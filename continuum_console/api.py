from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .catalog import catalog
from .auth import AuthManager
from .commands import build_command
from .store import RunStore

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).with_name("static")
DEFAULT_DATA = ROOT / ".continuum" / "runs.json"


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "ContinuumConsole/0.1"

    @property
    def store(self) -> RunStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def auth(self) -> AuthManager:
        return self.server.auth  # type: ignore[attr-defined]

    def log_message(self, message: str, *args: object) -> None:
        print(f"[console] {self.address_string()} {message % args}")

    def _json(self, payload: object, status: int = 200, cookie: str | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request body exceeds 1 MB")
        if not length:
            return {}
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json({"ok": True, "service": "continuum-console"})
        if path == "/api/auth/status":
            authenticated = self.auth.authenticated(self.headers.get("Cookie", ""))
            return self._json({"required": self.auth.required, "authenticated": authenticated, "username": self.auth.username if authenticated else None})
        if path.startswith("/api/") and not self._authorized():
            return
        if path == "/api/catalog":
            return self._json(catalog())
        if path == "/api/runs":
            runs = self.store.list_runs()
            summaries = [{key: run[key] for key in ("id", "name", "status", "created_at", "updated_at", "config", "metrics")} for run in runs]
            return self._json({"runs": summaries})
        if path.startswith("/api/runs/"):
            run_id = unquote(path.removeprefix("/api/runs/")).split("/")[0]
            run = self.store.get(run_id)
            return self._json(run if run else {"error": "run not found"}, 200 if run else 404)
        self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._body()
            self._pending_payload = payload
            if path == "/api/auth/login":
                return self._login(payload)
            if path == "/api/auth/logout":
                return self._json({"ok": True}, cookie=self.auth.clear_cookie())
            if not self._authorized() or not self._same_origin():
                return
            if path == "/api/runs":
                return self._json(self.store.create(payload), HTTPStatus.CREATED)
            if path.startswith("/api/runs/") and path.endswith("/launch"):
                run_id = unquote(path.removeprefix("/api/runs/").removesuffix("/launch").rstrip("/"))
                return self._launch(run_id)
            if path.startswith("/api/runs/") and path.endswith("/ingest"):
                run_id = unquote(path.removeprefix("/api/runs/").removesuffix("/ingest").rstrip("/"))
                return self._ingest(run_id, payload)
            self._json({"error": "not found"}, 404)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)

    def _launch(self, run_id: str) -> None:
        run = self.store.get(run_id)
        if not run:
            return self._json({"error": "run not found"}, 404)
        if os.environ.get("CONTINUUM_ENABLE_LAUNCH") != "1":
            return self._json(
                {"error": "launching is disabled", "command": run["command"], "help": "Set CONTINUUM_ENABLE_LAUNCH=1 to allow Modal jobs."},
                HTTPStatus.FORBIDDEN,
            )
        if run["status"] not in {"draft", "failed"}:
            return self._json({"error": f"cannot launch a {run['status']} run"}, 409)
        confirmation = self._pending_payload.get("confirm") if hasattr(self, "_pending_payload") else None
        if confirmation != run_id:
            return self._json({"error": "launch confirmation must exactly match the run id"}, 400)
        command = build_command(run["config"])
        self.store.update(run_id, status="queued", command=command, logs=run["logs"] + ["Submitting detached Modal job…"])
        threading.Thread(target=self._execute, args=(run_id, command), daemon=True).start()
        self._json(self.store.get(run_id), HTTPStatus.ACCEPTED)

    def _execute(self, run_id: str, command: list[str]) -> None:
        self.store.update(run_id, status="running")
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if "--smoke-gpu" in command:
                env["VERL_IMAGE_TAG"] = os.environ.get("CONTINUUM_SMOKE_VERL_IMAGE_TAG", "verlai/verl:vllm020.dev1")
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            self.store.update(run_id, job={"pid": process.pid, "backend": "modal", "command": command, "image": env.get("VERL_IMAGE_TAG")})
            lines: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line.rstrip())
                lines = lines[-1000:]
                metrics = _parse_metrics(lines)
                app_id = _parse_modal_app_id(lines)
                changes = {"logs": lines, "metrics": metrics}
                if app_id:
                    changes["job"] = {"pid": process.pid, "backend": "modal", "app_id": app_id, "command": command, "image": env.get("VERL_IMAGE_TAG")}
                self.store.update(run_id, **changes)
            return_code = process.wait()
            status = "completed" if return_code == 0 else "failed"
            self.store.update(run_id, status=status, logs=lines, metrics=_parse_metrics(lines))
        except Exception as exc:  # keep the detached worker failure visible in the run
            self.store.update(run_id, status="failed", logs=[f"Launch failed: {exc}"])

    def _login(self, payload: dict) -> None:
        if not self.auth.required:
            return self._json({"ok": True, "authenticated": True, "username": self.auth.username})
        username = str(payload.get("username", ""))
        password = str(payload.get("password", ""))
        if not self.auth.verify_credentials(username, password):
            return self._json({"error": "invalid username or password"}, HTTPStatus.UNAUTHORIZED)
        return self._json(
            {"ok": True, "authenticated": True, "username": self.auth.username},
            cookie=self.auth.session_cookie(self.auth.issue()),
        )

    def _authorized(self) -> bool:
        if self.auth.authenticated(self.headers.get("Cookie", "")):
            return True
        self._json({"error": "authentication required"}, HTTPStatus.UNAUTHORIZED)
        return False

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        expected_http = f"http://{self.headers.get('Host', '')}"
        expected_https = f"https://{self.headers.get('Host', '')}"
        if origin in {expected_http, expected_https}:
            return True
        self._json({"error": "cross-origin request rejected"}, HTTPStatus.FORBIDDEN)
        return False

    def _ingest(self, run_id: str, payload: dict) -> None:
        run = self.store.get(run_id)
        if not run:
            return self._json({"error": "run not found"}, 404)
        metrics = payload.get("metrics", run["metrics"])
        traces = payload.get("traces", run["traces"])
        logs = payload.get("logs", run["logs"])
        if not isinstance(metrics, list) or not isinstance(traces, list) or not isinstance(logs, list):
            raise ValueError("metrics, traces, and logs must be arrays")
        updated = self.store.update(run_id, metrics=metrics, traces=traces, logs=logs, status=str(payload.get("status", run["status"])))
        self._json(updated)

    def _static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else unquote(path.lstrip("/"))
        target = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in target.parents or not target.is_file():
            target = STATIC_ROOT / "index.html"
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _parse_metrics(lines: list[str]) -> list[dict]:
    metrics: dict[int, dict] = {}
    for line in lines:
        step_match = re.search(r"(?:training/global_step|global_step|step)\s*[:=]\s*(\d+)", line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        item = metrics.setdefault(step, {"step": step})
        patterns = {
            "distillation_loss": r"(?:distillation/loss|distillation_loss)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            "reward": r"(?:critic/rewards/mean|train/critic/rewards/mean|reward)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            "teacher_kl": r"(?:teacher_kl|kl_loss|kl)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                item[key] = float(match.group(1))
    return [metrics[key] for key in sorted(metrics)]


def _parse_modal_app_id(lines: list[str]) -> str | None:
    for line in reversed(lines):
        match = re.search(r"\bap-[A-Za-z0-9]+\b", line)
        if match:
            return match.group(0)
    return None


def create_server(host: str = "127.0.0.1", port: int = 8787, data_path: Path = DEFAULT_DATA, require_auth: bool | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), ConsoleHandler)
    server.store = RunStore(data_path)  # type: ignore[attr-defined]
    server.auth = AuthManager.from_environment(required=require_auth)  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ContinuumAI experiment console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    server = create_server(args.host, args.port, args.data)
    print(f"Continuum console: http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
