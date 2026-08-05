from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from continuum_console.api import _parse_metrics, create_server
from continuum_console.auth import AuthManager
from continuum_console.commands import build_command
from continuum_console.store import RunStore


class CommandTest(unittest.TestCase):
    def test_sdpo_command_targets_existing_launcher(self) -> None:
        command = build_command({
            "algorithm": "sdpo",
            "model": "Qwen/Qwen3.5-0.8B",
            "dataset": "feedback",
            "steps": 3,
            "train_rows": 8,
            "val_rows": 2,
            "hint": "Use the schema",
        })
        self.assertIn("SDPO/modal_verl_sdpo.py::main", command)
        self.assertIn("--total-training-steps", command)
        self.assertIn("--static-feedback", command)

    def test_sdpo_smoke_command_selects_guarded_executor(self) -> None:
        command = build_command({
            "algorithm": "sdpo",
            "model": "Qwen/Qwen3.5-0.8B",
            "dataset": "gsm8k_sdpo",
            "steps": 1,
            "train_rows": 2,
            "val_rows": 1,
            "smoke_gpu": True,
        })
        self.assertIn("--smoke-gpu", command)

    def test_kimi_command_maps_topology_to_mode(self) -> None:
        command = build_command({
            "algorithm": "kimi-sdpo",
            "model": "moonshotai/Kimi-K2.5",
            "dataset": "kimi_sdpo",
            "steps": 2,
            "topology": "h200-lora",
        })
        self.assertIn("SDPO/modal_verl_kimi_k26_sdpo.py", command)
        self.assertEqual("h200-lora", command[command.index("--mode") + 1])


class AuthTest(unittest.TestCase):
    def test_signed_session_round_trip_and_tamper_rejection(self) -> None:
        auth = AuthManager("admin", "correct horse battery", "s" * 32)
        token = auth.issue()
        self.assertTrue(auth.verify_token(token))
        self.assertFalse(auth.verify_token(token + "x"))

    def test_required_auth_rejects_short_environment_secrets(self) -> None:
        with patch.dict("os.environ", {"CONTINUUM_ADMIN_PASSWORD": "short", "CONTINUUM_SESSION_SECRET": "short"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Authentication requires"):
                AuthManager.from_environment(required=True)


class MetricParsingTest(unittest.TestCase):
    def test_extracts_verl_step_metrics(self) -> None:
        metrics = _parse_metrics([
            "step:1 - distillation/loss:0.114 - critic/rewards/mean:0.78 - teacher_kl:0.081 - training/global_step:1"
        ])
        self.assertEqual(1, metrics[0]["step"])
        self.assertEqual(0.114, metrics[0]["distillation_loss"])
        self.assertEqual(0.78, metrics[0]["reward"])


class StoreTest(unittest.TestCase):
    def test_create_persists_a_draft_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs.json")
            created = store.create({
                "name": "tool_recovery",
                "algorithm": "sdpo",
                "model": "Qwen/Qwen3.5-0.8B",
                "dataset": "traces",
                "steps": 4,
            })
            self.assertEqual("draft", created["status"])
            self.assertEqual(created["id"], store.get(created["id"])["id"])
            self.assertEqual(4, created["config"]["steps"])

    def test_rejects_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs.json")
            with self.assertRaisesRegex(ValueError, "model and dataset"):
                store.create({"algorithm": "sdpo", "dataset": "traces"})


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.server = create_server("127.0.0.1", 0, Path(self.tmp.name) / "runs.json")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path) as response:
            return json.load(response)

    def test_health_catalog_and_seed_run(self) -> None:
        self.assertTrue(self.get_json("/api/health")["ok"])
        self.assertGreaterEqual(len(self.get_json("/api/catalog")["algorithms"]), 4)
        runs = self.get_json("/api/runs")["runs"]
        self.assertEqual("run_sdpo_airline_001", runs[0]["id"])

    def test_create_and_read_run(self) -> None:
        request = urllib.request.Request(
            self.base + "/api/runs",
            data=json.dumps({
                "name": "api_created",
                "algorithm": "opd",
                "model": "Qwen/Qwen3.5-0.8B",
                "teacher_model": "Qwen/Qwen3.5-4B",
                "dataset": "gsm8k",
                "steps": 2,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            created = json.load(response)
        fetched = self.get_json("/api/runs/" + created["id"])
        self.assertEqual("api_created", fetched["name"])
        self.assertIn("OPD/modal_verl_qwen35_opd.py", fetched["command"])

    def test_launch_is_safely_gated(self) -> None:
        request = urllib.request.Request(
            self.base + "/api/runs/run_sdpo_airline_001/launch",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(403, raised.exception.code)


if __name__ == "__main__":
    unittest.main()
