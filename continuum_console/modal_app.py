from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "continuum-training-console"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/opt/continuum"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("continuum-console-data", create_if_missing=True)
auth_secret = modal.Secret.from_name("continuum-console-auth")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal>=1.4,<1.5")
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_ROOT,
        copy=True,
        ignore=[".git", ".continuum", "__pycache__", ".pytest_cache", "logs"],
    )
    .env(
        {
            "PYTHONPATH": REMOTE_ROOT,
            "CONTINUUM_REQUIRE_AUTH": "1",
            "CONTINUUM_SECURE_COOKIE": "1",
            "CONTINUUM_ENABLE_LAUNCH": "1",
            "CONTINUUM_MODAL_BIN": "modal",
        }
    )
)


@app.function(
    image=image,
    secrets=[auth_secret],
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(8787, startup_timeout=60)
def console() -> None:
    """Serve the authenticated console and its JSON control-plane API."""
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "continuum_console.api",
            "--host",
            "0.0.0.0",
            "--port",
            "8787",
            "--data",
            "/data/runs.json",
        ],
        cwd=REMOTE_ROOT,
    )
