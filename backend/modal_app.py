"""
Modal deployment entrypoint.

Wraps the existing FastAPI app with a GPU-backed Modal function.
Deploy with: modal deploy backend/modal_app.py
"""
import modal
from pathlib import Path

backend_dir = Path(__file__).parent

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel")
    .pip_install("fastapi==0.115.12", "uvicorn==0.34.2", "pydantic")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(backend_dir, remote_path="/app")  # must be last
)

app = modal.App("borgcompiler-backend")


@app.function(
    image=image,
    gpu="A10G",
    timeout=300,
    secrets=[modal.Secret.from_name("borgcompiler-secrets")],
)
@modal.asgi_app()
def fastapi_app():
    from main import app as _app  # noqa: PLC0415
    return _app
