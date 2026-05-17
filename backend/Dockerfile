# pytorch/pytorch images include the CUDA toolkit + cuDNN + PyTorch pre-installed.
# We use the 'devel' variant because torch.compile / Triton needs the full
# toolkit at runtime (not just the runtime .so files) to compile GPU kernels.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /app

# Install FastAPI and uvicorn. torch is already in the base image.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

# Triton's JIT cache lives here by default; expose it as a volume so kernels
# persist across container restarts (avoids recompile on every startup).
VOLUME ["/root/.cache/triton"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
