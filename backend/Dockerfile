# pytorch/pytorch images include the CUDA toolkit + cuDNN + PyTorch pre-installed.
# We use the 'devel' variant because torch.compile / Triton needs the full
# toolkit at runtime (not just the runtime .so files) to compile GPU kernels.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /app

# Install FastAPI and uvicorn. torch is already in the base image.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

RUN chmod +x entrypoint.sh

VOLUME ["/root/.cache/triton"]

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
