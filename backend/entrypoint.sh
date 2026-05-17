#!/bin/bash
# In Docker + WSL2, NVIDIA injects the CUDA driver library at
# /usr/lib/wsl/drivers/<hash>/libcuda.so.1.1 rather than the standard
# /usr/lib/wsl/lib/libcuda.so.1 path. The pytorch image has a stub
# libcuda at /usr/lib/x86_64-linux-gnu/libcuda.so.1 (version 570.x)
# that lacks symbols the newer driver exposes, causing "named symbol
# not found" when PyTorch tries to init CUDA.
#
# Fix: find the real WSL2 driver library and symlink it into the
# standard path so the dynamic linker picks it up instead of the stub.

WSL_CUDA=$(find /usr/lib/wsl/drivers -name "libcuda.so*" 2>/dev/null | head -1)

if [ -n "$WSL_CUDA" ]; then
    echo "Preloading WSL2 CUDA driver: $WSL_CUDA"
    # LD_PRELOAD injects this library's symbols before the stub in
    # /usr/lib/x86_64-linux-gnu/libcuda.so.1 loads, overriding it.
    export LD_PRELOAD="$WSL_CUDA"
else
    echo "No WSL2 CUDA driver found, using container default (CPU mode)"
fi

exec "$@"
