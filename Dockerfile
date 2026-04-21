FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        ffmpeg libsndfile1 curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/speech-api

ENV PATH="/workspace/speech-api/.venv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/workspace/speech-api/.venv/lib/python3.11/site-packages/onnxruntime/capi:${LD_LIBRARY_PATH}"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
