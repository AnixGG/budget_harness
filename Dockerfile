FROM python:3.13.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HARNESS_CONTAINER=1 HOME=/tmp
WORKDIR /app
COPY requirements.lock /app/requirements.lock
COPY requirements-build.lock /app/requirements-build.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /app/requirements-build.lock \
    && uv pip install --system --no-cache --require-hashes -r /app/requirements.lock
COPY harness /app/harness
COPY configs /app/configs
COPY run_agent.py collect_worker.py grade_examples.py config.example.json /app/
ENTRYPOINT ["python", "/app/run_agent.py"]
