# Dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn

COPY scripts/mesh_server.py ./scripts/mesh_server.py
RUN mkdir -p /data/artifacts/mesh/inbox /data/artifacts/mesh/outbox

ENV ARTIFACTS_ROOT=/data/artifacts \
    SIGMA_TOKEN=CHANGE_ME_SECRET \
    SIGMA_NODE_ID=node-docker \
    PORT=8000

EXPOSE 8000
CMD ["python", "scripts/mesh_server.py"]
