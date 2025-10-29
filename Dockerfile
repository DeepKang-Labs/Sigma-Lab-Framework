# ===== Sigma-Lab Dockerfile =====
FROM python:3.11-slim

# Evite les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ARTIFACTS_ROOT=/data/artifacts

WORKDIR /app

# Copie et installation des dépendances en premier (meilleur cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copie du reste du code
COPY . /app

# Pré-crée les dossiers de travail
RUN mkdir -p $ARTIFACTS_ROOT/heartbeat \
             $ARTIFACTS_ROOT/mesh/inbox \
             $ARTIFACTS_ROOT/mesh/outbox \
             $ARTIFACTS_ROOT/mesh/archive \
             $ARTIFACTS_ROOT/model

# Expose FastAPI
EXPOSE 8080

# Lance le serveur mesh (FastAPI)
CMD ["python", "scripts/mesh_server.py"]
