# ===== Sigma-Lab Dockerfile =====
FROM python:3.11-slim

WORKDIR /app

# Copie du code source
COPY . .

# Installation des dépendances
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Variables d’environnement
ENV ARTIFACTS_ROOT=/data/artifacts
ENV PYTHONUNBUFFERED=1

# Création des dossiers de travail
RUN mkdir -p $ARTIFACTS_ROOT/heartbeat \
             $ARTIFACTS_ROOT/mesh/inbox \
             $ARTIFACTS_ROOT/mesh/outbox \
             $ARTIFACTS_ROOT/mesh/archive \
             $ARTIFACTS_ROOT/model

# Port exposé (FastAPI)
EXPOSE 8080

# Entrypoint : lance le serveur mesh
CMD ["python", "scripts/mesh_server.py"]
