FROM python:3.11-slim

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY ai_agent/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY ai_agent/ai_agent /app/ai_agent
COPY ai_agent/rag /app/rag
COPY ai_agent/run.sh /app/run.sh
COPY ai_agent/outputs /app/outputs

# Ensure run.sh is executable
RUN chmod +x /app/run.sh

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

CMD ["bash", "-lc", "./run.sh"]
