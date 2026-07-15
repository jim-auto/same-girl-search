FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY apps/api/requirements.txt /workspace/apps/api/requirements.txt
RUN pip install --no-cache-dir -r /workspace/apps/api/requirements.txt

COPY libs /workspace/libs
COPY apps/api /workspace/apps/api

ENV PYTHONPATH=/workspace:/workspace/libs
WORKDIR /workspace/apps/api

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

