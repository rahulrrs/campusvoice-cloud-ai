FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.base.txt requirements.cpu.txt requirements.gpu.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.cpu.txt

COPY . .

EXPOSE 8000

RUN chmod +x /app/entrypoint.sh

CMD ["sh", "/app/entrypoint.sh"]
