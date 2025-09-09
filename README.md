# GroupMind

<p align="center">
  <a href="https://www.youtube.com/watch?v=5kWei3I7hZ4">
    <img src="https://img.youtube.com/vi/5kWei3I7hZ4/maxresdefault.jpg" alt="Watch the demo" width="600">
  </a>
</p>



Telegram group assistant powered by Mistral. Production-ready single-file bot with Redis durable reminders, rate limiting, health/ready endpoints, Prometheus metrics, Grafana dashboard, Docker/Compose, and CI.

## Features

- @mentions: Replies when mentioned (like `/ask`)
- Commands: `/ask`, `/summarize`, `/translate`, `/remind|/reminder`, `/memory`, `/help`, `/archetype`
- Memory: Per-chat toggle for contextual answers (on/off)
- Reminders: Natural times; durable with Redis
- Rate limiting: Per-user and per-chat (Redis-backed or in-memory)
- Observability: `/ready`, `/healthz`, `/metrics` + Grafana dashboard
- Packaging: Dockerfile, docker-compose (Redis/Prom/Grafana), GitHub Actions CI

## Environment

Create `.env` (see `env.example`):

- `MISTRAL_API_KEY` (required)
- `TELEGRAM_BOT_TOKEN` (required)
- `REDIS_URL` (optional; e.g., `redis://redis:6379/0` under compose)
- `HEALTH_PORT` (default `8080`)
- Rate limits (optional): `RATE_LIMIT_USER`, `RATE_WINDOW_USER`, `RATE_LIMIT_CHAT`, `RATE_WINDOW_CHAT`

## Run (Docker Compose)

```bash
cp env.example .env
docker compose up --build
# Bot: polling + health/metrics on :8080
# Redis: :6379, Prometheus: :9090, Grafana: :3000 (admin/admin)
```

Health/metrics:
- `GET http://localhost:8080/ready` → 200 when ready
- `GET http://localhost:8080/healthz` → JSON (uptime, username, Redis, etc.)
- `GET http://localhost:8080/metrics` → Prometheus exposition

Grafana: `http://localhost:3000` → dashboard “GroupMind Overview”

<p align="center">
  <img src=""<img width="1601" height="776" alt="Screenshot 2025-09-09 at 16 17 32" src="https://github.com/user-attachments/assets/86355dcd-7d86-4ee6-b18f-600f5fdab3a6" />
 alt="Grafana Dashboard" width="800">
</p>

## Run (local)

```bash
pip install -r requirements.txt
python bot.py
```

## Commands

- `/ask <question>`
- `/summarize [N]`
- `/translate <language> <text>`
- `/remind|/reminder <time> <text>` (e.g., `10m Take a break`, `1 hour Call Ana`, `14:30 Standup`, `tomorrow 09:00 Flight`)
- `/memory on|off`
- `/archetype [@username|me]`
- `/help`

Notes:
- Memory ON affects Q&A context; `/summarize` uses recent recorded text regardless.
- Reminders are durable when `REDIS_URL` is set; otherwise they persist until process restarts.

## CI

GitHub Actions (`.github/workflows/ci.yml`): flake8 → pytest (coverage) → Docker build.
Push to GitHub and watch the “CI” workflow.

## Deploy (Render)

- Create a Web Service from this repo (Dockerfile)
- Env: `MISTRAL_API_KEY`, `TELEGRAM_BOT_TOKEN`, optional `REDIS_URL`
- Port: 8080 (health/metrics). Polling by default; for replicas switch to webhooks and use Redis
