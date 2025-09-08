# GroupMind

A simple Telegram group assistant powered by Mistral AI. Responds to @mentions and commands in group chats.

## Features

- @mentions: Responds when mentioned in groups
- Commands: `/ask`, `/summarize`, `/translate`, `/memory`, `/help`, `/remind`
- Memory: Per-chat memory toggle with conversation context
- DMs: Full conversation support in DMs
- Rate limiting: Per-user and per-chat, Redis-backed or in-memory fallback
- Health & metrics: HTTP endpoints for readiness and basic counters

## Quick Setup

1. **Create conda environment:**
   ```bash
   conda create -n groupmind python=3.11 -y
   conda activate groupmind
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get API keys:**
   - **Mistral API**: Get from [console.mistral.ai](https://console.mistral.ai/)
   - **Telegram Bot**: Message [@BotFather](https://t.me/BotFather) on Telegram

4. **Create `.env` file:**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

5. **Run the bot:**
   ```bash
   # Option 1: Use the run script (recommended)
   ./run.sh
   
   # Option 2: Manual activation
   conda activate groupmind
   python bot.py
   ```

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/ask <question>` | Ask anything | `/ask What's the weather?` |
| `/summarize [N]` | Summarize last N messages | `/summarize 10` |
| `/translate <lang> <text>` | Translate text to language | `/translate Spanish Hello world` |
| `/memory on\|off` | Toggle memory for chat | `/memory off` |
| `/remind <time> <text>` | Set reminder (flexible parsing with LLM fallback) | `/remind 10m Stand up` |
| `/help` | Show all commands | `/help` |

## Usage

- **In groups**: Mention your bot (e.g., `groupmind_mistral_bot hello`) or use commands
- **In DMs**: Just send any message
- **Memory**: When on, remembers conversation context
- **Style**: Adapts to group communication style

## How it works

1. **Simple storage**: Uses in-memory dictionaries (no database needed)
2. **Mistral AI**: Powers all responses with `mistral-large-latest` (updated to latest API)
3. **Context awareness**: Remembers recent messages when memory is enabled
4. **Group-friendly**: Brief responses in groups, detailed in DMs

## Docker

Build and run with Docker:

```bash
docker build -t groupmind .
docker run --rm -p 8080:8080 \
  -e MISTRAL_API_KEY=... -e TELEGRAM_BOT_TOKEN=... groupmind
```

## Docker Compose (with Redis)

```bash
docker-compose up --build
```

## Health & Metrics

- GET `/healthz` → `ok`
- GET `/metrics` → `updates_total`, `commands_total`

## Render Deployment

- Create a new Web Service on Render from this repo. Use Dockerfile.
- Environment: set `MISTRAL_API_KEY`, `TELEGRAM_BOT_TOKEN`, optionally `REDIS_URL`.
- Port: 8080 (for health/metrics only; Telegram uses polling by default).
- For horizontal scaling later: switch to webhook mode and keep Redis enabled.

## Troubleshooting

**Import errors**: Make sure you're using the conda environment:
```bash
conda activate groupmind
python bot.py
```

**"'Chat' object is not callable" error**: This was fixed by updating the Mistral AI library usage. The bot now uses the correct synchronous API calls.

## File Structure

```
mistral_bot/
├── bot.py              # Main bot file (everything in one place)
├── requirements.txt    # Dependencies
├── env.example        # Environment variables template
└── README.md          # This file
```

That's it! No complex setup, no databases, just a simple bot that works.