#!/usr/bin/env python3
"""
GroupMind - Simple Telegram Group Assistant powered by Mistral AI

This bot responds to @mentions and commands in Telegram groups.
Commands: /ask, /summarize, /translate, /memory, /help
"""

import os
import logging
import json
import asyncio
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from typing import Dict, List
from datetime import datetime, timedelta

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from mistralai import Mistral
from mistralai.models import UserMessage, AssistantMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Simple rate limit config (can tune via env)
RATE_LIMIT_USER = int(os.getenv("RATE_LIMIT_USER", "8"))  # requests
RATE_WINDOW_USER = int(os.getenv("RATE_WINDOW_USER", "30"))  # seconds
RATE_LIMIT_CHAT = int(os.getenv("RATE_LIMIT_CHAT", "20"))  # requests
RATE_WINDOW_CHAT = int(os.getenv("RATE_WINDOW_CHAT", "60"))  # seconds

# Optional Redis URL for durability and rate limit backing
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if REDIS_URL:
    try:
        import redis

        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        logging.warning(f"Redis unavailable: {e}")
        redis_client = None

if not MISTRAL_API_KEY or not TELEGRAM_BOT_TOKEN:
    raise ValueError(
        "Please set MISTRAL_API_KEY and TELEGRAM_BOT_TOKEN in your .env file"
    )

# Initialize Mistral client
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Simple in-memory storage for chat settings and recent messages
chat_settings: Dict[str, Dict] = (
    {}
)  # chat_id -> {memory_on: bool, style: str, recent_messages: list}
MAX_RECENT_MESSAGES = 10

# Metrics (Prometheus)
registry = CollectorRegistry()
MET_UPDATES = Counter("updates_total", "Total updates processed", registry=registry)
MET_MESSAGES = Counter(
    "messages_total", "Total text messages stored", registry=registry
)
MET_COMMANDS = Counter(
    "commands_total", "Total commands handled", ["command"], registry=registry
)
MET_COMMANDS_AGG = Counter(
    "commands_handled_total", "Total commands handled (aggregate)", registry=registry
)
MET_READY = Gauge("ready", "Readiness flag (1 ready, 0 not ready)", registry=registry)
MET_UPTIME = Gauge("uptime_seconds", "Process uptime in seconds", registry=registry)
MET_RL_BLOCKS = Counter(
    "rate_limit_blocks_total", "Requests blocked by rate limit", registry=registry
)
MET_MISTRAL_ERRORS = Counter(
    "mistral_errors_total", "Mistral API call failures", ["code"], registry=registry
)
MET_MISTRAL_LATENCY = Histogram(
    "mistral_latency_seconds",
    "Latency of Mistral chat.complete calls",
    buckets=(0.1, 0.2, 0.5, 1, 2, 5, 10, 20),
    registry=registry,
)
STARTED_AT = time.time()
READY = False

# Bot username (will be set when bot starts)
BOT_USERNAME = None

# System prompt for the bot
SYSTEM_PROMPT = """You are GroupMind, an ambient AI assistant in Telegram groups.

Core behavior:
- Be helpful, concise, and context-aware.
- In groups: Keep responses brief (1-3 sentences).
- In DMs: Be more detailed when needed.
- Match the group's communication style when possible.

Commands:
- /ask <question>
- /summarize [N]
- /translate <language> <text>
- /remind|/reminder <time> <text>
- /archetype [@username|me]
- /memory on|off
- /help

Remember: You are ambient and non-intrusive."""


def get_chat_id(update: Update) -> str:
    """Get chat ID as string."""
    return str(update.effective_chat.id)


def get_chat_settings(chat_id: str) -> Dict:
    """Get or create chat settings."""
    if chat_id not in chat_settings:
        chat_settings[chat_id] = {
            "memory_on": True,
            "style": "friendly",
            "recent_messages": [],
        }
    return chat_settings[chat_id]


def sanitize_output(text: str) -> str:
    """Remove bold markers and trim extra whitespace from model output."""
    if not isinstance(text, str):
        return text
    return text.replace("**", "").strip()


class InMemoryRateLimiter:
    def __init__(self):
        self.user_hits: Dict[str, list] = {}
        self.chat_hits: Dict[str, list] = {}

    def _prune(self, arr: list, now_ts: float, window: int):
        while arr and now_ts - arr[0] > window:
            arr.pop(0)

    def allow(self, user_id: int, chat_id: str) -> bool:
        now_ts = time.time()
        # Per-user
        ukey = str(user_id)
        arr_u = self.user_hits.setdefault(ukey, [])
        self._prune(arr_u, now_ts, RATE_WINDOW_USER)
        # Per-chat
        ckey = str(chat_id)
        arr_c = self.chat_hits.setdefault(ckey, [])
        self._prune(arr_c, now_ts, RATE_WINDOW_CHAT)

        if len(arr_u) >= RATE_LIMIT_USER:
            return False
        if len(arr_c) >= RATE_LIMIT_CHAT:
            return False

        arr_u.append(now_ts)
        arr_c.append(now_ts)
        return True


rate_limiter = InMemoryRateLimiter()


def rate_limited(handler_fn):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            command_label = handler_fn.__name__
        except Exception:
            command_label = "unknown"
        MET_COMMANDS.labels(command=command_label).inc()
        MET_COMMANDS_AGG.inc()
        user_id = update.effective_user.id if update.effective_user else 0
        chat_id = get_chat_id(update)
        # Redis-backed option
        if redis_client is not None:
            now = int(time.time())
            pipe = redis_client.pipeline()
            # user key
            ku = f"rl:user:{user_id}"
            pipe.zremrangebyscore(ku, 0, now - RATE_WINDOW_USER)
            pipe.zadd(ku, {str(now): now})
            pipe.zcard(ku)
            pipe.expire(ku, RATE_WINDOW_USER)
            # chat key
            kc = f"rl:chat:{chat_id}"
            pipe.zremrangebyscore(kc, 0, now - RATE_WINDOW_CHAT)
            pipe.zadd(kc, {str(now): now})
            pipe.zcard(kc)
            pipe.expire(kc, RATE_WINDOW_CHAT)
            res = pipe.execute()
            user_count = res[2]
            chat_count = res[6]
            if user_count > RATE_LIMIT_USER or chat_count > RATE_LIMIT_CHAT:
                MET_RL_BLOCKS.inc()
                await update.effective_message.reply_text(
                    "Rate limit exceeded. Please try again shortly."
                )
                return
            return await handler_fn(update, context)
        # In-memory fallback
        if not rate_limiter.allow(user_id, chat_id):
            MET_RL_BLOCKS.inc()
            await update.effective_message.reply_text(
                "Rate limit exceeded. Please try again shortly."
            )
            return
        return await handler_fn(update, context)

    return wrapper


def add_message_to_memory(chat_id: str, role: str, content: str, user_name: str = None):
    """Add message to chat memory."""
    settings = get_chat_settings(chat_id)
    message = {
        "role": role,
        "content": content,
        "user_name": user_name,
        "timestamp": datetime.now().isoformat(),
    }
    settings["recent_messages"].append(message)
    # Metrics: count only real text-like entries
    if role in ("message", "user", "assistant", "command") and content:
        MET_MESSAGES.inc()
    # Keep only recent messages
    if len(settings["recent_messages"]) > MAX_RECENT_MESSAGES:
        settings["recent_messages"] = settings["recent_messages"][-MAX_RECENT_MESSAGES:]


def get_mistral_response(messages: List[Dict], max_tokens: int = 1000) -> str:
    """Get response from Mistral AI."""
    try:
        # Prepare messages for Mistral
        current_time_iso = datetime.utcnow().isoformat() + "Z"
        mistral_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=f"Current date/time (UTC): {current_time_iso}"),
        ]

        for msg in messages:
            if msg["role"] == "user":
                mistral_messages.append(UserMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                mistral_messages.append(AssistantMessage(content=msg["content"]))
            elif msg["role"] == "system":
                mistral_messages.append(SystemMessage(content=msg["content"]))

        # Get response (single attempt) and record latency
        start_ts = time.time()
        response = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=mistral_messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        MET_MISTRAL_LATENCY.observe(time.time() - start_ts)

        return response.choices[0].message.content

    except Exception as e:
        # Attempt to parse a code hint from message
        code_label = "unknown"
        msg = str(e)
        if "429" in msg:
            code_label = "429"
        elif "401" in msg:
            code_label = "401"
        elif "5" in msg:
            code_label = "5xx"
        MET_MISTRAL_ERRORS.labels(code=code_label).inc()
        return f"Sorry, I encountered an error: {str(e)}"


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        (
            "GroupMind is active.\n\n"
            f"Username: @{BOT_USERNAME or 'unknown'}\n\n"
            "To mention: @USERNAME hello or USERNAME hello\n"
            "Use /help to see commands."
        ).replace("USERNAME", BOT_USERNAME or "your_bot")
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_lines = [
        "Commands:",
        "/ask <question> - Ask a question",
        "/help - Show this help",
        "/summarize [N] - Summarize last N messages (default 25)",
        "/translate <language> <text> - Translate text to language",
        "/remind|/reminder <time> <text> - Set a reminder",
        (
            "  Examples: /remind 10m Take a break | /reminder 1 hour Call Ana "
            "| /remind tomorrow 09:00 Flight"
        ),
        "/archetype [@username|me] - Analyze a user's chat archetype",
        "/memory on|off - Toggle memory for this chat",
        "",
        "Usage:",
        "- Use slash commands in groups",
        "- Send any message in DMs",
        "- When memory is on, recent context may be used",
    ]
    help_text = "\n".join(help_lines)
    await update.message.reply_text(help_text)


@rate_limited
async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ask command."""

    chat_id = get_chat_id(update)
    settings = get_chat_settings(chat_id)

    # Get the question from command arguments
    question = " ".join(context.args) if context.args else "Hello!"

    # Prepare messages
    messages = [{"role": "user", "content": question}]

    # Add recent context if memory is on
    if settings["memory_on"] and settings["recent_messages"]:
        # Add last few messages as context
        recent_context = settings["recent_messages"][-5:]  # Last 5 messages
        context_text = "Recent conversation context:\n"
        for msg in recent_context:
            context_text += f"{msg['role']}: {msg['content']}\n"
        messages.insert(0, {"role": "system", "content": context_text})

    # Get response from Mistral
    response = get_mistral_response(messages)

    # Send response
    await update.message.reply_text(sanitize_output(response))

    # Add to memory
    add_message_to_memory(chat_id, "user", question, update.effective_user.username)
    add_message_to_memory(chat_id, "assistant", response)


@rate_limited
async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /summarize command."""
    chat_id = get_chat_id(update)
    settings = get_chat_settings(chat_id)

    # Get number of messages to summarize
    num_messages = 25  # default
    if context.args:
        try:
            num_messages = int(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "Please provide a valid number of messages to summarize."
            )
            return

    if not settings["recent_messages"]:
        await update.message.reply_text("No recent messages to summarize.")
        return

    # Get messages to summarize
    messages_to_summarize = settings["recent_messages"][-num_messages:]

    # Prepare for summarization
    conversation_text = "\n".join(
        [
            f"{msg['user_name'] or msg['role']}: {msg['content']}"
            for msg in messages_to_summarize
        ]
    )

    summary_prompt = (
        f"Please provide a concise summary of this conversation:\n\n{conversation_text}"
    )

    # Get summary from Mistral
    response = get_mistral_response(
        [{"role": "user", "content": summary_prompt}], max_tokens=500
    )

    await update.message.reply_text(f"Summary:\n\n{sanitize_output(response)}")


async def archetype_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze a user's chat archetype based on recent messages in this chat."""
    chat_id = get_chat_id(update)
    settings = get_chat_settings(chat_id)
    # Determine target username
    target_username = None
    if context.args:
        arg = context.args[0]
        if arg.lower() == "me":
            target_username = update.effective_user.username
        else:
            target_username = arg.lstrip("@")
    else:
        target_username = update.effective_user.username

    if not target_username:
        await update.message.reply_text(
            "User has no username. Mention them or use /archetype @username"
        )
        return

    # Collect recent messages from this user from in-memory buffer
    # If memory is off and buffer is empty, we still try with what we have
    recent = [
        m
        for m in settings["recent_messages"]
        if m.get("user_name") == target_username and m.get("content")
    ]
    if not recent:
        await update.message.reply_text("No recent messages found for that user.")
        return
    # Limit to last 20 messages from that user
    user_msgs = recent[-20:]
    sample_text = "\n".join(f"- {m['content']}" for m in user_msgs)

    prompt = (
        "You will assign a short, viral chat archetype based on recent messages."
        " Keep it playful but kind."
        "Strict format (3 lines max):\n"
        "Archetype: <2-3 word catchy name>\n"
        "Tagline: <one short punchline>\n"
        "Traits: <three ultra-short traits, comma-separated>\n\n"
        f"User: @{target_username}\n"
        f"Messages:\n{sample_text}"
    )

    response = get_mistral_response(
        [{"role": "user", "content": prompt}], max_tokens=120
    )
    await update.message.reply_text(sanitize_output(response))


@rate_limited
async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /translate command."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Please specify a language and text to translate.\n\n"
            "Usage: /translate <language> <text>\n"
            "Example: /translate Spanish Hello world"
        )
        return

    # First argument is the language, rest is the text to translate
    target_language = context.args[0]
    text_to_translate = " ".join(context.args[1:])

    # Prepare translation request
    translate_prompt = (
        "Translate the following text to "
        f"{target_language}. Only return the translation, no explanations:\n\n"
        f"{text_to_translate}"
    )

    # Get translation from Mistral
    response = get_mistral_response(
        [{"role": "user", "content": translate_prompt}], max_tokens=200
    )

    await update.message.reply_text(
        f"Translation to {target_language}:\n\n{sanitize_output(response)}"
    )


@rate_limited
async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /memory command."""
    chat_id = get_chat_id(update)
    settings = get_chat_settings(chat_id)

    if not context.args:
        status = "on" if settings["memory_on"] else "off"
        await update.message.reply_text(
            f"Memory is currently {status} for this chat.\n\nUse /memory on or /memory off to change."
        )
        return

    command = context.args[0].lower()

    if command in ["on", "enable", "true"]:
        settings["memory_on"] = True
        await update.message.reply_text(
            "Memory enabled for this chat. I will remember recent context."
        )
    elif command in ["off", "disable", "false"]:
        settings["memory_on"] = False
        settings["recent_messages"] = []  # Clear existing memory
        await update.message.reply_text(
            "Memory disabled for this chat. I will respond without context."
        )
    else:
        await update.message.reply_text("Please use /memory on or /memory off")


def _parse_time_spec(args: List[str]) -> (datetime, str):
    """Parse reminder time and text.

    Supported forms:
    - '10m text', '1h30m text', '45s text', '2d text'
    - '14:30 text' (UTC today or tomorrow if past)
    - 'tomorrow 09:00 text'
    Returns: (when_utc_datetime, text)
    """
    if not args:
        raise ValueError("Usage: /remind <duration|HH:MM|tomorrow HH:MM> [text]")

    # Normalize leading 'in'
    if args and args[0].lower() == "in":
        args = args[1:]
        if not args:
            raise ValueError("Missing time after 'in'")

    UNIT_WORDS = {
        "s": "s",
        "sec": "s",
        "secs": "s",
        "second": "s",
        "seconds": "s",
        "m": "m",
        "min": "m",
        "mins": "m",
        "minute": "m",
        "minutes": "m",
        "h": "h",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "d": "d",
        "day": "d",
        "days": "d",
    }

    def parse_duration(token: str) -> timedelta:
        # Supports 10m, 45s, 2h, 1h30m, 90m, etc.
        total = timedelta()
        num = ""
        for ch in token:
            if ch.isdigit():
                num += ch
            else:
                if not num:
                    raise ValueError("Invalid duration")
                value = int(num)
                if ch == "h":
                    total += timedelta(hours=value)
                elif ch == "m":
                    total += timedelta(minutes=value)
                elif ch == "s":
                    total += timedelta(seconds=value)
                elif ch == "d":
                    total += timedelta(days=value)
                else:
                    raise ValueError("Unsupported duration unit")
                num = ""
        if num:
            # bare number means minutes
            total += timedelta(minutes=int(num))
        if total.total_seconds() <= 0:
            raise ValueError("Duration must be > 0")
        return total

    # First token handling
    first = args[0].lower()

    # Case B: 'tomorrow HH:MM' (handle before duration heuristics)
    if first == "tomorrow" and len(args) >= 2:
        time_str = args[1]
        try:
            hour, minute = map(int, time_str.split(":"))
        except Exception:
            raise ValueError("Time must be HH:MM")
        now = datetime.utcnow()
        when = now.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        ) + timedelta(days=1)
        text = " ".join(args[2:]).strip() or "Reminder"
        return when, text

    # Case C: 'HH:MM' today (UTC), tomorrow if past
    if ":" in first:
        try:
            hour, minute = map(int, first.split(":"))
        except Exception:
            raise ValueError("Time must be HH:MM in 24h")
        now = datetime.utcnow()
        when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if when <= now:
            when = when + timedelta(days=1)
        text = " ".join(args[1:]).strip() or "Reminder"
        return when, text

    # Case A: duration first token (after explicit date/time checks)
    # If numeric followed by a unit word, join them (e.g., '15' 's' -> '15s')
    if first.isdigit() and len(args) > 1 and args[1].lower() in UNIT_WORDS:
        first = first + UNIT_WORDS[args[1].lower()]
        args = [first] + args[2:]
    # Detect compact duration tokens like 10m, 1h30m, 45s, 2d
    if first.isdigit() or any(ch in first for ch in ["h", "m", "s", "d"]):
        when = datetime.utcnow() + parse_duration(first)
        text = " ".join(args[1:]).strip() or "Reminder"
        return when, text

    raise ValueError(
        "Unrecognized time format. Use duration (e.g., 10m, 1h30m) or HH:MM or 'tomorrow HH:MM'."
    )


def _format_duration(total_seconds: int) -> str:
    """Format a positive number of seconds as a compact human string (e.g., '1h 5m', '3m', '45s')."""
    total_seconds = max(0, int(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds and not parts:
        # show seconds only if less than a minute total
        parts.append(f"{seconds}s")
    return " ".join(parts) or "0s"


async def remind_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set a reminder. Flexible parsing with LLM fallback."""
    # Try local parser first
    try:
        when, text = _parse_time_spec(context.args)
    except ValueError:
        # Fallback: ask Mistral to parse time and text from freeform input
        raw = " ".join(context.args).strip()
        if not raw:
            await update.message.reply_text("Usage: /remind <time> <text>")
            return
        current_utc = datetime.utcnow().isoformat() + "Z"
        prompt = (
            "Extract reminder time and text from the command below. Treat the exact unit "
            "immediately after the first number literally (e.g., '15 s' means seconds, not minutes).\n"
            "Return ONLY JSON: {\"when_utc\":\"YYYY-MM-DDTHH:MM:SSZ\",\"text\":\"...\"}.\n"
            "If time is relative (e.g., '3 min', '1 hour'), add to current time.\n"
            "Assume UTC. If only a date is given, use 09:00 UTC.\n"
            f"Current time (UTC): {current_utc}\n"
            f"Command: {raw}"
        )
        parsed = get_mistral_response(
            [{"role": "user", "content": prompt}], max_tokens=120
        )
        parsed = sanitize_output(parsed)
        try:
            start = parsed.find("{")
            end = parsed.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = parsed[start : end + 1]
            data = json.loads(parsed)
            when_str = data.get("when_utc")
            text = data.get("text") or "Reminder"
            if not when_str:
                raise ValueError("Missing when_utc")
            # Basic parse for YYYY-MM-DDTHH:MM:SSZ
            when = datetime.strptime(when_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            await update.message.reply_text(
                "Could not parse time. Try formats like '10m', '1h30m', '14:30', or '22 September'."
            )
            return
    delay_seconds = max(0, int((when - datetime.utcnow()).total_seconds()))
    if redis_client is not None:
        # store durable reminder
        due_ts = int(time.time()) + delay_seconds
        payload = json.dumps({"chat_id": update.effective_chat.id, "text": text})
        try:
            redis_client.zadd("reminders", {payload: due_ts})
        except Exception as e:
            logging.warning(f"Failed to persist reminder, falling back in-memory: {e}")

            async def _delayed_send():
                try:
                    await asyncio.sleep(delay_seconds)
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id, text=f"Reminder: {text}"
                    )
                except Exception as e:
                    logging.warning(f"Failed to send reminder: {e}")

            asyncio.create_task(_delayed_send())
    else:

        async def _delayed_send():
            try:
                await asyncio.sleep(delay_seconds)
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text=f"Reminder: {text}"
                )
            except Exception as e:
                logging.warning(f"Failed to send reminder: {e}")

        asyncio.create_task(_delayed_send())
    await update.message.reply_text(
        f"Reminder set. In {_format_duration(delay_seconds)} will remind about: {text}"
    )


async def handle_dm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle direct messages."""
    MET_UPDATES.inc()
    chat_id = get_chat_id(update)
    settings = get_chat_settings(chat_id)

    question = update.message.text or ""

    if not question:
        question = "Hello! How can I help?"

    # Prepare messages
    messages = [{"role": "user", "content": question}]

    # Add recent context if memory is on
    if settings["memory_on"] and settings["recent_messages"]:
        recent_context = settings["recent_messages"][-5:]  # Last 5 messages for DMs
        context_text = "Recent conversation context:\n"
        for msg in recent_context:
            context_text += f"{msg['role']}: {msg['content']}\n"
        messages.insert(0, {"role": "system", "content": context_text})

    # Get response from Mistral
    response = get_mistral_response(messages)

    # Send response
    await update.message.reply_text(sanitize_output(response))

    # Add to memory
    add_message_to_memory(chat_id, "user", question, update.effective_user.username)
    add_message_to_memory(chat_id, "assistant", response)


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular group messages (for memory only)."""
    text = update.message.text or ""
    MET_UPDATES.inc()

    # Store ALL messages in memory (regardless of memory setting)
    chat_id = get_chat_id(update)

    if text:
        # Determine message type
        if text.startswith("/"):
            message_type = "command"
        else:
            message_type = "message"

        add_message_to_memory(
            chat_id, message_type, text, update.effective_user.username
        )

    if BOT_USERNAME and text:
        tl = text.lower()
        bn = BOT_USERNAME.lower()
        mentioned = (f"@{bn}" in tl) or tl.startswith(bn) or (f" {bn}" in tl)
        if mentioned:
            import re

            clean_text = re.sub(
                rf"@?{re.escape(BOT_USERNAME)}", "", text, flags=re.IGNORECASE
            ).strip()
            if not clean_text:
                clean_text = "Hello! How can I help?"
            Ctx = type("Ctx", (), {})
            fake_context = Ctx()
            fake_context.args = clean_text.split()
            await ask_command(update, fake_context)
            return


async def post_init(application):
    """Post-initialization handler."""
    global BOT_USERNAME
    global READY
    me = await application.bot.get_me()
    BOT_USERNAME = me.username
    logging.info(f"Bot initialized. Username: @{BOT_USERNAME}")
    READY = True
    MET_READY.set(1)

    # Start Redis durable reminder loop if configured
    if redis_client is not None:

        async def reminder_loop():
            while True:
                try:
                    now = int(time.time())
                    # fetch due reminders
                    items = redis_client.zrangebyscore("reminders", 0, now)
                    if items:
                        pipe = redis_client.pipeline()
                        for item in items:
                            pipe.zrem("reminders", item)
                        pipe.execute()
                        for item in items:
                            try:
                                data = json.loads(item)
                                chat_id = data.get("chat_id")
                                text = data.get("text")
                                if chat_id and text:
                                    await application.bot.send_message(
                                        chat_id=chat_id, text=f"Reminder: {text}"
                                    )
                            except Exception as e:
                                logging.warning(f"Failed to send durable reminder: {e}")
                    await asyncio.sleep(2)
                except Exception as e:
                    logging.warning(f"Reminder loop error: {e}")
                    await asyncio.sleep(5)

        asyncio.create_task(reminder_loop())

    # Start HTTP health server in background thread
    def _start_health_server():
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Simple routing (no query parsing)
                if self.path == "/healthz":
                    # Liveness with subsystem info
                    status = {
                        "status": "ok",
                        "ready": READY,
                        "uptime_s": int(time.time() - STARTED_AT),
                        "bot_username": BOT_USERNAME or "unknown",
                        "redis": (
                            "up"
                            if (redis_client and self._redis_ok())
                            else ("down" if redis_client else "disabled")
                        ),
                    }
                    body = json.dumps(status).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/ready":
                    # Readiness (returns 200 only when bot post-init completed)
                    if READY:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"ready")
                    else:
                        self.send_response(503)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"not ready")
                elif self.path == "/metrics":
                    # Update uptime gauge just-in-time
                    MET_UPTIME.set(int(time.time() - STARTED_AT))
                    data = generate_latest(registry)
                    self.send_response(200)
                    self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Reduce noise from health probes
                return

            def _redis_ok(self) -> bool:
                try:
                    if not redis_client:
                        return False
                    # lightweight ping
                    return bool(redis_client.ping())
                except Exception:
                    return False

        port = int(os.getenv("HEALTH_PORT", "8080"))
        httpd = HTTPServer(("0.0.0.0", port), _Handler)
        logging.info(f"Health/metrics server on :{port}")
        httpd.serve_forever()

    threading.Thread(target=_start_health_server, daemon=True).start()


def main():
    """Main function to run the bot."""
    logging.info("Starting GroupMind bot...")

    # Create application
    application = (
        Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    )

    # Add command handlers
    logging.info("Registering command handlers...")
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("summarize", summarize_command))
    application.add_handler(CommandHandler("translate", translate_command))
    application.add_handler(CommandHandler("memory", memory_command))
    application.add_handler(CommandHandler(["remind", "reminder"], remind_command))
    # sentiment command removed
    application.add_handler(CommandHandler("archetype", archetype_command))
    logging.info("Command handlers registered")

    # Add message handlers (order matters - more specific first)
    application.add_handler(MessageHandler(filters.ChatType.PRIVATE, handle_dm))
    application.add_handler(
        MessageHandler(filters.ChatType.GROUPS, handle_group_message)
    )

    logging.info(
        "Message handlers registered: DMs -> handle_dm; Groups -> handle_group_message"
    )

    # Start the bot
    logging.info("Bot is running. Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()
