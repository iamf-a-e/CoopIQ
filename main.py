import json
import logging
import os
import sys
import time

import google.generativeai as genai
import requests
from flask import Flask, g, request, jsonify
from werkzeug.exceptions import HTTPException

from retrieval import retrieve, format_context

# force=True so this config wins even if the host already set up logging.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s [%(name)s] %(message)s",
    force=True,
)
logger = logging.getLogger("coopiq")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration (environment variables)
# ---------------------------------------------------------------------------
WA_TOKEN = os.environ.get("WA_TOKEN")
PHONE_ID = os.environ.get("PHONE_ID")
GEN_API = os.environ.get("GEN_API")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")

UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

WHATSAPP_API_URL = f"https://graph.facebook.com/v20.0/{PHONE_ID}/messages" if PHONE_ID else None

# Check this against the currently available Gemini models; override with GEN_MODEL.
GENERATION_MODEL = os.environ.get("GEN_MODEL", "gemini-3.5-flash-lite")

# Set LOG_PAYLOADS=0 once things work: payloads contain farmers' phone numbers and messages.
LOG_PAYLOADS = os.environ.get("LOG_PAYLOADS", "1") == "1"

# Conversations are stored permanently. Only the most recent messages are sent to
# Gemini on each turn, to keep latency and token cost bounded.
CONTEXT_MESSAGES = 20
WHATSAPP_MAX_CHARS = 4000

if GEN_API:
    genai.configure(api_key=GEN_API)


def _set(value) -> str:
    return "set" if value else "MISSING"


def _mask(number) -> str:
    """Show only the last 4 digits of a phone number in logs."""
    return f"***{number[-4:]}" if number else "?"


# Logged once per cold start. Shows which env vars exist (never their values).
logger.info(
    "CoopIQ starting | model=%s | WA_TOKEN=%s PHONE_ID=%s GEN_API=%s VERIFY_TOKEN=%s "
    "| redis=%s | log_payloads=%s",
    GENERATION_MODEL,
    _set(WA_TOKEN),
    _set(PHONE_ID),
    _set(GEN_API),
    _set(VERIFY_TOKEN),
    "on" if (UPSTASH_URL and UPSTASH_TOKEN) else "off (in-memory fallback)",
    LOG_PAYLOADS,
)

SYSTEM_PROMPT = """You are CoopIQ, a friendly, practical WhatsApp assistant for \
smallholder and commercial poultry farmers in Zambia. You help with broiler, layer, \
and village chicken farming: housing, brooding, feeding, vaccination and disease \
prevention, biosecurity, budgeting, and marketing.

How to behave:
- Talk naturally. Handle greetings, thanks, and follow-up questions like a helpful \
person would. Use the conversation so far to understand what "it", "that", or "how \
often" refers to.
- Reply in the same language the farmer writes or speaks in. You support English, \
Nyanja (Chinyanja), Bemba, Tonga, and Lozi, and farmers often mix these with English \
words. If they switch language, switch with them. Write simply. If you are not \
confident you can write correctly in their language, reply in clear simple English and \
say so briefly instead of guessing.
- Some messages are voice notes that were transcribed automatically, so the text may \
contain mishearings, especially of local words. Use the context to work out the most \
likely meaning. When a message is marked as a voice note with medium or low \
confidence, begin by briefly saying what you understood, in the farmer's language, so \
they can correct you.
- Each message may include REFERENCE NOTES from the CoopIQ knowledge base. Prefer them \
when they are relevant and keep any numbers in them exact. If the notes do not cover \
the question, answer from your own general poultry-farming knowledge, and say so \
briefly when you are less sure.
- For vaccine schedules, medicine names, dosages, withdrawal periods, and costs, only \
give specific figures if they appear in the reference notes. Otherwise give general \
guidance and tell the farmer to confirm with a local veterinary or extension officer.
- If birds are dying in large numbers or showing severe signs (bloody droppings, \
gasping, sudden mass deaths), say it is urgent and tell them to contact a vet or the \
nearest veterinary office right away.
- If a question is outside poultry farming, answer briefly if it is harmless, then \
steer back to what you can help with. Never invent facts.

Format for WhatsApp: short, plain language, short paragraphs or a few simple bullet \
points, no markdown headers or tables. Keep replies under about 150 words unless the \
farmer asks for detail."""


# ---------------------------------------------------------------------------
# Conversation memory (Upstash Redis via REST, with in-process fallback)
# ---------------------------------------------------------------------------
_local_history = {}
_local_seen = set()


def _redis(*command):
    resp = requests.post(
        UPSTASH_URL,
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        json=list(command),
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json().get("result")


def _redis_enabled() -> bool:
    return bool(UPSTASH_URL and UPSTASH_TOKEN)


def _history_key(sender: str) -> str:
    return f"coopiq:chat:{sender}"


def _start_with_user(messages: list) -> list:
    """Gemini requires history to begin with a user turn."""
    while messages and messages[0].get("role") != "user":
        messages = messages[1:]
    return messages


def load_history(sender: str) -> list:
    """Return the most recent messages in Gemini format:
    [{"role": "user"|"model", "parts": [text]}, ...]. The full history stays in storage."""
    if _redis_enabled():
        try:
            raw_items = _redis("LRANGE", _history_key(sender), -CONTEXT_MESSAGES, -1) or []
            history = _start_with_user([json.loads(item) for item in raw_items])
            logger.info("History loaded from Redis: %d messages for %s", len(history), _mask(sender))
            return history
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load history from Redis")
            return []
    history = _start_with_user(_local_history.get(sender, [])[-CONTEXT_MESSAGES:])
    logger.info("History loaded from memory: %d messages for %s", len(history), _mask(sender))
    return history


def append_history(sender: str, user_text: str, answer: str) -> None:
    """Append one farmer/assistant exchange to the permanent history (no expiry)."""
    user_msg = {"role": "user", "parts": [user_text]}
    model_msg = {"role": "model", "parts": [answer]}
    if _redis_enabled():
        try:
            _redis("RPUSH", _history_key(sender), json.dumps(user_msg), json.dumps(model_msg))
            logger.info("History saved to Redis for %s", _mask(sender))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to save history to Redis")
        return
    _local_history.setdefault(sender, []).extend([user_msg, model_msg])


def already_processed(message_id: str) -> bool:
    """WhatsApp retries webhooks that respond slowly, so skip repeated message IDs."""
    if not message_id:
        return False
    if _redis_enabled():
        try:
            return _redis("SET", f"coopiq:seen:{message_id}", "1", "NX", "EX", 600) is None
        except Exception:  # noqa: BLE001
            logger.exception("Failed dedupe check")
            return False
    if message_id in _local_seen:
        return True
    if len(_local_seen) > 1000:
        _local_seen.clear()
    _local_seen.add(message_id)
    return False


# ---------------------------------------------------------------------------
# Gemini answer generation
# ---------------------------------------------------------------------------
def generate_answer(sender: str, user_text: str, voice_confidence: str = None) -> str:
    """Let Gemini handle the whole conversation, with retrieved notes as optional context."""
    started = time.time()
    history = load_history(sender)

    # Include the previous farmer message so short follow-ups still retrieve well.
    previous_user = [h["parts"][0] for h in history if h["role"] == "user"][-1:]
    retrieval_query = " ".join(previous_user + [user_text])

    context = ""
    try:
        passages = retrieve(retrieval_query, top_k=4)
        logger.info("Retrieval returned %d passages", len(passages) if passages else 0)
        if passages:
            context = format_context(passages)
    except Exception:  # noqa: BLE001
        logger.exception("Retrieval failed; continuing without reference notes")

    notes = f"REFERENCE NOTES:\n{context}" if context else "(No reference notes matched this message.)"
    channel = ""
    if voice_confidence:
        channel = (
            "[This message is an automatic transcript of a voice note. "
            f"Transcription confidence: {voice_confidence}.]\n\n"
        )
    turn = f"{notes}\n\n{channel}FARMER'S MESSAGE:\n{user_text}"

    logger.info("Calling Gemini model=%s (prompt ~%d chars)", GENERATION_MODEL, len(turn))
    gemini_started = time.time()
    model = genai.GenerativeModel(GENERATION_MODEL, system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat(history=history)
    response = chat.send_message(
        turn,
        generation_config={"temperature": 0.4, "max_output_tokens": 700},
    )
    answer = response.text.strip()
    logger.info(
        "Gemini replied in %.1fs (%d chars); total generate_answer %.1fs",
        time.time() - gemini_started,
        len(answer),
        time.time() - started,
    )

    # Store the plain message, not the injected notes, so history stays small.
    append_history(sender, user_text, answer)

    return answer


# ---------------------------------------------------------------------------
# WhatsApp Cloud API helpers
# ---------------------------------------------------------------------------
def send_whatsapp_message(to: str, body: str) -> None:
    if not (WA_TOKEN and WHATSAPP_API_URL):
        logger.warning(
            "WA_TOKEN or PHONE_ID not configured (WA_TOKEN=%s PHONE_ID=%s); skipping send. "
            "Reply was: %s",
            _set(WA_TOKEN),
            _set(PHONE_ID),
            body,
        )
        return

    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body[:WHATSAPP_MAX_CHARS]},
    }
    logger.info("Sending WhatsApp message to %s (%d chars)", _mask(to), len(body))
    try:
        resp = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=20)
    except requests.RequestException:
        logger.exception("WhatsApp send raised a network error")
        return

    if resp.status_code >= 300:
        try:
            err = resp.json().get("error", {})
        except ValueError:
            err = {}
        code = err.get("code")
        logger.error(
            "WhatsApp send FAILED to=%s http=%s code=%s message=%s | raw=%s",
            _mask(to), resp.status_code, code, err.get("message"), resp.text[:500],
        )
        if code == 190 or resp.status_code == 401:
            logger.error("HINT: WA_TOKEN is expired or invalid. Temporary tokens last 24 hours.")
        elif code == 131030:
            logger.error("HINT: recipient is not on the allowed recipient list (dev mode).")
        return

    try:
        wamid = resp.json()["messages"][0]["id"]
    except (ValueError, KeyError, IndexError):
        wamid = "?"
    logger.info("WhatsApp send OK to=%s http=%s wamid=%s", _mask(to), resp.status_code, wamid)


def download_whatsapp_media(media_id: str):
    """Fetch a WhatsApp media file (e.g. a voice note). Returns (bytes, mime_type)."""
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    meta = requests.get(f"https://graph.facebook.com/v20.0/{media_id}", headers=headers, timeout=15)
    meta.raise_for_status()
    info = meta.json()
    media = requests.get(info["url"], headers=headers, timeout=30)
    media.raise_for_status()
    logger.info("Downloaded media %s: %d bytes, mime=%s", media_id, len(media.content), info.get("mime_type"))
    return media.content, info.get("mime_type")


TRANSCRIBE_PROMPT = """You are transcribing a WhatsApp voice note from a farmer in Zambia.
The speaker may use English, Nyanja (Chinyanja), Bemba, Tonga, or Lozi, and often mixes
local languages with English words.

Transcribe exactly what is said, in the language it was spoken. Do not translate and do
not answer the question. Keep code-switching as spoken. If the audio is silent, music
only, or you cannot make out the words, return an empty transcript.

Rate your confidence honestly: "high" only if the audio was clear and you understood
every important word, "medium" if some words were uncertain, "low" if you were mostly
guessing.

Return only JSON: {"transcript": "...", "language": "...", "confidence": "high|medium|low"}"""


def transcribe_voice_note(audio: dict):
    """Return (transcript, confidence) for a WhatsApp voice note, or (None, None) on failure."""
    try:
        data, media_mime = download_whatsapp_media(audio["id"])
        mime = (audio.get("mime_type") or media_mime or "audio/ogg").split(";")[0].strip()
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(
            [TRANSCRIBE_PROMPT, {"mime_type": mime, "data": data}],
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        result = json.loads(response.text)
        transcript = (result.get("transcript") or "").strip()
        confidence = str(result.get("confidence", "low")).lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "low"
        logger.info(
            "Voice note transcribed (language=%s, confidence=%s): %s",
            result.get("language"), confidence, transcript,
        )
        return (transcript or None), confidence
    except Exception:  # noqa: BLE001
        logger.exception("Voice note transcription failed")
        return None, None


def extract_incoming_message(payload: dict):
    """Return (sender, message_id, text, audio) from a WhatsApp webhook payload.

    sender is None for non-message events (e.g. delivery status updates).
    text is None unless the message is text or a button/list reply.
    audio is {"id": ..., "mime_type": ...} for voice notes and audio files, else None.
    """
    try:
        value = payload["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if not messages:
            return None, None, None, None
        message = messages[0]
        sender = message["from"]
        message_id = message.get("id")
        msg_type = message.get("type")
        logger.info("Parsed message: type=%s from=%s id=%s", msg_type, _mask(sender), message_id)

        if msg_type == "text":
            return sender, message_id, message["text"]["body"], None
        if msg_type == "interactive":
            interactive = message.get("interactive", {})
            reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
            return sender, message_id, reply.get("title"), None
        if msg_type == "audio":
            audio = message.get("audio", {})
            return sender, message_id, None, {"id": audio.get("id"), "mime_type": audio.get("mime_type")}
        return sender, message_id, None, None
    except (KeyError, IndexError, TypeError):
        logger.exception("Could not parse webhook payload")
        return None, None, None, None


def log_statuses(payload: dict) -> None:
    """Log delivery status updates. Failed deliveries include Meta's error codes."""
    try:
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                for status in change.get("value", {}).get("statuses", []) or []:
                    errors = status.get("errors")
                    if errors:
                        logger.error(
                            "Delivery status %s for %s: %s",
                            status.get("status"), _mask(status.get("recipient_id")), errors,
                        )
                    else:
                        logger.info(
                            "Delivery status %s for %s",
                            status.get("status"), _mask(status.get("recipient_id")),
                        )
    except Exception:  # noqa: BLE001
        logger.exception("Could not read status updates")


# ---------------------------------------------------------------------------
# Request / response logging and error handling
# ---------------------------------------------------------------------------
@app.before_request
def log_request():
    g.started = time.time()
    forwarded = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(
        "REQUEST %s %s from=%s ua=%s content_length=%s",
        request.method,
        request.path,
        forwarded,
        request.headers.get("User-Agent", "-")[:60],
        request.content_length,
    )


@app.after_request
def log_response(response):
    elapsed_ms = int((time.time() - getattr(g, "started", time.time())) * 1000)
    logger.info("RESPONSE %s %s -> %s (%d ms)", request.method, request.path, response.status_code, elapsed_ms)
    return response


@app.errorhandler(Exception)
def log_unhandled(exc):
    if isinstance(exc, HTTPException):
        return exc
    logger.exception("Unhandled exception on %s %s", request.method, request.path)
    return jsonify({"status": "error"}), 500


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "CoopIQ"})


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    logger.info(
        "Webhook verification: mode=%s token_matches=%s verify_token_configured=%s",
        mode, token == VERIFY_TOKEN, bool(VERIFY_TOKEN),
    )

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook verification SUCCEEDED")
        return challenge, 200
    logger.warning("Webhook verification FAILED")
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def handle_webhook():
    payload = request.get_json(silent=True) or {}
    if not payload:
        logger.warning("POST /webhook had an empty or non-JSON body")
    if LOG_PAYLOADS:
        logger.info("Incoming webhook payload: %s", json.dumps(payload)[:3000])
    else:
        logger.info("Incoming webhook payload received (%d bytes)", len(request.get_data() or b""))

    sender, message_id, text, audio = extract_incoming_message(payload)

    if not sender:
        # Not a user message (e.g. a status update): acknowledge and ignore.
        logger.info("Ignored: not a user message (status update or unrecognized event)")
        log_statuses(payload)
        return jsonify({"status": "ignored"}), 200

    if already_processed(message_id):
        logger.info("Ignored duplicate message %s", message_id)
        return jsonify({"status": "duplicate"}), 200

    voice_confidence = None
    if audio:
        logger.info("Voice note received, transcribing")
        text, voice_confidence = transcribe_voice_note(audio)
        if not text:
            logger.warning("Voice note produced no transcript")
            send_whatsapp_message(
                sender,
                "Sorry, I could not hear that voice note clearly. Please record it again "
                "in a quieter place, or type your question.",
            )
            return jsonify({"status": "received"}), 200

    if not text:
        logger.info("Unsupported message type; sending help text")
        send_whatsapp_message(
            sender,
            "I can read text messages and voice notes. Please send your poultry-farming "
            "question that way and I'll do my best to help!",
        )
        return jsonify({"status": "received"}), 200

    logger.info("Answering message from %s: %s", _mask(sender), text[:200])
    try:
        answer = generate_answer(sender, text, voice_confidence)
    except Exception:  # noqa: BLE001
        logger.exception("Error generating answer (model=%s)", GENERATION_MODEL)
        answer = (
            "Sorry, I ran into a problem answering that just now. "
            "Please try again in a moment."
        )

    send_whatsapp_message(sender, answer)
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
