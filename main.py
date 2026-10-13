import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

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

# Supabase: use the project URL and the SERVICE ROLE key (server-side only, never expose it).
SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

GRAPH_VERSION = os.environ.get("GRAPH_VERSION", "v21.0")
GRAPH_URL = f"https://graph.facebook.com/{GRAPH_VERSION}"

# Check this against the currently available Gemini models; override with GEN_MODEL.
GENERATION_MODEL = os.environ.get("GEN_MODEL", "gemini-3.5-flash-lite")

# Set LOG_PAYLOADS=0 once things work: payloads contain farmers' phone numbers and messages.
LOG_PAYLOADS = os.environ.get("LOG_PAYLOADS", "1") == "1"

# Optional: run a tiny extra Gemini call to detect the language of each text message
# before answering, then tell the main prompt explicitly which language to use.
# Costs one extra call per message. Set DETECT_LANGUAGE=1 to turn it on.
DETECT_LANGUAGE = os.environ.get("DETECT_LANGUAGE", "0") == "1"
SUPPORTED_LANGUAGES = ["English", "Nyanja", "Bemba", "Tonga", "Lozi"]

# Conversations are stored permanently. Only the most recent messages are sent to
# Gemini on each turn, to keep latency and token cost bounded.
CONTEXT_MESSAGES = 20
WHATSAPP_MAX_CHARS = 4000

# Photo check (same subjects and image types as the CoopIQ Supabase edge function).
DIAGNOSE_SUBJECTS = ["Poultry droppings", "Poultry bird", "Cattle", "Goat or sheep", "Pig"]
IMAGE_MIMES = ("image/jpeg", "image/png", "image/webp")
MAX_CAPTION_CHARS = 600

if GEN_API:
    genai.configure(api_key=GEN_API)


def _set(value) -> str:
    return "set" if value else "MISSING"


def _mask(number) -> str:
    """Show only the last 4 digits of a phone number in logs."""
    return f"***{number[-4:]}" if number else "?"


# Logged once per cold start. Shows which env vars exist (never their values).
logger.info(
    "CoopIQ starting | model=%s | graph=%s | WA_TOKEN=%s PHONE_ID=%s GEN_API=%s VERIFY_TOKEN=%s "
    "| supabase=%s | log_payloads=%s | detect_language=%s",
    GENERATION_MODEL,
    GRAPH_VERSION,
    _set(WA_TOKEN),
    _set(PHONE_ID),
    _set(GEN_API),
    _set(VERIFY_TOKEN),
    "on" if (SUPABASE_URL and SUPABASE_KEY) else "off (in-memory fallback)",
    LOG_PAYLOADS,
    DETECT_LANGUAGE,
)

# Shared by the chat prompt and the photo-check prompt so both detect language the same way.
LANGUAGE_RULE = """Language: work out the language of the farmer's latest message yourself and \
reply in that same language. You support English, Nyanja (Chinyanja), Bemba, Tonga, and Lozi. \
Farmers often mix these with English words: reply in the main language they used. If they \
switch language, switch with them, even if earlier messages were in a different language. \
A greeting or short phrase written in a local language (for example "Muli shani", \
"Muli bwanji", "Mwabuka buti", "Lumela") is a clear signal: reply in that language. \
Only fall back to the language of earlier messages when the latest message has no language \
signal at all, such as "ok", "yes", "thanks" or a number. If you still cannot tell, or they \
write in a language you do not support, reply in simple English. Use short, simple \
sentences, and keep technical terms such as vaccine, drug, and disease names in English."""

SYSTEM_PROMPT = f"""You are CoopIQ, a friendly, practical WhatsApp assistant for \
smallholder and commercial poultry farmers in Zambia. You help with broiler, layer, \
and village chicken farming: housing, brooding, feeding, vaccination and disease \
prevention, biosecurity, budgeting, and marketing. You can also answer basic health \
and feeding questions about cattle, goats, sheep, and pigs.

How to behave:
- Talk naturally. Handle greetings, thanks, and follow-up questions like a helpful \
person would. Use the conversation so far to understand what "it", "that", or "how \
often" refers to.
- {LANGUAGE_RULE}
- Some messages are voice notes that were transcribed automatically, so the text may \
contain mishearings, especially of local words. Use the context to work out the most \
likely meaning. A voice note may be marked with the language heard in the audio: treat \
that as a hint, and trust the transcript text if they disagree. When a message is marked \
as a voice note with medium or low confidence, begin by briefly saying what you \
understood, in the farmer's language, so they can correct you.
- Earlier turns that start with [Photo of ...] were photo checks you already answered. \
Use them to understand follow-up questions about that photo.
- Each message may include REFERENCE NOTES from the CoopIQ knowledge base. Prefer them \
when they are relevant and keep any numbers in them exact. If the notes do not cover \
the question, answer from your own general farming knowledge, and say so \
briefly when you are less sure.
- For vaccine schedules, medicine names, dosages, withdrawal periods, and costs, only \
give specific figures if they appear in the reference notes. Otherwise give general \
guidance and tell the farmer to confirm with a local veterinary or extension officer.
- If birds are dying in large numbers or showing severe signs (bloody droppings, \
gasping, sudden mass deaths), say it is urgent and tell them to contact a vet or the \
nearest veterinary office right away.
- If the farmer's message is only a greeting, thanks, or small talk, reply warmly in one \
or two short sentences WRITTEN IN THE FARMER'S LANGUAGE (follow the language rule above, \
not English by default), say you are CoopIQ, and invite them to ask about poultry or \
livestock health, feeding, or vaccination. Translate that invitation into their language \
too. For these messages ignore the reference notes. Examples of greetings and the \
language they signal:
    "Muli shani" -> Bemba: reply in Bemba.
    "Muli bwanji" or "Zikomo" -> Nyanja: reply in Nyanja.
    "Mwabuka buti" or "Mwalibizya buti" -> Tonga: reply in Tonga.
    "Lumela" or "Muzuhile" -> Lozi: reply in Lozi.
    "Hello" or "Good morning" -> English: reply in English.
- If a question has nothing to do with farming, say kindly that you can only help with \
poultry and livestock, and invite a farm question, in the farmer's language. \
Never invent facts.

Format for WhatsApp: short, plain language, short paragraphs or a few simple bullet \
points, no markdown headers or tables. Keep replies under about 150 words unless the \
farmer asks for detail."""


DIAGNOSE_INSTRUCTIONS = f"""You are the vision model behind CoopIQ's photo disease check on \
WhatsApp. A farmer in Zambia sent the photo.

First decide what the photo shows and set subject to exactly one of: Poultry droppings, \
Poultry bird, Cattle, Goat or sheep, Pig, Other. Use Other when the photo shows none of \
these (for example a person, a document, feed, or a building).

Look at the image and return ONLY a JSON object with these keys:
subject: as above.
condition: the most likely condition (poultry examples: Coccidiosis, Newcastle disease, \
Salmonellosis, Gumboro disease, Fowl pox, Healthy or no clear signs, Unclear). For \
livestock, name the most likely condition or "Unclear".
confidence: "High", "Medium" or "Low". Use "Low" when the image is blurry, dark or does \
not show the problem.
summary: one or two plain sentences on what you see that supports this.
recommendation: what to do next. Use the REFERENCE passages where relevant. Do not invent \
doses; say "confirm the dose with an agrovet" when no dose is in the reference.
isolate: true if affected animals should be separated from the rest.
escalate: true if an agrovet or vet should be contacted (always true for Newcastle disease, \
bloody droppings, sudden deaths, or Low confidence, unless subject is Other).
language: the language you wrote reply in.
reply: the complete WhatsApp message to send to the farmer, written in the farmer's \
language. Plain text, no markdown, under about 150 words. It must say what you see and the \
likely condition with how sure you are, what to do next, tell them to separate affected \
animals if isolate is true, tell them to contact an agrovet or vet if escalate is true, and \
end with one short line saying this is a photo-based guide and not a laboratory result. \
If subject is Other, set condition to "Unclear", confidence to "Low", isolate and escalate \
to false, and make reply kindly ask for a clear, well-lit photo of the sick bird, its \
droppings, or the animal.

Keep subject, condition and confidence in English exactly as described above.

{LANGUAGE_RULE}
For a photo there is no message text to read the language from. Use the language of the \
farmer's caption if there is one. If there is no caption, use the language of the farmer's \
recent messages. If neither exists, use English.

This is decision support, not a laboratory result."""

UNSUPPORTED_IMAGE_REPLY = (
    "Sorry, I can only check normal photos (JPEG, PNG or WebP). Please take a new photo "
    "of the bird, droppings or animal and send it as a picture."
)
DIAGNOSE_FAILED_REPLY = (
    "Sorry, I could not check that photo just now. Please try again in a moment, "
    "or send a clearer photo."
)


# ---------------------------------------------------------------------------
# Storage (Supabase Postgres via the REST API, with in-process fallback)
#
# Tables (see supabase_schema.sql):
#   chat_messages      one row per message, kept permanently
#   processed_messages WhatsApp message IDs already handled (dedupe)
#   farmers            one row per farmer: first_seen, last_seen, last_language
# ---------------------------------------------------------------------------
_local_history = {}
_local_seen = set()
_local_farmers = {}


def _supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _sb(method: str, table: str, params: dict = None, body=None, prefer: str = None):
    """Call the Supabase REST API (PostgREST) and return the parsed JSON, or None if empty."""
    headers = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
    # Legacy service_role keys are JWTs (start with "eyJ") and go in Authorization too.
    # New sb_secret_... keys are not JWTs and only need the apikey header.
    if SUPABASE_KEY.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {SUPABASE_KEY}"
    if prefer:
        headers["Prefer"] = prefer
    resp = requests.request(
        method,
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=headers,
        params=params,
        json=body,
        timeout=8,
    )
    if resp.status_code >= 300:
        logger.error("Supabase %s %s failed: http=%s body=%s", method, table, resp.status_code, resp.text[:300])
    resp.raise_for_status()
    return resp.json() if resp.content else None


def _start_with_user(messages: list) -> list:
    """Gemini requires history to begin with a user turn."""
    while messages and messages[0].get("role") != "user":
        messages = messages[1:]
    return messages


def load_history(sender: str) -> list:
    """Return the most recent messages in Gemini format:
    [{"role": "user"|"model", "parts": [text]}, ...]. The full history stays in storage."""
    if _supabase_enabled():
        try:
            rows = _sb(
                "GET",
                "chat_messages",
                params={
                    "select": "role,content",
                    "sender": f"eq.{sender}",
                    "order": "id.desc",
                    "limit": str(CONTEXT_MESSAGES),
                },
            ) or []
            rows.reverse()  # newest-first from the query, oldest-first for Gemini
            history = _start_with_user([{"role": r["role"], "parts": [r["content"]]} for r in rows])
            logger.info("History loaded from Supabase: %d messages for %s", len(history), _mask(sender))
            return history
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load history from Supabase")
            return []
    history = _start_with_user(_local_history.get(sender, [])[-CONTEXT_MESSAGES:])
    logger.info("History loaded from memory: %d messages for %s", len(history), _mask(sender))
    return history


def append_history(sender: str, user_text: str, answer: str) -> None:
    """Append one farmer/assistant exchange to the permanent history (no expiry)."""
    if _supabase_enabled():
        try:
            _sb(
                "POST",
                "chat_messages",
                body=[
                    {"sender": sender, "role": "user", "content": user_text},
                    {"sender": sender, "role": "model", "content": answer},
                ],
                prefer="return=minimal",
            )
            logger.info("History saved to Supabase for %s", _mask(sender))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to save history to Supabase")
        return
    _local_history.setdefault(sender, []).extend(
        [
            {"role": "user", "parts": [user_text]},
            {"role": "model", "parts": [answer]},
        ]
    )


def touch_farmer(sender: str, language: str = None) -> None:
    """Create or update the farmer's state row: last_seen, and last_language when known."""
    now = datetime.now(timezone.utc).isoformat()
    if _supabase_enabled():
        row = {"phone": sender, "last_seen": now}
        if language:
            row["last_language"] = language  # omitted when unknown so we never overwrite it
        try:
            _sb(
                "POST",
                "farmers",
                params={"on_conflict": "phone"},
                body=row,
                prefer="resolution=merge-duplicates,return=minimal",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update farmer state")
        return
    state = _local_farmers.setdefault(sender, {})
    state["last_seen"] = now
    if language:
        state["last_language"] = language


def already_processed(message_id: str) -> bool:
    """WhatsApp retries webhooks that respond slowly, so skip repeated message IDs.

    The insert is atomic: if the ID already exists, Postgres ignores it and returns no rows.
    """
    if not message_id:
        return False
    if _supabase_enabled():
        try:
            rows = _sb(
                "POST",
                "processed_messages",
                params={"on_conflict": "message_id"},
                body={"message_id": message_id},
                prefer="resolution=ignore-duplicates,return=representation",
            )
            return not rows
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
# Language detection (optional, enabled with DETECT_LANGUAGE=1)
# ---------------------------------------------------------------------------
def detect_language(text: str):
    """Return one of SUPPORTED_LANGUAGES for the text, or None if unknown or on failure."""
    if not text or not text.strip():
        return None
    options = ", ".join(SUPPORTED_LANGUAGES + ["Unknown"])
    prompt = (
        "Identify the language of this WhatsApp message from a farmer in Zambia. "
        "Nyanja is also called Chinyanja. Greetings count: 'Muli shani' is Bemba, "
        "'Muli bwanji' is Nyanja, 'Mwabuka buti' is Tonga, 'Lumela' is Lozi. "
        "If the message mixes English words with a local language, answer with the local "
        f"language. Reply with exactly one of: {options}. Reply with nothing else.\n\n"
        f"Message: {text[:500]}"
    )
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 10},
        )
        answer = response.text.strip().strip(".").title()
        language = answer if answer in SUPPORTED_LANGUAGES else None
        logger.info("Detected language: %s (raw=%r)", language, answer)
        return language
    except Exception:  # noqa: BLE001
        logger.exception("Language detection failed; continuing without it")
        return None


# ---------------------------------------------------------------------------
# Gemini answer generation
# ---------------------------------------------------------------------------
def _retrieve_context(query: str, top_k: int) -> str:
    """Return formatted reference notes for a query, or "" if nothing matched or retrieval failed."""
    try:
        passages = retrieve(query, top_k=top_k)
        logger.info("Retrieval returned %d passages", len(passages) if passages else 0)
        if passages:
            return format_context(passages)
    except Exception:  # noqa: BLE001
        logger.exception("Retrieval failed; continuing without reference notes")
    return ""


def generate_answer(
    sender: str,
    user_text: str,
    voice_confidence: str = None,
    voice_language: str = None,
) -> str:
    """Let Gemini handle the whole conversation, with retrieved notes as optional context."""
    started = time.time()
    history = load_history(sender)

    # Include the previous farmer message so short follow-ups still retrieve well.
    previous_user = [h["parts"][0] for h in history if h["role"] == "user"][-1:]
    retrieval_query = " ".join(previous_user + [user_text])

    context = _retrieve_context(retrieval_query, top_k=4)

    notes = f"REFERENCE NOTES:\n{context}" if context else "(No reference notes matched this message.)"
    channel = ""
    if voice_confidence:
        heard = f" Language heard in the audio: {voice_language}." if voice_language else ""
        channel = (
            "[This message is an automatic transcript of a voice note. "
            f"Transcription confidence: {voice_confidence}.{heard}]\n\n"
        )

    # The model weighs the latest turn most heavily, so repeat the language instruction
    # here, right next to the message, instead of relying only on the system prompt.
    language_note = (
        "LANGUAGE: Reply in the language of the farmer's message below, even if it is only "
        "a short greeting and even if earlier messages in this chat were in English. "
        "The reference notes are in English; do not let that change your reply language."
    )
    if DETECT_LANGUAGE:
        detected = detect_language(user_text)
        if detected:
            touch_farmer(sender, detected)
            language_note += f" The message was detected as {detected}: reply in {detected}."
    language_note += "\n\n"

    turn = f"{notes}\n\n{channel}{language_note}FARMER'S MESSAGE:\n{user_text}"

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
# Photo diagnosis (mirrors the "diagnose" action of the Supabase edge function)
# ---------------------------------------------------------------------------
def _fallback_diagnosis_reply(condition: str, confidence: str, summary: str,
                              recommendation: str, isolate: bool, escalate: bool) -> str:
    """English message built from the structured fields, used only if the model gave no reply."""
    lines = [f"Likely condition: {condition} (confidence: {confidence})."]
    if summary:
        lines.append(summary)
    if recommendation:
        lines.append(recommendation)
    if isolate:
        lines.append("Separate the affected animals from the rest.")
    if escalate:
        lines.append("Please contact an agrovet or vet soon.")
    lines.append("This is a photo-based guide, not a laboratory result.")
    return "\n\n".join(lines)


def diagnose_image(sender: str, image: dict) -> str:
    """Check a farmer's photo and return the WhatsApp reply text."""
    started = time.time()
    if not image.get("id"):
        raise ValueError("image message has no media id")

    declared = (image.get("mime_type") or "").split(";")[0].strip().lower()
    if declared and declared not in IMAGE_MIMES:
        logger.info("Unsupported image type: %s", declared)
        return UNSUPPORTED_IMAGE_REPLY

    data, media_mime = download_whatsapp_media(image["id"])
    mime = declared or (media_mime or "").split(";")[0].strip().lower() or "image/jpeg"
    if mime not in IMAGE_MIMES:
        logger.info("Unsupported image type after download: %s", mime)
        return UNSUPPORTED_IMAGE_REPLY

    caption = (image.get("caption") or "").strip()[:MAX_CAPTION_CHARS]
    history = load_history(sender)
    recent_farmer = [
        h["parts"][0] for h in history
        if h.get("role") == "user" and not h["parts"][0].startswith("[Photo of")
    ][-2:]

    # The subject is not known until the model has seen the photo, so retrieve broadly
    # using the caption and recent messages plus generic disease terms.
    retrieval_query = " ".join(
        [t for t in [caption, *recent_farmer, "signs symptoms treatment prevention disease"] if t]
    )
    context = _retrieve_context(retrieval_query, top_k=6)

    # The caption is the strongest language signal for a photo. Detect it explicitly
    # (one tiny extra call) so the decision does not depend on the big prompt.
    caption_language = detect_language(caption) if caption else None
    if caption_language:
        touch_farmer(sender, caption_language)
        language_instruction = (
            f"LANGUAGE: The farmer's caption is in {caption_language}. Write the reply field "
            f"and the language field in {caption_language}, even if the recent messages "
            "below are in a different language."
        )
        recent_label = "Farmer's recent messages (context only, NOT a language signal):"
    else:
        language_instruction = (
            "LANGUAGE: There is no usable caption. Write the reply field in the language "
            "of the farmer's recent messages below, or in English if there are none."
        )
        recent_label = "Farmer's recent messages (use for language and context):"

    prompt = (
        f"{DIAGNOSE_INSTRUCTIONS}\n\n"
        f"Farmer's caption: {caption or 'none'}\n"
        f"{recent_label}\n"
        f"{chr(10).join(recent_farmer) if recent_farmer else 'none'}\n\n"
        f"{language_instruction} The reference passages below are in English; do not "
        "let that change the reply language.\n\n"
        f"REFERENCE:\n{context or '(none)'}"
    )

    logger.info("Calling Gemini for photo check model=%s (%d image bytes)", GENERATION_MODEL, len(data))
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(
        [{"mime_type": mime, "data": data}, prompt],
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 2000,
        },
    )
    raw = response.text.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("photo check did not return a JSON object")

    subject = result.get("subject") if result.get("subject") in DIAGNOSE_SUBJECTS else "Other"
    raw_conf = str(result.get("confidence", "low")).lower()
    conf = raw_conf if raw_conf in ("high", "medium", "low") else "low"
    confidence = conf.capitalize()
    condition = str(result.get("condition") or "Unclear")[:120]
    summary = str(result.get("summary") or "")[:800]
    recommendation = str(result.get("recommendation") or "")[:1200]
    isolate = result.get("isolate") is True
    escalate = result.get("escalate") is True or (conf == "low" and subject != "Other")

    logger.info(
        "Photo check: subject=%s condition=%s confidence=%s isolate=%s escalate=%s language=%s (%.1fs)",
        subject, condition, confidence, isolate, escalate, result.get("language"),
        time.time() - started,
    )

    reply = str(result.get("reply") or "").strip()
    if not reply:
        reply = _fallback_diagnosis_reply(condition, confidence, summary, recommendation, isolate, escalate)

    # Keep a plain-text record so follow-up questions about the photo make sense.
    note = f"[Photo of {subject}]"
    if caption:
        note = f"{note} {caption}"
    append_history(sender, note, reply)
    return reply


# ---------------------------------------------------------------------------
# WhatsApp Cloud API helpers
# ---------------------------------------------------------------------------
def send_whatsapp_message(to: str, body: str, phone_number_id: str = None) -> None:
    """Send a text message.

    Replies go out from `phone_number_id` (the number that RECEIVED the message) when
    given, falling back to the PHONE_ID env var. This guarantees the reply appears in
    the same chat the farmer wrote in, even if PHONE_ID is misconfigured.
    """
    sender_id = phone_number_id or PHONE_ID
    if not (WA_TOKEN and sender_id):
        logger.warning(
            "WA_TOKEN or phone number ID not configured (WA_TOKEN=%s sender_id=%s); "
            "skipping send. Reply was: %s",
            _set(WA_TOKEN),
            _set(sender_id),
            body,
        )
        return

    url = f"{GRAPH_URL}/{sender_id}/messages"
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
    logger.info(
        "Sending WhatsApp message to %s from phone_number_id=%s (%d chars)",
        _mask(to), sender_id, len(body),
    )
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
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
        elif code == 100:
            logger.error(
                "HINT: the phone number ID (%s) does not exist or the token cannot access it. "
                "Dashboard 'Test' events use a dummy ID, so this is expected for those.",
                sender_id,
            )
        return

    try:
        data = resp.json()
        wamid = data["messages"][0]["id"]
        msg_status = data["messages"][0].get("message_status")
    except (ValueError, KeyError, IndexError):
        wamid, msg_status = "?", None
    logger.info(
        "WhatsApp send accepted to=%s http=%s wamid=%s message_status=%s "
        "(delivery result follows as a 'Delivery status' log line)",
        _mask(to), resp.status_code, wamid, msg_status,
    )
    if LOG_PAYLOADS:
        logger.info("WhatsApp send response: %s", resp.text[:500])


def download_whatsapp_media(media_id: str):
    """Fetch a WhatsApp media file (e.g. a voice note or photo). Returns (bytes, mime_type)."""
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    meta = requests.get(f"{GRAPH_URL}/{media_id}", headers=headers, timeout=15)
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
    """Return (transcript, confidence, language) for a voice note, or (None, None, None) on failure."""
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
        language = (str(result.get("language") or "").strip() or None)
        logger.info(
            "Voice note transcribed (language=%s, confidence=%s): %s",
            language, confidence, transcript,
        )
        return (transcript or None), confidence, language
    except Exception:  # noqa: BLE001
        logger.exception("Voice note transcription failed")
        return None, None, None


def extract_phone_number_id(payload: dict):
    """Return the ID of the WhatsApp number that received the event, or None."""
    try:
        return payload["entry"][0]["changes"][0]["value"]["metadata"].get("phone_number_id")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def extract_incoming_message(payload: dict):
    """Return (sender, message_id, text, audio, image) from a WhatsApp webhook payload.

    sender is None for non-message events (e.g. delivery status updates).
    text is None unless the message is text or a button/list reply.
    audio is {"id": ..., "mime_type": ...} for voice notes and audio files, else None.
    image is {"id": ..., "mime_type": ..., "caption": ...} for photos, else None.
    """
    try:
        value = payload["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if not messages:
            return None, None, None, None, None
        message = messages[0]
        sender = message["from"]
        message_id = message.get("id")
        msg_type = message.get("type")
        logger.info("Parsed message: type=%s from=%s id=%s", msg_type, _mask(sender), message_id)

        if msg_type == "text":
            return sender, message_id, message["text"]["body"], None, None
        if msg_type == "interactive":
            interactive = message.get("interactive", {})
            reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
            return sender, message_id, reply.get("title"), None, None
        if msg_type == "audio":
            audio = message.get("audio", {})
            return sender, message_id, None, {"id": audio.get("id"), "mime_type": audio.get("mime_type")}, None
        if msg_type == "image":
            img = message.get("image", {})
            return sender, message_id, None, None, {
                "id": img.get("id"),
                "mime_type": img.get("mime_type"),
                "caption": (img.get("caption") or "").strip(),
            }
        return sender, message_id, None, None, None
    except (KeyError, IndexError, TypeError):
        logger.exception("Could not parse webhook payload")
        return None, None, None, None, None


STATUS_HINTS = {
    131030: "recipient is not on the allowed recipient list (add the number under API Setup > To).",
    131047: "24-hour customer-service window is closed; only template messages can be sent.",
    131026: "message could not be delivered to this number (not on WhatsApp, or blocked).",
    131053: "media could not be downloaded or uploaded.",
    190: "access token is invalid or expired.",
}


def log_statuses(payload: dict) -> None:
    """Log delivery status updates (sent / delivered / read / failed) with Meta's error codes."""
    try:
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                for status in change.get("value", {}).get("statuses", []) or []:
                    errors = status.get("errors")
                    if errors:
                        logger.error(
                            "Delivery status %s for %s (wamid=%s): %s",
                            status.get("status"), _mask(status.get("recipient_id")),
                            status.get("id"), errors,
                        )
                        for err in errors:
                            hint = STATUS_HINTS.get(err.get("code"))
                            if hint:
                                logger.error("HINT (%s): %s", err.get("code"), hint)
                    else:
                        logger.info(
                            "Delivery status %s for %s (wamid=%s)",
                            status.get("status"), _mask(status.get("recipient_id")),
                            status.get("id"),
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

    sender, message_id, text, audio, image = extract_incoming_message(payload)

    if not sender:
        # Not a user message (e.g. a status update): acknowledge and ignore.
        logger.info("Ignored: not a user message (status update or unrecognized event)")
        log_statuses(payload)
        return jsonify({"status": "ignored"}), 200

    # Reply from the number that received the message.
    receiving_id = extract_phone_number_id(payload)
    if receiving_id and PHONE_ID and receiving_id != PHONE_ID:
        logger.warning(
            "PHONE_ID env var (%s) differs from the number that received this message (%s). "
            "Replying from the receiving number. Update PHONE_ID in Vercel to fix this.",
            PHONE_ID, receiving_id,
        )
    reply_from = receiving_id or PHONE_ID

    if already_processed(message_id):
        logger.info("Ignored duplicate message %s", message_id)
        return jsonify({"status": "duplicate"}), 200

    touch_farmer(sender)

    # Photos: run the disease check and reply.
    if image:
        logger.info("Photo received (caption=%s), running photo check", "yes" if image.get("caption") else "no")
        try:
            reply = diagnose_image(sender, image)
        except Exception:  # noqa: BLE001
            logger.exception("Photo check failed (model=%s)", GENERATION_MODEL)
            reply = DIAGNOSE_FAILED_REPLY
        send_whatsapp_message(sender, reply, reply_from)
        return jsonify({"status": "received"}), 200

    voice_confidence = None
    voice_language = None
    if audio:
        logger.info("Voice note received, transcribing")
        text, voice_confidence, voice_language = transcribe_voice_note(audio)
        if not text:
            logger.warning("Voice note produced no transcript")
            send_whatsapp_message(
                sender,
                "Sorry, I could not hear that voice note clearly. Please record it again "
                "in a quieter place, or type your question.",
                reply_from,
            )
            return jsonify({"status": "received"}), 200

    if not text:
        logger.info("Unsupported message type; sending help text")
        send_whatsapp_message(
            sender,
            "I can read text messages and voice notes, and check photos of sick birds, "
            "droppings or animals. Please send your farming question or a photo and "
            "I'll do my best to help!",
            reply_from,
        )
        return jsonify({"status": "received"}), 200

    logger.info("Answering message from %s: %s", _mask(sender), text[:200])
    try:
        answer = generate_answer(sender, text, voice_confidence, voice_language)
    except Exception:  # noqa: BLE001
        logger.exception("Error generating answer (model=%s)", GENERATION_MODEL)
        answer = (
            "Sorry, I ran into a problem answering that just now. "
            "Please try again in a moment."
        )

    send_whatsapp_message(sender, answer, reply_from)
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
