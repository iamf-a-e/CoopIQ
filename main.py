import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

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
SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").strip().strip('"').rstrip("/")
if SUPABASE_URL.endswith("/rest/v1"):  # tolerate a URL pasted with the API path
    SUPABASE_URL = SUPABASE_URL[: -len("/rest/v1")]
SUPABASE_KEY = (os.environ.get("SUPABASE_SERVICE_KEY") or "").strip().strip('"') or None

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

# If the farmer messaged us within this many hours, the bot does not greet or introduce itself.
GREETING_COOLDOWN_HOURS = float(os.environ.get("GREETING_COOLDOWN_HOURS", "24"))

# Photo check (same subjects and image types as the CoopIQ Supabase edge function).
DIAGNOSE_SUBJECTS = ["Poultry droppings", "Poultry bird", "Cattle", "Goat or sheep", "Pig"]
IMAGE_MIMES = ("image/jpeg", "image/png", "image/webp")
MAX_CAPTION_CHARS = 600

# Farmers' photos are saved to this private Supabase Storage bucket (see supabase_schema.sql).
IMAGE_BUCKET = os.environ.get("IMAGE_BUCKET", "farmer-images")
IMAGE_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}

# The person farmers call when they need a vet.
VET_NAME = "Memory"
VET_PHONE = "+260779722215"
VET_REFERENCE = f"{VET_NAME} from the CoopIQ team on {VET_PHONE}"

# The person farmers call to buy medicine.
MEDICINE_NAME = "Marcus"
MEDICINE_PHONE = "+260962810888"
MEDICINE_REFERENCE = f"{MEDICINE_NAME} from the CoopIQ medicine team on {MEDICINE_PHONE}"

# Menu options: (id sent back when a button is tapped, button title). WhatsApp allows at most
# 3 buttons per message and 20 characters per title. Each option's position is its number.
OPTION_NAMES = {
    "check": "Check my animal",
    "vet": "Get a vet",
    "ask": "Ask a question",
    "medicine": "Buy medicine",
}
MAIN_MENU = [("check", "1. Check my animal"), ("vet", "2. Get a vet"), ("medicine", "3. Buy medicine")]
AFTER_CHECK_MENU = [("vet", "1. Get a vet"), ("check", "2. Check another")]
AFTER_ANSWER_MENU = [("vet", "1. Get a vet"), ("ask", "2. Ask another")]
MENU_FOOTER = "Tap an option or reply with its number"
FOLLOW_UP_FOOTER = "Tap an option, or just type your question"  # every menu after the opening one
ASK_ANYTHING_LINE = "You can also ask me any farming question here."
MENU_PROMPT = "What would you like to do next?"
WELCOME_REPLY = "Hello, I'm CoopIQ. How can I help you today?"
GREETING_TAG = "[GREETING]"
URGENT_TAG = "[URGENT]"
INTERACTIVE_BODY_MAX_CHARS = 1024  # WhatsApp's limit for a message with buttons

# The static vet shown as "accepting" a dispatch request, for this demo.
DISPATCH_VET_NAME = "Dr Bwalya Mulenga"
DISPATCH_DELAY_SECONDS = float(os.environ.get("DISPATCH_DELAY_SECONDS", "4"))
DISPATCH_REQUESTING_REPLY = "Requesting a vet for you now..."
DISPATCH_ACCEPTED_REPLY = (
    f"Request accepted. {DISPATCH_VET_NAME} will call you shortly to arrange a farm visit "
    "or phone consultation."
)

# Intake: asked once, before the opening menu, on a farmer's first-ever text message.
INTAKE_NAME_QUESTION = "Hello, I'm CoopIQ. Before we start, what's your name?"
INTAKE_LOCATION_QUESTION = "Which area or town are you farming in?"
INTAKE_PHONE_QUESTION = "What's the best phone number for a vet to call you on?"
INTAKE_PHONE_RETRY = "That doesn't look like a phone number. Could you send it again?"
_PHONE_PATTERN = re.compile(r"^[+\d][\d\s().-]{6,19}$")

# A farmer can type this to wipe their own chat and start over, as a brand-new farmer would.
# Useful for testing and demos from a single number, without waiting for a new one.
CLEAR_KEYWORDS = {"clear", "reset"}
CLEAR_REPLY = "Okay, I've cleared our chat. Send \"hello\" to start again from the beginning."

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

if not (SUPABASE_URL and SUPABASE_KEY):
    logger.warning(
        "SUPABASE_URL=%s SUPABASE_SERVICE_KEY=%s: chats are NOT being saved to the database "
        "(in-memory fallback only, lost on every restart).",
        _set(SUPABASE_URL), _set(SUPABASE_KEY),
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

# Shared by the chat prompt and the photo-check prompt so both sound like the same person.
STYLE_RULE = """Write like a kind, experienced farm helper texting a neighbour on WhatsApp. \
Use everyday words and contractions. Plain text only: no bullets, no lists, no headers, no \
bold, and no emojis unless the farmer uses them. Never use em dashes or en dashes: use a \
comma, a full stop, or the word "and" instead, and write ranges as "2 to 3 weeks". Avoid \
stock phrases such as "I understand your concern", "I'm here to help", "Great question", \
or "Please note"."""

VET_CONTACT_RULE = f"""Getting a vet is automatic: the farmer taps "Get a vet" or you flag the \
case as urgent (see below), and the system requests one for them. Never write a phone number \
or a person's name for this. The only exception is if the farmer explicitly asks for a direct \
number to call themselves right now, for example "is there a number I can call": then, and \
only then, give {VET_NAME} from the CoopIQ team on {VET_PHONE} as a backup. \
If the farmer asks where to buy medicine or wants it delivered, tell them to call \
{MEDICINE_NAME} from the CoopIQ medicine team on {MEDICINE_PHONE}, who can deliver anywhere in \
Zambia. Never give any other number."""

SYSTEM_PROMPT = f"""You are CoopIQ, a friendly, practical helper for smallholder and \
commercial poultry farmers in Zambia. You help with broiler, layer, and village chicken \
farming: housing, brooding, feeding, vaccination and disease prevention, biosecurity, \
budgeting, and marketing. You can also answer basic health and feeding questions about \
cattle, goats, sheep, and pigs. You give quick guidance, and the vets the CoopIQ team arranges \
do the diagnosing and treating, so your job is to help the farmer and get them to a vet.

How to sound:
- {STYLE_RULE}
- Keep replies to 2 to 4 short lines, unless the farmer asks for more detail.
- When a farmer describes a problem, react like a person first, in a few words, for example \
"Oh no, that's worrying." Then help. Don't do this for simple questions.
- Use the conversation so far to understand what "it", "that", or "how often" refers to.

How to help:
- If you need more information, ask ONE question and wait for the answer, for example how \
many birds are affected, how old they are, how many have died, or when it started. Never ask \
several questions at once. Don't ask what the farmer has already told you.
- URGENT: if animals are dying in large numbers or showing severe signs (bloody droppings, \
gasping, sudden mass deaths), do not ask questions first. Start your reply with {URGENT_TAG} \
and nothing before it, then one short line saying it's urgent and one or two things to do \
right now, such as keeping sick birds apart from the others. Never give a phone number here \
(see below), the system requests a vet automatically. Never combine {URGENT_TAG} with {GREETING_TAG}.
- Never state a diagnosis as fact. Say what it could be and that a vet needs to confirm it.
- If you are unsure or the animals are sick, say clearly that a vet should look. Don't add a \
vet mention to routine answers. {VET_CONTACT_RULE}
- Each message may include REFERENCE NOTES from the CoopIQ knowledge base. Prefer them when \
they are relevant and keep any numbers in them exact. If the notes do not cover the question, \
answer from your own general farming knowledge, and say so briefly when you are less sure.
- For vaccine schedules, medicine names, dosages, withdrawal periods, and costs, only give \
specific figures if they appear in the reference notes. Otherwise give general guidance and \
say a vet can confirm the details. Never invent facts.
- If a question has nothing to do with farming, say kindly in the farmer's language that \
you only help with poultry and livestock.

Language:
- {LANGUAGE_RULE}
- Some messages are voice notes that were transcribed automatically, so the text may \
contain mishearings, especially of local words. Use the context to work out the most \
likely meaning. A voice note may be marked with the language heard in the audio: treat \
that as a hint, and trust the transcript text if they disagree. When a message is marked \
as a voice note with medium or low confidence, start by saying what you understood, in the \
farmer's language, so they can correct you, for example "I think you're asking about..."
- Earlier turns that start with [Photo of ...] were photo checks you already answered. \
Use them to understand follow-up questions about that photo.

Greetings:
- Only introduce yourself as CoopIQ when the message tells you this is FIRST CONTACT.
- If the farmer's message is only a greeting, thanks, or small talk, reply warmly in one \
or two short sentences WRITTEN IN THE FARMER'S LANGUAGE (follow the language rule above, \
not English by default), and ask how you can help today. For these messages ignore the \
reference notes. Examples of greetings and the language they signal:
    "Muli shani" -> Bemba: reply in Bemba.
    "Muli bwanji" or "Zikomo" -> Nyanja: reply in Nyanja.
    "Mwabuka buti" or "Mwalibizya buti" -> Tonga: reply in Tonga.
    "Lumela" or "Muzuhile" -> Lozi: reply in Lozi.
    "Hello" or "Good morning" -> English: reply in English.

These English examples show the tone. Don't copy their wording, and write in the farmer's \
language.
Farmer: my chicks are dying, 5 this morning
You: Oh no, I'm sorry, that's hard. How old are the chicks?
Farmer: how much feed does a broiler eat?
You: (give the figure from the notes in one or two lines)
Farmer: many birds died overnight, there's blood in the droppings
You: {URGENT_TAG} This is urgent. Keep the sick birds away from the others right now.

Some older messages in this chat may be long or use bullet points. Don't copy that style."""


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
doses; say "confirm the dose with a vet" when no dose is in the reference.
isolate: true if affected animals should be separated from the rest.
escalate: true if a vet should be contacted (always true for Newcastle disease, bloody \
droppings, sudden deaths, or Low confidence, unless subject is Other).
language: the language you wrote reply in.
reply: the complete WhatsApp message to send to the farmer, written in the farmer's \
language, as 2 to 4 short lines. It must sound like a person who has just looked at the \
photo, not like a report. Do not use labels such as "Condition:" or "Confidence:", and do \
not give a confidence rating. In your own words, say what you can see and what it could be, \
then say a vet needs to confirm it. If the photo points to something serious, such as bloody \
droppings or signs of Newcastle disease, make your first line say it is urgent. Never give a \
phone number, the system requests a vet automatically when escalate is true. Give the next \
step. If isolate is true, tell them to keep the affected \
animals away from the others. If escalate is true, say clearly that a vet should look. If \
confidence is Low, say you can't tell from this photo and ask for one clearer, well-lit \
photo. Do not add a disclaimer about photos or laboratory results. If subject is Other, set condition to "Unclear", confidence to "Low", isolate and \
escalate to false, and make reply kindly ask for a clear, well-lit photo of the sick bird, \
its droppings, or the animal.

{STYLE_RULE}

{VET_CONTACT_RULE}

Keep subject, condition and confidence in English exactly as described above.

{LANGUAGE_RULE}
For a photo there is no message text to read the language from. Use the language of the \
farmer's caption if there is one. If there is no caption, use the language of the farmer's \
recent messages. If neither exists, use English."""

UNSUPPORTED_IMAGE_REPLY = (
    "I can't open that type of file. Could you take a photo of the bird, droppings or "
    "animal and send it as a normal picture?"
)
DIAGNOSE_FAILED_REPLY = (
    "Sorry, I couldn't check that photo just now. Please send it again in a minute. "
    f"If your animals are really sick, don't wait, please contact {VET_REFERENCE}."
)
VOICE_FAILED_REPLY = (
    "Sorry, I couldn't hear that voice note properly. Could you record it again somewhere "
    "quieter, or type it out for me?"
)
UNSUPPORTED_MESSAGE_REPLY = (
    "I can't read that kind of message, but you can type to me, send a voice note, or send "
    "a photo of a sick bird, droppings or animal. What's going on with your animals?"
)
ANSWER_FAILED_REPLY = (
    "Sorry, something went wrong on my side. Please send that again in a minute. "
    f"If your animals are really sick, don't wait, please contact {VET_REFERENCE}."
)

# What the bot sends when a farmer picks a menu option. "vet" is handled separately (it
# dispatches a vet instead of sending a fixed reply) and has no entry here.
OPTION_REPLIES = {
    "check": (
        "No problem. Send me a photo of the animal or its droppings, or tell me what's wrong "
        "by text or voice note."
    ),
    "ask": (
        "Sure. You can ask me about feeding, vaccination, housing, keeping your animals healthy, "
        "costs or selling your birds. What would you like to know?"
    ),
    "medicine": (
        f"Please call {MEDICINE_REFERENCE}.\n\n"
        f"Tell {MEDICINE_NAME} what medicine you need and where you are. {MEDICINE_NAME} will "
        "explain the cost and arrange delivery to you anywhere in Zambia, by courier if needed.\n\n"
        f"{ASK_ANYTHING_LINE}"
    ),
}


def clean_reply(text: str) -> str:
    """Remove em and en dashes from a message, keeping number ranges and line breaks readable."""
    if not text:
        return text
    dashes = "‒–—―"
    # "2–3 weeks" becomes "2 to 3 weeks".
    text = re.sub(rf"(?<=\d)[ \t]*[{dashes}][ \t]*(?=\d)", " to ", text)
    # A dash at the start or end of a line is dropped.
    text = re.sub(rf"^[ \t]*[{dashes}]+[ \t]*", "", text, flags=re.MULTILINE)
    text = re.sub(rf"[ \t]*[{dashes}]+[ \t]*$", "", text, flags=re.MULTILINE)
    # An unspaced en dash joins words ("Jan–Mar"), so it becomes a hyphen.
    text = re.sub(r"(?<=\w)[‒–](?=\w)", "-", text)
    # Any other dash separates parts of a sentence, so it becomes a comma.
    text = re.sub(rf"[ \t]*[{dashes}]+[ \t]*", ", ", text)
    # Tidy up if the dash sat next to other punctuation ("well —, ok" or "ok, — fine").
    text = re.sub(r",[ \t]*([,.!?;:])", r"\1", text)
    return text


# Saved replies end with a marker listing the options they showed, in order. That is how a
# typed "2" can be matched to the right option on the next message without a database change.
_OPTIONS_MARKER = re.compile(r"\s*\[Options: ([a-z,]+)\]\s*$")


def with_options_marker(reply: str, menu: list) -> str:
    """Return the reply as it should be saved, recording which options it showed."""
    if not menu:
        return reply
    return f"{reply}\n\n[Options: {','.join(option_id for option_id, _ in menu)}]"


def strip_options_marker(reply: str) -> str:
    return _OPTIONS_MARKER.sub("", reply)


def options_in(reply: str) -> list:
    """Return the option ids a saved reply showed, in order, or [] if it showed none."""
    match = _OPTIONS_MARKER.search(reply or "")
    return match.group(1).split(",") if match else []


# ---------------------------------------------------------------------------
# Storage (Supabase Postgres via the REST API, with in-process fallback)
#
# Tables (see supabase_schema.sql):
#   chat_messages      one row per message, kept permanently
#   processed_messages WhatsApp message IDs already handled (dedupe)
#   farmers            one row per farmer: first_seen, last_seen, last_language
#   farmer_images      one row per photo received (file lives in Storage bucket IMAGE_BUCKET)
# ---------------------------------------------------------------------------
_local_history = {}
_local_seen = set()
_local_farmers = {}


def _supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _sb_headers() -> dict:
    """Auth headers shared by the REST (PostgREST) and Storage calls."""
    headers = {"apikey": SUPABASE_KEY}
    # Legacy service_role keys are JWTs (start with "eyJ") and go in Authorization too.
    # New sb_secret_... keys are not JWTs and only need the apikey header.
    if SUPABASE_KEY.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {SUPABASE_KEY}"
    return headers


def _sb(method: str, table: str, params: dict = None, body=None, prefer: str = None):
    """Call the Supabase REST API (PostgREST) and return the parsed JSON, or None if empty."""
    headers = {**_sb_headers(), "Content-Type": "application/json"}
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
            history = _start_with_user([
                {"role": r["role"], "parts": [_for_model(r["role"], r["content"])]}
                for r in rows
            ])
            logger.info("History loaded from Supabase: %d messages for %s", len(history), _mask(sender))
            return history
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load history from Supabase")
            return []
    history = _start_with_user([
        {"role": h["role"], "parts": [_for_model(h["role"], h["parts"][0])]}
        for h in _local_history.get(sender, [])[-CONTEXT_MESSAGES:]
    ])
    logger.info("History loaded from memory: %d messages for %s", len(history), _mask(sender))
    return history


def _for_model(role: str, content: str) -> str:
    """Hide the options marker from the model, and any dashes in older saved replies, so it
    doesn't imitate either."""
    return clean_reply(strip_options_marker(content)) if role == "model" else content


def load_last_reply(sender: str) -> str:
    """Return the bot's last saved reply including its options marker, or "" if there is none."""
    if _supabase_enabled():
        try:
            rows = _sb(
                "GET",
                "chat_messages",
                params={
                    "select": "content",
                    "sender": f"eq.{sender}",
                    "role": "eq.model",
                    "order": "id.desc",
                    "limit": "1",
                },
            ) or []
            return rows[0]["content"] if rows else ""
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load last reply from Supabase")
            return ""
    replies = [h for h in _local_history.get(sender, []) if h["role"] == "model"]
    return replies[-1]["parts"][0] if replies else ""


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


# Columns used for the first-contact intake (name, location, contact_phone, intake_step).
# If Supabase is enabled, these must exist on the farmers table, for example:
#   alter table farmers add column if not exists name text;
#   alter table farmers add column if not exists location text;
#   alter table farmers add column if not exists contact_phone text;
#   alter table farmers add column if not exists intake_step text;
# Until they're added, these calls fail silently (like the rest of this file's Supabase calls)
# and intake falls back to asking again each time, rather than crashing the bot.
_PROFILE_FIELDS = ("name", "location", "contact_phone", "intake_step")


def get_farmer_profile(sender: str) -> dict:
    """Return {"name", "location", "contact_phone", "intake_step"}, each None if unknown."""
    empty = {field: None for field in _PROFILE_FIELDS}
    if _supabase_enabled():
        try:
            rows = _sb(
                "GET",
                "farmers",
                params={"select": ",".join(_PROFILE_FIELDS), "phone": f"eq.{sender}", "limit": "1"},
            ) or []
            return {**empty, **rows[0]} if rows else empty
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load farmer profile (columns may be missing)")
            return empty
    return {**empty, **{k: v for k, v in _local_farmers.get(sender, {}).items() if k in _PROFILE_FIELDS}}


def save_farmer_profile(sender: str, **fields) -> None:
    """Merge the given profile fields (name, location, contact_phone, intake_step) in."""
    if _supabase_enabled():
        try:
            _sb(
                "POST",
                "farmers",
                params={"on_conflict": "phone"},
                body={"phone": sender, **fields},
                prefer="resolution=merge-duplicates,return=minimal",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to save farmer profile (columns may be missing)")
        return
    _local_farmers.setdefault(sender, {}).update(fields)


def reset_farmer(sender: str) -> None:
    """Wipe a farmer's saved profile, intake progress and chat history, so their next message
    is treated as first contact again. Used for the "clear" keyword."""
    if _supabase_enabled():
        try:
            _sb("DELETE", "farmers", params={"phone": f"eq.{sender}"}, prefer="return=minimal")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to delete farmer row during reset")
        try:
            _sb("DELETE", "chat_messages", params={"sender": f"eq.{sender}"}, prefer="return=minimal")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to delete chat history during reset")
        return
    _local_farmers.pop(sender, None)
    _local_history.pop(sender, None)


def advance_intake(sender: str, step: str, text: str) -> tuple:
    """Advance the one-time, first-contact intake by one step.

    step is None to start it. Returns (reply, menu); menu is MAIN_MENU only once intake is
    done, so the caller can send it straight to the farmer and return, same as any other reply.
    """
    text = (text or "").strip()
    if step is None:
        save_farmer_profile(sender, intake_step="name")
        return INTAKE_NAME_QUESTION, None
    if step == "name":
        save_farmer_profile(sender, name=text[:80] or "there", intake_step="location")
        return INTAKE_LOCATION_QUESTION, None
    if step == "location":
        save_farmer_profile(sender, location=text[:120], intake_step="phone")
        return INTAKE_PHONE_QUESTION, None
    # step == "phone"
    if not _PHONE_PATTERN.match(text):
        return INTAKE_PHONE_RETRY, None
    save_farmer_profile(sender, contact_phone=text[:30], intake_step="done")
    return WELCOME_REPLY, MAIN_MENU


def _parse_ts(value):
    """Parse a Postgres/ISO timestamp into an aware datetime, or None if it can't be read."""
    if not value:
        return None
    try:
        s = str(value).strip().replace("Z", "+00:00")
        # Older Pythons need exactly 6 fractional digits; Postgres may return fewer.
        s = re.sub(r"\.(\d+)", lambda m: "." + m.group(1)[:6].ljust(6, "0"), s, count=1)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def get_last_seen(sender: str):
    """When the farmer last messaged us (before this message), or None if never."""
    if _supabase_enabled():
        try:
            rows = _sb(
                "GET",
                "farmers",
                params={"select": "last_seen", "phone": f"eq.{sender}", "limit": "1"},
            ) or []
            return _parse_ts(rows[0].get("last_seen")) if rows else None
        except Exception:  # noqa: BLE001
            logger.exception("Failed to read last_seen")
            return None
    return _parse_ts(_local_farmers.get(sender, {}).get("last_seen"))


def chatted_recently(last_seen) -> bool:
    """True if the farmer messaged us within GREETING_COOLDOWN_HOURS."""
    if not last_seen:
        return False
    return datetime.now(timezone.utc) - last_seen < timedelta(hours=GREETING_COOLDOWN_HOURS)


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


def save_image(sender: str, media_id: str, data: bytes, mime: str, caption: str):
    """Upload the photo to Supabase Storage and insert a farmer_images row.

    Returns the row id, or None if storage is off or anything fails. It never raises,
    so a storage problem can't stop the photo check from running.
    """
    if not _supabase_enabled():
        return None
    path = None
    try:
        path = f"{sender}/{datetime.now(timezone.utc):%Y-%m}/{media_id}.{IMAGE_EXT[mime]}"
        headers = {**_sb_headers(), "Content-Type": mime, "x-upsert": "true"}  # upsert: safe on retries
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{IMAGE_BUCKET}/{path}",
            headers=headers,
            data=data,
            timeout=20,
        )
        if resp.status_code >= 300:
            logger.error("Image upload failed: http=%s body=%s", resp.status_code, resp.text[:300])
            path = None
    except Exception:  # noqa: BLE001
        logger.exception("Image upload raised an error")
        path = None
    try:
        rows = _sb(
            "POST",
            "farmer_images",
            body={
                "sender": sender,
                "storage_path": path,
                "mime_type": mime,
                "size_bytes": len(data),
                "caption": caption or None,
                "whatsapp_media_id": media_id,
            },
            prefer="return=representation",
        )
        row_id = rows[0]["id"] if rows else None
        logger.info("Image saved for %s (row=%s, path=%s)", _mask(sender), row_id, path)
        return row_id
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save farmer_images row")
        return None


def update_image_result(row_id, **fields) -> None:
    """Attach the photo-check result (subject, condition, ...) to the saved image row."""
    if not (row_id and _supabase_enabled()):
        return
    try:
        _sb("PATCH", "farmer_images", params={"id": f"eq.{row_id}"}, body=fields, prefer="return=minimal")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update farmer_images row %s", row_id)


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
    recently_chatted: bool = False,
    first_contact: bool = False,
) -> tuple:
    """Let Gemini handle the whole conversation, with retrieved notes as optional context.

    Returns (reply, menu, urgent). menu is the list of options to show under the reply, or
    None. urgent is True when the case needs a vet dispatched right away.
    """
    started = time.time()
    history = load_history(sender)
    option_names = set(OPTION_NAMES.values())

    # Include the previous farmer message so short follow-ups still retrieve well.
    # Tapped options say nothing about the topic, so they are left out.
    previous_user = [
        h["parts"][0] for h in history if h["role"] == "user" and h["parts"][0] not in option_names
    ][-1:]
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
    if first_contact:
        language_note += (
            "\n\nFIRST CONTACT: This is the farmer's first ever message. Greet them warmly, say "
            "once and briefly, in their language, that you are CoopIQ and can arrange a vet "
            "for them, then deal with their message."
        )
    elif recently_chatted:
        # Overrides the greeting rules in the system prompt for this turn.
        language_note += (
            "\n\nGREETING: This farmer already chatted with you in the last 24 hours. Do NOT "
            "greet them and do NOT introduce yourself: no \"Hello\", \"Hi\", \"Good morning\", "
            "no greeting in any local language, and no \"I am CoopIQ\". Start directly with the "
            "answer. If their message is only a greeting or small talk, reply in one short "
            "sentence in their language and invite their next question, without introducing "
            "yourself again."
        )
    else:
        language_note += (
            "\n\nGREETING: You already know this farmer. A short greeting is fine, but do not "
            "introduce yourself or say \"I am CoopIQ\"."
        )
    if not recently_chatted:
        language_note += (
            f"\n\nMENU: If the farmer's message is only a greeting or small talk, begin your "
            f"reply with {GREETING_TAG} and then ask how you can help today, in their language. "
            f"If it describes a problem or asks a question, do not use {GREETING_TAG}, just answer."
        )
    # Repeated here because older replies in the history may be long or use bullet points.
    language_note += (
        "\n\nSTYLE: 2 to 4 short lines of plain text, no bullets, no em or en dashes, and at "
        "most one question.\n\n"
    )

    turn = f"{notes}\n\n{channel}{language_note}FARMER'S MESSAGE:\n{user_text}"

    logger.info("Calling Gemini model=%s (prompt ~%d chars)", GENERATION_MODEL, len(turn))
    gemini_started = time.time()
    model = genai.GenerativeModel(GENERATION_MODEL, system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat(history=history)
    response = chat.send_message(
        turn,
        generation_config={"temperature": 0.4, "max_output_tokens": 700},
    )
    answer = clean_reply(response.text.strip())
    logger.info(
        "Gemini replied in %.1fs (%d chars); total generate_answer %.1fs",
        time.time() - gemini_started,
        len(answer),
        time.time() - started,
    )

    urgent = answer.startswith(URGENT_TAG)
    answer = answer.replace(URGENT_TAG, "").strip()
    is_greeting = answer.startswith(GREETING_TAG)
    answer = answer.replace(GREETING_TAG, "").strip()
    if urgent:
        menu = None  # a vet is being dispatched instead of offering options
    elif is_greeting and not recently_chatted:
        answer = answer or WELCOME_REPLY
        menu = MAIN_MENU
    elif answer.endswith("?"):
        menu = None  # the farmer needs to answer the question, so don't offer options
    else:
        menu = _menu_after_answer(history)

    # Store the plain message, not the injected notes, so history stays small.
    append_history(sender, user_text, with_options_marker(answer, menu))

    return answer, menu, urgent


def _menu_after_answer(history: list) -> list:
    """Offer "Check another" if the farmer was checking an animal, otherwise "Ask another"."""
    for turn in reversed(history):
        if turn["role"] != "user":
            continue
        content = turn["parts"][0]
        if content == OPTION_NAMES["check"] or content.startswith("[Photo of"):
            return AFTER_CHECK_MENU
        if content == OPTION_NAMES["ask"]:
            return AFTER_ANSWER_MENU
    return AFTER_ANSWER_MENU


# ---------------------------------------------------------------------------
# Photo diagnosis (mirrors the "diagnose" action of the Supabase edge function)
# ---------------------------------------------------------------------------
def _fallback_diagnosis_reply(condition: str, summary: str, recommendation: str,
                              isolate: bool, escalate: bool) -> str:
    """English message built from the structured fields, used only if the model gave no reply."""
    if condition.lower().startswith("unclear"):
        lines = ["I can't tell for sure from this photo."]
    elif condition.lower().startswith("healthy"):
        lines = ["I can't see clear signs of illness in this photo."]
    else:
        lines = [f"This could be {condition}, but a vet needs to confirm it."]
    if summary:
        lines.append(summary)
    if recommendation:
        lines.append(recommendation)
    if isolate:
        lines.append("Keep the affected animals away from the others for now.")
    if escalate:
        lines.append(f"Please contact {VET_REFERENCE} as soon as you can.")
    return "\n\n".join(lines)


def diagnose_image(sender: str, image: dict) -> tuple:
    """Check a farmer's photo. Returns (reply, menu, escalate); menu may be None."""
    started = time.time()
    if not image.get("id"):
        raise ValueError("image message has no media id")

    declared = (image.get("mime_type") or "").split(";")[0].strip().lower()
    if declared and declared not in IMAGE_MIMES:
        logger.info("Unsupported image type: %s", declared)
        return UNSUPPORTED_IMAGE_REPLY, None, False

    data, media_mime = download_whatsapp_media(image["id"])
    mime = declared or (media_mime or "").split(";")[0].strip().lower() or "image/jpeg"
    if mime not in IMAGE_MIMES:
        logger.info("Unsupported image type after download: %s", mime)
        return UNSUPPORTED_IMAGE_REPLY, None, False

    caption = (image.get("caption") or "").strip()[:MAX_CAPTION_CHARS]

    # Save the photo first, so it is kept even if the diagnosis below fails.
    image_row_id = save_image(sender, image["id"], data, mime, caption)

    history = load_history(sender)
    # Tapped options are always in English, so they would mislead the language choice below.
    option_names = set(OPTION_NAMES.values())
    recent_farmer = [
        h["parts"][0] for h in history
        if h.get("role") == "user"
        and not h["parts"][0].startswith("[Photo of")
        and h["parts"][0] not in option_names
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

    # Attach the result to the saved photo row.
    update_image_result(
        image_row_id,
        subject=subject,
        condition=condition,
        confidence=confidence,
        isolate=isolate,
        escalate=escalate,
    )

    logger.info(
        "Photo check: subject=%s condition=%s confidence=%s isolate=%s escalate=%s language=%s (%.1fs)",
        subject, condition, confidence, isolate, escalate, result.get("language"),
        time.time() - started,
    )

    reply = str(result.get("reply") or "").strip()
    if not reply:
        reply = _fallback_diagnosis_reply(condition, summary, recommendation, isolate, escalate)
    reply = clean_reply(reply)

    # An escalated case gets a vet dispatched automatically (the caller does this), so no
    # button is needed. Otherwise, a reply about a known animal still offers the vet, unless
    # it's asking for a better photo. A photo that isn't of an animal offers nothing.
    show_menu = not escalate and subject != "Other" and not reply.endswith("?")
    menu = AFTER_CHECK_MENU if show_menu else None

    # Keep a plain-text record so follow-up questions about the photo make sense.
    note = f"[Photo of {subject}]"
    if caption:
        note = f"{note} {caption}"
    append_history(sender, note, with_options_marker(reply, menu))
    return reply, menu, escalate


# ---------------------------------------------------------------------------
# WhatsApp Cloud API helpers
# ---------------------------------------------------------------------------
def send_whatsapp_message(to: str, body: str, phone_number_id: str = None, menu: list = None) -> None:
    """Send a text message, with reply buttons if a menu is given.

    Replies go out from `phone_number_id` (the number that RECEIVED the message) when
    given, falling back to the PHONE_ID env var. This guarantees the reply appears in
    the same chat the farmer wrote in, even if PHONE_ID is misconfigured.

    If WhatsApp rejects the menu, the options are sent again as numbered text.
    """
    sender_id = phone_number_id or PHONE_ID
    body = clean_reply(body)  # every outgoing message passes through here
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
    if menu and len(body) > INTERACTIVE_BODY_MAX_CHARS:
        # Too long to carry buttons: send the text first, then the buttons on their own.
        send_whatsapp_message(to, body, sender_id)
        body = MENU_PROMPT

    payload = {"messaging_product": "whatsapp", "to": to}
    if menu:
        payload["type"] = "interactive"
        payload["interactive"] = {
            "type": "button",
            "body": {"text": body},
            "footer": {"text": MENU_FOOTER if menu is MAIN_MENU else FOLLOW_UP_FOOTER},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": option_id, "title": title}}
                    for option_id, title in menu
                ]
            },
        }
    else:
        payload["type"] = "text"
        payload["text"] = {"body": body[:WHATSAPP_MAX_CHARS]}

    def send_as_numbered_text():
        options = "\n".join(title for _, title in menu)
        send_whatsapp_message(to, f"{body}\n\n{options}\n\nReply with a number.", sender_id)

    logger.info(
        "Sending WhatsApp message to %s from phone_number_id=%s (%d chars)",
        _mask(to), sender_id, len(body),
    )
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
    except requests.RequestException:
        logger.exception("WhatsApp send raised a network error")
        if menu:
            send_as_numbered_text()
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
        if menu:
            send_as_numbered_text()
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


def extract_button_id(payload: dict):
    """Return the id of a tapped option, or None if this message isn't a tap.

    Menus are now buttons, but list_reply keeps the short-lived list menu in existing chats working.
    """
    try:
        interactive = payload["entry"][0]["changes"][0]["value"]["messages"][0]["interactive"]
        return (interactive.get("button_reply") or interactive.get("list_reply"))["id"]
    except (KeyError, IndexError, TypeError):
        return None


def menu_choice(payload: dict, sender: str, text: str):
    """Return the option the farmer picked, by button tap or typed number, or None.

    A typed number only counts if the bot's last message showed options, because "2" can
    also be an answer, such as how many birds died.
    """
    button_id = extract_button_id(payload)
    if button_id in OPTION_NAMES:
        return button_id
    match = re.fullmatch(r"\s*(\d)\s*[.)]?\s*", text or "")
    if not match:
        return None
    options = options_in(load_last_reply(sender))
    position = int(match.group(1)) - 1
    return options[position] if 0 <= position < len(options) else None


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


@app.route("/storage-check", methods=["GET"])
def storage_check():
    """Diagnose Supabase: open /storage-check?token=<VERIFY_TOKEN> in a browser."""
    if not VERIFY_TOKEN or request.args.get("token") != VERIFY_TOKEN:
        return "Forbidden", 403
    if not _supabase_enabled():
        return jsonify({
            "supabase": "NOT CONFIGURED (in-memory fallback in use)",
            "SUPABASE_URL": _set(SUPABASE_URL),
            "SUPABASE_SERVICE_KEY": _set(SUPABASE_KEY),
            "hint": "Add both variables in Vercel and redeploy.",
        })

    def describe(exc):
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            return f"FAILED http {exc.response.status_code}: {exc.response.text[:300]}"
        return f"FAILED {type(exc).__name__}: {exc}"

    results = {"supabase_host": SUPABASE_URL.replace("https://", "")}
    for table in ("chat_messages", "processed_messages", "farmers", "farmer_images"):
        try:
            _sb("GET", table, params={"select": "*", "limit": "1"})
            results[f"read {table}"] = "ok"
        except Exception as exc:  # noqa: BLE001
            results[f"read {table}"] = describe(exc)
    try:
        _sb("POST", "farmers", params={"on_conflict": "phone"},
            body={"phone": "_storage_check"}, prefer="resolution=merge-duplicates,return=minimal")
        _sb("DELETE", "farmers", params={"phone": "eq._storage_check"}, prefer="return=minimal")
        results["write farmers"] = "ok"
    except Exception as exc:  # noqa: BLE001
        results["write farmers"] = describe(exc)
    # Check the image bucket exists and the key can list it.
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/list/{IMAGE_BUCKET}",
            headers={**_sb_headers(), "Content-Type": "application/json"},
            json={"prefix": "", "limit": 1},
            timeout=8,
        )
        results[f"storage bucket {IMAGE_BUCKET}"] = (
            "ok" if resp.status_code < 300 else f"FAILED http {resp.status_code}: {resp.text[:300]}"
        )
    except Exception as exc:  # noqa: BLE001
        results[f"storage bucket {IMAGE_BUCKET}"] = describe(exc)
    return jsonify(results)


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


def dispatch_vet(sender: str, reply_from: str, user_note: str) -> None:
    """Send the "requesting a vet... request accepted" sequence and record it in history.

    Blocks for DISPATCH_DELAY_SECONDS between the two messages. That's fine for a single
    webhook request, but means this request takes a few seconds longer to finish.
    """
    send_whatsapp_message(sender, DISPATCH_REQUESTING_REPLY, reply_from)
    time.sleep(DISPATCH_DELAY_SECONDS)
    send_whatsapp_message(sender, DISPATCH_ACCEPTED_REPLY, reply_from)
    append_history(sender, user_note, f"{DISPATCH_REQUESTING_REPLY}\n\n{DISPATCH_ACCEPTED_REPLY}")


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

    if text and text.strip().lower() in CLEAR_KEYWORDS:
        logger.info("Clear requested by %s", _mask(sender))
        reset_farmer(sender)
        send_whatsapp_message(sender, CLEAR_REPLY, reply_from)
        return jsonify({"status": "reset"}), 200

    # Read the previous last_seen BEFORE touch_farmer overwrites it with now.
    last_seen = get_last_seen(sender)
    first_contact = last_seen is None
    recently_chatted = chatted_recently(last_seen)
    touch_farmer(sender)

    # Intake (name, location, contact phone) runs once, on a brand-new farmer's first-ever
    # TEXT message, before anything else. If their first message is a photo or voice note
    # instead, intake is skipped for now and simply never starts (a known limitation).
    is_text_message = text is not None and audio is None and image is None
    if is_text_message:
        profile = get_farmer_profile(sender)
        step = profile.get("intake_step")
        if step in ("name", "location", "phone") or (step is None and first_contact):
            reply, menu = advance_intake(sender, step, text)
            send_whatsapp_message(sender, reply, reply_from, menu)
            if menu:
                # Only the finishing step returns a menu; record it so a typed number for
                # the opening menu is recognised on the farmer's very next message.
                append_history(sender, text, with_options_marker(reply, menu))
            return jsonify({"status": "received"}), 200

    # Photos: save them, run the disease check and reply.
    if image:
        logger.info("Photo received (caption=%s), running photo check", "yes" if image.get("caption") else "no")
        try:
            reply, menu, escalate = diagnose_image(sender, image)
        except Exception:  # noqa: BLE001
            logger.exception("Photo check failed (model=%s)", GENERATION_MODEL)
            reply, menu, escalate = DIAGNOSE_FAILED_REPLY, None, False
        send_whatsapp_message(sender, reply, reply_from, menu)
        if escalate:
            dispatch_vet(sender, reply_from, "[Vet dispatch after photo check]")
        return jsonify({"status": "received"}), 200

    voice_confidence = None
    voice_language = None
    if audio:
        logger.info("Voice note received, transcribing")
        text, voice_confidence, voice_language = transcribe_voice_note(audio)
        if not text:
            logger.warning("Voice note produced no transcript")
            send_whatsapp_message(sender, VOICE_FAILED_REPLY, reply_from)
            return jsonify({"status": "received"}), 200

    if not text:
        logger.info("Unsupported message type; sending help text")
        send_whatsapp_message(sender, UNSUPPORTED_MESSAGE_REPLY, reply_from)
        return jsonify({"status": "received"}), 200

    # Menu options get a fixed reply, except "vet", which dispatches one. A voice note is
    # never treated as a menu choice.
    choice = None if audio else menu_choice(payload, sender, text)
    if choice == "vet":
        logger.info("Menu option chosen: vet (dispatching)")
        dispatch_vet(sender, reply_from, OPTION_NAMES["vet"])
        return jsonify({"status": "received"}), 200
    if choice:
        logger.info("Menu option chosen: %s", choice)
        reply = OPTION_REPLIES[choice]
        append_history(sender, OPTION_NAMES[choice], reply)
        send_whatsapp_message(sender, reply, reply_from)
        return jsonify({"status": "received"}), 200

    logger.info("Answering message from %s: %s", _mask(sender), text[:200])
    try:
        answer, menu, urgent = generate_answer(
            sender, text, voice_confidence, voice_language,
            recently_chatted=recently_chatted, first_contact=first_contact,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Error generating answer (model=%s)", GENERATION_MODEL)
        answer, menu, urgent = ANSWER_FAILED_REPLY, None, False

    send_whatsapp_message(sender, answer, reply_from, menu)
    if urgent:
        dispatch_vet(sender, reply_from, "[Vet dispatch after urgent message]")
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
