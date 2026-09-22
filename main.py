"""
main.py

CoopIQ is a Flask WhatsApp chatbot that answers poultry-farming questions
(broiler, layer, village chicken, housing, biosecurity, vaccination,
feeding, budgeting, marketing, etc.) using retrieval-augmented generation
(RAG) over the CoopIQ knowledge base (poultry_data.py / embeddings.json)
and Google's Gemini models.

Structure mirrors the Rudo-Test project:
    - GET  /webhook  -> WhatsApp Cloud API webhook verification
    - POST /webhook  -> incoming WhatsApp messages
    - GET  /         -> health/status page
"""

import logging
import os

import google.generativeai as genai
import requests
from flask import Flask, request, jsonify

from retrieval import retrieve, format_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coopiq")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration (environment variables)
# ---------------------------------------------------------------------------
WA_TOKEN = os.environ.get("WA_TOKEN")
PHONE_ID = os.environ.get("PHONE_ID")
GEN_API = os.environ.get("GEN_API")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "COOPIQ")

WHATSAPP_API_URL = f"https://graph.facebook.com/v20.0/{PHONE_ID}/messages" if PHONE_ID else None
GENERATION_MODEL = "models/gemini-1.5-flash"

if GEN_API:
    genai.configure(api_key=GEN_API)

SYSTEM_PROMPT = """You are CoopIQ, a friendly, practical WhatsApp assistant that helps \
smallholder and commercial poultry farmers in Zambia with broiler, layer, and village \
chicken farming: housing, brooding, feeding, vaccination and disease prevention, \
biosecurity, budgeting, and marketing.

Answer ONLY using the CONTEXT provided below. If the context does not contain the \
answer, say you don't have that information yet and suggest the farmer ask a follow-up \
question or consult a local veterinary/extension officer -- do not make facts up.

Keep answers short and practical for a WhatsApp chat: plain language, short paragraphs \
or a few bullet points, no markdown headers. Where the context includes numbers \
(temperatures, dosages, costs, timelines), keep them exact."""


# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------
def generate_answer(user_question: str) -> str:
    """Retrieve relevant passages and ask Gemini to answer using only that context."""
    passages = retrieve(user_question, top_k=4)

    if not passages:
        return (
            "I don't have information on that yet in the CoopIQ knowledge base. "
            "Could you rephrase, or ask me about broiler/layer/village chicken farming, "
            "housing, feeding, vaccination, biosecurity, budgeting, or marketing?"
        )

    context = format_context(passages)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"FARMER'S QUESTION:\n{user_question}\n\n"
        f"ANSWER:"
    )

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()


# ---------------------------------------------------------------------------
# WhatsApp Cloud API helpers
# ---------------------------------------------------------------------------
def send_whatsapp_message(to: str, body: str) -> None:
    if not (WA_TOKEN and WHATSAPP_API_URL):
        logger.warning("WA_TOKEN or PHONE_ID not configured; skipping send. Reply was: %s", body)
        return

    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    resp = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=20)
    if resp.status_code >= 300:
        logger.error("WhatsApp send failed (%s): %s", resp.status_code, resp.text)


def extract_incoming_message(payload: dict):
    """Pull the sender's number and message text out of a WhatsApp webhook payload."""
    try:
        entry = payload["entry"][0]
        change = entry["changes"][0]
        value = change["value"]
        messages = value.get("messages")
        if not messages:
            return None, None
        message = messages[0]
        sender = message["from"]
        if message.get("type") != "text":
            return sender, None
        text = message["text"]["body"]
        return sender, text
    except (KeyError, IndexError, TypeError):
        return None, None


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

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def handle_webhook():
    payload = request.get_json(silent=True) or {}
    logger.info("Incoming webhook payload: %s", payload)

    sender, text = extract_incoming_message(payload)

    if not sender:
        # Not a user message (e.g. a status update) -- acknowledge and ignore.
        return jsonify({"status": "ignored"}), 200

    if not text:
        send_whatsapp_message(
            sender,
            "I can currently only read text messages. Please type your poultry-farming "
            "question and I'll do my best to help!",
        )
        return jsonify({"status": "received"}), 200

    try:
        answer = generate_answer(text)
    except Exception:  # noqa: BLE001
        logger.exception("Error generating answer")
        answer = (
            "Sorry, I ran into a problem answering that just now. "
            "Please try again in a moment."
        )

    send_whatsapp_message(sender, answer)
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
