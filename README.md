# CoopIQ

CoopIQ is a Flask WhatsApp chatbot for poultry farmers. It answers questions on
broiler, layer, and village chicken farming -- housing, brooding, feeding,
vaccination and disease prevention, biosecurity, budgeting, and marketing --
using retrieval-augmented generation (RAG) over a curated poultry-farming
knowledge base (Zambia-focused) and Google's Gemini models.

## Requirements

- Python 3.11 or newer
- WhatsApp Cloud API credentials
- Gemini API key

## Setup

Create and activate a virtual environment:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```
python -m pip install -r requirements.txt
```

Set environment variables:

```
$env:WA_TOKEN="your-whatsapp-token"
$env:PHONE_ID="your-phone-number-id"
$env:GEN_API="your-gemini-api-key"
$env:VERIFY_TOKEN="your-webhook-verify-token"
```

## Build the knowledge base embeddings

The RAG layer retrieves passages from `poultry_data.py` by vector search.
Before running the bot, generate the embedding cache once (and again
whenever `poultry_data.py` changes):

```
python build_embeddings.py
```

This writes `embeddings.json`, which `retrieval.py` loads at runtime.

## Run Locally

```
python main.py
```

The health/status page is available at:

```
http://127.0.0.1:5000/
```

Test WhatsApp webhook verification:

```
Invoke-WebRequest "http://127.0.0.1:5000/webhook?hub.mode=subscribe&hub.verify_token=your-webhook-verify-token&hub.challenge=12345"
```

Expected response:

```
12345
```

For live WhatsApp webhook testing, expose the local Flask server using a
tunnel such as ngrok, then configure the tunnel URL in the WhatsApp Cloud
API webhook settings.

## Project Structure

- `main.py` -- Flask app: webhook verification, incoming message handling,
  WhatsApp sends
- `retrieval.py` -- runtime vector search (cosine similarity) over cached
  embeddings
- `build_embeddings.py` -- offline script that embeds every passage in
  `poultry_data.py` with Gemini and writes `embeddings.json`
- `poultry_data.py` -- canonical knowledge base (source of truth for RAG
  content)
- `embeddings.json` -- generated cache of passage embeddings (not committed
  until you run `build_embeddings.py`)

## Important Notes

- Knowledge base content should be reviewed periodically for accuracy and
  currency, especially figures like vaccination schedules, dosages, and
  costs, which change over time.
- Answers are grounded strictly in `poultry_data.py` via the retrieval step;
  if nothing relevant is found, CoopIQ tells the farmer rather than
  guessing.

## Deploy

The project includes `vercel.json` for Vercel deployment. Configure the
same environment variables in Vercel before deploying, and run
`build_embeddings.py` locally to commit an up-to-date `embeddings.json`
before you deploy (Vercel's serverless functions should not call the
embeddings API on every cold start).
