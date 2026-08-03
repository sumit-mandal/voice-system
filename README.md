# Healthcare Voice Intake

Inbound phone intake: **Twilio → self-hosted LiveKit (OSS) → FastAPI/LangGraph → PocketTTS + faster-whisper → SQLite**.

This project targets **open-source LiveKit** via Docker — not LiveKit Cloud.

## Stack

| Piece | Role |
| --- | --- |
| FastAPI | Twilio webhooks, debug chat, call lookup |
| Twilio | PSTN inbound calling |
| LiveKit Server (OSS) | Realtime audio rooms |
| LiveKit SIP (OSS, optional) | Bridge Twilio phone ↔ rooms |
| LangGraph + ChatNVIDIA | Slot-filling + clarification |
| faster-whisper | STT |
| PocketTTS | TTS |
| SQLite | Persist name, age, transcript |
| Human handoff | LangGraph detects intent → Twilio cold-transfers to `TWILIO_HUMAN_AGENT_NUMBER` |
| Email agent | SES + LangGraph classify/tools → SES auto-reply (`POST /email/debug`) |

## Setup

```bash
cd voice_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill NVIDIA + Twilio secrets
```

### 1) Start open-source LiveKit

```bash
docker compose up -d
```

This starts Redis + LiveKit Server in `--dev` mode:

| Env | Value |
| --- | --- |
| `LIVEKIT_URL` | `ws://localhost:7880` |
| `LIVEKIT_API_KEY` | `devkey` |
| `LIVEKIT_API_SECRET` | `secret` |

### 2) Run API + worker

```bash
./scripts/run_api.sh
./scripts/run_worker.sh
```

### 3) Test without a phone

```bash
curl -s localhost:8000/debug/chat \
  -H 'content-type: application/json' \
  -d '{"call_sid":"debug-1","text":"hi my name is Alex","reset":true}'
```

## Phone calls (Twilio + LiveKit SIP)

Phone bridging needs the **LiveKit SIP** service (also open-source), plus a **public IP** with ports open:

- UDP/TCP `5060` (SIP signaling)
- UDP `10000-20000` (RTP media)

On a laptop (especially macOS Docker), this is awkward. Use a cheap Linux VPS for SIP, or keep local LiveKit for agent/dev and add SIP later.

Steps when you have a public host:

1. Uncomment `livekit-sip` in `docker-compose.yml` (or run SIP on the VPS).
2. Set `LIVEKIT_SIP_HOST=<public-ip>:5060` in `.env`.
3. Point Twilio Elastic SIP / TwiML dial at that host.
4. Create trunk:

```bash
PYTHONPATH=. python scripts/create_sip_trunk.py
# paste printed LIVEKIT_SIP_TRUNK_ID into .env
```

5. Expose FastAPI (`ngrok http 8000`) and set Twilio voice webhook to  
   `POST https://<tunnel>/twilio/voice/inbound`

Official refs:

- [Self-host LiveKit](https://docs.livekit.io/home/self-hosting/local/)
- [Self-host SIP](https://docs.livekit.io/transport/self-hosting/sip-server/)

## Email agent (SES sandbox auto-reply)

While SES is in **sandbox**, you can only send *to* verified addresses (e.g. your Gmail).

1. Fill `.env`:
   - `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - `SES_FROM_ADDRESS` — verified identity (`intake@sumitmandal.in` or verified Gmail)
2. Ensure **both** From and To are verified in SES (sandbox).
3. `pip install boto3` (or `pip install -r requirements.txt`)
4. Start API: `./scripts/run_api.sh`
5. Simulate inbound + send reply:

```bash
curl -s localhost:8000/email/debug \
  -H 'content-type: application/json' \
  -d '{
    "from_address":"sumit.mandal13100@gmail.com",
    "subject":"My history",
    "body_text":"Please tell me about my history records.",
    "send_reply":true
  }'
```

Check Gmail for the auto-reply from `SES_FROM_ADDRESS`.

Optional real inbound (Gmail → intake@sumitmandal.in):

1. MX for `sumitmandal.in` must point to SES inbound in **the same region** as `AWS_REGION`:
   `10 inbound-smtp.<region>.amazonaws.com`
2. SES **receipt rule** for `intake@sumitmandal.in` → **S3** (`SES_INBOUND_BUCKET`)
3. After mailing from Gmail, check objects landed:

```bash
curl -s localhost:8000/email/ses/inbound
```

4. Process newest mail + send auto-reply:

```bash
curl -s -X POST 'localhost:8000/email/ses/process-latest'
```

5. (Optional auto) S3 event or SES → SNS → `POST https://voice.sumitmandal.in/email/ses/sns`

**Important:** Sending mail to `intake@…` does nothing by itself — the API only runs when S3/SNS triggers it (or you call `/email/debug` / `/email/ses/process-latest`).

## Project layout

```
app/
  main.py                 FastAPI
  config.py               Settings
  agent/                  Voice LangGraph intake
  email_agent/            Email LangGraph
  email_app/              SES send + /email/debug + SNS webhook
  voice/                  faster-whisper + PocketTTS
  livekit_app/            room helpers + worker
  twilio_app/             inbound webhooks
  db/                     SQLite
docker-compose.yml        OSS LiveKit + Redis
```
