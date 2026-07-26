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

## Project layout

```
app/
  main.py                 FastAPI
  config.py               Settings
  agent/                  LangGraph + ChatNVIDIA
  voice/                  faster-whisper + PocketTTS
  livekit_app/            room helpers + worker
  twilio_app/             inbound webhooks
  db/                     SQLite
docker-compose.yml        OSS LiveKit + Redis
```
