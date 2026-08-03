"""Application settings — loaded from environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"
    log_level: str = "DEBUG"
    public_base_url: str = "http://localhost:8000"

    # Gemini (primary LLM for intake agent)
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.3
    gemini_top_p: float = 0.95
    gemini_max_tokens: int = 1024

    # Optional NVIDIA (unused while Gemini is primary)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "openai/gpt-oss-120b"
    nvidia_temperature: float = 1.0
    nvidia_top_p: float = 1.0
    nvidia_max_tokens: int = 1024
    nvidia_seed: int = 42

    database_url: str = "sqlite:///./data/healthcare_voice.db"

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    # E.164 number to dial when the AI hands off to a human (cold transfer).
    twilio_human_agent_number: str = ""

    livekit_url: str = "ws://localhost:7880"
    # Browser clients need a publicly reachable WSS URL (not localhost).
    # Leave empty to derive from public_base_url (https → wss).
    livekit_public_url: str = ""
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"
    livekit_sip_trunk_id: str = ""
    livekit_sip_host: str = ""

    stt_model_size: str = "base.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    tts_voice: str = "alba"
    tts_sample_rate: int = 24000

    # --- AWS SES (email agent) ---
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    # Verified SES identity used as From for auto-replies (domain or email).
    ses_from_address: str = ""
    # Optional: S3 bucket used by SES receipt rule for raw inbound MIME.
    ses_inbound_bucket: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
