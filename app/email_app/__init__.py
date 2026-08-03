"""Email HTTP + SES integration (separate from voice)."""

from app.email_app.routes import router

__all__ = ["router"]
