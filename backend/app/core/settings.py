import os
from pydantic import BaseModel


def _parse_origins(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


class Settings(BaseModel):
    # App
    APP_NAME: str = "Auth API"
    VERSION: str = "1.0.0"

    # Security
    JWT_SECRET: str | None = os.getenv("JWT_SECRET")
    JWT_ALG: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

    # CORS
    CORS_ORIGINS: list[str] = _parse_origins(os.getenv("CORS_ORIGINS"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

    # Cloudinary
    CLOUDINARY_CLOUD_NAME: str | None = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: str | None = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: str | None = os.getenv("CLOUDINARY_API_SECRET")
    CLOUDINARY_UNSIGNED_PRESET: str = os.getenv("CLOUDINARY_UNSIGNED_PRESET", "fyp_ac_unsigned")


settings = Settings()

if not settings.JWT_SECRET:
    raise RuntimeError("JWT_SECRET is missing. Set it in environment variables (.env or deployment config).")
