from fastapi import APIRouter, HTTPException
from app.core.settings import settings
import os

router = APIRouter(prefix="/api/files", tags=["Files"])

@router.get("/cloudinary-config")
def cloudinary_config():
    if not settings.CLOUDINARY_CLOUD_NAME or not settings.CLOUDINARY_UNSIGNED_PRESET:
        raise HTTPException(status_code=500, detail="Cloudinary is not configured on server")

    return {
        "cloud_name": settings.CLOUDINARY_CLOUD_NAME,
        "unsigned_preset": settings.CLOUDINARY_UNSIGNED_PRESET,
        "input_folder": os.getenv("CLOUDINARY_INPUT_FOLDER", "fyp_ac/inputs"),
    }
