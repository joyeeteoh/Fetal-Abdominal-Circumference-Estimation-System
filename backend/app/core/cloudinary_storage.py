import os
from typing import Any, Dict, Optional

import cloudinary
import cloudinary.uploader

_configured = False


def init_cloudinary() -> None:
    """
    Configure Cloudinary once per process.
    """
    global _configured
    if _configured:
        return

    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")

    if not cloud_name or not api_key or not api_secret:
        raise RuntimeError(
            "Missing Cloudinary env vars: CLOUDINARY_CLOUD_NAME / CLOUDINARY_API_KEY / CLOUDINARY_API_SECRET"
        )

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    _configured = True


def upload_image_bytes(
    data: bytes,
    *,
    public_id: Optional[str] = None,
    folder: Optional[str] = None,
    overwrite: bool = True,
    resource_type: str = "image",
    **options: Any,
) -> Dict[str, Any]:
    """
    Upload raw bytes to Cloudinary and return the upload response dict.
    """
    init_cloudinary()

    upload_opts: Dict[str, Any] = {
        "resource_type": resource_type,
        "overwrite": overwrite,
        **options,
    }
    if folder:
        upload_opts["folder"] = folder
    if public_id:
        upload_opts["public_id"] = public_id

    return cloudinary.uploader.upload(data, **upload_opts)
