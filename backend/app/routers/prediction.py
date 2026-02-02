import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import time

import requests
import numpy as np
import cv2

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.prediction import PredictionRecords
from app.schemas.prediction import (
    PredictionRunRequest,
    PredictionRunResponse,
    PredictionRecordsCreate,
    PredictionRecordsOut,
)

from app.core.cloudinary_storage import upload_image_bytes

from model.inference import run_pipeline_bmi_ge_30_inmemory, run_pipeline_bmi_lt_30_inmemory

router = APIRouter(prefix="/api/prediction", tags=["Prediction"])


def _download_image_bytes(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download input image: {e}")


def _decode_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Uploaded URL is not a valid image")
    return img


@router.post("/run", response_model=PredictionRunResponse)
def run_prediction(
    payload: PredictionRunRequest,
    current_user: User = Depends(get_current_user),
):
    t0 = time.perf_counter()

    image_bytes = _download_image_bytes(str(payload.input_image_url))
    image_bgr = _decode_to_bgr(image_bytes)

    try:
        if payload.bmi >= 30.0:
            ac_str, mask_png_bytes, predicted_pixels = run_pipeline_bmi_ge_30_inmemory(
                image_bgr=image_bgr,
                bmi=payload.bmi,
                scale_pixels_per_cm=payload.scale,
            )
        else:
            ac_str, mask_png_bytes, predicted_pixels = run_pipeline_bmi_lt_30_inmemory(
                image_bgr=image_bgr,
                bmi=payload.bmi,
                scale_pixels_per_cm=payload.scale,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    upload_result = upload_image_bytes(
        data=mask_png_bytes,
        folder=os.getenv("CLOUDINARY_OUTPUT_FOLDER", "fyp_ac/outputs"),
        public_id=None,
        overwrite=False,
        resource_type="image",
    )

    segmented_url = upload_result.get("secure_url")
    if not segmented_url:
        raise HTTPException(status_code=500, detail="Cloudinary upload failed: missing secure_url")

    processing_time_sec = round(time.perf_counter() - t0, 3)

    return PredictionRunResponse(
        message="Abdominal Circumference Predicted",
        ac_result=ac_str,
        segmented_image_url=segmented_url,
        processing_time_sec=processing_time_sec,
    )


@router.post("/records", response_model=PredictionRecordsOut)
def save_records(
    payload: PredictionRecordsCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = PredictionRecords(
        user_id=current_user.id,
        image_filename=payload.image_filename,
        scale=payload.scale,
        bmi=payload.bmi,
        patient_rn=payload.patient_rn,
        ac_result=payload.ac_result,
        input_image_url=payload.input_image_url,
        segmented_image_url=payload.segmented_image_url,
        processing_time_sec=payload.processing_time_sec,
    )
    db.add(records)
    db.commit()
    db.refresh(records)
    return records


@router.get("/records", response_model=List[PredictionRecordsOut])
def list_records(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = (
        db.query(PredictionRecords)
        .filter(PredictionRecords.user_id == current_user.id)
        .order_by(PredictionRecords.created_at.desc())
        .all()
    )
    return items
