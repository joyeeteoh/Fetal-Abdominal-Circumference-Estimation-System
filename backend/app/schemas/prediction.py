from pydantic import BaseModel, HttpUrl
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, HttpUrl, Field

class PredictionRunRequest(BaseModel):
    input_image_url: HttpUrl
    scale: float = Field(gt=0)
    bmi: float = Field(gt=0)

class PredictionRunResponse(BaseModel):
    message: str
    ac_result: str
    segmented_image_url: Optional[str] = None
    processing_time_sec: Optional[float] = None

class PredictionRecordsCreate(BaseModel):
    image_filename: str
    scale: float
    bmi: float
    patient_rn: str
    ac_result: str
    input_image_url: Optional[str] = None
    segmented_image_url: Optional[str] = None
    processing_time_sec: Optional[float] = None

class PredictionRecordsOut(PredictionRecordsCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
