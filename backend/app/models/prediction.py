from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from app.db.base import Base

class PredictionRecords(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    image_filename = Column(String, nullable=False)
    scale = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    patient_rn = Column(String, nullable=True)
    ac_result = Column(String, nullable=True)

    # URLs for stored input image and output mask
    input_image_url = Column(String, nullable=True)
    segmented_image_url = Column(String, nullable=True)

    # store processing time in seconds
    processing_time_sec = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


