from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.settings import settings
from app.db.session import engine
from app.db.base import Base

from app.models import user  # noqa
from app.models import prediction  # noqa

from app.routers.auth import router as auth_router
from app.routers.prediction import router as prediction_router
from app.routers.files import router as files_router

# Auto-create tables for local development
if os.getenv("AUTO_CREATE_TABLES", "false").lower() == "true":
    Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(prediction_router)
app.include_router(files_router)

@app.get("/")
def root():
    return {"status": "ok"}