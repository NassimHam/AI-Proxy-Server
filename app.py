from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import CORS_ALLOW_ORIGINS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import handlers to register routes
import handlers  # noqa: F401
