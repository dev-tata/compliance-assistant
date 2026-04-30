from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

from app.api import routes_cases, routes_deliverables, routes_documents, routes_llm

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(
    title="compliance assistant API",
    version="0.1.0",
    description="Upload documents, extract requirements, assemble cases, and run compliance analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_documents.router)
app.include_router(routes_cases.router)
app.include_router(routes_deliverables.router)
app.include_router(routes_llm.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "compliance assistant API is running"}
