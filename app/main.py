from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query as QueryParam
from app.instrumentation import verify_langfuse_connection
from app.rag_pipeline import run_rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    verify_langfuse_connection()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/query")
async def query_endpoint(
    q: str = QueryParam(..., min_length=3, description="User query string"),
    user_id: str = QueryParam(
        default="anonymous", description="User identifier for cost attribution"
    ),
):
    result = run_rag_pipeline(query=q, user_id=user_id)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
