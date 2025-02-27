import time
from typing import List
import json
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
from agent.agent import process_query
from agent.model import MODEL
# Initialize
app = FastAPI()
logger = None


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

def call_model(input_question):
    logger.info(f"Processing question {input_question}")
    res = process_query(input_question, 1)
    logger.info(f"Got res {input_question}")
    output = json.loads(res)
    logger.info(f"Got output {output}")
    return output['answer'], output['reasoning'], output['sources']
@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")
        # Здесь будет вызов вашей модели
        answer, reasoning, sources = call_model(body.query)
        # answer = 1  # Замените на реальный вызов модели
        # sources: List[HttpUrl] = [
        #     HttpUrl("https://itmo.ru/ru/"),
        #     HttpUrl("https://abit.itmo.ru/"),
        # ]

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=f"{MODEL}: {reasoning}",
            sources=sources,
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
