from fastapi import FastAPI, HTTPException
import boto3
import json
import asyncio
import torch
from transformers import pipeline
from memory_store import update_memory, retrieve_memory, load_memory

# Initialize FastAPI
app = FastAPI()

# AWS Clients
sqs = boto3.client("sqs")
dynamodb = boto3.resource("dynamodb")

# Load FAISS Memory at Startup
load_memory()

# Load LLaMA 2 Model
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
generator = pipeline("text-generation", model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

# Environment Variables
CHATBOT_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/chatbot-requests"
KALEIDOSCOPE_ENGINE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/kaleidoscope-processing"
PERSPECTIVE_ENGINE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/perspective-processing"

@app.post("/chat")
async def chatbot_query(request: dict):
    """Handles chatbot queries using AI engines and memory retrieval."""
    user_query = request.get("query", "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query.")

    # Check Memory First
    memory_response = retrieve_memory(user_query)
    if memory_response:
        return {"response": memory_response[0]}

    # Generate AI Response
    response = generator(user_query, max_length=250, temperature=0.7, num_return_sequences=1)[0]["generated_text"].strip()

    # Store interaction in memory
    update_memory(user_query, response)

    # Send query to AI engines for deeper insights
    message = {"query": user_query, "chat_response": response}
    await send_to_sqs(KALEIDOSCOPE_ENGINE_URL, message)
    await send_to_sqs(PERSPECTIVE_ENGINE_URL, message)

    return {"response": response}

async def send_to_sqs(queue_url: str, message: dict):
    """Send message to AWS SQS."""
    try:
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message))
    except Exception as e:
        print(f"Failed to send message to {queue_url}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

