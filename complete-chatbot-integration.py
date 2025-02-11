import asyncio
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import boto3
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
import faiss
from scipy.spatial.distance import cosine
from collections import deque
import redis.asyncio as redis
import msgpack
from prometheus_client import Counter, Histogram, start_http_server

@dataclass
class ChatbotMetrics:
    response_time: Histogram = Histogram('response_time_seconds', 'Response time in seconds')
    message_count: Counter = Counter('message_count', 'Total messages processed')
    token_usage: Counter = Counter('token_usage', 'Total tokens used')
    error_count: Counter = Counter('error_count', 'Total errors encountered')

@dataclass
class ChatbotConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_history: int = 50
    embedding_dim: int = 384
    redis_url: str = "redis://localhost:6379"
    enrichment_bucket: str = "kaleidoscope-enrichment"
    index_table: str = "enrichment-index"

class Message(BaseModel):
    content: str
    metadata: Optional[Dict] = None

class KaleidoscopeChatbot:
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.metrics = ChatbotMetrics()
        self.dynamo = boto3.resource('dynamodb')
        self.s3 = boto3.client('s3')
        self.logger = logging.getLogger('KaleidoscopeChatbot')
        self.conversation_history = {}
        self.active_connections = set()
        self.supernode_queue = asyncio.Queue()
        self.redis_client = redis.Redis.from_url(config.redis_url)
        self.vector_index = faiss.IndexFlatL2(config.embedding_dim)
        self.knowledge_cache = {}
        
        # Initialize FastAPI
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Start metrics server
        start_http_server(9090)

    async def initialize(self):
        await self._initialize_routes()
        await self._load_knowledge_base()
        await self._start_background_tasks()

    async def _initialize_routes(self):
        @self.app.websocket("/ws/chat/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.handle_websocket_connection(websocket, client_id)

        @self.app.post("/api/chat/{client_id}")
        async def chat_endpoint(client_id: str, message: Message, background_tasks: BackgroundTasks):
            with self.metrics.response_time.time():
                response = await self.process_message(message.content, client_id)
                background_tasks.add_task(self._process_message_background, message, response, client_id)
                return response

    async def _load_knowledge_base(self):
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.config.enrichment_bucket)
        
        async for page in pages:
            for obj in page.get('Contents', []):
                data = await self._get_document_from_s3(obj['Key'])
                if data and 'embedding' in data:
                    self.vector_index.add(np.array([data['embedding']]))
                    self.knowledge_cache[obj['Key']] = data

    async def _start_background_tasks(self):
        asyncio.create_task(self._process_supernode_queue())
        asyncio.create_task(self._update_knowledge_base())
        asyncio.create_task(self._cleanup_old_conversations())

    async def handle_websocket_connection(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.add(websocket)
        
        try:
            while True:
                message = await websocket.receive_text()
                with self.metrics.response_time.time():
                    response = await self.process_message(message, client_id)
                await websocket.send_text(json.dumps(response))
                self.metrics.message_count.inc()
        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
            self.metrics.error_count.inc()
        finally:
            self.active_connections.remove(websocket)

    async def process_message(self, message: str, client_id: str) -> Dict:
        inputs = self.tokenizer(message, return_tensors="pt", padding=True)
        self.metrics.token_usage.inc(inputs['input_ids'].shape[1])
        
        with torch.no_grad():
            message_embedding = self.model.get_input_embeddings()(inputs['input_ids']).mean(dim=1)

        context = await self._get_relevant_context(message_embedding)
        response = await self._generate_response(message, context, client_id)
        
        await self._store_interaction(message, response, client_id)
        await self.supernode_queue.put({
            'message': message,
            'response': response,
            'embedding': message_embedding.cpu().numpy().tolist(),
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        })
        
        return response

    async def _get_relevant_context(self, query_embedding: torch.Tensor) -> List[str]:
        cache_key = f"context:{hash(str(query_embedding.cpu().numpy().tobytes()))}"
        cached_context = await self.redis_client.get(cache_key)
        
        if cached_context:
            return msgpack.unpackb(cached_context)
            
        D, I = self.vector_index.search(
            query_embedding.cpu().numpy().reshape(1, -1),
            k=5
        )
        
        relevant_docs = []
        for idx in I[0]:
            if idx != -1:
                doc_key = list(self.knowledge_cache.keys())[idx]
                doc = self.knowledge_cache[doc_key]
                relevant_docs.append(doc['content'])
        
        await self.redis_client.setex(
            cache_key,
            3600,  # Cache for 1 hour
            msgpack.packb(relevant_docs)
        )
        
        return relevant_docs

    async def _generate_response(self, message: str, context: List[str], client_id: str) -> Dict:
        history = await self._get_conversation_history(client_id)
        prompt = self._construct_prompt(message, context, history)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        await self._update_conversation_history(client_id, message, response_text)
        
        return {
            'response': response_text,
            'metadata': {
                'context_used': len(context),
                'tokens_used': outputs.shape[1],
                'timestamp': datetime.now().isoformat()
            }
        }

    async def _process_supernode_queue(self):
        while True:
            try:
                data = await self.supernode_queue.get()
                await self._enrich_knowledge_base(data)
                self.supernode_queue.task_done()
            except Exception as e:
                self.logger.error(f"Supernode queue error: {str(e)}")
                self.metrics.error_count.inc()
            await asyncio.sleep(0.1)

    async def _enrich_knowledge_base(self, data: Dict):
        embedding = np.array(data['embedding'])
        self.vector_index.add(embedding.reshape(1, -1))
        
        key = f"enrichment/{datetime.now().isoformat()}/{hash(str(embedding))}"
        self.knowledge_cache[key] = {
            'content': data['message'],
            'embedding': data['embedding'],
            'metadata': {
                'source': 'conversation',
                'client_id': data['client_id'],
                'timestamp': data['timestamp']
            }
        }
        
        await self._store_enrichment(key, self.knowledge_cache[key])

    async def _store_enrichment(self, key: str, data: Dict):
        self.s3.put_object(
            Bucket=self.config.enrichment_bucket,
            Key=key,
            Body=json.dumps(data)
        )
        
        self.dynamo.Table(self.config.index_table).put_item(Item={
            'key': key,
            'timestamp': data['metadata']['timestamp'],
            'embedding_hash': hash(str(data['embedding']))
        })

    async def _cleanup_old_conversations(self):
        while True:
            try:
                current_time = datetime.now()
                for client_id in list(self.conversation_history.keys()):
                    history = self.conversation_history[client_id]
                    if isinstance(history, deque) and len(history) > 0:
                        oldest_message_time = datetime.fromisoformat(history[0]['timestamp'])
                        if (current_time - oldest_message_time).days > 7:
                            del self.conversation_history[client_id]
            except Exception as e:
                self.logger.error(f"Cleanup error: {str(e)}")
            await asyncio.sleep(3600)  # Run every hour

    def _construct_prompt(self, message: str, context: List[str], history: List[Dict]) -> str:
        context_text = "\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(context))
        history_text = "\n".join(
            f"User: {h['user']}\nAssistant: {h['bot']}"
            for h in history[-5:]  # Last 5 messages
        )
        
        return f"""Context:
{context_text}

Previous conversation:
{history_text}

User: {message}
Assistant:"""

if __name__ == "__main__":
    config = ChatbotConfig()
    chatbot = KaleidoscopeChatbot(config)
    
    async def main():
        await chatbot.initialize()
        uvicorn.run(chatbot.app, host="0.0.0.0", port=8000)
    
    asyncio.run(main())