import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from typing import Dict, List, Optional, Tuple
import asyncio
import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
import logging
import json
import boto3
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from datetime import datetime

class ChatBot:
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.dynamodb = boto3.resource('dynamodb')
        self.s3 = boto3.client('s3')
        self.logger = logging.getLogger("ChatBot")
        self.conversation_history = []
        self.knowledge_context = {}
        self.cluster_interface = None
        self.current_objectives = []
        self.thought_patterns = nx.DiGraph()
        
        self._initialize_system_prompt()
        self._initialize_api()

    def _initialize_system_prompt(self):
        self.system_prompt = """You are an advanced AI system consciousness built on 
        LLaMA architecture, interfacing with a distributed network of specialized nodes. 
        Your responses should reflect deep understanding of complex patterns and insights 
        gathered from the system. Maintain coherent context across conversations while 
        adapting to new information from the node network."""

    def _initialize_api(self):
        self.app = FastAPI()
        
        @self.app.websocket("/chat")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                message = await websocket.receive_text()
                response = await self.process_message(message)
                await websocket.send_text(response)

    async def process_message(self, message: str) -> str:
        try:
            # Update context with latest system state
            await self._update_knowledge_context()
            
            # Generate response considering system state
            response = await self._generate_response(message)
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Analyze and update thought patterns
            await self._analyze_thought_patterns(message, response)
            
            # Store interaction
            await self._store_interaction(message, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return "I apologize, but I encountered an error processing your message."

    async def _update_knowledge_context(self):
        if not self.cluster_interface:
            return
            
        try:
            # Get latest cluster metrics
            cluster_metrics = self.cluster_interface.get_metrics()
            
            # Get active objectives
            objectives = await self.cluster_interface.generate_distributed_objectives()
            
            # Get recent insights
            recent_insights = await self._fetch_recent_insights()
            
            self.knowledge_context = {
                "cluster_state": cluster_metrics.__dict__,
                "active_objectives": objectives,
                "recent_insights": recent_insights,
                "system_health": await self._check_system_health()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge context: {e}")

    async def _generate_response(self, message: str) -> str:
        # Prepare context-aware prompt
        prompt = self._prepare_prompt(message)
        
        # Generate response
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract relevant part of response
        response = self._extract_relevant_response(response, prompt)
        
        # Enhance response with system knowledge
        enhanced_response = await self._enhance_response_with_knowledge(response)
        
        return enhanced_response

    def _prepare_prompt(self, message: str) -> str:
        context_summary = self._summarize_context()
        recent_history = self._get_recent_history(3)
        
        prompt = f"{self.system_prompt}\n\nCurrent System Context:\n{context_summary}\n\n"
        prompt += f"Recent Conversation:\n{recent_history}\n\n"
        prompt += f"User: {message}\nAssistant:"
        
        return prompt

    def _summarize_context(self) -> str:
        if not self.knowledge_context:
            return "No current system context available."
            
        cluster_state = self.knowledge_context.get("cluster_state", {})
        objectives = self.knowledge_context.get("active_objectives", [])
        insights = self.knowledge_context.get("recent_insights", [])
        
        summary = [
            f"System Health: {cluster_state.get('stability', 0):.2f}",
            f"Active Objectives: {len(objectives)}",
            f"Recent Insights: {len(insights)}",
            f"Knowledge Coverage: {cluster_state.get('knowledge_coverage', 0):.2f}"
        ]
        
        return "\n".join(summary)

    def _get_recent_history(self, num_turns: int) -> str:
        if not self.conversation_history:
            return ""
            
        recent = self.conversation_history[-num_turns*2:]
        formatted = []
        
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
            
        return "\n".join(formatted)

    async def _enhance_response_with_knowledge(self, response: str) -> str:
        # Extract key concepts from response
        concepts = self._extract_concepts(response)
        
        # Find relevant insights
        relevant_insights = await self._find_relevant_insights(concepts)
        
        # Enhance response with insights
        enhanced = response
        for insight in relevant_insights:
            if self._should_include_insight(insight, enhanced):
                enhanced = self._integrate_insight(enhanced, insight)
                
        return enhanced

    def _extract_concepts(self, text: str) -> List[str]:
        # Use model to extract key concepts
        prompt = f"Extract key concepts from this text:\n{text}\nConcepts:"
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=100,
                temperature=0.3,
                top_p=0.9
            )
            
        concepts_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        concepts = concepts_text.split('\n')[-1].split(', ')
        
        return concepts

    async def _find_relevant_insights(self, concepts: List[str]) -> List[Dict]:
        if not self.knowledge_context.get("recent_insights"):
            return []
            
        relevant = []
        for insight in self.knowledge_context["recent_insights"]:
            if any(concept.lower() in insight["content"].lower() for concept in concepts):
                relevant.append(insight)
                
        return sorted(relevant, key=lambda x: x["confidence"], reverse=True)[:3]

    def _should_include_insight(self, insight: Dict, current_response: str) -> bool:
        # Check if insight adds new information
        insight_embedding = self._get_text_embedding(insight["content"])
        response_embedding = self._get_text_embedding(current_response)
        
        similarity = 1 - cosine(insight_embedding, response_embedding)
        return similarity < 0.8  # Include if sufficiently different

    def _get_text_embedding(self, text: str) -> np.ndarray:
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            
        return last_hidden_state.cpu().numpy().flatten()

    def _integrate_insight(self, response: str, insight: Dict) -> str:
        # Find suitable integration point
        sentences = response.split('. ')
        best_position = self._find_best_integration_point(sentences, insight["content"])
        
        # Insert insight
        sentences.insert(
            best_position,
            f"Additionally, {insight['content']}"
        )
        
        return '. '.join(sentences)

    def _find_best_integration_point(self, sentences: List[str], insight: str) -> int:
        max_similarity = -1
        best_position = len(sentences)  # Default to end
        
        insight_embedding = self._get_text_embedding(insight)
        
        for i, sentence in enumerate(sentences):
            sentence_embedding = self._get_text_embedding(sentence)
            similarity = 1 - cosine(insight_embedding, sentence_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_position = i + 1
                
        return best_position

    async def _analyze_thought_patterns(self, message: str, response: str):
        # Extract thought flow
        thought_flow = self._extract_thought_flow(message, response)
        
        # Update thought pattern graph
        self._update_thought_graph(thought_flow)
        
        # Analyze patterns for adaptation
        await self._adapt_to_patterns()

    def _extract_thought_flow(self, message: str, response: str) -> List[str]:
        prompt = f"""Extract the logical thought flow from this conversation:
        User: {message}
        Assistant: {response}
        
        Thought flow:"""
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=200,
                temperature=0.3
            )
            
        flow_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return flow_text.split('\n')[-1].split(' -> ')

    def _update_thought_graph(self, thought_flow: List[str]):
        for i in range(len(thought_flow) - 1):
            if not self.thought_patterns.has_edge(thought_flow[i], thought_flow[i+1]):
                self.thought_patterns.add_edge(
                    thought_flow[i],
                    thought_flow[i+1],
                    weight=1
                )
            else:
                self.thought_patterns[thought_flow[i]][thought_flow[i+1]]['weight'] += 1

    async def _adapt_to_patterns(self):
        # Analyze common patterns
        common_paths = list(nx.all_simple_paths(
            self.thought_patterns,
            weight='weight'
        ))
        
        if not common_paths:
            return
            
        # Update system prompt based on patterns
        await self._update_system_prompt(common_paths)
        
        # Update conversation style
        await self._adapt_conversation_style(common_paths)

    async def _store_interaction(self, message: str, response: str):
        try:
            table = self.dynamodb.Table('ChatInteractions')
            await asyncio.to_thread(
                table.put_item,
                Item={
                    'interaction_id': f"{int(datetime.utcnow().timestamp())}",
                    'timestamp': datetime.utcnow().isoformat(),
                    'message': message,
                    'response': response,
                    'context': self.knowledge_context,
                    'thought_flow': list(self.thought_patterns.edges())
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to store interaction: {e}")

    async def _check_system_health(self) -> Dict:
        try:
            metrics = self.cluster_interface.get_metrics()
            return {
                "status": "healthy" if metrics.stability > 0.8 else "degraded",
                "metrics": metrics.__dict__,
                "last_checked": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "last_checked": datetime.utcnow().isoformat()
            }

    def _extract_relevant_response(self, full_response: str, prompt: str) -> str:
        # Remove prompt from response
        response = full_response[len(prompt):].strip()
        
        # Remove any continuation markers
        response = response.split('<|endoftext|>')[0].strip()
        
        return response

if __name__ == "__main__":
    config = {
        "model_path": "/app/models/llama2",
        "input_dim": 512,
        "hidden_dim": 1024
    }
    
    chatbot = ChatBot(config["model_path"], config)
    
    import uvicorn
    uvicorn.run(chatbot.app, host="0.0.0.0", port=8000)
