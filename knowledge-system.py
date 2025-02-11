import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, List, Optional, Tuple
import networkx as nx
from dataclasses import dataclass
import asyncio
import logging

@dataclass
class KnowledgeFragment:
    embedding: torch.Tensor
    connections: List[int]
    confidence: float
    source_generation: int
    last_accessed: float

class AdaptiveTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.transformer(x, mask)
        return self.projection(x)

class KnowledgeBase:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fragments: List[KnowledgeFragment] = []
        self.knowledge_graph = nx.DiGraph()
        self.transformer = AdaptiveTransformer(
            d_model=config['embedding_dim'],
            nhead=config['num_heads'],
            num_layers=config['num_layers']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=config['learning_rate']
        )
        self.logger = logging.getLogger("KnowledgeBase")

    async def integrate_insights(self, insights: List[Dict]) -> Tuple[bool, float]:
        embeddings = []
        for insight in insights:
            embedding = self._encode_insight(insight)
            embeddings.append(embedding)
        
        embeddings_tensor = torch.stack(embeddings).to(self.device)
        novelty_scores = self._calculate_novelty(embeddings_tensor)
        
        if novelty_scores.mean() > self.config['novelty_threshold']:
            await self._update_knowledge(embeddings_tensor, insights)
            return True, novelty_scores.mean().item()
        
        return False, novelty_scores.mean().item()

    def _encode_insight(self, insight: Dict) -> torch.Tensor:
        values = []
        
        def extract_values(obj):
            if isinstance(obj, (int, float)):
                values.append(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_values(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    extract_values(v)
        
        extract_values(insight)
        tensor = torch.tensor(values, dtype=torch.float32)
        
        # Pad or truncate to fixed dimension
        if len(tensor) < self.config['embedding_dim']:
            tensor = nn.functional.pad(tensor, (0, self.config['embedding_dim'] - len(tensor)))
        else:
            tensor = tensor[:self.config['embedding_dim']]
            
        return tensor

    def _calculate_novelty(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.fragments:
            return torch.ones(embeddings.size(0))
            
        existing = torch.stack([f.embedding for f in self.fragments])
        
        similarities = torch.cdist(embeddings, existing)
        min_distances = similarities.min(dim=1)[0]
        
        # Normalize to [0,1]
        novelty = min_distances / min_distances.max()
        return novelty

    async def _update_knowledge(self, embeddings: torch.Tensor, insights: List[Dict]):
        # Process embeddings through transformer
        processed = self.transformer(embeddings)
        
        for i, (embedding, insight) in enumerate(zip(processed, insights)):
            fragment = KnowledgeFragment(
                embedding=embedding,
                connections=[],
                confidence=1.0,
                source_generation=insight.get('generation', 0),
                last_accessed=asyncio.get_event_loop().time()
            )
            
            # Add to knowledge graph
            self.fragments.append(fragment)
            idx = len(self.fragments) - 1
            self.knowledge_graph.add_node(
                idx,
                embedding=embedding.detach().cpu().numpy(),
                generation=fragment.source_generation
            )
            
            # Connect to related fragments
            self._update_connections(idx)
        
        # Optimize knowledge structure
        await self._optimize_knowledge()

    def _update_connections(self, new_idx: int):
        new_embedding = self.fragments[new_idx].embedding
        
        for i, fragment in enumerate(self.fragments[:-1]):  # Exclude the new fragment
            similarity = torch.cosine_similarity(
                new_embedding.unsqueeze(0),
                fragment.embedding.unsqueeze(0)
            )
            
            if similarity > self.config['connection_threshold']:
                self.fragments[new_idx].connections.append(i)
                self.fragments[i].connections.append(new_idx)
                
                self.knowledge_graph.add_edge(
                    new_idx,
                    i,
                    weight=similarity.item()
                )

    async def _optimize_knowledge(self):
        # Get embeddings
        embeddings = torch.stack([f.embedding for f in self.fragments])
        
        # Train transformer to compress and enhance knowledge
        self.transformer.train()
        for _ in range(self.config['optimization_steps']):
            self.optimizer.zero_grad()
            
            output = self.transformer(embeddings)
            
            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(output, embeddings)
            
            # Connectivity loss based on knowledge graph
            adj_matrix = nx.adjacency_matrix(self.knowledge_graph).todense()
            adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(self.device)
            
            pred_adj = torch.matmul(output, output.transpose(0, 1))
            conn_loss = nn.functional.mse_loss(pred_adj, adj_tensor)
            
            # Combined loss
            loss = recon_loss + self.config['connectivity_weight'] * conn_loss
            loss.backward()
            
            self.optimizer.step()
        
        self.transformer.eval()

    async def query_knowledge(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict]:
        if not self.fragments:
            return []
            
        # Update access times
        current_time = asyncio.get_event_loop().time()
        
        # Calculate similarities
        similarities = []
        for fragment in self.fragments:
            sim = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                fragment.embedding.unsqueeze(0)
            )
            similarities.append(sim.item())
            fragment.last_accessed = current_time
            
        # Get top-k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            fragment = self.fragments[idx]
            
            # Get connected knowledge
            connected = [self.fragments[i] for i in fragment.connections]
            connected_embeddings = torch.stack([f.embedding for f in connected]) if connected else None
            
            results.append({
                'embedding': fragment.embedding.detach().cpu().numpy(),
                'confidence': fragment.confidence * similarities[idx],
                'generation': fragment.source_generation,
                'connected_knowledge': connected_embeddings.detach().cpu().numpy() if connected_embeddings is not None else None
            })
            
        return results

    async def prune_knowledge(self):
        current_time = asyncio.get_event_loop().time()
        
        # Calculate access scores
        scores = []
        for fragment in self.fragments:
            time_factor = np.exp(-(current_time - fragment.last_accessed) / self.config['time_decay'])
            connectivity = len(fragment.connections) / len(self.fragments)
            score = time_factor * fragment.confidence * connectivity
            scores.append(score)
            
        # Remove low-scoring fragments
        threshold = np.percentile(scores, self.config['prune_percentile'])
        keep_indices = [i for i, score in enumerate(scores) if score >= threshold]
        
        # Update fragments and graph
        self.fragments = [self.fragments[i] for i in keep_indices]
        self.knowledge_graph = self.knowledge_graph.subgraph(keep_indices).copy()
        
        # Renumber nodes
        mapping = {old: new for new, old in enumerate(keep_indices)}
        self.knowledge_graph = nx.relabel_nodes(self.knowledge_graph, mapping)
        
        # Update connections
        for fragment in self.fragments:
            fragment.connections = [mapping[c] for c in fragment.connections if c in mapping]

async def main():
    config = {
        'embedding_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'learning_rate': 1e-4,
        'novelty_threshold': 0.3,
        'connection_threshold': 0.7,
        'connectivity_weight': 0.5,
        'optimization_steps': 100,
        'time_decay': 3600,
        'prune_percentile': 10
    }
    
    kb = KnowledgeBase(config)
    
    # Example insights
    insights = [
        {'data': [1, 2, 3], 'generation': 1},
        {'data': [4, 5, 6], 'generation': 1}
    ]
    
    # Integrate insights
    updated, novelty = await kb.integrate_insights(insights)
    print(f"Knowledge updated: {updated}, Novelty: {novelty:.3f}")
    
    # Query example
    query = torch.randn(config['embedding_dim'])
    results = await kb.query_knowledge(query)
    print(f"Query results: {len(results)} matches")
    
    # Prune knowledge
    await kb.prune_knowledge()

if __name__ == "__main__":
    asyncio.run(main())
