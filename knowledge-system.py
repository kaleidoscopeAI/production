import json
import os
import faiss
import torch
from sentence_transformers import SentenceTransformer
from supernode_core import SuperNode
from typing import Dict, List

class KnowledgeSystem:
    """Manages AI Knowledge Storage, Learning Feedback, and SuperNode Relationships."""

    def __init__(self, memory_store_path="knowledge_memory.json", embedding_model="all-MiniLM-L6-v2"):
        self.memory_store_path = memory_store_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self.memory_index = faiss.IndexFlatL2(384)  # 384-D embeddings for efficient search
        self.memory_data = []
        self.supernodes = {}

        if os.path.exists(self.memory_store_path):
            self._load_memory()

    def _load_memory(self):
        """Loads AI knowledge memory from disk."""
        with open(self.memory_store_path, "r") as f:
            data = json.load(f)
            self.memory_data = data["knowledge"]
            self.memory_index.add(torch.tensor(data["vectors"]).numpy())

    def store_knowledge(self, text: str, supernode_id: str):
        """Stores AI insights in the knowledge system and maps to SuperNodes."""
        embedding = self.embedding_model.encode(text).tolist()
        self.memory_index.add(torch.tensor([embedding]).numpy())
        self.memory_data.append({"supernode_id": supernode_id, "text": text, "vector": embedding})
        self._save_memory()
        print(f"Stored knowledge: {text}")

    def retrieve_similar_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieves relevant AI knowledge based on input query."""
        query_embedding = self.embedding_model.encode(query).reshape(1, -1)
        _, idxs = self.memory_index.search(query_embedding, top_k)
        return [self.memory_data[i]["text"] for i in idxs[0] if i < len(self.memory_data)]

    def _save_memory(self):
        """Persists AI memory to disk."""
        with open(self.memory_store_path, "w") as f:
            json.dump({"knowledge": self.memory_data, "vectors": [item["vector"] for item in self.memory_data]}, f)

    def integrate_supernode(self, supernode: SuperNode):
        """Integrates SuperNodes into the AI Knowledge System."""
        self.supernodes[supernode.id] = supernode
        print(f"SuperNode {supernode.id} integrated into the Knowledge System.")

if __name__ == "__main__":
    ks = KnowledgeSystem()
    
    # Example usage
    ks.store_knowledge("Quantum resonance analysis improves insight clustering.", supernode_id="SN-001")
    retrieved_knowledge = ks.retrieve_similar_knowledge("resonance insights", top_k=2)
    print(f"Retrieved Knowledge: {retrieved_knowledge}")

