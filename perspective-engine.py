import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
import networkx as nx
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@dataclass
class Perspective:
    scenario: torch.Tensor
    probability: float
    impact: float
    timeline: str
    supporting_evidence: List[str]
    counter_evidence: List[str]

class GenerativeSpeculator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.scenario_generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=4
        )
        
        self.probability_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.impact_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, num_scenarios: int = 5) -> List[torch.Tensor]:
        encoded = self.encoder(x)
        
        scenarios = []
        for _ in range(num_scenarios):
            noise = torch.randn_like(encoded) * 0.1
            scenario = self.scenario_generator(encoded + noise, encoded)
            scenarios.append(scenario)
            
        return scenarios

class PerspectiveEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speculator = GenerativeSpeculator(
            config['input_dim'],
            config['hidden_dim']
        ).to(self.device)
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.knowledge_graph = nx.DiGraph()

    async def generate_perspectives(self, data: torch.Tensor) -> List[Perspective]:
        # Generate scenarios
        scenarios = self.speculator(data)
        
        perspectives = []
        for scenario in scenarios:
            # Calculate probability and impact
            prob = self.speculator.probability_head(scenario).item()
            impact = self.speculator.impact_head(scenario).item()
            
            # Generate narrative
            timeline = await self._generate_timeline(scenario)
            
            # Analyze evidence
            supporting, counter = await self._analyze_evidence(scenario, data)
            
            perspectives.append(Perspective(
                scenario=scenario,
                probability=prob,
                impact=impact,
                timeline=timeline,
                supporting_evidence=supporting,
                counter_evidence=counter
            ))
            
        return perspectives

    async def _generate_timeline(self, scenario: torch.Tensor) -> str:
        # Convert scenario to text prompt
        prompt = self._scenario_to_text(scenario)
        
        # Generate timeline using GPT-2
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.gpt.generate(
            inputs, 
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0])

    def _scenario_to_text(self, scenario: torch.Tensor) -> str:
        # Extract key features
        features = scenario.mean(dim=0)
        top_k = torch.topk(features, k=5)
        
        # Convert to narrative prompt
        prompts = []
        for idx, value in zip(top_k.indices, top_k.values):
            prompts.append(f"Feature {idx}: {value:.2f}")
            
        return "Given the following scenario:\n" + "\n".join(prompts)

    async def _analyze_evidence(self, scenario: torch.Tensor, data: torch.Tensor) -> tuple:
        # Calculate correlation
        correlation = torch.corrcoef(
            torch.stack([scenario.flatten(), data.flatten()])
        )[0,1].item()
        
        # Find supporting patterns
        supporting = []
        if correlation > 0.3:
            top_features = self._find_top_features(scenario, data)
            supporting.extend(
                f"Strong correlation in feature {idx}" 
                for idx in top_features
            )
            
        # Find contradicting patterns
        counter = []
        if correlation < -0.3:
            opposing_features = self._find_opposing_features(scenario, data)
            counter.extend(
                f"Negative correlation in feature {idx}"
                for idx in opposing_features
            )
            
        return supporting, counter

    def _find_top_features(self, scenario: torch.Tensor, data: torch.Tensor) -> List[int]:
        similarities = torch.cosine_similarity(
            scenario.unsqueeze(0),
            data.unsqueeze(0)
        )
        return torch.topk(similarities, k=3).indices.tolist()

    def _find_opposing_features(self, scenario: torch.Tensor, data: torch.Tensor) -> List[int]:
        differences = torch.abs(scenario - data)
        return torch.topk(differences, k=3).indices.tolist()

    def update_knowledge_graph(self, perspectives: List[Perspective]):
        for p in perspectives:
            # Add scenario node
            self.knowledge_graph.add_node(
                id(p),
                probability=p.probability,
                impact=p.impact
            )
            
            # Connect to related scenarios
            for other in self.knowledge_graph.nodes():
                if other != id(p):
                    similarity = torch.cosine_similarity(
                        p.scenario.flatten(),
                        self.knowledge_graph.nodes[other]['scenario'].flatten(),
                        dim=0
                    )
                    if similarity > 0.7:
                        self.knowledge_graph.add_edge(
                            id(p),
                            other,
                            weight=similarity.item()
                        )

    async def find_high_impact_paths(self) -> List[List[Perspective]]:
        paths = []
        for source in self.knowledge_graph.nodes():
            for target in self.knowledge_graph.nodes():
                if source != target:
                    path = nx.shortest_path(
                        self.knowledge_graph,
                        source,
                        target,
                        weight='weight'
                    )
                    impact = sum(
                        self.knowledge_graph.nodes[node]['impact']
                        for node in path
                    )
                    if impact > 0.8:  # High impact threshold
                        paths.append(path)
        return paths

async def main():
    config = {
        'input_dim': 256,
        'hidden_dim': 512
    }
    
    engine = PerspectiveEngine(config)
    
    # Example usage
    data = torch.randn(100, 256)
    perspectives = await engine.generate_perspectives(data)
    
    print(f"Generated {len(perspectives)} perspectives")
    for p in perspectives:
        print(f"Probability: {p.probability:.2f}, Impact: {p.impact:.2f}")
        print(f"Timeline: {p.timeline[:100]}...")
        print("Supporting Evidence:", p.supporting_evidence)
        print("Counter Evidence:", p.counter_evidence)
        print()

if __name__ == "__main__":
    asyncio.run(main())
