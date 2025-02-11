import numpy as np
import torch
import boto3
from pathlib import Path
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import yaml
from scipy.optimize import minimize

@dataclass
class NodeConfig:
    memory_threshold: float
    input_dim: int
    processing_capacity: float
    kaleidoscope_queue_url: str
    mirror_queue_url: str

class MembraneCore:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sqs = boto3.client('sqs')
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.nodes = []
        self.insight_threshold = self.config['insight_threshold']
        self.logger = logging.getLogger("Membrane")
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("membrane.log"),
                logging.StreamHandler()
            ]
        )

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    async def process_input_data(self, data_path: str) -> Tuple[int, float]:
        """Calculate optimal number of nodes and memory threshold"""
        data_size = await self._get_data_size(data_path)
        
        def objective(x):
            nodes, memory = x
            insights = data_size / (nodes * memory)
            return abs(insights - self.insight_threshold)

        constraints = [
            {'type': 'ineq', 'fun': lambda x: self.config['max_nodes'] - x[0]},
            {'type': 'ineq', 'fun': lambda x: x[0] - self.config['min_nodes']},
            {'type': 'ineq', 'fun': lambda x: self.config['max_memory'] - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[1] - self.config['min_memory']}
        ]

        result = minimize(
            objective,
            x0=[self.config['min_nodes'], self.config['min_memory']],
            constraints=constraints,
            method='SLSQP'
        )

        num_nodes = int(np.ceil(result.x[0]))
        memory_threshold = result.x[1]

        self.logger.info(f"Calculated {num_nodes} nodes with {memory_threshold:.2f} memory threshold")
        return num_nodes, memory_threshold

    async def initialize_nodes(self, num_nodes: int, memory_threshold: float):
        """Initialize nodes with calculated parameters"""
        node_configs = []
        for _ in range(num_nodes):
            config = NodeConfig(
                memory_threshold=memory_threshold,
                input_dim=self.config['input_dimension'],
                processing_capacity=self.config['processing_capacity_per_node'],
                kaleidoscope_queue_url=self.config['kaleidoscope_queue_url'],
                mirror_queue_url=self.config['mirror_queue_url']
            )
            node_configs.append(config)

        # Initialize nodes in parallel
        self.nodes = await asyncio.gather(*[
            self._create_node(config) for config in node_configs
        ])

    async def _create_node(self, config: NodeConfig) -> 'Node':
        """Create and initialize a new node"""
        from node import Node  # Imported here to avoid circular dependency
        node = Node(config)
        await node.initialize()
        return node

    async def distribute_data(self, data_path: str):
        """Distribute data chunks to nodes"""
        chunk_size = self._calculate_chunk_size()
        async for chunk in self._stream_data(data_path, chunk_size):
            node = self._select_optimal_node()
            await node.process_data_chunk(chunk)

    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on node capacity"""
        avg_memory = np.mean([node.get_available_memory() for node in self.nodes])
        return int(avg_memory * 0.1)  # Use 10% of average available memory

    async def _stream_data(self, data_path: str, chunk_size: int):
        """Stream data in chunks to avoid memory overload"""
        file_content = await self._read_file_content(data_path)
        data = torch.load(file_content) if torch.is_tensor(file_content) else torch.tensor(file_content)
        
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    async def _read_file_content(self, data_path: str):
        """Read file content using appropriate method based on file type"""
        try:
            file_content = await window.fs.readFile(data_path, {'encoding': 'utf8'})
            return file_content
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            raise

    def _select_optimal_node(self) -> 'Node':
        """Select optimal node for processing next chunk"""
        return min(
            self.nodes,
            key=lambda node: (
                node.get_current_load(),
                -node.get_available_memory()
            )
        )

    async def monitor_nodes(self):
        """Monitor node health and performance"""
        while True:
            for node in self.nodes:
                metrics = await node.get_metrics()
                if metrics['health_status'] != 'healthy':
                    await self._handle_unhealthy_node(node)
                if metrics['load'] > self.config['max_load_threshold']:
                    await self._rebalance_nodes()
            await asyncio.sleep(self.config['monitoring_interval'])

    async def _handle_unhealthy_node(self, node: 'Node'):
        """Handle unhealthy node by redistributing its data"""
        healthy_nodes = [n for n in self.nodes if n != node and n.get_metrics()['health_status'] == 'healthy']
        if not healthy_nodes:
            self.logger.error("No healthy nodes available for redistribution")
            return

        data = await node.get_current_data()
        chunks = torch.chunk(data, len(healthy_nodes))
        
        await asyncio.gather(*[
            healthy_node.process_data_chunk(chunk)
            for healthy_node, chunk in zip(healthy_nodes, chunks)
        ])

        await node.reset()

    async def _rebalance_nodes(self):
        """Rebalance data across nodes"""
        total_load = sum(node.get_current_load() for node in self.nodes)
        target_load = total_load / len(self.nodes)

        overloaded = [n for n in self.nodes if n.get_current_load() > target_load * 1.1]
        underloaded = [n for n in self.nodes if n.get_current_load() < target_load * 0.9]

        for over_node in overloaded:
            if not underloaded:
                break
                
            excess_data = await over_node.get_excess_data(target_load)
            chunks = torch.chunk(excess_data, len(underloaded))
            
            await asyncio.gather(*[
                under_node.process_data_chunk(chunk)
                for under_node, chunk in zip(underloaded, chunks)
            ])

    async def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'total_nodes': len(self.nodes),
            'healthy_nodes': sum(1 for node in self.nodes if node.get_metrics()['health_status'] == 'healthy'),
            'total_processed_data': sum(node.get_metrics()['processed_data_size'] for node in self.nodes),
            'average_load': np.mean([node.get_current_load() for node in self.nodes]),
            'timestamp': time.time()
        }
