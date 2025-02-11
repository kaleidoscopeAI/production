from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
from pathlib import Path

@dataclass
class SecurityConfig:
    aws_access_key: str
    aws_secret_key: str
    aws_region: str
    encryption_key: Optional[str] = None

@dataclass
class MembraneConfig:
    input_dimension: int
    processing_capacity: int
    memory_threshold: float
    batch_size: int

@dataclass
class SuperNodeConfig:
    initial_nodes: int
    dimension: int
    memory_per_node: int
    min_resonance: float

@dataclass
class EngineConfig:
    input_dimension: int
    hidden_dimension: int
    num_layers: int
    batch_size: int

@dataclass
class PipelineConfig:
    s3_bucket: str
    queue_url: str
    max_batch_size: int
    processing_threads: int

@dataclass
class ChatbotConfig:
    model_name: str
    max_length: int
    temperature: float
    top_p: float

@dataclass
class MonitoringConfig:
    prometheus_port: int
    grafana_port: int
    log_level: str
    metrics_interval: int

@dataclass
class VisualizationConfig:
    host: str
    port: int
    update_interval: float

@dataclass
class SystemConfig:
    security: SecurityConfig
    membrane: MembraneConfig
    supernode: SuperNodeConfig
    kaleidoscope: EngineConfig
    mirror: EngineConfig
    pipeline: PipelineConfig
    chatbot: ChatbotConfig
    monitoring: MonitoringConfig
    visualization: VisualizationConfig

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SystemConfig':
        return cls(
            security=SecurityConfig(**config_dict['security']),
            membrane=MembraneConfig(**config_dict['membrane']),
            supernode=SuperNodeConfig(**config_dict['supernode']),
            kaleidoscope=EngineConfig(**config_dict['kaleidoscope']),
            mirror=EngineConfig(**config_dict['mirror']),
            pipeline=PipelineConfig(**config_dict['pipeline']),
            chatbot=ChatbotConfig(**config_dict['chatbot']),
            monitoring=MonitoringConfig(**config_dict['monitoring']),
            visualization=VisualizationConfig(**config_dict['visualization'])
        )

# Create default configuration
default_config = {
    'security': {
        'aws_access_key': 'AKIA4WJPWX757RLAGXU7',
        'aws_secret_key': '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',
        'aws_region': 'us-east-2'
    },
    'membrane': {
        'input_dimension': 512,
        'processing_capacity': 1000,
        'memory_threshold': 0.85,
        'batch_size': 64
    },
    'supernode': {
        'initial_nodes': 4,
        'dimension': 512,
        'memory_per_node': 1024,
        'min_resonance': 0.7
    },
    'kaleidoscope': {
        'input_dimension': 512,
        'hidden_dimension': 1024,
        'num_layers': 6,
        'batch_size': 32
    },
    'mirror': {
        'input_dimension': 512,
        'hidden_dimension': 1024,
        'num_layers': 4,
        'batch_size': 32
    },
    'pipeline': {
        's3_bucket': 'kaleidoscope-data',
        'queue_url': 'https://sqs.us-east-2.amazonaws.com/YOUR_ACCOUNT_ID/kaleidoscope-queue',
        'max_batch_size': 128,
        'processing_threads': 4
    },
    'chatbot': {
        'model_name': 'meta-llama/Llama-2-7b-chat-hf',
        'max_length': 2048,
        'temperature': 0.7,
        'top_p': 0.9
    },
    'monitoring': {
        'prometheus_port': 9090,
        'grafana_port': 3000,
        'log_level': 'INFO',
        'metrics_interval': 60
    },
    'visualization': {
        'host': '0.0.0.0',
        'port': 8000,
        'update_interval': 1.0
    }
}

# Save default configuration
config_dir = Path(__file__).parent.parent / 'config'
config_dir.mkdir(exist_ok=True)

with open(config_dir / 'base.yml', 'w') as f:
    yaml.dump(default_config, f, default_flow_style=False)
