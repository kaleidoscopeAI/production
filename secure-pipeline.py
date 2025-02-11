import asyncio
import torch
from typing import Dict, List, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import msgpack
import zmq
import zmq.asyncio
from dataclasses import dataclass
import logging

@dataclass
class SecureChannel:
    id: str
    pub_key: rsa.RSAPublicKey
    priv_key: rsa.RSAPrivateKey
    session_key: bytes
    fernet: Fernet

class SecurePipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.channels: Dict[str, SecureChannel] = {}
        self.context = zmq.asyncio.Context()
        self.logger = logging.getLogger("SecurePipeline")
        
    async def create_channel(self, channel_id: str) -> SecureChannel:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'secure_pipeline_salt',
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(channel_id.encode()))
        fernet = Fernet(key)
        
        channel = SecureChannel(
            id=channel_id,
            pub_key=public_key,
            priv_key=private_key,
            session_key=key,
            fernet=fernet
        )
        
        self.channels[channel_id] = channel
        return channel

class SecureDataPipeline:
    def __init__(self, pipeline: SecurePipeline):
        self.pipeline = pipeline
        self.socket = None
        
    async def connect(self, endpoint: str):
        self.socket = self.pipeline.context.socket(zmq.DEALER)
        self.socket.connect(endpoint)
        
    async def send_data(self, channel_id: str, data: Dict):
        if channel_id not in self.pipeline.channels:
            await self.pipeline.create_channel(channel_id)
            
        channel = self.pipeline.channels[channel_id]
        
        # Serialize and encrypt data
        serialized = msgpack.packb(data)
        encrypted = channel.fernet.encrypt(serialized)
        
        # Sign the encrypted data
        signature = channel.priv_key.sign(
            encrypted,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        message = {
            'channel_id': channel_id,
            'data': encrypted,
            'signature': signature
        }
        
        await self.socket.send_multipart([
            channel_id.encode(),
            msgpack.packb(message)
        ])
        
    async def receive_data(self) -> Optional[Dict]:
        if not self.socket:
            return None
            
        channel_id, message = await self.socket.recv_multipart()
        channel_id = channel_id.decode()
        
        if channel_id not in self.pipeline.channels:
            return None
            
        channel = self.pipeline.channels[channel_id]
        message = msgpack.unpackb(message)
        
        # Verify signature
        try:
            channel.pub_key.verify(
                message['signature'],
                message['data'],
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except:
            self.logger.error(f"Invalid signature for channel {channel_id}")
            return None
            
        # Decrypt and deserialize
        decrypted = channel.fernet.decrypt(message['data'])
        return msgpack.unpackb(decrypted)

class SecureNodeCommunication:
    def __init__(self, node_id: str, pipeline: SecurePipeline):
        self.node_id = node_id
        self.pipeline = pipeline
        self.peers: Dict[str, SecureChannel] = {}
        
    async def connect_to_peer(self, peer_id: str, endpoint: str):
        channel = await self.pipeline.create_channel(f"{self.node_id}_{peer_id}")
        self.peers[peer_id] = channel
        
        socket = self.pipeline.context.socket(zmq.DEALER)
        socket.connect(endpoint)
        
        return socket
        
    async def send_to_peer(self, peer_id: str, data: Dict):
        if peer_id not in self.peers:
            self.logger.error(f"Unknown peer {peer_id}")
            return
            
        channel = self.peers[peer_id]
        
        # Encrypt data
        serialized = msgpack.packb(data)
        encrypted = channel.fernet.encrypt(serialized)
        
        # Create signature
        signature = channel.priv_key.sign(
            encrypted,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        message = {
            'sender': self.node_id,
            'data': encrypted,
            'signature': signature
        }
        
        return msgpack.packb(message)

class SecureDataTransformer:
    def __init__(self, pipeline: SecurePipeline):
        self.pipeline = pipeline
        
    async def transform_tensor(self, data: torch.Tensor, channel_id: str) -> bytes:
        if channel_id not in self.pipeline.channels:
            await self.pipeline.create_channel(channel_id)
            
        channel = self.pipeline.channels[channel_id]
        
        # Serialize tensor
        buffer = io.BytesIO()
        torch.save(data, buffer)
        serialized = buffer.getvalue()
        
        # Encrypt
        encrypted = channel.fernet.encrypt(serialized)
        
        # Sign
        signature = channel.priv_key.sign(
            encrypted,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return msgpack.packb({
            'data': encrypted,
            'signature': signature
        })
        
    async def restore_tensor(self, encrypted_data: bytes, channel_id: str) -> Optional[torch.Tensor]:
        if channel_id not in self.pipeline.channels:
            return None
            
        channel = self.pipeline.channels[channel_id]
        message = msgpack.unpackb(encrypted_data)
        
        # Verify signature
        try:
            channel.pub_key.verify(
                message['signature'],
                message['data'],
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except:
            return None
            
        # Decrypt and deserialize
        decrypted = channel.fernet.decrypt(message['data'])
        buffer = io.BytesIO(decrypted)
        return torch.load(buffer)

async def main():
    config = {
        'encryption_key_size': 2048,
        'kdf_iterations': 100000
    }
    
    pipeline = SecurePipeline(config)
    data_pipeline = SecureDataPipeline(pipeline)
    
    # Example usage
    await data_pipeline.connect("tcp://localhost:5555")
    
    data = {
        'tensor': torch.randn(100, 100),
        'metadata': {'type': 'test'}
    }
    
    await data_pipeline.send_data("channel1", data)
    received = await data_pipeline.receive_data()
    
    if received:
        print("Data received successfully")

if __name__ == "__main__":
    asyncio.run(main())
