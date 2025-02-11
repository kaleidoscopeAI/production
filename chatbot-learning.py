import os
import boto3
import base64
import logging
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurePipeline:
    """Handles AI data security, encryption, and AWS-integrated storage protection."""

    def __init__(self, aws_region="us-east-1"):
        self.kms_client = boto3.client("kms", region_name=aws_region)
        self.s3 = boto3.client("s3", region_name=aws_region)
        self.logger = logging.getLogger("SecurePipeline")
        self.private_key, self.public_key = self.generate_key_pair()
    
    def generate_key_pair(self):
        """Generates an RSA key pair for AI data encryption."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def encrypt_data(self, data: str) -> bytes:
        """Encrypts AI data using RSA public key encryption."""
        encrypted_data = self.public_key.encrypt(
            data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted_data)

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypts AI data using RSA private key."""
        decrypted_data = self.private_key.decrypt(
            base64.b64decode(encrypted_data),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data.decode()

    def store_secure_data(self, bucket: str, key: str, data: str):
        """Encrypts and stores AI data securely in an AWS S3 bucket."""
        encrypted_data = self.encrypt_data(data)
        self.s3.put_object(Bucket=bucket, Key=key, Body=encrypted_data)
        self.logger.info(f"Securely stored data at: s3://{bucket}/{key}")

    def retrieve_secure_data(self, bucket: str, key: str) -> str:
        """Retrieves and decrypts AI data from AWS S3."""
        response = self.s3.get_object(Bucket=bucket, Key=key)
        encrypted_data = response["Body"].read()
        return self.decrypt_data(encrypted_data)

if __name__ == "__main__":
    pipeline = SecurePipeline(aws_region="us-east-1")
    
    # Example: Securely storing AI insights
    insight_data = "AI insights generated from SuperNodes"
    pipeline.store_secure_data(bucket="secure-ai-storage", key="insights/encrypted_data.txt", data=insight_data)

    # Example: Retrieving and decrypting data
    retrieved_data = pipeline.retrieve_secure_data(bucket="secure-ai-storage", key="insights/encrypted_data.txt")
    print(f"Decrypted AI Insight: {retrieved_data}")

