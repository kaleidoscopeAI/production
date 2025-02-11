import boto3
import json
import logging
import asyncio
from botocore.exceptions import ClientError

class AWSIntegration:
    """Handles AWS Lambda, S3, SQS, and DynamoDB integrations for AI system."""

    def __init__(self, region="us-east-1"):
        self.s3 = boto3.client("s3", region_name=region)
        self.sqs = boto3.client("sqs", region_name=region)
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.lambda_client = boto3.client("lambda", region_name=region)
        self.logger = logging.getLogger("AWSIntegration")

    def upload_to_s3(self, bucket_name, file_name, data):
        """Uploads AI data to AWS S3 for storage & retrieval."""
        try:
            self.s3.put_object(Bucket=bucket_name, Key=file_name, Body=json.dumps(data))
            self.logger.info(f"Uploaded {file_name} to S3 bucket {bucket_name}.")
        except ClientError as e:
            self.logger.error(f"S3 Upload Error: {e}")

    def invoke_lambda(self, function_name, payload):
        """Triggers AWS Lambda function for serverless AI processing."""
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                Payload=json.dumps(payload)
            )
            return json.loads(response["Payload"].read())
        except ClientError as e:
            self.logger.error(f"Lambda Invocation Error: {e}")
            return None

    async def send_message(self, queue_url, message):
        """Sends AI processing tasks to AWS SQS queue."""
        try:
            self.sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message)
            )
            self.logger.info(f"Sent message to SQS queue {queue_url}.")
        except ClientError as e:
            self.logger.error(f"SQS Message Error: {e}")

    async def receive_messages(self, queue_url, max_messages=10):
        """Receives processed AI insights from AWS SQS queue."""
        try:
            response = self.sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=5
            )
            return [json.loads(msg["Body"]) for msg in response.get("Messages", [])]
        except ClientError as e:
            self.logger.error(f"SQS Receive Error: {e}")
            return []

    def store_node_state(self, table_name, node_id, state_data):
        """Stores AI node state in AWS DynamoDB for persistence."""
        try:
            table = self.dynamodb.Table(table_name)
            table.put_item(Item={"node_id": node_id, "state": json.dumps(state_data)})
            self.logger.info(f"Stored node state in DynamoDB table {table_name}.")
        except ClientError as e:
            self.logger.error(f"DynamoDB Store Error: {e}")

if __name__ == "__main__":
    aws = AWSIntegration(region="us-east-1")
    
    # Example: Upload AI insight to S3
    aws.upload_to_s3(bucket_name="kaleidoscope-ai-insights", file_name="insight.json", data={"insight": "AI-generated patterns"})

    # Example: Invoke AI Lambda function
    response = aws.invoke_lambda("KaleidoscopeLambdaFunction", {"task": "process_insight"})
    print(f"Lambda Response: {response}")

    # Example: Store AI Node state in DynamoDB
    aws.store_node_state("NodeStatusTable", "node_42", {"status": "active", "tasks_completed": 10})

