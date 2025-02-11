#!/usr/bin/env python3
import boto3
import os
import json
from botocore.exceptions import ClientError
from typing import Dict, List
import asyncio
import yaml

class AWSDeployer:
    def __init__(self):
        self.session = boto3.Session(
            region_name='us-east-2',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        self.ecr = self.session.client('ecr')
        self.ecs = self.session.client('ecs')
        self.secretsmanager = self.session.client('secretsmanager')
        self.cloudformation = self.session.client('cloudformation')

    async def create_secret(self, secret_name: str, secret_value: Dict) -> None:
        try:
            self.secretsmanager.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value)
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                self.secretsmanager.update_secret(
                    SecretId=secret_name,
                    SecretString=json.dumps(secret_value)
                )

    async def deploy_infrastructure(self):
        # Store credentials in AWS Secrets Manager
        await self.create_secret('kaleidoscope/credentials', {
            'AWS_ACCOUNT_ID': '872515289083',
            'AWS_ACCESS_KEY_ID': os.environ.get('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': ' os.environ.get('WgS1dAUCdPnblbA+lZgyl8ww/0boB7zIJ2Z8Quh0')
        })

        # Create ECR repositories
        repositories = ['supernode', 'kaleidoscope', 'mirror', 'chatbot', 'monitoring']
        for repo in repositories:
            try:
                self.ecr.create_repository(
                    repositoryName=f'kaleidoscope-{repo}',
                    imageScanningConfiguration={'scanOnPush': True},
                    encryptionConfiguration={'encryptionType': 'AES256'}
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'RepositoryAlreadyExistsException':
                    raise

        # Deploy CloudFormation stack
        with open('infrastructure/cloudformation.yaml', 'r') as f:
            template = f.read()

        stack_name = 'kaleidoscope-infrastructure'
        try:
            self.cloudformation.create_stack(
                StackName=stack_name,
                TemplateBody=template,
                Capabilities=['CAPABILITY_NAMED_IAM'],
                Parameters=[
                    {
                        'ParameterKey': 'Environment',
                        'ParameterValue': 'production'
                    }
                ]
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'AlreadyExistsException':
                self.cloudformation.update_stack(
gt cevxd3z23                    StackName=stack_name,
                    TemplateBody=template,
                    Capabilities=['CAPABILITY_NAMED_IAM'],
                    Parameters=[
                        {
                            'ParameterKey': 'Environment',
                            'ParameterValue': 'production'
                        }
                    ]
                )

    async def deploy_services(self):
        cluster_name = 'kaleidoscope-cluster'
        service_configs = {
            'supernode': {'cpu': '4096', 'memory': '8192'},
            'kaleidoscope': {'cpu': '2048', 'memory': '4096'},
            'mirror': {'cpu': '2048', 'memory': '4096'},
            'chatbot': {'cpu': '1024', 'memory': '2048'},
            'monitoring': {'cpu': '512', 'memory': '1024'}
        }

        for service, config in service_configs.items():
            task_definition = {
                'family': f'kaleidoscope-{service}',
                'networkMode': 'awsvpc',
                'requiresCompatibilities': ['FARGATE'],
                'cpu': config['cpu'],
                'memory': config['memory'],
                'containerDefinitions': [{
                    'name': service,
                    'image': f'{os.environ.get("AWS_ACCOUNT_ID")}.dkr.ecr.us-east-2.amazonaws.com/kaleidoscope-{service}:latest',
                    'essential': True,
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': f'/ecs/kaleidoscope-{service}',
                            'awslogs-region': 'us-east-2',
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                }]
            }

            response = self.ecs.register_task_definition(**task_definition)
            task_definition_arn = response['taskDefinition']['taskDefinitionArn']

            try:
                self.ecs.create_service(
                    cluster=cluster_name,
                    serviceName=f'kaleidoscope-{service}',
                    taskDefinition=task_definition_arn,
                    desiredCount=1,
                    launchType='FARGATE',
                    networkConfiguration={
                        'awsvpcConfiguration': {
                            'subnets': ['subnet-xxxxxx'],  # Replace with actual subnet IDs
                            'securityGroups': ['sg-xxxxxx'],  # Replace with actual security group IDs
                            'assignPublicIp': 'ENABLED'
                        }
                    }
                )
            except ClientError as e:
                if e.response['Error']['Code'] == 'ServiceAlreadyExists':
                    self.ecs.update_service(
                        cluster=cluster_name,
                        service=f'kaleidoscope-{service}',
                        taskDefinition=task_definition_arn
                    )

if __name__ == '__main__':
    deployer = AWSDeployer()
    asyncio.run(deployer.deploy_infrastructure())
    asyncio.run(deployer.deploy_services())
