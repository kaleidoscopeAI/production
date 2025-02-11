#!/usr/bin/env python3
import boto3
import asyncio
import yaml
import os
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
    name: str
    region: str
    domain: str
    instance_type: str
    min_nodes: int
    max_nodes: int
    vpc_cidr: str

class MultiEnvironmentDeployer:
    def __init__(self):
        self.environments = {
            'ote': EnvironmentConfig(
                name='ote',
                region='us-east-2',
                domain='ote.artificialthinker.com',
                instance_type='t3.large',
                min_nodes=1,
                max_nodes=3,
                vpc_cidr='10.0.0.0/16'
            ),
            'production': EnvironmentConfig(
                name='production',
                region='us-east-2',
                domain='artificialthinker.com',
                instance_type='c5.2xlarge',
                min_nodes=3,
                max_nodes=10,
                vpc_cidr='172.16.0.0/16'
            )
        }
        
        self.aws_clients = {}
        for env in self.environments.values():
            self.aws_clients[env.name] = {
                'ecs': boto3.client('ecs', region_name=env.region),
                'ec2': boto3.client('ec2', region_name=env.region),
                'rds': boto3.client('rds', region_name=env.region),
                'elasticache': boto3.client('elasticache', region_name=env.region),
                'route53': boto3.client('route53', region_name=env.region)
            }

    async def deploy_environment(self, env_name: str):
        env = self.environments[env_name]
        clients = self.aws_clients[env_name]

        # Create VPC and networking
        vpc_id = await self._create_vpc(env, clients['ec2'])
        
        # Deploy database
        db_instance = await self._deploy_database(env, clients['rds'])
        
        # Deploy Redis cluster
        redis_cluster = await self._deploy_redis(env, clients['elasticache'])
        
        # Deploy ECS cluster
        cluster_arn = await self._deploy_ecs_cluster(env, clients['ecs'])
        
        # Deploy services
        await self._deploy_services(env, clients['ecs'], vpc_id)
        
        # Setup DNS
        await self._configure_dns(env, clients['route53'])

        return {
            'vpc_id': vpc_id,
            'cluster_arn': cluster_arn,
            'db_endpoint': db_instance['Endpoint'],
            'redis_endpoint': redis_cluster['CacheNodes'][0]['Endpoint']
        }

    async def _create_vpc(self, env: EnvironmentConfig, ec2):
        vpc = ec2.create_vpc(CidrBlock=env.vpc_cidr)
        waiter = ec2.get_waiter('vpc_available')
        waiter.wait(VpcIds=[vpc['Vpc']['VpcId']])
        
        # Enable DNS hostnames
        ec2.modify_vpc_attribute(
            VpcId=vpc['Vpc']['VpcId'],
            EnableDnsHostnames={'Value': True}
        )
        
        return vpc['Vpc']['VpcId']

    async def _deploy_database(self, env: EnvironmentConfig, rds):
        instance = rds.create_db_instance(
            DBInstanceIdentifier=f'kaleidoscope-{env.name}',
            DBInstanceClass='db.r5.xlarge' if env.name == 'production' else 'db.t3.large',
            Engine='postgres',
            EngineVersion='13.7',
            MultiAZ=env.name == 'production',
            StorageType='gp3',
            AllocatedStorage=100 if env.name == 'production' else 20,
            MasterUsername='kaleidoscope',
            MasterUserPassword=os.environ.get('DB_PASSWORD', 'defaultpass'),
            BackupRetentionPeriod=7 if env.name == 'production' else 1,
            VpcSecurityGroupIds=[self._create_db_security_group(env)],
            Tags=[{'Key': 'Environment', 'Value': env.name}]
        )
        
        waiter = rds.get_waiter('db_instance_available')
        waiter.wait(DBInstanceIdentifier=instance['DBInstance']['DBInstanceIdentifier'])
        
        return instance['DBInstance']

    async def _deploy_redis(self, env: EnvironmentConfig, elasticache):
        cluster = elasticache.create_cache_cluster(
            CacheClusterId=f'kaleidoscope-{env.name}',
            Engine='redis',
            CacheNodeType='cache.r5.xlarge' if env.name == 'production' else 'cache.t3.medium',
            NumCacheNodes=3 if env.name == 'production' else 1,
            VpcSecurityGroupIds=[self._create_redis_security_group(env)],
            Tags=[{'Key': 'Environment', 'Value': env.name}]
        )
        
        waiter = elasticache.get_waiter('cache_cluster_available')
        waiter.wait(CacheClusterId=cluster['CacheCluster']['CacheClusterId'])
        
        return cluster['CacheCluster']

    async def _deploy_ecs_cluster(self, env: EnvironmentConfig, ecs):
        cluster = ecs.create_cluster(
            clusterName=f'kaleidoscope-{env.name}',
            capacityProviders=['FARGATE', 'FARGATE_SPOT'],
            defaultCapacityProviderStrategy=[
                {
                    'capacityProvider': 'FARGATE',
                    'weight': 1,
                    'base': 1
                },
                {
                    'capacityProvider': 'FARGATE_SPOT',
                    'weight': 4
                }
            ],
            tags=[{'key': 'Environment', 'value': env.name}]
        )
        
        return cluster['cluster']['clusterArn']

    async def _deploy_services(self, env: EnvironmentConfig, ecs, vpc_id):
        services = ['supernode', 'kaleidoscope', 'mirror', 'chatbot', 'frontend']
        
        for service in services:
            task_def = self._create_task_definition(env, service)
            ecs.register_task_definition(**task_def)
            
            ecs.create_service(
                cluster=f'kaleidoscope-{env.name}',
                serviceName=f'{service}-{env.name}',
                taskDefinition=f'{service}-{env.name}',
                desiredCount=3 if env.name == 'production' else 1,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': self._get_subnets(vpc_id),
                        'securityGroups': [self._create_service_security_group(env, service)],
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )

    def _create_task_definition(self, env: EnvironmentConfig, service: str) -> Dict:
        return {
            'family': f'{service}-{env.name}',
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '2048' if env.name == 'production' else '512',
            'memory': '4096' if env.name == 'production' else '1024',
            'containerDefinitions': [{
                'name': service,
                'image': f'{os.environ["AWS_ACCOUNT_ID"]}.dkr.ecr.{env.region}.amazonaws.com/kaleidoscope-{service}:latest',
                'environment': [
                    {'name': 'ENVIRONMENT', 'value': env.name},
                    {'name': 'REGION', 'value': env.region}
                ],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': f'/ecs/kaleidoscope/{env.name}/{service}',
                        'awslogs-region': env.region,
                        'awslogs-stream-prefix': 'ecs'
                    }
                }
            }]
        }

if __name__ == "__main__":
    deployer = MultiEnvironmentDeployer()
    
    # Deploy OTE first
    asyncio.run(deployer.deploy_environment('ote'))
    
    # Run tests against OTE
    # ... testing code here ...
    
    # If tests pass, deploy to production
    asyncio.run(deployer.deploy_environment('production'))