#!/usr/bin/env python3
import asyncio
import aiodns
import OpenSSL
import dns.resolver
import aiohttp
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import boto3
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import logging
from dataclasses import dataclass

@dataclass
class DomainConfig:
    domain: str
    subdomains: List[str]
    cert_email: str
    dns_servers: List[str] = None
    check_interval: int = 300
    propagation_timeout: int = 600
    cert_renewal_days: int = 30

class DomainAutomation:
    def __init__(self, config: DomainConfig):
        self.config = config
        self.session = aiohttp.ClientSession()
        self.resolver = aiodns.DNSResolver()
        self.route53 = boto3.client('route53')
        self.acm = boto3.client('acm')
        self.cloudfront = boto3.client('cloudfront')
        self.logger = logging.getLogger('DomainAutomation')
        
        if not self.config.dns_servers:
            self.config.dns_servers = ['8.8.8.8', '1.1.1.1', '208.67.222.222']

    async def check_dns_propagation(self, record_type: str, name: str, expected_value: str) -> bool:
        tasks = []
        for server in self.config.dns_servers:
            self.resolver.nameservers = [server]
            tasks.append(self.resolver.query(name, record_type))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            propagated = all(
                any(str(answer.host) == expected_value for answer in result)
                for result in results if not isinstance(result, Exception)
            )
            return propagated
        except Exception as e:
            self.logger.error(f"DNS check error: {e}")
            return False

    async def wait_for_propagation(self, record_type: str, name: str, value: str) -> bool:
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < self.config.propagation_timeout:
            if await self.check_dns_propagation(record_type, name, value):
                return True
            await asyncio.sleep(30)
        return False

    async def provision_certificates(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        
        # Generate CSR
        builder = x509.CertificateSigningRequestBuilder()
        builder = builder.subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, self.config.domain),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Kaleidoscope AI"),
        ]))
        
        # Add SANs
        builder = builder.add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(self.config.domain),
                *[x509.DNSName(f"{sub}.{self.config.domain}") for sub in self.config.subdomains]
            ]),
            critical=False,
        )
        
        csr = builder.sign(private_key, hashes.SHA256())
        
        # Request certificate using ACME
        certbot_cmd = [
            "certbot", "certonly",
            "--non-interactive",
            "--agree-tos",
            "--email", self.config.cert_email,
            "--preferred-challenges", "dns",
            "--authenticator", "dns-route53",
            "--domain", self.config.domain,
            *[f"--domain", f"*.{self.config.domain}"],
            "--cert-name", self.config.domain,
            "--keep-until-expiring",
            "--rsa-key-size", "4096",
            "--must-staple"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *certbot_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Certificate provisioning failed: {stderr.decode()}")
        
        # Import certificate to ACM
        cert_path = f"/etc/letsencrypt/live/{self.config.domain}/fullchain.pem"
        key_path = f"/etc/letsencrypt/live/{self.config.domain}/privkey.pem"
        
        with open(cert_path, 'rb') as cert_file, open(key_path, 'rb') as key_file:
            cert_response = self.acm.import_certificate(
                Certificate=cert_file.read(),
                PrivateKey=key_file.read(),
                Tags=[{'Key': 'Domain', 'Value': self.config.domain}]
            )
        
        return cert_response['CertificateArn']

    async def setup_cloudfront(self, cert_arn: str):
        distribution_config = {
            'CallerReference': str(datetime.now().timestamp()),
            'Origins': {
                'Quantity': 1,
                'Items': [{
                    'Id': 'ELB-Origin',
                    'DomainName': f"{self.config.domain}",
                    'CustomOriginConfig': {
                        'HTTPPort': 80,
                        'HTTPSPort': 443,
                        'OriginProtocolPolicy': 'https-only',
                        'OriginSslProtocols': {'Quantity': 1, 'Items': ['TLSv1.2']}
                    }
                }]
            },
            'DefaultCacheBehavior': {
                'TargetOriginId': 'ELB-Origin',
                'ViewerProtocolPolicy': 'redirect-to-https',
                'MinTTL': 0,
                'ForwardedValues': {
                    'QueryString': True,
                    'Cookies': {'Forward': 'all'}
                }
            },
            'ViewerCertificate': {
                'ACMCertificateArn': cert_arn,
                'SSLSupportMethod': 'sni-only',
                'MinimumProtocolVersion': 'TLSv1.2_2021'
            },
            'Enabled': True
        }
        
        response = self.cloudfront.create_distribution(
            DistributionConfig=distribution_config
        )
        return response['Distribution']['Id']

    async def monitor_certificates(self):
        while True:
            certs = self.acm.list_certificates()['CertificateSummaryList']
            for cert in certs:
                cert_detail = self.acm.describe_certificate(CertificateArn=cert['CertificateArn'])
                not_after = cert_detail['Certificate']['NotAfter']
                
                if (not_after - datetime.now(not_after.tzinfo)).days <= self.config.cert_renewal_days:
                    self.logger.info(f"Renewing certificate for {self.config.domain}")
                    await self.provision_certificates()
            
            await asyncio.sleep(86400)  # Check daily

    async def setup_health_checks(self):
        response = self.route53.create_health_check(
            CallerReference=str(datetime.now().timestamp()),
            HealthCheckConfig={
                'IPAddress': await self._resolve_domain(),
                'Port': 443,
                'Type': 'HTTPS',
                'ResourcePath': '/health',
                'FullyQualifiedDomainName': self.config.domain,
                'RequestInterval': 30,
                'FailureThreshold': 3,
                'MeasureLatency': True,
                'EnableSNI': True
            }
        )
        return response['HealthCheck']['Id']

    async def _resolve_domain(self) -> str:
        resolver = dns.resolver.Resolver()
        answers = resolver.query(self.config.domain, 'A')
        return str(answers[0])

async def main():
    config = DomainConfig(
        domain="artificialthinker.com",
        subdomains=["www", "api", "admin"],
        cert_email="jmgraham1000@gmail.com"
    )
    
    automation = DomainAutomation(config)
    
    # Run all tasks concurrently
    await asyncio.gather(
        automation.monitor_certificates(),
        automation.setup_health_checks()
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())