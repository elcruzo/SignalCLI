#!/usr/bin/env python3
"""Production deployment script for SignalCLI."""

import asyncio
import subprocess
import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages SignalCLI deployment process."""

    def __init__(self, config_path: Path, environment: str = "production"):
        self.config_path = config_path
        self.environment = environment
        self.project_root = config_path.parent.parent
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            
            env_config = config.get(self.environment, {})
            base_config = config.get("base", {})
            
            # Merge base and environment-specific config
            merged_config = {**base_config, **env_config}
            
            logger.info(f"Loaded config for environment: {self.environment}")
            return merged_config
        
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

    def run_command(self, command: str, check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run shell command."""
        logger.info(f"Running: {command}")
        
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            check=check,
            cwd=cwd or self.project_root
        )
        
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        if result.stderr and check:
            logger.error(f"STDERR: {result.stderr}")
        
        return result

    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites."""
        logger.info("Checking deployment prerequisites...")
        
        checks = []
        
        # Check Docker
        try:
            self.run_command("docker --version")
            checks.append("✓ Docker installed")
        except subprocess.CalledProcessError:
            checks.append("✗ Docker not installed")
            return False

        # Check Docker Compose
        try:
            self.run_command("docker-compose --version")
            checks.append("✓ Docker Compose installed")
        except subprocess.CalledProcessError:
            checks.append("✗ Docker Compose not installed")
            return False

        # Check Python dependencies
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            checks.append("✓ Requirements file found")
        else:
            checks.append("✗ Requirements file missing")
            return False

        # Check config files
        docker_compose_file = self.project_root / "docker" / "docker-compose.prod.yml"
        if docker_compose_file.exists():
            checks.append("✓ Docker compose production config found")
        else:
            checks.append("✗ Docker compose production config missing")
            return False

        for check in checks:
            logger.info(check)

        return True

    def setup_environment(self) -> None:
        """Setup deployment environment."""
        logger.info("Setting up deployment environment...")
        
        # Create necessary directories
        directories = [
            "logs",
            "data/vector_store", 
            "data/cache",
            "data/models",
            "monitoring/prometheus",
            "monitoring/grafana",
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

        # Set permissions
        self.run_command("chmod -R 755 logs data monitoring")

        # Generate environment file
        self._generate_env_file()

    def _generate_env_file(self) -> None:
        """Generate .env file from config."""
        env_file = self.project_root / ".env"
        
        env_vars = {
            "ENVIRONMENT": self.environment,
            "LOG_LEVEL": self.config.get("log_level", "INFO"),
            "API_HOST": self.config.get("api", {}).get("host", "0.0.0.0"),
            "API_PORT": str(self.config.get("api", {}).get("port", 8000)),
            "MCP_PORT": str(self.config.get("mcp", {}).get("port", 8001)),
            "WEAVIATE_HOST": self.config.get("vector_store", {}).get("host", "weaviate"),
            "WEAVIATE_PORT": str(self.config.get("vector_store", {}).get("port", 8080)),
            "REDIS_HOST": self.config.get("cache", {}).get("host", "redis"),
            "REDIS_PORT": str(self.config.get("cache", {}).get("port", 6379)),
            "PROMETHEUS_PORT": str(self.config.get("monitoring", {}).get("prometheus_port", 9090)),
            "GRAFANA_PORT": str(self.config.get("monitoring", {}).get("grafana_port", 3000)),
        }

        # Add LLM configuration
        llm_config = self.config.get("llm", {})
        if llm_config.get("openai_api_key"):
            env_vars["OPENAI_API_KEY"] = llm_config["openai_api_key"]
        if llm_config.get("model_path"):
            env_vars["LLM_MODEL_PATH"] = llm_config["model_path"]

        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        logger.info("Generated .env file")

    def build_images(self) -> None:
        """Build Docker images."""
        logger.info("Building Docker images...")
        
        # Build main application image
        self.run_command("docker-compose -f docker/docker-compose.prod.yml build")

    def run_tests(self) -> None:
        """Run test suite."""
        logger.info("Running test suite...")
        
        # Run pytest in container
        test_command = (
            "docker-compose -f docker/docker-compose.test.yml run --rm "
            "signalcli-test pytest tests/ -v --cov=src --cov-report=term-missing"
        )
        
        try:
            self.run_command(test_command)
            logger.info("✓ All tests passed")
        except subprocess.CalledProcessError:
            logger.error("✗ Tests failed")
            if not self.config.get("deploy_on_test_failure", False):
                sys.exit(1)

    def deploy_services(self) -> None:
        """Deploy all services."""
        logger.info("Deploying services...")
        
        # Pull latest images for dependencies
        self.run_command("docker-compose -f docker/docker-compose.prod.yml pull")
        
        # Start services
        self.run_command("docker-compose -f docker/docker-compose.prod.yml up -d")
        
        # Wait for services to be ready
        self._wait_for_services()

    def _wait_for_services(self) -> None:
        """Wait for services to be ready."""
        import time
        import requests
        
        services = [
            {
                "name": "SignalCLI API",
                "url": f"http://localhost:{self.config.get('api', {}).get('port', 8000)}/health",
                "timeout": 60
            },
            {
                "name": "MCP Server", 
                "url": f"http://localhost:{self.config.get('mcp', {}).get('port', 8001)}/health",
                "timeout": 60
            },
            {
                "name": "Weaviate",
                "url": f"http://localhost:{self.config.get('vector_store', {}).get('port', 8080)}/v1/.well-known/ready",
                "timeout": 120
            }
        ]
        
        for service in services:
            logger.info(f"Waiting for {service['name']} to be ready...")
            
            start_time = time.time()
            while time.time() - start_time < service["timeout"]:
                try:
                    response = requests.get(service["url"], timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✓ {service['name']} is ready")
                        break
                except requests.RequestException:
                    pass
                
                time.sleep(5)
            else:
                logger.warning(f"⚠ {service['name']} may not be ready")

    def setup_monitoring(self) -> None:
        """Setup monitoring and alerting."""
        logger.info("Setting up monitoring...")
        
        # Copy monitoring configs
        monitoring_config = self.project_root / "config" / "monitoring"
        if monitoring_config.exists():
            self.run_command(f"cp -r {monitoring_config}/* monitoring/")
        
        # Restart monitoring services
        self.run_command(
            "docker-compose -f docker/docker-compose.prod.yml restart prometheus grafana"
        )

    def run_health_checks(self) -> bool:
        """Run post-deployment health checks."""
        logger.info("Running health checks...")
        
        import requests
        
        checks = []
        
        # API health check
        try:
            api_port = self.config.get('api', {}).get('port', 8000)
            response = requests.get(f"http://localhost:{api_port}/health", timeout=10)
            if response.status_code == 200:
                checks.append("✓ API health check passed")
            else:
                checks.append(f"✗ API health check failed: {response.status_code}")
        except Exception as e:
            checks.append(f"✗ API health check failed: {e}")

        # MCP server health check
        try:
            mcp_port = self.config.get('mcp', {}).get('port', 8001)
            response = requests.get(f"http://localhost:{mcp_port}/health", timeout=10)
            if response.status_code == 200:
                checks.append("✓ MCP server health check passed")
            else:
                checks.append(f"✗ MCP server health check failed: {response.status_code}")
        except Exception as e:
            checks.append(f"✗ MCP server health check failed: {e}")

        # Vector store health check
        try:
            weaviate_port = self.config.get('vector_store', {}).get('port', 8080)
            response = requests.get(f"http://localhost:{weaviate_port}/v1/.well-known/ready", timeout=10)
            if response.status_code == 200:
                checks.append("✓ Vector store health check passed")
            else:
                checks.append(f"✗ Vector store health check failed: {response.status_code}")
        except Exception as e:
            checks.append(f"✗ Vector store health check failed: {e}")

        # Log results
        for check in checks:
            if "✓" in check:
                logger.info(check)
            else:
                logger.error(check)

        # Return True if all checks passed
        return all("✓" in check for check in checks)

    def show_deployment_info(self) -> None:
        """Show deployment information."""
        logger.info("Deployment complete!")
        
        api_port = self.config.get('api', {}).get('port', 8000)
        mcp_port = self.config.get('mcp', {}).get('port', 8001)
        grafana_port = self.config.get('monitoring', {}).get('grafana_port', 3000)
        
        print("\n" + "="*60)
        print("🚀 SignalCLI Deployment Information")
        print("="*60)
        print(f"Environment: {self.environment}")
        print(f"API Server: http://localhost:{api_port}")
        print(f"MCP Server: http://localhost:{mcp_port}")
        print(f"API Documentation: http://localhost:{api_port}/docs")
        print(f"Health Check: http://localhost:{api_port}/health")
        print(f"Metrics: http://localhost:{api_port}/metrics")
        print(f"Grafana Dashboard: http://localhost:{grafana_port}")
        print("\nLogs:")
        print(f"  docker-compose -f docker/docker-compose.prod.yml logs -f")
        print("\nManagement:")
        print(f"  Stop: docker-compose -f docker/docker-compose.prod.yml down")
        print(f"  Restart: docker-compose -f docker/docker-compose.prod.yml restart")
        print("="*60 + "\n")

    def rollback(self) -> None:
        """Rollback to previous deployment."""
        logger.info("Rolling back deployment...")
        
        # Stop current services
        self.run_command("docker-compose -f docker/docker-compose.prod.yml down")
        
        # Restore from backup (if available)
        backup_dir = self.project_root / "backups" / "latest"
        if backup_dir.exists():
            logger.info("Restoring from backup...")
            # Implement backup restoration logic
        
        logger.info("Rollback completed")

    def backup_data(self) -> None:
        """Backup important data."""
        logger.info("Creating backup...")
        
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.project_root / "backups" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup vector store data
        if (self.project_root / "data" / "vector_store").exists():
            self.run_command(f"cp -r data/vector_store {backup_dir}/")
        
        # Backup configs
        if (self.project_root / "config").exists():
            self.run_command(f"cp -r config {backup_dir}/")
        
        # Create latest symlink
        latest_link = self.project_root / "backups" / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(timestamp)
        
        logger.info(f"Backup created: {backup_dir}")

    def full_deploy(self) -> None:
        """Run full deployment process."""
        try:
            # Pre-deployment
            if not self.check_prerequisites():
                sys.exit(1)
            
            self.backup_data()
            self.setup_environment()
            
            # Build and test
            self.build_images()
            
            if self.config.get("run_tests", True):
                self.run_tests()
            
            # Deploy
            self.deploy_services()
            self.setup_monitoring()
            
            # Post-deployment
            if self.run_health_checks():
                self.show_deployment_info()
                logger.info("🎉 Deployment successful!")
            else:
                logger.error("❌ Health checks failed")
                if self.config.get("rollback_on_failure", True):
                    self.rollback()
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            if self.config.get("rollback_on_failure", True):
                self.rollback()
            sys.exit(1)


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="Deploy SignalCLI")
    parser.add_argument("--config", "-c", type=Path, default="config/deployment.yml",
                       help="Deployment config file")
    parser.add_argument("--environment", "-e", default="production",
                       help="Deployment environment")
    parser.add_argument("--action", "-a", default="deploy",
                       choices=["deploy", "rollback", "health-check", "backup"],
                       help="Deployment action")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip test execution")
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Create deployment manager
    deployer = DeploymentManager(config_path, args.environment)
    
    # Override config based on args
    if args.skip_tests:
        deployer.config["run_tests"] = False
    
    # Execute action
    if args.action == "deploy":
        deployer.full_deploy()
    elif args.action == "rollback":
        deployer.rollback()
    elif args.action == "health-check":
        if deployer.run_health_checks():
            print("All health checks passed ✓")
        else:
            print("Some health checks failed ✗")
            sys.exit(1)
    elif args.action == "backup":
        deployer.backup_data()


if __name__ == "__main__":
    main()