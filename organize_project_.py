#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import re
import logging

class ProjectOrganizer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.logger = logging.getLogger("ProjectOrganizer")
        
        # Define package structure
        self.packages = {
            'core': ['*engine*.py', 'core*.py'],
            'chatbot': ['*chatbot*.py', '*chat*.py'],
            'engines': ['*engine*.py', '*kaleidoscope*.py', '*mirror*.py'],
            'membrane': ['*membrane*.py'],
            'pipeline': ['*pipeline*.py'],
            'security': ['*secure*.py', '*security*.py'],
            'supernode': ['*supernode*.py', '*node*.py'],
            'monitoring': ['*monitor*.py', '*metrics*.py'],
            'visualization': ['*visual*.py', '*frontend*.tsx', '*graph*.tsx'],
            'knowledge': ['*knowledge*.py', '*molecular*.py'],
            'infrastructure': ['*aws*.py', '*deploy*.py', '*terraform*', '*docker*'],
            'services': ['*service*.py'],
            'tests': ['*test*.py'],
            'config': ['*.yml', '*.yaml', '*.json'],
            'scripts': ['*.sh']
        }

    def create_structure(self):
        """Create basic package structure"""
        for package in self.packages:
            package_dir = self.root_dir / package
            package_dir.mkdir(exist_ok=True)
            
            # Create __init__.py if it doesn't exist
            init_file = package_dir / '__init__.py'
            if not init_file.exists():
                init_file.touch()

    def should_move_file(self, filename: str, patterns: list) -> bool:
        """Check if file matches any pattern"""
        return any(re.match(pattern.replace('*', '.*'), filename, re.IGNORECASE) 
                  for pattern in patterns)

    def organize_files(self):
        """Organize files into appropriate packages"""
        self.logger.info("Starting file organization...")
        
        # Get all files in root directory
        files = [f for f in self.root_dir.glob('*') if f.is_file()]
        
        for file in files:
            filename = file.name
            moved = False
            
            # Skip the organizer script itself
            if filename == Path(__file__).name:
                continue
                
            # Find appropriate package for file
            for package, patterns in self.packages.items():
                if self.should_move_file(filename, patterns):
                    target_dir = self.root_dir / package
                    target_file = target_dir / filename
                    
                    # Create numbered backup if file exists
                    if target_file.exists():
                        backup_num = 1
                        while (target_dir / f"{filename}.{backup_num}").exists():
                            backup_num += 1
                        target_file = target_dir / f"{filename}.{backup_num}"
                    
                    try:
                        shutil.move(str(file), str(target_file))
                        self.logger.info(f"Moved {filename} to {package}/")
                        moved = True
                        break
                    except Exception as e:
                        self.logger.error(f"Error moving {filename}: {str(e)}")

            if not moved:
                self.logger.warning(f"No matching package for {filename}")

    def create_requirements(self):
        """Create requirements.txt file"""
        requirements = [
            'numpy>=1.21.0',
            'torch>=1.9.0',
            'fastapi>=0.68.0',
            'uvicorn>=0.15.0',
            'boto3>=1.18.0',
            'pyyaml>=5.4.0',
            'prometheus-client>=0.11.0',
            'transformers>=4.9.0',
            'scipy>=1.7.0',
            'networkx>=2.6.0',
            'pandas>=1.3.0',
            'plotly>=5.1.0',
            'requests>=2.26.0',
            'python-dotenv>=0.19.0',
            'asyncio>=3.4.3',
            'aiohttp>=3.7.4',
            'scikit-learn>=0.24.2'
        ]
        
        with open(self.root_dir / 'requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))

    def create_gitignore(self):
        """Create .gitignore file"""
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# AWS
.aws/
*.pem

# Logs
*.log
logs/
log/

# Node (for frontend)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Build files
build/
dist/
*.pyc

# Local configuration
config/local.yml
"""
        with open(self.root_dir / '.gitignore', 'w') as f:
            f.write(gitignore_content)

    def run(self):
        """Run the complete organization process"""
        try:
            self.logger.info("Starting project organization...")
            
            # Create basic structure
            self.create_structure()
            self.logger.info("Created basic package structure")
            
            # Organize files
            self.organize_files()
            
            # Create additional files
            self.create_requirements()
            self.logger.info("Created requirements.txt")
            
            self.create_gitignore()
            self.logger.info("Created .gitignore")
            
            self.logger.info("Project organization complete!")
            
        except Exception as e:
            self.logger.error(f"Error organizing project: {str(e)}")
            raise

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create and run organizer
    organizer = ProjectOrganizer(root_dir)
    organizer.run()

if __name__ == "__main__":
    main()

