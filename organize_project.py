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
        
        # Files that should be .py files
        self.py_patterns = [
            r'.*engine.*\.txt$',
            r'.*metrics.*\.txt$',
            r'.*crawler.*\.txt$',
            r'.*monitor.*\.txt$',
            r'.*integration.*\.txt$',
            r'.*automation.*\.txt$',
            r'.*deployment.*\.txt$',
            r'.*script.*\.txt$'
        ]
        
        # Define package structure with enhanced patterns
        self.packages = {
            'core': ['*engine*.py', '*core*.py', '*engine*.txt'],
            
            'chatbot': ['*chatbot*.py', '*chat*.py', '*chatbot*.txt'],
            
            'engines': ['*engine*.py', '*kaleidoscope*.py', '*mirror*.py', '*engine*.txt'],
            
            'membrane': ['*membrane*.py', '*membrane*.txt'],
            
            'pipeline': ['*pipeline*.py', '*pipeline*.txt'],
            
            'security': [
                '*secure*.py', '*security*.py', 
                '*godaddy*.py', '*godaddy*.txt',  # Add GoDaddy security files
                'GDCREDZ.txt'  # Specific credential file
            ],
            
            'supernode': ['*supernode*.py', '*node*.py', '*node-matrix.tsx'],
            
            'monitoring': [
                '*monitor*.py', '*metrics*.py', '*metrics*.txt',
                'monitoring-rules.txt'  # Specific monitoring file
            ],
            
            'visualization': [
                '*visual*.py', '*frontend*.tsx', '*graph*.tsx',
                'node-matrix.tsx'  # UI component
            ],
            
            'knowledge': ['*knowledge*.py', '*molecular*.py', '*ml-optimization.py'],
            
            'infrastructure': [
                '*aws*.py', '*aws*.txt',
                '*deploy*.py', '*deploy*.txt',
                '*terraform*', 
                '*docker*',
                '*cloudformation*.txt',
                'launch-script.txt',
                '*automation*.py'
            ],
            
            'services': ['*service*.py'],
            
            'tests': ['*test*.py'],
            
            'config': [
                '*.yml', '*.yaml', '*.json',
                'file-structure.txt',  # Project structure documentation
                '*github-cicd*.txt'    # CI/CD configuration
            ],
            
            'scripts': [
                '*.sh',
                'launch-script.txt',
                '*script*.py',
                'automation*.py'
            ]
        }

        # Files to keep in root directory
        self.root_files = [
            'setup.py',
            'requirements.txt',
            '.gitignore',
            'README.md',
            'organize_project.py'
        ]

    def should_stay_in_root(self, filename: str) -> bool:
        """Check if file should remain in root directory"""
        return filename in self.root_files

    def create_structure(self):
        """Create package structure"""
        for package in self.packages:
            package_dir = self.root_dir / package
            package_dir.mkdir(exist_ok=True)
            init_file = package_dir / '__init__.py'
            if not init_file.exists():
                init_file.touch()

    def should_be_py_file(self, filename: str) -> bool:
        """Check if a .txt file should be converted to .py"""
        return any(re.match(pattern, filename) for pattern in self.py_patterns)

    def convert_to_py(self, file_path: Path) -> Path:
        """Convert a .txt file to .py if it contains Python code"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                # Basic check if file looks like Python code
                python_indicators = ['import ', 'def ', 'class ', 'async ', 'await ', '#!/usr/bin']
                if any(indicator in content for indicator in python_indicators):
                    new_path = file_path.with_suffix('.py')
                    return new_path
        except Exception as e:
            self.logger.error(f"Error checking {file_path}: {str(e)}")
        return file_path

    def organize_files(self):
        """Organize files into appropriate packages"""
        self.logger.info("Starting file organization...")
        
        # Get all files in root directory
        files = [f for f in self.root_dir.glob('*') if f.is_file()]
        
        for file in files:
            filename = file.name
            
            # Skip files that should stay in root
            if self.should_stay_in_root(filename):
                continue
                
            moved = False
            
            # Check if .txt file should be .py
            if file.suffix == '.txt' and self.should_be_py_file(filename):
                new_file = self.convert_to_py(file)
                if new_file != file:
                    try:
                        file = file.rename(new_file)
                        filename = file.name
                        self.logger.info(f"Converted {file.stem}.txt to {filename}")
                    except Exception as e:
                        self.logger.error(f"Error converting {filename}: {str(e)}")
                        continue
            
            # Find appropriate package for file
            for package, patterns in self.packages.items():
                if any(re.match(pattern.replace('*', '.*'), filename, re.IGNORECASE) 
                      for pattern in patterns):
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

    def run(self):
        """Run the complete organization process"""
        try:
            self.logger.info("Starting project organization...")
            self.create_structure()
            self.organize_files()
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
