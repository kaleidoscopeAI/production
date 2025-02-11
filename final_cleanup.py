#!/usr/bin/env python3
import os
from pathlib import Path
import logging
import shutil

class FinalCleaner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.logger = logging.getLogger("FinalCleaner")
        
        # Specific file fixes needed
        self.file_fixes = {
            'docker-composedocker-compose.yml': 'docker-compose.yml',
            'Dockerfile (ai engine)Dockerfile.ai-engine': 'Dockerfile.ai-engine',
            'Dockerfile (chatbot APIDockerfile.chatbot': 'Dockerfile.chatbot',
            'terraform_infra.tf.tf': 'terraform_infra.tf'
        }

    def fix_init_files(self):
        """Fix incorrectly named __init__.py files"""
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            if '**init**.py' in filenames:
                old_path = Path(dirpath) / '**init**.py'
                new_path = Path(dirpath) / '__init__.py'
                try:
                    if new_path.exists():
                        new_path.unlink()
                    old_path.rename(new_path)
                    self.logger.info(f"Fixed init file in {dirpath}")
                except Exception as e:
                    self.logger.error(f"Error fixing init file in {dirpath}: {e}")

    def fix_specific_files(self):
        """Fix specific file naming issues"""
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename in self.file_fixes:
                    old_path = Path(dirpath) / filename
                    new_path = Path(dirpath) / self.file_fixes[filename]
                    try:
                        if new_path.exists():
                            new_path.unlink()
                        old_path.rename(new_path)
                        self.logger.info(f"Fixed filename: {filename} -> {self.file_fixes[filename]}")
                    except Exception as e:
                        self.logger.error(f"Error fixing {filename}: {e}")

    def run(self):
        """Run final cleanup operations"""
        try:
            self.logger.info("Starting final cleanup...")
            self.fix_init_files()
            self.fix_specific_files()
            self.logger.info("Final cleanup complete!")
        except Exception as e:
            self.logger.error(f"Error during final cleanup: {e}")
            raise

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cleaner = FinalCleaner(root_dir)
    cleaner.run()

if __name__ == "__main__":
    main()
