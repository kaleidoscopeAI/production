#!/usr/bin/env python3
import os
from pathlib import Path
import logging

class FinalCleaner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.logger = logging.getLogger("FinalCleaner")

    def fix_init_files(self):
        """Fix incorrectly named __init__.py files"""
        for root, dirs, files in os.walk(str(self.root_dir)):
            for file in files:
                # Use literal string comparison instead of glob
                if file == '**init**.py':
                    old_path = Path(root) / file
                    new_path = old_path.parent / '__init__.py'
                    try:
                        if new_path.exists():
                            new_path.unlink()
                        old_path.rename(new_path)
                        self.logger.info(f"Fixed init file in {old_path.parent}")
                    except Exception as e:
                        self.logger.error(f"Error fixing init file in {old_path.parent}: {e}")

    def cleanup_duplicate_scripts(self):
        """Remove or merge duplicate organization scripts"""
        if (self.root_dir / 'organize_project_.py').exists():
            try:
                (self.root_dir / 'organize_project_.py').unlink()
                self.logger.info("Removed duplicate organizer script")
            except Exception as e:
                self.logger.error(f"Error removing duplicate script: {e}")

    def convert_remaining_txt(self):
        """Convert remaining txt files to appropriate formats"""
        skip_files = {'requirements.txt', 'GDCREDZ.txt'}
        
        for root, dirs, files in os.walk(str(self.root_dir)):
            for file in files:
                if file.endswith('.txt') and file not in skip_files:
                    txt_path = Path(root) / file
                    try:
                        with open(txt_path, 'r') as f:
                            content = f.read()
                        
                        # Determine file type based on content
                        new_ext = None
                        if any(x in content for x in ['def ', 'class ', 'import ']):
                            new_ext = '.py'
                        elif any(x in content for x in ['provider "aws"', 'resource "']):
                            new_ext = '.tf'
                        elif any(x in content for x in ['#!/bin/bash', 'apt-get']):
                            new_ext = '.sh'
                        
                        if new_ext:
                            new_path = txt_path.with_suffix(new_ext)
                            txt_path.rename(new_path)
                            self.logger.info(f"Converted {txt_path} to {new_path}")
                    except Exception as e:
                        self.logger.error(f"Error processing {txt_path}: {e}")

    def run(self):
        """Run all cleanup operations"""
        try:
            self.logger.info("Starting final cleanup...")
            self.fix_init_files()
            self.cleanup_duplicate_scripts()
            self.convert_remaining_txt()
            self.logger.info("Final cleanup complete!")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
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
