#!/bin/bash

# Find all **init**.py files and rename them to __init__.py
find . -name "\*\*init\*\*.py" -type f | while read file; do
    dir=$(dirname "$file")
    mv "$file" "$dir/__init__.py"
    echo "Fixed init file in: $dir"
done

# Delete cleanup scripts
rm -f cleanup_project.py final_cleanup.py fix_init__.py organize_project_.py

# Show results
echo "Done! All init files have been fixed."
tree
