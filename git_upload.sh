#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory (which should be the git repo root)
cd "$SCRIPT_DIR"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository!"
    exit 1
fi

echo "Current directory: $(pwd)"
echo "Adding files to git..."
git add .

echo "Committing changes..."
git commit -m "Ran exp 6 and 7"

echo "Pushing to origin main..."
git push origin main

echo "Git upload completed successfully!"
