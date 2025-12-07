#!/bin/bash
# Launcher script for Gradio app
# This script ensures the app runs from the correct directory

# Find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if gradio_app.py exists
if [ ! -f "gradio_app.py" ]; then
    echo "Error: gradio_app.py not found in $SCRIPT_DIR"
    exit 1
fi

# Run the app
python gradio_app.py

