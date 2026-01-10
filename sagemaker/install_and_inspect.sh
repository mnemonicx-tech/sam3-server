#!/bin/bash
set -e

echo "🔧 Upgrading pip..."
python3 -m pip install --upgrade pip

echo "⬇️ Installing Dependencies..."
python3 -m pip install wget
# Install from the git repo
python3 -m pip install "ultralytics>=8.3.237"

echo "🔍 Running Inspection..."
python3 sagemaker/inspect_sam3.py
