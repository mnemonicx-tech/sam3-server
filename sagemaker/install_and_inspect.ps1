Write-Host "🔧 Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "⬇️ Installing Dependencies..."
python -m pip install wget "ultralytics>=8.3.237"

Write-Host "🔍 Running Inspection..."
python sagemaker/inspect_sam3.py
