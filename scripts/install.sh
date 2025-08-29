python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
export PYTORCH_ENABLE_MPS_FALLBACK=1