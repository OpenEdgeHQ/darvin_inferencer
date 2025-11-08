# Darvin Inferencer

Automated inference worker that monitors and processes blockchain inference orders.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export INFERENCER_PRIVATE_KEY="your_private_key"
export MODEL_ID="1"
export MODEL_CID="QmYourModelCID"
export PRICE="1"
```

### 3. Run Worker
```bash
python inferencer_worker.py
```

The worker will automatically:
- ✅ Register node (first run)
- ✅ Register service (first run)
- ✅ Download and cache model
- ✅ Monitor and process orders

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INFERENCER_PRIVATE_KEY` | ✅ | - | Wallet private key |
| `MODEL_ID` | ✅ | - | Trained model ID |
| `MODEL_CID` | ✅ | - | IPFS CID for trained model to download |
| `PRICE` | ✅ | - | Price per call in tokens |
| `SYSTEM_PROMPT` | ❌ | Translation assistant | System prompt for the model |

### System Prompt Examples

**Translation (default):**
```bash
export SYSTEM_PROMPT="You are a helpful assistant that translates English text to Chinese."
```

**Note:** User input from blockchain orders will be sent as the user message. The model will respond based on the system prompt.

## How It Works

```
1. Start Worker
   ↓
2. Check node registration → Auto-register if needed
   ↓
3. Check service registration → Auto-register if needed
   ↓
4. Download model → Auto-cached by CID
   ↓
5. Poll for orders → Check every N seconds
   ↓
6. Process orders → Run inference → Submit results
   ↓
7. Repeat from step 5
```

## Testing

### Manual Inference Test
```python
from model_utils import download_model
from inference import ModelInference

# Download model
model_dir = download_model(
    model_cid="QmYourCID",
    private_key="0x...",
    api_server_url="https://darvin-backend-test.gradient.network"
)

# Initialize with system prompt
inference = ModelInference(
    model_path=model_dir,
    system_prompt="You are a helpful assistant that translates English text to Chinese."
)

# Run inference (input will be sent as user message)
result = inference.generate("hello")
print(result)  # Output: "你好"
```

## Model Caching

Models are automatically cached at `/tmp/darvin_models/{CID}/`:
- **First download:** Authenticate → Download → Extract → Cache
- **Subsequent runs:** Use cached model directly

## Troubleshooting

**Q: How to confirm service is registered?**  
A: Check logs for `Service registered: Service ID xxx`

**Q: Orders not processing?**  
A: Verify service is active on blockchain explorer and price is competitive

**Q: How to change service price?**  
A: Restart worker with new `--price` parameter

**Q: How to stop worker?**  
A: Press `Ctrl+C` to see processing statistics and gracefully exit

## Project Structure

```
darvin_inferencer/
├── abi/
│   └── inferencer_manager.json    # Contract ABI
├── inferencer_worker.py           # Main worker
├── model_utils.py                 # Model download utilities
├── inference.py                   # Inference engine
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Security

- ⚠️ Never commit private keys to version control
- ✅ Use environment variables for secrets
- ✅ Rotate keys regularly
- ✅ Monitor transaction activity
