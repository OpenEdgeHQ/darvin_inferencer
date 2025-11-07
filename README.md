# Darvin Inferencer

Automated inference worker that monitors and processes blockchain orders.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set Private Key

```bash
export INFERENCER_PRIVATE_KEY="your_private_key_here"
```

### 3. Run Worker

```bash
python inferencer_worker.py \
  --model-id 1 \
  --model-cid QmExample123... \
  --price 0.001
```

**That's it!** The worker will:
- ✅ Auto-register your node (if not registered)
- ✅ Auto-register your service (if not registered)
- ✅ Start monitoring and processing orders

## Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--model-id` | Yes | Model ID from whitelist | `1` |
| `--model-cid` | Yes | IPFS CID for model download | `QmAbc...` |
| `--price` | Yes | Price per call in tokens | `0.001` |
| `--interval` | No | Polling interval in seconds (default: 3) | `5` |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INFERENCER_PRIVATE_KEY` | ✅ Yes | - | Your wallet private key |

## How It Works

1. **First Run**: Automatically registers node + service with default settings
2. **Subsequent Runs**: Detects existing registration and starts monitoring
3. **Order Processing**: Polls for new orders and fulfills them automatically
4. **Error Handling**: Failed orders retry in next polling cycle

## Customization

### Change Model Processing

Edit `process_new_orders()` function in `inferencer_worker.py`:

```python
# Current: Echo mode (line ~341)
result_bytes = order.inputs

# Replace with your inference logic:
result_bytes = run_model_inference(model_cid, order.inputs)
```

### Change Device Description

Edit `DEFAULT_DEVICE_DESCRIPTION` constant (line ~47):

```python
DEFAULT_DEVICE_DESCRIPTION = "Your Custom Description"
```

## Troubleshooting

**Connection Error**
- Check `RPC_URL` is accessible
- Verify network connectivity

**Registration Failed**
- Ensure wallet has sufficient balance for gas + stake
- Verify `model-id` exists in whitelist (contact admin)

**No Orders Processing**
- Confirm service is active: Check blockchain explorer
- Verify service price matches market expectations

## Project Structure

```
darvin_inferencer/
├── abi/
│   └── inferencer_manager.json  # Contract ABI
├── inferencer_worker.py         # Main worker script
├── model_utils.py               # Model download utilities
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Model Download & Caching

The `download_model()` function automatically:
1. Checks if model is already cached (by CID)
2. If cached, returns cached path immediately
3. If not cached, authenticates and downloads from API server
4. Extracts and caches model for future use

Example usage in your inference code:
```python
from model_utils import download_model

# Download model (auto-login and caching handled internally)
model_dir = download_model(model_cid, private_key, api_server_url)
if model_dir:
    # Run your inference
    result = run_inference(model_dir, input_data)
```

**Cache Location**: Models are cached in system temp directory under `darvin_models/`. Each model is identified by its CID, so the same model is only downloaded once.

## Security

- ⚠️ Never commit private keys to git
- ✅ Use environment variables for secrets
- ✅ Rotate keys regularly
- ✅ Monitor transaction activity

## License

[Add your license]

## Support

For issues and questions, please contact [add contact info].
