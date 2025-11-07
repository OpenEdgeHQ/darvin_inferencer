# Darvin Inferencer

An automated worker service that monitors and fulfills orders for inference services on the blockchain.

## Overview

The Darvin Inferencer Worker continuously monitors new orders for a specified service and automatically processes them by calling the smart contract's `fulfillService` function.

## Features

- **Automated Order Processing**: Continuously monitors and responds to new orders
- **Configurable Polling**: Adjustable monitoring interval
- **Echo Mode**: Currently returns the same payload as received (can be customized)
- **Error Handling**: Automatic retry for failed orders
- **Blockchain Integration**: Direct interaction with InferencerManager smart contract

## Prerequisites

- Python 3.8+
- Web3 provider access (RPC endpoint)
- Private key for transaction signing
- Active service registration on the blockchain

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd darvin_inferencer
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The worker can be configured using environment variables:

- `RPC_URL`: Blockchain RPC endpoint (default: `http://43.143.212.26:8545`)
- `CONTRACT_ADDRESS`: InferencerManager contract address (default: `0x3ff54294A0a78BD21cc6aDE9Af0363B1Bb0E64Fc`)
- `INFERENCER_PRIVATE_KEY`: Private key for signing transactions (recommended way to provide key)

## Usage

### Basic Usage

Run the worker for a specific service:

```bash
python inferencer_worker.py --service-id 1
```

### Advanced Options

```bash
python inferencer_worker.py \
  --service-id 1 \
  --private-key YOUR_PRIVATE_KEY \
  --interval 5
```

### Command Line Options

- `--service-id`: (Required) Service ID to monitor
- `--private-key`: Transaction private key (defaults to `INFERENCER_PRIVATE_KEY` environment variable)
- `--interval`: Polling interval in seconds (default: 3)

### Using Environment Variables

For better security, use environment variables:

```bash
export INFERENCER_PRIVATE_KEY="your_private_key_here"
export RPC_URL="http://your-rpc-endpoint:8545"
export CONTRACT_ADDRESS="0xYourContractAddress"

python inferencer_worker.py --service-id 1
```

## How It Works

1. **Initialization**: Connects to the blockchain and verifies service existence
2. **Monitoring**: Continuously polls for new orders at specified intervals
3. **Processing**: When a new order is detected:
   - Retrieves order details from the smart contract
   - Processes the order (currently echoes the input)
   - Submits the result back to the blockchain via `asyncfulfillService`
4. **Tracking**: Maintains a set of processed orders to avoid duplicates
5. **Error Handling**: Failed orders are retried in the next polling cycle

## Project Structure

```
darvin_inferencer/
├── abi/
│   └── inferencer_manager.json  # Contract ABI
├── inferencer_worker.py        # Main worker script
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Development

### Customizing Order Processing

The current implementation echoes the input payload. To customize the processing logic, modify the `process_new_orders` function in `inferencer_worker.py`:

```python
# Current implementation (line ~241):
result_bytes = order.inputs if isinstance(order.inputs, (bytes, bytearray)) else b""

# Replace with your custom logic:
result_bytes = your_custom_processing_function(order.inputs)
```

## Troubleshooting

### Connection Issues

If you encounter connection errors:
- Verify the RPC URL is accessible
- Check network connectivity
- Ensure the contract address is correct

### Transaction Failures

If transactions fail:
- Verify the private key has sufficient balance for gas fees
- Check that the wallet address is authorized for the service
- Ensure the service is active and not paused

### Order Processing Errors

If orders fail to process:
- Check the order status (may already be fulfilled)
- Verify service configuration
- Review transaction logs for specific error messages

## Security Notes

- Never commit private keys to version control
- Use environment variables for sensitive configuration
- Regularly rotate private keys
- Monitor transaction activity for unexpected behavior

## License

[Add your license here]

## Support

For issues and questions, please [add contact information or issue tracker link].
