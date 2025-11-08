"""Darvin Inferencer Worker

An automated inference worker that:
1. Auto-registers node and service if not already registered
2. Continuously monitors and processes inference orders

Prerequisites:
- Python 3.8+
- Set required environment variables

Usage:
    export INFERENCER_PRIVATE_KEY="0x..."
    export MODEL_ID="1"
    export MODEL_CID="QmExample123..."
    export PRICE="0.001"
    python inferencer_worker.py

Environment Variables:
    INFERENCER_PRIVATE_KEY: Your private key (required)
    MODEL_ID: Whitelisted model ID (required)
    MODEL_CID: Model IPFS CID (required)
    PRICE: Price per call in tokens (required)
    SYSTEM_PROMPT: System prompt for model (optional, default: translation)
    MAX_NEW_TOKENS: Maximum tokens to generate (optional, default: 4096)
    INTERVAL: Polling interval in seconds (optional, default: 3)
    RPC_URL: Blockchain RPC endpoint (optional)
    CONTRACT_ADDRESS: InferencerManager contract address (optional)
    API_SERVER_URL: API server URL (optional)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional, Set

import click
from web3 import Web3

from model_utils import download_model
from inference import ModelInference


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """Configure logging with timestamp and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DEVICE_DESCRIPTION = "Darvin Inference Node"


def default_rpc_url() -> str:
    return os.getenv("RPC_URL", "http://43.143.212.26:8545")


def default_contract_address() -> str:
    return os.getenv("CONTRACT_ADDRESS", "0x3ff54294A0a78BD21cc6aDE9Af0363B1Bb0E64Fc")


def default_api_server_url() -> str:
    return os.getenv("API_SERVER_URL", "https://darvin-backend-test.gradient.network")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ServiceOrderSummary:
    order_id: int
    service_id: int
    caller: str
    inputs: bytes
    compressed_result: bytes
    price: int
    timestamp: int
    status: int


# ============================================================================
# Contract Interaction
# ============================================================================

@lru_cache(maxsize=1)
def _get_inferencer_abi():
    abi_path = Path(__file__).resolve().parent / "abi" / "inferencer_manager.json"
    with abi_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    abi = payload.get("abi")
    if not abi:
        raise RuntimeError(f"Missing abi field in file {abi_path}")
    return abi


def get_inferencer_contract(w3: Web3, address: str):
    return w3.eth.contract(
        address=Web3.to_checksum_address(address),
        abi=_get_inferencer_abi(),
    )


def init_web3_and_contract() -> tuple[Web3, Any]:
    rpc_url = default_rpc_url()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"Unable to connect to node: {rpc_url}")
    
    contract = get_inferencer_contract(w3, default_contract_address())
    return w3, contract


def build_fee_fields(w3: Web3) -> dict:
    return {"gasPrice": w3.eth.gas_price}


def ensure_private_key() -> str:
    key = os.getenv("INFERENCER_PRIVATE_KEY")
    if not key:
        raise RuntimeError(
            "Private key not provided. Set environment variable INFERENCER_PRIVATE_KEY"
        )
    return key


def format_timestamp(ts: int) -> str:
    if not ts:
        return "N/A"
    tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(ts, tz).isoformat()


# ============================================================================
# Auto-Registration Functions
# ============================================================================

def check_node_registered(w3: Web3, contract, account_address: str) -> Optional[int]:
    """Check if account has registered node. Returns node_id or None."""
    try:
        node_id = contract.functions.ownerToNodeId(account_address).call()
        if node_id > 0:
            return node_id
        return None
    except Exception:
        return None


def register_node(w3: Web3, contract, private_key: str) -> int:
    """Register a new inference node. Returns node_id."""
    account = w3.eth.account.from_key(private_key)
    
    # Get stake amount from contract
    stake_amount = contract.functions.BASE_STAKE_AMOUNT().call()
    
    logger.info("Registering node...")
    logger.info(f"Wallet: {account.address}")
    logger.info(f"Stake: {stake_amount / 1e18:.6f} tokens")
    
    tx_options = {
        "from": account.address,
        "value": stake_amount,
    }
    tx_options.update(build_fee_fields(w3))
    
    device_descriptions = [DEFAULT_DEVICE_DESCRIPTION]
    gas_estimate = contract.functions.registerAsInferencerNode(
        device_descriptions
    ).estimate_gas(tx_options)
    tx_options["gas"] = int(gas_estimate * 1.1)
    tx_options["nonce"] = w3.eth.get_transaction_count(account.address)
    
    tx = contract.functions.registerAsInferencerNode(
        device_descriptions
    ).build_transaction(tx_options)
    
    signed = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    
    logger.info(f"Transaction sent: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    if receipt.status != 1:
        raise RuntimeError(f"Node registration failed with status {receipt.status}")
    
    # Parse event to get node_id
    events = contract.events.InferencerRegistered().process_receipt(receipt)
    if events:
        node_id = events[0]["args"].get("nodeId")
        logger.info(f"Node registered successfully (Node ID: {node_id})")
        return node_id
    else:
        raise RuntimeError("Node registration event not found")


def check_service_registered(contract, node_id: int, model_id: int) -> Optional[int]:
    """Check if service exists for node and model. Returns service_id or None."""
    try:
        next_service_id = contract.functions.nextServiceId().call()
        for service_id in range(1, next_service_id):
            service = contract.functions.inferenceServices(service_id).call()
            if service[0] == node_id and service[2] == model_id:  # nodeId, modelId
                return service_id
        return None
    except Exception:
        return None


def register_service(
    w3: Web3, 
    contract, 
    private_key: str, 
    model_id: int, 
    node_id: int, 
    price_token: float
) -> int:
    """Register a new service. Returns service_id."""
    account = w3.eth.account.from_key(private_key)
    
    # Convert price to wei
    price_decimal = Decimal(str(price_token))
    price_wei = int((price_decimal * Decimal(10 ** 18)).to_integral_value(rounding=ROUND_DOWN))
    
    deploy_uri = default_rpc_url()
    
    logger.info("Registering service...")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Node ID: {node_id}")
    logger.info(f"Price: {price_token} tokens/call")
    
    tx_options = {"from": account.address}
    tx_options.update(build_fee_fields(w3))
    
    gas_estimate = contract.functions.deployModelAndRegisterService(
        model_id, node_id, price_wei, deploy_uri
    ).estimate_gas(tx_options)
    tx_options["gas"] = int(gas_estimate * 1.1)
    tx_options["nonce"] = w3.eth.get_transaction_count(account.address)
    
    tx = contract.functions.deployModelAndRegisterService(
        model_id, node_id, price_wei, deploy_uri
    ).build_transaction(tx_options)
    
    signed = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    
    logger.info(f"Transaction sent: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    if receipt.status != 1:
        raise RuntimeError(f"Service registration failed with status {receipt.status}")
    
    # Parse event to get service_id
    service_events = contract.events.ServiceRegistered().process_receipt(receipt)
    if service_events:
        event_args = service_events[0]["args"]
        service_id = event_args.get("serviceId")
        subnet_id = event_args.get("subnetId")
        logger.info("Service registered successfully")
        logger.info(f"  Service ID: {service_id}")
        logger.info(f"  Subnet ID: {subnet_id}")
        logger.info(f"  Model ID: {model_id}")
        return service_id
    else:
        raise RuntimeError("Service registration event not found")


def ensure_registration(
    w3: Web3,
    contract,
    private_key: str,
    model_id: int,
    price_token: float
) -> int:
    """Ensure node and service are registered. Returns service_id."""
    account = w3.eth.account.from_key(private_key)
    
    # Check node registration
    node_id = check_node_registered(w3, contract, account.address)
    if node_id is None:
        node_id = register_node(w3, contract, private_key)
    else:
        logger.info(f"Node already registered (Node ID: {node_id})")
    
    # Check service registration
    service_id = check_service_registered(contract, node_id, model_id)
    if service_id is None:
        service_id = register_service(w3, contract, private_key, model_id, node_id, price_token)
    else:
        # Get service info to show subnet_id
        try:
            service_info = contract.functions.inferenceServices(service_id).call()
            subnet_id = service_info[1] if len(service_info) > 1 else "Unknown"
            logger.info("Service already registered")
            logger.info(f"  Service ID: {service_id}")
            logger.info(f"  Subnet ID: {subnet_id}")
        except Exception:
            logger.info(f"Service already registered (Service ID: {service_id})")
    
    return service_id


# ============================================================================
# Order Processing
# ============================================================================

def iter_service_orders(
    contract, 
    service_id: Optional[int] = None
) -> Iterator[ServiceOrderSummary]:
    """Iterate through orders, optionally filtered by service_id."""
    next_order_id = contract.functions.nextOrderId().call()
    for order_id in range(1, next_order_id):
        order = contract.functions.serviceOrders(order_id).call()
        if not order[0]:
            continue
        if service_id is not None and order[0] != service_id:
            continue
        yield ServiceOrderSummary(
            order_id=order_id,
            service_id=order[0],
            caller=order[1],
            inputs=order[2],
            compressed_result=order[3],
            price=order[4],
            timestamp=order[5],
            status=order[6],
        )


def fulfill_order(
    w3: Web3,
    contract,
    private_key: str,
    order_id: int,
    result_bytes: bytes,
) -> bool:
    """Fulfill an order. Returns success status."""
    try:
        account = w3.eth.account.from_key(private_key)
        
        # Log result size for debugging
        result_size_kb = len(result_bytes) / 1024
        logger.info(f"[Order {order_id}] Result size: {result_size_kb:.2f} KB ({len(result_bytes)} bytes)")
        
        # Check if result is too large (Ethereum transaction limit ~128KB)
        # Reserve some space for transaction overhead (~10KB)
        max_result_size = 118 * 1024  # ~118KB
        if len(result_bytes) > max_result_size:
            logger.warning(
                f"[Order {order_id}] Result size ({result_size_kb:.2f} KB) exceeds recommended limit "
                f"({max_result_size/1024:.2f} KB). Transaction may fail."
            )
        
        tx_options = {"from": account.address}
        tx_options.update(build_fee_fields(w3))
        
        gas_estimate = contract.functions.asyncfulfillService(
            order_id, result_bytes
        ).estimate_gas(tx_options)
        
        # Check gas limit (Ethereum block gas limit is typically 30M)
        max_gas_limit = 30_000_000
        estimated_gas = int(gas_estimate * 1.1)
        if estimated_gas > max_gas_limit:
            logger.error(
                f"[Order {order_id}] Estimated gas ({estimated_gas:,}) exceeds block limit "
                f"({max_gas_limit:,}). Transaction will fail."
            )
            return False
        
        tx_options["gas"] = estimated_gas
        tx_options["nonce"] = w3.eth.get_transaction_count(account.address)
        
        logger.info(f"[Order {order_id}] Estimated gas: {estimated_gas:,}")
        
        tx = contract.functions.asyncfulfillService(
            order_id, result_bytes
        ).build_transaction(tx_options)
        
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"[Order {order_id}] Transaction sent: {tx_hash.hex()}")
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            logger.error(f"[Order {order_id}] Transaction failed (status {receipt.status})")
            return False
        
        logger.info(f"[Order {order_id}] Fulfilled successfully")
        return True
        
    except Exception as exc:
        logger.error(f"[Order {order_id}] Error: {exc}")
        return False


def parse_order_input(input_bytes: bytes) -> str:
    """Parse order input from bytes to prompt string.
    
    Expected JSON format: {"prompt": "Your custom prompt here"}
    or raw text
    
    Args:
        input_bytes: Input bytes from order
        
    Returns:
        Prompt string
    """
    try:
        input_str = input_bytes.decode('utf-8')
        input_data = json.loads(input_str)
        # Try to get prompt field first, fallback to raw text
        return input_data.get("prompt", input_str)
    except Exception:
        # Not JSON, treat as raw prompt
        return input_bytes.decode('utf-8', errors='ignore')


def process_new_orders(
    w3: Web3,
    contract,
    private_key: str,
    service_id: int,
    model_inference: ModelInference,
    processed_orders: Set[int],
) -> int:
    """Process new orders. Returns count of processed orders."""
    new_order_count = 0
    
    for order in iter_service_orders(contract, service_id):
        order_id = order.order_id
        
        if order_id in processed_orders:
            continue
        
        if order.status != 0:  # Already fulfilled
            processed_orders.add(order_id)
            continue
        
        # New order detected
        created_at = format_timestamp(order.timestamp)
        inputs_hex = order.inputs.hex() if isinstance(order.inputs, (bytes, bytearray)) else ""
        inputs_display = f"0x{inputs_hex[:20]}..." if len(inputs_hex) > 20 else f"0x{inputs_hex}"
        
        logger.info(f"New Order {order_id}")
        logger.info(f"  Caller: {order.caller}")
        logger.info(f"  Price: {order.price / 1e18:.6f} tokens")
        logger.info(f"  Time: {created_at}")
        logger.info(f"  Input: {inputs_display}")
        
        try:
            # Parse input to get prompt
            prompt = parse_order_input(order.inputs)
            logger.info(f"  Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"  Prompt: {prompt}")
            
            # Run inference
            result = model_inference.generate(prompt)
            logger.info(f"  Result: {result[:100]}..." if len(result) > 100 else f"  Result: {result}")
            
            # Encode result
            result_bytes = result.encode('utf-8')
            
        except Exception as exc:
            logger.error(f"[Order {order_id}] Inference error: {exc}")
            # Fallback to echo mode on error
            result_bytes = order.inputs if isinstance(order.inputs, (bytes, bytearray)) else b""
        
        # Fulfill order
        success = fulfill_order(w3, contract, private_key, order_id, result_bytes)
        
        if success:
            processed_orders.add(order_id)
            new_order_count += 1
        else:
            logger.warning(f"[Order {order_id}] Will retry in next loop")
    
    return new_order_count


# ============================================================================
# Main
# ============================================================================

@click.command()
def main() -> None:
    """Darvin Inferencer Worker - Auto-registers and processes orders."""
    
    # Read from environment variables
    model_id = os.getenv("MODEL_ID")
    model_cid = os.getenv("MODEL_CID")
    price_str = os.getenv("PRICE")
    system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant that translates English text to Chinese.")
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "65536"))
    interval = int(os.getenv("INTERVAL", "1"))
    
    # Validate required parameters
    if not model_id:
        logger.error("MODEL_ID environment variable is required")
        sys.exit(1)
    if not model_cid:
        logger.error("MODEL_CID environment variable is required")
        sys.exit(1)
    if not price_str:
        logger.error("PRICE environment variable is required")
        sys.exit(1)
    
    try:
        model_id = int(model_id)
        price = float(price_str)
    except ValueError as e:
        logger.error(f"Invalid parameter format: {e}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Darvin Inferencer Worker Starting")
    logger.info("=" * 60)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Model CID: {model_cid}")
    logger.info(f"Price: {price} tokens/call")
    logger.info(f"System Prompt: {system_prompt}")
    logger.info(f"Max New Tokens: {max_new_tokens}")
    logger.info(f"Polling interval: {interval}s")
    
    try:
        # Initialize
        w3, contract = init_web3_and_contract()
        logger.info(f"Connected to blockchain: {default_rpc_url()}")
        
        private_key = ensure_private_key()
        account = w3.eth.account.from_key(private_key)
        logger.info(f"Using wallet: {account.address}")
        
        # Auto-register node and service
        logger.info("Checking registration...")
        service_id = ensure_registration(w3, contract, private_key, model_id, price)
        
        # Download and load model
        logger.info(f"Preparing model (CID: {model_cid})...")
        model_dir = download_model(model_cid, private_key, default_api_server_url())
        if not model_dir:
            logger.error("Failed to download model")
            sys.exit(1)
        
        logger.info("Initializing model inference...")
        model_inference = ModelInference(
            model_path=model_dir,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        logger.info("Model ready for inference")
        
        # Start monitoring
        logger.info(f"Monitoring orders for Service ID {service_id}...")
        logger.info("Press Ctrl+C to stop")
        
        processed_orders: Set[int] = set()
        loop_count = 0
        
        while True:
            loop_count += 1
            
            try:
                new_count = process_new_orders(
                    w3, contract, private_key, service_id, model_inference, processed_orders
                )
                
                if new_count > 0:
                    logger.info(f"Loop #{loop_count}: Processed {new_count} order(s), Total: {len(processed_orders)}")
                
            except Exception as exc:
                logger.warning(f"Loop #{loop_count} error: {exc}")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("=" * 60)
        logger.info("Worker stopped by user")
        logger.info(f"Total orders processed: {len(processed_orders) if 'processed_orders' in locals() else 0}")
        logger.info("=" * 60)
        sys.exit(0)
    
    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
