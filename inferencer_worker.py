"""Inferencer Worker: Continuously monitors and automatically responds to orders for a specified service.

This is a standalone script with no external project dependencies.

Prerequisites:
- Activate project virtual environment: ``source .venv/bin/activate``
- Install web3 dependency: ``pip install web3``
- Provide private key (recommended via environment variable INFERENCER_PRIVATE_KEY)

Examples:
    python inferencer_worker.py --service-id 123

Description:
- The script continuously monitors new orders for the specified service-id
- When a new order is detected, it automatically calls fulfillService to return the payload
- Current logic: the returned payload is the same as the order's inputs (echo mode)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional, Set

import click
from web3 import Web3


# ============================================================================
# Data Structure Definitions
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
# Utility Functions
# ============================================================================

def default_rpc_url() -> str:
    """Return default RPC URL, read from environment variable or use default value."""
    return os.getenv("RPC_URL", "http://43.143.212.26:8545")


def default_contract_address() -> str:
    """Return InferencerManager contract address, read from environment variable or use default value."""
    return os.getenv("CONTRACT_ADDRESS", "0x3ff54294A0a78BD21cc6aDE9Af0363B1Bb0E64Fc")


@lru_cache(maxsize=1)
def _get_inferencer_abi():
    """Load InferencerManager contract ABI."""
    abi_path = Path(__file__).resolve().parent / "abi" / "inferencer_manager.json"
    with abi_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    abi = payload.get("abi")
    if not abi:
        raise RuntimeError(f"Missing abi field in file {abi_path}")
    return abi


def get_inferencer_contract(w3: Web3, address: str):
    """Return InferencerManager contract instance."""
    return w3.eth.contract(
        address=Web3.to_checksum_address(address),
        abi=_get_inferencer_abi(),
    )


def init_web3_and_contract() -> tuple[Web3, Any]:
    """Initialize Web3 instance and return InferencerManager contract."""
    rpc_url = default_rpc_url()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"Unable to connect to node: {rpc_url}")

    contract = get_inferencer_contract(w3, default_contract_address())
    return w3, contract


def build_fee_fields(w3: Web3) -> dict:
    """Build gas-related fields for transactions."""
    return {"gasPrice": w3.eth.gas_price}


def ensure_private_key(args_private_key: Optional[str], env_var: str = "INFERENCER_PRIVATE_KEY") -> str:
    """Read private key from arguments or environment variable."""
    key = args_private_key or os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Private key not provided: set via --private-key or environment variable {env_var}")
    return key


def format_timestamp(ts: int) -> str:
    """Format timestamp to UTC+8 ISO string."""
    if not ts:
        return "N/A"
    tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(ts, tz).isoformat()


def iter_service_orders(contract, service_id: Optional[int] = None, scan_limit: int = 0) -> Iterator[ServiceOrderSummary]:
    """Iterate through orders, optionally filtered by service_id. Limit scan count when scan_limit>0."""
    next_order_id = contract.functions.nextOrderId().call()
    scanned = 0
    for order_id in range(1, next_order_id):
        if scan_limit and scanned >= scan_limit:
            break
        order = contract.functions.serviceOrders(order_id).call()
        scanned += 1
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


# ============================================================================
# Worker Core Logic
# ============================================================================

def fulfill_order(
    w3: Web3,
    contract,
    private_key: str,
    order_id: int,
    result_bytes: bytes,
) -> bool:
    """Fulfill specified order, return whether successful."""
    try:
        account = w3.eth.account.from_key(private_key)

        tx_options = {"from": account.address}
        fee_fields = build_fee_fields(w3)
        tx_options.update(fee_fields)

        gas_estimate = contract.functions.asyncfulfillService(
            order_id,
            result_bytes,
        ).estimate_gas(tx_options)
        tx_options["gas"] = int(gas_estimate * 1.1)

        tx_options["nonce"] = w3.eth.get_transaction_count(account.address)

        tx = contract.functions.asyncfulfillService(
            order_id,
            result_bytes,
        ).build_transaction(tx_options)

        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)

        print(f"  [Order {order_id}] Transaction sent: {tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            print(f"  [Order {order_id}] Transaction failed, status code {receipt.status}")
            return False

        events = contract.events.ServiceFulfilled().process_receipt(receipt)
        if events:
            event_args = events[0]["args"]
            print(f"  [Order {order_id}] Success: success={event_args.get('success')}")
        else:
            print(f"  [Order {order_id}] Success (no event parsed)")

        return True

    except Exception as exc:
        print(f"  [Order {order_id}] Failed: {exc}")
        return False


def process_new_orders(
    w3: Web3,
    contract,
    private_key: str,
    service_id: int,
    processed_orders: Set[int],
) -> int:
    """Process new orders, return count of orders processed."""
    new_order_count = 0

    for order in iter_service_orders(contract, service_id):
        order_id = order.order_id

        # Skip already processed orders
        if order_id in processed_orders:
            continue

        # status=0 means unfulfilled, status=1 means fulfilled
        if order.status != 0:
            # Order already fulfilled, mark as processed
            processed_orders.add(order_id)
            continue

        # New order detected
        created_at = format_timestamp(order.timestamp)
        inputs_hex = (
            order.inputs.hex() if isinstance(order.inputs, (bytes, bytearray)) else ""
        )
        inputs_display = f"0x{inputs_hex}" if inputs_hex else "None"

        print(f"\n[New Order] orderId={order_id}, caller={order.caller}, price={order.price}, timestamp={created_at}")
        print(f"  inputs={inputs_display}")

        # Return fixed payload (currently echoing the request payload)
        result_bytes = order.inputs if isinstance(order.inputs, (bytes, bytearray)) else b""

        result_hex = result_bytes.hex() if result_bytes else ""
        result_display = f"0x{result_hex}" if result_hex else "None"
        print(f"  [Processing] Returning payload: {result_display}")

        # Fulfill order
        success = fulfill_order(w3, contract, private_key, order_id, result_bytes)

        if success:
            processed_orders.add(order_id)
            new_order_count += 1
        else:
            # Failed orders are not marked as processed, will retry in next loop
            print(f"  [Order {order_id}] Will retry in next loop")

    return new_order_count


# ============================================================================
# CLI Entry Point
# ============================================================================

@click.command()
@click.option("--service-id", type=int, required=True, help="Service ID to monitor")
@click.option(
    "--private-key",
    help="Transaction private key (defaults to environment variable INFERENCER_PRIVATE_KEY)",
)
@click.option(
    "--interval",
    type=int,
    default=3,
    show_default=True,
    help="Polling interval in seconds",
)
def main(service_id: int, private_key: Optional[str], interval: int) -> None:
    """Inferencer Worker: Continuously monitors and automatically responds to orders for a specified service."""
    print(f"=== Inferencer Worker Started ===")
    print(f"Service ID: {service_id}")
    print(f"Polling interval: {interval} seconds")
    print()

    # Initialize Web3 and contract
    try:
        w3, contract = init_web3_and_contract()
        print(f"Connected to blockchain node")
    except Exception as exc:
        print(f"Failed to connect to blockchain node: {exc}")
        sys.exit(1)

    # Verify private key
    try:
        private_key_str = ensure_private_key(private_key)
        account = w3.eth.account.from_key(private_key_str)
        print(f"Using wallet address: {account.address}")
    except Exception as exc:
        print(f"Private key verification failed: {exc}")
        sys.exit(1)

    # Verify service exists
    try:
        service = contract.functions.inferenceServices(service_id).call()
        if not service[0]:  # node_id of 0 means service doesn't exist
            print(f"Service ID {service_id} does not exist")
            sys.exit(1)
        print(f"Service info: nodeId={service[0]}, subnetId={service[1]}, modelId={service[2]}, price={service[3]}")
    except Exception as exc:
        print(f"Failed to query service info: {exc}")
        sys.exit(1)

    print()
    print("Started monitoring orders...")
    print()

    processed_orders: Set[int] = set()
    loop_count = 0

    try:
        while True:
            loop_count += 1

            try:
                new_count = process_new_orders(
                    w3,
                    contract,
                    private_key_str,
                    service_id,
                    processed_orders,
                )

                if new_count > 0:
                    print(f"[Loop #{loop_count}] Processed {new_count} new order(s), total processed: {len(processed_orders)}")

            except Exception as exc:
                print(f"[Loop #{loop_count}] Error processing orders: {exc}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print()
        print("=== Inferencer Worker Stopped ===")
        print(f"Total processed {len(processed_orders)} order(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()

