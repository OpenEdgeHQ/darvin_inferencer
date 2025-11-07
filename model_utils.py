"""Model Download and Management Utilities

Handles authentication and model file downloads from API server.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from typing import Optional, Tuple
from urllib import error, request

from eth_account import Account
from eth_account.messages import encode_defunct

logger = logging.getLogger(__name__)


# ============================================================================
# Authentication
# ============================================================================

def build_login_message(domain: str, address: str, timestamp: int) -> str:
    """Build login message for signing."""
    parts = [
        'LOGIN',
        f'Domain: {domain}',
        f'Address: {address}',
        f'Timestamp: {timestamp}',
    ]
    return '\n'.join(parts)


def sign_login_message(private_key: str, domain: str) -> Tuple[str, str, int]:
    """Sign login message with private key.
    
    Args:
        private_key: Ethereum private key
        domain: Domain for login
        
    Returns:
        Tuple of (signature, message_text, timestamp)
    """
    account = Account.from_key(private_key)
    wallet_address = account.address
    timestamp = int(time.time())
    
    message_text = build_login_message(domain, wallet_address, timestamp)
    message = encode_defunct(text=message_text)
    signed = account.sign_message(message)
    
    return signed.signature.hex(), message_text, timestamp


def perform_login(
    login_url: str,
    address: str,
    timestamp: int,
    signature: str,
    domain: str
) -> Tuple[int, dict, Optional[str]]:
    """Perform login request to API server.
    
    Args:
        login_url: Login endpoint URL
        address: Wallet address
        timestamp: Unix timestamp
        signature: Signed message
        domain: Domain string
        
    Returns:
        Tuple of (status_code, response_body, cookie)
    """
    payload = {
        'address': address,
        'timestamp': timestamp,
        'signature': signature,
        'domain': domain,
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(
        login_url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    
    try:
        with request.urlopen(req) as resp:
            body = resp.read().decode('utf-8')
            content = json.loads(body)
            cookie = resp.headers.get('Set-Cookie')
            return resp.getcode(), content, cookie
    except error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')
        try:
            content = json.loads(body)
        except json.JSONDecodeError:
            content = {'raw': body}
        cookie = exc.headers.get('Set-Cookie') if exc.headers else None
        return exc.code, content, cookie


def login_and_get_token(private_key: str, api_server_url: str) -> Optional[str]:
    """Login to API server and get JWT token.
    
    Args:
        private_key: Ethereum private key
        api_server_url: API server base URL
        
    Returns:
        JWT token or None if login failed
    """
    try:
        domain = api_server_url.replace('https://', '').replace('http://', '')
        login_url = f"{api_server_url}/api/auth/login"
        
        account = Account.from_key(private_key)
        
        # Generate signature
        signature, message_text, timestamp = sign_login_message(private_key, domain)
        
        logger.info(f"Logging in with wallet: {account.address}")
        
        # Perform login
        status, body, cookie = perform_login(
            login_url=login_url,
            address=account.address,
            timestamp=timestamp,
            signature=signature,
            domain=domain
        )
        
        if status == 200:
            token = body.get('data', {}).get('token')
            if token:
                logger.info("Login successful")
                return token
            else:
                logger.error("Login response missing token")
                return None
        else:
            logger.error(f"Login failed: HTTP {status}")
            logger.error(f"Response: {body}")
            return None
            
    except Exception as exc:
        logger.error(f"Login error: {exc}")
        return None


# ============================================================================
# Model Download
# ============================================================================

def download_model(
    model_cid: str,
    private_key: str,
    api_server_url: str,
    cache_dir: Optional[str] = None
) -> Optional[str]:
    """Download and extract model files from API server.
    
    Automatically handles authentication with the API server.
    Uses cache to avoid re-downloading the same model.
    
    Args:
        model_cid: Model file CID (hash ID)
        private_key: Ethereum private key for authentication
        api_server_url: API server base URL
        cache_dir: Directory for caching models (defaults to system temp dir)
        
    Returns:
        Path to extracted model directory, or None if download failed
    """
    try:
        import requests
        
        # 1. Check if model already cached
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "darvin_models")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use model_cid as cache key
        cached_model_dir = os.path.join(cache_dir, model_cid[:16])  # Use first 16 chars
        
        if os.path.exists(cached_model_dir) and os.listdir(cached_model_dir):
            logger.info(f"Using cached model: {cached_model_dir}")
            return cached_model_dir
        
        # 2. Model not cached, need to download
        logger.info(f"Model not cached, downloading: {model_cid}")
        
        # Auto-login to get JWT token
        logger.info("Authenticating for model download...")
        jwt_token = login_and_get_token(private_key, api_server_url)
        if not jwt_token:
            logger.error("Failed to authenticate with API server")
            return None
        
        download_url = f"{api_server_url}/api/files/download"
        headers = {'Authorization': f'Bearer {jwt_token}'}
        
        # 3. Download model zip file
        logger.info(f"Downloading model: {model_cid}")
        response = requests.get(
            download_url,
            params={'cid': model_cid},
            headers=headers,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code != 200:
            logger.error(f"Download failed: HTTP {response.status_code}")
            return None
        
        # 4. Create temp directory and save zip
        temp_dir = tempfile.mkdtemp(prefix=f'model_download_{model_cid[:8]}_')
        zip_path = os.path.join(temp_dir, "model.zip")
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded {len(response.content)} bytes")
        
        # 5. Verify zip file
        if not zipfile.is_zipfile(zip_path):
            logger.error("Downloaded file is not a valid zip file")
            shutil.rmtree(temp_dir)
            return None
        
        # 6. Extract model files to cache directory
        os.makedirs(cached_model_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            zip_ref.extractall(cached_model_dir)
            logger.info(f"Extracted {len(file_list)} files to cache: {cached_model_dir}")
        
        # 7. Clean up temp directory
        shutil.rmtree(temp_dir)
        
        return cached_model_dir
        
    except zipfile.BadZipFile:
        logger.error("Corrupted zip file")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None
    except Exception as exc:
        logger.error(f"Download error: {exc}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None
