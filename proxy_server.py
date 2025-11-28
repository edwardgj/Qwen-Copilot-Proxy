import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage
from typing import AsyncGenerator

# Load version from VERSION file
def get_version() -> str:
    try:
        version_path = Path(__file__).parent / "VERSION"
        return version_path.read_text().strip()
    except:
        return "1.1.0"  # Fallback version

# --- Constants from Cline's source code ---
QWEN_OAUTH_BASE_URL = "https://chat.qwen.ai"
QWEN_OAUTH_TOKEN_ENDPOINT = f"{QWEN_OAUTH_BASE_URL}/api/v1/oauth2/token"
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_DIR = ".qwen"
QWEN_CREDENTIAL_FILENAME = "oauth_creds.json"
TOKEN_REFRESH_BUFFER_MS = 30 * 1000  # 30s buffer

# Global variable to cache credentials
credentials = None

def get_qwen_cached_credential_path() -> Path:
    home_dir = Path.home()
    return home_dir / QWEN_DIR / QWEN_CREDENTIAL_FILENAME

async def load_cached_qwen_credentials():
    global credentials
    cred_path = get_qwen_cached_credential_path()
    try:
        if not cred_path.exists():
            raise RuntimeError(f"Qwen OAuth credentials file not found at {cred_path}")

        creds_str = await asyncio.to_thread(cred_path.read_text)
        credentials = json.loads(creds_str)

        # Validate required fields
        required_fields = ["access_token", "refresh_token", "token_type", "expiry_date"]
        missing_fields = [field for field in required_fields if field not in credentials]
        if missing_fields:
            raise RuntimeError(f"Invalid OAuth credentials: missing required fields: {missing_fields}")

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in OAuth credentials file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen OAuth credentials from {cred_path}: {e}")

def is_token_valid(creds: dict) -> bool:
    """Check if the token is still valid with buffer time."""
    if not creds.get("expiry_date"):
        return False
    current_time_ms = time.time() * 1000
    return current_time_ms < creds["expiry_date"] - TOKEN_REFRESH_BUFFER_MS

async def refresh_access_token():
    global credentials
    if not credentials or not credentials.get("refresh_token"):
        raise RuntimeError("No refresh token available in credentials. Please re-authenticate with Qwen.")

    refresh_token = credentials["refresh_token"]
    body_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": QWEN_OAUTH_CLIENT_ID,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                QWEN_OAUTH_TOKEN_ENDPOINT,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json"
                },
                data=body_data,
            )
            response.raise_for_status()
            token_data = response.json()

    except httpx.HTTPStatusError as e:
        error_msg = f"Token refresh failed with HTTP {e.response.status_code}"
        try:
            error_data = e.response.json()
            error_msg += f": {error_data.get('error', 'Unknown error')} - {error_data.get('error_description', '')}"
        except:
            error_msg += f": {e.response.text}"
        raise RuntimeError(error_msg)
    except httpx.RequestError as e:
        raise RuntimeError(f"Token refresh failed due to network error: {e}")

    if "error" in token_data:
        raise RuntimeError(f"Token refresh failed: {token_data.get('error')} - {token_data.get('error_description')}")

    # Validate required fields in response
    required_response_fields = ["access_token", "token_type", "expires_in"]
    missing_fields = [field for field in required_response_fields if field not in token_data]
    if missing_fields:
        raise RuntimeError(f"Invalid token refresh response: missing fields: {missing_fields}")

    new_credentials = {
        **credentials,
        "access_token": token_data["access_token"],
        "token_type": token_data["token_type"],
        "refresh_token": token_data.get("refresh_token", refresh_token),
        "expiry_date": int(time.time() * 1000) + token_data["expires_in"] * 1000,
    }

    try:
        cred_path = get_qwen_cached_credential_path()
        # Ensure the directory exists
        cred_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(cred_path.write_text, json.dumps(new_credentials, indent=2))
        credentials = new_credentials
        print("Successfully refreshed Qwen access token")
    except Exception as e:
        raise RuntimeError(f"Failed to save refreshed credentials: {e}")

async def ensure_authenticated():
    global credentials
    try:
        if not credentials:
            await load_cached_qwen_credentials()
            print("Loaded Qwen OAuth credentials")

        if not is_token_valid(credentials):
            print("Qwen token expired or close to expiring. Refreshing...")
            await refresh_access_token()

    except Exception as e:
        print(f"Authentication error: {e}")
        raise

def get_base_url() -> str:
    if not credentials:
        raise RuntimeError("Credentials not loaded.")

    base_url = credentials.get("resource_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        base_url = f"https://{base_url}"
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"

# --- FastAPI App ---
app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Qwen-Copilot-Proxy is running",
        "status": "active",
        "models": ["qwen3-coder-plus", "vision-model"],
        "version": "0.6.4"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check if credentials are loaded and valid
        if not credentials:
            return {"status": "unhealthy", "reason": "No credentials loaded"}

        if not is_token_valid(credentials):
            return {"status": "degraded", "reason": "Token expired or needs refresh"}

        return {
            "status": "healthy",
            "authenticated": True,
            "models_supported": ["qwen3-coder-plus", "vision-model"]
        }
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

@app.get("/api/ps")
async def list_running_models():
    # Return empty list as we don't actually run models locally
    return {"models": []}

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("🚀 Starting Qwen-Copilot-Proxy Server")
    print("=" * 50)

    try:
        print("🔐 Checking Qwen OAuth authentication...")
        await ensure_authenticated()
        print("✅ Successfully authenticated with Qwen")
        print("📋 Available models:")
        print("   • qwen3-coder-plus (Code generation with tool support)")
        print("   • vision-model (Code + Vision with tool support)")
        print("🌐 Server will be available at: http://localhost:11434")
        print("=" * 50)
        print("✅ Qwen proxy is ready and authenticated.")
    except Exception as e:
        print("❌ Failed to authenticate with Qwen:")
        print(f"   Error: {e}")
        print("Please ensure your Qwen OAuth credentials are properly set up.")
        print("See README.md for setup instructions.")
        print("=" * 50)
        # Optionally, shut down if auth fails
        # exit(1)

@app.get("/api/version")
async def get_ollama_version():
    # Return Ollama version for compatibility (0.6.4 minimum requirement)
    return {"version": "0.6.4"}

@app.get("/version")
async def get_proxy_version():
    # Return proxy version information
    return {
        "proxy_version": get_version(),
        "ollama_compatibility": "0.6.4",
        "supported_models": ["qwen3-coder-plus", "vision-model"]
    }

# Add this route to provide a list of available models to the client
@app.get("/api/tags")
@app.get("/api/list")
async def list_models():
    # Provide a static list of models that the Qwen proxy supports
    # Match Ollama's exact response format
    return {
        "models": [
            {
                "name": "qwen3-coder-plus",
                "model": "qwen3-coder-plus",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 0,
                "digest": "qwen3-coder-plus",
                "details": {
                    "format": "gguf",
                    "family": "qwen",
                    "families": ["qwen"],
                    "parameter_size": "0B",
                    "quantization_level": "Q4_0"
                }
            },
            {
                "name": "vision-model",
                "model": "vision-model",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 0,
                "digest": "vision-model",
                "details": {
                    "format": "gguf",
                    "family": "qwen",
                    "families": ["qwen"],
                    "parameter_size": "0B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }

# Add this route to handle potential model info requests from Copilot
@app.post("/api/show")
async def show_model(request: Request):
    await ensure_authenticated() # Ensure token is fresh before making calls
    try:
        body = await request.json()
        model_name = body.get("model")
    except:
        # If we can't parse the body, return a default response
        model_name = "qwen3-coder-plus"

    # Determine capabilities based on model
    if model_name == "vision-model":
        capabilities = ["tools", "vision"]  # Vision model supports both tools and vision
        context_length = 32768
    else:
        # Default to qwen3-coder-plus capabilities
        capabilities = ["tools"]  # Coder model supports tools but not vision
        context_length = 32768

    # Return Ollama-like model information
    # Provide detailed model info that matches Ollama's format
    return {
        "template": "{{ .System }}\n{{ .Prompt }}",
        "capabilities": capabilities,
        "details": {
            "family": "qwen",
            "families": ["qwen"],
            "format": "gguf",
            "parameter_size": "0B",
            "quantization_level": "Q4_0"
        },
        "model_info": {
            "general.basename": model_name,
            "general.architecture": "qwen",
            "qwen.context_length": context_length
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await ensure_authenticated()

    try:
        body = await request.json()
    except Exception as e:
        raise RuntimeError(f"Invalid JSON request body: {e}")

    # Extract OpenAI-compatible request details
    model = body.get("model")
    messages = body.get("messages")
    stream = body.get("stream", False)

    # Validate request
    if not model:
        raise RuntimeError("Model name is required")
    if not messages or not isinstance(messages, list):
        raise RuntimeError("Messages field is required and must be an array")

    # Verify that the model is supported
    supported_models = ["qwen3-coder-plus", "vision-model"]
    if model not in supported_models:
        raise RuntimeError(f"Unsupported model: {model}. Supported models: {', '.join(supported_models)}")

    # Use httpx for the client, mimicking the openai library call
    async def generate_chunks() -> AsyncGenerator[bytes, None]:
        """Generate streaming response with retry logic."""
        max_retries = 2
        retry_count = 0

        while retry_count <= max_retries:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    # Prepare request to Qwen's OpenAI-compatible endpoint
                    qwen_url = f"{get_base_url()}/chat/completions"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {credentials['access_token']}"
                    }

                    # Forward the entire request body as-is (they are both OpenAI-compatible)
                    response = await client.post(qwen_url, headers=headers, json=body, timeout=None)
                    response.raise_for_status()

                    # Stream response back
                    async for chunk in response.aiter_bytes():
                        yield chunk
                    break  # Success, exit retry loop

            except httpx.HTTPStatusError as e:
                # Handle 401 specifically for retries, mimicking Cline's logic
                if e.response.status_code == 401 and retry_count < max_retries:
                    print(f"Qwen token expired during stream (attempt {retry_count + 1}), attempting refresh and retry.")
                    try:
                        await refresh_access_token()
                        retry_count += 1
                        continue  # Retry with new token
                    except Exception as refresh_error:
                        print(f"Failed to refresh token: {refresh_error}")
                        raise RuntimeError(f"Authentication failed and token refresh failed: {refresh_error}")
                else:
                    # For other HTTP errors or max retries reached
                    error_details = f"HTTP {e.response.status_code}"
                    try:
                        error_data = e.response.json()
                        error_details += f": {error_data.get('error', {}).get('message', e.response.text)}"
                    except:
                        error_details += f": {e.response.text[:200]}"
                    raise RuntimeError(f"Qwen API error: {error_details}")

            except httpx.RequestError as e:
                if retry_count < max_retries:
                    print(f"Network error during streaming (attempt {retry_count + 1}): {e}. Retrying...")
                    retry_count += 1
                    continue
                else:
                    raise RuntimeError(f"Network error after {max_retries + 1} attempts: {e}")

            except Exception as e:
                raise RuntimeError(f"Unexpected error during streaming: {e}")

    if stream:
        return StreamingResponse(generate_chunks(), media_type="text/event-stream")
    else:
        # Handle non-streaming requests with retry logic
        max_retries = 2
        retry_count = 0

        while retry_count <= max_retries:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    qwen_url = f"{get_base_url()}/chat/completions"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {credentials['access_token']}"
                    }

                    response = await client.post(qwen_url, headers=headers, json=body)
                    response.raise_for_status()
                    return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and retry_count < max_retries:
                    print(f"Qwen token expired during request (attempt {retry_count + 1}), attempting refresh and retry.")
                    try:
                        await refresh_access_token()
                        retry_count += 1
                        continue
                    except Exception as refresh_error:
                        print(f"Failed to refresh token: {refresh_error}")
                        raise RuntimeError(f"Authentication failed and token refresh failed: {refresh_error}")
                else:
                    error_details = f"HTTP {e.response.status_code}"
                    try:
                        error_data = e.response.json()
                        error_details += f": {error_data.get('error', {}).get('message', e.response.text)}"
                    except:
                        error_details += f": {e.response.text[:200]}"
                    raise RuntimeError(f"Qwen API error: {error_details}")

            except httpx.RequestError as e:
                if retry_count < max_retries:
                    print(f"Network error during request (attempt {retry_count + 1}): {e}. Retrying...")
                    retry_count += 1
                    continue
                else:
                    raise RuntimeError(f"Network error after {max_retries + 1} attempts: {e}")

            except Exception as e:
                raise RuntimeError(f"Unexpected error during request: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("proxy_server:app", host="localhost", port=11434, reload=True)
