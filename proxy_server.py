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
        creds_str = await asyncio.to_thread(cred_path.read_text)
        credentials = json.loads(creds_str)
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen OAuth credentials: {e}")

async def refresh_access_token():
    global credentials
    if not credentials or not credentials.get("refresh_token"):
        raise ValueError("No refresh token available in credentials.")

    refresh_token = credentials["refresh_token"]
    body_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": QWEN_OAUTH_CLIENT_ID,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            QWEN_OAUTH_TOKEN_ENDPOINT,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data=body_data,
        )
        response.raise_for_status()
        token_data = response.json()

    if "error" in token_data:
        raise RuntimeError(f"Token refresh failed: {token_data.get('error')} - {token_data.get('error_description')}")

    new_credentials = {
        **credentials,
        "access_token": token_data["access_token"],
        "token_type": token_data["token_type"],
        "refresh_token": token_data.get("refresh_token", refresh_token),
        "expiry_date": int(time.time() * 1000) + token_data["expires_in"] * 1000,
    }

    cred_path = get_qwen_cached_credential_path()
    await asyncio.to_thread(cred_path.write_text, json.dumps(new_credentials, indent=2))
    credentials = new_credentials

async def ensure_authenticated():
    global credentials
    if not credentials:
        await load_cached_qwen_credentials()

    if time.time() * 1000 > credentials["expiry_date"] - TOKEN_REFRESH_BUFFER_MS:
        print("Qwen token expired or close to expiring. Refreshing...")
        await refresh_access_token()

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
    return {"message": "Ollama is running"}

@app.get("/api/ps")
async def list_running_models():
    # Return empty list as we don't actually run models locally
    return {"models": []}

@app.on_event("startup")
async def startup_event():
    try:
        await ensure_authenticated()
        print("Qwen proxy is ready and authenticated.")
    except Exception as e:
        print(f"Failed to authenticate with Qwen: {e}")
        # Optionally, shut down if auth fails
        # exit(1)

@app.get("/api/version")
async def get_version():
    # Return a version that meets or exceeds Ollama's minimum requirement (0.6.4)
    return {"version": "0.6.4"}

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

    # Return Ollama-like model information
    # Since we don't have actual Qwen model info, we'll provide a static response
    # that matches Ollama's format
    return {
        "template": "{{ .System }}\n{{ .Prompt }}",
        "capabilities": ["tools"],  # Qwen models support tool calling
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
            "qwen.context_length": 32768  # Default context length for Qwen models
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await ensure_authenticated()
    body = await request.json()

    # Extract OpenAI-compatible request details
    model = body.get("model")
    messages = body.get("messages")
    stream = body.get("stream", False)

    # Use httpx for the client, mimicking the openai library call
    async def generate_chunks() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
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

            except httpx.HTTPStatusError as e:
                # Handle 401 specifically for retries, mimicking Cline's logic
                if e.response.status_code == 401:
                    print("Qwen token expired during stream, attempting refresh and retry.")
                    await refresh_access_token()

                    # Re-create and re-send the request with new token
                    headers["Authorization"] = f"Bearer {credentials['access_token']}"
                    response = await client.post(qwen_url, headers=headers, json=body, timeout=None)
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
                else:
                    raise e

    if stream:
        return StreamingResponse(generate_chunks(), media_type="text/event-stream")
    else:
        # Handle non-streaming requests
        async with httpx.AsyncClient(timeout=300.0) as client:
            qwen_url = f"{get_base_url()}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {credentials['access_token']}"
            }
            response = await client.post(qwen_url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("proxy_server:app", host="localhost", port=11434, reload=True)
