# Qwen-Copilot-Proxy

A simple proxy server that enables Qwen Code models to work with GitHub Copilot Chat by mimicking the Ollama API interface.

## What it does

This proxy server acts as a bridge between GitHub Copilot Chat and Qwen's Code models API. It fools Copilot Chat into thinking it's communicating with Ollama while actually forwarding requests to Qwen's API.

```
GitHub Copilot Chat ↔️ localhost:1434 (Ollama endpoint) 
                              ↓
                    Qwen-Copilot-Proxy Server
                              ↓
                      Qwen API (OAuth2)
```

## Quick Start

### Prerequisites

1. **Qwen Account**: You need to install the Qwen-code CLI and create an account with the OAuth option to get access to Qwen-Code models
2. **Qwen OAuth Credentials**: Stored in `~/.qwen/oauth_creds.json`
3. **Python 3.8+**: Required to run the proxy server

### Installation

1. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv qwen-proxy-venv
   source qwen-proxy-venv/bin/activate  # On Windows: qwen-proxy-venv\Scripts\activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Proxy

1. **Start the proxy server:**
   ```bash
   python proxy_server.py
   ```

2. **The server will start on `http://localhost:11434`** (same port as Ollama, make sure ollama is not running at the time you run the script, else you'll get errors because you will be trying to run two services on the same port)

3. **Configure GitHub Copilot Chat:**
   - Open VS Code GitHub Copilot Chat
   - Click the name of the model you're currently running to see the list of available models
   - Click 'Manage Models...'
   - A screen showing you the list of providers will appear, choose 'ollama'
   - Then a screen letting you choose the model will appear, choose 'qwen3-coder-plus'

## Supported Models

- `qwen3-coder-plus`

## Troubleshooting

**Common Issues:**

1. **"Failed to authenticate with Qwen"**
   - Verify your OAuth credentials file exists
   - Check that the file is at `~/.qwen/oauth_creds.json`(different path on windows, if you're running this on Windows, you may need to edit the script for it to use the correct Path)

2. **"Connection refused" in Copilot Chat**
   - Make sure the proxy server is running
   - Check that no other service is using port 11434

3. **Network errors**
   - These are normal and can happen with any internet connection
   - Simply retry the request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
