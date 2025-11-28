# Changelog

All notable changes to Qwen-Copilot-Proxy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-11-28

### Added
- **Vision Model Support**: Added support for `vision-model` (qwen3-vl-plus-2025-09-23) with full vision capabilities
- **Enhanced Error Handling**: Comprehensive error handling with automatic retry logic (up to 3 attempts)
- **Robust OAuth Token Management**: Improved token validation, automatic refresh with 30-second buffer, and better error messages
- **Health Monitoring**: Added `/health` endpoint for monitoring authentication status and model availability
- **Up to date Documentation**: Complete README overhaul with technical explanations and proper model specifications
- **Model Capabilities**: Dynamic capability reporting (tools, vision support) per model
- **Version Information**: Added `/version` endpoint for proxy version and compatibility information
- **Better Logging**: Enhanced startup logging with detailed status information
- **Request Validation**: Input validation for model names and required fields

### Fixed
- **Model Name Resolution**: Fixed vision model name from `qwen3-vl-plus` to correct API identifier `vision-model`
- **OAuth Credential Handling**: Better error handling for missing or invalid credential files
- **Streaming Reliability**: Improved streaming error recovery and reconnection logic
- **Network Resilience**: Better handling of network failures with exponential backoff

### Improved
- **Performance**: More efficient token management and request handling
- **User Experience**: Clear error messages and better troubleshooting guidance
- **Code Quality**: Incorporated best practices from both Cline's Qwen implementation and GitHub Copilot's Ollama provider
- **Monitoring**: Real-time status reporting and authentication state tracking

## [1.0.0] - Initial Release

### Added
- **Basic Proxy Functionality**: Ollama API compatibility layer for Qwen Code models
- **qwen3-coder-plus Support**: Initial support for code generation model
- **OAuth Authentication**: Basic Qwen OAuth2 token management
- **Ollama API Endpoints**: Core `/api/tags`, `/api/show`, `/v1/chat/completions` endpoints
- **Streaming Support**: Basic streaming response handling
- **Configuration**: Simple configuration with OAuth credential file support

### Technical Details
- FastAPI-based proxy server
- OAuth2 token refresh mechanism
- OpenAI-compatible chat completion interface
- Local port 11434 (Ollama default)