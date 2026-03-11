from typing import List

# Remote API target
REMOTE_API_URL = "https://chat.dphn.ai/api/chat"

# CORS configuration for the FastAPI app
CORS_ALLOW_ORIGINS: List[str] = ["*"]

# Defaults used by the proxy
DEFAULT_MODEL = "dolphinserver:24B"
DEFAULT_TEMPLATE = "creative"

# HTTP client timeout (seconds)
HTTP_TIMEOUT = 60.0
