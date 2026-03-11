import json
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Allow cross-origin requests from the Web UI (development-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION (from your images) ---
# Image 0: The target remote URL
REMOTE_API_URL = "https://chat.dphn.ai/api/chat"

# --- DATA STRUCTURES (Models) ---
# Image 1 shows a JSON payload. Pydantic models validate incoming data.
# This ensures we are sending the exact format the remote server expects.

class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    messages: List[Message]
    model: str = "dolphinserver:24B" # Defaulting from Image 1
    template: str = "creative"        # Defaulting from Image 1


# --- THE PROXY ENDPOINT ---
# Your local script will listen for a POST request on this path
@app.post("/proxy/chat")
async def proxy_to_remote_api(user_input: ChatPayload):
    """
    Receives user input locally, forwards it to the remote API,
    and relays the response.
    """

    # 1. Prepare the exact payload the remote API expects.
    # We use our validated model to ensure structure.
    # Use Pydantic V2 API to produce a dict representation
    payload_to_remote = user_input.model_dump()
    # Remove any 'system' messages when forwarding from local clients / UI
    msgs = payload_to_remote.get("messages", [])
    payload_to_remote["messages"] = [m for m in msgs if (m.get("role") or "").lower() != "system"]

    # 2. Setup the headers for the request to the remote server.
    # Image 0 shows 'Access-Control-Allow-Headers: Content-Type, Authorization'
    # The browser will handle CORS, but when making a direct backend-to-backend
    # call, we *must* manually set the Content-Type.
    # WARNING: See 'The Crucial Missing Authorization Token' section below.
    target_headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer YOUR_API_KEY_HERE" # (UNCOMMENT AND FIX THIS LATER)
    }

    # 3. Create an async client to send the request from the proxy
    # This prevents your local server from blocking while waiting for a response.
    async with httpx.AsyncClient() as client:
        try:
            # 4. Make the POST request to the remote API
            # Image 0 confirms this is a POST request.
            # We send the REMOTE_API_URL, target_headers, and our JSON payload.
            print(f"--> Forwarding request to remote API...")
            remote_response = await client.post(
                REMOTE_API_URL,
                json=payload_to_remote,
                headers=target_headers,
                timeout=60.0 # Remote AI models can take time to respond
            )

            # 5. Handle the remote response
            # Image 0 shows a "200 OK" status code. Let's check for success.
            if remote_response.status_code == 200:
                print(f"<-- Received successful (200 OK) response from remote API.")
                # Try to parse JSON; if remote returned plain text or an empty body,
                # relay the raw text back to the caller instead of raising.
                try:
                    return remote_response.json()
                except (ValueError, json.JSONDecodeError):
                    text = remote_response.text or ""
                    # Detect Server-Sent Events (SSE) / chunked lines starting with 'data:'
                    if text.lstrip().startswith("data:"):
                        assembled = ""
                        for line in text.splitlines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                continue
                            try:
                                obj = json.loads(payload)
                            except Exception:
                                # Non-JSON payload — skip or append raw
                                continue
                            # Extract incremental content from common fields
                            content_piece = None
                            try:
                                choices = obj.get("choices") if isinstance(obj, dict) else None
                                if choices and isinstance(choices, list) and len(choices) > 0:
                                    delta = choices[0].get("delta", {})
                                    if isinstance(delta, dict):
                                        content_piece = delta.get("content")
                                    if not content_piece and "message" in choices[0]:
                                        msg = choices[0].get("message", {})
                                        if isinstance(msg, dict):
                                            content_piece = msg.get("content")
                                if not content_piece:
                                    content_piece = obj.get("content") if isinstance(obj, dict) else None
                            except Exception:
                                content_piece = None

                            if content_piece:
                                assembled += content_piece

                        # Return structured JSON with the assembled assistant output
                        return JSONResponse(content={"content": assembled})

                    print("<-- Remote returned non-JSON response; relaying raw text.")
                    media_type = remote_response.headers.get("content-type", "text/plain")
                    return PlainTextResponse(content=text, status_code=200, media_type=media_type)
            else:
                print(f"<-- WARNING: Remote API returned non-200 status code: {remote_response.status_code}")
                # Try to parse JSON error body; fall back to raw text if parsing fails.
                try:
                    error_json = remote_response.json()
                    return JSONResponse(status_code=remote_response.status_code, content=error_json)
                except (ValueError, json.JSONDecodeError):
                    return PlainTextResponse(content=remote_response.text or "", status_code=remote_response.status_code, media_type=remote_response.headers.get("content-type", "text/plain"))

        except httpx.RequestError as exc:
            # If the remote API can't be reached at all, report a 500
            raise HTTPException(status_code=500, detail=f"Failed to connect to remote API: {exc}")


@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: Request):
    """OpenAI-compatible endpoint for the web UI. Forwards the body to the remote
    Dolphin API. If `stream: true` is set in the incoming body, this endpoint will
    stream the remote response back to the client (preserving SSE/event-stream).
    """
    body = await request.json()

    # Forward the Authorization header if the client provided one
    incoming_auth = request.headers.get("authorization")
    headers = {"Content-Type": "application/json"}
    if incoming_auth:
        headers["Authorization"] = incoming_auth

    # If the client requested streaming, stream the remote response back
    stream_requested = bool(body.get("stream"))

    # Map OpenAI-style body to the remote API shape the Dolphin server expects.
    # Ensure messages are [{role, content}, ...]
    def normalize_messages(msgs):
        out = []
        for m in msgs or []:
            if isinstance(m, dict):
                role = (m.get('role') or m.get('speaker') or 'user')
                # Skip system messages for the remote Dolphin API
                if str(role).lower() == 'system':
                    continue
                # OpenAI uses {role, content}
                content = m.get('content') or m.get('message') or ''
                # Some UIs put the actual text in m.message.content
                if not content and isinstance(m.get('message'), dict):
                    content = m['message'].get('content') or ''
                out.append({'role': role, 'content': content})
        return out

    forward_body = {
        'messages': normalize_messages(body.get('messages') or []),
        'model': body.get('model') or 'dolphinserver:24B',
        'template': body.get('template') or body.get('template_name') or 'creative',
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            if stream_requested:
                # Stream the remote response and proxy chunks directly
                print("--> Forwarding streaming request to remote API...", forward_body['model'])
                remote_stream = await client.post(REMOTE_API_URL, json=forward_body, headers=headers, stream=True)

                # If remote returned non-streaming JSON, return it normally
                if remote_stream.status_code >= 400:
                    # Read error body and surface it
                    err_text = await remote_stream.aread()
                    try:
                        err_json = json.loads(err_text)
                        return JSONResponse(status_code=remote_stream.status_code, content={"error": err_json})
                    except Exception:
                        return JSONResponse(status_code=remote_stream.status_code, content={"error": err_text.decode('utf-8', errors='ignore')})

                if remote_stream.headers.get("content-type", "").startswith("application/json") and not remote_stream.headers.get("content-type", "").startswith("text/event-stream"):
                    # Read and return JSON
                    data = await remote_stream.aread()
                    try:
                        return JSONResponse(content=json.loads(data), status_code=remote_stream.status_code)
                    except Exception:
                        return PlainTextResponse(content=data.decode("utf-8", errors="ignore"), status_code=remote_stream.status_code)

                # Otherwise, stream the response body as-is (SSE/event-stream)
                async def stream_generator():
                    try:
                        async for chunk in remote_stream.aiter_bytes():
                            if chunk:
                                yield chunk
                    except Exception as e:
                        # If streaming breaks, yield an error message as plain text
                        yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n".encode('utf-8')

                media_type = remote_stream.headers.get("content-type", "text/event-stream")
                return StreamingResponse(stream_generator(), media_type=media_type, status_code=remote_stream.status_code)
            else:
                # Non-streaming: forward request and return parsed JSON or raw text
                print("--> Forwarding non-streaming request to remote API...", forward_body['model'])
                remote_resp = await client.post(REMOTE_API_URL, json=forward_body, headers=headers)
                # If non-2xx, surface body as structured error
                if remote_resp.status_code >= 400:
                    text = await remote_resp.aread()
                    try:
                        return JSONResponse(status_code=remote_resp.status_code, content={"error": json.loads(text)})
                    except Exception:
                        return JSONResponse(status_code=remote_resp.status_code, content={"error": text.decode('utf-8', errors='ignore')})

                # If the remote service returned an event-stream (SSE) even though
                # the client didn't request streaming, convert the SSE chunks into
                # a single OpenAI-compatible JSON response so the UI can consume it.
                content_type = remote_resp.headers.get("content-type", "")
                if content_type.startswith("text/event-stream"):
                    raw = await remote_resp.aread()
                    text = raw.decode('utf-8', errors='ignore')
                    assembled = ""
                    for line in text.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            continue
                        try:
                            obj = json.loads(payload)
                        except Exception:
                            continue
                        # pull delta content or message content
                        piece = None
                        try:
                            choices = obj.get('choices') if isinstance(obj, dict) else None
                            if choices and isinstance(choices, list) and len(choices) > 0:
                                delta = choices[0].get('delta', {})
                                if isinstance(delta, dict):
                                    piece = delta.get('content')
                                if not piece and 'message' in choices[0]:
                                    msg = choices[0].get('message', {})
                                    if isinstance(msg, dict):
                                        piece = msg.get('content')
                            if not piece:
                                piece = obj.get('content') if isinstance(obj, dict) else None
                        except Exception:
                            piece = None
                        if piece:
                            assembled += piece

                    # Build an OpenAI-compatible response body
                    openai_like = {
                        "id": "chatcmpl-proxy",
                        "object": "chat.completion",
                        "model": forward_body.get('model'),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": assembled},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    return JSONResponse(content=openai_like)

                # Otherwise try to return JSON directly; if that fails, fall back to text
                try:
                    return JSONResponse(content=remote_resp.json(), status_code=remote_resp.status_code)
                except Exception:
                    return PlainTextResponse(content=remote_resp.text or "", status_code=remote_resp.status_code, media_type=remote_resp.headers.get("content-type", "text/plain"))

        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to connect to remote API: {exc}")


@app.get("/v1/models")
async def openai_models():
    """Return a models list compatible with the web UI's /models check.
    Attempts to query a remote /models endpoint; falls back to a default list.
    """
    # Try to query the remote service's models endpoint
    models_url = REMOTE_API_URL.rstrip("/\n")
    # Heuristic: replace '/chat' with '/models' if present
    models_url = models_url.replace('/chat', '/models')

    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            resp = await client.get(models_url)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    # Normalize to { data: [...] }
                    if isinstance(data, dict) and (data.get('data') or data.get('models')):
                        return JSONResponse(content=data)
                    # If it's a list, return as data
                    if isinstance(data, list):
                        return JSONResponse(content={"data": data})
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback model list
    return JSONResponse(content={"data": [{"id": "dolphinserver:24B", "name": "dolphinserver:24B"}, {"id": "dp3:flash", "name": "GLM-4.7-Flash-Beta"}]})


if __name__ == "__main__":
    # Start the server on localhost port 8000
    # You can then send requests to http://localhost:8000/proxy/chat
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)