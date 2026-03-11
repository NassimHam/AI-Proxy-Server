import json
import httpx
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app import app
from config import REMOTE_API_URL, DEFAULT_MODEL, DEFAULT_TEMPLATE, HTTP_TIMEOUT
from models import ChatPayload
from utils import normalize_messages, sse_assemble


@app.post("/proxy/chat")
async def proxy_to_remote_api(user_input: ChatPayload):
    payload_to_remote = user_input.model_dump()
    msgs = payload_to_remote.get("messages", [])
    payload_to_remote["messages"] = [m for m in msgs if (m.get("role") or "").lower() != "system"]

    target_headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        try:
            remote_response = await client.post(
                REMOTE_API_URL,
                json=payload_to_remote,
                headers=target_headers,
                timeout=HTTP_TIMEOUT,
            )

            if remote_response.status_code == 200:
                try:
                    return remote_response.json()
                except (ValueError, json.JSONDecodeError):
                    text = remote_response.text or ""
                    if text.lstrip().startswith("data:"):
                        assembled = sse_assemble(text)
                        return JSONResponse(content={"content": assembled})

                    media_type = remote_response.headers.get("content-type", "text/plain")
                    return PlainTextResponse(content=text, status_code=200, media_type=media_type)
            else:
                try:
                    error_json = remote_response.json()
                    return JSONResponse(status_code=remote_response.status_code, content=error_json)
                except (ValueError, json.JSONDecodeError):
                    return PlainTextResponse(content=remote_response.text or "", status_code=remote_response.status_code, media_type=remote_response.headers.get("content-type", "text/plain"))

        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to connect to remote API: {exc}")


@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: Request):
    body = await request.json()

    incoming_auth = request.headers.get("authorization")
    headers = {"Content-Type": "application/json"}
    if incoming_auth:
        headers["Authorization"] = incoming_auth

    stream_requested = bool(body.get("stream"))

    forward_body = {
        'messages': normalize_messages(body.get('messages') or []),
        'model': body.get('model') or DEFAULT_MODEL,
        'template': body.get('template') or body.get('template_name') or DEFAULT_TEMPLATE,
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            if stream_requested:
                remote_stream = await client.post(REMOTE_API_URL, json=forward_body, headers=headers, stream=True)

                if remote_stream.status_code >= 400:
                    err_text = await remote_stream.aread()
                    try:
                        err_json = json.loads(err_text)
                        return JSONResponse(status_code=remote_stream.status_code, content={"error": err_json})
                    except Exception:
                        return JSONResponse(status_code=remote_stream.status_code, content={"error": err_text.decode('utf-8', errors='ignore')})

                if remote_stream.headers.get("content-type", "").startswith("application/json") and not remote_stream.headers.get("content-type", "").startswith("text/event-stream"):
                    data = await remote_stream.aread()
                    try:
                        return JSONResponse(content=json.loads(data), status_code=remote_stream.status_code)
                    except Exception:
                        return PlainTextResponse(content=data.decode("utf-8", errors="ignore"), status_code=remote_stream.status_code)

                async def stream_generator():
                    try:
                        async for chunk in remote_stream.aiter_bytes():
                            if chunk:
                                yield chunk
                    except Exception as e:
                        yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n".encode('utf-8')

                media_type = remote_stream.headers.get("content-type", "text/event-stream")
                return StreamingResponse(stream_generator(), media_type=media_type, status_code=remote_stream.status_code)

            else:
                remote_resp = await client.post(REMOTE_API_URL, json=forward_body, headers=headers)
                if remote_resp.status_code >= 400:
                    text = await remote_resp.aread()
                    try:
                        return JSONResponse(status_code=remote_resp.status_code, content={"error": json.loads(text)})
                    except Exception:
                        return JSONResponse(status_code=remote_resp.status_code, content={"error": text.decode('utf-8', errors='ignore')})

                content_type = remote_resp.headers.get("content-type", "")
                if content_type.startswith("text/event-stream"):
                    raw = await remote_resp.aread()
                    text = raw.decode('utf-8', errors='ignore')
                    assembled = sse_assemble(text)

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

                try:
                    return JSONResponse(content=remote_resp.json(), status_code=remote_resp.status_code)
                except Exception:
                    return PlainTextResponse(content=remote_resp.text or "", status_code=remote_resp.status_code, media_type=remote_resp.headers.get("content-type", "text/plain"))

        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to connect to remote API: {exc}")


@app.get("/v1/models")
async def openai_models():
    models_url = REMOTE_API_URL.rstrip("/\n").replace('/chat', '/models')

    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            resp = await client.get(models_url)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, dict) and (data.get('data') or data.get('models')):
                        return JSONResponse(content=data)
                    if isinstance(data, list):
                        return JSONResponse(content={"data": data})
                except Exception:
                    pass
        except Exception:
            pass

    return JSONResponse(content={"data": [{"id": "dolphinserver:24B", "name": "dolphinserver:24B"}, {"id": "dp3:flash", "name": "GLM-4.7-Flash-Beta"}]})
