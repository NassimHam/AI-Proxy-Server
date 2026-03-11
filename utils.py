import json
from typing import Any, Dict, List


def normalize_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs or []:
        if isinstance(m, dict):
            role = (m.get('role') or m.get('speaker') or 'user')
            if str(role).lower() == 'system':
                continue
            content = m.get('content') or m.get('message') or ''
            if not content and isinstance(m.get('message'), dict):
                content = m['message'].get('content') or ''
            out.append({'role': role, 'content': content})
    return out


def sse_assemble(text: str) -> str:
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

    return assembled
