from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    role: str
    content: str


class ChatPayload(BaseModel):
    messages: List[Message]
    model: str = "dolphinserver:24B"
    template: str = "creative"
