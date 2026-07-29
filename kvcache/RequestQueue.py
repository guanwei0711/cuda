from dataclasses import dataclass
from collections import deque
from typing import Optional
from torch import Tensor

@dataclass
class Request:
    req_id: int
    prompt: Tensor
    max_decode_tokens: int
    terminal_len: int # simulating EOS token

class RequestQueue:
    def __init__(self, capacity: int):
        self._queue: deque[Request] = deque(maxlen=capacity)
        self.capacity = capacity

    def __len__(self):
        return len(self._queue)

    def enqueue(self, req: Request) -> bool:
        if len(self._queue) >= self.capacity:
            return False
        self._queue.append(req)
        return True

    def requeue(self, req: Request) -> None:
        self._queue.appendleft(req)

    def dequeue(self) -> Request:
        return self._queue.popleft()
