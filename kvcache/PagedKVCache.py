from dataclasses import dataclass, field, InitVar
import torch
from torch import Tensor

@dataclass
class PagedCachePool:
    num_pages: int
    page_size: int
    dim: int
    def __post_init__(self, device=None, dtype=torch.float32):
        self.k = torch.empty(self.num_pages, self.page_size, self.dim, device=device, dtype=dtype)
        self.v = torch.empty(self.num_pages, self.page_size, self.dim, device=device, dtype=dtype)
        self._free_pages = list(range(self.num_pages))

    def alloc(self) -> int:
        if not self._free_pages:
            raise RuntimeError("no available pages")
        return self._free_pages.pop()

    def release(self, page_ids: list[int]) -> None:
        self._free_pages.extend(page_ids)

    @property
    def capacity(self) -> int:
        return self.num_pages * self.page_size

@dataclass
class PagedKVCache:
    pool: PagedCachePool

    init_len: InitVar[int]

    _page_ids: list[int] = field(default_factory=list)
    length: int = 0
    _allocated_length: int = 0

    def __post_init__(self, init_len: int):
        while self._allocated_length < init_len:
            self._page_ids.append(self.pool.alloc())
            self._allocated_length += self.pool.page_size

    def append(self, k: Tensor, v: Tensor):
        add_len = k.shape[0]
        while self._allocated_length < self.length + add_len:
            self._page_ids.append(self.pool.alloc())
            self._allocated_length += self.pool.page_size

        for i in range(add_len):
            pos = self.length + i
            page = pos // self.pool.page_size
            slot_id = pos % self.pool.page_size
            self.pool.k[self._page_ids[page], slot_id] = k[i]
            self.pool.v[self._page_ids[page], slot_id] = v[i]
        self.length += add_len

    def release(self):
        self.pool.release(self._page_ids)

    def _gather(self, target: Tensor):
        if self.length == 0:
            return target.new_empty(0, target.shape[-1])

        pages = target[self._page_ids] # deep copy because non-contiguous memory
        return pages.reshape(-1, pages.shape[-1])[:self.length]

    @property
    def keys(self) -> Tensor:
        return self._gather(self.pool.k)

    @property
    def values(self) -> Tensor:
        return self._gather(self.pool.v)
        