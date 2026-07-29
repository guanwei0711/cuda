from dataclasses import dataclass, field, InitVar
from sortedcontainers import SortedDict
import torch
from torch import Tensor

@dataclass
class ContiguousCachePool:
    max_size: int
    dim: int
    device: InitVar[object] = None
    dtype: InitVar[torch.dtype] = torch.float32
    k: Tensor = field(init=False, repr=False)
    v: Tensor = field(init=False, repr=False)
    _occupied_entries: SortedDict = field(default_factory=SortedDict, init=False)

    def __post_init__(self, device, dtype):
        self.k = torch.empty(self.max_size, self.dim, device=device, dtype=dtype)
        self.v = torch.empty(self.max_size, self.dim, device=device, dtype=dtype)
        
        self._occupied_entries[self.max_size] = 1
    def alloc(self, size: int) -> int:
        cand_start = 0
        for occupied_start, occupied_size in self._occupied_entries.items():
            if cand_start + size <= occupied_start:
                self._occupied_entries[cand_start] = size
                return cand_start
            cand_start = occupied_start + occupied_size

        raise RuntimeError("no available memory")

    def release(self, start: int) -> None:
        self._occupied_entries.pop(start)

    @property
    def capacity(self) -> int:
        return self.max_size

@dataclass
class ContiguousKVCache:
    max_length: int
    pool: ContiguousCachePool
    length: int = 0
    pool_mem_start: int = field(init=False)

    def __post_init__(self):
        self.pool_mem_start = self.pool.alloc(self.max_length)

    def append(self, k: Tensor, v: Tensor):
        if self.length == self.max_length:
            raise RuntimeError("reserved memory exhausted")
        add_len = k.shape[0]
        self.pool.k[self.pool_mem_start + self.length: self.pool_mem_start + self.length + add_len] = k
        self.pool.v[self.pool_mem_start + self.length: self.pool_mem_start + self.length + add_len] = v
        self.length += add_len

    def release(self):
        self.pool.release(self.pool_mem_start)

    @property
    def keys(self) -> Tensor:
        return self.pool.k[self.pool_mem_start:self.pool_mem_start + self.length]

    @property
    def values(self) -> Tensor:
        return self.pool.v[self.pool_mem_start:self.pool_mem_start + self.length]
