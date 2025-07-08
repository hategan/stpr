from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Union

from aiofile import async_open


class _FileChunksIter:
    def __init__(self, f, chunk_sz):
        self.f = f
        self.chunk_sz = chunk_sz

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self.f.read(self.chunk_sz)
        if chunk is None or len(chunk) == 0:
            raise StopAsyncIteration()
        return chunk


class _FileWrapper:
    def __init__(self, f):
        self.f = f

    def chunks(self, chunk_sz: int = 16384):
        return _FileChunksIter(self.f, chunk_sz)

    async def write(self, chunk) -> None:
        await self.f.write(chunk)

    async def close(self) -> None:
        await self.f.close()

    def __aiter__(self):
        return self.f.__aiter__()

    async def readline(self):
        return await self.f.readline()


class open(AbstractAsyncContextManager):
    def __init__(self, path: Union[str, Path], mode: str = 'r') -> None:
        self.path = path
        self.mode = mode

    async def __aenter__(self):
        self.f = await async_open(self.path, self.mode)
        return _FileWrapper(self.f)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.f.close()

