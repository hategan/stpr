import asyncio
import collections
import concurrent
import threading
from asyncio import events
from typing import TypeVar


T = TypeVar('T')
"""A generic type alias."""


class Lock:
    """
    A thread and asyncio safe lock.

    It is recommended that the lock be used as a context manager. Synchronous functions should
    use the synchronous context manager protocol (i.e., `with lock`) whereas coroutines should
    use the asynchronous context manager protocol (i.e., `async with lock`). If used in a `stpr`
    function, the asynchronous protocol will be used automatically.

    Alternatively, the functions :meth:`acquire_sync`, :meth:`release_sync`,
    :func:`acquire_async`, and :func:`release_async` can be used explicitly.

    At this time, this lock is only re-entrant if re-entered either in the same thread or in an
    async context. It is not re-entrant if invoked from both a sync and async context. Consider
    the following example:

    .. code-block:: Python

        lock = stpr.Lock()

        def sync_function():
            with lock:
                ...

        @stpr.fn
        def async_function():
            with lock:
                sync_function()

        stpr.run(async_function)

    In the above case, `sync_function` is executed in a thread pool. This effectively causes
    `sync_function` to run in a thread that is different from the thread that executes the
    async loop and in which `async_function` is running. In this case, re-entrance is not easily
    distinguishable from the lock being acquired in two different threads.
    """
    def __init__(self):
        """
        Create a new `Lock` instance.

        :meth:`acquire_sync`
        """
        self._lock = threading.Lock()
        self._waiters = collections.deque()
        self._owner = None
        self._alock = asyncio.Lock()
        self._count = 0

    async def acquire_async(self) -> None:
        """
        Acquire the lock from an async context.
        """
        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            await self._alock.acquire()
            return
        else:
            with self._lock:
                if self._owner is None:
                    self._owner = me
                    self._count = 1
                    await self._alock.acquire()
                    return
                else:
                    loop = events.get_running_loop()
                    future = loop.create_future()
                    self._waiters.append((future, me))
            await future

    def acquire_sync(self) -> None:
        """
        Acquire the lock from a synchronous context.
        """
        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            return
        else:
            with self._lock:
                if self._owner is None:
                    self._owner = me
                    self._count = 1
                    return
                else:
                    future = concurrent.futures.Future()
                    self._waiters.append((future, me))
            future.result()

    def release_async(self) -> None:
        """
        Release the lock from an asynchronous context.

        The lock must have previously been acquired using :func:`acquire_async`.
        """
        me = threading.get_ident()
        with self._lock:
            if self._owner != me:
                raise RuntimeError('Lock is not acquired.')
            else:
                self._count -= 1
                self._alock.release()
                if self._count == 0:
                    if len(self._waiters) == 0:
                        self._owner = None
                        return
                    else:
                        (future, thread) = self._waiters.popleft()
                        self._owner = thread
                        self._count = 1
                        future.set_result(None)

    def release_sync(self) -> None:
        """
        Release the lock from a synchronous context.

        The lock must have previously been acquired using :func:`acquire_sync`.
        """
        me = threading.get_ident()
        with self._lock:
            if self._owner != me:
                raise RuntimeError('Lock is not acquired.')
            else:
                self._count -= 1
                if self._count == 0:
                    if len(self._waiters) == 0:
                        self._owner = None
                        return
                    else:
                        (future, thread) = self._waiters.popleft()
                        self._owner = thread
                        self._count = 1
                        future.set_result(None)

    async def __aenter__(self):
        await self.acquire_async()
        return None

    async def __aexit__(self, exc_type, exc_value, tb):
        self.release_async()

    def __enter__(self):
        self.acquire_sync()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_sync()


class Throttle:
    """
    Represents a throttle.

    A throttle is a mechanism that places an upper limit on the number of concurrent operations that
    can occur at a given point in the code. The following code shows an example of throttle use:

    .. code-block:: Python

        throttle = Throttle(5)

        @stpr.fn
        def process(i: int) -> None:
            with throttle:
                print(f'start {i}')
                stpr.wait(1)
                print(f'end {i}')

        @stpr.fn
        def throttle_example() -> None:
            with stpr.parallelFor(range(20)) as i:
                process(i)

    In this example, at most 5 concurrent instances of the `with` block in the `process` function
    will be allowed.

    At this time, `Throttle` only implements the async context manager. This is automatically
    used in `stpr` functions when using the `with` keyword.
    """
    def __init__(self, n: int) -> None:
        """
        :param n: Represents the maximum number of simultaneous parallel threads that this throttle
            will allow.
        """
        self.n = n
        self._queue = asyncio.Queue(maxsize=n)

    async def __aenter__(self):
        await self._queue.put(True)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._queue.get_nowait()


