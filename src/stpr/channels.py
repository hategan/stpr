import asyncio
from typing import TypeVar, Tuple, Optional, AsyncIterator

import stpr
from stpr.types import T


class _Guard:
    def __init__(self, exception: Optional[Exception]) -> None:
        self.exception = exception


_GUARD = _Guard(StopAsyncIteration())


class Channel(AsyncIterator[T]):
    """
    This class implements a channel.

    A channel is an ordered collection that can be iterated over asynchronously. Values consumed
    from a channel are removed completely and cannot be iterated over again. The typical
    scenario is that values are added to a channel in a thread and consumed in another:

    .. code-block:: Python

        @stpr.fn
        def consumer(c: Channel[int]) -> None:
            [async] for n in c:
                print(f'Received {n}')
            print('Channel closed')

        @stpr.fn
        def producer(c: Channel[int]) -> None:
            for n in range(10):
                c += n
                stpr.wait(1)
            c.close()

        @stpr.fn
        def run():
            c = Channel()
            with stpr.parallel:
                consumer(c)
                producer(c)

    Where `[async]` is enclosed in square brackets because it is optional. Iteration over a channel
    will continue as a long as the `close()` or `fail()` methods on the channel are not called.
    While `close()` and `fail()` can be invoked explicitly, it generally safer to use the channel
    as a context manager, which automatically takes care of closing and/or failing the channel:

    .. code-block:: Python

        @stpr.fn
        def producer(c: Channel[int]) -> None:
            with c:
                for n in range(10):
                    c += f(n)
                    stpr.wait(1)

    In the above example, the channel is closed automatically when the `with` statement completes.
    Any exceptions thrown by `f()` will propagate out of `producer()`, but also be raised in the
    consumer function after all other values in the channel are consumed.
    """
    def __init__(self):
        self._q = asyncio.Queue()

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        if exc_val is not None:
            self.fail(exc_val)
        else:
            self.close()

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        value = await self._q.get()
        if isinstance(value, _Guard):
            raise value.exception
        return value

    async def append(self, value: T) -> None:
        """
        Appends a value to this channel.

        :param value: The value to append.
        """
        await self._q.put(value)

    def _append_now(self, value: T) -> None:
        self._q.put_nowait(value)

    def __iadd__(self, value: T) -> None:
        self._q.put_nowait(value)

    def close(self) -> None:
        """
        Closes this channel.

        When the channel is closed and all previously added items are consumed, the channel will
        raise :py:exc:`StopAsyncIteration` gracefully terminating the iteration
        that is used to consume values from this channel.
        """
        self._q.put_nowait(_GUARD)

    def fail(self, exception: Exception) -> None:
        """
        Fails (poisons) this channel.

        When this method is called, a special values is added to the channel. When attempts are
        made to consume that value, ``exception`` is raised instead.
        """
        self._q.put_nowait(_Guard(exception))


async def select(*args: Channel[T]) -> Channel[Tuple[T, Channel[T]]]:
    """
    Implements a channel selector.

    A channel selector is a construct that will wait until a value is added to any one of the
    selected channels. For each such value in each channel, the returned channel will produce
    a tuple ``(value, channel)`` with the value added and the channel it was added to. For
    example:

    .. code-block:: Python

        @stpr.fn
        def select_example() -> None:
            a = Channel()
            b = Channel()

            s = stpr.select(a, b)

            with stpr.parallel:
                for value, channel in s:
                    print(f'{value} received on {channel}')

                with stpr.seq:
                    a.append(1)
                    b.append(2)
                    a.close()
                    b.close()

        stpr.run(select_example)

    This example will produce an output similar to the following:

    .. code-block:: shell

        1 received on <stpr.channels.Channel object at 0x77985d383620>
        2 received on <stpr.channels.Channel object at 0x77985bf6ef90>


    :param args: One or more channels to select from.
    :return: A channel with ``(value, channel)`` tuples corresponding to values added to the
        selected channels.
    """
    r = Channel()
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    async def select_one(c: Channel[T]) -> None:
        async for v in c:
            await r.append((v, c))

    async def selector() -> None:
        async with asyncio.TaskGroup() as tg:
            for c in args:
                tg.create_task(select_one(c))
        r.close()

    stpr.start(selector)

    return r

