import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager

from stpr._debug import debug_print, Color
from stpr.functions import _LOOP


def _sync_initializer():
    # There are libraries with functions that assume they are called from a loop in a context
    # that does not indicate that they should be (e.g., a constructor). For those, we
    # set the running loop so that any setup involving that loop does not break things.
    asyncio._set_running_loop(_LOOP)


_EXEC = ThreadPoolExecutor(thread_name_prefix='STPR_SYNC', initializer=_sync_initializer)


def _run_sync(fn, *args, **kwargs):
    # TODO: since we often run sync pieces of code in the executor, both the future
    # wrapping and the thread pool submission should be fast. Furthermore, the thread
    # pool should scale both up and down.
    return asyncio.wait_for(asyncio.wrap_future(_EXEC.submit(fn, *args, **kwargs)), timeout=None)


def _start(fn):
    asyncio.create_task(fn)


class _Aiter:
    def __init__(self, iter):
        self.iter = iter.__iter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return self.iter.__next__()
        except StopIteration:
            raise StopAsyncIteration()


class _ACM:
    def __init__(self, cm):
        self.cm = cm

    async def __aenter__(self):
        return self.cm.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.cm.__exit__(exc_type, exc_val, exc_tb)


def _to_aiter(iter):
    if hasattr(iter, '__aiter__'):
        debug_print(f'No need to wrap {iter}', Color.MAGENTA)
        return iter
    else:
        debug_print(f'iter: {iter}', Color.MAGENTA)
        return _Aiter(iter)


def _to_acm(cm):
    if isinstance(cm, AbstractAsyncContextManager):
        return cm
    else:
        return _ACM(cm)


async def _await(obj):
    if asyncio.iscoroutine(obj):
        return await obj
    else:
        return obj


async def _call(fn, *args, **kwargs):
    if inspect.iscoroutine(fn) or inspect.iscoroutinefunction(fn):
        #print('Calling coro %s with %s, %s' % (fn, args, kwargs))
        return await fn(*args, **kwargs)
    else:
        #print('Wrap-calling %s with %s' % (fn, args))
        return await _run_sync(fn, *args, **kwargs)
