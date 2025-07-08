import asyncio
import concurrent
import inspect
import threading
import time
import traceback

from stpr import _debug
from stpr._debug import debug_print, Color


_LOOP = asyncio.new_event_loop()


class _LoopThread(threading.Thread):
    def __init__(self):
        super().__init__(name='STPR Loop')
        self.daemon = True

    def run(self):
        try:
            _LOOP.run_forever()
        except:
            _LOOP.close()


_LOOP_THREAD = _LoopThread()
_LOOP_THREAD.start()


async def wait(delay: int):
    await asyncio.sleep(delay)


def _start(fn) -> concurrent.futures.Future:
    debug_print(type(fn), Color.BLUE)
    _debug._TS = time.time()

    if inspect.iscoroutine(fn):
        coro = fn
    elif callable(fn):
        coro = fn()
    else:
        raise ValueError('Cannot run %s' % fn)

    #print(f'running {coro} in {id(_LOOP)}')
    return asyncio.run_coroutine_threadsafe(coro, _LOOP)


def run(fn) -> object | None:
    """
    Runs a coroutine.

    The coroutine (or coroutine function) is scheduled for execution and waited for synchronously.
    For example:

    .. code-block:: Python

        async def f(x: float) -> float:
            return x * x + 5

        r = stpr.run(f(2))
        print(r)  # 9

    If `fn` raises an otherwise uncaught exception, this function re-raises that exception.

    :param fn: A coroutine or coroutine function to run.
    :return: This function returns the result returned by the coroutine.
    """
    future = _start(fn)
    return future.result()


def _done_callback(f: concurrent.futures.Future) -> None:
    exception = f.exception(timeout=0)
    if exception is not None:
        print(f'Exception: {exception}')
        traceback.print_exception(exception)


def start(fn) -> None:
    """
    Starts a coroutine.

    This method schedules `fn` for execution. The argument must be either a coroutine or a coroutine
    function. This function returns immediately after scheduling `fn`. If `fn` raises an exception
    that is not caught, the exception is printed on the console.

    :param fn: A coroutine or coroutine function to start.
    """
    future = _start(fn)
    future.add_done_callback(_done_callback)


def async_partial(f, /, *args, **kwargs):
    """
    Like :func:`functools.partial`, except for async functions.

    :param f: The function to apply partially.
    :param args: The arguments to apply.
    :param kwargs: The keyword arguments to apply.
    :return: The function f with the specified arguments partially applied.
    """
    async def w(*dargs, **dkwargs):
        xkwargs = {**kwargs, **dkwargs}
        #print(f'ap: {f}({args}, {dargs}, {xkwargs})')
        try:
            return await f(*args, *dargs, **xkwargs)
        except TypeError as e:
            raise TypeError(f'Failed to invoke callback {f}{inspect.signature(f)}: {e}')

    return w


async def exec(executable: str, *args: str, stdin=None, stdout=None, stderr=None, **kwargs):
    return await asyncio.create_subprocess_exec(executable, *args, stdin=stdin, stdout=stdout,
                                                stderr=stderr, **kwargs)