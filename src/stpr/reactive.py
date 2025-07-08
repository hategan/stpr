"""
This module implements reactive values and functions. Reactive functions automatically update their
results whenever their inputs change. A basic usage scenario is when one wants to compute a
function on some data that is periodically updated:

.. code-block:: Python

    @stpr.reactive.fn
    def f(x: float) -> float:
        return math.sin(x)

    @stpr.fn
    main() -> None:
        x: Reactive[float] = Reactive()

        result = f(x)

        with stpr.parallel:
            for (r, value, old) in result.updates():
                print(f'Result updated from {old} to {value}.')
            for value in [0.1, 0.2, 0.3, 0.4]:
                x.update(value)
                stpr.wait(1.0)

    stpr.run(main)
"""

import asyncio
import inspect
import operator
import traceback

import stpr
from . import Channel
from ._debug import Color, _print
from .functions import async_partial, start
from .transformer import fn as stprfn
from .types import Lock, T
from typing import Callable, Awaitable, Optional, List, Generic, Union


class Undefined:
    """
    This class is the type of the :py:const:`UNDEFINED` value which is used to represent a reactive
    that has no particular value. Since ``None`` is a possible value of a reactive,
    :py:const:`UNDEFINED` is distinct from it.
    """
    def __str__(self) -> str:
        return 'UNDEFINED'

    def __repr__(self) -> str:
        return self.__str__()


UNDEFINED = Undefined()
"""A special value that indicates that a reactive does not have an explicit value."""


class Reactive(Generic[T]):
    """
    Implements a reactive.

    A reactive is a wrapper around other python objects that can be used to implement functions
    whose results get updated automatically when a reactive parameter changes. The main mechanism
    of updating a `Reactive` is by calling the :py:func:`Reactive.update` method.

    The base `Reactive` class overrides most operators to properly handle reactivity. For example:

    .. code-block:: Python

        a = Reactive()
        b = Reactive()

        r = a + b  # r is a reactive that is updated whenever a or b are.
    """
    def __init__(self, value: Union[T, Undefined] = UNDEFINED) -> None:
        """
        Creates a new reactive and optionally initializes it.

        By default, a new reactive will have a value of :py:const:`UNDEFINED`. A reactive with
        an :py:const:`UNDEFINED` value prevents reactive functions that depend on it from being
        invoked. Alternatively, the reactive can be initialized with a value using the ``value``
        parameter.

        :param value: A value to initialize this `Reactive` with.
        """

        self._callbacks = []
        self.value = value

    def add_callback(self, cb: Callable[['Reactive[T]', T, T], Awaitable[object]]) -> None:
        """
        Adds a callback to this reactive.

        A callback can be used to get asynchronous notifications when a `Reactive` is updated. This
        method is mostly used for implementing low-level functionality.

        Callbacks take three arguments: a `Reactive`, the new value, and the old value. The
        `Reactive` argument can be useful if multiplexing a single callback function with updates
        from multiple reactives.

        :param cb: A callback function to be called when this reactive is updated.
        """
        if cb is None:
            raise ValueError(f'Invalid callback {cb}')
        if not asyncio.iscoroutinefunction(cb):
            raise ValueError(f'Callback {cb} is not a coroutine.')
        self._callbacks.append(cb)

    async def update(self, value: T, force: bool = False) -> None:
        """
        Updates the value of this reactive.

        The update triggers updates to all functions that have this `Reactive` as input. In
        principle, if the updated value is the same as the existing value of this `Reactive`, the
        update is silently discarded. This behavior can be overridden with the `force` parameter.

        :param value: The value to update this `Reactive` with.
        :param force: Setting this to `True` forces an update even if the updated value is the same
            as the existing value of this `Reactive`.
        """
        old = self.value
        self.value = value
        if force or old != value:
            await self._notify(value, old)

    async def _notify(self, value: T, old: T) -> None:
        for cb in self._callbacks:
            try:
                await cb(self, value, old)
            except TypeError as e:
                raise TypeError(f'Failed to invoke callback {cb}: {e}')

    def chain(self, other: 'Reactive[T]') -> None:
        """
        Chains this reactive to another reactive.

        When chaining reactives, the chained reactive will be updated whenever this reactive is
        updated.

        :param other: The `Reactive` to chain.
        """
        async def chain_cb(r: 'Reactive[T]', value: T, old: T) -> None:
            await other.update(value)
        self.add_callback(chain_cb)

    def updates(self) -> Channel[T, T]:
        """
        Returns a channel that produces this reactive's updates.

        The returned channel can be iterated over. Each iteration produces a tuple
        `(new_val, old_val)` which represent the updated value and the old value of the reactive,
        respectively.

        :return:
        """
        channel = Channel()
        async def callback(r: 'Reactive[T]', value: T, old: T):
            channel.append((value, old))
        self.add_callback(callback)
        return channel

    def __lt__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.lt)

    def __le__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.le)

    def __eq__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.eq)

    def __ne__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.ne)

    def __ge__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.ge)

    def __gt__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.gt)

    def __not__(self) -> 'Reactive[T]':
        return _ReactiveUnaryOp(self, operator.not_)

    def __abs__(self) -> 'Reactive[T]':
        return _ReactiveUnaryOp(self, operator.abs)

    def __invert__(self) -> 'Reactive[T]':
        return _ReactiveUnaryOp(self, operator.inv)

    def __neg__(self) -> 'Reactive[T]':
        return _ReactiveUnaryOp(self, operator.neg)

    def __pos__(self) -> 'Reactive[T]':
        return _ReactiveUnaryOp(self, operator.pos)

    def __add__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.add)

    def __and__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.and_)

    def __lshift__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.lshift)

    def __mod__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.mod)

    def __mul__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.mul)

    def __matmul__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.matmul)

    def __or__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.or_)

    def __pow__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.pow)

    def __sub__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.sub)

    def __truediv__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.truediv)

    def __xor__(self, other: 'Reactive[T]') -> 'Reactive[T]':
        return _ReactiveBinOp(self, other, operator.xor)


class _ReactiveBinOp(Reactive[T]):
    def __init__(self, a: Reactive[T], b: Union[Reactive[T], T], op: Callable[[T, T], T]) -> None:
        super().__init__()
        self.va = a.value
        if isinstance(b, Reactive):
            self.vb = b.value
        else:
            self.vb = b
        self.op = op

        stpr._start(self._update_op())

        a.add_callback(self._cba)
        if isinstance(b, Reactive):
            b.add_callback(self._cbb)

    async def _update_op(self) -> None:
        if self.va == UNDEFINED or self.vb == UNDEFINED:
            return
        await self.update(self.op(self.va, self.vb))

    async def _cba(self, r: Reactive[T], value: T, old: T) -> None:
        self.va = value
        await self._update_op()

    async def _cbb(self, r: Reactive[T], value: T, old: T) -> None:
        self.vb = value
        await self._update_op()


class _ReactiveUnaryOp(Reactive[T]):
    def __init__(self, r: Reactive[T], op: Callable[[T], T]) -> None:
        super().__init__()
        self.op = op

        stpr._start(self._update_op())

        r.add_callback(self._cb)

    async def _update_op(self) -> None:
        if self.va == UNDEFINED or self.vb == UNDEFINED:
            return
        await self.update(self.op(self.va, self.vb))

    async def _cb(self, r: Reactive[T], value: T, old: T) -> None:
        self.update(self.op(value))


class Debounce(Reactive[T]):
    """
    Implements a de-bounced `Reactive`.

    The de-bounced reactive only updates if the source reactive has not changed its value for
    a pre-determined amount of time. This can be used to filter-out rapid changes in a reactive's
    value.
    """
    def __init__(self, r: Reactive[T], time: int, value: Union[T, Undefined] = UNDEFINED) -> None:
        """
        Constructs a new `Debounce`.

        :param r: The source `Reactive` which is to be de-bounced.
        :param time: An amount of time, in seconds, that the source reactive needs to have held
            single value for this instance to update.
        """

        super().__init__()
        self.time = time
        self._lock = Lock()
        r.add_callback(self._cb)
        self.seq = 0
        self.value = value
        stpr.start(self._cb2(r.value))

    async def _cb(self, r: Reactive, value: T, old: T) -> None:
        stpr._start(self._cb2(value))

    async def _cb2(self, value: T) -> None:
        with self._lock:
            self.seq += 1
            seq = self.seq
            old = self.value
        await stpr.wait(self.time)
        with self._lock:
            if self.seq == seq:
                await self._notify(value, old)

    async def update(self, value: T) -> None:
        raise TypeError('Debounce cannot be directly updated')


class Delay(Reactive[T]):
    """
    Allows delaying a reactive update.
    """
    def __init__(self, source: Reactive[T], time: int) -> None:
        """
        Constructs a new `Delay` instance.

        This reactive will receive the same updates as the source reactive. However, each update
        will happen `time` seconds after the source reactive was updated.

        :param source: A source reactive to follow.
        :param time: The delay, in seconds.
        """
        super().__init__()
        self.time = time
        self.r = source

    async def update(self, value: T) -> None:
        stpr._start(self._update_later(value))

    async def _update_later(self, value: T) -> None:
        await stpr.wait(self.time)
        await self.r.update(value)


class _RunningArgs:
    def __init__(self, f: object, arg_names: Optional[List[str]] = None) -> None:
        self.f = f
        self.arg_names = arg_names
        self.undefined = 0
        self.args = []
        self.kwargs = {}

    def add_arg_r(self, value: object) -> int:
        if value == UNDEFINED:
            self.undefined += 1
        ix = len(self.args)
        self.args.append(value)
        return ix

    def add_arg(self, value: object) -> None:
        self.args.append(value)

    def arr_kwarg_r(self, key: str, value: object) -> None:
        if value == UNDEFINED:
            self.undefined += 1
        self.kwargs[key] = value

    def add_kwarg(self, key: str, value: object) -> None:
        self.kwargs[key] = value

    def set_arg(self, ix: int, value: object) -> bool:
        if self.undefined != 0:
            if self.args[ix] == UNDEFINED:
                self.undefined -= 1
        self.args[ix] = value
        if self.arg_names:
            _print(f'Reactive[{self.f}]: set_arg({self.arg_names[ix]} = {value}), '
                   f'undef: {self.undefined}', color=Color.BRIGHT_WHITE, background=Color.RED)
            _print(f'Reactive[{self.f}]: {self._arg_vals()}', color=Color.BRIGHT_WHITE,
                   background=Color.RED)
        return self.undefined == 0

    def set_kwarg(self, key: str, value: object) -> bool:
        if self.undefined != 0:
            if self.kwargs[key] == UNDEFINED:
                self.undefined -= 1
        self.kwargs[key] = value
        if self.arg_names:
            _print(f'Reactive[{self.f}]: set_arg({key}={value}), '
                   f'undef: {self.undefined}', color=Color.BRIGHT_WHITE, background=Color.RED)
            _print(f'Reactive[{self.f}]: {self._arg_vals()}', color=Color.BRIGHT_WHITE,
                   background=Color.RED)
        return self.undefined == 0

    def _arg_vals(self) -> str:
        s = ''
        for ix in range(len(self.args)):
            s += f'{self.arg_names[ix]} = {self.args[ix]}, '
        for k, v in self.kwargs:
            s += f'{k} = {v}, '
        return s


def fn(*args, trace: bool = False):
    """
    A decorator that makes a function reactive.

    The decorated function will become reactive for any args (or kwargs) that are reactive. That is,
    whenever any of its reactive arguments are updated, and if all other reactive arguments have a
    value, it will be invoked. If the function returns a value, it will be wrapped in a `Reactive`
    and any invocation of the decorated function will update the returned reactive. All arguments
    that are not an instance of `Reactive` when the decorated function is invoked will be passed
    as-is to every reactive invocation of the function:

    .. code-block:: Python

        @stpr.reactive.fn
        def f(a: int, b: int, c: int) -> int:
            return a + b - c


        a = Reactive()
        b = Reactive()

        r = fn(a, b, 2)

    In the above example, `fn` will be invoked every time `a` or `b` are updated. The third
    parameter (`c`) will always be equal to `2`.

    The decorated function is automatically processed as if decorated by :py:func:`stpr.fn`, so
    :ref:`concurrency primitives <concurrency-primitives>` defined by Stpr can be used directly.

    If the `trace` argument to the decorator is set to `True`, invocations of the decorated function
    will be printed with details about the reactive that triggered the invocation.

    :param trace: If set to `True`, print traces of the invocation of the decorated function.
    """
    crt_frame = inspect.currentframe()

    def inner(f):
        if trace:
            _print(f'Reactive[{f}]: tracing enabled', color=Color.BRIGHT_WHITE, background=Color.RED)
            arg_names = inspect.getfullargspec(f)[0]
        else:
            arg_names = None
        f = stprfn(f, crt_frame=crt_frame)

        lock = Lock()

        ret = Reactive()

        async def invoke(*args, **kwargs):
            try:
                await ret.update(await f(*args, **kwargs))
            except Exception as ex:
                traceback.print_exception(ex)

        async def update_arg(running_args: _RunningArgs, ix: int, r: Reactive,
                             value: object, old: object) -> None:
            with lock:
                if running_args.set_arg(ix, value):
                    args = [*running_args.args]
                    kwargs = {**running_args.kwargs}
                else:
                    return
            if trace:
                _print(f'Reactive[{f}]: invoke due to {arg_names[ix]} -> {value}',
                             color=Color.BRIGHT_WHITE, background=Color.RED)
            await invoke(*args, **kwargs)

        async def update_kwarg(running_args: _RunningArgs, key: str, r: Reactive,
                               value: object, old: object) -> None:
            with lock:
                if running_args.set_kwarg(key, value):
                    args = [*running_args.args]
                    kwargs = {**running_args.kwargs}
                else:
                    return
            if trace:
                _print(f'Reactive[{f}]: invoke due to {key} -> {value}',
                             color=Color.BRIGHT_WHITE, background=Color.RED)
            await invoke(*args, **kwargs)

        def wrapper(*args, **kwargs):
            running_args = _RunningArgs(f, arg_names)
            with lock:
                # this runs once when the pipeline is set up
                for arg in args:
                    if isinstance(arg, Reactive):
                        ix = running_args.add_arg_r(arg.value)
                        arg.add_callback(async_partial(update_arg, running_args, ix))
                    else:
                        running_args.add_arg(arg)

                for key, value in kwargs.items():
                    if isinstance(value, Reactive):
                        running_args.add_kwarg_r(key, value.value)
                        arg.add_callback(async_partial(update_kwarg, running_args, key))
                    else:
                        running_args.add_kwarg(key, value)

                if running_args.undefined == 0:
                    start(invoke(*running_args.args, **running_args.kwargs))

            return ret

        return wrapper

    if len(args) > 1:
        raise TypeError()
    if len(args) == 0:
        return inner
    else:
        return inner(args[0])
