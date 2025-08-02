Getting Started
===============

This is a beginner's guide to the Stpr Python library. Stpr is a library
that makes concurrent programming using :py:mod:`asyncio` intuitive and
the resulting code maintainable.

Installation
------------

Using `PIP <https://packaging.python.org/en/latest/key_projects/#pip>`_:

.. code-block:: Bash

    pip install stpr


Why Stpr?
---------

Stpr allows a programmer to write scalable concurrent and asyncio based
programs while keeping the conceptual simplicity of synchronous programming.

Writing scalable concurrent software is difficult. Traditionally, this involves
the use of multiprocessing and/or threads, both of which are heavyweight
entities that are limited in number by the operating system. In Python,
the :py:mod:`asyncio` module, with support from the language, offers a
better solution. Let us look at a simple example. We start by writing a quick
HTTP server using the `aiohttp <https://docs.aiohttp.org/en/stable/>`_ library:

.. code-block:: Python

    import asyncio
    from aiohttp import web
    from threading import Thread

    async def handle(request):
        await asyncio.sleep(0.5)
        return web.Response(text='Hello from the async server')

    def run_server():
        app = web.Application()
        app.add_routes([web.get('/', handle),
                        web.get('/{name}', handle)])
        web.run_app(app, host='localhost', port=8080, 
                    handle_signals=False)

    Thread(target=run_server, daemon=True).start()

The server simulates a delay typical of a somewhat distant server somewhere
on the Internet. This simple server should be started so that the client examples
-that follow can talk to it. We want to compare code making concurrent requests to this
server using threads, plain ``asyncio`` and ``stpr``. A possible thread based
solution is:

.. code-block:: Python

    def threads_1():
        contents = urllib.request.urlopen(
            "http://localhost:8080").read()

    def threads_n(n):
        threads = []
        for i in range(n):
            t = Thread(target=threads_1)
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()

This allows us to run ``n`` concurrent requests. We can use the following
convenience function to time the execution and display the result:

.. code-block:: Python

    def time(fn, n):
        t = timeit.timeit(functools.partial(fn, n), number=1)
        print(f'{fn.__name__}({n}): {t:.2f}')

We can now run the ``threads`` method:

.. code-block:: Python

    for n in [1, 10, 100, 1000, 10000]:
        time(threads, n)

The result should be something like this:

.. code-block::

    threads_n(1): 0.54s
    threads_n(10): 0.51s
    threads_n(100): 0.54s
    threads_n(1000): 1.65s
    threads_n(10000): 16.26s

We see that the performance starts degrading around 1000 threads and becomes
significant at 10000 threads. What is not shown is that there is also a
significant penalty in terms of memory use. Each thread typically requires
8 megabytes for its stack, which means that 10000 threads will need somewhere
around 80 GB of virtual memory. Luckily, only a small portion of the thread
stack is used by the example above, and that portion fits in the physical
memory of a typical computer.

A possible ``asyncio`` based alternative is as follows:

.. code-block:: Python

    async def aio_1():
        async with ClientSession() as session:
            async with session.get('http://localhost:8080') as resp:
                contents = await resp.text()

    async def aio_n(n):
        async with asyncio.TaskGroup() as tg:
            for i in range(n):
                tg.create_task(aio_1())

    def aio_main(n):
        asyncio.run(aio_n(n))

A few things can be noted:

* Functions become coroutines, defined with ``async def``. Without the
  ``async def``, we would not be able to use ``async with`` or ``await`` in
  a function. 
* Perhaps subtle, we cannot use ``ClientSession`` with a plain ``with``. Doing
  so will produce an error.
* We must, at some point, ``await`` coroutines. Not doing so also results in
  ``asyncio`` complaining. In fact, none of the code inside a coroutine
  executes until an ``await`` statement is invoked on it. In other words,
  ``await`` is more like ``thread.start(); thread.join()`` than simply
  ``thread.join()``.
* The ``TaskGroup`` context manager automatically awaits for all the tasks to
  complete; there is no need to explicitly ``await`` the tasks.

Using asynchronous operations shows a clear improvement in scaling:

.. code-block::

    aio_n(1): 0.51s
    aio_n(10): 0.51s
    aio_n(100): 0.61s
    aio_n(1000): 1.16s
    aio_n(10000): 6.77s

We can also use ``stpr``:

.. code-block:: Python

    import stpr

    @stpr.fn
    def stpr_1():
        with ClientSession() as session:
            with session.get('http://localhost:8080') as resp:
                contents = resp.text()

    @stpr.fn
    def stpr_n(n):
        with st.parallelFor(range(n)):
            stpr_1()

    def stpr_main(n):
        stpr.run(stpr_n(n))

Some notable things are:

* Stpr is, at its core, an `AST <https://docs.python.org/3/library/ast.html>`_
  transformer. It takes an AST tree representation of a function, transforms it
  to a new AST tree, and compiles the result in place. This requires Stpr
  functions to be decorated using ``@stpr.fn``.
* We do not need any special syntax for invoking coroutines, which are
  automatically awaited for. Stpr abstracts *how* computations are run and,
  instead, allows the programmer to focus on *what* computations are run and
  how their parallelism is structured.

The timing results of ``stpr_main(n)`` should look similar to those of
``aio_main(n)``:

.. code-block::

    stpr_n(1): 0.62s
    stpr_n(10): 0.52s
    stpr_n(100): 0.58s
    stpr_n(1000): 1.06s
    stpr_n(10000): 6.79s


The Basics
----------

Running Stpr Functions
^^^^^^^^^^^^^^^^^^^^^^

Using Stpr starts with decorating a function or method with the :func:`stpr.fn`
decorator. The decorator analyzes the function code, processes ``with stpr.xx``
statements as well as other calls to Stpr functions, and inserts ``await``
statements as needed. A Stpr function can be called from normal Python code
using :func:`stpr.run` and directly from another Stpr function:

.. code-block:: Python

    @stpr.fn
    def foo():
        pass

    @stpr.fn
    def bar():
        foo()  # can be called directly

    stpr.run(bar())
    stpr.run(bar)  # if there are no arguments, this works, too


Running Things in Parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^

Basic parallel execution can be achieved with :class:`stpr.parallel`, which
runs all of its child statements in parallel:

.. code-block:: Python

    @stpr.fn
    def foo(n: int):
        with stpr.parallel:
            f()
            g()
            if n == 3:
                h()
                l()

The above code runs ``f()``, ``g()``, and the ``if`` statement in parallel.
Because ``h()`` and ``l()`` are not immediate sub-statements of the ``with``
statement, they will run sequentially which is the default. This can be
visualized in the activity diagram below:

.. uml::
    :align: center

    @startuml
    skinparam DefaultFontSize 15
    skinparam defaultFontName Ubuntu Mono
    start
    fork
        :f();
    fork again
        :g();
    fork again
        if (n == 3) then (yes)
            :h();
            :l();
        else (no)
        endif
    end fork
    stop
    @enduml

The functions ``a``, ``b``, ``c``, and ``d`` can be either Stpr functions,
``async`` functions, or normal Python functions. If the latter, they will be
run in separate threads such that the diagram above is always satisfied. You
can read more about this mechanism in :ref:`sync-fn-exec`.

Variables assigned inside a ``stpr.parallel`` statement can be used as if they
were assigned in any other ``with`` statement:

.. code-block:: Python

    with stpr.parallel:
        a = f()
        b = g()
    print(a + b)

A function version of ``stpr.parallel`` also exists:

.. code-block:: Python

    a, b = stpr.parallel(f(), g())

In this case, ``stpr.parallel`` acts as the identity function and, aside from
executing ``f()`` and ``g()`` in parallel, it is equivalent to the
statement ``a, b = f(), g()``.


Running Things Sequentially
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Statements inside Stpr functions are, by default, executed sequentially.
However, as we have seen above, cases may exist when we want to run some
a sequence of statements in parallel with others. We could simply rely on
the fact that ``stpr.parallel`` treats an ``if`` statement as a whole and write
something like this:

.. code-block:: Python

    with stpr.parallel:
        a()
        if True:
            c()
            d()

However, the ``if`` statement does not intuitively represent a sequence and can
be confusing. Instead, we can use ``stpr.seq`` to clearly indicate that a
sequential execution of statements is desired:

.. code-block:: Python

    with stpr.parallel:
        a()
        with stpr.seq:
            c()
            d()


Parallel "Iterations"
^^^^^^^^^^^^^^^^^^^^^

Parallel iterations can be achieved with :func:`stpr.parallelFor`:

.. code-block:: Python

    with stpr.parallelFor(range(10)) as i:
        f(i)

In the above example, ten parallel invocations of ``f()`` will be run, each
with a distinct number between 0 and 9 as argument.


Controlled Races
^^^^^^^^^^^^^^^^

It is often the case that we have two or more concurrent operations but care
only about the result of the operation that finishes first. This is essentially
the case with all timeout-guarded operations. Stpr allows implementing such
functionality using :func:`stpr.race`:

.. code-block:: Python

    a = stpr.race(a(), b())

The code above, similar to ``stpr.parallel``, runs both ``a()`` and ``b()`` in
parallel. However, it only returns the result of the functions the finishes
first. This can be used to quickly implement timeouts:

.. code-block:: Python

    result = stpr.race(f(), stpr.wait(2))
    if result is None:
        print('Operation timed out')
    else:
        print(f'Result is {result}')

The code above assumes that ``f()`` returns a non-null result. When the first
function under ``stpr.race()`` completes, ``race`` will cancel all other
still-running functions. This is generally intended to be used with functions
whose cancellation will not leave the system in an inconsistent state.


Exception Handling
^^^^^^^^^^^^^^^^^^

One of the defining features that distinguishes structured parallelism from
traditional threading is that exceptions that occur in a parallel context are
propagated just as they would in sequential code:

.. code-block:: Python

    try:
        with stpr.parallel:
            f()
            g()
    except Exception:
        print('Exception!')

In the above example, if either ``f()`` or ``g()`` raise an exception, all
actively running coroutines started by :class:`stpr.parallel` are canceled
and the exception is propagated to the enclosing ``except`` branch.


Critical Sections
^^^^^^^^^^^^^^^^^

It is often the case that two (or more) concurrent computations must access
a shared state while ensuring that a set of operations is only performed in
one of those computations. The classic example of this is the read-modify-write
pattern. In order to ensure correctness, these operations must be executed
in a so called *critical section*, which is usually done with a *lock*.
There are two ``Lock`` classes in the Python standard library:
:class:`threading.Lock` (with its re-entrant twin :class:`threading.RLock`)
and :class:`asyncio.Lock`. Unfortunately, :class:`asyncio.Lock` does not
prevent two Python threads from running concurrently and
:class:`threading.RLock` does not prevent two coroutines from running
concurrently. Since Stpr employs both Python Threads and coroutines, it
provides a :class:`stpr.Lock`, which ensures that both threads and coroutines
are prevented from concurrently accessing a critical section. The semantics
are similar to the other ``Lock`` classes:

.. code-block:: Python

    def g(lock: stpr.Lock) -> None:
        # this is not a coroutine, so it will run in
        # a thread
        with lock:
            # atomic operations

    @stpr.fn
    def f(lock: stpr.Lock) -> None:
        # this is a coroutine, so it will run in an asyncio
        # loop
        with lock:
            atomic operations

In most cases, the internal details of what runs how are not important. What is
important is that one should use :class:`stpr.Lock` with Stpr.


Throttling
^^^^^^^^^^

A throttle is a generalization of a *lock* in that it allows at most a certain
number ``n`` of computations to run concurrently. A lock is, therefore, a
throttle with ``n == 1``.



.. _sync-fn-exec:

Synchronous Function Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
