# Stpr

* [**Website**](https://hategan.github.io/stpr)
* [**Documentation**](https://hategan.github.io/stpr/docs)

Stpr (probably pronounced "steeper") is an experimental library for
structured parallelism in Python. It combines the efficiency of the
asyncio library with the intuitiveness of sequential programming.

Stpr works by dynamically re-compiling functions and methods. The
`@stpr.fn` decorator parses the source code of the function it
decorates, analyzes the [Abstract Syntax
Tree](https://docs.python.org/3/library/ast.html) (AST), transforms it,
and transparently compiles the resulting AST to bytecode. This allows it
to augment the standard Python semantics and introduce new parallelism
constructs.

A simple example is running functions in parallel:

```Python
import asyncio
import stpr
import time

@stpr.fn
def a():
    stpr.wait(1)

async def b():
    await asyncio.sleep(1)

def c():
    time.sleep(1)

@stpr.fn
def parallel():
    with stpr.parallel:
        a()
        b()
        c()

stpr.run(parallel)
```

Stpr-decorated functions are coroutines. Stpr schedules coroutines, such
as `a()` and `b()` in an asyncio event loop. Synchronous functions, such
as `c` are run in a thread pool.

Another example is starting a server socket, accepting, and handling
connections:

```Python
import stpr

@stpr.fn
def handle_connection(conn):
    ...

@stpr.fn
def server():
    with stpr.Socket('localhost', 8080) as sock:
        with stpr.parallelFor(sock.connections()) as conn:
            handle_connection(conn)

stpr.run(server)
```

