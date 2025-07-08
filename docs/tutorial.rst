********
Tutorial
********


The magic required to enable |project_name| functionality is the `@stpr.fn` decorator:

.. code-block:: Python

    import stpr

    @stpr.fn
    def f():
        print('Hello world!')

The `sp.fn` decorator instruments the function it is attached to. The first thing to notice after
for a function decorated with `sp.fn` is that it cannot be run synchronously any more. Attempting
to invoke `f` as `f()` results in an error message:

.. code-block::

     RuntimeWarning: coroutine 'f' was never awaited

That is because all `@stpr.fn` decorated functions become coroutines. In order to run `f`, you can
use

.. code-block:: Python

    stpr.run(f())

.. code-block::

    Hello world!

The parentheses are optional. The use of `stpr.run` is unnecessary if a `stpr` function is invoked
from another `stpr` function, as the following example illustrates:


.. code-block:: Python

    @stpr.fn
    def a():
        print('A')

    @stpr.fn
    def b():
        a()
        print('B')

    stpr.run(b)

.. code-block::

    A
    B

The `stpr.fn` decorator automatically takes care of awaiting coroutines when they are invoked from
a decorated function. In fact, this applies to all other coroutines.