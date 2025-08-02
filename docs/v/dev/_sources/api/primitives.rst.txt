:tocdepth: 3

.. _concurrency-primitives:

Concurrency Primitives
----------------------

The main starting point in using Stpr is the :py:func:`stpr.fn` decorator which
transforms a function definition and allows it to use the Stpr concurrency
constructs.

The `stpr.fn` Decorator
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: stpr.fn(f)


Concurrency Primitives
^^^^^^^^^^^^^^^^^^^^^^

.. rst-class:: stprtr
.. autoclass:: stpr.seq
    :no-show-inheritance:

.. rst-class:: stprtr
.. autoclass:: stpr.parallel
    :no-show-inheritance:

.. rst-class:: stprtr
.. autoclass:: stpr.parallelFor
    :no-show-inheritance:

.. rst-class:: stprtr
.. autoclass:: stpr.fork
    :no-show-inheritance:

.. rst-class:: stprtr
.. autoclass:: stpr.race
    :no-show-inheritance:

.. autoclass:: stpr.Lock
    :members:

.. autoclass:: stpr.Throttle
    :members:

