###############################################################
Structured Parallelism in Python
###############################################################


Stpr is a library that provides structured parallelism constructs for
Python. In short, structured parallelism is to fork/join as structured
programming is to programming with the GOTO instruction.



Structured parallelism, like structured programming, provides an equivalence
between the parallelism semantics of a program and its syntactical structure.
In structured programming, a loop is clearly delineated by some beginning and
ending token, and so are conditional branches; In a similar spirit, in
structured parallelism, there is a direct correspondence between the structure
of the program and the structure of the parallelism. Let's look at the
following example:

.. code-block:: Python

    import stpr

    with stpr.parallel:
        with stpr.seq:
            a()
            b()
        with stpr.seq:
            c()
            d()

Here, the two `with stpr.seq` blocks run in parallel, while :code:`b` is
invoked after :code:`a` and :code:`d` is invoked after :code:`c`. It is easy to
see that the code above is represented by the following UML diagram:

.. uml::
    :align: center

    @startuml
    skinparam DefaultFontSize 15
    skinparam defaultFontName Ubuntu Mono
    start
    fork
        :a();
        :b();
    fork again
        :c();
        :d();
    end fork
    stop
    @enduml

where the thick horizontal bars enclose parallel execution.

Stpr provides a number of :ref:`concurrency primitives <concurrency-primitives>` and
:ref:`utility functions <functions>`. :ref:`Reactives <reactives>` can be used to
automatically propagate data updates through computations.

The full list of the documentation index is provided below.


.. toctree::
    :maxdepth: 3

    getting_started
    tutorial

.. toctree::
    :maxdepth: 4
    :class: merge-up

    api/primitives.rst

.. toctree::
    :maxdepth: 2
    :class: merge-up

    api/types.rst
    api/functions.rst
    api/reactives.rst

