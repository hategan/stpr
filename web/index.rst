.. raw:: html

    <div style="height: 0; visibility: hidden;">

Stpr
====

.. raw:: html

    </div>

    <div class="main-content">
        <div class="main-section">
            <div class="left">
                <h1>A structured parallelism library for Python</h1>
            </div>
            <div class="right">
                <img src="/stpr/_static/main.png" width="80%"/>
            </div>
        </div>
        <div class="main-section">
            <div class="left">
                <p>
    Stpr is a Python concurrency library. It simplifies asyncio programming
    significantly and adds useful concurrency tools, data structures, and
    functions that can be used to achieve highly scalable I/O.
                </p>
                <p>
    Easily transform synchronous code to async code by simply decorating
    existing functions with the Stpr decorator.
                </p>
                <p>
    <a class="button" href="docs">View Documentation</a>
                </p>
            </div>
            <div class="right">
                <pre class="main-code">
    <span class="k">with</span> <span class="f">stpr.parallel</span>:
        <span class="k">with</span> <span class="f">stpr.seq</span>:
            a()
            b()
            c()
        <span class="k">with</span> <span class="f">stpr.seq</span>:
            d()
            <span class="k">with</span> <span class="f">stpr.parallel</span>:
                e()
                f()
            g()</pre>
            </div>
        </div>
    </div>