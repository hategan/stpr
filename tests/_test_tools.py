
from contextlib import contextmanager
from time import time


@contextmanager
def expect_time(t):
    start = time()
    try:
        yield
    finally:
        end = time()
        print('Expected time: %s, actual: %s' % (t, end - start))
        assert abs((end - start) - t) < 0.01, 'Expected time: %s, actual: %s' % (t, end - start)
