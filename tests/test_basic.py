import time

import stpr as st
from _test_tools import expect_time


@st.fn
def _test_seq():
    with st.seq:
        st.wait(0.1)
        st.wait(0.1)


def test_seq():
    with expect_time(0.2):
        st.run(_test_seq())


@st.fn
def _test_parallel():
    with st.parallel:
        st.wait(0.1)
        st.wait(0.1)


def test_parallel():
    with expect_time(0.1):
        st.run(_test_parallel())


@st.fn
def _test_nesting_1():
    with st.seq:
        st.wait(0.1)
        with st.parallel:
            st.wait(0.1)
            st.wait(0.1)
        st.wait(0.1)


def test_nesting_1():
    with expect_time(0.3):
        st.run(_test_nesting_1())


@st.fn
def _test_nesting_2():
    with st.parallel:
        with st.seq:
            st.wait(0.1)
            st.wait(0.1)
        with st.seq:
            st.wait(0.1)
            st.wait(0.1)


def test_nesting_2():
    with expect_time(0.2):
        st.run(_test_nesting_2)


@st.fn
def _test_parallel_2():
    a = 0
    b = 0
    with st.parallel:
        a = 1
        b = 2
    assert a == 1
    assert b == 2


def test_parallel_2():
    st.run(_test_parallel_2)


@st.fn
def _test_parallel_3():
    with st.parallel:
        a = 1
        b = 2
    c = a + b
    assert c == 3


def test_parallel_3():
    st.run(_test_parallel_3)


@st.fn
def _fa():
    st.wait(0.1)
    return 1


@st.fn
def _fb():
    st.wait(0.2)
    return 2


@st.fn
def _test_parallel_fn():
    a = 0
    b = 0
    with expect_time(0.2):
        a, b = st.parallel(_fa(), _fb())
    assert a == 1
    assert b == 2


def test_parallel_fn():
    st.run(_test_parallel_fn)


@st.fn
def _test_pfor_1():
    with st.parallelFor(range(10)):
        st.wait(0.2)


def test_pfor_1():
    with expect_time(0.2):
        st.run(_test_pfor_1)


@st.fn
def _test_pfor_2():
    l = []
    with st.parallelFor(range(10)) as i:
        st.wait(0.2)
        l.append(i)
    return l


def test_pfor_2():
    with expect_time(0.2):
        l = st.run(_test_pfor_2)
    assert l == [i for i in range(10)]


@st.fn
def _test_parallel_3():
    with st.parallel:
        if True:
            st.wait(0.2)
        else:
            st.wait(0.1)
        if not False:
            st.wait(0.2)
        else:
            st.wait(0.1)


def test_parallel_3():
    with expect_time(0.2):
        st.run(_test_parallel_3)


def _sync_wait(delay):
    time.sleep(delay)


@st.fn
def _test_parallel_4():
    with st.parallel:
        _sync_wait(0.2)
        _sync_wait(0.2)


def test_parallel_4():
    with expect_time(0.2):
        st.run(_test_parallel_4)


@st.fn
def _test_lock_1():
    lock = st.Lock()
    with lock:
        with lock:
            pass


def test_lock_1():
    _test_lock_1()


def _dummy():
    pass


@st.fn
def _test_lock_2():
    lock = st.Lock()
    a = 0
    with st.parallel:
        for i in range(1000):
            _dummy()
            with lock:
                if a != 0:
                    raise Exception(f'Lock is broken i: {i}')
                a = 1
                _dummy()
                a = 0
            _dummy()
        for j in range(1000):
            _dummy()
            with lock:
                if a != 0:
                    raise Exception(f'Lock is broken j: {j}')
                a = 1
                _dummy()
                a = 0
            _dummy()


def test_lock_2():
    st.run(_test_lock_2)


class _Flag:
    def __init__(self):
        self.value = 0


def _test_lock_sync(lock: st.Lock, f: _Flag) -> None:
    for i in range(1000):
        _dummy()
        with lock:
            if f.value != 0:
                raise Exception('Lock is broken')
            f.value = 1
            _dummy()
            f.value = 0
        _dummy()


@st.fn
def _test_lock_3():
    lock = st.Lock()
    f = _Flag()

    with st.parallel:
        _test_lock_sync(lock, f)
        _test_lock_sync(lock, f)


def test_lock_3():
    st.run(_test_lock_3)


@st.fn
def _test_lock_async(lock: st.Lock, f: _Flag) -> None:
    for i in range(1000):
        _dummy()
        with lock:
            if f.value != 0:
                raise Exception('Lock is broken')
            f.value = 1
            _dummy()
            f.value = 0
        _dummy()


@st.fn
def _test_lock_4():
    lock = st.Lock()
    f = _Flag()

    with st.parallel:
        _test_lock_sync(lock, f)
        _test_lock_async(lock, f)


def test_lock_4():
    st.run(_test_lock_4)


@st.fn
def _test_start_1_sub(f: _Flag) -> None:
    st.wait(0.05)
    f.value = 1
    st.wait(0.1)


@st.fn
def _test_start_1() -> None:
    f = _Flag()
    with st.fork:
        _test_start_1_sub(f)
    assert f.value == 0
    st.wait(0.1)
    assert f.value == 1


def test_start_1() -> None:
    st.run(_test_start_1)


@st.fn
def _test_throttle() -> None:
    t = st.Throttle(10)
    n = 0
    with st.parallelFor(range(20)) as i:
        with t:
            n += 1
            if n > 10:
                raise Exception('Broken throttle')
            st.wait(0.1)
            n -= 1


def test_throttle() -> None:
    st.run(_test_throttle)


@st.fn
def _test_pass() -> None:
    pass

def test_pass() -> None:
    st.run(_test_pass)