#!/usr/bin/env python

import sys
sys.path.append("./scripts")

from utils import *

def test_replay_buffer_cyclic():
    buf = ReplayBuffer(5)
    buf.add_one((0, 1, 2, 3, 4))
    buf.add([(1, 2, 3, 4, 5)] * 5)

    assert all(
        tup == (1, 2, 3, 4, 5)
        for tup in buf.get_tuples()
        )
    
def test_replay_buffer_all_arrays():
    buf = ReplayBuffer(5)
    buf.add([(1, 2, 3, 4, 5)] * 5)

    s, a, r, sp, d = buf.get_arrays()

    assert s == (1, 1, 1, 1, 1)
    assert a == (2, 2, 2, 2, 2)
    assert r == (3, 3, 3, 3, 3)
    assert sp == (4, 4, 4, 4, 4)
    assert d == (5, 5, 5, 5, 5)
