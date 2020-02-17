# -*- coding: utf-8 -*-

import pytest
from python__k_nearest_neighbours.skeleton import fib

__author__ = "LBielecki"
__copyright__ = "LBielecki"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
