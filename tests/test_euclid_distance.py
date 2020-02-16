# -*- coding: utf-8 -*-

import pytest
from python__k_nearest_neighbours.skeleton import euclid_distance

__author__ = "JMarcinkowski"
__copyright__ = "JMarcinkowski"
__license__ = "AGH"


def test_euclid_distance():
    assert euclid_distance([1,1],[1,1]) == 0
    assert euclid_distance([0,0],[5,0]) == 5
    assert euclid_distance([2,5],[5,9]) == 5
    assert euclid_distance([-2,-5],[1,-1]) == 5
    with pytest.raises(AssertionError):
        euclid_distance([0,0],[0,0])
