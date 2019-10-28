# -*- coding: utf-8 -*-

import pytest
from lexical_investigation.skeleton import fib

__author__ = "Vincent Nguyen"
__copyright__ = "Vincent Nguyen"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
