# -*- coding: utf-8 -*-
"""
Tests of neo.io.exdirio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import ExdirIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestExdirIO(BaseTestIO, unittest.TestCase):
    files_to_test = ['/tmp/test/test.exdir']
    # files_to_download = files_to_test
    ioclass = ExdirIO


if __name__ == "__main__":
    unittest.main()
