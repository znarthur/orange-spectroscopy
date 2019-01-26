import unittest

import numpy as np

from orangecontrib.spectroscopy.widgets.owstackalign import \
    alignstack, RegisterTranslation, shift_fill


def test_image():
    return np.hstack((np.zeros((5, 1)),
                      np.diag([1, 5., 3., 1, 1]),
                      np.ones((5, 1))))


def _up(im, fill=0):
    return np.vstack((im[1:],
                      np.ones_like(im[0]) * fill))


def _down(im, fill=0):
    return np.vstack((np.ones_like(im[0]) * fill,
                      im[:-1]))


def _right(im, fill=0):
    return np.hstack((np.ones_like(im[:, :1]) * fill,
                      im[:, :-1]))


def _left(im, fill=0):
    return np.hstack((im[:, 1:],
                      np.ones_like(im[:, :1]) * fill))


class TestUtils(unittest.TestCase):

    def test_image_shift(self):
        im = test_image()
        calculate_shift = RegisterTranslation()
        s = calculate_shift(im, _up(im))
        np.testing.assert_equal(s, (1, 0))
        s = calculate_shift(im, _down(im))
        np.testing.assert_equal(s, (-1, 0))
        s = calculate_shift(im, _left(im))
        np.testing.assert_equal(s, (0, 1))
        s = calculate_shift(im, _right(im))
        np.testing.assert_equal(s, (0, -1))
        s = calculate_shift(im, _left(_left(im)))
        np.testing.assert_equal(s, (0, 2))

    def test_alignstack(self):
        im = test_image()
        _, aligned = alignstack([im, _up(im), _down(im), _right(im)],
                                shiftfn=RegisterTranslation())
        self.assertEqual(aligned.shape, (4, 5, 7))

    def test_shift_fill(self):
        im = test_image()

        # shift down
        a = shift_fill(im, (1, 0))
        np.testing.assert_almost_equal(a, _down(im, np.nan))
        a = shift_fill(im, (0.55, 0))
        np.testing.assert_equal(np.isnan(a), np.isnan(_down(im, np.nan)))
        a = shift_fill(im, (0.45, 0))
        np.testing.assert_equal(np.isnan(a), False)

        # shift up
        a = shift_fill(im, (-1, 0))
        np.testing.assert_almost_equal(a, _up(im, np.nan))
        a = shift_fill(im, (-0.55, 0))
        np.testing.assert_equal(np.isnan(a), np.isnan(_up(im, np.nan)))
        a = shift_fill(im, (-0.45, 0))
        np.testing.assert_equal(np.isnan(a), False)

        # shift right
        a = shift_fill(im, (0, 1))
        np.testing.assert_almost_equal(a, _right(im, np.nan))
        a = shift_fill(im, (0, 0.55))
        np.testing.assert_equal(np.isnan(a), np.isnan(_right(im, np.nan)))
        a = shift_fill(im, (0, 0.45))
        np.testing.assert_equal(np.isnan(a), False)

        # shift left
        a = shift_fill(im, (0, -1))
        np.testing.assert_almost_equal(a, _left(im, np.nan))
        a = shift_fill(im, (0, -0.55))
        np.testing.assert_equal(np.isnan(a), np.isnan(_left(im, np.nan)))
        a = shift_fill(im, (0, -0.45))
        np.testing.assert_equal(np.isnan(a), False)
