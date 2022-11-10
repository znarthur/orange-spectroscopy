from unittest import TestCase

from Orange.data import Table

from orangecontrib.spectroscopy.preprocess.utils import reference_eq_X


class TestEq(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")
        cls.iris2 = Table("iris")
        cls.iris_changed = Table("iris")
        with cls.iris_changed.unlocked():
            cls.iris_changed[0][0] = 42

    def test_reference_eq_X_none(self):
        data = self.iris
        self.assertTrue(reference_eq_X(None, None))
        self.assertFalse(reference_eq_X(data, None))
        self.assertFalse(reference_eq_X(None, data))

    def test_reference_eq_X_same(self):
        self.assertTrue(reference_eq_X(self.iris, self.iris))
        self.assertTrue(reference_eq_X(self.iris, self.iris2))
        self.assertFalse(reference_eq_X(self.iris, self.iris_changed))
