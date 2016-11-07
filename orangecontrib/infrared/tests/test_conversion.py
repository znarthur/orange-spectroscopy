import unittest

import numpy as np
import Orange
from Orange.classification import LogisticRegressionLearner
from Orange.evaluation.testing import TestOnTestData
from Orange.evaluation.scoring import AUC

import sklearn.model_selection as ms

from orangecontrib.infrared.preprocess import Interpolate
from orangecontrib.infrared.data import getx


def seperate_learn_test(data):
    sf = ms.ShuffleSplit(n_splits=1, test_size=0.2, random_state=np.random.RandomState(0))
    (traini, testi), = sf.split(y=data.Y, X=data.X)
    return data[traini], data[testi]


def destroy_atts_conversion(data):
    natts = [ a.copy() for a in data.domain.attributes ]
    ndomain = Orange.data.Domain(natts, data.domain.class_vars,
                                metas=data.domain.metas)
    ndata = Orange.data.Table(ndomain, data)
    ndata.X = data.X
    return ndata


class TestConversion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.collagen = Orange.data.Table("collagen")

    def test_predict_same_domain(self):
        train, test = seperate_learn_test(self.collagen)
        auc = AUC(TestOnTestData(train, test, [LogisticRegressionLearner]))
        self.assertGreater(auc, 0.9) # easy dataset

    def test_predict_samename_domain(self):
        train, test = seperate_learn_test(self.collagen)
        test = destroy_atts_conversion(test)
        aucdestroyed = AUC(TestOnTestData(train, test, [LogisticRegressionLearner]))
        self.assertTrue(0.45 < aucdestroyed < 0.55)

    def test_predict_samename_domain_interpolation(self):
        train, test = seperate_learn_test(self.collagen)
        aucorig = AUC(TestOnTestData(train, test, [LogisticRegressionLearner]))
        test = destroy_atts_conversion(test)
        train = Interpolate(train, points=getx(train))
        auc = AUC(TestOnTestData(train, test, [LogisticRegressionLearner]))
        self.assertEqual(aucorig, auc)
