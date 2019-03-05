from enum import Enum

import numpy as np
import Orange.data
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainRef,\
    WrongReferenceException, replace_infs


class SpecTypes(Enum):
    """
    Spectral types possibly supported by spectral transforms
    """
    ABSORBANCE = "Absorbance"
    TRANSMITTANCE = "Transmittance"
    SINGLECHANNEL = "Single Channel"


class AbsorbanceFeature(SelectColumn):
    pass


class _AbsorbanceCommon(CommonDomainRef):

    def transformed(self, data):
        if self.ref is not None:
            # Calculate from single-channel data
            ref_X = self.interpolate_extend_to(self.ref, getx(data))
            if len(data):  # numpy does not like to divide shapes (0, b) by (a, b)
                absd = ref_X / data.X
                np.log10(absd, absd)
            else:
                absd = data
        else:
            # Calculate from transmittance data
            absd = np.log10(data.X)
            absd *= -1
        # Replace infs from either np.true_divide or np.log10
        return replace_infs(absd)


class TransformOptionalReference(Preprocess):

    def __init__(self, ref=None):
        if ref is not None and len(ref) != 1:
            raise WrongReferenceException("Reference data should have length 1")
        self.ref = ref

    def __call__(self, data):
        common = self._cl_common(self.ref, data.domain)
        newattrs = [Orange.data.ContinuousVariable(
            name=var.name, compute_value=self._cl_feature(i, common))
            for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class Absorbance(TransformOptionalReference):
    """
    Convert data to absorbance.

    Set ref to calculate from single-channel spectra, otherwise convert from transmittance.

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    _cl_common = _AbsorbanceCommon
    _cl_feature = AbsorbanceFeature

    from_types = (SpecTypes.TRANSMITTANCE, SpecTypes.SINGLECHANNEL)


class TransmittanceFeature(SelectColumn):
    pass


class _TransmittanceCommon(CommonDomainRef):

    def transformed(self, data):
        if self.ref is not None:
            # Calculate from single-channel data
            if len(data):  # numpy does not like to divide shapes (0, b) by (a, b)
                ref_X = self.interpolate_extend_to(self.ref, getx(data))
                transd = data.X / ref_X
            else:
                transd = data
        else:
            # Calculate from absorbance data
            transd = data.X.copy()
            transd *= -1
            np.power(10, transd, transd)
        # Replace infs from either np.true_divide or np.log10
        return replace_infs(transd)


class Transmittance(TransformOptionalReference):
    """
    Convert data to transmittance.

    Set ref to calculate from single-channel spectra, otherwise convert from absorbance.

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    from_types = (SpecTypes.ABSORBANCE, SpecTypes.SINGLECHANNEL)

    _cl_common = _TransmittanceCommon
    _cl_feature = TransmittanceFeature
