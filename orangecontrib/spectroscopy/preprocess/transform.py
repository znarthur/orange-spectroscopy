from enum import Enum

import numpy as np
import Orange.data
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainRef


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
            absd = ref_X / data.X
            np.log10(absd, absd)
        else:
            # Calculate from transmittance data
            absd = np.log10(data.X)
            absd *= -1
        return absd


class Absorbance(Preprocess):
    """
    Convert data to absorbance.

    Set ref to calculate from single-channel spectra, otherwise convert from transmittance.

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    from_types = (SpecTypes.TRANSMITTANCE, SpecTypes.SINGLECHANNEL)

    def __init__(self, ref=None):
        if ref:
            # Only single reference spectra are supported
            self.ref = ref[0:1]
        else:
            self.ref = ref

    def __call__(self, data):
        common = _AbsorbanceCommon(self.ref, data.domain)
        newattrs = [Orange.data.ContinuousVariable(
            name=var.name, compute_value=AbsorbanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class TransmittanceFeature(SelectColumn):
    pass


class _TransmittanceCommon(CommonDomainRef):

    def transformed(self, data):
        if self.ref is not None:
            # Calculate from single-channel data
            ref_X = self.interpolate_extend_to(self.ref, getx(data))
            transd = data.X / ref_X
        else:
            # Calculate from absorbance data
            transd = data.X.copy()
            transd *= -1
            np.power(10, transd, transd)
        return transd


class Transmittance(Preprocess):
    """
    Convert data to transmittance.

    Set ref to calculate from single-channel spectra, otherwise convert from absorbance.

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    from_types = (SpecTypes.ABSORBANCE, SpecTypes.SINGLECHANNEL)

    def __init__(self, ref=None):
        if ref:
            # Only single reference spectra are supported
            self.ref = ref[0:1]
        else:
            self.ref = ref

    def __call__(self, data):
        common = _TransmittanceCommon(self.ref, data.domain)
        newattrs = [Orange.data.ContinuousVariable(
            name=var.name, compute_value=TransmittanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)