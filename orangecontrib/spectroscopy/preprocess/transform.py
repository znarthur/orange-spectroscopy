from enum import Enum

import numpy as np
import Orange.data
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomain

class SpecTypes(Enum):
    """
    Spectral types possibly supported by spectral transforms
    """
    ABSORBANCE = "Absorbance"
    TRANSMITTANCE = "Transmittance"
    SINGLECHANNEL = "Single Channel"


class CommonDomainRef(CommonDomain):
    """CommonDomain which also ensures reference domain transformation"""
    def __init__(self, ref, domain):
        super().__init__(domain)
        self.ref = ref

    def __call__(self, data):
        data = self.transform_domain(data)
        if self.ref is not None:
            ref = self.transform_domain(self.ref)
        else:
            ref = self.ref
        return self.transformed(data, ref)

    def transformed(self, data, ref):
        raise NotImplemented


class AbsorbanceFeature(SelectColumn):
    pass


class _AbsorbanceCommon(CommonDomainRef):

    def transformed(self, data, ref):
        if ref is not None:
            # Calculate from single-channel data
            absd = ref.X / data.X
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

    def transformed(self, data, ref):
        if ref is not None:
            # Calculate from single-channel data
            transd = data.X / ref.X
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
        self.ref = ref

    def __call__(self, data):
        common = _TransmittanceCommon(self.ref, data.domain)
        newattrs = [Orange.data.ContinuousVariable(
            name=var.name, compute_value=TransmittanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)