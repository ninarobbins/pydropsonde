# this adds functionality that is not (yet) in the moist_thermodynamics repo, but should be replaced if added there
import numpy as np
from moist_thermodynamics import constants


triple_point_water = 273.16  # Triple point temperature in K


def q2vmr(q):
    """
    returns the volume mixing ratio from specific humidity
    """
    return q / ((1 - q) * constants.molar_mass_h2o / constants.md + q)


def vmr2q(vmr):
    """
    returns specific humidity from volume mixing ratio
    """
    return vmr / ((1 - vmr) * constants.md / constants.molar_mass_h2o + vmr)


def q2mr(q):
    """
    returns specific humidity from mixing ratio
    """
    return q / (1 - q)


def mr2q(mr):
    """
    returns mixing ratio from specific humidity
    """
    return mr / (1 + mr)


def density_from_mr(p, T, mr, eps=None):
    """
    returns density for given pressure, temperature and R
    """
    Rd = constants.dry_air_gas_constant
    if eps is None:
        eps = constants.eps1
    return eps * p * (1 + mr) / (Rd * T * (mr + eps))  # water vapor density


def density_from_q(p, T, q):
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    return p / ((Rd + (Rv - Rd) * q) * T)


def theta2ta(theta, P, qv=0.0, ql=0.0, qi=0.0):
    """Returns the temperature for an unsaturated moist fluid, given the potential temperature
    (reverse of Bjorn stevens moist thermodynamicts theta())

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qv: specific vapor mass
        ql: specific liquid mass
        qi: specific ice mass

    """
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cpv = constants.isobaric_water_vapor_specific_heat
    cl = constants.liquid_water_specific_heat
    ci = constants.frozen_water_specific_heat
    P0 = constants.P0

    qd = 1.0 - qv - ql - qi
    kappa = (qd * Rd + qv * Rv) / (qd * cpd + qv * cpv + ql * cl + qi * ci)
    return theta / (P0 / P) ** kappa


def integrate_water_vapor(p, q, T=None, z=None, axis=0):
    """Returns the integrated water vapor for given specific humidity
    Args:
        p: pressure in Pa
        either: (hydrostatic)
            q: specific humidity
        or: (non-hydrostatic)
            q: specific humidity
            T: temperature
            z: height

    """

    def integrate_column(y, x, axis=0):
        if np.all(x[:-1] >= x[1:]):
            return -np.trapz(y, x, axis=axis)
        else:
            return np.trapz(y, x, axis=axis)

    if T is None and z is None:
        # Calculate IWV assuming hydrostatic equilibrium.
        g = constants.gravity_earth
        return integrate_column(q, p, axis=axis) / g
    elif T is None or z is None:
        raise ValueError(
            "Pass both `T` and `z` for non-hydrostatic calculation of the IWV."
        )
    else:
        # Integrate the water vapor mass density for non-hydrostatic cases.
        rho = density_from_q(p, T, q)  # water vapor density
        return integrate_column(q * rho, z, axis=axis)
