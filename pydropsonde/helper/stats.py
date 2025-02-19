import numpy as np
import xarray as xr
from scipy.signal import correlate


def get_dist_to_nonan(ds, alt_dim, variable):
    masked_alt = ds[alt_dim].where(~np.isnan(ds[variable]))
    masked_alt.name = "int_alt"
    int_masked = masked_alt.interpolate_na(
        dim=alt_dim,
        method="nearest",
        fill_value="extrapolate",
    )
    return np.abs(ds[alt_dim] - int_masked)


def get_autocorr_sp(da, tau):
    vals = da
    mask = (~np.isnan(vals[:-tau])) & (~np.isnan(vals[tau:]))
    n = np.sqrt(
        np.sum(np.abs(vals[:-tau][mask] ** 2)) * np.sum(np.abs(vals[tau:][mask] ** 2))
    )
    if np.any(mask):
        return correlate(vals[:-tau][mask], vals[tau:][mask], mode="valid") / n
    else:
        return np.nan


def apply_autocorr_to_xr(autocorr_fct, ds, tau, alt_dim):
    return xr.apply_ufunc(
        autocorr_fct,
        ds,
        input_core_dims=[[alt_dim]],
        kwargs={"tau": int(tau)},
        vectorize=True,
        dask="allowed",
    )


def calc_autocorrelation(
    ds,
    alt_dim,
    maxalt,
    variables,
):
    if variables is None:
        variables = ["u", "v", "p", "theta", "q", "rh", "ta"]
    interp_step = np.abs(ds[alt_dim].diff(dim=alt_dim)[0]).values
    taus = ds[alt_dim].where(ds[alt_dim] < maxalt, drop=True).values[1:] / interp_step
    autocorr = {alt_dim: {"dims": (alt_dim), "data": taus * interp_step}}

    for var in variables:
        autocorr[f"{var}_autocorr"] = {
            "dims": (alt_dim),
        }
        res = [
            apply_autocorr_to_xr(get_autocorr_sp, ds[var], tau, alt_dim=alt_dim)
            for tau in taus
        ]
        autocorr[f"{var}_autocorr"]["data"] = [np.nanmean(corr) for corr in res]
        autocorr[f"{var}_std_autocorr"] = {
            "dims": (alt_dim),
            "data": [np.nanstd(corr) for corr in res],
        }
    return xr.Dataset.from_dict(autocorr)
