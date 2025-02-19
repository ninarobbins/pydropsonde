import numpy as np


def get_dist_to_nonan(ds, alt_dim, variable):
    masked_alt = ds[alt_dim].where(~np.isnan(ds[variable]))
    masked_alt.name = "int_alt"
    int_masked = masked_alt.interpolate_na(
        dim=alt_dim,
        method="nearest",
        fill_value="extrapolate",
    )
    return np.abs(ds[alt_dim] - int_masked)
