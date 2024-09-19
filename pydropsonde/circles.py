from dataclasses import dataclass
import numpy as np
import xarray as xr
import circle_fit as cf
import warnings
import tqdm

_no_default = object()


@dataclass(order=True)
class Circle:
    """Class identifying a circle and containing its metadata.

    A `Circle` identifies the circle data for a circle on a given flight
    """

    circle_ds: str
    flight_id: str
    platform_id: str
    segment_id: str

    def dummy_circle_function(self):
        print(self.flight_id, self.segment_id, self.platform_id)
        return self

    def get_xy_coords_for_circles(self):
        x_coor = (
            self.circle_ds.lon * 111.320 * np.cos(np.radians(self.circle_ds.lat)) * 1000
        )
        y_coor = self.circle_ds.lat * 110.54 * 1000

        # converting from lat, lon to coordinates in metre from (0,0).

        c_xc = np.full(np.size(x_coor, 1), np.nan)
        c_yc = np.full(np.size(x_coor, 1), np.nan)
        c_r = np.full(np.size(x_coor, 1), np.nan)

        for j in range(np.size(x_coor, 1)):
            a = ~np.isnan(x_coor.values[:, j])
            if a.sum() > 4:
                c_xc[j], c_yc[j], c_r[j], _ = cf.least_squares_circle(
                    [
                        (x, y)
                        for x, y in zip(x_coor.values[:, j], y_coor.values[:, j])
                        if ~np.isnan(x)
                    ]
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            circle_y = np.nanmean(c_yc) / (110.54 * 1000)
            circle_x = np.nanmean(c_xc) / (
                111.320 * np.cos(np.radians(circle_y)) * 1000
            )
            circle_diameter = np.nanmean(c_r) * 2

        xc = [None] * len(x_coor.T)
        yc = [None] * len(y_coor.T)

        xc = np.mean(x_coor, axis=0)
        yc = np.mean(y_coor, axis=0)

        delta_x = x_coor - xc  # *111*1000 # difference of sonde long from mean long
        delta_y = y_coor - yc  # *111*1000 # difference of sonde lat from mean lat

        object.__setattr__(self, "circle_lon", circle_x)
        object.__setattr__(self, "circle_lat", circle_y)
        object.__setattr__(self, "circle_diameter", circle_diameter)
        object.__setattr__(self, "dx", delta_x)
        object.__setattr__(self, "dy", delta_y)

        return self

    def fit2d(self, var: str = None):
        """
        Estimate a 2D linear model to calculate u-values from x-y coordinates.

        :param var: Variable name to extract from the dataset.
        :returns: intercept, dudx, dudy, and the original dataset.
        """
        u = self.circle_ds[var]
        u_ = u
        # Fix NaNs
        u = np.array(u, copy=True)

        # a is created by stacking ones_like(self.dx), self.dx, and self.dy
        a = np.stack([np.ones_like(self.dx), self.dx, self.dy], axis=-1)

        # Handle missing values
        invalid = np.isnan(u) | np.isnan(self.dx) | np.isnan(self.dy)
        under_constraint = np.sum(~invalid, axis=-1) < 6
        u[invalid] = 0
        a[invalid] = 0

        a_inv = np.linalg.pinv(a)

        # Fit the model using np.einsum
        intercept, dudx, dudy = np.einsum("...rm,...m->r...", a_inv, u)

        # Set under-constrained values to NaN
        intercept[under_constraint] = np.nan
        dudx[under_constraint] = np.nan
        dudy[under_constraint] = np.nan

        return intercept, dudx, dudy, u_

    def fit2d_xr(
        self,
        var: str = None,
        input_core_dims=["sonde_id"],
        output_core_dims=["sonde_id"],
    ):
        """
        Apply the fit2d method to the xarray object using xr.apply_ufunc.

        :param var: Variable name for the dataset.
        :returns: Xarray object with intercept, dudx, dudy, and original dataset.
        """
        # Note that we do not pass self.dx and self.dy here; they are used internally by fit2d.
        return xr.apply_ufunc(
            lambda v: self.fit2d(v),
            var,
            input_core_dims=[input_core_dims],
            output_core_dims=[(), (), (), output_core_dims],
            vectorize=True,
        )

    def fit_multiple_vars(self, variables: list = ["u", "v", "q", "ta", "p"]):
        """
        Apply the 2D linear fit for multiple variables (e.g., u, v, q, ta, p) for this circle.

        Parameters
        ----------
        variables : list
            A list of variable names (e.g., ["u", "v", "q", "ta", "p"]) to apply fit2d_xr to.

        Returns
        -------
        dict
            A dictionary where keys are the variable names and values are the results (intercept, dudx, dudy, sounding).
        """

        # Loop over each variable and apply the fit2d_xr method
        for var in tqdm.tqdm(variables):
            mean_var_name = var
            var_dx_name = "d" + var + "dx"
            var_dy_name = "d" + var + "dy"

            # Apply the fit2d_xr method for each variable (assuming fit2d_xr is defined in the class)
            intercept, dudx, dudy, sounding = self.fit2d_xr(var)

            object.__setattr__(self, mean_var_name, intercept)
            object.__setattr__(self, var_dx_name, dudx)
            object.__setattr__(self, var_dy_name, dudy)
            object.__setattr__(self, var + "_sounding", sounding)

        return self
