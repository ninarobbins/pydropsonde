from dataclasses import dataclass
import numpy as np
import xarray as xr
import circle_fit as cf
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

        circle_y = np.nanmean(c_yc) / (110.54 * 1000)
        circle_x = np.nanmean(c_xc) / (111.320 * np.cos(np.radians(circle_y)) * 1000)
        circle_diameter = np.nanmean(c_r) * 2

        xc = [None] * len(x_coor.T)
        yc = [None] * len(y_coor.T)

        xc = np.mean(x_coor, axis=0)
        yc = np.mean(y_coor, axis=0)

        delta_x = x_coor - xc  # *111*1000 # difference of sonde long from mean long
        delta_y = y_coor - yc  # *111*1000 # difference of sonde lat from mean lat

        new_vars = dict(
            circle_flight_altitude=self.circle_ds["aircraft_geopotential_altitude"]
            .mean()
            .values,
            circle_time=self.circle_ds["launch_time"].mean().values,
            circle_lon=circle_x,
            circle_lat=circle_y,
            circle_diameter=circle_diameter,
            dx=(["sonde_id", "alt"], delta_x.values),
            dy=(["sonde_id", "alt"], delta_y.values),
        )
        self.circle_ds = self.circle_ds.assign(new_vars)

        return self

    def fit2d(self, var: str = None):
        """
        Estimate a 2D linear model to calculate u-values from x-y coordinates.

        :param var: Variable name to extract from the dataset.
        :returns: intercept, dudx, dudy, and the original dataset.
        """
        u = self.circle_ds[var]
        # to fix nans, do a copy
        u_cal = u.copy()
        # a does not need to be copied as this creates a copy already
        a = np.stack(
            [np.ones_like(self.circle_ds.dx), self.circle_ds.dx, self.circle_ds.dy],
            axis=-1,
        )

        # for handling missing values, both u and a are set to 0, that way
        # these items don't influence the fit
        invalid = (
            np.isnan(u_cal) | np.isnan(self.circle_ds.dx) | np.isnan(self.circle_ds.dy)
        )
        under_constraint = np.sum(~invalid, axis=-1) < 6
        # Use `.where()` to mask invalid elements with 0
        u_cal = u_cal.where(~invalid, 0)
        a[invalid] = 0  # Broadcasting the mask

        a_inv = np.linalg.pinv(a)

        intercept, dudx, dudy = np.einsum("...rm,...m->r...", a_inv, u_cal)

        intercept[under_constraint] = np.nan
        dudx[under_constraint] = np.nan
        dudy[under_constraint] = np.nan

        return intercept, dudx, dudy

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
            intercept, dudx, dudy = self.fit2d_xr(var)

            object.__setattr__(self, mean_var_name, intercept)
            object.__setattr__(self, var_dx_name, dudx)
            object.__setattr__(self, var_dy_name, dudy)

        return self
