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
        if self.circle_ds.lon.size == 0 or self.circle_ds.lat.size == 0:
            print("Empty segment: 'lon' or 'lat' is empty.")
            return None  # or some default value like [], np.array([]), etc.

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

    @staticmethod
    def fit2d(x, y, u, var=None):
        # Function logic goes here
        # No need for self here since it's now a static method
        a = np.stack([np.ones_like(x), x, y], axis=-1)

        invalid = np.isnan(u) | np.isnan(x) | np.isnan(y)
        u_cal = np.where(invalid, 0, u)
        a[invalid] = 0

        a_inv = np.linalg.pinv(a)
        intercept, dudx, dudy = np.einsum("...rm,...m->r...", a_inv, u_cal)

        return intercept, dudx, dudy

    def fit2d_xr(self, x, y, u, sonde_dim="sonde_id"):
        # Apply the static method fit2d without passing self
        return xr.apply_ufunc(
            self.__class__.fit2d,  # Call the static method without passing `self`
            x,
            y,
            u,
            input_core_dims=[
                [sonde_dim],
                [sonde_dim],
                [sonde_dim],
            ],  # Specify input dims
            output_core_dims=[(), (), ()],  # Output dimensions as scalars
        )

    def apply_fit2d(self):
        # Loop over the parameters to apply fit2d_xr to each of them
        for par in tqdm.tqdm(["u", "v", "q", "ta", "p"]):
            varnames = [par + "0", "d" + par + "dx", "d" + par + "dy"]

            # Apply fit2d_xr to each variable, and assign the result to circle_ds
            results = self.fit2d_xr(
                x=self.circle_ds.dx,
                y=self.circle_ds.dy,
                u=self.circle_ds[par],
                sonde_dim="sonde_id",
            )

            # Assign the results using the varnames list for the output variables
            self.circle_ds = self.circle_ds.assign(
                {varname: result for varname, result in zip(varnames, results)}
            )

        return self
