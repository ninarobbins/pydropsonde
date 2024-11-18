from dataclasses import dataclass
import numpy as np
import xarray as xr
import tqdm
import circle_fit as cf

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

    def drop_m_N_vars(self):
        """
        drop m and N variables from level 3 from circle dataset
        """
        ds = self.circle_ds
        ds = ds.drop_vars(
            [f"m{var}" for var in ds.variables],
            errors="ignore",
        ).drop_vars(
            [f"N{var}" for var in ds.variables],
            errors="ignore",
        )
        self.circle_ds = ds
        return self

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

        delta_x_attrs = {
            "long_name": "x",
            "description": "Difference of sonde longitude from mean longitude",
            "units": self.circle_ds.lon.attrs["units"],
        }
        delta_y_attrs = {
            "long_name": "y",
            "description": "Difference of sonde latitude from mean latitude",
            "units": self.circle_ds.lat.attrs["units"],
        }
        circle_diameter_attrs = {
            "long_name": "circle_diameter",
            "description": "Diameter of fitted circle for all regressed sondes in circle",
            "units": "m",
        }
        circle_lon_attrs = {
            "long_name": "circle_lon",
            "description": "Longitude of fitted circle for all regressed sondes in circle",
            "units": self.circle_ds.lon.attrs["units"],
        }
        circle_lat_attrs = {
            "long_name": "circle_lat",
            "description": "Latitude of fitted circle for all regressed sondes in circle",
            "units": self.circle_ds.lat.attrs["units"],
        }
        circle_altitude_attrs = {
            "long_name": "circle_altitude",
            "description": "Mean altitude of the aircraft during the circle",
            "units": self.circle_ds.alt.attrs["units"],
        }
        circle_time_attrs = {
            "long_name": "circle_time",
            "description": "Mean launch time of all sondes in circle",
        }

        new_vars = dict(
            circle_altitude=(
                [],
                self.circle_ds["aircraft_msl_altitude"].mean().values,
                circle_altitude_attrs,
            ),
            circle_time=(
                [],
                self.circle_ds["launch_time"].mean().values,
                circle_time_attrs,
            ),
            circle_lon=([], circle_x, circle_lon_attrs),
            circle_lat=([], circle_y, circle_lat_attrs),
            circle_diameter=([], circle_diameter, circle_diameter_attrs),
            x=(["sonde_id", "alt"], delta_x.values, delta_x_attrs),
            y=(["sonde_id", "alt"], delta_y.values, delta_y_attrs),
        )

        self.circle_ds = self.circle_ds.assign(new_vars)
        return self

    @staticmethod
    def fit2d(x, y, u):
        a = np.stack([np.ones_like(x), x, y], axis=-1)

        invalid = np.isnan(u) | np.isnan(x) | np.isnan(y)
        u_cal = np.where(invalid, 0, u)
        a[invalid] = 0

        a_inv = np.linalg.pinv(a)
        intercept, dux, duy = np.einsum("...rm,...m->r...", a_inv, u_cal)

        return intercept, dux, duy

    def fit2d_xr(self, x, y, u, sonde_dim="sonde_id"):
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

    def apply_fit2d(self, alt_var="alt"):
        variables = ["u", "v", "q", "ta", "p"]
        alt_attrs = self.circle_ds[alt_var].attrs

        for par in tqdm.tqdm(variables):
            long_name = self.circle_ds[par].attrs.get("long_name")
            standard_name = self.circle_ds[par].attrs.get("standard_name")
            varnames = ["mean_" + par, "d" + par + "dx", "d" + par + "dy"]
            var_units = self.circle_ds[par].attrs.get("units", None)
            long_names = [
                "circle mean of " + long_name,
                "zonal gradient of " + long_name,
                "meridional gradient of " + long_name,
            ]
            use_names = [
                standard_name + "_circle_mean",
                "derivative_of_" + standard_name + "_wrt_x",
                "derivative_of_" + standard_name + "_wrt_y",
            ]

            results = self.fit2d_xr(
                x=self.circle_ds.x,
                y=self.circle_ds.y,
                u=self.circle_ds[par],
                sonde_dim="sonde_id",
            )

            assign_dict = {}
            for varname, result, long_name, use_name in zip(
                varnames, results, long_names, use_names
            ):
                if "mean" in varname:
                    assign_dict[varname] = (
                        [alt_var],
                        result.data,
                        {
                            "long_name": long_name,
                            "units": var_units,
                        },
                    )
                else:
                    assign_dict[varname] = (
                        [alt_var],
                        result.data,
                        {
                            "standard_name": use_name,
                            "long_name": long_name,
                            "units": f"{var_units} m-1",
                        },
                    )

            ds = self.circle_ds.assign(assign_dict)
        ds[alt_var].attrs.update(alt_attrs)
        self.circle_ds = ds
        return self
