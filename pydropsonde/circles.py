from dataclasses import dataclass
import numpy as np
import xarray as xr
import tqdm
import circle_fit as cf
import pydropsonde.helper.physics as hp

_no_default = object()


@dataclass(order=True)
class Circle:
    """Class identifying a circle and containing its metadata.

    A `Circle` identifies the circle data for a circle on a given flight
    """

    circle_ds: str
    clon: float
    clat: float
    crad: float
    flight_id: str
    platform_id: str
    segment_id: str
    alt_dim: str

    def drop_m_N_vars(self):
        """
        drop m and N variables from level 3 from circle dataset
        """
        ds = self.circle_ds
        ds = (
            ds.drop_vars(
                [f"{var}_m_qc" for var in ds.variables],
                errors="ignore",
            )
            .drop_vars(
                [f"{var}_N_qc" for var in ds.variables],
                errors="ignore",
            )
            .drop_vars(
                ["gps_m_qc", "gps_N_qc", "gpspos_N_qc", "gpspos_m_qc"], errors="ignore"
            )
        )
        self.circle_ds = ds
        return self

    def get_xy_coords_for_circles(self):
        if self.circle_ds.lon.size == 0 or self.circle_ds.lat.size == 0:
            print("Empty segment: 'lon' or 'lat' is empty.")
            return None  # or some default value like [], np.array([]), etc.

        x_coor = (
            self.circle_ds.lon * 111.32 * np.cos(np.radians(self.circle_ds.lat)) * 1000
        )
        y_coor = self.circle_ds.lat * 110.54 * 1000

        # converting from lat, lon to coordinates in metre from (0,0).
        if self.clat is None:
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

            self.clat = np.nanmean(c_yc) / (110.54 * 1000)
            self.clon = np.nanmean(c_xc) / (
                111.32 * np.cos(np.radians(self.clat)) * 1000
            )

            self.crad = np.nanmean(c_r)
            attr_descr = "fitted circle for all regressed sondes in circle (mean)"
        else:
            attr_descr = "circle from flight segmentation"

        yc = self.clat * 110.54 * 1000
        xc = self.clon * (111.32 * np.cos(np.radians(self.clat)) * 1000)

        delta_x = x_coor - xc
        delta_y = y_coor - yc

        delta_x_attrs = {
            "long_name": "x",
            "description": "Distance of sonde longitude to mean circle longitude",
            "units": "m",
        }
        delta_y_attrs = {
            "long_name": "y",
            "description": "Distance of sonde latitude to mean circle latitude",
            "units": "m",
        }
        circle_radius_attrs = {
            "long_name": "circle_radius",
            "description": f"Radius of {attr_descr}",
            "units": "m",
        }
        circle_lon_attrs = {
            "long_name": "circle_lon",
            "description": f"Longitude of {attr_descr}",
            "units": self.circle_ds.lon.attrs["units"],
        }
        circle_lat_attrs = {
            "long_name": "circle_lat",
            "description": f"Latitude of {attr_descr}",
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
            circle_lon=([], self.clon, circle_lon_attrs),
            circle_lat=([], self.clat, circle_lat_attrs),
            circle_radius=([], self.crad, circle_radius_attrs),
            x=(["sonde_id", self.alt_dim], delta_x.values, delta_x_attrs),
            y=(["sonde_id", self.alt_dim], delta_y.values, delta_y_attrs),
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

    def apply_fit2d(self, variables=None):
        if variables is None:
            variables = ["u", "v", "q", "ta", "p", "density"]
        alt_var = self.alt_dim
        alt_attrs = self.circle_ds[alt_var].attrs

        assign_dict = {}

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
        ds = ds.set_coords("circle_time")

        self.circle_ds = ds
        return self

    def add_density(self, sonde_dim="sonde_id", alt_dim="gpsalt"):
        """
        Calculate and add the density to the circle dataset.

        This method computes each sondes density.
        The result is added to the dataset.

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        assert ds.p.attrs["units"] == "Pa"
        assert ds.ta.attrs["units"] == "K"
        density = hp.density(
            ds.p,
            ds.ta,
            hp.q2mr(ds.q),
        )
        density_attrs = {
            "standard_name": "air_density",
            "long_name": "Air density",
            "units": "kg m-3",
        }
        self.circle_ds = ds.assign(
            dict(
                density=(ds.ta.dims, density.values, density_attrs),
            )
        )
        return self

    def add_divergence(self):
        """
        Calculate and add the divergence to the circle dataset.

        This method computes the area-averaged horizontal mass divergence.
        The result is added to the dataset.

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        D = ds.dudx + ds.dvdy
        D_attrs = {
            "standard_name": "divergence_of_wind",
            "long_name": "Area-averaged horizontal mass divergence",
            "units": "s-1",
        }
        self.circle_ds = ds.assign(div=(ds.dudx.dims, D.values, D_attrs))
        return self

    def add_vorticity(self):
        """
        Calculate and add the vorticity to the circle dataset.

        This method computes the area-averaged horizontal vorticity.
        The result is added to the dataset.

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        vor = ds.dvdx - ds.dudy
        vor_attrs = {
            "standard_name": "atmosphere_relative_vorticity",
            "long_name": "Area-averaged horizontal relative vorticity",
            "units": "s-1",
        }
        self.circle_ds = ds.assign(vor=(ds.dudx.dims, vor.values, vor_attrs))
        return self

    def add_omega(self):
        """
        Calculate vertical pressure velocity as
        \int div dp

        This calculates the vertical pressure velocity as described in
        Bony and Stevens 2019

        Returns:
            self: circle object with updated circle_ds
        """
        ds = self.circle_ds
        alt_dim = self.alt_dim
        div = ds.div.where(~np.isnan(ds.div), drop=True).sortby(alt_dim)
        p = ds.mean_p.where(~np.isnan(ds.div), drop=True).sortby(alt_dim)
        zero_vel = xr.DataArray(data=[0], dims=alt_dim, coords={alt_dim: [0]})
        pres_diff = xr.concat([zero_vel, p.diff(dim=alt_dim)], dim=alt_dim)
        del_omega = -div * pres_diff.values
        omega = del_omega.cumsum(dim=alt_dim) * 0.01 * 60**2
        omega_attrs = {
            "standard_name": "vertical_air_velocity_expressed_as_tendency_of_pressure",
            "long_name": "Area-averaged atmospheric pressure velocity (omega)",
            "units": "hPa hr-1",
        }
        self.circle_ds = ds.assign(
            dict(omega_p=(ds.div.dims, omega.values, omega_attrs))
        )
        return self
