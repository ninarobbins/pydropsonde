from dataclasses import dataclass
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import metpy.constants as mpconst


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
        print(self.flight_id, self.segment_id)
        return self

    def get_div_and_vor(self):
        D = self.circle_ds.dudx + self.circle_ds.dvdy
        vor = self.circle_ds.dvdx - self.circle_ds.dudy

        vor_attrs = {
            "standard_name": "atmosphere_relative_vorticity",
            "long_name": "Area-averaged horizontal relative vorticity",
            "units": str(1 / units.second),
        }
        D_attrs = {
            "standard_name": "divergence_of_wind",
            "long_name": "Area-averaged horizontal mass divergence",
            "units": str(1 / units.second),
        }
        self.circle_ds = self.circle_ds.assign(
            dict(D=(["alt"], D.values, D_attrs), vor=(["alt"], vor.values, vor_attrs))
        )
        return self

    def get_density(self, sonde_dim="sonde_id"):
        mr = mpcalc.mixing_ratio_from_specific_humidity(
            self.circle_ds.q.values,
        )
        density = mpcalc.density(
            self.circle_ds.p.values * units.Pa,
            self.circle_ds.ta.values * units.kelvin,
            mr,
        )
        density_attrs = {
            "standard_name": "air_density",
            "long_name": "Air density",
            "units": str(density.units),
        }
        mean_density_attrs = {
            "standard_name": "mean_air_density",
            "long_name": "Mean air density in circle",
            "units": str(density.units),
        }
        self.circle_ds = self.circle_ds.assign(
            dict(
                density=(self.circle_ds.ta.dims, density.magnitude, density_attrs),
                mean_density=(
                    ["alt"],
                    self.circle_ds.density.mean(sonde_dim),
                    mean_density_attrs,
                ),
            )
        )

        return self

    def get_vertical_velocity(self):
        div = self.circle_ds.div.where(~np.isnan(self.circle_ds.div), drop=True).sortby(
            "alt"
        )
        zero_vel = xr.DataArray(data=[0], dims="alt", coords={"alt": [0]})

        height = xr.concat([zero_vel, div["alt"]], dim="alt")
        height_diff = height.diff(dim="alt")

        del_w = -div * height_diff.values

        w_vel = del_w.cumsum(dim="alt")
        w_vel_attrs = {
            "standard_name": "upward_air_velocity",
            "long_name": "Area-averaged vertical air velocity",
            "units": str(units.meter / units.second),
        }
        self.circle_ds = self.circle_ds.assign(
            w_vel=(["alt"], w_vel.values, w_vel_attrs)
        )

        return self

    def get_omega(self):
        # Ensure density and vertical velocity are properly aligned and use correct units
        rho = self.circle_ds.mean_density * units(
            self.circle_ds.mean_density.attrs["units"]
        )
        w = self.circle_ds.w_vel * units(self.circle_ds.w_vel.attrs["units"])

        # Calculate omega using the correct formula: omega = - rho * g * w
        g = mpconst.earth_gravity  # gravitational constant (metpy constant)

        # Compute omega (vertical pressure velocity)
        p_vel = -(rho * g * w)

        # Assign omega to the dataset with correct attributes
        omega_attrs = {
            "standard_name": "atmosphere_vertical_velocity",
            "long_name": "Area-averaged atmospheric pressure velocity",
            "units": str(p_vel.units),
        }
        self.circle_ds = self.circle_ds.assign(
            omega=(self.circle_ds.w_vel.dims, p_vel.magnitude, omega_attrs)
        )

        return self
