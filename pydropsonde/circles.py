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

        self.circle_ds = self.circle_ds.assign(
            dict(div=(["alt"], D.values), vor=(["alt"], vor.values))
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
        self.circle_ds = self.circle_ds.assign(
            dict(density=(self.circle_ds.ta.dims, density.magnitude))
        )
        self.circle_ds["density"].attrs = {
            "standard_name": "density",
            "units": str(density.units),
        }
        self.circle_ds = self.circle_ds.assign(
            dict(mean_density=self.circle_ds["density"].mean(sonde_dim))
        )
        self.circle_ds["mean_density"].attrs = {
            "standard_name": "mean density",
            "units": str(density.units),
        }

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
        self.circle_ds = self.circle_ds.assign(dict(w_vel=w_vel))
        self.circle_ds["w_vel"].attrs = {
            "standard name": "vertical velocity",
            "units": str(units.meter / units.second),
        }

        return self

    def get_omega(self):
        p_vel = (
            -self.circle_ds.mean_density.values
            * units(self.circle_ds.mean_density.attrs["units"])
            * self.circle_ds.w_vel.values
            * units(self.circle_ds.w_vel.attrs["units"])
            * mpconst.earth_gravity
        )
        self.circle_ds = self.circle_ds.assign(
            dict(omega=(self.circle_ds.w_vel.dims, p_vel.magnitude))
        )

        return self
