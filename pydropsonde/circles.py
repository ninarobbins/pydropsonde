from dataclasses import dataclass
import metpy.calc as mpcalc
from metpy.units import units

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
