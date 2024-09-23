from dataclasses import dataclass

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
