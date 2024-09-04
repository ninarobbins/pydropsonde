from dataclasses import dataclass

import xarray as xr


_no_default = object()


@dataclass(order=True)
class Circle:
    """Class identifying a circle and containing its metadata.

    A `Circle` identifies the circle averaged products from all sondes within the circle.

    Every `Circle` mandatorily has a `circle` identifier in the format "HALO-240811-c1".
    """

    circle_ds: xr.Dataset
    circle: str
    flight_id: str
    platform_id: str
