from dataclasses import dataclass
import numpy as np
import circle_fit as cf

_no_default = object()


@dataclass(order=True)
class Circle:
    """Class identifying a circle and containing its metadata.

    A `Circle` identifies the circle averaged products from all sondes within the circle.

    Every `Circle` mandatorily has a `circle` identifier in the format "HALO-240811-c1".
    """

    circle: str
    circle_dict: str
    flight_id: str
    platform_id: str

    def get_circle_ds(self):
        """
        Retrieve the dataset for the circle from the provided dataset dictionary.

        Returns
        -------
        Dataset corresponding to the circle.
        """
        if self.circle in self.circle_dict:
            object.__setattr__(self, "circle_ds", self.circle_dict[self.circle])
        else:
            raise KeyError(f"Dataset for circle {self.circle} not found.")
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

        circle_y = np.nanmean(c_yc) / (110.54 * 1000)
        circle_x = np.nanmean(c_xc) / (111.320 * np.cos(np.radians(circle_y)) * 1000)

        circle_diameter = np.nanmean(c_r) * 2

        xc = [None] * len(x_coor.T)
        yc = [None] * len(y_coor.T)

        xc = np.mean(x_coor, axis=0)
        yc = np.mean(y_coor, axis=0)

        delta_x = x_coor - xc  # *111*1000 # difference of sonde long from mean long
        delta_y = y_coor - yc  # *111*1000 # difference of sonde lat from mean lat

        self.circle_ds.platform_id = self.circle_ds.platform_id.values[0]
        self.circle_ds.flight_altitude = self.circle_ds.flight_altitude.mean().values
        self.circle_ds.circle_time = self.circle_ds.launch_time.mean().values.astype(
            "datetime64"
        )

        object.__setattr__(self, "circle_lon", circle_x)
        object.__setattr__(self, "circle_lat", circle_y)
        object.__setattr__(self, "circle_diameter", circle_diameter)
        object.__setattr__(self, "dx", (["sounding", "alt"], delta_x))
        object.__setattr__(self, "dy", (["sounding", "alt"], delta_y))

        return self
