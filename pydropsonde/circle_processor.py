from dataclasses import dataclass


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
