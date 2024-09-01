import ast
from dataclasses import dataclass, field, KW_ONLY
from datetime import datetime
from typing import Any, Optional, List
import os
import subprocess
import warnings
import yaml
import glob

import numpy as np
import xarray as xr

import pydropsonde.helper as hh

from sklearn import linear_model
import metpy.calc as mpcalc
from metpy.units import units
from tqdm import tqdm
import circle_fit as cf

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


@dataclass(order=True)
class Circle_Gridded:

    circles: dict

    def concat_circles(self):
        """
        function to concatenate all circles using the combination of all measurement times and launch times
        """
        list_of_circle_ds = [circle for circle in self.circles.values()]
        combined = xr.combine_by_coords(list_of_circle_ds)
        self._interim_l4_ds = combined
        return self

    def get_l4_dir(self, l4_dir: str = None):
        if l4_dir:
            self.l4_dir = l4_dir
        elif not self.sondes is None:
            self.l4_dir = (
                list(self.sondes.values())[0]
                .l3_dir.replace("Level_3", "Level_4")
                .replace(list(self.sondes.values())[0].flight_id, "")
                .replace(list(self.sondes.values())[0].platform_id, "")
            )
        else:
            raise ValueError("No sondes and no l3 directory given, cannot continue ")
        return self

    def get_l4_filename(
        self, l4_filename_template: str = None, l4_filename: str = None
    ):
        if l4_filename is None:
            if l4_filename_template is None:
                l4_filename = "some_default_template_{platform}_{flight_id}.nc".format(
                    platform=self.platform_id,
                    flight_id=self.flight_id,
                )
            else:
                l4_filename = l4_filename_template.format(
                    platform=self.platform_id,
                    flight_id=self.flight_id,
                )

        self.l4_filename = l4_filename

        return self

    def write_l4(self, l4_dir: str = None):
        if l4_dir is None:
            l4_dir = self.l4_dir

        if not os.path.exists(l4_dir):
            os.makedirs(l4_dir)

        self._interim_l4_ds.to_netcdf(os.path.join(l4_dir, self.l4_filename))

        return self
