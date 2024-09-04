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
