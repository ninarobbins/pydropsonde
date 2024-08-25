import pytest
import os
import xarray as xr
from pydropsonde.processor import Gridded

sondes = None
l3_default = f"Level_3.nc"


@pytest.fixture
def gridded():
    return Gridded(sondes)


def test_l3_dir(gridded):
    with pytest.raises(ValueError):
        gridded.get_l3_dir()


def test_l3_dir_name(gridded):
    gridded.get_l3_dir(l3_dir="test")
    assert gridded.l3_dir == "test"


def test_l3_default(gridded):
    gridded.get_l3_filename()
    assert gridded.l3_filename == l3_default
