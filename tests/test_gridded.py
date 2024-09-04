import pytest
from pydropsonde.processor import Gridded

sondes = None
circles = None
l3_default = "Level_3.nc"
l4_default = "Level_4.nc"


@pytest.fixture
def gridded():
    return Gridded(sondes, circles)


def test_l3_dir(gridded):
    with pytest.raises(ValueError):
        gridded.get_l3_dir()


def test_l3_dir_name(gridded):
    gridded.get_l4_dir(l3_dir="test")
    assert gridded.l4_dir == "test"


def test_l3_default(gridded):
    gridded.get_l3_filename()
    assert gridded.l4_filename == l3_default


def test_l4_dir(gridded):
    with pytest.raises(ValueError):
        gridded.get_l4_dir()


def test_l4_dir_name(gridded):
    gridded.get_l4_dir(l3_dir="test")
    assert gridded.l4_dir == "test"


def test_l4_default(gridded):
    gridded.get_l4_filename()
    assert gridded.l4_filename == l4_default
