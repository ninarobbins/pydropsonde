import pytest
from pydropsonde.pipeline import iterate_Sonde_method_over_dict_of_Sondes_objects
from pydropsonde.processor import Sonde
import configparser

config = configparser.ConfigParser()
config.add_section("MANDATORY")
config.set("MANDATORY", "a", "1")


@pytest.fixture
def sonde_dict():
    return {i: Sonde(str(i)) for i in range(3)}


def test_sonde_iterator(sonde_dict):
    res = []

    def collect_sonde_serial_id(sonde: Sonde) -> Sonde:
        res.append(sonde.serial_id)
        return sonde

    sonde_dict = iterate_Sonde_method_over_dict_of_Sondes_objects(
        obj=sonde_dict,
        functions=[collect_sonde_serial_id],
        config=config,
    )
    assert res == ["0", "1", "2"]
    assert len(sonde_dict) == 3
