import pytest
from pydropsonde.pipeline import iterate_Sonde_method_over_list_of_Sondes_objects
from pydropsonde.processor import Sonde
import configparser

config = configparser.ConfigParser()
config.add_section("MANDATORY")
config.set("MANDATORY", "a", "1")


@pytest.fixture
def sondes_list():
    return [Sonde(str(i)) for i in range(3)]


def test_sonde_iterator_accepts_callable_as_function(sondes_list):
    res = []

    def collect_sonde_serial_id(sonde: Sonde) -> Sonde:
        res.append(sonde.serial_id)
        return sonde

    sondes = iterate_Sonde_method_over_list_of_Sondes_objects(
        sondes=sondes_list,
        functions=[collect_sonde_serial_id],
        config=config,
    )
    assert res == ["0", "1", "2"]
    assert len(sondes) == 3
