import configparser
import pytest
from pydropsonde.pipeline import (
    get_mandatory_args,
    get_mandatory_values_from_config,
    get_nondefaults_from_config,
    get_args_for_function,
)


# Define a function for testing
def some_func(self, a, b=2):
    pass


@pytest.fixture
def config_and_function():
    # Create a ConfigParser object and add a section for the test function
    config = configparser.ConfigParser()
    config.add_section("test_inspect_function.some_func")
    config.set("test_inspect_function.some_func", "b", "3")
    config.add_section("MANDATORY")
    config.set("MANDATORY", "a", "1")

    if "pydropsonde" not in some_func.__module__:
        some_func.__module__ = f"pydropsonde.{some_func.__module__}"
    return config, some_func


def test_get_mandatory_args(config_and_function):
    _, test_func = config_and_function
    result = get_mandatory_args(test_func)
    assert result == ["a"]


def test_get_mandatory_values_from_config(config_and_function):
    config, _ = config_and_function
    result = get_mandatory_values_from_config(config, ["a"])
    assert result == {"a": "1"}


def test_get_nondefaults_from_config(config_and_function):
    config, test_func = config_and_function
    result = get_nondefaults_from_config(config, test_func)
    assert result == {"b": "3"}


def test_get_args_for_function(config_and_function):
    config, test_func = config_and_function
    result = get_args_for_function(config, test_func)
    assert result == {"a": "1", "b": "3"}


def test_get_mandatory_values_from_config_no_section(config_and_function):
    config, _ = config_and_function
    config.remove_section("MANDATORY")
    with pytest.raises(ValueError, match="MANDATORY section not found in config file"):
        get_mandatory_values_from_config(config, ["a"])


def test_get_mandatory_values_from_config_no_arg(config_and_function):
    config, _ = config_and_function
    config.remove_option("MANDATORY", "a")
    with pytest.raises(
        ValueError, match="Mandatory argument a not found in config file"
    ):
        get_mandatory_values_from_config(config, ["a"])


def test_get_args_for_function_no_mandatory(config_and_function):
    config, test_func = config_and_function
    config.remove_option("MANDATORY", "a")
    with pytest.raises(
        ValueError, match="Mandatory argument a not found in config file"
    ):
        get_args_for_function(config, test_func)
