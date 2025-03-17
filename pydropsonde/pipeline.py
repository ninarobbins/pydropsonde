from .helper.paths import Platform, Flight
from .helper.__init__ import (
    path_to_flight_ids,
    path_to_l0_files,
)
from .processor import Sonde, Gridded
from .circles import Circle
import configparser
import inspect
from tqdm import tqdm
import numpy as np
import os
import xarray as xr
from typing import Protocol, runtime_checkable, Optional
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class CircleProcess(Protocol):
    def __call__(self, circle: Circle, **kwargs) -> Optional[Circle]: ...


@runtime_checkable
class SondeProcess(Protocol):
    def __call__(self, sonde: Sonde, **kwargs) -> Optional[Sonde]: ...


def get_mandatory_args(function):
    """
    Get a list of all arguments that do not have a default value for each function in the list.

    Parameters
    ----------
    list_of_functions : list
        A list of functions to inspect.

    Returns
    -------
    list
        A list of argument names that do not have a default value.

    Examples
    --------
    >>> def func1(a, b=2):
    ...     pass
    >>> def func2(c, d=4, e=5):
    ...     pass
    >>> mandatory_args([func1, func2])
    ['a', 'c']
    """
    mandatory_args = []
    sig = inspect.signature(function)
    for name, param in list(sig.parameters.items())[1:]:
        if param.default == inspect.Parameter.empty:
            mandatory_args.append(name)
    return mandatory_args


def get_mandatory_values_from_config(config, mandatory_args):
    """
    Extracts mandatory values from the 'MANDATORY' section of a configuration file.

    Parameters
    ----------
    config : ConfigParser
        The configuration file parser.
    mandatory_args : list
        A list of argument names that are expected to be in the 'MANDATORY' section of the config file.

    Returns
    -------
    dict
        A dictionary where the keys are the argument names and the values are the corresponding values from the config file.

    Raises
    ------
    ValueError
        If the 'MANDATORY' section is not found in the config file or if a mandatory argument is not found in the 'MANDATORY' section.

    Examples
    --------
    >>> import configparser
    >>> config = configparser.ConfigParser()
    >>> config.read_string('[MANDATORY]\\narg1=value1\\narg2=value2')
    >>> mandatory_values_from_config(config, ['arg1', 'arg2'])
    {'arg1': 'value1', 'arg2': 'value2'}
    """
    if not config.has_section("MANDATORY"):
        raise ValueError("MANDATORY section not found in config file")
    else:
        mandatory_dict = {}
        for arg in mandatory_args:
            if config.has_option("MANDATORY", arg):
                mandatory_dict[arg] = config.get("MANDATORY", arg)
            else:
                raise ValueError(f"Mandatory argument {arg} not found in config file")
    return mandatory_dict


def get_nondefaults_from_config(
    config: configparser.ConfigParser, obj: callable
) -> dict:
    """
    Get the non-default arguments for a given function from a ConfigParser object.

    Parameters
    ----------
    config : configparser.ConfigParser
        A ConfigParser object containing configuration settings.
    obj : callable
        The function for which to get the non-default arguments.

    Returns
    -------
    dict
        A dictionary of non-default arguments for the function.
    """

    try:
        section_name = f"{obj.__module__}.{obj.__qualname__}".split("pydropsonde.")[1]
    except IndexError:
        section_name = None

    if section_name in config.sections():
        nondefault_args = config[section_name]
    else:
        nondefault_args = {}
    return nondefault_args


def get_args_for_function(config, function):
    """
    Get the arguments for a given function.

    This function first checks if the qualified name of the function exists in the nondefaults dictionary.
    If it does, it uses the corresponding value as the arguments for the function.
    Otherwise, it initializes an empty dictionary as the arguments.

    Then, it gets the list of mandatory arguments for the function.
    If there are any mandatory arguments, it gets their values from the config and updates the arguments dictionary with them.

    Parameters
    ----------
    function : function
        The function for which to get the arguments.
    nondefaults : dict
        A dictionary mapping function qualified names to dictionaries of arguments.

    Returns
    -------
    dict
        A dictionary of arguments for the function.
    """
    args = get_nondefaults_from_config(config, function)
    mandatory = get_mandatory_args(function)
    if mandatory:
        mandatory_args = get_mandatory_values_from_config(config, mandatory)
        args.update(mandatory_args)

    return args


def get_platforms(config):
    """
    Get platforms based on the directory names in `data_directory` or the user-provided `platforms` values.

    Parameters
    ----------
    config : ConfigParser instance
        The configuration file parser.

    Returns
    -------
    dict
        A dictionary where keys are platform names and values are `Platform` objects.

    Raises
    ------
    ValueError
        If `platforms` is specified in the config file but `platform_directory_names` is not, or
        if a value in `platform_directory_names` does not correspond to a directory in `data_directory`.

    """
    data_directory = config.get("MANDATORY", "data_directory")
    path_structure = config.get(
        "OPTIONAL", "path_to_flight_ids", fallback=path_to_flight_ids
    )
    if config.has_option("OPTIONAL", "platforms"):
        if not config.has_option("OPTIONAL", "platform_directory_names"):
            raise ValueError(
                "platform_directory_names must be provided in the config file when platforms is specified"
            )
        platforms = config.get("OPTIONAL", "platforms").split(",")
        platform_directory_names = config.get(
            "OPTIONAL", "platform_directory_names"
        ).split(",")
        platforms = dict(zip(platforms, platform_directory_names))
        for directory_name in platform_directory_names:
            if not os.path.isdir(os.path.join(data_directory, directory_name)):
                raise ValueError(
                    f"No directory found for {directory_name} in data_directory"
                )
        platform_objects = {}
        for platform, platform_directory_name in platforms.items():
            platform_objects[platform] = Platform(
                data_directory=data_directory,
                platform_id=platform,
                platform_directory_name=platform_directory_name,
                path_structure=path_structure,
            )
    else:
        platforms = [
            name
            for name in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory, name))
        ]
        platform_objects = {}
        for platform in platforms:
            platform_objects[platform] = Platform(
                data_directory=data_directory,
                platform_id=platform,
                path_structure=path_structure,
            )
    return platform_objects


def create_and_populate_flight_object(
    config: configparser.ConfigParser,
) -> (dict[Platform], dict[Sonde]):
    """
    Creates a Flight object and populates it with D-files and optional information from A-files.

    Parameters
    ----------
    config : configparser.ConfigParser
        A ConfigParser object containing configuration settings.

    Returns
    -------
    Flight
        A Flight object.
    """
    output = {}

    platform_objects = get_platforms(config)
    path_structure = config.get(
        "OPTIONAL", "path_to_l0_files", fallback=path_to_l0_files
    )
    output["platforms"] = platform_objects
    output["sondes"] = []

    for platform in platform_objects:
        for flight_id in platform_objects[platform].flight_ids:
            flight = Flight(
                platform_objects[platform].data_directory,
                flight_id,
                platform,
                path_structure=path_structure,
            )

            output["sondes"] += flight.populate_sonde_instances(config)
    return output["platforms"], output["sondes"]


def create_and_populate_circle_object(
    gridded: Gridded, config: configparser.ConfigParser
) -> dict[Circle]:
    """
    create circle objects for further analysis
    """

    circles = {}
    ds = gridded.interim_l4_ds
    for segment in gridded.segments:
        extra_sondes = ds.where(
            ds["sonde_id"].isin(segment.get("extra_sondes")), drop=True
        )

        circle_ds = ds.where(
            (ds["sonde_time"] >= np.datetime64(segment["start"]))
            & (ds["sonde_time"] < np.datetime64(segment["end"])),
            drop=True,
        )
        circle_ds = xr.concat(
            [circle_ds, extra_sondes],
            dim=gridded.sonde_dim,
            join="exact",
            combine_attrs="no_conflicts",
        )
        if circle_ds.sonde_id.size > 0:
            circle = Circle(
                circle_ds=circle_ds.sortby("sonde_time"),
                flight_id=segment["flight_id"],
                platform_id=segment["platform_id"],
                segment_id=segment["segment_id"],
                alt_dim=gridded.alt_dim,
                sonde_dim=gridded.sonde_dim,
                clon=segment.get("clon"),
                clat=segment.get("clat"),
                crad=segment.get("radius"),
            )
            circles[segment["segment_id"]] = circle
        else:
            print(f"No data for segment {segment['segment_id']}")

    gridded.circles = circles

    return gridded


def iterate_Sonde_method_over_list_of_Sondes_objects(
    sondes: list[Sonde], functions: list, config: configparser.ConfigParser
) -> list[Sonde]:
    """
    Iterates over a list of Sonde objects and applies a list of methods to each Sonde.

    For each Sonde object in the list, this function
    applies each method listed in the 'functions' key of the substep dictionary.
    If the method returns a value, it stores the value in a new list.
    If the method returns None, it does not store the value in the new list.

    The arguments for each method are determined by the `get_args_for_function` function,
    which uses the nondefaults dictionary and the config object.
    """

    for function in functions:
        new_sondes = []
        for sonde in tqdm(sondes):
            if not callable(function):
                function = getattr(Sonde, function)
                assert isinstance(function, SondeProcess)
            if sonde.cont:
                logger.debug(f"Applying {function.__name__} to Sonde {sonde}")
                result = function(sonde, **get_args_for_function(config, function))
                if result is not None:
                    new_sondes.append(result)
                else:
                    logger.debug(
                        f"Sonde {sonde} was dropped after applying {function.__name__}"
                    )
            else:
                new_sondes.append(sonde)
        sondes = new_sondes
    return sondes


def iterate_Circle_method_over_dict_of_Circle_objects(
    obj: Gridded,
    functions: list[str | CircleProcess],
    config: configparser.ConfigParser,
) -> object:
    """
    Iterates over a dictionary of Circle objects and applies a list of methods to each Circle.

    For each Circle object in the dictionary, this function
    applies each method listed in the 'functions' key of the substep dictionary.
    If the method returns a value, it stores the value in a new dictionary.
    If the method returns None, it does not store the value in the new dictionary.

    The arguments for each method are determined by the `get_args_for_function` function,
    which uses the nondefaults dictionary and the config object.

    Parameters
    ----------
    obj : dict
        A dictionary of Circle objects.
    functions : list
        a list of method names.
    nondefaults : dict
        A dictionary mapping function qualified names to dictionaries of arguments.
    config : configparser.ConfigParser
        A ConfigParser object containing configuration settings.

    Returns
    -------
    dict
        A dictionary of Circle objects with the results of the methods applied to them (keys where results are None are not included).
    """

    my_dict = obj.circles

    for function in functions:
        new_dict = {}
        for key, value in my_dict.items():
            if not callable(function):
                function = getattr(Circle, function)
                assert isinstance(function, CircleProcess)
            result = function(value, **get_args_for_function(config, function))
            if result is not None:
                new_dict[key] = result

        my_dict = new_dict

    obj.circles.update(my_dict)
    return obj


def sondes_to_gridded(sondes: list[Sonde], config: configparser.ConfigParser):
    gridded = Gridded(sondes, global_attrs=sondes[0].global_attrs)
    return gridded


def apply_method_to_dataset(
    obj: Gridded,
    functions: list,
    config: configparser.ConfigParser,
) -> xr.Dataset:
    """
    This is NOT what the function should do in the end. only used to save the base-l3
    """
    for function_name in functions:
        function = getattr(Gridded, function_name)
        result = function(obj, **get_args_for_function(config, function))
    return result


def run_substep(
    previous_substep_output, substep: dict, config: configparser.ConfigParser
):
    """
    This function applies a specified function to the input data and stores the output for use in subsequent steps.

    Parameters
    ----------
    previous_substep_output : dict
        A dictionary storing the output data from previous steps. The input data for this step is retrieved from this dictionary using the key specified in substep['intake'], and the output of this step is stored in this dictionary under the key(s) specified in substep['output'].
    substep : dict
        A dictionary containing information about the current processing step. It should have the following keys:
        - 'apply': a function to apply to the input data.
        - 'intake': the key in the previous_substep_output dictionary that corresponds to the input data for this step.
        - 'output': the key(s) under which to store the output of this step in the previous_substep_output dictionary. If this is a list, the function should return a list of outputs of the same length.
        - 'functions' (optional): a list of functions to apply to the input data. If this key is present, the 'apply' function should take this list as an additional argument.
    config : object
        A configuration object used by the function.

    Returns
    -------
    dict
        The updated dictionary with the output data from the current step. The output data is stored under the key(s) specified in substep['output'].

    Notes
    -----
    This function assumes that the 'apply' function returns a list of outputs if substep['output'] is a list, and a single output otherwise. If substep['output'] is a list but the 'apply' function does not return a list of outputs, or if the lengths of the two lists do not match, this function will raise an exception.
    """
    function = substep["apply"]
    intake = substep["intake"]
    if "functions" not in substep:
        if intake is None:
            previous_substep_output = {}
            output = function(config)
        else:
            output = function(previous_substep_output[intake], config)
    else:
        output = function(previous_substep_output[intake], substep["functions"], config)

    if isinstance(substep["output"], list):
        for i, key in enumerate(substep["output"]):
            previous_substep_output[key] = output[i]
    else:
        previous_substep_output[substep["output"]] = output

    return previous_substep_output


def verify_unique_sonde_ids(
    sondes: list[Sonde], config: configparser.ConfigParser
) -> list[Sonde]:
    ids_so_far = set()
    for sonde in sondes:
        if sonde.id in ids_so_far:
            raise ValueError(f"Duplicate sonde ID: {sonde.id}")
        ids_so_far.add(sonde.id)
    return sondes


def run_pipeline(pipeline: dict, config: configparser.ConfigParser):
    """
    Executes a pipeline of processing steps.

    Parameters:
    ----------
    pipeline : dict
        A dictionary representing the pipeline
        where each key is a substep and the value is a dictionary with the configurations of that substep.
    config : configparser.ConfigParser
        Configuration settings for the package.

    Returns:
    ----------
    dict:
        The output of the last substep in the pipeline.
    """
    previous_substep_output = None
    for step in pipeline:
        print(f"Running {step}...")
        substep = pipeline[step]
        if previous_substep_output is None:
            previous_substep_output = run_substep(None, substep, config)
        else:
            previous_substep_output = run_substep(
                previous_substep_output, substep, config
            )
    return previous_substep_output


pipeline = {
    "create_flight": {
        "intake": None,
        "apply": create_and_populate_flight_object,
        "output": ["platforms", "sondes"],
    },
    "create_L1": {
        "intake": "sondes",
        "apply": iterate_Sonde_method_over_list_of_Sondes_objects,
        "functions": [
            "filter_no_launch_detect",
            "run_aspen",
            "add_aspen_ds",
            "add_aspen_history",
        ],
        "output": "sondes",
    },
    "qc": {
        "intake": "sondes",
        "apply": iterate_Sonde_method_over_list_of_Sondes_objects,
        "functions": [
            "init_qc",
            "detect_floater",
            "crop_aspen_ds_to_landing_time",
            "get_flight_attributes",
            "set_alt_dim",
            "create_interim_l2_ds",
            "get_l2_variables",
            "convert_to_si",
            "below_aircraft_qc",
            "get_qc",
            #            "remove_non_qc_sondes",
        ],
        "output": "sondes",
    },
    "build_L2": {
        "intake": "sondes",
        "apply": iterate_Sonde_method_over_list_of_Sondes_objects,
        "functions": [
            "get_sonde_attributes",
            "add_l2_attributes_to_interim_l2_ds",
            "add_sonde_id_variable",
            "add_platform_id_variable",
            "add_flight_id_variable",
            "add_qc_to_l2",
            "get_l2_filename",
            "update_history_l2",
            "write_l2",
        ],
        "output": "sondes",
        "comment": "This steps creates and saves the L2 data after the QC.",
    },
    "process_L2": {
        "intake": "sondes",
        "apply": iterate_Sonde_method_over_list_of_Sondes_objects,
        "functions": [
            "check_interim_l3",
            "get_l2_filename",
            "add_l2_ds",
            "create_interim_l3",
            "replace_alt_dim",
            "remove_above_aircraft",
            "remove_unphysical",
            "add_q_and_theta_to_l2_ds",
            "remove_non_mono_incr_alt",
            "swap_alt_dimension",
            "interpolate_alt",
            "recalc_rh_and_ta",
            "add_ids",
            "add_wind",
            "add_attributes_as_var",
            "make_attr_coordinates",
            "add_qc_to_interim_l3",
            "add_iwv",
            "add_Nm_to_vars",
            "update_history_l3",
            "add_expected_coords",
            "save_interim_l3",
        ],
        "output": "sondes",
        "comment": "This step reads from the saved L2 files and prepares individual sonde datasets before they can be concatenated to create L3.",
    },
    "verify_unique_sonde_ids": {
        "intake": "sondes",
        "apply": verify_unique_sonde_ids,
        "output": "sondes",
        "comment": "This steps checks L3 interim sondes for uniqueness.",
    },
    "create_gridded": {
        "intake": "sondes",
        "apply": sondes_to_gridded,
        "output": "gridded",
        "comment": "This step concatenates the individual sonde datasets to create the L3 dataset.",
    },
    "create_L3": {
        "intake": "gridded",
        "apply": apply_method_to_dataset,
        "functions": [
            "check_aspen_version",
            "check_pydropsonde_version",
            "add_history_to_gridded",
            "add_dim_names",
            "concat_sondes",
            "get_l3_dir",
            "get_l3_filename",
            "update_history_concat_l3",
            "write_l3",
        ],
        "output": "gridded",
        "comment": "This step creates the L3 dataset after adding additional products.",
    },
    "create_interim_l4": {
        "intake": "gridded",
        "apply": apply_method_to_dataset,
        "functions": [
            "add_l3_ds",
            "create_interim_l4",
        ],
        "output": "gridded",
        "comment": "prepare level 4 processing",
    },
    "get_circles": {
        "intake": "gridded",
        "apply": apply_method_to_dataset,
        "functions": ["get_circle_times_from_segmentation"],
        "output": "gridded",
        "comment": "get circle times and add to gridded",
    },
    "create_circles": {
        "intake": "gridded",
        "apply": create_and_populate_circle_object,
        "output": "gridded",
        "comment": "This step creates a dictionary of patterns by creating the pattern with the flight-phase segmentation file.",
    },
    "prepare_circle_dataset": {
        "intake": "gridded",
        "apply": iterate_Circle_method_over_dict_of_Circle_objects,
        "functions": [
            "get_xy_coords_for_circles",
            "drop_vars",
            "interpolate_na_sondes",
        ],
        "output": "gridded",
        "comment": "prepare circle dataset for calculation",
    },
    "calculate_circle_data": {
        "intake": "gridded",
        "apply": iterate_Circle_method_over_dict_of_Circle_objects,
        "functions": [
            "apply_fit2d",
            "add_divergence",
            "add_vorticity",
            "add_omega",
            "add_wvel",
            "add_circle_variables_to_ds",
            "add_regression_stderr",
        ],
        "output": "gridded",
        "comment": "calculate circle products",
    },
    "concatenate_circles": {
        "intake": "gridded",
        "apply": apply_method_to_dataset,
        "functions": ["concat_circles"],
        "output": "gridded",
        "comment": "This step concatenates the individual circle datasets to create the L4 dataset.",
    },
    "create_L4": {
        "intake": "gridded",
        "apply": apply_method_to_dataset,
        "functions": [
            "get_l4_dir",
            "get_l4_filename",
            "update_history_l4",
            "write_l4",
        ],
        "output": "gridded",
        "comment": "This step concatenates circles and creates the L4 dataset after adding additional products.",
    },
}
