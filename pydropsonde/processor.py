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
from ._version import __version__

from sklearn import linear_model
import metpy.calc as mpcalc
from metpy.units import units
from tqdm import tqdm
import circle_fit as cf


_no_default = object()


@dataclass(order=True, frozen=True)
class Sonde:
    """Class identifying a sonde and containing its metadata

    A `Sonde` identifies an instrument that has been deployed. This means that pre-initialization sondes do not exist for this class.

    Every `Sonde` mandatorily has a `serial_id` which is unique. Therefore, all instances with the same `serial_id` are to be considered as having the same metadata and data.

    Optionally, the `sonde` also has metadata attributes, which can be broadly classified into:

    - campaign and flight information
    - location and time information of launch
    - performance of the instrument and sensors
    - other information such as reconditioning status, signal strength, etc.
    """

    sort_index: np.datetime64 = field(init=False, repr=False)
    serial_id: str
    cont: bool = True
    _: KW_ONLY
    launch_time: Optional[Any] = None

    def __post_init__(self):
        """
        Initializes the 'qc' attribute as an empty object and sets the 'sort_index' attribute based on 'launch_time'.

        The 'sort_index' attribute is only applicable when 'launch_time' is available. If 'launch_time' is None, 'sort_index' will not be set.
        """
        object.__setattr__(self, "qc", type("", (), {})())
        if self.launch_time is not None:
            object.__setattr__(self, "sort_index", self.launch_time)

    def add_flight_id(self, flight_id: str, flight_template: str = None) -> None:
        """Sets attribute of flight ID

        Parameters
        ----------
        flight_id : str
            The flight ID of the flight during which the sonde was launched
        """
        if not flight_template is None:
            flight_id = flight_template.format(flight_id=flight_id)

        object.__setattr__(self, "flight_id", flight_id)

    def add_platform_id(self, platform_id: str) -> None:
        """Sets attribute of platform ID

        Parameters
        ----------
        platform_id : str
            The platform ID of the flight during which the sonde was launched
        """
        object.__setattr__(self, "platform_id", platform_id)

    def add_spatial_coordinates_at_launch(self, launch_coordinates: List) -> None:
        """Sets attributes of spatial coordinates at launch

        Expected units for altitude, latitude and longitude are
        meter above sea level, degree north and degree east, respectively.

        Parameters
        ----------
        launch_coordinates : List
            List must be provided in the order of [`launch_alt`,`launch_lat`,`launch_lon`]
        """
        try:
            launch_alt, launch_lat, launch_lon = launch_coordinates
            object.__setattr__(self, "launch_alt", launch_alt)
            object.__setattr__(self, "launch_lat", launch_lat)
            object.__setattr__(self, "launch_lon", launch_lon)
        except (AttributeError, TypeError, ValueError) as err:
            print(f"Error: {err}")
            print(
                "Check if the sonde detected a launch, otherwise launch coordinates cannot be set"
            )

    def add_launch_detect(self, launch_detect_bool: bool) -> None:
        """Sets bool attribute of whether launch was detected

        Parameters
        ----------
        launch_detect_bool : bool
            True if launch detected, else False
        """
        object.__setattr__(self, "launch_detect", launch_detect_bool)

    def add_afile(self, path_to_afile: str) -> None:
        """Sets attribute with path to A-file of the sonde

        Parameters
        ----------
        path_to_afile : str
            Path to the sonde's A-file
        """
        object.__setattr__(self, "afile", path_to_afile)
        return self

    def add_level_dir(self, l0_dir: str = None, l1_dir: str = None, l2_dir: str = None):
        if l0_dir is None:
            if not hasattr(self, "afile"):
                raise ValueError("No afile in sonde. Cannot continue")
            l0_dir = os.path.dirname(self.afile)
        if l1_dir is None:
            l1_dir = l0_dir.replace("Level_0", "Level_1")
        else:
            l1_dir = l1_dir.format(flight_id=self.flight_id)
        if l2_dir is None:
            l2_dir = l0_dir.replace("Level_0", "Level_2")
        else:
            l2_dir = l2_dir.format(flight_id=self.flight_id)

        object.__setattr__(self, "l0_dir", l0_dir)
        object.__setattr__(self, "l1_dir", l1_dir)
        object.__setattr__(self, "l2_dir", l2_dir)

    def run_aspen(self, path_to_postaspenfile: str = None) -> None:
        """Runs aspen and sets attribute with path to post-ASPEN file of the sonde

        If the A-file path is known for the sonde, i.e. if the attribute `path_to_afile` exists,
        then the function will attempt to look for a post-ASPEN file of the same date-time as in the A-file's name.
        Sometimes, the post-ASPEN file might not exist (e.g. because launch was not detected), and in
        such cases, ASPEN will run in a docker image and create the file.

        If the A-file path is not known for the sonde, the function will expect the argument
        `path_to_postaspenfile` to be not empty.

        Parameters
        ----------
        path_to_postaspenfile : str, optional
            The path to the post-ASPEN file. If not provided, the function will attempt to construct the path from the `afile` attribute.

        Attributes Set
        --------------
        postaspenfile : str
            The path to the post-ASPEN file. This attribute is set if the file exists at the constructed or provided path.
        """

        l0_dir = self.l0_dir  # os.path.dirname(self.afile)
        aname = os.path.basename(self.afile)
        dname = "D" + aname[1:]
        l1_dir = self.l1_dir
        l1_name = dname.split(".")[0] + "QC.nc"

        if path_to_postaspenfile is None:
            path_to_postaspenfile = os.path.join(l1_dir, l1_name)

        if not os.path.exists(path_to_postaspenfile):
            os.makedirs(l1_dir, exist_ok=True)
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--mount",
                    f"type=bind,source={l0_dir},target=/input",
                    "--mount",
                    f"type=bind,source={l1_dir},target=/output",
                    "ghcr.io/atmdrops/aspenqc:4.0.2",
                    "-i",
                    f"/input/{dname}",
                    "-n",
                    f"/output/{l1_name}",
                ],
                check=True,
            )

        object.__setattr__(self, "postaspenfile", path_to_postaspenfile)
        return self

    def add_aspen_ds(self) -> None:
        """Sets attribute with an xarray Dataset read from post-ASPEN file

        The function will first check if the serial ID of the instance and that obtained from the
        global attributes of the post-ASPEN file match. If they don't, function will print out an error.

        If the `postaspenfile` attribute doesn't exist, function will print out an error
        """

        if hasattr(self, "postaspenfile"):
            try:
                ds = xr.open_dataset(self.postaspenfile)
            except ValueError:
                warnings.warn(f"No valid l1 file for sonde {self.serial_id}")
                return None
            if "SondeId" not in ds.attrs:
                if ds.attrs["SoundingDescription"].split(" ")[1] == self.serial_id:
                    object.__setattr__(self, "aspen_ds", ds)
                else:
                    raise ValueError(
                        f"I didn't find the `SondeId` attribute, so checked the `SoundingDescription` attribute. I found the ID in the `SoundingDescription` global attribute ({ds.attrs['SoundingDescription'].split(' ')[1]}) to not match with this instance's `serial_id` attribute ({self.serial_id}). Therefore, I am not storing the xarray dataset as an attribute."
                    )
            elif ds.attrs["SondeId"] == self.serial_id:
                object.__setattr__(self, "aspen_ds", ds)
            else:
                raise ValueError(
                    f"I found the `SondeId` global attribute ({ds.attrs['SondeId']}) to not match with this instance's `serial_id` attribute ({self.serial_id}). Therefore, I am not storing the xarray dataset as an attribute."
                )
        else:
            raise ValueError(
                f"I didn't find the `postaspenfile` attribute for Sonde {self.serial_id}, therefore I can't store the xarray dataset as an attribute"
            )
        return self

    def filter_no_launch_detect(self) -> None:
        """
        Filter out sondes that did not detect a launch

        The function will check if the `launch_detect` attribute exists and if it is False.
        If the attribute doesn't exist, the function will raise an error.
        If the attribute exists and is False, the function will print a no-launch detected message.
        If the attribute exists and is True, the function will return the object.

        This function serves as a checkpoint for filtering out sondes
        that did not detect a launch before running functions
        that will require `aspen_ds`, e.g. the QC functions.

        Parameters
        ----------
        None

        Returns
        -------
        self : Sonde object
            The Sonde object itself, if the launch was detected, else None

        Raises
        ------
        ValueError
            If the `launch_detect` attribute does not exist.
        """
        if hasattr(self, "launch_detect"):
            if not self.launch_detect:
                print(
                    f"No launch detected for Sonde {self.serial_id}. I am not running QC checks for this Sonde."
                )
            else:
                return self
        else:
            raise ValueError(
                f"The attribute `launch_detect` does not exist for Sonde {self.serial_id}."
            )

    def detect_floater(
        self,
        gpsalt_threshold: float = 25,
        consecutive_time_steps: int = 3,
        skip: bool = False,
    ):
        """
        Detects if a sonde is a floater.

        Parameters
        ----------
        gpsalt_threshold : float, optional
            The gpsalt altitude below which the sonde will check for time periods when gpsalt and pres have not changed. Default is 25.
        skip : bool, optional
            If True, the function will return the object without performing any operations. Default is False.

        Returns
        -------
        self
            The object itself with the new `is_floater` attribute added based on the function parameters.
        """
        if hh.get_bool(skip):
            return self
        else:
            if isinstance(gpsalt_threshold, str):
                gpsalt_threshold = float(gpsalt_threshold)

            if hasattr(self, "aspen_ds"):
                surface_ds = (
                    self.aspen_ds.where(
                        self.aspen_ds.gpsalt < gpsalt_threshold, drop=True
                    )
                    .sortby("time")
                    .dropna(dim="time", how="any", subset=["pres", "gpsalt"])
                )
                gpsalt_diff = np.diff(surface_ds.gpsalt)
                pressure_diff = np.diff(surface_ds.pres)
                gpsalt_diff_below_threshold = (
                    np.abs(gpsalt_diff) < 1
                )  # GPS altitude value at surface shouldn't change by more than 1 m
                pressure_diff_below_threshold = (
                    np.abs(pressure_diff) < 1
                )  # Pressure value at surface shouldn't change by more than 1 hPa
                floater = gpsalt_diff_below_threshold & pressure_diff_below_threshold
                if np.any(floater):
                    object.__setattr__(self, "is_floater", True)
                    for time_index in range(len(floater) - consecutive_time_steps + 1):
                        if np.all(
                            floater[time_index : time_index + consecutive_time_steps]
                        ):
                            landing_time = surface_ds.time[time_index - 1].values
                            object.__setattr__(self, "landing_time", landing_time)
                            print(
                                f"{self.serial_id}: Floater detected! The landing time is estimated as {landing_time}."
                            )
                            break
                        if not hasattr(self, "landing_time"):
                            print(
                                f"{self.serial_id}: Floater detected! However, the landing time could not be estimated. Therefore setting landing time as {surface_ds.time[0].values}"
                            )
                            object.__setattr__(
                                self, "landing_time", surface_ds.time[0].values
                            )
                else:
                    object.__setattr__(self, "is_floater", False)
            else:
                raise ValueError(
                    "The attribute `aspen_ds` does not exist. Please run `add_aspen_ds` method first."
                )
            return self

    def crop_aspen_ds_to_landing_time(self):
        """
        Crops the aspen_ds to the time period before landing.

        Parameters
        ----------
        None

        Returns
        -------
        self
            The object itself with the new `cropped_aspen_ds` attribute added if the sonde is a floater.
        """
        if hasattr(self, "is_floater"):
            if self.is_floater:
                if hasattr(self, "landing_time"):
                    object.__setattr__(
                        self,
                        "cropped_aspen_ds",
                        self.aspen_ds.sel(time=slice(self.landing_time, None)),
                    )
        else:
            raise ValueError(
                "The attribute `is_floater` does not exist. Please run `detect_floater` method first."
            )
        return self

    def profile_fullness(
        self,
        variable_dict={"u_wind": 4, "v_wind": 4, "rh": 2, "tdry": 2, "pres": 2},
        time_dimension="time",
        timestamp_frequency=4,
        fullness_threshold=0.75,
        add_fullness_fraction_attribute=True,
        skip=False,
    ):
        """
        Calculates the profile coverage for a given set of variables, considering their sampling frequency.
        If the sonde is a floater, the function will take the `cropped_aspen_ds` attribute
        (calculated with the `crop_aspen_ds_to_landing_time` method) as the dataset to calculate the profile coverage.

        This function assumes that the time_dimension coordinates are spaced over 0.25 seconds,
        implying a timestamp_frequency of 4 hertz. This is applicable for ASPEN-processed QC and PQC files,
        specifically for RD41.

        For each variable in the variable_dict, the function calculates the fullness fraction. If the fullness
        fraction is less than the fullness_threshold, it sets an attribute in `self.qc` named
        "profile_fullness_{variable}" to False. Otherwise, it sets this attribute to True.

        If add_fullness_fraction_attribute is True, the function also sets an attribute in `self` named
        "profile_fullness_fraction_{variable}" to the calculated fullness fraction.

        Parameters
        ----------
        variable_dict : dict, optional
            Dictionary containing the variables in `self.aspen_ds` and their respective sampling frequencies.
            The function will estimate the weighted profile-coverage for these variables.
            Default is {'u_wind':4,'v_wind':4,'rh':2,'tdry':2,'pres':2}.
        time_dimension : str, optional
            The independent dimension of the profile. Default is "time".
        timestamp_frequency : numeric, optional
            The sampling frequency of `time_dimension` in hertz. Default is 4.
        fullness_threshold : float or str, optional
            The threshold for the fullness fraction. If the calculated fullness fraction is less than this threshold,
            the profile is considered not full. Default is 0.8.
        add_fullness_fraction_attribute : bool or str, optional
            If True, the function will add the fullness fraction as an attribute to the object. Default is True.
            If provided as string, it should be possible to convert it to a boolean value with the helper get_bool function.
        skip : bool, optional
            If True, the function will return the object without performing any operations. Default is False.

        Returns
        -------
        self
            The object itself, possibly with new attributes added based on the function parameters.
        """
        if hh.get_bool(skip):
            return self
        else:
            if isinstance(fullness_threshold, str):
                fullness_threshold = float(fullness_threshold)

            for variable, sampling_frequency in variable_dict.items():
                if self.is_floater:
                    if not hasattr(self, "cropped_aspen_ds"):
                        self.crop_aspen_ds_to_landing_time()
                    dataset = self.cropped_aspen_ds[variable]
                else:
                    dataset = self.aspen_ds[variable]

                weighed_time_size = len(dataset[time_dimension]) / (
                    timestamp_frequency / sampling_frequency
                )
                fullness_fraction = (
                    np.sum(~np.isnan(dataset.values)) / weighed_time_size
                )
                if fullness_fraction < fullness_threshold:
                    object.__setattr__(
                        self.qc,
                        f"profile_fullness_{variable}",
                        False,
                    )
                else:
                    object.__setattr__(
                        self.qc,
                        f"profile_fullness_{variable}",
                        True,
                    )
                if hh.get_bool(add_fullness_fraction_attribute):
                    object.__setattr__(
                        self,
                        f"profile_fullness_fraction_{variable}",
                        fullness_fraction,
                    )
            return self

    def near_surface_coverage(
        self,
        variables=["u_wind", "v_wind", "rh", "tdry", "pres"],
        alt_bounds=[0, 1000],
        alt_dimension_name="alt",
        count_threshold=50,
        add_near_surface_count_attribute=True,
        skip=False,
    ):
        """
        Calculates the fraction of non-null values in specified variables near the surface.

        Parameters
        ----------
        variables : list, optional
            The variables to consider for the calculation. Defaults to ["u_wind","v_wind","rh","tdry","pres"].
        alt_bounds : list, optional
            The lower and upper bounds of altitude in meters to consider for the calculation. Defaults to [0,1000].
        alt_dimension_name : str, optional
            The name of the altitude dimension. Defaults to "alt". If the sonde is a floater, this will be set to "gpsalt" regardless of user-provided value.
        count_threshold : int, optional
            The minimum count of non-null values required for a variable to be considered as having near surface coverage. Defaults to 50.
        add_near_surface_count_attribute : bool, optional
            If True, adds the count of non-null values as an attribute for every variable to the object. Defaults to True.
        skip : bool, optional
            If True, skips the calculation and returns the object as is. Defaults to False.

        Returns
        -------
        self
            The object with updated attributes.

        Raises
        ------
        ValueError
            If the attribute `aspen_ds` does not exist. The `add_aspen_ds` method should be run first.
        """
        if hh.get_bool(skip):
            return self
        else:
            if not hasattr(self, "aspen_ds"):
                raise ValueError(
                    "The attribute `aspen_ds` does not exist. Please run `add_aspen_ds` method first."
                )

            if not hasattr(self, "is_floater"):
                raise ValueError(
                    "The attribute `is_floater` does not exist. Please run `detect_floater` method first."
                )

            if self.is_floater:
                alt_dimension_name = "gpsalt"

            if isinstance(alt_bounds, str):
                alt_bounds = alt_bounds.split(",")
                alt_bounds = [float(alt_bound) for alt_bound in alt_bounds]
            if isinstance(count_threshold, str):
                count_threshold = int(count_threshold)
            if isinstance(variables, str):
                variables = variables.split(",")

            for variable in variables:
                dataset = self.aspen_ds[[variable, alt_dimension_name]]
                near_surface = dataset.where(
                    (dataset[alt_dimension_name] > alt_bounds[0])
                    & (dataset[alt_dimension_name] < alt_bounds[1]),
                    drop=True,
                )

                near_surface_count = np.sum(~np.isnan(near_surface[variable].values))
                if near_surface_count < count_threshold:
                    object.__setattr__(
                        self.qc,
                        f"near_surface_coverage_{variable}",
                        False,
                    )
                else:
                    object.__setattr__(
                        self.qc,
                        f"near_surface_coverage_{variable}",
                        True,
                    )
                if hh.get_bool(add_near_surface_count_attribute):
                    object.__setattr__(
                        self,
                        f"near_surface_count_{variable}",
                        near_surface_count,
                    )
            return self

    def filter_qc_fail(self, filter_flags=None):
        """
        Filters the sonde based on a list of QC flags. If any of the flags are False, the sonde will be filtered out from creating L2.
        If the sonde passes all the QC checks, the attributes listed in filter_flags will be removed from the sonde object.

        Parameters
        ----------
        filter_flags : str or list, optional
            Comma-separated string or list of QC-related attribute names to be checked. Each item can be a specific attribute name or a prefix to include all attributes starting with that prefix. You can also provide 'all_except_<prefix>' to filter all flags except those starting with '<prefix>'. If 'all_except_<prefix>' is provided, it should be the only value in filter_flags. If not provided, no sondes will be filtered.

        Returns
        -------
        self : object
            The sonde object itself, with the attributes listed in filter_flags removed if it passes all the QC checks.

        Raises
        ------
        ValueError
            If a flag in filter_flags does not exist as an attribute of the sonde object, or if 'all_except_<prefix>' is provided in filter_flags along with other values. Please ensure that the flag names provided in 'filter_flags' correspond to existing QC attributes. If you're using a prefix to filter attributes, make sure the prefix is correct. Check your skipped QC functions or your provided list of filter flags.
        """
        all_qc_attributes = [attr for attr in dir(self.qc) if not attr.startswith("__")]

        if filter_flags is None:
            filter_flags = []
        elif isinstance(filter_flags, str):
            filter_flags = filter_flags.split(",")
        elif isinstance(filter_flags, list):
            pass
        else:
            raise ValueError(
                "Invalid type for filter_flags. It must be one of the following:\n"
                "- None: If you want to filter against all QC attributes.\n"
                "- A string: If you want to provide a comma-separated list of individual flag values or prefixes of flag values.\n"
                "- A list: If you want to provide individual flag values or prefixes of flag values."
            )

        if (
            any(flag.startswith("all_except_") for flag in filter_flags)
            and len(filter_flags) > 1
        ):
            raise ValueError(
                "If 'all_except_<prefix>' is provided in filter_flags, it should be the only value."
            )

        new_filter_flags = []
        for flag in filter_flags:
            if flag.startswith("all_except_"):
                prefix = flag.replace("all_except_", "")
                new_filter_flags.extend(
                    [attr for attr in all_qc_attributes if not attr.startswith(prefix)]
                )
            else:
                new_filter_flags.extend(
                    [attr for attr in all_qc_attributes if attr.startswith(flag)]
                )

        filter_flags = new_filter_flags

        for flag in filter_flags:
            if not hasattr(self.qc, flag):
                raise ValueError(
                    f"The attribute '{flag}' does not exist in the QC attributes of the sonde object. "
                    "Please ensure that the flag names provided in 'filter_flags' correspond to existing QC attributes. "
                    "If you're using a prefix to filter attributes, make sure the prefix is correct. "
                    "Check your skipped QC functions or your provided list of filter flags."
                )
            if not bool(getattr(self.qc, flag)):
                print(
                    f"{flag} returned False. Therefore, filtering this sonde ({self.serial_id}) out from L2"
                )
                return None

        # If the sonde passes all the QC checks, remove all attributes listed in filter_flags
        for flag in filter_flags:
            delattr(self.qc, flag)

        return self

    def create_interim_l2_ds(self):
        """
        Creates an interim L2 dataset from the aspen_ds or cropped_aspen_ds attribute.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with the interim L2 dataset added as an attribute.
        """
        if self.is_floater:
            if not hasattr(self, "cropped_aspen_ds"):
                self.crop_aspen_ds_to_landing_time()
            ds = self.cropped_aspen_ds
        else:
            ds = self.aspen_ds

        object.__setattr__(self, "_interim_l2_ds", ds)

        return self

    def convert_to_si(self, variables=["rh", "pres", "tdry"], skip=False):
        """
        Converts variables to SI units.

        Parameters
        ----------
        variables : list or str, optional
            The variables to convert to SI units. If a string is provided, it should be a comma-separated list of variables.
            The default variables are 'rh', 'pres', and 'tdry'.

        skip : bool, optional
            If set to True, the function will skip the conversion process but will still ensure that the '_interim_l2_ds' attribute is set.
            If '_interim_l2_ds' is not already an attribute of the object, it will be set to 'aspen_ds'.
            Default is False.

        Returns
        -------
        self : object
            Returns the sonde object with the specified variables in aspen_ds converted to SI units.
            If 'skip' is set to True, it returns the sonde object with '_interim_l2_ds' set to 'aspen_ds' if it wasn't already present.
        """
        if hh.get_bool(skip):
            if hasattr(self, "_interim_l2_ds"):
                return self
            else:
                object.__setattr__(self, "_interim_l2_ds", self.aspen_ds)
                return self
        else:
            if isinstance(variables, str):
                variables = variables.split(",")

            if hasattr(self, "_interim_l2_ds"):
                ds = self._interim_l2_ds
            else:
                ds = self.aspen_ds

            for variable in variables:
                func = hh.get_si_converter_function_based_on_var(variable)
                ds = ds.assign({f"{variable}": func(self.aspen_ds[variable])})

            object.__setattr__(self, "_interim_l2_ds", ds)

            return self

    def get_l2_variables(self, l2_variables: dict = hh.l2_variables):
        """
        Gets the variables from aspen_ds to create L2.

        Parameters
        ----------
        l2_variables : dict or str, optional
            A dictionary where the keys are the variables in aspen_ds to keep for L2.
            If dictionary items contain a 'rename_to' key, the variable will be renamed to the value of 'rename_to'.
            If dictionary items contain a 'attributes' key, the variable will be assigned the attributes in the value of 'attributes'.
            The default is the l2_variables dictionary from the helper module.

        Returns
        -------
        self : object
            Returns the sonde object with only the specified variables (renamed if dictionary has 'rename_to' key and attributes added if dictionary has 'attributes' key) in _interim_l2_ds attribute.
            If '_interim_l2_ds' is not already an attribute of the object, it will first be set to 'aspen_ds' before reducing to the variables and renaming.
        """
        if isinstance(l2_variables, str):
            l2_variables = ast.literal_eval(l2_variables)

        l2_variables_list = list(l2_variables.keys())

        if hasattr(self, "_interim_l2_ds"):
            ds = self._interim_l2_ds
        else:
            ds = self.aspen_ds

        ds = ds[l2_variables_list]

        for variable, variable_dict in l2_variables.items():
            if "attributes" in variable_dict:
                ds[variable].attrs = variable_dict["attributes"]
            if "rename_to" in variable_dict:
                ds = ds.rename({variable: variable_dict["rename_to"]})

        object.__setattr__(self, "_interim_l2_ds", ds)

        return self

    def add_sonde_id_variable(self, variable_name="sonde_id"):
        """
        Adds a variable and related attributes to the sonde object with the Sonde object (self)'s serial_id attribute.

        Parameters
        ----------
        variable_name : str, optional
            The name of the variable to be added. Default is 'sonde_id'.

        Returns
        -------
        self : object
            Returns the sonde object with a variable containing serial_id. Name of the variable provided by 'variable_name'.
        """
        if hasattr(self, "_interim_l2_ds"):
            ds = self._interim_l2_ds
        else:
            ds = self.aspen_ds

        ds = ds.assign({variable_name: self.serial_id})
        ds[variable_name].attrs = {
            "descripion": "unique sonde ID",
            "long_name": "sonde identifier",
            "cf_role": "trajectory_id",
        }

        object.__setattr__(self, "_interim_l2_ds", ds)

        return self

    def get_flight_attributes(
        self, l2_flight_attributes_map: dict = hh.l2_flight_attributes_map
    ) -> None:
        """
        Gets flight attributes from the A-file and adds them to the sonde object.

        Parameters
        ----------
        l2_flight_attributes_map : dict or str, optional
            A dictionary where the keys are the flight attributes in the A-file
            and the values are the corresponding (renamed) attribute names to be used for the L2 file.
            The default is the l2_flight_attributes_map dictionary from the helper module.

        Returns
        -------
        self : object
            Returns the sonde object with the flight attributes added as attributes.
        """
        flight_attrs = {}

        with open(self.afile, "r") as f:
            lines = f.readlines()

        for attr in l2_flight_attributes_map.keys():
            for line_id, line in enumerate(lines):
                if attr in line:
                    break

            attr = l2_flight_attributes_map.get(attr, attr)

            value = lines[line_id].split("= ")[1]
            flight_attrs[attr] = float(value) if "AVAPS" not in attr else value

        object.__setattr__(self, "flight_attrs", flight_attrs)

        return self

    def get_other_global_attributes(self):
        nc_global_attrs = {
            # "title": "Level-2",
            # "doi": f"{pydropsonde.data_doi}",
            # "created with": f"pipeline.py doi:{pydropsonde.software_doi}",
            "Conventions": "CF-1.8",
            "platform_id": self.platform_id,
            # "instrument_id": "Vaisala RD-41",
            "product_id": "Level-2",
            # "AVAPS_Software_version": "Version 4.1.2",
            "ASPEN_version": (
                self.aspen_ds.AspenVersion
                if hasattr(self.aspen_ds, "AspenVersion")
                else self.aspen_ds.AvapsEditorVersion
            ),
            "ASPEN_processing_time": self.aspen_ds.ProcessingTime,
            # "JOANNE_version": joanne.__version__,
            # "launch_date": str(pd.to_datetime(self.launch_time).date()),
            "launch_time_(UTC)": (
                str(self.aspen_ds.launch_time.values)
                if hasattr(self.aspen_ds, "launch_time")
                else str(self.aspen_ds.base_time.values)
            ),
            "is_floater": self.is_floater.__str__(),
            "sonde_serial_ID": self.serial_id,
            "author": "Geet George",
            "author_email": "g.george@tudelft.nl",
            "featureType": "trajectory",
            # "reference": pydropsonde.reference_study,
            "creation_time": str(datetime.utcnow()) + " UTC",
        }

        for attr in dir(self):
            if attr.startswith("near_surface_count_"):
                nc_global_attrs[attr] = getattr(self, attr)
            if attr.startswith("profile_fullness_fraction_"):
                nc_global_attrs[attr] = getattr(self, attr)

        for attr in dir(self.qc):
            if not attr.startswith("__"):
                nc_global_attrs[f"qc_{attr}"] = int(getattr(self.qc, attr))

        object.__setattr__(self, "nc_global_attrs", nc_global_attrs)

        return self

    def add_global_attributes_to_interim_l2_ds(self):
        """
        Adds global attributes to _interim_l2_ds.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with global attributes added to _interim_l2_ds.
        """
        ds = self._interim_l2_ds

        attrs_to_del = []
        for attr in ds.attrs.keys():
            attrs_to_del.append(attr)

        for attr in attrs_to_del:
            del ds.attrs[attr]

        if hasattr(self, "flight_attrs"):
            for attr, value in self.flight_attrs.items():
                ds.attrs[attr] = value
        if hasattr(self, "nc_global_attrs"):
            for attr, value in self.nc_global_attrs.items():
                ds.attrs[attr] = value

        object.__setattr__(self, "_interim_l2_ds", ds)

        return self

    def add_compression_and_encoding_properties(
        self,
        encoding_variables: dict = hh.encoding_variables,
        default_variable_compression_properties: dict = hh.variable_compression_properties,
    ):
        """
        Adds compression and encoding properties to _interim_l2_ds.

        Parameters
        ----------
        comp : dict or str, optional
            A dictionary containing the compression properties to be used for the L2 file.
            The default is the comp dictionary from the helper module.

        Returns
        -------
        self : object
            Returns the sonde object with compression and encoding properties added to _interim_l2_ds.
        """

        for var in encoding_variables:
            self._interim_l2_ds[var].encoding = encoding_variables[var]

        for var in self._interim_l2_ds.data_vars:
            if not encoding_variables.get(var):
                self._interim_l2_ds[
                    var
                ].encoding = default_variable_compression_properties

        return self

    def get_l2_filename(
        self, l2_filename: str = None, l2_filename_template: str = None
    ):
        """
        Gets the L2 filename from the template provided.

        Parameters
        ----------
        l2_filename : str, optional
            The L2 filename. The default is the l2_filename_template from the helper module.

        Returns
        -------
        self : object
            Returns the sonde object with the L2 filename added as an attribute.
        """

        if l2_filename is None:
            if l2_filename_template:
                l2_filename = l2_filename_template.format(
                    platform=self.platform_id,
                    serial_id=self.serial_id,
                    flight_id=self.flight_id,
                    launch_time=self.launch_time.astype(datetime).strftime(
                        "%Y-%m-%d_%H-%M"
                    ),
                )
            else:
                l2_filename = hh.l2_filename_template.format(
                    platform=self.platform_id,
                    serial_id=self.serial_id,
                    flight_id=self.flight_id,
                    launch_time=self.launch_time.astype(datetime).strftime(
                        "%Y-%m-%d_%H-%M"
                    ),
                )

        object.__setattr__(self, "l2_filename", l2_filename)

        return self

    def write_l2(self, l2_dir: str = None):
        """
        Writes the L2 file to the specified directory.

        Parameters
        ----------
        l2_dir : str, optional
            The directory to write the L2 file to. The default is the directory of the A-file with '0' replaced by '2'.

        Returns
        -------
        self : object
            Returns the sonde object with the L2 file written to the specified directory using the l2_filename attribute to set the name.
        """

        if l2_dir is None:
            l2_dir = self.l2_dir

        if not os.path.exists(l2_dir):
            os.makedirs(l2_dir)

        self._interim_l2_ds.to_netcdf(os.path.join(l2_dir, self.l2_filename))

        return self

    def add_l2_ds(self, l2_dir: str = None):
        """
        Adds the L2 dataset as an attribute to the sonde object.

        Parameters
        ----------
        l2_dir : str, optional
            The directory to read the L2 file from. The default is the directory of the A-file with '0' replaced by '2'.

        Returns
        -------
        self : object
            Returns the sonde object with the L2 dataset added as an attribute.
        """
        if l2_dir is None:
            l2_dir = self.l2_dir

        try:
            object.__setattr__(
                self, "l2_ds", xr.open_dataset(os.path.join(l2_dir, self.l2_filename))
            )

            return self
        except FileNotFoundError:
            return None

    def create_prep_l3(self):
        _prep_l3_ds = self.l2_ds.assign_coords(
            {"sonde_id": ("sonde_id", [self.l2_ds.sonde_id.values])}
        ).sortby("time")
        object.__setattr__(self, "_prep_l3_ds", _prep_l3_ds)
        return self

    def check_interim_l3(
        self, interim_l3_path: str = None, interim_l3_filename: str = None
    ):
        if interim_l3_path is None:
            interim_l3_path = self.l2_dir.replace("Level_2", "Level_3_interim").replace(
                self.flight_id, ""
            )
        if interim_l3_filename is None:
            interim_l3_filename = "interim_l3_{sonde_id}_{version}.nc".format(
                sonde_id=self.serial_id, version=__version__
            )
        else:
            interim_l3_filename = interim_l3_filename.format(
                sonde_id=self.serial_id, version=__version__
            )
        if os.path.exists(os.path.join(interim_l3_path, interim_l3_filename)):
            ds = xr.open_dataset(os.path.join(interim_l3_path, interim_l3_filename))
            object.__setattr__(self, "_interim_l3_ds", ds)
            object.__setattr__(self, "cont", False)
            return self
        else:
            return self

    def add_q_and_theta_to_l2_ds(self):
        """
        Adds potential temperature and specific humidity to the L2 dataset.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with potential temperature and specific humidity added to the L2 dataset.
        """
        ds = self._prep_l3_ds

        ds = hh.calc_q_from_rh_sondes(ds)
        ds = hh.calc_theta_from_T(ds)

        object.__setattr__(self, "_prep_l3_ds", ds)

        return self

    def recalc_rh_and_ta(self):
        ds = self._prep_l3_ds
        ds = hh.calc_rh_from_q(ds)
        ds = hh.calc_T_from_theta(ds)
        object.__setattr__(self, "_prep_l3_ds", ds)
        return self

    def add_iwv(self):
        ds = self._prep_l3_ds
        ds = hh.calc_iwv(ds)
        object.__setattr__(self, "_prep_l3_ds", ds)

        return self

    def add_wind(self):

        ds = self._prep_l3_ds
        ds = hh.calc_wind_dir_and_speed(ds)
        object.__setattr__(self, "_prep_l3_ds", ds)
        return self

    def remove_non_mono_incr_alt(self, alt_var="alt"):
        """
        This function removes the indices in the some height variable that are not monotonically increasing
        """
        _prep_l3_ds = self._prep_l3_ds.load()
        alt = _prep_l3_ds[alt_var]
        curr_alt = alt.isel(time=0)
        for i in range(len(alt)):
            if alt[i] > curr_alt:
                alt[i] = np.nan
            elif ~np.isnan(alt[i]):
                curr_alt = alt[i]
        _prep_l3_ds[alt_var] = alt

        mask = ~np.isnan(alt)
        object.__setattr__(
            self,
            "_prep_l3_ds",
            _prep_l3_ds.sel(time=mask),
        )
        return self

    def interpolate_alt(
        self,
        alt_var="alt",
        interp_start=-5,
        interp_stop=14600,
        interp_step=10,
        max_gap_fill: int = 50,
        p_log=True,
        method: str = "bin",
    ):
        """
        Ineterpolate sonde data along comon altitude grid to prepare concatenation
        """
        interpolation_grid = np.arange(interp_start, interp_stop, interp_step)

        if not (self._prep_l3_ds[alt_var].diff(dim="time") < 0).any():
            warnings.warn(
                f"your altitude for sonde {self._prep_l3_ds.sonde_id.values} is not sorted."
            )
        ds = (self._prep_l3_ds.swap_dims({"time": alt_var})).load()
        if p_log:
            ds = ds.assign(p=(ds.p.dims, np.log(ds.p.values), ds.p.attrs))
        if method == "linear_interpolate":
            interp_ds = ds.interp({alt_var: interpolation_grid})
        elif method == "bin":
            interpolation_bins = interpolation_grid.astype("int")
            interpolation_label = np.arange(
                interp_start + interp_step / 2,
                interp_stop - interp_step / 2,
                interp_step,
            )
            try:
                interp_ds = ds.groupby_bins(
                    alt_var,
                    interpolation_bins,
                    labels=interpolation_label,
                ).mean(dim=alt_var)
            except ValueError:
                warnings.warn(f"No level 2 data for sonde {self.serial_id}")
                return None
            # somehow coordinates are lost and need to be added again
            for coord in ["lat", "lon", "time", "gpsalt"]:
                interp_ds = interp_ds.assign_coords(
                    {
                        coord: (
                            alt_var,
                            ds[coord]
                            .groupby_bins(
                                alt_var, interpolation_bins, labels=interpolation_label
                            )
                            .mean(alt_var)
                            .values,
                            ds[coord].attrs,
                        )
                    }
                )

            interp_ds = (
                interp_ds.transpose()
                .interpolate_na(
                    dim=f"{alt_var}_bins", max_gap=max_gap_fill, use_coordinate=True
                )
                .rename({f"{alt_var}_bins": alt_var, "time": "interp_time"})
                .reset_coords(["interp_time", "lat", "lon", "gpsalt"])
            )

        if p_log:
            interp_ds = interp_ds.assign(
                p=(interp_ds.p.dims, np.exp(interp_ds.p.values), interp_ds.p.attrs)
            )

        object.__setattr__(self, "_prep_l3_ds", interp_ds)
        return self

    def add_attributes_as_var(self):
        """
        Prepares l2 datasets to be concatenated to gridded.
        adds all attributes as variables to avoid conflicts when concatenating because attributes are different
        (and not lose information)
        """
        _prep_l3_ds = self._prep_l3_ds
        for attr, value in self._prep_l3_ds.attrs.items():
            _prep_l3_ds[attr] = value

        _prep_l3_ds.attrs.clear()
        object.__setattr__(self, "_prep_l3_ds", _prep_l3_ds)
        return self

    def make_prep_interim(self):
        object.__setattr__(self, "_interim_l3_ds", self._prep_l3_ds)
        return self

    def save_interim_l3(self, interim_l3_path: str = None, interim_l3_name: str = None):
        if interim_l3_path is None:
            interim_l3_path = self.l2_dir.replace("Level_2", "Level_3_interim").replace(
                self.flight_id, ""
            )
        if interim_l3_name is None:
            interim_l3_name = "interim_l3_{sonde_id}_{version}.nc".format(
                sonde_id=self.serial_id, version=__version__
            )
        os.makedirs(interim_l3_path, exist_ok=True)
        self._interim_l3_ds.to_netcdf(os.path.join(interim_l3_path, interim_l3_name))
        return self


@dataclass(order=True)
class Gridded:
    sondes: dict

    def concat_sondes(self):
        """
        function to concatenate all sondes using the combination of all measurement times and launch times
        """
        list_of_l2_ds = [sonde._interim_l3_ds for sonde in self.sondes.values()]

        combined = xr.concat(list_of_l2_ds, dim="sonde_id", join="exact")
        combined = combined.assign(
            dict(iwv=("sonde_id", combined.iwv.mean("alt").values, combined.iwv.attrs))
        )
        self._interim_l3_ds = combined
        return self

    def get_l3_dir(self, l3_dir: str = None):
        if l3_dir:
            self.l3_dir = l3_dir
        elif not self.sondes is None:
            self.l3_dir = (
                list(self.sondes.values())[0]
                .l2_dir.replace("Level_2", "Level_3")
                .replace(list(self.sondes.values())[0].flight_id, "")
                .replace(list(self.sondes.values())[0].platform_id, "")
            )
        else:
            raise ValueError("No sondes and no l3 directory given, cannot continue ")
        return self

    def get_l3_filename(self, l3_filename: str = None):
        if l3_filename is None:
            l3_filename = hh.l3_filename
        else:
            l3_filename = l3_filename

        self.l3_filename = l3_filename

        return self

    def write_l3(self, l3_dir: str = None):
        if l3_dir is None:
            l3_dir = self.l3_dir

        if not os.path.exists(l3_dir):
            os.makedirs(l3_dir)

        self._interim_l3_ds.to_netcdf(os.path.join(l3_dir, self.l3_filename))

        return self


@dataclass(order=True)
class Circle:
    """Class identifying a circle and containing its metadata.

    A `Circle` identifies the circle averaged products from all sondes within the circle.

    Every `Circle` mandatorily has a `circle` identifier in the format "HALO-240811-c1".
    """

    _interim_l3_ds: xr.Dataset
    flight_id: str
    platform_id: str
    l4_dir: str = field(default=None)
    l4_filename: str = field(default=None)
    level3_ds: xr.Dataset = field(default=None)
    yaml_directory: str = field(default=None)
    sonde_ids: list = field(default_factory=list)
    circle_times: list = field(default_factory=list)
    flight_date: list = field(default_factory=list)
    platform_name: list = field(default_factory=list)
    segment_id: list = field(default_factory=list)
    circles: list = field(default_factory=list)
    all_sondes: xr.Dataset = field(default=None)
    mean_parameter: np.ndarray = field(default_factory=lambda: np.array([]))
    m_parameter: np.ndarray = field(default_factory=lambda: np.array([]))
    c_parameter: np.ndarray = field(default_factory=lambda: np.array([]))
    Ns: np.ndarray = field(default_factory=lambda: np.array([]))
    D: xr.DataArray = field(default=None)
    vor: xr.DataArray = field(default=None)
    intercept: np.ndarray = field(default_factory=lambda: np.array([]))
    dudx: np.ndarray = field(default_factory=lambda: np.array([]))
    dudy: np.ndarray = field(default_factory=lambda: np.array([]))
    u_: np.ndarray = field(default_factory=lambda: np.array([]))
    u: xr.DataArray = field(default=None)
    v: xr.DataArray = field(default=None)
    q: xr.DataArray = field(default=None)
    dx: xr.DataArray = field(default=None)
    dy: xr.DataArray = field(default=None)
    ta: xr.DataArray = field(default=None)
    p: xr.DataArray = field(default=None)
    alt: xr.DataArray = field(default=None)

    def get_l4_dir(self, l4_dir: str = None):
        if l4_dir:
            self.l4_dir = l4_dir
        elif self._interim_l3_ds is not None:
            self.l4_dir = list(self._interim_l3_ds.values())[0].l4_dir
        else:
            raise ValueError("No sondes and no l4 directory given, cannot continue")
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

    def get_level3_dataset(self, lv3_directory, lv3_filename):
        self.level3_ds = xr.open_dataset(lv3_directory + "/" + lv3_filename)
        return self

    def get_circle_times_from_yaml(self, yaml_directory):

        self.yaml_directory = yaml_directory

        allyamlfiles = sorted(glob.glob(self.yaml_directory + "*.yaml"))

        circle_times = []
        sonde_ids = []
        flight_date = []
        platform_name = []
        segment_id = []

        for i in allyamlfiles:
            with open(i) as source:
                flightinfo = yaml.load(source, Loader=yaml.SafeLoader)

            circle_times.append(
                [
                    (c["start"], c["end"])
                    for c in flightinfo["segments"]
                    if "circle" in c["kinds"]
                    if len(c["dropsondes"]["GOOD"]) >= 6
                ]
            )

            sonde_ids.append(
                [
                    c["dropsondes"]["GOOD"]
                    for c in flightinfo["segments"]
                    if "circle" in c["kinds"]
                    if len(c["dropsondes"]["GOOD"]) >= 6
                ]
            )

            segment_id.append(
                [
                    (c["segment_id"])
                    for c in flightinfo["segments"]
                    if "circle" in c["kinds"]
                    if len(c["dropsondes"]["GOOD"]) >= 6
                ]
            )

            if "HALO" in i:
                platform_name.append("HALO")
            elif "P3" in i:
                platform_name.append("P3")
            else:
                platform_name.append("")

            flight_date.append(np.datetime64(flightinfo["date"]))

        self.sonde_ids = sonde_ids
        self.circle_times = circle_times
        self.flight_date = flight_date
        self.platform_name = platform_name
        self.segment_id = segment_id

        return self

    def dim_ready_ds(self):

        dims_to_drop = ["sounding"]

        all_sondes = self.level3_ds.swap_dims({"sounding": "sonde_id"}).drop(
            dims_to_drop
        )

        self.all_sondes = all_sondes
        return self

    def get_circles(self):

        self.get_level3_dataset(self.l3_dir, self.l3_filename)

        self.get_circle_times_from_yaml(self.yaml_directory)

        circles = []

        for i in range(len(self.flight_date)):
            for j in range(len(self.circle_times[i])):
                if len(self.sonde_ids[i]) != 0:
                    circle_ds = self.level3_ds.sel(sonde_id=self.sonde_ids[i][j])
                    circle_ds["segment_id"] = self.segment_id[i][j]
                    circle_ds = circle_ds.pad(
                        sonde_id=(0, 13 - int(len(circle_ds.sonde_id))), mode="constant"
                    )
                    circle_ds["sounding"] = (
                        ["sonde_id"],
                        np.arange(0, 13, 1, dtype="int"),
                    )
                    circle_ds = circle_ds.swap_dims({"sonde_id": "sounding"})
                    circles.append(circle_ds)

        self.circles = circles
        return self

    def reswap_launchtime_sounding(self):

        for circle in self.circles:
            circle["sounding"] = (
                ["launch_time"],
                np.arange(1, len(circle.launch_time) + 1, 1),
            )
            circle = circle.swap_dims({"launch_time": "sounding"})

        return self

    def get_xy_coords_for_circles(self):

        for i in range(len(self.circles)):

            x_coor = (
                self.circles[i]["lon"]
                * 111.320
                * np.cos(np.radians(self.circles[i]["lat"]))
                * 1000
            )
            y_coor = self.circles[i]["lat"] * 110.54 * 1000
            # converting from lat, lon to coordinates in metre from (0,0).

            c_xc = np.full(np.size(x_coor, 1), np.nan)
            c_yc = np.full(np.size(x_coor, 1), np.nan)
            c_r = np.full(np.size(x_coor, 1), np.nan)

            for j in range(np.size(x_coor, 1)):
                a = ~np.isnan(x_coor.values[:, j])
                if a.sum() > 4:
                    c_xc[j], c_yc[j], c_r[j], _ = cf.least_squares_circle(
                        [
                            (xcoord, ycoord)
                            for xcoord, ycoord in zip(
                                x_coor.values[:, j], y_coor.values[:, j]
                            )
                            if ~np.isnan(xcoord)
                        ]
                    )

            circle_y = np.nanmean(c_yc) / (110.54 * 1000)
            circle_x = np.nanmean(c_xc) / (
                111.320 * np.cos(np.radians(circle_y)) * 1000
            )

            circle_diameter = np.nanmean(c_r) * 2

            xc = np.mean(x_coor, axis=0)
            yc = np.mean(y_coor, axis=0)

            delta_x = x_coor - xc  # difference of sonde long from mean long
            delta_y = y_coor - yc  # difference of sonde lat from mean lat

            self.circles[i]["platform_id"] = self.circles[i].platform_id.values[0]
            self.circles[i]["flight_altitude"] = (
                self.circles[i].flight_altitude.mean().values
            )
            self.circles[i]["circle_time"] = (
                self.circles[i].launch_time.mean().values.astype("datetime64")
            )
            self.circles[i]["circle_lon"] = circle_x
            self.circles[i]["circle_lat"] = circle_y
            self.circles[i]["circle_diameter"] = circle_diameter
            self.circles[i]["dx"] = (["sounding", "alt"], delta_x)
            self.circles[i]["dy"] = (["sounding", "alt"], delta_y)

        print("Circles ready for regression")
        return self

    def fit2d(self, x, y, u):
        """
        Estimate a 2D linear model to calculate u-values from x-y coordinates.
        :param x: x coordinates of data points. shape: (...,M)
        :param y: y coordinates of data points. shape: (...,M)
        :param u: data values. shape: (...,M)
        :returns: intercept, dudx, dudy. all shapes: (...)
        """
        u_ = u
        u = np.array(u, copy=True)
        a = np.stack([np.ones_like(x), x, y], axis=-1)

        invalid = np.isnan(u) | np.isnan(x) | np.isnan(y)
        under_constraint = np.sum(~invalid, axis=-1) < 6
        u[invalid] = 0
        a[invalid] = 0

        a_inv = np.linalg.pinv(a)

        intercept, dudx, dudy = np.einsum("...rm,...m->r...", a_inv, u)

        intercept[under_constraint] = np.nan
        dudx[under_constraint] = np.nan
        dudy[under_constraint] = np.nan

        self.intercept, self.dudx, self.dudy, self.u_ = intercept, dudx, dudy, u_

        return self

    def fit2d_xr(self, x, y, u, input_core_dims, output_core_dims):
        xr.apply_ufunc(
            self.fit2d,
            x,
            y,
            u,
            input_core_dims=[input_core_dims, input_core_dims, input_core_dims],
            output_core_dims=[(), (), (), output_core_dims],
        )
        return self

    def run_regression(self, parameter):
        """
        Input :
            parameter : string
                        the parameter on which regression is to be carried out

        Output :
            mean_parameter : mean of parameter (intercept)
            m_parameter, c_parameter    : coefficients of regression
        """
        id_u = ~np.isnan(self.u.values)
        id_v = ~np.isnan(self.v.values)
        id_q = ~np.isnan(self.q.values)
        id_x = ~np.isnan(self.dx.values)
        # id_y = ~np.isnan(self.dy.values)
        id_t = ~np.isnan(self.ta.values)
        id_p = ~np.isnan(self.p.values)

        id_quxv = np.logical_and(np.logical_and(id_q, id_u), np.logical_and(id_x, id_v))
        id_ = np.logical_and(np.logical_and(id_t, id_p), id_quxv)

        mean_parameter = np.full(len(self.alt), np.nan)
        m_parameter = np.full(len(self.alt), np.nan)
        c_parameter = np.full(len(self.alt), np.nan)

        Ns = np.full(len(self.alt), np.nan)  # number of sondes available for regression

        for k in range(len(self.alt)):
            Ns[k] = id_[:, k].sum()
            if Ns[k] > 6:
                X_dx = self["dx"].isel(alt=k).isel(sounding=id_[:, k]).values
                X_dy = self["dy"].isel(alt=k).isel(sounding=id_[:, k]).values

                X = list(zip(X_dx, X_dy))

                Y_parameter = (
                    self[parameter].isel(alt=k).isel(sounding=id_[:, k]).values
                )

                regr = linear_model.LinearRegression()
                regr.fit(X, Y_parameter)

                mean_parameter[k] = regr.intercept_
                m_parameter[k], c_parameter[k] = regr.coef_
            else:
                Ns[k] = 0

        self.mean_parameter = mean_parameter
        self.m_parameter = m_parameter
        self.c_parameter = c_parameter
        self.Ns = Ns

        return self

    def regress_for_all_parameters(self, list_of_parameters):

        save_directory = "/Users/geet/Documents/JOANNE/Data/Level_4/Interim_files/"

        file_name = (
            "PERCUSION_HALO_Dropsonde-RD41_"
            + str(self.circle_time.values)
            + "Level_4.nc"
        )

        if os.path.exists(save_directory + file_name):
            self = xr.open_dataset(save_directory + file_name)
        else:
            for par in list_of_parameters:
                self.run_regression(par)

                self[par] = (["alt"], self.mean_parameter)
                self["d" + par + "dx"] = (["alt"], self.m_parameter)
                self["d" + par + "dy"] = (["alt"], self.c_parameter)

                if "sondes_regressed" not in list(self.data_vars):
                    self["sondes_regressed"] = (["alt"], self.Ns)

            self.to_netcdf(save_directory + file_name)

        print("Finished all parameters ...")

        return self

    @classmethod
    def regress_for_all_circles(cls, circles, list_of_parameters):
        """
        Class method to regress over all Circle instances in the circles list.
        """
        for id_, circle in enumerate(circles):
            circle.regress_for_all_parameters(list_of_parameters)
            print(f"{id_+1}/{len(circles)} circles completed ...")

        print("Regressed over all circles ...")

        return cls

    def get_div_and_vor(self):

        D = self.dudx + self.dvdy
        vor = self.dvdx - self.dudy

        self.D = (["circle", "alt"], D)
        self.vor = (["circle", "alt"], vor)

        print("Finished estimating divergence and vorticity for all circles....")

        return self

    def get_density_vertical_velocity_and_omega(self):

        den_m = [None] * len(self.sounding)

        for n in range(len(self.sounding)):
            if len(self.isel(sounding=n).sonde_id.values) > 1:
                mr = mpcalc.mixing_ratio_from_specific_humidity(
                    self.isel(sounding=n).q_sounding.values
                )
                den_m[n] = mpcalc.density(
                    self.isel(sounding=n).p_sounding.values * units.Pa,
                    self.isel(sounding=n).ta_sounding.values * units.kelvin,
                    mr,
                ).magnitude
            else:
                den_m[n] = np.nan

        self["density"] = (["sounding", "circle", "alt"], den_m)
        self["mean_density"] = (["circle", "alt"], np.nanmean(den_m, axis=0))

        D = self.D.values
        # mean_den = self.mean_density

        nan_ids = np.where(np.isnan(D))

        w_vel = np.full([len(self["circle"]), len(self.alt)], np.nan)
        p_vel = np.full([len(self["circle"]), len(self.alt)], np.nan)

        w_vel[:, 0] = 0

        for cir in range(len(self["circle"])):
            last = 0
            for m in range(1, len(self.alt)):

                if (
                    len(
                        np.intersect1d(
                            np.where(nan_ids[1] == m)[0], np.where(nan_ids[0] == cir)[0]
                        )
                    )
                    > 0
                ):

                    ids_for_nan_ids = np.intersect1d(
                        np.where(nan_ids[1] == m)[0], np.where(nan_ids[0] == cir)[0]
                    )
                    w_vel[
                        nan_ids[0][ids_for_nan_ids], nan_ids[1][ids_for_nan_ids]
                    ] = np.nan

                else:
                    w_vel[cir, m] = w_vel[cir, last] - self.D.isel(circle=cir).isel(
                        alt=m
                    ).values * 10 * (m - last)
                    last = m

            for n in range(1, len(self.alt)):

                p_vel[cir, n] = (
                    -self.mean_density.isel(circle=cir).isel(alt=n)
                    * 9.81
                    * w_vel[cir, n]
                )

        self.W = (["circle", "alt"], w_vel)
        self.omega = (["circle", "alt"], p_vel)

        print("Finished estimating density, W and omega ...")

    def add_std_err_terms(self):

        dx_mean = self.dx.mean(dim="sounding")
        dy_mean = self.dy.mean(dim="sounding")

        dx_denominator = np.sqrt(((self.dx - dx_mean) ** 2).sum(dim="sounding"))
        dy_denominator = np.sqrt(((self.dy - dy_mean) ** 2).sum(dim="sounding"))

        for par in tqdm(["u", "v", "p", "q", "ta"]):

            par_err = self[par + "_sounding"] - (
                self[par]
                + (self["d" + par + "dx"] * self.dx)
                + (self["d" + par + "dy"] * self.dy)
            )

            par_sq_sum = np.nansum((par_err**2), axis=2)
            par_n = (~np.isnan(par_err)).sum(axis=2)

            par_numerator = np.sqrt(par_sq_sum / (par_n - 3))

            se_dpardx = par_numerator / dx_denominator
            se_dpardy = par_numerator / dy_denominator

            var_name_dx = "se_d" + par + "dx"
            var_name_dy = "se_d" + par + "dy"

            self[var_name_dx] = (["circle", "alt"], se_dpardx)
            self[var_name_dy] = (["circle", "alt"], se_dpardy)

        se_div = np.sqrt((self.se_dudx) ** 2 + (self.se_dvdy) ** 2)
        se_vor = np.sqrt((self.se_dudy) ** 2 + (self.se_dvdx) ** 2)

        self.se_D = se_div
        self.se_vor = se_vor

        se_W = np.nancumsum(
            np.sqrt((np.sqrt(self.se_D**2 / self.D**2) * self.D) ** 2),
            axis=1,
        )
        self.se_W = (["circle", "alt"], se_W)

        return self

    def get_advection(self, list_of_parameters=["u", "v", "q", "ta", "p"]):

        for var in list_of_parameters:
            adv_dicts = {}
            adv_dicts[f"h_adv_{var}"] = -(self.u * eval(f"self.d{var}dx")) - (
                self.v * eval(f"self.d{var}dy")
            )
            self[f"h_adv_{var}"] = (["alt"], adv_dicts[f"h_adv_{var}"])

        print("Finished estimating advection terms ...")

        return self

    def get_circle_products(self):

        self.get_div_and_vor()

        self.get_density_vertical_velocity_and_omega()

        self.circle_with_std_err = self.add_std_err_terms()

        print("All circle products retrieved!")

        return self
