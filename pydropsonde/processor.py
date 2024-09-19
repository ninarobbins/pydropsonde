import ast
from dataclasses import dataclass, field, KW_ONLY

from datetime import datetime, timezone
from typing import Any, Optional, List
import os
import subprocess
import warnings
import yaml
import numpy as np
import xarray as xr
import glob

import pydropsonde.helper as hh
import pydropsonde.helper.xarray_helper as hx
from importlib.metadata import version

__version__ = version("pydropsonde")

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
        if flight_template is not None:
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
        except (ValueError, TypeError):
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
                ds = xr.open_dataset(self.postaspenfile, engine="netcdf4")
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

    def add_aspen_history(self):
        history = getattr(self, "history", "")
        if hasattr(self.aspen_ds, "AspenVersion"):
            aspen_version = self.aspen_ds.AspenVersion
        else:
            aspen_version = self.aspen_ds.AvapsEditorVersion
        assert self.aspen_ds.ProcessingTime[-3:] == "UTC"

        aspen_time = datetime.strptime(
            self.aspen_ds.ProcessingTime, "%d %b %Y %H:%M %Z"
        ).replace(tzinfo=timezone.utc)

        history = (
            history
            + aspen_time.isoformat()
            + f" ASPEN processing with {aspen_version} \n"
        )
        object.__setattr__(self, "history", history)
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
        alt_dimension_name="gpsalt",
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

    def alt_near_gpsalt(self):
        dataset = self.aspen_ds[["alt", "gpsalt"]]
        mean_diff = np.abs((dataset.alt - dataset.gpsalt).mean(skipna=True))
        if mean_diff > 150:
            object.__setattr__(
                self.qc,
                "alt_near_gpsalt",
                False,
            )
        else:
            object.__setattr__(
                self.qc,
                "alt_near_gpsalt",
                True,
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

    def add_global_attrs(self, attributes=None):
        """
        function to add global attributes to sonde object. Whenever a dataset is written, those are added as global attributes
        Input:
        attributes : dictionary containing global attributes or None
        Returns:
        -------
        self: object
            Returns the sonde with a new attribute "global_attrs"
        """
        if attributes is None:
            attributes = {}
        object.__setattr__(self, "global_attrs", attributes)

        return self

    def get_sonde_attributes(self):
        """
        function to get sonde specific attributes from ASPEN software and quality control
        Parameters:
        None
        Returns:
        -------
        self: object
            Returns the sonde with a new attribute "sonde_attrs"
        """
        sonde_attrs = {
            "platform_id": self.platform_id,
            "launch_time_(UTC)": (
                str(self.aspen_ds.launch_time.values)
                if hasattr(self.aspen_ds, "launch_time")
                else np.datetime64(self.aspen_ds.base_time.values)
            ),
            "is_floater": self.is_floater.__str__(),
            "sonde_serial_ID": self.serial_id,
        }

        for attr in dir(self):
            if attr.startswith("near_surface_count_"):
                sonde_attrs[attr] = getattr(self, attr)
            if attr.startswith("profile_fullness_fraction_"):
                sonde_attrs[attr] = getattr(self, attr)

        for attr in dir(self.qc):
            if not attr.startswith("__"):
                sonde_attrs[f"qc_{attr}"] = int(getattr(self.qc, attr))

        object.__setattr__(self, "sonde_attrs", sonde_attrs)

        return self

    def add_l2_attributes_to_interim_l2_ds(self):
        """
        Adds flight, sonde and global attributes to _interim_l2_ds.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with flight, sonde and global attributes added to _interim_l2_ds.
        """
        ds = self._interim_l2_ds

        attrs_to_del = []
        for attr in ds.attrs.keys():
            attrs_to_del.append(attr)

        for attr in attrs_to_del:
            del ds.attrs[attr]

        if hasattr(self, "flight_attrs"):
            ds = ds.assign_attrs(self.flight_attrs)
        if hasattr(self, "sonde_attrs"):
            ds = ds.assign_attrs(self.sonde_attrs)
        if hasattr(self, "global_attrs"):
            ds = ds.assign_attrs(self.global_attrs)

        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" quality control with pydropsonde {__version__} \n"
        )
        object.__setattr__(self, "history", history)
        ds = ds.assign_attrs({"history": history})

        object.__setattr__(self, "_interim_l2_ds", ds)

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

        hx.write_ds(
            ds=self._interim_l2_ds,
            dir=l2_dir,
            filename=self.l2_filename,
            object_dim="sonde_id",
            alt_dim="time",
        )
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
                self, "l2_ds", hx.open_dataset(os.path.join(l2_dir, self.l2_filename))
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
        self, interim_l3_dir: str = None, interim_l3_filename: str = None
    ):
        if interim_l3_dir is None:
            interim_l3_dir = self.l2_dir.replace("Level_2", "Level_3_interim").replace(
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
        object.__setattr__(self, "interim_l3_dir", interim_l3_dir)
        object.__setattr__(self, "interim_l3_filename", interim_l3_filename)
        if os.path.exists(os.path.join(interim_l3_dir, interim_l3_filename)):
            object.__setattr__(
                self,
                "_interim_l3_ds",
                hx.open_dataset(os.path.join(interim_l3_dir, interim_l3_filename)),
            )
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

        ds = hh.calc_q_from_rh(ds)
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

    def add_thetas(self):
        ds = self._prep_l3_ds
        ds = hh.calc_theta_e(ds)
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
        Interpolate sonde data along comon altitude grid to prepare concatenation
        """
        interpolation_grid = np.arange(interp_start, interp_stop, interp_step)

        if not (self._prep_l3_ds[alt_var].diff(dim="time") < 0).any():
            warnings.warn(
                f"your altitude for sonde {self.serial_id.values} is not sorted."
            )
        ds = (self._prep_l3_ds.swap_dims({"time": alt_var})).load()
        if method == "linear_interpolate":
            if p_log:
                ds = ds.assign(p=(ds.p.dims, np.log(ds.p.values), ds.p.attrs))
            interp_ds = ds.interp({alt_var: interpolation_grid})
        elif method == "bin":
            interpolation_bins = interpolation_grid.astype("int")
            interpolation_label = np.arange(
                interp_start + interp_step / 2,
                interp_stop - interp_step / 2,
                interp_step,
            )
            interp_ds = ds.reset_coords().drop_vars("time")
            binned_ds = interp_ds.groupby_bins(
                alt_var,
                interpolation_bins,
                labels=interpolation_label,
            )
            interp_ds = binned_ds.mean(dim=alt_var, skipna=True)

            if p_log:
                interp_ds = interp_ds.assign(
                    p=(interp_ds.p.dims, np.log(interp_ds.p.values), interp_ds.p.attrs)
                )
            interp_ds = (
                interp_ds.transpose()
                .interpolate_na(
                    dim=f"{alt_var}_bins", max_gap=max_gap_fill, use_coordinate=True
                )
                .rename({f"{alt_var}_bins": alt_var})
            )

            time_type = ds["time"].values.dtype

            binned_time = (
                ds["time"]
                .astype(float)
                .groupby_bins(alt_var, interpolation_bins, labels=interpolation_label)
                .map(lambda x: x.mean(skipna=True))
            )
            interp_ds = interp_ds.assign(
                {
                    "interp_time": (
                        alt_var,
                        binned_time.where(
                            binned_time > -1
                        )  # replace missing values with nan
                        .interpolate_na(
                            dim=f"{alt_var}_bins",
                            max_gap=max_gap_fill,
                            use_coordinate=True,
                        )
                        .astype(time_type)
                        .values,
                        ds["time"].attrs,
                    )
                }
            )
        if p_log:
            interp_ds = interp_ds.assign(
                p=(interp_ds.p.dims, np.exp(interp_ds.p.values), interp_ds.p.attrs)
            )

        object.__setattr__(self, "_binned_ds", binned_ds)
        object.__setattr__(self, "_prep_l3_ds", interp_ds)

        return self

    def get_N_m_values(self, alt_var="alt"):
        binned_ds = self._binned_ds
        prep_l3 = self._prep_l3_ds

        N = binned_ds.count().transpose().rename({f"{alt_var}_bins": alt_var})

        for variable in [var for var in N.variables if var not in N.coords]:
            N_name = f"N{variable}"
            Nvar = N[variable].where(N[variable] > 0, 0)
            N_attrs = dict(
                long_name=f"Number of values per bin for {variable}",
            )
            prep_l3 = prep_l3.assign(
                {
                    N_name: (
                        prep_l3[variable].dims,
                        Nvar.values.astype(int),
                        N_attrs,
                    )
                }
            )
            prep_l3 = hx.add_ancillary_var(prep_l3, variable, N_name)
            # get m
            N2m = Nvar
            n_mask = N2m.where(~np.isnan(N2m), 0)
            int_mask = prep_l3[variable].where(~np.isnan(prep_l3[variable]), 0)

            m_mask = np.invert(n_mask.astype(bool)) & int_mask.astype(bool)
            m = xr.where(N2m > 0, x=2, y=0)
            m = xr.where(m_mask, x=1, y=m)

            m_name = f"m{variable}"
            m_attrs = {
                "long_name": f"interp method for {variable}",
                "0": "no data",
                "1": "no raw data, interpolated",
                "2": "average over raw data",
            }
            prep_l3 = prep_l3.assign(
                {
                    m_name: (
                        N2m.dims,
                        m.values.astype(int),
                        m_attrs,
                    )
                }
            )
            prep_l3 = hx.add_ancillary_var(prep_l3, variable, m_name)
        object.__setattr__(self, "_prep_l3_ds", prep_l3)

        return self

    def add_attributes_as_var(self, essential_attrs=None):
        """
        Prepares l2 datasets to be concatenated to gridded.
        adds all attributes as variables to avoid conflicts when concatenating because attributes are different
        (and not lose information)
        """
        ds = self._prep_l3_ds
        if essential_attrs is None:
            try:
                essential_attrs = hh.l3_coords
            except AttributeError:
                essential_attrs = {
                    "launch_time": {
                        "time_zone": "UTC",
                        "long_name": "dropsonde launch time",
                    }
                }

        for attr, value in ds.attrs.items():
            splt = attr.split("(")
            var_name = splt[0][:-1]
            if var_name in list(essential_attrs.keys()):
                var_attrs = essential_attrs[var_name]
                ds = ds.assign({var_name: ("sonde_id", [ds.attrs[attr]], var_attrs)})
        ds.attrs.clear()
        ds = ds.assign(
            dict(
                launch_time=(
                    "sonde_id",
                    ds.launch_time.astype(np.datetime64).values,
                    essential_attrs["launch_time"],
                )
            )
        )
        object.__setattr__(self, "attrs", ds.attrs.keys())
        object.__setattr__(self, "_prep_l3_ds", ds)
        return self

    def make_attr_coordinates(self):
        ds = self._prep_l3_ds
        new_coords = {
            coord: ("sonde_id", np.reshape(ds[coord].values, (1,)), ds[coord].attrs)
            for coord in hh.l3_coords
        }
        ds = ds.assign_coords(new_coords)
        object.__setattr__(self, "_prep_l3_ds", ds)
        return self

    def make_prep_interim(self):
        ds = self._prep_l3_ds
        if hasattr(self, "global_attrs"):
            ds = ds.assign_attrs(self.global_attrs)
        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" Level 3 processing with pydropsonde {__version__} \n"
        )
        object.__setattr__(self, "history", history)
        ds = ds.assign_attrs({"history": history})
        object.__setattr__(self, "_interim_l3_ds", ds)
        return self

    def save_interim_l3(self):
        hx.write_ds(
            ds=self._interim_l3_ds,
            dir=self.interim_l3_dir,
            filename=self.interim_l3_filename,
        )

        return self


@dataclass(order=True)
class Gridded:
    sondes: dict
    global_attrs: dict

    def __post_init__(self):
        if self.global_attrs is None:
            self.global_attrs = {}

    def check_aspen_version(self):
        list_of_l2_hist = [
            sonde._interim_l3_ds.attrs["history"].splitlines()[0]
            for sonde in self.sondes.values()
        ]
        aspen_versions = [asp.split("Aspen ")[1] for asp in list_of_l2_hist]
        if not aspen_versions.count(aspen_versions[0]) == len(aspen_versions):
            raise ValueError(
                "Not all sondes have been processed with the same Aspen version"
            )
        return self

    def check_pydropsonde_version(self):
        list_of_l2_hist = [
            sonde._interim_l3_ds.attrs["history"].splitlines()[1]
            for sonde in self.sondes.values()
        ]
        pydrop_versions = [pydr.split("pydropsonde ")[-1] for pydr in list_of_l2_hist]
        if not pydrop_versions.count(pydrop_versions[0]) == len(pydrop_versions):
            raise ValueError(
                "Not all sondes have been processed with the same pydropsonde version to get level 3"
            )
        return self

    def add_history_to_ds(self):
        first_sonde_history = list(self.sondes.values())[0]._interim_l3_ds.attrs[
            "history"
        ]
        new_hist = ""
        for line_nb, line in enumerate(first_sonde_history.splitlines()):
            split_line = line.split(" ", 1)
            print(split_line[0])
            try:
                datetime.fromisoformat(split_line[0])
            except ValueError:
                warnings.warn(
                    f"The first part of line {line_nb} in the history is not a date. It was removed from the attribute"
                )
            new_hist += split_line[1] + "\n"
        self.history = new_hist
        return self

    def concat_sondes(self, sortby=None):
        """
        function to concatenate all sondes using the combination of all measurement times and launch times and add global attributes to resulting dataset
        """
        if sortby is None:
            sortby = list(hh.l3_coords.keys())[0]
        list_of_l2_ds = [sonde._interim_l3_ds for sonde in self.sondes.values()]
        ds = xr.concat(list_of_l2_ds, dim="sonde_id", join="exact").sortby(sortby)

        if hasattr(self, "global_attrs"):
            ds = ds.assign_attrs(self.global_attrs)

        self._interim_l3_ds = ds
        return self

    def get_all_attrs(self):
        attrs = set()
        for sonde in list(self.sondes.values()):
            attrs = set(sonde.attrs) | attrs
        self.attrs = list(attrs)
        return self

    def get_l3_dir(self, l3_dir: str = None):
        if l3_dir:
            self.l3_dir = l3_dir
        elif self.sondes is not None:
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
        """
        Writes the L3 file to the specified directory.

        Parameters
        ----------
        l3_dir : str, optional
            The directory to write the L3 file to.

        Returns
        -------
        self : object
            Returns the sonde object with the L3 file written to the specified directory using the l3_filename attribute to set the name.
        """

        if l3_dir is None:
            l3_dir = self.l3_dir
        ds = self._interim_l3_ds
        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" level3 concatenation with pydropsonde {__version__} \n"
        )
        object.__setattr__(self, "history", history)
        ds.attrs.update({"history": history})

        hx.write_ds(
            ds=self._interim_l3_ds,
            dir=l3_dir,
            filename=self.l3_filename,
            object_dim="sonde_id",
            alt_dim="alt",
        )
        return self

    def add_l3_ds(self, l3_dir: str = None):
        if l3_dir is None:
            self.l3_ds = self._interim_l3_ds.copy()
        else:
            self.l3_ds = hx.open_dataset(l3_dir)
        return self

    def get_simple_circle_times_from_yaml(self, yaml_file: str = None):
        with open(yaml_file) as source:
            flightinfo = yaml.load(source, Loader=yaml.SafeLoader)

        circle_times = []
        sonde_ids = []
        segment_ids = []
        platform_ids = []
        flight_ids = []
        for c in flightinfo["segments"]:
            circle_times.append([(np.datetime64(c["start"]), np.datetime64(c["end"]))])
            segment_ids.append(c["segment_id"])
            platform_ids.append(flightinfo["platform"])
            flight_ids.append(flightinfo["flight_id"])

            try:
                ds_c = self.l3_ds.where(
                    self.l3_ds["launch_time"] > np.datetime64(c["start"]),
                    drop=True,
                ).where(
                    self.l3_ds["launch_time"] < np.datetime64(c["end"]),
                    drop=True,
                )
            except ValueError:
                c_id = c["segment_id"]
                print(f"No sondes for circle {c_id}. It is omitted")
                sonde_ids.append([])
            else:
                print(ds_c.sonde_id.values)
                sonde_ids.append(list(ds_c.sonde_id.values))

        self.circle_times = circle_times
        self.sonde_ids = sonde_ids
        self.segment_ids = segment_ids
        self.platform_ids = platform_ids
        self.flight_ids = flight_ids

        return self

    def get_circle_info_from_yaml(self, yaml_dir: str = None):
        allyamlfiles = sorted(glob.glob(yaml_dir + "*.yaml"))

        circle_times = []
        sonde_ids = []
        flight_ids = []
        platform_ids = []
        segment_ids = []
        segment_ids = []

        for i in allyamlfiles:
            with open(i) as source:
                flightinfo = yaml.load(source, Loader=yaml.SafeLoader)
            platform_ids.append(flightinfo.platform)
            flight_ids.append(flightinfo["flight_id"])
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

            segment_ids.append(
                [
                    (c["segment_id"])
                    for c in flightinfo["segments"]
                    if "circle" in c["kinds"]
                    if len(c["dropsondes"]["GOOD"]) >= 6
                ]
            )

        self.circle_times = circle_times
        self.sonde_ids = sonde_ids
        self.segment_ids = segment_ids
        self.platform_ids = platform_ids
        self.flight_ids = flight_ids

        return self
