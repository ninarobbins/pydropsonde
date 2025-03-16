import ast
from dataclasses import dataclass, field, KW_ONLY

from datetime import datetime, timezone
from typing import Any, Optional, List
import os
import subprocess
import warnings
import yaml
import hashlib
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram

import pydropsonde.helper as hh
from pydropsonde.helper.quality import QualityControl
import pydropsonde.helper.xarray_helper as hx
import pydropsonde.helper.rawreader as rr
import pydropsonde.helper.stats as hs
from importlib.metadata import version
from pathlib import Path

__version__ = version("pydropsonde")

_no_default = object()


@dataclass(order=True)
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
    _serial_id: str
    cont: bool = True
    _: KW_ONLY
    _launch_time: Optional[Any] = None
    sonde_rev: Optional[str] = None

    @property
    def flight_id(self):
        return self._flight_id

    @property
    def serial_id(self):
        return self._serial_id

    def set_serial_id(self, serial_id):
        self._serial_id = serial_id

    @property
    def platform_id(self):
        return self._platform_id

    @property
    def launch_time(self):
        return self._launch_time

    def set_launch_time(self, launch_time):
        self._launch_time = launch_time

    @property
    def aspen_ds(self):
        return self._aspen_ds

    def set_aspen_ds(self, value):
        self._aspen_ds = value

    @property
    def l2_ds(self):
        return self._l2_ds

    def set_l2_ds(self, ds):
        self._l2_ds = ds

    @property
    def is_minisonde(self):
        return self.sonde_rev == "N1"

    @property
    def sonde_hash(self):
        with open(self.dfile, "r") as f:
            header = f.readline()
            return hashlib.sha256(header.encode("ascii")).hexdigest()[-8:]

    @property
    def id(self):
        return self.sonde_hash

    def __post_init__(self):
        """
        Initializes the 'qc' attribute as an empty object and sets the 'sort_index' attribute based on 'launch_time'.

        The 'sort_index' attribute is only applicable when 'launch_time' is available. If 'launch_time' is None, 'sort_index' will not be set.
        """
        self.qc = QualityControl()
        if self.launch_time is not None:
            self.sort_index = self.launch_time
        self.sonde_dim = "sonde"

    def add_flight_id(self, flight_id: str, flight_template: str = None) -> None:
        """Sets attribute of flight ID

        Parameters
        ----------
        flight_id : str
            The flight ID of the flight during which the sonde was launched
        """
        if flight_template is not None:
            flight_id = flight_template.format(flight_id=flight_id)

        self._flight_id = flight_id

    def add_platform_id(self, platform_id: str) -> None:
        """Sets attribute of platform ID

        Parameters
        ----------
        platform_id : str
            The platform ID of the flight during which the sonde was launched
        """
        self._platform_id = platform_id

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
            self.launch_alt = launch_alt
            self.launch_lat = launch_lat
            self.launch_lon = launch_lon
        except (ValueError, TypeError):
            print(
                "Check if the sonde detected a launch, otherwise launch coordinates cannot be set"
            )

    def add_launch_detect(self, launch_detect_bool: bool | None) -> None:
        """Sets bool attribute of whether launch was detected

        Parameters
        ----------
        launch_detect_bool : bool
            True if launch detected, else False
        """
        self.launch_detect = launch_detect_bool

    def add_afile(self, path_to_afile: str | None) -> None:
        """Sets attribute with path to A-file of the sonde

        Parameters
        ----------
        path_to_afile : str
            Path to the sonde's A-file
        """
        self.afile = path_to_afile
        return self

    def add_level_dir(self, l0_dir: str = None, l1_dir: str = None, l2_dir: str = None):
        """
        Sets the directory paths for different data levels (Level 0, Level 1, Level 2)
        within the object. If specific directories are not provided, default paths
        are generated based on the existing 'afile' attribute or by replacing
        'Level_0' in the path with 'Level_1' or 'Level_2'.

        Parameters:
        - l0_dir (str, optional): The directory path for Level 0 data. If not provided,
        it defaults to the directory of 'D-file'.
        - l1_dir (str, optional): The directory path for Level 1 data. If not provided,
        it defaults to the Level 0 directory with 'Level_0' replaced by 'Level_1'.
        Can include a placeholder '{flight_id}' for dynamic replacement.
        - l2_dir (str, optional): The directory path for Level 2 data. If not provided,
        it defaults to the Level 0 directory with 'Level_0' replaced by 'Level_2'.
        Can include a placeholder '{flight_id}' for dynamic replacement.
        """
        if l0_dir is None:
            l0_dir = os.path.dirname(self.dfile)
        if l1_dir is None:
            l1_dir = l0_dir.replace("Level_0", "Level_1")
        else:
            l1_dir = l1_dir.format(flight_id=self.flight_id)
        if l2_dir is None:
            l2_dir = l0_dir.replace("Level_0", "Level_2")
        else:
            l2_dir = l2_dir.format(flight_id=self.flight_id)
        self.l0_dir = l0_dir
        self.l1_dir = l1_dir
        self.l2_dir = l2_dir

    def add_broken(self, broken_sondes: dict):
        """
        Assigns a dictionary of broken sondes to this sonde.

        This method sets the 'broken_sondes' attribute of this sonde
        to the provided dictionary of broken sondes.

        Parameters:
        - broken_sondes (dict): A dictionary containing information about
        broken sondes, where keys are identifiers and values are details
        about the broken sondes.
        """
        self.broken_sondes = broken_sondes

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

        l0_dir = os.path.abspath(self.l0_dir)
        dname = os.path.basename(self.dfile)
        l1_dir = os.path.abspath(self.l1_dir)
        l1_name = dname.split(".")[0] + "QC.nc"

        if path_to_postaspenfile is None:
            path_to_postaspenfile = os.path.join(l1_dir, l1_name)

        if not os.path.exists(path_to_postaspenfile):
            if os.path.getsize(os.path.join(l0_dir, dname)) > 0:
                os.makedirs(l1_dir, exist_ok=True)

                command = [
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
                ]

                if self.is_minisonde:
                    command += ["-1", "mini-dropsonde"]

                subprocess.run(
                    command,
                    check=True,
                )
            else:
                warnings.warn(
                    f"L0 file for sonde {self.serial_id} on {self.flight_id} is empty. No processing done"
                )
        self.postaspenfile = path_to_postaspenfile
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
            except OSError:
                warnings.warn(
                    f"Empty l1 file for sonde {self.serial_id} on {self.flight_id}. This might be fixed using the ASPEN software manually. "
                )
                return None
            if "SondeId" not in ds.attrs:
                if ds.attrs["SoundingDescription"].split(" ")[1] == self.serial_id:
                    self.set_aspen_ds(ds)
                else:
                    raise ValueError(
                        f"I didn't find the `SondeId` attribute, so checked the `SoundingDescription` attribute. I found the ID in the `SoundingDescription` global attribute ({ds.attrs['SoundingDescription'].split(' ')[1]}) to not match with this instance's `serial_id` attribute ({self.serial_id}). Therefore, I am not storing the xarray dataset as an attribute."
                    )
            elif ds.attrs["SondeId"] == self.serial_id:
                self.set_aspen_ds(ds)

            elif self.launch_detect is None:
                warnings.warn(
                    f"I found the `SondeId` global attribute ({ds.attrs['SondeId']}) to not match with this instance's `serial_id` attribute ({self.serial_id}). This could be due to no afile for this sonde. Serial id is updated."
                )
                self.set_serial_id(ds.attrs["SondeId"])
                self.set_aspen_ds(ds)

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
        """
        Append ASPEN processing history to the object's history attribute.

        This method retrieves the ASPEN version and processing time from the `aspen_ds` attribute
        of the sonde and appends a formatted history entry to the `history` attribute. The processing
        time is expected to be in UTC format.

        Returns:
            self: The sonde object with updated history.

        Raises:
            AssertionError: If the processing time does not end with "UTC".
        """

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
        self.history = history
        return self

    def filter_no_launch_detect(self) -> Optional["Sonde"]:
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
            if self.launch_detect is True or self.launch_detect is None:
                return self
            else:
                print(
                    f"No launch detected for Sonde {self.serial_id}. I am not running QC checks for this Sonde."
                )
        else:
            raise ValueError(
                f"The attribute `launch_detect` does not exist for Sonde {self.serial_id}."
            )

    def detect_floater(
        self,
        skip: bool = False,
    ):
        """
        Detects if a sonde is a floater.

        Parameters
        ----------
        skip : bool, optional
            If True, the function will return the object without performing any operations. Default is False.

        Returns
        -------
        self
            The object itself with the new `is_floater` entry added to the quality control object
        """
        if hh.get_bool(skip):
            return self
        else:
            if hasattr(self, "aspen_ds"):
                landing_time = self.qc.get_is_floater()
                self.landing_time = landing_time
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
        if hasattr(self.qc, "is_floater"):
            if self.qc.is_floater:
                cropped_ds = self.aspen_ds.sel(time=slice(self.landing_time, None))
                self.cropped_aspen_ds = cropped_ds
                self.qc.set_qc_ds(cropped_ds)

        else:
            raise ValueError(
                "The attribute `is_floater` does not exist. Please run `detect_floater` method first."
            )
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
        if self.qc.is_floater:
            ds = self.cropped_aspen_ds
        else:
            ds = self.aspen_ds
        self.interim_l2_ds = ds

        return self

    def below_aircraft_qc(self, max_alt=15000):
        """
        Remove measurements that are above the aircraft altitude or some threshold value if aircraft altitude is not given.

        Parameters
        ----------
        max_alt :
            maximum realistic meassured altitude

        """
        self.qc.set_qc_ds(self.interim_l2_ds)

        aircraft_alt = self.flight_attrs.get(
            "aircraft_msl_altitude_(m)", float(max_alt)
        )
        self.qc.alt_below_aircraft(aircraft_alt)

        return self

    def init_qc(self, qc_vars=None):
        """
        set the variables for which to run the quality control.

        Parameters
        ----------
        qc_vars : list or string
            contains the Level1 variables for which to do the qc. Can be a list of strings or a string with comma-separated variable names
        """

        if qc_vars is None:
            qc_vars = {"u": "m s-1", "v": "m s-1", "rh": "1", "ta": "K", "p": "Pa"}
        self.qc.set_qc_variables(qc_vars)
        self.qc.set_qc_ds(self.aspen_ds)
        return self

    def get_qc(self, run_qc=None):
        """
        Run qc functions.

        Parameters
        ----------
        run_qc : list or string
            contains qc functions in the QualityControl class that should be run on the qc_variables. Can be a list of strings or a string with comma-separated variable names
        """
        if run_qc is None:
            run_qc = [
                "profile_sparsity",
                "profile_extent",
                "near_surface_coverage",
                "alt_near_gpsalt",
                "sfc_physics",
            ]
        elif isinstance(run_qc, str):
            run_qc = run_qc.split(",")
        for fct in run_qc:
            qc_fct = getattr(self.qc, fct)
            qc_fct()
        return self

    def remove_non_qc_sondes(self, used_flags=None, remove_ugly=True):
        """
        Removes sondes that do not pass quality control checks.

        This method checks the quality control (QC) status of the sonde using the specified flags.
        If the sonde passes the QC checks, it is retained; otherwise, it is filtered out.

        Parameters:
        - used_flags (optional): A list of flags to be used for the QC check. If not provided, default flags are used.
        - remove_ugly (bool, optional): A flag indicating whether drop 'UGLY' sondes.
        Defaults to True.

        Returns:
        - self: If the sonde passes the QC checks.
        - None: If the sonde fails the QC checks

        Prints a message indicating the sonde has been filtered out if it fails the QC checks.
        """
        if self.qc.check_qc(used_flags, check_ugly=remove_ugly):
            return self
        else:
            print(
                f"Quality control returned False. Therefore, filtering this sonde ({self.serial_id}) out from L2"
            )
            return None

    def convert_to_si(self, variables=["rh", "p", "ta"], skip=False):
        """
        Converts variables to SI units.

        Parameters
        ----------
        variables : list or str, optional
            The variables to convert to SI units. If a string is provided, it should be a comma-separated list of variables.
            The default variables are 'rh', 'pres', and 'tdry'.

        skip : bool, optional
            If set to True, the function will skip the conversion process but will still ensure that the 'interim_l2_ds' attribute is set.
            If 'interim_l2_ds' is not already an attribute of the object, it will be set to 'aspen_ds'.
            Default is False.

        Returns
        -------
        self : object
            Returns the sonde object with the specified variables in aspen_ds converted to SI units.
            If 'skip' is set to True, it returns the sonde object with 'interim_l2_ds' set to 'aspen_ds' if it wasn't already present.
        """
        if hh.get_bool(skip):
            if hasattr(self, "interim_l2_ds"):
                return self
            else:
                self.interim_l2_ds = self.aspen_ds.copy()
                return self
        else:
            if isinstance(variables, str):
                variables = variables.split(",")

            if hasattr(self, "interim_l2_ds"):
                ds = self.interim_l2_ds
            else:
                ds = self.aspen_ds

            for variable in variables:
                func = hh.get_si_converter_function_based_on_var(variable)
                var_attrs = ds[variable].attrs
                ds = ds.assign({f"{variable}": func(ds[variable])})
                ds[variable].attrs.update(var_attrs)
            self.interim_l2_ds = ds

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
            Returns the sonde object with only the specified variables (renamed if dictionary has 'rename_to' key and attributes added if dictionary has 'attributes' key) in interim_l2_ds attribute.
            If 'interim_l2_ds' is not already an attribute of the object, it will first be set to 'aspen_ds' before reducing to the variables and renaming.
        """
        if isinstance(l2_variables, str):
            l2_variables = ast.literal_eval(l2_variables)

        l2_variables_list = list(l2_variables.keys())

        if hasattr(self, "interim_l2_ds"):
            ds = self.interim_l2_ds
        else:
            ds = self.aspen_ds

        ds = ds[l2_variables_list]

        for variable, variable_dict in l2_variables.items():
            if "attributes" in variable_dict:
                ds[variable].attrs = variable_dict["attributes"]
            ds = ds.rename({variable: variable_dict["rename_to"]})
        self.interim_l2_ds = ds

        return self

    def get_flight_attributes(
        self, l2_flight_attributes_map: dict = hh.l2_flight_attributes_map
    ) -> "Sonde":
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

        if not self.afile:
            print(
                f"No flight attributes for sonde {self.serial_id} on {self.flight_id}"
            )
        else:
            with open(self.afile, "r") as f:
                lines = f.readlines()

            for attr in l2_flight_attributes_map.keys():
                for line_id, line in enumerate(lines):
                    if attr in line:
                        break

                attr = l2_flight_attributes_map.get(attr, attr)
                try:
                    value = lines[line_id].split("= ")[1]
                    flight_attrs[attr] = float(value) if "AVAPS" not in attr else value
                except UnboundLocalError:
                    print(
                        f"No flight attributes for sonde {self.serial_id} on {self.flight_id}"
                    )
                    break
        self.flight_attrs = flight_attrs

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
        self.global_attrs = attributes

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
            "sonde_time": (
                str(self.aspen_ds.launch_time.values)
                if hasattr(self.aspen_ds, "launch_time")
                else np.datetime64(self.aspen_ds.base_time.values)
            ),
            "is_floater": self.qc.is_floater.__str__(),
            "sonde_serial_ID": self.serial_id,
            "sonde_ID": self.id,
        }
        self.sonde_attrs = sonde_attrs

        return self

    def add_sonde_id_variable(self, variable_name="sonde_id"):
        """
        Adds a variable and related attributes to the sonde object with the Sonde object (self)'s id attribute.

        Parameters
        ----------
        variable_name : str, optional
            The name of the variable to be added. Default is 'sonde_id'.

        Returns
        -------
        self : object
            Returns the sonde object with a variable containing id. Name of the variable provided by 'variable_name'.
        """
        if hasattr(self, "interim_l2_ds"):
            ds = self.interim_l2_ds
        else:
            ds = self.aspen_ds
        attrs = {
            "descripion": "unique sonde ID",
            "long_name": "sonde identifier",
            "cf_role": "trajectory_id",
        }
        ds = ds.assign({variable_name: self.id})
        ds[variable_name] = ds[variable_name].assign_attrs(attrs)

        ds = ds.assign({"vaisala_serial_id": self.serial_id})
        ds["vaisala_serial_id"] = ds["vaisala_serial_id"].assign_attrs(
            {
                "description": "device serial ID provided by the manufacturer Vaisala",
                "long_name": "vaisala serial identifier",
            }
        )
        self.interim_l2_ds = ds

        return self

    def add_platform_id_variable(self, variable_name="platform_id"):
        """
        Adds a variable and related attributes to the sonde object with the Sonde object (self)'s platform_id attribute.

        Parameters
        ----------
        variable_name : str, optional
            The name of the variable to be added. Default is 'platform_id'.

        Returns
        -------
        self : object
            Returns the sonde object with a variable containing platform_id. Name of the variable provided by 'variable_name'.
        """
        if hasattr(self, "interim_l2_ds"):
            ds = self.interim_l2_ds
        else:
            ds = self.aspen_ds

        attrs = dict(
            description="unique platform ID",
            long_name="platform identifier",
        )
        ds = ds.assign({variable_name: self.platform_id})
        ds[variable_name] = ds[variable_name].assign_attrs(attrs)
        self.interim_l2_ds = ds
        return self

    def add_flight_id_variable(self, variable_name="flight_id"):
        """
        Adds a variable and related attributes to the sonde object with the Sonde object (self)'s flight_id attribute.

        Parameters
        ----------
        variable_name : str, optional
            The name of the variable to be added. Default is 'flight_id'.

        Returns
        -------
        self : object
            Returns the sonde object with a variable containing flight_id. Name of the variable provided by 'variable_name'.
        """
        if hasattr(self, "interim_l2_ds"):
            ds = self.interim_l2_ds
        else:
            ds = self.aspen_ds

        attrs = dict(
            description="unique flight ID",
            long_name="flight identifier",
        )

        ds = ds.assign({variable_name: self.flight_id})
        ds[variable_name] = ds[variable_name].assign_attrs(attrs)
        self.interim_l2_ds = ds
        return self

    def update_history_l2(self):
        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" quality control with pydropsonde {__version__} \n"
        )
        self.history = history
        return self

    def add_l2_attributes_to_interim_l2_ds(self):
        """
        Adds flight, sonde and global attributes to interim_l2_ds.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with flight, sonde and global attributes added to interim_l2_ds.
        """
        ds = self.interim_l2_ds

        attrs_to_del = []
        for attr in ds.attrs.keys():
            attrs_to_del.append(attr)

        for attr in attrs_to_del:
            del ds.attrs[attr]

        if hasattr(self, "flight_attrs"):
            ds = ds.assign_attrs(self.flight_attrs)
        if hasattr(self, "sonde_attrs"):
            ds = ds.assign_attrs(self.sonde_attrs)

        self.interim_l2_ds = ds

        return self

    def add_qc_to_l2(
        self,
        add_var_qc=True,
        add_details=True,
    ):
        """
        Adds quality control (QC) flags to the level 2 dataset.

        This function updates the internal level 2 dataset (`interim_l2_ds`) by adding
        quality control flags. If `add_as_vars` is set to True, it adds the QC flags as
        variables within the dataset. The function assigns an overall quality flag named
        "sonde_all_qc" and updates its attributes to describe the flag values and meanings.
        It also iterates over the QC variables, adding individual QC flags and their
        details to the dataset.

        Parameters:
        - add_as_vars (bool): If True, QC flags are added as variables in the dataset.

        Returns:
        - self: The instance of the class with the updated dataset.

        """
        ds = self.interim_l2_ds
        if add_var_qc:
            for variable in self.qc.qc_vars:
                ds = self.qc.add_variable_flags_to_ds(ds, variable, details=add_details)

            ds = self.qc.add_non_var_qc_to_ds(ds)
        ds = self.qc.add_sonde_flag_to_ds(ds, "sonde_qc")
        self.interim_l2_ds = ds
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

        self.l2_filename = l2_filename or (
            l2_filename_template or hh.l2_filename_template
        ).format(
            platform=self.platform_id,
            flight_id=self.flight_id,
            sonde_hash=self.sonde_hash,
            id=self.id,
        )

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

        ds = self.interim_l2_ds
        if hasattr(self, "broken_sondes"):
            if self.serial_id in self.broken_sondes:
                ds.attrs.update(
                    {"comment": self.broken_sondes[self.serial_id]["error"]}
                )

        ds.attrs.update(
            self.global_attrs["global"],
        )
        ds.attrs.update(
            self.global_attrs["l2"],
        )
        ds.attrs.update(
            dict(
                history=self.history,
                title=self.global_attrs["l2"].get(
                    "title",
                    self.global_attrs.get("title", "Dropsonde Data") + " Level_2",
                )
                + f", {self.id}",
            )
        )

        hx.write_ds(
            ds=ds,
            dir=l2_dir,
            filename=self.l2_filename,
            object_dims=(self.sonde_dim,),
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
            self.set_l2_ds(hx.open_dataset(os.path.join(l2_dir, self.l2_filename)))

            return self
        except FileNotFoundError:
            return None

    def create_interim_l3(self):
        """
        Assigns sonde_id coordinate to the  Level 2 dataset (`l2_ds`)  and sorts the dataset by time.
        The resulting dataset is stored in`interim_l3_ds` within the object.

        Returns:
            self: A sonde object with the updated `interim_l3_ds` attribute.
        """
        self.interim_l3_ds = self.l2_ds.sortby("time")

        return self

    def check_interim_l3(
        self,
        interim_l3_dir: str = None,
        interim_l3_filename: str = None,
        skip=False,
    ):
        """
        sets interim level 3 directory and filename.
        Checks if an interim level 3 file already exists and sets a cont attribute in the sonde
        to skip level 2 to level 3 processing if so.

        Parameters:
        - interim_l3_dir (str, optional): The directory for interim Level 3 files.
        Defaults to a modified version of the Level 2 directory.
        - interim_l3_filename (str, optional): The filename for the interim Level 3
        file. Defaults to a formatted string using `sonde_id` and `version`.
        - skip (bool, optional): If True, skips checking for the file's existence.
        Defaults to False.

        Returns:
        - self: The sonde instance with updated attributes.
        """
        self.interim_l3_dir = interim_l3_dir or self.l2_dir.replace(
            "Level_2", "Level_3_interim"
        ).replace(self.flight_id, "")
        self.interim_l3_filename = (
            interim_l3_filename or "interim_l3_{id}_{version}.zarr"
        ).format(id=self.id, version=__version__)

        if (not skip) and os.path.exists(
            os.path.join(self.interim_l3_dir, self.interim_l3_filename)
        ):
            self.interim_l3_ds = hx.open_dataset(
                os.path.join(self.interim_l3_dir, self.interim_l3_filename)
            )
            self.cont = False
            self.alt_dim = list(self.interim_l3_ds.sizes.keys())[-1]
            return self
        else:
            return self

    def remove_above_aircraft(self, max_alt=15000):
        """
        remove measured values above aircraft
        """
        variables = ["lat", "lon", self.alt_dim, "u", "v"]
        maxalt = self.flight_attrs.get("aircraft_msl_altitude_(m)", float(max_alt))
        self.interim_l3_ds = hx.remove_above_alt(
            self.interim_l3_ds, variables, alt_dim=self.alt_dim, maxalt=maxalt
        )

        return self

    def remove_unphysical(self):
        """
        remove sondes that did not pass the sfc physics qc
        """
        variables = ["rh", "p", "ta"]
        ds = self.interim_l3_ds
        for var in variables:
            if ds[f"{var}_qc"].values == 8:
                ds = ds.assign(
                    {
                        var: (
                            ds[var].dims,
                            ds[var].where(np.isnan(ds[var])).values,
                            ds[var].attrs,
                        )
                    }
                )
        self.interim_l3_ds = ds
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
        ds = self.interim_l3_ds

        ds = hh.calc_theta_from_T(ds)
        ds = hh.calc_q_from_rh_sonde(ds)

        self.interim_l3_ds = ds

        return self

    def recalc_rh_and_ta(self):
        """
        Recalculates relative humidity and temperature after the interpolation and
        adds it to the interim level 3 dataset

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with potential temperature and specific humidity added to the interim l3 dataset.
        """
        ds = self.interim_l3_ds
        ds = hh.calc_T_from_theta(ds)
        ds = hh.calc_rh_from_q(ds)
        self.interim_l3_ds = ds
        return self

    def add_iwv(self):
        """
        Calculates interpolated water vapor from the interim l3 dataset.

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with integrated water vapour added to the interim l3 dataset.
        """
        self.interim_l3_ds = hh.calc_iwv(
            self.interim_l3_ds,
            qc_var=["rh_qc", "ta_qc"],
            alt_dim=self.alt_dim,
            sonde_dim=self.sonde_dim,
        )

        return self

    def add_thetas(self):
        """
        Calculates theta_e from the interim l3 dataset and adds it to the interim l3 dataset

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with theta_e added to the interim l3 dataset.
        """
        self.interim_l3_ds = hh.calc_theta_e(self.interim_l3_ds)

        return self

    def add_wind(self):
        """
        Calculates wind direction and speed from the interim l3 dataset
        and adds it to the interim l3 dataset

        Parameters
        ----------
        None

        Returns
        -------
        self : object
            Returns the sonde object with wind direction and speed added to the interim l3 dataset.
        """
        self.interim_l3_ds = hh.calc_wind_dir_and_speed(self.interim_l3_ds)
        return self

    def set_alt_dim(self, alt_dim="alt"):
        """
        Set the altitude dimension attribute for the sonde

        This method updates the `alt_dim` attribute of the `qc` and the sonde.
        It allows for customization of the altitude dimension used as a dimension from level 3 onwards.

        Parameters:
        alt_dim (str): The name of the altitude dimension to set. Defaults to "alt".

        Returns:
        self: Returns the sonde instance
        """
        self.qc.alt_dim = alt_dim
        self.alt_dim = alt_dim
        return self

    def replace_alt_dim(self, interpolate=True):
        """
        Replaces the altitude dimension in the dataset if one altitude coordinate is worse than the other

        -> if no gpsalt values are present and the sonde probably didn't reach the ground it is dropped
        -> if the sonde did not reach the ground, but gpsalt is available, gpsalt is used
        -> if gpsalt does not go until the surface, but pressure measurements suggest that the sonde reached the surface, alt is used


        Returns:
        - self: Returns the object itself if the altitude dimension is successfully replaced or remains valid.
        - None: Returns None if the dataset is dropped due to NaN altitude values.
        """
        alt_dim = self.alt_dim
        ds = self.interim_l3_ds
        alt_attrs = ds[alt_dim].attrs
        if (not self.qc.qc_flags["p_sfc_physics"]) and (np.all(np.isnan(ds["gpsalt"]))):
            print(
                f"No gpsalt values and no reliable alt values.  Sonde {self.serial_id} from {self.flight_id} is dropped"
            )
            return None
        elif alt_dim == "alt":
            self.qc.qc_flags.update({"altitude_source": "alt"})
            if not self.qc.qc_flags["p_sfc_physics"]:
                for var in ["rh", "ta", "p"]:
                    self.qc.qc_flags[f"{var}_near_surface"] = False
                    self.qc.qc_details[f"{var}_near_surface_count"] = np.nan

                ds = ds.assign({"alt": ds["gpsalt"]})
                self.qc.qc_flags.update({"altitude_source": "gpsalt"})
            ds = ds.rename({"alt": "altitude"}).drop_vars(["gpsalt"])

        elif alt_dim == "gpsalt":
            self.qc.qc_flags.update({"altitude_source": "gpsalt"})
            if (
                (
                    (not self.qc.qc_flags["u_near_surface"])
                    and (self.qc.qc_flags["p_near_surface"])
                )
                or (
                    (not self.qc.qc_flags["u_profile_extent"])
                    and (self.qc.qc_flags["p_profile_extent"])
                )
            ) and (self.qc.qc_flags["p_sfc_physics"]):
                ds = ds.assign({alt_dim: ds["alt"]})
                self.qc.qc_flags.update({"altitude_source": "alt"})
            elif not self.qc.qc_flags["p_sfc_physics"]:
                for var in ["rh", "ta", "p"]:
                    self.qc.qc_flags[f"{var}_near_surface"] = False
                    self.qc.qc_details[f"{var}_near_surface_count"] = np.nan
            ds = ds.rename({"gpsalt": "altitude"}).drop_vars(["alt"])
        else:
            self.qc.qc_flags.update({"altitude_source": self.alt_dim})
        if hh.get_bool(interpolate):
            ds = ds.assign(
                {"altitude": ds["altitude"].sortby("time").interpolate_na(dim="time")}
            )
        alt_attrs["long_name"] = "altitude"
        alt_attrs["description"] = (
            "derived from either gpsalt or alt in Level 2. See alt_source"
        )
        ds.altitude.attrs.update(alt_attrs)
        self.alt_dim = "altitude"
        self.qc.alt_dim = "altitude"
        self.interim_l3_ds = self.qc.add_alt_source_to_ds(ds)

        return self

    def swap_alt_dimension(self):
        """
        Swap the 'time' dimension with an alternative dimension (either alt or gpsalt) in the dataset.

        This method swaps the 'time' dimension of the dataset with an alternative
        dimension specified by the `alt_dim` attribute of the object. It then loads
        the dataset with the new dimension configuration and updates the object's
        internal dataset attribute `interim_l3_ds`.

        Returns:
            self: The instance of the object with the updated dataset.
        """
        alt_dim = self.alt_dim
        self.interim_l3_ds = self.interim_l3_ds.swap_dims({"time": alt_dim})

        return self

    def remove_non_mono_incr_alt(self, bottom_up=True):
        """
        This function removes the indices in the some height variable that are not monotonically increasing
        """
        alt_dim = self.alt_dim

        ds = self.interim_l3_ds

        diff_array = ds[alt_dim].sortby("time").dropna(dim="time").diff(dim="time")
        if not np.all(diff_array < 0):
            warnings.warn(
                f"your altitude for sonde {self.serial_id} on {
                    self.launch_time
                } is not sorted."
            )
            if bottom_up:
                alt = ds[alt_dim].sortby("time", ascending=False).values
                idx = (
                    diff_array.sortby("time", ascending=False)
                    .where(diff_array > 0)
                    .argmin(dim="time")
                    .values
                )
                curr_alt = alt[idx]
                idx = idx + 1
                while idx < ds.sizes["time"]:
                    if alt[idx] < curr_alt:
                        alt[idx] = np.nan
                    elif ~np.isnan(alt[idx]):
                        curr_alt = alt[idx]
                    idx += 1
                ds = ds.assign({alt_dim: ("time", alt[::-1], ds[alt_dim].attrs)})

            else:
                alt = ds[alt_dim]
                curr_alt = alt.isel(time=0)
                for i in range(len(alt)):
                    if alt[i] > curr_alt:
                        alt[i] = np.nan
                    elif ~np.isnan(alt[i]):
                        curr_alt = alt[i]
                ds[alt_dim] = alt

        self.interim_l3_ds = ds

        return self

    def interpolate_alt(
        self,
        interp_start=-5,
        interp_stop=14600,
        interp_step=10,
        max_gap_fill: int = 50,
        interpolate=False,
        p_log=True,
        method: str = "bin",
    ):
        """
        Interpolate sonde data along comon altitude grid to prepare concatenation
        """
        alt_dim = self.alt_dim
        interpolation_grid = np.arange(interp_start, interp_stop, interp_step)
        ds = self.interim_l3_ds

        if p_log:
            ds = ds.assign(p=(ds.p.dims, np.log(ds.p.values), ds.p.attrs))
        if method == "linear_interpolate":
            interp_ds = ds.interp({alt_dim: interpolation_grid})
        elif method == "bin":
            mean_ds = {}
            count_dict = {}
            # bin variables along height, bins are right-open intervals, except the last
            for var in [
                "u",
                "v",
                "q",
                "p",
                "theta",
                "lat",
                "lon",
                "gpsalt",
                "time",
                "alt",
            ]:
                if (var in ds.variables) and (var not in ds.dims):
                    count_dict[var] = histogram(
                        ds[alt_dim].where(~np.isnan(ds[var])),
                        bins=interpolation_grid,
                        dim=[alt_dim],
                    )

                    new_ds = (
                        histogram(
                            ds[alt_dim].where(~np.isnan(ds[var])),
                            bins=interpolation_grid,
                            dim=[alt_dim],
                            weights=ds[var]
                            .astype(np.float64)
                            .where(~np.isnan(ds[var])),
                        )  # casting necessary for time
                        / count_dict[var]
                    )
                    new_ds.name = var
                    new_ds = new_ds.assign_attrs(ds[var].attrs)
                    mean_ds[var] = new_ds.copy()
            interp_ds = xr.Dataset(mean_ds)
            count_dict.pop("time")
            self.count_dict = count_dict
            interp_ds = self.add_N_values(interp_ds)
            # interpolate missing values up to max_gap_fill meters
            if interpolate:
                interp_ds = (
                    interp_ds.transpose()
                    .interpolate_na(
                        dim=f"{alt_dim}_bin", max_gap=max_gap_fill, use_coordinate=True
                    )
                    .rename({f"{alt_dim}_bin": alt_dim})
                )
                interp_ds = self.add_m_values(interp_ds)

            else:
                interp_ds = interp_ds.rename({f"{alt_dim}_bin": alt_dim})
            interp_ds[alt_dim].attrs.update(ds[alt_dim].attrs)
            time_type = ds["time"].values.dtype
            time_attrs = interp_ds.time.attrs
            time_attrs.update({"long_name": f"time after binning in {alt_dim}"})
            interp_ds = interp_ds.assign(
                bin_average_time=(
                    interp_ds.time.dims,
                    interp_ds.time.astype(time_type).values,
                    time_attrs,
                )
            ).drop_vars("time")

        if p_log:
            interp_ds = interp_ds.assign(
                p=(interp_ds.p.dims, np.exp(interp_ds.p.values), interp_ds.p.attrs)
            )
        self.interim_l3_ds = interp_ds
        return self

    def add_N_values(self, prep_l3):
        """
        Updates the internal dataset with the number of values per bin

        Attributes:
            - alt_dim (str): The name of the altitude dimension.
            _count_dict (dict): A dictionary containing variables and their corresponding count data arrays.
            interim_l3_ds (xarray.Dataset): The dataset to be updated with new variables.

        Returns:
            self: The updated sonde with the modified dataset.
        """
        alt_dim = self.alt_dim
        count_dict = self.count_dict

        for variable, Nvar in count_dict.items():
            N_name = f"{variable}_N_qc"
            N_attrs = dict(
                long_name=f"Number of values per bin for {variable}",
                units="1",
            )
            Nvar = Nvar.rename({alt_dim + "_bin": alt_dim})

            prep_l3 = prep_l3.assign(
                {
                    N_name: (
                        alt_dim,
                        Nvar.astype(int).values,
                        N_attrs,
                    )
                }
            )
        return prep_l3

    def add_m_values(self, prep_l3):
        """
        Updates the internal dataset with interpolation method flags for each variable

        Attributes:
            - alt_dim (str): The name of the altitude dimension.
            _count_dict (dict): A dictionary containing variables and their corresponding count data arrays.
            interim_l3_ds (xarray.Dataset): The dataset to be updated with new variables.

        Returns:
            self: The updated sonde with the modified dataset.
        """
        alt_dim = self.alt_dim
        count_dict = self.count_dict
        for variable in count_dict.keys():
            Nvar = prep_l3[f"{variable}_N_qc"]
            n_mask = Nvar.where(~np.isnan(Nvar), 0)
            int_mask = prep_l3[variable].where(~np.isnan(prep_l3[variable]), 0)

            m_mask = np.invert(n_mask.astype(bool)) & int_mask.astype(bool)
            m = xr.where(Nvar > 0, x=2, y=0)
            m = xr.where(m_mask, x=1, y=m)

            m_name = f"{variable}_m_qc"
            m_attrs = {
                "long_name": f"interp method for {variable}",
                "standard_name": "status_flag",
                "flag_values": "0, 1, 2",
                "flag_meanings": "no_data interpolated_no_raw_data average_over_raw_data",
            }
            prep_l3 = prep_l3.assign(
                {
                    m_name: (
                        alt_dim,
                        m.astype(int).values,
                        m_attrs,
                    )
                }
            )
        return prep_l3

    def remove_N_m_duplicates(self):
        """
        Removes duplicate N and m quality control variables (for variables from the same sensor).


        Attributes:
            interim_l3_ds (xarray.Dataset): The dataset from which duplicates are removed and variables are renamed.

        Returns:
            self: The updated sonde with the modified dataset.
        """
        ds = self.interim_l3_ds
        nm_vars = ["lat", "u", "p", "q", "theta"]
        if "lat_m_qc" in ds.variables:
            np.testing.assert_array_equal(
                ds.lat_m_qc.values,
                ds.lon_m_qc.values,
                err_msg="lat_m_qc and lon_m_qc not identical",
            )
            np.testing.assert_array_equal(
                ds.u_m_qc.values,
                ds.v_m_qc.values,
                err_msg="v_m_qc and u_m_qc not identical",
            )
            ds = ds.rename(
                {
                    "lat_m_qc": "gpspos_m_qc",
                    "u_m_qc": "gps_m_qc",
                }
            ).drop_vars(
                [f"{var}_m_qc" for var in ds.variables if var not in nm_vars],
                errors="ignore",
            )
        np.testing.assert_array_equal(
            ds.u_N_qc.values,
            ds.v_N_qc.values,
            err_msg="v_N_qc and u_N_qc not identical",
        )

        ds = ds.drop_vars(
            [f"{var}_N_qc" for var in ds.variables if var not in nm_vars],
            errors="ignore",
        ).rename(
            {
                "lat_N_qc": "gpspos_N_qc",
                "u_N_qc": "gps_N_qc",
            }
        )
        self.interim_l3_ds = ds

        return self

    def add_Nm_to_vars(self, add_m=False):
        """
        Adds ancillary N and m quality control variables to essential variables in the internal dataset.

        Attributes:
           interim_l3_ds (xarray.Dataset): The dataset to which ancillary variables are added.

        Returns:
            self: The updated instance with the modified dataset.
        """
        self.remove_N_m_duplicates()
        ds = self.interim_l3_ds

        essential_vars = ["u", "v", "q", "p", "theta", "lat", "lon"]
        mN_vars = ["gps", "gps", "q", "p", "theta", "gpspos", "gpspos"]

        for essential_var, mNvar in zip(essential_vars, mN_vars):
            if add_m:
                ds = hx.add_ancillary_var(ds, essential_var, mNvar + "_m_qc")
            ds = hx.add_ancillary_var(ds, essential_var, mNvar + "_N_qc")
        self.interim_l3_ds = ds
        return self

    def add_ids(self):
        """
        add sonde_id, platform_id and flight_id to interim_l3_ds
        """
        ds = self.interim_l3_ds
        source_ds = self.l2_ds
        self.interim_l3_ds = ds.assign(
            {
                "sonde_id": (
                    self.sonde_dim,
                    [source_ds.sonde_id.values],
                    source_ds.sonde_id.attrs,
                ),
                "vaisala_serial_id": (
                    self.sonde_dim,
                    [source_ds.vaisala_serial_id.values],
                    source_ds.vaisala_serial_id.attrs,
                ),
                "platform_id": (
                    self.sonde_dim,
                    [source_ds.platform_id.values],
                    source_ds.platform_id.attrs,
                ),
                "flight_id": (
                    self.sonde_dim,
                    [source_ds.flight_id.values],
                    source_ds.flight_id.attrs,
                ),
            }
        )
        return self

    def add_attributes_as_var(self, essential_attrs=None):
        """
        Prepares l2 datasets to be concatenated to gridded.
        adds all attributes as variables to avoid conflicts when concatenating because attributes are different
        (and not lose information)
        """
        ds = self.interim_l3_ds
        l2_ds = self.l2_ds
        if essential_attrs is None:
            try:
                essential_attrs = hh.l3_coords
            except AttributeError:
                essential_attrs = {
                    "sonde_time": {
                        "time_zone": "UTC",
                        "long_name": "dropsonde launch time",
                    }
                }

        for attr, value in l2_ds.attrs.items():
            splt = attr.split("(")
            var_name = splt[0][:-1]
            if var_name in list(essential_attrs.keys()):
                var_attrs = essential_attrs[var_name]
                ds = ds.assign(
                    {var_name: (self.sonde_dim, [l2_ds.attrs[attr]], var_attrs)}
                )
        ds = ds.assign(
            dict(
                sonde_time=(
                    self.sonde_dim,
                    [np.datetime64(self.launch_time, "ns")],
                    essential_attrs["sonde_time"],
                )
            )
        )
        self.attrs = ds.attrs.keys()
        self.interim_l3_ds = ds

        return self

    def make_attr_coordinates(self):
        """
        Reshape and assign coordinats as defined in helper.__init__ to the level 3 dataset.

        Returns:
            self: The instance with updated coordinates in `interim_l3_ds`.
        """
        ds = self.interim_l3_ds
        new_coords = {
            coord: (self.sonde_dim, np.reshape(ds[coord].values, (1,)), ds[coord].attrs)
            for coord in hh.l3_coords
            if coord in ds.variables
        }
        self.interim_l3_ds = ds.assign_coords(new_coords)
        return self

    def add_qc_to_interim_l3(self, keep=None):
        """
        Add quality control flags to the interim Level 3 dataset.


        Args:
            keep (list or str): A list of quality control flags to keep. If 'all',
                                all available flags are kept. If a string, it is
                                split by commas to form a list.
                                Default: sonde_qc

        Returns:
            self: The instance with updated `interim_l3_ds` including quality control flags.
        """
        ds = self.interim_l3_ds

        for var in ds.variables:
            if (var != self.sonde_dim) and (var != "sonde_id"):
                ds[var].attrs.pop("ancillary_variables", None)
        if keep is None:
            keep = []
        else:
            if keep == "all":
                keep = (
                    [f"{var}_qc" for var in list(self.qc.qc_by_var.keys())]
                    + list(self.qc.qc_details.keys())
                    + ["alt_near_gpsalt"]
                )
                for variable in self.qc.qc_vars:
                    ds = self.qc.add_variable_flags_to_ds(ds, variable, details=True)
                if (not np.isin("q", self.qc.qc_vars)) and np.isin(
                    "rh", self.qc.qc_vars
                ):
                    ds = self.qc.add_variable_flags_to_ds(
                        ds, "rh", add_to="q", details=True
                    )
                if (not np.isin("theta", self.qc.qc_vars)) and np.isin(
                    "ta", self.qc.qc_vars
                ):
                    ds = self.qc.add_variable_flags_to_ds(
                        ds, "ta", add_to="theta", details=True
                    )
                ds = self.qc.add_non_var_qc_to_ds(ds)
            elif keep == "var_flags":
                keep = [f"{var}_qc" for var in list(self.qc.qc_by_var.keys())] + [
                    "sonde_qc"
                ]
                for var in self.qc.qc_by_var.keys():
                    ds = hx.add_ancillary_var(ds, var, var + "_qc")
                if (not np.isin("q", self.qc.qc_vars)) and np.isin(
                    "rh", self.qc.qc_vars
                ):
                    ds = hx.add_ancillary_var(ds, "q", "rh_qc")
                if (not np.isin("theta", self.qc.qc_vars)) and np.isin(
                    "ta", self.qc.qc_vars
                ):
                    ds = hx.add_ancillary_var(ds, "theta", "ta_qc")

            else:
                warnings.warn(
                    "your keep argument for the qc flags in level 3 is not valid, no qc added"
                )
                keep = []
        keep = keep + ["sonde_qc"]

        ds = self.qc.add_alt_source_to_ds(ds)
        ds_qc = self.interim_l2_ds[keep]  # .expand_dims(self.sonde_dim)
        assert np.all(np.isin(ds.rh_qc - ds_qc.rh_qc, [0, 4]))
        assert np.all(np.isin(ds.p_qc - ds_qc.p_qc, [0, 4]))
        assert np.all(np.isin(ds.ta_qc - ds_qc.ta_qc, [0, 4]))
        self.interim_l3_ds = xr.merge([ds, ds_qc], compat="override")

        return self

    def update_history_l3(self):
        """
        Update history for Level 3
        """
        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" Level 3 processing with pydropsonde {__version__} \n"
        )
        self.history = history
        return self

    def save_interim_l3(self):
        """
        Save the interim Level 3 dataset to a specified directory.

        Returns:
            self: The instance after saving the `interim_l3_ds`.
        """
        ds = self.interim_l3_ds
        ds.attrs = {}
        ds.attrs.update(
            self.global_attrs["global"],
        )
        ds.attrs.update(
            self.global_attrs["l3"],
        )
        ds.attrs.update(
            dict(
                history=self.history,
                title=self.global_attrs["l3"].get(
                    "title",
                    self.global_attrs.get("title", "Dropsonde Data") + " Level_3",
                ),
            )
        )
        hx.write_ds(
            ds=ds,
            dir=self.interim_l3_dir,
            filename=self.interim_l3_filename,
            object_dims=(self.sonde_dim,),
            alt_dim=self.alt_dim,
        )

        return self

    def add_expected_coords(self):
        """
        Add missing expected coordinates to the dataset.

        This method checks for any missing coordinates in the dataset (`interim_l3_ds`)
        by comparing it to the l3 coordinats as in helper.__init__ (`l3_coords`).
        For each missing coordinate, it assigns a new coordinate filled with NaN values.

        Returns:
            self: The instance with the updated dataset containing all expected coordinates.
        """
        ds = self.interim_l3_ds
        missing_coords = set(hh.l3_coords.keys()) - set(ds.coords)
        for coord in missing_coords:
            ds = ds.assign_coords(
                {
                    coord: (
                        (self.sonde_dim,),
                        np.full(ds.sizes[self.sonde_dim], np.nan),
                        hh.l3_coords[coord],
                    )
                }
            )
        self.interim_l3_ds = ds
        return self


@dataclass(order=True)
class Gridded:
    sondes: dict
    global_attrs: dict
    circles: dict = None

    @property
    def l3_ds(self):
        return self._l3_ds

    def set_l3_ds(self, ds):
        self._l3_ds = ds

    def __post_init__(self):
        """
        Initializes the Gridded object, ensuring that global attributes are set to an empty dictionary if not provided.
        """
        if self.global_attrs is None:
            self.global_attrs = {}

    def check_aspen_version(self):
        """
        Checks if all sondes have been processed with the same Aspen version.

        Raises
        ------
        ValueError
            If sondes have been processed with different Aspen versions.

        Returns
        -------
        self : Gridded
            Returns the Gridded object.
        """
        list_of_l2_hist = [
            sonde.interim_l3_ds.attrs["history"].splitlines()[0]
            for sonde in self.sondes.values()
        ]
        aspen_versions = [asp.split("Aspen ")[1] for asp in list_of_l2_hist]
        if not aspen_versions.count(aspen_versions[0]) == len(aspen_versions):
            raise ValueError(
                "Not all sondes have been processed with the same Aspen version"
            )
        return self

    def check_pydropsonde_version(self):
        """
        Checks if all sondes have been processed with the same pydropsonde version.

        Raises
        ------
        ValueError
            If sondes have been processed with different pydropsonde versions.

        Returns
        -------
        self : Gridded
            Returns the Gridded object.
        """
        list_of_l2_hist = [
            sonde.interim_l3_ds.attrs["history"].splitlines()[1]
            for sonde in self.sondes.values()
        ]
        pydrop_versions = [pydr.split("pydropsonde ")[-1] for pydr in list_of_l2_hist]
        if not pydrop_versions.count(pydrop_versions[0]) == len(pydrop_versions):
            raise ValueError(
                "Not all sondes have been processed with the same pydropsonde version to get level 3"
            )
        return self

    def add_history_to_gridded(self):
        """
        Adds history information to the dataset by processing the history attribute of the first sonde.

        Returns
        -------
        self : Gridded
            Returns the Gridded object.
        """
        first_sonde_history = list(self.sondes.values())[0].interim_l3_ds.attrs[
            "history"
        ]
        new_hist = ""
        for line_nb, line in enumerate(first_sonde_history.splitlines()):
            split_line = line.split(" ", 1)
            try:
                datetime.fromisoformat(split_line[0])
            except ValueError:
                warnings.warn(
                    f"The first part of line {line_nb} in the history is not a date. It was removed from the attribute"
                )
            new_hist += split_line[1] + "\n"
        self.history = new_hist
        return self

    def add_dim_names(self):
        """
        Adds altitude dimension from the first sonde to the Gridded object.

        Returns
        -------
        self : Gridded
            Returns the Gridded object.
        """
        sonde = list(self.sondes.values())[0]
        self.alt_dim = sonde.alt_dim
        self.sonde_dim = sonde.sonde_dim
        return self

    def concat_sondes(self, sortby=None, coords=None):
        """
        Concatenates all sondes using the combination of all measurement times and launch times, and adds global attributes to the resulting dataset.

        Parameters
        ----------
        sortby : str, optional
            The coordinate to sort the concatenated dataset by.
        coords : dict, optional
            Coordinates to assign to the dataset if missing.

        Returns
        -------
        self : Gridded
            Returns the Gridded object with concatenated sondes.
        """
        if sortby is None:
            sortby = list(hh.l3_coords.keys())[0]
        list_of_l2_ds = [sonde.interim_l3_ds for sonde in self.sondes.values()]
        try:
            ds = xr.concat(
                list_of_l2_ds,
                dim=self.sonde_dim,
                join="exact",
                combine_attrs="drop_conflicts",
            ).sortby(sortby)
        except AttributeError:
            if coords is None:
                coords = hh.l3_coords
            for i, l2_ds in enumerate(list_of_l2_ds):
                missing_coords = set(coords.keys()) - set(l2_ds.coords)
                list_of_l2_ds[i] = l2_ds.assign_coords(
                    {
                        coord: (
                            (self.sonde_dim,),
                            np.full(ds.sizes[self.sonde_dim], np.nan),
                            coords[coord],
                        )
                        for coord in missing_coords
                    }
                )
            ds = xr.concat(
                list_of_l2_ds,
                dim=self.sonde_dim,
                join="exact",
                combine_attrs="drop_conflicts",
            ).sortby(sortby)

        self.concat_sonde_ds = ds
        return self

    def add_circles(self, circles: dict):
        """
        Add a dictionary of circles to the Gridded object.
        """
        self.circles = circles
        return self

    def concat_circles(self):
        count = []
        data = []
        circle_ids = []

        for circle_id, circle in self.circles.items():
            circle_ds = circle.circle_ds
            circle_ds = circle_ds.sortby("sonde_time")

            vars_sonde_dim = []
            vars_circle_dim = []

            for var in circle_ds.data_vars:
                if "sonde" in circle_ds[var].dims:
                    vars_sonde_dim.append(var)
                else:
                    vars_circle_dim.append(var)

            count.append(len(circle_ds.sonde_time))
            data.append(circle_ds)
            circle_ids.append(circle_id)

        concatenated_sonde_ds = xr.concat(
            [ds[vars_sonde_dim] for ds in data],
            dim="sonde",
            data_vars=vars_sonde_dim,
            coords="all",
            compat="override",
        )

        concatenated_circle_ds = xr.concat(
            [ds[vars_circle_dim] for ds in data],
            dim="circle",
            data_vars=vars_circle_dim,
            coords="all",
            compat="override",
        )

        sonde_ds_filtered = concatenated_sonde_ds[vars_sonde_dim]
        circle_ds_filtered = concatenated_circle_ds[vars_circle_dim]

        concatenated_ds = xr.merge(
            [sonde_ds_filtered, circle_ds_filtered], compat="override", join="outer"
        )
        concatenated_ds = concatenated_ds.assign(circle_id=("circle", circle_ids))

        concatenated_ds = concatenated_ds.set_coords(
            ["circle_time", "circle_lon", "circle_lat"]
        )
        concatenated_ds = concatenated_ds.reset_coords(
            ["aircraft_latitude", "aircraft_longitude", "aircraft_msl_altitude"]
        )
        concatenated_ds = concatenated_ds.assign_coords(
            sondes_per_circle=("circle", count)
        )

        concatenated_ds.sondes_per_circle.attrs["sample_dimension"] = "sonde"

        self._interim_l4_ds = concatenated_ds

        return self

    def get_l3_dir(self, l3_dir: str = None):
        """
        Determines the Level 3 directory for the Gridded object.

        Parameters
        ----------
        l3_dir : str, optional
            The Level 3 directory to set.

        Raises
        ------
        ValueError
            If no sondes and no Level 3 directory are provided.

        Returns
        -------
        self : Gridded
            Returns the Gridded object with the Level 3 directory set.
        """
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
        """
        Sets the Level 3 filename for the Gridded object.

        Parameters
        ----------
        l3_filename : str, optional
            The Level 3 filename to set.

        Returns
        -------
        self : Gridded
            Returns the Gridded object with the Level 3 filename set.
        """
        if l3_filename is None:
            l3_filename = hh.l3_filename
        else:
            l3_filename = l3_filename

        self.l3_filename = l3_filename

        return self

    def update_history_concat_l3(self):
        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" level3 concatenation with pydropsonde {__version__} \n"
        )
        self.history = history
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
        ds = self.concat_sonde_ds

        ds.attrs.update(
            history=self.history,
        )
        hx.write_ds(
            ds=self.concat_sonde_ds,
            dir=l3_dir,
            filename=self.l3_filename,
            object_dims=(self.sonde_dim,),
            alt_dim=self.alt_dim,
        )
        return self

    def add_l3_ds(self, l3_dir: str = None):
        """
        Adds the Level 3 dataset to the Gridded object.

        Parameters
        ----------
        l3_dir : str, optional
            The directory to load the Level 3 dataset from.

        Returns
        -------
        self : Gridded
            Returns the Gridded object with the Level 3 dataset added.
        """
        if l3_dir is None:
            self.set_l3_ds(self.concat_sonde_ds.copy())
        else:
            self.set_l3_ds(hx.open_dataset(l3_dir))
        return self

    def create_interim_l4(self):
        self.interim_l4_ds = self.l3_ds

        return self

    def add_distances(self):
        res = {}
        for var in ["u", "v", "p", "theta", "q", "rh", "ta"]:
            res[var] = hs.get_dist_to_nonan(
                self.interim_l4_ds, alt_dim=self.alt_dim, variable=var
            )
            res[var] = xr.where(res[var], res[var], 0)
            res[var].name = f"{var}_dist"
        self.distances = xr.merge(res.values(), join="exact").transpose(
            self.sonde_dim, self.alt_dim
        )
        return self

    def add_autocorrelation(
        self,
        autocorr_dir=None,
        filename="autocorrelation.zarr",
        maxalt=10000,
        variables=None,
    ):
        if autocorr_dir is None:
            autocorr_dir = self.l3_dir
        try:
            self.autocorrelation = hx.open_dataset(Path(autocorr_dir, filename))

        except FileNotFoundError:
            self.autocorrelation = hs.calc_autocorrelation(
                self.interim_l4_ds,
                alt_dim=self.alt_dim,
                maxalt=maxalt,
                variables=variables,
            )
            hx.write_ds(
                self.autocorrelation,
                dir=autocorr_dir,
                filename=filename,
                alt_dim=self.alt_dim,
                object_dims=("sonde",),
            )
        return self

    def get_autocorrelation_weight(self, variable):
        autocorrelation = self.autocorrelation
        dist = xr.where(self.distances, self.distances, 0)
        return xr.where(
            dist[f"{variable}_dist"] == 0,
            1,
            autocorrelation[f"{variable}_autocorr"]
            .sel(altitude=dist[f"{variable}_dist"], method="nearest")
            .values,
        )

    def get_no_change_weights(self, variable):
        one_array = xr.ones_like(self.interim_l4_ds[variable])
        one_array.name = f"{variable}_weights"
        return one_array

    def add_weights(self, method="autocorrelation"):
        w = []
        for var in ["u", "v", "p", "theta", "q", "rh", "ta"]:
            if method == "autocorrelation":
                weights = self.get_autocorrelation_weight(variable=var).transpose(
                    self.sonde_dim, self.alt_dim
                )
            elif method == "no_weights":
                weights = self.get_no_change_weights(variable=var)
            weights = weights.clip(0, 1)
            weights.name = f"{var}_weights"
            w.append(weights)

        self.weights = xr.merge(w, join="exact", compat="no_conflicts")
        self.interim_l4_ds = xr.merge(
            [self.interim_l4_ds, self.weights], join="exact", compat="no_conflicts"
        )
        return self

    def get_simple_circle_times_from_yaml(self, yaml_file: str = None):
        """
        Extracts circle times and related information from a simplified YAML file.
        This can be used for intermediated processing until the full flight
        segmentation is available

        Parameters:
        - yaml_file (str): The path to the YAML file containing flight information.

        Returns:
        - self: The instance of the class with updated attributes:
        - circle_times: List of tuples containing start and end times for each segment.
        - sonde_ids: List of sonde IDs for each segment.
        - segment_ids: List of segment IDs.
        - platform_ids: List of platform IDs.
        - flight_ids: List of flight IDs.
        """
        with open(yaml_file) as source:
            flightinfo = yaml.load(source, Loader=yaml.SafeLoader)
        segments = []
        for c in flightinfo["segments"]:
            segments.append(
                dict(
                    segment_id=c["segment_id"],
                    platform_id=flightinfo["platform"],
                    flight_id=flightinfo["flight_id"],
                    start=c["start"],
                    end=c["end"],
                )
            )
        self.segments = segments

        return self

    def get_circle_times_from_segmentation(self, yaml_file: str = None):
        if yaml_file is None:
            warnings.warn("No segmentation file provided. No circle analysis done")
            return None
        segmentation = rr.get_flight_segmentation(yaml_file)
        platform_ids = set(self.l3_ds.platform_id.values)
        flight_ids = set(self.l3_ds.flight_id.values)
        self.segments = sorted(
            [
                {
                    **s,
                    "platform_id": platform_id,
                    "flight_id": flight_id,
                }
                for platform_id in platform_ids
                for flight_id in flight_ids
                for s in segmentation.get(platform_id, {})
                .get(flight_id, {})
                .get("segments", [])
                if "circle" in s["kinds"]
            ],
            key=lambda s: s["start"],
        )

        return self

    def get_l4_dir(self, l4_dir: str = None):
        if l4_dir:
            self.l4_dir = l4_dir
        elif self.circles is not None:
            self.l4_dir = self.l3_dir.replace("Level_3", "Level_4")
        else:
            raise ValueError("No circles and no l4 directory given, cannot continue")
        return self

    def get_l4_filename(self, l4_filename: str = None):
        if l4_filename is None:
            l4_filename = hh.l4_filename
        else:
            l4_filename = l4_filename

        self.l4_filename = l4_filename
        return self

    def update_history_l4(self):
        history = getattr(self, "history", "")
        history = (
            history
            + datetime.now(timezone.utc).isoformat()
            + f" level4 computation with pydropsonde {__version__} \n"
        )
        self.history = history
        return self

    def write_l4(self, l4_dir: str = None, _interim_l4_ds: xr.Dataset = None):
        if l4_dir is None:
            l4_dir = self.l4_dir
        ds = self._interim_l4_ds
        ds.attrs = {}
        ds.attrs.update(
            self.global_attrs["global"],
        )
        ds.attrs.update(
            self.global_attrs["l4"],
        )
        ds.attrs.update(
            dict(
                history=self.history,
                title=self.global_attrs["l4"].get(
                    "title",
                    self.global_attrs.get("title", "Dropsonde Data") + " Level_4",
                ),
            )
        )
        hx.write_ds(
            ds=ds,
            dir=l4_dir,
            filename=self.l4_filename,
            object_dims=("sonde", "circle"),
            alt_dim=self.alt_dim,
        )
        return self
