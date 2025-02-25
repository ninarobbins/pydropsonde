import numpy as np
import warnings

import pydropsonde.helper.xarray_helper as hx


class QualityControl:
    """
    Helper class to handle quality control functions and flags in a sonde object
    """

    def __init__(
        self,
    ) -> None:
        self.qc_vars = {}
        self.qc_flags = {}
        self.qc_details = {}
        self.qc_by_var = {}
        self.alt_dim = "time"
        self.qc_ds = None

    def set_qc_variables(self, qc_variables):
        """
        set qc variables
        Parameters
        ----------
        qc_variables : dictionary of the form {<var>:<unit>}
        """
        self.qc_vars.update(qc_variables)

        for variable in qc_variables.keys():
            self.qc_by_var.update({variable: dict(qc_flags={}, qc_details={})})

    def set_qc_ds(self, ds):
        if "time" in ds.dims:
            self.qc_ds = ds.sortby("time")
        else:
            self.qc_ds = ds.sortby(self.alt_dim, ascending=False)

    def get_is_floater(
        self,
        gpsalt_threshold: float = 25,
        consecutive_time_steps: int = 3,
    ):
        """
        Add a qc flag, whether a sonde is a floater to a qc_flag object

        Parameters
        ----------
        gpsalt_threshold : float, optional
            The gpsalt altitude below which the sonde will check for time periods when gpsalt and pres have not changed. Default is 25.
        consecutive_time_steps : float, optional
            The number of timestapes that have to be at roughly  the same height and pressure to set landing time. default is 3

        Return
        ------
        Estimated landing time for floater or None
        """
        ds = self.qc_ds
        gpsalt_threshold = float(gpsalt_threshold)

        surface_ds = (
            ds.where(ds.gpsalt < gpsalt_threshold, drop=True)
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
            self.is_floater = True
        else:
            self.is_floater = False
            return None
        for time_index in range(len(floater) - consecutive_time_steps + 1):
            if np.all(floater[time_index : time_index + consecutive_time_steps]):
                landing_time = surface_ds.time[time_index - 1].values
                print(
                    f"{ds.attrs['SondeId']}: Floater detected! The landing time is estimated as {landing_time}."
                )
                return landing_time
        print(
            f"{ds.attrs['SondeId']}: Floater detected! However, the landing time could not be estimated. Therefore setting landing time as {surface_ds.time[0].values}"
        )
        return surface_ds.time[0].values

    def alt_below_aircraft(
        self,
        maxalt,
    ):
        """
        check if any measurements have been taken above the aircraft
        """
        ds = self.qc_ds
        self.qc_flags["altitude_below_aircraft"] = (
            np.nanmax(ds["gpsalt"].values) < maxalt
        )
        if not self.qc_flags["altitude_below_aircraft"]:
            variables = ["lat", "lon", "gpsalt", "u", "v"]

            self.set_qc_ds(
                hx.remove_above_alt(ds, variables, alt_dim="gpsalt", maxalt=maxalt)
            )

    def profile_extent(
        self,
        extent_min=8000,
    ):
        """
        Calculate the profile extent quality control flag and details for a given variable.

        This function calculates the altitude of the first measurement for a given variable and checks
        that this is above a given ratio.
        Adds the qc flag to the object.

        Parameters:
            self (object): The object containing the necessary attributes and methods.
            variable (str): The name of the variable being processed.
            ds (xarray.Dataset): The dataset containing the variable data.
            alt_dim (str): The dimension name of the altitude coordinate.
            extent_min (float): The minimum maximal height that is accepted for a good sonde

        Returns:
            None
        """
        ds = self.qc_ds
        alt_dim = self.alt_dim
        variables = self.qc_vars
        for variable in variables:
            no_na = ds.dropna(dim="time", subset=[variable])[alt_dim].values
            if no_na.size > 0:
                max_alt = np.nanmax(no_na)
            else:
                max_alt = np.nan

            self.qc_flags[f"{variable}_profile_extent"] = max_alt > extent_min
            self.qc_details[f"{variable}_profile_extent_max"] = max_alt

    def profile_sparsity(
        self,
        variable_dict={"u": 4, "v": 4, "rh": 2, "ta": 2, "p": 2},
        time_dimension="time",
        timestamp_frequency=4,
        sparsity_threshold=0.2,
    ):
        """
        Calculates the profile coverage for a given set of variables, considering their sampling frequency.

        This function assumes that the time_dimension coordinates are spaced over 0.25 seconds,
        implying a timestamp_frequency of 4 hertz. This is applicable for ASPEN-processed QC and PQC files,
        specifically for RD41.

        For each variable in the variable_dict that is in self.qc_vars, the function calculates the sparsity fraction. If the sparsity
        fraction is less than the sparsity_threshold, it sets the entry "profile_sparsity_{variable}" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in the variable_dict  that is in self.qc_vars, the function adds the sparsity fraction to the qc_details dictionary

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
        sparsity_threshold : float or str, optional
            The threshold for the sparsity fraction. If the calculated sparsity fraction is less than this threshold,
            the profile is considered not full. Default is 0.2.


        """
        ds = self.qc_ds.sortby(time_dimension)
        var_keys = set(variable_dict.keys())
        if set(var_keys) != set(self.qc_vars.keys()):
            var_keys = set(var_keys) & set(self.qc_vars.keys())
            warnings.warn(
                f"variables for which frequency is given do not match the qc_variables. Continue for the intersection  {var_keys}"
            )
        for variable in var_keys:
            min_valid_idx = ds[variable].notnull().argmax(dim=time_dimension).values
            dataset = ds[variable].isel(time=slice(min_valid_idx, None))
            sampling_frequency = variable_dict[variable]
            weighed_time_size = len(dataset[time_dimension]) / (
                timestamp_frequency / sampling_frequency
            )
            sparsity_fraction = 1 - (
                dataset.count(dim=time_dimension).values / weighed_time_size
            )
            self.qc_flags[f"{variable}_profile_sparsity"] = (
                sparsity_fraction < sparsity_threshold
            )
            self.qc_details[f"{variable}_profile_sparsity_fraction"] = sparsity_fraction

    def near_surface_coverage(
        self,
        alt_bounds=[0, 1000],
        count_threshold=50,
    ):
        """
        Calculates the fraction of non-null values in specified variables near the surface.


        For each variable in self.qc_variable, the function calculates the near surface count. If the near surface count is less than
        the count_threshold, it sets the entry "near_surface_{variable}" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in self.qc_vars, the function adds the near surface count to the qc_details dictionary


        Parameters
        ----------
        alt_bounds : list, optional
            The lower and upper bounds of altitude in meters to consider for the calculation. Defaults to [0,1000].
        alt_dim : str, optional
            The name of the altitude dimension. Defaults to "alt". If the sonde is a floater, this will be set to "gpsalt" regardless of user-provided value.
        count_threshold : int, optional
            The minimum count of non-null values required for a variable to be considered as having near surface coverage. Defaults to 50.


        """
        ds = self.qc_ds
        alt_dim = self.alt_dim

        count_threshold = int(count_threshold)

        if isinstance(alt_bounds, str):
            alt_bounds = alt_bounds.split(",")
            alt_bounds = [float(alt_bound) for alt_bound in alt_bounds]
        try:
            if self.is_floater and not (alt_dim == "gpsalt"):
                warnings.warn(
                    f"{ds.attrs['SondeId']} was detected as a floater but you did not chose gpsalt as altdim in the near surface coverage qc"
                )
        except KeyError:
            warnings.warn(
                f"{ds.attrs['SondeId']} has not been checked for being a floater. Please run is_floater first."
            )

        for variable in self.qc_vars.keys():
            if variable in ["u", "v"]:
                alt_dim = "gpsalt"
            else:
                alt_dim = "alt"
            dataset = ds.where(
                (ds[alt_dim] > alt_bounds[0]) & (ds[alt_dim] < alt_bounds[1]), drop=True
            )
            near_surface_count = dataset[variable].count()
            if near_surface_count < count_threshold:
                self.qc_flags[f"{variable}_near_surface"] = False

            else:
                self.qc_flags[f"{variable}_near_surface"] = True
            self.qc_details[f"{variable}_near_surface_count"] = (
                near_surface_count.values
            )

    def alt_near_gpsalt(self, diff_threshold=150):
        """
        Calculates the mean difference between msl altitude and gpsaltitude in the dataset

        For each variable in self.qc_variable, the function calculates the mean difference between altitude and gpsaltitude.
        If the difference is greater than the diff_threshold, it sets the entry "mean_alt_gpsalt_diff" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in self.qc_vars, the function adds the mean alt to gpsalt difference to the qc_details dictionary


        Parameters
        ----------
        diff_threshold : accepted difference between altitude and gpsaltitude. Default is 150m

        """
        ds = self.qc_ds
        dataset = ds[["alt", "gpsalt"]]
        if not self.qc_flags.get(f"{self.alt_dim}_values", True):
            return 0

        max_diff = np.abs((dataset.alt - dataset.gpsalt).max(skipna=True))
        if max_diff < diff_threshold:
            self.qc_flags["alt_near_gpsalt"] = True
        else:
            self.qc_flags["alt_near_gpsalt"] = False
        self.qc_details["alt_near_gpsalt_max_diff"] = max_diff.values

    def sfc_physics(
        self,
        min_vals={"rh": 0.3, "p": 100500, "ta": 293.15},
        max_vals={"rh": 1, "p": 102000, "ta": 310},
    ):
        """
        Checks that temperature, rh and p at the surface are within a certain range


        Parameters
        ----------
        self : object
        The object containing the necessary attributes and methods.
        alt_dim : str, optional
        The dimension name of the altitude coordinate (default is "gpsalt").

        Returns
        -------
        None
        """
        ds_check = self.qc_ds[["rh", "ta", "p"]].sortby("time", ascending=False)
        for var in ["p", "rh", "ta"]:
            if ds_check[var].dropna(dim="time").sizes["time"] == 0:
                self.qc_flags[f"{var}_sfc_physics"] = False
                self.qc_details[f"{var}_sfc_physics_val"] = np.nan
            else:
                sfc_var = ds_check[var].dropna(dim="time").values[0]
                self.qc_flags[f"{var}_sfc_physics"] = (sfc_var > min_vals[var]) and (
                    sfc_var < max_vals[var]
                )
                self.qc_details[f"{var}_sfc_physics_val"] = sfc_var

    def check_qc(self, used_flags=None, check_ugly=True):
        """
        check if any qc check has failed.
        If any has failed, return False, if not True

        Parameters:
        -----------
        used_flags: string or list
            list of qc flags to check
        """
        if used_flags is None:
            used_flags = []
        elif used_flags == "all":
            used_flags = list(self.qc_flags.keys()).copy()
        elif isinstance(used_flags, str):
            used_flags = used_flags.split(",")
            if (len(used_flags) == 1) and used_flags[0].startswith("all_except_"):
                all_flags = self.qc_flags.copy()
                all_flags.pop(used_flags[0].replace("all_except_", ""))
                used_flags = all_flags.copy()
            elif used_flags[0].startswith("all_except_"):
                raise ValueError(
                    "If 'all_except_<prefix>' is provided in filter_flags, it should be the only value."
                )
        if not all(flag in self.qc_flags for flag in used_flags):
            raise ValueError(
                "not all flags are in the qc dict. please check you ran all qc tests"
            )

        used_flags = {key: self.qc_flags[key] for key in used_flags}
        if check_ugly and all(used_flags.values()):
            return True
        elif (not check_ugly) and any(used_flags.values()):
            return True
        else:
            return False

    def get_qc_by_var(self):
        """
        Organizes quality control (QC) flags and details by variable.

        This method iterates over each variable in `self.qc_vars` and filters the
        `self.qc_flags` and `self.qc_details` dictionaries to include only the keys
        that are associated with the current variable. The keys are identified by
        checking if they contain the variable name as a prefix, followed by an
        underscore. The filtered dictionaries are then stored in `self.qc_flags`
        and `self.qc_details` under the corresponding variable name.

        Attributes:
            self.qc_vars (list): A list of variable names to filter QC data by.
            self.qc_flags (dict): A dictionary containing QC flags, which will be
                filtered and organized by variable.
            self.qc_details (dict): A dictionary containing QC details, which will
                be filtered and organized by variable.

        """
        for variable in self.qc_vars.keys():
            self.qc_by_var[variable]["qc_flags"].update(
                {
                    key: self.qc_flags.get(key)
                    for key in list(self.qc_flags.keys())
                    if f"{variable}_" in key
                }
            )
            self.qc_by_var[variable]["qc_details"].update(
                {
                    key: self.qc_details.get(key)
                    for key in list(self.qc_details.keys())
                    if f"{variable}_" in key
                }
            )

    def get_byte_array(self, variable):
        """
        Generate a byte array and associated attributes for a given variable's quality control flags.

        This function checks if quality control flags for the specified variable are available.
        If not, it retrieves them. It then calculates a byte value representing the quality control
        status by iterating over the flags and their values. Additionally, it constructs a dictionary
        of attributes that describe the quality control flags.

        Parameters:
        - variable (str): The name of the variable for which to generate the byte array and attributes.

        Returns:
        - tuple: A tuple containing:
            - np.byte: The calculated byte value representing the quality control status.
            - dict: A dictionary of attributes with the following keys:
                - long_name (str): A descriptive name for the quality control of the variable.
                - standard_name (str): A standard name indicating the type of flag.
                - flag_masks (str): A comma-separated string of binary masks for each flag.
                - flag_meanings (str): A comma-separated string of the meanings of each flag.
        """
        if not self.qc_by_var.get(variable, {}).get("qc_flags"):
            self.get_qc_by_var()
        qc_val = np.byte(0)
        keys = []
        for i, (key, value) in enumerate(
            self.qc_by_var.get(variable).get("qc_flags").items()
        ):
            qc_val = qc_val + (2**i) * np.byte(not value)
            keys.append(key.split("_", 1)[1])
        if qc_val == 0:
            qc_status = "GOOD"
        elif qc_val == 2 ** (i + 1) - 1:
            qc_status = "BAD"
        else:
            qc_status = "UGLY"
        attrs = dict(
            long_name=f"qc for {variable}",
            standard_name="status_flag",
            flag_masks=", ".join([f"{2**x}b" for x in range(i + 1)]),
            flag_meanings=", ".join(keys),
            description="if non-zero, this sonde should be used with care.",
            qc_status=qc_status,
        )
        return np.byte(qc_val), attrs

    def get_unit_for_qc(self, qc_name, var_name=None):
        """
        get the correct unit for the detailed qc value. Depends on the last bit of the qc detail name
        """
        var_unit = self.qc_vars[var_name]
        if (
            (qc_name.endswith("diff"))
            or (qc_name.endswith("min"))
            or (qc_name.endswith("max"))
            or (qc_name.endswith("sfc"))
            or (qc_name.endswith("val"))
        ):
            return var_unit
        elif (
            qc_name.endswith("count")
            or qc_name.endswith("fraction")
            or qc_name.endswith("ratio")
        ):
            return "1"
        else:
            warnings.warn("qc ending not specified. can't return a unit.")

    def get_details_var(self, variable):
        """
        Retrieve quality control details and attributes for a specified variable.

        This method checks if the quality control (QC) details for the given variable are available. If not, it invokes the `get_qc_by_var` method to populate them. It then constructs a dictionary of attributes for each QC key associated with the variable, providing a descriptive long name and units.

        Parameters:
            variable (str): The name of the variable for which QC details are to be retrieved.

        Returns:
            tuple: A tuple containing:
                - dict: The QC details for the specified variable.
                - dict: A dictionary of attributes for each QC key, including:
                    - long_name (str): A descriptive name for the QC key.
                    - units (str): The units for the QC key, defaulted to "1".
        """
        if self.qc_by_var.get(variable, {}).get("qc_details") is not None:
            self.get_qc_by_var()
        attrs = {}
        for key in list(self.qc_by_var.get(variable).get("qc_details").keys()):
            name = key.split("_", 1)[1]
            attrs.update(
                {
                    key: dict(
                        long_name=f"value for qc  {variable} " + name.replace("_", " "),
                        units=self.get_unit_for_qc(key, variable),
                    )
                }
            )
        return self.qc_by_var.get(variable).get("qc_details"), attrs

    def add_variable_flags_to_ds(self, ds, variable, add_to=None, details=True):
        if add_to is None:
            add_to = variable
        name = f"{variable}_qc"
        value, attrs = self.get_byte_array(variable)
        ds = ds.assign({name: value})
        ds[name].attrs.update(attrs)
        ds = hx.add_ancillary_var(ds, add_to, name)
        # get detail
        if details:
            qc_dict, attrs = self.get_details_var(variable)
            for key in list(qc_dict.keys()):
                ds = ds.assign({key: qc_dict.get(key)})
                ds[key].attrs.update(attrs.get(key))
                ds = hx.add_ancillary_var(ds, add_to, key)

        return ds

    def add_alt_near_gpsalt_to_ds(self, ds):
        if self.qc_flags.get("alt_near_gpsalt") is not None:
            ds = ds.assign(
                {"alt_near_gpsalt": np.byte(not self.qc_flags.get("alt_near_gpsalt"))}
            )
            ds["alt_near_gpsalt"].attrs.update(
                dict(
                    long_name="maximal difference between alt and gpsalt",
                    flag_values="0 1 ",
                    flag_meaning="GOOD BAD",
                )
            )

            ds = ds.assign(
                {
                    "alt_near_gpsalt_max_diff": self.qc_details.get(
                        "alt_near_gpsalt_max_diff"
                    )
                }
            )
            ds["alt_near_gpsalt_max_diff"].attrs.update(
                dict(
                    long_name="maximal difference between alt and gpsalt",
                    units="m",
                )
            )

            ds = hx.add_ancillary_var(
                ds, self.alt_dim, "alt_near_gpsalt alt_near_gpsalt_max_diff"
            )
        return ds

    def add_below_aircraft_to_ds(self, ds):
        """
        add quality flag whether any measurement is above aircraft alt to ds
        """
        alt_dim = self.alt_dim
        ds = ds.assign(
            {
                f"{alt_dim}_below_aircraft": np.byte(
                    not self.qc_flags.get(f"{alt_dim}_below_aircraft")
                )
            }
        )
        ds[f"{alt_dim}_below_aircraft"].attrs.update(
            dict(
                long_name=f"qc heighest {alt_dim} measurement below aircraft",
                flag_values="0 1 ",
                flag_meaning="GOOD BAD",
            )
        )

        ds = hx.add_ancillary_var(ds, self.alt_dim, f"{alt_dim}_below_aircraft")
        return ds

    def add_alt_source_to_ds(self, ds):
        """
        Adds  an ancillary variable in the dataset for the altitude dimension.

        This function assigns a new variable to the dataset `ds` with a name based on
        the `alt_dim` attribute of the class. The new variable indicates whether values
        for the specified dimension are present in the raw data, using quality control
        flags. It updates the attributes of the new variable to include a long name,
        flag values, and flag meanings.

        Parameters:
        - ds: The dataset to which the ancillary variable will be added or replaced.

        Returns:
        - The updated dataset with the ancillary variable added or replaced.
        """
        ds = ds.assign(
            {f"{self.alt_dim}_source": self.qc_flags.get(f"{self.alt_dim}_source")}
        )
        ds[f"{self.alt_dim}_source"].attrs.update(
            dict(
                long_name=f"raw data dimension {self.alt_dim} is derived from",
                flag_values="alt gpsalt",
            )
        )

        ds = hx.add_ancillary_var(ds, self.alt_dim, f"{self.alt_dim}_source")
        return ds

    def add_non_var_qc_to_ds(self, ds):
        """
        Adds non-variable quality control (QC) data to the given dataset.

        This method performs the following operations on the input dataset `ds`:
        1. Adds altitude near GPS altitude to the dataset using the `add_alt_near_gpsalt_to_ds` method.
        2. Replaces altitude variable in the dataset using the `add_alt_source_to_ds` method.

        Parameters:
        - ds: The input dataset to which non-variable QC data will be added.

        Returns:
        - ds_out: The output dataset with added non-variable QC data.
        """
        ds_out = self.add_alt_near_gpsalt_to_ds(ds)
        ds_out = self.add_below_aircraft_to_ds(ds_out)

        return ds_out

    def add_sonde_flag_to_ds(self, ds, qc_name):
        flags = self.qc_flags.copy()
        flags.pop("altitude_below_aircraft", None)
        flags.pop("alt_below_aircraft", None)
        flags.pop("gpsalt_below_aircraft", None)
        if all(flags.values()):
            qc_val = 0
        elif any(flags.values()):
            flags.pop("p_sfc_physics", None)
            flags.pop("rh_sfc_physics", None)
            flags.pop("ta_sfc_physics", None)
            if all(flags.values()):
                qc_val = 1
            else:
                flags.pop("alt_near_gpsalt", None)
                if any(flags.values()):
                    qc_val = 2
                else:
                    qc_val = 1
        else:
            qc_val = 1

        ds = ds.assign({qc_name: qc_val})
        ds[qc_name].attrs.update(
            dict(
                standard_name="aggregate_quality_flag",
                long_name="aggregated quality flag for sonde",
                flag_values="0 1 2",
                flag_meaning="GOOD BAD UGLY",
                description="if not 0 some quality control has not been passed. Handle with care.",
            )
        )
        ds = hx.add_ancillary_var(ds, "sonde_id", qc_name)

        return ds
