import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from . import physics
import xarray as xr

# Keys in l2_variables should be variable names in aspen_ds attribute of Sonde object
l2_variables = {
    "u_wind": {
        "rename_to": "u",
        "attributes": {
            "standard_name": "eastward_wind",
            "long_name": "u component of winds",
            "units": "m s-1",
            "coordinates": "time lon lat alt",
        },
    },
    "v_wind": {
        "rename_to": "v",
        "attributes": {
            "standard_name": "northward_wind",
            "long_name": "v component of winds",
            "units": "m s-1",
            "coordinates": "time lon lat alt",
        },
    },
    "tdry": {
        "rename_to": "ta",
        "attributes": {
            "standard_name": "air_temperature",
            "long_name": "air temperature",
            "units": "K",
            "coordinates": "time lon lat alt",
        },
    },
    "pres": {
        "rename_to": "p",
        "attributes": {
            "standard_name": "air_pressure",
            "long_name": "atmospheric pressure",
            "units": "Pa",
            "coordinates": "time lon lat alt",
        },
    },
    "rh": {
        "attributes": {
            "standard_name": "relative_humidity",
            "long_name": "relative humidity",
            "units": "",
            "coordinates": "time lon lat alt",
        }
    },
    "lat": {
        "attributes": {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degree_north",
            "axis": "Y",
        }
    },
    "lon": {
        "attributes": {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degree_east",
            "axis": "X",
        }
    },
    "time": {
        "attributes": {
            "standard_name": "time",
            "long_name": "time of recorded measurement",
            "axis": "T",
        }
    },
    "gpsalt": {
        "attributes": {
            "standard_name": "altitude",
            "long_name": "gps reported altitude above MSL",
            "units": "m",
            "axis": "Z",
            "positive": "up",
        }
    },
    "alt": {
        "attributes": {
            "standard_name": "altitude",
            "long_name": "altitude above MSL",
            "units": "m",
            "axis": "Z",
            "positive": "up",
        }
    },
}

encoding_variables = {
    "time": {"units": "seconds since 1970-01-01", "dtype": "float"},
}

variable_compression_properties = dict(
    zlib=True,
    complevel=4,
    fletcher32=True,
    _FillValue=np.finfo("float32").max,
)


l2_flight_attributes_map = {
    "True Air Speed (m/s)": "true_air_speed_(ms-1)",
    "Ground Speed (m/s)": "ground_speed_(ms-1)",
    "Software Notes": "AVAPS_software_notes",
    "Format Notes": "AVAPS_format_notes",
    "True Heading (deg)": "true_heading_(deg)",
    "Ground Track (deg)": "ground_track_(deg)",
    "Longitude (deg)": "aircraft_longitude_(deg_E)",
    "Latitude (deg)": "aircraft_latitude_(deg_N)",
    "MSL Altitude (m)": "aircraft_msl_altitude_(m)",
    "Geopotential Altitude (m)": "aircraft_geopotential_altitude_(m)",
}
path_to_flight_ids = "{platform}/Level_0"
path_to_l0_files = "{platform}/Level_0/{flight_id}"

l2_filename_template = "{platform}_{launch_time}_{flight_id}_{serial_id}_Level_2.nc"

l3_filename = "Level_3.nc"


def get_chunks(ds, var):
    chunks = {
        "sonde_id": min(256, ds.sonde_id.size),
        "alt": min(350, ds.alt.size),
    }

    return tuple((chunks[d] for d in ds[var].dims))


def get_encoding(ds, exclude_vars=None):
    variables = [
        "u",
        "v",
        "ta",
        "p",
        "rh",
        "lat",
        "lon",
        "gpsalt",
        "alt",
        "sonde_id",
        "q",
        "iwv",
        "w_dir",
        "w_spd",
    ]
    if exclude_vars is None:
        exclude_vars = []

    enc_var = {
        var: {
            "compression": "zlib",
            "dtype": "float32",
            "chunksizes": get_chunks(ds, var),
        }
        for var in variables
        if var not in ds.dims
        if var not in exclude_vars
    }
    enc_time = {
        var: {
            "compression": "zlib",
            "chunksizes": get_chunks(ds, var),
            "_FillValue": np.datetime64("NaT"),
        }
        for var in ["interp_time", "launch_time"]
    }
    enc_var.update(enc_time)

    enc_attr = {
        var: {
            "compression": "zlib",
            "chunksizes": get_chunks(ds, var),
            "dtype": "float32",
        }
        for var in ds.variables
        if var not in ds.dims
        if var not in variables
        if var not in ["interp_time", "launch_time"]
        if ds[var].dtype == "float64"
    }
    enc_var.update(enc_attr)
    return enc_var


def get_bool(s):
    if isinstance(s, bool):
        return s
    elif isinstance(s, int):
        return bool(s)
    elif isinstance(s, str):
        lower_s = s.lower()
        if lower_s == "true":
            return True
        elif lower_s == "false":
            return False
        elif lower_s in ["0", "1"]:
            return bool(int(lower_s))
        else:
            raise ValueError(f"Cannot convert {s} to boolean")
    else:
        raise ValueError(f"Cannot convert {s} to boolean")


def convert_rh_to_si(value):
    """convert RH from % to fraction"""
    return value / 100


def convert_pres_to_si(value):
    """convert pressure from hPa to Pa"""
    return value * 100


def convert_tdry_to_si(value):
    """convert temperature from C to K"""
    return value + 273.15


def get_si_converter_function_based_on_var(var_name):
    """get the function to convert a variable to SI units based on its name"""
    func_name = f"convert_{var_name}_to_si"
    func = globals().get(func_name, None)
    if func is None:
        raise ValueError(f"No function named {func_name} found in the module")
    return func


def calc_saturation_pressure(temperature_K, method="hardy1998"):
    """
    Calculate saturation water vapor pressure

    Input
    -----
    temperature_K : array
        array of temperature in Kevlin or dew point temperature for actual vapor pressure
    method : str
        Formula used for calculating the saturation pressure
            'hardy1998' : ITS-90 Formulations for Vapor Pressure, Frostpoint Temperature,
                Dewpoint Temperature, and Enhancement Factors in the Range –100 to +100 C,
                Bob Hardy, Proceedings of the Third International Symposium on Humidity and Moisture,
                1998 (same as used in Aspen software after May 2018)

    Return
    ------
    e_sw : array
        saturation pressure (Pa)

    Examples
    --------
    >>> calc_saturation_pressure([273.15])
    array([ 611.2129107])

    >>> calc_saturation_pressure([273.15, 293.15, 253.15])
    array([  611.2129107 ,  2339.26239586,   125.58350529])
    """

    if method == "hardy1998":
        g = np.empty(8)
        g[0] = -2.8365744 * 10**3
        g[1] = -6.028076559 * 10**3
        g[2] = 1.954263612 * 10**1
        g[3] = -2.737830188 * 10 ** (-2)
        g[4] = 1.6261698 * 10 ** (-5)
        g[5] = 7.0229056 * 10 ** (-10)
        g[6] = -1.8680009 * 10 ** (-13)
        g[7] = 2.7150305

        e_sw = np.zeros_like(temperature_K)

        for t, temp in enumerate(temperature_K):
            ln_e_sw = np.sum([g[i] * temp ** (i - 2) for i in range(0, 7)]) + g[
                7
            ] * np.log(temp)
            e_sw[t] = np.exp(ln_e_sw)
        return e_sw


def calc_q_from_rh_sondes(ds):
    """
    Input :

        ds : Dataset

    Output :

        q : Specific humidity values

    Function to estimate specific humidity from the relative humidity, temperature and pressure in the given dataset.
    """
    e_s = calc_saturation_pressure(ds.ta.values)
    w_s = mpcalc.mixing_ratio(e_s * units.Pa, ds.p.values * units.Pa).magnitude
    w = ds.rh.values * w_s
    q = w / (1 + w)
    ds = ds.assign(q=(ds.rh.dims, q))
    ds["q"].attrs = dict(
        standard_name="specific humidity",
        long_name="specific humidity",
        units="kg/kg ",
    )
    return ds


def calc_q_from_rh(ds):
    """
    Input :

        ds : Dataset

    Output :

        ds : Dataset with q added

    Function to estimate specific humidity from the relative humidity, temperature and pressure in the given dataset.
    """
    vmr = physics.relative_humidity2vmr(
        RH=ds.rh.values,
        p=ds.p.values,
        T=ds.ta.values,
        e_eq=physics.e_eq_mixed_mk,
    )

    q = physics.vmr2specific_humidity(vmr)
    ds = ds.assign(q=(ds.ta.dims, q))
    ds["q"].attrs = dict(
        standard_name="specific humidity",
        long_name="specific humidity",
        units="kg/kg ",
    )
    return ds


def calc_rh_from_q(ds):
    vmr = physics.specific_humidity2vmr(q=ds.q.values)
    rh = physics.vmr2relative_humidity(
        vmr=vmr, p=ds.p.values, T=ds.ta.values, e_eq=physics.e_eq_mixed_mk
    )
    ds = ds.assign(rh=(ds.q.dims, rh))
    ds["rh"].attrs = dict(
        standard_name="relative humidity",
        long_name="relative humidity",
        units="",
        method="water until 0degC, ice below -23degC, mixed between",
    )
    return ds


def calc_iwv(ds, sonde_dim="sonde_id", alt_dim="alt"):
    pressure = ds.p.values
    temperature = ds.ta.values
    alt = ds[alt_dim].values

    vmr = physics.specific_humidity2vmr(
        q=ds.q.values,
    )
    mask_p = ~np.isnan(pressure)
    mask_t = ~np.isnan(temperature)
    mask_vmr = ~np.isnan(vmr)
    mask = mask_p & mask_t & mask_vmr
    iwv = physics.integrate_water_vapor(
        vmr[mask], pressure[mask], T=temperature[mask], z=alt[mask]
    )
    ds_iwv = xr.DataArray([iwv], dims=[sonde_dim], coords={})
    ds_iwv.name = "iwv"
    ds_iwv.attrs = {"standard name": "integrated water vapor", "units": "kg/m^2"}
    ds = xr.merge([ds, ds_iwv])
    return ds


def calc_theta_from_T(ds):
    """
    Input :

        dataset : Dataset

    Output :

        theta : Potential temperature values

    Function to estimate potential temperature from the temperature and pressure in the given dataset.
    """
    theta = mpcalc.potential_temperature(
        ds.p.values * units(ds.p.attrs["units"]),
        ds.ta.values * units(ds.ta.attrs["units"]),
    )
    ds = ds.assign(theta=(ds.ta.dims, theta.magnitude))
    ds["theta"].attrs = dict(
        standard_name="air_potential_temperature",
        long_name="potential temperature",
        units=str(theta.units),
    )

    return ds


def calc_T_from_theta(ds):
    """
    Input :

        dataset : Dataset

    Output :

        theta : Potential temperature values

    Function to estimate potential temperature from the temperature and pressure in the given dataset.
    """
    ta = mpcalc.temperature_from_potential_temperature(
        ds.p.values * units(ds.p.attrs["units"]),
        ds.theta.values * units(ds.theta.attrs["units"]),
    )
    ds = ds.assign(ta=(ds.ta.dims, ta.magnitude))
    ds["ta"].attrs = dict(
        standard_name="air_temperature",
        long_name="air temperature",
        units=str(ta.units),
    )
    return ds


def calc_theta_e(ds):
    dewpoint = mpcalc.dewpoint_from_specific_humidity(
        pressure=ds.p.values * units(ds.p.attrs["units"]),
        temperature=ds.ta.values * units(ds.ta.attrs["units"]),
        specific_humidity=ds.q.values * units(ds.q.attrs["units"]),
    )
    theta_e = mpcalc.equivalent_potential_temperature(
        pressure=ds.p.values * units(ds.p.attrs["units"]),
        temperature=ds.ta.values * units(ds.ta.attrs["units"]),
        dewpoint=dewpoint,
    )

    ds = ds.assign(
        theta_e=(
            ds.ta.dims,
            theta_e.magnitude,
            dict(
                standard_name="air_equivalent_potential_temperature",
                long_name="equivalent potential temperature",
                units=str(theta_e.units),
            ),
        )
    )
    return ds


def calc_T_v(ds):
    mr = mpcalc.mixing_ratio_from_specific_humidity(
        ds.q.values * units(ds.q.attrs["units"])
    )

    tv = mpcalc.virtual_temperature(
        temperature=ds.ta.values * units(ds.ta.attrs["units"]),
        mixing_ratio=mr,
    )
    ds = ds.assign(
        tv=(
            ds.ta.dims,
            tv.magnitude,
            dict(
                standard_name="virtual_temperature",
                long_name="virtual temperature",
                units=str(tv.units),
            ),
        )
    )
    return ds


def calc_theta_v(ds):
    mr = mpcalc.mixing_ratio_from_specific_humidity(
        ds.q.values * units(ds.q.attrs["units"])
    )

    theta_v = mpcalc.virtual_potential_temperature(
        pressure=ds.p.values * units(ds.p.attrs["units"]),
        temperature=ds.ta.values * units(ds.ta.attrs["units"]),
        mixing_ratio=mr,
    )
    ds = ds.assign(
        theta_v=(
            ds.ta.dims,
            theta_v.magnitude,
            dict(
                # standard_name="", to be added when official
                long_name="virtual potential temperature",
                units=str(theta_v.units),
            ),
        )
    )
    return ds


def calc_wind_dir_and_speed(ds):
    """
    Calculates wind direction between 0 and 360 according to https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398

    """
    w_dir = (180 + np.arctan2(ds.u.values, ds.v.values) * 180 / np.pi) % 360
    w_spd = np.sqrt(ds.u.values**2 + ds.v.values**2)

    ds = ds.assign(
        w_dir=(
            ds.u.dims,
            w_dir,
            dict(
                standard_name="wind_from_direction",
                units="degree",
            ),
        )
    )

    ds = ds.assign(
        w_spd=(
            ds.u.dims,
            w_spd,
            dict(
                standard_name="wind_speed",
                units="m s-1",
            ),
        )
    )
    return ds
