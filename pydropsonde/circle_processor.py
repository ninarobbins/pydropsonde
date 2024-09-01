@dataclass(order=True)
class Circle:
    """Class identifying a circle and containing its metadata.

    A `Circle` identifies the circle averaged products from all sondes within the circle.

    Every `Circle` mandatorily has a `circle` identifier in the format "HALO-240811-c1".
    """

    level3_ds: xr.Dataset
    circle: str
    flight_id: str
    platform_id: str

    def dim_ready_ds(self):

        dims_to_drop = ["sounding"]

        all_sondes = level3_ds.swap_dims({"sounding": "sonde_id"}).drop(dims_to_drop)

        self.all_sondes = all_sondes
        return self

    def reswap_launchtime_sounding(self):

        for circle in self.circles:
            circle["sounding"] = (
                ["launch_time"],
                np.arange(1, len(circle.launch_time) + 1, 1),
            )
            circle = circle.swap_dims({"launch_time": "sounding"})

        return self

    def get_xy_coords_for_circle(self):

        x_coor = (
            self.circle["lon"]
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
        circle_x = np.nanmean(c_xc) / (111.320 * np.cos(np.radians(circle_y)) * 1000)

        circle_diameter = np.nanmean(c_r) * 2

        xc = np.mean(x_coor, axis=0)
        yc = np.mean(y_coor, axis=0)

        delta_x = x_coor - xc  # difference of sonde long from mean long
        delta_y = y_coor - yc  # difference of sonde lat from mean lat

        self.circle["platform_id"] = self.circle.platform_id.values[0]
        self.circle["flight_altitude"] = self.circle.flight_altitude.mean().values
        self.circle["circle_time"] = self.circle.launch_time.mean().values.astype(
            "datetime64"
        )
        self.circle["circle_lon"] = circle_x
        self.circle["circle_lat"] = circle_y
        self.circle["circle_diameter"] = circle_diameter
        self.circle["dx"] = (["sounding", "alt"], delta_x)
        self.circle["dy"] = (["sounding", "alt"], delta_y)

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

    def get_div_and_vor(self):

        D = self.dudx + self.dvdy
        vor = self.dvdx - self.dudy

        self.D = (["circle", "alt"], D)
        self.vor = (["circle", "alt"], vor)

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

        self["density"] = (["sounding", "alt"], den_m)
        self["mean_density"] = (["alt"], np.nanmean(den_m, axis=0))

        D = self.D.values

        nan_ids = np.where(np.isnan(D))

        w_vel = np.full(len(self.alt), np.nan)
        p_vel = np.full(len(self.alt), np.nan)

        w_vel[0] = 0

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
                w_vel[nan_ids[0][ids_for_nan_ids], nan_ids[1][ids_for_nan_ids]] = np.nan

            else:
                w_vel[m] = w_vel[last] - self.D.isel(alt=m).values * 10 * (m - last)
                last = m

            for n in range(1, len(self.alt)):

                p_vel[n] = -self.mean_density.isel(alt=n) * 9.81 * w_vel[n]

        self.W = (["alt"], w_vel)
        self.omega = (["alt"], p_vel)

        return self

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

    def get_l4_dir(self, l4_dir: str = None):
        if l4_dir:
            self.l4_dir = l4_dir
        elif level3_ds is not None:
            self.l4_dir = list(level3_ds.values())[0].l4_dir
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
