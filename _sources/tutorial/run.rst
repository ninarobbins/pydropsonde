Running pydropsonde
===================

To run ``pydropsonde`` on a set of sonde profiles, you first need to create a configuration file (see :doc:`Config Files <configs>`).
Further, you can provide the path to the config file through the ``-c`` option.
By default, ``pydropsonde`` would search for `dropsonde.cfg` in the current directory.

.. code-block:: bash

        pydropsonde -c <path_to_config_file>

.. note::
   The processing from Level_0 to Level_1 using `Aspen <https://www.eol.ucar.edu/content/aspen>`_ is included in ``pydropsonde``.
   It makes use of a docker image containing the Aspen software.
   If you are unfamiliar with `docker`, have a look at the :doc:`../data/level1` description to learn how to install `docker` and start a `docker daemon`.

Using pydropsonde as a package
==============================

Sometimes it is not necessary to run the full pydropsonde processing pipeline. In that case, ``pydropsonde`` can be used as a regular Python package.
Here, we give an example of how to use ``pydropsonde`` to process from a Level 3 dataset to Level 4.

.. code-block:: python

        import xarray as xr

        from pydropsonde.processor import Gridded
        from pydropsonde import pipeline
        from pydropsonde.circles import Circle
        root = "QmTpimQBT8AngwDPRTWYqCXctbhqMiP3NiPeEfXt3JomuU"
        l3_ds = xr.open_dataset(
                f"ipfs://{root}/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
        engine="zarr",
        )

Then, the ``Gridded`` object used in ``pydropsonde`` has to be created and the circle times have to be added (in this case from the flight flight segmentation file).

.. code-block:: python

        gridded = Gridded(sondes={}, global_attrs={})
        gridded.set_l3_ds(l3_ds)
        gridded.get_circle_times_from_segmentation(
                "https://orcestra-campaign.github.io/flight_segmentation/all_flights.yaml"
        )
        gridded.alt_dim = "altitude"
        gridded.sonde_dim = "sonde"
        gridded.create_interim_l4()

Using the segment times, a dictionary with the circles can be created.

.. code-block:: python

        circles = pipeline.create_and_populate_circle_object(gridded, None).circles

Once this is done, the Level 4 processing can be conducted step by step.

.. code-block:: python

        def iterate_function_over_circles(circles, function, **kwargs):
                for id, circle in circles.items():
                        function(circle, **kwargs)

        iterate_function_over_circles(circles, Circle.get_xy_coords_for_circles)
        iterate_function_over_circles(circles, Circle.drop_vars)
        iterate_function_over_circles(circles, Circle.interpolate_na_sondes, method="akima")
        iterate_function_over_circles(circles, Circle.extrapolate_na_sondes, max_alt=300)
        iterate_function_over_circles(circles, Circle.apply_fit2d)
        iterate_function_over_circles(circles, Circle.add_divergence)
        iterate_function_over_circles(circles, Circle.add_vorticity)
        iterate_function_over_circles(circles, Circle.add_omega)
        iterate_function_over_circles(circles, Circle.add_wvel)
        iterate_function_over_circles(circles, Circle.add_circle_id_variable)
        iterate_function_over_circles(circles, Circle.drop_latlon)
        iterate_function_over_circles(circles, Circle.get_circle_flight_id)
        iterate_function_over_circles(circles, Circle.add_regression_stderr)
        iterate_function_over_circles(circles, Circle.add_circle_variables_to_ds)

.. warning::
        The pydropsonde functions that calculate the circle products modify the ``Circle`` objects in place and some of the above functions delete variables.
        Depending on what you are doing, copying before the processing steps might be necessary.

Finally, the circles can be concatenated and saved to a zarr file.

.. code-block:: python

        gridded.concat_circles()
        gridded.get_l4_dir(".")
        gridded.get_l4_filename("test_l4.zarr")
        gridded.update_history_l4()
        gridded.global_attrs = {
                "global": {"author": "it's me"},
                "l4": {"title": "this is a test"},
        }
        gridded.write_l4()
