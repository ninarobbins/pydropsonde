Level 2 (default)
=================


Quality Control
---------------

Between `Level_1` (product of the ASPEN software) and `Level_2` quality control checks can be applied to
remove sondes that don't fullfill a given standard.

.. warning::

    Per default NO quality control is applied. If quality control is required for the dataset, the functions have to be added to the `config_file`!

Currently, there are three quality control checks implemented that can be used:

1. `profile_fullness`: check if the coverage of each profiles is above a certain threshold (wrt their normal data frequency)
2. `near_surface_coverage`: check if the fraction of data in the bottom 1000m is above a certain threshold
3. `alt_near_gpsalt`: check that the `gpsalt` and `alt` variable after the ASPEN processing don't differ by more than 150m. If they do, the sonde most likely stopped meassuring before reaching the ground.

.. danger::

    If neither test 2 nor 3 are applied, sondes that did not meassure in the lower atmosphere are included in `Level_2` and the
    bottom most `alt` is set to zero, i.e., using `alt` as the height coordinate later on will lead to errors in the dataset.

To add tests to the config_file, a section has to be added:

.. code-block:: bash

    [processor.Sonde.filter_qc_fail]
    filter_flags = <qc1>,<qc2>,<qc3>
