
Level 3 (default)
=================

The Level 3 dataset contains all sondes from Level 2 on a uniform vertical grid. The default vertical coordinate is the MSL altitude ``alt``.

Change vertical dimension
-------------------------

The default-pipeline allows a change of the vertical coordinate. For example, to change the vertical coordinate to ``gpsalt``, the following lines have to be added to the config file

.. code-block:: ini

        [processor.Sonde.remove_non_mono_incr_alt]
        alt_var=gpsalt

        [processor.Sonde.interpolate_alt]
        alt_var=gpsalt

        [processor.Sonde.get_N_m_values]
        alt_var=gpsalt

        [processor.Sonde.save_interim_l3]
        alt_dim=gpsalt

        [processor.Gridded.write_l3]
        alt_dim=gpsalt
