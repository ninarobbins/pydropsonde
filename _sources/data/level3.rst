
Level 3 (default)
=================

The Level 3 dataset contains all sondes from Level 2 on a uniform vertical grid. The default vertical coordinate is the MSL altitude ``alt``.

Change vertical dimension
-------------------------

The default-pipeline allows a change of the vertical coordinate. For example, to change the vertical coordinate to ``gpsalt``, the following lines have to be added to the config file

.. code-block:: ini

        [processor.Sonde.set_alt_dim]
        alt_dim=gpsalt

If the altitude dimension is replaced, all sondes that do not have values in that dimension are dropped. The user can choose to replace the former altitude dimension instead, by setting

.. code-block:: ini

        [processor.Sonde.replace_alt_dim]
        drop_nan=False
