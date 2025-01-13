Config Files
============

A valid config file states at least the path to the dropsonde data in the mandatory section.
The below description shows exemplarily the general structure and lists further optional statements.
For illustration, you can find an example for a valid config file ``dropsonde.cfg`` that works with the example data ``example_data/`` in the `pydropsonde GitHub repository <https://github.com/atmdrops/pydropsonde>`_ that also hosts the pages of the documentation that you are currently reading.


.. _mandatory:

MANDATORY
---------
The mandatory section contains the data directory of the dropsonde data like

.. code-block:: ini

        [MANDATORY]
        data_directory = ./example_data

Technically, this is everything a valid config file needs. However, you can specify the place of your flight folders and of the actual dropsonde data in an optional section.

.. _optional:

OPTIONAL
--------

The optional arguments comprise statements on the present data structure, global attributes and a very powerful mechanism to shape the processing in various data levels e.g. by adding certain quality control tests.

Data structure
**************

A frequent usecase in this section is the definition of your folder structure if you have multiple sonde profiles from e.g.
different platforms or single flights and possibly also already existing preprocessed data levels (see :ref:`data_directory_structure` for details).
Based on the example data mentioned above, we could specify

#. the path to the folders containing sonde profiles of a singel flight ``path_to_flight_ids``
#. the path to the actual Level_0 files ``path_to_l0_files``

.. code-block:: ini

        [OPTIONAL]
        path_to_flight_ids = {platform}/Level_0
        path_to_l0_files = {platform}/Level_0/{flight_id}

Global attributes
*****************

Another usecase are global attributes that shall be added to any dataset that is saved.
You can specify any global attributes such as ``author`` or contact details ``author_email`` in a ``GLOBAL_ATTRS`` section.
For example

.. code-block:: ini

        [GLOBAL_ATTRS]
        author = Geet George
        author_email = g.george@tudelft.nl

The pydropsonde version used for processing will be added as a global attribute in any case.

Modifying the processing
************************

Any deviation from the default processing can be specified in the config file by stating the respective arguments to a function.
For example, you can specify the quality control filter applied when processing data to Level_2.

.. code-block:: ini

        [processor.Sonde.filter_qc_fail]
        filter_flags = profile_fullness,near_surface_coverage,alt_near_gpsalt

Also, you can customize the Level_2 and Level_3 file names as shown below.

.. code-block:: ini

        [processor.Sonde.get_l2_filename]
        l2_filename_template = campaign_name_{platform}_{launch_time}_{flight_id}_{serial_id}_Level_2.nc
        [processor.Gridded.get_l3_filename]
        l3_filename_template = campaign_name_{platform}_{flight_id}_Level_3.nc
