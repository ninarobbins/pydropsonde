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
