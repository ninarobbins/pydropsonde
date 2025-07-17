Level_1
=======

The Level_1 data is a direct output of the `Aspen <https://www.eol.ucar.edu/content/aspen>`_ software provided by NCAR.
It is used for a basic quality control and ``pydropsonde`` assumes that Aspen is run in it's default configuration.
For details on the basic quality control steps as well as the standard netCDF output files, please have a look at the `AspenDocs <https://ncar.github.io/aspendocs/index.html>`_.

There are two ways for processing sonde profiles from Level_0 to Level_1 via Aspen:

#. by default, ``pydropsonde`` will use an `Aspen docker image <https://github.com/atmdrops/aspenqc>`_ named `ghcr.io/atmdrops/aspenqc` and hosted on GitHub for running Aspen independent of your local operating system. It uses the latest Aspen software version available as docker image (v4.0.2 available since Jun 7, 2024) if not specified differently. The Aspen QC is run in default configuration and saves the output data in netCDF format.
#. Alternatively, you can process your raw files manually with Aspen. ``pydropsonde`` won't reprocess the files if they are saved within your :ref:`data_directory_structure`. Download your favourite version of Aspen from the `NCAR Software Center <https://www.eol.ucar.edu/software/aspen>`_ and either use the command line interface or the included GUI "BatchAspen" for processing individual or a group of files. For further processing with ``pydropsonde`` the output file format should be netCDF.

If you opt for processing with docker, you need to install `docker <https://www.docker.com/>`_ and start a docker daemon (dockerd) locally to be able to use the ASPEN docker image.
The simplest way would be to install Docker Desktop (on MacOS also possible via ``brew install --cask docker``) and open the application before executing ``pydropsonde``.
Further information can be found in the respective `Aspen docker image <https://github.com/atmdrops/aspenqc>`_ repository.

.. note::
   In case that sondes are recognized as "minisondes", processing via the Docker image or CLI will result in errors.
   The Level_0 to Level_1 processing can still be done manually with (Batch)Aspen with the minisonde configuration file.
   From Level_1 onwards those files are then included in the pipeline.
