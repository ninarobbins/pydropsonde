Getting Started
===============



This page explains how to install and run the ``pydropsonde`` python package. The package includes functions to process the dropsonde data from raw files to datasets on uniform grids and including several derived meteorological variables.

To run ``pydropsonde`` on a set of dropsonde data, you first need to create a respective configuration file. An example and further infos are provided below. To run the ``pydropsonde`` pipeline, you need to provide the path to the config file:

.. code-block:: bash

        pydropsonde -c <config_file>


The processing from Level_0 to Level_1 (using `aspen <https://www.eol.ucar.edu/content/aspen>`_) is included in ``pydropsonde``. It makes use of a docker image containing the ASPEN software. Before actually calling pydropsonde, you therefore need to install `docker <https://www.docker.com/>`_ and start a docker daemon (dockerd) locally to be able to use the ASPEN docker image.
The simplest way would be to install Docker Desktop (on MacOS also possible via ``brew install --cask docker``) and open the application before executing pydropsonde. Further information can be found in the respective `aspenqc <https://github.com/atmdrops/aspenqc>`_ repository.

.. toctree::
    :caption: Contents:

    installation
    configs
