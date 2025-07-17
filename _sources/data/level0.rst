Level 0
=======

Level_0 refers to the raw data files saved by the `AVAPS Dropsonde System <https://www.eol.ucar.edu/observing_facilities/avaps-dropsonde-system>`_ in its default configuration.
For each dropped sonde that system stores the data in different formats, partly with overlapping information.
In addition to the single profile files, the system saves metadata related to a measurement sequence, typically one flight, as defined by the start and closing of the AVAPS software.

An overview of typical files stored at Level_0 is shown in `George et al., 2021 <https://essd.copernicus.org/articles/13/5253/2021/essd-13-5253-2021.html>`_ and copied below:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - File Type
     - Description
   * - A files
     - Sounding attribute file; includes channel configuration, COM port data, hardware configuration, launch obs data, sensor errors, aircraft data, software config and firmware information
   * - B files
     - File containing binary data; same as D files
   * - C files
     - Sounding data stored as comma-separated-value files
   * - D files
     - Raw sounding data recorded for timestamp at every 0.25 s
   * - D_P files
     - Only post-launch raw data; same as D files
   * - R files
     - Receiver port data: signal strength and receiver frequency
   * - 0_SysLog files
     - Comma-separated-value file of all AVAPS system logs
   * - 1_Aircraft files
     - TXT file of aircraft position data in the IWGADTS format (IWG1)
   * - 2_GPSRef files
     - TXT file of GPS data: GPGGA (system fix data) and GPRMC (minimum specific GPS/Transit data)
   * - 3_SpecAnlyzr files
     - TXT file of logs of spectral analyzer

``pydropsonde`` uses the `D files` with the actual sonde profile data in the further processing.
Metadata information from the `A files`, e.g. a successful launch detect, is added within the processing.
In principle, the processing can run without `A files`, but obviously does not have metadata in that case.
