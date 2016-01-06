.. _sed-label:

Scanning Electron Diffraction (SED)
***********************************

The methods described here are specific to the following signals:

* :py:class:`~._signals.sed.SEDPattern`

This chapter described step by step the analysis of an EDS
spectrum (SEM or TEM).

Spectrum loading and parameters
-------------------------------

Data files used in the following examples can be downloaded using

.. code-block:: python

    >>> from urllib import urlretrieve
    >>> url = 'http://cook.msm.cam.ac.uk//~hyperspy//SED_tutorial//'
    >>> urlretrieve(url + 'GaAs_001.tif', 'GaAs_nanowire_002.rpl')

.. NOTE::

    The sample and the data used in this chapter are described in


Loading
^^^^^^^^

All data are loaded with the :py:func:`~.io.load` function, as described in details in
:ref:`Loading files<loading_files>`. HyperSpy is able to import different formats,
among them ".blo" (the format used by NanoMEGAS). Below are examples for loading a single diffraction pattern and loading a stack of diffraction patterns.

For a single diffraction pattern:

.. code-block:: python

    >>> dp = hs.load("GaAs_001.tif")
    >>> dp
    <Image, title: Image, dimensions: (|144, 144)>

For a SED dataset here stored as a stack of images:

.. code-block:: python

    >>> dps = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dps
    <Image, title: , dimensions: (100, 30|144, 144)>


Microscope and detector parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Microscope and detector parameters are stored in the :py:attr:`~.signal.Signal.metadata` attribute (see :ref:`metadata_structure`). These parameters can be displayed
as follows:

.. code-block:: python

    >>> dp = hs.load("Ni_superalloy_1pix.msa", signal_type="EDS_SEM")
    >>> dp.metadata.Acquisition_instrument.SEM
    ├── Detector
    │   └── EDS
    │       ├── azimuth_angle = 63.0
    │       ├── elevation_angle = 35.0
    │       ├── energy_resolution_MnKa = 130.0
    │       ├── live_time = 0.006855
    │       └── real_time = 0.0
    ├── beam_current = 0.0
    ├── beam_energy = 15.0
    └── tilt_stage = 38.0


Parameters can be specified directly:

.. code-block:: python

    >>> s = hs.load("Ni_superalloy_1pix.msa", signal_type="EDS_SEM")
    >>> s.metadata.Acquisition_instrument.SEM.beam_energy = 30

or with the
:py:meth:`~._signals.eds_tem.EDSTEMSpectrum.set_microscope_parameters` method:

.. code-block:: python

    >>> s = hs.load("Ni_superalloy_1pix.msa", signal_type="EDS_SEM")
    >>> s.set_microscope_parameters(beam_energy = 30)

or raising the gui:

.. code-block:: python

    >>> s = hs.load("Ni_superalloy_1pix.msa", signal_type="EDS_SEM")
    >>> s.set_microscope_parameters()

.. figure::  images/SED_microscope_parameters_gui.png
   :align:   center
   :width:   350

   SED microscope parameters preferences window.

If the microscope and detector parameters are not written in the original file, some
of them are set by default. The default values can be changed in the
:py:class:`~.defaults_parser.Preferences` class (see :ref:`preferences
<configuring-hyperspy-label>`).

.. code-block:: python

    >>> hs.preferences.SED.precession_angle = 36

or raising the gui:

.. code-block:: python

    >>> hs.preferences.gui()

.. figure::  images/SED_preferences_gui.png
   :align:   center
   :width:   400

   SED preferences window.


Machine Learning SED Data
-------------------------

Describe defaults and masking here.
