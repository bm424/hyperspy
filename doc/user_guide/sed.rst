.. _sed-label:

Scanning Electron Diffraction (SED)
***********************************

The methods described here are specific to the following signals:

* :py:class:`~._signals.sed.SEDPattern`

This chapter describes step-by-step the analysis of a SED dataset
acquired in a (S)TEM.

Spectrum loading and parameters
-------------------------------

Data files used in the following examples can be downloaded using

.. code-block:: python

    >>> from urllib import urlretrieve
    >>> url = 'http://cook.msm.cam.ac.uk//~hyperspy//SED_tutorial//'
    >>> urlretrieve(url + 'GaAs_001.tif', 'GaAs_nanowire_002.rpl')

.. NOTE::

    The sample and the data used in this chapter are described in...


Loading
^^^^^^^^

All data are loaded with the :py:func:`~.io.load` function, as described in details in
:ref:`Loading files<loading_files>`. HyperSpy is able to import different formats,
among them ".blo" (the format used by NanoMEGAS). Below are examples for loading a single 
diffraction pattern and loading a stack of diffraction patterns.

For a single diffraction pattern:

.. code-block:: python

    >>> dp = hs.load("GaAs_001.tif")
    >>> dp
    <Image, title: Image, dimensions: (|144, 144)>

For a SED dataset here stored as a stack of images:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dp
    <Image, title: , dimensions: (100, 30|144, 144)>


Microscope and detector parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Microscope and detector parameters are stored in the :py:attr:`~.signal.Signal.metadata` 
attribute (see :ref:`metadata_structure`). These parameters can be displayed as follows:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dp.metadata


Parameters can be specified directly:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dp.metadata.Acquisition_instrument.SED.convergence_angle = 5.

or with the
:py:meth:`~._signals.sed.SEDPattern.set_microscope_parameters` method:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dp.set_microscope_parameters(convergence_angle = 5.)

or raising the gui:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dp.set_microscope_parameters()

.. figure::  images/SED_microscope_parameters_gui.png
   :align:   center
   :width:   400

   SED microscope parameters preferences window.

If the microscope and detector parameters are not written in the original file, some
of them are set by default. The default values can be changed in the
:py:class:`~.defaults_parser.Preferences` class (see :ref:`preferences
<configuring-hyperspy-label>`).

.. code-block:: python

    >>> hs.preferences.SED.precession_angle = 36.

or raising the gui:

.. code-block:: python

    >>> hs.preferences.gui()

.. figure::  images/SED_preferences_gui.png
   :align:   center
   :width:   400

   SED preferences window.


Alignment and masking
---------------------

Basic preprocessing of SED datasets involves aligning the recorded patterns such that all 
have a common center and removing, by automated masking, parts of the dataset that are
problematic for further analysis. Alignment is based on determining the direct beam position
directly since it cannot be assumed, in general, that a recorded diffraction pattern is 
symmetric. Masking methods are provided to remove saturated pixels associated with the direct 
beam and to exclude data acquired in vacuum from further treatment.

The position of the direct beam can be estimated using the estimate_direct_beam_position()
method. This method implements the peak refinement algorithm originially described by
Zaeferrer [REF] to find the peak centre to pixel level accuracy.

Direct beam alignment
^^^^^^^^^^^^^^^^^^^^^

Alignment based on the direct beam position can be performed using the align_direct_beam()
method.

The align_direct_beam() method estimates the direct beam position in each SED pattern using 
the estimate_direct_beam_position() method, calculates the shift of each found position with
respect to a specified reference, and applies these shifts using the align2D() method.

Direct beam masking
^^^^^^^^^^^^^^^^^^^

A signal mask that excludes pixels in the SED patterns containing the direct beam can be
generated automatically using the direct_beam_mask() method. This is useful because pixels
associated with the direct beam are often saturated and this can lead to issues with further
analysis such as the application of unsupervised learning methods for decomposition.

The direct_beam_mask() method estimates the direct beam position in each SED pattern using 
the estimate_direct_beam_position() method and masks a circular region around that position
with a user specified radius.

The mask can be generated and checked as follows:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> dpmask = dp * dp.direct_beam_mask(radius=6)
    >>> dpmask.plot()

.. figure:: images/SED_direct_beam_mask.png
   :align: center
   :width: 400

   Automatically generated direct beam mask.


Vacuum masking
^^^^^^^^^^^^^^

A navigation mask to exclude SED patterns acquired in vacuum from further analysis can be
generated automatically using the vacuum_mask() method. Ignoring these patterns, which do 
not contain useful information, in later analysis is efficient in terms of computation time
and can improve results from statistical methods that use all of the selected data.

The vacuum_mask() method automatically determines whether a SED pattern was acquired in
vacuum by assessing whether or not any diffraction peaks exist in the region that does not
contain the direct beam. 

The method is applied as follows:

.. code-block:: python

    >>> dp = hs.load("GaAs_nanowire_002.rpl", signal_type="SED_Pattern")
    >>> 

.. figure:: images/SED_vacuum_mask.png
   :align: center
   :width: 400

   Automatically generated mask excluding SED patterns acquired in vacuum.
