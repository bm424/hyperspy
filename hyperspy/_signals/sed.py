# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division

import traits.api as t
import numpy as np


from hyperspy._signals.image import Image
from hyperspy.decorators import only_interactive
from hyperspy.gui.eds import TEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui


class SEDPattern(Image):
    _signal_type = "SED_Pattern"

    def __init__(self, *args, **kwards):
        Image.__init__(self, *args, **kwards)
        # Attributes defaults
        if 'Acquisition_instrument.TEM' not in self.metadata:
            if 'Acquisition_instrument.SEM' in self.metadata:
                self.metadata.set_item(
                    "Acquisition_instrument.TEM",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
        self._set_default_param()

    def _set_default_param(self):
        """Set microscope parameters to default values (defined in preferences)
        """

        md = self.metadata
        md.Signal.signal_type = 'EDS_TEM'

        if "md.Acquisition_instrument.TEM.accelerating_voltage" not in md:
            md.set_item(
                "Acquisition_instrument.TEM.accelerating_voltage",
                preferences.SED.sed_accelerating_voltage)
        if "Acquisition_instrument.TEM.convergence_angle" not in md:
            md.set_item(
                "Acquisition_instrument.TEM.convergence_angle",
                preferences.SED.sed_convergence_angle)
        if "Acquisition_instrument.TEM.precession_angle" not in md:
            md.set_item("Acquisition_instrument.TEM.precession_angle",
                        preferences.SED.sed_precession_angle)
        if "Acquisition_instrument.TEM.precession_frequency" not in md:
            md.set_item(
                "Acquisition_instrument.TEM.precession_frequency",
                preferences.SED.sed_precession_frequency)
        if "Acquisition_instrument.TEM.Detector.SED.exposure_time" not in md:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.SEDexposure_time",
                preferences.SED.sed_exposure_time)

    def set_microscope_parameters(self,
                                  accelerating_voltage=None,
                                  convergence_angle=None,
                                  precession_angle=None,
                                  precession_frequency=None,
                                  exposure_time=None):
        """Set the microscope parameters.

        If no arguments are given, raises an interactive mode to fill
        the values.

        Parameters
        ----------
        accelerating_voltage: float
            The energy of the electron beam in keV
        convergence_angle : float
            Convergence angle in mrad
        precession_angle : float
            Precession angle in mrad
        precession_frequency : float
            Precession frequency in Hz
        exposure_time : float
            Exposure time in ms.

        Examples
        --------
        >>> dp = hs.datasets.example_signals.SED_Pattern()
        >>> print(dp.metadata.Acquisition_instrument.TEM.precession_angle)
        >>> s.set_microscope_parameters(precession_angle=36.)
        >>> print(s.metadata.Acquisition_instrument.TEM.precession_angle)
        18.0
        36.0

        """
        md = self.metadata

        if accelerating_voltage is not None:
            md.set_item("Acquisition_instrument.TEM.accelerating_voltage",
                        accelerating_voltage)
        if convergence_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.convergence_angle",
                convergence_angle)
        if precession_angle is not None:
            md.set_item("Acquisition_instrument.TEM.precession_angle",
                        precession_angle)
        if precession_frequency is not None:
            md.set_item("Acquisition_instrument.TEM.precession_frequency",
                        precession_frequency)
        if exposure_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.SED.exposure_time",
                exposure_time)

        if set([accelerating_voltage, convergence_angle, precession_angle,
                precession_frequency, exposure_time]) == {None}:
            self._are_microscope_parameters_missing()

    @only_interactive
    def _set_microscope_parameters(self):
        tem_par = TEMParametersUI()
        mapping = {
            'Acquisition_instrument.TEM.accelerating_voltage':
            'tem_par.beam_energy',
            'Acquisition_instrument.TEM.convergence_angle':
            'tem_par.convergence_angle',
            'Acquisition_instrument.TEM.precession_angle':
            'tem_par.precession_angle',
            'Acquisition_instrument.TEM.precession_frequency':
            'tem_par.precession_frequency',
            'Acquisition_instrument.TEM.Detector.SED.exposure_time':
            'tem_par.exposure_time', }
        for key, value in mapping.iteritems():
            if self.metadata.has_item(key):
                exec('%s = self.metadata.%s' % (value, key))
        tem_par.edit_traits()

        mapping = {
            'Acquisition_instrument.TEM.accelerating_voltage':
            tem_par.beam_energy,
            'Acquisition_instrument.TEM.convergence_angle':
            tem_par.convergence_angle,
            'Acquisition_instrument.TEM.precession_angle':
            tem_par.precession_angle,
            'Acquisition_instrument.TEM.precession_frequency':
            tem_par.precession_frequency,
            'Acquisition_instrument.TEM.Detector.SED.exposure_time':
            tem_par.exposure_time, }

        for key, value in mapping.iteritems():
            if value != t.Undefined:
                self.metadata.set_item(key, value)
        self._are_microscope_parameters_missing()

    def _are_microscope_parameters_missing(self):
        """Check if the SED parameters necessary for further analysis
        are defined in metadata. Raise in interactive mode
         an UI item to fill or cahnge the values"""
        must_exist = (
            'Acquisition_instrument.TEM.accelerating_voltage', )

        missing_parameters = []
        for item in must_exist:
            exists = self.metadata.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters:
            if preferences.General.interactive is True:
                par_str = "The following parameters are missing:\n"
                for par in missing_parameters:
                    par_str += '%s\n' % par
                par_str += 'Please set them in the following wizard'
                is_ok = messagesui.information(par_str)
                if is_ok:
                    self._set_microscope_parameters()
                else:
                    return True
            else:
                return True
        else:
            return False

    def get_direct_beam_position(self):
        """

        """

    def direct_beam_mask(self, radius, center):
        """
        Generate a mask for the direct beam.

        Parameters
        ----------
        radius: float
            User specified radius for the circular mask.

        center: tuple
            User specified (y, x) position of the diffraction pattern center.
            i.e. the direct beam position.

        Return
        ------
        mask: signal
            The mask of the direct beam
        """

        r = radius

        y, x = np.ogrid[-center[0]:ny-center[0], -center[1]:nx-center[1]]
        mask = x*x + y*y <= r*r
        return mask

    def decomposition(self,
                      normalize_poissonian_noise=True,
                      direct_beam_mask=4.0,
                      *args,
                      **kwargs):
        """
        Decomposition with a choice of algorithms

        The results are stored in self.learning_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        direct_beam_mask : None or float or boolean numpy array
            The navigation locations marked as True are not used in the
            decompostion. If float is given the direct_beam_mask method is used
            to generate a mask with the float value as radius.
        closing: bool
            If true, applied a morphologic closing to the maks obtained by
            vacuum_mask.
        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca'
        output_dimension : None or int
            number of components to keep/calculate
        centre : None | 'variables' | 'trials'
            If None no centring is applied. If 'variable' the centring will be
            performed in the variable axis. If 'trials', the centring will be
            performed in the 'trials' axis. It only has effect when using the
            svd or fast_svd algorithms
        auto_transpose : bool
            If True, automatically transposes the data to boost performance.
            Only has effect when using the svd of fast_svd algorithms.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.
        polyfit :
        reproject : None | signal | navigation | both
            If not None, the results of the decomposition will be projected in
            the selected masked area.

        Examples
        --------
        >>> im = hs.datasets.example_signals.SED_Pattern()
        >>> ims = hs.stack([s]*3)
        >>> ims.change_dtype(float)
        >>> ims.decomposition()

        See also
        --------
        direct_beam_mask
        """
        if isinstance(direct_beam_mask, float):
            navigation_mask = self.direct_beam_mask(direct_beam_mask).data
        super(Image, self).decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            navigation_mask=navigation_mask, *args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)
