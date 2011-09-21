# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt

import numpy as np
import enthought.traits.api as t
import enthought.traits.ui.api as tu
from enthought.traits.ui.menu import OKButton, ApplyButton, CancelButton

from hyperspy import components
from hyperspy.misc import utils
from hyperspy import drawing
from hyperspy.misc.interactive_ns import interactive_ns
from hyperspy.gui.tools import SpanSelectorInSpectrum


class BackgroundRemoval(SpanSelectorInSpectrum):
    background_type = t.Enum('Power Law', 'Polynomial', default = 'Power Law')
    view = tu.View(
        tu.Group(
            'background_type',),
            kind = 'nonmodal',
            buttons= [OKButton, ApplyButton, CancelButton],)
                 
    def __init__(self, signal):
        super(BackgroundRemoval, self).__init__(signal)
        
        # Background
        self.pl = components.PowerLaw()
        self.bg_line = None

    def store_current_spectrum_bg_parameters(self, *args, **kwards):
        if self.span_selector.range is None or \
        self.span_selector.range is None: return
        pars = utils.two_area_powerlaw_estimation(
        self.signal, *self.span_selector.range,only_current_spectrum = True)
        self.pl.r.value = pars['r']
        self.pl.A.value = pars['A']
        
    def on_disabling_span_selector(self):
        if self.bg_line is not None:
            self.span_selector.ax.lines.remove(self.bg_line)
            self.bg_line = None
            
    def _ss_left_value_changed(self, old, new):
        self.plot_bg_removed_spectrum()
        
    def _ss_right_value_changed(self, old, new):
        self.plot_bg_removed_spectrum()
                      
    def plot_bg_removed_spectrum(self, *args, **kwards):
        if self.span_selector is None or \
            self.span_selector.range is None: return
        self.store_current_spectrum_bg_parameters()
        ileft = self.axis.value2index(self.span_selector.range[0])
        iright = self.axis.value2index(self.span_selector.range[1])
        ea = self.axis.axis[ileft:]
        if self.bg_line is not None:
            self.span_selector.ax.lines.remove(self.bg_line)
        self.bg_line, = self.signal._plot.spectrum_plot.left_ax.plot(
        ea, self.pl.function(ea), color = 'black')
        self.signal._plot.spectrum_plot.left_ax.figure.canvas.draw()

        
#class EgertonPanel(t.HasTraits):
#    define_background_window = t.Bool(False)
#    bg_window_size_variation = t.Button()
#    background_substracted_spectrum_name = t.Str('signal')
#    extract_background = t.Button()    
#    define_signal_window = t.Bool(False)
#    signal_window_size_variation = t.Button()
#    signal_name = t.Str('signal')
#    extract_signal = t.Button()
#    view = tu.View(tu.Group(
#        tu.Group('define_background_window',
#                 tu.Item('bg_window_size_variation', 
#                         label = 'window size effect', show_label=False),
#                 tu.Item('background_substracted_spectrum_name'),
#                 tu.Item('extract_background', show_label=False),
#                 ),
#        tu.Group('define_signal_window',
#                 tu.Item('signal_window_size_variation', 
#                         label = 'window size effect', show_label=False),
#                 tu.Item('signal_name', show_label=True),
#                 tu.Item('extract_signal', show_label=False)),))
#                 
#    def __init__(self, signal):
#        
#        self.signal = signal
#        
#        # Background
#        self.span_selector = None
#        self.pl = components.PowerLaw()
#        self.bg_line = None
#        self.bg_cube = None
#                
#        # Signal
#        self.signal_span_selector = None
#        self.signal_line = None
#        self.signal_map = None
#        self.map_ax = None
#    
#    def store_current_spectrum_bg_parameters(self, *args, **kwards):
#        if self.define_background_window is False or \
#        self.span_selector.range is None: return
#        pars = utils.two_area_powerlaw_estimation(
#        self.signal, *self.span_selector.range,only_current_spectrum = True)
#        self.pl.r.value = pars['r']
#        self.pl.A.value = pars['A']
#                     
#        if self.define_signal_window is True and \
#        self.signal_span_selector.range is not None:
#            self.plot_signal_map()
#                     
#    def _define_background_window_changed(self, old, new):
#        if new is True:
#            self.span_selector = \
#            drawing.widgets.ModifiableSpanSelector(
#            self.signal.hse.spectrum_plot.left_ax,
#            onselect = self.store_current_spectrum_bg_parameters,
#            onmove_callback = self.plot_bg_removed_spectrum)
#        elif self.span_selector is not None:
#            if self.bg_line is not None:
#                self.span_selector.ax.lines.remove(self.bg_line)
#                self.bg_line = None
#            if self.signal_line is not None:
#                self.span_selector.ax.lines.remove(self.signal_line)
#                self.signal_line = None
#            self.span_selector.turn_off()
#            self.span_selector = None
#                      
#    def _bg_window_size_variation_fired(self):
#        if self.define_background_window is False: return
#        left = self.span_selector.rect.get_x()
#        right = left + self.span_selector.rect.get_width()
#        energy_window_dependency(self.signal, left, right, min_width = 10)
#        
#    def _extract_background_fired(self):
#        if self.pl is None: return
#        signal = self.signal() - self.pl.function(self.signal.energy_axis)
#        i = self.signal.energy2index(self.span_selector.range[1])
#        signal[:i] = 0.
#        s = Spectrum({'calibration' : {'data_cube' : signal}})
#        s.get_calibration_from(self.signal)
#        interactive_ns[self.background_substracted_spectrum_name] = s       
#        
#    def _define_signal_window_changed(self, old, new):
#        if new is True:
#            self.signal_span_selector = \
#            drawing.widgets.ModifiableSpanSelector(
#            self.signal.hse.spectrum_plot.left_ax, 
#            onselect = self.store_current_spectrum_bg_parameters,
#            onmove_callback = self.plot_signal_map)
#            self.signal_span_selector.rect.set_color('blue')
#        elif self.signal_span_selector is not None:
#            self.signal_span_selector.turn_off()
#            self.signal_span_selector = None
#            
#    def plot_bg_removed_spectrum(self, *args, **kwards):
#        if self.span_selector.range is None: return
#        self.store_current_spectrum_bg_parameters()
#        ileft = self.signal.energy2index(self.span_selector.range[0])
#        iright = self.signal.energy2index(self.span_selector.range[1])
#        ea = self.signal.energy_axis[ileft:]
#        if self.bg_line is not None:
#            self.span_selector.ax.lines.remove(self.bg_line)
#            self.span_selector.ax.lines.remove(self.signal_line)
#        self.bg_line, = self.signal.hse.spectrum_plot.left_ax.plot(
#        ea, self.pl.function(ea), color = 'black')
#        self.signal_line, = self.signal.hse.spectrum_plot.left_ax.plot(
#        self.signal.energy_axis[iright:], self.signal()[iright:] - 
#        self.pl.function(self.signal.energy_axis[iright:]), color = 'black')
#        self.signal.hse.spectrum_plot.left_ax.figure.canvas.draw()

#        
#    def plot_signal_map(self, *args, **kwargs):
#        if self.define_signal_window is True and \
#        self.signal_span_selector.range is not None:
#            ileft = self.signal.energy2index(self.signal_span_selector.range[0])
#            iright = self.signal.energy2index(self.signal_span_selector.range[1])
#            signal_sp = self.signal.data_cube[ileft:iright,...].squeeze().copy()
#            if self.define_background_window is True:
#                pars = utils.two_area_powerlaw_estimation(
#                self.signal, *self.span_selector.range, only_current_spectrum = False)
#                x = self.signal.energy_axis[ileft:iright, np.newaxis, np.newaxis]
#                A = pars['A'][np.newaxis,...]
#                r = pars['r'][np.newaxis,...]
#                self.bg_sp = (A*x**(-r)).squeeze()
#                signal_sp -= self.bg_sp
#            self.signal_map = signal_sp.sum(0)
#            if self.map_ax is None:
#                f = plt.figure()
#                self.map_ax = f.add_subplot(111)
#                if len(self.signal_map.squeeze().shape) == 2:
#                    self.map = self.map_ax.imshow(self.signal_map.T, 
#                                                  interpolation = 'nearest')
#                else:
#                    self.map, = self.map_ax.plot(self.signal_map.squeeze())
#            if len(self.signal_map.squeeze().shape) == 2:
#                    self.map.set_data(self.signal_map.T)
#                    self.map.autoscale()
#                    
#            else:
#                self.map.set_ydata(self.signal_map.squeeze())
#            self.map_ax.figure.canvas.draw()
#            
#    def _extract_signal_fired(self):
#        if self.signal_map is None: return
#        if len(self.signal_map.squeeze().shape) == 2:
#            s = Image(
#            {'calibration' : {'data_cube' : self.signal_map.squeeze()}})
#            s.xscale = self.signal.xscale
#            s.yscale = self.signal.yscale
#            s.xunits = self.signal.xunits
#            s.yunits = self.signal.yunits
#            interactive_ns[self.signal_name] = s
#        else:
#            s = Spectrum(
#            {'calibration' : {'data_cube' : self.signal_map.squeeze()}})
#            s.energyscale = self.signal.xscale
#            s.energyunits = self.signal.xunits
#            interactive_ns[self.signal_name] = s
#    

#def energy_window_dependency(s, left, right, min_width = 10):
#    ins = s.energy2index(left)
#    ine = s.energy2index(right)
#    energies = s.energy_axis[ins:ine - min_width]
#    rs = []
#    As = []
#    for E in energies:
#        di = utils.two_area_powerlaw_estimation(s, E, ine)
#        rs.append(di['r'].mean())
#        As.append(di['A'].mean())
#    f = plt.figure()
#    ax1  = f.add_subplot(211)
#    ax1.plot(s.energy_axis[ins:ine - min_width], rs)
#    ax1.set_title('Rs')
#    ax1.set_xlabel('Energy')
#    ax2  = f.add_subplot(212, sharex = ax1)
#    ax2.plot(s.energy_axis[ins:ine - min_width], As)
#    ax2.set_title('As')
#    ax2.set_xlabel('Energy')
#    return rs, As
