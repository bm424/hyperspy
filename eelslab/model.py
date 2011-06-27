# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
# USA

import copy
import os
import tempfile

import numpy as np
from components.edge import Edge
from components.power_law import PowerLaw
from interactive_ns import interactive_ns
from defaults_parser import defaults
from utils import two_area_powerlaw_estimation
from estimators import Estimators
from optimizers import Optimizers
from model_controls import Controls
import messages
import drawing.spectrum
import progressbar

class Model(list, Optimizers, Estimators, Controls):
    """Build a fit a model
    
    Parameters
    ----------
    data : Spectrum or Experiment instance
    auto_background : boolean
        If True, it adds automatically a powerlaw to the model and estimate the 
        parameters by the two-area method.
    auto_add_edges : boolen
        If True (default), it will automatically add the ionization edges as 
        defined in the Spectrum instance.
    """
    
    __firstimetouch = True

    def __init__(self, data, auto_background = True, auto_add_edges = True):
        from experiments import Experiments
        from spectrum import Spectrum
        self.auto_update_plot = False
        if isinstance(data, Experiments):
            self.experiments = data
            self.hl = data.hl
            if hasattr(data, 'll'):
                self.ll = data.ll
            else:
                self.ll = None
        elif isinstance(data, Spectrum):
            self.hl = data
            self.ll = None
        else:
            messages.warning_exit(
            "The data must be a member of the Spectrum "
            "or Experiment classes")
         
        # Create some containers for the information generated by the fit             
        self.least_squares_fit_output = np.zeros((self.hl.xdimension,
        self.hl.ydimension)).tolist()
        self.free_parameters_boundaries = None
        self.channel_switches=np.array([True]*self.hl.energydimension)
        self.model_cube = np.zeros(self.hl.data_cube.shape, dtype = 'float')
        self.model_cube[:] = np.nan

        if auto_background:
            bg = PowerLaw()
            interactive_ns['bg'] = bg
            self.append(bg)

        if self.ll is not None:
            self.convolved = True
            if self.experiments.convolution_axis is None:
                self.experiments.set_convolution_axis()
        else:
            self.convolved = False
        self.coordinates = self.hl.coordinates
        self.coordinates.connect(self.charge)
        if self.hl.edges and auto_add_edges is True:
            self.extend(self.hl.edges)
        
    # Extend the list methods to call the _touch when the model is modified
    def append(self, object):
        list.append(self,object)
        try:
            self._touch()
        except:
            self.remove(object)
            messages.warning('The object was not added to the model because ' 
            'an error ocurred.\nCheck that the object was a valid component')
    
    def insert(self, object):
        list.insert(self,object)
        try:
            self._touch()
        except:
            self.remove(object)
            messages.warning('The object was not added to the model because ' 
            'an error ocurred.\nCheck that the object was a valid component')
   
    def extend(self, iterable):
        list.extend(self,iterable)
        try:
            self._touch()
        except:
            messages.warning('The objects were not added to the model because ' 
            'an error ocurred.\nCheck that the objects were valid components')
            for object in iterable:
                self.remove(object, touch = False)
        self._touch()
                
    def __delitem__(self, object):
        list.__delitem__(self,object)
        self._touch()
    
    def remove(self, object, touch = True):
        list.remove(self,object)
        if touch is True:
            self._touch() 

    def _touch(self):
        """Run model setup tasks
        
        This function must be called everytime that we add or remove components
        from the model.
        It creates the bookmarks self.edges and sef.__background_components and 
        configures the edges by setting the dispersion attribute and setting 
        the fine structure.
        """
        self.edges = []
        self.__background_components = []
        for component in self:
            if isinstance(component,Edge):
                component.dispersion = self.hl.energyscale
                component.setfslist()
                if component.edge_position() < \
                self.hl.energy_axis[self.channel_switches][0]:
                    component.isbackground = True
                if component.isbackground is not True:
                    self.edges.append(component)
                else :
                    component.fs_state = False
                    component.fslist.free = False
                    component.backgroundtype = "edge"
                    self.__background_components.append(component)

            elif isinstance(component,PowerLaw) or component.isbackground is True:
                self.__background_components.append(component)
            component.create_arrays(*self.hl.data_cube.shape[1:])

        if not self.edges:
            messages.warning("The model contains no edges")
        else:
            self.edges.sort(key = Edge.edge_position)
            self.resolve_fine_structure()
        if len(self.__background_components) > 1 :
            self.__backgroundtype = "mix"
        elif not self.__background_components:
            messages.warning("No background model has been defined")
        else :
            self.__backgroundtype = \
            self.__background_components[0].__repr__()
            if self.__firstimetouch and self.edges:
                self.two_area_background_estimation()
                self.__firstimetouch = False
        self.connect_parameters2update_plot()
                
    def connect_parameters2update_plot(self):   
        for component in self:
            for parameter in component.parameters:
                if self.hl.hse is not None:
                    for line in self.hl.hse.spectrum_plot.left_ax_lines:
                        parameter.connect(line.update)
                    parameter.connection_activated = False
    
    def disconnect_parameters2update_plot(self):
        for component in self:
            for parameter in component.parameters:
                if self.hl.hse is not None:
                    for line in self.hl.hse.spectrum_plot.left_ax_lines:
                        parameter.disconnect(line.update)
                    parameter.connection_activated = False
                            
    def set_auto_update_plot(self, tof):
        for component in self:
            for parameter in component.parameters:
                parameter.connection_activated = tof
        self.auto_update_plot = tof
                    
    def generate_cube(self):
        """Generate a SI with the current model
        
        The SI is stored in self.model_cube
        """
        for iy in range(self.model_cube.shape[2]):
            for ix in range(self.model_cube.shape[1]):
                print "x = %i\ty = %i" % (ix, iy)
                self.coordinates.ix = ix
                self.coordinates.iy = iy
                self.model_cube[self.channel_switches, self.coordinates.ix, self.coordinates.iy] = \
                self.__call__(
                non_convolved = not self.convolved, onlyactive = True)
                self.model_cube[self.channel_switches == False,:,:] = np.nan

    def resolve_fine_structure(self,preedge_safe_window_width = 
        defaults.preedge_safe_window_width, i1 = 0):
        """Adjust the fine structure of all edges to avoid overlapping
        
        This function is called automatically everytime the position of an edge
        changes
        
        Parameters
        ----------
        preedge_safe_window_width : float
            minimum distance between the fine structure of an ionization edge 
            and that of the following one.
        """

        while (self.edges[i1].fs_state is False or  
        self.edges[i1].active is False) and i1 < len(self.edges)-1 :
            i1+=1
        if i1 < len(self.edges)-1 :
            i2=i1+1
            while (self.edges[i2].fs_state is False or 
            self.edges[i2].active is False) and \
            i2 < len(self.edges)-1:
                i2+=1
            if self.edges[i2].fs_state is True:
                distance_between_edges = self.edges[i2].edge_position() - \
                self.edges[i1].edge_position()
                if self.edges[i1].fs_emax > distance_between_edges - \
                preedge_safe_window_width :
                    if (distance_between_edges - 
                    preedge_safe_window_width) <= \
                    defaults.min_distance_between_edges_for_fine_structure:
                        print " Automatically desactivating the fine \
                        structure of edge number",i2+1,"to avoid conflicts\
                         with edge number",i1+1
                        self.edges[i2].fs_state = False
                        self.edges[i2].fslist.free = False
                        self.resolve_fine_structure(i1 = i2)
                    else:
                        new_fs_emax = distance_between_edges - \
                        preedge_safe_window_width
                        print "Automatically changing the fine structure \
                        width of edge",i1+1,"from", \
                        self.edges[i1].fs_emax, "eV to", new_fs_emax, \
                        "eV to avoid conflicts with edge number", i2+1
                        self.edges[i1].fs_emax = new_fs_emax
                        self.resolve_fine_structure(i1 = i2)
                else:
                    self.resolve_fine_structure(i1 = i2)
        else:
            return

    def _set_p0(self):
        p0 = None
        for component in self:
            component.refresh_free_parameters()
            if component.active:
                for param in component.free_parameters:
                    if p0 is not None:
                        p0 = (p0 + [param.value,] 
                        if not isinstance(param.value, list) 
                        else p0 + param.value)
                    else:
                        p0 = ([param.value,] 
                        if not isinstance(param.value, list) 
                        else param.value)
        self.p0 = tuple(p0)
    
    def set_boundaries(self):
        """Generate the boundary list.
        
        Necessary before fitting with a boundary awared optimizer
        """
        self.free_parameters_boundaries = []
        for component in self:
            component.refresh_free_parameters()
            if component.active:
                for param in component.free_parameters:
                    if param._number_of_elements == 1:
                        self.free_parameters_boundaries.append((
                        param._bounds))
                    else:
                        self.free_parameters_boundaries.extend((
                        param._bounds))

    def set(self):
        """ Store the parameters of the current coordinates into the 
        parameters array.
        
        If the parameters array has not being defined yet it creates it filling 
        it with the current parameters."""
        for component in self:
            component.store_current_parameters_in_map(
            *self.axes_manager.axes)

    def charge(self, only_fixed = False):
        """Charge the parameters for the current spectrum from the parameters 
        array
        
        Parameters
        ----------
        only_fixed : bool
            If True, only the fixed parameters will be charged.
        """
        switch_aap = (False != self.auto_update_plot)
        if switch_aap is True:
            self.set_auto_update_plot(False)
        for component in self :
            component.charge_value_from_map(
            self.coordinates.ix,self.coordinates.iy, only_fixed = 
            only_fixed)
        if switch_aap is True:
            self.set_auto_update_plot(True)
            for line in self.hl.hse.spectrum_plot.left_ax_lines:
                line.update()

    def _charge_p0(self, p_std = None):
        """Charge the free data for the current coordinates (x,y) from the
        p0 array.
        
        Parameters
        ----------
        p_std : array
            array containing the corresponding standard deviation
        """
        comp_p_std = None
        counter = 0
        for component in self: # Cut the parameters list
            if component.active:
                if p_std is not None:
                    comp_p_std = p_std[counter: counter + component.nfree_param]
                component.charge(
                self.p0[counter: counter + component.nfree_param], True, 
                comp_p_std)
                counter += component.nfree_param

    # Defines the functions for the fitting process -------------------------
    def model2plot(self, coordinates, out_of_region2nans = True):
        old_coord = None
        if coordinates is not self.coordinates:
            old_coord = self.axes_manager.axes
            self.coordinates.ix, self.coordinates.iy = coordinates.coordinates
        s = self.__call__(non_convolved=False, onlyactive=True)
        if old_coord is not None:
            self.coordinates.ix, self.coordinates.iy = old_coord
            self.charge()
        if out_of_region2nans is True:
            ns = np.zeros((self.hl.energy_axis.shape))
            ns[:] = np.nan
            ns[self.channel_switches] = s
        return ns
    
    def __call__(self,non_convolved=False, onlyactive=False) :
        """Returns the corresponding model for the current coordinates
        
        Parameters
        ----------
        non_convolved : bool
            If True it will return the deconvolved model
        only_active : bool
            If True, only the active components will be used to build the model.
            
        cursor: 1 or 2
        
        Returns
        -------
        numpy array
        """
            
        if self.convolved is False or non_convolved is True:
            axis = self.hl.energy_axis[self.channel_switches]
            sum_ = np.zeros(len(axis))
            if onlyactive is True:
                for component in self: # Cut the parameters list
                    if component.active:
                        np.add(sum_, component.function(axis),
                        sum_)
                return sum_
            else:
                for component in self: # Cut the parameters list
                    np.add(sum_, component.function(axis),
                     sum_)
                return sum_

        else: # convolved
            counter = 0
            sum_convolved = np.zeros(len(self.experiments.convolution_axis))
            sum_ = np.zeros(len(self.hl.energy_axis))
            for component in self: # Cut the parameters list
                if onlyactive :
                    if component.active:
                        if component.convolved:
                            np.add(sum_convolved,
                            component.function(
                            self.experiments.convolution_axis), sum_convolved)
                        else:
                            np.add(sum_,
                            component.function(self.hl.energy_axis), sum_)
                        counter+=component.nfree_param
                else :
                    if component.convolved:
                        np.add(sum_convolved,
                        component.function(self.experiments.convolution_axis),
                        sum_convolved)
                    else:
                        np.add(sum_, component.function(self.hl.energy_axis),
                        sum_)
                    counter+=component.nfree_param
            to_return = sum_ + np.convolve(
                self.ll.data_cube[: , self.coordinates.ix, self.coordinates.iy], 
                sum_convolved, mode="valid")
            to_return = to_return[self.channel_switches]
            return to_return


    def set_energy_region(self, E1=None, E2=None):
        """Use only the selected area in the fitting routine.
        
        Parameters
        ----------
        E1 : None or float
        E2 : None or float
        
        Notes
        -----
        To use the full energy range call the function without arguments.
        """
        if E1 is not None :
            if E1 > self.hl.energy_axis[0]:
                start_index = self.hl.energy2index(E1)
            else :
                start_index = None
        else :
            start_index = None
        if E2 is not None :
            if E2 < self.hl.energy_axis[-1]:
                stop_index = self.hl.energy2index(E2)
            else :
                stop_index = None
        else:
            stop_index = None
        self.backup_channel_switches = copy.copy(self.channel_switches)
        self.channel_switches[:] = False
        self.channel_switches[start_index:stop_index] = True

    def remove_data_range(self,E1 = None,E2= None):
        """Do not use the data in the selected range in the fitting rountine
        
        Parameters
        ----------
        E1 : None or float
        E2 : None or float
        
        Notes
        -----
        To use the full energy range call the function without arguments.
        """
        if E1 is not None :
            start_index = self.hl.energy2index(E1)
        else :
            start_index = None
        if E2 is not None :
            stop_index = self.hl.energy2index(E2)
        else:
            stop_index = None
        self.channel_switches[start_index:stop_index] = False

    def _model_function(self,param):

        if self.convolved is True:
            counter = 0
            sum_convolved = np.zeros(len(self.experiments.convolution_axis))
            sum = np.zeros(len(self.hl.energy_axis))
            for component in self: # Cut the parameters list
                if component.active:
                    if component.convolved:
                        np.add(sum_convolved, component(param[\
                        counter:counter+component.nfree_param],
                        self.experiments.convolution_axis), sum_convolved)
                    else:
                        np.add(sum, component(param[counter:counter + \
                        component.nfree_param],self.hl.energy_axis), sum)
                    counter+=component.nfree_param

            return (sum + np.convolve(self.ll.data_cube[
            :,self.coordinates.ix,self.coordinates.iy],
        sum_convolved,mode="valid"))[self.channel_switches]

        else:
            axis = self.hl.energy_axis[self.channel_switches]
            counter = 0
            first = True
            for component in self: # Cut the parameters list
                if component.active:
                    if first:
                        sum = component(param[counter:counter + \
                        component.nfree_param],axis)
                        first = False
                    else:
                        sum += component(param[counter:counter + \
                        component.nfree_param], axis)
                    counter += component.nfree_param
            return sum

    def _jacobian(self,param, y, weights = None):
        if self.convolved is True:
            counter = 0
            grad = np.zeros(len(self.hl.energy_axis))
            for component in self: # Cut the parameters list
                if component.active:
                    component.charge(param[counter:counter + \
                    component.nfree_param] , onlyfree = True)
                    if component.convolved:
                        for parameter in component.free_parameters :
                            par_grad = np.convolve(
                            parameter.grad(self.experiments.convolution_axis), 
                            self.ll.data_cube[
                            :,self.coordinates.ix,self.coordinates.iy], 
                            mode="valid")
                            if parameter._twins:
                                for parameter in parameter._twins:
                                    np.add(par_grad, np.convolve(
                                    parameter.grad(
                                    self.experiments.convolution_axis), 
                                    self.ll.data_cube[
                                    :, self.coordinates.ix, self.coordinates.iy], 
                                    mode="valid"), par_grad)
                            grad = np.vstack((grad, par_grad))
                        counter += component.nfree_param

                    else:
                        for parameter in component.free_parameters :
                            par_grad = parameter.grad(self.hl.energy_axis)
                            if parameter._twins:
                                for parameter in parameter._twins:
                                    np.add(par_grad, parameter.grad(
                                    self.hl.energy_axis), par_grad)
                            grad = np.vstack((grad, par_grad))
                        counter += component.nfree_param
            if weights is None:
                return grad[1:, self.channel_switches]
            else:
                return grad[1:, self.channel_switches] * weights
        else:
            axis = self.hl.energy_axis[self.channel_switches]
            counter = 0
            grad = axis
            for component in self: # Cut the parameters list
                if component.active:
                    component.charge(param[counter:counter + \
                    component.nfree_param] , onlyfree = True)
                    for parameter in component.free_parameters :
                        par_grad = parameter.grad(axis)
                        if parameter._twins:
                            for parameter in parameter._twins:
                                np.add(par_grad, parameter.grad(
                                axis), par_grad)
                        grad = np.vstack((grad, par_grad))
                    counter+=component.nfree_param
            if weights is None:
                return grad[1:,:]
            else:
                return grad[1:,:] * weights
        
    def _function4odr(self,param,x):
        return self._model_function(param)
    
    def _jacobian4odr(self,param,x):
        return self._jacobian(param, x)

    def smart_fit(self, background_fit_E1 = None, **kwards):
        """ Fits everything in a cascade style."""

        # Fit background
        self.fit_background(background_fit_E1, **kwards)

        # Fit the edges
        for i in range(0,len(self.edges)) :
            self.fit_edge(i,background_fit_E1, **kwards)
                
    def fit_background(self,startenergy = None, kind = 'single', **kwards):
        """Fit an EELS spectrum ionization edge by ionization edge from left 
        to right to optimize convergence.
        
        Parameters
        ----------
        startenergy : float
        kind : {'single', 
        """
        ea = self.hl.energy_axis[self.channel_switches]

        print "Fitting the", self.__backgroundtype, "background"
        edges = copy.copy(self.edges)
        edge = edges.pop(0)
        if startenergy is None:
            startenergy = ea[0]
        i = 0
        while edge.edge_position() < startenergy or edge.active is False:
            i+=1
            edge = edges.pop(0)
        self.set_energy_region(startenergy,edge.edge_position() - \
        defaults.preedge_safe_window_width)
        active_edges = []
        for edge in self.edges[i:]:
            if edge.active:
                active_edges.append(edge)
        self.disable_edges(active_edges)
        if kind == 'single':
            self.fit(**kwards)
        if kind == 'multi':
            self.multifit(**kwards)
        self.channel_switches = copy.copy(self.backup_channel_switches)
        self.enable_edges(active_edges)
        
    def two_area_background_estimation(self, E1 = None, 
    E2 = None):
        """
        Estimates the parameters of a power law background with the two
        area method.
        """
        ea = self.hl.energy_axis[self.channel_switches]
        if E1 is None or E1 < ea[0]:
            E1 = ea[0]
        else:
            E1 = E1
        if E2 is None:
            if self.edges:
                i = 0
                while self.edges[i].edge_position() < E1 or \
                self.edges[i].active is False:
                    i += 1
                E2 = self.edges[i].edge_position() - \
                defaults.preedge_safe_window_width
            else:
                E2 = ea[-1]
        else:
            E2 = E2           
        print \
        "Estimating the parameters of the background by the two area method"
        print "E1 = %s\t E2 = %s" % (E1, E2)

        try:
            estimation = two_area_powerlaw_estimation(self.hl, E1, E2)
            bg = self.__background_components[0]
            bg.A.already_set_map = np.ones(
                (self.hl.xdimension,self.hl.ydimension))
            bg.r.already_set_map = np.ones(
                (self.hl.xdimension, self.hl.ydimension))
            bg.r.map = estimation['r']
            bg.A.map = estimation['A']
            bg.charge_value_from_map(self.coordinates.ix,self.coordinates.iy)
        except ValueError:
            messages.warning(
            "The power law background parameters could not be estimated\n"
            "Try choosing an energy range for the estimation")

    def fit_edge(self,edgenumber,startenergy = None, **kwards):
        backup_channel_switches = self.channel_switches.copy()
        ea = self.hl.energy_axis[self.channel_switches]
        if startenergy is None:
            startenergy = ea[0]
        preedge_safe_window_width = defaults.preedge_safe_window_width
        # Declare variables
        edge = self.edges[edgenumber]
        if edge.intensity.twin is not None or edge.active is False or \
        edge.edge_position() < startenergy or edge.edge_position() > ea[-1]:
            return 1
        print "Fitting edge ", edge.name 
        last_index = len(self.edges) - 1
        i = 1
        twins = []
        print "Last edge index", last_index
        while edgenumber + i <= last_index and (
        self.edges[edgenumber+i].intensity.twin is not None or 
        self.edges[edgenumber+i].active is False):
            if self.edges[edgenumber+i].intensity.twin is not None:
                twins.append(self.edges[edgenumber+i])
            i+=1
        print "twins", twins
        print "next_edge_index", edgenumber + i
        if  (edgenumber + i) > last_index:
            nextedgeenergy = ea[-1]
        else:
            nextedgeenergy = self.edges[edgenumber+i].edge_position() - \
            preedge_safe_window_width

        # Backup the fsstate
        to_activate_fs = []
        for edge_ in [edge,] + twins:
            if edge_.fs_state is True and edge_.fslist.free is True:
                to_activate_fs.append(edge_)
        self.disable_fine_structure(to_activate_fs)
        
        # Smart Fitting

        print("Fitting region: %s-%s" % (startenergy,nextedgeenergy))

        # Without fine structure to determine delta
        edges_to_activate = []
        for edge_ in self.edges[edgenumber+1:]:
            if edge_.active is True and edge_.edge_position() >= nextedgeenergy:
                edge_.active = False
                edges_to_activate.append(edge_)
        print "edges_to_activate", edges_to_activate
        print "Fine structure to fit", to_activate_fs
        
        self.set_energy_region(startenergy, nextedgeenergy)
        if edge.freedelta is True:
            print "Fit without fine structure, delta free"
            edge.delta.free = True
            self.fit(**kwards)
            edge.delta.free = False
            print "delta = ", edge.delta.value
            self._touch()
        elif edge.intensity.free is True:
            print "Fit without fine structure"
            self.enable_fine_structure(to_activate_fs)
            self.remove_fine_structure_data(to_activate_fs)
            self.disable_fine_structure(to_activate_fs)
            self.fit(**kwards)

        if len(to_activate_fs) > 0:
            self.set_energy_region(startenergy, nextedgeenergy)
            self.enable_fine_structure(to_activate_fs)
            print "Fit with fine structure"
            self.fit(**kwards)
            
        self.enable_edges(edges_to_activate)
        # Recover the channel_switches. Remove it or make it smarter.
        self.channel_switches = backup_channel_switches

    def multifit(self, background_fit_E1 = None, mask = None, kind = "normal", 
                 fitter = "leastsq", charge_only_fixed = False, grad = False, 
                 autosave = False, **kwargs):
        if autosave is not None:
            fd, autosave_fn = tempfile.mkstemp(prefix = 'eelslab_autosave-', 
            dir = '.', suffix = '.par')
            os.close(fd)
            autosave_fn = autosave_fn[:-4]
            messages.information(
            "Autosaving each %s in file: %s.par" % (autosave, autosave_fn))
            messages.information(
            "When multifit finishes its job the file will be deleted")
        if mask is not None and \
        (np.shape(mask) != (self.hl.xdimension,self.hl.ydimension)):
           messages.warning_exit(
           "The mask must be an array with the same espatial dimensions as the" 
           "data cube")
        pbar = progressbar.progressbar(
        maxval = (self.hl.ydimension * self.hl.xdimension))
        i = 0
        for y in np.arange(0,self.hl.ydimension):
            for x in np.arange(0,self.hl.xdimension):
                if mask is None or mask[x,y] :
                    self.coordinates.ix = x
                    self.coordinates.iy = y
                    self.charge(only_fixed=charge_only_fixed)
#                    print '-'*40
#                    print "Fitting x=",self.coordinates.ix," y=",self.coordinates.iy
                    if kind  == "smart" :
                        self.smart_fit(background_fit_E1 = None,
                         fitter = fitter, **kwargs)
                    elif kind == "normal" :
                        self.fit(fitter = fitter, grad = grad, **kwargs)
                    if autosave == 'pixel':
                        try:
                            # Saving can fail, e.g., if the std was not present 
                            # due to a current leastsq bug
                            # Therefore we only try to save...
                            self.save_parameters2file(autosave_fn)
                        except:
                            pass
                    i += 1
                    pbar.update(i)
                if autosave == 'row':
                        try:
                            self.save_parameters2file(autosave_fn)
                        except:
                            pass
        pbar.finish()

        messages.information(
        'Removing the temporary file %s' % (autosave_fn + 'par'))
        os.remove(autosave_fn + '.par')

    def generate_chisq(self, degrees_of_freedom = 'auto') :
        if self.hl.variance is None:
            self.hl.estimate_variance()
        variance = self.hl.variance[self.channel_switches]
        differences = (self.model_cube - self.hl.data_cube)[self.channel_switches]
        self.chisq = np.sum(differences**2 / variance, 0)
        if degrees_of_freedom == 'auto' :
            self.red_chisq = self.chisq / \
            (np.sum(np.ones(self.hl.energydimension)[self.channel_switches]) \
            - len(self.p0) -1)
            print "Degrees of freedom set to auto"
            print "DoF = ", len(self.p0)
        elif type(degrees_of_freedom) is int :
            self.red_chisq = self.chisq / \
            (np.sum(np.ones(self.hl.energydimension)[self.channel_switches]) \
            - degrees_of_freedom -1)
        else:
            print "degrees_of_freedom must be on interger type."
            print "The red_chisq could not been calculated"
            
    def save_parameters2file(self,filename):
        """Save the parameters array in binary format"""
        value_array = None
        std_array = None
        asm_array = None
        for component in self:
            for param in component.parameters:
                if value_array is not None:
                    value_array=np.concatenate((value_array,
                    np.atleast_3d(param.map)),2)
                    std_array=np.concatenate((std_array,
                    np.atleast_3d(param.std_map)),2)
                    asm_array=np.concatenate((asm_array,
                    np.atleast_3d(param.already_set_map)),2)
                else:
                    value_array = np.atleast_3d(param.map)
                    std_array = np.atleast_3d(param.std_map)
                    asm_array = np.atleast_3d(param.already_set_map)
        np.savez(filename, 
                 value_array = value_array,
                 std_array = std_array,
                 asm_array = asm_array,)

    def load_parameters_from_file(self,filename):
        """Loads the parameters array from  a binary file written with the
        'save_parameters2file' function"""
        
        f = np.load(filename)
        counter = 0
        counter_asm = 0
        for component in self: # Cut the parameters list
            component.update_number_parameters()
            component.charge2map(f['value_array'][:, :,
            counter:counter+component.nparam])
            component.charge2map(f['std_array'][:, :,
            counter:counter+component.nparam], array = 'std')
            component.charge2map(f['asm_array'][:, :,
            counter_asm:counter_asm+len(component.parameters)], array = 'asm')
            counter += component.nparam
            counter_asm += len(component.parameters)
        if f['value_array'].shape[2] != counter:
            messages.warning(
            "The total number of parameters of the model is not equal to the " 
            "number of parameters in the file.\n"
            "Probably the model is broken\n"
            "Parameters in the file: %i\n Parameters in the model: %i" % 
            (f['value_array'].shape[2], counter))
        print "\n%s parameters charged from %s" % (counter, filename)
        self.charge()
    
    def quantify(self):
        elements = {}
        for edge in self.edges:
            if edge.active and edge.intensity.twin is None:
                element = edge._Edge__element
                subshell = edge._Edge__subshell
                if element not in elements:
                    elements[element] = {}
                elements[element][subshell] = edge.intensity.value
        # Print absolute quantification
        print
        print "Absolute quantification:"
        print "Elem.\tAreal density (atoms/nm**2)"
        for element in elements:
            if len(elements[element]) == 1:
                for subshell in elements[element]:
                    print "%s\t%f" % (element, elements[element][subshell])
            else:
                for subshell in elements[element]:
                    print "%s_%s\t%f" % (element, subshell, 
                    elements[element][subshell])
       
    def plot(self, auto_update_plot = True):
        """Plots the current spectrum to the screen and a map with a cursor to 
        explore the SI.
        """
        
        # If new coordinates are assigned
        self.hl.plot()
        hse = self.hl.hse
        l1 = hse.spectrum_plot.left_ax_lines[0]
        color = l1.line.get_color()
        l1.line_properties_helper(color, 'scatter')
        l1.set_properties()
        
        l2 = drawing.spectrum.SpectrumLine()
        l2.data_function = self.model2plot
        l2.line_properties_helper('blue', 'line')        
        # Add the line to the figure
          
        hse.spectrum_plot.add_line(l2)
        l2.plot()
        self.connect_parameters2update_plot()
        drawing.utils.on_window_close(hse.spectrum_plot.figure, 
                                      self.disconnect_parameters2update_plot)
        self.set_auto_update_plot(True)
        # TODO Set autoupdate to False on close