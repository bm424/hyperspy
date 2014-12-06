# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

import matplotlib.pyplot as plt
import matplotlib.widgets
import matplotlib.transforms as transforms
import numpy as np
import traits

from utils import on_figure_window_close
from hyperspy.misc.math_tools import closest_nice_number


class DraggablePatch(object):

    """
    """

    def __init__(self, axes_manager=None):
        """
        Add a cursor to ax.
        """
        self.axes_manager = axes_manager
        self.ax = None
        self.picked = False
        self.size = 1.
        self.color = 'red'
        self.__is_on = True
        self._2D = True  # Whether the cursor lives in the 2D dimension
        self.patch = None
        self.cids = list()
        self.blit = True
        self.background = None

    def is_on(self):
        return self.__is_on

    def set_on(self, value):
        if value is not self.is_on():
            if value is True:
                self.add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                for container in [
                        self.ax.patches,
                        self.ax.lines,
                        self.ax.artists,
                        self.ax.texts]:
                    if self.patch in container:
                        container.remove(self.patch)
                self.disconnect(self.ax)
            self.__is_on = value
            try:
                self.ax.figure.canvas.draw()
            except:  # figure does not exist
                pass
            else:
                self.ax = None

    def set_patch(self):
        pass
        # Must be provided by the subclass

    def add_patch_to(self, ax):
        self.set_patch()
        ax.add_artist(self.patch)
        self.patch.set_animated(hasattr(ax, 'hspy_fig'))

    def add_axes(self, ax):
        self.ax = ax
        canvas = ax.figure.canvas
        if self.is_on() is True:
            self.add_patch_to(ax)
            self.connect(ax)
            canvas.draw()

    def connect(self, ax):
        canvas = ax.figure.canvas
        self.cids.append(
            canvas.mpl_connect('motion_notify_event', self.onmove))
        self.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        self.cids.append(canvas.mpl_connect(
            'button_release_event', self.button_release))
        self.axes_manager.connect(self.update_patch_position)
        on_figure_window_close(ax.figure, self.close)

    def disconnect(self, ax):
        for cid in self.cids:
            try:
                ax.figure.canvas.mpl_disconnect(cid)
            except:
                pass
        self.axes_manager.disconnect(self.update_patch_position)

    def close(self, window=None):
        self.set_on(False)

    def onpick(self, event):
        self.picked = (event.artist is self.patch)

    def onmove(self, event):
        """This method must be provided by the subclass"""
        pass

    def update_patch_position(self):
        """This method must be provided by the subclass"""
        pass

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1:
            return
        if self.picked is True:
            self.picked = False

    def draw_patch(self, *args):
        if hasattr(self.ax, 'hspy_fig'):
            self.ax.hspy_fig._draw_animated()
        else:
            self.ax.figure.canvas.draw_idle()


class ResizebleDraggablePatch(DraggablePatch):

    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)
        self.size = 1.

    def set_size(self, size):
        self.size = size
        self.update_patch_size()

    def increase_size(self):
        self.set_size(self.size + 1)

    def decrease_size(self):
        if self.size > 1:
            self.set_size(self.size - 1)

    def update_patch_size(self):
        """This method must be provided by the subclass"""
        pass

    def on_key_press(self, event):
        if event.key == "+":
            self.increase_size()
        if event.key == "-":
            self.decrease_size()

    def connect(self, ax):
        DraggablePatch.connect(self, ax)
        canvas = ax.figure.canvas
        self.cids.append(canvas.mpl_connect('key_press_event',
                                            self.on_key_press))


class DraggableSquare(ResizebleDraggablePatch):

    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)

    def set_patch(self):
        self.calculate_size()
        self.calculate_position()
        self.patch = plt.Rectangle(
            self._position, self._xsize, self._ysize,
            animated=self.blit,
            fill=False,
            lw=2,
            ec=self.color,
            picker=True,)

    def calculate_size(self):
        xaxis = self.axes_manager.navigation_axes[0]
        yaxis = self.axes_manager.navigation_axes[1]
        self._xsize = xaxis.scale * self.size
        self._ysize = yaxis.scale * self.size

    def calculate_position(self):
        coordinates = np.array(self.axes_manager.coordinates[:2])
        self._position = coordinates - (
            self._xsize / 2., self._ysize / 2.)

    def update_patch_size(self):
        self.calculate_size()
        self.patch.set_width(self._xsize)
        self.patch.set_height(self._ysize)
        self.update_patch_position()

    def update_patch_position(self):
        self.calculate_position()
        self.patch.set_xy(self._position)
        self.draw_patch()

    def onmove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            xaxis = self.axes_manager.navigation_axes[0]
            yaxis = self.axes_manager.navigation_axes[1]
            wxindex = xaxis.value2index(event.xdata)
            wyindex = yaxis.value2index(event.ydata)
            if self.axes_manager.indices[1] != wyindex:
                try:
                    yaxis.index = wyindex
                except traits.api.TraitError:
                    # Index out of range, we do nothing
                    pass

            if self.axes_manager.indices[0] != wxindex:
                try:
                    xaxis.index = wxindex
                except traits.api.TraitError:
                    # Index out of range, we do nothing
                    pass


class ResizebleDraggableRectangle(ResizebleDraggablePatch):

    def __init__(self, axes_manager):
        super(ResizebleDraggableRectangle, self).__init__(axes_manager)
        self.xsize = 1
        self.ysize = 1
        self.bounds = []
        self.pick_on_frame = False
        self.pick_offset = (0,0)
        self.resize_color = 'green'
        self.resizers = []

    def set_size(self, size):
        self.xsize = size
        self.ysize = size
        self.update_patch_size()
        self._apply_geometry()

    def set_xsize(self, xsize):
        self.xsize = xsize
        self.update_patch_size()
        self._apply_geometry()

    def increase_xsize(self):
        self.set_xsize(self.xsize + 1)

    def decrease_xsize(self):
        if self.xsize >= 2:
            self.set_xsize(self.xsize - 1)

    def set_ysize(self, ysize):
        self.ysize = ysize
        self.update_patch_size()
        self._apply_geometry()

    def increase_ysize(self):
        self.set_ysize(self.ysize + 1)

    def decrease_ysize(self):
        if self.ysize >= 2:
            self.set_ysize(self.ysize - 1)

    def on_key_press(self, event):
        if event.key == "x":
            self.increase_xsize()
        elif event.key == "c":
            self.decrease_xsize()
        elif event.key == "y":
            self.increase_ysize()
        elif event.key == "u":
            self.decrease_ysize()
        else:
            super(ResizebleDraggableRectangle, self).on_key_press(event)


    def set_patch(self):
        self.calculate_size()
        self.calculate_position()
        self.calculate_bounds()
        self.patch = plt.Rectangle(
            self._position, self._xsize, self._ysize,
            animated=self.blit,
            fill=False,
            lw=2,
            ec=self.color,
            picker=True,)
        self.resizers = []
        dx = self._xsize / self.xsize
        dy = self._ysize / self.ysize
        
        p = self._position - (dx, dy)
        r = plt.Rectangle(p, dx, dy, animated=self.blit, fill=True, lw=0,
                          fc=self.resize_color, picker=True,)
        self.resizers.append(r)
        
        p = self._position + (self._xsize + dx, -dy)
        r = plt.Rectangle(p, dx, dy, animated=self.blit, fill=True, lw=0,
                          fc=self.resize_color, picker=True,)
        self.resizers.append(r)
        
        p = self._position + (-dx, self._ysize + dy)
        r = plt.Rectangle(p, dx, dy, animated=self.blit, fill=True, lw=0,
                          fc=self.resize_color, picker=True,)
        self.resizers.append(r)
        
        p = self._position + (self._xsize + dx, self._ysize + dy)
        r = plt.Rectangle(p, dx, dy, animated=self.blit, fill=True, lw=0,
                          fc=self.resize_color, picker=True,)
        self.resizers.append(r)

    def calculate_size(self):
        xaxis = self.axes_manager.navigation_axes[0]
        yaxis = self.axes_manager.navigation_axes[1]
        self._xsize = xaxis.scale * self.xsize
        self._ysize = yaxis.scale * self.ysize

    def calculate_position(self):
        coordinates = np.array(self.axes_manager.coordinates[:2])
        self._position = coordinates - (
            self._xsize / (2.*self.xsize),
            self._ysize / (2.*self.ysize))

    def calculate_bounds(self):
        position = self._position
        x0 = position[0]
        x1 = position[0] + self._xsize
        y0 = position[1]
        y1 = position[1] + self._ysize
        self.bounds = [x0,y0,x1,y1]

    def update_patch_size(self):
        self.calculate_size()
        self.patch.set_width(self._xsize)
        self.patch.set_height(self._ysize)
        self.calculate_bounds()
        self.draw_patch()

    def update_patch_position(self):
        self.calculate_position()
        self.patch.set_xy(self._position)
        self.calculate_bounds()
        self.draw_patch()
        
    def update_patch_geometry(self):
        self.calculate_size()
        self.patch.set_width(self._xsize)
        self.patch.set_height(self._ysize)
        self.calculate_position()
        self.patch.set_xy(self._position)
        self.calculate_bounds()
        self.draw_patch()
        
            
    def _apply_geometry(self, x1=None, y1=None):
        xaxis = self.axes_manager.navigation_axes[0]
        yaxis = self.axes_manager.navigation_axes[1]
        if x1 is None: x1 = xaxis.index
        if y1 is None: y1 = yaxis.index
        x2 = x1 + self.xsize
        y2 = y1 + self.ysize
        if np.abs(x1 - x2) < 2:
            xaxis.slice = None
        else:
            xaxis.slice = slice(x1, x2)
        if np.abs(y1 - y2) < 2:
            yaxis.slice = None
        else:
            yaxis.slice = slice(y1, y2)
            
        try:
            xaxis.index = x1
        except traits.api.TraitError:
            # Index out of range, we do nothing
            pass
        try:
            yaxis.index = y1
        except traits.api.TraitError:
            # Index out of range, we do nothing
            pass

    def onpick(self, event):
        super(ResizebleDraggableRectangle, self).onpick(event)
        
        if event.artist in self.resizers:
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata
            dx = self._xsize / self.xsize
            dy = self._ysize / self.ysize
            xaxis = self.axes_manager.navigation_axes[0]
            yaxis = self.axes_manager.navigation_axes[1]
            ix = xaxis.value2index(x + 0.5*dx)
            iy = yaxis.value2index(y + 0.5*dy)
            self.pick_offset = (ix-xaxis.index, iy-yaxis.index)

            corner = self.resizers.index(event.artist)
            self.pick_on_frame = corner
        else:
            self.pick_on_frame = False
        
    def onmove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            xaxis = self.axes_manager.navigation_axes[0]
            yaxis = self.axes_manager.navigation_axes[1]
            dx = self._xsize / self.xsize
            dy = self._ysize / self.ysize
            ix = xaxis.value2index(event.xdata + 0.5*dx)
            iy = yaxis.value2index(event.ydata + 0.5*dy)
            ibounds = [xaxis.index, yaxis.index, xaxis.index + self.xsize,
                       yaxis.index + self.ysize]
            if self.pick_on_frame is not False:
                posx = None
                posy = None
                corner = self.pick_on_frame
                if corner % 2 == 0: # Left side start
                    if ix > ibounds[2]:    # flipped to right
                        posx = ibounds[2]
                        self.xsize = ix - ibounds[2]
                    else:
                        posx = ix
                        self.xsize = ibounds[2] - ix
                else:   # Right side start
                    if ix < ibounds[0]:  # Flipped to left
                        posx = ix
                        self.xsize = ibounds[0] - ix
                    else:
                        self.xsize = ix - ibounds[0]
                if corner // 2 == 0: # Top side start
                    if iy > ibounds[3]:    # flipped to botton
                        posy = ibounds[3]
                        self.ysize = iy - ibounds[3]
                    else:
                        posy = iy
                        self.ysize = ibounds[3] - iy
                else:   # Bottom side start
                    if iy < ibounds[1]:  # Flipped to top
                        posy = iy
                        self.ysize = ibounds[1] - iy
                    else:
                        self.ysize = iy - ibounds[1]
                if self.xsize < 1:
                    self.xsize = 1
                if self.ysize < 1:
                    self.ysize = 1
                self._apply_geometry(posx, posy)
                self.update_patch_geometry()
            else:
                ix -= self.pick_offset[0]
                iy -= self.pick_offset[1]
                self._apply_geometry(ix, iy)
                self.update_patch_position()
            


class DraggableHorizontalLine(DraggablePatch):

    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)
        self._2D = False
        # Despise the bug, we use blit for this one because otherwise the
        # it gets really slow

    def update_patch_position(self):
        if self.patch is not None:
            self.patch.set_ydata(self.axes_manager.coordinates[0])
            self.draw_patch()

    def set_patch(self):
        ax = self.ax
        self.patch = ax.axhline(
            self.axes_manager.coordinates[0],
            color=self.color,
            picker=5)

    def onmove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            try:
                self.axes_manager.navigation_axes[0].value = event.ydata
            except traits.api.TraitError:
                # Index out of range, we do nothing
                pass


class DraggableVerticalLine(DraggablePatch):

    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)
        self._2D = False

    def update_patch_position(self):
        if self.patch is not None:
            self.patch.set_xdata(self.axes_manager.coordinates[0])
            self.draw_patch()

    def set_patch(self):
        ax = self.ax
        self.patch = ax.axvline(self.axes_manager.coordinates[0],
                                color=self.color,
                                picker=5)

    def onmove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            try:
                self.axes_manager.navigation_axes[0].value = event.xdata
            except traits.api.TraitError:
                # Index out of range, we do nothing
                pass


class DraggableLabel(DraggablePatch):

    def __init__(self, axes_manager):
        DraggablePatch.__init__(self, axes_manager)
        self._2D = False
        self.string = ''
        self.y = 0.9
        self.text_color = 'black'
        self.bbox = None

    def update_patch_position(self):
        if self.patch is not None:
            self.patch.set_x(self.axes_manager.coordinates[0])
            self.draw_patch()

    def set_patch(self):
        ax = self.ax
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        self.patch = ax.text(
            self.axes_manager.coordinates[0],
            self.y,  # Y value in axes coordinates
            self.string,
            color=self.text_color,
            picker=5,
            transform=trans,
            horizontalalignment='right',
            bbox=self.bbox,
            animated=self.blit)


class Scale_Bar():

    def __init__(self, ax, units, pixel_size=None, color='white',
                 position=None, max_size_ratio=0.25, lw=2, lenght=None,
                 animated=False):
        """Add a scale bar to an image.

        Parameteres
        -----------
        ax : matplotlib axes
            The axes where to draw the scale bar.
        units : string
        pixel_size : {None, float}
            If None the axes of the image are supposed to be calibrated.
            Otherwise the pixel size must be specified.
        color : a valid matplotlib color
        position {None, (float, float)}
            If None the position is automatically determined.
        max_size_ratio : float
            The maximum size of the scale bar in respect to the
            lenght of the x axis
        lw : int
            The line width
        lenght : {None, float}
            If None the lenght is automatically calculated using the
            max_size_ratio.

        """

        self.animated = animated
        self.ax = ax
        self.units = units
        self.pixel_size = pixel_size
        self.xmin, self.xmax = ax.get_xlim()
        self.ymin, self.ymax = ax.get_ylim()
        self.text = None
        self.line = None
        self.tex_bold = False
        if lenght is None:
            self.calculate_size(max_size_ratio=max_size_ratio)
        else:
            self.lenght = lenght
        if position is None:
            self.position = self.calculate_line_position()
        else:
            self.position = position
        self.calculate_text_position()
        self.plot_scale(line_width=lw)
        self.set_color(color)

    def get_units_string(self):
        if self.tex_bold is True:
            if (self.units[0] and self.units[-1]) == '$':
                return r'$\mathbf{%g\,%s}$' % \
                    (self.lenght, self.units[1:-1])
            else:
                return r'$\mathbf{%g\,}$\textbf{%s}' % \
                    (self.lenght, self.units)
        else:
            return r'$%g\,$%s' % (self.lenght, self.units)

    def calculate_line_position(self, pad=0.05):
        return ((1 - pad) * self.xmin + pad * self.xmax,
                (1 - pad) * self.ymin + pad * self.ymax)

    def calculate_text_position(self, pad=1 / 100.):
        ps = self.pixel_size if self.pixel_size is not None else 1
        x1, y1 = self.position
        x2, y2 = x1 + self.lenght / ps, y1

        self.text_position = ((x1 + x2) / 2.,
                              y2 + (self.ymax - self.ymin) / ps * pad)

    def calculate_size(self, max_size_ratio=0.25):
        ps = self.pixel_size if self.pixel_size is not None else 1
        size = closest_nice_number(ps * (self.xmax - self.xmin) *
                                   max_size_ratio)
        self.lenght = size

    def remove(self):
        if self.line is not None:
            self.ax.lines.remove(self.line)
        if self.text is not None:
            self.ax.texts.remove(self.text)

    def plot_scale(self, line_width=1):
        self.remove()
        ps = self.pixel_size if self.pixel_size is not None else 1
        x1, y1 = self.position
        x2, y2 = x1 + self.lenght / ps, y1
        self.line, = self.ax.plot([x1, x2], [y1, y2],
                                  linestyle='-',
                                  lw=line_width,
                                  animated=self.animated)
        self.text = self.ax.text(*self.text_position,
                                 s=self.get_units_string(),
                                 ha='center',
                                 size='medium',
                                 animated=self.animated)
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.figure.canvas.draw()

    def set_position(self, x, y):
        self.position = x, y
        self.calculate_text_position()
        self.plot_scale(line_width=self.line.get_linewidth())

    def set_color(self, c):
        self.line.set_color(c)
        self.text.set_color(c)
        self.ax.figure.canvas.draw_idle()

    def set_lenght(self, lenght):
        color = self.line.get_color()
        self.lenght = lenght
        self.calculate_scale_size()
        self.calculate_text_position()
        self.plot_scale(line_width=self.line.get_linewidth())
        self.set_color(color)

    def set_tex_bold(self):
        self.tex_bold = True
        self.text.set_text(self.get_units_string())
        self.ax.figure.canvas.draw_idle()


def in_interval(number, interval):
    if number >= interval[0] and number <= interval[1]:
        return True
    else:
        return False


class ModifiableSpanSelector(matplotlib.widgets.SpanSelector):

    def __init__(self, ax, **kwargs):
        matplotlib.widgets.SpanSelector.__init__(
            self, ax, direction='horizontal', useblit=False, **kwargs)
        # The tolerance in points to pick the rectangle sizes
        self.tolerance = 1
        self.on_move_cid = None
        self.range = None

    def release(self, event):
        """When the button is realeased, the span stays in the screen and the
        iteractivity machinery passes to modify mode"""
        if self.pressv is None or (self.ignore(event) and not self.buttonDown):
            return
        self.buttonDown = False
        self.update_range()
        self.onselect()
        # We first disconnect the previous signals
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)

        # And connect to the new ones
        self.cids.append(
            self.canvas.mpl_connect('button_press_event', self.mm_on_press))
        self.cids.append(
            self.canvas.mpl_connect('button_release_event', self.mm_on_release))
        self.cids.append(
            self.canvas.mpl_connect('draw_event', self.update_background))

    def mm_on_press(self, event):
        if (self.ignore(event) and not self.buttonDown):
            return
        self.buttonDown = True

        # Calculate the point size in data units
        invtrans = self.ax.transData.inverted()
        x_pt = abs((invtrans.transform((1, 0)) -
                    invtrans.transform((0, 0)))[0])

        # Determine the size of the regions for moving and stretching
        rect = self.rect
        self.range = rect.get_x(), rect.get_x() + rect.get_width()
        left_region = self.range[0] - x_pt, self.range[0] + x_pt
        right_region = self.range[1] - x_pt, self.range[1] + x_pt
        middle_region = self.range[0] + x_pt, self.range[1] - x_pt

        if in_interval(event.xdata, left_region) is True:
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_left)
        elif in_interval(event.xdata, right_region):
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_right)
        elif in_interval(event.xdata, middle_region):
            self.pressv = event.xdata
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_rect)
        else:
            return

    def update_range(self):
        self.range = (self.rect.get_x(),
                      self.rect.get_x() + self.rect.get_width())

    def move_left(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        # Do not move the left edge beyond the right one.
        if event.xdata >= self.range[1]:
            return
        width_increment = self.range[0] - event.xdata
        self.rect.set_x(event.xdata)
        self.rect.set_width(self.rect.get_width() + width_increment)
        self.update_range()
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
        self.update()

    def move_right(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        # Do not move the right edge beyond the left one.
        if event.xdata <= self.range[0]:
            return
        width_increment = \
            event.xdata - self.range[1]
        self.rect.set_width(self.rect.get_width() + width_increment)
        self.update_range()
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
        self.update()

    def move_rect(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        x_increment = event.xdata - self.pressv
        self.rect.set_x(self.rect.get_x() + x_increment)
        self.update_range()
        self.pressv = event.xdata
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
        self.update()

    def mm_on_release(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        self.buttonDown = False
        self.canvas.mpl_disconnect(self.on_move_cid)
        self.on_move_cid = None

    def turn_off(self):
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        if self.on_move_cid is not None:
            self.canvas.mpl_disconnect(cid)
        self.ax.patches.remove(self.rect)
        self.ax.figure.canvas.draw()
