import pyqtgraph as pg
import numpy as np

from fflib import ff_time
from MagPy4.dynBase import GradLegend
import pyqtgraph as pg
from MagPy4.plotBase import DateAxis, MagPyPlotItem, MagPyAxisItem
from PyQt5 import QtCore, QtGui, QtWidgets

class ColorLabel(pg.LabelItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def set_color(self, color):
        self.setAttr('color', color)
        self.setText(self.text)

class HiddenAxis(MagPyAxisItem):
    # Axis item that hides all lines and only paints text items at ticks
    def __init__(self, interpolator, orientation='top', ofst=0):
        self.ofst = ofst
        self.matchedTicks = None
        self.interpolater = interpolator
        super().__init__(orientation=orientation)

    def tickStrings(self, values, scale, spacing):
        # Interpolate values along values that are within a valid range
        values = np.array([val + self.ofst for val in values])
        interp_data = self.interpolater(values)

        ## Replace out of range values with None
        string_arr = interp_data[~np.isnan(interp_data)]

        # Convert values to strings
        strings = []
        for val in string_arr:
            if val is None:
                strings.append('')
            else:
                txt = str(np.round(val, decimals=4))
                strings.append(txt)

        return strings

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        pen, p1, p2 = axisSpec
        p.setPen(pen)

        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)
        p.translate(0.5, 0)
        
        ## Draw all text
        if self.getTickFont() is not None:
            p.setFont(self.getTickFont())
        p.setPen(self.pen())
        for rect, flags, text in textSpecs:
            p.drawText(rect, flags & QtCore.Qt.AlignCenter, text)

    def matchTicks(self, ticks):
        self._set_matched_ticks(ticks)
        self.ignore_bounds = True
    
    def _set_matched_ticks(self, ticks):
        levels = []
        for level in ticks:
            spacing, values = level
            strings = self.tickStrings(values, None, None)
            levels.append(list(map(tuple, zip(values, strings))))
        self.setTicks(levels)

    def tickValues(self, *args, **kwargs):
        if self.matchedTicks is not None:
            res = self.matchedTicks
        else:
            res =  super().tickValues(*args, **kwargs)
        return res

class HiddenPlotAxis(MagPyPlotItem):
    def __init__(self, interpolator, ofst=0, orientation='top'):
        self.orientation = orientation
        axItems = {orientation:HiddenAxis(interpolator, orientation=orientation, ofst=ofst)}
        super().__init__(axisItems=axItems)
        self.layout.setVerticalSpacing(0.0)
        for i in range(4):
            self.layout.setRowStretchFactor(i, 1)
            self.layout.itemAt(i, 1).setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.getAxis('left').setHeight(0)
        self.getAxis('right').setHeight(0)
        self.vb.setMaximumHeight(0)
        self.vb.setMinimumHeight(0)
        self.vb.setMouseEnabled(y=False)
        self.setMaximumHeight(500)
        op = {'top':'bottom', 'bottom':'top'}
        self.hideAxis(op[orientation])
        self.showAxis(orientation)

    def setTickLevels(self, ticks):
        ax = self.getAxis(self.orientation)
        ax.matchTicks(ticks)

class Grid():
    def __init__(self, shape=None) -> None:
        self.items = []
        self.constraints = []
        self.spacings = [2, 5]
        if shape is not None:
            rows, cols = shape
            for i in range(rows-1):
                self.add_row()
            for i in range(cols-1):
                self.add_col()

    def shape(self):
        rows = len(self.items)
        if rows > 0:
            cols = len(self.items[0])
        else:
            cols = 0
        return (rows, cols)

    def add_row(self, index=None):
        # Create a new row of empty items and insert into grid
        rows, cols = self.shape()
        index = rows if index is None else index
        new_row = [None] * cols
        self.items.insert(index, new_row)

    def remove_row(self, index=None, delete=False):
        # Remove items in row
        rows, cols = self.shape()
        index = (rows - 1) if index is None else index
        v = self.items.pop(index)
        for e in v:
            if e is not None:
                self._remove_item(e)
        
        # Delete items in row
        if delete:
            for e in v:
                if e is not None:
                    e.deleteLater()

        self._update_state()        

    def add_col(self, index=None):
        rows, cols = self.shape()
        index = cols if index is None else index
        for i in range(rows):
            self.items[i].insert(index, None)

    def remove_col(self, index=None, delete=False):
        rows, cols = self.shape()
        index = (cols - 1) if index is None else index
        for i in range(rows):
            v = self.items[i].pop(index)
            self._remove_item(v)    
    
    def remove_item(self, row, col, delete=True):
        item = self.items[row][col]
        if item is not None:
            self._remove_item(item)
        self.items[row][col] = None

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return self.items[index]
        else:
            return [row[index[1]] for row in self.items]

    def __setitem__(self, index, item):
        shape = self.shape()
        if isinstance(index, (int, slice)):
            self.items[index] = item
        else:
            for row, row_item in zip(self.items, item):
                row[index[1]] = row_item
        self._reset_size(shape)
        self._update_state()

    def _reset_size(self, shape):
        # Adjust size so trailing empty rows and columns are clipped
        rows, cols = shape
        curr_rows = len(self.items)
        if rows < curr_rows:
            for i in range(curr_rows-rows):
                self.add_row()
        for item_row in self.items:
            curr_cols = len(item_row)
            if curr_cols < cols:
                item_row.extend([None]*(cols-curr_cols))
    
    def _update_state(self):
        pass
    
    def _remove_item(self, v):
        pass

    def update_shape(self):
        ncols = max(list(map(len, self.items)))
        for row in self.items:
            if len(row) < ncols:
                row.extend([None]*(ncols-len(row)))

def get_axis_width(ax):
    ''' Calculates the width of the plot axis '''
    width = ax.textWidth if ax.style['showValues'] else 1
    width += ax.style['tickTextOffset'][0] if ax.style['showValues'] else 0
    width += max(0, ax.style['tickLength'])
    width += 5
    if ax.label.isVisible():
        width += ax.label.boundingRect().height() * 1.0 ##
    return width

def get_axis_height(ax):
    ''' Calculates the height of the plot axis'''
    height = ax.textHeight if ax.style['showValues'] else 1
    height += ax.style['tickTextOffset'][1] if ax.style['showValues'] else 0
    height += max(0, ax.style['tickLength'])
    if ax.label.isVisible():
        height += ax.label.boundingRect().height() * 1.0
    return height

def set_axis_height(ax, w):
    ax.setHeight(w)

def set_axis_width(ax, w):
    ax.setWidth(w)

class JustifiedText(pg.LabelItem):
    def __init__(self, text, parent=None, angle=0, justify='center', **args):
        self.justify = justify
        super().__init__(text=text, parent=parent, angle=angle, 
            justify=justify, **args)
    
    def resizeEvent(self, ev):
        return super().resizeEvent(ev)
        
    def sizeHint(self, hint, *args):
        if hint not in self._sizeHint:
            return QtCore.QSizeF(0, 0)
        return QtCore.QSizeF(*self._sizeHint[hint])

class AxisGrid(Grid):
    ''' Grid made of additional axes to be displayed above or below plots '''
    def __init__(self, parent) -> None:
        self.parent = parent
        self.axes_labels = []
        super().__init__(shape=(1, 0))
    
    def count(self):
        ''' Returns the number of valid items in grid'''
        items = self[:,0]
        return len([e for e in items if e is not None])

    def add_axis(self, label, interpolator, ofst=0):
        ''' Sets up hidden plot axis and label item '''
        ax = HiddenPlotAxis(interpolator, ofst=ofst)
        label_item = JustifiedText(label, justify='right')
        self.add_row()
        self[-1] = [label_item, ax]
        self.axes_labels.append(label)

        return [label_item, ax]

    def update_ticks(self, ticks):
        ''' Matches ticks on all axes to given tick levels '''
        for ax in self[:,1]:
            if ax is not None:
                ax.setTickLevels(ticks)
    
    def remove_axis(self, label):
        ''' Removes an axis with the given label if it exists '''
        index = self.axes_labels.index(label)
        row = self.items.pop(index)
        self.axes_labels.pop(index)
        return [v for v in row if v is not None]
    
    def set_x_range(self, t0, t1):
        ''' Sets the range for all axes items '''
        for ax in self[:,1]:
            if ax is not None:
                ax.setXRange(t0, t1, 0.0)

    def get_labels(self):
        return self.axes_labels

class PlotGrid(Grid, QtGui.QGraphicsGridLayout):
    def __init__(self, shape=(1,1)) -> None:
        Grid.__init__(self, shape)
        QtGui.QGraphicsGridLayout.__init__(self)
        self.border = (0, 255, 0)
        self.time_label = None
        self.axis_grids = {'top': AxisGrid(self), 'bottom': AxisGrid(self)}
        self.ref_plot = None

        self.stretch = None

        self.wheel_enabled = [True, False]
        self._changing_signal = False
        self.column_bounds = {0 : (10, 70), 2 : (5, 50), 3 : (0, 10)}
        self.itemBorders = {}
        self.setBorder(self.border)

    def setBorder(self, *args, **kwds):
        """
        Set the pen used to draw border between cells.
        
        See :func:`mkPen <pyqtgraph.mkPen>` for arguments.        
        """
        self.border = pg.mkPen(*args, **kwds)

        for borderRect in self.itemBorders.values():
            borderRect.setPen(self.border)


    def set_height_factors(self, s):
        self.stretch = s
    
    def get_height_factors(self):
        return self.stretch

    def update_grid_range(self, vb, range):
        ''' Updates x range for all plots and axis grids '''
        plots = self.get_plots()
        for plot in plots:
            plot.blockSignals(True)
            plot.setXRange(*range, 0.0)
            plot.blockSignals(False)

        self._update_axis_grid_range(vb, range)
        parent = self.parentLayoutItem()
        parent.sigXRangeChanged.emit(range)
        parent.update_y_ranges()

    def set_x_range(self, t0, t1):
        self.update_grid_range(None, (t0, t1))
    
    def set_x_lim(self, t0, t1):
        plots = self.get_plots()
        for plot in plots:
            plot.setLimits(xMin=t0, xMax=t1)

    def add_axis(self, label, interpolator, loc='top'):
        ax_grid = self.axis_grids[loc]
        items = ax_grid.add_axis(label, interpolator)
        for elem in items:
            self.addChildLayoutItem(elem)
            self._connect_plot(elem, hidden=True)

        # Make sure time label in correct location
        self._adjust_time_label()

        # Adjust ticks
        ax = self.ref_plot.getAxis('bottom')
        levels = ax.get_levels()
        if levels is not None:
            self._update_axis_grids(levels)
            self._update_axis_grid_range(None, ax.range)

    def remove_axis(self, label, loc='top'):
        items = self.axis_grids[loc].remove_axis(label)
        for elem in items:
            elem.setParentItem(None)
            self._disconnect_plot(elem)

        self._adjust_time_label()

    def _update_axis_grids(self, levels):
        for key in self.axis_grids:
            self.axis_grids[key].update_ticks(levels)
        label = self.ref_plot.getAxis('bottom').get_label()
        self._update_left_time_label(label)
    
    def _update_axis_grid_range(self, vb, range):
        for key in self.axis_grids:
            self.axis_grids[key].set_x_range(*range)
    
    def _update_left_time_label(self, label):
        if self.time_label:
            self.time_label.setText(label)
    
    def left_time_label(self, val=True):
        if val:
            if self.time_label is None:
                self.time_label = JustifiedText('HH:MM:SS', justify='right')
                self.addChildLayoutItem(self.time_label)
        else:
            if self.time_label:
                self.time_label.setParentItem(None)
                self.time_label.deleteLater()
                self.time_label = None
    
    def _link_last_plot(self):
        ''' Links bottom plot changes so all grids are properly
            updated and unlinks any previously linked plots 
        '''
        # Find last plot item
        plots = []
        for row in self.items:
            for e in row:
                if isinstance(e, MagPyPlotItem):
                    plots.append(e)
        
        # Make sure style is properly set for all plots
        for plot in plots:
            for ax in ['top', 'bottom']:
                axis = plot.getAxis(ax)
                axis.setStyle(showValues=False)
                if ax == 'bottom':
                    axis.showLabel(False)

            vb = plot.getViewBox()
            vb.enable_yscroll(self.wheel_enabled[1])

        if plots[-1] != self.ref_plot:
            # Unlink previously linked plot
            parent = self.parentLayoutItem()
            if self.ref_plot is not None:
                ax = self.ref_plot.getAxis('bottom')
                ax.ticksChanged.disconnect(self._update_axis_grids)
                ax.axisClicked.disconnect(parent.openPlotAppr)
                ax.setCursor(QtCore.Qt.ArrowCursor)
                plots[-1].setXRange(*self.ref_plot.getAxis('bottom').range, 0.0)

            # Link bottom plot
            self.ref_plot = plots[-1]
            ax = self.ref_plot.getAxis('bottom')
            ax.ticksChanged.connect(self._update_axis_grids)
            ax.axisClicked.connect(parent.openPlotAppr)
            ax.setCursor(QtCore.Qt.PointingHandCursor)

        # Make sure bottom plot displays values
        plots[-1].getAxis('bottom').setStyle(showValues=True)

        # Adjust time label axis grids are enabled
        self._adjust_time_label()
    
    def _adjust_time_label(self):
        lower_count = self.axis_grids['bottom'].count()
        left = (lower_count > 0)
        self.left_time_label(left)
        self.ref_plot.getAxis('bottom').showLabel((not left))

    def _connect_plot(self, plot, hidden=False):
        parent = plot.parentLayoutItem()
        new_plot = (parent is None) or (hidden)
        if new_plot and isinstance(plot, MagPyPlotItem):
            plot.sigXRangeChanged.connect(self.update_grid_range)

    def _update_time_loc(self):
        ax_grid = self.axis_grids['bottom']
        if ax_grid.count() > 0:
            self.left_time_label(True)
        else:
            self.left_time_label(False)
    
    def _update_state(self):
        for row in self.items:
            for item in row:
                if item is not None:
                    self._connect_plot(item)
                    self.addChildLayoutItem(item)

        self._link_last_plot()
        self.parentLayoutItem()._update_state()
    
    def _disconnect_plot(self, plot):
        if isinstance(plot, MagPyPlotItem):
            plot.sigXRangeChanged.disconnect(self.update_grid_range)

    def _remove_item(self, v):
        v.setParentItem(None)
        self._disconnect_plot(v)
    
    def _align_colorbars_helper(self, grid, keys):
        func_map = {
            0 : get_axis_height,
            1 : get_axis_width
        }

        set_func_map = {
            0 : set_axis_height,
            1 : set_axis_width
        }

        if 'top' in keys:
            top = True
            get_func = func_map[0]
            set_func = set_func_map[0]
        else:
            top = False
            get_func = func_map[1]
            set_func = set_func_map[1]

        n = grid.shape()[0] if top else grid.shape()[1]
        row_heights = []
        plot_offsets = []
        for i in range(n):
            if top:
                row = grid.items[i]
            else:
                row = grid[:,i]

            row_plots = []
            row_legends = []
            for item in row:
                if isinstance(item, (MagPyPlotItem, HiddenPlotAxis)):
                    row_plots.append(item)
                elif isinstance(item, GradLegend):
                    row_legends.append(item)
            
            bottom_height = 0
            top_height = 0
            for plot in row_plots:
                curr_top_height = get_func(plot.getAxis(keys[0])) 
                curr_botm_height = get_func(plot.getAxis(keys[1]))
                top_height = max(top_height, curr_top_height)
                bottom_height = max(bottom_height, curr_botm_height)
            
            for plot in row_plots:
                set_func(plot.getAxis(keys[0]), top_height)
                set_func(plot.getAxis(keys[1]), bottom_height)

            if top:
                for plot in row_plots:
                    title_height = 0
                    if plot.titleLabel is not None and top:
                        title_height = max(0, plot.titleLabel.boundingRect().height())
                    top_height += title_height

            row_heights.append((top_height, bottom_height))  
            plot_offsets.append((top_height, bottom_height))
        
        return row_heights, plot_offsets

    def _axis_grid_sizes(self):
        ''' Computes the minimum sizes for axis grids in layout '''
        grid_mins = [None, None]
        i = 0
        for loc in ['top', 'bottom']:
            ax_grid = self.axis_grids[loc]
            if ax_grid.count() > 0:
                size_grid = self._get_size_grid(ax_grid)

                # Get min height for each row
                n = ax_grid.count()
                heights = []
                for j in range(n):
                    row = ax_grid[j]
                    h = max([e.height() for e in row if e is not None])
                    heights.append(h)
                
                # Get min width for each col
                widths = []
                for j in range(2):
                    col = ax_grid[:,j][:n]
                    w = max([e.width() for e in col])
                    widths.append(w)
                
                grid_mins[i] = (widths, sum(heights), n)
                    
            i += 1

        return grid_mins

    def _align_colorbars(self):
        # Determine the number of items in grid
        grid = self.items[:]
        top_ofst = 0
        if self.axis_grids['top'].count() > 0:
            n = self.axis_grids['top'].count()
            grid = self.axis_grids['top'].items[:n] + grid
            top_ofst = n

        botm_ofst = len(grid)
        if self.axis_grids['bottom'].count() > 0:
            n = self.axis_grids['bottom'].count()
            grid = grid + self.axis_grids['bottom'].items[:n]
            botm_ofst = len(grid) - n
        
        # Create a copy of grid object
        grid_obj = Grid()
        grid_obj[:] = grid[:]
        grid_obj.update_shape()

        # Align vertical and horizontal edges of colorbars
        row_heights, plot_offsets = self._align_colorbars_helper(grid_obj, ['top', 'bottom'])
        col_widths, col_offsets = self._align_colorbars_helper(grid_obj, ['left', 'right'])
        return row_heights[top_ofst:botm_ofst], plot_offsets[top_ofst:botm_ofst]

    def _get_size_grid(self, grid=None, min=False):
        constraint = QtCore.Qt.MinimumSize
        if not min:
            constraint = QtCore.Qt.MaximumSize

        if grid is None:
            grid = self.items

        rows = []
        for row in grid:
            row_sizes = []
            for elem in row:
                if elem is not None:
                    size = elem.sizeHint(constraint)
                else:
                    size = QtCore.QSizeF(0, 0)
                row_sizes.append(size)
            rows.append(row_sizes)
        
        size_grid = Grid(self.shape())
        size_grid[:] = rows
        return size_grid
    
    def _readjust_space(self, total_space, n, constraints, spacing=2):
        # Calculate the space available without spacing between items
        spacer_space = spacing * max(n-1, 0)
        true_space = total_space - spacer_space
        even_space = true_space / n

        # Extract the space ranges for each column
        min_space = [max(low, 0) for low, high in constraints]
        max_space = [max(high, 0) for low, high in constraints]

        # Initialize spacing so all columns have equal widths
        spaces = [even_space] * n
        adjustments = [0]*n
        
        # Calculate adjustment needed for each column so it is
        # not too large for the items
        loss_count = 0
        for j in range(n):
            val = max_space[j]
            if val < spaces[j]:
                adjustments[j] = val - spaces[j]
                loss_count += 1
    
        # Calculate adjustments needed so each column has
        # enough space for its items
        for j in range(n):
            val = min_space[j]
            if val > spaces[j]:
                adjustments[j] = val - spaces[j]
        
        # Determine the amount of space leftover for adjustments
        adjust_width = sum(adjustments)
        adjust_n = n - loss_count

        # Adjust each column space with calculated values above
        # and if no adjustments needed, make sure leftover space
        # is evenly distributed amongst rest
        for i in range(n):
            if adjustments[i] != 0:
                spaces[i] += adjustments[i]
            if adjust_n > 0 and adjustments[i] >= 0:
                even_adjust = -adjust_width / adjust_n
                spaces[i] += even_adjust
        
        return spaces
    
    def _adjust_row_heights(self, total_space, n, axis_heights, spacing=2, stretch=None):
        ''' Computes the height for each row using the total space and
            given stretch factors
        '''
        # Compute the space available for items
        true_space = total_space - (max(n-1, 0)*spacing)
        true_space -= sum([a+b for a, b in axis_heights])

        # Compute the fractional values of spacing assigned to each plot
        if stretch is not None:
            total_stretch = sum(stretch)
            fracs = [s/total_stretch for s in stretch]
        else:
            fracs = [1/n] * n
        
        # Compute the spacing for each plot (adjusted for axis heights)
        spaces = []
        for axis_height, frac in zip(axis_heights, fracs):
            a, b = axis_height
            space = (true_space * frac) + a + b
            spaces.append(space)
        
        return spaces

    def _set_geometries(self, rect, grid, col_widths, row_heights, col_space, row_space):
        rects = self._get_geometries(rect, grid, col_widths, row_heights, col_space, row_space)
        self._apply_geometries(rects, grid)

    def _get_geometries(self, rect, grid, col_widths, row_heights, col_space, row_space):
        ''' Given a rect of space available and any constraints, compute 
            geometry rects for item in grid
        '''
        rects = np.zeros((*grid.shape(), 4))
        rows, cols = len(row_heights), len(col_widths)

        # Adjust column widths
        for j in range(rows):
            x = 0
            for i in range(cols):
                width = col_widths[i]
                rects[j][i][0] = x
                rects[j][i][2] = x + width
                x += width + col_space

        # Adjust column heights
        y = rect.top()
        for i in range(rows):
            height = row_heights[i]
            for j in range(cols):
                rects[i][j][1] = y
                rects[i][j][3] = y + height

            y += height + row_space
        
        return rects

    def _apply_geometries(self, rects, grid):
        ''' Given the array of rects for each item, this function
            sets the geometry for each item in the grid
        '''
        rows, cols = len(grid.items), len(grid[0])
        for i in range(rows):
            for j in range(cols):
                size = rects[i][j]
                item = grid.items[i][j]
                if item is not None:
                    x1, y1, x2, y2 = size
                    rect = QtCore.QRectF(x1, y1, x2-x1, y2-y1)
                    item.setGeometry(rect)

    def setGeometry(self, val):
        rows, cols = self.shape()
        if cols == 0:
            return

        rect = val

        # Get sizes of items in main grid
        size_grid = self._get_size_grid(min=False)

        # Get axis grid column widths and min heights
        min_ax_grid_sizes = self._axis_grid_sizes()
        top_ax_size, bottom_ax_size = min_ax_grid_sizes
        if top_ax_size is not None:
            col_widths, height, n = top_ax_size
            top_row_heights = [((height-2) - n*2)/n]*n
            top_rect = QtCore.QRectF(val.left(), val.top(), rect.width(),  height - 2)
            rect.adjust(0, +height, 0, 0)

        if bottom_ax_size:
            col_widths, height, n = bottom_ax_size
            rect.adjust(0, 0, 0, -height)
            botm_row_heights = [(height)/n]*n
            bottom_rect = QtCore.QRectF(0, rect.bottom()-4, rect.width(), height)

        # Compute the column width boundaries
        n = self.shape()[1]
        constraints = []
        for col in range(n):
            lower, upper = None, None

            # Get items in column
            items = self[:,col]
            for ax_grid_key in self.axis_grids:
                ax_grid = self.axis_grids[ax_grid_key]

                # If along plot or labels col, use axis grids in calculations
                if ax_grid.count() > 0 and col < 2:
                    items += ax_grid[:,col]

            # Get the minimum size and max sizes for the entire column                
            min_sizes = [e.sizeHint(QtCore.Qt.MinimumSize) for e in items if e is not None]
            max_sizes = [e.sizeHint(QtCore.Qt.MaximumSize) for e in items if e is not None]
            
            if len(min_sizes) == 0:
                min_size, max_size = 0, 0
                constraints.append((min_size, max_size))
                continue
            else:
                min_size = max([s.width() for s in min_sizes])
                max_size = max([s.width() for s in max_sizes])

            # Adjust min and max sizes by column bounds if valid
            if col in self.column_bounds:
                lower, upper = self.column_bounds[col]
                if min_size > lower:
                    lower = min_size
                if min_size > upper:
                    upper = min_size
                
                if max_size < lower:
                    lower = max_size
                    upper = max_size
                
                min_size, max_size  = lower, upper
            
            # Keep track of constraints for current column
            constraints.append((min_size, max_size))

        # Compute the spacing alloted for each column
        spacing = 2
        col_spaces = self._readjust_space(rect.width(), n, constraints, spacing)
        
        # Compute the row space allotted for each row
        spacing = 2
        rows, cols = self.shape()
        plot_row_heights, plot_offsets = self._align_colorbars()
        n = len(plot_row_heights)
        stretch = self.get_height_factors()
        if stretch is not None and len(stretch) < n:
            stretch.extend([1]*(n-len(stretch)))
        row_heights = self._adjust_row_heights(rect.height(), n, plot_row_heights, 
            spacing=spacing, stretch=stretch)

        # Compute the geometry rect for each item using row spacing and column spacing
        rects = self._get_geometries(rect, self, col_spaces, row_heights, 2, 2)
        bottom_left = list(rects[-1][0])

        # Adjust rects so they do not go beyond bounds of viewbox in each row
        for i in range(len(rects)):
            row = self.items[i]
            row_rects = rects[i]
            top_ofst, botm_ofst = plot_offsets[i]
            j = 0
            for item, rect in zip(row, row_rects):
                if item is not None and not isinstance(item, MagPyPlotItem):
                    left, top, width, height = rect
                    top += top_ofst + 1
                    height -= (botm_ofst + 1)
                    rects[i][j] = [left, top, width, height]
                j += 1
        
        # Apply geometries and adjust color bar widths
        self._apply_geometries(rects, self)
        self._adjust_bar_widths(rects)

        # Set geometries for each axis grid available
        axis_spacing = 0
        if top_ax_size is not None:
            self._set_geometries(top_rect, self.axis_grids['top'], col_spaces[:2], top_row_heights, 2, axis_spacing)

        if bottom_ax_size is not None:
            self._set_geometries(bottom_rect, self.axis_grids['bottom'], col_spaces[:2], botm_row_heights, 2, axis_spacing)

        # Set geometry for time label if one exists
        if self.time_label:
            x1, y1, x2, y2 = bottom_left
            y1 = y2 - self.time_label.minimumSize().height()
            rect = QtCore.QRectF(x1, y1, x2-x1, y2-y1)
            self.time_label.setGeometry(rect)

    def _adjust_bar_widths(self, rects):
        # Check if any color bars exist
        col = 2
        if self.shape()[1] <= col:
            return

        # Get all color bars in column
        items = self[:,col]
        items = [e for e in items if e is not None]

        if len(items) > 1:
            # Set the widths of each axis item in grad legends to
            # be the same width
            rects = [rect for elem, rect in zip(self[:,col], rects[:,col]) if elem is not None]
            axes = [item.getAxis() for item in items]
            min_width = max([ax.minimumWidth() for ax in axes])
            for ax in axes:
                ax.setWidth(min_width)

    def get_plots(self):
        ''' Returns a list of MagPyPlotItems in grid '''
        plots = []
        for row in self.items:
            for e in row:
                mp_plot = isinstance(e, MagPyPlotItem)
                if mp_plot and (not isinstance(e, HiddenPlotAxis)):
                    plots.append(e)
        return plots
    
    def get_hidden_axes(self):
        ''' Returns a list of axes in Axis Grids'''
        axes = []
        for key in self.axis_grids:
            grid = self.axis_grids[key]
            if grid.shape()[1] > 0:
                axes += self.axis_grids [key][:,1]
        axes = [ax for ax in axes if ax is not None]
        return axes
    
    def get_grad_legends(self):
        ''' Returns a list of gradient legends in grid'''
        grads = []
        for row in self.items:
            for e in row:
                if isinstance(e, GradLegend):
                    grads.append(e)
        return grads

class PlotGridObject(pg.GraphicsWidget):
    sigXRangeChanged = QtCore.pyqtSignal(tuple)
    sigPlotColorsChanged = QtCore.pyqtSignal(tuple)
    def __init__(self, window=None):
        pg.GraphicsWidget.__init__(self)
        self.grid = PlotGrid()
        self.window = window
        self.links = None
        self.scale_to_range = True

        # Custom settings
        self.ax_styles = {'tickFont':None}
        self.bottom_ax_spacing = None
        self.tracking_enabled = True

        # Custom actions and helper windows
        self.plotApprAction = QtWidgets.QAction(self)
        self.plotApprAction.triggered.connect(self.openPlotAppr)
        self.plotApprAction.setText('Change Plot Appearance...')

        self.tickLabels = None
        self.plotAppr = None

        self.setLayout(self.grid)

    def set_links(self, links):
        self.links = links
        self.update_y_ranges()
    
    def get_links(self):
        return self.links if self.links is not None else []
    
    def set_window(self, window):
        self.window = window

    def _get_x_range(self):
        plots = self.get_plots()
        if len(plots) == 0:
            return (0, 0)
        ref_plot = plots[0]
        return ref_plot.getViewBox().viewRange()[0]

    def list_axis_grids(self):
        ''' Returns a dictionary of additional axes labels in each
            Axis Grid (top, botttom)
        '''
        label_dict = {}
        for key in self.grid.axis_grids:
            ax_grid = self.grid.axis_grids[key]
            labels = ax_grid.get_labels()
            label_dict[key] = labels
        return label_dict

    def update_y_ranges(self):
        plots = self.get_plots()
        ranges = []

        # Get data bounds for each plot in current (or given) x range
        if self.scale_to_range:
            xrange = self._get_x_range()
        else:
            xrange = None

        for plot in plots:
            bounds = plot.dataBounds(ax=1, orthoRange=xrange)
            ranges.append(bounds)

        # Set ranges for linked plots
        seen = set()
        if self.links is not None:
            for plot_set in self.links:
                # Get bounds for plot data items
                set_bounds = [ranges[i] for i in plot_set]
                set_bounds = [b for b in set_bounds if (b is not None and b[1] is not None)]

                if len(set_bounds) == 0:
                    continue

                # Determine scale and padding
                max_range = max([upper-lower for lower, upper in set_bounds])
                half_range = (max_range / 2) * 1.05

                # Spectrograms should have zero padding
                if True in [plots[index].isSpecialPlot() for index in plot_set]:
                    half_range = 0.0

                # Set bounds for each plot in link group
                for index in plot_set:
                    bound = ranges[index]
                    if bound is None or bound[1] is None:
                        continue
                    mid = (bound[-1]+bound[0])/2
                    plots[index].setYRange(mid-half_range, mid+half_range, 0.0)
                    seen.add(index)
        
        n = len(plots)
        for plot, bounds, index in zip(plots, ranges, range(n)):
            # Skip linked plots or those with no data
            if index in seen:
                continue

            if bounds is None or bounds[0] is None or bounds[1] is None:
                continue

            # Use regular range as data bounds
            diff = bounds[1] - bounds[0]
            half = diff * 0.01 
            if plot.isSpecialPlot():
                half = 0.0
            plot.setYRange(bounds[0] - half, bounds[1] + half, 0.0)

    def set_autoscale(self, val=True):
        self.scale_to_range = val
        self.update_y_ranges()

    def get_traces(self):
        ''' Returns trace info for each plot '''
        traces = []
        plots = self.grid.get_plots()
        index = 0
        for plot in plots:
            lines = plot.getLineInfo()
            traces.append((index, lines))
            index += 1
        return traces
    
    def update_label_color(self, plot, name, color):
        # Find index and corresponding label
        plot_col = self.grid[:,1]
        row = plot_col.index(plot)
        label = self.grid[row][0]
        if label is not None:
            labels = label.get_labels()
            # Find index for trace in label item
            if name in labels:
                index = labels.index(name)
                colors = label.get_colors()[:]
                colors[index] = color
                label.set_colors(colors)

    def get_plots(self):
        return self.grid.get_plots()

    def get_layout(self):
        return self.grid

    def set_tick_text_size(self, size):
        # Get new font
        font = QtGui.QFont()
        font.setPointSize(size)

        # Apply new styles
        self.set_plot_styles(tickFont=font)
    
    def set_plot_styles(self, **kwargs):
        self.ax_styles.update(kwargs)
        self._update_plot_styles()

    def _update_plot_styles(self):
        # Get all items to change
        plots = self.grid.get_plots()
        grads = self.grid.get_grad_legends()
        hidden = self.grid.get_hidden_axes()

        # Set style for all current plots
        all_plots = plots + hidden
        for plot in all_plots:
            for key in ['bottom', 'left', 'top', 'right']:
                ax = plot.getAxis(key)
                ax.setStyle(**self.ax_styles)
        
        # Set spacing
        self.set_tick_spacing(self.bottom_ax_spacing)

        # Enable tracking
        self.set_tracking(self.tracking_enabled)

    def set_tick_spacing(self, spacing):
        plots = self.get_plots()
        if len(plots) == 0:
            return
        
        for plot in plots:
            ax = plot.getAxis('bottom')
            ax = plot.getAxis('top')
            if spacing is not None:
                ax.setCstmTickSpacing(spacing)
            else:
                ax.resetTickSpacing()

    def _update_state(self):
        self._update_plot_styles()

    def getContextMenus(self, *args, **kwargs):
        menus = [self.plotApprAction]
        return menus
    
    def openPlotAppr(self):
        from .plotAppearance import PlotAppearance
        self.closePlotAppr()
        self.plotAppr = PlotAppearance(self)
        self.plotAppr.colorsChanged.connect(self.sigPlotColorsChanged)
        self.destroyed.connect(self.plotAppr.close)
        self.plotAppr.show()
    
    def closePlotAppr(self):
        if self.plotAppr:
            self.plotAppr.close()
            self.plotAppr = None

    def enable_tracking(self, val=True):
        self.tracking_enabled = val
        self.set_tracking(self.tracking_enabled)

    def set_tracking(self, val):
        plots = self.get_plots()
        for plot in plots:
            ax = plot.getAxis('bottom')
            if isinstance(ax, DateAxis):
                tick_to_date = lambda t : ff_time.tick_to_ts(t, plot.epoch)[:-3]
                format_funcs = {'x':tick_to_date}
            else:
                format_funcs = {}
            plot.enableTracking(val, format_funcs)

    def close(self, *args, **kwargs):
        self.closePlotAppr()
        super().close(*args, **kwargs)
