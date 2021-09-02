import pyqtgraph as pg
from PyQt5 import QtWidgets

class GraphicsView(pg.GraphicsView):
    def __init__(self, *args, **kwargs):
        self.shortcut_dict = {}
        super().__init__(*args, **kwargs)

    def add_shortcut(self, key, func):
        ''' Adds a shortcut to the GraphicsView that when triggered
            calls func()
        '''
        # Check if shortcut in dict
        item = self.shortcut_dict.get(key)

        # Create a new shortcut and action linked to func if
        # the shortcut hasn't been created yet
        if item is None:
            item = QtWidgets.QShortcut(key, self)
            self.shortcut_dict[key] = item
        # Otherwise, unlink the action's last signal connections
        # and connect it to func
        else:
            item.activated.disconnect()
        item.activated.connect(func)