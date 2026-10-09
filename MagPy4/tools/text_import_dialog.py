from PyQt5 import QtWidgets
from datetime import datetime, timedelta
import os

from ..data_import.text_import import preview_text_file, FORMATS
from ..plotbase.selectbase import TimeFormatWidget
from fflib import isoparser

class TextImportDialog(QtWidgets.QDialog):
    ''' Dialog that previews a text/CSV file's detected columns and lets the
        user confirm or correct how the timestamp is derived before the file
        is parsed in full. Call exec_() and, if it returns Accepted, read the
        chosen config back via get_time_config().
    '''
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.time_config = None

        self.setWindowTitle(f'Import Text File - {os.path.basename(filename)}')
        self.resize(700, 500)

        self.preview = None
        self.preview_error = None
        try:
            self.preview = preview_text_file(filename)
        except Exception as e:
            self.preview_error = str(e)

        self._setupUI()
        if self.preview is not None:
            self._populateFromPreview()
            self._updateModeVisibility()
            self._updateValidation()

    def _setupUI(self):
        layout = QtWidgets.QVBoxLayout(self)

        if self.preview is None:
            errLbl = QtWidgets.QLabel(self.preview_error or 'Could not read file.')
            errLbl.setWordWrap(True)
            errLbl.setStyleSheet('color: #b00020;')
            layout.addWidget(errLbl)

            btnBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
            btnBox.rejected.connect(self.reject)
            layout.addWidget(btnBox)
            return

        cols = self.preview['columns']
        rows = self.preview['rows']
        self.col_items = [c if c else f'(unnamed {i})' for i, c in enumerate(cols)]

        # Info line
        info = self.preview['file_info']
        fmt_name = 'CSV' if info['type'] == FORMATS.CSV else 'Fixed-width'
        infoLabel = QtWidgets.QLabel(
            f"Format: {fmt_name}    Comment lines skipped: {self.preview['skip_header']}    "
            f"Columns detected: {len(cols)}"
        )
        layout.addWidget(infoLabel)

        # Preview table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(self.col_items)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                if c < len(cols):
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(val))
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # Timestamp configuration
        timeBox = QtWidgets.QGroupBox('Timestamp')
        timeLt = QtWidgets.QVBoxLayout(timeBox)

        modeLt = QtWidgets.QHBoxLayout()
        modeLt.addWidget(QtWidgets.QLabel('Mode:'))
        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.addItems(['ISO 8601 column', 'Year + day-of-year', 'Custom format'])
        modeLt.addWidget(self.modeCombo)
        modeLt.addStretch()
        timeLt.addLayout(modeLt)

        # ISO 8601 mode
        self.isoFrame = QtWidgets.QWidget()
        isoLt = QtWidgets.QFormLayout(self.isoFrame)
        self.isoColCombo = QtWidgets.QComboBox()
        self.isoColCombo.addItems(self.col_items)
        isoLt.addRow('Timestamp column:', self.isoColCombo)
        timeLt.addWidget(self.isoFrame)

        # Year + day-of-year mode
        self.ydFrame = QtWidgets.QWidget()
        ydLt = QtWidgets.QFormLayout(self.ydFrame)
        self.yearCombo = QtWidgets.QComboBox()
        self.yearCombo.addItems(self.col_items)
        self.doyCombo = QtWidgets.QComboBox()
        self.doyCombo.addItems(self.col_items)
        self.fdayCombo = QtWidgets.QComboBox()
        self.fdayCombo.addItems(['(none)'] + self.col_items)
        ydLt.addRow('Year column:', self.yearCombo)
        ydLt.addRow('Day-of-year column:', self.doyCombo)
        ydLt.addRow('Fractional day column:', self.fdayCombo)
        timeLt.addWidget(self.ydFrame)

        # Custom format mode
        self.customFrame = QtWidgets.QWidget()
        custLt = QtWidgets.QFormLayout(self.customFrame)
        self.custCol1Combo = QtWidgets.QComboBox()
        self.custCol1Combo.addItems(self.col_items)
        self.custCol2Combo = QtWidgets.QComboBox()
        self.custCol2Combo.addItems(['(none)'] + self.col_items)
        self.custFmtWidget = TimeFormatWidget()
        custLt.addRow('Date/time column:', self.custCol1Combo)
        custLt.addRow('Second column (optional):', self.custCol2Combo)
        custLt.addRow(self.custFmtWidget)
        timeLt.addWidget(self.customFrame)

        layout.addWidget(timeBox)

        # Live validation feedback
        self.validationLabel = QtWidgets.QLabel()
        self.validationLabel.setWordWrap(True)
        layout.addWidget(self.validationLabel)

        # Buttons
        self.btnBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btnBox.accepted.connect(self._onAccept)
        self.btnBox.rejected.connect(self.reject)
        layout.addWidget(self.btnBox)

        # Signal wiring
        self.modeCombo.currentIndexChanged.connect(self._updateModeVisibility)
        self.modeCombo.currentIndexChanged.connect(self._updateValidation)
        for w in [self.isoColCombo, self.yearCombo, self.doyCombo, self.fdayCombo,
                  self.custCol1Combo, self.custCol2Combo]:
            w.currentIndexChanged.connect(self._updateValidation)
        self.custFmtWidget.getLineEdit().textChanged.connect(self._updateValidation)

    def _populateFromPreview(self):
        ''' Pre-selects widgets to match the auto-detected time_spec so the
            user only has to intervene when the guess is wrong
        '''
        spec = self.preview['time_spec']
        mode = spec.get('mode')
        if mode == 'year_doy':
            self.modeCombo.setCurrentIndex(1)
            self.yearCombo.setCurrentIndex(spec['year_col'])
            self.doyCombo.setCurrentIndex(spec['doy_col'])
            fday_col = spec.get('fday_col')
            self.fdayCombo.setCurrentIndex(0 if fday_col is None else fday_col + 1)
        elif mode == 'iso':
            self.modeCombo.setCurrentIndex(0)
            self.isoColCombo.setCurrentIndex(spec['col'])
        else:
            # Nothing auto-detected - default to custom format so the user
            # has to actively choose a column and format
            self.modeCombo.setCurrentIndex(2)

    def _updateModeVisibility(self):
        idx = self.modeCombo.currentIndex()
        self.isoFrame.setVisible(idx == 0)
        self.ydFrame.setVisible(idx == 1)
        self.customFrame.setVisible(idx == 2)

    def _currentTimeConfig(self):
        idx = self.modeCombo.currentIndex()
        if idx == 0:
            return {'mode': 'iso', 'col': self.isoColCombo.currentIndex()}
        elif idx == 1:
            fday_idx = self.fdayCombo.currentIndex()
            fday_col = None if fday_idx == 0 else fday_idx - 1
            return {
                'mode': 'year_doy',
                'year_col': self.yearCombo.currentIndex(),
                'doy_col': self.doyCombo.currentIndex(),
                'fday_col': fday_col,
            }
        else:
            cols = [self.custCol1Combo.currentIndex()]
            col2_idx = self.custCol2Combo.currentIndex()
            if col2_idx > 0:
                cols.append(col2_idx - 1)
            return {
                'mode': 'custom',
                'cols': cols,
                'format': self.custFmtWidget.text(),
            }

    def _parse_sample(self, config, rows):
        ''' Parses the timestamp out of each preview row using the same
            rules the real reader will use, for live validation feedback
        '''
        mode = config['mode']
        if mode == 'iso':
            col = config['col']
            parser = isoparser.ISOParser('T')
            return [parser.isoparse(row[col].strip()) for row in rows]
        elif mode == 'year_doy':
            year_col, doy_col = config['year_col'], config['doy_col']
            fday_col = config.get('fday_col')
            dates = []
            for row in rows:
                year = int(row[year_col])
                doy = int(row[doy_col])
                fday = float(row[fday_col]) if fday_col is not None else 0.0
                dates.append(datetime(year, 1, 1) + timedelta(days=doy - 1 + fday))
            return dates
        else:
            cols = config['cols']
            fmt = config['format']
            dates = []
            for row in rows:
                if len(cols) == 2:
                    raw = f'{row[cols[0]]} {row[cols[1]]}'
                else:
                    raw = row[cols[0]]
                dates.append(datetime.strptime(raw.strip(), fmt))
            return dates

    def _updateValidation(self):
        config = self._currentTimeConfig()
        rows = self.preview['rows']
        okButton = self.btnBox.button(QtWidgets.QDialogButtonBox.Ok)

        if not rows:
            self.validationLabel.setText('No sample rows available to validate against.')
            okButton.setEnabled(True)
            return

        try:
            sample = self._parse_sample(config, rows)
            text = ', '.join(d.isoformat() for d in sample[:3])
            self.validationLabel.setStyleSheet('color: #1a7f37;')
            self.validationLabel.setText(f'Parsed sample timestamps: {text}')
            okButton.setEnabled(True)
        except Exception as e:
            self.validationLabel.setStyleSheet('color: #b00020;')
            self.validationLabel.setText(f'Could not parse timestamps: {e}')
            okButton.setEnabled(False)

    def _onAccept(self):
        self.time_config = self._currentTimeConfig()
        self.accept()

    def get_time_config(self):
        ''' Returns the user-confirmed time_config, or None if cancelled '''
        return self.time_config
