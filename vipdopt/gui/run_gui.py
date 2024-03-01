"""Entrypoint running the GUI. Mainly for testing right now."""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMainWindow,
    QTreeWidgetItem,
    QWidget,
    QComboBox,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QLayout
)

import pickle
from typing import Callable
import subprocess

import vipdopt
from vipdopt.gui.config_editor import ConfigModel
from vipdopt.gui.ui_fom_dialog import Ui_Dialog as Ui_FomDialog
from vipdopt.gui.ui_settings import Ui_MainWindow as Ui_SettingsWindow
from vipdopt.gui.ui_dashboard import Ui_MainWindow as Ui_DashboardWindow
from vipdopt.monitor import Monitor
from vipdopt.optimization import FoM, BayerFilterFoM, GradientOptimizer
from vipdopt.project import Project
from vipdopt.simulation import LumericalSimulation, ISimulation
from vipdopt.utils import PathLike, read_config_file, subclasses, StoppableThread
from vipdopt.submit_job import generate_script

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

font = {
    'size': 12,
    'family': 'serif',
    'weight': 'bold',
    'serif': ['cmr10'],
}
mpl.rc('font', **font)

FOM_TYPES = subclasses(FoM)
SIM_TYPES = subclasses(ISimulation)
OPTIMIZER_TYPES = subclasses(GradientOptimizer)

PLOT_NAMES = [
    ['fom.pkl', 'overall_trans.pkl'],
    ['enorm.pkl', 'final_device_layer.pkl'],
]
PLOT_DIMS = (len(PLOT_NAMES), len(PLOT_NAMES[0]))


class FomDialog(QDialog, Ui_FomDialog):
    """Dialog window for configuring FoMs."""
    def __init__(
            self,
            parent: QWidget | None,
            f: Qt.WindowType = Qt.WindowType.Dialog,
    )-> None:
        """Initialize a FomDialog."""
        super().__init__(parent, f)

        self.setupUi(self)

    def populate_tree(self, sims: list[LumericalSimulation]):
        """Populate the tree widget with the FoMs."""
        self.treeWidget.clear()
        sim: LumericalSimulation
        for i, sim in enumerate(sims):
            sim_node = QTreeWidgetItem((f'Simulation {i}', ))
            source_nodes = [QTreeWidgetItem((src,)) for src in sim.monitor_names()]
            self.uncheck_items(*source_nodes)
            sim_node.addChildren(source_nodes)
            self.treeWidget.addTopLevelItem(sim_node)

    def check_items(self, *items: QTreeWidgetItem):
        """Check all items in the provided list."""
        for item in items:
            item.setCheckState(1, Qt.CheckState.Checked)

    def uncheck_items(self, *items: QTreeWidgetItem):
        """Uncheck all items in the provided list."""
        for item in items:
            item.setCheckState(1, Qt.CheckState.Unchecked)
            item.setExpanded(False)

    def check_monitors(
            self,
            src_to_sim_map: dict[str, LumericalSimulation],
            *monitors: Monitor,
    ):
        """Check all of the monitors."""
        all_children: list[QTreeWidgetItem] = self.treeWidget.findChildren(
            QTreeWidgetItem
        )
        self.uncheck_items(*all_children)
        mon_items = []
        for monitor in monitors:
            mon_nodes = self.treeWidget.findItems(
                monitor.monitor_name,
                Qt.MatchFlag.MatchRecursive | Qt.MatchFlag.MatchExactly,
                0,
            )
            mon_node = mon_nodes[
                list(src_to_sim_map.keys()).index(monitor.source_name)
            ]
            mon_node.parent().setExpanded(True)

            mon_items.append(mon_node)
        self.check_items(*mon_items)


class SettingsWindow(QMainWindow, Ui_SettingsWindow):
    """Wrapper class for the settings window."""
    def __init__(self):
        """Initialize a SettingsWindow."""
        super().__init__()
        self.setupUi(self)

        self.config_model = ConfigModel()
        self.config_treeView.setModel(self.config_model)
        self.sim_model = ConfigModel()
        self.sim_treeView.setModel(self.sim_model)
        self.device_model = ConfigModel()
        self.device_treeView.setModel(self.device_model)

        self.actionOpen.triggered.connect(self.open_project)

        self.config_pushButton.clicked.connect(self.load_yaml)
        self.sim_pushButton.clicked.connect(self.load_json)

        # Initialize Backend
        self.project = Project()

        self.fom_addrow_toolButton = QToolButton()
        self.fom_addrow_toolButton.setObjectName(u"fom_addrow_toolButton")
        self.fom_gridLayout.addWidget(self.fom_addrow_toolButton, 1, 0, 1, 1)
        self.fom_addrow_toolButton.setText(QCoreApplication.translate("MainWindow", u"Add Row", None))
        self.fom_addrow_toolButton.clicked.connect(lambda: self.new_fom())

        self.fom_widget_rows: list[tuple[QLineEdit, QComboBox, QPushButton, QPushButton, QLineEdit, QComboBox, QComboBox, QToolButton]]= []
        self.fom_locs: dict[str, tuple[int, int]] = {}  # index, row of each FoM
        self.new_fom()

    # General Methods
    def new_project(self):
        """Create a new project."""
        self.project = Project()
        self._update_values()

    def open_project(self):
        """Load optimization project into the GUI."""
        proj_dir = QFileDialog.getExistingDirectory(
            self,
            'Select Project Directory',
            './',
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if proj_dir:
            self.project.load_project(proj_dir)
            vipdopt.logger.info(f'Loaded project from {proj_dir}')
            self._update_values()
            vipdopt.logger.info(f'Updated GUI with values from {proj_dir}')

    def _update_values(self):
        """Update GUI to match values in Project."""
        # Config Tab
        self.config_model.load(self.project.config)

        # Simulation Tab
        self.sim_model.load(self.project.base_sim.as_dict())

        self.sim_config_treeWidget.clear()
        sim: LumericalSimulation
        for i, sim in enumerate(self.project.optimization.sims):
            sim_node = QTreeWidgetItem((f'Simulation {i}', ))
            source_nodes = [QTreeWidgetItem((src,)) for src in sim.source_names()]
            for src in source_nodes:
                src.setCheckState(1, Qt.CheckState.Unchecked)
            source_nodes[i].setCheckState(1, Qt.CheckState.Checked)
            sim_node.addChildren(source_nodes)
            self.sim_config_treeWidget.addTopLevelItem(sim_node)

        # FoM Tab
        self.clear_fom_rows()
        for i, fom in enumerate(self.project.foms):
            self.add_fom(i, fom)
            self.fom_widget_rows[i][4].setText(str(self.project.weights[i]))
            self.fom_widget_rows[i][5].addItems([f'Simulation {i}' for i in range(len(self.project.optimization.sims))])
            self.fom_widget_rows[i][6].addItems([f'Simulation {i}' for i in range(len(self.project.optimization.sims))])
        # Device Tab
        self.device_model.load(self.project.device.as_dict())

        # Optimization Tab
        self.opt_iter_lineEdit.setText(str(self.project.optimization.iteration))
        self.opt_iter_per_epoch_lineEdit.setText(str(self.project.optimization.iter_per_epoch))
        self.opt_max_epoch_lineEdit.setText(str(self.project.optimization.max_epochs))

    # Configuration Tab
    def load_yaml(self):
        """Load a yaml config file into the configuration tab."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            'Select Configuration File',
            './',
            'YAML (*.yaml *.yml);;All Files(*.*)',
            'YAML (*.yaml *.yml)',
        )
        if fname:
            cfg = read_config_file(fname)
            self.config_lineEdit.setText(fname)
            self.config_model.load(cfg)

    # Simulation Tab
    def load_json(self):
        """Load a yaml config file into the configuration tab."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            'Select Configuration File',
            './',
            'JSON (*.json);;All Files(*.*)',
            'JSON (*.json)',
        )
        if fname:
            cfg = read_config_file(fname)
            self.sim_lineEdit.setText(fname)
            self.sim_model.load(cfg)

    # FoM Tab
    def fom_dialog(self, name: str, mode: str='fom'):
        """Create a dialog for selecting monitors for FoMs."""
        dialog = FomDialog(self)
        dialog.populate_tree(self.project.optimization.sims)
        idx, _ = self.fom_locs[name]
        fom = self.project.foms[idx]
        match mode.lower():
            case 'fom':
                monitors = fom.fom_monitors
            case 'grad':
                monitors = fom.grad_monitors
            case _:
                raise ValueError(
                    'fom_dialog can only be called with mode "fom" or "grad"'
                )

        dialog.check_monitors(self.project.src_to_sim_map, *monitors)
        dialog.exec()

    def new_fom_row(self, name: str):
        """Create a new FoM row."""
        rows = self.fom_gridLayout.rowCount()

        # Name Line Edit
        name_lineEdit = QLineEdit()
        name_lineEdit.setObjectName(f'fom_name_lineEdit_{name}')
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(name_lineEdit.sizePolicy().hasHeightForWidth())
        name_lineEdit.setSizePolicy(sizePolicy)
        name_lineEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", name, None))
        self.fom_gridLayout.addWidget(name_lineEdit, rows, 0, 1, 1)

        # Type Combo Box
        type_comboBox = QComboBox()
        type_comboBox.setObjectName(f'fom_type_comboBox_{name}')
        type_comboBox.clear()
        type_comboBox.addItems(FOM_TYPES)
        self.fom_gridLayout.addWidget(type_comboBox, rows, 1, 1, 1)
        
        # Monitor Buttons
        fom_pushButton = QPushButton()
        fom_pushButton.setObjectName(f'fom_fom_pushButton_{name}')
        fom_pushButton.setText(QCoreApplication.translate("MainWindow", u"Choose Monitors...", None))
        self.fom_gridLayout.addWidget(fom_pushButton, rows, 2, 1, 1)
        fom_pushButton.clicked.connect(lambda: self.fom_dialog(name, 'fom'))
        
        grad_pushButton = QPushButton()
        grad_pushButton.setObjectName(f'fom_grad_pushButton_{name}')
        grad_pushButton.setText(QCoreApplication.translate("MainWindow", u"Choose Monitors...", None))
        self.fom_gridLayout.addWidget(grad_pushButton, rows, 3, 1, 1)
        grad_pushButton.clicked.connect(lambda: self.fom_dialog(name, 'grad'))

        # Weight Line Edit
        weight_lineEdit = QLineEdit()
        weight_lineEdit.setObjectName(f'fom_weight_lineEdit_{name}')
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(weight_lineEdit.sizePolicy().hasHeightForWidth())
        weight_lineEdit.setSizePolicy(sizePolicy1)
        name_lineEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"1.0", None))
        self.fom_gridLayout.addWidget(weight_lineEdit, rows, 4, 1, 1)

        # Simulation Combo Boxes
        fwd_sim_comboBox = QComboBox()
        fwd_sim_comboBox.setObjectName(f'fom_fwd_sim_comboBox_{name}')
        fwd_sim_comboBox.clear()
        self.fom_gridLayout.addWidget(fwd_sim_comboBox, rows, 5, 1, 1)

        adj_sim_comboBox = QComboBox()
        adj_sim_comboBox.setObjectName(f'fom_adj_sim_comboBox_{name}')
        adj_sim_comboBox.clear()
        self.fom_gridLayout.addWidget(adj_sim_comboBox, rows, 6, 1, 1)

        # Delete Row button
        delrow_toolButton = QToolButton()
        delrow_toolButton.setObjectName(f'fom_delrow_toolButton_{name}')
        self.fom_gridLayout.addWidget(delrow_toolButton, rows, 7, 1, 1)
        delrow_toolButton.setText(QCoreApplication.translate("MainWindow", u"Remove Row", None))
        delrow_toolButton.clicked.connect(lambda: self.remove_fom(name))

        self.fom_widget_rows.append(
            (
                name_lineEdit,
                type_comboBox,
                fom_pushButton,
                grad_pushButton,
                weight_lineEdit,
                fwd_sim_comboBox,
                adj_sim_comboBox,
                delrow_toolButton
            )
        )
        self.add_fom_add_button()
        return rows
    
    def new_fom(self):
        """Create a new FoM."""
        fom = BayerFilterFoM([], [], '', [])
        self.add_fom(len(self.fom_widget_rows), fom)
        self.project.foms.append(fom)
    
    def add_fom(self, idx: int, fom: FoM):
        """Create a new FoM row and populate with data from a FoM."""
        self.fom_gridLayout.removeWidget(self.fom_addrow_toolButton)
        row = self.new_fom_row(fom.name)
        self.fom_locs[fom.name] = (idx, row)
        name_lineEdit, type_comboBox, _, _, _, _, _, _ = \
              self.fom_widget_rows[idx]

        name_lineEdit.setText(fom.name)
        type_comboBox.setCurrentIndex(FOM_TYPES.index(type(fom).__name__))
    
    def add_fom_add_button(self):
        """Move the FoM add buttons to the correct location."""
        row = self.fom_gridLayout.rowCount()
        self.fom_gridLayout.removeWidget(self.fom_addrow_toolButton)
        self.fom_gridLayout.addWidget(self.fom_addrow_toolButton, row, 0)

    def clear_fom_rows(self):
        """Remove all FoMs."""
        for _, (_, r) in self.fom_locs.items():
            for c in range(self.fom_gridLayout.columnCount()):
                layout = self.fom_gridLayout.itemAtPosition(r, c)
                if layout is not None:
                    layout.widget().deleteLater()
                    self.fom_gridLayout.removeItem(layout)
        self.fom_widget_rows = []
        self.fom_locs = {}
        self.add_fom_add_button()

    def remove_fom(self, name: str):
        if len(self.fom_locs) == 1:
            raise ValueError(f'There must be at least one FoM.')

        idx, row = self.fom_locs.pop(name)
        for c in range(self.fom_gridLayout.columnCount()):
            layout = self.fom_gridLayout.itemAtPosition(row, c)
            if layout is not None:
                layout.widget().deleteLater()
                self.fom_gridLayout.removeItem(layout)

        # Shift remaining rows up
        for name, (i, r) in self.fom_locs.items():
            if i > idx:
                self.fom_locs[name] = (i - 1, r)

        self.add_fom_add_button()

    # Optimization Tab
    def load_optimization_settings(self, fname: PathLike):
        """Load settings for the optimization tab."""


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, figure: Figure= None, width=3, height=12/5, dpi=100, fontsize=12):
        # fig = Figure(figsize=(width, height), dpi=dpi)
        self.fontsize = fontsize
        if figure is not None:
            self.change_font(figure)
            self.fig = figure
        else:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.fig.set_layout_engine('constrained')
        super().__init__(self.fig)
        self.adjust_figure_size(width)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    def change_font(self, fig: Figure):
        for ax in fig.axes:
            ax.set_title(ax.get_title(), fontsize=self.fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=self.fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=self.fontsize)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=self.fontsize)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=self.fontsize)
    
    def adjust_figure_size(self, fig_width, fig_height=None):
        if fig_height is None:
            fig_height = fig_width*4/5
        l = self.fig.subplotpars.left
        r = self.fig.subplotpars.right
        t = self.fig.subplotpars.top
        b = self.fig.subplotpars.bottom
        figw = float(fig_width)/(r-l)
        figh = float(fig_height)/(t-b)
        self.fig.figure.set_size_inches(figw, figh, forward=True)

class StatusDashboard(QMainWindow, Ui_DashboardWindow):
    """Wrapper class for the status window."""
    def __init__(self):
        """Initialize a StatusWindow."""
        super().__init__()
        self.setupUi(self)
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)


        self.actionOpen.triggered.connect(self.open_project)

        self.plots = [
            [MplCanvas() for _ in range(PLOT_DIMS[1])] for _ in range(PLOT_DIMS[0])
        ]
        for i in range(PLOT_DIMS[0]):
            for j in range(PLOT_DIMS[1]):
                self.gridLayout.addWidget(self.plots[i][j], i, j)

        self.project = Project()
        self.running = False
        self.opt_thread = StoppableThread()

        self.start_stop_pushButton.setText('Start Optimization')

        self.start_stop_pushButton.clicked.connect(self.toggle_optimization)
        self.edit_pushButton.clicked.connect(self.settings_window)

    def open_project(self):
        """Load optimization project into the GUI."""
        proj_dir = QFileDialog.getExistingDirectory(
            self,
            'Select Project Directory',
            './',
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if proj_dir:
            self.project.load_project(proj_dir)
            vipdopt.logger.info(f'Loaded project from {proj_dir}')
            self._update_values()
            vipdopt.logger.info(f'Updated GUI with values from {proj_dir}')

    def settings_window(self):
        win = SettingsWindow()
        win.project = self.project
        try:
            win._update_values()
        except BaseException:
            pass
        win.setWindowModality(Qt.WindowModality.ApplicationModal)
        def wrap_func(f: Callable):
            def override_close(*args):
                f(*args)
                self._update_values()
            return override_close

        win.closeEvent = wrap_func(win.closeEvent)
        win.show()

    def create_submission_script(self):
        slurm_script = str(self.project.dir / 'slurm.sh'),
        generate_script(
            self.project.optimization.nsims,
            str(self.project.dir),
            'config.json',
        )
        subprocess.call(['sbatch', str(slurm_script)])



    def toggle_optimization(self):
        self.running = not self.running
        if self.running:
            self.create_submission_script()
            self.opt_thread = StoppableThread(target=self.project.start_optimization)
            self.opt_thread.run()
        else:
            self.opt_thread.stop()
            self.opt_thread = StoppableThread(target=self.project.stop_optimization)
            self.opt_thread.run()
        self._update_values()
    
    def _update_plots(self):
        plots_folder = self.project.subdirectories['opt_plots']

        for i in range(PLOT_DIMS[0]):
            for j, name in enumerate(PLOT_NAMES[i]):
                self.gridLayout.removeWidget(self.plots[i][j])
                fig_path = plots_folder / name
                if fig_path.exists():
                    self.plots[i][j].deleteLater()
                    new_fig = pickle.load(fig_path.open('rb'))
                    if isinstance(new_fig, tuple):
                        new_fig = new_fig[0]
                    self.plots[i][j] = MplCanvas(new_fig)
                self.gridLayout.addWidget(self.plots[i][j], i, j)

    def _update_values(self):

        self._update_plots()

        if self.running:
            self.start_stop_pushButton.setText('Stop Optimization')
        else:
            self.start_stop_pushButton.setText('Start Optimization')
        
        self.iter_label.setText(str(self.project.optimization.iteration))
        self.epoch_label.setText(str(self.project.optimization.epoch))
        self.avg_e_label.setText('unknown')
        self.avg_power_label.setText('unknown')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    # Create application windows
    status_window = StatusDashboard()
    status_window.show()
    # settings_window = SettingsWindow()
    # settings_window.show()
    app.exec()
