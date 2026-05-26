from PyQt6.QtWidgets import (
    QApplication, QLabel, QHBoxLayout, QVBoxLayout,
    QMainWindow, QWidget, QTabWidget,
)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt
import datetime
import logging
import os
import sys

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class KnwoledgeTracer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.style = "background-color: #d3d3d3;"
        self.apply_config()
        self.CreateGui()
        self.CreateMenubar()

    def apply_config(self):
        print("config done")

    def CreateGui(self):
        logging.info(self.__class__.__name__ + ':Create Gui')
        self.setWindowTitle("KnowledgeTracer")
        self.resize(2560, 1440)
        self.CreateVlayout()

    def CreateMenubar(self):
        self.menubar = self.menuBar()
        file_menu = self.menubar.addMenu("File")
        process_menu = self.menubar.addMenu("Process")
        analyse_menu = self.menubar.addMenu("Analyse")
        self.menubar.setStyleSheet(self.style)

    def CreateVlayout(self):
        main_layout = QHBoxLayout()
        central = QWidget()
        self.left_panel = QWidget()
        self.left_panel.setStyleSheet(self.style)
        left_layout = QVBoxLayout()
        
        self.info_label = QLabel("No dataset loaded")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

        left_layout.addStretch()
        self.left_panel.setLayout(left_layout)
        self.tabs = QTabWidget()
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.tabs.addTab(self.canvas, "Graph")


        self.data_tab = QLabel("Display data : head, ....")
        self.tabs.addTab(self.data_tab, "Data")
       
        self.model_tab = QLabel("Results: DAS3H / IRT / ...")
        self.tabs.addTab(self.model_tab, "Models")


        self.Heuristic_tab = QLabel("Heuristics / Simulation")
        self.tabs.addTab(self.Heuristic_tab, "Heuristics")
        main_layout.addWidget(self.left_panel, stretch=1)
        main_layout.addWidget(self.tabs, stretch=4)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def CreateToolBarDataSet(self):
        pass

    def CreateGraph(self):
        pass

    def Display_ManageData(self):
        pass

    def DisplayresultsModel(self):
        pass

    def ManageHeuristics(self):
        pass
    def SimulationHeuristics(self):
        pass



if __name__ == '__main__':
    now = datetime.datetime.now()
    if os.path.exists("/home/loubna/Temp"):
        logging.basicConfig(
            filename='/home/loubna/KnowledgeTracer_'
            + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '.log',
            level=logging.INFO,
        )
        logger.info('KnowledgeTracer Started at ' + str(now.hour) + 'h' + str(now.minute))

    app = QApplication(sys.argv)
    knwoledgetracer = KnwoledgeTracer()
    knwoledgetracer.show()
    sys.exit(app.exec())