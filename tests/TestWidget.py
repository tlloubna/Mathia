from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QMenu, QToolBar, QFrame
from PyQt6.QtGui import QIcon, QFont, QPixmap, QEnterEvent, QAction
from PyQt6.QtCore import QTimer, QPoint
import sys
 
"""
QLabel :  Creer les  labels
QHBoxLayout : pour la mise en oage horizontale
"""
 
 
 
class Window1(QWidget):
    def __init__(self,widget):
        super().__init__()
        self.widget = widget
 
        Title = QLabel("KnowLedgeTracer")
        Title.setFont(QFont("Lexend", 32, QFont.Weight.Bold))
        Title.setStyleSheet("color: black; qproperty-alignment: 'AlignCenter';")
 
        logo_pixmap = QPixmap("/home/neuro/Documents/Loic/qt.png")
        logo = QLabel(self)
        logo.setPixmap(logo_pixmap)
        logo.setFixedSize(70, 70)
        logo.setScaledContents(True)
 
        layout_T = QHBoxLayout()
        layout_T.addStretch()
        layout_T.addSpacing(10)        
        layout_T.addWidget(Title)  
        layout_T.addWidget(logo)
        layout_T.addStretch()  
 
        btn = QPushButton("Enseignant\nProf en Poche", self)
        btn.setFont(QFont("Times New Roman", 12, QFont.Weight.Bold))
        btn.setFixedHeight(200)
        btn.setFixedWidth(130)
        btn.setStyleSheet('background-color:grey')
        btn.clicked.connect(lambda: self.widget.setCurrentIndex(1))
 
        btn0 = QPushButton("Chercheur\nETIS", self)
        btn0.setFont(QFont("Times New Roman", 12, QFont.Weight.Bold))
        btn0.setFixedHeight(200)
        btn0.setFixedWidth(130)
        btn0.setStyleSheet('background-color:grey')
        btn0.clicked.connect(lambda: self.widget.setCurrentIndex(2))
 
        layout_H = QHBoxLayout()
        layout_H.addStretch()
        layout_H.addWidget(btn)
        layout_H.addSpacing(170)
        layout_H.addWidget(btn0)
        layout_H.addStretch()
 
        layout_V = QVBoxLayout()
        layout_V.addStretch()    
        layout_V.addLayout(layout_H)
        layout_V.addLayout(layout_T)
        layout_V.addStretch()    
 
        self.setLayout(layout_V)
 
class PageChercheur(QWidget):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
 
        toolbar_frame = QFrame(self)
        toolbar_frame.setStyleSheet("background-color: #e0e0e0; border-bottom: 1px solid #b0b0b0;")
        toolbar_frame.setFixedHeight(40) # Hauteur fixe pour faire comme une vraie toolbar
        
        # Le layout interne de notre barre d'outils
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(10, 0, 10, 0) # Petites marges internes
 
        # On remplace la QAction par un vrai QPushButton textuel
        toolbar_button = QPushButton("Your bon", self)
        toolbar_button.setStyleSheet("background-color: transparent; border: none; font-weight: bold;")
        toolbar_button.clicked.connect(self.toolbar_button_clicked)
        
        # On ajoute le bouton à notre barre et on pousse vers la gauche
        toolbar_layout.addWidget(toolbar_button)
        toolbar_layout.addStretch()
 
        Title = QLabel("Chercheur(euse)", self)
        Title.setFont(QFont("Lexend", 32, QFont.Weight.Bold))
        Title.setStyleSheet("color: black;")
 
        btn = QPushButton("Accueil", self)
        btn.setFont(QFont("Times New Roman", 12, QFont.Weight.Bold))
        btn.setFixedHeight(50)
        btn.setFixedWidth(200)
        btn.setStyleSheet('background-color:grey')
        btn.clicked.connect(lambda: self.widget.setCurrentIndex(0))
 
        layout_C2H = QHBoxLayout()
        layout_C2H.addWidget(btn)
        layout_C2H.addStretch()
        layout_C2H.addSpacing(150)
        layout_C2H.addWidget(Title)
        layout_C2H.addSpacing(400)
        layout_C2H.addStretch()
 
        layout_C2V = QVBoxLayout()
        layout_C2V.addSpacing(400)
        layout_C2V.addLayout(layout_C2H)
        layout_C2V.addStretch()
 
        self.setLayout(layout_C2V)
 
    def toolbar_button_clicked(self, s):
        print("click", s)
 
class PageProf(QWidget):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
 
        Title = QLabel("Prof de Poche", self)
        Title.setFont(QFont("Lexend", 32, QFont.Weight.Bold))
        Title.setStyleSheet("color: black;")
 
        btn = QPushButton("Accueil", self)
        btn.setFont(QFont("Times New Roman", 12, QFont.Weight.Bold))
        btn.setFixedHeight(50)
        btn.setFixedWidth(200)
        btn.setStyleSheet('background-color:grey')
        btn.clicked.connect(lambda: self.widget.setCurrentIndex(0))
 
        layout_P2H = QHBoxLayout()
        layout_P2H.addWidget(btn)
        layout_P2H.addStretch()
        layout_P2H.addSpacing(150)
        layout_P2H.addWidget(Title)
        layout_P2H.addSpacing(400)
        layout_P2H.addStretch()
 
        layout_P2V = QVBoxLayout()
        layout_P2V.addLayout(layout_P2H)
        layout_P2V.addStretch()
 
        self.setLayout(layout_P2V)
 
 
class Site(QWidget):
    def __init__(self):
        super().__init__()
 
        self.setWindowTitle("KnowLedgeTracer")
        #self.setWindowIcon(QIcon("/home/neuro/Documents/Inertface_learning/qt.png"))
        self.resize(1200, 700)
        self.setStyleSheet('background-color:white')
 
        self.widget = QStackedWidget()
        self.Window1 = Window1(self.widget)
        self.p_prof = PageProf(self.widget)
        self.p_chr = PageChercheur(self.widget)
 
        self.widget.addWidget(self.Window1)  
        self.widget.addWidget(self.p_prof)       
        self.widget.addWidget(self.p_chr)
 
        layout_Prcp = QVBoxLayout()
        layout_Prcp.addWidget(self.widget)
        self.setLayout(layout_Prcp)
        
app = QApplication([])
window = Site()
window.show()
sys.exit(app.exec())