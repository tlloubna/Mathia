from PyQt6.QtWidgets import (
    QApplication, QLabel, QHBoxLayout, QVBoxLayout,
    QMainWindow, QWidget, QTabWidget, QComboBox, QFileDialog, QFrame, QSplitter
)
from PyQt6.QtGui import QIcon, QPixmap, QAction
from PyQt6.QtCore import Qt
import datetime
import logging
import os
import sys
import json
import csv

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class KnwoledgeTracer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.style = "color: black; background-color: #d3d3d3;"
        # Dictionnaire qui stocke les meta-donnees ET les interactions de chaque fichier.
        # Cle = nom du fichier.
        # Valeur = dict : {type, eleves, exercices, competences, interactions}
        #   interactions = liste de dicts {"eleve":..., "kc":..., "reussite": 0 ou 1}
        # IMPORTANT : a creer AVANT CreateGui().
        self.donnees_fichiers = {}
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
        self.menubar.setStyleSheet("color: black; background-color: #d3d3d3;border: 1px solid black;")

        File_import = QAction("Importer un fichier", self)
        File_import.triggered.connect(self.ouvrir_fichier)

        file_menu.addAction(File_import)

    def CreateVlayout(self):

        main_layout = QHBoxLayout()
        central = QWidget()
        self.left_panel = QWidget()
        self.left_panel.setStyleSheet(self.style)
        self.left_layout = QVBoxLayout()
        self.rec = QFrame()
        self.rec.setStyleSheet("color: black; background-color: gray;")
        in_rec = QVBoxLayout(self.rec)
        in_rec.setSpacing(10)
        in_rec.setContentsMargins(10, 10, 10, 10)
        self.combobox()
        self.combobox.setStyleSheet("color: white; background-color: #595858; border: 1px solid black;")

        self.info = QLabel("            Information dataset selectionne")
        self.info_type = QLabel("Type : ")
        self.info_nbeleves = QLabel("Nombres d'eleves : ")
        self.info_nbexercise = QLabel("Nombres d'exercices : ")
        self.info_nbcompetance = QLabel("Nombres de competance : ")

        self.sep = QSplitter(Qt.Orientation.Vertical)

        self.sep.setStyleSheet("""
        QSplitter::handle {
        background-color: #595858;
        height: 2px; /* Ligne fine de 2 pixels */
                        }
            QLabel {
        margin: 0px;
        padding: 0px;
        line-height: 100%;
        }
            """)
        self.sep.addWidget(self.info)
        self.sep.addWidget(self.info_type)
        self.sep.addWidget(self.info_nbeleves)
        self.sep.addWidget(self.info_nbexercise)
        self.sep.addWidget(self.info_nbcompetance)

        in_rec.addWidget(self.combobox)
        in_rec.addSpacing(20)
        in_rec.addWidget(self.sep)
        in_rec.addStretch()

        self.left_layout.addWidget(self.rec)

        self.left_layout.addStretch()
        self.left_panel.setLayout(self.left_layout)

        # ---------- ONGLET GRAPH ----------
        self.tabs = QTabWidget()

        # On construit un widget "Graph" qui contient :
        #   - une barre du haut avec 2 ComboBox (eleve + KC)
        #   - la zone matplotlib en dessous
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)

        # Barre du haut avec les deux selecteurs
        barre_selecteurs = QHBoxLayout()

        barre_selecteurs.addWidget(QLabel("Eleve :"))
        self.combo_eleve = QComboBox()
        # Quand on change l'eleve -> on retrace le graphe
        self.combo_eleve.currentTextChanged.connect(self.tracer_courbe_apprentissage)
        barre_selecteurs.addWidget(self.combo_eleve)

        barre_selecteurs.addWidget(QLabel("KC :"))
        self.combo_kc = QComboBox()
        # Quand on change la KC -> on retrace le graphe
        self.combo_kc.currentTextChanged.connect(self.tracer_courbe_apprentissage)
        barre_selecteurs.addWidget(self.combo_kc)

        barre_selecteurs.addStretch()  # pousse les selecteurs vers la gauche

        graph_layout.addLayout(barre_selecteurs)

        # Zone matplotlib
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        graph_layout.addWidget(self.canvas)

        self.tabs.addTab(graph_widget, "Graph")

        # ---------- AUTRES ONGLETS ----------
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

    def combobox(self):
        self.combobox = QComboBox()
        # Quand on change de fichier -> on met a jour infos ET selecteurs eleve/KC.
        self.combobox.currentTextChanged.connect(self.changer_dataset)
        self.left_layout.addSpacing(20)
        self.left_layout.addWidget(self.combobox)

    # --------- AFFICHAGE / CHANGEMENT DE DATASET ---------

    def afficher_infos(self, nom_fichier):
        """Ecrit dans les labels les infos du fichier demande."""
        infos = self.donnees_fichiers.get(nom_fichier, {})
        self.info_type.setText("Type : " + infos.get("type", ""))
        self.info_nbeleves.setText(f"Nombres d'eleves : {infos.get('eleves', '')}")
        self.info_nbexercise.setText(f"Nombres d'exercices : {infos.get('exercices', '')}")
        self.info_nbcompetance.setText(f"Nombres de competences : {infos.get('competences', '')}")

    def changer_dataset(self, nom_fichier):
        """Appelee quand la combobox des fichiers change."""
        self.afficher_infos(nom_fichier)
        self.remplir_selecteurs(nom_fichier)

    def remplir_selecteurs(self, nom_fichier):
        """Remplit les ComboBox eleve et KC a partir des interactions du fichier."""
        infos = self.donnees_fichiers.get(nom_fichier, {})
        interactions = infos.get("interactions", [])

        # On recupere la liste triee des eleves et des KC presents.
        eleves = sorted({inter["eleve"] for inter in interactions})
        kcs = sorted({inter["kc"] for inter in interactions})

        # On bloque temporairement les signaux pour eviter de tracer
        # plein de fois pendant qu'on remplit les listes.
        self.combo_eleve.blockSignals(True)
        self.combo_kc.blockSignals(True)

        self.combo_eleve.clear()
        self.combo_eleve.addItems(eleves)
        self.combo_kc.clear()
        self.combo_kc.addItems(kcs)

        self.combo_eleve.blockSignals(False)
        self.combo_kc.blockSignals(False)

        # On trace une fois maintenant que tout est rempli.
        self.tracer_courbe_apprentissage()

    # --------- TRACE DU GRAPHE ---------

    def tracer_courbe_apprentissage(self):
        """Trace P(succes) au fil des tentatives pour l'eleve + KC selectionnes."""
        nom_fichier = self.combobox.currentText()
        eleve = self.combo_eleve.currentText()
        kc = self.combo_kc.currentText()

        # On vide la figure avant de redessiner.
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        infos = self.donnees_fichiers.get(nom_fichier, {})
        interactions = infos.get("interactions", [])

        # On filtre : on garde les interactions de cet eleve sur cette KC,
        # dans l'ordre ou elles ont ete lues (= ordre chronologique du fichier).
        suite = [inter["reussite"] for inter in interactions
                 if inter["eleve"] == eleve and inter["kc"] == kc]

        if not suite:
            ax.set_title("Aucune donnee pour cette selection")
            self.canvas.draw()
            return

        # Moyenne mobile cumulee : a chaque tentative, P(succes) = moyenne
        # des reussites depuis le debut. C'est ce qui donne la courbe qui monte.
        x = list(range(1, len(suite) + 1))
        y = []
        total = 0
        for i, r in enumerate(suite):
            total += r
            y.append(total / (i + 1))

        ax.plot(x, y, marker='o', color='#4f8ef7')
        ax.set_title(f"Courbe d'apprentissage - {eleve} / {kc}")
        ax.set_xlabel("Numero de tentative")
        ax.set_ylabel("P(succes) cumulee")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        self.canvas.draw()

    # --------- IMPORT DE FICHIER ---------

    def ouvrir_fichier(self):
        filtre = "*.json *.txt *.csv"
        chemin, _ = QFileDialog.getOpenFileName(self, "fichier", "", filtre)

        if chemin:
            extension = os.path.splitext(chemin)[1].lower()
            name = chemin.split("/")[-1]

            if extension == '.json':
                self.charger_json(chemin, name)
            elif extension in ['.csv', '.txt']:
                self.charger_texte_csv(chemin, name)

            self.combobox.addItem(name)
            self.combobox.setCurrentText(name)

    def charger_json(self, chemin, nom):
        try:
            with open(chemin, 'r', encoding='utf-8') as f:
                donnees = json.load(f)

            eleves = set()
            exercices = set()
            competences = set()
            interactions = []  # NOUVEAU : la liste des interactions

            statements = donnees if isinstance(donnees, list) else [donnees]

            for item in statements:
                st = item.get("statement", item)

                # Eleve
                actor = st.get("actor", {})
                agent_id = item.get("agents", [None])[0] if "agents" in item else actor.get("mbox")
                if agent_id:
                    eleves.add(agent_id)

                # Exercice
                obj = st.get("object", {})
                obj_id = obj.get("id")
                if obj_id:
                    exercices.add(obj_id)

                # Competences
                ext = obj.get("definition", {}).get("extensions", {})
                kc_courante = None
                for k, v in ext.items():
                    if "competence" in k.lower() or "skill" in k.lower():
                        competences.add(str(v))
                        kc_courante = str(v)

                # Reussite : on regarde le verbe ou le score xAPI.
                reussite = self._reussite_depuis_xapi(st)

                # On enregistre l'interaction si on a au moins eleve + KC + reussite.
                if agent_id and kc_courante is not None and reussite is not None:
                    interactions.append({
                        "eleve": agent_id,
                        "kc": kc_courante,
                        "reussite": reussite,
                    })

            self.donnees_fichiers[nom] = {
                "type": "JSON (xAPI)",
                "eleves": len(eleves),
                "exercices": len(exercices),
                "competences": len(competences) if competences else "Non detecte",
                "interactions": interactions,
            }

        except Exception as e:
            print(f"Erreur JSON : {e}")
            self.donnees_fichiers[nom] = {"type": "Erreur de lecture JSON", "interactions": []}

    def _reussite_depuis_xapi(self, statement):
        """Renvoie 1 (reussi), 0 (echoue) ou None (inconnu) pour un statement xAPI."""
        # 1. On essaie via le resultat (success / score)
        result = statement.get("result", {})
        if isinstance(result, dict):
            if "success" in result:
                return 1 if result["success"] else 0
            score = result.get("score", {})
            if isinstance(score, dict) and "scaled" in score:
                # scaled va de 0 a 1 : on considere >= 0.5 comme reussi.
                return 1 if score["scaled"] >= 0.5 else 0

        # 2. Sinon via le verbe
        verbe = statement.get("verb", {})
        verbe_id = (verbe.get("id") or "").lower()
        if "passed" in verbe_id or "completed" in verbe_id or "mastered" in verbe_id:
            return 1
        if "failed" in verbe_id:
            return 0

        return None

    def charger_texte_csv(self, chemin, nom):
        try:
            with open(chemin, 'r', encoding='utf-8') as f:
                echantillon = f.read(2048)
                delimiteur = '\t' if '\t' in echantillon else ','
                f.seek(0)

                lecteur = csv.reader(f, delimiter=delimiteur)
                entetes = next(lecteur, None)

                if not entetes:
                    return

                entetes_propres = [h.strip().lower() for h in entetes]

                idx_student = -1
                idx_problem = -1
                idx_kc = -1
                idx_result = -1  # NOUVEAU : colonne reussite

                for i, h in enumerate(entetes_propres):
                    if any(k in h for k in ["anon student id", "student", "eleve", "user"]):
                        idx_student = i
                    if any(k in h for k in ["problem name", "problem", "exercice", "task"]):
                        idx_problem = i
                    if any(k in h for k in ["kc(subskills)", "kc", "competence", "skill"]):
                        idx_kc = i
                    if any(k in h for k in ["correct first attempt", "outcome", "success", "correct", "result"]):
                        idx_result = i

                eleves = set()
                exercices = set()
                competences = set()
                interactions = []  # NOUVEAU

                for ligne in lecteur:
                    if not ligne:
                        continue
                    if idx_student != -1 and idx_student < len(ligne):
                        eleves.add(ligne[idx_student])
                    if idx_problem != -1 and idx_problem < len(ligne):
                        exercices.add(ligne[idx_problem])
                    if idx_kc != -1 and idx_kc < len(ligne):
                        if ligne[idx_kc].strip():
                            competences.add(ligne[idx_kc].strip())

                    # On enregistre l'interaction si on a eleve + KC + resultat.
                    if (idx_student != -1 and idx_kc != -1 and idx_result != -1
                            and idx_student < len(ligne) and idx_kc < len(ligne)
                            and idx_result < len(ligne)):
                        eleve = ligne[idx_student].strip()
                        kc = ligne[idx_kc].strip()
                        reussite = self._reussite_depuis_texte(ligne[idx_result])
                        if eleve and kc and reussite is not None:
                            interactions.append({
                                "eleve": eleve,
                                "kc": kc,
                                "reussite": reussite,
                            })

                type_str = "TSV (Tabulation)" if delimiteur == '\t' else "CSV"
                self.donnees_fichiers[nom] = {
                    "type": type_str,
                    "eleves": len(eleves),
                    "exercices": len(exercices),
                    "competences": len(competences),
                    "interactions": interactions,
                }

        except Exception as e:
            print(f"Erreur Texte : {e}")
            self.donnees_fichiers[nom] = {"type": "Erreur de lecture CSV", "interactions": []}

    def _reussite_depuis_texte(self, valeur):
        """Convertit une valeur texte (1, 0, 'correct', 'true'...) en 1, 0 ou None."""
        v = str(valeur).strip().lower()
        if v in ("1", "correct", "true", "ok", "yes", "y", "pass", "passed"):
            return 1
        if v in ("0", "incorrect", "false", "no", "n", "fail", "failed"):
            return 0
        return None


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