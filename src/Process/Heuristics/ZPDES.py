"""
=====================================================================
  Methode de recommandation d'exercices fondee sur la ZPD
  -------------------------------------------------------------------
  Alternative FRUGALE a DAS3H : ne predit rien, mesure le progres reel,
  recommande l'exercice qui maximise l'apprentissage dans la zone
  proximale, et consolide les acquis via un mecanisme d'oubli.

  Difference avec DAS3H :
    - DAS3H  : predictif, modele factoriel, gourmand en donnees
    - ICI    : reactif, taux de reussite empirique, sans entrainement

  Le coeur de la methode est le SCORE A TROIS TERMES (etape 3) :
      score(ex) = alpha * progres        (learning progress)
                + beta  * proximite_zone (ni trop facile ni trop dur)
                + gamma * besoin_revision (consolidation / anti-oubli)
=====================================================================
"""

import numpy as np


class RecoZPD:
    def __init__(self, Graph_ks: dict, Ex_Kc: dict,
                 eps_v: float = 0.4, eps_r: float = 0.85,
                 # --- poids du score a 3 termes (etape 3) ---
                 alpha: float = 1.0,   # poids du progres d'apprentissage
                 beta: float = 1.0,    # poids de la proximite a la zone
                 gamma: float = 0.6,   # poids du besoin de revision
                 zone_cible: float = 0.6,   # taux de reussite "ideal" (coeur ZPD)
                 # --- consolidation / oubli (etape 5) ---
                 horizon_oubli: int = 25,   # apres N etapes sans pratique, on revoit
                 explore: float = 0.1,      # part d'exploration aleatoire
                 window: int = 6,           # fenetre pour mesurer le progres
                 learn_rate: float = 0.08,  # vitesse d'apprentissage (SIMULATION)
                 oubli_rate: float = 0.0,   # vitesse d'oubli (SIMULATION, 0 = pas d'oubli)
                 seed: int = None):

        self.Graph_ks = Graph_ks
        self.Ex_Kc = dict(Ex_Kc)
        self.eps_v, self.eps_r = eps_v, eps_r
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.zone_cible = zone_cible
        self.horizon_oubli = horizon_oubli
        self.explore = explore
        self.window = window
        self.learn_rate = learn_rate
        self.oubli_rate = oubli_rate

        # Ensembles d'etat (ZPD)
        self.Z_e, self.D_e, self.V_e = set(), set(), set()
        self.Z_k, self.V_k = set(), set()

        # Statistiques par exercice
        self.su, self.attempts, self.success = {}, {}, {}
        self.recent = {}            # historique recent (pour le progres)
        self.last_seen = {}         # derniere etape ou l'exo a ete propose

        # Maitrise "vraie" de l'apprenant simule (cachee en usage reel)
        self.skill = {kc: 0.2 for kc in self.Graph_ks}

        self._t = 0                 # horloge globale (nb d'exercices proposes)
        self._atomic_counter = 0
        for ex in self.Ex_Kc:
            self._init_stats(ex)
        self.rng = np.random.default_rng(seed)

    def _init_stats(self, ex):
        self.su[ex] = 0.0
        self.attempts[ex] = 0
        self.success[ex] = 0
        self.recent[ex] = []
        self.last_seen[ex] = 0

    # ============================================================
    # ETAPE 1 : initialiser la zone proximale
    # ============================================================
    def initZPD(self):
        for kc, pre in self.Graph_ks.items():
            if not pre:
                self.Z_k.add(kc)
        self._ensure_atomic_exercises()
        for ex, kcs in self.Ex_Kc.items():
            if all(kc in self.Z_k for kc in kcs):
                self.Z_e.add(ex)

    def _ensure_atomic_exercises(self):
        """Garantit qu'aucune KC activee non-validee n'est orpheline."""
        for kc in list(self.Z_k):
            if kc in self.V_k:
                continue
            if not any(self.Ex_Kc[ex] == [kc] for ex in self.Ex_Kc):
                self._atomic_counter += 1
                name = f"Atom_{kc}_{self._atomic_counter}"
                self.Ex_Kc[name] = [kc]
                self._init_stats(name)

    # ============================================================
    # ETAPE 2 : estimer la maitrise (taux de reussite empirique)
    #           -> deja porte par self.su, rien de lourd a entrainer
    # ============================================================
    def learning_progress(self, ex):
        """Progres = reussite moyenne recente - reussite moyenne anterieure."""
        r = self.recent.get(ex, [])
        if len(r) < 2:
            return 0.5
        w = min(self.window, len(r) // 2)
        recent_half = np.mean(r[-w:])
        older_half = np.mean(r[-2 * w:-w]) if len(r) >= 2 * w else np.mean(r[:-w])
        return abs(recent_half - older_half)

    def proximite_zone(self, ex):
        """1.0 si le taux de reussite est au coeur de la ZPD, 0 aux extremes."""
        if self.attempts[ex] == 0:
            return 0.5                      # inconnu -> valeur moyenne, on explore
        return 1.0 - abs(self.su[ex] - self.zone_cible)

    def besoin_revision(self, ex):
        """Monte avec le temps ecoule depuis la derniere fois qu'on a vu l'exo."""
        age = self._t - self.last_seen[ex]
        return min(1.0, age / self.horizon_oubli)

    # ============================================================
    # ETAPE 3 : choisir l'exercice via le SCORE A TROIS TERMES
    # ============================================================
    def score(self, ex):
        return (self.alpha * self.learning_progress(ex)
                + self.beta * self.proximite_zone(ex)
                + self.gamma * self.besoin_revision(ex))

    def recommander(self):
        """Renvoie l'exercice recommande dans la zone active Z_e."""
        exos = sorted(self.Z_e)
        if not exos:
            return None
        if self.rng.uniform(0, 1) < self.explore:
            return self.rng.choice(exos)
        scores = np.array([self.score(e) for e in exos])
        # softmax pour transformer les scores en probabilites de choix
        w = np.exp(scores - scores.max())
        w /= w.sum()
        return self.rng.choice(exos, p=w)

    # ============================================================
    # ETAPE 4 : enregistrer le resultat (en usage reel : l'eleve repond)
    # ============================================================
    def soumettre(self, ex, reussi: bool):
        self._t += 1
        self.last_seen[ex] = self._t
        self.attempts[ex] += 1
        if reussi:
            self.success[ex] += 1
        self.su[ex] = self.success[ex] / self.attempts[ex]
        self.recent[ex].append(1 if reussi else 0)
        if len(self.recent[ex]) > 2 * self.window:
            self.recent[ex] = self.recent[ex][-2 * self.window:]
        self.update_zpd()

    # version simulee : l'apprenant repond selon sa maitrise (cachee)
    def _simuler_reponse(self, ex):
        p = np.mean([self.skill[kc] for kc in self.Ex_Kc[ex]])
        reussi = self.rng.uniform(0, 1) < p
        for kc in self.Ex_Kc[ex]:
            self.skill[kc] = min(1.0, self.skill[kc] + self.learn_rate)
        # oubli des autres KC (si oubli_rate > 0)
        if self.oubli_rate > 0:
            for kc in self.skill:
                if kc not in self.Ex_Kc[ex]:
                    self.skill[kc] = max(0.0, self.skill[kc] - self.oubli_rate)
        return reussi

    # ============================================================
    # ETAPE 5 : mettre a jour la ZPD ET consolider (revisions)
    # ============================================================
    def update_validations(self):
        for ex in list(self.Z_e):
            if self.su[ex] >= self.eps_v:
                self.V_e.add(ex)
                for kc in self.Ex_Kc[ex]:
                    self.V_k.add(kc)

    def update_activations(self):
        for kc, pre in self.Graph_ks.items():
            if kc not in self.Z_k and pre and all(p in self.V_k for p in pre):
                self.Z_k.add(kc)

    def update_added_exercises(self):
        for ex, kcs in self.Ex_Kc.items():
            if ex not in self.Z_e and ex not in self.D_e and all(kc in self.Z_k for kc in kcs):
                self.Z_e.add(ex)

    def update_removals(self):
        # on ne retire que si la KC est validee ET l'exo vraiment maitrise
        for ex in list(self.Z_e):
            if self.su[ex] >= self.eps_r and all(kc in self.V_k for kc in self.Ex_Kc[ex]):
                self.Z_e.discard(ex)
                self.D_e.add(ex)

    def reactiver_revisions(self):
        """CONSOLIDATION : un exo retire depuis trop longtemps revient
        dans la zone pour une revision (anti-oubli)."""
        for ex in list(self.D_e):
            age = self._t - self.last_seen[ex]
            if age >= self.horizon_oubli:
                self.D_e.discard(ex)
                self.Z_e.add(ex)

    def update_zpd(self):
        changed = True
        while changed:
            before = (len(self.V_e), len(self.V_k), len(self.Z_k), len(self.Z_e), len(self.Ex_Kc))
            self.update_validations()
            self.update_activations()
            self._ensure_atomic_exercises()
            self.update_added_exercises()
            after = (len(self.V_e), len(self.V_k), len(self.Z_k), len(self.Z_e), len(self.Ex_Kc))
            changed = before != after
        self.update_removals()
        self.reactiver_revisions()

    # ============================================================
    # boucle complete (SIMULATION pour tester la methode)
    # ============================================================
    def run(self, max_steps=2000, verbose=False):
        self.initZPD()
        self.update_zpd()
        step = 0
        while self.Z_e and step < max_steps:
            ex = self.recommander()
            if ex is None:
                break
            reussi = self._simuler_reponse(ex)
            self.soumettre(ex, reussi)
            step += 1
            if verbose:
                comp = (f"prog={self.learning_progress(ex):.2f} "
                        f"zone={self.proximite_zone(ex):.2f} "
                        f"rev={self.besoin_revision(ex):.2f}")
            if verbose:
                print(f"{step:3d} | {ex:14s} su={self.su[ex]:.2f} | {comp} | Z_e={len(self.Z_e)}")
        return step

    def report(self):
        print("KC validees :", sorted(self.V_k))
        print("Exos actifs :", sorted(self.Z_e))
        print("Exos retires:", sorted(self.D_e))
        print("Total exos  :", len(self.Ex_Kc))


# ---------- exemple du sujet ----------
if __name__ == "__main__":
    Ex_kc = {"Ex1": ["kc1", "kc3"], "Ex2": ["kc1"], "Ex3": ["kc1", "kc2"], "Ex4": ["kc4"]}
    Graph_kc = {"kc1": ["kc2", "kc3"], "kc2": ["kc4"], "kc3": [], "kc4": []}

    print("=== SANS oubli (apprentissage simple) ===")
    ob = RecoZPD(Graph_ks=Graph_kc, Ex_Kc=Ex_kc, seed=2, oubli_rate=0.0)
    n = ob.run(verbose=False)
    print(f"Termine en {n} essais")
    ob.report()