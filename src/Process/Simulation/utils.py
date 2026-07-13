

import numpy as np
import pandas as pd
import src.Process.DAS3H as das3H



PROFILES = {
    "maitrise":       {"n": 9, "window": (0.5, 1.0), "success_rate": 0.90},
    "en_cours":       {"n": 4, "window": (0.6, 1.0), "success_rate": 0.50},
    "ancien_oublie":  {"n": 5, "window": (0.0, 0.3), "success_rate": 0.70},
    "jamais_vu":      {"n": 0, "window": (0.0, 0.0), "success_rate": 0.0},
}

MASTERY_ORDER = ["jamais_vu", "ancien_oublie", "en_cours", "maitrise"]
RANK = {name: i for i, name in enumerate(MASTERY_ORDER)}
def assign_coherent_profiles(structure, leaf_profiles):
    profile_by_name = {}

    def resolve(name):
        if name in profile_by_name:
            return profile_by_name[name]
        children = structure.get(name, [])
        if not children:
            profile_by_name[name] = leaf_profiles.get(name, "jamais_vu")
            return profile_by_name[name]
        child_ranks = [RANK[resolve(child)] for child in children]
        parent_rank = min(child_ranks)
        profile_by_name[name] = MASTERY_ORDER[parent_rank]
        return profile_by_name[name]

    all_names = set(structure.keys())
    for kids in structure.values():
        all_names.update(kids)
    for name in all_names:
        resolve(name)
    return profile_by_name

def generate_qmatrix_controlled(n_kcs, items_per_kc=3):
    n_items = n_kcs * items_per_kc
    qmat = np.zeros((n_items, n_kcs), dtype=int)
    for kc in range(n_kcs):
        for k in range(items_per_kc):
            qmat[kc * items_per_kc + k, kc] = 1
    return qmat


def generate_skill_tree_json(structure):
    data_js = []
    name_to_id = {}
    all_names = []
    for parent, children in structure.items():
        if parent not in all_names:
            all_names.append(parent)
        for c in children:
            if c not in all_names:
                all_names.append(c)

    for i, name in enumerate(all_names, start=1):
        name_to_id[name] = str(i)
    parent_of = {}
    for parent, children in structure.items():
        for c in children:
            parent_of[c] = parent

    for name in all_names:
        pid = name_to_id[parent_of[name]] if name in parent_of else "0"
        data_js.append({
            "id": name_to_id[name],
            "name": name,
            "parent": pid,
        })
    kc_list = all_names
    return data_js, kc_list

def CreateHistoryStudent_Scenario(students, qmatrix, profile_per_kc,
                                   n_days=10, t0=0, seed=42):
    rng = np.random.default_rng(seed)
    total_seconds = n_days * 24 * 3600
    rows = []
    for std in students:
        prof_map = profile_per_kc[std] if std in profile_per_kc else profile_per_kc
        for kc, profile_name in prof_map.items():
            if profile_name not in PROFILES:
                raise ValueError(f"Profil inconnu '{profile_name}'")
            prof = PROFILES[profile_name]
            n_inter = prof["n"]
            if n_inter == 0:
                continue
            t_start = t0 + prof["window"][0] * total_seconds
            t_end = t0 + prof["window"][1] * total_seconds
            timestamps = np.sort(rng.uniform(t_start, t_end, size=n_inter))
            for ts in timestamps:
                item = _pick_item_for_kc(kc, qmatrix, rng)
                if item is None:
                    continue
                correct = int(rng.random() < prof["success_rate"])
                rows.append({
                    "user_id": std, "item_id": item, "KC": int(kc),
                    "timestamp": float(ts), "correct": correct, "inter_id": -1,
                })
    return _finalize_history(rows)

def _pick_item_for_kc(kc, qmatrix, rng):
    items = np.where(qmatrix[:, kc] == 1)[0]
    if len(items) == 0:
        return None
    return int(rng.choice(items))

def _finalize_history(rows):
    if not rows:
        return pd.DataFrame(columns=["user_id", "item_id", "KC", "timestamp",
                                      "correct", "inter_id"])
    df = pd.DataFrame(rows)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["inter_id"] = np.arange(len(df))
    return df


def compute_pmr_for_kc(kc, params, queues, t_eval, alpha_s, items_per_kc):
    items = items_per_kc.get(kc, [])
    if len(items) == 0:
        delta_j = -1
    else:
        delta_j = params["delta_j"][int(items[0])]

    beta = params["beta_j"].get(kc, 0)

    if kc in queues:
        cw = queues[kc]["wins"].get_counters(t_eval)
        ca = queues[kc]["attempts"].get_counters(t_eval)
    else:
        cw = [0] * 5
        ca = [0] * 5

    h = sum(
        params["theta_wins"][kc][i] * np.log(1 + cw[i]) +
        params["theta_attempts"][kc][i] * np.log(1 + ca[i])
        for i in range(5)
    )
    logit = alpha_s - delta_j + beta + h
    return 1 / (1 + np.exp(-logit))

def classify_mastery(pmr):
        if pmr >= 0.7:
            return "maîtrisé"
        elif pmr >= 0.4:
            return "en cours"
        elif pmr > 0:
            return "fragile"
        else:
            return "jamais vu"
def generate_random_students(structure, kc_list, n_students,
                             rng, start_id=1000):
    """
    Crée n_students élèves avec des profils de FEUILLES tirés au hasard,
    puis dérive les pères pour garantir la cohérence enfant >= père.
    Retourne (profile_per_eleve, alpha_par_eleve, student_ids).
    """
    feuilles = [n for n in kc_list if n not in structure]
    profile_per_eleve = {}
    alpha_par_eleve = {}
    student_ids = []

    for k in range(n_students):
        std = start_id + k
        student_ids.append(std)

        # profil aléatoire pour chaque feuille
        leaf_profiles = {
            f: rng.choice(MASTERY_ORDER) for f in feuilles
        }
        prof_names = assign_coherent_profiles(structure, leaf_profiles)

        # nom -> indice
        profile_per_eleve[std] = {
            idx: prof_names[name]
            for idx, name in enumerate(kc_list)
            if name in prof_names
        }
        # alpha (capacité de l'élève) tiré dans une plage réaliste
        alpha_par_eleve[std] = float(rng.normal(0, 1))

    return profile_per_eleve, alpha_par_eleve, student_ids