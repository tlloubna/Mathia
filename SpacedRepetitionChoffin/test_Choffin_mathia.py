import sys
import os

extra_path = os.path.join(os.path.dirname(__file__), "..")
if extra_path not in sys.path:
    sys.path.append(extra_path)

import numpy as np
import joblib
import pandas as pd
from scipy import sparse
from collections import defaultdict
import SpacedRepetitionChoffin.student_simulator as student_simulator
import src.datamodel.Studentdata as STD
import matplotlib.pyplot as plt

# Chargement
NAME_FOLDER = "Mathiadata"
DATA_FOLDER = os.path.join("data", NAME_FOLDER)
N_STUDENTS = 25351

df = pd.read_csv(os.path.join(DATA_FOLDER, f"preprocessed_data_{N_STUDENTS}std.csv"))
q_matrix = sparse.load_npz(os.path.join(DATA_FOLDER, f"q_mat_{N_STUDENTS}std.npz")).toarray()
model = joblib.load(os.path.join(DATA_FOLDER, f"das3h_model_C0.1_{N_STUDENTS}std.pkl"))["model"]
kc_list = np.load(os.path.join(DATA_FOLDER, f"history_metadata_{N_STUDENTS}std.npz"),
                  allow_pickle=True)["kc_list"]

params = model.get_params()
kc_to_idx = {kc: i for i, kc in enumerate(kc_list)}

B = 10
list_of_ri = np.arange(1, 7)
reviews_per_step = 3

# Choisir un étudiant
StudentModel = STD.StudentDATA(df, q_matrix)
std = df["user_id"].sample(1).iloc[0]
alpha = params["alpha_s"].get(std, 0.0)
item_deltas = np.array([params["delta_j"].get(i, -1.0) for i in range(q_matrix.shape[0])])

# KCs de l'étudiant — dédoublonnés et avec au moins 1 item
student_kcs = StudentModel.get_student_kcs(df, std)
student_kcs_unique = list(dict.fromkeys(student_kcs))  # supprime doublons

kc_names = []
for kc in student_kcs_unique:
    if kc in kc_to_idx:
        global_idx = kc_to_idx[kc]
        nb_items = np.sum(q_matrix[:, global_idx])
        if nb_items > 0:
            kc_names.append(kc)
    if len(kc_names) == B:
        break

print(f"KCs sélectionnés : {kc_names}")
assert len(kc_names) == B, f"Pas assez de KCs avec items : {len(kc_names)}"

kc_global_indices = [kc_to_idx[kc] for kc in kc_names]
global_to_local = {global_idx: local_idx
                   for local_idx, global_idx in enumerate(kc_global_indices)}

# Reconstruire qmat et inv_qmat avec indices locaux
inv_qmat_choffin = {local_idx: [] for local_idx in range(B)}
for item_id in range(q_matrix.shape[0]):
    for global_idx, local_idx in global_to_local.items():
        if q_matrix[item_id, global_idx] == 1:
            if item_id not in inv_qmat_choffin[local_idx]:
                inv_qmat_choffin[local_idx].append(item_id)

qmat_choffin = {}
for item_id in range(q_matrix.shape[0]):
    local_kcs = [global_to_local[g] for g in np.where(q_matrix[item_id] == 1)[0]
                 if g in global_to_local]
    if local_kcs:
        qmat_choffin[item_id] = local_kcs

# Vérification
for i in range(B):
    assert len(inv_qmat_choffin[i]) > 0, f"KC local {i} ({kc_names[i]}) a 0 items !"
print("Tous les KCs ont des items ✓")

# Paramètres par KC local
skill_betas = np.array([params["beta_k"].get(kc_names[i], 0.0) for i in range(B)])
win_params = np.array([params["theta_wins"].get(kc_names[i], np.zeros(5)) for i in range(B)])
att_params = np.array([params["theta_attempts"].get(kc_names[i], np.zeros(5)) for i in range(B)])

print("skill_betas shape:", skill_betas.shape)
print("win_params shape:", win_params.shape)
print("att_params shape:", att_params.shape)

# Simuler no_review
simul_no = student_simulator.single_student(
    "no_review", 0, B, list_of_ri, alpha, skill_betas,
    win_params, att_params, qmat_choffin, inv_qmat_choffin,
    item_deltas.reshape(-1, 1), B,
    item_deltas.reshape(-1, 1), skill_betas, win_params, att_params,
    reviews_per_step)
res_no = simul_no.learn_and_review()

# Simuler theta_threshold
simul_th = student_simulator.single_student(
    "theta_thres_multiskill", 0.4, B, list_of_ri, alpha, skill_betas,
    win_params, att_params, qmat_choffin, inv_qmat_choffin,
    item_deltas.reshape(-1, 1), B,
    item_deltas.reshape(-1, 1), skill_betas, win_params, att_params,
    reviews_per_step)
res_th = simul_th.learn_and_review()

# Plot
weeks_ap = list(range(B))
weeks_ret = [B - 1 + ri for ri in list_of_ri]
weeks = weeks_ap + weeks_ret

pmr_no = res_no[0] + [simul_no.get_performance_metric(B - 1 + ri)[0] for ri in list_of_ri]
pmr_th = res_th[0] + [simul_th.get_performance_metric(B - 1 + ri)[0] for ri in list_of_ri]

plt.plot(weeks, pmr_no, label="no_review", color="red")
plt.plot(weeks, pmr_th, label="theta_threshold", color="blue")
plt.axvline(x=B - 1, color='gray', linestyle='--', label="Fin apprentissage")
plt.xlabel("Semaines")
plt.ylabel("PMR")
plt.title("Simulation Choffin avec paramètres MathIA")
plt.legend()
plt.show()
print("done")