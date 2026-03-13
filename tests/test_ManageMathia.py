from pymongo import MongoClient
import csv
import json

client = MongoClient("mongodb://localhost:27017/")
db = client["mathia"]
col = db["statements"]

# Toutes les clés du tableau
COMP_KEY         = "https://xapi&46;mathia&46;education/extensions/competences"
COMP_VAL_KEY     = "https://xapi&46;mathia&46;education/extensions/competences_validees"
PREREQ_KEY       = "https://xapi&46;mathia&46;education/extensions/prerequis"
ERR_KEY          = "https://xapi&46;mathia&46;education/extensions/erreurs_type"
ERR_VAL_KEY      = "https://xapi&46;mathia&46;education/extensions/erreurs_type_validees"
ID_KEY           = "https://xapi&46;mathia&46;education/extensions/id"
NOM_EXO_KEY      = "https://xapi&46;mathia&46;education/extensions/nom_exercice"
NOM_ACT_KEY      = "https://xapi&46;mathia&46;education/extensions/nom_activite"
ID_ACT_KEY       = "https://xapi&46;mathia&46;education/extensions/id_activite"
SESSION_KEY      = "https://xapi&46;mathia&46;education/extensions/id_session"
CODE_CLASSE_KEY  = "https://xapi&46;mathia&46;education/extensions/codeclasse"
MODE_JEU_KEY     = "https://xapi&46;mathia&46;education/extensions/mode_jeu"
MODE_REP_KEY     = "https://xapi&46;mathia&46;education/extensions/mode_reponse"
REPONSE_KEY      = "https://xapi&46;mathia&46;education/extensions/reponse"
BINOME_KEY       = "https://xapi&46;mathia&46;education/extensions/binome"
HORS_LIGNE_KEY   = "https://xapi&46;mathia&46;education/extensions/hors_ligne"
RECOMM_KEY       = "https://xapi&46;mathia&46;education/extensions/recommandation"
PCT_BON_KEY      = "https://xapi&46;mathia&46;education/extensions/pourcentage_bonnes_reponses"
PCT_MAU_KEY      = "https://xapi&46;mathia&46;education/extensions/pourcentage_mauvaises_reponses"
DUR_KEY          = "https://learninglocker&46;net/result-duration"

def extract_kcs(comp_raw):
    try:
        if isinstance(comp_raw, str):
            parsed = json.loads(comp_raw)
        else:
            parsed = comp_raw
        if isinstance(parsed, dict):
            return [(str(k), str(v)) for k, v in parsed.items()]
        elif isinstance(parsed, list):
            return [(str(c.get("id","")), str(c.get("name",""))) 
                    for c in parsed if "id" in c]
    except:
        pass
    return []

query = {"verbs": "https://xapi.mathia.education/verbs/finished"}

projection = {
    "_id": 0,
    "statement.actor.account.name": 1,
    "statement.result.success": 1,
    "statement.result.duration": 1,
    "statement.timestamp": 1,
    "statement.object.definition.name": 1,
    "statement.object.definition.extensions": 1,
    "statement.context.extensions": 1,
    "metadata": 1,
}

output_path = "/home/loubna/Code Projet Mathia/Mathia/data/Mathiadata/data.csv"
count_ok, count_skip = 0, 0

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "student_id",
        "item_id",
        "item_name",
        "correct",
        "timestamp",
        "duration_sec",
        "kc_ids",
        "kc_names",
        "kc_validees_ids",
        "kc_validees_names",
        "prerequis_ids",
        "prerequis_names",
        "erreurs_type_ids",
        "erreurs_type_names",
        "erreurs_type_validees_ids",
        "erreurs_type_validees_names",
        "nom_exercice",
        "nom_activite",
        "id_activite",
        "session_id",
        "code_classe",
        "mode_jeu",
        "mode_reponse",
        "reponse",
        "binome",
        "hors_ligne",
        "recommandation",
        "pct_bonnes_reponses",
        "pct_mauvaises_reponses",
    ])

    cursor = col.find(query, projection).batch_size(500)

    for i, doc in enumerate(cursor):
        try:
            stmt    = doc["statement"]
            ctx_ext = stmt["context"]["extensions"]
            obj_ext = stmt["object"]["definition"]["extensions"]

            student_id  = stmt["actor"]["account"]["name"]
            item_id     = ctx_ext.get(ID_KEY, "")
            item_name   = stmt["object"]["definition"].get("name", {}).get("en-US", "")
            correct     = int(stmt["result"]["success"])
            timestamp   = stmt["timestamp"]

            # Durée en secondes depuis metadata
            duration_sec = doc.get("metadata", {}).get(DUR_KEY, {}).get("seconds", 0)

            # Compétences ciblées
            kcs = extract_kcs(obj_ext.get(COMP_KEY, "[]"))

            # Compétences validées
            kcs_val = extract_kcs(obj_ext.get(COMP_VAL_KEY, "[]"))

            # Prérequis
            prereqs = extract_kcs(obj_ext.get(PREREQ_KEY, "[]"))

            # Erreurs type
            errs = extract_kcs(obj_ext.get(ERR_KEY, "[]"))

            # Erreurs type validées
            errs_val = extract_kcs(obj_ext.get(ERR_VAL_KEY, "[]"))

            nom_exercice      = ctx_ext.get(NOM_EXO_KEY, "")
            nom_activite      = ctx_ext.get(NOM_ACT_KEY, "")
            id_activite       = ctx_ext.get(ID_ACT_KEY, "")
            session_id        = ctx_ext.get(SESSION_KEY, "")
            code_classe       = ctx_ext.get(CODE_CLASSE_KEY, "")
            mode_jeu          = ctx_ext.get(MODE_JEU_KEY, "")
            mode_reponse      = ctx_ext.get(MODE_REP_KEY, "")
            reponse           = ctx_ext.get(REPONSE_KEY, "")
            binome            = ctx_ext.get(BINOME_KEY, "")
            hors_ligne        = ctx_ext.get(HORS_LIGNE_KEY, "")
            recommandation    = ctx_ext.get(RECOMM_KEY, "")
            pct_bon           = ctx_ext.get(PCT_BON_KEY, "")
            pct_mau           = ctx_ext.get(PCT_MAU_KEY, "")

            if not student_id or not item_id or not kcs:
                count_skip += 1
                continue

            writer.writerow([
                student_id,
                item_id,
                item_name,
                correct,
                timestamp,
                duration_sec,
                "~~".join([k[0] for k in kcs]),
                "~~".join([k[1] for k in kcs]),
                "~~".join([k[0] for k in kcs_val]),
                "~~".join([k[1] for k in kcs_val]),
                "~~".join([k[0] for k in prereqs]),
                "~~".join([k[1] for k in prereqs]),
                "~~".join([k[0] for k in errs]),
                "~~".join([k[1] for k in errs]),
                "~~".join([k[0] for k in errs_val]),
                "~~".join([k[1] for k in errs_val]),
                nom_exercice,
                nom_activite,
                id_activite,
                session_id,
                code_classe,
                mode_jeu,
                mode_reponse,
                reponse,
                binome,
                hors_ligne,
                recommandation,
                pct_bon,
                pct_mau,
            ])
            count_ok += 1

        except Exception:
            count_skip += 1
            continue

        if i % 100_000 == 0 and i > 0:
            print(f"{i} docs | {count_ok} lignes | {count_skip} skippés")

print(f"Terminé : {count_ok} lignes, {count_skip} skippées")