from pymongo import MongoClient
import csv
import json
import re


BASE_DOT = "https://xapi.mathia.education"
BASE_46  = "https://xapi&46;mathia&46;education"
BASES    = [BASE_DOT, BASE_46]

LL_DOT = "https://learninglocker.net/result-duration"
LL_46  = "https://learninglocker&46;net/result-duration"


def variants(suffix):
    return [f"{b}{suffix}" for b in BASES]


COMP_KEYS     = variants("/extensions/competences")
COMP_VAL_KEYS = variants("/extensions/competences_validees")
PREREQ_KEYS   = variants("/extensions/prerequis")
ERR_KEYS      = variants("/extensions/erreurs_type")
ERR_VAL_KEYS  = variants("/extensions/erreurs_type_validees")
ID_KEYS       = variants("/extensions/id")
CLASSE_KEYS   = variants("/extensions/codeclasse")
MODE_JEU_KEYS = variants("/extensions/mode_jeu")
TYPE_PARC_KEYS= variants("/extensions/type_parcours")
ACTIVITE_KEYS = variants("/extensions/activite")
NOM_EXO_KEYS  = variants("/extensions/nom_exercice")

PCT_BON_KEYS = (variants("/pourcentage_bonnes_reponses")
                + variants("/extensions/pourcentage_bonnes_reponses"))
PCT_MAU_KEYS = (variants("/pourcentage_mauvaises_reponses")
                + variants("/extensions/pourcentage_mauvaises_reponses"))

VERB_IDS = variants("/verbs/completed")

KC_FIELDS = [
    (COMP_KEYS,     "competences"),
    (COMP_VAL_KEYS, "competences_validees"),
    (PREREQ_KEYS,   "prerequis"),
    (ERR_KEYS,      "erreurs_type"),
    (ERR_VAL_KEYS,  "erreurs_type_validees"),
]


SEUIL_DEFAUT = 75.0
SEUILS_PAR_MODE = [
    ("aventure",   50.0),
    ("adaptatif",  75.0),
    ("bilan",      75.0),
    ("remediation",75.0),
    ("entrainement",75.0),
]


def resolve_seuil(mode_jeu, type_parcours, activite):
    """Retourne (seuil, mode_detecte) en cherchant un mot-cle connu."""
    blob = " ".join(str(x).lower() for x in (mode_jeu, type_parcours, activite) if x)
    for mot, seuil in SEUILS_PAR_MODE:
        if mot in blob:
            return seuil, mot
    return SEUIL_DEFAUT, "defaut"


CSV_HEADER = ["student_id", "item_id", "item_name", "correct", "timestamp",
              "duration_sec", "exo_url"]
for _, prefix in KC_FIELDS:
    CSV_HEADER.append(f"{prefix}_ids")
    CSV_HEADER.append(f"{prefix}_names")
CSV_HEADER += ["pct_bonnes_reponses", "pct_mauvaises_reponses", "pct_source",
               "seuil", "mode_detecte", "mode_jeu", "type_parcours",
               "activite", "code_classe"]



def first_present(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def lookup_multi(dicts, keys, default=None):
    for d in dicts:
        v = first_present(d, keys, None)
        if v is not None:
            return v
    return default


def iso8601_to_seconds(dur):
    if dur is None:
        return None
    if isinstance(dur, (int, float)):
        return float(dur)
    if not isinstance(dur, str):
        return None
    m = re.match(r'^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$', dur)
    if not m:
        return None
    days, hours, minutes, seconds = m.groups()
    total = 0.0
    if days:    total += int(days) * 86400
    if hours:   total += int(hours) * 3600
    if minutes: total += int(minutes) * 60
    if seconds: total += float(seconds)
    return total


def parse_kc(raw):
    if raw is None:
        return []
    val = raw
    if isinstance(val, str):
        val = val.strip()
        if val in ("", "[]", "{}"):
            return []
        try:
            val = json.loads(val)
        except Exception:
            return []
    pairs = []
    if isinstance(val, dict):
        pairs = [(str(k), str(v)) for k, v in val.items()]
    elif isinstance(val, list):
        for c in val:
            if isinstance(c, dict) and "id" in c:
                pairs.append((str(c.get("id", "")), str(c.get("name", ""))))
            else:
                pairs.append((str(c), str(c)))
    else:
        return []
    pairs = [(i, n) for i, n in pairs if i or n]
    pairs.sort(key=lambda p: int(p[0]) if p[0].isdigit() else 10**9)
    return pairs


def kc_ids_and_names(raw):
    pairs = parse_kc(raw)
    return "~~".join(i for i, _ in pairs), "~~".join(n for _, n in pairs)


def to_number(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "."))
    except Exception:
        return None


def extract_item_id(object_id, ctx_ext, def_ext):
    if object_id:
        m = re.search(r'/exercise/id/(\d+)', str(object_id))
        if m:
            return int(m.group(1))
    v = lookup_multi([ctx_ext, def_ext], ID_KEYS)
    n = to_number(v)
    if n is not None and float(n).is_integer():
        return int(n)
    return None


def resolve_pct(ctx_ext, def_ext):
    pct_bon = to_number(lookup_multi([ctx_ext, def_ext], PCT_BON_KEYS))
    pct_mau = to_number(lookup_multi([ctx_ext, def_ext], PCT_MAU_KEYS))
    if pct_bon is not None and pct_mau is None:
        pct_mau = 100.0 - pct_bon
    return pct_bon, pct_mau


def resolve_correct(pct_bon, seuil, success):
   
    if pct_bon is not None:
        return (1 if pct_bon >= seuil else 0), "pct"
    if isinstance(success, bool):
        return (1 if success else 0), "success"
    return None, ""

def export_mathia_to_csv(output_path,
                         mongo_uri="mongodb://localhost:27017/",
                         db_name="ralph",
                         collection="statements",
                         exclude_classe=None,
                         require_kc=True,
                         require_real_pct=False):
    client = MongoClient(mongo_uri)
    col = client[db_name][collection]
    query = {"statement.verb.id": {"$in": VERB_IDS}}
    projection = {
        "_id": 0,
        "statement.actor.account.name": 1,
        "statement.object.id": 1,
        "statement.object.definition.name": 1,
        "statement.object.definition.extensions": 1,
        "statement.result.success": 1,
        "statement.result.duration": 1,
        "statement.timestamp": 1,
        "statement.context.extensions": 1,
        "metadata": 1,
    }

    count_ok = 0
    skips = {"no_student": 0, "no_item": 0, "no_kc": 0,
             "no_correct": 0, "classe": 0, "not_exercise": 0, "erreur": 0}
    stats_src  = {"pct": 0, "success": 0}
    stats_mode = {}

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        cursor = col.find(query, projection, no_cursor_timeout=True).batch_size(500)
        for i, doc in enumerate(cursor):
            try:
                src = doc.get("statement", {}) or {}
                obj = src.get("object", {}) or {}
                definition = obj.get("definition", {}) or {}
                def_ext = definition.get("extensions", {}) or {}
                ctx_ext = (src.get("context", {}) or {}).get("extensions", {}) or {}
                result  = src.get("result", {}) or {}
                student_id = ((src.get("actor", {}) or {}).get("account", {}) or {}).get("name", "")
                student_id = str(student_id).strip()
                if not student_id:
                    skips["no_student"] += 1
                    continue
                exo_url = obj.get("id", "")
                if "/exercise/" not in str(exo_url):
                    skips["not_exercise"] += 1
                    continue
                item_id = extract_item_id(exo_url, ctx_ext, def_ext)
                if item_id is None:
                    skips["no_item"] += 1
                    continue

                name_obj = definition.get("name", {}) or {}
                item_name = name_obj.get("en-US") or name_obj.get("fr-FR") or ""

                timestamp = src.get("timestamp", "")
                if not timestamp:
                    skips["erreur"] += 1
                    continue
                success = result.get("success")
                duration_sec = iso8601_to_seconds(result.get("duration"))
                if duration_sec is None:
                    meta = doc.get("metadata", {}) or {}
                    for llk in (LL_DOT, LL_46):
                        node = meta.get(llk)
                        if isinstance(node, dict) and node.get("seconds") is not None:
                            duration_sec = to_number(node.get("seconds"))
                            break
                if duration_sec is None:
                    duration_sec = ""
                kc_cols = []
                comp_ids = ""
                for keys, prefix in KC_FIELDS:
                    raw = lookup_multi([def_ext, ctx_ext], keys)
                    ids, names = kc_ids_and_names(raw)
                    if prefix == "competences":
                        comp_ids = ids
                    kc_cols.append(ids)
                    kc_cols.append(names)
                if require_kc and not comp_ids:
                    skips["no_kc"] += 1
                    continue
                mode_jeu      = lookup_multi([ctx_ext, def_ext], MODE_JEU_KEYS, "")
                type_parcours = lookup_multi([ctx_ext, def_ext], TYPE_PARC_KEYS, "")
                activite      = lookup_multi([ctx_ext, def_ext], ACTIVITE_KEYS, "")
                seuil, mode_detecte = resolve_seuil(mode_jeu, type_parcours, activite)
                pct_bon, pct_mau = resolve_pct(ctx_ext, def_ext)
                correct, pct_source = resolve_correct(pct_bon, seuil, success)
                if correct is None:
                    skips["no_correct"] += 1
                    continue
                if require_real_pct and pct_source != "pct":
                    skips["no_correct"] += 1
                    continue

                code_classe = lookup_multi([ctx_ext, def_ext], CLASSE_KEYS, "")
                if exclude_classe is not None and str(code_classe) == str(exclude_classe):
                    skips["classe"] += 1
                    continue

                stats_src[pct_source] = stats_src.get(pct_source, 0) + 1
                stats_mode[mode_detecte] = stats_mode.get(mode_detecte, 0) + 1

                row = ([student_id, item_id, item_name, correct, timestamp,
                        duration_sec, exo_url] + kc_cols
                       + ["" if pct_bon is None else pct_bon,
                          "" if pct_mau is None else pct_mau,
                          pct_source, seuil, mode_detecte,
                          mode_jeu, type_parcours, activite, code_classe])
                writer.writerow(row)
                count_ok += 1

            except Exception:
                skips["erreur"] += 1
                continue

            if i % 100_000 == 0 and i > 0:
                print(f"  {i} docs lus | {count_ok} lignes retenues")

    client.close()
    total_skip = sum(skips.values())
    print(f"\nTermine : {count_ok} lignes -> {output_path}")
    print(f"Skippees : {total_skip}")
    for k, v in skips.items():
        if v:
            print(f"   {k:14s} : {v}")
    print("Origine de correct :")
    for k, v in stats_src.items():
        print(f"   {k:14s} : {v}")
    print("Seuil applique (mode detecte) :")
    for k, v in sorted(stats_mode.items(), key=lambda x: -x[1]):
        print(f"   {k:14s} : {v}")
    return output_path
if __name__ == "__main__":
    export_mathia_to_csv(
        output_path="data/Mathiadata_completed/data.csv",
        mongo_uri= "mongodb://localhost:27017/",
        db_name="mathia",
        collection="statements",
        exclude_classe=None,
        require_kc=True,
        require_real_pct=False,
    )
    print("done")