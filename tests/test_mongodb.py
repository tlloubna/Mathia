from pymongo import MongoClient
import json
import pandas as pd
import csv
client=MongoClient("mongodb://localhost:27017/")
print(client.list_database_names())

db = client["mathia"]
col = db["statements"]
#print(col.count_documents({})) # 11 662 285 interactions
doc = col.find_one()
#print(json.dumps(doc, indent=2, default=str))

#print(col.distinct("statement.verb.id")) #verb [completed,failed,failed-on-first-attempt,passed-on-retry,passed-with_help,resume]

#print(len(col.distinct("statement.actor.account.name"))) #élèves 25351
#nb_exercices = len(col.distinct("statement.context.extensions.https://xapi&46;mathia&46;education/extensions/id"))
#print("Nb exercices :", nb_exercices) 708

#plage de temps de la base de données est 01/09/2024 au 02/03/2026
# Garder seulement les verbes avec une réponse claire
"""verbes_utiles = [
    "https://xapi.mathia.education/verbs/passed",
    "https://xapi.mathia.education/verbs/passed-on-retry",
    "https://xapi.mathia.education/verbs/passed-with-help",
    "https://xapi.mathia.education/verbs/failed",
    "https://xapi.mathia.education/verbs/failed-on-first-attempt",
   
]

nb_utiles = col.count_documents({
    "statement.verb.id": {"$in": verbes_utiles}
})
nb_total = col.count_documents({})

print("Total documents     :", nb_total) # 11 662 285 interactions
print("Interactions utiles :", nb_utiles) # 4 823 536
print("Pourcentage gardé   :", round(nb_utiles / nb_total * 100, 1), "%") #41.4
#Taux de réussite global : 80.4%%"""

"""
passed                    | True:  3599038 | False:      193 | Absent:        0
passed-on-retry           | True:    53685 | False:        3 | Absent:        0
passed-with-help          | True:   245576 | False:       35 | Absent:        0
failed                    | True:       43 | False:   239832 | Absent:        0
failed-on-first-attempt   | True:      183 | False:   684948 | Absent:        0
finished                  | True:  3879536 | False:   943436 | Absent:        0
completed                 | True:   540234 | False:        0 | Absent:    43228
initialized               | True:        0 | False:        0 | Absent:  1071854
resume                    | True:        0 | False:        0 | Absent:   360461"""

#Compute the number of competences (Total)

"""competences = col.distinct(
    "statement.object.definition.extensions.https://xapi&46;mathia&46;education/extensions/competences"
)

kcs_uniques = set()

for valeur in competences:
    if not valeur:
        continue
    
    if isinstance(valeur, str):
        try:
            parsed = json.loads(valeur)
        except json.JSONDecodeError:
            continue
    else:
        parsed = valeur
    
    # Format 1 : dict {"132": "Nommer..."} ou {"64": "...", "165": "..."}
    if isinstance(parsed, dict):
        for kc_id in parsed.keys():
            kcs_uniques.add(str(kc_id))
    
    # Format 2 : liste [{"id": "2570", "name": "..."}]
    elif isinstance(parsed, list):
        for kc in parsed:
            if isinstance(kc, dict) and "id" in kc:
                kcs_uniques.add(str(kc["id"]))

print("Nb KCs uniques :", len(kcs_uniques)) # 184"""
# Extraire les interactions avec student_id, kc_id, timestamp
# Calculer les stats par paire (élève, KC) directement dans MongoDB


# Combien de documents finished n'ont pas de compétences ?
nb_sans_comp = col.count_documents({
    "verbs": "https://xapi.mathia.education/verbs/finished",
    "statement.object.definition.extensions.https://xapi&46;mathia&46;education/extensions/competences": {"$exists": False}
})

nb_comp_vide = col.count_documents({
    "verbs": "https://xapi.mathia.education/verbs/finished",
    "statement.object.definition.extensions.https://xapi&46;mathia&46;education/extensions/competences": "[]"
})

print("Sans champ competences :", nb_sans_comp)
print("Competences vide []    :", nb_comp_vide)
print("Total skippés probable :", nb_sans_comp + nb_comp_vide)
