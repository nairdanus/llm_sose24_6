import csv
import json

# with open("dataset.csv", "w") as out_file:
#     writer = csv.writer(out_file, delimiter=";")
#     writer.writerow(["id", "prompt", "promt_without_def", "gold"])

result = []

ORIGINAL_DATA = "data/HateWiC_T5Defs_MajorityLabels.csv"
OUTPUT_FILE = "data/data.json"

with open(ORIGINAL_DATA, "r") as f:
    spamreader = csv.reader(f, delimiter=';')
    i = 0
    for row in spamreader:
        if i == 0:
            i = 1
            continue


        _id = row[0]
        sent = row[1].replace("\"", "'")
        _def = row[4].replace("\"", "'")
        T5_def = row[5].replace("\"", "'")

        tgt = row[2]
        gold = row[6]

        prompt = f"Given this context: '{sent}', is the word '{tgt}' considered hate speech and give an explanation why."
        
        answer1 = f"{'Yes' if gold else 'No'}, because here the definition of '{tgt}' is '{_def}'."
        answer2 = f"{'No' if gold else 'Yes'}, because here the definition of '{tgt}' is '{T5_def}'."

        out = {
            "id": _id,
            "gold":gold,
            "data": {
                "prompt": prompt, 
                "answer1": answer1, 
                "answer2": answer2, 
                }
            }

        result.append(out)

        # with open("dataset.csv", "a", newline='') as out_file:
        #     writer = csv.writer(out_file, delimiter=";")
        #     writer.writerow(out)

with open(OUTPUT_FILE, "w") as f:
    json.dump({"annotations": result}, f, indent=4)