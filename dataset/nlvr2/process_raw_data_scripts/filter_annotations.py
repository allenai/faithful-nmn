import csv
import sys

FIELDNAMES = ["sentence", "annotation", "tokens", "all", "simple_count"]

outfile = open("filtered_annotations.tsv", "w")
sentences = {}
for fname in sys.argv[1:]:
    finput = open(fname)
    reader = csv.DictReader(finput, delimiter="\t")
    for item in reader:
        sentence = item["sentence"].lower()
        if (
            sentence not in sentences
            and "annotation" in item
            and item["annotation"] is not None
            and len(item["annotation"]) > 0
        ):
            keep = True
            if keep:
                sentences[sentence] = item
    finput.close()
writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=FIELDNAMES)
writer.writerow({f: f for f in FIELDNAMES})
for item in sentences.values():
    res = {}
    for f in FIELDNAMES:
        if f == "all":
            res[f] = "1"
        elif f == "simple_count":
            if f in item:
                res[f] = item[f]
            else:
                res[f] = "0"
        else:
            res[f] = item[f]
    writer.writerow(res)
outfile.close()
