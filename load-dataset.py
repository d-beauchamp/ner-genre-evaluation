import argparse
import glob
from pathlib import Path
import re
import sys
import pandas as pd

#!/usr/bin/env python3

from datasets import Dataset

print("loading Litbank dataset from local files...")

all_data_litbank = []

for f in glob.glob("LitBank/entities/tsv/*.tsv"):
    tokens = [] 
    labels = [] 
    with open(f, "r", encoding="utf-8") as file: 
        for line in file: 
            line = line.strip()
            if not line:
                continue 
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
            labels.append(parts[1])
    if tokens:
        all_data_litbank.append({"tokens": tokens, "labels": labels})

litbank_dataset = Dataset.from_list(all_data_litbank)

print("Done.")

print("loading OntoNotes dataset from local files...")

f = "OntoNotes/dataset/test.json"

ontonotes_dataset = Dataset.from_json(f)

ontonotes_id2label = {
    0: "O",
    1: "B-CARDINAL",
    2: "B-DATE",
    3: "I-DATE",
    4: "B-PERSON",
    5: "I-PERSON",
    6: "B-NORP",
    7: "B-GPE",
    8: "I-GPE",
    9: "B-LAW",
    10: "I-LAW",
    11: "B-ORG",
    12: "I-ORG", 
    13: "B-PERCENT",
    14: "I-PERCENT", 
    15: "B-ORDINAL", 
    16: "B-MONEY", 
    17: "I-MONEY", 
    18: "B-WORK_OF_ART", 
    19: "I-WORK_OF_ART", 
    20: "B-FAC", 
    21: "B-TIME", 
    22: "I-CARDINAL", 
    23: "B-LOC", 
    24: "B-QUANTITY", 
    25: "I-QUANTITY", 
    26: "I-NORP", 
    27: "I-LOC", 
    28: "B-PRODUCT", 
    29: "I-TIME", 
    30: "B-EVENT",
    31: "I-EVENT",
    32: "I-FAC",
    33: "B-LANGUAGE",
    34: "I-PRODUCT",
    35: "I-ORDINAL",
    36: "I-LANGUAGE"
}

ontonotes_dataset = ontonotes_dataset.map(lambda x: {"labels": [ontonotes_id2label.get(t) for t in x["tags"]]})

print("Done.")

print("Simplifying labels...")

def reduce_labels(dataset):
    reduced = []
    for label in dataset["labels"]:
        if label.startswith("B-") or label.startswith("I-"):
            label = label[:5]
        if label.endswith("PER") or label.endswith("LOC") or label.endswith("ORG"):
            reduced.append(label)
        else:
            reduced.append("O")
    dataset["reduced_labels"] = reduced
    return dataset

def simplify_labels(dataset):
    simple = []
    for label in dataset["labels"]:
        if label.endswith("PER"):
            simple.append("PER")
        elif label.endswith("LOC"):
            simple.append("LOC")
        elif label.endswith("ORG"):
            simple.append("ORG")
        else:
            simple.append("O")
    dataset["simple_labels"] = simple
    return dataset

print("Done.")
print("mapping and saving datasets...")
litbank_dataset = litbank_dataset.map(simplify_labels)
litbank_dataset = litbank_dataset.map(reduce_labels)
litbank_dataset.save_to_disk("converted_datasets/litbank_dataset")
ontonotes_dataset = ontonotes_dataset.map(simplify_labels)
ontonotes_dataset = ontonotes_dataset.map(reduce_labels)
ontonotes_dataset.save_to_disk("converted_datasets/ontonotes_dataset")
print("Datasets saved to disk.")
