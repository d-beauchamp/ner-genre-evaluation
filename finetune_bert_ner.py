# TODO: check size of CoNLL to compare with LitBank/OntoNotes training set
# TODO: check where to correctly add to(device)
# TODO: remove extraneous columns from data
# TODO: add validation data for training

from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer,
                          DataCollatorForTokenClassification)
from datasets import Dataset, load_from_disk, concatenate_datasets

# Perchance create a separate dataloader file to avoid these imports and stuff
from eval_bert_ner import litbank_data, split_data

label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC", 5: "B-ORG", 6: "I-ORG"}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=7,
    id2label=id2label,
    label2id=label2id)
data_collator = DataCollatorForTokenClassification(tokenizer)


def combine_and_extract_train_sets(dirs):
    loaded_datasets = []
    for d in dirs:
        loaded_file = load_from_disk(f"converted_datasets/{d}")
        loaded_datasets.append(loaded_file)
    combined_set = concatenate_datasets(loaded_datasets)
    train, labels = combined_set["tokens"], combined_set["reduced_labels"]
    return train, labels


def token_label_alignment(features, labels, label2id):
    words_ids = features.word_ids()
    aligned_labels = []

    prev_word_idx = None
    for word_idx in words_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != prev_word_idx:
            aligned_labels.append(label2id[labels[word_idx]])
        else:
            aligned_labels.append(-100)

        prev_word_idx = word_idx

    return aligned_labels


def tokenize_and_align(texts, labels):
    encodings = []
    for text_segment, label_sequence in zip(texts, labels):
        encoding = tokenizer(text_segment, is_split_into_words=True, truncation=True)
        alignment = token_label_alignment(encoding, label_sequence, label2id)

        encoding["labels"] = alignment

        encodings.append(encoding)

    features = Dataset.from_list(encodings)
    return features


def main():
    litbank_split = split_data(litbank_data)
    litbank_train = litbank_split[0]
    litbank_train_labels = litbank_split[2]

    ontonotes_sets = ["ontonotes_dataset_train00", "ontonotes_dataset_train01"]
    ontonotes_train, ontonotes_labels = combine_and_extract_train_sets(ontonotes_sets)

    litbank_feats = tokenize_and_align(litbank_train, litbank_train_labels)
    ontonotes_feats = tokenize_and_align(ontonotes_train, ontonotes_labels)
    mixed_feats = concatenate_datasets([litbank_feats, ontonotes_feats])

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mixed_feats,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # trainer.train()

    finetuned_tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-11280")
    finetuned_model = AutoModelForTokenClassification.from_pretrained("./results/checkpoint-11280")

    finetuned_tokenizer.save_pretrained("./finetuned_model")
    finetuned_model.save_pretrained("./finetuned_model")


if __name__ == "__main__":
    main()
