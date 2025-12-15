from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer,
                          DataCollatorForTokenClassification)
from datasets import Dataset, concatenate_datasets

from data_prep import load_splits, chunker

label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC", 5: "B-ORG", 6: "I-ORG"}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=7,
    id2label=id2label,
    label2id=label2id)
data_collator = DataCollatorForTokenClassification(tokenizer)


def token_label_alignment(features, labels, label2id):
    """Align subword tokens with their original labels."""
    words_ids = features.word_ids()
    aligned_labels = []

    prev_word_idx = None
    for word_idx in words_ids:
        if word_idx is None:  # CLS or SEP
            aligned_labels.append(-100)
        elif word_idx != prev_word_idx:  # Indicates new word
            aligned_labels.append(label2id[labels[word_idx]])
        else:
            aligned_labels.append(-100)

        prev_word_idx = word_idx

    return aligned_labels


def tokenize_and_align(docs, all_labels):
    """Chunk text and label inputs, then tokenize and align them to create a feature set."""
    encodings = []
    for doc, labels in zip(docs, all_labels):

        # Chunk input documents into chunks of 50 words
        # Especially important for LitBank inputs, whose ~2000 word length far exceeds
        # the BERT limit of 512 even before subword tokenization
        chunks = chunker(doc, labels, max_len=50)

        for text_segment, label_sequence in chunks:

            # If a new chunk begins with an entity continuation, convert it to the start of a new entity
            if label_sequence[0].startswith("I-"):
                label_sequence[0] = "B-" + label_sequence[0][2:]

            encoding = tokenizer(text_segment, is_split_into_words=True, truncation=True)
            alignment = token_label_alignment(encoding, label_sequence, label2id)

            encoding["labels"] = alignment

            encodings.append(encoding)

    features = Dataset.from_list(encodings)
    return features


def main():
    """Create mixed-domain feature lists for fine-tuning, then train using Hugging Face
    Trainer and save the best resulting model."""

    splits = load_splits()

    litbank_train, litbank_train_labels = splits["litbank"]["train"]
    ontonotes_train, ontonotes_train_labels = splits["ontonotes"]["train"]

    litbank_feats = tokenize_and_align(litbank_train, litbank_train_labels)

    # Use simple oversampling by duplicating LitBank inputs by a factor of 3
    # To account for the fiction-nonfiction domain imbalance
    litbank_feats_oversampled = concatenate_datasets([litbank_feats] * 3)

    ontonotes_feats = tokenize_and_align(ontonotes_train, ontonotes_train_labels)
    mixed_training_feats = concatenate_datasets([litbank_feats_oversampled, ontonotes_feats]).shuffle(seed=42)

    # Validation set drawn from LitBank only
    val_tokens, val_labels = splits["litbank"]["validation"]
    val_feats = tokenize_and_align(val_tokens, val_labels)

    training_args = TrainingArguments(
        output_dir="./results2",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mixed_training_feats,
        eval_dataset=val_feats,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # trainer.train()

    # First epoch was the best according to eval_loss, so it's loaded and saved for evaluation
    # finetuned_tokenizer = AutoTokenizer.from_pretrained("./results2/checkpoint-5024")
    # finetuned_model = AutoModelForTokenClassification.from_pretrained("./results2/checkpoint-5024")

    # finetuned_tokenizer.save_pretrained("./finetuned_model")
    # finetuned_model.save_pretrained("./finetuned_model")


if __name__ == "__main__":
    main()
