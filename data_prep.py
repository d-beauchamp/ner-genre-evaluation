from sklearn.model_selection import train_test_split
from datasets import load_from_disk, concatenate_datasets

litbank_data = load_from_disk("converted_datasets/litbank_dataset_test")
ontonotes_test_data = load_from_disk("converted_datasets/ontonotes_dataset_test")
ontonotes_train_sets = ["ontonotes_dataset_train00", "ontonotes_dataset_train01"]


def split_train_val_test(dataset):
    """Split labels and tokens into test and train sets."""
    tokens = dataset["tokens"]
    labels = dataset["reduced_labels"]
    X_temp, X_test, y_temp, y_test = train_test_split(tokens, labels,
                                                      test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                      test_size=0.125, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def combine_and_extract_train_sets(dirs):
    loaded_datasets = []
    for d in dirs:
        loaded_file = load_from_disk(f"converted_datasets/{d}")
        loaded_datasets.append(loaded_file)
    combined_set = concatenate_datasets(loaded_datasets)
    train, labels = combined_set["tokens"], list(combined_set["reduced_labels"])
    return train, labels


def chunker(tokens, labels=None, max_len=50):
    """
    Split token list into chunks. Useful for LitBank inputs which are usually well over
    the BERT limit of 512.
    """
    chunks = []
    for i in range(0, len(tokens), max_len):
        token_chunk = tokens[i:i + max_len]

        if labels is not None:
            label_chunk = labels[i:i + max_len]
            chunks.append((token_chunk, label_chunk))
        else:
            chunks.append(token_chunk)

    return chunks


def load_splits():
    litbank_split = split_train_val_test(litbank_data)
    litbank_train, litbank_train_labels = litbank_split[0], litbank_split[3]
    val_tokens, val_labels = litbank_split[1], litbank_split[4]
    litbank_test, litbank_test_labels = litbank_split[2], litbank_split[5]

    ontonotes_train, ontonotes_train_labels = combine_and_extract_train_sets(ontonotes_train_sets)
    ontonotes_test, ontonotes_test_labels = (ontonotes_test_data["tokens"],
                                        ontonotes_test_data["reduced_labels"])

    return {
        "litbank": {
            "train": (litbank_train, litbank_train_labels),
            "validation": (val_tokens, val_labels),
            "test": (litbank_test, litbank_test_labels)
        },
        "ontonotes": {
            "train": (ontonotes_train, ontonotes_train_labels),
            "test": (ontonotes_test, ontonotes_test_labels)
        }
    }