from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.model_selection import train_test_split
import torch
from datasets import load_from_disk
from seqeval.metrics import classification_report

# TODO: remove MISC from test data to have direct comparison w/ finetuned labels

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

baseline_tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
baseline_model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

finetuned_tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")
finetuned_model = AutoModelForTokenClassification.from_pretrained("./finetuned_model")

baseline_model.eval()
finetuned_model.eval()

litbank_data = load_from_disk("converted_datasets/litbank_dataset_test")
# OntoNotes is test set only - no split needed
ontonotes_data = load_from_disk("converted_datasets/ontonotes_dataset_test")


def chunk_tokens(tokens, max_len=510):
    """
    Split token list into chunks <= max_len (leave room for [CLS] + [SEP])
    """
    chunks = []
    for i in range(0, len(tokens), max_len):
        chunks.append(tokens[i:i + max_len])
    return chunks


def align_predictions(pred_ids, encoding, model):
    """Maps subword predictions back to original token."""
    word_ids = encoding.word_ids()
    previous_word_idx = None
    labels = []
    temp_labels = []

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            if temp_labels:
                labels.append(temp_labels[0])
                temp_labels = []
            previous_word_idx = word_idx
        temp_labels.append(model.config.id2label[pred_ids[0, idx].item()])

    if temp_labels:
        labels.append(temp_labels[0])
    return labels


def split_data(dataset):
    tokens = dataset["tokens"]
    labels = dataset["reduced_labels"]
    X_train, X_test, y_train, y_test = train_test_split(tokens, labels,
                                                        test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def predictions(data, tokenizer, model):
    all_preds = []

    for sentence_tokens in data:
        sentence_preds = []
        chunks = chunk_tokens(sentence_tokens, max_len=510)  # BERT limit 512

        for chunk in chunks:
            encoding = tokenizer(chunk, return_tensors="pt", is_split_into_words=True,
                                 truncation=True, padding=False)
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)
                pred_labels = align_predictions(pred_ids, encoding, model)
                sentence_preds.extend(pred_labels)

        if len(sentence_preds) != len(sentence_tokens):
            # Sometimes align_predictions may drop tokens; pad with 'O' to match length
            diff = len(sentence_tokens) - len(sentence_preds)
            sentence_preds.extend(['O'] * diff)

        all_preds.append(sentence_preds)
    return all_preds


def model_eval(tokenizer, model, test_set, labels):
    """Abstract the evaluation process to allow tests with different models and datasets."""
    preds = predictions(test_set, tokenizer, model)
    print(classification_report(labels, preds))


def main():
    litbank_split = split_data(litbank_data)

    litbank_test, litbank_labels = litbank_split[1], litbank_split[3]
    ontonotes_test, ontonotes_labels = ontonotes_data["tokens"], ontonotes_data["reduced_labels"] # TEST Labels

    print("LitBank Evaluation (Baseline):")
    model_eval(baseline_tokenizer, baseline_model, litbank_test, litbank_labels)

    print("OntoNotes Evaluation (Baseline):")
    model_eval(baseline_tokenizer, baseline_model, ontonotes_test, ontonotes_labels)

    print("LitBank Evaluation (Finetuned):")
    model_eval(finetuned_tokenizer, finetuned_model, litbank_test, litbank_labels)

    print("OntoNotes Evaluation (Finetuned):")
    model_eval(finetuned_tokenizer, finetuned_model, ontonotes_test, ontonotes_labels)


if __name__ == "__main__":
    main()
