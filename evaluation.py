from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from seqeval.metrics import classification_report, f1_score

from data_prep import load_splits, chunker

# GPU acceleration on Mac M1 using mps
device = "cpu"
available = torch.backends.mps.is_available()
built = torch.backends.mps.is_built()
if available and built:
    device = "mps"

baseline_tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
baseline_model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

finetuned_tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")
finetuned_model = AutoModelForTokenClassification.from_pretrained("./finetuned_model")

baseline_model.to(device).eval()
finetuned_model.to(device).eval()


def align_predictions(pred_ids, encoding, model):
    """Map subword predictions back to original token."""
    word_ids = encoding.word_ids()
    previous_word_idx = None
    labels = []
    temp_labels = []

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            if temp_labels:
                labels.append(temp_labels[0])  # First subword heuristic
                temp_labels = []
            previous_word_idx = word_idx
        temp_labels.append(model.config.id2label[pred_ids[0, idx].item()])

    if temp_labels:
        labels.append(temp_labels[0])
    return labels


def predictions(data, tokenizer, model):
    """Tokenize and predict labels for chunks of inputs."""
    all_preds = []

    for doc in data:
        doc_preds = []

        # Chunk input documents into chunks of 50 words
        # Especially important for LitBank inputs, whose ~2000 word length far exceeds
        # the BERT limit of 512 even before subword tokenization
        chunks = chunker(doc, max_len=50)

        for chunk in chunks:

            # Inputs are already split into word tokens
            encoding = tokenizer(chunk, return_tensors="pt", is_split_into_words=True,
                                 truncation=True, padding=False).to(device)

            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)
                pred_labels = align_predictions(pred_ids, encoding, model)
                doc_preds.extend(pred_labels)

        all_preds.append(doc_preds)
    return all_preds


def model_eval(tokenizer, model, test_set, labels):
    """Abstract the evaluation process to allow tests with different models and datasets."""
    preds = predictions(test_set, tokenizer, model)
    print(classification_report(labels, preds))


def domain_f1_difference(tokenizer, model, test1, labels1, test2, labels2):
    """Calculate difference in F1 score between domains."""
    preds_litbank = predictions(test1, tokenizer, model)
    preds_ontonotes = predictions(test2, tokenizer, model)

    f1_litbank = f1_score(labels1, preds_litbank, average="weighted")
    f1_ontonotes = f1_score(labels2, preds_ontonotes, average="weighted")
    print(f1_ontonotes - f1_litbank)


def main():
    """Print classification reports and F1 domain differences for both baseline
    and finetuned models, using predefined test splits."""

    splits = load_splits()

    litbank_test, litbank_test_labels = splits["litbank"]["test"]
    ontonotes_test, ontonotes_labels = splits["ontonotes"]["test"]

    print("LitBank Evaluation (Baseline):")
    model_eval(baseline_tokenizer, baseline_model, litbank_test, litbank_test_labels)

    print("OntoNotes Evaluation (Baseline):")
    model_eval(baseline_tokenizer, baseline_model, ontonotes_test, ontonotes_labels)

    print("LitBank Evaluation (Finetuned):")
    model_eval(finetuned_tokenizer, finetuned_model, litbank_test, litbank_test_labels)

    print("OntoNotes Evaluation (Finetuned):")
    model_eval(finetuned_tokenizer, finetuned_model, ontonotes_test, ontonotes_labels)

    print("Domain F1 Difference (Baseline):")
    domain_f1_difference(baseline_tokenizer, baseline_model, litbank_test, litbank_test_labels,
                         ontonotes_test, ontonotes_labels)

    print("Domain F1 Difference (Finetuned):")
    domain_f1_difference(finetuned_tokenizer, finetuned_model, litbank_test, litbank_test_labels,
                         ontonotes_test, ontonotes_labels)


if __name__ == "__main__":
    main()
