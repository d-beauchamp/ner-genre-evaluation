from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.model_selection import train_test_split
import torch
from datasets import load_from_disk
from seqeval.metrics import classification_report

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

model.eval()

litbank_data = load_from_disk("converted_datasets/litbank_dataset")
# OntoNotes is test set only - no split needed
ontonotes_data = load_from_disk("converted_datasets/ontonotes_dataset")


def chunk_tokens(tokens, max_len=510):
    """
    Split token list into chunks <= max_len (leave room for [CLS] + [SEP])
    """
    chunks = []
    for i in range(0, len(tokens), max_len):
        chunks.append(tokens[i:i + max_len])
    return chunks

def align_predictions(pred_ids, encoding, model):
    """
    Maps subword predictions back to original tokens
    """
    word_ids = encoding.word_ids()  # Maps each WordPiece token -> original word index
    previous_word_idx = None
    labels = []
    temp_labels = []

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue  # Skip special tokens ([CLS], [SEP])
        if word_idx != previous_word_idx:
            if temp_labels:
                # Take first subword label as the word label
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

def prep_bert_data(data):
    features = []
    for sentence_tokens in data:
        chunks = chunk_tokens(sentence_tokens, max_len=510)  # BERT limit 512

        for chunk in chunks:
            encoding = tokenizer(text=chunk,return_tensors="pt",is_split_into_words=True,
                                 truncation=True, max_length=512, padding=False)
            features.append(encoding)
    return features

def model_eval(data):
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

def main():
    # print((litbank_data[0]))
    litbank_split = split_data(litbank_data)

    litbank_test, litbank_labels = litbank_split[1], litbank_split[3]
    ontonotes_test, ontonotes_labels = ontonotes_data["tokens"], ontonotes_data["reduced_labels"] # TEST Labels

    litbank_preds = model_eval(litbank_test)
    ontonotes_preds = model_eval(ontonotes_test)

    print("Litbank Evaluation:")
    print(classification_report(litbank_labels, litbank_preds))

    print("OntoNotes Evaluation:")
    print(classification_report(ontonotes_labels, ontonotes_preds))

if __name__ == "__main__":
    main()
    # print(ontonotes_data)