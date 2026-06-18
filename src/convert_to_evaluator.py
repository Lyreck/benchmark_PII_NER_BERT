## This file intends to take an input dataset with input text and privacy mask (with offsets);
## and convert it to a new dataset following a CoLNN format compatible with HuggingFace's 
## Evaluator class (https://huggingface.co/docs/evaluate/v0.4.6/en/package_reference/evaluator_classes#evaluate.TokenClassificationEvaluator)

from datasets import Dataset, Features, Sequence, Value, ClassLabel

# For example, the following dataset format is accepted by the evaluator (and is thus the target format):

# dataset = Dataset.from_dict(
#     mapping={
#         "tokens": [["New", "York", "is", "a", "city", "and", "Felix", "a", "person", "."]],
#         "ner_tags": [[1, 2, 0, 0, 0, 0, 3, 0, 0, 0]],
#     },
#     features=Features({
#         "tokens": Sequence(feature=Value(dtype="string")),
#         "ner_tags": Sequence(feature=ClassLabel(names=["O", "B-LOC", "I-LOC", "B-PER", "I-PER"])),
#         }),
# )

from select_labels_and_language import create_benchmark_datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def load_model_and_tokenizer(model_id):

    if model_id == "yonigo/deberta-v3-base-pii-en":

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)

        pipeline = pipeline(
            "token-classification", 
            model=model_id, 
            aggregation_strategy="first", 
            ignore_labels=["USERNAME", "COUNTRY", "STATE", "PASS", "BOD", "IP", "SECADRESS", "GEOCOORD", "CARDISSUER"]
            )

    elif model_id == "Ar86Bat/multilang-pii-ner":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first", ignore_labels=["AGE", "TAXNUM", "CREDITCARDNUMBER", "GENDER"])

    else:
        raise AttributeError("Unexpected model_id provided (model not yet supported).")
    
    return tokenizer, model, pipeline


def tokenize_and_align_labels_batched(examples, tokenizer, label2id):
    """
    Batched version of tokenization and alignment for dataset.map(batched=True). 
    Adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    """
    # Initialize lists to store the batched outputs
    batch_words = []
    batch_labels = []

    # Access the underlying Hugging Face pre-tokenizer
    pre_tokenizer = tokenizer._tokenizer.pre_tokenizer

    # Iterate through each example in the batch
    for i in range(len(examples['source_text'])):
        source_text = examples['source_text'][i]
        privacy_mask = examples['privacy_mask'][i]

        # 1. Pre-tokenize the current string
        pre_tokenized = pre_tokenizer.pre_tokenize_str(source_text)
        pre_tokenized_words = [w for w, span in pre_tokenized]
        pre_tokenized_spans = [span for w, span in pre_tokenized]

        # 2. Assign labels to each word according to the offsets
        labels = []
        label_idx = 0
        print(f"Privacy mask: {privacy_mask}")

        if len(privacy_mask) != 0:  # if there are labeled entities
            for j, w in enumerate(pre_tokenized_words):
                span = pre_tokenized_spans[j]
                start, end = span[0], span[1]

                # Safeguard against running out of labels in the privacy_mask list
                if label_idx < len(privacy_mask):
                    next_non_O_label = privacy_mask[label_idx]
                else:
                    next_non_O_label = None

                if start == end:  # special character or empty span
                    labels.append(-100)

                elif next_non_O_label and (start >= next_non_O_label['start']) and (end <= next_non_O_label['end']):
                    print(f"Adding label {next_non_O_label["label"]} to word/token {w}")
                    labels.append(label2id[f"B-{next_non_O_label["label"]}"])
                    
                    # If we reached or passed the end of the current entity, move to the next one
                    if end >= next_non_O_label['end']:
                        label_idx += 1
                else:
                    labels.append(label2id["O"])
        else:
            # If there are no labeled entities, everything is labeled as "O"
            labels = [label2id["O"] for _ in pre_tokenized_words]

        # Append this example's results to our batch lists
        batch_words.append(pre_tokenized_words)
        batch_labels.append(labels)

    # dataset.map expects a dictionary of lists when batched=True
    return {
        "pre_tokenized_words": batch_words,
        "labels": batch_labels
    }

def format_benchmark_datasets():
    """Start from the benchmark datasets with good languages and labels, and format them such that they can be used with HuggingFace's Evaluator.

    Returns:
        tuple: 300k and 500K formatted benchmark datasets.
    """

    benchmark_ds_3OOk, benchmark_ds_5OOk = create_benchmark_datasets()

    # Start with DeBERTa
    model_id_deberta = "yonigo/deberta-v3-base-pii-en"
    tokenizer_deberta, model_deberta, pipeline_deberta = load_model_and_tokenizer(model_id_deberta)

    # Tokenize sentences and attribute each token its label with the map function
    benchmark_ds_300k = benchmark_ds_3OOk.map(
        tokenize_and_align_labels_batched,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer_deberta, "label2id": model_deberta.config.label2id}
    )

    # Change columns and features of the dataset to match the target format.
    label_names_300k = list(model_deberta.config.id2label.values()) #label names in order of the ids.

    final_benchmark_ds_300k = Dataset.from_dict(
        mapping={
            "tokens": benchmark_ds_300k["pre_tokenized_words"],
            "ner_tags": benchmark_ds_300k["labels"],
        },
        features=Features({
            "tokens": Sequence(feature=Value(dtype="string")), # Tokens are words (strings)
            "ner_tags": Sequence(feature=ClassLabel(names=label_names_300k)),
        }),
    )

    # Now, RoBERTa
    model_id_roberta = "Ar86Bat/multilang-pii-ner"
    tokenizer_roberta, model_roberta, pipeline_roberta = load_model_and_tokenizer(model_id_roberta)

    # Tokenize sentences and attribute each token its label with the map function
    benchmark_ds_5OOk = benchmark_ds_5OOk.map(
        tokenize_and_align_labels_batched,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer_roberta, "label2id": model_roberta.config.label2id}
    )

    # Change columns and features of the dataset to match the target format.
    label_names_500k = list(model_roberta.config.id2label.values()) #label names in order of the ids.

    final_benchmark_ds_5OOk =  Dataset.from_dict(
        mapping={
            "tokens": benchmark_ds_5OOk["pre_tokenized_words"],
            "ner_tags": benchmark_ds_5OOk["labels"],
        },
        features=Features({
            "tokens": Sequence(feature=Value(dtype="string")), # Tokens are words (strings)
            "ner_tags": Sequence(feature=ClassLabel(names=label_names_500k)),
        }),
    )

    return [(final_benchmark_ds_300k, tokenizer_deberta,model_deberta), (final_benchmark_ds_5OOk, tokenizer_roberta, model_roberta)]


if __name__ == "__main__":

    benchmark_ds_3OOk, benchmark_ds_5OOk = format_benchmark_datasets()

    ## Export datasets. Next step: evaluate using HuggingFace's Evaluator !
    benchmark_ds_3OOk.to_csv('benchmark_ds_300k.csv')
    benchmark_ds_5OOk.to_csv('benchmark_ds_500k.csv')


