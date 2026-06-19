## This file intends to take an input dataset with input text and privacy mask (with offsets);
## and convert it to a new dataset following a CoLNN format compatible with HuggingFace's 
## Evaluator class (https://huggingface.co/docs/evaluate/v0.4.6/en/package_reference/evaluator_classes#evaluate.TokenClassificationEvaluator)

from datasets import Features, Sequence, Value, ClassLabel

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

from .select_labels_and_language import create_benchmark_datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def load_model_and_tokenizer(model_id):

    if model_id == "yonigo/deberta-v3-base-pii-en":

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)

        pipe = pipeline(
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
    
    return tokenizer, model, pipe


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
        #print(f"Privacy mask: {privacy_mask}")
        # Sanity check: verify that the len of privacy mask is equal to the number of labels being added.
        # print(f"Length of Privacy mask = {len(privacy_mask)}")
        num_of_labels_added=0
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
                    #print(f"Adding label {next_non_O_label["label"]} to word/token {w}")
                    num_of_labels_added +=1
                    labels.append(label2id[f"B-{next_non_O_label["label"]}"]) #[TODO] potential issue xhen evaluating; whatabout I- tags ?
                    
                    # If we reached or passed the end of the current entity, move to the next one
                    if end >= next_non_O_label['end']:
                        label_idx += 1
                else:
                    labels.append(label2id["O"])
        else:
            # If there are no labeled entities, everything is labeled as "O"
            labels = [label2id["O"] for _ in pre_tokenized_words]

        #print(f"Number of labels added = {num_of_labels_added}")
        if num_of_labels_added!=len(privacy_mask):
            print(f"There might be an issue. Length of privacy mask: {len(privacy_mask)}. Number of labels added: {num_of_labels_added}")
            print(f"Pre-tokenized words and spans: {pre_tokenized}")
            print(f"Privacy mask: {privacy_mask}")
            print("=======================================================================")
        # Append this example's results to our batch lists
        batch_words.append(pre_tokenized_words)
        batch_labels.append(labels)

    # dataset.map expects a dictionary of lists when batched=True
    return {
        "pre_tokenized_words": batch_words,
        "labels": batch_labels
    }

# trying the alignment function of yonigo. The approach sounds more robust.

def is_subword(text, tokenized, tokenizer, index):
    word = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][index])
    start_ind, end_ind = tokenized["offset_mapping"][index]
    word_ref = text[start_ind:end_ind]
    is_subword = len(word) != len(word_ref)
    return is_subword

def tokenize_robust(example, label2id, tokenizer, iob=True, ignore_subwords=True): #adapted from https://github.com/yonigottesman/pii-model/blob/main/train.py

    text, labels = example["source_text"], example["privacy_mask"] #runs only on one example: no batching!
    print(text)
    print(labels)

    i = 0
    token_labels = []

    tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    start_token_to_label = {
        tokenized.char_to_token(label["start"]): (label["start"], label["end"], label["label"]) for label in labels
    }
    num_labels_added = 0
    while i < len(tokenized["input_ids"]):
        if tokenized["special_tokens_mask"][i] == 1:
            token_labels.append(-100)
            i += 1
        elif i not in start_token_to_label:
            if ignore_subwords and is_subword(text, tokenized, tokenizer, i):
                token_labels.append(-100)
            else:
                token_labels.append(label2id["O"])
            i += 1
        else:
            start, end, label = start_token_to_label[i]
            start_token = tokenized.char_to_token(start)
            assert start_token == i
            j = start_token
            while j < (len(tokenized["input_ids"]) - 1) and tokenized.token_to_chars(j).start < end:
                if j == start_token:
                    if iob:
                        token_labels.append(label2id["B-" + label])
                        num_labels_added +=1
                    else:
                        token_labels.append(label2id[label])
                elif ignore_subwords and is_subword(text, tokenized, tokenizer, j):
                    token_labels.append(-100)
                else:
                    if iob:
                        token_labels.append(label2id["I-" + label])
                        num_labels_added +=1
                    else:
                        token_labels.append(label2id[label])

                j += 1
            i = j
        
    if num_labels_added!=len(labels):
            print(f"There might be an issue. Length of privacy mask: {len(labels)}. Number of labels added: {num_labels_added}")
            print(f"Tokenized words and spans: {tokenized}")
            print(f"Privacy mask: {labels}")
            print("=======================================================================")

    tokenized["labels"] = token_labels
    return tokenized

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
    print(f"Launching DeBERTa label alignment. label2id dict: {model_deberta.config.id2label}.")
    final_benchmark_ds_300k = benchmark_ds_3OOk.map(
        tokenize_robust,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer_deberta, "label2id": {v:k for k,v in model_deberta.config.id2label.items()}} #label2id in DeBERTa is not using the right k,v pairs.
    )

    print(final_benchmark_ds_300k)

    # Change columns and features of the dataset to match the target format.
    # label_names_300k = list(model_deberta.config.id2label.values()) #label names in order of the ids.
    # features_300k = Features({
    #     "tokens": Sequence(feature=Value(dtype="string")),
    #     "ner_tags": Sequence(feature=ClassLabel(names=label_names_300k)),
    # })

    # final_benchmark_ds_300k = (
    #     benchmark_ds_300k
    #     .rename_column("pre_tokenized_words", "tokens")
    #     .rename_column("labels", "ner_tags")
    #     .select_columns(["tokens", "ner_tags"])  # Drops source_text, privacy_mask, etc.
    #     .cast(features_300k)                      # Enforces the ClassLabel string map
    # )

    # # Now, RoBERTa
    # model_id_roberta = "Ar86Bat/multilang-pii-ner"
    # tokenizer_roberta, model_roberta, pipeline_roberta = load_model_and_tokenizer(model_id_roberta)

    # # Tokenize sentences and attribute each token its label with the map function
    # benchmark_ds_5OOk = benchmark_ds_5OOk.map(
    #     tokenize_and_align_labels_batched,
    #     batched=True,
    #     fn_kwargs={"tokenizer": tokenizer_roberta, "label2id": model_roberta.config.label2id}
    # )

    # # Change columns and features of the dataset to match the target format.
    # label_names_500k = list(model_roberta.config.id2label.values()) #label names in order of the ids.
    # features_500k = Features({
    #     "tokens": Sequence(feature=Value(dtype="string")),
    #     "ner_tags": Sequence(feature=ClassLabel(names=label_names_500k)),
    # })

    # final_benchmark_ds_5OOk = (
    #     benchmark_ds_5OOk
    #     .rename_column("pre_tokenized_words", "tokens")
    #     .rename_column("labels", "ner_tags")
    #     .select_columns(["tokens", "ner_tags"])
    #     .cast(features_500k)
    # )

    # return [(final_benchmark_ds_300k, tokenizer_deberta,model_deberta), (final_benchmark_ds_5OOk, tokenizer_roberta, model_roberta)]
    return [(final_benchmark_ds_300k, tokenizer_deberta,model_deberta), (-1, -1, -1)]


if __name__ == "__main__":

    out = format_benchmark_datasets()

    benchmark_ds_3OOk, benchmark_ds_5OOk = out[0][0], out[1][0]

    ## Export datasets. Next step: evaluate using HuggingFace's Evaluator !
    benchmark_ds_3OOk.to_csv('benchmark_ds_300k.csv')
    benchmark_ds_5OOk.to_csv('benchmark_ds_500k.csv')


