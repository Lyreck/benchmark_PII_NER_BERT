########################### FILE 3 ###########################
## This file intends to take an input dataset with input text and privacy mask (with offsets);
## and convert it to a new dataset that is accepted by scikit-learn's classification report.

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
from transformers.pipelines.pt_utils import KeyDataset
import torch

def load_model_and_tokenizer(model_id):

    device = 0 if torch.cuda.is_available() else -1 #CPU fallback

    if model_id == "yonigo/deberta-v3-base-pii-en":

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)

        pipe = pipeline(
            "ner", 
            model=model_id, 
            aggregation_strategy="none", 
            ignore_labels=["USERNAME", "COUNTRY", "STATE", "PASS", "BOD", "IP", "SECADDRESS", "GEOCOORD", "CARDISSUER"],
            device=device
            )

    elif model_id == "Ar86Bat/multilang-pii-ner":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none", ignore_labels=["AGE", "TAXNUM", "CREDITCARDNUMBER", "GENDER"], device=device)

    else:
        raise AttributeError("Unexpected model_id provided (model not yet supported).")
    
    return tokenizer, model, pipe


# trying the alignment function of yonigo. The approach seems more robust.

# def is_subword(text, tokenized, tokenizer, index):
#     word = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][index])
#     start_ind, end_ind = tokenized["offset_mapping"][index]
#     word_ref = text[start_ind:end_ind]
#     is_subword = len(word) != len(word_ref)
#     return is_subword

def is_subword(text, tokenized, tokenizer, index):
    # A token is a subword if its offset_mapping start doesn't match
    # the end of the previous non-subword token's offset_mapping.
    # Simpler: check if the token string starts with the continuation marker.
    word = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][index])
    # RoBERTa: Ġ = word boundary, no Ġ = subword
    # BERT: no ## = word boundary, ## = subword  
    # SentencePiece: ▁ = word boundary, no ▁ = subword
    return not word.startswith(("Ġ", "▁", "##"))

def tokenize_robust(example, label2id, tokenizer, iob=True, ignore_subwords=True, model_id=""): #adapted from https://github.com/yonigottesman/pii-model/blob/main/train.py
    """Tokenize the dataset and create the columns "true labels" and "predicted labels" to then evaluate the models.

    Args:
        example (_type_): a line in a dataset
        label2id (dict): a dictionary mapping each label to its id, given my config.json
        tokenizer (_type_): tokenizer for the given model that we want to evaluate
        iob (bool, optional): whether we use iob tagging. Defaults to True.
        ignore_subwords (bool, optional): whether we operate at subword level or not. Defaults to True.
        model_id (str, optional): to know which model is being evaluated and enforce specific cases.. Defaults to "".

    Returns:
        _type_: dataset with input text, predicted and true labels (among other things)
    """


    text, labels = example["source_text"], example["privacy_mask"] #runs only on one example: no batching!
    pred_labels = example["predicted_mask"] #run the model on the text
    pred_token_labels = [label2id[label["entity"]] for label in pred_labels]

    i = 0
    true_token_labels = []

    tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    start_token_to_label = {
        tokenized.char_to_token(label["start"]): (label["start"], label["end"], label["label"]) for label in labels #each token has a label which starts and ends at some point. From what I understand, char_to_token maps the index number where we are in the word to a label.
    }
    num_labels_added = 0
    # num_special_labels = 0
    while i < len(tokenized["input_ids"]):
        if tokenized["special_tokens_mask"][i] == 1:
            # num_special_labels +=1 #disclaimer: this count is probably not accurate. It was used for debugging.
            true_token_labels.append(-100)
            i += 1
        elif i not in start_token_to_label: #if token number i is not a labeled entity
            if ignore_subwords and is_subword(text, tokenized, tokenizer, i):
                true_token_labels.append(-100) #this means the subwords are labeled -100. Mostly important for training. but [TODO] beware of mismatch between subwords and second words: it depends on how the model outputs results !!
            else:
                true_token_labels.append(label2id["O"])
            i += 1
        else:
            start, end, label = start_token_to_label[i]
            start_token = tokenized.char_to_token(start)
            assert start_token == i
            j = start_token
            while j < (len(tokenized["input_ids"]) - 1) and tokenized.token_to_chars(j).start < end:
                if j == start_token:
                    if iob:
                        true_token_labels.append(label2id["B-" + label])
                        num_labels_added +=1
                    else:
                        true_token_labels.append(label2id[label])
                elif ignore_subwords and is_subword(text, tokenized, tokenizer, j): #this might be where roberta fails
                    true_token_labels.append(-100)
                else:
                    if iob:
                        if (label in ["PASSPORTNUM", "AGE", "CREDITCARDNUMBER", "GENDER", "IDCARDNUM", "SEX"]) and model_id == "Ar86Bat/multilang-pii-ner": #special case for 500k dataset: RoBERTa model does not have I-labels for these elements in its id2label.
                            true_token_labels.append(label2id["B-" + label])
                        else:
                            true_token_labels.append(label2id["I-" + label])
                    else:
                        true_token_labels.append(label2id[label])

                j += 1
            i = j
        
    # safety checks on the number of labels added to the "real" labels.
    if num_labels_added!=len(labels):
            id2label = {v:k for k,v in label2id.items()}
            id2label[-100] = "SPEC"
            print(f"There might be an issue. Length of privacy mask: {len(labels)}. Number of labels added: {num_labels_added}. However, note that this might just be due to the dataset's flaws: it contains some overlapping labels and annotation errors.")
            print(f"Added labels: {[(tokenizer.decode(input),id2label[l]) for input,l in zip(tokenized["input_ids"],true_token_labels)]}")
            print(f"Privacy mask: {labels}")
            print("=======================================================================")

    tokenized["true_labels"] = true_token_labels
    tokenized["pred_labels"] = [-100] + [label2id[label["entity"]] for label in pred_labels] + [-100] # add padding to match true labels with start and end special characters.

    # assert len(true_token_labels) == len(pred_token_labels), f"Issue: There are {len(true_token_labels)} true labels, and {len(pred_token_labels)} predicted labels. Labels should be attributed token-wise, there should be no discrepancy on the number. Here is the text {text}: and the predictions:{pred_labels}"

    # safety checks to verify that the predicted labels list and the true labels list have the same length, and see whether any discrepancy could be explained by special characters.
    
    try:
        assert len(true_token_labels) != len(pred_token_labels) - 2
    except:
        diff = len(true_token_labels) - len(pred_token_labels)
        print(f"There are {len(true_token_labels)} true labels and {len(pred_token_labels)} predicted labels (diff = {diff})")
    
    # if len(true_token_labels) != len(pred_token_labels):
    #     diff = len(true_token_labels) - len(pred_token_labels)
    #     if diff >= 3:
    #         print(f"There are {len(true_token_labels)} true labels and {len(pred_token_labels)} predicted labels (diff = {diff})")
    #     # assert num_special_labels == len(true_token_labels) - len(pred_token_labels), f"I am counting too many special labels. Wrong counting ? \n There are {num_special_labels} special labels, for {len(true_token_labels)} true labels and {len(pred_token_labels)} predicted labels (diff = {len(true_token_labels) - len(pred_token_labels)})"
    #     pass 
    return tokenized

def replicate_original_benchmark():
    
    (_, benchmark_ds_3OOk_only), (_, benchmark_ds_5OOk_only) = create_benchmark_datasets()

    # Filter empty strings: This prevents the 'ValueError: At least one input is required' error
    benchmark_ds_3OOk = benchmark_ds_3OOk_only.filter(lambda x: x["source_text"] is not None and len(x["source_text"].strip()) > 0)
    benchmark_ds_5OOk = benchmark_ds_5OOk_only.filter(lambda x: x["source_text"] is not None and len(x["source_text"].strip()) > 0)

    # Start with DeBERTa
    model_id_deberta = "yonigo/deberta-v3-base-pii-en"
    tokenizer_deberta, model_deberta, pipeline_deberta = load_model_and_tokenizer(model_id_deberta)

    # Tokenize sentences and attribute each token its label with the map function
    print(f"Launching DeBERTa label alignment. label2id dict: {model_deberta.config.id2label}.")

    # run the model efficiently on GPU
    deberta_predictions = []
    # Adjust batch_size based on the GPU's VRAM (e.g., 16, 32, 64)
    for out in pipeline_deberta(KeyDataset(benchmark_ds_3OOk, "source_text"), batch_size=32):
        deberta_predictions.append(out)
    
        
    benchmark_ds_3OOk = benchmark_ds_3OOk.add_column("predicted_mask", deberta_predictions)

    final_benchmark_ds_300k = benchmark_ds_3OOk.map(
        tokenize_robust,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer_deberta, "label2id": {v:k for k,v in model_deberta.config.id2label.items()}, "model_id":model_id_deberta}, #label2id in DeBERTa is not using the right k,v pairs.
        remove_columns=[
            "source_text",
            "privacy_mask"
        ]
    )

    # Now, RoBERTa
    model_id_roberta = "Ar86Bat/multilang-pii-ner"
    tokenizer_roberta, model_roberta, pipeline_roberta = load_model_and_tokenizer(model_id_roberta)

    print(f"Launching RoBERTa label alignment. label2id dict: {model_roberta.config.label2id}")

    # batched predictions with RoBERTa
    roberta_predictions = []
    for out in pipeline_roberta(KeyDataset(benchmark_ds_5OOk, "source_text"), batch_size=32):
        roberta_predictions.append(out)
        
    benchmark_ds_5OOk = benchmark_ds_5OOk.add_column("predicted_mask", roberta_predictions)

    # Tokenize sentences and attribute each token its label with the map function
    final_benchmark_ds_5OOk = benchmark_ds_5OOk.map(
        tokenize_robust,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer_roberta, "label2id": model_roberta.config.label2id, "model_id":model_id_roberta},
        remove_columns=[
            "source_text",
            "privacy_mask"
        ]
    )

    print(final_benchmark_ds_300k)
    print(final_benchmark_ds_5OOk)
    
    final_benchmark_ds_300k.to_parquet('benchmark_ds_300k_only.parquet')
    final_benchmark_ds_5OOk.to_parquet('benchmark_ds_500k_only.parquet')

    return [(final_benchmark_ds_300k, tokenizer_deberta,model_deberta), (final_benchmark_ds_5OOk, tokenizer_roberta, model_roberta)]



def format_benchmark_datasets():
    """Start from the benchmark datasets with good languages and labels, and format them such that they can be used with HuggingFace's Evaluator.

    Returns:
        tuple: 300k and 500K formatted benchmark datasets.
    """

    (benchmark_ds_3OOk, _), (benchmark_ds_5OOk, _) = create_benchmark_datasets()

    # Filter empty strings: This prevents the 'ValueError: At least one input is required' error
    benchmark_ds_3OOk = benchmark_ds_3OOk.filter(lambda x: x["source_text"] is not None and len(x["source_text"].strip()) > 0)
    benchmark_ds_5OOk = benchmark_ds_5OOk.filter(lambda x: x["source_text"] is not None and len(x["source_text"].strip()) > 0)

    # Start with DeBERTa
    model_id_deberta = "yonigo/deberta-v3-base-pii-en"
    tokenizer_deberta, model_deberta, pipeline_deberta = load_model_and_tokenizer(model_id_deberta)

    # Tokenize sentences and attribute each token its label with the map function
    print(f"Launching DeBERTa label alignment. label2id dict: {model_deberta.config.id2label}.")

    # run the model efficiently on GPU
    deberta_predictions = []
    # Adjust batch_size based on the GPU's VRAM (e.g., 16, 32, 64)
    for out in pipeline_deberta(KeyDataset(benchmark_ds_3OOk, "source_text"), batch_size=32):
        deberta_predictions.append(out)
    
        
    benchmark_ds_3OOk = benchmark_ds_3OOk.add_column("predicted_mask", deberta_predictions)

    final_benchmark_ds_300k = benchmark_ds_3OOk.map(
        tokenize_robust,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer_deberta, "label2id": {v:k for k,v in model_deberta.config.id2label.items()}, "model_id":model_id_deberta}, #label2id in DeBERTa is not using the right k,v pairs.
        remove_columns=[
            "source_text",
            "privacy_mask"
        ]
    )

    # Now, RoBERTa
    model_id_roberta = "Ar86Bat/multilang-pii-ner"
    tokenizer_roberta, model_roberta, pipeline_roberta = load_model_and_tokenizer(model_id_roberta)

    print(f"Launching RoBERTa label alignment. label2id dict: {model_roberta.config.label2id}")

    # batched predictions with RoBERTa
    roberta_predictions = []
    for out in pipeline_roberta(KeyDataset(benchmark_ds_5OOk, "source_text"), batch_size=32):
        roberta_predictions.append(out)
        
    benchmark_ds_5OOk = benchmark_ds_5OOk.add_column("predicted_mask", roberta_predictions)

    # Tokenize sentences and attribute each token its label with the map function
    final_benchmark_ds_5OOk = benchmark_ds_5OOk.map(
        tokenize_robust,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer_roberta, "label2id": model_roberta.config.label2id, "model_id":model_id_roberta},
        remove_columns=[
            "source_text",
            "privacy_mask"
        ]
    )

    print(final_benchmark_ds_300k)
    print(final_benchmark_ds_5OOk)

    return [(final_benchmark_ds_300k, tokenizer_deberta,model_deberta), (final_benchmark_ds_5OOk, tokenizer_roberta, model_roberta)]

if __name__ == "__main__":

    replicate_original_benchmark() #create the original datasets only filtered to english and restrictive classes.

    out = format_benchmark_datasets()

    benchmark_ds_3OOk, benchmark_ds_5OOk = out[0][0], out[1][0]

    print(benchmark_ds_3OOk["input_ids"])
    print("===========================")
    print(benchmark_ds_3OOk["true_labels"])

    ## Export datasets. Next step: evaluate using HuggingFace's Evaluator !
    benchmark_ds_3OOk.to_parquet('benchmark_ds_300k.parquet')
    benchmark_ds_5OOk.to_parquet('benchmark_ds_500k.parquet')


