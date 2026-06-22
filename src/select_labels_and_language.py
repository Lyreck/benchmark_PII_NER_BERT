########################### FILE 2 ###########################
## This code aims to filter both the datasets to english, 
## and keep only the labels identified thanks to analyze_datasets_labels.py.

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
from pathlib import Path
import json

def get_hf_hub_token():
    # Dynamically locate the script's directory
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent  # This points to 'src/'

    #get hf token and log in to access the dataset
    with open(PROJECT_ROOT / "secrets.json", "r") as f:
        jsonf = json.load(f)
        hf_hub_token = jsonf["hf_hub_token"]

    return hf_hub_token

def keep_only_labels(example, labels_to_keep:list[str]):
    # returns a dict with only the labels to keep mentioned in the input list. Intended to be used with .map().
    # We also remove the "label_index" from the 500k dataset to enable future concatenations.
    # Requires batched to be set to "False", because expects to get examples one-by-one.
    updated_annotations = []
    for annotation in example["privacy_mask"]:
        if annotation["label"] in labels_to_keep:
            annotation = {k:v for k,v in annotation.items() if k!="label_index"} # remove "label_index" from the 500k dataset to enable concatenation afterwards.
            updated_annotations.append(annotation)

    example["privacy_mask"] = updated_annotations
    return example #ici, les annotations sont correctes pour 500k (pas de label_index)

def get_unique_labels(ds, col_name:str="privacy_mask") -> list[str]:
    # get the unique labels, stored in column col_name
    labels_list = []
    labels_col = ds[col_name]

    for annotations in labels_col:
        for annotation in annotations:
            curr_label = annotation["label"]
            if curr_label not in labels_list:
                labels_list.append(curr_label)
    
    return labels_list


def create_filtered_datasets():
    # Filter the language of the datasets to english, and keep only the relevant columns.

    try:
        hf_hub_token = get_hf_hub_token()
        login(hf_hub_token)
    except:
        pass # loading without hf hub token. That's ok, just slow.

    ds300k = load_dataset("ai4privacy/pii-masking-300k")
    ds500k = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy")

    ds300k = ds300k["validation"].filter(lambda x: x["language"]=="English")
    ds500k = ds500k["validation"].filter(lambda x: x["language"]=="en")

    labels_to_keep_300k = ["TIME", "DATE", "EMAIL", "STREET", "CITY", "SEX", "TITLE", "SOCIALNUMBER", "IDCARD", "BUILDING", "POSTCODE", "PASSPORT", "TEL", "DRIVERLICENSE", "GIVENNAME1", "LASTNAME1", "LASTNAME2", "LASTNAME3"]#, "BOD"]
    labels_to_keep_500k = ["TIME", "DATE", "EMAIL", "STREET", "CITY", "SEX", "TITLE", "SOCIALNUM", "IDCARDNUM", "BUILDINGNUM", "ZIPCODE", "PASSPORTNUM", "TELEPHONENUM", "DRIVERLICENSENUM", "GIVENNAME", "SURNAME"]

    ds300k = ds300k.map(keep_only_labels, batched=False, fn_kwargs={"labels_to_keep" : labels_to_keep_300k})
    ## Fix of the data schema bug (Gemini)
    # a. Create new features for ds500k by copying its current features, 
    # but replacing the privacy_mask schema with ds300k's schema (which lacks 'label_index')
    new_features_500k = ds500k.features.copy()
    new_features_500k["privacy_mask"] = ds300k.features["privacy_mask"]
    # b. Explicitly pass the new features to the map function
    ds500k = ds500k.map(
        keep_only_labels, 
        batched=False, 
        fn_kwargs={"labels_to_keep" : labels_to_keep_500k},
        features=new_features_500k # <--- This was the missing piece that put all label_index to "None".
    )

    ## Sanity check with the unique labels function that I defined in analyze_datasets_labels.py
    labels_list_300k = get_unique_labels(ds300k)
    labels_list_500k = get_unique_labels(ds500k)

    assert set(labels_list_300k) == set(["TIME", "DATE", "EMAIL", "STREET", "CITY", "SEX", "TITLE", "SOCIALNUMBER", "IDCARD", "BUILDING", "POSTCODE", "PASSPORT", "TEL", "DRIVERLICENSE", "GIVENNAME1", "LASTNAME1", "LASTNAME2", "LASTNAME3"]), f"Unexpected labels for 300k dataset: {labels_list_300k}"    
    assert set(labels_to_keep_500k) == set(["TIME", "DATE", "EMAIL", "STREET", "CITY", "SEX", "TITLE", "SOCIALNUM", "IDCARDNUM", "BUILDINGNUM", "ZIPCODE", "PASSPORTNUM", "TELEPHONENUM", "DRIVERLICENSENUM", "GIVENNAME", "SURNAME"]), f"Unexpected labels for 500k datadet: {labels_list_500k}"    

    return ds300k, ds500k

def standardize_labels(example, labels_mapping:dict):
    # standardize labels according to a label mapping.
    updated_annotations = []
    for annotation in example["privacy_mask"]:
        label = annotation["label"]
        if label in labels_mapping.keys():
            annotation["label"] = labels_mapping[label] # change to the label in the mapping

    return example


def create_dataset_with_uniform_labels(ds1, ds2, labels_mapping:dict):
    # create fusion of 500k and 300k validation datasets, but with labels determined by the labels_mapping.
    # ds1 is the one whose labels do not change.

    # map ds500k to the labels of ds300k
    ds2 = ds2.map(standardize_labels, batched=False, fn_kwargs={"labels_mapping" : labels_mapping})

    # keep only the same columns
    # 'source_text', 'privacy_mask'

    ds2 = ds2.select_columns(["source_text", "privacy_mask"])
    ds1 = ds1.select_columns(["source_text", "privacy_mask"])


    # concatenate the datasets and return
    benchmark_ds = concatenate_datasets([ds1,ds2])

    ## Sanity check: verify that there is no label specific to ds2 that stays in the concatenated dataset.
    unique_labels = set(get_unique_labels(benchmark_ds))
    labels_unique_to_ds2 = set(labels_mapping.keys())
    assert len(unique_labels.intersection(labels_unique_to_ds2)) == 0, f"There are some labels from ds2 ({labels_unique_to_ds2}) that are remaining in the concatenated dataset ({unique_labels})."

    return benchmark_ds, ds1 #return also ds1 to get the original dataset but only with the good labels, to compare with the original benchmarks.

def create_benchmark_datasets():
    ds300k, ds500k = create_filtered_datasets()

    labels_mapping_300kTo500k = {
        "SOCIALNUMBER": "SOCIALNUM",
        "IDCARD": "IDCARDNUM",
        "BUILDING" : "BUILDINGNUM", 
        "POSTCODE" : "ZIPCODE",
        "PASSPORT" : "PASSPORTNUM",
        "TEL" : "TELEPHONENUM",
        "DRIVERLICENSE" : "DRIVERLICENSENUM",
        "GIVENNAME1" : "GIVENNAME",
        "LASTNAME1": "SURNAME",
        "LASTNAME2": "SURNAME",
        "LASTNAME3": "SURNAME"
        #"BOD":"DATE"
    }

    labels_mapping_500kTo300k = {
        "SOCIALNUM": "SOCIALNUMBER",
        "IDCARDNUM": "IDCARD",
        "BUILDINGNUM": "BUILDING",
        "ZIPCODE": "POSTCODE",
        "PASSPORTNUM": "PASSPORT",
        "TELEPHONENUM": "TEL",
        "DRIVERLICENSENUM": "DRIVERLICENSE",
        "GIVENNAME": "GIVENNAME1",
        "SURNAME": "LASTNAME1"  # map LASTNAME1 to SURNAME.
    }

    benchmark_ds_3OOk = create_dataset_with_uniform_labels(ds300k, ds500k, labels_mapping_500kTo300k)
    benchmark_ds_5OOk = create_dataset_with_uniform_labels(ds500k, ds300k, labels_mapping_300kTo500k)

    return benchmark_ds_3OOk, benchmark_ds_5OOk

if __name__ == "__main__":

    (benchmark_ds_3OOk, benchmark_ds_3OOk_only), (benchmark_ds_5OOk, benchmark_ds_5OOk_only) = create_benchmark_datasets()

    benchmark_ds_3OOk.to_csv("benchmark_ds_3OOk.csv")
    benchmark_ds_5OOk.to_csv("benchmark_ds_5OOk.csv")


