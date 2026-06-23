########################### FILE 1 ###########################
## Simple analysis of each dataset's labels to see which ones are overlapping.

from datasets import load_dataset
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

def get_unique_labels(ds, col_name:str="privacy_mask") -> list[str]:
    # get the unique labels, stored in column col_name
    labels_list = []
    labels_col = ds["train"][col_name]

    for annotations in labels_col:
        for annotation in annotations:
            curr_label = annotation["label"]
            if curr_label not in labels_list:
                labels_list.append(curr_label)
    
    return labels_list

if __name__ == "__main__":

    hf_hub_token = get_hf_hub_token()
    login(hf_hub_token)


    ds300k = load_dataset("ai4privacy/pii-masking-300k")
    ds500k = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy")


    print(ds300k)
    print(ds500k)
    
    labels_list_300k = get_unique_labels(ds300k)
    labels_list_500k = get_unique_labels(ds500k)

    print(labels_list_300k)
    # ['USERNAME', 'TIME', 'DATE', 'LASTNAME1', 
    # 'LASTNAME2', 'EMAIL', 'SOCIALNUMBER', 
    # 'IDCARD', 'COUNTRY', 'BUILDING', 
    # 'STREET', 'CITY', 'STATE', 
    # 'POSTCODE', 'PASS', 'PASSPORT', 
    # 'TEL', 'DRIVERLICENSE', 'BOD', 
    # 'SEX', 'IP', 'SECADDRESS', 
    # 'LASTNAME3', 'GIVENNAME1', 'GIVENNAME2', 
    # 'TITLE', 'GEOCOORD', 'CARDISSUER']
    print(labels_list_500k)
    # ['TIME', 'GIVENNAME', 'SURNAME', 
    # 'DATE', 'CITY', 'STREET', 
    # 'DRIVERLICENSENUM', 'SOCIALNUM', 'TELEPHONENUM', 
    # 'AGE', 'SEX', 'IDCARDNUM', 
    # 'ZIPCODE', 'TAXNUM', 'EMAIL', 
    # 'BUILDINGNUM', 'TITLE', 'PASSPORTNUM', 
    # 'CREDITCARDNUMBER', 'GENDER']





