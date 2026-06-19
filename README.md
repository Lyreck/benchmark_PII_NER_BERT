# Objective of this repo:

Evaluate NER models on GPU on two dimensions: precision, and speed.

Models to evaluate:
- Ar86Bat/multilang-pii-ner
- yonigo/deberta-v3-base-pii-en

To begin with, evaluation will only be in English. The test set will be the intersection of OpenPII 300k and 500k, since it is the training set of each of the aforementioned models (keeping only labels that intersect).

The first task is to analyse both datasets' labels and only keep the intersection.

The second task is to process the dataset in order to feed it to the model and compare it to the validation labels.

Then, we will be able to run the precision and speed benchmark.

I'll maybe add some small tests to get a sense of some specific issues: 
- 0.85763847 that are scores in code but could be interpreted as phone numbers (as it has been in some of my experiments)

Both the models have a github repo with code. [This one](https://github.com/Ar86Bat/multilang-pii-ner) gives a code to put daa in CoNLL format.

----

Following this benchmark, I will
- take the best model, convert it to ONNX if necessary
- Add it to the website, preferably with a WebWorker that runs in the "choice" and "editor" pages

Following this first real-world test, I could have time to
- finetune my own BERT model using HF's Trainer API. Tune the good pre-processing and hyperparameters. Use multiple datasets to avoid relying only on ai4privacy's sometimes ill-labeled dataset, and non-realistic situations (there's a reason why Open AI corrected the dataset and used its own synthetic data source). **I think this will be very important for precision in our case because PII might be hidden in difficult context.
- finetune LFM2.5-350M according to OpenAI's Privacy Filter model card. For fun and to compare similar-sized models for precision and speed. (PyTorch training loop, causal attention, LoRA...)

# Step 1: benchmark dataset creation

## Comparison of PII 300k and 500k
300k:
 ```
 ['USERNAME', 'TIME', 'DATE', 'LASTNAME1', 
    'LASTNAME2', 'EMAIL', 'SOCIALNUMBER', 
    'IDCARD', 'COUNTRY', 'BUILDING', 
    'STREET', 'CITY', 'STATE', 
    'POSTCODE', 'PASS', 'PASSPORT', 
    'TEL', 'DRIVERLICENSE', 'BOD', 
    'SEX', 'IP', 'SECADDRESS', 
    'LASTNAME3', 'GIVENNAME1', 'GIVENNAME2', 
    'TITLE', 'GEOCOORD', 'CARDISSUER']
```
500k:
```
['TIME', 'GIVENNAME', 'SURNAME', 
    'DATE', 'CITY', 'STREET', 
    'DRIVERLICENSENUM', 'SOCIALNUM', 'TELEPHONENUM', 
    'AGE', 'SEX', 'IDCARDNUM', 
    'ZIPCODE', 'TAXNUM', 'EMAIL', 
    'BUILDINGNUM', 'TITLE', 'PASSPORTNUM', 
    'CREDITCARDNUMBER', 'GENDER']
```
Important remark: if 500k model, then the following labels DO NOT exist: ["I-PASSPORTNUM", "I-AGE", "I-CREDITCARDNUMBER", "I-GENDER", "I-IDCARDNUM", "I-SEX"].

Present in 300k AND in 500k: `TIME, DATE, EMAIL, STREET, CITY, SEX, TITLE`
Present in 300k and NOT in 500k: `USERNAME, LASTNAME1, LASTNAME2, COUNTRY, BUILDING, STATE, PASS, PASSPORT, TEL, DRIVERLICENSE, BOD (??), IP, SECADRESS, LASTNAME3, GIVENNAME1, GEOCOORD, CARDISSUER`
Present in 500k and NOT in 300k: `SOCIALNUM, IDCARDNUM, BUILDINGNUM, ZIPCODE, PASSPORTNUM, TELEPHONENUM, DRIVERLICENSENUM, GIVENNAME. SURNAME, AGE, TAXNUM, CREDITCARDNUMBER, GENDER`

Resulting mapping (300k:500k):
{
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

Final labels for benchmark (using the label mapping to get as much as possible.): `["TIME", "DATE", "EMAIL", "STREET", "CITY", "SEX", "TITLE", "SOCIALNUM", "IDCARDNUM", "BUILDINGNUM", "ZIPCODE", "PASSPORTNUM", "TELEPHONENUM", "DRIVERLICENSENUM", "GIVENNAME", "SURNAME"]`

Labels to ignore when running the model trained on 300k (DeBERTa): `["USERNAME", "COUNTRY", "STATE", "PASS", "BOD", "IP", "SECADRESS", "GEOCOORD", "CARDISSUER"]`

Labels to ignore when running the model trained on 500k (RoBERTa): `["AGE", "TAXNUM", "CREDITCARDNUMBER", "GENDER"]`

