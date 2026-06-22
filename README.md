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
- Add it to the website, preferably with a WebWorker that runs in the "choice" and "editor" pages.

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


# Results

## Expected results
Based on the benchmarks provided by the people who trained each model.

### DeBERTa

| Training Loss | Epoch   | Step  | Validation Loss | Bod F1 | Building F1 | Cardissuer F1 | City F1 | Country F1 | Date F1 | Driverlicense F1 | Email F1 | Geocoord F1 | Givenname1 F1 | Givenname2 F1 | Idcard F1 | Ip F1  | Lastname1 F1 | Lastname2 F1 | Lastname3 F1 | Pass F1 | Passport F1 | Postcode F1 | Secaddress F1 | Sex F1 | Socialnumber F1 | State F1 | Street F1 | Tel F1 | Time F1 | Title F1 | Username F1 | Precision | Recall | F1     | Accuracy |
|:-------------:|:-------:|:-----:|:---------------:|:------:|:-----------:|:-------------:|:-------:|:----------:|:-------:|:----------------:|:--------:|:-----------:|:-------------:|:-------------:|:---------:|:------:|:------------:|:------------:|:------------:|:-------:|:-----------:|:-----------:|:-------------:|:------:|:---------------:|:--------:|:---------:|:------:|:-------:|:--------:|:-----------:|:---------:|:------:|:------:|:--------:|
| 0.0007        | 32.0856 | 30000 | 0.0767          | 0.9705 | 0.9869      | 1.0           | 0.9781  | 0.9773     | 0.9374  | 0.9645           | 0.9850   | 0.9769      | 0.8810        | 0.7996        | 0.9443    | 0.9873 | 0.8433       | 0.7641       | 0.7696       | 0.9603  | 0.9619      | 0.9820      | 0.9791        | 0.9782 | 0.9615          | 0.9878   | 0.9815    | 0.9767 | 0.9762  | 0.9668   | 0.9606      | 0.9504    | 0.9625 | 0.9564 | 0.9904   |

### RoBERTa

- Overall accuracy: 99.24%
- Macro F1-score: 0.954
- Weighted F1-score: 0.992

Entity-level scores:
- High F1-scores (>0.97) for common entities: AGE, BUILDINGNUM, CITY, DATE, EMAIL, GIVENNAME, STREET, TELEPHONENUM, TIME
- Excellent performance on EMAIL and DATE (F1 ≈ 0.999)
- Lower F1-scores for challenging/rare entities: DRIVERLICENSENUM (F1 ≈ 0.85), GENDER (F1 ≈ 0.83), PASSPORTNUM (F1 ≈ 0.88), SURNAME (F1 ≈ 0.85), SEX (F1 ≈ 0.84)

## Our results

Here is the output of the classification report onthe concatenated datasets.


# Conversion to ONNX

## DeBERTa
```
`torch_dtype` is deprecated! Use `dtype` instead!
Weight deduplication check in the ONNX export requires accelerate. Please install accelerate to run it.
		-[x] values not close enough, max diff: 2.765655517578125e-05 (atol: 1e-05)
The ONNX export succeeded with the warning: The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance 1e-05:
- logits: max diff = 2.765655517578125e-05.
 The exported model was saved at: /tmp/tmpr1zbo54x


Quantization results:
model_fp16: ok
model_int8: ok
model_uint8: ok
model_quantized: ok (copy of model_int8.onnx)
model_q4: ok
model_q4f16: ok
model_bnb4: ok
```