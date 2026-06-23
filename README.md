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




# Debugging the annotations

A weird example among many

```There might be an issue. Length of privacy mask: 3. Number of labels added: 2. However, note that this might just be due to the dataset's flaws: it contains some overlapping labels and annotation errors.
Added labels: [('<s>', 'SPEC'), ('Design', 'SPEC'), ('an', 'SPEC'), ('exhibi', 'SPEC'), ('t', 'O'), ('featuring', 'SPEC'), ('the', 'SPEC'), ('art', 'SPEC'), ('work', 'O'), ('of', 'SPEC'), ('Gerard', 'B-GIVENNAME'), ('ine', 'I-GIVENNAME'), ('Sum', 'SPEC'), ('eja', 'I-GIVENNAME'), ('al', 'B-SURNAME'), ('Do', 'SPEC'), ('uri', 'I-SURNAME'), (',', 'O'), ('a', 'SPEC'), ('50-', 'SPEC'), ('year', 'O'), ('-', 'O'), ('old', 'O'), ('artist', 'SPEC'), ('from', 'SPEC'), ('South', 'SPEC'), ('Sur', 'SPEC'), ('rey', 'O'), ('.', 'O'), ('</s>', 'SPEC')]
Privacy mask: [{'value': 'Gerardine Sumeja', 'start': 43, 'end': 59, 'label': 'GIVENNAME'}, {'value': 'al Douri', 'start': 60, 'end': 68, 'label': 'SURNAME'}, {'value': ' South Surrey', 'start': 96, 'end': 109, 'label': 'CITY'}]```

Al-Douri : Do is classified as SPEC
South Surre:y: classified as SPEC for SOUth and Sur !! . 
Only <s> and </s> should be labeled as SPEC. What is happening ??



This seems to be a problem more focused on RoBERTa. Bugs with DeBERTa happen, but they are more seldom and SPEC characters usually don't go further than the second token, e.g.:
```There might be an issue. Length of privacy mask: 5. Number of labels added: 4. However, note that this might just be due to the dataset's flaws: it contains some overlapping labels and annotation errors.
Added labels: [('[CLS]', 'SPEC'), ('C', 'SPEC'), ('7', 'O'), ('4', 'O'), ('.', 'O'), ('Applicant', 'O'), ('ID', 'O'), (':', 'O'), ('266', 'B-SOCIALNUMBER'), ('107', 'I-SOCIALNUMBER'), ('49', 'I-SOCIALNUMBER'), ('45', 'I-SOCIALNUMBER'), ('-', 'O'), ('IP', 'O'), ('Address', 'O'), (':', 'O'), ('d', 'O'), ('8', 'O'), ('cd', 'O'), (':', 'O'), ('1', 'O'), ('aaa', 'O'), (':', 'O'), ('63', 'O'), ('ba', 'O'), (':', 'O'), ('844', 'O'), ('8', 'O'), (':', 'O'), ('7', 'O'), ('fe', 'O'), ('2', 'O'), (':', 'O'), ('6', 'O'), ('ea', 'O'), ('5', 'O'), (':', 'O'), ('21', 'O'), ('a', 'O'), ('1', 'O'), (':', 'O'), ('393', 'O'), ('b', 'O'), ('-', 'O'), ('Password', 'O'), (':', 'O'), ('J', 'O'), ('3', 'O'), ('b', 'O'), ('X', 'O'), ('$', 'O'), ('1', 'O'), ('q', 'O'), ('0', 'O'), ('&', 'O'), (']', 'O'), ('5', 'O'), ('.', 'O'), ('Applicant', 'O'), ('ID', 'O'), (':', 'O'), ('306', 'B-SOCIALNUMBER'), ('272', 'I-SOCIALNUMBER'), ('5', 'I-SOCIALNUMBER'), ('393', 'I-SOCIALNUMBER'), ('-', 'O'), ('IP', 'O'), ('Address', 'O'), (':', 'O'), ('225', 'O'), ('.', 'O'), ('5', 'O'), ('.', 'O'), ('202', 'O'), ('.', 'O'), ('173', 'O'), ('-', 'O'), ('Password', 'O'), (':', 'O'), ('V', 'O'), ('k', 'O'), ('.', 'O'), ('.', 'O'), ('.', 'O'), ('and', 'O'), ('so', 'O'), ('on', 'O'), ('for', 'O'), ('all', 'O'), ('the', 'O'), ('application', 'O'), ('IDs', 'O'), ('listed', 'O'), ('.', 'O'), ('Please', 'O'), ('be', 'O'), ('informed', 'O'), ('that', 'O'), ('a', 'O'), ('scholarship', 'O'), ('award', 'O'), ('ceremony', 'O'), ('will', 'O'), ('be', 'O'), ('held', 'O'), ('on', 'O'), ('March', 'B-DATE'), ('22', 'I-DATE'), ('nd', 'I-DATE'), (',', 'I-DATE'), ('20', 'I-DATE'), ('78', 'I-DATE'), ('at', 'I-DATE'), ('21', 'I-DATE'), ('o', 'I-DATE'), ("'", 'I-DATE'), ('clock', 'I-DATE'), ('at', 'I-DATE'), ('the', 'I-DATE'), ('following', 'I-DATE'), ('location', 'O'), (':', 'O'), ('Venue', 'O'), (':', 'O'), ('Rue', 'B-STREET'), ('des', 'I-STREET'), ('École', 'I-STREET'), ('s', 'I-STREET'), (',', 'I-STREET'), ('POST', 'I-STREET'), ('CODE', 'I-STREET'), ('_', 'I-STREET'), ('BG', 'I-STREET'), ('(', 'I-STREET'), ('TA', 'I-STREET'), ('9', 'I-STREET'), (',', 'O'), ('Great', 'O'), ('Britain', 'O'), ('[SEP]', 'SPEC')]
Privacy mask: [{'value': '266 107 4945', 'start': 21, 'end': 33, 'label': 'SOCIALNUMBER'}, {'value': '306 272 5393', 'start': 135, 'end': 147, 'label': 'SOCIALNUMBER'}, {'value': "March 22nd, 2078 at TIME_BG(21 o'clock", 'start': 317, 'end': 355, 'label': 'DATE'}, {'value': "21 o'clock at the following location:\n\nVenue: STREET_BG(Rue des Écoles", 'start': 337, 'end': 407, 'label': 'TIME'}, {'value': 'Rue des Écoles, POSTCODE_BG(TA9', 'start': 383, 'end': 414, 'label': 'STREET'}]```

Things like these explain most of the issues with DeBERTa:
```{'value': 'IDCARD_B(728700586', 'start': 67, 'end': 85, 'label': 'POSTCODE'}, {'value': '728700586', 'start': 67, 'end': 76, 'label': 'IDCARD'}```. Two labels for one thing ! the POSTCODE is weird.



After correcting the is_subowrd method, only 3 annotation errors happen on the original dataset.

```
Added labels: [('<s>', 'SPEC'), ('Yo', 'O'), (',', 'SPEC'), ('El', 'B-GIVENNAME'), ('ica', 'SPEC'), ('Qu', 'I-GIVENNAME'), ("'", 'SPEC'), ('s', 'SPEC'), ('birthday', 'O'), ('is', 'O'), ('22', 'B-DATE'), ('/04/', 'SPEC'), ('2022', 'SPEC'), (',', 'SPEC'), ('we', 'O'), ("'", 'SPEC'), ('re', 'SPEC'), ('planning', 'O'), ('a', 'O'), ('surprise', 'O'), ('at', 'O'), ('Saint', 'O'), ('-', 'SPEC'), ('Ni', 'SPEC'), ('cola', 'SPEC'), ('s', 'SPEC'), ("'", 'SPEC'), ('s', 'SPEC'), ('fin', 'O'), ('est', 'SPEC'), ('restaurant', 'O'), ('</s>', 'SPEC')]
Privacy mask: [{'value': 'Elica Qu', 'start': 4, 'end': 12, 'label': 'GIVENNAME'}, {'value': '22/04/2022', 'start': 27, 'end': 37, 'label': 'DATE'}, {'value': ' Saint-Nicolas', 'start': 68, 'end': 82, 'label': 'CITY'}]```
=> this one is due to the blank space before "Saint-Nicolas" in the original privacy mask ;) The same goes with the 2 other problematic examples ! Mystery solved on this.

WIth this new subword method, DeBERTa has XXX problematic cases in alignement.


With the concatrenated datasets, we have a total of 15 problematic entries. That sounds reasonable.