########################### FILE 4 ###########################
import pandas as pd
import ast
import time
from sklearn.metrics import classification_report
from transformers import pipeline



def compute_metrics(eval_pred, label_list, seqeval_metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    results_flat = {f"{k}_f1": v["f1"] for k, v in results.items() if isinstance(v, dict)}
    results_flat.update(
        {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    )
    return results_flat



def load_and_prep_data(csv_path):
    df = pd.read_csv(csv_path)
    # Convert string representation of lists back to actual Python lists
    df['tokens'] = df['input_ids'].apply(ast.literal_eval)
    df['ner_tags'] = df['labels'].apply(ast.literal_eval)
    return df

def evaluate_model_speed_and_accuracy(df, model_id, pipeline_task="token-classification"):
    print(f"\n=== Evaluating Model: {model_id} ===")
    
    # Load pipeline directly
    pipe = pipeline(pipeline_task, model=model_id, aggregation_strategy="none")
    
    # Extract labels dictionary mapping from the model
    id2label = pipe.model.config.id2label
    
    y_true = []
    y_pred = []
    
    start_time = time.perf_counter()
    
    # Process text entry by entry
    for _, row in df.iterrows():
        tokens = row['tokens']
        tags = row['ner_tags']
        
        # Reconstruct full sentence string for the pipeline
        text = " ".join(tokens)
        
        # Predict
        predictions = pipe(text)
        
        # Build predicted labels map to align against our words
        # Note: If your baseline has mismatched token lengths, we fallback safely
        aligned_preds = []
        for j, word in enumerate(tokens):
            # Fallback to outside tag if model prediction runs out of alignments
            pred_label = "O" 
            
            # Simple matching logic: find prediction matching current token offsets
            for pred in predictions:
                if pred.get('word', '').strip() in word:
                    pred_label = pred['entity']
                    break
            aligned_preds.append(pred_label)
            
        # Map target tags (integers) to string labels to match predictions
        true_labels = [id2label[t] if t != -100 else "O" for t in tags]
        
        y_true.extend(true_labels)
        y_pred.extend(aligned_preds)
        
    end_time = time.perf_counter()
    
    # 1. LATENCY REPORT
    total_time = end_time - start_time
    avg_time = total_time / len(df)
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Average Latency per Sample: {avg_time * 1000:.2f} ms")
    
    # 2. ACCURACY REPORT
    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

def run_original_benchmark():
    print("=================================")
    print("Launching the original benchmark for comparison with the announced model performances...")

    # Test on 300k 
    # 1. Load the data
    df_test_300k = pd.read_parquet('benchmark_ds_300k_only.parquet')#.head(100)

    id2label_deberta = {
        "1": "I-BUILDING",
        "3": "I-CITY",
        "5": "I-DATE",
        "6": "I-DRIVERLICENSE",
        "7": "I-EMAIL",
        "9": "I-GIVENNAME1",
        "10": "I-GIVENNAME2",
        "11": "I-IDCARD",
        "13": "I-LASTNAME1",
        "14": "I-LASTNAME2",
        "15": "I-LASTNAME3",
        "17": "I-PASSPORT",
        "18": "I-POSTCODE",
        "20": "I-SEX",
        "21": "I-SOCIALNUMBER",
        "23": "I-STREET",
        "24": "I-TEL",
        "25": "I-TIME",
        "26": "I-TITLE",
        "29": "B-BUILDING",
        "31": "B-CITY",
        "33": "B-DATE",
        "34": "B-DRIVERLICENSE",
        "35": "B-EMAIL",
        "37": "B-GIVENNAME1",
        "38": "B-GIVENNAME2",
        "39": "B-IDCARD",
        "41": "B-LASTNAME1",
        "42": "B-LASTNAME2",
        "43": "B-LASTNAME3",
        "45": "B-PASSPORT",
        "46": "B-POSTCODE",
        "48": "B-SEX",
        "49": "B-SOCIALNUMBER",
        "51": "B-STREET",
        "52": "B-TEL",
        "53": "B-TIME",
        "54": "B-TITLE",
        "56": "O"
    }

    # 2. Flatten the lists AND filter out the -100 tokens simultaneously
    true_labels_flat = []
    pred_labels_flat = []

    for true_seq, pred_seq in zip(df_test_300k["true_labels"], df_test_300k["pred_labels"]):
        # Iterate through pairs of (true, predicted) tokens in each sentence
        for true_token, pred_token in zip(true_seq, pred_seq):
            if true_token != -100:  # Ignore special tokens and subwords
                true_labels_flat.append(true_token)
                pred_labels_flat.append(pred_token)

    # 4. Generate the classification report
    print("---------------------------------------------------------------")
    print("DeBERTa classification report, original training dataset only:")
    clfreport = classification_report(true_labels_flat, pred_labels_flat, labels=[k for k in id2label_deberta.keys()], target_names=[v for v in id2label_deberta.values()])
    print(clfreport)



    ### and for RoBERTa !

    # 1. Load the data
    df_test_500k = pd.read_parquet('benchmark_ds_500k_only.parquet')#.head(100)

    id2label_roberta = {
        "1": "B-BUILDINGNUM",
        "2": "B-CITY",
        "4": "B-DATE",
        "5": "B-DRIVERLICENSENUM",
        "6": "B-EMAIL",
        "8": "B-GIVENNAME",
        "9": "B-IDCARDNUM",
        "10": "B-PASSPORTNUM",
        "11": "B-SEX",
        "12": "B-SOCIALNUM",
        "13": "B-STREET",
        "14": "B-SURNAME",
        "16": "B-TELEPHONENUM",
        "17": "B-TIME",
        "18": "B-TITLE",
        "19": "B-ZIPCODE",
        "20": "I-BUILDINGNUM",
        "21": "I-CITY",
        "22": "I-DATE",
        "23": "I-DRIVERLICENSENUM",
        "24": "I-EMAIL",
        "25": "I-GIVENNAME",
        "26": "I-SOCIALNUM",
        "27": "I-STREET",
        "28": "I-SURNAME",
        "30": "I-TELEPHONENUM",
        "31": "I-TIME",
        "32": "I-TITLE",
        "33": "I-ZIPCODE",
        "34": "O"
    }

    # 2. Flatten the lists AND filter out the -100 tokens simultaneously
    true_labels_flat = []
    pred_labels_flat = []

    for true_seq, pred_seq in zip(df_test_500k["true_labels"], df_test_500k["pred_labels"]):
        # Iterate through pairs of (true, predicted) tokens in each sentence
        for true_token, pred_token in zip(true_seq, pred_seq):
            if true_token != -100:  # Ignore special tokens and subwords
                true_labels_flat.append(true_token)
                pred_labels_flat.append(pred_token)

    # 4. Generate the classification report
    print("-----------------------------------------------------------------")
    print("RoBERTa classification report, original training dataset only:")
    clfreport = classification_report(true_labels_flat, pred_labels_flat, labels = [k for k in id2label_roberta.keys()], target_names= [v for v in id2label_roberta.values()])
    print(clfreport)

def run_our_benchmark():
    print("=================================")
    print("Launching our benchmark to compare the two models on the same dataset...")


    # Test on 300k 
    # 1. Load the data
    df_test_300k = pd.read_parquet('benchmark_ds_300k.parquet')#.head(100)

    id2label_deberta = {
        "1": "I-BUILDING",
        "3": "I-CITY",
        "5": "I-DATE",
        "6": "I-DRIVERLICENSE",
        "7": "I-EMAIL",
        "9": "I-GIVENNAME1",
        "10": "I-GIVENNAME2",
        "11": "I-IDCARD",
        "13": "I-LASTNAME1",
        "14": "I-LASTNAME2",
        "15": "I-LASTNAME3",
        "17": "I-PASSPORT",
        "18": "I-POSTCODE",
        "20": "I-SEX",
        "21": "I-SOCIALNUMBER",
        "23": "I-STREET",
        "24": "I-TEL",
        "25": "I-TIME",
        "26": "I-TITLE",
        "29": "B-BUILDING",
        "31": "B-CITY",
        "33": "B-DATE",
        "34": "B-DRIVERLICENSE",
        "35": "B-EMAIL",
        "37": "B-GIVENNAME1",
        "38": "B-GIVENNAME2",
        "39": "B-IDCARD",
        "41": "B-LASTNAME1",
        "42": "B-LASTNAME2",
        "43": "B-LASTNAME3",
        "45": "B-PASSPORT",
        "46": "B-POSTCODE",
        "48": "B-SEX",
        "49": "B-SOCIALNUMBER",
        "51": "B-STREET",
        "52": "B-TEL",
        "53": "B-TIME",
        "54": "B-TITLE",
        "56": "O"
    }

    # 2. Flatten the lists AND filter out the -100 tokens simultaneously
    true_labels_flat = []
    pred_labels_flat = []

    for true_seq, pred_seq in zip(df_test_300k["true_labels"], df_test_300k["pred_labels"]):
        # Iterate through pairs of (true, predicted) tokens in each sentence
        for true_token, pred_token in zip(true_seq, pred_seq):
            if true_token != -100:  # Ignore special tokens and subwords
                true_labels_flat.append(true_token)
                pred_labels_flat.append(pred_token)

    # 4. Generate the classification report
    print("------------------------------------------------------------------------")
    print("DeBERTa classification report, original concatenated 300k+500k datasets:")
    clfreport = classification_report(true_labels_flat, pred_labels_flat, labels=[k for k in id2label_deberta.keys()], target_names=[v for v in id2label_deberta.values()])
    print(clfreport)



    ### and for RoBERTa !

    # 1. Load the data
    df_test_500k = pd.read_parquet('benchmark_ds_500k.parquet')#.head(100)

    id2label_roberta = {
        "1": "B-BUILDINGNUM",
        "2": "B-CITY",
        "4": "B-DATE",
        "5": "B-DRIVERLICENSENUM",
        "6": "B-EMAIL",
        "8": "B-GIVENNAME",
        "9": "B-IDCARDNUM",
        "10": "B-PASSPORTNUM",
        "11": "B-SEX",
        "12": "B-SOCIALNUM",
        "13": "B-STREET",
        "14": "B-SURNAME",
        "16": "B-TELEPHONENUM",
        "17": "B-TIME",
        "18": "B-TITLE",
        "19": "B-ZIPCODE",
        "20": "I-BUILDINGNUM",
        "21": "I-CITY",
        "22": "I-DATE",
        "23": "I-DRIVERLICENSENUM",
        "24": "I-EMAIL",
        "25": "I-GIVENNAME",
        "26": "I-SOCIALNUM",
        "27": "I-STREET",
        "28": "I-SURNAME",
        "30": "I-TELEPHONENUM",
        "31": "I-TIME",
        "32": "I-TITLE",
        "33": "I-ZIPCODE",
        "34": "O"
    }

    # 2. Flatten the lists AND filter out the -100 tokens simultaneously
    true_labels_flat = []
    pred_labels_flat = []

    for true_seq, pred_seq in zip(df_test_500k["true_labels"], df_test_500k["pred_labels"]):
        # Iterate through pairs of (true, predicted) tokens in each sentence
        for true_token, pred_token in zip(true_seq, pred_seq):
            if true_token != -100:  # Ignore special tokens and subwords
                true_labels_flat.append(true_token)
                pred_labels_flat.append(pred_token)

    # 4. Generate the classification report
    print("------------------------------------------------------------------------")
    print("RoBERTa classification report, original concatenated 500k+300k datasets:")
    clfreport = classification_report(true_labels_flat, pred_labels_flat, labels = [k for k in id2label_roberta.keys()], target_names= [v for v in id2label_roberta.values()])
    print(clfreport)


if __name__ == "__main__":
    
    run_original_benchmark()

    run_our_benchmark()
    