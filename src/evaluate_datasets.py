import pandas as pd
import ast
import time
from sklearn.metrics import classification_report
from transformers import pipeline

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

if __name__ == "__main__":
    # Test on your 300k or 500k csv
    df_test_300k = load_and_prep_data('benchmark_ds_300k.csv').head(100) # test on first 100 lines first!
    # df_test_500k = load_and_prep_data('benchmark_ds_500k.csv').head(100) # test on first 100 lines first!

    
    evaluate_model_speed_and_accuracy(df_test_300k, "yonigo/deberta-v3-base-pii-en")
    # evaluate_model_speed_and_accuracy(df_test_500k, "Ar86Bat/multilang-pii-ner")