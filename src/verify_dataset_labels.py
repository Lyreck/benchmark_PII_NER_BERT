#!/usr/bin/env python3
"""
Standalone verification script for benchmark dataset labels.

Loads the benchmark parquet files, replicates the tokenization logic from
create_true_and_pred_datasets.py, and compares annotation counts
at every pipeline stage to identify discrepancies.

Usage:
    python src/verify_dataset_labels.py
    python src/verify_dataset_labels.py --sample 1000
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PARQUET_300K =  "benchmark_ds_300k.parquet"
DEFAULT_PARQUET_500K =  "benchmark_ds_500k.parquet"

ROBERTA_SPECIAL_NO_I = {"PASSPORTNUM", "AGE", "CREDITCARDNUMBER", "GENDER", "IDCARDNUM", "SEX"}

# DeBERTa id2label (from model config, after Transformers casts string keys to int)
DEBERTA_ID2LABEL = {
    0: "I-BOD", 1: "I-BUILDING", 2: "I-CARDISSUER", 3: "I-CITY", 4: "I-COUNTRY",
    5: "I-DATE", 6: "I-DRIVERLICENSE", 7: "I-EMAIL", 8: "I-GEOCOORD",
    9: "I-GIVENNAME1", 10: "I-GIVENNAME2", 11: "I-IDCARD", 12: "I-IP",
    13: "I-LASTNAME1", 14: "I-LASTNAME2", 15: "I-LASTNAME3", 16: "I-PASS",
    17: "I-PASSPORT", 18: "I-POSTCODE", 19: "I-SECADDRESS", 20: "I-SEX",
    21: "I-SOCIALNUMBER", 22: "I-STATE", 23: "I-STREET", 24: "I-TEL",
    25: "I-TIME", 26: "I-TITLE", 27: "I-USERNAME",
    28: "B-BOD", 29: "B-BUILDING", 30: "B-CARDISSUER", 31: "B-CITY",
    32: "B-COUNTRY", 33: "B-DATE", 34: "B-DRIVERLICENSE", 35: "B-EMAIL",
    36: "B-GEOCOORD", 37: "B-GIVENNAME1", 38: "B-GIVENNAME2", 39: "B-IDCARD",
    40: "B-IP", 41: "B-LASTNAME1", 42: "B-LASTNAME2", 43: "B-LASTNAME3",
    44: "B-PASS", 45: "B-PASSPORT", 46: "B-POSTCODE", 47: "B-SECADDRESS",
    48: "B-SEX", 49: "B-SOCIALNUMBER", 50: "B-STATE", 51: "B-STREET",
    52: "B-TEL", 53: "B-TIME", 54: "B-TITLE", 55: "B-USERNAME", 56: "O",
}

# RoBERTa id2label (from model config)
ROBERTA_ID2LABEL = {
    0: "B-AGE", 1: "B-BUILDINGNUM", 2: "B-CITY", 3: "B-CREDITCARDNUMBER",
    4: "B-DATE", 5: "B-DRIVERLICENSENUM", 6: "B-EMAIL", 7: "B-GENDER",
    8: "B-GIVENNAME", 9: "B-IDCARDNUM", 10: "B-PASSPORTNUM", 11: "B-SEX",
    12: "B-SOCIALNUM", 13: "B-STREET", 14: "B-SURNAME", 15: "B-TAXNUM",
    16: "B-TELEPHONENUM", 17: "B-TIME", 18: "B-TITLE", 19: "B-ZIPCODE",
    20: "I-BUILDINGNUM", 21: "I-CITY", 22: "I-DATE", 23: "I-DRIVERLICENSENUM",
    24: "I-EMAIL", 25: "I-GIVENNAME", 26: "I-SOCIALNUM", 27: "I-STREET",
    28: "I-SURNAME", 29: "I-TAXNUM", 30: "I-TELEPHONENUM", 31: "I-TIME",
    32: "I-TITLE", 33: "I-ZIPCODE", 34: "O",
}

# Labels the DeBERTa pipeline ignores (not in benchmark set)
DEBERTA_IGNORE = {"USERNAME", "COUNTRY", "STATE", "PASS", "BOD", "IP", "SECADDRESS", "GEOCOORD", "CARDISSUER"}

# ---------------------------------------------------------------------------
# is_subword — self-contained copy matching the fixed version in the codebase
# ---------------------------------------------------------------------------

def is_subword(tokenizer, token_id):
    """Return True if token_id represents a subword (continuation) token."""
    word = tokenizer.convert_ids_to_tokens([token_id])[0]
    return not word.startswith(("\u0120", "\u2581", "##"))  # Ġ, ▁, ##


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

def load_parquet(path, max_rows=None):
    """Load benchmark parquet. Returns list of (source_text, privacy_mask) tuples."""
    import pandas as pd

    df = pd.read_parquet(path)
    if max_rows is not None and max_rows < len(df):
        df = df.head(max_rows)

    rows = []
    for _, row in df.iterrows():
        text = row["source_text"]
        mask = row["privacy_mask"]
        if isinstance(mask, str):
            import ast
            mask = ast.literal_eval(mask)
        rows.append((text, mask))
    return rows


def count_annotations(rows):
    """Count character-level annotations per label and collect entity lengths."""
    counts = Counter()
    lengths = defaultdict(list)
    samples = defaultdict(list)

    for text, mask in rows:
        for ann in mask:
            label = ann["label"]
            counts[label] += 1
            entity_len = ann["end"] - ann["start"]
            lengths[label].append(entity_len)
            if len(samples[label]) < 5:
                samples[label].append(text[ann["start"]:ann["end"]])

    return counts, lengths, samples


# ---------------------------------------------------------------------------
# Tokenizer-level analysis (replicates tokenize_robust without model inference)
# ---------------------------------------------------------------------------

def tokenize_and_count(rows, tokenizer, id2label, model_id, ignore_subwords=True):
    """Tokenize all texts and count B-/I-/O/-100 labels per entity type.

    Replicates the logic of tokenize_robust from create_true_and_pred_datasets.py.
    """
    label2id = {v: k for k, v in id2label.items()}

    b_counts = Counter()
    i_counts = Counter()
    o_count = 0
    ignore_count = 0
    char_to_token_failures = 0
    entity_warnings = 0

    for text, mask in rows:
        tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)

        # Build start_token_to_label
        start_token_to_label = {}
        for ann in mask:
            tok_idx = tokenized.char_to_token(ann["start"])
            if tok_idx is not None:
                start_token_to_label[tok_idx] = (ann["start"], ann["end"], ann["label"])
            else:
                char_to_token_failures += 1

        i = 0
        while i < len(tokenized["input_ids"]):
            if tokenized["special_tokens_mask"][i] == 1:
                ignore_count += 1
                i += 1
            elif i not in start_token_to_label:
                if ignore_subwords and is_subword(tokenizer, tokenized["input_ids"][i]):
                    ignore_count += 1
                else:
                    o_count += 1
                i += 1
            else:
                start, end, label = start_token_to_label[i]
                start_token = tokenized.char_to_token(start)
                assert start_token == i
                j = start_token
                while j < (len(tokenized["input_ids"]) - 1) and tokenized.token_to_chars(j).start < end:
                    if j == start_token:
                        b_counts[label] += 1
                    elif ignore_subwords and is_subword(tokenizer, tokenized["input_ids"][j]):
                        ignore_count += 1
                    else:
                        if label in ROBERTA_SPECIAL_NO_I and model_id == "Ar86Bat/multilang-pii-ner":
                            b_counts[label] += 1  # B-tag for continuation (RoBERTa special case)
                        else:
                            i_counts[label] += 1
                    j += 1
                i = j

    return {
        "b_counts": b_counts,
        "i_counts": i_counts,
        "o_count": o_count,
        "ignore_count": ignore_count,
        "char_to_token_failures": char_to_token_failures,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_analysis1(csv_counts, csv_lengths, csv_samples, csv_name):
    print(f"\n{'=' * 80}")
    print(f"Analysis 1: CSV annotation counts — {csv_name}")
    print(f"{'=' * 80}")
    print(f"{'Label':<25} {'Count':>8} {'Char_len(min,max,mean)':>25}")
    print(f"{'-' * 60}")
    for label in sorted(csv_counts.keys()):
        cnt = csv_counts[label]
        lens = csv_lengths[label]
        min_l = min(lens) if lens else 0
        max_l = max(lens) if lens else 0
        mean_l = sum(lens) / len(lens) if lens else 0
        print(f"{label:<25} {cnt:>8} {f'({min_l}, {max_l}, {mean_l:.1f})':>25}")

    print(f"\nEntity text samples:")
    for label in sorted(csv_samples.keys()):
        print(f"  {label}: {csv_samples[label]}")


def print_analysis2(result, tokenizer_name, ignore_subwords):
    b = result["b_counts"]
    i = result["i_counts"]
    o = result["o_count"]
    ign = result["ignore_count"]
    c2t_fails = result["char_to_token_failures"]
    total_b_i = sum(b.values()) + sum(i.values())

    print(f"\n{'=' * 80}")
    print(f"Analysis 2: Tokenizer-level labels — {tokenizer_name} (ignore_subwords={ignore_subwords})")
    print(f"{'=' * 80}")
    print(f"{'Label':<25} {'B-count':>8} {'I-count':>8} {'B+I':>8}")
    print(f"{'-' * 52}")
    all_labels = sorted(set(list(b.keys()) + list(i.keys())))
    for label in all_labels:
        bc = b.get(label, 0)
        ic = i.get(label, 0)
        print(f"{label:<25} {bc:>8} {ic:>8} {bc + ic:>8}")

    print(f"\nO tokens: {o}")
    print(f"Ignored tokens (-100): {ign}")
    print(f"Total B+I labels: {total_b_i}")
    print(f"char_to_token failures (annotations with unmappable start): {c2t_fails}")


def print_analysis3(csv_counts_300k, csv_counts_500k, deberta_result, roberta_result, ignore_subwords):
    print(f"\n{'=' * 80}")
    print(f"Analysis 3: Consolidated comparison table (ignore_subwords={ignore_subwords})")
    print(f"{'=' * 80}")

    # --- DeBERTa on 300k labels ---
    print(f"\n--- DeBERTa tokenizer vs 300k data ---")
    print(f"{'Label':<25} {'Ann':>8} {'DeB_B':>8} {'DeB_I':>8} {'DeB_B+I':>8} {'Match':>8}")
    print(f"{'-' * 68}")
    all_labels_300k = sorted(set(list(csv_counts_300k.keys()) + list(deberta_result["b_counts"].keys()) + list(deberta_result["i_counts"].keys())))
    for label in all_labels_300k:
        csv_ann = csv_counts_300k.get(label, 0)
        bc = deberta_result["b_counts"].get(label, 0)
        ic = deberta_result["i_counts"].get(label, 0)
        bi = bc + ic
        if csv_ann > 0:
            match = f"{bi / csv_ann * 100:.0f}%"
        elif bi == 0:
            match = "OK"
        else:
            match = f"+{bi}"
        print(f"{label:<25} {csv_ann:>8} {bc:>8} {ic:>8} {bi:>8} {match:>8}")

    # --- RoBERTa on 500k labels ---
    print(f"\n--- RoBERTa tokenizer vs 500k data ---")
    print(f"{'Label':<25} {'Ann':>8} {'RoB_B':>8} {'RoB_I':>8} {'RoB_B+I':>8} {'Match':>8}")
    print(f"{'-' * 68}")
    all_labels_500k = sorted(set(list(csv_counts_500k.keys()) + list(roberta_result["b_counts"].keys()) + list(roberta_result["i_counts"].keys())))
    for label in all_labels_500k:
        csv_ann = csv_counts_500k.get(label, 0)
        bc = roberta_result["b_counts"].get(label, 0)
        ic = roberta_result["i_counts"].get(label, 0)
        bi = bc + ic
        if csv_ann > 0:
            match = f"{bi / csv_ann * 100:.0f}%"
        elif bi == 0:
            match = "OK"
        else:
            match = f"+{bi}"
        print(f"{label:<25} {csv_ann:>8} {bc:>8} {ic:>8} {bi:>8} {match:>8}")


def print_analysis4(rows, tokenizer, id2label, model_id, labels_of_interest):
    """Spot-check specific labels: show tokenization details for up to 3 examples each."""
    print(f"\n{'=' * 80}")
    print(f"Analysis 4: Spot-check examples for labels of interest")
    print(f"{'=' * 80}")

    label2id = {v: k for k, v in id2label.items()}

    examples_shown = defaultdict(int)
    MAX_EXAMPLES = 3

    for text, mask in rows:
        if all(examples_shown[l] >= MAX_EXAMPLES for l in labels_of_interest):
            break

        relevant_anns = [a for a in mask if a["label"] in labels_of_interest]
        if not relevant_anns:
            continue

        tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)

        for ann in relevant_anns:
            label = ann["label"]
            if examples_shown[label] >= MAX_EXAMPLES:
                continue

            tok_idx = tokenized.char_to_token(ann["start"])
            if tok_idx is None:
                print(f"\n  [{label}] char_to_token({ann['start']}) returned None!")
                print(f"    Entity: '{text[ann['start']:ann['end']]}'")
                examples_shown[label] += 1
                continue

            # Walk entity tokens
            entity_tokens = []
            j = tok_idx
            end = ann["end"]
            while j < (len(tokenized["input_ids"]) - 1) and tokenized.token_to_chars(j).start < end:
                raw_token = tokenizer.convert_ids_to_tokens([tokenized["input_ids"][j]])[0]
                decoded = tokenizer.decode([tokenized["input_ids"][j]])
                offset = tokenized["offset_mapping"][j]
                sub = is_subword(tokenizer, tokenized["input_ids"][j])
                is_first = (j == tok_idx)
                entity_tokens.append({
                    "index": j,
                    "raw_token": raw_token,
                    "decoded": decoded,
                    "offset": offset,
                    "is_subword": sub,
                    "is_first": is_first,
                })
                j += 1

            print(f"\n  [{label}] Entity: '{text[ann['start']:ann['end']]}' (start={ann['start']}, end={ann['end']})")
            print(f"  Token index: {tok_idx}, entity spans tokens {tok_idx}..{j - 1}")
            for t in entity_tokens:
                flag = "FIRST" if t["is_first"] else ("SUB" if t["is_subword"] else "CONT")
                print(f"    token[{t['index']}] raw={t['raw_token']:<15} decoded={t['decoded']:<10} "
                      f"offset={t['offset']}  {flag}")
            examples_shown[label] += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify benchmark dataset labels")
    parser.add_argument("--sample", type=int, default=None,
                        help="Analyze only N random rows (default: all)")
    parser.add_argument("--parquet-300k", type=str, default=str(DEFAULT_PARQUET_300K))
    parser.add_argument("--parquet-500k", type=str, default=str(DEFAULT_PARQUET_500K))
    args = parser.parse_args()

    # Lazy imports so the script can run with minimal deps for --help
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers package required. Install with: pip install transformers")
        sys.exit(1)

    print("Loading parquet files...")
    rows_300k = load_parquet(args.parquet_300k, max_rows=args.sample)
    rows_500k = load_parquet(args.parquet_500k, max_rows=args.sample)
    print(f"  300k: {len(rows_300k)} rows")
    print(f"  500k: {len(rows_500k)} rows")

    # --- Analysis 1 ---
    csv_counts_300k, csv_lengths_300k, csv_samples_300k = count_annotations(rows_300k)
    csv_counts_500k, csv_lengths_500k, csv_samples_500k = count_annotations(rows_500k)

    print_analysis1(csv_counts_300k, csv_lengths_300k, csv_samples_300k, "300k")
    print_analysis1(csv_counts_500k, csv_lengths_500k, csv_samples_500k, "500k")

    # --- Analysis 2 ---
    print("\nLoading tokenizers...")
    tok_deberta = AutoTokenizer.from_pretrained("yonigo/deberta-v3-base-pii-en")
    tok_roberta = AutoTokenizer.from_pretrained("Ar86Bat/multilang-pii-ner")

    # DeBERTa on 300k
    print("\nTokenizing 300k CSV with DeBERTa tokenizer...")
    deberta_300k = tokenize_and_count(rows_300k, tok_deberta, DEBERTA_ID2LABEL,
                                       "yonigo/deberta-v3-base-pii-en", ignore_subwords=True)
    print_analysis2(deberta_300k, "DeBERTa", ignore_subwords=True)

    # RoBERTa on 500k
    print("\nTokenizing 500k CSV with RoBERTa tokenizer...")
    roberta_500k = tokenize_and_count(rows_500k, tok_roberta, ROBERTA_ID2LABEL,
                                       "Ar86Bat/multilang-pii-ner", ignore_subwords=True)
    print_analysis2(roberta_500k, "RoBERTa", ignore_subwords=True)

    # Also run without ignore_subwords for comparison
    print("\n--- (Re-running with ignore_subwords=False for comparison) ---")
    deberta_300k_noignore = tokenize_and_count(rows_300k, tok_deberta, DEBERTA_ID2LABEL,
                                                "yonigo/deberta-v3-base-pii-en", ignore_subwords=False)
    roberta_500k_noignore = tokenize_and_count(rows_500k, tok_roberta, ROBERTA_ID2LABEL,
                                                "Ar86Bat/multilang-pii-ner", ignore_subwords=False)

    print(f"\nDeBERTa (ignore_subwords=False) — Total O: {deberta_300k_noignore['o_count']}, "
          f"Ignored: {deberta_300k_noignore['ignore_count']}, "
          f"B+I: {sum(deberta_300k_noignore['b_counts'].values()) + sum(deberta_300k_noignore['i_counts'].values())}")
    print(f"RoBERTa (ignore_subwords=False) — Total O: {roberta_500k_noignore['o_count']}, "
          f"Ignored: {roberta_500k_noignore['ignore_count']}, "
          f"B+I: {sum(roberta_500k_noignore['b_counts'].values()) + sum(roberta_500k_noignore['i_counts'].values())}")

    # --- Analysis 3 ---
    print_analysis3(csv_counts_300k, csv_counts_500k, deberta_300k, roberta_500k, ignore_subwords=True)

    # --- Analysis 4 ---
    # Spot-check labels that had surprising results
    labels_of_interest_deberta = {"GIVENNAME1", "LASTNAME1", "SEX", "TITLE", "BUILDING", "EMAIL"}
    labels_of_interest_roberta = {"GIVENNAME", "SURNAME", "TITLE", "BUILDINGNUM", "SEX", "EMAIL"}

    print("\n--- Spot-checks for DeBERTa tokenizer on 300k data ---")
    print_analysis4(rows_300k, tok_deberta, DEBERTA_ID2LABEL,
                    "yonigo/deberta-v3-base-pii-en", labels_of_interest_deberta)

    print("\n--- Spot-checks for RoBERTa tokenizer on 500k data ---")
    print_analysis4(rows_500k, tok_roberta, ROBERTA_ID2LABEL,
                    "Ar86Bat/multilang-pii-ner", labels_of_interest_roberta)

    # --- Summary of anomalies ---
    print(f"\n{'=' * 80}")
    print("Summary of anomalies")
    print(f"{'=' * 80}")

    for label in sorted(csv_counts_300k.keys()):
        csv_ann = csv_counts_300k[label]
        bc = deberta_300k["b_counts"].get(label, 0)
        if csv_ann > 0 and bc == 0:
            print(f"  [!] {label}: {csv_ann} CSV annotations but 0 B-tokens in DeBERTa tokenization")
        elif csv_ann > 0 and abs(bc - csv_ann) / csv_ann > 0.05:
            print(f"  [?] {label}: {csv_ann} CSV annotations vs {bc} B-tokens ({bc/csv_ann*100:.0f}%)")

    for label in sorted(csv_counts_500k.keys()):
        csv_ann = csv_counts_500k[label]
        bc = roberta_500k["b_counts"].get(label, 0)
        if csv_ann > 0 and bc == 0:
            print(f"  [!] {label}: {csv_ann} CSV annotations but 0 B-tokens in RoBERTa tokenization")
        elif csv_ann > 0 and abs(bc - csv_ann) / csv_ann > 0.05:
            print(f"  [?] {label}: {csv_ann} CSV annotations vs {bc} B-tokens ({bc/csv_ann*100:.0f}%)")

    if deberta_300k["char_to_token_failures"] > 0:
        print(f"\n  [!] {deberta_300k['char_to_token_failures']} annotations had char_to_token return None (DeBERTa)")
    if roberta_500k["char_to_token_failures"] > 0:
        print(f"  [!] {roberta_500k['char_to_token_failures']} annotations had char_to_token return None (RoBERTa)")

    print("\nDone.")


if __name__ == "__main__":
    main()
