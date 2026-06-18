from evaluate import evaluator
from convert_to_evaluator import format_benchmark_datasets

if __name__ == "__main__":
    task_evaluator = evaluator("token-classification")

    [(benchmark_ds_3OOk, tokenizer_deberta,model_deberta), (benchmark_ds_5OOk, tokenizer_roberta, model_roberta)] = format_benchmark_datasets()

    results = task_evaluator.compute(
        model_or_pipeline = model_deberta,
        data=benchmark_ds_3OOk
        metric=seqeval
    )

    results = task_evaluator.compute(
        model_or_pipeline = model_roberta,
        data=benchmark_ds_5OOk
        metric=seqeval
    )