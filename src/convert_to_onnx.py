######## FILE 5 ########
#Convert a PyTorch model to ONNX, 
# I first try with HF's Optimum: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model
# Then, If it does not work, Ill try following these docs: https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html

## needed: uv add onnx onnxscript

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def get_hf_hub_token():
    # Dynamically locate the script's directory
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent  # This points to 'src/'

    #get hf token and log in to access the dataset
    with open(PROJECT_ROOT / "secrets.json", "r") as f:
        jsonf = json.load(f)
        hf_hub_token = jsonf["hf_hub_token"]

    return hf_hub_token


if __name__ == "__main__":
    ## 1. Get the HF model

    print("Getting model...")
    tokenizer = AutoTokenizer.from_pretrained("yonigo/deberta-v3-base-pii-en")#, torchscript = True)
    model = AutoModelForTokenClassification.from_pretrained("yonigo/deberta-v3-base-pii-en")#, torchscript=True)

    model.eval()

    # create sample input for onnx export
    print("Creating sample input...")
    sample_text = "Hello, my name is John Smith."
    sample_tokens = tokenizer(sample_text, return_tensors="pt")
    example_inputs = (sample_tokens["input_ids"], sample_tokens["attention_mask"])

    # export to onnx
    print("Converting to ONNX...")
    onnx_program = torch.onnx.export(
        model,
        example_inputs,
        dynamo=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
    )
    # ## 2. Get the PyTorch Model

    # # traced_model = torch.jit.trace(model)#, [tokens_tensor, segments_tensors]) #[TODO] check that the commented arguments are not vital.
    # # torch.jit.save(traced_model, "traced_bert.pt")

    # ## 3. Convert to ONNX

    # # Create example inputs for exporting the model. The inputs should be a tuple of tensors.
    # example_inputs = pass #TODO
    # onnx_program = torch.onnx.export(traced_model, example_inputs, dynamo=True)

    #### qstion: "eval" mode as I saw on some couple forums ? What does it do ?

    ## 4. Save to a file (optional)

    onnx_program.save("deberta_v3_PII.onnx")

    # INT8 quantization
    print("Quantizing to INT8...")
    quantize_dynamic(
        "deberta_v3_PII.onnx",
        "deberta_v3_PII_int8.onnx",
        weight_type=QuantType.QInt8,
    )

    # UINT8 quantization
    print("Quantizing to UINT8...")
    quantize_dynamic(
        "deberta_v3_PII.onnx",
        "deberta_v3_PII_uint8.onnx",
        weight_type=QuantType.QUInt8,
    )

    # FP16
    onnx_program_fp16 = torch.onnx.export(
        model,
        example_inputs,
        dynamo=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        fp16=True,
    )
    onnx_program_fp16.save("deberta_v3_PII_fp16.onnx")


    ## 5. Push to the Hub
    # hf_hub_token = get_hf_hub_token()
    # onnx_program.push_to_hub("test-onnx-export-deberta-v3")

    ## 6. Compare performances