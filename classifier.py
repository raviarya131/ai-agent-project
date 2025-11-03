import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import sys
import json

# --- Configuration ---
# This is the base model you trained on
BASE_MODEL_ID = "google/gemma-2b-it"
# This is the path to your downloaded adapter (the unzipped folder)
ADAPTER_PATH = "./email-classifier-gemma-2b-lora-final"

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Classifier Tool: Using device: {device}")

# --- Load Model & Tokenizer (with 4-bit) ---
print("Classifier Tool: Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    num_labels=3,  # Urgent, To-Do, FYI
    device_map={"": 0}  # Load on the first GPU
)
model.config.pad_token_id = tokenizer.pad_token_id

# --- Load the LoRA Adapter ---
print(f"Classifier Tool: Loading adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()  # Set to evaluation mode

# --- Labels ---
id2label = {0: "Urgent", 1: "To-Do", 2: "FYI"}


def classify_email(email_text):
    """
    Classifies a single email text using the fine-tuned LoRA model.
    """
    print(f"Classifier Tool: Received text: '{email_text[:50]}...'")

    # Tokenize
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the highest probability class
    prediction = torch.argmax(logits, dim=-1).item()
    label = id2label[prediction]

    print(f"Classifier Tool: Classification result: {label}")
    return {"classification": label}


if __name__ == "__main__":
    # This allows the script to be called from the command line
    # by our agent.
    if len(sys.argv) > 1:
        # Read text from command line arguments
        text_to_classify = sys.argv[1]
        result = classify_email(text_to_classify)
        # Print the result as a JSON string
        print(json.dumps(result))
    else:
        print("Classifier Tool: No text provided. Running a test...")
        test_result = classify_email("Subject: Urgent meeting\n\nWe need to meet now.")
        print(f"Classifier Tool: Test result: {test_result}")