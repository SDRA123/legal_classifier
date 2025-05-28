import runpod
import torch
import fitz  # PyMuPDF
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from huggingface_hub import hf_hub_download

# Constants
CONFIDENCE_THRESHOLD = 0.8
REQUIRED_CORPORATE_RATIO = 0.4
CHUNK_SIZE = 1000
MAX_CHUNKS = 10

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "daxa-ai/pebblo-classifier-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

# Load label encoder
LABEL_ENCODER_FILE = "label_encoder.joblib"
REPO_NAME = "daxa-ai/pebblo-classifier-v2"
filename = hf_hub_download(repo_id=REPO_NAME, filename=LABEL_ENCODER_FILE)
label_encoder = joblib.load(filename)

# --- Helper Functions ---
def get_text_chunks(text, chunk_size=1000):
    chunks = []
    text = text.strip()
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def classify_chunk(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    conf, predicted_idx = torch.max(probs, dim=-1)
    predicted_label = label_encoder.inverse_transform(predicted_idx.cpu().numpy())[0]
    return predicted_label, conf.item()

def is_corporate_document(file_path):
    chunks = get_text_chunks(file_path)
    if not chunks:
        return False

    sampled_chunks = random.sample(chunks, min(MAX_CHUNKS, len(chunks)))
    corporate_count = 0

    for i, chunk in enumerate(sampled_chunks, 1):
        label, confidence = classify_chunk(chunk)
        print(f"Chunk {i}: Predicted Label = {label}, Confidence = {confidence:.2f}")
        if label == "CORPORATE_DOCUMENTS" and confidence >= CONFIDENCE_THRESHOLD:
            corporate_count += 1

    ratio = corporate_count / len(sampled_chunks)
    print(f"Corporate Chunk Ratio: {ratio:.2f}")
    return ratio >= REQUIRED_CORPORATE_RATIO

# --- RunPod Handler ---
def handler(job):
    try:
        job_input = job["input"]
        file_path = job_input.get("prompt", "")
        if not file_path:
            return {"error": "No 'file_path' provided in input."}

        if is_corporate_document(file_path):
            return {"status": "accepted", "message": "Corporate document detected."}
        else:
            return {"status": "rejected", "message": "Not a corporate document."}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
