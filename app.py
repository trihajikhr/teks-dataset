import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io

app = FastAPI(title="NoteScanin AI Service", description="OCR API using TrOCR")

# Constants
# Load from .env if available, fallback to default model
MODEL_DIR = os.getenv("MODEL_DIR", "microsoft/trocr-base-handwritten")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = None
model = None

@app.on_event("startup")
def load_model():
    global processor, model
    print(f"Loading model from {MODEL_DIR} to {device}...")
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Not exiting so the server can still run and we can test the endpoints,
        # but OCR will fail if model is None

import cv2
import numpy as np

def segment_image_into_lines(pil_image):
    img_gray = np.array(pil_image.convert('L'))
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    hist = np.sum(thresh, axis=1)
    
    lines = []
    in_line = False
    start_y = 0
    threshold_hist = np.max(hist) * 0.05 if np.max(hist) > 0 else 50
    
    for y, val in enumerate(hist):
        if not in_line and val > threshold_hist:
            in_line = True
            start_y = y
        elif in_line and val <= threshold_hist:
            in_line = False
            if y - start_y > 15:
                margin = 5
                y1 = max(0, start_y - margin)
                y2 = min(img_gray.shape[0], y + margin)
                lines.append((y1, y2))
                
    if in_line and (img_gray.shape[0] - start_y > 15):
        lines.append((max(0, start_y - 5), img_gray.shape[0]))
        
    if not lines:
        return [pil_image]
        
    line_images = []
    rgb_image = np.array(pil_image)
    for (y1, y2) in lines:
        crop_np = rgb_image[y1:y2, :, :]
        line_images.append(Image.fromarray(crop_np))
    return line_images

@app.post("/ocr")
def process_ocr(file: UploadFile = File(...)):
    if not model or not processor:
        raise HTTPException(status_code=500, detail="Model is not loaded properly")
        
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        from PIL import ImageOps, ImageEnhance
        
        # 1. Invert if background is dark
        image_gray = ImageOps.grayscale(image)
        if np.mean(np.array(image_gray)) < 127:
            image_gray = ImageOps.invert(image_gray)
            
        image = image_gray.convert("RGB")
        
        # 2. Contrast enhancement
        image = ImageOps.autocontrast(image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # 3. Line Segmentation
        line_images = segment_image_into_lines(image)
        print(f"Memproses {len(line_images)} baris teks...")
        
        full_text = []
        confidences = []
        
        # 4. Loop Inference per baris
        for line_img in line_images:
            pixel_values = processor(images=line_img, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_length=128
                )
                
            generated_ids = outputs.sequences
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if text.strip():
                full_text.append(text.strip())
                
            # Confidence calculation
            if outputs.scores:
                probs = []
                for i, score in enumerate(outputs.scores):
                    token_probs = torch.nn.functional.softmax(score, dim=-1)
                    if i + 1 < len(generated_ids[0]):
                        token_id = generated_ids[0][i + 1]
                        prob = token_probs[0, token_id].item()
                        probs.append(prob)
                if probs:
                    confidences.append(sum(probs) / len(probs))
            else:
                confidences.append(0.95)
                
        final_text = "\n".join(full_text)
        final_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "raw_text": final_text,
            "clean_text": final_text,
            "confidence": round(final_confidence, 4)
        }
        
    except Exception as e:
        print(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
