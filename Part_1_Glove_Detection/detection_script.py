import argparse
import json
import os
import sys
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# --- LABEL MAPPING CONFIGURATION ---
# Handles dataset noise. Maps 'surgical-gloves' and 'gloverotation' 
# (often white gloves) to the required 'gloved_hand' class.
LABEL_MAP = {
    "surgical-gloves": "gloved_hand",
    "glove": "gloved_hand",
    "gloverotation": "gloved_hand", # Mapping fix for white gloves
    "hand": "bare_hand",
    "bare": "bare_hand"
}

# BGR Colors: Green for Glove, Red for Bare
COLORS = {"gloved_hand": (0, 255, 0), "bare_hand": (0, 0, 255)}

def normalize_label(raw_label):
    """Normalize dataset labels to project requirements."""
    if raw_label in LABEL_MAP:
        return LABEL_MAP[raw_label]
    # Fallback string matching
    if "glove" in raw_label.lower():
        return "gloved_hand"
    return "bare_hand"

def run_inference(input_dir, output_dir, model_path, conf_thres):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    logs_path = Path("submission/Part_1_Glove_Detection/logs")

    # Ensure directories exist
    output_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        sys.exit(1)

    # Gather images
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in input_path.iterdir() if p.suffix.lower() in valid_exts]
    
    if not images:
        print("No images found in input directory.")
        return

    print(f"Processing {len(images)} images...")
    
    # Run Inference (Stream mode for memory efficiency)
    results = model.predict(
        source=[str(p) for p in images],
        conf=conf_thres,
        stream=True,
        verbose=False
    )

    for result in tqdm(results, total=len(images)):
        filename = os.path.basename(result.path)
        img_out = result.orig_img.copy()
        
        # JSON Schema
        log_entry = {
            "filename": filename,
            "detections": []
        }

        for box in result.boxes:
            # Extract Integer Coordinates
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            
            # Resolve Class Name
            raw_class = model.names[int(box.cls[0])]
            label = normalize_label(raw_class)

            # Update Log
            log_entry["detections"].append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

            # Draw Annotation
            color = COLORS.get(label, (255, 255, 255))
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_out, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(
                img_out, label_text, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Save Artifacts
        cv2.imwrite(str(output_path / filename), img_out)
        
        json_path = logs_path / f"{Path(filename).stem}.json"
        with open(json_path, 'w') as f:
            json.dump(log_entry, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder")
    parser.add_argument("--output", default="submission/Part_1_Glove_Detection/output")
    parser.add_argument("--weights", required=True, help="Path to .pt model")
    parser.add_argument("--confidence", type=float, default=0.45)
    
    args = parser.parse_args()
    run_inference(args.input, args.output, args.weights, args.confidence)
