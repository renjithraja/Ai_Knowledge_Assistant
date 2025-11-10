"""
Handles basic image understanding using OpenAI CLIP.
Generates short semantic descriptions (labels) for images.
Lightweight and CPU-friendly, ideal for Streamlit deployment.
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import torch


class VisionAgent:
    def __init__(self):
        """
        Initialize the CLIP model and processor.
        Automatically selects GPU if available, otherwise CPU.
        """
        try:
            print("Loading CLIP model (openai/clip-vit-base-patch16)...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"Vision model loaded successfully on {self.device.upper()}.")
        except Exception as e:
            print(f"Error initializing CLIP model: {e}")
            raise RuntimeError("VisionAgent initialization failed.")

    def describe_image(self, image_path):
        """
        Generates a simple textual description for the given image.
        Safely handles unreadable or missing image files.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return "image file missing"

        try:
            # Attempt to open the image safely
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"Unreadable image skipped: {image_path} ({e})")
            return "unreadable or unsupported image"

        try:
            # Candidate text labels to classify the image meaning
            candidate_labels = [
                "a photo of a chart or graph",
                "a photo of a document page",
                "a photo of a natural scene",
                "a photo of a person",
                "a photo of a diagram or infographic",
                "a photo of a technical figure"
            ]

            # Prepare CLIP inputs (image + text)
            inputs = self.processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass through CLIP
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Select the label with the highest probability
            best_label = candidate_labels[probs.argmax().item()]
            confidence = probs.max().item()

            # Free memory (important on Streamlit Cloud)
            del inputs, outputs
            torch.cuda.empty_cache()

            return f"This image likely shows {best_label} (confidence: {confidence:.2f})."

        except Exception as e:
            print(f"Error describing image {image_path}: {e}")
            return "image analysis failed"
