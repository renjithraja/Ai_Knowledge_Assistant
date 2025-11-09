"""
Handles image understanding using a Vision-Language Model (BLIP-2).
Generates text descriptions (captions) for visual content.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import torch


class VisionAgent:
    def __init__(self):
        """
        Load the BLIP-2 image captioning model and processor.
        Runs on GPU if available, otherwise on CPU.
        """
        try:
            print("Loading BLIP-2 model (Salesforce/blip-image-captioning-base)...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"Vision model loaded successfully on {self.device.upper()}.")
        except Exception as e:
            print(f"Model initialization failed: {e}")
            raise RuntimeError("VisionAgent initialization failed.")

    def describe_image(self, image_path):
        """
        Generate a caption for a given image using the BLIP-2 model.
        Safely handles missing or unreadable image files.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return "image file missing"

        try:
            # Open and convert the image to RGB
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"Unreadable image skipped: {image_path} ({e})")
            return "unreadable or unsupported image"

        try:
            # Prepare model inputs
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Generate caption from image features
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50)

            caption = self.processor.decode(output[0], skip_special_tokens=True).strip()

            # Free memory (especially important when running on GPU)
            del inputs, output
            torch.cuda.empty_cache()

            return caption or "no caption generated"

        except Exception as e:
            print(f"Caption generation failed for {image_path}: {e}")
            return "image captioning failed"
