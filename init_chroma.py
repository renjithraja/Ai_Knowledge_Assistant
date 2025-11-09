"""
Initializes the multimodal knowledge base by extracting text and images
from PDFs, generating captions, and storing everything in ChromaDB.
"""

from pathlib import Path
from utils.pdf_utils import extract_text_and_images
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent

# Define data directories
PDF_DIR = Path("data/reports")
TEXT_DIR = Path("data/processed/text")
IMAGE_DIR = Path("data/processed/images")

# Create output directories if they don’t exist
TEXT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Extract text and images from all PDFs
for pdf_path in PDF_DIR.glob("*.pdf"):
    print(f"Extracting from {pdf_path.name}...")
    extract_text_and_images(pdf_path, TEXT_DIR, IMAGE_DIR)

# Step 2: Initialize retrieval and vision agents
retrieval = RetrievalAgent()
vision = VisionAgent()

# Step 3: Add text chunks to the Chroma vector store
for txt_file in TEXT_DIR.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split document into clean text chunks
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    ids = [f"{txt_file.stem}_{i}" for i in range(len(chunks))]
    metas = [{"source": str(txt_file), "type": "text"} for _ in chunks]

    retrieval.add_documents(ids, chunks, metas)

# Step 4: Generate image captions and store them
for img_path in IMAGE_DIR.glob("*.*"):
    caption = vision.describe_image(str(img_path))
    retrieval.add_documents(
        ids=[f"{img_path.stem}_img"],
        texts=[caption],
        metadatas=[{"source": str(img_path), "type": "image"}]
    )

print("Knowledge base successfully populated.")
