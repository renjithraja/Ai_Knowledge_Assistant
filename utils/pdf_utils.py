"""

Extracts text and embedded images from PDF documents using PyMuPDF.
Saves extracted content into organized text and image directories.
"""

import fitz  # PyMuPDF
from pathlib import Path


def extract_text_and_images(pdf_path, text_dir, image_dir):
    """
    Extract text and images from a given PDF file.

    Args:
        pdf_path (Path): Path to the source PDF file.
        text_dir (Path): Directory to store extracted text files.
        image_dir (Path): Directory to store extracted images.

    Saves:
        - A .txt file containing all page text.
        - Individual image files extracted from each page.
    """
    doc = fitz.open(pdf_path)
    text_output = []

    for page_num, page in enumerate(doc):
        # Extract plain text from the page
        text = page.get_text("text")
        text_output.append(text)

        # Extract all embedded images on the page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Build image file path
            img_path = image_dir / f"{Path(pdf_path).stem}_p{page_num}_img{img_index}.{image_ext}"

            try:
                # Convert to RGB if needed for unsupported color formats
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # Handles CMYK or alpha channels
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                pix.save(str(img_path))
                pix = None  # Cleanup to free memory

            except Exception as e:
                print(f"Skipping image {img_index} on page {page_num}: {e}")
                continue

    # Save extracted text to file
    text_path = text_dir / f"{Path(pdf_path).stem}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_output))

    doc.close()
