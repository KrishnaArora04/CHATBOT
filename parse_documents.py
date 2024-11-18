import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a given PDF file.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages])
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Example of calling the function with print output
pdf_paths = [
    "/Users/krishnaarora/Desktop/content-engine/data/goog-10-k-2023 (1).pdf",
    "/Users/krishnaarora/Desktop/content-engine/data/tsla-20231231-gen.pdf",
    "/Users/krishnaarora/Desktop/content-engine/data/uber-10-k-2023.pdf"
]

for path in pdf_paths:
    extracted_text = extract_text_from_pdf(path)
    if extracted_text:
        print(f"Extracted text from {path}:\n{extracted_text[:500]}...")  # Print first 500 characters
    else:
        print(f"Failed to extract text from {path}")
