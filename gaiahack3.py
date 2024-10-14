import os
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
import re
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        logger.info(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def split_text_into_sections(text: str) -> List[Tuple[str, str]]:
    # More flexible regex pattern
    sections = re.split(r"^([SG]?\s*(\d+\.\d+))\s+(.*)", text)
    result = [(sections[i], sections[i+1]) for i in range(1, len(sections)-1, 2)]
    logger.info(f"Split text into {len(result)} sections")
    return result

def compare_texts(text1: str, text2: str) -> str:
    prompt = f"""
    Compare the following two versions of a policy document and identify any additions, removals, or meaningful changes. 
    Meaningful changes are those that alter the semantic meaning and would require internal policies to be updated. 
    Ignore changes in words that have similar meanings.

    Old version:
    {text1[:3000]}  # Limiting to 3000 characters to avoid token limits

    New version:
    {text2[:3000]}  # Limiting to 3000 characters to avoid token limits

    Provide the output in the following format:
    Additions: [List any added content]
    Removals: [List any removed content]
    Meaningful Changes: [List any meaningful changes]

    If there are no changes in a category, write "None".
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant specialized in comparing policy documents."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 500) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def compare_policy_documents(old_pdf_path: str, new_pdf_path: str) -> str:
    old_text = extract_text_from_pdf(old_pdf_path)
    new_text = extract_text_from_pdf(new_pdf_path)

    if not old_text or not new_text:
        return "Error: Could not extract text from one or both PDFs"

    old_sections = split_text_into_sections(old_text)
    new_sections = split_text_into_sections(new_text)

    output = ""

    if len(old_sections) > 0 and len(new_sections) > 0:
        logger.info("Comparing by sections")
        for i, (old, new) in enumerate(zip(old_sections, new_sections)):
            logger.info(f"Comparing section {i+1}")
            comparison = compare_texts(old[1], new[1])
            if comparison and "None" not in comparison:
                output += f"{old[0]}\n{comparison}\n\n"
    else:
        logger.info("Falling back to chunk comparison")
        old_chunks = chunk_text(old_text)
        new_chunks = chunk_text(new_text)
        for i, (old_chunk, new_chunk) in enumerate(zip(old_chunks, new_chunks)):
            logger.info(f"Comparing chunk {i+1}")
            comparison = compare_texts(old_chunk, new_chunk)
            if comparison and "None" not in comparison:
                output += f"Chunk {i+1}\n{comparison}\n\n"

    if not output:
        output = "No significant changes found or error occurred during comparison"

    return output

# Usage
old_pdf_path = "ekyc_2020_06.pdf"
new_pdf_path = "ekyc_2024_04.pdf"

logger.info("Starting document comparison...")
result = compare_policy_documents(old_pdf_path, new_pdf_path)
logger.info("Comparison result:")
print(result)