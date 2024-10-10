import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util #for semantic comparison

#For better semantic comparison, install sentence-transformers: pip install sentence-transformers

def extract_sections(pdf_path):
    """Extracts text sections from a PDF based on subsection numbering."""
    try:
        doc = fitz.open(pdf_path)
        sections = {}
        current_section = None
        for page in doc:
            text = page.get_text()
            for line in text.splitlines():
                match = re.match(r"(\d+\.\d+)\s+(.*)", line)  # Matches subsection numbers like "1.1"
                if match:
                    section_number = match.group(1)
                    section_text = match.group(2)
                    current_section = section_number
                    sections[current_section] = section_text.strip() #strip whitespace
                elif current_section:
                    sections[current_section] += " " + line.strip()  # Append to the current section
        return sections
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def map_sections(sections1, sections2):
    """Maps corresponding sections between two PDFs using fuzzy matching."""
    mapping = {}
    #Using fuzzy matching for more robust section mapping
    for section1_num in sections1:
        best_match = None
        best_similarity = 0
        for section2_num in sections2:
            similarity = calculate_similarity(section1_num,section2_num) #Custom function to compare section numbers
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = section2_num
        if best_similarity > 0.8: # Adjust threshold as needed
            mapping[section1_num] = best_match
    return mapping

def calculate_similarity(num1, num2):
    """Calculates similarity between section numbers (fuzzy matching)."""
    try:
        parts1 = [int(x) for x in num1.split('.')]
        parts2 = [int(x) for x in num2.split('.')]
        #Simple fuzzy matching: checks if the first parts match and the second part is close
        if parts1[0] == parts2[0] and abs(parts1[1]-parts2[1]) <= 1:
            return 0.9
        elif parts1[0] == parts2[0]:
            return 0.7
        else:
            return 0
    except (ValueError, IndexError):
        return 0

def compare_sections(sections1, sections2, mapping, model):
    """Compares corresponding sections using semantic similarity."""
    results = {}
    for section1_num, section2_num in mapping.items():
        section1_text = sections1.get(section1_num, "")
        section2_text = sections2.get(section2_num, "")

        embeddings1 = model.encode(section1_text, convert_to_tensor=True)
        embeddings2 = model.encode(section2_text, convert_to_tensor=True)
        cosine_sim = util.cos_sim(embeddings1, embeddings2)
        similarity_score = cosine_sim.item()

        if similarity_score > 0.8: # Adjust threshold as needed
            results[section1_num] = {"change": "no change", "explanation": f"Sections are semantically similar (cosine similarity: {similarity_score:.2f})"}
        else:
            results[section1_num] = {"change": "semantic change", "explanation": f"Sections differ semantically (cosine similarity: {similarity_score:.2f})"}
    return results


# Example usage:
pdf_1_path = "ekyc_2024_04.pdf"  # Replace with your PDF file paths
pdf_2_path = "ekyc_2020_06.pdf"

#Load Sentence Transformer model (all-mpnet-base-v2 is a good option)
model = SentenceTransformer('all-mpnet-base-v2')

sections1 = extract_sections(pdf_1_path)
sections2 = extract_sections(pdf_2_path)

mapping = map_sections(sections1, sections2)

comparison_results = compare_sections(sections1, sections2, mapping, model)

# print(comparison_results)

def save_results_to_file(comparison_results, output_file):
    """Saves the comparison results to a text file."""
    with open(output_file, 'w') as f:
        for section, result in comparison_results.items():
            f.write(f"Section: {section}\n")
            f.write(f"Change: {result['change']}\n")
            f.write(f"Explanation: {result['explanation']}\n")
            f.write("\n")

# Example usage to save results
output_file = "comparison_results.txt"
save_results_to_file(comparison_results, output_file)

