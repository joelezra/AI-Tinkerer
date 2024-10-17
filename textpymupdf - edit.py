import re
import pymupdf  # PyMuPDF
# from sentence_transformers import SentenceTransformer, util #for semantic comparison

#For better semantic comparison, install sentence-transformers: pip install sentence-transformers

def test_print(pdf_path):
    try:
        alltext = []
        doc = pymupdf.open(pdf_path)
        for page in doc:
            text = page.get_text(sort=True)
            for line in text.splitlines():
                #line = line.strip()
                alltext.append(line)
        return alltext
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

def test_html(pdf_path):
    doc = pymupdf.open(pdf_path)
    htmlout = doc[3].get_text("html", sort=True)
    print(htmlout)

def extract_sections(pdf_path):

    doc = pymupdf.open(pdf_path)
    sections = {}
    current_section = None
    previous_section = None  # Keep track of the previous section number
    full_stop = False
    in_appendix = False
    for page in doc:
        text = page.get_text(sort=True)
        for line in text.splitlines():
            # Improved regex to match section numbers with leading letters and spaces
            match = re.match(r"^(([SG]\s\s)?(\d+\.\d+))(.*)$", line)
            """
            Groups: 1 - S 8.9
                    2 - S
                    3 - 8.9
                    4-  Text (Until end of line)
            """
            #Also check Appendix match
            if not match:
                match = re.match(r"^Appendix\s+(\d+):(.*)$", line)
                if match:
                    in_appendix = True
            if match:
                if not in_appendix:
                    section_number = match.group(3).strip()
                    if match.group(2) != None:
                        section_type = match.group(2).strip()
                    else:
                        section_type = None
                    section_text = match.group(4)  # Capture the text after the section number
                else:
                    section_number= "Appendix " + match.group(1)
                    section_type = None
                    section_text = match.group(2)  # Appendix name
                print(section_number)
                stop_point = "aaa"
                if section_number == stop_point:
                    print("Stopping at section " + stop_point)
                    full_stop = True
                    break
                # # Check if the current section is in ascending order
                # if previous_section and not is_section_in_order(previous_section, section_number):
                #     print(f"Warning: Section {section_number} is out of order after {previous_section}. Skipping.")
                #     ## code to concatenate section text with previous section text
                #     sections[previous_section]['text']  += " " + section_number + " " + tidy_line(section_text)
                #     continue  # Skip this section if it is out of order

                # Update the sections dictionary
                if not section_number in sections:
                    sections[section_number] = {}
                    sections[section_number]['text'] = []
                sections[section_number]['text'] += tidy_line(section_text)
                if section_type:
                    sections[section_number]['type'] = section_type
                else:
                    sections[section_number]['type'] = None
                previous_section = section_number  # Update previous_section
                current_section = section_number  # Update current_section for subsequent lines
            elif current_section:
                # If there's no match but we have a current section, append the line to it
                sections[current_section]['text'] += " " + tidy_line(line)  # Append to the current section
        if full_stop:
            break
    return sections

def tidy_line(line):
    tidyline = line.strip()
    if re.match(r"\([ivx]{1,4}\)", tidyline):
        tidyline = "\n" + tidyline #Tries to put a linebreak in front of (i), (iv), etc - currently detects it properly but inserting \n does not work
    tidyline = re.sub(r"(\s+)"," ", tidyline)
    return tidyline
    
# function to ensure that the section numbers are in ascending order
# meaning if current is 1.3 then the next section number should be either 1.4 or 2.1
def is_section_in_order(prev_section, curr_section):
    """Check if the current section number is in ascending order compared to the previous one."""
    try:
        prev_parts = [int(x) for x in prev_section.split('.')]  # Convert to integer list like [1, 3]
        curr_parts = [int(x) for x in curr_section.split('.')]  # Convert to integer list like [1, 4] or [2, 1]

        # Case 1: Same major section (e.g., 1.3 to 1.4)
        if curr_parts[0] == prev_parts[0] and curr_parts[1] == prev_parts[1] + 1:
            return True  # Valid next minor section

        # Case 2: Next major section (e.g., 1.3 to 2.1)
        elif curr_parts[0] == prev_parts[0] + 1 and curr_parts[1] == 1:
            return True  # Valid start of next major section

        # Otherwise, the section is out of order
        return False

    except (ValueError, IndexError):
        return False  # Return False if parsing fails (e.g., malformed section number)


# Example usage:
pdf_1_path = "ekyc_2024_04.pdf"  # Replace with your PDF file paths
pdf_2_path = "ekyc_2020_06.pdf"
pdf_3_path = ""
pdf_4_path = ""

def save_sections_to_file(sections, output_file):
    """Saves the extracted sections to a text file."""
    with open(output_file, 'w', encoding="utf-8") as f:
        for section_number, section_contents in sections.items():
            f.write(f"Section: {section_number}\n")
            f.write(f"Type: {section_contents['type']}\n")
            f.write(f"Text: {"".join(section_contents['text'])}\n")
            f.write("\n")  # Add an empty line between sections
    print(f"Wrote to {output_file}")

def test_extract_to_file():
    to_write = test_print(pdf_1_path)
    with open("test_extract.txt", 'w', encoding="utf-8") as f:
        for i in to_write:
            f.write(f"{i}\n")
    return
# Example usage

#test_extract_to_file()
#test_html(pdf_1_path)

sections1 = extract_sections(pdf_1_path )
#sections2 = extract_sections(pdf_2_path )

output_file2 = "test_output.txt"
save_sections_to_file(sections1, output_file2)
