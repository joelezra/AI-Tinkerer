import re
import pymupdf  # PyMuPDF
import html
# from sentence_transformers import SentenceTransformer, util #for semantic comparison

#For better semantic comparison, install sentence-transformers: pip install sentence-transformers

def extract_sections(pdf_path):
    section_number = 0
    doc = pymupdf.open(pdf_path)
    sections = {}
    current_section = None
    # Get document title to filter it from future pages; this is grabbed by looking at the first line on the 2nd page
    global doc_title
    page_1 = doc[1].get_text(sort=True)
    for line in page_1.splitlines():
        if line.strip() != "":
            doc_title = re.sub(r"(\s{5,}).*", "", line).strip()
            break
    # print("Doc title: " + doc_title)

    stop_point = "none"  # Set this to a section name for early stop for testing
    full_stop = False  # Don't touch, this is for the above
    in_appendix = False

    for page in doc:
        text = page.get_text(sort=True)
        for line in text.splitlines():
            # Improved regex to match section numbers with leading letters and spaces
            match = re.match(r"^(([SG]\s+)?(\d+\.\d+))(.*)$", line)
            """
            Groups: 1 - S 8.9
                    2 - S
                    3 - 8.9
                    4 - Text (Until end of line)
            """
            # Also check Appendix match
            if not match:
                match = re.match(r"^Appendix\s+(\d+):(.*)$", line)
                if match:
                    in_appendix = True

            if match:
                if not in_appendix:
                    section_number = match.group(3).strip()
                    section_text = section_number + ':' + match.group(4)  # Capture the text after the section number
                    section_type = match.group(2).strip() if match.group(2) else None
                else:
                    section_number = "Appendix " + match.group(1)
                    section_text = section_number + ':' + match.group(2)  # Appendix name
                    section_type = None

                # print(section_number)
                if section_number == stop_point:
                    print("Stopping at section " + stop_point)
                    full_stop = True
                    break

                # Initialize section in dictionary
                if section_number not in sections:
                    sections[section_number] = {
                        'text': "",
                        'type': section_type,
                        'footnotes': {}  # Initialize footnotes as an empty dictionary
                    }
                # Append the text for the section
                sections[section_number]['text'] += tidy_line(section_text)
                sections[section_number]['type'] = section_type or None

                current_section = section_number  # Update current_section for subsequent lines

            elif current_section:
                # If there's no match but we have a current section, append the line to it
                sections[current_section]['text'] += " " + tidy_line(line)  # Append to the current section

        if full_stop:
            break

        # Footnote Handling
        htmltext = page.get_text("html")
        footnote_section = None
        footnotes_in_page = {}

        for line in htmltext.splitlines():
            # Find current section again by checking the line starts left enough
            marginmatch = re.match(r"^.*?left:(\d+)", line)
            if marginmatch:
                margin = int(marginmatch.group(1))
                if margin < 100:
                    trimline = re.sub(r"<.*?>", "", line).strip()
                    section_match = re.match(r"^.*?(\d+\.\d+)", trimline)
                    if section_match:
                        # Also check it isn't a footnote
                        sizematch = int(re.match(r"^.*?font-size:(\d+)", line).group(1))
                        if sizematch > 10:
                            current_section_html = section_match.group(1)

            # Actual Footnote Check
            footmatch = re.match(r"^.*<span.*font-size:([\d\.]+).*?>(\d+)<\/span>(.*)", line)
            if footmatch:
                fontsize = float(footmatch.group(1))
                if fontsize < 9:
                    notenum = footmatch.group(2)
                    if not notenum in footnotes_in_page:
                        # First appearance of small number (footnote indicator)
                        footnotes_in_page[notenum] = current_section if "A" in current_section else current_section_html
                    else:
                        # Actual footnote text
                        foot_text = re.sub(r"<.*?>", "", footmatch.group(3))
                        linked_section = footnotes_in_page[notenum]
                        if sections[linked_section]['footnotes'] is None:
                            sections[linked_section]['footnotes'] = {}

                        # print("FOOTNOTE " + notenum)
                        foot_text = tidy_html(foot_text)

                        if not footnote_section:  # The first time footnotes are found on the page
                            footnote_start_regex = notenum + r"\s+" + foot_text[:10] + ".*"
                            sections[current_section]['text'] = re.sub(footnote_start_regex, "", sections[current_section]['text'])

                        sections[linked_section]['footnotes'].update({notenum: foot_text})
                        footnote_section = True  # Start of the footnotes section

            elif footnote_section:
                foot_text = tidy_html(re.sub(r"<.*?>", "", line))
                sections[linked_section]['footnotes'][notenum] += " " + ''.join(foot_text)

    return sections

#Trims lines
def tidy_line(line):
    tidyline = line.strip()
    if re.match(r"\([ivx]{1,4}\)", tidyline):
        tidyline = "\n" + tidyline #Tries to put a linebreak in front of (i), (iv), etc - currently detects it properly but inserting \n does not work
    #Remove spaces
    tidyline = re.sub(r"(\s+)"," ", tidyline)
    #We see whether the line is blank after removing page number and title; if it's blank then delete the whole line
    #Remove 'Page x of y' or 'x of y'
    maybeline = re.sub(r"([Pp]age)?\s+\d+ of \d+", "", tidyline)
    #Remove line if it is only the doc title
    maybeline = maybeline.replace(doc_title,"")
    maybeline = re.sub(r"Issued on:\s+\d+\s+\w+\s+\d+","", maybeline)
    if maybeline.strip() == "":
        tidyline = ""
    return tidyline

#Unescapes html entities like &#x2019; to convert them to normal
def tidy_html(line):
    tidyhtml = line.strip()
    entities = re.findall(r'&#[xX]?[0-9a-fA-F]+;', tidyhtml)
    # Replace each entity with its corresponding symbol
    for entity in entities:
        symbol = html.unescape(entity)
        tidyhtml = tidyhtml.replace(entity, symbol)
    return tidyhtml

def get_title(page):
    return


# Example usage:
pdf_1_path = "ekyc_2024_04.pdf"  # Replace with your PDF file paths
pdf_2_path = "sample/ekyc_2020_06.pdf"
pdf_3_path = "sample/rmit_2023_06.pdf"
pdf_4_path = "sample/rmit_2020_06.pdf"

def save_sections_to_file(pdf, output_file):
    """Saves the extracted sections to a text file."""
    sections = extract_sections(pdf)
    with open(output_file, 'w', encoding="utf-8") as f:
        for section_number, section_contents in sections.items():
            f.write(f"Section: {section_number}\n")
            f.write(f"Type: {section_contents['type']}\n")
            f.write(f"Text: { section_contents['text'] }\n")
            # if 'footnotes' in section_contents:
            # Ensure footnotes are present for every section, even if empty
            footnotes = section_contents.get('footnotes', {})
            if not footnotes:
                f.write("Footnotes: None\n")
            else:
                for num, text in footnotes.items():
                    f.write(f"Footnote {num}: {text}\n")
                    
            f.write("\n")  # Add an empty line between sections
    print(f"Wrote to {output_file}")

# def test_extract_to_file(pdf_path,output='test_extract'):
#     to_write = test_print_all(pdf_path)
#     with open(output+".txt", 'w', encoding="utf-8") as f:
#         for i in to_write:
#             f.write(f"{i}\n")
#     print(f"Wrote {output}.txt")
#     return

# def test_print_1pg(pdf_path):
#     doc = pymupdf.open(pdf_path)
#     text = doc[1].get_text(sort=True)
#     lines = []
#     for line in text.splitlines():
#         lines.append(line)
#     return lines

# def test_print_all(pdf_path):
#     doc = pymupdf.open(pdf_path)
#     lines = []
#     for page in doc:
#         text = page.get_text(sort=True)
#         for line in text.splitlines():
#             lines.append(line)
#     return lines

# def test_html(pdf_path, page):
#     doc = pymupdf.open(pdf_path)
#     htmlout = doc[page].get_text("html", sort=True)
#     with open("test_htmlout.txt", 'w', encoding="utf-8") as f:
#         for line in htmlout.splitlines():
#             f.write(f"{line}\n")
#     print(f"Wrote to test_htmlout.txt")

# def test_html_clean(pdf_path, page):
#     doc = pymupdf.open(pdf_path)
#     htmlout = doc[page].get_text("html", sort=True)
#     with open("test_htmlout2.txt", 'w', encoding="utf-8") as f:
#         for line in htmlout.splitlines():
#             hline = re.sub(r"<(?!b>)(?!\/b>).*?>","",line)
#             f.write(f"{hline}\n\n")
#     print(f"Wrote to test_htmlout2.txt")
# Example usage

#sections1 = extract_sections(pdf_1_path )
#sections2 = extract_sections(pdf_2_path )

output_file = "test_output.txt"
save_sections_to_file(pdf_1_path, output_file)

#test_html(pdf_3_path, 7)
#test_html_clean(pdf_1_path, 8)
# test_extract_to_file(pdf_1_path,"pdf4raw")
extract_sections(pdf_1_path)
# save_sections_to_file(pdf_1_path, 'GAIA2.txt')
