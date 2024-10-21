import re
import pymupdf
import html
# from sentence_transformers import SentenceTransformer, util #for semantic comparison

#For better semantic comparison, install sentence-transformers: pip install sentence-transformers

def extract_sections(pdf_path):
    section_number = 0
    doc = pymupdf.open(pdf_path)
    sections = {}
    current_section = None
    current_appendix = None
    appendix_letter_part = None
    appendix_num_part = None
    #Get document title to filter it from future pages; this is grabbed by looking at the first line on the 2nd page
    global doc_title
    page_1 = doc[1].get_text(sort=True)
    for line in page_1.splitlines():
        if line.strip() != "":
            doc_title = re.sub(r"(\s{5,}).*","", line).strip()
            break
    print("Doc title: " + doc_title)
    in_appendix = False
    section_type_hold = None #Used to remember whether an S or G was mentioned before a section

    for page in doc:
        page_text = page.get_text("dict", sort=True)
        footnote_section = False
        footnotes_in_page = {} #The first time a small number is found, it's put into this dict as footnote_num : current_section; the second time is the actual footnote so it's left for later

        for block in page_text['blocks']:
            if not 'lines' in block: #Ignore image blocks
                continue
            for blockpart in block['lines']:
                for span in blockpart['spans']:
                    text = span['text']
                    #Check for Bold S or G
                    match = re.match(r"^\s*([SG])\s*$", text)
                    if match and span['flags'] & 2**4: #uses bitwise & operator to tell if the flags contain Bold (2**4) 
                        section_type_hold = match.group(1).strip()
                    #Check for subsection number
                    match = re.match(r"^\s*(\d+\.\d+)\s*(.*)$", text)
                    if match and span['origin'][0] < 80: #Section number match; 80 is the margin cutoff
                        section_number = match.group(1).strip()
                        if not section_number in sections: #Update the sections dictionary
                            sections[section_number] = {}
                            if match.group(2):
                                add_text = tidy_line(match.group(2))
                            else:
                                add_text = ""
                            sections[section_number]['text'] = add_text
                            sections[section_number]['type'] = section_type_hold
                            print("Section " + section_number)
                        section_type_hold = None
                        continue
                    #Check for Appendix
                    match = re.match(r"^\s*Appendix\s+(\d+):?(.*)$", text)
                    if not span['flags'] & 2**4: #The bold check is needed otherwise it might pick up the Contents page
                        match = None
                    if match: 
                        in_appendix = True
                        current_appendix = match.group(1)
                        appendix_letter_part = "0"
                        appendix_num_part = "0"
                    elif in_appendix: #Check for part starting
                        match = re.match(r"^\s{0,2}P[Aa][Rr][Tt]\s+(\w)(.*)$", text) #Allows up to 2 spaces before the number
                        if match:
                            appendix_letter_part = match.group(1)
                            appendix_num_part = "0"
                        else:
                            match = re.match(r"^\s{0,2}?(\d+)\.\s+(.*)$", text) #Allows up to 2 spaces before the number
                            if not span['origin'][0] < 80:
                                match = None
                            if match:
                                appendix_num_part = match.group(1)
                    if match: #Update appendix number
                        section_number= "Appendix " + current_appendix + "." + appendix_letter_part + "." + appendix_num_part
                        if not section_number in sections: #Update the sections dictionary
                            sections[section_number] = {}
                            sections[section_number]['text'] = ""
                            sections[section_number]['type'] = None
                            print("Section " + section_number)
                    #Check for footnote number
                    footmatch = re.match(r"^\s*(\d+)\s*$", text)
                    if footmatch and span['size'] < 9: #Footnote number match; set this 9 to a settings variable
                        notenum = footmatch.group(1)
                        if not notenum in footnotes_in_page: #First appearance of small number; ie the indicator and not the actual footnote
                            footnotes_in_page[notenum] = section_number
                            sections[section_number]['text'] += "(" + notenum + ")"
                        else: #Actual footnote
                            footnote_section = True
                            linked_section = footnotes_in_page[notenum]
                            if not 'footnotes' in sections[linked_section]: #If this is the first footnote for the section, create 'footnotes' key
                                sections[linked_section]['footnotes'] = {}
                            sections[linked_section]['footnotes'].update( { notenum: "" } )
                            print("Footnote " + notenum)
                        continue
                    #Normal text, add it to the section
                    if not match and section_number: 
                        if section_type_hold: #This skips the adding if we're in the space between S/G and the section number
                            continue
                        add_text = tidy_line(text)
                                
                        if add_text != "":
                                add_text = " " + add_text
                        if not footnote_section: #This might cause errors if there are any footnotes before the very first section
                            sections[section_number]['text'] += add_text
                        else: 
                            sections[linked_section]['footnotes'][notenum] += add_text

    return sections

#Trims lines
def tidy_line(line):
    tidyline = line.strip()
    #if re.match(r"\([ivx]{1,4}\)", tidyline):
    #    tidyline = "\n" + tidyline #Tries to put a linebreak in front of (i), (iv), etc - currently detects it properly but inserting \n does not work
    #Remove spaces
    tidyline = re.sub(r"(\s+)"," ", tidyline)
    #We see whether the line is blank after removing page number and title; if it's blank then delete the whole line
    #Remove 'Page x of y' or 'x of y'
    maybeline = re.sub(r"([Pp]age)?\s+\d+ of \d+", "", tidyline)
    #Remove the doc title
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
pdf_1_path = "sample/ekyc_2024_04.pdf"  # Replace with your PDF file paths
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
            if 'footnotes' in section_contents:
                for num, text in section_contents['footnotes'].items():
                    f.write(f"Footnote {num}: {text}\n")
            f.write("\n")  # Add an empty line between sections
    print(f"Wrote to {output_file}")

def test_dict(pdf_path, output_file, page, mode="default"):
    doc = pymupdf.open(pdf_path)
    dictout = doc[page].get_text("dict", sort=True)
    #print(dictout)
    with open(output_file, 'w', encoding="utf-8") as f:
        for block in dictout['blocks']:
            for blockpart in block['lines']:
                for span in blockpart['spans']:
                    if mode=="txtonly":
                        f.write(f"{span['text']}\n")
                    else:
                        f.write(f"{span}\n")
    print(f"Wrote to {output_file}")


output_file = "test_output2.txt"

mode=1
if mode == 1:
    save_sections_to_file(pdf_2_path, output_file)
else:
    output_file = "test_dictout3.py"
    test_dict(pdf_2_path, output_file, 10, "txtonsly")

#test_extract_to_file(pdf_4_path,"pdf4raw")

def test_table_extract(pdf_path,page):
    doc = pymupdf.open(pdf_path)
    x = doc[page].find_tables().tables
    if x != []:
        x = x[0]
    print(x.extract())

#test_table_extract(pdf_1_path,21)
