import re
import pymupdf  # PyMuPDF
# from sentence_transformers import SentenceTransformer, util #for semantic comparison

#For better semantic comparison, install sentence-transformers: pip install sentence-transformers

def extract_sections(pdf_path):
    section_number = 0
    doc = pymupdf.open(pdf_path)
    sections = {}
    current_section = None
    #previous_section = None  # Keep track of the previous section number
    full_stop = False
    stop_point = "8.1a6" #Set this for early stop for testing
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
                if section_number == stop_point:
                    print("Stopping at section " + stop_point)
                    full_stop = True
                    break

                # Update the sections dictionary
                if not section_number in sections:
                    sections[section_number] = {}
                    sections[section_number]['text'] = ""
                sections[section_number]['text'] += tidy_line(section_text)
                if section_type:
                    sections[section_number]['type'] = section_type
                else:
                    sections[section_number]['type'] = None
                #previous_section = section_number  # Update previous_section
                current_section = section_number  # Update current_section for subsequent lines
            elif current_section:
                # If there's no match but we have a current section, append the line to it
                sections[current_section]['text'] += " " + tidy_line(line)  # Append to the current section
        if full_stop:
            break
        #Footnote Handling
        htmltext = page.get_text("html")
        footnote_section = False
        footnotes_in_page = {} #The first time a small number is found, it's put into this dict as footnote_num : current_section; the second time is the actual footnote so it's left for later
        for line in htmltext.splitlines():
            #
            #Have to find current section again, a different way: checking the line starts left enough
            #
            marginmatch = re.match(r"^.*?left:(\d+)", line)
            if marginmatch:
                margin = int(marginmatch.group(1))
                if margin < 100:
                    trimline = re.sub(r"<.*?>","",line).strip() #Check the text of the line if it's far enough to the left
                    section_match = re.match(r"^.*?(\d+\.\d+)", trimline)
                    if section_match:
                        #Also check it isn't a footnote
                        sizematch = int(re.match(r"^.*?font-size:(\d+)", line).group(1))
                        if sizematch > 10:
                            current_section_html = section_match.group(1)
            #
            #Actual Footnote Check
            footmatch = re.match(r"^.*<span.*font-size:([\d\.]+).*?>(\d+)<\/span>(.*)", line) #I don't know why it needs the ^.*? in front but it does
            if footmatch:
                fontsize = float(footmatch.group(1))
                if fontsize < 9:
                    notenum = footmatch.group(2)
                    if not notenum in footnotes_in_page: #First appearance of small number; ie the indicator and not the actual footnote
                        footnotes_in_page[notenum] = current_section if (current_section.find("A")>-1) else current_section_html
                    else:
                        footnote_section = True #The first repeated small number shows the start of the footnotes section                    
                        foot_text = footmatch.group(3)
                        foot_text = re.sub(r"<.*?>","",foot_text)
                        linked_section = footnotes_in_page[notenum]
                        if not 'footnotes' in sections[linked_section]:
                            sections[linked_section]['footnotes'] = {}
                        print("FOOTNOTE " + notenum)
                        sections[linked_section]['footnotes'].update( { notenum: foot_text.strip() } )
            elif footnote_section:
                    foot_text = re.sub(r"<.*?>","",line)
                    sections[linked_section]['footnotes'][notenum] += " " + ''.join(foot_text.strip())

    return sections

def tidy_line(line):
    tidyline = line.strip()
    if re.match(r"\([ivx]{1,4}\)", tidyline):
        tidyline = "\n" + tidyline #Tries to put a linebreak in front of (i), (iv), etc - currently detects it properly but inserting \n does not work
    tidyline = re.sub(r"(\s+)"," ", tidyline)
    return tidyline
    


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

def test_extract_to_file(): #Currently broken, ignore
    to_write = test_print(pdf_1_path)
    with open("test_extract.txt", 'w', encoding="utf-8") as f:
        for i in to_write:
            f.write(f"{i}\n")
    return

def test_print(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = doc[7].get_text(sort=True)
    for line in text.splitlines():
        print(line)

def test_html(pdf_path, page):
    doc = pymupdf.open(pdf_path)
    htmlout = doc[page].get_text("html", sort=True)
    with open("test_htmlout.txt", 'w', encoding="utf-8") as f:
        for line in htmlout.splitlines():
            f.write(f"{line}\n")
    print(f"Wrote to test_htmlout.txt")

def test_html_clean(pdf_path, page):
    doc = pymupdf.open(pdf_path)
    htmlout = doc[page].get_text("html", sort=True)
    with open("test_htmlout2.txt", 'w', encoding="utf-8") as f:
        for line in htmlout.splitlines():
            hline = re.sub(r"<(?!b>)(?!\/b>).*?>","",line)
            f.write(f"{hline}\n\n")
    print(f"Wrote to test_htmlout2.txt")
# Example usage

#test_extract_to_file()

#sections1 = extract_sections(pdf_1_path )
#sections2 = extract_sections(pdf_2_path )

output_file = "test_output.txt"
save_sections_to_file(pdf_1_path, output_file)

#test_html(pdf_3_path, 7)
#test_html_clean(pdf_1_path, 8)
#test_print(pdf_3_path)