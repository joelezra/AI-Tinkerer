from pypdf import PdfReader
from difflib import HtmlDiff
from bs4 import BeautifulSoup

reader1 = PdfReader('ekyc_2020_06.pdf')
reader2 = PdfReader('ekyc_2024_04.pdf')

text1 = ''
text2 = ''

for page in reader1.pages:
  text1 += page.extract_text(extraction_mode='layout') + '\n'

for page in reader2.pages:
  text2 += page.extract_text(extraction_mode='layout') + '\n'

d = HtmlDiff()
html_diff = d.make_file(text1.splitlines(), text2.splitlines())
with open ('diff2.html', 'w', encoding="utf-8") as f:
  f.write(html_diff)

soup = BeautifulSoup(html_diff, 'html.parser')

# Initialize arrays for added, removed, and changed subsections
added_subsections = []
removed_subsections = []
changed_subsections = []

# Function to extract a full subsection given a starting span
def extract_full_subsection(span):
    subsection = span.find_parent('td').text.strip()
    return subsection

# Extract highlighted text and map them to their subsections
for span in soup.find_all('span'):
    style = span.get('class')
    
    # Check for added (green), removed (red), or changed (yellow) text
    if style == ['diff_add']:  # Added text (typically green)
        added_subsections.append(extract_full_subsection(span))
    elif style == ['diff_sub']:  # Removed text (typically red)
        removed_subsections.append(extract_full_subsection(span))
    elif style == ['diff_chg']:  # Changed text (typically yellow)
        changed_subsections.append(extract_full_subsection(span))

# Create a function to generate the HTML content
def generate_html_summary(added_subsections, removed_subsections, changed_subsections):
    # Start the HTML string
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Changes Summary</title>
        <style>
            body { font-family: Arial, sans-serif; }
            h2 { color: #333; }
            .added { color: green; }
            .removed { color: red; }
            .changed { color: orange; }
            ul { margin: 0; padding: 0; }
            li { margin: 5px 0; }
        </style>
    </head>
    <body>
        <h2>=== Changes Summary ===</h2>
    """

    # Added Text Section
    html_content += "<h3>Added Text:</h3><ul>"
    if added_subsections:
        for item in added_subsections:
            if item:
              html_content += f"<li class='added'>- {item}</li>"
    else:
        html_content += "<li>No text was added.</li>"
    html_content += "</ul>"

    # Removed Text Section
    html_content += "<h3>Removed Text:</h3><ul>"
    if removed_subsections:
        for item in removed_subsections:
            if item:
              html_content += f"<li class='removed'>- {item}</li>"
    else:
        html_content += "<li>No text was removed.</li>"
    html_content += "</ul>"

    # Changed Text Section
    html_content += "<h3>Changed Text:</h3><ul>"
    if changed_subsections:
        for item in changed_subsections:
            if item:
              html_content += f"<li class='changed'>- {item}</li>"
    else:
        html_content += "<li>No text was changed.</li>"
    html_content += "</ul>"

    # Close the HTML tags
    html_content += """
    </body>
    </html>
    """

    return html_content

# Use the function to generate the HTML summary
html_summary = generate_html_summary(added_subsections, removed_subsections, changed_subsections)

# Write the HTML to a file
with open('changes_summary2.html', 'w', encoding='utf-8') as f:
    f.write(html_summary)

print("HTML summary has been written to 'changes_summary.html'.")

