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



# for i in range(len(reader1.pages)):
#   page1 = reader1.pages[i]
#   text1 = page1.extract_text(extraction_mode='layout')
#   # print(f"Page {i+1}: \n{text1}\n")

# for i in range(len(reader2.pages)):
#   page2 = reader2.pages[i]
#   text2 = page2.extract_text(extraction_mode='layout')
#   # print(f"Page {i+1}: \n{text2}\n")

d = HtmlDiff()
html_diff = d.make_file(text1.splitlines(), text2.splitlines())
with open ('diff2.html', 'w', encoding="utf-8") as f:
  f.write(html_diff)

soup = BeautifulSoup(html_diff, 'html.parser')

# Initialize arrays for added, removed, and changed text
added = []
removed = []
changed = []

# Extract green-highlighted text for additions, red for removals, and yellow for changes
for span in soup.find_all('span'):
    style = span.get('class')
    
    if style == ['diff_add']:  # Added (usually green)
        added.append(span.text.strip())
    elif style == ['diff_sub']:  # Removed (usually red)
        removed.append(span.text.strip())
    elif style == ['diff_chg']:  # Changed (usually yellow)
        changed.append(span.text.strip())

# # Output the extracted arrays
# print("Added:", added)
# print("Removed:", removed)
# print("Changed:", changed)

# Output the extracted arrays in a readable format
print("=== Changes Summary ===\n")

if added:
    print("Added Text:")
    for item in added:
        print(f"- {item}")
else:
    print("No text was added.")

print("\nRemoved Text:")
if removed:
    for item in removed:
        print(f"- {item}")
else:
    print("No text was removed.")

print("\nChanged Text:")
if changed:
    for item in changed:
        print(f"- {item}")
else:
    print("No text was changed.")