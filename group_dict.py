from text_sections_extract2 import extract_sections
from collections import defaultdict

# Original dictionary of sections and subsections
sections = extract_sections('ekyc_2024_04.pdf')

# Dictionary to hold grouped sections
grouped_sections = defaultdict(dict)

# Group subsections under their respective main sections
for subsection, content in sections.items():
    main_section = subsection.split('.')[0]  # Extract main section (e.g., '1' from '1.1')
    grouped_sections[main_section][subsection] = content

# Convert defaultdict back to regular dict (optional)
grouped_sections = dict(grouped_sections)

# # Output the grouped dictionary
# for section, subsections in grouped_sections.items():
#     print(f"Section {section}:")
#     for subsection, content in subsections.items():
#         print(f"  Subsection {subsection}: {content}")

        

def group_subsections(sections):
    """
    Groups subsections into their respective main sections.
    
    :param sections: A dictionary where keys are subsection numbers (e.g., '1.1') and values are dictionaries with content.
    :return: A dictionary where keys are main section numbers (e.g., '1') and values are dictionaries containing subsections.
    """
    grouped_sections = defaultdict(dict)

    # Group subsections under their respective main sections
    for subsection, content in sections.items():
        main_section = subsection.split('.')[0]  # Extract main section (e.g., '1' from '1.1')
        grouped_sections[main_section][subsection] = content

    # Convert defaultdict back to regular dict
    return dict(grouped_sections)

group_sections = group_subsections(sections)
# print(group_sections)
output_file = 'testcheck3.txt'

with open(output_file, 'w', encoding="utf-8") as f:
        for section, subsections in grouped_sections.items():
            f.write(f"Section: {section}\n")
            for subsection, content in subsections.items():
              f.write(f"Subsection: {subsection}\n")
              f.write(f"Type: {content['type']}\n")
              f.write(f"Text: { content['text'] }\n")
              if 'footnotes' in content:
                  for num, text in content['footnotes'].items():
                      f.write(f"Footnote {num}: {text}\n")
              f.write("\n")  # Add an empty line between sections
        print(f"Wrote to {output_file}")

