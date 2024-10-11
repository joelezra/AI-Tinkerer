import networkx as nx
import matplotlib.pyplot as plt
from textpymupdf import extract_sections, map_sections, compare_sections  # Assuming you're using sections from textpymupdf
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model (all-mpnet-base-v2 is a good option)
model = SentenceTransformer('all-mpnet-base-v2')

# Example usage
pdf_1_path = "ekyc_2024_04.pdf"  # Replace with your PDF file paths
pdf_2_path = "ekyc_2020_06.pdf"

sections1 = extract_sections(pdf_1_path)
sections2 = extract_sections(pdf_2_path)

mapping = map_sections(sections1, sections2)
comparison_results = compare_sections(sections1, sections2, mapping, model)

# Create a graph
G = nx.Graph()

# Add nodes for sections in both PDFs
for section in sections1:
    G.add_node(f"PDF1_{section}", label=section)

for section in sections2:
    G.add_node(f"PDF2_{section}", label=section)

# Add edges based on section mappings and comparison results
for section1, section2 in mapping.items():
    similarity = comparison_results[section1]['change']
    if similarity == 'no change':
        edge_color = 'green'
    elif similarity == 'semantic change':
        edge_color = 'red'
    else:
        edge_color = 'gray'

    G.add_edge(f"PDF1_{section1}", f"PDF2_{section2}", color=edge_color)

# Draw the graph
colors = [G[u][v]['color'] for u, v in G.edges()]
pos = nx.spring_layout(G)  # Layout for a better visual representation
nx.draw(G, pos, edge_color=colors, with_labels=True, node_size=3000, font_size=10, font_weight='bold', node_color='lightblue')

# Show the plot
plt.show()

