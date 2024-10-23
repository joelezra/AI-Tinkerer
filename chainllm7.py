import os
from pinecone import Pinecone, pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
import openai
from dotenv import load_dotenv
import asyncio
from text_sections_extract2 import extract_sections
from group_dict import group_subsections
from pydantic import BaseModel, ValidationError
import re
import logging

# Load environment variables
load_dotenv()

# Set up logging for error tracking
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Set up OpenAI API key
client = AsyncOpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY") or 'PINECONE_API_KEY'
)
index_name='pdf-gaia-test'

index = pc.Index(index_name, 'https://pdf-gaia-test-f3fg8ao.svc.aped-4627-b74a.pinecone.io')

parser = StrOutputParser()
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()

class SubsectionComparison(BaseModel):
    subsection: str  # Subsection number
    meaningful_changes_comparison: Optional[str]  # LLM output

class SectionComparison(BaseModel):
    section: str  # Section number
    comparisons: List[SubsectionComparison]  # List of comparisons for subsections

class DocumentComparisonResult(BaseModel):
    document_title: str
    sections: List[SectionComparison]  # List of section comparisons

# Define a Pydantic model to structure the response
class LLMResponseModel(BaseModel):
    section_number: str
    meaningful_changes: list[str]

# Extract text from new policy in subsection dictionary
pdf_sections1 = extract_sections("ekyc_2020_06.pdf")
pdf_sections2 = extract_sections("ekyc_2024_04.pdf")

# Group subsection dict into section dict
group_sections1 = group_subsections(pdf_sections1) # Old policy
group_sections2 = group_subsections(pdf_sections2) # revised policy

def extract_sections_and_chunk(policy_dict, chunk_size=1500, overlap=100):
    section_chunks = {}
    
    # Iterate through each section in the policy documents
    for section, subsections in policy_dict.items():
        section_chunks[section] = {}  # Initialize dictionary for each section
        
        combined_text = ""  # To store concatenated text from all subsections
        
        subsection_boundaries = []  # List to keep track of where subsections start and end in the combined text
        
        # Concatenate subsections and keep track of where they start and end in the combined text
        current_position = 0
        for subsection, content in subsections.items():
            text = content['text']
            combined_text += text
            subsection_boundaries.append({
                'start': current_position,
                'end': current_position + len(text),
                'subsection': subsection,
                'type': content['type'],
                'footnotes': content['footnotes']
            })
            current_position += len(text)
        
        # Split the combined text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = text_splitter.split_text(combined_text)
        
        # Store the chunks and assign the correct metadata based on subsection boundaries
        for i, chunk in enumerate(chunks):
            chunk_start = combined_text.find(chunk)
            chunk_end = chunk_start + len(chunk)
            # Find the subsections that this chunk overlaps with
            relevant_subsections = []
            for boundary in subsection_boundaries:
                # Check if chunk falls within or overlaps this boundary
                if (boundary['start'] <= chunk_start < boundary['end']) or (boundary['start'] < chunk_end <= boundary['end']) or (chunk_start <= boundary['start'] and chunk_end >= boundary['end']):
                    relevant_subsections.append({
                        'subsection': boundary['subsection'],
                        'type': boundary['type'],
                        'footnotes': boundary['footnotes']
                    })
            
            # Store the chunk with its relevant subsection metadata
            section_chunks[section][i + 1] = {
                'text': chunk,
                'subsections': relevant_subsections  # List of subsections the chunk spans
            }
    
    return section_chunks

section_chunked_old = extract_sections_and_chunk(group_sections1)
section_chunked_new = extract_sections_and_chunk(group_sections2)

# Function to get embeddings asynchronously from OpenAI
async def get_embedding_async(text):
    try:
        embedding_response = await client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        # Extract the first embedding from the response
        embedding = embedding_response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise
    
# Utility function to flatten subsections metadata into a list of strings or single string
def flatten_subsection_metadata(subsections):
    flattened = []
    for subsection in subsections:
        # Combine subsection, type, and footnotes into a readable string
        subsection_str = f"Subsection: {subsection['subsection']}, Type: {subsection['type']}, Footnotes: {subsection['footnotes']}"
        flattened.append(subsection_str)
    return flattened

# Utility function to flatten subsections metadata into individual fields
def extract_subsection_fields(subsections):
    types = []
    footnotes = []
    
    for subsection in subsections:
        # Add the type and footnotes to their respective lists
        types.append(str(subsection['type']))
        footnotes.append(str(subsection['footnotes']) if subsection['footnotes'] else "")
    
    return types, footnotes

# Function to store a single chunk in Pinecone asynchronously
async def store_chunk_in_pinecone(index, chunk_id, embedding, namespace, chunk_data_to_store):
    await index.upsert(
        vectors=[(chunk_id, embedding, chunk_data_to_store)],  # Pass metadata with the vector
        namespace=namespace
    )

# Function to process and store all chunks asynchronously
async def store_chunks_in_pinecone(index, section_chunks, namespace):
    tasks = []
    
    for section, chunks in section_chunks.items():
        print(f"Processing section: {section}")
        for chunk_number, chunk_data in chunks.items():
            print(f"Processing chunk {chunk_number} in section {section}")
            chunk_id = f"{section}_{chunk_number}"  # Unique ID for each chunk

            # Get embedding asynchronously
            embedding = await get_embedding_async(chunk_data['text'])
            
            # Extract types and footnotes as separate fields
            types, footnotes = extract_subsection_fields(chunk_data['subsections'])
            
            # Prepare the metadata and vector
            chunk_data_to_store = {
                "text": chunk_data['text'],        # Keep text as is
                "subsections": [sub["subsection"] for sub in chunk_data['subsections']],  # Only store subsection titles
                "types": types,                    # Store types as a separate field
                "footnotes": footnotes             # Store footnotes as a separate field
            }
            
            # Add async task to store this chunk
            tasks.append(store_chunk_in_pinecone(index, chunk_id, embedding, namespace, chunk_data_to_store))
    
    # Run all storage tasks concurrently
    await asyncio.gather(*tasks)

# Function to retrieve all chunks from Pinecone (old)
def retrieve_all_chunks(index, namespace, subsections=None):
    # If you want to retrieve specific sections, you can pass the section filter
    query_response = index.query(
        vector=None,  # We don't need a specific vector, we retrieve all with metadata
        namespace=namespace,
        filter={"subsection": subsections} if subsections else None,
        top_k=1000  # Large number to retrieve all chunks
    )
    
    return query_response['matches']

# Function to query Pinecone and get the top similar chunks from the new document namespace
async def find_similar_chunks(index, new_chunk_vector, namespace, top_k=1):
    query_response = await index.query(
        vector=new_chunk_vector,  # Query using the old chunk embedding vector
        namespace=namespace,  # Specify the namespace for the new document embeddings
        top_k=top_k,  # Retrieve the top_k most similar chunks
        include_metadata=True  # Include metadata in the response
    )
    
    return query_response['matches']

# Function to match corresponding old and new chunks using vector similarity across namespaces
async def match_chunks_using_similarity(index, old_chunks, new_doc_namespace, top_k=1):
    matched_chunks = []
    
    for old_chunk in old_chunks:
        old_vector = old_chunk['values']  # Use the embedding vector for the old chunk
        
        # Find the top similar chunk from the new document using vector similarity
        similar_chunks = await find_similar_chunks(index, old_vector, namespace=new_doc_namespace, top_k=top_k)
        
        for similar_chunk in similar_chunks:
            matched_chunks.append({
                'old_chunk': old_chunk['metadata']['text'],  # Old chunk text
                'new_chunk': similar_chunk['metadata']['text'],  # New chunk text
                'old_metadata': old_chunk['metadata'],  # Metadata for old chunk
                'new_metadata': similar_chunk['metadata'],  # Metadata for the matched chunk
                'similarity_score': similar_chunk['score']  # Similarity score from the Pinecone query
            })
    
    return matched_chunks

# Function to compare old and new chunks using OpenAI LLM
async def compare_chunks_with_llm(old_chunk, new_chunk, subsection):
    # LLM prompt to focus on meaningful changes requiring internal policy updates
    prompt = f"Compare the following subsections and list only the meaningful changes that would require internal policy updates:\n\nOld Subsection:\n{old_chunk}\n\nNew Subsection:\n{new_chunk}"
    
    # Query OpenAI for comparison analysis
    openai_response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in legal compliance and document comparison."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.0
    )
    
    # Return the comparison result as a dictionary for Pydantic model
    return {
        "subsection": subsection,
        "meaningful_changes_comparison": openai_response.choices[0].message.content.strip()
    }

# Main function to handle the entire comparison and output the results in Pydantic JSON format
async def compare_documents_section_by_section(old_chunks, new_doc_namespace, index):
    comparison_results = []

    # Match the old and new document chunks using vector similarity across namespaces
    matched_chunks = await match_chunks_using_similarity(index, old_chunks, new_doc_namespace)

    # For each matched chunk, compare them using the LLM
    for match in matched_chunks:
        old_text = match['old_chunk']
        new_text = match['new_chunk']
        subsection_old = match['old_metadata']['subsection']
        subsection_new = match['new_metadata']['subsection']

        # Get the meaningful comparison from the LLM
        comparison_result = await compare_chunks_with_llm(old_text, new_text, subsection_old, subsection_new)

        # Append to the result list (using the Pydantic model later)
        comparison_results.append(SubsectionComparison(**comparison_result))

    return comparison_results


async def main():
    # Step 1: Initialize Pinecone (with environment variable for API key)
    pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY") or 'PINECONE_API_KEY'
    )
    index_name='pdf-gaia-test'

    index = pc.Index(index_name, 'https://pdf-gaia-test-f3fg8ao.svc.aped-4627-b74a.pinecone.io')

    # Define the old policy namespace and index
    old_policy_namespace = "ekyc2020n"

    section_chunked_old = extract_sections_and_chunk(group_sections1)
    section_chunked_new = extract_sections_and_chunk(group_sections2)


#     # Retrieve all old chunks
#     all_old_chunks = await retrieve_all_old_chunks(old_policy_namespace, index_name)
#     print(all_old_chunks)

#     # Output the number of retrieved chunks and a sample
#     print(f"Retrieved {len(all_old_chunks)} chunks from the old policy namespace.")
#     print("Sample chunk:", all_old_chunks[0] if all_old_chunks else "No chunks found.")

#     # Check if the index exists, if not, create it
#     # if index_name not in pinecone.list_indexes():
#     #     pinecone.create_index(index_name, dimension=1536)  # Dimension of 'text-embedding-ada-002' embeddings

    # Step 3: Chunk and extract sections from both policy PDFs
    # section_chunked_old = extract_sections_and_chunk(group_sections1)
    # section_chunked_new = extract_sections_and_chunk(group_sections2)

#     # Store chunks in Pinecone
#     # await store_chunks_in_pinecone(index, section_chunked_old, 'ekyc2020nn')
#     # await store_chunks_in_pinecone(index, section_chunked_new, 'ekyc2024nn')

#     # Close the Pinecone connection
#     # pinecone.deinit()

# Run the main function
asyncio.run(main())