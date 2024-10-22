from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
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

embeddings = OpenAIEmbeddings()
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY") or 'PINECONE_API_KEY'
)
index_name='pdf-gaia-test'

index = pc.Index(index_name, 'https://pdf-gaia-test-f3fg8ao.svc.aped-4627-b74a.pinecone.io')

parser = StrOutputParser()
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# Define a Pydantic model to structure the response
class LLMResponseModel(BaseModel):
    section_number: str
    meaningful_changes: list[str]

# Function to parse the LLM response and extract bullet points
def parse_openai_response(openai_response_text: str) -> list[str]:
    bullet_points = re.split(r"[\n•*-]\s+", openai_response_text.strip())
    # Clean up the list and remove any empty strings
    bullet_points = [point.strip() for point in bullet_points if point.strip()]
    
    return bullet_points

# Extract text from new policy in subsection dictionary
pdf_sections1 = extract_sections("ekyc_2020_06.pdf")
pdf_sections2 = extract_sections("ekyc_2024_04.pdf")

# Group subsection dict into section dict
group_sections1 = group_subsections(pdf_sections1) # Old policy
group_sections2 = group_subsections(pdf_sections2) # revised policy

# print(group_sections1)

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Move the window with overlap

    return chunks

def extract_and_chunk_policy(policy_dict, chunk_size=1000, overlap=100):
    section_chunks = {}

    # Iterate through each section in the policy documents
    for section, subsections in policy_dict.items():
        section_number = section  # Extract the section number
        
        combined_text = ""  # To combine text across subsections
        metadata_list = []  # To keep track of subsections and their metadata for this section

        # Iterate through each subsection and concatenate the content
        for subsection, content in subsections.items():
            # content = content_dict['content']
            
            # Combine text and footnotes into a single string
            combined_text += content['text']
            # if content['footnotes']:
            #     combined_text += " " + " ".join(content['footnotes'])  # Append footnotes
            
            # Collect metadata for this subsection
            metadata_list.append({
                'subsection': subsection,
                'type': content['type'],
                # 'text': combined_text,
                'footnotes': content['footnotes']
            })

            print(f"metadata: {metadata_list}")
        
        # Now chunk the combined text for this entire section
        chunks = chunk_text(combined_text, chunk_size, overlap)
        print(chunks)
        
        # Store the chunks for this section in a dictionary
        section_chunks[section_number] = []  # Initialize the list of chunks for the section
        
        # Store each chunk with metadata, numbered as chunk 1, 2, 3, etc.
        for i, chunk in enumerate(chunks):
            section_chunks[section_number].append({
                'chunk_number': i + 1,
                'chunk_text': chunk,
                'metadata': metadata_list  # Metadata for all subsections in this section
            })

    return section_chunks

section_object_old = extract_and_chunk_policy(group_sections1)
section_object_new = extract_and_chunk_policy(group_sections2)

print(type(section_object_new))

async def store_chunks_in_pinecone(section_chunks, namespace):
    # Check the type of section_chunks (which should be a dict)
    print(f"Type of section_chunks: {type(section_chunks)}")

    for section, chunks in section_chunks.items():
        # Check the type of chunks
        print(f"Processing section: {section}, chunks type: {type(chunks)}, chunks: {chunks}")
        
        if not isinstance(chunks, list):
            print(f"Error: Expected a list of chunks, but got {type(chunks)} for section {section}")
            continue  # Skip this section if chunks are not properly formatted

        for chunk in chunks:
            # Check if each chunk is a dictionary
            print(f"Processing chunk: {chunk}, chunk type: {type(chunk)}")

            if 'metadata' not in chunk:
                print(f"Error: No 'metadata' key in chunk: {chunk}")
                continue

            # Check the type of metadata and its content
            metadata = chunk['metadata']
            print(f"Metadata: {metadata}, metadata type: {type(metadata)}")

            if isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
                chunk_metadata = {
                    'section': section,
                    # 'subsection': metadata['subsection'],
                    'chunk_number': chunk['chunk_number'],
                    'type': metadata['type'],
                    'text': metadata['text']
                }
                print(f"Chunk metadata to be stored: {chunk_metadata}")
            else:
                print(f"Unexpected metadata format: {metadata}, skipping this chunk.")
                continue  # Skip this chunk if metadata is malformed
            
            # Get the embedding for the chunk (simulated function)
            chunk_text = chunk['chunk_text']  # Ensure chunk_text is correct
            embedding = await get_embedding(chunk_text)
            vector_id = f"{section}_{chunk['chunk_number']}"  # Unique ID for each chunk

            if embedding:
                print(f"Storing vector ID: {vector_id} with embedding size: {len(embedding)}")
            else:
                print(f"Failed to get embedding for chunk: {chunk}")
                continue  # Skip storing this chunk if embedding failed

            # Try to upsert into Pinecone
            try:
                index.upsert(vectors=[(vector_id, embedding)], namespace=namespace, metadata=chunk_metadata)
                print(f"Successfully upserted vector for section: {section}, chunk number: {chunk['chunk_number']}")
            except Exception as e:
                print(f"Failed to upsert vector: {e}")

# Store old and new policy into vector database in chunks
async def store_old_policy():
    await store_chunks_in_pinecone(section_object_old, 'ekyc2020nn')
async def store_new_policy():
    await store_chunks_in_pinecone(section_object_new, 'ekyc2024nn')

async def process_policies():
    await asyncio.gather(store_old_policy(), store_new_policy())



# Query Pinecone for matching chunks from the old policy
async def query_old_policy_for_chunks(chunk_text, old_policy_namespace="ekyc2020n"):
    # for section, subsections in policy_chunks.items():
        # for subsection, chunk_text in subsections.items():
    embedding = await get_embedding(chunk_text)
    try:
        response = index.query(
            vector=embedding,
            namespace=old_policy_namespace,  # Old policy namespace
            top_k=3,  # Retrieve top k matches
            include_metadata=True
        )
    except Exception as e:
        print(f"Query failed: {e}")
        return "Query failed"

# Handle response

    if response.matches:
        closest_match_text = "\n".join([match.metadata.get('text', 'No text found') for match in response.matches[:1] if match.metadata])
    else:
        closest_match_text = "No corresponding section found in old policy"

    return closest_match_text

# Function to compare chunks and generate the result
async def process_and_compare_chunks(section_chunks, retries=3):
    comparison_results = {}

    # Iterate over every section in the chunked_policy
    for section, chunked_subsections in section_chunks.items():
        section_results = {}  # Store results for each subsection within a section
    
        for subsection, chunks in chunked_subsections.items():
            subsection_results = []
            
            for idx, chunk in enumerate(chunks):
                # Query the old policy for matching content
                closest_match_text = await query_old_policy_for_chunks(chunk, old_policy_namespace="ekyc2020n")

                # Check if it's a new subsection that doesn't exist in the old policy
                if closest_match_text == "No corresponding section found in old policy":
                    result = f"New Subsection: {section}.{subsection} has no corresponding section in the old policy."
                # Check token limit before making API call
                if check_token_limit(chunk, closest_match_text):
                    # Generate comparison prompt only if token limits are respected
                    prompt = (f"Revised policy Section {section}, Subsection {subsection}, Chunk {idx}:\n"
                              f"Revised Text: {chunk}\n"
                              f"Old Policy Text: {closest_match_text}\n"
                              "Identify and highlight all semantic changes between the two versions. Be concise. Only consider changes that would impact internal policy.")

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

                    # Parse and store the comparison result for this chunk
                    result = parse_openai_response(openai_response.choices[0].message.content.strip())
                else: 
                    result = "Token limit exceeded. Chunk too large for comparison."

                subsection_results.append(result)

                try:
                    # Step 5: Create a structured response using Pydantic
                    llm_response = LLMResponseModel(
                        section_number=section,
                        meaningful_changes=result  # Store the parsed bullet points
                    )
                    
                    return llm_response
                
                except (openai.RateLimitError, openai.APIError, openai.APIConnectionError) as e:
                    # Handle errors and retry if needed globally for the loop
                    logging.error(f"Error in section {section}, retrying: {e}")
                    if retries > 0:
                        await asyncio.sleep(10)  # Add delay before retry
                        return await process_and_compare_chunks(section_chunks, retries=retries-1)

            # Store results for each subsection
            section_results[subsection] = subsection_results
        
        # Store results for each section
        comparison_results[section] = section_results
    
    return comparison_results # Return the results for all sections

# Asynchronous function to get embeddings (simulated here, replace with actual embedding logic)
async def get_embedding(text):
    try:
        embedding_response = await client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        # return embedding_response['data'][0]['embedding']
        # Extract the first embedding from the response
        embedding = embedding_response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

def check_token_limit(revised_text, old_text, max_tokens=4096):
    total_tokens = len(revised_text.split()) + len(old_text.split())
    return total_tokens < max_tokens

# Asynchronous function to manage tasks and display results as they complete
async def process_sections_async(section_chunks):
    
    tasks = []

    for chunk in section_chunks:
        task = asyncio.create_task(process_and_compare_chunks(chunk))
        tasks.append(task)

    # Iterate over tasks as they complete using asyncio.as_completed()
    for task in asyncio.as_completed(tasks):
        try:
            llm_response = await task  # Get the result of the completed task
            if llm_response:
                display_result(llm_response)  # Display result to the frontend immediately
            else: 
                logger.info(f"An error occured with task {section_chunks['section']}, moving onto the next.")
        except Exception as e:
            logger.error(f"An exepction occurred: {e}")

# Function to simulate displaying results to the front end (pagination simulation)
def display_result(llm_response):
    # Check if llm_response is a dictionary or an instance of LLMResponseModel
    if isinstance(llm_response, dict):
        section_number = llm_response.get('section_number', 'Unknown Section')
        meaningful_changes = llm_response.get('meaningful_changes', [])
    else:
        section_number = llm_response.section_number
        meaningful_changes = llm_response.meaningful_changes

    result = f"Section: {section_number}\n"
    
    # Loop through the meaningful changes (assuming these are bullet points extracted from OpenAI response)
    for bullet_point in meaningful_changes:
        result += f"{bullet_point}\n"
    
    print(result)  # Output the formatted result
    # print(f"Displaying result for {llm_response.section_number}...")
    # print(f"Page {llm_response.section_number}: {llm_response.meaningful_changes}")
    # Add logic to update your frontend's pagination or UI here (e.g., send via WebSocket to frontend)

async def main():
    await process_sections_async(section_object_new)
    await process_policies()
    await process_and_compare_chunks(section_object_new)

# Run the asynchronous processing function
if __name__ == "__main__":
    asyncio.run(main())


