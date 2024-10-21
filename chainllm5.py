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

index = pc.Index(index_name)

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

# Chunk text into manageable sizes (consistent for both old and revised policies)
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Move the window with overlap
        print(len(chunks))

    return chunks

# Example of combining 'text' and 'footnotes' before chunking
def chunk_text_with_footnotes(content, chunk_size=1000, overlap=100):
    # Check if content is already a string or a dictionary
    print(type(content))
    print
    if isinstance(content, str):
        full_text = content  # If it's a string, use it directly
    elif isinstance(content, dict):
        full_text = content['text']  # If it's a dictionary, extract the 'text'
        # if 'footnotes' in content and content['footnotes']:
        if 'footnotes' in content: 
            print(content['footnotes'])
            footnotes = content['footnotes']
            if isinstance(footnotes, list):
                footnotes = "\n".join(footnotes)  # If footnotes are a list, join them
            full_text += "\nFootnotes:\n" + footnotes  # Append footnotes to the main text
    else:
        raise TypeError("Unexpected content type. Must be string or dictionary.")
    
    return chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)


# Function to chunk the entire policy (both old and revised)
def chunk_policy_sections(policy_dict):
    chunked_policy = {}
    
    for section, subsections in policy_dict.items():
        chunked_subsections = {}
        for subsection, content in subsections.items():
            text_chunks = chunk_text_with_footnotes(content['text'], chunk_size=1000, overlap=100)
            print(text_chunks)
            chunked_subsections[subsection] = text_chunks
        
        chunked_policy[section] = chunked_subsections
    
    return chunked_policy

# Chunk both old and revised policies
chunked_old_policy = chunk_policy_sections(group_sections1)
# chunked_revised_policy = chunk_policy_sections(group_sections2)

# print(chunked_old_policy)
# print(chunked_revised_policy)

# Function to store policy chunks into Pinecone
async def store_policy_in_pinecone(policy_chunks, namespace):
    for section, subsections in policy_chunks.items():
        for subsection, chunks in subsections.items():
            for idx, chunk in enumerate(chunks):
                embedding = await get_embedding(chunk)
                vector_id = f"{section}_{subsection}_{idx}"  # Unique ID for each chunk
                if embedding:
                    print(f"Storing vector ID: {vector_id} with embedding size: {len(embedding)}")
                else:
                    print(f"Failed to get embedding for chunk: {chunk}")

             
                # Store the embedding in Pinecone
                try:
                    await index.upsert(vectors=[(vector_id, embedding)], namespace=namespace, metadata={'section': section, 'subsection': subsection, 'text': chunk})
                    print(f"Upserted vector for section: {section}, subsection: {subsection}, index: {idx}")
                except Exception as e:
                    print(f"Failed to upsert vector: {e}")

# Separate storage of old and revised policies to avoid race conditions
async def store_old_policy():
    await store_policy_in_pinecone(chunked_old_policy, namespace='ekyc2020n')

async def store_revised_policy():
    await store_policy_in_pinecone(chunked_revised_policy, namespace='ekyc2024n')

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
async def process_and_compare_chunks(policy_chunks, retries=3):
    comparison_results = {}

    # Iterate over every section in the chunked_policy
    for section, chunked_subsections in policy_chunks.items():
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
                        return await process_and_compare_chunks(policy_chunks, retries=retries-1)

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
async def process_sections_async(policy_chunks):
    
    tasks = []
    task = asyncio.create_task(process_and_compare_chunks(policy_chunks))
    tasks.append(task)

    # Iterate over tasks as they complete using asyncio.as_completed()
    for task in asyncio.as_completed(tasks):
        llm_response = await task  # Get the result of the completed task
        if llm_response:
            display_result(llm_response)  # Display result to the frontend immediately
        else: 
            logger.info(f"An error occured with task {policy_chunks['section']}, moving onto the next.")

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

# Run the asynchronous processing function
if __name__ == "__main__":
    asyncio.run(process_sections_async(chunked_revised_policy))

