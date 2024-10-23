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
    section_number: int
    meaningful_changes: list[str]

# Function to parse the LLM response and extract bullet points
def parse_openai_response(openai_response_text: str) -> list[str]:
    # Use regular expressions to split the text into bullet points, assuming bullet points are lines starting with '-', '*', or numbered.
    # You can customize this based on how the response is formatted.
    bullet_points = re.split(r"[\n•*-]\s+", openai_response_text.strip())
    # Clean up the list and remove any empty strings
    bullet_points = [point.strip() for point in bullet_points if point.strip()]
    
    return bullet_points

# Extract text from new policy in subsection dictionary
pdf_sections = extract_sections("ekyc_2024_04.pdf")

# Group subsection dict into section dict
group_sections = group_subsections(pdf_sections)

# Asynchronous function to query the vector database and then call OpenAI for detailed analysis
async def query_openai_llm_async(section, subsection, content, retries=3):
    # Step 1: Query Pinecone to get the closest matching vectors
    # Assume section_text is embedded before querying (using your embedding model)
    # Here, we simulate an embedding call. Use your actual embedding logic.

    context = (f"Section: {section} Subsection: {subsection}"
               f"Type: {content['type']}"
               f"Text: {content['text']} Footnotes: {content['footnotes']}")

    try:
        embedding = await get_embedding(context)
        response = index.query( # Query the Pinecone vector index
            vector=embedding, 
            namespace='ekyc2020', 
            top_k=3, 
            include_metadata=True
            )
    
        # Get the closest 2 matches
        # closest_match_text = "\n".join(response['matches'][:1]['metadata']['text'])
        if response.matches:
            # Ensure 'metadata' is available and handle cases where it may not exist
            closest_match_text = "\n".join([match.metadata.get('text', 'No text found') for match in response.matches if match.metadata])
        else:
            closest_match_text = "No matching sections found"

        # Step 2: Formulate the prompt using the section text and the closest match from the vector database
        prompt = (f"This is a section from the revised policy. Section: {section} Subsection: {subsection} Type: {content['type']} Text: {content['text']} Footnotes: {content['footnotes']}"
                  f"Find the corresponding section from the document database: {closest_match_text}."
                  "Identify and highlight all semantic changes between the two versions in bullet form. Only output changes that would impact internal policy. Be concise. Cite section numbers (eg 1.2, 2.2, 8.1) for traceability."
                  "Example: a)Section 2.1 : The addition of 'and any other institution that may be specified by the Bank' in the revised policy expands the scope of applicability beyond just financial institutions as defined in paragraph 5.2."
                  "b)Section 2.2: The date of the Agent Banking Policy Document has been updated from 30 April 2015 to 30 June 2022 in the revised version.")

        # Step 3: Query OpenAI for detailed comparison
        openai_response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in legal compliance and document comparison."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            
            max_tokens=2000
        )

        # Step 4: Parse the OpenAI response to extract bullet points
        bullet_points = parse_openai_response(openai_response.choices[0].message.content.strip())

        try:
            # Step 5: Create a structured response using Pydantic
            llm_response = LLMResponseModel(
                section_number=section,
                meaningful_changes=bullet_points  # Store the parsed bullet points
            )
            
            return llm_response
        
        except ValidationError as ve:
            # Handle pydantic validation errors
            logging.error(f"Validation Error in section {section}: {ve}")
            return {"error": f"Validation Error in section {section}"}

    except openai.RateLimitError as e:
        logger.error(f"Rate limit exceeded for section {section}: {e}")
        await asyncio.sleep(10)  # Retry after a delay
        if retries > 0:
            return await query_openai_llm_async(section, subsection, content, retries=retries-1)    
    except openai.APIError as ae:
        logger.error(f"OpenAI API Error for section {section} : {ae}")
        return {"error": f"API Error for section {section} : {ae}"}
    except openai.APIConnectionError as pe:
        logger.error(f"Pinecone API connection error for section {section} : {pe}")
        return {"error": f"Pinecone API Error for section {section} : {pe}"}
    except Exception as ex:
        logging.error(f"Unexpected error for section {section}: {ex}")
        return {"error": f"Unexpected error for section: {section}"}

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

# Asynchronous function to manage tasks and display results as they complete
async def process_sections_async():
    
    tasks = []

    for section, subsections in group_sections.items():
        for subsection, content in subsections.items():
                task = asyncio.create_task(query_openai_llm_async(section, subsection, content))
                tasks.append(task)

    # Iterate over tasks as they complete using asyncio.as_completed()
    for task in asyncio.as_completed(tasks):
        llm_response = await task  # Get the result of the completed task
        if llm_response:
            display_result(llm_response)  # Display result to the frontend immediately
        else: 
            logger.info(f"An error occured with task {section}, moving onto the next.")

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
    asyncio.run(process_sections_async())

