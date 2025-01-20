import os
import importlib
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, abort
from typing import Dict, Any
from flasgger import Swagger

# Azure OpenAI Imports
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.settings import Settings

# Custom tools and utility functions
import src.tools as tools
importlib.reload(tools)
from src.tools import (
    download_sharepoint_files,
    get_index_docs_summary,
    multimodal_query_engine
)

# Load environment variables
load_dotenv(override=True)

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Environment Variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
LOCAL_BASE_DIR = os.getenv("LOCAL_BASE_DIR")

azure_openai_endpoint = (
    f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
    f"{AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME}/chat/completions?api-version=2024-08-01-preview"
)

llm = AzureOpenAI(
    model=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-08-01-preview",
)
embed_model = AzureOpenAIEmbedding(
    model=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-08-01-preview",
)

Settings.llm = llm
Settings.embed_model = embed_model

app = Flask(__name__)
app.config['SWAGGER'] = {
    'title': 'Query API',
    'uiversion': 3  # Use Swagger UI version 3
}
swagger = Swagger(app)

@app.route("/", methods=["GET"])
def home():
    return "Flask app is running. Use the `/query` endpoint to interact with the application."

@app.route("/query", methods=["POST"])
def handle_query():
    """
    Process a user query.
    ---
    tags:
      - Query
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              description: The query string to process.
              example: What is the total assets within the balance sheet?
    responses:
      200:
        description: Successfully processed query.
      400:
        description: Query cannot be empty.
      500:
        description: An error occurred while processing the query.
    """
    data = request.get_json()
    if not data or "query" not in data:
        abort(400, description="Query cannot be empty.")
    query = data["query"].strip()
    if not query:
        abort(400, description="Query cannot be empty.")

    document_url_dict: Dict[str, Any] = {}
    document_summary_dict: Dict[str, str] = {}
    indexes: Dict[str, Any] = {}

    logger.info("Starting up and preparing environment...")
    # 1. Download files from SharePoint
    document_url_dict = download_sharepoint_files()
    logger.info(f"Downloaded documents. Found {len(document_url_dict)} files.")

    # 2. Build or load indexes and summaries
    indexes, document_summary_dict = get_index_docs_summary()
    if not indexes:
        logger.warning("No indexes were created. Check if documents were successfully processed.")
    else:
        logger.info("Indexes and summaries prepared successfully.")

    if not document_summary_dict:
        logger.error("Document summaries are empty. Ensure indexes were built successfully.")
        abort(500, description="No document summaries found.")

    # Prepare the prompt for selecting the index
    prompt = (
        f"Given the query '{query}', determine the most appropriate index (key) in the dictionary "
        f"based on the description (value) that corresponds best to the query. "
        f"Return only the index (key) as the output, nothing else. "
        f"Options: {', '.join([f'Index: {key}, Description: {value}' for key, value in document_summary_dict.items()])}"
    )

    try:
        selected_index = llm.complete(prompt).text.strip()
        logger.info(f"Selected index: {selected_index}")

        if selected_index not in indexes:
            logger.warning(f"LLM returned an invalid index: {selected_index}. Using a fallback if available.")
            if len(indexes) == 1:
                selected_index = next(iter(indexes))
                logger.info(f"Falling back to the only available index: {selected_index}")
            else:
                abort(500, description="LLM did not return a valid index.")

        index = indexes[selected_index]

        # Create and initialize the query engine
        query_engine = multimodal_query_engine(index)
        logger.info("Query engine has been initialized.")

        # Execute the query
        response = query_engine.query(query)
        return jsonify({"response": response.response})

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        abort(500, description="An error occurred while processing the query.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
