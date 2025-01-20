import os
import importlib
import logging
from dotenv import load_dotenv
from IPython.display import Markdown
import streamlit as st

# Azure OpenAI Imports
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.settings import Settings

# Custom tools and utility functions
import src.tools as tools
importlib.reload(tools)
from src.tools import (
    display_query_and_multimodal_response_saved_images,
    download_sharepoint_files,
    get_index_docs_summary,
    multimodal_query_engine
)

# Environment setup
load_dotenv(override=True)

# Logging configuration
logging.basicConfig(
    level=logging.ERROR,
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

# Derived Configuration
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
    api_version="2024-08-01-preview"
)

Settings.llm = llm
Settings.embed_model = embed_model

st.title("Advanced Multi-Modal Retrieval Augmented Generation (RAG)")
st.markdown(
    """This application processes and indexes documents of all types, from Arcurve SharePoint
             by extracting text and images, summarizing content, and storing the data in Azure Blob Storage and Azure Cognitive Search
             for multimodal querying and retrieval using Azure OpenAI models."""
)

query_str = "In terms of scaling AI/ML what percentage of executives disagree with this statement?"
query = st.text_area("**Enter your query:**", query_str)


if st.button("Submit"):
    with st.spinner("In progress..."):

        # Get all files in the LOCAL_BASE_DIR
        document_url_dict = download_sharepoint_files()

        # Get the indexes, document summary and url dictionaries
        indexes, document_summary_dict = get_index_docs_summary()

        prompt = (
            f"Given the query '{query}', determine the most appropriate index (key) in the dictionary "
            f"based on the description (value) that corresponds best to the query. "
            f"Return only the index (key) as the output, nothing else. "
            f"Options: {', '.join([f'Index: {key}, Description: {value}' for key, value in document_summary_dict.items()])}"
        )

        # Call the LLM endpoint with the prompt
        selected_index = llm.complete(prompt).text.strip()
        logger.info(f"Selected index: {selected_index}")

        index = indexes[selected_index]

        # Create and initialize the query engine
        query_engine = multimodal_query_engine(index)
        logger.info("query engine has been initialized.")

        response = query_engine.query(query)

        st.subheader("Response:")
        st.markdown(response.response)

        st.subheader("Images from Cached Responses:")    
        display_query_and_multimodal_response_saved_images(response)
        
        image_folder = "cached/responses"
        if os.path.exists(image_folder):
            images = [
                img
                for img in os.listdir(image_folder)
                if img.endswith((".png", ".jpg", ".jpeg"))
            ]
            if images:
                for image in images:
                    image_path = os.path.join(image_folder, image)
                    st.image(image_path, caption=image, use_container_width=True)

                # Display the sources
                st.markdown(f"Retrieved document:\n{selected_index}\n")

                st.subheader(f"Url to the document:")
                st.markdown(f"{document_url_dict[selected_index]}\n")

                st.subheader(f"Summary of the document:")
                st.markdown(document_summary_dict[selected_index])

            else:
                st.write("No images found in the cached responses.")
        else:
            st.write("Cached responses folder not found.")
