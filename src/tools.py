import os
import re
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import requests
from llama_index.core.base.response.schema import Response
from PIL import Image
from azure.identity import ClientSecretCredential

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode, ImageNode, NodeWithScore, MetadataMode
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import BlobServiceClient as BlobServiceClientSync
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta, timezone
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_parse import LlamaParse
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from llama_index.vector_stores.azureaisearch import MetadataIndexFieldType
from llama_index.core.schema import  MetadataMode
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.settings import Settings

import asyncio
import nest_asyncio
nest_asyncio.apply()

import logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from dotenv import load_dotenv
load_dotenv(override=True)

BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_STORAGE_ACCOUNT_KEY = os.getenv("BLOB_STORAGE_ACCOUNT_KEY")
SHAREPOINT_DOMAIN = os.getenv("SHAREPOINT_DOMAIN")
SHAREPOINT_SITE = os.getenv("SHAREPOINT_SITE")
BASE_FOLDER = os.getenv("BASE_FOLDER")
LOCAL_BASE_DIR = os.getenv("LOCAL_BASE_DIR")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
azure_openai_endpoint = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME}/chat/completions?api-version=2024-08-01-preview"
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
DOCUMENT_SUMMARY_DICT_PATH = os.getenv("DOCUMENT_SUMMARY_DICT_PATH", "cached/document_summary.json")
DOCUMENT_URL_DICT_PATH = os.getenv("DOCUMENT_URL_DICT_PATH", "cached/document_url.json")
IMAGES_DIR = os.getenv("IMAGES_DIR", "images")
CONCURRENT_UPLOADS = int(os.getenv("CONCURRENT_UPLOADS", 5)) 
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 3))
SHAREPOINT_FILES_CONTAINER = os.getenv("SHAREPOINT_FILES_CONTAINER")
METADATA_FILES_CONTAINER = os.getenv("METADATA_FILES_CONTAINER")

llm = AzureOpenAI(
    model=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-08-01-preview"
)

azure_openai_mm_llm = AzureOpenAIMultiModal(
    engine=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    api_version="2024-08-01-preview",
    model=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    max_new_tokens=4096,
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
)

llm_light_task = AzureOpenAI(
    model='gpt-4o-mini',
    deployment_name='gpt-4o-mini',
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-08-01-preview"
)

embed_model = AzureOpenAIEmbedding(
    model=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-08-01-preview"
)

parser = LlamaParse(
    result_type="markdown",
    use_vendor_multimodal_model=True,
    azure_openai_deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    azure_openai_endpoint=azure_openai_endpoint,
    azure_openai_api_version="2024-08-01-preview",
    azure_openai_key=AZURE_OPENAI_API_KEY,
    parsing_instruction = """
    Extract all visible text from the document, preserving the structure, formatting, and hierarchy of content, including headers, subheaders, paragraphs, lists, and tables. Ensure the following:
    1. Complete Capture: Retain all text exactly as it appears, including citations, references, disclaimers, and attributions.
    2. Visual Elements: Describe visual elements (e.g., logos, charts, illustrations) with alternative text or metadata, if applicable.
    3. Formatting: Maintain inline formatting such as bold, italics, and underlines, as well as the logical flow of lists, URLs, and numbered items.
    4. Tables: Extract tables row by row, preserving readability and structure.
    5. Annotations: Provide proper attribution for illustrations and flag unparseable sections as "[Unparsed content: description]" for manual review.
    """
)

Settings.llm = llm
Settings.embed_model = embed_model

# Initialize search clients
credential = AzureKeyCredential(SEARCH_SERVICE_API_KEY)
index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=credential)

# Define metadata fields mapping
metadata_fields = {
    "page_num": ("page_num", MetadataIndexFieldType.INT64),
    "image_path": ("image_path", MetadataIndexFieldType.STRING),
    "parsed_text_markdown": ("parsed_text_markdown", MetadataIndexFieldType.STRING),
}

# Define QA prompt template
QA_PROMPT_TMPL = """\
Below we give parsed text from slides in parsed markdown format, as well as the image.

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
from the parsed markdown or raw text or image, and if there's discrepancies, and your reasoning for the final answer.
Always return the math equations and terms within the math equations in LATEX markdown (between $$).
If the information is retrieved from a design guideline, the final answer MUST include the section number and page number where the information was retrieved from (Section 2.4 or Appendix B.1 and page number 75).

Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

import time

def robust_get_json_result(parser, file_path, max_retries=3, sleep_secs=2):
    for attempt in range(max_retries):
        md_json_objs = parser.get_json_result(file_path)
        if md_json_objs and "pages" in md_json_objs[0]:
            return md_json_objs
        time.sleep(sleep_secs)
    raise ValueError(f"Parser failed to return 'pages' after {max_retries} attempts.")

def get_images_from_doc(file_path: str, DOWNLOAD_PATH: str, parser) -> List[Dict]:

    # Parse document and extract images
    md_json_objs = robust_get_json_result(parser, file_path)
    md_json_list = md_json_objs[0]["pages"]
    image_dicts = parser.get_images(md_json_objs, download_path=DOWNLOAD_PATH)
    return  md_json_list, image_dicts

def parse_connection_string(connection_string: str) -> Dict[str, str]:
    """Parse an Azure Storage connection string into its components."""
    components = {}
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            components[key.strip()] = value.strip()
    return components


def generate_sas_token(container_name, blob_name, connection_string, expiry_days=180):
    """Generate a SAS token for a specific blob."""
    connection_data = parse_connection_string(connection_string)
    account_name = connection_data['AccountName']  # Note the capital 'A' and 'N'
    account_key = connection_data['AccountKey']    # Note the capital 'A' and 'K'

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(days=expiry_days)
    )
    return sas_token

def check_container_exists(container_name: str) -> bool:
    blob_service_client = BlobServiceClientSync.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(container_name)
    return container_client.exists()

async def create_container_if_not_exists(connection_string: str, container_name: str):
    """Create container if it doesn't exist, handling the operation once."""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    async with blob_service_client:
        container_client = blob_service_client.get_container_client(container_name)
        if not await container_client.exists():
            try:
                await container_client.create_container()
                print(f"Container {container_name} created successfully")
            except Exception as e:
                print(f"Error creating container: {e}")

async def upload_image_to_blob_storage(image, blob_name, semaphore, BLOB_CONTAINER_NAME):
    """Upload a single image to blob storage."""
    async with semaphore:
        try:
            image_path = image.get('path') or image.get('original_file_path')
            if not image_path or not os.path.exists(image_path):
                print(f"Image path not found or invalid: {image_path}")
                return None

            blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
            async with blob_service_client:
                container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
                blob_client = container_client.get_blob_client(blob_name)

                with open(image_path, "rb") as f:
                    await blob_client.upload_blob(f, overwrite=True)
                    print(f"Successfully uploaded {blob_name}")

                    # Generate SAS token for the uploaded blob
                    sas_token = generate_sas_token(
                        container_name=BLOB_CONTAINER_NAME,
                        blob_name=blob_name,
                        connection_string=BLOB_CONNECTION_STRING
                    )

                    # Construct the SAS URL
                    sas_url = f"{blob_client.url}?{sas_token}"
                    # sas_url = blob_client.url
                return sas_url

        except Exception as e:
            print(f"Failed to upload {blob_name}: {str(e)}")
            return None

async def main(BLOB_CONTAINER_NAME, CONCURRENT_UPLOADS, image_dicts):
    """Main function to handle container creation and image uploads."""
    # First, ensure the container exists
    await create_container_if_not_exists(BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME)
    
    # Then proceed with uploads
    semaphore = asyncio.Semaphore(CONCURRENT_UPLOADS)
    upload_tasks = [upload_image_to_blob_storage(image, image["name"], semaphore, BLOB_CONTAINER_NAME) 
                   for image in image_dicts]
    
    results = await asyncio.gather(*upload_tasks)
    
    # Create dictionary of successful uploads
    successful_uploads = {
        image["name"]: url
        for image, url in zip(image_dicts, results)
        if url is not None
    }
    
    # Print summary
    print(f"\nUpload Summary:")
    print(f"Total images: {len(image_dicts)}")
    print(f"Successfully uploaded: {len(successful_uploads)}")
    print(f"Failed uploads: {len(image_dicts) - len(successful_uploads)}")
    
    return successful_uploads

def upload_json_dict_to_blob_storage(
    dict_data: dict, 
    blob_name: str
) -> None:
    """
    Upload a Python dictionary as JSON directly to a blob container, without saving locally.
    Creates the container if it does not exist.
    """
    # Convert dict to JSON bytes
    json_bytes = json.dumps(dict_data).encode("utf-8")
    data_stream = BytesIO(json_bytes)

    # Initialize blob service
    blob_service_client = BlobServiceClientSync.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(METADATA_FILES_CONTAINER)

    # Create the container if it does not exist
    if not container_client.exists():
        container_client.create_container()

    # Upload the bytes to the specified blob
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data_stream, overwrite=True)
    print(f"Uploaded {blob_name} to container '{METADATA_FILES_CONTAINER}' from memory successfully.")
    
def upload_images_to_blob_storage(BLOB_CONTAINER_NAME, CONCURRENT_UPLOADS, image_dicts):
    image_urls = asyncio.run(main(BLOB_CONTAINER_NAME, CONCURRENT_UPLOADS, image_dicts))
    return image_urls

def get_page_number(file_name: str) -> int:
    """Extract page number from blob name."""
    match = re.search(r"page_(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0

def _get_sorted_blob_urls(image_urls: Dict[str, str]) -> List[str]:
    """Get blob URLs sorted by page number."""
    sorted_items = sorted(image_urls.items(), key=lambda x: get_page_number(x[0]))
    return [url for _, url in sorted_items]

def get_text_nodes(image_urls: Dict[str, str], json_dicts: List[dict]) -> List[TextNode]:
    """Create TextNodes with metadata including blob URLs as image_path."""
    nodes = []
    
    sorted_urls = _get_sorted_blob_urls(image_urls)
    md_texts = [d["md"] for d in json_dicts]

    for idx, md_text in enumerate(md_texts):
        if idx >= len(sorted_urls):
            continue
            
        node = TextNode(
            text=md_text,
            metadata={
                "page_num": idx + 1,
                "image_path": sorted_urls[idx],
                "parsed_text_markdown": md_texts[idx],
            }
        )
        nodes.append(node)

    return nodes

def create_vector_store(
    index_client,
    index_name: str,
    metadata_fields: dict,
    use_existing_index: bool = False
) -> AzureAISearchVectorStore:
    """Create or get existing Azure AI Search vector store."""
    return AzureAISearchVectorStore(
        search_or_index_client=index_client,
        index_name=index_name,
        index_management=IndexManagement.VALIDATE_INDEX if use_existing_index 
                        else IndexManagement.CREATE_IF_NOT_EXISTS,
        id_field_key="id",
        chunk_field_key="parsed_text_markdown",
        embedding_field_key="embedding",
        embedding_dimensionality=1536,
        metadata_string_field_key="metadata",
        doc_id_field_key="doc_id",
        filterable_metadata_field_keys=metadata_fields,
        language_analyzer="en.lucene",
        vector_algorithm_type="exhaustiveKnn",
    )

def create_or_load_index(
    text_nodes,
    index_client,
    index_name: str,
    embed_model,
    llm,
    metadata_fields: dict,
    use_existing_index: bool = False
) -> VectorStoreIndex:
    """Create new index or load existing one."""
    vector_store = create_vector_store(index_client, index_name, metadata_fields, use_existing_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if use_existing_index:
        return VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
        )
    else:
        return VectorStoreIndex(
            nodes=text_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            llm=llm,
            show_progress=True,
        )

class MultimodalQueryEngine(CustomQueryEngine):
    """Custom multimodal Query Engine for public blob storage."""

    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: AzureOpenAIMultiModal
    reranker: Optional[SentenceTransformerRerank] = None  # Add reranker

    def __init__(
        self,
        qa_prompt: Optional[PromptTemplate] = None,
        reranker_top_n: int = 3,  # Default top_n for reranker
        **kwargs,
    ) -> None:
        """Initialize."""
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=reranker_top_n
        )  # Initialize the reranker

    def custom_query(self, query_str: str) -> Response:
        # Retrieve relevant nodes
        nodes = self.retriever.retrieve(query_str)

        # Rerank the nodes using the specified reranker
        if self.reranker:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=query_str)

        # Create ImageNode items directly using the blob URLs
        image_nodes = []
        for n in nodes:
            if "image_path" in n.metadata:
                try:
                    image_nodes.append(
                        NodeWithScore(
                            node=ImageNode(image_url=n.metadata["image_path"])
                        )
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to create ImageNode for {n.metadata['image_path']}: {str(e)}"
                    )
                    continue

        # Create context string from text nodes
        context_str = "\n\n".join(
            [node.get_content(metadata_mode=MetadataMode.LLM) for node in nodes]
        )

        # Format the prompt
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        # Get response from multimodal LLM
        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[image_node.node for image_node in image_nodes],
        )

        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": nodes, "image_nodes": image_nodes},
        )
    
def display_query_and_multimodal_response(
    response: Response, plot_height: int = 2, plot_width: int = 5
) -> None:
    """For displaying a query and its multi-modal response."""
    if response.metadata:
        image_nodes = response.metadata["image_nodes"] or []
    else:
        image_nodes = []
    num_subplots = len(image_nodes)
    if num_subplots == 0:
        # Handle the case where there are no images to display
        print("No images found to display.")
    else:
        f, axarr = plt.subplots(1, num_subplots)
        f.set_figheight(plot_height)
        f.set_figwidth(plot_width)
        ix = 0
        for ix, scored_img_node in enumerate(image_nodes):
            img_node = scored_img_node.node
            image = None
            if img_node.image_url:
                img_response = requests.get(img_node.image_url)
                image = Image.open(BytesIO(img_response.content)).convert("RGB")
            elif img_node.image_path:
                image = Image.open(img_node.image_path).convert("RGB")
            else:
                raise ValueError(
                    "A retrieved image must have image_path or image_url specified."
                )

            if num_subplots > 1:
                axarr[ix].imshow(image)
                axarr[ix].set_title(f"Retrieved Position: {ix}", pad=10, fontsize=9)
            else:
                axarr.imshow(image)
                axarr.set_title(f"Retrieved Position: {ix}", pad=10, fontsize=9)

def display_query_and_multimodal_response_saved_images(
    response: Response
) -> None:
    """For displaying a query and its multi-modal response."""
    if response.metadata:
        image_nodes = response.metadata["image_nodes"] or []
    else:
        image_nodes = []
    num_subplots = len(image_nodes)
    if num_subplots == 0:
        # Handle the case where there are no images to display
        print("No images found to display.")
    else:
        ix = 0
        for ix, scored_img_node in enumerate(image_nodes):
            img_node = scored_img_node.node
            image = None
            if img_node.image_url:
                img_response = requests.get(img_node.image_url)
                image = Image.open(BytesIO(img_response.content)).convert("RGB")
            elif img_node.image_path:
                image = Image.open(img_node.image_path).convert("RGB")
            else:
                raise ValueError(
                    "A retrieved image must have image_path or image_url specified."
                )

def load_json_dict_from_blob_storage(blob_name: str) -> dict:
    """
    Loads a JSON blob from METADATA_FILES_CONTAINER into a Python dict.
    If the container or blob doesn't exist, returns an empty dict.
    """
    blob_service_client = BlobServiceClientSync.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(METADATA_FILES_CONTAINER)

    # If container doesn't exist, return empty
    if not container_client.exists():
        return {}

    blob_client = container_client.get_blob_client(blob_name)

    # If blob doesn't exist, return empty
    if not blob_client.exists():
        return {}

    # Download and parse JSON
    downloaded_blob = blob_client.download_blob().readall()
    return json.loads(downloaded_blob)

def download_sharepoint_files():
        
    document_url_dict = {}
    document_url_dict = load_json_dict_from_blob_storage("document_url.json")

    # Authenticate using the Service Principal
    credential = ClientSecretCredential(tenant_id=TENANT_ID, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    access_token = credential.get_token("https://graph.microsoft.com/.default").token
    headers = {"Authorization": f"Bearer {access_token}"}

    # Get the site ID
    site_url = f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_DOMAIN}:/sites/{SHAREPOINT_SITE}"
    site_response = requests.get(site_url, headers=headers)
    site_response.raise_for_status()
    site_id = site_response.json().get('id')

    # Get the drive (document library) ID
    drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    drives_response = requests.get(drives_url, headers=headers)
    drives_response.raise_for_status()
    drives = drives_response.json().get('value', [])

    # Select the 'Documents' drive
    drive_id = next((drive['id'] for drive in drives if drive['name'] == "Documents"), None)
    if not drive_id:
        raise ValueError("Drive 'Documents' not found.")

    # Fetch files from SharePoint
    files_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{BASE_FOLDER}:/children"
    files_response = requests.get(files_url, headers=headers)
    files_response.raise_for_status()
    files = files_response.json()

    # Download files
    for file in files.get('value', []):
        # Add to URL dictionary
        document_url_dict[os.path.splitext(file['name'].lower())[0]] = file['webUrl']

    # Save URL dictionary to a JSON file in the cached directory
    blob_service_client = BlobServiceClientSync.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(SHAREPOINT_FILES_CONTAINER)

    if not container_client.exists():
        container_client.create_container()

    # Synchronous upload
    for file in files.get('value', []):
        file_url = file['@microsoft.graph.downloadUrl']
        response = requests.get(file_url)  # blocking download
        blob_client = container_client.get_blob_client(file['name'])
        blob_client.upload_blob(response.content, overwrite=True)
    upload_json_dict_to_blob_storage(document_url_dict, "document_url.json")
  
    return document_url_dict

def get_index_docs_summary():
    # Get all files in the LOCAL_BASE_DIR
    # files = os.listdir(LOCAL_BASE_DIR)

    # Grab a reference to your SharePoint files container
    blob_service_client = BlobServiceClientSync.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(SHAREPOINT_FILES_CONTAINER)

    # List all blobs (files) in the SharePoint container
    all_blobs = container_client.list_blobs()

    # Define supported extensions
    SUPPORTED_EXTENSIONS = [".pdf", ".pptx"]

    # Initialize the dictionary
    document_summary_dict = {}
    document_summary_dict = load_json_dict_from_blob_storage("document_summary.json")
    indexes = {} 

    # Loop through each file in the directory
    for blob in all_blobs:
        file_name = blob.name
        file_name_lower = file_name.lower()
        # Process files with supported extensions
        if any(file_name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS): 
            curated_file_name = os.path.splitext(file_name.lower())[0]
            BLOB_CONTAINER_NAME = f"{curated_file_name.replace('_', '-')}"
            logger.info(f"Blob Container Name: {BLOB_CONTAINER_NAME}")

            INDEX_NAME = f"{curated_file_name.replace('-', '_')}"
            logger.info(f"Index Name: {INDEX_NAME}")

            file_path = os.path.join(LOCAL_BASE_DIR, file_name)
            images_path = os.path.join(IMAGES_DIR, curated_file_name)
            
            if not check_container_exists(BLOB_CONTAINER_NAME):

                # logging.info or process the paths and container name
                logger.info(f"File Path: {file_path}")
                logger.info(f"Images Path: {images_path}")
                logger.info(f"Blob Container Name: {BLOB_CONTAINER_NAME}")

                # Download the blob to a temporary file to pass it to LlamaParse
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name_lower)[1]) as temp_file:
                    temp_filepath = temp_file.name
                    download_stream = container_client.download_blob(file_name)
                    temp_file.write(download_stream.readall())

                md_json_list, image_dicts = get_images_from_doc(temp_filepath, images_path, parser)

                image_urls = upload_images_to_blob_storage(BLOB_CONTAINER_NAME, CONCURRENT_UPLOADS, image_dicts)

                # Create text nodes
                text_nodes = get_text_nodes(image_urls=image_urls, json_dicts=md_json_list)

                document_content = " ".join(node.get_content() for node in text_nodes)
                summary = llm_light_task.complete(f"summarize this text: {document_content}").text

                # Add to dictionary
                document_summary_dict[INDEX_NAME] = summary

                # search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=credential)

                indexes[INDEX_NAME] = create_or_load_index(
                    text_nodes=text_nodes,
                    index_client=index_client,
                    index_name=INDEX_NAME,
                    embed_model=embed_model,
                    llm=llm,
                    metadata_fields=metadata_fields,
                    use_existing_index=False
                )
                print("===============================================")
            else:
                indexes[INDEX_NAME] = create_or_load_index(
                text_nodes=[],
                index_client=index_client,
                index_name=INDEX_NAME,
                embed_model=embed_model,
                llm=llm,
                metadata_fields=metadata_fields,
                use_existing_index=True
                )
                logger.info(f"{INDEX_NAME} already exists")

                logger.info("===============================================")
    
    upload_json_dict_to_blob_storage(document_summary_dict, "document_summary.json")

    return indexes, document_summary_dict

def multimodal_query_engine(index):
    query_engine = MultimodalQueryEngine(
        retriever=index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
            similarity_top_k=SIMILARITY_TOP_K,
        ),
        multi_modal_llm=azure_openai_mm_llm,
        reranker_top_n=3,  # Specify the number of nodes to consider after reranking
    )
    return query_engine