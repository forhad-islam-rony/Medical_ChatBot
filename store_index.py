from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINCONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file("Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

pc.create_index(
  name=index_name,
  dimension=384,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
)

docsearch = PineconeVectorStore.from_documents(
  documents=text_chunks,
  embedding=embeddings,
  index_name=index_name
)