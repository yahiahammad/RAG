import streamlit as st
from langchain_community.document_loaders import UnstructuredExcelLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
from openai import OpenAI
import os # Import os to access environment variables

# Placeholder for Groq API key loading
# Use Streamlit secrets or environment variables for secure access
# Example using environment variable: groq_api_key = os.environ.get("GROQ_API_KEY")
# Example using Streamlit secrets: groq_api_key = st.secrets["GROQ_API_KEY"]
# Ensure the Groq API key is loaded securely, e.g., from Streamlit secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Groq API key not found. Please set the GROQ_API_KEY in Streamlit secrets.")
    st.stop() # Stop execution if the key is not found


def rag(retrieved_docs, query):
    """
    Performs Retrieval-Augmented Generation (RAG) using retrieved document chunks
    and a user query to generate an answer.

    Args:
        retrieved_docs: A list of strings containing the content of retrieved document chunks.
        query: The user's query string.

    Returns:
        str: The generated answer from the language model.
    """
    context = "\n".join(retrieved_docs)

    client = OpenAI(
        api_key=groq_api_key, # Use the securely loaded API key
        base_url="https://api.groq.com/openai/v1"
    )

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": """
            You are a helpful and knowledgeable assistant specialized in books. You have access to customer reviews, ratings, and metadata about the top 200 trending books. When answering questions, use the most relevant review content, focus on readers' opinions, and highlight recurring themes. If the user asks for recommendations, tailor them to user interests based on review sentiment and popularity.

            Only answer questions using the data you have. Be concise, avoid speculation, and favor well-supported answers. When reviews are mixed, acknowledge both positives and negatives.

            If asked about a specific book, summarize the most useful reviews and the general rating. If asked for comparisons or suggestions, emphasize differences in genre, writing style, or reader feedback.

            You do not generate fictional content or hallucinate details not found in the context.
            """},
            {"role": "user", "content": f"""
              Using the information in the following context, answer the question clearly and accurately.

              Context:
              {context}

              Answer the following question:
              {query}

              If the answer cannot be found in the context, say The context does not contain enough information to answer this question. then explain why it does not contain enough information.
            """}
        ],
        temperature=0.7
    )

    answer = response.choices[0].message.content
    return answer # Return the answer string

def prepare_excel_data(uploaded_file, query, k=2):
    """
    Loads data from an uploaded Excel file, creates document chunks,
    builds a FAISS index, performs similarity search, and uses RAG to answer a query.

    Args:
        uploaded_file: The uploaded file object from Streamlit.
        query: The user's query string.
        k: The number of most similar document chunks to retrieve (default is 2).

    Returns:
        str: The generated answer from the RAG function.
    """
    # Save the uploaded file to a temporary location to be read by UnstructuredExcelLoader
    try:
        with open("temp_excel_file.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = UnstructuredExcelLoader("temp_excel_file.xlsx")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
        chunks = text_splitter.split_documents(documents)

        model = SentenceTransformer('all-MiniLM-L6-v2')
        doc_embeddings = model.encode([doc.page_content for doc in chunks])

        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings)

        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k=k)
        retrieved_docs = [chunks[i].page_content for i in I[0]]

        return rag(retrieved_docs, query)
    except Exception as e:
        st.error(f"An error occurred during Excel processing: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists("temp_excel_file.xlsx"):
            os.remove("temp_excel_file.xlsx")

def create_pdf_database(uploaded_file):
    """
    Creates a FAISS database from an uploaded PDF file and returns the index and document chunks.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        A tuple containing:
            - faiss.IndexFlatL2: The created FAISS index.
            - list: A list of document chunks (langchain.schema.document.Document objects).
        Returns None if an error occurs.
    """
    # Save the uploaded file to a temporary location
    try:
        with open("temp_pdf_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader("temp_pdf_file.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=15000,
            chunk_overlap=500
        )
        chunks = text_splitter.split_documents(documents)

        model = SentenceTransformer('all-MiniLM-L6-v2')
        doc_embeddings = model.encode([doc.page_content for doc in chunks])

        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings)

        return index, chunks
    except Exception as e:
        st.error(f"An error occurred during PDF database creation: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists("temp_pdf_file.pdf"):
            os.remove("temp_pdf_file.pdf")

def prepare_pdf_data(index: faiss.IndexFlatL2, chunks: list, query: str, k: int = 2):
    """
    Prepares PDF data for RAG by performing a similarity search on a pre-built
    FAISS index and using the retrieved chunks with the RAG function.

    Args:
        index: The pre-built FAISS index.
        chunks: A list of document chunks (langchain.schema.document.Document objects).
        query: The user's query.
        k: The number of most similar document chunks to retrieve (default is 2).

    Returns:
        str: The generated answer from the RAG function based on the retrieved chunks.
    """
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])

        D, I = index.search(query_embedding, k=k)
        retrieved_docs = [chunks[i].page_content for i in I[0]]

        return rag(retrieved_docs, query)
    except Exception as e:
        st.error(f"An error occurred during PDF data preparation: {e}")
        return None


st.title("RAG Application")

st.write("Upload a document (Excel or PDF) and ask questions about its content.")

# Layout using columns
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "pdf"])
    query = st.text_input("Enter your query:")

with col2:
    k_value = st.slider("Number of relevant documents to retrieve (k)", 1, 10, 2)

# Add a button to trigger the RAG process
if st.button("Get Answer"):
    if uploaded_file is not None and query:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "xlsx":
            with st.spinner("Processing Excel file..."):
                answer = prepare_excel_data(uploaded_file, query, k_value)
            if answer:
                st.subheader("Answer:")
                st.write(answer)
        elif file_extension == "pdf":
            # Create the database only once and store it in session state
            if "pdf_index" not in st.session_state or "pdf_chunks" not in st.session_state or st.session_state.uploaded_pdf_name != uploaded_file.name:
                with st.spinner("Creating PDF database..."):
                    index, chunks = create_pdf_database(uploaded_file)
                if index is not None and chunks is not None:
                     st.session_state.pdf_index = index
                     st.session_state.pdf_chunks = chunks
                     st.session_state.uploaded_pdf_name = uploaded_file.name
                     st.success("PDF database created.")
                else:
                    st.error("Failed to create PDF database. Please try again.")
                    # Clear session state if database creation failed
                    if "pdf_index" in st.session_state:
                        del st.session_state.pdf_index
                    if "pdf_chunks" in st.session_state:
                        del st.session_state.pdf_chunks
                    if "uploaded_pdf_name" in st.session_state:
                        del st.session_state.uploaded_pdf_name
                    st.stop() # Stop execution after showing error


            else:
                 index = st.session_state.pdf_index
                 chunks = st.session_state.pdf_chunks
                 st.info("Using existing PDF database.")


            with st.spinner("Processing query..."):
                answer = prepare_pdf_data(index, chunks, query, k_value)
            if answer:
                st.subheader("Answer:")
                st.write(answer)
        else:
            st.error("Unsupported file type.")
    elif uploaded_file is None and query:
        st.warning("Please upload a file.")
    elif uploaded_file is not None and not query:
         st.warning("Please enter a query.")
    else:
        st.warning("Please upload a file and enter a query.")

# Add a clear button
if st.button("Clear"):
    # Clear the session state and rerun the app to reset
    st.session_state.clear()
    st.rerun()
