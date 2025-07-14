import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
from openai import OpenAI
import os # Import os to access environment variables
import sqlite3
import tempfile

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
            You are a helpful and knowledgeable assistant. When answering questions, use the most relevant information from the provided context. Be concise, avoid speculation, and favor well-supported answers.

            For structured data (like database results), present the information clearly and highlight key findings. When dealing with mixed or contradictory information, acknowledge both sides.

            If asked about specific data points, summarize the most relevant information. If asked for comparisons or analysis, emphasize the differences and similarities based on the data provided.

            Only answer questions using the data you have. Do not generate fictional content or hallucinate details not found in the context.
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

def create_excel_database(uploaded_file):
    """
    Creates a SQLite database from an uploaded Excel file and returns the database connection and schema.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        A tuple containing:
            - sqlite3.Connection: The SQLite database connection.
            - str: The database schema information.
        Returns None if an error occurs.
    """
    try:
        # Read Excel file into pandas DataFrame
        df = pd.read_excel(uploaded_file)
        
        # Create a temporary SQLite database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        conn = sqlite3.connect(temp_db.name)
        
        # Convert DataFrame to SQL table
        df.to_sql('data_table', conn, index=False, if_exists='replace')
        
        # Generate schema information
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(data_table)")
        columns_info = cursor.fetchall()
        
        schema_parts = []
        schema_parts.append("Database Schema:")
        schema_parts.append("Table: data_table")
        schema_parts.append("Columns:")
        
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]
            schema_parts.append(f"  - {col_name} ({col_type})")
        
        # Add sample data
        cursor.execute("SELECT * FROM data_table LIMIT 5")
        sample_data = cursor.fetchall()
        
        schema_parts.append("\nSample Data (first 5 rows):")
        column_names = [desc[0] for desc in cursor.description]
        schema_parts.append(f"Columns: {', '.join(column_names)}")
        
        for i, row in enumerate(sample_data, 1):
            schema_parts.append(f"Row {i}: {row}")
        
        # Add total row count
        cursor.execute("SELECT COUNT(*) FROM data_table")
        total_rows = cursor.fetchone()[0]
        schema_parts.append(f"\nTotal rows in database: {total_rows}")
        
        schema = "\n".join(schema_parts)
        
        return conn, schema
    except Exception as e:
        st.error(f"An error occurred during Excel database creation: {e}")
        return None

def prepare_excel_data(conn: sqlite3.Connection, schema: str, query: str, k: int = 2):
    """
    Prepares Excel data for RAG by executing SQL queries on the database
    and using the results with the RAG function.

    Args:
        conn: The SQLite database connection.
        schema: The database schema information.
        query: The user's natural language query.
        k: Not used for SQL-based approach, kept for compatibility.

    Returns:
        str: The generated answer from the RAG function based on the SQL query results.
    """
    try:
        # First, let the LLM generate an appropriate SQL query
        context = f"""
        You have access to a SQLite database with the following schema:
        
        {schema}
        
        Based on the user's question, you need to write a SQL query to retrieve relevant data.
        Only respond with the SQL query, nothing else. Use proper SQL syntax for SQLite.
        The table name is 'data_table'.
        """

        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        sql_response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate only the SQL query needed to answer the user's question. Do not include any explanations or markdown formatting."},
                {"role": "user", "content": f"{context}\n\nUser question: {query}\n\nSQL query:"}
            ],
            temperature=0.1
        )

        sql_query = sql_response.choices[0].message.content.strip()
        
        # Clean up the SQL query (remove any markdown formatting)
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()

        # Execute the SQL query
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]

        # Format the results for the RAG function
        if results:
            result_text = f"SQL Query: {sql_query}\n\n"
            result_text += f"Columns: {', '.join(column_names)}\n\n"
            result_text += "Results:\n"
            for i, row in enumerate(results, 1):
                row_data = dict(zip(column_names, row))
                result_text += f"Row {i}: {row_data}\n"
        else:
            result_text = f"SQL Query: {sql_query}\n\nNo results found."

        # Use RAG to generate a natural language answer
        return rag([result_text], query)
        
    except Exception as e:
        st.error(f"An error occurred during Excel data preparation: {e}")
        return f"Error executing query: {str(e)}"

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
            # Create the database only once and store it in session state
            if "excel_conn" not in st.session_state or "excel_schema" not in st.session_state or st.session_state.uploaded_excel_name != uploaded_file.name:
                with st.spinner("Creating Excel database..."):
                    result = create_excel_database(uploaded_file)
                if result is not None:
                    conn, schema = result
                    st.session_state.excel_conn = conn
                    st.session_state.excel_schema = schema
                    st.session_state.uploaded_excel_name = uploaded_file.name
                    st.success("Excel database created.")
                else:
                    st.error("Failed to create Excel database. Please try again.")
                    # Clear session state if database creation failed
                    if "excel_conn" in st.session_state:
                        del st.session_state.excel_conn
                    if "excel_schema" in st.session_state:
                        del st.session_state.excel_schema
                    if "uploaded_excel_name" in st.session_state:
                        del st.session_state.uploaded_excel_name
                    st.stop() # Stop execution after showing error
            else:
                conn = st.session_state.excel_conn
                schema = st.session_state.excel_schema
                st.info("Using existing Excel database.")

            with st.spinner("Processing query..."):
                answer = prepare_excel_data(conn, schema, query, k_value)
            if answer:
                st.subheader("Answer:")
                st.write(answer)
        elif file_extension == "pdf":
            # Create the database only once and store it in session state
            if "pdf_index" not in st.session_state or "pdf_chunks" not in st.session_state or st.session_state.uploaded_pdf_name != uploaded_file.name:
                with st.spinner("Creating PDF database..."):
                    result = create_pdf_database(uploaded_file)
                if result is not None:
                    index, chunks = result
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
    # Close database connections if they exist
    if "excel_conn" in st.session_state:
        try:
            st.session_state.excel_conn.close()
        except:
            pass
    # Clear the session state and rerun the app to reset
    st.session_state.clear()
    st.rerun()
