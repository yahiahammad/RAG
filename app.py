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
import re

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


def rag(retrieved_docs, query, conversation_history=None):
    
    context = "\n".join(retrieved_docs)

    client = OpenAI(
        api_key=groq_api_key, # Use the securely loaded API key
        base_url="https://api.groq.com/openai/v1"
    )

    # Build the conversation messages
    messages = [
        {"role": "system", "content": """
        You are a helpful and knowledgeable assistant. When answering questions, use the most relevant information from the provided context. Be concise, avoid speculation, and favor well-supported answers.

        For structured data (like database results), present the information clearly and highlight key findings. When dealing with mixed or contradictory information, acknowledge both sides.

        If asked about specific data points, summarize the most relevant information. If asked for comparisons or analysis, emphasize the differences and similarities based on the data provided.

        Only answer questions using the data you have. Do not generate fictional content or hallucinate details not found in the context.
        
        Consider the conversation history when answering follow-up questions, but always prioritize the current context data.
        """}
    ]

    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current query with context
    messages.append({
        "role": "user", 
        "content": f"""
          Using the information in the following context, answer the question clearly and accurately.

          Context:
          {context}

          Answer the following question:
          {query}

          If the answer cannot be found in the context, say The context does not contain enough information to answer this question. then explain why it does not contain enough information.
        """
    })

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
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
        
        # Clean column names to avoid SQL issues
        df.columns = df.columns.astype(str)  # Convert to string
        df.columns = [col.strip().replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]
        
        # Remove any special characters that might cause SQL issues
        import re
        df.columns = [re.sub(r'[^\w_]', '_', col) for col in df.columns]
        
        # Ensure column names don't start with numbers
        df.columns = [f"col_{col}" if col[0].isdigit() else col for col in df.columns]
        
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
        schema_parts.append("=== DATABASE SCHEMA ===")
        schema_parts.append("Table: data_table")
        schema_parts.append("\nColumn Information:")
        
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]
            not_null = "NOT NULL" if col_info[3] else "NULLABLE"
            schema_parts.append(f"  • {col_name}: {col_type} ({not_null})")
        
        # Add sample data with more detail
        cursor.execute("SELECT * FROM data_table LIMIT 3")
        sample_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        schema_parts.append("\n=== SAMPLE DATA ===")
        schema_parts.append(f"Columns: {', '.join(column_names)}")
        
        for i, row in enumerate(sample_data, 1):
            row_dict = dict(zip(column_names, row))
            schema_parts.append(f"Sample Row {i}: {row_dict}")
        
        # Add data type examples and unique values for categorical columns
        schema_parts.append("\n=== DATA INSIGHTS ===")
        
        for col_name in column_names:
            # Get unique value count - use quoted column names to handle special cases
            cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM data_table')
            unique_count = cursor.fetchone()[0]
            
            # Get null count
            cursor.execute(f'SELECT COUNT(*) FROM data_table WHERE "{col_name}" IS NULL')
            null_count = cursor.fetchone()[0]
            
            schema_parts.append(f"  • {col_name}: {unique_count} unique values, {null_count} nulls")
            
            # If it's a categorical column (low unique count), show sample values
            if unique_count <= 10 and unique_count > 1:
                cursor.execute(f'SELECT DISTINCT "{col_name}" FROM data_table WHERE "{col_name}" IS NOT NULL LIMIT 5')
                unique_values = [str(row[0]) for row in cursor.fetchall()]
                schema_parts.append(f"    Sample values: {', '.join(unique_values)}")
        
        # Add total row count
        cursor.execute("SELECT COUNT(*) FROM data_table")
        total_rows = cursor.fetchone()[0]
        schema_parts.append(f"\n=== SUMMARY ===")
        schema_parts.append(f"Total rows: {total_rows}")
        schema_parts.append(f"Total columns: {len(column_names)}")
        
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
        # Generate SQL query using LLM
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        sql_response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": """You are an expert SQL developer specializing in SQLite. Your task is to generate precise, efficient SQL queries.

CRITICAL INSTRUCTIONS:
- Return ONLY the SQL query, nothing else
- No explanations, comments, or markdown formatting
- Use proper SQLite syntax and functions
- The table name is always 'data_table'
- ALWAYS use double quotes around column names to handle special characters (e.g., "column_name")
- Use appropriate SQLite data types and functions
- For text searches, use LIKE with wildcards (%) when appropriate
- For numerical operations, use proper arithmetic operators
- For date/time operations, use SQLite date functions if needed

QUERY REQUIREMENTS:
- Write efficient queries that directly answer the user's question
- Use appropriate WHERE clauses for filtering
- Use GROUP BY and aggregate functions (COUNT, SUM, AVG, MAX, MIN) when needed
- Use ORDER BY for sorting results when relevant
- Use LIMIT only if the user specifically asks for a limited number of results
- Handle NULL values appropriately
- Remember to quote ALL column names with double quotes

EXAMPLES:
- SELECT "Name", "Age" FROM data_table WHERE "Age" > 25
- SELECT COUNT(*) FROM data_table WHERE "Status" = 'Active'
- SELECT "Category", AVG("Price") FROM data_table GROUP BY "Category"
"""},
                {"role": "user", "content": f"""Database Schema and Context:
{schema}

User Question: {query}

Generate the SQL query:"""}
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
        conversation_history = st.session_state.get('conversation_history', [])
        answer = rag([result_text], query, conversation_history)
        
        # Update conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        st.session_state.conversation_history.append({"role": "user", "content": query})
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
        
        # Keep only the last 10 exchanges (20 messages) to prevent context from getting too long
        if len(st.session_state.conversation_history) > 20:
            st.session_state.conversation_history = st.session_state.conversation_history[-20:]
        
        return answer
        
    except Exception as e:
        st.error(f"An error occurred during Excel data preparation: {e}")
        return f"Error executing query: {str(e)}"

def create_pdf_database(uploaded_file):
    
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
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])

        D, I = index.search(query_embedding, k=k)
        retrieved_docs = [chunks[i].page_content for i in I[0]]

        conversation_history = st.session_state.get('conversation_history', [])
        answer = rag(retrieved_docs, query, conversation_history)
        
        # Update conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        st.session_state.conversation_history.append({"role": "user", "content": query})
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
        
        # Keep only the last 10 exchanges (20 messages) to prevent context from getting too long
        if len(st.session_state.conversation_history) > 20:
            st.session_state.conversation_history = st.session_state.conversation_history[-20:]
        
        return answer
    except Exception as e:
        st.error(f"An error occurred during PDF data preparation: {e}")
        return None


st.title("RAG Application")

st.write("Upload a document (Excel or PDF) and ask questions about its content.")

# Display conversation history if it exists
if 'conversation_history' in st.session_state and st.session_state.conversation_history:
    with st.expander("💬 Conversation History", expanded=False):
        for i in range(0, len(st.session_state.conversation_history), 2):
            if i + 1 < len(st.session_state.conversation_history):
                user_msg = st.session_state.conversation_history[i]
                assistant_msg = st.session_state.conversation_history[i + 1]
                
                st.write(f"**Q{(i//2)+1}:** {user_msg['content']}")
                st.write(f"**A{(i//2)+1}:** {assistant_msg['content']}")
                st.write("---")

# Layout using columns
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "pdf"])
    query = st.text_input("Enter your query:")

with col2:
    k_value = st.slider("Number of relevant documents to retrieve (k)", 1, 10, 2)
    
    # Add a button to clear conversation history
    if st.button("Clear Chat History"):
        if 'conversation_history' in st.session_state:
            st.session_state.conversation_history = []
            st.success("Chat history cleared!")
            st.rerun()

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
