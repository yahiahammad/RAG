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
from datetime import datetime

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

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# CSS for chat interface
st.markdown("""
<style>
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 10px;
    background-color: #f9f9f9;
    margin-bottom: 20px;
}

.user-message {
    background-color: #007bff;
    color: white;
    padding: 10px 15px;
    border-radius: 18px;
    margin: 5px 0;
    max-width: 70%;
    margin-left: auto;
    margin-right: 0;
    display: block;
    text-align: right;
}

.assistant-message {
    background-color: #e9ecef;
    color: #333;
    padding: 10px 15px;
    border-radius: 18px;
    margin: 5px 0;
    max-width: 70%;
    margin-left: 0;
    margin-right: auto;
    display: block;
}

.timestamp {
    font-size: 0.8em;
    color: #666;
    margin: 2px 0;
}

.message-wrapper {
    margin: 10px 0;
}

.user-wrapper {
    text-align: right;
}

.assistant-wrapper {
    text-align: left;
}
</style>
""", unsafe_allow_html=True)


def rag(retrieved_docs, query, conversation_history=None):
    """
    Performs Retrieval-Augmented Generation (RAG) using retrieved document chunks
    and a user query to generate an answer, with optional conversation history.

    Args:
        retrieved_docs: A list of strings containing the content of retrieved document chunks.
        query: The user's query string.
        conversation_history: List of previous conversation messages.

    Returns:
        str: The generated answer from the language model.
    """
    context = "\n".join(retrieved_docs)

    client = OpenAI(
        api_key=groq_api_key, # Use the securely loaded API key
        base_url="https://api.groq.com/openai/v1"
    )

    # Build conversation context
    messages = [
        {"role": "system", "content": """
        You are a helpful and knowledgeable assistant. When answering questions, use the most relevant information from the provided context. Be concise, avoid speculation, and favor well-supported answers.

        For structured data (like database results), present the information clearly and highlight key findings. When dealing with mixed or contradictory information, acknowledge both sides.

        If asked about specific data points, summarize the most relevant information. If asked for comparisons or analysis, emphasize the differences and similarities based on the data provided.

        Only answer questions using the data you have. Do not generate fictional content or hallucinate details not found in the context.
        
        When the user asks follow-up questions, consider the conversation history to provide contextually relevant answers.
        """}
    ]
    
    # Add conversation history if available
    if conversation_history:
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append({"role": "assistant", "content": msg["content"]})
    
    # Add current query with context
    messages.append({"role": "user", "content": f"""
              Using the information in the following context, answer the question clearly and accurately.

              Context:
              {context}

              Answer the following question:
              {query}

              If the answer cannot be found in the context, say The context does not contain enough information to answer this question. then explain why it does not contain enough information.
            """})

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
        
        # Store original column names for reference
        original_columns = df.columns.tolist()
        
        # Clean column names to avoid SQL issues
        # Replace spaces with underscores and remove special characters
        df.columns = df.columns.str.replace(' ', '_', regex=False)
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # Handle empty column names
        df.columns = ['col_' + str(i) if col == '' else col for i, col in enumerate(df.columns)]
        
        # Ensure column names don't start with numbers
        df.columns = ['col_' + col if col and col[0].isdigit() else col for col in df.columns]
        
        # Handle reserved SQL keywords by prefixing with 'col_'
        sql_keywords = {'name', 'order', 'group', 'select', 'from', 'where', 'table', 'column', 'index', 'key', 'primary', 'foreign', 'unique', 'null', 'not', 'and', 'or', 'in', 'is', 'like', 'between', 'exists', 'all', 'any', 'some', 'union', 'intersect', 'except', 'join', 'inner', 'outer', 'left', 'right', 'full', 'cross', 'natural', 'on', 'using', 'case', 'when', 'then', 'else', 'end', 'if', 'count', 'sum', 'avg', 'min', 'max', 'distinct', 'having', 'limit', 'offset', 'create', 'alter', 'drop', 'insert', 'update', 'delete', 'truncate', 'begin', 'commit', 'rollback', 'transaction', 'savepoint', 'release', 'grant', 'revoke', 'references', 'constraint', 'check', 'default', 'collate', 'autoincrement', 'temporary', 'temp', 'view', 'trigger', 'procedure', 'function', 'database', 'schema', 'pragma', 'explain', 'analyze', 'vacuum', 'reindex', 'attach', 'detach', 'conflict', 'fail', 'ignore', 'replace', 'abort', 'rollback', 'cascade', 'restrict', 'set', 'immediate', 'deferred', 'exclusive', 'shared', 'reserved', 'pending', 'unlocked'}
        df.columns = ['col_' + col if col.lower() in sql_keywords else col for col in df.columns]
        
        # Ensure all column names are unique
        seen = set()
        new_columns = []
        for col in df.columns:
            original_col = col
            counter = 1
            while col in seen:
                col = f"{original_col}_{counter}"
                counter += 1
            seen.add(col)
            new_columns.append(col)
        df.columns = new_columns
        
        # Create a temporary SQLite database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        conn = sqlite3.connect(temp_db.name)
        
        # Debug: Show the column transformation
        st.info(f"Debug: Original columns: {original_columns}")
        st.info(f"Debug: Cleaned columns: {df.columns.tolist()}")
        
        # Convert DataFrame to SQL table
        try:
            df.to_sql('data_table', conn, index=False, if_exists='replace')
        except Exception as sql_error:
            st.warning(f"Initial database creation failed: {sql_error}")
            st.info("Attempting with safer column names...")
            
            # Fallback: Use generic column names
            df.columns = [f'column_{i}' for i in range(len(df.columns))]
            try:
                df.to_sql('data_table', conn, index=False, if_exists='replace')
                st.success("Database created with generic column names")
            except Exception as fallback_error:
                st.error(f"Fallback database creation also failed: {fallback_error}")
                return None
        
        # Test the database by running a simple query
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM data_table")
            row_count = cursor.fetchone()[0]
            st.success(f"Database created successfully with {row_count} rows")
        except Exception as test_error:
            st.error(f"Database creation test failed: {test_error}")
            return None
        
        # Generate schema information
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(data_table)")
        columns_info = cursor.fetchall()
        
        schema_parts = []
        schema_parts.append("=== DATABASE SCHEMA ===")
        schema_parts.append("Table: data_table")
        schema_parts.append("\nColumn Information:")
        schema_parts.append("(Note: Column names have been cleaned for SQL compatibility)")
        
        for i, col_info in enumerate(columns_info):
            col_name = col_info[1]
            col_type = col_info[2]
            not_null = "NOT NULL" if col_info[3] else "NULLABLE"
            original_name = original_columns[i] if i < len(original_columns) else col_name
            
            # Show different info based on whether we used generic names
            if col_name.startswith('column_'):
                schema_parts.append(f'  • "{col_name}" (originally: "{original_name}"): {col_type} ({not_null})')
            else:
                schema_parts.append(f'  • "{col_name}" (originally: "{original_name}"): {col_type} ({not_null})')
        
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
            try:
                # Get unique value count with proper escaping
                cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM data_table')
                unique_count = cursor.fetchone()[0]
                
                # Get null count with proper escaping
                cursor.execute(f'SELECT COUNT(*) FROM data_table WHERE "{col_name}" IS NULL')
                null_count = cursor.fetchone()[0]
                
                schema_parts.append(f"  • {col_name}: {unique_count} unique values, {null_count} nulls")
                
                # If it's a categorical column (low unique count), show sample values
                if unique_count <= 10 and unique_count > 1:
                    cursor.execute(f'SELECT DISTINCT "{col_name}" FROM data_table WHERE "{col_name}" IS NOT NULL LIMIT 5')
                    unique_values = [str(row[0]) for row in cursor.fetchall()]
                    schema_parts.append(f"    Sample values: {', '.join(unique_values)}")
            except Exception as col_error:
                schema_parts.append(f"  • {col_name}: Error analyzing column - {str(col_error)}")
                # Continue with other columns even if one fails
        
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
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
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
- ALWAYS wrap column names in double quotes to handle special characters and reserved words
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
- Always use double quotes around column names like "column_name"

EXAMPLE:
Instead of: SELECT Name, Age FROM data_table
Use: SELECT "Name", "Age" FROM data_table"""},
                {"role": "user", "content": f"""Database Schema and Context:
{schema}

User Question: {query}

Generate the SQL query (remember to wrap all column names in double quotes):"""}
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
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
        except sqlite3.Error as sql_error:
            st.error(f"SQL execution error: {sql_error}")
            st.error(f"Generated SQL query: {sql_query}")
            return f"Error executing SQL query: {str(sql_error)}"

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
        return rag([result_text], query, st.session_state.conversation_history)
        
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

        return rag(retrieved_docs, query, st.session_state.conversation_history)
    except Exception as e:
        st.error(f"An error occurred during PDF data preparation: {e}")
        return None

def add_to_conversation(role, content):
    """Add a message to the conversation history."""
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })

def display_conversation():
    """Display the conversation history in a chat-like interface."""
    if st.session_state.conversation_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="message-wrapper user-wrapper">
                    <div class="timestamp">{message["timestamp"]}</div>
                    <div class="user-message">{message["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="message-wrapper assistant-wrapper">
                    <div class="timestamp">{message["timestamp"]}</div>
                    <div class="assistant-message">{message["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No conversation history yet. Ask your first question!")

def clear_conversation():
    """Clear the conversation history."""
    st.session_state.conversation_history = []


st.title("🤖 RAG Chat Application")

st.write("Upload a document (Excel or PDF) and have a conversation about its content.")

# File upload section
uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "pdf"])

# Settings sidebar
with st.sidebar:
    st.header("Settings")
    k_value = st.slider("Number of relevant documents to retrieve (k)", 1, 10, 2)
    
    if st.button("🗑️ Clear Conversation"):
        clear_conversation()
        st.rerun()
    
    if st.button("🔄 Clear All Data"):
        # Close database connections if they exist
        if "excel_conn" in st.session_state:
            try:
                st.session_state.excel_conn.close()
            except:
                pass
        # Clear the session state and rerun the app to reset
        st.session_state.clear()
        st.rerun()

# Display conversation history
st.subheader("💬 Conversation")
display_conversation()

# Chat input section
if uploaded_file is not None:
    # Show file info
    file_extension = uploaded_file.name.split(".")[-1].lower()
    st.info(f"📄 File loaded: {uploaded_file.name} ({file_extension.upper()})")
    
    # Text input for queries
    query = st.text_input("Ask a question about your document:", key="query_input", placeholder="Type your question here...")
    
    # Process query when Enter is pressed or button is clicked
    if st.button("Send 📤") or query:
        if query.strip():
            # Add user message to conversation
            add_to_conversation("user", query)
            
            # Process the query based on file type
            with st.spinner("🔍 Processing your question..."):
                answer = None
                
                if file_extension == "xlsx":
                    # Create the database only once and store it in session state
                    if "excel_conn" not in st.session_state or "excel_schema" not in st.session_state or st.session_state.get("uploaded_excel_name") != uploaded_file.name:
                        with st.spinner("📊 Creating Excel database..."):
                            result = create_excel_database(uploaded_file)
                        if result is not None:
                            conn, schema = result
                            st.session_state.excel_conn = conn
                            st.session_state.excel_schema = schema
                            st.session_state.uploaded_excel_name = uploaded_file.name
                        else:
                            st.error("Failed to create Excel database. Please try again.")
                            st.stop()
                    else:
                        conn = st.session_state.excel_conn
                        schema = st.session_state.excel_schema
                    
                    answer = prepare_excel_data(conn, schema, query, k_value)
                    
                elif file_extension == "pdf":
                    # Create the database only once and store it in session state
                    if "pdf_index" not in st.session_state or "pdf_chunks" not in st.session_state or st.session_state.get("uploaded_pdf_name") != uploaded_file.name:
                        with st.spinner("📑 Creating PDF database..."):
                            result = create_pdf_database(uploaded_file)
                        if result is not None:
                            index, chunks = result
                            st.session_state.pdf_index = index
                            st.session_state.pdf_chunks = chunks
                            st.session_state.uploaded_pdf_name = uploaded_file.name
                        else:
                            st.error("Failed to create PDF database. Please try again.")
                            st.stop()
                    else:
                        index = st.session_state.pdf_index
                        chunks = st.session_state.pdf_chunks
                    
                    answer = prepare_pdf_data(index, chunks, query, k_value)
                
                else:
                    st.error("Unsupported file type.")
                    answer = None
            
            # Add assistant response to conversation
            if answer:
                add_to_conversation("assistant", answer)
                st.rerun()  # Refresh to show new messages
            else:
                st.error("Failed to get an answer. Please try again.")
        else:
            st.warning("Please enter a question.")
else:
    st.info("👆 Please upload a file to start the conversation.")

# Add some helpful tips
with st.expander("💡 Tips for better conversations"):
    st.markdown("""
    - **Follow-up questions**: Ask related questions to dive deeper into the content
    - **Be specific**: The more specific your question, the better the answer
    - **Reference previous answers**: You can refer to previous parts of the conversation
    - **Excel files**: Ask about data analysis, calculations, trends, and comparisons
    - **PDF files**: Ask about specific content, summaries, and detailed explanations
    - **Use natural language**: Ask questions as you would to a human assistant
    """)

# Display file status
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "xlsx" and "excel_conn" in st.session_state:
        st.success("✅ Excel database ready for queries")
    elif file_extension == "pdf" and "pdf_index" in st.session_state:
        st.success("✅ PDF database ready for queries")
    else:
        st.info("⏳ Database will be created when you ask your first question")
