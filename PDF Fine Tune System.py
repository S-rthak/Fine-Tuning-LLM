# main.py - Streamlit Cloud Optimized RAG System
import streamlit as st
import os
import warnings
import gc
warnings.filterwarnings('ignore')

# Import only essential libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from typing import List, Dict, Any
import io
from pathlib import Path
import time
import re

# Try to import optional heavy dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ sentence-transformers not available. Using TF-IDF fallback.")

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# Page config
st.set_page_config(
    page_title="Lightweight RAG Q&A System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentProcessor:
    """Handle document loading and text extraction"""
    
    def __init__(self):
        self.supported_formats = []
        if PDF_SUPPORT:
            self.supported_formats.append('.pdf')
        if DOCX_SUPPORT:
            self.supported_formats.append('.docx')
        self.supported_formats.append('.txt')
    
    def load_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF file bytes"""
        if not PDF_SUPPORT:
            raise ValueError("PDF support not available")
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def load_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX file bytes"""
        if not DOCX_SUPPORT:
            raise ValueError("DOCX support not available")
        try:
            doc = Document(io.BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def load_txt(self, file_bytes: bytes) -> str:
        """Load text from TXT file bytes"""
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_bytes.decode('latin-1')
            except Exception as e:
                st.error(f"Error reading TXT file: {str(e)}")
                return ""
    
    def load_document(self, file_name: str, file_bytes: bytes) -> str:
        """Load document based on file extension"""
        extension = Path(file_name).suffix.lower()
        
        if extension == '.pdf' and PDF_SUPPORT:
            return self.load_pdf(file_bytes)
        elif extension == '.docx' and DOCX_SUPPORT:
            return self.load_docx(file_bytes)
        elif extension == '.txt':
            return self.load_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

class TextChunker:
    """Split documents into manageable chunks"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk.split()) + len(sentence.split()) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    words = current_chunk.split()
                    if len(words) > self.overlap:
                        current_chunk = ' '.join(words[-self.overlap:]) + ' ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += '. ' + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.split()) > 5]
        return chunks

@st.cache_resource
def load_embedding_model():
    """Load embedding model with fallback options"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Try lightweight models first
        models_to_try = [
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2",
            "all-MiniLM-L12-v2"
        ]
        
        for model_name in models_to_try:
            try:
                model = SentenceTransformer(model_name)
                st.success(f"✅ Loaded embedding model: {model_name}")
                return model
            except Exception as e:
                st.warning(f"Failed to load {model_name}: {str(e)}")
                continue
        
        return None
    except Exception as e:
        st.error(f"Failed to load any embedding model: {str(e)}")
        return None

class LightweightVectorStore:
    """Lightweight vector database using TF-IDF or sentence transformers"""
    
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        if not self.use_embeddings:
            # Fallback to TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = None
    
    def clear(self):
        """Clear all stored data"""
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.tfidf_matrix = None
        # Force garbage collection
        gc.collect()
    
    def add_documents(self, texts: List[str], embeddings=None, metadatas: List[Dict] = None):
        """Add documents to the vector store"""
        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(texts))]
        
        # Generate IDs
        new_ids = [f"doc_{len(self.ids) + i}_{int(time.time())}" for i in range(len(texts))]
        
        if self.use_embeddings and embeddings is not None:
            # Use embeddings
            if len(self.embeddings) == 0:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
        else:
            # Use TF-IDF
            all_docs = self.documents + texts
            self.tfidf_matrix = self.vectorizer.fit_transform(all_docs)
        
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(new_ids)
        
        return len(texts)
    
    def similarity_search(self, query: str, query_embedding=None, k: int = 5) -> Dict[str, Any]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        if self.use_embeddings and query_embedding is not None:
            # Use embedding similarity
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        else:
            # Use TF-IDF similarity
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return results
        results = {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]],
            "distances": [[1 - similarities[i] for i in top_indices]]
        }
        
        return results
    
    def count(self) -> int:
        """Return number of documents"""
        return len(self.documents)

class LightweightRAGPipeline:
    """Lightweight RAG pipeline optimized for Streamlit Cloud"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.text_chunker = TextChunker(chunk_size=400, overlap=50)
        self.embedding_model = load_embedding_model()
        self.vector_store = LightweightVectorStore(use_embeddings=(self.embedding_model is not None))
    
    def process_uploaded_files(self, uploaded_files) -> int:
        """Process uploaded files and add to vector store"""
        if not uploaded_files:
            return 0
        
        # Clear previous documents
        self.vector_store.clear()
        
        all_chunks = []
        all_metadatas = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {uploaded_file.name}")
            
            # Check file size (limit for Streamlit Cloud)
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                st.warning(f"File {uploaded_file.name} is too large (>10MB). Skipping.")
                continue
            
            # Read file bytes
            file_bytes = uploaded_file.read()
            
            # Extract text
            try:
                text = self.doc_processor.load_document(uploaded_file.name, file_bytes)
                if not text.strip():
                    st.warning(f"No text extracted from {uploaded_file.name}")
                    continue
                
                chunks = self.text_chunker.chunk_text(text)
                
                if not chunks:
                    st.warning(f"No valid chunks created from {uploaded_file.name}")
                    continue
                
                # Limit chunks per file to avoid memory issues
                if len(chunks) > 100:
                    chunks = chunks[:100]
                    st.info(f"Limited {uploaded_file.name} to first 100 chunks")
                
                # Create metadata
                metadatas = [
                    {
                        "source": uploaded_file.name,
                        "chunk_id": j,
                        "file_type": Path(uploaded_file.name).suffix.lower(),
                    } 
                    for j in range(len(chunks))
                ]
                
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                
                st.success(f"✅ {uploaded_file.name}: {len(chunks)} chunks")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_chunks:
            status_text.text("Creating search index...")
            
            try:
                if self.embedding_model is not None:
                    # Generate embeddings in small batches
                    batch_size = 16
                    total_chunks = 0
                    
                    for i in range(0, len(all_chunks), batch_size):
                        batch_chunks = all_chunks[i:i + batch_size]
                        batch_metadatas = all_metadatas[i:i + batch_size]
                        
                        embeddings = self.embedding_model.encode(
                            batch_chunks, 
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        
                        added = self.vector_store.add_documents(
                            batch_chunks, embeddings, batch_metadatas
                        )
                        total_chunks += added
                        
                        progress_bar.progress(min(1.0, (i + batch_size) / len(all_chunks)))
                        
                        # Force garbage collection
                        if i % (batch_size * 4) == 0:
                            gc.collect()
                else:
                    # Use TF-IDF
                    total_chunks = self.vector_store.add_documents(
                        all_chunks, None, all_metadatas
                    )
                
                status_text.text(f"✅ Ready! {total_chunks} chunks indexed")
                return total_chunks
                
            except Exception as e:
                st.error(f"Error creating search index: {str(e)}")
                return 0
        
        return 0
    
    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant context for a query"""
        try:
            if self.embedding_model is not None:
                query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
                results = self.vector_store.similarity_search(query, query_embedding, k=k)
            else:
                results = self.vector_store.similarity_search(query, None, k=k)
                
            contexts = results['documents'][0] if results['documents'] else []
            return contexts
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using retrieved context"""
        if not context:
            return "No relevant information found in your documents."
        
        # Simple answer generation
        context_text = "\n\n".join(context[:2])  # Use top 2 contexts
        
        answer = f"""**Based on your documents:**

{context_text}

---

*The above information was found in your uploaded documents and is most relevant to your question: "{query}"*

**💡 Tip:** Try asking more specific questions for better results!"""
        
        return answer
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Complete RAG pipeline query"""
        # Retrieve context
        context = self.retrieve_context(question, k=k)
        
        if not context:
            return {
                "question": question,
                "answer": "No relevant context found. Please upload some documents first.",
                "context": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context": context
        }

def main():
    st.title("🚀 Lightweight RAG Q&A System")
    st.markdown("*Optimized for Streamlit Cloud deployment*")
    
    # Show system capabilities
    with st.expander("💡 System Information"):
        st.write("**Available Features:**")
        st.write(f"📄 PDF Support: {'✅' if PDF_SUPPORT else '❌'}")
        st.write(f"📝 DOCX Support: {'✅' if DOCX_SUPPORT else '❌'}")
        st.write(f"📃 TXT Support: ✅")
        st.write(f"🧠 Semantic Search: {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌ (using TF-IDF fallback)'}")
        st.write(f"💾 Memory Optimized: ✅")
        
        if not PDF_SUPPORT:
            st.info("💡 To enable PDF support, add `PyPDF2` to requirements.txt")
        if not DOCX_SUPPORT:
            st.info("💡 To enable DOCX support, add `python-docx` to requirements.txt")
    
    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.rag_pipeline = LightweightRAGPipeline()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize document count
    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        # Show supported formats
        supported_formats = st.session_state.rag_pipeline.doc_processor.supported_formats
        st.info(f"Supported: {', '.join(supported_formats)}")
        
        uploaded_files = st.file_uploader(
            "Choose files (max 10MB each)",
            type=[fmt[1:] for fmt in supported_formats],  # Remove dots
            accept_multiple_files=True,
            help="Upload documents to analyze"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents", type="primary"):
                with st.spinner("Processing..."):
                    chunk_count = st.session_state.rag_pipeline.process_uploaded_files(uploaded_files)
                    st.session_state.document_count = chunk_count
                    if chunk_count > 0:
                        st.success(f"✅ Ready! {chunk_count} chunks")
                        st.session_state.messages = []
                        st.rerun()
        
        # Status
        if st.session_state.document_count > 0:
            st.success(f"📊 {st.session_state.document_count} chunks indexed")
            
            # Show documents
            vector_store = st.session_state.rag_pipeline.vector_store
            if hasattr(vector_store, 'metadatas') and vector_store.metadatas:
                sources = list(set([meta.get('source', 'Unknown') for meta in vector_store.metadatas]))
                with st.expander("📚 Uploaded Documents"):
                    for source in sources:
                        st.write(f"• {source}")
        else:
            st.info("Upload documents to get started")
        
        # Settings
        st.header("⚙️ Settings")
        max_results = st.slider("Results per query", 1, 5, 3)
        
        # Clear options
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🗑️ Clear All"):
            if hasattr(st.session_state, 'rag_pipeline'):
                st.session_state.rag_pipeline.vector_store.clear()
            st.session_state.document_count = 0
            st.session_state.messages = []
            st.success("All data cleared!")
            st.rerun()
    
    # Main interface
    st.header("💬 Ask Questions")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "context" in message:
                with st.expander("📖 Source Excerpts"):
                    for i, ctx in enumerate(message["context"][:2]):
                        st.write(f"**Excerpt {i+1}:**")
                        st.write(ctx[:300] + "..." if len(ctx) > 300 else ctx)
                        if i < len(message["context"][:2]) - 1:
                            st.write("---")
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        if st.session_state.document_count == 0:
            st.error("Please upload documents first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    result = st.session_state.rag_pipeline.query(prompt, k=max_results)
                    
                    st.markdown(result["answer"])
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "context": result["context"]
                    })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 **Lightweight RAG System** | "
        "🔍 Semantic Search | "
        "☁️ Streamlit Cloud Optimized | "
        "🔒 Privacy-First"
    )

if __name__ == "__main__":
    main()
