import streamlit as st
from typing import Dict, Any, Optional
import tempfile
import os


class PaperUploadComponent:
    """Component for uploading and managing research papers."""
    
    def __init__(self, research_system):
        self.research_system = research_system
    
    def render(self) -> Optional[Dict[str, Any]]:
        """Render the paper upload interface."""
        
        st.subheader("📄 Upload Research Paper")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a research paper in PDF format"
        )
        
        if uploaded_file is not None:
            # Display file information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", uploaded_file.type)
            
            # Processing options
            st.subheader("Processing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                extract_figures = st.checkbox("Extract Figures", value=True)
                extract_tables = st.checkbox("Extract Tables", value=True)
            
            with col2:
                extract_equations = st.checkbox("Extract Equations", value=True)
                generate_summary = st.checkbox("Generate Summary", value=True)
            
            # Process button
            if st.button("📊 Process Document", type="primary", use_container_width=True):
                return self._process_document(
                    uploaded_file,
                    {
                        "extract_figures": extract_figures,
                        "extract_tables": extract_tables,
                        "extract_equations": extract_equations,
                        "generate_summary": generate_summary
                    }
                )
        
        return None
    
    def _process_document(self, uploaded_file, options: Dict[str, bool]) -> Dict[str, Any]:
        """Process the uploaded document."""
        
        with st.spinner("🔄 Processing document..."):
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            try:
                # Process with research system
                # In real implementation: 
                # result = asyncio.run(self.research_system.analyze_document(tmp_path))
                
                # Mock result for demonstration
                result = {
                    "title": "Advanced Multi-Modal Learning for Computer Vision",
                    "abstract": "This paper presents a novel approach to multi-modal learning that combines visual and textual information for improved computer vision tasks...",
                    "authors": ["Dr. Smith", "Dr. Johnson", "Dr. Brown"],
                    "publication_year": 2024,
                    "figures": [
                        {"page": 1, "type": "architecture_diagram", "description": "Model architecture overview"},
                        {"page": 3, "type": "results_chart", "description": "Performance comparison"},
                        {"page": 5, "type": "ablation_study", "description": "Component analysis"}
                    ] if options["extract_figures"] else [],
                    "tables": [
                        {"page": 2, "title": "Dataset Statistics", "rows": 5, "cols": 4},
                        {"page": 4, "title": "Results Comparison", "rows": 8, "cols": 6}
                    ] if options["extract_tables"] else [],
                    "equations": [
                        {"page": 2, "type": "loss_function", "content": "L = -log(p(y|x))"},
                        {"page": 3, "type": "attention", "content": "Attention(Q,K,V) = softmax(QK^T/√d_k)V"}
                    ] if options["extract_equations"] else [],
                    "key_concepts": ["multi-modal learning", "attention mechanisms", "computer vision", "deep learning"],
                    "summary": "The paper introduces a multi-modal learning framework that effectively combines visual and textual information, achieving state-of-the-art results on several benchmark datasets." if options["generate_summary"] else "",
                    "page_count": 12,
                    "processing_time": "2.3 seconds"
                }
                
                st.success("✅ Document processed successfully!")
                
                # Display results
                self._display_results(result)
                
                return result
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    def _display_results(self, result: Dict[str, Any]):
        """Display processing results."""
        
        # Basic information
        st.subheader("📋 Document Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Title:**", result.get("title", "Unknown"))
            st.write("**Authors:**", ", ".join(result.get("authors", [])))
            st.write("**Year:**", result.get("publication_year", "Unknown"))
        
        with col2:
            st.metric("Pages", result.get("page_count", 0))
            st.metric("Figures", len(result.get("figures", [])))
            st.metric("Tables", len(result.get("tables", [])))
            st.metric("Equations", len(result.get("equations", [])))
        
        # Abstract
        if result.get("abstract"):
            with st.expander("📝 Abstract"):
                st.write(result["abstract"])
        
        # Summary
        if result.get("summary"):
            with st.expander("📄 Summary"):
                st.write(result["summary"])
        
        # Figures
        if result.get("figures"):
            with st.expander(f"🖼️ Figures ({len(result['figures'])})"):
                for i, figure in enumerate(result["figures"], 1):
                    st.write(f"**Figure {i}** (Page {figure['page']}): {figure['description']}")
                    st.write(f"Type: {figure['type']}")
                    st.divider()
        
        # Tables
        if result.get("tables"):
            with st.expander(f"📊 Tables ({len(result['tables'])})"):
                for i, table in enumerate(result["tables"], 1):
                    st.write(f"**Table {i}** (Page {table['page']}): {table['title']}")
                    st.write(f"Dimensions: {table['rows']} rows × {table['cols']} columns")
                    st.divider()
        
        # Equations
        if result.get("equations"):
            with st.expander(f"🔢 Equations ({len(result['equations'])})"):
                for i, equation in enumerate(result["equations"], 1):
                    st.write(f"**Equation {i}** (Page {equation['page']}):")
                    st.latex(equation["content"])
                    st.write(f"Type: {equation['type']}")
                    st.divider()
        
        # Key concepts
        if result.get("key_concepts"):
            with st.expander("🏷️ Key Concepts"):
                for concept in result["key_concepts"]:
                    st.write(f"• {concept}")
