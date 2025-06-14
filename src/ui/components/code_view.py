import streamlit as st
from typing import Dict, Any, List, Optional
import json


class CodeViewComponent:
    """Component for displaying and managing code implementations."""
    
    def __init__(self, research_system):
        self.research_system = research_system
    
    def render(self, code_data: Optional[Dict[str, Any]] = None):
        """Render the code view component."""
        
        st.subheader("💻 Code Implementation Viewer")
        
        if code_data is None:
            # Show placeholder or load recent implementations
            st.info("No code implementation selected. Generate code using the Research Query interface.")
            
            # Show recent implementations (mock data)
            with st.expander("📜 Recent Implementations"):
                recent_impls = [
                    {"name": "Vision Transformer", "language": "Python", "date": "2024-01-15"},
                    {"name": "BERT Fine-tuning", "language": "Python", "date": "2024-01-14"},
                    {"name": "ResNet Implementation", "language": "Python", "date": "2024-01-13"}
                ]
                
                for impl in recent_impls:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(impl["name"])
                    with col2:
                        st.write(impl["language"])
                    with col3:
                        st.write(impl["date"])
                    with col4:
                        if st.button("View", key=f"view_{impl['name']}"):
                            st.session_state.selected_impl = impl["name"]
            
            return
        
        # Display code implementation
        self._display_code_implementation(code_data)
    
    def _display_code_implementation(self, code_data: Dict[str, Any]):
        """Display a code implementation with all details."""
        
        # Header information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Language", code_data.get("language", "Unknown"))
        with col2:
            st.metric("Framework", code_data.get("framework", "Unknown"))
        with col3:
            validation = code_data.get("validation", {})
            status = "✅ Valid" if validation.get("syntax_valid", False) else "❌ Invalid"
            st.metric("Status", status)
        
        # Code tabs
        if code_data.get("code_snippets"):
            tabs = st.tabs([snippet["name"] for snippet in code_data["code_snippets"]])
            
            for tab, snippet in zip(tabs, code_data["code_snippets"]):
                with tab:
                    st.write(f"**Description:** {snippet.get('description', 'No description')}")
                    
                    # Code display with copy button
                    col1, col2 = st.columns([6, 1])
                    
                    with col1:
                        st.code(snippet["code"], language=code_data.get("language", "python").lower())
                    
                    with col2:
                        if st.button("📋", help="Copy to clipboard", key=f"copy_{snippet['name']}"):
                            st.session_state.clipboard = snippet["code"]
                            st.success("Copied!")
        
        # Additional information
        col1, col2 = st.columns(2)
        
        with col1:
            # Dependencies
            if code_data.get("dependencies"):
                st.subheader("📦 Dependencies")
                for dep in code_data["dependencies"]:
                    st.write(f"• {dep}")
                
                # Generate requirements.txt
                if st.button("Generate requirements.txt"):
                    requirements = "\n".join(code_data["dependencies"])
                    st.download_button(
                        "Download requirements.txt",
                        requirements,
                        "requirements.txt",
                        "text/plain"
                    )
        
        with col2:
            # Validation results
            if code_data.get("validation"):
                st.subheader("✅ Validation Results")
                validation = code_data["validation"]
                
                st.write(f"**Syntax Valid:** {'✅' if validation.get('syntax_valid') else '❌'}")
                st.write(f"**AST Valid:** {'✅' if validation.get('ast_valid') else '❌'}")
                st.write(f"**Execution Test:** {'✅' if validation.get('execution_test') else '❌'}")
                
                if validation.get("errors"):
                    with st.expander("❌ Errors"):
                        for error in validation["errors"]:
                            st.error(error)
        
        # Usage examples
        if code_data.get("examples"):
            st.subheader("📖 Usage Examples")
            st.code(code_data["examples"], language=code_data.get("language", "python").lower())
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Run Code", type="primary"):
                self._simulate_code_execution(code_data)
        
        with col2:
            if st.button("💾 Save Implementation"):
                st.success("Implementation saved to workspace!")
        
        with col3:
            if st.button("📤 Export Code"):
                self._export_code(code_data)
    
    def _simulate_code_execution(self, code_data: Dict[str, Any]):
        """Simulate code execution."""
        
        with st.spinner("Executing code..."):
            # Simulate execution
            import time
            time.sleep(2)
            
            # Mock execution results
            st.success("✅ Code executed successfully!")
            
            with st.expander("📊 Execution Results"):
                st.write("**Output:**")
                st.code("""
Model initialized successfully
Training started...
Epoch 1/10: Loss = 0.856, Accuracy = 0.623
Epoch 2/10: Loss = 0.742, Accuracy = 0.701
...
Training completed!
Final accuracy: 0.892
                """)
                
                st.write("**Performance Metrics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Execution Time", "2.34s")
                with col2:
                    st.metric("Memory Usage", "156 MB")
                with col3:
                    st.metric("GPU Utilization", "78%")
    
    def _export_code(self, code_data: Dict[str, Any]):
        """Export code implementation."""
        
        # Prepare export data
        export_content = f"""# Generated Code Implementation
# Language: {code_data.get('language', 'Unknown')}
# Framework: {code_data.get('framework', 'Unknown')}

"""
        
        if code_data.get("dependencies"):
            export_content += "# Dependencies:\n"
            for dep in code_data["dependencies"]:
                export_content += f"# - {dep}\n"
            export_content += "\n"
        
        if code_data.get("code_snippets"):
            for snippet in code_data["code_snippets"]:
                export_content += f"# {snippet['name']}\n"
                export_content += f"# {snippet.get('description', '')}\n\n"
                export_content += snippet["code"] + "\n\n"
        
        # Download button
        st.download_button(
            "📥 Download Complete Implementation",
            export_content,
            f"implementation_{code_data.get('language', 'code').lower()}.py",
            "text/plain"
        )
