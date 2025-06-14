import streamlit as st
import asyncio
import json
from typing import Dict, Any, Optional
import base64
from PIL import Image
import io
import pandas as pd

# Import main system components
import sys
sys.path.append('..')


class ResearchApp:
    """Main Streamlit application for the research assistant."""
    
    def __init__(self, research_system):
        self.research_system = research_system
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Multi-Agent Research Assistant",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the Streamlit application."""
        
        # Main title
        st.title("🔬 Multi-Agent Research Assistant")
        st.markdown("Advanced research analysis powered by specialized AI agents")
        
        # Sidebar for navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox(
                "Choose a page",
                ["Home", "Document Analysis", "Research Query", "Agent Status", "Visualization Tools"]
            )
        
        # Route to appropriate page
        if page == "Home":
            self.home_page()
        elif page == "Document Analysis":
            self.document_analysis_page()
        elif page == "Research Query":
            self.research_query_page()
        elif page == "Agent Status":
            self.agent_status_page()
        elif page == "Visualization Tools":
            self.visualization_page()
    
    def home_page(self):
        """Display the home page."""
        
        st.header("Welcome to the Multi-Agent Research Assistant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 What can this system do?")
            st.markdown("""
            - **Document Analysis**: Upload and analyze research papers
            - **Multi-Modal RAG**: Search across text, figures, tables, and equations
            - **Code Implementation**: Generate code from research descriptions
            - **Experiment Design**: Design and simulate experiments
            - **Critical Analysis**: Evaluate research quality and identify gaps
            - **Visual Understanding**: Parse and analyze scientific visualizations
            """)
        
        with col2:
            st.subheader("🤖 Specialized Agents")
            st.markdown("""
            - **Research Agent**: Literature analysis and synthesis
            - **Implementation Agent**: Code generation and validation
            - **Experiment Agent**: Experimental design and execution
            - **Critique Agent**: Quality assessment and recommendations
            - **Orchestrator**: Coordinates all agents intelligently
            """)
        
        st.subheader("🚀 Quick Start")
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Analyze Document", use_container_width=True):
                st.switch_page("Document Analysis")
        
        with col2:
            if st.button("❓ Ask Research Question", use_container_width=True):
                st.switch_page("Research Query")
        
        with col3:
            if st.button("📊 View Agent Status", use_container_width=True):
                st.switch_page("Agent Status")
        
        # Recent activity (if any)
        if hasattr(self.research_system, 'get_recent_activity'):
            st.subheader("📈 Recent Activity")
            activity = self.research_system.get_recent_activity()
            if activity:
                for item in activity[-5:]:  # Show last 5 items
                    st.text(f"• {item}")
            else:
                st.text("No recent activity")
    
    def document_analysis_page(self):
        """Document analysis and upload page."""
        
        st.header("📄 Document Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a research paper (PDF)",
            type=['pdf'],
            help="Upload a PDF research paper for analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with st.spinner("Processing document..."):
                # In a real implementation, you'd save the file and process it
                st.success("Document uploaded successfully!")
                
                # Show basic file info
                st.subheader("Document Information")
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.1f} KB",
                    "File type": uploaded_file.type
                }
                
                for key, value in file_details.items():
                    st.text(f"{key}: {value}")
                
                # Simulate document processing
                if st.button("Analyze Document"):
                    with st.spinner("Analyzing document with AI agents..."):
                        # Simulate analysis
                        analysis_results = {
                            "title": "Sample Research Paper Title",
                            "abstract": "This is a sample abstract extracted from the document...",
                            "key_concepts": ["machine learning", "neural networks", "classification"],
                            "figures": 3,
                            "tables": 2,
                            "equations": 5,
                            "page_count": 12
                        }
                        
                        st.subheader("Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Figures", analysis_results["figures"])
                            st.metric("Tables", analysis_results["tables"])
                            st.metric("Equations", analysis_results["equations"])
                        
                        with col2:
                            st.metric("Pages", analysis_results["page_count"])
                            st.metric("Key Concepts", len(analysis_results["key_concepts"]))
                        
                        st.subheader("Extracted Content")
                        
                        with st.expander("Title"):
                            st.write(analysis_results["title"])
                        
                        with st.expander("Abstract"):
                            st.write(analysis_results["abstract"])
                        
                        with st.expander("Key Concepts"):
                            for concept in analysis_results["key_concepts"]:
                                st.write(f"• {concept}")
    
    def research_query_page(self):
        """Research query interface."""
        
        st.header("❓ Research Query Interface")
        
        # Query input
        query = st.text_area(
            "Enter your research question",
            placeholder="e.g., 'What are the latest developments in transformer architectures for computer vision?'",
            height=100
        )
        
        # Query options
        col1, col2 = st.columns(2)
        
        with col1:
            include_implementation = st.checkbox("Include code implementation", value=True)
            include_experiments = st.checkbox("Include experiment design", value=True)
        
        with col2:
            include_critique = st.checkbox("Include critical analysis", value=True)
            max_sources = st.slider("Maximum sources to consider", 5, 50, 20)
        
        if st.button("Submit Query", use_container_width=True):
            if query.strip():
                with st.spinner("Processing query with multi-agent system..."):
                    # Simulate query processing
                    results = self.simulate_query_processing(query, {
                        "include_implementation": include_implementation,
                        "include_experiments": include_experiments,
                        "include_critique": include_critique,
                        "max_sources": max_sources
                    })
                    
                    self.display_query_results(results)
            else:
                st.error("Please enter a research question.")
    
    def simulate_query_processing(self, query: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate query processing (replace with actual system call)."""
        
        # This would be replaced with:
        # return asyncio.run(self.research_system.process_research_query(query, options))
        
        return {
            "query": query,
            "status": "completed",
            "steps_completed": ["research", "implementation", "experiment", "critique", "synthesis"],
            "results": {
                "research": {
                    "findings": {
                        "summary": "Found 15 relevant papers on transformer architectures for computer vision",
                        "insights": [
                            "Vision Transformers (ViTs) show promising results for image classification",
                            "Hybrid CNN-Transformer architectures are emerging as effective solutions",
                            "Attention mechanisms help models focus on relevant image regions"
                        ]
                    }
                },
                "implementation": {
                    "implementation": {
                        "request": "Implement Vision Transformer architecture",
                        "code": {
                            "main_code": "class VisionTransformer(nn.Module):\n    def __init__(self, ...):\n        # Implementation here\n        pass"
                        }
                    }
                } if options["include_implementation"] else None,
                "experiment": {
                    "experiment": {
                        "design": {
                            "hypothesis": "Vision Transformers will outperform CNNs on image classification tasks"
                        },
                        "results": {
                            "metrics_calculated": {
                                "accuracy": {"mean": 87.5, "std": 2.1}
                            }
                        }
                    }
                } if options["include_experiments"] else None,
                "critique": {
                    "critique": {
                        "methodological_score": 7,
                        "evidence_score": 8,
                        "recommendations": {
                            "priority_improvements": [
                                "Consider larger datasets for evaluation",
                                "Include more baseline comparisons"
                            ]
                        }
                    }
                } if options["include_critique"] else None
            },
            "synthesis": {
                "executive_summary": f"Comprehensive analysis completed for: {query}",
                "key_findings": [
                    "Vision Transformers show strong performance",
                    "Hybrid approaches are promising",
                    "Further research needed on efficiency"
                ],
                "confidence_score": 82.5
            }
        }
    
    def display_query_results(self, results: Dict[str, Any]):
        """Display query results."""
        
        st.subheader("🎯 Query Results")
        
        # Executive summary
        if "synthesis" in results:
            synthesis = results["synthesis"]
            
            st.subheader("Executive Summary")
            st.write(synthesis.get("executive_summary", "No summary available"))
            
            # Confidence score
            confidence = synthesis.get("confidence_score", 0)
            st.metric("Confidence Score", f"{confidence:.1f}%")
            
            # Key findings
            if "key_findings" in synthesis:
                st.subheader("Key Findings")
                for finding in synthesis["key_findings"]:
                    st.write(f"• {finding}")
        
        # Detailed results by agent
        st.subheader("Detailed Results")
        
        # Research results
        if "research" in results["results"]:
            with st.expander("🔍 Research Analysis"):
                research_data = results["results"]["research"]
                if "findings" in research_data:
                    findings = research_data["findings"]
                    st.write("**Summary:**", findings.get("summary", "No summary"))
                    
                    if "insights" in findings:
                        st.write("**Key Insights:**")
                        for insight in findings["insights"]:
                            st.write(f"• {insight}")
        
        # Implementation results
        if "implementation" in results["results"] and results["results"]["implementation"]:
            with st.expander("💻 Implementation"):
                impl_data = results["results"]["implementation"]["implementation"]
                st.write("**Request:**", impl_data.get("request", "No description"))
                
                if "code" in impl_data and "main_code" in impl_data["code"]:
                    st.code(impl_data["code"]["main_code"], language="python")
        
        # Experiment results
        if "experiment" in results["results"] and results["results"]["experiment"]:
            with st.expander("🧪 Experiment Results"):
                exp_data = results["results"]["experiment"]["experiment"]
                
                if "design" in exp_data:
                    st.write("**Hypothesis:**", exp_data["design"].get("hypothesis", "No hypothesis"))
                
                if "results" in exp_data and "metrics_calculated" in exp_data["results"]:
                    metrics = exp_data["results"]["metrics_calculated"]
                    st.write("**Results:**")
                    for metric, values in metrics.items():
                        if isinstance(values, dict):
                            st.write(f"• {metric}: {values.get('mean', 'N/A')} ± {values.get('std', 'N/A')}")
        
        # Critique results
        if "critique" in results["results"] and results["results"]["critique"]:
            with st.expander("📊 Critical Analysis"):
                critique_data = results["results"]["critique"]["critique"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Methodology Score", f"{critique_data.get('methodological_score', 0)}/10")
                
                with col2:
                    st.metric("Evidence Score", f"{critique_data.get('evidence_score', 0)}/10")
                
                if "recommendations" in critique_data and "priority_improvements" in critique_data["recommendations"]:
                    st.write("**Priority Improvements:**")
                    for improvement in critique_data["recommendations"]["priority_improvements"]:
                        st.write(f"• {improvement}")
    
    def agent_status_page(self):
        """Display agent status and system information."""
        
        st.header("🤖 Agent Status Dashboard")
        
        # System status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Research Agent", "Active", delta="Ready")
        
        with col2:
            st.metric("Implementation Agent", "Active", delta="Ready")
        
        with col3:
            st.metric("Experiment Agent", "Active", delta="Ready")
        
        with col4:
            st.metric("Critique Agent", "Active", delta="Ready")
        
        # Memory status
        st.subheader("📊 System Memory")
        
        memory_data = {
            "Content Type": ["Documents", "Findings", "Implementations", "Experiments", "Critiques"],
            "Count": [12, 25, 8, 15, 20],
            "Last Updated": ["2 hours ago", "1 hour ago", "30 min ago", "45 min ago", "1 hour ago"]
        }
        
        df = pd.DataFrame(memory_data)
        st.dataframe(df, use_container_width=True)
        
        # Agent interaction history
        st.subheader("🔄 Recent Agent Interactions")
        
        interactions = [
            {"Time": "14:30", "Agent": "Research", "Action": "Analyzed 'transformer architecture' query", "Status": "Completed"},
            {"Time": "14:25", "Agent": "Implementation", "Action": "Generated Vision Transformer code", "Status": "Completed"},
            {"Time": "14:20", "Agent": "Experiment", "Action": "Designed classification experiment", "Status": "Completed"},
            {"Time": "14:15", "Agent": "Critique", "Action": "Evaluated methodology quality", "Status": "Completed"},
            {"Time": "14:10", "Agent": "Orchestrator", "Action": "Coordinated multi-agent workflow", "Status": "Completed"}
        ]
        
        interaction_df = pd.DataFrame(interactions)
        st.dataframe(interaction_df, use_container_width=True)
        
        # System configuration
        with st.expander("⚙️ System Configuration"):
            config_info = {
                "Embedding Model": "sentence-transformers/all-MiniLM-L6-v2",
                "LLM Model": "gpt-3.5-turbo", 
                "Vector Store": "ChromaDB",
                "Max Iterations": "5 per agent",
                "Temperature": "0.3 (Research), 0.1 (Implementation)"
            }
            
            for key, value in config_info.items():
                st.text(f"{key}: {value}")
    
    def visualization_page(self):
        """Visualization tools and utilities."""
        
        st.header("📊 Visualization Tools")
        
        tab1, tab2, tab3 = st.tabs(["Generate Visualization", "Analyze Image", "Convert Charts"])
        
        with tab1:
            st.subheader("Generate New Visualization")
            
            # Data input
            col1, col2 = st.columns(2)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["bar_chart", "line_chart", "scatter_plot", "pie_chart", "heatmap"]
                )
                
                title = st.text_input("Chart Title", "Sample Chart")
            
            with col2:
                x_label = st.text_input("X-axis Label", "X Values")
                y_label = st.text_input("Y-axis Label", "Y Values")
            
            # Sample data input
            st.subheader("Data Input")
            
            if chart_type in ["bar_chart", "pie_chart"]:
                categories = st.text_input("Categories (comma-separated)", "A,B,C,D,E")
                values = st.text_input("Values (comma-separated)", "10,20,15,25,30")
                
                if st.button("Generate Chart"):
                    cat_list = [c.strip() for c in categories.split(",")]
                    val_list = [float(v.strip()) for v in values.split(",")]
                    
                    if len(cat_list) == len(val_list):
                        # In real implementation, would call visualization generator
                        st.success("Chart generated successfully!")
                        st.info("Chart would be displayed here using the VisualizationGenerator")
                    else:
                        st.error("Categories and values must have the same length")
        
        with tab2:
            st.subheader("Analyze Uploaded Image")
            
            uploaded_image = st.file_uploader(
                "Upload an image to analyze",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a chart or scientific visualization"
            )
            
            if uploaded_image is not None:
                # Display the image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Analyze Image"):
                    # In real implementation, would call visual parser
                    st.success("Image analysis completed!")
                    
                    # Mock analysis results
                    analysis_results = {
                        "Chart Type": "Bar Chart (85% confidence)",
                        "Data Points": "5 bars detected",
                        "Text Elements": "Title, X-axis label, Y-axis label detected",
                        "Color Scheme": "Multi-color with 4 distinct colors",
                        "Insights": [
                            "Well-structured bar chart with clear labels",
                            "Data shows increasing trend from left to right",
                            "Professional formatting with good contrast"
                        ]
                    }
                    
                    for key, value in analysis_results.items():
                        if key == "Insights":
                            st.write(f"**{key}:**")
                            for insight in value:
                                st.write(f"• {insight}")
                        else:
                            st.write(f"**{key}:** {value}")
        
        with tab3:
            st.subheader("Convert Chart Types")
            
            st.info("Upload a chart and convert it to a different visualization type")
            
            source_chart = st.file_uploader(
                "Upload source chart",
                type=['png', 'jpg', 'jpeg'],
                key="source_chart"
            )
            
            if source_chart is not None:
                image = Image.open(source_chart)
                st.image(image, caption="Source Chart", use_column_width=True)
                
                target_type = st.selectbox(
                    "Convert to:",
                    ["bar_chart", "line_chart", "scatter_plot", "pie_chart", "heatmap"]
                )
                
                if st.button("Convert Chart"):
                    # In real implementation, would call visual converter
                    st.success(f"Chart converted to {target_type}!")
                    st.info("Converted chart would be displayed here using the VisualConverter")
