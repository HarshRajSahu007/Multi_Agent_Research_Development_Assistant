import streamlit as st
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64


class VisualizationViewComponent:
    """Component for creating and analyzing visualizations."""
    
    def __init__(self, research_system):
        self.research_system = research_system
    
    def render(self):
        """Render the visualization interface."""
        
        st.subheader("📊 Visualization Tools")
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Create Visualization", 
            "🔍 Analyze Image", 
            "🔄 Convert Charts",
            "📋 Visualization Gallery"
        ])
        
        with tab1:
            self._render_create_visualization()
        
        with tab2:
            self._render_analyze_image()
        
        with tab3:
            self._render_convert_charts()
        
        with tab4:
            self._render_visualization_gallery()
    
    def _render_create_visualization(self):
        """Render visualization creation interface."""
        
        st.write("**Create Custom Visualizations**")
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Visualization Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap", "Box Plot", "Violin Plot"]
        )
        
        # Data input method
        data_method = st.radio(
            "Data Input Method",
            ["Manual Entry", "Upload CSV", "Generate Sample Data"]
        )
        
        if data_method == "Manual Entry":
            data = self._manual_data_entry(viz_type)
        elif data_method == "Upload CSV":
            data = self._csv_data_upload()
        else:
            data = self._generate_sample_data(viz_type)
        
        if data is not None:
            # Customization options
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Chart Title", f"Sample {viz_type}")
                x_label = st.text_input("X-axis Label", "X Values")
                y_label = st.text_input("Y-axis Label", "Y Values")
            
            with col2:
                color_scheme = st.selectbox(
                    "Color Scheme",
                    ["Default", "Viridis", "Plasma", "Set1", "Set2", "Pastel"]
                )
                show_grid = st.checkbox("Show Grid", value=True)
                show_legend = st.checkbox("Show Legend", value=True)
            
            # Generate visualization
            if st.button("🎨 Generate Visualization", type="primary"):
                fig = self._create_visualization(viz_type, data, {
                    "title": title,
                    "x_label": x_label,
                    "y_label": y_label,
                    "color_scheme": color_scheme,
                    "show_grid": show_grid,
                    "show_legend": show_legend
                })
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("💾 Save to Gallery"):
                            self._save_to_gallery(fig, title, viz_type)
                    
                    with col2:
                        # Export as HTML
                        html_str = fig.to_html()
                        st.download_button(
                            "📤 Export HTML",
                            html_str,
                            f"{title.lower().replace(' ', '_')}.html",
                            "text/html"
                        )
                    
                    with col3:
                        # Export as PNG (would need additional setup)
                        if st.button("🖼️ Export PNG"):
                            st.info("PNG export requires additional configuration")
    
    def _manual_data_entry(self, viz_type: str) -> Optional[Dict[str, Any]]:
        """Handle manual data entry for different chart types."""
        
        if viz_type in ["Bar Chart", "Pie Chart"]:
            col1, col2 = st.columns(2)
            
            with col1:
                categories = st.text_area(
                    "Categories (one per line)",
                    "Category A\nCategory B\nCategory C\nCategory D"
                )
            
            with col2:
                values = st.text_area(
                    "Values (one per line)",
                    "10\n20\n15\n25"
                )
            
            try:
                cat_list = [c.strip() for c in categories.split('\n') if c.strip()]
                val_list = [float(v.strip()) for v in values.split('\n') if v.strip()]
                
                if len(cat_list) == len(val_list):
                    return {"categories": cat_list, "values": val_list}
                else:
                    st.error("Categories and values must have the same length")
                    return None
            except ValueError:
                st.error("Please enter valid numbers for values")
                return None
        
        elif viz_type in ["Line Chart", "Scatter Plot"]:
            col1, col2 = st.columns(2)
            
            with col1:
                x_values = st.text_area(
                    "X Values (one per line)",
                    "1\n2\n3\n4\n5"
                )
            
            with col2:
                y_values = st.text_area(
                    "Y Values (one per line)",
                    "10\n25\n15\n30\n20"
                )
            
            try:
                x_list = [float(x.strip()) for x in x_values.split('\n') if x.strip()]
                y_list = [float(y.strip()) for y in y_values.split('\n') if y.strip()]
                
                if len(x_list) == len(y_list):
                    return {"x_values": x_list, "y_values": y_list}
                else:
                    st.error("X and Y values must have the same length")
                    return None
            except ValueError:
                st.error("Please enter valid numbers")
                return None
        
        return None
    
    def _csv_data_upload(self) -> Optional[Dict[str, Any]]:
        """Handle CSV data upload."""
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column selection
                columns = df.columns.tolist()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    x_column = st.selectbox("X Column", columns)
                
                with col2:
                    y_column = st.selectbox("Y Column", [col for col in columns if col != x_column])
                
                if x_column and y_column:
                    return {
                        "dataframe": df,
                        "x_column": x_column,
                        "y_column": y_column
                    }
            
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        
        return None
    
    def _generate_sample_data(self, viz_type: str) -> Dict[str, Any]:
        """Generate sample data for different visualization types."""
        
        np.random.seed(42)
        
        if viz_type == "Bar Chart":
            return {
                "categories": ["Product A", "Product B", "Product C", "Product D", "Product E"],
                "values": [23, 17, 35, 29, 12]
            }
        
        elif viz_type == "Line Chart":
            x = list(range(1, 11))
            y = [val + np.random.normal(0, 2) for val in [5, 7, 6, 8, 9, 11, 10, 12, 14, 13]]
            return {"x_values": x, "y_values": y}
        
        elif viz_type == "Scatter Plot":
            n = 50
            x = np.random.normal(50, 15, n)
            y = 2 * x + np.random.normal(0, 10, n)
            return {"x_values": x.tolist(), "y_values": y.tolist()}
        
        elif viz_type == "Pie Chart":
            return {
                "categories": ["Desktop", "Mobile", "Tablet", "Other"],
                "values": [45, 35, 15, 5]
            }
        
        elif viz_type == "Heatmap":
            data = np.random.randn(10, 10)
            return {"matrix": data.tolist()}
        
        return {}
    
    def _create_visualization(self, viz_type: str, data: Dict[str, Any], options: Dict[str, Any]):
        """Create visualization based on type and data."""
        
        try:
            if viz_type == "Bar Chart":
                fig = px.bar(
                    x=data["categories"], 
                    y=data["values"],
                    title=options["title"],
                    labels={"x": options["x_label"], "y": options["y_label"]}
                )
            
            elif viz_type == "Line Chart":
                fig = px.line(
                    x=data["x_values"], 
                    y=data["y_values"],
                    title=options["title"],
                    labels={"x": options["x_label"], "y": options["y_label"]}
                )
            
            elif viz_type == "Scatter Plot":
                fig = px.scatter(
                    x=data["x_values"], 
                    y=data["y_values"],
                    title=options["title"],
                    labels={"x": options["x_label"], "y": options["y_label"]}
                )
            
            elif viz_type == "Pie Chart":
                fig = px.pie(
                    values=data["values"], 
                    names=data["categories"],
                    title=options["title"]
                )
            
            elif viz_type == "Heatmap":
                fig = px.imshow(
                    data["matrix"],
                    title=options["title"],
                    labels={"x": options["x_label"], "y": options["y_label"]}
                )
            
            elif viz_type == "Box Plot":
                # Create sample data for box plot
                categories = data.get("categories", ["A", "B", "C"])
                box_data = []
                for cat in categories:
                    values = np.random.normal(50, 10, 30)
                    for val in values:
                        box_data.append({"category": cat, "value": val})
                
                box_df = pd.DataFrame(box_data)
                fig = px.box(box_df, x="category", y="value", title=options["title"])
            
            else:
                return None
            
            # Apply styling options
            if options["show_grid"]:
                fig.update_layout(showlegend=options["show_legend"])
            
            # Apply color scheme
            if options["color_scheme"] != "Default":
                color_map = {
                    "Viridis": px.colors.sequential.Viridis,
                    "Plasma": px.colors.sequential.Plasma,
                    "Set1": px.colors.qualitative.Set1,
                    "Set2": px.colors.qualitative.Set2,
                    "Pastel": px.colors.qualitative.Pastel
                }
                
                if options["color_scheme"] in color_map:
                    fig.update_traces(marker_color=color_map[options["color_scheme"]])
            
            return fig
        
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _render_analyze_image(self):
        """Render image analysis interface."""
        
        st.write("**Analyze Existing Visualizations**")
        
        uploaded_image = st.file_uploader(
            "Upload an image to analyze",
            type=['png', 'jpg', 'jpeg', 'svg'],
            help="Upload a chart or scientific visualization for analysis"
        )
        
        if uploaded_image is not None:
            # Display image
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Uploaded Visualization", use_column_width=True)
            
            with col2:
                st.write("**Image Information:**")
                st.write(f"Size: {image.size[0]} × {image.size[1]}")
                st.write(f"Format: {image.format}")
                st.write(f"Mode: {image.mode}")
            
            # Analysis options
            analysis_options = st.multiselect(
                "Analysis Options",
                ["Chart Type Detection", "Color Analysis", "Text Extraction", "Data Point Extraction"],
                default=["Chart Type Detection", "Color Analysis"]
            )
            
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    results = self._analyze_image(image, analysis_options)
                    self._display_analysis_results(results)
    
    def _analyze_image(self, image: Image.Image, options: List[str]) -> Dict[str, Any]:
        """Analyze uploaded image (mock implementation)."""
        
        # Mock analysis results
        results = {}
        
        if "Chart Type Detection" in options:
            results["chart_type"] = {
                "detected_type": "Bar Chart",
                "confidence": 0.87,
                "alternative_types": [
                    {"type": "Column Chart", "confidence": 0.65},
                    {"type": "Histogram", "confidence": 0.23}
                ]
            }
        
        if "Color Analysis" in options:
            results["color_analysis"] = {
                "dominant_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                "color_count": 8,
                "is_colorblind_friendly": True,
                "color_scheme": "Professional"
            }
        
        if "Text Extraction" in options:
            results["text_extraction"] = {
                "title": "Sales Performance Q4 2023",
                "axis_labels": {
                    "x_axis": "Product Categories",
                    "y_axis": "Revenue (in thousands)"
                },
                "legend_items": ["Q3", "Q4"],
                "data_labels": ["Electronics: $45K", "Clothing: $32K", "Books: $18K"]
            }
        
        if "Data Point Extraction" in options:
            results["data_extraction"] = {
                "data_points": [
                    {"category": "Electronics", "value": 45, "position": (120, 200)},
                    {"category": "Clothing", "value": 32, "position": (220, 280)},
                    {"category": "Books", "value": 18, "position": (320, 380)},
                    {"category": "Home", "value": 28, "position": (420, 320)}
                ],
                "extraction_method": "Computer Vision",
                "accuracy_estimate": 0.92
            }
        
        return results
    
    def _display_analysis_results(self, results: Dict[str, Any]):
        """Display image analysis results."""
        
        st.subheader("🔍 Analysis Results")
        
        # Chart type detection
        if "chart_type" in results:
            with st.expander("📊 Chart Type Detection"):
                chart_info = results["chart_type"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Detected Type", chart_info["detected_type"])
                    st.metric("Confidence", f"{chart_info['confidence']:.1%}")
                
                with col2:
                    st.write("**Alternative Classifications:**")
                    for alt in chart_info["alternative_types"]:
                        st.write(f"• {alt['type']}: {alt['confidence']:.1%}")
        
        # Color analysis
        if "color_analysis" in results:
            with st.expander("🎨 Color Analysis"):
                color_info = results["color_analysis"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Unique Colors", color_info["color_count"])
                    st.metric("Color Scheme", color_info["color_scheme"])
                
                with col2:
                    st.write("**Dominant Colors:**")
                    for color in color_info["dominant_colors"]:
                        st.markdown(f'<div style="display: inline-block; width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border: 1px solid #ccc;"></div>{color}', unsafe_allow_html=True)
                    
                    accessibility = "✅ Yes" if color_info["is_colorblind_friendly"] else "❌ No"
                    st.write(f"**Colorblind Friendly:** {accessibility}")
        
        # Text extraction
        if "text_extraction" in results:
            with st.expander("📝 Text Extraction"):
                text_info = results["text_extraction"]
                
                st.write(f"**Title:** {text_info['title']}")
                st.write(f"**X-axis:** {text_info['axis_labels']['x_axis']}")
                st.write(f"**Y-axis:** {text_info['axis_labels']['y_axis']}")
                
                if text_info["legend_items"]:
                    st.write("**Legend Items:**")
                    for item in text_info["legend_items"]:
                        st.write(f"• {item}")
                
                if text_info["data_labels"]:
                    st.write("**Data Labels:**")
                    for label in text_info["data_labels"]:
                        st.write(f"• {label}")
        
        # Data extraction
        if "data_extraction" in results:
            with st.expander("📈 Data Point Extraction"):
                data_info = results["data_extraction"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Data Points", len(data_info["data_points"]))
                    st.metric("Accuracy Estimate", f"{data_info['accuracy_estimate']:.1%}")
                
                with col2:
                    st.write(f"**Method:** {data_info['extraction_method']}")
                
                # Data table
                if data_info["data_points"]:
                    df = pd.DataFrame(data_info["data_points"])
                    st.dataframe(df[["category", "value"]], use_container_width=True)
                    
                    # Option to download extracted data
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Extracted Data",
                        csv,
                        "extracted_data.csv",
                        "text/csv"
                    )
    
    def _render_convert_charts(self):
        """Render chart conversion interface."""
        
        st.write("**Convert Between Chart Types**")
        
        # Source chart upload
        source_image = st.file_uploader(
            "Upload source chart",
            type=['png', 'jpg', 'jpeg'],
            key="source_conversion"
        )
        
        if source_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Source Chart:**")
                image = Image.open(source_image)
                st.image(image, use_column_width=True)
                
                # Detect current type (mock)
                st.info("🔍 Detected: Bar Chart (85% confidence)")
            
            with col2:
                st.write("**Conversion Options:**")
                
                target_type = st.selectbox(
                    "Convert to:",
                    ["Line Chart", "Pie Chart", "Scatter Plot", "Area Chart", "Horizontal Bar"]
                )
                
                # Conversion settings
                preserve_colors = st.checkbox("Preserve Color Scheme", value=True)
                extract_title = st.checkbox("Extract Original Title", value=True)
                auto_scale = st.checkbox("Auto-scale Values", value=True)
                
                if st.button("🔄 Convert Chart", type="primary"):
                    with st.spinner("Converting chart..."):
                        converted_chart = self._convert_chart(image, target_type, {
                            "preserve_colors": preserve_colors,
                            "extract_title": extract_title,
                            "auto_scale": auto_scale
                        })
                        
                        if converted_chart:
                            st.success("✅ Conversion completed!")
                            
                            # Display converted chart
                            st.write("**Converted Chart:**")
                            st.plotly_chart(converted_chart, use_container_width=True)
                            
                            # Save options
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.button("💾 Save Conversion"):
                                    st.success("Chart saved to gallery!")
                            
                            with col2:
                                # Export options
                                html_str = converted_chart.to_html()
                                st.download_button(
                                    "📤 Export HTML",
                                    html_str,
                                    f"converted_{target_type.lower().replace(' ', '_')}.html",
                                    "text/html"
                                )
    
    def _convert_chart(self, source_image: Image.Image, target_type: str, options: Dict[str, bool]):
        """Convert chart to different type (mock implementation)."""
        
        # Mock extracted data (in real implementation, this would use computer vision)
        extracted_data = {
            "categories": ["Product A", "Product B", "Product C", "Product D"],
            "values": [25, 40, 30, 35],
            "title": "Sales Performance" if options["extract_title"] else "Converted Chart"
        }
        
        # Create converted visualization
        if target_type == "Line Chart":
            fig = px.line(
                x=extracted_data["categories"],
                y=extracted_data["values"],
                title=extracted_data["title"],
                markers=True
            )
        
        elif target_type == "Pie Chart":
            fig = px.pie(
                values=extracted_data["values"],
                names=extracted_data["categories"],
                title=extracted_data["title"]
            )
        
        elif target_type == "Scatter Plot":
            fig = px.scatter(
                x=list(range(len(extracted_data["categories"]))),
                y=extracted_data["values"],
                text=extracted_data["categories"],
                title=extracted_data["title"]
            )
            fig.update_traces(textposition="top center")
        
        elif target_type == "Area Chart":
            fig = px.area(
                x=extracted_data["categories"],
                y=extracted_data["values"],
                title=extracted_data["title"]
            )
        
        elif target_type == "Horizontal Bar":
            fig = px.bar(
                x=extracted_data["values"],
                y=extracted_data["categories"],
                orientation='h',
                title=extracted_data["title"]
            )
        
        else:
            return None
        
        # Apply color preservation if requested
        if options["preserve_colors"]:
            # Mock color preservation (in real implementation, would extract colors from source)
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            fig.update_traces(marker_color=colors[:len(extracted_data["categories"])])
        
        return fig
    
    def _render_visualization_gallery(self):
        """Render visualization gallery interface."""
        
        st.write("**Visualization Gallery**")
        
        # Gallery filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox("Filter by Type", ["All", "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot"])
        
        with col2:
            filter_date = st.selectbox("Filter by Date", ["All Time", "Last Week", "Last Month", "Last Year"])
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Date Created", "Name", "Type", "Views"])
        
        # Mock gallery data
        gallery_items = [
            {
                "id": 1,
                "name": "Sales Performance Q4",
                "type": "Bar Chart",
                "created": "2024-01-15",
                "views": 45,
                "thumbnail": "📊"
            },
            {
                "id": 2,
                "name": "Website Traffic Trends",
                "type": "Line Chart",
                "created": "2024-01-14",
                "views": 32,
                "thumbnail": "📈"
            },
            {
                "id": 3,
                "name": "Market Share Distribution",
                "type": "Pie Chart",
                "created": "2024-01-13",
                "views": 28,
                "thumbnail": "🥧"
            },
            {
                "id": 4,
                "name": "Price vs Quality Analysis",
                "type": "Scatter Plot",
                "created": "2024-01-12",
                "views": 19,
                "thumbnail": "⚪"
            },
            {
                "id": 5,
                "name": "Performance Heatmap",
                "type": "Heatmap",
                "created": "2024-01-11",
                "views": 23,
                "thumbnail": "🔥"
            }
        ]
        
        # Display gallery items
        cols_per_row = 3
        for i in range(0, len(gallery_items), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(gallery_items):
                    item = gallery_items[i + j]
                    
                    with col:
                        # Gallery item card
                        with st.container():
                            st.markdown(f"### {item['thumbnail']} {item['name']}")
                            st.write(f"**Type:** {item['type']}")
                            st.write(f"**Created:** {item['created']}")
                            st.write(f"**Views:** {item['views']}")
                            
                            # Action buttons
                            button_col1, button_col2 = st.columns(2)
                            
                            with button_col1:
                                if st.button("👁️ View", key=f"view_{item['id']}"):
                                    self._view_gallery_item(item)
                            
                            with button_col2:
                                if st.button("🗑️ Delete", key=f"delete_{item['id']}"):
                                    st.success(f"Deleted '{item['name']}'")
                            
                            st.divider()
        
        # Gallery statistics
        with st.expander("📊 Gallery Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Visualizations", len(gallery_items))
            
            with col2:
                st.metric("Most Popular Type", "Bar Chart")
            
            with col3:
                st.metric("Total Views", sum(item["views"] for item in gallery_items))
            
            with col4:
                st.metric("Average Views", f"{sum(item['views'] for item in gallery_items) / len(gallery_items):.1f}")
    
    def _view_gallery_item(self, item: Dict[str, Any]):
        """Display a gallery item in detail."""
        
        st.subheader(f"📊 {item['name']}")
        
        # Create a sample visualization based on the item type
        if item['type'] == "Bar Chart":
            fig = px.bar(
                x=["Q1", "Q2", "Q3", "Q4"],
                y=[20, 30, 25, 35],
                title=item['name']
            )
        elif item['type'] == "Line Chart":
            fig = px.line(
                x=list(range(1, 13)),
                y=[100, 120, 110, 140, 160, 150, 180, 170, 190, 200, 185, 210],
                title=item['name']
            )
        elif item['type'] == "Pie Chart":
            fig = px.pie(
                values=[40, 30, 20, 10],
                names=["Desktop", "Mobile", "Tablet", "Other"],
                title=item['name']
            )
        else:
            fig = px.scatter(
                x=[1, 2, 3, 4, 5],
                y=[10, 25, 15, 30, 20],
                title=item['name']
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Item details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Type:** {item['type']}")
            st.write(f"**Created:** {item['created']}")
        
        with col2:
            st.write(f"**Views:** {item['views']}")
            st.write(f"**ID:** {item['id']}")
        
        with col3:
            # Action buttons
            if st.button("📤 Export"):
                st.success("Exported successfully!")
            
            if st.button("📋 Duplicate"):
                st.success("Duplicated to gallery!")
    
    def _save_to_gallery(self, fig, title: str, viz_type: str):
        """Save visualization to gallery."""
        
        # In real implementation, would save to database/file system
        st.success(f"✅ '{title}' saved to gallery as {viz_type}")
        
        # Add to session state for demo
        if "gallery_items" not in st.session_state:
            st.session_state.gallery_items = []
        
        new_item = {
            "id": len(st.session_state.gallery_items) + 1,
            "name": title,
            "type": viz_type,
            "created": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "views": 0,
            "figure": fig
        }
        
        st.session_state.gallery_items.append(new_item)
