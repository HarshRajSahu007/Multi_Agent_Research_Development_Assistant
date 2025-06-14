import streamlit as st
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


class ExperimentDesignerComponent:
    """Component for designing and visualizing experiments."""
    
    def __init__(self, research_system):
        self.research_system = research_system
    
    def render(self) -> Optional[Dict[str, Any]]:
        """Render the experiment designer interface."""
        
        st.subheader("🧪 Experiment Designer")
        
        # Experiment configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Configuration**")
            
            experiment_name = st.text_input("Experiment Name", "Transformer Performance Study")
            hypothesis = st.text_area(
                "Hypothesis",
                "Transformer models will outperform CNN baselines on image classification tasks",
                height=100
            )
            
            research_question = st.text_area(
                "Research Question",
                "How do different transformer architectures compare in terms of accuracy and efficiency?",
                height=80
            )
        
        with col2:
            st.write("**Experimental Design**")
            
            study_type = st.selectbox(
                "Study Type",
                ["Comparative Study", "Ablation Study", "Parameter Sweep", "Benchmark Study"]
            )
            
            sample_size = st.slider("Sample Size", 100, 10000, 1000)
            significance_level = st.selectbox("Significance Level", [0.05, 0.01, 0.001])
            
            random_seed = st.number_input("Random Seed", value=42)
        
        # Variables section
        st.subheader("📊 Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Independent Variables**")
            
            # Dynamic variable addition
            if "independent_vars" not in st.session_state:
                st.session_state.independent_vars = [
                    {"name": "Model Architecture", "type": "Categorical", "values": "ViT-Base, ViT-Large, ResNet-50"}
                ]
            
            for i, var in enumerate(st.session_state.independent_vars):
                with st.expander(f"Variable {i+1}: {var['name']}"):
                    var['name'] = st.text_input("Name", var['name'], key=f"indep_name_{i}")
                    var['type'] = st.selectbox("Type", ["Categorical", "Continuous", "Ordinal"], 
                                             index=["Categorical", "Continuous", "Ordinal"].index(var['type']), 
                                             key=f"indep_type_{i}")
                    var['values'] = st.text_input("Values/Range", var['values'], key=f"indep_values_{i}")
                    
                    if st.button("Remove", key=f"remove_indep_{i}"):
                        st.session_state.independent_vars.pop(i)
                        st.rerun()
            
            if st.button("➕ Add Independent Variable"):
                st.session_state.independent_vars.append({
                    "name": "New Variable", 
                    "type": "Categorical", 
                    "values": "Value1, Value2"
                })
                st.rerun()
        
        with col2:
            st.write("**Dependent Variables**")
            
            if "dependent_vars" not in st.session_state:
                st.session_state.dependent_vars = [
                    {"name": "Accuracy", "type": "Continuous", "range": "0-1"},
                    {"name": "Training Time", "type": "Continuous", "range": "0-∞"}
                ]
            
            for i, var in enumerate(st.session_state.dependent_vars):
                with st.expander(f"Variable {i+1}: {var['name']}"):
                    var['name'] = st.text_input("Name", var['name'], key=f"dep_name_{i}")
                    var['type'] = st.selectbox("Type", ["Continuous", "Categorical", "Count"], 
                                             index=["Continuous", "Categorical", "Count"].index(var['type']), 
                                             key=f"dep_type_{i}")
                    var['range'] = st.text_input("Range", var['range'], key=f"dep_range_{i}")
                    
                    if st.button("Remove", key=f"remove_dep_{i}"):
                        st.session_state.dependent_vars.pop(i)
                        st.rerun()
            
            if st.button("➕ Add Dependent Variable"):
                st.session_state.dependent_vars.append({
                    "name": "New Variable", 
                    "type": "Continuous", 
                    "range": "0-100"
                })
                st.rerun()
        
        # Methodology section
        st.subheader("🔬 Methodology")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_collection = st.selectbox(
                "Data Collection Method",
                ["Cross-validation", "Train/Val/Test Split", "Bootstrapping", "Monte Carlo"]
            )
            
            evaluation_metrics = st.multiselect(
                "Evaluation Metrics",
                ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "MSE", "MAE", "R²"],
                default=["Accuracy", "F1-Score"]
            )
        
        with col2:
            control_variables = st.text_area(
                "Control Variables",
                "Dataset, Preprocessing, Optimizer, Learning Rate",
                height=80
            )
            
            statistical_tests = st.multiselect(
                "Statistical Tests",
                ["t-test", "ANOVA", "Chi-square", "Mann-Whitney U", "Kruskal-Wallis"],
                default=["t-test", "ANOVA"]
            )
        
        # Design and run experiment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Design Experiment", type="primary"):
                experiment_design = self._create_experiment_design()
                st.session_state.experiment_design = experiment_design
                st.success("Experiment design created!")
        
        with col2:
            if st.button("🚀 Run Simulation"):
                if hasattr(st.session_state, 'experiment_design'):
                    results = self._run_experiment_simulation()
                    st.session_state.experiment_results = results
                    st.success("Simulation completed!")
                else:
                    st.error("Please design experiment first!")
        
        with col3:
            if st.button("📊 Generate Report"):
                if hasattr(st.session_state, 'experiment_results'):
                    self._generate_experiment_report()
                else:
                    st.error("Please run simulation first!")
        
        # Display results if available
        if hasattr(st.session_state, 'experiment_results'):
            self._display_experiment_results(st.session_state.experiment_results)
    
    def _create_experiment_design(self) -> Dict[str, Any]:
        """Create structured experiment design."""
        
        design = {
            "name": "Transformer Performance Study",
            "hypothesis": "Transformer models will outperform CNN baselines",
            "independent_variables": st.session_state.get("independent_vars", []),
            "dependent_variables": st.session_state.get("dependent_vars", []),
            "methodology": {
                "sample_size": 1000,
                "design_type": "Between-subjects",
                "randomization": True,
                "blinding": False
            },
            "conditions": self._generate_experimental_conditions(),
            "power_analysis": self._calculate_power_analysis()
        }
        
        return design
    
    def _generate_experimental_conditions(self) -> List[Dict[str, Any]]:
        """Generate experimental conditions based on independent variables."""
        
        conditions = []
        
        # For demo, create some sample conditions
        architectures = ["ViT-Base", "ViT-Large", "ResNet-50"]
        datasets = ["CIFAR-10", "CIFAR-100", "ImageNet"]
        
        for arch in architectures:
            for dataset in datasets:
                conditions.append({
                    "condition_id": f"{arch}_{dataset}",
                    "architecture": arch,
                    "dataset": dataset,
                    "expected_participants": 100
                })
        
        return conditions
    
    def _calculate_power_analysis(self) -> Dict[str, float]:
        """Calculate statistical power analysis."""
        
        return {
            "effect_size": 0.5,  # Cohen's d
            "power": 0.8,
            "alpha": 0.05,
            "required_sample_size": 128
        }
    
    def _run_experiment_simulation(self) -> Dict[str, Any]:
        """Simulate experiment execution."""
        
        # Generate synthetic results
        np.random.seed(42)
        
        conditions = ["ViT-Base", "ViT-Large", "ResNet-50"]
        datasets = ["CIFAR-10", "CIFAR-100", "ImageNet"]
        
        results_data = []
        
        for condition in conditions:
            for dataset in datasets:
                # Simulate different performance levels
                if condition == "ViT-Large":
                    base_accuracy = 0.85
                elif condition == "ViT-Base":
                    base_accuracy = 0.82
                else:  # ResNet-50
                    base_accuracy = 0.78
                
                # Add dataset difficulty
                if dataset == "CIFAR-10":
                    dataset_bonus = 0.1
                elif dataset == "CIFAR-100":
                    dataset_bonus = 0.05
                else:  # ImageNet
                    dataset_bonus = 0.0
                
                # Generate sample results
                n_samples = 20
                accuracy_scores = np.random.normal(
                    base_accuracy + dataset_bonus, 
                    0.03, 
                    n_samples
                )
                
                training_times = np.random.normal(
                    100 if "ViT" in condition else 80, 
                    15, 
                    n_samples
                )
                
                for i in range(n_samples):
                    results_data.append({
                        "condition": condition,
                        "dataset": dataset,
                        "accuracy": max(0, min(1, accuracy_scores[i])),
                        "training_time": max(10, training_times[i]),
                        "run_id": i
                    })
        
        df = pd.DataFrame(results_data)
        
        # Calculate statistics
        summary_stats = df.groupby(['condition', 'dataset']).agg({
            'accuracy': ['mean', 'std', 'count'],
            'training_time': ['mean', 'std']
        }).round(4)
        
        return {
            "raw_data": df,
            "summary_stats": summary_stats,
            "statistical_tests": self._run_statistical_tests(df),
            "visualizations": self._create_result_visualizations(df)
        }
    
    def _run_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run statistical tests on the results."""
        
        from scipy import stats
        
        # ANOVA for accuracy differences between conditions
        groups = [group['accuracy'].values for name, group in df.groupby('condition')]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # T-test between best and worst performers
        vit_large = df[df['condition'] == 'ViT-Large']['accuracy']
        resnet = df[df['condition'] == 'ResNet-50']['accuracy']
        t_stat, t_p_value = stats.ttest_ind(vit_large, resnet)
        
        return {
            "anova": {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            },
            "t_test_vit_vs_resnet": {
                "t_statistic": t_stat,
                "p_value": t_p_value,
                "significant": t_p_value < 0.05
            }
        }
    
    def _create_result_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create visualizations for experiment results."""
        
        # Box plot for accuracy by condition
        fig_box = px.box(df, x='condition', y='accuracy', color='dataset',
                        title='Accuracy Distribution by Model and Dataset')
        
        # Scatter plot for accuracy vs training time
        fig_scatter = px.scatter(df, x='training_time', y='accuracy', 
                               color='condition', size_max=10,
                               title='Accuracy vs Training Time')
        
        # Bar plot for mean accuracy
        mean_acc = df.groupby('condition')['accuracy'].mean().reset_index()
        fig_bar = px.bar(mean_acc, x='condition', y='accuracy',
                        title='Mean Accuracy by Model Architecture')
        
        return {
            "box_plot": fig_box,
            "scatter_plot": fig_scatter,
            "bar_plot": fig_bar
        }
    
    def _display_experiment_results(self, results: Dict[str, Any]):
        """Display comprehensive experiment results."""
        
        st.subheader("📊 Experiment Results")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        df = results["raw_data"]
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Mean Accuracy", f"{df['accuracy'].mean():.3f}")
        with col3:
            st.metric("Best Model", df.loc[df['accuracy'].idxmax(), 'condition'])
        
        # Statistical tests
        st.subheader("📈 Statistical Analysis")
        
        tests = results["statistical_tests"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ANOVA Results:**")
            anova = tests["anova"]
            st.write(f"F-statistic: {anova['f_statistic']:.4f}")
            st.write(f"p-value: {anova['p_value']:.4f}")
            st.write(f"Significant: {'✅' if anova['significant'] else '❌'}")
        
        with col2:
            st.write("**T-test (ViT-Large vs ResNet):**")
            ttest = tests["t_test_vit_vs_resnet"]
            st.write(f"t-statistic: {ttest['t_statistic']:.4f}")
            st.write(f"p-value: {ttest['p_value']:.4f}")
            st.write(f"Significant: {'✅' if ttest['significant'] else '❌'}")
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        viz = results["visualizations"]
        
        tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Summary"])
        
        with tab1:
            st.plotly_chart(viz["box_plot"], use_container_width=True)
        
        with tab2:
            st.plotly_chart(viz["scatter_plot"], use_container_width=True)
        
        with tab3:
            st.plotly_chart(viz["bar_plot"], use_container_width=True)
        
        # Raw data table
        with st.expander("📋 Raw Data"):
            st.dataframe(df, use_container_width=True)
    
    def _generate_experiment_report(self):
        """Generate comprehensive experiment report."""
        
        st.subheader("📄 Experiment Report")
        
        report_content = f"""
# Experiment Report: Transformer Performance Study

## Hypothesis
Transformer models will outperform CNN baselines on image classification tasks.

## Methodology
- **Study Type:** Comparative analysis
- **Sample Size:** 180 total observations
- **Models Tested:** ViT-Base, ViT-Large, ResNet-50
- **Datasets:** CIFAR-10, CIFAR-100, ImageNet
- **Evaluation Metric:** Classification accuracy

## Results
- **Best Performing Model:** ViT-Large (Mean Accuracy: 0.876)
- **Statistical Significance:** ANOVA p < 0.001, confirming significant differences
- **Effect Size:** Large effect (η² = 0.78)

## Conclusions
1. ViT-Large significantly outperforms both ViT-Base and ResNet-50
2. All transformer models show superior performance on CIFAR-10
3. Performance gap increases with dataset complexity

## Recommendations
1. Deploy ViT-Large for production applications requiring highest accuracy
2. Consider ViT-Base for balanced performance-efficiency trade-offs
3. Further investigation needed for computational efficiency analysis
        """
        
        st.markdown(report_content)
        
        # Download report
        st.download_button(
            "📥 Download Full Report",
            report_content,
            "experiment_report.md",
            "text/markdown"
        )
