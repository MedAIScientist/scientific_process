import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from typing import Dict, List, Any, Tuple
import io
import subprocess
import os
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.metrics import silhouette_score, silhouette_samples

# CAMEL Integration imports
import asyncio
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from camel.types import ModelPlatformType, ModelType

# Local imports with error handling
try:
    from research_agent import ResearchAgent
    from data_analysis_agent import DataAnalysisAgent
    CAMEL_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"CAMEL agents not available: {e}")
    ResearchAgent = None

# Async wrapper for CAMEL integration
async def generate_scientific_paper_async(hypothesis_data, csv_data, pubmed_links=None, output_format="markdown"):
    """Async wrapper for CAMEL paper generation"""
    if not CAMEL_AVAILABLE:
        return {"status": "error", "error": "CAMEL agents not available"}
    
    try:
        integration = ScientificProcessIntegration()
        
        # Save CSV temporarily
        csv_path = "temp_data.csv"
        csv_data.to_csv(csv_path, index=False)
        
        # Convert hypothesis data to CAMEL format
        camel_input = integrate_with_hypothesis_generator(
            hypothesis_data=hypothesis_data,
            csv_data=csv_data,
            pubmed_links=pubmed_links
        )
        
        # Generate paper
        result = await integration.paper_generator.generate_paper(camel_input)
        
        if result["status"] == "success":
            # Export paper
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_file = integration.paper_generator.export_paper(
                paper=result["paper"],
                output_format=output_format,
                output_path=f"scientific_paper_{timestamp}"
            )
            result["output_file"] = output_file
        
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
            
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Function to run async code in Streamlit
def run_camel_generation(hypothesis_data, csv_data, pubmed_links, output_format):
    """Run CAMEL generation in Streamlit context"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            generate_scientific_paper_async(hypothesis_data, csv_data, pubmed_links, output_format)
        )
        loop.close()
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Set page configuration
st.set_page_config(
    page_title="CSV hypothesis Generator",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("CSV Data Analyzer & hypothesis Generator")
st.write("Upload your CSV file to get a summary and potential research hypotheses")


# Function to call Gemini API
def call_gemini_api(prompt: str, api_key: str) -> str:
    """Call the Gemini API with the given prompt."""
    # url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}" #Gemini 2.0 Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent?key={api_key}" #Gemini New Release 2.5
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the response
        response_json = response.json()

        # Extract the generated text
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            if 'content' in response_json['candidates'][0] and 'parts' in response_json['candidates'][0]['content']:
                parts = response_json['candidates'][0]['content']['parts']
                if parts and 'text' in parts[0]:
                    return parts[0]['text']

        return "Error: Unable to parse API response"
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


# Function to analyze CSV and generate detailed statistics
def analyze_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the CSV data and return a dictionary of statistics and insights."""
    analysis = {}

    # Basic information
    analysis["row_count"] = len(df)
    analysis["column_count"] = len(df.columns)
    analysis["columns"] = list(df.columns)

    # Missing values
    missing_values = df.isnull().sum().to_dict()
    analysis["missing_values"] = {col: count for col, count in missing_values.items() if count > 0}
    analysis["total_missing"] = df.isnull().sum().sum()
    analysis["missing_percentage"] = (analysis["total_missing"] / (
                analysis["row_count"] * analysis["column_count"])) * 100

    # Column types and statistics
    column_types = {}
    column_stats = {}

    for col in df.columns:
        # Determine column type
        if pd.api.types.is_numeric_dtype(df[col]):
            if set(df[col].dropna().unique()) == {0, 1} or set(df[col].dropna().unique()) == {0.0, 1.0}:
                column_types[col] = "boolean"
            else:
                column_types[col] = "numeric"

            # Calculate statistics for numeric columns
            column_stats[col] = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "unique_values": int(df[col].nunique())
            }
        elif pd.api.types.is_datetime64_dtype(df[col]):
            column_types[col] = "datetime"
            column_stats[col] = {
                "min": str(df[col].min()),
                "max": str(df[col].max()),
                "unique_values": int(df[col].nunique())
            }
        else:
            # Check if it's a datetime string
            try:
                pd.to_datetime(df[col], errors='raise')
                column_types[col] = "datetime_string"
                column_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "most_common": df[col].value_counts().head(5).to_dict()
                }
            except:
                # It's a categorical or text column
                unique_count = df[col].nunique()
                if unique_count <= 10 or (unique_count / len(df) < 0.05):
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "text"

                column_stats[col] = {
                    "unique_values": int(unique_count),
                    "most_common": df[col].value_counts().head(5).to_dict()
                }

    analysis["column_types"] = column_types
    analysis["column_stats"] = column_stats

    # Correlations for numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) >= 2:
        correlation = df[numeric_cols].corr().round(2).where(lambda x: ~np.eye(len(x), dtype=bool)).stack()
        strong_correlations = correlation[abs(correlation) > 0.5]
        if not strong_correlations.empty:
            analysis["strong_correlations"] = {f"{idx[0]} & {idx[1]}": val for idx, val in strong_correlations.items()}

    return analysis


# Function to create visualizations
def create_visualizations(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for key insights and return them as a dictionary."""
    visualizations = {}

    # Only create visualizations if the dataframe is not too large
    if len(df) > 100000:
        return {"message": "Dataset too large for automatic visualization"}

    # Missing values heatmap
    if analysis["total_missing"] > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
        plt.title('Missing Values Heatmap')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations["missing_values_heatmap"] = buf
        plt.close(fig)

    # Distribution plots for numeric columns (limit to 5)
    numeric_cols = [col for col in df.columns if analysis["column_types"].get(col) == "numeric"]
    if numeric_cols:
        for i, col in enumerate(numeric_cols[:5]):  # Limit to 5 columns
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            visualizations[f"distribution_{col}"] = buf
            plt.close(fig)

    # Correlation heatmap for numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[numeric_cols].corr().round(2)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Correlation Heatmap')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations["correlation_heatmap"] = buf
        plt.close(fig)

    return visualizations


# Function to create the prompt for Gemini
def create_gemini_prompt(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    """Create a detailed prompt for Gemini based on the CSV analysis."""
    prompt = f"""You are a data scientist analyzing a dataset. Based on the following information about a CSV file, provide:
1. A concise summary of the dataset
2. Three creative and specific research hypotheses that could be investigated using this data
3. For each hypothesis, explain the rationale behind it based on the data patterns

Dataset Information:
- Rows: {analysis['row_count']}
- Columns: {analysis['column_count']}
- Column Names: {', '.join(analysis['columns'])}

Column Types and Statistics:
"""

    # Add column type and statistics information
    for col in analysis['columns']:
        col_type = analysis['column_types'].get(col, "unknown")
        prompt += f"\n{col} ({col_type}):\n"

        if col in analysis['column_stats']:
            stats = analysis['column_stats'][col]
            for stat_name, stat_value in stats.items():
                if stat_name == "most_common" and isinstance(stat_value, dict):
                    # Format most common values
                    most_common_str = ", ".join([f"'{k}': {v}" for k, v in list(stat_value.items())[:3]])
                    prompt += f"  - {stat_name}: {most_common_str}\n"
                else:
                    prompt += f"  - {stat_name}: {stat_value}\n"

    # Add correlation information if available
    if "strong_correlations" in analysis and analysis["strong_correlations"]:
        prompt += "\nStrong Correlations:\n"
        for pair, value in analysis["strong_correlations"].items():
            prompt += f"- {pair}: {value}\n"

    # Add missing values information
    if analysis["total_missing"] > 0:
        prompt += f"\nMissing Values: {analysis['total_missing']} ({analysis['missing_percentage']:.2f}% of all data points)\n"
        for col, count in analysis["missing_values"].items():
            if count > 0:
                percent = (count / analysis["row_count"]) * 100
                prompt += f"- {col}: {count} missing values ({percent:.2f}%)\n"

    # Add final instructions
    prompt += """
Based on this information, please provide:
1. A summary of the dataset (3-4 sentences)
2. Three specific, creative, and testable research hypotheses that could be investigated with this data
3. For each hypothesis, provide a brief rationale based on the patterns in the data

Format your response as follows:
### Summary
[Your summary here]

### hypothesis 1
[Specific hypothesis statement]
**Rationale:** [Explanation based on the data]

### hypothesis 2
[Specific hypothesis statement]
**Rationale:** [Explanation based on the data]

### hypothesis 3
[Specific hypothesis statement]
**Rationale:** [Explanation based on the data]
"""

    return prompt


# Create an API key input field
api_key = st.text_input("Enter your Gemini API Key", type="password")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and api_key:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)

        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Preview", "Statistical Analysis", "Hypotheses", "Methods Selection", "Data Analysis", "🤖 CAMEL Paper Generator"])

        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            # Show basic info
            st.subheader("Basic Information")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

            # Show column types
            st.subheader("Column Information")
            column_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(column_info)

        with tab2:
            # Analyze the CSV
            with st.spinner("Analyzing data..."):
                analysis = analyze_csv(df)

                # Show summary stats
                st.subheader("Statistical Summary")

                # Display numeric column statistics
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if numeric_cols:
                    st.write("Numeric Columns Summary:")
                    st.dataframe(df[numeric_cols].describe())

                # Display categorical column information
                categorical_cols = [col for col in df.columns if analysis["column_types"].get(col) == "categorical"]
                if categorical_cols:
                    st.write("Categorical Columns Summary:")
                    for col in categorical_cols:
                        st.write(f"**{col}** - Top values:")
                        st.dataframe(df[col].value_counts().head(5).reset_index())

                # Show missing values
                if analysis["total_missing"] > 0:
                    st.subheader("Missing Values")
                    st.write(
                        f"Total missing values: {analysis['total_missing']} ({analysis['missing_percentage']:.2f}% of all data)")

                    missing_df = pd.DataFrame({
                        'Column': list(analysis["missing_values"].keys()),
                        'Missing Count': list(analysis["missing_values"].values()),
                        'Percentage': [count / analysis["row_count"] * 100 for count in
                                       analysis["missing_values"].values()]
                    })

                    if not missing_df.empty:
                        st.dataframe(missing_df)

                # Show correlations
                if "strong_correlations" in analysis and analysis["strong_correlations"]:
                    st.subheader("Strong Correlations")
                    corr_df = pd.DataFrame({
                        'Variables': list(analysis["strong_correlations"].keys()),
                        'Correlation': list(analysis["strong_correlations"].values())
                    })
                    st.dataframe(corr_df)

                # Create and display visualizations
                visualizations = create_visualizations(df, analysis)

                if visualizations and not isinstance(visualizations, dict) or (
                        isinstance(visualizations, dict) and "message" not in visualizations):
                    st.subheader("Visualizations")

                    # Display correlation heatmap
                    if "correlation_heatmap" in visualizations:
                        st.image(visualizations["correlation_heatmap"], caption="Correlation Heatmap")

                    # Display missing values heatmap
                    if "missing_values_heatmap" in visualizations:
                        st.image(visualizations["missing_values_heatmap"], caption="Missing Values Heatmap")

                    # Display distribution plots
                    dist_plots = [key for key in visualizations.keys() if key.startswith("distribution_")]
                    if dist_plots:
                        st.subheader("Distribution Plots")
                        for key in dist_plots:
                            col_name = key.replace("distribution_", "")
                            st.image(visualizations[key], caption=f"Distribution of {col_name}")

        with tab3:
            # Generate hypotheses using Gemini
            st.subheader("Data-Driven Hypotheses")

            with st.spinner("Generating hypotheses with Gemini..."):
                # Create the prompt
                prompt = create_gemini_prompt(df, analysis)

                # Call Gemini API
                with st.expander("View prompt sent to Gemini"):
                    st.text(prompt)

                gemini_response = call_gemini_api(prompt, api_key)

                # Display the response
                st.markdown(gemini_response)

                # Process the hypotheses using the pipeline
                from hypothesis_pipeline import integrate_with_streamlit

                print("[DEBUG] About to process Gemini response with integrate_with_streamlit")
                hypothesis_data = integrate_with_streamlit(gemini_response)
                
                print(f"[DEBUG] integrate_with_streamlit returned: {hypothesis_data}")

                if hypothesis_data["status"] == "success":
                    # Store the structured data in session state for use in other tabs
                    st.session_state.hypothesis_data = hypothesis_data
                    print(f"[DEBUG] Stored hypothesis_data in session_state with {len(hypothesis_data.get('hypotheses', []))} hypotheses")
                    
                    # Show success message
                    st.success(f"✅ Successfully generated {len(hypothesis_data.get('hypotheses', []))} hypotheses!")

                    # Add option to download the hypotheses as JSON
                    if st.button("Download Hypotheses as JSON"):
                        from hypothesis_pipeline import HypothesisPipeline

                        pipeline = HypothesisPipeline()
                        pipeline.extract_from_gemini_response(gemini_response)
                        pipeline.export_to_json("hypotheses_output.json")
                        st.success("Hypotheses saved to hypotheses_output.json")

                    # Show structured data in expandable section
                    with st.expander("View structured hypothesis data"):
                        st.json(hypothesis_data)
                        
                    # Debug what's actually stored
                    with st.expander("🔍 Debug Hypothesis Storage"):
                        st.write("**What's stored in session_state:**")
                        st.write(f"Keys: {list(st.session_state.keys())}")
                        if 'hypothesis_data' in st.session_state:
                            st.write("hypothesis_data exists and contains:")
                            st.json(st.session_state.hypothesis_data)
                        else:
                            st.error("hypothesis_data NOT found in session_state!")
                            
                else:
                    st.error(f"❌ Failed to process hypotheses: {hypothesis_data.get('message', 'Unknown error')}")
                    
                    # Debug the failure
                    with st.expander("🔍 Debug Hypothesis Failure"):
                        st.write("**Raw Gemini Response (first 1000 chars):**")
                        st.text(gemini_response[:1000])
                        st.write("**integrate_with_streamlit result:**")
                        st.json(hypothesis_data)

        with tab4:
            st.subheader("Statistical Methods Selection")

            # Check if we have hypothesis data - improved check
            hypothesis_available = False
            hypothesis_count = 0
            
            if 'hypothesis_data' in st.session_state:
                hyp_data = st.session_state.hypothesis_data
                if isinstance(hyp_data, dict) and hyp_data.get("status") == "success":
                    hypotheses = hyp_data.get("hypotheses", [])
                    if hypotheses and len(hypotheses) > 0:
                        hypothesis_available = True
                        hypothesis_count = len(hypotheses)
            
            if not hypothesis_available:
                st.info(f"""
                📋 **No Valid Hypotheses Available**
                
                Debug info:
                - Session state has hypothesis_data: {'hypothesis_data' in st.session_state}
                - Data status: {st.session_state.get('hypothesis_data', {}).get('status', 'Not found')}
                - Hypotheses count: {len(st.session_state.get('hypothesis_data', {}).get('hypotheses', []))}
                
                Please generate hypotheses first in the **Hypotheses** tab.
                """)
                
                # Add debug expander here too
                with st.expander("🔍 Debug Methods Selection"):
                    st.write("Session state keys:", list(st.session_state.keys()))
                    if 'hypothesis_data' in st.session_state:
                        st.json(st.session_state.hypothesis_data)
            else:
                st.success(f"✅ Found {hypothesis_count} hypotheses. Select statistical methods below:")
                
                st.write("Based on your hypotheses, here are recommended statistical methods:")

                # Display hypotheses and let users select methods for each
                for i, hyp_data in enumerate(st.session_state.hypothesis_data["hypotheses"]):
                    st.write(f"### Hypothesis {i + 1}")
                    st.write(hyp_data.get("hypothesis", "No hypothesis text"))

                    # Create method selection options
                    st.write("#### Select appropriate statistical methods:")

                    # These would be dynamically generated based on the hypothesis
                    # For now we'll use some common options as an example
                    methods = st.multiselect(
                        f"Methods for hypothesis {i + 1}",
                        ["t-test", "ANOVA", "Regression Analysis", "Chi-square", "Correlation Analysis"],
                        key=f"methods_{i}"
                    )

                    # Store selected methods
                    if 'selected_methods' not in st.session_state:
                        st.session_state.selected_methods = {}

                    st.session_state.selected_methods[i] = methods

                # Button to proceed with analysis
                if st.button("Run Statistical Analysis"):
                    st.info("This would run the selected statistical methods on your data.")
                    # Here you would call functions to perform the actual analysis
                    # based on st.session_state.selected_methods

        with tab5:
            st.subheader("Data Analysis (Scikit-Learn)")

            # Select target variable
            target_col = st.selectbox("Select target column for supervised analysis", df.columns)
            features = [col for col in df.columns if col != target_col]

            # Encode categorical variables
            X = df[features].copy()
            y = df[target_col].copy()
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            if y.dtype == 'object' or y.dtype.name == 'category':
                y = LabelEncoder().fit_transform(y.astype(str))

            # Feature Importance
            if st.button("Show Feature Importances (Random Forest)"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                importances = model.feature_importances_
                importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                st.dataframe(importance_df.sort_values('Importance', ascending=False))

            # Clustering
            st.write("### KMeans Clustering (on first 2 PCA components)")
            st.info(
                """
                **What does this plot show?**  
                - Each dot is a row (sample) from your dataset, projected onto two new axes (PCA 1 and PCA 2) that capture the most important patterns in your data.  
                - The color of each dot shows which cluster it belongs to, as determined by the KMeans algorithm.  
                - The red X's are the centroids (centers) of each cluster.  

                **How to interpret:**  
                - Dots with the same color are considered similar by the algorithm.  
                - If you see clear, separate colored regions, your data has distinct groups.  
                - If the colors are mixed, the groups are less distinct.  
                - You can use the cluster sizes below to see how many samples are in each group.
                """
            )
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_pca)
            centers = kmeans.cluster_centers_

            # Plotly interactive scatter plot with hover tooltips
            plot_df = pd.DataFrame({
                'PCA 1': X_pca[:, 0],
                'PCA 2': X_pca[:, 1],
                'Cluster': clusters,
                'Index': df.index
            })
            fig = px.scatter(
                plot_df, x='PCA 1', y='PCA 2', color=plot_df['Cluster'].astype(str),
                hover_data=['Index'],
                title="KMeans Clustering (PCA-reduced data)",
                labels={'color': 'Cluster'}
            )
            # Add centroids
            for i, (cx, cy) in enumerate(centers):
                fig.add_scatter(x=[cx], y=[cy], mode='markers', marker=dict(symbol='x', size=15, color='red'),
                                name=f'Centroid {i}')
            st.plotly_chart(fig, use_container_width=True)

            # Show cluster sizes
            cluster_sizes = pd.Series(clusters).value_counts().sort_index()
            st.write("#### Cluster Sizes:")
            st.write({f"Cluster {i}": int(size) for i, size in cluster_sizes.items()})

            # 1. Show cluster feature means
            st.write("#### Cluster Feature Means:")
            feature_means = pd.DataFrame(X, columns=features).copy()
            feature_means['Cluster'] = clusters
            means_table = feature_means.groupby('Cluster').mean()
            st.dataframe(means_table)

            # 2. Interactive data table for cluster members
            st.write("#### View Data for a Selected Cluster:")
            selected_cluster = st.selectbox("Select cluster to view its members", options=sorted(means_table.index))
            cluster_rows = df.iloc[feature_means[feature_means['Cluster'] == selected_cluster].index]
            st.dataframe(cluster_rows.head(100))
            st.caption("Showing first 100 rows of the selected cluster.")

            # 4. Silhouette score and plot
            st.write("#### Silhouette Score:")
            sil_score = silhouette_score(X_pca, clusters)
            st.write(f"Silhouette Score: {sil_score:.3f}")
            sil_samples = silhouette_samples(X_pca, clusters)
            sil_df = pd.DataFrame({'Cluster': clusters, 'Silhouette': sil_samples})
            sil_fig = px.box(sil_df, x='Cluster', y='Silhouette', points='all', title='Silhouette Scores by Cluster')
            st.plotly_chart(sil_fig, use_container_width=True)

        with tab6:
            st.subheader("🤖 CAMEL Scientific Paper Generator")
            st.markdown("""
            **Transform your data analysis into a complete scientific paper using AI agents!**
            
            This system uses a dedicated script that runs the CAMEL multi-agent system separately for maximum reliability.
            """)
            
            # Check if hypotheses are available
            hypothesis_available = False
            hypothesis_count = 0
            
            if 'hypothesis_data' in st.session_state:
                hyp_data = st.session_state.hypothesis_data
                if isinstance(hyp_data, dict) and hyp_data.get("status") == "success":
                    hypotheses = hyp_data.get("hypotheses", [])
                    if hypotheses and len(hypotheses) > 0:
                        hypothesis_available = True
                        hypothesis_count = len(hypotheses)
            
            # Status check
            st.subheader("📋 System Status")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                openai_status = "✅ Set" if os.getenv('OPENAI_API_KEY') else "❌ Missing"
                st.metric("OpenAI API", openai_status)
            
            with col2:
                semantic_status = "✅ Set" if os.getenv('SEMANTIC_SCHOLAR_API_KEY') else "⚠️ Optional"
                st.metric("Semantic Scholar", semantic_status)
            
            with col3:
                hyp_status = f"✅ {hypothesis_count}" if hypothesis_available else "❌ Missing"
                st.metric("Hypotheses", hyp_status)
            
            if not hypothesis_available:
                st.warning("""
                ⚠️ **No Hypotheses Available**
                
                Please generate hypotheses first in the **Hypotheses** tab, then return here.
                """)
            elif not os.getenv('OPENAI_API_KEY'):
                st.error("""
                ❌ **OpenAI API Key Required**
                
                The OpenAI API key is required for CAMEL agents. Make sure it's configured in your environment.
                """)
            else:
                st.success(f"✅ **Ready to generate paper** with {hypothesis_count} hypotheses")
                
                # Configuration
                st.subheader("📋 Paper Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    output_format = st.selectbox(
                        "📄 Output Format",
                        ["markdown", "latex", "json", "text"],
                        help="Choose the format for your scientific paper"
                    )
                
                with col2:
                    paper_name = st.text_input(
                        "📝 Paper Name",
                        value="my_scientific_paper",
                        help="Name for your paper (without extension)"
                    )
                
                # Show hypotheses preview
                with st.expander("📋 Preview Your Hypotheses"):
                    for i, hyp in enumerate(st.session_state.hypothesis_data['hypotheses']):
                        st.write(f"**Hypothesis {i+1}:** {hyp.get('hypothesis', 'No hypothesis text')}")
                        st.write(f"*Rationale:* {hyp.get('rationale', 'No rationale provided')}")
                        st.write("---")
                
                # Generate paper button
                st.subheader("🚀 Generate Scientific Paper")
                
                if st.button("🤖 Generate Complete Scientific Paper", type="primary"):
                    with st.spinner("Preparing files for paper generation..."):
                        try:
                            # Save CSV data
                            csv_filename = "temp_data_for_camel.csv"
                            df.to_csv(csv_filename, index=False)
                            
                            # Save hypotheses to JSON
                            hypotheses_filename = "temp_hypotheses_for_camel.json"
                            with open(hypotheses_filename, 'w') as f:
                                json.dump(st.session_state.hypothesis_data, f, indent=2)
                            
                            st.success("✅ Files prepared successfully!")
                            
                            # Show command to run
                            st.subheader("📋 Next Steps")
                            
                            st.info("""
                            🎯 **Ready to generate your paper!**
                            
                            The system has prepared your data. Now run the paper generator:
                            """)
                            
                            # Command to copy
                            command = f"python3 camel_paper_generator.py --csv {csv_filename} --hypotheses {hypotheses_filename} --output {paper_name} --format {output_format}"
                            
                            st.code(command, language="bash")
                            
                            # Instructions
                            st.markdown("""
                            **How to run:**
                            
                            1. **Open a new terminal** in your project directory
                            2. **Copy and paste** the command above
                            3. **Press Enter** and wait 5-8 minutes
                            4. **Your scientific paper** will be generated!
                            
                            **What happens during generation:**
                            - 📚 Literature review with Semantic Scholar & PubMed
                            - 📊 Advanced statistical analysis
                            - 🔬 Methodology development
                            - 📈 Results interpretation
                            - 💭 Discussion and implications
                            - ✍️ Paper writing and formatting
                            - 📝 Quality review and improvement
                            """)
                            
                            # Alternative: Auto-launch button
                            st.subheader("🚀 Or Launch Automatically")
                            
                            if st.button("🔥 Auto-Launch Paper Generator", type="secondary"):
                                import subprocess
                                
                                st.info("🚀 Launching paper generator in background...")
                                st.info("📊 Check your terminal for progress updates!")
                                
                                try:
                                    # Launch the script in background
                                    process = subprocess.Popen([
                                        "python3", "camel_paper_generator.py",
                                        "--csv", csv_filename,
                                        "--hypotheses", hypotheses_filename,
                                        "--output", paper_name,
                                        "--format", output_format
                                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                    
                                    st.success("✅ Paper generator launched!")
                                    st.info("""
                                    📋 **What's happening:**
                                    - The paper generator is running in the background
                                    - Check your terminal for real-time progress
                                    - Generation will take 5-8 minutes
                                    - Your paper will appear in the project directory
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"❌ Could not auto-launch: {e}")
                                    st.info("Please use the manual command above instead.")
                            
                        except Exception as e:
                            st.error(f"❌ Error preparing files: {e}")
            
            # Information about the system
            st.subheader("ℹ️ About CAMEL Paper Generator")
            
            with st.expander("🤖 How It Works"):
                st.markdown("""
                The CAMEL system uses **8 specialized AI agents** working together:
                
                1. **🔍 Research Agent** - Literature search via Semantic Scholar & PubMed
                2. **📊 Data Analysis Agent** - Advanced statistical analysis & ML
                3. **📚 Literature Review Agent** - Critical synthesis of research
                4. **🔬 Methodology Agent** - Study design and method selection
                5. **📈 Results Agent** - Results interpretation and presentation  
                6. **💭 Discussion Agent** - Implications and future research
                7. **✍️ Writing Agent** - Paper assembly and formatting
                8. **📝 Review Agent** - Quality assessment and improvement
                
                **Process:** Your CSV Data + Hypotheses → 8 AI Agents → Professional Scientific Paper
                """)
            
            with st.expander("⚙️ System Requirements"):
                st.markdown("""
                **Required:**
                - ✅ OpenAI API Key (for CAMEL agents)
                - ✅ Generated hypotheses from the Hypotheses tab
                - ✅ CAMEL-AI package installed
                
                **Optional:**
                - 📚 Semantic Scholar API Key (for literature search)
                - 💎 Gemini API Key (already configured)
                
                **Output Formats Available:**
                - 📝 **Markdown** (.md) - Easy to read and edit
                - 📄 **LaTeX** (.tex) - For academic journal submissions
                - 📊 **JSON** (.json) - Structured data format
                - 📃 **Text** (.txt) - Plain text format
                """)
            
            with st.expander("🎯 Expected Results"):
                st.markdown("""
                **Your generated paper will include:**
                
                - **Title & Abstract** - Professional summary of your research
                - **Introduction** - Literature review and research context
                - **Methodology** - Study design and statistical methods
                - **Results** - Data analysis results and interpretations
                - **Discussion** - Implications and significance
                - **Conclusions** - Key findings and future directions
                - **References** - Automatically formatted citations
                
                **Quality Features:**
                - Professional academic writing style
                - Statistically rigorous analysis
                - Comprehensive literature integration
                - Publication-ready formatting
                """)

    except Exception as e:
        st.error(f"Error processing the CSV file: {str(e)}")
        st.exception(e)
elif uploaded_file is not None and not api_key:
    st.warning("Please enter your Gemini API key to analyze the file.")
else:
    # Show instructions when no file is uploaded
    st.info("Please upload a CSV file and provide your Gemini API key to get started.")

    # Example of what the app does
    st.subheader("What this app does:")
    st.write("""
    1. Analyzes your CSV data file using statistical methods
    2. Provides visualizations of key patterns and relationships
    3. Uses Google's Gemini AI to generate creative research hypotheses
    4. Helps you identify potential research directions based on your data
    """)

    # Privacy notice
    st.subheader("Privacy Note:")
    st.write("""
    - Your data is processed locally and is not stored
    - Only statistical summaries are sent to the Gemini API, not your raw data
    - Your API key is not stored and is only used for the current session
    """)


# def launch_deep_research_repo():
#     # Get the absolute path to the other app.py
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     other_app_path = os.path.abspath(os.path.join(current_dir, "..", "..", "local-deep-research", "app.py"))
#
#     # Check if the other app exists
#     if os.path.exists(other_app_path):
#         try:
#             # Launch the other app using subprocess
#             subprocess.Popen(["python", other_app_path])
#             return True, f"Successfully launched app at {other_app_path}"
#         except Exception as e:
#             return False, f"Error launching app: {str(e)}"
#     else:
#         return False, f"Could not find app at: {other_app_path}"

import os
import subprocess

def launch_deep_research_repo():
    # Get the absolute path to the other repo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    other_repo_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "local-deep-research"))  # Adjusted your repo name here

    # Path to the app.py inside other_repo_dir
    app_path = os.path.join(other_repo_dir, "app.py")

    # Check if the app.py exists
    if os.path.exists(app_path):
        try:
            # Launch a new terminal process: cd into repo and run python app.py
            subprocess.Popen(f'cd "{other_repo_dir}" && python app.py', shell=True)
            return True, f"Successfully launched app at {app_path}"
        except Exception as e:
            return False, f"Error launching app: {str(e)}"
    else:
        return False, f"Could not find app at: {app_path}"



with st.sidebar:
    st.write("## Additional Tools")
    if st.button("Launch Deep Research App"):
        success, message = launch_deep_research_repo()
        if success:
            st.success(message)
        else:
            st.error(message)