import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from typing import Dict, List, Any, Tuple
import io
import os

# Set page configuration
st.set_page_config(
    page_title="CSV Hypothesis Generator",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("CSV Data Analyzer & Hypothesis Generator")
st.write("Upload your CSV file to get a summary and potential research hypotheses")


# Function to call Gemini API
def call_gemini_api(prompt: str, api_key: str) -> str:
    """Call the Gemini API with the given prompt."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
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

### Hypothesis 1
[Specific hypothesis statement]
**Rationale:** [Explanation based on the data]

### Hypothesis 2
[Specific hypothesis statement]
**Rationale:** [Explanation based on the data]

### Hypothesis 3
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
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Statistical Analysis", "Hypotheses"])

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
