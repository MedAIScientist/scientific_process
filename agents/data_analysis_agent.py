"""
Data Analysis Agent for Advanced Statistical Analysis

This agent specializes in conducting sophisticated data analysis beyond basic statistics,
including advanced statistical tests, machine learning insights, and data interpretation.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

from camel.agents import ChatAgent
from camel.messages import BaseMessage

# Statistical and ML imports
try:
    from scipy import stats
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import classification_report, regression_metrics
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Warning: Some statistical libraries not available: {e}")


@dataclass
class StatisticalTest:
    """Represents a statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    assumptions_met: bool = True


@dataclass
class AnalysisResult:
    """Comprehensive analysis result"""
    summary: str
    statistical_tests: List[StatisticalTest]
    model_results: Dict[str, Any]
    visualizations: List[str]
    insights: List[str]
    recommendations: List[str]


class DataAnalysisAgent:
    """
    Specialized agent for advanced data analysis and statistical testing
    """
    
    def __init__(self, model):
        """
        Initialize the data analysis agent
        
        Args:
            model: CAMEL model instance
        """
        self.model = model
        
        # Initialize the chat agent with specialized role
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Senior Data Scientist",
                content="""You are an expert data scientist and statistician specializing in:
                1. Advanced statistical analysis and hypothesis testing
                2. Machine learning model development and interpretation
                3. Data visualization and pattern recognition
                4. Causal inference and experimental design
                5. Statistical assumption validation
                
                Your analyses should be:
                - Methodologically rigorous
                - Clearly explained for scientific audiences
                - Focused on addressing research hypotheses
                - Mindful of statistical assumptions and limitations
                - Actionable and insightful
                
                Always provide statistical justification for your chosen methods."""
            )
        )
    
    async def conduct_advanced_analysis(self, 
                                      data: pd.DataFrame,
                                      hypotheses: List[Dict[str, str]],
                                      summary: str) -> Dict[str, Any]:
        """
        Conduct comprehensive advanced analysis of the dataset
        
        Args:
            data: The dataset to analyze
            hypotheses: Research hypotheses to test
            summary: Data summary from initial analysis
            
        Returns:
            Comprehensive analysis results
        """
        print("🔬 Conducting advanced data analysis...")
        
        # Clean and prepare data
        cleaned_data = await self._clean_and_prepare_data(data)
        
        # Select appropriate statistical tests
        test_plan = await self._create_statistical_test_plan(cleaned_data, hypotheses)
        
        # Execute statistical tests
        statistical_results = await self._execute_statistical_tests(cleaned_data, test_plan)
        
        # Conduct machine learning analysis
        ml_results = await self._conduct_ml_analysis(cleaned_data, hypotheses)
        
        # Generate insights and interpretations
        insights = await self._generate_insights(
            cleaned_data, statistical_results, ml_results, hypotheses
        )
        
        # Create visualizations
        visualizations = await self._create_advanced_visualizations(cleaned_data, hypotheses)
        
        return {
            "data_summary": self._summarize_data(cleaned_data),
            "statistical_tests": statistical_results,
            "machine_learning": ml_results,
            "insights": insights,
            "visualizations": visualizations,
            "methodology": test_plan,
            "recommendations": await self._generate_recommendations(insights, hypotheses)
        }
    
    async def _clean_and_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        
        cleaned_data = data.copy()
        
        # Handle missing values intelligently
        prompt = f"""
        Given this dataset with {data.shape[0]} rows and {data.shape[1]} columns:
        
        Columns: {list(data.columns)}
        Missing values per column: {data.isnull().sum().to_dict()}
        
        Recommend the best strategy for handling missing values for each column.
        Consider:
        - Data type (numeric, categorical, text)
        - Percentage of missing values
        - Pattern of missingness
        - Impact on analysis
        
        Provide recommendations in JSON format with column names as keys and strategies as values.
        Use strategies: 'drop_column', 'drop_rows', 'mean_impute', 'median_impute', 'mode_impute', 'forward_fill', 'interpolate'
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            strategies = json.loads(response.msgs[0].content)
            
            for column, strategy in strategies.items():
                if column in cleaned_data.columns:
                    if strategy == 'drop_column':
                        cleaned_data = cleaned_data.drop(columns=[column])
                    elif strategy == 'drop_rows':
                        cleaned_data = cleaned_data.dropna(subset=[column])
                    elif strategy == 'mean_impute':
                        cleaned_data[column].fillna(cleaned_data[column].mean(), inplace=True)
                    elif strategy == 'median_impute':
                        cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)
                    elif strategy == 'mode_impute':
                        cleaned_data[column].fillna(cleaned_data[column].mode()[0], inplace=True)
                    elif strategy == 'forward_fill':
                        cleaned_data[column].fillna(method='ffill', inplace=True)
                    elif strategy == 'interpolate':
                        cleaned_data[column].interpolate(inplace=True)
                        
        except Exception as e:
            print(f"Using default missing value strategy: {e}")
            # Default strategy: drop rows with too many missing values
            cleaned_data = cleaned_data.dropna(thresh=len(cleaned_data.columns) * 0.7)
        
        return cleaned_data
    
    async def _create_statistical_test_plan(self, 
                                          data: pd.DataFrame, 
                                          hypotheses: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create a comprehensive statistical testing plan"""
        
        data_info = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(data.select_dtypes(include=['object', 'category']).columns)
        }
        
        hypotheses_text = "\n".join([f"- {h['hypothesis']}" for h in hypotheses])
        
        prompt = f"""
        Create a comprehensive statistical testing plan for this dataset and hypotheses:
        
        Dataset Info:
        - Shape: {data_info['shape']}
        - Numeric columns: {data_info['numeric_columns']}
        - Categorical columns: {data_info['categorical_columns']}
        
        Hypotheses to test:
        {hypotheses_text}
        
        For each hypothesis, recommend specific statistical tests considering:
        1. Data types involved
        2. Sample size and distribution
        3. Independence assumptions
        4. Effect size estimation
        5. Multiple comparison corrections
        
        Provide a detailed plan in JSON format with:
        - hypothesis_id: corresponding to each hypothesis
        - primary_tests: list of main statistical tests to perform
        - secondary_tests: additional exploratory tests
        - variables_involved: which columns to analyze
        - assumptions_to_check: what to validate before testing
        - effect_size_measures: appropriate effect size calculations
        """
        
        response = self.agent.step(prompt)
        
        try:
            test_plan = json.loads(response.msgs[0].content)
        except:
            # Default test plan
            test_plan = {
                "general_analysis": {
                    "primary_tests": ["correlation_analysis", "distribution_tests"],
                    "secondary_tests": ["clustering_analysis"],
                    "variables_involved": data_info['numeric_columns'][:5],
                    "assumptions_to_check": ["normality", "independence"]
                }
            }
        
        return test_plan
    
    async def _execute_statistical_tests(self, 
                                       data: pd.DataFrame, 
                                       test_plan: Dict[str, Any]) -> List[StatisticalTest]:
        """Execute the planned statistical tests"""
        
        test_results = []
        
        for hypothesis_id, plan in test_plan.items():
            variables = plan.get('variables_involved', [])
            primary_tests = plan.get('primary_tests', [])
            
            for test_name in primary_tests:
                try:
                    if test_name == "correlation_analysis" and len(variables) >= 2:
                        result = self._perform_correlation_analysis(data, variables)
                        test_results.extend(result)
                    
                    elif test_name == "t_test" and len(variables) >= 2:
                        result = self._perform_t_test(data, variables)
                        if result:
                            test_results.append(result)
                    
                    elif test_name == "anova" and len(variables) >= 2:
                        result = self._perform_anova(data, variables)
                        if result:
                            test_results.append(result)
                    
                    elif test_name == "chi_square" and len(variables) >= 2:
                        result = self._perform_chi_square(data, variables)
                        if result:
                            test_results.append(result)
                    
                    elif test_name == "distribution_tests":
                        result = self._perform_distribution_tests(data, variables)
                        test_results.extend(result)
                    
                except Exception as e:
                    print(f"Error executing {test_name}: {e}")
        
        return test_results
    
    def _perform_correlation_analysis(self, data: pd.DataFrame, variables: List[str]) -> List[StatisticalTest]:
        """Perform correlation analysis"""
        results = []
        numeric_vars = [col for col in variables if col in data.select_dtypes(include=[np.number]).columns]
        
        if len(numeric_vars) >= 2:
            correlation_matrix = data[numeric_vars].corr()
            
            for i, var1 in enumerate(numeric_vars):
                for j, var2 in enumerate(numeric_vars):
                    if i < j:  # Avoid duplicates
                        corr_value = correlation_matrix.loc[var1, var2]
                        
                        # Perform significance test
                        try:
                            stat, p_value = stats.pearsonr(data[var1].dropna(), data[var2].dropna())
                            
                            # Effect size interpretation
                            if abs(corr_value) < 0.1:
                                interpretation = "negligible correlation"
                            elif abs(corr_value) < 0.3:
                                interpretation = "small correlation"
                            elif abs(corr_value) < 0.5:
                                interpretation = "medium correlation"
                            else:
                                interpretation = "large correlation"
                            
                            result = StatisticalTest(
                                test_name=f"Pearson correlation: {var1} vs {var2}",
                                statistic=corr_value,
                                p_value=p_value,
                                effect_size=abs(corr_value),
                                interpretation=interpretation
                            )
                            results.append(result)
                            
                        except Exception as e:
                            print(f"Error in correlation test for {var1} vs {var2}: {e}")
        
        return results
    
    def _perform_t_test(self, data: pd.DataFrame, variables: List[str]) -> Optional[StatisticalTest]:
        """Perform t-test analysis"""
        if len(variables) < 2:
            return None
        
        try:
            # Assuming first variable is continuous, second is binary grouping
            continuous_var = variables[0]
            group_var = variables[1]
            
            if data[group_var].nunique() == 2:
                groups = data[group_var].unique()
                group1_data = data[data[group_var] == groups[0]][continuous_var].dropna()
                group2_data = data[data[group_var] == groups[1]][continuous_var].dropna()
                
                # Perform independent t-test
                stat, p_value = stats.ttest_ind(group1_data, group2_data)
                
                # Calculate Cohen's d (effect size)
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                    (len(group2_data) - 1) * group2_data.var()) / 
                                   (len(group1_data) + len(group2_data) - 2))
                cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
                
                return StatisticalTest(
                    test_name=f"Independent t-test: {continuous_var} by {group_var}",
                    statistic=stat,
                    p_value=p_value,
                    effect_size=abs(cohens_d),
                    interpretation=f"Mean difference between groups: {group1_data.mean():.3f} vs {group2_data.mean():.3f}"
                )
        except Exception as e:
            print(f"Error in t-test: {e}")
        
        return None
    
    def _perform_anova(self, data: pd.DataFrame, variables: List[str]) -> Optional[StatisticalTest]:
        """Perform ANOVA analysis"""
        if len(variables) < 2:
            return None
        
        try:
            continuous_var = variables[0]
            group_var = variables[1]
            
            if data[group_var].nunique() > 2:
                groups = [data[data[group_var] == group][continuous_var].dropna() 
                         for group in data[group_var].unique()]
                
                stat, p_value = stats.f_oneway(*groups)
                
                # Calculate eta-squared (effect size)
                grand_mean = data[continuous_var].mean()
                ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in groups)
                ss_total = sum((data[continuous_var] - grand_mean)**2)
                eta_squared = ss_between / ss_total
                
                return StatisticalTest(
                    test_name=f"One-way ANOVA: {continuous_var} by {group_var}",
                    statistic=stat,
                    p_value=p_value,
                    effect_size=eta_squared,
                    interpretation=f"Variance explained by groups: {eta_squared:.3f}"
                )
        except Exception as e:
            print(f"Error in ANOVA: {e}")
        
        return None
    
    def _perform_chi_square(self, data: pd.DataFrame, variables: List[str]) -> Optional[StatisticalTest]:
        """Perform chi-square test for independence"""
        if len(variables) < 2:
            return None
        
        try:
            var1, var2 = variables[0], variables[1]
            
            # Create contingency table
            contingency_table = pd.crosstab(data[var1], data[var2])
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate Cramér's V (effect size)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            return StatisticalTest(
                test_name=f"Chi-square test: {var1} vs {var2}",
                statistic=chi2,
                p_value=p_value,
                effect_size=cramers_v,
                interpretation=f"Association strength (Cramér's V): {cramers_v:.3f}"
            )
        except Exception as e:
            print(f"Error in chi-square test: {e}")
        
        return None
    
    def _perform_distribution_tests(self, data: pd.DataFrame, variables: List[str]) -> List[StatisticalTest]:
        """Test for normality and other distribution properties"""
        results = []
        
        numeric_vars = [col for col in variables if col in data.select_dtypes(include=[np.number]).columns]
        
        for var in numeric_vars:
            try:
                variable_data = data[var].dropna()
                
                if len(variable_data) > 3:
                    # Shapiro-Wilk test for normality
                    stat, p_value = stats.shapiro(variable_data)
                    
                    result = StatisticalTest(
                        test_name=f"Shapiro-Wilk normality test: {var}",
                        statistic=stat,
                        p_value=p_value,
                        interpretation="Normal distribution" if p_value > 0.05 else "Non-normal distribution"
                    )
                    results.append(result)
                    
            except Exception as e:
                print(f"Error in distribution test for {var}: {e}")
        
        return results
    
    async def _conduct_ml_analysis(self, 
                                 data: pd.DataFrame, 
                                 hypotheses: List[Dict[str, str]]) -> Dict[str, Any]:
        """Conduct machine learning analysis for pattern discovery"""
        
        ml_results = {}
        
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] >= 2:
                # PCA Analysis
                ml_results['pca'] = self._perform_pca_analysis(numeric_data)
                
                # Clustering Analysis
                ml_results['clustering'] = self._perform_clustering_analysis(numeric_data)
                
                # Feature importance (if target variable can be inferred)
                ml_results['feature_importance'] = await self._analyze_feature_importance(data, hypotheses)
        
        except Exception as e:
            print(f"Error in ML analysis: {e}")
            ml_results['error'] = str(e)
        
        return ml_results
    
    def _perform_pca_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.fillna(data.mean()))
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            return {
                "explained_variance_ratio": explained_variance.tolist(),
                "cumulative_variance": cumulative_variance.tolist(),
                "n_components_95": int(np.argmax(cumulative_variance >= 0.95) + 1),
                "n_components_90": int(np.argmax(cumulative_variance >= 0.90) + 1),
                "loadings": pca.components_[:3].tolist()  # First 3 components
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _perform_clustering_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis"""
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.fillna(data.mean()))
            
            # Try different numbers of clusters
            silhouette_scores = []
            K_range = range(2, min(10, len(data) // 2))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                silhouette_avg = silhouette_score(scaled_data, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Find optimal number of clusters
            optimal_k = K_range[np.argmax(silhouette_scores)]
            
            # Perform final clustering
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
            final_labels = kmeans_final.fit_predict(scaled_data)
            
            return {
                "optimal_clusters": optimal_k,
                "silhouette_scores": dict(zip(K_range, silhouette_scores)),
                "cluster_centers": kmeans_final.cluster_centers_.tolist(),
                "cluster_sizes": [int(x) for x in np.bincount(final_labels)]
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_feature_importance(self, 
                                        data: pd.DataFrame, 
                                        hypotheses: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze feature importance using Random Forest"""
        
        # Ask the agent to identify potential target variables
        prompt = f"""
        Based on these hypotheses and the dataset columns, identify the most likely target variable(s) 
        for predictive modeling:
        
        Hypotheses:
        {chr(10).join([f"- {h['hypothesis']}" for h in hypotheses])}
        
        Available columns: {list(data.columns)}
        
        Suggest the column that would be the best target variable for prediction.
        Respond with just the column name.
        """
        
        response = self.agent.step(prompt)
        target_column = response.msgs[0].content.strip()
        
        if target_column in data.columns:
            try:
                # Prepare features and target
                X = data.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
                y = data[target_column]
                
                # Handle missing values
                X = X.fillna(X.mean())
                y = y.fillna(y.mean() if y.dtype in ['int64', 'float64'] else y.mode()[0])
                
                # Determine if regression or classification
                if y.dtype in ['int64', 'float64'] and y.nunique() > 10:
                    # Regression
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    importances = model.feature_importances_
                    
                    return {
                        "target_variable": target_column,
                        "task_type": "regression",
                        "feature_importances": dict(zip(X.columns, importances)),
                        "top_features": X.columns[np.argsort(importances)[-5:]].tolist()
                    }
                else:
                    # Classification
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    importances = model.feature_importances_
                    
                    return {
                        "target_variable": target_column,
                        "task_type": "classification",
                        "feature_importances": dict(zip(X.columns, importances)),
                        "top_features": X.columns[np.argsort(importances)[-5:]].tolist()
                    }
                    
            except Exception as e:
                return {"error": f"Feature importance analysis failed: {e}"}
        
        return {"message": "No suitable target variable identified"}
    
    async def _generate_insights(self, 
                               data: pd.DataFrame,
                               statistical_results: List[StatisticalTest],
                               ml_results: Dict[str, Any],
                               hypotheses: List[Dict[str, str]]) -> List[str]:
        """Generate insights from analysis results"""
        
        # Prepare summary of results
        significant_tests = [test for test in statistical_results if test.p_value < 0.05]
        
        results_summary = f"""
        Dataset: {data.shape[0]} rows, {data.shape[1]} columns
        
        Statistical Tests Performed: {len(statistical_results)}
        Significant Results (p<0.05): {len(significant_tests)}
        
        Key Findings:
        {chr(10).join([f"- {test.test_name}: p={test.p_value:.4f}, effect size={test.effect_size:.3f if test.effect_size else 'N/A'}" 
                      for test in significant_tests[:5]])}
        
        Machine Learning Results:
        {json.dumps(ml_results, indent=2) if ml_results else 'No ML analysis performed'}
        """
        
        prompt = f"""
        Based on the comprehensive analysis results, generate 5-7 key insights that:
        1. Directly address the research hypotheses
        2. Highlight the most important statistical findings
        3. Explain practical significance beyond statistical significance
        4. Identify unexpected patterns or relationships
        5. Suggest implications for further research
        
        Analysis Results:
        {results_summary}
        
        Original Hypotheses:
        {chr(10).join([f"- {h['hypothesis']}" for h in hypotheses])}
        
        Provide insights as a numbered list.
        """
        
        response = self.agent.step(prompt)
        insights = [insight.strip() for insight in response.msgs[0].content.split('\n') 
                   if insight.strip() and any(char.isdigit() for char in insight[:3])]
        
        return insights
    
    async def _create_advanced_visualizations(self, 
                                            data: pd.DataFrame, 
                                            hypotheses: List[Dict[str, str]]) -> List[str]:
        """Create advanced visualizations"""
        visualization_descriptions = []
        
        # This would create actual plots in a real implementation
        # For now, we'll describe what visualizations would be created
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) >= 2:
            visualization_descriptions.append(
                f"Correlation heatmap showing relationships between {len(numeric_cols)} numeric variables"
            )
            visualization_descriptions.append(
                f"Scatter plot matrix for key variables: {', '.join(numeric_cols[:4])}"
            )
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            visualization_descriptions.append(
                f"Box plots showing distribution of {numeric_cols[0]} across {categorical_cols[0]} categories"
            )
        
        if len(numeric_cols) >= 3:
            visualization_descriptions.append(
                "3D PCA visualization showing data structure in reduced dimensions"
            )
        
        return visualization_descriptions
    
    async def _generate_recommendations(self, 
                                      insights: List[str], 
                                      hypotheses: List[Dict[str, str]]) -> List[str]:
        """Generate actionable recommendations"""
        
        prompt = f"""
        Based on these insights and hypotheses, provide 5-7 specific, actionable recommendations for:
        1. Further statistical analysis that should be conducted
        2. Additional data that should be collected
        3. Methodological improvements for future studies
        4. Practical applications of the findings
        5. Potential limitations to address
        
        Insights:
        {chr(10).join(insights)}
        
        Original Hypotheses:
        {chr(10).join([f"- {h['hypothesis']}" for h in hypotheses])}
        
        Provide recommendations as a numbered list.
        """
        
        response = self.agent.step(prompt)
        recommendations = [rec.strip() for rec in response.msgs[0].content.split('\n') 
                          if rec.strip() and any(char.isdigit() for char in rec[:3])]
        
        return recommendations
    
    def _summarize_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create a comprehensive data summary"""
        return {
            "shape": data.shape,
            "missing_values": data.isnull().sum().to_dict(),
            "numeric_summary": data.describe().to_dict(),
            "categorical_summary": {col: data[col].value_counts().head().to_dict() 
                                  for col in data.select_dtypes(include=['object', 'category']).columns}
        } 