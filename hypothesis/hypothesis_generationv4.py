import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from typing import Dict, List, Any, Tuple, Optional
import io
import subprocess
import os
import threading
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Statistical Analysis Libraries
from scipy import stats
from scipy.stats import (
    normaltest, shapiro, kstest, anderson, jarque_bera,
    levene, bartlett, kruskal, mannwhitneyu, chi2_contingency,
    pearsonr, spearmanr, kendalltau, pointbiserialr
)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.signal import find_peaks
import itertools

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectKBest, f_classif
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Advanced CSV Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("🔬 Advanced CSV Data Analyzer & Hypothesis Generator")
st.write("Upload your CSV file for comprehensive statistical analysis and AI-powered hypothesis generation")


class PatternDetector:
    """Advanced pattern detection for comprehensive data analysis"""

    def __init__(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
        self.df = df
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.patterns = {}

    def detect_all_patterns(self) -> Dict[str, Any]:
        """Run comprehensive pattern detection"""
        patterns = {}

        # Clustering patterns
        patterns['clustering_analysis'] = self._advanced_clustering_analysis()

        # Feature interaction patterns
        patterns['feature_interactions'] = self._detect_feature_interactions()

        # Outlier ensemble detection
        patterns['ensemble_outliers'] = self._ensemble_outlier_detection()

        # Association patterns (categorical data)
        patterns['association_patterns'] = self._association_rule_mining()

        # Temporal patterns (if datetime exists)
        patterns['temporal_patterns'] = self._detect_temporal_patterns()

        # Dimensionality patterns
        patterns['dimensionality_insights'] = self._advanced_dimensionality_analysis()

        # Data quality patterns
        patterns['data_quality_patterns'] = self._detect_data_quality_patterns()

        self.patterns = patterns
        return patterns

    def _advanced_clustering_analysis(self) -> Dict[str, Any]:
        """Comprehensive clustering analysis with multiple algorithms"""
        if len(self.numeric_cols) < 2:
            return {"message": "Insufficient numeric columns for clustering"}

        numeric_data = self.df[self.numeric_cols].dropna()
        if len(numeric_data) < 10:
            return {"message": "Insufficient data for clustering"}

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        clustering_results = {}

        # 1. K-Means with optimal cluster detection
        kmeans_results = self._optimal_kmeans_analysis(scaled_data)
        clustering_results['kmeans'] = kmeans_results

        # 2. DBSCAN clustering
        dbscan_results = self._dbscan_analysis(scaled_data)
        clustering_results['dbscan'] = dbscan_results

        # 3. Hierarchical clustering
        hierarchical_results = self._hierarchical_clustering_analysis(scaled_data)
        clustering_results['hierarchical'] = hierarchical_results

        return clustering_results

    def _optimal_kmeans_analysis(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """Find optimal number of clusters using multiple methods"""
        max_clusters = min(10, len(scaled_data) // 5)

        inertias = []
        silhouette_scores = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_data)

            inertias.append(kmeans.inertia_)
            if len(set(labels)) > 1:
                sil_score = silhouette_score(scaled_data, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(-1)

        # Find optimal k
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2
        final_k = optimal_k_silhouette if max(silhouette_scores) > 0.3 else 3

        # Fit final model
        final_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
        final_labels = final_kmeans.fit_predict(scaled_data)

        return {
            'optimal_clusters': int(final_k),
            'optimal_labels': final_labels.tolist(),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'final_silhouette_score': float(silhouette_score(scaled_data, final_labels)),
            'cluster_centers': final_kmeans.cluster_centers_.tolist(),
            'cluster_sizes': np.bincount(final_labels).tolist()
        }

    def _dbscan_analysis(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """DBSCAN clustering for density-based patterns"""
        eps_values = [0.3, 0.5, 0.7, 1.0]
        best_eps = 0.5
        best_score = -1
        best_labels = None

        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(scaled_data)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters > 1 and n_noise < len(scaled_data) * 0.5:
                if n_clusters > 1:
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        score = silhouette_score(scaled_data[non_noise_mask], labels[non_noise_mask])
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_labels = labels

        if best_labels is not None:
            n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise = list(best_labels).count(-1)

            return {
                'n_clusters': int(n_clusters),
                'n_noise_points': int(n_noise),
                'noise_percentage': float((n_noise / len(best_labels)) * 100),
                'best_eps': float(best_eps),
                'silhouette_score': float(best_score),
                'labels': best_labels.tolist(),
                'cluster_sizes': np.bincount(best_labels[best_labels != -1]).tolist() if n_clusters > 0 else []
            }
        else:
            return {
                'message': 'No suitable DBSCAN clustering found',
                'n_clusters': 0
            }

    def _hierarchical_clustering_analysis(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """Hierarchical clustering analysis"""
        if len(scaled_data) > 1000:
            sample_indices = np.random.choice(len(scaled_data), 1000, replace=False)
            sample_data = scaled_data[sample_indices]
        else:
            sample_data = scaled_data

        linkage_matrix = linkage(sample_data, method='ward')

        max_clusters = min(10, len(sample_data) // 5)
        best_k = 3
        best_score = -1

        for k in range(2, max_clusters + 1):
            cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
            if len(set(cluster_labels)) == k:
                score = silhouette_score(sample_data, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        final_labels = fcluster(linkage_matrix, best_k, criterion='maxclust')

        return {
            'optimal_clusters': int(best_k),
            'silhouette_score': float(best_score),
            'cluster_sizes': np.bincount(final_labels).tolist(),
            'linkage_info': {
                'method': 'ward',
                'sample_size': len(sample_data)
            }
        }

    def _detect_feature_interactions(self) -> Dict[str, Any]:
        """Detect meaningful feature interactions"""
        if len(self.numeric_cols) < 2:
            return {"message": "Insufficient numeric columns for interaction analysis"}

        interactions = {}
        numeric_data = self.df[self.numeric_cols].dropna()

        # Polynomial feature importance
        if len(self.numeric_cols) <= 5:
            poly_interactions = self._polynomial_feature_analysis(numeric_data)
            interactions['polynomial_features'] = poly_interactions

        # Feature multiplication interactions
        multiplicative_interactions = self._multiplicative_interactions(numeric_data)
        interactions['multiplicative_interactions'] = multiplicative_interactions

        return interactions

    def _polynomial_feature_analysis(self, numeric_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze polynomial feature importance"""
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(numeric_data)
        feature_names = poly.get_feature_names_out(numeric_data.columns)

        interaction_terms = []
        for i, name in enumerate(feature_names):
            if ' ' in name and '*' in name:
                interaction_terms.append({
                    'feature_name': name,
                    'variance': float(np.var(poly_features[:, i])),
                    'feature_index': i
                })

        interaction_terms.sort(key=lambda x: x['variance'], reverse=True)

        return {
            'top_interactions': interaction_terms[:10],
            'total_interactions': len(interaction_terms)
        }

    def _multiplicative_interactions(self, numeric_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze multiplicative interactions between features"""
        interactions = []

        for i, col1 in enumerate(numeric_data.columns):
            for col2 in numeric_data.columns[i + 1:]:
                interaction = numeric_data[col1] * numeric_data[col2]

                corr1 = interaction.corr(numeric_data[col1])
                corr2 = interaction.corr(numeric_data[col2])

                interaction_strength = max(abs(corr1), abs(corr2))

                interactions.append({
                    'feature1': col1,
                    'feature2': col2,
                    'interaction_strength': float(interaction_strength),
                    'correlation_with_feature1': float(corr1),
                    'correlation_with_feature2': float(corr2),
                    'interaction_variance': float(interaction.var())
                })

        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)

        return {
            'strong_interactions': [i for i in interactions if i['interaction_strength'] > 0.7],
            'moderate_interactions': [i for i in interactions if 0.4 < i['interaction_strength'] <= 0.7],
            'all_interactions': interactions[:20]
        }

    def _ensemble_outlier_detection(self) -> Dict[str, Any]:
        """Ensemble outlier detection using multiple algorithms"""
        if len(self.numeric_cols) < 2:
            return {"message": "Insufficient numeric columns for outlier detection"}

        numeric_data = self.df[self.numeric_cols].dropna()
        if len(numeric_data) < 10:
            return {"message": "Insufficient data for outlier detection"}

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        outlier_methods = {}

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_outliers = iso_forest.fit_predict(scaled_data)
        outlier_methods['isolation_forest'] = {
            'outlier_count': int(np.sum(iso_outliers == -1)),
            'outlier_percentage': float((np.sum(iso_outliers == -1) / len(scaled_data)) * 100),
            'outlier_indices': np.where(iso_outliers == -1)[0].tolist()
        }

        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_outliers = lof.fit_predict(scaled_data)
        outlier_methods['local_outlier_factor'] = {
            'outlier_count': int(np.sum(lof_outliers == -1)),
            'outlier_percentage': float((np.sum(lof_outliers == -1) / len(scaled_data)) * 100),
            'outlier_indices': np.where(lof_outliers == -1)[0].tolist()
        }

        # One-Class SVM
        try:
            svm = OneClassSVM(nu=0.1, gamma='scale')
            svm_outliers = svm.fit_predict(scaled_data)
            outlier_methods['one_class_svm'] = {
                'outlier_count': int(np.sum(svm_outliers == -1)),
                'outlier_percentage': float((np.sum(svm_outliers == -1) / len(scaled_data)) * 100),
                'outlier_indices': np.where(svm_outliers == -1)[0].tolist()
            }
        except Exception as e:
            outlier_methods['one_class_svm'] = {'error': str(e)}

        # Ensemble consensus
        all_outlier_indices = []
        for method_data in outlier_methods.values():
            if 'outlier_indices' in method_data:
                all_outlier_indices.extend(method_data['outlier_indices'])

        outlier_consensus = {}
        for idx in set(all_outlier_indices):
            count = all_outlier_indices.count(idx)
            outlier_consensus[idx] = count

        high_confidence_outliers = [idx for idx, count in outlier_consensus.items() if count >= 2]

        return {
            'methods': outlier_methods,
            'ensemble_results': {
                'high_confidence_outliers': high_confidence_outliers,
                'outlier_consensus_scores': outlier_consensus,
                'total_suspected_outliers': len(set(all_outlier_indices))
            }
        }

    def _association_rule_mining(self) -> Dict[str, Any]:
        """Mine association rules from categorical data"""
        if len(self.categorical_cols) < 2:
            return {"message": "Insufficient categorical columns for association analysis"}

        cat_cols_subset = self.categorical_cols[:5]
        categorical_data = self.df[cat_cols_subset].dropna()

        if len(categorical_data) < 10:
            return {"message": "Insufficient categorical data"}

        associations = {}

        # Pairwise associations
        pairwise_associations = []
        for i, col1 in enumerate(cat_cols_subset):
            for col2 in cat_cols_subset[i + 1:]:
                contingency = pd.crosstab(categorical_data[col1], categorical_data[col2])

                if contingency.size > 0:
                    chi2, p_val, dof, expected = chi2_contingency(contingency)

                    n = contingency.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

                    pairwise_associations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'chi_square': float(chi2),
                        'p_value': float(p_val),
                        'cramers_v': float(cramers_v),
                        'association_strength': self._interpret_cramers_v(cramers_v),
                        'contingency_shape': contingency.shape
                    })

        associations['pairwise'] = pairwise_associations

        return associations

    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramér's V strength"""
        if cramers_v >= 0.3:
            return "Strong"
        elif cramers_v >= 0.1:
            return "Moderate"
        else:
            return "Weak"

    def _detect_temporal_patterns(self) -> Dict[str, Any]:
        """Detect temporal patterns if datetime columns exist"""
        datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()

        if not datetime_cols:
            return {"message": "No datetime columns found"}

        temporal_patterns = {}

        for dt_col in datetime_cols[:2]:
            dt_series = self.df[dt_col].dropna()

            if len(dt_series) < 10:
                continue

            temporal_info = {
                'column': dt_col,
                'date_range': {
                    'start': str(dt_series.min()),
                    'end': str(dt_series.max()),
                    'span_days': (dt_series.max() - dt_series.min()).days
                },
                'frequency_analysis': self._analyze_temporal_frequency(dt_series)
            }

            if self.numeric_cols:
                temporal_info['trends'] = self._analyze_temporal_trends(dt_col)

            temporal_patterns[dt_col] = temporal_info

        return temporal_patterns

    def _analyze_temporal_frequency(self, dt_series: pd.Series) -> Dict[str, Any]:
        """Analyze frequency patterns in datetime data"""
        df_temp = pd.DataFrame({'datetime': dt_series})
        df_temp['month'] = df_temp['datetime'].dt.month
        df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek

        frequency_patterns = {}

        if df_temp['month'].nunique() > 1:
            monthly_counts = df_temp['month'].value_counts().sort_index()
            frequency_patterns['monthly'] = {
                'pattern': monthly_counts.to_dict(),
                'peak_month': int(monthly_counts.idxmax()),
                'variation_coefficient': float(monthly_counts.std() / monthly_counts.mean())
            }

        return frequency_patterns

    def _analyze_temporal_trends(self, dt_col: str) -> Dict[str, Any]:
        """Analyze trends over time for numeric variables"""
        trends = {}

        for num_col in self.numeric_cols[:3]:
            temp_df = self.df[[dt_col, num_col]].dropna().sort_values(dt_col)

            if len(temp_df) < 5:
                continue

            x = np.arange(len(temp_df))
            y = temp_df[num_col].values

            slope, intercept = np.polyfit(x, y, 1)
            correlation = np.corrcoef(x, y)[0, 1]

            trends[num_col] = {
                'trend_slope': float(slope),
                'trend_strength': float(abs(correlation)),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'correlation_with_time': float(correlation)
            }

        return trends

    def _advanced_dimensionality_analysis(self) -> Dict[str, Any]:
        """Advanced dimensionality reduction and analysis"""
        if len(self.numeric_cols) < 3:
            return {"message": "Insufficient numeric columns for dimensionality analysis"}

        numeric_data = self.df[self.numeric_cols].dropna()
        if len(numeric_data) < 10:
            return {"message": "Insufficient data for dimensionality analysis"}

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        dimensionality_results = {}

        # PCA Analysis
        pca_results = self._enhanced_pca_analysis(scaled_data)
        dimensionality_results['pca'] = pca_results

        return dimensionality_results

    def _enhanced_pca_analysis(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """Enhanced PCA analysis with component interpretation"""
        pca = PCA()
        pca_transformed = pca.fit_transform(scaled_data)

        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        components_80 = np.where(cumulative_var >= 0.80)[0][0] + 1 if len(cumulative_var) > 0 else len(explained_var)

        component_interpretation = []
        for i in range(min(5, len(pca.components_))):
            loadings = pca.components_[i]
            loading_abs = np.abs(loadings)
            top_variables_idx = np.argsort(loading_abs)[-3:]

            top_variables = []
            for idx in reversed(top_variables_idx):
                top_variables.append({
                    'variable': self.numeric_cols[idx],
                    'loading': float(loadings[idx]),
                    'abs_loading': float(loading_abs[idx])
                })

            component_interpretation.append({
                'component': f'PC{i + 1}',
                'explained_variance': float(explained_var[i]),
                'top_contributing_variables': top_variables
            })

        return {
            'explained_variance_ratio': explained_var.tolist(),
            'cumulative_variance_ratio': cumulative_var.tolist(),
            'components_for_80_percent': int(components_80),
            'component_interpretation': component_interpretation,
            'dimensionality_reduction_potential': {
                'original_dimensions': len(self.numeric_cols),
                'effective_dimensions_80': int(components_80),
                'compression_ratio_80': float(components_80 / len(self.numeric_cols))
            }
        }

    def _detect_data_quality_patterns(self) -> Dict[str, Any]:
        """Detect data quality patterns"""
        quality_patterns = {}

        # Missing value patterns
        missing_patterns = {}
        missing_correlations = []

        # Check for correlated missingness
        missing_df = self.df.isnull()
        if missing_df.sum().sum() > 0:
            missing_corr = missing_df.corr()

            for i in range(len(missing_corr.columns)):
                for j in range(i + 1, len(missing_corr.columns)):
                    corr_val = missing_corr.iloc[i, j]
                    if abs(corr_val) > 0.5 and not np.isnan(corr_val):
                        missing_correlations.append({
                            'variable1': missing_corr.columns[i],
                            'variable2': missing_corr.columns[j],
                            'correlation': float(corr_val)
                        })

        missing_patterns['missing_correlations'] = missing_correlations
        quality_patterns['missing_value_patterns'] = missing_patterns

        return quality_patterns


class DeepStatisticalAnalyzer:
    """Advanced statistical analysis engine for comprehensive data exploration"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        self.analysis_results = {}

        # Performance optimizations
        self.max_correlations = 20
        self.max_categorical_analysis = 10
        self.sample_size_threshold = 10000

        # Pre-compute cleaned datasets
        self.numeric_data = df[self.numeric_cols].dropna() if self.numeric_cols else pd.DataFrame()
        self.is_large_dataset = len(df) > self.sample_size_threshold

    def comprehensive_analysis(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run comprehensive statistical analysis with performance optimizations"""

        results = {
            'basic_info': self._basic_info(),
            'univariate_analysis': self._univariate_analysis(),
        }

        # Advanced pattern detection
        if len(self.numeric_cols) >= 2:
            pattern_detector = PatternDetector(self.df, self.numeric_cols, self.categorical_cols)
            results['advanced_patterns'] = pattern_detector.detect_all_patterns()

        # For large datasets or quick mode, limit analysis scope
        if quick_mode or self.is_large_dataset:
            results.update({
                'correlation_analysis': self._quick_correlation_analysis(),
                'outlier_analysis': self._quick_outlier_analysis(),
                'distribution_analysis': self._quick_distribution_analysis(),
                'recommendations': self._generate_recommendations()
            })
        else:
            results.update({
                'bivariate_analysis': self._bivariate_analysis(),
                'multivariate_analysis': self._multivariate_analysis(),
                'outlier_analysis': self._outlier_analysis(),
                'distribution_analysis': self._distribution_analysis(),
                'correlation_analysis': self._advanced_correlation_analysis(),
                'statistical_tests': self._statistical_tests(),
                'feature_importance': self._feature_importance_analysis(),
                'recommendations': self._generate_recommendations()
            })

        self.analysis_results = results
        return results

    def _basic_info(self) -> Dict[str, Any]:
        """Enhanced basic information about the dataset"""
        return {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'missing_values': {
                'total': self.df.isnull().sum().sum(),
                'percentage': (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100,
                'by_column': self.df.isnull().sum().to_dict()
            },
            'duplicates': {
                'count': self.df.duplicated().sum(),
                'percentage': (self.df.duplicated().sum() / len(self.df)) * 100
            },
            'data_types': {
                'numeric': len(self.numeric_cols),
                'categorical': len(self.categorical_cols),
                'datetime': len(self.datetime_cols)
            },
            'column_types': {
                'numeric_columns': self.numeric_cols,
                'categorical_columns': self.categorical_cols,
                'datetime_columns': self.datetime_cols
            }
        }

    def _univariate_analysis(self) -> Dict[str, Any]:
        """Detailed univariate analysis for each column"""
        analysis = {'numeric': {}, 'categorical': {}}

        # Numeric columns analysis
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            # Basic statistics
            basic_stats = {
                'count': len(series),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'var': float(series.var()),
                'min': float(series.min()),
                'max': float(series.max()),
                'range': float(series.max() - series.min()),
                'q1': float(series.quantile(0.25)),
                'q3': float(series.quantile(0.75)),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25))
            }

            # Advanced statistics
            advanced_stats = {
                'skewness': float(stats.skew(series)),
                'kurtosis': float(stats.kurtosis(series)),
                'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else np.inf,
                'mad': float(stats.median_abs_deviation(series)),
                'unique_values': int(series.nunique()),
                'unique_percentage': float((series.nunique() / len(series)) * 100)
            }

            analysis['numeric'][col] = {**basic_stats, **advanced_stats}

        # Categorical columns analysis
        for col in self.categorical_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            value_counts = series.value_counts()
            analysis['categorical'][col] = {
                'count': len(series),
                'unique_values': int(series.nunique()),
                'unique_percentage': float((series.nunique() / len(series)) * 100),
                'mode': str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                'most_frequent': str(value_counts.index[0]),
                'most_frequent_count': int(value_counts.iloc[0]),
                'least_frequent': str(value_counts.index[-1]),
                'least_frequent_count': int(value_counts.iloc[-1]),
                'entropy': float(stats.entropy(value_counts.values)),
                'concentration_ratio': float(value_counts.iloc[0] / len(series)),
                'top_5_categories': {str(k): int(v) for k, v in value_counts.head().items()}
            }

        return analysis

    def _bivariate_analysis(self) -> Dict[str, Any]:
        """Comprehensive bivariate analysis"""
        analysis = {
            'numeric_numeric': {},
            'numeric_categorical': {},
            'categorical_categorical': {}
        }

        # Numeric-Numeric relationships
        for i, col1 in enumerate(self.numeric_cols):
            for col2 in self.numeric_cols[i + 1:]:
                series1 = self.df[col1].dropna()
                series2 = self.df[col2].dropna()

                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) < 3:
                    continue

                s1_common = series1.loc[common_idx]
                s2_common = series2.loc[common_idx]

                # Correlation analyses
                pearson_r, pearson_p = pearsonr(s1_common, s2_common)
                spearman_r, spearman_p = spearmanr(s1_common, s2_common)
                kendall_tau, kendall_p = kendalltau(s1_common, s2_common)

                analysis['numeric_numeric'][f"{col1}_vs_{col2}"] = {
                    'pearson_correlation': float(pearson_r),
                    'pearson_p_value': float(pearson_p),
                    'spearman_correlation': float(spearman_r),
                    'spearman_p_value': float(spearman_p),
                    'kendall_tau': float(kendall_tau),
                    'kendall_p_value': float(kendall_p),
                    'sample_size': len(common_idx),
                    'relationship_strength': self._interpret_correlation(abs(pearson_r))
                }

        # Numeric-Categorical relationships
        for num_col in self.numeric_cols:
            for cat_col in self.categorical_cols:
                try:
                    # Point-biserial correlation for binary categorical
                    if self.df[cat_col].nunique() == 2:
                        cat_encoded = pd.get_dummies(self.df[cat_col], drop_first=True).iloc[:, 0]
                        common_idx = self.df[num_col].dropna().index.intersection(cat_encoded.dropna().index)

                        if len(common_idx) > 3:
                            r_pb, p_pb = pointbiserialr(cat_encoded.loc[common_idx],
                                                        self.df[num_col].loc[common_idx])

                            analysis['numeric_categorical'][f"{num_col}_vs_{cat_col}"] = {
                                'point_biserial_correlation': float(r_pb),
                                'point_biserial_p_value': float(p_pb),
                                'sample_size': len(common_idx)
                            }

                    # ANOVA for multi-category
                    groups = [group[num_col].dropna() for name, group in self.df.groupby(cat_col)]
                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
                        f_stat, p_val = stats.f_oneway(*groups)

                        analysis['numeric_categorical'][f"{num_col}_vs_{cat_col}"] = {
                            **analysis['numeric_categorical'].get(f"{num_col}_vs_{cat_col}", {}),
                            'anova_f_statistic': float(f_stat),
                            'anova_p_value': float(p_val),
                            'effect_size_eta_squared': float(self._calculate_eta_squared(groups, f_stat))
                        }
                except Exception as e:
                    continue

        # Categorical-Categorical relationships
        for i, col1 in enumerate(self.categorical_cols):
            for col2 in self.categorical_cols[i + 1:]:
                try:
                    contingency_table = pd.crosstab(self.df[col1], self.df[col2])

                    if contingency_table.size > 0:
                        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                        n = contingency_table.sum().sum()

                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

                        analysis['categorical_categorical'][f"{col1}_vs_{col2}"] = {
                            'chi_square': float(chi2),
                            'chi_square_p_value': float(p_val),
                            'degrees_of_freedom': int(dof),
                            'cramers_v': float(cramers_v),
                            'sample_size': int(n),
                            'contingency_table_shape': contingency_table.shape
                        }
                except Exception as e:
                    continue

        return analysis

    def _multivariate_analysis(self) -> Dict[str, Any]:
        """Multivariate analysis including PCA and clustering insights"""
        analysis = {}

        if len(self.numeric_cols) >= 2:
            numeric_data = self.df[self.numeric_cols].dropna()

            if len(numeric_data) > 0:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)

                pca = PCA()
                pca_transformed = pca.fit_transform(scaled_data)

                explained_var = pca.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var)

                n_components_95 = np.where(cumulative_var >= 0.95)[0][0] + 1 if len(cumulative_var) > 0 else len(
                    explained_var)

                analysis['pca'] = {
                    'explained_variance_ratio': explained_var.tolist(),
                    'cumulative_variance_ratio': cumulative_var.tolist(),
                    'n_components_95_variance': int(n_components_95),
                    'total_components': len(explained_var),
                    'principal_component_loadings': {
                        f'PC{i + 1}': {col: float(loading) for col, loading in
                                       zip(self.numeric_cols, pca.components_[i])}
                        for i in range(min(3, len(pca.components_)))
                    }
                }

                # Multicollinearity check
                corr_matrix = numeric_data.corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append({
                                'variable1': corr_matrix.columns[i],
                                'variable2': corr_matrix.columns[j],
                                'correlation': float(corr_val)
                            })

                analysis['multicollinearity'] = {
                    'high_correlation_pairs': high_corr_pairs,
                    'multicollinearity_concern': len(high_corr_pairs) > 0
                }

        return analysis

    def _outlier_analysis(self) -> Dict[str, Any]:
        """Comprehensive outlier detection using multiple methods"""
        analysis = {}

        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if len(series) < 4:
                continue

            outliers = {}

            # IQR Method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]

            outliers['iqr'] = {
                'count': len(iqr_outliers),
                'percentage': float((len(iqr_outliers) / len(series)) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }

            # Z-Score Method
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]

            outliers['z_score'] = {
                'count': len(z_outliers),
                'percentage': float((len(z_outliers) / len(series)) * 100),
                'threshold': 3
            }

            # Modified Z-Score (using MAD)
            median = np.median(series)
            mad = stats.median_abs_deviation(series)
            modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros_like(series)
            modified_z_outliers = series[np.abs(modified_z_scores) > 3.5]

            outliers['modified_z_score'] = {
                'count': len(modified_z_outliers),
                'percentage': float((len(modified_z_outliers) / len(series)) * 100),
                'threshold': 3.5
            }

            analysis[col] = outliers

        # Multivariate outlier detection using Isolation Forest
        if len(self.numeric_cols) >= 2:
            numeric_data = self.df[self.numeric_cols].dropna()
            if len(numeric_data) > 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(numeric_data)

                analysis['multivariate_outliers'] = {
                    'isolation_forest': {
                        'count': int(np.sum(outlier_labels == -1)),
                        'percentage': float((np.sum(outlier_labels == -1) / len(numeric_data)) * 100),
                        'contamination_rate': 0.1
                    }
                }

        return analysis

    def _distribution_analysis(self) -> Dict[str, Any]:
        """Analyze distributions and test for normality and other properties"""
        analysis = {}

        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if len(series) < 8:
                continue

            # Normality tests
            normality_tests = {}

            # Shapiro-Wilk (best for small samples)
            if len(series) <= 5000:
                shapiro_stat, shapiro_p = shapiro(series)
                normality_tests['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }

            # D'Agostino-Pearson normality test
            try:
                dagostino_stat, dagostino_p = normaltest(series)
                normality_tests['dagostino_pearson'] = {
                    'statistic': float(dagostino_stat),
                    'p_value': float(dagostino_p),
                    'is_normal': dagostino_p > 0.05
                }
            except:
                pass

            # Jarque-Bera test
            try:
                jb_stat, jb_p = jarque_bera(series)
                normality_tests['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > 0.05
                }
            except:
                pass

            # Anderson-Darling test
            try:
                ad_result = anderson(series, dist='norm')
                normality_tests['anderson_darling'] = {
                    'statistic': float(ad_result.statistic),
                    'critical_values': ad_result.critical_values.tolist(),
                    'significance_levels': [15.0, 10.0, 5.0, 2.5, 1.0]
                }
            except:
                pass

            analysis[col] = {
                'normality_tests': normality_tests,
                'distribution_summary': {
                    'likely_normal': self._assess_normality(normality_tests),
                    'skewness_interpretation': self._interpret_skewness(stats.skew(series)),
                    'kurtosis_interpretation': self._interpret_kurtosis(stats.kurtosis(series))
                }
            }

        return analysis

    def _advanced_correlation_analysis(self) -> Dict[str, Any]:
        """Advanced correlation analysis including partial correlations and mutual information"""
        analysis = {}

        if len(self.numeric_cols) >= 2:
            numeric_data = self.df[self.numeric_cols].dropna()

            # Standard correlation matrix
            corr_matrix = numeric_data.corr()
            analysis['correlation_matrix'] = corr_matrix.to_dict()

            # Mutual Information
            if len(numeric_data) > 10:
                mi_scores = {}
                for target_col in self.numeric_cols:
                    features = [col for col in self.numeric_cols if col != target_col]
                    if features:
                        X = numeric_data[features]
                        y = numeric_data[target_col]
                        mi = mutual_info_regression(X, y, random_state=42)
                        mi_scores[target_col] = {feat: float(score) for feat, score in zip(features, mi)}

                analysis['mutual_information'] = mi_scores

            # Identify strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': float(corr_val),
                            'strength': self._interpret_correlation(abs(corr_val))
                        })

            analysis['strong_correlations'] = strong_correlations

        return analysis

    def _statistical_tests(self) -> Dict[str, Any]:
        """Perform various statistical tests based on data characteristics"""
        tests = {}

        # Test for equal variances (homoscedasticity) between groups
        for cat_col in self.categorical_cols:
            for num_col in self.numeric_cols:
                groups = [group[num_col].dropna() for name, group in self.df.groupby(cat_col)]
                groups = [g for g in groups if len(g) >= 3]

                if len(groups) >= 2:
                    # Levene's test for equal variances
                    try:
                        levene_stat, levene_p = levene(*groups)
                        tests[f"levene_{num_col}_by_{cat_col}"] = {
                            'test_type': 'levene_equal_variances',
                            'statistic': float(levene_stat),
                            'p_value': float(levene_p),
                            'equal_variances': levene_p > 0.05,
                            'groups_tested': len(groups)
                        }
                    except:
                        pass

                    # Kruskal-Wallis (non-parametric ANOVA)
                    try:
                        kw_stat, kw_p = kruskal(*groups)
                        tests[f"kruskal_wallis_{num_col}_by_{cat_col}"] = {
                            'test_type': 'kruskal_wallis',
                            'statistic': float(kw_stat),
                            'p_value': float(kw_p),
                            'significant_difference': kw_p < 0.05,
                            'groups_tested': len(groups)
                        }
                    except:
                        pass

        return tests

    def _feature_importance_analysis(self) -> Dict[str, Any]:
        """Analyze feature importance using various methods"""
        analysis = {}

        if len(self.numeric_cols) >= 2:
            numeric_data = self.df[self.numeric_cols].dropna()

            if len(numeric_data) > 10:
                for target_col in self.numeric_cols:
                    features = [col for col in self.numeric_cols if col != target_col]
                    if not features:
                        continue

                    X = numeric_data[features]
                    y = numeric_data[target_col]

                    # Random Forest feature importance
                    try:
                        y_binned = pd.cut(y, bins=min(5, len(y.unique())), labels=False)
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf.fit(X, y_binned)

                        importance_scores = {feat: float(score) for feat, score in
                                             zip(features, rf.feature_importances_)}

                        analysis[f"feature_importance_for_{target_col}"] = {
                            'method': 'random_forest',
                            'scores': importance_scores,
                            'most_important': max(importance_scores.items(), key=lambda x: x[1])[0]
                        }
                    except:
                        pass

        return analysis

    def _quick_correlation_analysis(self) -> Dict[str, Any]:
        """Quick correlation analysis for large datasets"""
        analysis = {}

        if len(self.numeric_cols) >= 2:
            numeric_data = self.df[self.numeric_cols].dropna()

            if len(numeric_data) > 0:
                # Sample if too large
                if len(numeric_data) > self.sample_size_threshold:
                    numeric_data = numeric_data.sample(n=self.sample_size_threshold, random_state=42)

                corr_matrix = numeric_data.corr()
                analysis['correlation_matrix'] = corr_matrix.to_dict()

                # Identify strong correlations
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strong_correlations.append({
                                'variable1': corr_matrix.columns[i],
                                'variable2': corr_matrix.columns[j],
                                'correlation': float(corr_val),
                                'strength': self._interpret_correlation(abs(corr_val))
                            })

                analysis['strong_correlations'] = strong_correlations[:self.max_correlations]

        return analysis

    def _quick_outlier_analysis(self) -> Dict[str, Any]:
        """Quick outlier analysis for large datasets"""
        analysis = {}

        for col in self.numeric_cols[:10]:  # Limit columns
            series = self.df[col].dropna()
            if len(series) < 4:
                continue

            # Sample if too large
            if len(series) > self.sample_size_threshold:
                series = series.sample(n=self.sample_size_threshold, random_state=42)

            outliers = {}

            # IQR Method only for quick analysis
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]

            outliers['iqr'] = {
                'count': len(iqr_outliers),
                'percentage': float((len(iqr_outliers) / len(series)) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }

            analysis[col] = outliers

        return analysis

    def _quick_distribution_analysis(self) -> Dict[str, Any]:
        """Quick distribution analysis for large datasets"""
        analysis = {}

        for col in self.numeric_cols[:10]:  # Limit columns
            series = self.df[col].dropna()
            if len(series) < 8:
                continue

            # Sample if too large
            if len(series) > 5000:
                series = series.sample(n=5000, random_state=42)

            normality_tests = {}

            # Shapiro-Wilk test only
            if len(series) <= 5000:
                shapiro_stat, shapiro_p = shapiro(series)
                normality_tests['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }

            analysis[col] = {
                'normality_tests': normality_tests,
                'distribution_summary': {
                    'likely_normal': self._assess_normality(normality_tests),
                    'skewness_interpretation': self._interpret_skewness(stats.skew(series)),
                    'kurtosis_interpretation': self._interpret_kurtosis(stats.kurtosis(series))
                }
            }

        return analysis

    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate statistical recommendations based on analysis"""
        recommendations = {
            'data_quality': [],
            'statistical_methods': [],
            'further_analysis': [],
            'pattern_insights': []
        }

        # Data quality recommendations
        basic_info = self.analysis_results.get('basic_info', {})
        if basic_info.get('missing_values', {}).get('percentage', 0) > 5:
            recommendations['data_quality'].append(
                "Consider investigating missing value patterns and implementing appropriate imputation strategies")

        if basic_info.get('duplicates', {}).get('percentage', 0) > 1:
            recommendations['data_quality'].append("Remove or investigate duplicate records")

        # Pattern-based insights
        patterns = self.analysis_results.get('advanced_patterns', {})
        if patterns:
            clustering = patterns.get('clustering_analysis', {})
            if clustering and clustering.get('kmeans', {}).get('optimal_clusters', 0) > 1:
                n_clusters = clustering['kmeans']['optimal_clusters']
                recommendations['pattern_insights'].append(
                    f"Dataset shows {n_clusters} distinct clusters - consider segmentation analysis")

            interactions = patterns.get('feature_interactions', {})
            strong_int = interactions.get('multiplicative_interactions', {}).get('strong_interactions', [])
            if strong_int:
                recommendations['pattern_insights'].append(
                    f"Found {len(strong_int)} strong feature interactions - consider interaction terms in models")

            outliers = patterns.get('ensemble_outliers', {})
            if outliers and 'ensemble_results' in outliers:
                high_conf = len(outliers['ensemble_results'].get('high_confidence_outliers', []))
                if high_conf > 0:
                    recommendations['pattern_insights'].append(
                        f"{high_conf} high-confidence outliers detected by multiple methods")

        # Statistical method recommendations
        normality_issues = []
        for col, dist_analysis in self.analysis_results.get('distribution_analysis', {}).items():
            if not dist_analysis.get('distribution_summary', {}).get('likely_normal', True):
                normality_issues.append(col)

        if normality_issues:
            recommendations['statistical_methods'].append(
                f"Consider non-parametric tests for variables: {', '.join(normality_issues[:3])}")

        # Strong correlation findings
        strong_corrs = self.analysis_results.get('correlation_analysis', {}).get('strong_correlations', [])
        if strong_corrs:
            recommendations['further_analysis'].append(
                "Investigate strong correlations for potential causality or confounding")

        return recommendations

    # Helper methods
    def _interpret_correlation(self, corr_value: float) -> str:
        """Interpret correlation strength"""
        if corr_value >= 0.7:
            return "Very Strong"
        elif corr_value >= 0.5:
            return "Strong"
        elif corr_value >= 0.3:
            return "Moderate"
        elif corr_value >= 0.1:
            return "Weak"
        else:
            return "Very Weak"

    def _calculate_eta_squared(self, groups: List, f_stat: float) -> float:
        """Calculate eta-squared effect size"""
        try:
            total_n = sum(len(g) for g in groups)
            between_df = len(groups) - 1
            within_df = total_n - len(groups)
            eta_squared = (between_df * f_stat) / (between_df * f_stat + within_df)
            return eta_squared
        except:
            return 0.0

    def _assess_normality(self, normality_tests: Dict) -> bool:
        """Assess if distribution is likely normal based on multiple tests"""
        normal_votes = 0
        total_tests = 0

        for test_name, test_result in normality_tests.items():
            if 'p_value' in test_result:
                total_tests += 1
                if test_result['p_value'] > 0.05:
                    normal_votes += 1

        return normal_votes / total_tests > 0.5 if total_tests > 0 else True

    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value"""
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif abs(skewness) < 1:
            return "Moderately skewed"
        else:
            return "Highly skewed"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value"""
        if abs(kurtosis) < 0.5:
            return "Approximately normal (mesokurtic)"
        elif kurtosis > 0.5:
            return "Heavy-tailed (leptokurtic)"
        else:
            return "Light-tailed (platykurtic)"


class ReportGenerator:
    """Generate comprehensive reports from statistical analysis"""

    def __init__(self, analysis_results: Dict[str, Any], df: pd.DataFrame):
        self.analysis_results = analysis_results
        self.df = df
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_json_report(self) -> str:
        """Generate comprehensive JSON report"""
        report = {
            'metadata': {
                'timestamp': self.timestamp,
                'dataset_shape': self.df.shape,
                'analysis_version': '1.0'
            },
            'analysis_results': self.analysis_results,
            'summary': self._generate_executive_summary()
        }
        return json.dumps(report, indent=2, default=str)

    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of key findings"""
        summary = {
            'key_findings': [],
            'data_quality_score': self._calculate_data_quality_score(),
            'statistical_significance_findings': self._extract_significant_findings(),
            'recommended_next_steps': self.analysis_results.get('recommendations', {})
        }

        # Extract key findings
        basic_info = self.analysis_results.get('basic_info', {})
        if basic_info.get('missing_values', {}).get('percentage', 0) > 10:
            summary['key_findings'].append(f"High missing data rate: {basic_info['missing_values']['percentage']:.1f}%")

        strong_corrs = self.analysis_results.get('correlation_analysis', {}).get('strong_correlations', [])
        if strong_corrs:
            summary['key_findings'].append(f"Found {len(strong_corrs)} strong correlations requiring investigation")

        # Outlier findings
        outlier_summary = self._summarize_outliers()
        if outlier_summary:
            summary['key_findings'].extend(outlier_summary)

        return summary

    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        basic_info = self.analysis_results.get('basic_info', {})

        # Penalize for missing values
        missing_pct = basic_info.get('missing_values', {}).get('percentage', 0)
        score -= missing_pct * 2

        # Penalize for duplicates
        duplicate_pct = basic_info.get('duplicates', {}).get('percentage', 0)
        score -= duplicate_pct * 3

        # Penalize for excessive outliers
        outlier_penalty = 0
        for col, outlier_data in self.analysis_results.get('outlier_analysis', {}).items():
            if col != 'multivariate_outliers':
                iqr_pct = outlier_data.get('iqr', {}).get('percentage', 0)
                if iqr_pct > 10:
                    outlier_penalty += 5
        score -= outlier_penalty

        return max(0.0, min(100.0, score))

    def _extract_significant_findings(self) -> List[Dict[str, Any]]:
        """Extract statistically significant findings"""
        findings = []

        # Significant correlations
        strong_corrs = self.analysis_results.get('correlation_analysis', {}).get('strong_correlations', [])
        for corr in strong_corrs:
            findings.append({
                'type': 'correlation',
                'description': f"Strong {corr['strength'].lower()} correlation between {corr['variable1']} and {corr['variable2']}",
                'strength': corr['correlation'],
                'variables': [corr['variable1'], corr['variable2']]
            })

        # Significant statistical tests
        for test_name, test_result in self.analysis_results.get('statistical_tests', {}).items():
            if test_result.get('p_value', 1) < 0.05:
                findings.append({
                    'type': 'statistical_test',
                    'test_name': test_name,
                    'description': f"Significant result in {test_result.get('test_type', test_name)}",
                    'p_value': test_result['p_value'],
                    'significant': True
                })

        return findings

    def _summarize_outliers(self) -> List[str]:
        """Summarize outlier findings"""
        summaries = []
        for col, outlier_data in self.analysis_results.get('outlier_analysis', {}).items():
            if col != 'multivariate_outliers':
                iqr_pct = outlier_data.get('iqr', {}).get('percentage', 0)
                z_pct = outlier_data.get('z_score', {}).get('percentage', 0)
                if iqr_pct > 5 or z_pct > 2:
                    summaries.append(f"Variable '{col}' has {iqr_pct:.1f}% outliers (IQR method)")
        return summaries


# Enhanced Gemini prompt creation function
def create_pattern_enhanced_gemini_prompt(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    """Create comprehensive prompt for Gemini including advanced pattern insights"""

    prompt = f"""You are a world-class data scientist with expertise in advanced pattern recognition and hypothesis generation. Based on the comprehensive statistical analysis and pattern detection below, provide:

1. EXECUTIVE SUMMARY (3-4 sentences highlighting the most significant patterns)
2. TOP 5 EVIDENCE-BASED HYPOTHESES ranked by statistical strength and novelty
3. METHODOLOGY ROADMAP for testing each hypothesis
4. CONFOUNDING VARIABLES and bias considerations
5. FOLLOW-UP RESEARCH DIRECTIONS

=== DATASET OVERVIEW ===
• Dimensions: {analysis['basic_info']['shape'][0]:,} rows × {analysis['basic_info']['shape'][1]} columns
• Data Quality Score: {ReportGenerator(analysis, df)._calculate_data_quality_score():.1f}/100
• Missing Data: {analysis['basic_info']['missing_values']['percentage']:.1f}%
• Variable Types: {len(analysis['basic_info']['column_types']['numeric_columns'])} numeric, {len(analysis['basic_info']['column_types']['categorical_columns'])} categorical

=== PHASE 1: STATISTICAL FOUNDATIONS ===
"""

    # Add key statistical findings
    strong_corrs = analysis.get('correlation_analysis', {}).get('strong_correlations', [])
    if strong_corrs:
        prompt += f"\n🔗 SIGNIFICANT CORRELATIONS ({len(strong_corrs)} found):\n"
        for corr in strong_corrs[:3]:
            prompt += f"• {corr['variable1']} ↔ {corr['variable2']}: r={corr['correlation']:.3f} ({corr['strength']})\n"

    # Add univariate insights
    univariate = analysis.get('univariate_analysis', {})
    if univariate.get('numeric'):
        prompt += f"\n📊 KEY VARIABLE CHARACTERISTICS:\n"
        for col, stats in list(univariate['numeric'].items())[:3]:
            skew = stats.get('skewness', 0)
            cv = stats.get('coefficient_of_variation', 0)
            prompt += f"• {col}: μ={stats.get('mean', 0):.2f}, σ={stats.get('std', 0):.2f}, skew={skew:.2f}, CV={cv:.2f}\n"

    # Phase 2 Pattern Detection Results
    patterns = analysis.get('advanced_patterns', {})
    if patterns:
        prompt += f"\n=== PHASE 2: ADVANCED PATTERN DETECTION ===\n"

        # Clustering patterns
        clustering = patterns.get('clustering_analysis', {})
        if clustering and 'message' not in clustering:
            kmeans_info = clustering.get('kmeans', {})
            if kmeans_info.get('optimal_clusters', 0) > 1:
                n_clusters = kmeans_info['optimal_clusters']
                silhouette = kmeans_info.get('final_silhouette_score', 0)
                prompt += f"\n🎯 CLUSTERING INSIGHTS:\n"
                prompt += f"• Optimal clusters detected: {n_clusters} (silhouette score: {silhouette:.3f})\n"
                prompt += f"• Cluster sizes: {kmeans_info.get('cluster_sizes', [])}\n"

                # DBSCAN insights
                dbscan_info = clustering.get('dbscan', {})
                if dbscan_info.get('n_clusters', 0) > 0:
                    prompt += f"• DBSCAN found {dbscan_info['n_clusters']} density-based clusters with {dbscan_info.get('noise_percentage', 0):.1f}% noise\n"

        # Feature interaction patterns
        interactions = patterns.get('feature_interactions', {})
        if interactions:
            strong_int = interactions.get('multiplicative_interactions', {}).get('strong_interactions', [])
            if strong_int:
                prompt += f"\n⚡ FEATURE INTERACTIONS:\n"
                for interaction in strong_int[:2]:
                    prompt += f"• {interaction['feature1']} × {interaction['feature2']}: strength={interaction['interaction_strength']:.3f}\n"

            # Polynomial interactions
            poly_int = interactions.get('polynomial_features', {})
            if poly_int.get('top_interactions'):
                prompt += f"• Top polynomial interactions: {poly_int['total_interactions']} detected\n"

        # Outlier ensemble results
        outliers = patterns.get('ensemble_outliers', {})
        if outliers and 'message' not in outliers:
            ensemble_results = outliers.get('ensemble_results', {})
            high_conf_outliers = len(ensemble_results.get('high_confidence_outliers', []))
            if high_conf_outliers > 0:
                prompt += f"\n⚠️ OUTLIER CONSENSUS:\n"
                prompt += f"• {high_conf_outliers} high-confidence outliers detected by multiple algorithms\n"

                # Individual method results
                methods = outliers.get('methods', {})
                for method, result in methods.items():
                    if 'outlier_percentage' in result:
                        prompt += f"• {method.replace('_', ' ').title()}: {result['outlier_percentage']:.1f}% outliers\n"

        # Association patterns
        associations = patterns.get('association_patterns', {})
        if associations and 'message' not in associations:
            pairwise = associations.get('pairwise', [])
            strong_assoc = [a for a in pairwise if a.get('cramers_v', 0) > 0.3]
            if strong_assoc:
                prompt += f"\n🔗 CATEGORICAL ASSOCIATIONS:\n"
                for assoc in strong_assoc[:2]:
                    prompt += f"• {assoc['variable1']} ↔ {assoc['variable2']}: Cramér's V={assoc['cramers_v']:.3f} ({assoc.get('association_strength', 'Strong')})\n"

        # Temporal patterns
        temporal = patterns.get('temporal_patterns', {})
        if temporal and 'message' not in temporal:
            prompt += f"\n📅 TEMPORAL PATTERNS:\n"
            for dt_col, info in temporal.items():
                date_range = info.get('date_range', {})
                if date_range:
                    prompt += f"• {dt_col}: {date_range.get('span_days', 0)} days range\n"

                trends = info.get('trends', {})
                for var, trend_info in trends.items():
                    direction = trend_info.get('trend_direction', 'stable')
                    strength = trend_info.get('trend_strength', 0)
                    prompt += f"• {var} shows {direction} trend over time (strength: {strength:.3f})\n"

        # Dimensionality insights
        dimensionality = patterns.get('dimensionality_insights', {})
        pca_info = dimensionality.get('pca', {})
        if pca_info:
            components_80 = pca_info.get('dimensionality_reduction_potential', {}).get('components_for_80_percent', 0)
            compression = pca_info.get('dimensionality_reduction_potential', {}).get('compression_ratio_80', 1)
            if compression < 0.8:
                prompt += f"\n🎯 DIMENSIONALITY INSIGHTS:\n"
                prompt += f"• {components_80} components capture 80% of variance (compression ratio: {compression:.2f})\n"

                # Component interpretation
                comp_interp = pca_info.get('component_interpretation', [])
                if comp_interp:
                    prompt += f"• PC1 dominated by: {', '.join([v['variable'] for v in comp_interp[0].get('top_contributing_variables', [])])}\n"

        # Data quality patterns
        quality_patterns = patterns.get('data_quality_patterns', {})
        if quality_patterns:
            consistency = quality_patterns.get('consistency_patterns', {})
            issues = consistency.get('total_issues_found', 0)
            if issues > 0:
                prompt += f"\n🚨 DATA QUALITY ALERTS:\n"
                prompt += f"• {issues} data consistency issues detected\n"

            # Missing value patterns
            missing_patterns = quality_patterns.get('missing_value_patterns', {})
            strong_missing_corr = missing_patterns.get('missing_correlations', [])
            if strong_missing_corr:
                prompt += f"• Correlated missingness patterns detected in {len(strong_missing_corr)} variable pairs\n"

    # Enhanced recommendations
    recommendations = analysis.get('recommendations', {})
    pattern_insights = recommendations.get('pattern_insights', [])
    if pattern_insights:
        prompt += f"\n💡 PATTERN-BASED INSIGHTS:\n"
        for insight in pattern_insights:
            prompt += f"• {insight}\n"

    prompt += f"""
=== HYPOTHESIS GENERATION FRAMEWORK ===
Using the comprehensive analysis above, generate exactly 5 hypotheses that leverage BOTH traditional statistical relationships AND advanced pattern insights. Prioritize hypotheses that:

✓ Are supported by multiple types of evidence (correlations + patterns + clusters)
✓ Suggest novel relationships not obvious from basic statistics
✓ Can be tested with the available data and appropriate statistical methods
✓ Consider the detected patterns (clusters, interactions, temporal trends)
✓ Account for data quality issues and potential confounders

For EACH hypothesis, provide:
1. **Hypothesis Statement** (specific, testable prediction)
2. **Statistical Evidence** (quantitative support from analysis)
3. **Pattern Evidence** (clustering, interactions, or temporal patterns supporting this)
4. **Recommended Statistical Test** (specific method with parameters)
5. **Expected Effect Size** (small/medium/large with justification)
6. **Confounding Variables** (potential threats to validity)
7. **Follow-up Questions** (what this hypothesis would reveal)

Format your response as:

### EXECUTIVE SUMMARY
[3-4 sentences highlighting the most compelling patterns and their implications]

### HYPOTHESIS 1: [Descriptive Title]
**Statement:** [Clear, testable hypothesis]
**Statistical Evidence:** [Correlations, tests, effect sizes]
**Pattern Evidence:** [Clusters, interactions, temporal trends]
**Recommended Test:** [Specific statistical method]
**Effect Size:** [Expected magnitude with reasoning]
**Confounders:** [Variables that could affect results]
**Follow-up:** [Research questions this would answer]

[Continue for hypotheses 2-5...]

### METHODOLOGY RECOMMENDATIONS
[Overall testing strategy considering all patterns detected]

### BIAS CONSIDERATIONS
[Potential sources of bias and mitigation strategies]
"""

    return prompt


def call_gemini_api(prompt: str, api_key: str) -> str:
    """Call the Gemini API with the given prompt."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()

        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            if 'content' in response_json['candidates'][0] and 'parts' in response_json['candidates'][0]['content']:
                parts = response_json['candidates'][0]['content']['parts']
                if parts and 'text' in parts[0]:
                    return parts[0]['text']

        return "Error: Unable to parse API response"
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


# Enhanced visualization functions
class AdvancedVisualizer:
    """Create comprehensive visualizations for statistical analysis"""

    def __init__(self, df: pd.DataFrame, analysis: Dict[str, Any]):
        self.df = df
        self.analysis = analysis
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def create_all_visualizations(self) -> Dict[str, Any]:
        """Create comprehensive set of visualizations"""
        visualizations = {}

        # Distribution plots with statistical annotations
        visualizations.update(self._create_distribution_plots())

        # Advanced correlation visualizations
        visualizations.update(self._create_correlation_plots())

        # Outlier visualizations
        visualizations.update(self._create_outlier_plots())

        # Bivariate relationship plots
        visualizations.update(self._create_bivariate_plots())

        # Statistical test results plots
        visualizations.update(self._create_statistical_plots())

        return visualizations

    def _create_distribution_plots(self) -> Dict[str, Any]:
        """Create enhanced distribution plots with statistical annotations"""
        plots = {}

        for col in self.numeric_cols[:6]:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(f'{col} - Histogram', f'{col} - Box Plot',
                                f'{col} - Q-Q Plot', f'{col} - Violin Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            data = self.df[col].dropna()

            # Histogram with KDE
            fig.add_trace(
                go.Histogram(x=data, name='Distribution', nbinsx=30, opacity=0.7),
                row=1, col=1
            )

            # Box plot
            fig.add_trace(
                go.Box(y=data, name='Box Plot', boxpoints='outliers'),
                row=1, col=2
            )

            # Q-Q plot data
            qq_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
            qq_sample = np.sort(stats.zscore(data))

            fig.add_trace(
                go.Scatter(x=qq_theoretical, y=qq_sample, mode='markers',
                           name='Q-Q Plot', marker=dict(size=3)),
                row=2, col=1
            )

            # Add reference line for Q-Q plot
            fig.add_trace(
                go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines',
                           name='Reference Line', line=dict(dash='dash')),
                row=2, col=1
            )

            # Violin plot
            fig.add_trace(
                go.Violin(y=data, name='Violin Plot', box_visible=True),
                row=2, col=2
            )

            # Add statistical annotations
            stats_text = self._get_distribution_stats_text(col)
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )

            fig.update_layout(
                title=f"Statistical Distribution Analysis: {col}",
                height=800,
                showlegend=False
            )

            plots[f"distribution_analysis_{col}"] = fig

        return plots

    def _create_correlation_plots(self) -> Dict[str, Any]:
        """Create advanced correlation visualizations"""
        plots = {}

        if len(self.numeric_cols) >= 2:
            # Enhanced correlation heatmap
            corr_matrix = self.df[self.numeric_cols].corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title="Enhanced Correlation Matrix",
                xaxis_title="Variables",
                yaxis_title="Variables",
                height=600
            )

            plots["enhanced_correlation_heatmap"] = fig

            # Correlation network (for strong correlations)
            strong_corrs = self.analysis.get('correlation_analysis', {}).get('strong_correlations', [])
            if strong_corrs:
                plots["correlation_network"] = self._create_correlation_network(strong_corrs)

        return plots

    def _create_outlier_plots(self) -> Dict[str, Any]:
        """Create comprehensive outlier visualizations"""
        plots = {}

        for col in self.numeric_cols[:4]:
            data = self.df[col].dropna()

            # Multi-method outlier detection plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('IQR Method', 'Z-Score Method',
                                'Modified Z-Score', 'Box Plot with Outliers'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # IQR method
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]

            fig.add_trace(
                go.Scatter(x=data.index, y=data, mode='markers',
                           name='Normal', marker=dict(color='blue', size=3)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=outliers_iqr.index, y=outliers_iqr, mode='markers',
                           name='IQR Outliers', marker=dict(color='red', size=5)),
                row=1, col=1
            )

            # Z-Score method
            z_scores = np.abs(stats.zscore(data))
            outliers_z = data[z_scores > 3]

            fig.add_trace(
                go.Scatter(x=data.index, y=z_scores, mode='markers',
                           name='Z-Scores', marker=dict(color='green', size=3)),
                row=1, col=2
            )
            fig.add_hline(y=3, line_dash="dash", line_color="red", row=1, col=2)

            # Box plot with outliers highlighted
            fig.add_trace(
                go.Box(y=data, name='Box Plot', boxpoints='all'),
                row=2, col=1
            )

            plots[f"outlier_analysis_{col}"] = fig

        return plots

    def _create_bivariate_plots(self) -> Dict[str, Any]:
        """Create bivariate relationship visualizations"""
        plots = {}

        # Scatter plot matrix for numeric variables (limited)
        if len(self.numeric_cols) >= 2:
            cols_subset = self.numeric_cols[:4]

            fig = make_subplots(
                rows=len(cols_subset), cols=len(cols_subset),
                subplot_titles=[f'{col1} vs {col2}' for col1 in cols_subset for col2 in cols_subset]
            )

            for i, col1 in enumerate(cols_subset):
                for j, col2 in enumerate(cols_subset):
                    if i != j:
                        fig.add_trace(
                            go.Scatter(x=self.df[col2], y=self.df[col1],
                                       mode='markers', name=f'{col1} vs {col2}',
                                       marker=dict(size=3, opacity=0.6)),
                            row=i + 1, col=j + 1
                        )
                    else:
                        fig.add_trace(
                            go.Histogram(x=self.df[col1], name=f'{col1} dist'),
                            row=i + 1, col=j + 1
                        )

            fig.update_layout(
                title="Bivariate Relationship Matrix",
                height=800,
                showlegend=False
            )

            plots["bivariate_matrix"] = fig

        return plots

    def _create_statistical_plots(self) -> Dict[str, Any]:
        """Create plots for statistical test results"""
        plots = {}

        # ANOVA results visualization
        anova_results = []
        for test_name, result in self.analysis.get('statistical_tests', {}).items():
            if 'anova' in test_name.lower():
                anova_results.append((test_name, result))

        if anova_results:
            fig = go.Figure()

            test_names = [result[0].replace('anova_', '').replace('_', ' ') for result in anova_results]
            p_values = [result[1].get('p_value', 1) for result in anova_results]

            colors = ['red' if p < 0.05 else 'blue' for p in p_values]

            fig.add_trace(go.Bar(
                x=test_names,
                y=[-np.log10(p) for p in p_values],
                marker_color=colors,
                name='Statistical Tests'
            ))

            fig.add_hline(y=-np.log10(0.05), line_dash="dash",
                          annotation_text="Significance Threshold (p=0.05)")

            fig.update_layout(
                title="Statistical Test Results (-log10 p-values)",
                xaxis_title="Tests",
                yaxis_title="-log10(p-value)",
                height=500
            )

            plots["statistical_test_results"] = fig

        return plots

    def _get_distribution_stats_text(self, col: str) -> str:
        """Get formatted statistical text for distribution plots"""
        univariate = self.analysis.get('univariate_analysis', {}).get('numeric', {}).get(col, {})
        distribution = self.analysis.get('distribution_analysis', {}).get(col, {})

        text = f"<b>{col} Statistics:</b><br>"
        text += f"Mean: {univariate.get('mean', 0):.2f}<br>"
        text += f"Std: {univariate.get('std', 0):.2f}<br>"
        text += f"Skewness: {univariate.get('skewness', 0):.2f}<br>"
        text += f"Kurtosis: {univariate.get('kurtosis', 0):.2f}<br>"

        # Add normality test result if available
        normality = distribution.get('distribution_summary', {})
        if normality.get('likely_normal') is not None:
            text += f"Likely Normal: {normality.get('likely_normal', 'Unknown')}<br>"

        return text

    def _create_correlation_network(self, strong_corrs: List[Dict]) -> go.Figure:
        """Create network visualization of strong correlations"""
        fig = go.Figure()

        # Get unique variables
        variables = list(set([corr['variable1'] for corr in strong_corrs] +
                             [corr['variable2'] for corr in strong_corrs]))

        # Simple circular layout
        n = len(variables)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            text=variables,
            textposition="middle center",
            marker=dict(size=20, color='lightblue'),
            name='Variables'
        ))

        # Add edges for correlations
        for corr in strong_corrs:
            var1_idx = variables.index(corr['variable1'])
            var2_idx = variables.index(corr['variable2'])

            fig.add_trace(go.Scatter(
                x=[x_pos[var1_idx], x_pos[var2_idx]],
                y=[y_pos[var1_idx], y_pos[var2_idx]],
                mode='lines',
                line=dict(width=abs(corr['correlation']) * 5,
                          color='red' if corr['correlation'] < 0 else 'green'),
                name=f"r={corr['correlation']:.2f}"
            ))

        fig.update_layout(
            title="Strong Correlation Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )

        return fig


# Save/Export Functions
def save_analysis_results(analysis_results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, str]:
    """Save analysis results to files and return file paths"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory
    results_dir = f"analysis_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    file_paths = {}

    # Save JSON report
    report_generator = ReportGenerator(analysis_results, df)
    json_report = report_generator.generate_json_report()

    json_path = os.path.join(results_dir, "statistical_analysis_report.json")
    with open(json_path, 'w') as f:
        f.write(json_report)
    file_paths['json_report'] = json_path

    # Save visualizations as HTML (interactive) and PNG (static)
    visualizer = AdvancedVisualizer(df, analysis_results)
    visualizations = visualizer.create_all_visualizations()

    for viz_name, fig in visualizations.items():
        if hasattr(fig, 'write_html'):
            html_path = os.path.join(results_dir, f"{viz_name}.html")
            fig.write_html(html_path)
            file_paths[f'{viz_name}_html'] = html_path

            # Save as PNG (requires kaleido: pip install kaleido)
            try:
                png_path = os.path.join(results_dir, f"{viz_name}.png")
                fig.write_image(png_path, width=1200, height=800)
                file_paths[f'{viz_name}_png'] = png_path
            except Exception as e:
                st.warning(f"Could not save {viz_name} as PNG: {e}")

    return file_paths


# Launch Deep Research App Function
def launch_deep_research_repo():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    other_repo_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "local-deep-research"))
    app_path = os.path.join(other_repo_dir, "app.py")

    if os.path.exists(app_path):
        try:
            subprocess.Popen(f'cd "{other_repo_dir}" && python app.py', shell=True)
            return True, f"Successfully launched app at {app_path}"
        except Exception as e:
            return False, f"Error launching app: {str(e)}"
    else:
        return False, f"Could not find app at: {app_path}"


# Main Application
def main():
    # Create API key input field
    api_key = st.text_input("Enter your Gemini API Key", type="password")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and api_key:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📋 Data Preview",
                "📊 Deep Statistical Analysis",
                "🧪 AI Hypothesis Generation",
                "🔬 Method Selection",
                "📈 Advanced Visualizations",
                "💾 Export Results"
            ])

            with tab1:
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

                # Show column information
                st.subheader("Column Information")
                column_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Unique Values': [df[col].nunique() for col in df.columns],
                    'Missing %': [f"{(df[col].isnull().sum() / len(df) * 100):.1f}%" for col in df.columns]
                })
                st.dataframe(column_info, use_container_width=True)

            with tab2:
                st.subheader("🔬 Deep Statistical Analysis")

                if st.button("🚀 Run Comprehensive Analysis", type="primary"):
                    with st.spinner("Performing deep statistical analysis..."):
                        # Initialize analyzer
                        analyzer = DeepStatisticalAnalyzer(df)

                        # Run comprehensive analysis
                        analysis_results = analyzer.comprehensive_analysis()

                        # Store in session state
                        st.session_state.analysis_results = analysis_results

                        st.success("✅ Analysis complete!")

                # Display results if available
                if 'analysis_results' in st.session_state:
                    analysis = st.session_state.analysis_results

                    # Executive Summary
                    st.subheader("📋 Executive Summary")
                    summary = analysis.get('summary', {})

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        score = ReportGenerator(analysis, df)._calculate_data_quality_score()
                        st.metric("Data Quality Score", f"{score:.1f}/100")
                    with col2:
                        strong_corrs = len(analysis.get('correlation_analysis', {}).get('strong_correlations', []))
                        st.metric("Strong Correlations", strong_corrs)
                    with col3:
                        sig_tests = len([t for t in analysis.get('statistical_tests', {}).values()
                                         if t.get('p_value', 1) < 0.05])
                        st.metric("Significant Tests", sig_tests)

                    # Key Findings
                    findings = ReportGenerator(analysis, df)._extract_significant_findings()
                    if findings:
                        st.subheader("🔍 Key Statistical Findings")
                        for finding in findings[:5]:
                            if finding['type'] == 'correlation':
                                st.info(f"🔗 {finding['description']}")
                            elif finding['type'] == 'statistical_test':
                                st.success(f"📊 {finding['description']} (p={finding['p_value']:.4f})")

                    # Detailed Analysis Sections
                    with st.expander("📈 Univariate Analysis"):
                        # Numeric variables
                        if analysis.get('univariate_analysis', {}).get('numeric'):
                            st.write("**Numeric Variables:**")
                            numeric_df = pd.DataFrame.from_dict(
                                analysis['univariate_analysis']['numeric'],
                                orient='index'
                            ).round(3)
                            st.dataframe(numeric_df, use_container_width=True)

                        # Categorical variables
                        if analysis.get('univariate_analysis', {}).get('categorical'):
                            st.write("**Categorical Variables:**")
                            for col, stats in analysis['univariate_analysis']['categorical'].items():
                                st.write(f"**{col}:** {stats['unique_values']} unique values, "
                                         f"entropy: {stats['entropy']:.2f}")

                    with st.expander("🔗 Bivariate Analysis"):
                        # Numeric-Numeric correlations
                        num_num = analysis.get('bivariate_analysis', {}).get('numeric_numeric', {})
                        if num_num:
                            st.write("**Numeric-Numeric Relationships:**")
                            corr_df = pd.DataFrame.from_dict(num_num, orient='index')
                            if not corr_df.empty:
                                st.dataframe(corr_df.round(4), use_container_width=True)

                        # Numeric-Categorical relationships
                        num_cat = analysis.get('bivariate_analysis', {}).get('numeric_categorical', {})
                        if num_cat:
                            st.write("**Numeric-Categorical Relationships:**")
                            numcat_df = pd.DataFrame.from_dict(num_cat, orient='index')
                            if not numcat_df.empty:
                                st.dataframe(numcat_df.round(4), use_container_width=True)

                    with st.expander("📊 Distribution Analysis"):
                        dist_analysis = analysis.get('distribution_analysis', {})
                        if dist_analysis:
                            for col, dist_info in dist_analysis.items():
                                st.write(f"**{col}:**")
                                summary = dist_info.get('distribution_summary', {})
                                st.write(f"- Likely Normal: {summary.get('likely_normal', 'Unknown')}")
                                st.write(f"- Skewness: {summary.get('skewness_interpretation', 'Unknown')}")
                                st.write(f"- Kurtosis: {summary.get('kurtosis_interpretation', 'Unknown')}")

                    with st.expander("⚠️ Outlier Analysis"):
                        outlier_analysis = analysis.get('outlier_analysis', {})
                        if outlier_analysis:
                            for col, outlier_info in outlier_analysis.items():
                                if col != 'multivariate_outliers':
                                    st.write(f"**{col}:**")
                                    for method, stats in outlier_info.items():
                                        st.write(f"- {method.upper()}: {stats['count']} outliers "
                                                 f"({stats['percentage']:.1f}%)")

                    with st.expander("🎯 Multivariate Analysis"):
                        multi_analysis = analysis.get('multivariate_analysis', {})
                        if multi_analysis:
                            pca_info = multi_analysis.get('pca', {})
                            if pca_info:
                                st.write("**PCA Results:**")
                                st.write(
                                    f"- Components for 95% variance: {pca_info.get('n_components_95_variance', 'N/A')}")
                                st.write(f"- Total components: {pca_info.get('total_components', 'N/A')}")

                                # Show explained variance
                                exp_var = pca_info.get('explained_variance_ratio', [])
                                if exp_var:
                                    exp_var_df = pd.DataFrame({
                                        'Component': [f'PC{i + 1}' for i in range(len(exp_var))],
                                        'Explained Variance': exp_var,
                                        'Cumulative Variance': pca_info.get('cumulative_variance_ratio', exp_var)
                                    })
                                    st.dataframe(exp_var_df.head(10), use_container_width=True)

                    # Advanced Patterns Section
                    patterns = analysis.get('advanced_patterns', {})
                    if patterns:
                        with st.expander("🧬 Advanced Pattern Detection"):
                            # Clustering insights
                            clustering = patterns.get('clustering_analysis', {})
                            if clustering and 'message' not in clustering:
                                st.write("**Clustering Analysis:**")
                                kmeans_info = clustering.get('kmeans', {})
                                if kmeans_info:
                                    st.write(f"- Optimal clusters: {kmeans_info.get('optimal_clusters', 'N/A')}")
                                    st.write(f"- Silhouette score: {kmeans_info.get('final_silhouette_score', 0):.3f}")

                                dbscan_info = clustering.get('dbscan', {})
                                if dbscan_info and dbscan_info.get('n_clusters', 0) > 0:
                                    st.write(f"- DBSCAN clusters: {dbscan_info['n_clusters']}")
                                    st.write(f"- Noise percentage: {dbscan_info.get('noise_percentage', 0):.1f}%")

                            # Feature interactions
                            interactions = patterns.get('feature_interactions', {})
                            if interactions:
                                st.write("**Feature Interactions:**")
                                strong_int = interactions.get('multiplicative_interactions', {}).get(
                                    'strong_interactions', [])
                                if strong_int:
                                    for interaction in strong_int[:3]:
                                        st.write(f"- {interaction['feature1']} × {interaction['feature2']}: "
                                                 f"strength={interaction['interaction_strength']:.3f}")

                                poly_int = interactions.get('polynomial_features', {})
                                if poly_int:
                                    st.write(
                                        f"- Total polynomial interactions: {poly_int.get('total_interactions', 0)}")

                            # Outlier ensemble
                            outliers = patterns.get('ensemble_outliers', {})
                            if outliers and 'message' not in outliers:
                                st.write("**Ensemble Outlier Detection:**")
                                ensemble_results = outliers.get('ensemble_results', {})
                                high_conf = len(ensemble_results.get('high_confidence_outliers', []))
                                if high_conf > 0:
                                    st.write(f"- High-confidence outliers: {high_conf}")

                                methods = outliers.get('methods', {})
                                for method, result in methods.items():
                                    if 'outlier_percentage' in result:
                                        st.write(f"- {method.replace('_', ' ').title()}: "
                                                 f"{result['outlier_percentage']:.1f}% outliers")

                    with st.expander("💡 Recommendations"):
                        recommendations = analysis.get('recommendations', {})
                        for category, recs in recommendations.items():
                            if recs:
                                st.write(f"**{category.replace('_', ' ').title()}:**")
                                for rec in recs:
                                    st.write(f"- {rec}")

            def parse_hypotheses_to_json(gemini_response: str) -> dict:
                """
                Parse the AI-generated hypotheses text into a structured JSON format
                """
                try:
                    lines = gemini_response.split('\n')
                    structured_data = {
                        "executive_summary": "",
                        "hypotheses": [],
                        "methodology_recommendations": "",
                        "bias_considerations": "",
                        "follow_up_research_directions": ""
                    }

                    current_hypothesis = {}
                    current_section = None
                    current_key = None
                    buffer_text = ""

                    for line in lines:
                        line = line.strip()

                        # Skip empty lines
                        if not line:
                            continue

                        # Check for main sections
                        if line.startswith("### EXECUTIVE SUMMARY"):
                            current_section = "executive_summary"
                            buffer_text = ""
                            continue
                        elif line.startswith("### HYPOTHESIS"):
                            # Save previous hypothesis if exists
                            if current_hypothesis:
                                structured_data["hypotheses"].append(current_hypothesis)

                            # Extract hypothesis number and title
                            title_match = line.replace("### HYPOTHESIS", "").strip()
                            current_hypothesis = {
                                "id": len(structured_data["hypotheses"]) + 1,
                                "title": title_match,
                                "statement": "",
                                "statistical_evidence": "",
                                "pattern_evidence": "",
                                "recommended_test": "",
                                "effect_size": "",
                                "confounders": "",
                                "follow_up": ""
                            }
                            current_section = "hypothesis"
                            current_key = None
                            continue
                        elif line.startswith("### METHODOLOGY RECOMMENDATIONS"):
                            # Save last hypothesis
                            if current_hypothesis:
                                structured_data["hypotheses"].append(current_hypothesis)
                                current_hypothesis = {}
                            current_section = "methodology"
                            buffer_text = ""
                            continue
                        elif line.startswith("### BIAS CONSIDERATIONS"):
                            current_section = "bias"
                            buffer_text = ""
                            continue
                        elif line.startswith("### FOLLOW-UP RESEARCH DIRECTIONS"):
                            current_section = "follow_up_directions"
                            buffer_text = ""
                            continue

                        # Parse hypothesis components
                        if current_section == "hypothesis" and current_hypothesis:
                            if line.startswith("**Statement:**"):
                                current_key = "statement"
                                current_hypothesis[current_key] = line.replace("**Statement:**", "").strip()
                            elif line.startswith("**Statistical Evidence:**"):
                                current_key = "statistical_evidence"
                                current_hypothesis[current_key] = line.replace("**Statistical Evidence:**", "").strip()
                            elif line.startswith("**Pattern Evidence:**"):
                                current_key = "pattern_evidence"
                                current_hypothesis[current_key] = line.replace("**Pattern Evidence:**", "").strip()
                            elif line.startswith("**Recommended Test:**"):
                                current_key = "recommended_test"
                                current_hypothesis[current_key] = line.replace("**Recommended Test:**", "").strip()
                            elif line.startswith("**Effect Size:**"):
                                current_key = "effect_size"
                                current_hypothesis[current_key] = line.replace("**Effect Size:**", "").strip()
                            elif line.startswith("**Confounders:**"):
                                current_key = "confounders"
                                current_hypothesis[current_key] = line.replace("**Confounders:**", "").strip()
                            elif line.startswith("**Follow-up:**"):
                                current_key = "follow_up"
                                current_hypothesis[current_key] = line.replace("**Follow-up:**", "").strip()
                            elif current_key and not line.startswith("**") and not line.startswith("###"):
                                # Continue previous field
                                current_hypothesis[current_key] += " " + line
                            elif not current_key and not line.startswith("**"):
                                # This might be part of the title or description
                                current_hypothesis["title"] += " " + line

                        # Parse other sections - accumulate in buffer
                        elif current_section in ["executive_summary", "methodology", "bias", "follow_up_directions"]:
                            if not line.startswith("###"):
                                buffer_text += line + " "

                    # Don't forget the last hypothesis
                    if current_hypothesis:
                        structured_data["hypotheses"].append(current_hypothesis)

                    # Set accumulated text for sections
                    if current_section == "executive_summary":
                        structured_data["executive_summary"] = buffer_text.strip()
                    elif current_section == "methodology":
                        structured_data["methodology_recommendations"] = buffer_text.strip()
                    elif current_section == "bias":
                        structured_data["bias_considerations"] = buffer_text.strip()
                    elif current_section == "follow_up_directions":
                        structured_data["follow_up_research_directions"] = buffer_text.strip()

                    # Clean up hypothesis fields
                    for hypothesis in structured_data["hypotheses"]:
                        for key in hypothesis:
                            if isinstance(hypothesis[key], str):
                                hypothesis[key] = hypothesis[key].strip()

                    # If no structured hypotheses were found, try a simpler approach
                    if not structured_data["hypotheses"]:
                        # Look for any hypothesis patterns
                        hypothesis_sections = []
                        current_hyp_text = ""

                        for line in gemini_response.split('\n'):
                            if "HYPOTHESIS" in line and (
                                    "1:" in line or "2:" in line or "3:" in line or "4:" in line or "5:" in line):
                                if current_hyp_text:
                                    hypothesis_sections.append(current_hyp_text.strip())
                                current_hyp_text = line + "\n"
                            elif current_hyp_text and not line.startswith("###"):
                                current_hyp_text += line + "\n"
                            elif line.startswith("### METHODOLOGY") or line.startswith("### BIAS"):
                                if current_hyp_text:
                                    hypothesis_sections.append(current_hyp_text.strip())
                                    current_hyp_text = ""
                                break

                        if current_hyp_text:
                            hypothesis_sections.append(current_hyp_text.strip())

                        # Parse each hypothesis section
                        for i, hyp_text in enumerate(hypothesis_sections):
                            structured_data["hypotheses"].append({
                                "id": i + 1,
                                "title": f"Hypothesis {i + 1}",
                                "full_text": hyp_text,
                                "statement": "",
                                "statistical_evidence": "",
                                "pattern_evidence": "",
                                "recommended_test": "",
                                "effect_size": "",
                                "confounders": "",
                                "follow_up": ""
                            })

                    return structured_data

                except Exception as e:
                    # Fallback: return basic structure with raw text
                    return {
                        "executive_summary": "Error parsing response - see raw text below",
                        "hypotheses": [{
                            "id": 1,
                            "title": "Raw AI Response",
                            "full_text": gemini_response,
                            "statement": "Unable to parse structured format",
                            "statistical_evidence": "",
                            "pattern_evidence": "",
                            "recommended_test": "",
                            "effect_size": "",
                            "confounders": "",
                            "follow_up": ""
                        }],
                        "methodology_recommendations": "",
                        "bias_considerations": "",
                        "follow_up_research_directions": "",
                        "parsing_error": str(e)
                    }

            with tab3:
                st.subheader("🧪 AI-Powered Hypothesis Generation")

                if 'analysis_results' not in st.session_state:
                    st.warning("⚠️ Please run the statistical analysis first in the 'Deep Statistical Analysis' tab.")
                else:
                    if st.button("🤖 Generate Enhanced Hypotheses", type="primary"):
                        with st.spinner("🧠 AI is analyzing patterns and generating hypotheses..."):
                            # Create enhanced prompt
                            enhanced_prompt = create_pattern_enhanced_gemini_prompt(df,
                                                                                    st.session_state.analysis_results)

                            # Show prompt preview
                            with st.expander("🔍 View Enhanced Prompt (Preview)"):
                                st.text(
                                    enhanced_prompt[:2000] + "..." if len(enhanced_prompt) > 2000 else enhanced_prompt)

                            # Call Gemini API
                            gemini_response = call_gemini_api(enhanced_prompt, api_key)

                            # Store response
                            st.session_state.gemini_response = gemini_response

                            # Display response
                            st.markdown("### 🎯 AI-Generated Hypotheses")
                            st.markdown(gemini_response)

                            st.success("✅ Hypotheses generated successfully!")

                    # Show previous response if available
                    if 'gemini_response' in st.session_state:
                        st.markdown("### 🎯 Generated Hypotheses")
                        st.markdown(st.session_state.gemini_response)

                        # Add save functionality section
                        st.markdown("---")
                        st.subheader("💾 Save Analysis Results")

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            # Option to choose what to save
                            save_options = st.multiselect(
                                "Select components to save:",
                                [
                                    "Statistical Analysis Results",
                                    "AI-Generated Hypotheses",
                                    "Dataset Metadata",
                                    "Executive Summary",
                                    "Data Quality Report"
                                ],
                                default=["Statistical Analysis Results", "AI-Generated Hypotheses"]
                            )

                            # Custom filename option
                            custom_filename = st.text_input(
                                "Custom filename (optional):",
                                placeholder="my_analysis_results"
                            )

                        with col2:
                            st.markdown("")  # spacing
                            st.markdown("")  # spacing

                            if st.button("📥 Save to JSON", type="secondary"):
                                try:
                                    # Generate the comprehensive report
                                    report_generator = ReportGenerator(st.session_state.analysis_results, df)

                                    # Create comprehensive findings dictionary
                                    findings_data = {}

                                    if "Dataset Metadata" in save_options:
                                        findings_data["dataset_metadata"] = {
                                            "shape": df.shape,
                                            "columns": df.columns.tolist(),
                                            "dtypes": df.dtypes.astype(str).to_dict(),
                                            "memory_usage": df.memory_usage(deep=True).sum(),
                                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        }

                                    if "Statistical Analysis Results" in save_options:
                                        findings_data["statistical_analysis"] = st.session_state.analysis_results

                                    if "AI-Generated Hypotheses" in save_options:
                                        findings_data["ai_hypotheses"] = {
                                            "response": st.session_state.gemini_response,
                                            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "model": "gemini-2.5-flash-preview"
                                        }

                                    if "Executive Summary" in save_options:
                                        findings_data[
                                            "executive_summary"] = report_generator._generate_executive_summary()

                                    if "Data Quality Report" in save_options:
                                        findings_data["data_quality"] = {
                                            "overall_score": report_generator._calculate_data_quality_score(),
                                            "significant_findings": report_generator._extract_significant_findings(),
                                            "outlier_summary": report_generator._summarize_outliers()
                                        }

                                    # Generate filename
                                    if custom_filename:
                                        filename = f"{custom_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                    else:
                                        filename = f"analysis_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                                    # Convert to JSON string
                                    json_string = json.dumps(findings_data, indent=2, default=str)

                                    # Create download button
                                    st.download_button(
                                        label="⬇️ Download JSON Report",
                                        data=json_string,
                                        file_name=filename,
                                        mime="application/json",
                                        help="Click to download your complete analysis findings as JSON"
                                    )

                                    # Success message with file details
                                    st.success(f"✅ Report prepared successfully!")
                                    st.info(
                                        f"📊 File size: {len(json_string.encode('utf-8'))} bytes\n📁 Components: {len(save_options)} selected")

                                except Exception as e:
                                    st.error(f"❌ Error generating report: {str(e)}")

                        # Preview section
                        with st.expander("👁️ Preview JSON Structure"):
                            if save_options:
                                try:
                                    # Generate a preview of the JSON structure
                                    preview_data = {}

                                    if "Dataset Metadata" in save_options:
                                        preview_data[
                                            "dataset_metadata"] = "Dataset information (shape, columns, types, etc.)"

                                    if "Statistical Analysis Results" in save_options:
                                        preview_data["statistical_analysis"] = "Complete statistical analysis results"

                                    if "AI-Generated Hypotheses" in save_options:
                                        preview_data["ai_hypotheses"] = "AI-generated hypotheses and insights"

                                    if "Executive Summary" in save_options:
                                        preview_data["executive_summary"] = "Key findings and recommendations"

                                    if "Data Quality Report" in save_options:
                                        preview_data["data_quality"] = "Data quality metrics and assessments"

                                    st.json(preview_data)

                                except Exception as e:
                                    st.error(f"Error generating preview: {str(e)}")
                            else:
                                st.info("Select components above to see preview")

                        # Additional export options
                        st.markdown("---")
                        st.subheader("📄 Additional Export Options")

                        col3, col4 = st.columns(2)

                        with col3:
                            # Save just the hypotheses as markdown
                            if st.button("📝 Save Hypotheses as Markdown"):
                                markdown_content = f"""# AI-Generated Hypotheses Report
            Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            ## Dataset Overview
            - Shape: {df.shape[0]} rows × {df.shape[1]} columns
            - Analysis Version: 1.0

            ## AI-Generated Hypotheses
            {st.session_state.gemini_response}

            ---
            *Generated using Advanced Statistical Analysis Pipeline*
            """

                                st.download_button(
                                    label="⬇️ Download Markdown",
                                    data=markdown_content,
                                    file_name=f"hypotheses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )

                        with col4:
                            # Save structured hypotheses
                            if st.button("🔬 Save Structured Hypotheses"):
                                try:
                                    structured_hypotheses = parse_hypotheses_to_json(st.session_state.gemini_response)

                                    hypotheses_json = json.dumps({
                                        "structured_hypotheses": structured_hypotheses,
                                        "metadata": {
                                            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "dataset_shape": df.shape,
                                            "total_hypotheses": len(structured_hypotheses.get("hypotheses", [])),
                                            "model_used": "gemini-2.5-flash-preview"
                                        }
                                    }, indent=2, default=str)

                                    st.download_button(
                                        label="⬇️ Download Structured Hypotheses",
                                        data=hypotheses_json,
                                        file_name=f"structured_hypotheses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json"
                                    )

                                except Exception as e:
                                    st.error(f"Error generating structured hypotheses: {str(e)}")
            with tab4:
                st.subheader("🔬 Statistical Methods Selection")

                if 'gemini_response' not in st.session_state:
                    st.info("💡 Generate hypotheses first to get method recommendations.")
                else:
                    st.markdown("### 🛠️ Recommended Statistical Methods")

                    # Extract and display recommended methods from Gemini response
                    response_text = st.session_state.gemini_response

                    # Simple method extraction
                    if "Test:" in response_text or "test" in response_text.lower():
                        st.info("📋 Based on your hypotheses, here are statistical methods to consider:")

                        # Common statistical methods based on data types
                        st.subheader("🔢 For Numeric Variables:")
                        st.write("- **t-tests**: Compare means between groups")
                        st.write("- **ANOVA**: Compare means across multiple groups")
                        st.write("- **Regression Analysis**: Model relationships between variables")
                        st.write("- **Correlation Analysis**: Measure linear relationships")
                        st.write("- **Mann-Whitney U**: Non-parametric alternative to t-test")

                        st.subheader("📊 For Categorical Variables:")
                        st.write("- **Chi-square Tests**: Test independence between categorical variables")
                        st.write("- **Fisher's Exact Test**: For small sample sizes")
                        st.write("- **McNemar's Test**: For paired categorical data")

                        st.subheader("🔄 For Mixed Data Types:")
                        st.write("- **Point-biserial Correlation**: Numeric vs binary categorical")
                        st.write("- **Logistic Regression**: Predict categorical outcomes")
                        st.write("- **ANCOVA**: Compare groups while controlling for covariates")

                        # Method selection interface
                        st.subheader("✅ Select Methods for Analysis:")

                        method_categories = {
                            "Comparison Tests": ["t-test", "ANOVA", "Mann-Whitney U", "Kruskal-Wallis"],
                            "Association Tests": ["Pearson Correlation", "Spearman Correlation", "Chi-square"],
                            "Regression Methods": ["Linear Regression", "Logistic Regression", "Multiple Regression"],
                            "Non-parametric Tests": ["Wilcoxon", "Mann-Whitney U", "Kruskal-Wallis"]
                        }

                        selected_methods = {}
                        for category, methods in method_categories.items():
                            st.write(f"**{category}:**")
                            selected_methods[category] = st.multiselect(
                                f"Select {category.lower()}:",
                                methods,
                                key=f"methods_{category}"
                            )

                        if st.button("💾 Save Method Selection"):
                            st.session_state.selected_methods = selected_methods
                            st.success("✅ Methods saved! Ready for analysis.")

            with tab5:
                st.subheader("📈 Advanced Visualizations")

                if 'analysis_results' not in st.session_state:
                    st.warning("⚠️ Please run the statistical analysis first.")
                else:
                    if st.button("🎨 Generate All Visualizations", type="primary"):
                        with st.spinner("🎨 Creating advanced visualizations..."):
                            # Create visualizations
                            visualizer = AdvancedVisualizer(df, st.session_state.analysis_results)
                            visualizations = visualizer.create_all_visualizations()

                            # Store in session state
                            st.session_state.visualizations = visualizations

                            st.success("✅ Visualizations created!")

                    # Display visualizations if available
                    if 'visualizations' in st.session_state:
                        viz_dict = st.session_state.visualizations

                        # Distribution Analysis Plots
                        st.subheader("📊 Distribution Analysis")
                        dist_plots = {k: v for k, v in viz_dict.items() if 'distribution_analysis' in k}
                        for name, fig in dist_plots.items():
                            st.plotly_chart(fig, use_container_width=True)

                        # Correlation Plots
                        st.subheader("🔗 Correlation Analysis")
                        if 'enhanced_correlation_heatmap' in viz_dict:
                            st.plotly_chart(viz_dict['enhanced_correlation_heatmap'], use_container_width=True)
                        if 'correlation_network' in viz_dict:
                            st.plotly_chart(viz_dict['correlation_network'], use_container_width=True)

                        # Outlier Plots
                        st.subheader("⚠️ Outlier Analysis")
                        outlier_plots = {k: v for k, v in viz_dict.items() if 'outlier_analysis' in k}
                        for name, fig in outlier_plots.items():
                            st.plotly_chart(fig, use_container_width=True)

                        # Bivariate Plots
                        st.subheader("📈 Bivariate Relationships")
                        if 'bivariate_matrix' in viz_dict:
                            st.plotly_chart(viz_dict['bivariate_matrix'], use_container_width=True)

                        # Statistical Test Results
                        if 'statistical_test_results' in viz_dict:
                            st.subheader("🧪 Statistical Test Results")
                            st.plotly_chart(viz_dict['statistical_test_results'], use_container_width=True)

            with tab6:
                st.subheader("💾 Export Analysis Results")

                if 'analysis_results' not in st.session_state:
                    st.warning("⚠️ Please run the analysis first.")
                else:
                    st.write("📁 Export your comprehensive analysis results:")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("📊 Export JSON Report", type="primary"):
                            with st.spinner("📝 Generating JSON report..."):
                                report_gen = ReportGenerator(st.session_state.analysis_results, df)
                                json_report = report_gen.generate_json_report()

                                # Download button
                                st.download_button(
                                    label="⬇️ Download JSON Report",
                                    data=json_report,
                                    file_name=f"statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                                st.success("✅ JSON report ready for download!")

                    with col2:
                        if st.button("🎨 Export Visualizations"):
                            if 'visualizations' in st.session_state:
                                st.info("🔧 Visualization export functionality ready. "
                                        "Individual plots can be downloaded using the Plotly toolbar.")
                            else:
                                st.warning("⚠️ Generate visualizations first in the previous tab.")

                    # Full export option
                    st.subheader("📦 Complete Analysis Package")
                    if st.button("🚀 Generate Complete Analysis Package"):
                        if 'analysis_results' in st.session_state:
                            with st.spinner("📦 Creating complete analysis package..."):
                                try:
                                    file_paths = save_analysis_results(st.session_state.analysis_results, df)

                                    st.success("✅ Complete analysis package created!")
                                    st.json(file_paths)
                                    st.info("💡 Files have been saved to your local directory. "
                                            "Check the generated folder for all exports.")
                                except Exception as e:
                                    st.error(f"❌ Error creating package: {str(e)}")
                        else:
                            st.warning("⚠️ No analysis results to export.")

        except Exception as e:
            st.error(f"❌ Error processing the CSV file: {str(e)}")
            st.exception(e)

    elif uploaded_file is not None and not api_key:
        st.warning("🔑 Please enter your Gemini API key to analyze the file.")

    else:
        # Show instructions when no file is uploaded
        st.info("📤 Please upload a CSV file and provide your Gemini API key to get started.")

        # Enhanced feature description
        st.subheader("🌟 What this Enhanced App Does:")

        features = [
            "🔬 **Deep Statistical Analysis**: Comprehensive univariate, bivariate, and multivariate analysis",
            "📊 **Advanced Visualizations**: Interactive plots with statistical annotations",
            "🤖 **AI-Powered Hypotheses**: Enhanced prompts for more sophisticated hypothesis generation",
            "📈 **Outlier Detection**: Multiple methods including IQR, Z-score, and Isolation Forest",
            "🔗 **Correlation Analysis**: Pearson, Spearman, Kendall, and mutual information",
            "🧪 **Statistical Tests**: Normality tests, ANOVA, Chi-square, and more",
            "💾 **Comprehensive Export**: JSON reports and interactive visualizations",
            "🎯 **Method Recommendations**: Statistical test suggestions based on data characteristics",
            "🧬 **Pattern Detection**: Advanced clustering, feature interactions, and temporal analysis"
        ]

        for feature in features:
            st.write(feature)

        # Privacy notice
        st.subheader("🔒 Privacy & Security:")
        st.write("""
        - 🏠 Your data is processed locally and is not stored permanently
        - 📊 Only statistical summaries (not raw data) are sent to the Gemini API
        - 🔑 Your API key is not stored and is only used for the current session
        - 💾 All exported files remain on your local machine
        """)


# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ Additional Tools")

    if st.button("🚀 Launch Deep Research App"):
        success, message = launch_deep_research_repo()
        if success:
            st.success(message)
        else:
            st.error(message)

    st.markdown("---")
    st.markdown("## 📋 Analysis Checklist")
    st.markdown("""
    ✅ Upload CSV file  
    ✅ Enter Gemini API key  
    ✅ Run statistical analysis  
    ✅ Generate AI hypotheses  
    ✅ Create visualizations  
    ✅ Export results  
    """)

    st.markdown("---")
    st.markdown("## 💡 Tips")
    st.markdown("""
    - **Large datasets**: Analysis may take longer for files >50MB
    - **API costs**: Monitor your Gemini API usage
    - **Export**: Save results before closing the app
    - **Visualization**: Use Plotly toolbar to interact with charts
    - **Pattern Detection**: Advanced patterns require at least 2 numeric columns
    """)

    st.markdown("---")
    st.markdown("## 🔧 Requirements")
    st.markdown("""
    ```bash
    pip install streamlit pandas numpy scipy
    pip install scikit-learn plotly requests
    pip install kaleido  # for PNG exports
    ```
    """)

# Run the main application
if __name__ == "__main__":
    main()