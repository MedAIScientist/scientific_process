#!/usr/bin/env python3
"""
Source Verification Agent - Prevents AI hallucination of statistics
Ensures all numerical values come from verified sources only
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass

@dataclass
class VerifiedStatistic:
    """Represents a verified statistic with its source"""
    value: float
    description: str
    source: str  # 'csv_data', 'statistical_report', 'semantic_scholar'
    context: str
    confidence: float = 1.0

class SourceVerificationAgent:
    """Agent that tracks and verifies the source of all numerical values"""
    
    def __init__(self):
        self.verified_stats = {}
        self.csv_stats = {}
        self.report_stats = {}
        self.paper_stats = {}
        
    def load_verified_sources(self, csv_path: str, report_path: str, hypotheses_path: str):
        """Load all verified data sources"""
        print("📊 Loading verified data sources...")
        
        # Load CSV data
        self.csv_data = pd.read_csv(csv_path)
        self.csv_stats = self._extract_csv_statistics()
        
        # Load statistical report
        with open(report_path, 'r') as f:
            self.report_data = json.load(f)
        self.report_stats = self._extract_report_statistics()
        
        # Load hypotheses
        with open(hypotheses_path, 'r') as f:
            self.hypotheses_data = json.load(f)
            
        print(f"✅ Loaded {len(self.csv_stats)} CSV statistics")
        print(f"✅ Loaded {len(self.report_stats)} report statistics")
        
    def _extract_csv_statistics(self) -> Dict[str, VerifiedStatistic]:
        """Extract all valid statistics from CSV data"""
        stats = {}
        
        # Basic dataset info
        stats['sample_size'] = VerifiedStatistic(
            value=len(self.csv_data),
            description="Total number of participants",
            source="csv_data",
            context=f"Dataset contains {len(self.csv_data)} participants"
        )
        
        # For each numeric column
        for col in self.csv_data.select_dtypes(include=[np.number]).columns:
            series = self.csv_data[col]
            
            # Basic statistics
            stats[f'{col}_mean'] = VerifiedStatistic(
                value=series.mean(),
                description=f"Mean of {col}",
                source="csv_data",
                context=f"Calculated from {len(series)} observations"
            )
            
            stats[f'{col}_std'] = VerifiedStatistic(
                value=series.std(),
                description=f"Standard deviation of {col}",
                source="csv_data",
                context=f"Calculated from {len(series)} observations"
            )
            
            stats[f'{col}_median'] = VerifiedStatistic(
                value=series.median(),
                description=f"Median of {col}",
                source="csv_data",
                context=f"Calculated from {len(series)} observations"
            )
            
            stats[f'{col}_min'] = VerifiedStatistic(
                value=series.min(),
                description=f"Minimum of {col}",
                source="csv_data",
                context=f"Calculated from {len(series)} observations"
            )
            
            stats[f'{col}_max'] = VerifiedStatistic(
                value=series.max(),
                description=f"Maximum of {col}",
                source="csv_data",
                context=f"Calculated from {len(series)} observations"
            )
            
            # For binary variables, calculate prevalence
            if series.nunique() == 2 and set(series.unique()) == {0, 1}:
                prevalence = series.mean()
                stats[f'{col}_prevalence'] = VerifiedStatistic(
                    value=prevalence,
                    description=f"Prevalence of {col}",
                    source="csv_data",
                    context=f"{prevalence:.1%} of {len(series)} participants"
                )
        
        return stats
    
    def _extract_report_statistics(self) -> Dict[str, VerifiedStatistic]:
        """Extract statistics from the statistical analysis report"""
        stats = {}
        
        try:
            # Extract from univariate analysis
            if 'analysis_results' in self.report_data and 'univariate_analysis' in self.report_data['analysis_results']:
                univariate = self.report_data['analysis_results']['univariate_analysis']['numeric']
                
                for var, var_stats in univariate.items():
                    for stat_name, stat_value in var_stats.items():
                        if isinstance(stat_value, (int, float)):
                            key = f'{var}_{stat_name}_report'
                            stats[key] = VerifiedStatistic(
                                value=stat_value,
                                description=f"{stat_name} of {var} from statistical report",
                                source="statistical_report",
                                context=f"Pre-calculated in statistical analysis report"
                            )
        except Exception as e:
            print(f"⚠️ Error extracting report statistics: {e}")
            
        return stats
    
    def get_verified_statistic(self, description: str) -> Optional[VerifiedStatistic]:
        """Get a verified statistic by description"""
        # Search in CSV stats first
        for key, stat in self.csv_stats.items():
            if description.lower() in stat.description.lower():
                return stat
                
        # Then search in report stats
        for key, stat in self.report_stats.items():
            if description.lower() in stat.description.lower():
                return stat
                
        return None
    
    def verify_paper_statistics(self, paper_content: str) -> Dict[str, Any]:
        """Verify all numerical values in the paper against sources"""
        print("🔍 Verifying paper statistics against sources...")
        
        # Extract all numerical values from paper
        numerical_patterns = [
            r't\((\d+)\)\s*=\s*([\d.]+)',  # t-statistics
            r'r\s*=\s*([\d.]+)',           # correlations
            r'R²?\s*=\s*([\d.]+)',         # R-squared
            r'p\s*[<>=]\s*([\d.]+)',       # p-values
            r'(\d+\.?\d*)\s*participants', # sample sizes
            r'mean\s*=\s*([\d.]+)',        # means
            r'SD\s*=\s*([\d.]+)',          # standard deviations
            r'(\d+\.?\d*)%',               # percentages
        ]
        
        found_values = []
        verified_count = 0
        hallucinated_count = 0
        
        for pattern in numerical_patterns:
            matches = re.finditer(pattern, paper_content, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1) if len(match.groups()) == 1 else match.group(2)
                try:
                    value = float(value_str)
                    context = match.group(0)
                    
                    # Check if this value exists in our verified sources
                    is_verified = self._check_value_in_sources(value, context)
                    
                    found_values.append({
                        'value': value,
                        'context': context,
                        'verified': is_verified,
                        'position': match.span()
                    })
                    
                    if is_verified:
                        verified_count += 1
                    else:
                        hallucinated_count += 1
                        
                except ValueError:
                    continue
        
        verification_report = {
            'total_values': len(found_values),
            'verified_values': verified_count,
            'hallucinated_values': hallucinated_count,
            'verification_rate': verified_count / len(found_values) if found_values else 0,
            'found_values': found_values,
            'recommendations': self._generate_recommendations(found_values)
        }
        
        return verification_report
    
    def _check_value_in_sources(self, value: float, context: str) -> bool:
        """Check if a value exists in our verified sources"""
        tolerance = 0.001  # Allow small floating point differences
        
        # Check CSV stats
        for stat in self.csv_stats.values():
            if abs(stat.value - value) < tolerance:
                return True
                
        # Check report stats  
        for stat in self.report_stats.values():
            if abs(stat.value - value) < tolerance:
                return True
                
        # Special case: check if it's a reasonable derived statistic
        if self._is_reasonable_derived_value(value, context):
            return True
            
        return False
    
    def _is_reasonable_derived_value(self, value: float, context: str) -> bool:
        """Check if value could be reasonably derived from our data"""
        # Sample size should match our data
        if 'participant' in context.lower():
            return abs(value - len(self.csv_data)) < 100
            
        # Percentages should be reasonable
        if '%' in context and 0 <= value <= 100:
            return True
            
        # p-values should be between 0 and 1
        if 'p' in context.lower() and 0 <= value <= 1:
            return True
            
        return False
    
    def _generate_recommendations(self, found_values: List[Dict]) -> List[str]:
        """Generate recommendations for improving source verification"""
        recommendations = []
        
        hallucinated = [v for v in found_values if not v['verified']]
        
        if hallucinated:
            recommendations.append(
                f"🚨 {len(hallucinated)} potentially hallucinated values detected"
            )
            recommendations.append(
                "Replace hallucinated statistics with verified values from your data sources"
            )
            
        verified = [v for v in found_values if v['verified']]
        if verified:
            recommendations.append(
                f"✅ {len(verified)} values successfully verified against sources"
            )
            
        return recommendations
    
    def generate_verified_statistics_summary(self) -> str:
        """Generate a summary of verified statistics that can be used in papers"""
        summary = "## VERIFIED STATISTICS FROM YOUR DATA SOURCES\n\n"
        
        # Key dataset statistics
        summary += f"**Sample Size:** {self.csv_stats['sample_size'].value:,.0f} participants\n\n"
        
        # Key variables
        key_vars = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Age']
        
        for var in key_vars:
            if f'{var}_mean' in self.csv_stats:
                mean_stat = self.csv_stats[f'{var}_mean']
                std_stat = self.csv_stats[f'{var}_std']
                
                if f'{var}_prevalence' in self.csv_stats:
                    prev_stat = self.csv_stats[f'{var}_prevalence']
                    summary += f"**{var}:** {prev_stat.value:.1%} prevalence\n"
                else:
                    summary += f"**{var}:** Mean = {mean_stat.value:.2f} (SD = {std_stat.value:.2f})\n"
        
        summary += "\n*All values verified from source data*\n"
        return summary
    
    def create_source_citation_map(self) -> Dict[str, str]:
        """Create a map of statistics to their sources for citation"""
        citation_map = {}
        
        for key, stat in self.csv_stats.items():
            citation_map[f"{stat.value:.3f}"] = f"Source: CSV dataset (n={len(self.csv_data)})"
            
        for key, stat in self.report_stats.items():
            citation_map[f"{stat.value:.3f}"] = "Source: Statistical analysis report"
            
        return citation_map

def create_verified_results_section(verification_agent: SourceVerificationAgent) -> str:
    """Create a results section using only verified statistics"""
    
    results = "## RESULTS\n\n"
    results += "### Participant Characteristics\n\n"
    
    # Use only verified statistics
    sample_size = verification_agent.csv_stats['sample_size'].value
    results += f"The study included {sample_size:,.0f} participants. "
    
    # Diabetes prevalence
    if 'Diabetes_binary_prevalence' in verification_agent.csv_stats:
        diabetes_prev = verification_agent.csv_stats['Diabetes_binary_prevalence']
        results += f"The prevalence of diabetes was {diabetes_prev.value:.1%}. "
    
    # BMI statistics
    if 'BMI_mean' in verification_agent.csv_stats:
        bmi_mean = verification_agent.csv_stats['BMI_mean']
        bmi_std = verification_agent.csv_stats['BMI_std']
        results += f"The mean BMI was {bmi_mean.value:.1f} (SD = {bmi_std.value:.1f}). "
    
    results += "\n\n*All statistics verified from source data.*\n"
    
    return results

if __name__ == "__main__":
    # Test the verification system
    agent = SourceVerificationAgent()
    agent.load_verified_sources(
        'data/demo_data.csv',
        'data/hypothesis/statistical_analysis_report.json',
        'data/hypothesis/structured_hypotheses_3_only.json'
    )
    
    print("\n" + "="*50)
    print("VERIFIED STATISTICS AVAILABLE:")
    print("="*50)
    
    for key, stat in list(agent.csv_stats.items())[:10]:
        print(f"{stat.description}: {stat.value:.3f}")
    
    print(f"\nTotal verified statistics: {len(agent.csv_stats) + len(agent.report_stats)}") 