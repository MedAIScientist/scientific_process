"""Results Agent for Data Interpretation and Presentation"""

from typing import Dict, List, Any
import pandas as pd
from camel.agents import ChatAgent
from camel.messages import BaseMessage


class ResultsAgent:
    def __init__(self, model):
        self.model = model
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Results Analyst",
                content="""You are an expert in interpreting and presenting research results.
                You specialize in connecting statistical findings to research hypotheses and
                creating clear, scientifically accurate results sections."""
            )
        )
    
    async def interpret_results(self, 
                              data: pd.DataFrame,
                              hypotheses: List[Dict[str, str]],
                              analysis_results: Dict[str, Any],
                              methodology: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret and present analysis results"""
        
        # Extract key findings
        key_findings = await self._extract_key_findings(analysis_results, hypotheses)
        
        # Create results narrative
        results_text = await self._write_results_section(key_findings, analysis_results, hypotheses)
        
        # Generate tables and figures descriptions
        visualizations = await self._create_results_visualizations(analysis_results, hypotheses)
        
        return {
            "key_findings": key_findings,
            "results_text": results_text,
            "statistical_summary": self._summarize_statistical_tests(analysis_results),
            "visualizations": visualizations,
            "hypothesis_outcomes": await self._evaluate_hypotheses(key_findings, hypotheses)
        }
    
    async def _extract_key_findings(self, analysis_results: Dict[str, Any], hypotheses: List[Dict[str, str]]) -> List[str]:
        """Extract key findings from analysis results"""
        
        prompt = f"""
        Extract 5-7 key findings from these analysis results that directly address the research hypotheses:
        
        Statistical Tests: {len(analysis_results.get('statistical_tests', []))} tests performed
        Significant Results: {len([t for t in analysis_results.get('statistical_tests', []) if hasattr(t, 'p_value') and t.p_value < 0.05])}
        
        Hypotheses:
        {chr(10).join([f"- {h['hypothesis']}" for h in hypotheses])}
        
        Machine Learning Results: {analysis_results.get('machine_learning', {})}
        
        Provide key findings as numbered list with statistical evidence.
        """
        
        response = self.agent.step(prompt)
        findings = [finding.strip() for finding in response.msgs[0].content.split('\n') 
                   if finding.strip() and any(char.isdigit() for char in finding[:3])]
        
        return findings
    
    async def _write_results_section(self, key_findings: List[str], 
                                   analysis_results: Dict[str, Any],
                                   hypotheses: List[Dict[str, str]]) -> str:
        """Write comprehensive results section"""
        
        prompt = f"""
        Write a comprehensive results section that:
        1. Presents findings in logical order
        2. Connects results to specific hypotheses
        3. Reports statistical details appropriately
        4. References tables and figures
        
        Key Findings:
        {chr(10).join(key_findings)}
        
        Hypotheses:
        {chr(10).join([f"- {h['hypothesis']}" for h in hypotheses])}
        
        Write 4-5 paragraphs in formal academic style.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    def _summarize_statistical_tests(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize statistical test results"""
        tests = analysis_results.get('statistical_tests', [])
        
        return {
            "total_tests": len(tests),
            "significant_tests": len([t for t in tests if hasattr(t, 'p_value') and t.p_value < 0.05]),
            "test_types": list(set([t.test_name.split(':')[0] for t in tests if hasattr(t, 'test_name')])),
            "effect_sizes": [t.effect_size for t in tests if hasattr(t, 'effect_size') and t.effect_size]
        }
    
    async def _create_results_visualizations(self, analysis_results: Dict[str, Any], 
                                           hypotheses: List[Dict[str, str]]) -> List[str]:
        """Create descriptions of results visualizations"""
        return [
            "Statistical test results summary table",
            "Effect size visualization for significant findings",
            "Correlation matrix heatmap for key variables",
            "Distribution plots for primary outcome variables"
        ]
    
    async def _evaluate_hypotheses(self, key_findings: List[str], 
                                 hypotheses: List[Dict[str, str]]) -> Dict[str, str]:
        """Evaluate each hypothesis based on findings"""
        
        evaluations = {}
        for i, hypothesis in enumerate(hypotheses):
            prompt = f"""
            Based on the key findings, evaluate this hypothesis:
            
            Hypothesis: {hypothesis['hypothesis']}
            
            Key Findings:
            {chr(10).join(key_findings)}
            
            Provide evaluation: "Supported", "Not Supported", or "Partially Supported" with brief justification.
            """
            
            response = self.agent.step(prompt)
            evaluations[f"Hypothesis {i+1}"] = response.msgs[0].content.strip()
        
        return evaluations 