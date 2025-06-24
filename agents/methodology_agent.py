"""
Methodology Agent for Scientific Method Development

This agent specializes in developing comprehensive methodology sections
for scientific papers, including study design, statistical methods, and validation procedures.
"""

from typing import Dict, List, Any
import pandas as pd
from camel.agents import ChatAgent
from camel.messages import BaseMessage


class MethodologyAgent:
    """
    Specialized agent for methodology development and validation
    """
    
    def __init__(self, model):
        self.model = model
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Research Methodologist",
                content="""You are an expert research methodologist specializing in:
                1. Study design and experimental methodology
                2. Statistical method selection and justification
                3. Data collection and validation procedures
                4. Ethical considerations and limitations
                5. Reproducibility and methodological rigor
                
                Your methodology sections should be detailed, scientifically sound, and reproducible."""
            )
        )
    
    async def develop_methodology(self, 
                                data: pd.DataFrame,
                                hypotheses: List[Dict[str, str]],
                                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Develop comprehensive methodology"""
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(data)
        
        # Develop study design
        study_design = await self._develop_study_design(data_characteristics, hypotheses)
        
        # Select statistical methods
        statistical_methods = await self._select_statistical_methods(
            data_characteristics, hypotheses, analysis_results
        )
        
        # Create methodology text
        methodology_text = await self._write_methodology_section(
            study_design, statistical_methods, data_characteristics
        )
        
        return {
            "study_design": study_design,
            "statistical_methods": statistical_methods,
            "data_characteristics": data_characteristics,
            "methodology_text": methodology_text,
            "limitations": await self._identify_limitations(data_characteristics, hypotheses)
        }
    
    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics for methodology planning"""
        return {
            "sample_size": len(data),
            "variables": len(data.columns),
            "numeric_variables": len(data.select_dtypes(include=['number']).columns),
            "categorical_variables": len(data.select_dtypes(include=['object', 'category']).columns),
            "missing_data_percentage": (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
            "data_types": data.dtypes.to_dict()
        }
    
    async def _develop_study_design(self, data_char: Dict[str, Any], hypotheses: List[Dict[str, str]]) -> str:
        """Develop study design description"""
        
        hypotheses_text = "\n".join([f"- {h['hypothesis']}" for h in hypotheses])
        
        prompt = f"""
        Based on the data characteristics and hypotheses, describe an appropriate study design:
        
        Data: {data_char['sample_size']} observations, {data_char['variables']} variables
        Hypotheses: {hypotheses_text}
        
        Describe:
        1. Study type (observational, experimental, etc.)
        2. Sampling method and population
        3. Data collection procedures
        4. Study timeline and setting
        
        Write 2-3 paragraphs suitable for a methodology section.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _select_statistical_methods(self, 
                                        data_char: Dict[str, Any],
                                        hypotheses: List[Dict[str, str]],
                                        analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select and justify statistical methods"""
        
        hypotheses_text = "\n".join([f"- {h['hypothesis']}" for h in hypotheses])
        
        prompt = f"""
        Select appropriate statistical methods for this study:
        
        Data characteristics: {data_char}
        Hypotheses: {hypotheses_text}
        
        Provide:
        1. Primary statistical tests with justification
        2. Significance level and effect size measures
        3. Multiple comparison corrections
        4. Software and packages to be used
        
        Format as JSON with clear justifications.
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            return json.loads(response.msgs[0].content)
        except:
            return {"methods": response.msgs[0].content}
    
    async def _write_methodology_section(self,
                                       study_design: str,
                                       statistical_methods: Dict[str, Any],
                                       data_char: Dict[str, Any]) -> str:
        """Write complete methodology section"""
        
        prompt = f"""
        Write a comprehensive methodology section including:
        
        Study Design:
        {study_design}
        
        Statistical Methods:
        {statistical_methods}
        
        Data Characteristics:
        - Sample size: {data_char['sample_size']}
        - Variables: {data_char['variables']}
        - Missing data: {data_char['missing_data_percentage']:.1f}%
        
        Include subsections for:
        1. Study Design
        2. Data Collection
        3. Statistical Analysis
        4. Ethical Considerations
        
        Write in formal academic style.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _identify_limitations(self, data_char: Dict[str, Any], hypotheses: List[Dict[str, str]]) -> List[str]:
        """Identify methodological limitations"""
        
        prompt = f"""
        Identify 4-6 methodological limitations for this study:
        
        Data: {data_char['sample_size']} samples, {data_char['missing_data_percentage']:.1f}% missing data
        Hypotheses: {len(hypotheses)} research hypotheses
        
        Consider:
        - Sample size and power
        - Data quality and completeness
        - Study design limitations
        - Generalizability issues
        - Causal inference limitations
        
        Provide as numbered list.
        """
        
        response = self.agent.step(prompt)
        limitations = [lim.strip() for lim in response.msgs[0].content.split('\n') 
                      if lim.strip() and any(char.isdigit() for char in lim[:3])]
        
        return limitations 