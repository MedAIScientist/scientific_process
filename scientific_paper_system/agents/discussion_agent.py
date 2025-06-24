"""
ENHANCED Discussion Agent for Creating Comprehensive Scientific Discussions

This agent creates detailed discussion sections with literature integration,
statistical interpretation, and clinical implications.
"""

from typing import Dict, List, Any
from camel.agents import ChatAgent
from camel.messages import BaseMessage


class DiscussionAgent:
    def __init__(self, model):
        self.model = model
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Senior Research Scientist and Clinical Expert",
                content="""You are a senior research scientist expert in scientific discussion writing.
                Your role is to:
                
                1. Interpret statistical results in clinical and practical contexts
                2. Compare findings with existing literature using specific citations
                3. Discuss methodological strengths and limitations honestly
                4. Propose clinical and public health implications
                5. Suggest specific directions for future research
                6. Address potential confounding factors and alternative explanations
                
                Your discussions must be:
                - Evidence-based with specific statistical interpretations
                - Critical and balanced in assessment of findings
                - Clinically relevant and practically applicable
                - Well-structured with clear logical flow
                - Honest about study limitations and uncertainties
                
                Always provide specific, actionable insights rather than vague generalizations."""
            )
        )
    
    async def create_discussion(self, 
                              results: Dict[str, Any],
                              literature_context: Dict[str, Any],
                              hypotheses: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create comprehensive discussion section"""
        
        print("💭 Creating comprehensive discussion...")
        
        # Generate main findings interpretation
        main_findings = await self._interpret_main_findings(results, hypotheses)
        
        # Create literature comparison
        literature_comparison = await self._compare_with_literature(results, literature_context, hypotheses)
        
        # Discuss methodological considerations
        methodological_discussion = await self._discuss_methodology(results)
        
        # Generate clinical implications
        clinical_implications = await self._discuss_clinical_implications(results, hypotheses)
        
        # Create limitations discussion
        limitations = await self._discuss_limitations(results)
        
        # Suggest future research
        future_research = await self._suggest_future_research(results, hypotheses)
        
        # Compile complete discussion
        discussion_text = await self._compile_discussion_section(
            main_findings, literature_comparison, methodological_discussion,
            clinical_implications, limitations, future_research
        )
        
        # Generate conclusions
        conclusions = await self._generate_conclusions(results, hypotheses)
        
        return {
            "discussion_text": discussion_text,
            "conclusions": conclusions,
            "main_findings": main_findings,
            "clinical_implications": clinical_implications,
            "limitations": limitations,
            "future_research": future_research
        }
    
    async def _interpret_main_findings(self, results: Dict[str, Any], 
                                     hypotheses: List[Dict[str, str]]) -> str:
        """Interpret main statistical findings"""
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        results_summary = str(results.get('detailed_statistics', {}))[:1000]
        
        prompt = f"""
        Interpret the main findings from this study in a comprehensive discussion format:
        
        Research Hypotheses:
        {hypotheses_text}
        
        Statistical Results Summary:
        {results_summary}
        
        Provide a detailed interpretation that:
        
        1. **Summarizes Key Findings** (2-3 sentences)
        - State the main results clearly with specific statistics
        - Indicate which hypotheses were supported/not supported
        - Highlight the most important discoveries
        
        2. **Statistical Interpretation** (3-4 sentences)
        - Explain the clinical/practical meaning of effect sizes
        - Discuss the strength of associations found
        - Comment on statistical significance and confidence intervals
        - Address any unexpected findings
        
        3. **Hypothesis Evaluation** (2-3 sentences)
        - Explicitly address each hypothesis with evidence
        - Explain why hypotheses were or were not supported
        - Discuss partial support if applicable
        
        Use formal academic language with specific statistical details.
        Write as if this is the opening of a Discussion section.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _compare_with_literature(self, results: Dict[str, Any], 
                                     literature_context: Dict[str, Any],
                                     hypotheses: List[Dict[str, str]]) -> str:
        """Compare findings with existing literature"""
        
        literature_review = literature_context.get('detailed_literature_review', '')[:2000]
        
        prompt = f"""
        Compare the current study findings with existing literature:
        
        Literature Context:
        {literature_review}
        
        Study Results: {str(results.get('detailed_statistics', {}))[:1000]}
        
        Create a literature comparison using short, focused paragraphs:
        
        **Paragraph 1: Consistent Findings** (3-4 sentences)
        - Identify findings that align with previous research
        - Cite specific studies with similar results and strengthen confidence through replication
        
        **Paragraph 2: Novel Contributions** (3-4 sentences)  
        - Highlight new or unique findings that extend current knowledge
        - Explain original aspects of the research and their significance
        
        **Paragraph 3: Contradictory Results** (3-4 sentences, if applicable)
        - Address any findings that differ from literature with proposed explanations
        - Discuss methodological or population differences that may explain discrepancies
        
        **Paragraph 4: Methodological Context** (3-4 sentences)
        - Compare analytical approaches with previous studies
        - Highlight methodological improvements or innovations used in this research
        
        Requirements:
        - Write 3-4 focused paragraphs
        - Each paragraph should contain 3-4 sentences maximum
        - Reference studies by number [1], [2], etc. throughout
        - Use formal academic comparison language
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _discuss_methodology(self, results: Dict[str, Any]) -> str:
        """Discuss methodological strengths and considerations"""
        
        prompt = f"""
        Discuss the methodological aspects of this study:
        
        Statistical Analysis Performed: {str(results.get('detailed_statistics', {}))[:800]}
        
        Address methodological aspects using short, focused paragraphs:
        
        **Paragraph 1: Methodological Strengths** (3-4 sentences)
        - Describe robust statistical approaches used and appropriate sample size
        - Highlight comprehensive variable assessment and quality control measures
        
        **Paragraph 2: Analytical Rigor** (3-4 sentences)
        - Discuss multiple testing corrections applied and appropriate statistical test selection
        - Emphasize effect size reporting and confidence interval utilization
        
        **Paragraph 3: Design Considerations** (3-4 sentences)
        - Address study design appropriateness for research questions
        - Describe data collection standardization and bias minimization strategies
        
        Requirements:
        - Write 3 focused paragraphs
        - Each paragraph should contain 3-4 sentences maximum
        - Use formal academic discussion style
        - Be honest about both strengths and potential methodological considerations
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _discuss_clinical_implications(self, results: Dict[str, Any], 
                                           hypotheses: List[Dict[str, str]]) -> str:
        """Discuss clinical and public health implications"""
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        
        prompt = f"""
        Discuss the clinical and public health implications of these findings:
        
        Research Focus: {hypotheses_text}
        
        Key Results: {str(results.get('detailed_statistics', {}))[:800]}
        
        Address:
        
        1. **Clinical Practice Implications** (3-4 sentences)
        - How findings inform clinical decision-making
        - Specific recommendations for healthcare providers
        - Patient assessment and risk stratification guidance
        - Treatment or intervention considerations
        
        2. **Public Health Significance** (3-4 sentences)
        - Population-level health implications
        - Prevention strategy recommendations
        - Health policy considerations
        - Resource allocation guidance
        
        3. **Practical Applications** (2-3 sentences)
        - Implementation considerations
        - Cost-effectiveness implications
        - Feasibility of recommended interventions
        - Scalability to different populations
        
        4. **Risk Communication** (1-2 sentences)
        - How to communicate findings to patients/public
        - Important caveats or limitations to emphasize
        
        Use specific, actionable language that clinicians and policymakers can implement.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _discuss_limitations(self, results: Dict[str, Any]) -> str:
        """Discuss study limitations honestly and comprehensively"""
        
        prompt = f"""
        Provide an honest, comprehensive discussion of study limitations:
        
        Study Context: {str(results.get('detailed_statistics', {}))[:600]}
        
        Address these limitation categories:
        
        1. **Design Limitations** (2-3 sentences)
        - Cross-sectional vs. longitudinal design constraints
        - Causal inference limitations
        - Temporal relationship uncertainties
        - Selection bias considerations
        
        2. **Measurement Limitations** (2-3 sentences)
        - Self-report data limitations
        - Measurement error possibilities
        - Missing data implications
        - Recall bias considerations
        
        3. **Statistical Limitations** (2-3 sentences)
        - Multiple comparison considerations
        - Effect size interpretation caveats
        - Confounding variable limitations
        - Power analysis considerations
        
        4. **Generalizability Limitations** (2-3 sentences)
        - Population representativeness
        - Geographic or demographic constraints
        - Setting-specific considerations
        - External validity concerns
        
        5. **Methodological Considerations** (1-2 sentences)
        - Unmeasured confounding possibilities
        - Residual bias sources
        - Alternative explanation considerations
        
        Be specific and honest while maintaining scientific credibility.
        Explain how limitations affect interpretation of results.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _suggest_future_research(self, results: Dict[str, Any], 
                                     hypotheses: List[Dict[str, str]]) -> str:
        """Suggest specific directions for future research"""
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        
        prompt = f"""
        Suggest specific, actionable directions for future research:
        
        Current Study Focus: {hypotheses_text}
        
        Study Results: {str(results.get('detailed_statistics', {}))[:600]}
        
        Propose:
        
        1. **Methodological Improvements** (2-3 sentences)
        - Longitudinal study designs needed
        - Experimental or intervention studies
        - Larger or more diverse sample requirements
        - Improved measurement approaches
        
        2. **Research Questions to Address** (3-4 sentences)
        - Specific unanswered questions from current findings
        - Causal mechanism investigations needed
        - Dose-response relationship studies
        - Intervention effectiveness trials
        
        3. **Population-Specific Studies** (2-3 sentences)
        - Different demographic groups to study
        - Geographic or cultural contexts to explore
        - Special populations requiring investigation
        - Comparative effectiveness research needs
        
        4. **Mechanistic Research** (2-3 sentences)
        - Biological or behavioral mechanisms to investigate
        - Mediating pathway studies needed
        - Molecular or genetic research directions
        - Systems-level analysis requirements
        
        5. **Implementation Research** (1-2 sentences)
        - Real-world effectiveness studies
        - Implementation strategy research
        - Health system integration studies
        
        Be specific and feasible in your suggestions.
        Connect recommendations directly to current findings and limitations.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _compile_discussion_section(self, main_findings: str, literature_comparison: str,
                                        methodological_discussion: str, clinical_implications: str,
                                        limitations: str, future_research: str) -> str:
        """Compile all discussion components into coherent section"""
        
        prompt = f"""
        Compile these discussion components into a well-structured Discussion section:
        
        Main Findings:
        {main_findings}
        
        Literature Comparison:
        {literature_comparison}
        
        Methodological Discussion:
        {methodological_discussion}
        
        Clinical Implications:
        {clinical_implications}
        
        Limitations:
        {limitations}
        
        Future Research:
        {future_research}
        
        Create a cohesive Discussion section using shorter, focused paragraphs for better readability:
        
        **Opening discussion (2-3 short paragraphs):**
        - Paragraph 1: Interpret the principal findings and their direct meaning (3-4 sentences)
        - Paragraph 2: Connect statistical results to practical significance (3-4 sentences)
        - Paragraph 3: Address how results answer the research questions (3-4 sentences)
        
        **Literature integration (2-3 short paragraphs):**
        - Paragraph 4: Compare findings with previous research showing confirmatory results (3-4 sentences)
        - Paragraph 5: Discuss novel contributions and extensions to existing knowledge (3-4 sentences)
        - Paragraph 6: Address any contradictory findings or methodological considerations (3-4 sentences)
        
        **Broader implications (2-3 short paragraphs):**
        - Paragraph 7: Present clinical implications and practical applications (3-4 sentences)
        - Paragraph 8: Discuss public health implications and policy recommendations (3-4 sentences)
        - Paragraph 9: Provide recommendations for different stakeholders (3-4 sentences)
        
        **Honest assessment (2 short paragraphs):**
        - Paragraph 10: Acknowledge study limitations with specific examples (3-4 sentences)
        - Paragraph 11: Suggest future research directions as natural extensions (3-4 sentences)
        
        Requirements:
        - Each paragraph should focus on ONE main concept
        - Use 3-4 sentences per paragraph maximum
        - Include smooth transitions between paragraphs
        - Use formal academic writing style
        - Include specific statistical details and practical implications
        - Write 9-11 focused paragraphs for comprehensive yet readable discussion
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _generate_conclusions(self, results: Dict[str, Any], 
                                  hypotheses: List[Dict[str, str]]) -> str:
        """Generate comprehensive conclusions section"""
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        
        prompt = f"""
        Generate a comprehensive Conclusions section based on:
        
        Research Hypotheses:
        {hypotheses_text}
        
        Key Results: {str(results.get('detailed_statistics', {}))[:800]}
        
        The Conclusions section should:
        
        1. **Main Conclusions** (2-3 sentences)
        - Clearly state the primary findings
        - Indicate level of evidence and confidence
        - Summarize hypothesis testing outcomes
        
        2. **Practical Significance** (2-3 sentences)
        - Highlight clinical and public health importance
        - Emphasize actionable implications
        - Note contribution to scientific knowledge
        
        3. **Implementation Relevance** (1-2 sentences)
        - How findings can be applied in practice
        - Recommendations for different stakeholders
        
        4. **Future Impact** (1-2 sentences)
        - Potential influence on future research
        - Expected impact on clinical practice or policy
        
        Write in declarative, confident language while acknowledging appropriate limitations.
        Make conclusions specific and actionable.
        Ensure conclusions are directly supported by the study results.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content 