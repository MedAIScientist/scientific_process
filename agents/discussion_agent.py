"""Discussion Agent for Scientific Implications and Context"""

from typing import Dict, List, Any
from camel.agents import ChatAgent
from camel.messages import BaseMessage


class DiscussionAgent:
    def __init__(self, model):
        self.model = model
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Research Discussion Expert",
                content="""You are an expert in scientific discussion and interpretation.
                You specialize in contextualizing findings, discussing implications,
                and connecting results to broader scientific knowledge."""
            )
        )
    
    async def generate_discussion(self,
                                hypotheses: List[Dict[str, str]],
                                results: Dict[str, Any],
                                literature_synthesis: str,
                                research_gaps: List[str]) -> Dict[str, Any]:
        """Generate comprehensive discussion section"""
        
        # Main discussion content
        discussion_text = await self._write_discussion_section(
            hypotheses, results, literature_synthesis, research_gaps
        )
        
        # Implications
        implications = await self._identify_implications(results, hypotheses)
        
        # Future research directions
        future_research = await self._suggest_future_research(results, research_gaps)
        
        # Conclusions
        conclusions = await self._write_conclusions(results, hypotheses, implications)
        
        return {
            "discussion_text": discussion_text,
            "implications": implications,
            "future_research": future_research,
            "conclusions": conclusions,
            "study_contributions": await self._identify_contributions(results, research_gaps)
        }
    
    async def _write_discussion_section(self,
                                      hypotheses: List[Dict[str, str]],
                                      results: Dict[str, Any],
                                      literature_synthesis: str,
                                      research_gaps: List[str]) -> str:
        """Write comprehensive discussion section"""
        
        key_findings = results.get('key_findings', [])
        hypothesis_outcomes = results.get('hypothesis_outcomes', {})
        
        prompt = f"""
        Write a comprehensive discussion section that:
        1. Interprets findings in context of existing literature
        2. Discusses each hypothesis outcome
        3. Addresses study limitations
        4. Explains theoretical and practical implications
        
        Key Findings:
        {chr(10).join(key_findings)}
        
        Hypothesis Outcomes:
        {chr(10).join([f"{k}: {v}" for k, v in hypothesis_outcomes.items()])}
        
        Literature Context:
        {literature_synthesis[:500]}...
        
        Research Gaps Addressed:
        {chr(10).join(research_gaps[:3])}
        
        Write 5-6 paragraphs in academic style.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _identify_implications(self, results: Dict[str, Any], 
                                   hypotheses: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Identify theoretical and practical implications"""
        
        key_findings = results.get('key_findings', [])
        
        prompt = f"""
        Based on the research findings, identify implications in these categories:
        
        Findings:
        {chr(10).join(key_findings)}
        
        Provide implications for:
        1. Theoretical understanding
        2. Practical applications
        3. Policy recommendations
        4. Clinical/applied practice
        
        Format as JSON with categories as keys and lists of implications as values.
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            return json.loads(response.msgs[0].content)
        except:
            return {
                "theoretical": ["Advances understanding in the field"],
                "practical": ["Provides actionable insights"],
                "policy": ["Informs evidence-based policy"],
                "applied": ["Guides professional practice"]
            }
    
    async def _suggest_future_research(self, results: Dict[str, Any], 
                                     research_gaps: List[str]) -> List[str]:
        """Suggest future research directions"""
        
        prompt = f"""
        Based on the study results and identified research gaps, suggest 5-7 specific
        future research directions:
        
        Study findings: {results.get('key_findings', [])[:3]}
        Research gaps: {research_gaps[:3]}
        
        Suggest research that:
        - Addresses limitations of current study
        - Explores unexpected findings
        - Extends findings to new populations
        - Tests causal mechanisms
        - Develops practical applications
        
        Provide as numbered list.
        """
        
        response = self.agent.step(prompt)
        suggestions = [sug.strip() for sug in response.msgs[0].content.split('\n') 
                      if sug.strip() and any(char.isdigit() for char in sug[:3])]
        
        return suggestions
    
    async def _write_conclusions(self, results: Dict[str, Any],
                               hypotheses: List[Dict[str, str]],
                               implications: Dict[str, List[str]]) -> str:
        """Write study conclusions"""
        
        prompt = f"""
        Write a concise conclusions section that:
        1. Summarizes main findings
        2. States conclusions about each hypothesis
        3. Highlights key contributions
        4. Notes important implications
        
        Main findings: {results.get('key_findings', [])[:3]}
        Hypothesis outcomes: {results.get('hypothesis_outcomes', {})}
        Key implications: {implications.get('theoretical', [])[:2]}
        
        Write 2-3 paragraphs for conclusion section.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _identify_contributions(self, results: Dict[str, Any], 
                                    research_gaps: List[str]) -> List[str]:
        """Identify study's scientific contributions"""
        
        prompt = f"""
        Identify the key scientific contributions of this study:
        
        Results: {results.get('key_findings', [])[:3]}
        Gaps addressed: {research_gaps[:3]}
        
        What are the main contributions to:
        - Scientific knowledge
        - Methodological advancement
        - Practical understanding
        
        Provide 3-5 specific contributions as numbered list.
        """
        
        response = self.agent.step(prompt)
        contributions = [cont.strip() for cont in response.msgs[0].content.split('\n') 
                        if cont.strip() and any(char.isdigit() for char in cont[:3])]
        
        return contributions 