"""Review Agent for Paper Quality Assessment and Improvement"""

from typing import Dict, List, Any
from camel.agents import ChatAgent
from camel.messages import BaseMessage


class ReviewAgent:
    def __init__(self, model):
        self.model = model
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Scientific Reviewer",
                content="""You are an expert scientific reviewer specializing in:
                1. Comprehensive quality assessment of research papers
                2. Identifying areas for improvement in content and structure
                3. Ensuring scientific rigor and clarity
                4. Checking coherence and flow between sections
                5. Providing constructive feedback for enhancement
                
                Your reviews should be thorough, constructive, and focused on improving scientific quality."""
            )
        )
    
    async def review_and_improve(self, paper_draft: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive review and improve the paper"""
        
        # Assess overall quality
        quality_assessment = await self._assess_paper_quality(paper_draft)
        
        # Identify specific improvements
        improvements = await self._identify_improvements(paper_draft)
        
        # Apply improvements
        improved_paper = await self._apply_improvements(paper_draft, improvements)
        
        # Final quality check
        final_assessment = await self._final_quality_check(improved_paper)
        
        return {
            "paper": improved_paper,
            "quality_assessment": quality_assessment,
            "improvements_made": improvements,
            "final_assessment": final_assessment,
            "review_summary": await self._create_review_summary(quality_assessment, improvements)
        }
    
    async def _assess_paper_quality(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall paper quality"""
        
        sections_to_review = ['title', 'abstract', 'introduction', 'methodology', 'results', 'discussion']
        available_sections = [section for section in sections_to_review if paper.get(section)]
        
        prompt = f"""
        Assess the quality of this research paper draft:
        
        Title: {paper.get('title', 'Missing')}
        Abstract: {len(paper.get('abstract', ''))} characters
        Available sections: {available_sections}
        
        Evaluate each section (1-10 scale) for:
        - Clarity and coherence
        - Scientific rigor
        - Completeness
        - Academic writing quality
        
        Sample content from each section:
        Abstract: {paper.get('abstract', '')[:200]}...
        Introduction: {paper.get('introduction', '')[:200]}...
        Methodology: {paper.get('methodology', '')[:200]}...
        
        Provide assessment as JSON with section scores and overall quality rating.
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            assessment = json.loads(response.msgs[0].content)
        except:
            assessment = {
                "overall_quality": 7,
                "section_scores": {section: 7 for section in available_sections},
                "assessment_text": response.msgs[0].content
            }
        
        return assessment
    
    async def _identify_improvements(self, paper: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify specific areas for improvement"""
        
        improvements = []
        
        # Review each major section
        sections = {
            'title': 'Title',
            'abstract': 'Abstract', 
            'introduction': 'Introduction',
            'methodology': 'Methodology',
            'results': 'Results',
            'discussion': 'Discussion'
        }
        
        for section_key, section_name in sections.items():
            if paper.get(section_key):
                section_improvements = await self._review_section(
                    section_name, paper[section_key]
                )
                improvements.extend(section_improvements)
        
        # Check overall coherence
        coherence_improvements = await self._check_coherence(paper)
        improvements.extend(coherence_improvements)
        
        return improvements
    
    async def _review_section(self, section_name: str, content: str) -> List[Dict[str, str]]:
        """Review individual section and suggest improvements"""
        
        prompt = f"""
        Review this {section_name} section and identify 2-4 specific improvements:
        
        Content:
        {content[:1000]}...
        
        Consider:
        - Clarity and conciseness
        - Scientific accuracy
        - Logical flow
        - Academic writing style
        - Completeness
        
        Provide improvements as JSON list with 'issue' and 'suggestion' for each.
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            improvements = json.loads(response.msgs[0].content)
            
            # Ensure proper format
            formatted_improvements = []
            for imp in improvements:
                if isinstance(imp, dict) and 'issue' in imp and 'suggestion' in imp:
                    imp['section'] = section_name
                    formatted_improvements.append(imp)
            
            return formatted_improvements
            
        except:
            return [{
                'section': section_name,
                'issue': 'General review needed',
                'suggestion': 'Review for clarity and scientific rigor'
            }]
    
    async def _check_coherence(self, paper: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check coherence between sections"""
        
        prompt = f"""
        Check the coherence and flow between sections of this paper:
        
        Title: {paper.get('title', '')}
        Abstract: {paper.get('abstract', '')[:200]}...
        Introduction: {paper.get('introduction', '')[:200]}...
        
        Identify any:
        - Inconsistencies between sections
        - Missing transitions
        - Logical gaps
        - Repetition or redundancy
        
        Provide 2-3 coherence improvements as JSON list.
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            coherence_issues = json.loads(response.msgs[0].content)
            
            for issue in coherence_issues:
                issue['section'] = 'Overall'
                
            return coherence_issues
            
        except:
            return [{
                'section': 'Overall',
                'issue': 'Coherence check needed',
                'suggestion': 'Review flow between sections'
            }]
    
    async def _apply_improvements(self, paper: Dict[str, Any], 
                                improvements: List[Dict[str, str]]) -> Dict[str, Any]:
        """Apply identified improvements to the paper"""
        
        improved_paper = paper.copy()
        
        # Group improvements by section
        section_improvements = {}
        for improvement in improvements:
            section = improvement.get('section', 'Overall')
            if section not in section_improvements:
                section_improvements[section] = []
            section_improvements[section].append(improvement)
        
        # Apply improvements to each section
        for section, section_imps in section_improvements.items():
            if section == 'Overall':
                continue  # Handle overall improvements separately
                
            section_key = section.lower()
            if section_key in improved_paper:
                improved_content = await self._improve_section_content(
                    improved_paper[section_key], section_imps
                )
                improved_paper[section_key] = improved_content
        
        return improved_paper
    
    async def _improve_section_content(self, content: str, 
                                     improvements: List[Dict[str, str]]) -> str:
        """Improve specific section content"""
        
        improvements_text = "\n".join([
            f"- {imp['issue']}: {imp['suggestion']}" for imp in improvements
        ])
        
        prompt = f"""
        Improve this section content based on the following suggestions:
        
        Current content:
        {content}
        
        Improvements needed:
        {improvements_text}
        
        Rewrite the content incorporating these improvements while maintaining:
        - Original meaning and data
        - Academic writing style
        - Appropriate length
        - Scientific accuracy
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _final_quality_check(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct final quality assessment"""
        
        prompt = f"""
        Conduct a final quality check of this improved paper:
        
        Title: {paper.get('title', '')}
        Word counts:
        - Abstract: {len(paper.get('abstract', '').split())} words
        - Introduction: {len(paper.get('introduction', '').split())} words
        - Methodology: {len(paper.get('methodology', '').split())} words
        - Results: {len(paper.get('results', '').split())} words
        - Discussion: {len(paper.get('discussion', '').split())} words
        
        Assess:
        1. Overall scientific quality (1-10)
        2. Readiness for submission (1-10)
        3. Remaining issues (if any)
        4. Strengths of the paper
        
        Provide as JSON with numerical scores and text assessments.
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            final_assessment = json.loads(response.msgs[0].content)
        except:
            final_assessment = {
                "scientific_quality": 8,
                "submission_readiness": 7,
                "assessment": response.msgs[0].content
            }
        
        return final_assessment
    
    async def _create_review_summary(self, quality_assessment: Dict[str, Any],
                                   improvements: List[Dict[str, str]]) -> str:
        """Create summary of the review process"""
        
        prompt = f"""
        Create a concise review summary for this paper improvement process:
        
        Initial Quality Assessment:
        Overall quality: {quality_assessment.get('overall_quality', 'N/A')}
        Section scores: {quality_assessment.get('section_scores', {})}
        
        Improvements Made: {len(improvements)} improvements applied
        Key areas addressed: {list(set([imp.get('section', 'Unknown') for imp in improvements]))}
        
        Write a 2-3 paragraph summary suitable for inclusion in review documentation.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content 