"""
Literature Review Agent for Synthesizing Scientific Literature

This agent specializes in analyzing and synthesizing scientific literature
to create comprehensive literature reviews and identify research gaps.
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from camel.agents import ChatAgent
from camel.messages import BaseMessage


@dataclass
class LiteratureTheme:
    """Represents a theme found in literature"""
    theme_name: str
    description: str
    supporting_papers: List[str]
    key_findings: List[str]
    contradictions: Optional[List[str]] = None


class LiteratureReviewAgent:
    """
    Specialized agent for literature review and synthesis
    """
    
    def __init__(self, model):
        """
        Initialize the literature review agent
        
        Args:
            model: CAMEL model instance
        """
        self.model = model
        
        # Initialize the chat agent with specialized role
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Literature Review Expert",
                content="""You are an expert in scientific literature review and synthesis. Your role is to:
                1. Analyze collections of scientific papers and identify key themes
                2. Synthesize findings across multiple studies
                3. Identify contradictions and gaps in the literature
                4. Create comprehensive literature reviews
                5. Connect findings to current research hypotheses
                
                Your reviews should be:
                - Systematic and comprehensive
                - Critical and analytical
                - Well-structured and coherent
                - Focused on identifying research opportunities
                - Written in academic style suitable for scientific papers"""
            )
        )
    
    async def synthesize_literature(self, 
                                  search_results: List[Any],
                                  hypotheses: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Synthesize literature findings into a comprehensive review
        
        Args:
            search_results: Results from literature search
            hypotheses: Research hypotheses to address
            
        Returns:
            Comprehensive literature synthesis
        """
        print("📚 Synthesizing literature review...")
        
        # Extract key themes from literature
        themes = await self._identify_literature_themes(search_results)
        
        # Create comprehensive synthesis
        synthesis = await self._create_literature_synthesis(search_results, hypotheses, themes)
        
        # Identify research gaps
        gaps = await self._identify_research_gaps(search_results, hypotheses, themes)
        
        # Generate introduction content
        introduction = await self._generate_introduction_content(synthesis, hypotheses)
        
        return {
            "themes": themes,
            "synthesis": synthesis,
            "research_gaps": gaps,
            "introduction_content": introduction,
            "quality_assessment": await self._assess_literature_quality(search_results)
        }
    
    async def _identify_literature_themes(self, search_results: List[Any]) -> List[LiteratureTheme]:
        """Identify key themes in the literature"""
        
        if not search_results:
            return []
        
        # Prepare literature summary for analysis
        papers_summary = []
        for i, paper in enumerate(search_results[:15]):  # Limit for prompt size
            if hasattr(paper, 'title') and hasattr(paper, 'abstract'):
                summary = f"Paper {i+1}:\nTitle: {paper.title}\nAbstract: {paper.abstract[:300]}...\n"
                papers_summary.append(summary)
        
        prompt = f"""
        Analyze these research papers and identify 5-7 key themes or research areas.
        For each theme, provide:
        1. Theme name
        2. Brief description
        3. Which papers support this theme
        4. Key findings related to this theme
        
        Papers to analyze:
        {chr(10).join(papers_summary)}
        
        Format your response as JSON with the following structure:
        {{
            "themes": [
                {{
                    "theme_name": "Theme Name",
                    "description": "Description of the theme",
                    "supporting_papers": ["Paper 1", "Paper 2"],
                    "key_findings": ["Finding 1", "Finding 2"]
                }}
            ]
        }}
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            themes_data = json.loads(response.msgs[0].content)
            themes = []
            
            for theme_data in themes_data.get("themes", []):
                theme = LiteratureTheme(
                    theme_name=theme_data.get("theme_name", ""),
                    description=theme_data.get("description", ""),
                    supporting_papers=theme_data.get("supporting_papers", []),
                    key_findings=theme_data.get("key_findings", [])
                )
                themes.append(theme)
            
            return themes
            
        except Exception as e:
            print(f"Error parsing themes: {e}")
            return []
    
    async def _create_literature_synthesis(self, 
                                         search_results: List[Any],
                                         hypotheses: List[Dict[str, str]],
                                         themes: List[LiteratureTheme]) -> str:
        """Create a comprehensive literature synthesis"""
        
        hypotheses_text = "\n".join([f"- {h['hypothesis']}" for h in hypotheses])
        themes_text = "\n".join([f"- {t.theme_name}: {t.description}" for t in themes])
        
        prompt = f"""
        Create a comprehensive literature synthesis that:
        1. Summarizes the current state of research in this field
        2. Discusses how existing findings relate to our hypotheses
        3. Highlights consensus and contradictions in the literature
        4. Establishes theoretical framework for our research
        
        Our Research Hypotheses:
        {hypotheses_text}
        
        Identified Themes:
        {themes_text}
        
        Write a 4-5 paragraph literature synthesis suitable for a scientific paper's introduction or literature review section.
        Use formal academic writing style and include logical transitions between ideas.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _identify_research_gaps(self, 
                                    search_results: List[Any],
                                    hypotheses: List[Dict[str, str]],
                                    themes: List[LiteratureTheme]) -> List[str]:
        """Identify research gaps in the literature"""
        
        hypotheses_text = "\n".join([f"- {h['hypothesis']}" for h in hypotheses])
        themes_text = "\n".join([f"- {t.theme_name}: {t.description}" for t in themes])
        
        prompt = f"""
        Based on the literature analysis and our research hypotheses, identify 5-7 specific research gaps that our study could address.
        
        Consider:
        - Methodological limitations in existing studies
        - Unexplored relationships between variables
        - Limited sample sizes or populations
        - Contradictory findings that need resolution
        - Theoretical gaps or underdeveloped frameworks
        
        Our Research Hypotheses:
        {hypotheses_text}
        
        Literature Themes:
        {themes_text}
        
        Provide gaps as a numbered list with brief explanations.
        """
        
        response = self.agent.step(prompt)
        gaps = [gap.strip() for gap in response.msgs[0].content.split('\n') 
               if gap.strip() and any(char.isdigit() for char in gap[:3])]
        
        return gaps
    
    async def _generate_introduction_content(self, 
                                           synthesis: str,
                                           hypotheses: List[Dict[str, str]]) -> str:
        """Generate introduction content for the paper"""
        
        hypotheses_text = "\n".join([f"- {h['hypothesis']}" for h in hypotheses])
        
        prompt = f"""
        Based on the literature synthesis, create an introduction section that:
        1. Establishes the context and importance of the research topic
        2. Reviews relevant literature and theory
        3. Identifies research gaps
        4. Presents our specific hypotheses
        5. Outlines the study's contribution
        
        Literature Synthesis:
        {synthesis}
        
        Our Hypotheses:
        {hypotheses_text}
        
        Write a comprehensive introduction (5-7 paragraphs) suitable for a scientific paper.
        End with a clear statement of the study's objectives and hypotheses.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _assess_literature_quality(self, search_results: List[Any]) -> Dict[str, Any]:
        """Assess the quality of the literature collection"""
        
        if not search_results:
            return {"quality_score": 0, "assessment": "No literature found"}
        
        # Prepare basic statistics
        total_papers = len(search_results)
        papers_with_abstracts = sum(1 for paper in search_results 
                                  if hasattr(paper, 'abstract') and paper.abstract)
        
        recent_papers = sum(1 for paper in search_results 
                          if hasattr(paper, 'year') and paper.year and paper.year >= 2020)
        
        high_citation_papers = sum(1 for paper in search_results 
                                 if hasattr(paper, 'citations') and paper.citations and paper.citations > 10)
        
        prompt = f"""
        Assess the quality of this literature collection for a scientific review:
        
        Statistics:
        - Total papers: {total_papers}
        - Papers with abstracts: {papers_with_abstracts}
        - Recent papers (2020+): {recent_papers}
        - High-citation papers (>10 citations): {high_citation_papers}
        
        Provide assessment covering:
        1. Overall quality score (1-10)
        2. Strengths of the collection
        3. Weaknesses or limitations
        4. Recommendations for improvement
        
        Format as JSON with keys: quality_score, strengths, weaknesses, recommendations.
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            assessment = json.loads(response.msgs[0].content)
        except:
            assessment = {
                "quality_score": 7,
                "assessment": response.msgs[0].content,
                "statistics": {
                    "total_papers": total_papers,
                    "papers_with_abstracts": papers_with_abstracts,
                    "recent_papers": recent_papers,
                    "high_citation_papers": high_citation_papers
                }
            }
        
        return assessment 