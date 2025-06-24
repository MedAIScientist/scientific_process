"""
Research Agent for Scientific Literature Search and Analysis

This agent specializes in finding and analyzing relevant scientific literature
using Semantic Scholar API, PubMed, and other academic databases.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
from urllib.parse import quote

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType

try:
    import semanticscholar as sch
except ImportError:
    sch = None

try:
    from Bio import Entrez
except ImportError:
    Entrez = None


@dataclass
class LiteratureSource:
    """Represents a scientific literature source"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    doi: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    citations: Optional[int] = None
    keywords: Optional[List[str]] = None
    relevance_score: Optional[float] = None


class ResearchAgent:
    """
    Specialized agent for scientific literature research and analysis
    """
    
    def __init__(self, model, semantic_scholar_api_key: Optional[str] = None):
        """
        Initialize the research agent
        
        Args:
            model: CAMEL model instance
            semantic_scholar_api_key: API key for Semantic Scholar
        """
        self.model = model
        self.semantic_scholar_api_key = semantic_scholar_api_key
        
        # Initialize the chat agent with specialized role
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Research Scientist",
                content="""You are an expert research scientist specializing in literature review and analysis. 
                Your role is to:
                1. Search for relevant scientific literature based on research hypotheses
                2. Analyze and summarize research papers
                3. Identify research gaps and opportunities
                4. Extract key findings and methodologies
                5. Assess the quality and relevance of sources
                
                You should provide comprehensive, accurate, and well-structured analyses 
                that help build the foundation for new research."""
            )
        )
        
        # Initialize external APIs
        if sch and semantic_scholar_api_key:
            try:
                self.semantic_scholar = sch.SemanticScholar(api_key=semantic_scholar_api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Semantic Scholar: {e}")
                self.semantic_scholar = None
        else:
            self.semantic_scholar = None
            
        # Configure Entrez for PubMed access
        if Entrez:
            Entrez.email = "research@example.com"  # Should be configurable
    
    async def search_literature(self, 
                              hypotheses: List[Dict[str, str]], 
                              pubmed_links: Optional[List[str]] = None,
                              max_papers: int = 20) -> List[LiteratureSource]:
        """
        Search for relevant literature based on hypotheses
        
        Args:
            hypotheses: List of research hypotheses
            pubmed_links: Optional list of specific PubMed links
            max_papers: Maximum number of papers to retrieve
            
        Returns:
            List of relevant literature sources
        """
        all_sources = []
        
        # Extract key terms from hypotheses
        search_terms = await self._extract_search_terms(hypotheses)
        
        # Search using different sources
        if self.semantic_scholar:
            semantic_results = await self._search_semantic_scholar(search_terms, max_papers // 2)
            all_sources.extend(semantic_results)
        
        if Entrez:
            pubmed_results = await self._search_pubmed(search_terms, max_papers // 2)
            all_sources.extend(pubmed_results)
        
        # Process specific PubMed links if provided
        if pubmed_links:
            specific_papers = await self._process_pubmed_links(pubmed_links)
            all_sources.extend(specific_papers)
        
        # Remove duplicates and rank by relevance
        unique_sources = await self._deduplicate_and_rank(all_sources, hypotheses)
        
        return unique_sources[:max_papers]
    
    async def _extract_search_terms(self, hypotheses: List[Dict[str, str]]) -> List[str]:
        """Extract relevant search terms from hypotheses"""
        
        prompt = f"""
        Based on the following research hypotheses, extract the most relevant scientific search terms 
        that would be useful for literature search. Focus on:
        - Key concepts and variables
        - Methodological terms
        - Domain-specific terminology
        - Related research areas
        
        Hypotheses:
        {chr(10).join([f"- {h['hypothesis']}" for h in hypotheses])}
        
        Provide a list of 10-15 search terms, each on a new line, without numbering.
        """
        
        response = self.agent.step(prompt)
        search_terms = [term.strip() for term in response.msgs[0].content.split('\n') 
                       if term.strip() and not term.strip().startswith('-')]
        
        return search_terms[:15]  # Limit to top 15 terms
    
    async def _search_semantic_scholar(self, search_terms: List[str], max_results: int) -> List[LiteratureSource]:
        """Search using Semantic Scholar API"""
        if not self.semantic_scholar:
            return []
        
        sources = []
        
        try:
            # Combine search terms for better results
            query = " ".join(search_terms[:5])  # Use top 5 terms
            
            results = self.semantic_scholar.search_paper(
                query, 
                limit=max_results,
                fields=['title', 'abstract', 'authors', 'year', 'venue', 'citationCount', 'url', 'externalIds']
            )
            
            for paper in results:
                if paper.abstract:  # Only include papers with abstracts
                    authors = [author.name for author in paper.authors] if paper.authors else []
                    
                    source = LiteratureSource(
                        title=paper.title or "Unknown Title",
                        authors=authors,
                        abstract=paper.abstract or "",
                        url=paper.url or "",
                        doi=paper.externalIds.get('DOI') if paper.externalIds else None,
                        year=paper.year,
                        journal=paper.venue,
                        citations=paper.citationCount
                    )
                    sources.append(source)
                    
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
        
        return sources
    
    async def _search_pubmed(self, search_terms: List[str], max_results: int) -> List[LiteratureSource]:
        """Search using PubMed/Entrez"""
        if not Entrez:
            return []
        
        sources = []
        
        try:
            # Combine search terms
            query = " AND ".join(search_terms[:3])  # Use top 3 terms
            
            # Search PubMed
            search_handle = Entrez.esearch(
                db="pubmed", 
                term=query, 
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            # Fetch details
            if search_results["IdList"]:
                fetch_handle = Entrez.efetch(
                    db="pubmed", 
                    id=search_results["IdList"], 
                    rettype="medline", 
                    retmode="text"
                )
                
                # Parse results (simplified parsing)
                records = fetch_handle.read().split('\n\n')
                fetch_handle.close()
                
                for record in records:
                    if 'TI  -' in record:  # Has title
                        source = self._parse_pubmed_record(record)
                        if source:
                            sources.append(source)
                            
        except Exception as e:
            print(f"Error searching PubMed: {e}")
        
        return sources
    
    def _parse_pubmed_record(self, record: str) -> Optional[LiteratureSource]:
        """Parse a PubMed record into a LiteratureSource"""
        try:
            lines = record.split('\n')
            title = ""
            authors = []
            abstract = ""
            year = None
            journal = ""
            
            for line in lines:
                if line.startswith('TI  - '):
                    title = line[6:].strip()
                elif line.startswith('AU  - '):
                    authors.append(line[6:].strip())
                elif line.startswith('AB  - '):
                    abstract = line[6:].strip()
                elif line.startswith('DP  - '):
                    year_match = re.search(r'(\d{4})', line)
                    if year_match:
                        year = int(year_match.group(1))
                elif line.startswith('TA  - '):
                    journal = line[6:].strip()
            
            if title and abstract:
                return LiteratureSource(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url="",  # Would need PMID to construct URL
                    year=year,
                    journal=journal
                )
        except Exception as e:
            print(f"Error parsing PubMed record: {e}")
        
        return None
    
    async def _process_pubmed_links(self, pubmed_links: List[str]) -> List[LiteratureSource]:
        """Process specific PubMed links provided by user"""
        sources = []
        
        for link in pubmed_links:
            try:
                # Extract PMID from link
                pmid_match = re.search(r'(\d+)/?$', link)
                if pmid_match:
                    pmid = pmid_match.group(1)
                    
                    # Fetch paper details
                    if Entrez:
                        fetch_handle = Entrez.efetch(
                            db="pubmed", 
                            id=pmid, 
                            rettype="medline", 
                            retmode="text"
                        )
                        record = fetch_handle.read()
                        fetch_handle.close()
                        
                        source = self._parse_pubmed_record(record)
                        if source:
                            source.url = link
                            sources.append(source)
                            
            except Exception as e:
                print(f"Error processing PubMed link {link}: {e}")
        
        return sources
    
    async def _deduplicate_and_rank(self, sources: List[LiteratureSource], 
                                  hypotheses: List[Dict[str, str]]) -> List[LiteratureSource]:
        """Remove duplicates and rank sources by relevance"""
        
        # Simple deduplication by title similarity
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            title_key = source.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_sources.append(source)
        
        # Rank by relevance using the agent
        if unique_sources:
            ranked_sources = await self._rank_by_relevance(unique_sources, hypotheses)
            return ranked_sources
        
        return unique_sources
    
    async def _rank_by_relevance(self, sources: List[LiteratureSource], 
                                hypotheses: List[Dict[str, str]]) -> List[LiteratureSource]:
        """Rank sources by relevance to hypotheses"""
        
        hypothesis_text = "\n".join([f"- {h['hypothesis']}" for h in hypotheses])
        
        for i, source in enumerate(sources):
            prompt = f"""
            Rate the relevance of this research paper to our hypotheses on a scale of 1-10.
            
            Our Hypotheses:
            {hypothesis_text}
            
            Paper Title: {source.title}
            Abstract: {source.abstract[:500]}...
            
            Consider:
            - Methodological relevance
            - Conceptual overlap
            - Similar research questions
            - Applicable findings
            
            Provide only a number from 1-10.
            """
            
            try:
                response = self.agent.step(prompt)
                score_text = response.msgs[0].content.strip()
                score = float(re.search(r'(\d+(?:\.\d+)?)', score_text).group(1))
                sources[i].relevance_score = score
            except:
                sources[i].relevance_score = 5.0  # Default score
        
        # Sort by relevance score (descending)
        sources.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        return sources
    
    async def analyze_literature_quality(self, sources: List[LiteratureSource]) -> Dict[str, Any]:
        """Analyze the quality and characteristics of the literature collection"""
        
        if not sources:
            return {"quality_score": 0, "analysis": "No sources to analyze"}
        
        # Prepare data for analysis
        source_summary = []
        for source in sources:
            summary = f"Title: {source.title}\n"
            summary += f"Authors: {', '.join(source.authors[:3])}{'...' if len(source.authors) > 3 else ''}\n"
            summary += f"Year: {source.year or 'Unknown'}\n"
            summary += f"Journal: {source.journal or 'Unknown'}\n"
            summary += f"Citations: {source.citations or 'Unknown'}\n"
            summary += f"Abstract: {source.abstract[:200]}...\n"
            source_summary.append(summary)
        
        prompt = f"""
        Analyze the quality and characteristics of this literature collection:
        
        {chr(10).join(source_summary[:10])}  # Limit to first 10 for analysis
        
        Provide an analysis covering:
        1. Overall quality assessment (1-10 scale)
        2. Temporal distribution (are sources recent?)
        3. Methodological diversity
        4. Geographic/institutional diversity
        5. Key themes and patterns
        6. Potential gaps or limitations
        
        Format as JSON with keys: quality_score, temporal_analysis, methodology_diversity, 
        key_themes, gaps_identified, recommendations.
        """
        
        response = self.agent.step(prompt)
        
        try:
            # Try to parse as JSON, fallback to text analysis
            import json
            analysis = json.loads(response.msgs[0].content)
        except:
            analysis = {
                "quality_score": 7,
                "analysis": response.msgs[0].content,
                "source_count": len(sources)
            }
        
        analysis["source_count"] = len(sources)
        return analysis 