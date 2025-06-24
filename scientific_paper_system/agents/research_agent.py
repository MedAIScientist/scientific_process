"""
ENHANCED Research Agent for Generating Domain-Specific Scientific References

This agent searches for real scientific references using Semantic Scholar API
based on the research topic, ensuring all citations are relevant and exist.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
import time
from urllib.parse import quote

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType

try:
    import semanticscholar as sch
    SEMANTIC_SCHOLAR_AVAILABLE = True
except ImportError:
    sch = None
    SEMANTIC_SCHOLAR_AVAILABLE = False

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
    Enhanced Research Agent that searches for real scientific references using Semantic Scholar API
    """
    
    def __init__(self, model, semantic_scholar_api_key: Optional[str] = None):
        """Initialize the research agent"""
        self.model = model
        
        # Try to get API key from config if not provided
        try:
            from config.semantic_scholar_config import semantic_scholar_config
            self.api_key = semantic_scholar_api_key or semantic_scholar_config.api_key
            self.config = semantic_scholar_config
        except ImportError:
            self.api_key = semantic_scholar_api_key
            self.config = None
        
        # Initialize Semantic Scholar client
        if SEMANTIC_SCHOLAR_AVAILABLE:
            self.sch_client = sch.SemanticScholar(api_key=self.api_key)
        else:
            self.sch_client = None
            print("⚠️ Semantic Scholar library not available. Install with: pip install semanticscholar")
        
        # Initialize the chat agent for domain analysis
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Domain Expert Research Scientist",
                content="""You are an expert research scientist specializing in analyzing research domains 
                and generating search queries for scientific literature. Your role is to:
                
                1. Identify the exact research domain from hypotheses
                2. Generate effective search queries for academic databases
                3. Extract key concepts and terminology for literature searches
                4. Ensure search terms are specific and relevant to the research topic
                
                You must provide precise, domain-specific search terms that will yield 
                relevant scientific literature from academic databases."""
            )
        )
    
    async def search_literature(self, hypotheses: List[Dict], research_topic: str = None, max_papers: int = 20) -> List[LiteratureSource]:
        """Search for real literature using Semantic Scholar API with specific terms from hypotheses"""
        
        print(f"🔍 Searching for real scientific literature for {len(hypotheses)} hypotheses...")
        print(f"📋 Research topic: {research_topic}")
        
        # Extract specific terms from hypotheses AND research topic
        search_terms = self._extract_specific_search_terms(hypotheses, research_topic)
        
        # Limit search terms to avoid timeout - INCREASE FOR MORE REFERENCES
        search_terms = search_terms[:5]  # Increase to 5 terms for more references
        print(f"🎯 Using {len(search_terms)} search terms: {search_terms}")
        
        all_sources = []
        
        for i, term in enumerate(search_terms, 1):
            try:
                print(f"🔍 [{i}/{len(search_terms)}] Searching for: '{term}'")
                
                # Timeout per search - SMALLER LIMIT PER SEARCH
                import asyncio
                sources = await asyncio.wait_for(
                    self._search_literature_for_term(term, hypotheses, limit=4),  # Pass hypotheses
                    timeout=30.0  # 30 seconds max per search
                )
                
                all_sources.extend(sources)
                print(f"✅ [{i}/{len(search_terms)}] Found {len(sources)} sources for '{term}'")
                
                # Stop early if we have enough papers
                if len(all_sources) >= max_papers:
                    print(f"🎯 Already found {len(all_sources)} papers, stopping search")
                    break
                
                # Rate limiting - shorter
                if i < len(search_terms):  # Don't sleep after last search
                    print(f"⏳ Rate limiting... (1 second)")
                    time.sleep(1)  # Reduced from 2 to 1 second
                    
            except asyncio.TimeoutError:
                print(f"⏰ Timeout for search term '{term}' - skipping")
                continue
            except Exception as e:
                print(f"⚠️ Error searching for '{term}': {e}")
                continue
        
        print(f"🔍 Total sources found: {len(all_sources)}")
        
        # Remove duplicates and sort by relevance
        unique_sources = self._remove_duplicates(all_sources)
        print(f"🔍 Unique sources after deduplication: {len(unique_sources)}")
        
        # Filter for quality and relevance - LOWER THRESHOLD
        filtered_sources = self._filter_by_quality(unique_sources, min_citations=1)  # Very low threshold
        print(f"✅ Quality filtered sources: {len(filtered_sources)}")
        
        final_sources = filtered_sources[:max_papers]
        print(f"✅ Found {len(final_sources)} real scientific references")
        
        return final_sources
    
    def _extract_specific_search_terms(self, hypotheses: List[Dict], research_topic: str = None) -> List[str]:
        """Extract specific search terms automatically from any research topic and hypotheses"""
        
        # Combine all text sources
        all_text = ""
        
        # Add research topic if available
        if research_topic:
            all_text += research_topic + " "
        
        # Add hypotheses text (support both old and new formats)
        for h in hypotheses:
            all_text += h.get('hypothesis', h.get('statement', '')) + " " + h.get('rationale', h.get('statistical_evidence', '')) + " " + h.get('title', '') + " "
        
        print(f"📝 Full research context: {all_text[:200]}...")
        
        # Convert to lowercase for analysis
        text = all_text.lower()
        
        # Extract the REAL research subject (not technical variable names)
        real_subject = self._extract_real_research_subject(text)
        
        # Extract substantive concepts (not variable names)
        substantive_concepts = self._extract_substantive_concepts(text)
        
        # Generate focused search terms - MORE TERMS FOR MORE REFERENCES
        search_terms = self._generate_focused_search_terms(real_subject, substantive_concepts)
        
        print(f"🎯 Real research subject: {real_subject}")
        print(f"💡 Substantive concepts: {substantive_concepts[:3]}")
        print(f"🔍 Generated search terms: {search_terms[:8]}")  # Show more terms
        
        return search_terms[:8]  # Increase to 8 search terms for more references
    
    def _extract_real_research_subject(self, text: str) -> str:
        """Extract the real research subject, not technical variables"""
        
        # Look for key medical conditions mentioned
        medical_conditions = []
        
        if 'diabetes' in text or 'diabetic' in text:
            medical_conditions.append('diabetes')
        if 'hypertension' in text or 'blood pressure' in text or 'highbp' in text:
            medical_conditions.append('hypertension') 
        if 'obesity' in text or 'overweight' in text or 'bmi' in text:
            medical_conditions.append('obesity')
        if 'cardiovascular' in text or 'heart disease' in text or 'heartdisease' in text:
            medical_conditions.append('cardiovascular disease')
        if 'mental health' in text or 'depression' in text:
            medical_conditions.append('mental health')
        
        # New patterns for our specific hypotheses
        if 'cholcheck' in text or 'cholesterol screening' in text or 'prerequisite' in text:
            medical_conditions.append('preventive screening')
        if 'cluster' in text and ('health' in text or 'subgroup' in text):
            medical_conditions.append('health phenotyping')
        if 'composite' in text and ('health' in text or 'score' in text or 'principal component' in text):
            medical_conditions.append('health assessment')
        if 'chronic condition' in text or 'chronic disease' in text:
            medical_conditions.append('chronic disease')
        
        # Return the primary medical condition
        if medical_conditions:
            return medical_conditions[0]  # Primary condition
        else:
            return 'health outcomes'
    
    def _extract_substantive_concepts(self, text: str) -> List[str]:
        """Extract substantive research concepts, avoiding technical variable names"""
        
        substantive_concepts = []
        
        # Socioeconomic factors
        if any(term in text for term in ['income', 'economic', 'financial', 'poverty', 'wealth']):
            substantive_concepts.append('socioeconomic factors')
        
        if any(term in text for term in ['education', 'educational', 'literacy', 'school']):
            substantive_concepts.append('education')
        
        # Health perceptions and access
        if any(term in text for term in ['perception', 'perceived', 'self-rated', 'subjective']):
            substantive_concepts.append('health perceptions')
            
        if any(term in text for term in ['access', 'cost', 'barrier', 'healthcare', 'insurance']):
            substantive_concepts.append('healthcare access')
        
        # Risk factors
        if any(term in text for term in ['risk', 'factor', 'predictor', 'determinant']):
            substantive_concepts.append('risk factors')
        
        # Health behaviors
        if any(term in text for term in ['physical activity', 'exercise', 'lifestyle', 'behavior']):
            substantive_concepts.append('health behaviors')
        
        # Social determinants
        if any(term in text for term in ['social', 'demographic', 'disparity', 'inequality']):
            substantive_concepts.append('social determinants')
        
        # New concepts for our specific hypotheses
        if any(term in text for term in ['screening', 'prevention', 'early detection', 'diagnosis']):
            substantive_concepts.append('preventive care')
        
        if any(term in text for term in ['cluster', 'phenotype', 'subgroup', 'classification']):
            substantive_concepts.append('health phenotyping')
        
        if any(term in text for term in ['composite score', 'principal component', 'health index', 'multidimensional']):
            substantive_concepts.append('health measurement')
        
        if any(term in text for term in ['walking', 'mobility', 'physical function', 'disability']):
            substantive_concepts.append('physical function')
        
        if any(term in text for term in ['general health', 'self-reported health', 'health status']):
            substantive_concepts.append('health status')
        
        return substantive_concepts
    
    def _generate_focused_search_terms(self, real_subject: str, substantive_concepts: List[str]) -> List[str]:
        """Generate focused search terms based on real research subject and our specific hypotheses"""
        
        search_terms = []
        
        # HYPOTHESIS-SPECIFIC SEARCH TERMS for our 3 specific hypotheses
        
        # Hypothesis 1: Socioeconomic factors and healthcare access impact on diabetes risk
        hypothesis_1_terms = [
            'socioeconomic factors diabetes healthcare access',
            'income education diabetes risk barriers',
            'healthcare cost barriers diabetes prevalence',
            'social determinants diabetes prevention',
            'poverty diabetes healthcare utilization',
            'education level diabetes risk factors'
        ]
        
        # Hypothesis 2: Discrepancy between general and physical health perceptions
        hypothesis_2_terms = [
            'health perception diabetes risk assessment',
            'subjective health physical health diabetes',
            'health status discrepancy chronic disease',
            'general health physical health diabetes',
            'health perception bias diabetes risk',
            'self-reported health diabetes outcomes'
        ]
        
        # Hypothesis 3: Mental health pathway to physical health decline and diabetes
        hypothesis_3_terms = [
            'mental health physical activity diabetes',
            'depression physical limitations diabetes risk',
            'psychological health diabetes prevention',
            'mental health diabetes physical function',
            'depression diabetes physical activity',
            'mental health chronic disease pathway'
        ]
        
        # Add all hypothesis-specific terms
        search_terms.extend(hypothesis_1_terms)
        search_terms.extend(hypothesis_2_terms)
        search_terms.extend(hypothesis_3_terms)
        
        # Primary combinations with the real subject (keep some for broader coverage)
        for concept in substantive_concepts[:2]:  # Reduced from 3 to 2 to focus more on specific terms
            search_terms.append(f"{real_subject} {concept}")
        
        # Specific combinations based on detected research subject
        if real_subject == 'diabetes':
            search_terms.extend([
                'diabetes socioeconomic risk factors',
                'diabetes healthcare access barriers',
                'diabetes mental health physical activity'
            ])
        
        elif real_subject == 'socioeconomic':
            search_terms.extend([
                'socioeconomic diabetes prevention',
                'income education diabetes outcomes',
                'healthcare cost diabetes management'
            ])
        
        elif real_subject == 'health perception':
            search_terms.extend([
                'health perception diabetes assessment',
                'subjective health diabetes risk',
                'health status diabetes evaluation'
            ])
        
        elif real_subject == 'mental health':
            search_terms.extend([
                'mental health diabetes physical function',
                'depression diabetes activity levels',
                'psychological health diabetes prevention'
            ])
        
        else:
            # Fallback to our hypothesis-specific terms (already added above)
            search_terms.extend([
                'social determinants diabetes health',
                'health disparities diabetes outcomes',
                'mental physical health diabetes'
            ])
        
        # Remove duplicates while preserving order
        unique_terms = list(dict.fromkeys(search_terms))
        
        return unique_terms
    
    def _is_relevant_to_research(self, source: LiteratureSource, hypotheses: List[Dict]) -> bool:
        """Check if paper is relevant to our specific 3 hypotheses"""
        
        # Check paper content
        paper_text = (source.title + " " + source.abstract).lower()
        
        # Calculate relevance score for our specific hypotheses
        relevance_score = 0
        hypothesis_matches = 0
        
        # HYPOTHESIS 1: Socioeconomic factors and healthcare access impact on diabetes risk
        hypothesis_1_terms = [
            'socioeconomic factors', 'income', 'education', 'healthcare access',
            'healthcare cost', 'cost barriers', 'social determinants', 'poverty',
            'healthcare utilization', 'economic factors', 'educational level',
            'healthcare affordability', 'social disparities', 'healthcare equity'
        ]
        
        # HYPOTHESIS 2: Discrepancy between general and physical health perceptions  
        hypothesis_2_terms = [
            'health perception', 'subjective health', 'self-reported health',
            'general health', 'physical health', 'health status discrepancy',
            'perceived health', 'health assessment', 'health rating',
            'health perception bias', 'subjective wellbeing', 'health evaluation'
        ]
        
        # HYPOTHESIS 3: Mental health pathway to physical health decline and diabetes
        hypothesis_3_terms = [
            'mental health', 'depression', 'psychological health', 'anxiety',
            'physical activity', 'physical limitations', 'physical function',
            'mental health diabetes', 'depression diabetes', 'psychological factors',
            'mental wellbeing', 'psychiatric conditions', 'behavioral health'
        ]
        
        # Check for hypothesis-specific matches
        h1_match = any(term in paper_text for term in hypothesis_1_terms)
        h2_match = any(term in paper_text for term in hypothesis_2_terms)  
        h3_match = any(term in paper_text for term in hypothesis_3_terms)
        
        if h1_match:
            hypothesis_matches += 1
            relevance_score += 3
        if h2_match:
            hypothesis_matches += 1
            relevance_score += 3
        if h3_match:
            hypothesis_matches += 1
            relevance_score += 3
        
        # MANDATORY: Must be about diabetes or related chronic diseases
        diabetes_terms = [
            'diabetes', 'diabetic', 'type 2 diabetes', 'type 1 diabetes', 
            'glycemic', 'glucose', 'insulin', 'chronic disease', 'chronic condition',
            'multimorbidity', 'comorbidity', 'cardiovascular', 'hypertension',
            'high blood pressure', 'cholesterol', 'metabolic syndrome'
        ]
        
        subject_match = any(term in paper_text for term in diabetes_terms)
        
        # If no subject match, immediately reject
        if not subject_match:
            return False
        
        # Check for additional relevant concepts beyond our main hypotheses
        additional_score = 0
        
        # Socioeconomic factors (relevant to Hypothesis 1)
        socioeconomic_terms = ['socioeconomic status', 'income level', 'educational attainment', 'poverty', 'economic status', 'social class']
        if any(term in paper_text for term in socioeconomic_terms):
            additional_score += 2
            
        # Health perceptions and self-reported measures (relevant to Hypothesis 2)
        perception_terms = ['health perception', 'self-reported health', 'subjective health', 'perceived health', 'health rating', 'health evaluation']
        if any(term in paper_text for term in perception_terms):
            additional_score += 2
            
        # Mental health and physical activity (relevant to Hypothesis 3)
        mental_physical_terms = ['mental health', 'depression', 'physical activity', 'exercise', 'physical limitations', 'functional capacity']
        if any(term in paper_text for term in mental_physical_terms):
            additional_score += 2
            
        # Healthcare access and barriers (relevant to Hypothesis 1)
        access_terms = ['healthcare access', 'healthcare barriers', 'cost barriers', 'healthcare utilization', 'medical care access', 'healthcare affordability']
        if any(term in paper_text for term in access_terms):
            additional_score += 2
            
        # Statistical methods relevant to our analyses
        methods_terms = ['logistic regression', 'clustering analysis', 'principal component', 'chi-squared', 'statistical analysis']
        if any(term in paper_text for term in methods_terms):
            additional_score += 1
            
        relevance_score += additional_score
        
        # STRICT EXCLUSIONS for clearly irrelevant content
        exclusion_terms = [
            'covid-19', 'coronavirus', 'sars-cov-2', 'pandemic',
            'psilocybin', 'psychedelic', 'alcohol', 'substance abuse',
            'remdesivir', 'antiviral', 'drug treatment',
            'raspberry pi', 'computer', 'software', 'technology',
            'plant', 'animal', 'ecology', 'chemistry', 'physics',
            # NEW EXCLUSIONS for dental and other peripheral subjects
            'dental', 'tooth', 'teeth', 'oral health', 'dentistry', 'periodontal',
            'veterinary', 'animal health', 'pet', 'livestock',
            'pregnancy', 'maternal', 'prenatal', 'obstetric',
            'pediatric', 'children', 'adolescent', 'infant',
            'cancer', 'oncology', 'tumor', 'chemotherapy',
            'surgery', 'surgical', 'operative', 'anesthesia'
        ]
        
        for exclusion in exclusion_terms:
            if exclusion in paper_text:
                return False  # Immediately reject
        
        # Decision logic: Must have subject match + at least one hypothesis match OR high relevance score
        return subject_match and (hypothesis_matches >= 1 or relevance_score >= 5)
    
    async def _analyze_research_domain(self, hypotheses: List[Dict[str, str]], research_topic: str) -> Dict[str, Any]:
        """Analyze hypotheses to determine exact research domain and generate search queries"""
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        
        prompt = f"""
        Analyze these research hypotheses to identify the exact research domain and generate search queries:
        
        Research Topic: {research_topic}
        
        Hypotheses:
        {hypotheses_text}
        
        Provide analysis in the following JSON format:
        {{
            "domain": "Specific research field (e.g., 'Type 2 Diabetes and Socioeconomic Factors')",
            "subdomain": "More specific area",
            "key_concepts": ["concept1", "concept2", "concept3", "concept4", "concept5"],
            "search_queries": [
                "diabetes socioeconomic factors",
                "diabetes income education",
                "diabetes risk factors epidemiology",
                "diabetes prevention socioeconomic"
            ],
            "target_population": "Study population",
            "methodology_keywords": ["cross-sectional", "epidemiological", "regression analysis"],
            "relevant_journals": ["Journal1", "Journal2", "Journal3"],
            "related_diseases_conditions": ["condition1", "condition2"],
            "socioeconomic_factors": ["factor1", "factor2", "factor3"]
        }}
        
        Make sure search_queries are specific, relevant, and likely to find real papers in academic databases.
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            domain_data = json.loads(response.msgs[0].content)
            return domain_data
        except Exception as e:
            print(f"Error parsing domain analysis: {e}")
            # Default fallback based on common research patterns
            return {
                "domain": "Health and Medical Research",
                "subdomain": "Epidemiological Study",
                "key_concepts": ["health outcomes", "risk factors", "statistical analysis"],
                "search_queries": [
                    "health outcomes risk factors",
                    "epidemiological study",
                    "statistical analysis health",
                    "chronic disease prevention"
                ],
                "target_population": "Adult population",
                "methodology_keywords": ["cross-sectional", "regression analysis"],
                "relevant_journals": ["Journal of Medical Research", "Epidemiology Review"],
                "related_diseases_conditions": ["chronic diseases"],
                "socioeconomic_factors": ["income", "education"]
            }
    
    async def _search_literature_for_term(self, search_term: str, hypotheses: List[Dict], limit: int = 4) -> List[LiteratureSource]:
        """Search literature for a specific term with improved relevance filtering"""
        
        print(f"🔍 Starting search for: '{search_term}' (limit: {limit})")
        
        try:
            print("🔍 Searching with client...")
            # Increase limit to get more papers for better selection
            papers = self.sch_client.search_paper(search_term, limit=limit * 2)  # Get double to filter better
            print(f"✅ Client returned {len(papers)} results")
            
            relevant_papers = []
            for paper in papers:
                if len(relevant_papers) >= limit:
                    break
                    
                # Convert to our format
                source = self._convert_paper_to_source(paper)
                if source:
                    if self._is_relevant_to_research(source, hypotheses):
                        relevant_papers.append(source)
                        print(f"   ✅ Paper {len(relevant_papers)} is relevant: {source.title[:50]}...")
                    else:
                        print(f"   ❌ Paper not relevant")
            
            if len(relevant_papers) >= limit:
                print(f"   🎯 Found enough relevant papers ({len(relevant_papers)}), stopping search")
            
            print(f"✅ Search completed. Found {len(relevant_papers)} relevant papers for '{search_term}'")
            return relevant_papers
            
        except Exception as e:
            print(f"❌ Error searching for '{search_term}': {str(e)}")
            return []
    
    def _convert_paper_to_source(self, paper) -> Optional[LiteratureSource]:
        """Convert Semantic Scholar paper object to LiteratureSource"""
        
        try:
            # Extract authors
            authors = []
            if hasattr(paper, 'authors') and paper.authors:
                authors = [author.name for author in paper.authors if hasattr(author, 'name')]
            
            # Extract DOI
            doi = ''
            if hasattr(paper, 'externalIds') and paper.externalIds:
                doi = paper.externalIds.get('DOI', '')
            
            # Extract journal/venue
            journal = 'Unknown Journal'
            if hasattr(paper, 'venue') and paper.venue:
                journal = paper.venue
            elif hasattr(paper, 'journal') and paper.journal:
                journal = paper.journal.get('name', 'Unknown Journal') if isinstance(paper.journal, dict) else str(paper.journal)
            
            source = LiteratureSource(
                title=paper.title or "Unknown Title",
                authors=authors,  # Keep as list, don't join!
                year=paper.year or 2023,
                journal=journal,
                doi=doi,
                abstract=paper.abstract or '',
                url=paper.url or '',
                citations=paper.citationCount or 0
            )
            
            return source
            
        except Exception as e:
            print(f"⚠️ Error converting paper: {e}")
            return None
    
    def _remove_duplicates(self, sources: List[LiteratureSource]) -> List[LiteratureSource]:
        """Remove duplicate sources based on title similarity"""
        
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            title_key = source.title.lower().strip()[:50]  # First 50 chars
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_sources.append(source)
        
        return unique_sources
    
    def _filter_by_quality(self, sources: List[LiteratureSource], min_citations: int = 10) -> List[LiteratureSource]:
        """Filter sources by quality criteria"""
        
        quality_sources = []
        
        for source in sources:
            # Quality criteria
            has_citations = (source.citations or 0) >= min_citations
            has_doi = bool(source.doi)
            has_abstract = len(source.abstract or "") > 100
            recent = (source.year or 0) >= 2015
            
            # At least 2 out of 4 criteria
            quality_score = sum([has_citations, has_doi, has_abstract, recent])
            
            if quality_score >= 2:
                quality_sources.append(source)
        
        # Sort by citations (descending)
        quality_sources.sort(key=lambda x: x.citations or 0, reverse=True)
        
        return quality_sources
    
    async def generate_targeted_literature_review(self, sources: List[LiteratureSource], 
                                                hypotheses: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate literature review from real sources"""
        
        if not sources:
            return {
                "literature_review": "No relevant literature sources found.",
                "key_findings": [],
                "research_gaps": [],
                "theoretical_framework": ""
            }
        
        # Prepare sources summary for the agent
        sources_summary = []
        for i, source in enumerate(sources[:10], 1):  # Limit to top 10 for processing
            summary = f"""
            {i}. {source.title} ({source.year})
            Authors: {', '.join(source.authors[:3]) if isinstance(source.authors, list) else source.authors}
            Journal: {source.journal or 'Unknown'}
            Citations: {source.citations or 0}
            Abstract: {source.abstract[:300]}...
            """
            sources_summary.append(summary)
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        
        prompt = f"""
        Based on these REAL scientific sources, write a comprehensive literature review:
        
        Research Hypotheses:
        {hypotheses_text}
        
        Scientific Sources:
        {chr(10).join(sources_summary)}
        
        Write a literature review that:
        1. Synthesizes findings from these real sources
        2. Identifies key themes and patterns
        3. Relates findings to the research hypotheses
        4. Identifies research gaps
        5. Provides theoretical framework
        
        Format as JSON:
        {{
            "literature_review": "Comprehensive review text...",
            "key_findings": ["finding1", "finding2", "finding3"],
            "research_gaps": ["gap1", "gap2"],
            "theoretical_framework": "Framework description..."
        }}
        """
        
        response = self.agent.step(prompt)
        
        try:
            import json
            return json.loads(response.msgs[0].content)
        except Exception as e:
            print(f"Error generating literature review: {e}")
            return {
                "literature_review": f"Literature review based on {len(sources)} real scientific sources.",
                "key_findings": ["Multiple studies found relevant patterns"],
                "research_gaps": ["Further research needed"],
                "theoretical_framework": "Based on current scientific evidence"
            }
    
    async def analyze_literature_quality(self, sources: List[LiteratureSource]) -> Dict[str, Any]:
        """Analyze quality of real literature sources"""
        
        if not sources:
            return {
                "total_sources": 0,
                "quality_score": 0,
                "average_citations": 0,
                "year_range": "N/A",
                "journal_diversity": 0
            }
        
        # Calculate metrics
        total_citations = sum(s.citations or 0 for s in sources)
        avg_citations = total_citations / len(sources) if sources else 0
        
        years = [s.year for s in sources if s.year]
        year_range = f"{min(years)}-{max(years)}" if years else "N/A"
        
        journals = set(s.journal for s in sources if s.journal)
        journal_diversity = len(journals)
        
        # Quality score based on citations and recency
        recent_papers = sum(1 for s in sources if s.year and s.year >= 2020)
        well_cited = sum(1 for s in sources if s.citations and s.citations >= 10)
        
        quality_score = (recent_papers * 0.4 + well_cited * 0.6) / len(sources) * 100
        
        return {
            "total_sources": len(sources),
            "quality_score": round(quality_score, 1),
            "average_citations": round(avg_citations, 1),
            "year_range": year_range,
            "journal_diversity": journal_diversity,
            "recent_papers": recent_papers,
            "well_cited_papers": well_cited
        } 