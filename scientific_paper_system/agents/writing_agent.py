"""Writing Agent for Scientific Paper Composition"""

from typing import Dict, List, Any
from camel.agents import ChatAgent
from camel.messages import BaseMessage
import json
import os


class WritingAgent:
    def __init__(self, model):
        self.model = model
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Scientific Writer",
                content="""You are an expert scientific writer specializing in:
                1. Assembling comprehensive research papers
                2. Ensuring coherent flow between sections
                3. Proper academic formatting and style
                4. Creating abstracts and titles
                5. Managing references and citations
                
                Your writing should be clear, precise, and follow academic standards."""
            )
        )
    
    async def write_complete_paper(self,
                                 abstract_data: Dict[str, Any],
                                 introduction_data: Dict[str, Any],
                                 methodology_data: Dict[str, Any],
                                 results_data: Dict[str, Any],
                                 discussion_data: Dict[str, Any],
                                 literature_sources: List[Any]) -> Dict[str, Any]:
        """Assemble complete scientific paper"""
        
        # Generate title
        title = await self._generate_title(abstract_data['hypotheses'])
        
        # Write abstract
        abstract = await self._write_abstract(abstract_data)
        
        # Assemble all sections
        complete_paper = {
            "title": title,
            "abstract": abstract,
            "keywords": await self._generate_keywords(abstract_data.get('hypotheses', [])),
            "introduction": introduction_data.get('introduction_section', introduction_data.get('detailed_literature_review', introduction_data.get('introduction_content', ''))),
            "methodology": methodology_data.get('methodology_text', ''),
            "results": results_data.get('results_text', ''),
            "discussion": discussion_data.get('discussion_text', ''),
            "conclusions": discussion_data.get('conclusions', ''),
            "references": await self._format_references(literature_sources),
            "acknowledgments": await self._write_acknowledgments(),
            "tables": self._create_tables_list(results_data),
            "figures": self._create_figures_list(results_data)
        }
        
        return complete_paper
    
    async def _generate_title(self, hypotheses: List[Dict[str, str]]) -> str:
        """Generate paper title"""
        
        if not hypotheses:
            return "Research Article: Statistical Analysis and Findings"
        
        # Extract key themes from hypotheses
        all_hypotheses_text = " ".join([h.get('hypothesis', '') for h in hypotheses]).lower()
        
        # Generate title based on detected themes
        if 'socioeconomic' in all_hypotheses_text and 'diabetes' in all_hypotheses_text:
            if 'mental health' in all_hypotheses_text:
                return "Socioeconomic Factors, Mental Health, and Diabetes Risk: A Comprehensive Analysis"
            elif 'healthcare access' in all_hypotheses_text or 'cost' in all_hypotheses_text:
                return "Socioeconomic Barriers to Healthcare and Diabetes Risk: A Population-Based Study"
            else:
                return "Socioeconomic Determinants of Type 2 Diabetes Risk: A Cross-Sectional Analysis"
        elif 'health perception' in all_hypotheses_text or 'physical health' in all_hypotheses_text:
            return "Health Perceptions and Diabetes Risk Assessment: A Statistical Analysis"
        elif 'mental health' in all_hypotheses_text and 'physical activity' in all_hypotheses_text:
            return "Mental Health, Physical Activity, and Chronic Disease Risk: A Pathway Analysis"
        else:
            # Fallback to AI generation for other cases
            hypotheses_text = "; ".join([h.get('hypothesis', 'Research hypothesis') for h in hypotheses[:2]])
            
            prompt = f"""
            Generate a concise, informative title for a scientific paper based on these hypotheses:
            
            {hypotheses_text}
            
            The title should:
            - Be 10-15 words
            - Clearly indicate the research focus
            - Be engaging but professional
            - Include key variables or concepts
            - Focus on health outcomes, risk factors, or social determinants
            
            Provide just the title, no quotes or additional text.
            """
            
            response = self.agent.step(prompt)
            return response.msgs[0].content.strip()
    
    async def _write_abstract(self, abstract_data: Dict[str, Any]) -> str:
        """Write structured abstract"""
        
        hypotheses = abstract_data.get('hypotheses', [])
        if not hypotheses:
            hypotheses_text = "Research hypotheses to be determined"
        else:
            hypotheses_text = "\n".join([f"- {h.get('hypothesis', 'Research hypothesis')}" for h in hypotheses])
        
        key_findings = abstract_data.get('key_findings', [])
        summary = abstract_data.get('summary', 'Data analysis conducted')
        
        prompt = f"""
        Write a structured abstract (250-300 words) using short, focused paragraphs:
        
        Background: (2-3 sentences): Based on the summary: {summary}
        
        Objectives: (2-3 sentences): Based on hypotheses: {hypotheses_text}
        
        Methods: (2-3 sentences): Describe study design and analysis approach concisely
        
        Results: (2-3 sentences): Key findings: {key_findings[:3] if key_findings else ['Statistical analysis conducted']}
        
        Conclusions: (2-3 sentences): Summarize main conclusions and implications
        
        Requirements:
        - Each section should be 2-3 sentences maximum
        - Write in formal academic style with clear structure
        - Ensure smooth flow between sections
        - Total word count: 250-300 words
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _generate_keywords(self, hypotheses: List[Dict[str, str]]) -> List[str]:
        """Generate relevant keywords"""
        
        if not hypotheses:
            return ["research", "statistical analysis", "data analysis", "empirical study"]
        
        hypotheses_text = "; ".join([h.get('hypothesis', 'research hypothesis') for h in hypotheses])
        
        prompt = f"""
        Generate 5-8 relevant keywords for a research paper with these hypotheses:
        
        {hypotheses_text}
        
        Keywords should be:
        - Specific to the research domain
        - Useful for database searching
        - Mix of broad and specific terms
        
        Provide as comma-separated list.
        """
        
        response = self.agent.step(prompt)
        keywords = [kw.strip() for kw in response.msgs[0].content.split(',')]
        return keywords[:8] if keywords else ["research", "analysis", "statistics"]
    
    async def _format_references(self, literature_sources: List[Any]) -> str:
        """Format references in academic style - supports Semantic Scholar objects"""
        
        if not literature_sources:
            return "References will be added based on literature review."
        
        formatted_refs = []
        for i, source in enumerate(literature_sources[:20]):  # Limit references
            try:
                # Handle Semantic Scholar objects and dict formats
                if isinstance(source, dict):
                    # Dict format from API
                    authors = source.get('authors', [])
                    if isinstance(authors, list) and authors:
                        if isinstance(authors[0], dict):
                            authors_str = ", ".join([a.get('name', 'Unknown') for a in authors[:3]])
                        else:
                            authors_str = ", ".join(authors[:3])
                    else:
                        authors_str = "Unknown"
                    
                    year = source.get('year', 'Year')
                    title = source.get('title', 'Unknown Title')
                    journal = source.get('journal', {})
                    if isinstance(journal, dict):
                        journal_name = journal.get('name', 'Journal')
                    else:
                        journal_name = str(journal) if journal else 'Journal'
                    
                    # Add DOI if available
                    doi = source.get('doi', '')
                    doi_str = f" DOI: {doi}" if doi else ""
                    
                    ref = f"{authors_str} ({year}). {title}. {journal_name}.{doi_str}"
                    formatted_refs.append(ref)
                    
                elif hasattr(source, 'title'):
                    # Object format from LiteratureSource
                    authors = getattr(source, 'authors', [])
                    if isinstance(authors, list) and authors:
                        # Authors is already a list of strings
                        authors_str = ", ".join(authors[:3])
                        if len(authors) > 3:
                            authors_str += " et al."
                    elif isinstance(authors, str):
                        # Handle case where authors is a string (shouldn't happen now)
                        authors_str = authors
                    else:
                        authors_str = "Unknown"
                    
                    year = getattr(source, 'year', 'Year')
                    title = getattr(source, 'title', 'Unknown Title')
                    
                    # Handle journal - can be object or string
                    journal = getattr(source, 'journal', None)
                    if hasattr(journal, 'name'):
                        journal_name = journal.name
                    elif isinstance(journal, dict):
                        journal_name = journal.get('name', 'Journal')
                    else:
                        journal_name = str(journal) if journal else 'Journal'
                    
                    # Add DOI if available
                    doi = getattr(source, 'doi', '')
                    doi_str = f" DOI: {doi}" if doi else ""
                    
                    ref = f"{authors_str} ({year}). {title}. {journal_name}.{doi_str}"
                    formatted_refs.append(ref)
                
            except Exception as e:
                # Fallback for problematic sources
                print(f"Warning: Could not format reference {i+1}: {e}")
                formatted_refs.append(f"[{i+1}] Reference formatting error - source available")
        
        if formatted_refs:
            return "\n".join(formatted_refs)
        else:
            return "References to be formatted."
    
    async def _write_acknowledgments(self) -> str:
        """Write acknowledgments section"""
        return """The authors thank all participants and contributors to this research. 
        We acknowledge the use of computational resources and statistical software packages 
        that made this analysis possible."""
    
    def _create_tables_list(self, results_data: Dict[str, Any]) -> List[str]:
        """Create list of tables to be included"""
        return [
            "Table 1: Descriptive statistics for all variables",
            "Table 2: Statistical test results summary",
            "Table 3: Correlation matrix for key variables",
            "Table 4: Hypothesis testing outcomes"
        ]
    
    def _create_figures_list(self, results_data: Dict[str, Any]) -> List[str]:
        """Create list of figures to be included"""
        return [
            "Figure 1: Data distribution and outlier analysis",
 
            "Figure 3: Statistical test results visualization",
            "Figure 4: Effect size comparison across hypotheses"
        ]
    
    def export_paper(self, paper: Dict[str, Any], 
                    output_format: str = "markdown",
                    output_path: str = "generated_paper") -> str:
        """Export paper to specified format"""
        
        if output_format == "markdown":
            return self._export_markdown(paper, output_path)
        elif output_format == "latex":
            return self._export_latex(paper, output_path)
        elif output_format == "json":
            return self._export_json(paper, output_path)
        else:
            return self._export_text(paper, output_path)
    
    def _export_markdown(self, paper: Dict[str, Any], output_path: str) -> str:
        """Export as Markdown"""
        
        filepath = f"{output_path}.md"
        
        markdown_content = f"""# {paper.get('title', 'Research Paper')}

## Abstract
{paper.get('abstract', '')}

**Keywords:** {', '.join(paper.get('keywords', []))}

## Introduction
{paper.get('introduction', '')}

## Methodology
{paper.get('methodology', '')}

## Results
{paper.get('results', '')}

## Discussion
{paper.get('discussion', '')}

## Conclusions
{paper.get('conclusions', '')}

## Tables
{chr(10).join([f"- {table}" for table in paper.get('tables', [])])}

## Figures
{chr(10).join([f"- {figure}" for figure in paper.get('figures', [])])}

## References
{paper.get('references', '')}

## Acknowledgments
{paper.get('acknowledgments', '')}
"""
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            return filepath
        except Exception as e:
            print(f"Error exporting markdown: {e}")
            return ""
    
    def _export_json(self, paper: Dict[str, Any], output_path: str) -> str:
        """Export as JSON"""
        
        filepath = f"{output_path}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(paper, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            print(f"Error exporting JSON: {e}")
            return ""
    
    def _export_text(self, paper: Dict[str, Any], output_path: str) -> str:
        """Export as plain text"""
        
        filepath = f"{output_path}.txt"
        
        text_content = f"""{paper.get('title', 'Research Paper')}

ABSTRACT
{paper.get('abstract', '')}

Keywords: {', '.join(paper.get('keywords', []))}

INTRODUCTION
{paper.get('introduction', '')}

METHODOLOGY
{paper.get('methodology', '')}

RESULTS
{paper.get('results', '')}

DISCUSSION
{paper.get('discussion', '')}

CONCLUSIONS
{paper.get('conclusions', '')}

REFERENCES
{paper.get('references', '')}

ACKNOWLEDGMENTS
{paper.get('acknowledgments', '')}
"""
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)
            return filepath
        except Exception as e:
            print(f"Error exporting text: {e}")
            return ""
    
    def _export_latex(self, paper: Dict[str, Any], output_path: str) -> str:
        """Export as LaTeX (basic template)"""
        
        filepath = f"{output_path}.tex"
        
        latex_content = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}

\\title{{{paper.get('title', 'Research Paper')}}}
\\author{{Generated by CAMEL Scientific Agents}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{paper.get('abstract', '')}
\\end{{abstract}}

\\section{{Introduction}}
{paper.get('introduction', '')}

\\section{{Methodology}}
{paper.get('methodology', '')}

\\section{{Results}}
{paper.get('results', '')}

\\section{{Discussion}}
{paper.get('discussion', '')}

\\section{{Conclusions}}
{paper.get('conclusions', '')}

\\section{{References}}
{paper.get('references', '')}

\\end{{document}}
"""
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            return filepath
        except Exception as e:
            print(f"Error exporting LaTeX: {e}")
            return ""

    async def compile_paper(self, components: Dict[str, Any], literature_sources: List[Any]) -> Dict[str, Any]:
        """Compile all paper components into a complete scientific paper"""
        
        # Extract components
        abstract_data = components.get('abstract_data', {})
        introduction_data = components.get('literature_synthesis', {})
        methodology_data = components.get('methodology', {})
        results_data = components.get('results', {})
        discussion_data = components.get('discussion', {})
        
        # Use the existing write_complete_paper method
        complete_paper = await self.write_complete_paper(
            abstract_data=abstract_data,
            introduction_data=introduction_data,
            methodology_data=methodology_data,
            results_data=results_data,
            discussion_data=discussion_data,
            literature_sources=literature_sources
        )
        
        return complete_paper 