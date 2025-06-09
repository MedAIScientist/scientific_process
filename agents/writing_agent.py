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
            "keywords": await self._generate_keywords(abstract_data['hypotheses']),
            "introduction": introduction_data.get('introduction_content', ''),
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
        
        hypotheses_text = "; ".join([h['hypothesis'] for h in hypotheses[:2]])
        
        prompt = f"""
        Generate a concise, informative title for a scientific paper based on these hypotheses:
        
        {hypotheses_text}
        
        The title should:
        - Be 10-15 words
        - Clearly indicate the research focus
        - Be engaging but professional
        - Include key variables or concepts
        
        Provide just the title, no quotes or additional text.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content.strip()
    
    async def _write_abstract(self, abstract_data: Dict[str, Any]) -> str:
        """Write structured abstract"""
        
        hypotheses_text = "\n".join([f"- {h['hypothesis']}" for h in abstract_data['hypotheses']])
        key_findings = abstract_data.get('key_findings', [])
        
        prompt = f"""
        Write a structured abstract (250-300 words) with these sections:
        
        Background: Based on the summary: {abstract_data['summary']}
        Objectives: Based on hypotheses: {hypotheses_text}
        Methods: [Describe study design and analysis approach]
        Results: Key findings: {key_findings[:3] if key_findings else ['Statistical analysis conducted']}
        Conclusions: [Summarize main conclusions and implications]
        
        Write in formal academic style with clear structure.
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content
    
    async def _generate_keywords(self, hypotheses: List[Dict[str, str]]) -> List[str]:
        """Generate relevant keywords"""
        
        hypotheses_text = "; ".join([h['hypothesis'] for h in hypotheses])
        
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
        return keywords[:8]
    
    async def _format_references(self, literature_sources: List[Any]) -> str:
        """Format references in academic style"""
        
        if not literature_sources:
            return "References will be added based on literature review."
        
        formatted_refs = []
        for i, source in enumerate(literature_sources[:20]):  # Limit references
            if hasattr(source, 'title') and hasattr(source, 'authors'):
                authors_str = ", ".join(source.authors[:3]) if source.authors else "Unknown"
                year = f"({source.year})" if hasattr(source, 'year') and source.year else "(Year)"
                title = source.title if source.title else "Unknown Title"
                journal = source.journal if hasattr(source, 'journal') and source.journal else "Journal"
                
                ref = f"{authors_str} {year}. {title}. {journal}."
                formatted_refs.append(ref)
        
        return "\n".join(formatted_refs) if formatted_refs else "References to be formatted."
    
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
            "Figure 2: Correlation heatmap of main variables", 
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