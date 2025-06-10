# Scientific Paper Generator

**Generate professional scientific papers in PDF format from your CSV data and research hypotheses.**

## Quick Start

```bash
cd scientific_paper_system
python3 generate_paper.py
```

Your PDF paper will be created in the `outputs/` folder automatically.

## Requirements

### Data Files
- `data/demo_data.csv` - Your research data
- `data/hypothesis/hypotheses_output.json` - Your research hypotheses

### Dependencies
```bash
pip install camel-ai pandas numpy requests reportlab markdown 
```
### API Keys
```bash
export OPENAI_API_KEY="your_openai_key"
export SEMANTIC_SCHOLAR_API_KEY="your_key"   
```

## How It Works

1. **Load Data**: Reads your CSV data and hypotheses
2. **Literature Search**: Finds relevant academic papers (via Semantic Scholar)
3. **Generate Paper**: Creates a complete scientific paper with:
   - Abstract, Introduction, Methods, Results, Discussion
   - Real academic references
   - Professional PDF format

## Project Structure

```
scientific_paper_system/
├── generate_paper.py              # Main script - run this
├── data/
│   ├── demo_data.csv              # Your data
│   └── hypothesis/
│       └── hypotheses_output.json # Your hypotheses
├── scripts/                       # Core generation logic
├── agents/                        # Specialized AI agents
└── outputs/                       # Generated papers
```

## Output

- **Markdown**: `scientific_article_YYYYMMDD_HHMMSS.md`
- **PDF**: `scientific_article_YYYYMMDD_HHMMSS.pdf` ← **Your final paper**
- **Metadata**: `article_metadata_YYYYMMDD_HHMMSS.json`

## Performance

- **Time**: 2-5 minutes
- **Cost**: ~$0.50-1.00 per paper
- **Quality**: Publication-ready academic format

## Troubleshooting

**No PDF generated?**
```bash
pip install reportlab markdown
```

**No sources found?**
- Add Semantic Scholar API key
- Check internet connection

**Generation fails?**
- Verify OpenAI API key
- Check data files exist

## Customization

- Edit `agents/*.py` to modify paper sections
- Modify `scripts/scientific_paper_generator.py` for format changes
- Update hypotheses in `data/hypothesis/hypotheses_output.json`

---

**That's it!** Run `python3 generate_paper.py` and get your professional scientific paper. 
@yunus and ibrahim    