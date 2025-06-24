# 🎯 AI-Powered Scientific Paper Generator

**Complete system for automatically generating professional scientific papers with statistical verification and real academic references.**

## 🚀 Quick Start

```bash
# 1. Clone the project
git clone <your-repo>
cd scientific_paper_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
export OPENAI_API_KEY="your_openai_key"
export SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_key"  

# 4. Generate your paper
python generate_paper_complete_fixed.py
```

**✅ Your PDF paper will be created in the `outputs/` folder in 5-7 minutes**

## 📋 Prerequisites

### 🔑 Required API Keys

1. **OpenAI API Key** 
   - Create an account on [OpenAI Platform](https://platform.openai.com/)
   - Generate an API key
   - Add credit (~$5-10 recommended)

2. **Semantic Scholar API Key** 
   - Sign up on [Semantic Scholar API](https://www.semanticscholar.org/product/api)
   - Improves academic reference quality

### 💻 System Configuration

```bash
# Environment variables (Linux/Mac)
export OPENAI_API_KEY="sk-..."
export SEMANTIC_SCHOLAR_API_KEY="your_key"

# Environment variables (Windows)
set OPENAI_API_KEY=sk-...
set SEMANTIC_SCHOLAR_API_KEY=your_key
```

### 📦 Dependencies Installation

```bash
# Complete installation
pip install -r requirements.txt

# OR minimal installation
pip install camel-ai pandas numpy requests reportlab beautifulsoup4 matplotlib seaborn
```

## 📊 Required Data

### 1. CSV Data File
- **Location**: `data/demo_data.csv`
- **Format**: CSV with headers
- **Size**: Current system uses 253,680 participants, 22 variables
- **Example structure**:
```csv
Diabetes_binary,HighBP,HighChol,BMI,Age,Income,Education,...
0,1,0,25.5,45,75000,4,...
1,1,1,32.1,52,45000,2,...
```

### 2. Research Hypotheses
- **Location**: `data/hypothesis/structured_hypotheses_3_only.json`
- **Format**: Structured JSON
- **Content**: 3 research hypotheses

### 3. Statistical Analysis Report
- **Location**: `data/hypothesis/statistical_analysis_report.json`
- **Format**: JSON with verified statistics
- **Usage**: Prevents statistical hallucinations

## 🎯 Main Scripts

### 🏆 Main Script 
```bash
python generate_paper_complete_fixed.py
```
**Features:**
- ✅ 8 specialized AI agents
- ✅ Anti-hallucination verification
- ✅ Real academic references
- ✅ Professional visualizations
- ✅ PDF with figures and tables


### 🚀 Alternative Scripts

```bash
# Quick generation (2-3 minutes)
python quick_generate.py

# Verified generation only
python quick_verified_paper.py

# Interactive demonstration
python demo_for_professor.py
```

## 🏗️ System Architecture

```
scientific_paper_system/
├── 🎯 generate_paper_complete_fixed.py    # MAIN SCRIPT
├── 📊 data/
│   ├── demo_data.csv                      # Your research data
│   └── hypothesis/
│       ├── structured_hypotheses_3_only.json      # Hypotheses (3)
│       └── statistical_analysis_report.json       # Verified stats
├── 🤖 agents/                             # 8 Specialized AI Agents
│   ├── research_agent.py                 # Literature search
│   ├── literature_review_agent.py        # Literature review
│   ├── data_analysis_agent.py            # Statistical analysis
│   ├── methodology_agent.py              # Methodology
│   ├── results_agent.py                  # Results + visualizations
│   ├── discussion_agent.py               # Discussion
│   ├── writing_agent.py                  # Final writing
│   └── source_verification_agent.py      # Anti-hallucination verification
├── 📄 outputs/                           # Generated papers
│   ├── complete_enhanced_article_*.md    # Markdown format
│   ├── complete_enhanced_article_*.pdf   # Final PDF
│   ├── figure_1_*.png                    # Visualizations
│   └── table_*.html                      # Statistical tables
└── 🛠️ scripts/                          # Utility scripts
```

## 🎨 Advanced Features

### 🛡️ Anti-Hallucination Verification System
- **126 verified CSV statistics**
- **374 verified report statistics**
- **Real-time validation** of every numerical value
- **Verification rate**: 80-100%

### 📚 Real Academic References
- **20 papers** from Semantic Scholar
- **Numbered citations** [1], [2], [3]...
- **Verifiable DOIs** for each reference
- **Average 10 citations** per paper

### 📊 Professional Visualizations
- **Figure 1**: Prevalence charts
- **Table 1**: Descriptive statistics
- **Table 2**: Correlation matrix
- **Table 3**: Logistic regression results

### 📝 Complete Academic Structure
```
1. Introduction
   1.1 Context and Public Health Significance
   1.2 Literature Review - Key Findings
   1.3 Research Gaps and Theoretical Framework
   1.4 Study Objectives and Hypotheses

2. Methodology
   2.1 Study Foundation
   2.2 Participant Selection
   2.3 Data Collection Approach
   2.4 Measurement Framework
   2.5 Analytical Strategy

3. Results
   - Sample characteristics
   - Primary analysis
   - Secondary analyses
   - Hypothesis testing outcomes

4. Discussion
   - Results interpretation
   - Literature comparison
   - Clinical implications
   - Limitations

5. Conclusions

6. References (20 academic references)
```

## ⚡ Performance and Costs

| Metric | Value |
|--------|-------|
| **Generation time** | 5-7 minutes |
| **OpenAI cost** | $1-3 per paper |
| **Paper size** | 15-20 pages PDF |
| **References** | 20 academic papers |
| **Participants analyzed** | 253,680 |
| **Variables** | 22 |
| **Hypotheses** | 3 |

## 🔧 Troubleshooting

### ❌ Common Errors

**1. API key error**
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $SEMANTIC_SCHOLAR_API_KEY
```

**2. Missing modules**
```bash
pip install camel-ai pandas reportlab beautifulsoup4
```

**3. No PDF generated**
```bash
# Install PDF dependencies
pip install reportlab markdown
```

**4. Data error**
```bash
# Verify files exist
ls data/demo_data.csv
ls data/hypothesis/structured_hypotheses_3_only.json
```

**5. Generation timeout**
```bash
# Restart script - it will resume automatically
python generate_paper_complete_fixed.py
```

### 🐛 Debug Logs

The system displays detailed logs:
```
🎯 COMPLETE SCIENTIFIC PAPER GENERATOR - WITH VERIFICATION
📊 Loading research data with verification...
✅ Dataset: 253680 participants, 22 variables
📚 VERIFIED LITERATURE REVIEW
✅ Found 20 verified references
🔬 ADVANCED DATA ANALYSIS
🛡️ VERIFICATION & CORRECTION OF ALL CONTENT
📝 FINAL PAPER COMPILATION
💾 Saving complete paper...
✅ Markdown saved: complete_enhanced_article_*.md
✅ PDF with images saved: complete_enhanced_article_*.pdf
```

## 🎛️ Customization

### 📝 Modify Hypotheses
Edit `data/hypothesis/structured_hypotheses_3_only.json`:
```json
{
  "structured_hypotheses": {
    "hypotheses": [
      {
        "statement": "Your new hypothesis here",
        "variables": ["variable1", "variable2"],
        "expected_relationship": "positive/negative"
      }
    ]
  }
}
```

### 🤖 Modify Agents
- **Introduction**: `agents/literature_review_agent.py`
- **Methodology**: `agents/methodology_agent.py`
- **Results**: `agents/results_agent.py`
- **Discussion**: `agents/discussion_agent.py`

### 🎨 Modify PDF Format
Edit `generate_paper_complete_fixed.py` section `save_complete_paper()`.

## 📈 Output Examples

### 📄 Generated Files
```
outputs/
├── complete_enhanced_article_20250623_015232.md    # Markdown source
├── complete_enhanced_article_20250623_015232.pdf   # Final PDF
├── figure_1_prevalence_by_bmi.png                  # Chart
├── table_1_descriptive_statistics.html             # Statistics
├── table_2_correlations.html                       # Correlations
└── table_3_logistic_regression.html               # Regression
```

### 📊 Quality Statistics
```
📊 VERIFICATION RESULTS:
   📈 Total values found: 19
   ✅ Verified values: 15
   ❌ Hallucinated values: 0
   📈 Verification rate: 100%

🔗 DOI verification: 20/20
📚 References: 20 (ALL real from Semantic Scholar)
👥 Participants: 253,680
```

## 🤝 Support and Contribution

### 🆘 Getting Help
1. Check debug logs
2. Consult Troubleshooting section
3. Verify API keys
4. Ensure data files exist

### 🔄 Updates
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## 🏆 Summary

**This system generates publication-quality scientific papers with:**
- ✅ **Anti-hallucination verification** (100% verified statistics)
- ✅ **20 real academic references** with DOI
- ✅ **Professional visualizations** (figures + tables)
- ✅ **Complete PDF format** (15-20 pages)
- ✅ **Detailed numbered citations**
- ✅ **Standard academic structure**

**Single command to generate everything:**
```bash
python generate_paper_complete_fixed.py
```

**Developed by @yunus and @ibrahim** 🚀