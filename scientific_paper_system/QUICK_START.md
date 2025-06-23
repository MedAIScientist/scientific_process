# ⚡ Quick Start Guide

## 🚀 In 4 simple steps

### 1️⃣ Installation
```bash
git clone <your-repo>
cd scientific_paper_system
pip install -r requirements.txt
```

### 2️⃣ API Configuration
```bash
# Linux/Mac
export OPENAI_API_KEY="sk-your_key_here"

# Windows
set OPENAI_API_KEY=sk-your_key_here
```

### 3️⃣ Data Verification
```bash
# Check that these files exist:
ls data/demo_data.csv
ls data/hypothesis/structured_hypotheses_3_only.json
```

### 4️⃣ Generation
```bash
python generate_paper_complete_fixed.py
```

## ✅ Expected Output

```
🎯 COMPLETE SCIENTIFIC PAPER GENERATOR - WITH VERIFICATION
📊 Loading research data with verification...
✅ Dataset: 253680 participants, 22 variables
📚 VERIFIED LITERATURE REVIEW
✅ Found 20 verified references
🔬 ADVANCED DATA ANALYSIS
📈 RESULTS GENERATION
💬 DISCUSSION DEVELOPMENT
🛡️ VERIFICATION & CORRECTION OF ALL CONTENT
📝 FINAL PAPER COMPILATION
💾 Saving complete paper...
✅ Markdown saved: complete_enhanced_article_*.md
✅ PDF with images saved: complete_enhanced_article_*.pdf
🚀 Opening generated article: complete_enhanced_article_*.pdf
```

## 📄 Your paper will be in `outputs/`

- **Final PDF**: `complete_enhanced_article_YYYYMMDD_HHMMSS.pdf`
- **Markdown source**: `complete_enhanced_article_YYYYMMDD_HHMMSS.md`
- **Visualizations**: `figure_1_*.png`, `table_*.html`

## 🔧 Common Issues

**API key error?**
```bash
echo $OPENAI_API_KEY  # Should display your key
```

**Missing module?**
```bash
pip install camel-ai pandas reportlab
```

**No PDF?**
```bash
pip install reportlab beautifulsoup4
```

---

**⏱️ Total time: 5-7 minutes**  
**💰 Cost: $1-3 per paper**  
**📄 Result: 15-20 page PDF paper** 