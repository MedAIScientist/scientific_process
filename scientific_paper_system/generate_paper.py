#!/usr/bin/env python3
"""
🎯 PROFESSIONAL SCIENTIFIC PAPER GENERATOR

Main script to generate professional scientific papers in PDF format
from your CSV data and hypotheses.

USAGE: python3 generate_paper.py
"""

import sys
import os
import asyncio
import subprocess
from datetime import datetime

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

def print_banner():
    print("""
🎯 PROFESSIONAL SCIENTIFIC PAPER GENERATOR
==========================================
Creation of professional academic papers in PDF format
Organized project structure
""")

def check_project_structure():
    """Check project structure"""
    print("🔍 Checking project structure...")
    
    required_dirs = ['data', 'scripts', 'agents', 'outputs', 'tests']
    base_dir = os.path.dirname(__file__)
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - MISSING")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0

def check_data_files():
    """Check data files"""
    print("\n📊 Checking data files...")
    
    base_dir = os.path.dirname(__file__)
    required_files = [
        os.path.join('data', 'demo_data.csv'),
        os.path.join('data', 'hypothesis', 'structured_hypotheses_3_only.json'),
        os.path.join('data', 'hypothesis', 'statistical_analysis_report.json')
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print("\n⚠️ Missing data files!")
        print("Make sure you have:")
        print("- data/demo_data.csv (your data)")
        print("- data/hypothesis/structured_hypotheses_3_only.json (your hypotheses)")
        print("- data/hypothesis/statistical_analysis_report.json (statistical analysis)")
        return False
    
    return True

def check_dependencies():
    """Check dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = ['reportlab', 'markdown', 'pandas', 'camel', 'semanticscholar', 'requests']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'camel':
                import camel
            elif package == 'semanticscholar':
                import semanticscholar
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n💡 Install missing packages:")
        if 'camel' in missing:
            missing[missing.index('camel')] = 'camel-ai'
        print(f"pip3 install {' '.join(missing)}")
        return False
    
    # Check Semantic Scholar configuration
    try:
        from config.semantic_scholar_config import setup_semantic_scholar_api_key
        setup_semantic_scholar_api_key()
    except Exception as e:
        print(f"⚠️ Semantic Scholar config warning: {e}")
    
    return True

async def generate_paper():
    """Generate the scientific paper"""
    print("\n🚀 SCIENTIFIC PAPER GENERATION")
    print("=" * 50)
    
    try:
        # Change working directory to script folder
        base_dir = os.path.dirname(__file__)
        original_cwd = os.getcwd()
        os.chdir(base_dir)
        
        # Import and launch ENHANCED generator with verified references
        from scripts.enhanced_scientific_paper_generator import main as generate_main
        
        success = await generate_main()
        
        # Return to original directory
        os.chdir(original_cwd)
        
        return success
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def open_generated_pdf():
    """Open generated PDF"""
    print("\n📖 OPENING PDF...")
    
    base_dir = os.path.dirname(__file__)
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    # Find most recent PDF file
    import glob
    pdf_pattern = os.path.join(outputs_dir, "scientific_article_*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if pdf_files:
        latest_pdf = max(pdf_files, key=os.path.getctime)
        print(f"📄 Opening: {os.path.basename(latest_pdf)}")
        
        try:
            # Open according to OS
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", latest_pdf])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["start", latest_pdf], shell=True)
            else:  # Linux
                subprocess.run(["xdg-open", latest_pdf])
            
            print("✅ PDF opened in your default reader")
            return latest_pdf
            
        except Exception as e:
            print(f"⚠️ Cannot open automatically: {e}")
            print(f"📖 Open manually: {latest_pdf}")
            return latest_pdf
    else:
        print("❌ No PDF file found in outputs/")
        return None

def show_project_structure():
    """Display project structure"""
    print("\n📁 PROJECT STRUCTURE")
    print("=" * 40)
    
    structure = """
scientific_paper_system/
├── 📊 data/                    # Input data
│   ├── demo_data.csv           # Your research data
│   └── hypothesis/             # Hypothesis folder
│       ├── structured_hypotheses_3_only.json
│       └── statistical_analysis_report.json
│
├── 🚀 scripts/                 # Generation scripts
│   ├── generate_scientific_paper.py
│   └── scientific_paper_generator.py
│
├── 🤖 agents/                  # Specialized agents
│   ├── research_agent.py       
│   ├── data_analysis_agent.py
│   ├── literature_review_agent.py
│   ├── methodology_agent.py
│   ├── results_agent.py
│   ├── discussion_agent.py
│   ├── writing_agent.py
│   └── review_agent.py
│
├── 📄 outputs/                 # Generated files
│   ├── scientific_article_*.pdf    # FINAL PAPERS
│   ├── scientific_article_*.md
│   └── article_metadata_*.json
│
├── 🧪 tests/                   # Test scripts
│   ├── test_final_system.py
│   ├── validation_complete_semantic.py
│   ├── diagnostic_semantic_scholar.py
│   └── test_research_agent_fixed.py
│
├── ⚙️ config/                  # Configuration
│
└── 🎯 generate_paper.py        # MAIN SCRIPT
"""
    print(structure)

def print_summary(pdf_file):
    """Display final summary"""
    print("\n" + "=" * 60)
    print("🎉 PROFESSIONAL SCIENTIFIC PAPER CREATED!")
    print("=" * 60)
    
    if pdf_file:
        print(f"📄 PDF file: outputs/{os.path.basename(pdf_file)}")
        
        # File size
        try:
            size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
            print(f"💾 Size: {size_mb:.1f} MB")
        except:
            pass
    
    print("\n🎯 FEATURES:")
    print("✅ Professional academic format")
    print("✅ No technical mentions")
    print("✅ Real bibliographic references")
    print("✅ Organized structure")
    print("✅ Ready for publication")

async def main():
    """Main function"""
    print_banner()
    
    # Checks
    if not check_project_structure():
        print("\n❌ Incomplete project structure")
        return False
    
    if not check_data_files():
        return False
    
    if not check_dependencies():
        return False
    
    print("\n✅ All prerequisites are satisfied")
    print("🚀 Starting generation...")
    
    # Generation
    success = await generate_paper()
    
    if success:
        # Open PDF
        pdf_file = open_generated_pdf()
        
        # Summary
        print_summary(pdf_file)
        show_project_structure()
        
        print("\n🏆 MISSION ACCOMPLISHED!")
        print("Your professional scientific paper is ready!")
        
        return True
    else:
        print("\n❌ Generation failed")
        return False

if __name__ == "__main__":
    print("🎯 Scientific Paper Generator")
    print(f"⏰ Started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1) 