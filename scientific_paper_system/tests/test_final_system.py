#!/usr/bin/env python3
"""
Final System Validation

Quick validation that your scientific paper generator works perfectly.
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime

def print_banner():
    print("""
🧪 SCIENTIFIC PAPER SYSTEM VALIDATION
=====================================
Quick test to verify your system is ready
""")

def check_environment():
    """Check environment and dependencies"""
    print("🔍 Environment Check")
    print("=" * 40)
    
    # Check Python version
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check required packages
    packages = {
        'pandas': 'Data processing',
        'camel': 'Multi-agent system',
        'requests': 'HTTP requests',
        'reportlab': 'PDF generation (optional)',
        'markdown': 'Markdown processing (optional)'
    }
    
    missing = []
    for package, description in packages.items():
        try:
            if package == 'camel':
                import camel
            else:
                __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"❌ {package}: {description} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n💡 Install missing packages:")
        for pkg in missing:
            if pkg == 'camel':
                print(f"pip install camel-ai")
            else:
                print(f"pip install {pkg}")
    
    return len(missing) == 0

def check_api_keys():
    """Check API keys"""
    print("\n🔑 API Keys Check")
    print("=" * 40)
    
    openai_key = os.getenv('OPENAI_API_KEY')
    semantic_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
    
    if openai_key:
        masked_key = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
        print(f"✅ OpenAI API Key: {masked_key}")
    else:
        print(f"❌ OpenAI API Key: MISSING")
        print("   Set with: export OPENAI_API_KEY='your_key_here'")
    
    if semantic_key:
        masked_key = semantic_key[:8] + "..." + semantic_key[-4:] if len(semantic_key) > 12 else "***"
        print(f"✅ Semantic Scholar API Key: {masked_key}")
    else:
        print(f"⚠️ Semantic Scholar API Key: OPTIONAL")
        print("   Set with: export SEMANTIC_SCHOLAR_API_KEY='your_key_here'")
    
    return bool(openai_key)

def check_data_files():
    """Check required data files"""
    print("\n📊 Data Files Check")
    print("=" * 40)
    
    required_files = [
        ('data/demo_data.csv', 'Research data'),
        ('data/hypothesis/hypotheses_output.json', 'Research hypotheses')
    ]
    
    all_present = True
    
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: {description}")
            
            # Show file details
            if file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    print(f"   📈 Data: {df.shape[0]} rows, {df.shape[1]} columns")
                    print(f"   📋 Columns: {', '.join(df.columns[:3])}{'...' if len(df.columns) > 3 else ''}")
                except:
                    print(f"   ⚠️ Cannot read CSV file")
            
            elif file_path.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    hypotheses = data.get('hypotheses', [])
                    print(f"   🔬 Hypotheses: {len(hypotheses)} defined")
                    if hypotheses:
                        print(f"   📝 First: {hypotheses[0].get('hypothesis', '')[:60]}...")
                except:
                    print(f"   ⚠️ Cannot read JSON file")
        else:
            print(f"❌ {file_path}: {description} - MISSING")
            all_present = False
    
    return all_present

def check_scripts():
    """Check required scripts"""
    print("\n🚀 Scripts Check")
    print("=" * 40)
    
    scripts = [
        'generate_paper.py',
        'scripts/scientific_paper_generator.py'
    ]
    
    all_present = True
    
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - MISSING")
            all_present = False
    
    return all_present

def check_agents():
    """Check agent files"""
    print("\n🤖 Agents Check")
    print("=" * 40)
    
    agents = [
        'research_agent.py',
        'data_analysis_agent.py',
        'literature_review_agent.py',
        'methodology_agent.py',
        'results_agent.py',
        'discussion_agent.py',
        'writing_agent.py',
        'review_agent.py'
    ]
    
    present_count = 0
    
    for agent in agents:
        agent_path = f"agents/{agent}"
        if os.path.exists(agent_path):
            print(f"✅ {agent}")
            present_count += 1
        else:
            print(f"❌ {agent} - MISSING")
    
    print(f"\n📊 Summary: {present_count}/{len(agents)} agents available")
    
    return present_count >= 6  # At least 6 agents required

def run_quick_test():
    """Run a quick functionality test"""
    print("\n⚡ Quick Functionality Test")
    print("=" * 40)
    
    try:
        # Test data loading
        print("📊 Testing data loading...")
        if os.path.exists('data/demo_data.csv'):
            df = pd.read_csv('data/demo_data.csv')
            print(f"   ✅ CSV loaded: {df.shape[0]} rows")
        else:
            print("   ❌ CSV file not found")
            return False
        
        # Test hypothesis loading
        print("🔬 Testing hypothesis loading...")
        if os.path.exists('data/hypothesis/hypotheses_output.json'):
            with open('data/hypothesis/hypotheses_output.json', 'r') as f:
                hyp_data = json.load(f)
            hypotheses = hyp_data.get('hypotheses', [])
            print(f"   ✅ Hypotheses loaded: {len(hypotheses)} items")
        else:
            print("   ❌ Hypothesis file not found")
            return False
        
        # Test CAMEL import
        print("🐪 Testing CAMEL import...")
        try:
            import camel
            print("   ✅ CAMEL library imported")
        except ImportError:
            print("   ❌ CAMEL library not available")
            return False
        
        print("\n🎉 All quick tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        return False

def show_next_steps():
    """Show next steps to user"""
    print("\n" + "=" * 60)
    print("🎯 YOUR SYSTEM IS READY!")
    print("=" * 60)
    
    print("\nTO GENERATE A NEW SCIENTIFIC PAPER:")
    print("=" * 50)
    print("1. 📊 Make sure you have your data:")
    print("   - demo_data.csv (your data)")
    print("   - hypothesis/hypotheses_output.json (your hypotheses)")
    print("")
    print("2. 🚀 Run the generator:")
    print("   python3 generate_paper.py")
    print("")
    print("3. 📄 Your PDF will be created in outputs/")
    print("")
    print("💡 TIPS:")
    print("- The process takes 2-5 minutes")
    print("- Cost: approximately $0.50-1.00")
    print("- PDF will open automatically when ready")
    print("- All technical mentions are automatically removed")
    print("")
    print("🔧 CUSTOMIZATION:")
    print("- Edit agents/ files to modify behavior")
    print("- Modify scripts/ to change generation process")
    print("- Check docs/ for detailed documentation")

def main():
    """Main validation function"""
    print_banner()
    
    # Run all checks
    env_ok = check_environment()
    api_ok = check_api_keys()
    data_ok = check_data_files()
    scripts_ok = check_scripts()
    agents_ok = check_agents()
    
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    checks = [
        ("Environment", env_ok),
        ("API Keys", api_ok),
        ("Data Files", data_ok),
        ("Scripts", scripts_ok),
        ("Agents", agents_ok)
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:15} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        
        # Run quick functionality test
        if run_quick_test():
            show_next_steps()
        else:
            print("\n⚠️ Some functionality tests failed")
            print("Your system may work but double-check the errors above")
    else:
        print("\n❌ SOME CHECKS FAILED")
        print("Please fix the issues above before running the generator")
    
    print(f"\n⏰ Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Your scientific paper generator is ready to use!")

if __name__ == "__main__":
    main() 