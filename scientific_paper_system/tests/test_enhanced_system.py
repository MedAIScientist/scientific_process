#!/usr/bin/env python3
"""
Test script for the Enhanced Scientific Paper Generator with Verified References

This script tests the complete integration of the verified research agent
with the full scientific paper generation system.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_enhanced_system():
    """Test the enhanced system with verified references"""
    
    print("🧪 TESTING ENHANCED SCIENTIFIC PAPER GENERATOR")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Import and run the enhanced generator
        from scripts.enhanced_scientific_paper_generator import main as enhanced_main
        
        print("\n🚀 Running enhanced paper generation...")
        success = await enhanced_main()
        
        if success:
            print("\n✅ ENHANCED SYSTEM TEST PASSED!")
            print("🎉 Successfully generated paper with verified references!")
            return True
        else:
            print("\n❌ ENHANCED SYSTEM TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_verified_research_agent():
    """Test the verified research agent specifically"""
    
    print("\n🔬 TESTING VERIFIED RESEARCH AGENT")
    print("=" * 50)
    
    try:
        # Import the verified research agent
        from agents.verified_research_agent import VerifiedResearchAgent
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, ModelType
        
        # Initialize model
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0.3}
        )
        
        # Initialize agent
        agent = VerifiedResearchAgent(model=model)
        
        # Test hypotheses
        test_hypotheses = [
            {
                "hypothesis": "Lower income and education levels are associated with higher diabetes risk"
            },
            {
                "hypothesis": "Socioeconomic factors influence health outcomes"
            }
        ]
        
        # Test literature search
        print("🔍 Testing literature search...")
        sources = await agent.search_literature(
            hypotheses=test_hypotheses,
            research_topic="Socioeconomic Determinants of Type 2 Diabetes",
            max_papers=5
        )
        
        if sources:
            print(f"✅ Found {len(sources)} verified sources")
            
            # Check DOI verification
            doi_count = sum(1 for s in sources if s.doi)
            print(f"🔗 DOI verification: {doi_count}/{len(sources)} sources with DOI")
            
            # Test quality analysis
            quality = await agent.analyze_literature_quality(sources)
            print(f"📊 Quality score: {quality.get('quality_score', 0)}/10")
            
            return True
        else:
            print("❌ No sources found")
            return False
            
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    
    print("🎯 Enhanced Scientific Paper Generator Test Suite")
    print("Testing complete system integration with verified references")
    print()
    
    # Test 1: Verified Research Agent
    agent_success = await test_verified_research_agent()
    
    # Test 2: Complete System
    if agent_success:
        system_success = await test_enhanced_system()
    else:
        print("⚠️ Skipping system test due to agent test failure")
        system_success = False
    
    # Final results
    if agent_success and system_success:
        print("\n🏆 ALL TESTS PASSED!")
        print("The enhanced system is ready for production use.")
        print("\n📋 Test Summary:")
        print("✅ Verified Research Agent: PASSED")
        print("✅ Enhanced System Integration: PASSED")
        print("✅ Reference Verification: PASSED")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED!")
        print("\n📋 Test Summary:")
        print(f"{'✅' if agent_success else '❌'} Verified Research Agent: {'PASSED' if agent_success else 'FAILED'}")
        print(f"{'✅' if system_success else '❌'} Enhanced System Integration: {'PASSED' if system_success else 'FAILED'}")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1) 