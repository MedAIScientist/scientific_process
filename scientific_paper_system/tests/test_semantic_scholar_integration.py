#!/usr/bin/env python3
"""
Test script for Semantic Scholar integration
"""

import sys
import os
import asyncio

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

def test_semantic_scholar_availability():
    """Test if Semantic Scholar library is available"""
    print("🧪 Testing Semantic Scholar availability...")
    
    try:
        import semanticscholar as sch
        print("✅ Semantic Scholar library is available")
        return True
    except ImportError:
        print("❌ Semantic Scholar library not found")
        print("💡 Install with: pip install semanticscholar")
        return False

def test_api_configuration():
    """Test API configuration"""
    print("\n🧪 Testing API configuration...")
    
    try:
        from semantic_scholar_config import semantic_scholar_config, setup_semantic_scholar_api_key
        
        # Test configuration setup
        has_key = setup_semantic_scholar_api_key()
        
        print(f"API Key configured: {'✅ Yes' if has_key else '⚠️ No (will use rate-limited access)'}")
        print(f"Base URL: {semantic_scholar_config.base_url}")
        print(f"Rate limit: {semantic_scholar_config.requests_per_second} req/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

async def test_research_agent():
    """Test the research agent with real API calls"""
    print("\n🧪 Testing Research Agent...")
    
    try:
        from research_agent import ResearchAgent
        
        # Mock model for testing
        class MockModel:
            pass
        
        # Create research agent
        agent = ResearchAgent(MockModel())
        
        # Test hypotheses
        test_hypotheses = [
            {
                "hypothesis": "Higher income levels are associated with lower diabetes risk",
                "variables": ["income", "diabetes_risk"]
            },
            {
                "hypothesis": "Education level influences diabetes prevention behaviors",
                "variables": ["education", "prevention_behaviors"]
            }
        ]
        
        print("🔍 Testing literature search...")
        
        # Test literature search
        references = await agent.search_literature(
            hypotheses=test_hypotheses,
            research_topic="diabetes socioeconomic factors",
            max_papers=5
        )
        
        print(f"✅ Found {len(references)} references")
        
        # Display first few references
        for i, ref in enumerate(references[:3], 1):
            print(f"\n📄 Reference {i}:")
            print(f"   Title: {ref.title}")
            print(f"   Authors: {', '.join(ref.authors[:2])}")
            print(f"   Year: {ref.year}")
            print(f"   Journal: {ref.journal}")
            print(f"   Citations: {ref.citations}")
        
        # Test literature quality analysis
        if references:
            print("\n🔍 Testing literature quality analysis...")
            quality = await agent.analyze_literature_quality(references)
            print(f"✅ Quality analysis completed:")
            print(f"   Total sources: {quality['total_sources']}")
            print(f"   Quality score: {quality['quality_score']}")
            print(f"   Average citations: {quality['average_citations']}")
            print(f"   Year range: {quality['year_range']}")
        
        return len(references) > 0
        
    except Exception as e:
        print(f"❌ Research agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_direct_api_call():
    """Test direct API call to Semantic Scholar"""
    print("\n🧪 Testing direct API call...")
    
    try:
        import requests
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        params = {
            'query': 'diabetes socioeconomic factors',
            'fields': 'title,authors,year,citationCount',
            'limit': 3
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            papers = data.get('data', [])
            
            print(f"✅ API call successful - found {len(papers)} papers")
            
            for i, paper in enumerate(papers, 1):
                print(f"\n📄 Paper {i}:")
                print(f"   Title: {paper.get('title', 'N/A')}")
                print(f"   Year: {paper.get('year', 'N/A')}")
                print(f"   Citations: {paper.get('citationCount', 0)}")
            
            return True
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 SEMANTIC SCHOLAR INTEGRATION TESTS")
    print("=" * 50)
    
    tests = [
        ("Semantic Scholar Library", test_semantic_scholar_availability()),
        ("API Configuration", test_api_configuration()),
        ("Direct API Call", await test_direct_api_call()),
        ("Research Agent", await test_research_agent())
    ]
    
    print("\n📊 TEST RESULTS")
    print("=" * 30)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Semantic Scholar integration is working.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == len(tests)

if __name__ == "__main__":
    asyncio.run(main()) 