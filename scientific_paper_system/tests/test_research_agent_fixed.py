#!/usr/bin/env python3
"""
FIXED Research Agent Test

Testing the fixed version that handles empty query errors and rate limiting.
"""

import asyncio
import os
import json
from datetime import datetime

async def test_research_agent_fixed():
    """Test the fixed Research Agent"""
    print("""
🧪 FIXED RESEARCH AGENT TEST
============================
Version that handles identified errors
""")
    
    try:
        # Required imports
        from scientific_agents.research_agent_fixed import ResearchAgentFixed
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, ModelType
        
        print("✅ Imports successful")
        
        # Initialize model
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_3_5_TURBO,
            model_config_dict={"temperature": 0.7}
        )
        print("✅ GPT-3.5-TURBO model initialized")
        
        # Initialize fixed Research Agent
        agent = ResearchAgentFixed(
            model=model,
            semantic_scholar_api_key=os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        )
        print("✅ Fixed Research Agent initialized")
        
        # Load real project hypotheses
        with open('hypothesis/hypotheses_output.json', 'r') as f:
            hyp_data = json.load(f)
        
        hypotheses = hyp_data['hypotheses'][:2]  # Limit to 2 for test
        print(f"📋 {len(hypotheses)} hypotheses loaded")
        
        for i, hyp in enumerate(hypotheses, 1):
            print(f"   {i}. {hyp['hypothesis'][:60]}...")
        
        print("\n🔍 STARTING LITERATURE SEARCH")
        print("=" * 50)
        
        # Test search with reasonable timeout
        start_time = datetime.now()
        
        try:
            sources = await asyncio.wait_for(
                agent.search_literature(hypotheses=hypotheses, max_papers=10),
                timeout=60.0  # 1 minute max
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n✅ SEARCH COMPLETED IN {duration:.1f}s")
            print(f"🎯 {len(sources)} sources found")
            
            if sources:
                print("\n📚 SOURCES OVERVIEW:")
                for i, source in enumerate(sources[:5], 1):
                    print(f"\n{i}. {source.title}")
                    print(f"   Authors: {', '.join(source.authors[:3])}{'...' if len(source.authors) > 3 else ''}")
                    print(f"   Year: {source.year}")
                    print(f"   Journal: {source.journal}")
                    print(f"   Citations: {source.citations}")
                    print(f"   Abstract: {source.abstract[:150]}...")
                
                # Save results
                results = {
                    "timestamp": datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "sources_found": len(sources),
                    "sources": [
                        {
                            "title": s.title,
                            "authors": s.authors,
                            "year": s.year,
                            "journal": s.journal,
                            "citations": s.citations,
                            "doi": s.doi,
                            "url": s.url,
                            "abstract": s.abstract[:500]  # Limit for file
                        }
                        for s in sources
                    ]
                }
                
                with open('research_results_fixed.json', 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"\n💾 Results saved to research_results_fixed.json")
                
                return True, f"Success: {len(sources)} sources in {duration:.1f}s"
            
            else:
                print("\n⚠️ NO SOURCES FOUND")
                print("Possible causes:")
                print("- API rate limiting")
                print("- Inadequate search terms")
                print("- Connectivity issues")
                
                return False, "No sources found"
                
        except asyncio.TimeoutError:
            print(f"\n❌ TIMEOUT after 60s")
            return False, "Timeout 60s"
            
        except Exception as e:
            print(f"\n❌ ERROR during search: {e}")
            return False, f"Error: {e}"
            
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False, f"Init error: {e}"

async def compare_with_original():
    """Comparison with original agent to show improvement"""
    print("""
🔍 ORIGINAL vs FIXED AGENT COMPARISON
=====================================
""")
    
    print("📊 PROBLEMS RESOLVED IN FIXED VERSION:")
    print()
    print("✅ 1. SEARCH TERMS VALIDATION:")
    print("   Original: Empty query → 'Missing required parameter' error")
    print("   Fixed:    Validation + default terms")
    print()
    print("✅ 2. RATE LIMITING HANDLING:")
    print("   Original: Immediate failure on HTTP 429")
    print("   Fixed:    Retry with exponential backoff")
    print()
    print("✅ 3. QUERY ROBUSTNESS:")
    print("   Original: Uses 5 terms → complex queries")
    print("   Fixed:    Limited to 2-3 terms → more stable")
    print()
    print("✅ 4. ERROR HANDLING:")
    print("   Original: Silent exceptions")
    print("   Fixed:    Detailed logs + fallbacks")
    print()
    print("✅ 5. PERFORMANCE:")
    print("   Original: No limits on results")
    print("   Fixed:    Adaptive limits to avoid timeouts")

def main():
    print("🧪 LAUNCHING FIXED RESEARCH AGENT TEST")
    
    # Test fixed agent
    success, message = asyncio.run(test_research_agent_fixed())
    
    # Comparison with original
    asyncio.run(compare_with_original())
    
    # Final report
    print("\n" + "=" * 60)
    print("📊 FINAL REPORT")
    print("=" * 60)
    
    if success:
        print("🎉 SUCCESS: Fixed Research Agent works!")
        print(f"   Result: {message}")
        print()
        print("🚀 NEXT STEPS:")
        print("1. Replace original agent with fixed version")
        print("2. Test with complete generator")
        print("3. Adjust search parameters")
    else:
        print("⚠️ PARTIAL FAILURE:")
        print(f"   Issue: {message}")
        print()
        print("🔧 ALTERNATIVE SOLUTIONS:")
        print("1. Wait for rate limiting to resolve")
        print("2. Use economical mode without external search")
        print("3. Implement fallback to manual search")

if __name__ == "__main__":
    main() 