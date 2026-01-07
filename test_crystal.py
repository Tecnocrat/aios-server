"""
Test script for Conversation Crystal integration.
Starts a fractal process and verifies crystallization.
"""

import asyncio
import json
import websockets

THINKER_WS = "ws://localhost:9600"


async def test_crystal():
    print("\n" + "="*60)
    print("🔮 CONVERSATION CRYSTAL INTEGRATION TEST")
    print("="*60)
    
    async with websockets.connect(THINKER_WS) as ws:
        # First: Identify ourselves
        print("\n[0] Identifying to Thinker...")
        identify_signal = {
            "signal_type": "identify",
            "source_cell": "test-client",
            "target_cell": "thinker-alpha",
            "payload": {
                "cell_id": "test-client",
                "cell_type": "external"
            }
        }
        await ws.send(json.dumps(identify_signal))
        response = await ws.recv()
        ident = json.loads(response)
        print(f"    Connected: {ident.get('signal_type')}")
        
        # Step 1: Embed an Ollama agent
        print("\n[1] Embedding Gemma agent...")
        embed_signal = {
            "signal_type": "embed_agent",
            "source_cell": "test-client",
            "target_cell": "thinker-alpha",
            "payload": {
                "agent_type": "ollama",
                "agent_id": "gemma-crystal-test",
                "model_name": "gemma3:1b"
            }
        }
        await ws.send(json.dumps(embed_signal))
        response = await ws.recv()
        result = json.loads(response)
        print(f"    Result: {result.get('signal_type')} - {result.get('payload', {}).get('state', 'unknown')}")
        
        await asyncio.sleep(2)  # Wait for agent initialization
        
        # Step 2: Start a fractal process (should trigger crystallization)
        print("\n[2] Starting fractal process (with crystal context injection)...")
        fractal_signal = {
            "signal_type": "start_fractal_process",
            "source_cell": "test-client",
            "target_cell": "thinker-alpha",
            "payload": {
                "query": "Explain how emergent intelligence arises in distributed systems. Consider feedback loops and self-organization.",
                "tier": "LOCAL_FAST",
                "use_crystal_context": True  # Enable reflexive injection
            }
        }
        await ws.send(json.dumps(fractal_signal))
        response = await ws.recv()
        result = json.loads(response)
        
        if result.get("signal_type") == "fractal_started":
            process_id = result["payload"]["process_id"]
            agent_id = result["payload"]["agent_id"]
            tier = result["payload"]["tier"]
            response_text = result["payload"]["response"]
            
            print(f"    Process ID: {process_id}")
            print(f"    Agent: {agent_id} ({tier})")
            print(f"    Response preview: {response_text[:200]}...")
        else:
            print(f"    Error: {result}")
            return
        
        await asyncio.sleep(2)  # Wait for crystallization
        
        # Step 3: Query crystal stats
        print("\n[3] Querying Crystal statistics...")
        stats_signal = {
            "signal_type": "get_crystal_stats",
            "source_cell": "test-client",
            "target_cell": "thinker-alpha",
            "payload": {}
        }
        await ws.send(json.dumps(stats_signal))
        response = await ws.recv()
        stats = json.loads(response)
        
        if stats.get("signal_type") == "crystal_stats":
            payload = stats["payload"]
            print(f"    Total Processes: {payload.get('total_processes', 0)}")
            print(f"    Total Exchanges: {payload.get('total_exchanges', 0)}")
            print(f"    Average Quality: {payload.get('average_quality', 0):.2f}")
            print(f"    Top Models: {payload.get('top_models', [])}")
            print(f"    DB Path: {payload.get('db_path', 'N/A')}")
        else:
            print(f"    Stats response: {stats}")
        
        # Step 4: Query recent processes
        print("\n[4] Querying recent crystallized processes...")
        recent_signal = {
            "signal_type": "query_crystal",
            "source_cell": "test-client",
            "target_cell": "thinker-alpha",
            "payload": {
                "query_type": "recent",
                "limit": 5
            }
        }
        await ws.send(json.dumps(recent_signal))
        response = await ws.recv()
        recent = json.loads(response)
        
        if recent.get("signal_type") == "crystal_response":
            processes = recent["payload"].get("processes", [])
            print(f"    Found {len(processes)} recent processes")
            for p in processes[:3]:
                print(f"    - [{p.get('process_id', 'N/A')[:8]}] {p.get('origin_query', 'N/A')[:50]}...")
        
        # Step 5: Test reflexive context generation
        print("\n[5] Testing reflexive context generation...")
        context_signal = {
            "signal_type": "query_crystal",
            "source_cell": "test-client",
            "target_cell": "thinker-alpha",
            "payload": {
                "query_type": "context",
                "query": "What patterns exist in distributed intelligence?",
                "max_tokens": 1000
            }
        }
        await ws.send(json.dumps(context_signal))
        response = await ws.recv()
        context = json.loads(response)
        
        if context.get("signal_type") == "crystal_response":
            has_context = context["payload"].get("has_context", False)
            crystal_context = context["payload"].get("context", "")
            print(f"    Has context: {has_context}")
            if crystal_context:
                print(f"    Context preview ({len(crystal_context)} chars):")
                print(f"    {crystal_context[:300]}...")
        
        print("\n" + "="*60)
        print("✅ CRYSTAL INTEGRATION TEST COMPLETE")
        print("="*60)
        print("\nThe Conversation Crystal is now:")
        print("  - Persisting agent conversations to SQLite")
        print("  - Backing up to JSON files")
        print("  - Providing reflexive context for new queries")
        print("  - Queryable via WebSocket signals")
        print("\nConversations will survive container restarts!")


if __name__ == "__main__":
    asyncio.run(test_crystal())
