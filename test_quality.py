"""Test quality scoring in the Conversation Crystal."""
import asyncio
import websockets
import json

async def test_quality_scoring():
    uri = 'ws://localhost:9600'
    async with websockets.connect(uri) as ws:
        # Identify first
        identify = {
            'id': 'id-1',
            'signal_type': 'identify',
            'source_cell': 'test-client',
            'timestamp': '2025-01-05T18:30:00Z',
            'payload': {'cell_type': 'external', 'cell_id': 'test-client'}
        }
        await ws.send(json.dumps(identify))
        resp = await ws.recv()
        print(f'Identity: {resp[:100]}...')
        
        # Embed an Ollama agent first
        embed_signal = {
            'id': 'embed-1',
            'signal_type': 'embed_agent',
            'source_cell': 'test-client',
            'timestamp': '2025-01-05T18:30:00Z',
            'payload': {
                'agent_type': 'ollama',
                'model_name': 'gemma3:1b',
                'tier': 'LOCAL_FAST',
                'agent_id': 'test-gemma-agent'
            }
        }
        await ws.send(json.dumps(embed_signal))
        resp = await ws.recv()
        print(f'Embed response: {json.loads(resp).get("signal_type")}')
        
        # Start a fractal process with a complex query
        signal = {
            'id': 'quality-test-1',
            'signal_type': 'start_fractal_process',
            'source_cell': 'test-client',
            'timestamp': '2025-01-05T18:30:00Z',
            'payload': {
                'query': 'Explain the concept of evolutionary selection pressure in biological systems. How does fitness determine which traits propagate to future generations? Consider examples from both natural selection and artificial selection in domesticated species.',
                'max_elevations': 1
            }
        }
        await ws.send(json.dumps(signal))
        
        # Wait for complete response
        while True:
            resp = await asyncio.wait_for(ws.recv(), timeout=120)
            data = json.loads(resp)
            signal_type = data.get('signal_type')
            
            if signal_type == 'fractal_complete':
                print('\n=== FRACTAL COMPLETE ===')
                payload = data.get('payload', {})
                print(f"Process ID: {payload.get('process_id')}")
                print(f"Final Tier: {payload.get('final_tier')}")
                print(f"Elevations: {payload.get('elevation_count')}")
                break
            elif signal_type == 'fractal_started':
                print('\n=== FRACTAL STARTED ===')
                payload = data.get('payload', {})
                print(f"Process ID: {payload.get('process_id')}")
                print(f"Tier: {payload.get('tier')}")
                print(f"Agent: {payload.get('agent_id')}")
                response = payload.get('response', '')
                print(f"Response preview: {response[:500]}...")
                break
            elif signal_type == 'thought_stream':
                chunk = data.get('payload', {}).get('chunk', '')
                print(chunk, end='', flush=True)
            elif signal_type == 'error' or signal_type == 'fractal_error':
                print(f'ERROR: {data}')
                break
            else:
                print(f'Received: {signal_type}')
        
        # Now check crystal stats and quality
        print('\n\n=== CHECKING CRYSTAL STATS ===')
        stats_signal = {
            'id': 'stats-1',
            'signal_type': 'get_crystal_stats',
            'source_cell': 'test-client',
            'timestamp': '2025-01-05T18:31:00Z',
            'payload': {}
        }
        await ws.send(json.dumps(stats_signal))
        
        resp = await asyncio.wait_for(ws.recv(), timeout=10)
        data = json.loads(resp)
        print(json.dumps(data.get('payload', {}), indent=2))

if __name__ == '__main__':
    asyncio.run(test_quality_scoring())
