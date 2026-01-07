"""Test fractal process with thought elevation in Thinker cell."""
import asyncio
import websockets
import json
import uuid


async def main():
    uri = 'ws://localhost:9600'
    async with websockets.connect(uri, ping_interval=20, ping_timeout=120) as ws:
        # 1. Identify
        identify = {
            'signal_type': 'identify',
            'source_cell': 'fractal-test',
            'target_cell': 'thinker-alpha',
            'payload': {'cell_type': 'external'},
            'id': uuid.uuid4().hex
        }
        await ws.send(json.dumps(identify))
        print('✓ Identified')
        ack = await ws.recv()
        
        # 2. Embed LOCAL_FAST agent (Gemma)
        embed_gemma = {
            'signal_type': 'embed_agent',
            'source_cell': 'fractal-test',
            'target_cell': 'thinker-alpha',
            'payload': {
                'agent_id': 'gemma-fast',
                'agent_type': 'ollama',
                'model_name': 'gemma3:1b'
            },
            'id': uuid.uuid4().hex
        }
        await ws.send(json.dumps(embed_gemma))
        resp = await asyncio.wait_for(ws.recv(), timeout=20)
        print(f'✓ Gemma (LOCAL_FAST): {json.loads(resp).get("signal_type")}')
        
        # 3. Embed LOCAL_REASONING agent (Mistral)
        embed_mistral = {
            'signal_type': 'embed_agent',
            'source_cell': 'fractal-test',
            'target_cell': 'thinker-alpha',
            'payload': {
                'agent_id': 'mistral-reason',
                'agent_type': 'ollama',
                'model_name': 'mistral:7b'
            },
            'id': uuid.uuid4().hex
        }
        await ws.send(json.dumps(embed_mistral))
        resp = await asyncio.wait_for(ws.recv(), timeout=30)
        print(f'✓ Mistral (LOCAL_REASONING): {json.loads(resp).get("signal_type")}')
        
        # 4. Start fractal process with Gemma
        fractal_signal = {
            'signal_type': 'start_fractal_process',
            'source_cell': 'fractal-test',
            'target_cell': 'thinker-alpha',
            'payload': {
                'query': 'What is the nature of emergent intelligence in distributed systems?',
                'tier': 'LOCAL_FAST',
                'max_elevations': 2
            },
            'id': uuid.uuid4().hex
        }
        await ws.send(json.dumps(fractal_signal))
        print('\n🔮 [TIER 1: LOCAL_FAST] Starting fractal process with Gemma...')
        
        resp = await asyncio.wait_for(ws.recv(), timeout=120)
        data = json.loads(resp)
        
        if data.get('signal_type') == 'fractal_error':
            print(f'Error: {data.get("payload", {}).get("error")}')
            return
        
        payload = data.get('payload', {})
        process_id = payload.get('process_id')
        print(f'\n📨 Process ID: {process_id}')
        print(f'   Tier: {payload.get("tier")}')
        print(f'   Agent: {payload.get("agent_id")}')
        print(f'   Can elevate: {payload.get("can_elevate")}')
        
        response1 = payload.get('response', '')
        print(f'\n🧠 Gemma Response ({len(response1)} chars):')
        print('─' * 60)
        print(response1[:600] + '...' if len(response1) > 600 else response1)
        print('─' * 60)
        
        # 5. Elevate thought to Mistral
        if payload.get('can_elevate'):
            print('\n⬆️ [TIER 2: LOCAL_REASONING] Elevating thought to Mistral...')
            
            elevate_signal = {
                'signal_type': 'elevate_thought',
                'source_cell': 'fractal-test',
                'target_cell': 'thinker-alpha',
                'payload': {
                    'process_id': process_id,
                    'reason': 'Need deeper philosophical analysis'
                },
                'id': uuid.uuid4().hex
            }
            await ws.send(json.dumps(elevate_signal))
            
            resp = await asyncio.wait_for(ws.recv(), timeout=180)  # Mistral is slower
            data = json.loads(resp)
            
            if data.get('signal_type') == 'fractal_error':
                print(f'Elevation error: {data.get("payload", {}).get("error")}')
            elif data.get('signal_type') == 'thought_elevated':
                payload = data.get('payload', {})
                print(f'\n📨 Elevated!')
                print(f'   New tier: {payload.get("new_tier")}')
                print(f'   Elevation count: {payload.get("elevation_count")}')
                print(f'   Agent: {payload.get("agent_id")}')
                
                response2 = payload.get('response', '')
                print(f'\n🧠 Mistral Response ({len(response2)} chars):')
                print('═' * 60)
                print(response2[:800] + '...' if len(response2) > 800 else response2)
                print('═' * 60)
        
        # 6. Get fractal status
        status_signal = {
            'signal_type': 'get_fractal_status',
            'source_cell': 'fractal-test',
            'target_cell': 'thinker-alpha',
            'payload': {'process_id': process_id},
            'id': uuid.uuid4().hex
        }
        await ws.send(json.dumps(status_signal))
        
        resp = await asyncio.wait_for(ws.recv(), timeout=10)
        data = json.loads(resp)
        
        if data.get('signal_type') == 'fractal_status':
            summary = data.get('payload', {}).get('summary', {})
            print(f'\n📊 Fractal Process Summary:')
            print(f'   Total exchanges: {summary.get("total_exchanges")}')
            print(f'   Elevations: {summary.get("elevations")}')
            print(f'   Current tier: {summary.get("current_tier")}')


if __name__ == '__main__':
    asyncio.run(main())
