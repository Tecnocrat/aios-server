"""Read full fractal process conversations."""
import asyncio
import websockets
import json
import uuid


async def get_full_lineage(process_id: str = None):
    """Retrieve full conversation lineage from a fractal process."""
    uri = 'ws://localhost:9600'
    async with websockets.connect(uri, ping_interval=20, ping_timeout=30) as ws:
        # Identify
        identify = {
            'signal_type': 'identify',
            'source_cell': 'lineage-reader',
            'target_cell': 'thinker-alpha',
            'payload': {'cell_type': 'external'},
            'id': uuid.uuid4().hex
        }
        await ws.send(json.dumps(identify))
        await ws.recv()

        # Get fractal status
        status_signal = {
            'signal_type': 'get_fractal_status',
            'source_cell': 'lineage-reader',
            'target_cell': 'thinker-alpha',
            'payload': {'process_id': process_id} if process_id else {},
            'id': uuid.uuid4().hex
        }
        await ws.send(json.dumps(status_signal))

        resp = await asyncio.wait_for(ws.recv(), timeout=10)
        data = json.loads(resp)

        if data.get('signal_type') == 'fractal_status':
            payload = data.get('payload', {})

            if 'process' in payload:
                # Single process - full details
                process = payload['process']
                print(f"\n{'═' * 80}")
                print(f"🔮 FRACTAL PROCESS: {process['id']}")
                print(f"{'═' * 80}")
                print(f"Origin Query: {process.get('origin_query', 'N/A')}")
                print(f"Started: {process.get('started_at', 'N/A')}")
                print(f"Current Tier: {process.get('current_tier')}")
                print(f"Elevations: {process.get('elevation_count')} / {process.get('max_elevations')}")

                print(f"\n{'─' * 80}")
                print("📜 FULL EXCHANGE HISTORY")
                print(f"{'─' * 80}")

                for i, exchange in enumerate(process.get('exchanges', []), 1):
                    print(f"\n[Exchange {i}] Tier: {exchange.get('tier')} | Agent: {exchange.get('agent_id')}")
                    print(f"Timestamp: {exchange.get('timestamp')}")
                    print(f"\n📥 Query:\n{exchange.get('query', 'N/A')}")
                    print(f"\n📤 Response:\n{exchange.get('response', 'N/A')}")
                    print(f"\n{'─' * 40}")

                if process.get('distillations'):
                    print(f"\n{'─' * 80}")
                    print("⬆️ ELEVATION HISTORY")
                    print(f"{'─' * 80}")
                    for d in process.get('distillations', []):
                        print(f"  {d.get('from_tier')} → {d.get('to_tier')}")
                        print(f"  Reason: {d.get('reason')}")
                        print(f"  Time: {d.get('timestamp')}")
            else:
                # All processes summary
                print(f"\n📊 ALL FRACTAL PROCESSES ({payload.get('total_processes', 0)} total)")
                print(f"Current active: {payload.get('current_process', 'None')}")
                print()
                for p in payload.get('processes', []):
                    print(f"  [{p['id'][:8]}...] {p['tier']} | Elevations: {p['elevations']}")
                    print(f"    Query: {p['origin']}")
                    print()


if __name__ == '__main__':
    import sys
    process_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(get_full_lineage(process_id))
