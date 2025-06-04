import asyncio
from pathlib import Path
from codette_quantum_multicore2 import analyse_cocoons_async

def test_async_analysis():
    cocoon_dir = Path('.')
    result = asyncio.run(analyse_cocoons_async(cocoon_dir))
    assert isinstance(result, list)
