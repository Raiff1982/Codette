import os
import sys
sys.path.append(os.path.dirname(__file__))
import json
import tempfile

from UniversalReasoning import UniversalReasoning
from cognition_cocooner import CognitionCocooner


def test_generate_response_creates_cocoon(tmp_path):
    config = {
        "enabled_perspectives": ["echo"],
        "ethical_considerations": "Test ethics"
    }
    cocoon_dir = tmp_path / "cocoons"
    ur = UniversalReasoning(config=config, cocooner=CognitionCocooner(str(cocoon_dir)))
    result = ur.generate_response("hello")
    assert "[Echo] hello" in result
    assert "Test ethics" in result
    cocoon_files = list(cocoon_dir.iterdir())
    assert len(cocoon_files) == 1
    with open(cocoon_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["data"]["question"] == "hello"
