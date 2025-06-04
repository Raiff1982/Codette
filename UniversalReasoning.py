import json
import os
from typing import List, Dict, Any

from cognition_cocooner import CognitionCocooner

class EchoPerspective:
    """Simple perspective that echoes the question."""
    def __init__(self, config: Dict[str, Any]):
        pass
    def generate_response(self, question: str) -> str:
        return f"[Echo] {question}"

class UniversalReasoning:
    """Minimal reasoning engine leveraging configurable perspectives."""

    DEFAULT_CONFIG_PATH = "Codetteconfig.json"

    def __init__(self, config_path: str | None = None, config: Dict[str, Any] | None = None, cocooner: CognitionCocooner | None = None):
        if config is None:
            cfg_path = config_path or self.DEFAULT_CONFIG_PATH
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                config = {}
        self.config = config
        self.perspectives = self.initialize_perspectives()
        self.cocooner = cocooner or CognitionCocooner()

    def initialize_perspectives(self) -> List[Any]:
        mapping = {
            "echo": EchoPerspective,
        }
        enabled = self.config.get("enabled_perspectives", ["echo"])
        return [mapping[name](self.config) for name in enabled if name in mapping]

    def generate_response(self, question: str) -> str:
        responses = [p.generate_response(question) for p in self.perspectives]
        ethical = self.config.get(
            "ethical_considerations",
            "Act transparently and respect user privacy."
        )
        responses.append(f"**Ethical Considerations:**\n{ethical}")
        final = "\n\n".join(responses)
        # Store in cocoon
        self.cocooner.wrap_and_store({"question": question, "response": final})
        return final
