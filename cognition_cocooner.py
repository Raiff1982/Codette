import json
import os
import uuid
from datetime import datetime

class CognitionCocooner:
    """Utility to wrap and store cognitive states in .cocoon files."""

    def __init__(self, storage_path="cocoons"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def wrap_and_store(self, data, label="cocoon"):
        """Wraps the given data and writes it to a uniquely named file."""
        meta = {
            "label": label,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        fname = f"{label}_{uuid.uuid4().hex}.cocoon"
        fpath = os.path.join(self.storage_path, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return fpath
