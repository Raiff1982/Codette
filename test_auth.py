import os
import sys

sys.path.append(os.path.dirname(__file__))
from cognitive_auth import CognitiveAuthManager


def test_register_and_validate():
    auth = CognitiveAuthManager()
    cocoon_id = auth.register_user(
        username="JonathanH",
        password="Quantum#2025",
        metadata={"role": "developer", "access": "full"},
    )
    assert cocoon_id == "JonathanH"
    assert auth.validate_user("JonathanH", "Quantum#2025")
    assert auth.collapse_user_node("JonathanH") is not None
