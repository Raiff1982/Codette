from cognitive_auth import CognitiveAuthManager

if __name__ == "__main__":
    auth = CognitiveAuthManager()

    # Register
    cocoon_id = auth.register_user(
        username="JonathanH",
        password="Quantum#2025",
        metadata={"role": "developer", "access": "full"}
    )
    print("[REGISTERED]", cocoon_id)

    # Validate
    result = auth.validate_user("JonathanH", "Quantum#2025")
    print("[LOGIN RESULT]", result)

    # Collapse
    collapsed = auth.collapse_user_node("JonathanH")
    print("[COLLAPSED STATE]", collapsed)