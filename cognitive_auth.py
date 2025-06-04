class CognitiveAuthManager:
    """Simple in-memory user management stub for tests."""
    def __init__(self):
        self.users = {}

    def register_user(self, username: str, password: str, metadata=None):
        self.users[username] = {
            "password": password,
            "metadata": metadata or {}
        }
        return username

    def validate_user(self, username: str, password: str) -> bool:
        info = self.users.get(username)
        return info is not None and info["password"] == password

    def collapse_user_node(self, username: str):
        return self.users.pop(username, None)
