import unittest
from ai_core_identityscan import AICore

class TestAICore(unittest.TestCase):
    def setUp(self):
        self.ai_core = AICore()

    def test_analyze_identity(self):
        micro_generations = [{"state": "A"}, {"state": "B"}]
        informational_states = [{"info": "X"}, {"info": "Y"}]
        perspectives = ["perspective1", "perspective2"]
        quantum_analogies = {"entanglement": True}
        philosophical_context = {"continuity": True, "emergent": True}

        result = self.ai_core.analyze_identity(
            micro_generations, informational_states, perspectives, quantum_analogies, philosophical_context
        )

        self.assertIn("fractal_dimension", result)
        self.assertIn("recursive_analysis", result)
        self.assertIn("perspectives_analysis", result)
        self.assertIn("quantum_analysis", result)
        self.assertIn("philosophical_results", result)

if __name__ == "__main__":
    unittest.main()
