import unittest
from agireasoning import CognitiveProcessor, AgileAGIFunctionality, UniversalReasoning

class TestCognitiveProcessor(unittest.TestCase):
    def test_generate_insights(self):
        processor = CognitiveProcessor(["scientific", "creative", "emotional"])
        query = "How can AGI improve healthcare?"
        insights = processor.generate_insights(query)
        self.assertEqual(len(insights), 3)
        self.assertIn("Scientific Analysis", insights[0])
        self.assertIn("Creative Insight", insights[1])
        self.assertIn("Emotional Interpretation", insights[2])

class TestAgileAGIFunctionality(unittest.TestCase):
    def setUp(self):
        self.learning_capabilities = {
            "experience_learning": True,
            "flexibility": True,
            "generalization": True
        }
        self.action_execution = {
            "goal_directed_behavior": True,
            "problem_solving": True,
            "task_autonomy": True
        }
        self.ethical_alignment = {
            "value_alignment": True,
            "self_awareness": True,
            "transparency": True
        }
        self.cognitive_modes = ["scientific", "creative", "emotional"]
        self.agi_functionality = AgileAGIFunctionality(
            self.learning_capabilities, self.action_execution, self.ethical_alignment, self.cognitive_modes
        )

    def test_analyze_learning_capabilities(self):
        analysis = self.agi_functionality.analyze_learning_capabilities()
        self.assertEqual(analysis["experience_learning"], True)
        self.assertEqual(analysis["flexibility"], True)
        self.assertEqual(analysis["generalization"], True)

    def test_analyze_action_execution(self):
        analysis = self.agi_functionality.analyze_action_execution()
        self.assertEqual(analysis["goal_directed_behavior"], True)
        self.assertEqual(analysis["problem_solving"], True)
        self.assertEqual(analysis["task_autonomy"], True)

    def test_analyze_ethical_alignment(self):
        analysis = self.agi_functionality.analyze_ethical_alignment()
        self.assertEqual(analysis["value_alignment"], True)
        self.assertEqual(analysis["self_awareness"], True)
        self.assertEqual(analysis["transparency"], True)

    def test_combined_analysis(self):
        query = "How can AGI improve healthcare?"
        combined_analysis = self.agi_functionality.combined_analysis(query)
        self.assertIn("learning_capabilities", combined_analysis)
        self.assertIn("action_execution", combined_analysis)
        self.assertIn("ethical_alignment", combined_analysis)
        self.assertIn("cognitive_insights", combined_analysis)

class TestUniversalReasoning(unittest.TestCase):
    def setUp(self):
        learning_capabilities = {
            "experience_learning": True,
            "flexibility": True,
            "generalization": True
        }
        action_execution = {
            "goal_directed_behavior": True,
            "problem_solving": True,
            "task_autonomy": True
        }
        ethical_alignment = {
            "value_alignment": True,
            "self_awareness": True,
            "transparency": True
        }
        cognitive_modes = ["scientific", "creative", "emotional"]
        agi_functionality = AgileAGIFunctionality(
            learning_capabilities, action_execution, ethical_alignment, cognitive_modes
        )
        self.universal_reasoning = UniversalReasoning(agi_functionality)

    def test_perform_reasoning(self):
        query = "How can AGI improve healthcare?"
        reasoning_results = self.universal_reasoning.perform_reasoning(query)
        self.assertIn("analysis_results", reasoning_results)
        self.assertIn("reasoning_summary", reasoning_results)
        self.assertIn("learning_capabilities", reasoning_results["analysis_results"])
        self.assertIn("action_execution", reasoning_results["analysis_results"])
        self.assertIn("ethical_alignment", reasoning_results["analysis_results"])
        self.assertIn("cognitive_insights", reasoning_results["analysis_results"])

if __name__ == "__main__":
    unittest.main()
