import unittest
from unittest.mock import patch, AsyncMock
from ai_core_system import AICore

class TestAICore(unittest.TestCase):
    def setUp(self):
        self.ai_core = AICore()

    @patch('ai_core_system.AICore._load_config')
    @patch('ai_core_system.AICore._initialize_models')
    def test_init(self, mock_initialize_models, mock_load_config):
        mock_load_config.return_value = {"model_name": "test_model"}
        mock_initialize_models.return_value = {"mistralai": None, "tokenizer": None}
        ai_core = AICore(config_path="test_config.json")
        self.assertEqual(ai_core.config, {"model_name": "test_model"})
        self.assertIn("mistralai", ai_core.models)
        self.assertIn("tokenizer", ai_core.models)

    @patch('ai_core_system.AICore._generate_local_model_response', new_callable=AsyncMock)
    @patch('ai_core_system.AICore._process_perspectives', new_callable=AsyncMock)
    @patch('ai_core_system.AICore.sentiment_analyzer')
    @patch('ai_core_system.AICore.database')
    @patch('ai_core_system.AICore.context_manager')
    @patch('ai_core_system.AICore.user_personalizer')
    @patch('ai_core_system.AICore.ethical_decision_maker')
    @patch('ai_core_system.AICore.explainable_ai')
    @patch('ai_core_system.AICore.self_healing')
    async def test_generate_response(self, mock_self_healing, mock_explainable_ai, mock_ethical_decision_maker, mock_user_personalizer, mock_context_manager, mock_database, mock_sentiment_analyzer, mock_process_perspectives, mock_generate_local_model_response):
        mock_generate_local_model_response.return_value = "model response"
        mock_process_perspectives.return_value = ["perspective1", "perspective2"]
        mock_sentiment_analyzer.detailed_analysis.return_value = "positive"
        mock_database.get_latest_feedback.return_value = None
        mock_self_healing.check_health.return_value = "healthy"
        mock_explainable_ai.explain_decision.return_value = "explanation"
        mock_ethical_decision_maker.enforce_policies.return_value = "final response"
        mock_user_personalizer.personalize_response.return_value = "personalized response"
        mock_context_manager.update_environment.return_value = None
        mock_database.log_interaction.return_value = None

        response = await self.ai_core.generate_response("test query", 1)

        self.assertIn("insights", response)
        self.assertIn("response", response)
        self.assertIn("sentiment", response)
        self.assertIn("security_level", response)
        self.assertIn("health_status", response)
        self.assertIn("explanation", response)

if __name__ == "__main__":
    unittest.main()
