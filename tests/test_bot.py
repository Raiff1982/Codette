import unittest
from unittest.mock import AsyncMock, MagicMock
from bot import MyBot
from botbuilder.core import TurnContext, ActivityHandler
from botbuilder.schema import Activity, ChannelAccount

class TestMyBot(unittest.TestCase):
    def setUp(self):
        self.ai_core_mock = MagicMock()
        self.bot = MyBot(self.ai_core_mock)
        self.turn_context_mock = MagicMock(spec=TurnContext)
        self.turn_context_mock.activity = Activity()
        self.turn_context_mock.activity.from_property = ChannelAccount()
        self.turn_context_mock.activity.from_property.id = "user1"
        self.turn_context_mock.activity.text = "Hello"

    def test_inheritance(self):
        self.assertIsInstance(self.bot, ActivityHandler)

    async def test_on_message_activity(self):
        self.ai_core_mock.generate_response = AsyncMock(return_value={"response": "Hi there!"})
        await self.bot.on_message_activity(self.turn_context_mock)
        self.ai_core_mock.generate_response.assert_called_once_with("Hello", "user1")
        self.turn_context_mock.send_activity.assert_called_once_with("Hi there!")

    async def test_on_members_added_activity(self):
        members_added = [ChannelAccount(id="user1"), ChannelAccount(id="user2")]
        self.turn_context_mock.activity.recipient.id = "bot"
        await self.bot.on_members_added_activity(members_added, self.turn_context_mock)
        self.turn_context_mock.send_activity.assert_called_once_with("Hello and welcome!")

if __name__ == "__main__":
    unittest.main()
