import unittest
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from aiohttp import web
from app import APP, messages

class TestApp(AioHTTPTestCase):
    async def get_application(self):
        return APP

    @unittest_run_loop
    async def test_messages(self):
        headers = {"Content-Type": "application/json"}
        payload = {
            "type": "message",
            "text": "Hello, bot!",
            "from": {"id": "user1"},
            "recipient": {"id": "bot1"},
            "conversation": {"id": "conv1"},
            "channelId": "emulator",
            "serviceUrl": "http://localhost:3978"
        }
        request = await self.client.post("/api/messages", json=payload, headers=headers)
        assert request.status == 201

if __name__ == "__main__":
    unittest.main()
