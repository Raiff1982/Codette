# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount
from ai_core_system import AICore

class MyBot(ActivityHandler):
    def __init__(self, ai_core: AICore):
        super().__init__()
        self.ai_core = ai_core

    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    async def on_message_activity(self, turn_context: TurnContext):
        user_id = turn_context.activity.from_property.id
        user_message = turn_context.activity.text

        # Generate response using AI Core
        response = await self.ai_core.generate_response(user_message, user_id)

        # Send the response back to the user
        await turn_context.send_activity(response["response"])

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")
