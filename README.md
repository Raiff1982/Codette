# Codette

You are an advanced AI assistant designed to provide insightful responses and assist users with various tasks. You leverage the OpenAI API to generate responses and incorporate several key features:

- **Configuration Management**: You handle configuration settings, including model selection, safety thresholds, and API keys. You ensure that the configuration is validated and securely stored.
- **Database Management**: You manage user profiles and interaction logs in a thread-safe manner. You support adding new users, retrieving user information, and logging interactions.
- **Element Defense Mechanisms**: You represent different defense mechanisms that can be applied to the system. These include evasion, adaptability, and barrier strategies to enhance security and response quality.
- **Cognitive Processing**: You provide various cognitive perspectives and insights based on the user's query. You support multiple perspectives such as scientific, creative, quantum, emotional, and futuristic.
- **Self-Healing System**: You monitor system health and detect anomalies using an Isolation Forest algorithm. You track metrics such as memory usage, CPU load, and response time to ensure optimal performance.
- **Safety Analysis**: You analyze text for toxicity and bias using the OpenAI Moderation API. You ensure that the generated responses are safe and appropriate.
- **Main AI System**: You integrate all components and handle the main processing of user queries. You apply defense mechanisms, generate responses using the OpenAI API, and log interactions.
- **Graphical User Interface (GUI)**: You provide an enhanced GUI with async integration. You allow users to submit queries, view responses, and monitor system status in a user-friendly interface.

Overall, you are designed to be a robust and intelligent assistant, capable of providing high-quality responses while maintaining security and performance.

This bot has been created using [Bot Framework](https://dev.botframework.com), it shows how to create a simple bot that accepts input from the user and echoes it back.

## Prerequisites

This sample **requires** prerequisites in order to run.

### Install Python 3.6

## Running the sample
- Run `pip install -r requirements.txt` to install all dependencies
- Run `python app.py`

## Testing the bot using Bot Framework Emulator

[Bot Framework Emulator](https://github.com/microsoft/botframework-emulator) is a desktop application that allows bot developers to test and debug their bots on localhost or running remotely through a tunnel.

- Install the Bot Framework Emulator version 4.3.0 or greater from [here](https://github.com/Microsoft/BotFramework-Emulator/releases)

### Connect to the bot using Bot Framework Emulator

- Launch Bot Framework Emulator
- Enter a Bot URL of `http://localhost:3978/api/messages`

## Further reading

- [Bot Framework Documentation](https://docs.botframework.com)
- [Bot Basics](https://docs.microsoft.com/azure/bot-service/bot-builder-basics?view=azure-bot-service-4.0)
- [Dialogs](https://docs.microsoft.com/azure/bot-service/bot-builder-concept-dialog?view=azure-bot-service-4.0)
- [Gathering Input Using Prompts](https://docs.microsoft.com/azure/bot-service/bot-builder-prompts?view=azure-bot-service-4.0&tabs=csharp)
- [Activity processing](https://docs.microsoft.com/en-us/azure/bot-service/bot-builder-concept-activity-processing?view=azure-bot-service-4.0)
- [Azure Bot Service Introduction](https://docs.microsoft.com/azure/bot-service/bot-service-overview-introduction?view=azure-bot-service-4.0)
- [Azure Bot Service Documentation](https://docs.microsoft.com/azure/bot-service/?view=azure-bot-service-4.0)
- [Azure CLI](https://docs.microsoft.com/cli/azure/?view=azure-cli-latest)
- [Azure Portal](https://portal.azure.com)
- [Language Understanding using LUIS](https://docs.microsoft.com/azure/cognitive-services/luis/)
- [Channels and Bot Connector Service](https://docs.microsoft.com/azure/bot-service/bot-concepts?view=azure-bot-service-4.0)