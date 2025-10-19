# app.py
import sys
import traceback
import gradio as gr
import asyncio
from datetime import datetime
from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    TurnContext,
    BotFrameworkAdapter,
)
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.schema import Activity, ActivityTypes
from bot import MyBot
from config import DefaultConfig
from ai_core import AICore
from aegis_integration import AegisBridge
from aegis_integration.config import AEGIS_CONFIG
from aegis_integration.routes import register_aegis_endpoints
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = DefaultConfig()

# Initialize AI Core and AEGIS
ai_core = AICore()
aegis_bridge = AegisBridge(ai_core, AEGIS_CONFIG)
ai_core.set_aegis_bridge(aegis_bridge)

# Force fallback to gpt2 for text generation
ai_core.model_id = 'gpt2'

# Bot Framework Setup
SETTINGS = BotFrameworkAdapterSettings(CONFIG.APP_ID, CONFIG.APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)

# Create Gradio interface with AEGIS integration
app = gr.Interface(
    fn=lambda x: ai_core.generate_text(x),
    inputs="text",
    outputs=[
        gr.Textbox(label="Response"),
        gr.JSON(label="AEGIS Analysis", visible=True)
    ],
    title="Codette with AEGIS",
    description="An ethical AI assistant enhanced with AEGIS analysis"
)

class CodetteGradioApp:
    def __init__(self, ai_core: AICore):
        self.ai_core = ai_core
        self.chat_history = []
    
    def process_message(self, message: str, history: list, cocoon_mode: bool = False) -> Tuple[str, list]:
        """Process a message and update chat history, with optional cocoon-powered creativity"""
        try:
            # Generate response (cocoon-powered if enabled)
            if cocoon_mode:
                # Ensure cocoons are loaded
                if not hasattr(self.ai_core, 'cocoon_data') or not self.ai_core.cocoon_data:
                    self.ai_core.load_cocoon_data()
                response = self.ai_core.remix_and_randomize_response(message, cocoon_mode=True)
            else:
                response = self.ai_core.generate_text(message)
            try:
                # Analyze sentiment
                sentiment = self.ai_core.analyze_sentiment(message)
                label = sentiment.get('label', '').upper()
                score = sentiment.get('score', 0.0)
                # Use transformers to generate a unique, sentiment-aware reply
                if label == 'POS':
                    prompt = f"The user said something positive: '{message}'. Respond in a cheerful, encouraging, and unique way."
                elif label == 'NEG':
                    prompt = f"The user said something negative: '{message}'. Respond with empathy, support, and a unique comforting message."
                elif label == 'NEU':
                    prompt = f"The user said something neutral: '{message}'. Respond in a thoughtful, neutral, and unique way."
                else:
                    prompt = f"The user's sentiment is unclear: '{message}'. Respond in a curious, open-minded, and unique way."
                char_response = self.ai_core.generate_text(prompt, max_length=60)
                sentiment_info = f"\n[Sentiment: {label} ({score:.2f})] {char_response}"
            except Exception as sent_e:
                logger.error(f"Sentiment analysis error: {sent_e}")
                sentiment_info = "\n[Sentiment: error (0.00)] 🤖 Sorry, I couldn't analyze the sentiment."
            # Update history in Gradio 'messages' format
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response + sentiment_info}
            ]
            return "", history
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Add error as assistant message
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Error: {str(e)}"}
            ]
            return "", history
    
    def analyze_text(self, text: str):
        """Perform comprehensive text analysis"""
        try:
            # Get sentiment
            sentiment = self.ai_core.analyze_sentiment(text)
            # Get embeddings
            embeddings = self.ai_core.get_embeddings(text)
            if embeddings:
                # Convert embeddings to 2D visualization
                embedding_viz = self._visualize_embeddings(embeddings)
            else:
                embedding_viz = None
            # Generate creative expansion
            expansion = self.ai_core.generate_text(
                f"Creative expansion of the concept: {text}",
                max_length=150
            )
            return (
                f"Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.2f})",
                embedding_viz,
                expansion
            )
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return (
                "Error analyzing sentiment",
                None,
                str(e)
            )
    
    def _visualize_embeddings(self, embeddings: list) -> np.ndarray:
        """Create a simple 2D visualization of embeddings"""
        # Convert to numpy array and reshape to 2D
        emb_array = np.array(embeddings)
        if len(emb_array.shape) > 2:
            emb_array = emb_array.reshape(-1, emb_array.shape[-1])
        
        # Simple dimensionality reduction (mean across dimensions)
        viz_data = emb_array.mean(axis=1)
        
        # Create a simple heatmap-style visualization
        size = int(np.sqrt(len(viz_data)))
        heatmap = viz_data[:size*size].reshape(size, size)
        return heatmap

# Create Gradio Interface
gradio_app = CodetteGradioApp(ai_core)

# Create the Bot
BOT = MyBot(ai_core)

# Bot Framework message handler
async def messages(req: Request) -> Response:
    if "application/json" in req.headers["Content-Type"]:
        body = await req.json()
    else:
        return Response(status=415)

    activity = Activity().deserialize(body)
    auth_header = req.headers["Authorization"] if "Authorization" in req.headers else ""

    response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
    if response:
        return json_response(data=response.body, status=response.status)
    return Response(status=201)

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Codette AI Assistant", theme="default") as interface:
        gr.Markdown("""
        # 🤖 Codette AI Assistant
        A sophisticated AI assistant powered by Hugging Face models.
        
        ## Features:
        - 💬 Interactive Chat
        - 📊 Sentiment Analysis
        - 🧠 Semantic Understanding
        - 🎨 Creative Generation
        """)
        
        with gr.Tabs():
            # Chat Interface
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=400,
                    type="messages"
                )
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message here...",
                        container=False
                    )
                with gr.Row():
                    cocoon_toggle = gr.Checkbox(label="Enable Cocoon-Powered Creativity", value=False)
                txt.submit(
                    gradio_app.process_message,
                    [txt, chatbot, cocoon_toggle],
                    [txt, chatbot]
                )
                clear = gr.Button("Clear")
                clear.click(lambda: [], None, chatbot)
            
            # Analysis Interface
            with gr.Tab("Analysis"):
                with gr.Row():
                    with gr.Column():
                        analysis_input = gr.Textbox(
                            label="Text to Analyze",
                            placeholder="Enter text for comprehensive analysis...",
                            lines=3
                        )
                        analyze_btn = gr.Button("Analyze")
                    
                    with gr.Column():
                        sentiment_output = gr.Textbox(label="Sentiment Analysis")
                        embedding_output = gr.Plot(label="Semantic Embedding Visualization")
                        expansion_output = gr.Textbox(
                            label="Creative Expansion",
                            lines=3
                        )
                
                analyze_btn.click(
                    gradio_app.analyze_text,
                    inputs=analysis_input,
                    outputs=[
                        sentiment_output,
                        embedding_output,
                        expansion_output
                    ]
                )
    
    return interface

# Main app setup
async def main():
    # Set up aiohttp web app
    app = web.Application(middlewares=[aiohttp_error_middleware])
    app.router.add_post("/api/messages", messages)
    
    # Launch Gradio interface in a separate thread
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        auth=None,
        favicon_path=None
    )
    
    # Start the web app
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", 3978).start()
    
    # Keep the server running
    while True:
        await asyncio.sleep(3600)  # Sleep for an hour

if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except Exception as error:
        logger.error(f"Application error: {error}")
        raise error
