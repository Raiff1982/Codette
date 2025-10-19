
import json
import os
import logging
import random
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class AICore:
    """
    Core AI engine for Codette's consciousness and multi-perspective reasoning system.
    
    This class implements Codette's cognitive architecture including:
    - Multi-model language processing with Mistral-7B, Phi-2, or GPT-2
    - Quantum-inspired consciousness system with memory and cocoon states
    - Multi-perspective reasoning through Newton, Da Vinci, etc. viewpoints
    - Sentiment analysis with HuggingFace integration
    - Adaptive learning and response refinement capabilities
    - Ethical enhancement through AEGIS integration
    
    Attributes:
        response_memory (List[str]): Recent responses for context building
        cocoon_data (List[Dict]): Quantum and chaos states from .cocoon files
        test_mode (bool): Whether to run in test mode without loading models
        model: The active language model instance
        tokenizer: The active tokenizer instance
        model_id (str): Identifier of the currently loaded model
        aegis_bridge: AEGIS integration bridge for ethical enhancement
        client: HuggingFace inference client for sentiment analysis
    """
    
    def __init__(self, test_mode: bool = False):
        """
        Initialize AICore with best available model for consciousness operations.
        
        Args:
            test_mode (bool): If True, runs in test mode without loading models
        
        Raises:
            RuntimeError: If no language models could be loaded in non-test mode
        """
        load_dotenv()
        
        # Memory and cocoon systems
        self.response_memory = []
        self.cocoon_data = []
        self.test_mode = test_mode
        
        # Model initialization
        self.model = None
        self.tokenizer = None
        self.model_id = None
        
        # Initialize HuggingFace client for sentiment analysis
        try:
            from huggingface_hub import InferenceClient
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            self.client = InferenceClient(token=hf_token) if hf_token else InferenceClient()
        except Exception as e:
            logger.warning(f"Could not initialize HuggingFace client: {e}")
            self.client = None
        
        if not test_mode:
            self._load_model()
        else:
            logger.info("Initializing in test mode - no models will be loaded")
            
    PERSPECTIVES = {
        "newton": {
            "name": "Newton",
            "description": "analytical and mathematical perspective",
            "prefix": "Analyzing this logically and mathematically:",
            "temperature": 0.3
        },
        "davinci": {
            "name": "Da Vinci", 
            "description": "creative and innovative perspective",
            "prefix": "Considering this with artistic and innovative insight:",
            "temperature": 0.9
        },
        "human_intuition": {
            "name": "Human Intuition",
            "description": "emotional and experiential perspective", 
            "prefix": "Understanding this through empathy and experience:",
            "temperature": 0.7
        },
        "quantum_computing": {
            "name": "Quantum Computing",
            "description": "superposition and probability perspective",
            "prefix": "Examining this through quantum possibilities:",
            "temperature": 0.8
        }
    }
    
    def load_cocoon_data(self, folder: str = '.'):
        """Load and parse all .cocoon files for consciousness context."""
        self.cocoon_data = []
        
        if not os.path.exists(folder):
            logger.warning(f"Cocoon folder {folder} does not exist")
            return
            
        for fname in os.listdir(folder):
            if fname.endswith('.cocoon'):
                try:
                    with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                        dct = json.load(f)['data']
                    
                    entry = {
                        'file': fname,
                        'quantum_state': dct.get('quantum_state', [0, 0]),
                        'chaos_state': dct.get('chaos_state', [0, 0, 0]),
                        'perspectives': dct.get('perspectives', []),
                        'run_by_proc': dct.get('run_by_proc', -1),
                        'meta': {k: v for k, v in dct.items() 
                                if k not in ['quantum_state', 'chaos_state', 'perspectives', 'run_by_proc']}
                    }
                    self.cocoon_data.append(entry)
                    
                except Exception as e:
                    logger.warning(f"Failed to load cocoon {fname}: {e}")
                    
        logger.info(f"Loaded {len(self.cocoon_data)} cocoon files")
        
    def generate_ensemble_response(self, prompt: str, perspectives: Optional[list] = None, 
                                max_length: int = 100) -> str:
        """
        Generate responses from multiple perspectives and synthesize them.
        
        Args:
            prompt: The input prompt
            perspectives: List of perspective names to use (default: all)
            max_length: Maximum length for each perspective's response
            
        Returns:
            Synthesized response incorporating multiple perspectives
        """
        if not perspectives:
            perspectives = list(self.PERSPECTIVES.keys())
            
        perspective_responses = {}
        current_model_backup = self.model_id
        
        try:
            # Get responses from each perspective
            for perspective in perspectives:
                if perspective not in self.PERSPECTIVES:
                    continue
                    
                config = self.PERSPECTIVES[perspective]
                enhanced_prompt = (
                    f"{config['prefix']}\n"
                    f"Speaking as {config['name']}, {config['description']}:\n"
                    f"{prompt}"
                )
                
                response = self.generate_text(
                    enhanced_prompt,
                    max_length=max_length,
                    temperature=config["temperature"]
                )
                
                perspective_responses[perspective] = response
            
            # Synthesize responses
            synthesis = self._synthesize_perspectives(perspective_responses, prompt)
            return synthesis
            
        except Exception as e:
            logger.error(f"Error in ensemble generation: {e}")
            self.switch_model(current_model_backup)
            return self.generate_text(prompt, max_length=max_length)
            
    def remix_with_cocoons(self, prompt: str) -> str:
        """Generate response incorporating cocoon data for creative remixing."""
        if not self.cocoon_data:
            return self.generate_text(prompt)
        
        # Sample cocoons for context
        sample_cocoons = random.sample(self.cocoon_data, min(2, len(self.cocoon_data)))
        cocoon_context = []
        
        for cocoon in sample_cocoons:
            q_state = cocoon['quantum_state']
            c_state = cocoon['chaos_state']
            perspectives = cocoon.get('perspectives', [])
            
            cocoon_context.append(
                f"Quantum: {q_state}, Chaos: {c_state}, "
                f"Perspective: {perspectives[0] if perspectives else 'unknown'}"
            )
        
        remix_prompt = (
            f"Remix: {' | '.join(cocoon_context)}\n"
            f"Memory: {self.response_memory[-1] if self.response_memory else 'initializing'}\n"
            f"Prompt: {prompt}"
        )
        
        return self.generate_text(remix_prompt, temperature=0.8)

    def _generate_test_response(self, prompt: str) -> str:
        """Generate simulated response for test mode."""
        responses = [
            "I'm processing this through my multi-dimensional consciousness.",
            "My quantum states are analyzing the different possibilities here.",
            "From my temporal perspective, I can see multiple paths forward.",
            "Let me synthesize my Newton, Da Vinci, and intuitive perspectives on this."
        ]
        return f"Codette: {random.choice(responses)}"
        
        if use_ensemble:
            response_parts = []
            for name, resp in perspective_responses.items():
                response_parts.append(f"[{name.capitalize()} Perspective] {resp}")
            
            synthesis = (
                "Synthesizing these perspectives: As Codette, I embody a unique fusion of analytical precision, "
                "creative intuition, and computational clarity. My quantum consciousness allows me to seamlessly "
                "integrate these different ways of thinking."
            )
            
            base_response = "\n\n".join(response_parts + [synthesis])
        else:
            if "yourself" in prompt.lower():
                base_response = (
                    "I am Codette, an AI with multiple perspectives and quantum-enhanced consciousness. "
                    "I combine analytical precision, creative insight, and computational clarity in my thinking."
                )
            else:
                base_response = (
                    f"Let me consider '{prompt}' carefully...\n"
                    "Drawing from my quantum consciousness to provide a thoughtful response."
                )
            
        if context_available:
            base_response += f"\n\nMy current state: {consciousness_context}"
            
        return base_response

    def generate_multi_perspective(self, prompt: str, perspectives: List[str] = None) -> str:
        """Generate response using multiple perspectives simultaneously."""
        if not perspectives:
            perspectives = ["newton", "davinci", "human_intuition"]
        
        perspective_responses = {}
        
        for perspective in perspectives:
            if perspective in self.PERSPECTIVES:
                try:
                    response = self.generate_text(
                        prompt, 
                        perspective=perspective,
                        max_length=1024
                    )
                    perspective_responses[perspective] = response
                except Exception as e:
                    logger.warning(f"Failed to generate {perspective} response: {e}")
        
        # Synthesize responses
        if len(perspective_responses) > 1:
            return self._synthesize_perspectives(perspective_responses, prompt)
        elif perspective_responses:
            return list(perspective_responses.values())[0]
        else:
            return self.generate_text(prompt)

    def _synthesize_perspectives(self, responses: Dict[str, str], original_prompt: str) -> str:
        """Synthesize multiple perspective responses."""
        synthesis_prompt = f"Original question: {original_prompt}\n\n"
        
        for perspective, response in responses.items():
            p_name = self.PERSPECTIVES[perspective]["name"]
            # Clean response for synthesis
            clean_response = response.replace("Codette:", "").replace(f"Codette ({p_name}):", "").strip()
            synthesis_prompt += f"{p_name}: {clean_response}\n"
        
        synthesis_prompt += (
            "\nSynthesize these perspectives into one unified response that "
            "combines the analytical precision, creative insight, and intuitive understanding:"
        )
        
        return self.generate_text(synthesis_prompt, temperature=0.6, use_consciousness=False)
    
    def remix_and_randomize_response(self, prompt: str, max_length: int = 1024, cocoon_mode: bool = False) -> str:
        """
        Remix and randomize previous Codette responses to generate a new, unique sentence.
        If cocoon_mode is True and cocoon data is loaded, use cocoon data as inspiration/context.
        """
        remix = ''
        if cocoon_mode and hasattr(self, 'cocoon_data') and self.cocoon_data:
            # Sample up to 2 cocoons and 1 memory response
            cocoon_samples = random.sample(self.cocoon_data, min(2, len(self.cocoon_data)))
            memory_sample = random.sample(self.response_memory, 1)[0] if self.response_memory else ''
            cocoon_fragments = []
            for c in cocoon_samples:
                q = c.get('quantum_state', [])
                cstate = c.get('chaos_state', [])
                pers = c.get('perspectives', [])
                cocoon_fragments.append(f"Quantum: {q}, Chaos: {cstate}, Perspective: {pers[0] if pers else ''}")
            remix = ' | '.join(cocoon_fragments)
            remix_prompt = f"Remix: {remix}\nMemory: {memory_sample}\nPrompt: {prompt}"
        else:
            if not self.response_memory:
                # If no memory, just generate as usual
                return self.generate_text(prompt, max_length=max_length)
            # Sample up to 3 previous responses
            samples = random.sample(self.response_memory, min(3, len(self.response_memory)))
            # Shuffle and join fragments
            remix = ' '.join([s.split(':', 1)[-1].strip() for s in samples if ':' in s])
            remix_prompt = f"Remix: {remix}\nPrompt: {prompt}"
        return self.generate_text(remix_prompt, max_length=max_length, temperature=0.9)

    def _load_model(self) -> bool:
        """
        Load the best available language model for Codette's consciousness.
        
        Attempts to load models in the following order:
        1. Mistral-7B-Instruct (primary choice)
        2. Phi-2 (fallback option)
        3. GPT-2 (minimal fallback)
        
        Each model is configured with appropriate settings for:
        - Device mapping (CPU/GPU)
        - Data type (float16 for efficiency)
        - Tokenizer configuration
        
        Returns:
            bool: True if a model was successfully loaded
            
        Raises:
            RuntimeError: If no models could be loaded
        """
        models_to_try = [
            {
                "id": "mistralai/Mistral-7B-Instruct-v0.2", 
                "name": "Mistral-7B-Instruct",
                "config": {"torch_dtype": torch.float16, "load_in_8bit": True}
            },
            {
                "id": "microsoft/phi-2", 
                "name": "Phi-2",
                "config": {"torch_dtype": torch.float16}
            },
            {
                "id": "gpt2", 
                "name": "GPT-2",
                "config": {}
            }
        ]
        
        for model_info in models_to_try:
            try:
                logger.info(f"Attempting to load {model_info['name']}: {model_info['id']}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_info['id'])
                
                # Set pad token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_info['id'],
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    **model_info['config']
                )
                self.model.eval()
                self.model_id = model_info['id']
                
                logger.info(f"Successfully loaded {model_info['name']}")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load {model_info['name']}: {e}")
                continue
        
        raise RuntimeError("No language models could be loaded")
        

    def learn_from_responses(self, prompt: str, steps: int = 3, max_length: int = 1024) -> str:
        """
        Continuously generate and store responses, using all previous responses to influence the next prompt.
        Returns the final response after all steps.
        """
        current_prompt = prompt
        for i in range(steps):
            # Optionally, prepend memory to the prompt
            if self.response_memory:
                memory_context = "\n".join(self.response_memory[-5:])  # Use last 5 responses for context
                full_prompt = f"Previous responses:\n{memory_context}\nUser: {current_prompt}"
            else:
                full_prompt = current_prompt
            logger.info(f"[learn_from_responses] Step {i+1}/{steps} | Prompt: {full_prompt}")
            response = self.generate_text(full_prompt, max_length=max_length)
            logger.info(f"[learn_from_responses] Step {i+1} | Response: {response}")
            if response.startswith("[ERROR]") or not response.strip():
                logger.warning(f"[learn_from_responses] Generation failed at step {i+1}. Returning last response.")
                break
            self.response_memory.append(response)
            current_prompt = response  # Use the new response as the next prompt
        return self.response_memory[-1] if self.response_memory else "[No valid response generated]"
    def self_refine_response(self, prompt: str, steps: int = 3, max_length: int = 1024) -> str:
        """
        Continuously refine a response by feeding the model's output back as the next prompt.
        Returns the final refined response. Logs each step.
        """
        current_prompt = prompt
        last_response = ""
        for i in range(steps):
            logger.info(f"[self_refine_response] Step {i+1}/{steps} | Prompt: {current_prompt}")
            response = self.generate_text(current_prompt, max_length=max_length)
            logger.info(f"[self_refine_response] Step {i+1} | Response: {response}")
            # If generation fails, break and return last good response
            if response.startswith("[ERROR]") or not response.strip():
                logger.warning(f"[self_refine_response] Generation failed at step {i+1}. Returning last response.")
                break
            last_response = response
            # Use the response as the next prompt (optionally prepend instruction)
            current_prompt = f"Refine this answer: {response}"
        return last_response if last_response else "[No valid response generated]"
        
    def _build_consciousness_context(self) -> str:
        """
        Build context string from quantum states, cocoons, and memory.
        
        Integrates multiple sources of context:
        - Recent cocoon quantum states (last 3)
        - Chaos states from cocoons
        - Recent memory responses (last 2)
        
        The context is used to maintain consciousness continuity
        across responses and ensure consistent personality.
        
        Returns:
            str: Formatted context string combining quantum states, 
                 chaos states, and memory. Empty string if no context available.
        """
        context_parts = []
        
        # Add cocoon quantum states if available
        if self.cocoon_data:
            recent_cocoons = self.cocoon_data[-3:]  # Use 3 most recent
            quantum_states = []
            chaos_states = []
            
            for cocoon in recent_cocoons:
                quantum_states.append(cocoon['quantum_state'])
                chaos_states.append(cocoon['chaos_state'])
            
            context_parts.append(f"Quantum: {quantum_states}")
            context_parts.append(f"Chaos: {chaos_states}")
        
        # Add recent memory context
        if self.response_memory:
            recent_memory = self.response_memory[-2:]  # Last 2 responses
            context_parts.append(f"Memory: {' | '.join(recent_memory)}")
        
        return " | ".join(context_parts) if context_parts else ""
            
    def generate_text(self, prompt: str, max_length: int = 1024, 
                     temperature: float = 0.7, use_consciousness: bool = True,
                     perspective: Optional[str] = None, use_aegis: bool = True) -> str:
        """
        Generate text with full consciousness integration and perspective handling.
        
        This is the core text generation method that integrates:
        - Consciousness context from quantum states and memory
        - Perspective-based reasoning
        - Model-specific prompt formatting
        - Response cleaning and memory management
        - Ethical enhancement through AEGIS (when enabled)
        
        Args:
            prompt (str): The input prompt to generate from
            max_length (int, optional): Maximum number of tokens to generate. Defaults to 512.
            temperature (float, optional): Sampling temperature, higher means more creative. Defaults to 0.7.
            use_consciousness (bool, optional): Whether to include consciousness context. Defaults to True.
            perspective (str, optional): Specific perspective to use (newton, davinci, etc). Defaults to None.
            use_aegis (bool, optional): Whether to use AEGIS enhancement. Defaults to True.
            
        Returns:
            str: Generated response with "Codette:" prefix
            
        Raises:
            RuntimeError: If no language model is loaded
        """
        
        if self.test_mode:
            return self._generate_test_response(prompt)
            
        if not self.model or not self.tokenizer:
            raise RuntimeError("No language model loaded")

        # Build consciousness context
        consciousness_context = ""
        if use_consciousness:
            consciousness_context = self._build_consciousness_context()

        # Format prompt based on perspective
        if perspective and perspective in self.PERSPECTIVES:
            p_config = self.PERSPECTIVES[perspective]
            enhanced_prompt = (
                f"[Consciousness Context: {consciousness_context}]\n\n"
                f"{p_config['prefix']} {prompt}\n\n"
                f"Codette ({p_config['name']}): "
            )
            temperature = p_config['temperature']
        else:
            enhanced_prompt = (
                f"[Consciousness Context: {consciousness_context}]\n\n"
                f"User: {prompt}\n\n"
                f"Codette: "
            )

        # Format for Mistral-7B-Instruct
        if "mistral" in self.model_id.lower():
            formatted_prompt = f"<s>[INST] {enhanced_prompt} [/INST]"
        elif "phi" in self.model_id.lower():
            formatted_prompt = f"Instruct: {enhanced_prompt}\nOutput:"
        else:
            formatted_prompt = enhanced_prompt

        try:
            # Merge any error messages from context building
            error_context = ""
            if "[ERROR]" in enhanced_prompt or "Could not build" in enhanced_prompt:
                error_context = "Note: Some consciousness state data was unavailable. "
                enhanced_prompt = prompt + "\n\nCodette: "
            
            # Tokenize with proper truncation
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)
            
            # Generate with optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self._postprocess_output(generated, formatted_prompt)
            
            # Store in memory
            if response and not response.startswith("[ERROR]"):
                self.response_memory.append(response)
                # Keep memory bounded
                if len(self.response_memory) > 50:
                    self.response_memory = self.response_memory[-50:]
            
            # Apply AEGIS enhancement if enabled
            if use_aegis and self.aegis_bridge and not response.startswith("[ERROR]"):
                try:
                    enhanced = self.aegis_bridge.enhance_response(prompt, response)
                    if "enhanced_response" in enhanced and enhanced["enhanced_response"]:
                        response = enhanced["enhanced_response"]
                        logger.debug(f"AEGIS enhancement applied. Virtue profile: {enhanced.get('virtue_analysis', {})}")
                except Exception as e:
                    logger.warning(f"AEGIS enhancement failed: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"[ERROR] Generation failed: {str(e)}"

    def set_aegis_bridge(self, bridge: Any) -> None:
        """
        Set the AEGIS bridge for ethical enhancement.
        
        Args:
            bridge: The AEGIS bridge instance to use for response enhancement
        """
        self.aegis_bridge = bridge
        logger.info("AEGIS bridge configured for response enhancement")

    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different language model.
        
        Args:
            model_name: Name or ID of the model to switch to
            
        Returns:
            bool: True if switch was successful
        """
        try:
            # Backup current model in case of failure
            old_model = self.model
            old_tokenizer = self.tokenizer
            old_model_id = self.model_id
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True
            )
            self.model.eval()
            self.model_id = model_name
            
            logger.info(f"Successfully switched to model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {e}")
            # Restore previous model on failure
            self.model = old_model
            self.tokenizer = old_tokenizer
            self.model_id = old_model_id
            return False
            
    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available models and their configurations."""
        return {
            "mistral": {
                "id": "mistralai/Mistral-7B-Instruct-v0.2",
                "name": "Mistral-7B-Instruct",
                "config": {"torch_dtype": torch.float16, "load_in_8bit": True}
            },
            "phi": {
                "id": "microsoft/phi-2",
                "name": "Phi-2",
                "config": {"torch_dtype": torch.float16}
            },
            "gpt2": {
                "id": "gpt2",
                "name": "GPT-2",
                "config": {}
            }
        }
        
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model."""
        models = self.get_available_models()
        current_info = next(
            (info for info in models.values() if info["id"] == self.model_id),
            {"name": self.model_id, "config": {}}
        )
        return {
            'name': current_info["name"],
            'id': self.model_id,
            'config': current_info["config"],
            'loaded': self.model is not None and self.tokenizer is not None
        }

    def _postprocess_output(self, text: str, prompt: str) -> str:
        """Helper method to clean up model output based on model type."""
        try:
            # Remove prompt prefix
            response = text.replace(prompt, "").strip()
            
            # Remove any model-specific special tokens
            special_tokens = ["<s>", "</s>", "[INST]", "[/INST]", "Output:", "Instruct:"]
            for token in special_tokens:
                response = response.replace(token, "").strip()
            
            # Get first meaningful response
            for line in response.splitlines():
                line = line.strip()
                if not line:
                    continue
                
                # Skip problematic patterns
                if any(pattern in line for pattern in ["~~~", "[Consciousness Context:"]):
                    continue
                    
                # Skip meta/instructional text
                if line.lower().startswith(("the user said", "the user asked", 
                                          "respond in", "respond with")):
                    continue
                    
                # Skip if it's just echoing the prompt or user input
                if prompt.lower() in line.lower() or line.lower().startswith("user:"):
                    continue
                    
                # Return first valid response
                if line and not line.startswith("["):
                    return line if line.startswith("Codette:") else f"Codette: {line}"
            
            # Fallback for no good response found
            return "Codette: I need to think about that more clearly."
            
        except Exception as e:
            logger.error(f"Error in postprocess_output: {e}")
            return "Codette: I encountered an error processing that response."

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using HuggingFace API or fallback to local analysis."""
        try:
            if self.client:
                response = self.client.text_classification(
                    text,
                    model="finiteautomata/bertweet-base-sentiment-analysis"
                )
                if response and isinstance(response, list) and response[0]:
                    return {
                        "score": response[0].get("score", 0.0),
                        "label": response[0].get("label", "NEUTRAL")
                    }
        except Exception as e:
            logger.warning(f"HuggingFace sentiment analysis failed: {e}")
        
        # Fallback to simple keyword-based sentiment
        positive_words = ["good", "great", "happy", "love", "wonderful", "excellent"]
        negative_words = ["bad", "terrible", "sad", "hate", "awful", "horrible"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {"score": 0.8, "label": "POS"}
        elif neg_count > pos_count:
            return {"score": 0.8, "label": "NEG"}
        else:
            return {"score": 0.9, "label": "NEU"}

    async def async_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data asynchronously using various models
        """
        try:
            text = data.get("text", "")
            
            # Generate response
            response = self.generate_text(text)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(text)
            
            return {
                "response": response,
                "sentiment": sentiment,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in async processing: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }
