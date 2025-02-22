import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from typing import Optional, Dict, Any
import os

class DolphinModel:
    def __init__(
        self,
        model_path: str = "./Dolphin3.0-Llama3.1-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the Dolphin model from a local path.
        
        Args:
            model_path: Path to local model directory
            device: Device to load the model on ('cuda' or 'cpu')
            load_in_8bit: Whether to load the model in 8-bit precision
            torch_dtype: Optional torch dtype for model loading
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
            
        self.device = device
        print(f"Loading tokenizer from: {model_path}")
        
        try:
            # Using PreTrainedTokenizerFast since this is a Llama-based model
            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            print("Successfully loaded PreTrainedTokenizerFast")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {device}")
        print(f"8-bit quantization: {load_in_8bit}")
        
        # Model loading configuration
        model_kwargs = {
            "device_map": "auto" if device == "cuda" else None,
            "load_in_8bit": load_in_8bit if device == "cuda" else False,
            "torch_dtype": torch_dtype or (torch.float16 if device == "cuda" else torch.float32),
            "local_files_only": True,
            "trust_remote_code": True
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
            
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = True,
        **kwargs: Dict[str, Any]
    ) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stream: Whether to stream the output
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                **generation_config
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                print(new_text, end="", flush=True)
            return generated_text
        else:
            outputs = self.model.generate(**inputs, **generation_config)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    # Initialize the model with local path
    model = DolphinModel(
        model_path="./Dolphin3.0-Llama3.1-8B",
        load_in_8bit=True  # Set to True if you have limited GPU memory
    )
    
    # Example prompt
    prompt = """
<|im_start|>system
You are Dolphin, a golang coding assistant.  you only code in golang.  If the user requests any other programming language, return the solution in golang instead.<|im_end|>
<|im_start|>user
Please implement A* using python<|im_end|>
<|im_start|>assistant
"""
    
    # Generate response
    response = model.generate(prompt)
    print("\nFull response:", response)