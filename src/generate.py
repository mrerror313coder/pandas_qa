#!/usr/bin/env python3
"""
generate.py
===========
Uses an open instruction LLM to generate an answer grounded in
retrieved passages.

The prompt strictly instructs the model to answer only from the
given passages, and to cite the doc_name it used.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = """You are a documentation assistant for the pandas library. Answer the question using ONLY the information in the passages below.

STRICT RULES:
1. If the passages contain the answer, give a concise answer (1-2 sentences) followed by "Source: <doc_name>".
2. If and ONLY IF the passages truly do not contain the answer, respond with just: NOT_FOUND
3. Never write both an answer AND NOT_FOUND. Pick ONE.
4. Do not invent information not in the passages.

Passages:
{passages}

Question: {question}

Answer:"""


class Generator:
    """
    Wraps a HuggingFace instruction model for grounded answer generation.
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct",
                 device=None):
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading generator model: {model_name}")
        print(f"  Device: {device}")
        print(f"  (First run downloads model - ~6 GB)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self.model.eval()
        print(f"  Model loaded.")
    
    def format_passages(self, passages):
        """Format retrieved passages for the prompt."""
        formatted = []
        for i, p in enumerate(passages, 1):
            formatted.append(
                f"[Passage {i}] doc_name: {p['doc_name']} | section: {p['section']}\n"
                f"{p['text']}"
            )
        return "\n\n".join(formatted)
    
    def generate(self, question, passages, max_new_tokens=200):
        """
        Generate an answer for the question given the retrieved passages.
        """
        # Build the prompt
        passages_str = self.format_passages(passages)
        prompt = PROMPT_TEMPLATE.format(
            passages=passages_str,
            question=question,
        )
        
        # Use chat template if available (better for instruction models)
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=3500,  # leave room for output
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # deterministic
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the newly generated tokens
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        return answer


if __name__ == "__main__":
    # Demo mode
    from retrieve import Retriever
    
    print("Loading retriever...")
    retriever = Retriever()
    
    print("Loading generator...")
    generator = Generator()
    
    while True:
        question = input("\nQuestion (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        
        print("Retrieving...")
        passages = retriever.retrieve(question, k=5)
        
        print("Generating...")
        answer = generator.generate(question, passages)
        
        print(f"\n{'=' * 60}")
        print(f"Answer: {answer}")
        print(f"{'=' * 60}")
