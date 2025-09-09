from dataclasses import dataclass
from typing import List, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline


@dataclass
class GenerationConfig:
    model_name: str = "google/flan-t5-small"
    device: str = "cpu"  # "cpu" or "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    use_fp16: bool = False


class SimpleGenerator:
    """Tiny wrapper around HF transformers for RAG answer generation.

    Chooses text2text-generation for seq2seq models (e.g., T5/FLAN), otherwise text-generation.
    Builds a compact instruction prompt using retrieved context.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        name = self.config.model_name

        # Detect task
        task = "text2text-generation" if any(tag in name.lower() for tag in ["t5", "flan"]) else "text-generation"

        kwargs = {}
        if self.config.device == "cuda" and torch.cuda.is_available():
            kwargs["device"] = 0
            if self.config.use_fp16:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["device"] = -1

        self.logger.info(f"Loading generator model: {name} (task={task})")
        self._pipe = pipeline(
            task,
            model=name,
            tokenizer=name,
            **kwargs,
        )

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        ctx_join = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts) if c.strip()])
        instruction = (
            "You are a helpful assistant. Answer the question using ONLY the provided contexts. "
            "If the answer is not in the contexts, say you don't know. Be concise.\n\n"
        )
        prompt = f"{instruction}{ctx_join}\n\nQuestion: {question}\nAnswer:"
        return prompt

    def generate(self, question: str, contexts: List[str]) -> str:
        self._ensure_loaded()

        prompt = self.build_prompt(question, contexts)
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.temperature > 0,
        }

        outputs = self._pipe(prompt, **gen_kwargs)
        if isinstance(outputs, list) and outputs:
            out = outputs[0]
            # text2text pipe returns {"generated_text": ...}; text-generation returns {"generated_text": full_prompt+continuation}
            text = out.get("generated_text", "")
            if not text:
                # Some pipelines nest differently
                text = out.get("summary_text", "") or out.get("text", "")
            # If the pipe included the prompt, try to strip it
            if text.startswith(prompt):
                text = text[len(prompt):].lstrip()
            return text.strip()
        return ""
