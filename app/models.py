from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelRuntime:
    model_name: str
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    device: str = "cpu"
    loaded: bool = False

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        self.loaded = True
        return self


@dataclass
class ZaynModel:
    runtime: ModelRuntime = field(default_factory=lambda: ModelRuntime(model_name="Qwen/Qwen2.5-7B-Instruct"))

    def load(self):
        self.runtime.load()
        return self
