

models_info = {
    "gpt2": {"params": "124M"},
    "distilgpt2": {"params": "82M"},
    "tiiuae/falcon-7b-instruct": {"params": "7B"},
    "mosaicml/mpt-7b": {"params": "7B"},
    "EleutherAI/gpt-neo-1.3B": {"params": "1.3B"},
    "EleutherAI/gpt-neo-2.7B": {"params": "2.7B"},
    "facebook/opt-1.3b": {"params": "1.3B"},
    "facebook/opt-6.7b": {"params": "6.7B"},
}

import shutil
from pathlib import Path
from typing import Dict, Optional, Union

from transformers import pipeline
try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None
try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None
from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

DEFAULT_OFFLOAD_DIR = Path(__file__).resolve().parent / "model_offload"


def _ensure_fresh_offload_dir(folder: Path) -> Path:
    """
    Delete any leftover offload cache and recreate a clean directory.
    """
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def _normalize_pipeline_device(device: Union[int, "torch.device", str, None]) -> Optional[int]:
    """
    Convertit les differentes representations de device en un format compatible
    avec transformers.pipeline. Retourne None si aucun device explicite n'est requis.
    """
    if device is None:
        return None

    if torch is None:
        return device

    if isinstance(device, torch.device):
        if device.type == "cuda":
            return 0 if device.index is None else device.index
        return -1

    if isinstance(device, str):
        if device.startswith("cuda"):
            suffix = device.replace("cuda", "").replace(":", "")
            if suffix:
                try:
                    return int(suffix)
                except ValueError:
                    return 0
            return 0
        if device.startswith("cpu"):
            return -1
        return None

    return device


def _model_uses_accelerate(model_kwargs: Optional[Dict]) -> bool:
    """
    Detecte si le chargement du modele s'appuie sur accelerate/device_map.
    Dans ce cas, la pipeline ne doit pas recevoir de parametre `device`.
    """
    if not isinstance(model_kwargs, dict):
        return False

    if model_kwargs.get("quantization_config") is not None:
        return True

    device_map = model_kwargs.get("device_map")
    if device_map not in (None, "auto"):
        return True

    if model_kwargs.get("llm_int8_enable_fp32_cpu_offload"):
        return True

    if model_kwargs.get("offload_folder"):
        return True

    return False


# ==========================================================
# 🔹 Classe de base pour tous les LLMs
# ==========================================================
class BaseLLM:
    def __init__(self, model_name: str, device: Union[int, "torch.device", str] = -1, task: str = "text-generation", pipeline_kwargs=None, generation_defaults=None, **kwargs):
        """
        :param model_name: Nom du modèle HF (ex: 'Qwen/Qwen2-1.5B-Instruct')
        :param device: 0 pour GPU, -1 pour CPU (ou torch.device/'cuda:X')
        :param task: 'text-generation' ou 'text2text-generation'
        """
        self.model_name = model_name
        self.pipeline_kwargs = pipeline_kwargs or {}
        self.generation_defaults = generation_defaults or {}
        normalized_device = _normalize_pipeline_device(device)
        model_kwargs_param = kwargs.get("model_kwargs")
        uses_accelerate = _model_uses_accelerate(model_kwargs_param)
        if normalized_device is not None and not uses_accelerate:
            self.pipeline_kwargs.setdefault("device", normalized_device)
        self.generator = pipeline(
            task,
            model=model_name,
            **self.pipeline_kwargs,
            **kwargs
        )
        self.llm = HuggingFacePipeline(pipeline=self.generator)


    def get_pipeline(self):
        return self.llm
    

    def generate(self, prompt: str, max_new_tokens: int = 500, **gen_kwargs):
        cfg = {**self.generation_defaults, **gen_kwargs}
        if "max_new_tokens" not in cfg:
            cfg["max_new_tokens"] = max_new_tokens
        output = self.generator(prompt, **cfg)
        # output = self.generator(prompt, max_new_tokens=max_new_tokens, **gen_kwargs)
        return output[0]["generated_text"] if "generated_text" in output[0] else output[0]["summary_text"]


def _prepare_quant_kwargs(
    quantized: bool,
    device: Union[int, "torch.device", str],
    *,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    offload_folder: Optional[Union[str, Path]] = None,
    reset_offload_cache: bool = False,
) -> Dict:
    """
    Construit les kwargs de quantification basés sur BitsAndBytes.
    """
    if not quantized:
        return {}

    # if device < 0:
    #     print("[WARN] Option 'quantized' ignorée (GPU requis).")
    #     return {}

    if BitsAndBytesConfig is None:
        raise ImportError(
            "BitsAndBytesConfig indisponible. Installez 'bitsandbytes' et 'accelerate' pour utiliser quantized=True."
        )

    quant_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        **(
            {"llm_int8_enable_fp32_cpu_offload": True}
            if load_in_8bit
            else {}
        ),
        **(
            {
                "bnb_4bit_compute_dtype": getattr(torch, "float16", None),
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
            if load_in_4bit and torch is not None
            else {}
        ),
    )

    quant_kwargs: Dict = {
        "quantization_config": quant_config,
        "device_map": "auto",
    }

    if torch is not None:
        quant_kwargs.setdefault("torch_dtype", getattr(torch, "float16", None))

    if offload_folder:
        offload_path = Path(offload_folder)
        if reset_offload_cache:
            offload_path = _ensure_fresh_offload_dir(offload_path)
        else:
            offload_path.mkdir(parents=True, exist_ok=True)
        quant_kwargs["offload_folder"] = str(offload_path)
    elif load_in_8bit or load_in_4bit:
        offload_path = _ensure_fresh_offload_dir(DEFAULT_OFFLOAD_DIR)
        quant_kwargs["offload_folder"] = str(offload_path)

    quant_kwargs.setdefault("low_cpu_mem_usage", True)
    quant_kwargs.setdefault("tie_word_embeddings", False)

    normalized_device = _normalize_pipeline_device(device)
    if normalized_device == -1:
        quant_kwargs.setdefault("device_map", "cpu")

    return quant_kwargs


# ==========================================================
# 🧠 Qwen2-1.5B (excellent multilingue, très efficace RAG)
# ==========================================================
class QwenLLM(BaseLLM):
    def __init__(
        self,
        device: int = -1,
        quantized: bool = False,
        pipeline_kwargs=None,
        generation_defaults=None,
        **kwargs,
    ):
        model_kwargs = kwargs.pop("model_kwargs", {})
        reset_cache = kwargs.pop("reset_offload_cache", True)
        offload_target = model_kwargs.get("offload_folder")
        model_kwargs.update(
            _prepare_quant_kwargs(
                quantized,
                device,
                load_in_4bit=True,
                offload_folder=offload_target,
                reset_offload_cache=reset_cache,
            )
        )

        super().__init__(
            "Qwen/Qwen2-1.5B-Instruct",
            device=device,
            task="text-generation",
            pipeline_kwargs=pipeline_kwargs,
            generation_defaults=generation_defaults,
            model_kwargs=model_kwargs,
            **kwargs,
        )

# ==========================================================
# 🌍 Salamandra-2B (multilingue européen, très bon français)
# ==========================================================
class SalamandraLLM(BaseLLM):
    def __init__(
        self,
        device: int = -1,
        quantized: bool = False,
        pipeline_kwargs=None,
        generation_defaults=None,
        **kwargs,
    ):
        model_kwargs = kwargs.pop("model_kwargs", {})
        reset_cache = kwargs.pop("reset_offload_cache", True)
        offload_target = model_kwargs.get("offload_folder")
        model_kwargs.update(
            _prepare_quant_kwargs(
                quantized,
                device,
                load_in_4bit=True,
                offload_folder=offload_target,
                reset_offload_cache=reset_cache,
            )
        )

        super().__init__(
            "BSC-LT/salamandra-2b-instruct",
            device=device,
            task="text-generation",
            pipeline_kwargs=pipeline_kwargs,
            generation_defaults=generation_defaults,
            model_kwargs=model_kwargs,
            **kwargs,
        )

# ==========================================================
# 🔤 BLOOMZ-1.7B (multilingue, bon généraliste)
# ==========================================================
class BloomzLLM(BaseLLM):
    def __init__(self, device: int = -1):
        super().__init__("bigscience/bloomz-1b7", device=device, task="text-generation")


# ==========================================================
# 🧩 mT0-base (T5 multilingue, adapté résumé/Q-R)
# ==========================================================
class MT0LLM(BaseLLM):
    def __init__(self, device: int = -1):
        super().__init__("bigscience/mt0-base", device=device, task="text2text-generation")


# ==========================================================
# ⚡ FLAN-T5 Large (petit, rapide, idéal résumé)
# ==========================================================
class FlanT5LLM(BaseLLM):
    def __init__(self, device: int = -1):
        super().__init__("google/flan-t5-large", device=device, task="text2text-generation")


# ==========================================================
# ⚡ LLaMA 2 7B (bon généraliste, nécessite GPU 16Go+)          ######### Attention aux droits d'utilisation #########
# ==========================================================
class LLaMA2LLM(BaseLLM):
    def __init__(
        self,
        device: int = -1,
        quantized: bool = False,
        pipeline_kwargs=None,
        generation_defaults=None,
        **kwargs,
    ):
        model_kwargs = kwargs.pop("model_kwargs", {})
        reset_cache = kwargs.pop("reset_offload_cache", True)
        offload_target = model_kwargs.get("offload_folder")
        model_kwargs.update(
            _prepare_quant_kwargs(
                quantized,
                device,
                load_in_8bit=True,
                offload_folder=offload_target,
                reset_offload_cache=reset_cache,
            )
        )

        super().__init__(
            "meta-llama/Llama-2-7b-chat-hf",
            device=device,
            task="text-generation",
            pipeline_kwargs=pipeline_kwargs,
            generation_defaults=generation_defaults,
            model_kwargs=model_kwargs,
            **kwargs,
        )

# ==========================================================
# ⚡ LLaMA 2 7B (bon généraliste, nécessite GPU 16Go+)
# ==========================================================
class Mistral7BLLM(BaseLLM):
    def __init__(
        self,
        device: int = -1,
        quantized: bool = False,
        pipeline_kwargs=None,
        generation_defaults=None,
        **kwargs,
    ):
        model_kwargs = kwargs.pop("model_kwargs", {})
        reset_cache = kwargs.pop("reset_offload_cache", True)
        offload_target = model_kwargs.get("offload_folder")
        model_kwargs.update(
            _prepare_quant_kwargs(
                quantized,
                device,
                load_in_8bit=True,
                offload_folder=offload_target,
                reset_offload_cache=reset_cache,
            )
        )

        super().__init__(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device=device,
            task="text-generation",
            pipeline_kwargs=pipeline_kwargs,
            generation_defaults=generation_defaults,
            model_kwargs=model_kwargs,
            **kwargs,
        )

# ==========================================================
# 🧪 Exemple d'utilisation
# ==========================================================

if __name__ == "__main__":

    import utilisation_GPU as test_GPU
    device = test_GPU.test_utilisation_GPU()

    llm = Mistral7BLLM(device=device, quantized=True)

    print("\n Tapez 'exit' pour quitter.")
    while True:
        prompt = input("\nVous: ")
        if prompt.lower() == "exit":
            break
        response = llm.generate(prompt)
        print("\nLLM:", response)
