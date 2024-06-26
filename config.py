import torch
from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelBackendConfig, ModelChatConfig, ModelConfig, ModelFrontendConfig

default_chat_config = ModelChatConfig(
    max_session_length=100000,
    sep_token="###",
    stop_token="###",
    extra_stop_sequences=["</s>"],
    generation_params=dict(do_sample=1, temperature=0.4, top_p=0.9, repetition_penalty=1.17),
)

MODEL_FAMILIES = {
    "Llama 2": [
        ModelConfig(
            ModelBackendConfig(repository="petals-team/StableBeluga2", aliases=["stabilityai/StableBeluga2"]),
            ModelFrontendConfig(
                name="Stable Beluga 2 (70B)",
                model_card="https://huggingface.co/stabilityai/StableBeluga2",
                license="https://huggingface.co/stabilityai/StableBeluga2/blob/main/LICENSE.txt",
            ),
            default_chat_config,
        ),
        ModelConfig(
            ModelBackendConfig(repository="codellama/CodeLlama-34b-Instruct-hf", aliases=["codellama/CodeLlama-34b-Instruct-hf"]),
            ModelFrontendConfig(
                name="Code Llama (34B)",
                model_card="https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf",
                license="https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/blob/main/LICENSE.txt",
            ),
            default_chat_config,
        ),
    ],
}

# INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
# --initial_peers /ip4/49.194.167.186/tcp/31337/p2p/QmNbWmdMF4mrHYBZvaX1aWmEHw1i6Sh7E1sVZdqE1LFbFm
INITIAL_PEERS = ['/dns/ai.gettingitalldone.com/tcp/31337/p2p/QmTbqDrBxCioZMYCjTUHu5GLVERw369VkY7fBTMiKFFDXu']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from cpufeature import CPUFeature

    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
