import torch
from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelBackendConfig, ModelChatConfig, ModelConfig, ModelFrontendConfig

default_chat_config = ModelChatConfig(
    max_session_length=100000,
    sep_token="###",
    stop_token="###",
    extra_stop_sequences=["</s>"],
    generation_params=dict(do_sample=1, temperature=0.7, top_p=0.9),
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
            ModelChatConfig(
                max_session_length=8192,
                generation_params=dict(do_sample=1, temperature=0.4, top_p=0.9, repetition_penalty=1.17),
            ),
            default_chat_config,
        ),
    ],
    "Falcon": [
        ModelConfig(
            ModelBackendConfig(repository="tiiuae/falcon-180B-chat", public_api=False),
            ModelFrontendConfig(
                name="Falcon 180B-Chat",
                model_card="https://huggingface.co/tiiuae/falcon-180B-chat",
                license="https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt",
            ),
            ModelChatConfig(
                max_session_length=8192,
                sep_token="\n",
                stop_token="\n",
                extra_stop_sequences=["<|endoftext|>", "\nFalcon:", " Falcon:", "\nUser:", " User:", "###"],
                generation_params=dict(do_sample=1, temperature=0.75, top_p=0.9, repetition_penalty=1.2),
            ),
        ),
    ],
}

# INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
INITIAL_PEERS = ['/ip4/49.194.167.186/tcp/31337/p2p/QmcTB9tM37HrKWDaCKEDGtbiGYEpN8mqNcBiNnoF929wWM']

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
