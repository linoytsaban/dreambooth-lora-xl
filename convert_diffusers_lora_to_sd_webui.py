from pathlib import Path
from diffusers import StableDiffusionXLPipeline
import torch
from safetensors.torch import save_file

# text_encoder.text_model.encoder.layers.0.self_attn.k_proj.lora_linear_layer.down.weight
# lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight
# 1. text_encoder -> lora_te, text_encoder_2 -> lora_te2
# 2. map
# 3. .weight -> 2 .alpha -> 1 and replace . -> _
# test:
# 1. lora_te.text_model.encoder.layers.0.self_attn.k_proj.lora_linear_layer.down.weight
# 2. lora_te.text_model.encoder.layers.0.self_attn.k_proj.lora_down.weight
# 2. lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight

# unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor.to_k_lora.down.weight
# lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight
# 1. unet -> lora_unet
# 2. map
# 4. .weight -> 2 .alpha -> 1 and replace . -> _
# test:
# 1. lora_unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor.to_k_lora.down.weight
# 2. lora_unet.down_blocks_1_attentions_0_transformer_blocks_0_attn1.to_k.lora_down.weight
# 4. lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight


pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, local_files_only=True
)
state_dict, _ = pipe.lora_state_dict(
    Path("<your_lora.safetensors>"), local_files_only=True
)

# DIFFUSERS -> webui and OLD_DIFFUSERS -> webui
LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
    "lora_linear_layer.down": "lora_down",
    "lora_linear_layer.up": "lora_up",
}

# PEFT -> webui
LORA_CLIP_MAP_PEFT = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
    "lora_A": "lora_down",
    "lora_B": "lora_up",
}

# (original) OLD_DIFFUSERS -> webui
LORA_UNET_MAP_OLD_DIFFUSERS = {
    "to_q_lora.down": "to_q.lora_down",
    "to_q_lora.up": "to_q.lora_up",
    "to_k_lora.down": "to_k.lora_down",
    "to_k_lora.up": "to_k.lora_up",
    "to_v_lora.down": "to_v.lora_down",
    "to_v_lora.up": "to_v.lora_up",
    "to_out_lora.down": "to_out_0.lora_down",
    "to_out_lora.up": "to_out_0.lora_up",
    "to_q.alpha": "to_q.alpha",
    "to_k.alpha": "to_k.alpha",
    "to_v.alpha": "to_v.alpha",
}

# (new) DIFFUSERS -> webui
LORA_UNET_MAP_DIFFUSERS = {
    "q_proj.lora_linear_layer.down": "to_q.lora_down",
    "q_proj.lora_linear_layer.up": "to_q.lora_up",
    "k_proj.lora_linear_layer.down": "to_k.lora_down",
    "k_proj.lora_linear_layer.up": "to_k.lora_up",
    "v_proj.lora_linear_layer.down": "to_v.lora_down",
    "v_proj.lora_linear_layer.up": "to_v.lora_up",
    "out_proj.lora_linear_layer.down": "to_out_0.lora_down",
    "out_proj.lora_linear_layer.up": "to_out_0.lora_up",
    "to_q.alpha": "to_q.alpha",
    "to_k.alpha": "to_k.alpha",
    "to_v.alpha": "to_v.alpha",

}

# (new) PEFT -> webui
LORA_UNET_MAP_PEFT = {
    "q_proj.lora_A": "to_q.lora_down",
    "q_proj.lora_B": "to_q.lora_up",
    "k_proj.lora_A": "to_k.lora_down",
    "k_proj.lora_B": "to_k.lora_up",
    "v_proj.lora_A": "to_v.lora_down",
    "v_proj.lora_B": "to_v.lora_up",
    "out_proj.lora_A": "to_out_0.lora_down",
    "out_proj.lora_B": "to_out_0.lora_up",
    "to_q.alpha": "to_q.alpha",
    "to_k.alpha": "to_k.alpha",
    "to_v.alpha": "to_v.alpha",
}

# copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/state_dict_utils.py
DIFFUSERS_OLD_TO_DIFFUSERS = {
    ".to_q_lora.up": ".q_proj.lora_linear_layer.up",
    ".to_q_lora.down": ".q_proj.lora_linear_layer.down",
    ".to_k_lora.up": ".k_proj.lora_linear_layer.up",
    ".to_k_lora.down": ".k_proj.lora_linear_layer.down",
    ".to_v_lora.up": ".v_proj.lora_linear_layer.up",
    ".to_v_lora.down": ".v_proj.lora_linear_layer.down",
    ".to_out_lora.up": ".out_proj.lora_linear_layer.up",
    ".to_out_lora.down": ".out_proj.lora_linear_layer.down",
}

DIFFUSERS_TO_PEFT = {
    ".q_proj.lora_linear_layer.up": ".q_proj.lora_B",
    ".q_proj.lora_linear_layer.down": ".q_proj.lora_A",
    ".k_proj.lora_linear_layer.up": ".k_proj.lora_B",
    ".k_proj.lora_linear_layer.down": ".k_proj.lora_A",
    ".v_proj.lora_linear_layer.up": ".v_proj.lora_B",
    ".v_proj.lora_linear_layer.down": ".v_proj.lora_A",
    ".out_proj.lora_linear_layer.up": ".out_proj.lora_B",
    ".out_proj.lora_linear_layer.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

DIFFUSERS_OLD_TO_PEFT = {
    ".to_q_lora.up": ".q_proj.lora_B",
    ".to_q_lora.down": ".q_proj.lora_A",
    ".to_k_lora.up": ".k_proj.lora_B",
    ".to_k_lora.down": ".k_proj.lora_A",
    ".to_v_lora.up": ".v_proj.lora_B",
    ".to_v_lora.down": ".v_proj.lora_A",
    ".to_out_lora.up": ".out_proj.lora_B",
    ".to_out_lora.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

PEFT_TO_DIFFUSERS = {
    ".q_proj.lora_B": ".q_proj.lora_linear_layer.up",
    ".q_proj.lora_A": ".q_proj.lora_linear_layer.down",
    ".k_proj.lora_B": ".k_proj.lora_linear_layer.up",
    ".k_proj.lora_A": ".k_proj.lora_linear_layer.down",
    ".v_proj.lora_B": ".v_proj.lora_linear_layer.up",
    ".v_proj.lora_A": ".v_proj.lora_linear_layer.down",
    ".out_proj.lora_B": ".out_proj.lora_linear_layer.up",
    ".out_proj.lora_A": ".out_proj.lora_linear_layer.down",
    "to_k.lora_A": "to_k.lora.down",
    "to_k.lora_B": "to_k.lora.up",
    "to_q.lora_A": "to_q.lora.down",
    "to_q.lora_B": "to_q.lora.up",
    "to_v.lora_A": "to_v.lora.down",
    "to_v.lora_B": "to_v.lora.up",
    "to_out.0.lora_A": "to_out.0.lora.down",
    "to_out.0.lora_B": "to_out.0.lora.up",
}

KEYS_TO_ALWAYS_REPLACE = {
    ".processor.": ".",
}
#############################################################


webui_lora_state_dict = {}

for k, v in state_dict.items():

    # First, filter out the keys that we always want to replace
    for pattern in KEYS_TO_ALWAYS_REPLACE.keys():
        if pattern in k:
            new_pattern = KEYS_TO_ALWAYS_REPLACE[pattern]
            k = k.replace(pattern, new_pattern)

    is_text_encoder = False
    prefix = k.split(".")[0]
    if prefix == "text_encoder":
        k = k.replace("text_encoder", "lora_te1")
        is_text_encoder = True
    elif prefix == "text_encoder_2":
        k = k.replace("text_encoder_2", "lora_te2")
        is_text_encoder = True
    elif prefix == "unet":
        k = k.replace("unet", "lora_unet")

    if is_text_encoder:
        # PEFT -> webui
        if any("lora_A" in k for k in state_dict.keys()):
            for map_k, map_v in LORA_CLIP_MAP_PEFT.items():
                k = k.replace(map_k, map_v)
        else:
            for map_k, map_v in LORA_CLIP_MAP.items():
                k = k.replace(map_k, map_v)
    else:
        # OLD_DIFFUSERS -> webui
        if any("to_out_lora" in k for k in state_dict.keys()):
            for map_k, map_v in LORA_UNET_MAP_OLD_DIFFUSERS.items():
                k = k.replace(map_k, map_v)
        # PEFT -> webui
        elif any(".lora_A.weight" in k for k in state_dict.keys()):
            for map_k, map_v in LORA_UNET_MAP_PEFT.items():
                k = k.replace(map_k, map_v)
        # DIFFUSERS -> webui
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            for map_k, map_v in LORA_UNET_MAP_DIFFUSERS.items():
                k = k.replace(map_k, map_v)

    keep_dots = 0
    if k.endswith(".alpha"):
        keep_dots = 1
    elif k.endswith(".weight"):
        keep_dots = 2
    parts = k.split(".")
    k = "_".join(parts[:-keep_dots]) + "." + ".".join(parts[-keep_dots:])

    webui_lora_state_dict[k] = v

save_file(webui_lora_state_dict, "<your_lora_for_webui.safetensors>")