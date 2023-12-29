from pathlib import Path
from diffusers.utils import state_dict_utils
import torch
from safetensors.torch import load_file, save_file

state_dict = load_file("animals.safetensors")

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


# (new) DIFFUSERS -> webui
LORA_UNET_MAP = {
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

# LORA_UNET_MAP = {
#     ".to_q_lora.down": "to_q.lora_down",
#     ".to_q_lora.up": "to_q.lora_up",
#     ".to_k_lora.down": "to_k.lora_down",
#     ".to_k_lora.up": "to_k.lora_up",
#     ".to_v_lora.down": "to_v.lora_down",
#     ".to_v_lora.up": "to_v.lora_up",
#     ".to_out_lora.down": "to_out_0.lora_down",
#     ".to_out_lora.up": "to_out_0.lora_up",
#     ".to_q.alpha": "to_q.alpha",
#     ".to_k.alpha": "to_k.alpha",
#     ".to_v.alpha": "to_v.alpha",
# }

# intermediate dict to convert from the current format of the lora to DIFFUSERS
# we will then convert from DIFFUSERS to webui
diffusers_state_dict = state_dict_utils.convert_state_dict_to_diffusers(state_dict)
with open('test_conversion_script_peft.txt', 'w') as f:
    for line in list(diffusers_state_dict.keys()):
        f.write(f"{line}\n")

webui_lora_state_dict = {}

for k, v in diffusers_state_dict.items():
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
        for map_k, map_v in LORA_CLIP_MAP.items():
            k = k.replace(map_k, map_v)
    else:
        for map_k, map_v in LORA_UNET_MAP.items():
            k = k.replace(map_k, map_v)

    keep_dots = 0
    if k.endswith(".alpha"):
        keep_dots = 1
    elif k.endswith(".weight"):
        keep_dots = 2
    parts = k.split(".")
    k = "_".join(parts[:-keep_dots]) + "." + ".".join(parts[-keep_dots:])

    webui_lora_state_dict[k] = v

with open('test_conversion_script_webui_not_peft.txt', 'w') as f:
    for line in list(webui_lora_state_dict.keys()):
        f.write(f"{line}\n")
save_file(webui_lora_state_dict, "diffusers_animals.safetensors")