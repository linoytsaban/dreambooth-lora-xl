from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.utils import state_dict_utils
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


# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, local_files_only=True
# )
# state_dict, _ = pipe.lora_state_dict(
#     Path("<your_lora.safetensors>"), local_files_only=True
# )

pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")

state_dict,_  = pipe.lora_state_dict("multimodalart/medieval-animals",
                                    weight_name="pytorch_lora_weights.safetensors"
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


#############################################################

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
save_file(webui_lora_state_dict, "<your_lora_for_webui.safetensors>")