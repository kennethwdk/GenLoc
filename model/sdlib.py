import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from peft import LoraConfig

class AttentionStore():
    def __call__(self, attn):
        self.step_store["attn"].append(attn) 

    def reset(self):
        self.step_store = {"attn": [],}

    def __init__(self):
        self.step_store = {
            "attn": [],
        }

def register_attention_control(model, controller, feature_upsample_res=256):
    def ca_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = True
            if encoder_hidden_states is None:
                is_cross = False
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            x = hidden_states
            context = encoder_hidden_states
            if (
                is_cross
                # and x.shape[1] <= 32**2
                # and len(controller.step_store["attn"]) < 4
            ):
                bs, seq_len, dim = x.shape
                x_reshaped = x.reshape(
                    bs,
                    int(seq_len**0.5),
                    int(seq_len**0.5),
                    dim,
                ).permute(0, 3, 1, 2)
                # upsample to feature_upsample_res**2
                x_reshaped = (
                    F.interpolate(
                        x_reshaped,
                        size=(feature_upsample_res, feature_upsample_res),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(bs, -1, dim)
                )

                q = self.to_q(x_reshaped)
                bs, seq_len, dim = q.shape
                # head_size = self.heads
                # q = q.reshape(bs, seq_len, head_size, dim // head_size)[:, :, 0, :]
                # assert q.shape[0] == bs and q.shape[1] == seq_len and q.shape[2] == dim // head_size

                k = self.to_k(context)
                bs, seq_len, dim = k.shape
                # assert seq_len == 1000
                # head_size = self.heads
                # k = k.reshape(bs, seq_len, head_size, dim // head_size)[:, :, 0, :]
                # assert k.shape[0] == bs and k.shape[1] == seq_len and k.shape[2] == dim // head_size

                # sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                # attn = torch.nn.Softmax(dim=-1)(sim)
                # attn = attn.clone()
                # controller(attn)
                sim = torch.einsum("b i d, b j d -> b i j", q, k) * dim**-0.5#self.scale
                # sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn = torch.nn.Softmax(dim=1)(sim)
                # attn2 = torch.nn.Softmax(dim=-1)(sim)
                attn = attn.clone()
                controller(attn)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)
            # # dropout
            # hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count)
        return count
    
    def register_crossattnblocks(net_):
        count = 0
        if net_.__class__.__name__ == "CrossAttnUpBlock2D":
            count += register_recr(net_, 0)
            return count
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count += register_crossattnblocks(net__)
        return count

    cross_att_count = 0
    cross_att_count = register_crossattnblocks(model)
    # print('Find {} attention layers in this model'.format(cross_att_count))
    controller.num_att_layers = cross_att_count
    
    # create assertion with message
    assert cross_att_count != 0, "No cross attention layers found in the model. Please check to make sure you're using diffusers==0.8.0."

def load_ldm(type="CompVis/stable-diffusion-v1-4", feature_upsample_res=256, lora_path=""):
    NUM_DDIM_STEPS = 50

    if "stable-diffusion-xl-base-1.0" in type:
        ldm = DiffusionPipeline.from_pretrained(type)
        ldm.unet.config.addition_embed_type = "null"
        ldm.text_encoder = None
        ldm.text_encoder_2 = None
        ldm.tokenizer = None
        ldm.tokenizer_2 = None
    else:
        ldm = StableDiffusionPipeline.from_pretrained(type)
        ldm.text_encoder = None
        ldm.tokenizer = None

    ldm.scheduler.set_timesteps(NUM_DDIM_STEPS)

    for param in ldm.vae.parameters():
        param.requires_grad = False
    for param in ldm.unet.parameters():
        param.requires_grad = False
    
    if lora_path:
        ldm.load_lora_weights(lora_path)
    else:
        unet_lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        ldm.unet.add_adapter(unet_lora_config)
    
    controller = AttentionStore()

    def hook_fn(module, input):
        _device = input[0].device
        # if device not in patched_devices:
        register_attention_control(module, controller, feature_upsample_res=feature_upsample_res)

    ldm.unet.register_forward_pre_hook(hook_fn)

    return ldm, controller

def image2latent(model, image):
    image = image * 2 - 1
    latents = model.encode(image)["latent_dist"].mean
    latents = latents * 0.18215
    return latents

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def collect_maps(
    controller,
    layers=[0, 1, 2, 3, 4, 5, 6, 7],
):
    """
    returns the bilinearly upsampled attention map of size upsample_res x upsample_res for the first word in the prompt
    """

    attention_maps = controller.step_store['attn']
    
    attention_maps_list = []

    layer_overall = -1

    for layer in range(len(attention_maps)):
        layer_overall += 1

        if layer_overall not in layers:
            continue
        
        data = attention_maps[layer]

        data = data.reshape(
            data.shape[0], int(data.shape[1] ** 0.5), int(data.shape[1] ** 0.5), data.shape[2]
        )
        
        data = data.permute(0, 3, 1, 2)

        attention_maps_list.append(data)

    attention_maps = torch.stack(attention_maps_list, dim=0)
    attention_maps = attention_maps.mean(dim=0)

    controller.reset()

    return attention_maps