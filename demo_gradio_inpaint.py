from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import cv2

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan, sample_hunyuan_inpaint
from diffusers_helper.memory import cpu, gpu, GPU_TYPE, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

print(args)

print(f'GPU Type: {GPU_TYPE}')
print(f'GPU Device: {gpu}')

if GPU_TYPE == 'cuda':
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60
    print(f'Free VRAM {free_mem_gb} GB')
elif GPU_TYPE == 'mps':
    # MPS uses unified memory, conservative mode recommended
    free_mem_gb = 16.0  # Assume reasonable unified memory
    high_vram = False  # Use memory-efficient mode on MPS
    print(f'MPS unified memory mode (conservative)')
else:
    free_mem_gb = 0
    high_vram = False
    print('No GPU available, using CPU')

print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

# MPS has limited bfloat16 support, use float16 instead
_transformer_load_dtype = torch.float16 if (torch.backends.mps.is_available() and not torch.cuda.is_available()) else torch.bfloat16
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=_transformer_load_dtype).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# MPS has limited bfloat16 support, use float16 instead
if GPU_TYPE == 'mps':
    transformer_dtype = torch.float16
    print('Using float16 for transformer (MPS compatibility)')
else:
    transformer_dtype = torch.bfloat16
    print('Using bfloat16 for transformer')

transformer.to(dtype=transformer_dtype)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


def load_video_frames(video_path, target_width, target_height, max_frames=None):
    """Load video frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize and center crop
        frame = resize_and_center_crop(frame, target_width=target_width, target_height=target_height)
        frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    return frames


def resize_and_center_crop_grayscale(image, target_width, target_height):
    """Resize and center crop a grayscale image."""
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def load_mask_frames(video_path, target_width, target_height, num_frames):
    """Load mask video frames and convert to binary mask (white=1, black=0)."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            # If mask video is shorter, repeat the last frame
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                # If no frames at all, create a white mask (inpaint everything)
                frames.append(np.ones((target_height, target_width), dtype=np.float32))
            continue

        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize and center crop using grayscale-compatible function
        frame = resize_and_center_crop_grayscale(frame, target_width=target_width, target_height=target_height)

        # Normalize to [0, 1] - white (255) becomes 1, black (0) becomes 0
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return frames[:num_frames]


def frames_to_latent_mask(mask_frames, latent_height, latent_width):
    """Convert frame-space masks to latent-space masks.

    The VAE compresses spatial dimensions by 8x and temporal by 4x.
    """
    num_frames = len(mask_frames)
    # Temporal compression: 4x, so we take every 4th frame or average
    num_latent_frames = (num_frames + 3) // 4

    latent_masks = []
    for i in range(num_latent_frames):
        # Average masks for frames that compress to this latent frame
        start_idx = i * 4
        end_idx = min(start_idx + 4, num_frames)
        frame_masks = mask_frames[start_idx:end_idx]

        # Average the masks
        avg_mask = np.mean(frame_masks, axis=0)

        # Resize to latent resolution (8x spatial compression)
        latent_mask = cv2.resize(avg_mask, (latent_width, latent_height), interpolation=cv2.INTER_AREA)
        latent_masks.append(latent_mask)

    # Stack to [T, H, W] and convert to tensor [1, 1, T, H, W]
    latent_mask_np = np.stack(latent_masks, axis=0)
    latent_mask = torch.from_numpy(latent_mask_np).float()
    latent_mask = latent_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]

    return latent_mask


@torch.no_grad()
def worker(input_video, mask_video, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, inpaint_strength):
    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Load input video frames
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Loading video frames ...'))))

        # First pass to get video info
        cap = cv2.VideoCapture(input_video)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Find nearest bucket for resolution
        height, width = find_nearest_bucket(original_height, original_width, resolution=640)

        # Load video frames
        video_frames = load_video_frames(input_video, target_width=width, target_height=height)
        num_frames = len(video_frames)

        print(f'Loaded {num_frames} frames at {width}x{height}')

        # Load mask frames
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Loading mask frames ...'))))
        mask_frames = load_mask_frames(mask_video, target_width=width, target_height=height, num_frames=num_frames)

        # Save first frame for reference
        Image.fromarray(video_frames[0]).save(os.path.join(outputs_folder, f'{job_id}_first_frame.png'))

        # Convert frames to tensor [B, C, T, H, W]
        video_np = np.stack(video_frames, axis=0)  # [T, H, W, C]
        video_pt = torch.from_numpy(video_np).float() / 127.5 - 1  # Normalize to [-1, 1]
        video_pt = video_pt.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        # Encode the video to latent space
        video_latents = vae_encode(video_pt, vae)  # [1, 16, T', H', W']

        latent_height = video_latents.shape[3]
        latent_width = video_latents.shape[4]
        latent_frames = video_latents.shape[2]

        print(f'Video latent shape: {video_latents.shape}')

        # Create latent-space mask
        latent_mask = frames_to_latent_mask(mask_frames, latent_height, latent_width)
        # Ensure mask has correct number of temporal frames
        if latent_mask.shape[2] != latent_frames:
            # Interpolate temporally if needed
            latent_mask = torch.nn.functional.interpolate(
                latent_mask,
                size=(latent_frames, latent_height, latent_width),
                mode='trilinear',
                align_corners=False
            )

        print(f'Latent mask shape: {latent_mask.shape}')

        # CLIP Vision encoding using first frame
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(video_frames[0], feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype conversion
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start inpainting ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)

        # For inpainting, we work with the full video at once
        # Number of pixel frames from latent frames
        num_pixel_frames = latent_frames * 4 - 3

        if not high_vram:
            unload_complete_models()
            move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        if use_teacache:
            transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
        else:
            transformer.initialize_teacache(enable_teacache=False)

        def callback(d):
            preview = d['denoised']
            preview = vae_decode_fake(preview)

            preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                raise KeyboardInterrupt('User ends the task.')

            current_step = d['i'] + 1
            percentage = int(100.0 * current_step / steps)
            hint = f'Inpainting {current_step}/{steps}'
            desc = f'Processing {num_frames} frames at {width}x{height}'
            stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
            return

        # Use inpainting sampler
        generated_latents = sample_hunyuan_inpaint(
            transformer=transformer,
            sampler='unipc',
            width=width,
            height=height,
            frames=num_pixel_frames,
            real_guidance_scale=cfg,
            distilled_guidance_scale=gs,
            guidance_rescale=rs,
            num_inference_steps=steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=gpu,
            dtype=transformer.dtype,
            image_embeddings=image_encoder_last_hidden_state,
            # Inpainting specific
            original_latents=video_latents,
            inpaint_mask=latent_mask,
            inpaint_strength=inpaint_strength,
            callback=callback,
        )

        if not high_vram:
            offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
            load_model_as_complete(vae, target_device=gpu)

        # Decode latents to pixels
        output_pixels = vae_decode(generated_latents, vae).cpu()

        if not high_vram:
            unload_complete_models()

        output_filename = os.path.join(outputs_folder, f'{job_id}_inpainted.mp4')
        save_bcthw_as_mp4(output_pixels, output_filename, fps=30, crf=mp4_crf)

        print(f'Saved inpainted video to {output_filename}')

        stream.output_queue.push(('file', output_filename))

    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_video, mask_video, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, inpaint_strength):
    global stream
    assert input_video is not None, 'No input video!'
    assert mask_video is not None, 'No mask video!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_video, mask_video, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, inpaint_strength)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The scene continues naturally with smooth motion.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack Video Inpainting')
    gr.Markdown('Upload an input video and a mask video. White regions in the mask will be inpainted/regenerated based on the prompt.')
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(sources='upload', label="Input Video", height=320)
            mask_video = gr.Video(sources='upload', label="Mask Video (white=inpaint, black=keep)", height=320)
            prompt = gr.Textbox(label="Prompt", value='The scene continues naturally with smooth motion.')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Inpainting")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but may reduce quality slightly.')
                inpaint_strength = gr.Slider(label="Inpaint Strength", minimum=0.0, maximum=1.0, value=1.0, step=0.01, info='1.0 = full inpainting, lower values blend more with original')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                seed = gr.Number(label="Seed", value=31337, precision=0)

                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed.")

        with gr.Column():
            preview_image = gr.Image(label="Preview", height=200, visible=False)
            result_video = gr.Video(label="Inpainted Video", autoplay=True, show_share_button=False, height=512, loop=True)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    ips = [input_video, mask_video, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, inpaint_strength]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
