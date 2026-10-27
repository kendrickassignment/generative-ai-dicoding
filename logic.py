"""Generative Image Suite — Kendrick Filbert.

SD 1.5 submission backend. Model imports are lazy so geometry tests need no GPU.
White mask pixels are editable; black pixels are preserved exactly.
"""
import gc
import threading
from contextlib import contextmanager
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
PROMPT = ("2D digital illustration of a cute astronaut in a white spacesuit, "
          "standing on the moon, full body on the left, Earth in the starry sky, "
          "empty lunar ground on the right, clean outlines, colorful digital art")
NEGATIVE_PROMPT = ("photorealistic, realistic, photograph, 3d render, messy, blurry, "
                   "low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration")
SCHEDULERS = ("Euler A", "DPM++", "DDIM")
INFERENCE_LOCK = threading.RLock()


def flush_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _configure(pipe):
    import torch
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    # Keep weights on CPU between calls; activate only the current pipeline.
    if not torch.cuda.is_available():
        pipe.to("cpu")
    return pipe


@lru_cache(maxsize=2)
def get_pipeline(task="generation"):
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
    if task not in ("generation", "inpainting"):
        raise ValueError("Task harus generation atau inpainting.")
    cls = StableDiffusionPipeline if task == "generation" else StableDiffusionInpaintPipeline
    model = BASE_MODEL if task == "generation" else INPAINT_MODEL
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = cls.from_pretrained(model, torch_dtype=dtype, use_safetensors=True)
    return _configure(pipe)


def load_models_cached():
    """Compatibility with the template. Prefer lazy get_pipeline in the UI."""
    return get_pipeline("generation"), get_pipeline("inpainting")


@contextmanager
def active_pipeline(pipe):
    import torch
    with INFERENCE_LOCK:
        try:
            if torch.cuda.is_available():
                pipe.to("cuda")
            yield pipe
        finally:
            if torch.cuda.is_available():
                pipe.to("cpu")
            flush_memory()


def load_scheduler(pipe, scheduler_name):
    from diffusers import (EulerAncestralDiscreteScheduler,
                           DPMSolverMultistepScheduler, DDIMScheduler)
    classes = {"Euler A": EulerAncestralDiscreteScheduler,
               "DPM++": DPMSolverMultistepScheduler, "DDIM": DDIMScheduler}
    if scheduler_name not in classes:
        raise ValueError(f"Scheduler tidak dikenal: {scheduler_name}")
    if not hasattr(pipe, "_original_scheduler_config"):
        pipe._original_scheduler_config = dict(pipe.scheduler.config)
    options = {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True} if scheduler_name == "DPM++" else {}
    pipe.scheduler = classes[scheduler_name].from_config(pipe._original_scheduler_config, **options)
    return pipe


set_scheduler = load_scheduler


def _validate(prompt, seed, steps, cfg, num_images=1):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt tidak boleh kosong.")
    if not 0 <= int(seed) <= 2**32 - num_images:
        raise ValueError("Seed di luar rentang uint32.")
    if not 1 <= int(steps) <= 100 or not 1 <= int(num_images) <= 4:
        raise ValueError("Steps 1–100 dan jumlah gambar 1–4.")
    if not 1 <= float(cfg) <= 20:
        raise ValueError("CFG harus 1–20.")


def _check_tokens(pipe, prompt):
    tokens = pipe.tokenizer(prompt, truncation=False).input_ids
    if len(tokens) > pipe.tokenizer.model_max_length:
        raise ValueError("Prompt terlalu panjang untuk CLIP. Ringkas menjadi maksimal 75 token isi.")


def generate_image(pipe, prompt, neg_prompt, seed, steps, cfg,
                   num_images=1, scheduler_name="Euler A"):
    import torch
    _validate(prompt, seed, steps, cfg, num_images)
    _check_tokens(pipe, prompt)
    _check_tokens(pipe, neg_prompt)
    # One call, true batch inference; a fresh generator per output.
    generators = [torch.Generator(device="cpu").manual_seed(int(seed) + i)
                  for i in range(int(num_images))]
    with active_pipeline(pipe), torch.inference_mode():
        load_scheduler(pipe, scheduler_name)
        output = pipe(prompt=prompt, negative_prompt=neg_prompt,
                      generator=generators, num_images_per_prompt=int(num_images),
                      num_inference_steps=int(steps), guidance_scale=float(cfg),
                      width=512, height=512)
    if any(getattr(output, "nsfw_content_detected", None) or []):
        raise ValueError("Safety checker memblokir salah satu hasil; ubah prompt.")
    return output.images


def generate_simple_image(prompt, negative_prompt=NEGATIVE_PROMPT, seed=222):
    return generate_image(get_pipeline(), prompt, negative_prompt, seed, 30, 7.5)[0]


def generate_advanced_image(prompt, negative_prompt=NEGATIVE_PROMPT, seed=222,
                            guidance_scale=9.0, num_inference_step=40):
    return generate_image(get_pipeline(), prompt, negative_prompt, seed,
                          num_inference_step, guidance_scale)[0]


def manual_mask(image, box=(285, 230, 485, 450)):
    """Hardcoded starting rectangle for 512px output; adjust after visual review."""
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def normalize_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if w < 64 or h < 64 or max(w, h) > 1536:
        raise ValueError("Ukuran gambar harus 64–1536 piksel per sisi.")
    if w % 8 or h % 8:
        raise ValueError("Ukuran gambar harus kelipatan 8; jangan merusak geometri mask.")
    return image


def run_inpainting(pipe, image, mask, prompt, strength=1.0, seed=9,
                   steps=40, cfg=9.0, negative_prompt=NEGATIVE_PROMPT,
                   feather=4):
    import torch
    _validate(prompt, seed, steps, cfg)
    if not 0 < strength <= 1 or int(steps * strength) < 1:
        raise ValueError("Strength harus menghasilkan minimal satu denoising step.")
    _check_tokens(pipe, prompt)
    image = normalize_image(image)
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask = mask.point(lambda p: 255 if p >= 128 else 0)
    if mask.getbbox() is None:
        raise ValueError("Mask kosong: tandai area yang akan diedit.")
    with active_pipeline(pipe), torch.inference_mode():
        load_scheduler(pipe, "DPM++")
        output = pipe(prompt=prompt, negative_prompt=negative_prompt,
                      image=image, mask_image=mask, width=image.width, height=image.height,
                      strength=float(strength), guidance_scale=float(cfg),
                      num_inference_steps=int(steps),
                      generator=torch.Generator("cpu").manual_seed(int(seed)))
    if any(getattr(output, "nsfw_content_detected", None) or []):
        raise ValueError("Safety checker memblokir hasil edit.")
    # Blur only inside editable pixels, preserving every black-mask pixel.
    alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(feather))).copy()
    alpha[np.asarray(mask) == 0] = 0
    return Image.composite(output.images[0], image, Image.fromarray(alpha))


def inpaint_engine(image, mask, prompt, **kwargs):
    return run_inpainting(get_pipeline("inpainting"), image, mask, prompt, **kwargs)


def prepare_outpainting(image, expand_pixels=128, direction="all", overlap=24):
    """Expand geometrically; overlap lets diffusion blend the original boundary."""
    image = normalize_image(image)
    if direction not in ("left", "right", "top", "bottom", "all"):
        raise ValueError("Arah perluasan tidak valid.")
    if expand_pixels < 8 or expand_pixels % 8:
        raise ValueError("Ekspansi harus kelipatan 8 dan minimal 8 piksel.")
    if not 0 <= overlap < min(image.size) // 2:
        raise ValueError("Overlap terlalu besar.")
    left = expand_pixels if direction in ("left", "all") else 0
    right = expand_pixels if direction in ("right", "all") else 0
    top = expand_pixels if direction in ("top", "all") else 0
    bottom = expand_pixels if direction in ("bottom", "all") else 0
    w, h = image.size
    if max(w + left + right, h + top + bottom) > 1536:
        raise ValueError("Kanvas melebihi batas 1536px; gunakan gambar lebih kecil.")
    # Edge extension avoids a solid-color border interpreted as a picture frame.
    pixels = np.pad(np.asarray(image), ((top, bottom), (left, right), (0, 0)), mode="edge")
    canvas = Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(18))
    canvas.paste(image, (left, top))
    mask = Image.new("L", canvas.size, 255)
    keep = (left + (overlap if left else 0), top + (overlap if top else 0),
            left + w - (overlap if right else 0), top + h - (overlap if bottom else 0))
    # PIL rectangle includes the final coordinate; subtract one for exclusive bounds.
    ImageDraw.Draw(mask).rectangle((keep[0], keep[1], keep[2]-1, keep[3]-1), fill=0)
    return canvas, mask


@lru_cache(maxsize=1)
def load_segmentation():
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    model_id = "CIDAS/clipseg-rd64-refined"
    return (CLIPSegProcessor.from_pretrained(model_id),
            CLIPSegForImageSegmentation.from_pretrained(model_id).eval())


def segmentation_mask(image, text="astronaut", threshold=0.45):
    import torch
    if not 0 < threshold < 1:
        raise ValueError("Threshold harus antara 0 dan 1.")
    processor, model = load_segmentation()
    inputs = processor(text=[text], images=[image.convert("RGB")],
                       padding=True, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits.reshape(1, 1, 352, 352)
        probabilities = torch.nn.functional.interpolate(
            logits, size=(image.height, image.width), mode="bilinear", align_corners=False
        ).sigmoid()[0, 0].numpy()
    mask = Image.fromarray((probabilities >= threshold).astype("uint8") * 255)
    fraction = np.count_nonzero(np.asarray(mask)) / (image.width * image.height)
    if not 0.005 <= fraction <= 0.8:
        raise ValueError(f"Mask segmentasi tidak layak ({fraction:.1%}); sesuaikan teks/threshold.")
    return mask.filter(ImageFilter.MaxFilter(7)), probabilities


def refine_image(image, prompt, negative_prompt=NEGATIVE_PROMPT, seed=222,
                 strength=0.25, steps=40, cfg=7.5):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    base = get_pipeline()
    # Share the base components; never load a second copy of the checkpoint.
    with INFERENCE_LOCK:
        refiner = StableDiffusionImg2ImgPipeline(**base.components)
        _validate(prompt, seed, steps, cfg)
        if not 0 < strength <= 0.4 or int(steps * strength) < 1:
            raise ValueError("Refiner strength harus kecil: 0 < strength <= 0.4.")
        with active_pipeline(refiner), torch.inference_mode():
            load_scheduler(refiner, "DPM++")
            output = refiner(prompt=prompt, negative_prompt=negative_prompt,
                             image=image.convert("RGB"), strength=strength,
                             num_inference_steps=steps, guidance_scale=cfg,
                             generator=torch.Generator("cpu").manual_seed(seed))
        if any(getattr(output, "nsfw_content_detected", None) or []):
            raise ValueError("Safety checker memblokir hasil refiner.")
        return output.images[0]
