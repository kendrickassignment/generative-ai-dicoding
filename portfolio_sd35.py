"""Optional SD 3.5 portfolio runner; excluded from Dicoding submission ZIP.

Accept the model's access terms in Hugging Face yourself, then use HF_TOKEN.
Example: python portfolio_sd35.py --prompt "A futuristic city at night"
"""
import argparse
import json
import os
from pathlib import Path


def main():
    import torch
    from diffusers import StableDiffusion3Pipeline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="A futuristic cyberpunk city at night, neon lights, flying cars, detailed digital art")
    parser.add_argument("--negative-prompt", default="blurry, low quality, watermark")
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--output", default="artifacts/sd35.png")
    parser.add_argument("--low-memory", action="store_true", help="Omit T5; changes prompt encoding quality")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Jalankan SD 3.5 di runtime GPU. CPU laptop tidak memadai untuk alur ini.")
    if not args.prompt.strip() or not 1 <= args.steps <= 100 or not 1 <= args.cfg <= 20:
        raise ValueError("Periksa prompt, steps (1–100), dan CFG (1–20).")
    if not 0 <= args.seed <= 2**32-1:
        raise ValueError("Seed harus uint32.")
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    options = dict(torch_dtype=dtype, token=os.environ.get("HF_TOKEN"))
    if args.low_memory:
        options.update(text_encoder_3=None, tokenizer_3=None)
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **options)
    # SD3.5 uses its native flow-matching scheduler, never SD1.5 schedulers.
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    with torch.inference_mode():
        image = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt,
                     num_inference_steps=args.steps, guidance_scale=args.cfg,
                     width=1024, height=1024, max_sequence_length=256,
                     generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    path.with_suffix(".json").write_text(json.dumps(dict(vars(args), model=model_id,
        scheduler=type(pipe.scheduler).__name__), indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
