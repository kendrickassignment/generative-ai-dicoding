"""Streamlit interface adapted from the supplied BFGAI template."""
import io
import json
import time
import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas
import logic

st.set_page_config(page_title="Lunar Studio · Kendrick Filbert", page_icon="🌕", layout="wide")
st.title("Lunar Studio")
st.caption("Generative Image Suite · Kendrick Filbert · Stable Diffusion 1.5")

@st.cache_resource
def get_model(task):
    return logic.get_pipeline(task)

def png_bytes(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()

def show_error(exc):
    if "out of memory" in str(exc).lower():
        st.error("Memori GPU tidak cukup. Kurangi batch atau ukuran ekspansi, lalu coba kembali.")
    else:
        st.error(f"Proses belum berhasil: {exc}")

with st.sidebar:
    st.header("Generation settings")
    steps = st.slider("Inference steps", 5, 50, 30)
    cfg = st.slider("Guidance scale (CFG)", 1.0, 20.0, 7.5, 0.5)
    seed = int(st.number_input("Seed", min_value=0, max_value=4294967292, value=222))
    scheduler_name = st.selectbox("Scheduler", logic.SCHEDULERS)
    num_images = st.slider("Jumlah gambar", 1, 4, 1)
    st.caption("Seed tiap gambar = seed awal + indeks. Empat gambar diproses dalam satu batch.")
    if st.button("Clear Memory"):
        logic.flush_memory()
        st.success("Cache alokasi yang tidak dipakai dibersihkan.")

tab_gen, tab_edit = st.tabs(["Generation", "Edit"])
with tab_gen:
    inputs, outputs = st.columns(2, gap="large")
    with inputs:
        st.subheader("Describe your scene")
        with st.form("generation"):
            prompt = st.text_area("Prompt", logic.PROMPT, height=150)
            negative = st.text_area("Negative prompt", logic.NEGATIVE_PROMPT, height=100)
            generate = st.form_submit_button("Generate", type="primary")
        if generate:
            try:
                start = time.perf_counter()
                with st.spinner("Menghasilkan gambar dengan Stable Diffusion…"):
                    images = logic.generate_image(get_model("generation"), prompt,
                        negative, seed, steps, cfg, num_images, scheduler_name)
                st.session_state.generated_images = images
                st.session_state.current_image = images[0]
                st.session_state.canvas_revision = st.session_state.get("canvas_revision", 0) + 1
                st.session_state.generation_meta = dict(model=logic.BASE_MODEL,
                    prompt=prompt, negative_prompt=negative, seed=seed, steps=steps,
                    cfg=cfg, scheduler=scheduler_name, num_images=num_images,
                    seconds=round(time.perf_counter()-start, 2))
            except Exception as exc:
                show_error(exc)
    with outputs:
        st.subheader("Visual output")
        images = st.session_state.get("generated_images", [])
        if images:
            cols = st.columns(2) if len(images) > 1 else [st.container()]
            meta = st.session_state.generation_meta
            for i, image in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(image, caption=f"Gambar {i+1} · seed {meta['seed']+i}", use_column_width=True)
                    if st.button(f"Edit gambar {i+1}", key=f"select_{i}"):
                        st.session_state.current_image = image
                        st.session_state.canvas_revision = st.session_state.get("canvas_revision", 0) + 1
                        st.success("Gambar dipilih. Buka tab Edit.")
                    st.download_button("Download PNG", png_bytes(image),
                        file_name=f"generation_{i+1}.png", mime="image/png", key=f"download_{i}")
            st.caption(f"Inference selesai dalam {meta['seconds']} detik.")
            st.download_button("Download parameter", json.dumps(meta, indent=2),
                               "generation_metadata.json", "application/json")
        else:
            st.info("Isi prompt lalu klik Generate. Hasil ditampilkan di sini.")

with tab_edit:
    source = st.session_state.get("current_image")
    if source is None:
        st.info("Buat gambar pada tab Generation terlebih dahulu.")
    else:
        mode = st.radio("Mode edit", ["Inpainting", "Outpainting / Zoom Out"], horizontal=True)
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Gambar sumber")
            if mode == "Inpainting":
                scale = min(1.0, 512 / max(source.size))
                display_size = (round(source.width*scale), round(source.height*scale))
                preview = source.resize(display_size, Image.Resampling.LANCZOS)
                st.caption("Sapukan kuas putih di area yang ingin diubah.")
                drawing = st_canvas(fill_color="rgba(255,255,255,1)", stroke_width=24,
                    stroke_color="#FFFFFF", background_image=preview,
                    update_streamlit=True, height=preview.height, width=preview.width,
                    drawing_mode="freedraw", key=f"mask_{st.session_state.get('canvas_revision', 0)}")
            else:
                st.image(source, use_column_width=True)
        with right:
            st.subheader("Edit settings")
            with st.form("edit"):
                edit_prompt = st.text_area("Prompt edit", "2D illustration of a broken satellite with blue solar panels on the moon, Earth in the starry sky")
                edit_seed = int(st.number_input("Seed edit", min_value=0, max_value=4294967295, value=9))
                strength = st.slider("Strength", 0.1, 1.0, 1.0, 0.05)
                if mode != "Inpainting":
                    direction = st.selectbox("Arah perluasan", ["all", "right", "left", "top", "bottom"])
                    expansion = st.slider("Ekspansi (piksel)", 64, 256, 128, 64)
                apply = st.form_submit_button("Apply edit", type="primary")
            if apply:
                try:
                    if mode == "Inpainting":
                        if drawing.image_data is None:
                            raise ValueError("Gambar mask terlebih dahulu.")
                        alpha = drawing.image_data[:, :, 3]
                        if not np.any(alpha > 0):
                            raise ValueError("Mask masih kosong.")
                        mask = Image.fromarray((alpha > 0).astype("uint8") * 255)
                        mask = mask.resize(source.size, Image.Resampling.NEAREST).filter(ImageFilter.MaxFilter(9))
                        canvas = source
                    else:
                        canvas, mask = logic.prepare_outpainting(source, expansion, direction)
                    with st.spinner("Menjalankan inpainting…"):
                        result = logic.run_inpainting(get_model("inpainting"), canvas, mask,
                            edit_prompt, strength, seed=edit_seed, steps=steps, cfg=cfg)
                    st.session_state.before_edit = source
                    st.session_state.current_image = result
                    st.session_state.last_mask = mask
                    st.session_state.canvas_revision = st.session_state.get("canvas_revision", 0) + 1
                    st.rerun()
                except Exception as exc:
                    show_error(exc)
        if "before_edit" in st.session_state:
            before, after = st.columns(2)
            before.image(st.session_state.before_edit, caption="Before", use_column_width=True)
            after.image(st.session_state.current_image, caption="After", use_column_width=True)
            st.download_button("Download hasil edit", png_bytes(st.session_state.current_image),
                               "edited_image.png", "image/png")
            with st.expander("Lihat mask terakhir"):
                st.image(st.session_state.last_mask, width=256)
