# Lunar Studio — Generative Image Suite

**Kendrick Filbert** · Dicoding BFGAI · Python / Diffusers / Streamlit

Eksperimen Stable Diffusion dan aplikasi interaktif untuk text-to-image, inpainting,
outpainting satu sisi, dan zoom out. Notebook mempertahankan urutan dan struktur
template yang diberikan.

**Status:** implementasi tersedia; eksekusi GPU, observasi visual aktual, dan video
demo autentik masih harus diselesaikan. Repository ini belum menyatakan submission
siap dikumpulkan. Gambar referensi bawaan template bukan hasil generasi proyek.

## Jalankan di Google Colab

GPU lokal tidak diperlukan. Buka notebook, pilih **Runtime → Change runtime type →
T4 GPU → Save**, lalu hubungkan runtime. Ketersediaan GPU mengikuti kuota akun Colab.

1. [Buka notebook Pipeline](https://colab.research.google.com/github/kendrickassignment/generative-ai-dicoding/blob/main/Pipeline_submission_BFGAI_Kendrick_Filbert.ipynb).
2. Jalankan sel berurutan. Jika instalasi meminta restart, restart runtime lalu lanjut
   dari sel load model. Periksa hasil, sesuaikan koordinat mask, dan tulis observasi
   visual pada tiga bagian penjelasan. Semua eksperimen memakai seed dan parameter
   tercatat. Unduh notebook melalui **File → Download → Download .ipynb**.
3. [Buka notebook Streamlit](https://colab.research.google.com/github/kendrickassignment/generative-ai-dicoding/blob/main/Streamlit_submission_BFGAI_Kendrick_Filbert.ipynb)
   pada runtime GPU. Jalankan sel berurutan; tautan aplikasi muncul setelah health
   check. Proxy Colab dapat dipakai tanpa Ngrok. Untuk Ngrok, simpan token di Colab
   Secrets bernama `NGROK_AUTHTOKEN` dan berikan akses notebook. Jangan commit token.
4. Rekam **60–120 detik** aplikasi nyata: tampilkan prompt, CFG/steps/seed, klik
   Generate, tunggu hasil, dan tunjukkan Edit. Jangan mengganti hasil dengan gambar
   referensi atau video slideshow. Simpan sebagai `video_demo_aplikasi_BFGAI.mp4`.
5. Unduh notebook Streamlit dengan output, letakkan kedua notebook dan video di folder
   proyek, lalu jalankan `python package_submission.py`. Validator menolak notebook
   belum dieksekusi, output error, gambar eksperimen hilang, observasi belum diisi,
   atau video tidak valid. Validasi teknis tidak menjamin kualitas visual/nilai.

## Cakupan rubric

| Area | Implementasi |
| --- | --- |
| Text-to-image | `generate_simple_image`, `generate_advanced_image`, prompt sama, seed 222, negative prompt sesuai instruksi |
| Eksperimen | CFG 2/7.5/16; steps 10/30/50; kontrol parameter lain tetap |
| Batch dan scheduler | Satu batch empat gambar, grid 2×2; Euler A, DPM++, DDIM tanpa reload bobot |
| Inpainting | Checkpoint inpainting khusus; mask koordinat manual; broken satellite dengan seed 9 |
| Automasking | CLIPSeg pada hasil inpainting manual, lalu inpainting hasil segmentasi |
| Outpainting | Hasil edit → perluasan kanan → perluasan semua arah; overlap untuk sambungan |
| Refiner | Gambar baru → img2img strength 0.25 menggunakan komponen base yang sama |
| Streamlit | Generation/Edit, kuas mask, batch grid, kontrol parameter, Clear Memory, download PNG |

## Model dan keputusan engineering

Submission memakai `stable-diffusion-v1-5/stable-diffusion-v1-5` serta
`stable-diffusion-v1-5/stable-diffusion-inpainting` sesuai rubrik. Resolusi base
512×512. Model disimpan di CPU ketika tidak sedang inferensi; hanya pipeline aktif
dipindahkan ke GPU. Lock melindungi pergantian scheduler dan penggunaan resource
bersama. Cache model menghindari reload setiap klik; Clear Memory hanya membersihkan
alokasi yang sudah tidak terpakai, bukan menghapus bobot model.

Mask putih berarti edit, hitam berarti pertahankan. Hasil dikompositkan dengan
alpha yang tidak mengubah piksel mask hitam. Outpainting membuat overlap di tepi
yang diperluas untuk mengurangi seam; kualitas sambungan tetap perlu dinilai visual.
Refiner menggunakan kembali komponen model base. Seed dibuat ulang per eksperimen;
batch menggunakan seed 222–225. Reproduksibilitas lintas hardware/versi tidak dijamin.

## Stable Diffusion 3.5 untuk portfolio

`portfolio_sd35.py` menyediakan runner **SD 3.5 Medium**, terpisah dari submission.
SD3.5 menggunakan `StableDiffusion3Pipeline` dan scheduler flow-matching bawaan;
Euler A/DPM++/DDIM SD1.5 tidak dipasang ke pipeline ini. Runner ini belum dinyatakan
telah teruji inference. SD3.5 memerlukan GPU dan RAM lebih besar; T4 gratis belum
tentu cukup dengan semua text encoder. `--low-memory` menghilangkan T5 dan dapat
mengubah kualitas pemahaman prompt.

Terima syarat akses model sendiri di Hugging Face, lalu set `HF_TOKEN` lewat secret
environment. Jangan menyimpan token dalam notebook, video, atau repository.

```bash
python portfolio_sd35.py --prompt "A futuristic cyberpunk city at night, neon lights, flying cars"
```

## Lokal dan pengujian

Python 3.11/3.12 dianjurkan. Gunakan virtual environment. Instal versi PyTorch CUDA
yang sesuai hardware; Colab sudah menyediakan PyTorch sehingga requirements tidak
memaksa menggantinya dengan build CPU. CPU lokal cukup untuk tes geometri, bukan
target inference interaktif. Streamlit dan drawable canvas dipasangkan pada versi
yang dipin untuk menjaga API canvas.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Notebook bersifat standalone: menulis `logic.py`, `app.py`, dan requirements sendiri.
`artifacts/` menyimpan PNG, hash, parameter, dan environment dari eksekusi Pipeline.
Folder ini tidak otomatis diunggah; output PNG notebook disimpan bersama `.ipynb`.

## Referensi

- [Diffusers: Stable Diffusion inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)
- [Diffusers: Stable Diffusion 3](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)
- [Stable Diffusion 3.5 Medium model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- [CLIPSeg model](https://huggingface.co/CIDAS/clipseg-rd64-refined)

Template dan gambar referensi berasal dari materi submission Dicoding yang diberikan
pengguna. Lisensi model mengikuti masing-masing model card; bobot model tidak
disertakan dalam repository ini.
