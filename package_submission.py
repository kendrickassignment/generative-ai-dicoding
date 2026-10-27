"""Package only authentic, executed submission evidence. Never invent outputs."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
NAMES = ['Pipeline_submission_BFGAI_Kendrick_Filbert.ipynb',
         'Streamlit_submission_BFGAI_Kendrick_Filbert.ipynb',
         'video_demo_aplikasi_BFGAI.mp4', 'requirements.txt']


def validate():
    for name in NAMES:
        path=ROOT/name
        if not path.is_file(): raise ValueError(f'Berkas wajib belum tersedia: {name}')
    for name in NAMES[:2]:
        nb=nbformat.read(ROOT/name,as_version=4)
        nbformat.validate(nb)
        for i,cell in enumerate(nb.cells):
            if cell.cell_type!='code' or not cell.source.strip(): continue
            if cell.execution_count is None:
                raise ValueError(f'{name}: sel {i} belum dijalankan.')
            if any(o.output_type=='error' for o in cell.outputs):
                raise ValueError(f'{name}: sel {i} memiliki error.')
        if name.startswith('Pipeline'):
            for i in [7,10,12,15,18,20,28,30,35,37,41,43,46,48,50]:
                if not any('image/png' in o.get('data',{}) for o in nb.cells[i].outputs):
                    raise ValueError(f'{name}: sel {i} belum memiliki output PNG eksperimen.')
            for i in [13,16,21]:
                if 'diisi setelah' in nb.cells[i].source.lower():
                    raise ValueError('Observasi visual aktual belum dilengkapi.')
    import imageio_ffmpeg
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-i',str(ROOT/NAMES[2]),
        '-f','null','-'],capture_output=True,text=True)
    if result.returncode: raise ValueError('Video gagal didekode oleh FFmpeg.')
    import re
    match=re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)',result.stderr)
    if not match: raise ValueError('Durasi video tidak dapat diperiksa.')
    h,m,s=map(float,match.groups()); duration=h*3600+m*60+s
    if not 60 <= duration <= 120: raise ValueError(f'Gunakan video 60–120 detik; aktual {duration:.1f}s.')
    return duration


if __name__=='__main__':
    try:
        duration=validate()
    except ValueError as exc:
        sys.exit(f'BELUM SIAP SUBMIT: {exc}')
    archive=ROOT.parent/'BFGAI_Kendrick_Filbert.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in NAMES: z.write(ROOT/name, f'BFGAI_Kendrick_Filbert/{name}')
    print(f'Created {archive}; video {duration:.1f}s. Periksa kualitas visual sebelum upload.')
