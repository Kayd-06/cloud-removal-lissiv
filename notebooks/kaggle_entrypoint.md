# Kaggle entrypoint (copy-paste)

Free GPU, 30 GPU-h/week, sessions up to 9h, datasets persist. This is your
primary training environment.

> ⚠️ **Use GPU T4 x2, NOT P100.** Kaggle's current PyTorch build dropped support for
> the older P100 (compute capability sm_60) and will crash with
> `no kernel image is available for execution on the device`. The T4 (sm_75) works.

## 1. Setup

Create a new Kaggle Notebook → **Settings** menu (top bar) → Accelerator = **GPU T4 x2**,
Internet = **On**. (Our code uses a single GPU; the second T4 just sits idle, which is fine.)

```python
# clone your repo (push this project to GitHub first, or upload as a Kaggle Dataset)
!git clone https://github.com/<you>/cloud-removal-lissiv.git
%cd cloud-removal-lissiv
!pip install -q -r requirements.txt
```

## 2. Validate the pipeline on dummy data (2 min, do this first)

```python
!python scripts/smoke_test.py
!python scripts/make_dummy_data.py --out data/clear --n 40
!python -m src.preprocess.synthetic_clouds --clear_dir data/clear --out_dir data/patches --per_patch 3
!python -m src.train.train_gan --data data/patches --ckpt_dir /kaggle/working/ckpts --epochs 2
```

If that runs clean, the whole scaffold works — now swap in real data.

## 3. Real data

- Upload preprocessed LISS-IV **clear** patches (`.npz` with `clear`,`sar`) as a **Kaggle Dataset**
  so they persist across sessions. Build them locally/here with `src.preprocess.tiling`.
- Then regenerate paired data and train:

```python
!python -m src.preprocess.synthetic_clouds --clear_dir /kaggle/input/lissiv-clear --out_dir data/patches
!python -m src.train.train_gan       --data data/patches --ckpt_dir /kaggle/working/ckpts
!python -m src.train.train_diffusion --data data/patches --ckpt_dir /kaggle/working/ckpts \
        --init /kaggle/input/sen12mscr-weights/pretrained.pt   # optional transfer
```

## 4. Persist checkpoints across sessions

`/kaggle/working` is saved as notebook output. To resume next session, add **this
notebook's output** (or a checkpoint dataset) as an input, then point `--ckpt_dir` at it.
Training auto-resumes from `*_last.pt`.

## 5. Evaluate + demo

```python
!python -m src.eval.compare --data data/patches \
        --gan /kaggle/working/ckpts/gan_last.pt \
        --diff /kaggle/working/ckpts/diff_last.pt --out results
# view results/cmp_*.png inline
```

## GPU-hour budget tips

- Diffusion validation samples every epoch — that's the expensive part. `CFG.sample_steps=50`
  and only 4 val batches keeps it cheap. Lower `sample_steps` to 20 for faster dev loops.
- Drop `CFG.patch_size` to 128 and raise `batch_size` to 16 if you hit OOM or want speed.
- Colab free is your overflow when the 30h/week runs out — same commands, mount Drive
  for `--ckpt_dir` instead of `/kaggle/working`.
```
