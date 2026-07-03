# SAM Checkpoint

Automatic segmentation requires:

```text
sam_vit_h_4b8939.pth
```

This file is normally too large for a standard GitHub repository. Host it outside GitHub or use Git LFS if your account limits allow it.

Place the checkpoint in this folder before running the app, or set `SAM_CHECKPOINT_URL` in Streamlit secrets so the app can download it on startup.
