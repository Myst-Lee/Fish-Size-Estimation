# GitHub and Streamlit Deployment Checklist

Before pushing to GitHub:

1. Confirm these files are present in `models/`:

```text
final_random_forest_model.joblib
final_scaler.joblib
final_species_columns.json
final_feature_columns.json
```

2. Keep the main app path as:

```text
streamlit_app.py
```

3. Use `requirements.txt` for the simplest Streamlit Community Cloud deployment.

4. If you want SAM automatic segmentation, install the optional dependencies from `requirements-sam.txt` and provide the checkpoint:

```text
sam/sam_vit_h_4b8939.pth
```

The SAM checkpoint is usually too large for a normal GitHub repository. Without it, the app still supports manual binary mask upload.

5. On Streamlit Community Cloud, set:

```text
Repository: your GitHub repository
Branch: main
Main file path: streamlit_app.py
```
