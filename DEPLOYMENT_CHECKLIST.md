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

3. Use `requirements.txt` for Streamlit Community Cloud deployment.

4. Provide the SAM checkpoint:

```text
sam/sam_vit_h_4b8939.pth
```

The SAM checkpoint is usually too large for a normal GitHub repository. The app requires SAM because the binary mask is generated automatically from the uploaded fish image.

If you host the checkpoint outside GitHub, add this in Streamlit Community Cloud secrets:

```toml
SAM_CHECKPOINT_URL = "https://your-host/sam_vit_h_4b8939.pth"
```

5. On Streamlit Community Cloud, set:

```text
Repository: your GitHub repository
Branch: main
Main file path: streamlit_app.py
```
