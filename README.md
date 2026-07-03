# Fish Length Streamlit Deployment

This is a deployment scaffold extracted from `24062516_Lee_Ming_Yi.ipynb`.

The app uses the stronger final model from the notebook:

- RandomForestRegressor
- morphology + species + ratio features
- reported notebook performance: R2 0.9745, RMSE 0.8632, MAE 0.5198 cm

## Required Model Artifacts

Commit these files in `models/` before deploying from GitHub:

```text
models/
  final_random_forest_model.joblib
  final_scaler.joblib
  final_species_columns.json
  final_feature_columns.json
```

In the notebook, these were saved under:

```text
/content/drive/MyDrive/Colab Notebooks/final_fish_model/
```

Streamlit will not be able to run the app until these files are present.

## Required SAM Artifact

The app generates the binary mask automatically with SAM. Place the SAM checkpoint here:

```text
sam/sam_vit_h_4b8939.pth
```

The notebook used Meta's `vit_h` checkpoint.

The SAM checkpoint is usually too large for normal GitHub commits. For Streamlit Community Cloud, host the checkpoint outside GitHub and add this secret in the Streamlit app settings:

```toml
SAM_CHECKPOINT_URL = "https://your-host/sam_vit_h_4b8939.pth"
```

If the checkpoint already exists at `sam/sam_vit_h_4b8939.pth`, the app will use the local file instead.

If neither a local checkpoint nor a secret is provided, the app tries the public Meta SAM checkpoint URL used in the notebook:

```text
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment Notes

Deploy from GitHub by setting:

```text
Main file path: streamlit_app.py
```

SAM is large and can be slow or memory-heavy on Streamlit Community Cloud. If deployment resources are limited, replace SAM with a lighter segmentation model before deployment.

The prediction is sensitive to segmentation quality and image scale. The app shows the selected mask overlay before reporting the predicted length.
