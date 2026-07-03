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

## Optional SAM Artifact

For automatic segmentation, place the SAM checkpoint here:

```text
sam/sam_vit_h_4b8939.pth
```

The notebook used Meta's `vit_h` checkpoint. If you do not want to deploy SAM, select **Upload binary mask** in the app and provide a mask manually.

The SAM checkpoint is usually too large for normal GitHub commits. For Streamlit Community Cloud, the manual mask mode is the simplest deployment path unless you host the checkpoint elsewhere and add custom download logic.

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

SAM is large and can be slow or memory-heavy on Streamlit Community Cloud. If deployment resources are limited, use the manual mask upload mode, or replace SAM with a lighter segmentation model.

The prediction is sensitive to segmentation quality and image scale. The app shows the selected mask overlay before reporting the predicted length.
