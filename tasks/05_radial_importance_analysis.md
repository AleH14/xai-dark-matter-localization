# Add explainability for halo mass regression

## Goal
Apply explainability methods to the trained halo mass regression model.

## Context
The model predicts a scalar halo-related property. XAI should identify which image regions contribute to that prediction.

## Files to create
- src/explainability/gradcam.py
- src/explainability/integrated_gradients.py
- notebooks/08_generate_xai_maps.ipynb

## Requirements
- Implement Grad-CAM for the last convolutional layer of ResNet18.
- Implement Integrated Gradients if feasible.
- Generate attribution maps for selected test images.
- Save heatmaps as PNG.
- Save raw attribution arrays as .npy.
- Overlay heatmaps on original galaxy images.
- Include a clear note:
  "Attribution maps show regions influential for the model prediction, not direct dark matter detections."

## Do not
- Do not call heatmaps "dark matter maps".
- Do not modify the trained model architecture.
- Do not use test labels during attribution generation except for reporting.

## Acceptance criteria
- For a trained checkpoint, heatmaps are generated for sample test images.
- Outputs are saved under results/xai/.