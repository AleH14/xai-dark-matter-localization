# Project Context — XAI Dark Matter Localization

## 1. Project Identity

### Provisional Title

**Spatial Localization of Information Associated with Dark Matter in Galaxy Images Using Explainable Artificial Intelligence**

Spanish title:

**Localización espacial de la información asociada a materia oscura en imágenes de galaxias mediante IA explicable**

---

## 2. Core Vision

This project aims to become a strong scientific research project, suitable for academic presentation and potentially for a conference paper such as **CONESCAPAN**.

The project is not intended to be a generic machine learning exercise. It is a research-oriented project that combines:

- computational astrophysics;
- galaxy image analysis;
- deep learning;
- explainable artificial intelligence;
- physical interpretation of model behavior;
- simulation-to-observation reasoning.

The main idea is to investigate whether deep learning models can infer halo-related physical properties from galaxy images and, more importantly, identify **where in the image** the predictive information is located.

The project should feel like a serious scientific paper, not a basic image classification project.

---

## 3. Main Research Motivation

Dark matter cannot be directly observed through electromagnetic radiation. However, its gravitational influence affects the formation, structure, and dynamics of galaxies.

Galaxy morphology may contain indirect information related to the dark matter halo in which the galaxy resides. A neural network may learn correlations between visible galaxy structure and halo-related physical properties.

The central interest of this project is not only whether a model can predict a halo-related property, but also which spatial regions of the galaxy image contribute most to that prediction.

The key research idea is:

> Instead of only asking “how much dark matter-related information can be inferred?”, this project asks “where is that information located in the image?”

---

## 4. Critical Scientific Framing

This project **does not attempt to directly detect dark matter**.

This is a strict scientific rule.

Dark matter is not visible in galaxy images, and the project must not claim that the model sees, detects, segments, or localizes dark matter directly.

### Correct framing

The project investigates whether AI models can identify spatial regions in galaxy images that contain **predictive information correlated with halo-related physical properties**.

Recommended phrasing:

> The model identifies image regions that contribute to the prediction of physical properties associated with the dark matter halo.

Another acceptable phrasing:

> The attribution maps highlight regions containing information predictive of halo-related properties, not direct dark matter detections.

Spanish phrasing:

> El modelo identifica regiones de la imagen que contienen información predictiva asociada a propiedades físicas relacionadas con el halo de materia oscura.

### Incorrect framing

Avoid these claims:

- The model detects dark matter.
- The model localizes dark matter directly.
- The heatmaps show where dark matter is.
- The segmentation masks represent dark matter locations.
- The network sees the dark matter halo.

These statements are scientifically unsafe and should not appear in the code comments, README, notebooks, paper draft, figures, captions, or documentation.

---

## 5. Main Research Question

The guiding research question is:

> In which regions of galaxy images is the information located that allows a model to infer physical properties associated with dark matter halos?

Spanish version:

> ¿En qué regiones de una galaxia se encuentra la información que permite inferir propiedades físicas relacionadas con el halo de materia oscura?

This question defines the project’s identity.

The project is not primarily about achieving the lowest possible prediction error. Model performance matters, but explainability and spatial interpretation are central.

---

## 6. Main Hypothesis

The working hypothesis is:

> Deep learning models can learn spatial patterns in galaxy images that are statistically correlated with halo-related physical properties, and explainability methods can reveal which image regions contribute most to those predictions.

Spanish version:

> Las redes neuronales pueden aprender patrones espaciales en imágenes de galaxias que están correlacionados con propiedades físicas del halo de materia oscura, y estas regiones pueden ser identificadas mediante técnicas de explicabilidad.

---

## 7. Scientific Objective

### General Objective

Develop an explainable artificial intelligence pipeline capable of learning relationships between galaxy morphology and halo-related physical properties, then spatially analyzing which regions of the galaxy image contribute most to the model’s prediction.

### Specific Objectives

1. Build a dataset of galaxy images from astrophysical simulations, paired with physical metadata related to dark matter halos.
2. Train a deep learning regression model to infer a halo-related property from galaxy images.
3. Evaluate the model using regression metrics such as MAE, RMSE, and R².
4. Apply explainability techniques to generate attribution maps for individual predictions.
5. Quantify the spatial distribution of attribution using radial regions such as center, middle, and outer galaxy zones.
6. Interpret the results from both a machine learning and astrophysical perspective.
7. Avoid overclaiming by clearly distinguishing between predictive information and direct dark matter detection.

---

## 8. Main Pipeline

The intended scientific pipeline is:

```txt
Astrophysical simulation data
        ↓
Galaxy image extraction
        ↓
Physical metadata extraction
        ↓
Dataset construction
        ↓
Image → halo-property regression
        ↓
Model evaluation
        ↓
Explainability methods
        ↓
Spatial attribution maps
        ↓
Radial importance analysis
        ↓
Physical interpretation
        ↓
Paper-ready results