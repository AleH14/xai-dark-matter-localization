# Fix scientific narrative in README and dataset card

## Goal
Update the project documentation so it no longer claims that binary masks represent dark matter locations.

## Context
The current project uses masks generated from radial regions or image thresholding. These masks are not physical dark matter ground truth. They should be described as spatial analysis masks or morphology masks.

## Files to modify
- README.md
- docs/dataset_card.md
- any notebook markdown cells that mention "dark matter masks" or "dark matter localization masks"

## Requirements
- Replace claims of direct dark matter localization with careful scientific language.
- Use this framing:
  "The project identifies spatial regions containing predictive information associated with dark-matter-related halo properties."
- Explicitly state that the project does not directly detect dark matter.
- Explain that radial masks are used for attribution analysis: center, middle, outer.

## Do not
- Do not claim that masks are dark matter ground truth.
- Do not remove the project motivation.
- Do not change code in this task.

## Acceptance criteria
- No documentation says that the model directly detects dark matter.
- README clearly describes the pipeline: image → halo property regression → XAI → radial analysis.