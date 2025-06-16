Potential Areas of Further Research
- Use nnUNet as a baseline instead of UNet (A semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided training cases and automatically configure a matching U-Net-based segmentation pipeline)
- Investigation into better Model Architecture/Design (Using a Basic 5 Layer U-Net for now, can look to expand/improve on this once the research topic is narrowed)

Novel Ideas
- Anatomically-Guided 3D Jigsaw Pretext Task
  - Create a bounding box around the kidney region, perform Jigsaw SSL within the bounding box only; this way, we only want to encourage learning of the kidney-tumour relationship.
  - Add On: Once dividing into patches, add some form of noise to force the model to de-noise it
https://arxiv.org/pdf/2308.05770
https://arxiv.org/html/2404.07292v1 (This one masks specific patches to force the model to fully regenerate it)

- Cross-Scale Feature Alignment (Multi-Resolution SSL)
  - Divide the image into Patches, perform feature extraction of the same patch at different scales or resolutions and align them in feature space. Enables ability to find multi-scale features (Global Kidney vs Local Tumor)
https://arxiv.org/pdf/2401.15855 (Multi-Scale)
https://www.researchgate.net/publication/377321015_Multi-Scale_Image-_and_Feature-Level_Alignment_for_Cross-Resolution_Person_Re-Identification
https://www.sciencedirect.com/science/article/pii/S0031320324003728

- Ask Dr. Akilan for feedback on research topics, additionally ask him if he has any ideas
