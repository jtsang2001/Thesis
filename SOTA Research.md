Model Guideline
- Develop using nnU-Net: A semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided training cases and automatically configure a matching U-Net-based segmentation pipeline.
https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file
- Develop using Auto3DSeg: Analyzes the global information such as intensity, data size, and data spacing of the dataset, and then generates algorithm folders in MONAI bundle format based on data statistics and algorithm templates
https://github.com/Project-MONAI/tutorials/tree/main/auto3dseg

SOTA Implementations
- Myronenko et al. DICE: 0.835, Uses Auto3DSeg: Which will automatically analyze the dataset, generate hyperparameter configurations for several supported algorithms, train them, and produce inference and ensembling. The system automatically scales to all available GPUs and supports multi-node training. https://arxiv.org/pdf/2310.04110
- Swapna et al., AUC: 0.9, Uses MedSwim: Swim transformer-based framework https://ieeexplore-ieee-org.ezproxy.lakeheadu.ca/document/11042017
- Matos et al. DICE: 0.91, Uses CPP-UNet: Combined Pyramid Pooling + U-Net, https://ieeexplore-ieee-org.ezproxy.lakeheadu.ca/stamp/stamp.jsp?tp=&arnumber=10620387
- Uhm et al., DICE: 0.887, Uses Low + High Res U-Net Segmentation https://arxiv.org/pdf/2312.05528
- Karunanayake et al., DICE: 0.90, Uses Vision Transformers and CNNs https://pmc.ncbi.nlm.nih.gov/articles/PMC11769543/#app1-tomography-11-00003
- Ziaee et al. DICE: 0.90, Uses UA-UNet (Uncertainty Aware Pseudo-Label Generation) https://mail.google.com/mail/u/1/?ogbl#search/takilan%40lakeheadu.ca/FMfcgzQbfxjtQjlHlHXvJBlQTqgQLkDr?projector=1&messagePartId=0.1


SOTA Models
-https://github.com/yulequan/UA-MT/tree/master
-https://github.com/kleinzcy/SASSnet?tab=readme-ov-file
-https://github.com/koncle/CoraNet?tab=readme-ov-file

Current Model:
- DeepLabV3 Based Model, Image Size was scaled down to reduce computation time. Was able to achieve a computational DICE Score of 0.75. This may be a good starting baseline, investigation into nnU-Net and Auto3DSeg required to determine best baseline.
<img width="1560" height="798" alt="image" src="https://github.com/user-attachments/assets/2564612d-7d55-4705-927b-5722d528a6e4" />
