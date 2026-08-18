# Exploratory Data Analysis Summary

## Per-Dataset Aggregated Statistics

This table verifies all 88 images successfully processed. Stats are aggregated means.

| Dataset | Images | Native Size | Brightness | Contrast | Native Vessel % | Post-Resize Vessel % | Δ Vessel % |
|---------|--------|-------------|------------|----------|-----------------|----------------------|------------|
| DRIVE | 40 | 565x584 | 83.6 | 54.2 | 8.63% | 8.63% | +0.003% |
| CHASE | 28 | 999x960 | 59.9 | 46.8 | 6.93% | 6.94% | +0.001% |
| STARE | 20 | 700x605 | 98.2 | 52.3 | 7.60% | 7.61% | +0.002% |

## t-SNE Interpretation
A pre-trained ResNet-50 (ImageNet weights, no fine-tuning) extracted 2048-dim features from the preprocessed images. t-SNE (`random_state=42`) reduced this to 2D.

**Note on t-SNE**: t-SNE distances are not strictly meaningful metrics. The algorithm preserves local neighborhood structure, meaning we only interpret the *relative grouping* and formation of distinct clusters as empirical evidence of a domain gap, not the absolute distance between clusters.