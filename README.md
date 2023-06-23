# CVAE-BRCA

Overview
----------
RNA-seq generation model | [Paper](https://doi.org/10.5909/JBE.2023.28.3.275) | [Code](https://github.com/HyunSBong/CVAE-RNA-seq)
- <img width="749" alt="cvae" src="https://github.com/HyunSBong/CVAE-BRCA/assets/69189272/82f10eb6-8cd4-4d3e-acb5-763c43996529">
----------
Modified it to Pytorch (gene level self-attention)
- <img width="749" alt="bio" src="https://github.com/HyunSBong/CVAE-BRCA/assets/69189272/d2ef2ea4-7f16-4e2b-9b50-4d5fb6c8d8ed">
- Original moBRCA-net | [Paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05273-5) | [Code](https://github.com/cbi-bioinfo/moBRCA-net)
- ![fig1_v7](https://github.com/HyunSBong/CVAE-BRCA/assets/69189272/ea63e488-9f97-4384-a7b0-1bb3e75aa6c1)
----------
BRCA_weighted_DEGs.R - Top 1,000 genes with high weights were selected from the miRNA-TF-mRNA gene regulatory networks.
- Reference Statistical Analysis | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7644310/)
- |Subtypes|Basal-like|Her2|LumA|LumB|Normal-like|
  |---|---|---|---|---|---|
  |Weighted DEGs|376|157|249|206|249|
----------
Dataset
----------
- GTEx(Genotype-Tissue Expression) Dataset
- TCGA(Cancer Genome Atlas) Dataset
- L1000 landmark gene set
- RNA-seq(human transcriptomics) Dataset (9147 samples and 18154 genes)
- TCGA-BRCA (PAM50)

Install dependencies
----------
- torch >= 1.12.1
- umap-learn >= 0.5.3
- scikit-learn >= 1.1.1
