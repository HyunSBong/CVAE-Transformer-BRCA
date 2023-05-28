# CVAE-BRCA

Overview
----------
RNA-seq generation model : [Code](https://github.com/HyunSBong/CVAE-RNA-seq)
- <img width="749" alt="스크린샷" src="https://user-images.githubusercontent.com/69189272/229360369-fd217d1c-6749-462f-b617-30adc314c4f1.png">


Modified it to Pytorch (gene level self-attention)
- Original moBRCA-net | [Paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05273-5) | [Code](https://github.com/cbi-bioinfo/moBRCA-net)
- <img width="749" alt="bio" src="https://github.com/HyunSBong/CVAE-BRCA/assets/69189272/d2ef2ea4-7f16-4e2b-9b50-4d5fb6c8d8ed">


Dataset
----------
- GTEx(Genotype-Tissue Expression) Dataset
- TCGA(Cancer Genome Atlas) Dataset
- L1000 landmark 
- RNA-seq(human transcriptomics) Dataset (9147 samples and 18154 genes)
- TCGA-BRCA (PAM50)

Install dependencies
----------
- torch >= 1.12.1
- umap-learn >= 0.5.3
- scikit-learn >= 1.1.1
