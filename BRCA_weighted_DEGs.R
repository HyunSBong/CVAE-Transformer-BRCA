library(dplyr)
library(CancerSubtypes)

### load the brca tumor FPKM data
brca_train = read.csv("tcga_brca_new_sampleid.csv")
brca_tumor_counts = read.csv("brca_tumor_counts_new.csv")

### labels of PAM50
subtypes = brca_train$PAM50
subtypes_names <- unique(subtypes)

### transpose
brca_train = brca_train[,2:401]
brca_train = t(brca_train)
colnames(brca_train) = brca_train[1,]
brca_train = brca_train[-c(1,2),]

brca_tumor_counts = brca_tumor_counts[,2:1087]
colnames(brca_tumor_counts) <- gsub("\\.", "-", colnames(brca_tumor_counts))
rownames(brca_tumor_counts) = brca_tumor_counts[,1]
brca_tumor_counts = brca_tumor_counts[,-c(1)]

### conduct the differential expression analysis using edgeR
### save the DEGs after filtering the logFC and pvalue cutoff
subtype_DEGs_filter_list <- list()
subtype_count <- 0 # counter
for (val in subtypes_names) {
  subtype_count <- subtype_count + 1
  idx_subtype <- which(subtypes==val)
  
  ### get the results of DEGs using vomm+limma for RNA-Seq Counts data
  subtype_results=DiffExp.limma(Tumor_Data=brca_tumor_counts[,-idx_subtype], # control group
                                Normal_Data=brca_tumor_counts[,idx_subtype], # experiment group
                                group=NULL,topk=NULL,RNAseq=TRUE)
  print(subtype_results)
  
  ### retain the genes whose logFC >=0.5 and p-vlalue <-0.05
  index_logFC <- which(abs(subtype_results[[1]]$logFC)>=0.5)
  
  ### index_Pvalue <- which(abs(subtype_results[[1]]$P.Value)<=0.05)
  index_Pvalue <- which(abs(subtype_results[[1]]$adj.P.Val)<=0.01)
  
  ### conduct the intersection operation
  index_logFC_Pvalue <- intersect(index_logFC,index_Pvalue) 
  
  ### get the initial DEGs
  DEGs_filter <- subtype_results[[1]]$ID[index_logFC_Pvalue] 
  
  ### save the initial to the list variable
  subtype_DEGs_filter_list[[subtype_count]] <- DEGs_filter 
}
names(subtype_DEGs_filter_list) <- subtypes_names

############## Step 3: Acuqire the weights of genes from gene regulatroy network
############## Step 3.1: get the weights(ranking) for all the genes of tumor data
### load the gene regulatory network
data(Ranking)
### find the corresponding index of all genes of tumor data from Ranking
index_genes <- match(rownames(brca_train),Ranking$mRNA_TF_miRNA.v21_SYMBOL)
### remove the NA value for the rownames(brca_train)
index_genes_nonna <- index_genes[which(!(is.na(index_genes)))] 
### get the dataframe including the gene names and ranking(weight) data
DEGs_ranking <- data.frame(GeneName = rownames(brca_train)[which(!(is.na(index_genes)))],
                           Ranking[index_genes_nonna,],stringsAsFactors=FALSE)
### reorder the DEGs_ranking by descending order of ranking
DEGs_ranking_order <- DEGs_ranking[order(DEGs_ranking$ranking_default,decreasing = TRUE),]

### assgin the minimum to the iterm whose ranking_default is NA
index_rank=which(is.na(DEGs_ranking_order$ranking_default))
DEGs_ranking_order$ranking_default[index_rank]=min(DEGs_ranking_order$ranking_default,na.rm =TRUE)

############## Step 3.2:conduct the intersction between inital DEGs and top 2000 genes with high weightes, 
##############          and get the weighted DEGs for classification 
### the list for saving the weighted DEGs for classification
subtype_weighted_DEGs_classification <- list()
for (kk in c(1:length(subtypes_names))) {
  subtype_weighted_DEGs_classification[[kk]] <- intersect(subtype_DEGs_filter_list[[kk]],DEGs_ranking_order$GeneName[c(1:300)])
}
names(subtype_weighted_DEGs_classification) <- subtypes_names

############## Step 3.3:conduct the intersction between inital DEGs and top 4000 genes with high weightes, 
##############          and get the weighted DEGs for classification 
### the list for saving the weighted DEGs for pathway enrichment analysis
subtype_weighted_DEGs_pathway <- list()
for (kk in c(1:length(subtypes_names))) {
  subtype_weighted_DEGs_pathway[[kk]] <- intersect(subtype_DEGs_filter_list[[kk]],DEGs_ranking_order$GeneName[c(1:3000)])
}
names(subtype_weighted_DEGs_pathway) <- subtypes_names

############## Step 4: save the final results
### save the data of weighted DEGs for classification
save(subtype_weighted_DEGs_classification,file = "weighted_DEGs_classification.Rdata")
write.csv(subtype_weighted_DEGs_classification, "weighted_DEGs_classification.csv", row.names=FALSE)


basal = c(subtype_weighted_DEGs_classification$Basal)
her2 = c(subtype_weighted_DEGs_classification$Her2)
luma = c(subtype_weighted_DEGs_classification$LumA)
lumb = c(subtype_weighted_DEGs_classification$LumB)
normal = c(subtype_weighted_DEGs_classification$Normal)
intersected_items <- Reduce(intersect, list(basal, her2, luma, lumb, normal))

output_degs = data.frame(basal, her2, luma, lumb, normal)
