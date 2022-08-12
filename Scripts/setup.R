# Set other variables.
id_type = "HUGO"
preprocessing_type = "YuGene"
deg_type = "All"
all_genes = ifelse(deg_type == "All", "all_genes", "") 
disease_network_folder = ""
deg_cutoff = 2 # Number of SDs away from the mean rank change to be considered a differentially expressed gene. 
GSE = "GSE32474"
# TCGA/Gao cancer types we do not have GTEx normal tissue for:
cancer_no_normal = c(
  "CHOL", # Cholangiocarcinoma -> bile duct
  "HNSC", # Head and neck squamous-cell carcinoma -> mouth, nose, throat
  #"DLBC", # Scratch that. I guess we can use spleen for lymph nodes? https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4277406/
  "MESO", # Mesothelioma -> pleura
  "SARC", # Sarcoma -> soft connective tissue (bone, fat)
  #"READ", # Scratch that. It's OK to use colon tissue, I guess. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3229689/
  "THYM", # Thymoma -> thymus
  "UCS", # Uterine carcinosarcoma -> muscles of the uterine
  "UVM" # Uveal melanoma -> eye
)
# GEO GSEs.
GSEs = c("GSE30161", "GSE18728", "GSE28702", "GSE69657", "GSE72970", "GSE39345") # "GSE69657", "GSE32877", "GSE52735"

# Load required libraries.
library(affy)
library(annotate)
library(biomaRt) 
require(caTools) 
library(CePa) # Reading .gct files. 
library(corrplot) # Generating correlation-matrix plots for the network scores. 
library(doParallel) # Setting up parallel backends.
library(plyr)
library(dplyr)
library(magrittr) # https://stackoverflow.com/questions/64885956/after-after-updating-packages-running-librarytidyverse-pipe-operator-not-wor
library(data.table) # Faster and more memory-friendly version of data frames. 
library(dendextend) # Prettying up dendrograms. 
library(egg)
library(eulerr)
library(extraDistr) # Discrete uniform distribution. 
library(FactoMineR) # PCA. 
library(factoextra) # PCA.
library(flextable)
library(forcats) # fct_inorder; https://stackoverflow.com/a/51537246
library(future.apply) 
#print(Sys.setenv(R_FUTURE_FORK_ENABLE = TRUE))
#plan(multiprocess, workers = 4) # Run in parallel on local computer.
library(GEOquery) # Fetching data sets from GEO. 
library(gplots)
library(ggplot2)
library(ggpubr)
library(ggsci)
library(ggsignif)
library(gridExtra)
library(GSA)
library(gtools) # For making permutations. 
library(hgu133plus2.db)
library(igraph) # Network visualization and calculation. 
library(jetset)
library(KEGGgraph)
library(KEGGREST)
library(limma)
library(lucy) # Tools for networks/graphs.
library(MASS)
library(matrixcalc) 
library(miceadds)
library(officer)
library(org.Hs.eg.db)
hs = org.Hs.eg.db
library(parallel)
library(plotly)
library(plotrix) 
library(progeny) # For dysregulated pathway identification.
library(qdap)
library(RcppHungarian) # For solving bipartite graphs.
library(readxl) # Reading Excel files. 
library(regexPipes)
library(reshape) # Melting data frames for graphing. 
library(reshape2)
library(reticulate)
library(Rgraphviz)
library(RMariaDB)
library(sampling)
library(scales)
library(showtext)
library(STRINGdb) # Network generation.
library(stringi)
library(stringr)
library(survival)
library(survminer)
library(tibble)
library(tidyr)
library(vegan) # Jaccard index as distance measurement. 
library(xlsx)
source.all("NetworkScores/")
source.all("Helper/")
source.all("Data_processing/")
source.all("Config/")

R_version = as.numeric(R.Version()$major) + (0.1 * as.numeric(R.Version()$minor))
R_version_major = as.numeric(R.Version()$major)
R_version_minor = as.numeric(R.Version()$minor)
if(R_version < 3.7) {
  library(YuGene) # Processing microarrays.
  library(TCGA2STAT)
}

# Graphical parameters.
# https://stackoverflow.com/questions/39508304/margins-between-plots-in-grid-arrange
margin = theme(plot.margin = unit(c(2,2,2,2), "cm"))

# Create a list holding the different annotation databases.
if(R_version_major < 4 & R_version_minor < 7) {
  library(illuminaHumanv4.db)
  library(hgu219.db)
  illuminaHumanv4_db = illuminaHumanv4.db
  hgu219_db = hgu219.db
} else {
  illuminaHumanv4_db = NA
  hgu219_db = NA
}
annotation_dbs = list(
  GPL570 = hgu133plus2.db,
  GPL6104 = illuminaHumanv4_db,
  GPL13667 = hgu219_db
)