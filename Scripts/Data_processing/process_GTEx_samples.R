### GTEx
#Subset raw GTEx data.

# Read in the normal tissue sample. 
# normal_tissue = read.gct(paste0(data_dir, "GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"))
normal_tissue = readRDS(paste0(data_dir, "GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.rds"))
# Save as an RDS file. 
# saveRDS(normal_tissue, paste0(data_dir, "GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.rds"))
colnames(normal_tissue) = base::gsub("\\.", "-", colnames(normal_tissue)) # Convert separators from full stops to dashes.

# Read in the sample-annotation table. 
sample_annotation = read.csv(paste0(data_dir, "GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.csv"), header=TRUE)
sample_annotation$SMTS = sample_annotation$SMTS %>% regexPipes::gsub("[[:space:]]", "_")
# Read in the cancer-types table.
cancer_types = read.csv(paste0(tcga_data_dir, "tcga_cancer_types.csv"), stringsAsFactors = F)

# For each tissue type, create a subset of the original GTEx file containing only samples from that tissue.
if(preprocessing_method == "None") {
  GTEx_list_filename = paste0(data_dir, "GTEx/GTEx_normal_tissues_list_unprocessed.rds")
} else {
  GTEx_list_filename = paste0(data_dir, "GTEx/GTEx_normal_tissues_list_processed_", preprocessing_method, ".rds")
}

if(!file.exists(GTEx_list_filename)) {
  normal_tissues = list()
} else {
  if(!exists("normal_tissues")) {
    normal_tissues = readRDS(GTEx_list_filename)
  }
}
names(normal_tissues) = names(normal_tissues) %>% regexPipes::gsub("[[:space:]]", "_")

tissue_types = cancer_types$GTExNormalTissue %>% .[complete.cases(.)] %>% unique()
for(tissue_type in tissue_types) {
  makeGTExSubset(tissue_type)
}
# Save.
saveRDS(normal_tissues, GTEx_list_filename)

# Delete the original GTC file ... it's huge!
rm(normal_tissue)
gc()

# Process raw GTEx data.
for(tissue_type in tissue_types) {
  if(preprocessing_method == "None") {
    file_name = paste0(data_dir, "GTEx/Unprocessed_samples/GTEx_normal_samples_unprocessed_", tissue_type, ".rds")
  } else {
    file_name = paste0(data_dir, "GTEx/Processed_samples/GTEx_normal_samples_processed_", preprocessing_method, "_", tissue_type, ".rds")
  }
  
  processGTEx(tissue_type, file_name, preprocessing_method)
}