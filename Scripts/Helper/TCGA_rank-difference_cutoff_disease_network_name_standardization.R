# TCGA rank-difference_cutoff disease network name standardization.
# 2021-11-10.
deg_type = "SD"
if(deg_type == "All")  {
  all_genes = "_all-genes"
} else if(deg_type == "SD") {
  all_genes = paste0(deg_cutoff, "SD")
} else {
  all_genes = paste0(deg_type)
}

file_names_original = list.files(folder_generator(data_dir = data_dir, data_source = "TCGA", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"), full.names = T) 
file_names_new = file_names_original %>% regexPipes::gsub("TCGA-", "TCGA_TCGA-") %>% regexPipes::gsub("disease_network", paste0("disease-network_processed")) %>% regexPipes::gsub("all_genes", paste0(preprocessing_type, "_", id_type, "_", all_genes))

# Rename.
file.rename(file_names_original, file_names_new)