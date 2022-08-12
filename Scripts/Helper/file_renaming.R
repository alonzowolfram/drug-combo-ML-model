# Set directories. 
# https://stackoverflow.com/a/4463291
win_username = Sys.info()[['user']]
switch(Sys.info()[['sysname']],
       Windows = {tilde_expansion = paste0("C:/Users/", win_username, "/")},
       Linux   = {tilde_expansion = "~/"},
       Darwin  = {tilde_expansion = "~/"})

# Set path variables.
data_dir = paste0(tilde_expansion, "Dropbox/Work/Thesis_projects/Common_material/Data/")
results_dir = paste0(tilde_expansion, "Dropbox/Work/Thesis_projects/Common_material/Results/")
script_dir = paste0(tilde_expansion, "Dropbox/Work/Thesis_projects/Common_material/Scripts/")
error_log_dir = paste0(tilde_expansion, "Dropbox/Work/Thesis_projects/Common_material/Error_logs/")

# Set other variables.
id_type = "HUGO"
preprocessing_method = "YuGene"
deg_type = "All"
all_genes = ifelse(deg_type == "All", "_all_genes", "") 
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
source.all(paste0(script_dir, "Helper/"))

# 2021/09/14: Rename NCI-60 disease network files.
deg_type = "SD"
deg_cutoff = 2
if(deg_type == "All")  {
  all_genes = "_all-genes"
} else if(deg_type == "SD") {
  all_genes = paste0(deg_cutoff, "SD")
} else {
  all_genes = paste0(deg_type)
}
GSE = "GSE32474"
setwd(folder_generator(data_dir = data_dir, data_source = "NCI-60", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"))
for(network_filename in list.files(folder_generator(data_dir = data_dir, data_source = "NCI-60", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"), full.names = F)) {
  sample = network_filename %>% str_split("_") %>% unlist %>% .[1]
  new_filename = filename_generator(data_dir = data_dir, data_source = "NCI-60", data_type = "Expression", data_subtype = "disease-network", extension = ".rds", data_set = GSE, sample = sample, processing_stage = "Processed", processing = preprocessing_type, gene_identifier = id_type, DE_criterion = all_genes, full_path = F)
  
  print(paste("Old filename:", network_filename))
  print(paste("New filename:", new_filename))
  
  file.rename(network_filename, new_filename)
}

# 2022/06/13: Rename NCI-60 disease network files ... again. 
setwd(folder_generator(data_dir = data_dir, data_source = "NCI-60", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"))
for(network_filename in list.files(folder_generator(data_dir = data_dir, data_source = "NCI-60", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"), full.names = F)) {
  new_filename = network_filename %>% regexPipes::gsub("NCI-60_GSE32474_", "") %>% regexPipes::gsub("disease-network_processed_YuGene_HUGO", "disease_network")
  
  print(paste("Old filename:", network_filename))
  print(paste("New filename:", new_filename))
  
  file.rename(network_filename, new_filename)
}

# 2022/06/30: Rename GEO and TCGA 2SD disease network files. 
setwd(folder_generator(data_dir = data_dir, data_source = "GEO", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"))
for(network_filename in list.files(folder_generator(data_dir = data_dir, data_source = "GEO", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"), full.names = F)) {
  new_filename = network_filename %>% regexPipes::gsub("all_genes", "2SD")
  
  print(paste("Old filename:", network_filename))
  print(paste("New filename:", new_filename))
  
  file.rename(network_filename, new_filename)
}

setwd(folder_generator(data_dir = data_dir, data_source = "TCGA", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"))
for(network_filename in list.files(folder_generator(data_dir = data_dir, data_source = "TCGA", data_type = "Expression", processing_stage = "Processed", additional_paths = "Disease_networks/Rank-difference_cutoff/"), full.names = F)) {
  #new_filename = network_filename %>% regexPipes::gsub("TCGA_", "") %>% regexPipes::gsub("_processed_YuGene_HUGO", "")
  new_filename = network_filename %>% regexPipes::gsub("disease\\-network", "disease_network")
  
  print(paste("Old filename:", network_filename))
  print(paste("New filename:", new_filename))
  
  file.rename(network_filename, new_filename)
}
