folder_generator = function(data_dir, data_source, data_type, processing_stage = "", additional_paths = "", full_path = T) {
  # data_dir: character
  # data_set: c("NCI60", "MDAMB438", "GDSC", "CCLE", "Gao", "GEO", "GTEx", "TCGA", "ALMANAC", "SPEED2", "STRINGDB", "COSMIC")
  # data_type: c("pData", "Expression", "Mutations", "CN", "Esets", "Drug_targets", "ML", "Metadata", "Misc")
  #         pData: phenotypic data about cell lines, animal models, or patient samples.
  #         Expression: gene-expression data, usually measured using either a microarray or RNA-seq.
  #         Genomic: mutational and copy-number data.
  #         Esets: [used only in the GEO folder] GEO Esets.
  #         Drug: [used only in the FDA, CTD, DrugBank-KEGG, and TRRUST folders] data about drugs, usually either their structural information or targets.
  #         Network: biological network (e.g. PPI) data. 
  #         ML: machine-learning data, usually the inputs for the machine-learning models.
  #         Metadata: data about the data sets themselves.
  #         Misc: anything that doesn't belong in the above categories. 
  # extension: character
  # additional_paths: character
  
  # Naming convention: [data_set]_[data_type_final]_[processing_stage_final]_
  # data_type_final = str_to_lower(data_type) if data_type != "Machine_learning"
  # processing_stage_final = str_to_lower(processing_stage) if data_type != "Machine_learning"
  
  if(processing_stage %in% c("Raw", "Processed")) {
    processing_stage_dir = processing_stage
  } else {
    processing_stage_dir = "Processed"
  }
  
  additional_paths = ifelse((base::grepl(".+\\/$", additional_paths) | additional_paths==""), additional_paths, paste0(additional_paths, "/"))
  foldername = ifelse(full_path, paste0(data_dir, data_source, "/", data_type, "/", processing_stage_dir, "/", additional_paths), paste0(data_source, "/", data_type, "/", processing_stage_dir, "/", additional_paths))
  
  return(foldername)
}

filename_generator = function(data_dir, data_source, data_type, extension, additional_paths = "", data_set = "", sample = "", data_subtype = "", additional_info = "", processing_stage = "", processing = "", gene_identifier = "", DE_criterion = "", input_or_output = "", input_data_type = "", data_subset_type = "", ML_data_processing_type = "", model_building_type = "", output_type = "", full_path = T) {
  # This is mostly used for processed data. 
  
  # data_dir: character
  # data_source: c("NCI60", "MDAMB438", "GDSC", "CCLE", "Gao", "GEO", "GTEx", "TCGA", "ALMANAC", "SPEED2", "STRINGDB", "COSMIC")
  # data_type: c("pData", "Expression", "Genomic", "Esets", "Drugs", "Network", "ML", "Metadata", "Logs", "Misc")
  #         pData: phenotypic data about cell lines, animal models, or patient samples.
  #         Expression: gene-expression data, usually measured using either a microarray or RNA-seq.
  #         Genomic: mutational and copy-number data.
  #         Esets: [used only in the GEO folder] GEO Esets.
  #         Drug: data about drugs, usually either their structural information or targets.
  #         Network: biological network (e.g. PPI) data. 
  #         ML: machine-learning data, usually the inputs for the machine-learning models.
  #         Metadata: data about the data sets themselves.
  #         Misc: anything that doesn't belong in the above categories. 
  # extension: character
  # additional_paths: character
  # data_set: character (e.g. GSE...)
  # sample: character
  # data_subtype: c("differential_expression", "disease_network")
  # additional_info: character. Any additional qualifiers necessary. 
  # processing_stage: c("Raw", "Processed", paste0("Intermed", i))
  # processing: character
  # gene_identifier: c("ENTREZ", "HUGO")
  # DE_criterion: c("all_genes", paste0(i, "SD")). NOTE: all genes are retained in the main network by default. This refers to the genes that are included in the disease module. 
  #         all_genes: All genes are included in the disease module.
  #         paste0(i, "SD"): Genes +/- iSD of the mean are included in the disease module. 
  
  # For ML only:
  # input_or_output: c("Input", "Output")
  # input_data_type: c("Combined", "1", "2", "Singles")
  # data_subset_type: c("Full", "Sample")
  # ML_data_processing_type: c("PROGENy")
  # model_building_type: c("Spotchecking", "Hyperparameter_tuning", "Complete")
  # output_type: c("Results", "Raw_results", "Raw_results_by_group")
  
  # Naming convention: [data_set]_[data_type_final]_[processing_stage_final]_
  # data_type_final = str_to_lower(data_type) if data_type != "Machine_learning"
  # processing_stage_final = str_to_lower(processing_stage) if data_type != "Machine_learning"
  
  if(processing_stage %in% c("Raw", "Processed")) {
    processing_stage_dir = processing_stage
  } else {
    processing_stage_dir = "Processed"
  }
  if(extension=="") {
    extension_final = extension
  } else {
    extension_final = ifelse(base::grepl("\\.", extension), extension, paste0(".", extension %>% stringr::str_to_lower)) 
  }
  
  data_type_for_filename = data_type %>% str_to_lower %>% regexPipes::gsub("[s]{1}$", "") # Plural -> singular
  data_type_final = ifelse(data_subtype=="", paste0("_",  data_type_for_filename), "") %>% regexPipes::gsub("pdata", "pData")
  data_subtype_final = ifelse(data_subtype=="", "", paste0("_", data_subtype))
  additional_info_final = ifelse(additional_info=="", "", paste0("_", additional_info))
  data_set_final = ifelse(data_set=="", "", paste0("_", data_set))
  sample_final = ifelse(sample=="", sample, paste0("_", sample))
  processing_stage_final = ifelse(processing_stage=="", processing_stage, paste0("_", stringr::str_to_lower(processing_stage)))
  processing_final = ifelse(processing=="", processing, paste0("_", processing))
  gene_identifier_final = ifelse(gene_identifier=="", gene_identifier, paste0("_", gene_identifier))
  DE_criterion_final = ifelse(DE_criterion=="", DE_criterion, paste0("_", DE_criterion))
  
  input_or_output_final = ifelse(input_or_output=="", input_or_output, paste0("_", stringr::str_to_lower(input_or_output)))
  input_data_type_final = ifelse(input_data_type=="", input_data_type, paste0("_", stringr::str_to_lower(input_data_type)))
  data_subset_type_final = ifelse(data_subset_type=="", data_subset_type, paste0("_", stringr::str_to_lower(data_subset_type)))
  ML_data_processing_type_final = ifelse(ML_data_processing_type=="", ML_data_processing_type, paste0("_", ML_data_processing_type))
  model_building_type_final = ifelse(model_building_type=="", model_building_type, paste0("_", stringr::str_to_lower(model_building_type)))
  output_type_final = ifelse(output_type=="", output_type, paste0("_", stringr::str_to_lower(output_type)))
  
  additional_paths = ifelse((base::grepl(".+\\/$", additional_paths) | additional_paths==""), additional_paths, paste0(additional_paths, "/"))
    
  absolute_path = folder_generator(data_dir = data_dir, data_source = data_source, data_type = data_type, processing_stage = processing_stage, additional_paths = additional_paths, full_path = full_path)
  
  filename = ifelse(full_path, 
                    paste0(absolute_path, data_source, data_set_final, sample_final, data_type_final, data_subtype_final, additional_info_final, processing_stage_final, processing_final, gene_identifier_final, input_data_type_final, data_subset_type_final, DE_criterion_final, ML_data_processing_type_final, model_building_type_final, output_type_final, extension_final),
                    paste0(data_source, data_set_final, sample_final, data_type_final, data_subtype_final, processing_stage_final, processing_final, gene_identifier_final, input_data_type_final, data_subset_type_final, DE_criterion_final, additional_info_final, ML_data_processing_type_final, model_building_type_final, output_type_final, extension_final))
  
  return(filename)
  
}