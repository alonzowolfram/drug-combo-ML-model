makeGTExSubset = function(tissue_type) {
  if(!exists("normal_tissues") | !exists("normal_tissue")) {
    print("Either the normal_tissues or normal_tissue variable is missing! Please have both normal_tissues and normal_tissue variables available in your global environment.")
    # normal_tissues = list in which each element is the raw normal samples for a certain tissue type. 
    return(F)
  } else {
    if(!(tissue_type %in% names(normal_tissues))) { # Only create subset if it hasn't been created yet. 
      print(paste0("Creating subset for ", tissue_type, "."))
      
      normal_sample_names = sample_annotation[sample_annotation$SMTS==tissue_type,"SAMPID",drop=TRUE]
      normal_samples = normal_tissue[,colnames(normal_tissue) %in% normal_sample_names]
      normal_tissues[[tissue_type]] <<- normal_samples # <<- because we're affecting the global variable. 
      return(T)
    } else {
      print(paste0(tissue_type, " has already been processed!"))
      return(T)
    }
  }
  
}

processGTEx = function(tissue_type, file_name, preprocessing_method) {
  # Check if the file exists.
  if(!file.exists(file_name)) {
    print(paste0("Processing ", tissue_type, "."))
    
    # Check if the subset exists. 
    if(!tissue_type %in% names(normal_tissues)) {
      print(paste0("No subset for ", tissue_type, " found. Creating subset."))
      make_subset_result = makeGTExSubset(tissue_type)
    } else {
      make_subset_result = T
    }
    
    if(!make_subset_result) {
      print("Error in creating subset for ", tissue_type, ". Skipping this tissue type.")
      next
    }
    
    # Load the expression matrix of the normal samples. 
    normal_samples = normal_tissues[[tissue_type]]
    
    if(preprocessing_method != "None") {
      # Perform log2 transformation. 
      print(paste0("Performing log2 transformation for ", tissue_type, "."))
      normal_samples_log2 = log2transform(normal_samples)
      # Perform YuGene normalization.
      print(paste0("Performing YuGene normalization for ", tissue_type, "."))
      normal_samples_YuGene = YuGene(normal_samples_log2)
      normal_samples_YuGene = normal_samples_YuGene[1:nrow(normal_samples_YuGene),1:ncol(normal_samples_YuGene)] # This step is needed because apparently YuGene() returns a 'YuGene' object, not a matrix, and using colnames() does something funky
      normal_samples = normal_samples_YuGene
    }
    
    # Make sure the gene names are the same between the patient sample and the normal tissue. 
    # Clean the GTEx Ensembl IDs (GTEx adds .version part to the Ensembl ID.)
    ids = cleanid(rownames(normal_samples))
    df = grex(ids)
    # Add the Entrez/HUGO ID to the normal_samples matrix.
    if(base::grepl("^Hugo$", id_type, ignore.case = T)) {
      normal_samples = cbind(df$hgnc_symbol, normal_samples) 
    } else if(base::grepl("^Entrez$", id_type, ignore.case = T)) {
      normal_samples = cbind(df$entrez_id, normal_samples)
    } else {
      normal_samples = cbind(df$hgnc_symbol, normal_samples) 
    }
    # Remove rows (Ensembl IDs) without a corresponding Entrez/HUGO ID.
    normal_samples = normal_samples[complete.cases(normal_samples),]
    # Merge duplicate Entrez IDs.
    # For each gene (yes, all 20,000, or however many have an Entrez/HUGO ID), find all the probesets that map to that gene, then average the expression of those probesets. Add the average expression to a new matrix. 
    ncol_normal_samples = ncol(normal_samples)   # Columns = samples (cell lines.)
    map = data.frame(Ensembl=rownames(normal_samples), ID=normal_samples[,1])
    normal_samples_rownames = rownames(normal_samples)
    normal_samples = apply(normal_samples, 2, as.numeric)
    rownames(normal_samples) = normal_samples_rownames
    #normal_samples2 = matrix(, nrow = length(unique(map$Entrez)), ncol = (ncol_normal_samples - 1)) # ncol_normal_samples - 1 because the first column of normal_samples is the Entrez ID. 
    
    # Run the avgExprs() function.   
    #print(paste0("Performing benchmark test for avgExprs() function."))
    #system.time({
    #  test = foreach(i = 1:20, .combine = 'cbind') %dopar% {
    #    avgExprs(i, normal_samples, map)
    #  }
    #})
    print(paste0("Merging duplicate probes for genes for tissue ", tissue_type, "."))
    normal_samples2 = foreach(i = 1:length(unique(map$ID)), .combine = 'cbind') %dopar% {
      avgExprs(i, normal_samples, map)
    }
    #system.time({test = future_sapply(1:2, avgExprs, normal_samples = normal_samples)})
    #normal_samples2 = future_sapply(1:length(unique(map$ID)), avgExprs, normal_samples = normal_samples)
    
    print(paste0("Setting row/column names for ", tissue_type, "."))
    # Set the _colnames_ of the expression matrix to be the Entrez/HUGO IDs. 
    colnames(normal_samples2) = unique(map$ID)
    # Set the _rownames_ of the expression matrix to be the samples.
    rownames(normal_samples2) = colnames(normal_samples)[2:ncol(normal_samples)]
    
    print(paste0("Saving the matrix of merged probes for ", tissue_type, "."))
    # Save the matrix of merged probes.
    saveRDS(normal_samples2, file_name)
    # Save the matrix of merged probes to the list.
    #normal_samples_processed[[tissue_type]] = normal_samples2
    
    # Clean up.
    print(paste0("Cleaning up."))
    rm(normal_samples2, normal_samples, normal_samples_log2, normal_samples_YuGene)
    gc()
    
  } else {
    print(paste0(tissue_type, " has already been processed!"))
  }
}

# Functions.
generateRandCombos = function(i, pData, drug_col_name, n_rand, unique_drugs) {
  # Even though the function is named "generateRandCombos", it will include the one actual experimental combination as well. 
  
  combos_GSM = c()
  # Get the combo for this GSM. 
  drugs = strsplit(pData[i,drug_col_name][[1]], "_")[[1]]
  combos_GSM = c(combos_GSM, paste(drugs, collapse = "_")) 
  
  # Create a table of combinations with the same number of drugs as the combination for this GSM.
  comb_table = as.matrix(t(combn(unique_drugs, length(drugs))))
  # For each combination in comb_table, check if it is the same as the combination tested.
  is_test_combo = future_sapply(1:nrow(comb_table), FUN = checkIsTestCombo, comb_table = comb_table, drugs = drugs)
  # Remove any combinations that are the same as the test combination.
  comb_table = as.matrix(comb_table[!is_test_combo,])
  
  # Select a random subset of the combinations.
  set.seed(i)
  rand_combos = as.matrix(comb_table[sample(1:nrow(comb_table), n_rand),])
  # Add these combinations to combos_GSM
  if(ncol(rand_combos) > 1) {
    combos_GSM = c(combos_GSM, pasteCols(t(rand_combos), sep="_"))
  } else {
    combos_GSM = c(combos_GSM, rand_combos[,1])
  } 
  
  return(combos_GSM)
  # Add combos_GSM to final_combos.
  #final_combos <<- c(final_combos, combos_GSM)
}

checkIsTestCombo = function(j, comb_table, drugs) {
  drugs_j = comb_table[j,]
  return(ifelse(setequal(drugs_j, drugs), TRUE, FALSE))
  #is_test_combo = c(is_test_combo, ifelse(setequal(drugs_j, drugs), TRUE, FALSE))
}
