# Read in the CCLE metadata file. 
CCLE_metadata = read.csv(CCLE_metadata_filename)

cel_file_list = list.files(path = CCLE_file_list_directory, pattern = NULL,
                           full.names = TRUE, recursive = TRUE,
                           ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)
GPL = "GPL13667"

# Create matrix to hold gene-expression data. 
CCLE_exprs_complete = matrix(, nrow = 19456, ncol = nrow(CCLE_metadata))

# Process each cell line individually.
for(i in 1:nrow(CCLE_metadata)) {
  cell_line_name = CCLE_metadata[i, "Characteristics.cell.line."]
  cel_file_name = CCLE_metadata[i, "Array.Data.File"]
  cel_file_path = cel_file_list %>% regexPipes::grep(cel_file_name, value = T) %>% .[1]
  
  # Check if this one's been done already. 
  if(file.exists(paste0(processed_data_dir, "CCLE/exprs_", cell_line_name, "_", preprocessing_method, "_", id_type, ".rds"))) next
  
  print(paste0("Processing cell line ", cell_line_name, ", ", i, " of ", nrow(CCLE_metadata), "."))
  
  # 1. Read in raw data and normalize using YuGene. 
  possibleError = tryCatch( # https://stackoverflow.com/a/8094059, https://stackoverflow.com/a/12195574
    {
      # Read in the raw data and BG correct.
      exprs_bgcorrected = processAffySingleCEL(cel_file_path)
      
      # Perform log2 transformation. 
      exprs_log2 = log2transform(exprs_bgcorrected)
      # Perform YuGene normalization.
      exprs_YuGene = YuGene(exprs_log2)
      exprs_YuGene = exprs_YuGene[1:nrow(exprs_YuGene),1:ncol(exprs_YuGene)] # This step is needed because apparently YuGene() returns a 'YuGene' object, not a matrix, and using colnames() does something funky
      
    }, error=function(cond) {
      message(paste0("There was a problem processing the cell line ", cell_line_name, ". Skipping this set."))
      message(cond)
    }
  )
  if(inherits(possibleError, "error")) {
    print(paste0("There was a problem processing the cell line ", cell_line_name, ". Skipping this set."))
    
    # Write the error to the appropriate file. 
    fileConn = file(GSE_error_file)
    writeLines(paste0("There was a problem processing the cell line ", cell_line_name, ". Skipping this set."), fileConn)
    close(fileConn)
    
  } 
  
  # 2. Annotate the probes.
  exprs_YuGene = as.matrix(exprs_YuGene)
  exprs_annotated = annotateAffySingleCEL(exprs_YuGene, GPL)
  
  # 3. Save the annotated expression matrix (well, more like expression vector since it's for a single cell line.)
  saveRDS(exprs_annotated, paste0(processed_data_dir, "CCLE/exprs_", cell_line_name, "_", preprocessing_method, "_", id_type, ".rds"))
  
  # 4. Also add the expression vector to the matrix. 
  colnames(exprs_annotated)[1] = as.character(cell_line_name)
  CCLE_exprs_complete[,i] = exprs_annotated
  
  if(!exists("gene_names")) gene_names = rownames(exprs_annotated)
  
  # Clean up.
  rm(exprs_YuGene, exprs_annotated)
  gc()
}

# Set the row/column names.
dimnames(CCLE_exprs_complete) = list(gene_names, CCLE_metadata$Characteristics.cell.line.)

# Save the final expression matrix.
saveRDS(CCLE_exprs_complete, CCLE_exprs_complete_filename)

# Correct cell-line names if necessary. 
# ...