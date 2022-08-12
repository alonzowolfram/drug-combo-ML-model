processAffy = function(path, is_GSE, data_dir, filename_list) {
  # Read in .CEL files as AffyBatch object. 
  if(is_GSE) {
    cel_files_path = paste0(folder_generator(
      data_dir = data_dir,
      data_set = "GEO",
      data_type = "Expression",
      processing_stage = "Raw",
      additional_paths = paste0(path, "/", path, "_RAW/")
    ))
  } else {
    cel_files_path = paste0(path)
  }
  
  cel_files = list.files(path = cel_files_path, pattern = "^.+\\.CEL$", ignore.case = T, full.names = T)
  affy_raw = ReadAffy(filenames=cel_files)
  # Background correct. 
  affy_bgcorrected = expresso(affy_raw, bg.correct=TRUE, bgcorrect.method="rma", normalize=FALSE, pmcorrect.method="pmonly", summary.method="avgdiff")
  
  return(exprs(affy_bgcorrected))
}





processLumi = function(path, is_GSE, data_dir, filename_list) {
  if(is_GSE) {
    raw_files_path = paste0(folder_generator(
      data_dir = data_dir,
      data_set = "GEO",
      data_type = "Expression",
      processing_stage = "Raw",
      additional_paths = paste0(path, "/", path, "_RAW/")
    ))
  } else {
    raw_files_path = paste0(path)
  }
  # Read in raw files as lumibatch object.
  raw_files = list.files(path = raw_files_path, pattern = "^.+\\.txt$", full.names = T)
  lumi_raw = lumiR(raw_files[1], lib.mapping = 'lumiHumanIDMapping')
  # Background correct. 
  lumi_bgcorrected = lumiB(lumi_raw, "bgAdjust.affy")
  
  # Filter by detection score cutoff.
  cutoff = detectionCall(lumi_raw) # Get the count of probes which passed the detection threshold per sample.
  identical(featureNames(lumi_bgcorrected), names(cutoff)) # Make sure the probes are in the right order in both the normalized ExpressionSet and the cutoff counts.
  exprs = exprs(lumi_bgcorrected)[which(rowSums(exprs(lumi_bgcorrected)) > 0),] # Drop any probes where none of the samples passed detection threshold. https://www.biostars.org/p/89891/#383862
  # Drop unannotated probes.
  symbols = getSYMBOL(rownames(exprs), 'lumiHumanAll.db') %>% is.na 
  exprs_annot = exprs[!symbols,] 
  
  return(exprs_annot)
}





processAffySingleCEL = function(path, data_dir) {
  # Read in .CEL file as AffyBatch object. 
  affy_raw = ReadAffy(filenames=path)
  # Background correct. 
  affy_bgcorrected = expresso(affy_raw, bg.correct=TRUE, bgcorrect.method="rma", normalize=FALSE, pmcorrect.method="pmonly", summary.method="avgdiff")
  
  return(exprs(affy_bgcorrected))
}




annotateAffy = function(GSE, exprs, GPL, annotation) {
  # id_type should be set globally, do not need to make it an argument ... right?
  
  # We don't need to remove the prefixes since we will be converting the probeset/probes to ENTREZ IDs ourselves, and the databases (hgu133plus2.db etc.) have the full probeset/probe names including the prefixes.
  # Remove the file extensions.
  colnames(exprs) = base::gsub("\\..*$", "", colnames(exprs))
  colnames(exprs) = base::gsub("_.*$", "", colnames(exprs))
  
  # Remove AFFX probesets.
  # eset = eset[!base::grepl("AFFX", featureNames(eset)),]
  exprs = exprs[!base::grepl("AFFX", rownames(exprs)),,drop=F]
  
  # Get the appropriate annotation database.
  # https://www.bioconductor.org/packages/devel/bioc/manuals/AnnotationDbi/man/AnnotationDbi.pdf See "Accnum" section. 
  if(base::grepl("^Hugo$", id_type, ignore.case = T)) {
    IDs = AnnotationDbi::select(annotation_dbs[[GPL]], rownames(exprs), "SYMBOL") 
  } else if(base::grepl("^Entrez$", id_type, ignore.case = T)) {
    IDs = AnnotationDbi::select(annotation_dbs[[GPL]], rownames(exprs), "ENTREZID") 
  } else {
    IDs = AnnotationDbi::select(annotation_dbs[[GPL]], rownames(exprs), "SYMBOL") 
  }
  # Rename from "ENTREZID" or "SYMBOL" to "ID".
  colnames(IDs)[2] = "ID"
  
  # Remove all the probesets without an Entrez/HUGO ID.
  IDs = IDs[complete.cases(IDs),,drop=FALSE]
  probes_with_entrez = intersect(rownames(exprs), IDs$PROBEID)
  exprs = exprs[which(rownames(exprs) %in% probes_with_entrez),,drop=FALSE]
  IDs = IDs[which(IDs$PROBEID %in% probes_with_entrez),,drop=FALSE]
  
  # Remove duplicate PROBESETS in the entrez ID table. We will just keep the first annotation of each multiple-mapping probe, per https://support.bioconductor.org/p/87032/.
  IDs = IDs[!duplicated(IDs$PROBEID),,drop=FALSE]
  identical(IDs$PROBEID, rownames(exprs))
  # We won't set the rownames of the expression matrix to be entrez IDs yet, because we still need the probeset names when we merge duplicate entrez IDs.
  
  # Merge duplicate ENTREZ/HUGO IDS. 
  # For each gene (yes, all 20,000, or however many have an Entrez/HUGO ID), find all the probesets that map to that gene, then average the expression of those probesets. Add the average expression to a new matrix. 
  ncol_exprs = ncol(exprs)   # Columns = samples (cell lines.)
  
  exprs2 = matrix(, nrow = 0, ncol = ncol_exprs)
  for(i in unique(IDs$ID)) {
    # Get all the probesets that map to gene i. This will be a vector. 
    probesets_i = IDs[which(IDs$ID==i),1] # The first column is the one with the probesets.
    # Get the expression of all probesets_i across all cell lines. This will be an m x n matrix, with cell lines as the columns and probesets as the rows. 
    gene_i_matrix = matrix(, nrow = 0, ncol = ncol_exprs) # Since we'll be rbinding, don't set nrow = length(probesets_i). 
    for(j in probesets_i) {
      gene_i_matrix = rbind(gene_i_matrix, exprs[which(rownames(exprs)==j),])
    }
    
    # Average the expressions column-wise, giving us a 1 x n vector. 
    avg_i_expression = apply(gene_i_matrix, 2, mean)
    # Add this vector containing the average expression of all probesets for gene i to the exprs matrix. 
    exprs2 = rbind(exprs2, avg_i_expression)
  }
  
  # Set the rownames of the expression matrix to be the entrez IDs. 
  rownames(exprs2) = unique(IDs$ID)
  
  return(exprs2)
}






annotateLumi = function(GSE, exprs) {
  # Convert nuIDs to Entrez IDs.
  # https://www.bioconductor.org/packages//2.7/bioc/vignettes/lumi/inst/doc/IlluminaAnnotation.pdf
  lumiHumanIDMapping_nuID()
  mappingInfo = as.data.frame(nuID2RefSeqID(rownames(exprs), lib.mapping = 'lumiHumanIDMapping', returnAllInfo = TRUE))
  View(head(mappingInfo))
  if(base::grepl("^Entrez$", id_type, ignore.case = T)) {
    mappingInfo = mappingInfo[,c("Accession", "EntrezID")]
  } else {
    mappingInfo = mappingInfo[,c("Accession", "Symbol")]
  }
  colnames(mappingInfo)[2] = "ID"
  # Remove all the nuIDs without an Entrez ID/gene symbol.
  mappingInfo[mappingInfo==""] = NA
  mappingInfo = mappingInfo[complete.cases(mappingInfo),]
  exprs = exprs[rownames(exprs) %in% rownames(mappingInfo),]
  
  # Since each nuID maps to only one Entrez ID/gene symbol, we don't have to remove duplicate nuIDs the way we had to remove duplicate Affy probes.
  # However, each Entrez ID/gene symbol can have more than one nuID mapping to it, so for each Entrez ID/gene symbol, we will take the average of all the nuIDs mapping to it. 
  ncol_exprs = ncol(exprs)   # Columns = samples.
  exprs2 = matrix(, nrow = 0, ncol = ncol_exprs)
  for(i in unique(mappingInfo$ID)) {
    # Get all the probesets that map to gene i. This will be a vector. 
    probesets_i = rownames(mappingInfo)[which(mappingInfo$ID==i)] 
    # Get the expression of all probesets_i across all samples. This will be an m x n matrix, with cell lines as the columns and probesets as the rows. 
    gene_i_matrix = matrix(, nrow = 0, ncol = ncol_exprs) # Since we'll be rbinding, don't set nrow = length(probesets_i). 
    for(j in probesets_i) {
      gene_i_matrix = rbind(gene_i_matrix, exprs[which(rownames(exprs)==j),])
    }
    
    # Average the expressions column-wise, giving us a 1 x n vector. 
    avg_i_expression = apply(gene_i_matrix, 2, mean)
    # Add this vector containing the average expression of all probesets for gene i to the exprs matrix. 
    exprs2 = rbind(exprs2, avg_i_expression)
  }
  # Set the rownames of the expression matrix to be the entrez IDs. 
  rownames(exprs2) = unique(mappingInfo$ID)
  
  return(exprs2)
  
}





annotateAffySingleCEL = function(exprs, GPL) {
  # We don't need to remove the prefixes since we will be converting the probeset/probes to ENTREZ IDs ourselves, and the databases (hgu133plus2.db etc.) have the full probeset/probe names including the prefixes.
  # Remove the file extensions.
  colnames(exprs) = base::gsub("\\..*$", "", colnames(exprs))
  colnames(exprs) = base::gsub("_.*$", "", colnames(exprs))
  
  # Remove AFFX probesets.
  # eset = eset[!base::grepl("AFFX", featureNames(eset)),]
  exprs = exprs[!base::grepl("AFFX", rownames(exprs)),,drop=F]
  
  # Get the appropriate annotation database.
  # https://www.bioconductor.org/packages/devel/bioc/manuals/AnnotationDbi/man/AnnotationDbi.pdf See "Accnum" section. 
  if(base::grepl("^Hugo$", id_type, ignore.case = T)) {
    IDs = AnnotationDbi::select(annotation_dbs[[GPL]], rownames(exprs), "SYMBOL") 
  } else if(base::grepl("^Entrez$", id_type, ignore.case = T)) {
    IDs = AnnotationDbi::select(annotation_dbs[[GPL]], rownames(exprs), "ENTREZID") 
  } else {
    IDs = AnnotationDbi::select(annotation_dbs[[GPL]], rownames(exprs), "SYMBOL") 
  }
  # Rename from "ENTREZID" or "SYMBOL" to "ID".
  colnames(IDs)[2] = "ID"
  
  # Remove all the probesets without an Entrez/HUGO ID.
  IDs = IDs[complete.cases(IDs),,drop=FALSE]
  probes_with_entrez = intersect(rownames(exprs), IDs$PROBEID)
  exprs = exprs[which(rownames(exprs) %in% probes_with_entrez),,drop=FALSE]
  IDs = IDs[which(IDs$PROBEID %in% probes_with_entrez),,drop=FALSE]
  
  # Remove duplicate PROBESETS in the entrez ID table. We will just keep the first annotation of each multiple-mapping probe, per https://support.bioconductor.org/p/87032/.
  IDs = IDs[!duplicated(IDs$PROBEID),,drop=FALSE]
  identical(IDs$PROBEID, rownames(exprs))
  # We won't set the rownames of the expression matrix to be entrez IDs yet, because we still need the probeset names when we merge duplicate entrez IDs.
  
  # Merge duplicate ENTREZ/HUGO IDS. 
  # For each gene (yes, all 20,000, or however many have an Entrez/HUGO ID), find all the probesets that map to that gene, then average the expression of those probesets. Add the average expression to a new matrix. 
  ncol_exprs = ncol(exprs)   # Columns = samples (cell lines.)
  
  exprs2 = matrix(, nrow = 0, ncol = ncol_exprs)
  for(i in unique(IDs$ID)) {
    # Get all the probesets that map to gene i. This will be a vector. 
    probesets_i = IDs[which(IDs$ID==i),1] # The first column is the one with the probesets.
    # Get the expression of all probesets_i across all cell lines. This will be an m x n matrix, with cell lines as the columns and probesets as the rows. 
    gene_i_matrix = matrix(, nrow = 0, ncol = ncol_exprs) # Since we'll be rbinding, don't set nrow = length(probesets_i). 
    for(j in probesets_i) {
      gene_i_matrix = rbind(gene_i_matrix, exprs[which(rownames(exprs)==j),])
    }
    
    # Average the expressions column-wise, giving us a 1 x n vector. 
    avg_i_expression = apply(gene_i_matrix, 2, mean)
    # Add this vector containing the average expression of all probesets for gene i to the exprs matrix. 
    exprs2 = rbind(exprs2, avg_i_expression)
  }
  
  # Set the rownames of the expression matrix to be the entrez IDs. 
  rownames(exprs2) = unique(IDs$ID)
  
  return(exprs2)
}





avgExprs = function(i, normal_samples, map, parallel = "none") {
  # Get all the probesets that map to (Entrez/HUGO) gene i. This will be a vector. 
  id = as.character(unique(map$ID)[i])
  
  probesets_i = as.character(map[which(map$ID==id),1]) # The first column is the one with the Ensembl names. Probeset = Ensembl ID, because I was too lazy to change it. 
  # Get the expression of all probesets_i across all cell lines. This will be an m x n matrix, with cell lines as the rows and probesets as the columns. (Originally cell lines as columns and probesets as rows, which is why we have to use 1 as the margin for avg_i_expression = future_apply() below.)
  # Use different parallelization (or no parallelization) based on the "parallel" argument passed to avgExprs.
  if(parallel == "none") {
    gene_i_matrix = sapply(1:length(probesets_i), createGeneiMatrix, probesets_i = probesets_i, normal_samples = normal_samples)
    avg_i_expression = as.matrix(apply(gene_i_matrix, 1, mean))
    
  } else if(parallel == "future") {
    gene_i_matrix = future_sapply(1:length(probesets_i), createGeneiMatrix, probesets_i = probesets_i, normal_samples = normal_samples)
    avg_i_expression = as.matrix(future_apply(gene_i_matrix, 1, mean))
    
  } else if(parallel == "foreach") {
    gene_i_matrix = as.matrix(foreach(i = 1:length(probesets_i), .combine = 'cbind') %dopar% {
      createGeneiMatrix(i, probesets_i, normal_samples) # Returns matrix with the same setup as normal_samples, which is genes as rows, samples as columns.
    })
    avg_i_expression = foreach(i = 1:nrow(gene_i_matrix), .combine = 'rbind') %dopar% {
      mean(gene_i_matrix[i,])
    }
    
  } else {
    # If an argument is passed that is not "none," "future," or "foreach," default to none.
    gene_i_matrix = sapply(1:length(probesets_i), createGeneiMatrix, probesets_i = probesets_i, normal_samples = normal_samples)
    avg_i_expression = as.matrix(apply(gene_i_matrix, 1, mean))
  }
  
  rownames(avg_i_expression) = rownames(gene_i_matrix)
  
  # Return the vector containing the average expression of all probesets for gene i. 
  return(avg_i_expression) 
}





createGeneiMatrix = function(j, probesets_i, normal_samples) {
  ensembl = probesets_i[j]
  return(normal_samples[which(rownames(normal_samples)==ensembl),2:ncol(normal_samples)])
}





log2transform = function(x, parallel = "none") {
  # x = matrix
  min = min(x)
  correction_factor = ifelse(min <= 0, -min + 1, 0)
  
  if(parallel == "none") {
    return(apply(x + correction_factor, c(1,2), log2))
    
  } else if(parallel == "future") {
    return(future_apply(x + correction_factor, c(1,2), log2))
    
  } else if(parallel == "foreach") {
    x_new = foreach(j = 1:ncol(x), .combine='cbind') %:%
      foreach(i = 1:nrow(x), .combine='c') %dopar% {
        log2((x[i,j]+correction_factor))
      }
    rownames(x_new) = rownames(x)
    colnames(x_new) = colnames(x)
    return(x_new)
    
  } else {
    return(apply(x + correction_factor, c(1,2), log2))
  }
}