# Function for checking if ALMANAC data has been loaded.
checkALMANACLoaded = function() {
  if(!exists("almanac_dat")) {
    if(!file.exists(paste0(data_dir, "NCI-ALMANAC/almanac_dat.rds"))) {
      print("almanac_dat variable and almanac_dat.rds file not found. Creating file from scratch.")
      loadAndCleanALMANACData()
    } else {
      # Alternatively, if it is already saved to an RDS file:
      print("almanac_dat variable not found, but almanac_dat.rds file exists. Loading from file.")
      almanac_dat = readRDS(paste0(data_dir, "NCI-ALMANAC/almanac_dat.rds"))
      assign("almanac_dat", almanac_dat, envir = .GlobalEnv)
      # Check that it is in the data.table format.
      if("data.table" %in% class(almanac_dat)) {
        # Do nothing. 
      } else {
        print("almanac_dat variable not found, and almanac_dat.rds file exists but not formatted correctly. Creating file from scratch.")
        loadAndCleanALMANACData()
      }
    }
  } else {
    print("almanac_dat is already loaded.")
  }
}




# --------------------------------------------------------




# Function for loading raw ALMANAC data and cleaning. 
loadAndCleanALMANACData = function() {
  print("Loading and cleaning ALMANAC data.")
  
  # Load the ALMANAC data.
  almanac_dat = read.csv(paste0(data_dir, "NCI-ALMANAC/ComboDrugGrowth_Nov2017.csv"))
  # Convert it to a data.table. 
  setDT(almanac_dat)
  
  # Standardize the cell-line names so they match those in the NCI-60 data. Specifically, make the following changes manually:
  # Remove the forward slashes when they're not part of the cell name.
  # When slashes (forward or backslashes) are part of the cell name, change them to dashes. 
  nci60_names = read.csv(paste0(data_dir, "NCI-60_cell_line_names_ALMANAC_GEO_GDSC_CCLE.csv"), colClasses = rep("character", 6)) # Maps the different cell line names to the standardized names.
  almanac_dat$CELLNAME = as.factor(base::gsub("NCI/", "NCI-", almanac_dat$CELLNAME))
  almanac_dat$CELLNAME = as.factor(base::gsub("/.+$", "", almanac_dat$CELLNAME))
  almanac_dat$CELLNAME = as.factor(base::gsub("\\\032", "-032", almanac_dat$CELLNAME))
  almanac_dat$CELLNAME = as.factor(base::gsub("HL-60\\(TB\\)", "HL-60", almanac_dat$CELLNAME))
  
  # Now we can standardize using the table.
  # Replace a with b. 
  a = nci60_names$ALMANAC_name
  b = nci60_names$Standardized_name
  names(b) = a
  almanac_dat$CELLNAME = as.factor(str_replace_all(almanac_dat$CELLNAME, b))
  
  # Make sure the order of the cell lines in the ALMANAC data matches those in the eset. 
  cols = c("CELLNAME", "NSC1", "NSC2")
  setkeyv(almanac_dat, cols, verbose=getOption("datatable.verbose"))
  # identical(unique(as.character(almanac_dat$CELLNAME)), colnames(exprs(GSE32474)))
  # Well, the order may be the same, but the ALMANAC data has 2 more cell lines than the GEO data.
  
  # Calculate the CORRECT PercentGrowth. T_T
  almanac_dat$PERCENTGROWTHCORRECTED = 100 * (almanac_dat$TESTVALUE - almanac_dat$TZVALUE) / (almanac_dat$CONTROLVALUE - almanac_dat$TZVALUE)
  sum(round(almanac_dat$PERCENTGROWTHCORRECTED, 2)==round(almanac_dat$PERCENTGROWTH, 2))/nrow(almanac_dat) # See if the corrected PercentGrowth generally matches the supplied PercentGrowth. 
  
  # Save the ALMANAC data. 
  saveRDS(almanac_dat, paste0(data_dir, "NCI-ALMANAC/almanac_dat.rds")) # Originally just in data_dir.
  # As of 06/15/2019, the saved ALMANAC data has the cell-line names standardized and the PERCENTGROWTH column is NOT in decimal format (i.e. it's in percentage format.) 
  assign("almanac_dat", almanac_dat, envir = .GlobalEnv)

}




# --------------------------------------------------------



# Function for checking if the gene-descriptor subset has been loaded.
checkGeneDescriptorSubsetLoaded = function() {
  if(!exists("exprs_sub")) {
    if(!file.exists(paste0(data_dir, "NCBI-GEO/NCI-60_exprs_", gene_subset_type, "_sub.rds"))) {
      print(paste0("exprs_sub not loaded, and no RDS file found for the gene subset ", gene_subset_type, ". Preparing the data from scratch."))
      prepareGeneDescriptorSubset()
    } else {
      # Alternatively, if it is already saved to an RDS file:
      print(paste0("exprs_sub not loaded, but an RDS file has been found for the gene subset ", gene_subset_type, ". Loading data from the RDS file."))
      exprs_sub = readRDS(paste0(data_dir, "NCBI-GEO/NCI-60_exprs_", gene_subset_type, "_sub.rds"))
      assign("exprs_sub", exprs_sub, envir = .GlobalEnv)
    }
  } else {
    print("exprs_sub is already loaded.")
  } 
}




# --------------------------------------------------------



# Function for preparing gene-descriptor subset.
prepareGeneDescriptorSubset = function() {
  if(!file.exists(paste0(data_dir, "NCBI-GEO/GSE32474_exprs_", gene_subset_type, "_sub.rds"))) {
    if(!exists("exprs")) {
      eset = readRDS(paste0(data_dir, "NCBI-GEO/GSE32474_cleaned.rds"))
      exprs = exprs(eset)
    }
    
    # Load the LINCS L1000 landmark genes.
    L1000_landmark_genes = read.csv(paste0(data_dir, "978_L1000_landmark_genes.csv"))
    predictor_genes = L1000_landmark_genes$Entrez.ID
    # Subset the GSE32474 expression matrix to include only the LINCS L1000 landmark genes. 
    exprs_sub = exprs[rownames(exprs) %in% predictor_genes,,drop=FALSE]
    # Save the subset.
    saveRDS(exprs_sub, paste0(data_dir, "NCBI-GEO/GSE32474_exprs_", gene_subset_type, "_sub.rds"))
  } else {
    exprs_sub = readRDS(paste0(data_dir, "NCBI-GEO/GSE32474_exprs_", gene_subset_type, "_sub.rds"))
  }
  
  if(!file.exists(paste0(data_dir, "GDSC/Transcriptional_profiling_of_1000_cell_lines/Esets/GDSC_MDA-MB-468_exprs_", gene_subset_type, "_sub.rds"))) {
    eset_MDAMB468 = readRDS(paste0(data_dir, "GDSC/Transcriptional_profiling_of_1000_cell_lines/Esets/GDSC_MDA-MB-468_cleaned.rds"))
    exprs_MDAMB468 = exprs(eset_MDAMB468)
    
    # Load the LINCS L1000 landmark genes.
    L1000_landmark_genes = read.csv(paste0(data_dir, "978_L1000_landmark_genes.csv"))
    predictor_genes = L1000_landmark_genes$Entrez.ID
    # Subset the GSE32474 expression matrix to include only the LINCS L1000 landmark genes. 
    exprs_MDAMB468_sub = exprs_MDAMB468[rownames(exprs_MDAMB468) %in% predictor_genes,,drop=FALSE]
    # Save the subset.
    saveRDS(exprs_MDAMB468_sub, paste0(data_dir, "GDSC/Transcriptional_profiling_of_1000_cell_lines/Esets/GDSC_MDA-MB-468_exprs_", gene_subset_type, "_sub.rds"))
  } else {
    exprs_MDAMB468_sub = readRDS(paste0(data_dir, "GDSC/Transcriptional_profiling_of_1000_cell_lines/Esets/GDSC_MDA-MB-468_exprs_", gene_subset_type, "_sub.rds"))
  }
  
  # Check if there are any missing genes in the MDA-MB-468 expression that need to be imputed.
  if(length(setdiff(rownames(exprs_sub), rownames(exprs_MDAMB468_sub))) > 0) {
    # Add to exprs_MDAMB468_sub the _rows_ that are missing. 
    exprs_MDAMB468_sub = rbind(exprs_MDAMB468_sub, matrix(, nrow = 1, ncol = 1, dimnames = list(setdiff(rownames(exprs_sub), rownames(exprs_MDAMB468_sub)))))
    # Make sure the rownames of exprs_MDAMB468_sub and exprs_sub are in the same order.
    exprs_sub = exprs_sub[order(rownames(exprs_sub)),]
    exprs_MDAMB468_sub = exprs_MDAMB468_sub[order(rownames(exprs_MDAMB468_sub)),,drop=F]
    identical(rownames(exprs_MDAMB468_sub), rownames(exprs_sub))
    
    # Cbind exprs_MDAMB468_sub to exprs_sub and impute missing values.
    exprs_sub = cbind(exprs_sub, exprs_MDAMB468_sub)
    exprs_sub = impute.knn(exprs_sub)$data
    
    # Order columns (cell lines) by name. 
    exprs_sub = exprs_sub[,order(colnames(exprs_sub))]
    
    # Save as RDS.
    saveRDS(exprs_sub, paste0(data_dir, "NCBI-GEO/NCI-60_exprs_", gene_subset_type, "_sub.rds"))
    assign("exprs_sub", exprs_sub, envir = .GlobalEnv)
    
  } else {
    # No missing genes, just cbind and order.
    # Make sure the rownames of exprs_MDAMB468_sub and exprs_sub are in the same order.
    exprs_sub = exprs_sub[order(rownames(exprs_sub)),]
    exprs_MDAMB468_sub = exprs_MDAMB468_sub[order(rownames(exprs_MDAMB468_sub)),,drop=F]
    identical(rownames(exprs_MDAMB468_sub), rownames(exprs_sub))
    
    # Cbind.
    exprs_sub = cbind(exprs_sub, exprs_MDAMB468_sub)
    
    # Order columns (cell lines) by name. 
    exprs_sub = exprs_sub[,order(colnames(exprs_sub))]
    
    # Save as RDS.
    saveRDS(exprs_sub, paste0(data_dir, "NCBI-GEO/NCI-60_exprs_", gene_subset_type, "_sub.rds"))
    assign("exprs_sub", exprs_sub, envir = .GlobalEnv)
  }
}




# --------------------------------------------------------




# Function for checking if reduced chemical descriptors have been loaded.
checkChemDescriptorsLoaded = function() {
  if(!exists("almanac_compound_descriptors_reduced")) {
    if(!file.exists(paste0(data_dir, "NCI-ALMANAC/almanac_compounds_", descriptor_type, "_descriptors_individual_compounds_reduced.rds"))) {
      loadAndCleanChemDescriptors()
    } else {
      # Alternatively, if it is already saved to an RDS file:
      almanac_compound_descriptors_reduced = readRDS(paste0(data_dir, "NCI-ALMANAC/almanac_compounds_", descriptor_type, "_descriptors_individual_compounds_reduced.rds"))
      assign("almanac_compound_descriptors_reduced", almanac_compound_descriptors_reduced, envir = .GlobalEnv)
    }
  } else {
    print("almanac_compound_descriptors_reduced is already loaded.")
  } 
}




# --------------------------------------------------------




# Function for loading and cleaning chemical descriptors.
loadAndCleanChemDescriptors = function() {
  if(!file.exists(paste0(data_dir, chemical_descriptor_reduced_filename))) {
    # Read in the original matrix of descriptors for the single compounds.
    single_cmpd_descriptors = read.csv(paste0(data_dir, chemical_descriptor_filename)) # Originally just in data_dir. 
    
    # Set the rownames to be the Name variable, and then remove the Name variable as a column.
    rownames(single_cmpd_descriptors) = single_cmpd_descriptors$Name
    single_cmpd_descriptors = single_cmpd_descriptors[,-1]
    # Convert "na"s to NAs and take out those columns. 
    single_cmpd_descriptors[single_cmpd_descriptors=="na"] = NA
    single_cmpd_descriptors = t(single_cmpd_descriptors)
    single_cmpd_descriptors = single_cmpd_descriptors[complete.cases(single_cmpd_descriptors),]
    single_cmpd_descriptors = t(single_cmpd_descriptors)
    # Save the rownames.
    cmpd_names = rownames(single_cmpd_descriptors)
    
    # Convert to numeric. https://stackoverflow.com/a/19146419
    single_cmpd_descriptors = apply(single_cmpd_descriptors, 2, as.numeric)
    # Remove columns (variables) with 0 variance. https://stackoverflow.com/a/40317343
    single_cmpd_descriptors = single_cmpd_descriptors[,which(apply(single_cmpd_descriptors, 2, var) != 0)]
    
    # Remove collinear variables. 
    # https://stackoverflow.com/a/18275778
    descriptor_cor = cor(single_cmpd_descriptors)
    descriptor_cor[upper.tri(descriptor_cor)] = 0
    diag(descriptor_cor) = 0
    almanac_compound_descriptors_reduced = single_cmpd_descriptors[,!apply(descriptor_cor,2,function(x) any(x > 0.8))] # 0.8 chosen as cutoff based on http://courses.washington.edu/urbdp520/UDP520/Lab7_MultipleRegression.pdf. But we could go with 0.9 if there are too few. 
    # For right now, we won't consider multicollinearity, but if at some point in the future we do, here are some resources: https://www.r-bloggers.com/dealing-with-the-problem-of-multicollinearity-in-r/
    # Reset the rownames
    rownames(almanac_compound_descriptors_reduced) = cmpd_names
    View(head(almanac_compound_descriptors_reduced))
    # This will leave us with a couple hundred descriptors.
    
    # Save this dimension-reduced matrix (input for create_compound_descriptor_matrix_TACC.R).
    saveRDS(almanac_compound_descriptors_reduced, paste0(data_dir, chemical_descriptor_reduced_filename))
    assign("almanac_compound_descriptors_reduced", almanac_compound_descriptors_reduced, envir = .GlobalEnv)
  } else {
    almanac_compound_descriptors_reduced = readRDS(paste0(data_dir, chemical_descriptor_reduced_filename))
    assign("almanac_compound_descriptors_reduced", almanac_compound_descriptors_reduced, envir = .GlobalEnv)
    View(head(almanac_compound_descriptors_reduced))
  }
}




# --------------------------------------------------------



# Function for checking if the cell-line-drug permutations have been loaded.
checkCLDPReducedLoaded = function() {
  if(!exists("cell_line_drug_perms_reduced")) {
    if(!file.exists(paste0(data_dir, "NCI-ALMANAC/cell_line_drug_perms_reduced.rds"))) {
      print("cell_line_drug_perms_reduced variable and cell_line_drug_perms_reduced.rds file not found. Creating file from scratch.")
      loadCLDPReduced()
    } else {
      # Alternatively, if it is already saved to an RDS file:
      print("cell_line_drug_perms_reduced variable not found, but cell_line_drug_perms_reduced.rds file exists. Loading from file.")
      cell_line_drug_perms_reduced = readRDS(paste0(data_dir, "NCI-ALMANAC/cell_line_drug_perms_reduced.rds"))
      assign("cell_line_drug_perms_reduced", cell_line_drug_perms_reduced, envir = .GlobalEnv)
    }
  }
}




# --------------------------------------------------------



# Function for creating the cell-line-drug permutations table.
loadCLDPReduced = function() {
  print("Creating reduced cell-line-drug-permutation table.")
  
  # We will need almanac_dat for this, so make sure it's loaded.
  checkALMANACLoaded()
  
  # So we don't have to loop through every single drug permutation, in both almanac_dat and cell_line_drug_perms we will create a new column consisting of drug1 and drug2 paste0()-ed together. This way, we can compare this new column between the almanac_dat and cell_line_drug_perms. 
  almanac_dat$DRUGPERM = paste0(almanac_dat$NSC1, almanac_dat$NSC2)
  cell_line_drug_perms$DrugPerm = paste0(cell_line_drug_perms$Drug1, cell_line_drug_perms$Drug2)
  
  cell_line_drug_perms_reduced = cell_line_drug_perms[cell_line_drug_perms$DrugPerm %in% almanac_dat$DRUGPERM,,drop=F]
  
  saveRDS(cell_line_drug_perms_reduced, paste0(data_dir, "NCI-ALMANAC/cell_line_drug_perms_reduced.rds"))
  assign("cell_line_drug_perms_reduced", cell_line_drug_perms_reduced, envir = .GlobalEnv)
  # As of 01/18/2020 cell_line_drug_perms_unique.rds should have the standardized cell-line names and only the permutations actually represented in the ALMANAC data. 
  
}



# --------------------------------------------------------




# I forgot what this one does. T_T
keepCellLineDrugPerm = function(x) {
  cell_line = cell_line_drug_combos[x, "CellLine"]
  drug_perm = cell_line_drug_combos[x, "DrugPerm"]
  
  print(paste0("Adding entry ", x, " of ", nrow(cell_line_drug_combos), "."))
  return(ifelse(nrow(almanac_dat[almanac_dat$CELLNAME==cell_line & almanac_dat$DRUGPERM==drug_perm,]) > 0, 1, NA))
}




# --------------------------------------------------------




# This is the algorithm for calculating the "BestComboScore" as described by Xia et al (BMC Bioinformatics 2018.)

# Variables.
# cellline will be the ith cell line.
# drug1_NSC will be drug A.
# drug2_NSC will be drug B.

calcBestComboScore <- function(cellline, drug1_NSC, drug2_NSC, BestComboScoreOnly) {
  # Check if this combination of cell line, first drug, and second drug was even tested. 
  if(nrow(almanac_dat[which(almanac_dat$CELLNAME==cellline & almanac_dat$NSC1==drug1_NSC & almanac_dat$NSC2==drug2_NSC),]) > 0 | nrow(almanac_dat[which(almanac_dat$CELLNAME==cellline & almanac_dat$NSC1==drug2_NSC & almanac_dat$NSC2==drug1_NSC),]) > 0) {
    # The combination was tested; calculate the ComboScore and add it to the ComboScores data frame. 
    
    print(paste0("Calculating the BestComboScore for the drugs ", drug1_NSC, " and ", drug2_NSC, " for the cell line ", cellline, "."))
    
    # For the ith cell line exposed to drug pair A and B, get the percent growth fraction, which is just "PERCENTGROWTHCORRECTED" in the ALAMANAC data and which we will call "MinComboGrowth" (y_i^AB in the notation of Xia et al.) Furthermore, it is the LOWEST "PERCENTGROWTHCORRECTED" for ALL concentrations for a given drug pair and given cell line. (I.e., we will only consider the concentration that gives the lowest "PERCENTGROWTHCORRECTED.")
    # Get the lowest PERCENTGROWTHCORRECTED for the ith cell line. This will be called as part of a for() loop that loops through the cell lines. 
    MinComboGrowth <- ifelse(sum(!is.na(almanac_dat[J(cellline, drug1_NSC, drug2_NSC),PERCENTGROWTHCORRECTED])) > 0, min(almanac_dat[J(cellline, drug1_NSC, drug2_NSC),PERCENTGROWTHCORRECTED]), min(almanac_dat[J(cellline, drug2_NSC, drug1_NSC),PERCENTGROWTHCORRECTED]))
    
    # For the ith cell line exposed to drug pair A and B, get the lowest growth fraction for drug A and B separately. They will be used to calculate the "ExpectedGrowth." y_i^A will be the LOWEST "PERCENTGROWTHCORRECTED" for ALL concentrations for a given drug A and given cell line; similarly, y_i^B will be the LOWEST "PERCENTGROWTHCORRECTED" for ALL concentrations for a given drug B and given cell line. I.e., for each drug A and B, we will only consider the concentrations that give the lowest respective "PERCENTGROWTHCORRECTED"s. 
    # We have already calculated the minimum growth percentages for each drug individually for each cell line, so we will just fetch the appropriate minimum growth percentage from the matrix. 
    MinAGrowth <- MinGrowth_single_drug[which(MinGrowth_single_drug[,1]==drug1_NSC & MinGrowth_single_drug[,2]==cellline),3]
    MinBGrowth <- MinGrowth_single_drug[which(MinGrowth_single_drug[,1]==drug2_NSC & MinGrowth_single_drug[,2]==cellline),3]
    
    # Calculate the ExpectedGrowth per Xia et al's equation. 
    if(MinAGrowth <= 0 | MinBGrowth <= 0) {
      ExpectedGrowth <- min(MinAGrowth, MinBGrowth)
    } else {
      MinAGrowth <- min(MinAGrowth, 100) # This truncates the growth fraction at 1. I.e., it does not allow for growth in excess of 100%. The same with the next line.
      MinBGrowth <- min(MinBGrowth, 100)
      ExpectedGrowth <- MinAGrowth * MinBGrowth / 100 # Dividing by 10000 would get it into decimal form. We want it in percent form.
    }
    
    # Now the "BestComboScore" (C_i^AB in the notation of Xia et al) is simply (MinComboGrowth - ExpectedGrowth) * 100, where MinComboGrowth is the experimentally measured growth after administration of the combo, and ExpectedGrowth is the theoretical growth after administration of the combo. The larger the difference (and therefore the larger the score), the better the drug combo performed than expected, meaning that there's a greater level of synergy between the drugs used for the combo.
    BestComboScore <- ExpectedGrowth - MinComboGrowth # (MinComboGrowth - ExpectedGrowth) * 100 <- don't need to multiply by 100 since we're no longer using decimal format. Also, it should be ExpectedGrowth - MinComboGrowth, because more growth = worse drug. 
    if(!BestComboScoreOnly) return(list(CellLine=cellline, Drug1=drug1_NSC, Drug2=drug2_NSC, BestComboScore=BestComboScore)) else return(BestComboScore)
    
  } else {
    if(!BestComboScoreOnly) return(list(CellLine=cellline, Drug1=drug1_NSC, Drug2=drug2_NSC, BestComboScore=NA)) else return(NA)
    
  }
}




# --------------------------------------------------------




# Function for getting the PercentGrowth. 
getBestPercentGrowth = function(cellline, drug1_NSC, drug2_NSC) {
  # Check if this combination of cell line, first drug, and second drug was even tested. 
  if(nrow(almanac_dat[which(almanac_dat$CELLNAME==cellline & almanac_dat$NSC1==drug1_NSC & almanac_dat$NSC2==drug2_NSC),]) > 0 | nrow(almanac_dat[which(almanac_dat$CELLNAME==cellline & almanac_dat$NSC1==drug2_NSC & almanac_dat$NSC2==drug1_NSC),]) > 0) {
    # The combination was tested; calculate the PercentGrowth and add it to the PercentGrowth data frame.
    print(paste0("Calculating the PercentGrowth for the drugs ", drug1_NSC, " and ", drug2_NSC, " for the cell line ", cellline, "."))
    
    # For the ith cell line exposed to drug pair A and B, get the percent growth fraction, which is just "PERCENTGROWTHCORRECTED" in the ALAMANAC data and which we will call "MinComboGrowth" (y_i^AB in the notation of Xia et al.) Furthermore, it is the LOWEST "PERCENTGROWTHCORRECTED" for ALL concentrations for a given drug pair and given cell line. (I.e., we will only consider the concentration that gives the lowest "PERCENTGROWTHCORRECTED.")
    # Get the lowest PERCENTGROWTHCORRECTED for the ith cell line. 
    MinComboGrowth = ifelse(sum(!is.na(almanac_dat[J(cellline, drug1_NSC, drug2_NSC),PERCENTGROWTHCORRECTED])) > 0, min(almanac_dat[J(cellline, drug1_NSC, drug2_NSC),PERCENTGROWTHCORRECTED]), min(almanac_dat[J(cellline, drug2_NSC, drug1_NSC),PERCENTGROWTHCORRECTED]))
    
    return(list(CellLine=cellline, Drug1=drug1_NSC, Drug2=drug2_NSC, BestPercentGrowth=MinComboGrowth))
    
  } else {
    return(list(CellLine=cellline, Drug1=drug1_NSC, Drug2=drug2_NSC, BestPercentGrowth=NA))
    
  }
}




# --------------------------------------------------------




# Generic function for getting the best instance of the target variable for each cell line x drug combination. 
getBestTargetVariable = function(cellline, drug1, drug2) {
  # Check if this combination of cell line, first drug, and second drug was even tested. 
  if(nrow(original_table[which(original_table$CellLine==cellline & original_table$Drug1==drug1 & original_table$Drug2==drug2),]) > 0) { # | nrow(original_table[which(original_table$CellLine==cellline & original_table$Drug1==drug2 & original_table$Drug2==drug1),]) > 0
    # The combination was tested; calculate the PercentGrowth and add it to the PercentGrowth data frame.
    print(paste0("Calculating the PercentGrowth for the drugs ", drug1, " and ", drug2, " for the cell line ", cellline, "."))
    
    # For the ith cell line exposed to drug pair A and B, get the percent growth fraction, which is just "Target" in the original data and which we will call "MinComboGrowth" (y_i^AB in the notation of Xia et al.) Furthermore, it is the LOWEST "Target" for ALL concentrations for a given drug pair and given cell line. (I.e., we will only consider the concentration that gives the lowest "Target.")
    # Get the lowest Target for the ith cell line. 
    MinComboGrowth = ifelse(sum(!is.na(original_table[J(cellline, drug1, drug2),Target])) > 0, min(original_table[J(cellline, drug1, drug2),Target]), min(original_table[J(cellline, drug2, drug1),Target]))
    
    return(list(CellLine=cellline, Drug1=drug1, Drug2=drug2, BestTargetVariable=MinComboGrowth))
    
  } else {
    return(list(CellLine=cellline, Drug1=drug1, Drug2=drug2, BestTargetVariable=NA))
    
  }
}




# --------------------------------------------------------




# Function for calculating the (corrected) ExpectedGrowth.
calcExpectedGrowthCorrected = function(cell_line, drugA, drugB, concindex1, concindex2, i, ExpectedGrowthCorrectedOnly=F) {
  print(paste0("Calculating corrected ExpectedGrowth for row ", i, " of ", nrow(almanac_dat_sub), "."))
  
  if(!is.na(drugB)) {
    # Calculate the ExpectedGrowth.
    # Get the growth (PERCENTGROWTHCORRECTED) of drug 1 and drug 2 for cell line cell_line. 
    DrugAGrowth = fifelse(
      nrow(MedianSingleDrugGrowth_data[CellLine==cell_line & Drug==drugA & ConcentrationIndex==concindex1,]) > 0,
      MedianSingleDrugGrowth_data[CellLine==cell_line & Drug==drugA & ConcentrationIndex==concindex1,]$MedianDrugGrowth,
      Inf
      )   
    
    DrugBGrowth= fifelse(
      nrow(MedianSingleDrugGrowth_data[CellLine==cell_line & Drug==drugB & ConcentrationIndex==concindex2,]) > 0,
      MedianSingleDrugGrowth_data[CellLine==cell_line & Drug==drugB & ConcentrationIndex==concindex2,]$MedianDrugGrowth,
      Inf
      )
    
    # Calculate the ExpectedGrowth per Holbeck et al's equation. 
    if(is.finite(DrugAGrowth) & is.finite(DrugBGrowth)) {
      if(DrugAGrowth <= 0 | DrugBGrowth <= 0) {
        ExpectedGrowthCorrected = min(DrugAGrowth, DrugBGrowth)
      } else {
        DrugAGrowth = min(DrugAGrowth, 100) # This truncates the growth fraction at 1. I.e., it does not allow for growth in excess of 100%. The same with the next line.
        DrugBGrowth = min(DrugBGrowth, 100)
        ExpectedGrowthCorrected = DrugAGrowth * DrugBGrowth / 100
      }
    } else {
      ExpectedGrowthCorrected = NA
    } # End if/else DrugAGrowth and DrugBGrowth are both defined.
    
  } else {
    ExpectedGrowthCorrected = NA
  } # End if/else DrugB is defined. 
  
  return(list(
    CellLine=cell_line, 
    Drug1=drugA, 
    Drug2=drugB, 
    ExpectedGrowthCorrected=ExpectedGrowthCorrected
  ))
  
}
  




# --------------------------------------------------------




# Function for getting the best (highest) ComboScore (corrected.)
getBestComboScoreCorrected = function(cellline, drug1_NSC, drug2_NSC) {
  # Check if this combination of cell line, first drug, and second drug was even tested. 
  if(nrow(almanac_dat[which(almanac_dat$CELLNAME==cellline & almanac_dat$NSC1==drug1_NSC & almanac_dat$NSC2==drug2_NSC),]) > 0 | nrow(almanac_dat[which(almanac_dat$CELLNAME==cellline & almanac_dat$NSC1==drug2_NSC & almanac_dat$NSC2==drug1_NSC),]) > 0) {
    # The combination was tested; calculate the BestComboScoreCorrected and add it to the BestComboScoreCorrected data frame.
    print(paste0("Calculating the best corrected ComboScore for the drugs ", drug1_NSC, " and ", drug2_NSC, " for the cell line ", cellline, "."))
    
    # Get the highest SCORECORRECTED for the ith cell line. 
    BestComboScoreCorrected = ifelse(sum(!is.na(almanac_dat[J(cellline, drug1_NSC, drug2_NSC),SCORECORRECTED])) > 0, max(almanac_dat[J(cellline, drug1_NSC, drug2_NSC),SCORECORRECTED]), max(almanac_dat[J(cellline, drug2_NSC, drug1_NSC),SCORECORRECTED]))
    
    return(list(CellLine=cellline, Drug1=drug1_NSC, Drug2=drug2_NSC, BestComboScoreCorrected=BestComboScoreCorrected))
    
  } else {
    return(list(CellLine=cellline, Drug1=drug1_NSC, Drug2=drug2_NSC, BestComboScoreCorrected=NA))
    
  }
}





# --------------------------------------------------------




