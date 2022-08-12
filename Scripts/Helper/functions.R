# Load and process SPEED2 database. 
loadSPEED2 = function(filepath) {
  # Load SPEED2 database.
  speed2 = read.csv(filepath)
  # Make the regulation numeric. 
  a = c("UP", "DOWN")
  b = c(1, -1)
  names(b) = a
  speed2$RegulationNumeric = b[match(speed2$Regulation, a)]
  # Replace HUGO gene IDs with ENTREZ.
  hs = org.Hs.eg.db
  keys = speed2$Gene %>% as.character()
  x = AnnotationDbi::select(hs, 
                            keys = keys,
                            columns = c("SYMBOL", "ENTREZID"),
                            keytype = "SYMBOL") %>% .[complete.cases(.),] %>% distinct()
  # https://stackoverflow.com/a/3905442
  a = x$SYMBOL
  b = x$ENTREZID
  speed2$GeneEntrez = b[match(speed2$Gene, a)]
  
  return(speed2)
}

# Get a vector containing the disease-module status of the genes for a  given sample. 
get_disease_module_status = function(sample) {
  # This function returns a vector containing the disease-module status of the genes for a given sample. 
  disease_networks[[sample]]$Map$DiseaseModuleGene %>% unlist %>% as.factor %>% as.numeric %>%  - 1
}

# Calculate the overlap using the equation from Maron et al (Nat Com 2021.) 
# overlap(A, B) = size(intersection(A, B)) / min(size(A), size(B))
maron_overlap = function(A, B) {
  # A = a character vector containing the names of the disease module genes in the first of the two samples to be compared.
  # B = a character vector containing the names of the disease module genes in the second of the two samples to be compared. 
  
  length(intersect(A, B)) / min(length(A), length(B))
}
overlap_matrix = function(A, B_list) {
  # A = a character vector containing the names of the disease module genes in the first of the two samples to be compared via maron_overlap().
  # B_list = a list of character vectors, each vector containing the names of the disease module genes in the second of the two samples to be compared via maron_overlap().
  
  lapply(B_list, FUN =  maron_overlap, A = A) %>% unlist
  
}
get_disease_genes = function(sample) {
  # This function returns a vector containing the disease-module genes for a given sample. 
  disease_networks[[sample]]$DiseaseModuleGene %>% 
    .[.$DiseaseModuleGene=="Y","GeneSymbol"] %>% #dplyr::filter(DiseaseModuleGene=="Y") %>% 
    #dplyr::select(GeneSymbol) %>% unlist
    as.character
}

create_disease_network_list = function(cell_line_names) {
  disease_networks = list()
  for(cell_line_name in cell_line_names) {
    disease_networks[[cell_line_name]] = paste0(aim_2_data_dir, "NCI-ALMANAC/Disease_networks/", disease_network_folder, cell_line_name,  "_disease_network", all_genes, ".rds") %>% readRDS
  }

  return(disease_networks)
}

# Calculation of average clustering accuracy for randomized labels.
random_assignment_aca = function(i, y) {
  # y = ground truth vector (i.e. correct label assignments.)
  set.seed(i)
  
  # Scramble the labels.
  y_prime_rand = sample(y)
  
  # Calculate average clustering accuracy.
  avg_cluster_accuracy(y, y_prime_rand)
}

create_disease_network = function(sample, input_dir, output_dir) {
  if(file.exists(paste0(input_dir, sample, "_disease_network", all_genes, ".rds")) | file.exists(paste0(output_dir, sample, "_disease_network", all_genes, ".rds"))) {
    return(paste0("Network ", sample, "_disease_network", all_genes, ".rds already exists. Moving on to the next sample."))
  } else {
    # Set up a list to hold the network, the mapping, and the hits. 
    network_list = list()
    
    # Differential gene expression data.
    de_genes = as.data.frame(deg_list[[sample]][["DEGs"]])
    de_genes$GeneSymbol = rownames(de_genes)
    colnames(de_genes)[1] = "RankChange"
    
    # Map gene names from the table of differentially expressed genes to STRING identifiers. 
    #system.time({
    mapped = string_db$map(de_genes, "GeneSymbol", removeUnmappedRows=TRUE)
    #}) # Time: user system elapsed 7.412   0.248   7.978 
    hits = mapped$STRING_id # These are the hits. To select the top n hits, simply subset using [1:n]. Because we start with ~150 genes, which should be enough to make a decent PPI network, we will just use all of these genes without subsetting. 
    # Plot the network, showing the number of proteins (genes), the number of interactions, and the p-value (probability that we would get, by chance, as many interactions as we did in this network.)
    # string_db$plot_network(hits)
    
    # Create an iGraph network object. This is the disease network, which we can now run iGraph functions on.
    #system.time({
    network = string_db$get_subnetwork(hits) # user system elapsed 0.017   0.066   0.192
    #})
    # Save network to list.
    network_list[["Network"]] = network 
    network_list[["Map"]] = mapped
    network_list[["Hits"]] = hits
    
    # New addition 11/01/2020; need the intensity for the WINTHER score.
    # Add intensity.
    intensity = as.data.frame(deg_list[[sample]][["Intensity"]])
    intensity = intensity[rownames(intensity) %in% mapped$GeneSymbol,,drop=F]
    colnames(intensity) = "Intensity"
    network_list[["Intensity"]] = intensity
    
    # Save the list to RDS.
    saveRDS(network_list, paste0(output_dir, sample, "_disease_network", all_genes, ".rds"))
    
    # Clean up.
    rm(mapped, hits, network_list)
    gc()
    
    return(paste0("Successfully created network ", sample, "_disease_network", all_genes, ".rds"))
  }
}

# Replace Inf values with a given replacement.
replaceNonnumeric = function(v, replacement, types = c("All")) {
  if("All" %in% types) v[!is.finite(v)] = replacement
  if("Inf" %in% types) v[is.infinite(v)] = replacement
  if("NaN" %in% types) v[is.nan(v)] = replacement
  if("NA" %in% types) v[is.na(v)] = replacement
  
  return(v)
}