handleError = function(sample, i, error_type, export_file) {
  print(paste0("Error for sample", sample, ": ", error_type, "Skipping row ", i, "."))
  
  # Write the error to the appropriate file. 
  sample_error_file = paste0(results_dir, "NetworkScore/Logs/", sample, "_log.txt")
  fileConn = file(sample_error_file)
  writeLines(paste0("Error in calcNetworkScores() loading the network for sample ", sample, ". Skipping row ", i, "."), fileConn)
  close(fileConn)
  
  score_table = data.frame(
    DrugNetworkOverlapWeighted = NA, 
    DrugNetworkOverlapUnweighted = NA,
    WINTHERscore = NA,
    CentralityRaw = NA,
    CentralityWeighted = NA,
    CentralityUnweighted = NA,
    #BridgingCentralityWeighted = NA,
    #BridgingCentralityUnweighted = NA,
    Sample = NA, 
    Dataset = NA, 
    Drugs = NA)
  
  # Save to CSV file if indicated.
  if(export_file) write.table(score_table, file_name, row.names = F, col.names = F, sep = ',')
  
  return(score_table)
}