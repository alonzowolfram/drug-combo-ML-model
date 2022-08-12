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
