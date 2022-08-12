generateRandomDrugCombos <- function(num_control_drug_combos, num_drugs_per_combo, drug_list, exp_drug_combo) {
  # num_control_drug_combos = the number of combinations we want to generate.
  # num_drugs_per_combo = the number of drugs each combination will have.
  # drug_list = the list of drugs from which to randomly sample drugs.
  # exp_drug_combo = [OPTIONAL] a combination of drugs which should NOT be duplicated. Its length must equal num_drugs_per_combo.
  
  if(length(exp_drug_combo) != num_drugs_per_combo) stop("Please make sure that the num_drugs_per_combo argument matches the number of drugs in the exp_drug_combo argument.")
  
  # First the control drugs. 
  # Create a table to hold the n + 1 drug combinations where n = num_control_drug_combos and the 1 refers to the combination we're testing. 
  exp_and_control_drug_combos <- matrix(, ncol = num_drugs_per_combo, nrow = 0)
  
  # Use a while() loop to populate the table (not a for() loop, since we don't know how many iterations it will take to populate.)
  while(nrow(exp_and_control_drug_combos) < num_control_drug_combos) {
    # Generate a combo j of n random drugs, where n = num_drugs_per_combo.
    combo_j <- c()
    for(i in 1:num_drugs_per_combo) {
      # Generate random drug i. 
      rand_drug_i <- sample(setdiff(unique(drug_list), combo_j), 1)
      
      # Add it to the combo_j vector.
      combo_j <- c(combo_j, rand_drug_i)
    }
    
    # Check if combo j already exists (i.e. is the experimental drug combination), has an NA, or is not present in the original ALMANAC data.
    # Check if combo j is the experimental drug combination. 
    # It might be in a different order; generate all possible permutations of that n-plet of drugs. 
    all_tuplets <- createListOfDrugTuplets(combo_j, "permutation")
    # Check if exp_drug_combo is in all_tuplets.
    # https://stackoverflow.com/a/4294420/8497173
    combo_is_exp_combo <- F
    if(sum(apply(all_tuplets, 2, function(x, want) isTRUE(all.equal(x, want)), exp_drug_combo)) > 0) combo_is_exp_combo <- T
    
    # Check if combo j is already in the table. 
    # https://stackoverflow.com/a/4294420/8497173
    combo_exists <- F
    if(sum(apply(exp_and_control_drug_combos, 2, function(x, want) isTRUE(all.equal(x, want)), combo_j)) > 0) combo_exists <- T
    
    # Check if combo j has an NA.
    combo_has_NA <- F
    if(sum(is.na(combo_j)) > 0) combo_has_NA <- T
    
    if(sum(combo_is_exp_combo, combo_exists, combo_has_NA) > 0) {
      # If any of the above apply, do nothing.
      print(paste0("The drug combination just generated either already exists or has a missing value. Generating a new random combination."))
    } else {
      # Else, add it to the control_drug_combos table. 
      exp_and_control_drug_combos <- rbind(exp_and_control_drug_combos, combo_j)
    } 
    
  }

  return(exp_and_control_drug_combos)
}