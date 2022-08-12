createListOfDrugPairs = function(drugs, type) {
  # Creates a p x 2 matrix of all possible drug pairs in the vector argument "drugs," where p = the number of pairwise combinations that can be made from length(drugs) drugs. 
  # The parameter "type" determines whether we generate a combination or permutation of the drugs.
  
  drugs = unique(drugs) # In case we have duplicate drugs (looking at you, Vemurafenib T_T)
  
  if(type=="combination") {
    return(t(combn(drugs, 2)))
  } else if(type=="permutation") {
    return(gtools::permutations(length(drugs), 2, drugs))
  } else {
    print(paste0("Warning: type ", type, " is not an accepted argument. Please specify either 'combination' or 'permutation'."))
    return(NULL)
  }
  
}