createListOfDrugTuplets = function(drugs, type, m) {
  # Creates a p x m matrix of all possible drug tuplets in the vector argument "drugs," where p = the number of tuplet-wise combinations that can be made from length(drugs) drugs and m = the number of drugs per tuplet, to be specified by the user. If the user does not provide a value for m, it defaults to the length of the vector of drugs from which to generate the tuplets.
  # The parameter "type" determines whether we generate a combination or permutation of the drugs.
  
  drugs = unique(drugs)
  
  if(missing(m)) {
    m = length(drugs)
  } 
  
  if(type=="combination") {
    return(t(combn(drugs, m)))
  } else if(type=="permutation") {
    return(gtools::permutations(length(drugs), m, drugs))
  } else {
    print(paste0("Warning: type ", type, " is not an accepted argument. Please specify either 'combination' or 'permutation'."))
    return(NULL)
  }
  
}