# We will tune the best-performing model.
# In this case, RandomForests.
# Note that we use the "Spotchecking" input data set to build the model. Even though filenames.R generates a filename for a "Hyperparameter_tuning" input data set, it does not exist. 

library(reticulate)
library(plyr)
library(dplyr)
library(regexPipes)

reticulate::conda_list()
use_condaenv("r-reticulate", required = TRUE)
py_config()
system("which python")

# Set the variables. 
# Remember that Python numbering starts at 0, unlike R, which starts at 1, so we will need to subtract 1 from whatever index we would use in R to get the equivalent Python index.
# Also, apparently numbers in R are sent over as type float by default, even the ones that look like integers. So we will have to explicitly convert to integers before calling functions in Python.
meta_vars = list()
meta_vars[["target_column"]] = list()
meta_vars[["pred_start_column"]] = list()
meta_vars[["grouping_column"]] = list()
meta_vars[["num_folds"]] = list()
data_sets = c("ALMANAC", "GDSC1", "GDSC2")
target_columns = c(as.integer(3-1), as.integer(4-1), as.integer(4-1))
pred_start_columns = c(as.integer(4-1), as.integer(5-1), as.integer(5-1))
grouping_columns = c(as.integer(1-1), as.integer(3-1), as.integer(3-1))
num_folds = rep(as.integer(5), length(data_sets))
for(i in 1:length(data_sets)) {
  data_set = data_sets[i]
  for(j in 1:length(names(meta_vars))) {
    meta_var = names(meta_vars)[j]
    
    meta_var_name = ifelse(meta_var=="num_folds", meta_var, paste0(meta_var, "s"))
    meta_vars[[meta_var]][[data_set]] = get(meta_var_name)[i] # get(): https://stackoverflow.com/a/10430015
  }
}

for(ds in c("ALMANAC")) {
  print(paste0("Working on data set ", ds, "."))
  
  timestamp = Sys.time() %>% str_split(" ") %>% unlist %>% regexPipes::gsub(":", "-")
  stdout_file = paste0(ds, "_hyperparameter-tuning_stdout_", timestamp[1], "_", timestamp[2], ".txt")
  if(!file.exists(stdout_file)) file.create(stdout_file)
  
  if(class(input_filename_list[[ds]]) != "list") {
    # It's one of the GDSC ones.
    # Check if the input and output file exist.
    # If the input file does not, or if both the output files do, skip.
    input_filename = ""
    output_raw_filename = ""
    output_raw_by_group_filename = ""
    if(!file.exists(input_filename) | (file.exists(output_raw_filename) & file.exists(output_raw_by_group_filename))) {
      print(paste0("Either the input file ", input_filename, " does not exist or this data set has been hyperparameter-tuned already. Skipping to the next one."))
      next
    }
    
    print(paste0("Now performing hyperparameter tuning for the data set file ", input_filename, "."))
    # Make the command.
    command = paste0(
      "python ",
      #modeling_functions_dir,
      "hyperparameter_tuning.py -d ",
      input_filename, 
      " -t ", meta_vars$target_column[[ds]],
      " -s ", meta_vars$pred_start_column[[ds]], 
      " -g ", meta_vars$grouping_column[[ds]],
      " -r ", output_raw_filename, 
      " -b ", output_raw_by_group_filename,
      " -n ", meta_vars$num_folds[[ds]],
      " >> ", stdout_file, " 2>&1"
    )
    
    # Run.
    system(command)
    
  } else {
    for(dt in c("S1")) { # "combined", "S1", "S2", "singles"
      # Check if the input and output file exist.
      # If the input file does not, or if both the output files do, skip.
      input_filename = "ALMANAC_PROGENy_ml_S1_processed_subset.csv"
      output_raw_filename = "ALMANAC_PROGENy_ml_S1_hyperparameter-tuning_raw_results_by_group.csv"
      output_raw_by_group_filename = "ALMANAC_PROGENy_ml_S1_hyperparameter-tuning_raw_results.csv"
      if(!file.exists(input_filename) | (file.exists(output_raw_filename) & file.exists(output_raw_by_group_filename))) {
        print(paste0("Either the input file ", input_filename, " does not exist or this data set has been hyperparameter-tuned already. Skipping to the next one."))
        next
      }
      
      print(paste0("Now performing hyperparameter tuning for the data set file ", input_filename, "."))
      
      # Make the command.
      command = paste0(
        "python ",
        #modeling_functions_dir,
        "hyperparameter_tuning.py -d ",
        input_filename, 
        " -t ", meta_vars$target_column[[ds]],
        " -s ", meta_vars$pred_start_column[[ds]], 
        " -g ", meta_vars$grouping_column[[ds]],
        " -r ", output_raw_filename, 
        " -b ", output_raw_by_group_filename,
        " -n ", meta_vars$num_folds[[ds]],
        " >> ", stdout_file, " 2>&1"
      )
      
      # Run.
      system(command)
      
    } # End for() loop looping over the different descriptor types. 
  } # End else it's the ALMANAC data set. 
}