library(RMariaDB)
library(XML)
library(xml2)
library(pryr)
library(plyr)
library(dplyr)
cloud = FALSE
if(cloud) {
  # Set directories. 
  # https://stackoverflow.com/a/4463291
  # File folders for input/output data
  data_dir = "/home/lon/Pharmacogenomics_drug_combos/Data/"
  script_dir = "/home/lon/Pharmacogenomics_drug_combos/Scripts/Network_and_gene_expression_predictor/"
  results_dir = "/home/lon/Pharmacogenomics_drug_combos/Results/"
  programs_dir = "/home/lon/Pharmacogenomics_drug_combos/Common_programs/"
  #data_dir = "/work/05034/lwfong/lonestar/Pharmacogenomics_drug_combos/Data/"
  #script_dir = "/work/05034/lwfong/lonestar/Pharmacogenomics_drug_combos/Scripts/Network_and_gene_expression_predictor/"
  #results_dir = "/work/05034/lwfong/lonestar/Pharmacogenomics_drug_combos/Results/NetworkScore/"
  #programs_dir = "/work/05034/lwfong/lonestar/Common_programs/"
} else {
  # Local.
  # Set directories. 
  # https://stackoverflow.com/a/4463291
  switch(Sys.info()[['sysname']],
         Windows = {tilde_expansion = "C:/Users/lwfong/"},
         Linux   = {tilde_expansion = "~/"},
         Darwin  = {tilde_expansion = "~/"})
  
  # File folders for input/output data
  data_dir = paste0(tilde_expansion, "Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Data/")
  script_dir = paste0(tilde_expansion, "Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Scripts/Network_and_gene_expression_predictor/")
  results_dir = paste0(tilde_expansion, "Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Results/")
}

# Function to use with xmlEventParse.
# NOTE! If Xpath is not working, see https://mirekdlugosz.com/blog/2015/when-xml2-returns-no-matches-for-correct-xpath-expression/. Make sure the namespace stuff is removed from the XML document's root node. 
ELTdrugTargets = function(drug) {
  gc()
  #print(paste0("Memory used after garbage collection: ", mem_used()))
  #print(ls())
  
  # tryCatch block 1: getting the drug name, ID, and targets.
  possible_error_1 = tryCatch({
    # Name the current drug for easy reference
    current_drug = read_xml(toString.XMLNode(drug)); # current_drug = xml2::xml_children(test_file_xml)[1] for test
    
    # Get the name of the drug. 
    # NOTE! We might have to standardize these names in the final drug_targets_table, since the DrugBank name might differ from the Gao name. ... 
    drug_name = xml_text(xml_find_first(current_drug, './name'))
    drug_name = base::gsub("'", "''", drug_name) # https://stackoverflow.com/a/40258033
    
  }, error = function(cond) {
    print(paste0("Error! Reading in the drug node encountered the following problem: "))
    print(cond)
    cond
  })
  # End tryCatch block 1. 
  
  if(!inherits(possible_error_1, "error") & exists("drug_name")) {
    # tryCatch block 2: getting the drug ID and targets.
    possible_error_2 = tryCatch({
      current_drug_db_id = xml_text(xml_find_first(current_drug, './drugbank-id')) # xml_find_first() finds the first node matching the Xpath query (and returns an XML node); xml_text() converts the text in that node into a character vector. 
      # current_drug_db_id = xml_text(xml_find_first(current_drug, './drugbank-id')) for test. 
      
      print(paste0("Getting DrugBank targets for drug ", drug_name, "."))
      # Get the targets.
      drug_targets = xml_text(xml_find_all(current_drug, './targets/target/polypeptide/gene-name'))
    }, error = function(cond) {
      drugs_with_error <<- c(drugs_with_error, drug_name)
      print(paste0("Error! Getting the ID for drug ", drug_name, " encountered the following error: "))
      print(cond)
      cond
    })
    # End tryCatch block 2.
    
    if(!inherits(possible_error_2, "error") & exists("drug_targets")) {
      # Each target may have > 1 action per target, so run a for() loop to handle each action individually. 
      for(target_name in drug_targets) {
        # tryCatch block 3: getting the organism and pharmacological actions. 
        possible_error_3 = tryCatch({
          organism = xml_text(xml_find_all(current_drug, paste0("./targets/target/polypeptide[gene-name='", target_name, "']/../organism"))) %>% paste0(collapse = "|")
          drug_actions = xml_text(xml_find_all(current_drug, paste0("./targets/target/polypeptide[gene-name='", target_name, "']/../actions/action")))
        }, error = function(cond) {
          drugs_with_error <<- c(drugs_with_error, drug_name)
          print(paste0("Error! Getting the organism and pharmacological actions for target", target_name, " for drug ", drug_name, " encountered the following error: "))
          print(cond)
          cond
        })
        # End tryCatch block 3.
        
        if(!inherits(possible_error_3, "error") & exists("drug_actions")) {
          if(length(drug_actions) > 0) {
            for(drug_action in drug_actions) {
              # Create the SQL query.
              query = paste0(
                "INSERT INTO DrugBank (
                DrugName,
                DrugBankID,
                TargetName,
                Organism,
                PharmacologicalAction)
                VALUES('",drug_name,"',
                '",current_drug_db_id,"',
                '",target_name,"',
                '",organism,"',
                '",drug_action,"')"
              )
              
              # Execute the query on the storiesDb that we connected to above.
              possible_error = tryCatch({
                rsInsert = dbSendQuery(DrugTargetsDB, query)
              }, error = function(cond) {
                drugs_with_error <<- c(drugs_with_error, drug_name)
                print(paste0("Error! The SQL query for action ", drug_action, " for target ", target_name, " for drug ", drug_name,  " encountered the following problem: "))
                print(cond)
                cond
              })
              
              if(!inherits(possible_error, "error") & exists("rsInsert")) {
                # Clear the result.
                dbClearResult(rsInsert)
              }
            }
          } else {
            # Create the SQL query.
            drug_action = "Unknown"
            query = paste0(
              "INSERT INTO DrugBank (
                DrugName,
                DrugBankID,
                TargetName,
                Organism,
                PharmacologicalAction)
                VALUES('",drug_name,"',
                '",current_drug_db_id,"',
                '",target_name,"',
                '",organism,"',
                '",drug_action,"')"
            )
            
            # Execute the query on the storiesDb that we connected to above.
            possible_error = tryCatch({
              rsInsert = dbSendQuery(DrugTargetsDB, query)
            }, error = function(cond) {
              drugs_with_error <<- c(drugs_with_error, drug_name)
              print(paste0("Error! The SQL query for target ", target_name, " for drug ", drug_name,  " encountered the following problem: "))
              print(cond)
              cond
            })
            
            if(!inherits(possible_error, "error") & exists("rsInsert")) {
              # Clear the result.
              dbClearResult(rsInsert)
            } # End if everything was completed successfully. 
            
          } # End else the target has no pharmacological action associated with it. 
          
        } # End if the drug actions for target target_name were obtained successfully. 
        
      } # End for target_name in drug_targets loop. 
      
    } # End if the drug ID was obtained successfully. 
    
  } # End if the drug name was obtained successfully. 
  
  # remove the current node from memory when finished with it
  #print(paste0("Memory used before garbage collection: ", mem_used()))
  #print(ls())
  #rm(drug)
  rm(list=setdiff(ls(), c("db_xml", "test_xml", "user_password", "DrugTargetsDB", "ELTdrugTargets", "drugs_with_error")))
  gc()
}

# Load the DrugBank XML file.
db_xml = paste0(data_dir, "DrugBank/DrugBank_all_full_database_5_1_7.xml")
test_xml = paste0(data_dir, "DrugBank/DrugBank_sample_entry.xml")

# https://programminghistorian.org/en/lessons/getting-started-with-mysql-using-r#create-an-r-script-that-connects-to-the-database
# Make the connection to the MySQL database. 
# The connection method below uses a password stored in a settings file.
user_password = "Kibougaaru2020"
DrugTargetsDB = dbConnect(RMariaDB::MariaDB(), 
                          user='root', 
                          password=user_password, 
                          dbname='DrugTargets', 
                          host='localhost')
dbListTables(DrugTargetsDB)

# Test.
#xmlEventParse(file = test_xml, handlers = NULL, trim = FALSE, branches = list(drug = ELTdrugTargets))
# Run the ELTdrugTargets() function on the XML document. 
drugs_with_error = c()
xmlEventParse(file = test_xml, handlers = NULL, trim = FALSE, branches = list(drug = ELTdrugTargets), addFinalizer = F)

# Close the connection to the database.
dbDisconnect(DrugTargetsDB)
