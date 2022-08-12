#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:20:53 2020

@author: alonzowolf
"""

# https://stackoverflow.com/questions/36804794/iterparse-large-xml-using-python

import xml.etree.cElementTree as et # XML parsing. 
import mysql.connector # Connecting to MySQL database. 
import re # String functions analogous to R's gsub.  

# File paths.
data_dir = "/Users/alonzowolf/Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Data/"
script_dir = "/Users/alonzowolf/Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Scripts/Network_and_gene_expression_predictor/"
results_dir = "/Users/alonzowolf/Dropbox/Work/Thesis_projects/Pharmacogenomics_drug_combos/Results/"

# Load the DrugBank XML file(s).
db_xml = data_dir + "DrugBank/DrugBank_all_full_database_5_1_7.xml"
test_xml = data_dir + "DrugBank/DrugBank_sample_entry.xml"
test_xml_2 = data_dir + "DrugBank/DrugBank_sample_entry_2.xml"
test_targets_xml = data_dir + "DrugBank/test_targets.xml"

# Connect to the MySQL database.
drugbank_db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Kibougaaru2020",
  database="DrugTargets"
)

db_uri_qname = '{http://www.drugbank.ca}'
db_uri = 'http://www.drugbank.ca'
ns = {'db': db_uri}
with open(db_xml) as xml_file:
    for event, drug in et.iterparse(xml_file):
        if drug.tag == et.QName(db_uri, 'drug') and 'type' in drug.attrib:
            # Get all the drugs, i.e. all the elements named "drug" that are IMMEDIATE CHILDREN of the root element ("drugbank").
                # Get the name of the current drug. 
                drug_name = drug.find("db:name", ns).text
                # Replace single quotes with double quotes.
                drug_name = re.sub("'", "''", drug_name)
                # Get the DrugBank ID of the current drug. 
                drug_db_id = drug.find("db:drugbank-id", ns).text
                
                print("Obtaining targets for drug " + drug_name + ".")
                
                # Loop over the targets for the drug.
                for target in drug.findall("db:targets/" + db_uri_qname +  "target", ns):
                    try:
                        target_name = target.find("db:polypeptide/" + db_uri_qname +  "gene-name", ns).text
                        print("Target " + target_name + " for drug " + drug_name + ".")
                    except:
                        print("Skipping a non-protein/non-gene target for drug " + drug_name + ".")
                        continue
                    
                    if target_name is None:
                        print("Skipping a target for drug " + drug_name + " without a name.")
                        continue
                    
                    # Get the organism(s). Sometimes there will be >1 organism, so join into a single string.
                    if target.findall("db:organism", ns) is not None:
                        sep = "|"
                        num_organisms = len(target.findall("db:organism", ns))
                        organisms = []
                        for i in range(num_organisms):
                            organism_i = target.findall("db:organism", ns)[i].text
                            if organism_i is not None:
                                organisms.append(organism_i)
                    
                        organism = sep.join(organisms)
                        print("Target " + target_name + " has organism " + organism + ".")
                        
                        # Loop over the pharmacological actions for the target. 
                        actions = target.findall("db:actions/" + db_uri_qname +  "action", ns)
                        if len(actions) < 1: # There are no actions for this target. 
                            action_name = "unknown"
                            print("No pharmacological action listed for the target " + target_name + " for drug " + drug_name + ". Setting the action to 'unknown.'")
                            try:
                                # Insert the row.
                                drugbank_db_cursor = drugbank_db.cursor()
                                
                                query = "INSERT INTO DrugBank (DrugName, DrugBankID, TargetName, Organism, PharmacologicalAction) VALUES (%s, %s, %s, %s, %s)"
                                values = (drug_name, drug_db_id, target_name, organism, action_name)
                                drugbank_db_cursor.execute(query, values)
                                drugbank_db.commit()
                                print("Drug action " + action_name + " for target " + target_name + " for drug " + drug_name + " inserted into the database.")
                            except:
                                print("Error inserting the row into the database.")
                                
                        else: # There are actions for this target. 
                            for action in target.findall("db:actions/" + db_uri_qname +  "action", ns):
                                try:
                                    # Get the name of the action.
                                    action_name = action.text
                                except:
                                    action_name = "unknown"
                                    print("There was an error getting the action for target " + target_name + " for drug " + drug_name + ". Setting the action to 'unknown.'")
                                
                                try:
                                    # Insert the row.
                                    drugbank_db_cursor = drugbank_db.cursor()
                                
                                    query = "INSERT INTO DrugBank (DrugName, DrugBankID, TargetName, Organism, PharmacologicalAction) VALUES (%s, %s, %s, %s, %s)"
                                    values = (drug_name, drug_db_id, target_name, organism, action_name)
                                    drugbank_db_cursor.execute(query, values)
                                    drugbank_db.commit()
                                    print("Drug action " + action_name + " for target " + target_name + " for drug " + drug_name + " inserted into the database.")
                                except:
                                    print("Error inserting the row into the database.")
                        
    # Note: only clears <drugbank> elements and their children.
    # There is a memory leak of any elements not children of <drugbank>
    drug.clear()   
