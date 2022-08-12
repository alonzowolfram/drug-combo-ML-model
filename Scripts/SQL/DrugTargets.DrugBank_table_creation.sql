CREATE TABLE DrugTargets.DrugBank (
InteractionID INT NOT NULL AUTO_INCREMENT,
DrugName VARCHAR(255) NULL,
DrugBankID VARCHAR(255) NULL,
TargetName VARCHAR(255) NULL,
Organism VARCHAR(255) NULL,
PharmacologicalAction VARCHAR(255) NULL,
PRIMARY KEY (InteractionID));