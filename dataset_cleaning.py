import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

input_file = r'C:\Users\karan\PycharmProjects\QSAR_toxicity_prediction\Acute_toxicity.csv'

df = pd.read_csv(input_file)
print(df.head())

df = df[["Canonical_Smiles", "LD50 (rat oral, -log10 mol_kg-bw)"]].rename(columns={"Canonical_Smiles": "SMILES", "LD50 (rat oral, -log10 mol_kg-bw)": "LD50"})
print(df.head())
print("shape of the dataset ",df.shape)
print()
#droping the missing values
df = df.dropna(subset=['SMILES',"LD50"])
df = df.drop_duplicates('SMILES')
df = df.reset_index(drop=True)

salt_remover = SaltRemover.SaltRemover()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
print("Before cleaning")
print(df.head())

cleaned_smiles = []
cleaned_LD50 = []

for i, j in df.iterrows():
    smiles = j["SMILES"]
    LD50 = j["LD50"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue

    #sanitizing the smiles
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        continue

    #remove a salt
    mol = salt_remover.StripMol(mol, dontRemoveEverything=True)

    ##tautomer standardization
    mol = tautomer_enumerator.Canonicalize(mol)

    #canonical SMILES
    mol_smiles = Chem.MolToSmiles(mol, canonical=True)

    cleaned_smiles.append(mol_smiles)
    cleaned_LD50.append(LD50)

new_df = pd.DataFrame({"SMILES": cleaned_smiles,"LD50":cleaned_LD50})
print("after cleaning : ")
print(new_df.head(10))

new_df= new_df.drop_duplicates(subset=['SMILES']).reset_index(drop=True)

output_file = r'C:\Users\karan\PycharmProjects\QSAR_toxicity_prediction\cleaned_toxicity.csv'

new_df.to_csv(output_file,index=False)

print(df.shape)
print("the cleaned file is saved to a ",output_file)




