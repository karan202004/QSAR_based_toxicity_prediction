import pandas as pd
import numpy as np
import os
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from mordred import Calculator, descriptors
if not hasattr(np, 'product'):
    np.product = np.prod

input_file = r'C:\Users\karan\PycharmProjects\QSAR_toxicity_prediction'
model_path = os.path.join(input_file, "QSAR_Toxicity_model.pkl")
feature_path = os.path.join(input_file, "QSAR_feature_list.pkl")

#loading the model
model = joblib.load(model_path)
selected_features = joblib.load(feature_path)

#mordred descriptors calculation
calc = Calculator(descriptors, ignore_3D=True)

salt_remover = SaltRemover.SaltRemover()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def predict_toxicity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid smiles"

    #sanitization
    try:
        Chem.SanitizeMol(mol)
    except:
        return "sanitization failed"

    #salt removal
    mol = salt_remover.StripMol(mol, dontRemoveEverything=True)

    #tautomer standardization
    mol = tautomer_enumerator.Canonicalize(mol)

    #molecular weight calculation
    mw = Descriptors.MolWt(mol)

    #calculate descriptors
    cal_descriptors = calc.pandas([mol], quiet=True)
    desc_numeric = cal_descriptors.apply(pd.to_numeric, errors='coerce').fillna(0)

    #select only filter features
    try:
        calculation = desc_numeric[selected_features]
    except KeyError as e:
        return f"Missing required descriptor: {e}"

    #prediction
    predicted_pLD50 = model.predict(calculation)[0]

    mg_kg = (10 ** (3 - predicted_pLD50)) * mw

    return mg_kg


if __name__ == "__main__":
    test_smiles = input("Enter a SMILES string to predict Toxicity: ")
    result = predict_toxicity(test_smiles)

    if isinstance(result, str):
        print(f"Prediction failed: {result}")
    else:
        print(f"Predicted LD50: {result:.2f} mg/kg")

        if result <= 5:
            print("Category 1 (Extremely Fatal)")
        elif result <= 50:
            print("Category 2 (Fatal)")
        elif result <= 300:
            print("Category 3 (Toxic)")
        elif result <= 2000:
            print("Category 4 (Harmful)")
        elif result <= 5000:
            print("Category 5 (May be harmful)")
        else:
            print("Not Classified (Low Toxicity)")