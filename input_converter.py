import pandas as pd
import pickle
# Load training column order
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
def convert_user_input(
        rooms,
        location,
        crime,
        property_age,
        area_quality
    ):
    """
    Convert user-friendly inputs to model features.
    All string inputs are normalized to lowercase for consistency.
    """
    # Normalize inputs to lowercase
    location = location.lower() if isinstance(location, str) else location
    crime = crime.lower() if isinstance(crime, str) else crime
    property_age = property_age.lower() if isinstance(property_age, str) else property_age
    area_quality = area_quality.lower() if isinstance(area_quality, str) else area_quality
    # ---------- Base Defaults ----------
    data = {
        'CRIM': 0.1,
        'ZN': 10,
        'INDUS': 8,
        'CHAS': 0,
        'NOX': 0.5,
        'RM': rooms,
        'AGE': 30,
        'DIS': 5,
        'RAD': 4,
        'TAX': 300,
        'PTRATIO': 18,
        'LSTAT': 12
    }
    # ---------- Location ----------
    if location == "urban":
        data['DIS'] = 3
        data['NOX'] = 0.7
        data['TAX'] = 400
    elif location == "suburban":
        data['DIS'] = 6
        data['NOX'] = 0.5
        data['TAX'] = 300
    else:  # rural
        data['DIS'] = 9
        data['NOX'] = 0.3
        data['TAX'] = 200
    # ---------- Crime ----------
    if crime == "low":
        data['CRIM'] = 0.1
    elif crime == "medium":
        data['CRIM'] = 3
    else:  # high
        data['CRIM'] = 8
    # ---------- Property Age ----------
    if property_age == "new":
        data['AGE'] = 10
    elif property_age == "moderate":
        data['AGE'] = 40
    else:  # old
        data['AGE'] = 80
    # ---------- Area Quality ----------
    if area_quality == "premium":
        data['LSTAT'] = 5
        data['PTRATIO'] = 14
    elif area_quality == "average":
        data['LSTAT'] = 12
        data['PTRATIO'] = 18
    else:  # developing
        data['LSTAT'] = 20
        data['PTRATIO'] = 22
    # ---------- Feature Engineering ----------
    data['LivabilityScore'] = (
        data['RM'] * 2
        - data['CRIM']
        - data['LSTAT']
    )
    data['LocationScore'] = (
        data['DIS']
        + data['RAD']
        - data['NOX'] * 10
    )
    data['PropertyScore'] = (
        data['RM']
        - data['AGE'] * 0.01
        - data['TAX'] * 0.001
    )
    # ---------- Convert to DataFrame ----------
    df = pd.DataFrame([data])
    # Ensure same column order as training
    df = df[feature_columns]
    return df