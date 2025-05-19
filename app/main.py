from fastapi import FastAPI,UploadFile,File
import pandas as pd
from io import StringIO
from app.model import model,encoder,scalar

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Titanic Survival Prediction API is running"}

@app.post("/predict")
async def predict_csv(file:UploadFile=File(...)):
    if not file.filename.endswith('.csv'):
        return {"message":"Please upload a csv file"}
    
    contents=await file.read()
    data_str=contents.decode('utf-8')
    new_data=pd.read_csv(StringIO(data_str))

    new_data.columns=new_data.columns.str.strip().str.lower()
    new_data.rename(columns={
        'passengerid':'PassengerId',
        'pclass':'Pclass',
        'sex':'Sex',
        'age':'Age',
        'sibsp':'SibSp',
        'parch':'Parch',
        'embarked':'Embarked'
        },inplace=True)
    
    drop_cols=['PassengerId',"name","ticket","fare","cabin"]
    new_data.drop(drop_cols,axis=1,inplace=True)

    if "Age"in new_data.columns:
        new_data["Age"].fillna(new_data["Age"].mean(),inplace=True)

    if "Embarked" in new_data.columns:
        new_data["Embarked"].fillna(new_data["Embarked"].mode()[0],inplace=True)

    if "SibSp" in new_data.columns and "Parch" in new_data.columns:
        new_data["Family"] = new_data["SibSp"] + new_data["Parch"]
        new_data.drop(["SibSp","Parch"],axis=1,inplace=True)

    catagorical_cols=["Sex","Embarked"]
    new_data[catagorical_cols]=encoder.transform(new_data[catagorical_cols])
   
    feature_cols=["Pclass", "Sex", "Age", "Embarked", "Family"]
    
    
  
    X_test_final=new_data[feature_cols]
    X_test_final=scalar.transform(X_test_final)
    
    prediction=model.predict(X_test_final)
    new_data["Survived"]=prediction
    return new_data[["Pclass", "Sex", "Age", "Embarked", "Family","Survived"]].to_dict(orient="records")
