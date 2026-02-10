import pandas as pd 
import pickle 
from sklearn.linear_model import LinearRegression 

#load data 
df = pd.read_csv(r"C:\Users\kotap\Desktop\CI_CD_pipeline\data\data.csv")

x= df[["area","bedrooms"]]
y = df["price"]

#train model
model = LinearRegression()
model.fit(x,y)

# #save model 
with open("models\model.pkl","wb") as f:
    pickle.dump(model,f)