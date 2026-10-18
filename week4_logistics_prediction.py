"""Week 4 Logistics Prediction. Install: pip install pandas numpy matplotlib scikit-learn
Run alongside week4_logistics_prediction_dataset.csv: python week4_logistics_prediction.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("week4_logistics_prediction_dataset.csv")
features = ["Distance_km","Weight_kg","Shipment_Volume_units","Transport_Mode","Region","Weather","Traffic_Level"]
X, y = df[features], df["Delivery_Time_days"]
num = ["Distance_km","Weight_kg","Shipment_Volume_units"]
cat = ["Transport_Mode","Region","Weather","Traffic_Level"]
prep = ColumnTransformer([
 ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cat)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
models={"Linear Regression":LinearRegression(),"Decision Tree":DecisionTreeRegressor(max_depth=8,min_samples_leaf=4,random_state=42),"Random Forest":RandomForestRegressor(n_estimators=160,min_samples_leaf=3,max_features=.9,random_state=42)}
for name, model in models.items():
 pipe=Pipeline([("prep",prep),("model",model)]); pipe.fit(X_train,y_train); p=pipe.predict(X_test)
 print(name,"MAE",mean_absolute_error(y_test,p),"RMSE",mean_squared_error(y_test,p)**.5,"R2",r2_score(y_test,p))
pipe=Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=120,random_state=42))])
grid=GridSearchCV(pipe,{"model__max_depth":[6,10,None],"model__min_samples_leaf":[2,4,6]},cv=KFold(n_splits=5,shuffle=True,random_state=42),scoring="neg_mean_absolute_error",n_jobs=-1)
grid.fit(X_train,y_train); p=grid.best_estimator_.predict(X_test)
print("Best parameters",grid.best_params_)
print("Tuned Random Forest", "MAE",mean_absolute_error(y_test,p),"RMSE",mean_squared_error(y_test,p)**.5,"R2",r2_score(y_test,p))
for mode in ["Road","Rail","Air","Sea"]:
 scenario=X_test.iloc[:100].copy(); scenario["Transport_Mode"]=mode
 print(mode,"average predicted days",grid.best_estimator_.predict(scenario).mean())
