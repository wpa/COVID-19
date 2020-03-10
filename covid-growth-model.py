import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
#import matplotlib.pyplot as plt
#%matplotlib inline

url = "https://raw.githubusercontent.com/wpa/COVID-19/master/national-trend-data/covid19-pl-trends-national.csv"
#url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url)


df = df.loc[:,['date','total_infected']]
FMT = '%Y-%m-%d %H:%M:%S'
date = df['date']
df['date'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

def daynumber_to_date(daynumber):
    dt = datetime(2020,1,1)
    dtdelta = timedelta(days=daynumber)
    return dt + dtdelta

print(df)
x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
fit = curve_fit(logistic_model,x,y,p0=[2,100,20000])

a,b,c = fit[0]
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
print(errors)

sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
print(fit[1])
print(sol)
#now_date = datetime.now().timetuple().tm_yday
print(daynumber_to_date(b))
print(daynumber_to_date(sol))