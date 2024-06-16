from flask import Flask,render_template,request,jsonify
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\anjan\Downloads\data.csv')

X = data[['Age','Annual Income (k$)','Spending Score (1-100)']].values


def km(n):
    model = KMeans(n_clusters=n)
    a=model.fit_predict(X)
   
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def main():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def form():
    if request.method=="POST":
        n=request.form.get("clusters")
        a=km(n)
        plt.scatter(X[a==0, 0], X[a==0, 1], c='red')
        plt.scatter(X[a==1, 0], X[a==1, 1], c='blue')
        plt.scatter(X[a==2, 0], X[a==2, 1], c='green')
        plt.scatter(X[a==3, 0], X[a==3, 1], c='cyan')
        plt.scatter(X[a==4, 0], X[a==4, 1], c='magenta')
        plt.show()
        return("done")
        return render_template("index.html")
    
if __name__=="__main__":
    app.run(host='localhost',port=5000)
    


