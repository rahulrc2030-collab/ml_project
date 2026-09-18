from flask import Flask,request,render_template
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import custom_data,predictpipeline


app=Flask(__name__)

application=app


@application.route("/")
def index():
    return render_template("index.html")


@application.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        data=custom_data(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )

        pred_df=data.get_data_as_data_frame()

        print(pred_df)

        pridict_pipeline=predictpipeline()
        results=pridict_pipeline.predict(pred_df)
        return render_template("home.html",results=results[0])


    
    
        



    return  render_template("home.html")






if __name__=="__main__":
    application.run()
