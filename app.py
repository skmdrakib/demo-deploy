import numpy as np
from flask import request, Flask, jsonify, render_template
import pickle

app = Flask(__name__)
model=pickle.load(open("model.pkl",'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    feature1=int(request.form['feature1'])
    feature2=int(request.form['feature2'])
    feature3=int(request.form['feature3'])
    feature4=int(request.form['feature4'])
    feature5=int(request.form['feature5'])

    user_input=np.array([[feature1,feature2,feature3,feature4,feature5]])

    prediction=model.predict(user_input)

    return render_template('index.html',predicition_text="Predicted cost $ {}".format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)