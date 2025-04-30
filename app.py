from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from  sklearn.linear_model import LogisticRegression

app = Flask(__name__)


diabetes = pd.read_csv("diabetes.csv")
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:

        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        input_np = np.asarray(input_data).reshape(1, -1)


        scaled_input = scaler.transform(input_np)


        prediction = model.predict(scaled_input)

        result = 'This person has diabetes.' if prediction[0] == 1 else 'This person does NOT have diabetes.'

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)


