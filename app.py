from flask import Flask, render_template, request
import LinearRegression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("Home.html")

@app.route('/use-cases')
def use_cases():
    return render_template('Use_Cases.html')

@app.route("/use-case1")
def Use_Case1():
    return render_template('Use_Case1.html')

@app.route('/use-case2')
def Use_Case2():
    return render_template('Use_Case2.html')

@app.route('/use-case3')
def Use_Case3():
    return render_template('Use_Case3.html')

@app.route('/use-case4')
def Use_Case4():
    return render_template('Use_Case4.html')

@app.route("/LinearRegression/")
def HomeLinearRegression():
    return render_template('HomeLinearRegression.html')

@app.route("/linear-regression-concepts")
def LinearRegressionConcept():
    return render_template('LinearRegressionConcept.html')

@app.route('/LinearRegression/application', methods=["GET", "POST"])
def application():
    result = None

    if request.method == "POST":
        grid = float(request.form["grid"])
        laps = float(request.form["laps"])
        result = LinearRegression.calculatePosition(grid, laps)

    plot = LinearRegression.generate_plot()

    return render_template(
        'LinearRegressionApplication.html',
        result=result,
        plot=plot
    )

@app.route("/LogisticRegression")
def logistic_menu():
    return render_template('HomeLogisticRegression.html')

@app.route("/LogisticRegressionConcepts")
def LogisticRegressionConcept():
    return render_template('LogisticRegressionConcept.html')

@app.route('/LogisticRegressionApplication', methods=["GET", "POST"])
def logistic_application():

    data = pd.read_csv('diabetes.csv')

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred), 2)
    recall = round(recall_score(y_test, y_pred), 2)
    f1 = round(f1_score(y_test, y_pred), 2)

    prediction = None

    if request.method == "POST":
        values = [float(request.form[x]) for x in [
            'Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age'
        ]]

        pred = model.predict([values])[0]

        prediction = "Diabetes" if pred == 1 else "No Diabetes"

    return render_template(
        'LogisticRegressionApplication.html',
        prediction=prediction,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )
@app.route("/SVM")
def SVM():
    return render_template('HomeSVM.html')
@app.route("/SVMConcepts")
def SVMConcepts():
    return render_template('SVMConcepts.html')
@app.route('/SVMApplication', methods=["GET", "POST"])
def SVMApplication():

    data = pd.read_csv('Titanic-Dataset.csv')

    data = data.drop(['Name','Ticket','Cabin','PassengerId'], axis=1)

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2})

    data = data.dropna()

    X = data.drop('Survived', axis=1)
    y = data['Survived']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred), 2)
    recall = round(recall_score(y_test, y_pred), 2)
    f1 = round(f1_score(y_test, y_pred), 2)

    prediction = None

    if request.method == "POST":
        values = [
            float(request.form["Pclass"]),
            float(request.form["Sex"]),
            float(request.form["Age"]),
            float(request.form["SibSp"]),
            float(request.form["Parch"]),
            float(request.form["Fare"]),
            float(request.form["Embarked"])
        ]

        values = scaler.transform([values])
        pred = model.predict(values)[0]

        prediction = "Survived" if pred == 1 else "Did Not Survive"

    return render_template(
        'SVMApplication.html',
        prediction=prediction,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )