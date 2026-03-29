from flask import Flask, render_template, request
import LinearRegression
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
def LogisticRegression():
    return render_template('HomeLogisticRegression.html')