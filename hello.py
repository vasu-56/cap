from flask import Flask, render_template, request
import pickle
import pandas as pd
import sklearn

with open(f'model/lrmodel.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template("hello12.html")
    if request.method == 'POST':
        temp = request.form['temperature']
        humidity = request.form['humidity']
        windspeed = request.form['windspeed']
        rain = request.form['rain']
        visibility = request.form['visibility']
        invariables = pd.DataFrame([['temp', 'humidity', 'windspeed', 'rain', 'visibility']],
                                   columns=['temperature', 'humidity', 'windspeed', 'rain', 'visibility'], dtype=float)

        predict = model.predict(invariables)[0]

        return render_template('hello12.html', original_input={'Temperature': temp,
                                                               'Humidity': humidity,
                                                               'Windspeed': windspeed,
                                                               'Rain': rain,
                                                               'Visibility': visibility}, result=predict)


if __name__ == '__main__':
    app.run()
