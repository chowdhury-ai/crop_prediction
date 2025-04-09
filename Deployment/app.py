from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model/crop_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(model)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data and convert to float
            data = [
                float(request.form['nitrogen']),
                float(request.form['phosphorous']),
                float(request.form['potassium']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]

            prediction = model.predict(np.array(data).reshape(1, -1))
            return render_template('index.html', prediction=prediction[0])
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
