import pickle
import numpy as np
import sklearn
from flask import Flask, render_template, request
app = Flask(__name__)
word_list = pickle.load(open('mystrings.pkl', 'rb'))
classifier = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('email')
    sample = []
    for i in word_list:
        sample.append(email.split(" ").count(i[0]))
    x = classifier.predict(np.array(sample).reshape(1, 3000))
    x = x[0]
    return render_template('index.html', label=str(x))

if __name__ == "__main__":
    app.run(debug=True)

