from flask import Flask, request, jsonify, render_template
import joblib

model=joblib.load('model.pkl')



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    sentences = [x for x in request.form.values()]
    accent= model.predict(sentences).tolist()[0]
    result= f'The accent is {accent}'

    return render_template('index.html', prediction_text='{}'.format(result))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict(data).tolist()

    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
    

