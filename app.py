from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)


try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print("ERROR loading model/scaler:", str(e))


@app.route('/', methods=['GET'])
def home():
    return "Human Productivity Estimator is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([[ 
            data['Sleep_Hours'],
            data['Start_Work_Hour'],
            data['Total_Work_Hours'],
            data['Meetings_Count'],
            data['Interruptions_Count'],
            data['Break_Minutes'],
            data['Task_Completion_Rate'],
            data['SocialMedia_Min'],
            data['Emails_Handled']
        ]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        return jsonify({'productivity_score': round(prediction, 2)})

    except Exception as e:
        print("ERROR in /predict:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
