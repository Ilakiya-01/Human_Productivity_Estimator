from flask import Flask, request, jsonify
import numpy as np
import pickle

# lets Load our saved model
model = pickle.load(open('model.pkl', 'rb'))

# I used scaler.transform, so we must also load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "🎯 Human Productivity Estimator is LIVE!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # let's Extract  features in the correct order
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

    # This step is to Scale features before prediction
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    return jsonify({'productivity_score': round(prediction, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
