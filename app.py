from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('ipl_data.csv')

# Encoding categorical variables
label_encoder = LabelEncoder()
df['team1'] = label_encoder.fit_transform(df['team1'])
df['team2'] = label_encoder.fit_transform(df['team2'])
df['venue'] = label_encoder.fit_transform(df['venue'])
df['winner'] = label_encoder.fit_transform(df['winner'])

# Splitting the data into features and target variable
X = df[['team1', 'team2', 'venue']]
y = df['winner']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to make predictions
def predict_winner(team1, team2, venue):
    team1_encoded = label_encoder.transform([team1])[0]
    team2_encoded = label_encoder.transform([team2])[0]
    venue_encoded = label_encoder.transform([venue])[0]
    prediction = model.predict([[team1_encoded, team2_encoded, venue_encoded]])
    winner = label_encoder.inverse_transform(prediction)
    return winner[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    team1 = request.form['team1']
    team2 = request.form['team2']
    venue = request.form['venue']
    prediction = predict_winner(team1, team2, venue)
    return jsonify({'winner': prediction})

if __name__ == '__main__':
    app.run(debug=True)
