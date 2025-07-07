import io
import base64
import os
from datetime import datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression # type: ignore
import firebase_admin
import google.generativeai as genai
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import pandas as pd
import yfinance as yf
from firebase_admin import credentials, db
from flask import Flask, render_template, request
from flask import jsonify, session
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('/Users/krishimehta/Desktop/project/money-3f17d-firebase-adminsdk-acnwl-9b1bf13ea8.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://money-3f17d-default-rtdb.asia-southeast1.firebasedatabase.app/'
})


# Firebase database reference
database_reference = db.reference('Users')
app.secret_key = os.urandom(24)
@app.route('/')
def home():
    # Serve the login page
    return render_template('index4.html')



@app.route('/login',methods=['GET', 'POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Simple validation
    if not username or not password:
        return jsonify({"message": "Both fields are required"}), 400

    # Query Firebase Realtime Database
    query_ref = database_reference.order_by_child('username').equal_to(username)
    users = query_ref.get()

    if users:
        for user_id, user_data in users.items():
            if user_data.get('password') == password:
                # Login successful
                session['user_details'] = {
                    'username': username,
                    'age': user_data.get('age', 'N/A'),
                    'profession': user_data.get('profession', 'N/A'),
                    'income': user_data.get('income', 'N/A'),
                    'expenditure': user_data.get('expenditure', 'N/A'),
                    'risk': user_data.get('risk', 'N/A')
                }
                session['chat_history'] = []  # Initialize chat history

                return jsonify({"message": "Correct login", "user_id": user_id}), 200

    return jsonify({"message": "Invalid username or password"}), 401


@app.route('/profile/<user_id>', methods=['GET', 'POST'])
def profile(user_id):
    if request.method == 'GET':
        # Fetch user data
        user_data = database_reference.child(user_id).get()
        if user_data:
            return render_template('index5.html', user=user_data, user_id=user_id)
        return jsonify({"message": "User not found"}), 404

    if request.method == 'POST':
        try:
            data = request.json
            password = data.get('password')
            age = data.get('age')

            risk = data.get('risk')
            income = data.get('income')
            expenditure = data.get('expenditure')
            profession = data.get('profession')


            if not password or not age:
                return jsonify({"error": "All fields are required"}), 400

            age = int(age)
            if age <= 18:
                return jsonify({"error": "Please enter a valid age"}), 400

            expenditure = int(expenditure)
            if expenditure <= 0:
                return jsonify({"error": "Please enter a valid age"}), 400
            income = int(income)
            if income <= 0:
                return jsonify({"error": "Please enter a valid age"}), 400


            updated_user = {
                'password': password,
                'age': age,
                'income': income,
                'expenditure': expenditure,
                'risk': risk,
                'profession': profession
            }

            # Check if username needs to be updated
            username = data.get('username')
            if username:
                updated_user['username'] = username

            existing_user = database_reference.child(user_id).get()
            if existing_user:
                database_reference.child(user_id).update(updated_user)
                return jsonify({"success": True, "message": "Profile updated successfully!"}), 200
            else:
                return jsonify({"error": "User not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500



#for budget asset
# Function to assess risk level based on user inputs
def risk_assessment(time_horizon, loss_reaction, financial_situation, experience, age):
    horizon_score = 1 if time_horizon < 3 else 2 if time_horizon <= 10 else 3
    reaction_score = 1 if loss_reaction == 'a' else 2 if loss_reaction == 'b' else 3
    financial_score = 3 if financial_situation == 'yes' else 1
    experience_score = 1 if experience == 'a' else 2 if experience == 'b' else 3
    age_score = 3 if age < 35 else 2 if age <= 50 else 1

    risk_score = horizon_score + reaction_score + financial_score + experience_score + age_score

    if risk_score <= 6:
        return "conservative", age
    elif 7 <= risk_score <= 11:
        return "balanced", age
    else:
        return "aggressive", age

# Fetch historical data and calculate returns and volatility
def fetch_historical_data(ticker, period='5y'):
    try:
        data = yf.Ticker(ticker)
        historical_data = data.history(period=period)
        returns = historical_data['Close'].pct_change()
        annualized_return = returns.mean() * 252  # Annualized return
        annualized_volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        return annualized_return, annualized_volatility, historical_data['Close']
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None, None

# Fetch VIX data
def fetch_vix_data(period='1mo'):
    try:
        vix_data = yf.Ticker("^VIX")
        vix_history = vix_data.history(period=period)
        vix_volatility = vix_history['Close'].mean()
        return vix_volatility
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return None

# Train a simple linear regression model
def predict_future_returns(ticker, historical_prices):
    historical_returns = historical_prices.pct_change().dropna()
    X = np.array(range(len(historical_returns))).reshape(-1, 1)  # Time index
    y = historical_returns.values

    model = LinearRegression()
    model.fit(X, y)

    future_time = np.array([[len(historical_returns) + i] for i in range(1, 13)])  # Predict for next 12 months
    future_returns = model.predict(future_time)

    return future_returns.mean() * 12  # Annualized predicted return

# Asset allocation based on risk profile, income, and age
def asset_allocation(income, risk_profile, age):
    asset_classes = {
        "equities": "SPY",
        "bonds": "BND",
        "cash": "SHV",
        "blue_chip": "VIG",
        "mid_cap": "MDY",
        "small_cap": "IJR",
        "mutual_fund": "VFIAX"
    }

    expected_returns = {}
    expected_volatility = {}

    for asset_class, ticker in asset_classes.items():
        annualized_return, annualized_volatility, historical_prices = fetch_historical_data(ticker)
        if annualized_return is not None:
            future_return = predict_future_returns(ticker, historical_prices)
            expected_returns[asset_class] = (annualized_return + future_return) / 2  # Average historical and predicted
            expected_volatility[asset_class] = annualized_volatility

    allocation = {
        "conservative": {"equities": 0.2, "bonds": 0.6, "cash": 0.2},
        "balanced": {"equities": 0.5, "bonds": 0.4, "cash": 0.1},
        "aggressive": {"equities": 0.8, "bonds": 0.15, "cash": 0.05}
    }

    equity_weight = allocation[risk_profile]["equities"]
    bond_weight = allocation[risk_profile]["bonds"]
    cash_weight = allocation[risk_profile]["cash"]

    if age < 35:
        equity_weight += 0.1
    elif age > 50:
        equity_weight -= 0.1

    if income > 100000:
        equity_weight += 0.1
    elif income < 50000:
        bond_weight += 0.1

    total_weight = equity_weight + bond_weight + cash_weight
    equity_weight /= total_weight
    bond_weight /= total_weight
    cash_weight /= total_weight

    # **Adding blue-chip, mid-cap, and small-cap stocks**
    blue_chip_weight = 0.3 if risk_profile == "aggressive" else 0.5
    mid_cap_weight = 0.2
    small_cap_weight = 0.2

    portfolio_return_percentage = (equity_weight * expected_returns.get("equities", 0) +
                                    bond_weight * expected_returns.get("bonds", 0) +
                                    cash_weight * expected_returns.get("cash", 0))

    portfolio_return_percentage += (
        blue_chip_weight * expected_returns.get("blue_chip", 0) +
        mid_cap_weight * expected_returns.get("mid_cap", 0) +
        small_cap_weight * expected_returns.get("small_cap", 0)
    )

    portfolio_return_absolute = (
        equity_weight * income * expected_returns.get("equities", 0) +
        bond_weight * income * expected_returns.get("bonds", 0) +
        cash_weight * income * expected_returns.get("cash", 0) +
        blue_chip_weight * income * expected_returns.get("blue_chip", 0) +
        mid_cap_weight * income * expected_returns.get("mid_cap", 0) +
        small_cap_weight * income * expected_returns.get("small_cap", 0)
    )
    

    vix_volatility = fetch_vix_data()

    sharpe_ratio = portfolio_return_percentage / (
        equity_weight * expected_volatility.get("equities", 1) +
        bond_weight * expected_volatility.get("bonds", 1) +
        cash_weight * expected_volatility.get("cash", 1)
    )

    return portfolio_return_percentage, vix_volatility, expected_returns, sharpe_ratio


@app.route('/ba/<user_id>', methods=['GET', 'POST'])
def ba(user_id):
    if request.method == 'GET':
        # Fetch user data
        user_data = database_reference.child(user_id).get()
        if user_data:
            return render_template('ba.html', user=user_data, user_id=user_id, var={})  # Pass an empty var for GET
        return jsonify({"message": "User not found"}), 404

    if request.method == 'POST':
        try:
            data = request.json
            # Convert incoming data to appropriate types
            income = int(data.get('income'))  # Ensure this is an integer
            age = int(data.get('age'))  # Ensure this is an integer
            time_horizon = data.get('time_horizon')  # This may need conversion depending on your logic
            loss_reaction = data.get('loss_reaction')  # Ensure this is in expected format
            financial_situation = data.get('financial_situation')  # Ensure this is in expected format
            experience = data.get('experience')  # Ensure this is in expected format

            updated_user = {
                'income': income,
                'time_horizon': time_horizon,
                'loss_reaction': loss_reaction,
                'financial_situation': financial_situation,
                'experience': experience,
                'age': age
            }

            # Risk assessment
            risk_profile, user_age = risk_assessment(time_horizon, loss_reaction, financial_situation, experience, age)

            # Asset allocation
            portfolio_return_percentage, vix_volatility, expected_returns = asset_allocation(income, risk_profile, user_age)

            # Example calculations for portfolio returns
            portfolio_return_absolute = income * portfolio_return_percentage
            risk_free_rate = 0.01  # Define your risk-free rate
            volatility = 0.2  # Define your volatility
            sharpe_ratio = (portfolio_return_percentage - risk_free_rate) / volatility

            # Create allocation details (without formatted strings)
            allocation_details = {
                "Equities": 0.4 * 100,
                "Bonds": 0.3 * 100,
                "Cash": 0.1 * 100,
                "Blue-Chip Stocks": 0.1 * 100,
                "Mid-Cap Stocks": 0.05 * 100,
                "Small-Cap Stocks": 0.05 * 100,
                "Total Expected Portfolio Return": portfolio_return_percentage * 100,
                "Expected Return in Absolute Terms": portfolio_return_absolute,
                "Sharpe Ratio": sharpe_ratio
            }

            # Plot expected returns
            plt.figure(figsize=(10, 6))
            plt.bar(expected_returns.keys(), [return_ * 100 for return_ in expected_returns.values()], color='skyblue')
            plt.xlabel("Asset Class")
            plt.ylabel("Expected Return (%)")
            plt.title("Expected Returns by Asset Class")
            plt.axhline(y=portfolio_return_percentage * 100, color='r', linestyle='--', label='Total Portfolio Expected Return')
            plt.legend()

            # Save the plot as an image
            plot_path = os.path.join('static', 'expected_returns_plot.png')
            plt.savefig(plot_path)
            plt.close()

            # Store values in new variables
            var = {
                "equities_value": allocation_details["Equities"],
                "bonds_value": allocation_details["Bonds"],
                "cash_value": allocation_details["Cash"],
                "blue_chip_value": allocation_details["Blue-Chip Stocks"],
                "mid_cap_value": allocation_details["Mid-Cap Stocks"],
                "small_cap_value": allocation_details["Small-Cap Stocks"],
                "total_return_value": allocation_details["Total Expected Portfolio Return"],
                "absolute_return_value": allocation_details["Expected Return in Absolute Terms"],
                "sharpe_ratio_value": allocation_details["Sharpe Ratio"]
            }

            # Check if username needs to be updated
            existing_user = database_reference.child(user_id).get()
            if existing_user:
                database_reference.child(user_id).update(updated_user)
            else:
                return jsonify({"error": "User not found"}), 404

            # Return the values as JSON for the front-end
            return jsonify({"success": True, "var": var, "plot_path": plot_path})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



