import base64
import io
import os
import random
from flask import Flask, request, session, jsonify, render_template
from twilio.rest import Client
from datetime import datetime
from datetime import timedelta
import firebase_admin
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')  # Switch to a non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from firebase_admin import credentials, db
from flask import Flask, render_template, request
from flask import jsonify, session
from flask_cors import CORS


import spacy
import pandas as pd
from difflib import get_close_matches

# Twilio credentials directly embedded in the code
TWILIO_ACCOUNT_SID = "ACb4f743d0c6f50e56798235bb2799c1b5"
TWILIO_AUTH_TOKEN = "d704c833d0e776b4f7458fe5a39554c6"
TWILIO_PHONE_NUMBER = "+12495440508"

# Initialize Twilio Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        # Serve the signup page
        return render_template('index3.html')

    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        phone = data.get('phone')
        age = data.get('age')


        if not username or not password or not phone or not age:
            return jsonify({"error": "All fields are required"}), 400

        try:
            age = int(age)
            if age < 18:
                raise ValueError()
        except ValueError:
            return jsonify({"error": "Please enter a valid legal age"}), 400

        user_ref = db.reference('Users')
        user_ref.push({
            'username': username,
            'password': password,
            'phone': phone,
            'age': age,

        })

        return jsonify({"success": True, "message": "Sign-up successful!"}), 201


@app.route('/login', methods=['GET', 'POST'])
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
                session['user_details'] = {key: user_data.get(key, 'N/A') for key in user_data}
                session['chat_history'] = []  # Initialize chat history

                # Generate a 6-digit OTP
                otp = random.randint(1000, 9999)
                session['otp'] = otp  # Store OTP in session

                # Send OTP via SMS to user's phone number
                user_phone_number = user_data.get('phone')
                if user_phone_number:
                    try:
                        message = client.messages.create(
                            body=f"Your OTP is: {otp}",
                            from_=TWILIO_PHONE_NUMBER,
                            to=user_phone_number
                        )
                        return jsonify({"message": "OTP sent successfully!", "user_id": user_id}), 200
                    except Exception as e:
                        return jsonify({"error": "Failed to send OTP", "details": str(e)}), 500
                else:
                    return jsonify({"message": "User phone number not found"}), 400

    return jsonify({"message": "Invalid username or password"}), 401

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    entered_otp = request.form.get('otp')  # Accept OTP via form data (application/x-www-form-urlencoded)
    stored_otp = session.get('otp')

    if stored_otp and int(entered_otp) == stored_otp:
        session.pop('otp')  # Clear OTP from session after successful verification
        return jsonify({"message": "OTP verified! Login complete."}), 200
    else:
        return jsonify({"message": "Invalid OTP. Please try again."}), 400



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






# Load the Spacy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Predefined set of words related to "prediction"
prediction_keywords = {"prediction", "predictions", "forecast", "forecasts", "projection", "projections"}

bse_file_path = '/Users/krishimehta/Desktop/project/indlist.xlsx'
nasdaq_file_path = '/Users/krishimehta/Desktop/project/iamericalist.csv'

# Read the BSE Excel file (assuming first sheet contains the needed data)
bse_data = pd.read_excel(bse_file_path, sheet_name=0)

# Read the NASDAQ CSV file
nasdaq_data = pd.read_csv(nasdaq_file_path)

# Create the dictionary for BSE (Company Name to Ticker Symbol)
bse_dict = dict(zip(bse_data['Issuer Name'], bse_data['Security Id']))

# Create the dictionary for NASDAQ (Company Name to Ticker Symbol)
nasdaq_dict = dict(zip(nasdaq_data['Name'], nasdaq_data['Symbol']))







#stock code
def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    info = stock.info
    return hist, info


# Fundamental Analysis Tools
def calculate_fundamentals(info):
    pe_ratio = info.get("trailingPE", np.nan)
    eps = info.get("trailingEps", np.nan)
    de_ratio = info.get("debtToEquity", np.nan)
    roe = info.get("returnOnEquity", np.nan)
    dividend_yield = info.get("dividendYield", np.nan) * 100 if info.get("dividendYield", np.nan) else np.nan
    market_cap = info.get("marketCap", np.nan)
    beta = info.get("beta", np.nan)
    return pe_ratio, eps, de_ratio, roe, dividend_yield, market_cap, beta


# Technical Analysis Tools
def calculate_technicals(hist):
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    hist['BB_upper'], hist['BB_lower'] = compute_bollinger_bands(hist['Close'])
    hist['ATR'] = compute_atr(hist)
    rsi = compute_rsi(hist['Close'])
    macd, signal, macd_histogram = compute_macd(hist['Close'])
    stochastic_k, stochastic_d = compute_stochastic_oscillator(hist)
    momentum = hist['Close'].pct_change().fillna(0)
    return hist, hist['MA50'], hist['MA200'], rsi, macd, signal, macd_histogram, stochastic_k, stochastic_d, momentum


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd - signal
    return macd, signal, macd_histogram


def compute_stochastic_oscillator(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=3).mean()
    return k, d


def compute_bollinger_bands(data, window=20, num_std=2):
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, lower_band


def compute_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


# Weight assignment
def assign_weights(fundamentals, technicals):
    fundamental_weights = [0.2] * 7  # 7 weights for 7 fundamentals
    technical_weights = [0.1] * 8  # 8 weights for 8 technicals

    fundamental_score = np.nansum(np.array(fundamentals) * np.array(fundamental_weights))
    technical_score = np.nansum(np.array(technicals) * np.array(technical_weights))

    return fundamental_score, technical_score


# Combining scores
def combine_scores(fundamental_score, technical_score):
    combined_score = (fundamental_score + technical_score) / 2
    return combined_score


# Future price prediction based on combined score and trend confirmation
def predict_future_price(hist, combined_score, trend_confirmation, future_date):
    current_price = hist['Close'].iloc[-1]
    days_to_predict = (future_date - datetime.now()).days

    if trend_confirmation:
        trend_factor = 1 + (combined_score / 100)
    else:
        trend_factor = 1 - (combined_score / 100)

    future_price = current_price * (trend_factor ** (days_to_predict / 365))
    return future_price


#for 2 stock
# Fetch historical data using yfinance
def get_stock_data2(symbol, end):
    start = "2021-01-01"  # Hardcoded start date
    try:
        stock_data = yf.download(symbol, start=start, end=end)
        if stock_data.empty:
            raise ValueError(f"No data found for {symbol}.")
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


# Calculate indicators
def moving_average2(data, window):
    return data['Close'].rolling(window=window).mean()


def calculate_macd2(data, fastperiod=12, slowperiod=26, signalperiod=9):
    data['EMA12'] = data['Close'].ewm(span=fastperiod, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=slowperiod, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACDSignal'] = data['MACD'].ewm(span=signalperiod, adjust=False).mean()
    return data


def calculate_rsi2(data, period=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Prevent divide-by-zero error
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    data['RSI'] = 100 - (100 / (1 + rs))
    return data


def calculate_vwap2(data):
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return data


def calculate_bollinger_bands2(data, window=20):
    data['MiddleBand'] = moving_average2(data, window)
    data['StdDev'] = data['Close'].rolling(window=window, min_periods=1).std()
    data['UpperBand'] = data['MiddleBand'] + (2 * data['StdDev'])
    data['LowerBand'] = data['MiddleBand'] - (2 * data['StdDev'])
    return data


def calculate_risk2(data):
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=14).std() * np.sqrt(252)
    return data


# Monte Carlo Simulation for future price prediction
def monte_carlo_simulation2(data, num_simulations, num_days):
    last_price = data['Close'].iloc[-1]
    returns = data['Returns'].dropna()

    # Parameters for simulations
    mean_return = returns.mean()
    volatility = returns.std()

    # Simulating future price paths
    simulation_results = np.zeros((num_simulations, num_days))

    for sim in range(num_simulations):
        future_prices = [last_price]

        for day in range(1, num_days):
            # Random factor for price movement
            price_change = np.random.normal(mean_return, volatility)
            new_price = future_prices[-1] * (1 + price_change)
            future_prices.append(new_price)

        simulation_results[sim, :] = future_prices

    # Getting statistics
    price_range = {
        'Minimum Price': np.min(simulation_results[:, -1]),
        'Maximum Price': np.max(simulation_results[:, -1]),
        'Average Price': np.mean(simulation_results[:, -1])
    }

    return price_range, simulation_results


# Make prediction based on indicators
def make_prediction2(data):
    current_data = data.iloc[-1]

    # Adjust weights for indicators
    weights = {
        'MACD': 0.25,
        'RSI': 0.25,
        'VWAP': 0.25,
        'BollingerBands': 0.25
    }

    bullish_score = 0
    bearish_score = 0

    # MACD
    if current_data['MACD'] > current_data['MACDSignal']:
        bullish_score += weights['MACD']
    else:
        bearish_score += weights['MACD']

    # RSI
    if current_data['RSI'] < 30:
        bullish_score += weights['RSI']
    elif current_data['RSI'] > 70:
        bearish_score += weights['RSI']

    # VWAP
    if current_data['Close'] > current_data['VWAP']:
        bullish_score += weights['VWAP']
    else:
        bearish_score += weights['VWAP']

    # Bollinger Bands
    if current_data['Close'] < current_data['LowerBand']:
        bullish_score += weights['BollingerBands']
    elif current_data['Close'] > current_data['UpperBand']:
        bearish_score += weights['BollingerBands']

    # Final Prediction
    trend = "Uptrend (Bullish)" if bullish_score > bearish_score else "Downtrend (Bearish)"

    return {
        'Trend': trend,
        'Reason': f"Bullish Score: {bullish_score}, Bearish Score: {bearish_score}"
    }


# Plot data and indicators
def plot_data2(data, simulation_results=None):
    plt.figure(figsize=(14, 7))

    # Plot historical data
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['MA50'], label='50-Day MA', linestyle='--', color='green')
    plt.plot(data['MA200'], label='200-Day MA', linestyle='--', color='orange')
    plt.plot(data['VWAP'], label='VWAP', linestyle='--', color='purple')
    plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2, label='Bollinger Bands')

    # Plot Monte Carlo simulation paths
    if simulation_results is not None:
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(simulation_results.shape[1])]
        for sim in simulation_results:
            plt.plot(future_dates, sim, alpha=0.1, color='red')

    plt.legend()
    plt.title(' Indicators ')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)

    # Save plot to a PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

@app.route('/stock', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        future_date_str = request.form['future_date']
        future_date = datetime.strptime(future_date_str, "%Y-%m-%d")



        #2
        symbol = request.form['ticker'].upper()
        end = datetime.today().strftime('%Y-%m-%d')

        # Fetch stock data
        data = get_stock_data2(symbol, end)

        data['MA50'] = moving_average2(data, 50)
        data['MA200'] = moving_average2(data, 200)
        data = calculate_macd2(data)
        data = calculate_rsi2(data)
        data = calculate_vwap2(data)
        data = calculate_bollinger_bands2(data)
        data = calculate_risk2(data)

        # Make prediction
        prediction = make_prediction2(data)

        # Run Monte Carlo simulation
        num_simulations = 1000
        num_days = 30
        price_range, simulation_results = monte_carlo_simulation2(data, num_simulations, num_days)

        # Plot the data and indicators
        plot_url = plot_data2(data, simulation_results)


        #2 end



        hist, info = fetch_data(ticker)
        fundamentals = calculate_fundamentals(info)
        hist, ma50, ma200, rsi, macd, signal, macd_histogram, stochastic_k, stochastic_d, momentum = calculate_technicals(hist)

        trend_confirmation = ma50.iloc[-1] > ma200.iloc[-1]

        technicals = [ma50.iloc[-1], ma200.iloc[-1], rsi.iloc[-1], macd.iloc[-1], macd_histogram.iloc[-1],
                      stochastic_k.iloc[-1], stochastic_d.iloc[-1], momentum.iloc[-1]]

        fundamental_score, technical_score = assign_weights(fundamentals, technicals)
        combined_score = combine_scores(fundamental_score, technical_score)

        future_price = predict_future_price(hist, combined_score, trend_confirmation, future_date)

        # Generate the graph
        plt.figure(figsize=(10, 6))
        plt.plot(hist.index, hist['Close'], label='Historical Price')
        plt.axhline(y=ma50.iloc[-1], color='r', linestyle='-', label='50-Day MA')
        plt.axhline(y=ma200.iloc[-1], color='b', linestyle='-', label='200-Day MA')
        plt.scatter(future_date, future_price, color='green', label=f'Predicted Price: ${future_price:.2f}')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save the plot to a bytes object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        pe_ratio, eps, de_ratio, roe, dividend_yield, market_cap, beta = fundamentals
        pat = info.get("netIncomeToCommon", np.nan)  # PAT
        capex_sales = info.get("capexToSales", np.nan)  # CapEx/Sales
        nopat_tic = info.get("nopatToTIC", np.nan)  # NOPAT/Total Invested Capital


        return render_template('index6.html', ticker=ticker, future_date=future_date_str,
                               future_price=f"${future_price:.2f}", symbol=symbol, prediction=prediction,
                               price_range=price_range, plot_url=plot_url,
                               graph_url=f"data:image/png;base64,{graph_url}",
                               pe_ratio=pe_ratio, eps=eps, de_ratio=de_ratio, roe=roe, dividend_yield=dividend_yield,
                               market_cap=market_cap, beta=beta, pat=pat, capex_sales=capex_sales, nopat_tic=nopat_tic)


    return render_template('index6.html')










def fetch_mutual_fund_datamf(ticker, period="5y"):
    fund = yf.Ticker(ticker)
    histmf = fund.history(period=period)
    infomf = fund.info
    return histmf, infomf

# Fundamental Analysis for mutual funds
def calculate_fundamentalsmf(infomf):
    expense_ratio_mf = float(infomf.get("expenseRatio", np.nan))  # Ensure this is float
    category_mf = infomf.get("category", "Unknown")
    net_assets_mf = float(infomf.get("totalAssets", np.nan))  # Ensure this is float
    return expense_ratio_mf, category_mf, net_assets_mf

# Technical Analysis for mutual funds (based on NAV)
def calculate_technicalsmf(histmf):
    histmf['MA50_mf'] = histmf['Close'].rolling(window=50).mean()
    histmf['MA200_mf'] = histmf['Close'].rolling(window=200).mean()
    histmf['BB_upper_mf'], histmf['BB_lower_mf'] = compute_bollinger_bandsmf(histmf['Close'])
    histmf['ATR_mf'] = compute_atrmf(histmf)
    rsi_mf = compute_rsimf(histmf['Close'])
    macd_mf, signal_mf, macd_histogram_mf = compute_macdmf(histmf['Close'])
    return histmf, histmf['MA50_mf'], histmf['MA200_mf'], rsi_mf, macd_mf, signal_mf, macd_histogram_mf

# Calculate RSI
def compute_rsimf(data_mf, window=14):
    delta_mf = data_mf.diff()
    gain_mf = (delta_mf.where(delta_mf > 0, 0)).rolling(window=window).mean()
    loss_mf = (-delta_mf.where(delta_mf < 0, 0)).rolling(window=window).mean()
    rs_mf = gain_mf / loss_mf.replace(0, np.nan)  # Avoid division by zero
    return 100 - (100 / (1 + rs_mf))

# Calculate MACD
def compute_macdmf(data_mf, short_window=12, long_window=26, signal_window=9):
    short_ema_mf = data_mf.ewm(span=short_window, adjust=False).mean()
    long_ema_mf = data_mf.ewm(span=long_window, adjust=False).mean()
    macd_mf = short_ema_mf - long_ema_mf
    signal_mf = macd_mf.ewm(span=signal_window, adjust=False).mean()
    macd_histogram_mf = macd_mf - signal_mf
    return macd_mf, signal_mf, macd_histogram_mf

# Calculate Bollinger Bands
def compute_bollinger_bandsmf(data_mf, window=20, num_std=2):
    ma_mf = data_mf.rolling(window=window).mean()
    std_mf = data_mf.rolling(window=window).std()
    upper_band_mf = ma_mf + (std_mf * num_std)
    lower_band_mf = ma_mf - (std_mf * num_std)
    return upper_band_mf, lower_band_mf

# Calculate ATR
def compute_atrmf(data_mf, window=14):
    high_low_mf = data_mf['High'] - data_mf['Low']
    high_close_mf = np.abs(data_mf['High'] - data_mf['Close'].shift())
    low_close_mf = np.abs(data_mf['Low'] - data_mf['Close'].shift())
    ranges_mf = pd.concat([high_low_mf, high_close_mf, low_close_mf], axis=1)
    true_range_mf = ranges_mf.max(axis=1)
    atr_mf = true_range_mf.rolling(window=window).mean()
    return atr_mf

# Monte Carlo Simulation for future price prediction
def monte_carlo_simulationmf(histmf, num_simulations_mf, num_days_mf):
    last_price_mf = histmf['Close'].iloc[-1]
    returns_mf = histmf['Close'].pct_change().dropna()

    mean_return_mf = returns_mf.mean()
    volatility_mf = returns_mf.std()

    simulation_results_mf = np.zeros((num_simulations_mf, num_days_mf))
    for sim_mf in range(num_simulations_mf):
        future_prices_mf = [last_price_mf]
        for day_mf in range(1, num_days_mf):
            price_change_mf = np.random.normal(mean_return_mf, volatility_mf)
            new_price_mf = future_prices_mf[-1] * (1 + price_change_mf)
            future_prices_mf.append(new_price_mf)
        simulation_results_mf[sim_mf, :] = future_prices_mf

    price_range_mf = {
        'Minimum Price': np.min(simulation_results_mf[:, -1]),
        'Maximum Price': np.max(simulation_results_mf[:, -1]),
        'Average Price': np.mean(simulation_results_mf[:, -1])
    }
    return price_range_mf, simulation_results_mf

# Weight assignment for mutual funds based on category
# Weight assignment for mutual funds based on category
def assign_weightsmf(fundamentals_mf, technicals_mf, category_mf):
    fundamentals_mf = [f_mf if isinstance(f_mf, (int, float)) else np.nan for f_mf in fundamentals_mf]
    technicals_mf = [t_mf if isinstance(t_mf, (int, float)) else np.nan for t_mf in technicals_mf]

    fundamentals_mf = np.array(fundamentals_mf, dtype=float)
    technicals_mf = np.array(technicals_mf, dtype=float)

    # Define weights based on the number of metrics
    fundamental_weights_mf = [0.3, 0.4, 0.3]  # 3 metrics
    technical_weights_mf = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]  # 6 metrics

    # Check the lengths before calculating scores
    if len(fundamental_weights_mf) != len(fundamentals_mf):
        raise ValueError("Fundamental weights length does not match the number of fundamental metrics.")
    if len(technical_weights_mf) != len(technicals_mf):
        raise ValueError("Technical weights length does not match the number of technical metrics.")

    fundamental_score_mf = np.nansum(fundamentals_mf * fundamental_weights_mf)
    technical_score_mf = np.nansum(technicals_mf * technical_weights_mf)
    return fundamental_score_mf, technical_score_mf


# Combine scores
def combine_scoresmf(fundamental_score_mf, technical_score_mf):
    combined_score_mf = (fundamental_score_mf + technical_score_mf) / 2
    return combined_score_mf

# Predict future price based on combined score
def predict_future_navmf(histmf, combined_score_mf, future_date_mf):
    current_price_mf = histmf['Close'].iloc[-1]
    days_to_predict_mf = (future_date_mf - datetime.now()).days
    trend_factor_mf = 1 + (combined_score_mf / 100)
    future_price_mf = current_price_mf * (trend_factor_mf ** (days_to_predict_mf / 365))
    return future_price_mf

# Plot mutual fund data
def plot_datamf(histmf, simulation_results_mf=None):
    plt.figure(figsize=(14, 7))
    plt.plot(histmf['Close'], label='NAV', color='blue')
    plt.plot(histmf['MA50_mf'], label='50-Day MA', linestyle='--', color='green')
    plt.plot(histmf['MA200_mf'], label='200-Day MA', linestyle='--', color='orange')
    plt.fill_between(histmf.index, histmf['BB_upper_mf'], histmf['BB_lower_mf'], color='gray', alpha=0.2, label='Bollinger Bands')

    if simulation_results_mf is not None:
        future_dates_mf = [histmf.index[-1] + timedelta(days=i) for i in range(simulation_results_mf.shape[1])]
        for sim_mf in simulation_results_mf:
            plt.plot(future_dates_mf, sim_mf, alpha=0.1, color='red')

    plt.legend()
    plt.title('Mutual Fund NAV and Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)

    img_mf = io.BytesIO()
    plt.savefig(img_mf, format='png')
    img_mf.seek(0)
    plot_url_mf = base64.b64encode(img_mf.getvalue()).decode()
    plt.close()  # Close the plot to avoid display issues
    return plot_url_mf

@app.route('/mf', methods=['GET', 'POST'])
def indexmf():
    if request.method == 'POST':
        ticker_mf = request.form['ticker']
        future_date_str_mf = request.form['future_date']
        future_date_mf = datetime.strptime(future_date_str_mf, "%Y-%m-%d")

        # Fetch historical data
        histmf, infomf = fetch_mutual_fund_datamf(ticker_mf)
        fundamentals_mf = calculate_fundamentalsmf(infomf)

        # Calculate technical indicators
        histmf, ma50_mf, ma200_mf, rsi_mf, macd_mf, signal_mf, macd_histogram_mf = calculate_technicalsmf(histmf)
        technicals_mf = [
            ma50_mf.iloc[-1],
            ma200_mf.iloc[-1],
            rsi_mf.iloc[-1],
            macd_mf.iloc[-1],
            signal_mf.iloc[-1],
            macd_histogram_mf.iloc[-1]
        ]

        # Assign weights and combine scores based on category
        fundamental_score_mf, technical_score_mf = assign_weightsmf(fundamentals_mf, technicals_mf, fundamentals_mf[1])
        combined_score_mf = combine_scoresmf(fundamental_score_mf, technical_score_mf)

        # Predict future price (NAV)
        future_nav_mf = predict_future_navmf(histmf, combined_score_mf, future_date_mf)

        # Run Monte Carlo simulation
        num_simulations_mf = 1000
        num_days_mf = max((future_date_mf - datetime.now()).days, 1)  # Ensure at least 1 day
        price_range_mf, simulation_results_mf = monte_carlo_simulationmf(histmf, num_simulations_mf, num_days_mf)

        # Plot the data
        plot_url_mf = plot_datamf(histmf, simulation_results_mf)

        expense_ratio_mf, category_mf, net_assets_mf = fundamentals_mf

        return render_template('index8.html',
                               ticker=ticker_mf,
                               future_date=future_date_mf,
                               future_nav=future_nav_mf,
                               price_range=price_range_mf,
                               plot_url=plot_url_mf,
                               expense_ratio=expense_ratio_mf,
                               category=category_mf,
                               net_assets=net_assets_mf)
    return render_template('index8.html')






api_key = 'AIzaSyB3QjNktbfPxDvcLHIYRWKuKSN_X04nhTA'

genai.configure(api_key=api_key)

# Create the model
generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
)
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':



        user_input = request.form['user_input']

        words_in_text = set(user_input.lower().split())
        if prediction_keywords & words_in_text:
            doc = nlp(user_input)
            company_name = None
            for ent in doc.ents:
                if ent.label_ == "ORG":  # Look for organizations
                    company_name = ent.text
                    break

            if company_name:
                company_name = company_name.strip().lower()  # Convert to lowercase and strip spaces

                # Find the closest match for BSE
                bse_match = get_close_matches(company_name, bse_dict.keys(), n=1, cutoff=0.6)
                if bse_match:
                    ticker = bse_dict[bse_match[0]]  # Save the ticker
                    ticker = ticker+".BO"
                else:
                    # Find the closest match for NASDAQ
                    nasdaq_match = get_close_matches(company_name, nasdaq_dict.keys(), n=1, cutoff=0.6)
                    if nasdaq_match:
                        ticker = nasdaq_dict[nasdaq_match[0]]  # Save the ticker

# give me predictions for reliance
                response_ready = True
                future_date_str = "2024-10-21"
                future_date = datetime.strptime(future_date_str, "%Y-%m-%d")
                # 2
                symbol = ticker
                end = datetime.today().strftime('%Y-%m-%d')

                # Fetch stock data
                data = get_stock_data2(symbol, end)

                data['MA50'] = moving_average2(data, 50)
                data['MA200'] = moving_average2(data, 200)
                data = calculate_macd2(data)
                data = calculate_rsi2(data)
                data = calculate_vwap2(data)
                data = calculate_bollinger_bands2(data)
                data = calculate_risk2(data)

                # Make prediction
                prediction = make_prediction2(data)

                # Run Monte Carlo simulation
                num_simulations = 1000
                num_days = 30
                price_range, simulation_results = monte_carlo_simulation2(data, num_simulations, num_days)

                # Plot the data and indicators
                plot_url = plot_data2(data, simulation_results)

                # 2 end

                hist, info = fetch_data(ticker)
                fundamentals = calculate_fundamentals(info)
                hist, ma50, ma200, rsi, macd, signal, macd_histogram, stochastic_k, stochastic_d, momentum = calculate_technicals(
                    hist)

                trend_confirmation = ma50.iloc[-1] > ma200.iloc[-1]

                technicals = [ma50.iloc[-1], ma200.iloc[-1], rsi.iloc[-1], macd.iloc[-1], macd_histogram.iloc[-1],
                              stochastic_k.iloc[-1], stochastic_d.iloc[-1], momentum.iloc[-1]]

                fundamental_score, technical_score = assign_weights(fundamentals, technicals)
                combined_score = combine_scores(fundamental_score, technical_score)

                future_price = predict_future_price(hist, combined_score, trend_confirmation, future_date)

                # Generate the graph
                plt.figure(figsize=(10, 6))
                plt.plot(hist.index, hist['Close'], label='Historical Price')
                plt.axhline(y=ma50.iloc[-1], color='r', linestyle='-', label='50-Day MA')
                plt.axhline(y=ma200.iloc[-1], color='b', linestyle='-', label='200-Day MA')
                plt.scatter(future_date, future_price, color='green', label=f'Predicted Price: ${future_price:.2f}')
                plt.title(f'{ticker} Stock Price Prediction')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)

                # Save the plot to a bytes object
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                graph_url = base64.b64encode(img.getvalue()).decode()
                pe_ratio, eps, de_ratio, roe, dividend_yield, market_cap, beta = fundamentals
                pat = info.get("netIncomeToCommon", np.nan)  # PAT
                capex_sales = info.get("capexToSales", np.nan)  # CapEx/Sales
                nopat_tic = info.get("nopatToTIC", np.nan)  # NOPAT/Total Invested Capital

                response_ready = True

            return render_template('index.html', chat_history=session.get('chat_history', []), ticker=ticker,
                                   future_date=future_date_str,
                                   future_price=f"${future_price:.2f}", symbol=symbol, prediction=prediction,
                                   price_range=price_range, plot_url=plot_url,
                                   graph_url=f"data:image/png;base64,{graph_url}",
                                   pe_ratio=pe_ratio, eps=eps, de_ratio=de_ratio, roe=roe,
                                   dividend_yield=dividend_yield,
                                   market_cap=market_cap, beta=beta, pat=pat, capex_sales=capex_sales,
                                   nopat_tic=nopat_tic, response_ready=response_ready)


        else:
            chat_history = session.get('chat_history', [])

            # Include user details with the instruction
            user_details = session.get('user_details', {})
            user_details['instruction'] = "Give short to the point answers. Do not ask the user any questions."
            chat_history.append({
                'role': 'user',
                'parts': [{'text': f"User details: {user_details}"}]  # Ensure it's wrapped in 'parts'
            })

            # Append the user input in the correct structure
            chat_history.append({
                'role': 'user',
                'parts': [{'text': user_input}]  # The 'parts' key contains a list of dictionaries with 'text'
            })

            # Start or continue the chat session with the correct history format
            chat_session = model.start_chat(history=chat_history)
            response = chat_session.send_message(user_input)
            response = response.text.replace('*', '')
            # Append the assistant's response to the chat history in the correct structure
            chat_history.append({
                'role': 'model',  # Ensure the role is what the API expects
                'parts': [{'text': response}]  # The assistant's response is also wrapped in 'parts'
            })
            session['chat_history'] = chat_history  # Update session with the correct format

        return render_template('index.html', chat_history=chat_history)

    return render_template('index.html', chat_history=session.get('chat_history', []))



#goallll
def fetch_asset_data(ticker):
    data = yf.Ticker(ticker).history(period="5y")["Close"]
    return data


def calculate_asset_return(ticker):
    data = fetch_asset_data(ticker)
    returns = data.pct_change().dropna()
    annualized_return = returns.mean() * 252
    return annualized_return


def get_best_asset(asset_tickers):
    best_return = float('-inf')
    best_ticker = ""

    for ticker in asset_tickers:
        annual_return = calculate_asset_return(ticker)
        if annual_return > best_return:
            best_return = annual_return
            best_ticker = ticker

    return best_ticker, best_return


def perform_asset_allocation():
    asset_classes = {
        "Low Risk": ["SHV", "BIL", "TLT"],
        "Mid Risk": ["IJR", "SCHA", "VTWO"],
        "High Risk": ["ARKK", "TQQQ", "SPY"]
    }

    best_assets = {}
    for asset_class, tickers in asset_classes.items():
        best_assets[asset_class] = get_best_asset(tickers)

    return best_assets


def calculate_investment_period(goal_date):
    today = datetime.today()
    investment_period_years = (goal_date - today).days / 365
    return investment_period_years


def calculate_required_return(goal_amount, lump_sum, savings_per_month, investment_period_years):
    required_return = (goal_amount - lump_sum - (savings_per_month * 12 * investment_period_years)) / lump_sum
    required_return /= investment_period_years

    if required_return > 0.3:
        return None

    return required_return


def prompt_risk_factor(risk_input):
    if risk_input == 'low':
        return 0.05
    elif risk_input == 'medium':
        return 0.15
    elif risk_input == 'high':
        return 0.25
    else:
        return 0.15


def calculate_expected_returns():
    best_assets = perform_asset_allocation()
    expected_returns = {asset_class: annual_return for asset_class, (_, annual_return) in best_assets.items()}
    return expected_returns


def calculate_required_monthly_investment(goal_amount, lump_sum, savings_per_month, investment_period_years):
    investment_needed = goal_amount - lump_sum
    total_months = investment_period_years * 12
    required_monthly_investment = investment_needed / total_months

    if required_monthly_investment > savings_per_month:
        return None

    return required_monthly_investment


def calculate_asset_allocations(lump_sum, required_monthly_investment, risk_factor):
    allocation_percentages = {
        "Low Risk": (1 - risk_factor) * 0.5,
        "Mid Risk": risk_factor * 0.6,
        "High Risk": risk_factor * 0.4
    }

    total_percentage = sum(allocation_percentages.values())
    for key in allocation_percentages:
        allocation_percentages[key] /= total_percentage

    lump_sum_allocations = {asset_class: lump_sum * percent for asset_class, percent in allocation_percentages.items()}
    monthly_allocations = {asset_class: required_monthly_investment * percent for asset_class, percent in
                           allocation_percentages.items()}

    return lump_sum_allocations, monthly_allocations


# Plot data for expected returns and amount invested
def plot_investment_data(expected_returns, amounts_invested):
    plt.figure(figsize=(14, 7))

    # Set the bar width
    bar_width = 0.35
    index = np.arange(len(expected_returns))

    # Create bars for expected returns
    bars1 = plt.bar(index, expected_returns.values(), bar_width, label='Expected Returns', color='blue')

    # Create a second Y-axis for amounts invested
    ax2 = plt.gca().twinx()  # Create a twin Axes sharing the x-axis

    # Create bars for amounts invested, offsetting them
    bars2 = ax2.bar(index + bar_width, amounts_invested.values(), bar_width, label='Amount Invested', color='orange')

    # Adding labels and title
    plt.xlabel('Asset Class')
    plt.ylabel('Expected Return (%)', color='blue')
    ax2.set_ylabel('Amount Invested ($)', color='orange')
    plt.title('Expected Returns and Amount Invested per Asset Class')

    # Set x-ticks to asset classes
    plt.xticks(index + bar_width / 2, expected_returns.keys())

    # Adding legends for both y-axes
    plt.legend(handles=[bars1[0], bars2[0]], labels=['Expected Returns', 'Amount Invested'], loc='upper left')
    ax2.legend(loc='upper right')

    # Adding grid for better visibility
    plt.grid(True)

    # Save plot to a PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()  # Close the plot to free memory
    return plot_url


@app.route("/goal/<user_id>", methods=["GET", "POST"])
def goal_investment_calculator(user_id):
    if request.method == 'GET':
        # Fetch user data
        user_data = database_reference.child(user_id).get()
        if user_data:
            # Fetch existing goal data
            goal_data = database_reference.child(user_id).child(
                "goals").get()  # Assuming goals are stored under the user ID
            return render_template('goal.html', user=user_data, goal_data=goal_data, user_id=user_id)
        return jsonify({"message": "User not found"}), 404

    if request.method == "POST":
        try:
            # Get form data
            goal_name = request.form["goal_name"]
            goal_amount = float(request.form["goal_amount"])
            lump_sum = float(request.form["lump_sum"])
            savings_per_month = float(request.form["savings_per_month"])
            goal_date_str = request.form["goal_date"]
            goal_date = datetime.strptime(goal_date_str, "%Y-%m-%d")

            today = datetime.today()
            investment_period_years = (goal_date - today).days / 365

            # Calculate required return
            required_return = (goal_amount - lump_sum - (savings_per_month * 12 * investment_period_years)) / lump_sum
            required_return /= investment_period_years

            if required_return > 0.3:
                return render_template("goal.html",
                                       error="Achieving this goal is highly unlikely as it requires an annual return greater than 30%.")

            risk_factor = request.form["risk_factor"]
            risk_factor_map = {
                "low": 0.05,
                "medium": 0.15,
                "high": 0.25
            }
            risk_factor = risk_factor_map.get(risk_factor, 0.15)

            asset_classes = {
                "Low Risk": ["SHV", "BIL", "TLT"],
                "Mid Risk": ["IJR", "SCHA", "VTWO"],
                "High Risk": ["ARKK", "TQQQ", "SPY"]
            }

            best_assets = {}
            for asset_class, tickers in asset_classes.items():
                best_ticker, annual_return = get_best_asset(tickers)
                best_assets[asset_class] = (best_ticker, annual_return)

            expected_returns = {asset_class: annual_return for asset_class, (_, annual_return) in best_assets.items()}

            required_monthly_investment = (goal_amount - lump_sum) / (investment_period_years * 12)
            if required_monthly_investment > savings_per_month:
                return render_template("goal.html",
                                       error=f"The required monthly investment of ${required_monthly_investment:.2f} exceeds your savings of ${savings_per_month:.2f}.")

            allocation_percentages = {
                "Low Risk": (1 - risk_factor) * 0.5,
                "Mid Risk": risk_factor * 0.6,
                "High Risk": risk_factor * 0.4
            }

            total_percentage = sum(allocation_percentages.values())
            for key in allocation_percentages:
                allocation_percentages[key] /= total_percentage

            lump_sum_allocations = {asset_class: lump_sum * percent for asset_class, percent in
                                    allocation_percentages.items()}
            monthly_allocations = {asset_class: required_monthly_investment * percent for asset_class, percent in
                                   allocation_percentages.items()}

            # Save the goal data to the database
            goal_data = {
                'goal_name': goal_name,
                'goal_amount': goal_amount,
                'lump_sum': lump_sum,
                'savings_per_month': savings_per_month,
                'goal_date': goal_date_str,
                'expected_returns': expected_returns,
                'lump_sum_allocations': lump_sum_allocations,
                'monthly_allocations': monthly_allocations
            }

            database_reference.child(user_id).child("goals").push(goal_data)  # Save goal data under user ID

            # Generate the plot URL
            plot_url = plot_investment_data(expected_returns, lump_sum_allocations)

            # Calculate portfolio composition and risk level
            portfolio_allocation = best_assets  # This can be replaced with your own logic for portfolio allocation
            portfolio_composition = []
            for asset_class, (ticker, _) in portfolio_allocation.items():
                portfolio_composition.append(f"{asset_class} (Best Asset: {ticker})")

            portfolio_return = sum(
                expected_returns[asset_class] * allocation_percentages[asset_class] for asset_class in
                allocation_percentages)
            portfolio_warning = ""
            if portfolio_return < goal_amount:
                portfolio_warning = f"Warning: Total expected return is below your goal amount of ${goal_amount:.2f}."
            else:
                portfolio_warning = "Success: Total expected return meets or exceeds your goal."

            return render_template(
                "goal.html",
                goal_name=goal_name,
                goal_amount=goal_amount,
                investment_period_years=investment_period_years,
                expected_returns=expected_returns,
                lump_sum_allocations=lump_sum_allocations,
                monthly_allocations=monthly_allocations,
                goal_amount_display=goal_amount,
                plot_url=plot_url,  # Pass the plot URL to the template
                portfolio_composition=portfolio_composition,
                portfolio_return=portfolio_return,
                portfolio_warning=portfolio_warning
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("goal.html")








'''
@app.route('/home', methods=['GET', 'POST'])
def homepage():
    # Get user details from session
    user_details = session.get('user_details', {'username': 'Guest'})
    username = user_details.get('username')

    # Query the database using the username
    query_ref = database_reference.order_by_child('username').equal_to(username)
    users = query_ref.get()

    user_id = None
    for user_id, user_data in users.items():
        # Update session with user data
        session['user_details'] = {
            'username': user_data.get('username', 'Guest'),
            'age': user_data.get('age', 'N/A'),
            'profession': user_data.get('profession', 'N/A'),
            'income': user_data.get('income', 'N/A'),
            'expenditure': user_data.get('expenditure', 'N/A'),
            'risk': user_data.get('risk', 'N/A')
        }
        session['chat_history'] = []

    if not user_id:
        return jsonify({"message": "User not found"}), 404

    # Render the template with user details
    return render_template('home.html', user=user_details, user_id=user_id)
'''

# for budget asset
EXPECTED_RETURNS = {
    'High Risk': 0.08,
    'Mid Risk': 0.06,
    'Low Risk': 0.04,
    'Very High': 0.10,
    'Bonds': 0.03,
    'Cash': 0.01
}

ALLOC_PERCENTAGES = {
    'High Risk': (0.70, 0.20, 0.10),
    'Moderate Risk': (0.50, 0.30, 0.20),
    'Low Risk': (0.30, 0.50, 0.20)
}


# Function to validate positive values
def validate_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


# Function to validate expected return
def validate_expected_return(option):
    option_mapping = {
        1: 'Low Risk',
        2: 'Mid Risk',
        3: 'High Risk',
        4: 'Very High'
    }
    if option not in option_mapping:
        raise ValueError("Invalid option for expected return. Please choose 1, 2, 3, or 4.")
    return EXPECTED_RETURNS[option_mapping[option]]


# Function to assign risk label based on user input
def assign_risk_label(age, expected_return, self_assessed_risk):
    if self_assessed_risk in ['high', 'moderate', 'low']:
        return self_assessed_risk.capitalize() + " Risk"

    if age < 30 and expected_return >= 0.08:
        return 'High Risk'
    elif 30 <= age < 50 and expected_return >= 0.06:
        return 'Moderate Risk'
    else:
        return 'Low Risk'


# Function to fetch real-time data
def get_real_time_data(asset_classes):
    asset_data = {}
    for asset_class, tickers in asset_classes.items():
        try:
            volatilities = [
                yf.Ticker(ticker).history(period='6mo')['Close'].pct_change().std()
                for ticker in tickers
            ]
            historical_volatility = np.mean(volatilities) if volatilities else 0
            asset_data[asset_class] = historical_volatility
        except Exception as e:
            print(f"Error fetching data for {asset_class}: {e}")
            asset_data[asset_class] = 0
    return asset_data


# Function to calculate asset allocation
def calculate_allocation(user_profile):
    total_investment = user_profile['investable_amount']
    equities_pct, bonds_pct, cash_pct = ALLOC_PERCENTAGES[user_profile['risk_label']]

    return {
        'Equities': total_investment * equities_pct,
        'Bonds': total_investment * bonds_pct,
        'Cash': total_investment * cash_pct
    }


# Function to recommend funds based on allocation
def recommend_funds(allocation_strategy):
    funds = {
        'Equities High Risk': ['S&P 500 Growth Fund', 'Tech Growth ETF', 'Emerging Markets Fund'],
        'Equities Mid Risk': ['S&P 500 Index Fund', 'Total Market Index Fund', 'Balanced Fund'],
        'Equities Low Risk': ['Dividend Growth Fund', 'Conservative Allocation Fund', 'Value Fund'],
        'Bonds': 'Government Bond Index',
        'Cash': ['Money Market Fund', 'Short-Term Bond Fund']
    }

    recommendations = {}
    equities_amount = allocation_strategy.get('Equities', 0)

    high_risk_amount = equities_amount * 0.6
    mid_risk_amount = equities_amount * 0.3
    low_risk_amount = equities_amount * 0.1

    if high_risk_amount > 0:
        recommendations['Equities High Risk'] = funds['Equities High Risk']
    if mid_risk_amount > 0:
        recommendations['Equities Mid Risk'] = funds['Equities Mid Risk']
    if low_risk_amount > 0:
        recommendations['Equities Low Risk'] = funds['Equities Low Risk']

    if allocation_strategy.get('Bonds', 0) > 0:
        recommendations['Bonds'] = funds['Bonds']
    if allocation_strategy.get('Cash', 0) > 0:
        recommendations['Cash'] = funds['Cash']

    return recommendations


# Function to plot the investment strategy
def plot_investment_strategy(allocation):
    asset_classes = ['Equities High Risk', 'Equities Mid Risk', 'Equities Low Risk', 'Bonds', 'Cash']
    expected_returns_dollars = {
        'Equities High Risk': allocation['Equities'] * 0.6 * EXPECTED_RETURNS['High Risk'],
        'Equities Mid Risk': allocation['Equities'] * 0.3 * EXPECTED_RETURNS['Mid Risk'],
        'Equities Low Risk': allocation['Equities'] * 0.1 * EXPECTED_RETURNS['Low Risk'],
        'Bonds': allocation['Bonds'] * EXPECTED_RETURNS['Bonds'],
        'Cash': allocation['Cash'] * EXPECTED_RETURNS['Cash']
    }
    invested_amounts = {
        'Equities High Risk': allocation['Equities'] * 0.6,
        'Equities Mid Risk': allocation['Equities'] * 0.3,
        'Equities Low Risk': allocation['Equities'] * 0.1,
        'Bonds': allocation['Bonds'],
        'Cash': allocation['Cash']
    }

    plt.figure(figsize=(14, 7))
    bar_width = 0.35
    index = np.arange(len(asset_classes))

    bars1 = plt.bar(index, expected_returns_dollars.values(), bar_width, label='Expected Returns ($)', color='blue')
    ax2 = plt.gca().twinx()
    bars2 = ax2.bar(index + bar_width, invested_amounts.values(), bar_width, label='Amount Invested ($)',
                    color='orange')

    plt.xlabel('Asset Class')
    plt.ylabel('Expected Return ($)', color='blue')
    ax2.set_ylabel('Amount Invested ($)', color='orange')
    plt.title('Expected Returns and Amount Invested per Asset Class')
    plt.xticks(index + bar_width / 2, asset_classes)
    plt.legend(handles=[bars1[0]], labels=['Expected Returns ($)'], loc='upper left')
    ax2.legend(handles=[bars2[0]], labels=['Amount Invested ($)'], loc='upper right')
    plt.grid(True)

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return plot_url


@app.route('/ba', methods=['GET', 'POST'])
def ba():
    results = None  # Initialize results variable
    if request.method == 'POST':
        # Take user input
        age = int(request.form['age'])
        income = float(request.form['income'])
        expected_return = int(request.form['expected_return'])
        investable_amount = float(request.form['investable_amount'])
        self_assessed_risk = request.form['self_assessed_risk'].strip().lower()

        # Create user profile
        try:
            expected_return_value = validate_expected_return(expected_return)
            risk_label = assign_risk_label(age, expected_return_value, self_assessed_risk)

            user_profile = {
                'age': age,
                'income': income,
                'expected_return': expected_return_value,
                'investable_amount': validate_positive(investable_amount, "Investable Amount"),
                'risk_label': risk_label
            }

            # Calculate allocation
            allocation_strategy = calculate_allocation(user_profile)
            fund_recommendations = recommend_funds(allocation_strategy)

            # Generate plot
            plot_url = plot_investment_strategy(allocation_strategy)

            # Create a results dictionary to pass to the template
            results = {
                'allocation': allocation_strategy,
                'funds': fund_recommendations,
                'plot_url': plot_url
            }
        except ValueError as e:
            return render_template('ba.html', error=str(e), results=results)

    return render_template('ba.html', results=results)


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/landing/<user_id>', methods=['GET', 'POST'])
def dashboard():
    return render_template('landing.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
