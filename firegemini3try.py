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

    

    portfolio_return_percentage = (equity_weight * expected_returns.get("equities", 0) +
                                    bond_weight * expected_returns.get("bonds", 0) +
                                    cash_weight * expected_returns.get("cash", 0))

    vix_volatility = fetch_vix_data()

    return portfolio_return_percentage, vix_volatility, expected_returns

# Route to display form and get inputs
@app.route('/ba', methods=['GET', 'POST'])
def portfolio():
    if request.method == 'POST':
        # Collect form data
        income = float(request.form['income'])
        time_horizon = int(request.form['time_horizon'])
        loss_reaction = request.form['loss_reaction']
        financial_situation = request.form['financial_situation']
        experience = request.form['experience']
        age = int(request.form['age'])

        # Assess risk
        risk_profile, age = risk_assessment(time_horizon, loss_reaction, financial_situation, experience, age)

        # Asset allocation
        portfolio_return_percentage, vix_volatility, expected_returns = asset_allocation(income, risk_profile, age)

        # Generate a bar chart
        img = io.BytesIO()
        plt.bar(expected_returns.keys(), [ret * 100 for ret in expected_returns.values()], color='skyblue')
        plt.xlabel("Asset Class")
        plt.ylabel("Expected Return (%)")
        plt.title("Expected Returns by Asset Class")
        plt.axhline(y=portfolio_return_percentage * 100, color='r', linestyle='--', label='Total Portfolio Expected Return')
        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('ba.html', portfolio_return_percentage=portfolio_return_percentage * 100, vix_volatility=vix_volatility, plot_url=plot_url)

    return render_template('ba.html')





html:<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            height: 100vh;
            background-color: #f4f4f4;
        }

        /* Header */
        header {
            background: linear-gradient(90deg, #002852, #3A6E8F 50%, #002852);
            width: 100%;
            padding: 20px;
            text-align: center;
            color: white;
            position: fixed;
            top: 0;
            left: 0;
            z-index: 1000;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            animation: fadeIn 1s ease-in-out;
        }

        .header-content h1 {
            margin: 0;
            font-size: 2.5rem;
            line-height: 1.2;
            filter: drop-shadow(0px 0px 10px rgba(0,0,0, 0.5));
            font-weight: bold;
        }

        footer {
            background: linear-gradient(90deg, #002852, #3A6E8F 50%, #002852);
            width: 100%;
            padding: 10px 0;
            text-align: center;
            color: white;
            position: fixed;
            bottom: 0;
            left: 0;
            display: flex;
            animation: fadeIn 1s ease-in-out;
        }

        /* Card container */
        .card-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr); /* 4 columns */
            grid-gap: 20px;
            max-width: 1200px;
            margin: 100px auto; /* Adjust the margin as per your header height */
            padding: 20px;
            background-color: #e5ddd5;
            border-radius: 8px;
        }

        .employee-card {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1), 0px 1px 3px rgba(0, 0, 0, 0.6);
            max-width: 100%;
        }

        .employee-card strong {
            display: block;
            font-size: 1.2rem;
            margin-bottom: 5px;
        }

        .employee-card p {
            margin: 5px 0;
        }

        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin-top: 20px; /* Adjust as needed to center vertically */
        }

        button {
            width: 200px;
            padding: 12px;
            border-radius: 20px;
            cursor: pointer;
            background-color: #3C8DAD;
            color: white;
            border: none;
            transition: background-color 0.3s ease-in-out, transform 0.3s ease;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1), 0px 1px 3px rgba(0, 0, 0, 0.6);
            margin-bottom: 20px;
        }

        button:hover {
            background-color: #3C8DAD;
            transform: scale(1.05);
        }

        h2 {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            font-size: 2rem;
            color: #002852;
            margin: 20px auto;
            margin-top: 40px;
        }
    </style>
</head>
<body>
<header>
    <div class="header-content">
        <h1>MoneyZZ, money made easy</h1>
    </div>
</header>

<div class="contact-container" id="contactContainer">
    <h1>Portfolio Page</h1>
    <form method="POST">
        <h2>Enter your portfolio details: </h2>

        <div class="card-container">
            <div class="employee-card">
                <strong>Income:</strong>
                <input type="number" name="income" required>
            </div>
            <div class="employee-card">
                <strong>Time Horizon (in years):</strong>
                <input type="number" name="time_horizon" required>
            </div>
            <div class="employee-card">
                <strong>Loss Reaction (a, b, c):</strong>
                <select name="loss_reaction" required>
                    <option value="a">A - Avoid Losses</option>
                    <option value="b">B - Accept Small Losses</option>
                    <option value="c">C - Can Tolerate Significant Losses</option>
                </select>
            </div>
            <div class="employee-card">
                <strong>Financial Situation (yes or no):</strong>
                <select name="financial_situation" required>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                </select>
            </div>
            <div class="employee-card">
                <strong>Experience Level (a, b, c):</strong>
                <select name="experience" required>
                    <option value="a">A - Beginner</option>
                    <option value="b">B - Intermediate</option>
                    <option value="c">C - Experienced</option>
                </select>
            </div>
            <div class="employee-card">
                <strong>Age:</strong>
                <input type="number" name="age" required>
            </div>
        </div>

        <div class="button-container">
            <button type="submit">Calculate Portfolio</button>
        </div>
    </form>

    {% if portfolio_return_percentage %}
    <h2>Your Portfolio Return Percentage: {{ portfolio_return_percentage }}%</h2>
    <h2>VIX Volatility: {{ vix_volatility }}</h2>
    <h2>Expected Returns by Asset Class:</h2>
    <img src="data:image/png;base64,{{ plot_url }}" alt="Expected Returns Chart">
    {% endif %}
    
    <div class="button-container">
        <button onclick="window.location.href = '/';">Back to HomePage</button>
    </div>
</div>

<footer>
    <p>&copy; MoneyZZ, a Capstone Project. All rights reserved.</p>
</footer>
</body>
</html>






new code:
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

    portfolio_return_percentage = (equity_weight * expected_returns.get("equities", 0) +
                                    bond_weight * expected_returns.get("bonds", 0) +
                                    cash_weight * expected_returns.get("cash", 0))

    vix_volatility = fetch_vix_data()

    return portfolio_return_percentage, vix_volatility, expected_returns

# Route to display form and get inputs
@app.route('/ba/<user_id>', methods=['GET', 'POST'])
def ba(user_id):
    if request.method == 'GET':
        # Fetch user data from Firebase (assuming you have a Firebase reference)
        user_data = database_reference.child(user_id).get()
        if user_data:
            return render_template('ba.html', user=user_data, user_id=user_id)
        return jsonify({"message": "User not found"}), 404

    if request.method == 'POST':
        try:
            data = request.json  # Get the data from the request
            age = data.get('age')
            income = data.get('income')
            time_horizon = data.get('time_horizon')
            loss_reaction = data.get('loss_reaction')
            financial_situation = data.get('financial_situation')
            experience = data.get('experience')

            # Update user data in Firebase
            updated_user = {
                'income': income,
                'time_horizon': time_horizon,
                'loss_reaction': loss_reaction,
                'financial_situation': financial_situation,
                'experience': experience,
                'age': age
            }

            existing_user = database_reference.child(user_id).get()
            if existing_user:
                database_reference.child(user_id).update(updated_user)

                # Assess risk using a custom risk assessment function
                risk_profile, age = risk_assessment(
                    time_horizon, loss_reaction, financial_situation, experience, age)

                # Asset allocation using a custom asset allocation function
                portfolio_return_percentage, vix_volatility, expected_returns = asset_allocation(
                    income, risk_profile, age)

                # Generate a bar chart using matplotlib
                img = io.BytesIO()
                plt.bar(expected_returns.keys(), [ret * 100 for ret in expected_returns.values()], color='skyblue')
                plt.xlabel("Asset Class")
                plt.ylabel("Expected Return (%)")
                plt.title("Expected Returns by Asset Class")
                plt.axhline(y=portfolio_return_percentage * 100, color='r', linestyle='--', label='Total Portfolio Expected Return')
                plt.legend()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')

                # Return the updated template with portfolio details and chart
                return render_template(
                    'ba.html',
                    portfolio_return_percentage=portfolio_return_percentage * 100,
                    vix_volatility=vix_volatility,
                    plot_url=plot_url,
                    user={'income': income, 'age': age}
                )
            else:
                return jsonify({"error": "User not found"}), 404

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Invalid request method"}), 405




    new html:
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            height: 100vh;
            background-color: #f4f4f4;
        }

        header {
            background: linear-gradient(90deg, #002852, #3A6E8F 50%, #002852);
            width: 100%;
            padding: 20px;
            text-align: center;
            color: white;
            position: fixed;
            top: 0;
            left: 0;
            z-index: 1000;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .header-content h1 {
            margin: 0;
            font-size: 2.5rem;
            line-height: 1.2;
            filter: drop-shadow(0px 0px 10px rgba(0,0,0, 0.5));
            font-weight: bold;
        }

        footer {
            background: linear-gradient(90deg, #002852, #3A6E8F 50%, #002852);
            width: 100%;
            padding: 10px 0;
            text-align: center;
            color: white;
            position: fixed;
            bottom: 0;
            left: 0;
        }

        .card-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-gap: 20px;
            max-width: 1200px;
            margin: 100px auto;
            padding: 20px;
            background-color: #e5ddd5;
            border-radius: 8px;
        }

        .employee-card {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1), 0px 1px 3px rgba(0, 0, 0, 0.6);
            max-width: 100%;
        }

        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin-top: 20px;
        }

        button {
            width: 200px;
            padding: 12px;
            border-radius: 20px;
            cursor: pointer;
            background-color: #3C8DAD;
            color: white;
            border: none;
            transition: background-color 0.3s ease-in-out, transform 0.3s ease;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1), 0px 1px 3px rgba(0, 0, 0, 0.6);
            margin-bottom: 20px;
        }

        button:hover {
            background-color: #3C8DAD;
            transform: scale(1.05);
        }

        h2 {
            text-align: center;
            font-size: 2rem;
            color: #002852;
            margin: 20px auto;
            margin-top: 40px;
        }
    </style>
    <script>
        // This function can be removed if you're no longer fetching data via JavaScript
        function fetchIncomeAndAge() {
            const user_id = document.getElementById('user_id').value;
            const age = document.getElementById('age').value;
            const income = document.getElementById('income').value;
            const time_horizon = document.getElementById('time_horizon').value;
            const loss_reaction = document.getElementById('loss_reaction').value;
            const financial_situation = document.getElementById('financial_situation').value;
            const experience = document.getElementById('experience').value;

            fetch(`/ba/${user_id}`, {
                method: 'POST', // Change to 'GET' since you're fetching data
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ age, income, time_horizon, loss_reaction, financial_situation, experience}),
            })
            .then(async response => {
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText);
                }
                return response.json();
            })
            .then(data => {
            if (data.success) {
                alert('data got');
            } else {
                alert(data.error || 'Failed to get data');
            }
        })
            .catch(error => {
                alert('Error fetching data: ' + error.message);
            });
        }
    </script>
</head>

<body>
<header>
    <div class="header-content">
        <h1>MoneyZZ, money made easy</h1>
        <nav class="nav-bar">
            <a href="/about">About</a>
            <a href="/contact">Contact Us</a>
            <a href="/social">Social Media</a>
        </nav>
    </div>
</header>

<div class="contact-container" id="contactContainer">
    <h1>Portfolio Page</h1>
    <form onsubmit="event.preventDefault(); fetchIncomeAndAge();">
        <h2>Enter your portfolio details: </h2>
        <!-- Hidden field to store the user_id -->
        <input type="hidden" id="user_id" value="{{ user_id }}">

        <div class="card-container">
            <div class="employee-card">
                <strong>Income</strong>
                <input type="number" id="income" placeholder="Income" value="{{ user['income'] }}">
            </div>
            <div class="employee-card">
                <strong>Time Horizon (in years):</strong>
                <input type="number" name="time_horizon" placeholder="Time Horizon" value="{{ user['time_horizon'] }} " required>
            </div>
            <div class="employee-card">
                <strong>Loss Reaction (a, b, c):</strong>
                <select name="Loss Reaction" placeholder="Loss Reaction" value="{{ user['loss_reaction'] }} " required>
                    <option value="a">A - Avoid Losses</option>
                    <option value="b">B - Accept Small Losses</option>
                    <option value="c">C - Can Tolerate Significant Losses</option>
                </select>
            </div>
            <div class="employee-card">
                <strong>Financial Situation (yes or no):</strong>
                <select name="financial_situation" placeholder="financial_situation" value="{{ user['financial_situation'] }} " required>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                </select>
            </div>
            <div class="employee-card">
                <strong>Experience Level (a, b, c):</strong>
                <select name="experience" placeholder="experience" value="{{ user['experience'] }} " required>
                    <option value="a">A - Beginner</option>
                    <option value="b">B - Intermediate</option>
                    <option value="c">C - Experienced</option>
                </select>
            </div>
            <div class="employee-card">
                <strong>Age:</strong>
                <input type="number" id="age" name="age" placeholder="Age" value="{{ user['age'] }}" readonly required>
            </div>
        </div>
         <div class="button-container">
            <button type="submit">Calculate Portfolio</button>
        </div>
</form>
{% if portfolio_return_percentage %}
    <h2>Your Portfolio Return Percentage: {{ portfolio_return_percentage }}%</h2>
    <h2>VIX Volatility: {{ vix_volatility }}</h2>
    <h2>Expected Returns by Asset Class:</h2>
    <img src="data:image/png;base64,{{ plot_url }}" alt="Expected Returns Chart">
    {% endif %}
    
    <button onclick="window.location.href = '/';">Back to HomePage</button>
</div>

<footer>
    <p>&copy; MoneyZZ, a Capstone Project. All rights reserved.</p>
</footer>
</body>
</html>





onsubmit="event.preventDefault(); fetchIncomeAndAge();"







clean code:
@app.route('/ba/<user_id>', methods=['GET', 'POST'])
def ba(user_id):
    if request.method == 'GET':
        # Fetch user data
        user_data = database_reference.child(user_id).get()
        if user_data:
            return render_template('ba.html', user=user_data, user_id=user_id)
        return jsonify({"message": "User not found"}), 404
    
    if request.method == 'POST':
        try:# Collect form data
            data = request.json  # Get the data from the request
            age = data.get('age')
            income = data.get('income')
            time_horizon = data.get('time_horizon')
            loss_reaction = data.get('loss_reaction')
            financial_situation = data.get('financial_situation')
            experience = data.get('experience')

            updated_user = {
                'income': income,
                'time_horizon': time_horizon,
                'loss_reaction': loss_reaction,
                'financial_situation': financial_situation,
                'experience': experience,
                'age': age
            }
            existing_user = database_reference.child(user_id).get()
            if existing_user:
                database_reference.child(user_id).update(updated_user)
                # Assess risk
                risk_profile, age = risk_assessment(time_horizon, loss_reaction, financial_situation, experience, age)

                # Asset allocation
                portfolio_return_percentage, vix_volatility, expected_returns = asset_allocation(income, risk_profile, age)

                # Generate a bar chart
                img = io.BytesIO()
                plt.bar(expected_returns.keys(), [ret * 100 for ret in expected_returns.values()], color='skyblue')
                plt.xlabel("Asset Class")
                plt.ylabel("Expected Return (%)")
                plt.title("Expected Returns by Asset Class")
                plt.axhline(y=portfolio_return_percentage * 100, color='r', linestyle='--', label='Total Portfolio Expected Return')
                plt.legend()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                return render_template('ba.html', portfolio_return_percentage=portfolio_return_percentage * 100, vix_volatility=vix_volatility, plot_url=plot_url, user={'income': income, 'age': age})
        
            else:
                return jsonify({"error": "User not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Invalid request method"}), 405       




latest edit:




@app.route('/ba/<user_id>', methods=['GET', 'POST'])
def ba(user_id):
    if request.method == 'GET':
        # Fetch user data
        user_data = database_reference.child(user_id).get()
        if user_data:
            return render_template('ba.html', user=user_data, user_id=user_id)
        return jsonify({"message": "User not found"}), 404

    if request.method == 'POST':
        try:
            data = request.json
            income = data.get('income')
            age = data.get('age')
            time_horizon = data.get('time_horizon')
            loss_reaction = data.get('loss_reaction')
            financial_situation = data.get('financial_situation')
            experience = data.get('experience')

            # Validate input data
            if not all([income, age, time_horizon, loss_reaction, financial_situation, experience]):
                return jsonify({"error": "Missing required data"}), 400

            updated_user = {
                'income': income,
                'time_horizon': time_horizon,
                'loss_reaction': loss_reaction,
                'financial_situation': financial_situation,
                'experience': experience,
                'age': age
            }

            # Check if user exists
            existing_user = database_reference.child(user_id).get()
            if existing_user:
                database_reference.child(user_id).update(updated_user)
                return jsonify({"success": True, "message": "Profile updated successfully!"}), 200
            else:
                return jsonify({"error": "User not found"}), 404

        except Exception as e:
            return jsonify({"error": str(e)}), 500







        risk_profile, user_age = risk_assessment(time_horizon, loss_reaction, financial_situation, experience, age)

        # Asset allocation
        portfolio_return_percentage, vix_volatility, expected_returns = asset_allocation(income, risk_profile, user_age)

                    # Calculate weights and portfolio return (Dummy calculations for illustration)
        equity_weight = 0.4  # Example weight
        bond_weight = 0.3
        cash_weight = 0.1
        blue_chip_weight = 0.1
        mid_cap_weight = 0.05
        small_cap_weight = 0.05
                    
                    # Example calculations for portfolio returns
        portfolio_return_absolute = income * portfolio_return_percentage
        sharpe_ratio = (portfolio_return_percentage - risk_free_rate) / volatility  # Define risk_free_rate and volatility

        # Create a response
        allocation_details = {
            "Equities": f"{equity_weight * 100:.2f}% (Expected Return: {expected_returns.get('equities', 0) * 100:.2f}%)",
            "Bonds": f"{bond_weight * 100:.2f}% (Expected Return: {expected_returns.get('bonds', 0) * 100:.2f}%)",
            "Cash": f"{cash_weight * 100:.2f}% (Expected Return: {expected_returns.get('cash', 0) * 100:.2f}%)",
            "Blue-Chip Stocks": f"{blue_chip_weight * 100:.2f}% (Expected Return: {expected_returns.get('blue_chip', 0) * 100:.2f}%)",
            "Mid-Cap Stocks": f"{mid_cap_weight * 100:.2f}% (Expected Return: {expected_returns.get('mid_cap', 0) * 100:.2f}%)",
            "Small-Cap Stocks": f"{small_cap_weight * 100:.2f}% (Expected Return: {expected_returns.get('small_cap', 0) * 100:.2f}%)",
            "Total Expected Portfolio Return": f"{portfolio_return_percentage * 100:.2f}%",
            "Expected Return in Absolute Terms": f"${portfolio_return_absolute:.2f}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}"
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
        
        equities_value = equity_weight * 100
        bonds_value = bond_weight * 100
        cash_value = cash_weight * 100
        blue_chip_value = blue_chip_weight * 100
        mid_cap_value = mid_cap_weight * 100
        small_cap_value = small_cap_weight * 100
        total_return_value = portfolio_return_percentage * 100
        absolute_return_value = portfolio_return_absolute
        sharpe_ratio_value = sharpe_ratio

        return render_template(equities_value=equities_value, bonds_value=bonds_value, cash_value=cash_value, blue_chip_value=blue_chip_value, mid_cap_value=mid_cap_value, 
                            small_cap_value=small_cap_value, total_return_value=total_return_value, absolute_return_value=absolute_return_value, sharpe_ratio_value=sharpe_ratio_value, plot_path=plot_path)

