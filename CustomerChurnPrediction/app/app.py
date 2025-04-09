#Flask App can be run from from the terminal with: "python app.py"
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
lda = joblib.load('lda.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    CONTRACT_CATEGORIES = ["Monthly", "Quarterly"]
    SUBSCRIPTION_CATEGORIES = ['Premium', 'Standard']
    num_fields = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    
    for field in num_fields:
        data[field] = float(data[field])
   
    data['Gender_Male'] = bool(data['Gender_Male'])

    subscription_type = data.pop('Subscription Type')
    one_hot_subscription = {f"Subscription Type_{sub}": subscription_type == sub for sub in SUBSCRIPTION_CATEGORIES}
    data.update(one_hot_subscription)
    
    contract_length = data.pop('Contract Length')
    one_hot_contract = {f"Contract Length_{cat}": contract_length == cat for cat in CONTRACT_CATEGORIES}
    data.update(one_hot_contract)
    
    input_data = pd.DataFrame([data])

    # #Debugging purpose
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)
    # print(input_data)
    
    input_data = scaler.transform(input_data)
    input_data = lda.transform(input_data)
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    result = {
        'prediction': 'Churn' if prediction[0] == 1 else 'Not Churn',
        'probability': f'{probability * 100:.2f}%' 
    }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)