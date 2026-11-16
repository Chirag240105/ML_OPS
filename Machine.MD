## what we are doing 
building production grade real-time fraud detection MLOPs Pipeline named RiskStream using PaySim financial transactions dataset 




## Maching Learning in Model -: Dataset URL: https://www.kaggle.com/datasets/ealaxi/paysim1

how I approach first loading data set and know about it 
1. Understanding Data 

final report : 
{
    "rows": 6362620,
    "columns": 11,
    "column_names": [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud"
    ],
    "missing_values": {
        "step": 0,
        "type": 0,
        "amount": 0,
        "nameOrig": 0,
        "oldbalanceOrg": 0,
        "newbalanceOrig": 0,
        "nameDest": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
        "isFraud": 0,
        "isFlaggedFraud": 0
    },
    "duplicate_rows": 0,
    "data_types": {
        "step": "int64",
        "type": "object",
        "amount": "float64",
        "nameOrig": "object",
        "oldbalanceOrg": "float64",
        "newbalanceOrig": "float64",
        "nameDest": "object",
        "oldbalanceDest": "float64",
        "newbalanceDest": "float64",
        "isFraud": "int64",
        "isFlaggedFraud": "int64"
    },
    "Fraud_data": {
        "0": 6354407,
        "1": 8213
    }
}



2. Exploratory Data Analysis (EDA)

What have we learned?
Observation 1: TRANSFER has the highest fraud rate.

Approximately 0.77% of transfers are fraudulent.

Observation 2: CASH_OUT has the highest number of fraud cases.

It accounts for 4,116 fraudulent transactions, slightly more than TRANSFER.

Observation 3: The dataset contains no fraud examples for CASH_IN, DEBIT or PAYMENT.

This is particularly interesting.

A model could learn that certain transaction types are strongly associated with fraud in this dataset.

However, don't assume those transaction types can never be fraudulent in real financial systems. This is a property of the PaySim simulation and its fraud-generation mechanism.
Our decision
Keep type

We will retain transaction type as a categorical feature.

We should also report model performance separately across transaction types, so we know whether the model is simply exploiting this dataset pattern.
2. Daily transaction analysis
Your daily graph shows something unusual.
- Day 0 and Day 1 have extremely high transaction volumes.
- Days 2–4 have much lower volumes.
- Later days have irregular transaction counts.
- The last day shows a 100% fraud rate.
The last point is particularly important.

A 100% fraud rate does not necessarily mean a major fraud attack occurred. It could mean that the final group contains very few transactions.

This is a denominator problem.
99.45% of fraudulent transactions have an exact sender balance match (orig_error ≈ 0).



Only 20.27% of legitimate transactions have the same match.



All 272 transactions in the final simulation period (steps 721–743) are fraudulent.

3. Data Processing 
 a. Feature Selection -> b. feature Engineering -> c. Data validation -> d. Splitting  -> e. Class Imbalance strategy -> f. Save processed artifact


a. 

 Feature	Decision	Reason
step	Keep	Transaction timing
type	Keep	Transaction category
amount	Keep	Transaction amount
oldbalanceOrg	Keep	Sender's pre-transaction balance
oldbalanceDest	Keep provisionally	Recipient's pre-transaction balance
nameOrig	Exclude	High-cardinality account ID
nameDest	Exclude	High-cardinality account ID
newbalanceOrig	Exclude	Post-transaction information
newbalanceDest	Exclude	Post-transaction information
isFlaggedFraud	Exclude	Existing detection-system output
isFraud	Target	What we predict

