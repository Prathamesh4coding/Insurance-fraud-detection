# Healthcare Fraud Detection 🏥

## What is this?
Insurance fraud costs the healthcare industry billions every year. 
This project builds a machine learning model to flag suspicious claims 
before they get paid using only 4 features and achieving 96% precision.

## What I found before even building the model
Before touching any ML code I explored the data manually. Two things stood out:

**Fraudsters file claims faster.**
Legitimate claims take ~15 days between service and filing. 
Fraudulent ones take ~3 days. They're not dealing with paperwork  
they're fabricating records and want payment quickly.

**Fraudsters overbill significantly.**
Average legitimate claim: 533 INR. Average fraudulent claim: 981 dollars. 
Nearly double. But they only get approved for slightly more than 
legitimate claimants the overbilling gets caught at approval stage 
but some still slip through.

These two observations alone predicted what the model later confirmed.

## Model Results
| Metric | Score |
|---|---|
| Accuracy | 98.4% |
| Precision | 96.4% |
| Recall | 83.7% |
| F1 Score | 89.6% |
| AUC | 99.7% |

I prioritised Recall over Accuracy intentionally. In fraud detection, 
missing a fraud case costs money. A false alarm just means extra review. 
So I used class_weight='balanced' instead of artificially inflating 
the dataset with synthetic data — which could introduce patterns that don't exist.

## What the model actually means in business terms
- Dataset had 630 fraud cases out of 7,599 claims (~8% fraud rate)
- Model catches 527 of those 630 fraud cases
- Average fraudulent claim: 981 INR
- Estimated fraud caught per cycle: ~516,987 $
- Missed fraud (16%): ~103 cases = ~101,043 $ still slipping through

## Feature Importance
The model confirmed what manual analysis predicted:

| Feature | Importance |
|---|---|
| Days Between Service and Claim | 63% |
| Claim Amount | 19% |
| Approved Amount | 14% |
| Claims Per Provider Monthly | 4% |

## Why I excluded Self-Pay patients
Self-Pay patients don't go through an insurance company 
they literally cannot commit insurance fraud. Including them 
would add noise without signal. I verified this first: 
their fraud rate (8.3%) was identical to the overall rate, 
confirming they carry no useful fraud signal.

## Limitations
This is a post-payment model — it identifies fraud after the claim 
is processed. In healthcare where claims settle in days not months, 
a real deployment would need pre-payment scoring at submission stage 
using features available at the moment of filing.

The next version of this project applies the same logic to 
general insurance (motor claims) where the longer settlement 
tail gives the model time to flag fraud before payment.

## Tech Stack
- Python, Pandas, Scikit-learn, Matplotlib
- Random Forest Classifier
- No SMOTE — class weights used instead
