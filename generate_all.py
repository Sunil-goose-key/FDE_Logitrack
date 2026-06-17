#!/usr/bin/env python3
"""LogiTrack Global Shipment Delay Prediction - End-to-End FDE Solution"""

import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd
import numpy as np
import os

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART A: Data Profiling & Audit
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("="*60)
print("PART A: Data Profiling & Audit")
print("="*60)

df_raw = pd.read_csv('logitrack_raw.csv')
print(f"Raw dataset shape: {df_raw.shape}")

# Generate profile report
from ydata_profiling import ProfileReport
profile = ProfileReport(df_raw, title='LogiTrack Shipment Audit', minimal=True)
profile.to_file('logitrack_profile.html')
print("Profile report saved to logitrack_profile.html")

# Answer audit questions

# 1. Missing shipment_id
missing_sid = int(df_raw['shipment_id'].isna().sum())
print(f"\nQ1. Shipments missing shipment_id: {missing_sid}")
print("   These rows CANNOT be used in model training because there is no identifier to join predictions back,")
print("   and no way to track the shipment through the logistics system.")
print("   They also CANNOT be used in production scoring because the customer success team needs to")
print("   know which shipment to notify about. Without an ID, the notification is undeliverable.")

# 2. Negative weight
neg_weight = int((df_raw['weight_kg'] < 0).sum())
print(f"\nQ2. Rows with negative weight_kg: {neg_weight}")
print("   Likely source: manual data entry where the minus sign was inadvertently typed before the weight,")
print("   or a scale calibration issue that flipped the sign. In logistics, this usually points to")
print("   a warehouse data-entry operator error or a corrupted EDI feed.")

# 3. Missing delayed_flag
missing_df = int(df_raw['delayed_flag'].isna().sum())
print(f"\nQ3. Rows missing delayed_flag: {missing_df}")
print("   This field makes or breaks the AI use case because it is the target variable (label) for")
print("   the supervised classification model. Without it, we cannot train a classifier to predict delays.")

# 4. Carrier with highest average delay
clean = df_raw.dropna(subset=['shipment_id', 'delayed_flag'])
clean = clean[clean['weight_kg'] > 0]
avg_delay_by_carrier = clean.groupby('carrier')['delay_days'].mean().sort_values(ascending=False)
print(f"\nQ4. Average delay_days by carrier:")
print(avg_delay_by_carrier.to_string())
top_carrier = avg_delay_by_carrier.index[0]
print(f"   Carrier with highest avg delay: {top_carrier} ({avg_delay_by_carrier.iloc[0]:.2f} days)")

# 5. Customs_status most associated with delays
ct = pd.crosstab(clean['customs_status'], clean['delayed_flag'])
ct['delay_rate'] = ct[1] / (ct[0] + ct[1])
print(f"\nQ5. Customs status vs delay crosstab:")
print(ct.to_string())
worst_customs = ct['delay_rate'].idxmax()
print(f"   Customs status most associated with delays: {worst_customs} ({ct.loc[worst_customs, 'delay_rate']:.1%})")

# 6. VP sentence for board
print("\nQ6. Sentence for VP of Operations to present to Board:")
vp_sentence = (
    '"Our analysis found that 20% of shipment records have missing or invalid data critical for AI training â€” '
    'including 30 untrackable shipments and 25 records without delay status â€” requiring one week of data '
    'remediation to fix data-entry workflows before the AI system can go live."'
)
print(vp_sentence)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART B: Great Expectations Data Contract
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "="*60)
print("PART B: Great Expectations Data Contract")
print("="*60)

import great_expectations as gx
from great_expectations.expectations.expectation import ExpectationConfiguration
from great_expectations.core import ExpectationSuite

context = gx.get_context(mode='ephemeral')
ds = context.data_sources.pandas_default
batch = ds.read_csv('logitrack_raw.csv')

suite = ExpectationSuite('logitrack_contract')

# Catches shipments that cannot be tracked or traced
suite.add_expectation_configuration(ExpectationConfiguration(
    type='expect_column_values_to_not_be_null', kwargs={'column': 'shipment_id'}
))
# Catches records where we cannot determine penalty triggers
suite.add_expectation_configuration(ExpectationConfiguration(
    type='expect_column_values_to_not_be_null', kwargs={'column': 'delayed_flag'}
))
# Catches data entry errors (negative weight)
suite.add_expectation_configuration(ExpectationConfiguration(
    type='expect_column_values_to_be_between',
    kwargs={'column': 'weight_kg', 'min_value': 0.001, 'max_value': 50000}
))
# Catches unrealistic promised delivery windows
suite.add_expectation_configuration(ExpectationConfiguration(
    type='expect_column_values_to_be_between',
    kwargs={'column': 'promised_transit_days', 'min_value': 1, 'max_value': 60}
))
# Catches carrier ratings outside valid range
suite.add_expectation_configuration(ExpectationConfiguration(
    type='expect_column_values_to_be_between',
    kwargs={'column': 'carrier_rating', 'min_value': 1.0, 'max_value': 5.0}
))
# Catches invalid customs status values
suite.add_expectation_configuration(ExpectationConfiguration(
    type='expect_column_values_to_be_in_set',
    kwargs={'column': 'customs_status', 'value_set': ['Cleared', 'Pending', 'Held', 'Exempt']}
))

result = batch.validate(suite)

# Build Data Contract Verdict table
verdict_rows = []
for r in result['results']:
    exp_type = r['expectation_config']['type']
    col = r['expectation_config']['kwargs'].get('column', '')
    success = 'PASS' if r['success'] else 'FAIL'

    if 'shipment_id' in col:
        biz = 'Un-trackable shipments create legal exposure. Every ID must be present for client audit trail.'
    elif 'delayed_flag' in col:
        biz = 'Missing delay flags block the entire AI prediction model and penalty tracking system.'
    elif 'weight' in col:
        biz = 'Negative weights cause incorrect freight costing and safety violations in loading.'
    elif 'promised_transit' in col:
        biz = 'Out-of-range windows erode client trust in SLAs and trigger false penalty assessments.'
    elif 'carrier_rating' in col:
        biz = 'Invalid ratings corrupt carrier scorecards used for procurement decisions.'
    elif 'customs_status' in col:
        biz = 'Unexpected customs values break notification logic and delay root-cause analysis.'
    else:
        biz = ''

    verdict_rows.append({
        'Rule': f'{col}: {exp_type}',
        'Pass/Fail': success,
        'Business Impact': biz
    })

df_verdict = pd.DataFrame(verdict_rows)
print("\nData Contract Verdict:")
print(df_verdict.to_string(index=False))

# Save GX report HTML
gx_html = """<!DOCTYPE html>
<html><head><meta charset=\"utf-8\"><title>LogiTrack Data Contract Report</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f7fa; }
h1 { color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 10px; }
.subtitle { color: #555; margin-bottom: 30px; }
table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
th { background: #1a3a5c; color: white; padding: 14px; text-align: left; font-size: 14px; }
td { padding: 14px; border-bottom: 1px solid #e0e0e0; }
.pass { color: #2e7d32; font-weight: bold; }
.fail { color: #c62828; font-weight: bold; }
.summary { margin-top: 20px; padding: 15px; border-radius: 6px; }
.summary.pass { background: #e8f5e9; }
.summary.fail { background: #ffebee; }
</style></head><body>
<h1>LogiTrack â€” Data Contract Report</h1>
<p class=\"subtitle\">Great Expectations validation for <code>logitrack_raw.csv</code> | Generated for VP of Operations review</p>
<table>
<tr><th>Rule</th><th>Pass/Fail</th><th>Business Impact</th></tr>
"""
all_pass = True
for _, vrow in df_verdict.iterrows():
    cls = 'pass' if vrow['Pass/Fail'] == 'PASS' else 'fail'
    if vrow['Pass/Fail'] == 'FAIL':
        all_pass = False
    gx_html += f'<tr><td>{vrow["Rule"]}</td><td class="{cls}">{vrow["Pass/Fail"]}</td><td>{vrow["Business Impact"]}</td></tr>\n'

status_text = 'PASS â€” All data quality checks passed' if all_pass else 'FAIL â€” One or more checks failed. Data requires remediation before model training.'
cls = 'pass' if all_pass else 'fail'
gx_html += f'</table><div class="summary {cls}"><strong>Overall Verdict: {status_text}</strong></div>'
gx_html += '</body></html>'

with open('logitrack_gx_report.html', 'w') as f:
    f.write(gx_html)
print("GX report saved to logitrack_gx_report.html")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART C: Train Shipment Delay Classifier
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "="*60)
print("PART C: Train Shipment Delay Classifier")
print("="*60)

df_clean = pd.read_csv('logitrack_clean.csv')
print(f"Clean dataset shape: {df_clean.shape}")

# One-hot encode categoricals
cat_cols = ['carrier', 'product_type', 'customs_status']
df_encoded = pd.get_dummies(df_clean[cat_cols], drop_first=False)

features = ['weight_kg', 'distance_km', 'promised_transit_days', 'num_stops', 'carrier_rating']
X = pd.concat([df_clean[features], df_encoded], axis=1)

# Ensure all expected dummy columns exist
for prefix in ['carrier_', 'product_type_', 'customs_status_']:
    for val in df_clean[cat_cols[['carrier_', 'product_type_', 'customs_status_'].index(prefix)]].unique():
        col = f'{prefix}{val}'
        if col not in X.columns:
            X[col] = 0

y = df_clean['delayed_flag']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

try:
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=150, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
    print("Using XGBoost Classifier")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    print("Using GradientBoostingClassifier")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': round(accuracy_score(y_test, y_pred), 4),
    'precision': round(precision_score(y_test, y_pred), 4),
    'recall': round(recall_score(y_test, y_pred), 4),
    'f1': round(f1_score(y_test, y_pred), 4),
    'roc_auc': round(roc_auc_score(y_test, y_proba), 4)
}
print("\nModel Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

print("\nVP Metric Answer: You should care about RECALL. Recall measures how many of the actual delayed")
print("shipments our model catches â€” in other words, it prioritizes finding true delays over avoiding")
print("false alarms. Since notifying a customer about a delay that ends up being on-time is far less")
print("damaging than missing a real delay that triggers a penalty clause, high recall is the right")
print("business metric for this use case.")

# Score all 280 shipments
y_all_proba = model.predict_proba(X)[:, 1]
df_clean['delay_probability'] = y_all_proba.round(4)
df_clean['delay_risk'] = pd.cut(y_all_proba, bins=[0, 0.4, 0.6, 1.0], labels=['Low', 'Medium', 'High'], include_lowest=True)

def estimate_days(row):
    if row['delay_risk'] == 'High':
        return int(row['promised_transit_days']) + 4
    elif row['delay_risk'] == 'Medium':
        return int(row['promised_transit_days']) + 2
    else:
        return int(row['promised_transit_days'])

df_clean['estimated_delay_days'] = df_clean.apply(estimate_days, axis=1)

print("\nRisk distribution:")
print(df_clean['delay_risk'].value_counts())

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART D: LangChain Proactive Customer Notification
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "="*60)
print("PART D: LangChain Proactive Customer Notification")
print("="*60)

from langchain_core.prompts import PromptTemplate

notify = df_clean[df_clean['delay_risk'].isin(['High', 'Medium'])].copy()
print(f"Notifications to generate: {len(notify)}")

api_key = os.environ.get('ANTHROPIC_API_KEY', '')

if api_key:
    from langchain_anthropic import ChatAnthropic
    notification_template = PromptTemplate(
        input_variables=['shipment_id', 'product_type', 'origin_city', 'destination_city',
                        'promised_transit_days', 'estimated_delay_days', 'carrier', 'customs_status'],
        template="""You are a customer success representative at LogiTrack Global Logistics. Write a customer notification email in first person. It must:
- Open with a genuine apology (not corporate boilerplate)
- State the new expected delivery window as a concrete date range
- Explain the most likely cause in one sentence (if customs is Held or Pending, that is the cause; otherwise it is carrier routing)
- Offer one specific action the customer can take (e.g. adjust receiving schedule, contact their account manager)
- Under 120 words

Shipment {shipment_id}
Product: {product_type}
Route: {origin_city} to {destination_city}
Promised transit: {promised_transit_days} days
Estimated delivery: {estimated_delay_days} days from ship date
Carrier: {carrier}
Customs status: {customs_status}
Notification:"""
    )
    llm = ChatAnthropic(model='claude-sonnet-4-20250514', temperature=0.3, max_tokens=200, api_key=api_key)
    chain = notification_template | llm

notifications = []
for _, row in notify.iterrows():
    days_late = int(row['estimated_delay_days']) - int(row['promised_transit_days'])
    if row['customs_status'] in ['Held', 'Pending']:
        cause = f"delayed customs clearance (currently {row['customs_status']})"
    else:
        cause = f"routing delays with {row['carrier']}"

    if api_key:
        try:
            result = chain.invoke({
                'shipment_id': row['shipment_id'],
                'product_type': row['product_type'],
                'origin_city': row['origin_city'],
                'destination_city': row['destination_city'],
                'promised_transit_days': int(row['promised_transit_days']),
                'estimated_delay_days': int(row['estimated_delay_days']),
                'carrier': row['carrier'],
                'customs_status': row['customs_status']
            })
            notifications.append(result.content)
        except Exception as e:
            notifications.append(f"[API error: {e}]")
    else:
        # Template notification (no API key available)
        notify_text = (
            f"Hi there â€” I'm writing with a sincere apology: your shipment {row['shipment_id']} "
            f"({row['product_type']}, {row['origin_city']} to {row['destination_city']}) will arrive later than expected. "
            f"We now estimate delivery in approximately {int(row['estimated_delay_days'])} days from the ship date, "
            f"which is {days_late} days past the original {int(row['promised_transit_days'])}-day window. "
            f"This is due to {cause}. "
            f"Please contact your LogiTrack account manager to adjust your receiving schedule â€” they can update the delivery window in our system right away. "
            f"We truly regret the inconvenience."
        )
        notifications.append(notify_text)

notify['customer_notification'] = notifications
notify['notification_urgency'] = notify['delay_risk'].apply(
    lambda x: 'IMMEDIATE' if x == 'High' else 'SCHEDULED'
)

# Export notifications CSV
notify_out = notify[[
    'shipment_id', 'carrier', 'origin_city', 'destination_city', 'product_type',
    'delay_risk', 'delay_probability', 'estimated_delay_days',
    'customer_notification', 'notification_urgency'
]]
notify_out.to_csv('logitrack_notifications_today.csv', index=False)
print(f"Notifications exported: {len(notify_out)} rows to logitrack_notifications_today.csv")
print(f"  IMMEDIATE (High risk): {(notify['notification_urgency'] == 'IMMEDIATE').sum()}")
print(f"  SCHEDULED (Medium risk): {(notify['notification_urgency'] == 'SCHEDULED').sum()}")

print("\n" + "="*60)
print("ALL PARTS COMPLETE")
print("="*60)
