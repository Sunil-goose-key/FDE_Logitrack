import pandas as pd
import great_expectations as gx

context = gx.get_context(mode='ephemeral')
ds = context.data_sources.pandas_default
validator = ds.read_csv('logitrack_raw.csv')

# Catches shipments that cannot be tracked or traced in the logistics system
validator.expect_column_values_to_not_be_null('shipment_id')

# Catches records where we cannot determine if a penalty clause should be triggered
validator.expect_column_values_to_not_be_null('delayed_flag')

# Catches data entry errors where negative weight was entered instead of positive value
validator.expect_column_values_to_be_between('weight_kg', min_value=0.001, max_value=50000)

# Catches unrealistic promised delivery windows outside logistics norms
validator.expect_column_values_to_be_between('promised_transit_days', min_value=1, max_value=60)

# Catches carrier ratings outside the valid 1.0-5.0 range used by LogiTrack's procurement system
validator.expect_column_values_to_be_between('carrier_rating', min_value=1.0, max_value=5.0)

# Catches invalid customs status values that would break downstream routing logic
validator.expect_column_values_to_be_in_set('customs_status', value_set=['Cleared', 'Pending', 'Held', 'Exempt'])

results = validator.validate()

# Build Data Contract Verdict table
verdict_rows = []
for r in results['results']:
    exp_type = r['expectation_config']['type']
    col = ''
    for k in ['column', 'columns']:
        if k in r['expectation_config']['kwargs']:
            col = r['expectation_config']['kwargs'][k]
            break
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
    
    verdict_rows.append({'Rule': f'{col}: {exp_type}', 'Pass/Fail': success, 'Business Impact': biz})

df_verdict = pd.DataFrame(verdict_rows)
print(df_verdict.to_string(index=False))
print(f"\nTotal validations: {len(results['results'])}")
print(f"All passed: {all(r['success'] for r in results['results'])}")

# Build Data Docs - generate HTML report
gx_html = """
<html><head><style>
body { font-family: Arial; margin: 40px; }
h1 { color: #1a3a5c; }
table { border-collapse: collapse; width: 100%; margin-top: 20px; }
th { background: #1a3a5c; color: white; padding: 10px; text-align: left; }
td { padding: 10px; border-bottom: 1px solid #ddd; }
.pass { color: green; font-weight: bold; }
.fail { color: red; font-weight: bold; }
</style></head><body>
<h1>LogiTrack Data Contract Report</h1>
<p>Great Expectations validation for logitrack_raw.csv</p>
<table>
<tr><th>Rule</th><th>Pass/Fail</th><th>Business Impact</th></tr>
"""
for _, vrow in df_verdict.iterrows():
    cls = 'pass' if vrow['Pass/Fail'] == 'PASS' else 'fail'
    gx_html += f'<tr><td>{vrow["Rule"]}</td><td class="{cls}">{vrow["Pass/Fail"]}</td><td>{vrow["Business Impact"]}</td></tr>\n'
gx_html += "</table></body></html>"

with open('logitrack_gx_report.html', 'w') as f:
    f.write(gx_html)
print("GX report saved to logitrack_gx_report.html")
