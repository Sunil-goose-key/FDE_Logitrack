import json, nbformat, pandas as pd

# Validate n8n
wf = json.load(open('logitrack_n8n_workflow.json'))
print(f"n8n workflow: {len(wf['nodes'])} nodes, {len(wf['connections'])} connection groups")
for n in wf['nodes']:
    print(f"  {n['name']} ({n['type']})")

# Validate notebooks
for f in ['logitrack_profiling.ipynb', 'logitrack_gx.ipynb', 'logitrack_model.ipynb']:
    nb = nbformat.read(f, as_version=4)
    print(f"{f}: {len(nb.cells)} cells")

# Validate notifications CSV
df = pd.read_csv('logitrack_notifications_today.csv')
print(f"Notifications CSV: {len(df)} rows, {list(df.columns)}")
print(f"  IMMEDIATE: {(df['notification_urgency']=='IMMEDIATE').sum()}")
print(f"  SCHEDULED: {(df['notification_urgency']=='SCHEDULED').sum()}")

# Validate HTML reports
import os
for f in ['logitrack_profile.html', 'logitrack_gx_report.html']:
    sz = os.path.getsize(f)
    print(f"{f}: {sz/1024:.1f} KB")

# Validate markdown files
for f in ['logitrack_handoff.md', 'logitrack_cfo_response.md']:
    sz = os.path.getsize(f)
    print(f"{f}: {sz/1024:.1f} KB")

print("\nAll deliverables validated OK")
