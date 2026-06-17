# LogiTrack Delay Prediction System - Operations Handoff

**To:** VP of Operations, LogiTrack Global Logistics  
**From:** FDE Team  
**Date:** June 2026

## 1. What the Model Predicts

The model uses shipment weight, distance, promised transit days, number of stops, carrier rating, carrier, product type, and customs status to estimate the probability that a shipment will arrive more than two days late. It outputs a delay probability plus a plain-language risk tier: High risk means at least a 60% chance of a delay, Medium risk means 40%-59%, and Low risk means below 40%.

## 2. What Happens Automatically Each Morning

1. At 06:00, the daily workflow starts before the operations shift begins.
2. It pulls the latest shipment file and checks for missing shipment IDs, missing delay flags, and non-positive weights.
3. If the file fails those checks, the workflow stops and sends a plain-language alert to the data-ops Slack channel.
4. If the file passes, the workflow reads the scored shipment notifications for the day.
5. Every High-risk shipment sends three alerts in parallel: Slack to `#ops-alerts`, Gmail to the customer success team, and WhatsApp to the client's logistics contact.
6. Every Medium-risk shipment is logged to the scheduled notifications sheet for next-morning outreach.

## 3. When to Trust the Model and When Not To

1. If customs status changes after scoring, the prediction may be stale because the model scored an older operating picture. The ops team should re-score that shipment before contacting the customer.
2. If a new carrier appears that was not in the training data, the prediction may be unreliable because the model has no historical pattern for that carrier. The ops team should route those shipments for manual review until the model is retrained.

## 4. How to Know It's Working in 30 Days

The 30-day metric should be the share of delayed shipments where the customer was notified before the promised delivery date. Pull it from the shipment system by joining notification timestamps to shipment IDs and comparing them with the promised delivery date. If at least 80% of delayed shipments are warned in advance after 30 days, the workflow is delivering measurable operational value.
