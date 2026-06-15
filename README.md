\---

title: Customer Support Ticket Classifier

colorFrom: blue

colorTo: indigo

sdk: gradio

sdk\_version: 4.44.0

app\_file: app.py

pinned: false

license: mit

\---



\# Customer Support Ticket Classifier



Hierarchical multi-label classifier for customer-support tickets. Predicts three label levels using a cascaded classifier:



\- \*\*Type 2\*\*: Suggestion / Problem-Fault / Others

\- \*\*Type 3\*\*: Payment, Refund, AppGallery-Install/Upgrade, etc.

\- \*\*Type 4\*\*: Subscription cancellation, Can't install Apps, etc.



\## Evaluation on the held-out test set (42 rows)



| Metric              | Random Forest | Logistic Regression |

|---------------------|---------------|---------------------|

| Chained accuracy    | 0.683         | \*\*0.694\*\*           |

| Type 2 macro F1     | \*\*0.833\*\*     | 0.769               |

| Type 3 macro F1     | 0.553         | 0.583               |

| Type 4 macro F1     | 0.537         | 0.448               |



\## Limitations



\- Trained on 206 samples. Predictions on out-of-domain text will be unreliable.



