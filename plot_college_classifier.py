import pandas as pd
import matplotlib.pyplot as plt
# Step 1: Load the classified CSV
df = pd.read_csv("college_feedback_output.csv")
# Step 2: Count actual and predicted values
actual_counts = df["category"].value_counts()
predicted_counts = df["predicted_category"].value_counts()
# Ensure all categories are included
all_categories = sorted(set(actual_counts.index).union(set(predicted_counts.index)))
actual = [actual_counts.get(cat, 0) for cat in all_categories]
predicted = [predicted_counts.get(cat, 0) for cat in all_categories]
# Step 3: Plot
x = range(len(all_categories))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x, actual, width=width, label="Actual", align="center")
plt.bar([i + width for i in x], predicted, width=width, label="Predicted", align="center")
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Actual vs Predicted Feedback Categories")
plt.xticks([i + width / 2 for i in x], all_categories)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
