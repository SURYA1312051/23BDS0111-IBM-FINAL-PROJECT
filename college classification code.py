import pandas as pd
from transformers import pipeline

# Step 1: Load your feedback CSV file
df = pd.read_csv("college_feedback.csv")  


# Step 4: Few-shot examples with explicit label instruction
few_shot_prompt = """Classify the feedback into one of the following categories: Facilities, Academics, Administration.

Feedback: The classrooms are clean and well-maintained.
Category: Facilities

Feedback: The professors explain concepts very clearly.
Category: Academics

Feedback: There is a lot of delay in processing scholarship applications.
Category: Administration

Feedback: {input}
Category:"""

# Step 2: Load FLAN-T5 model for text2text classification
classifier = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=10)

# Step 5: Function to classify each feedback
VALID_LABELS = ["Facilities", "Academics", "Administration"]

def clean_prediction(prediction):
    for label in VALID_LABELS:
        if label.lower() in prediction.lower():
            return label
    return "Unknown"

# Classify each feedback
predicted_categories = []
for feedback in df["feedback"]:
    prompt = few_shot_prompt.format(input=feedback)
    raw_output = classifier(prompt)[0]["generated_text"].strip()
    cleaned = clean_prediction(raw_output)
    predicted_categories.append(cleaned)

# Step 6: Save results
df["predicted_category"] = predicted_categories

#step 7: export data
df.to_csv("classified_feedback_output.csv", index=False)
