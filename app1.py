import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# -----------------------------
# Load and prepare training data
# -----------------------------
df = pd.read_csv("expenses.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s₹]', '', text)
    return text

df['Clean_Text'] = df['Text'].apply(clean_text)

# -----------------------------
# Train ML model
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Clean_Text'])
y = df['Category']
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Helper functions
# -----------------------------
def predict_category(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

def extract_amount(text):
    match = re.search(r'₹\s?(\d+(?:,\d{3})*)', text)
    if match:
        return int(match.group(1).replace(',', ''))
    return 0

# -----------------------------
# Streamlit App
# -----------------------------
st.title("💸 Smart Expense Categorizer")

# Store expense history using session_state
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Enter your SMS or Transaction Text:")

if st.button("Categorize"):
    category = predict_category(user_input)
    amount = extract_amount(user_input)

    # Show results
    st.success(f"Category: {category}")
    st.info(f"Amount: ₹{amount}")

    # Alert if spending is high
    if category == "Food" and amount > 2000:
        st.warning("⚠️ High Food Spending!")
    elif category == "Rent" and amount > 10000:
        st.warning("⚠️ High Rent Spending!")
    elif category == "Travel" and amount > 3000:
        st.warning("⚠️ High Travel Spending!")

    # Add to session history
    st.session_state.history.append({"Text": user_input, "Category": category, "Amount": amount})

# -----------------------------
# Show updated pie chart
# -----------------------------
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.subheader("📊 Updated Spending Chart")
    chart_data = history_df.groupby("Category")["Amount"].sum()
    fig, ax = plt.subplots()
    chart_data.plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    # Optional: Show full table
    st.subheader("📋 Expense History")
    st.dataframe(history_df)
