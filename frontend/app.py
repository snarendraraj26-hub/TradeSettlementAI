print("========== NEW FRONTEND APP LOADED ==========")
import streamlit as st
import requests

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Trade Settlement AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Trade Settlement AI Assistant")

st.markdown("---")

# =====================================================
# Sidebar
# =====================================================

menu = st.sidebar.radio(

    "Select Option",

    [

        "Settlement Owner Prediction",

        "Trade Settlement Chat"

    ]

)

# =====================================================
# Prediction Page
# =====================================================

if menu == "Settlement Owner Prediction":

    st.header("Settlement Owner Prediction")

    amount = st.number_input(

        "Amount",

        min_value=1000,

        value=15000

    )

    currency = st.selectbox(

        "Currency",

        [

            "USD",

            "EUR",

            "GBP",

            "INR"

        ]

    )

    product = st.selectbox(

        "Product",

        [

            "Equity",

            "Bond",

            "ETF",

            "Derivative"

        ]

    )

    country = st.selectbox(

        "Country",

        [

            "India",

            "Germany",

            "France",

            "UK",

            "USA"

        ]

    )

    settlement = st.selectbox(

        "Settlement Type",

        [

            "T+1",

            "T+2",

            "T+3"

        ]

    )

    priority = st.selectbox(

        "Priority",

        [

            "High",

            "Medium",

            "Low"

        ]

    )

    if st.button("Predict Owner"):

        data = {

            "amount": amount,

            "currency": currency,

            "product": product,

            "country": country,

            "settlement_type": settlement,

            "priority": priority

        }

        try:

            response = requests.post(

                "http://127.0.0.1:8000/predict",

                json=data

            )

            if response.status_code == 200:

                result = response.json()

                st.success(

                    f"Predicted Settlement Owner : {result['predicted_owner']}"

                )

            else:

                st.error(response.text)

        except Exception as e:

            st.error(f"Cannot connect to FastAPI.\n\n{e}")

# =====================================================
# Chat Page
# =====================================================

else:

    st.header("Trade Settlement Knowledge Assistant")

    question = st.text_area(

        "Ask any trade settlement question"

    )

    if st.button("Ask AI"):

        if question.strip() == "":

            st.warning("Please enter a question.")

        else:

            try:

                response = requests.post(

                    "http://127.0.0.1:8000/chat",

                    json={

                        "question": question

                    }

                )

                if response.status_code == 200:

                    answer = response.json()["answer"]

                    st.markdown("### 🤖 AI Response")

                    st.write(answer)

                else:

                    st.error(response.text)

            except Exception as e:

                st.error(f"Cannot connect to FastAPI.\n\n{e}")