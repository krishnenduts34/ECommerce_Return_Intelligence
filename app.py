import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Return Intelligence",
    page_icon="📦",
    layout="wide"
)


# --------------------------------------------------
# Load model and prepared data
# --------------------------------------------------

model = joblib.load("model/return_prediction_model.pkl")
df = pd.read_csv("data/prepared_data.csv")


# --------------------------------------------------
# Dashboard title
# --------------------------------------------------

st.title("📦 E-Commerce Return Intelligence")
st.subheader("Customer Return Prediction Dashboard")

st.write(
    "Enter order and customer details to predict whether "
    "the order is likely to be returned."
)

st.success("Machine Learning Model Loaded Successfully!")


# --------------------------------------------------
# Prediction form
# --------------------------------------------------

st.header("🔍 Enter Order Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        product_category = st.selectbox(
            "Product Category",
            sorted(df["Product_Category"].unique())
        )

        product_price = st.number_input(
            "Product Price",
            min_value=0.0,
            value=1000.0
        )

        order_quantity = st.number_input(
            "Order Quantity",
            min_value=1,
            value=1,
            step=1
        )

        discount_applied = st.number_input(
            "Discount Applied",
            min_value=0.0,
            value=0.0
        )

        shipping_method = st.selectbox(
            "Shipping Method",
            sorted(df["Shipping_Method"].unique())
        )

        payment_method = st.selectbox(
            "Payment Method",
            sorted(df["Payment_Method"].unique())
        )

        order_value = st.number_input(
            "Order Value",
            min_value=0.0,
            value=1000.0
        )

    with col2:
        user_age = st.number_input(
            "Customer Age",
            min_value=1,
            max_value=100,
            value=25
        )

        user_gender = st.selectbox(
            "Customer Gender",
            sorted(df["User_Gender"].unique())
        )

        user_location = st.selectbox(
            "Customer Location",
            sorted(df["User_Location"].unique())
        )

        co2_emissions = st.number_input(
            "CO2 Emissions",
            min_value=0.0,
            value=10.0
        )

        packaging_waste = st.number_input(
            "Packaging Waste",
            min_value=0.0,
            value=1.0
        )

        order_year = st.number_input(
            "Order Year",
            min_value=2022,
            max_value=2030,
            value=2025
        )

        order_month = st.number_input(
            "Order Month",
            min_value=1,
            max_value=12,
            value=1
        )

        order_day_of_week = st.number_input(
            "Order Day of Week",
            min_value=0,
            max_value=6,
            value=0
        )

    predict_button = st.form_submit_button(
        "🔮 Predict Return"
    )


# --------------------------------------------------
# Make prediction
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame({
        "Product_Category": [product_category],
        "Product_Price": [product_price],
        "Order_Quantity": [order_quantity],
        "Discount_Applied": [discount_applied],
        "Shipping_Method": [shipping_method],
        "Payment_Method": [payment_method],
        "User_Age": [user_age],
        "User_Gender": [user_gender],
        "User_Location": [user_location],
        "Order_Value": [order_value],
        "CO2_Emissions": [co2_emissions],
        "Packaging_Waste": [packaging_waste],
        "Order_Year": [order_year],
        "Order_Month": [order_month],
        "Order_DayOfWeek": [order_day_of_week]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # --------------------------------------------------
    # Display prediction
    # --------------------------------------------------

    st.header("📊 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Return Predicted")

        st.write(
            f"The model predicts that this order is likely to be returned."
        )

    else:

        st.success("✅ No Return Predicted")

        st.write(
            f"The model predicts that this order is unlikely to be returned."
        )

    st.metric(
        "Return Probability",
        f"{probability * 100:.2f}%"
    )
    # --------------------------------------------------
# --------------------------------------------------
# Bulk CSV Prediction
# --------------------------------------------------

st.header("📁 Bulk Return Prediction")

st.write(
    "Upload a CSV file containing multiple orders "
    "to predict returns for all orders at once."
)

required_columns = [
    "Product_Category",
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "Shipping_Method",
    "Payment_Method",
    "User_Age",
    "User_Gender",
    "User_Location",
    "Order_Value",
    "CO2_Emissions",
    "Packaging_Waste",
    "Order_Year",
    "Order_Month",
    "Order_DayOfWeek"
]

template_df = pd.DataFrame(columns=required_columns)

template_csv = template_df.to_csv(index=False)

st.download_button(
    label="📄 Download CSV Template",
    data=template_csv,
    file_name="bulk_prediction_template.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    bulk_data = pd.read_csv(uploaded_file)

    st.write("Uploaded data:")
    st.dataframe(bulk_data.head())

    if st.button("🔮 Predict All Orders"):

        missing_columns = [
            column
            for column in required_columns
            if column not in bulk_data.columns
        ]

        if missing_columns:

            st.error("❌ Missing required columns:")

            for column in missing_columns:
                st.write(f"- {column}")

        else:

            try:

                prediction_data = bulk_data[required_columns]

                bulk_predictions = model.predict(
                    prediction_data
                )

                bulk_probabilities = model.predict_proba(
                    prediction_data
                )[:, 1]

                bulk_data["Return_Prediction"] = bulk_predictions

                bulk_data["Return_Probability"] = (
                    bulk_probabilities * 100
                ).round(2)

                bulk_data["Return_Prediction"] = bulk_data[
                    "Return_Prediction"
                ].map({
                    0: "No Return",
                    1: "Return"
                })

                st.success(
                    f"Prediction completed for {len(bulk_data)} orders!"
                )

                st.subheader("📊 Prediction Results")

                st.dataframe(bulk_data)

                csv = bulk_data.to_csv(index=False)

                st.download_button(
                    label="📥 Download Prediction Results",
                    data=csv,
                    file_name="return_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    "An error occurred while making predictions."
                )

                st.write("Error:", e)