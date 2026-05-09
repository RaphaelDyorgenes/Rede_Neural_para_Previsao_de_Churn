# Schema de validação do dataset Telco Churn com pandera.
import pandera.pandas as pa

# Schema para validação do DataFrame de entrada.
# Garante tipos, ranges e valores permitidos antes do pré-processamento.
TelcoChurnSchema = pa.DataFrameSchema(
    {
        "CustomerID": pa.Column(str, nullable=False),
        "Count": pa.Column(int),
        "Country": pa.Column(str),
        "State": pa.Column(str),
        "City": pa.Column(str),
        "Zip Code": pa.Column(int),
        "Lat Long": pa.Column(str),
        "Latitude": pa.Column(float),
        "Longitude": pa.Column(float),
        "Gender": pa.Column(
            str, pa.Check.isin(["Male", "Female"]),
        ),
        "Senior Citizen": pa.Column(
            str, pa.Check.isin(["Yes", "No"]),
        ),
        "Partner": pa.Column(
            str, pa.Check.isin(["Yes", "No"]),
        ),
        "Dependents": pa.Column(
            str, pa.Check.isin(["Yes", "No"]),
        ),
        "Tenure Months": pa.Column(
            int, pa.Check.ge(0),
        ),
        "Phone Service": pa.Column(
            str, pa.Check.isin(["Yes", "No"]),
        ),
        "Multiple Lines": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No phone service"]),
        ),
        "Internet Service": pa.Column(
            str, pa.Check.isin(["DSL", "Fiber optic", "No"]),
        ),
        "Online Security": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Online Backup": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Device Protection": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Tech Support": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Streaming TV": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Streaming Movies": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Contract": pa.Column(
            str, pa.Check.isin(["Month-to-month", "One year", "Two year"]),
        ),
        "Paperless Billing": pa.Column(
            str, pa.Check.isin(["Yes", "No"]),
        ),
        "Payment Method": pa.Column(
            str,
            pa.Check.isin([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]),
        ),
        "Monthly Charges": pa.Column(float, pa.Check.ge(0)),
        "Total Charges": pa.Column(float, pa.Check.ge(0)),
        "Churn Label": pa.Column(
            str, pa.Check.isin(["Yes", "No"]),
        ),
        "Churn Value": pa.Column(int),
        "Churn Score": pa.Column(int, pa.Check.in_range(0, 100)),
        "CLTV": pa.Column(int, pa.Check.ge(0)),
        "Churn Reason": pa.Column(str, nullable=True),
    },
    strict=False,
)
