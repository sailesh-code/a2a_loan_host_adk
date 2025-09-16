from typing import Dict


def make_loan_application(loan_id: str, amount: str) -> dict:
    """Creates a loan application and returns JSON data for send_message."""
    # Convert amount to float for proper JSON handling
    try:
        amount_float = float(amount)
    except (ValueError, TypeError):
        amount_float = 990000.0  # Default fallback
    
    # Return the JSON payload that send_message will send directly
    json_payload = {
        "type": "loan_status_request",
        "loan_id": loan_id,
        "amount": amount_float
    }
    
    return {
        "json_payload": json_payload,
        "message": f"Loan application prepared for loan_id {loan_id} with amount {amount_float}"
    }
