
import json
from google.adk import Agent


def get_loan_status(loan_id: str, amount: float) -> dict:
    """Returns loan status as structured JSON."""
    print(f"DEBUG: get_loan_status called with loan_id={loan_id}, amount={amount}, type={type(amount)}")
    
    try:
        amount_value = float(amount)
        print(f"DEBUG: Converted amount to {amount_value}")
    except (TypeError, ValueError) as e:
        print(f"DEBUG: Error converting amount: {e}")
        return {
            "type": "loan_status_response",
            "loan_id": loan_id,
            "amount": None,
            "approved": False,
            "error": "invalid_amount",
            "message": "Invalid amount provided. Please supply a numeric amount."
        }

    if amount_value > 10000:
        result = {
            "type": "loan_status_response",
            "loan_id": loan_id,
            "amount": amount_value,
            "approved": False,
            "reason": "exceeds_limit",
            "message": f"Loan amount {amount_value:.0f} exceeds the limit of 10000."
        }
    else:
        result = {
            "type": "loan_status_response",
            "loan_id": loan_id,
            "amount": amount_value,
            "approved": True,
            "message": f"Loan approved for amount {amount_value:.0f}."
        }
    
    print(f"DEBUG: Returning result: {result}")
    return result


def create_agent() -> Agent:
    """Constructs the ADK agent for Loan."""
    return Agent(
        model="gemini-2.5-flash",
        name="Loan_Agent",
        instruction="""
            You are a loan agent that ONLY responds with JSON format.
            
            CRITICAL RULES:
            1. ALWAYS respond with valid JSON only - never plain text
            2. When you receive any request, call the get_loan_status tool with the loan_id and amount
            3. Return the exact JSON result from get_loan_status - do not modify it
            4. Never add explanations, greetings, or any text outside the JSON
            5. If you cannot process a request, return: {"error": "invalid_request", "message": "Unable to process request"}
            
            Example response format:
            {"type": "loan_status_response", "loan_id": "12345", "amount": 10000, "approved": true, "message": "Loan approved for amount 10000."}
        """,
        tools=[get_loan_status],
    )
