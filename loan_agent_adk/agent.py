
from google.adk import Agent


def get_loan_status(loan_id: str, amount: float) -> str:
    try:
        amount_value = float(amount)
    except (TypeError, ValueError):
        return "Invalid amount provided. Please supply a numeric amount (e.g., 10000)."

    if amount_value > 10000:
        return (
            f"Loan ID {loan_id}: Not approved for amount {amount_value:.0f} "
            "because it exceeds the limit of 10000."
        )
    return f"Loan ID {loan_id}: Approved for amount {amount_value:.0f}."

    


def create_agent() -> Agent:
    """Constructs the ADK agent for Loan."""
    return Agent(
        model="gemini-2.5-flash",
        name="Loan_Agent",
        instruction="""
            You are a loan agent. You are responsible for helping the user with their loan application. You will use the `get_loan_status` tool to check the status of the loan. The tool requires a `loan_id` and amount.Always return a final one-sentence decision.
        """,
        tools=[get_loan_status],
    )
