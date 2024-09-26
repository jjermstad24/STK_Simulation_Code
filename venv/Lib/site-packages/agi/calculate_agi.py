def calculate_agi(gross_income, adjustments):
    """
    Calculate Adjusted Gross Income (AGI).

    Parameters:
    gross_income (float): The total gross income.
    adjustments (float): The total adjustments to income.

    Returns:
    float: The adjusted gross income.
    """
    return gross_income - adjustments
