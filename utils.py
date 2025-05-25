def ask_user_profile():
    """Ask the user about their investment preferences."""
    print("📋 Tell me about your investment profile:\n")

    # 1. Investment Horizon
    print("1) Investment Horizon")
    print("   (1) Short-term (up to 1 year)")
    print("   (2) Long-term (1+ years)")
    horizon = input("   Choose 1 or 2: ").strip()

    # 2. Risk Tolerance
    print("\n2) Risk Tolerance")
    print("   (1) Low  (focus on capital preservation)")
    print("   (2) Medium")
    print("   (3) High (comfortable with big swings)")
    risk = input("   Choose 1, 2, or 3: ").strip()

    # 3. Income Needs
    print("\n3) Do you need dividend income?")
    print("   (y) Yes")
    print("   (n) No")
    income = input("   y or n: ").strip().lower()

    return {
        "horizon": "short" if horizon == "1" else "long",
        "risk":     {"1":"low","2":"medium","3":"high"}.get(risk, "medium"),
        "income":   income == "y"
    }
