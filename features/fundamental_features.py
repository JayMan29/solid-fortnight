"""
Growth, quality, financial-strength, and valuation features built from
SEC XBRL concepts. All lookups respect point-in-time filing dates via
`data.fundamentals.extract_quarterly_metric(..., as_of_date=...)`.
"""
from __future__ import annotations
from datetime import date
import numpy as np
import pandas as pd
from data.fundamentals import extract_quarterly_metric


def _latest_value(df: pd.DataFrame) -> float:
    return float(df.iloc[-1]["val"]) if not df.empty else np.nan


def _yoy_growth(df: pd.DataFrame) -> float:
    if len(df) < 5:
        return np.nan
    current = df.iloc[-1]["val"]
    year_ago = df.iloc[-5]["val"]
    if year_ago in (0, None) or pd.isna(year_ago):
        return np.nan
    return float(current / year_ago - 1.0)


def calculate_fundamental_features(
    company_facts: dict,
    as_of_date: date,
    market_cap: float | None = None,
) -> dict[str, float]:
    revenue = extract_quarterly_metric(company_facts, "Revenues", as_of_date=as_of_date)
    if revenue.empty:
        # Some filers use RevenueFromContractWithCustomerExcludingAssessedTax instead
        revenue = extract_quarterly_metric(
            company_facts, "RevenueFromContractWithCustomerExcludingAssessedTax", as_of_date=as_of_date
        )
    gross_profit = extract_quarterly_metric(company_facts, "GrossProfit", as_of_date=as_of_date)
    op_income = extract_quarterly_metric(company_facts, "OperatingIncomeLoss", as_of_date=as_of_date)
    fcf_proxy = extract_quarterly_metric(
        company_facts, "NetCashProvidedByUsedInOperatingActivities", as_of_date=as_of_date
    )
    total_debt = extract_quarterly_metric(company_facts, "LongTermDebtNoncurrent", as_of_date=as_of_date)
    cash = extract_quarterly_metric(company_facts, "CashAndCashEquivalentsAtCarryingValue", as_of_date=as_of_date)
    rnd = extract_quarterly_metric(company_facts, "ResearchAndDevelopmentExpense", as_of_date=as_of_date)
    shares = extract_quarterly_metric(company_facts, "CommonStockSharesOutstanding", as_of_date=as_of_date)

    latest_revenue = _latest_value(revenue)
    latest_gross_profit = _latest_value(gross_profit)
    latest_op_income = _latest_value(op_income)
    latest_fcf = _latest_value(fcf_proxy)
    latest_debt = _latest_value(total_debt)
    latest_cash = _latest_value(cash)

    gross_margin = (
        latest_gross_profit / latest_revenue
        if latest_revenue not in (0, None) and not pd.isna(latest_revenue) and not pd.isna(latest_gross_profit)
        else np.nan
    )
    operating_margin = (
        latest_op_income / latest_revenue
        if latest_revenue not in (0, None) and not pd.isna(latest_revenue) and not pd.isna(latest_op_income)
        else np.nan
    )
    fcf_margin = (
        latest_fcf / latest_revenue
        if latest_revenue not in (0, None) and not pd.isna(latest_revenue) and not pd.isna(latest_fcf)
        else np.nan
    )
    net_debt = (
        latest_debt - latest_cash
        if not pd.isna(latest_debt) and not pd.isna(latest_cash)
        else np.nan
    )

    share_dilution_yoy = _yoy_growth(shares) if not shares.empty else np.nan
    rnd_growth_yoy = _yoy_growth(rnd) if not rnd.empty else np.nan

    ev_to_revenue = (
        (market_cap + (net_debt if not pd.isna(net_debt) else 0)) / latest_revenue
        if market_cap and latest_revenue not in (0, None) and not pd.isna(latest_revenue)
        else np.nan
    )

    return {
        "revenue_growth_yoy": _yoy_growth(revenue),
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "free_cash_flow_margin": fcf_margin,
        "rnd_growth_yoy": rnd_growth_yoy,
        "share_dilution_yoy": share_dilution_yoy,
        "net_debt": net_debt,
        "ev_to_revenue": ev_to_revenue,
    }
