"""
Ticker <-> company-name alias table.

This matters more than it looks. Searching news for a ticker like "ON"
(ON Semiconductor), "ALL" (Allstate), "CAT" (Caterpillar), or "AI"
(C3.ai) will pull in huge amounts of irrelevant text if you're not
careful. Always require the *company name* to also match, and keep an
explicit exclude list of common-word tickers.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyAlias:
    ticker: str
    company_name: str
    aliases: tuple[str, ...] = ()
    ambiguous_ticker: bool = False  # True if ticker collides with a common word


UNIVERSE: dict[str, CompanyAlias] = {
    "NVDA": CompanyAlias("NVDA", "NVIDIA", ("Nvidia Corporation",)),
    "TSLA": CompanyAlias("TSLA", "Tesla", ("Tesla Inc", "Tesla Motors")),
    "AVGO": CompanyAlias("AVGO", "Broadcom", ("Broadcom Inc",)),
    "QCOM": CompanyAlias("QCOM", "Qualcomm", ("Qualcomm Incorporated",)),
    "AMBA": CompanyAlias("AMBA", "Ambarella", ("Ambarella Inc",)),
    "TER":  CompanyAlias("TER", "Teradyne", ("Teradyne Inc",)),
    "ROK":  CompanyAlias("ROK", "Rockwell Automation", ()),
    "ABB":  CompanyAlias("ABB", "ABB Ltd", ("ABB Group",)),
    "ISRG": CompanyAlias("ISRG", "Intuitive Surgical", ()),
    "STM":  CompanyAlias("STM", "STMicroelectronics", ("STMicro",)),
    "TXN":  CompanyAlias("TXN", "Texas Instruments", ()),
    "HON":  CompanyAlias("HON", "Honeywell", ("Honeywell International",)),
    "TDY":  CompanyAlias("TDY", "Teledyne Technologies", ()),
    "NOC":  CompanyAlias("NOC", "Northrop Grumman", ()),
    "AME":  CompanyAlias("AME", "Ametek", ("AMETEK Inc",)),
}


def get_search_query(ticker: str) -> str:
    """Build a news-search query that requires the company name, not just the ticker."""
    entry = UNIVERSE.get(ticker)
    if entry is None:
        raise KeyError(f"Unknown ticker: {ticker}. Add it to UNIVERSE first.")
    names = [entry.company_name, *entry.aliases]
    name_clause = " OR ".join(f'"{n}"' for n in names)
    return f"({name_clause})"
