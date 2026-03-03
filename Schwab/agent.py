import os
import json
import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta

# Schwab API credentials should be loaded from environment variables
SCHWAB_CLIENT_ID = os.getenv("SCHWAB_CLIENT_ID")
SCHWAB_SECRET = os.getenv("SCHWAB_SECRET")
SCHWAB_REFRESH_TOKEN = os.getenv("SCHWAB_REFRESH_TOKEN")
SCHWAB_BASE_URL = "https://api.schwab.com"

class SchwabAgent:
    def __init__(self):
        self.session = requests.Session()
        self._authenticate()

    def _authenticate(self):
        """Obtain an access token using the refresh token."""
        token_url = f"{SCHWAB_BASE_URL}/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "client_id": SCHWAB_CLIENT_ID,
            "client_secret": SCHWAB_SECRET,
            "refresh_token": SCHWAB_REFRESH_TOKEN,
        }
        response = self.session.post(token_url, data=payload)
        response.raise_for_status()
        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token", SCHWAB_REFRESH_TOKEN)

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        url = f"{SCHWAB_BASE_URL}{endpoint}"
        resp = self.session.get(url, headers=self._auth_headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, json_body: Dict) -> Dict:
        url = f"{SCHWAB_BASE_URL}{endpoint}"
        resp = self.session.post(url, headers=self._auth_headers(), json=json_body)
        resp.raise_for_status()
        return resp.json()

    # ----------------------------------------------------------------------
    # Options chain and expirations
    # ----------------------------------------------------------------------
    def get_available_expirations(self, ticker: str) -> List[str]:
        """Return a list of expiration dates (as YYYY-MM-DD strings) for the given ticker."""
        data = self._get(f"/marketdata/v1/options/expirations", {"symbol": ticker})
        return [exp["expirationDate"] for exp in data.get("expirations", [])]

    def get_options_chain(self, ticker: str, expiration: str) -> Dict:
        """Fetch the full options chain for a ticker and expiration date."""
        data = self._get(
            f"/marketdata/v1/options/chains",
            {"symbol": ticker, "expiration": expiration},
        )
        return data

    # ----------------------------------------------------------------------
    # Covered Call Strategy
    # ----------------------------------------------------------------------
    def execute_covered_call(
        self,
        ticker: str,
        strike: float,
        expiration: str,
        contracts: int = 1,
        premium: Optional[float] = None,
        underlying_cost_basis: Optional[float] = None,
    ) -> Dict:
        """
        Execute a covered call.

        Parameters
        ----------
        ticker : str
            Underlying stock symbol.
        strike : float
            Strike price of the call option.
        expiration : str
            Expiration date in YYYY-MM-DD format.
        contracts : int, optional
            Number of contracts (default 1).
        premium : float, optional
            Premium received per share. If not provided, the agent will query the market.
        underlying_cost_basis : float, optional
            Cost basis per share of the underlying stock you own.

        Returns
        -------
        dict
            Summary of the trade including max profit, max loss, and order status.
        """
        # 1. Validate inputs
        if not (0 < strike <= underlying_cost_basis * 2):  # simple sanity check
            raise ValueError("Strike price seems unrealistic.")
        if not expiration:
            raise ValueError("Expiration date is required.")
        if contracts <= 0:
            raise ValueError("Number of contracts must be positive.")

        # 2. If premium not supplied, fetch the current bid for the call
        if premium is None:
            chain = self.get_options_chain(ticker, expiration)
            calls = chain.get("callExpirations", [{}])[0].get("calls", [])
            premium = next(
                (c["bid"] for c in calls if float(c["strike"]) == strike), None
            )
            if premium is None:
                raise RuntimeError(f"Could not find market premium for strike {strike}.")
            premium = float(premium)

        # 3. Calculate theoretical max profit and max loss
        #    - Max profit = premium * 100 * contracts (if underlying is called away)
        #    - Max loss = (underlying_cost_basis - strike) * 100 * contracts - premium
        #      (only applies if you bought the stock at cost_basis)
        premium_total = premium * 100 * contracts
        if underlying_cost_basis is not None:
            # Loss if stock price falls to $0 (worst case) minus premium received
            max_loss = (underlying_cost_basis * 100 * contracts) - premium_total
        else:
            max_loss = None  # cannot compute without cost basis

        # 4. Place the order (simplified – actual order creation is more involved)
        order_payload = {
            "symbol": ticker,
            "orderType": "LIMIT",
            "session": "GTC",
            "price": premium,  # limit price for the option
            "quantity": contracts * 100,
            "instrument": "OPTION",
            "optionChains": [
                {
                    "strike": strike,
                    "expirationDate": expiration,
                    "type": "CALL",
                }
            ],
        }
        response = self._post("/trading/v1/orders", order_payload)
        order_status = response.get("orderStatus", "UNKNOWN")

        # 5. Return a summary
        return {
            "ticker": ticker,
            "strike": strike,
            "expiration": expiration,
            "contracts": contracts,
            "premium_received_per_contract": premium,
            "premium_total": premium_total,
            "max_profit": premium_total,  # simplified
            "max_loss": max_loss,
            "order_status": order_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

# ----------------------------------------------------------------------
# Example usage (will only run when this file is executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple demo – replace with real values or environment variables
    agent = SchwabAgent()
    result = agent.execute_covered_call(
        ticker="AAPL",
        strike=150.0,
        expiration="2025-01-17",
        contracts=1,
        underlying_cost_basis=140.0,  # example purchase price
    )
    print(json.dumps(result, indent=2))
