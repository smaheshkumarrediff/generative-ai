import os
import json
import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta

# Schwab API credentials should be loaded from environment variables
SCHWAB_CLIENT_ID = os.getenv("SCHWAB_CLIENT_ID")
SCHWAB_SECRET = os.getenv("SCHWAB_SECRET")
SCHWAB_REFRESH_TOKEN = os.getenv("SCHWAB_REFRESH_TOKEN")
SCHWAB_BASE_URL = os.getenv("SCHWAB_BASE_URL", "https://api.schwab.com")

class SchwabAgent:
    """
    SchwabAgent wraps the Schwab API OAuth flow and provides methods for
    trading options (e.g., covered calls).  The access token is obtained
    using the refresh token via the ``/oauth/token`` endpoint.

    **Obtaining Schwab Access & Refresh Tokens**

    Schwab's API uses OAuth 2.0.  The typical flow is:

    1. **Register an application** on the Schwab Developer Portal to obtain:
       - ``client_id`` (also called ``apiKey``)
       - ``client_secret``

    2. **Request an authorization code**:
       - Direct the user (or yourself in a test script) to the
         authorization endpoint:
         ``https://api.schwab.com/oauth/authorize``
       - Include the ``response_type=code``, ``client_id``, ``redirect_uri``,
         and the ``scope`` you need (e.g., ``accounts read``).
       - After the user consents, Schwab redirects to the ``redirect_uri``
         with a ``code`` query parameter.

    3. **Exchange the code for tokens**:
       - POST to ``https://api.schwab.com/oauth/token`` with:
         - ``grant_type=authorization_code``
         - ``code`` (the code from step 2)
         - ``redirect_uri`` (must match the one used in step 2)
         - ``client_id`` and ``client_secret`` (Basic Auth header or form data)
       - The response contains:
         - ``access_token`` – short‑lived (typically 1 hour)
         - ``refresh_token`` – long‑lived token used to obtain new access tokens
         - ``expires_in`` – seconds until expiration
         - ``token_type`` (usually ``Bearer``)

    4. **Refresh the access token** when it expires:
       - POST to ``https://api.schwab.com/oauth/token`` again, this time with:
         - ``grant_type=refresh_token``
         - ``refresh_token`` (the refresh token you stored)
         - ``client_id`` and ``client_secret``
       - The response gives a new ``access_token`` (and optionally a new
         ``refresh_token``).

    **Storing Tokens**

    - For local development you can keep the tokens in environment variables:
      ``SCHWAB_CLIENT_ID``, ``SCHWAB_SECRET``, ``SCHWAB_REFRESH_TOKEN``.
    - In production store them securely (e.g., secret manager,
      encrypted database) and rotate them regularly.

    **Example (pseudo‑code)**

    .. code-block:: python

        import requests
        import base64
        import os

        CLIENT_ID = os.getenv("SCHWAB_CLIENT_ID")
        CLIENT_SECRET = os.getenv("SCHWAB_SECRET")
        REDIRECT_URI = os.getenv("SCHWAB_REDIRECT_URI")

        # 1️⃣ Get authorization code (user interaction required)
        auth_url = (
            "https://api.schwab.com/oauth/authorize"
            f"?response_type=code&client_id={CLIENT_ID}"
            f"&redirect_uri={REDIRECT_URI}&scope=accounts%20read"
        )
        # ... open browser, user logs in, gets redirected with ?code=...

        # 2️⃣ Exchange code for tokens
        token_url = "https://api.schwab.com/oauth/token"
        auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        token_payload = {
            "grant_type": "authorization_code",
            "code": AUTHORIZATION_CODE,
            "redirect_uri": REDIRECT_URI,
        }
        headers = {"Authorization": f"Basic {auth_header}"}
        resp = requests.post(token_url, data=token_payload, headers=headers)
        tokens = resp.json()
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

        # 3️⃣ Use the access token in subsequent API calls
        #    When it expires, repeat step 2 with grant_type=refresh_token.
    """

    def __init__(self):
        self.session = requests.Session()
        # Attempt to authenticate during initialization; if it fails we keep
        # ``self.authenticated`` as False so that later calls raise a clear error.
        self.authenticated = self._authenticate()

        if not self.authenticated:
            raise RuntimeError(
                "Failed to obtain Schwab access token. "
                "Check that SCHWAB_CLIENT_ID, SCHWAB_SECRET, and SCHWAB_REFRESH_TOKEN "
                "are set correctly and that the token endpoint URL is correct."
            )

    def _authenticate(self) -> bool:
        """
        Obtain an access token using the stored refresh token.

        This method contacts the Schwab token endpoint
        ``{SCHWAB_BASE_URL}/oauth/token`` with a payload of type
        ``grant_type=refresh_token``.  On success the returned JSON is parsed
        and ``self.access_token`` and ``self.refresh_token`` are set.

        Returns
        -------
        bool
            ``True`` if authentication succeeded, ``False`` otherwise.

        Raises
        ------
        RuntimeError
            If the HTTP request fails or the response indicates an error.
        """
        token_url = f"{SCHWAB_BASE_URL}/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "client_id": SCHWAB_CLIENT_ID,
            "client_secret": SCHWAB_SECRET,
            "refresh_token": SCHWAB_REFRESH_TOKEN,
        }
        try:
            response = self.session.post(token_url, data=payload, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            # Surface the actual status code and body for debugging.
            raise RuntimeError(
                f"Authentication request failed with status {response.status_code}: {response.text}"
            ) from http_err
        except requests.exceptions.RequestException as req_err:
            raise RuntimeError(f"Network error while authenticating: {req_err}") from req_err

        token_data = response.json()
        self.access_token = token_data["access_token"]
        # Schwab may return a new refresh token; if not, keep the old one.
        self.refresh_token = token_data.get("refresh_token", SCHWAB_REFRESH_TOKEN)
        return True

    def _auth_headers(self) -> Dict[str, str]:
        """Return the Authorization header containing the current access token."""
        return {"Authorization": f"Bearer {self.access_token}"}

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Perform a GET request to ``{SCHWAB_BASE_URL}{endpoint}`` with auth headers."""
        url = f"{SCHWAB_BASE_URL}{endpoint}"
        resp = self.session.get(url, headers=self._auth_headers(), params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, json_body: Dict) -> Dict:
        """Perform a POST request to ``{SCHWAB_BASE_URL}{endpoint}`` with auth headers."""
        url = f"{SCHWAB_BASE_URL}{endpoint}"
        resp = self.session.post(url, headers=self._auth_headers(), json=json_body, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ----------------------------------------------------------------------
    # Options chain and expirations
    # ----------------------------------------------------------------------
    def get_available_expirations(self, ticker: str) -> List[str]:
        """Return a list of expiration dates (YYYY‑MM‑DD) for the given ticker."""
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
            Expiration date in YYYY‑MM‑DD format.
        contracts : int, optional
            Number of contracts (default 1).
        premium : float, optional
            Premium received per share. If not provided, the method queries the
            market for the current bid.
        underlying_cost_basis : float, optional
            Cost basis per share of the underlying stock you own.

        Returns
        -------
        dict
            Summary of the trade including max profit, max loss, and order status.
        """
        # 1. Validate inputs
        if not (0 < strike):
            raise ValueError("Strike price must be positive.")
        if not expiration:
            raise ValueError("Expiration date is required.")
        if contracts <= 0:
            raise ValueError("Number of contracts must be positive.")

        # 2. If premium not supplied, fetch the current bid for the call
        if premium is None:
            chain = self.get_options_chain(ticker, expiration)
            calls = chain.get("callExpirations", [{}])[0].get("calls", [])
            premium = next(
                (float(c["bid"]) for c in calls if float(c["strike"]) == strike), None
            )
            if premium is None:
                raise RuntimeError(f"Could not find market premium for strike {strike}.")
        
        # 3. Calculate theoretical max profit and max loss
        premium_total = premium * 100 * contracts
        if underlying_cost_basis is not None:
            max_loss = (underlying_cost_basis * 100 * contracts) - premium_total
        else:
            max_loss = None

        # 4. Place the order (simplified – actual order creation is more involved)
        order_payload = {
            "symbol": ticker,
            "orderType": "LIMIT",
            "session": "GTC",
            "price": premium,
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
            "max_profit": premium_total,
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
        underlying_cost_basis=140.0,
    )
    print(json.dumps(result, indent=2))

# Expose a root_agent for ADK discovery
root_agent = SchwabAgent()
