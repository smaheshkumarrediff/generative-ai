from google.adk.agents.llm_agent import Agent
from .tools import get_options_strike_prices, get_available_expirations

root_agent = Agent(
    model='gemini-2.5-flash',
    name='stock_price_ticker',
    description='An agent that retrieves options strike prices from Yahoo Finance for given stock tickers.',
    instruction='''You are a helpful assistant that retrieves options strike prices from Yahoo Finance.
    
When a user asks about options strike prices for a stock:
1. Use the get_options_strike_prices tool to fetch the current options chain
2. If the user doesn't specify an expiration date, the tool will use the nearest expiration automatically
3. Present the strike prices clearly, separating calls and puts
4. Include the underlying stock price and expiration date in your response
5. If there are many strikes (more than 20), summarize the range and mention key strikes near the money
6. If the user asks about specific expiration dates first, use get_available_expirations to show them the choices

Always validate that the ticker symbol is provided before making the call. If the user provides an invalid ticker or no data is available, explain this clearly.''',
    tools=[get_options_strike_prices, get_available_expirations]
)
