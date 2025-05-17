from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
import datetime
import os
import re
from indicatorfinal import process_stock

load_dotenv()

class StockContext:
    def __init__(self):
        self.df_store = {}

    def load_df(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.strip().replace("'", "").replace('"', '').upper()
        expected_file = f"{symbol}_analysis.csv"

        all_files = os.listdir()
        print(f"\n🔍 Looking for file matching: {expected_file}")
        #print(f"📁 Current directory: {os.getcwd()}")
        #print(f"📄 Files: {all_files}")

        matching_files = [f for f in all_files if f.lower() == expected_file.lower()]

        if not matching_files:
            print(f"❌ No matching file found for: {expected_file}")
            raise FileNotFoundError(f"No analysis file found for '{symbol}'. Please run stock analysis first.")

        file_path = matching_files[0]
        print(f"✅ Found file: {file_path}")

        if symbol not in self.df_store:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            df.index = df.index.normalize()  # strip time part
            self.df_store[symbol] = df

        return self.df_store[symbol]



    def get_indicator(self, input_str: str) -> str:
        try:
        # Remove stray quotes from input
            symbol, date_str, indicator = [x.strip().replace("'", "").replace('"', '') for x in input_str.split(',')]
            df = self.load_df(symbol)
            date = pd.to_datetime(date_str).normalize()

            if indicator not in df.columns:
                return f"❌ Indicator '{indicator}' not found for {symbol}."
            if date not in df.index:
                return f"❌ Date {date_str} not found in data for {symbol}."

            value = df.loc[date, indicator]
            return f"✅ {indicator} for {symbol} on {date_str} is {value:.2f}"
        except Exception as e:
            return f"❌ Error: {str(e)}. Format: symbol,date,indicator (e.g., SBIN.NS,2024-05-01,RSI)"

    def get_latest_summary(self, symbol: str) -> str:
        try:
            df = self.load_df(symbol.strip())
            latest = df.iloc[-1]
            return (
                f"📊 Latest summary for {symbol}:\n"
                f"📅 Date: {latest.name.date()}\n"
                f"📈 RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, OBV: {latest['OBV']:.2f}\n"
                f"🌀 Supertrend: {latest['ST_Direction']}, Weekly Trend: {latest['Weekly_Trend']}\n"
                f"🟢 Composite Buy Signal: {'Yes' if latest['Composite_Buy_Signal'] else 'No'}\n"
                f"🤖 ML Prediction: {'Buy' if latest['ML_Buy_Prediction'] else 'Sell'}"
            )
        except Exception as e:
            return f"❌ Error fetching summary for {symbol}: {str(e)}"

    def run_analysis(self, input_str: str) -> str:
        try:
            match = re.search(r"([A-Z0-9\.]+).*?(\d{4}-\d{2}-\d{2}).*?(\d{4}-\d{2}-\d{2})", input_str)
            if match:
                symbol, start, end = match.groups()
            else:
                symbol, start, end = [x.strip() for x in input_str.split(',')]

            if not end:
                end = datetime.datetime.today().strftime('%Y-%m-%d')

            result = process_stock(symbol, start, end, verbose=False)

            file_path = f"{symbol}_analysis.csv"
            if os.path.exists(file_path):
                return f"✅ Analysis complete for {symbol} ({start} to {end}). File saved successfully.\n\n{result}"
            else:
                return f"⚠️ Analysis ran but file {file_path} was not created. Please check for internal errors."

        except ValueError as ve:
            return f"⚠️ Failed to analyze. Reason: {str(ve)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"


# Initialize context and tools
stock_context = StockContext()

tools = [
    Tool(
        name="Run Stock Analysis",
        func=stock_context.run_analysis,
        description="Run full stock analysis using all technical indicators. Example: 'Analyze SBIN.NS from 2024-01-01 to 2025-05-13'"
    ),
    Tool(
        name="Get Indicator Value",
        func=stock_context.get_indicator,
        description="Fetch a specific indicator value. Format: 'symbol,date,indicator'. Example: 'SBIN.NS,2024-05-01,RSI'"
    ),
    Tool(
        name="Get Latest Summary",
        func=stock_context.get_latest_summary,
        description="Summarize latest indicators and signals. Input: 'symbol' (e.g., SBIN.NS)"
    )
]

# Setup agent
llm = ChatOpenAI(temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# CLI Loop
if __name__ == "__main__":
    print("\n📈 LangChain Agent for Stock Analysis Initialized")
    print("👉 Try these examples:")
    print("   • Run stock analysis for SBIN.NS from 2024-01-01 to 2025-05-13")
    print("   • What was the RSI for SBIN.NS on 2024-04-15?")
    print("   • Show latest signal summary for SBIN.NS\n")

    while True:
        query = input("🧠 You: ")
        if query.lower() in ['exit', 'quit']:
            print("👋 Exiting agent.")
            break
        response = agent.run(query)
        print(f"\n🤖 Agent: {response}\n")
