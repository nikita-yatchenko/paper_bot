# ArXiv Paper Analyzer Bot 🤖📄📚

A Telegram bot that fetches, analyzes, and summarizes research papers from arXiv.

✨ Features
/search – Find papers on arXiv by keyword

/get [arxiv_id] – Download a paper as PDF

/summarize – Generate a concise summary of a paper

/analyze – Extract key insights (methods, results, etc.)

/save – Store papers for later reading

## 🚀 Setup & Installation
1. Clone the repo

```
git clone https://github.com/yourusername/arxiv-telegram-bot.git
cd arxiv-telegram-bot
```
2. Configure environment variables

```
cp .env.example .env
```
Edit .env with:
```
TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"  # From @BotFather
ARXIV_MAX_RESULTS=5                  # Max papers per search
```
3. Set up Poetry

Install Poetry if you don’t have it:
```
curl -sSL https://install.python-poetry.org | python3 -
```
4. Install dependencies

```
poetry install  # Creates virtual env and installs packages
```
5. Run the bot

```
poetry run python src/main.py
```

# Add other configuration variables as needed
Available Commands
```
/start - Welcome message and brief instructions

/help - Show available commands

/search [query] - Search arXiv for papers

/analyze [arxiv_id] - Analyze a specific paper

/summary [arxiv_id] - Get paper summary

/download [arxiv_id] - Download paper PDF
```

Development
Project Structure
```
arxiv-bot/
├── bot/                  # Bot handlers
├── services/             # arXiv API services
├── utils/                # Helper functions
├── main.py               # Entry point
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

Contributing
Pull requests are welcome! For major changes, please open an issue first.
