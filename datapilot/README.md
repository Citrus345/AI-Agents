# DataPilot - AI-Powered Data Analysis Agent

**By [Vizible Results](https://www.vizibleresults.com)**

DataPilot is a free, local-first AI agent that helps you analyze CSV and Excel files using natural language. Your data never leaves your machine - only your questions are sent to the AI provider.

## Features

- **Natural Language Analysis**: Ask questions about your data in plain English
- **Data Cleaning**: Automatically detect and fix data quality issues
- **Report Generation**: Get comprehensive summaries and statistics
- **Chart Creation**: Generate bar, line, pie, scatter, and histogram charts
- **Privacy-First**: Your data stays local - only queries go to the AI

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your API Key

Copy the example config and add your API key:

```bash
cp config.example.py config.py
```

Edit `config.py` and add your OpenAI or Anthropic API key:

```python
PROVIDER = 'openai'  # or 'anthropic'
OPENAI_API_KEY = 'sk-your-key-here'
```

**Get your API key:**
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

### 3. Run DataPilot

```bash
# Interactive mode
python datapilot.py your_data.csv

# Ask a single question
python datapilot.py sales.csv -q "What were the top 5 products by revenue?"

# Show data summary
python datapilot.py data.csv --summary

# Check data quality
python datapilot.py data.csv --quality

# Clean data and save
python datapilot.py data.csv --clean --save cleaned_data.csv
```

## Usage Examples

### Interactive Mode

```bash
$ python datapilot.py sample_data/sample.csv

DataPilot initialized with openai (gpt-4-turbo-preview)
Loaded: sample.csv
Shape: 1000 rows x 8 columns

==================================================
DataPilot Interactive Mode
==================================================

Commands:
  ask <question>  - Ask a question about your data
  summary         - Show data summary
  quality         - Show data quality report
  clean [type]    - Clean data (all/duplicates/missing/text)
  save [path]     - Save cleaned data
  help            - Show this help
  exit            - Exit DataPilot

DataPilot> What is the average sales amount?

Result: 1523.45

DataPilot> Show me a bar chart of sales by region

Chart saved to: bar_chart_20260110_143052.png

DataPilot> Which products have the highest profit margin?

Result:
   product_name  profit_margin
0  Premium Widget       42.3%
1  Pro Service        38.7%
2  Enterprise Plan    35.2%
```

### Command Line Mode

```bash
# Quick question
python datapilot.py data.csv -q "How many unique customers do we have?"

# With a different AI provider
python datapilot.py data.csv --provider anthropic -q "Summarize this data"

# Specify model
python datapilot.py data.csv --model gpt-4 -q "What trends do you see?"
```

## Environment Variables

You can also configure DataPilot using environment variables:

```bash
export DATAPILOT_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key-here
export DATAPILOT_MODEL=gpt-4-turbo-preview
```

## Supported File Formats

- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

## Data Privacy

DataPilot is designed with privacy in mind:

1. **Your data stays local**: All data processing happens on your machine
2. **Only queries go to AI**: The AI only sees your questions and a summary of your data structure
3. **No data storage**: We don't store or have access to your data
4. **Self-hosted**: Run it entirely within your firewall

## Cleaning Operations

DataPilot can automatically clean your data:

| Command | Description |
|---------|-------------|
| `clean all` | Run all cleaning operations |
| `clean duplicates` | Remove duplicate rows |
| `clean missing` | Fill missing values (median for numbers, mode for text) |
| `clean text` | Standardize text (trim whitespace, lowercase) |

## Chart Types

| Type | Usage |
|------|-------|
| Bar | `Create a bar chart of sales by category` |
| Line | `Show a line chart of revenue over time` |
| Pie | `Make a pie chart of market share` |
| Scatter | `Plot price vs quantity as a scatter` |
| Histogram | `Show the distribution of order values` |

## Troubleshooting

### "No API key found"
Make sure you've either:
- Created `config.py` with your API key, or
- Set the environment variable (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)

### "Module not found"
Run `pip install -r requirements.txt` to install all dependencies.

### "Unsupported file format"
DataPilot only supports `.csv`, `.xlsx`, and `.xls` files.

## Need More?

DataPilot is a free tool from Vizible Results. For enterprise AI solutions, custom agents, or consulting services:

- **Website**: [vizibleresults.com](https://www.vizibleresults.com)
- **Email**: info@vizibleresults.com
- **Phone**: 857-297-2945

---

Made with AI by [Vizible Results](https://www.vizibleresults.com) - AI Agents & Enterprise Consulting
