#!/usr/bin/env python3
"""
DataPilot - AI-Powered Data Analysis Agent
By Vizible Results (www.vizibleresults.com)

A local-first AI agent that analyzes your CSV/Excel files using natural language.
Your data stays on your machine - only queries are sent to the AI provider.

Features:
- Natural language data analysis
- Automatic data cleaning
- Report generation
- Chart creation

Supports: OpenAI (GPT-4) and Anthropic (Claude)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving charts
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

def load_config() -> Dict[str, str]:
    """Load configuration from config.py or environment variables."""
    config = {
        'provider': os.getenv('DATAPILOT_PROVIDER', 'openai'),
        'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
        'model': os.getenv('DATAPILOT_MODEL', ''),
    }

    # Try to load from config.py if exists
    config_path = Path(__file__).parent / 'config.py'
    if config_path.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            if hasattr(config_module, 'PROVIDER'):
                config['provider'] = config_module.PROVIDER
            if hasattr(config_module, 'OPENAI_API_KEY'):
                config['openai_api_key'] = config_module.OPENAI_API_KEY
            if hasattr(config_module, 'ANTHROPIC_API_KEY'):
                config['anthropic_api_key'] = config_module.ANTHROPIC_API_KEY
            if hasattr(config_module, 'MODEL'):
                config['model'] = config_module.MODEL
        except Exception as e:
            print(f"Warning: Could not load config.py: {e}")

    return config

# =============================================================================
# AI Provider Abstraction
# =============================================================================

class AIProvider:
    """Abstraction layer for AI providers (OpenAI/Anthropic)."""

    def __init__(self, provider: str, api_key: str, model: str = None):
        self.provider = provider.lower()
        self.api_key = api_key

        if self.provider == 'openai':
            self.model = model or 'gpt-4-turbo-preview'
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        elif self.provider in ['anthropic', 'claude']:
            self.provider = 'anthropic'
            self.model = model or 'claude-sonnet-4-20250514'
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'")

    def query(self, prompt: str, system_prompt: str = None) -> str:
        """Send a query to the AI provider and return the response."""
        if self.provider == 'openai':
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=4096
            )
            return response.choices[0].message.content

        elif self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt or "You are a helpful data analysis assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

# =============================================================================
# Data Analyzer
# =============================================================================

class DataAnalyzer:
    """Handles data loading, analysis, and transformations."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = self._load_data()
        self.original_df = self.df.copy()

    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        suffix = self.file_path.suffix.lower()

        if suffix == '.csv':
            return pd.read_csv(self.file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .xlsx")

    def get_summary(self) -> str:
        """Generate a comprehensive data summary."""
        summary_parts = []

        # Basic info
        summary_parts.append(f"Dataset: {self.file_path.name}")
        summary_parts.append(f"Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        summary_parts.append("")

        # Column info
        summary_parts.append("Columns:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            unique_count = self.df[col].nunique()
            summary_parts.append(f"  - {col}: {dtype} ({unique_count} unique, {null_count} nulls)")
        summary_parts.append("")

        # Numeric statistics
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append("Numeric Statistics:")
            stats = self.df[numeric_cols].describe().round(2)
            summary_parts.append(stats.to_string())
            summary_parts.append("")

        # Sample data
        summary_parts.append("Sample Data (first 5 rows):")
        summary_parts.append(self.df.head().to_string())

        return "\n".join(summary_parts)

    def get_column_values(self, column: str, limit: int = 20) -> List:
        """Get sample values from a column."""
        if column not in self.df.columns:
            return []
        return self.df[column].dropna().unique()[:limit].tolist()

    def execute_query(self, query: str) -> Tuple[Any, str]:
        """Execute a pandas query and return result with explanation."""
        try:
            # Create a safe environment for eval
            safe_env = {
                'df': self.df,
                'pd': pd,
            }
            result = eval(query, {"__builtins__": {}}, safe_env)
            return result, "Query executed successfully"
        except Exception as e:
            return None, f"Error executing query: {str(e)}"

# =============================================================================
# Data Cleaner
# =============================================================================

class DataCleaner:
    """Handles data cleaning operations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cleaning_log = []

    def analyze_quality(self) -> Dict[str, Any]:
        """Analyze data quality issues."""
        issues = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': {},
            'duplicates': 0,
            'data_types': {},
            'recommendations': []
        }

        # Missing values
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                issues['missing_values'][col] = {
                    'count': int(null_count),
                    'percentage': round(null_count / len(self.df) * 100, 2)
                }

        # Duplicates
        issues['duplicates'] = int(self.df.duplicated().sum())

        # Data types
        for col in self.df.columns:
            issues['data_types'][col] = str(self.df[col].dtype)

        # Recommendations
        if issues['duplicates'] > 0:
            issues['recommendations'].append(
                f"Found {issues['duplicates']} duplicate rows. Consider removing with 'clean duplicates'."
            )

        for col, info in issues['missing_values'].items():
            if info['percentage'] > 50:
                issues['recommendations'].append(
                    f"Column '{col}' has {info['percentage']}% missing values. Consider dropping or imputing."
                )
            elif info['percentage'] > 0:
                issues['recommendations'].append(
                    f"Column '{col}' has {info['count']} missing values ({info['percentage']}%)."
                )

        return issues

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = original_count - len(self.df)
        self.cleaning_log.append(f"Removed {removed} duplicate rows")
        return self.df

    def fill_missing(self, strategy: str = 'auto') -> pd.DataFrame:
        """Fill missing values based on strategy."""
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    if strategy == 'auto' or strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                        self.cleaning_log.append(f"Filled missing values in '{col}' with median")
                    elif strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                        self.cleaning_log.append(f"Filled missing values in '{col}' with mean")
                else:
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                        self.cleaning_log.append(f"Filled missing values in '{col}' with mode")
        return self.df

    def standardize_text(self, columns: List[str] = None) -> pd.DataFrame:
        """Standardize text columns (trim, lowercase)."""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            if col in self.df.columns and self.df[col].dtype == 'object':
                self.df[col] = self.df[col].str.strip().str.lower()
                self.cleaning_log.append(f"Standardized text in '{col}'")

        return self.df

    def get_cleaning_report(self) -> str:
        """Get a report of all cleaning operations performed."""
        if not self.cleaning_log:
            return "No cleaning operations performed."
        return "Cleaning operations:\n" + "\n".join(f"  - {log}" for log in self.cleaning_log)

# =============================================================================
# Chart Generator
# =============================================================================

class ChartGenerator:
    """Generates charts and visualizations."""

    def __init__(self, df: pd.DataFrame, output_dir: str = "."):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _save_chart(self, fig, name: str) -> str:
        """Save chart to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return str(filepath)

    def bar_chart(self, x: str, y: str = None, title: str = None) -> str:
        """Create a bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if y is None:
            # Count plot
            data = self.df[x].value_counts().head(20)
            data.plot(kind='bar', ax=ax, color='#6366f1')
            ax.set_ylabel('Count')
        else:
            # Aggregated bar plot
            data = self.df.groupby(x)[y].sum().head(20)
            data.plot(kind='bar', ax=ax, color='#6366f1')
            ax.set_ylabel(y)

        ax.set_xlabel(x)
        ax.set_title(title or f'{x} Bar Chart')
        plt.xticks(rotation=45, ha='right')

        return self._save_chart(fig, 'bar_chart')

    def line_chart(self, x: str, y: str, title: str = None) -> str:
        """Create a line chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        self.df.plot(x=x, y=y, kind='line', ax=ax, color='#6366f1', marker='o')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{y} over {x}')
        ax.grid(True, alpha=0.3)

        return self._save_chart(fig, 'line_chart')

    def pie_chart(self, column: str, title: str = None) -> str:
        """Create a pie chart."""
        fig, ax = plt.subplots(figsize=(10, 8))

        data = self.df[column].value_counts().head(10)
        colors = plt.cm.Set3(range(len(data)))

        ax.pie(data.values, labels=data.index, autopct='%1.1f%%', colors=colors)
        ax.set_title(title or f'{column} Distribution')

        return self._save_chart(fig, 'pie_chart')

    def scatter_chart(self, x: str, y: str, title: str = None) -> str:
        """Create a scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(self.df[x], self.df[y], alpha=0.6, color='#6366f1')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f'{x} vs {y}')
        ax.grid(True, alpha=0.3)

        return self._save_chart(fig, 'scatter_chart')

    def histogram(self, column: str, bins: int = 30, title: str = None) -> str:
        """Create a histogram."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(self.df[column].dropna(), bins=bins, color='#6366f1', edgecolor='white')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(title or f'{column} Distribution')

        return self._save_chart(fig, 'histogram')

# =============================================================================
# Main DataPilot Agent
# =============================================================================

class DataPilot:
    """Main DataPilot agent orchestrating all components."""

    SYSTEM_PROMPT = """You are DataPilot, an AI data analysis assistant. You help users analyze their data through natural language.

You have access to a pandas DataFrame called 'df'. When the user asks a question:
1. Understand what they want to know
2. Generate the appropriate pandas code to answer their question
3. Explain the results in plain English

Available columns and their types will be provided in the context.

IMPORTANT:
- Only output valid Python/pandas code when asked to analyze data
- For charts, specify the chart type and columns
- Be concise and helpful
- Focus on actionable insights

When generating code, output ONLY the code without markdown formatting or explanations.
Format: Just the pandas expression that can be evaluated (e.g., df['column'].mean())"""

    def __init__(self, file_path: str, config: Dict[str, str] = None):
        self.config = config or load_config()
        self.analyzer = DataAnalyzer(file_path)
        self.cleaner = DataCleaner(self.analyzer.df)
        self.chart_gen = ChartGenerator(self.analyzer.df)

        # Initialize AI provider
        provider = self.config['provider']
        api_key = (self.config['openai_api_key'] if provider == 'openai'
                   else self.config['anthropic_api_key'])

        if not api_key:
            raise ValueError(f"No API key found for provider '{provider}'. "
                           "Set it in config.py or as environment variable.")

        self.ai = AIProvider(provider, api_key, self.config.get('model'))
        print(f"DataPilot initialized with {provider} ({self.ai.model})")
        print(f"Loaded: {file_path}")
        print(f"Shape: {self.analyzer.df.shape[0]} rows x {self.analyzer.df.shape[1]} columns\n")

    def _get_context(self) -> str:
        """Get context about the current dataset."""
        return f"""Current Dataset Context:
{self.analyzer.get_summary()}
"""

    def ask(self, question: str) -> str:
        """Process a natural language question about the data."""
        context = self._get_context()
        prompt = f"""{context}

User Question: {question}

If this requires analysis, provide the pandas code to answer it.
If this is a general question, answer directly.
If a chart is requested, specify: CHART_TYPE: [bar/line/pie/scatter/histogram], X: [column], Y: [column or None]"""

        response = self.ai.query(prompt, self.SYSTEM_PROMPT)

        # Check if response contains chart request
        if 'CHART_TYPE:' in response:
            return self._handle_chart_request(response)

        # Try to execute as pandas code
        try:
            # Clean the response (remove markdown formatting if present)
            code = response.strip()
            if code.startswith('```'):
                code = code.split('\n', 1)[1]
                code = code.rsplit('```', 1)[0]
            code = code.strip()

            if code.startswith('df'):
                result, status = self.analyzer.execute_query(code)
                if result is not None:
                    return f"Result:\n{result}\n\n(Code: {code})"
        except:
            pass

        return response

    def _handle_chart_request(self, response: str) -> str:
        """Handle chart generation requests."""
        try:
            lines = response.split('\n')
            chart_type = None
            x_col = None
            y_col = None

            for line in lines:
                if 'CHART_TYPE:' in line:
                    chart_type = line.split(':')[1].strip().lower()
                elif 'X:' in line:
                    x_col = line.split(':')[1].strip()
                elif 'Y:' in line:
                    y_val = line.split(':')[1].strip()
                    y_col = None if y_val.lower() == 'none' else y_val

            if chart_type and x_col:
                if chart_type == 'bar':
                    path = self.chart_gen.bar_chart(x_col, y_col)
                elif chart_type == 'line':
                    path = self.chart_gen.line_chart(x_col, y_col)
                elif chart_type == 'pie':
                    path = self.chart_gen.pie_chart(x_col)
                elif chart_type == 'scatter':
                    path = self.chart_gen.scatter_chart(x_col, y_col)
                elif chart_type == 'histogram':
                    path = self.chart_gen.histogram(x_col)
                else:
                    return f"Unknown chart type: {chart_type}"

                return f"Chart saved to: {path}"
        except Exception as e:
            return f"Error generating chart: {str(e)}"

        return response

    def clean(self, operation: str = 'all') -> str:
        """Run data cleaning operations."""
        self.cleaner = DataCleaner(self.analyzer.df)

        if operation in ['all', 'duplicates']:
            self.cleaner.remove_duplicates()

        if operation in ['all', 'missing']:
            self.cleaner.fill_missing()

        if operation in ['all', 'text']:
            self.cleaner.standardize_text()

        # Update analyzer with cleaned data
        self.analyzer.df = self.cleaner.df
        self.chart_gen.df = self.cleaner.df

        return self.cleaner.get_cleaning_report()

    def quality_report(self) -> str:
        """Generate a data quality report."""
        issues = self.cleaner.analyze_quality()

        report = ["=" * 50]
        report.append("DATA QUALITY REPORT")
        report.append("=" * 50)
        report.append(f"\nDataset: {self.analyzer.file_path.name}")
        report.append(f"Total Rows: {issues['total_rows']}")
        report.append(f"Total Columns: {issues['total_columns']}")
        report.append(f"Duplicate Rows: {issues['duplicates']}")

        if issues['missing_values']:
            report.append("\nMissing Values:")
            for col, info in issues['missing_values'].items():
                report.append(f"  - {col}: {info['count']} ({info['percentage']}%)")
        else:
            report.append("\nNo missing values found!")

        if issues['recommendations']:
            report.append("\nRecommendations:")
            for rec in issues['recommendations']:
                report.append(f"  - {rec}")

        report.append("\n" + "=" * 50)
        return "\n".join(report)

    def summary(self) -> str:
        """Get a summary of the data."""
        return self.analyzer.get_summary()

    def save(self, output_path: str = None) -> str:
        """Save the current (potentially cleaned) data to a file."""
        if output_path is None:
            stem = self.analyzer.file_path.stem
            output_path = f"{stem}_cleaned.csv"

        self.analyzer.df.to_csv(output_path, index=False)
        return f"Data saved to: {output_path}"

# =============================================================================
# CLI Interface
# =============================================================================

def interactive_mode(pilot: DataPilot):
    """Run DataPilot in interactive mode."""
    print("\n" + "=" * 50)
    print("DataPilot Interactive Mode")
    print("=" * 50)
    print("\nCommands:")
    print("  ask <question>  - Ask a question about your data")
    print("  summary         - Show data summary")
    print("  quality         - Show data quality report")
    print("  clean [type]    - Clean data (all/duplicates/missing/text)")
    print("  save [path]     - Save cleaned data")
    print("  help            - Show this help")
    print("  exit            - Exit DataPilot")
    print("\nOr just type your question directly!\n")

    while True:
        try:
            user_input = input("DataPilot> ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            elif user_input.lower() == 'help':
                print("\nCommands: ask, summary, quality, clean, save, exit")
                print("Or just type your question!\n")

            elif user_input.lower() == 'summary':
                print("\n" + pilot.summary() + "\n")

            elif user_input.lower() == 'quality':
                print("\n" + pilot.quality_report() + "\n")

            elif user_input.lower().startswith('clean'):
                parts = user_input.split(maxsplit=1)
                operation = parts[1] if len(parts) > 1 else 'all'
                print("\n" + pilot.clean(operation) + "\n")

            elif user_input.lower().startswith('save'):
                parts = user_input.split(maxsplit=1)
                path = parts[1] if len(parts) > 1 else None
                print("\n" + pilot.save(path) + "\n")

            elif user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                print("\n" + pilot.ask(question) + "\n")

            else:
                # Treat as a question
                print("\n" + pilot.ask(user_input) + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DataPilot - AI-Powered Data Analysis Agent by Vizible Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python datapilot.py data.csv
  python datapilot.py data.xlsx --question "What is the average sales?"
  python datapilot.py data.csv --summary
  python datapilot.py data.csv --quality
  python datapilot.py data.csv --clean

For more information, visit: www.vizibleresults.com
        """
    )

    parser.add_argument('file', help='Path to CSV or Excel file')
    parser.add_argument('-q', '--question', help='Ask a single question and exit')
    parser.add_argument('-s', '--summary', action='store_true', help='Show data summary')
    parser.add_argument('--quality', action='store_true', help='Show data quality report')
    parser.add_argument('--clean', action='store_true', help='Clean the data')
    parser.add_argument('--save', help='Save cleaned data to file')
    parser.add_argument('--provider', choices=['openai', 'anthropic'],
                       help='AI provider to use')
    parser.add_argument('--model', help='Model to use (e.g., gpt-4, claude-3-sonnet)')

    args = parser.parse_args()

    # Build config from args
    config = load_config()
    if args.provider:
        config['provider'] = args.provider
    if args.model:
        config['model'] = args.model

    try:
        pilot = DataPilot(args.file, config)

        if args.summary:
            print(pilot.summary())
        elif args.quality:
            print(pilot.quality_report())
        elif args.clean:
            print(pilot.clean())
            if args.save:
                print(pilot.save(args.save))
        elif args.question:
            print(pilot.ask(args.question))
        else:
            interactive_mode(pilot)

    except FileNotFoundError:
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
