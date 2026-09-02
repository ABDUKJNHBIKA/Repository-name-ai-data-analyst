import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from data_analyzer import DataAnalyzer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class DataAnalystAgent:
    """
    Hybrid agent:
    - Python/Pandas performs factual calculations.
    - Optional OpenAI model turns calculated results into natural-language insights.
    - If no API key is configured, deterministic local responses still work.
    """

    def __init__(self, df):
        self.df = df.copy()
        self.analyzer = DataAnalyzer(self.df)
        self.client = None

        key = os.getenv("OPENAI_API_KEY")
        if key and OpenAI:
            self.client = OpenAI(api_key=key)

    def _ai_explain(self, user_question, analysis_result):
        if not self.client:
            return None

        prompt = f"""
You are an AI Data Analyst. Answer the user's question using ONLY the supplied
computed analysis. Do not invent numbers. Be concise and actionable.

User question:
{user_question}

Computed analysis:
{json.dumps(analysis_result, default=str, indent=2)[:15000]}
"""

        try:
            response = self.client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                input=prompt
            )
            return response.output_text
        except Exception:
            return None

    def full_analysis(self):
        result = self.analyzer.full_analysis()
        ai = self._ai_explain("Give me a complete executive summary and actionable recommendations.", result)

        if ai:
            result["summary"] = ai
        else:
            result["summary"] = self._local_summary(result)

        return result

    def _local_summary(self, r):
        s = r["overview"]
        lines = [
            "### 📊 Dataset Analysis",
            f"- **Rows:** {s['rows']:,}",
            f"- **Columns:** {s['columns']}",
            f"- **Duplicate rows:** {s['duplicate_rows']:,}",
            f"- **Missing cells:** {s['missing_cells']:,}",
        ]

        if r["numeric_summary"]:
            lines.append("\n### 🔢 Numeric Findings")
            for col, vals in list(r["numeric_summary"].items())[:6]:
                lines.append(
                    f"- **{col}:** mean {vals['mean']:.2f}, "
                    f"min {vals['min']:.2f}, max {vals['max']:.2f}"
                )

        if r["recommendations"]:
            lines.append("\n### ✅ Actionable Recommendations")
            lines.extend([f"- {x}" for x in r["recommendations"]])

        return "\n".join(lines)

    def ask(self, question):
        q = question.lower().strip()

        # Full/general analysis
        if any(x in q for x in [
            "analyze", "analysis", "overview", "summary",
            "insight", "recommendation", "recommendations"
        ]):
            result = self.full_analysis()
            return {"text": result["summary"]}

        # Dataset structure
        if any(x in q for x in ["columns", "column names", "fields", "structure"]):
            cols = list(self.df.columns)
            return {"text": "### Dataset Columns\n" + "\n".join(f"- `{c}`" for c in cols)}

        if "missing" in q or "null" in q:
            table = self.analyzer.missing_values()
            return {
                "text": "### Missing Values\nHere is the missing-value count by column.",
                "table": table
            }

        if "duplicate" in q:
            n = int(self.df.duplicated().sum())
            return {"text": f"### Duplicate Rows\nThere are **{n:,} duplicate rows**."}

        # Top/bottom rows
        top_match = re.search(r"(?:top|highest|best)\s*(\d+)?", q)
        bottom_match = re.search(r"(?:bottom|lowest|worst)\s*(\d+)?", q)

        if top_match or bottom_match:
            n = int((top_match or bottom_match).group(1) or 10)
            ascending = bool(bottom_match)
            numeric_cols = list(self.df.select_dtypes(include="number").columns)

            # Try to use a column explicitly named in the question.
            chosen = self.analyzer.find_column_in_text(q, numeric_cols)
            if not chosen and numeric_cols:
                chosen = numeric_cols[0]

            if chosen:
                table = self.df.sort_values(chosen, ascending=ascending).head(n)
                direction = "lowest" if ascending else "highest"
                return {
                    "text": f"### {direction.title()} {n} by **{chosen}**",
                    "table": table
                }

        # Chart/trend requests
        if any(x in q for x in ["chart", "graph", "plot", "trend", "visual"]):
            fig = self.analyzer.make_chart(q)
            if fig is not None:
                return {"text": "### 📈 Chart\nI generated a chart from the most relevant columns.", "figure": fig}
            return {"text": "I couldn't find suitable columns for that chart."}

        # Correlation
        if "correlation" in q or "correlat" in q:
            table = self.analyzer.correlations()
            return {
                "text": "### 🔗 Correlations\nThese are the strongest numeric relationships I found.",
                "table": table
            }

        # Statistics
        if any(x in q for x in ["statistics", "stats", "average", "mean", "median", "std"]):
            table = self.analyzer.numeric_statistics()
            return {
                "text": "### 📐 Numeric Statistics",
                "table": table
            }

        # Let the LLM interpret computed overview for natural questions.
        result = self.analyzer.full_analysis()
        ai = self._ai_explain(question, result)
        if ai:
            return {"text": ai}

        return {
            "text": (
                "I can analyze the dataset, find top/bottom records, calculate "
                "statistics, check missing/duplicate values, find correlations, "
                "create charts, and generate recommendations.\n\n"
                "Try: **'Analyze my dataset'**, **'Show top 10'**, "
                "**'Find missing values'**, or **'Create a sales trend chart'**."
            )
        }
