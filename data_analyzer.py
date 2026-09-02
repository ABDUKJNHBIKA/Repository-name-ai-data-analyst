import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def overview(self):
        return {
            "rows": int(len(self.df)),
            "columns": int(len(self.df.columns)),
            "column_names": [str(x) for x in self.df.columns],
            "duplicate_rows": int(self.df.duplicated().sum()),
            "missing_cells": int(self.df.isna().sum().sum()),
            "numeric_columns": [str(x) for x in self.df.select_dtypes(include="number").columns],
            "text_columns": [str(x) for x in self.df.select_dtypes(exclude="number").columns],
        }

    def missing_values(self):
        out = pd.DataFrame({
            "column": self.df.columns,
            "missing": self.df.isna().sum().values,
            "missing_percent": (self.df.isna().mean().values * 100).round(2)
        })
        return out.sort_values("missing", ascending=False)

    def numeric_statistics(self):
        num = self.df.select_dtypes(include="number")
        if num.empty:
            return pd.DataFrame()
        return num.describe().T.reset_index().rename(columns={"index": "column"}).round(3)

    def numeric_summary_dict(self):
        num = self.df.select_dtypes(include="number")
        result = {}
        for col in num.columns:
            x = pd.to_numeric(num[col], errors="coerce").dropna()
            if len(x):
                result[str(col)] = {
                    "mean": float(x.mean()),
                    "median": float(x.median()),
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "std": float(x.std()) if len(x) > 1 else 0.0
                }
        return result

    def correlations(self):
        num = self.df.select_dtypes(include="number")
        if num.shape[1] < 2:
            return pd.DataFrame({"message": ["At least two numeric columns are required."]})

        corr = num.corr(numeric_only=True)
        pairs = []
        cols = list(corr.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = corr.iloc[i, j]
                if pd.notna(value):
                    pairs.append({
                        "column_1": cols[i],
                        "column_2": cols[j],
                        "correlation": round(float(value), 4),
                        "absolute_correlation": round(abs(float(value)), 4)
                    })

        return pd.DataFrame(pairs).sort_values(
            "absolute_correlation", ascending=False
        ).head(20)

    def find_column_in_text(self, text, columns):
        low = text.lower()
        for col in columns:
            if str(col).lower() in low:
                return col

        aliases = {
            "sales": ["sales", "revenue", "amount", "income"],
            "profit": ["profit", "margin"],
            "price": ["price", "cost"],
            "quantity": ["quantity", "qty", "units"],
        }

        for canonical, words in aliases.items():
            if any(w in low for w in words):
                for col in columns:
                    if canonical in str(col).lower():
                        return col
        return None

    def make_chart(self, question):
        num_cols = list(self.df.select_dtypes(include="number").columns)
        if not num_cols:
            return None

        chosen = self.find_column_in_text(question.lower(), num_cols) or num_cols[0]

        # If there is a date-like column, create a time trend.
        date_col = None
        for col in self.df.columns:
            parsed = pd.to_datetime(self.df[col], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                date_col = col
                break

        fig, ax = plt.subplots(figsize=(10, 5))

        if date_col:
            temp = self.df[[date_col, chosen]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp[chosen] = pd.to_numeric(temp[chosen], errors="coerce")
            temp = temp.dropna().sort_values(date_col)

            if len(temp) > 100:
                temp = temp.groupby(temp[date_col].dt.to_period("M"))[chosen].sum().reset_index()
                temp[date_col] = temp[date_col].dt.to_timestamp()

            ax.plot(temp[date_col], temp[chosen])
            ax.set_xlabel(str(date_col))
            ax.set_ylabel(str(chosen))
            ax.set_title(f"{chosen} Trend")
            fig.autofmt_xdate()
        else:
            values = self.df[chosen].dropna().head(30)
            ax.bar(range(len(values)), values)
            ax.set_title(f"{chosen} — First 30 Values")
            ax.set_xlabel("Record")
            ax.set_ylabel(str(chosen))

        fig.tight_layout()
        return fig

    def recommendations(self):
        recs = []
        missing = self.missing_values()
        bad = missing[missing["missing_percent"] > 20]

        if not bad.empty:
            recs.append(
                "Prioritize columns with more than 20% missing values; investigate "
                "the source before using them for decisions."
            )

        duplicates = int(self.df.duplicated().sum())
        if duplicates:
            recs.append(f"Remove or review the {duplicates:,} duplicate rows before final reporting.")

        num = self.df.select_dtypes(include="number")
        if not num.empty:
            for col in num.columns:
                x = pd.to_numeric(num[col], errors="coerce").dropna()
                if len(x) >= 8:
                    q1, q3 = x.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
                        if outliers:
                            recs.append(
                                f"Review approximately {int(outliers):,} potential outliers in **{col}**."
                            )

        if not recs:
            recs.append("Data quality looks reasonable from the automated checks. Continue with domain-specific validation.")

        return recs[:8]

    def full_analysis(self):
        return {
            "overview": self.overview(),
            "missing_values": self.missing_values().to_dict(orient="records"),
            "numeric_summary": self.numeric_summary_dict(),
            "correlations": self.correlations().to_dict(orient="records"),
            "recommendations": self.recommendations()
        }
