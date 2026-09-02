from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def generate_report(df, analysis, output_dir="/tmp"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("AI Data Analyst Report", styles["Title"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 18))

    ov = analysis["overview"]
    story.append(Paragraph("1. Dataset Overview", styles["Heading2"]))
    overview_data = [
        ["Metric", "Value"],
        ["Rows", f"{ov['rows']:,}"],
        ["Columns", str(ov["columns"])],
        ["Duplicate rows", f"{ov['duplicate_rows']:,}"],
        ["Missing cells", f"{ov['missing_cells']:,}"],
    ]
    t = Table(overview_data, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("2. Numeric Analysis", styles["Heading2"]))
    for col, vals in list(analysis["numeric_summary"].items())[:12]:
        text = (
            f"<b>{col}</b>: mean={vals['mean']:.2f}, "
            f"median={vals['median']:.2f}, "
            f"min={vals['min']:.2f}, max={vals['max']:.2f}"
        )
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 5))

    story.append(Paragraph("3. Correlations", styles["Heading2"]))
    corr = analysis["correlations"][:10]
    if corr:
        data = [["Column 1", "Column 2", "Correlation"]]
        for row in corr:
            data.append([str(row["column_1"]), str(row["column_2"]), str(row["correlation"])])
        t = Table(data, colWidths=[2.2*inch, 2.2*inch, 1.3*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No correlation results available.", styles["BodyText"]))

    story.append(Spacer(1, 16))
    story.append(Paragraph("4. Actionable Recommendations", styles["Heading2"]))
    for rec in analysis["recommendations"]:
        story.append(Paragraph("• " + rec, styles["BodyText"]))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "This report was generated automatically. Validate important business "
        "decisions against domain knowledge and source systems.",
        styles["Italic"]
    ))

    doc = SimpleDocTemplate(
        str(filename), pagesize=A4,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    doc.build(story)
    return str(filename)
