import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from agent import DataAnalystAgent
from report_generator import generate_report

load_dotenv()

st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Analyst")
st.caption("Upload CSV/Excel data and chat with an AI-powered analysis agent.")

if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Initialize / Refresh Agent", use_container_width=True):
                st.session_state.agent = DataAnalystAgent(df)
                st.session_state.messages = []
                st.session_state.last_analysis = None
                st.success("Agent is ready.")

        except Exception as e:
            st.error(f"Could not read the file: {e}")

    if st.session_state.agent:
        st.divider()
        st.write("**Quick actions**")
        if st.button("Run full analysis", use_container_width=True):
            with st.spinner("Analyzing dataset..."):
                result = st.session_state.agent.full_analysis()
                st.session_state.last_analysis = result
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["summary"]
                })
            st.rerun()

        if st.button("Generate PDF report", use_container_width=True):
            if st.session_state.last_analysis is None:
                with st.spinner("Running analysis first..."):
                    st.session_state.last_analysis = st.session_state.agent.full_analysis()

            path = generate_report(
                st.session_state.agent.df,
                st.session_state.last_analysis
            )
            with open(path, "rb") as f:
                st.download_button(
                    "Download PDF",
                    f,
                    file_name=os.path.basename(path),
                    mime="application/pdf",
                    use_container_width=True
                )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("table") is not None:
            st.dataframe(message["table"], use_container_width=True)
        if message.get("figure") is not None:
            st.pyplot(message["figure"])

if st.session_state.agent is None:
    st.info("Upload a dataset from the sidebar and click 'Initialize / Refresh Agent'.")
else:
    prompt = st.chat_input(
        "Ask about your dataset, e.g. 'Find the top 10 products' or 'Show monthly sales'"
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking and analyzing..."):
                answer = st.session_state.agent.ask(prompt)

            st.markdown(answer["text"])
            if answer.get("table") is not None:
                st.dataframe(answer["table"], use_container_width=True)
            if answer.get("figure") is not None:
                st.pyplot(answer["figure"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer["text"],
            "table": answer.get("table"),
            "figure": answer.get("figure")
        })
