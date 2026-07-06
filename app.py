import streamlit as st

st.set_page_config(page_title="Spotify Review Discovery Engine", page_icon="🎧", layout="wide")

pages = [
    st.Page("views/chat.py", title="Chat", icon="💬", default=True),
    st.Page("views/analytics.py", title="Analytics", icon="📊"),
    st.Page("views/how_it_works.py", title="How it works", icon="🧭"),
]
st.navigation(pages).run()
