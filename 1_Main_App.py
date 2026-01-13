import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from chat_parser import preprocess_chat
from wordcloud import WordCloud
import emoji
from collections import Counter
from admin_panel import admin_panel

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")


# ================== WhatsApp Chat Analyzer ==================
def whatsapp_chat_analyzer():
    if not st.session_state.get("admin_logged_in", False):
        st.warning("⚠ You must log in as admin to access the Chat Analyzer.")
        return

    st.title("📱 WhatsApp Chat Analyzer")

    uploaded_file = st.file_uploader("Upload your WhatsApp chat file (.txt)", type="txt")

    if uploaded_file is not None:
        try:
            chat_data = uploaded_file.read().decode("utf-8")
            df = preprocess_chat(chat_data)

            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

            # Store chat DF for sentiment/toxicity pages
            st.session_state['chat_df'] = df

            # ================== Chat Preview ==================
            st.subheader("📄 Chat Preview")
            st.dataframe(df.head(20), use_container_width=True)

            # ================== Basic Statistics ==================
            st.subheader("📊 Basic Statistics")
            total_messages = df.shape[0]
            total_words = df['message'].apply(lambda x: len(x.split())).sum()
            total_media = df['message'].str.contains('<Media omitted>').sum()
            total_links = df['message'].str.contains(r'http[s]?://', regex=True).sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Messages", total_messages)
            col2.metric("Words", total_words)
            col3.metric("Media Shared", total_media)
            col4.metric("Links Shared", total_links)

            # ================== Navigation Buttons ==================
            st.subheader("⚙️ More Analysis Options")
            colA, colB = st.columns(2)

            with colA:
                if st.button("💬 Check Sentiment Analysis", key="sent_btn"):
                    st.session_state['go_to_page'] = "sentiment"

            with colB:
                if st.button("☠ Check Toxicity Analysis", key="tox_btn"):
                    st.session_state['go_to_page'] = "toxicity"

            # ================== EXTRA CHARTS ==================
            st.subheader("📆 Messages by Day of Week")
            df['day_name'] = df['date'].dt.day_name()
            fig, ax = plt.subplots()
            df['day_name'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

            st.subheader("🕒 Messages by Hour")
            df['hour'] = df['date'].dt.hour
            fig, ax = plt.subplots()
            df['hour'].value_counts().sort_index().plot(kind='line', marker='o', ax=ax)
            st.pyplot(fig)

            st.subheader("📅 Messages per Month")
            df['month'] = df['date'].dt.to_period('M')
            fig, ax = plt.subplots()
            df['month'].value_counts().sort_index().plot(kind='bar', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("✍ Message Length Distribution")
            df['msg_length'] = df['message'].apply(len)
            fig, ax = plt.subplots()
            sns.histplot(df['msg_length'], bins=20, kde=False, ax=ax)
            st.pyplot(fig)

            st.subheader("📈 Daily Messages Trend")
            daily_counts = df.groupby(df['date'].dt.date).size()
            st.line_chart(daily_counts)

            st.subheader("🔥 Weekly Activity Heatmap")
            heat_df = df.pivot_table(
                index=df['date'].dt.day_name(),
                columns=df['date'].dt.hour,
                values='message',
                aggfunc='count'
            ).fillna(0)
            fig, ax = plt.subplots(figsize=(10,4))
            sns.heatmap(heat_df, ax=ax)
            st.pyplot(fig)

            st.subheader("🗣 Most Common Words")
            words = []
            for msg in df['message']:
                if 'http' not in msg and not msg.startswith('<Media'):
                    words.extend(msg.lower().split())

            stop = {"the","a","to","is","and","i","you","of","in","for","on","me","my","it","this","that","so","at","be"}
            cleaned = [w for w in words if w not in stop]
            freq = Counter(cleaned).most_common(15)
            if freq:
                wc_df = pd.DataFrame(freq, columns=['Word','Count'])
                fig, ax = plt.subplots()
                sns.barplot(x='Count', y='Word', data=wc_df, ax=ax)
                st.pyplot(fig)

            st.subheader("📊 User Activity Over Time")
            df['day'] = df['date'].dt.date
            pivot = df.pivot_table(index='day', columns='user', values='message', aggfunc='count').fillna(0)
            st.area_chart(pivot)

            st.subheader("☁️ Word Cloud")
            text = " ".join(msg for msg in df['message'] if not msg.startswith('<Media') and 'http' not in msg)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            st.subheader("📊 Messages Distribution by User")
            fig, ax = plt.subplots()
            df['user'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.axis('equal')
            st.pyplot(fig)

            st.subheader("😊 Emoji Usage")
            def extract_emojis(s):
                return [c for c in s if c in emoji.EMOJI_DATA]
            all_emojis = []
            for message in df['message']:
                all_emojis += extract_emojis(message)

            st.write(f"Total Emojis Used: {len(all_emojis)}")

            if len(all_emojis):
                emo_df = pd.DataFrame(Counter(all_emojis).most_common(10), columns=['Emoji','Count'])
                fig, ax = plt.subplots()
                sns.barplot(x='Count', y='Emoji', data=emo_df, ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    # ================== PAGE REDIRECT LOGIC ==================
    if st.session_state.get('go_to_page') == "sentiment":
        st.switch_page("pages/2_Sentiment_Analysis")

    if st.session_state.get('go_to_page') == "toxicity":
        st.switch_page("pages/3_Toxicity_Analysis")


# ================== Main Navigation ==================
def main():
    menu = ["Chat Analyzer", "Admin Panel"]
    choice = st.sidebar.selectbox("📌 Navigation", menu)
    if choice == "Chat Analyzer":
        whatsapp_chat_analyzer()
    else:
        admin_panel()

if __name__ == "__main__":
    main()
