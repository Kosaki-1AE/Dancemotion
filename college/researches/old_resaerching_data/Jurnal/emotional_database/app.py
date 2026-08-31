import streamlit as st

from prompt_helper import classify_air, generate_mock_poem

st.set_page_config(page_title="音響文学AI（ローカル版）", page_icon="🎧")
st.title("🎧 音響文学AI - ローカルテンプレ版")
st.markdown("入力された文章の“空気”を読み取り、詩的な響きを返します。")

user_input = st.text_area("🌫️ あなたの文章を入力してください:")

if st.button("🌀 響きを生成"):
    if user_input.strip() == "":
        st.warning("文章を入力してください。")
    else:
        with st.spinner("空気を感じ取っています..."):
            tags = classify_air(user_input)
            poem = generate_mock_poem(tags)

            st.markdown("### 🌬 感じ取った空気感：")
            st.markdown(f"**{tags}**")

            st.markdown("### ✨ 響きの詩：")
            st.markdown(f"> {poem}")
