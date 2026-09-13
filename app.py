"""社内情報特化型生成AI検索アプリ (Streamlit + LangChain + OpenAI API)"""
import json
import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import CONVERSATIONS_DIR, DATA_DIR, EMBEDDING_MODEL, PERSIST_DIR, RETRIEVER_K

MTG_CATEGORY = "MTG議事録"
NO_MATCH_MESSAGE = "該当する資料はありません。"
TITLE_MAX_LEN = 24

load_dotenv()

st.set_page_config(page_title="社内情報特化型検索AI", page_icon="🔍", layout="wide")

if not os.environ.get("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

CONVERSATIONS_DIR.mkdir(exist_ok=True)


# --- 会話履歴の永続化 ---
def list_conversations():
    conversations = []
    for path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            conversations.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(conversations, key=lambda c: c.get("updated_at", ""), reverse=True)


def make_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip().replace("\n", " ")
            return text[:TITLE_MAX_LEN] + ("…" if len(text) > TITLE_MAX_LEN else "")
    return "新しい会話"


def save_conversation(conversation_id, messages):
    if not messages:
        return
    path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    now = datetime.now().isoformat(timespec="seconds")
    created_at = now
    if path.exists():
        try:
            created_at = json.loads(path.read_text(encoding="utf-8")).get("created_at", now)
        except (json.JSONDecodeError, OSError):
            pass
    data = {
        "id": conversation_id,
        "title": make_title(messages),
        "created_at": created_at,
        "updated_at": now,
        "messages": messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_conversation(conversation_id):
    path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if path.exists():
        path.unlink()


def messages_to_chat_history(messages):
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def start_new_conversation():
    st.session_state.current_conversation_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_history = []


if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- カテゴリー ---
def list_categories():
    """data フォルダ直下のフォルダ名(カテゴリー)と、MTG議事録直下のフォルダ名(サブカテゴリー)を取得する。"""
    if not DATA_DIR.exists():
        return [], []

    top_categories = sorted(
        p.name for p in DATA_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    mtg_subcategories = []
    mtg_dir = DATA_DIR / MTG_CATEGORY
    if mtg_dir.exists():
        mtg_subcategories = sorted(
            p.name for p in mtg_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        if any(p.is_file() and not p.name.startswith(".") for p in mtg_dir.iterdir()):
            mtg_subcategories.append("その他")

    return top_categories, mtg_subcategories


def category_label(doc):
    category = doc.metadata.get("category", "その他")
    subcategory = doc.metadata.get("subcategory")
    if category == MTG_CATEGORY and subcategory:
        return f"{MTG_CATEGORY} > {subcategory}"
    return category


# --- サイドバー: 会話履歴 ---
st.sidebar.title("💬 会話履歴")
if st.sidebar.button("➕ 新しい会話", use_container_width=True):
    start_new_conversation()
    st.rerun()

conversations = list_conversations()
if conversations:
    for conv in conversations:
        is_current = conv["id"] == st.session_state.current_conversation_id
        label = ("▶ " if is_current else "") + conv.get("title", "新しい会話")
        if st.sidebar.button(
            label, key=f"conv_{conv['id']}", use_container_width=True, disabled=is_current
        ):
            st.session_state.current_conversation_id = conv["id"]
            st.session_state.messages = conv.get("messages", [])
            st.session_state.chat_history = messages_to_chat_history(conv.get("messages", []))
            st.rerun()
else:
    st.sidebar.caption("まだ会話履歴がありません。")

if st.sidebar.button("🗑️ この会話を削除", use_container_width=True):
    delete_conversation(st.session_state.current_conversation_id)
    start_new_conversation()
    st.rerun()

# --- サイドバー: カテゴリー ---
st.sidebar.divider()
st.sidebar.title("📁 カテゴリー")
st.sidebar.caption("チェックしたカテゴリーのみが回答の対象になります。")

top_categories, mtg_subcategories = list_categories()
non_mtg_categories = [c for c in top_categories if c != MTG_CATEGORY]
mtg_sub_keys = [f"mtgsub_{s}" for s in mtg_subcategories]
all_controlled_keys = [f"cat_{c}" for c in non_mtg_categories]
if MTG_CATEGORY in top_categories:
    all_controlled_keys.append(f"cat_{MTG_CATEGORY}")
all_controlled_keys += mtg_sub_keys


def on_toggle_all():
    value = st.session_state["cat_all"]
    for key in all_controlled_keys:
        st.session_state[key] = value


def on_toggle_mtg():
    value = st.session_state[f"cat_{MTG_CATEGORY}"]
    for key in mtg_sub_keys:
        st.session_state[key] = value


st.sidebar.checkbox("すべて", value=True, key="cat_all", on_change=on_toggle_all)

selected_categories = []
for category in non_mtg_categories:
    checked = st.sidebar.checkbox(category, value=True, key=f"cat_{category}")
    if checked:
        selected_categories.append(category)

selected_mtg_subcategories = []
if MTG_CATEGORY in top_categories:
    mtg_checked = st.sidebar.checkbox(
        MTG_CATEGORY, value=True, key=f"cat_{MTG_CATEGORY}", on_change=on_toggle_mtg
    )
    with st.sidebar.expander(f"📂 {MTG_CATEGORY} の詳細", expanded=False):
        for sub in mtg_subcategories:
            checked = st.checkbox(sub, value=True, key=f"mtgsub_{sub}")
            if checked:
                selected_mtg_subcategories.append(sub)

category_filter_conditions = [{"category": c} for c in selected_categories] + [
    {"$and": [{"category": MTG_CATEGORY}, {"subcategory": s}]} for s in selected_mtg_subcategories
]
has_category_selected = bool(category_filter_conditions)
if len(category_filter_conditions) == 1:
    category_filter = category_filter_conditions[0]
elif len(category_filter_conditions) > 1:
    category_filter = {"$or": category_filter_conditions}
else:
    category_filter = None

# --- サイドバー: 設定 ---
st.sidebar.divider()
st.sidebar.title("⚙️ 設定")

model_name = st.sidebar.selectbox("チャットモデル", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
k = st.sidebar.slider("検索件数 (k)", min_value=2, max_value=10, value=RETRIEVER_K)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.0, 0.1)

st.sidebar.divider()
if st.sidebar.button("🔄 インデックスを再作成 (data フォルダを再読み込み)"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.sidebar.error(".env ファイルに OPENAI_API_KEY を設定してください。")
    else:
        with st.spinner("data フォルダを読み込みインデックスを作成しています..."):
            from ingest import build_vectorstore

            build_vectorstore()
        st.cache_resource.clear()
        st.sidebar.success("インデックスを再作成しました。")

st.title("🔍 社内情報特化型生成AI検索アプリ")
st.caption("`data` フォルダ内の社内ドキュメント(PDF / Word / CSV / テキスト)をもとに質問へ回答します。")

if not os.environ.get("OPENAI_API_KEY"):
    st.error(".env ファイルに OPENAI_API_KEY を設定してからアプリを起動してください。")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not PERSIST_DIR.exists():
        return None
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


vectorstore = load_vectorstore()

if vectorstore is None:
    st.warning(
        "インデックスが未作成です。サイドバーの「インデックスを再作成」ボタンを押すか、"
        "ターミナルで `python ingest.py` を実行してください。"
    )
    st.stop()

if not has_category_selected:
    st.warning("サイドバーで少なくとも1つカテゴリーを選択してください。")

search_kwargs = {"k": k}
if category_filter is not None:
    search_kwargs["filter"] = category_filter
retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
llm = ChatOpenAI(model=model_name, temperature=temperature)

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "これまでの会話履歴と最新の質問を踏まえて、会話履歴が無くても意味が伝わる独立した質問文に"
            "書き換えてください。質問に答える必要はなく、書き換えのみ行ってください。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは社内情報に詳しい検索アシスタントです。以下の「参考情報」(社内ドキュメントの抜粋)"
            "のみを根拠に、日本語で簡潔かつ正確に回答してください。"
            f"参考情報に答えが含まれない場合や、質問がdataフォルダの内容と関係ない場合は、推測で答えず"
            f"「{NO_MATCH_MESSAGE}」とだけ回答してください。\n\n"
            "--- 参考情報 ---\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def render_reference(categories, sources):
    if categories:
        st.caption("参照カテゴリー: " + " / ".join(categories))
    if sources:
        with st.expander("参照元ドキュメント"):
            for s in sources:
                st.write(f"- {s}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_reference(msg.get("categories", []), msg.get("sources", []))

user_input = st.chat_input(
    "社内情報について質問してください(例: 経費精算のルールは？)",
    disabled=not has_category_selected,
)
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("社内ドキュメントを検索中..."):
            result = rag_chain.invoke(
                {"input": user_input, "chat_history": st.session_state.chat_history}
            )
            answer = result["answer"]
            context_docs = result.get("context", [])
            categories = sorted({category_label(doc) for doc in context_docs})
            sources = sorted({doc.metadata.get("source", "unknown") for doc in context_docs})
            if answer.strip() == NO_MATCH_MESSAGE:
                categories, sources = [], []
        st.markdown(answer)
        render_reference(categories, sources)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "categories": categories, "sources": sources}
    )
    save_conversation(st.session_state.current_conversation_id, st.session_state.messages)
    st.rerun()
