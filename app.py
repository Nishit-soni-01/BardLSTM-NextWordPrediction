import streamlit as st
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Hamlet — Next Word Oracle",
    page_icon="🗡️",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700&family=Cinzel:wght@400;600&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');

        :root {
            --parchment: #f1e6cf;
            --parchment-dark: #e6d7b8;
            --ink: #241c15;
            --ink-soft: #4a3f33;
            --wine: #5c1a1a;
            --wine-bright: #7a2323;
            --gold: #b8912f;
        }

        html, body, [class*="css"] {
            font-family: 'EB Garamond', serif;
        }

        /* Overall app background: dark stone with soft vignette */
        .stApp {
            background: radial-gradient(ellipse at top, #241f2b 0%, #100d13 70%);
            color: var(--parchment);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1520 0%, #0d0a10 100%);
            border-right: 1px solid #3a2f22;
        }
        section[data-testid="stSidebar"] * {
            color: var(--parchment-dark) !important;
            font-family: 'EB Garamond', serif;
        }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            font-family: 'Cinzel', serif !important;
            color: var(--gold) !important;
            letter-spacing: 1px;
        }

        /* Title block */
        .hamlet-header {
            text-align: center;
            padding: 1.2rem 0 0.6rem 0;
            border-bottom: 1px solid #4a3f33;
            margin-bottom: 1.4rem;
        }
        .hamlet-title {
            font-family: 'Cinzel Decorative', serif;
            font-size: 2.4rem;
            color: var(--gold);
            text-shadow: 0 0 18px rgba(184, 145, 47, 0.35);
            margin-bottom: 0.2rem;
        }
        .hamlet-subtitle {
            font-family: 'Cinzel', serif;
            font-size: 0.95rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #a89b7f;
        }
        .hamlet-quote {
            font-style: italic;
            color: #cbb994;
            font-size: 1.05rem;
            margin-top: 0.8rem;
        }
        .hamlet-quote-attr {
            font-family: 'Cinzel', serif;
            font-size: 0.78rem;
            letter-spacing: 2px;
            color: #7d7060;
            margin-top: 0.15rem;
        }

        /* Input label */
        .stTextInput label {
            font-family: 'Cinzel', serif !important;
            color: var(--gold) !important;
            font-size: 1rem !important;
            letter-spacing: 1px;
        }

        /* Input box styled like parchment */
        .stTextInput > div > div > input {
            background-color: var(--parchment);
            color: var(--ink);
            font-family: 'EB Garamond', serif;
            font-size: 1.15rem;
            border: 1px solid var(--gold);
            border-radius: 4px;
            padding: 0.6rem 0.8rem;
        }
        .stTextInput > div > div > input:focus {
            box-shadow: 0 0 0 2px var(--gold);
        }

        /* Buttons */
        .stButton > button {
            font-family: 'Cinzel', serif;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            background: linear-gradient(180deg, var(--wine-bright) 0%, var(--wine) 100%);
            color: var(--parchment);
            border: 1px solid var(--gold);
            border-radius: 4px;
            padding: 0.55rem 1.6rem;
            font-size: 0.9rem;
            width: 100%;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            background: linear-gradient(180deg, var(--wine) 0%, #3d1010 100%);
            border-color: #d9b25a;
            box-shadow: 0 0 12px rgba(184, 145, 47, 0.4);
            transform: translateY(-1px);
        }

        /* Result card */
        .result-card {
            margin-top: 1.6rem;
            background: linear-gradient(135deg, #2a2330 0%, #1c1720 100%);
            border: 1px solid var(--gold);
            border-radius: 8px;
            padding: 1.4rem 1.6rem;
            text-align: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        }
        .result-label {
            font-family: 'Cinzel', serif;
            font-size: 0.8rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #a89b7f;
        }
        .result-word {
            font-family: 'Cinzel Decorative', serif;
            font-size: 2.2rem;
            color: var(--gold);
            margin: 0.3rem 0;
        }
        .result-sequence {
            font-style: italic;
            color: #cbb994;
            font-size: 1.05rem;
        }
        .result-sequence .predicted {
            color: var(--gold);
            font-weight: 600;
            font-style: normal;
        }

        /* Divider ornament */
        .ornament {
            text-align: center;
            color: var(--gold);
            font-size: 1.1rem;
            letter-spacing: 8px;
            margin: 1.4rem 0;
            opacity: 0.6;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model & tokenizer (cached so the castle doesn't rebuild on every click)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Summoning the ghost of the LSTM...")
def load_assets():
    model = load_model("next_word_lstm.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


model, tokenizer = load_assets()

# ----------------------------------------------------------------------------
# Prediction logic (unchanged)
# ----------------------------------------------------------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


# ----------------------------------------------------------------------------
# Sidebar — lore & controls
# ----------------------------------------------------------------------------
FAMOUS_LINES = [
    "To be, or not to be, that is the question",
    "Something is rotten in the state of Denmark",
    "Though this be madness, yet there is method in't",
    "The lady doth protest too much, methinks",
    "Brevity is the soul of wit",
    "There is nothing either good or bad, but thinking makes it so",
    "Frailty, thy name is woman",
    "What a piece of work is a man",
    "The rest is silence",
    "Alas, poor Yorick! I knew him, Horatio",
]

with st.sidebar:
    st.markdown("## 🕯️ Elsinore Archive")
    st.markdown(
        "This oracle was trained upon the full text of **Shakespeare's "
        "*Hamlet*** using an LSTM network, and whispers the word it "
        "believes should follow yours."
    )
    st.markdown("---")
    st.markdown("### Try a line from the play")
    chosen_line = st.selectbox(
        "Famous lines",
        options=["— choose a line —"] + FAMOUS_LINES,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚰️ About the Model")
    st.markdown(
        f"- **Vocabulary:** {len(tokenizer.word_index):,} words\n"
        f"- **Input length:** {model.input_shape[1]} tokens\n"
        "- **Architecture:** LSTM (with early stopping)"
    )
    st.markdown("---")
    st.caption("Act well your part, there all the honour lies.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
random_quote = random.choice(FAMOUS_LINES)
st.markdown(
    f"""
    <div class="hamlet-header">
        <div class="hamlet-title">🗡️ Hamlet — Next Word Oracle</div>
        <div class="hamlet-subtitle">An LSTM Divination Chamber</div>
        <div class="hamlet-quote">"{random_quote}"</div>
        <div class="hamlet-quote-attr">— William Shakespeare, Hamlet</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Main interaction
# ----------------------------------------------------------------------------
default_text = (
    chosen_line if chosen_line != "— choose a line —" else "To be or not to"
)

input_text = st.text_input(
    "Speak thy sequence of words, and the oracle shall divine the next:",
    value=default_text,
)

st.markdown('<div class="ornament">❧ ❧ ❧</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("⚔️ Consult the Oracle")

if predict_clicked:
    if not input_text.strip():
        st.warning("Speak first, good sir or lady — the oracle needs words to divine from.")
    else:
        with st.spinner("The ghost stirs in the machine..."):
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

        if next_word:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">The Oracle Foretells</div>
                    <div class="result-word">{next_word}</div>
                    <div class="result-sequence">
                        "{input_text} <span class="predicted">{next_word}</span>"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(
                "The spirits are silent... no word could be divined. "
                "Try a phrase more familiar to Elsinore's halls."
            )
