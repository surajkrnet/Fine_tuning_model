import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from datasets import Dataset

st.set_page_config(
    page_title="LLM Fine-Tuning Demo",
    layout="wide"
)

st.title("🤖 LLM Fine-Tuning Demo")
st.markdown(
    """
    Demonstrates:
    - Training Data
    - Base Model Output
    - Fine-Tuning
    - Training Loss
    - Fine-Tuned Output
    """
)

# ==================================================
# SIDEBAR PARAMETERS
# ==================================================

st.sidebar.header("Training Parameters")

learning_rate = st.sidebar.number_input(
    "Learning Rate",
    value=0.00005,
    format="%.6f",
    help="Controls how much the model changes after each training step."
)

epochs = st.sidebar.slider(
    "Epochs",
    min_value=1,
    max_value=30,
    value=10,
    help="Number of times the model reads the complete dataset."
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [1, 2, 4, 8],
    index=1,
    help="Number of samples processed before updating weights."
)

max_length = st.sidebar.selectbox(
    "Max Sequence Length",
    [32, 64, 128, 256],
    index=1,
    help="Maximum number of tokens seen per training example."
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Parameter Guide

**Learning Rate**
- How fast the model learns

**Epochs**
- Number of dataset passes

**Batch Size**
- Examples processed together

**Max Sequence Length**
- Maximum text length the model sees
""")

# ==================================================
# TRAINING DATA
# ==================================================

st.header("📚 Fine-Tuning Dataset")

default_data = """User: My order is late
Bot: Sorry for the delay. Please share your order ID.

User: I want refund
Bot: We can help process your refund request.

User: Where is my package?
Bot: Please send your tracking number.

User: Cancel my order
Bot: Sure. Please provide your order number.
"""

dataset_text = st.text_area(
    "Edit Training Data",
    value=default_data,
    height=250
)

# ==================================================
# PROMPT
# ==================================================

st.header("💬 Test Prompt")

prompt = st.text_area(
    "Prompt",
    value="User: My order is late\nBot:",
    height=120
)

# ==================================================
# MODEL LOADING
# ==================================================

@st.cache_resource
def load_base_model():

    model_name = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model

with st.spinner("Loading DistilGPT2..."):
    tokenizer, model = load_base_model()

# ==================================================
# BEFORE OUTPUT
# ==================================================

st.header("🔵 Before Fine-Tuning")

inputs = tokenizer(prompt, return_tensors="pt")

before_output = model.generate(
    **inputs,
    max_new_tokens=20
)

before_text = tokenizer.decode(
    before_output[0],
    skip_special_tokens=True
)

st.code(before_text)

# ==================================================
# TRAIN BUTTON
# ==================================================

if st.button("🚀 Start Fine-Tuning"):

    examples = []

    blocks = dataset_text.strip().split("\n\n")

    for block in blocks:
        block = block.strip()

        if block:
            examples.append({
                "text": block
            })

    dataset = Dataset.from_list(examples)

    def tokenize(example):

        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        tokens["labels"] = tokens["input_ids"].copy()

        return tokens

    tokenized_dataset = dataset.map(tokenize)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    progress = st.progress(0)

    with st.spinner("Training Model..."):

        trainer.train()

    progress.progress(100)

    # ============================================
    # LOSS DATA
    # ============================================

    losses = []

    for log in trainer.state.log_history:

        if "loss" in log:

            losses.append({
                "step": log["step"],
                "loss": log["loss"]
            })

    loss_df = pd.DataFrame(losses)

    st.header("📉 Training Loss")

    if len(loss_df) > 0:

        st.dataframe(
            loss_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(8,4))

        ax.plot(
            loss_df["step"],
            loss_df["loss"],
            marker="o"
        )

        ax.set_title("Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        st.pyplot(fig)

    # ============================================
    # AFTER OUTPUT
    # ============================================

    st.header("🟢 After Fine-Tuning")

    after_output = model.generate(
        **inputs,
        max_new_tokens=20
    )

    after_text = tokenizer.decode(
        after_output[0],
        skip_special_tokens=True
    )

    st.code(after_text)

    # ============================================
    # COMPARISON
    # ============================================

    st.header("⚖️ Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before")
        st.code(before_text)

    with col2:
        st.subheader("After")
        st.code(after_text)

    st.success(
        "Fine-tuning completed successfully."
    )

# ==================================================
# EDUCATIONAL NOTES
# ==================================================

with st.expander("📖 Fine-Tuning Concepts"):

    st.markdown("""
### Learning Rate
Controls how much the model changes after each training step.

### Epochs
Number of complete passes through the dataset.

### Batch Size
How many samples are processed before updating model weights.

### Max Sequence Length
Maximum text length the model can process at once.

### Loss
Measures model error. Lower loss generally means better learning.

### Fine-Tuning
Teaching a pre-trained model new behavior using custom examples.
""")
