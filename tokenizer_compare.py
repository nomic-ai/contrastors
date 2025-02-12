import os
import pandas as pd
import plotly.graph_objects as go
import json
import tiktoken
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


languages = ["ar", "de", "en", "es", "fr", "hi", "ja", "ko", "ru", "zh"]
tokenizers = [
    "FacebookAI/xlm-roberta-base",
    "CohereForAI/c4ai-command-r-plus-08-2024",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    # most recent openai tokenizers use cl100k_base except 4o
    "gpt-4",
    "gpt-4o",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2-0.5B-Instruct"
]

tokens_per_lang = {}
for tok in tokenizers:
    for lang in languages:
        tokens_per_lang[tok] = {}

num_samples = 100_000
if not os.path.exists("tokenizer_data.json"):
    for tokenizer_name in tqdm(tokenizers):
        if "gpt" not in tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        print(f"Using {tokenizer_name=}")
        for lang in languages:
            print(f"Processing {lang=}")
            ds = load_dataset("allenai/c4", lang, split="validation", streaming=True)
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            if "gpt" not in tokenizer_name:
                ds = ds.map(lambda x: tokenizer(x["text"], add_special_tokens=False))
            else:
                ds = ds.map(lambda x: {"input_ids": tokenizer.encode(x["text"])})

            ds = ds.map(lambda x: {"num_tokens": len(x["input_ids"])})

            ds = list(ds.take(num_samples))

            tokens_per_lang[tokenizer_name][lang] = sum(x["num_tokens"] for x in ds)
    print(json.dumps(tokens_per_lang, indent=2))
    with open("tokenizer_data.json", "w") as f:
        json.dump(tokens_per_lang, f)
else:
    with open("tokenizer_data.json", "r") as f:
        tokens_per_lang = json.load(f)

# Calculate ratios
ratios = {tok: {} for tok in tokenizers if tok != "FacebookAI/xlm-roberta-base"}
for lang in languages:
    xlm_roberta_tokens = tokens_per_lang["FacebookAI/xlm-roberta-base"][lang]
    for tok in tokenizers:
        if tok != "FacebookAI/xlm-roberta-base":
            ratios[tok][lang] = tokens_per_lang[tok][lang] / xlm_roberta_tokens

# Prepare data for plotting
df = pd.DataFrame(ratios).T
df = df.reset_index().melt(id_vars=["index"], var_name="Language", value_name="Ratio")
df = df.rename(columns={"index": "Tokenizer"})

fig = go.Figure()

for tokenizer in df["Tokenizer"].unique():
    tokenizer_data = df[df["Tokenizer"] == tokenizer]
    fig.add_trace(
        go.Bar(
            x=tokenizer_data["Language"],
            y=tokenizer_data["Ratio"],
            name=tokenizer,
            text=tokenizer_data["Ratio"].round(2),
            textposition="auto",
        )
    )

# Add horizontal line at y=1
fig.add_shape(
    type="line",
    x0=0,
    y0=1,
    x1=1,
    y1=1,
    line=dict(
        color="red",
        width=2,
        dash="dash",
    ),
    xref='paper',
    yref='y'
)

fig.update_layout(
    title="Multilingual Token Cost (Ratio to XLM-RoBERTa)",
    xaxis_title="Language",
    yaxis_title="Tokens Required (Ratio)",
    barmode="group",
    legend_title="Tokenizer",
    template="plotly_dark",
    width=1200,  # Increase the width of the figure
    height=800,  # Increase the height of the figure
)

# Set font sizes
fig.update_layout(
    title_font_size=24,
    xaxis_title_font_size=18,
    yaxis_title_font_size=18,
    legend_title_font_size=18,
    font_size=14  # This sets the base font size for tick labels and legend text
)

# Add annotation for the reference line
fig.add_annotation(
    x=1.02,
    y=1,
    xref="paper",
    yref="y",
    text="XLM-RoBERTa Reference",
    showarrow=False,
    font=dict(size=12, color="red"),
    textangle=-90,
    xanchor="left",
    yanchor="middle"
)

# Increase the resolution
fig.write_image(f"tokenizer_compare_{num_samples=}.webp", scale=4, format="webp", engine="kaleido")

