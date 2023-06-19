from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple
import gradio as gr
import librosa

REMOVE_COLS = ["lang_id", "english_transcription"]
LANGUAGE = "ko-KR"

minds = load_dataset("PolyAI/minds14", name=LANGUAGE, split="train")
minds = minds.remove_columns(REMOVE_COLS)

id2label = minds.features["intent_class"].int2str

def generate_audio() -> Tuple[Tuple[int, np.ndarray], Figure]:
    example = minds.shuffle()[0]
    sr = example["audio"]["sampling_rate"]
    audio_arr = example["audio"]["array"]

    fig,ax = plt.subplots(1,1)
    librosa.display.waveshow(audio_arr, sr=sr, ax=ax)
    return (sr, audio_arr), fig


iface = gr.Interface(
    fn=generate_audio,
    inputs=[],
    outputs=["audio", "plot"],
    title=f"Listen and see Audio in {LANGUAGE}"
).launch()

