import gradio as gr
from transformers import pipeline
from transformers import AutoProcessor, BarkModel
import torch
from openai import OpenAI
import numpy as np
from IPython.display import Audio, display
import numpy as np
import re
from nltk.tokenize import sent_tokenize


WORDS_PER_CHUNK = 25


# Setup Whisper client
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    torch_dtype=torch.float16,
    device="cuda:0"
)

voice_processor = AutoProcessor.from_pretrained("suno/bark")
voice_model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to("cuda:0")

voice_model =  voice_model.to_bettertransformer()
voice_preset = "v2/en_speaker_9"


system_prompt = "You are a helpful AI. You must answer the questino user asks briefly."


client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-xxx")  # Placeholder, replace 
sample_rate = 48000

def transcribe_and_query_llm_voice(audio_file_path):

    transcription = pipe(audio_file_path)['text']
    
    response = client.chat.completions.create(
        model="mistral",
        messages=[
            {"role": "system", "content": system_prompt},  # Update this as per your needs
            {"role": "user", "content": transcription}
        ],
    )
    llm_response = response.choices[0].message.content

    sampling_rate = voice_model.generation_config.sample_rate
    silence = np.zeros(int(0.25 * sampling_rate))

    BATCH_SIZE = 12
    model_input = sent_tokenize(llm_response)

    pieces = []
    for i in range(0, len(model_input), BATCH_SIZE):
        inputs = model_input[BATCH_SIZE*i:min(BATCH_SIZE*(i+1), len(model_input))]
        
        if len(inputs) != 0:
            inputs = voice_processor(inputs, voice_preset=voice_preset)
            
            speech_output, output_lengths = voice_model.generate(**inputs.to("cuda:0"), return_output_lengths=True, min_eos_p=0.2)
            
            speech_output = [output[:length].cpu().numpy() for (output,length) in zip(speech_output, output_lengths)]
            
            pieces += [*speech_output, silence.copy()]
        
        
    whole_ouput = np.concatenate(pieces)

    audio_output = (sampling_rate, whole_ouput) 

    return llm_response, audio_output


def transcribe_and_query_llm_text(text_input):

    transcription = text_input
    
    response = client.chat.completions.create(
        model="mistral",
        messages=[
            {"role": "system", "content": system_prompt},  # Update this as per your needs
            {"role": "user", "content": transcription + "\n Answer briefly."}
        ],
    )

    llm_response = response.choices[0].message.content

    sampling_rate = voice_model.generation_config.sample_rate
    silence = np.zeros(int(0.25 * sampling_rate))

    BATCH_SIZE = 12
    model_input = sent_tokenize(llm_response)

    pieces = []
    for i in range(0, len(model_input), BATCH_SIZE):
        inputs = model_input[BATCH_SIZE*i:min(BATCH_SIZE*(i+1), len(model_input))]
        
        if len(inputs) != 0:
            inputs = voice_processor(inputs, voice_preset=voice_preset)
            
            speech_output, output_lengths = voice_model.generate(**inputs.to("cuda:0"), return_output_lengths=True, min_eos_p=0.2)
            
            speech_output = [output[:length].cpu().numpy() for (output,length) in zip(speech_output, output_lengths)]
            
            
            pieces += [*speech_output, silence.copy()]
        
        
    whole_ouput = np.concatenate(pieces)

    audio_output = (sampling_rate, whole_ouput)  

    return llm_response, audio_output



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Type your request", placeholder="Type here or use the microphone...")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Or record your speech")
        with gr.Column():
            output_text = gr.Textbox(label="LLM Response")
            output_audio = gr.Audio(label="LLM Response as Speech", type="numpy")
    
    submit_btn_text = gr.Button("Submit Text")
    submit_btn_voice = gr.Button("Submit Voice")
    

    submit_btn_voice.click(fn=transcribe_and_query_llm_voice, inputs=[audio_input], outputs=[output_text, output_audio])
    submit_btn_text.click(fn=transcribe_and_query_llm_text, inputs=[text_input], outputs=[output_text, output_audio])

demo.launch(ssl_verify=False,
            share=False,
            debug=False)