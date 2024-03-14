# personal_llm_assistant

First install a new conda environment:
```
conda create --name assistant python=3.10
```

Activate the new conda env:
```
conda activate assistant
```

Run the ```install.sh``` bash script to install the required packages and libraries:
```
chmod +x install.sh
bash install.sh
```

Download your gguf model to serve:
```
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ./models/ --local-dir-use-symlinks False
```

Start the llm engine (based on your GPU available RAM, you might need to change the ```--n_gpu_layers``` parameter value):
```
python3 -m llama_cpp.server --model ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf --n_gpu_layers -1 --chat_format chatml
```

Finally, in another terminal, run the python code:
```
python gradio_app.py
```
