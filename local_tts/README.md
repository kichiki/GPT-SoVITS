# local TTS script with GPT-SoVITS

This is a simple CLI script for TTS (Text-to-Speech) with GPT-SoVITS.

# Preparation

* you need to complete the set-up for GPT-SoVITS environment
* install additional packages 
<pre>
pip install sounddevice pyperclip
</pre>

* that is it.

* optionally, you may use your own fine-tuned models.
    * ex. my model is here: https://huggingface.co/ichiki/GPT-SoVITS-v2-kengo

# How to run

* run the script `local_tts/tts.py` like
<pre>
% python local_tts/tts.py 
✓ TTS models initialized            
✓ Model warm-up complete     

✓ Model ready for input

● Interactive TTS started in clipboard mode
i Monitoring clipboard. Press Ctrl+C to end
</pre>

# Help
<pre>
% python local_tts/tts.py -h
usage: tts.py [-h] [--gpt_model GPT_MODEL] [--sovits_model SOVITS_MODEL] [--ref_audio REF_AUDIO] [--ref_text REF_TEXT]
              [--ref_language {中文,英文,日文}] [--target_language {中文,英文,日文,中英混合,日英混合,多语种混合}] [--input_mode {stdin,clipboard}]
              [--max_chunk_size MAX_CHUNK_SIZE] [--speaking_rate SPEAKING_RATE] [--temperature TEMPERATURE]

Interactive GPT-SoVITS Text-to-Speech

options:
  -h, --help            show this help message and exit
  --gpt_model GPT_MODEL
                        Path to the GPT model file
  --sovits_model SOVITS_MODEL
                        Path to the SoVITS model file
  --ref_audio REF_AUDIO
                        Path to the reference audio file
  --ref_text REF_TEXT   Path to the reference text file
  --ref_language {中文,英文,日文}
                        Language of the reference audio
  --target_language {中文,英文,日文,中英混合,日英混合,多语种混合}
                        Language of the target text
  --input_mode {stdin,clipboard}
                        Input mode: 'stdin' for command line, 'clipboard' for clipboard monitoring
  --max_chunk_size MAX_CHUNK_SIZE
                        Max length of chunk for synthesis
  --speaking_rate SPEAKING_RATE
                        Speed factor for speech (0.8-1.5)
  --temperature TEMPERATURE
                        Temperature for generation (0.5-1.5)
</pre>

