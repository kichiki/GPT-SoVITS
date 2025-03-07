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
  --device {cpu,cuda,mps}
                        either 'cpu', 'cuda', 'mps'
</pre>

# Misc

`GPT-SoVITS` のインストール、使い方については、一通り説明したビデオがあります。

* YouTube: [GPT-SoVITS でローカル TTS / 《復活》こんにちわ AI フォーラム・スペシャル 2025年2月28日](https://youtu.be/rrVvoFryYK0)

そこで喋っていた、学習済みモデルのダウンロードのシェルコマンドは、こちらになります：
<pre>
pip install -U "huggingface_hub[cli]"

# v2 models
huggingface-cli download lj1995/GPT-SoVITS \
chinese-hubert-base/config.json \
chinese-hubert-base/preprocessor_config.json \
chinese-hubert-base/pytorch_model.bin \
chinese-roberta-wwm-ext-large/config.json \
chinese-roberta-wwm-ext-large/pytorch_model.bin \
chinese-roberta-wwm-ext-large/tokenizer.json \
gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt \
gsv-v2final-pretrained/s2D2333k.pth \
gsv-v2final-pretrained/s2G2333k.pth \
--local-dir ./GPT_SoVITS/pretrained_models

# Faster Whisper
huggingface-cli download Systran/faster-whisper-large-v3 \
--local-dir ./tools/asr/models

# v3 models
huggingface-cli download lj1995/GPT-SoVITS s1v3.ckpt --local-dir ./GPT_SoVITS/pretrained_models
huggingface-cli download lj1995/GPT-SoVITS s2Gv3.pth --local-dir ./GPT_SoVITS/pretrained_models

huggingface-cli download lj1995/GPT-SoVITS models--nvidia--bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt --local-dir ./GPT_SoVITS/pretrained_models
huggingface-cli download lj1995/GPT-SoVITS models--nvidia--bigvgan_v2_24khz_100band_256x/config.json --local-dir ./GPT_SoVITS/pretrained_models
</pre>

参考にしてください。


# Update History

## 2025/03/07

* merge the latest `GPT-SoVITS` v3 release
  * minor fix for the latest code
    * on `GPT_SoVITS/text/LangSegmenter/langsegmenter.py`
* update `local_tts/tts.py`
  * add MPS optimization -- only T2S is still done on CPU
* update `local_tts/README.md`

## 2025/03/01

* Start the project `local_tts`
