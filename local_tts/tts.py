import argparse
import os
import sounddevice as sd
import threading
import time
import queue
import pyperclip

import contextlib

os.environ["TQDM_DISABLE"] = "1"

# to hide 'OMP' warnings, probably happening in the library
os.environ["OMP_NUM_THREADS"] = "1"

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS
import random
import numpy as np
import itertools
import sys
import math
import torch
from functools import partial


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def colored_text(text, color):
    """Wraps text with color codes"""
    return f"{color}{text}{Colors.RESET}"

def success(message):
    """Print a success message with green checkmark"""
    print(f"{colored_text('✓', Colors.GREEN)} {message}")

def warning(message):
    """Print a warning message with yellow exclamation mark"""
    print(f"{colored_text('!', Colors.YELLOW)} {message}")

def error(message):
    """Print an error message with red X"""
    print(f"{colored_text('✗', Colors.RED)} {message}")

def info(message):
    """Print an info message with blue dot"""
    print(f"{colored_text('•', Colors.BLUE)} {message}")

class Spinner:
    def __init__(self, message="Loading...", delay=0.1):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None

    def spin(self):
        while self.running:
            sys.stdout.write(f"\r{colored_text(next(self.spinner), Colors.CYAN)} {self.message}")
            sys.stdout.flush()  # Force flush stdout
            time.sleep(self.delay)

    def start(self):
        self.running = True
        # Clear any existing output on this line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()
        
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        # Small delay to ensure thread starts
        time.sleep(0.1)

    def stop(self, success_message=None):
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')  # Clear line
        if success_message:
            success(success_message)
        sys.stdout.flush()


i18n = I18nAuto()

dict_language_v2 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("粤语"): "all_yue",#全部按中文识别
    i18n("韩文"): "all_ko",#全部按韩文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("粤英混合"): "yue",#按粤英混合识别####不变
    i18n("韩英混合"): "ko",#按韩英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
}
dict_language = dict_language_v2


# Global queues for synthesis and playback
synthesis_queue = queue.Queue(maxsize=50)
playback_queue = queue.Queue()

# Global playback lock to ensure sequential playback
playback_lock = threading.Lock()

class GPT_SoVITS_TTS:
    def __init__(
        self,
        gpt_path,
        sovits_path,
        ref_audio_path,
        ref_text_path,
        ref_language,
        device='cpu',
        is_half=False,
        version='v2',
        seed=-1, keep_random=True,
    ):
        tts_config = TTS_Config()
        tts_config.device = device
        tts_config.is_half = is_half
        tts_config.version = version
        if gpt_path is not None:
            tts_config.t2s_weights_path = gpt_path
        if sovits_path is not None:
            tts_config.vits_weights_path = sovits_path

        tts_config.cnhuhbert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        tts_config.bert_base_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                # Store the original torch.load
                original_torch_load = torch.load

                # Create a patched version that always uses weights_only=False
                def load_with_code(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)

                torch.load = load_with_code

                try:
                    self.pipeline = TTS(tts_config)
                finally:
                    torch.load = original_torch_load

        # Read reference text
        if ref_text_path is None:
            ref_text = None
        else:
            with open(ref_text_path, 'r', encoding='utf-8') as file:
                ref_text = file.read()

        seed = -1 if keep_random else seed
        actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
        self.inputs = {
            "text": "",
            "text_lang": "",
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_text": ref_text,
            "prompt_lang": dict_language[i18n(ref_language)],
            "top_k": 5,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": 'cut1',
            "batch_size": 20,
            #"speed_factor": 1.0,
            "speed_factor": 1.1,
            "split_bucket": True,
            "return_fragment": True,
            "fragment_interval": 0.3,
            "seed": actual_seed,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
        }


# A simple function to split text into chunks of a maximum character length.
def split_text(text, max_chars=200):
    """
    Splits text into chunks, trying not to break sentences.
    """
    #words = text.split()
    delimilers = [' ', '\n', '.', '?', '!', ':', ';', '。', '、', '？', '！', '：', '；', '…']
    t = text
    for d in delimilers[1:]:
        t = t.replace(d, f'{d} ')
    words = t.split()

    is_first = True
    current_limit = 10
    chunks = []
    current_chunk = []
    current_len = 0
    for word in words:
        # Add one for the space.
        if is_first:
            if current_len > current_limit and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = len(word)

                current_limit = min(max_chars, current_limit*2)
                if current_limit > max_chars:
                    is_first = False
            else:
                current_chunk.append(word)
                current_len += len(word) + 1
        else:
            if current_len > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = len(word)
            else:
                current_chunk.append(word)
                current_len += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    print(f'*** {[len(c) for c in chunks]}')
    return chunks

def warm_up_model(tts: GPT_SoVITS_TTS, target_language):
    try:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                tts.inputs['text'] = "ツナマヨは人類の叡智だよ"
                tts.inputs['text_lang'] = dict_language[i18n(target_language)]
        return True
    except Exception as e:
        error(f"Warm-up failed: {e}")
        return False

def synthesis_worker(tts: GPT_SoVITS_TTS, target_language, synthesis_q, playback_q):
    while True:
        chunk_text = synthesis_q.get()
        if chunk_text is None:
            synthesis_q.task_done()
            break
        info(f"Synthesizing: {chunk_text[:30]}..." + ("" if len(chunk_text) <= 30 else ""))
        try:
            tts.inputs['text'] = chunk_text
            tts.inputs['text_lang'] = dict_language[i18n(target_language)]
            with open(os.devnull, 'w') as fnull:
                with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                    for block in tts.pipeline.run(tts.inputs):
                        playback_q.put(block)
        except Exception as e:
            print(f"Synthesis error for chunk '{chunk_text}': {e}")
        success(f"Completed")
        synthesis_q.task_done()

def playback_worker_continuous(playback_q, playback_lock):
    """
    Continuous playback worker that opens a persistent output stream using the
    parameters from the first audio block and then writes subsequent audio blocks
    continuously.
    """
    # Wait for the first audio block to initialize the stream.
    first_item = playback_q.get()
    if first_item is None:
        playback_q.task_done()
        return
    sampling_rate, audio_data = first_item
    playback_q.task_done()

    # Determine the number of channels.
    # If audio_data is 1D, assume mono; if 2D, assume shape (frames, channels).
    if audio_data.ndim == 1:
        channels = 1
        audio_data = audio_data.reshape(-1, 1)
    else:
        channels = audio_data.shape[1]

    # Create the output stream with the correct parameters.
    with sd.OutputStream(samplerate=sampling_rate, channels=channels, dtype=audio_data.dtype) as stream:
        # Write the first block.
        with playback_lock:
            stream.write(audio_data)

        # Now continuously process the rest of the audio blocks.
        while True:
            item = playback_q.get()
            if item is None:
                playback_q.task_done()
                break  # Sentinel to exit
            sr, block = item
            # Optional: Check if sr matches the initialized sampling_rate.
            if sr != sampling_rate:
                print("Warning: sampling rate changed from", sampling_rate, "to", sr)
                # Optionally, handle resampling here if needed.
            # Ensure block has the right shape (2D)
            if block.ndim == 1:
                block = block.reshape(-1, 1)
            with playback_lock:
                stream.write(block)
            playback_q.task_done()

import numpy as np
import sounddevice as sd

def create_ready_tone(duration=0.7, frequency=500, sample_rate=22050, volume=0.2):
    """
    Creates a notification tone with smooth fade-in and fade-out.
    
    Parameters:
    - duration: Length of tone in seconds
    - frequency: Tone frequency in Hz
    - sample_rate: Audio sample rate
    - volume: Maximum amplitude (0.0-1.0)
    
    Returns:
    - Numpy array with the tone waveform
    """
    # Create time array
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, False)
    #sine_90 = np.sin(np.linspace(0, np.pi/2, n, False))
    #f = (500 - 400) * sine_90 + 400
    f = np.linspace(400, 500, n, False)
    
    # Generate sine wave
    tone = np.sin(2 * np.pi * frequency * t)
    #tone = np.sin(2 * np.pi * f * t)

    dlf_oct = math.log(2)
    dlf_ht = dlf_oct / 12
    f_M3 = frequency * math.exp(dlf_ht * 4)
    tone_M3 = np.sin(2 * np.pi * f_M3 * t)

    f_p5 = frequency * math.exp(dlf_ht * 7)
    tone_p5 = np.sin(2 * np.pi * f_p5 * t)
    f_p5_drop = f_p5 / 2
    tone_p5_drop = np.sin(2 * np.pi * f_p5_drop * t)

    f_M7 = frequency * math.exp(dlf_ht * 11)
    tone_M7 = np.sin(2 * np.pi * f_M7 * t)

    f_bass = frequency / 2
    tone_bass = np.sin(2 * np.pi * f_bass * t)
    
    # Create fade envelope
    fade_in_samples = int(sample_rate * duration * 0.05)
    fade_out_samples = int(sample_rate * duration * 0.6)
    
    envelope = np.ones_like(tone)
    # Fade in: linear ramp from 0 to 1
    envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
    # Fade out: linear ramp from 1 to 0
    envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
    
    # Apply envelope and volume
    #tone = (tone/4.2 + tone_M3/4.5 + tone_p5/4. + tone_M7/4) * envelope * volume
    tone = (tone_bass/4.2 + tone_M3/4.5 + tone_p5_drop/4. + tone_M7/4) * envelope * volume
    
    return tone

def play_notification(frequency=440):
    """Plays a gentle notification tone to indicate the system is ready"""
    tone = create_ready_tone(frequency=frequency)
    sd.play(tone, 22050)
    sd.wait()  # Wait for the tone to finish playing

def main():
    # script should run on the top of the repository
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))

    try:
        parser = argparse.ArgumentParser(description="local Text-to-Speech script with GPT-SoVITS")
        parser.add_argument('--gpt_model', default='GPT_weights_v2/kengo-e15.ckpt', help="Path to the GPT model file")
        parser.add_argument('--sovits_model', default='SoVITS_weights_v2/kengo_e8_s184.pth', help="Path to the SoVITS model file")
        parser.add_argument('--ref_audio', default='local_tts/kengo-ref.wav', help="Path to the reference audio file")
        parser.add_argument('--ref_text', default='local_tts/kengo-ref.txt', help="Path to the reference text file")
        parser.add_argument('--ref_language', choices=["中文", "英文", "日文"], default='日文', help="Language of the reference audio")
        parser.add_argument('--target_language', choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], default='日英混合', help="Language of the target text")
        parser.add_argument('--input_mode', choices=['stdin', 'clipboard'], default='clipboard', help="Input mode: 'stdin' for command line, 'clipboard' for clipboard monitoring")
        parser.add_argument('--max_chunk_size', default='160', help="Max length of chunk for synthesis")
        parser.add_argument('--speaking_rate', type=float, default=1.1, 
                        help="Speed factor for speech (0.8-1.5)")
        parser.add_argument('--temperature', type=float, default=1.0,
                        help="Temperature for generation (0.5-1.5)")

        args = parser.parse_args()

        # Show a loading spinner during initialization
        spinner = Spinner("Initializing TTS models...")
        spinner.start()

        tts = GPT_SoVITS_TTS(
            args.gpt_model,
            args.sovits_model,
            args.ref_audio,
            args.ref_text,
            args.ref_language,
        )
        target_language = args.target_language
        input_mode = args.input_mode
        max_chunk_size = int(args.max_chunk_size)

        # Stop the first spinner
        spinner.stop("TTS models initialized")

        # Start a new spinner for warm-up
        spinner = Spinner("Warming up model...")
        spinner.start()

        # Warm up the model
        warm_up_result = warm_up_model(tts, target_language)

        # Stop the warm-up spinner
        if warm_up_result:
            spinner.stop("Model warm-up complete")
        else:
            spinner.stop()
            error("Model warm-up failed")
            return

    except Exception as e:
        spinner.stop()
        error(f"Failed to initialize models: {e}")
        return

    print("\n✓ Model ready for input")
    play_notification(frequency=500)
    
    # Start long-running worker threads once
    synth_thread = threading.Thread(
        target=synthesis_worker,
        args=(tts, target_language, synthesis_queue, playback_queue))
    play_thread = threading.Thread(
        target=playback_worker_continuous,
        args=(playback_queue, playback_lock))
    synth_thread.start()
    play_thread.start()

    print(f"\n{colored_text('●', Colors.GREEN)} Interactive TTS started in {colored_text(input_mode, Colors.CYAN)} mode")

    if input_mode == 'stdin':
        print(f"{colored_text('i', Colors.BLUE)} Type {colored_text('exit', Colors.YELLOW)} or {colored_text('quit', Colors.YELLOW)} to end")
        while True:
            full_text = input("> ")
            if full_text.lower() in ["exit", "quit"]:
                break
            chunks = split_text(full_text, max_chars=max_chunk_size)
            for chunk in chunks:
                synthesis_queue.put(chunk)
    elif input_mode == 'clipboard':
        last_clipboard_text = None
        print(f"{colored_text('i', Colors.BLUE)} Monitoring clipboard. Press {colored_text('Ctrl+C', Colors.YELLOW)} to end")
        try:
            while True:
                time.sleep(0.5)
                clipboard_text = pyperclip.paste().strip()
                if clipboard_text.lower() in ["exit", "quit"]:
                    break
                # Only process if new and non-empty text is found.
                if clipboard_text:
                    if last_clipboard_text is not None and clipboard_text != last_clipboard_text:
                        chunks = split_text(clipboard_text, max_chars=max_chunk_size)
                        for chunk in chunks:
                            synthesis_queue.put(chunk)
                    last_clipboard_text = clipboard_text
        except KeyboardInterrupt:
            print("\nExiting clipboard monitoring mode.")

    # closing notification
    play_notification(frequency=440)

    # Signal shutdown to the workers (by putting sentinels)
    synthesis_queue.put(None)  # for synthesis worker
    playback_queue.put(None)   # for playback worker
    synth_thread.join()
    play_thread.join()

if __name__ == '__main__':
    main()
