"""
The engine: listen to a video (any language) -> text -> translate into the language you pick.
Used by both app.py (web page) and translate_video.py (command line).

IMPORTANT: import this file FIRST in your program. The moment it is imported it
  1) tells the AI libraries to stay offline, and
  2) blocks every network connection that is not to this same computer.
"""

# ---------------------------------------------------------------------------
# SECURITY LAYER 1: AI libraries must never go online.
# (Set BEFORE the libraries are imported, so this block must stay at the top.)
# ---------------------------------------------------------------------------
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# SECURITY LAYER 2: only talk to this same computer (127.0.0.1), nobody else.
# This is a safety net inside Python. It cannot stop programs written in C
# that open their own connections; for a guarantee at system level, run the
# command-line version in Docker with --network none (see Dockerfile).
# ---------------------------------------------------------------------------
import socket

_LOCAL_NAMES = {"127.0.0.1", "::1", "localhost"}


def _is_local(sock, address):
    if sock.family == getattr(socket, "AF_UNIX", None):
        return True
    return isinstance(address, tuple) and len(address) > 0 and address[0] in _LOCAL_NAMES


def _blocked():
    return RuntimeError("Blocked: this program may only talk to your own computer (127.0.0.1).")


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_sendto = socket.socket.sendto
_real_getaddrinfo = socket.getaddrinfo


def _guarded_connect(self, address):
    if not _is_local(self, address):
        raise _blocked()
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if not _is_local(self, address):
        raise _blocked()
    return _real_connect_ex(self, address)


def _guarded_sendto(self, data, *args):          # UDP
    if not _is_local(self, args[-1] if args else None):
        raise _blocked()
    return _real_sendto(self, data, *args)


def _guarded_getaddrinfo(host, *args, **kwargs):  # name lookups
    if host is not None and host not in _LOCAL_NAMES:
        raise _blocked()
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.socket.sendto = _guarded_sendto
socket.getaddrinfo = _guarded_getaddrinfo

# ---------------------------------------------------------------------------
import gc
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"

# (whisper code, NLLB code, name).  Whisper code "" = can be a target only.
# NLLB-200 is the translation model: it translates between all of these.
LANGUAGES = [
    ("af", "afr_Latn", "Afrikaans"), ("sq", "als_Latn", "Albanian"),
    ("am", "amh_Ethi", "Amharic"), ("ar", "arb_Arab", "Arabic"),
    ("hy", "hye_Armn", "Armenian"), ("as", "asm_Beng", "Assamese"),
    ("az", "azj_Latn", "Azerbaijani"), ("ba", "bak_Cyrl", "Bashkir"),
    ("eu", "eus_Latn", "Basque"), ("be", "bel_Cyrl", "Belarusian"),
    ("bn", "ben_Beng", "Bengali"), ("bs", "bos_Latn", "Bosnian"),
    ("bg", "bul_Cyrl", "Bulgarian"), ("my", "mya_Mymr", "Burmese"),
    ("yue", "yue_Hant", "Cantonese"), ("ca", "cat_Latn", "Catalan"),
    ("zh", "zho_Hans", "Chinese (Simplified)"), ("", "zho_Hant", "Chinese (Traditional)"),
    ("hr", "hrv_Latn", "Croatian"), ("cs", "ces_Latn", "Czech"),
    ("da", "dan_Latn", "Danish"), ("nl", "nld_Latn", "Dutch"),
    ("en", "eng_Latn", "English"), ("et", "est_Latn", "Estonian"),
    ("fo", "fao_Latn", "Faroese"), ("fi", "fin_Latn", "Finnish"),
    ("fr", "fra_Latn", "French"), ("gl", "glg_Latn", "Galician"),
    ("ka", "kat_Geor", "Georgian"), ("de", "deu_Latn", "German"),
    ("el", "ell_Grek", "Greek"), ("gu", "guj_Gujr", "Gujarati"),
    ("ht", "hat_Latn", "Haitian Creole"), ("ha", "hau_Latn", "Hausa"),
    ("he", "heb_Hebr", "Hebrew"), ("hi", "hin_Deva", "Hindi"),
    ("hu", "hun_Latn", "Hungarian"), ("is", "isl_Latn", "Icelandic"),
    ("id", "ind_Latn", "Indonesian"), ("it", "ita_Latn", "Italian"),
    ("ja", "jpn_Jpan", "Japanese"), ("jw", "jav_Latn", "Javanese"),
    ("kn", "kan_Knda", "Kannada"), ("kk", "kaz_Cyrl", "Kazakh"),
    ("km", "khm_Khmr", "Khmer"), ("ko", "kor_Hang", "Korean"),
    ("lo", "lao_Laoo", "Lao"), ("lv", "lvs_Latn", "Latvian"),
    ("ln", "lin_Latn", "Lingala"), ("lt", "lit_Latn", "Lithuanian"),
    ("lb", "ltz_Latn", "Luxembourgish"), ("mk", "mkd_Cyrl", "Macedonian"),
    ("mg", "plt_Latn", "Malagasy"), ("ms", "zsm_Latn", "Malay"),
    ("ml", "mal_Mlym", "Malayalam"), ("mt", "mlt_Latn", "Maltese"),
    ("mi", "mri_Latn", "Maori"), ("mr", "mar_Deva", "Marathi"),
    ("mn", "khk_Cyrl", "Mongolian"), ("ne", "npi_Deva", "Nepali"),
    ("no", "nob_Latn", "Norwegian"), ("ps", "pbt_Arab", "Pashto"),
    ("fa", "pes_Arab", "Persian"), ("pl", "pol_Latn", "Polish"),
    ("pt", "por_Latn", "Portuguese"), ("pa", "pan_Guru", "Punjabi"),
    ("ro", "ron_Latn", "Romanian"), ("ru", "rus_Cyrl", "Russian"),
    ("sa", "san_Deva", "Sanskrit"), ("sr", "srp_Cyrl", "Serbian"),
    ("sn", "sna_Latn", "Shona"), ("sd", "snd_Arab", "Sindhi"),
    ("si", "sin_Sinh", "Sinhala"), ("sk", "slk_Latn", "Slovak"),
    ("sl", "slv_Latn", "Slovenian"), ("so", "som_Latn", "Somali"),
    ("es", "spa_Latn", "Spanish"), ("su", "sun_Latn", "Sundanese"),
    ("sw", "swh_Latn", "Swahili"), ("sv", "swe_Latn", "Swedish"),
    ("tl", "tgl_Latn", "Tagalog"), ("tg", "tgk_Cyrl", "Tajik"),
    ("ta", "tam_Taml", "Tamil"), ("tt", "tat_Cyrl", "Tatar"),
    ("te", "tel_Telu", "Telugu"), ("th", "tha_Thai", "Thai"),
    ("bo", "bod_Tibt", "Tibetan"), ("tr", "tur_Latn", "Turkish"),
    ("tk", "tuk_Latn", "Turkmen"), ("uk", "ukr_Cyrl", "Ukrainian"),
    ("ur", "urd_Arab", "Urdu"), ("uz", "uzn_Latn", "Uzbek"),
    ("vi", "vie_Latn", "Vietnamese"), ("cy", "cym_Latn", "Welsh"),
    ("yi", "ydd_Hebr", "Yiddish"), ("yo", "yor_Latn", "Yoruba"),
]
# Not in the table because the translator has no code for them:
# Latin, Breton, Occitan, Hawaiian, Nynorsk.

WHISPER_TO_NLLB = {w: n for w, n, _ in LANGUAGES if w}
NLLB_NAMES = {n: name for _, n, name in LANGUAGES}
WHISPER_NAMES = {w: name for w, _, name in LANGUAGES if w}

WHISPER_ORDER = ["large-v3", "medium", "small"]      # best first
TRANSLATOR_ORDER = ["nllb-1.3b", "nllb-600m"]        # best first


class Cancelled(Exception):
    """The user pressed Cancel."""


class UnsupportedLanguage(Exception):
    pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def resolve_language(text: str) -> str:
    """'English', 'en' or 'eng_Latn'  ->  'eng_Latn'."""
    t = (text or "").strip().lower()
    for w, n, name in LANGUAGES:
        if t in (n.lower(), name.lower()) or (w and t == w):
            return n
    raise UnsupportedLanguage(f"Unknown language: '{text}'. Use --list-languages to see all.")


def short_code(nllb: str) -> str:
    """'eng_Latn' -> 'en' (for file names)."""
    for w, n, _ in LANGUAGES:
        if n == nllb:
            return w or nllb
    return nllb


def find_whisper_models(models_dir: Path = MODELS_DIR) -> list[str]:
    return [n for n in WHISPER_ORDER if (models_dir / f"whisper-{n}").is_dir()]


def find_translator(models_dir: Path = MODELS_DIR):
    for n in TRANSLATOR_ORDER:
        if (models_dir / n).is_dir():
            return models_dir / n
    return None


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(segments: list[dict], kind: str = "text") -> str:
    """kind: 'text' = translated, 'src' = original language, 'both' = both."""
    out = []
    for i, seg in enumerate(segments, start=1):
        if kind == "both":
            body = f"{seg['src']}\n{seg['text']}"
        else:
            body = seg[kind]
        out.append(f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{body}\n")
    return "\n".join(out)


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _report(progress, stage, fraction, message):
    if progress:
        progress(stage, max(0.0, min(1.0, fraction)), message)


_cuda_libs_done = False


def _preload_cuda_libs():
    """
    The NVIDIA helper libraries (cuBLAS, cuDNN) come with the PyTorch install, but they sit
    inside the environment folder (.../site-packages/nvidia/...), where the speech engine
    (ctranslate2) does not look. Load them here first; then the engine finds them already loaded.
    Safe to call when they are not there: it just does nothing.
    """
    global _cuda_libs_done
    if _cuda_libs_done:
        return
    _cuda_libs_done = True
    import ctypes
    import glob
    import importlib.util

    try:
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations or []) if spec else []
    except Exception:
        roots = []
    for name in ("libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9"):
        for root in roots:
            hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
            if hits:
                try:
                    ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    print(f"[cuda] could not load {hits[0]}: {e}", file=sys.stderr)
                break


def _decode_with_ffmpeg(path):
    """Read the sound with the ffmpeg program. Returns None if ffmpeg is not installed.
    Only local files are allowed (-protocol_whitelist file): it cannot open web addresses."""
    import shutil
    import subprocess
    import numpy as np

    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    proc = subprocess.run(
        [exe, "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", str(path),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg could not read it: " + proc.stderr.decode(errors="replace").strip()[:300])
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("ffmpeg found no sound in it")
    return audio


# ---------------------------------------------------------------------------
# PART 1: audio -> text in the ORIGINAL language (language found automatically)
# ---------------------------------------------------------------------------
def transcribe(audio_path, whisper_dir, source=None, progress=None, cancel=None):
    """
    source: Whisper language code ('zh', 'ja' ...) or None = detect automatically.
    Returns (segments, language_code, confidence 0..1).
    Each segment: {"start": s, "end": s, "src": "text in the original language"}
    """
    _preload_cuda_libs()
    import ctranslate2
    from faster_whisper import WhisperModel, decode_audio

    # Read the sound ONCE, on the CPU, before anything else. If the file cannot be read,
    # you get a clear message here instead of a confusing "graphics card failed".
    _report(progress, "transcribing", 0.0, "Reading the sound from the file ...")
    try:
        audio = decode_audio(str(audio_path), sampling_rate=16000)
    except Exception as first_error:
        # The 'av' package and faster-whisper sometimes do not match each other.
        # If the ffmpeg program is installed, use it instead.
        print(f"[audio] av could not read the file ({type(first_error).__name__}: {first_error}); "
              f"trying ffmpeg.", file=sys.stderr)
        try:
            audio = _decode_with_ffmpeg(audio_path)
        except Exception as e:
            raise RuntimeError(f"Could not read the sound from this file ({e}).") from e
        if audio is None:
            raise RuntimeError(
                "Could not read the sound from this file. The 'av' package does not work with this "
                f"faster-whisper ({type(first_error).__name__}: {first_error}). "
                "Fix: install ffmpeg (Ubuntu: sudo apt install ffmpeg), then start the app again; "
                "the app will use it automatically.") from first_error
    _check(cancel)

    def run(device, compute_type):
        _report(progress, "transcribing", 0.0, f"Loading speech model on {device.upper()} ...")
        model = WhisperModel(str(whisper_dir), device=device, compute_type=compute_type,
                             local_files_only=True)
        try:
            seg_iter, info = model.transcribe(
                audio,                              # sound already read above
                language=source,                    # None = detect from the audio
                task="transcribe",                  # write what was said, do NOT translate here
                vad_filter=True,                    # skip silence and music
                beam_size=5,
                condition_on_previous_text=False,   # stops the "same line repeated" problem
            )
            segs, total = [], (info.duration or 1.0)
            for seg in seg_iter:
                _check(cancel)
                text = seg.text.strip()
                if text:
                    segs.append({"start": seg.start, "end": seg.end, "src": text})
                _report(progress, "transcribing", seg.end / total,
                        f"Listening ... {len(segs)} lines so far")
            return segs, info
        finally:
            del model
            gc.collect()

    if ctranslate2.get_cuda_device_count() > 0:
        try:
            segs, info = run("cuda", "float16")
        except Cancelled:
            raise
        except Exception as e:                      # e.g. graphics card too new / missing libraries
            print(f"[whisper] graphics card failed ({type(e).__name__}: {e}); using CPU.",
                  file=sys.stderr)
            _report(progress, "transcribing", 0.0,
                    "Graphics card could not be used for listening; using CPU (slower) ...")
            segs, info = run("cpu", "int8")
    else:
        segs, info = run("cpu", "int8")

    if not segs:
        raise ValueError("No speech was found in this file.")
    return segs, info.language, float(info.language_probability or 0.0)


# ---------------------------------------------------------------------------
# PART 2: text -> text in the language you picked
# ---------------------------------------------------------------------------
def translate(segments, src_whisper, tgt_nllb, translator_dir, progress=None, cancel=None,
              batch_size=16):
    """Adds seg['text'] (translated) to every segment."""
    src_nllb = WHISPER_TO_NLLB.get(src_whisper)
    if src_nllb is None:
        raise UnsupportedLanguage(
            f"The speech was detected as '{src_whisper}', which the translator does not support.")
    if src_nllb == tgt_nllb:                        # already in the wanted language
        for s in segments:
            s["text"] = s["src"]
        return

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _report(progress, "translating", 0.0, f"Loading translation model on {device.upper()} ...")

    tokenizer = AutoTokenizer.from_pretrained(str(translator_dir), src_lang=src_nllb,
                                              local_files_only=True)
    tokenizer.src_lang = src_nllb
    for code in (src_nllb, tgt_nllb):
        if tokenizer.convert_tokens_to_ids(code) == tokenizer.unk_token_id:
            raise UnsupportedLanguage(f"This translation model does not know '{code}'.")
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

    model = AutoModelForSeq2SeqLM.from_pretrained(str(translator_dir), local_files_only=True)
    if device == "cuda":
        model = model.half()
    model = model.to(device).eval()

    lines = [s["src"] for s in segments]
    order = sorted(range(len(lines)), key=lambda i: len(lines[i]))   # similar lengths together
    result = [""] * len(lines)
    try:
        for start in range(0, len(order), batch_size):
            _check(cancel)
            idx = order[start:start + batch_size]
            inputs = tokenizer([lines[i] for i in idx], return_tensors="pt", padding=True,
                               truncation=True, max_length=256).to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, forced_bos_token_id=tgt_id,
                                     num_beams=4, max_new_tokens=256)
            for i, text in zip(idx, tokenizer.batch_decode(out, skip_special_tokens=True)):
                result[i] = text.strip()
            done = min(start + batch_size, len(order))
            _report(progress, "translating", done / len(order),
                    f"Translating ... {done}/{len(order)} lines")
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    for seg, text in zip(segments, result):
        seg["text"] = text