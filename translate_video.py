"""
Command-line version (no web page). Same engine as app.py.
This is the one to run in Docker with --network none for the strongest isolation.

  python translate_video.py my_video.mp4                      # -> English
  python translate_video.py my_video.mp4 --target Bengali
  python translate_video.py my_video.mp4 --target fr --source ja --bilingual
  python translate_video.py --list-languages

Files saved next to the video:
  my_video.<spoken>.srt        what was said (original language)
  my_video.<target>.srt        the translation
  my_video.<spoken>-<target>.srt   both together (only with --bilingual)
"""
import pipeline   # FIRST: switches on the offline / local-only locks

import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Offline video subtitles: any language -> any language.")
    ap.add_argument("video", nargs="?", type=Path, help="Path to the video (or audio) file")
    ap.add_argument("--target", default="English", help="Language to translate INTO (name or code)")
    ap.add_argument("--source", default="auto", help="Spoken language (name or code); default: auto-detect")
    ap.add_argument("--whisper", help="Speech model: large-v3, medium or small (default: best downloaded)")
    ap.add_argument("--bilingual", action="store_true", help="Also save original + translation together")
    ap.add_argument("--list-languages", action="store_true")
    args = ap.parse_args()

    if args.list_languages:
        for _, nllb, name in sorted(pipeline.LANGUAGES, key=lambda x: x[2]):
            print(f"{name:24} {pipeline.short_code(nllb)}")
        return
    if args.video is None:
        ap.error("please give a video file")

    video = args.video.resolve()
    if not video.is_file():
        sys.exit(f"Video not found: {video}")

    available = pipeline.find_whisper_models()
    whisper_name = args.whisper or (available[0] if available else None)
    translator_dir = pipeline.find_translator()
    if whisper_name not in available or translator_dir is None:
        sys.exit("Models not found in ./models . Run 'python download_models.py' once (with internet).")

    try:
        target = pipeline.resolve_language(args.target)
        source = None
        if args.source.lower() != "auto":
            source = pipeline.short_code(pipeline.resolve_language(args.source))
            if source not in pipeline.WHISPER_NAMES:
                sys.exit(f"'{args.source}' cannot be used as a spoken language.")
    except pipeline.UnsupportedLanguage as e:
        sys.exit(str(e))

    def progress(stage, frac, msg):
        print(f"\r  {msg:<70}", end="", flush=True)

    started = time.time()
    segments, lang, conf = pipeline.transcribe(
        video, pipeline.MODELS_DIR / f"whisper-{whisper_name}", source, progress)
    print(f"\n  Spoken language: {pipeline.WHISPER_NAMES.get(lang, lang)} ({conf:.0%} sure)")
    if source is None and conf < 0.5:
        print("  Warning: detection is unsure. Use --source to set it yourself.")

    out_base = video.with_suffix("")
    tgt_short = pipeline.short_code(target)

    pipeline.translate(segments, lang, target, translator_dir, progress)
    print()
    files = {f"{out_base}.{lang}.srt": "src", f"{out_base}.{tgt_short}.srt": "text"}
    if args.bilingual:
        files[f"{out_base}.{lang}-{tgt_short}.srt"] = "both"
    for path, kind in files.items():
        Path(path).write_text(pipeline.to_srt(segments, kind), encoding="utf-8")
        print(f"  saved: {path}")
    print(f"Finished in {time.time() - started:.0f} s.")


if __name__ == "__main__":
    main()
