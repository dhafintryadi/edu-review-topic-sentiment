import sys
sys.path.insert(0, "")
from google_play_scraper import app

candidates = [
    "com.ruangguru.tv",
    "com.quipper.android",
    "com.pahamify.pahamify",
]
for aid in candidates:
    print("TRY", aid)
    try:
        info = app(aid, lang="en", country="id")
        print("OK", info.get("title"))
    except Exception as exc:
        print("ERR", type(exc).__name__, exc)
