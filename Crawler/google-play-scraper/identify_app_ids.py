import sys
sys.path.insert(0, "")
from google_play_scraper import search

for name in ["ruangguru", "quipper", "pahamify"]:
    print("\nSEARCH", name)
    results = search(name, n_hits=10)
    for item in results:
        print(item.get("appId"), "-", item.get("title"))
