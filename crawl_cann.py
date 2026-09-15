#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recursively crawl Huawei CANN Kit HarmonyOS guide pages and save as markdown."""
import os
import re
import sys
import time
import urllib.request
import urllib.parse

BASE = "https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/"
START = BASE + "cann-kit-guide"
OUT = "/home/ma-user/workspace/fdh/mobiinfer/cann_web_doc"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def fetch(url, retries=3):
    """Fetch url (appending .md) and return text."""
    if url.endswith("/"):
        url = url[:-1]
    md_url = url + ".md"
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(md_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1 + i)
    raise RuntimeError("fetch failed for %s: %s" % (url, last_err))

def normalize(url):
    """Strip fragment/query and trailing slash."""
    u = urllib.parse.urlsplit(url)
    path = u.path.rstrip("/")
    if not path:
        return None
    return urllib.parse.urlunsplit((u.scheme, u.netloc, path, "", ""))

def is_in_scope(url):
    """Keep only CANN Kit pages under harmonyos-guides."""
    u = urllib.parse.urlsplit(url)
    if u.netloc != "developer.huawei.com":
        return False
    path = u.path.strip("/")
    if not path.startswith("consumer/cn/doc/harmonyos-guides/"):
        return False
    slug = path[len("consumer/cn/doc/harmonyos-guides/"):]
    if slug == "cann-kit-guide":
        return True
    return slug.startswith("cannkit-")

def file_name(url):
    """Map a doc URL to a local .md file name."""
    u = urllib.parse.urlsplit(url)
    path = u.path.strip("/")
    slug = path[len("consumer/cn/doc/harmonyos-guides/"):]
    return slug.replace("/", "__") + ".md"

LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")

def url_from_file(fn):
    slug = fn[:-3].replace("__", "/")
    return BASE + slug

def main():
    os.makedirs(OUT, exist_ok=True)
    queue = [START]
    visited = set()
    failed = []

    # resume: mark already-downloaded non-empty files as visited
    existing = []
    for fn in sorted(os.listdir(OUT)):
        path = os.path.join(OUT, fn)
        if fn.endswith(".md") and os.path.getsize(path) > 0:
            existing.append(fn)
            visited.add(url_from_file(fn))

    # rediscover links from already-downloaded pages so unseen pages get queued
    for fn in existing:
        path = os.path.join(OUT, fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for m in LINK_RE.finditer(text):
            target = normalize(m.group(2))
            if target and is_in_scope(target) and target not in visited:
                queue.append(target)

    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        fn = file_name(url)
        print("[fetch] %s -> %s" % (url, fn), flush=True)
        try:
            text = fetch(url)
        except Exception as e:  # noqa: BLE001
            print("[fail] %s: %s" % (url, e), flush=True)
            failed.append((url, str(e)))
            continue

        # discover subpage links (absolute doc URLs), in scope
        for m in LINK_RE.finditer(text):
            target = normalize(m.group(2))
            if target and is_in_scope(target) and target not in visited:
                queue.append(target)

        # remove a possible leading "# " H1 that duplicates the file name?
        # keep content as-is; save now, rewrite later.
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
            f.write(text)

    print("crawled %d pages" % len(visited), flush=True)

    # second pass: rewrite internal doc links to relative local .md files
    downloaded = set(os.listdir(OUT))
    rewritten = 0
    for fn in sorted(downloaded):
        path = os.path.join(OUT, fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        def repl(m):
            nonlocal rewritten
            label, link = m.group(1), m.group(2)
            target = normalize(link)
            if target and is_in_scope(target):
                local = file_name(target)
                if local in downloaded:
                    rewritten += 1
                    return "[%s](./%s)" % (label, local)
            return m.group(0)

        text = LINK_RE.sub(repl, text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    print("rewrote %d internal links" % rewritten, flush=True)
    print("total files: %d" % len(downloaded), flush=True)

    if failed:
        print("FAILED PAGES:", flush=True)
        for u, e in failed:
            print("  %s : %s" % (u, e), flush=True)
        sys.exit(2)

if __name__ == "__main__":
    main()
