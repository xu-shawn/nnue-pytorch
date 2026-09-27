import concurrent.futures, hashlib, json, pathlib, time, urllib.request
root = pathlib.Path('/workspace/data'); root.mkdir(exist_ok=True)
items = json.load(open('/workspace/datasets.json'))
def download(item):
    start = time.time()
    url = 'https://huggingface.co/datasets/' + item['repo'] + '/resolve/main/' + item['file']
    path = root / item['file']
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as src, open(path, 'wb') as dst:
        while chunk := src.read(8*1024*1024):
            dst.write(chunk); digest.update(chunk)
    assert path.stat().st_size == item['size']
    assert digest.hexdigest() == item['sha256']
    print(item['file'], path.stat().st_size, time.time()-start, flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(download, items))
