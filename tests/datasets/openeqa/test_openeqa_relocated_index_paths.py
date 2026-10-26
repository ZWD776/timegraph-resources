import json
from pathlib import Path
import sqlite3

import faiss
import numpy as np

from vlkg.datasets.openeqa.runtime.visual_object_index import ObjectVisualSearchIndex


def test_runtime_manifest_can_reference_unchanged_binary(tmp_path):
    original = tmp_path / 'original'
    runtime = tmp_path / 'runtime'
    original.mkdir(); runtime.mkdir()
    binary = original / 'index.faiss'
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1., 0.]], dtype=np.float32))
    faiss.write_index(index, str(binary))
    metadata = runtime / 'rows.sqlite'
    with sqlite3.connect(metadata) as conn:
        conn.execute('CREATE TABLE object_metadata(faiss_id INTEGER PRIMARY KEY, episode_history TEXT)')
        conn.execute('INSERT INTO object_metadata VALUES (0, "episode")')
    manifest = {'index_path': str(binary), 'metadata_path': str(metadata)}
    (runtime / 'openeqa_object_siglip_flat_ip.manifest.json').write_text(json.dumps(manifest))
    search = ObjectVisualSearchIndex(index_dir=runtime, text_encoder=lambda texts: [[1., 0.] for _ in texts])
    try:
        assert search.index.ntotal == 1
        assert search.episode_size('episode') == 1
        assert search.index_path == binary
        assert not (runtime / 'openeqa_object_siglip_flat_ip.faiss').exists()
    finally:
        search.close()
