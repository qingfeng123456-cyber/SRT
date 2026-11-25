"""Checksum-only verification; imports no experiment or numerical modules."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(destination):
    destination = destination.resolve()
    expected = json.loads((ROOT / 'data/expected_source_manifest.json').read_text(encoding='utf-8'))
    assert expected['cutoff'] == '2021-12-31'
    for relative, item in expected['files'].items():
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination):
            raise ValueError('Manifest path leaves data directory: ' + relative)
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError('Missing or incorrectly sized data file: ' + relative)
        if digest(path) != item['sha256']:
            raise ValueError('Data checksum mismatch: ' + relative)
    for relative in expected['missing_source_files']:
        if (destination / relative).exists():
            raise ValueError('Recorded missing file is present: ' + relative)
    actual = json.loads((destination / 'source_manifest.json').read_text(encoding='utf-8'))
    if actual != expected:
        raise ValueError('source_manifest.json differs from the recorded manifest')
    return len(expected['files'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    print(json.dumps({'verified_files': verify(args.destination), 'cutoff': '2021-12-31',
                      'scope': 'checksums only; no model execution'}))
