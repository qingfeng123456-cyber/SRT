"""Prepare the recorded public-data prefixes; imports no experiment modules."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from verify_data import verify

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Source and destination must be separate, non-nested directories')
    if (destination / 'qlib_cn').exists():
        raise FileExistsError('Destination qlib_cn already exists; select an empty destination')
    expected_path = ROOT / 'data/expected_source_manifest.json'
    expected = json.loads(expected_path.read_text(encoding='utf-8'))
    formats = json.loads((ROOT / 'docs/data_text_format.json').read_text(encoding='utf-8'))
    cutoff = expected['cutoff']
    assert cutoff == '2021-12-31'
    days = [day for day in (source / 'calendars/day.txt').read_text(encoding='utf-8').splitlines()
            if day <= cutoff]
    if len(days) != expected['calendar_dates'] or days != sorted(set(days)):
        raise ValueError('Provider calendar does not match the recorded snapshot')
    rows = [line.split() for line in (source / 'instruments/csi300.txt').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    rows = [(code.upper(), begin, min(end, cutoff)) for code, begin, end in rows
            if begin <= cutoff and end >= '2007-01-01']
    texts = {'qlib_cn/calendars/day.txt': '\n'.join(days) + '\n',
             'qlib_cn/instruments/csi300.txt': '\n'.join('\t'.join(row) for row in rows) + '\n'}
    destination.mkdir(parents=True, exist_ok=True)
    for relative, item in expected['files'].items():
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Manifest path leaves destination: ' + relative)
        if relative in texts:
            text = texts[relative]
            if formats[relative] == 'CRLF':
                text = text.replace('\n', '\r\n')
            payload = text.encode('utf-8')
        else:
            origin = source / Path(relative).relative_to('qlib_cn')
            with origin.open('rb') as stream:
                header = stream.read(4)
                if len(header) != 4:
                    raise ValueError('Truncated Qlib header: ' + relative)
                first = struct.unpack('<f', header)[0]
                start = int(first)
                values = item['bytes'] // 4 - 1
                if start != first or start < 0 or item['bytes'] % 4 or start + values > len(days):
                    raise ValueError('Recorded prefix exceeds the closed calendar: ' + relative)
                payload = header + stream.read(item['bytes'] - 4)
        if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Provider payload differs from the recorded snapshot: ' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (destination / 'source_manifest.json').write_bytes(expected_path.read_bytes())
    return verify(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    print(json.dumps({'prepared_files': prepare(args.source, args.destination),
                      'cutoff': '2021-12-31', 'scope': 'public-data prefixes only'}))
