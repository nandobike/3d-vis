"""
Check that kernel data never changes without a version bump.

data/versions.lock.json records the sha256 of every data file for each released
version ("n2_77k@1.1.0", "structures@1.0.0", ...).

    python tools/data_lock.py           # check; exit 1 on any problem
    python tools/data_lock.py --record  # record versions not in the lock yet

--record only adds versions that are new in data/manifest.toml. It never overwrites a
recorded version: if a file changed, bump the version in the manifest (and describe the
change in data/CHANGELOG.md) instead.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kernel_data  # noqa: E402


def check(manifest, lock):
    problems, new = [], []
    pinned = manifest['structures']['version']
    for kernel in manifest['kernels']:
        if kernel['structures'] != pinned:
            problems.append(f"{kernel['key']}@{kernel['version']} was computed on structures "
                            f"{kernel['structures']}, but the library is {pinned}")
    for entry, files in kernel_data.versioned_files(manifest).items():
        current = {f: kernel_data.file_sha256(f) for f in files}
        if entry not in lock:
            new.append((entry, current))
            continue
        for f, digest in current.items():
            if lock[entry].get(f) != digest:
                problems.append(f'{f} changed but {entry} is already released: bump the version')
    return problems, new


def main():
    record = '--record' in sys.argv
    manifest = kernel_data.load_manifest()
    lock = json.loads(kernel_data.LOCK.read_text()) if kernel_data.LOCK.exists() else {}
    problems, new = check(manifest, lock)
    for p in problems:
        print(f'ERROR: {p}')
    for entry, _ in new:
        print(f'{"recording" if record else "UNRECORDED"}: {entry}')
    if record and not problems and new:
        lock.update(dict(new))
        kernel_data.LOCK.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + '\n')
    if problems or (new and not record):
        sys.exit(1)
    print('OK')


if __name__ == '__main__':
    main()
