"""Fetch the pinned Utonia source/weights as ordinary files (SMB has no symlinks)."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path('/ehsan/m.sadegh/simlingo_lidar')
COMMIT = 'de1f9deef4ee65e508d74347f8d0eb848d878abb'
REVISION = '9457f025e36c7001ee962a542f0d27a5b600e1e7'
WEIGHT_SHA = 'a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353'
SOURCE_SHA = '31be76ea81af90789f8759078b997eadb951cb3d18fdfb112cda902a74c5a1bd'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected=None):
    if path.exists() and (expected is None or sha256(path) == expected):
        return
    part = path.with_name(path.name + '.part')
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as stream:
        while block := response.read(8 * 1024 * 1024):
            if shutil.disk_usage(ROOT).free < 150 * 1024**3:
                raise RuntimeError('Shared filesystem free-space reserve reached')
            stream.write(block)
    actual = sha256(part)
    if expected and actual != expected:
        raise RuntimeError(f'Checksum mismatch for {path.name}: {actual}')
    part.replace(path)


def main():
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    (ROOT / 'source').mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'cache' / f'utonia-{COMMIT}.tar.gz'
    source = ROOT / 'source' / f'Utonia-{COMMIT}'
    download(f'https://codeload.github.com/Pointcept/Utonia/tar.gz/{COMMIT}', archive, SOURCE_SHA)
    if not source.exists():
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            for member in members:
                target = (ROOT / 'source' / member.name).resolve()
                if not target.is_relative_to((ROOT / 'source').resolve()):
                    raise RuntimeError('Unsafe source archive path')
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError('Source archive contains a link or special file')
            tar.extractall(ROOT / 'source', members=members)
    weight = ROOT / 'cache' / 'utonia.pth'
    download(f'https://huggingface.co/Pointcept/Utonia/resolve/{REVISION}/utonia.pth', weight, WEIGHT_SHA)
    report = {'source_commit': COMMIT, 'source_archive_sha256': sha256(archive),
              'source_directory': str(source), 'weight_revision': REVISION,
              'weight_path': str(weight), 'weight_sha256': sha256(weight)}
    (ROOT / 'reports' / 'utonia-assets.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
