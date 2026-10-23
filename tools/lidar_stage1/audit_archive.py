"""Validate one sensor archive in place; extract only one small baseline fixture."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import laspy
import numpy as np
from PIL import Image

ARTIFACTS = Path('/ehsan/m.sadegh/simlingo_lidar')
DATASET = Path('/ehsan/m.sadegh/datasets/simlingo')
ARCHIVE = 'data_simlingo_lb1_split_routes_training_VehicleTurningRoute_chunk_002.tar.gz'


def check_fixture_coordinates():
    fixture = ARTIFACTS / 'audit' / 'fixture'
    xyz = np.load(fixture / 'points.npy').astype(np.float64)
    boxes = json.loads(gzip.decompress((fixture / 'boxes.json.gz').read_bytes()))
    rows = []
    for box in boxes:
        if box.get('num_points', -1) < 0 or not all(k in box for k in ('position', 'yaw', 'extent')):
            continue
        c, s = np.cos(box['yaw']), np.sin(box['yaw'])
        rotation = np.array([[c,-s,0], [s,c,0], [0,0,1]])
        local = (xyz - np.array(box['position'])) @ rotation
        count = int((np.abs(local) < np.array(box['extent'])).all(axis=1).sum())
        rows.append({'class':box['class'], 'position':box['position'],
                     'recorded_hits':box['num_points'], 'decoded_hits':count,
                     'difference':count - box['num_points']})
    report = {'note':'Small hit-count differences can follow 1 cm LAZ quantization; no mount transform applied', 'boxes':rows}
    (ARTIFACTS / 'reports/coordinate-check.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def audit_annotations(manifest, sensor_frames):
    coverage = {}
    for family in ('commentary', 'drivelm', 'dreamer'):
        matched = defaultdict(set)
        archives = [x for x in manifest['files'] if x['path'].startswith(family + '_') and 'VehicleTurningRoute_chunk_' in x['path']]
        for entry in archives:
            path = DATASET / 'archives' / entry['path']
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != entry['sha256']:
                raise RuntimeError(f'Annotation archive checksum failed: {path}')
            with tarfile.open(path, 'r|gz') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith('.json.gz'):
                        continue
                    name = PurePosixPath(member.name)
                    route = 'data/' + '/'.join(name.parts[1:-2])
                    if route not in sensor_frames:
                        continue
                    json.loads(gzip.decompress(archive.extractfile(member).read()))
                    frame = name.name.split('.')[0]
                    if frame in sensor_frames[route]['measurements']:
                        matched[route].add(frame)
        count = sum(map(len, matched.values()))
        if not count:
            raise RuntimeError(f'No matching {family} annotations; inspect archive/path correspondence')
        coverage[family] = {'archives_checked': [x['path'] for x in archives],
                            'matched_frames': count, 'matched_routes': len(matched),
                            'coverage_by_route': {route: len(ids) for route, ids in sorted(matched.items())}}
    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ARTIFACTS / 'reports' / 'archive-audit.json')
    parser.add_argument('--coordinate-only', action='store_true')
    args = parser.parse_args()
    if args.coordinate_only:
        print(json.dumps(check_fixture_coordinates(), indent=2))
        return
    manifest = json.loads((DATASET / 'download_manifest.json').read_text())
    entry = next(x for x in manifest['files'] if x['path'] == ARCHIVE)
    path = DATASET / 'archives' / ARCHIVE
    if path.stat().st_size != entry['size']:
        raise RuntimeError('Archive size differs from download manifest')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != entry['sha256']:
        raise RuntimeError('Archive SHA-256 differs from download manifest')
    print('Archive checksum verified', flush=True)
    frames = defaultdict(lambda: defaultdict(set))
    metadata = {}
    measurements = {}
    counts = Counter()
    scans = []
    extents = []
    fixture_route = None
    fixture = ARTIFACTS / 'audit' / 'fixture'
    fixture.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f'Unsafe archive member: {name}')
            if member.isdir():
                continue
            counts['files'] += 1
            data = archive.extractfile(member).read()
            if name.name == 'results.json.gz':
                metadata[str(name.parent)] = json.loads(gzip.decompress(data))
                continue
            kind = name.parent.name
            if kind not in ('rgb', 'rgb_augmented', 'lidar', 'measurements', 'boxes'):
                continue
            route, frame = str(name.parent.parent), name.name.split('.')[0]
            if frame in frames[route][kind]:
                raise RuntimeError(f'Duplicate frame: {name}')
            frames[route][kind].add(frame)
            counts[kind] += 1
            if fixture_route is None:
                fixture_route = route
            selected = route == fixture_route and frame == '0010'
            if kind in ('measurements', 'boxes'):
                decoded = json.loads(gzip.decompress(data))
                if kind == 'measurements':
                    if not np.isfinite(np.asarray(decoded['ego_matrix'])).all():
                        raise RuntimeError(f'Non-finite pose: {name}')
                    measurements.setdefault(route, set()).update(decoded)
            elif kind in ('rgb', 'rgb_augmented'):
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if image.size != (1024, 512):
                        raise RuntimeError(f'Unexpected image dimensions: {name}, {image.size}')
            elif kind == 'lidar':
                cloud = laspy.read(io.BytesIO(data))
                xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
                if len(xyz) == 0 or not np.isfinite(xyz).all():
                    raise RuntimeError(f'Empty/non-finite point cloud: {name}')
                if np.any(cloud.intensity):
                    raise RuntimeError(f'Unexpected meaningful intensity: {name}')
                scans.append({'route': route, 'frame': frame, 'points': len(xyz),
                              'front_points': int((xyz[:, 0] >= 0).sum()),
                              'rear_points': int((xyz[:, 0] < 0).sum())})
                extents.append([xyz.min(0).tolist(), xyz.max(0).tolist()])
                if selected:
                    np.save(fixture / 'points.npy', xyz.astype(np.float32))
                    (fixture / 'laz-header.json').write_text(json.dumps({
                        'scales': cloud.header.scales.tolist(),
                        'offsets': cloud.header.offsets.tolist(),
                        'point_format': cloud.header.point_format.id}, indent=2))
            if selected:
                (fixture / f'{kind}{".jpg" if kind.startswith("rgb") else ".laz" if kind == "lidar" else ".json.gz"}').write_bytes(data)
    # tar stops at end-of-archive markers; consume the enclosing gzip to verify its CRC/footer.
    with gzip.open(path, 'rb') as stream:
        while stream.read(8 * 1024**2):
            pass
    route_reports = []
    for route, kinds in sorted(frames.items()):
        reference = kinds['measurements']
        if not reference:
            raise RuntimeError(f'No measurements: {route}')
        for kind in ('rgb', 'lidar', 'boxes'):
            if kinds[kind] != reference:
                raise RuntimeError(f'Unmatched {kind} frames: {route}')
        indexes = sorted(map(int, reference))
        if indexes != list(range(0, indexes[-1] + 1)):
            raise RuntimeError(f'Non-contiguous frame IDs: {route}')
        result = metadata.get(route)
        eligible = False
        reason = 'missing results'
        if result:
            score = result['scores']
            allowed = len(result['infractions']['min_speed_infractions']) + len(result['infractions']['outside_route_lanes'])
            eligible = score['score_composed'] >= 100 or (score['score_route'] > 94 and result['num_infractions'] == allowed)
            reason = 'accepted' if eligible else 'existing route-quality filter'
        # Existing hist_len=1, pred_len=11, skip_first_n_frames=10 indexing.
        windows = len(range(10, len(indexes) - 11 - 1 - 1))
        route_reports.append({'route': route, 'frames': len(indexes), 'quality_pass': eligible,
                              'quality_reason': reason, 'eligible_windows': windows if eligible else 0})
    first = [s['points'] for s in scans if s['frame'] == '0000']
    later = [s['points'] for s in scans if s['frame'] != '0000']
    print('Sensor decoding complete; checking corresponding annotation archives', flush=True)
    annotation_coverage = audit_annotations(manifest, frames)
    check_fixture_coordinates()
    report = {'archive': str(path), 'sha256': digest.hexdigest(), 'integrity': 'SHA-256, tar members and gzip CRC verified',
              'dataset_revision': manifest['revision'], 'counts': dict(counts), 'route_count': len(frames),
              'quality_pass_routes': sum(r['quality_pass'] for r in route_reports),
              'eligible_windows_existing_indexing': sum(r['eligible_windows'] for r in route_reports),
              'point_count_initial_median': float(np.median(first)),
              'point_count_complete_median': float(np.median(later)),
              'point_count_complete_min_max': [min(later), max(later)],
              'xyz_min': np.min(np.asarray(extents)[:, 0], axis=0).tolist(),
              'xyz_max': np.max(np.asarray(extents)[:, 1], axis=0).tolist(),
              'complete_scans_with_rear_returns': sum(s['rear_points'] > 0 for s in scans if s['frame'] != '0000'),
              'cadence': '4 Hz inferred from collector 20 Hz / data_save_freq=5; no frame timestamps in measurements',
              'measurement_keys': sorted(set.union(*measurements.values())), 'routes': route_reports,
              'annotation_coverage': annotation_coverage,
              'fixture_route': fixture_route, 'fixture_frame': '0010', 'fixture_directory': str(fixture),
              'archive_extracted': False, 'free_bytes_after': shutil.disk_usage(ARTIFACTS).free}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('routes','measurement_keys','annotation_coverage')}, indent=2))


if __name__ == '__main__':
    main()
