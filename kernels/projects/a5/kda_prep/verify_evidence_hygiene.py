"""Check every reachable PR commit for oversized blobs and private path/IP text."""
import argparse
import hashlib
import ipaddress
import json
import re
import subprocess
from pathlib import Path


PRIVATE_HOME = re.compile(rb'/(?:data/)?home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/')
IPV4 = re.compile(rb'(?<![A-Za-z0-9_.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![A-Za-z0-9_.])')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', default='HEAD')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(repo), *arguments])

    base = git('rev-parse', '--verify', args.base + '^{commit}').decode().strip()
    head = git('rev-parse', '--verify', args.head + '^{commit}').decode().strip()
    commits = git('rev-list', base + '..' + head).decode().splitlines()
    blobs = {}
    for commit in commits:
        changed = git('diff-tree', '--no-commit-id', '-r', '-z', '--diff-filter=AM', commit).split(b'\0')
        for index in range(0, len(changed) - 1, 2):
            header, path = changed[index], changed[index + 1]
            if not header:
                continue
            object_id = header.split()[3].decode()
            blobs.setdefault(object_id, set()).add(path.decode())
    reader = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    violations, largest = [], 0
    try:
        for object_id, paths in blobs.items():
            reader.stdin.write((object_id + '\n').encode())
            reader.stdin.flush()
            header = reader.stdout.readline().decode().split()
            assert header[0] == object_id and header[1] == 'blob'
            size = int(header[2])
            data = reader.stdout.read(size)
            assert len(data) == size and reader.stdout.read(1) == b'\n'
            largest = max(largest, size)
            reasons = []
            if size > 5_000_000:
                reasons.append('blob_over_5_MB')
            if PRIVATE_HOME.search(data):
                reasons.append('private_home_path_including_tracebacks_or_logs')
            if any(ipaddress.ip_address(m.group().decode()).version == 4 for m in IPV4.finditer(data)
                   if all(int(part) <= 255 for part in m.group().split(b'.'))):
                reasons.append('IPv4_literal_requires_review')
            if reasons:
                violations.append(dict(object_id=object_id, paths=sorted(paths), reasons=reasons))
    finally:
        reader.stdin.close()
        reader.wait()
    report = dict(base=base, head=head, commits=len(commits), unique_blobs=len(blobs),
                  largest_blob_bytes=largest, passed=not violations, violations=violations,
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Reachable PR commits only. This does not erase or qualify previously published unreachable objects.')
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
