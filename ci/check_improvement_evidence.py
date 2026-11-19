#!/usr/bin/env python3
"""Verify improvement-plan links, evidence digests, and recorded distribution inputs."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


MANIFEST = Path("benchmarks/evidence/20261119-improvement-qualification/manifest.json")
PLANS = Path("docs/improvement-plan")
INLINE_LINK = re.compile(r"\[[^\]\n]+\]\((<[^>]+>|[^\s)]+)\)")
ATX_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def evidence_entries(value):
    if isinstance(value, dict):
        if "path" in value:
            yield value
        for child in value.values():
            yield from evidence_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_entries(child)


def check_evidence(root, manifest):
    count = 0
    for entry in evidence_entries(manifest):
        name, expected = entry.get("path"), entry.get("sha256")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError(f"invalid evidence path: {name!r}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid evidence SHA-256: {name}")
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError(f"evidence file missing from repository: {name}")
        with artifact.open("rb") as stream:
            actual = file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"evidence SHA-256 differs: {name}")
        count += 1
    if count == 0:
        raise ValueError("no checked evidence files in manifest")
    return count


def check_distribution_inputs(root, manifest):
    candidate = manifest["current_distribution_candidate"]
    revisions = (candidate["audited_main"], candidate["source_revision"])
    if any(not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
           for revision in revisions):
        raise ValueError("expected full Git revisions for the recorded distribution comparison")
    inputs = candidate["matching_distribution_input_git_objects"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("no recorded distribution inputs")
    expected = {}
    for path, object_id in inputs.items():
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"invalid distribution input path: {path!r}")
        if not isinstance(object_id, str) or not re.fullmatch(r"[0-9a-f]{40}", object_id):
            raise ValueError(f"invalid recorded Git object: {path}")
        for revision in revisions:
            expected[f"{revision}:{path}"] = object_id
    result = subprocess.run(["git", "rev-parse", *expected], cwd=root, capture_output=True, text=True)
    entries = result.stdout.splitlines()
    if result.returncode or len(entries) != len(expected):
        raise ValueError(f"unable to resolve the recorded distribution inputs: {result.stderr.strip()}")
    for (expression, object_id), actual in zip(expected.items(), entries, strict=True):
        if actual != object_id:
            raise ValueError(f"recorded distribution input differs: {expression}")
    return len(inputs)


def prose_lines(path):
    fence = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker.startswith(fence) and not line[match.end():].strip():
                fence = None
            continue
        if fence is None:
            yield number, line


def markdown_anchors(path):
    counts = {}
    for _, line in prose_lines(path):
        match = ATX_HEADING.match(line)
        if match:
            text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", match[1])
            slug = re.sub(r"[^\w -]", "", text.lower()).replace(" ", "-")
            ordinal = counts.get(slug, 0)
            counts[slug] = ordinal + 1
            yield f"{slug}-{ordinal}" if ordinal else slug


def check_plan_links(root):
    count = 0
    anchors = {}
    for plan in sorted((root / PLANS).glob("*.md")):
        for number, line in prose_lines(plan):
            for href in INLINE_LINK.findall(line):
                destination = urlsplit(href.strip("<>"))
                if destination.scheme or destination.netloc:
                    continue
                target = (plan.parent / unquote(destination.path)).resolve() if destination.path else plan
                location = f"{plan.relative_to(root)}:{number}"
                if not target.is_relative_to(root) or not target.exists():
                    raise ValueError(f"{location}: unresolved local link {href}")
                if destination.fragment and target.suffix.lower() == ".md":
                    if target not in anchors:
                        anchors[target] = set(markdown_anchors(target))
                    if unquote(destination.fragment) not in anchors[target]:
                        raise ValueError(f"{location}: unresolved heading in {href}")
                count += 1
    if count == 0:
        raise ValueError("no local improvement-plan links checked")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().repo_root.resolve()
    try:
        manifest = json.loads((root / MANIFEST).read_bytes(), object_pairs_hook=unique_object)
        files = check_evidence(root, manifest)
        inputs = check_distribution_inputs(root, manifest)
        links = check_plan_links(root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"improvement evidence check failed: {error}", file=sys.stderr)
        return 1
    print(f"improvement evidence verified: {files} file hashes, {inputs} recorded Git inputs, {links} local links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
