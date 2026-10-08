"""Narrow provenance reconciliation; original archived metadata remains unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTUAL = '61402ebc42e9d0ea5bde4e6415a1a9c25e3db8a1'
REPORTED = '7a8409177c9efc210b7271963e109a26a62022b8'
BINARY = '23820eb2df5006c2a522625083a86247a46731f8312cee6072be35be1be26fd6'

def reconciliation():
    receipt = json.loads((ROOT / 'kernel-batch05-source-metadata-reconciliation-attempt01.json').read_text())
    source_path = ROOT / 'kernel-batch05-terminal-source-binary-reverification-attempt01.json'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == receipt['source_reverification_sha256']
    source = json.loads(source_path.read_text())
    original = json.loads((ROOT / 'kernel-batch05-source-receipt-attempt01.json').read_text())
    assert source['source_commit'] == original['source_commit'] == receipt['actual_source_commit'] == ACTUAL
    assert source['files'] == original['files'] and len(source['files']) == source['verified_true_git_blobs'] == 464
    assert source['source_clean'] and source['binary_sha256'] == receipt['actual_binary_sha256'] == BINARY
    assert receipt['reported_source_commits'] == {'v52': '06302d8d8396b8f2f4996fec8595bbfa1dcd7450', 'candidate': REPORTED}
    return receipt

def require_original_archive(archive):
    assert hashlib.sha256(Path(archive).read_bytes()).hexdigest() == reconciliation()['archive_sha256']

def reconcile(prep, plan, binary_receipt):
    receipt = reconciliation()
    assert prep['source_commits'] == receipt['reported_source_commits']
    assert plan['candidate_source_commit'] == binary_receipt['source_commit'] == ACTUAL
    assert plan['candidate_binary_sha256'] == binary_receipt['sha256'] == BINARY
    assert prep['pins']['/data/riley-serving-261007/kernel-batch05-target-attempt01/release/riley'] == BINARY
    return {'field': 'preparation.source_commits.candidate', 'original_reported_value': REPORTED,
            'actual_source_from_plan_build_binary_and_true_blobs': ACTUAL,
            'actual_binary_sha256': BINARY, 'provenance_reconciliation': receipt,
            'original_metadata_and_raw_archive_modified': False}
