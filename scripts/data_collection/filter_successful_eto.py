"""Export original ETO records that won during a complete, verified replay audit.

This creates a separate success-only dataset; it does not change the source,
or the audit. Preparation reads this output via common.TRAIN_SOURCE.
"""
import argparse
import json
from pathlib import Path

from scripts.common import ETO_DATA, ETO_AUDIT_DIR, SUCCESSFUL_ETO_DIR, SUCCESSFUL_ETO_DATA, sha256_file, write_json


def select_successful(rows, audit):
    source_ids = [row['id'] for row in rows]
    audit_ids = [record['id'] for record in audit]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('duplicate source trajectory IDs')
    if len(set(audit_ids)) != len(audit_ids):
        raise ValueError('duplicate audit trajectory IDs')
    if set(source_ids) != set(audit_ids):
        raise ValueError('audit IDs must exactly cover the source IDs')
    for record in audit:
        if type(record.get('success')) is not bool:
            raise ValueError(f"{record['id']}: missing or invalid success flag")
    by_id = {record['id']: record for record in audit}
    return [
        row for row in rows
        if by_id[row['id']]['success'] is True
        and by_id[row['id']].get('reason') == 'won'
        and 'error' not in by_id[row['id']]
        and 'close_error' not in by_id[row['id']]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=ETO_DATA, help='original ETO JSON')
    parser.add_argument('--audit-dir', default=ETO_AUDIT_DIR,
                        help='complete audit directory with config, summary and replay')
    parser.add_argument('--out-dir', default=SUCCESSFUL_ETO_DIR,
                        help='new output directory; existing directories are never overwritten')
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    replay_path = audit_dir / 'replay.jsonl'
    config_path = audit_dir / 'config.json'
    summary_path = audit_dir / 'summary.json'
    with open(args.source) as stream:
        rows = json.load(stream)
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    audit = [json.loads(line) for line in replay_path.read_text().splitlines()]
    source_hash = sha256_file(args.source)
    replay_hash = sha256_file(str(replay_path))
    if config.get('source_sha256') != source_hash or summary.get('source_sha256') != source_hash:
        raise ValueError('audit source hash does not match the source dataset')
    if summary.get('replay_sha256') != replay_hash:
        raise ValueError('audit replay hash does not match summary')
    if (summary.get('complete_source') is not True
            or config.get('source_records') != len(rows)
            or summary.get('records') != len(rows)):
        raise ValueError('a complete source audit is required')
    selected = select_successful(rows, audit)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    dataset = out / Path(SUCCESSFUL_ETO_DATA).name
    write_json(str(dataset), selected)
    manifest = {
        'source': str(Path(args.source).resolve()),
        'source_sha256': source_hash,
        'audit': str(replay_path.resolve()),
        'audit_sha256': replay_hash,
        'audit_config_sha256': sha256_file(str(config_path)),
        'audit_summary_sha256': sha256_file(str(summary_path)),
        'script_sha256': sha256_file(__file__),
        'source_records': len(rows),
        'selected_records': len(selected),
        'excluded_records': len(rows) - len(selected),
        'selection_rule': 'success is true, reason is won, no error or close_error',
        'observation_mismatch_filtering': False,
        'record_rewriting': 'none; original record content and order preserved',
        'output_sha256': sha256_file(str(dataset)),
        'scope': 'Current-environment replay success only; not model evaluation or verified paper training data.',
    }
    write_json(str(out / 'manifest.json'), manifest)
    print(json.dumps({'source': len(rows), 'selected': len(selected),
                      'excluded': len(rows) - len(selected), 'output': str(dataset)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
