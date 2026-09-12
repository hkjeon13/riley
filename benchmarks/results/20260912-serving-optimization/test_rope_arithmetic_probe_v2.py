"""CPU-only contract checks; synthetic v2 rows are never GPU evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


v1, v2 = module('rope_arithmetic_probe'), module('rope_arithmetic_probe_v2')


def synthetic_rows():
    rows = [json.loads(line) for line in (HERE/'raw/rope-arithmetic-probe/run/native.jsonl').read_text().splitlines()]
    for row in rows:
        row['schema'] = 'riley.rope-arithmetic-native.v2'
        if row['kind'] == 'case':
            for item in row['candidates']:
                item.update(equal=True, mismatch_words=0, region_mismatches=[0, 0, 0],
                            first_mismatch=None, mismatch_bit_pairs=[])
    pairs = [{'oracle_bits': 0, 'candidate_bits': 32768, 'count': 1},
             {'oracle_bits': 1, 'candidate_bits': 2, 'count': 2}]
    rows[1]['candidates'][0].update(equal=False, mismatch_words=3, region_mismatches=[3, 0, 0],
        first_mismatch={'region': 'q_rotary', 'word': 0, 'oracle_bits': 0, 'candidate_bits': 32768},
        mismatch_bit_pairs=pairs)
    rows[-1].update(candidate_mismatch_cases=[1, 0], candidate_mismatch_words=[3, 0],
                    candidate_equal=[False, True], candidate_mismatch_bit_pairs=[copy.deepcopy(pairs), []])
    return rows


def validate_rows(rows):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'synthetic-not-gpu.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return v2.validate_raw(path)


class Contracts(unittest.TestCase):
    def test_all_input_bytes_remain_identical(self):
        self.assertEqual(list(v1.cases()), list(v2.cases()))
        for index, phase, position, batch in v1.cases():
            self.assertEqual(v1.qkv_bytes(index, phase, batch), v2.qkv_bytes(index, phase, batch))
            self.assertEqual(v1.metadata_bytes(index, position), v2.metadata_bytes(index, position))
        for phase in range(7):
            for sine in (False, True):
                self.assertEqual(v1.table_bytes(phase, sine), v2.table_bytes(phase, sine))

    def test_histograms_preserve_signed_zero_and_nonzero_mismatches(self):
        result = validate_rows(synthetic_rows())
        self.assertEqual(result['mismatch_classification'][0], {
            'candidate': 0, 'mismatch_words': 3, 'signed_zero_only_words': 1,
            'other_mismatch_words': 2, 'all_mismatches_signed_zero': False})
        self.assertIsNone(result['mismatch_classification'][1]['all_mismatches_signed_zero'])
        rows = synthetic_rows()
        rows[1]['candidates'][0]['mismatch_bit_pairs'] = [{'oracle_bits': 0, 'candidate_bits': 32768, 'count': 3}]
        rows[-1]['candidate_mismatch_bit_pairs'][0] = copy.deepcopy(rows[1]['candidates'][0]['mismatch_bit_pairs'])
        self.assertTrue(validate_rows(rows)['mismatch_classification'][0]['all_mismatches_signed_zero'])
        self.assertFalse(validate_rows(rows)['summary']['candidate_equal'][0])

    def test_missing_duplicate_or_unreconciled_histogram_is_rejected(self):
        mutations = [
            lambda rows: rows[1]['candidates'][0].pop('mismatch_bit_pairs'),
            lambda rows: rows[1]['candidates'][0]['mismatch_bit_pairs'].append({'oracle_bits': 0, 'candidate_bits': 32768, 'count': 1}),
            lambda rows: rows[1]['candidates'][0]['mismatch_bit_pairs'][1].update(count=1),
            lambda rows: rows[-1]['candidate_mismatch_bit_pairs'][0][1].update(candidate_bits=3),
            lambda rows: rows[1]['candidates'][0]['mismatch_bit_pairs'][0].update(candidate_bits=0),
        ]
        for mutate in mutations:
            rows = synthetic_rows()
            mutate(rows)
            with self.assertRaises((ValueError, RuntimeError, KeyError)):
                validate_rows(rows)

    def test_actual_v1_basis_is_pinned_and_cannot_masquerade_as_v2(self):
        with patch.object(v2, 'HERE', HERE/'raw'):
            evidence = v2.basis_evidence()
        self.assertEqual(evidence['inspect/oracle-rope.sass']['sha256'],
                         evidence['inspect/production-0-rope.sass']['sha256'])
        with self.assertRaises((ValueError, RuntimeError)):
            v2.validate_raw(HERE/'raw/rope-arithmetic-probe/run/native.jsonl')

    def test_commands_preserve_precise_and_fast_math_flags_and_shared_runtime(self):
        compiled = json.loads((HERE/'raw/rope-arithmetic-probe/compile.json').read_text())
        manifest = {'source_root': '/isolated/source', 'tools': v2.tool_evidence()}
        commands = v2.compile_commands(manifest, Path('/cuda/bin/nvcc'), compiled['production_flags'], '/isolated/v2-build')
        self.assertEqual(commands[0][1:1+len(compiled['production_flags']['oracle_precise']['flags'])],
                         compiled['production_flags']['oracle_precise']['flags'])
        self.assertNotIn('--use_fast_math', commands[0])
        self.assertEqual(commands[1].count('--use_fast_math'), 1)
        self.assertIn(str(HERE/'rope_arithmetic_candidates_v2.cu'), commands[1])
        self.assertIn(str(HERE/'rope_arithmetic_probe_v2.cu'), commands[2])
        self.assertIn('--cudart=shared', commands[3])
        self.assertEqual(commands[3][-1], '/isolated/v2-build/rope_arithmetic_probe_v2')


if __name__ == '__main__':
    unittest.main()
