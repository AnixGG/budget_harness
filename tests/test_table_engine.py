"""Contract checks including exact arithmetic and independent SQLite queries."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from harness.table_engine import execute


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'sales.csv').write_text(
            'product,code,units,price\nA,007,2,0.1\nA,7,3,0.2\nB,007,4,2.5\n')
        self.q = dict(tool='table_query', table='sales.csv', schema={'units': 'number', 'price': 'number'},
                      aggregates=[{'as': 'total', 'op': 'sum', 'column': 'units'}])

    def run_query(self, **changes):
        return execute(dict(self.q, **changes), self.root)

    def test_exact_derived_sum(self):
        r = self.run_query(filters=[{'column': 'product', 'op': 'eq', 'value': 'A'}],
                           derive=[{'as': 'revenue', 'op': 'mul', 'left': 'units', 'right': 'price'}],
                           aggregates=[{'as': 'total', 'op': 'sum', 'column': 'revenue'}])
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['result'], [{'total': '4/5'}])
        self.assertEqual((r['scanned_rows'], r['matched_rows']), (3, 2))

    def test_text_codes_stay_distinct(self):
        r = self.run_query(filters=[{'column': 'code', 'op': 'eq', 'value': '007'}])
        self.assertEqual(r['result'], [{'total': '6'}])

    def test_string_alias_matches_text_without_mutating_request(self):
        q = dict(self.q, schema={'units': 'number', 'code': 'string'},
                 filters=[{'column': 'code', 'op': 'eq', 'value': '007'}])
        aliased = execute(q, self.root)
        canonical = execute(dict(q, schema={'units': 'number', 'code': 'text'}), self.root)
        self.assertEqual(aliased, canonical)
        self.assertEqual(aliased['result'], [{'total': '6'}])
        self.assertEqual(q['schema']['code'], 'string')

    def test_string_alias_does_not_enable_numeric_aggregates_on_text(self):
        self.assertFalse(self.run_query(schema={'units': 'string'})['ok'])

    def test_sqlite_oracle_group_sort_limit(self):
        q = dict(self.q, group_by=['product'], sort=[{'column': 'total', 'descending': True}], limit=1)
        r = execute(q, self.root)
        with closing(sqlite3.connect(':memory:')) as db:
            db.execute('create table sales (product text, units integer)')
            db.executemany('insert into sales values (?, ?)', [('A', 2), ('A', 3), ('B', 4)])
            expected = [{'product': p, 'total': str(t)} for p, t in db.execute(
                'select product, sum(units) as total from sales group by product order by total desc limit 1')]
        self.assertEqual(r['result'], expected)
        self.assertTrue(r['truncated'])
        self.assertEqual(r['total_groups'], 2)

    def test_numeric_sort_not_lexical(self):
        (self.root / 'sales.csv').write_text('product,units\nA,2\nB,10\n')
        self.q['schema'] = {'units': 'number'}
        self.assertEqual(self.run_query(group_by=['product'], sort=[{'column': 'total'}])['result'],
                         [{'product': 'A', 'total': '2'}, {'product': 'B', 'total': '10'}])

    def test_mean_and_chained_derive(self):
        r = self.run_query(derive=[{'as': 'r', 'op': 'mul', 'left': 'units', 'right': 'price'},
                                  {'as': 'r2', 'op': 'add', 'left': 'r', 'right': 'r'}],
                           aggregates=[{'as': 'avg', 'op': 'mean', 'column': 'r2'}])
        self.assertEqual(r['result'], [{'avg': '36/5'}])

    def test_missing_error_exclude_and_irrelevant_cells(self):
        (self.root / 'sales.csv').write_text('product,units,unused\nA,2,\nB,,\n')
        self.q['schema'] = {'units': 'number'}
        self.assertFalse(self.run_query()['ok'])
        self.assertEqual(self.run_query(missing='exclude')['result'], [{'total': '2'}])
        self.assertEqual(self.run_query(filters=[{'column': 'product', 'op': 'eq', 'value': 'A'}])['result'],
                         [{'total': '2'}])

    def test_missing_filter_and_false_is_order_independent(self):
        (self.root / 'sales.csv').write_text('product,units\nB,\nA,2\n')
        self.q['schema'] = {'units': 'number'}
        filters = [{'column': 'units', 'op': 'gt', 'value': '1'},
                   {'column': 'product', 'op': 'eq', 'value': 'A'}]
        self.assertEqual(self.run_query(filters=filters), self.run_query(filters=list(reversed(filters))))
        self.assertTrue(self.run_query(filters=filters)['ok'])

    def test_empty_sum_zero_mean_error(self):
        f = [{'column': 'product', 'op': 'eq', 'value': 'absent'}]
        self.assertEqual(self.run_query(filters=f)['result'], [{'total': '0'}])
        self.assertFalse(self.run_query(filters=f, aggregates=[{'as': 'avg', 'op': 'mean', 'column': 'units'}])['ok'])

    def test_unknown_columns_bad_types_and_schema(self):
        for changes in [dict(schema={'units': 'guess'}), dict(group_by=['absent']),
                        dict(aggregates=[{'op': 'sum', 'as': 'total', 'column': 'code'}]),
                        dict(aggregates=[3]), dict(limit=True), dict(extra='ignored?')]:
            with self.subTest(changes=changes):
                self.assertFalse(self.run_query(**changes)['ok'])

    def test_duplicate_headers_ragged_invalid_number(self):
        for data in ['units,units\n1,2\n', 'units\n1,2\n', 'units\nNaN\n', 'units\n1e9999\n']:
            (self.root / 'sales.csv').write_text(data)
            self.assertFalse(execute(dict(self.q, schema={'units': 'number'}), self.root)['ok'])

    def test_no_path_escape_or_links(self):
        (self.root / 'link.csv').symlink_to(self.root / 'sales.csv')
        (self.root / 'sub').symlink_to(self.root, target_is_directory=True)
        for name in ['../sales.csv', '/tmp/a.csv', 'link.csv', 'sub/sales.csv']:
            self.assertFalse(self.run_query(table=name)['ok'])

    def test_division_zero_no_partial_answer(self):
        (self.root / 'sales.csv').write_text('units,price\n2,1\n4,0\n')
        r = self.run_query(derive=[{'as': 'ratio', 'op': 'truediv', 'left': 'units', 'right': 'price'}])
        self.assertFalse(r['ok'])
        self.assertIsNone(r['result'])

    def test_semicolon_bom(self):
        (self.root / 'sales.csv').write_text('\ufeffproduct;units\nA;2\n')
        self.assertEqual(self.run_query(delimiter=';', schema={'units': 'number'})['result'], [{'total': '2'}])

    def test_backend_boundary_and_unavailable_k(self):
        self.assertEqual(execute(self.q, self.root, backend='python'), self.run_query())
        r = execute(self.q, self.root, backend='k')
        self.assertFalse(r['ok'])
        self.assertEqual(r['error'], 'Backend unavailable')

    def test_count_min_max_and_empty_groups(self):
        r = self.run_query(aggregates=[{'as': 'n', 'op': 'count'},
                                      {'as': 'low', 'op': 'min', 'column': 'units'},
                                      {'as': 'high', 'op': 'max', 'column': 'units'}])
        self.assertEqual(r['result'], [{'n': '3', 'low': '2', 'high': '4'}])
        self.assertEqual(self.run_query(group_by=['product'], filters=[
            {'column': 'product', 'op': 'eq', 'value': 'absent'}])['result'], [])

    def test_no_executable_expression(self):
        r = self.run_query(derive=[{'as': 'x', 'op': 'eval', 'left': 'units', 'right': 'price'}])
        self.assertFalse(r['ok'])
        self.assertIsNone(r['result'])

    def test_cli_stdout_json_and_exit_codes(self):
        command = [sys.executable, '-m', 'harness.table_cli', '--workspace', str(self.root)]
        r = subprocess.run(command, input=json.dumps(self.q), text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['result'], [{'total': '9'}])
        r = subprocess.run(command, input='{', text=True, capture_output=True)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(json.loads(r.stdout)['ok'])


if __name__ == '__main__':
    unittest.main()
