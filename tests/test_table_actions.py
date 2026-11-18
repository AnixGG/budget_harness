"""Check direct query submission, explicit fallback and error boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

from harness import agent, core
from harness.table_actions import translate
from harness.query_policy import scope as query_scope


class DirectTests(unittest.TestCase):
    def run_action(self, action, data='product,units\nA,2\nB,10\n'):
        translated,kind=translate(json.dumps(action))
        self.assertEqual(kind,'table_query')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'sales.csv').write_text(data)
            with query_scope(['sales.csv'],p):
                obs=agent.run_python(json.loads(translated)['code'],p,core.Config(),True)
            return obs

    def action(self, **query):
        return dict(action='table_query',query=dict(table='sales.csv',schema={'units':'number'},
            aggregates=[{'as':'total','op':'sum','column':'units'}],**query),column='total')

    def test_scalar_submits_without_another_model_turn(self):
        obs=self.run_action(self.action())
        self.assertEqual(obs['terminal'],{'action':'answer','answer':'12'})

    def test_top_group_can_select_explicit_top_one(self):
        a=self.action(group_by=['product'],sort=[{'column':'total','descending':True}],limit=1)
        a['column']='product'
        self.assertEqual(self.run_action(a)['terminal']['answer'],'B')

    def test_multiple_rows_and_missing_data_never_submit(self):
        for a,data in [(self.action(group_by=['product']),'product,units\nA,2\nB,10\n'),
                       (self.action(),'product,units\nA,\n')]:
            obs=self.run_action(a,data)
            self.assertNotEqual(obs['exit_code'],0)
            self.assertIsNone(obs['terminal'])

    def test_exact_fraction(self):
        a=self.action();a['query']['aggregates'][0]['op']='mean'
        self.assertEqual(self.run_action(a,'product,units\nA,1\nB,2\n')['terminal']['answer'],'3/2')

    def test_model_string_schema_submits_exact_text_code(self):
        a=self.action()
        a['query']['schema']['code']='string'
        a['query']['filters']=[{'column':'code','op':'eq','value':'007'}]
        obs=self.run_action(a,'product,code,units\nA,007,2\nB,7,10\n')
        self.assertEqual(obs['terminal'],{'action':'answer','answer':'2'})

    def test_python_fallback_explicit_skip_finishes_without_answer(self):
        raw=json.dumps({'action':'python','reason':'required value unknown',
                        'code':"skip_task('missing price')"})
        action,kind=translate(raw)
        self.assertEqual(kind,'fallback')
        with tempfile.TemporaryDirectory() as tmp:
            obs=agent.run_python(json.loads(action)['code'],Path(tmp),core.Config(),True)
            self.assertEqual(obs['terminal'],{'action':'skip','reason':'missing price'})
            self.assertFalse((Path(tmp)/'answer.txt').exists())

    def test_fractional_revenue_direct_action(self):
        a=self.action()
        a['query']['schema']['price']='number'
        a['query']['derive']=[{'as':'revenue','op':'mul','left':'units','right':'price'}]
        a['query']['aggregates'][0]['column']='revenue'
        obs=self.run_action(a,'product,units,price\nA,2,0.1\nB,3,0.2\n')
        self.assertEqual(obs['terminal'],{'action':'answer','answer':'4/5'})

    def test_fallback_requires_reason(self):
        action={'action':'python','code':'submit_answer(42)'}
        self.assertEqual(translate(json.dumps(action))[1],'invalid')
        action['reason']='unsupported window operation'
        text,kind=translate(json.dumps(action))
        self.assertEqual(kind,'fallback')
        self.assertEqual(json.loads(text),{'action':'python','code':'submit_answer(42)'})

    def test_bare_answer_and_markdown_rejected(self):
        self.assertEqual(translate('{"action":"answer","answer":"42"}')[1],'invalid')
        self.assertEqual(translate('```json\n{}\n```')[1],'invalid')

    def test_skip_remains_explicit(self):
        raw='{"action":"skip","reason":"required value unknown"}'
        self.assertEqual(translate(raw),(raw,'skip'))


if __name__=='__main__':unittest.main()
