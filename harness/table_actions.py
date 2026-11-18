"""Direct table JSON contract and deterministic action compilation."""
import json

SYSTEM = '''Solve the data task using only its public files. Return ONE JSON action, no fences.
For supported table operations you MUST use action table_query, not Python or a guessed answer:
{"action":"table_query","query":{"table":"sales.csv","schema":{"units":"number"},"aggregates":[{"as":"total","op":"sum","column":"units"}]},"column":"total"}
The executor reads the FULL CSV, computes, selects column from the single result row and
submits immediately. Numeric answers are exact strings (integer or reduced fraction p/q).
Decimal prices and exact fractional revenue ARE supported: declare units and price as
number, derive revenue=mul(units,price), then sum revenue. Use table_query, not Python.
Unspecified columns are text; codes 007 and 7 remain distinct. Numeric columns need schema:number.
query supports filters:[{column,op:eq|ne|lt|le|gt|ge,value:string}] (AND),
derive:[{as,op:add|sub|mul|truediv,left:column,right:column}], group_by:[column],
aggregates:[{as,op:sum|mean|min|max,column}] or {as,op:count},
sort:[{column,descending:bool}], limit:1..100, delimiter comma/semicolon/tab.
Order: filters -> derive -> group/aggregate -> sort -> limit. No arbitrary expressions.
For top group: sort by aggregate then group key for ties, set limit:1 and select group column.
missing:error is default. missing:exclude is allowed ONLY if the task explicitly excludes
unknown rows. Empty sum/count=0; empty mean/min/max errors. Empty cells are unknown values.
If a needed value is unknown with no default, return {"action":"skip","reason":"..."}.
For unsupported semantics (joins, latest-per-key/version, dates, custom expressions or output
formatting), return {"action":"python","reason":"unsupported operation","code":"..."}.
That Python action must compute and call submit_answer(value) or write answer.txt immediately.
If Python discovers missing required data, call skip_task(reason); print('skip') and exit()
do not declare a skip. Never substitute zero or exclude unknown rows without task permission.
Never answer from previews only, invent missing data, run network calls or install packages.
'''

def translate(content):
    """Strict action envelope; no model code is interpolated into generated query code."""
    try:
        action = json.loads(content)
        if not isinstance(action, dict):
            return content, 'invalid'
        kind = action.get('action')
        if kind == 'skip' and set(action)=={'action','reason'} and isinstance(action['reason'],str) and action['reason'].strip():
            return content, 'skip'
        if kind == 'python' and set(action)=={'action','reason','code'} and all(isinstance(action[k],str) and action[k].strip() for k in ('reason','code')):
            return json.dumps({'action':'python','code':action['code']}), 'fallback'
        if kind != 'table_query' or set(action)!={'action','query','column'}:
            return '{}', 'invalid'
        q, column = action['query'], action['column']
        if not isinstance(q,dict) or not isinstance(column,str) or not column:
            return '{}', 'invalid'
        q = dict(q, tool='table_query')
        code=(f"r=table_query({q!r})\n"
              "if not r['ok']: raise ValueError('Table query failed')\n"
              "if len(r['result'])!=1: raise ValueError('Select exactly one result row')\n"
              f"if r.get('truncated') and not ({bool(q.get('sort'))!r} and {q.get('limit') == 1!r}): raise ValueError('Incomplete result')\n"
              f"value=r['result'][0][{column!r}]\n"
              "if value is None: raise ValueError('Missing answer')\n"
              "submit_answer(value)\n")
        return json.dumps({'action':'python','code':code}), 'table_query'
    except (ValueError,TypeError):
        return content, 'invalid'
