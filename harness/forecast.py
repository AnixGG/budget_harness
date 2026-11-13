"""Portable CPU inference. Bundles contain fitted parameters, never task labels."""
from __future__ import annotations
import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
from .core import predict

NUMERIC = ('text_bytes','text_chars','numeric_literals','file_count','file_bytes','python_files','csv_files',
           'max_calls','max_output_tokens','max_retry_output_tokens','max_generation_retries','max_prompt_bytes',
           'max_tool_output_bytes','temperature','timeout_seconds','tool_timeout_seconds','reasoning_in_completion',
           'max_preview_bytes','single_action','auto_submit_coding')
CATEGORICAL = ('task_type','model','workflow')
ALLOWED = set(NUMERIC+CATEGORICAL+('text',))


def features(task, config):
    text=task['instruction']
    return {'task_type':task['task_type'],'text':text,'text_bytes':len(text.encode()),'text_chars':len(text),
            'numeric_literals':len(re.findall(r'\b\d+(?:\.\d+)?\b',text)),
            'file_count':len(task['files']),'file_bytes':sum(f['bytes'] for f in task['files']),
            'python_files':sum(f['path'].endswith('.py') for f in task['files']),
            'csv_files':sum(f['path'].endswith('.csv') for f in task['files']),
            'model':config.model,'workflow':config.workflow,
            **{k:getattr(config,k) for k in NUMERIC if hasattr(config,k)}}


def tokens(text):
    return re.findall(r'(?u)\b\w\w+\b',text.lower())


def encode(feature_rows, spec):
    numeric=np.array([[float(r[k]) for k in spec['numeric']] for r in feature_rows])
    numeric=(numeric-np.array(spec['mean']))/np.array(spec['scale'])
    cats=np.array([[float(r[k]==v) for k,vs in spec['categories'].items() for v in vs] for r in feature_rows])
    parts=[numeric,cats]
    if spec['tfidf']:
        counts=[Counter(tokens(r['text'])) for r in feature_rows]
        text=np.array([[c[t] for t in spec['vocabulary']] for c in counts],dtype=float)
        text*=np.array(spec['idf'])
        norm=np.linalg.norm(text,axis=1,keepdims=True)
        parts.append(text/np.where(norm==0,1,norm))
    return np.concatenate(parts,axis=1)


def sigmoid(value):
    return 1/(1+np.exp(-np.clip(value,-40,40)))


def tree_value(tree, row):
    while 'leaf_value' not in tree:
        if tree['decision_type']!='<=':
            raise ValueError('Only numerical LightGBM splits are supported')
        value=row[tree['split_feature']]
        left=tree.get('default_left',True) if math.isnan(value) else value<=float(tree['threshold'])
        tree=tree['left_child'] if left else tree['right_child']
    return tree['leaf_value']


def infer(model, feature_rows, config_ids):
    kind=model['kind']
    if kind=='constant':
        return [model['value']]*len(feature_rows)
    if kind=='statistics':
        results=[]
        for f,cid in zip(feature_rows,config_ids):
            group=next((g for g in model['groups'] if g['task_type']==f['task_type'] and g['config_id']==cid),None)
            if group is None:
                group=model['by_type'].get(f['task_type'],model['global'])
            results.append(group[model['field']])
        return results
    x=encode(feature_rows,model['preprocessing'])
    if kind=='ridge':
        return np.maximum(0,(x-np.array(model['xmean'])) @ np.array(model['coef'])+model['ymean']).tolist()
    if kind=='logistic':
        return sigmoid(x @ np.array(model['coef'])+model['intercept']).tolist()
    if kind=='lightgbm':
        dump=model['trees']
        if dump.get('average_output',False):
            raise ValueError('Random-forest LightGBM exports are unsupported')
        values=np.array([sum(tree_value(t['tree_structure'],row) for t in dump['tree_info']) for row in x])
        if model['objective']=='binary':
            return sigmoid(values).tolist()
        return np.maximum(0,values).tolist()
    raise ValueError('Unknown estimator kind')


class Forecaster:
    def __init__(self, bundle, *, allow_fixture=False):
        self.bundle=json.loads(Path(bundle).read_text()) if isinstance(bundle,(str,Path)) else bundle
        if self.bundle['schema_version']!=1 or not self.bundle['supported_configs']:
            raise ValueError('Invalid forecast bundle')
        if self.bundle['provenance']!='real_provider' and not allow_fixture:
            raise ValueError('Fixture models cannot drive real execution')

    def estimate(self, task, config):
        if config.config_id not in self.bundle['supported_configs']:
            raise ValueError('No measured support for this model/caps/policy configuration')
        f=features(task,config)
        cost=(predict(task,config)['budget_units'] if self.bundle['cost']['kind']=='formula'
              else infer(self.bundle['cost'],[f],[config.config_id])[0])
        p=infer(self.bundle['success'],[f],[config.config_id])[0]
        if not math.isfinite(cost) or cost<0 or not math.isfinite(p) or not 0<=p<=1:
            raise ValueError('Invalid forecast output')
        return {'budget_units':max(.001,float(cost)),'p_success':float(p),'method':self.bundle['method'],
                'prediction_point':'before_agent_start','config_id':config.config_id,
                'uncertainty':self.bundle['warning']}
