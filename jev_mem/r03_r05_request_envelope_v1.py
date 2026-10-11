"""Transport envelope shared by the versioned R03/R05 experiment.

Scientific signatures are regenerated from the frozen pure planners before
dispatch. This module only validates the systemone wire format and its scores.
"""
from __future__ import annotations
from jev_mem.evidence_exchange_r01 import serialized_signature, parse_scores

MODEL = 'jev-1.13.0'


def validate_request(signature, *, allow_oversize=False):
    if (not isinstance(signature, dict)
        or set(signature) != {'provider','model','operation','body'}
        or (signature['provider'],signature['model'],signature['operation'])
           != ('jev',MODEL,'systemone')):
        raise ValueError('Frozen JEV systemone envelope required')
    body=signature['body']
    if (not isinstance(body,dict) or set(body)!={'model','state','questions'}
        or body['model']!=MODEL or not isinstance(body['state'],dict)
        or not isinstance(body['questions'],dict) or not 1<=len(body['questions'])<=80):
        raise ValueError('Bounded Noul envelope required')
    for key,value in body['questions'].items():
        if (not isinstance(key,str) or not key
            or not isinstance(value,dict) or set(value)!={'type','instructions'}
            or value['type']!='noul' or not isinstance(value['instructions'],str)
            or not value['instructions'].strip()):
            raise ValueError('Only exact Noul question records permitted')
    if not allow_oversize and len(serialized_signature(signature))>131072:
        raise ValueError('Signature exceeds frozen cap; never truncate')


def parse_response(response, signature):
    validate_request(signature)
    return parse_scores(response,signature)
