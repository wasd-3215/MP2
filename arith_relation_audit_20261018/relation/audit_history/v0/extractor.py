"""Conservative relation probes with source spans; unknown is not a contradiction.

These are feasibility probes, not a semantic verifier or a deployed rule set.
Rate/count differences need entity/scope audit. A repeated condition is retrieval only.
"""
from pathlib import Path
import re
import sys
import ast
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'arithmetic'))
from evidence import arithmetic_evidence, _clean
from common import parse_answer

NUMBER=r'(?<![\w./])(?:\d+/\d+|\d[\d,]*(?:\.\d+)?)(?![\w./])'
UNITS=r'(?:dollars?|cents?|balloons?|chairs?|miles?|hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|pounds?|liters?|feet|meters?|episodes?|records?|feathers?|pillows?|oranges?|apples?|trips?|seasons?)'
RATE=re.compile(r'(?P<value>'+NUMBER+r')\s+(?P<num>'+UNITS+r')\s*(?:per|every|an?|each|/)\s*(?:(?P<denvalue>\d+)\s+)?(?P<den>'+UNITS+r')\b',re.I)
CASH_RATE=re.compile(r'\$(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?:per|every|an?|each)\s*(?:(?P<denvalue>\d+)\s+)?(?P<den>'+UNITS+r')\b',re.I)
GENDER=re.compile(r'(?P<value>'+NUMBER+r')\s+(?P<entity>boys?|girls?|males?|females?|men|women|husbands?|wives?)\b',re.I)
CUE=re.compile(r'\b(?:twice|double|half|second|except|only|before|after|remaining)\b',re.I)

def canonical(s):
    s=s.lower()
    return {'men':'male','man':'male','boys':'male','boy':'male','males':'male','husband':'male','husbands':'male',
      'women':'female','woman':'female','girls':'female','girl':'female','females':'female','wife':'female','wives':'female',
      'hrs':'hour','hr':'hour','min':'minute','mins':'minute','feet':'foot'}.get(s,s[:-1] if s.endswith('s') else s)

def lines(text):
    offset=0
    for raw in text.splitlines(keepends=True):
        plain=raw.replace('**','').replace('\\,',' ').replace('\\times','*').replace('\\div','/')
        plain=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r'\1',plain)
        # Every record keeps the exact full raw source line and its original offsets.
        yield plain,{'start':offset,'end':offset+len(raw),'text':raw}
        offset+=len(raw)

def rates(text):
    rows=[]
    for plain,source in lines(text):
        for pattern,cash in [(RATE,False),(CASH_RATE,True)]:
            for m in pattern.finditer(plain):
                value=Fraction(m['value'].replace(',',''))/int(m['denvalue'] or 1)
                rows.append({'value':str(value),'numerator':'dollar' if cash else canonical(m['num']),
                  'denominator':canonical(m['den']),'source':source,'matched':m.group()})
    return rows

def counts(text):
    rows=[]
    for plain,source in lines(text):
        for m in GENDER.finditer(plain):
            # Percentages/fractional subgroups do not become a literal person count.
            if '/' in m['value'] or re.search(r'[%/]',plain[max(0,m.start()-2):m.start()]):continue
            rows.append({'value':str(Fraction(m['value'].replace(',',''))),'entity':canonical(m['entity']),
                         'source':source,'matched':m.group()})
    return rows

def target(question):
    matches=list(re.finditer(r'\b(?:how|what)\b[^?]*\?',question,re.I))
    if not matches:return {'type':'unknown','source':None}
    m=matches[-1];q=m.group().lower()
    if re.search(r'\b(?:each|per)\b',q):kind='per_unit'
    elif re.search(r'\b(?:more|fewer|less|difference|additional)\b',q):kind='difference'
    elif re.search(r'\b(?:left|remaining|remain|still holding|to spare)\b',q):kind='remaining'
    elif re.search(r'\b(?:total|in all|altogether|combined)\b',q):kind='total'
    else:kind='unknown'
    return {'type':kind,'source':{'start':m.start(),'end':m.end(),'text':m.group()}}

def final_operation(text):
    answer=parse_answer(text)
    pairs=arithmetic_evidence(text)[1]['pairs']
    matching=[p for p in pairs if p['right_value']==answer]
    if not matching:return {'type':'unknown','source':None}
    pair=matching[-1]
    try:
        tree=ast.parse(_clean(pair['left']),mode='eval')
        ops={type(n.op).__name__ for n in ast.walk(tree) if isinstance(n,ast.BinOp)}
    except (ValueError,SyntaxError):return {'type':'unknown','source':None}
    kind={frozenset({'Add'}):'sum',frozenset({'Sub'}):'difference',frozenset({'Mult'}):'product',frozenset({'Div'}):'quotient'}.get(frozenset(ops),'mixed')
    offset=pair['source_offset']
    return {'type':kind,'source':{'start':offset,'end':min(len(text),offset+300),'text':text[offset:offset+300]},'pair':pair,
      'caveat':'Equals official parsed final value; may be an earlier repeated value, not proven final dependency.'}

def cues(text):
    return [{'cue':m.group().lower(),'source':{'start':m.start(),'end':m.end(),'text':m.group()}} for m in CUE.finditer(text)]

def extract(question,text):
    qr=rates(question);cr=rates(text);qc=counts(question);cc=counts(text)
    aligned_rates=[]
    for key in sorted({(r['numerator'],r['denominator']) for r in qr}):
        q=[r for r in qr if (r['numerator'],r['denominator'])==key]
        if len({r['value'] for r in q})!=1:continue
        for c in cr:
            if (c['numerator'],c['denominator'])==key:
                aligned_rates.append({'kind':'explicit_rate_value_comparison','question':q[0],'candidate':c,
                  'status':'same_value' if q[0]['value']==c['value'] else 'different_value',
                  'binding_proven':False,'caveat':'Entity, stage and aggregate-vs-individual scope require review.'})
    aligned_counts=[]
    for entity in sorted({r['entity'] for r in qc}):
        q=[r for r in qc if r['entity']==entity]
        if len({r['value'] for r in q})!=1:continue
        for c in cc:
            if c['entity']==entity:
                aligned_counts.append({'kind':'explicit_entity_count_comparison','question':q[0],'candidate':c,
                  'status':'same_value' if q[0]['value']==c['value'] else 'different_value','binding_proven':False,
                  'caveat':'Subgroup, time and total scope require review.'})
    qt=target(question);op=final_operation(text)
    target_probe=None
    if qt['type']!='unknown' and op['type']!='unknown':
        expected={'difference':{'difference','mixed'},'remaining':{'difference','mixed'},'total':{'sum','mixed'},'per_unit':{'quotient','mixed'}}[qt['type']]
        target_probe={'kind':'target_operation_compatibility','question':qt,'candidate':op,
          'status':'compatible_operation' if op['type'] in expected else 'different_operation',
          'binding_proven':False,'caveat':'Operation form alone does not establish entity/target correspondence; equivalent calculations may differ.'}
    qcue=cues(question);ccue=cues(text)
    # Do not turn repetition or absence into a claim of actual condition use/omission.
    condition={'question_cues':qcue,'candidate_cues':ccue,'actual_use':'unknown',
      'reason':'No reliable entity/stage-to-equation binding established by these lightweight probes.'}
    return {'scope_unit':aligned_rates,'entity_binding':aligned_counts,'condition_use':condition,'target_alignment':target_probe}

def check_sources(question,text,value):
    def visit(v):
        if isinstance(v,dict):
            if 'source' in v and v['source']:
                s=v['source'];assert s['text'] in [question[s['start']:s['end']],text[s['start']:s['end']]]
            for c in v.values():visit(c)
        elif isinstance(v,list):
            for c in v:visit(c)
    visit(value)
