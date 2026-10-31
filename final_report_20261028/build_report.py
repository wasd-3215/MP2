"""Build the course PDF from its editable Markdown and existing experiment records."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Flowable
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / 'code'
POPPLER = Path('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
FONT_DIR = Path('C:/Windows/Fonts')
pdfmetrics.registerFont(TTFont('Body', str(FONT_DIR / 'simsun.ttc'), subfontIndex=0))
pdfmetrics.registerFont(TTFont('Sans', str(FONT_DIR / 'msyh.ttc'), subfontIndex=0))
pdfmetrics.registerFont(TTFont('Bold', str(FONT_DIR / 'msyhbd.ttc'), subfontIndex=0))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def inventory():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*')
            if p.is_file() and HERE not in p.parents}


def chart():
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    path = out / 'consensus_effects.pdf'
    c = canvas.Canvas(str(path), pagesize=(680, 210))
    c.setTitle('共识特征的配对效应')
    panels = [
        (62, 245, (-2.5, 1.6), '完整验证 798题', [-.752, -.627, .301],
         [(-2.180, .677), (-1.980, .677), (-.802, 1.429)], [-2, -1, 0, 1]),
        (397, 245, (-52, 3), '错误最高票 60题', [-37.667, -32., -23.667],
         [(-48.667, -26.658), (-43., -21.667), (-31.667, -16.)], [-50, -40, -30, -20, -10, 0])]
    for x, width, limits, title, vals, intervals, ticks in panels:
        def xx(v):
            return x + (v - limits[0]) / (limits[1] - limits[0]) * width
        c.setFillColor(colors.HexColor('#111827'))
        c.setFont('Bold', 12)
        c.drawCentredString(x + width / 2, 188, title)
        for tick in ticks:
            c.setStrokeColor(colors.HexColor('#e5e7eb'))
            c.setLineWidth(.6)
            c.line(xx(tick), 52, xx(tick), 166)
            c.setFillColor(colors.HexColor('#4b5563'))
            c.setFont('Sans', 9)
            c.drawCentredString(xx(tick), 36, str(tick))
        c.setDash(3, 3)
        c.setStrokeColor(colors.HexColor('#6b7280'))
        c.line(xx(0), 52, xx(0), 166)
        c.setDash()
        for j, name in enumerate(['I', 'Pool', 'S2']):
            y = 150 - j * 41
            c.setFont('Sans', 11)
            c.setFillColor(colors.black)
            c.drawRightString(x - 11, y - 4, name)
            low, high = intervals[j]
            c.setStrokeColor(colors.HexColor('#365d7b'))
            c.setLineWidth(1.7)
            c.line(xx(low), y, xx(high), y)
            for v in [low, high]:
                c.line(xx(v), y-4, xx(v), y+4)
            c.setFillColor(colors.HexColor('#365d7b'))
            c.circle(xx(vals[j]), y, 3.7, fill=1, stroke=0)
            c.setFont('Sans', 9)
            c.drawCentredString(xx(vals[j]), y + 9, f'{vals[j]:+.3f}')
        c.setFillColor(colors.black)
        c.setFont('Sans', 10)
        c.drawCentredString(x + width/2, 14, 'C − P 准确率变化  百分点')
    c.save()
    subprocess.run([str(POPPLER/'pdftoppm.exe'), '-r', '200', '-png', '-singlefile', str(path), str(out/'consensus_effects')], check=True, capture_output=True)


PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 100
styles = {
    'body': ParagraphStyle('body', fontName='Body', fontSize=10.5, leading=16.2,
                           spaceAfter=7, wordWrap='CJK', allowWidows=0, allowOrphans=0),
    'title': ParagraphStyle('title', fontName='Bold', fontSize=21, leading=29, spaceAfter=9),
    'h1': ParagraphStyle('h1', fontName='Bold', fontSize=14.5, leading=22, spaceAfter=10, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='Bold', fontSize=11.3, leading=18, spaceBefore=7, spaceAfter=7, keepWithNext=True),
    'caption': ParagraphStyle('caption', fontName='Sans', fontSize=9, leading=13.5, spaceBefore=3, spaceAfter=7, wordWrap='CJK', keepWithNext=True),
    'note': ParagraphStyle('note', fontName='Body', fontSize=9, leading=13, spaceAfter=8, wordWrap='CJK'),
    'cell': ParagraphStyle('cell', fontName='Body', fontSize=9.2, leading=13.7, wordWrap='CJK'),
    'head': ParagraphStyle('head', fontName='Sans', fontSize=9, leading=13.5, wordWrap='CJK', textColor=colors.black),
    'ref': ParagraphStyle('ref', fontName='Body', fontSize=8.6, leading=12.4, spaceAfter=4.5, wordWrap='CJK'),
    'code': ParagraphStyle('code', fontName='Sans', fontSize=7.6, leading=11.2, spaceAfter=1, wordWrap='CJK'),
}


def inline(text):
    text = text.replace('−', '-').replace('–', '-').replace('p̄_i', 'p_mean,i')
    text = escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = text.replace('βᵀ', 'β<super>T</super>')
    text = text.replace('p_mean,i', 'p<sub>mean,i</sub>')
    for variable in ['x', 'y', 's', 'f']:
        text = text.replace(variable + '_i', variable + '<sub>i</sub>')
    return text


def table(lines, caption):
    rows = [[x.strip() for x in line.strip().strip('|').split('|')] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r'[:\- ]+', x) for x in row)]
    n = len(rows[0])
    ratios = {
        1:[.20,.18,.31,.31], 2:[.22,.54,.24], 3:[.24,.30,.24,.22],
        4:[.35,.17,.29,.19], 5:[.43,.19,.19,.19], 6:[.20,.35,.24,.21],
        7:[.20,.46,.34], 8:[.39,.11,.39,.11], 9:[.29,.14,.15,.10,.32],
        10:[.29,.12,.14,.23,.11,.11], 11:[.32,.18,.19,.14,.17], 12:[.28,.36,.36],
    }.get(caption, [1/n]*n)
    content = []
    for ri, row in enumerate(rows):
        prow=[]
        for ci,value in enumerate(row):
            key='head' if ri==0 else 'cell'
            sty=ParagraphStyle(f'{key}-{ri}-{ci}', parent=styles[key])
            is_numeric = bool(re.fullmatch(r'[\d\s.,%+−±/\[\]–-]+',value))
            sty.alignment = TA_CENTER if (ci>0 and (is_numeric or caption in [1,4,5,6,9,10,11,12])) else TA_LEFT
            prow.append(Paragraph(inline(value),sty))
        content.append(prow)
    t=Table(content,colWidths=[WIDTH*r for r in ratios],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#D9D9D9')),
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf2')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    return t


def footer(c,doc):
    c.saveState()
    c.setFillColor(colors.HexColor('#4b5563'))
    c.setFont('Sans',8)
    c.drawString(50,22,'DASE7506 MP2  固定候选数学解答的可靠选择')
    c.drawRightString(PAGE_W-50,22,str(doc.page))
    c.restoreState()


def build():
    lines=(HERE/'report.md').read_text(encoding='utf-8').splitlines()
    story=[]; i=0; caption=0; logical_page=1
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1; continue
        if line=='<!-- PAGE -->':
            story.append(PageBreak()); logical_page+=1; i+=1; continue
        if line.startswith('```'):
            i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                story.append(Paragraph(escape(lines[i]).replace(' ','&#160;'),styles['code']))
                i+=1
            story.append(Spacer(1,8)); i+=1; continue
        if line.startswith('|'):
            tlines=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                tlines.append(lines[i]); i+=1
            story.append(table(tlines,caption)); story.append(Spacer(1,9)); continue
        if line.startswith('!['):
            rel=re.search(r'\]\((.*?)\)',line).group(1)
            story.append(Image(str(HERE/rel),width=WIDTH,height=WIDTH*210/680))
            story.append(Spacer(1,3)); i+=1; continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]),styles['title'])); i+=1; continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]),styles['h1'])); i+=1; continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]),styles['h2'])); i+=1; continue
        if re.match(r'^表\d+ ',line):
            caption=int(re.match(r'^表(\d+)',line).group(1))
            story.append(Paragraph(inline(line),styles['caption'])); i+=1; continue
        kind='ref' if logical_page==10 and re.match(r'^\[\d+\]',line) else ('note' if line.startswith('图1 ') else 'body')
        story.append(Paragraph(inline(line),styles[kind])); i+=1
    pdf=HERE/'output/pdf/DASE7506_MP2_Report.pdf'
    pdf.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=41,bottomMargin=40,
                          title='固定候选数学解答的可靠选择',author='',subject='DASE7506 MP2 实验报告')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(str(pdf))
    texts=[p.extract_text() for p in reader.pages]
    (HERE/'qa').mkdir(exist_ok=True)
    (HERE/'qa/pages_text.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pdf':str(pdf),'pages':len(reader.pages),'characters_per_page':[len(t) for t in texts]},ensure_ascii=False))


def manifest():
    sources=json.loads((CODE/'experiment_review_20261028/sources.json').read_text(encoding='utf-8'))
    paths=[ROOT/s['path'] for s in sources]
    paths += [ROOT/'guide/GUIDE.md',CODE/'data/manifest.json',CODE/'final_test_20261028/frozen_before_test.json']
    paths += sorted((CODE/'final_test_20261028/results').glob('*/score.json'))
    paths += [p for p in (CODE/'arith_relation_audit_20261018/portable').rglob('*') if p.is_file()]
    result={'date':'2026-10-28','primary_frozen_variant':'V','files':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    (HERE/'source_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    import sys
    op=sys.argv[1]
    if op=='before':
        snap=inventory()
        (HERE/'protected_before.json').write_text(json.dumps(snap,indent=2),encoding='utf-8')
        print(json.dumps({'protected_files':len(snap)}))
    elif op=='build':
        chart(); build(); manifest()
    elif op=='verify':
        old=json.loads((HERE/'protected_before.json').read_text(encoding='utf-8'))
        changed=[]; missing=[]
        for rel,value in old.items():
            p=ROOT/rel
            if not p.exists(): missing.append(rel)
            elif sha(p)!=value: changed.append(rel)
        pdf=HERE/'output/pdf/DASE7506_MP2_Report.pdf'
        reader=PdfReader(str(pdf))
        text='\n'.join(p.extract_text() for p in reader.pages)
        expected={'V':374,'H':377,'NV':377,'B':368,'S2_route':383,'Pool_P':375}
        for name,n in expected.items():
            result=json.loads((CODE/f'final_test_20261028/results/{name}/score.json').read_text())
            assert result['correct']==n and result['n']==467
        assert sum([48,48,60,42,48,36,140,30,30,40,25,30])==577
        assert len(reader.pages)<=10
        result={'pages':len(reader.pages),'protected_files':len(old),'changed':changed,'missing':missing,
                'source_report_count':18,'test_scores_matched':True,'new_experiments_executed':False,
                'pdf_sha256':sha(pdf),'markdown_sha256':sha(HERE/'report.md'),
                'passed':not changed and not missing,'visual_review':'recorded separately after PNG inspection'}
        (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False))
