"""Join explicit manual evidence with statistics and verify original-file protection."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / 'results'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    automatic = read(HERE / 'automatic_summary.json')
    review = read(HERE / 'manual_reviews.json')
    format_audit = read(HERE / 'format_audit.json')
    cards = [json.loads(s) for s in (HERE / 'evidence_cards.jsonl').read_text(encoding='utf-8').splitlines()]
    reviews = {r['case']: r for r in review['cases']}
    assert len(reviews) == 42 and sorted(reviews) == list(range(1, 43))
    assert [c['case'] for c in cards] == list(range(1, 43))
    categories = Counter(r['category'] for r in reviews.values())
    assert sum(categories.values()) == 42
    joined, compact, reviewed_characters = [], [], 0
    reviewed_candidates = 0
    for card in cards:
        r = reviews[card['case']]
        assert r['category'] in review['categories']
        indices = sorted(set(card['correct_indices']+[card['method_choices'][e] for e in ['Pool_P', 'S2_C']]))
        reviewed_candidates += len(indices)
        reviewed_characters += len(card['question'])+sum(len(c['text']) for c in card['candidates'] if c['index'] in indices)
        joined.append({**card, 'human_review_status': 'REVIEWED', 'review': r,
            'reviewed_candidate_indices': indices, 'reviewed_question_and_these_candidates_in_full': True,
            'confidence': r.get('confidence', review['default_confidence']),
            'needs_more_context_for_recorded_observation': False,
            'model_internal_cause_verified': False})
        compact.append({'case': card['case'], 'row': card['row'], 'source_line': card['row']+1, 'id': card['id'],
            'category': review['categories'][r['category']], 'tags': '|'.join(r['tags']),
            'question': card['question'], 'gold': card['gold'], 'evidence': r['evidence'],
            'correct_computation_or_caveat': r['correct_computation'],
            'correct_trace_status': r.get('correct_trace_status', 'no_defect_recorded_in_this_review'),
            'correct_trace_evidence': r.get('correct_trace_evidence', ''), 'reference_flag': r.get('reference_flag', ''),
            'official_score_or_label_changed': False})
    with (OUT / 'reviewed_cases.jsonl').open('w', encoding='utf-8') as f:
        for row in joined:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (OUT / 'case_reviews.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(compact[0])); w.writeheader(); w.writerows(compact)
    substantive = [r['case'] for r in reviews.values() if r.get('correct_trace_status') == 'substantive_defect']
    wording = [r['case'] for r in reviews.values() if r.get('correct_trace_status') == 'wording_defect_only']
    strong_ref = [r['case'] for r in reviews.values() if r.get('reference_flag') == 'strong_wording_conflict']
    result = {**automatic, 'causal_reviews_completed': 42, 'manual_primary_categories': dict(categories),
        'manual_categories_not_independently_adjudicated': True, 'reviewed_candidates': reviewed_candidates,
        'reviewed_full_question_and_candidate_characters': reviewed_characters,
        'reviewed_characters_exclude_metadata_and_additional_reference_reading': True,
        'substantively_flawed_officially_correct_trace_cases': substantive,
        'additional_wording_or_variable_label_defect_cases': wording, 'strong_reference_wording_flags': strong_ref,
        'semantics_scope_condition_target_primary_total': sum(categories[k] for k in ['quantity_scope', 'relation_binding', 'condition_fidelity', 'target_binding']),
        'new_model_fit_or_new_predictions': False, 'official_scores_recomputed_or_changed': False,
        'training_format_recurrence': format_audit['splits']['train']['flags'],
        'test_used': False}
    write(OUT / 'diagnostic_summary.json', result)
    lines = ['# 42道共同失败题的逐题诊断', '',
      '**结论：优先补充候选文本中的局部计算和题意约束证据，不建议继续扩大专家集成或仅凭票数做更复杂选择。** 这是对冻结Pool-P与S2-C共同失败样本的诊断，不是新模型成绩，也不能证明固定向量不含正确性信息。', '',
      '## 阅读范围与口径', '',
      f'全部42题的题干、两代表选中的错误候选以及所有官方数值正确候选均已完整阅读，共{reviewed_candidates}份候选文本；并补看案例1、17、24、32、36、41的官方解答。其他未被选择的错误候选没有全部逐步验算。先用本地索引筛选、去重，再读必要文本；题干加已读候选为{reviewed_characters:,}字符，四候选全部文本加题干为108,330字符。此处不包含元数据与额外参考阅读，也不是精确token统计。', '',
      '统计标签从保存的预测及官方解析结果计算；数学错因由本次阅读人工标注。每题给一个主类别以便计数，次要错误另列，不把统计标签等同于模型内部原因。分类没有独立第二位评审，且只代表这42题的条件分布。源码和原始数据只读，未重新训练、推理、评分或更改官方标签。', '',
      '## 结构与现有方法的覆盖', '',
      '- 42题中38题只有一个正确候选；40题两种代表选中同一错误数值。',
      '- 22题有错误的唯一最高票答案，另外20题最高票并列；没有正确的唯一最高票答案。',
      '- 16题所有有效答案组均为单例（含解析失败候选时有效组可能少于4）；这些题的组内max/mean/sum相同，不能借重复答案提供新证据。其余26题有重复答案，但多数票会误导其中22题。不能由此断言所有答案组网络无效。',
      '- 11题含解析无效候选，但两个代表选中的错误候选均可被官方解析，因此此前过滤None的规则已不能处理这些失败。',
      '- 6个既有I/Pool/S2 × P/C五种子集成的并集中，只另覆盖5题；再加S2-route共覆盖6题，仍有36题全部失败。S2-route单独能修复其中4题。这只是已知正确性后的覆盖统计，不是可部署选择器成绩。',
      '- 30个已有单种子模型中，25题没有任何一个模型选对。Pool-P五种子全错35题、五种子一致选同一错误数值34题；S2-C分别为35题和32题。置信一致可能来自共同偏差。', '',
      '## 人工主分类', '', '| 主类别 | 题数 | 案例编号 |', '|---|---:|---|']
    for name in ['quantity_scope', 'relation_binding', 'condition_fidelity', 'target_binding', 'arithmetic', 'units', 'parser_output', 'reference_wording']:
        examples = [r['case'] for r in reviews.values() if r['category'] == name]
        lines.append(f'| {review["categories"][name]} | {categories[name]} | '+', '.join(map(str, examples))+' |')
    lines += ['', '数量范围、关系绑定、条件和所求量四类合计33题。例如当前余额又加一次收入；将总耗时当每层耗时；把“姐姐是Lexie两倍”反写；已经算出未康复人数却把它当剩余人数。它们通常不是再算一次等式就能发现的。', '',
      '5题主因是明确局部算术错误，另有单位案例24和简单利息案例42伴随算术错误。例：案例4把3+5+7+9+11算36；案例5把12×0.50算60；案例28所有分项都正确，和却由54变63。候选间区别可落在极小的数字或关系片段上，支持尝试更细的文本证据；尚不能证明整段embedding已经丢失这些信息。', '',
      '## 两个容易误判的发现', '',
      '**输出解析错位。** 案例41的被选候选正确计算240、250、220，并最终写boxed250，但其最后一个`####`是Job3标题，官方优先从这里读到3，因而该候选判错。候选0既有正确数值也可正确解析。案例19中S2选中的截断候选最后boxed为`32-d`，被解析为32，虽然尾部已试出31、32、33；Pool另选中直接把中间年龄32当最小年龄的候选。两例不能笼统记成数学能力不足，也不能更改解析器或候选文本来消除错误。', '',
      '**最终正确不等于过程无错。** 案例2、13、17、20、24、27共6题，唯一官方正确候选有显著数量、算术或逻辑问题，最终数值却与参考相同。案例17正确候选用错误的总数32和袋数4得到8，官方有效计算应为40/5；案例20正确候选误读现金为10、算出-5后又直接输出5。另4题存在变量或单位表述失准，单独记录，未混入这6题。因此“发现一个过程错误就否决该候选”会和最终数值评分目标冲突；一致性证据应作软特征，仍由训练集最终答案正确性监督。', '',
      '**参考题面疑点。** 案例1的`1/4 more than`通常得到17，参考按四分之一再加7得到9；案例36的`at20% of`字面应120，参考按`20% off`得到480。但案例36当前被选720按两种读法都错，疑点不能替其免责。这两题保留固定标签和分数，只在诊断中标注，不据此改写准确率。案例32另有不顺的题面语法，但没有将其计作第三个明确参考冲突。', '',
      '## 格式问题在训练集中是否复现', '',
      '增加一次只读的格式普查：只识别最后一个非嵌套boxed中的完整数值字面量，且它在最后一个####之后；若与官方解析结果不同，则记录冲突。另记录以数字开头的代数boxed。这只是有限覆盖的诊断探针，不是替代解析器。', '',
      '- 训练集19,020个候选中，有22个数值框与官方解析冲突，涉及17题；其中16个框内数值符合参考，但官方解析不符。22个候选按当前官方口径均为错。',
      '- 既有训练OOF选择中，Pool-P选中过这类候选7次，S2-C选中过5次。它们在训练集上确实可见，格式现象不是只在这42道验证题出现。',
      '- 另有11个数字前缀的代数boxed，其中1个按官方解析碰巧正确；不能将所有表达式框硬性视为错误。',
      '- 验证全体候选的对应冲突有7个、涉及6题，其中4个框内数值匹配参考。没有运行任何新过滤规则或计算新方法准确率；这些数量不等于潜在净提升。', '',
      '## 建议的最小下一步', '',
      '1. **先补信息，再改选择结构。** 保持S2-C主模型冻结，给每个候选添加小量可复核的文本证据：官方实际取数位置与末尾答案是否一致、非纯数值/未解变量框、明显截断、保守支持的纯数值等式是否成立。用训练OOF学习很小的候选分数校正，目标仍为最终数值正确。和上一组“选择哪位专家”相比，这里新增的是原模型没有显式读取的内容证据。',
      '2. **四组足够检验证据来源。** 原S2-C、仅输出结构证据、仅算术证据、两者合并；先做提取覆盖和误报审计，再固定定义及训练设置。数学表达式只处理明确支持的运算，无法解析时标未知，不能eval候选代码、补生成解答或把未解析当错。不得针对42题手写分支或拟合参数。',
      '3. **主体难点是题目与解答的数量关系。** 上述小实验仅覆盖部分问题。如果继续投入训练，更值得试题干—解答的数字、实体、单位及关系对齐，或定位四候选真正分歧的局部片段，再与既有向量结合；使用训练数据建立词表/映射，不引入外部预训练模型。仅检测数字是否出现不够，案例7和26已复述条件却未使用，案例23已算税却未纳入最终答案。此方向目前是研究假设，不是已证明可学习。',
      '4. **答案组建模暂不优先。** 16题没有可利用的重复答案，22题唯一多数反而错；单纯强化组票数或换聚合层，未提供上述局部计算和约束证据。后续若进入组建模，应明确控制组大小先验，并验证是否能保留少数正确候选。',
      '5. **评估须计算损害。** 全部验证题与355道多答案题都报告净增，另看单正确候选、错误多数、格式冲突等分组，并复核新增证据是否误伤那些最终正确但过程有错的候选。不能只对这42题报修复数，也不能把本次人工错因当训练标签。', '',
      '## 逐题证据', '',
      '候选索引为0起始；source_line是原始验证候选JSONL的1起始行号。原始长文本留在evidence_cards.jsonl，复核后的完整卡在results/reviewed_cases.jsonl；精简索引在results/case_reviews.csv。以下公式是人工诊断证据，不是新的候选生成或部署规则。', '']
    for card in joined:
        r = card['review']
        lines += [f'### 案例{card["case"]}（原数据行{card["row"]+1}）', '',
            f'- 主类别：{review["categories"][r["category"]]}；次级标签：'+', '.join(r['tags'])+'。',
            f'- 题目：{card["question"]}',
            f'- 官方目标：{card["gold"]}；四候选解析：{card["parsed_answers"]}；Pool-P选{card["method_choices"]["Pool_P"]}，S2-C选{card["method_choices"]["S2_C"]}。',
            '- 证据：'+r['evidence'], '- 核对：'+r['correct_computation']]
        if r.get('correct_trace_evidence'):
            lines.append('- 数值正确候选的过程问题：'+r['correct_trace_evidence'])
        if r.get('reference_flag'):
            lines.append('- 题面标记：'+r['reference_flag']+'；标签保持原样。')
        lines.append('')
    lines += ['## 复现与保护', '',
      'Python3.12下设置PYTHONDONTWRITEBYTECODE=1，所有命令带-B。依次运行prepare.py、format_audit.py、finalize.py；manual_reviews.json是本次人工阅读的固定标注输入，重新运行脚本不会自动重做人工判断。没有API调用、外部数据或新模型推理；测试集语义内容未读取。保护扫描仅按字节核对开始时已有文件。最终保护结果见verification.json，未通过时保留差异且不更新保护快照。']
    report = OUT / 'REPORT.md'
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    sources = read(HERE / 'sources.json')
    assert sha(HERE / 'prepare.py') == sources['script_sha256']
    assert all(sha(v['path']) == v['sha256'] for v in sources['predictions'].values())
    assert sha(sources['probabilities']['path']) == sources['probabilities']['sha256']
    before = read(HERE / 'protected_before.json')
    changed = [p for p,h in before.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    verification = {'passed': not changed, 'protected_files': len(before), 'protected_changes': changed,
      'all_42_manually_reviewed': True, 'review_scope': review['review_scope'],
      'reviewed_candidates': reviewed_candidates, 'primary_labels_cover_all_42_exactly_once': True,
      'source_predictions_unchanged': True, 'scores_and_official_parser_unchanged': True,
      'new_training_or_inference': False, 'new_official_method_scores': False, 'test_used': False,
      'sources': {p.name: sha(p) for p in HERE.iterdir() if p.suffix in ('.py', '.json') and p.name != 'protected_before.json'},
      'report_sha256': sha(report)}
    write(OUT / 'verification.json', verification)
    print(json.dumps({'manual_categories': dict(categories), 'reviewed_candidates': reviewed_candidates,
      'reviewed_characters': reviewed_characters, 'protected_passed': not changed,
      'protected_files': len(before), 'changes': changed}, ensure_ascii=False))
    assert not changed, changed

if __name__ == '__main__':
    main()
