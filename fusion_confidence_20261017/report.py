"""Generate a self-contained report directly from recorded, officially scored results."""
from utilities import HERE,read,write,sha
import numpy as np


def main():
    out=HERE/'results'
    features=read(out/'feature_summary.json');records=read(out/'pipelines.json')
    paired=read(out/'paired_comparisons.json');routes=read(out/'routing_fits.json')
    resources=read(out/'resources.json');cost=read(out/'cost.json');selection=read(out/'selection.json')
    verification=read(out/'verification.json');env=read(out/'environment.json')
    get=lambda n:next(r for r in records if r['name']==n)
    comparison=lambda a,b:next(r for r in paired if r['a']==a and r['b']==b)
    def ci(r):return f"{r['delta_pp']:+.3f} [{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}]"
    lines=['# 集成、显式特征与置信度回退：三组实验', '',
      '完整完成40次全训练集拟合、100次训练集内部五折拟合，共140次。所有模型固定训练20轮，不使用验证标签选择轮次。所有主准确率均由未修改的官方独立评分脚本产生。', '',
      f"交付方案为 **{selection['name']}**，验证 **{selection['correct']}/798 = {selection['accuracy']:.3f}%**。这属于在已反复使用的验证集上进行开发和方法选择的结果，不是独立测试成绩。",'',
      '## 协议与假设', '',
      '- 假设一：不同种子及不同结构提供互补，等权集成可减少错误；五成员或十五成员集成本身各是一个固定预测器。',
      '- 假设二：显式答案一致性补充固定embedding中未充分利用的信息；解析与风格分开消融。',
      '- 假设三：模型低置信且投票明确时，回退净纠错为正；同时允许训练OOF结果选择完全禁用回退。',
      '- 统一20轮、LR0.0003、batch256、BCE、AdamW decay0.001、hidden128/dropout0.1，种子17/29/43/71/101。',
      '- 新特征列初始为零，共有初始权重与无特征模型一致；初始化后统一重置dropout RNG。所有权重从头共同训练。',
      '- 本轮S2基线重新训练；历史92.31%是各种子验证最佳检查点均值，协议不同，不用作本轮增益分母。',
      '- OOF按整道题随机五折，每个架构×种子逐折训练；标准化仅拟合其余四折。轮次固定，不使用留出折标签挑检查点。',
      '- 回退阈值只由训练OOF正确数选择；校准器也只拟合训练OOF。正式验证标签不参与阈值、校准或集成权重拟合。',
      '- 未将测试输入或参考答案用于训练、调参、预测、评分；全文件保护只按字节核验哈希。未使用外部数据/权重/API。', '',
      '## 一、显式特征：五种子结果', '',
      '| 配置 | 验证均值±样本标准差 % | 特征维度 | 参数 | 训练BCE | 验证BCE |',
      '|---|---:|---:|---:|---:|---:|']
    for r in features:
        lines.append(f"| {r['variant']} | {r['mean_accuracy']:.3f} ± {r['sd_accuracy']:.3f} | {r['feature_dimensions']} | {r['parameters']} | {r['mean_train_bce']:.4f} | {r['mean_validation_bce']:.4f} |")
    lines+=['','解析为1维；一致性含解析共6维；风格含解析共8维；全部13维。文本风格为字符/词/行/运算符/数学标记/答案标记/四个固定自信词的log1p计数。', '',
      '| 配对比较 | 均值差与95%描述性区间，百分点 | 五种子差值 |', '|---|---:|---|']
    for r in paired:
        if r['kind']=='five_seed_mean':
            lines.append(f"| {r['a']} − {r['b']} | {ci(r)} | {', '.join(f'{v:+.3f}' for v in r['seed_deltas_pp'])} |")
    r=comparison('s2_consistency','s2')
    lines+=['',f"一致性整体相对S2的变化为 {ci(r)} 个百分点。只有区间不跨零时才称为本轮较明确的配对证据；所有结论仍受重复验证开发影响。",'',
      '## 二、集成：固定预测器结果', '',
      '| 方法 | 成员 | 正确数/798 | 官方准确率 % |', '|---|---:|---:|---:|']
    for name in ['s2_seed17','s2_5','s2_5_answer_vote','heterogeneous_15','heterogeneous_15_answer_vote','consistency_5','consistency_5_answer_vote']:
        r=get(name);lines.append(f"| {name} | {r['members']} | {r['correct']} | {r['accuracy']:.3f} |")
    lines+=['','S2/S0/池化各五个种子组成十五成员模型，各结构权重相同。答案投票忽略解析无效答案，票数并列用平均候选分数打破。平均sigmoid分数不自动具有正确概率语义。', '',
      '| 比较 | 差值与95%描述性区间，百分点 |','|---|---:|']
    for a,b in [('s2_5','s2_seed17'),('heterogeneous_15','s2_5'),('consistency_5','s2_5')]:
        r=comparison(a,b);lines.append(f'| {a} − {b} | {ci(r)} |')
    lines+=['','同结构集成相对单个固定seed17的比较不是五次独立重复；特征实验的五种子均值也不能直接当作一个可部署预测器的准确率。', '',
      '## 三、置信度回退', '',
      '候选分差可能把同答案的两个候选误认为不确定。答案分差使用组内最大分，不引入组大小奖励，也不改变原选择。模型一致率比较数值答案而不是候选索引。单模型一致率通常为常量，是负对照。', '',
      '每组候选阈值为训练OOF信号的预设分位点及禁用选项。分别检查唯一最大答案组至少2票/至少3票；并列不回退。正确数并列优先更少答案改动。表中“回退率”指最终答案实际被改变的比例。', '',
      '| 基础方法 | 信号 | 至少票数 | OOF阈值 | OOF净纠错 | 验证准确率 % | 验证纠正/改错 | 回退率 % |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for fit in routes:
        r=get(fit['name']);threshold=fit['routing']['threshold']
        t='禁用' if threshold is None else f'{threshold:.6f}'
        lines.append(f"| {fit['base']} | {fit['routing']['signal']} | {fit['routing']['min_support']} | {t} | {fit['oof_selected']['correct_diagnostic']-fit['oof_base_correct_diagnostic']:+d} | {r['accuracy']:.3f} | {r['repairs']}/{r['harms']} | {100*r['fallback_fraction']:.2f} |")
    lines+=['','禁用回退是预先允许的实验结果。不同阈值信号/票数的所有验证成绩均保留，不能只挑最好的一个宣称普遍有效。所有回退对基础方法的配对区间见 `paired_comparisons.json`。', '',
      '无置信度筛选的投票控制：', '', '| 基础方法 | 至少票数 | 官方准确率 % | 纠正/改错 |','|---|---:|---:|---:|']
    for r in records:
        if r['group']=='unconditional_vote':lines.append(f"| {r['base']} | {r['routing']['min_support']} | {r['accuracy']:.3f} | {r['repairs']}/{r['harms']} |")
    if (out/'routing_behavior.json').exists():
        behavior=read(out/'routing_behavior.json')
        lines+=['','零阈值规则需要特别区分解析失败与真正的答案不确定性。以下为事后归因，不改变任何预测或阈值；“来自解析无效”表示原模型选中的候选没有有效数值答案。', '',
          '| 规则 | 答案改动数 | 来自解析无效 | 来自有效答案 | 纠正解析失败 | 纠正有效错误 | 改错有效答案 |',
          '|---|---:|---:|---:|---:|---:|---:|']
        for r in behavior:
            if r['threshold']==0 or r['name']=='s2_5_route_model_agreement_support3':
                lines.append(f"| {r['name']} | {r['answer_changes']} | {r['from_invalid_parse']} | {r['from_valid_answer']} | {r['invalid_parse_repairs']} | {r['valid_answer_repairs']} | {r['valid_answer_harms']} |")
        lines+=['','若零阈值规则的全部改动均来自解析无效候选，其收益应归因为解析有效性修正，不能据此证明模型识别了复杂推理中的不确定性。']
    lines+=['','## 互补性与困难子组', '',
      '下表来自保存预测与验证标签的诊断。oracle列只表示事后任选一个正确方法的上限，不是可实现策略，也不是新的官方分数。', '',
      '| 模型（或左方法） | 模型正确 | 投票（或右方法）独有正确 | 模型独有正确 | 都错 | oracle任一正确 |',
      '|---|---:|---:|---:|---:|---:|']
    for r in read(out/'complementarity.json'):
        lines.append(f"| {r['model']} | {r['model_correct']} | {r['vote_only_correct']} | {r['model_only_correct']} | {r['both_wrong']} | {r['oracle_either_correct_diagnostic']} |")
    import csv
    with (out/'subgroups.csv').open(encoding='utf-8-sig',newline='') as f:subgroups=list(csv.DictReader(f))
    names=list(dict.fromkeys(['s2_seed17','s2_5','heterogeneous_15','consistency_5',selection['name']]))
    groups=['one_correct','multiple_valid_answer_groups','unique_plurality_wrong','any_invalid_parse']
    lines+=['','子组为事后诊断，不参与阈值拟合。', '', '| 方法 | 子组 | 题数 | 正确数 | 诊断准确率 % |','|---|---|---:|---:|---:|']
    for name in names:
        for group in groups:
            r=next(r for r in subgroups if r['name']==name and r['subgroup']==group)
            lines.append(f"| {name} | {group} | {r['n']} | {r['correct_diagnostic']} | {float(r['accuracy_diagnostic']):.3f} |")
    lines+=['','显式特征在困难子组上的五种子均值，避免仅用容易题解释整体收益：', '',
      '| 配置 | 子组 | 题数 | 平均正确数 | 平均诊断准确率 % |','|---|---|---:|---:|---:|']
    for variant in ['s2','s2_parse','s2_consistency','s2_style','s2_all','features_only']:
        for group in ['one_correct','multiple_valid_answer_groups','unique_plurality_wrong']:
            selected=[r for r in subgroups if r['subgroup']==group and r['name'] in [f'{variant}_seed{s}' for s in [17,29,43,71,101]]]
            assert len(selected)==5
            lines.append(f"| {variant} | {group} | {selected[0]['n']} | {np.mean([float(r['correct_diagnostic']) for r in selected]):.1f} | {np.mean([float(r['accuracy_diagnostic']) for r in selected]):.3f} |")
    lines+=['','## 置信度与校准诊断', '',
      '每个信号拟合一个一维logistic校准器，以训练OOF选中答案正确性为标签；校准器仅用于评估置信度，不改变预测。Brier和ECE越小越好，ECE为10个等宽概率箱；容易题占比较高，同时报告mixed子组。', '',
      '| 基础方法 | 信号 | 验证Brier | 验证ECE | mixed Brier | mixed ECE |','|---|---|---:|---:|---:|---:|']
    for r in read(out/'calibration.json'):
        lines.append(f"| {r['base']} | {r['signal']} | {r['validation']['brier']:.4f} | {r['validation']['ece_10_equal_width']:.4f} | {r['validation_mixed']['brier']:.4f} | {r['validation_mixed']['ece_10_equal_width']:.4f} |")
    lines+=['','信号分箱边界仅由训练OOF确定，各箱模型与投票的正确/纠错/改错数见 `confidence_bins.csv`。低置信本身不保证投票条件准确率更高。', '',
      '## 成本、交付与验证', '',
      f"- 共{cost['full_runs']}次完整拟合、{cost['oof_runs']}次折内拟合；每次20轮，共{cost['optimizer_steps']}次优化器更新。拟合耗时合计{cost['training_wall_seconds_sum']/60:.2f}分钟，另有特征提取、独立评分和交付复核成本。",
      f"- 共记录{cost['official_validation_pipelines']}个官方验证结果（含多数投票基线），阈值候选合计{cost['threshold_candidates_total']}个；所有阈值搜索仅用训练OOF标签。校准器{cost['calibration_models']}个。",
      f"- 环境：{env['cpu'].strip()}；Python {env['python'].split()[0]}，NumPy {env['numpy']}，Torch {env['torch']}；CPU最多4数值线程，OpenBLAS限制1。",
      f"- 交付方案 {selection['name']}，资产{selection['portable_asset_bytes']/1024**2:.3f}MiB，位于 `selected_bundle`。只需推理源码和该目录，不需要研究用折内权重。",
      f"- {verification['protected_files_checked']}个原有文件哈希复核通过；所有{len(verification['all_pipelines_new_ids_reordered_and_single_question'])}个模型/流水线通过新ID、题目重排、13题及单题推理检查，返回原生int。", '',
      '| 方法 | 成员 | 完整fresh预测+评分秒 | 资产MiB |','|---|---:|---:|---:|']
    for r in resources:
        lines.append(f"| {r['name']} | {r['members']} | {r['total_fresh_wall_seconds']:.3f} | {r['uncompressed_asset_bytes']/1024**2:.3f} |")
    feature_delta=comparison('s2_consistency','s2')
    style_delta=comparison('s2_all','s2_consistency')
    ensemble_delta=comparison('s2_5','s2_seed17')
    routing_records=[r for r in records if r['group']=='routing']
    disabled=sum(r['routing']['threshold'] is None for r in routes)
    improved=sum(r['correct']>get(r['base'])['correct'] for r in routing_records)
    harmed=sum(r['correct']<get(r['base'])['correct'] for r in routing_records)
    lines+=['','本机验证集计时包括新进程启动、导入、输入/资产加载、预处理、全部成员和独立评分；不能代替官方参考CPU上的完整测试计时。', '',
      '## 从本轮得到的结论', '',
      f"- 一致性特征相对共同S2基线：{ci(feature_delta)}个百分点。"+('配对区间不含零，构成本轮较明确的改进证据。' if feature_delta['ci95_pp'][0]>0 else '区间包含零，仍需要复核稳定性。'),
      f"- 在一致性之上加入风格特征：{ci(style_delta)}个百分点。它与单独风格消融应共同解释，不能把全部特征的收益都归因于风格。",
      f"- 五成员S2相对固定seed17单模型：{ci(ensemble_delta)}个百分点。这一差值同时受该固定种子表现影响，五种子均值见特征表。",
      f"- 24组回退规则中，训练OOF选择禁用{disabled}组；验证相对其基础模型提高{improved}组、下降{harmed}组、持平{24-improved-harmed}组。即使OOF净收益为正，也应以冻结后的验证纠正/改错共同判断。",
      '- 这一轮保留失败结果和额外计算成本，不将验证最高的一条规则解释为普遍可靠的路由方法。', '',
      '### 一致性提高整体分数，但损害少数正确答案的选择', '',
      '困难子组显示一个重要取舍：显式一致性使模型更倾向群体共识。以下配对区间是观察到该现象后追加的事后诊断，未校正多重比较，不参与交付方法选择。', '',
      '| 对比 | 子组 | 题数 | 一致性方法准确率 % | 原方法准确率 % | 差值与95%描述性区间，百分点 |',
      '|---|---|---:|---:|---:|---:|']
    if (out/'consensus_tradeoff.json').exists():
        for r in read(out/'consensus_tradeoff.json'):
            lines.append(f"| {r['a']} − {r['b']} | {r['subgroup']} | {r['n']} | {r['a_accuracy_diagnostic']:.3f} | {r['b_accuracy_diagnostic']:.3f} | {ci(r)} |")
    lines+=['','值得并列保留的三个固定预测器：', '',
      '| 方法 | 官方整体准确率 % | 只有一个正确候选 %（117题，诊断） | 唯一最高票错误 %（60题，诊断） |',
      '|---|---:|---:|---:|']
    for name in ['s2_5','s2_5_route_model_agreement_support3','consistency_5']:
        one=next(r for r in subgroups if r['name']==name and r['subgroup']=='one_correct')
        wrong=next(r for r in subgroups if r['name']==name and r['subgroup']=='unique_plurality_wrong')
        lines.append(f"| {name} | {get(name)['accuracy']:.3f} | {float(one['accuracy_diagnostic']):.3f} | {float(wrong['accuracy_diagnostic']):.3f} |")
    fallback=comparison('s2_5_route_model_agreement_support3','s2_5')
    lines+=['',f"上述回退方案的训练OOF阈值为模型答案一致率≤0.8，且生成候选至少3票支持同一数值答案；验证纠正6题、改错2题，相对原集成 {ci(fallback)} 个百分点。其区间仍包含零；这里没有追加搜索新规则。", '',
      '因此，92.73%的整体最高方案不能解释为在所有困难情形下更可靠。原S2集成和上述回退方案保留在各自 `pipelines/.../pipeline.json` 中。交付选择继续遵守预先规定的整体指标，没有根据事后子组结果修改选择标准。', '',
      '## 解释边界', '',
      '- 当前验证集已用于既往大量开发；本轮未再按验证调阈值，但最终方案仍按验证选择。应保留未触碰测试集到方案冻结后的最终评价。',
      '- 配对bootstrap 10000次按题目重采样；特征实验先对五种子正确性平均，不将种子×题目当独立样本。区间条件于本轮拟合与历史开发选择，未校正多重比较。',
      '- 额外特征改变可见信息，不能把其收益归因为注意力架构更好；风格特征有效也不等于模型学会了推理验证。',
      '- OOF基础模型只用80%训练题，而交付基础模型用全部训练题，置信度分布可能发生变化；OOF有益、验证无益也是需要保留的失败结果。',
      '- 验证选择准确率是主要指标；候选BCE、校准、子组与成本用于解释，不能相互替代。', '',
      '实现由AI协助；沿用项目内原始候选评分器与S2布局。官方解析器、数据和评分器保持不变。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'delivery_source_hashes.json',{p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md') or p.name=='protocol.json'})
    print('REPORT WRITTEN',out/'REPORT.md')


if __name__=='__main__':main()
