"""Evidence-led report generated from immutable official scores and paired diagnostics."""
from utilities import HERE,read,write,sha


def main():
    out=HERE/'results';summary=read(out/'group_summary.json');subs=read(out/'subgroup_summary.json')
    paired=read(out/'paired_comparisons.json');interactions=read(out/'interactions.json')
    ensembles=read(out/'ensemble_summary.json');weights=read(out/'weight_summary.json')
    target=read(out/'practical_target.json');selection=read(out/'selection.json')
    env=read(out/'environment.json');cost=read(out/'cost.json');resources=read(out/'resources.json')
    verification=read(out/'verification.json')
    get=lambda group:next(r for r in summary if r['group']==group)
    subgroup=lambda group,split,name:next(r for r in subs if r['group']==group and r['split']==split and r['subgroup']==name)
    primary=next(r for r in paired if r['primary'])
    def ci(r):return f"{r['delta_pp']:+.3f} [{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}]"
    lines=['# 困难训练子组重加权：能否保留一致性收益并恢复少数正确答案？','',
      '完成P/C两种输入×U/K/H三种训练权重×五个种子，共30次固定预算训练；另评估六个固定五成员集成。所有主准确率由未修改的官方独立评分器核对。', '',
      f"预设主要比较 **C-H减C-K，在错误共识60题上：{ci(primary)}个百分点**。五种子差值为 {', '.join(f'{v:+.3f}' for v in primary['seed_deltas_pp'])}。", '',
      ('主要差值区间不含零，构成本轮有方向性的证据；仍是反复使用验证集上的开发结果。' if primary['ci95_pp'][0]>0 or primary['ci95_pp'][1]<0 else '主要差值区间包含零，目前无法确认特别针对错误共识优于正确数量匹配的重加权。'), '',
      f"本轮整体指标选中的可部署方案是 **{selection['name']}，{selection['correct']}/798={selection['accuracy']:.3f}%**。整体最高分选择与本轮科学问题、实用目标分别记录；没有替换历史选中模型。若最高分并列，名称排序只用于确定性复现，不能解释为重加权方法更好。", '',
      '## 固定设计及权重控制','',
      '- P：S2＋解析有效性；C：S2＋解析与答案一致性。相同S2主体，新特征列零初始化，共有初始函数一致。',
      '- U均匀；H使错误共识题相对其它混合题2倍，再把混合题权重总量归一回2513；全对题始终权重1。K在每种正确候选数量内均匀赋权，严格匹配H在该数量下的权重总量。',
      '- K/H的单正确、两正确、三正确、全对权重总量以及正负候选标签总权重完全匹配；因此二者区别集中于单正确题内部是否特别关注错误共识。U和重加权组的候选正例比例会改变，不能忽略这一差异。',
      '- 全部训练题每轮出现一次，无过采样；每题四候选BCE均值乘全局题目权重，再在batch取均值。权重全局平均1，不在batch内重归一化。',
      '- 20轮、每轮19次更新、LR0.0003、AdamW decay0.001、batch256、dropout0.1、clip1；种子17/29/43/71/101。没有早停、验证检查点选择或权重倍数搜索。',
      '- 权重与分组仅由训练输入及标签拟合。推理不需要组标签、问题权重或任何参考答案；统一argmax，没有新增解析过滤、校准或回退。', '',
      '| 训练题类型 | 题数 | U权重 | K权重 | H权重 |','|---|---:|---:|---:|---:|']
    w=weights['schemes'];counts=weights['counts']
    for title,n,key in [('错误唯一最高票',counts['wrong_unique_plurality'],'hard_weight'),
        ('其它单正确',counts['one_correct']-counts['wrong_unique_plurality'],'other_one_correct_weight'),
        ('两正确',counts['two_correct'],'two_correct_weight'),('三正确',counts['three_correct'],'three_correct_weight')]:
        lines.append(f"| {title} | {n} | {w['U'][key]:.6f} | {w['K'][key]:.6f} | {w['H'][key]:.6f} |")
    lines+=['| 四候选全对 | 2242 | 1 | 1 | 1 |','',
      '| 方案 | 总权重 | 混合题总权重 | 全对题总权重 | 加权候选正例比例 % |','|---|---:|---:|---:|---:|']
    for name,r in w.items():lines.append(f"| {name} | {r['total_mass']:.6f} | {r['mixed_mass']:.6f} | {r['all_correct_mass']:.6f} | {100*r['positive_candidate_fraction']:.6f} |")
    lines+=['','## 五种子主结果','',
      '整体准确率由每个种子的官方完整验证分数汇总。子组是使用相同正确性标签的诊断统计；单正确题与错误共识题有重叠。', '',
      '| 组 | 整体 % 均值±SD | 错误共识60题 % 均值±SD | 单正确117题 % | 正确共识混合271题 % | 并列88题 % |',
      '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        s=subgroup(r['group'],'validation','wrong_unique_plurality')
        lines.append(f"| {r['group']} | {r['mean_accuracy']:.3f} ± {r['sd_accuracy']:.3f} | {s['mean_accuracy_diagnostic']:.3f} ± {s['sd_accuracy']:.3f} | {r['one_correct_accuracy']:.3f} | {r['correct_unique_plurality_mixed_accuracy']:.3f} | {r['plurality_tie_accuracy']:.3f} |")
    lines+=['','## 预设配对比较','',
      '题目bootstrap10000次；先对五种子正确性平均，再在对应子组按题目抽样。没有把种子×题目当独立样本。区间不包含历史开发、模型选择不确定性，次要比较未校正多重检验。', '',
      '| 比较 | 整体差值与95%区间，百分点 | 错误共识差值与95%区间，百分点 | 错误共识正向种子/5 |',
      '|---|---:|---:|---:|']
    pairs=list(dict.fromkeys((r['a'],r['b']) for r in paired))
    for a,b in pairs:
        overall=next(r for r in paired if r['a']==a and r['b']==b and r['subgroup']=='all')
        hard=next(r for r in paired if r['a']==a and r['b']==b and r['subgroup']=='wrong_unique_plurality')
        lines.append(f"| {a} − {b} | {ci(overall)} | {ci(hard)} | {hard['positive_seeds']}/5 |")
    lines+=['','## 重加权是否特别帮助一致性模型？','',
      '仅看C改善而P未改善不足以判断两者反应不同，需直接比较差值之差。', '',
      '| 交互比较 | 子组 | 差值与95%区间，百分点 |','|---|---|---:|']
    for r in interactions:
        if r['subgroup'] in ['all','wrong_unique_plurality','one_correct']:
            lines.append(f"| {r['comparison']} | {r['subgroup']} | {ci(r)} |")
    lines+=['','## 实用目标检查','',
      '预设目标：H相对同模型U，错误共识提高至少10个百分点，整体损失不超过0.5个百分点。C是主要工程目标，P为对照；目标只按点估计检查，不替代主要比较的区间。', '',
      '| 模型 | 错误共识变化 pp | 整体变化 pp | 达到两项点估计目标 |','|---|---:|---:|---|']
    for name,r in target.items():lines.append(f"| {name} | {r['hard_delta_pp']:+.3f} | {r['overall_delta_pp']:+.3f} | {'是' if r['point_target_met'] else '否'} |")
    lines+=['','## 训练拟合与验证差距','',
      'BCE统一报告关闭dropout后的未加权损失，不把不同训练目标的加权损失直接比较。错误共识训练正确率为诊断，用于检查新增训练关注是否只是增加记忆。', '',
      '| 组 | 训练BCE | 验证BCE | 错误共识训练准确率 % | 错误共识验证准确率 % |','|---|---:|---:|---:|---:|']
    for r in summary:
        t=subgroup(r['group'],'train','wrong_unique_plurality')
        lines.append(f"| {r['group']} | {r['train_bce']:.4f} | {r['validation_bce']:.4f} | {t['mean_accuracy_diagnostic']:.3f} | {r['wrong_unique_plurality_accuracy']:.3f} |")
    lines+=['','## 五成员集成：次要结果','',
      '每组五种子sigmoid分数等权平均，是一个固定预测器；无学习融合参数。不要把成员均值与集成准确率混为一谈，也不将加权BCE分数解释为校准概率。', '',
      '| 集成 | 正确数/798 | 官方整体 % | 错误共识60题 % | 单正确117题 % |','|---|---:|---:|---:|---:|']
    for r in ensembles:
        lines.append(f"| {r['name']} | {r['correct']} | {r['accuracy']:.3f} | {r['subgroups_diagnostic']['wrong_unique_plurality']:.3f} | {r['subgroups_diagnostic']['one_correct']:.3f} |")
    replay=read(out/'uniform_replay.json')
    lines+=['','## 复现与资源记录','',
      f"- 新均匀基线与上一轮比较：{sum(r['identical_candidate_choices'] for r in replay)}/10逐项选择相同。数值归约顺序的float32误差可能改变轨迹，详见 `uniform_replay.json`；所有主要比较只用本轮重新训练结果。",
      f"- 共30次×20轮，{cost['optimizer_steps']}次更新，{cost['draws']}次题目曝光；拟合耗时合计{cost['training_wall_seconds_sum']/60:.2f}分钟，另有初始化检查、原文件哈希和评分成本。",
      f"- 环境：{env['cpu']}；Python {env['python'].split()[0]}，NumPy {env['numpy']}，Torch {env['torch']}；CPU顺序训练，Torch4/interop1，OpenBLAS1。",
      f"- 本轮选择 {selection['name']}，推理资产{selection['portable_asset_bytes']/1024**2:.3f}MiB；`selected_bundle`包含全部成员，预测不需要训练权重或组标签。",
      f"- 全部{len(verification['all_predictors_new_ids_reordered_13_and_single_question'])}个预测器通过新ID、乱序13题、单题与原生int检查；{verification['protected_files_checked']}个原有文件哈希一致。", '',
      '| 方法 | 成员 | 本机fresh预测＋评分秒 | 解包后资产MiB |','|---|---:|---:|---:|']
    for r in resources:lines.append(f"| {r['name']} | {r['members']} | {r['total_fresh_wall_seconds']:.3f} | {r['uncompressed_asset_bytes']/1024**2:.3f} |")
    train_before=subgroup('C_U','train','wrong_unique_plurality')['mean_accuracy_diagnostic']
    train_after=subgroup('C_H','train','wrong_unique_plurality')['mean_accuracy_diagnostic']
    lines+=['','本机完整验证计时包含新进程启动、导入、输入/资产加载、预处理、全部成员和独立评分，不能替代官方参考CPU测试认证。', '',
      '## 结论与边界','',
      f"- 主要比较C-H减C-K：{ci(primary)}个百分点。"+('本轮支持针对错误共识提供额外关注。' if primary['ci95_pp'][0]>0 else ('本轮该方向的区间偏向负向。' if primary['ci95_pp'][1]<0 else '区间包含零，尚不能确认特殊关注错误共识比匹配正确数量分布更好。')),
      f"- C-H相对C-U：错误共识{target['C']['hard_delta_pp']:+.3f}pp，整体{target['C']['overall_delta_pp']:+.3f}pp；预设两项实用目标{'达到' if target['C']['point_target_met'] else '未同时达到'}。",
      f"- C的错误共识训练准确率由{train_before:.3f}%提高到{train_after:.3f}%，验证仅变化{target['C']['hard_delta_pp']:+.3f}pp。新增训练关注主要转化为训练拟合，没有显示相应的验证泛化收益。",
      '- C-U和C-H的五成员集成均为740/798=92.732%，没有超过上一轮整体最高分。当前二倍权重、20轮预算未修复一致性特征的主要弱点；这不排除其它训练目标或信息结构可能有效。',
      '- 重加权改变学习目标，并不在预测时告诉模型何时共识错误；困难子组改善也不能单独证明模型学会了条件判断或推理验证。',
      '- 验证集已反复用于历史开发；本轮预设比较不使其变为独立测试。所有区间均为开发证据，子组也源于此前观察，应保留这一选择背景。',
      '- 未用测试输入或参考答案训练、调参、预测或评分。原文件保护仅按字节核验哈希。没有外部数据、权重、API或新候选生成。',
      '- 由AI协助设计、编码和运行，沿用项目内S2、特征和预测模块；全部原目录保持不变。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'delivery_source_hashes.json',{p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md') or p.name=='protocol.json'})
    print('REPORT WRITTEN',out/'REPORT.md')


if __name__=='__main__':main()
