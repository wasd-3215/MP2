"""Build a report from immutable official scores and recorded reading judgments."""
from pathlib import Path
import json
import csv
import hashlib
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True)
    ar=read(HERE/'arithmetic/results/analysis.json');fit=read(HERE/'arithmetic/results/fit_results.json')
    rel=read(HERE/'relation/results/relation_summary.json');pop=read(HERE/'relation/results/population_coverage.json')
    verify=read(OUT/'verification.json');env=read(OUT/'environment.json');assert verify['passed']
    sampling=read(HERE/'relation/results/split_and_sampling.json')
    records=ar['results'];v=next(r for r in records if r['name']=='V')
    with (OUT/'arithmetic_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['name','correct','n','accuracy','repairs','harms','net','delta_pp','ci95_pp','numerical_changes','fresh_total_seconds','asset_bytes_uncompressed']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in records)
    with (OUT/'relation_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['family','sampled_questions','questions_with_probes','reviewed_assertions','literal_faithful','semantic_binding_supported','confirmed_supported_multianswer_questions','confirmed_discriminating_questions','correct_wrong_contrast_questions','decision']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for name,r in rel['families'].items():w.writerow({'family':name,**{k:r[k] for k in cols[1:]}})
    cost={'new_base_model_training_runs':0,'new_residual_fits':3,'hyperparameter_searches':0,
      'arithmetic_fit_preparation_seconds':fit['fit_preparation_seconds'],
      'arithmetic_five_fresh_prediction_and_score_seconds':sum(r['fresh_total_seconds'] for r in records),
      'arithmetic_five_portable_replay_seconds':sum(r['portable_replay']['fresh_total_seconds'] for r in records),
      'arithmetic_validation_feature_analysis_seconds':ar['validation_extraction_seconds'],
      'relation_development_final_extraction_seconds':read(HERE/'relation/results/development_checks.json')['seconds'],
      'relation_all_train_extraction_seconds':pop['train_extraction_seconds'],
      'relation_controlled_thread_full_train_replay_seconds':verify['relation_controlled_thread_train_replay_seconds'],
      'manual_development_questions':48,'manual_confirmation_unique_questions':106,'confirmation_candidate_contexts':424,
      'confirmation_deduplicated_assertions':107,'reference_labels_loaded_after_first_review':True,
      'cost_scope':'Recorded phase costs; excludes implementation, interactive reading, source inspections and integrity hashing. Historical training/OOF/evidence reused; not charged as new work.'}
    write(OUT/'cost_accounting.json',cost)
    interpretation={'phase':'post-evaluation explanation only; no refit',
      'V_changes':[
        {'validation_row_zero_based':302,'effect':'repair','reason':'Detected9+12+8+8+3=32 rather than40. New selected answer8matches reference, but its reasoning uses32/4and has counting/semantic flaws; not proof of whole-trace reasoning verification.'},
        {'validation_row_zero_based':739,'effect':'wrong_to_wrong','reason':'Detected5600+392=6000. Replaced6420by6411.44; still incorrectly compounds interest despite original-principal condition.'},
        {'validation_row_zero_based':757,'effect':'repair','reason':'Detected15*58940=883100 rather than884100. New candidate follows884100-130000and/2to377050.'}],
      'N_changes':{'repairs_rows':[206,478],'harm_rows':[17,320,359],'wrong_to_wrong_rows':[655],
        'all_selected_pair_contributions_from_count_only':True},
      'NV_and_H_choices_identical':read(HERE/'arithmetic/results/pipelines/NV/predictions.json')['selected']==read(HERE/'arithmetic/results/pipelines/H/predictions.json')['selected'],
      'exact_two_sided_discordance_sign_test_V_vs_B':0.5,
      'sign_test_note':'Two repairs, zero harms:2*(1/2)^2=.5; small descriptive bootstrap lower bound0is not strong confirmation.',
      'scope':'Feature subsets each refitted on training OOF; not a fixed-coefficient deletion experiment.'}
    write(OUT/'arithmetic_interpretation.json',interpretation)
    names={'scope_unit':'总量／每份量及单位','entity_binding':'实体与数量绑定','condition_use':'条件使用','target_alignment':'所求量与末步运算'}
    lines=['# 算术代理消融与训练集关系证据可行性审计','',
      '完成日期：2026-10-18（北京时间）。先完成算术协议冻结、拟合、官方评价及解释，再开展训练集关系开发与冻结检查。全部新增文件位于本独立实验目录。','',
      '**主要结果：仅矛盾证据V为742/798，比基线净增2题（修复2、损害0）；仅等式数量N为739，数量＋矛盾NV为741。四类当前关系探针均未达到预设建模门槛。本轮没有训练关系选择模型，也没有替换既有预测入口。**','',
      '**算术对照与冻结设置**','',
      '冻结S2-C五种子17/29/43/71/101，平均sigmoid概率。官方解析、有效性过滤和最终算术提取器逐字节复用。训练仅使用现有训练OOF与训练最终数值标签，新增3次确定性线性残差拟合，主模型训练0次。分数为logit(平均概率)+特征·β，固定基线系数1、无截距、原始尺度、BCE求和＋0.5||β||²、L2=1，无超参或阈值搜索。','',
      'B无新增特征；N为log1p可检查等式数；V为log1p矛盾数及末条可检查等式矛盾标记；NV合并三项。主实验去掉错误等式比例，避免分母重新引入等式数量。H直接复用历史完整四维算术校正系数，包含比例，只作复现参照。无证据或共同偏移保留原排序。','',
      '| 方案 | 正确/798 | 官方准确率 | 修复／损害 | 净增 | 数值答案变化 | 对B的95%描述区间pp |','|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        lo,hi=r['ci95_pp'];lines.append(f'| {r["name"]} | {r["correct"]}/798 | {r["accuracy"]:.3f}% | {r["repairs"]}／{r["harms"]} | {r["net"]:+d} | {r["numerical_changes"]} | [{lo:.3f}, {hi:.3f}] |')
    lines+=['','| 预设比较 | 差值pp | 95%描述区间pp |','|---|---:|---|']
    for name,c in ar['comparisons'].items():lines.append(f'| {name} | {c["delta_pp"]:+.3f} | [{c["ci95_pp"][0]:.3f}, {c["ci95_pp"][1]:.3f}] |')
    lines+=['','区间为10000次整题配对bootstrap，seed20261018。基线740及历史741的选择均逐项复现；NV与H最终选择也完全相同。V的区间接触0，只出现2个得分分歧，精确双侧配对符号检验p=.5。验证集反复用于开发，不能将这一结果认定为独立确认的稳定泛化提升。不同特征子集分别重新拟合，不能解释为单纯删除旧系数后的因果贡献。','',
      '| 方案 | 多答案355 | 错误最高票60 | 唯一正确117 | 历史共同错42修复 | 过程有错但数值正确6题修复 |','|---|---:|---:|---:|---:|---:|']
    for r in records:
        g=r['subgroups'];lines.append(f'| {r["name"]} | {g["multi_answer"]["correct"]} | {g["wrong_plurality"]["correct"]} | {g["one_correct"]["correct"]} | {g["common42"]["correct"]} | {g["flawed_correct6"]["correct"]} |')
    lines+=['','子组重叠，不能相加。V的2次修复：M&M题排除了9+12+8+8+3=32的错误求和，但新选数值正确候选自身仍有计数解释问题；人口题排除了15×58940=883100的错误乘法，新选377050数值正确。另一次把6420改为6411.44仍然错误：算术更一致，但仍违背按原本金计息条件。完整原文、公式及分数贡献保存在arithmetic/results/answer_changes.jsonl，人工解释在results/arithmetic_interpretation.json。','',
      'N单独修复2题、损害3题，净−1；所有选择变化的直接新增贡献都来自数量。NV相对V增加数量后少对1题。证据支持在这一冻结基线下保留明确矛盾校正、谨慎对待等式数量代理；不证明任意算术检查均有效，也不意味着V验证了整段推理。','',
      '**关系审计：样本、探针与阅读边界**','',
      '仅使用4755道训练输入，按整题seed20261018划为开发2853、冻结检查1902。该划分仅留出本轮提取器开发，既有主模型已经使用这些训练题，不是主模型的新独立留出集。正例按各族筛选命中的多有效答案题随机抽样，不使用标签或分数；开发每族12题共48道。冻结检查为25/16/25/25题，另20道全族筛选未命中的随机对照，去重106道、424份候选最终上下文。','',
      '第一遍不读取训练参考答案或模型分数：阅读全部题目、四候选最终上下文和全部107条去重后的提取源片段；4道有疑问的题另展开四候选全文。阅读范围不等于逐步证明全部424份完整解答。逐题判断由Codex单独完成，没有独立第二位人工评审。标注先冻结并记录哈希，再由finalize.py读取训练参考答案做数值标签关联；人工标注不用于拟合。','',
      '当前探针范围刻意有限：单位族为显式数字费率的单位元组；实体族为相邻数字的性别计数；条件族为明确序号年/月/周/日的倍数修饰；目标族为问题所求类别与可解析数值等式运算形状。重复条件或缺少词语均不被认定为实际使用或遗漏。开发阶段保留v0并修复嵌入what提问截取，加入阶段倍数和带单位的诊断运算提取；冻结检查之后未再修改。','',
      '| 关系族 | 抽样题 | 有探针题 | 判断条数 | 字面／类型提取正确 | 可确认语义对应 | 支持题 | 可区分题 | 决定 |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,r in rel['families'].items():
        lines.append(f'| {names[name]} | {r["sampled_questions"]} | {r["questions_with_probes"]} | {r["reviewed_assertions"]} | {r["literal_faithful"]}/{r["reviewed_assertions"]} | {r["semantic_binding_supported"]}/{r["reviewed_assertions"]} | {r["confirmed_supported_multianswer_questions"]} | {r["confirmed_discriminating_questions"]} | 不进入建模 |')
    lines+=['','表内只统计各族预定正例样本。20道负对照另贡献23条判断，总审阅107条，字面／类型正确101条、可确认语义对应74条；这些比例不代表总体精度。重复关系在每候选内按关系键去重，避免同一事实反复复述充大样本。相同题内判断仍相关，给出的Wilson区间仅作描述，不是独立样本置信保证。','',
      '预设继续门槛为：至少20道确认支持的多答案题，至少40条判断来自20道题，语义证据支持比例至少95%，至少10道可区分候选题，并存在与训练最终正确性有关的对照。本轮未知或未绑定不能计为可靠证据，失败不等于已证明该关系本身错误。四族均未通过。实体族冻结检查输入池本身只有16道，未追加或重抽以越过门槛。','',
      '负对照也发现筛选漏项：平均每地点、总量、差额等题未被正例筛选覆盖。20个负对照不能估计总体召回率。全部训练输入上，冻结探针分别在215、29、14、1572题产生原始判断；原始覆盖不是经人工确认的可用覆盖。条件族整个冻结检查1902题中仅4题有原始阶段探针，而抽样25题均没有，表明这类安全模板覆盖不足，不能扩张为任何条件提取均无效的结论。','',
      '**失败机制与少量可用现象**','',
      '- 单位族：售价0.40、成本0.25、利润0.15均为美元/苹果，单位相同不代表同一关系；冻结检查还发现负费率被取成正数，原样记录，未修补重测。\n- 实体族：多数候选准确复述人数却在后续计算出错。row3810的增加1名男生却把女生13改14提供了1道真正区分，但覆盖不足。\n- 条件族：开发木瓜树例可比较第4年twice与50%more，但泛化到命名月份、人物、折扣或剩余量需要更多绑定能力。\n- 目标族：求剩余人数可除以2，求剩余花瓣可乘以每朵花瓣数，求净总收入可最后作减法；粗运算形状会误伤等价表达。反之，原数额与剩余数额、所需总量与每份量有少量清楚对照，冻结检查只确认2道可区分题。','',
      '这些是对当前轻量探针的可行性结论，不能解释为所有题目侧关系方法无效。下一次若继续，应先在新的开发/检查安排下解决实体、阶段、单位角色和计算依赖的绑定，不能直接把本轮失效模式改成硬规则再在同批检查题上宣称通过。','',
      '**资源、保护与复现**','',
      f'本机{env["cpu"]}，Python3.12.14、NumPy2.5.3、PyTorch2.7.1+cpu；算术数值任务串行，Torch4/interop1、BLAS1。确认阶段全训练提取另在导入NumPy前限制线程的独立进程重放，与全部424份保存证据逐项相同，耗时{verify["relation_controlled_thread_train_replay_seconds"]:.2f}秒；这是全4755题的审计提取成本，不属于部署预测成本。','',
      '| 方案 | 完整验证新进程预测＋独立评分秒 | 展开后推理资产MiB |','|---|---:|---:|']
    for r in records:lines.append(f'| {r["name"]} | {r["fresh_total_seconds"]:.3f} | {r["asset_bytes_uncompressed"]/1024**2:.5f} |')
    lines += ['',f'3次校正拟合及准备{cost["arithmetic_fit_preparation_seconds"]:.2f}秒；5次主评价合计{cost["arithmetic_five_fresh_prediction_and_score_seconds"]:.2f}秒，5次便携回放合计{cost["arithmetic_five_portable_replay_seconds"]:.2f}秒。旧主模型、OOF和训练算术特征复用，历史成本未伪记为本轮新训练。完整阶段成本见cost_accounting.json，未计实现、阅读与哈希耗时。','',
      f'已核对{verify["protected_files"]}个开始前既有文件，变更0。官方数据、解析器、评分器、预测入口及旧模型/实验全部保留。新增便携包不含参考答案、OOF或预测缓存，五套配置共享相同5个主模型；每个方案小于16MiB。完成26项旧提取器检查、9项关系边界检查、5次官方评价与5次便携官方回放、改ID/重排/单题/空输入/候选置换/解析回退检查。未使用测试集语义内容、预测或标签；仅按文件字节校验其哈希。本机验证集时间不替代官方Xeon完整测试评测的30秒认证。','',
      '所有代码与审阅由Codex AI辅助完成，未引入外部数据、预训练权重或推理API。参考README.md。可移植推理从portable/inference/method.py与portable/pipelines/V加载；复现脚本会建立新的同级实验目录，不覆盖本轮记录。人工标注作为固定输入回放，不会自动产生独立第二次人工审阅。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'delivery_hashes.json',{'files':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
      for p in HERE.rglob('*') if p.is_file() and p.name!='delivery_hashes.json'},'test_used':False})
    print({'report':str(OUT/'REPORT.md'),'V_correct':v['correct'],'protected_files':verify['protected_files'],'all_current_relation_probes_no_go':all(s['decision']=='NO_GO_CURRENT_PROBE' for s in rel['families'].values())},flush=True)

if __name__=='__main__':main()
