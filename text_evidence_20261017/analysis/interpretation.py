"""Posthoc score decomposition only; never refits or evaluates another predictor."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
from utils import HERE,OUT,CODE,read,write,sha,sources
import numpy as np
from common import load_inputs,parse_answer
from decisions import base_logits
from evidence import NAMES

def main():
    frozen=read(OUT/'frozen_before_validation.json')
    assert sources()==frozen['sources']
    bank,_=load_inputs('validation')
    f=np.load(OUT/'validation-evidence.npy')
    p_path=CODE/'method_complementarity_20261017/results/validation-probabilities.npz'
    with np.load(p_path) as a:p=a['S2_C'].mean(0)
    z=base_logits(p)
    baseline=np.asarray(read(OUT/'pipelines/baseline/predictions.json')['selected'])
    new=np.asarray(read(OUT/'pipelines/arithmetic/predictions.json')['selected'])
    beta=np.asarray(read(OUT/'pipelines/arithmetic/pipeline.json')['beta'])
    rows=[]
    for q,(old,i) in enumerate(zip(baseline,new)):
        answers=[parse_answer(c['text']) for c in bank[q]['candidates']]
        if answers[int(old)]==answers[int(i)]:continue
        difference=f[q,int(i),3:]-f[q,int(old),3:]
        contribution=difference*beta
        rows.append({'row':q,'question':bank[q]['question'],'old_candidate':int(old),'new_candidate':int(i),
          'old_answer':answers[int(old)],'new_answer':answers[int(i)],'baseline_old_minus_new_logit':float(z[q,int(old)]-z[q,int(i)]),
          'new_minus_old_feature_contributions':dict(zip(NAMES[3:],contribution.tolist())),
          'old_new_both_have_no_detected_violation':bool(f[q,int(old),4]==0 and f[q,int(i),4]==0),
          'only_pairwise_contribution_from_equation_count':bool(np.all(contribution[1:]==0))})
    format_cases=[]
    cfg=read(OUT/'pipelines/format/pipeline.json')
    for q in [331,733]:
        old=int(baseline[q]);right=[]
        # Reference-independent diagnosis of rank movement: compare with previously identified correct candidate0.
        comparison=0
        shifts=f[q,:,:3]@np.asarray(cfg['beta'])
        format_cases.append({'row':q,'old_candidate':old,'comparison_candidate':comparison,
          'old_probability':float(p[q,old]),'comparison_probability':float(p[q,comparison]),
          'old_minus_comparison_logit_before':float(z[q,old]-z[q,comparison]),
          'old_minus_comparison_logit_after':float(z[q,old]+shifts[old]-z[q,comparison]-shifts[comparison]),
          'old_offset':float(shifts[old]),'comparison_offset':float(shifts[comparison])})
    supplement={'phase':'after frozen official evaluation; diagnostic decomposition only','no_new_predictor_or_fit':True,
      'arithmetic_numerical_changes':rows,'format_case_rank_margins':format_cases,
      'input_probability_sha256':sha(p_path),'top_level_frozen_sources_unchanged':True}
    write(OUT/'posthoc_interpretation.json',supplement)
    summary=read(OUT/'summary.json')
    report=OUT/'REPORT.md';text=report.read_text(encoding='utf-8')
    lead='**本组结果：算术及合并校正均为741/798，比740基线净增1题（修复3、损害2）；输出结构校正没有改变最终数值选择。这个幅度未证明稳定改进，且多数变化含有等式数量代理的影响。**\n\n'
    text=text.replace(lead,'').split('\n## 得失解释与研究结论')[0].rstrip()+'\n'
    text=text.replace('# 输出结构与算术证据校正实验\n\n','# 输出结构与算术证据校正实验\n\n'+lead,1)
    audit_passes=[read(HERE/'audit_history'/v/'training_audit.json')['extraction_seconds'] for v in ['v0','v1']]
    audit_passes.append(read(OUT/'training_audit.json')['extraction_seconds'])
    records=read(OUT/'official_results.json')
    cost={'new_base_training_runs':0,'linear_residual_fits':3,'hyperparameter_searches':0,
      'training_only_extractor_audit_passes':3,'training_extraction_seconds_each':audit_passes,
      'training_extraction_seconds_total':sum(audit_passes),'residual_preparation_and_fitting_seconds':read(OUT/'fit_results.json')['total_seconds'],
      'four_fresh_official_prediction_and_score_seconds':sum(r['fresh_total_seconds'] for r in records),
      'four_portable_replays_seconds':sum(r['portable_replay']['fresh_total_seconds'] for r in records),
      'scope_note':'Recorded process phase costs; excludes manual review time and extra implementation verification.'}
    write(OUT/'cost_accounting.json',cost)
    notes=['','## 得失解释与研究结论','',
      '1. **净增1题是很弱的正向信号。** 741/798=92.857%，较基线增加0.125个百分点；配对描述区间[-0.376,+0.752]跨0。arithmetic与combined的数值选择完全相同，加入格式证据没有额外收益。不能将本轮开发集+1题当作独立确认的泛化提升，也没有据此替换旧正式入口。','',
      '2. **应区分矛盾检查与风格代理。** 算术族包含可检查等式数量，训练出的该项系数为负。在5处正确性变化中，原数据行303（M&M题）确有错误等式从所选候选中被排到后面：9+12+8+8+3被写成32，校正后从6.4选到官方正确8。另两处修复（票价、布料）和两处损害（测验平均分、每周车流）对比的旧新候选均无检测到的矛盾，直接排序翻转只来自等式数量差异。这项代理偏好较少可解析算式，产生2修复2损害；它不代表理解了算术真伪。诊断分解见posthoc_interpretation.json，未另外拟合或评价删去该项的新模型。','',
      '3. **42题中的改善同样有限。** 修复2/42：M&M题通过显式错误求和证据，票价题通过等式数量代理。另有两处把错误答案换成另一错误答案，仍不计修复。简单利息题从6420换到6411.44，后者算术更一致，却仍按复利计算，违背“每年按原本金”的条件；这直接展示算术一致性不足以检查题意。纯数值提取仅在7/42题的某候选中找到矛盾、4/42题的原选候选中找到矛盾，不能覆盖此前33题数量关系/条件/目标的主体问题。','',
      '4. **格式线索没有变成选择收益，不能仅解释为线索不存在。** 训练的数值框冲突系数约-3.23，但在原数据行734的Job3/boxed250案例，校正后原错候选仍高于可正确解析的候选；行332的截断代数框也未被纠正。下表给出同两候选的冻结分数差：','',
      '| 原数据行 | 校正前错误候选领先logit | 格式校正后领先logit |','|---|---:|---:|']
    for c in format_cases:notes.append(f'| {c["row"]+1} | {c["old_minus_comparison_logit_before"]:.4f} | {c["old_minus_comparison_logit_after"]:.4f} |')
    notes+=['','这些稀少信号（训练仅22个数值框冲突、11个符号框）由最终正确性BCE校正学习，符号框和末尾运算符系数甚至为正；这是当前基线条件下拟合的残差，不是人为设定的“坏格式必扣分”。本轮没有据验证结果加强处罚、改成硬过滤或搜索阈值。','',
      '5. **下一步应先澄清证据，而非扩大本组。** 如果沿算术方向继续，最小机制对照是去掉“可检查等式数量”代理，仅保留明确矛盾证据，并在训练内冻结；这项尚未执行，不能从本轮事后拆解推算它的新分数。若投入更大工作量，优先题干—解答的数量关系、单位及条件对齐；不建议继续堆选专家层或仅调整答案组票数。','',
      '本轮完成26项提取器检查、28张最终训练卡人工审阅、4组新进程官方评分、4组便携重放，以及新ID、题目重排、单题、候选置换和解析回退检查。11,179个既有文件哈希全部不变；测试集未使用。训练阶段的两次提取修复分别保留audit_history/v0和v1，冻结后没有再改动提取器或拟合配置。', '',
      f'成本补充：三版训练提取审计总计{cost["training_extraction_seconds_total"]:.2f}秒，校正准备及拟合{cost["residual_preparation_and_fitting_seconds"]:.2f}秒；4组原始官方预测评分合计{cost["four_fresh_official_prediction_and_score_seconds"]:.2f}秒，4组便携重放合计{cost["four_portable_replays_seconds"]:.2f}秒。阶段成本未计人工阅读和额外实现验证，详见cost_accounting.json。最终解释复现需在analyze.py、verify.py之后运行analysis/interpretation.py；该脚本仅拆解冻结分数，不产生新的模型成绩。']
    report.write_text(text+'\n'.join(notes)+'\n',encoding='utf-8')
    verification=read(OUT/'verification.json')
    verification['posthoc_analysis_did_not_change_frozen_sources_or_predictions']=True
    verification['posthoc_interpretation_script_sha256']=sha(Path(__file__))
    verification['final_report_sha256']=sha(report)
    write(OUT/'verification.json',verification)
    print({'changed_rows':[r['row'] for r in rows],'format_margins':format_cases,'protected_passed':verification['passed']},flush=True)

if __name__=='__main__':main()
