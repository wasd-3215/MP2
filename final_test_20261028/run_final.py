"""One frozen six-method final test; official runner/scorer remain unchanged."""
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import winreg
import zipfile

HERE=Path(__file__).resolve().parent
CODE=HERE.parent
ROOT=CODE.parent
for key,value in {'PYTHONDONTWRITECODE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8',
                  'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
                  'BLIS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE'}.items():
    os.environ[key]=value

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def asset_size(paths):
    result=0
    for p in set(paths):
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as archive:result+=sum(item.file_size for item in archive.infolist())
        else:result+=p.stat().st_size
    return result
def command(args,log):
    start=time.perf_counter()
    result=subprocess.run(args,cwd=CODE,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8')
    seconds=time.perf_counter()-start
    Path(log).write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(f'Command failed: {log}\n{result.stderr}')
    return seconds

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if (HERE/'frozen_before_test.json').exists():
        raise RuntimeError('This final test is already frozen. Preserve it; do not overwrite or adapt methods.')
    # Snapshot covers every file that existed before this new evaluation folder.
    original={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and HERE not in p.parents}
    write(HERE/'protected_before.json',original)
    manifest=read(CODE/'PACKAGE_MANIFEST.json')
    release_mismatch=[name for name,h in manifest['files'].items() if sha(ROOT/name)!=h]
    assert not release_mismatch,release_mismatch
    arith=CODE/'arith_relation_audit_20261018'
    variants=[]
    labels={'V':'S2-C＋仅矛盾证据','H':'S2-C＋历史完整算术校正','NV':'S2-C＋数量与矛盾','B':'S2-C五成员集成'}
    for name in ['V','H','NV','B']:
        variants.append({'name':name,'label':labels[name],'method':arith/'portable/inference/method.py',
          'checkpoint':arith/'portable/pipelines'/name,'validation_score':arith/'arithmetic/results/pipelines'/name/'score.json'})
    variants.extend([
      {'name':'S2_route','label':'S2五成员＋固定回退＋有效性过滤',
       'method':CODE/'validity_filter_20261017/portable/inference/method.py',
       'checkpoint':CODE/'validity_filter_20261017/portable/s2_ensemble5_route_filtered',
       'validation_score':CODE/'validity_filter_20261017/results/pipelines/s2_ensemble5_route_filtered/score.json'},
      {'name':'Pool_P','label':'Pool-P五成员＋有效性过滤',
       'method':CODE/'interaction_consensus_20261017/portable/inference/method.py',
       'checkpoint':CODE/'interaction_consensus_20261017/portable/Pool_P_ensemble5_filtered',
       'validation_score':CODE/'interaction_consensus_20261017/results/pipelines/Pool_P_ensemble5_filtered/score.json'}])
    frozen_assets={};records=[]
    for v in variants:
        config=read(v['checkpoint']/'pipeline.json')
        members=[(v['checkpoint']/p).resolve() for p in config['members']]
        paths=[v['checkpoint']/'pipeline.json']+[p for m in members for p in m.rglob('*') if p.is_file()]
        paths += [p for p in v['method'].parent.glob('*.py')]
        for p in paths:frozen_assets[str(p)]=sha(p)
        count=asset_size([p for p in paths if p.suffix!='.py'])
        assert count<16*1024**2
        score=read(v['validation_score']);assert score['split']=='validation' and score['n']==798
        folder=HERE/'results'/v['name'];folder.mkdir(parents=True)
        predict=[sys.executable,'-B',str(CODE/'predict.py'),'--method',str(v['method']),
                 '--checkpoint',str(v['checkpoint']),'--split','test','--output',str(folder/'predictions.json')]
        scoring=[sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','test',
                 '--predictions',str(folder/'predictions.json'),'--output',str(folder/'score.json')]
        records.append({'name':v['name'],'label':v['label'],'method':str(v['method']),'checkpoint':str(v['checkpoint']),
          'validation_score_file':str(v['validation_score']),'validation_correct':score['correct'],'validation_n':score['n'],
          'validation_accuracy':score['accuracy'],'asset_bytes_uncompressed':count,
          'prediction_command':predict,'scoring_command':scoring})
    for p in [CODE/'predict.py',CODE/'score_predictions.py',CODE/'common.py',Path(__file__)]:frozen_assets[str(p)]=sha(p)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    frozen={'date_client_timezone':'2026-10-28 Asia/Shanghai','primary_selected_using_validation_only':'V',
      'selection_reason':'V has highest frozen validation accuracy742/798; other five are fixed final scientific comparisons.',
      'test_result_will_not_select_or_modify_methods':True,'all_test_predictions_before_any_test_scoring':True,
      'variants':records,'hashes':frozen_assets,'official_release_manifest_verified':True,
      'environment':{'python':sys.version,'executable':sys.executable,'cpu':cpu,'platform':platform.platform(),
        'thread_environment':{key:os.environ[key] for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS']},
        'numerical_processes':'serial, PyTorch intra4/interop1 in frozen runtime, BLAS1','reference_machine':False}}
    write(HERE/'frozen_before_test.json',frozen)
    print(f'Frozen six methods; primary V; protected {len(original)} files.',flush=True)
    timings={}
    # Complete every frozen choice before any test reference is semantically read.
    for record in records:
        folder=HERE/'results'/record['name']
        assert all(sha(p)==h for p,h in frozen_assets.items())
        seconds=command(record['prediction_command'],folder/'predict.log')
        prediction=read(folder/'predictions.json')
        assert prediction['split']=='test' and len(prediction['selected'])==467
        assert all(type(i) is int and 0<=i<4 for i in prediction['selected'])
        timings[record['name']]={'prediction_wall_seconds':seconds,'prediction_sha256':sha(folder/'predictions.json')}
        write(HERE/'prediction_completion.json',timings)
        print(record['name']+' prediction complete',flush=True)
    write(HERE/'all_predictions_frozen_before_scoring.json',{'prediction_hashes':{n:v['prediction_sha256'] for n,v in timings.items()},
      'test_reference_answers_read_by_orchestrator':False,'all_methods_already_frozen':True})
    results=[]
    for record in records:
        folder=HERE/'results'/record['name'];assert sha(folder/'predictions.json')==timings[record['name']]['prediction_sha256']
        seconds=command(record['scoring_command'],folder/'score.log');score=read(folder/'score.json')
        assert score['split']=='test' and score['n']==467
        result={**record,'test_score':score,**timings[record['name']],'scoring_wall_seconds':seconds,
          'prediction_plus_scoring_wall_seconds':timings[record['name']]['prediction_wall_seconds']+seconds,
          'official_score_file':str(folder/'score.json'),'score_sha256':sha(folder/'score.json')}
        results.append(result);write(HERE/'results/summary.json',results)
        print(f'{record["name"]}: {score["correct"]}/{score["n"]} = {score["accuracy"]:.6f}%',flush=True)
    assert all(sha(p)==h for p,h in frozen_assets.items())
    changed=[p for p,h in original.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    write(HERE/'results/verification.json',{'passed':not changed,'pre_existing_files_checked':len(original),'changed':changed,
      'official_release_manifest_mismatches':release_mismatch,'frozen_sources_and_weights_unchanged':True,
      'all_test_predictions_frozen_before_first_scoring':True,'test_case_error_analysis_or_new_training':False,
      'primary_remains_V_selected_on_validation':True,'fresh_prediction_and_scoring_processes_per_method':True,
      'timing_note':'Sum of externally measured fresh prediction and fresh scoring process durations; excludes intervening other-method jobs.',
      'reference_machine_time_certification':False})
    assert not changed,changed
    columns=['name','label','validation_correct','validation_accuracy','test_correct','test_n','test_accuracy','fresh_seconds','asset_MiB']
    with (HERE/'results/summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in results:writer.writerow({'name':r['name'],'label':r['label'],'validation_correct':r['validation_correct'],
          'validation_accuracy':r['validation_accuracy'],'test_correct':r['test_score']['correct'],'test_n':r['test_score']['n'],
          'test_accuracy':r['test_score']['accuracy'],'fresh_seconds':r['prediction_plus_scoring_wall_seconds'],
          'asset_MiB':r['asset_bytes_uncompressed']/1024**2})
    lines=['# 冻结方法最终测试记录','',
      '日期：2026-10-28（北京时间）。用户请求对现有较好方案进行最终测试。名单、顺序、代码、权重、参数和主候选V均在测试评分前冻结。六份预测全部生成并记录哈希后，才逐一调用原版独立评分器。未进行测试错题驱动迭代、参数搜索、融合或重新训练。','',
      '| 冻结方案 | 验证集/798 | 测试集/467 | 测试准确率 | 本机预测＋评分秒 | 资产MiB |','|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f'| {r["name"]}：{r["label"]} | {r["validation_correct"]} | {r["test_score"]["correct"]} | {r["test_score"]["accuracy"]:.6f}% | {r["prediction_plus_scoring_wall_seconds"]:.3f} | {r["asset_bytes_uncompressed"]/1024**2:.5f} |')
    primary=next(r for r in results if r['name']=='V')
    lines+=['',f'验证集选定的主候选仍为V：正式测试{primary["test_score"]["correct"]}/467，官方accuracy={primary["test_score"]["accuracy"]}。其他五组只作为预先冻结的最终比较，不根据测试排名改写方法选择理由。','',
      '验证与测试的题目构成不同，验证含单有效答案组的容易题，而课程说明测试包含至少两个不同有效数值答案。验证和测试准确率不能直接作同分布差值解释；本轮不展开测试题目或逐题错误来开发方法。','',
      f'本机：{cpu}，Python{sys.version.split()[0]}；进程串行，冻结运行时Torch4/interop1、BLAS1。时间为完整新预测进程与独立评分进程外部墙钟之和，包括启动、导入、数据及资产加载和全部成员。每套资产按NPZ展开后统计，均小于16MiB；本机时间不等于官方Xeon30秒认证。','',
      f'核验{len(original)}个开始前既有文件，变更0；官方发布清单全部匹配。预测器只读取输入及冻结资产，参考答案由官方score_predictions.py读取。测试输出位于各方案results/<名称>/predictions.json和score.json。','',
      '复现命令完整保存在frozen_before_test.json各方案prediction_command/scoring_command。再次复现时必须使用新的输出路径，保留本次正式记录。主候选使用现有arith_relation_audit_20261018/portable/inference/method.py与portable/pipelines/V，依赖NumPy2.5.3、PyTorch2.7.1+cpu和Python3.12。','',
      '本次只完成本地最终评分与证据保存，未执行网页提交、公开发布或更换预测默认入口。实验运行与报告由Codex AI辅助。']
    (HERE/'results/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Complete; original files and frozen methods unchanged.',flush=True)

if __name__=='__main__':main()
