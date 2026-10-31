# 42道共同失败题诊断

诊断冻结Pool-P和S2-C五种子集成在验证集上共同失败的42题。报告位于results/REPORT.md；42题人工证据在manual_reviews.json以及results/case_reviews.csv，完整复核卡在results/reviewed_cases.jsonl。

本目录为用户本次诊断请求授权的新输出路径，不修改任何既有实验。未做新训练、推理或评分；训练集仅用于只读格式现象复现与既有OOF选择审计，不拟合新规则。验证错因不用于生成部署查表或人工修改标签。

复现：Python3.12，设置PYTHONDONTWRITEBYTECODE=1，依次用-B执行prepare.py、format_audit.py、finalize.py。manual_reviews.json是明确的人工标注输入，程序不会自动产生因果判断。

prepare.py汇总官方解析、保存的选择与概率，并导出42张完整证据卡。format_audit.py检测有限范围内的答案格式冲突，保留原官方解析器；finalize.py合并人工证据、生成报告并复核开始时11,162个既有文件哈希。

完整阅读范围是42个题干、两代表选中的错误候选、全部数值正确候选；其他未选错误候选未全部逐步检查。主类别不代表独立证明的模型内部机制。全部测试集语义内容均未读取，保护快照包含文件的字节哈希。
