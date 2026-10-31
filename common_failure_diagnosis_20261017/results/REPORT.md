# 42道共同失败题的逐题诊断

**结论：优先补充候选文本中的局部计算和题意约束证据，不建议继续扩大专家集成或仅凭票数做更复杂选择。** 这是对冻结Pool-P与S2-C共同失败样本的诊断，不是新模型成绩，也不能证明固定向量不含正确性信息。

## 阅读范围与口径

全部42题的题干、两代表选中的错误候选以及所有官方数值正确候选均已完整阅读，共95份候选文本；并补看案例1、17、24、32、36、41的官方解答。其他未被选择的错误候选没有全部逐步验算。先用本地索引筛选、去重，再读必要文本；题干加已读候选为65,598字符，四候选全部文本加题干为108,330字符。此处不包含元数据与额外参考阅读，也不是精确token统计。

统计标签从保存的预测及官方解析结果计算；数学错因由本次阅读人工标注。每题给一个主类别以便计数，次要错误另列，不把统计标签等同于模型内部原因。分类没有独立第二位评审，且只代表这42题的条件分布。源码和原始数据只读，未重新训练、推理、评分或更改官方标签。

## 结构与现有方法的覆盖

- 42题中38题只有一个正确候选；40题两种代表选中同一错误数值。
- 22题有错误的唯一最高票答案，另外20题最高票并列；没有正确的唯一最高票答案。
- 16题所有有效答案组均为单例（含解析失败候选时有效组可能少于4）；这些题的组内max/mean/sum相同，不能借重复答案提供新证据。其余26题有重复答案，但多数票会误导其中22题。不能由此断言所有答案组网络无效。
- 11题含解析无效候选，但两个代表选中的错误候选均可被官方解析，因此此前过滤None的规则已不能处理这些失败。
- 6个既有I/Pool/S2 × P/C五种子集成的并集中，只另覆盖5题；再加S2-route共覆盖6题，仍有36题全部失败。S2-route单独能修复其中4题。这只是已知正确性后的覆盖统计，不是可部署选择器成绩。
- 30个已有单种子模型中，25题没有任何一个模型选对。Pool-P五种子全错35题、五种子一致选同一错误数值34题；S2-C分别为35题和32题。置信一致可能来自共同偏差。

## 人工主分类

| 主类别 | 题数 | 案例编号 |
|---|---:|---|
| 数量、实体或累计范围混淆 | 13 | 2, 9, 11, 13, 14, 16, 18, 21, 22, 27, 31, 34, 35 |
| 题意到数量关系的绑定错误 | 8 | 6, 8, 12, 25, 29, 32, 36, 42 |
| 遗漏、改写或添加题目条件 | 8 | 7, 10, 15, 20, 26, 30, 37, 40 |
| 计算中间量但未回答所求量 | 4 | 23, 33, 38, 39 |
| 局部算术等式错误 | 5 | 3, 4, 5, 17, 28 |
| 单位换算错误（可伴随算术错误） | 1 | 24 |
| 输出结构与官方解析口径不匹配（可伴随目标错误） | 2 | 19, 41 |
| 题面通常含义与参考计算存在疑点 | 1 | 1 |

数量范围、关系绑定、条件和所求量四类合计33题。例如当前余额又加一次收入；将总耗时当每层耗时；把“姐姐是Lexie两倍”反写；已经算出未康复人数却把它当剩余人数。它们通常不是再算一次等式就能发现的。

5题主因是明确局部算术错误，另有单位案例24和简单利息案例42伴随算术错误。例：案例4把3+5+7+9+11算36；案例5把12×0.50算60；案例28所有分项都正确，和却由54变63。候选间区别可落在极小的数字或关系片段上，支持尝试更细的文本证据；尚不能证明整段embedding已经丢失这些信息。

## 两个容易误判的发现

**输出解析错位。** 案例41的被选候选正确计算240、250、220，并最终写boxed250，但其最后一个`####`是Job3标题，官方优先从这里读到3，因而该候选判错。候选0既有正确数值也可正确解析。案例19中S2选中的截断候选最后boxed为`32-d`，被解析为32，虽然尾部已试出31、32、33；Pool另选中直接把中间年龄32当最小年龄的候选。两例不能笼统记成数学能力不足，也不能更改解析器或候选文本来消除错误。

**最终正确不等于过程无错。** 案例2、13、17、20、24、27共6题，唯一官方正确候选有显著数量、算术或逻辑问题，最终数值却与参考相同。案例17正确候选用错误的总数32和袋数4得到8，官方有效计算应为40/5；案例20正确候选误读现金为10、算出-5后又直接输出5。另4题存在变量或单位表述失准，单独记录，未混入这6题。因此“发现一个过程错误就否决该候选”会和最终数值评分目标冲突；一致性证据应作软特征，仍由训练集最终答案正确性监督。

**参考题面疑点。** 案例1的`1/4 more than`通常得到17，参考按四分之一再加7得到9；案例36的`at20% of`字面应120，参考按`20% off`得到480。但案例36当前被选720按两种读法都错，疑点不能替其免责。这两题保留固定标签和分数，只在诊断中标注，不据此改写准确率。案例32另有不顺的题面语法，但没有将其计作第三个明确参考冲突。

## 格式问题在训练集中是否复现

增加一次只读的格式普查：只识别最后一个非嵌套boxed中的完整数值字面量，且它在最后一个####之后；若与官方解析结果不同，则记录冲突。另记录以数字开头的代数boxed。这只是有限覆盖的诊断探针，不是替代解析器。

- 训练集19,020个候选中，有22个数值框与官方解析冲突，涉及17题；其中16个框内数值符合参考，但官方解析不符。22个候选按当前官方口径均为错。
- 既有训练OOF选择中，Pool-P选中过这类候选7次，S2-C选中过5次。它们在训练集上确实可见，格式现象不是只在这42道验证题出现。
- 另有11个数字前缀的代数boxed，其中1个按官方解析碰巧正确；不能将所有表达式框硬性视为错误。
- 验证全体候选的对应冲突有7个、涉及6题，其中4个框内数值匹配参考。没有运行任何新过滤规则或计算新方法准确率；这些数量不等于潜在净提升。

## 建议的最小下一步

1. **先补信息，再改选择结构。** 保持S2-C主模型冻结，给每个候选添加小量可复核的文本证据：官方实际取数位置与末尾答案是否一致、非纯数值/未解变量框、明显截断、保守支持的纯数值等式是否成立。用训练OOF学习很小的候选分数校正，目标仍为最终数值正确。和上一组“选择哪位专家”相比，这里新增的是原模型没有显式读取的内容证据。
2. **四组足够检验证据来源。** 原S2-C、仅输出结构证据、仅算术证据、两者合并；先做提取覆盖和误报审计，再固定定义及训练设置。数学表达式只处理明确支持的运算，无法解析时标未知，不能eval候选代码、补生成解答或把未解析当错。不得针对42题手写分支或拟合参数。
3. **主体难点是题目与解答的数量关系。** 上述小实验仅覆盖部分问题。如果继续投入训练，更值得试题干—解答的数字、实体、单位及关系对齐，或定位四候选真正分歧的局部片段，再与既有向量结合；使用训练数据建立词表/映射，不引入外部预训练模型。仅检测数字是否出现不够，案例7和26已复述条件却未使用，案例23已算税却未纳入最终答案。此方向目前是研究假设，不是已证明可学习。
4. **答案组建模暂不优先。** 16题没有可利用的重复答案，22题唯一多数反而错；单纯强化组票数或换聚合层，未提供上述局部计算和约束证据。后续若进入组建模，应明确控制组大小先验，并验证是否能保留少数正确候选。
5. **评估须计算损害。** 全部验证题与355道多答案题都报告净增，另看单正确候选、错误多数、格式冲突等分组，并复核新增证据是否误伤那些最终正确但过程有错的候选。不能只对这42题报修复数，也不能把本次人工错因当训练标签。

## 逐题证据

候选索引为0起始；source_line是原始验证候选JSONL的1起始行号。原始长文本留在evidence_cards.jsonl，复核后的完整卡在results/reviewed_cases.jsonl；精简索引在results/case_reviews.csv。以下公式是人工诊断证据，不是新的候选生成或部署规则。

### 案例1（原数据行33）

- 主类别：题面通常含义与参考计算存在疑点；次级标签：reference_wording_conflict, ratio_semantics。
- 题目：Yun had 20 paperclips initially, but then lost 12. Marion has 1/4 more than what Yun currently has, plus 7. How many paperclips does Marion have?
- 官方目标：9；四候选解析：['9', '17', '17', '17']；Pool-P选3，S2-C选3。
- 证据：现有Yun=8；题面说Marion有1/4 more than Yun，再加7。被选17按通常英文含义为8+8/4+7；官方参考却按8/4+7=9。
- 核对：通常more than读法=17；官方固定口径=9。保留疑点，不更改标签。
- 题面标记：strong_wording_conflict；标签保持原样。

### 案例2（原数据行45）

- 主类别：数量、实体或累计范围混淆；次级标签：pairs_vs_people, entity_count。
- 题目：Three different 6th grade classes are combining for a square dancing unit.  If possible, the teachers would like each male student to partner with a female student for the unit.  The first class has 17 males and 13 females, while the second class has 14 males and 18 females, and the third class has 15 males and 17 females.  When the classes are combined, how many students will be unable to partner with a student of the opposite gender?
- 官方目标：2；四候选解析：['48', '48', '45', '2']；Pool-P选0，S2-C选0。
- 证据：男46女48，共94人；错误候选把46对当46名已配对学生，用94-46=48。
- 核对：94-2*46=2。
- 数值正确候选的过程问题：唯一正确候选3也写总学生48、能配对学生46，再作48-46=2；最终数值对，数量标签和过程解释有错。

### 案例3（原数据行53）

- 主类别：局部算术等式错误；次级标签：fraction_equality。
- 题目：Erik's dog can run 24 miles per hour. It is chasing a rabbit that can run 15 miles per hour. The rabbit has a head start of .6 miles. How many minutes does it take for the dog to catch up to the rabbit?
- 官方目标：4；四候选解析：['8', '4', '8', '8']；Pool-P选2，S2-C选2。
- 证据：错误候选把0.6/9写成2/15小时，继而得到8分钟。
- 核对：0.6/9=1/15小时；乘60得4分钟。

### 案例4（原数据行66）

- 主类别：局部算术等式错误；次级标签：sum_equality。
- 题目：A snail kite is a kind of bird that eats apple snails. On the first day, a snail kite ate 3 snails. Then it eats 2 more snails than it did the day before. How many apple snails in all did a snail kite eat for 5 days?
- 官方目标：35；四候选解析：['31', '36', '35', '36']；Pool-P选3，S2-C选3。
- 证据：每日序列3,5,7,9,11完全正确，却把3+5+7+9+11写成36。
- 核对：3+5+7+9+11=35。

### 案例5（原数据行67）

- 主类别：局部算术等式错误；次级标签：decimal_product。
- 题目：You can buy 4 apples or 1 watermelon for the same price. You bought 36 fruits evenly split between oranges, apples and watermelons, and the price of 1 orange is $0.50. How much does 1 apple cost if your total bill was $66?
- 官方目标：1；四候选解析：[None, '1', '133/100', '1/10']；Pool-P选3，S2-C选3。
- 证据：设苹果价x、12个橙子单价0.50，错误候选写12*0.50=60，形成60+60x=66。
- 核对：12*0.50=6；6+60x=66，x=1。

### 案例6（原数据行98）

- 主类别：题意到数量关系的绑定错误；次级标签：cashflow_sign。
- 题目：A football club has a balance of $100 million. The club then sells 2 of its players at $10 million each, and buys 4 more at $15 million each. How much money is left in the club register in millions of dollars?
- 官方目标：60；四候选解析：['20', '60', '20', '20']；Pool-P选3，S2-C选2。
- 证据：两方法选的不同错误文本都把卖球员所得20和买球员花费60相加为支出80。
- 核对：100+2*10-4*15=60。

### 案例7（原数据行134）

- 主类别：遗漏、改写或添加题目条件；次级标签：unused_condition, second_coat。
- 题目：Jason and Jeremy want to paint their wall white and agreed to split the cost of the paint. A gallon of paint costs $45 and can cover up to 400 square feet. How much will each of them contribute to the cost of the paint if their walls have a total area of 1600 square feet and will need a second coat?
- 官方目标：180；四候选解析：[None, '180', '90', '45']；Pool-P选2，S2-C选2。
- 证据：错误候选复述second coat，但只按1600/400=4加仑算一次涂刷，再平分为90。
- 核对：1600/400*2*45/2=180。

### 案例8（原数据行162）

- 主类别：题意到数量关系的绑定错误；次级标签：state_transition, additional_vs_remaining。
- 题目：Patricia is growing her long very long to donate it to a charity that makes wigs for cancer survivors. Her hair is 14 inches long. She needs to donate 23 inches to make a wig. She wants her hair to be 12 inches long after the donation. How much longer does she have to grow her hair?
- 官方目标：21；四候选解析：['21', '2', '2', '3']；Pool-P选3，S2-C选3。
- 证据：错误候选先算23-14=9的不足量，却用12-9=3来回答需要再长多少。
- 核对：设再长x，14+x-23=12，x=21。

### 案例9（原数据行163）

- 主类别：数量、实体或累计范围混淆；次级标签：total_vs_per_unit, time_scope。
- 题目：Michael is stuck in an elevator that is slowly moving to the bottom floor. The elevator needs to move down 20 floors to reach the bottom. It takes 15 minutes for the elevator to travel down the first half of the floors. The elevator then takes 5 minutes per floor to travel down the next 5 floors. It then takes 16 minutes per floor for the elevator to finish traveling down the final 5 floors. In hours, how long did it take for the elevator to reach the bottom?
- 官方目标：2；四候选解析：['17/4', '2', '17/4', '2']；Pool-P选2，S2-C选2。
- 证据：题面15分钟是前10层的合计，错误候选改成10*15=150分钟。
- 核对：(15+5*5+16*5)/60=2小时。

### 案例10（原数据行171）

- 主类别：遗漏、改写或添加题目条件；次级标签：unused_entity。
- 题目：During the Easter egg hunt, Kevin found 5 eggs, Bonnie found 13 eggs, George found 9 and Cheryl found 56.  How many more eggs did Cheryl find than the other three children found?
- 官方目标：29；四候选解析：['34', '29', '42', '42']；Pool-P选2，S2-C选2。
- 证据：错误候选在56-(5+9)中漏掉Bonnie的13个蛋，尽管开头已复述。
- 核对：56-(5+13+9)=29。

### 案例11（原数据行172）

- 主类别：数量、实体或累计范围混淆；次级标签：population_state, survivors_vs_deaths。
- 题目：A frog pond has five frogs living in it. Some frogs laid eggs, and now there are three times as many tadpoles growing into more frogs as there are frogs. Two-thirds of the tadpoles will survive to maturity as frogs. The pond can only sustain eight frogs. How many frogs will have to find a new pond when the tadpoles grow up?
- 官方目标：7；四候选解析：['5', '1', '3', '7']；Pool-P选2，S2-C选2。
- 证据：先算15只蝌蚪中10只成活，随后将15-10解释成会变蝌蚪的青蛙，用8-5=3回答需迁出数。
- 核对：现有5+新增10-容量8=7。

### 案例12（原数据行176）

- 主类别：题意到数量关系的绑定错误；次级标签：subject_object_reversal, ratio_direction。
- 题目：Lexie is 6 years older than her brother and her sister is twice her age. If Lexie is 8, what is the age difference between her brother and her sister?
- 官方目标：14；四候选解析：['2', '2', '2', '14']；Pool-P选1，S2-C选1。
- 证据：题面姐姐是Lexie年龄的两倍，错误候选改成Lexie是姐姐的两倍，将姐姐算成4而不是16。
- 核对：弟弟8-6=2；姐姐2*8=16；差14。

### 案例13（原数据行207）

- 主类别：数量、实体或累计范围混淆；次级标签：double_counting, unused_savings。
- 题目：Isabelle gets a job so she can save enough money to go to a concert with her two brothers. Her ticket costs $20. Her brothers both get the children’s discount, and each of their tickets cost $10. Her brothers have saved $5 between the two of them and Isabelle has saved $5. If her job pays $3 per week, how many weeks must she work to afford the tickets?
- 官方目标：10；四候选解析：['10', '20', '11', '6']；Pool-P选1，S2-C选1。
- 证据：兄弟票共20、本人票20，错误候选却写20+20+20=60，且后续未减已存10。
- 核对：(20+2*10-5-5)/3=10周。
- 数值正确候选的过程问题：唯一正确候选0把票价当存款，称兄弟票20加存款10得到总存款30，再30/3=10；最终数值符合参考，但解释没有正确建立40-10的差额。

### 案例14（原数据行210）

- 主类别：数量、实体或累计范围混淆；次级标签：calendar_count。
- 题目：Tom hasn't been sleeping well lately. He figures he has been getting about 5 hours of sleep each weeknight and 6 hours each night on the weekend. If Tom would ideally like to get 8 hours of sleep each night on both weeknights and weekends, how many hours of sleep is Tom behind on from the last week?
- 官方目标：19；四候选解析：['19', '10', '5', '10']；Pool-P选1，S2-C选1。
- 证据：错误候选把weeknights算成2晚、整周算4晚，得到32-22=10。
- 核对：(8-5)*5+(8-6)*2=19。

### 案例15（原数据行211）

- 主类别：遗漏、改写或添加题目条件；次级标签：unused_condition, phase_identification。
- 题目：Billy and Margaret are competing with each other to see who can swim 10 laps the fastest. Billy swims his first 5 laps in 2 minutes, swims the next 3 laps in 4 minutes, swims the next lap in 1 minute, then swims his final lap. Margaret finishes swimming all of her laps in 10 minutes. Billy wins the competition by finishing his laps 30 seconds before Margaret does. In seconds, how long did it take Billy to swim his final lap?
- 官方目标：150；四候选解析：['420', '150', '540', '180']；Pool-P选3，S2-C选3。
- 证据：错误候选把第9圈1分钟称最后一圈，最后用600-420=180，未扣领先的30秒。
- 核对：10*60-30-(2+4+1)*60=150。

### 案例16（原数据行276）

- 主类别：数量、实体或累计范围混淆；次级标签：double_counting, stock_vs_increment。
- 题目：Texas Integrated School has 15 classes and has 20 students per class.  They added five more classes, how many students will they have now?
- 官方目标：400；四候选解析：['400', '700', '700', '400']；Pool-P选1，S2-C选1。
- 证据：原15班300人，加5班后总20班400人；错误候选又把300+400算成700。
- 核对：(15+5)*20=400。
- 数值正确候选的过程问题：正确候选0有干净计算；另一个正确候选3把新增后的20班称为20 new classes，但最终400正确。

### 案例17（原数据行303）

- 主类别：局部算术等式错误；次级标签：sum_equality。
- 题目：Robby doesn't like the color brown and will not eat the brown M&M's.  On his first bag, he sets aside  9 brown M&M's.  On the second bag, he sets aside 12.  On his third & fourth bag, there are 8 brown M&M's that he doesn't eat. On his fifth bag, there are only 3 brown M&M's that he doesn't eat.  What's the average number of brown M&M's in a bag?
- 官方目标：8；四候选解析：[None, '38/5', '32/5', '8']；Pool-P选1，S2-C选2。
- 证据：Pool候选将9+12+8+8+3写38；S2候选把同一式写32。官方参考合计40。
- 核对：(9+12+8+8+3)/5=40/5=8。
- 数值正确候选的过程问题：唯一正确候选3将五袋变四袋、用32/4=8，还说Robby吃了不吃的棕色糖；最后答案偶然一致。

### 案例18（原数据行329）

- 主类别：数量、实体或累计范围混淆；次级标签：entity_count, self_inclusion。
- 题目：Manny had 3 birthday cookie pies to share with his 24 classmates and his teacher, Mr. Keith.   If each of the cookie pies were cut into 10 slices and Manny, his classmates, and Mr. Keith all had 1 piece, how many slices are left?
- 官方目标：4；四候选解析：['5', '5', '240', '4']；Pool-P选0，S2-C选0。
- 证据：错误候选写Manny+24 classmates+teacher=25人，实际为26人。
- 核对：3*10-(1+24+1)=4。

### 案例19（原数据行332）

- 主类别：输出结构与官方解析口径不匹配（可伴随目标错误）；次级标签：symbolic_box_prefix, truncated_trace, middle_vs_youngest。
- 题目：The ages of three brothers are consecutive integers with the sum of 96.  How old is the youngest brother?
- 官方目标：31；四候选解析：['31', '32', None, '32']；Pool-P选3，S2-C选1。
- 证据：S2选候选1，最后一个boxed是32-d，文本截断在31+32+33=；官方解析读到32。Pool选候选3，把中间年龄32当最小年龄。
- 核对：连续整数为31,32,33；最小31。候选0具有可正确解析的最终31。

### 案例20（原数据行339）

- 主类别：遗漏、改写或添加题目条件；次级标签：multiplicity_dropped, cash_amount。
- 题目：Marcus is having a water balloon party. He has 100 balloons. Each balloon holds 3 ounces of water. He can buy 50 ounces of water for $2.5 a bottle. If he walks into the store with 2 $10 bills, how much change will he have after he buys all the water he needs?
- 官方目标：5；四候选解析：['-5', '-5990', '5', '75']；Pool-P选0，S2-C选0。
- 证据：题面2张10元，错误候选只算10元；需水6瓶花15元，从而给出-5。
- 核对：2*10-(100*3/50)*2.5=5。
- 数值正确候选的过程问题：唯一正确候选2也只按10元，算出-5并称买不起，再无依据给出boxed5；数值正确并不表示推理正确。

### 案例21（原数据行363）

- 主类别：数量、实体或累计范围混淆；次级标签：double_counting, current_vs_initial_balance。
- 题目：For his birthday, Geoffrey received clothes and money. His grandmother gave him €20, his aunt €25 and his uncle €30. With what he already had, he now has €125 in his wallet. He goes to a video game store and gets 3 games that cost 35 euros each. How much does he have left after this purchase?
- 官方目标：20；四候选解析：['20', '20', '95', '95']；Pool-P选2，S2-C选2。
- 证据：125是加完礼金后的余额，错误候选把它当原有钱，再加75礼金得200。
- 核对：125-3*35=20。

### 案例22（原数据行372）

- 主类别：数量、实体或累计范围混淆；次级标签：monthly_vs_annual_base。
- 题目：Gillian’s phone bill is usually $50 a month. If the monthly bill increases by 10%, what will be her phone bill for the entire next year?
- 官方目标：660；四候选解析：['660', '110', None, '14763/50']；Pool-P选1，S2-C选1。
- 证据：错误候选将月基数50与全年涨额5*12=60相加得110，没有把基数乘12。
- 核对：50*1.1*12=660。

### 案例23（原数据行398）

- 主类别：计算中间量但未回答所求量；次级标签：tax_not_propagated, final_answer_binding。
- 题目：Mark is buying asphalt to pave a new section of road. The road will be 2000 feet long and 20 feet wide. Each truckload of asphalt will cover 800 square feet of road. If each truckload costs $75, and there's a 20% sales tax, how much will Mark need to pay for asphalt?
- 官方目标：4500；四候选解析：[None, '3750', '288000', '4500']；Pool-P选1，S2-C选1。
- 证据：错误候选已正确算税前3750和税750，却将3750直接作为含税最终支付额。
- 核对：3750+750=4500。

### 案例24（原数据行414）

- 主类别：单位换算错误（可伴随算术错误）；次级标签：ounces_pounds, sum_equality。
- 题目：Melanie does her weekly shopping at the farmer's market.  She starts with an 8-ounce wheel of brie cheese.  Next is a 1 pound loaf of bread.  She grabs a pound of tomatoes and 2 pounds of zucchini.  After that, she grabs 1 1/2 pounds of chicken breasts and treats herself to 8 ounces of fresh raspberries and 8 ounces of fresh blueberries.  How many pounds of food does she buy?
- 官方目标：7；四候选解析：['7', '15/2', '13/2', '13/2']；Pool-P选3，S2-C选3。
- 证据：错误候选写8盎司=1磅，且列出的1+1+1+2+1.5+1+1又声称合计6.5，单位和求和均错。
- 核对：3*8/16+1+1+2+1.5=7磅。
- 数值正确候选的过程问题：唯一正确候选0也写8盎司=1磅、遗漏一项并写1+1+1+2+1.5+1=7；此式实际7.5。

### 案例25（原数据行445）

- 主类别：题意到数量关系的绑定错误；次级标签：route_segment_binding。
- 题目：Andy walks 50 meters from his house to school. After school, he comes back to the house and goes to the market. If he walks 140 meters in total, how many meters is the distance between the house and the market?
- 官方目标：40；四候选解析：[None, '45', '40', None]；Pool-P选1，S2-C选1。
- 证据：返家段已知仍为50米，错误候选却把它设成市场距离x，列50+x+x=140。
- 核对：50+50+x=140，x=40。

### 案例26（原数据行470）

- 主类别：遗漏、改写或添加题目条件；次级标签：unused_multiplier。
- 题目：Lara is baking cookies using four baking trays. She places five rows of cookies on a baking tray where there are six cookies in one row. How many cookies is she baking?
- 官方目标：120；四候选解析：['120', '24', '30', '30']；Pool-P选2，S2-C选2。
- 证据：错误候选复述four baking trays，最终只算每盘5*6=30，漏乘4盘。
- 核对：4*5*6=120。

### 案例27（原数据行498）

- 主类别：数量、实体或累计范围混淆；次级标签：entity_count, host_inclusion。
- 题目：Tom invites his parents and 3 siblings to his house.  They each eat 3 times a day.  How many plates do Tom and his guests use while they are there for the 4 days if each person uses 2 plates per meal?
- 官方目标：144；四候选解析：['144', '120', '96', '120']；Pool-P选2，S2-C选1。
- 证据：S2候选只算5位客人未含Tom；Pool候选把parents+3 siblings算4人且未含Tom。
- 核对：(Tom1+parents2+siblings3)*3*2*4=144。
- 数值正确候选的过程问题：唯一正确候选0写3 parents+3 siblings=6人，捏造3位父母且未列Tom，数量6碰巧与真实总人数相同。

### 案例28（原数据行523）

- 主类别：局部算术等式错误；次级标签：sum_equality。
- 题目：After trick-or-treating, Brent dumped his candy on the table.  He had received 5 Kit-Kat bars, three times that amount in Hershey kisses, 8 boxes of Nerds,  and 11 lollipops.  He also had 10 Baby Ruths and half that amount in Reese Peanut butter cups.  After giving his little sister 5 lollipops, how many pieces of candy did Brent have left?
- 官方目标：49；四候选解析：['53', '49', '58', '58']；Pool-P选2，S2-C选2。
- 证据：所有糖果分项5,15,8,11,10,5正确，错误候选把和写63再减5得58。
- 核对：5+15+8+11+10+5=54；54-5=49。

### 案例29（原数据行551）

- 主类别：题意到数量关系的绑定错误；次级标签：ratio_direction, entity_binding。
- 题目：It takes a butterfly egg 120 days to become a butterfly. If each butterfly spends 3 times as much time as a larva as in a cocoon, how long does each butterfly spend in a cocoon?
- 官方目标：30；四候选解析：['180', '90', None, '30']；Pool-P选1，S2-C选1。
- 证据：设x为茧期、y为幼虫期后，错误候选将幼虫期3倍茧期写反为x=3y。
- 核对：幼虫L=3C；L+C=120，C=30。

### 案例30（原数据行568）

- 主类别：遗漏、改写或添加题目条件；次级标签：invented_rate。
- 题目：Hayes does 3 loads of laundry a week using a detergent pod for each load.  His detergent pods come 39 to a pack.  How many packs of detergent pods will he need to do a full year of laundry?
- 官方目标：4；四候选解析：['4', '12', '13', '17']；Pool-P选1，S2-C选1。
- 证据：题面每次洗衣1颗，错误候选改为每次3颗，把156次再乘3得468颗。
- 核对：3*52*1/39=4包。

### 案例31（原数据行587）

- 主类别：数量、实体或累计范围混淆；次级标签：subset_vs_total, missing_present_girls。
- 题目：On a particular day, Rose's teacher read the register and realized there were twice as many girls as boys present on that day. The class has 250 students, and all the 140 girls were present. If all the absent students were boys, how many boys were absent that day?
- 官方目标：40；四候选解析：['180', '180', '180', '40']；Pool-P选2，S2-C选1。
- 证据：已算到场男生70后，错误候选用250-70=180，把到场女生140漏掉。
- 核对：250-140-70=40。
- 数值正确候选的过程问题：唯一正确候选3的开头把140写作全部到场人数，与后续正确250-140-70不一致；后续公式确实正确。

### 案例32（原数据行614）

- 主类别：题意到数量关系的绑定错误；次级标签：subject_object_reversal, awkward_question_wording。
- 题目：Issac has to buy pens and pencils for the upcoming school year. Issac buys 108 writing utensils total. He buys 12 more than 5 times the number of pencils than pens. How many pens did Issac buy?
- 官方目标：16；四候选解析：['16', '92', '92', '16']；Pool-P选2，S2-C选2。
- 证据：按参考含义铅笔=5*钢笔+12，错误候选反写钢笔=5*铅笔+12，得到钢笔92。题面语法不顺，参考说明意图。
- 核对：p+(5p+12)=108，p=16。
- 数值正确候选的过程问题：两个正确候选都把实际代表铅笔数的w称为总writing utensils，但后续按铅笔数列p+w=108。
- 题面标记：awkward_wording_without_separate_reference_mismatch_claim；标签保持原样。

### 案例33（原数据行618）

- 主类别：计算中间量但未回答所求量；次级标签：phase_vs_total, unused_first_phase。
- 题目：James burned his hand in a freak accident.  It takes him 4 weeks to heal enough to get a skin graft.  After that it takes him 50% longer than that to heal from the skin graft.  How long did he take to recover from everything?
- 官方目标：10；四候选解析：['6', '6', '6', '10']；Pool-P选0，S2-C选2。
- 证据：错误候选算出移植后恢复6周，就把它当整个过程总时间，漏掉之前4周。
- 核对：4+4*1.5=10周。

### 案例34（原数据行634）

- 主类别：数量、实体或累计范围混淆；次级标签：whole_bill_vs_own_meal。
- 题目：James goes out to eat.  He orders a steak and egg meal for $16.  He is with his friend, who orders chicken fried steak for $14.  His friend pays for half the bill and James pays the tip along with his half of the bill. They tip 20%.  How much did James pay?
- 官方目标：21；四候选解析：['14', '40', '21', '27/2']；Pool-P选0，S2-C选0。
- 证据：已算整单30和小费6，错误候选却用自己餐费16的一半8代替整单的一半15。
- 核对：(16+14)/2+0.2*(16+14)=21。

### 案例35（原数据行656）

- 主类别：数量、实体或累计范围混淆；次级标签：per_item_vs_subtotal, missing_multiplicity。
- 题目：John decides to start collecting art.  He pays the same price for his first 3 pieces of art and the total price came to $45,000.  The next piece of art was 50% more expensive than those.  How much did all the art cost?
- 官方目标：67500；四候选解析：['67500', '45000', '37500', '45000']；Pool-P选2，S2-C选2。
- 证据：已经算出旧画每幅15000、新画22500，却只加一幅旧画与新画，未加原三幅45000。
- 核对：45000+(45000/3)*1.5=67500。

### 案例36（原数据行662）

- 主类别：题意到数量关系的绑定错误；次级标签：percent_of_vs_off, reference_wording_conflict。
- 题目：After negotiating for the price of a phone, Karsyn bought a phone at 20% of the initial price. If the phone's initial price was $600, calculate how much she paid after negotiating.
- 官方目标：480；四候选解析：['720', '720', '120', '480']；Pool-P选0，S2-C选1。
- 证据：题面at20% of initial通常为120，官方参考按20% off得480；被选两条却将600+120=720，按两种读法都不对。
- 核对：题面字面600*0.2=120；官方口径600*(1-0.2)=480；当前720仍是错误。
- 题面标记：strong_wording_conflict；标签保持原样。

### 案例37（原数据行673）

- 主类别：遗漏、改写或添加题目条件；次级标签：changed_input_rate。
- 题目：Austin receives $5 for every hour he helps his father build a deck in their backyard. He works for 2 hours on Mondays, an hour on Wednesdays, and 3 hours on Fridays. If Austin wants to buy a bicycle that costs $180, how many weeks does he have to work?
- 官方目标：6；四候选解析：[None, None, '6', '30']；Pool-P选3，S2-C选3。
- 证据：题面时薪5，错误候选改成1，周收入算6而非30，继而180/6=30周。
- 核对：180/((2+1+3)*5)=6周。
- 数值正确候选的过程问题：正确候选2把分母30美元/周称为hourly rate；公式与最后6周正确。

### 案例38（原数据行676）

- 主类别：计算中间量但未回答所求量；次级标签：deaths_vs_survivors。
- 题目：A group of 30 tourists sets out on a walking tour of the rainforest. Two tourists are eaten by anacondas, then half the remaining tourists try to pet poisonous dart frogs and get poisoned. If only 1/7 of the poisoned tourists recovered, how many tourists are left at the end of the tour?
- 官方目标：16；四候选解析：['14', '12', '16', '12']；Pool-P选1，S2-C选1。
- 证据：错误候选算14中毒、2康复、14-2=12后停止；12是未康复人数，不是最后留下人数。
- 核对：30-2-(14-2)=16。

### 案例39（原数据行680）

- 主类别：计算中间量但未回答所求量；次级标签：target_total_vs_additional。
- 题目：Billy and Tiffany are having a contest to see how can run the most miles in a week. On Sunday, Monday, and Tuesday, Billy runs 1 mile each day and Tiffany runs 2 miles each day. On Wednesday, Thursday, and Friday, Billy runs 1 mile each day and Tiffany runs a 1/3 of a mile each day. On Saturday Tiffany assumes she's going to win and takes the day off. How many miles does Billy have to run on Saturday to tie Tiffany?
- 官方目标：1；四候选解析：[None, '1', None, '7']；Pool-P选3，S2-C选3。
- 证据：错误候选正确算Billy已有6、Tiffany7，但最终让Billy周六再跑7而不是差额1。
- 核对：7-6=1。

### 案例40（原数据行729）

- 主类别：遗漏、改写或添加题目条件；次级标签：unused_comparator_raise。
- 题目：Hansel makes $30,000 a year and just received a 10% raise.  Gretel makes the same amount as Hansel but received a 15% raise.  How much more money will Gretel make compared to Hansel?
- 官方目标：1500；四候选解析：['4500', '4500', '1500', '4500']；Pool-P选1，S2-C选1。
- 证据：错误候选把Gretel涨后34500与Hansel原薪30000比较，未应用Hansel的10%涨薪。
- 核对：30000*1.15-30000*1.10=1500。

### 案例41（原数据行734）

- 主类别：输出结构与官方解析口径不匹配（可伴随目标错误）；次级标签：markdown_heading_as_answer, visible_final_vs_official_parse。
- 题目：A plumber is trying to decide which of three different jobs he should take to make the most money.  The first job involves fixing three toilets and three sinks at an apartment complex.  The second involves two toilets and five sinks at a different apartment complex.  The third involves fixing one toilet, two showers and three sinks at a third apartment complex.  If the plumber charges $30 to fix a sink, $40 to fix a shower and $50 to fix a toilet, what is the most money the plumber can earn if he can only choose one of these three jobs?
- 官方目标：250；四候选解析：['250', '3', None, '360']；Pool-P选1，S2-C选1。
- 证据：被选候选1正确算三个工作240、250、220并最终boxed250；但最后一个####是Job3标题，官方先从其后读到3，因而该候选标签为错。
- 核对：需选择官方解析也为250的候选0；不能修改候选1或解析器来改变评分。

### 案例42（原数据行740）

- 主类别：题意到数量关系的绑定错误；次级标签：simple_vs_compound_interest, sum_equality。
- 题目：A person puts $5600 in a bank for two years. Each year he earns interest of 7% of the original amount deposited. How much will this person have in the bank after two years?
- 官方目标：6384；四候选解析：['160286/25', '6384', '160286/25', '6420']；Pool-P选3，S2-C选3。
- 证据：题面每年按原本金计息；错误候选先把5600+392写6000，又用6000而非5600算次年利息。
- 核对：5600+2*(5600*0.07)=6384。

## 复现与保护

Python3.12下设置PYTHONDONTWRITEBYTECODE=1，所有命令带-B。依次运行prepare.py、format_audit.py、finalize.py；manual_reviews.json是本次人工阅读的固定标注输入，重新运行脚本不会自动重做人工判断。没有API调用、外部数据或新模型推理；测试集语义内容未读取。保护扫描仅按字节核对开始时已有文件。最终保护结果见verification.json，未通过时保留差异且不更新保护快照。
