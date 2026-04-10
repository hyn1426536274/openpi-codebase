
## research
pi05三板斧 ：fast + subtask + stop gradient （vlm data）
### pi0 vs pi05
因为 pi05 有subtask等预训练，而pi0没有。
因此在post training时，三板斧是否对pi05是必要的，而对于pi0没有那么必要（提升不大）
（因为业内总是直接用pi05模型但是却还是直接只用flowmatching loss，遵循着pi0的训练）

### 三板斧对pi05效果分别的影响，组合的影响
例如：
（1）：joint training + 只用subtask | 只用fast | 只用vlm data | （subtask + fast） | （subtask+vlm data） | （fast + vlm data）| （all）
（2）：stop gradient  + 只用subtask | 只用fast | 只用vlm data | ……


### 训练数据单任务setting 和 多任务setting
(场景：post pretraining)
多任务到单任务的迁移中, 这三板斧分别对成功率，遵循能力的影响（以往只是说明多任务训练可以提升单任务表现:直接只用flowmatching loss）
实验过程：
先进行一次post pretraining，再在单任务finetune，这里主要对post pretraining消融
例如：
多任务的only subtask（language）or
多任务的only fast ……
对单任务的表现有没有提升,泛化性如何
（多任务可能不止是pick plate和pick bowl，还有close draw等无关任务）


### 执行
（LIBERO等其他benchmark实验,简单点的bench，可以看到效果和指标变化）
（subtask方面，看看有无开源项目或使用 LLM 生成subtask标注）

