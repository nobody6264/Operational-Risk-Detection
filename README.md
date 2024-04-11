# Behavioral Logic Heterograph Learning Enhanced with Expert Experience to Identify Operational Risks
## Abstract
Operational risks are a growing concern across various industries, which cause substantial losses annually. Despite the increasing attention towards their identification and prevention, three key characteristics make this task challenging: (a) involvement of multiple interrelated heterogeneous entities, (b) subtle yet damaging, (c) insufficient risk labels due to their low frequency, and (d) dynamically changing risk patterns.
For addressing such four issues, we propose a novel risk identification framework which combines graph pretraining with finetuning. In the pretraining stage, behavioral logic is depicted through heterographs, which are subsequently transformed into low-dimensional vectors via a heterograph learning algorithm. This algorithm is enhanced with expert experience, enabling it to capture subtle changes in operational risk features.
During the finetuning phase, we introduce a risk identification model grounded in psychological behavioral analysis. This model exhibits superior generalization performance, particularly in scenarios with limited risk labels and shifting risk patterns.
Extensive experiments on a series of datasets validate outstanding performance of our framework.
## Framework
In this framework, we first use a large language model (LLM) to transform expert experience into a series of abnormal operational graphs and inject them into the knowledge base. Using such data, we train a risk scoring model, which guides the learning of operational logic heterogeneous graphs that are encoded into low-dimensional vectors. Next, we estimate the distribution of these vectors to identify operational risks. Finally, risk experts further review the model results and update the knowledge base.

<p align="center">
   <img src="https://github.com/nobody6264/Operational-Risk-Detection/blob/main/images/overview-8.jpg" width="600">
</p>

<p align="center">
   <img src="https://github.com/nobody6264/Operational-Risk-Detection/blob/main/images/initial stage.jpg" width="600">
</p>

The heterogeneous graphs, consisting of different types of nodes and edges, are a powerful data structure usually used for abstracting and depicting complex systems in which objects of different types interact with each other in various ways.
In this study, we apply heterographs to model the streaming operational logic.


<p align="center">
   <img src="https://github.com/nobody6264/Operational-Risk-Detection/blob/main/images/operation-heterograph-3.jpeg" width="600">
</p>

Then we introduce a novel behavioral logic heterograph learning enhanced with expert experience. we initially develop a risk scoring model, informed by expert experience, capable of allocating scores within the interval $(0,1)$ corresponding to varying risk levels. Subsequently, during the graph contrastive learning phase, we employ these expert-derived scores to fine-tune the spatial distribution of graph features within a low-dimensional space. This strategy ensures that all normal samples are oriented towards a concentrated and evenly distribution in the space, while directing risk-bearing samples towards distinctly separate regions.

<p align="center">
   <img src="https://github.com/nobody6264/Operational-Risk-Detection/blob/main/images/he-9.jpg" width="600">
</p>

## Dataset
we provide an financial operational risk graph dataset called **XWRisk**.The dataset contains daily operational logsfrom 01/01/2021 to 31/12/2021. Each daily operational log file comprises approximately 6,148,000 activities performed on 14 internal systems by 800 employees affiliated with 67 departments.
The example data and labels *XWRisk/data_and_labels.pkl* are saved on <a href='https://pan.baidu.com/s/1HLjbrDYWYNOG1wWgrd0Wcw?pwd=g6ta'><img height="15" src="./images/baidu_icon.jpeg"/></a>.
