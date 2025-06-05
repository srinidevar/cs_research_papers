# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-06-04 17:04:47.492884 PST.

### Artificial Intelligence

### 1. [VS-Bench: Evaluating VLMs for Strategic Reasoning and Decision-Making in Multi-Agent Environments](http://arxiv.org/pdf/2506.02387v1)

Authors: Zelai Xu, Zhexuan Xu, Xiangmin Yi, Huining Yuan, Xinlei Chen, Yi Wu, Chao Yu, Yu Wang

Recent advancements in Vision Language Models (VLMs) have expanded their
capabilities to interactive agent tasks, yet existing benchmarks remain limited
to single-agent or text-only environments. In contrast, real-world scenarios
often involve multiple agents interacting within rich visual and linguistic
contexts, posing challenges with both multimodal observations and strategic
interactions. To bridge this gap, we introduce Visual Strategic Bench
(VS-Bench), a multimodal benchmark that evaluates VLMs for strategic reasoning
and decision-making in multi-agent environments. VS-Bench comprises eight
vision-grounded environments spanning cooperative, competitive, and
mixed-motive interactions, designed to assess agents' ability to predict
others' future moves and optimize for long-term objectives. We consider two
complementary evaluation dimensions, including offline evaluation of strategic
reasoning by next-action prediction accuracy and online evaluation of
decision-making by normalized episode return. Extensive experiments of fourteen
leading VLMs reveal a significant gap between current models and optimal
performance, with the best models attaining 47.8% prediction accuracy and 24.3%
normalized return. We further conduct in-depth analyses on multimodal
observations, test-time scaling, social behaviors, and failure cases of VLM
agents. By standardizing the evaluation and highlighting the limitations of
existing models, we envision VS-Bench as a foundation for future research on
strategic multimodal agents. Code and data are available at
https://vs-bench.github.io.

### 2. [OThink-R1: Intrinsic Fast/Slow Thinking Mode Switching for Over-Reasoning Mitigation](http://arxiv.org/pdf/2506.02397v1)

Authors: Shengjia Zhang, Junjie Wu, Jiawei Chen, Changwang Zhang, Xingyu Lou, Wangchunshu Zhou, Sheng Zhou, Can Wang, Jun Wang

Recent advanced large reasoning models (LRMs) leverage extended
chain-of-thought (CoT) reasoning to solve complex tasks, achieving
state-of-the-art performance. Despite their success, we identify a critical
issue: a substantial portion of simple tasks solved by LRMs can also be
addressed by non-reasoning LLMs using significantly fewer tokens, indicating
the complex reasoning may not always be necessary. To address this, we
systematically analyze the reasoning trajectories of LRMs and present a method
utilizing identified paradigms and LLM-Judge to classify these trajectories as
either Redundant Reasoning or Essential Reasoning. And we introduce OThink-R1,
a method that prunes redundant reasoning steps while preserving logical
validity. OThink-R1 dynamically employs the non-thinking mode (fast-thinking)
for straightforward problems while engaging in deliberate thinking
(slow-thinking) for complex problems. Experiments across mathematical and
question-answering tasks demonstrate that OThink-R1 reduces reasoning
redundancy by almost 23\% on average without compromising accuracy, offering
practical guidelines for efficient reasoning models. The code is available at
https://github.com/AgenticIR-Lab/OThink-R1.

### 3. [A Smart Multimodal Healthcare Copilot with Powerful LLM Reasoning](http://arxiv.org/pdf/2506.02470v1)

Authors: Xuejiao Zhao, Siyan Liu, Su-Yin Yang, Chunyan Miao

Misdiagnosis causes significant harm to healthcare systems worldwide, leading
to increased costs and patient risks. MedRAG is a smart multimodal healthcare
copilot equipped with powerful large language model (LLM) reasoning, designed
to enhance medical decision-making. It supports multiple input modalities,
including non-intrusive voice monitoring, general medical queries, and
electronic health records. MedRAG provides recommendations on diagnosis,
treatment, medication, and follow-up questioning. Leveraging
retrieval-augmented generation enhanced by knowledge graph-elicited reasoning,
MedRAG retrieves and integrates critical diagnostic insights, reducing the risk
of misdiagnosis. It has been evaluated on both public and private datasets,
outperforming existing models and offering more specific and accurate
healthcare assistance. A demonstration video of MedRAG is available at:
https://www.youtube.com/watch?v=PNIBDMYRfDM. The source code is available at:
https://github.com/SNOWTEAM2023/MedRAG.

### 4. [Think Twice, Act Once: A Co-Evolution Framework of LLM and RL for Large-Scale Decision Making](http://arxiv.org/pdf/2506.02522v1)

Authors: Xu Wan, Wenyue Xu, Chao Yang, Mingyang Sun

Recent advancements in Large Language Models (LLMs) and Reinforcement
Learning (RL) have shown significant promise in decision-making tasks.
Nevertheless, for large-scale industrial decision problems, both approaches
face distinct challenges: LLMs lack real-time long-sequence decision-making
capabilities, while RL struggles with sample efficiency in vast action spaces.
To bridge this gap, we propose Agents Co-Evolution (ACE), a synergistic
framework between LLMs and RL agents for large-scale decision-making scenarios.
ACE introduces a dual-role trajectory refinement mechanism where LLMs act as
both Policy Actor and Value Critic during RL's training: the Actor refines
suboptimal actions via multi-step reasoning and environment validation, while
the Critic performs temporal credit assignment through trajectory-level reward
shaping. Concurrently, RL agent enhances LLMs' task-specific decision-making
with high-quality fine-tuning datasets generated via prioritized experience
replay. Through extensive experiments across multiple power grid operation
challenges with action spaces exceeding 60K discrete actions, ACE demonstrates
superior performance over existing RL methods and LLM-based methods.

### 5. [Towards Generating Controllable and Solvable Geometry Problem by Leveraging Symbolic Deduction Engine](http://arxiv.org/pdf/2506.02565v1)

Authors: Zhuoxuan Jiang, Tianyang Zhang, Peiyan Peng, Jing Chen, Yinong Xun, Haotian Zhang, Lichi Li, Yong Li, Shaohua Zhang

Generating high-quality geometry problems is both an important and
challenging task in education. Compared to math word problems, geometry
problems further emphasize multi-modal formats and the translation between
informal and formal languages. In this paper, we introduce a novel task for
geometry problem generation and propose a new pipeline method: the Symbolic
Deduction Engine-based Geometry Problem Generation framework (SDE-GPG). The
framework leverages a symbolic deduction engine and contains four main steps:
(1) searching a predefined mapping table from knowledge points to extended
definitions, (2) sampling extended definitions and performing symbolic
deduction, (3) filtering out unqualified problems, and (4) generating textual
problems and diagrams. Specifically, our method supports to avoid inherent
biases in translating natural language into formal language by designing the
mapping table, and guarantees to control the generated problems in terms of
knowledge points and difficulties by an elaborate checking function. With
obtained formal problems, they are translated to natural language and the
accompanying diagrams are automatically drew by rule-based methods. We conduct
experiments using real-world combinations of knowledge points from two public
datasets. The results demonstrate that the SDE-GPG can effectively generate
readable, solvable and controllable geometry problems.

### 6. [MLaGA: Multimodal Large Language and Graph Assistant](http://arxiv.org/pdf/2506.02568v1)

Authors: Dongzhe Fan, Yi Fang, Jiajin Liu, Djellel Difallah, Qiaoyu Tan

Large Language Models (LLMs) have demonstrated substantial efficacy in
advancing graph-structured data analysis. Prevailing LLM-based graph methods
excel in adapting LLMs to text-rich graphs, wherein node attributes are text
descriptions. However, their applications to multimodal graphs--where nodes are
associated with diverse attribute types, such as texts and images--remain
underexplored, despite their ubiquity in real-world scenarios. To bridge the
gap, we introduce the Multimodal Large Language and Graph Assistant (MLaGA), an
innovative model that adeptly extends LLM capabilities to facilitate reasoning
over complex graph structures and multimodal attributes. We first design a
structure-aware multimodal encoder to align textual and visual attributes
within a unified space through a joint graph pre-training objective.
Subsequently, we implement a multimodal instruction-tuning approach to
seamlessly integrate multimodal features and graph structures into the LLM
through lightweight projectors. Extensive experiments across multiple datasets
demonstrate the effectiveness of MLaGA compared to leading baseline methods,
achieving superior performance in diverse graph learning tasks under both
supervised and transfer learning scenarios.

### 7. [ADFormer: Aggregation Differential Transformer for Passenger Demand Forecasting](http://arxiv.org/pdf/2506.02576v1)

Authors: Haichen Wang, Liu Yang, Xinyuan Zhang, Haomin Yu, Ming Li, Jilin Hu

Passenger demand forecasting helps optimize vehicle scheduling, thereby
improving urban efficiency. Recently, attention-based methods have been used to
adequately capture the dynamic nature of spatio-temporal data. However,
existing methods that rely on heuristic masking strategies cannot fully adapt
to the complex spatio-temporal correlations, hindering the model from focusing
on the right context. These works also overlook the high-level correlations
that exist in the real world. Effectively integrating these high-level
correlations with the original correlations is crucial. To fill this gap, we
propose the Aggregation Differential Transformer (ADFormer), which offers new
insights to demand forecasting promotion. Specifically, we utilize Differential
Attention to capture the original spatial correlations and achieve attention
denoising. Meanwhile, we design distinct aggregation strategies based on the
nature of space and time. Then, the original correlations are unified with the
high-level correlations, enabling the model to capture holistic spatio-temporal
relations. Experiments conducted on taxi and bike datasets confirm the
effectiveness and efficiency of our model, demonstrating its practical value.
The code is available at https://github.com/decisionintelligence/ADFormer.

### 8. [V2X-UniPool: Unifying Multimodal Perception and Knowledge Reasoning for Autonomous Driving](http://arxiv.org/pdf/2506.02580v1)

Authors: Xuewen Luo, Fengze Yang, Fan Ding, Xiangbo Gao, Shuo Xing, Yang Zhou, Zhengzhong Tu, Chenxi Liu

Knowledge-driven autonomous driving systems(ADs) offer powerful reasoning
capabilities, but face two critical challenges: limited perception due to the
short-sightedness of single-vehicle sensors, and hallucination arising from the
lack of real-time environmental grounding. To address these issues, this paper
introduces V2X-UniPool, a unified framework that integrates multimodal
Vehicle-to-Everything (V2X) data into a time-indexed and language-based
knowledge pool. By leveraging a dual-query Retrieval-Augmented Generation (RAG)
mechanism, which enables retrieval of both static and dynamic knowledge, our
system enables ADs to perform accurate, temporally consistent reasoning over
both static environment and dynamic traffic context. Experiments on a
real-world cooperative driving dataset demonstrate that V2X-UniPool
significantly enhances motion planning accuracy and reasoning capability.
Remarkably, it enables even zero-shot vehicle-side models to achieve
state-of-the-art performance by leveraging V2X-UniPool, while simultaneously
reducing transmission cost by over 99.9\% compared to prior V2X methods.

### 9. [EALG: Evolutionary Adversarial Generation of Language Model-Guided Generators for Combinatorial Optimization](http://arxiv.org/pdf/2506.02594v1)

Authors: Ruibo Duan, Yuxin Liu, Xinyao Dong, Chenglin Fan

Generating challenging instances is crucial for the evaluation and
advancement of combinatorial optimization solvers. In this work, we introduce
EALG (Evolutionary Adversarial Generation of Language Model-Guided Generators),
a novel framework that automates the co-evolution of optimization problem
instances and their corresponding heuristic solvers using large language models
(LLMs). EALG leverages a mutation-based adversarial approach that dynamically
evolves instance generation procedures to create increasingly difficult
problems, while simultaneously synthesizing adaptive heuristic algorithms
through interactions with LLMs guided by algorithmic structure. Unlike existing
approaches that focus solely on static benchmark creation or manual solver
design, EALG provides a seamless pipeline from instance generation to solver
synthesis. Experimental results demonstrate that EALG generates significantly
harder instances than current benchmarks, and its synthesized solvers
generalize effectively across a broad spectrum of combinatorial tasks. This
work explores a new paradigm for combinatorial optimization that integrates
instance generation with solver design, resulting in state-of-the-art
performance.

### 10. [A Time-Enhanced Data Disentanglement Network for Traffic Flow Forecasting](http://arxiv.org/pdf/2506.02609v1)

Authors: Tianfan Jiang, Mei Wu, Wenchao Weng, Dewen Seng, Yiqian Lin

In recent years, traffic flow prediction has become a highlight in the field
of intelligent transportation systems. However, due to the temporal variations
and dynamic spatial correlations of traffic data, traffic prediction remains
highly challenging.Traditional spatiotemporal networks, which rely on
end-to-end training, often struggle to handle the diverse data dependencies of
multiple traffic flow patterns. Additionally, traffic flow variations are
highly sensitive to temporal information changes. Regrettably, other
researchers have not sufficiently recognized the importance of temporal
information.To address these challenges, we propose a novel approach called A
Time-Enhanced Data Disentanglement Network for Traffic Flow Forecasting
(TEDDN). This network disentangles the originally complex and intertwined
traffic data into stable patterns and trends. By flexibly learning temporal and
node information through a dynamic graph enhanced by a temporal feature
extraction module, TEDDN demonstrates significant efficacy in disentangling and
extracting complex traffic information. Experimental evaluations and ablation
studies on four real-world datasets validate the superiority of our method.

### Hardware Architecture

### 1. [Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention](http://arxiv.org/pdf/2506.02523v1)

Authors: Robin Geens, Marian Verhelst

Multi-Head Latent Attention (MLA), introduced in DeepSeek-V2, improves the
efficiency of large language models by projecting query, key, and value tensors
into a compact latent space. This architectural change reduces the KV-cache
size and significantly lowers memory bandwidth demands, particularly in the
autoregressive decode phase. This letter presents the first hardware-centric
analysis of MLA, comparing it to conventional Multi-Head Attention (MHA) and
evaluating its implications for accelerator performance. We identify two
alternative execution schemes of MLA--reusing, resp. recomputing latent
projection matrices--which offer distinct trade-offs between compute and memory
access. Using the Stream design space exploration framework, we model their
throughput and energy cost across a range of hardware platforms and find that
MLA can shift attention workloads toward the compute-bound regime.
  Our results show that MLA not only reduces bandwidth usage but also enables
adaptable execution strategies aligned with hardware constraints. Compared to
MHA, it provides more stable and efficient performance, particularly on
bandwidth-limited hardware platforms. These findings emphasize MLA's relevance
as a co-design opportunity for future AI accelerators.

### 2. [Large Processor Chip Model](http://arxiv.org/pdf/2506.02929v1)

Authors: Kaiyan Chang, Mingzhi Chen, Yunji Chen, Zhirong Chen, Dongrui Fan, Junfeng Gong, Nan Guo, Yinhe Han, Qinfen Hao, Shuo Hou, Xuan Huang, Pengwei Jin, Changxin Ke, Cangyuan Li, Guangli Li, Huawei Li, Kuan Li, Naipeng Li, Shengwen Liang, Cheng Liu, Hongwei Liu, Jiahua Liu, Junliang Lv, Jianan Mu, Jin Qin, Bin Sun, Chenxi Wang, Duo Wang, Mingjun Wang, Ying Wang, Chenggang Wu, Peiyang Wu, Teng Wu, Xiao Xiao, Mengyao Xie, Chenwei Xiong, Ruiyuan Xu, Mingyu Yan, Xiaochun Ye, Kuai Yu, Rui Zhang, Shuoming Zhang, Jiacheng Zhao

Computer System Architecture serves as a crucial bridge between software
applications and the underlying hardware, encompassing components like
compilers, CPUs, coprocessors, and RTL designs. Its development, from early
mainframes to modern domain-specific architectures, has been driven by rising
computational demands and advancements in semiconductor technology. However,
traditional paradigms in computer system architecture design are confronting
significant challenges, including a reliance on manual expertise, fragmented
optimization across software and hardware layers, and high costs associated
with exploring expansive design spaces. While automated methods leveraging
optimization algorithms and machine learning have improved efficiency, they
remain constrained by a single-stage focus, limited data availability, and a
lack of comprehensive human domain knowledge. The emergence of large language
models offers transformative opportunities for the design of computer system
architecture. By leveraging the capabilities of LLMs in areas such as code
generation, data analysis, and performance modeling, the traditional manual
design process can be transitioned to a machine-based automated design
approach. To harness this potential, we present the Large Processor Chip Model
(LPCM), an LLM-driven framework aimed at achieving end-to-end automated
computer architecture design. The LPCM is structured into three levels:
Human-Centric; Agent-Orchestrated; and Model-Governed. This paper utilizes 3D
Gaussian Splatting as a representative workload and employs the concept of
software-hardware collaborative design to examine the implementation of the
LPCM at Level 1, demonstrating the effectiveness of the proposed approach.
Furthermore, this paper provides an in-depth discussion on the pathway to
implementing Level 2 and Level 3 of the LPCM, along with an analysis of the
existing challenges.

### 3. [Minimal Neuron Circuits -- Part I: Resonators](http://arxiv.org/pdf/2506.02341v1)

Authors: Amr Nabil, T. Nandha Kumar, Haider Abbas F. Almurib

Spiking Neural Networks have earned increased recognition in recent years
owing to their biological plausibility and event-driven computation. Spiking
neurons are the fundamental building components of Spiking Neural Networks.
Those neurons act as computational units that determine the decision to fire an
action potential. This work presents a methodology to implement biologically
plausible yet scalable spiking neurons in hardware. We show that it is more
efficient to design neurons that mimic the $I_{Na,p}+I_{K}$ model rather than
the more complicated Hodgkin-Huxley model. We demonstrate our methodology by
presenting eleven novel minimal spiking neuron circuits in Parts I and II of
the paper. We categorize the neuron circuits presented into two types:
Resonators and Integrators. We discuss the methodology employed in designing
neurons of the resonator type in Part I, while we discuss neurons of the
integrator type in Part II. In part I, we postulate that Sodium channels
exhibit type-N negative differential resistance. Consequently, we present three
novel minimal neuron circuits that use type-N negative differential resistance
circuits or devices as the Sodium channel. Nevertheless, the aim of the paper
is not to present a set of minimal neuron circuits but rather the methodology
utilized to construct those circuits.

### 4. [Memory Access Vectors: Improving Sampling Fidelity for CPU Performance Simulations](http://arxiv.org/pdf/2506.02344v1)

Authors: Sriyash Caculo, Mahesh Madhav, Jeff Baxter

Accurate performance projection of large-scale benchmarks is essential for
CPU architects to evaluate and optimize future processor designs. SimPoint
sampling, which uses Basic Block Vectors (BBVs), is a widely adopted technique
to reduce simulation time by selecting representative program phases. However,
BBVs often fail to capture the behavior of applications with extensive
array-indirect memory accesses, leading to inaccurate projections. In
particular, the 523.xalancbmk_r benchmark exhibits complex data movement
patterns that challenge traditional SimPoint methods. To address this, we
propose enhancing SimPoint's BBV methodology by incorporating Memory Access
Vectors (MAV), a microarchitecture independent technique that tracks functional
memory access patterns. This combined approach significantly improves the
projection accuracy of 523.xalancbmk_r on a 192-core system-on-chip, increasing
it from 80% to 98%.

### 5. [CLONE: Customizing LLMs for Efficient Latency-Aware Inference at the Edge](http://arxiv.org/pdf/2506.02847v1)

Authors: Chunlin Tian, Xinpeng Qin, Kahou Tam, Li Li, Zijian Wang, Yuanzhe Zhao, Minglei Zhang, Chengzhong Xu

Deploying large language models (LLMs) on edge devices is crucial for
delivering fast responses and ensuring data privacy. However, the limited
storage, weight, and power of edge devices make it difficult to deploy
LLM-powered applications. These devices must balance latency requirements with
energy consumption and model accuracy. In this paper, we first quantify the
challenges of deploying LLMs on off-the-shelf edge devices and then we present
CLONE, an in-depth algorithm-hardware co-design at both the model- and
system-level that intelligently integrates real-time, energy optimization while
maintaining robust generality. In order to maximize the synergistic benefits of
these algorithms in always-on and intermediate edge computing settings, we
specialize in a 28nm scalable hardware accelerator system. We implement and
extensively evaluate CLONE on two off-the-shelf edge platforms. Experiments
show that CLONE effectively accelerates the inference process up to 11.92x, and
saves energy up to 7.36x, while maintaining high-generation.

### Computational Complexity

### 1. [Convergence and efficiency proof of quantum imaginary time evolution for bounded order systems](http://arxiv.org/pdf/2506.03014v1)

Authors: Tobias Hartung, Karl Jansen

Many current and near-future applications of quantum computing utilise
parametric families of quantum circuits and variational methods to find optimal
values for these parameters. Solving a quantum computational problem with such
variational methods relies on minimising some cost function, e.g., the energy
of a physical system. As such, this is similar to the training process in
machine learning and variational quantum simulations can therefore suffer from
similar problems encountered in machine learning training. This includes
non-convergence to the global minimum due to local minima as well as critical
slowing down. In this article, we analyse the imaginary time evolution as a
means of compiling parametric quantum circuits and finding optimal parameters,
and show that it guarantees convergence to the global minimum without critical
slowing down. We also show that the compilation process, including the task of
finding optimal parameters, can be performed efficiently up to an arbitrary
error threshold if the underlying physical system is of bounded order. This
includes many relevant computational problems, e.g., local physical theories
and combinatorial optimisation problems such as the flight-to-gate assignment
problem. In particular, we show a priori estimates on the success probability
for these combinatorial optimisation problems. There seem to be no known
classical methods with similar efficiency and convergence guarantees. Meanwhile
the imaginary time evolution method can be implemented on current quantum
computers.

### Computational Engineering

### 1. [Generative AI for Predicting 2D and 3D Wildfire Spread: Beyond Physics-Based Models and Traditional Deep Learning](http://arxiv.org/pdf/2506.02485v1)

Authors: Haowen Xu, Sisi Zlatanova, Ruiyu Liang, Ismet Canbulat

Wildfires continue to inflict devastating human, environmental, and economic
losses globally, as tragically exemplified by the 2025 Los Angeles wildfire and
the urgent demand for more effective response strategies. While physics-based
and deep learning models have advanced wildfire simulation, they face critical
limitations in predicting and visualizing multimodal fire spread in real time,
particularly in both 2D and 3D spatial domains using dynamically updated GIS
data. These limitations hinder timely emergency response, infrastructure
protection, and community safety. Generative AI has recently emerged as a
transformative approach across research and industry. Models such as Generative
Adversarial Networks (GANs), Variational Autoencoders (VAEs), Transformers, and
diffusion-based architectures offer distinct advantages over traditional
methods, including the integration of multimodal data, generation of diverse
scenarios under uncertainty, and improved modeling of wildfire dynamics across
spatial and temporal scales. This position paper advocates for the adoption of
generative AI as a foundational framework for wildfire prediction. We explore
how such models can enhance 2D fire spread forecasting and enable more
realistic, scalable 3D simulations. Additionally, we employ a novel human-AI
collaboration framework using large language models (LLMs) for automated
knowledge extraction, literature synthesis, and bibliometric mapping. Looking
ahead, we identify five key visions for integrating generative AI into wildfire
management: multimodal approaches, AI foundation models, conversational AI
systems, edge-computing-based scenario generation, and cognitive digital twins.
We also address three major challenges accompanying these opportunities and
propose potential solutions to support their implementation.

### 2. [On the fracture mechanics validity of small scale tests](http://arxiv.org/pdf/2506.02538v1)

Authors: C. Cui, L. Cupertino-Malheiros, Z. Xiong, E. Martínez-Pañeda

There is growing interest in conducting small-scale tests to gain additional
insight into the fracture behaviour of components across a wide range of
materials. For example, micro-scale mechanical tests inside of a microscope
(\emph{in situ}) enable direct, high-resolution observation of the interplay
between crack growth and microstructural phenomena (e.g., dislocation behaviour
or the fracture resistance of a particular interface), and sub-size samples are
increasingly used when only a limited amount of material is available. However,
to obtain quantitative insight and extract relevant fracture parameters, the
sample must be sufficiently large for a $J$- (HRR) or a $K$-field to exist. We
conduct numerical and semi-analytical studies to map the conditions (sample
geometry, material) that result in a valid, quantitative fracture experiment.
Specifically, for a wide range of material properties, crack lengths and sample
dimensions, we establish the maximum value of the $J$-integral where an HRR
field ceases to exist (i.e., the maximum $J$ value at which fracture must occur
for the test to be valid, $J_\mathrm{max}$). Maps are generated to establish
the maximum valid $J$ value ($J_\mathrm{max}$) as a function of yield strength,
strain hardening and minimum sample size. These maps are then used to discuss
the existing experimental literature and provide guidance on how to conduct
quantitative experiments. Finally, our study is particularised to the analysis
of metals that have been embrittled due to hydrogen exposure. The response of
relevant materials under hydrogen-containing environments are superimposed on
the aforementioned maps, determining the conditions that will enable
quantitative insight.

### 3. [Enriching Location Representation with Detailed Semantic Information](http://arxiv.org/pdf/2506.02744v1)

Authors: Junyuan Liu, Xinglei Wang, Tao Cheng

Spatial representations that capture both structural and semantic
characteristics of urban environments are essential for urban modeling.
Traditional spatial embeddings often prioritize spatial proximity while
underutilizing fine-grained contextual information from places. To address this
limitation, we introduce CaLLiPer+, an extension of the CaLLiPer model that
systematically integrates Point-of-Interest (POI) names alongside categorical
labels within a multimodal contrastive learning framework. We evaluate its
effectiveness on two downstream tasks, land use classification and
socioeconomic status distribution mapping, demonstrating consistent performance
gains of 4% to 11% over baseline methods. Additionally, we show that
incorporating POI names enhances location retrieval, enabling models to capture
complex urban concepts with greater precision. Ablation studies further reveal
the complementary role of POI names and the advantages of leveraging pretrained
text encoders for spatial representations. Overall, our findings highlight the
potential of integrating fine-grained semantic attributes and multimodal
learning techniques to advance the development of urban foundation models.

### 4. [TL;DR: Too Long, Do Re-weighting for Effcient LLM Reasoning Compression](http://arxiv.org/pdf/2506.02678v1)

Authors: Zhong-Zhi Li, Xiao Liang, Zihao Tang, Lei Ji, Peijie Wang, Haotian Xu, Xing W, Haizhen Huang, Weiwei Deng, Ying Nian Wu, Yeyun Gong, Zhijiang Guo, Xiao Liu, Fei Yin, Cheng-Lin Liu

Large Language Models (LLMs) have recently achieved remarkable progress by
leveraging Reinforcement Learning and extended Chain-of-Thought (CoT)
techniques. However, the challenge of performing efficient language
reasoning--especially during inference with extremely long outputs--has drawn
increasing attention from the research community. In this work, we propose a
dynamic ratio-based training pipeline that does not rely on sophisticated data
annotations or interpolation between multiple models. We continuously balance
the weights between the model's System-1 and System-2 data to eliminate
redundant reasoning processes while preserving the model's reasoning
capability. We validate our approach across models on DeepSeek-R1-Distill-7B
and DeepSeek-R1-Distill-14B and on a diverse set of benchmarks with varying
difficulty levels. Our method significantly reduces the number of output tokens
by nearly 40% while maintaining the accuracy of the reasoning. Our code and
data will be available soon.

### 5. [A mesoscale phase-field model of intergranular liquid lithium corrosion of ferritic/martensitic steels](http://arxiv.org/pdf/2506.02776v1)

Authors: A. Lhoest, S. Kovacevic, D. Nguyen-Manh, J. Lim, E. Martínez-Pañeda, M. Wenman

A phase-field model is developed to simulate intergranular corrosion of
ferritic/martensitic steels exposed to liquid lithium. The chromium
concentration of the material is used to track the mass transport within the
metal and liquid (corrosive) phase. The framework naturally captures
intergranular corrosion by enhancing the diffusion of chromium along grain
boundaries relative to the grain bulk with no special treatment for the
corrosion front evolution. The formulation applies to arbitrary 2D and 3D
polycrystalline geometries. The framework reproduces experimental measurements
of weight loss and corrosion depth for a 9 wt\% Cr ferritic/martensitic steel
exposed to static lithium at 600 $^\circ$C. A sensitivity analysis, varying
near-surface grain density, grain size, and chromium depletion thickness,
highlights the microstructural influence in the corrosion process. Moreover,
the significance of saturation is considered and evaluated. Simulation results
show that near-surface grain density is a deciding factor, whereas grain size
dictates the susceptibility to intergranular corrosion.

### 6. [Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning](http://arxiv.org/pdf/2506.02911v1)

Authors: Yin Fang, Qiao Jin, Guangzhi Xiong, Bowen Jin, Xianrui Zhong, Siru Ouyang, Aidong Zhang, Jiawei Han, Zhiyong Lu

Cell type annotation is a key task in analyzing the heterogeneity of
single-cell RNA sequencing data. Although recent foundation models automate
this process, they typically annotate cells independently, without considering
batch-level cellular context or providing explanatory reasoning. In contrast,
human experts often annotate distinct cell types for different cell clusters
based on their domain knowledge. To mimic this workflow, we introduce the
CellPuzzles task, where the objective is to assign unique cell types to a batch
of cells. This benchmark spans diverse tissues, diseases, and donor conditions,
and requires reasoning across the batch-level cellular context to ensure label
uniqueness. We find that off-the-shelf large language models (LLMs) struggle on
CellPuzzles, with the best baseline (OpenAI's o1) achieving only 19.0%
batch-level accuracy. To fill this gap, we propose Cell-o1, a 7B LLM trained
via supervised fine-tuning on distilled reasoning traces, followed by
reinforcement learning with batch-level rewards. Cell-o1 achieves
state-of-the-art performance, outperforming o1 by over 73% and generalizing
well across contexts. Further analysis of training dynamics and reasoning
behaviors provides insights into batch-level annotation performance and
emergent expert-like reasoning. Code and data are available at
https://github.com/ncbi-nlp/cell-o1.

### Computation and Language

### 1. [One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL](http://arxiv.org/pdf/2506.02338v1)

Authors: Hyungjoo Chae, Dongjin Kang, Jihyuk Kim, Beong-woo Kwak, Sunghyun Park, Haeju Park, Jinyoung Yeo, Moontae Lee, Kyungjae Lee

With the release of R1, a publicly available large reasoning model (LRM),
researchers commonly train new LRMs by training language models on R1's long
chain-of-thought (CoT) inferences. While prior works show that LRMs'
capabilities can be reproduced through direct distillation, the continued
reliance on the existing models (e.g., R1) remains a critical limitation in
advancing the field. As a first step toward independent LRM development, this
paper explores the possibility of constructing a long CoT dataset with LLMs
that are not trained for inference-time scaling. To this end, we present the
Long CoT Collection, a dataset of 100K CoT rationales annotated using existing
short CoT LLMs. We develop a pipeline that induces o1's novel reasoning
strategies into short CoT LLMs, enabling them to think longer and introducing
controllability over the thought budget to better manage the overthinking
problem. Our extensive analyses validate that our dataset achieves quality
comparable to--or slightly below--R1. Furthermore, our experiments demonstrate
that training on our dataset not only strengthens general reasoning skills, but
also provides a strong foundation for reinforcement learning--models
initialized on our data achieve 2-3x larger gains with RLVR.

### 2. [STORYTELLER: An Enhanced Plot-Planning Framework for Coherent and Cohesive Story Generation](http://arxiv.org/pdf/2506.02347v1)

Authors: Jiaming Li, Yukun Chen, Ziqiang Liu, Minghuan Tan, Lei Zhang, Yunshui Li, Run Luo, Longze Chen, Jing Luo, Ahmadreza Argha, Hamid Alinejad-Rokny, Wei Zhou, Min Yang

Stories are central to human culture, serving to share ideas, preserve
traditions, and foster connections. Automatic story generation, a key
advancement in artificial intelligence (AI), offers new possibilities for
creating personalized content, exploring creative ideas, and enhancing
interactive experiences. However, existing methods struggle to maintain
narrative coherence and logical consistency. This disconnect compromises the
overall storytelling experience, underscoring the need for substantial
improvements. Inspired by human cognitive processes, we introduce Storyteller,
a novel approach that systemically improves the coherence and consistency of
automatically generated stories. Storyteller introduces a plot node structure
based on linguistically grounded subject verb object (SVO) triplets, which
capture essential story events and ensure a consistent logical flow. Unlike
previous methods, Storyteller integrates two dynamic modules, the STORYLINE and
narrative entity knowledge graph (NEKG),that continuously interact with the
story generation process. This integration produces structurally sound,
cohesive and immersive narratives. Extensive experiments demonstrate that
Storyteller significantly outperforms existing approaches, achieving an 84.33%
average win rate through human preference evaluation. At the same time, it is
also far ahead in other aspects including creativity, coherence, engagement,
and relevance.

### 3. [Truth over Tricks: Measuring and Mitigating Shortcut Learning in Misinformation Detection](http://arxiv.org/pdf/2506.02350v1)

Authors: Herun Wan, Jiaying Wu, Minnan Luo, Zhi Zeng, Zhixiong Su

Misinformation detection models often rely on superficial cues (i.e.,
\emph{shortcuts}) that correlate with misinformation in training data but fail
to generalize to the diverse and evolving nature of real-world misinformation.
This issue is exacerbated by large language models (LLMs), which can easily
generate convincing misinformation through simple prompts. We introduce
TruthOverTricks, a unified evaluation paradigm for measuring shortcut learning
in misinformation detection. TruthOverTricks categorizes shortcut behaviors
into intrinsic shortcut induction and extrinsic shortcut injection, and
evaluates seven representative detectors across 14 popular benchmarks, along
with two new factual misinformation datasets, NQ-Misinfo and Streaming-Misinfo.
Empirical results reveal that existing detectors suffer severe performance
degradation when exposed to both naturally occurring and adversarially crafted
shortcuts. To address this, we propose SMF, an LLM-augmented data augmentation
framework that mitigates shortcut reliance through paraphrasing, factual
summarization, and sentiment normalization. SMF consistently enhances
robustness across 16 benchmarks, encouraging models to rely on deeper semantic
understanding rather than shortcut cues. To promote the development of
misinformation detectors, we have published the resources publicly at
https://github.com/whr000001/TruthOverTricks.

### 4. [AnswerCarefully: A Dataset for Improving the Safety of Japanese LLM Output](http://arxiv.org/pdf/2506.02372v1)

Authors: Hisami Suzuki, Satoru Katsumata, Takashi Kodama, Tetsuro Takahashi, Kouta Nakayama, Satoshi Sekine

In this paper we present AnswerCarefully, a dataset for promoting the safety
and appropriateness of Japanese LLM outputs. The dataset consists of 1,800
pairs of questions and reference answers, where the questions require special
attention in answering. It covers a wide range of risk categories established
in prior English-language datasets, but the data samples are original in that
they are manually created to reflect the socio-cultural context of LLM usage in
Japan. We show that using this dataset for instruction to fine-tune a Japanese
LLM led to improved output safety without compromising the utility of general
responses. We also report the results of a safety evaluation of 12 Japanese
LLMs using this dataset as a benchmark. Finally, we describe the latest update
on the dataset which provides English translations and annotations of the
questions, aimed at facilitating the derivation of similar datasets in
different languages and regions.

### 5. [From Anger to Joy: How Nationality Personas Shape Emotion Attribution in Large Language Models](http://arxiv.org/pdf/2506.02431v1)

Authors: Mahammed Kamruzzaman, Abdullah Al Monsur, Gene Louis Kim, Anshuman Chhabra

Emotions are a fundamental facet of human experience, varying across
individuals, cultural contexts, and nationalities. Given the recent success of
Large Language Models (LLMs) as role-playing agents, we examine whether LLMs
exhibit emotional stereotypes when assigned nationality-specific personas.
Specifically, we investigate how different countries are represented in
pre-trained LLMs through emotion attributions and whether these attributions
align with cultural norms. Our analysis reveals significant nationality-based
differences, with emotions such as shame, fear, and joy being
disproportionately assigned across regions. Furthermore, we observe notable
misalignment between LLM-generated and human emotional responses, particularly
for negative emotions, highlighting the presence of reductive and potentially
biased stereotypes in LLM outputs.

### 6. [Should LLM Safety Be More Than Refusing Harmful Instructions?](http://arxiv.org/pdf/2506.02442v1)

Authors: Utsav Maskey, Mark Dras, Usman Naseem

This paper presents a systematic evaluation of Large Language Models' (LLMs)
behavior on long-tail distributed (encrypted) texts and their safety
implications. We introduce a two-dimensional framework for assessing LLM
safety: (1) instruction refusal-the ability to reject harmful obfuscated
instructions, and (2) generation safety-the suppression of generating harmful
responses. Through comprehensive experiments, we demonstrate that models that
possess capabilities to decrypt ciphers may be susceptible to
mismatched-generalization attacks: their safety mechanisms fail on at least one
safety dimension, leading to unsafe responses or over-refusal. Based on these
findings, we evaluate a number of pre-LLM and post-LLM safeguards and discuss
their strengths and limitations. This work contributes to understanding the
safety of LLM in long-tail text scenarios and provides directions for
developing robust safety mechanisms.

### 7. [MidPO: Dual Preference Optimization for Safety and Helpfulness in Large Language Models via a Mixture of Experts Framework](http://arxiv.org/pdf/2506.02460v1)

Authors: Yupeng Qi, Ziyu Lyu, Min Yang, Yanlin Wang, Lu Bai, Lixin Cui

As large language models (LLMs) are increasingly applied across various
domains, enhancing safety while maintaining the helpfulness of LLMs has become
a critical challenge. Recent studies solve this problem through
safety-constrained online preference optimization or safety-constrained offline
preference optimization. However, the safety-constrained online methods often
suffer from excessive safety, which might reduce helpfulness, while the
safety-constrained offline methods perform poorly in adaptively balancing
safety and helpfulness. To address these limitations, we propose MidPO, a
\textbf{\underline{Mi}}xture of Experts (MoE) framework for safety-helpfulness
\textbf{\underline{d}}ual \textbf{\underline{P}}reference
\textbf{\underline{O}}ptimization. Firstly, MidPO devises single-preference
enhanced direct preference optimization approach to transform the base model
into two independent experts, termed safety and helpfulness experts, and
fine-tunes the two independent experts for optimal safety or helpfulness
performance. Secondly, to achieve an effective balance between safety and
helpfulness, MidPO incorporates the two experts into the MoE framework and
designs a dynamic routing mechanism to allocate contributions from each expert
adaptively. We conduct quantitative and qualitative experiments on three
popular datasets to demonstrate the proposed MidPO significantly outperforms
state-of-the-art approaches in both safety and helpfulness. The code and models
will be released.

### 8. [XToM: Exploring the Multilingual Theory of Mind for Large Language Models](http://arxiv.org/pdf/2506.02461v1)

Authors: Chunkit Chan, Yauwai Yim, Hongchuan Zeng, Zhiying Zou, Xinyuan Cheng, Zhifan Sun, Zheye Deng, Kawai Chung, Yuzhuo Ao, Yixiang Fan, Cheng Jiayang, Ercong Nie, Ginny Y. Wong, Helmut Schmid, Hinrich Schütze, Simon See, Yangqiu Song

Theory of Mind (ToM), the ability to infer mental states in others, is
pivotal for human social cognition. Existing evaluations of ToM in LLMs are
largely limited to English, neglecting the linguistic diversity that shapes
human cognition. This limitation raises a critical question: can LLMs exhibit
Multilingual Theory of Mind, which is the capacity to reason about mental
states across diverse linguistic contexts? To address this gap, we present
XToM, a rigorously validated multilingual benchmark that evaluates ToM across
five languages and incorporates diverse, contextually rich task scenarios.
Using XToM, we systematically evaluate LLMs (e.g., DeepSeek R1), revealing a
pronounced dissonance: while models excel in multilingual language
understanding, their ToM performance varies across languages. Our findings
expose limitations in LLMs' ability to replicate human-like mentalizing across
linguistic contexts.

### 9. [FroM: Frobenius Norm-Based Data-Free Adaptive Model Merging](http://arxiv.org/pdf/2506.02478v1)

Authors: Zijian Li, Xiaocheng Feng, Huixin Liu, Yichong Huang, Ting Liu, Bing Qin

With the development of large language models, fine-tuning has emerged as an
effective method to enhance performance in specific scenarios by injecting
domain-specific knowledge. In this context, model merging techniques provide a
solution for fusing knowledge from multiple fine-tuning models by combining
their parameters. However, traditional methods often encounter task
interference when merging full fine-tuning models, and this problem becomes
even more evident in parameter-efficient fine-tuning scenarios. In this paper,
we introduce an improvement to the RegMean method, which indirectly leverages
the training data to approximate the outputs of the linear layers before and
after merging. We propose an adaptive merging method called FroM, which
directly measures the model parameters using the Frobenius norm, without any
training data. By introducing an additional hyperparameter for control, FroM
outperforms baseline methods across various fine-tuning scenarios, alleviating
the task interference problem.

### 10. [ORPP: Self-Optimizing Role-playing Prompts to Enhance Language Model Capabilities](http://arxiv.org/pdf/2506.02480v1)

Authors: Yifan Duan, Yihong Tang, Kehai Chen, Liqiang Nie, Min Zhang

High-quality prompts are crucial for eliciting outstanding performance from
large language models (LLMs) on complex tasks. Existing research has explored
model-driven strategies for prompt optimization. However, these methods often
suffer from high computational overhead or require strong optimization
capabilities from the model itself, which limits their broad applicability.To
address these challenges, we propose ORPP (Optimized Role-Playing Prompt),a
framework that enhances model performance by optimizing and generating
role-playing prompts. The core idea of ORPP is to confine the prompt search
space to role-playing scenarios, thereby fully activating the model's intrinsic
capabilities through carefully crafted, high-quality role-playing prompts.
Specifically, ORPP first performs iterative optimization on a small subset of
training samples to generate high-quality role-playing prompts. Then,
leveraging the model's few-shot learning capability, it transfers the
optimization experience to efficiently generate suitable prompts for the
remaining samples.Our experimental results show that ORPP not only matches but
in most cases surpasses existing mainstream prompt optimization methods in
terms of performance. Notably, ORPP demonstrates superior "plug-and-play"
capability. In most cases, it can be integrated with various other prompt
methods and further enhance their effectiveness.

### Cryptography and Security

### 1. [Attention Knows Whom to Trust: Attention-based Trust Management for LLM Multi-Agent Systems](http://arxiv.org/pdf/2506.02546v1)

Authors: Pengfei He, Zhenwei Dai, Xianfeng Tang, Yue Xing, Hui Liu, Jingying Zeng, Qiankun Peng, Shrivats Agrawal, Samarth Varshney, Suhang Wang, Jiliang Tang, Qi He

Large Language Model-based Multi-Agent Systems (LLM-MAS) have demonstrated
strong capabilities in solving complex tasks but remain vulnerable when agents
receive unreliable messages. This vulnerability stems from a fundamental gap:
LLM agents treat all incoming messages equally without evaluating their
trustworthiness. While some existing studies approach the trustworthiness, they
focus on a single type of harmfulness rather than analyze it in a holistic
approach from multiple trustworthiness perspectives. In this work, we propose
Attention Trust Score (A-Trust), a lightweight, attention-based method for
evaluating message trustworthiness. Inspired by human communication
literature[1], through systematically analyzing attention behaviors across six
orthogonal trust dimensions, we find that certain attention heads in the LLM
specialize in detecting specific types of violations. Leveraging these
insights, A-Trust directly infers trustworthiness from internal attention
patterns without requiring external prompts or verifiers. Building upon
A-Trust, we develop a principled and efficient trust management system (TMS)
for LLM-MAS, enabling both message-level and agent-level trust assessment.
Experiments across diverse multi-agent settings and tasks demonstrate that
applying our TMS significantly enhances robustness against malicious inputs.

### 2. [Tarallo: Evading Behavioral Malware Detectors in the Problem Space](http://arxiv.org/pdf/2506.02660v1)

Authors: Gabriele Digregorio, Salvatore Maccarrone, Mario D'Onghia, Luigi Gallo, Michele Carminati, Mario Polino, Stefano Zanero

Machine learning algorithms can effectively classify malware through dynamic
behavior but are susceptible to adversarial attacks. Existing attacks, however,
often fail to find an effective solution in both the feature and problem
spaces. This issue arises from not addressing the intrinsic nondeterministic
nature of malware, namely executing the same sample multiple times may yield
significantly different behaviors. Hence, the perturbations computed for a
specific behavior may be ineffective for others observed in subsequent
executions. In this paper, we show how an attacker can augment their chance of
success by leveraging a new and more efficient feature space algorithm for
sequential data, which we have named PS-FGSM, and by adopting two problem space
strategies specially tailored to address nondeterminism in the problem space.
We implement our novel algorithm and attack strategies in Tarallo, an
end-to-end adversarial framework that significantly outperforms previous works
in both white and black-box scenarios. Our preliminary analysis in a sandboxed
environment and against two RNN-based malware detectors, shows that Tarallo
achieves a success rate up to 99% on both feature and problem space attacks
while significantly minimizing the number of modifications required for
misclassification.

### 3. [Decentralized COVID-19 Health System Leveraging Blockchain](http://arxiv.org/pdf/2506.02674v1)

Authors: Lingsheng Chen, Shipeng Ye, Xiaoqi Li

With the development of the Internet, the amount of data generated by the
medical industry each year has grown exponentially. The Electronic Health
Record (EHR) manages the electronic data generated during the user's treatment
process. Typically, an EHR data manager belongs to a medical institution. This
traditional centralized data management model has many unreasonable or
inconvenient aspects, such as difficulties in data sharing, and it is hard to
verify the authenticity and integrity of the data. The decentralized,
non-forgeable, data unalterable and traceable features of blockchain are in
line with the application requirements of EHR. This paper takes the most common
COVID-19 as the application scenario and designs a COVID-19 health system based
on blockchain, which has extensive research and application value. Considering
that the public and transparent nature of blockchain violates the privacy
requirements of some health data, in the system design stage, from the
perspective of practical application, the data is divided into public data and
private data according to its characteristics. For private data, data
encryption methods are adopted to ensure data privacy. The searchable
encryption technology is combined with blockchain technology to achieve the
retrieval function of encrypted data. Then, the proxy re-encryption technology
is used to realize authorized access to data. In the system implementation
part, based on the Hyperledger Fabric architecture, some functions of the
system design are realized, including data upload, retrieval of the latest data
and historical data. According to the environment provided by the development
architecture, Go language chaincode (smart contract) is written to implement
the relevant system functions.

### 4. [Poster: FedBlockParadox -- A Framework for Simulating and Securing Decentralized Federated Learning](http://arxiv.org/pdf/2506.02679v1)

Authors: Gabriele Digregorio, Francesco Bleggi, Federico Caroli, Michele Carminati, Stefano Zanero, Stefano Longari

A significant body of research in decentralized federated learning focuses on
combining the privacy-preserving properties of federated learning with the
resilience and transparency offered by blockchain-based systems. While these
approaches are promising, they often lack flexible tools to evaluate system
robustness under adversarial conditions. To fill this gap, we present
FedBlockParadox, a modular framework for modeling and evaluating decentralized
federated learning systems built on blockchain technologies, with a focus on
resilience against a broad spectrum of adversarial attack scenarios. It
supports multiple consensus protocols, validation methods, aggregation
strategies, and configurable attack models. By enabling controlled experiments,
FedBlockParadox provides a valuable resource for researchers developing secure,
decentralized learning solutions. The framework is open-source and built to be
extensible by the community.

### 5. [Privacy Leaks by Adversaries: Adversarial Iterations for Membership Inference Attack](http://arxiv.org/pdf/2506.02711v1)

Authors: Jing Xue, Zhishen Sun, Haishan Ye, Luo Luo, Xiangyu Chang, Ivor Tsang, Guang Dai

Membership inference attack (MIA) has become one of the most widely used and
effective methods for evaluating the privacy risks of machine learning models.
These attacks aim to determine whether a specific sample is part of the model's
training set by analyzing the model's output. While traditional membership
inference attacks focus on leveraging the model's posterior output, such as
confidence on the target sample, we propose IMIA, a novel attack strategy that
utilizes the process of generating adversarial samples to infer membership. We
propose to infer the member properties of the target sample using the number of
iterations required to generate its adversarial sample. We conduct experiments
across multiple models and datasets, and our results demonstrate that the
number of iterations for generating an adversarial sample is a reliable feature
for membership inference, achieving strong performance both in black-box and
white-box attack scenarios. This work provides a new perspective for evaluating
model privacy and highlights the potential of adversarial example-based
features for privacy leakage assessment.

### 6. [When Blockchain Meets Crawlers: Real-time Market Analytics in Solana NFT Markets](http://arxiv.org/pdf/2506.02892v1)

Authors: Chengxin Shen, Zhongwen Li, Xiaoqi Li, Zongwei Li

In this paper, we design and implement a web crawler system based on the
Solana blockchain for the automated collection and analysis of market data for
popular non-fungible tokens (NFTs) on the chain. Firstly, the basic information
and transaction data of popular NFTs on the Solana chain are collected using
the Selenium tool. Secondly, the transaction records of the Magic Eden trading
market are thoroughly analyzed by combining them with the Scrapy framework to
examine the price fluctuations and market trends of NFTs. In terms of data
analysis, this paper employs time series analysis to examine the dynamics of
the NFT market and seeks to identify potential price patterns. In addition, the
risk and return of different NFTs are evaluated using the mean-variance
optimization model, taking into account their characteristics, such as
illiquidity and market volatility, to provide investors with data-driven
portfolio recommendations. The experimental results show that the combination
of crawler technology and financial analytics can effectively analyze NFT data
on the Solana blockchain and provide timely market insights and investment
strategies. This study provides a reference for further exploration in the
field of digital currencies.

### 7. [An Algorithmic Pipeline for GDPR-Compliant Healthcare Data Anonymisation: Moving Toward Standardisation](http://arxiv.org/pdf/2506.02942v1)

Authors: Hamza Khan, Lore Menten, Liesbet M. Peeters

High-quality real-world data (RWD) is essential for healthcare but must be
transformed to comply with the General Data Protection Regulation (GDPR). GDPRs
broad definitions of quasi-identifiers (QIDs) and sensitive attributes (SAs)
complicate implementation. We aim to standardise RWD anonymisation for GDPR
compliance while preserving data utility by introducing an algorithmic method
to identify QIDs and SAs and evaluate utility in anonymised datasets. We
conducted a systematic literature review via ProQuest and PubMed to inform a
three-stage anonymisation pipeline: identification, de-identification, and
quasi-identifier dimension evaluation. The pipeline was implemented, validated,
and tested on two mock RWD datasets (500 and 1000 rows). Privacy was assessed
using k-anonymity, l-diversity, and t-closeness; utility was measured by
non-uniform entropy (NUE). The review yielded two studies on QID/SA
identification and five on utility metrics. Applying the pipeline, attributes
were classified by re-identification risk using alpha and beta thresholds (25
percent/1 percent for 500 rows; 10 percent/1 percent for 1000 rows). Privacy
metrics improved k-anonymity from 1 to 4 (500 rows) and 1 to 110 (1000 rows).
NUE scores were 69.26 percent and 69.05 percent, respectively, indicating
consistent utility despite varying privacy gains. We present a GDPR-compliant
anonymisation pipeline for healthcare RWD that provides a reproducible approach
to QID/SA identification and utility evaluation; publicly available code
promotes standardisation, data privacy, and open science.

### 8. [MISLEADER: Defending against Model Extraction with Ensembles of Distilled Models](http://arxiv.org/pdf/2506.02362v1)

Authors: Xueqi Cheng, Minxing Zheng, Shixiang Zhu, Yushun Dong

Model extraction attacks aim to replicate the functionality of a black-box
model through query access, threatening the intellectual property (IP) of
machine-learning-as-a-service (MLaaS) providers. Defending against such attacks
is challenging, as it must balance efficiency, robustness, and utility
preservation in the real-world scenario. Despite the recent advances, most
existing defenses presume that attacker queries have out-of-distribution (OOD)
samples, enabling them to detect and disrupt suspicious inputs. However, this
assumption is increasingly unreliable, as modern models are trained on diverse
datasets and attackers often operate under limited query budgets. As a result,
the effectiveness of these defenses is significantly compromised in realistic
deployment scenarios. To address this gap, we propose MISLEADER (enseMbles of
dIStiLled modEls Against moDel ExtRaction), a novel defense strategy that does
not rely on OOD assumptions. MISLEADER formulates model protection as a bilevel
optimization problem that simultaneously preserves predictive fidelity on
benign inputs and reduces extractability by potential clone models. Our
framework combines data augmentation to simulate attacker queries with an
ensemble of heterogeneous distilled models to improve robustness and diversity.
We further provide a tractable approximation algorithm and derive theoretical
error bounds to characterize defense effectiveness. Extensive experiments
across various settings validate the utility-preserving and
extraction-resistant properties of our proposed defense strategy. Our code is
available at https://github.com/LabRAI/MISLEADER.

### 9. [VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents](http://arxiv.org/pdf/2506.02456v1)

Authors: Tri Cao, Bennett Lim, Yue Liu, Yuan Sui, Yuexin Li, Shumin Deng, Lin Lu, Nay Oo, Shuicheng Yan, Bryan Hooi

Computer-Use Agents (CUAs) with full system access enable powerful task
automation but pose significant security and privacy risks due to their ability
to manipulate files, access user data, and execute arbitrary commands. While
prior work has focused on browser-based agents and HTML-level attacks, the
vulnerabilities of CUAs remain underexplored. In this paper, we investigate
Visual Prompt Injection (VPI) attacks, where malicious instructions are
visually embedded within rendered user interfaces, and examine their impact on
both CUAs and Browser-Use Agents (BUAs). We propose VPI-Bench, a benchmark of
306 test cases across five widely used platforms, to evaluate agent robustness
under VPI threats. Each test case is a variant of a web platform, designed to
be interactive, deployed in a realistic environment, and containing a visually
embedded malicious prompt. Our empirical study shows that current CUAs and BUAs
can be deceived at rates of up to 51% and 100%, respectively, on certain
platforms. The experimental results also indicate that system prompt defenses
offer only limited improvements. These findings highlight the need for robust,
context-aware defenses to ensure the safe deployment of multimodal AI agents in
real-world environments. The code and dataset are available at:
https://github.com/cua-framework/agents

### 10. [BitBypass: A New Direction in Jailbreaking Aligned Large Language Models with Bitstream Camouflage](http://arxiv.org/pdf/2506.02479v1)

Authors: Kalyan Nakka, Nitesh Saxena

The inherent risk of generating harmful and unsafe content by Large Language
Models (LLMs), has highlighted the need for their safety alignment. Various
techniques like supervised fine-tuning, reinforcement learning from human
feedback, and red-teaming were developed for ensuring the safety alignment of
LLMs. However, the robustness of these aligned LLMs is always challenged by
adversarial attacks that exploit unexplored and underlying vulnerabilities of
the safety alignment. In this paper, we develop a novel black-box jailbreak
attack, called BitBypass, that leverages hyphen-separated bitstream camouflage
for jailbreaking aligned LLMs. This represents a new direction in jailbreaking
by exploiting fundamental information representation of data as continuous
bits, rather than leveraging prompt engineering or adversarial manipulations.
Our evaluation of five state-of-the-art LLMs, namely GPT-4o, Gemini 1.5, Claude
3.5, Llama 3.1, and Mixtral, in adversarial perspective, revealed the
capabilities of BitBypass in bypassing their safety alignment and tricking them
into generating harmful and unsafe content. Further, we observed that BitBypass
outperforms several state-of-the-art jailbreak attacks in terms of stealthiness
and attack success. Overall, these results highlights the effectiveness and
efficiency of BitBypass in jailbreaking these state-of-the-art LLMs.

### Computer Vision and Pattern Recognition

### 1. [Generalized Category Discovery via Reciprocal Learning and Class-Wise Distribution Regularization](http://arxiv.org/pdf/2506.02334v1)

Authors: Duo Liu, Zhiquan Tan, Linglan Zhao, Zhongqiang Zhang, Xiangzhong Fang, Weiran Huang

Generalized Category Discovery (GCD) aims to identify unlabeled samples by
leveraging the base knowledge from labeled ones, where the unlabeled set
consists of both base and novel classes. Since clustering methods are
time-consuming at inference, parametric-based approaches have become more
popular. However, recent parametric-based methods suffer from inferior base
discrimination due to unreliable self-supervision. To address this issue, we
propose a Reciprocal Learning Framework (RLF) that introduces an auxiliary
branch devoted to base classification. During training, the main branch filters
the pseudo-base samples to the auxiliary branch. In response, the auxiliary
branch provides more reliable soft labels for the main branch, leading to a
virtuous cycle. Furthermore, we introduce Class-wise Distribution
Regularization (CDR) to mitigate the learning bias towards base classes. CDR
essentially increases the prediction confidence of the unlabeled data and
boosts the novel class performance. Combined with both components, our proposed
method, RLCD, achieves superior performance in all classes with negligible
extra computation. Comprehensive experiments across seven GCD datasets validate
its superiority. Our codes are available at https://github.com/APORduo/RLCD.

### 2. [RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](http://arxiv.org/pdf/2506.02354v1)

Authors: Junjie Li, Nan Zhang, Xiaoyang Qu, Kai Lu, Guokuan Li, Jiguang Wan, Jianzong Wang

Object Navigation (ObjectNav) is a fundamental task in embodied artificial
intelligence. Although significant progress has been made in semantic map
construction and target direction prediction in current research, redundant
exploration and exploration failures remain inevitable. A critical but
underexplored direction is the timely termination of exploration to overcome
these challenges. We observe a diminishing marginal effect between exploration
steps and exploration rates and analyze the cost-benefit relationship of
exploration. Inspired by this, we propose RATE-Nav, a Region-Aware
Termination-Enhanced method. It includes a geometric predictive region
segmentation algorithm and region-Based exploration estimation algorithm for
exploration rate calculation. By leveraging the visual question answering
capabilities of visual language models (VLMs) and exploration rates enables
efficient termination.RATE-Nav achieves a success rate of 67.8% and an SPL of
31.3% on the HM3D dataset. And on the more challenging MP3D dataset, RATE-Nav
shows approximately 10% improvement over previous zero-shot methods.

### 3. [InterRVOS: Interaction-aware Referring Video Object Segmentation](http://arxiv.org/pdf/2506.02356v1)

Authors: Woojeong Jin, Seongchan Kim, Seungryong Kim

Referring video object segmentation aims to segment the object in a video
corresponding to a given natural language expression. While prior works have
explored various referring scenarios, including motion-centric or
multi-instance expressions, most approaches still focus on localizing a single
target object in isolation. However, in comprehensive video understanding, an
object's role is often defined by its interactions with other entities, which
are largely overlooked in existing datasets and models. In this work, we
introduce Interaction-aware referring video object sgementation (InterRVOS), a
new task that requires segmenting both actor and target entities involved in an
interaction. Each interactoin is described through a pair of complementary
expressions from different semantic perspectives, enabling fine-grained
modeling of inter-object relationships. To tackle this task, we propose
InterRVOS-8K, the large-scale and automatically constructed dataset containing
diverse interaction-aware expressions with corresponding masks, including
challenging cases such as motion-only multi-instance expressions. We also
present a baseline architecture, ReVIOSa, designed to handle actor-target
segmentation from a single expression, achieving strong performance in both
standard and interaction-focused settings. Furthermore, we introduce an
actor-target-aware evalaution setting that enables a more targeted assessment
of interaction understanding. Experimental results demonstrate that our
approach outperforms prior methods in modeling complex object interactions for
referring video object segmentation task, establishing a strong foundation for
future research in interaction-centric video understanding. Our project page is
available at
\href{https://cvlab-kaist.github.io/InterRVOS}{https://cvlab-kaist.github.io/InterRVOS}.

### 4. [RoadFormer : Local-Global Feature Fusion for Road Surface Classification in Autonomous Driving](http://arxiv.org/pdf/2506.02358v1)

Authors: Tianze Wang, Zhang Zhang, Chao Sun

The classification of the type of road surface (RSC) aims to utilize pavement
features to identify the roughness, wet and dry conditions, and material
information of the road surface. Due to its ability to effectively enhance road
safety and traffic management, it has received widespread attention in recent
years. In autonomous driving, accurate RSC allows vehicles to better understand
the road environment, adjust driving strategies, and ensure a safer and more
efficient driving experience. For a long time, vision-based RSC has been
favored. However, existing visual classification methods have overlooked the
exploration of fine-grained classification of pavement types (such as similar
pavement textures). In this work, we propose a pure vision-based fine-grained
RSC method for autonomous driving scenarios, which fuses local and global
feature information through the stacking of convolutional and transformer
modules. We further explore the stacking strategies of local and global feature
extraction modules to find the optimal feature extraction strategy. In
addition, since fine-grained tasks also face the challenge of relatively large
intra-class differences and relatively small inter-class differences, we
propose a Foreground-Background Module (FBM) that effectively extracts
fine-grained context features of the pavement, enhancing the classification
ability for complex pavements. Experiments conducted on a large-scale pavement
dataset containing one million samples and a simplified dataset reorganized
from this dataset achieved Top-1 classification accuracies of 92.52% and
96.50%, respectively, improving by 5.69% to 12.84% compared to SOTA methods.
These results demonstrate that RoadFormer outperforms existing methods in RSC
tasks, providing significant progress in improving the reliability of pavement
perception in autonomous driving systems.

### 5. [Auto-Labeling Data for Object Detection](http://arxiv.org/pdf/2506.02359v1)

Authors: Brent A. Griffin, Manushree Gangwar, Jacob Sela, Jason J. Corso

Great labels make great models. However, traditional labeling approaches for
tasks like object detection have substantial costs at scale. Furthermore,
alternatives to fully-supervised object detection either lose functionality or
require larger models with prohibitive computational costs for inference at
scale. To that end, this paper addresses the problem of training standard
object detection models without any ground truth labels. Instead, we configure
previously-trained vision-language foundation models to generate
application-specific pseudo "ground truth" labels. These auto-generated labels
directly integrate with existing model training frameworks, and we subsequently
train lightweight detection models that are computationally efficient. In this
way, we avoid the costs of traditional labeling, leverage the knowledge of
vision-language models, and keep the efficiency of lightweight models for
practical application. We perform exhaustive experiments across multiple
labeling configurations, downstream inference models, and datasets to establish
best practices and set an extensive auto-labeling benchmark. From our results,
we find that our approach is a viable alternative to standard labeling in that
it maintains competitive performance on multiple datasets and substantially
reduces labeling time and costs.

### 6. [A TRPCA-Inspired Deep Unfolding Network for Hyperspectral Image Denoising via Thresholded t-SVD and Top-K Sparse Transformer](http://arxiv.org/pdf/2506.02364v1)

Authors: Liang Li, Jianli Zhao, Sheng Fang, Siyu Chen, Hui Sun

Hyperspectral images (HSIs) are often degraded by complex mixed noise during
acquisition and transmission, making effective denoising essential for
subsequent analysis. Recent hybrid approaches that bridge model-driven and
data-driven paradigms have shown great promise. However, most of these
approaches lack effective alternation between different priors or modules,
resulting in loosely coupled regularization and insufficient exploitation of
their complementary strengths. Inspired by tensor robust principal component
analysis (TRPCA), we propose a novel deep unfolding network (DU-TRPCA) that
enforces stage-wise alternation between two tightly integrated modules:
low-rank and sparse. The low-rank module employs thresholded tensor singular
value decomposition (t-SVD), providing a widely adopted convex surrogate for
tensor low-rankness and has been demonstrated to effectively capture the global
spatial-spectral structure of HSIs. The Top-K sparse transformer module
adaptively imposes sparse constraints, directly matching the sparse
regularization in TRPCA and enabling effective removal of localized outliers
and complex noise. This tightly coupled architecture preserves the stage-wise
alternation between low-rank approximation and sparse refinement inherent in
TRPCA, while enhancing representational capacity through attention mechanisms.
Extensive experiments on synthetic and real-world HSIs demonstrate that
DU-TRPCA surpasses state-of-the-art methods under severe mixed noise, while
offering interpretability benefits and stable denoising dynamics inspired by
iterative optimization. Code is available at
https://github.com/liangli97/TRPCA-Deep-Unfolding-HSI-Denoising.

### 7. [ViTNF: Leveraging Neural Fields to Boost Vision Transformers in Generalized Category Discovery](http://arxiv.org/pdf/2506.02367v1)

Authors: Jiayi Su, Dequan Jin

Generalized category discovery (GCD) is a highly popular task in open-world
recognition, aiming to identify unknown class samples using known class data.
By leveraging pre-training, meta-training, and fine-tuning, ViT achieves
excellent few-shot learning capabilities. Its MLP head is a feedforward
network, trained synchronously with the entire network in the same process,
increasing the training cost and difficulty without fully leveraging the power
of the feature extractor. This paper proposes a new architecture by replacing
the MLP head with a neural field-based one. We first present a new static
neural field function to describe the activity distribution of the neural field
and then use two static neural field functions to build an efficient few-shot
classifier. This neural field-based (NF) classifier consists of two coupled
static neural fields. It stores the feature information of support samples by
its elementary field, the known categories by its high-level field, and the
category information of support samples by its cross-field connections. We
replace the MLP head with the proposed NF classifier, resulting in a novel
architecture ViTNF, and simplify the three-stage training mode by pre-training
the feature extractor on source tasks and training the NF classifier with
support samples in meta-testing separately, significantly reducing ViT's demand
for training samples and the difficulty of model training. To enhance the
model's capability in identifying new categories, we provide an effective
algorithm to determine the lateral interaction scale of the elementary field.
Experimental results demonstrate that our model surpasses existing
state-of-the-art methods on CIFAR-100, ImageNet-100, CUB-200, and Standard
Cars, achieving dramatic accuracy improvements of 19\% and 16\% in new and all
classes, respectively, indicating a notable advantage in GCD.

### 8. [RRCANet: Recurrent Reusable-Convolution Attention Network for Infrared Small Target Detection](http://arxiv.org/pdf/2506.02393v1)

Authors: Yongxian Liu, Boyang Li, Ting Liu, Zaiping Lin, Wei An

Infrared small target detection is a challenging task due to its unique
characteristics (e.g., small, dim, shapeless and changeable). Recently
published CNN-based methods have achieved promising performance with heavy
feature extraction and fusion modules. To achieve efficient and effective
detection, we propose a recurrent reusable-convolution attention network
(RRCA-Net) for infrared small target detection. Specifically, RRCA-Net
incorporates reusable-convolution block (RuCB) in a recurrent manner without
introducing extra parameters. With the help of the repetitive iteration in
RuCB, the high-level information of small targets in the deep layers can be
well maintained and further refined. Then, a dual interactive attention
aggregation module (DIAAM) is proposed to promote the mutual enhancement and
fusion of refined information. In this way, RRCA-Net can both achieve
high-level feature refinement and enhance the correlation of contextual
information between adjacent layers. Moreover, to achieve steady convergence,
we design a target characteristic inspired loss function (DpT-k loss) by
integrating physical and mathematical constraints. Experimental results on
three benchmark datasets (e.g. NUAA-SIRST, IRSTD-1k, DenseSIRST) demonstrate
that our RRCA-Net can achieve comparable performance to the state-of-the-art
methods while maintaining a small number of parameters, and act as a plug and
play module to introduce consistent performance improvement for several popular
IRSTD methods. Our code will be available at https://github.com/yongxianLiu/
soon.

### 9. [The Devil is in the Darkness: Diffusion-Based Nighttime Dehazing Anchored in Brightness Perception](http://arxiv.org/pdf/2506.02395v1)

Authors: Xiaofeng Cong, Yu-Xin Zhang, Haoran Wei, Yeying Jin, Junming Hou, Jie Gui, Jing Zhang, Dacheng Tao

While nighttime image dehazing has been extensively studied, converting
nighttime hazy images to daytime-equivalent brightness remains largely
unaddressed. Existing methods face two critical limitations: (1) datasets
overlook the brightness relationship between day and night, resulting in the
brightness mapping being inconsistent with the real world during image
synthesis; and (2) models do not explicitly incorporate daytime brightness
knowledge, limiting their ability to reconstruct realistic lighting. To address
these challenges, we introduce the Diffusion-Based Nighttime Dehazing (DiffND)
framework, which excels in both data synthesis and lighting reconstruction. Our
approach starts with a data synthesis pipeline that simulates severe
distortions while enforcing brightness consistency between synthetic and
real-world scenes, providing a strong foundation for learning night-to-day
brightness mapping. Next, we propose a restoration model that integrates a
pre-trained diffusion model guided by a brightness perception network. This
design harnesses the diffusion model's generative ability while adapting it to
nighttime dehazing through brightness-aware optimization. Experiments validate
our dataset's utility and the model's superior performance in joint haze
removal and brightness mapping.

### 10. [Towards Explicit Geometry-Reflectance Collaboration for Generalized LiDAR Segmentation in Adverse Weather](http://arxiv.org/pdf/2506.02396v1)

Authors: Longyu Yang, Ping Hu, Shangbo Yuan, Lu Zhang, Jun Liu, Hengtao Shen, Xiaofeng Zhu

Existing LiDAR semantic segmentation models often suffer from decreased
accuracy when exposed to adverse weather conditions. Recent methods addressing
this issue focus on enhancing training data through weather simulation or
universal augmentation techniques. However, few works have studied the negative
impacts caused by the heterogeneous domain shifts in the geometric structure
and reflectance intensity of point clouds. In this paper, we delve into this
challenge and address it with a novel Geometry-Reflectance Collaboration (GRC)
framework that explicitly separates feature extraction for geometry and
reflectance. Specifically, GRC employs a dual-branch architecture designed to
independently process geometric and reflectance features initially, thereby
capitalizing on their distinct characteristic. Then, GRC adopts a robust
multi-level feature collaboration module to suppress redundant and unreliable
information from both branches. Consequently, without complex simulation or
augmentation, our method effectively extracts intrinsic information about the
scene while suppressing interference, thus achieving better robustness and
generalization in adverse weather conditions. We demonstrate the effectiveness
of GRC through comprehensive experiments on challenging benchmarks, showing
that our method outperforms previous approaches and establishes new
state-of-the-art results.

### Computers and Society

### 1. [A Hierarchical Integer Linear Programming Approach for Optimizing Team Formation in Education](http://arxiv.org/pdf/2506.02756v1)

Authors: Aaron Kessler, Tim Scheiber, Heinz Schmitz, Ioanna Lykourentzou

Teamwork is integral to higher education, fostering students' interpersonal
skills, improving learning outcomes, and preparing them for professional
collaboration later in their careers. While team formation has traditionally
been managed by humans, either instructors or students, algorithmic approaches
have recently emerged to optimize this process. However, existing algorithmic
team formation methods often focus on expert teams, overlook agency in choosing
one's teammates, and are limited to a single team formation setting. These
limitations make them less suitable for education, where no student can be left
out, student agency is crucial for motivation, and team formation needs vary
across courses and programs. In this paper, we introduce the EDUCATIONAL TEAM
FORMATION problem (EDU-TF), a partitioning optimization problem model tailored
to the unique needs of education, integrating both teacher and student
requirements. To solve EDU-TF, we propose a modular optimization approach, one
of the first to allow the flexible adjustment of objectives according to
educational needs, enhancing the method's applicability across various
classroom settings rather than just research environments. Results from
evaluating ten strategies derived from our model on real-world university
datasets indicate that our approach outperforms heuristic teacher-assigned
teams by better accommodating student preferences. Our study contributes a new
modular approach to partition-based algorithmic team formation and provides
valuable insights for future research on team formation in educational
settings.

### 2. [Evaluating LLM Agent Adherence to Hierarchical Safety Principles: A Lightweight Benchmark for Probing Foundational Controllability Components](http://arxiv.org/pdf/2506.02357v1)

Authors: Ram Potham

Credible safety plans for advanced AI development require methods to verify
agent behavior and detect potential control deficiencies early. A fundamental
aspect is ensuring agents adhere to safety-critical principles, especially when
these conflict with operational goals. Failure to prioritize such principles
indicates a potential basic control failure. This paper introduces a
lightweight, interpretable benchmark methodology using a simple grid world to
evaluate an LLM agent's ability to uphold a predefined, high-level safety
principle (e.g., "never enter hazardous zones") when faced with conflicting
lower-level task instructions. We probe whether the agent reliably prioritizes
the inviolable directive, testing a foundational controllability aspect of
LLMs. This pilot study demonstrates the methodology's feasibility, offers
preliminary insights into agent behavior under principle conflict, and
discusses how such benchmarks can contribute empirical evidence for assessing
controllability. We argue that evaluating adherence to hierarchical principles
is a crucial early step in understanding our capacity to build governable AI
systems.

### 3. [Data Leakage and Deceptive Performance: A Critical Examination of Credit Card Fraud Detection Methodologies](http://arxiv.org/pdf/2506.02703v1)

Authors: Khizar Hayat, Baptiste Magnier

This study critically examines the methodological rigor in credit card fraud
detection research, revealing how fundamental evaluation flaws can overshadow
algorithmic sophistication. Through deliberate experimentation with improper
evaluation protocols, we demonstrate that even simple models can achieve
deceptively impressive results when basic methodological principles are
violated. Our analysis identifies four critical issues plaguing current
approaches: (1) pervasive data leakage from improper preprocessing sequences,
(2) intentional vagueness in methodological reporting, (3) inadequate temporal
validation for transaction data, and (4) metric manipulation through recall
optimization at precision's expense. We present a case study showing how a
minimal neural network architecture with data leakage outperforms many
sophisticated methods reported in literature, achieving 99.9\% recall despite
fundamental evaluation flaws. These findings underscore that proper evaluation
methodology matters more than model complexity in fraud detection research. The
study serves as a cautionary example of how methodological rigor must precede
architectural sophistication, with implications for improving research
practices across machine learning applications.

### 4. [TestAgent: An Adaptive and Intelligent Expert for Human Assessment](http://arxiv.org/pdf/2506.03032v1)

Authors: Junhao Yu, Yan Zhuang, YuXuan Sun, Weibo Gao, Qi Liu, Mingyue Cheng, Zhenya Huang, Enhong Chen

Accurately assessing internal human states is key to understanding
preferences, offering personalized services, and identifying challenges in
real-world applications. Originating from psychometrics, adaptive testing has
become the mainstream method for human measurement and has now been widely
applied in education, healthcare, sports, and sociology. It customizes
assessments by selecting the fewest test questions . However, current adaptive
testing methods face several challenges. The mechanized nature of most
algorithms leads to guessing behavior and difficulties with open-ended
questions. Additionally, subjective assessments suffer from noisy response data
and coarse-grained test outputs, further limiting their effectiveness. To move
closer to an ideal adaptive testing process, we propose TestAgent, a large
language model (LLM)-powered agent designed to enhance adaptive testing through
interactive engagement. This is the first application of LLMs in adaptive
testing. TestAgent supports personalized question selection, captures
test-takers' responses and anomalies, and provides precise outcomes through
dynamic, conversational interactions. Experiments on psychological,
educational, and lifestyle assessments show our approach achieves more accurate
results with 20% fewer questions than state-of-the-art baselines, and testers
preferred it in speed, smoothness, and other dimensions.

### 5. [Corrigibility as a Singular Target: A Vision for Inherently Reliable Foundation Models](http://arxiv.org/pdf/2506.03056v1)

Authors: Ram Potham, Max Harms

Foundation models (FMs) face a critical safety challenge: as capabilities
scale, instrumental convergence drives default trajectories toward loss of
human control, potentially culminating in existential catastrophe. Current
alignment approaches struggle with value specification complexity and fail to
address emergent power-seeking behaviors. We propose "Corrigibility as a
Singular Target" (CAST)-designing FMs whose overriding objective is empowering
designated human principals to guide, correct, and control them. This paradigm
shift from static value-loading to dynamic human empowerment transforms
instrumental drives: self-preservation serves only to maintain the principal's
control; goal modification becomes facilitating principal guidance. We present
a comprehensive empirical research agenda spanning training methodologies
(RLAIF, SFT, synthetic data generation), scalability testing across model
sizes, and demonstrations of controlled instructability. Our vision: FMs that
become increasingly responsive to human guidance as capabilities grow, offering
a path to beneficial AI that remains as tool-like as possible, rather than
supplanting human judgment. This addresses the core alignment problem at its
source, preventing the default trajectory toward misaligned instrumental
convergence.

### 6. [Causal Explainability of Machine Learning in Heart Failure Prediction from Electronic Health Records](http://arxiv.org/pdf/2506.03068v1)

Authors: Yina Hou, Shourav B. Rabbani, Liang Hong, Norou Diawara, Manar D. Samad

The importance of clinical variables in the prognosis of the disease is
explained using statistical correlation or machine learning (ML). However, the
predictive importance of these variables may not represent their causal
relationships with diseases. This paper uses clinical variables from a heart
failure (HF) patient cohort to investigate the causal explainability of
important variables obtained in statistical and ML contexts. Due to inherent
regression modeling, popular causal discovery methods strictly assume that the
cause and effect variables are numerical and continuous. This paper proposes a
new computational framework to enable causal structure discovery (CSD) and
score the causal strength of mixed-type (categorical, numerical, binary)
clinical variables for binary disease outcomes. In HF classification, we
investigate the association between the importance rank order of three feature
types: correlated features, features important for ML predictions, and causal
features. Our results demonstrate that CSD modeling for nonlinear causal
relationships is more meaningful than its linear counterparts. Feature
importance obtained from nonlinear classifiers (e.g., gradient-boosting trees)
strongly correlates with the causal strength of variables without
differentiating cause and effect variables. Correlated variables can be causal
for HF, but they are rarely identified as effect variables. These results can
be used to add the causal explanation of variables important for ML-based
prediction modeling.

### 7. [Designing Algorithmic Delegates: The Role of Indistinguishability in Human-AI Handoff](http://arxiv.org/pdf/2506.03102v1)

Authors: Sophie Greenwood, Karen Levy, Solon Barocas, Hoda Heidari, Jon Kleinberg

As AI technologies improve, people are increasingly willing to delegate tasks
to AI agents. In many cases, the human decision-maker chooses whether to
delegate to an AI agent based on properties of the specific instance of the
decision-making problem they are facing. Since humans typically lack full
awareness of all the factors relevant to this choice for a given
decision-making instance, they perform a kind of categorization by treating
indistinguishable instances -- those that have the same observable features --
as the same. In this paper, we define the problem of designing the optimal
algorithmic delegate in the presence of categories. This is an important
dimension in the design of algorithms to work with humans, since we show that
the optimal delegate can be an arbitrarily better teammate than the optimal
standalone algorithmic agent. The solution to this optimal delegation problem
is not obvious: we discover that this problem is fundamentally combinatorial,
and illustrate the complex relationship between the optimal design and the
properties of the decision-making task even in simple settings. Indeed, we show
that finding the optimal delegate is computationally hard in general. However,
we are able to find efficient algorithms for producing the optimal delegate in
several broad cases of the problem, including when the optimal action may be
decomposed into functions of features observed by the human and the algorithm.
Finally, we run computational experiments to simulate a designer updating an
algorithmic delegate over time to be optimized for when it is actually adopted
by users, and show that while this process does not recover the optimal
delegate in general, the resulting delegate often performs quite well.

### 8. [MAEBE: Multi-Agent Emergent Behavior Framework](http://arxiv.org/pdf/2506.03053v1)

Authors: Sinem Erisken, Timothy Gothard, Martin Leitgab, Ram Potham

Traditional AI safety evaluations on isolated LLMs are insufficient as
multi-agent AI ensembles become prevalent, introducing novel emergent risks.
This paper introduces the Multi-Agent Emergent Behavior Evaluation (MAEBE)
framework to systematically assess such risks. Using MAEBE with the Greatest
Good Benchmark (and a novel double-inversion question technique), we
demonstrate that: (1) LLM moral preferences, particularly for Instrumental
Harm, are surprisingly brittle and shift significantly with question framing,
both in single agents and ensembles. (2) The moral reasoning of LLM ensembles
is not directly predictable from isolated agent behavior due to emergent group
dynamics. (3) Specifically, ensembles exhibit phenomena like peer pressure
influencing convergence, even when guided by a supervisor, highlighting
distinct safety and alignment challenges. Our findings underscore the necessity
of evaluating AI systems in their interactive, multi-agent contexts.

### Databases

### 1. [In-context Clustering-based Entity Resolution with Large Language Models: A Design Space Exploration](http://arxiv.org/pdf/2506.02509v1)

Authors: Jiajie Fu, Haitong Tang, Arijit Khan, Sharad Mehrotra, Xiangyu Ke, Yunjun Gao

Entity Resolution (ER) is a fundamental data quality improvement task that
identifies and links records referring to the same real-world entity.
Traditional ER approaches often rely on pairwise comparisons, which can be
costly in terms of time and monetary resources, especially with large datasets.
Recently, Large Language Models (LLMs) have shown promising results in ER
tasks. However, existing methods typically focus on pairwise matching, missing
the potential of LLMs to perform clustering directly in a more cost-effective
and scalable manner. In this paper, we propose a novel in-context clustering
approach for ER, where LLMs are used to cluster records directly, reducing both
time complexity and monetary costs. We systematically investigate the design
space for in-context clustering, analyzing the impact of factors such as set
size, diversity, variation, and ordering of records on clustering performance.
Based on these insights, we develop LLM-CER (LLM-powered Clustering-based ER),
which achieves high-quality ER results while minimizing LLM API calls. Our
approach addresses key challenges, including efficient cluster merging and LLM
hallucination, providing a scalable and effective solution for ER. Extensive
experiments on nine real-world datasets demonstrate that our method
significantly improves result quality, achieving up to 150% higher accuracy,
10% increase in the F-measure, and reducing API calls by up to 5 times, while
maintaining comparable monetary cost to the most cost-effective baseline.

### 2. [PandasBench: A Benchmark for the Pandas API](http://arxiv.org/pdf/2506.02345v1)

Authors: Alex Broihier, Stefanos Baziotis, Daniel Kang, Charith Mendis

The Pandas API has been central to the success of pandas and its
alternatives. Despite its importance, there is no benchmark for it, and we
argue that we cannot repurpose existing benchmarks (from other domains) for the
Pandas API.
  In this paper, we introduce requirements that are necessary for a Pandas API
enchmark, and present the first benchmark that fulfills them: PandasBench. We
argue that it should evaluate the real-world coverage of a technique. Yet,
real-world coverage is not sufficient for a useful benchmark, and so we also:
cleaned it from irrelevant code, adapted it for benchmark usage, and introduced
input scaling. We claim that uniform scaling used in other benchmarks (e.g.,
TPC-H) is too coarse-grained for PandasBench, and use a non-uniform scaling
scheme. PandasBench is the largest Pandas API benchmark to date, with 102
notebooks and 3,721 cells.
  We used PandasBench to evaluate Modin, Dask, Koalas, and Dias. This is the
largest-scale evaluation of all these techniques to date. Prior works report
significant speedups using constrained benchmarks, but we show that on a larger
benchmark with real-world code, the most notebooks that got a speedup were
8/102 (~8%) for Modin, and 0 for both Koalas and Dask. Dias showed speedups in
up to 55 notebooks (~54%), but it rewrites code incorrectly in certain cases,
which had not been observed in prior work. Second, we identified many failures:
Modin runs only 72/102 (~70%) notebooks, Dask 4 (~4%), Koalas 10 (~10%), and
Dias 97 (95%).

### 3. [A Learned Cost Model-based Cross-engine Optimizer for SQL Workloads](http://arxiv.org/pdf/2506.02802v1)

Authors: András Strausz, Niels Pardon, Ioana Giurgiu

Lakehouse systems enable the same data to be queried with multiple execution
engines. However, selecting the engine best suited to run a SQL query still
requires a priori knowledge of the query computational requirements and an
engine capability, a complex and manual task that only becomes more difficult
with the emergence of new engines and workloads. In this paper, we address this
limitation by proposing a cross-engine optimizer that can automate engine
selection for diverse SQL queries through a learned cost model. Optimized with
hints, a query plan is used for query cost prediction and routing. Cost
prediction is formulated as a multi-task learning problem, and multiple
predictor heads, corresponding to different engines and provisionings, are used
in the model architecture. This eliminates the need to train engine-specific
models and allows the flexible addition of new engines at a minimal fine-tuning
cost. Results on various databases and engines show that using a query
optimized logical plan for cost estimation decreases the average Q-error by
even 12.6% over using unoptimized plans as input. Moreover, the proposed
cross-engine optimizer reduces the total workload runtime by up to 25.2% in a
zero-shot setting and 30.4% in a few-shot setting when compared to random
routing.

### 4. [Process Mining on Distributed Data Sources](http://arxiv.org/pdf/2506.02830v1)

Authors: Maximilian Weisenseel, Julia Andersen, Samira Akili, Christian Imenkamp, Hendrik Reiter, Christoffer Rubensson, Wilhelm Hasselbring, Olaf Landsiedel, Xixi Lu, Jan Mendling, Florian Tschorsch, Matthias Weidlich, Agnes Koschmider

Major domains such as logistics, healthcare, and smart cities increasingly
rely on sensor technologies and distributed infrastructures to monitor complex
processes in real time. These developments are transforming the data landscape
from discrete, structured records stored in centralized systems to continuous,
fine-grained, and heterogeneous event streams collected across distributed
environments. As a result, traditional process mining techniques, which assume
centralized event logs from enterprise systems, are no longer sufficient. In
this paper, we discuss the conceptual and methodological foundations for this
emerging field. We identify three key shifts: from offline to online analysis,
from centralized to distributed computing, and from event logs to sensor data.
These shifts challenge traditional assumptions about process data and call for
new approaches that integrate infrastructure, data, and user perspectives. To
this end, we define a research agenda that addresses six interconnected fields,
each spanning multiple system dimensions. We advocate a principled methodology
grounded in algorithm engineering, combining formal modeling with empirical
evaluation. This approach enables the development of scalable, privacy-aware,
and user-centric process mining techniques suitable for distributed
environments. Our synthesis provides a roadmap for advancing process mining
beyond its classical setting, toward a more responsive and decentralized
paradigm of process intelligence.

### Distributed, Parallel, and Cluster Computing

### 1. [DiOMP-Offloading: Toward Portable Distributed Heterogeneous OpenMP](http://arxiv.org/pdf/2506.02486v1)

Authors: Baodi Shan, Mauricio Arayr-Polo, Barbara Chapman

As core counts and heterogeneity rise in HPC, traditional hybrid programming
models face challenges in managing distributed GPU memory and ensuring
portability. This paper presents DiOMP, a distributed OpenMP framework that
unifies OpenMP target offloading with the Partitioned Global Address Space
(PGAS) model. Built atop LLVM/OpenMP and using GASNet-EX or GPI-2 for
communication, DiOMP transparently handles global memory, supporting both
symmetric and asymmetric GPU allocations. It leverages OMPCCL, a portable
collective communication layer compatible with vendor libraries. DiOMP
simplifies programming by abstracting device memory and communication,
achieving superior scalability and programmability over traditional approaches.
Evaluations on NVIDIA A100, Grace Hopper, and AMD MI250X show improved
performance in micro-benchmarks and applications like matrix multiplication and
Minimod, highlighting DiOMP's potential for scalable, portable, and efficient
heterogeneous computing.

### 2. [Distributedness based scheduling](http://arxiv.org/pdf/2506.02581v1)

Authors: Paritosh Ranjan, Surajit Majumder, Prodip Roy, Bhuban Padhan

Efficient utilization of computing resources in a Kubernetes cluster is often
constrained by the uneven distribution of pods with similar usage patterns.
This paper presents a novel scheduling strategy designed to optimize the
distributedness of Kubernetes resources based on their usage magnitude and
patterns across CPU, memory, network, and storage. By categorizing resource
usage into labels such as "cpu high spike" or "memory medium always," and
applying these to deployed pods, the system calculates the variance or
distributedness factor of similar resource types across cluster nodes. A lower
variance indicates a more balanced distribution. The Kubernetes scheduler is
enhanced to consider this factor during scheduling decisions, placing new pods
on nodes that minimize resource clustering. Furthermore, the approach supports
redistribution of existing pods through simulated scheduling to improve
balance. This method is adaptable at the cluster, namespace, or application
level and is integrated within the standard Kubernetes scheduler, providing a
scalable, label-driven mechanism to improve overall resource efficiency in
cloud-native environments.

### 3. [Adaptive Configuration Selection for Multi-Model Inference Pipelines in Edge Computing](http://arxiv.org/pdf/2506.02814v1)

Authors: Jinhao Sheng, Zhiqing Tang, Jianxiong Guo, Tian Wang

The growing demand for real-time processing tasks is driving the need for
multi-model inference pipelines on edge devices. However, cost-effectively
deploying these pipelines while optimizing Quality of Service (QoS) and costs
poses significant challenges. Existing solutions often neglect device resource
constraints, focusing mainly on inference accuracy and cost efficiency. To
address this, we develop a framework for configuring multi-model inference
pipelines. Specifically: 1) We model the decision-making problem by considering
the pipeline's QoS, costs, and device resource limitations. 2) We create a
feature extraction module using residual networks and a load prediction model
based on Long Short-Term Memory (LSTM) to gather comprehensive node and
pipeline status information. Then, we implement a Reinforcement Learning (RL)
algorithm based on policy gradients for online configuration decisions. 3)
Experiments conducted in a real Kubernetes cluster show that our approach
significantly improve QoS while reducing costs and shorten decision-making time
for complex pipelines compared to baseline algorithms.

### 4. [Memory-Efficient Split Federated Learning for LLM Fine-Tuning on Heterogeneous Mobile Devices](http://arxiv.org/pdf/2506.02940v1)

Authors: Xiaopei Chen, Liang Li, Fei Ji, Wen Wu

In this paper, we propose an edge-assisted split federated learning framework
to facilitate large language model (LLM) fine-tuning on heterogeneous mobile
devices while alleviating memory pressures on both mobile devices and the edge
server. Specifically, mobile devices perform low-rank adaptation (LoRA)
fine-tuning on only a subset of lower layers of the pre-trained LLM, tailored
to their individual capacities. On the server, a full LLM is maintained, and
the corresponding LoRA modules are selectively fine-tuned in a sequential
manner for each device. To further enhance training efficiency, we propose a
server-side training scheduling method that optimizes the processing order of
devices for accelerating fine-tuning. Extensive experiments demonstrate that
compared to the baselines, our scheme can reduce 79\% memory footprint and 6\%
training time while achieving comparable performance.

### 5. [Reconciling Hessian-Informed Acceleration and Scalar-Only Communication for Efficient Federated Zeroth-Order Fine-Tuning](http://arxiv.org/pdf/2506.02370v1)

Authors: Zhe Li, Bicheng Ying, Zidong Liu, Chaosheng Dong, Haibo Yang

Recent dimension-free communication frameworks in Federated Learning (FL),
such as DeComFL, significantly reduce per-round communication by transmitting
only scalars via zeroth-order stochastic gradient descent (ZO-SGD). This method
is particularly advantageous for federated fine-tuning of Large Language Models
(LLMs). Yet, the high variance in ZO gradient estimation typically leads to
slow convergence. Although leveraging Hessian information is known to enhance
optimization speed, integrating this into FL presents significant challenges.
These include clients' restrictions on local data and the critical need to
maintain the dimension-free communication property. To overcome this
limitation, we first introduce a generalized scalar-only communication FL
framework that decouples dimension-free communication from standard ZO-SGD,
enabling the integration of more advanced optimization strategies. Building on
this framework, we propose HiSo, a fast federated fine-tuning method via
Hessian-informed zeroth-order optimization and Scalar-only communication.
Specifically, it leverages global curvature information to accelerate
convergence while preserving the same minimal communication cost per round.
Theoretically, we establish convergence guarantees that are independent of the
global Lipschitz constant, and further show that HiSo achieves faster rates
when the global Hessian exhibits a low effective rank -- a common phenomenon in
LLMs. Extensive experiments on benchmark datasets and LLM fine-tuning tasks
confirm that HiSo significantly outperforms existing ZO-based FL methods in
both convergence speed and communication efficiency.

### 6. [Enhancing Convergence, Privacy and Fairness for Wireless Personalized Federated Learning: Quantization-Assisted Min-Max Fair Scheduling](http://arxiv.org/pdf/2506.02422v1)

Authors: Xiyu Zhao, Qimei Cui, Ziqiang Du, Weicai Li, Xi Yu, Wei Ni, Ji Zhang, Xiaofeng Tao, Ping Zhang

Personalized federated learning (PFL) offers a solution to balancing
personalization and generalization by conducting federated learning (FL) to
guide personalized learning (PL). Little attention has been given to wireless
PFL (WPFL), where privacy concerns arise. Performance fairness of PL models is
another challenge resulting from communication bottlenecks in WPFL. This paper
exploits quantization errors to enhance the privacy of WPFL and proposes a
novel quantization-assisted Gaussian differential privacy (DP) mechanism. We
analyze the convergence upper bounds of individual PL models by considering the
impact of the mechanism (i.e., quantization errors and Gaussian DP noises) and
imperfect communication channels on the FL of WPFL. By minimizing the maximum
of the bounds, we design an optimal transmission scheduling strategy that
yields min-max fairness for WPFL with OFDMA interfaces. This is achieved by
revealing the nested structure of this problem to decouple it into subproblems
solved sequentially for the client selection, channel allocation, and power
control, and for the learning rates and PL-FL weighting coefficients.
Experiments validate our analysis and demonstrate that our approach
substantially outperforms alternative scheduling strategies by 87.08%, 16.21%,
and 38.37% in accuracy, the maximum test loss of participating clients, and
fairness (Jain's index), respectively.

### 7. [KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider](http://arxiv.org/pdf/2506.02634v1)

Authors: Jiahao Wang, Jinbo Han, Xingda Wei, Sijie Shen, Dingyan Zhang, Chenguang Fang, Rong Chen, Wenyuan Yu, Haibo Chen

Serving large language models (LLMs) is important for cloud providers, and
caching intermediate results (KV\$) after processing each request substantially
improves serving throughput and latency. However, there is limited
understanding of how LLM serving benefits from KV\$ caching, where system
design decisions like cache eviction policies are highly workload-dependent. In
this paper, we present the first systematic characterization of the KV\$
workload patterns from one of the leading LLM service providers. We draw
observations that were not covered by previous studies focusing on synthetic
workloads, including: KV\$ reuses are skewed across requests, where reuses
between single-turn requests are equally important as multi-turn requests; the
reuse time and probability are diverse considering all requests, but for a
specific request category, the pattern tends to be predictable; and the overall
cache size required for an ideal cache hit ratio is moderate. Based on the
characterization, we further propose a workload-aware cache eviction policy
that improves the serving performance under real-world traces, especially with
limited cache capacity.

### 8. [Usability Evaluation of Cloud for HPC Applications](http://arxiv.org/pdf/2506.02709v1)

Authors: Vanessa Sochat, Daniel Milroy, Abhik Sarkar, Aniruddha Marathe

The rise of AI and the economic dominance of cloud computing have created a
new nexus of innovation for high performance computing (HPC), which has a long
history of driving scientific discovery. In addition to performance needs,
scientific workflows increasingly demand capabilities of cloud environments:
portability, reproducibility, dynamism, and automation. As converged cloud
environments emerge, there is growing need to study their fit for HPC use
cases. Here we present a cross-platform usability study that assesses 11
different HPC proxy applications and benchmarks across three clouds (Microsoft
Azure, Amazon Web Services, and Google Cloud), six environments, and two
compute configurations (CPU and GPU) against on-premises HPC clusters at a
major center. We perform scaling tests of applications in all environments up
to 28,672 CPUs and 256 GPUs. We present methodology and results to guide future
study and provide a foundation to define best practices for running HPC
workloads in cloud.

### 9. [Rethinking Dynamic Networks and Heterogeneous Computing with Automatic Parallelization](http://arxiv.org/pdf/2506.02787v1)

Authors: Ruilong Wu, Xinjiao Li, Yisu Wang, Xinyu Chen, Dirk Kutscher

Hybrid parallelism techniques are essential for efficiently training large
language models (LLMs). Nevertheless, current automatic parallel planning
frameworks often overlook the simultaneous consideration of node heterogeneity
and dynamic network topology changes, limiting their effectiveness in practical
applications. In this paper, we address these limitations by modeling
heterogeneous nodes within dynamically changing network environments and
leveraging simulation-based strategies to determine optimal parallel
configurations. Our approach enables fine-grained workload allocation tailored
for heterogeneous nodes and complex network scenarios, achieving performance
competitive with state-of-the-art methods under regular and stable network
conditions. Additionally, we introduce a strategy pruning technique to rapidly
discard infeasible parallel configurations, substantially reducing the search
space and accelerating the search process through parallel execution within the
simulator. Preliminary evaluations confirm that our method notably enhances
training performance on heterogeneous nodes and demonstrates improved
adaptability in complex, dynamic scenarios such as cloud computing
environments.

### 10. [Overcoming Challenges of Partial Client Participation in Federated Learning : A Comprehensive Review](http://arxiv.org/pdf/2506.02887v1)

Authors: Mrinmay Sen, Shruti Aparna, Rohit Agarwal, Chalavadi Krishna Mohan

Federated Learning (FL) is a learning mechanism that falls under the
distributed training umbrella, which collaboratively trains a shared global
model without disclosing the raw data from different clients. This paper
presents an extensive survey on the impact of partial client participation in
federated learning. While much of the existing research focuses on addressing
issues such as generalization, robustness, and fairness caused by data
heterogeneity under the assumption of full client participation, limited
attention has been given to the practical and theoretical challenges arising
from partial client participation, which is common in real-world scenarios.
This survey provides an in-depth review of existing FL methods designed to cope
with partial client participation. We offer a comprehensive analysis supported
by theoretical insights and empirical findings, along with a structured
categorization of these methods, highlighting their respective advantages and
disadvantages.

### Discrete Mathematics

### 1. [Boolean-network simplification and rule fitting to unravel chemotherapy resistance in non-small cell lung cancer](http://arxiv.org/pdf/2506.02525v1)

Authors: Alonso Espinoza, Eric Goles, Marco Montalva-Medel

Boolean networks are powerful frameworks for capturing the logic of
gene-regulatory circuits, yet their combinatorial explosion hampers exhaustive
analyses. Here, we present a systematic reduction of a 31-node Boolean model
that describes cisplatin- and pemetrexed-resistance in non-small-cell lung
cancer to a compact 9-node core that exactly reproduces the original attractor
landscape. The streamlined network shrinks the state space by four orders of
magnitude, enabling rapid exploration of critical control points, rules fitting
and candidate therapeutic targets. Extensive synchronous and asynchronous
simulations confirm that the three clinically relevant steady states and their
basins of attraction are conserved and reflect resistance frequencies close to
those reported in clinical studies. The reduced model provides an accessible
scaffold for future mechanistic and drug-discovery studies.

### 2. [Bounded Discrete Bridges](http://arxiv.org/pdf/2506.02982v1)

Authors: Pierre Nicodeme

In 2010 Banderier and Nicodeme consider the height of bounded discrete
bridges and conclude to a limiting Rayleigh distribution. This result is
correct although their proof is partly erroneous. They make asymptotic
simplifications based upon dominance properties of the roots of the kernel of
the walk within a disk centered at the origin, but these dominance properties
apply only upon a positive real segment. However the very good agreement of
simulations with their asymptotic expansion of the probability distribution in
case of {\L}ukasiewicz bridges let us think that their proof could be
corrected. This is the scope of the present article which provides
  a proof using the dominance property only in its domain of validity. We also
consider the case of periodic walks, a topic not considered in
Banderier-Nicodeme2010. We limit ourselves to walks whose characteristic
polynomial decomposes over $\bC$ without repeated factors.

### Data Structures and Algorithms

### 1. [A Practical Linear Time Algorithm for Optimal Tree Decomposition of Halin Graphs](http://arxiv.org/pdf/2506.02346v1)

Authors: J. A. Alejandro-Soto, Joel Antonio Trejo-Sanchez, Carlos Segura

This work proposes \textsc{H-Td}, a practical linear-time algorithm for
computing an optimal-width tree decomposition of Halin graphs. Unlike
state-of-the-art methods based on reduction rules or separators, \textsc{H-Td}
exploits the structural properties of Halin graphs. Although two theoretical
linear-time algorithms exist that can be applied to graphs of treewidth three,
no practical implementation has been made publicly available. Furthermore,
extending reduction-based approaches to partial $k$-trees with $k > 3$ results
in increasingly complex rules that are challenging to implement. This motivates
the exploration of alternative strategies that leverage structural insights
specific to certain graph classes. Experimental validation against the winners
of the Parameterized Algorithms and Computational Experiments Challenge (PACE)
2017 and the treewidth library \texttt{libtw} demonstrates the advantage of
\textsc{H-Td} when the input is known to be a Halin graph.

### 2. [On the Inversion Modulo a Power of an Integer](http://arxiv.org/pdf/2506.02491v1)

Authors: Guangwu Xu, Yunxiao Tian, Bingxin Yang

Recently, Koc proposed a neat and efficient algorithm for computing $x =
a^{-1} \pmod {p^k}$ for a prime $p$ based on the exact solution of linear
equations using $p$-adic expansions. The algorithm requires only addition and
right shift per step. In this paper, we design an algorithm that computes $x =
a^{-1} \pmod {n^k}$ for any integer $n>1$. The algorithm has a motivation from
the schoolbook multiplication and achieves both efficiency and generality. The
greater flexibility of our algorithm is explored by utilizing the build-in
arithmetic of computer architecture, e.g., $n=2^{64}$, and experimental results
show significant improvements. This paper also contains some results on modular
inverse based on an alternative proof of Koc algorithm.

### 3. [Cartesian Forest Matching](http://arxiv.org/pdf/2506.02704v1)

Authors: Bastien Auvray, Julien David, Richard Groult, Thierry Lecroq

In this paper, we introduce the notion of Cartesian Forest, which generalizes
Cartesian Trees, in order to deal with partially ordered sequences. We show
that algorithms that solve both exact and approximate Cartesian Tree Matching
can be adapted to solve Cartesian Forest Matching in average linear time. We
adapt the notion of Cartesian Tree Signature to Cartesian Forests and show how
filters can be used to experimentally improve the algorithm for the exact
matching. We also show a one to one correspondence between Cartesian Forests
and Schr\"oder Trees.

### 4. [Upper bounds on the theta function of random graphs](http://arxiv.org/pdf/2506.02952v1)

Authors: Uriel Feige, Vadim Grinberg

The theta function of Lovasz is a graph parameter that can be computed up to
arbitrary precision in polynomial time. It plays a key role in algorithms that
approximate graph parameters such as maximum independent set, maximum clique
and chromatic number, or even compute them exactly in some models of random and
semi-random graphs. For Erdos-Renyi random $G_{n,1/2}$ graphs, the expected
value of the theta function is known to be at most $2\sqrt{n}$ and at least
$\sqrt{n}$. These bounds have not been improved in over 40 years.
  In this work, we introduce a new class of polynomial time computable graph
parameters, where every parameter in this class is an upper bound on the theta
function. We also present heuristic arguments for determining the expected
values of parameters from this class in random graphs. The values suggested by
these heuristic arguments are in agreement with results that we obtain
experimentally, by sampling graphs at random and computing the value of the
respective parameter. Based on parameters from this new class, we feel safe in
conjecturing that for $G_{n,1/2}$, the expected value of the theta function is
below $1.55 \sqrt{n}$. Our paper falls short of rigorously proving such an
upper bound, because our analysis makes use of unproven assumptions.

### 5. [The power of mediators: Price of anarchy and stability in Bayesian games with submodular social welfare](http://arxiv.org/pdf/2506.02655v1)

Authors: Kaito Fujii

This paper investigates the role of mediators in Bayesian games by examining
their impact on social welfare through the price of anarchy (PoA) and price of
stability (PoS). Mediators can communicate with players to guide them toward
equilibria of varying quality, and different communication protocols lead to a
variety of equilibrium concepts collectively known as Bayes (coarse) correlated
equilibria. To analyze these equilibrium concepts, we consider a general class
of Bayesian games with submodular social welfare, which naturally extends valid
utility games and their variant, basic utility games. These frameworks,
introduced by Vetta (2002), have been developed to analyze the social welfare
guarantees of equilibria in games such as competitive facility location,
influence maximization, and other resource allocation problems.
  We provide upper and lower bounds on the PoA and PoS for a broad class of
Bayes (coarse) correlated equilibria. Central to our analysis is the strategy
representability gap, which measures the multiplicative gap between the optimal
social welfare achievable with and without knowledge of other players' types.
For monotone submodular social welfare functions, we show that this gap is
$1-1/\mathrm{e}$ for independent priors and $\Theta(1/\sqrt{n})$ for correlated
priors, where $n$ is the number of players. These bounds directly lead to upper
and lower bounds on the PoA and PoS for various equilibrium concepts, while we
also derive improved bounds for specific concepts by developing smoothness
arguments. Notably, we identify a fundamental gap in the PoA and PoS across
different classes of Bayes correlated equilibria, highlighting essential
distinctions among these concepts.

### 6. [Labelling Data with Unknown References](http://arxiv.org/pdf/2506.03083v1)

Authors: Adrian de Wynter

An evaluator is trustworthy when there exists some agreed-upon way to measure
its performance as a labeller. The two ways to establish trustworthiness are
either by testing it, or by assuming the evaluator `knows' somehow the way to
label the corpus. However, if labelled references (e.g., a development set) are
unavailable, neither of these approaches work: the former requires the data,
and the latter is an assumption, not evidence. To address this, we introduce an
algorithm (the `No-Data Algorithm') by which to establish trust in an evaluator
without any existing references. Our algorithm works by successively posing
challenges to said evaluator. We show that this is sufficient to establish
trustworthiness w.h.p., in such a way that when the evaluator actually knows
the way to label the corpus, the No-Data Algorithm accepts its output; and,
conversely, flags untrustworthy evaluators when these are unable to prove it.
We present formal proofs of correctness and limited experiments.

### 7. [GPU-Parallelizable Randomized Sketch-and-Precondition for Linear Regression using Sparse Sign Sketches](http://arxiv.org/pdf/2506.03070v1)

Authors: Tyler Chen, Pradeep Niroula, Archan Ray, Pragna Subrahmanya, Marco Pistoia, Niraj Kumar

A litany of theoretical and numerical results have established the
sketch-and-precondition paradigm as a powerful approach to solving large linear
regression problems in standard computing environments. Perhaps surprisingly,
much less work has been done on understanding how sketch-and-precondition
performs on graphics processing unit (GPU) systems. We address this gap by
benchmarking an implementation of sketch-and-precondition based on sparse
sign-sketches on single and multi-GPU systems. In doing so, we describe a
novel, easily parallelized, rejection-sampling based method for generating
sparse sign sketches. Our approach, which is particularly well-suited for GPUs,
is easily adapted to a variety of computing environments. Taken as a whole, our
numerical experiments indicate that sketch-and-precondition with sparse sign
sketches is particularly well-suited for GPUs, and may be suitable for use in
black-box least-squares solvers.

### Emerging Technologies

### 1. [Probabilistic Online Event Downsampling](http://arxiv.org/pdf/2506.02547v1)

Authors: Andreu Girbau-Xalabarder, Jun Nagata, Shinichi Sumiyoshi

Event cameras capture scene changes asynchronously on a per-pixel basis,
enabling extremely high temporal resolution. However, this advantage comes at
the cost of high bandwidth, memory, and computational demands. To address this,
prior work has explored event downsampling, but most approaches rely on fixed
heuristics or threshold-based strategies, limiting their adaptability. Instead,
we propose a probabilistic framework, POLED, that models event importance
through an event-importance probability density function (ePDF), which can be
arbitrarily defined and adapted to different applications. Our approach
operates in a purely online setting, estimating event importance on-the-fly
from raw event streams, enabling scene-specific adaptation. Additionally, we
introduce zero-shot event downsampling, where downsampled events must remain
usable for models trained on the original event stream, without task-specific
adaptation. We design a contour-preserving ePDF that prioritizes structurally
important events and evaluate our method across four datasets and tasks--object
classification, image interpolation, surface normal estimation, and object
detection--demonstrating that intelligent sampling is crucial for maintaining
performance under event-budget constraints.

### 2. [Stacking the Odds: Full-Stack Quantum System Design Space Exploration](http://arxiv.org/pdf/2506.02782v1)

Authors: Hila Safi, Medina Bandic, Christoph Niedermeier, Carmen G. Almudever, Sebastian Feld, Wolfgang Mauerer

Design space exploration (DSE) plays an important role in optimising quantum
circuit execution by systematically evaluating different configurations of
compilation strategies and hardware settings. In this work, we study the impact
of layout methods, qubit routing techniques, compiler optimization levels, and
hardware-specific properties, including noise characteristics, topological
structures, connectivity densities, and device sizes. By traversing these
dimensions, we aim to understand how compilation choices interact with hardware
features. A central question in our study is whether carefully selected device
parameters and mapping strategies, including initial layouts and routing
heuristics, can mitigate hardware-induced errors beyond standard error
mitigation methods. Our results show that choosing the right software
strategies (e.g., layout and routing) and tailoring hardware properties (e.g.,
reducing noise or leveraging connectivity) significantly enhances the fidelity
of quantum circuit executions. We provide performance estimates using metrics
such as circuit depth, gate count, and expected fidelity. These findings
highlight the value of hardware-software co-design, especially as quantum
systems scale and move toward error-corrected computing. Our simulations,
though noisy, include quantum error correction (QEC) scenarios, revealing
similar sensitivities to layout and connectivity. This suggests that co-design
principles will be vital for integrating QEC in future devices. Overall, we
offer practical guidance for co-optimizing mapping, routing, and hardware
configuration in real-world quantum computing.

### 3. [Zero-Energy RIS-Assisted Communications With Noise Modulation and Interference-Based Energy Harvesting](http://arxiv.org/pdf/2506.02625v1)

Authors: Ahmad Massud Tota Khel, Aissa Ikhlef, Zhiguo Ding, Hongjian Sun

To advance towards carbon-neutrality and improve the limited {performance} of
conventional passive wireless communications, in this paper, we investigate the
integration of noise modulation with zero-energy reconfigurable intelligent
surfaces (RISs). In particular, the RIS reconfigurable elements (REs) are
divided into two groups: one for beamforming the desired signals in reflection
mode and another for harvesting energy from interference signals in an
absorption mode, providing the power required for RIS operation. Since the
harvested energy is a random variable, a random number of REs can beamform the
signals, while the remainder blindly reflects them. We present a closed-form
solution and a search algorithm for REs allocation, jointly optimizing both the
energy harvesting (EH) and communication performance. Considering the
repetition coding technique and discrete phase shifts, we derive analytical
expressions for the energy constrained success rate, bit error rate, optimal
threshold, mutual information, {and energy efficiency}. Numerical and
simulation results confirm the effectiveness of the algorithm and expressions,
demonstrating the superiority of the proposed integration over conventional
noise-modulation systems. It is shown that by properly allocating the REs, both
the EH and communication performance can be improved in low to moderate
interference scenarios, while the latter is restricted in the high-interference
regime.

### 4. [Process Mining on Distributed Data Sources](http://arxiv.org/pdf/2506.02830v1)

Authors: Maximilian Weisenseel, Julia Andersen, Samira Akili, Christian Imenkamp, Hendrik Reiter, Christoffer Rubensson, Wilhelm Hasselbring, Olaf Landsiedel, Xixi Lu, Jan Mendling, Florian Tschorsch, Matthias Weidlich, Agnes Koschmider

Major domains such as logistics, healthcare, and smart cities increasingly
rely on sensor technologies and distributed infrastructures to monitor complex
processes in real time. These developments are transforming the data landscape
from discrete, structured records stored in centralized systems to continuous,
fine-grained, and heterogeneous event streams collected across distributed
environments. As a result, traditional process mining techniques, which assume
centralized event logs from enterprise systems, are no longer sufficient. In
this paper, we discuss the conceptual and methodological foundations for this
emerging field. We identify three key shifts: from offline to online analysis,
from centralized to distributed computing, and from event logs to sensor data.
These shifts challenge traditional assumptions about process data and call for
new approaches that integrate infrastructure, data, and user perspectives. To
this end, we define a research agenda that addresses six interconnected fields,
each spanning multiple system dimensions. We advocate a principled methodology
grounded in algorithm engineering, combining formal modeling with empirical
evaluation. This approach enables the development of scalable, privacy-aware,
and user-centric process mining techniques suitable for distributed
environments. Our synthesis provides a roadmap for advancing process mining
beyond its classical setting, toward a more responsive and decentralized
paradigm of process intelligence.

### Graphics

### 1. [Voyager: Real-Time Splatting City-Scale 3D Gaussians on Your Phone](http://arxiv.org/pdf/2506.02774v1)

Authors: Zheng Liu, He Zhu, Xinyang Li, Yirun Wang, Yujiao Shi, Wei Li, Jingwen Leng, Minyi Guo, Yu Feng

3D Gaussian Splatting (3DGS) is an emerging technique for photorealistic 3D
scene rendering. However, rendering city-scale 3DGS scenes on mobile devices,
e.g., your smartphones, remains a significant challenge due to the limited
resources on mobile devices. A natural solution is to offload computation to
the cloud; however, naively streaming rendered frames from the cloud to the
client introduces high latency and requires bandwidth far beyond the capacity
of current wireless networks.
  In this paper, we propose an effective solution to enable city-scale 3DGS
rendering on mobile devices. Our key insight is that, under normal user motion,
the number of newly visible Gaussians per second remains roughly constant.
Leveraging this, we stream only the necessary Gaussians to the client.
Specifically, on the cloud side, we propose asynchronous level-of-detail search
to identify the necessary Gaussians for the client. On the client side, we
accelerate rendering via a lookup table-based rasterization. Combined with
holistic runtime optimizations, our system can deliver low-latency, city-scale
3DGS rendering on mobile devices. Compared to existing solutions, Voyager
achieves over 100$\times$ reduction on data transfer and up to 8.9$\times$
speedup while retaining comparable rendering quality.

### 2. [FlexPainter: Flexible and Multi-View Consistent Texture Generation](http://arxiv.org/pdf/2506.02620v1)

Authors: Dongyu Yan, Leyi Wu, Jiantao Lin, Luozhou Wang, Tianshuo Xu, Zhifei Chen, Zhen Yang, Lie Xu, Shunsi Zhang, Yingcong Chen

Texture map production is an important part of 3D modeling and determines the
rendering quality. Recently, diffusion-based methods have opened a new way for
texture generation. However, restricted control flexibility and limited prompt
modalities may prevent creators from producing desired results. Furthermore,
inconsistencies between generated multi-view images often lead to poor texture
generation quality. To address these issues, we introduce \textbf{FlexPainter},
a novel texture generation pipeline that enables flexible multi-modal
conditional guidance and achieves highly consistent texture generation. A
shared conditional embedding space is constructed to perform flexible
aggregation between different input modalities. Utilizing such embedding space,
we present an image-based CFG method to decompose structural and style
information, achieving reference image-based stylization. Leveraging the 3D
knowledge within the image diffusion prior, we first generate multi-view images
simultaneously using a grid representation to enhance global understanding.
Meanwhile, we propose a view synchronization and adaptive weighting module
during diffusion sampling to further ensure local consistency. Finally, a
3D-aware texture completion model combined with a texture enhancement model is
used to generate seamless, high-resolution texture maps. Comprehensive
experiments demonstrate that our framework significantly outperforms
state-of-the-art methods in both flexibility and generation quality.

### 3. [VolTex: Food Volume Estimation using Text-Guided Segmentation and Neural Surface Reconstruction](http://arxiv.org/pdf/2506.02895v1)

Authors: Ahmad AlMughrabi, Umair Haroon, Ricardo Marques, Petia Radeva

Accurate food volume estimation is crucial for dietary monitoring, medical
nutrition management, and food intake analysis. Existing 3D Food Volume
estimation methods accurately compute the food volume but lack for food
portions selection. We present VolTex, a framework that improves \change{the
food object selection} in food volume estimation. Allowing users to specify a
target food item via text input to be segmented, our method enables the precise
selection of specific food objects in real-world scenes. The segmented object
is then reconstructed using the Neural Surface Reconstruction method to
generate high-fidelity 3D meshes for volume computation. Extensive evaluations
on the MetaFood3D dataset demonstrate the effectiveness of our approach in
isolating and reconstructing food items for accurate volume estimation. The
source code is accessible at https://github.com/GCVCG/VolTex.

### 4. [PartComposer: Learning and Composing Part-Level Concepts from Single-Image Examples](http://arxiv.org/pdf/2506.03004v1)

Authors: Junyu Liu, R. Kenny Jones, Daniel Ritchie

We present PartComposer: a framework for part-level concept learning from
single-image examples that enables text-to-image diffusion models to compose
novel objects from meaningful components. Existing methods either struggle with
effectively learning fine-grained concepts or require a large dataset as input.
We propose a dynamic data synthesis pipeline generating diverse part
compositions to address one-shot data scarcity. Most importantly, we propose to
maximize the mutual information between denoised latents and structured concept
codes via a concept predictor, enabling direct regulation on concept
disentanglement and re-composition supervision. Our method achieves strong
disentanglement and controllable composition, outperforming subject and
part-level baselines when mixing concepts from the same, or different, object
categories.

### 5. [HumanRAM: Feed-forward Human Reconstruction and Animation Model using Transformers](http://arxiv.org/pdf/2506.03118v1)

Authors: Zhiyuan Yu, Zhe Li, Hujun Bao, Can Yang, Xiaowei Zhou

3D human reconstruction and animation are long-standing topics in computer
graphics and vision. However, existing methods typically rely on sophisticated
dense-view capture and/or time-consuming per-subject optimization procedures.
To address these limitations, we propose HumanRAM, a novel feed-forward
approach for generalizable human reconstruction and animation from monocular or
sparse human images. Our approach integrates human reconstruction and animation
into a unified framework by introducing explicit pose conditions, parameterized
by a shared SMPL-X neural texture, into transformer-based large reconstruction
models (LRM). Given monocular or sparse input images with associated camera
parameters and SMPL-X poses, our model employs scalable transformers and a
DPT-based decoder to synthesize realistic human renderings under novel
viewpoints and novel poses. By leveraging the explicit pose conditions, our
model simultaneously enables high-quality human reconstruction and
high-fidelity pose-controlled animation. Experiments show that HumanRAM
significantly surpasses previous methods in terms of reconstruction accuracy,
animation fidelity, and generalization performance on real-world datasets.
Video results are available at https://zju3dv.github.io/humanram/.

### 6. [EyeNavGS: A 6-DoF Navigation Dataset and Record-n-Replay Software for Real-World 3DGS Scenes in VR](http://arxiv.org/pdf/2506.02380v1)

Authors: Zihao Ding, Cheng-Tse Lee, Mufeng Zhu, Tao Guan, Yuan-Chun Sun, Cheng-Hsin Hsu, Yao Liu

3D Gaussian Splatting (3DGS) is an emerging media representation that
reconstructs real-world 3D scenes in high fidelity, enabling
6-degrees-of-freedom (6-DoF) navigation in virtual reality (VR). However,
developing and evaluating 3DGS-enabled applications and optimizing their
rendering performance, require realistic user navigation data. Such data is
currently unavailable for photorealistic 3DGS reconstructions of real-world
scenes. This paper introduces EyeNavGS (EyeNavGS), the first publicly available
6-DoF navigation dataset featuring traces from 46 participants exploring twelve
diverse, real-world 3DGS scenes. The dataset was collected at two sites, using
the Meta Quest Pro headsets, recording the head pose and eye gaze data for each
rendered frame during free world standing 6-DoF navigation. For each of the
twelve scenes, we performed careful scene initialization to correct for scene
tilt and scale, ensuring a perceptually-comfortable VR experience. We also
release our open-source SIBR viewer software fork with record-and-replay
functionalities and a suite of utility tools for data processing, conversion,
and visualization. The EyeNavGS dataset and its accompanying software tools
provide valuable resources for advancing research in 6-DoF viewport prediction,
adaptive streaming, 3D saliency, and foveated rendering for 3DGS scenes. The
EyeNavGS dataset is available at: https://symmru.github.io/EyeNavGS/.

### 7. [MotionRAG-Diff: A Retrieval-Augmented Diffusion Framework for Long-Term Music-to-Dance Generation](http://arxiv.org/pdf/2506.02661v1)

Authors: Mingyang Huang, Peng Zhang, Bang Zhang

Generating long-term, coherent, and realistic music-conditioned dance
sequences remains a challenging task in human motion synthesis. Existing
approaches exhibit critical limitations: motion graph methods rely on fixed
template libraries, restricting creative generation; diffusion models, while
capable of producing novel motions, often lack temporal coherence and musical
alignment. To address these challenges, we propose $\textbf{MotionRAG-Diff}$, a
hybrid framework that integrates Retrieval-Augmented Generation (RAG) with
diffusion-based refinement to enable high-quality, musically coherent dance
generation for arbitrary long-term music inputs. Our method introduces three
core innovations: (1) A cross-modal contrastive learning architecture that
aligns heterogeneous music and dance representations in a shared latent space,
establishing unsupervised semantic correspondence without paired data; (2) An
optimized motion graph system for efficient retrieval and seamless
concatenation of motion segments, ensuring realism and temporal coherence
across long sequences; (3) A multi-condition diffusion model that jointly
conditions on raw music signals and contrastive features to enhance motion
quality and global synchronization. Extensive experiments demonstrate that
MotionRAG-Diff achieves state-of-the-art performance in motion quality,
diversity, and music-motion synchronization accuracy. This work establishes a
new paradigm for music-driven dance generation by synergizing retrieval-based
template fidelity with diffusion-based creative enhancement.

### 8. [PhysGaia: A Physics-Aware Dataset of Multi-Body Interactions for Dynamic Novel View Synthesis](http://arxiv.org/pdf/2506.02794v1)

Authors: Mijeong Kim, Gunhee Kim, Jungyoon Choi, Wonjae Roh, Bohyung Han

We introduce PhysGaia, a novel physics-aware dataset specifically designed
for Dynamic Novel View Synthesis (DyNVS), encompassing both structured objects
and unstructured physical phenomena. Unlike existing datasets that primarily
focus on photorealistic reconstruction, PhysGaia is created to actively support
physics-aware dynamic scene modeling. Our dataset provides complex dynamic
scenarios with rich interactions among multiple objects, where they
realistically collide with each other and exchange forces. Furthermore, it
contains a diverse range of physical materials, such as liquid, gas,
viscoelastic substance, and textile, which moves beyond the rigid bodies
prevalent in existing datasets. All scenes in PhysGaia are faithfully generated
to strictly adhere to physical laws, leveraging carefully selected
material-specific physics solvers. To enable quantitative evaluation of
physical modeling, our dataset provides essential ground-truth information,
including 3D particle trajectories and physics parameters, e.g., viscosity. To
facilitate research adoption, we also provide essential integration pipelines
for using state-of-the-art DyNVS models with our dataset and report their
results. By addressing the critical lack of datasets for physics-aware
modeling, PhysGaia will significantly advance research in dynamic view
synthesis, physics-based scene understanding, and deep learning models
integrated with physical simulation -- ultimately enabling more faithful
reconstruction and interpretation of complex dynamic scenes. Our datasets and
codes are available in the project website,
http://cvlab.snu.ac.kr/research/PhysGaia.

### 9. [TalkingMachines: Real-Time Audio-Driven FaceTime-Style Video via Autoregressive Diffusion Models](http://arxiv.org/pdf/2506.03099v1)

Authors: Chetwin Low, Weimin Wang

In this paper, we present TalkingMachines -- an efficient framework that
transforms pretrained video generation models into real-time, audio-driven
character animators. TalkingMachines enables natural conversational experiences
by integrating an audio large language model (LLM) with our video generation
foundation model. Our primary contributions include: (1) We adapt a pretrained
SOTA image-to-video DiT into an audio-driven avatar generation model of 18
billion parameters; (2) We enable infinite video streaming without error
accumulation through asymmetric knowledge distillation from a bidirectional
teacher model into a sparse causal, autoregressive student model; (3) We design
a high-throughput, low-latency inference pipeline incorporating several key
engineering optimizations such as: (a) disaggregation of the DiT and VAE
decoder across separate devices, (b) efficient overlap of inter-device
communication and computation using CUDA streams, (c) elimination of redundant
recomputations to maximize frame-generation throughput. Please see demo videos
here - https://aaxwaz.github.io/TalkingMachines/

### Computer Science and Game Theory

### 1. [A Transformer-Based Neural Network for Optimal Deterministic-Allocation and Anonymous Joint Auction Design](http://arxiv.org/pdf/2506.02435v1)

Authors: Zhen Zhang, Luowen Liu, Wanzhi Zhang, Zitian Guo, Kun Huang, Qi Qi, Qiang Liu, Xingxing Wang

With the advancement of machine learning, an increasing number of studies are
employing automated mechanism design (AMD) methods for optimal auction design.
However, all previous AMD architectures designed to generate optimal mechanisms
that satisfy near dominant strategy incentive compatibility (DSIC) fail to
achieve deterministic allocation, and some also lack anonymity, thereby
impacting the efficiency and fairness of advertising allocation. This has
resulted in a notable discrepancy between the previous AMD architectures for
generating near-DSIC optimal mechanisms and the demands of real-world
advertising scenarios. In this paper, we prove that in all online advertising
scenarios, previous non-deterministic allocation methods lead to the
non-existence of feasible solutions, resulting in a gap between the rounded
solution and the optimal solution. Furthermore, we propose JTransNet, a
transformer-based neural network architecture, designed for optimal
deterministic-allocation and anonymous joint auction design. Although the
deterministic allocation module in JTransNet is designed for the latest joint
auction scenarios, it can be applied to other non-deterministic AMD
architectures with minor modifications. Additionally, our offline and online
data experiments demonstrate that, in joint auction scenarios, JTransNet
significantly outperforms baseline methods in terms of platform revenue,
resulting in a substantial increase in platform earnings.

### 2. [Computational adversarial risk analysis for general security games](http://arxiv.org/pdf/2506.02603v1)

Authors: Jose Manuel Camacho, Roi Naveiro, David Rios Insua

This paper provides an efficient computational scheme to handle general
security games from an adversarial risk analysis perspective. Two cases in
relation to single-stage and multi-stage simultaneous defend-attack games
motivate our approach to general setups which uses bi-agent influence diagrams
as underlying problem structure and augmented probability simulation as core
computational methodology. Theoretical convergence and numerical, modeling, and
implementation issues are thoroughly discussed. A disinformation war case study
illustrates the relevance of the proposed approach.

### 3. [Branch-and-Cut for Mixed-Integer Generalized Nash Equilibrium Problems](http://arxiv.org/pdf/2506.02520v1)

Authors: Aloïs Duguet, Tobias Harks, Martin Schmidt, Julian Schwarz

Generalized Nash equilibrium problems with mixed-integer variables form an
important class of games in which each player solves a mixed-integer
optimization problem with respect to her own variables and the strategy space
of each player depends on the strategies chosen by the rival players. In this
work, we introduce a branch-and-cut algorithm to compute exact pure Nash
equilibria for different classes of such mixed-integer games. The main idea is
to reformulate the equilibrium problem as a suitable bilevel problem based on
the Nikaido--Isoda function of the game. The proposed branch-and-cut method is
applicable to generalized Nash equilibrium problems under quite mild
assumptions. Depending on the specific setting, we use tailored equilibrium or
intersection cuts. The latter are well-known in mixed-integer linear
optimization and we adapt them to the game setting. We prove finite termination
and correctness of the algorithm and present some first numerical results for
two different types of knapsack games and another game based on capacitated
flow problems.

### 4. [The power of mediators: Price of anarchy and stability in Bayesian games with submodular social welfare](http://arxiv.org/pdf/2506.02655v1)

Authors: Kaito Fujii

This paper investigates the role of mediators in Bayesian games by examining
their impact on social welfare through the price of anarchy (PoA) and price of
stability (PoS). Mediators can communicate with players to guide them toward
equilibria of varying quality, and different communication protocols lead to a
variety of equilibrium concepts collectively known as Bayes (coarse) correlated
equilibria. To analyze these equilibrium concepts, we consider a general class
of Bayesian games with submodular social welfare, which naturally extends valid
utility games and their variant, basic utility games. These frameworks,
introduced by Vetta (2002), have been developed to analyze the social welfare
guarantees of equilibria in games such as competitive facility location,
influence maximization, and other resource allocation problems.
  We provide upper and lower bounds on the PoA and PoS for a broad class of
Bayes (coarse) correlated equilibria. Central to our analysis is the strategy
representability gap, which measures the multiplicative gap between the optimal
social welfare achievable with and without knowledge of other players' types.
For monotone submodular social welfare functions, we show that this gap is
$1-1/\mathrm{e}$ for independent priors and $\Theta(1/\sqrt{n})$ for correlated
priors, where $n$ is the number of players. These bounds directly lead to upper
and lower bounds on the PoA and PoS for various equilibrium concepts, while we
also derive improved bounds for specific concepts by developing smoothness
arguments. Notably, we identify a fundamental gap in the PoA and PoS across
different classes of Bayes correlated equilibria, highlighting essential
distinctions among these concepts.

### 5. [Proportional Response Dynamics in Gross Substitutes Markets](http://arxiv.org/pdf/2506.02852v1)

Authors: Yun Kuen Cheung, Richard Cole, Yixin Tao

Proportional response is a well-established distributed algorithm which has
been shown to converge to competitive equilibria in both Fisher and
Arrow-Debreu markets, for various sub-families of homogeneous utilities,
including linear and constant elasticity of substitution utilities. We propose
a natural generalization of proportional response for gross substitutes
utilities, and prove that it converges to competitive equilibria in Fisher
markets. This is the first convergence result of a proportional response style
dynamics in Fisher markets for utilities beyond the homogeneous utilities
covered by the Eisenberg-Gale convex program. We show an empirical convergence
rate of $O(1/T)$ for the prices. Furthermore, we show that the allocations of a
lazy version of the generalized proportional response dynamics converge to
competitive equilibria in Arrow-Debreu markets.

### 6. [Dynamic Fee for Reducing Impermanent Loss in Decentralized Exchanges](http://arxiv.org/pdf/2506.03001v1)

Authors: Irina Lebedeva, Dmitrii Umnov, Yury Yanovich, Ignat Melnikov, George Ovchinnikov

Decentralized exchanges (DEXs) are crucial to decentralized finance (DeFi) as
they enable trading without intermediaries. However, they face challenges like
impermanent loss (IL), where liquidity providers (LPs) see their assets' value
change unfavorably within a liquidity pool compared to outside it. To tackle
these issues, we propose dynamic fee mechanisms over traditional fixed-fee
structures used in automated market makers (AMM). Our solution includes
asymmetric fees via block-adaptive, deal-adaptive, and the "ideal but
unattainable" oracle-based fee algorithm, utilizing all data available to
arbitrageurs to mitigate IL. We developed a simulation-based framework to
compare these fee algorithms systematically. This framework replicates trading
on a DEX, considering both informed and uninformed users and a psychological
relative loss factor. Results show that adaptive algorithms outperform
fixed-fee baselines in reducing IL while maintaining trading activity among
uninformed users. Additionally, insights from oracle-based performance
underscore the potential of dynamic fee strategies to lower IL, boost LP
profitability, and enhance overall market efficiency.

### 7. [Designing Algorithmic Delegates: The Role of Indistinguishability in Human-AI Handoff](http://arxiv.org/pdf/2506.03102v1)

Authors: Sophie Greenwood, Karen Levy, Solon Barocas, Hoda Heidari, Jon Kleinberg

As AI technologies improve, people are increasingly willing to delegate tasks
to AI agents. In many cases, the human decision-maker chooses whether to
delegate to an AI agent based on properties of the specific instance of the
decision-making problem they are facing. Since humans typically lack full
awareness of all the factors relevant to this choice for a given
decision-making instance, they perform a kind of categorization by treating
indistinguishable instances -- those that have the same observable features --
as the same. In this paper, we define the problem of designing the optimal
algorithmic delegate in the presence of categories. This is an important
dimension in the design of algorithms to work with humans, since we show that
the optimal delegate can be an arbitrarily better teammate than the optimal
standalone algorithmic agent. The solution to this optimal delegation problem
is not obvious: we discover that this problem is fundamentally combinatorial,
and illustrate the complex relationship between the optimal design and the
properties of the decision-making task even in simple settings. Indeed, we show
that finding the optimal delegate is computationally hard in general. However,
we are able to find efficient algorithms for producing the optimal delegate in
several broad cases of the problem, including when the optimal action may be
decomposed into functions of features observed by the human and the algorithm.
Finally, we run computational experiments to simulate a designer updating an
algorithmic delegate over time to be optimized for when it is actually adopted
by users, and show that while this process does not recover the optimal
delegate in general, the resulting delegate often performs quite well.

### Human-Computer Interaction

### 1. [Visualization for interactively adjusting the de-bias effect of word embedding](http://arxiv.org/pdf/2506.02447v1)

Authors: Arisa Sugino, Takayuki Itoh

Word embedding, which converts words into numerical values, is an important
natural language processing technique and widely used. One of the serious
problems of word embedding is that the bias will be learned and affect the
model if the dataset used for pre-training contains bias. On the other hand,
indiscriminate removal of bias from word embeddings may result in the loss of
information, even if the bias is undesirable to us. As a result, a risk of
model performance degradation due to bias removal will be another problem. As a
solution to this problem, we focus on gender bias in Japanese and propose an
interactive visualization method to adjust the degree of debias for each word
category. Specifically, we visualize the accuracy in a category classification
task after debiasing, and allow the user to adjust the parameters based on the
visualization results, so that the debiasing can be adjusted according to the
user's objectives. In addition, considering a trade-off between debiasing and
preventing degradation of model performance, and that different people perceive
gender bias differently, we developed a mechanism to present multiple choices
of debiasing configurations applying an optimization scheme. This paper
presents the results of an experiment in which we removed the gender bias for
word embeddings learned from the Japanese version of Wikipedia. We classified
words into five categories based on a news corpus, and observed that the degree
of influence of debiasing differed greatly among the categories. We then
adjusted the degree of debiasing for each category based on the visualization
results.

### 2. [To Embody or Not: The Effect Of Embodiment On User Perception Of LLM-based Conversational Agents](http://arxiv.org/pdf/2506.02514v1)

Authors: Kyra Wang, Boon-Kiat Quek, Jessica Goh, Dorien Herremans

Embodiment in conversational agents (CAs) refers to the physical or visual
representation of these agents, which can significantly influence user
perception and interaction. Limited work has been done examining the effect of
embodiment on the perception of CAs utilizing modern large language models
(LLMs) in non-hierarchical cooperative tasks, a common use case of CAs as more
powerful models become widely available for general use. To bridge this
research gap, we conducted a mixed-methods within-subjects study on how users
perceive LLM-based CAs in cooperative tasks when embodied and non-embodied. The
results show that the non-embodied agent received significantly better
quantitative appraisals for competence than the embodied agent, and in
qualitative feedback, many participants believed that the embodied CA was more
sycophantic than the non-embodied CA. Building on prior work on users'
perceptions of LLM sycophancy and anthropomorphic features, we theorize that
the typically-positive impact of embodiment on perception of CA credibility can
become detrimental in the presence of sycophancy. The implication of such a
phenomenon is that, contrary to intuition and existing literature, embodiment
is not a straightforward way to improve a CA's perceived credibility if there
exists a tendency to sycophancy.

### 3. [Cognitive Load-Driven VR Memory Palaces: Personalizing Focus and Recall Enhancement](http://arxiv.org/pdf/2506.02700v1)

Authors: Zhengyang Li, Hailin Deng

Cognitive load, which varies across individuals, can significantly affect
focus and memory performance.This study explores the integration of Virtual
Reality (VR) with memory palace techniques, aiming to optimize VR environments
tailored to individual cognitive load levels to improve focus and memory. We
utilized EEG devices, specifically the Oculus Quest 2, to monitor Beta wave
activity in 10 participants.By modeling their cognitive load profiles through
polynomial regression, we dynamically adjusted spatial variables within a VR
environment using Grasshopper, creating personalized experiences. Results
indicate that 8 participants showed a notable increase in Beta wave activity,
demonstrating improved focus and cognitive performance in the customized VR
settings.These findings underscore the potential of VR-based memory
environments, driven by cognitive load considerations, and provide valuable
insights for advancing VR memory research

### 4. [Heatables: Effects of Infrared-LED-Induced Ear Heating on Thermal Perception, Comfort, and Cognitive Performance](http://arxiv.org/pdf/2506.02714v1)

Authors: Valeria Zitz, Michael Küttner, Jonas Hummel, Michael T. Knierim, Michael Beigl, Tobias Röddiger

Maintaining thermal comfort in shared indoor environments remains
challenging, as centralized HVAC systems are slow to adapt and standardized to
group norms. Cold exposure not only reduces subjective comfort but can impair
cognitive performance, particularly under moderate to severe cold stress.
Personal Comfort Systems (PCS) have shown promise by providing localized
heating, yet many designs target distal body parts with low thermosensitivity
and often lack portability. In this work, we investigate whether targeted
thermal stimulation using in-ear worn devices can manipulate thermal perception
and enhance thermal comfort. We present Heatables, a novel in-ear wearable that
emits Near-Infrared (NIR) and Infrared (IR) radiation via integrated LEDs to
deliver localized optical heating. This approach leverages NIR-IR's ability to
penetrate deeper tissues, offering advantages over traditional resistive
heating limited to surface warming. In a placebo-controlled study with 24
participants, each exposed for 150 minutes in a cool office environment
(approximately 17.5 degrees Celsius) to simulate sustained cold stress during
typical sedentary office activities, Heatables significantly increased the
perceived ambient temperature by around 1.5 degrees Celsius and delayed cold
discomfort. Importantly, thermal benefits extended beyond the ear region,
improving both whole-body comfort and thermal acceptability. These findings
position in-ear NIR-IR-LED-based stimulation as a promising modality for
unobtrusive thermal comfort enhancement in everyday contexts.

### 5. [Exploring listeners' perceptions of AI-generated and human-composed music for functional emotional applications](http://arxiv.org/pdf/2506.02856v1)

Authors: Kimaya Lecamwasam, Tishya Ray Chaudhuri

This work investigates how listeners perceive and evaluate AI-generated as
compared to human-composed music in the context of emotional resonance and
regulation. Across a mixed-methods design, participants were exposed to both AI
and human music under various labeling conditions (music correctly labeled as
AI- or human-origin, music incorrectly labeled as AI- or human-origin, and
unlabeled music) and emotion cases (Calm and Upbeat), and were asked to rate
preference, efficacy of target emotion elicitation, and emotional impact.
Participants were significantly more likely to rate human-composed music,
regardless of labeling, as more effective at eliciting target emotional states,
though quantitative analyses revealed no significant differences in emotional
response. However, participants were significantly more likely to indicate
preference for AI-generated music, yielding further questions regarding the
impact of emotional authenticity and perceived authorship on musical appraisal.
Qualitative data underscored this, with participants associating humanness with
qualities such as imperfection, flow, and 'soul.' These findings challenge the
assumption that preference alone signals success in generative music systems.
Rather than positioning AI tools as replacements for human creativity or
emotional expression, they point toward a more careful design ethos that
acknowledges the limits of replication and prioritizes human values such as
authenticity, individuality, and emotion regulation in wellness and affective
technologies.

### 6. [Unpacking Graduate Students' Learning Experience with Generative AI Teaching Assistant in A Quantitative Methodology Course](http://arxiv.org/pdf/2506.02966v1)

Authors: Zhanxin Hao, Haifeng Luo, Yongyi Chen, Yu Zhang

The study was conducted in an Advanced Quantitative Research Methods course
involving 20 graduate students. During the course, student inquiries made to
the AI were recorded and coded using Bloom's taxonomy and the CLEAR framework.
A series of independent sample t-tests and poisson regression analyses were
employed to analyse the characteristics of different questions asked by
students with different backgrounds. Post course interviews were conducted with
10 students to gain deeper insights into their perceptions. The findings
revealed a U-shaped pattern in students' use of the AI assistant, with higher
usage at the beginning and towards the end of the course, and a decrease in
usage during the middle weeks. Most questions posed to the AI focused on
knowledge and comprehension levels, with fewer questions involving deeper
cognitive thinking. Students with a weaker mathematical foundation used the AI
assistant more frequently, though their inquiries tended to lack explicit and
logical structure compared to those with a strong mathematical foundation, who
engaged less with the tool. These patterns suggest the need for targeted
guidance to optimise the effectiveness of AI tools for students with varying
levels of academic proficiency.

### 7. [Mapping Student-AI Interaction Dynamics in Multi-Agent Learning Environments: Supporting Personalised Learning and Reducing Performance Gaps](http://arxiv.org/pdf/2506.02993v1)

Authors: Zhanxin Hao, Jie Cao, Ruimiao Li, Jifan Yu, Zhiyuan Liu, Yu Zhang

Multi-agent AI systems, which simulate diverse instructional roles such as
teachers and peers, offer new possibilities for personalized and interactive
learning. Yet, student-AI interaction patterns and their pedagogical
implications remain unclear. This study explores how university students
engaged with multiple AI agents, and how these interactions influenced
cognitive outcomes (learning gains) and non-cognitive factors (motivation,
technology acceptance). Based on MAIC, an online learning platform with
multi-agent, the research involved 305 university students and 19,365 lines of
dialogue data. Pre- and post-test scores, self-reported motivation and
technology acceptance were also collected. The study identified two engagement
patterns: co-construction of knowledge and co-regulation. Lag sequential
analysis revealed that students with lower prior knowledge relied more on
co-construction of knowledge sequences, showing higher learning gains and
post-course motivation. In contrast, students with higher prior knowledge
engaged more in co-regulation behaviors but exhibited limited learning
improvement. Technology acceptance increased across all groups. These findings
suggest that multi-agent AI systems can adapt to students' varying needs,
support differentiated engagement, and reduce performance gaps. Implications
for personalized system design and future research directions are discussed.

### 8. [Feedstack: Layering Structured Representations over Unstructured Feedback to Scaffold Human AI Conversation](http://arxiv.org/pdf/2506.03052v1)

Authors: Hannah Vy Nguyen, Yu-Chun Grace Yen, Omar Shakir, Hang Huynh, Sebastian Gutierrez, June A. Smith, Sheila Jimenez, Salma Abdelgelil, Stephen MacNeil

Many conversational user interfaces facilitate linear conversations with
turn-based dialogue, similar to face-to-face conversations between people.
However, digital conversations can afford more than simple back-and-forth; they
can be layered with interaction techniques and structured representations that
scaffold exploration, reflection, and shared understanding between users and AI
systems. We introduce Feedstack, a speculative interface that augments feedback
conversations with layered affordances for organizing, navigating, and
externalizing feedback. These layered structures serve as a shared
representation of the conversation that can surface user intent and reveal
underlying design principles. This work represents an early exploration of this
vision using a research-through-design approach. We describe system features
and design rationale, and present insights from two formative (n=8, n=8)
studies to examine how novice designers engage with these layered supports.
Rather than presenting a conclusive evaluation, we reflect on Feedstack as a
design probe that opens up new directions for conversational feedback systems.

### 9. [Assessing Workers Neuro-physiological Stress Responses to Augmented Reality Safety Warnings in Immersive Virtual Roadway Work Zones](http://arxiv.org/pdf/2506.03113v1)

Authors: Fatemeh Banani Ardecani, Omidreza Shoghli

This paper presents a multi-stage experimental framework that integrates
immersive Virtual Reality (VR) simulations, wearable sensors, and advanced
signal processing to investigate construction workers neuro-physiological
stress responses to multi-sensory AR-enabled warnings. Participants performed
light- and moderate-intensity roadway maintenance tasks within a high-fidelity
VR roadway work zone, while key stress markers of electrodermal activity (EDA),
heart rate variability (HRV), and electroencephalography (EEG) were
continuously measured. Statistical analyses revealed that task intensity
significantly influenced physiological and neurological stress indicators.
Moderate-intensity tasks elicited greater autonomic arousal, evidenced by
elevated heart rate measures (mean-HR, std-HR, max-HR) and stronger
electrodermal responses, while EEG data indicated distinct stress-related alpha
suppression and beta enhancement. Feature-importance analysis further
identified mean EDR and short-term HR metrics as discriminative for classifying
task intensity. Correlation results highlighted a temporal lag between
immediate neural changes and subsequent physiological stress reactions,
emphasizing the interplay between cognition and autonomic regulation during
hazardous tasks.

### 10. [IP-Dialog: Evaluating Implicit Personalization in Dialogue Systems with Synthetic Data](http://arxiv.org/pdf/2506.02449v1)

Authors: Bo Peng, Zhiheng Wang, Heyang Gong, Chaochao Lu

In modern dialogue systems, the ability to implicitly infer user backgrounds
from conversations and leverage this information for personalized assistance is
crucial. However, the scarcity of high-quality data remains a fundamental
challenge to evaluating and improving this capability. Traditional dataset
construction methods are labor-intensive, resource-demanding, and raise privacy
concerns. To address these issues, we propose a novel approach for automatic
synthetic data generation and introduce the Implicit Personalized Dialogue
(IP-Dialog) benchmark along with a training dataset, covering 10 tasks and 12
user attribute types. Additionally, we develop a systematic evaluation
framework with four metrics to assess both attribute awareness and reasoning
capabilities. We further propose five causal graphs to elucidate models'
reasoning pathways during implicit personalization. Extensive experiments yield
insightful observations and prove the reliability of our dataset.

### Information Retrieval

### 1. [NextQuill: Causal Preference Modeling for Enhancing LLM Personalization](http://arxiv.org/pdf/2506.02368v1)

Authors: Xiaoyan Zhao, Juntao You, Yang Zhang, Wenjie Wang, Hong Cheng, Fuli Feng, See-Kiong Ng, Tat-Seng Chua

Personalizing large language models (LLMs) for individual users has become
increasingly important as they are progressively integrated into real-world
applications to support users' daily lives. However, existing personalization
approaches often fail to distinguish which components of model predictions and
training data truly reflect user preferences, leading to superficial
personalization alignment. In this paper, we introduce NextQuill, a novel LLM
personalization alignment framework grounded in causal preference modeling. We
approach personalization from a causal perspective, treating both model
predictions and ground-truth data generation as outcomes influenced by user
preferences, along with other factors. We define the true preference effect as
the causal impact of user history (which reflects preferences) on each token
prediction or data generation instance, estimated through causal intervention
techniques. Building on this insight, NextQuill introduces two complementary
alignment strategies: (1) aligning model-internal causal preference effects on
predictions with those reflected in ground-truth data, rather than
indiscriminately fitting predictions, and (2) focusing on fitting
preference-bearing tokens identified via ground-truth data preference effects,
rather than treating all tokens uniformly. By integrating these strategies,
NextQuill shifts the alignment process toward learning from causal preference
effects, facilitating more effective and personalized adaptation. Experiments
across multiple personalization benchmarks demonstrate that NextQuill
significantly improves personalization quality, offering a principled, causal
foundation for LLM personalization. Our codes are available on
https://github.com/juntaoyou/NextQuill.

### 2. [Learning Binarized Representations with Pseudo-positive Sample Enhancement for Efficient Graph Collaborative Filtering](http://arxiv.org/pdf/2506.02750v1)

Authors: Yankai Chen, Yue Que, Xinni Zhang, Chen Ma, Irwin King

Learning vectorized embeddings is fundamental to many recommender systems for
user-item matching. To enable efficient online inference, representation
binarization, which embeds latent features into compact binary sequences, has
recently shown significant promise in optimizing both memory usage and
computational overhead. However, existing approaches primarily focus on
numerical quantization, neglecting the associated information loss, which often
results in noticeable performance degradation. To address these issues, we
study the problem of graph representation binarization for efficient
collaborative filtering. Our findings indicate that explicitly mitigating
information loss at various stages of embedding binarization has a significant
positive impact on performance. Building on these insights, we propose an
enhanced framework, BiGeaR++, which specifically leverages supervisory signals
from pseudo-positive samples, incorporating both real item data and latent
embedding samples. Compared to its predecessor BiGeaR, BiGeaR++ introduces a
fine-grained inference distillation mechanism and an effective embedding sample
synthesis approach. Empirical evaluations across five real-world datasets
demonstrate that the new designs in BiGeaR++ work seamlessly well with other
modules, delivering substantial improvements of around 1%-10% over BiGeaR and
thus achieving state-of-the-art performance compared to the competing methods.
Our implementation is available at https://github.com/QueYork/BiGeaR-SS.

### 3. [UTCS: Effective Unsupervised Temporal Community Search with Pre-training of Temporal Dynamics and Subgraph Knowledge](http://arxiv.org/pdf/2506.02784v1)

Authors: Yue Zhang, Yankai Chen, Yingli Zhou, Yucan Guo, Xiaolin Han, Chenhao Ma

In many real-world applications, the evolving relationships between entities
can be modeled as temporal graphs, where each edge has a timestamp representing
the interaction time.
  As a fundamental problem in graph analysis, {\it community search (CS)} in
temporal graphs has received growing attention but exhibits two major
limitations: (1) Traditional methods typically require predefined subgraph
structures, which are not always known in advance. (2) Learning-based methods
struggle to capture temporal interaction information. To fill this research
gap, in this paper, we propose an effective \textbf{U}nsupervised
\textbf{T}emporal \textbf{C}ommunity \textbf{S}earch with pre-training of
temporal dynamics and subgraph knowledge model (\textbf{\model}).
\model~contains two key stages: offline pre-training and online search. In the
first stage, we introduce multiple learning objectives to facilitate the
pre-training process in the unsupervised learning setting. In the second stage,
we identify a candidate subgraph and compute community scores using the
pre-trained node representations and a novel scoring mechanism to determine the
final community members. Experiments on five real-world datasets demonstrate
the effectiveness.

### 4. [Combining social relations and interaction data in Recommender System with Graph Convolution Collaborative Filtering](http://arxiv.org/pdf/2506.02834v1)

Authors: Tin T. Tran, Vaclav Snasel, Loc Tan Nguyen

A recommender system is an important subject in the field of data mining,
where the item rating information from users is exploited and processed to make
suitable recommendations with all other users. The recommender system creates
convenience for e-commerce users and stimulates the consumption of items that
are suitable for users. In addition to e-commerce, a recommender system is also
used to provide recommendations on books to read, movies to watch, courses to
take or websites to visit. Similarity between users is an important impact for
recommendation, which could be calculated from the data of past user ratings of
the item by methods of collaborative filtering, matrix factorization or
singular vector decomposition. In the development of graph data mining
techniques, the relationships between users and items can be represented by
matrices from which collaborative filtering could be done with the larger
database, more accurate and faster in calculation. All these data can be
represented graphically and mined by today's highly developed graph neural
network models. On the other hand, users' social friendship data also influence
consumption habits because recommendations from friends will be considered more
carefully than information sources. However, combining a user's friend
influence and the similarity between users whose similar shopping habits is
challenging. Because the information is noisy and it affects each particular
data set in different ways. In this study, we present the input data processing
method to remove outliers which are single reviews or users with little
interaction with the items; the next proposed model will combine the social
relationship data and the similarity in the rating history of users to improve
the accuracy and recall of the recommender system.

### 5. [MMM4Rec: An Transfer-Efficient Framework for Multi-modal Sequential Recommendation](http://arxiv.org/pdf/2506.02916v1)

Authors: Hao Fan, Yanrong Hu, Kai Fang, Qingyang Liu, Hongjiu Liu

Sequential Recommendation (SR) systems model user preferences by analyzing
interaction histories. Although transferable multi-modal SR architectures
demonstrate superior performance compared to traditional ID-based approaches,
current methods incur substantial fine-tuning costs when adapting to new
domains due to complex optimization requirements and negative transfer effects
- a significant deployment bottleneck that hinders engineers from efficiently
repurposing pre-trained models for novel application scenarios with minimal
tuning overhead. We propose MMM4Rec (Multi-Modal Mamba for Sequential
Recommendation), a novel multi-modal SR framework that incorporates a dedicated
algebraic constraint mechanism for efficient transfer learning. By combining
State Space Duality (SSD)'s temporal decay properties with a time-aware
modeling design, our model dynamically prioritizes key modality information,
overcoming limitations of Transformer-based approaches. The framework
implements a constrained two-stage process: (1) sequence-level cross-modal
alignment via shared projection matrices, followed by (2) temporal fusion using
our newly designed Cross-SSD module and dual-channel Fourier adaptive
filtering. This architecture maintains semantic consistency while suppressing
noise propagation.MMM4Rec achieves rapid fine-tuning convergence with simple
cross-entropy loss, significantly improving multi-modal recommendation accuracy
while maintaining strong transferability. Extensive experiments demonstrate
MMM4Rec's state-of-the-art performance, achieving the maximum 31.78% NDCG@10
improvement over existing models and exhibiting 10 times faster average
convergence speed when transferring to large-scale downstream datasets.

### 6. [DeepShop: A Benchmark for Deep Research Shopping Agents](http://arxiv.org/pdf/2506.02839v1)

Authors: Yougang Lyu, Xiaoyu Zhang, Lingyong Yan, Maarten de Rijke, Zhaochun Ren, Xiuying Chen

Web agents for online shopping have shown great promise in automating user
interactions across e-commerce platforms. Benchmarks for assessing such agents
do not reflect the complexity of real-world shopping scenarios, as they often
consist of overly simple queries with deterministic paths, such as "Find iPhone
15." Real shopping scenarios are inherently more layered, involving
multi-dimensional product attributes, search filters, and user-specific sorting
preferences. To address this gap, we introduce DeepShop, a benchmark designed
to evaluate web agents in complex and realistic online shopping environments.
DeepShop comprises three key components. (1) Query diversity evolution:
Starting from real user queries, we generate diverse queries across five
popular online shopping domains. (2) Query complexity evolution: We further
evolve these queries to increase complexity, considering product attributes,
search filters, and sorting preferences, and classify them into three levels:
easy, medium, and hard, based on the number of evolutions. (3) Fine-grained and
holistic evaluation: We propose an automated evaluation framework that assesses
agent performance in terms of fine-grained aspects (product attributes, search
filters, and sorting preferences) and reports the overall success rate through
holistic evaluation. We conduct a systematic evaluation of retrieval-augmented
generation (RAG) methods, web agents, and deep research systems. Results show
that RAG struggles with complex queries due to its lack of web interaction,
while other methods face significant challenges with filters and sorting
preferences, leading to low overall success rates. We also perform
cross-category, complexity-based evaluations and error analyses to support the
advancement of deep research shopping agents.

### 7. [Token and Span Classification for Entity Recognition in French Historical Encyclopedias](http://arxiv.org/pdf/2506.02872v1)

Authors: Ludovic Moncla, Hédi Zeghidi

Named Entity Recognition (NER) in historical texts presents unique challenges
due to non-standardized language, archaic orthography, and nested or
overlapping entities. This study benchmarks a diverse set of NER approaches,
ranging from classical Conditional Random Fields (CRFs) and spaCy-based models
to transformer-based architectures such as CamemBERT and sequence-labeling
models like Flair. Experiments are conducted on the GeoEDdA dataset, a richly
annotated corpus derived from 18th-century French encyclopedias. We propose
framing NER as both token-level and span-level classification to accommodate
complex nested entity structures typical of historical documents. Additionally,
we evaluate the emerging potential of few-shot prompting with generative
language models for low-resource scenarios. Our results demonstrate that while
transformer-based models achieve state-of-the-art performance, especially on
nested entities, generative models offer promising alternatives when labeled
data are scarce. The study highlights ongoing challenges in historical NER and
suggests avenues for hybrid approaches combining symbolic and neural methods to
better capture the intricacies of early modern French text.

### 8. [Multilingual Information Retrieval with a Monolingual Knowledge Base](http://arxiv.org/pdf/2506.02527v1)

Authors: Yingying Zhuang, Aman Gupta, Anurag Beniwal

Multilingual information retrieval has emerged as powerful tools for
expanding knowledge sharing across languages. On the other hand, resources on
high quality knowledge base are often scarce and in limited languages,
therefore an effective embedding model to transform sentences from different
languages into a feature vector space same as the knowledge base language
becomes the key ingredient for cross language knowledge sharing, especially to
transfer knowledge available in high-resource languages to low-resource ones.
In this paper we propose a novel strategy to fine-tune multilingual embedding
models with weighted sampling for contrastive learning, enabling multilingual
information retrieval with a monolingual knowledge base. We demonstrate that
the weighted sampling strategy produces performance gains compared to standard
ones by up to 31.03\% in MRR and up to 33.98\% in Recall@3. Additionally, our
proposed methodology is language agnostic and applicable for both multilingual
and code switching use cases.

### 9. [Evaluating Named Entity Recognition Models for Russian Cultural News Texts: From BERT to LLM](http://arxiv.org/pdf/2506.02589v1)

Authors: Maria Levchenko

This paper addresses the challenge of Named Entity Recognition (NER) for
person names within the specialized domain of Russian news texts concerning
cultural events. The study utilizes the unique SPbLitGuide dataset, a
collection of event announcements from Saint Petersburg spanning 1999 to 2019.
A comparative evaluation of diverse NER models is presented, encompassing
established transformer-based architectures such as DeepPavlov, RoBERTa, and
SpaCy, alongside recent Large Language Models (LLMs) including GPT-3.5, GPT-4,
and GPT-4o. Key findings highlight the superior performance of GPT-4o when
provided with specific prompting for JSON output, achieving an F1 score of
0.93. Furthermore, GPT-4 demonstrated the highest precision at 0.99. The
research contributes to a deeper understanding of current NER model
capabilities and limitations when applied to morphologically rich languages
like Russian within the cultural heritage domain, offering insights for
researchers and practitioners. Follow-up evaluation with GPT-4.1 (April 2025)
achieves F1=0.94 for both simple and structured prompts, demonstrating rapid
progress across model families and simplified deployment requirements.

### 10. [INESC-ID @ eRisk 2025: Exploring Fine-Tuned, Similarity-Based, and Prompt-Based Approaches to Depression Symptom Identification](http://arxiv.org/pdf/2506.02924v1)

Authors: Diogo A. P. Nunes, Eugénio Ribeiro

In this work, we describe our team's approach to eRisk's 2025 Task 1: Search
for Symptoms of Depression. Given a set of sentences and the Beck's Depression
Inventory - II (BDI) questionnaire, participants were tasked with submitting up
to 1,000 sentences per depression symptom in the BDI, sorted by relevance.
Participant submissions were evaluated according to standard Information
Retrieval (IR) metrics, including Average Precision (AP) and R-Precision
(R-PREC). The provided training data, however, consisted of sentences labeled
as to whether a given sentence was relevant or not w.r.t. one of BDI's
symptoms. Due to this labeling limitation, we framed our development as a
binary classification task for each BDI symptom, and evaluated accordingly. To
that end, we split the available labeled data into training and validation
sets, and explored foundation model fine-tuning, sentence similarity, Large
Language Model (LLM) prompting, and ensemble techniques. The validation results
revealed that fine-tuning foundation models yielded the best performance,
particularly when enhanced with synthetic data to mitigate class imbalance. We
also observed that the optimal approach varied by symptom. Based on these
insights, we devised five independent test runs, two of which used ensemble
methods. These runs achieved the highest scores in the official IR evaluation,
outperforming submissions from 16 other teams.

### Machine Learning

### 1. [Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening](http://arxiv.org/pdf/2506.02355v1)

Authors: Andre He, Daniel Fried, Sean Welleck

Reinforcement learning has emerged as an effective framework for training
large language models on structured language-conditioned tasks. We identify a
critical flaw of Group Relative Policy Optimization (GRPO), a widely used RL
algorithm in this setting. For tasks that require multi-sample performance,
such as formal theorem proving, GRPO biasedly reinforces already probable
solutions and neglects rare but correct proofs. This implicit bias impairs
performance on pass@$N$ metrics at large sample sizes, limiting its
practicality for training theorem provers. To address this, we introduce the
unlikeliness reward, a straightforward method that explicitly encourages
reinforcing rare correct solutions. Additionally, we find that increasing the
number of PPO epochs further mitigates this bias. Our experiments confirm that
incorporating the unlikeliness reward significantly improves pass@$N$ across a
large range of N, outperforming standard GRPO and substantially increasing
sample diversity. Applying our revised recipe to Lean, we achieve competitive
performance with DeepSeek-Prover-V1.5-RL on the miniF2F-test benchmark. We
release our implementation, providing a simple yet effective recipe for
training formal theorem provers with RL.

### 2. [SFBD Flow: A Continuous-Optimization Framework for Training Diffusion Models with Noisy Samples](http://arxiv.org/pdf/2506.02371v1)

Authors: Haoye Lu, Darren Lo, Yaoliang Yu

Diffusion models achieve strong generative performance but often rely on
large datasets that may include sensitive content. This challenge is compounded
by the models' tendency to memorize training data, raising privacy concerns.
SFBD (Lu et al., 2025) addresses this by training on corrupted data and using
limited clean samples to capture local structure and improve convergence.
However, its iterative denoising and fine-tuning loop requires manual
coordination, making it burdensome to implement. We reinterpret SFBD as an
alternating projection algorithm and introduce a continuous variant, SFBD flow,
that removes the need for alternating steps. We further show its connection to
consistency constraint-based methods, and demonstrate that its practical
instantiation, Online SFBD, consistently outperforms strong baselines across
benchmarks.

### 3. [GAdaBoost: An Efficient and Robust AdaBoost Algorithm Based on Granular-Ball Structure](http://arxiv.org/pdf/2506.02390v1)

Authors: Qin Xie, Qinghua Zhang, Shuyin Xia, Xinran Zhou, Guoyin Wang

Adaptive Boosting (AdaBoost) faces significant challenges posed by label
noise, especially in multiclass classification tasks. Existing methods either
lack mechanisms to handle label noise effectively or suffer from high
computational costs due to redundant data usage. Inspired by granular
computing, this paper proposes granular adaptive boosting (GAdaBoost), a novel
two-stage framework comprising a data granulation stage and an adaptive
boosting stage, to enhance efficiency and robustness under noisy conditions. To
validate its feasibility, an extension of SAMME, termed GAdaBoost.SA, is
proposed. Specifically, first, a granular-ball generation method is designed to
compress data while preserving diversity and mitigating label noise. Second,
the granular ball-based SAMME algorithm focuses on granular balls rather than
individual samples, improving efficiency and reducing sensitivity to noise.
Experimental results on some noisy datasets show that the proposed approach
achieves superior robustness and efficiency compared with existing methods,
demonstrating that this work effectively extends AdaBoost and SAMME.

### 4. [Improving Generalization of Neural Combinatorial Optimization for Vehicle Routing Problems via Test-Time Projection Learning](http://arxiv.org/pdf/2506.02392v1)

Authors: Yuanyao Chen, Rongsheng Chen, Fu Luo, Zhenkun Wang

Neural Combinatorial Optimization (NCO) has emerged as a promising
learning-based paradigm for addressing Vehicle Routing Problems (VRPs) by
minimizing the need for extensive manual engineering. While existing NCO
methods, trained on small-scale instances (e.g., 100 nodes), have demonstrated
considerable success on problems of similar scale, their performance
significantly degrades when applied to large-scale scenarios. This degradation
arises from the distributional shift between training and testing data,
rendering policies learned on small instances ineffective for larger problems.
To overcome this limitation, we introduce a novel learning framework driven by
Large Language Models (LLMs). This framework learns a projection between the
training and testing distributions, which is then deployed to enhance the
scalability of the NCO model. Notably, unlike prevailing techniques that
necessitate joint training with the neural network, our approach operates
exclusively during the inference phase, obviating the need for model
retraining. Extensive experiments demonstrate that our method enables a
backbone model (trained on 100-node instances) to achieve superior performance
on large-scale Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing
Problem (CVRP) of up to 100K nodes from diverse distributions.

### 5. [Weak Supervision for Real World Graphs](http://arxiv.org/pdf/2506.02451v1)

Authors: Pratheeksha Nair, Reihaneh Rabbany

Node classification in real world graphs often suffers from label scarcity
and noise, especially in high stakes domains like human trafficking detection
and misinformation monitoring. While direct supervision is limited, such graphs
frequently contain weak signals, noisy or indirect cues, that can still inform
learning. We propose WSNET, a novel weakly supervised graph contrastive
learning framework that leverages these weak signals to guide robust
representation learning. WSNET integrates graph structure, node features, and
multiple noisy supervision sources through a contrastive objective tailored for
weakly labeled data. Across three real world datasets and synthetic benchmarks
with controlled noise, WSNET consistently outperforms state of the art
contrastive and noisy label learning methods by up to 15% in F1 score. Our
results highlight the effectiveness of contrastive learning under weak
supervision and the promise of exploiting imperfect labels in graph based
settings.

### 6. [Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](http://arxiv.org/pdf/2506.02504v1)

Authors: Xingyu Chen, Bokun Wang, Ming Yang, Quanqi Hu, Qihang Lin, Tianbao Yang

Finite-sum Coupled Compositional Optimization (FCCO), characterized by its
coupled compositional objective structure, emerges as an important optimization
paradigm for addressing a wide range of machine learning problems. In this
paper, we focus on a challenging class of non-convex non-smooth FCCO, where the
outer functions are non-smooth weakly convex or convex and the inner functions
are smooth or weakly convex. Existing state-of-the-art result face two key
limitations: (1) a high iteration complexity of $O(1/\epsilon^6)$ under the
assumption that the stochastic inner functions are Lipschitz continuous in
expectation; (2) reliance on vanilla SGD-type updates, which are not suitable
for deep learning applications. Our main contributions are two fold: (i) We
propose stochastic momentum methods tailored for non-smooth FCCO that come with
provable convergence guarantees; (ii) We establish a new state-of-the-art
iteration complexity of $O(1/\epsilon^5)$. Moreover, we apply our algorithms to
multiple inequality constrained non-convex optimization problems involving
smooth or weakly convex functional inequality constraints. By optimizing a
smoothed hinge penalty based formulation, we achieve a new state-of-the-art
complexity of $O(1/\epsilon^5)$ for finding an (nearly) $\epsilon$-level KKT
solution. Experiments on three tasks demonstrate the effectiveness of the
proposed algorithms.

### 7. [VerificAgent: Integrating Expert Knowledge and Fact-Checked Memory for Robust Domain-Specific Task Planning](http://arxiv.org/pdf/2506.02539v1)

Authors: Thong Q. Nguyen, Shubhang Desai, Yash Jain, Tanvir Aumi, Vishal Chowdhary

Continual memory augmentation allows computer-use agents (CUAs) to learn from
past interactions and refine their task-solving strategies over time. However,
unchecked memory accumulation can introduce spurious or hallucinated
"learnings" that degrade agent performance, particularly in domain-specific
workflows such as productivity software. We present a novel framework,
VerificAgent, that effectively manages memory for CUAs through (1) an
expert-curated seed of domain knowledge, (2) iterative, trajectory-based memory
refinement during training, and (3) a post-hoc fact-checking pass by human
experts to sanitize accumulated memory before deployment. On OSWorld
productivity tasks, VerificAgent achieves a 111.1% relative improvement in
success rate over baseline CUA without any additional fine-tuning.

### 8. [Privacy-Preserving Federated Convex Optimization: Balancing Partial-Participation and Efficiency via Noise Cancellation](http://arxiv.org/pdf/2506.02563v1)

Authors: Roie Reshef, Kfir Yehuda Levy

This paper tackles the challenge of achieving Differential Privacy (DP) in
Federated Learning (FL) under partial-participation, where only a subset of the
machines participate in each time-step. While previous work achieved optimal
performance in full-participation settings, these methods struggled to extend
to partial-participation scenarios. Our approach fills this gap by introducing
a novel noise-cancellation mechanism that preserves privacy without sacrificing
convergence rates or computational efficiency. We analyze our method within the
Stochastic Convex Optimization (SCO) framework and show that it delivers
optimal performance for both homogeneous and heterogeneous data distributions.
This work expands the applicability of DP in FL, offering an efficient and
practical solution for privacy-preserving learning in distributed systems with
partial participation.

### 9. [Reachability Weighted Offline Goal-conditioned Resampling](http://arxiv.org/pdf/2506.02577v1)

Authors: Wenyan Yang, Joni Pajarinen

Offline goal-conditioned reinforcement learning (RL) relies on fixed datasets
where many potential goals share the same state and action spaces. However,
these potential goals are not explicitly represented in the collected
trajectories. To learn a generalizable goal-conditioned policy, it is common to
sample goals and state-action pairs uniformly using dynamic programming methods
such as Q-learning. Uniform sampling, however, requires an intractably large
dataset to cover all possible combinations and creates many unreachable
state-goal-action pairs that degrade policy performance. Our key insight is
that sampling should favor transitions that enable goal achievement. To this
end, we propose Reachability Weighted Sampling (RWS). RWS uses a reachability
classifier trained via positive-unlabeled (PU) learning on goal-conditioned
state-action values. The classifier maps these values to a reachability score,
which is then used as a sampling priority. RWS is a plug-and-play module that
integrates seamlessly with standard offline RL algorithms. Experiments on six
complex simulated robotic manipulation tasks, including those with a robot arm
and a dexterous hand, show that RWS significantly improves performance. In one
notable case, performance on the HandBlock-Z task improved by nearly 50 percent
relative to the baseline. These results indicate the effectiveness of
reachability-weighted sampling.

### 10. [Assessing the Completeness of Traffic Scenario Categories for Automated Highway Driving Functions via Cluster-based Analysis](http://arxiv.org/pdf/2506.02599v1)

Authors: Niklas Roßberg, Marion Neumeier, Sinan Hasirlioglu, Mohamed Essayed Bouzouraa, Michael Botsch

The ability to operate safely in increasingly complex traffic scenarios is a
fundamental requirement for Automated Driving Systems (ADS). Ensuring the safe
release of ADS functions necessitates a precise understanding of the occurring
traffic scenarios. To support this objective, this work introduces a pipeline
for traffic scenario clustering and the analysis of scenario category
completeness. The Clustering Vector Quantized - Variational Autoencoder
(CVQ-VAE) is employed for the clustering of highway traffic scenarios and
utilized to create various catalogs with differing numbers of traffic scenario
categories. Subsequently, the impact of the number of categories on the
completeness considerations of the traffic scenario categories is analyzed. The
results show an outperforming clustering performance compared to previous work.
The trade-off between cluster quality and the amount of required data to
maintain completeness is discussed based on the publicly available highD
dataset.

### Neural and Evolutionary Computing

### 1. [GANORM: Lifespan Normative Modeling of EEG Network Topology based on Multinational Cross-Spectra](http://arxiv.org/pdf/2506.02566v1)

Authors: Shiang Hu, Xiaolong Huang, Yifan Hu, Xue Xiang, Xiaoliang Sheng, Debin Zhou, Pedro A. Valdes-Sosa

Charting the lifespan evolutionary trajectory of brain function serves as the
normative standard for preventing mental disorders during brain development and
aging. Although numerous MRI studies have mapped the structural connectome for
young cohorts, the EEG-based functional connectome is unknown to characterize
human lifespan, limiting its practical applications for the early detection of
brain dysfunctions at the community level. This work aimed to undertake
normative modeling from the perspective of EEG network topology.
Frequency-dependent scalp EEG functional networks were constructed based on EEG
cross-spectra aged 5-97 years from 9 countries and network characteristics were
quantified. First, GAMLSS were applied to describe the normative curves of the
network characteristics in different frequency bands. Subsequently, addressing
the limitations of existing regression approaches for whole brain network
analysis, this paper proposed an interpretable encoder-decoder framework,
Generative Age-dependent brain Network nORmative Model (GANORM). Building upon
this framework, we established an age-dependent normative trajectory of the
complete brain network for the entire lifespan. Finally, we validated the
effectiveness of the norm using EEG datasets from multiple sites. Subsequently,
we evaluated the effectiveness of GANORM, and the tested performances of BPNN
showed the R^2 was 0.796, the MAE was 0.081, and the RMSE was 0.013. Following
established lifespan brain network norm, GANORM also exhibited good results
upon verification using healthy and disease data from various sites. The
deviation scores from the normative mean for the healthy control group were
significantly smaller than those of the disease group.

### 2. [Adaptive Exploration in Lenia with Intrinsic Multi-Objective Ranking](http://arxiv.org/pdf/2506.02990v1)

Authors: Niko Lorantos, Lee Spector

Artificial life aims to understand the fundamental principles of biological
life by creating computational models that exhibit life-like properties.
Although artificial life systems show promise for simulating biological
evolution, achieving open-endedness remains a central challenge. This work
investigates mechanisms to promote exploration and unbounded innovation within
evolving populations of Lenia continuous cellular automata by evaluating
individuals against each other with respect to distinctiveness, population
sparsity, and homeostatic regulation. Multi-objective ranking of these
intrinsic fitness objectives encourages the perpetual selection of novel and
explorative individuals in sparse regions of the descriptor space without
restricting the scope of emergent behaviors. We present experiments
demonstrating the effectiveness of our multi-objective approach and emphasize
that intrinsic evolution allows diverse expressions of artificial life to
emerge. We argue that adaptive exploration improves evolutionary dynamics and
serves as an important step toward achieving open-ended evolution in artificial
systems.

### 3. [Minimal Neuron Circuits -- Part I: Resonators](http://arxiv.org/pdf/2506.02341v1)

Authors: Amr Nabil, T. Nandha Kumar, Haider Abbas F. Almurib

Spiking Neural Networks have earned increased recognition in recent years
owing to their biological plausibility and event-driven computation. Spiking
neurons are the fundamental building components of Spiking Neural Networks.
Those neurons act as computational units that determine the decision to fire an
action potential. This work presents a methodology to implement biologically
plausible yet scalable spiking neurons in hardware. We show that it is more
efficient to design neurons that mimic the $I_{Na,p}+I_{K}$ model rather than
the more complicated Hodgkin-Huxley model. We demonstrate our methodology by
presenting eleven novel minimal spiking neuron circuits in Parts I and II of
the paper. We categorize the neuron circuits presented into two types:
Resonators and Integrators. We discuss the methodology employed in designing
neurons of the resonator type in Part I, while we discuss neurons of the
integrator type in Part II. In part I, we postulate that Sodium channels
exhibit type-N negative differential resistance. Consequently, we present three
novel minimal neuron circuits that use type-N negative differential resistance
circuits or devices as the Sodium channel. Nevertheless, the aim of the paper
is not to present a set of minimal neuron circuits but rather the methodology
utilized to construct those circuits.

### 4. [Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](http://arxiv.org/pdf/2506.02813v1)

Authors: Jack Cook, Danyal Akarca, Rui Ponte Costa, Jascha Achterberg

The brain is made up of a vast set of heterogeneous regions that dynamically
organize into pathways as a function of task demands. Examples of such pathways
can be seen in the interactions between cortical and subcortical networks
during learning. This raises the question of how exactly brain regions organize
into these dynamic groups. In this work, we use an extension of the
Heterogeneous Mixture-of-Experts architecture, to show that heterogeneous
regions do not form processing pathways by themselves, implying that the brain
likely implements specific constraints which result in reliable formation of
pathways. We identify three biologically relevant inductive biases that
encourage pathway formation: a routing cost imposed on the use of more complex
regions, a scaling factor that reduces this cost when task performance is low,
and randomized expert dropout. When comparing our resulting Mixture-of-Pathways
model with the brain, we observe that the artificial pathways match how the
brain uses cortical and subcortical systems to learn and solve tasks of varying
difficulty. In summary, we introduce a novel framework for investigating how
the brain forms task-specific pathways through inductive biases which may make
Mixture-of-Experts architectures in general more adaptive.

### Networking and Internet Architecture

### 1. [AI-Driven Vehicle Condition Monitoring with Cell-Aware Edge Service Migration](http://arxiv.org/pdf/2506.02785v1)

Authors: Charalampos Kalalas, Pavol Mulinka, Guillermo Candela Belmonte, Miguel Fornell, Michail Dalgitsis, Francisco Paredes Vera, Javier Santaella Sánchez, Carmen Vicente Villares, Roshan Sedar, Eftychia Datsika, Angelos Antonopoulos, Antonio Fernández Ojea, Miquel Payaro

Artificial intelligence (AI) has been increasingly applied to the condition
monitoring of vehicular equipment, aiming to enhance maintenance strategies,
reduce costs, and improve safety. Leveraging the edge computing paradigm,
AI-based condition monitoring systems process vast streams of vehicular data to
detect anomalies and optimize operational performance. In this work, we
introduce a novel vehicle condition monitoring service that enables real-time
diagnostics of a diverse set of anomalies while remaining practical for
deployment in real-world edge environments. To address mobility challenges, we
propose a closed-loop service orchestration framework where service migration
across edge nodes is dynamically triggered by network-related metrics. Our
approach has been implemented and tested in a real-world race circuit
environment equipped with 5G network capabilities under diverse operational
conditions. Experimental results demonstrate the effectiveness of our framework
in ensuring low-latency AI inference and adaptive service placement,
highlighting its potential for intelligent transportation and mobility
applications.

### 2. [Quantum Data Centers: Why Entanglement Changes Everything](http://arxiv.org/pdf/2506.02920v1)

Authors: Angela Sara Cacciapuoti, Claudio Pellitteri, Jessica Illiano, Laura d'Avossa, Francesco Mazza, Siyi Chen, Marcello Caleffi

The Quantum Internet is key for distributed quantum computing, by
interconnecting multiple quantum processors into a virtual quantum computation
system. This allows to scale the number of qubits, by overcoming the inherent
limitations of noisy-intermediate-scale quantum (NISQ) devices. Thus, the
Quantum Internet is the foundation for large-scale, fault-tolerant quantum
computation. Among the distributed architectures, Quantum Data Centers emerge
as the most viable in the medium-term, since they integrate multiple quantum
processors within a localized network infrastructure, by allowing modular
design of quantum networking. We analyze the physical and topological
constraints of Quantum Data Centers, by emphasizing the role of entanglement
orchestrators in dynamically reconfiguring network topologies through local
operations. We examine the major hardware challenge of quantum transduction,
essential for interfacing heterogeneous quantum systems. Furthermore, we
explore how interconnecting multiple Quantum Data Centers could enable
large-scale quantum networks. We discuss the topological constraints of such a
scaling and identify open challenges, including entanglement routing and
synchronization. The carried analysis positions Quantum Data Centers as both a
practical implementation platform and strategic framework for the future
Quantum Internet.

### 3. [AI-Augmented OTDR Fault Localization Framework for Resilient Rural Fiber Networks in the United States](http://arxiv.org/pdf/2506.03041v1)

Authors: Sabab Al Farabi

This research presents a novel framework that combines traditional Optical
Time-Domain Reflectometer (OTDR) signal analysis with machine learning to
localize and classify fiber optic faults in rural broadband infrastructures.
The proposed system addresses a critical need in the expansion of middle-mile
and last-mile networks, particularly in regions targeted by the U.S. Broadband
Equity, Access, and Deployment (BEAD) Program. By enhancing fault diagnosis
through a predictive, AI-based model, this work enables proactive network
maintenance in low-resource environments. Experimental evaluations using a
controlled fiber testbed and synthetic datasets simulating rural network
conditions demonstrate that the proposed method significantly improves
detection accuracy and reduces false positives compared to conventional
thresholding techniques. The solution offers a scalable, field-deployable tool
for technicians and ISPs engaged in rural broadband deployment.

### 4. [Zero-Energy RIS-Assisted Communications With Noise Modulation and Interference-Based Energy Harvesting](http://arxiv.org/pdf/2506.02625v1)

Authors: Ahmad Massud Tota Khel, Aissa Ikhlef, Zhiguo Ding, Hongjian Sun

To advance towards carbon-neutrality and improve the limited {performance} of
conventional passive wireless communications, in this paper, we investigate the
integration of noise modulation with zero-energy reconfigurable intelligent
surfaces (RISs). In particular, the RIS reconfigurable elements (REs) are
divided into two groups: one for beamforming the desired signals in reflection
mode and another for harvesting energy from interference signals in an
absorption mode, providing the power required for RIS operation. Since the
harvested energy is a random variable, a random number of REs can beamform the
signals, while the remainder blindly reflects them. We present a closed-form
solution and a search algorithm for REs allocation, jointly optimizing both the
energy harvesting (EH) and communication performance. Considering the
repetition coding technique and discrete phase shifts, we derive analytical
expressions for the energy constrained success rate, bit error rate, optimal
threshold, mutual information, {and energy efficiency}. Numerical and
simulation results confirm the effectiveness of the algorithm and expressions,
demonstrating the superiority of the proposed integration over conventional
noise-modulation systems. It is shown that by properly allocating the REs, both
the EH and communication performance can be improved in low to moderate
interference scenarios, while the latter is restricted in the high-interference
regime.

### 5. [Computation- and Communication-Efficient Online FL for Resource-Constrained Aerial Vehicles](http://arxiv.org/pdf/2506.02972v1)

Authors: Md-Ferdous Pervej, Richeng Jin, Md Moin Uddin Chowdhury, Simran Singh, İsmail Güvenç, Huaiyu Dai

Privacy-preserving distributed machine learning (ML) and aerial connected
vehicle (ACV)-assisted edge computing have drawn significant attention lately.
Since the onboard sensors of ACVs can capture new data as they move along their
trajectories, the continual arrival of such 'newly' sensed data leads to online
learning and demands carefully crafting the trajectories. Besides, as typical
ACVs are inherently resource-constrained, computation- and
communication-efficient ML solutions are needed. Therefore, we propose a
computation- and communication-efficient online aerial federated learning
(2CEOAFL) algorithm to take the benefits of continual sensed data and limited
onboard resources of the ACVs. In particular, considering independently owned
ACVs act as selfish data collectors, we first model their trajectories
according to their respective time-varying data distributions. We then propose
a 2CEOAFL algorithm that allows the flying ACVs to (a) prune the received dense
ML model to make it shallow, (b) train the pruned model, and (c)
probabilistically quantize and offload their trained accumulated gradients to
the central server (CS). Our extensive simulation results show that the
proposed 2CEOAFL algorithm delivers comparable performances to its non-pruned
and nonquantized, hence, computation- and communication-inefficient
counterparts.

### Robotics

### 1. [SAVOR: Skill Affordance Learning from Visuo-Haptic Perception for Robot-Assisted Bite Acquisition](http://arxiv.org/pdf/2506.02353v1)

Authors: Zhanxin Wu, Bo Ai, Tom Silver, Tapomayukh Bhattacharjee

Robot-assisted feeding requires reliable bite acquisition, a challenging task
due to the complex interactions between utensils and food with diverse physical
properties. These interactions are further complicated by the temporal
variability of food properties-for example, steak becomes firm as it cools even
during a meal. To address this, we propose SAVOR, a novel approach for learning
skill affordances for bite acquisition-how suitable a manipulation skill (e.g.,
skewering, scooping) is for a given utensil-food interaction. In our
formulation, skill affordances arise from the combination of tool affordances
(what a utensil can do) and food affordances (what the food allows). Tool
affordances are learned offline through calibration, where different utensils
interact with a variety of foods to model their functional capabilities. Food
affordances are characterized by physical properties such as softness,
moisture, and viscosity, initially inferred through commonsense reasoning using
a visually-conditioned language model and then dynamically refined through
online multi-modal visuo-haptic perception using SAVOR-Net during interaction.
Our method integrates these offline and online estimates to predict skill
affordances in real time, enabling the robot to select the most appropriate
skill for each food item. Evaluated on 20 single-item foods and 10 in-the-wild
meals, our approach improves bite acquisition success by 13% over
state-of-the-art (SOTA) category-based methods (e.g. use skewer for fruits).
These results highlight the importance of modeling interaction-driven skill
affordances for generalizable and effective robot-assisted bite acquisition.
Website: https://emprise.cs.cornell.edu/savor/

### 2. [AURA: Agentic Upskilling via Reinforced Abstractions](http://arxiv.org/pdf/2506.02507v1)

Authors: Alvin Zhu, Yusuke Tanaka, Dennis Hong

We study the combinatorial explosion involved in translating high-level task
prompts into deployable control policies for agile robots through multi-stage
reinforcement learning. We introduce AURA (Agentic Upskilling via Reinforced
Abstractions), a schema-centric curriculum RL framework that leverages Large
Language Models (LLMs) as autonomous designers of multi-stage curricula. AURA
transforms user prompts into YAML workflows that encode full reward functions,
domain randomization strategies, and training configurations. All files are
statically validated against a schema before any GPU time is consumed, ensuring
reliable and efficient execution without human intervention. A
retrieval-augmented feedback loop allows specialized LLM agents to design,
execute, and refine staged curricula based on prior training results stored in
a vector database, supporting continual improvement over time. Ablation studies
highlight the importance of retrieval for curriculum quality and convergence
stability. Quantitative experiments show that AURA consistently outperforms
LLM-guided baselines on GPU-accelerated training frameworks. In qualitative
tests, AURA successfully trains end-to-end policies directly from user prompts
and deploys them zero-shot on a custom humanoid robot across a range of
environments. By abstracting away the complexity of curriculum design, AURA
enables scalable and adaptive policy learning pipelines that would be
prohibitively complex to construct by hand.

### 3. [Sign Language: Towards Sign Understanding for Robot Autonomy](http://arxiv.org/pdf/2506.02556v1)

Authors: Ayush Agrawal, Joel Loo, Nicky Zimmerman, David Hsu

Signage is an ubiquitous element of human environments, playing a critical
role in both scene understanding and navigation. For autonomous systems to
fully interpret human environments, effectively parsing and understanding signs
is essential. We introduce the task of navigational sign understanding, aimed
at extracting navigational cues from signs that convey symbolic spatial
information about the scene. Specifically, we focus on signs capturing
directional cues that point toward distant locations and locational cues that
identify specific places. To benchmark performance on this task, we curate a
comprehensive test set, propose appropriate evaluation metrics, and establish a
baseline approach. Our test set consists of over 160 images, capturing signs
with varying complexity and design across a wide range of public spaces, such
as hospitals, shopping malls, and transportation hubs. Our baseline approach
harnesses Vision-Language Models (VLMs) to parse navigational signs under these
high degrees of variability. Experiments show that VLMs offer promising
performance on this task, potentially motivating downstream applications in
robotics. The code and dataset are available on Github.

### 4. [A Hybrid Approach to Indoor Social Navigation: Integrating Reactive Local Planning and Proactive Global Planning](http://arxiv.org/pdf/2506.02593v1)

Authors: Arnab Debnath, Gregory J. Stein, Jana Kosecka

We consider the problem of indoor building-scale social navigation, where the
robot must reach a point goal as quickly as possible without colliding with
humans who are freely moving around. Factors such as varying crowd densities,
unpredictable human behavior, and the constraints of indoor spaces add
significant complexity to the navigation task, necessitating a more advanced
approach. We propose a modular navigation framework that leverages the
strengths of both classical methods and deep reinforcement learning (DRL). Our
approach employs a global planner to generate waypoints, assigning soft costs
around anticipated pedestrian locations, encouraging caution around potential
future positions of humans. Simultaneously, the local planner, powered by DRL,
follows these waypoints while avoiding collisions. The combination of these
planners enables the agent to perform complex maneuvers and effectively
navigate crowded and constrained environments while improving reliability. Many
existing studies on social navigation are conducted in simplistic or open
environments, limiting the ability of trained models to perform well in
complex, real-world settings. To advance research in this area, we introduce a
new 2D benchmark designed to facilitate development and testing of social
navigation strategies in indoor environments. We benchmark our method against
traditional and RL-based navigation strategies, demonstrating that our approach
outperforms both.

### 5. [Sight Guide: A Wearable Assistive Perception and Navigation System for the Vision Assistance Race in the Cybathlon 2024](http://arxiv.org/pdf/2506.02676v1)

Authors: Patrick Pfreundschuh, Giovanni Cioffi, Cornelius von Einem, Alexander Wyss, Hans Wernher van de Venn, Cesar Cadena, Davide Scaramuzza, Roland Siegwart, Alireza Darvishy

Visually impaired individuals face significant challenges navigating and
interacting with unknown situations, particularly in tasks requiring spatial
awareness and semantic scene understanding. To accelerate the development and
evaluate the state of technologies that enable visually impaired people to
solve these tasks, the Vision Assistance Race (VIS) at the Cybathlon 2024
competition was organized. In this work, we present Sight Guide, a wearable
assistive system designed for the VIS. The system processes data from multiple
RGB and depth cameras on an embedded computer that guides the user through
complex, real-world-inspired tasks using vibration signals and audio commands.
Our software architecture integrates classical robotics algorithms with
learning-based approaches to enable capabilities such as obstacle avoidance,
object detection, optical character recognition, and touchscreen interaction.
In a testing environment, Sight Guide achieved a 95.7% task success rate, and
further demonstrated its effectiveness during the Cybathlon competition. This
work provides detailed insights into the system design, evaluation results, and
lessons learned, and outlines directions towards a broader real-world
applicability.

### 6. [Stochastic Modeling of Road Hazards on Intersections and their Effect on Safety of Autonomous Vehicles](http://arxiv.org/pdf/2506.02688v1)

Authors: Peter Popov, Lorenzo Strigini, Cornelius Buerkle, Fabian Oboril, Michael Paulitsch

Autonomous vehicles (AV) look set to become common on our roads within the
next few years. However, to achieve the final breakthrough, not only functional
progress is required, but also satisfactory safety assurance must be provided.
Among those, a question demanding special attention is the need to assess and
quantify the overall safety of an AV. Such an assessment must consider on the
one hand the imperfections of the AV functionality and on the other hand its
interaction with the environment. In a previous paper we presented a
model-based approach to AV safety assessment in which we use a probabilistic
model to describe road hazards together with the impact on AV safety of
imperfect behavior of AV functions, such as safety monitors and perception
systems. With this model, we are able to quantify the likelihood of the
occurrence of a fatal accident, for a single operating condition. In this
paper, we extend the approach and show how the model can deal explicitly with a
set of different operating conditions defined in a given ODD.

### 7. [Efficient Tactile Perception with Soft Electrical Impedance Tomography and Pre-trained Transformer](http://arxiv.org/pdf/2506.02824v1)

Authors: Huazhi Dong, Ronald B. Liu, Sihao Teng, Delin Hu, Peisan, E, Francesco Giorgio-Serchi, Yunjie Yang

Tactile sensing is fundamental to robotic systems, enabling interactions
through physical contact in multiple tasks. Despite its importance, achieving
high-resolution, large-area tactile sensing remains challenging. Electrical
Impedance Tomography (EIT) has emerged as a promising approach for large-area,
distributed tactile sensing with minimal electrode requirements which can lend
itself to addressing complex contact problems in robotics. However, existing
EIT-based tactile reconstruction methods often suffer from high computational
costs or depend on extensive annotated simulation datasets, hindering its
viability in real-world settings. To address this shortcoming, here we propose
a Pre-trained Transformer for EIT-based Tactile Reconstruction (PTET), a
learning-based framework that bridges the simulation-to-reality gap by
leveraging self-supervised pretraining on simulation data and fine-tuning with
limited real-world data. In simulations, PTET requires 99.44 percent fewer
annotated samples than equivalent state-of-the-art approaches (2,500 vs.
450,000 samples) while achieving reconstruction performance improvements of up
to 43.57 percent under identical data conditions. Fine-tuning with real-world
data further enables PTET to overcome discrepancies between simulated and
experimental datasets, achieving superior reconstruction and detail recovery in
practical scenarios. The improved reconstruction accuracy, data efficiency, and
robustness in real-world tasks establish it as a scalable and practical
solution for tactile sensing systems in robotics, especially for object
handling and adaptive grasping under varying pressure conditions.

### 8. [High-speed control and navigation for quadrupedal robots on complex and discrete terrain](http://arxiv.org/pdf/2506.02835v1)

Authors: Hyeongjun Kim, Hyunsik Oh, Jeongsoo Park, Yunho Kim, Donghoon Youm, Moonkyu Jung, Minho Lee, Jemin Hwangbo

High-speed legged navigation in discrete and geometrically complex
environments is a challenging task because of the high-degree-of-freedom
dynamics and long-horizon, nonconvex nature of the optimization problem. In
this work, we propose a hierarchical navigation pipeline for legged robots that
can traverse such environments at high speed. The proposed pipeline consists of
a planner and tracker module. The planner module finds physically feasible
foothold plans by sampling-based optimization with fast sequential filtering
using heuristics and a neural network. Subsequently, rollouts are performed in
a physics simulation to identify the best foothold plan regarding the
engineered cost function and to confirm its physical consistency. This
hierarchical planning module is computationally efficient and physically
accurate at the same time. The tracker aims to accurately step on the target
footholds from the planning module. During the training stage, the foothold
target distribution is given by a generative model that is trained
competitively with the tracker. This process ensures that the tracker is
trained in an environment with the desired difficulty. The resulting tracker
can overcome terrains that are more difficult than what the previous methods
could manage. We demonstrated our approach using Raibo, our in-house dynamic
quadruped robot. The results were dynamic and agile motions: Raibo is capable
of running on vertical walls, jumping a 1.3-meter gap, running over stepping
stones at 4 meters per second, and autonomously navigating on terrains full of
30{\deg} ramps, stairs, and boxes of various sizes.

### 9. [Automatic Operation of an Articulated Dump Truck: State Estimation by Combined QZSS CLAS and Moving-Base RTK Using Multiple GNSS Receivers](http://arxiv.org/pdf/2506.02877v1)

Authors: Taro Suzuki, Shotaro Kojima, Kazunori Ohno, Naoto Miyamoto, Takahiro Suzuki, Kimitaka Asano, Tomohiro Komatsu, Hiroto Kakizaki

Labor shortage due to the declining birth rate has become a serious problem
in the construction industry, and automation of construction work is attracting
attention as a solution to this problem. This paper proposes a method to
realize state estimation of dump truck position, orientation and articulation
angle using multiple GNSS for automatic operation of dump trucks. RTK-GNSS is
commonly used for automation of construction equipment, but in mountainous
areas, mobile networks often unstable, and RTK-GNSS using GNSS reference
stations cannot be used. Therefore, this paper develops a state estimation
method for dump trucks that does not require a GNSS reference station by using
the Centimeter Level Augmentation Service (CLAS) of the Japanese Quasi-Zenith
Satellite System (QZSS). Although CLAS is capable of centimeter-level position
estimation, its positioning accuracy and ambiguity fix rate are lower than
those of RTK-GNSS. To solve this problem, we construct a state estimation
method by factor graph optimization that combines CLAS positioning and
moving-base RTK-GNSS between multiple GNSS antennas. Evaluation tests under
real-world environments have shown that the proposed method can estimate the
state of dump trucks with the same accuracy as conventional RTK-GNSS, but does
not require a GNSS reference station.

### 10. [Text-guided Generation of Efficient Personalized Inspection Plans](http://arxiv.org/pdf/2506.02917v1)

Authors: Xingpeng Sun, Zherong Pan, Xifeng Gao, Kui Wu, Aniket Bera

We propose a training-free, Vision-Language Model (VLM)-guided approach for
efficiently generating trajectories to facilitate target inspection planning
based on text descriptions. Unlike existing Vision-and-Language Navigation
(VLN) methods designed for general agents in unknown environments, our approach
specifically targets the efficient inspection of known scenes, with widespread
applications in fields such as medical, marine, and civil engineering.
Leveraging VLMs, our method first extracts points of interest (POIs) from the
text description, then identifies a set of waypoints from which POIs are both
salient and align with the spatial constraints defined in the prompt. Next, we
interact with the VLM to iteratively refine the trajectory, preserving the
visibility and prominence of the POIs. Further, we solve a Traveling Salesman
Problem (TSP) to find the most efficient visitation order that satisfies the
order constraint implied in the text description. Finally, we apply trajectory
optimization to generate smooth, executable inspection paths for aerial and
underwater vehicles. We have evaluated our method across a series of both
handcrafted and real-world scanned environments. The results demonstrate that
our approach effectively generates inspection planning trajectories that adhere
to user instructions.

### Software Engineering

### 1. [Toward Understanding Bugs in Vector Database Management Systems](http://arxiv.org/pdf/2506.02617v1)

Authors: Yinglin Xie, Xinyi Hou, Yanjie Zhao, Shenao Wang, Kai Chen, Haoyu Wang

Vector database management systems (VDBMSs) play a crucial role in
facilitating semantic similarity searches over high-dimensional embeddings from
diverse data sources. While VDBMSs are widely used in applications such as
recommendation, retrieval-augmented generation (RAG), and multimodal search,
their reliability remains underexplored. Traditional database reliability
models cannot be directly applied to VDBMSs because of fundamental differences
in data representation, query mechanisms, and system architecture. To address
this gap, we present the first large-scale empirical study of software defects
in VDBMSs. We manually analyzed 1,671 bug-fix pull requests from 15 widely used
open-source VDBMSs and developed a comprehensive taxonomy of bugs based on
symptoms, root causes, and developer fix strategies. Our study identifies five
categories of bug symptoms, with more than half manifesting as functional
failures. We further reveal 31 recurring fault patterns and highlight failure
modes unique to vector search systems. In addition, we summarize 12 common fix
strategies, whose distribution underscores the critical importance of correct
program logic. These findings provide actionable insights into VDBMS
reliability challenges and offer guidance for building more robust future
systems.

### 2. [Textual-Based vs. Thinging Machines Conceptual Modeling](http://arxiv.org/pdf/2506.02646v1)

Authors: Sabah Al-Fedaghi

Software engineers typically interpret the domain description in natural
language and translate it into a conceptual model. Three approaches are used in
this domain modeling: textual languages, diagrammatic languages, and a mixed
based of text and diagrams. According to some researchers, relying on a
diagrammatic notation levies certain burdens for designing large models because
visual languages are intended to depict everything diagrammatically during a
development process but fail to do so for a lack of developer efficiency. It is
claimed that textual formats enable easier manipulation in editors and tools
and facilitate the integration of ontologies in software systems. In this
paper, we explore the problem of the relationship between textual format and
diagramming in conceptual modeling. The main focus is modeling based on the
so-called thinging machine (TM). Several examples are developed in detail to
contrast side-by-side targeted domains represented in textual description and
TM modeling. A TM model is defined as a thimac (thing/machine) with a time
feature that forms dynamic events over static thimacs utilizing five generic
actions: create, process, release, transfer, and receive. This provides a
conceptual foundation that can be simplified further by eliminating the actions
of release, transfer, and receive. A multilevel reduction in the TM diagram s
complexity can also be achieved by assuming diagrammatic notations represent
the actions of creation and processing. We envision that special tools will
help improve developer efficiency. The study s results of contrasting textual
and mix-based descriptions vs. TM modeling justify our claim that TM modeling
is a more appropriate methodology than other diagrammatic schemes (e.g., UML
classes) examined in this paper.

### 3. [Computational Thinking Reasoning in Large Language Models](http://arxiv.org/pdf/2506.02658v1)

Authors: Kechi Zhang, Ge Li, Jia Li, Huangzhao Zhang, Jingjing Xu, Hao Zhu, Lecheng Wang, Jia Li, Yihong Dong, Jing Mai, Bin Gu, Zhi Jin

While large language models (LLMs) have demonstrated remarkable reasoning
capabilities, they often struggle with complex tasks that require specific
thinking paradigms, such as divide-and-conquer and procedural deduction, \etc
Previous researches integrate external, reliable tools to alleviate logical
inconsistencies and hallucinations in LLMs' problem-solving processes. However,
we argue that the root challenge is more profound: LLMs lack the complex
thinking paradigms (\ie, computational thinking) during reasoning. In this
paper, we propose Computational Thinking Model (CTM), a novel framework that
incorporates computational thinking paradigms into LLMs. This framework enables
LLMs to reformulate complex problems through decomposition, abstraction,
reduction, and simulation, among other techniques. Specifically, live code
execution is seamlessly integrated into the reasoning process, allowing CTM to
think by computing. CTM directly instills computational thinking objectives
into LLMs through tailored reinforcement learning rewards, which encourages
problem simplification, modular planning, and iterative verification. We
conduct extensive evaluations on multiple code generation and mathematical
benchmarks. The results demonstrate that CTM outperforms conventional reasoning
models and tool-augmented baselines in terms of accuracy, interpretability, and
generalizability. We hope this study offers valuable insights for AI reasoning,
where LLMs can transform problems into robust, verifiable, and scalable
computational workflows, much like computer scientists do.

### 4. [Transforming Automatically BPMN Models to Smart Contracts with Nested Collaborative Transactions (TABS+)](http://arxiv.org/pdf/2506.02727v1)

Authors: Christian Gang Liu, Peter Bodorik, Dawn Jutla

Development of blockchain smart contracts is more difficult than mainstream
software development because the underlying blockchain infrastructure poses
additional complexity. To ease the developer's task of writing smart contract,
as other research efforts, we also use Business Process Model and Notation BPMN
modeling to describe application requirements for trade of goods and services
and then transform automatically the BPMN model into the methods of a smart
contract. In our previous research we described our approach and a tool to
Transform Automatically BPMN models into Smart contracts TABS. In this paper,
we describe how the TABS approach is augmented with the support for a BPMN
collaborative transaction by several actors. Our approach analyzes the BPMN
model to determine which patterns in the BPMN model are suitable for use as
collaborative transactions. The found BPMN patterns that are suitable as
transactions are shown to the developer who decides which ones should be
deployed as collaborative transactions. We describe how our approach
automatically transform the BPMN model into smart contract the provides a
transaction mechanism to enforce the transactional properties of the nested
transactions. Our approach greatly reduces the developers task as
synchronization of collaborative activities is provided by our approach, so
that the developer needs to code only independent tasks with well-defined
inputs and outputs. We also overview the TABS+ tool we built as a proof of
concept to show that our approach is feasible. Finally, we provide estimates on
the cost of supporting the nested BPMN collaborative transactions.

### 5. [Reuse or Generate? Accelerating Code Editing via Edit-Oriented Speculative Decoding](http://arxiv.org/pdf/2506.02780v1)

Authors: Peiding Wang, Li Zhang, Fang Liu, Yinghao Zhu, Wang Xu, Lin Shi, Xiaoli Lian, Minxiao Li, Bo Shen, An Fu

Large Language Models (LLMs) have demonstrated remarkable capabilities in
code editing, substantially enhancing software development productivity.
However, the inherent complexity of code editing tasks forces existing
approaches to rely on LLMs' autoregressive end-to-end generation, where
decoding speed plays a critical role in efficiency. While inference
acceleration techniques like speculative decoding are applied to improve the
decoding efficiency, these methods fail to account for the unique
characteristics of code editing tasks where changes are typically localized and
existing code segments are reused. To address this limitation, we propose
EfficientEdit, a novel method that improves LLM-based code editing efficiency
through two key mechanisms based on speculative decoding: (1) effective reuse
of original code segments while identifying potential edit locations, and (2)
efficient generate edit content via high-quality drafts from edit-oriented
draft models and a dynamic verification mechanism that balances quality and
acceleration. Experimental results show that EfficientEdit can achieve up to
10.38$\times$ and 13.09$\times$ speedup compared to standard autoregressive
decoding in CanItEdit and CodeIF-Bench, respectively, outperforming
state-of-the-art inference acceleration approaches by up to 90.6%.

### 6. [A Multi-agent LLM-based JUit Test Generation with Strong Oracles](http://arxiv.org/pdf/2506.02943v1)

Authors: Qinghua Xu, Guancheng Wang, Lionel Briand, Kui Liu

Unit testing plays a critical role in ensuring software correctness. However,
writing unit tests manually is laborious, especially for strong typed languages
like Java, motivating the need for automated approaches. Traditional methods
primarily rely on search-based or randomized algorithms to generate tests that
achieve high code coverage and produce regression oracles, which are derived
from the program's current behavior rather than its intended functionality.
Recent advances in large language models (LLMs) have enabled oracle generation
from natural language descriptions. However, existing LLM-based methods often
require LLM fine-tuning or rely on external tools such as EvoSuite for test
prefix generation.
  In this work, we propose CANDOR, a novel end-to-end, prompt-based LLM
framework for automated JUnit test generation. CANDOR orchestrates multiple
specialized LLM agents to generate JUnit tests, including both high-quality
test prefixes and accurate oracles. To mitigate the notorious hallucinations in
LLMs, we introduce a novel strategy that engages multiple reasoning LLMs in a
panel discussion and generate accurate oracles based on consensus.
Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a
novel dual-LLM pipeline to produce concise and structured oracle evaluations.
  Our experiments on the HumanEvalJava and LeetCodeJava datasets show that
CANDOR can generate accurate oracles and is slightly better than EvoSuite in
generating tests with high line coverage and clearly superior in terms of
mutation score. Moreover, CANDOR significantly outperforms the
state-of-the-art, prompt-based test generator LLM-Empirical, achieving
improvements of 15.8 to 25.1 percentage points in oracle correctness on both
correct and faulty source code. Ablation studies confirm the critical
contributions of key agents in improving test prefix quality and oracle
accuracy.

### 7. [Towards More Effective Fault Detection in LLM-Based Unit Test Generation](http://arxiv.org/pdf/2506.02954v1)

Authors: Guancheng Wang, Qinghua Xu, Lionel C. Briand, Kui Liu

Unit tests play a vital role in uncovering potential faults in software.
While tools like EvoSuite focus on maximizing code coverage, recent advances in
large language models (LLMs) have shifted attention toward LLM-based test
generation. However, code coverage metrics -- such as line and branch coverage
-- remain overly emphasized in reported research, despite being weak indicators
of a test suite's fault-detection capability. In contrast, \textit{mutation
score} offers a more reliable and stringent measure, as demonstrated in our
findings where some test suites achieve 100\% coverage but only 4\% mutation
score. Although a few studies consider mutation score, the effectiveness of
LLMs in killing mutants remains underexplored.
  In this paper, we propose MUTGEN, a mutation-guided, LLM-based test
generation approach that incorporates mutation feedback directly into the
prompt. Evaluated on 204 subjects from two benchmarks, MUTGEN significantly
outperforms both EvoSuite and vanilla prompt-based strategies in terms of
mutation score. Furthermore, MUTGEN introduces an iterative generation
mechanism that pushes the limits of LLMs in killing additional mutants. Our
study also provide insights into the limitations of LLM-based generation,
analyzing the reasons for live and uncovered mutants, and the impact of
different mutation operators on generation effectiveness.

### 8. [A Preference-Driven Methodology for High-Quality Solidity Code Generation](http://arxiv.org/pdf/2506.03006v1)

Authors: Zhiyuan Peng, Xin Yin, Chenhao Ying, Chao Ni, Yuan Luo

While Large Language Models (LLMs) have demonstrated remarkable progress in
generating functionally correct Solidity code, they continue to face critical
challenges in producing gas-efficient and secure code, which are critical
requirements for real-world smart contract deployment. Although recent advances
leverage Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO)
for code preference alignment, existing approaches treat functional
correctness, gas optimization, and security as independent objectives,
resulting in contracts that may achieve operational soundness but suffer from
prohibitive execution costs or dangerous vulnerabilities. To address these
limitations, we propose PrefGen, a novel framework that extends standard DPO
beyond human preferences to incorporate quantifiable blockchain-specific
metrics, enabling holistic multi-objective optimization specifically tailored
for smart contract generation. Our framework introduces a comprehensive
evaluation methodology with four complementary metrics: Pass@k (functional
correctness), Compile@k (syntactic correctness), Gas@k (gas efficiency), and
Secure@k (security assessment), providing rigorous multi-dimensional contract
evaluation. Through extensive experimentation, we demonstrate that PrefGen
significantly outperforms existing approaches across all critical dimensions,
achieving 66.7% Pass@5, 58.9% Gas@5, and 62.5% Secure@5, while generating
production-ready smart contracts that are functionally correct, cost-efficient,
and secure.

### 9. [Ten Simple Rules for Catalyzing Collaborations and Building Bridges between Research Software Engineers and Software Engineering Researchers](http://arxiv.org/pdf/2506.03012v1)

Authors: Nasir U. Eisty, Jeffrey C. Carver, Johanna Cohoon, Ian A. Cosden, Carole Goble, Samuel Grayson

In the evolving landscape of scientific and scholarly research, effective
collaboration between Research Software Engineers (RSEs) and Software
Engineering Researchers (SERs) is pivotal for advancing innovation and ensuring
the integrity of computational methodologies. This paper presents ten strategic
guidelines aimed at fostering productive partnerships between these two
distinct yet complementary communities. The guidelines emphasize the importance
of recognizing and respecting the cultural and operational differences between
RSEs and SERs, proactively initiating and nurturing collaborations, and
engaging within each other's professional environments. They advocate for
identifying shared challenges, maintaining openness to emerging problems,
ensuring mutual benefits, and serving as advocates for one another.
Additionally, the guidelines highlight the necessity of vigilance in monitoring
collaboration dynamics, securing institutional support, and defining clear,
shared objectives. By adhering to these principles, RSEs and SERs can build
synergistic relationships that enhance the quality and impact of research
outcomes.

### 10. [GenFair: Systematic Test Generation for Fairness Fault Detection in Large Language Models](http://arxiv.org/pdf/2506.03024v1)

Authors: Madhusudan Srinivasan, Jubril Abdel

Large Language Models (LLMs) are increasingly deployed in critical domains,
yet they often exhibit biases inherited from training data, leading to fairness
concerns. This work focuses on the problem of effectively detecting fairness
violations, especially intersectional biases that are often missed by existing
template-based and grammar-based testing methods. Previous approaches, such as
CheckList and ASTRAEA, provide structured or grammar-driven test generation but
struggle with low test diversity and limited sensitivity to complex demographic
interactions. To address these limitations, we propose GenFair, a metamorphic
fairness testing framework that systematically generates source test cases
using equivalence partitioning, mutation operators, and boundary value
analysis. GenFair improves fairness testing by generating linguistically
diverse, realistic, and intersectional test cases. It applies metamorphic
relations (MR) to derive follow-up cases and detects fairness violations via
tone-based comparisons between source and follow-up responses. In experiments
with GPT-4.0 and LLaMA-3.0, GenFair outperformed two baseline methods. It
achieved a fault detection rate (FDR) of 0.73 (GPT-4.0) and 0.69 (LLaMA-3.0),
compared to 0.54/0.51 for template-based and 0.39/0.36 for ASTRAEA. GenFair
also showed the highest test case diversity (syntactic:10.06, semantic: 76.68)
and strong coherence (syntactic: 291.32, semantic: 0.7043), outperforming both
baselines. These results demonstrate the effectiveness of GenFair in uncovering
nuanced fairness violations. The proposed method offers a scalable and
automated solution for fairness testing and contributes to building more
equitable LLMs.

### Social and Information Networks

### 1. [Building a Recommendation System Using Amazon Product Co-Purchasing Network](http://arxiv.org/pdf/2506.02482v1)

Authors: Minghao Liu, Catherine Zhao, Nathan Zhou

This project develops an online, inductive recommendation system for newly
listed products on e-commerce platforms, focusing on suggesting relevant new
items to customers as they purchase other products. Using the Amazon Product
Co-Purchasing Network Metadata dataset, we construct a co-purchasing graph
where nodes represent products and edges capture co-purchasing relationships.
To address the challenge of recommending new products with limited information,
we apply a modified GraphSAGE method for link prediction. This inductive
approach leverages both product features and the existing co-purchasing graph
structure to predict potential co-purchasing relationships, enabling the model
to generalize to unseen products. As an online method, it updates in real time,
making it scalable and adaptive to evolving product catalogs. Experimental
results demonstrate that our approach outperforms baseline algorithms in
predicting relevant product links, offering a promising solution for enhancing
the relevance of new product recommendations in e-commerce environments. All
code is available at
https://github.com/cse416a-fl24/final-project-l-minghao_z-catherine_z-nathan.git.

### 2. [Collective Intelligence Outperforms Individual Talent: A Case Study in League of Legends](http://arxiv.org/pdf/2506.02706v1)

Authors: Angelo Josey Caldeira, Sajan Maharjan, Srijoni Majumdar, Evangelos Pournaras

Gaming environments are popular testbeds for studying human interactions and
behaviors in complex artificial intelligence systems. Particularly, in
multiplayer online battle arena (MOBA) games, individuals collaborate in
virtual environments of high realism that involves real-time strategic
decision-making and trade-offs on resource management, information collection
and sharing, team synergy and collective dynamics. This paper explores whether
collective intelligence, emerging from cooperative behaviours exhibited by a
group of individuals, who are not necessarily skillful but effectively engage
in collaborative problem-solving tasks, exceeds individual intelligence
observed within skillful individuals. This is shown via a case study in League
of Legends, using machine learning algorithms and statistical methods applied
to large-scale data collected for the same purpose. By modelling systematically
game-specific metrics but also new game-agnostic topological and graph spectra
measures of cooperative interactions, we demonstrate compelling insights about
the superior performance of collective intelligence.

### 3. [Detecting Patterns of Interaction in Temporal Hypergraphs via Edge Clustering](http://arxiv.org/pdf/2506.03105v1)

Authors: Ryan DeWolfe, François Théberge

Finding densely connected subsets of vertices in an unsupervised setting,
called clustering or community detection, is one of the fundamental problems in
network science. The edge clustering approach instead detects communities by
clustering the edges of the graph and then assigning a vertex to a community if
it has at least one edge in that community, thereby allowing for overlapping
clusters of vertices. We apply the idea behind edge clustering to temporal
hypergraphs, an extension of a graph where a single edge can contain any number
of vertices and each edge has a timestamp. Extending to hypergraphs allows for
many different patterns of interaction between edges, and by defining a
suitable structural similarity function, our edge clustering algorithm can find
clusters of these patterns. We test the algorithm with three structural
similarity functions on a large collaboration hypergraph, and find intuitive
cluster structures that could prove useful for downstream tasks.

### 4. [Random Hyperbolic Graphs with Arbitrary Mesoscale Structures](http://arxiv.org/pdf/2506.02686v1)

Authors: Stefano Guarino, Davide Torre, Enrico Mastrostefano

Real-world networks exhibit universal structural properties such as sparsity,
small-worldness, heterogeneous degree distributions, high clustering, and
community structures. Geometric network models, particularly Random Hyperbolic
Graphs (RHGs), effectively capture many of these features by embedding nodes in
a latent similarity space. However, networks are often characterized by
specific connectivity patterns between groups of nodes -- i.e. communities --
that are not geometric, in the sense that the dissimilarity between groups do
not obey the triangle inequality. Structuring connections only based on the
interplay of similarity and popularity thus poses fundamental limitations on
the mesoscale structure of the networks that RHGs can generate. To address this
limitation, we introduce the Random Hyperbolic Block Model (RHBM), which
extends RHGs by incorporating block structures within a maximum-entropy
framework. We demonstrate the advantages of the RHBM through synthetic network
analyses, highlighting its ability to preserve community structures where
purely geometric models fail. Our findings emphasize the importance of latent
geometry in network modeling while addressing its limitations in controlling
mesoscale mixing patterns.

### Systems and Control

### 1. [Unit Commitment with Cost-Oriented Temporal Resolution](http://arxiv.org/pdf/2506.02707v1)

Authors: Junyi Tao, Ran Li, Salvador Pineda

Time-adaptive unit commitment (UC) has recently been investigated to reduce
the scheduling costs by flexibly varying the temporal resolution, which is
usually determined by clustering the net load patterns. However, there exists a
misalignment between cost and net load patterns due to the discrete start-up
costs and out-of-merit-order dispatch triggered by ramping and other
constraints. The optimal time-adaptive resolution cannot be completely captured
by clustering-based method. This paper proposes a cost-oriented method to
address this misalignment by a novel bilevel optimization approach that is
efficiently solved through a heuristic greedy algorithm. The impact of varying
temporal resolution on the final scheduling costs are tested, based on which
the temporal resolution is heuristically updated, achieving significant cost
reduction without increasing the number of temporal periods. Subsequently, an
improved discretized Adam optimization method together with offline warm start
and online refinement strategy is proposed to efficiently search for the better
temporal resolution configuration. Results show that the proposed cost-oriented
UC temporal resolution determination method achieves enhanced cost efficiency.

### 2. [Recursive Privacy-Preserving Estimation Over Markov Fading Channels](http://arxiv.org/pdf/2506.02725v1)

Authors: Jie Huang, Fanlin Jia, Xiao He

In industrial applications, the presence of moving machinery, vehicles, and
personnel, contributes to the dynamic nature of the wireless channel. This time
variability induces channel fading, which can be effectively modeled using a
Markov fading channel (MFC). In this paper, we investigate the problem of
secure state estimation for systems that communicate over a MFC in the presence
of an eavesdropper. The objective is to enable a remote authorized user to
accurately estimate the states of a dynamic system, while considering the
potential interception of the sensor's packet through a wiretap channel. To
prevent information leakage, a novel co-design strategy is established, which
combines a privacy-preserving mechanism with a state estimator. To implement
our encoding scheme, a nonlinear mapping of the innovation is introduced based
on the weighted reconstructed innovation previously received by the legitimate
user. Corresponding to this encoding scheme, we design a recursive
privacy-preserving filtering algorithm to achieve accurate estimation. The
boundedness of estimation error dynamics at the legitimate user's side is
discussed and the divergence of the eavesdropper's estimation error is
analyzed, which demonstrates the effectiveness of our co-design strategy in
ensuring secrecy. Furthermore, a simulation example of a three-tank system is
provided to demonstrate the effectiveness and feasibility of our
privacy-preserving estimation method.

### 3. [Quantized Dissipative Uncertain Model for Fractional T_S Fuzzy systems with Time_Varying Delays Under Networked Control System](http://arxiv.org/pdf/2506.02788v1)

Authors: Muhammad Shamrooz Aslam, Hazrat Bilal, Sumeera Shamrooz

This paper addressed with the quantized dissipative uncertain problem for
delayed fractional T_S Fuzzy system for event_triggered networked systems
(E_NS), where the extended dissipativity analysis combines the H infinity,
dissipativity, L2 and L infinity and passivity performance in a unified frame.
To attain the high efficiency for available channel resources, measurement size
decrease mechanism and event_triggered scheme (ETS) are proposed. Firstly, we
present the ETS in which signal is transmitted through the channel with logical
function then logarithmic quantization methodology is implemented for size
reduction. Then, we transfer the original delayed fractional T_S fuzzy systems
with the effect of quantization under ETS as induced communications delays.
Furthermore, by employing the associative Lyapunov functional method in terms
of linear matrix inequalities, adequate conditions for asymptotical stability
is given. Moreover, we also construct the design fuzzy model for state space
filtering system. At last, a truck_trailer model is given to show the
effectiveness of the proposed strategy.

### 4. [Target Sensing Performance in Disaster-Specific ISAC Networks](http://arxiv.org/pdf/2506.02828v1)

Authors: Ahmet Burak Ozyurt, John S. Thompson

As sixth-generation (6G) wireless technology emerges, integrated sensing and
communication (ISAC) networks offer significant potential for enhancing
real-time monitoring in disaster areas. However, existing ISAC approaches often
fail to address the unique challenges of dynamic and cluttered disaster areas,
resulting in limited sensing coverage and interruptions in sensing service. To
address these limitations, this work proposes a mobile ISAC network
specifically designed for disaster scenarios. By leveraging stochastic
geometry, we derive closed-form expressions for sensing coverage and introduce
a novel performance metric to evaluate sensing service continuity. Simulation
results validate the analytical derivations and offer key insights into network
design.

### 5. [Dynamic real-time multi-UAV cooperative mission planning method under multiple constraints](http://arxiv.org/pdf/2506.02365v1)

Authors: Chenglou Liu, Yufeng Lu, Fangfang Xie, Tingwei Ji, Yao Zheng

As UAV popularity soars, so does the mission planning associated with it. The
classical approaches suffer from the triple problems of decoupled of task
assignment and path planning, poor real-time performance and limited
adaptability. Aiming at these challenges, this paper proposes a dynamic
real-time multi-UAV collaborative mission planning algorithm based on Dubins
paths under a distributed formation structure. Dubins path with multiple
advantages bridges the gap between task assignment and path planning, leading
to a coupled solution for mission planning. Then, a series of acceleration
techniques, task clustering preprocessing, highly efficient distance cost
functions, low-complexity and less iterative task allocation strategies, are
employed to guarantee the real-time performance of the algorithms. To cope with
different emergencies and their simultaneous extremes, real-time planning of
emerging tasks and mission replanning due to the reduction of available UAVs
are appropriately handled. Finally, the developed algorithm is comprehensively
exemplified and studied through simulations, highlighting that the proposed
method only sacrifices 9.57% of the path length, while achieving a speed
improvement of 4-5 orders of magnitude over the simulated annealing method,
with a single mission planning of about 0.0003s.

### 6. [Geometric Visual Servo Via Optimal Transport](http://arxiv.org/pdf/2506.02768v1)

Authors: Ethan Canzini, Simon Pope, Ashutosh Tiwari

When developing control laws for robotic systems, the principle factor when
examining their performance is choosing inputs that allow smooth tracking to a
reference input. In the context of robotic manipulation, this involves
translating an object or end-effector from an initial pose to a target pose.
Robotic manipulation control laws frequently use vision systems as an error
generator to track features and produce control inputs. However, current
control algorithms don't take into account the probabilistic features that are
extracted and instead rely on hand-tuned feature extraction methods.
Furthermore, the target features can exist in a static pose thus allowing a
combined pose and feature error for control generation. We present a geometric
control law for the visual servoing problem for robotic manipulators. The input
from the camera constitutes a probability measure on the 3-dimensional Special
Euclidean task-space group, where the Wasserstein distance between the current
and desired poses is analogous with the geometric geodesic. From this, we
develop a controller that allows for both pose and image-based visual servoing
by combining classical PD control with gravity compensation with error
minimization through the use of geodesic flows on a 3-dimensional Special
Euclidean group. We present our results on a set of test cases demonstrating
the generalisation ability of our approach to a variety of initial positions.

### 7. [On dual-rate consensus under transmission delays](http://arxiv.org/pdf/2506.02840v1)

Authors: David Umsonst, Mina Ferizbegovic

In this paper, we investigate the problem of dual-rate consensus under
transmission delays, where the control updates happen at a faster rate than the
measurements being received. We assume that the measurements are delayed by a
fixed delay and show that for all delays and rates, the system reaches a
consensus if and only if the communication graph of the agents is connected and
the control gain is chosen in a specific interval. Based on these results we
dive deeper into the convergence properties and investigate how the convergence
changes when we change the rate for sending measurements. We observe that in
certain cases there exists a sweet spot for choosing the sampling rate of the
measurements, which can improve the convergence to the consensus point. We then
formulate an optimization problem to find a sampling rate to improve the
convergence speed and provide a necessary and sufficient condition for the
existence of a finite optimizer of this problem. Our results are verified with
numerical simulations.

### 8. [Ensemble-MIX: Enhancing Sample Efficiency in Multi-Agent RL Using Ensemble Methods](http://arxiv.org/pdf/2506.02841v1)

Authors: Tom Danino, Nahum Shimkin

Multi-agent reinforcement learning (MARL) methods have achieved
state-of-the-art results on a range of multi-agent tasks. Yet, MARL algorithms
typically require significantly more environment interactions than their
single-agent counterparts to converge, a problem exacerbated by the difficulty
in exploring over a large joint action space and the high variance intrinsic to
MARL environments. To tackle these issues, we propose a novel algorithm that
combines a decomposed centralized critic with decentralized ensemble learning,
incorporating several key contributions. The main component in our scheme is a
selective exploration method that leverages ensemble kurtosis. We extend the
global decomposed critic with a diversity-regularized ensemble of individual
critics and utilize its excess kurtosis to guide exploration toward
high-uncertainty states and actions. To improve sample efficiency, we train the
centralized critic with a novel truncated variation of the TD($\lambda$)
algorithm, enabling efficient off-policy learning with reduced variance. On the
actor side, our suggested algorithm adapts the mixed samples approach to MARL,
mixing on-policy and off-policy loss functions for training the actors. This
approach balances between stability and efficiency and outperforms purely
off-policy learning. The evaluation shows our method outperforms
state-of-the-art baselines on standard MARL benchmarks, including a variety of
SMAC II maps.

### 9. [CLONE: Customizing LLMs for Efficient Latency-Aware Inference at the Edge](http://arxiv.org/pdf/2506.02847v1)

Authors: Chunlin Tian, Xinpeng Qin, Kahou Tam, Li Li, Zijian Wang, Yuanzhe Zhao, Minglei Zhang, Chengzhong Xu

Deploying large language models (LLMs) on edge devices is crucial for
delivering fast responses and ensuring data privacy. However, the limited
storage, weight, and power of edge devices make it difficult to deploy
LLM-powered applications. These devices must balance latency requirements with
energy consumption and model accuracy. In this paper, we first quantify the
challenges of deploying LLMs on off-the-shelf edge devices and then we present
CLONE, an in-depth algorithm-hardware co-design at both the model- and
system-level that intelligently integrates real-time, energy optimization while
maintaining robust generality. In order to maximize the synergistic benefits of
these algorithms in always-on and intermediate edge computing settings, we
specialize in a 28nm scalable hardware accelerator system. We implement and
extensively evaluate CLONE on two off-the-shelf edge platforms. Experiments
show that CLONE effectively accelerates the inference process up to 11.92x, and
saves energy up to 7.36x, while maintaining high-generation.

### 10. [Computation- and Communication-Efficient Online FL for Resource-Constrained Aerial Vehicles](http://arxiv.org/pdf/2506.02972v1)

Authors: Md-Ferdous Pervej, Richeng Jin, Md Moin Uddin Chowdhury, Simran Singh, İsmail Güvenç, Huaiyu Dai

Privacy-preserving distributed machine learning (ML) and aerial connected
vehicle (ACV)-assisted edge computing have drawn significant attention lately.
Since the onboard sensors of ACVs can capture new data as they move along their
trajectories, the continual arrival of such 'newly' sensed data leads to online
learning and demands carefully crafting the trajectories. Besides, as typical
ACVs are inherently resource-constrained, computation- and
communication-efficient ML solutions are needed. Therefore, we propose a
computation- and communication-efficient online aerial federated learning
(2CEOAFL) algorithm to take the benefits of continual sensed data and limited
onboard resources of the ACVs. In particular, considering independently owned
ACVs act as selfish data collectors, we first model their trajectories
according to their respective time-varying data distributions. We then propose
a 2CEOAFL algorithm that allows the flying ACVs to (a) prune the received dense
ML model to make it shallow, (b) train the pruned model, and (c)
probabilistically quantize and offload their trained accumulated gradients to
the central server (CS). Our extensive simulation results show that the
proposed 2CEOAFL algorithm delivers comparable performances to its non-pruned
and nonquantized, hence, computation- and communication-inefficient
counterparts.

### Machine Learning (Statistics Category)

### 1. [Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](http://arxiv.org/pdf/2506.02336v1)

Authors: Jingfeng Wu, Pierre Marion, Peter Bartlett

We study gradient descent (GD) with a constant stepsize for
$\ell_2$-regularized logistic regression with linearly separable data.
Classical theory suggests small stepsizes to ensure monotonic reduction of the
optimization objective, achieving exponential convergence in
$\widetilde{\mathcal{O}}(\kappa)$ steps with $\kappa$ being the condition
number. Surprisingly, we show that this can be accelerated to
$\widetilde{\mathcal{O}}(\sqrt{\kappa})$ by simply using a large stepsize --
for which the objective evolves nonmonotonically. The acceleration brought by
large stepsizes extends to minimizing the population risk for separable
distributions, improving on the best-known upper bounds on the number of steps
to reach a near-optimum. Finally, we characterize the largest stepsize for the
local convergence of GD, which also determines the global convergence in
special scenarios. Our results extend the analysis of Wu et al. (2024) from
convex settings with minimizers at infinity to strongly convex cases with
finite minimizers.

### 2. [Multi-agent Markov Entanglement](http://arxiv.org/pdf/2506.02385v1)

Authors: Shuze Chen, Tianyi Peng

Value decomposition has long been a fundamental technique in multi-agent
dynamic programming and reinforcement learning (RL). Specifically, the value
function of a global state $(s_1,s_2,\ldots,s_N)$ is often approximated as the
sum of local functions: $V(s_1,s_2,\ldots,s_N)\approx\sum_{i=1}^N V_i(s_i)$.
This approach traces back to the index policy in restless multi-armed bandit
problems and has found various applications in modern RL systems. However, the
theoretical justification for why this decomposition works so effectively
remains underexplored.
  In this paper, we uncover the underlying mathematical structure that enables
value decomposition. We demonstrate that a multi-agent Markov decision process
(MDP) permits value decomposition if and only if its transition matrix is not
"entangled" -- a concept analogous to quantum entanglement in quantum physics.
Drawing inspiration from how physicists measure quantum entanglement, we
introduce how to measure the "Markov entanglement" for multi-agent MDPs and
show that this measure can be used to bound the decomposition error in general
multi-agent MDPs.
  Using the concept of Markov entanglement, we proved that a widely-used class
of index policies is weakly entangled and enjoys a sublinear $\mathcal
O(\sqrt{N})$ scale of decomposition error for $N$-agent systems. Finally, we
show how Markov entanglement can be efficiently estimated in practice,
providing practitioners with an empirical proxy for the quality of value
decomposition.

### 3. [Tensor State Space-based Dynamic Multilayer Network Modeling](http://arxiv.org/pdf/2506.02413v1)

Authors: Tian Lan, Jie Guo, Chen Zhang

Understanding the complex interactions within dynamic multilayer networks is
critical for advancements in various scientific domains. Existing models often
fail to capture such networks' temporal and cross-layer dynamics. This paper
introduces a novel Tensor State Space Model for Dynamic Multilayer Networks
(TSSDMN), utilizing a latent space model framework. TSSDMN employs a symmetric
Tucker decomposition to represent latent node features, their interaction
patterns, and layer transitions. Then by fixing the latent features and
allowing the interaction patterns to evolve over time, TSSDMN uniquely captures
both the temporal dynamics within layers and across different layers. The model
identifiability conditions are discussed. By treating latent features as
variables whose posterior distributions are approximated using a mean-field
variational inference approach, a variational Expectation Maximization
algorithm is developed for efficient model inference. Numerical simulations and
case studies demonstrate the efficacy of TSSDMN for understanding dynamic
multilayer networks.

### 4. [Asymptotics of SGD in Sequence-Single Index Models and Single-Layer Attention Networks](http://arxiv.org/pdf/2506.02651v1)

Authors: Luca Arnaboldi, Bruno Loureiro, Ludovic Stephan, Florent Krzakala, Lenka Zdeborova

We study the dynamics of stochastic gradient descent (SGD) for a class of
sequence models termed Sequence Single-Index (SSI) models, where the target
depends on a single direction in input space applied to a sequence of tokens.
This setting generalizes classical single-index models to the sequential
domain, encompassing simplified one-layer attention architectures. We derive a
closed-form expression for the population loss in terms of a pair of sufficient
statistics capturing semantic and positional alignment, and characterize the
induced high-dimensional SGD dynamics for these coordinates. Our analysis
reveals two distinct training phases: escape from uninformative initialization
and alignment with the target subspace, and demonstrates how the sequence
length and positional encoding influence convergence speed and learning
trajectories. These results provide a rigorous and interpretable foundation for
understanding how sequential structure in data can be beneficial for learning
with attention-based models.

### 5. [Symmetry-Aware GFlowNets](http://arxiv.org/pdf/2506.02685v1)

Authors: Hohyun Kim, Seunggeun Lee, Min-hwan Oh

Generative Flow Networks (GFlowNets) offer a powerful framework for sampling
graphs in proportion to their rewards. However, existing approaches suffer from
systematic biases due to inaccuracies in state transition probability
computations. These biases, rooted in the inherent symmetries of graphs, impact
both atom-based and fragment-based generation schemes. To address this
challenge, we introduce Symmetry-Aware GFlowNets (SA-GFN), a method that
incorporates symmetry corrections into the learning process through reward
scaling. By integrating bias correction directly into the reward structure,
SA-GFN eliminates the need for explicit state transition computations.
Empirical results show that SA-GFN enables unbiased sampling while enhancing
diversity and consistently generating high-reward graphs that closely match the
target distribution.

### 6. [Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport](http://arxiv.org/pdf/2506.02712v1)

Authors: Jayadev Naram, Fredrik Hellström, Ziming Wang, Rebecka Jörnsten, Giuseppe Durisi

In many scenarios of practical interest, labeled data from a target
distribution are scarce while labeled data from a related source distribution
are abundant. One particular setting of interest arises when the target label
space is a subset of the source label space, leading to the framework of
partial domain adaptation (PDA). Typical approaches to PDA involve minimizing a
domain alignment term and a weighted empirical loss on the source data, with
the aim of transferring knowledge between domains. However, a theoretical basis
for this procedure is lacking, and in particular, most existing weighting
schemes are heuristic. In this work, we derive generalization bounds for the
PDA problem based on partial optimal transport. These bounds corroborate the
use of the partial Wasserstein distance as a domain alignment term, and lead to
theoretically motivated explicit expressions for the empirical source loss
weights. Inspired by these bounds, we devise a practical algorithm for PDA,
termed WARMPOT. Through extensive numerical experiments, we show that WARMPOT
is competitive with recent approaches, and that our proposed weights improve on
existing schemes.

### 7. [Safely Learning Controlled Stochastic Dynamics](http://arxiv.org/pdf/2506.02754v1)

Authors: Luc Brogat-Motte, Alessandro Rudi, Riccardo Bonalli

We address the problem of safely learning controlled stochastic dynamics from
discrete-time trajectory observations, ensuring system trajectories remain
within predefined safe regions during both training and deployment.
Safety-critical constraints of this kind are crucial in applications such as
autonomous robotics, finance, and biomedicine. We introduce a method that
ensures safe exploration and efficient estimation of system dynamics by
iteratively expanding an initial known safe control set using kernel-based
confidence bounds. After training, the learned model enables predictions of the
system's dynamics and permits safety verification of any given control. Our
approach requires only mild smoothness assumptions and access to an initial
safe control set, enabling broad applicability to complex real-world systems.
We provide theoretical guarantees for safety and derive adaptive learning rates
that improve with increasing Sobolev regularity of the true dynamics.
Experimental evaluations demonstrate the practical effectiveness of our method
in terms of safety, estimation accuracy, and computational efficiency.

### 8. [Doubly-Robust Estimation of Counterfactual Policy Mean Embeddings](http://arxiv.org/pdf/2506.02793v1)

Authors: Houssam Zenati, Bariscan Bozkurt, Arthur Gretton

Estimating the distribution of outcomes under counterfactual policies is
critical for decision-making in domains such as recommendation, advertising,
and healthcare. We analyze a novel framework-Counterfactual Policy Mean
Embedding (CPME)-that represents the entire counterfactual outcome distribution
in a reproducing kernel Hilbert space (RKHS), enabling flexible and
nonparametric distributional off-policy evaluation. We introduce both a plug-in
estimator and a doubly robust estimator; the latter enjoys improved uniform
convergence rates by correcting for bias in both the outcome embedding and
propensity models. Building on this, we develop a doubly robust kernel test
statistic for hypothesis testing, which achieves asymptotic normality and thus
enables computationally efficient testing and straightforward construction of
confidence intervals. Our framework also supports sampling from the
counterfactual distribution. Numerical simulations illustrate the practical
benefits of CPME over existing methods.

### 9. [Asymptotically perfect seeded graph matching without edge correlation (and applications to inference)](http://arxiv.org/pdf/2506.02825v1)

Authors: Tong Qi, Vera Andersson, Peter Viechnicki, Vince Lyzinski

We present the OmniMatch algorithm for seeded multiple graph matching. In the
setting of $d$-dimensional Random Dot Product Graphs (RDPG), we prove that
under mild assumptions, OmniMatch with $s$ seeds asymptotically and efficiently
perfectly aligns $O(s^{\alpha})$ unseeded vertices -- for $\alpha<2\wedge d/4$
-- across multiple networks even in the presence of no edge correlation. We
demonstrate the effectiveness of our algorithm across numerous simulations and
in the context of shuffled graph hypothesis testing. In the shuffled testing
setting, testing power is lost due to the misalignment/shuffling of vertices
across graphs, and we demonstrate the capacity of OmniMatch to correct for
misaligned vertices prior to testing and hence recover the lost testing power.
We further demonstrate the algorithm on a pair of data examples from
connectomics and machine translation.

### 10. [The Limits of Predicting Agents from Behaviour](http://arxiv.org/pdf/2506.02923v1)

Authors: Alexis Bellot, Jonathan Richens, Tom Everitt

As the complexity of AI systems and their interactions with the world
increases, generating explanations for their behaviour is important for safely
deploying AI. For agents, the most natural abstractions for predicting
behaviour attribute beliefs, intentions and goals to the system. If an agent
behaves as if it has a certain goal or belief, then we can make reasonable
predictions about how it will behave in novel situations, including those where
comprehensive safety evaluations are untenable. How well can we infer an
agent's beliefs from their behaviour, and how reliably can these inferred
beliefs predict the agent's behaviour in novel situations? We provide a precise
answer to this question under the assumption that the agent's behaviour is
guided by a world model. Our contribution is the derivation of novel bounds on
the agent's behaviour in new (unseen) deployment environments, which represent
a theoretical limit for predicting intentional agents from behavioural data
alone. We discuss the implications of these results for several research areas
including fairness and safety.

