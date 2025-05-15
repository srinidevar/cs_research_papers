# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-14 18:02:56.796229 PST.

### Artificial Intelligence

### 1. [Foundation Models Knowledge Distillation For Battery Capacity Degradation Forecast](http://arxiv.org/pdf/2505.08151v1)

Authors: Joey Chan, Zhen Chen, Ershun Pan

Accurate estimation of lithium-ion battery capacity degradation is critical
for enhancing the reliability and safety of battery operations. Traditional
expert models, tailored to specific scenarios, provide isolated estimations.
With the rapid advancement of data-driven techniques, a series of
general-purpose time-series foundation models have been developed. However,
foundation models specifically designed for battery capacity degradation remain
largely unexplored. To enable zero-shot generalization in battery degradation
prediction using large model technology, this study proposes a
degradation-aware fine-tuning strategy for time-series foundation models. We
apply this strategy to fine-tune the Timer model on approximately 10 GB of
open-source battery charge discharge data. Validation on our released
CycleLife-SJTUIE dataset demonstrates that the fine-tuned Battery-Timer
possesses strong zero-shot generalization capability in capacity degradation
forecasting. To address the computational challenges of deploying large models,
we further propose a knowledge distillation framework that transfers the
knowledge of pre-trained foundation models into compact expert models.
Distillation results across several state-of-the-art time-series expert models
confirm that foundation model knowledge significantly improves the
multi-condition generalization of expert models.

### 2. [Efficient and Scalable Neural Symbolic Search for Knowledge Graph Complex Query Answering](http://arxiv.org/pdf/2505.08155v1)

Authors: Weizhi Fei, Zihao Wang, hang Yin, Shukai Zhao, Wei Zhang, Yangqiu Song

Complex Query Answering (CQA) aims to retrieve answer sets for complex
logical formulas from incomplete knowledge graphs, which is a crucial yet
challenging task in knowledge graph reasoning. While neuro-symbolic search
utilized neural link predictions achieve superior accuracy, they encounter
significant complexity bottlenecks: (i) Data complexity typically scales
quadratically with the number of entities in the knowledge graph, and (ii)
Query complexity becomes NP-hard for cyclic queries. Consequently, these
approaches struggle to effectively scale to larger knowledge graphs and more
complex queries. To address these challenges, we propose an efficient and
scalable symbolic search framework. First, we propose two constraint strategies
to compute neural logical indices to reduce the domain of variables, thereby
decreasing the data complexity of symbolic search. Additionally, we introduce
an approximate algorithm based on local search to tackle the NP query
complexity of cyclic queries. Experiments on various CQA benchmarks demonstrate
that our framework reduces the computational load of symbolic methods by 90\%
while maintaining nearly the same performance, thus alleviating both efficiency
and scalability issues.

### 3. [Behind the Noise: Conformal Quantile Regression Reveals Emergent Representations](http://arxiv.org/pdf/2505.08176v1)

Authors: Petrus H. Zwart, Tamas Varga, Odeta Qafoku, James A. Sethian

Scientific imaging often involves long acquisition times to obtain
high-quality data, especially when probing complex, heterogeneous systems.
However, reducing acquisition time to increase throughput inevitably introduces
significant noise into the measurements. We present a machine learning approach
that not only denoises low-quality measurements with calibrated uncertainty
bounds, but also reveals emergent structure in the latent space. By using
ensembles of lightweight, randomly structured neural networks trained via
conformal quantile regression, our method performs reliable denoising while
uncovering interpretable spatial and chemical features -- without requiring
labels or segmentation. Unlike conventional approaches focused solely on image
restoration, our framework leverages the denoising process itself to drive the
emergence of meaningful representations. We validate the approach on real-world
geobiochemical imaging data, showing how it supports confident interpretation
and guides experimental design under resource constraints.

### 4. [Evaluating LLM Metrics Through Real-World Capabilities](http://arxiv.org/pdf/2505.08253v1)

Authors: Justin K Miller, Wenjia Tang

As generative AI becomes increasingly embedded in everyday workflows, it is
important to evaluate its performance in ways that reflect real-world usage
rather than abstract notions of intelligence. Unlike many existing benchmarks
that assess general intelligence, our approach focuses on real-world utility,
evaluating how well models support users in everyday tasks. While current
benchmarks emphasize code generation or factual recall, users rely on AI for a
much broader range of activities-from writing assistance and summarization to
citation formatting and stylistic feedback. In this paper, we analyze
large-scale survey data and usage logs to identify six core capabilities that
represent how people commonly use Large Language Models (LLMs): Summarization,
Technical Assistance, Reviewing Work, Data Structuring, Generation, and
Information Retrieval. We then assess the extent to which existing benchmarks
cover these capabilities, revealing significant gaps in coverage, efficiency
measurement, and interpretability. Drawing on this analysis, we use
human-centered criteria to identify gaps in how well current benchmarks reflect
common usage that is grounded in five practical criteria: coherence, accuracy,
clarity, relevance, and efficiency. For four of the six capabilities, we
identify the benchmarks that best align with real-world tasks and use them to
compare leading models. We find that Google Gemini outperforms other
models-including OpenAI's GPT, xAI's Grok, Meta's LLaMA, Anthropic's Claude,
DeepSeek, and Qwen from Alibaba-on these utility-focused metrics.

### 5. [An Identifiable Cost-Aware Causal Decision-Making Framework Using Counterfactual Reasoning](http://arxiv.org/pdf/2505.08343v1)

Authors: Ruichu Cai, Xi Chen, Jie Qiao, Zijian Li, Yuequn Liu, Wei Chen, Keli Zhang, Jiale Zheng

Decision making under abnormal conditions is a critical process that involves
evaluating the current state and determining the optimal action to restore the
system to a normal state at an acceptable cost. However, in such scenarios,
existing decision-making frameworks highly rely on reinforcement learning or
root cause analysis, resulting in them frequently neglecting the cost of the
actions or failing to incorporate causal mechanisms adequately. By relaxing the
existing causal decision framework to solve the necessary cause, we propose a
minimum-cost causal decision (MiCCD) framework via counterfactual reasoning to
address the above challenges. Emphasis is placed on making counterfactual
reasoning processes identifiable in the presence of a large amount of mixed
anomaly data, as well as finding the optimal intervention state in a continuous
decision space. Specifically, it formulates a surrogate model based on causal
graphs, using abnormal pattern clustering labels as supervisory signals. This
enables the approximation of the structural causal model among the variables
and lays a foundation for identifiable counterfactual reasoning. With the
causal structure approximated, we then established an optimization model based
on counterfactual estimation. The Sequential Least Squares Programming (SLSQP)
algorithm is further employed to optimize intervention strategies while taking
costs into account. Experimental evaluations on both synthetic and real-world
datasets reveal that MiCCD outperforms conventional methods across multiple
metrics, including F1-score, cost efficiency, and ranking quality(nDCG@k
values), thus validating its efficacy and broad applicability.

### 6. [Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning](http://arxiv.org/pdf/2505.08361v1)

Authors: Xinyue Wang, Biwei Huang

Generalization in reinforcement learning (RL) remains a significant
challenge, especially when agents encounter novel environments with unseen
dynamics. Drawing inspiration from human compositional reasoning -- where known
components are reconfigured to handle new situations -- we introduce World
Modeling with Compositional Causal Components (WM3C). This novel framework
enhances RL generalization by learning and leveraging compositional causal
components. Unlike previous approaches focusing on invariant representation
learning or meta-learning, WM3C identifies and utilizes causal dynamics among
composable elements, facilitating robust adaptation to new tasks. Our approach
integrates language as a compositional modality to decompose the latent space
into meaningful components and provides theoretical guarantees for their unique
identification under mild assumptions. Our practical implementation uses a
masked autoencoder with mutual information constraints and adaptive sparsity
regularization to capture high-level semantic information and effectively
disentangle transition dynamics. Experiments on numerical simulations and
real-world robotic manipulation tasks demonstrate that WM3C significantly
outperforms existing methods in identifying latent processes, improving policy
learning, and generalizing to unseen tasks.

### 7. [Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation](http://arxiv.org/pdf/2505.08364v1)

Authors: Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, Qianchun Lu

Despite impressive progress in areas like mathematical reasoning, large
language models still face significant challenges in consistently solving
complex problems. Drawing inspiration from key human learning strategies, we
propose two novel strategies to enhance the capability of large language models
to solve these complex problems. First, Adaptive Difficulty Curriculum Learning
(ADCL) is a novel curriculum learning strategy that tackles the Difficulty
Shift phenomenon (i.e., a model's perception of problem difficulty dynamically
changes during training) by periodically re-estimating difficulty within
upcoming data batches to maintain alignment with the model's evolving
capabilities. Second, Expert-Guided Self-Reformulation (EGSR) is a novel
reinforcement learning strategy that bridges the gap between imitation learning
and pure exploration by guiding models to reformulate expert solutions within
their own conceptual framework, rather than relying on direct imitation,
fostering deeper understanding and knowledge assimilation. Extensive
experiments on challenging mathematical reasoning benchmarks, using Qwen2.5-7B
as the base model, demonstrate that these human-inspired strategies
synergistically and significantly enhance performance. Notably, their combined
application improves performance over the standard Zero-RL baseline by 10% on
the AIME24 benchmark and 16.6% on AIME25.

### 8. [Explaining Autonomous Vehicles with Intention-aware Policy Graphs](http://arxiv.org/pdf/2505.08404v1)

Authors: Sara Montese, Victor Gimenez-Abalos, Atia Cortés, Ulises Cortés, Sergio Alvarez-Napagao

The potential to improve road safety, reduce human driving error, and promote
environmental sustainability have enabled the field of autonomous driving to
progress rapidly over recent decades. The performance of autonomous vehicles
has significantly improved thanks to advancements in Artificial Intelligence,
particularly Deep Learning. Nevertheless, the opacity of their decision-making,
rooted in the use of accurate yet complex AI models, has created barriers to
their societal trust and regulatory acceptance, raising the need for
explainability. We propose a post-hoc, model-agnostic solution to provide
teleological explanations for the behaviour of an autonomous vehicle in urban
environments. Building on Intention-aware Policy Graphs, our approach enables
the extraction of interpretable and reliable explanations of vehicle behaviour
in the nuScenes dataset from global and local perspectives. We demonstrate the
potential of these explanations to assess whether the vehicle operates within
acceptable legal boundaries and to identify possible vulnerabilities in
autonomous driving datasets and models.

### 9. [Agent-as-a-Service based on Agent Network](http://arxiv.org/pdf/2505.08446v1)

Authors: Yuhan Zhu, Haojie Liu, Jian Wang, Bing Li, Zikang Yin, Yefei Liao

The rise of large model-based AI agents has spurred interest in Multi-Agent
Systems (MAS) for their capabilities in decision-making, collaboration, and
adaptability. While the Model Context Protocol (MCP) addresses tool invocation
and data exchange challenges via a unified protocol, it lacks support for
organizing agent-level collaboration. To bridge this gap, we propose
Agent-as-a-Service based on Agent Network (AaaS-AN), a service-oriented
paradigm grounded in the Role-Goal-Process-Service (RGPS) standard. AaaS-AN
unifies the entire agent lifecycle, including construction, integration,
interoperability, and networked collaboration, through two core components: (1)
a dynamic Agent Network, which models agents and agent groups as vertexes that
self-organize within the network based on task and role dependencies; (2)
service-oriented agents, incorporating service discovery, registration, and
interoperability protocols. These are orchestrated by a Service Scheduler,
which leverages an Execution Graph to enable distributed coordination, context
tracking, and runtime task management. We validate AaaS-AN on mathematical
reasoning and application-level code generation tasks, which outperforms
state-of-the-art baselines. Notably, we constructed a MAS based on AaaS-AN
containing agent groups, Robotic Process Automation (RPA) workflows, and MCP
servers over 100 agent services. We also release a dataset containing 10,000
long-horizon multi-agent workflows to facilitate future research on long-chain
collaboration in MAS.

### 10. [Adaptive Bias Generalized Rollout Policy Adaptation on the Flexible Job-Shop Scheduling Problem](http://arxiv.org/pdf/2505.08451v1)

Authors: Lotfi Kobrosly, Marc-Emmanuel Coupvent des Graviers, Christophe Guettier, Tristan Cazenave

The Flexible Job-Shop Scheduling Problem (FJSSP) is an NP-hard combinatorial
optimization problem, with several application domains, especially for
manufacturing purposes. The objective is to
  efficiently schedule multiple operations on dissimilar machines. These
operations are gathered into jobs, and operations pertaining to the same job
need to be scheduled sequentially. Different methods have been previously
tested to solve this problem, such as Constraint Solving, Tabu Search, Genetic
Algorithms, or Monte Carlo Tree Search (MCTS). We propose a novel algorithm
derived from the Generalized Nested Rollout Policy Adaptation, developed to
solve the FJSSP. We report encouraging experimental results, as our algorithm
performs better than other MCTS-based approaches, even if makespans obtained on
large instances are still far from known upper bounds.

### Hardware Architecture

### 1. [e-GPU: An Open-Source and Configurable RISC-V Graphic Processing Unit for TinyAI Applications](http://arxiv.org/pdf/2505.08421v1)

Authors: Simone Machetti, Pasquale Davide Schiavone, Lara Orlandic, Darong Huang, Deniz Kasap, Giovanni Ansaloni, David Atienza

Graphics processing units (GPUs) excel at parallel processing, but remain
largely unexplored in ultra-low-power edge devices (TinyAI) due to their power
and area limitations, as well as the lack of suitable programming frameworks.
To address these challenges, this work introduces embedded GPU (e-GPU), an
open-source and configurable RISC-V GPU platform designed for TinyAI devices.
Its extensive configurability enables area and power optimization, while a
dedicated Tiny-OpenCL implementation provides a lightweight programming
framework tailored to resource-constrained environments. To demonstrate its
adaptability in real-world scenarios, we integrate the e-GPU with the
eXtendible Heterogeneous Energy-Efficient Platform (X-HEEP) to realize an
accelerated processing unit (APU) for TinyAI applications. Multiple instances
of the proposed system, featuring varying e-GPU configurations, are implemented
in TSMC's 16 nm SVT CMOS technology and are operated at 300 MHz and 0.8 V.
Their area and leakage characteristics are analyzed to ensure alignment with
TinyAI constraints. To assess both runtime overheads and application-level
efficiency, we employ two benchmarks: General Matrix Multiply (GeMM) and
bio-signal processing (TinyBio) workloads. The GeMM benchmark is used to
quantify the scheduling overhead introduced by the Tiny-OpenCL framework. The
results show that the delay becomes negligible for matrix sizes larger than
256x256 (or equivalent problem sizes). The TinyBio benchmark is then used to
evaluate performance and energy improvements in the baseline host. The results
demonstrate that the high-range e-GPU configuration with 16 threads achieves up
to a 15.1x speed-up and reduces energy consumption by up to 3.1x, while
incurring only a 2.5x area overhead and operating within a 28 mW power budget.

### 2. [SpNeRF: Memory Efficient Sparse Volumetric Neural Rendering Accelerator for Edge Devices](http://arxiv.org/pdf/2505.08191v1)

Authors: Yipu Zhang, Jiawei Liang, Jian Peng, Jiang Xu, Wei Zhang

Neural rendering has gained prominence for its high-quality output, which is
crucial for AR/VR applications. However, its large voxel grid data size and
irregular access patterns challenge real-time processing on edge devices. While
previous works have focused on improving data locality, they have not
adequately addressed the issue of large voxel grid sizes, which necessitate
frequent off-chip memory access and substantial on-chip memory. This paper
introduces SpNeRF, a software-hardware co-design solution tailored for sparse
volumetric neural rendering. We first identify memory-bound rendering
inefficiencies and analyze the inherent sparsity in the voxel grid data of
neural rendering. To enhance efficiency, we propose novel preprocessing and
online decoding steps, reducing the memory size for voxel grid. The
preprocessing step employs hash mapping to support irregular data access while
maintaining a minimal memory size. The online decoding step enables efficient
on-chip sparse voxel grid processing, incorporating bitmap masking to mitigate
PSNR loss caused by hash collisions. To further optimize performance, we design
a dedicated hardware architecture supporting our sparse voxel grid processing
technique. Experimental results demonstrate that SpNeRF achieves an average
21.07$\times$ reduction in memory size while maintaining comparable PSNR
levels. When benchmarked against Jetson XNX, Jetson ONX, RT-NeRF.Edge and
NeuRex.Edge, our design achieves speedups of 95.1$\times$, 63.5$\times$,
1.5$\times$ and 10.3$\times$, and improves energy efficiency by 625.6$\times$,
529.1$\times$, 4$\times$, and 4.4$\times$, respectively.

### 3. [Area Comparison of CHERIoT and PMP in Ibex](http://arxiv.org/pdf/2505.08541v1)

Authors: Samuel Riedel, Marno van der Maas, John Thomson, Andreas Kurth, Pirmin Vogel

Memory safety is a critical concern for modern embedded systems, particularly
in security-sensitive applications. This paper explores the area impact of
adding memory safety extensions to the Ibex RISC-V core, focusing on physical
memory protection (PMP) and Capability Hardware Extension to RISC-V for
Internet of Things (CHERIoT). We synthesise the extended Ibex cores using a
commercial tool targeting the open FreePDK45 process and provide a detailed
area breakdown and discussion of the results.
  The PMP configuration we consider is one with 16 PMP regions. We find that
the extensions increase the core size by 24 thousand gate-equivalent (kGE) for
PMP and 33 kGE for CHERIoT. The increase is mainly due to the additional state
required to store information about protected memory. While this increase
amounts to 42% for PMP and 57% for CHERIoT in Ibex's area, its effect on the
overall system is minimal. In a complete system-on-chip (SoC), like the secure
microcontroller OpenTitan Earl Grey, where the core represents only a fraction
of the total area, the estimated system-wide overhead is 0.6% for PMP and 1%
for CHERIoT. Given the security benefits these extensions provide, the area
trade-off is justified, making Ibex a compelling choice for secure embedded
applications.

### 4. [MINIMALIST: switched-capacitor circuits for efficient in-memory computation of gated recurrent units](http://arxiv.org/pdf/2505.08599v1)

Authors: Sebastian Billaudelle, Laura Kriener, Filippo Moro, Tristan Torchet, Melika Payvand

Recurrent neural networks (RNNs) have been a long-standing candidate for
processing of temporal sequence data, especially in memory-constrained systems
that one may find in embedded edge computing environments. Recent advances in
training paradigms have now inspired new generations of efficient RNNs. We
introduce a streamlined and hardware-compatible architecture based on minimal
gated recurrent units (GRUs), and an accompanying efficient mixed-signal
hardware implementation of the model. The proposed design leverages
switched-capacitor circuits not only for in-memory computation (IMC), but also
for the gated state updates. The mixed-signal cores rely solely on commodity
circuits consisting of metal capacitors, transmission gates, and a clocked
comparator, thus greatly facilitating scaling and transfer to other technology
nodes.
  We benchmark the performance of our architecture on time series data,
introducing all constraints required for a direct mapping to the hardware
system. The direct compatibility is verified in mixed-signal simulations,
reproducing data recorded from the software-only network model.

### Computational Complexity

### 1. [Short and useful quantum proofs for sublogarithmic-space verifiers](http://arxiv.org/pdf/2505.08462v1)

Authors: A. C. Cem Say

Quantum Merlin-Arthur proof systems are believed to be stronger than both
their classical counterparts and ``stand-alone'' quantum computers when Arthur
is assumed to operate in $\Omega(\log n)$ space. No hint of such an advantage
over classical computation had emerged from research on smaller space bounds,
which had so far concentrated on constant-space verifiers. We initiate the
study of quantum Merlin-Arthur systems with space bounds in $\omega(1) \cap
o(\log n)$, and exhibit a problem family $\mathcal{F}$, whose yes-instances
have proofs that are verifiable by polynomial-time quantum Turing machines
operating in this regime. We show that no problem in $\mathcal{F}$ has proofs
that can be verified classically or is solvable by a stand-alone quantum
machine in polynomial time if standard complexity assumptions hold. Unlike
previous examples of small-space verifiers, our protocols require only
subpolynomial-length quantum proofs.

### Computational Engineering

### 1. [Topology and geometry optimization of grid-shells under self-weight loading](http://arxiv.org/pdf/2505.08645v1)

Authors: Helen E. Fairclough, Karol Bolbotowski, Linwei He, Andrew Liew, Matthew Gilbert

This manuscript presents an approach for simultaneously optimizing the
connectivity and elevation of grid-shell structures acting in pure compression
(or pure tension) under the combined effects of a prescribed external loading
and the design-dependent self-weight of the structure itself. The method
derived herein involves solving a second-order cone optimization problem,
thereby ensuring convexity and obtaining globally optimal results for a given
discretization of the design domain. Several numerical examples are presented,
illustrating characteristics of this class of optimal structures. It is found
that, as self-weight becomes more significant, both the optimal topology and
the optimal elevation profile of the structure change, highlighting the
importance of optimizing both topology and geometry simultaneously from the
earliest stages of design. It is shown that this approach can obtain solutions
with greater accuracy and several orders of magnitude more quickly than a
standard 3D layout/truss topology optimization approach.

### 2. [Improving Unsupervised Task-driven Models of Ventral Visual Stream via Relative Position Predictivity](http://arxiv.org/pdf/2505.08316v1)

Authors: Dazhong Rong, Hao Dong, Xing Gao, Jiyu Wei, Di Hong, Yaoyao Hao, Qinming He, Yueming Wang

Based on the concept that ventral visual stream (VVS) mainly functions for
object recognition, current unsupervised task-driven methods model VVS by
contrastive learning, and have achieved good brain similarity. However, we
believe functions of VVS extend beyond just object recognition. In this paper,
we introduce an additional function involving VVS, named relative position (RP)
prediction. We first theoretically explain contrastive learning may be unable
to yield the model capability of RP prediction. Motivated by this, we
subsequently integrate RP learning with contrastive learning, and propose a new
unsupervised task-driven method to model VVS, which is more inline with
biological reality. We conduct extensive experiments, demonstrating that: (i)
our method significantly improves downstream performance of object recognition
while enhancing RP predictivity; (ii) RP predictivity generally improves the
model brain similarity. Our results provide strong evidence for the involvement
of VVS in location perception (especially RP prediction) from a computational
perspective.

### 3. [Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations](http://arxiv.org/pdf/2505.08740v2)

Authors: Abdolmehdi Behroozi, Chaopeng Shen and, Daniel Kifer

Parametric differential equations of the form du/dt = f(u, x, t, p) are
fundamental in science and engineering. While deep learning frameworks such as
the Fourier Neural Operator (FNO) can efficiently approximate solutions, they
struggle with inverse problems, sensitivity estimation (du/dp), and concept
drift. We address these limitations by introducing a sensitivity-based
regularization strategy, called Sensitivity-Constrained Fourier Neural
Operators (SC-FNO). SC-FNO achieves high accuracy in predicting solution paths
and consistently outperforms standard FNO and FNO with physics-informed
regularization. It improves performance in parameter inversion tasks, scales to
high-dimensional parameter spaces (tested with up to 82 parameters), and
reduces both data and training requirements. These gains are achieved with a
modest increase in training time (30% to 130% per epoch) and generalize across
various types of differential equations and neural operators. Code and selected
experiments are available at: https://github.com/AMBehroozi/SC_Neural_Operators

### 4. [Addressing the Current Challenges of Quantum Machine Learning through Multi-Chip Ensembles](http://arxiv.org/pdf/2505.08782v1)

Authors: Junghoon Justin Park, Jiook Cha, Samuel Yen-Chi Chen, Huan-Hsin Tseng, Shinjae Yoo

Quantum Machine Learning (QML) holds significant promise for solving
computational challenges across diverse domains. However, its practical
deployment is constrained by the limitations of noisy intermediate-scale
quantum (NISQ) devices, including noise, limited scalability, and trainability
issues in variational quantum circuits (VQCs). We introduce the multi-chip
ensemble VQC framework, which partitions high-dimensional computations across
smaller quantum chips to enhance scalability, trainability, and noise
resilience. We show that this approach mitigates barren plateaus, reduces
quantum error bias and variance, and maintains robust generalization through
controlled entanglement. Designed to align with current and emerging quantum
hardware, the framework demonstrates strong potential for enabling scalable QML
on near-term devices, as validated by experiments on standard benchmark
datasets (MNIST, FashionMNIST, CIFAR-10) and real world dataset (PhysioNet
EEG).

### Computational Geometry

### 1. [Computing Projective Implicit Representations from Poset Towers](http://arxiv.org/pdf/2505.08755v1)

Authors: Tamal K. Dey, Florian Russold

A family of simplicial complexes, connected with simplicial maps and indexed
by a poset $P$, is called a poset tower. The concept of poset towers subsumes
classical objects of study in the persistence literature, as, for example,
one-critical multi-filtrations and zigzag filtrations, but also allows
multi-critical simplices and arbitrary simplicial maps. The homology of a poset
tower gives rise to a $P$-persistence module. To compute this homology globally
over $P$, in the spirit of the persistence algorithm, we consider the homology
of a chain complex of $P$-persistence modules,
$C_{\ell-1}\xleftarrow{}C_\ell\xleftarrow{}C_{\ell+1}$, induced by the
simplices of the poset tower. Contrary to the case of one-critical filtrations,
the chain-modules $C_\ell$ of a poset tower can have a complicated structure.
In this work, we tackle the problem of computing a representation of such a
chain complex segment by projective modules and $P$-graded matrices, which we
call a projective implicit representation (PiRep). We give efficient algorithms
to compute asymptotically minimal projective resolutions (up to the second
term) of the chain modules and the boundary maps and compute a PiRep from these
resolutions. Our algorithms are tailored to the chain complexes and resolutions
coming from poset towers and take advantage of their special structure. In the
context of poset towers, they are fully general and could potentially serve as
a foundation for developing more efficient algorithms on specific posets.

### 2. [Claycode: Stylable and Deformable 2D Scannable Codes](http://arxiv.org/pdf/2505.08666v1)

Authors: Marco Maida, Alberto Crescini, Marco Perronet, Elena Camuffo

This paper introduces Claycode, a novel 2D scannable code designed for
extensive stylization and deformation. Unlike traditional matrix-based codes
(e.g., QR codes), Claycodes encode their message in a tree structure. During
the encoding process, bits are mapped into a topology tree, which is then
depicted as a nesting of color regions drawn within the boundaries of a target
polygon shape. When decoding, Claycodes are extracted and interpreted in
real-time from a camera stream. We detail the end-to-end pipeline and show that
Claycodes allow for extensive stylization without compromising their
functionality. We then empirically demonstrate Claycode's high tolerance to
heavy deformations, outperforming traditional 2D scannable codes in scenarios
where they typically fail.

### Computation and Language

### 1. [Evaluating the Effectiveness of Black-Box Prompt Optimization as the Scale of LLMs Continues to Grow](http://arxiv.org/pdf/2505.08303v1)

Authors: Ziyu Zhou, Yihang Wu, Jingyuan Yang, Zhan Xiao, Rongjun Li

Black-Box prompt optimization methods have emerged as a promising strategy
for refining input prompts to better align large language models (LLMs),
thereby enhancing their task performance. Although these methods have
demonstrated encouraging results, most studies and experiments have primarily
focused on smaller-scale models (e.g., 7B, 14B) or earlier versions (e.g.,
GPT-3.5) of LLMs. As the scale of LLMs continues to increase, such as with
DeepSeek V3 (671B), it remains an open question whether these black-box
optimization techniques will continue to yield significant performance
improvements for models of such scale. In response to this, we select three
well-known black-box optimization methods and evaluate them on large-scale LLMs
(DeepSeek V3 and Gemini 2.0 Flash) across four NLU and NLG datasets. The
results show that these black-box prompt optimization methods offer only
limited improvements on these large-scale LLMs. Furthermore, we hypothesize
that the scale of the model is the primary factor contributing to the limited
benefits observed. To explore this hypothesis, we conducted experiments on LLMs
of varying sizes (Qwen 2.5 series, ranging from 7B to 72B) and observed an
inverse scaling law, wherein the effectiveness of black-box optimization
methods diminished as the model size increased.

### 2. [AM-Thinking-v1: Advancing the Frontier of Reasoning at 32B Scale](http://arxiv.org/pdf/2505.08311v1)

Authors: Yunjie Ji, Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Han Zhao, Xiangang Li

We present AM-Thinking-v1, a 32B dense language model that advances the
frontier of reasoning, embodying the collaborative spirit of open-source
innovation. Outperforming DeepSeek-R1 and rivaling leading Mixture-of-Experts
(MoE) models like Qwen3-235B-A22B and Seed1.5-Thinking, AM-Thinking-v1 achieves
impressive scores of 85.3 on AIME 2024, 74.4 on AIME 2025, and 70.3 on
LiveCodeBench, showcasing state-of-the-art mathematical and coding capabilities
among open-source models of similar scale.
  Built entirely from the open-source Qwen2.5-32B base model and publicly
available queries, AM-Thinking-v1 leverages a meticulously crafted
post-training pipeline - combining supervised fine-tuning and reinforcement
learning - to deliver exceptional reasoning capabilities. This work
demonstrates that the open-source community can achieve high performance at the
32B scale, a practical sweet spot for deployment and fine-tuning. By striking a
balance between top-tier performance and real-world usability, we hope
AM-Thinking-v1 inspires further collaborative efforts to harness mid-scale
models, pushing reasoning boundaries while keeping accessibility at the core of
innovation. We have open-sourced our model on
\href{https://huggingface.co/a-m-team/AM-Thinking-v1}{Hugging Face}.

### 3. [On the Geometry of Semantics in Next-token Prediction](http://arxiv.org/pdf/2505.08348v1)

Authors: Yize Zhao, Christos Thrampoulidis

Modern language models demonstrate a remarkable ability to capture linguistic
meaning despite being trained solely through next-token prediction (NTP). We
investigate how this conceptually simple training objective leads models to
extract and encode latent semantic and grammatical concepts. Our analysis
reveals that NTP optimization implicitly guides models to encode concepts via
singular value decomposition (SVD) factors of a centered data-sparsity matrix
that captures next-word co-occurrence patterns. While the model never
explicitly constructs this matrix, learned word and context embeddings
effectively factor it to capture linguistic structure. We find that the most
important SVD factors are learned first during training, motivating the use of
spectral clustering of embeddings to identify human-interpretable semantics,
including both classical k-means and a new orthant-based method directly
motivated by our interpretation of concepts. Overall, our work bridges
distributional semantics, neural collapse geometry, and neural network training
dynamics, providing insights into how NTP's implicit biases shape the emergence
of meaning representations in language models.

### 4. [Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring](http://arxiv.org/pdf/2505.08351v1)

Authors: Mina Almasi, Ross Deans Kristensen-McLachlan

This paper investigates the potentials of Large Language Models (LLMs) as
adaptive tutors in the context of second-language learning. In particular, we
evaluate whether system prompting can reliably constrain LLMs to generate only
text appropriate to the student's competence level. We simulate full
teacher-student dialogues in Spanish using instruction-tuned, open-source LLMs
ranging in size from 7B to 12B parameters. Dialogues are generated by having an
LLM alternate between tutor and student roles with separate chat histories. The
output from the tutor model is then used to evaluate the effectiveness of
CEFR-based prompting to control text difficulty across three proficiency levels
(A1, B1, C1). Our findings suggest that while system prompting can be used to
constrain model outputs, prompting alone is too brittle for sustained,
long-term interactional contexts - a phenomenon we term alignment drift. Our
results provide insights into the feasibility of LLMs for personalized,
proficiency-aligned adaptive tutors and provide a scalable method for low-cost
evaluation of model performance without human participants.

### 5. [Towards Contamination Resistant Benchmarks](http://arxiv.org/pdf/2505.08389v1)

Authors: Rahmatullah Musawi, Sheng Lu

The rapid development of large language models (LLMs) has transformed the
landscape of natural language processing. Evaluating LLMs properly is crucial
for understanding their potential and addressing concerns such as safety.
However, LLM evaluation is confronted by various factors, among which
contamination stands out as a key issue that undermines the reliability of
evaluations. In this work, we introduce the concept of contamination resistance
to address this challenge. We propose a benchmark based on Caesar ciphers
(e.g., "ab" to "bc" when the shift is 1), which, despite its simplicity, is an
excellent example of a contamination resistant benchmark. We test this
benchmark on widely used LLMs under various settings, and we find that these
models struggle with this benchmark when contamination is controlled. Our
findings reveal issues in current LLMs and raise important questions regarding
their true capabilities. Our work contributes to the development of
contamination resistant benchmarks, enabling more rigorous LLM evaluation and
offering insights into the true capabilities and limitations of LLMs.

### 6. [TUMS: Enhancing Tool-use Abilities of LLMs with Multi-structure Handlers](http://arxiv.org/pdf/2505.08402v1)

Authors: Aiyao He, Sijia Cui, Shuai Xu, Yanna Wang, Bo Xu

Recently, large language models(LLMs) have played an increasingly important
role in solving a wide range of NLP tasks, leveraging their capabilities of
natural language understanding and generating. Integration with external tools
further enhances LLMs' effectiveness, providing more precise, timely, and
specialized responses. However, LLMs still encounter difficulties with
non-executable actions and improper actions, which are primarily attributed to
incorrect parameters. The process of generating parameters by LLMs is confined
to the tool level, employing the coarse-grained strategy without considering
the different difficulties of various tools. To address this issue, we propose
TUMS, a novel framework designed to enhance the tool-use capabilities of LLMs
by transforming tool-level processing into parameter-level processing.
Specifically, our framework consists of four key components: (1) an intent
recognizer that identifies the user's intent to help LLMs better understand the
task; (2) a task decomposer that breaks down complex tasks into simpler
subtasks, each involving a tool call; (3) a subtask processor equipped with
multi-structure handlers to generate accurate parameters; and (4) an executor.
Our empirical studies have evidenced the effectiveness and efficiency of the
TUMS framework with an average of 19.6\% and 50.6\% improvement separately on
easy and hard benchmarks of ToolQA, meanwhile, we demonstrated the key
contribution of each part with ablation experiments, offering more insights and
stimulating future research on Tool-augmented LLMs.

### 7. [A document processing pipeline for the construction of a dataset for topic modeling based on the judgments of the Italian Supreme Court](http://arxiv.org/pdf/2505.08439v1)

Authors: Matteo Marulli, Glauco Panattoni, Marco Bertini

Topic modeling in Italian legal research is hindered by the lack of public
datasets, limiting the analysis of legal themes in Supreme Court judgments. To
address this, we developed a document processing pipeline that produces an
anonymized dataset optimized for topic modeling.
  The pipeline integrates document layout analysis (YOLOv8x), optical character
recognition, and text anonymization. The DLA module achieved a mAP@50 of 0.964
and a mAP@50-95 of 0.800. The OCR detector reached a mAP@50-95 of 0.9022, and
the text recognizer (TrOCR) obtained a character error rate of 0.0047 and a
word error rate of 0.0248. Compared to OCR-only methods, our dataset improved
topic modeling with a diversity score of 0.6198 and a coherence score of
0.6638.
  We applied BERTopic to extract topics and used large language models to
generate labels and summaries. Outputs were evaluated against domain expert
interpretations. Claude Sonnet 3.7 achieved a BERTScore F1 of 0.8119 for
labeling and 0.9130 for summarization.

### 8. [IterKey: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation](http://arxiv.org/pdf/2505.08450v1)

Authors: Kazuki Hayashi, Hidetaka Kamigaito, Shinya Kouda, Taro Watanabe

Retrieval-Augmented Generation (RAG) has emerged as a way to complement the
in-context knowledge of Large Language Models (LLMs) by integrating external
documents. However, real-world applications demand not only accuracy but also
interpretability. While dense retrieval methods provide high accuracy, they
lack interpretability; conversely, sparse retrieval methods offer transparency
but often fail to capture the full intent of queries due to their reliance on
keyword matching. To address these issues, we introduce IterKey, an LLM-driven
iterative keyword generation framework that enhances RAG via sparse retrieval.
IterKey consists of three LLM-driven stages: generating keywords for retrieval,
generating answers based on retrieved documents, and validating the answers. If
validation fails, the process iteratively repeats with refined keywords. Across
four QA tasks, experimental results show that IterKey achieves 5% to 20%
accuracy improvements over BM25-based RAG and simple baselines. Its performance
is comparable to dense retrieval-based RAG and prior iterative query refinement
methods using dense models. In summary, IterKey is a novel BM25-based approach
leveraging LLMs to iteratively refine RAG, effectively balancing accuracy with
interpretability.

### 9. [Reassessing Graph Linearization for Sequence-to-sequence AMR Parsing: On the Advantages and Limitations of Triple-Based Encoding](http://arxiv.org/pdf/2505.08504v1)

Authors: Jeongwoo Kang, Maximin Coavoux, Cédric Lopez, Didier Schwab

Sequence-to-sequence models are widely used to train Abstract Meaning
Representation (Banarescu et al., 2013, AMR) parsers. To train such models, AMR
graphs have to be linearized into a one-line text format. While Penman encoding
is typically used for this purpose, we argue that it has limitations: (1) for
deep graphs, some closely related nodes are located far apart in the linearized
text (2) Penman's tree-based encoding necessitates inverse roles to handle node
re-entrancy, doubling the number of relation types to predict. To address these
issues, we propose a triple-based linearization method and compare its
efficiency with Penman linearization. Although triples are well suited to
represent a graph, our results suggest room for improvement in triple encoding
to better compete with Penman's concise and explicit representation of a nested
graph structure.

### 10. [Are We Paying Attention to Her? Investigating Gender Disambiguation and Attention in Machine Translation](http://arxiv.org/pdf/2505.08546v1)

Authors: Chiara Manna, Afra Alishahi, Frédéric Blain, Eva Vanmassenhove

While gender bias in modern Neural Machine Translation (NMT) systems has
received much attention, traditional evaluation metrics do not to fully capture
the extent to which these systems integrate contextual gender cues. We propose
a novel evaluation metric called Minimal Pair Accuracy (MPA), which measures
the reliance of models on gender cues for gender disambiguation. MPA is
designed to go beyond surface-level gender accuracy metrics by focusing on
whether models adapt to gender cues in minimal pairs -- sentence pairs that
differ solely in the gendered pronoun, namely the explicit indicator of the
target's entity gender in the source language (EN). We evaluate a number of NMT
models on the English-Italian (EN--IT) language pair using this metric, we show
that they ignore available gender cues in most cases in favor of (statistical)
stereotypical gender interpretation. We further show that in anti-stereotypical
cases, these models tend to more consistently take masculine gender cues into
account while ignoring the feminine cues. Furthermore, we analyze the attention
head weights in the encoder component and show that while all models encode
gender information to some extent, masculine cues elicit a more diffused
response compared to the more concentrated and specialized responses to
feminine gender cues.

### Cryptography and Security

### 1. [GDNTT: an Area-Efficient Parallel NTT Accelerator Using Glitch-Driven Near-Memory Computing and Reconfigurable 10T SRAM](http://arxiv.org/pdf/2505.08162v1)

Authors: Hengyu Ding, Houran Ji, Jia Li, Jinhang Chen, Chin-Wing Sham, Yao Wang

With the rapid advancement of quantum computing technology, post-quantum
cryptography (PQC) has emerged as a pivotal direction for next-generation
encryption standards. Among these, lattice-based cryptographic schemes rely
heavily on the fast Number Theoretic Transform (NTT) over polynomial rings,
whose performance directly determines encryption/decryption throughput and
energy efficiency. However, existing software-based NTT implementations
struggle to meet the real-time performance and low-power requirements of IoT
and edge devices. To address this challenge, this paper proposes an
area-efficient highly parallel NTT accelerator with glitch-driven near-memory
computing (GDNTT). The design integrates a 10T SRAM for data storage, enabling
flexible row/column data access and streamlining circuit mapping strategies.
Furthermore, a glitch generator is incorporated into the near-memory computing
unit, significantly reducing the latency of butterfly operations. Evaluation
results show that the proposed NTT accelerator achieves a 1.5~28* improvement
in throughput-per-area compared to the state-of-the-art.

### 2. [LM-Scout: Analyzing the Security of Language Model Integration in Android Apps](http://arxiv.org/pdf/2505.08204v1)

Authors: Muhammad Ibrahim, Gűliz Seray Tuncay, Z. Berkay Celik, Aravind Machiry, Antonio Bianchi

Developers are increasingly integrating Language Models (LMs) into their
mobile apps to provide features such as chat-based assistants. To prevent LM
misuse, they impose various restrictions, including limits on the number of
queries, input length, and allowed topics. However, if the LM integration is
insecure, attackers can bypass these restrictions and gain unrestricted access
to the LM, potentially harming developers' reputations and leading to
significant financial losses.
  This paper presents the first systematic study of insecure usage of LMs by
Android apps. We first manually analyze a preliminary dataset of apps to
investigate LM integration methods, construct a taxonomy that categorizes the
LM usage restrictions implemented by the apps, and determine how to bypass
them. Alarmingly, we can bypass restrictions in 127 out of 181 apps. Then, we
develop LM-Scout, a fully automated tool to detect on a large-scale vulnerable
usage of LMs in 2,950 mobile apps. LM-Scout shows that, in many cases (i.e.,
120 apps), it is possible to find and exploit such security issues
automatically. Finally, we identify the root causes for the identified issues
and offer recommendations for secure LM integration.

### 3. [ABAC Lab: An Interactive Platform for Attribute-based Access Control Policy Analysis, Tools, and Datasets](http://arxiv.org/pdf/2505.08209v1)

Authors: Thang Bui, Anthony Matricia, Emily Contreras, Ryan Mauvais, Luis Medina, Israel Serrano

Attribute-Based Access Control (ABAC) provides expressiveness and
flexibility, making it a compelling model for enforcing fine-grained access
control policies. To facilitate the transition to ABAC, extensive research has
been conducted to develop methodologies, frameworks, and tools that assist
policy administrators in adapting the model. Despite these efforts, challenges
remain in the availability and benchmarking of ABAC datasets. Specifically,
there is a lack of clarity on how datasets can be systematically acquired, no
standardized benchmarking practices to evaluate existing methodologies and
their effectiveness, and limited access to real-world datasets suitable for
policy analysis and testing.
  This paper introduces ABAC Lab, an interactive platform that addresses these
challenges by integrating existing ABAC policy datasets with analytical tools
for policy evaluation. Additionally, we present two new ABAC datasets derived
from real-world case studies. ABAC Lab serves as a valuable resource for both
researchers studying ABAC policies and policy administrators seeking to adopt
ABAC within their organizations. By offering an environment for dataset
exploration and policy analysis, ABAC Lab facilitates research, aids policy
administrators in transitioning to ABAC, and promotes a more structured
approach to ABAC policy evaluation and development.

### 4. [On the Account Security Risks Posed by Password Strength Meters](http://arxiv.org/pdf/2505.08292v1)

Authors: Ming Xu, Weili Han, Jitao Yu, Jing Liu, Xinyi Zhang, Yun Lin, Jin Song Dong

Password strength meters (PSMs) have been widely used by websites to gauge
password strength, encouraging users to create stronger passwords. Popular
data-driven PSMs, e.g., based on Markov, Probabilistic Context-free Grammar
(PCFG) and neural networks, alarm strength based on a model learned from real
passwords. Despite their proven effectiveness, the secure utility that arises
from the leakage of trained passwords remains largely overlooked. To address
this gap, we analyze 11 PSMs and find that 5 data-driven meters are vulnerable
to membership inference attacks that expose their trained passwords, and
seriously, 3 rule-based meters openly disclose their blocked passwords. We
specifically design a PSM privacy leakage evaluation approach, and uncover that
a series of general data-driven meters are vulnerable to leaking between 10^4
to 10^5 trained passwords, with the PCFG-based models being more vulnerable
than other counterparts; furthermore, we aid in deriving insights that the
inherent utility-privacy tradeoff is not as severe as previously thought. To
further exploit the risks, we develop novel meter-aware attacks when a clever
attacker can filter the used passwords during compromising accounts on websites
using the meter, and experimentally show that attackers targeting websites that
deployed the popular Zxcvbn meter can compromise an additional 5.84% user
accounts within 10 attempts, demonstrating the urgent need for
privacy-preserving PSMs that protect the confidentiality of the meter's used
passwords. Finally, we sketch some counter-measures to mitigate these threats.

### 5. [Cryptologic Techniques and Associated Risks in Public and Private Security. An Italian and European Union Perspective with an Overview of the Current Legal Framework](http://arxiv.org/pdf/2505.08650v1)

Authors: Zana Kudriasova

This article examines the evolution of cryptologic techniques and their
implications for public and private security, focusing on the Italian and EU
legal frameworks. It explores the roles of cryptography, steganography, and
quantum technologies in countering cybersecurity threats, emphasising the need
for robust legislation to address emerging challenges. Special attention is
given to Italy's legislative reforms, including Law No. 90 of 2024, which
strengthens penalties for cybercrimes and establishes the National Cryptography
Centre within the Italian National Cybersecurity Agency. Additionally, the
article highlights international initiatives, such as the UN's draft convention
on cybercrime, emphasising the balance between security, privacy, and
fundamental human rights in a post-quantum era.

### 6. [Comparative Analysis of Blockchain Systems](http://arxiv.org/pdf/2505.08652v1)

Authors: Jiaqi Huang, Yuanzheng Niu, Xiaoqi Li, Zongwei Li

Blockchain is a type of decentralized distributed database. Unlike
traditional relational database management systems, it does not require
management or maintenance by a third party. All data management and update
processes are open and transparent, solving the trust issues of centralized
database management systems. Blockchain ensures network-wide consistency,
consensus, traceability, and immutability. Under the premise of mutual distrust
between nodes, blockchain technology integrates various technologies, such as
P2P protocols, asymmetric encryption, consensus mechanisms, and chain
structures. Data is distributed and stored across multiple nodes, maintained by
all nodes, ensuring transaction data integrity, undeniability, and security.
This facilitates trusted information sharing and supervision. The basic
principles of blockchain form the foundation for all related research.
Understanding the working principles is essential for further study of
blockchain technology. There are many platforms based on blockchain technology,
and they differ from one another. This paper will analyze the architecture of
blockchain systems at each layer, focusing on the principles and technologies
of blockchain platforms such as Bitcoin, Ethereum, and Hyperledger Fabric. The
analysis will cover their scalability and security and highlight their
similarities, differences, advantages, and disadvantages.

### 7. [Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted](http://arxiv.org/pdf/2505.08255v1)

Authors: Shuaiwei Yuan, Junyu Dong, Yuezun Li

With the advancement of AI generative techniques, Deepfake faces have become
incredibly realistic and nearly indistinguishable to the human eye. To counter
this, Deepfake detectors have been developed as reliable tools for assessing
face authenticity. These detectors are typically developed on Deep Neural
Networks (DNNs) and trained using third-party datasets. However, this protocol
raises a new security risk that can seriously undermine the trustfulness of
Deepfake detectors: Once the third-party data providers insert poisoned
(corrupted) data maliciously, Deepfake detectors trained on these datasets will
be injected ``backdoors'' that cause abnormal behavior when presented with
samples containing specific triggers. This is a practical concern, as
third-party providers may distribute or sell these triggers to malicious users,
allowing them to manipulate detector performance and escape accountability.
  This paper investigates this risk in depth and describes a solution to
stealthily infect Deepfake detectors. Specifically, we develop a trigger
generator, that can synthesize passcode-controlled, semantic-suppression,
adaptive, and invisible trigger patterns, ensuring both the stealthiness and
effectiveness of these triggers. Then we discuss two poisoning scenarios,
dirty-label poisoning and clean-label poisoning, to accomplish the injection of
backdoors. Extensive experiments demonstrate the effectiveness, stealthiness,
and practicality of our method compared to several baselines.

### 8. [Area Comparison of CHERIoT and PMP in Ibex](http://arxiv.org/pdf/2505.08541v1)

Authors: Samuel Riedel, Marno van der Maas, John Thomson, Andreas Kurth, Pirmin Vogel

Memory safety is a critical concern for modern embedded systems, particularly
in security-sensitive applications. This paper explores the area impact of
adding memory safety extensions to the Ibex RISC-V core, focusing on physical
memory protection (PMP) and Capability Hardware Extension to RISC-V for
Internet of Things (CHERIoT). We synthesise the extended Ibex cores using a
commercial tool targeting the open FreePDK45 process and provide a detailed
area breakdown and discussion of the results.
  The PMP configuration we consider is one with 16 PMP regions. We find that
the extensions increase the core size by 24 thousand gate-equivalent (kGE) for
PMP and 33 kGE for CHERIoT. The increase is mainly due to the additional state
required to store information about protected memory. While this increase
amounts to 42% for PMP and 57% for CHERIoT in Ibex's area, its effect on the
overall system is minimal. In a complete system-on-chip (SoC), like the secure
microcontroller OpenTitan Earl Grey, where the core represents only a fraction
of the total area, the estimated system-wide overhead is 0.6% for PMP and 1%
for CHERIoT. Given the security benefits these extensions provide, the area
trade-off is justified, making Ibex a compelling choice for secure embedded
applications.

### 9. [ROSA: Finding Backdoors with Fuzzing](http://arxiv.org/pdf/2505.08544v1)

Authors: Dimitri Kokkonis, Michaël Marcozzi, Emilien Decoux, Stefano Zacchiroli

A code-level backdoor is a hidden access, programmed and concealed within the
code of a program. For instance, hard-coded credentials planted in the code of
a file server application would enable maliciously logging into all deployed
instances of this application. Confirmed software supply chain attacks have led
to the injection of backdoors into popular open-source projects, and backdoors
have been discovered in various router firmware. Manual code auditing for
backdoors is challenging and existing semi-automated approaches can handle only
a limited scope of programs and backdoors, while requiring manual
reverse-engineering of the audited (binary) program. Graybox fuzzing (automated
semi-randomized testing) has grown in popularity due to its success in
discovering vulnerabilities and hence stands as a strong candidate for improved
backdoor detection. However, current fuzzing knowledge does not offer any means
to detect the triggering of a backdoor at runtime. In this work we introduce
ROSA, a novel approach (and tool) which combines a state-of-the-art fuzzer
(AFL++) with a new metamorphic test oracle, capable of detecting runtime
backdoor triggers. To facilitate the evaluation of ROSA, we have created
ROSARUM, the first openly available benchmark for assessing the detection of
various backdoors in diverse programs. Experimental evaluation shows that ROSA
has a level of robustness, speed and automation similar to classical fuzzing.
It finds all 17 authentic or synthetic backdooors from ROSARUM in 1h30 on
average. Compared to existing detection tools, it can handle a diversity of
backdoors and programs and it does not rely on manual reverse-engineering of
the fuzzed binary code.

### 10. [MUBox: A Critical Evaluation Framework of Deep Machine Unlearning](http://arxiv.org/pdf/2505.08576v1)

Authors: Xiang Li, Bhavani Thuraisingham, Wenqi Wei

Recent legal frameworks have mandated the right to be forgotten, obligating
the removal of specific data upon user requests. Machine Unlearning has emerged
as a promising solution by selectively removing learned information from
machine learning models. This paper presents MUBox, a comprehensive platform
designed to evaluate unlearning methods in deep learning. MUBox integrates 23
advanced unlearning techniques, tested across six practical scenarios with 11
diverse evaluation metrics. It allows researchers and practitioners to (1)
assess and compare the effectiveness of different machine unlearning methods
across various scenarios; (2) examine the impact of current evaluation metrics
on unlearning performance; and (3) conduct detailed comparative studies on
machine unlearning in a unified framework. Leveraging MUBox, we systematically
evaluate these unlearning methods in deep learning and uncover several key
insights: (a) Even state-of-the-art unlearning methods, including those
published in top-tier venues and winners of unlearning competitions,
demonstrate inconsistent effectiveness across diverse scenarios. Prior research
has predominantly focused on simplified settings, such as random forgetting and
class-wise unlearning, highlighting the need for broader evaluations across
more difficult unlearning tasks. (b) Assessing unlearning performance remains a
non-trivial problem, as no single evaluation metric can comprehensively capture
the effectiveness, efficiency, and preservation of model utility. Our findings
emphasize the necessity of employing multiple metrics to achieve a balanced and
holistic assessment of unlearning methods. (c) In the context of depoisoning,
our evaluation reveals significant variability in the effectiveness of existing
approaches, which is highly dependent on the specific type of poisoning
attacks.

### Computer Vision and Pattern Recognition

### 1. [MoKD: Multi-Task Optimization for Knowledge Distillation](http://arxiv.org/pdf/2505.08170v1)

Authors: Zeeshan Hayder, Ali Cheraghian, Lars Petersson, Mehrtash Harandi

Compact models can be effectively trained through Knowledge Distillation
(KD), a technique that transfers knowledge from larger, high-performing teacher
models. Two key challenges in Knowledge Distillation (KD) are: 1) balancing
learning from the teacher's guidance and the task objective, and 2) handling
the disparity in knowledge representation between teacher and student models.
To address these, we propose Multi-Task Optimization for Knowledge Distillation
(MoKD). MoKD tackles two main gradient issues: a) Gradient Conflicts, where
task-specific and distillation gradients are misaligned, and b) Gradient
Dominance, where one objective's gradient dominates, causing imbalance. MoKD
reformulates KD as a multi-objective optimization problem, enabling better
balance between objectives. Additionally, it introduces a subspace learning
framework to project feature representations into a high-dimensional space,
improving knowledge transfer. Our MoKD is demonstrated to outperform existing
methods through extensive experiments on image classification using the
ImageNet-1K dataset and object detection using the COCO dataset, achieving
state-of-the-art performance with greater efficiency. To the best of our
knowledge, MoKD models also achieve state-of-the-art performance compared to
models trained from scratch.

### 2. [Empowering Vision Transformers with Multi-Scale Causal Intervention for Long-Tailed Image Classification](http://arxiv.org/pdf/2505.08173v1)

Authors: Xiaoshuo Yan, Zhaochuan Li, Lei Meng, Zhuang Qi, Wei Wu, Zixuan Li, Xiangxu Meng

Causal inference has emerged as a promising approach to mitigate long-tail
classification by handling the biases introduced by class imbalance. However,
along with the change of advanced backbone models from Convolutional Neural
Networks (CNNs) to Visual Transformers (ViT), existing causal models may not
achieve an expected performance gain. This paper investigates the influence of
existing causal models on CNNs and ViT variants, highlighting that ViT's global
feature representation makes it hard for causal methods to model associations
between fine-grained features and predictions, which leads to difficulties in
classifying tail classes with similar visual appearance. To address these
issues, this paper proposes TSCNet, a two-stage causal modeling method to
discover fine-grained causal associations through multi-scale causal
interventions. Specifically, in the hierarchical causal representation learning
stage (HCRL), it decouples the background and objects, applying backdoor
interventions at both the patch and feature level to prevent model from using
class-irrelevant areas to infer labels which enhances fine-grained causal
representation. In the counterfactual logits bias calibration stage (CLBC), it
refines the optimization of model's decision boundary by adaptive constructing
counterfactual balanced data distribution to remove the spurious associations
in the logits caused by data distribution. Extensive experiments conducted on
various long-tail benchmarks demonstrate that the proposed TSCNet can eliminate
multiple biases introduced by data imbalance, which outperforms existing
methods.

### 3. [Monocular Depth Guided Occlusion-Aware Disparity Refinement via Semi-supervised Learning in Laparoscopic Images](http://arxiv.org/pdf/2505.08178v1)

Authors: Ziteng Liu, Dongdong He, Chenghong Zhang, Wenpeng Gao, Yili Fu

Occlusion and the scarcity of labeled surgical data are significant
challenges in disparity estimation for stereo laparoscopic images. To address
these issues, this study proposes a Depth Guided Occlusion-Aware Disparity
Refinement Network (DGORNet), which refines disparity maps by leveraging
monocular depth information unaffected by occlusion. A Position Embedding (PE)
module is introduced to provide explicit spatial context, enhancing the
network's ability to localize and refine features. Furthermore, we introduce an
Optical Flow Difference Loss (OFDLoss) for unlabeled data, leveraging temporal
continuity across video frames to improve robustness in dynamic surgical
scenes. Experiments on the SCARED dataset demonstrate that DGORNet outperforms
state-of-the-art methods in terms of End-Point Error (EPE) and Root Mean
Squared Error (RMSE), particularly in occlusion and texture-less regions.
Ablation studies confirm the contributions of the Position Embedding and
Optical Flow Difference Loss, highlighting their roles in improving spatial and
temporal consistency. These results underscore DGORNet's effectiveness in
enhancing disparity estimation for laparoscopic surgery, offering a practical
solution to challenges in disparity estimation and data limitations.

### 4. [ADC-GS: Anchor-Driven Deformable and Compressed Gaussian Splatting for Dynamic Scene Reconstruction](http://arxiv.org/pdf/2505.08196v1)

Authors: He Huang, Qi Yang, Mufan Liu, Yiling Xu, Zhu Li

Existing 4D Gaussian Splatting methods rely on per-Gaussian deformation from
a canonical space to target frames, which overlooks redundancy among adjacent
Gaussian primitives and results in suboptimal performance. To address this
limitation, we propose Anchor-Driven Deformable and Compressed Gaussian
Splatting (ADC-GS), a compact and efficient representation for dynamic scene
reconstruction. Specifically, ADC-GS organizes Gaussian primitives into an
anchor-based structure within the canonical space, enhanced by a temporal
significance-based anchor refinement strategy. To reduce deformation
redundancy, ADC-GS introduces a hierarchical coarse-to-fine pipeline that
captures motions at varying granularities. Moreover, a rate-distortion
optimization is adopted to achieve an optimal balance between bitrate
consumption and representation fidelity. Experimental results demonstrate that
ADC-GS outperforms the per-Gaussian deformation approaches in rendering speed
by 300%-800% while achieving state-of-the-art storage efficiency without
compromising rendering quality. The code is released at
https://github.com/H-Huang774/ADC-GS.git.

### 5. [Visual Watermarking in the Era of Diffusion Models: Advances and Challenges](http://arxiv.org/pdf/2505.08197v1)

Authors: Junxian Duan, Jiyang Guang, Wenkui Yang, Ran He

As generative artificial intelligence technologies like Stable Diffusion
advance, visual content becomes more vulnerable to misuse, raising concerns
about copyright infringement. Visual watermarks serve as effective protection
mechanisms, asserting ownership and deterring unauthorized use. Traditional
deepfake detection methods often rely on passive techniques that struggle with
sophisticated manipulations. In contrast, diffusion models enhance detection
accuracy by allowing for the effective learning of features, enabling the
embedding of imperceptible and robust watermarks. We analyze the strengths and
challenges of watermark techniques related to diffusion models, focusing on
their robustness and application in watermark generation. By exploring the
integration of advanced diffusion models and watermarking security, we aim to
advance the discourse on preserving watermark robustness against evolving
forgery threats. It emphasizes the critical importance of developing innovative
solutions to protect digital content and ensure the preservation of ownership
rights in the era of generative AI.

### 6. [HMPNet: A Feature Aggregation Architecture for Maritime Object Detection from a Shipborne Perspective](http://arxiv.org/pdf/2505.08231v1)

Authors: Yu Zhang, Fengyuan Liu, Juan Lyu, Yi Wei, Changdong Yu

In the realm of intelligent maritime navigation, object detection from a
shipborne perspective is paramount. Despite the criticality, the paucity of
maritime-specific data impedes the deployment of sophisticated visual
perception techniques, akin to those utilized in autonomous vehicular systems,
within the maritime context. To bridge this gap, we introduce Navigation12, a
novel dataset annotated for 12 object categories under diverse maritime
environments and weather conditions. Based upon this dataset, we propose
HMPNet, a lightweight architecture tailored for shipborne object detection.
HMPNet incorporates a hierarchical dynamic modulation backbone to bolster
feature aggregation and expression, complemented by a matrix cascading
poly-scale neck and a polymerization weight sharing detector, facilitating
efficient multi-scale feature aggregation. Empirical evaluations indicate that
HMPNet surpasses current state-of-the-art methods in terms of both accuracy and
computational efficiency, realizing a 3.3% improvement in mean Average
Precision over YOLOv11n, the prevailing model, and reducing parameters by 23%.

### 7. [G-MSGINet: A Grouped Multi-Scale Graph-Involution Network for Contactless Fingerprint Recognition](http://arxiv.org/pdf/2505.08233v2)

Authors: Santhoshkumar Peddi, Soham Bandyopadhyay, Debasis Samanta

This paper presents G-MSGINet, a unified and efficient framework for robust
contactless fingerprint recognition that jointly performs minutiae localization
and identity embedding directly from raw input images. Existing approaches rely
on multi-branch architectures, orientation labels, or complex preprocessing
steps, which limit scalability and generalization across real-world acquisition
scenarios. In contrast, the proposed architecture introduces the GMSGI layer, a
novel computational module that integrates grouped pixel-level involution,
dynamic multi-scale kernel generation, and graph-based relational modelling
into a single processing unit. Stacked GMSGI layers progressively refine both
local minutiae-sensitive features and global topological representations
through end-to-end optimization. The architecture eliminates explicit
orientation supervision and adapts graph connectivity directly from learned
kernel descriptors, thereby capturing meaningful structural relationships among
fingerprint regions without fixed heuristics. Extensive experiments on three
benchmark datasets, namely PolyU, CFPose, and Benchmark 2D/3D, demonstrate that
G-MSGINet consistently achieves minutiae F1-scores in the range of
$0.83\pm0.02$ and Rank-1 identification accuracies between 97.0% and 99.1%,
while maintaining an Equal Error Rate (EER) as low as 0.5%. These results
correspond to improvements of up to 4.8% in F1-score and 1.4% in Rank-1
accuracy when compared to prior methods, using only 0.38 million parameters and
6.63 giga floating-point operations, which represents up to ten times fewer
parameters than competitive baselines. This highlights the scalability and
effectiveness of G-MSGINet in real-world contactless biometric recognition
scenarios.

### 8. [EventDiff: A Unified and Efficient Diffusion Model Framework for Event-based Video Frame Interpolation](http://arxiv.org/pdf/2505.08235v1)

Authors: Hanle Zheng, Xujie Han, Zegang Peng, Shangbin Zhang, Guangxun Du, Zhuo Zou, Xilin Wang, Jibin Wu, Hao Guo, Lei Deng

Video Frame Interpolation (VFI) is a fundamental yet challenging task in
computer vision, particularly under conditions involving large motion,
occlusion, and lighting variation. Recent advancements in event cameras have
opened up new opportunities for addressing these challenges. While existing
event-based VFI methods have succeeded in recovering large and complex motions
by leveraging handcrafted intermediate representations such as optical flow,
these designs often compromise high-fidelity image reconstruction under subtle
motion scenarios due to their reliance on explicit motion modeling. Meanwhile,
diffusion models provide a promising alternative for VFI by reconstructing
frames through a denoising process, eliminating the need for explicit motion
estimation or warping operations. In this work, we propose EventDiff, a unified
and efficient event-based diffusion model framework for VFI. EventDiff features
a novel Event-Frame Hybrid AutoEncoder (HAE) equipped with a lightweight
Spatial-Temporal Cross Attention (STCA) module that effectively fuses dynamic
event streams with static frames. Unlike previous event-based VFI methods,
EventDiff performs interpolation directly in the latent space via a denoising
diffusion process, making it more robust across diverse and challenging VFI
scenarios. Through a two-stage training strategy that first pretrains the HAE
and then jointly optimizes it with the diffusion model, our method achieves
state-of-the-art performance across multiple synthetic and real-world event VFI
datasets. The proposed method outperforms existing state-of-the-art event-based
VFI methods by up to 1.98dB in PSNR on Vimeo90K-Triplet and shows superior
performance in SNU-FILM tasks with multiple difficulty levels. Compared to the
emerging diffusion-based VFI approach, our method achieves up to 5.72dB PSNR
gain on Vimeo90K-Triplet and 4.24X faster inference.

### 9. [Congenital Heart Disease recognition using Deep Learning/Transformer models](http://arxiv.org/pdf/2505.08242v1)

Authors: Aidar Amangeldi, Vladislav Yarovenko, Angsar Taigonyrov

Congenital Heart Disease (CHD) remains a leading cause of infant morbidity
and mortality, yet non-invasive screening methods often yield false negatives.
Deep learning models, with their ability to automatically extract features, can
assist doctors in detecting CHD more effectively. In this work, we investigate
the use of dual-modality (sound and image) deep learning methods for CHD
diagnosis. We achieve 73.9% accuracy on the ZCHSound dataset and 80.72%
accuracy on the DICOM Chest X-ray dataset.

### 10. [CNN and ViT Efficiency Study on Tiny ImageNet and DermaMNIST Datasets](http://arxiv.org/pdf/2505.08259v1)

Authors: Aidar Amangeldi, Angsar Taigonyrov, Muhammad Huzaid Jawad, Chinedu Emmanuel Mbonu

This study evaluates the trade-offs between convolutional and
transformer-based architectures on both medical and general-purpose image
classification benchmarks. We use ResNet-18 as our baseline and introduce a
fine-tuning strategy applied to four Vision Transformer variants (Tiny, Small,
Base, Large) on DermatologyMNIST and TinyImageNet. Our goal is to reduce
inference latency and model complexity with acceptable accuracy degradation.
Through systematic hyperparameter variations, we demonstrate that appropriately
fine-tuned Vision Transformers can match or exceed the baseline's performance,
achieve faster inference, and operate with fewer parameters, highlighting their
viability for deployment in resource-constrained environments.

### Computers and Society

### 1. [AI and Generative AI Transforming Disaster Management: A Survey of Damage Assessment and Response Techniques](http://arxiv.org/pdf/2505.08202v1)

Authors: Aman Raj, Lakshit Arora, Sanjay Surendranath Girija, Shashank Kapoor, Dipen Pradhan, Ankit Shetgaonkar

Natural disasters, including earthquakes, wildfires and cyclones, bear a huge
risk on human lives as well as infrastructure assets. An effective response to
disaster depends on the ability to rapidly and efficiently assess the intensity
of damage. Artificial Intelligence (AI) and Generative Artificial Intelligence
(GenAI) presents a breakthrough solution, capable of combining knowledge from
multiple types and sources of data, simulating realistic scenarios of disaster,
and identifying emerging trends at a speed previously unimaginable. In this
paper, we present a comprehensive review on the prospects of AI and GenAI in
damage assessment for various natural disasters, highlighting both its
strengths and limitations. We talk about its application to multimodal data
such as text, image, video, and audio, and also cover major issues of data
privacy, security, and ethical use of the technology during crises. The paper
also recognizes the threat of Generative AI misuse, in the form of
dissemination of misinformation and for adversarial attacks. Finally, we
outline avenues of future research, emphasizing the need for secure, reliable,
and ethical Generative AI systems for disaster management in general. We
believe that this work represents the first comprehensive survey of Gen-AI
techniques being used in the field of Disaster Assessment and Response.

### 2. [How Students Use AI Feedback Matters: Experimental Evidence on Physics Achievement and Autonomy](http://arxiv.org/pdf/2505.08672v1)

Authors: Xusheng Dai, Zhaochun Wen, Jianxiao Jiang, Huiqin Liu, Yu Zhang

Despite the precision and adaptiveness of generative AI (GAI)-powered
feedback provided to students, existing practice and literature might ignore
how usage patterns impact student learning. This study examines the
heterogeneous effects of GAI-powered personalized feedback on high school
students' physics achievement and autonomy through two randomized controlled
trials, with a major focus on usage patterns. Each experiment lasted for five
weeks, involving a total of 387 students. Experiment 1 (n = 121) assessed
compulsory usage of the personalized recommendation system, revealing that
low-achieving students significantly improved academic performance (d = 0.673,
p < 0.05) when receiving AI-generated heuristic solution hints, whereas
medium-achieving students' performance declined (d = -0.539, p < 0.05) with
conventional answers provided by workbook. Notably, high-achieving students
experienced a significant decline in self-regulated learning (d = -0.477, p <
0.05) without any significant gains in achievement. Experiment 2 (n = 266)
investigated the usage pattern of autonomous on-demand help, demonstrating that
fully learner-controlled AI feedback significantly enhanced academic
performance for high-achieving students (d = 0.378, p < 0.05) without
negatively impacting their autonomy. However, autonomy notably declined among
lower achievers exposed to on-demand AI interventions (d = -0.383, p < 0.05),
particularly in the technical-psychological dimension (d = -0.549, p < 0.05),
which has a large overlap with self-regulation. These findings underscore the
importance of usage patterns when applying GAI-powered personalized feedback to
students.

### 3. [Understanding Housing and Homelessness System Access by Linking Administrative Data](http://arxiv.org/pdf/2505.08743v1)

Authors: Geoffrey G. Messier, Sam Elliott, Dallas Seitz

This paper uses privacy preserving methods to link over 235,000 records in
the housing and homelessness system of care (HHSC) of a major North American
city. Several machine learning pairwise linkage and two clustering algorithms
are evaluated for merging the profiles for latent individuals in the data.
Importantly, these methods are evaluated using both traditional machine
learning metrics and HHSC system use metrics generated using the linked data.
The results demonstrate that privacy preserving linkage methods are an
effective and practical method for understanding how a single person interacts
with multiple agencies across an HHSC. They also show that performance
differences between linkage techniques are amplified when evaluated using HHSC
domain specific metrics like number of emergency homeless shelter stays, length
of time interacting with an HHSC and number of emergency shelters visited per
person.

### 4. [One Bad NOFO? AI Governance in Federal Grantmaking](http://arxiv.org/pdf/2505.08133v1)

Authors: Dan Bateyko, Karen Levy

Much scholarship considers how U.S. federal agencies govern artificial
intelligence (AI) through rulemaking and their own internal use policies. But
agencies have an overlooked AI governance role: setting discretionary grant
policy when directing billions of dollars in federal financial assistance.
These dollars enable state and local entities to study, create, and use AI.
This funding not only goes to dedicated AI programs, but also to grantees using
AI in the course of meeting their routine grant objectives. As discretionary
grantmakers, agencies guide and restrict what grant winners do -- a hidden
lever for AI governance. Agencies pull this lever by setting program
objectives, judging criteria, and restrictions for AI use. Using a novel
dataset of over 40,000 non-defense federal grant notices of funding opportunity
(NOFOs) posted to Grants.gov between 2009 and 2024, we analyze how agencies
regulate the use of AI by grantees. We select records mentioning AI and review
their stated goals and requirements. We find agencies promoting AI in notice
narratives, shaping adoption in ways other records of grant policy might fail
to capture. Of the grant opportunities that mention AI, we find only a handful
of AI-specific judging criteria or restrictions. This silence holds even when
agencies fund AI uses in contexts affecting people's rights and which, under an
analogous federal procurement regime, would result in extra oversight. These
findings recast grant notices as a site of AI policymaking -- albeit one that
is developing out of step with other regulatory efforts and incomplete in its
consideration of transparency, accountability, and privacy protections. The
paper concludes by drawing lessons from AI procurement scholarship, while
identifying distinct challenges in grantmaking that invite further study.

### 5. [The Failure of Plagiarism Detection in Competitive Programming](http://arxiv.org/pdf/2505.08244v1)

Authors: Ethan Dickey

Plagiarism in programming courses remains a persistent challenge, especially
in competitive programming contexts where assignments often have unique, known
solutions. This paper examines why traditional code plagiarism detection
methods frequently fail in these environments and explores the implications of
emerging factors such as generative AI (genAI). Drawing on the author's
experience teaching a Competitive Programming 1 (CP1) course over seven
semesters at Purdue University (with $\approx 100$ students each term) and
completely redesigning the CP1/2/3 course sequence, we provide an academically
grounded analysis. We review literature on code plagiarism in computer science
education, survey current detection tools (Moss, Kattis, etc.) and methods
(manual review, code-authorship interviews), and analyze their strengths and
limitations. Experience-based observations are presented to illustrate
real-world detection failures and successes. We find that widely-used automated
similarity checkers can be thwarted by simple code transformations or novel
AI-generated code, while human-centric approaches like oral interviews, though
effective, are labor-intensive. The paper concludes with opinions and
preliminary recommendations for improving academic integrity in programming
courses, advocating for a multi-faceted approach that combines improved
detection algorithms, mastery-based learning techniques, and authentic
assessment practices to better ensure code originality.

### 6. [A Comparison Between Human and Generative AI Decision-Making Attributes in Complex Health Services](http://arxiv.org/pdf/2505.08360v1)

Authors: Nandini Doreswamy, Louise Horstmanshof

A comparison between human and Generative AI decision-making attributes in
complex health services is a knowledge gap in the literature, at present.
Humans may possess unique attributes beneficial to decision-making in complex
health services such as health policy and health regulation, but are also
susceptible to decision-making flaws. The objective is to explore whether
humans have unique, and/or helpful attributes that contribute to optimal
decision-making in complex health services. This comparison may also shed light
on whether humans are likely to compete, cooperate, or converge with Generative
AI. The comparison is based on two published reviews: a scoping review of human
attributes [1] and a rapid review of Generative AI attributes [2]. The analysis
categorizes attributes by uniqueness and impact. The results are presented in
tabular form, comparing the sets and subsets of human and Generative AI
attributes. Humans and Generative AI decision-making attributes have
complementary strengths. Cooperation between these two entities seems more
likely than pure competition. To maintain meaningful decision-making roles,
humans could develop their unique attributes, with decision-making systems
integrating both human and Generative AI contributions. These entities may also
converge, in future.

### 7. [TikTok Search Recommendations: Governance and Research Challenges](http://arxiv.org/pdf/2505.08385v1)

Authors: Taylor Annabell, Robert Gorwa, Rebecca Scharlach, Jacob van de Kerkhof, Thales Bertaglia

Like other social media, TikTok is embracing its use as a search engine,
developing search products to steer users to produce searchable content and
engage in content discovery. Their recently developed product search
recommendations are preformulated search queries recommended to users on
videos. However, TikTok provides limited transparency about how search
recommendations are generated and moderated, despite requirements under
regulatory frameworks like the European Union's Digital Services Act. By
suggesting that the platform simply aggregates comments and common searches
linked to videos, it sidesteps responsibility and issues that arise from
contextually problematic recommendations, reigniting long-standing concerns
about platform liability and moderation. This position paper addresses the
novelty of search recommendations on TikTok by highlighting the challenges that
this feature poses for platform governance and offering a computational
research agenda, drawing on preliminary qualitative analysis. It sets out the
need for transparency in platform documentation, data access and research to
study search recommendations.

### 8. [Reciprocity as the Foundational Substrate of Society: How Reciprocal Dynamics Scale into Social Systems](http://arxiv.org/pdf/2505.08319v1)

Authors: Egil Diau

A major bottleneck in multi-agent AI is the lack of simulateable models for
the bottom-up emergence of social structure under realistic behavioral
constraints. Similarly, many foundational theories in economics and sociology
including the concepts of "institutions" and "norms" tend to describe social
structures post hoc, often relying on implicit assumptions of shared culture,
morality, or symbolic agreement. These concepts are often treated as primitives
rather than reconstructed from agent-level behavior, leaving both their origins
and operational definitions under-specified. To address this, we propose a
three-stage bottom-up framework: Reciprocal Dynamics, capturing
individual-level reciprocal exchanges; Norm Stabilization, the consolidation of
shared expectations; and Institutional Construction, the externalization of
stable patterns into scalable structures. By grounding social emergence in
agent-level reciprocity, our framework enables the systematic exploration of
how moral, cultural, and institutional structures emerge from cognitively
minimal interactions.

### 9. [Small but Significant: On the Promise of Small Language Models for Accessible AIED](http://arxiv.org/pdf/2505.08588v1)

Authors: Yumou Wei, Paulo Carvalho, John Stamper

GPT has become nearly synonymous with large language models (LLMs), an
increasingly popular term in AIED proceedings. A simple keyword-based search
reveals that 61% of the 76 long and short papers presented at AIED 2024
describe novel solutions using LLMs to address some of the long-standing
challenges in education, and 43% specifically mention GPT. Although LLMs
pioneered by GPT create exciting opportunities to strengthen the impact of AI
on education, we argue that the field's predominant focus on GPT and other
resource-intensive LLMs (with more than 10B parameters) risks neglecting the
potential impact that small language models (SLMs) can make in providing
resource-constrained institutions with equitable and affordable access to
high-quality AI tools. Supported by positive results on knowledge component
(KC) discovery, a critical challenge in AIED, we demonstrate that SLMs such as
Phi-2 can produce an effective solution without elaborate prompting strategies.
Hence, we call for more attention to developing SLM-based AIED approaches.

### 10. [Big Data and the Computational Social Science of Entrepreneurship and Innovation](http://arxiv.org/pdf/2505.08706v1)

Authors: Ningzi Li, Shiyang Lai, James Evans

As large-scale social data explode and machine-learning methods evolve,
scholars of entrepreneurship and innovation face new research opportunities but
also unique challenges. This chapter discusses the difficulties of leveraging
large-scale data to identify technological and commercial novelty, document new
venture origins, and forecast competition between new technologies and
commercial forms. It suggests how scholars can take advantage of new text,
network, image, audio, and video data in two distinct ways that advance
innovation and entrepreneurship research. First, machine-learning models,
combined with large-scale data, enable the construction of precision
measurements that function as system-level observatories of innovation and
entrepreneurship across human societies. Second, new artificial intelligence
models fueled by big data generate 'digital doubles' of technology and
business, forming laboratories for virtual experimentation about innovation and
entrepreneurship processes and policies. The chapter argues for the advancement
of theory development and testing in entrepreneurship and innovation by
coupling big data with big models.

### Databases

### 1. [A Unified Model for Cardinality Estimation by Learning from Data and Queries via Sum-Product Networks](http://arxiv.org/pdf/2505.08318v1)

Authors: Jiawei Liu, Ju Fan, Tongyu Liu, Kai Zeng, Jiannan Wang, Quehuan Liu, Tao Ye, Nan Tang

Cardinality estimation is a fundamental component in database systems,
crucial for generating efficient execution plans. Despite advancements in
learning-based cardinality estimation, existing methods may struggle to
simultaneously optimize the key criteria: estimation accuracy, inference time,
and storage overhead, limiting their practical applicability in real-world
database environments. This paper introduces QSPN, a unified model that
integrates both data distribution and query workload. QSPN achieves high
estimation accuracy by modeling data distribution using the simple yet
effective Sum-Product Network (SPN) structure. To ensure low inference time and
reduce storage overhead, QSPN further partitions columns based on query access
patterns. We formalize QSPN as a tree-based structure that extends SPNs by
introducing two new node types: QProduct and QSplit. This paper studies the
research challenges of developing efficient algorithms for the offline
construction and online computation of QSPN. We conduct extensive experiments
to evaluate QSPN in both single-table and multi-table cardinality estimation
settings. The experimental results have demonstrated that QSPN achieves
superior and robust performance on the three key criteria, compared with
state-of-the-art approaches.

### 2. [Information Leakage in Data Linkage](http://arxiv.org/pdf/2505.08596v1)

Authors: Peter Christen, Rainer Schnell, Anushka Vidanage

The process of linking databases that contain sensitive information about
individuals across organisations is an increasingly common requirement in the
health and social science research domains, as well as with governments and
businesses. To protect personal data, protocols have been developed to limit
the leakage of sensitive information. Furthermore, privacy-preserving record
linkage (PPRL) techniques have been proposed to conduct linkage on encoded
data. While PPRL techniques are now being employed in real-world applications,
the focus of PPRL research has been on the technical aspects of linking
sensitive data (such as encoding methods and cryptanalysis attacks), but not on
organisational challenges when employing such techniques in practice. We
analyse what sensitive information can possibly leak, either unintentionally or
intentionally, in traditional data linkage as well as PPRL protocols, and what
a party that participates in such a protocol can learn from the data it obtains
legitimately within the protocol. We also show that PPRL protocols can still
result in the unintentional leakage of sensitive information. We provide
recommendations to help data custodians and other parties involved in a data
linkage project to identify and prevent vulnerabilities and make their project
more secure.

### Distributed, Parallel, and Cluster Computing

### 1. [Kudzu: Fast and Simple High-Throughput BFT](http://arxiv.org/pdf/2505.08771v1)

Authors: Victor Shoup, Jakub Sliwinski, Yann Vonlanthen

We present Kudzu, a high-throughput atomic broadcast protocol with an
integrated fast path. Our contribution is based on the combination of two lines
of work. Firstly, our protocol achieves finality in just two rounds of
communication if all but $p$ out of $n = 3f + 2p + 1$ participating replicas
behave correctly, where $f$ is the number of Byzantine faults that are
tolerated. Due to the seamless integration of the fast path, even in the
presence of more than $p$ faults, our protocol maintains state-of-the-art
characteristics. Secondly, our protocol utilizes the bandwidth of participating
replicas in a balanced way, alleviating the bottleneck at the leader, and thus
enabling high throughput. This is achieved by disseminating blocks using
erasure codes. Despite combining a novel set of advantages, Kudzu is remarkably
simple: intricacies such as progress certificates, complex view changes, and
speculative execution are avoided.

### 2. [Leveraging AI for Productive and Trustworthy HPC Software: Challenges and Research Directions](http://arxiv.org/pdf/2505.08135v1)

Authors: Keita Teranishi, Harshitha Menon, William F. Godoy, Prasanna Balaprakash, David Bau, Tal Ben-Nun, Abhinav Bathele, Franz Franchetti, Michael Franusich, Todd Gamblin, Giorgis Georgakoudis, Tom Goldstein, Arjun Guha, Steven Hahn, Costin Iancu, Zheming Jin, Terry Jones, Tze Meng Low, Het Mankad, Narasinga Rao Miniskar, Mohammad Alaul Haque Monil, Daniel Nichols, Konstantinos Parasyris, Swaroop Pophale, Pedro Valero-Lara, Jeffrey S. Vetter, Samuel Williams, Aaron Young

We discuss the challenges and propose research directions for using AI to
revolutionize the development of high-performance computing (HPC) software. AI
technologies, in particular large language models, have transformed every
aspect of software development. For its part, HPC software is recognized as a
highly specialized scientific field of its own. We discuss the challenges
associated with leveraging state-of-the-art AI technologies to develop such a
unique and niche class of software and outline our research directions in the
two US Department of Energy--funded projects for advancing HPC Software via AI:
Ellora and Durban.

### 3. [Scaling Multi Agent Reinforcement Learning for Underwater Acoustic Tracking via Autonomous Vehicles](http://arxiv.org/pdf/2505.08222v1)

Authors: Matteo Gallici, Ivan Masmitja, Mario Martín

Autonomous vehicles (AV) offer a cost-effective solution for scientific
missions such as underwater tracking. Recently, reinforcement learning (RL) has
emerged as a powerful method for controlling AVs in complex marine
environments. However, scaling these techniques to a fleet--essential for
multi-target tracking or targets with rapid, unpredictable motion--presents
significant computational challenges. Multi-Agent Reinforcement Learning (MARL)
is notoriously sample-inefficient, and while high-fidelity simulators like
Gazebo's LRAUV provide 100x faster-than-real-time single-robot simulations,
they offer no significant speedup for multi-vehicle scenarios, making MARL
training impractical. To address these limitations, we propose an iterative
distillation method that transfers high-fidelity simulations into a simplified,
GPU-accelerated environment while preserving high-level dynamics. This approach
achieves up to a 30,000x speedup over Gazebo through parallelization, enabling
efficient training via end-to-end GPU acceleration. Additionally, we introduce
a novel Transformer-based architecture (TransfMAPPO) that learns multi-agent
policies invariant to the number of agents and targets, significantly improving
sample efficiency. Following large-scale curriculum learning conducted entirely
on GPU, we perform extensive evaluations in Gazebo, demonstrating that our
method maintains tracking errors below 5 meters over extended durations, even
in the presence of multiple fast-moving targets. This work bridges the gap
between large-scale MARL training and high-fidelity deployment, providing a
scalable framework for autonomous fleet control in real-world sea missions.

### 4. [Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing](http://arxiv.org/pdf/2505.08474v1)

Authors: Kuan-Cheng Chen, Chen-Yu Liu, Yu Shang, Felix Burt, Kin K. Leung

We introduce a distributed quantum-classical framework that synergizes
photonic quantum neural networks (QNNs) with matrix-product-state (MPS) mapping
to achieve parameter-efficient training of classical neural networks. By
leveraging universal linear-optical decompositions of $M$-mode interferometers
and photon-counting measurement statistics, our architecture generates neural
parameters through a hybrid quantum-classical workflow: photonic QNNs with
$M(M+1)/2$ trainable parameters produce high-dimensional probability
distributions that are mapped to classical network weights via an MPS model
with bond dimension $\chi$. Empirical validation on MNIST classification
demonstrates that photonic QT achieves an accuracy of $95.50\% \pm 0.84\%$
using 3,292 parameters ($\chi = 10$), compared to $96.89\% \pm 0.31\%$ for
classical baselines with 6,690 parameters. Moreover, a ten-fold compression
ratio is achieved at $\chi = 4$, with a relative accuracy loss of less than
$3\%$. The framework outperforms classical compression techniques (weight
sharing/pruning) by 6--12\% absolute accuracy while eliminating quantum
hardware requirements during inference through classical deployment of
compressed parameters. Simulations incorporating realistic photonic noise
demonstrate the framework's robustness to near-term hardware imperfections.
Ablation studies confirm quantum necessity: replacing photonic QNNs with random
inputs collapses accuracy to chance level ($10.0\% \pm 0.5\%$). Photonic
quantum computing's room-temperature operation, inherent scalability through
spatial-mode multiplexing, and HPC-integrated architecture establish a
practical pathway for distributed quantum machine learning, combining the
expressivity of photonic Hilbert spaces with the deployability of classical
neural networks.

### 5. [Multi-Layer Hierarchical Federated Learning with Quantization](http://arxiv.org/pdf/2505.08145v1)

Authors: Seyed Mohammad Azimi-Abarghouyi, Carlo Fischione

Almost all existing hierarchical federated learning (FL) models are limited
to two aggregation layers, restricting scalability and flexibility in complex,
large-scale networks. In this work, we propose a Multi-Layer Hierarchical
Federated Learning framework (QMLHFL), which appears to be the first study that
generalizes hierarchical FL to arbitrary numbers of layers and network
architectures through nested aggregation, while employing a layer-specific
quantization scheme to meet communication constraints. We develop a
comprehensive convergence analysis for QMLHFL and derive a general convergence
condition and rate that reveal the effects of key factors, including
quantization parameters, hierarchical architecture, and intra-layer iteration
counts. Furthermore, we determine the optimal number of intra-layer iterations
to maximize the convergence rate while meeting a deadline constraint that
accounts for both communication and computation times. Our results show that
QMLHFL consistently achieves high learning accuracy, even under high data
heterogeneity, and delivers notably improved performance when optimized,
compared to using randomly selected values.

### Digital Libraries

### 1. [How are research data referenced? The use case of the research data repository RADAR](http://arxiv.org/pdf/2505.08533v1)

Authors: Dorothea Strecker, Kerstin Soltau, Felix Bach

Publishing research data aims to improve the transparency of research results
and facilitate the reuse of datasets. In both cases, referencing the datasets
that were used is recommended. Research data repositories can support data
referencing through various measures and also benefit from it, for example
using this information to demonstrate their impact. However, the literature
shows that the practice of formally citing research data is not widespread,
data metrics are not yet established, and effective incentive structures are
lacking. This article examines how often and in what form datasets published
via the research data repository RADAR are referenced. For this purpose, the
data sources Google Scholar, DataCite Event Data and the Data Citation Corpus
were analyzed. The analysis shows that 27.9 % of the datasets in the repository
were referenced at least once. 21.4 % of these references were (also) present
in the reference lists and are therefore considered data citations. Datasets
were referenced often in data availability statements. A comparison of the
three data sources showed that there was little overlap in the coverage of
references. In most cases (75.8 %), data and referencing objects were published
in the same year. Two definition approaches were considered to investigate data
reuse. 118 RADAR datasets were referenced more than once. Only 21 references
had no overlaps in the authorship information -- these datasets were referenced
by researchers that were not involved in data collection.

### Discrete Mathematics

### 1. [Isolation Forest in Novelty Detection Scenario](http://arxiv.org/pdf/2505.08489v1)

Authors: Adam Ulrich, Jan Krňávek, Roman Šenkeřík, Zuzana Komínková Oplatková, Radek Vala

Data mining offers a diverse toolbox for extracting meaningful structures
from complex datasets, with anomaly detection emerging as a critical subfield
particularly in the context of streaming or real-time data. Within anomaly
detection, novelty detection focuses on identifying previously unseen patterns
after training solely on regular data. While classic algorithms such as
One-Class SVM or Local Outlier Factor (LOF) have been widely applied, they
often lack interpretability and scalability. In this work, we explore the
Half-Space Tree (HST) algorithm, originally proposed for streaming anomaly
detection, and propose a novel theoretical modification to adapt it
specifically for novelty detection tasks. Our approach is grounded in the idea
that anomalies i.e., novelties tend to appear in the higher leaves of the tree,
which are less frequently visited by regular instances. We analytically
demonstrate the effectiveness of this approach using probabilistic analysis,
expected depth (EXD) calculations, and combinatorial reasoning. A comparative
analysis of expected depths between our modified HST and the original Isolation
Forest highlights that novelty points are significantly more isolated in our
approach. This supports the hypothesis that HSTs, with appropriate structural
adaptation, can serve as interpretable and efficient novelty detectors. The
paper contributes a theoretical foundation and supporting analysis for this
adaptation, setting the stage for further application and experimentation.

### Data Structures and Algorithms

### 1. [Tensor Sketch: Fast and Scalable Polynomial Kernel Approximation](http://arxiv.org/pdf/2505.08146v1)

Authors: Ninh Pham, Rasmus Pagh

Approximation of non-linear kernels using random feature maps has become a
powerful technique for scaling kernel methods to large datasets. We propose
\textit{Tensor Sketch}, an efficient random feature map for approximating
polynomial kernels. Given $n$ training samples in $\R^d$ Tensor Sketch computes
low-dimensional embeddings in $\R^D$ in time $\BO{n(d+D \log{D})}$ making it
well-suited for high-dimensional and large-scale settings. We provide
theoretical guarantees on the approximation error, ensuring the fidelity of the
resulting kernel function estimates. We also discuss extensions and highlight
applications where Tensor Sketch serves as a central computational tool.

### 2. [Uniform Universal Sets, Splitters, and Bisectors](http://arxiv.org/pdf/2505.08308v1)

Authors: Elisabet Burjons, Peter Rossmanith

Given a subset of size $k$ of a very large universe a randomized way to find
this subset could consist of deleting half of the universe and then searching
the remaining part. With a probability of $2^{-k}$ one will succeed. By
probability amplification, a randomized algorithm needs about $2^k$ rounds
until it succeeds. We construct bisectors that derandomize this process and
have size~$2^{k+o(k)}$. One application is derandomization of reductions
between average case complexity classes. We also construct uniform
$(n,k)$-universal sets that generalize universal sets in such a way that they
are bisectors at the same time. This construction needs only linear time and
produces families of asymptotically optimal size without using advanced
combinatorial constructions as subroutines, which previous families did, but
are basedmainly on modulo functions and refined brute force search.

### Emerging Technologies

### 1. [Attention-based Generative Latent Replay: A Continual Learning Approach for WSI Analysis](http://arxiv.org/pdf/2505.08524v1)

Authors: Pratibha Kumari, Daniel Reisenbüchler, Afshin Bozorgpour, Nadine S. Schaadt, Friedrich Feuerhake, Dorit Merhof

Whole slide image (WSI) classification has emerged as a powerful tool in
computational pathology, but remains constrained by domain shifts, e.g., due to
different organs, diseases, or institution-specific variations. To address this
challenge, we propose an Attention-based Generative Latent Replay Continual
Learning framework (AGLR-CL), in a multiple instance learning (MIL) setup for
domain incremental WSI classification. Our method employs Gaussian Mixture
Models (GMMs) to synthesize WSI representations and patch count distributions,
preserving knowledge of past domains without explicitly storing original data.
A novel attention-based filtering step focuses on the most salient patch
embeddings, ensuring high-quality synthetic samples. This privacy-aware
strategy obviates the need for replay buffers and outperforms other buffer-free
counterparts while matching the performance of buffer-based solutions. We
validate AGLR-CL on clinically relevant biomarker detection and molecular
status prediction across multiple public datasets with diverse centers, organs,
and patient cohorts. Experimental results confirm its ability to retain prior
knowledge and adapt to new domains, offering an effective, privacy-preserving
avenue for domain incremental continual learning in WSI classification.

### 2. [Blockchain Technology: Core Mechanisms, Evolution, and Future Implementation Challenges](http://arxiv.org/pdf/2505.08772v1)

Authors: Aditya Pratap Singh

Blockchain technology has emerged as one of the most transformative digital
innovations of the 21st century. This paper presents a comprehensive review of
blockchain's fundamental architecture, tracing its development from Bitcoin's
initial implementation to current enterprise applications. We examine the core
technical components including distributed consensus algorithms, cryptographic
principles, and smart contract functionality that enable blockchain's unique
properties. The historical progression from cryptocurrency-focused systems to
robust platforms for decentralized applications is analyzed, highlighting
pivotal developments in scalability, privacy, and interoperability.
Additionally, we identify critical challenges facing widespread blockchain
adoption, including technical limitations, regulatory hurdles, and integration
complexities with existing systems. By providing this foundational
understanding of blockchain technology, this paper contributes to ongoing
research efforts addressing blockchain's potential to revolutionize data
management across industries.

### Formal Languages and Automata Theory

### 1. [Lost in Transmission: When and Why LLMs Fail to Reason Globally](http://arxiv.org/pdf/2505.08140v1)

Authors: Tobias Schnabel, Kiran Tomlinson, Adith Swaminathan, Jennifer Neville

Despite their many successes, transformer-based large language models (LLMs)
continue to struggle with tasks that require complex reasoning over large parts
of their input. We argue that these failures arise due to capacity limits on
the accurate flow of information within LLMs. To formalize this issue, we
introduce the bounded attention prefix oracle (BAPO) model, a new computational
framework that models bandwidth constraints on attention heads, the mechanism
for internal communication in LLMs. We show that several important reasoning
problems like graph reachability require high communication bandwidth for BAPOs
to solve; we call these problems BAPO-hard. Our experiments corroborate our
theoretical predictions: GPT-4, Claude, and Gemini succeed on BAPO-easy tasks
and fail even on relatively small BAPO-hard tasks. BAPOs also reveal another
benefit of chain of thought (CoT): we prove that breaking down a task using CoT
can turn any BAPO-hard problem into a BAPO-easy one. Our results offer
principled explanations for key LLM failures and suggest directions for
architectures and inference methods that mitigate bandwidth limits.

### Graphics

### 1. [ACT-R: Adaptive Camera Trajectories for 3D Reconstruction from Single Image](http://arxiv.org/pdf/2505.08239v1)

Authors: Yizhi Wang, Mingrui Zhao, Ali Mahdavi-Amiri, Hao Zhang

We introduce adaptive view planning to multi-view synthesis, aiming to
improve both occlusion revelation and 3D consistency for single-view 3D
reconstruction. Instead of generating an unordered set of views independently
or simultaneously, we generate a sequence of views, leveraging temporal
consistency to enhance 3D coherence. Most importantly, our view sequence is not
determined by a pre-determined camera setup. Instead, we compute an adaptive
camera trajectory (ACT), specifically, an orbit of camera views, which
maximizes the visibility of occluded regions of the 3D object to be
reconstructed. Once the best orbit is found, we feed it to a video diffusion
model to generate novel views around the orbit, which in turn, are passed to a
multi-view 3D reconstruction model to obtain the final reconstruction. Our
multi-view synthesis pipeline is quite efficient since it involves no run-time
training/optimization, only forward inferences by applying the pre-trained
models for occlusion analysis and multi-view synthesis. Our method predicts
camera trajectories that reveal occlusions effectively and produce consistent
novel views, significantly improving 3D reconstruction over SOTA on the unseen
GSO dataset, both quantitatively and qualitatively.

### 2. [Large Language Models for Computer-Aided Design: A Survey](http://arxiv.org/pdf/2505.08137v1)

Authors: Licheng Zhang, Bach Le, Naveed Akhtar, Siew-Kei Lam, Tuan Ngo

Large Language Models (LLMs) have seen rapid advancements in recent years,
with models like ChatGPT and DeepSeek, showcasing their remarkable capabilities
across diverse domains. While substantial research has been conducted on LLMs
in various fields, a comprehensive review focusing on their integration with
Computer-Aided Design (CAD) remains notably absent. CAD is the industry
standard for 3D modeling and plays a vital role in the design and development
of products across different industries. As the complexity of modern designs
increases, the potential for LLMs to enhance and streamline CAD workflows
presents an exciting frontier. This article presents the first systematic
survey exploring the intersection of LLMs and CAD. We begin by outlining the
industrial significance of CAD, highlighting the need for AI-driven innovation.
Next, we provide a detailed overview of the foundation of LLMs. We also examine
both closed-source LLMs as well as publicly available models. The core of this
review focuses on the various applications of LLMs in CAD, providing a taxonomy
of six key areas where these models are making considerable impact. Finally, we
propose several promising future directions for further advancements, which
offer vast opportunities for innovation and are poised to shape the future of
CAD technology. Github:
https://github.com/lichengzhanguom/LLMs-CAD-Survey-Taxonomy

### 3. [Claycode: Stylable and Deformable 2D Scannable Codes](http://arxiv.org/pdf/2505.08666v1)

Authors: Marco Maida, Alberto Crescini, Marco Perronet, Elena Camuffo

This paper introduces Claycode, a novel 2D scannable code designed for
extensive stylization and deformation. Unlike traditional matrix-based codes
(e.g., QR codes), Claycodes encode their message in a tree structure. During
the encoding process, bits are mapped into a topology tree, which is then
depicted as a nesting of color regions drawn within the boundaries of a target
polygon shape. When decoding, Claycodes are extracted and interpreted in
real-time from a camera stream. We detail the end-to-end pipeline and show that
Claycodes allow for extensive stylization without compromising their
functionality. We then empirically demonstrate Claycode's high tolerance to
heavy deformations, outperforming traditional 2D scannable codes in scenarios
where they typically fail.

### 4. [CAD-Coder:Text-Guided CAD Files Code Generation](http://arxiv.org/pdf/2505.08686v1)

Authors: Changqi He, Shuhan Zhang, Liguo Zhang, Jiajun Miao

Computer-aided design (CAD) is a way to digitally create 2D drawings and 3D
models of real-world products. Traditional CAD typically relies on hand-drawing
by experts or modifications of existing library files, which doesn't allow for
rapid personalization. With the emergence of generative artificial
intelligence, convenient and efficient personalized CAD generation has become
possible. However, existing generative methods typically produce outputs that
lack interactive editability and geometric annotations, limiting their
practical applications in manufacturing. To enable interactive generative CAD,
we propose CAD-Coder, a framework that transforms natural language instructions
into CAD script codes, which can be executed in Python environments to generate
human-editable CAD files (.Dxf). To facilitate the generation of editable CAD
sketches with annotation information, we construct a comprehensive dataset
comprising 29,130 Dxf files with their corresponding script codes, where each
sketch preserves both editability and geometric annotations. We evaluate
CAD-Coder on various 2D/3D CAD generation tasks against existing methods,
demonstrating superior interactive capabilities while uniquely providing
editable sketches with geometric annotations.

### 5. [M3G: Multi-Granular Gesture Generator for Audio-Driven Full-Body Human Motion Synthesis](http://arxiv.org/pdf/2505.08293v1)

Authors: Zhizhuo Yin, Yuk Hang Tsui, Pan Hui

Generating full-body human gestures encompassing face, body, hands, and
global movements from audio is a valuable yet challenging task in virtual
avatar creation. Previous systems focused on tokenizing the human gestures
framewisely and predicting the tokens of each frame from the input audio.
However, one observation is that the number of frames required for a complete
expressive human gesture, defined as granularity, varies among different human
gesture patterns. Existing systems fail to model these gesture patterns due to
the fixed granularity of their gesture tokens. To solve this problem, we
propose a novel framework named Multi-Granular Gesture Generator (M3G) for
audio-driven holistic gesture generation. In M3G, we propose a novel
Multi-Granular VQ-VAE (MGVQ-VAE) to tokenize motion patterns and reconstruct
motion sequences from different temporal granularities. Subsequently, we
proposed a multi-granular token predictor that extracts multi-granular
information from audio and predicts the corresponding motion tokens. Then M3G
reconstructs the human gestures from the predicted tokens using the MGVQ-VAE.
Both objective and subjective experiments demonstrate that our proposed M3G
framework outperforms the state-of-the-art methods in terms of generating
natural and expressive full-body human gestures.

### Computer Science and Game Theory

### 1. [Optimal Prize Design in Parallel Rank-order Contests](http://arxiv.org/pdf/2505.08342v1)

Authors: Xiaotie Deng, Ningyuan Li, Weian Li, Qi Qi

This paper investigates a two-stage game-theoretical model with multiple
parallel rank-order contests. In this model, each contest designer sets up a
contest and determines the prize structure within a fixed budget in the first
stage. Contestants choose which contest to participate in and exert costly
effort to compete against other participants in the second stage. First, we
fully characterize the symmetric Bayesian Nash equilibrium in the subgame of
contestants, accounting for both contest selection and effort exertion, under
any given prize structures. Notably, we find that, regardless of whether
contestants know the number of participants in their chosen contest, the
equilibrium remains unchanged in expectation. Next, we analyze the designers'
strategies under two types of objective functions based on effort and
participation, respectively. For a broad range of effort-based objectives, we
demonstrate that the winner-takes-all prize structure-optimal in the
single-contest setting-remains a dominant strategy for all designers. For the
participation objective, which maximizes the number of participants surpassing
a skill threshold, we show that the optimal prize structure is always a simple
contest. Furthermore, the equilibrium among designers is computationally
tractable when they share a common threshold.

### Human-Computer Interaction

### 1. [Investigating Resolution Strategies for Workspace-Occlusion in Augmented Virtuality](http://arxiv.org/pdf/2505.08312v1)

Authors: Nico Feld, Pauline Bimberg, Michael Feldmann, Matthias Wölwer, Eike Langbehn, Benjamin Weyers, Daniel Zielasko

Augmented Virtuality integrates physical content into virtual environments,
but the occlusion of physical by virtual content is a challenge. This unwanted
occlusion may disrupt user interactions with physical devices and compromise
safety and usability. This paper investigates two resolution strategies to
address this issue: Redirected Walking, which subtly adjusts the user's
movement to maintain physical-virtual alignment, and Automatic Teleport
Rotation, which realigns the virtual environment during travel. A user study
set in a virtual forest demonstrates that both methods effectively reduce
occlusion. While in our testbed, Automatic Teleport Rotation achieves higher
occlusion resolution, it is suspected to increase cybersickness compared to the
less intrusive Redirected Walking approach.

### 2. [Human-in-the-Loop Optimization for Inclusive Design: Balancing Automation and Designer Expertise](http://arxiv.org/pdf/2505.08375v1)

Authors: Pascal Jansen

Accessible and inclusive design has gained increased attention in HCI, yet
practical implementation remains challenging due to resource-intensive
prototyping methods. Traditional approaches such as workshops, A-B tests, and
co-design sessions struggle to capture the diverse and complex needs of users
with disabilities at scale. This position paper argues for an automated,
accessible Human-in-the-Loop (HITL) design optimization process that shifts the
designer's role from directly crafting prototypes to curating constraints for
algorithmic exploration. By pre-constraining the design space based on specific
user interaction needs, integrating adaptive multi-modal feedback channels, and
personalizing feedback prompts, the HITL approach could efficiently refine
design parameters, such as text size, color contrast, layout, and interaction
modalities, to achieve optimal accessibility. This approach promises scalable,
individualized design solutions while raising critical questions about
constraint curation, transparency, user agency, and ethical considerations,
making it essential to discuss and refine these ideas collaboratively at the
workshop.

### 3. [BizChat: Scaffolding AI-Powered Business Planning for Small Business Owners Across Digital Skill Levels](http://arxiv.org/pdf/2505.08493v1)

Authors: Quentin Romero Lauro, Aakash Gautam, Yasmine Kotturi

Generative AI can help small business owners automate tasks, increase
efficiency, and improve their bottom line. However, despite the seemingly
intuitive design of systems like ChatGPT, significant barriers remain for those
less comfortable with technology. To address these disparities, prior work
highlights accessory skills -- beyond prompt engineering -- users must master
to successfully adopt generative AI including keyboard shortcuts, editing
skills, file conversions, and browser literacy. Building on a design workshop
series and 15 interviews with small businesses, we introduce BizChat, a large
language model (LLM)-powered web application that helps business owners across
digital skills levels write their business plan -- an essential but often
neglected document. To do so, BizChat's interface embodies three design
considerations inspired by learning sciences: ensuring accessibility to users
with less digital skills while maintaining extensibility to power users
("low-floor-high-ceiling"), providing in situ micro-learning to support
entrepreneurial education ("just-in-time learning"), and framing interaction
around business activities ("contextualized technology introduction"). We
conclude with plans for a future BizChat deployment.

### 4. [Communication Styles and Reader Preferences of LLM and Human Experts in Explaining Health Information](http://arxiv.org/pdf/2505.08143v1)

Authors: Jiawei Zhou, Kritika Venkatachalam, Minje Choi, Koustuv Saha, Munmun De Choudhury

With the wide adoption of large language models (LLMs) in information
assistance, it is essential to examine their alignment with human communication
styles and values. We situate this study within the context of fact-checking
health information, given the critical challenge of rectifying conceptions and
building trust. Recent studies have explored the potential of LLM for health
communication, but style differences between LLMs and human experts and
associated reader perceptions remain under-explored. In this light, our study
evaluates the communication styles of LLMs, focusing on how their explanations
differ from those of humans in three core components of health communication:
information, sender, and receiver. We compiled a dataset of 1498 health
misinformation explanations from authoritative fact-checking organizations and
generated LLM responses to inaccurate health information. Drawing from health
communication theory, we evaluate communication styles across three key
dimensions of information linguistic features, sender persuasive strategies,
and receiver value alignments. We further assessed human perceptions through a
blinded evaluation with 99 participants. Our findings reveal that LLM-generated
articles showed significantly lower scores in persuasive strategies, certainty
expressions, and alignment with social values and moral foundations. However,
human evaluation demonstrated a strong preference for LLM content, with over
60% responses favoring LLM articles for clarity, completeness, and
persuasiveness. Our results suggest that LLMs' structured approach to
presenting information may be more effective at engaging readers despite
scoring lower on traditional measures of quality in fact-checking and health
communication.

### 5. [A Comparison Between Human and Generative AI Decision-Making Attributes in Complex Health Services](http://arxiv.org/pdf/2505.08360v1)

Authors: Nandini Doreswamy, Louise Horstmanshof

A comparison between human and Generative AI decision-making attributes in
complex health services is a knowledge gap in the literature, at present.
Humans may possess unique attributes beneficial to decision-making in complex
health services such as health policy and health regulation, but are also
susceptible to decision-making flaws. The objective is to explore whether
humans have unique, and/or helpful attributes that contribute to optimal
decision-making in complex health services. This comparison may also shed light
on whether humans are likely to compete, cooperate, or converge with Generative
AI. The comparison is based on two published reviews: a scoping review of human
attributes [1] and a rapid review of Generative AI attributes [2]. The analysis
categorizes attributes by uniqueness and impact. The results are presented in
tabular form, comparing the sets and subsets of human and Generative AI
attributes. Humans and Generative AI decision-making attributes have
complementary strengths. Cooperation between these two entities seems more
likely than pure competition. To maintain meaningful decision-making roles,
humans could develop their unique attributes, with decision-making systems
integrating both human and Generative AI contributions. These entities may also
converge, in future.

### 6. [CoVoL: A Cooperative Vocabulary Learning Game for Children with Autism](http://arxiv.org/pdf/2505.08515v1)

Authors: Pawel Chodkiewicz, Pragya Verma, Grischa Liebel

Children with Autism commonly face difficulties in vocabulary acquisition,
which can have an impact on their social communication. Using digital tools for
vocabulary learning can prove beneficial for these children, as they can
provide a predictable environment and effective individualized feedback. While
existing work has explored the use of technology-assisted vocabulary learning
for children with Autism, no study has incorporated turn-taking to facilitate
learning and use of vocabulary similar to that used in real-world social
contexts. To address this gap, we propose the design of a cooperative
two-player vocabulary learning game, CoVoL. CoVoL allows children to engage in
game-based vocabulary learning useful for real-world social communication
scenarios. We discuss our first prototype and its evaluation. Additionally, we
present planned features which are based on feedback obtained through ten
interviews with researchers and therapists, as well as an evaluation plan for
the final release of CoVoL.

### 7. [Integrating Natural Language Processing and Exercise Monitoring for Early Diagnosis of Metabolic Syndrome: A Deep Learning Approach](http://arxiv.org/pdf/2505.08628v1)

Authors: Yichen Zhao, Yuhua Wang, Xi Cheng, Junhao Fang, Yang Yang

Metabolic syndrome (MetS) is a medication condition characterized by
abdominal obesity, insulin resistance, hypertension and hyperlipidemia. It
increases the risk of majority of chronic diseases, including type 2 diabetes
mellitus, and affects about one quarter of the global population. Therefore,
early detection and timely intervention for MetS are crucial. Standard
diagnosis for MetS components requires blood tests conducted within medical
institutions. However, it is frequently underestimated, leading to unmet need
for care for MetS population. This study aims to use the least physiological
data and free texts about exercises related activities, which are obtained
easily in daily life, to diagnosis MetS. We collected the data from 40
volunteers in a nursing home and used data augmentation to reduce the
imbalance. We propose a deep learning framework for classifying MetS that
integrates natural language processing (NLP) and exercise monitoring. The
results showed that the best model reported a high positive result (AUROC=0.806
and REC=76.3%) through 3-fold cross-validation. Feature importance analysis
revealed that text and minimum heart rate on a daily basis contribute the most
in the classification of MetS. This study demonstrates the potential
application of data that are easily measurable in daily life for the early
diagnosis of MetS, which could contribute to reducing the cost of screening and
management for MetS population.

### 8. [Enhancing Software Development with Context-Aware Conversational Agents: A User Study on Developer Interactions with Chatbots](http://arxiv.org/pdf/2505.08648v1)

Authors: Glaucia Melo, Paulo Alencar, Donald Cowan

Software development is a cognitively intensive process requiring
multitasking, adherence to evolving workflows, and continuous learning. With
the rise of large language model (LLM)-based tools, such as conversational
agents (CAs), there is growing interest in supporting developers through
natural language interaction. However, little is known about the specific
features developers seek in these systems. We conducted a user study with 29
developers using a prototype text-based chatbot to investigate preferred
functionalities. Our findings reveal strong interest in task automation,
version control support, and contextual adaptability, especially the need to
tailor assistance for both novice and experienced users. We highlight the
importance of deep contextual understanding, historical interaction awareness,
and personalized support in CA design. This study contributes to the
development of context-aware chatbots that enhance productivity and
satisfaction, and it outlines opportunities for future research on human-AI
collaboration in software engineering.

### 9. [VizCV: AI-assisted visualization of researchers' publications tracks](http://arxiv.org/pdf/2505.08691v1)

Authors: Vladimír Lazárik, Marco Agus, Barbora Kozlíková, Pere-Pau Vázquez

Analyzing how the publication records of scientists and research groups have
evolved over the years is crucial for assessing their expertise since it can
support the management of academic environments by assisting with career
planning and evaluation. We introduce VizCV, a novel web-based end-to-end
visual analytics framework that enables the interactive exploration of
researchers' scientific trajectories. It incorporates AI-assisted analysis and
supports automated reporting of career evolution. Our system aims to model
career progression through three key dimensions: a) research topic evolution to
detect and visualize shifts in scholarly focus over time, b) publication record
and the corresponding impact, c) collaboration dynamics depicting the growth
and transformation of a researcher's co-authorship network. AI-driven insights
provide automated explanations of career transitions, detecting significant
shifts in research direction, impact surges, or collaboration expansions. The
system also supports comparative analysis between researchers, allowing users
to compare topic trajectories and impact growth. Our interactive, multi-tab and
multiview system allows for the exploratory analysis of career milestones under
different perspectives, such as the most impactful articles, emerging research
themes, or obtaining a detailed analysis of the contribution of the researcher
in a subfield. The key contributions include AI/ML techniques for: a) topic
analysis, b) dimensionality reduction for visualizing patterns and trends, c)
the interactive creation of textual descriptions of facets of data through
configurable prompt generation and large language models, that include key
indicators, to help understanding the career development of individuals or
groups.

### 10. [Large Language Model Psychometrics: A Systematic Review of Evaluation, Validation, and Enhancement](http://arxiv.org/pdf/2505.08245v1)

Authors: Haoran Ye, Jing Jin, Yuhang Xie, Xin Zhang, Guojie Song

The rapid advancement of large language models (LLMs) has outpaced
traditional evaluation methodologies. It presents novel challenges, such as
measuring human-like psychological constructs, navigating beyond static and
task-specific benchmarks, and establishing human-centered evaluation. These
challenges intersect with Psychometrics, the science of quantifying the
intangible aspects of human psychology, such as personality, values, and
intelligence. This survey introduces and synthesizes an emerging
interdisciplinary field of LLM Psychometrics, which leverages psychometric
instruments, theories, and principles to evaluate, understand, and enhance
LLMs. We systematically explore the role of Psychometrics in shaping
benchmarking principles, broadening evaluation scopes, refining methodologies,
validating results, and advancing LLM capabilities. This paper integrates
diverse perspectives to provide a structured framework for researchers across
disciplines, enabling a more comprehensive understanding of this nascent field.
Ultimately, we aim to provide actionable insights for developing future
evaluation paradigms that align with human-level AI and promote the advancement
of human-centered AI systems for societal benefit. A curated repository of LLM
psychometric resources is available at
https://github.com/valuebyte-ai/Awesome-LLM-Psychometrics.

### Information Retrieval

### 1. [Lost in Transliteration: Bridging the Script Gap in Neural IR](http://arxiv.org/pdf/2505.08411v1)

Authors: Andreas Chari, Iadh Ounis, Sean MacAvaney

Most human languages use scripts other than the Latin alphabet. Search users
in these languages often formulate their information needs in a transliterated
-- usually Latinized -- form for ease of typing. For example, Greek speakers
might use Greeklish, and Arabic speakers might use Arabizi. This paper shows
that current search systems, including those that use multilingual dense
embeddings such as BGE-M3, do not generalise to this setting, and their
performance rapidly deteriorates when exposed to transliterated queries. This
creates a ``script gap" between the performance of the same queries when
written in their native or transliterated form. We explore whether adapting the
popular ``translate-train" paradigm to transliterations can enhance the
robustness of multilingual Information Retrieval (IR) methods and bridge the
gap between native and transliterated scripts. By exploring various
combinations of non-Latin and Latinized query text for training, we investigate
whether we can enhance the capacity of existing neural retrieval techniques and
enable them to apply to this important setting. We show that by further
fine-tuning IR models on an even mixture of native and Latinized text, they can
perform this cross-script matching at nearly the same performance as when the
query was formulated in the native script. Out-of-domain evaluation and further
qualitative analysis show that transliterations can also cause queries to lose
some of their nuances, motivating further research in this direction.

### 2. [Interest Changes: Considering User Interest Life Cycle in Recommendation System](http://arxiv.org/pdf/2505.08471v1)

Authors: Yinjiang Cai, Jiangpan Hou, Yangping Zhu, Yuan Nie

In recommendation systems, user interests are always in a state of constant
flux. Typically, a user interest experiences a emergent phase, a stable phase,
and a declining phase, which are referred to as the "user interest life-cycle".
Recent papers on user interest modeling have primarily focused on how to
compute the correlation between the target item and user's historical
behaviors, without thoroughly considering the life-cycle features of user
interest. In this paper, we propose an effective method called Deep Interest
Life-cycle Network (DILN), which not only captures the interest life-cycle
features efficiently, but can also be easily integrated to existing ranking
models. DILN contains two key components: Interest Life-cycle Encoder Module
constructs historical activity histograms of the user interest and then encodes
them into dense representation. Interest Life-cycle Fusion Module injects the
encoded dense representation into multiple expert networks, with the aim of
enabling the specific phase of interest life-cycle to activate distinct
experts. Online A/B testing reveals that DILN achieves significant improvements
of +0.38% in CTR, +1.04% in CVR and +0.25% in duration per user, which
demonstrates its effectiveness. In addition, DILN inherently increase the
exposure of users' emergent and stable interests while decreasing the exposure
of declining interests. DILN has been deployed on the Lofter App.

### 3. [Hyperbolic Contrastive Learning with Model-augmentation for Knowledge-aware Recommendation](http://arxiv.org/pdf/2505.08157v1)

Authors: Shengyin Sun, Chen Ma

Benefiting from the effectiveness of graph neural networks (GNNs) and
contrastive learning, GNN-based contrastive learning has become mainstream for
knowledge-aware recommendation. However, most existing contrastive
learning-based methods have difficulties in effectively capturing the
underlying hierarchical structure within user-item bipartite graphs and
knowledge graphs. Moreover, they commonly generate positive samples for
contrastive learning by perturbing the graph structure, which may lead to a
shift in user preference learning. To overcome these limitations, we propose
hyperbolic contrastive learning with model-augmentation for knowledge-aware
recommendation. To capture the intrinsic hierarchical graph structures, we
first design a novel Lorentzian knowledge aggregation mechanism, which enables
more effective representations of users and items. Then, we propose three
model-level augmentation techniques to assist Hyperbolic contrastive learning.
Different from the classical structure-level augmentation (e.g., edge
dropping), the proposed model-augmentations can avoid preference shifts between
the augmented positive pair. Finally, we conduct extensive experiments to
demonstrate the superiority (maximum improvement of $11.03\%$) of proposed
methods over existing baselines.

### 4. [TikTok Search Recommendations: Governance and Research Challenges](http://arxiv.org/pdf/2505.08385v1)

Authors: Taylor Annabell, Robert Gorwa, Rebecca Scharlach, Jacob van de Kerkhof, Thales Bertaglia

Like other social media, TikTok is embracing its use as a search engine,
developing search products to steer users to produce searchable content and
engage in content discovery. Their recently developed product search
recommendations are preformulated search queries recommended to users on
videos. However, TikTok provides limited transparency about how search
recommendations are generated and moderated, despite requirements under
regulatory frameworks like the European Union's Digital Services Act. By
suggesting that the platform simply aggregates comments and common searches
linked to videos, it sidesteps responsibility and issues that arise from
contextually problematic recommendations, reigniting long-standing concerns
about platform liability and moderation. This position paper addresses the
novelty of search recommendations on TikTok by highlighting the challenges that
this feature poses for platform governance and offering a computational
research agenda, drawing on preliminary qualitative analysis. It sets out the
need for transparency in platform documentation, data access and research to
study search recommendations.

### 5. [Securing RAG: A Risk Assessment and Mitigation Framework](http://arxiv.org/pdf/2505.08728v1)

Authors: Lukas Ammann, Sara Ott, Christoph R. Landolt, Marco P. Lehmann

Retrieval Augmented Generation (RAG) has emerged as the de facto industry
standard for user-facing NLP applications, offering the ability to integrate
data without re-training or fine-tuning Large Language Models (LLMs). This
capability enhances the quality and accuracy of responses but also introduces
novel security and privacy challenges, particularly when sensitive data is
integrated. With the rapid adoption of RAG, securing data and services has
become a critical priority. This paper first reviews the vulnerabilities of RAG
pipelines, and outlines the attack surface from data pre-processing and data
storage management to integration with LLMs. The identified risks are then
paired with corresponding mitigations in a structured overview. In a second
step, the paper develops a framework that combines RAG-specific security
considerations, with existing general security guidelines, industry standards,
and best practices. The proposed framework aims to guide the implementation of
robust, compliant, secure, and trustworthy RAG systems.

### Machine Learning

### 1. [A Multi-scale Representation Learning Framework for Long-Term Time Series Forecasting](http://arxiv.org/pdf/2505.08199v1)

Authors: Boshi Gao, Qingjian Ni, Fanbo Ju, Yu Chen, Ziqi Zhao

Long-term time series forecasting (LTSF) offers broad utility in practical
settings like energy consumption and weather prediction. Accurately predicting
long-term changes, however, is demanding due to the intricate temporal patterns
and inherent multi-scale variations within time series. This work confronts key
issues in LTSF, including the suboptimal use of multi-granularity information,
the neglect of channel-specific attributes, and the unique nature of trend and
seasonal components, by introducing a proficient MLP-based forecasting
framework. Our method adeptly disentangles complex temporal dynamics using
clear, concurrent predictions across various scales. These multi-scale
forecasts are then skillfully integrated through a system that dynamically
assigns importance to information from different granularities, sensitive to
individual channel characteristics. To manage the specific features of temporal
patterns, a two-pronged structure is utilized to model trend and seasonal
elements independently. Experimental results on eight LTSF benchmarks
demonstrate that MDMixer improves average MAE performance by 4.64% compared to
the recent state-of-the-art MLP-based method (TimeMixer), while achieving an
effective balance between training efficiency and model interpretability.

### 2. [An Effective Flow-based Method for Positive-Unlabeled Learning: 2-HNC](http://arxiv.org/pdf/2505.08212v1)

Authors: Dorit Hochbaum, Torpong Nitayanont

In many scenarios of binary classification, only positive instances are
provided in the training data, leaving the rest of the data unlabeled. This
setup, known as positive-unlabeled (PU) learning, is addressed here with a
network flow-based method which utilizes pairwise similarities between samples.
The method we propose here, 2-HNC, leverages Hochbaum's Normalized Cut (HNC)
and the set of solutions it provides by solving a parametric minimum cut
problem. The set of solutions, that are nested partitions of the samples into
two sets, correspond to varying tradeoff values between the two goals: high
intra-similarity inside the sets and low inter-similarity between the two sets.
This nested sequence is utilized here to deliver a ranking of unlabeled samples
by their likelihood of being negative. Building on this insight, our method,
2-HNC, proceeds in two stages. The first stage generates this ranking without
assuming any negative labels, using a problem formulation that is constrained
only on positive labeled samples. The second stage augments the positive set
with likely-negative samples and recomputes the classification. The final label
prediction selects among all generated partitions in both stages, the one that
delivers a positive class proportion, closest to a prior estimate of this
quantity, which is assumed to be given. Extensive experiments across synthetic
and real datasets show that 2-HNC yields strong performance and often surpasses
existing state-of-the-art algorithms.

### 3. [Deep Probabilistic Modeling of User Behavior for Anomaly Detection via Mixture Density Networks](http://arxiv.org/pdf/2505.08220v1)

Authors: Lu Dai, Wenxuan Zhu, Xuehui Quan, Renzi Meng, Sheng Cai, Yichen Wang

To improve the identification of potential anomaly patterns in complex user
behavior, this paper proposes an anomaly detection method based on a deep
mixture density network. The method constructs a Gaussian mixture model
parameterized by a neural network, enabling conditional probability modeling of
user behavior. It effectively captures the multimodal distribution
characteristics commonly present in behavioral data. Unlike traditional
classifiers that rely on fixed thresholds or a single decision boundary, this
approach defines an anomaly scoring function based on probability density using
negative log-likelihood. This significantly enhances the model's ability to
detect rare and unstructured behaviors. Experiments are conducted on the
real-world network user dataset UNSW-NB15. A series of performance comparisons
and stability validation experiments are designed. These cover multiple
evaluation aspects, including Accuracy, F1- score, AUC, and loss fluctuation.
The results show that the proposed method outperforms several advanced neural
network architectures in both performance and training stability. This study
provides a more expressive and discriminative solution for user behavior
modeling and anomaly detection. It strongly promotes the application of deep
probabilistic modeling techniques in the fields of network security and
intelligent risk control.

### 4. [Clustering-based Low-Rank Matrix Approximation: An Adaptive Theoretical Analysis with Application to Data Compression](http://arxiv.org/pdf/2505.08256v1)

Authors: Sisipho Hamlomo, Marcellin Atemkeng

Low-rank matrix approximation (LoRMA) is a fundamental tool for compressing
high-resolution data matrices by extracting important features while
suppressing redundancy. Low-rank methods, such as global singular value
decomposition (SVD), apply uniform compression across the entire data matrix,
often ignoring important local variations and leading to the loss of fine
structural details. To address these limitations, we introduce an adaptive
LoRMA, which partitions data matrix into overlapping patches, groups
structurally similar patches into several clusters using k-means, and performs
SVD within each cluster. We derive the overall compression factor accounting
for patch overlap and analyze how patch size influences compression efficiency
and computational cost. While the proposed adaptive LoRMA method is applicable
to any data exhibiting high local variation, we focus on medical imaging due to
its pronounced local variability. We evaluate and compare our adaptive LoRMA
against global SVD across four imaging modalities: MRI, ultrasound, CT scan,
and chest X-ray. Results demonstrate that adaptive LoRMA effectively preserves
structural integrity, edge details, and diagnostic relevance, as measured by
peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), mean
squared error (MSE), intersection over union (IoU), and edge preservation index
(EPI). Adaptive LoRMA significantly minimizes block artifacts and residual
errors, particularly in pathological regions, consistently outperforming global
SVD in terms of PSNR, SSIM, IoU, EPI, and achieving lower MSE. Adaptive LoRMA
prioritizes clinically salient regions while allowing aggressive compression in
non-critical regions, optimizing storage efficiency. Although adaptive LoRMA
requires higher processing time, its diagnostic fidelity justifies the overhead
for high-compression applications.

### 5. [Rapid Overfitting of Multi-Pass Stochastic Gradient Descent in Stochastic Convex Optimization](http://arxiv.org/pdf/2505.08306v1)

Authors: Shira Vansover-Hager, Tomer Koren, Roi Livni

We study the out-of-sample performance of multi-pass stochastic gradient
descent (SGD) in the fundamental stochastic convex optimization (SCO) model.
While one-pass SGD is known to achieve an optimal $\Theta(1/\sqrt{n})$ excess
population loss given a sample of size $n$, much less is understood about the
multi-pass version of the algorithm which is widely used in practice. Somewhat
surprisingly, we show that in the general non-smooth case of SCO, just a few
epochs of SGD can already hurt its out-of-sample performance significantly and
lead to overfitting. In particular, using a step size $\eta =
\Theta(1/\sqrt{n})$, which gives the optimal rate after one pass, can lead to
population loss as large as $\Omega(1)$ after just one additional pass. More
generally, we show that the population loss from the second pass onward is of
the order $\Theta(1/(\eta T) + \eta \sqrt{T})$, where $T$ is the total number
of steps. These results reveal a certain phase-transition in the out-of-sample
behavior of SGD after the first epoch, as well as a sharp separation between
the rates of overfitting in the smooth and non-smooth cases of SCO.
Additionally, we extend our results to with-replacement SGD, proving that the
same asymptotic bounds hold after $O(n \log n)$ steps. Finally, we also prove a
lower bound of $\Omega(\eta \sqrt{n})$ on the generalization gap of one-pass
SGD in dimension $d = \smash{\widetilde O}(n)$, improving on recent results of
Koren et al.(2022) and Schliserman et al.(2024).

### 6. [SpecSphere: Dual-Pass Spectral-Spatial Graph Neural Networks with Certified Robustness](http://arxiv.org/pdf/2505.08320v2)

Authors: Yoonhyuk Choi, Chong-Kwon Kim

We introduce SpecSphere, the first dual-pass spectral-spatial GNN that
certifies every prediction against both $\ell\_{0}$ edge flips and
$\ell\_{\infty}$ feature perturbations, adapts to the full
homophily-heterophily spectrum, and surpasses the expressive power of
1-Weisfeiler-Lehman while retaining linear-time complexity. Our model couples a
Chebyshev-polynomial spectral branch with an attention-gated spatial branch and
fuses their representations through a lightweight MLP trained in a
cooperative-adversarial min-max game. We further establish (i) a uniform
Chebyshev approximation theorem, (ii) minimax-optimal risk across the
homophily-heterophily spectrum, (iii) closed-form robustness certificates, and
(iv) universal approximation strictly beyond 1-WL. SpecSphere achieves
state-of-the-art node-classification accuracy and delivers tighter certified
robustness guarantees on real-world benchmarks. These results demonstrate that
high expressivity, heterophily adaptation, and provable robustness can coexist
within a single, scalable architecture.

### 7. [Localization of Impacts on Thin-Walled Structures by Recurrent Neural Networks: End-to-end Learning from Real-World Data](http://arxiv.org/pdf/2505.08362v1)

Authors: Alexander Humer, Lukas Grasboeck, Ayech Benjeddou

Today, machine learning is ubiquitous, and structural health monitoring (SHM)
is no exception. Specifically, we address the problem of impact localization on
shell-like structures, where knowledge of impact locations aids in assessing
structural integrity. Impacts on thin-walled structures excite Lamb waves,
which can be measured with piezoelectric sensors. Their dispersive
characteristics make it difficult to detect and localize impacts by
conventional methods. In the present contribution, we explore the localization
of impacts using neural networks. In particular, we propose to use {recurrent
neural networks} (RNNs) to estimate impact positions end-to-end, i.e., directly
from {sequential sensor data}. We deal with comparatively long sequences of
thousands of samples, since high sampling rate are needed to accurately capture
elastic waves. For this reason, the proposed approach builds upon Gated
Recurrent Units (GRUs), which are less prone to vanishing gradients as compared
to conventional RNNs. Quality and quantity of data are crucial when training
neural networks. Often, synthetic data is used, which inevitably introduces a
reality gap. Here, by contrast, we train our networks using {physical data from
experiments}, which requires automation to handle the large number of
experiments needed. For this purpose, a {robot is used to drop steel balls}
onto an {aluminum plate} equipped with {piezoceramic sensors}. Our results show
remarkable accuracy in estimating impact positions, even with a comparatively
small dataset.

### 8. [InfoPO: On Mutual Information Maximization for Large Language Model Alignment](http://arxiv.org/pdf/2505.08507v1)

Authors: Teng Xiao, Zhen Ge, Sujay Sanghavi, Tian Wang, Julian Katz-Samuels, Marc Versage, Qingjun Cui, Trishul Chilimbi

We study the post-training of large language models (LLMs) with human
preference data. Recently, direct preference optimization and its variants have
shown considerable promise in aligning language models, eliminating the need
for reward models and online sampling. Despite these benefits, these methods
rely on explicit assumptions about the Bradley-Terry (BT) model, which makes
them prone to overfitting and results in suboptimal performance, particularly
on reasoning-heavy tasks. To address these challenges, we propose a principled
preference fine-tuning algorithm called InfoPO, which effectively and
efficiently aligns large language models using preference data. InfoPO
eliminates the reliance on the BT model and prevents the likelihood of the
chosen response from decreasing. Extensive experiments confirm that InfoPO
consistently outperforms established baselines on widely used open benchmarks,
particularly in reasoning tasks.

### 9. [Online Learning and Unlearning](http://arxiv.org/pdf/2505.08557v1)

Authors: Yaxi Hu, Bernhard Schölkopf, Amartya Sanyal

We formalize the problem of online learning-unlearning, where a model is
updated sequentially in an online setting while accommodating unlearning
requests between updates. After a data point is unlearned, all subsequent
outputs must be statistically indistinguishable from those of a model trained
without that point. We present two online learner-unlearner (OLU) algorithms,
both built upon online gradient descent (OGD). The first, passive OLU,
leverages OGD's contractive property and injects noise when unlearning occurs,
incurring no additional computation. The second, active OLU, uses an offline
unlearning algorithm that shifts the model toward a solution excluding the
deleted data. Under standard convexity and smoothness assumptions, both methods
achieve regret bounds comparable to those of standard OGD, demonstrating that
one can maintain competitive regret bounds while providing unlearning
guarantees.

### 10. [Clustering of Incomplete Data via a Bipartite Graph Structure](http://arxiv.org/pdf/2505.08594v1)

Authors: Amirhossein Javaheri, Daniel P. Palomar

There are various approaches to graph learning for data clustering,
incorporating different spectral and structural constraints through diverse
graph structures. Some methods rely on bipartite graph models, where nodes are
divided into two classes: centers and members. These models typically require
access to data for the center nodes in addition to observations from the member
nodes. However, such additional data may not always be available in many
practical scenarios. Moreover, popular Gaussian models for graph learning have
demonstrated limited effectiveness in modeling data with heavy-tailed
distributions, which are common in financial markets. In this paper, we propose
a clustering method based on a bipartite graph model that addresses these
challenges. First, it can infer clusters from incomplete data without requiring
information about the center nodes. Second, it is designed to effectively
handle heavy-tailed data. Numerical experiments using real financial data
validate the efficiency of the proposed method for data clustering.

### Neural and Evolutionary Computing

### 1. [Convolutional Spiking Neural Network for Image Classification](http://arxiv.org/pdf/2505.08514v1)

Authors: Mikhail Kiselev, Andrey Lavrentyev

We consider an implementation of convolutional architecture in a spiking
neural network (SNN) used to classify images. As in the traditional neural
network, the convolutional layers form informational "features" used as
predictors in the SNN-based classifier with CoLaNET architecture. Since weight
sharing contradicts the synaptic plasticity locality principle, the
convolutional weights are fixed in our approach. We describe a methodology for
their determination from a representative set of images from the same domain as
the classified ones. We illustrate and test our approach on a classification
task from the NEOVISION2 benchmark.

### 2. [ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus](http://arxiv.org/pdf/2505.08778v1)

Authors: Etienne Guichard, Felix Reimers, Mia Kvalsund, Mikkel Lepperød, Stefano Nichele

The Abstraction and Reasoning Corpus (ARC), later renamed ARC-AGI, poses a
fundamental challenge in artificial general intelligence (AGI), requiring
solutions that exhibit robust abstraction and reasoning capabilities across
diverse tasks, while only few (with median count of three) correct examples are
presented. While ARC-AGI remains very challenging for artificial intelligence
systems, it is rather easy for humans. This paper introduces ARC-NCA, a
developmental approach leveraging standard Neural Cellular Automata (NCA) and
NCA enhanced with hidden memories (EngramNCA) to tackle the ARC-AGI benchmark.
NCAs are employed for their inherent ability to simulate complex dynamics and
emergent patterns, mimicking developmental processes observed in biological
systems. Developmental solutions may offer a promising avenue for enhancing
AI's problem-solving capabilities beyond mere training data extrapolation.
ARC-NCA demonstrates how integrating developmental principles into
computational models can foster adaptive reasoning and abstraction. We show
that our ARC-NCA proof-of-concept results may be comparable to, and sometimes
surpass, that of ChatGPT 4.5, at a fraction of the cost.

### Networking and Internet Architecture

### 1. [Hybrid Wi-Fi/PDR Indoor Localization with Fingerprint Matching](http://arxiv.org/pdf/2505.08258v1)

Authors: Chunyi Zhang, Zongwei Li, Xiaoqi Li

Indoor position technology has become one of the research highlights in the
Internet of Things (IoT), but there is still a lack of universal, low-cost, and
high-precision solutions. This paper conducts research on indoor position
technology based on location fingerprints and proposes a practical hybrid
indoor positioning system. In this experiment, the location fingerprint
database is established by using RSS signal in the offline stage, the location
algorithm is improved and innovated in the online stage. The weighted k-nearest
neighbor algorithm is used for location fingerprint matching and pedestrian
dead reckoning technology is used for trajectory tracking. This paper designs
and implements an indoor position system that performs the functions of data
collection, positioning, and position tracking. Through the test, it is found
that it can meet the requirements of indoor positioning.

### 2. [AI-Driven Digital Twins: Optimizing 5G/6G Network Slicing with NTNs](http://arxiv.org/pdf/2505.08328v1)

Authors: Afan Ali, Huseyin Arslan

Network slicing in 5G/6G Non-Terrestrial Network (NTN) is confronted with
mobility and traffic variability. An artificial intelligence (AI)-based digital
twin (DT) architecture with deep reinforcement learning (DRL) using Deep
deterministic policy gradient (DDPG) is proposed for dynamic optimization of
resource allocation. DT virtualizes network states to enable predictive
analysis, while DRL changes bandwidth for eMBB slice. Simulations show a 25\%
latency reduction compared to static methods, with enhanced resource
utilization. This scalable solution supports 5G/6G NTN applications like
disaster recovery and urban blockage.

### 3. [Multi-Layer Hierarchical Federated Learning with Quantization](http://arxiv.org/pdf/2505.08145v1)

Authors: Seyed Mohammad Azimi-Abarghouyi, Carlo Fischione

Almost all existing hierarchical federated learning (FL) models are limited
to two aggregation layers, restricting scalability and flexibility in complex,
large-scale networks. In this work, we propose a Multi-Layer Hierarchical
Federated Learning framework (QMLHFL), which appears to be the first study that
generalizes hierarchical FL to arbitrary numbers of layers and network
architectures through nested aggregation, while employing a layer-specific
quantization scheme to meet communication constraints. We develop a
comprehensive convergence analysis for QMLHFL and derive a general convergence
condition and rate that reveal the effects of key factors, including
quantization parameters, hierarchical architecture, and intra-layer iteration
counts. Furthermore, we determine the optimal number of intra-layer iterations
to maximize the convergence rate while meeting a deadline constraint that
accounts for both communication and computation times. Our results show that
QMLHFL consistently achieves high learning accuracy, even under high data
heterogeneity, and delivers notably improved performance when optimized,
compared to using randomly selected values.

### Robotics

### 1. [CLTP: Contrastive Language-Tactile Pre-training for 3D Contact Geometry Understanding](http://arxiv.org/pdf/2505.08194v1)

Authors: Wenxuan Ma, Xiaoge Cao, Yixiang Zhang, Chaofan Zhang, Shaobo Yang, Peng Hao, Bin Fang, Yinghao Cai, Shaowei Cui, Shuo Wang

Recent advancements in integrating tactile sensing with vision-language
models (VLMs) have demonstrated remarkable potential for robotic multimodal
perception. However, existing tactile descriptions remain limited to
superficial attributes like texture, neglecting critical contact states
essential for robotic manipulation. To bridge this gap, we propose CLTP, an
intuitive and effective language tactile pretraining framework that aligns
tactile 3D point clouds with natural language in various contact scenarios,
thus enabling contact-state-aware tactile language understanding for
contact-rich manipulation tasks. We first collect a novel dataset of 50k+
tactile 3D point cloud-language pairs, where descriptions explicitly capture
multidimensional contact states (e.g., contact location, shape, and force) from
the tactile sensor's perspective. CLTP leverages a pre-aligned and frozen
vision-language feature space to bridge holistic textual and tactile
modalities. Experiments validate its superiority in three downstream tasks:
zero-shot 3D classification, contact state classification, and tactile 3D large
language model (LLM) interaction. To the best of our knowledge, this is the
first study to align tactile and language representations from the contact
state perspective for manipulation tasks, providing great potential for
tactile-language-action model learning. Code and datasets are open-sourced at
https://sites.google.com/view/cltp/.

### 2. [HandCept: A Visual-Inertial Fusion Framework for Accurate Proprioception in Dexterous Hands](http://arxiv.org/pdf/2505.08213v1)

Authors: Junda Huang, Jianshu Zhou, Honghao Guo, Yunhui Liu

As robotics progresses toward general manipulation, dexterous hands are
becoming increasingly critical. However, proprioception in dexterous hands
remains a bottleneck due to limitations in volume and generality. In this work,
we present HandCept, a novel visual-inertial proprioception framework designed
to overcome the challenges of traditional joint angle estimation methods.
HandCept addresses the difficulty of achieving accurate and robust joint angle
estimation in dynamic environments where both visual and inertial measurements
are prone to noise and drift. It leverages a zero-shot learning approach using
a wrist-mounted RGB-D camera and 9-axis IMUs, fused in real time via a
latency-free Extended Kalman Filter (EKF). Our results show that HandCept
achieves joint angle estimation errors between $2^{\circ}$ and $4^{\circ}$
without observable drift, outperforming visual-only and inertial-only methods.
Furthermore, we validate the stability and uniformity of the IMU system,
demonstrating that a common base frame across IMUs simplifies system
calibration. To support sim-to-real transfer, we also open-sourced our
high-fidelity rendering pipeline, which is essential for training without
real-world ground truth. This work offers a robust, generalizable solution for
proprioception in dexterous hands, with significant implications for robotic
manipulation and human-robot interaction.

### 3. [SKiD-SLAM: Robust, Lightweight, and Distributed Multi-Robot LiDAR SLAM in Resource-Constrained Field Environments](http://arxiv.org/pdf/2505.08230v1)

Authors: Hogyun Kim, Jiwon Choi, Juwon Kim, Geonmo Yang, Dongjin Cho, Hyungtae Lim, Younggun Cho

Distributed LiDAR SLAM is crucial for achieving efficient robot autonomy and
improving the scalability of mapping. However, two issues need to be considered
when applying it in field environments: one is resource limitation, and the
other is inter/intra-robot association. The resource limitation issue arises
when the data size exceeds the processing capacity of the network or memory,
especially when utilizing communication systems or onboard computers in the
field. The inter/intra-robot association issue occurs due to the narrow
convergence region of ICP under large viewpoint differences, triggering many
false positive loops and ultimately resulting in an inconsistent global map for
multi-robot systems. To tackle these problems, we propose a distributed LiDAR
SLAM framework designed for versatile field applications, called SKiD-SLAM.
Extending our previous work that solely focused on lightweight place
recognition and fast and robust global registration, we present a multi-robot
mapping framework that focuses on robust and lightweight inter-robot loop
closure in distributed LiDAR SLAM. Through various environmental experiments,
we demonstrate that our method is more robust and lightweight compared to other
state-of-the-art distributed SLAM approaches, overcoming resource limitation
and inter/intra-robot association issues. Also, we validated the field
applicability of our approach through mapping experiments in real-world
planetary emulation terrain and cave environments, which are in-house datasets.
Our code will be available at https://sparolab.github.io/research/skid_slam/.

### 4. [Motion Control of High-Dimensional Musculoskeletal Systems with Hierarchical Model-Based Planning](http://arxiv.org/pdf/2505.08238v1)

Authors: Yunyue Wei, Shanning Zhuang, Vincent Zhuang, Yanan Sui

Controlling high-dimensional nonlinear systems, such as those found in
biological and robotic applications, is challenging due to large state and
action spaces. While deep reinforcement learning has achieved a number of
successes in these domains, it is computationally intensive and time consuming,
and therefore not suitable for solving large collections of tasks that require
significant manual tuning. In this work, we introduce Model Predictive Control
with Morphology-aware Proportional Control (MPC^2), a hierarchical model-based
learning algorithm for zero-shot and near-real-time control of high-dimensional
complex dynamical systems. MPC^2 uses a sampling-based model predictive
controller for target posture planning, and enables robust control for
high-dimensional tasks by incorporating a morphology-aware proportional
controller for actuator coordination. The algorithm enables motion control of a
high-dimensional human musculoskeletal model in a variety of motion tasks, such
as standing, walking on different terrains, and imitating sports activities.
The reward function of MPC^2 can be tuned via black-box optimization,
drastically reducing the need for human-intensive reward engineering.

### 5. [Training Strategies for Efficient Embodied Reasoning](http://arxiv.org/pdf/2505.08243v1)

Authors: William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, Sergey Levine

Robot chain-of-thought reasoning (CoT) -- wherein a model predicts helpful
intermediate representations before choosing actions -- provides an effective
method for improving the generalization and performance of robot policies,
especially vision-language-action models (VLAs). While such approaches have
been shown to improve performance and generalization, they suffer from core
limitations, like needing specialized robot reasoning data and slow inference
speeds. To design new robot reasoning approaches that address these issues, a
more complete characterization of why reasoning helps policy performance is
critical. We hypothesize several mechanisms by which robot reasoning improves
policies -- (1) better representation learning, (2) improved learning
curricularization, and (3) increased expressivity -- then devise simple
variants of robot CoT reasoning to isolate and test each one. We find that
learning to generate reasonings does lead to better VLA representations, while
attending to the reasonings aids in actually leveraging these features for
improved action prediction. Our results provide us with a better understanding
of why CoT reasoning helps VLAs, which we use to introduce two simple and
lightweight alternative recipes for robot reasoning. Our proposed approaches
achieve significant performance gains over non-reasoning policies,
state-of-the-art results on the LIBERO-90 benchmark, and a 3x inference speedup
compared to standard robot reasoning.

### 6. [MA-ROESL: Motion-aware Rapid Reward Optimization for Efficient Robot Skill Learning from Single Videos](http://arxiv.org/pdf/2505.08367v1)

Authors: Xianghui Wang, Xinming Zhang, Yanjun Chen, Xiaoyu Shen, Wei Zhang

Vision-language models (VLMs) have demonstrated excellent high-level planning
capabilities, enabling locomotion skill learning from video demonstrations
without the need for meticulous human-level reward design. However, the
improper frame sampling method and low training efficiency of current methods
remain a critical bottleneck, resulting in substantial computational overhead
and time costs. To address this limitation, we propose Motion-aware Rapid
Reward Optimization for Efficient Robot Skill Learning from Single Videos
(MA-ROESL). MA-ROESL integrates a motion-aware frame selection method to
implicitly enhance the quality of VLM-generated reward functions. It further
employs a hybrid three-phase training pipeline that improves training
efficiency via rapid reward optimization and derives the final policy through
online fine-tuning. Experimental results demonstrate that MA-ROESL
significantly enhances training efficiency while faithfully reproducing
locomotion skills in both simulated and real-world settings, thereby
underscoring its potential as a robust and scalable framework for efficient
robot locomotion skill learning from video demonstrations.

### 7. [MDF: Multi-Modal Data Fusion with CNN-Based Object Detection for Enhanced Indoor Localization Using LiDAR-SLAM](http://arxiv.org/pdf/2505.08388v1)

Authors: Saqi Hussain Kalan, Boon Giin Lee, Wan-Young Chung

Indoor localization faces persistent challenges in achieving high accuracy,
particularly in GPS-deprived environments. This study unveils a cutting-edge
handheld indoor localization system that integrates 2D LiDAR and IMU sensors,
delivering enhanced high-velocity precision mapping, computational efficiency,
and real-time adaptability. Unlike 3D LiDAR systems, it excels with rapid
processing, low-cost scalability, and robust performance, setting new standards
for emergency response, autonomous navigation, and industrial automation.
Enhanced with a CNN-driven object detection framework and optimized through
Cartographer SLAM (simultaneous localization and mapping ) in ROS, the system
significantly reduces Absolute Trajectory Error (ATE) by 21.03%, achieving
exceptional precision compared to state-of-the-art approaches like SC-ALOAM,
with a mean x-position error of -0.884 meters (1.976 meters). The integration
of CNN-based object detection ensures robustness in mapping and localization,
even in cluttered or dynamic environments, outperforming existing methods by
26.09%. These advancements establish the system as a reliable, scalable
solution for high-precision localization in challenging indoor scenarios

### 8. [ORACLE-Grasp: Zero-Shot Task-Oriented Robotic Grasping using Large Multimodal Models](http://arxiv.org/pdf/2505.08417v1)

Authors: Avihai Giuili, Rotem Atari, Avishai Sintov

Grasping unknown objects in unstructured environments remains a fundamental
challenge in robotics, requiring both semantic understanding and spatial
reasoning. Existing methods often rely on dense training datasets or explicit
geometric modeling, limiting their scalability to real-world tasks. Recent
advances in Large Multimodal Models (LMMs) offer new possibilities for
integrating vision and language understanding, but their application to
autonomous robotic grasping remains largely unexplored. We present
ORACLE-Grasp, a zero-shot framework that leverages LMMs as semantic oracles to
guide grasp selection without requiring additional training or human input. The
system formulates grasp prediction as a structured, iterative decision process,
using dual-prompt tool calling to first extract high-level object context and
then select task-relevant grasp regions. By discretizing the image space and
reasoning over candidate areas, ORACLE-Grasp mitigates the spatial imprecision
common in LMMs and produces human-like, task-driven grasp suggestions. Early
stopping and depth-based refinement steps further enhance efficiency and
physical grasp reliability. Experiments demonstrate that the predicted grasps
achieve low positional and orientation errors relative to human-annotated
ground truth and lead to high success rates in real-world pick up tasks. These
results highlight the potential of combining language-driven reasoning with
lightweight vision techniques to enable robust, autonomous grasping without
task-specific datasets or retraining.

### 9. [HMR-ODTA: Online Diverse Task Allocation for a Team of Heterogeneous Mobile Robots](http://arxiv.org/pdf/2505.08419v1)

Authors: Ashish Verma, Avinash Gautam, Tanishq Duhan, V. S. Shekhawat, Sudeept Mohan

Coordinating time-sensitive deliveries in environments like hospitals poses a
complex challenge, particularly when managing multiple online pickup and
delivery requests within strict time windows using a team of heterogeneous
robots. Traditional approaches fail to address dynamic rescheduling or diverse
service requirements, typically restricting robots to single-task types. This
paper tackles the Multi-Pickup and Delivery Problem with Time Windows (MPDPTW),
where autonomous mobile robots are capable of handling varied service requests.
The objective is to minimize late delivery penalties while maximizing task
completion rates. To achieve this, we propose a novel framework leveraging a
heterogeneous robot team and an efficient dynamic scheduling algorithm that
supports dynamic task rescheduling. Users submit requests with specific time
constraints, and our decentralized algorithm, Heterogeneous Mobile Robots
Online Diverse Task Allocation (HMR-ODTA), optimizes task assignments to ensure
timely service while addressing delays or task rejections. Extensive
simulations validate the algorithm's effectiveness. For smaller task sets
(40-160 tasks), penalties were reduced by nearly 63%, while for larger sets
(160-280 tasks), penalties decreased by approximately 50%. These results
highlight the algorithm's effectiveness in improving task scheduling and
coordination in multi-robot systems, offering a robust solution for enhancing
delivery performance in structured, time-critical environments.

### 10. [Symbolically-Guided Visual Plan Inference from Uncurated Video Data](http://arxiv.org/pdf/2505.08444v1)

Authors: Wenyan Yang, Ahmet Tikna, Yi Zhao, Yuying Zhang, Luigi Palopoli, Marco Roveri, Joni Pajarinen

Visual planning, by offering a sequence of intermediate visual subgoals to a
goal-conditioned low-level policy, achieves promising performance on
long-horizon manipulation tasks. To obtain the subgoals, existing methods
typically resort to video generation models but suffer from model hallucination
and computational cost. We present Vis2Plan, an efficient, explainable and
white-box visual planning framework powered by symbolic guidance. From raw,
unlabeled play data, Vis2Plan harnesses vision foundation models to
automatically extract a compact set of task symbols, which allows building a
high-level symbolic transition graph for multi-goal, multi-stage planning. At
test time, given a desired task goal, our planner conducts planning at the
symbolic level and assembles a sequence of physically consistent intermediate
sub-goal images grounded by the underlying symbolic representation. Our
Vis2Plan outperforms strong diffusion video generation-based visual planners by
delivering 53\% higher aggregate success rate in real robot settings while
generating visual plans 35$\times$ faster. The results indicate that Vis2Plan
is able to generate physically consistent image goals while offering fully
inspectable reasoning steps.

### Software Engineering

### 1. [LLM-Based Detection of Tangled Code Changes for Higher-Quality Method-Level Bug Datasets](http://arxiv.org/pdf/2505.08263v1)

Authors: Md Nahidul Islam Opu, Shaowei Wang, Shaiful Chowdhury

Tangled code changes-commits that conflate unrelated modifications such as
bug fixes, refactorings, and enhancements-introduce significant noise into bug
datasets and adversely affect the performance of bug prediction models.
Addressing this issue at a fine-grained, method-level granularity remains
underexplored. This is critical to address, as recent bug prediction models,
driven by practitioner demand, are increasingly focusing on finer granularity
rather than traditional class- or file-level predictions. This study
investigates the utility of Large Language Models (LLMs) for detecting tangled
code changes by leveraging both commit messages and method-level code diffs. We
formulate the problem as a binary classification task and evaluate multiple
prompting strategies, including zero-shot, few-shot, and chain-of-thought
prompting, using state-of-the-art proprietary LLMs such as GPT-4o and
Gemini-2.0-Flash.
  Our results demonstrate that combining commit messages with code diffs
significantly enhances model performance, with the combined few-shot and
chain-of-thought prompting achieving an F1-score of 0.88. Additionally, we
explore embedding-based machine learning models trained on LLM-generated
embeddings, where a multi-layer perceptron classifier achieves superior
performance (F1-score: 0.906, MCC: 0.807). These findings are encouraging for
the research community, as method-level bug prediction remains an open research
problem, largely due to the lack of noise-free bug datasets. This research not
only contributes a novel method-level perspective to the untangling problem but
also highlights practical avenues for enhancing automated software quality
assessment tools.

### 2. [Exploring Challenges in Test Mocking: Developer Questions and Insights from StackOverflow](http://arxiv.org/pdf/2505.08300v1)

Authors: Mumtahina Ahmed, Md Nahidul Islam Opu, Chanchal Roy, Shaiful Chowdhury, Sujana Islam Suhi

Mocking is a common unit testing technique that is used to simplify tests,
reduce flakiness, and improve coverage by replacing real dependencies with
simplified implementations. Despite its widespread use in Open Source Software
projects, there is limited understanding of how and why developers use mocks
and the challenges they face. In this collaborative study, we have analyzed
25,302 questions related to Mocking on STACKOVERFLOW to identify the challenges
faced by developers. We have used Latent Dirichlet Allocation for topic
modeling, identified 30 key topics, and grouped the topics into five key
categories. Consequently, we analyzed the annual and relative probabilities of
each category to understand the evolution of mocking-related discussions. Trend
analysis reveals that category like Advanced Programming peaked between 2009
and 2012 but have since declined, while categories such as Mocking Techniques
and External Services have remained consistently dominant, highlighting
evolving developer priorities and ongoing technical challenges. Our findings
also show an inverse relationship between a topic's popularity and its
difficulty. Popular topics like Framework Selection tend to have lower
difficulty and faster resolution times, while complex topics like HTTP Requests
and Responses are more likely to remain unanswered and take longer to resolve.
A classification of questions into How, Why, What, and Other revealed that over
70% are How questions, particularly in practical domains like file access and
APIs, indicating a strong need for implementation guidance. Why questions are
more prevalent in error-handling contexts, reflecting conceptual challenges in
debugging, while What questions are rare and mostly tied to theoretical
discussions. These insights offer valuable guidance for improving developer
support, tooling, and educational content in the context of mocking and unit
testing.

### 3. [ICVul: A Well-labeled C/C++ Vulnerability Dataset with Comprehensive Metadata and VCCs](http://arxiv.org/pdf/2505.08503v1)

Authors: Chaomeng Lu, Tianyu Li, Toon Dehaene, Bert Lagaisse

Machine learning-based software vulnerability detection requires high-quality
datasets, which is essential for training effective models. To address
challenges related to data label quality, diversity, and comprehensiveness, we
constructed ICVul, a dataset emphasizing data quality and enriched with
comprehensive metadata, including Vulnerability-Contributing Commits (VCCs). We
began by filtering Common Vulnerabilities and Exposures from the NVD, retaining
only those linked to GitHub fix commits. Then we extracted functions and files
along with relevant metadata from these commits and used the SZZ algorithm to
trace VCCs. To further enhance label reliability, we developed the ESC
(Eliminate Suspicious Commit) technique, ensuring credible data labels. The
dataset is stored in a relational-like database for improved usability and data
integrity. Both ICVul and its construction framework are publicly accessible on
GitHub, supporting research in related field.

### 4. [Grouptuner: Efficient Group-Aware Compiler Auto-tuning](http://arxiv.org/pdf/2505.08598v1)

Authors: Bingyu Gao, Mengyu Yao, Ziming Wang, Dong Liu, Ding Li, Xiangqun Chen, Yao Guo

Modern compilers typically provide hundreds of options to optimize program
performance, but users often cannot fully leverage them due to the huge number
of options. While standard optimization combinations (e.g., -O3) provide
reasonable defaults, they often fail to deliver near-peak performance across
diverse programs and architectures. To address this challenge, compiler
auto-tuning techniques have emerged to automate the discovery of improved
option combinations. Existing techniques typically focus on identifying
critical options and prioritizing them during the search to improve efficiency.
However, due to limited tuning iterations, the resulting data is often sparse
and noisy, making it highly challenging to accurately identify critical
options. As a result, these algorithms are prone to being trapped in local
optima.
  To address this limitation, we propose GroupTuner, a group-aware auto-tuning
technique that directly applies localized mutation to coherent option groups
based on historically best-performing combinations, thus avoiding explicitly
identifying critical options. By forgoing the need to know precisely which
options are most important, GroupTuner maximizes the use of existing
performance data, ensuring more targeted exploration. Extensive experiments
demonstrate that GroupTuner can efficiently discover competitive option
combinations, achieving an average performance improvement of 12.39% over -O3
while requiring only 77.21% of the time compared to the random search
algorithm, significantly outperforming state-of-the-art methods.

### 5. [The Failure of Plagiarism Detection in Competitive Programming](http://arxiv.org/pdf/2505.08244v1)

Authors: Ethan Dickey

Plagiarism in programming courses remains a persistent challenge, especially
in competitive programming contexts where assignments often have unique, known
solutions. This paper examines why traditional code plagiarism detection
methods frequently fail in these environments and explores the implications of
emerging factors such as generative AI (genAI). Drawing on the author's
experience teaching a Competitive Programming 1 (CP1) course over seven
semesters at Purdue University (with $\approx 100$ students each term) and
completely redesigning the CP1/2/3 course sequence, we provide an academically
grounded analysis. We review literature on code plagiarism in computer science
education, survey current detection tools (Moss, Kattis, etc.) and methods
(manual review, code-authorship interviews), and analyze their strengths and
limitations. Experience-based observations are presented to illustrate
real-world detection failures and successes. We find that widely-used automated
similarity checkers can be thwarted by simple code transformations or novel
AI-generated code, while human-centric approaches like oral interviews, though
effective, are labor-intensive. The paper concludes with opinions and
preliminary recommendations for improving academic integrity in programming
courses, advocating for a multi-faceted approach that combines improved
detection algorithms, mastery-based learning techniques, and authentic
assessment practices to better ensure code originality.

### 6. [CoVoL: A Cooperative Vocabulary Learning Game for Children with Autism](http://arxiv.org/pdf/2505.08515v1)

Authors: Pawel Chodkiewicz, Pragya Verma, Grischa Liebel

Children with Autism commonly face difficulties in vocabulary acquisition,
which can have an impact on their social communication. Using digital tools for
vocabulary learning can prove beneficial for these children, as they can
provide a predictable environment and effective individualized feedback. While
existing work has explored the use of technology-assisted vocabulary learning
for children with Autism, no study has incorporated turn-taking to facilitate
learning and use of vocabulary similar to that used in real-world social
contexts. To address this gap, we propose the design of a cooperative
two-player vocabulary learning game, CoVoL. CoVoL allows children to engage in
game-based vocabulary learning useful for real-world social communication
scenarios. We discuss our first prototype and its evaluation. Additionally, we
present planned features which are based on feedback obtained through ten
interviews with researchers and therapists, as well as an evaluation plan for
the final release of CoVoL.

### 7. [ROSA: Finding Backdoors with Fuzzing](http://arxiv.org/pdf/2505.08544v1)

Authors: Dimitri Kokkonis, Michaël Marcozzi, Emilien Decoux, Stefano Zacchiroli

A code-level backdoor is a hidden access, programmed and concealed within the
code of a program. For instance, hard-coded credentials planted in the code of
a file server application would enable maliciously logging into all deployed
instances of this application. Confirmed software supply chain attacks have led
to the injection of backdoors into popular open-source projects, and backdoors
have been discovered in various router firmware. Manual code auditing for
backdoors is challenging and existing semi-automated approaches can handle only
a limited scope of programs and backdoors, while requiring manual
reverse-engineering of the audited (binary) program. Graybox fuzzing (automated
semi-randomized testing) has grown in popularity due to its success in
discovering vulnerabilities and hence stands as a strong candidate for improved
backdoor detection. However, current fuzzing knowledge does not offer any means
to detect the triggering of a backdoor at runtime. In this work we introduce
ROSA, a novel approach (and tool) which combines a state-of-the-art fuzzer
(AFL++) with a new metamorphic test oracle, capable of detecting runtime
backdoor triggers. To facilitate the evaluation of ROSA, we have created
ROSARUM, the first openly available benchmark for assessing the detection of
various backdoors in diverse programs. Experimental evaluation shows that ROSA
has a level of robustness, speed and automation similar to classical fuzzing.
It finds all 17 authentic or synthetic backdooors from ROSARUM in 1h30 on
average. Compared to existing detection tools, it can handle a diversity of
backdoors and programs and it does not rely on manual reverse-engineering of
the fuzzed binary code.

### 8. [Enhancing Software Development with Context-Aware Conversational Agents: A User Study on Developer Interactions with Chatbots](http://arxiv.org/pdf/2505.08648v1)

Authors: Glaucia Melo, Paulo Alencar, Donald Cowan

Software development is a cognitively intensive process requiring
multitasking, adherence to evolving workflows, and continuous learning. With
the rise of large language model (LLM)-based tools, such as conversational
agents (CAs), there is growing interest in supporting developers through
natural language interaction. However, little is known about the specific
features developers seek in these systems. We conducted a user study with 29
developers using a prototype text-based chatbot to investigate preferred
functionalities. Our findings reveal strong interest in task automation,
version control support, and contextual adaptability, especially the need to
tailor assistance for both novice and experienced users. We highlight the
importance of deep contextual understanding, historical interaction awareness,
and personalized support in CA design. This study contributes to the
development of context-aware chatbots that enhance productivity and
satisfaction, and it outlines opportunities for future research on human-AI
collaboration in software engineering.

### 9. [Leveraging AI for Productive and Trustworthy HPC Software: Challenges and Research Directions](http://arxiv.org/pdf/2505.08135v1)

Authors: Keita Teranishi, Harshitha Menon, William F. Godoy, Prasanna Balaprakash, David Bau, Tal Ben-Nun, Abhinav Bathele, Franz Franchetti, Michael Franusich, Todd Gamblin, Giorgis Georgakoudis, Tom Goldstein, Arjun Guha, Steven Hahn, Costin Iancu, Zheming Jin, Terry Jones, Tze Meng Low, Het Mankad, Narasinga Rao Miniskar, Mohammad Alaul Haque Monil, Daniel Nichols, Konstantinos Parasyris, Swaroop Pophale, Pedro Valero-Lara, Jeffrey S. Vetter, Samuel Williams, Aaron Young

We discuss the challenges and propose research directions for using AI to
revolutionize the development of high-performance computing (HPC) software. AI
technologies, in particular large language models, have transformed every
aspect of software development. For its part, HPC software is recognized as a
highly specialized scientific field of its own. We discuss the challenges
associated with leveraging state-of-the-art AI technologies to develop such a
unique and niche class of software and outline our research directions in the
two US Department of Energy--funded projects for advancing HPC Software via AI:
Ellora and Durban.

### Social and Information Networks

### 1. [Revisiting Information Diffusion Beyond Explicit Social Ties: A Study of Implicit-Link Diffusion on Twitter](http://arxiv.org/pdf/2505.08354v1)

Authors: Yuto Tamura, Sho Tsugawa, Kohei Watabe

Information diffusion on social media platforms is often assumed to occur
primarily through explicit social connections, such as follower or friend
relationships. However, information frequently propagates beyond these
observable ties -- via external websites, search engines, or algorithmic
recommendations -- forming implicit links between users who are not directly
connected. Despite their potential impact, the mechanisms and characteristics
of such implicit-link diffusion remain underexplored. In this study, we
investigate the dynamics of nontrivial information diffusion mediated by
implicit links on Twitter, using four large-scale datasets. We define
implicit-link diffusion as the reposting of content by users who are not
explicitly connected to the original poster. Our analysis reveals that users
located farther from the original source in the social network are more likely
to engage in diffusion through implicit links, suggesting that such links often
arise from sources outside direct social relationships. Moreover, while
implicit links contribute less to the overall diffusion size than explicit
links, they play a distinct role in disseminating content across diverse and
topologically distant communities. We further identify user groups who
predominantly engage in diffusion through either explicit or implicit links,
and demonstrate that the choice of diffusion channel exhibits strong patterns
of social homophily. These findings underscore the importance of incorporating
implicit-link dynamics into models of information diffusion and social
influence.

### 2. [A political cartography of news sharing: Capturing story, outlet and content level of news circulation on Twitter](http://arxiv.org/pdf/2505.08359v1)

Authors: Felix Gaisbauer, Armin Pournaki, Jakob Ohme

News sharing on digital platforms shapes the digital spaces millions of users
navigate. Trace data from these platforms also enables researchers to study
online news circulation. In this context, research on the types of news shared
by users of differential political leaning has received considerable attention.
We argue that most existing approaches (i) rely on an overly simplified
measurement of political leaning, (ii) consider only the outlet level in their
analyses, and/or (iii) study news circulation among partisans by making ex-ante
distinctions between partisan and non-partisan news. In this methodological
contribution, we introduce a research pipeline that allows a systematic mapping
of news sharing both with respect to source and content. As a proof of concept,
we demonstrate insights that otherwise remain unnoticed: Diversification of
news sharing along the second political dimension; topic-dependent sharing of
outlets; some outlets catering different items to different audiences.

### 3. [Community Detection on Noisy Stochastic Block Models](http://arxiv.org/pdf/2505.08251v1)

Authors: Washieu Anan, Gwyneth Liu

We study the problem of community detection in noisy stochastic block models.
We focus on two types of noise: (1) geometric noise where a latent-space kernel
affects edge formation, and (2) Erdos-Renyi model censoring where edges are
masked independently. We present a new algorithm DuoSpec that de-noises the
network to a pristine stochastic block model structure for better community
recovery. We demonstrate on synthetic data that our algorithm outperforms
existing community detection methods on noisy models. We test our algorithm on
the Amazon metadata dataset and demonstrate strong results on community
detection.

### 4. [Structural-Temporal Coupling Anomaly Detection with Dynamic Graph Transformer](http://arxiv.org/pdf/2505.08330v1)

Authors: Chang Zong, Yueting Zhuang, Jian Shao, Weiming Lu

Detecting anomalous edges in dynamic graphs is an important task in many
applications over evolving triple-based data, such as social networks,
transaction management, and epidemiology. A major challenge with this task is
the absence of structural-temporal coupling information, which decreases the
ability of the representation to distinguish anomalies from normal instances.
Existing methods focus on handling independent structural and temporal features
with embedding models, which ignore the deep interaction between these two
types of information. In this paper, we propose a structural-temporal coupling
anomaly detection architecture with a dynamic graph transformer model.
Specifically, we introduce structural and temporal features from two
integration levels to provide anomaly-aware graph evolutionary patterns. Then,
a dynamic graph transformer enhanced by two-dimensional positional encoding is
implemented to capture both discrimination and contextual consistency signals.
Extensive experiments on six datasets demonstrate that our method outperforms
current state-of-the-art models. Finally, a case study illustrates the strength
of our method when applied to a real-world task.

### 5. [The Truth Becomes Clearer Through Debate! Multi-Agent Systems with Large Language Models Unmask Fake News](http://arxiv.org/pdf/2505.08532v1)

Authors: Yuhan Liu, Yuxuan Liu, Xiaoqing Zhang, Xiuying Chen, Rui Yan

In today's digital environment, the rapid propagation of fake news via social
networks poses significant social challenges. Most existing detection methods
either employ traditional classification models, which suffer from low
interpretability and limited generalization capabilities, or craft specific
prompts for large language models (LLMs) to produce explanations and results
directly, failing to leverage LLMs' reasoning abilities fully. Inspired by the
saying that "truth becomes clearer through debate," our study introduces a
novel multi-agent system with LLMs named TruEDebate (TED) to enhance the
interpretability and effectiveness of fake news detection. TED employs a
rigorous debate process inspired by formal debate settings. Central to our
approach are two innovative components: the DebateFlow Agents and the
InsightFlow Agents. The DebateFlow Agents organize agents into two teams, where
one supports and the other challenges the truth of the news. These agents
engage in opening statements, cross-examination, rebuttal, and closing
statements, simulating a rigorous debate process akin to human discourse
analysis, allowing for a thorough evaluation of news content. Concurrently, the
InsightFlow Agents consist of two specialized sub-agents: the Synthesis Agent
and the Analysis Agent. The Synthesis Agent summarizes the debates and provides
an overarching viewpoint, ensuring a coherent and comprehensive evaluation. The
Analysis Agent, which includes a role-aware encoder and a debate graph,
integrates role embeddings and models the interactions between debate roles and
arguments using an attention mechanism, providing the final judgment.

### 6. [Large Language Models Meet Stance Detection: A Survey of Tasks, Methods, Applications, Challenges and Future Directions](http://arxiv.org/pdf/2505.08464v1)

Authors: Lata Pangtey, Anukriti Bhatnagar, Shubhi Bansal, Shahid Shafi Dar, Nagendra Kumar

Stance detection is essential for understanding subjective content across
various platforms such as social media, news articles, and online reviews.
Recent advances in Large Language Models (LLMs) have revolutionized stance
detection by introducing novel capabilities in contextual understanding,
cross-domain generalization, and multimodal analysis. Despite these
progressions, existing surveys often lack comprehensive coverage of approaches
that specifically leverage LLMs for stance detection. To bridge this critical
gap, our review article conducts a systematic analysis of stance detection,
comprehensively examining recent advancements of LLMs transforming the field,
including foundational concepts, methodologies, datasets, applications, and
emerging challenges. We present a novel taxonomy for LLM-based stance detection
approaches, structured along three key dimensions: 1) learning methods,
including supervised, unsupervised, few-shot, and zero-shot; 2) data
modalities, such as unimodal, multimodal, and hybrid; and 3) target
relationships, encompassing in-target, cross-target, and multi-target
scenarios. Furthermore, we discuss the evaluation techniques and analyze
benchmark datasets and performance trends, highlighting the strengths and
limitations of different architectures. Key applications in misinformation
detection, political analysis, public health monitoring, and social media
moderation are discussed. Finally, we identify critical challenges such as
implicit stance expression, cultural biases, and computational constraints,
while outlining promising future directions, including explainable stance
reasoning, low-resource adaptation, and real-time deployment frameworks. Our
survey highlights emerging trends, open challenges, and future directions to
guide researchers and practitioners in developing next-generation stance
detection systems powered by large language models.

### 7. [Big Data and the Computational Social Science of Entrepreneurship and Innovation](http://arxiv.org/pdf/2505.08706v1)

Authors: Ningzi Li, Shiyang Lai, James Evans

As large-scale social data explode and machine-learning methods evolve,
scholars of entrepreneurship and innovation face new research opportunities but
also unique challenges. This chapter discusses the difficulties of leveraging
large-scale data to identify technological and commercial novelty, document new
venture origins, and forecast competition between new technologies and
commercial forms. It suggests how scholars can take advantage of new text,
network, image, audio, and video data in two distinct ways that advance
innovation and entrepreneurship research. First, machine-learning models,
combined with large-scale data, enable the construction of precision
measurements that function as system-level observatories of innovation and
entrepreneurship across human societies. Second, new artificial intelligence
models fueled by big data generate 'digital doubles' of technology and
business, forming laboratories for virtual experimentation about innovation and
entrepreneurship processes and policies. The chapter argues for the advancement
of theory development and testing in entrepreneurship and innovation by
coupling big data with big models.

### Systems and Control

### 1. [Integrating Koopman theory and Lyapunov stability for enhanced model predictive control in nonlinear systems](http://arxiv.org/pdf/2505.08139v1)

Authors: Md Nur-A-Adam Dony, Minghui Zhu

This paper delves into the challenges posed by the increasing complexity of
modern control systems, specifically focusing on bilinear systems, a prevalent
subclass of non-linear systems characterized by state dynamics influenced by
the interaction of state and control variables. Traditional control strategies,
such as PID controllers, often fall short in adequately addressing the
intricacies of such systems due to their predictive limitations. To bridge this
gap, we introduce Model Predictive Control (MPC), a sophisticated technique
that utilizes system models to forecast future behaviors, allowing for the
computation of an optimal control sequence by minimizing deviations and control
efforts. The Koopman operator emerges as a pivotal tool in this framework by
providing a means to linearize the nonlinear dynamics of bilinear systems. By
integrating the principles of Lyapunov theory with the linearizing capabilities
of the Koopman operator into the MPC framework, we give rise to Koopman
Lyapunov-based Model Predictive Control (Koopman LMPC). This approach not only
retains MPC's predictive capabilities but also harnesses the Koopman operator's
ability to transform complex nonlinear behaviors into a linear framework,
thereby enhancing the robustness and applicability of LMPC. With the stability
assurances from Lyapunov theory, Koopman LMPC presents a robust solution to
effectively control and stabilize bilinear systems. The paper underscores the
efficacy of Koopman LMPC, emphasizing its significance in achieving optimal
performance and system stability, marking it as a promising approach for the
future of advanced control systems.

### 2. [Non-Blocking Robustness Analysis in Discrete Event Systems](http://arxiv.org/pdf/2505.08166v1)

Authors: Md Nur-A-Adam Dony, Satadru Dey

This paper presents a mathematical framework for characterizing state
blocking in discrete event systems (DES) under transition deletions. We
introduce a path-based analysis approach that determines whether systems
maintain non-blocking properties when transitions are removed. Through formal
analysis and case studies, we establish three key contributions: a mathematical
characterization of transition-induced blocking with necessary and sufficient
conditions, a definition of robust deviations that preserve non-blocking
properties, and an algorithm for identifying critical transitions and analyzing
system behavior under deletions. Our algorithm reduces computational complexity
by leveraging minimal blocking sets, achieving significant reduction in
computational requirements. We demonstrate the framework's effectiveness
through manufacturing system and autonomous vehicle case studies, showing
substantial improvements in identifying critical transitions and predicting
potential blocking scenarios across different application domains.

### 3. [On the Use of CVRP to Diagnose Faulty Elements in Antenna Arrays](http://arxiv.org/pdf/2505.08433v1)

Authors: Alejandro Antón Ruiz, John Kvarnstrand, Klas Arvidsson, Andrés Alayón Glazunov

This paper investigates the application of Constrained-View Radiated Power
(CVRP) for diagnosing phased array element failures, specifically focusing on
on-off element failure. CVRP, similar to Partial Radiated Power (PRP),
considers a specific Field-of-View (FoV) but normalizes it by the FoV area. The
study explores CVRP's effectiveness in detecting failures in a 2x8 cosine
element array under beam-steering conditions, accounting for random and
depointing errors, angular resolution, and pattern rotation. Results indicate
that CVRP can detect on-off failures based on angular resolution and error
severity, under the assumption of reduced Total Radiated Power (TRP) with
element failures. Additionally, CVRP is effective with partial far-field
patterns, making it suitable for near-field, indirect far-field, and far-field
measurement systems without requiring phase acquisition in the latter two.

### 4. [A Practical Approach to Generating First-Order Rician Channel Statistics in a RC plus CATR Chamber at mmWave](http://arxiv.org/pdf/2505.08447v1)

Authors: Alejandro Antón Ruiz, Samar Hosseinzadegan, John Kvarnstrand, Klas Arvidsson, Andrés Alayón Glazunov

This paper explores a novel hybrid configuration integrating a Reverberation
Chamber (RC) with a Compact Antenna Test Range (CATR) to achieve a controllable
Rician K-factor. The focus is testing directive antennas in the lower FR2
frequency bands (24.25-29.5 GHz) for 5G and beyond wireless applications. The
study meticulously evaluates 39 unique configurations, using a stationary horn
antenna for consistent reference K-factor characterization, and considers
variables like absorbers and CATR polarization. Results demonstrate that the
K-factor can be effectively adjusted within the hybrid setup, maintaining
substantial margins above the noise level across all configurations. Sample
independence is confirmed for at least 600 samples in all cases. The Bootstrap
Anderson-Darling goodness-of-fit test verifies that the data align with Rician
or Rayleigh distributions. Analysis of total received power, stirred and
unstirred power and frequency-dependent modeling reveals that power variables
are inversely related to frequency, while the K-factor remains
frequency-independent. The hybrid RC-CATR system achieves a wide range of
frequency-averaged K-factors from -9.2 dB to 40.8 dB, with an average
granularity of 1.3 dB. Notably, configurations using co-polarized CATR signals
yield large K-factors, reduced system losses, and improved frequency stability,
underscoring the system's efficacy for millimeter-wave over-the-air testing.
This research offers a cost-efficient and repeatable method for generating
complex Rician fading channels at mmWave frequencies, crucial for the effective
OTA testing of advanced wireless devices.

### 5. [The Quadrature Gaussian Sum Filter and Smoother for Wiener Systems](http://arxiv.org/pdf/2505.08469v1)

Authors: Angel L. Cedeño, Rodrigo A. González, Juan C. Agüero

Block-Oriented Nonlinear (BONL) models, particularly Wiener models, are
widely used for their computational efficiency and practicality in modeling
nonlinear behaviors in physical systems. Filtering and smoothing methods for
Wiener systems, such as particle filters and Kalman-based techniques, often
struggle with computational feasibility or accuracy. This work addresses these
challenges by introducing a novel Gaussian Sum Filter for Wiener system state
estimation that is built on a Gauss-Legendre quadrature approximation of the
likelihood function associated with the output signal. In addition to
filtering, a two-filter smoothing strategy is proposed, enabling accurate
computation of smoothed state distributions at single and consecutive time
instants. Numerical examples demonstrate the superiority of the proposed method
in balancing accuracy and computational efficiency compared to traditional
approaches, highlighting its benefits in control, state estimation and system
identification, for Wiener systems.

### 6. [Max-Min Fairness in Stacked Intelligent Metasurface-Aided Rate Splitting Networks](http://arxiv.org/pdf/2505.08521v1)

Authors: Abdullah Quran, Shimaa Naser, Maryam Tariq, Omar Alhussein, Sami Muhaidat

This paper investigates a downlink multiuser multiple-input single-output
system that integrates rate-splitting multiple access (RSMA) with a stacked
intelligent metasurface (SIM) to enable wave-domain beamforming. Unlike
conventional digital beamforming, the proposed system leverages the
programmable phase shifts of the SIM to perform beamforming entirely in the
wave domain. In contrast to existing literature, this work introduces a
fairness-centric SIM-RSMA design that shifts the emphasis from maximizing
sum-rate to ensuring fair allocation of resources. In particular, we formulate
a max-min rate optimization problem that jointly optimizes transmit power
coefficients at the base station and SIM phase shifts. Given the non-convex
nature of this problem, we develop an alternating optimization framework, where
the power allocation is optimized through successive convex approximation and
SIM beamforming is optimized using the Riemannian conjugate gradient method.
Simulation results indicate that combining SIM with RSMA yields superior
max-min performance compared to its integration with space division multiple
access or non-orthogonal multiple access.

### 7. [A High-Efficiency Reconfigurable Bidirectional Array Antenna Based on Transmit-Reflect Switchable Metasurface](http://arxiv.org/pdf/2505.08556v1)

Authors: Fan Qin, Jinyang Bi, Jiao Ma, Chao Gu, Hailin Zhang, Wenchi Cheng, Steven Gao

This paper proposes a reconfigurable bidirectional array antenna with
high-efficiency radiations and flexible beam-switching capability by employing
a novel transmit-reflect switchable metasurface (TRSM). To realize the
electromagnetic (EM) wave transmitted or reflected manipulation, a dedicated
transmit-reflect switch layer (TRSL) with periodically soldered PIN diodes is
introduced between two transmitted metasurfaces. By switching ON/OFF the
embedded diodes, the TRSL performs as a mesh-type ground layer or
polarization-grid layer, exhibiting a reflect or transmit property to the
incident wave respectively. Further, utilizing the above TRSM configuration in
conjunction with a microstrip feed antenna, bidirectional radiations are
obtained at the same frequency and polarization. To further reduce the number
of PIN diodes and control complexity, an enhanced TRSM using a single diode to
control two unit cells is also investigated, resulting in half PIN diodes
reduction. Since the bidirectional beam-switching is achieved by only
controlling PIN diodes integrated in the ground plane instead of directly
acting on the radiation element, which reduces insertion loss and avoids phase
quantization errors, the proposed antenna can maintain a high aperture
efficiency. To verify this concept, a prototype was designed, fabricated, and
measured, demonstrating a successful realization of backward and forward
patterns with peak gains of 22.3 and 22.1 dBi, and aperture efficiencies of
47.2% and 43.8%. The 3-dB gain bandwidths of reflected and transmitted modes
are 13.7% and 12.3%. This antenna has the advantages of high gain, high
aperture efficiency, simple configuration, cost-effectiveness, and flexible and
digital beam control.

### 8. [Robust Indoor Localization via Conformal Methods and Variational Bayesian Adaptive Filtering](http://arxiv.org/pdf/2505.08639v1)

Authors: Zhiyi Zhou, Dongzhuo Liu, Songtao Guo, Yuanyuan Yang

Indoor localization is critical for IoT applications, yet challenges such as
non-Gaussian noise, environmental interference, and measurement outliers hinder
the robustness of traditional methods. Existing approaches, including Kalman
filtering and its variants, often rely on Gaussian assumptions or static
thresholds, limiting adaptability in dynamic environments. This paper proposes
a hierarchical robust framework integrating Variational Bayesian (VB) parameter
learning, Huber M-estimation, and Conformal Outlier Detection (COD) to address
these limitations. First, VB inference jointly estimates state and noise
parameters, adapting to time-varying uncertainties. Second, Huber-based robust
filtering suppresses mild outliers while preserving Gaussian efficiency. Third,
COD provides statistical guarantees for outlier detection via dynamically
calibrated thresholds, ensuring a user-controlled false alarm rate.
Theoretically, we prove the Semi-positive Definiteness of Huber-based Kalman
filtering covariance and the coverage of sliding window conformal prediction.
Experiments on geomagnetic fingerprint datasets demonstrate significant
improvements: fingerprint matching accuracy increases from 81.25% to 93.75%,
and positioning errors decrease from 0.62-6.87 m to 0.03-0.35 m. Comparative
studies further validate the framework's robustness, showing consistent
performance gains under non-Gaussian noise and outlier conditions.

### 9. [Joint Communication Scheduling and Resource Allocation for Distributed Edge Learning: Seamless Integration in Next-Generation Wireless Networks](http://arxiv.org/pdf/2505.08682v1)

Authors: Paul Zheng, Navid Keshtiarast, Pradyumna Kumar Bishoyi, Yao Zhu, Yulin Hu, Marina Petrova, Anke Schmeink

Distributed edge learning (DL) is considered a cornerstone of intelligence
enablers, since it allows for collaborative training without the necessity for
local clients to share raw data with other parties, thereby preserving privacy
and security. Integrating DL into the 6G networks requires a coexistence design
with existing services such as high-bandwidth (HB) traffic like eMBB. Current
designs in the literature mainly focus on communication round-wise designs that
assume a rigid resource allocation throughout each communication round (CR).
However, rigid resource allocation within a CR is a highly inefficient and
inaccurate representation of the system's realistic behavior. This is due to
the heterogeneous nature of the system, as clients inherently may need to
access the network at different times. This work zooms into one arbitrary CR,
and demonstrates the importance of considering a time-dependent resource
sharing design with HB traffic. We first formulate a time-step-wise
optimization problem to minimize the consumed time by DL within the CR while
constrained by a DL energy budget. Due to its intractability, a session-based
optimization problem is formulated assuming a CR lasts less than a large-scale
coherence time. Some scheduling properties of such multi-server joint
communication scheduling and resource allocation framework have been
established. An iterative algorithm has been designed to solve such non-convex
and non-block-separable-constrained problems. Simulation results confirm the
importance of the efficient and accurate integration design proposed in this
work.

### 10. [Load-independent Metrics for Benchmarking Force Controllers](http://arxiv.org/pdf/2505.08730v1)

Authors: Victor Shime, Elisa G. Vergamini, Cícero Zanette, Leonardo F. dos Santos, Lucca Maitan, Andrea Calanca, Thiago Boaventura

Torque-controlled actuators are critical components in mechatronic systems
that closely interact with their environment, such as legged robots,
collaborative manipulators, and exoskeletons. The performance and stability of
these actuators depend not only on controller design and system dynamics but
also significantly on load characteristics, which may include interactions with
humans or unstructured environments. This load dependence highlights the need
for frameworks that properly assess and compare torque controllers independent
of specific loading conditions. In this short paper, we concisely present a
modeling approach that captures the impact of load on the closed-loop dynamics
of torque-controlled systems. Based on this model, we propose new methods and
quantitative metrics, including the Passivity Index Interval, which blends
passivity and small-gain theory to offer a less conservative measure of coupled
stability than passivity alone. These metrics can be used alongside traditional
control performance indicators, such as settling time and bandwidth, to provide
a more comprehensive characterization of torque-controlled systems. We
demonstrate the application of the proposed metrics through experimental
comparisons of linear actuator force controllers.

### Machine Learning (Statistics Category)

### 1. [SIM-Shapley: A Stable and Computationally Efficient Approach to Shapley Value Approximation](http://arxiv.org/pdf/2505.08198v1)

Authors: Wangxuan Fan, Siqi Li, Doudou Zhou, Yohei Okada, Chuan Hong, Molei Liu, Nan Liu

Explainable artificial intelligence (XAI) is essential for trustworthy
machine learning (ML), particularly in high-stakes domains such as healthcare
and finance. Shapley value (SV) methods provide a principled framework for
feature attribution in complex models but incur high computational costs,
limiting their scalability in high-dimensional settings. We propose Stochastic
Iterative Momentum for Shapley Value Approximation (SIM-Shapley), a stable and
efficient SV approximation method inspired by stochastic optimization. We
analyze variance theoretically, prove linear $Q$-convergence, and demonstrate
improved empirical stability and low bias in practice on real-world datasets.
In our numerical experiments, SIM-Shapley reduces computation time by up to 85%
relative to state-of-the-art baselines while maintaining comparable feature
attribution quality. Beyond feature attribution, our stochastic mini-batch
iterative framework extends naturally to a broader class of sample average
approximation problems, offering a new avenue for improving computational
efficiency with stability guarantees. Code is publicly available at
https://github.com/nliulab/SIM-Shapley.

### 2. [Lie Group Symmetry Discovery and Enforcement Using Vector Fields](http://arxiv.org/pdf/2505.08219v1)

Authors: Ben Shaw, Sasidhar Kunapuli, Abram Magner, Kevin R. Moon

Symmetry-informed machine learning can exhibit advantages over machine
learning which fails to account for symmetry. Additionally, recent attention
has been given to continuous symmetry discovery using vector fields which serve
as infinitesimal generators for Lie group symmetries. In this paper, we extend
the notion of non-affine symmetry discovery to functions defined by neural
networks. We further extend work in this area by introducing symmetry
enforcement of smooth models using vector fields. Finally, we extend work on
symmetry discovery using vector fields by providing both theoretical and
experimental material on the restriction of the symmetry search space to
infinitesimal isometries.

### 3. [Density Ratio-based Causal Discovery from Bivariate Continuous-Discrete Data](http://arxiv.org/pdf/2505.08371v1)

Authors: Takashi Nicholas Maeda, Shohei Shimizu, Hidetoshi Matsui

This paper proposes a causal discovery method for mixed bivariate data
consisting of one continuous and one discrete variable. Existing
constraint-based approaches are ineffective in the bivariate setting, as they
rely on conditional independence tests that are not suited to bivariate data.
Score-based methods either impose strong distributional assumptions or face
challenges in fairly comparing causal directions between variables of different
types, due to differences in their information content. We introduce a novel
approach that determines causal direction by analyzing the monotonicity of the
conditional density ratio of the continuous variable, conditioned on different
values of the discrete variable. Our theoretical analysis shows that the
conditional density ratio exhibits monotonicity when the continuous variable
causes the discrete variable, but not in the reverse direction. This property
provides a principled basis for comparing causal directions between variables
of different types, free from strong distributional assumptions and bias
arising from differences in their information content. We demonstrate its
effectiveness through experiments on both synthetic and real-world datasets,
showing superior accuracy compared to existing methods.

### 4. [Learning Treatment Allocations with Risk Control Under Partial Identifiability](http://arxiv.org/pdf/2505.08378v1)

Authors: Sofia Ek, Dave Zachariah

Learning beneficial treatment allocations for a patient population is an
important problem in precision medicine. Many treatments come with adverse side
effects that are not commensurable with their potential benefits. Patients who
do not receive benefits after such treatments are thereby subjected to
unnecessary harm. This is a `treatment risk' that we aim to control when
learning beneficial allocations. The constrained learning problem is challenged
by the fact that the treatment risk is not in general identifiable using either
randomized trial or observational data. We propose a certifiable learning
method that controls the treatment risk with finite samples in the partially
identified setting. The method is illustrated using both simulated and real
data.

### 5. [BAT: Benchmark for Auto-bidding Task](http://arxiv.org/pdf/2505.08485v1)

Authors: Alexandra Khirianova, Ekaterina Solodneva, Andrey Pudovikov, Sergey Osokin, Egor Samosvat, Yuriy Dorn, Alexander Ledovsky, Yana Zenkova

The optimization of bidding strategies for online advertising slot auctions
presents a critical challenge across numerous digital marketplaces. A
significant obstacle to the development, evaluation, and refinement of
real-time autobidding algorithms is the scarcity of comprehensive datasets and
standardized benchmarks.
  To address this deficiency, we present an auction benchmark encompassing the
two most prevalent auction formats. We implement a series of robust baselines
on a novel dataset, addressing the most salient Real-Time Bidding (RTB) problem
domains: budget pacing uniformity and Cost Per Click (CPC) constraint
optimization. This benchmark provides a user-friendly and intuitive framework
for researchers and practitioners to develop and refine innovative autobidding
algorithms, thereby facilitating advancements in the field of programmatic
advertising. The implementation and additional resources can be accessed at the
following repository (https://github.com/avito-tech/bat-autobidding-benchmark,
https://doi.org/10.5281/zenodo.14794182).

### 6. [A new methodology to decompose a parametric domain using reduced order data manifold in machine learning](http://arxiv.org/pdf/2505.08497v1)

Authors: Chetra Mang, Axel TahmasebiMoradi, Mouadh Yagoubi

We propose a new methodology for parametric domain decomposition using
iterative principal component analysis. Starting with iterative principle
component analysis, the high dimension manifold is reduced to the lower
dimension manifold. Moreover, two approaches are developed to reconstruct the
inverse projector to project from the lower data component to the original one.
Afterward, we provide a detailed strategy to decompose the parametric domain
based on the low dimension manifold. Finally, numerical examples of harmonic
transport problem are given to illustrate the efficiency and effectiveness of
the proposed method comparing to the classical meta-models such as neural
networks.

### 7. [Privacy-Preserving Analytics for Smart Meter (AMI) Data: A Hybrid Approach to Comply with CPUC Privacy Regulations](http://arxiv.org/pdf/2505.08237v1)

Authors: Benjamin Westrich

Advanced Metering Infrastructure (AMI) data from smart electric and gas
meters enables valuable insights for utilities and consumers, but also raises
significant privacy concerns. In California, regulatory decisions (CPUC
D.11-07-056 and D.11-08-045) mandate strict privacy protections for customer
energy usage data, guided by the Fair Information Practice Principles (FIPPs).
We comprehensively explore solutions drawn from data anonymization,
privacy-preserving machine learning (differential privacy and federated
learning), synthetic data generation, and cryptographic techniques (secure
multiparty computation, homomorphic encryption). This allows advanced
analytics, including machine learning models, statistical and econometric
analysis on energy consumption data, to be performed without compromising
individual privacy.
  We evaluate each technique's theoretical foundations, effectiveness, and
trade-offs in the context of utility data analytics, and we propose an
integrated architecture that combines these methods to meet real-world needs.
The proposed hybrid architecture is designed to ensure compliance with
California's privacy rules and FIPPs while enabling useful analytics, from
forecasting and personalized insights to academic research and econometrics,
while strictly protecting individual privacy. Mathematical definitions and
derivations are provided where appropriate to demonstrate privacy guarantees
and utility implications rigorously. We include comparative evaluations of the
techniques, an architecture diagram, and flowcharts to illustrate how they work
together in practice. The result is a blueprint for utility data scientists and
engineers to implement privacy-by-design in AMI data handling, supporting both
data-driven innovation and strict regulatory compliance.

### 8. [High-dimensional Bayesian Tobit regression for censored response with Horseshoe prior](http://arxiv.org/pdf/2505.08288v1)

Authors: The Tien Mai

Censored response variables--where outcomes are only partially observed due
to known bounds--arise in numerous scientific domains and present serious
challenges for regression analysis. The Tobit model, a classical solution for
handling left-censoring, has been widely used in economics and beyond. However,
with the increasing prevalence of high-dimensional data, where the number of
covariates exceeds the sample size, traditional Tobit methods become
inadequate. While frequentist approaches for high-dimensional Tobit regression
have recently been developed, notably through Lasso-based estimators, the
Bayesian literature remains sparse and lacks theoretical guarantees. In this
work, we propose a novel Bayesian framework for high-dimensional Tobit
regression that addresses both censoring and sparsity. Our method leverages the
Horseshoe prior to induce shrinkage and employs a data augmentation strategy to
facilitate efficient posterior computation via Gibbs sampling. We establish
posterior consistency and derive concentration rates under sparsity, providing
the first theoretical results for Bayesian Tobit models in high dimensions.
Numerical experiments show that our approach outperforms favorably with the
recent Lasso-Tobit method.
  Our method is implemented in the R package tobitbayes, which can be found on
Github.

### 9. [ConDiSim: Conditional Diffusion Models for Simulation Based Inference](http://arxiv.org/pdf/2505.08403v1)

Authors: Mayank Nautiyal, Andreas Hellander, Prashant Singh

We present a conditional diffusion model - ConDiSim, for simulation-based
inference of complex systems with intractable likelihoods. ConDiSim leverages
denoising diffusion probabilistic models to approximate posterior
distributions, consisting of a forward process that adds Gaussian noise to
parameters, and a reverse process learning to denoise, conditioned on observed
data. This approach effectively captures complex dependencies and
multi-modalities within posteriors. ConDiSim is evaluated across ten benchmark
problems and two real-world test problems, where it demonstrates effective
posterior approximation accuracy while maintaining computational efficiency and
stability in model training. ConDiSim offers a robust and extensible framework
for simulation-based inference, particularly suitable for parameter inference
workflows requiring fast inference methods.

### 10. [A note on concentration inequalities for the overlapped batch mean variance estimators for Markov chains](http://arxiv.org/pdf/2505.08456v1)

Authors: Eric Moulines, Alexey Naumov, Sergey Samsonov

In this paper, we study the concentration properties of quadratic forms
associated with Markov chains using the martingale decomposition method
introduced by Atchad\'e and Cattaneo (2014). In particular, we derive
concentration inequalities for the overlapped batch mean (OBM) estimators of
the asymptotic variance for uniformly geometrically ergodic Markov chains. Our
main result provides an explicit control of the $p$-th moment of the difference
between the OBM estimator and the asymptotic variance of the Markov chain with
explicit dependence upon $p$ and mixing time of the underlying Markov chain.

