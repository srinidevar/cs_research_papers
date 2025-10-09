# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-08 17:00:25.855659 PST.

### Artificial Intelligence

### 1. [From Agentification to Self-Evolving Agentic AI for Wireless Networks: Concepts, Approaches, and Future Research Directions](http://arxiv.org/pdf/2510.05596v1)

Authors: Changyuan Zhao, Ruichen Zhang, Jiacheng Wang, Dusit Niyato, Geng Sun, Xianbin Wang, Shiwen Mao, Abbas Jamalipour

Self-evolving agentic artificial intelligence (AI) offers a new paradigm for
future wireless systems by enabling autonomous agents to continually adapt and
improve without human intervention. Unlike static AI models, self-evolving
agents embed an autonomous evolution cycle that updates models, tools, and
workflows in response to environmental dynamics. This paper presents a
comprehensive overview of self-evolving agentic AI, highlighting its layered
architecture, life cycle, and key techniques, including tool intelligence,
workflow optimization, self-reflection, and evolutionary learning. We further
propose a multi-agent cooperative self-evolving agentic AI framework, where
multiple large language models (LLMs) are assigned role-specialized prompts
under the coordination of a supervisor agent. Through structured dialogue,
iterative feedback, and systematic validation, the system autonomously executes
the entire life cycle without human intervention. A case study on antenna
evolution in low-altitude wireless networks (LAWNs) demonstrates how the
framework autonomously upgrades fixed antenna optimization into movable antenna
optimization. Experimental results show that the proposed self-evolving agentic
AI autonomously improves beam gain and restores degraded performance by up to
52.02%, consistently surpassing the fixed baseline with little to no human
intervention and validating its adaptability and robustness for next-generation
wireless intelligence.

### 2. [Large Language Model-Based Uncertainty-Adjusted Label Extraction for Artificial Intelligence Model Development in Upper Extremity Radiography](http://arxiv.org/pdf/2510.05664v1)

Authors: Hanna Kreutzer, Anne-Sophie Caselitz, Thomas Dratsch, Daniel Pinto dos Santos, Christiane Kuhl, Daniel Truhn, Sven Nebelung

Objectives: To evaluate GPT-4o's ability to extract diagnostic labels (with
uncertainty) from free-text radiology reports and to test how these labels
affect multi-label image classification of musculoskeletal radiographs.
Methods: This retrospective study included radiography series of the clavicle
(n=1,170), elbow (n=3,755), and thumb (n=1,978). After anonymization, GPT-4o
filled out structured templates by indicating imaging findings as present
("true"), absent ("false"), or "uncertain." To assess the impact of label
uncertainty, "uncertain" labels of the training and validation sets were
automatically reassigned to "true" (inclusive) or "false" (exclusive).
Label-image-pairs were used for multi-label classification using ResNet50.
Label extraction accuracy was manually verified on internal (clavicle: n=233,
elbow: n=745, thumb: n=393) and external test sets (n=300 for each).
Performance was assessed using macro-averaged receiver operating characteristic
(ROC) area under the curve (AUC), precision recall curves, sensitivity,
specificity, and accuracy. AUCs were compared with the DeLong test. Results:
Automatic extraction was correct in 98.6% (60,618 of 61,488) of labels in the
test sets. Across anatomic regions, label-based model training yielded
competitive performance measured by macro-averaged AUC values for inclusive
(e.g., elbow: AUC=0.80 [range, 0.62-0.87]) and exclusive models (elbow:
AUC=0.80 [range, 0.61-0.88]). Models generalized well on external datasets
(elbow [inclusive]: AUC=0.79 [range, 0.61-0.87]; elbow [exclusive]: AUC=0.79
[range, 0.63-0.89]). No significant differences were observed across labeling
strategies or datasets (p>=0.15). Conclusion: GPT-4o extracted labels from
radiologic reports to train competitive multi-label classification models with
high accuracy. Detected uncertainty in the radiologic reports did not influence
the performance of these models.

### 3. [Joint Communication Scheduling and Velocity Control for Multi-UAV-Assisted Post-Disaster Monitoring: An Attention-Based In-Context Learning Approach](http://arxiv.org/pdf/2510.05698v1)

Authors: Yousef Emami, Seyedsina Nabavirazavi, Jingjing Zheng, Hao Zhou, Miguel Gutierrez Gaitan, Kai Li, Luis Almeida

Recently, Unmanned Aerial Vehicles (UAVs) are increasingly being investigated
to collect sensory data in post-disaster monitoring scenarios, such as
tsunamis, where early actions are critical to limit coastal damage. A major
challenge is to design the data collection schedules and flight velocities, as
unfavorable schedules and velocities can lead to transmission errors and buffer
overflows of the ground sensors, ultimately resulting in significant packet
loss. Meanwhile, online Deep Reinforcement Learning (DRL) solutions have a
complex training process and a mismatch between simulation and reality that
does not meet the urgent requirements of tsunami monitoring. Recent advances in
Large Language Models (LLMs) offer a compelling alternative. With their strong
reasoning and generalization capabilities, LLMs can adapt to new tasks through
In-Context Learning (ICL), which enables task adaptation through natural
language prompts and example-based guidance without retraining. However, LLM
models have input data limitations and thus require customized approaches. In
this paper, a joint optimization of data collection schedules and velocities
control for multiple UAVs is proposed to minimize data loss. The battery level
of the ground sensors, the length of the queues, and the channel conditions, as
well as the trajectories of the UAVs, are taken into account. Attention-Based
In-Context Learning for Velocity Control and Data Collection Schedule (AIC-VDS)
is proposed as an alternative to DRL in emergencies. The simulation results
show that the proposed AIC-VDS outperforms both the Deep-Q-Network (DQN) and
maximum channel gain baselines.

### 4. [Syn-Diag: An LLM-based Synergistic Framework for Generalizable Few-shot Fault Diagnosis on the Edge](http://arxiv.org/pdf/2510.05733v1)

Authors: Zijun Jia, Shuang Liang, Jinsong Yu

Industrial fault diagnosis faces the dual challenges of data scarcity and the
difficulty of deploying large AI models in resource-constrained environments.
This paper introduces Syn-Diag, a novel cloud-edge synergistic framework that
leverages Large Language Models to overcome these limitations in few-shot fault
diagnosis. Syn-Diag is built on a three-tiered mechanism: 1) Visual-Semantic
Synergy, which aligns signal features with the LLM's semantic space through
cross-modal pre-training; 2) Content-Aware Reasoning, which dynamically
constructs contextual prompts to enhance diagnostic accuracy with limited
samples; and 3) Cloud-Edge Synergy, which uses knowledge distillation to create
a lightweight, efficient edge model capable of online updates via a shared
decision space. Extensive experiments on six datasets covering different CWRU
and SEU working conditions show that Syn-Diag significantly outperforms
existing methods, especially in 1-shot and cross-condition scenarios. The edge
model achieves performance comparable to the cloud version while reducing model
size by 83% and latency by 50%, offering a practical, robust, and deployable
paradigm for modern intelligent diagnostics.

### 5. [ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming](http://arxiv.org/pdf/2510.05774v1)

Authors: Weichun Shi, Minghao Liu, Wanting Zhang, Langchen Shi, Fuqi Jia, Feifei Ma, Jian Zhang

Constraint programming (CP) is a crucial technology for solving real-world
constraint optimization problems (COPs), with the advantages of rich modeling
semantics and high solving efficiency. Using large language models (LLMs) to
generate formal modeling automatically for COPs is becoming a promising
approach, which aims to build trustworthy neuro-symbolic AI with the help of
symbolic solvers. However, CP has received less attention compared to works
based on operations research (OR) models. We introduce ConstraintLLM, the first
LLM specifically designed for CP modeling, which is trained on an open-source
LLM with multi-instruction supervised fine-tuning. We propose the
Constraint-Aware Retrieval Module (CARM) to increase the in-context learning
capabilities, which is integrated in a Tree-of-Thoughts (ToT) framework with
guided self-correction mechanism. Moreover, we construct and release IndusCP,
the first industrial-level benchmark for CP modeling, which contains 140
challenging tasks from various domains. Our experiments demonstrate that
ConstraintLLM achieves state-of-the-art solving accuracy across multiple
benchmarks and outperforms the baselines by 2x on the new IndusCP benchmark.
Code and data are available at: https://github.com/william4s/ConstraintLLM.

### 6. [Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies](http://arxiv.org/pdf/2510.05909v1)

Authors: Aksel Joonas Reedi, Corentin Léger, Julien Pourcel, Loris Gaven, Perrine Charriau, Guillaume Pourcel

Large Language Models (LLMs) optimized to output truthful answers often
overfit, producing brittle reasoning that fails to generalize. While
persuasion-based optimization has shown promise in debate settings, it has not
been systematically compared against mainstream truth-based approaches. We
introduce DebateQD, a minimal Quality-Diversity (QD) evolutionary algorithm
that evolves diverse debate strategies across different categories
(rationality, authority, emotional appeal, etc.) through tournament-style
competitions where two LLMs debate while a third judges. Unlike previously
proposed methods that require a population of LLMs, our approach maintains
diversity of opponents through prompt-based strategies within a single LLM
architecture, making it more accessible for experiments while preserving the
key benefits of population-based optimization. In contrast to prior work, we
explicitly isolate the role of the optimization objective by fixing the debate
protocol and swapping only the fitness function: persuasion rewards strategies
that convince the judge irrespective of truth, whereas truth rewards
collaborative correctness. Across three model scales (7B, 32B, 72B parameters)
and multiple dataset sizes from the QuALITY benchmark, persuasion-optimized
strategies achieve up to 13.94% smaller train-test generalization gaps, while
matching or exceeding truth optimization's test performance. These results
provide the first controlled evidence that competitive pressure to persuade,
rather than seek the truth collaboratively, fosters more transferable reasoning
skills, offering a promising path for improving LLM generalization.

### 7. [Training-Free Time Series Classification via In-Context Reasoning with LLM Agents](http://arxiv.org/pdf/2510.05950v1)

Authors: Songyuan Sui, Zihang Xu, Yu-Neng Chuang, Kwei-Herng Lai, Xia Hu

Time series classification (TSC) spans diverse application scenarios, yet
labeled data are often scarce, making task-specific training costly and
inflexible. Recent reasoning-oriented large language models (LLMs) show promise
in understanding temporal patterns, but purely zero-shot usage remains
suboptimal. We propose FETA, a multi-agent framework for training-free TSC via
exemplar-based in-context reasoning. FETA decomposes a multivariate series into
channel-wise subproblems, retrieves a few structurally similar labeled examples
for each channel, and leverages a reasoning LLM to compare the query against
these exemplars, producing channel-level labels with self-assessed confidences;
a confidence-weighted aggregator then fuses all channel decisions. This design
eliminates the need for pretraining or fine-tuning, improves efficiency by
pruning irrelevant channels and controlling input length, and enhances
interpretability through exemplar grounding and confidence estimation. On nine
challenging UEA datasets, FETA achieves strong accuracy under a fully
training-free setting, surpassing multiple trained baselines. These results
demonstrate that a multi-agent in-context reasoning framework can transform
LLMs into competitive, plug-and-play TSC solvers without any parameter
training. The code is available at https://github.com/SongyuanSui/FETATSC.

### 8. [ARISE: An Adaptive Resolution-Aware Metric for Test-Time Scaling Evaluation in Large Reasoning Models](http://arxiv.org/pdf/2510.06014v1)

Authors: Zhangyue Yin, Qiushi Sun, Zhiyuan Zeng, Zhiyuan Yu, Qipeng Guo, Xuanjing Huang, Xipeng Qiu

Test-time scaling has emerged as a transformative paradigm for enhancing the
performance of large reasoning models, enabling dynamic allocation of
computational resources during inference. However, as the landscape of
reasoning models rapidly expands, a critical question remains: how can we
systematically compare and evaluate the test-time scaling capabilities across
different models? In this paper, we introduce ARISE (Adaptive Resolution-aware
Scaling Evaluation), a novel metric specifically designed to assess the
test-time scaling effectiveness of large reasoning models. Unlike existing
evaluation approaches, ARISE incorporates two key innovations: (1) sample-level
awareness that effectively penalizes negative scaling behaviors where increased
computation leads to performance degradation, and (2) a dynamic sampling
mechanism that mitigates the impact of accuracy fluctuations and token count
instability on the final assessment. We conduct comprehensive experiments
evaluating state-of-the-art reasoning models across diverse domains including
mathematical reasoning, code generation, and agentic tasks. Our results
demonstrate that ARISE provides a reliable and fine-grained measurement of
test-time scaling capabilities, revealing significant variations in scaling
efficiency across models. Notably, our evaluation identifies Claude Opus as
exhibiting superior scaling characteristics compared to other contemporary
reasoning models.

### 9. [Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research](http://arxiv.org/pdf/2510.06056v1)

Authors: Gang Liu, Yihan Zhu, Jie Chen, Meng Jiang

Large language models hold promise as scientific assistants, yet existing
agents either rely solely on algorithm evolution or on deep research in
isolation, both of which face critical limitations. Pure algorithm evolution,
as in AlphaEvolve, depends only on the internal knowledge of LLMs and quickly
plateaus in complex domains, while pure deep research proposes ideas without
validation, resulting in unrealistic or unimplementable solutions. We present
DeepEvolve, an agent that integrates deep research with algorithm evolution,
uniting external knowledge retrieval, cross-file code editing, and systematic
debugging under a feedback-driven iterative loop. Each iteration not only
proposes new hypotheses but also refines, implements, and tests them, avoiding
both shallow improvements and unproductive over-refinements. Across nine
benchmarks in chemistry, mathematics, biology, materials, and patents,
DeepEvolve consistently improves the initial algorithm, producing executable
new algorithms with sustained gains. By bridging the gap between unguided
evolution and research without grounding, DeepEvolve provides a reliable
framework for advancing scientific algorithm discovery. Our code is available
at https://github.com/liugangcode/deepevolve.

### 10. [Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents](http://arxiv.org/pdf/2510.06078v1)

Authors: Tao Zhe, Rui Liu, Fateme Memar, Xiao Luo, Wei Fan, Xinyue Ye, Zhongren Peng, Dongjie Wang

Route recommendation aims to provide users with optimal travel plans that
satisfy diverse and complex requirements. Classical routing algorithms (e.g.,
shortest-path and constraint-aware search) are efficient but assume structured
inputs and fixed objectives, limiting adaptability to natural-language queries.
Recent LLM-based approaches enhance flexibility but struggle with spatial
reasoning and the joint modeling of route-level and POI-level preferences. To
address these limitations, we propose RouteLLM, a hierarchical multi-agent
framework that grounds natural-language intents into constraint-aware routes.
It first parses user queries into structured intents including POIs, paths, and
constraints. A manager agent then coordinates specialized sub-agents: a
constraint agent that resolves and formally check constraints, a POI agent that
retrieves and ranks candidate POIs, and a path refinement agent that refines
routes via a routing engine with preference-conditioned costs. A final verifier
agent ensures constraint satisfaction and produces the final route with an
interpretable rationale. This design bridges linguistic flexibility and spatial
structure, enabling reasoning over route feasibility and user preferences.
Experiments show that our method reliably grounds textual preferences into
constraint-aware routes, improving route quality and preference satisfaction
over classical methods.

### Hardware Architecture

### 1. [An opportunity to improve Data Center Efficiency: Optimizing the Server's Upgrade Cycle](http://arxiv.org/pdf/2510.05787v1)

Authors: Panagiota Nikolaou, Freddy Gabbay, Jawad Haj-Yahya, Yiannakis Sazeides

This work aims to improve a data center's efficiency by optimizing the server
upgrade plan: determine the optimal timing for replacing old servers with new
ones. The opportunity presented by this approach is demonstrated through a
study based on historical server data. The study establishes a significant
opportunity to increase the QPS/(TCOxCO2) metric by formulating a global
upgrade plan at the data center's design time covering its entire life cycle.
This plan leverages information, such as server entry year, performance, and
active power consumption for both existing and future servers. Our findings
reveal that an optimal global upgrade plan, may involve upgrades at non fixed
time periods and outperforms local upgrade plans. Local upgrade plans follow a
fixed, equal-length cycle and make decisions based only on currently available
server models. These local plans select the best available server at each
upgrade cycle without accounting for future server releases.

### 2. [From Principles to Practice: A Systematic Study of LLM Serving on Multi-core NPUs](http://arxiv.org/pdf/2510.05632v1)

Authors: Tianhao Zhu, Dahu Feng, Erhu Feng, Yubin Xia

With the widespread adoption of Large Language Models (LLMs), the demand for
high-performance LLM inference services continues to grow. To meet this demand,
a growing number of AI accelerators have been proposed, such as Google TPU,
Huawei NPU, Graphcore IPU, and Cerebras WSE, etc. Most of these accelerators
adopt multi-core architectures to achieve enhanced scalability, but lack the
flexibility of SIMT architectures. Therefore, without careful configuration of
the hardware architecture, as well as deliberate design of tensor parallelism
and core placement strategies, computational resources may be underutilized,
resulting in suboptimal inference performance.
  To address these challenges, we first present a multi-level simulation
framework with both transaction-level and performance-model-based simulation
for multi-core NPUs. Using this simulator, we conduct a systematic analysis and
further propose the optimal solutions for tensor parallelism strategies, core
placement policies, memory management methods, as well as the selection between
PD-disaggregation and PD-fusion on multi-core NPUs. We conduct comprehensive
experiments on representative LLMs and various NPU configurations. The
evaluation results demonstrate that, our solution can achieve 1.32x-6.03x
speedup compared to SOTA designs for multi-core NPUs across different hardware
configurations. As for LLM serving, our work offers guidance on designing
optimal hardware architectures and serving strategies for multi-core NPUs
across various LLM workloads.

### 3. [cMPI: Using CXL Memory Sharing for MPI One-Sided and Two-Sided Inter-Node Communications](http://arxiv.org/pdf/2510.05476v1)

Authors: Xi Wang, Bin Ma, Jongryool Kim, Byungil Koh, Hoshik Kim, Dong Li

Message Passing Interface (MPI) is a foundational programming model for
high-performance computing. MPI libraries traditionally employ network
interconnects (e.g., Ethernet and InfiniBand) and network protocols (e.g., TCP
and RoCE) with complex software stacks for cross-node communication. We present
cMPI, the first work to optimize MPI point-to-point communication (both
one-sided and two-sided) using CXL memory sharing on a real CXL platform,
transforming cross-node communication into memory transactions and data copies
within CXL memory, bypassing traditional network protocols. We analyze
performance across various interconnects and find that CXL memory sharing
achieves 7.2x-8.1x lower latency than TCP-based interconnects deployed in
small- and medium-scale clusters. We address challenges of CXL memory sharing
for MPI communication, including data object management over the dax
representation [50], cache coherence, and atomic operations. Overall, cMPI
outperforms TCP over standard Ethernet NIC and high-end SmartNIC by up to 49x
and 72x in latency and bandwidth, respectively, for small messages.

### 4. [Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting](http://arxiv.org/pdf/2510.05497v1)

Authors: Zhongkai Yu, Yue Guan, Zihao Yu, Chenyang Zhou, Shuyi Pei, Yangwook Kang, Yufei Ding, Po-An Tsai

Large Language Models (LLMs) with Mixture of Experts (MoE) architectures
achieve remarkable performance improvements, but their random expert selection
mechanism introduces significant data movement overhead that becomes the
dominant bottleneck in multi-unit serving systems. To forecast the patterns
underlying this data movement, we conduct comprehensive data-movement-centric
profiling across three state-of-the-art large-scale MoE models (200B- 671B)
using over 24,000 requests spanning diverse workloads. With the resulting
150GB+ trace files, we perform systematic analysis from both temporal and
spatial perspectives and distill six key insights to guide the design of
diverse future serving systems. Taking wafer-scale GPUs as a case study, we
demonstrate that minor architectural modifications leveraging our insights
achieve substantial performance gains, delivering 6.3X and 4.0X average
speedups on DeepSeek V3 and Qwen3, respectively. Our work provides the first
comprehensive data-centric analysis of MoE models at scale. Our profiling
traces and analysis results are publicly available at
{https://huggingface.co/datasets/core12345/MoE_expert_selection_trace. We will
also release our simulation framework shortly to facilitate future research in
this area.

### Computational Complexity

### 1. [Fundamental Limits of Crystalline Equivariant Graph Neural Networks: A Circuit Complexity Perspective](http://arxiv.org/pdf/2510.05494v1)

Authors: Yang Cao, Zhao Song, Jiahao Zhang, Jiale Zhao

Graph neural networks (GNNs) have become a core paradigm for learning on
relational data. In materials science, equivariant GNNs (EGNNs) have emerged as
a compelling backbone for crystalline-structure prediction, owing to their
ability to respect Euclidean symmetries and periodic boundary conditions.
Despite strong empirical performance, their expressive power in periodic,
symmetry-constrained settings remains poorly understood. This work
characterizes the intrinsic computational and expressive limits of EGNNs for
crystalline-structure prediction through a circuit-complexity lens. We analyze
the computations carried out by EGNN layers acting on node features, atomic
coordinates, and lattice matrices, and prove that, under polynomial precision,
embedding width $d=O(n)$ for $n$ nodes, $O(1)$ layers, and $O(1)$-depth,
$O(n)$-width MLP instantiations of the message/update/readout maps, these
models admit a simulation by a uniform $\mathsf{TC}^0$ threshold-circuit family
of polynomial size (with an explicit constant-depth bound). Situating EGNNs
within $\mathsf{TC}^0$ provides a concrete ceiling on the decision and
prediction problems solvable by such architectures under realistic resource
constraints and clarifies which architectural modifications (e.g., increased
depth, richer geometric primitives, or wider layers) are required to transcend
this regime. The analysis complements Weisfeiler-Lehman style results that do
not directly transfer to periodic crystals, and offers a complexity-theoretic
foundation for symmetry-aware graph learning on crystalline systems.

### 2. [On the Interplay of Cube Learning and Dependency Schemes in QCDCL Proof Systems](http://arxiv.org/pdf/2510.05876v1)

Authors: Abhimanyu Choudhury, Meena Mahajan

Quantified Conflict Driven Clause Leaning (QCDCL) is one of the main
approaches to solving Quantified Boolean Formulas (QBF). Cube-learning is
employed in this approach to ensure that true formulas can be verified.
Dependency Schemes help to detect spurious dependencies that are implied by the
variable ordering in the quantifier prefix of QBFs but are not essential for
constructing (counter)models. This detection can provably shorten refutations
in specific proof systems, and is expected to speed up runs of QBF solvers.
  The simplest underlying proof system [BeyersdorffB\"ohm-LMCS2023], formalises
the reasoning in the QCDCL approach on false formulas, when neither cube
learning nor dependency schemes is used. The work of
[B\"ohmPeitlBeyersdorff-AI2024] further incorporates cube-learning. The work of
[ChoudhuryMahajan-JAR2024] incorporates a limited use of dependency schemes,
but without cube-learning.
  In this work, proof systems underlying the reasoning of QCDCL solvers which
use cube learning, and which use dependency schemes at all stages, are
formalised. Sufficient conditions for soundness and completeness are presented,
and it is shown that using the standard and reflexive resolution path
dependency schemes ($D^{std}$ and $D^{rrs}$) to relax the decision order
provably shortens refutations.
  When the decisions are restricted to follow quantification order, but
dependency schemes are used in propagation and learning, in conjunction with
cube-learning, the resulting proof systems using the dependency schemes
$D^{std}$ and $D^{rrs}$ are investigated in detail and their relative strengths
are analysed.

### 3. [Computational Complexity in Property Testing](http://arxiv.org/pdf/2510.05927v1)

Authors: Renato Ferreira Pinto Jr., Diptaksho Palit, Sofya Raskhodnikova

We initiate a systematic study of the computational complexity of property
testing, focusing on the relationship between query and time complexity. While
traditional work in property testing has emphasized query complexity,
relatively little is known about the computational hardness of property
testers. Our goal is to chart the landscape of time-query interplay and develop
tools for proving time complexity lower bounds. Our first contribution is a
pair of time-query hierarchy theorems for property testing. For all suitable
nondecreasing functions $q(n)$ and $t(n)$ with $t(n)\geq q(n)$, we construct
properties with query complexity $\tilde{\Theta}(q(n))$ and time complexity
$\tilde\Omega(t(n))$. Our weak hierarchy holds unconditionally, whereas the
strong version-assuming the Strong Exponential Time Hypothesis-provides better
control over the time complexity of the constructed properties.
  We then turn to halfspaces in $\mathbb{R}^d$, a fundamental class in property
testing and learning theory. We study the problem of approximating the distance
from the input function to the nearest halfspace within additive error
$\epsilon$. For the distribution-free distance approximation problem, known
algorithms achieve query complexity $O(d/\epsilon^2)$, but take time
$\tilde{\Theta}(1/\epsilon^d)$. We provide a fine-grained justification for
this gap: assuming the $k$-SUM conjecture, any algorithm must have running time
${\Omega}(1/\epsilon^{d/2})$. This fine-grained lower bound yields a provable
separation between query and time complexity for a natural and well-studied
(tolerant) testing problem. We also prove that any Statistical Query (SQ)
algorithm under the standard Gaussian distribution requires
$(1/\epsilon)^{\Omega(d)}$ queries if the queries are answered with additive
error up to $\epsilon^{\Omega(d)}$, revealing a fundamental barrier even in the
distribution-specific setting.

### 4. [Learning stabilizer structure of quantum states](http://arxiv.org/pdf/2510.05890v1)

Authors: Srinivasan Arunachalam, Arkopal Dutt

We consider the task of learning a structured stabilizer decomposition of an
arbitrary $n$-qubit quantum state $|\psi\rangle$: for $\varepsilon > 0$, output
a state $|\phi\rangle$ with stabilizer-rank $\textsf{poly}(1/\varepsilon)$ such
that $|\psi\rangle=|\phi\rangle+|\phi'\rangle$ where $|\phi'\rangle$ has
stabilizer fidelity $< \varepsilon$. We firstly show the existence of such
decompositions using the recently established inverse theorem for the
Gowers-$3$ norm of states [AD,STOC'25].
  To learn this structure, we initiate the task of self-correction of a state
$|\psi\rangle$ with respect to a class of states $\textsf{C}$: given copies of
$|\psi\rangle$ which has fidelity $\geq \tau$ with a state in $\textsf{C}$,
output $|\phi\rangle \in \textsf{C}$ with fidelity $|\langle \phi | \psi
\rangle|^2 \geq \tau^C$ for a constant $C>1$. Assuming the algorithmic
polynomial Frieman-Rusza (APFR) conjecture (whose combinatorial version was
recently resolved [GGMT,Annals of Math.'25], we give a polynomial-time
algorithm for self-correction of stabilizer states. Given access to the state
preparation unitary $U_\psi$ for $|\psi\rangle$ and its controlled version
$cU_\psi$, we give a polynomial-time protocol that learns a structured
decomposition of $|\psi\rangle$. Without assuming APFR, we give a
quasipolynomial-time protocol for the same task.
  As our main application, we give learning algorithms for states
$|\psi\rangle$ promised to have stabilizer extent $\xi$, given access to
$U_\psi$ and $cU_\psi$. We give a protocol that outputs $|\phi\rangle$ which is
constant-close to $|\psi\rangle$ in time $\textsf{poly}(n,\xi^{\log \xi})$,
which can be improved to polynomial-time assuming APFR. This gives an
unconditional learning algorithm for stabilizer-rank $k$ states in time
$\textsf{poly}(n,k^{k^2})$. As far as we know, learning arbitrary states with
even stabilizer-rank $k \geq 2$ was unknown.

### 5. [Efficient Heuristics and Exact Methods for Pairwise Interaction Sampling](http://arxiv.org/pdf/2510.05955v1)

Authors: Sándor P. Fekete, Phillip Keldenich, Dominik Krupke, Michael Perk

We consider a class of optimization problems that are fundamental to testing
in modern configurable software systems, e.g., in automotive industries. In
pairwise interaction sampling, we are given a (potentially very large)
configuration space, in which each dimension corresponds to a possible Boolean
feature of a software system; valid configurations are the satisfying
assignments of a given propositional formula $\varphi$. The objective is to
find a minimum-sized family of configurations, such that each pair of features
is jointly tested at least once. Due to its relevance in Software Engineering,
this problem has been studied extensively for over 20 years. In addition to new
theoretical insights (we prove BH-hardness), we provide a broad spectrum of key
contributions on the practical side that allow substantial progress for the
practical performance. Remarkably, we are able to solve the largest instances
we found in published benchmark sets (with about 500000000 feasible
interactions) to provable optimality. Previous approaches were not even able to
compute feasible solutions.

### Computational Engineering

### 1. [Gaussian Ensemble Topology (GET): A New Explicit and Inherently Smooth Framework for Manufacture-Ready Topology Optimization](http://arxiv.org/pdf/2510.05572v1)

Authors: Xinyu Ma, Chengxin Wang, Meng Wang, Xu Guo, Liu Yang, Huajian Gao

We introduce the Gaussian Ensemble Topology (GET) method, a new explicit and
manufacture-ready framework for topology optimization in which design
geometries are represented as superpositions of anisotropic Gaussian functions.
By combining explicit Gaussian descriptions with a level-set-like Heaviside
projection, GET inherently generates smooth, curvature-continuous designs
without requiring post-processing steps such as mesh or corner smoothing and
feature extraction. The method is validated on standard compliance-minimization
and compliant mechanism benchmarks in two and three dimensions. The optimized
designs achieve objective values comparable to those obtained with classical
Moving Morphable Component (MMC) approaches, but with geometrically consistent,
refined boundaries. Numerical examples demonstrate additional advantages of the
GET framework, including mesh independence inherent to explicit
parameterizations, strong geometric expressiveness, and effective control over
smoothness, discreteness, and structural complexity through parameter tuning.
As a robust and manufacture-ready approach to explicit topology optimization,
GET opens avenues for tackling advanced and complex design problems.

### 2. [Code Smell Detection via Pearson Correlation and ML Hyperparameter Optimization](http://arxiv.org/pdf/2510.05835v1)

Authors: Moinuddin Muhammad Imtiaz Bhuiyan, Kazi Ekramul Hoque, Rakibul Islam, Md. Mahbubur Rahman Tusher, Najmul Hassan, Yoichi Tomioka, Satoshi Nishimura, Jungpil Shin, Abu Saleh Musa Miah

This study addresses the challenge of detecting code smells in large-scale
software systems using machine learning (ML). Traditional detection methods
often suffer from low accuracy and poor generalization across different
datasets. To overcome these issues, we propose a machine learning-based model
that automatically and accurately identifies code smells, offering a scalable
solution for software quality analysis. The novelty of our approach lies in the
use of eight diverse ML algorithms, including XGBoost, AdaBoost, and other
classifiers, alongside key techniques such as the Synthetic Minority
Over-sampling Technique (SMOTE) for class imbalance and Pearson correlation for
efficient feature selection. These methods collectively improve model accuracy
and generalization. Our methodology involves several steps: first, we
preprocess the data and apply SMOTE to balance the dataset; next, Pearson
correlation is used for feature selection to reduce redundancy; followed by
training eight ML algorithms and tuning hyperparameters through Grid Search,
Random Search, and Bayesian Optimization. Finally, we evaluate the models using
accuracy, F-measure, and confusion matrices. The results show that AdaBoost,
Random Forest, and XGBoost perform best, achieving accuracies of 100%, 99%, and
99%, respectively. This study provides a robust framework for detecting code
smells, enhancing software quality assurance, and demonstrating the
effectiveness of a comprehensive, optimized ML approach.

### 3. [A comprehensive comparison of neural operators for 3D industry-scale engineering designs](http://arxiv.org/pdf/2510.05995v1)

Authors: Weiheng Zhong, Qibang Liu, Diab Abueidda, Seid Koric, Hadi Meidani

Neural operators have emerged as powerful tools for learning nonlinear
mappings between function spaces, enabling real-time prediction of complex
dynamics in diverse scientific and engineering applications. With their growing
adoption in engineering design evaluation, a wide range of neural operator
architectures have been proposed for various problem settings. However, model
selection remains challenging due to the absence of fair and comprehensive
comparisons. To address this, we propose and standardize six representative 3D
industry-scale engineering design datasets spanning thermal analysis, linear
elasticity, elasto-plasticity, time-dependent plastic problems, and
computational fluid dynamics. All datasets include fully preprocessed inputs
and outputs for model training, making them directly usable across diverse
neural operator architectures. Using these datasets, we conduct a systematic
comparison of four types of neural operator variants, including
Branch-Trunk-based Neural Operators inspired by DeepONet, Graph-based Neural
Operators inspired by Graph Neural Networks, Grid-based Neural Operators
inspired by Fourier Neural Operators, and Point-based Neural Operators inspired
by PointNet. We further introduce practical enhancements to adapt these models
to different engineering settings, improving the fairness of the comparison.
Our benchmarking study evaluates each model strengths and limitations in terms
of predictive performance, computational efficiency, memory usage, and
deployment complexity. The findings provide actionable insights to guide future
neural operator development.

### 4. [Intertemporal Pricing of Time-Bound Stablecoins: Measuring and Controlling the Liquidity-of-Time Premium](http://arxiv.org/pdf/2510.05711v1)

Authors: Ailiya Borjigin, Cong He

Time-bound stablecoins are DeFi assets that temporarily tokenize traditional
securities during market off-hours, enabling continuous cross-market liquidity.
We introduce the Liquidity-of-Time Premium (TLP): the extra return or cost of
providing liquidity when the primary market is closed. We build a no-arbitrage
pricing model that yields a band for fair values over different expiries, and a
dynamic risk-control mechanism that adjusts loan-to-value (LTV) ratios in real
time to keep TLP within a target range. Our analysis blends financial
engineering (no-arbitrage conditions, option-style pricing) with empirical
finance (event studies on cross-listed stocks and futures) to measure TLP under
time-zone frictions. We define TLP formally, derive closed-form expressions for
its term structure under idealized assumptions, and simulate scenarios that
vary volatility and collateralization. We then propose an LTV policy that
raises or lowers collateral to expand or curtail time-bound stablecoin supply,
analogous to a central bank adjusting rates to defend a peg. We outline
empirical proxies for TLP, including ADR premiums, overseas index futures
versus cash index divergence, and pre-market versus official close gaps.
Results show that TLP grows with closure length and volatility, yet can be
contained by adaptive LTV. We provide backtests and figures (term-structure
curves, capital-efficiency versus tail-risk trade-offs, time-liquidity
heatmaps) and discuss protocol design (vault structure, closing-price oracles,
on-chain auction liquidations). The findings position time-bound stablecoins as
a tool to reduce temporal market inefficiencies and inform future research and
deployment.

### 5. [Physicochemically Informed Dual-Conditioned Generative Model of T-Cell Receptor Variable Regions for Cellular Therapy](http://arxiv.org/pdf/2510.05747v1)

Authors: Jiahao Ma, Hongzong Li, Ye-Fan Hu, Jian-Dong Huang

Physicochemically informed biological sequence generation has the potential
to accelerate computer-aided cellular therapy, yet current models fail to
\emph{jointly} ensure novelty, diversity, and biophysical plausibility when
designing variable regions of T-cell receptors (TCRs). We present
\textbf{PhysicoGPTCR}, a large generative protein Transformer that is
\emph{dual-conditioned} on peptide and HLA context and trained to
autoregressively synthesise TCR sequences while embedding residue-level
physicochemical descriptors. The model is optimised on curated
TCR--peptide--HLA triples with a maximum-likelihood objective and compared
against ANN, GPTCR, LSTM, and VAE baselines. Across multiple neoantigen
benchmarks, PhysicoGPTCR substantially improves edit-distance, similarity, and
longest-common-subsequence scores, while populating a broader region of
sequence space. Blind in-silico docking and structural modelling further reveal
a higher proportion of binding-competent clones than the strongest baseline,
validating the benefit of explicit context conditioning and physicochemical
awareness. Experimental results demonstrate that dual-conditioned,
physics-grounded generative modelling enables end-to-end design of functional
TCR candidates, reducing the discovery timeline from months to minutes without
sacrificing wet-lab verifiability.

### 6. [Distributional Semantics Tracing: A Framework for Explaining Hallucinations in Large Language Models](http://arxiv.org/pdf/2510.06107v1)

Authors: Gagan Bhatia, Somayajulu G Sripada, Kevin Allan, Jacobo Azcona

Large Language Models (LLMs) are prone to hallucination, the generation of
plausible yet factually incorrect statements. This work investigates the
intrinsic, architectural origins of this failure mode through three primary
contributions.First, to enable the reliable tracing of internal semantic
failures, we propose \textbf{Distributional Semantics Tracing (DST)}, a unified
framework that integrates established interpretability techniques to produce a
causal map of a model's reasoning, treating meaning as a function of context
(distributional semantics). Second, we pinpoint the model's layer at which a
hallucination becomes inevitable, identifying a specific \textbf{commitment
layer} where a model's internal representations irreversibly diverge from
factuality. Third, we identify the underlying mechanism for these failures. We
observe a conflict between distinct computational pathways, which we interpret
using the lens of dual-process theory: a fast, heuristic \textbf{associative
pathway} (akin to System 1) and a slow, deliberate \textbf{contextual pathway}
(akin to System 2), leading to predictable failure modes such as
\textit{Reasoning Shortcut Hijacks}. Our framework's ability to quantify the
coherence of the contextual pathway reveals a strong negative correlation
($\rho = -0.863$) with hallucination rates, implying that these failures are
predictable consequences of internal semantic weakness. The result is a
mechanistic account of how, when, and why hallucinations occur within the
Transformer architecture.

### Computational Geometry

### 1. [Algorithms and Lower Bounds for the Maximum Overlap of Two Polygons Under Translation](http://arxiv.org/pdf/2510.05896v1)

Authors: Mikkel Abrahamsen, Sujoy Bhore, Maike Buchin, Jacobus Conradi, Ce Jin, André Nusser, Carolin Rehs

A fundamental problem in shape matching and geometric similarity is computing
the maximum area overlap between two polygons under translation. For general
simple polygons, the best-known algorithm runs in $O((nm)^2 \log(nm))$ time
[Mount, Silverman, Wu 96], where $n$ and $m$ are the complexities of the input
polygons. In a recent breakthrough, Chan and Hair gave a linear-time algorithm
for the special case when both polygons are convex. A key challenge in
computational geometry is to design improved algorithms for other natural
classes of polygons. We address this by presenting an $O((nm)^{3/2}
\log(nm))$-time algorithm for the case when both polygons are orthogonal. This
is the first algorithm for polygon overlap on orthogonal polygons that is
faster than the almost 30 years old algorithm for simple polygons.
  Complementing our algorithmic contribution, we provide $k$-SUM lower bounds
for problems on simple polygons with only orthogonal and diagonal edges. First,
we establish that there is no algorithm for polygon overlap with running time
$O(\max(n^2,nm^2)^{1-\varepsilon})$, where $m\leq n$, unless the $k$-SUM
hypothesis fails. This matches the running time of our algorithm when $n=m$. We
use part of the above construction to also show a lower bound for the polygon
containment problem, a popular special case of the overlap problem. Concretely,
there is no algorithm for polygon containment with running time
$O(n^{2-\varepsilon})$ under the $3$-SUM hypothesis, even when the polygon to
be contained has $m=O(1)$ vertices. Our lower bound shows that polygon
containment for these types of polygons (i.e., with diagonal edges) is strictly
harder than for orthogonal polygons, and also strengthens the previously known
lower bounds for polygon containment. Furthermore, our lower bounds show
tightness of the algorithm of [Mount, Silverman, Wu 96] when $m=O(1)$.

### 2. [Minimal Unimodal Decomposition is NP-Hard on Graphs](http://arxiv.org/pdf/2510.05944v1)

Authors: Mishal Assif P K, Yuliy Baryshnikov

A function on a topological space is called unimodal if all of its
super-level sets are contractible. A minimal unimodal decomposition of a
function $f$ is the smallest number of unimodal functions that sum up to $f$.
The problem of decomposing a given density function into its minimal unimodal
components is fundamental in topological statistics. We show that finding a
minimal unimodal decomposition of an edge-linear function on a graph is
NP-hard. Given any $k \geq 2$, we establish the NP-hardness of finding a
unimodal decomposition consisting of $k$ unimodal functions. We also extend the
NP-hardness result to related variants of the problem, including restriction to
planar graphs, inapproximability results, and generalizations to higher
dimensions.

### Computation and Language

### 1. [Language Model as Planner and Formalizer under Constraints](http://arxiv.org/pdf/2510.05486v1)

Authors: Cassie Huang, Stuti Mohan, Ziyi Yang, Stefanie Tellex, Li Zhang

LLMs have been widely used in planning, either as planners to generate action
sequences end-to-end, or as formalizers to represent the planning domain and
problem in a formal language that can derive plans deterministically. However,
both lines of work rely on standard benchmarks that only include generic and
simplistic environmental specifications, leading to potential overestimation of
the planning ability of LLMs and safety concerns in downstream tasks. We bridge
this gap by augmenting widely used planning benchmarks with manually annotated,
fine-grained, and rich natural language constraints spanning four formally
defined categories. Over 4 state-of-the-art reasoning LLMs, 3 formal languages,
5 methods, and 4 datasets, we show that the introduction of constraints not
only consistently halves performance, but also significantly challenges
robustness to problem complexity and lexical shift.

### 2. [Prototype-Based Dynamic Steering for Large Language Models](http://arxiv.org/pdf/2510.05498v1)

Authors: Ceyhun Efe Kayan, Li Zhang

Despite impressive breadth, LLMs still rely on explicit reasoning
instructions or static, one-fits-all steering methods, leaving a gap for
adaptive, instruction-free reasoning amplification. We present Prototype-Based
Dynamic Steering (PDS), a test-time method that amplifies large language model
(LLM) reasoning without adding or altering instructions. We introduce
"reasoning prototypes" by clustering activation differences between
Chain-of-Thought (CoT) and neutral prompts. At inference, an input's hidden
state is projected onto these prototypes to form an instance-specific steering
vector. Evaluated on GSM8K, AQuA-RAT, and BIG-Bench tasks, PDS consistently
improves accuracy without fine-tuning or prompt engineering. Notably, the gains
persist even when CoT is explicitly suppressed to improve cost-efficiency,
indicating that the intervention strengthens latent reasoning processes rather
than inducing a superficial behavioral shift. These results position dynamic,
prototype-guided steering as a lightweight alternative to training-time
approaches for enhancing LLM reasoning.

### 3. [On the Role of Difficult Prompts in Self-Play Preference Optimization](http://arxiv.org/pdf/2510.05534v1)

Authors: Yao Xiao, Jung-jae Kim, Roy Ka-wei Lee, Lidong Bing

Self-play preference optimization has emerged as a prominent paradigm for
aligning large language models (LLMs). It typically involves a language model
to generate on-policy responses for prompts and a reward model (RM) to guide
the selection of chosen and rejected responses, which can be further trained
with direct preference optimization (DPO). However, the role of prompts remains
underexplored, despite being a core component in this pipeline. In this work,
we investigate how prompts of varying difficulty influence self-play preference
optimization. We first use the mean reward of $N$ sampled responses of a prompt
as a proxy for its difficulty. We find that difficult prompts exhibit
substantially inferior self-play optimization performance in comparison to easy
prompts for language models. Moreover, incorporating difficult prompts into
training fails to enhance overall performance and, in fact, leads to slight
degradation compared to training on easy prompts alone. We also observe that
the performance gap between difficult and easy prompts closes as the model
capacity increases, suggesting that difficulty interacts with the model
capacity. Building on these findings, we explore strategies to mitigate the
negative effect of difficult prompts on final performance. We demonstrate that
selectively removing an appropriate portion of challenging prompts enhances
overall self-play performance, while also reporting failed attempts and lessons
learned.

### 4. [Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations](http://arxiv.org/pdf/2510.05571v1)

Authors: Chengzhi Liu, Yuzhe Yang, Kaiwen Zhou, Zhen Zhang, Yue Fan, Yannan Xie, Peng Qi, Xin Eric Wang

The promotion of academic papers has become an important means of enhancing
research visibility. However, existing automated methods struggle limited
storytelling, insufficient aesthetic quality, and constrained self-adjustment,
making it difficult to achieve efficient and engaging dissemination. At the
heart of those challenges is a simple principle: \emph{there is no way to
improve it when you cannot evaluate it right}. To address this, we introduce
\textbf{EvoPresent}, a self-improvement agent framework that unifies coherent
narratives, aesthetic-aware designs, and realistic presentation delivery via
virtual characters. Central to EvoPresent is \textbf{PresAesth}, a multi-task
reinforcement learning (RL) aesthetic model that provides reliable aesthetic
scoring, defect adjustment, and comparative feedback, enabling iterative
self-improvement even under limited aesthetic training data. To systematically
evaluate the methods, we introduce \textbf{EvoPresent Benchmark}, a
comprehensive benchmark comprising: \textit{Presentation Generation Quality},
built on 650 top-tier AI conference papers with multimodal resources (slides,
videos and scripts) to assess both content and design; and \textit{Aesthetic
Awareness}, consisting of 2,000 slide pairs with varying aesthetic levels,
supporting joint training and evaluation on scoring, defect adjustment, and
comparison. Our findings highlight that (i) High-quality feedback is essential
for agent self-improvement, while initial capability alone does not guarantee
effective self-correction. (ii) Automated generation pipelines exhibit a
trade-off between visual design and content construction. (iii) Multi-task RL
training shows stronger generalization in aesthetic awareness tasks.

### 5. [Mission Impossible: Feedback-Guided Dynamic Interactive Planning for Improving Reasoning on LLMs](http://arxiv.org/pdf/2510.05577v1)

Authors: Dong Yan, Gaochen Wu, Bowen Zhou

Recent advancements in language agents have led to significant improvements
in multi-hop reasoning tasks. However, existing approaches often struggle with
handling open-domain problems, which require massive information retrieval due
to their reliance on a fixed sequence of actions. To address this, we propose
Feedback-Guided Dynamic Interactive Planning (FGDIP), a novel framework
tailored to enhance reasoning in LLMs by utilizing dynamic and adaptive
strategies for information exploration in open-domain multi-hop reasoning
tasks. Our approach begins by identifying key entities relevant to the problem,
which serve as the initial nodes in the reasoning process. From these initial
nodes, we then generate reasoning child nodes with the process being refined
through a combination of historical error analysis and real-time feedback,
which allows the framework to dynamically adjust and optimize its reasoning
strategies. By integrating depth-first search with an innovative node
generation technique, our framework adapts based on both prior error paths and
concurrently generated nodes at the same hierarchical level. This dynamic
strategy effectively expands the search space while ensuring the reasoning
process systematically converges toward accurate solutions. Experimental
results show that FGDIP achieved up to 54.47% F1 score on the HotpotQA dataset
and 70.05% on the StrategyQA dataset, surpassing the best baseline by 5.03% and
7.25% respectively, highlighting its versatility and potential to enhance
language agents in multi-hop reasoning tasks.

### 6. [A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks](http://arxiv.org/pdf/2510.05608v1)

Authors: Shuzheng Si, Haozhe Zhao, Kangyang Luo, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun

Agents based on large language models (LLMs) struggle with brainless
trial-and-error and generating hallucinatory actions due to a lack of global
planning in long-horizon tasks. In this paper, we introduce a plan-and-execute
framework and propose EAGLET, an efficient and effective planner training
method to enhance the executor agent's planning abilities without human effort.
Specifically, we train a plug-and-play global planner through a two-step
process: we first synthesize high-quality plans from an advanced LLM using our
proposed homologous consensus filtering strategy, and apply fine-tuning as a
cold start. Moreover, we further improve the planner with a rule-based
reinforcement learning stage using a novel executor capability gain reward,
ensuring it can handle task instructions of varying difficulty. Experiments on
three long-horizon agent tasks show that executor agents equipped with our
planner outperform existing methods, achieving new state-of-the-art
performance. Meanwhile, EAGLET reduces training costs by 8x compared to
RL-based baselines, and it does not require manual effort or extra training
data, offering an efficient and effective solution.

### 7. [DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision](http://arxiv.org/pdf/2510.05691v1)

Authors: Yongqi Leng, Yikun Lei, Xikai Liu, Meizhi Zhong, Bojian Xiong, Yurong Zhang, Yan Gao, Yi Wu, Yao Hu, Deyi Xiong

Agentic Retrieval-Augmented Generation (Agentic RAG) enhances the processing
capability for complex tasks through dynamic retrieval and adaptive workflows.
Recent advances (e.g., Search-R1) have shown that outcome-supervised
reinforcement learning demonstrate strong performance. However, this approach
still suffers from inefficient exploration, sparse reward signals, and
ambiguous global reward feedback. To address these challenges, we propose
DecEx-RAG, which models RAG as a Markov Decision Process (MDP) incorporating
decision-making and execution, while introducing an efficient pruning strategy
to optimize data expansion. Through comprehensive process-level policy
optimization, DecEx-RAG significantly enhances the autonomous task
decomposition, dynamic retrieval, and high-quality answer generation
capabilities of large language models (LLMs). Experiments show that DecEx-RAG
achieves an average absolute performance improvement of $6.2\%$ across six
datasets, significantly outperforming existing baselines. Moreover, the pruning
strategy improves data construction efficiency by nearly $6 \times$, providing
an efficient solution for process-supervised RAG training. The code is
available at https://github.com/sdsxdxl/DecEx-RAG.

### 8. [Diversity Is All You Need for Contrastive Learning: Spectral Bounds on Gradient Magnitudes](http://arxiv.org/pdf/2510.05767v1)

Authors: Peter Ochieng

We derive non-asymptotic spectral bands that bound the squared InfoNCE
gradient norm via alignment, temperature, and batch spectrum, recovering the
\(1/\tau^{2}\) law and closely tracking batch-mean gradients on synthetic data
and ImageNet. Using effective rank \(R_{\mathrm{eff}}\) as an anisotropy proxy,
we design spectrum-aware batch selection, including a fast greedy builder. On
ImageNet-100, Greedy-64 cuts time-to-67.5\% top-1 by 15\% vs.\ random (24\%
vs.\ Pool--P3) at equal accuracy; CIFAR-10 shows similar gains. In-batch
whitening promotes isotropy and reduces 50-step gradient variance by
\(1.37\times\), matching our theoretical upper bound.

### 9. [Mixture of Neuron Experts](http://arxiv.org/pdf/2510.05781v1)

Authors: Runxi Cheng, Yuchen Guan, Yucheng Ding, Qingguo Hu, Yongxian Wei, Chun Yuan, Yelong Shen, Weizhu Chen, Yeyun Gong

In this work, we first explore whether the parameters activated by the MoE
layer remain highly sparse at inference. We perform a sparsification study on
several representative MoE models. For each expert, we rank parameters by the
magnitude of their activations from the gate projection and progressively prune
the activated subset. Pruning up to 60% of parameters within that subset causes
only negligible task-performance degradation; substantial drops occur only
after more than 90% are removed. We further decompose experts into
neuron-granular MoE and visualize their activation values, finding that most
neuron activations are near zero. This observation motivates us to select only
high-activation neuron experts during pretraining. Based on this insight, we
propose Mixture of Neuron Experts (MoNE). MoNE achieves neuron-granular expert
selection by only applying a simple top-k selection within each expert, incurs
negligible latency, and requires no additional routing parameters or
inter-expert communication. Extensive experiments demonstrate that MoNE matches
traditional MoE performance while activating only 50% of the MoE-layer
parameters, and it consistently outperforms traditional MoE when compared at
equal numbers of activated parameters. These results suggest that MoNE is a
practical approach to improving parameter utilization and inference efficiency
in MoE-like models.

### 10. [EEPO: Exploration-Enhanced Policy Optimization via Sample-Then-Forget](http://arxiv.org/pdf/2510.05837v1)

Authors: Liang Chen, Xueting Han, Qizhou Wang, Bo Han, Jing Bai, Hinrich Schutze, Kam-Fai Wong

Balancing exploration and exploitation remains a central challenge in
reinforcement learning with verifiable rewards (RLVR) for large language models
(LLMs). Current RLVR methods often overemphasize exploitation, leading to
entropy collapse, diminished exploratory capacity, and ultimately limited
performance gains. Although techniques that increase policy stochasticity can
promote exploration, they frequently fail to escape dominant behavioral modes.
This creates a self-reinforcing loop-repeatedly sampling and rewarding dominant
modes-that further erodes exploration. We introduce Exploration-Enhanced Policy
Optimization (EEPO), a framework that promotes exploration via two-stage
rollouts with adaptive unlearning. In the first stage, the model generates half
of the trajectories; it then undergoes a lightweight unlearning step to
temporarily suppress these sampled responses, forcing the second stage to
explore different regions of the output space. This sample-then-forget
mechanism disrupts the self-reinforcing loop and promotes wider exploration
during rollouts. Across five reasoning benchmarks, EEPO outperforms GRPO,
achieving average relative gains of 24.3% on Qwen2.5-3B, 33.0% on
Llama3.2-3B-Instruct, and 10.4% on Qwen3-8B-Base.

### Cryptography and Security

### 1. [New Insights into Involutory and Orthogonal MDS Matrices](http://arxiv.org/pdf/2510.05766v1)

Authors: Yogesh Kumar, Susanta Samanta, Atul Gaur

MDS matrices play a critical role in the design of diffusion layers for block
ciphers and hash functions due to their optimal branch number. Involutory and
orthogonal MDS matrices offer additional benefits by allowing identical or
nearly identical circuitry for both encryption and decryption, leading to
equivalent implementation costs for both processes. These properties have been
further generalized through the notions of semi-involutory and semi-orthogonal
matrices. Specifically, we establish nontrivial interconnections between
semi-involutory and involutory matrices, as well as between semi-orthogonal and
orthogonal matrices. Exploiting these relationships, we show that the number of
semi-involutory MDS matrices can be directly derived from the number of
involutory MDS matrices, and vice versa. A similar correspondence holds for
semi-orthogonal and orthogonal MDS matrices. We also examine the intersection
of these classes and show that the number of $3 \times 3$ MDS matrices that are
both semi-involutory and semi-orthogonal coincides with the number of
semi-involutory MDS matrices over $\mathbb{F}_{2^m}$. Furthermore, we derive
the general structure of orthogonal matrices of arbitrary order $n$ over
$\mathbb{F}_{2^m}$. Based on this generic form, we provide a closed-form
expression for enumerating all $3 \times 3$ orthogonal MDS matrices over
$\mathbb{F}_{2^m}$. Finally, leveraging the aforementioned interconnections, we
present explicit formulas for counting $3 \times 3$ semi-involutory MDS
matrices and semi-orthogonal MDS matrices.

### 2. [Privacy-Preserving On-chain Permissioning for KYC-Compliant Decentralized Applications](http://arxiv.org/pdf/2510.05807v1)

Authors: Fabian Piper, Karl Wolf, Jonathan Heiss

Decentralized applications (dApps) in Decentralized Finance (DeFi) face a
fundamental tension between regulatory compliance requirements like Know Your
Customer (KYC) and maintaining decentralization and privacy. Existing
permissioned DeFi solutions often fail to adequately protect private attributes
of dApp users and introduce implicit trust assumptions, undermining the
blockchain's decentralization. Addressing these limitations, this paper
presents a novel synthesis of Self-Sovereign Identity (SSI), Zero-Knowledge
Proofs (ZKPs), and Attribute-Based Access Control to enable privacy-preserving
on-chain permissioning based on decentralized policy decisions. We provide a
comprehensive framework for permissioned dApps that aligns decentralized trust,
privacy, and transparency, harmonizing blockchain principles with regulatory
compliance. Our framework supports multiple proof types (equality, range,
membership, and time-dependent) with efficient proof generation through a
commit-and-prove scheme that moves credential authenticity verification outside
the ZKP circuit. Experimental evaluation of our KYC-compliant DeFi
implementation shows considerable performance improvement for different proof
types compared to baseline approaches. We advance the state-of-the-art through
a holistic approach, flexible proof mechanisms addressing diverse real-world
requirements, and optimized proof generation enabling practical deployment.

### 3. [Enhancing Automotive Security with a Hybrid Approach towards Universal Intrusion Detection System](http://arxiv.org/pdf/2510.05824v1)

Authors: Md Rezanur Islam, Mahdi Sahlabadi, Keunkyoung Kim, Kangbin Yim

Security measures are essential in the automotive industry to detect
intrusions in-vehicle networks. However, developing a one-size-fits-all
Intrusion Detection System (IDS) is challenging because each vehicle has unique
data profiles. This is due to the complex and dynamic nature of the data
generated by vehicles regarding their model, driving style, test environment,
and firmware update. To address this issue, a universal IDS has been developed
that can be applied to all types of vehicles without the need for
customization. Unlike conventional IDSs, the universal IDS can adapt to
evolving data security issues resulting from firmware updates. In this study, a
new hybrid approach has been developed, combining Pearson correlation with deep
learning techniques. This approach has been tested using data obtained from
four distinct mechanical and electronic vehicles, including Tesla, Sonata, and
two Kia models. The data has been combined into two frequency datasets, and
wavelet transformation has been employed to convert them into the frequency
domain, enhancing generalizability. Additionally, a statistical method based on
independent rule-based systems using Pearson correlation has been utilized to
improve system performance. The system has been compared with eight different
IDSs, three of which utilize the universal approach, while the remaining five
are based on conventional techniques. The accuracy of each system has been
evaluated through benchmarking, and the results demonstrate that the hybrid
system effectively detects intrusions in various vehicle models.

### 4. [Fairness in Token Delegation: Mitigating Voting Power Concentration in DAOs](http://arxiv.org/pdf/2510.05830v1)

Authors: Johnnatan Messias, Ayae Ide

Decentralized Autonomous Organizations (DAOs) aim to enable participatory
governance, but in practice face challenges of voter apathy, concentration of
voting power, and misaligned delegation. Existing delegation mechanisms often
reinforce visibility biases, where a small set of highly ranked delegates
accumulate disproportionate influence regardless of their alignment with the
broader community. In this paper, we conduct an empirical study of delegation
in DAO governance, combining on-chain data from five major protocols with
off-chain discussions from 14 DAO forums. We develop a methodology to link
forum participants to on-chain addresses, extract governance interests using
large language models, and compare these interests against delegates'
historical behavior. Our analysis reveals that delegations are frequently
misaligned with token holders' expressed priorities and that current
ranking-based interfaces exacerbate power concentration. We argue that
incorporating interest alignment into delegation processes could mitigate these
imbalances and improve the representativeness of DAO decision-making.

### 5. [PhishSSL: Self-Supervised Contrastive Learning for Phishing Website Detection](http://arxiv.org/pdf/2510.05900v1)

Authors: Wenhao Li, Selvakumar Manickam, Yung-Wey Chong, Shankar Karuppayah, Priyadarsi Nanda, Binyong Li

Phishing websites remain a persistent cybersecurity threat by mimicking
legitimate sites to steal sensitive user information. Existing machine
learning-based detection methods often rely on supervised learning with labeled
data, which not only incurs substantial annotation costs but also limits
adaptability to novel attack patterns. To address these challenges, we propose
PhishSSL, a self-supervised contrastive learning framework that eliminates the
need for labeled phishing data during training. PhishSSL combines hybrid
tabular augmentation with adaptive feature attention to produce semantically
consistent views and emphasize discriminative attributes. We evaluate PhishSSL
on three phishing datasets with distinct feature compositions. Across all
datasets, PhishSSL consistently outperforms unsupervised and self-supervised
baselines, while ablation studies confirm the contribution of each component.
Moreover, PhishSSL maintains robust performance despite the diversity of
feature sets, highlighting its strong generalization and transferability. These
results demonstrate that PhishSSL offers a promising solution for phishing
website detection, particularly effective against evolving threats in dynamic
Web environments.

### 6. [Power Mechanism: Private Tabular Representation Release for Model Agnostic Consumption](http://arxiv.org/pdf/2510.05581v1)

Authors: Praneeth Vepakomma, Kaustubh Ponkshe

Traditional collaborative learning approaches are based on sharing of model
weights between clients and a server. However, there are advantages to resource
efficiency through schemes based on sharing of embeddings (activations) created
from the data. Several differentially private methods were developed for
sharing of weights while such mechanisms do not exist so far for sharing of
embeddings. We propose Ours to learn a privacy encoding network in conjunction
with a small utility generation network such that the final embeddings
generated from it are equipped with formal differential privacy guarantees.
These privatized embeddings are then shared with a more powerful server, that
learns a post-processing that results in a higher accuracy for machine learning
tasks. We show that our co-design of collaborative and private learning results
in requiring only one round of privatized communication and lesser compute on
the client than traditional methods. The privatized embeddings that we share
from the client are agnostic to the type of model (deep learning, random
forests or XGBoost) used on the server in order to process these activations to
complete a task.

### 7. [AutoPentester: An LLM Agent-based Framework for Automated Pentesting](http://arxiv.org/pdf/2510.05605v1)

Authors: Yasod Ginige, Akila Niroshan, Sajal Jain, Suranga Seneviratne

Penetration testing and vulnerability assessment are essential industry
practices for safeguarding computer systems. As cyber threats grow in scale and
complexity, the demand for pentesting has surged, surpassing the capacity of
human professionals to meet it effectively. With advances in AI, particularly
Large Language Models (LLMs), there have been attempts to automate the
pentesting process. However, existing tools such as PentestGPT are still
semi-manual, requiring significant professional human interaction to conduct
pentests. To this end, we propose a novel LLM agent-based framework,
AutoPentester, which automates the pentesting process. Given a target IP,
AutoPentester automatically conducts pentesting steps using common security
tools in an iterative process. It can dynamically generate attack strategies
based on the tool outputs from the previous iteration, mimicking the human
pentester approach. We evaluate AutoPentester using Hack The Box and
custom-made VMs, comparing the results with the state-of-the-art PentestGPT.
Results show that AutoPentester achieves a 27.0% better subtask completion rate
and 39.5% more vulnerability coverage with fewer steps. Most importantly, it
requires significantly fewer human interactions and interventions compared to
PentestGPT. Furthermore, we recruit a group of security industry professional
volunteers for a user survey and perform a qualitative analysis to evaluate
AutoPentester against industry practices and compare it with PentestGPT. On
average, AutoPentester received a score of 3.93 out of 5 based on user reviews,
which was 19.8% higher than PentestGPT.

### 8. [Membership Inference Attacks on Tokenizers of Large Language Models](http://arxiv.org/pdf/2510.05699v1)

Authors: Meng Tong, Yuntao Du, Kejiang Chen, Weiming Zhang, Ninghui Li

Membership inference attacks (MIAs) are widely used to assess the privacy
risks associated with machine learning models. However, when these attacks are
applied to pre-trained large language models (LLMs), they encounter significant
challenges, including mislabeled samples, distribution shifts, and
discrepancies in model size between experimental and real-world settings. To
address these limitations, we introduce tokenizers as a new attack vector for
membership inference. Specifically, a tokenizer converts raw text into tokens
for LLMs. Unlike full models, tokenizers can be efficiently trained from
scratch, thereby avoiding the aforementioned challenges. In addition, the
tokenizer's training data is typically representative of the data used to
pre-train LLMs. Despite these advantages, the potential of tokenizers as an
attack vector remains unexplored. To this end, we present the first study on
membership leakage through tokenizers and explore five attack methods to infer
dataset membership. Extensive experiments on millions of Internet samples
reveal the vulnerabilities in the tokenizers of state-of-the-art LLMs. To
mitigate this emerging risk, we further propose an adaptive defense. Our
findings highlight tokenizers as an overlooked yet critical privacy threat,
underscoring the urgent need for privacy-preserving mechanisms specifically
designed for them.

### 9. [Empirical Comparison of Membership Inference Attacks in Deep Transfer Learning](http://arxiv.org/pdf/2510.05753v1)

Authors: Yuxuan Bai, Gauri Pradhan, Marlon Tobaben, Antti Honkela

With the emergence of powerful large-scale foundation models, the training
paradigm is increasingly shifting from from-scratch training to transfer
learning. This enables high utility training with small, domain-specific
datasets typical in sensitive applications.Membership inference attacks (MIAs)
provide an empirical estimate of the privacy leakage by machine learning
models. Yet, prior assessments of MIAs against models fine-tuned with transfer
learning rely on a small subset of possible attacks. We address this by
comparing performance of diverse MIAs in transfer learning settings to help
practitioners identify the most efficient attacks for privacy risk evaluation.
We find that attack efficacy decreases with the increase in training data for
score-based MIAs. We find that there is no one MIA which captures all privacy
risks in models trained with transfer learning. While the Likelihood Ratio
Attack (LiRA) demonstrates superior performance across most experimental
scenarios, the Inverse Hessian Attack (IHA) proves to be more effective against
models fine-tuned on PatchCamelyon dataset in high data regime.

### 10. [SBOMproof: Beyond Alleged SBOM Compliance for Supply Chain Security of Container Images](http://arxiv.org/pdf/2510.05798v1)

Authors: Jacopo Bufalino, Mario Di Francesco, Agathe Blaise, Stefano Secci

Supply chain security is extremely important for modern applications running
at scale in the cloud. In fact, they involve a large number of heterogeneous
microservices that also include third-party software. As a result, security
vulnerabilities are hard to identify and mitigate before they start being
actively exploited by attackers. For this reason, governments have recently
introduced cybersecurity regulations that require vendors to share a software
bill of material (SBOM) with end users or regulators. An SBOM can be employed
to identify the security vulnerabilities of a software component even without
access to its source code, as long as it is accurate and interoperable across
different tools. This work evaluates this issue through a comprehensive study
of tools for SBOM generation and vulnerability scanning, including both
open-source software and cloud services from major providers. We specifically
target software containers and focus on operating system packages in Linux
distributions that are widely used as base images due to their far-reaching
security impact. Our findings show that the considered tools are largely
incompatible, leading to inaccurate reporting and a large amount of undetected
vulnerabilities. We uncover the SBOM confusion vulnerability, a byproduct of
such fragmented ecosystem, where inconsistent formats prevent reliable
vulnerability detection across tools.

### Computer Vision and Pattern Recognition

### 1. [ArchitectHead: Continuous Level of Detail Control for 3D Gaussian Head Avatars](http://arxiv.org/pdf/2510.05488v1)

Authors: Peizhi Yan, Rabab Ward, Qiang Tang, Shan Du

3D Gaussian Splatting (3DGS) has enabled photorealistic and real-time
rendering of 3D head avatars. Existing 3DGS-based avatars typically rely on
tens of thousands of 3D Gaussian points (Gaussians), with the number of
Gaussians fixed after training. However, many practical applications require
adjustable levels of detail (LOD) to balance rendering efficiency and visual
quality. In this work, we propose "ArchitectHead", the first framework for
creating 3D Gaussian head avatars that support continuous control over LOD. Our
key idea is to parameterize the Gaussians in a 2D UV feature space and propose
a UV feature field composed of multi-level learnable feature maps to encode
their latent features. A lightweight neural network-based decoder then
transforms these latent features into 3D Gaussian attributes for rendering.
ArchitectHead controls the number of Gaussians by dynamically resampling
feature maps from the UV feature field at the desired resolutions. This method
enables efficient and continuous control of LOD without retraining.
Experimental results show that ArchitectHead achieves state-of-the-art (SOTA)
quality in self and cross-identity reenactment tasks at the highest LOD, while
maintaining near SOTA performance at lower LODs. At the lowest LOD, our method
uses only 6.2\% of the Gaussians while the quality degrades moderately (L1 Loss
+7.9\%, PSNR --0.97\%, SSIM --0.6\%, LPIPS Loss +24.1\%), and the rendering
speed nearly doubles.

### 2. [Human Action Recognition from Point Clouds over Time](http://arxiv.org/pdf/2510.05506v1)

Authors: James Dickens

Recent research into human action recognition (HAR) has focused predominantly
on skeletal action recognition and video-based methods. With the increasing
availability of consumer-grade depth sensors and Lidar instruments, there is a
growing opportunity to leverage dense 3D data for action recognition, to
develop a third way. This paper presents a novel approach for recognizing
actions from 3D videos by introducing a pipeline that segments human point
clouds from the background of a scene, tracks individuals over time, and
performs body part segmentation. The method supports point clouds from both
depth sensors and monocular depth estimation. At the core of the proposed HAR
framework is a novel backbone for 3D action recognition, which combines
point-based techniques with sparse convolutional networks applied to
voxel-mapped point cloud sequences. Experiments incorporate auxiliary point
features including surface normals, color, infrared intensity, and body part
parsing labels, to enhance recognition accuracy. Evaluation on the NTU RGB- D
120 dataset demonstrates that the method is competitive with existing skeletal
action recognition algorithms. Moreover, combining both sensor-based and
estimated depth inputs in an ensemble setup, this approach achieves 89.3%
accuracy when different human subjects are considered for training and testing,
outperforming previous point cloud action recognition methods.

### 3. [Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models](http://arxiv.org/pdf/2510.05509v1)

Authors: Shinnosuke Saito, Takashi Matsubara

Diffusion models are powerful deep generative models (DGMs) that generate
high-fidelity, diverse content. However, unlike classical DGMs, they lack an
explicit, tractable low-dimensional latent space that parameterizes the data
manifold. This absence limits manifold-aware analysis and operations, such as
interpolation and editing. Existing interpolation methods for diffusion models
typically follow paths through high-density regions, which are not necessarily
aligned with the data manifold and can yield perceptually unnatural
transitions. To exploit the data manifold learned by diffusion models, we
propose a novel Riemannian metric on the noise space, inspired by recent
findings that the Jacobian of the score function captures the tangent spaces to
the local data manifold. This metric encourages geodesics in the noise space to
stay within or run parallel to the learned data manifold. Experiments on image
interpolation show that our metric produces perceptually more natural and
faithful transitions than existing density-based and naive baselines.

### 4. [HoloScene: Simulation-Ready Interactive 3D Worlds from a Single Video](http://arxiv.org/pdf/2510.05560v1)

Authors: Hongchi Xia, Chih-Hao Lin, Hao-Yu Hsu, Quentin Leboutet, Katelyn Gao, Michael Paulitsch, Benjamin Ummenhofer, Shenlong Wang

Digitizing the physical world into accurate simulation-ready virtual
environments offers significant opportunities in a variety of fields such as
augmented and virtual reality, gaming, and robotics. However, current 3D
reconstruction and scene-understanding methods commonly fall short in one or
more critical aspects, such as geometry completeness, object interactivity,
physical plausibility, photorealistic rendering, or realistic physical
properties for reliable dynamic simulation. To address these limitations, we
introduce HoloScene, a novel interactive 3D reconstruction framework that
simultaneously achieves these requirements. HoloScene leverages a comprehensive
interactive scene-graph representation, encoding object geometry, appearance,
and physical properties alongside hierarchical and inter-object relationships.
Reconstruction is formulated as an energy-based optimization problem,
integrating observational data, physical constraints, and generative priors
into a unified, coherent objective. Optimization is efficiently performed via a
hybrid approach combining sampling-based exploration with gradient-based
refinement. The resulting digital twins exhibit complete and precise geometry,
physical stability, and realistic rendering from novel viewpoints. Evaluations
conducted on multiple benchmark datasets demonstrate superior performance,
while practical use-cases in interactive gaming and real-time digital-twin
manipulation illustrate HoloScene's broad applicability and effectiveness.
Project page: https://xiahongchi.github.io/HoloScene.

### 5. [CalibCLIP: Contextual Calibration of Dominant Semantics for Text-Driven Image Retrieval](http://arxiv.org/pdf/2510.05586v1)

Authors: Bin Kang, Bin Chen, Junjie Wang, Yulin Li, Junzhi Zhao, Zhuotao Tian

Existing Visual Language Models (VLMs) suffer structural limitations where a
few low contribution tokens may excessively capture global semantics,
dominating the information aggregation process and suppressing the
discriminative features in text-driven image retrieval tasks. To address this,
we introduce \textbf{CalibCLIP}, a training-free method designed to calibrate
the suppressive effect of dominant tokens. Specifically, in the visual space,
we propose the Contrastive Visual Enhancer (CVE), which decouples visual
features into target and low information regions. Subsequently, it identifies
dominant tokens and dynamically suppresses their representations.In the textual
space, we introduce the Discriminative Concept Calibrator (DCC), which aims to
differentiate between general and discriminative concepts within the text
query. By mitigating the challenges posed by generic concepts and improving the
representations of discriminative concepts, DCC strengthens the differentiation
among similar samples. Finally, extensive experiments demonstrate consistent
improvements across seven benchmarks spanning three image retrieval tasks,
underscoring the effectiveness of CalibCLIP. Code is available at:
https://github.com/kangbin98/CalibCLIP

### 6. [Efficient Conditional Generation on Scale-based Visual Autoregressive Models](http://arxiv.org/pdf/2510.05610v1)

Authors: Jiaqi Liu, Tao Huang, Chang Xu

Recent advances in autoregressive (AR) models have demonstrated their
potential to rival diffusion models in image synthesis. However, for complex
spatially-conditioned generation, current AR approaches rely on fine-tuning the
pre-trained model, leading to significant training costs. In this paper, we
propose the Efficient Control Model (ECM), a plug-and-play framework featuring
a lightweight control module that introduces control signals via a distributed
architecture. This architecture consists of context-aware attention layers that
refine conditional features using real-time generated tokens, and a shared
gated feed-forward network (FFN) designed to maximize the utilization of its
limited capacity and ensure coherent control feature learning. Furthermore,
recognizing the critical role of early-stage generation in determining semantic
structure, we introduce an early-centric sampling strategy that prioritizes
learning early control sequences. This approach reduces computational cost by
lowering the number of training tokens per iteration, while a complementary
temperature scheduling during inference compensates for the resulting
insufficient training of late-stage tokens. Extensive experiments on
scale-based AR models validate that our method achieves high-fidelity and
diverse control over image generation, surpassing existing baselines while
significantly improving both training and inference efficiency.

### 7. [TFM Dataset: A Novel Multi-task Dataset and Integrated Pipeline for Automated Tear Film Break-Up Segmentation](http://arxiv.org/pdf/2510.05615v1)

Authors: Guangrong Wan, Jun liu, Tang tang, Lianghao Shi, Wenjun Luo, TingTing Xu

Tear film break-up (TFBU) analysis is critical for diagnosing dry eye
syndrome, but automated TFBU segmentation remains challenging due to the lack
of annotated datasets and integrated solutions. This paper introduces the Tear
Film Multi-task (TFM) Dataset, the first comprehensive dataset for multi-task
tear film analysis, comprising 15 high-resolution videos (totaling 6,247
frames) annotated with three vision tasks: frame-level classification ('clear',
'closed', 'broken', 'blur'), Placido Ring detection, and pixel-wise TFBU area
segmentation. Leveraging this dataset, we first propose TF-Net, a novel and
efficient baseline segmentation model. TF-Net incorporates a MobileOne-mini
backbone with re-parameterization techniques and an enhanced feature pyramid
network to achieve a favorable balance between accuracy and computational
efficiency for real-time clinical applications. We further establish benchmark
performance on the TFM segmentation subset by comparing TF-Net against several
state-of-the-art medical image segmentation models. Furthermore, we design
TF-Collab, a novel integrated real-time pipeline that synergistically leverages
models trained on all three tasks of the TFM dataset. By sequentially
orchestrating frame classification for BUT determination, pupil region
localization for input standardization, and TFBU segmentation, TF-Collab fully
automates the analysis. Experimental results demonstrate the effectiveness of
the proposed TF-Net and TF-Collab, providing a foundation for future research
in ocular surface diagnostics. Our code and the TFM datasets are available at
https://github.com/glory-wan/TF-Net

### 8. [Combined Hyperbolic and Euclidean Soft Triple Loss Beyond the Single Space Deep Metric Learning](http://arxiv.org/pdf/2510.05643v1)

Authors: Shozo Saeki, Minoru Kawahara, Hirohisa Aman

Deep metric learning (DML) aims to learn a neural network mapping data to an
embedding space, which can represent semantic similarity between data points.
Hyperbolic space is attractive for DML since it can represent richer
structures, such as tree structures. DML in hyperbolic space is based on
pair-based loss or unsupervised regularization loss. On the other hand,
supervised proxy-based losses in hyperbolic space have not been reported yet
due to some issues in applying proxy-based losses in a hyperbolic space.
However, proxy-based losses are attractive for large-scale datasets since they
have less training complexity. To address these, this paper proposes the
Combined Hyperbolic and Euclidean Soft Triple (CHEST) loss. CHEST loss is
composed of the proxy-based losses in hyperbolic and Euclidean spaces and the
regularization loss based on hyperbolic hierarchical clustering. We find that
the combination of hyperbolic and Euclidean spaces improves DML accuracy and
learning stability for both spaces. Finally, we evaluate the CHEST loss on four
benchmark datasets, achieving a new state-of-the-art performance.

### 9. [SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets](http://arxiv.org/pdf/2510.05652v1)

Authors: Manolis Mylonas, Charalampia Zerva, Evlampios Apostolidis, Vasileios Mezaris

In this work, we extend a recent method for script-driven video
summarization, originally considering just the visual content of the video, to
take into account the relevance of the user-provided script also with the
video's spoken content. In the proposed method, SD-MVSum, the dependence
between each considered pair of data modalities, i.e., script-video and
script-transcript, is modeled using a new weighted cross-modal attention
mechanism. This explicitly exploits the semantic similarity between the paired
modalities in order to promote the parts of the full-length video with the
highest relevance to the user-provided script. Furthermore, we extend two
large-scale datasets for video summarization (S-VideoXum, MrHiSum), to make
them suitable for training and evaluation of script-driven multimodal video
summarization methods. Experimental comparisons document the competitiveness of
our SD-MVSum method against other SOTA approaches for script-driven and generic
video summarization. Our new method and extended datasets are available at:
https://github.com/IDT-ITI/SD-MVSum.

### 10. [A Hierarchical Geometry-guided Transformer for Histological Subtyping of Primary Liver Cancer](http://arxiv.org/pdf/2510.05657v1)

Authors: Anwen Lu, Mingxin Liu, Yiping Jiao, Hongyi Gong, Geyang Xu, Jun Chen, Jun Xu

Primary liver malignancies are widely recognized as the most heterogeneous
and prognostically diverse cancers of the digestive system. Among these,
hepatocellular carcinoma (HCC) and intrahepatic cholangiocarcinoma (ICC) emerge
as the two principal histological subtypes, demonstrating significantly greater
complexity in tissue morphology and cellular architecture than other common
tumors. The intricate representation of features in Whole Slide Images (WSIs)
encompasses abundant crucial information for liver cancer histological
subtyping, regarding hierarchical pyramid structure, tumor microenvironment
(TME), and geometric representation. However, recent approaches have not
adequately exploited these indispensable effective descriptors, resulting in a
limited understanding of histological representation and suboptimal subtyping
performance. To mitigate these limitations, ARGUS is proposed to advance
histological subtyping in liver cancer by capturing the macro-meso-micro
hierarchical information within the TME. Specifically, we first construct a
micro-geometry feature to represent fine-grained cell-level pattern via a
geometric structure across nuclei, thereby providing a more refined and precise
perspective for delineating pathological images. Then, a Hierarchical
Field-of-Views (FoVs) Alignment module is designed to model macro- and
meso-level hierarchical interactions inherent in WSIs. Finally, the augmented
micro-geometry and FoVs features are fused into a joint representation via
present Geometry Prior Guided Fusion strategy for modeling holistic phenotype
interactions. Extensive experiments on public and private cohorts demonstrate
that our ARGUS achieves state-of-the-art (SOTA) performance in histological
subtyping of liver cancer, which provide an effective diagnostic tool for
primary liver malignancies in clinical practice.

### Computers and Society

### 1. [Evaluating LLM Safety Across Child Development Stages: A Simulated Agent Approach](http://arxiv.org/pdf/2510.05484v1)

Authors: Abhejay Murali, Saleh Afroogh, Kevin Chen, David Atkinson, Amit Dhurandhar, Junfeng Jiao

Large Language Models (LLMs) are rapidly becoming part of tools used by
children; however, existing benchmarks fail to capture how these models manage
language, reasoning, and safety needs that are specific to various ages. We
present ChildSafe, a benchmark that evaluates LLM safety through simulated
child agents that embody four developmental stages. These agents, grounded in
developmental psychology, enable a systematic study of child safety without the
ethical implications of involving real children. ChildSafe assesses responses
across nine safety dimensions (including privacy, misinformation, and emotional
support) using age-weighted scoring in both sensitive and neutral contexts.
Multi-turn experiments with multiple LLMs uncover consistent vulnerabilities
that vary by simulated age, exposing shortcomings in existing alignment
practices. By releasing agent templates, evaluation protocols, and an
experimental corpus, we provide a reproducible framework for age-aware safety
research. We encourage the community to expand this work with real
child-centered data and studies, advancing the development of LLMs that are
genuinely safe and developmentally aligned.

### 2. [Assessing Human Rights Risks in AI: A Framework for Model Evaluation](http://arxiv.org/pdf/2510.05519v1)

Authors: Vyoma Raman, Camille Chabot, Betsy Popken

The Universal Declaration of Human Rights and other international agreements
outline numerous inalienable rights that apply across geopolitical boundaries.
As generative AI becomes increasingly prevalent, it poses risks to human rights
such as non-discrimination, health, and security, which are also central
concerns for AI researchers focused on fairness and safety. We contribute to
the field of algorithmic auditing by presenting a framework to computationally
assess human rights risk. Drawing on the UN Guiding Principles on Business and
Human Rights, we develop an approach to evaluating a model to make grounded
claims about the level of risk a model poses to particular human rights. Our
framework consists of three parts: selecting tasks that are likely to pose
human rights risks within a given context, designing metrics to measure the
scope, scale, and likelihood of potential risks from that task, and analyzing
rights with respect to the values of those metrics. Because a human rights
approach centers on real-world harms, it requires evaluating AI systems in the
specific contexts in which they are deployed. We present a case study of large
language models in political news journalism, demonstrating how our framework
helps to design an evaluation and benchmarking different models. We then
discuss the implications of the results for the rights of access to information
and freedom of thought and broader considerations for adopting this approach.

### 3. [Beyond Accessibility: How Intelligent Assistive Technologies Improve Activities of Daily Life for Visually Impaired People in South Africa](http://arxiv.org/pdf/2510.05998v1)

Authors: Ronaldo Nombakuse, Nils Messerschmidt, Pitso Tsibolane, Muhammad Irfan Khalid

Our study explores how intelligent assistive technologies (IATs) can enable
visually impaired people (VIPs) to overcome barriers to inclusion in a digital
society to ultimately improve their quality of life. Drawing on the Social
Model of Disability (SMD), which frames disability as a consequence of social
and institutional barriers rather than individual impairments, we employ
semi-structured interviews and an online qualitative survey with n=61 VIPs in
South Africa. Using descriptive statistics and Qualitative Comparative Analysis
(QCA), we uncover nine configurations, clustered along three broader
combinations of conditions, that support and hinder IAT-mediated inclusion.
Most notably, we identify that the autonomy of VIPs and the accessibility of
IATs are primary predictors of IAT's ability to achieve social participation.
Our findings contribute to Information Systems (IS) literature at the
intersection of technology and social participation. We further formulate
implications for research and policymakers to foster social inclusion of VIPs
in the Global South.

### 4. [A Possibility Frontier Approach to Diverse Talent Selection](http://arxiv.org/pdf/2510.06119v1)

Authors: Neil Natarajan, Kadeem Noray

Organizations (e.g., talent investment programs, schools, firms) are
perennially interested in selecting cohorts of talented people. And
organizations are increasingly interested in selecting diverse cohorts. Except
in trivial cases, measuring the tradeoff between cohort diversity and talent is
computationally difficult. Thus, organizations are presently unable to make
Pareto-efficient decisions about these tradeoffs. We introduce an algorithm
that approximates upper bounds on cohort talent and diversity. We call this
object the selection possibility frontier (SPF). We then use the SPF to assess
the efficiency of selection of a talent investment program. We show that, in
the 2021 and 2022 cycles, the program selected cohorts of finalists that could
have been better along both diversity and talent dimensions (i.e., considering
only these dimensions as we subsequently calculated them, they are
Pareto-inferior cohorts). But, when given access our approximation of the SPF
in the 2023 cycle, the program adjusted decisions and selected a cohort on the
SPF.

### 5. [EduVerse: A User-Defined Multi-Agent Simulation Space for Education Scenario](http://arxiv.org/pdf/2510.05650v1)

Authors: Yiping Ma, Shiyu Hu, Buyuan Zhu, Yipei Wang, Yaxuan Kang, Shiqing Liu, Kang Hao Cheong

Reproducing cognitive development, group interaction, and long-term evolution
in virtual classrooms remains a core challenge for educational AI, as real
classrooms integrate open-ended cognition, dynamic social interaction,
affective factors, and multi-session development rarely captured together.
Existing approaches mostly focus on short-term or single-agent settings,
limiting systematic study of classroom complexity and cross-task reuse. We
present EduVerse, the first user-defined multi-agent simulation space that
supports environment, agent, and session customization. A distinctive
human-in-the-loop interface further allows real users to join the space. Built
on a layered CIE (Cognition-Interaction-Evolution) architecture, EduVerse
ensures individual consistency, authentic interaction, and longitudinal
adaptation in cognition, emotion, and behavior-reproducing realistic classroom
dynamics with seamless human-agent integration. We validate EduVerse in
middle-school Chinese classes across three text genres, environments, and
multiple sessions. Results show: (1) Instructional alignment: simulated IRF
rates (0.28-0.64) closely match real classrooms (0.37-0.49), indicating
pedagogical realism; (2) Group interaction and role differentiation: network
density (0.27-0.40) with about one-third of peer links realized, while
human-agent tasks indicate a balance between individual variability and
instructional stability; (3) Cross-session evolution: the positive transition
rate R+ increase by 11.7% on average, capturing longitudinal shifts in
behavior, emotion, and cognition and revealing structured learning
trajectories. Overall, EduVerse balances realism, reproducibility, and
interpretability, providing a scalable platform for educational AI. The system
will be open-sourced to foster cross-disciplinary research.

### 6. [Artificially intelligent agents in the social and behavioral sciences: A history and outlook](http://arxiv.org/pdf/2510.05743v1)

Authors: Petter Holme, Milena Tsvetkova

We review the historical development and current trends of artificially
intelligent agents (agentic AI) in the social and behavioral sciences: from the
first programmable computers, and social simulations soon thereafter, to
today's experiments with large language models. This overview emphasizes the
role of AI in the scientific process and the changes brought about, both
through technological advancements and the broader evolution of science from
around 1950 to the present. Some of the specific points we cover include: the
challenges of presenting the first social simulation studies to a world unaware
of computers, the rise of social systems science, intelligent game theoretic
agents, the age of big data and the epistemic upheaval in its wake, and the
current enthusiasm around applications of generative AI, and many other topics.
A pervasive theme is how deeply entwined we are with the technologies we use to
understand ourselves.

### 7. [The Five Safes as a Privacy Context](http://arxiv.org/pdf/2510.05803v1)

Authors: James Bailie, Ruobin Gong

The Five Safes is a framework used by national statistical offices (NSO) for
assessing and managing the disclosure risk of data sharing. This paper makes
two points: Firstly, the Five Safes can be understood as a specialization of a
broader concept $\unicode{x2013}$ contextual integrity $\unicode{x2013}$ to the
situation of statistical dissemination by an NSO. We demonstrate this by
mapping the five parameters of contextual integrity onto the five dimensions of
the Five Safes. Secondly, the Five Safes contextualizes narrow, technical
notions of privacy within a holistic risk assessment. We demonstrate this with
the example of differential privacy (DP). This contextualization allows NSOs to
place DP within their Five Safes toolkit while also guiding the design of DP
implementations within the broader privacy context, as delineated by both their
regulation and the relevant social norms.

### 8. [Evaluating the Sensitivity of LLMs to Harmful Contents in Long Input](http://arxiv.org/pdf/2510.05864v1)

Authors: Faeze Ghorbanpour, Alexander Fraser

Large language models (LLMs) increasingly support applications that rely on
extended context, from document processing to retrieval-augmented generation.
While their long-context capabilities are well studied for reasoning and
retrieval, little is known about their behavior in safety-critical scenarios.
We evaluate LLMs' sensitivity to harmful content under extended context,
varying type (explicit vs. implicit), position (beginning, middle, end),
prevalence (0.01-0.50 of the prompt), and context length (600-6000 tokens).
Across harmful content categories such as toxic, offensive, and hate speech,
with LLaMA-3, Qwen-2.5, and Mistral, we observe similar patterns: performance
peaks at moderate harmful prevalence (0.25) but declines when content is very
sparse or dominant; recall decreases with increasing context length; harmful
sentences at the beginning are generally detected more reliably; and explicit
content is more consistently recognized than implicit. These findings provide
the first systematic view of how LLMs prioritize and calibrate harmful content
in long contexts, highlighting both their emerging strengths and the challenges
that remain for safety-critical use.

### 9. [Hire Your Anthropologist! Rethinking Culture Benchmarks Through an Anthropological Lens](http://arxiv.org/pdf/2510.05931v1)

Authors: Mai AlKhamissi, Yunze Xiao, Badr AlKhamissi, Mona Diab

Cultural evaluation of large language models has become increasingly
important, yet current benchmarks often reduce culture to static facts or
homogeneous values. This view conflicts with anthropological accounts that
emphasize culture as dynamic, historically situated, and enacted in practice.
To analyze this gap, we introduce a four-part framework that categorizes how
benchmarks frame culture, such as knowledge, preference, performance, or bias.
Using this lens, we qualitatively examine 20 cultural benchmarks and identify
six recurring methodological issues, including treating countries as cultures,
overlooking within-culture diversity, and relying on oversimplified survey
formats. Drawing on established anthropological methods, we propose concrete
improvements: incorporating real-world narratives and scenarios, involving
cultural communities in design and validation, and evaluating models in context
rather than isolation. Our aim is to guide the development of cultural
benchmarks that go beyond static recall tasks and more accurately capture the
responses of the models to complex cultural situations.

### 10. [InstaGeo: Compute-Efficient Geospatial Machine Learning from Data to Deployment](http://arxiv.org/pdf/2510.05617v1)

Authors: Ibrahim Salihu Yusuf, Iffanice Houndayi, Rym Oualha, Mohamed Aziz Cherif, Kobby Panford-Quainoo, Arnu Pretorius

Open-access multispectral imagery from missions like Landsat 8-9 and
Sentinel-2 has fueled the development of geospatial foundation models (GFMs)
for humanitarian and environmental applications. Yet, their deployment remains
limited by (i) the absence of automated geospatial data pipelines and (ii) the
large size of fine-tuned models. Existing GFMs lack workflows for processing
raw satellite imagery, and downstream adaptations often retain the full
complexity of the original encoder.
  We present InstaGeo, an open-source, end-to-end framework that addresses
these challenges by integrating: (1) automated data curation to transform raw
imagery into model-ready datasets; (2) task-specific model distillation to
derive compact, compute-efficient models; and (3) seamless deployment as
interactive web-map applications. Using InstaGeo, we reproduced datasets from
three published studies and trained models with marginal mIoU differences of
-0.73 pp for flood mapping, -0.20 pp for crop segmentation, and +1.79 pp for
desert locust prediction. The distilled models are up to 8x smaller than
standard fine-tuned counterparts, reducing FLOPs and CO2 emissions with minimal
accuracy loss.
  Leveraging InstaGeo's streamlined data pipeline, we also curated a larger
crop segmentation dataset, achieving a state-of-the-art mIoU of 60.65%, a 12 pp
improvement over prior baselines. Moreover, InstaGeo enables users to progress
from raw data to model deployment within a single working day.
  By unifying data preparation, model compression, and deployment, InstaGeo
transforms research-grade GFMs into practical, low-carbon tools for real-time,
large-scale Earth observation. This approach shifts geospatial AI toward data
quality and application-driven innovation. Source code, datasets, and model
checkpoints are available at:
https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git

### Databases

### 1. [Redefining Cost Estimation in Database Systems: The Role of Execution Plan Features and Machine Learning](http://arxiv.org/pdf/2510.05612v1)

Authors: Utsav Pathak, Amit Mankodi

Accurate query runtime prediction is a critical component of effective query
optimization in modern database systems. Traditional cost models, such as those
used in PostgreSQL, rely on static heuristics that often fail to reflect actual
query performance under complex and evolving workloads. This remains an active
area of research, with recent work exploring machine learning techniques to
replace or augment traditional cost estimators. In this paper, we present a
machine learning-based framework for predicting SQL query runtimes using
execution plan features extracted from PostgreSQL. Our approach integrates
scalar and structural features from execution plans and semantic
representations of SQL queries to train predictive models. We construct an
automated pipeline for data collection and feature extraction using
parameterized TPC-H queries, enabling systematic evaluation of multiple
modeling techniques. Unlike prior efforts that focus either on cardinality
estimation or on synthetic cost metrics, we model the actual runtimes using
fine-grained plan statistics and query embeddings derived from execution
traces, to improve the model accuracy. We compare baseline regressors, a
refined XGBoost model, and a sequential LSTM-based model to assess their
effectiveness in runtime prediction. Our dataset includes over 1000 queries
generated from TPC-H query templates executed in PostgreSQL with EXPLAIN
ANALYZE. Experimental results show that the XGBoost model significantly
outperforms others, achieving a mean squared error of 0.3002 and prediction
accuracy within 10% of the true runtime in over 65% of cases. The findings
highlight the potential of tree-based learning combined with execution plan
features for improving cost estimation in query optimizers.

### 2. [Improving Clinical Dataset Condensation with Mode Connectivity-based Trajectory Surrogates](http://arxiv.org/pdf/2510.05805v1)

Authors: Pafue Christy Nganjimi, Andrew Soltan, Danielle Belgrave, Lei Clifton, David A. Clifton, Anshul Thakur

Dataset condensation (DC) enables the creation of compact, privacy-preserving
synthetic datasets that can match the utility of real patient records,
supporting democratised access to highly regulated clinical data for developing
downstream clinical models. State-of-the-art DC methods supervise synthetic
data by aligning the training dynamics of models trained on real and those
trained on synthetic data, typically using full stochastic gradient descent
(SGD) trajectories as alignment targets; however, these trajectories are often
noisy, high-curvature, and storage-intensive, leading to unstable gradients,
slow convergence, and substantial memory overhead. We address these limitations
by replacing full SGD trajectories with smooth, low-loss parametric surrogates,
specifically quadratic B\'ezier curves that connect the initial and final model
states from real training trajectories. These mode-connected paths provide
noise-free, low-curvature supervision signals that stabilise gradients,
accelerate convergence, and eliminate the need for dense trajectory storage. We
theoretically justify B\'ezier-mode connections as effective surrogates for SGD
paths and empirically show that the proposed method outperforms
state-of-the-art condensation approaches across five clinical datasets,
yielding condensed datasets that enable clinically effective model development.

### 3. [Speeding up SQL subqueries via decoupling of non-correlated predicate (extended version)](http://arxiv.org/pdf/2510.05907v1)

Authors: Dmitrii Radivonchik, Yakov Kuzin, Anton Chizhov, Dmitriy Shcheka, Mikhail Firsov, Kirill Smirnov, George Chernishev

In this paper, we discuss a novel technique for processing correlated
subqueries in SQL. The core idea is to isolate the non-correlated part of the
predicate and use it to reduce the number of evaluations of the correlated
part. We begin by providing an overview of several classes of queries that may
benefit from this technique. For each class, we propose a potential rewrite and
discuss the conditions under which it is advantageous. Next, we address the
evaluation aspects of the proposed rewrites: 1) we describe our approach to
adapting the block-based Volcano query processing model, and 2) we discuss the
benefits of implementing that technique within a position-enabled column-store
with late materialization support. Finally, we present a simple cost model that
allows estimation of the benefits of said rewrites.
  Our evaluation has a quantitative part and a qualitative part. The former
focuses on studying the impact of non-correlated predicate selectivity on our
technique. The latter identifies the limitations of our approach by comparing
it with alternative approaches available in existing systems. Overall,
experiments conducted using PosDB (a position-enabled column-store) and
PostgreSQL demonstrated that, under suitable conditions, our technique can
achieve a 5x improvement.

### Distributed, Parallel, and Cluster Computing

### 1. [A Review of Ontology-Driven Big Data Analytics in Healthcare: Challenges, Tools, and Applications](http://arxiv.org/pdf/2510.05738v1)

Authors: Ritesh Chandra, Sonali Agarwal, Navjot Singh, Sadhana Tiwari

Exponential growth in heterogeneous healthcare data arising from electronic
health records (EHRs), medical imaging, wearable sensors, and biomedical
research has accelerated the adoption of data lakes and centralized
architectures capable of handling the Volume, Variety, and Velocity of Big Data
for advanced analytics. However, without effective governance, these
repositories risk devolving into disorganized data swamps. Ontology-driven
semantic data management offers a robust solution by linking metadata to
healthcare knowledge graphs, thereby enhancing semantic interoperability,
improving data discoverability, and enabling expressive, domain-aware access.
This review adopts a systematic research strategy, formulating key research
questions and conducting a structured literature search across major academic
databases, with selected studies analyzed and classified into six categories of
ontology-driven healthcare analytics: (i) ontology-driven integration
frameworks, (ii) semantic modeling for metadata enrichment, (iii)
ontology-based data access (OBDA), (iv) basic semantic data management, (v)
ontology-based reasoning for decision support, and (vi) semantic annotation for
unstructured data. We further examine the integration of ontology technologies
with Big Data frameworks such as Hadoop, Spark, Kafka, and so on, highlighting
their combined potential to deliver scalable and intelligent healthcare
analytics. For each category, recent techniques, representative case studies,
technical and organizational challenges, and emerging trends such as artificial
intelligence, machine learning, the Internet of Things (IoT), and real-time
analytics are reviewed to guide the development of sustainable, interoperable,
and high-performance healthcare data ecosystems.

### 2. [Toward Systems Foundations for Agentic Exploration](http://arxiv.org/pdf/2510.05556v1)

Authors: Jiakai Xu, Tianle Zhou, Eugene Wu, Kostis Kaffes

Agentic exploration, letting LLM-powered agents branch, backtrack, and search
across many execution paths, demands systems support well beyond today's
pass-at-k resets. Our benchmark of six snapshot/restore mechanisms shows that
generic tools such as CRIU or container commits are not fast enough even in
isolated testbeds, and they crumble entirely in real deployments where agents
share files, sockets, and cloud APIs with other agents and human users. In this
talk, we pinpoint three open fundamental challenges: fork semantics, which
concerns how branches reveal or hide tentative updates; external side-effects,
where fork awareness must be added to services or their calls intercepted; and
native forking, which requires cloning databases and runtimes in microseconds
without bulk copying.

### 3. [When Does Global Attention Help? A Unified Empirical Study on Atomistic Graph Learning](http://arxiv.org/pdf/2510.05583v1)

Authors: Arindam Chowdhury, Massimiliano Lupo Pasini

Graph neural networks (GNNs) are widely used as surrogates for costly
experiments and first-principles simulations to study the behavior of compounds
at atomistic scale, and their architectural complexity is constantly increasing
to enable the modeling of complex physics. While most recent GNNs combine more
traditional message passing neural networks (MPNNs) layers to model short-range
interactions with more advanced graph transformers (GTs) with global attention
mechanisms to model long-range interactions, it is still unclear when global
attention mechanisms provide real benefits over well-tuned MPNN layers due to
inconsistent implementations, features, or hyperparameter tuning. We introduce
the first unified, reproducible benchmarking framework - built on HydraGNN -
that enables seamless switching among four controlled model classes: MPNN, MPNN
with chemistry/topology encoders, GPS-style hybrids of MPNN with global
attention, and fully fused local - global models with encoders. Using seven
diverse open-source datasets for benchmarking across regression and
classification tasks, we systematically isolate the contributions of message
passing, global attention, and encoder-based feature augmentation. Our study
shows that encoder-augmented MPNNs form a robust baseline, while fused
local-global models yield the clearest benefits for properties governed by
long-range interaction effects. We further quantify the accuracy - compute
trade-offs of attention, reporting its overhead in memory. Together, these
results establish the first controlled evaluation of global attention in
atomistic graph learning and provide a reproducible testbed for future model
development.

### 4. [Decoupling Correctness from Policy: A Deterministic Causal Structure for Multi-Agent Systems](http://arxiv.org/pdf/2510.05621v1)

Authors: Zhiyuan Ren, Tao Zhang, Wenchi Chen

In distributed multi-agent systems, correctness is often entangled with
operational policies such as scheduling, batching, or routing, which makes
systems brittle since performance-driven policy evolution may break integrity
guarantees. This paper introduces the Deterministic Causal Structure (DCS), a
formal foundation that decouples correctness from policy. We develop a minimal
axiomatic theory and prove four results: existence and uniqueness,
policy-agnostic invariance, observational equivalence, and axiom minimality.
These results show that DCS resolves causal ambiguities that value-centric
convergence models such as CRDTs cannot address, and that removing any axiom
collapses determinism into ambiguity. DCS thus emerges as a boundary principle
of asynchronous computation, analogous to CAP and FLP: correctness is preserved
only within the expressive power of a join-semilattice. All guarantees are
established by axioms and proofs, with only minimal illustrative constructions
included to aid intuition. This work establishes correctness as a fixed,
policy-agnostic substrate, a Correctness-as-a-Chassis paradigm, on which
distributed intelligent systems can be built modularly, safely, and evolvably.

### 5. [Intertemporal Pricing of Time-Bound Stablecoins: Measuring and Controlling the Liquidity-of-Time Premium](http://arxiv.org/pdf/2510.05711v1)

Authors: Ailiya Borjigin, Cong He

Time-bound stablecoins are DeFi assets that temporarily tokenize traditional
securities during market off-hours, enabling continuous cross-market liquidity.
We introduce the Liquidity-of-Time Premium (TLP): the extra return or cost of
providing liquidity when the primary market is closed. We build a no-arbitrage
pricing model that yields a band for fair values over different expiries, and a
dynamic risk-control mechanism that adjusts loan-to-value (LTV) ratios in real
time to keep TLP within a target range. Our analysis blends financial
engineering (no-arbitrage conditions, option-style pricing) with empirical
finance (event studies on cross-listed stocks and futures) to measure TLP under
time-zone frictions. We define TLP formally, derive closed-form expressions for
its term structure under idealized assumptions, and simulate scenarios that
vary volatility and collateralization. We then propose an LTV policy that
raises or lowers collateral to expand or curtail time-bound stablecoin supply,
analogous to a central bank adjusting rates to defend a peg. We outline
empirical proxies for TLP, including ADR premiums, overseas index futures
versus cash index divergence, and pre-market versus official close gaps.
Results show that TLP grows with closure length and volatility, yet can be
contained by adaptive LTV. We provide backtests and figures (term-structure
curves, capital-efficiency versus tail-risk trade-offs, time-liquidity
heatmaps) and discuss protocol design (vault structure, closing-price oracles,
on-chain auction liquidations). The findings position time-bound stablecoins as
a tool to reduce temporal market inefficiencies and inform future research and
deployment.

### 6. [EARL: Efficient Agentic Reinforcement Learning Systems for Large Language Models](http://arxiv.org/pdf/2510.05943v1)

Authors: Zheyue Tan, Mustapha Abdullahi, Tuo Shi, Huining Yuan, Zelai Xu, Chao Yu, Boxun Li, Bo Zhao

Reinforcement learning (RL) has become a pivotal component of large language
model (LLM) post-training, and agentic RL extends this paradigm to operate as
agents through multi-turn interaction and tool use. Scaling such systems
exposes two practical bottlenecks: (1) context length grows rapidly during
training, inflating memory usage and latency, and triggering out-of-memory
(OOM) failures; and (2) intermediate tensors accumulate with context length,
making cross-device data movement a major system bottleneck.
  We present EARL, a scalable system for efficient agentic RL. EARL designs a
parallelism selector that dynamically adapts model and training parallelism
across RL stages based on sequence length and system load, and a data
dispatcher that performs layout-aware, decentralized exchange of intermediate
data batches. Together, these components increase throughput, reduce
long-context failures, and enable stable large-scale training of agentic LLMs
without relying on hard limits or penalties of context length.

### 7. [Optimal Good-Case Latency for Sleepy Consensus](http://arxiv.org/pdf/2510.06023v1)

Authors: Yuval Efron, Joachim Neu, Ling Ren, Ertem Nusret Tas

In the context of Byzantine consensus problems such as Byzantine broadcast
(BB) and Byzantine agreement (BA), the good-case setting aims to study the
minimal possible latency of a BB or BA protocol under certain favorable
conditions, namely the designated leader being correct (for BB), or all parties
having the same input value (for BA). We provide a full characterization of the
feasibility and impossibility of good-case latency, for both BA and BB, in the
synchronous sleepy model. Surprisingly to us, we find irrational resilience
thresholds emerging: 2-round good-case BB is possible if and only if at all
times, at least $\frac{1}{\varphi} \approx 0.618$ fraction of the active
parties are correct, where $\varphi = \frac{1+\sqrt{5}}{2} \approx 1.618$ is
the golden ratio; 1-round good-case BA is possible if and only if at least
$\frac{1}{\sqrt{2}} \approx 0.707$ fraction of the active parties are correct.

### 8. [cMPI: Using CXL Memory Sharing for MPI One-Sided and Two-Sided Inter-Node Communications](http://arxiv.org/pdf/2510.05476v1)

Authors: Xi Wang, Bin Ma, Jongryool Kim, Byungil Koh, Hoshik Kim, Dong Li

Message Passing Interface (MPI) is a foundational programming model for
high-performance computing. MPI libraries traditionally employ network
interconnects (e.g., Ethernet and InfiniBand) and network protocols (e.g., TCP
and RoCE) with complex software stacks for cross-node communication. We present
cMPI, the first work to optimize MPI point-to-point communication (both
one-sided and two-sided) using CXL memory sharing on a real CXL platform,
transforming cross-node communication into memory transactions and data copies
within CXL memory, bypassing traditional network protocols. We analyze
performance across various interconnects and find that CXL memory sharing
achieves 7.2x-8.1x lower latency than TCP-based interconnects deployed in
small- and medium-scale clusters. We address challenges of CXL memory sharing
for MPI communication, including data object management over the dax
representation [50], cache coherence, and atomic operations. Overall, cMPI
outperforms TCP over standard Ethernet NIC and high-end SmartNIC by up to 49x
and 72x in latency and bandwidth, respectively, for small messages.

### 9. [Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting](http://arxiv.org/pdf/2510.05497v1)

Authors: Zhongkai Yu, Yue Guan, Zihao Yu, Chenyang Zhou, Shuyi Pei, Yangwook Kang, Yufei Ding, Po-An Tsai

Large Language Models (LLMs) with Mixture of Experts (MoE) architectures
achieve remarkable performance improvements, but their random expert selection
mechanism introduces significant data movement overhead that becomes the
dominant bottleneck in multi-unit serving systems. To forecast the patterns
underlying this data movement, we conduct comprehensive data-movement-centric
profiling across three state-of-the-art large-scale MoE models (200B- 671B)
using over 24,000 requests spanning diverse workloads. With the resulting
150GB+ trace files, we perform systematic analysis from both temporal and
spatial perspectives and distill six key insights to guide the design of
diverse future serving systems. Taking wafer-scale GPUs as a case study, we
demonstrate that minor architectural modifications leveraging our insights
achieve substantial performance gains, delivering 6.3X and 4.0X average
speedups on DeepSeek V3 and Qwen3, respectively. Our work provides the first
comprehensive data-centric analysis of MoE models at scale. Our profiling
traces and analysis results are publicly available at
{https://huggingface.co/datasets/core12345/MoE_expert_selection_trace. We will
also release our simulation framework shortly to facilitate future research in
this area.

### 10. [How many more is different?](http://arxiv.org/pdf/2510.06011v1)

Authors: Jacob Calvert, Andréa W. Richa, Dana Randall

From the formation of ice in small clusters of water molecules to the mass
raids of army ant colonies, the emergent behavior of collectives depends
critically on their size. At the same time, common wisdom holds that such
behaviors are robust to the loss of individuals. This tension points to the
need for a more systematic study of how number influences collective behavior.
We initiate this study by focusing on collective behaviors that change abruptly
at certain critical numbers of individuals. We show that a subtle modification
of standard bifurcation analysis identifies such critical numbers, including
those associated with discreteness- and noise-induced transitions. By treating
them as instances of the same phenomenon, we show that critical numbers across
physical scales and scientific domains commonly arise from competing feedbacks
that scale differently with number. We then use this idea to find overlooked
critical numbers in past studies of collective behavior and explore the
implications for their conclusions. In particular, we highlight how
deterministic approximations of stochastic models can fail near critical
numbers. We close by distinguishing these qualitative changes from
density-dependent phase transitions and by discussing how our approach could
generalize to broader classes of collective behaviors.

### Digital Libraries

### 1. [The Software Observatory: aggregating and analysing software metadata for trend computation and FAIR assessment](http://arxiv.org/pdf/2510.05705v1)

Authors: Eva Martín del Pico, Josep Lluís Gelpí, Salvador Capella-Gutiérrez

In the ever-changing realm of research software development, it is crucial
for the scientific community to grasp current trends to identify gaps that can
potentially hinder scientific progress. The adherence to the FAIR (Findable,
Accessible, Interoperable, Reusable) principles can serve as a proxy to
understand those trends and provide a mechanism to propose specific actions.
  The Software Observatory at OpenEBench
(https://openebench.bsc.es/observatory) is a novel web portal that consolidates
software metadata from various sources, offering comprehensive insights into
critical research software aspects. Our platform enables users to analyse
trends, identify patterns and advancements within the Life Sciences research
software ecosystem, and understand its evolution over time. It also evaluates
research software according to FAIR principles for research software, providing
scores for different indicators.
  Users have the ability to visualise this metadata at different levels of
granularity, ranging from the entire software landscape to specific communities
to individual software entries through the FAIRsoft Evaluator. Indeed, the
FAIRsoft Evaluator component streamlines the assessment process, helping
developers efficiently evaluate and obtain guidance to improve their software's
FAIRness.
  The Software Observatory represents a valuable resource for researchers and
software developers, as well as stakeholders, promoting better software
development practices and adherence to FAIR principles for research software.

### Discrete Mathematics

### 1. [Weighted Food Webs Make Computing Phylogenetic Diversity So Much Harder](http://arxiv.org/pdf/2510.05911v1)

Authors: Jannik Schestag

Phylogenetic trees represent certain species and their likely ancestors. In
such a tree, present-day species are leaves and an edge from u to v indicates
that u is an ancestor of v. Weights on these edges indicate the phylogenetic
distance. The phylogenetic diversity (PD) of a set of species A is the total
weight of edges that are on any path between the root of the phylogenetic tree
and a species in A. Selecting a small set of species that maximizes
phylogenetic diversity for a given phylogenetic tree is an essential task in
preservation planning, where limited resources naturally prevent saving all
species. An optimal solution can be found with a greedy algorithm [Steel,
Systematic Biology, 2005; Pardi and Goldman, PLoS Genetics, 2005]. However,
when a food web representing predator-prey relationships is given, finding a
set of species that optimizes phylogenetic diversity subject to the condition
that each saved species should be able to find food among the preserved species
is NP-hard [Spillner et al., IEEE/ACM, 2008]. We present a generalization of
this problem, where, inspired by biological considerations, the food web has
weighted edges to represent the importance of predator-prey relationships. We
show that this version is NP-hard even when both structures, the food web and
the phylogenetic tree, are stars. To cope with this intractability, we proceed
in two directions. Firstly, we study special cases where a species can only
survive if a given fraction of its prey is preserved. Secondly, we analyze
these problems through the lens of parameterized complexity. Our results
include that finding a solution is fixed-parameter tractable with respect to
the vertex cover number of the food web, assuming the phylogenetic tree is a
star.

### 2. [Improved Streaming Algorithm for Fair $k$-Center Clustering](http://arxiv.org/pdf/2510.05937v1)

Authors: Longkun Guo, Zeyu Lin, Chaoqi Jia, Chao Chen

Many real-world applications pose challenges in incorporating fairness
constraints into the $k$-center clustering problem, where the dataset consists
of $m$ demographic groups, each with a specified upper bound on the number of
centers to ensure fairness. Focusing on big data scenarios, this paper
addresses the problem in a streaming setting, where data points arrive one by
one sequentially in a continuous stream. Leveraging a structure called the
$\lambda$-independent center set, we propose a one-pass streaming algorithm
that first computes a reserved set of points during the streaming process.
Then, for the post-streaming process, we propose an approach for selecting
centers from the reserved point set by analyzing all three possible cases,
transforming the most complicated one into a specially constrained vertex cover
problem in an auxiliary graph. Our algorithm achieves a tight approximation
ratio of 5 while consuming $O(k\log n)$ memory. It can also be readily adapted
to solve the offline fair $k$-center problem, achieving a 3-approximation ratio
that matches the current state of the art. Furthermore, we extend our approach
to a semi-structured data stream, where data points from each group arrive in
batches. In this setting, we present a 3-approximation algorithm for $m = 2$
and a 4-approximation algorithm for general $m$. Lastly, we conduct extensive
experiments to evaluate the performance of our approaches, demonstrating that
they outperform existing baselines in both clustering cost and runtime
efficiency.

### 3. [A Finer View of the Parameterized Landscape of Labeled Graph Contractions](http://arxiv.org/pdf/2510.06102v1)

Authors: Yashaswini Mathur, Prafullkumar Tale

We study the \textsc{Labeled Contractibility} problem, where the input
consists of two vertex-labeled graphs $G$ and $H$, and the goal is to determine
whether $H$ can be obtained from $G$ via a sequence of edge contractions.
  Lafond and Marchand~[WADS 2025] initiated the parameterized complexity study
of this problem, showing it to be \(\W[1]\)-hard when parameterized by the
number \(k\) of allowed contractions. They also proved that the problem is
fixed-parameter tractable when parameterized by the tree-width \(\tw\) of
\(G\), via an application of Courcelle's theorem resulting in a
non-constructive algorithm.
  In this work, we present a constructive fixed-parameter algorithm for
\textsc{Labeled Contractibility} with running time \(2^{\mathcal{O}(\tw^2)}
\cdot |V(G)|^{\mathcal{O}(1)}\). We also prove that unless the Exponential Time
Hypothesis (\ETH) fails, it does not admit an algorithm running in time
\(2^{o(\tw^2)} \cdot |V(G)|^{\mathcal{O}(1)}\). This result adds
\textsc{Labeled Contractibility} to a small list of problems that admit such a
lower bound and matching algorithm.
  We further strengthen existing hardness results by showing that the problem
remains \NP-complete even when both input graphs have bounded maximum degree.
We also investigate parameterizations by \((k + \delta(G))\) where
\(\delta(G)\) denotes the degeneracy of \(G\), and rule out the existence of
subexponential-time algorithms. This answers question raised in Lafond and
Marchand~[WADS 2025]. We additionally provide an improved \FPT\ algorithm with
better dependence on \((k + \delta(G))\) than previously known. Finally, we
analyze a brute-force algorithm for \textsc{Labeled Contractibility} with
running time \(|V(H)|^{\mathcal{O}(|V(G)|)}\), and show that this running time
is optimal under \ETH.

### 4. [Critical attention scaling in long-context transformers](http://arxiv.org/pdf/2510.05554v1)

Authors: Shi Chen, Zhengjiang Lin, Yury Polyanskiy, Philippe Rigollet

As large language models scale to longer contexts, attention layers suffer
from a fundamental pathology: attention scores collapse toward uniformity as
context length $n$ increases, causing tokens to cluster excessively, a
phenomenon known as rank-collapse. While $\textit{attention scaling}$
effectively addresses this deficiency by rescaling attention scores with a
polylogarithmic factor $\beta_n$, theoretical justification for this approach
remains lacking.
  We analyze a simplified yet tractable model that magnifies the effect of
attention scaling. In this model, attention exhibits a phase transition
governed by the scaling factor $\beta_n$: insufficient scaling collapses all
tokens to a single direction, while excessive scaling reduces attention to
identity, thereby eliminating meaningful interactions between tokens. Our main
result identifies the critical scaling $\beta_n \asymp \log n$ and provides a
rigorous justification for attention scaling in YaRN and Qwen, clarifying why
logarithmic scaling maintains sparse, content-adaptive attention at large
context lengths.

### 5. [Möbius transforms and Shapley values for vector-valued functions on weighted directed acyclic multigraphs](http://arxiv.org/pdf/2510.05786v1)

Authors: Patrick Forré, Abel Jansma

We generalize the concept of M\"obius inversion and Shapley values to
directed acyclic multigraphs and weighted versions thereof. We further allow
value functions (games) and thus their M\"obius transforms (synergy function)
and Shapley values to have values in any abelian group that is a module over a
ring that contains the graph weights, e.g. vector-valued functions. To achieve
this and overcome the obstruction that the classical axioms (linearity,
efficiency, null player, symmetry) are not strong enough to uniquely determine
Shapley values in this more general setting, we analyze Shapley values from two
novel points of view: 1) We introduce projection operators that allow us to
interpret Shapley values as the recursive projection and re-attribution of
higher-order synergies to lower-order ones; 2) we propose a strengthening of
the null player axiom and a localized symmetry axiom, namely the weak elements
and flat hierarchy axioms. The former allows us to remove coalitions with
vanishing synergy while preserving the rest of the hierarchical structure. The
latter treats player-coalition bonds uniformly in the corner case of
hierarchically flat graphs. Together with linearity these axioms already imply
a unique explicit formula for the Shapley values, as well as classical
properties like efficiency, null player, symmetry, and novel ones like the
projection property. This whole framework then specializes to finite inclusion
algebras, lattices, partial orders and mereologies, and also recovers certain
previously known cases as corner cases, and presents others from a new
perspective. The admission of general weighted directed acyclic multigraph
structured hierarchies and vector-valued functions and Shapley values opens up
the possibility for new analytic tools and application areas, like machine
learning, language processing, explainable artificial intelligence, and many
more.

### 6. [Parameterized Complexity of Temporal Connected Components: Treewidth and k-Path Graphs](http://arxiv.org/pdf/2510.05806v1)

Authors: Argyrios Deligkas, Michelle Döring, Eduard Eiben, Tiger-Lily Goldsmith, George Skretas, Georg Tennigkeit

We study the parameterized complexity of maximum temporal connected
components (tccs) in temporal graphs, i.e., graphs that deterministically
change over time. In a tcc, any pair of vertices must be able to reach each
other via a time-respecting path. We consider both problems of maximum open
tccs (openTCC), which allow temporal paths through vertices outside the
component, and closed tccs (closedTCC) which require at least one temporal path
entirely within the component for every pair. We focus on the structural
parameter of treewidth, tw, and the recently introduced temporal parameter of
temporal path number, tpn, which is the minimum number of paths needed to fully
describe a temporal graph. We prove that these parameters on their own are not
sufficient for fixed parameter tractability: both openTCC and closedTCC are
NP-hard even when tw=9, and closedTCC is NP-hard when tpn=6. In contrast, we
prove that openTCC is in XP when parameterized by tpn. On the positive side, we
show that both problem become fixed parameter tractable under various
combinations of structural and temporal parameters that include, tw plus tpn,
tw plus the lifetime of the graph, and tw plus the maximum temporal degree.

### Data Structures and Algorithms

### 1. [Time To Replace Your Filter: How Maplets Simplify System Design](http://arxiv.org/pdf/2510.05518v1)

Authors: Michael A. Bender, Alex Conway, Martín Farach-Colton, Rob Johnson, Prashant Pandey

Filters such as Bloom, quotient, and cuckoo filters are fundamental building
blocks providing space-efficient approximate set membership testing. However,
many applications need to associate small values with keys-functionality that
filters do not provide. This mismatch forces complex workarounds that degrade
performance. We argue that maplets-space-efficient data structures for
approximate key-value mappings-are the right abstraction. A maplet provides the
same space benefits as filters while natively supporting key-value associations
with one-sided error guarantees. Through detailed case studies of SplinterDB
(LSM-based key-value store), Squeakr (k-mer counter), and Mantis (genomic
sequence search), we identify the common patterns and demonstrate how a unified
maplet abstraction can lead to simpler designs and better performance. We
conclude that applications benefit from defaulting to maplets rather than
filters across domains including databases, computational biology, and
networking.

### 2. [Fast-Convergent Proximity Graphs for Approximate Nearest Neighbor Search](http://arxiv.org/pdf/2510.05975v1)

Authors: Binhong Li, Xiao Yan, Shangqi Lu

Approximate nearest neighbor (ANN) search in high-dimensional metric spaces
is a fundamental problem with many applications. Over the past decade,
proximity graph (PG)-based indexes have demonstrated superior empirical
performance over alternatives. However, these methods often lack theoretical
guarantees regarding the quality of query results, especially in the worst-case
scenarios. In this paper, we introduce the {\alpha}-convergent graph
({\alpha}-CG), a new PG structure that employs a carefully designed edge
pruning rule. This rule eliminates candidate neighbors for each data point p by
applying the shifted-scaled triangle inequalities among p, its existing
out-neighbors, and new candidates. If the distance between the query point q
and its exact nearest neighbor v* is at most {\tau} for some constant {\tau} >
0, our {\alpha}-CG finds the exact nearest neighbor in poly-logarithmic time,
assuming bounded intrinsic dimensionality for the dataset; otherwise, it can
find an ANN in the same time. To enhance scalability, we develop the
{\alpha}-convergent neighborhood graph ({\alpha}-CNG), a practical variant that
applies the pruning rule locally within each point's neighbors. We also
introduce optimizations to reduce the index construction time. Experimental
results show that our {\alpha}-CNG outperforms existing PGs on real-world
datasets. For most datasets, {\alpha}-CNG can reduce the number of distance
computations and search steps by over 15% and 45%, respectively, when compared
with the best-performing baseline.

### 3. [Local Search-based Individually Fair Clustering with Outliers](http://arxiv.org/pdf/2510.06130v1)

Authors: Binita Maity, Shrutimoy Das, Anirban Dasgupta

In this paper, we present a local search-based algorithm for individually
fair clustering in the presence of outliers. We consider the individual
fairness definition proposed in Jung et al., which requires that each of the
$n$ points in the dataset must have one of the $k$ centers within its $n/k$
nearest neighbors. However, if the dataset is known to contain outliers, the
set of fair centers obtained under this definition might be suboptimal for
non-outlier points. In order to address this issue, we propose a method that
discards a set of points marked as outliers and computes the set of fair
centers for the remaining non-outlier points. Our method utilizes a randomized
variant of local search, which makes it scalable to large datasets. We also
provide an approximation guarantee of our method as well as a bound on the
number of outliers discarded. Additionally, we demonstrate our claims
experimentally on a set of real-world datasets.

### 4. [A New Quantum Linear System Algorithm Beyond the Condition Number and Its Application to Solving Multivariate Polynomial Systems](http://arxiv.org/pdf/2510.05588v1)

Authors: Jianqiang Li

Given a matrix $A$ of dimension $M \times N$ and a vector $\vec{b}$, the
quantum linear system (QLS) problem asks for the preparation of a quantum state
$|\vec{y}\rangle$ proportional to the solution of $A\vec{y} = \vec{b}$.
Existing QLS algorithms have runtimes that scale linearly with the condition
number $\kappa(A)$, the sparsity of $A$, and logarithmically with inverse
precision, but often overlook structural properties of $\vec{b}$, whose
alignment with $A$'s eigenspaces can greatly affect performance.
  In this work, we present a new QLS algorithm that explicitly leverages the
structure of the right-hand side vector $\vec{b}$. The runtime of our algorithm
depends polynomially on the sparsity of the augmented matrix $H = [A,
-\vec{b}]$, the inverse precision, the $\ell_2$ norm of the solution $\vec{y} =
A^+ \vec{b}$, and a new instance-dependent parameter \[ ET= \sum_{i=1}^M p_i^2
\cdot d_i, \] where $\vec{p} = (AA^{\top})^+ \vec{b}$, and $d_i$ denotes the
squared $\ell_2$ norm of the $i$-th row of $H$. We also introduce a
structure-aware rescaling technique tailored to the solution $\vec{y} = A^+
\vec{b}$. Unlike left preconditioning methods, which transform the linear
system to $DA\vec{y} = D\vec{b}$, our approach applies a right rescaling
matrix, reformulating the linear system as $AD\vec{z} = \vec{b}$.
  As an application of our instance-aware QLS algorithm and new rescaling
scheme, we develop a quantum algorithm for solving multivariate polynomial
systems in regimes where prior QLS-based methods fail. This yields an
end-to-end framework applicable to a broad class of problems. In particular, we
apply it to the maximum independent set (MIS) problem, formulated as a special
case of a polynomial system, and show through detailed analysis that, under
certain conditions, our quantum algorithm for MIS runs in polynomial time.

### 5. [Computational Complexity in Property Testing](http://arxiv.org/pdf/2510.05927v1)

Authors: Renato Ferreira Pinto Jr., Diptaksho Palit, Sofya Raskhodnikova

We initiate a systematic study of the computational complexity of property
testing, focusing on the relationship between query and time complexity. While
traditional work in property testing has emphasized query complexity,
relatively little is known about the computational hardness of property
testers. Our goal is to chart the landscape of time-query interplay and develop
tools for proving time complexity lower bounds. Our first contribution is a
pair of time-query hierarchy theorems for property testing. For all suitable
nondecreasing functions $q(n)$ and $t(n)$ with $t(n)\geq q(n)$, we construct
properties with query complexity $\tilde{\Theta}(q(n))$ and time complexity
$\tilde\Omega(t(n))$. Our weak hierarchy holds unconditionally, whereas the
strong version-assuming the Strong Exponential Time Hypothesis-provides better
control over the time complexity of the constructed properties.
  We then turn to halfspaces in $\mathbb{R}^d$, a fundamental class in property
testing and learning theory. We study the problem of approximating the distance
from the input function to the nearest halfspace within additive error
$\epsilon$. For the distribution-free distance approximation problem, known
algorithms achieve query complexity $O(d/\epsilon^2)$, but take time
$\tilde{\Theta}(1/\epsilon^d)$. We provide a fine-grained justification for
this gap: assuming the $k$-SUM conjecture, any algorithm must have running time
${\Omega}(1/\epsilon^{d/2})$. This fine-grained lower bound yields a provable
separation between query and time complexity for a natural and well-studied
(tolerant) testing problem. We also prove that any Statistical Query (SQ)
algorithm under the standard Gaussian distribution requires
$(1/\epsilon)^{\Omega(d)}$ queries if the queries are answered with additive
error up to $\epsilon^{\Omega(d)}$, revealing a fundamental barrier even in the
distribution-specific setting.

### 6. [Improved Streaming Algorithm for Fair $k$-Center Clustering](http://arxiv.org/pdf/2510.05937v1)

Authors: Longkun Guo, Zeyu Lin, Chaoqi Jia, Chao Chen

Many real-world applications pose challenges in incorporating fairness
constraints into the $k$-center clustering problem, where the dataset consists
of $m$ demographic groups, each with a specified upper bound on the number of
centers to ensure fairness. Focusing on big data scenarios, this paper
addresses the problem in a streaming setting, where data points arrive one by
one sequentially in a continuous stream. Leveraging a structure called the
$\lambda$-independent center set, we propose a one-pass streaming algorithm
that first computes a reserved set of points during the streaming process.
Then, for the post-streaming process, we propose an approach for selecting
centers from the reserved point set by analyzing all three possible cases,
transforming the most complicated one into a specially constrained vertex cover
problem in an auxiliary graph. Our algorithm achieves a tight approximation
ratio of 5 while consuming $O(k\log n)$ memory. It can also be readily adapted
to solve the offline fair $k$-center problem, achieving a 3-approximation ratio
that matches the current state of the art. Furthermore, we extend our approach
to a semi-structured data stream, where data points from each group arrive in
batches. In this setting, we present a 3-approximation algorithm for $m = 2$
and a 4-approximation algorithm for general $m$. Lastly, we conduct extensive
experiments to evaluate the performance of our approaches, demonstrating that
they outperform existing baselines in both clustering cost and runtime
efficiency.

### 7. [A Finer View of the Parameterized Landscape of Labeled Graph Contractions](http://arxiv.org/pdf/2510.06102v1)

Authors: Yashaswini Mathur, Prafullkumar Tale

We study the \textsc{Labeled Contractibility} problem, where the input
consists of two vertex-labeled graphs $G$ and $H$, and the goal is to determine
whether $H$ can be obtained from $G$ via a sequence of edge contractions.
  Lafond and Marchand~[WADS 2025] initiated the parameterized complexity study
of this problem, showing it to be \(\W[1]\)-hard when parameterized by the
number \(k\) of allowed contractions. They also proved that the problem is
fixed-parameter tractable when parameterized by the tree-width \(\tw\) of
\(G\), via an application of Courcelle's theorem resulting in a
non-constructive algorithm.
  In this work, we present a constructive fixed-parameter algorithm for
\textsc{Labeled Contractibility} with running time \(2^{\mathcal{O}(\tw^2)}
\cdot |V(G)|^{\mathcal{O}(1)}\). We also prove that unless the Exponential Time
Hypothesis (\ETH) fails, it does not admit an algorithm running in time
\(2^{o(\tw^2)} \cdot |V(G)|^{\mathcal{O}(1)}\). This result adds
\textsc{Labeled Contractibility} to a small list of problems that admit such a
lower bound and matching algorithm.
  We further strengthen existing hardness results by showing that the problem
remains \NP-complete even when both input graphs have bounded maximum degree.
We also investigate parameterizations by \((k + \delta(G))\) where
\(\delta(G)\) denotes the degeneracy of \(G\), and rule out the existence of
subexponential-time algorithms. This answers question raised in Lafond and
Marchand~[WADS 2025]. We additionally provide an improved \FPT\ algorithm with
better dependence on \((k + \delta(G))\) than previously known. Finally, we
analyze a brute-force algorithm for \textsc{Labeled Contractibility} with
running time \(|V(H)|^{\mathcal{O}(|V(G)|)}\), and show that this running time
is optimal under \ETH.

### 8. [Efficient learning of bosonic Gaussian unitaries](http://arxiv.org/pdf/2510.05531v1)

Authors: Marco Fanizza, Vishnu Iyer, Junseo Lee, Antonio A. Mele, Francesco A. Mele

Bosonic Gaussian unitaries are fundamental building blocks of central
continuous-variable quantum technologies such as quantum-optic interferometry
and bosonic error-correction schemes. In this work, we present the first
time-efficient algorithm for learning bosonic Gaussian unitaries with a
rigorous analysis. Our algorithm produces an estimate of the unknown unitary
that is accurate to small worst-case error, measured by the physically
motivated energy-constrained diamond distance. Its runtime and query complexity
scale polynomially with the number of modes, the inverse target accuracy, and
natural energy parameters quantifying the allowed input energy and the
unitary's output-energy growth.
  The protocol uses only experimentally friendly photonic resources: coherent
and squeezed probes, passive linear optics, and heterodyne/homodyne detection.
We then employ an efficient classical post-processing routine that leverages a
symplectic regularization step to project matrix estimates onto the symplectic
group. In the limit of unbounded input energy, our procedure attains
arbitrarily high precision using only $2m+2$ queries, where $m$ is the number
of modes. To our knowledge, this is the first provably efficient learning
algorithm for a multiparameter family of continuous-variable unitaries.

### 9. [Parameterized Complexity of Temporal Connected Components: Treewidth and k-Path Graphs](http://arxiv.org/pdf/2510.05806v1)

Authors: Argyrios Deligkas, Michelle Döring, Eduard Eiben, Tiger-Lily Goldsmith, George Skretas, Georg Tennigkeit

We study the parameterized complexity of maximum temporal connected
components (tccs) in temporal graphs, i.e., graphs that deterministically
change over time. In a tcc, any pair of vertices must be able to reach each
other via a time-respecting path. We consider both problems of maximum open
tccs (openTCC), which allow temporal paths through vertices outside the
component, and closed tccs (closedTCC) which require at least one temporal path
entirely within the component for every pair. We focus on the structural
parameter of treewidth, tw, and the recently introduced temporal parameter of
temporal path number, tpn, which is the minimum number of paths needed to fully
describe a temporal graph. We prove that these parameters on their own are not
sufficient for fixed parameter tractability: both openTCC and closedTCC are
NP-hard even when tw=9, and closedTCC is NP-hard when tpn=6. In contrast, we
prove that openTCC is in XP when parameterized by tpn. On the positive side, we
show that both problem become fixed parameter tractable under various
combinations of structural and temporal parameters that include, tw plus tpn,
tw plus the lifetime of the graph, and tw plus the maximum temporal degree.

### 10. [Efficient Heuristics and Exact Methods for Pairwise Interaction Sampling](http://arxiv.org/pdf/2510.05955v1)

Authors: Sándor P. Fekete, Phillip Keldenich, Dominik Krupke, Michael Perk

We consider a class of optimization problems that are fundamental to testing
in modern configurable software systems, e.g., in automotive industries. In
pairwise interaction sampling, we are given a (potentially very large)
configuration space, in which each dimension corresponds to a possible Boolean
feature of a software system; valid configurations are the satisfying
assignments of a given propositional formula $\varphi$. The objective is to
find a minimum-sized family of configurations, such that each pair of features
is jointly tested at least once. Due to its relevance in Software Engineering,
this problem has been studied extensively for over 20 years. In addition to new
theoretical insights (we prove BH-hardness), we provide a broad spectrum of key
contributions on the practical side that allow substantial progress for the
practical performance. Remarkably, we are able to solve the largest instances
we found in published benchmark sets (with about 500000000 feasible
interactions) to provable optimality. Previous approaches were not even able to
compute feasible solutions.

### Formal Languages and Automata Theory

### 1. [Iterating Non-Aggregative Structure Compositions](http://arxiv.org/pdf/2510.06019v1)

Authors: Marius Bozga, Radu Iosif, Florian Zuleger

An aggregative composition is a binary operation obeying the
  principle that the whole is determined by the sum of its parts. The
  development of graph algebras, on which the theory of formal graph
  languages is built, relies on aggregative compositions that behave
  like disjoint union, except for a set of well-marked interface
  vertices from both sides, that are joined. The same style of
  composition has been considered in the context of relational
  structures, that generalize graphs and use constant symbols to label
  the interface.
  In this paper, we study a non-aggregative composition operation,
  called \emph{fusion}, that joins non-deterministically chosen
  elements from disjoint structures. The sets of structures obtained
  by iteratively applying fusion do not always have bounded
  tree-width, even when starting from a tree-width bounded set.
  First, we prove that the problem of the existence of a bound on the
  tree-width of the closure of a given set under fusion is decidable,
  when the input set is described inductively by a finite
  \emph{hyperedge-replacement} (HR) grammar, written using the
  operations of aggregative composition, forgetting and renaming of
  constants. Such sets are usually called \emph{context-free}.
  Second, assuming that the closure under fusion of a context-free set
  has bounded tree-width, we show that it is the language of an
  effectively constructible HR grammar. A possible application of the
  latter result is the possiblity of checking whether all structures
  from a non-aggregatively closed set having bounded tree-width
  satisfy a given monadic second order logic formula.

### Graphics

### 1. [Teamwork: Collaborative Diffusion with Low-rank Coordination and Adaptation](http://arxiv.org/pdf/2510.05532v1)

Authors: Sam Sartor, Pieter Peers

Large pretrained diffusion models can provide strong priors beneficial for
many graphics applications. However, generative applications such as neural
rendering and inverse methods such as SVBRDF estimation and intrinsic image
decomposition require additional input or output channels. Current solutions
for channel expansion are often application specific and these solutions can be
difficult to adapt to different diffusion models or new tasks. This paper
introduces Teamwork: a flexible and efficient unified solution for jointly
increasing the number of input and output channels as well as adapting a
pretrained diffusion model to new tasks. Teamwork achieves channel expansion
without altering the pretrained diffusion model architecture by coordinating
and adapting multiple instances of the base diffusion model (\ie, teammates).
We employ a novel variation of Low Rank-Adaptation (LoRA) to jointly address
both adaptation and coordination between the different teammates. Furthermore
Teamwork supports dynamic (de)activation of teammates. We demonstrate the
flexibility and efficiency of Teamwork on a variety of generative and inverse
graphics tasks such as inpainting, single image SVBRDF estimation, intrinsic
decomposition, neural shading, and intrinsic image synthesis.

### Computer Science and Game Theory

### 1. [Hallucinating Flows for Optimal Mechanisms](http://arxiv.org/pdf/2510.05474v1)

Authors: Marios Mertzanidis, Athina Terzoglou

Myerson's seminal characterization of the revenue-optimal auction for a
single item \cite{myerson1981optimal} remains a cornerstone of mechanism
design. However, generalizing this framework to multi-item settings has proven
exceptionally challenging. Even under restrictive assumptions, closed-form
characterizations of optimal mechanisms are rare and are largely confined to
the single-agent case \cite{pavlov2011optimal,hart2017approximate,
daskalakis2018transport, GIANNAKOPOULOS2018432}, departing from the two-item
setting only when prior distributions are uniformly distributed
\cite{manelli2006bundling, daskalakis2017strong,giannakopoulos2018sjm}. In this
work, we build upon the bi-valued setting introduced by Yao
\cite{YAO_BIC_DSIC}, where each item's value has support 2 and lies in $\{a,
b\}$. Yao's result provides the only known closed-form optimal mechanism for
multiple agents. We extend this line of work along three natural axes,
establishing the first closed-form optimal mechanisms in each of the following
settings: (i) $n$ i.i.d. agents and $m$ i.i.d. items (ii) $n$ non-i.i.d. agents
and two i.i.d. items and (iii) $n$ i.i.d. agents and two non-i.i.d. items. Our
results lie at the limit of what is considered possible, since even with a
single agent and m bi-valued non-i.i.d. items, finding the optimal mechanism is
$\#P$-Hard \cite{daskalakis2014complexity, xi2018soda}. We finally generalize
the discrete analog of a result from~\cite{daskalakis2017strong}, showing that
for a single agent with $m$ items drawn from arbitrary (non-identical) discrete
distributions, grand bundling is optimal when all item values are sufficiently
large. We further show that for any continuous product distribution, grand
bundling achieves $\mathrm{OPT} - \epsilon$ revenue for large enough values.

### 2. [Mechanism design and equilibrium analysis of smart contract mediated resource allocation](http://arxiv.org/pdf/2510.05504v1)

Authors: Jinho Cha, Justin Yoo, Eunchan Daniel Cha, Emily Yoo, Caedon Geoffrey, Hyoshin Song

Decentralized coordination and digital contracting are becoming critical in
complex industrial ecosystems, yet existing approaches often rely on ad hoc
heuristics or purely technical blockchain implementations without a rigorous
economic foundation. This study develops a mechanism design framework for smart
contract-based resource allocation that explicitly embeds efficiency and
fairness in decentralized coordination. We establish the existence and
uniqueness of contract equilibria, extending classical results in mechanism
design, and introduce a decentralized price adjustment algorithm with provable
convergence guarantees that can be implemented in real time. To evaluate
performance, we combine extensive synthetic benchmarks with a proof-of-concept
real-world dataset (MovieLens). The synthetic tests probe robustness under fee
volatility, participation shocks, and dynamic demand, while the MovieLens case
study illustrates how the mechanism can balance efficiency and fairness in
realistic allocation environments. Results demonstrate that the proposed
mechanism achieves substantial improvements in both efficiency and equity while
remaining resilient to abrupt perturbations, confirming its stability beyond
steady state analysis. The findings highlight broad managerial and policy
relevance for supply chains, logistics, energy markets, healthcare resource
allocation, and public infrastructure, where transparent and auditable
coordination is increasingly critical. By combining theoretical rigor with
empirical validation, the study shows how digital contracts can serve not only
as technical artifacts but also as institutional instruments for transparency,
accountability, and resilience in high-stakes resource allocation.

### 3. [A Small Collusion is All You Need](http://arxiv.org/pdf/2510.05986v1)

Authors: Yotam Gafni

Transaction Fee Mechanisms (TFMs) study auction design in the Blockchain
context, and emphasize robustness against miner and user collusion, moreso than
traditional auction theory. \cite{chung2023foundations} introduce the notion of
a mechanism being $c$-Side-Contract-Proof ($c$-SCP), i.e., robust to a
collusion of the miner and $c$ users. Later work
\cite{chung2024collusion,welfareIncreasingCollusion} shows a gap between the
$1$-SCP and $2$-SCP classes. We show that the class of $2$-SCP mechanisms
equals that of any $c$-SCP with $c\geq 2$, under a relatively minor assumption
of consistent tie-breaking. In essence, this implies that any mechanism
vulnerable to collusion, is also vulnerable to a small collusion.

### 4. [Möbius transforms and Shapley values for vector-valued functions on weighted directed acyclic multigraphs](http://arxiv.org/pdf/2510.05786v1)

Authors: Patrick Forré, Abel Jansma

We generalize the concept of M\"obius inversion and Shapley values to
directed acyclic multigraphs and weighted versions thereof. We further allow
value functions (games) and thus their M\"obius transforms (synergy function)
and Shapley values to have values in any abelian group that is a module over a
ring that contains the graph weights, e.g. vector-valued functions. To achieve
this and overcome the obstruction that the classical axioms (linearity,
efficiency, null player, symmetry) are not strong enough to uniquely determine
Shapley values in this more general setting, we analyze Shapley values from two
novel points of view: 1) We introduce projection operators that allow us to
interpret Shapley values as the recursive projection and re-attribution of
higher-order synergies to lower-order ones; 2) we propose a strengthening of
the null player axiom and a localized symmetry axiom, namely the weak elements
and flat hierarchy axioms. The former allows us to remove coalitions with
vanishing synergy while preserving the rest of the hierarchical structure. The
latter treats player-coalition bonds uniformly in the corner case of
hierarchically flat graphs. Together with linearity these axioms already imply
a unique explicit formula for the Shapley values, as well as classical
properties like efficiency, null player, symmetry, and novel ones like the
projection property. This whole framework then specializes to finite inclusion
algebras, lattices, partial orders and mereologies, and also recovers certain
previously known cases as corner cases, and presents others from a new
perspective. The admission of general weighted directed acyclic multigraph
structured hierarchies and vector-valued functions and Shapley values opens up
the possibility for new analytic tools and application areas, like machine
learning, language processing, explainable artificial intelligence, and many
more.

### Human-Computer Interaction

### 1. [Two Modes of Reflection: How Temporal, Spatial, and Social Distances Affect Reflective Writing in Family Caregiving](http://arxiv.org/pdf/2510.05510v1)

Authors: Shunpei Norihama, Yuka Iwane, Jo Takezawa, Simo Hosio, Mari Hirano, Naomi Yamashita, Koji Yatani

Writing about personal experiences can improve well-being, but for family
caregivers, fixed or user-initiated schedules often miss the right moments.
Drawing on Construal Level Theory, we conducted a three-week field study with
47 caregivers using a chatbot that delivered daily reflective writing prompts
and captured temporal, spatial, and social contexts. We collected 958 writing
entries, resulting in 5,412 coded segments. Our Analysis revealed two
reflective modes. Under proximal conditions, participants produced detailed,
emotion-rich, and care recipient-focused narratives that supported emotional
release. Under distal conditions, they generated calmer, self-focused, and
analytic accounts that enabled objective reflection and cognitive reappraisal.
Participants described trade-offs: proximity preserved vivid detail but limited
objectivity, while distance enabled analysis but risked memory loss. This work
contributes empirical evidence of how psychological distances shape reflective
writing and proposes design implications for distance-aware Just-in-Time
Adaptive Interventions for family caregivers' mental health support.

### 2. [Locability: An Ability-Based Ranking Model for Virtual Reality Locomotion Techniques](http://arxiv.org/pdf/2510.05679v1)

Authors: Rachel L. Franz, Jacob O. Wobbrock

There are over a hundred virtual reality (VR) locomotion techniques that
exist today, with new ones being designed as VR technology evolves. The
different ways of controlling locomotion techniques (e.g., gestures, button
inputs, body movements), along with the diversity of upper-body motor
impairments, can make it difficult for a user to know which locomotion
technique is best suited to their particular abilities. Moreover,
trial-and-error can be difficult, time-consuming, and costly. Using machine
learning techniques and data from 20 people with and without upper-body motor
impairments, we developed a modeling approach to predict a ranked list of a
user's fastest techniques based on questionnaire and interaction data. We found
that a user's fastest technique could be predicted based on interaction data
with 92% accuracy and that predicted locomotion times were within 12% of
observed times. The model we trained could also rank six locomotion techniques
based on speed with 61% accuracy and that predictions were within 8% of
observed times. Our findings contribute to growing research in VR accessibility
by taking an ability-based design approach to adapt systems to users'
abilities.

### 3. [Vipera: Blending Visual and LLM-Driven Guidance for Systematic Auditing of Text-to-Image Generative AI](http://arxiv.org/pdf/2510.05742v1)

Authors: Yanwei Huang, Wesley Hanwen Deng, Sijia Xiao, Motahhare Eslami, Jason I. Hong, Arpit Narechania, Adam Perer

Despite their increasing capabilities, text-to-image generative AI systems
are known to produce biased, offensive, and otherwise problematic outputs.
While recent advancements have supported testing and auditing of generative AI,
existing auditing methods still face challenges in supporting effectively
explore the vast space of AI-generated outputs in a structured way. To address
this gap, we conducted formative studies with five AI auditors and synthesized
five design goals for supporting systematic AI audits. Based on these insights,
we developed Vipera, an interactive auditing interface that employs multiple
visual cues including a scene graph to facilitate image sensemaking and inspire
auditors to explore and hierarchically organize the auditing criteria.
Additionally, Vipera leverages LLM-powered suggestions to facilitate
exploration of unexplored auditing directions. Through a controlled experiment
with 24 participants experienced in AI auditing, we demonstrate Vipera's
effectiveness in helping auditors navigate large AI output spaces and organize
their analyses while engaging with diverse criteria.

### 4. [The Interplay of Attention and Memory in Visual Enumeration](http://arxiv.org/pdf/2510.05833v1)

Authors: B. Sankar, Devottama Sen, Dibakar Sen

Humans navigate and understand complex visual environments by subconsciously
quantifying what they see, a process known as visual enumeration. However,
traditional studies using flat screens fail to capture the cognitive dynamics
of this process over the large visual fields of real-world scenes. To address
this gap, we developed an immersive virtual reality system with integrated
eye-tracking to investigate the interplay between attention and memory during
complex enumeration. We conducted a two-phase experiment where participants
enumerated scenes of either simple abstract shapes or complex real-world
objects, systematically varying the task intent (e.g., selective vs. exhaustive
counting) and the spatial layout of items. Our results reveal that task intent
is the dominant factor driving performance, with selective counting imposing a
significant cognitive cost that was dramatically amplified by stimulus
complexity. The semantic processing required for real-world objects reduced
accuracy and suppressed memory recall, while the influence of spatial layout
was secondary and statistically non-significant when a higher-order cognitive
task intent was driving the human behaviour. We conclude that real-world
enumeration is fundamentally constrained by the cognitive load of semantic
processing, not just the mechanics of visual search. Our findings demonstrate
that under high cognitive demand, the effort to understand what we are seeing
directly limits our capacity to remember it.

### 5. [From "Arbitrary Timberland" To "Skyline Charts": Is Visualization At Risk From The Pollution of Scientific Literature?](http://arxiv.org/pdf/2510.05844v1)

Authors: Lonni Besançon

In this essay, I argue that, while visualization research does not seem to be
directly at risk of being corrupted by the current massive wave of polluted
research, certain visualization concepts are being used in fraudulent fashions
and fields close to ours are being targeted. Worse, the society publishing our
work is overwhelmed by thousands of questionable papers that are being,
unfortunately, published. As a community, and if we want our research to remain
as good as it currently is, I argue that we should all get involved with our
variety of skills to help identify and correct the current scientific record. I
thus aim to present a few questionable practices that are worth knowing about
when reviewing for fields using visualization research, and hopefully will
never be useful when reviewing for our main venues. I also argue that our skill
set could become particularly relevant in the future and invite scholars of the
fields to try to get involved.

### 6. [Observing Interaction Rather Than Interfaces](http://arxiv.org/pdf/2510.06156v1)

Authors: Guillaume Rivière

The science of Human-Computer Interaction (HCI) is populated by isolated
empirical findings, often tied to specific technologies, designs, and tasks.
This situation probably lies in observing the wrong object of study, that is to
say, observing interfaces rather than interaction. This paper proposes an
experimental methodology, powered by a research methodology, that enables
tackling the ambition of observing interaction (rather than interfaces). These
observations are done during the treatment of applicative cases, allowing to
generate and replicate results covering various experimental conditions,
expressed from the need of end users and the evolution of technologies.
Performing these observations when developing applicative prototypes
illustrating novel technologies' utility allows, in the same time, to benefit
from an optimization of these prototypes to better accomplish end users tasks.
This paper depicts a long term research direction, from generating the initial
observations of interaction properties and their replication, to their
integration, that would then lead to exploring the possible relations existing
between those properties, to end toward the description of human-computer
interaction's physics.

### 7. [Taxonomy of User Needs and Actions](http://arxiv.org/pdf/2510.06124v1)

Authors: Renee Shelby, Fernando Diaz, Vinodkumar Prabhakaran

The growing ubiquity of conversational AI highlights the need for frameworks
that capture not only users' instrumental goals but also the situated,
adaptive, and social practices through which they achieve them. Existing
taxonomies of conversational behavior either overgeneralize, remain
domain-specific, or reduce interactions to narrow dialogue functions. To
address this gap, we introduce the Taxonomy of User Needs and Actions (TUNA),
an empirically grounded framework developed through iterative qualitative
analysis of 1193 human-AI conversations, supplemented by theoretical review and
validation across diverse contexts. TUNA organizes user actions into a
three-level hierarchy encompassing behaviors associated with information
seeking, synthesis, procedural guidance, content creation, social interaction,
and meta-conversation. By centering user agency and appropriation practices,
TUNA enables multi-scale evaluation, supports policy harmonization across
products, and provides a backbone for layering domain-specific taxonomies. This
work contributes a systematic vocabulary for describing AI use, advancing both
scholarly understanding and practical design of safer, more responsive, and
more accountable conversational systems.

### 8. [Evidence of Cognitive Biases in Capture-the-Flag Cybersecurity Competitions](http://arxiv.org/pdf/2510.05771v1)

Authors: Carolina Carreira, Anu Aggarwal, Alejandro Cuevas, Maria José Ferreira, Hanan Hibshi, Cleotilde Gonzalez

Understanding how cognitive biases influence adversarial decision-making is
essential for developing effective cyber defenses. Capture-the-Flag (CTF)
competitions provide an ecologically valid testbed to study attacker behavior
at scale, simulating real-world intrusion scenarios under pressure. We analyze
over 500,000 submission logs from picoCTF, a large educational CTF platform, to
identify behavioral signatures of cognitive biases with defensive implications.
Focusing on availability bias and the sunk cost fallacy, we employ a
mixed-methods approach combining qualitative coding, descriptive statistics,
and generalized linear modeling. Our findings show that participants often
submitted flags with correct content but incorrect formatting (availability
bias), and persisted in attempting challenges despite repeated failures and
declining success probabilities (sunk cost fallacy). These patterns reveal that
biases naturally shape attacker behavior in adversarial contexts. Building on
these insights, we outline a framework for bias-informed adaptive defenses that
anticipate, rather than simply react to, adversarial actions.

### 9. ["Your Doctor is Spying on You": An Analysis of Data Practices in Mobile Healthcare Applications](http://arxiv.org/pdf/2510.06015v1)

Authors: Luke Stevenson, Sanchari Das

Mobile healthcare (mHealth) applications promise convenient, continuous
patient-provider interaction but also introduce severe and often underexamined
security and privacy risks. We present an end-to-end audit of 272 Android
mHealth apps from Google Play, combining permission forensics, static
vulnerability analysis, and user review mining. Our multi-tool assessment with
MobSF, RiskInDroid, and OWASP Mobile Audit revealed systemic weaknesses: 26.1%
request fine-grained location without disclosure, 18.3% initiate calls
silently, and 73 send SMS without notice. Nearly half (49.3%) still use
deprecated SHA-1 encryption, 42 transmit unencrypted data, and 6 remain
vulnerable to StrandHogg 2.0. Analysis of 2.56 million user reviews found 28.5%
negative or neutral sentiment, with over 553,000 explicitly citing privacy
intrusions, data misuse, or operational instability. These findings demonstrate
the urgent need for enforceable permission transparency, automated pre-market
security vetting, and systematic adoption of secure-by-design practices to
protect Protected Health Information (PHI).

### 10. [Benchmark It Yourself (BIY): Preparing a Dataset and Benchmarking AI Models for Scatterplot-Related Tasks](http://arxiv.org/pdf/2510.06071v1)

Authors: João Palmeiro, Diogo Duarte, Rita Costa, Pedro Bizarro

AI models are increasingly used for data analysis and visualization, yet
benchmarks rarely address scatterplot-specific tasks, limiting insight into
performance. To address this gap for one of the most common chart types, we
introduce a synthetic, annotated dataset of over 18,000 scatterplots from six
data generators and 17 chart designs, and a benchmark based on it. We evaluate
proprietary models from OpenAI and Google using N-shot prompting on five
distinct tasks derived from annotations of cluster bounding boxes, their center
coordinates, and outlier coordinates. OpenAI models and Gemini 2.5 Flash,
especially when prompted with examples, are viable options for counting
clusters and, in Flash's case, outliers (90%+ Accuracy). However, the results
for localization-related tasks are unsatisfactory: Precision and Recall are
near or below 50%, except for Flash in outlier identification (65.01%).
Furthermore, the impact of chart design on performance appears to be a
secondary factor, but it is advisable to avoid scatterplots with wide aspect
ratios (16:9 and 21:9) or those colored randomly. Supplementary materials are
available at https://github.com/feedzai/biy-paper.

### Information Retrieval

### 1. [Automated Research Article Classification and Recommendation Using NLP and ML](http://arxiv.org/pdf/2510.05495v1)

Authors: Shadikur Rahman, Hasibul Karim Shanto, Umme Ayman Koana, Syed Muhammad Danish

In the digital era, the exponential growth of scientific publications has
made it increasingly difficult for researchers to efficiently identify and
access relevant work. This paper presents an automated framework for research
article classification and recommendation that leverages Natural Language
Processing (NLP) techniques and machine learning. Using a large-scale arXiv.org
dataset spanning more than three decades, we evaluate multiple feature
extraction approaches (TF--IDF, Count Vectorizer, Sentence-BERT, USE,
Mirror-BERT) in combination with diverse machine learning classifiers (Logistic
Regression, SVM, Na\"ive Bayes, Random Forest, Gradient Boosted Trees, and
k-Nearest Neighbour). Our experiments show that Logistic Regression with
TF--IDF consistently yields the best classification performance, achieving an
accuracy of 69\%. To complement classification, we incorporate a recommendation
module based on the cosine similarity of vectorized articles, enabling
efficient retrieval of related research papers. The proposed system directly
addresses the challenge of information overload in digital libraries and
demonstrates a scalable, data-driven solution to support literature discovery.

### 2. [Limitations of Current Evaluation Practices for Conversational Recommender Systems and the Potential of User Simulation](http://arxiv.org/pdf/2510.05624v1)

Authors: Nolwenn Bernard, Krisztian Balog

Research and development on conversational recommender systems (CRSs)
critically depends on sound and reliable evaluation methodologies. However, the
interactive nature of these systems poses significant challenges for automatic
evaluation. This paper critically examines current evaluation practices and
identifies two key limitations: the over-reliance on static test collections
and the inadequacy of existing evaluation metrics. To substantiate this
critique, we analyze real user interactions with nine existing CRSs and
demonstrate a striking disconnect between self-reported user satisfaction and
performance scores reported in prior literature. To address these limitations,
this work explores the potential of user simulation to generate dynamic
interaction data, offering a departure from static datasets. Furthermore, we
propose novel evaluation metrics, based on a general reward/cost framework,
designed to better align with real user satisfaction. Our analysis of different
simulation approaches provides valuable insights into their effectiveness and
reveals promising initial results, showing improved correlation with system
rankings compared to human evaluation. While these findings indicate a
significant step forward in CRS evaluation, we also identify areas for future
research and refinement in both simulation techniques and evaluation metrics.

### 3. [How public datasets constrain the development of diversity-aware news recommender systems, and what law could do about it](http://arxiv.org/pdf/2510.05952v1)

Authors: Max van Drunen, Sanne Vrijenhoek

News recommender systems increasingly determine what news individuals see
online. Over the past decade, researchers have extensively critiqued
recommender systems that prioritise news based on user engagement. To offer an
alternative, researchers have analysed how recommender systems could support
the media's ability to fulfil its role in democratic society by recommending
news based on editorial values, particularly diversity. However, there
continues to be a large gap between normative theory on how news recommender
systems should incorporate diversity, and technical literature that designs
such systems. We argue that to realise diversity-aware recommender systems in
practice, it is crucial to pay attention to the datasets that are needed to
train modern news recommenders. We aim to make two main contributions. First,
we identify the information a dataset must include to enable the development of
the diversity-aware news recommender systems proposed in normative literature.
Based on this analysis, we assess the limitations of currently available public
datasets, and show what potential they do have to expand research into
diversity-aware recommender systems. Second, we analyse why and how European
law and policy can be used to provide researchers with structural access to the
data they need to develop diversity-aware news recommender systems.

### 4. [KEO: Knowledge Extraction on OMIn via Knowledge Graphs and RAG for Safety-Critical Aviation Maintenance](http://arxiv.org/pdf/2510.05524v1)

Authors: Kuangshi Ai, Jonathan A. Karr Jr, Meng Jiang, Nitesh V. Chawla, Chaoli Wang

We present Knowledge Extraction on OMIn (KEO), a domain-specific knowledge
extraction and reasoning framework with large language models (LLMs) in
safety-critical contexts. Using the Operations and Maintenance Intelligence
(OMIn) dataset, we construct a QA benchmark spanning global sensemaking and
actionable maintenance tasks. KEO builds a structured Knowledge Graph (KG) and
integrates it into a retrieval-augmented generation (RAG) pipeline, enabling
more coherent, dataset-wide reasoning than traditional text-chunk RAG. We
evaluate locally deployable LLMs (Gemma-3, Phi-4, Mistral-Nemo) and employ
stronger models (GPT-4o, Llama-3.3) as judges. Experiments show that KEO
markedly improves global sensemaking by revealing patterns and system-level
insights, while text-chunk RAG remains effective for fine-grained procedural
tasks requiring localized retrieval. These findings underscore the promise of
KG-augmented LLMs for secure, domain-specific QA and their potential in
high-stakes reasoning.

### 5. [AgentDR Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents](http://arxiv.org/pdf/2510.05598v1)

Authors: Mingdai Yang, Nurendra Choudhary, Jiangshu Du, Edward W. Huang, Philip S. Yu, Karthik Subbian, Danai Kourta

Recent agent-based recommendation frameworks aim to simulate user behaviors
by incorporating memory mechanisms and prompting strategies, but they struggle
with hallucinating non-existent items and full-catalog ranking. Besides, a
largely underexplored opportunity lies in leveraging LLMs'commonsense reasoning
to capture user intent through substitute and complement relationships between
items, which are usually implicit in datasets and difficult for traditional
ID-based recommenders to capture. In this work, we propose a novel LLM-agent
framework, AgenDR, which bridges LLM reasoning with scalable recommendation
tools. Our approach delegates full-ranking tasks to traditional models while
utilizing LLMs to (i) integrate multiple recommendation outputs based on
personalized tool suitability and (ii) reason over substitute and complement
relationships grounded in user history. This design mitigates hallucination,
scales to large catalogs, and enhances recommendation relevance through
relational reasoning. Through extensive experiments on three public grocery
datasets, we show that our framework achieves superior full-ranking
performance, yielding on average a twofold improvement over its underlying
tools. We also introduce a new LLM-based evaluation metric that jointly
measures semantic alignment and ranking correctness.

### 6. [Peeking inside the Black-Box: Reinforcement Learning for Explainable and Accurate Relation Extraction](http://arxiv.org/pdf/2510.06198v1)

Authors: Xinyu Guo, Zhengliang Shi, Minglai Yang, Mahdi Rahimi, Mihai Surdeanu

This paper introduces a framework for relation extraction (RE) that enhances
both accuracy and explainability. The framework has two key components: (i) a
reasoning mechanism that formulates relation extraction as a series of
text-processing steps inspired by cognitive science, and (ii) an optimization
process driven by reinforcement learning (RL) with a novel reward function
designed to improve both task accuracy and explanation quality. We call our
approach CogRE. Our framework addresses the lack of supervision for
language-based explanations in traditional RE by promoting outputs that include
important relation keywords. These keywords are drawn from a high-quality
dictionary that is automatically constructed using an LLM. We evaluate our
approach for the task of one-shot RE using two LLMs and two RE datasets. Our
experiments show that CogRE improves explanation quality by addressing two
common failure patterns in one-shot RE: poor attention focus and limited
one-shot learning capability. For example, our cognitive-structured reasoning
with Qwen2.5-15B-Instruct on One-shot NYT29 achieves 24.65% F1, surpassing
prior reasoning-based designs. Optimizing this approach with RL using our
reward further improves performance by +23.46% (absolute). Finally, human
evaluation shows that our best model generates relational keywords closely
aligned with gold labels, increasing human explanation quality ratings by 54%
(relative).

### 7. [Deterministic Legal Retrieval: An Action API for Querying the SAT-Graph RAG](http://arxiv.org/pdf/2510.06002v1)

Authors: Hudson de Martim

The Structure-Aware Temporal Graph RAG (SAT-Graph RAG) addresses core
limitations of standard Retrieval-Augmented Generation in the legal domain by
providing a verifiable knowledge graph that models hierarchical structure,
temporal evolution, and causal events of legal norms. However, a critical gap
remains: how to reliably query this structured knowledge without sacrificing
its deterministic properties. This paper introduces the SAT-Graph API, a formal
query execution layer centered on canonical actions-atomic, composable, and
auditable primitives that isolate probabilistic discovery from deterministic
retrieval. These actions enable: (i) high-precision hybrid search; (ii) robust
reference resolution; (iii) point-in-time version retrieval; and (iv) auditable
causal tracing. We demonstrate how planner-guided agents can decompose complex
queries into Directed Acyclic Graphs (DAGs) of these actions. This two-layer
architecture transforms retrieval from an opaque black box to a transparent,
auditable process, directly addressing Explainable AI (XAI) requirements for
high-stakes domains.

### Machine Learning

### 1. [ATOM: A Pretrained Neural Operator for Multitask Molecular Dynamics](http://arxiv.org/pdf/2510.05482v1)

Authors: Luke Thompson, Davy Guan, Dai Shi, Slade Matthews, Junbin Gao, Andi Han

Molecular dynamics (MD) simulations underpin modern computational drug dis-
covery, materials science, and biochemistry. Recent machine learning models
provide high-fidelity MD predictions without the need to repeatedly solve
quantum mechanical forces, enabling significant speedups over conventional
pipelines. Yet many such methods typically enforce strict equivariance and rely
on sequential rollouts, thus limiting their flexibility and simulation
efficiency. They are also com- monly single-task, trained on individual
molecules and fixed timeframes, which restricts generalization to unseen
compounds and extended timesteps. To address these issues, we propose Atomistic
Transformer Operator for Molecules (ATOM), a pretrained transformer neural
operator for multitask molecular dynamics. ATOM adopts a quasi-equivariant
design that requires no explicit molecular graph and employs a temporal
attention mechanism, allowing for the accurate parallel decod- ing of multiple
future states. To support operator pretraining across chemicals and timescales,
we curate TG80, a large, diverse, and numerically stable MD dataset with over
2.5 million femtoseconds of trajectories across 80 compounds. ATOM achieves
state-of-the-art performance on established single-task benchmarks, such as
MD17, RMD17 and MD22. After multitask pretraining on TG80, ATOM shows
exceptional zero-shot generalization to unseen molecules across varying time
hori- zons. We believe ATOM represents a significant step toward accurate,
efficient, and transferable molecular dynamics models

### 2. [EEG-Based Acute Pain Classification: Machine Learning Model Comparison and Real-Time Clinical Feasibility](http://arxiv.org/pdf/2510.05511v1)

Authors: Aavid Mathrawala, Dhruv Kurup, Josie Lau

Current pain assessment within hospitals often relies on self-reporting or
non-specific EKG vital signs. This system leaves critically ill, sedated, and
cognitively impaired patients vulnerable to undertreated pain and opioid
overuse. Electroencephalography (EEG) offers a noninvasive method of measuring
brain activity. This technology could potentially be applied as an assistive
tool to highlight nociceptive processing in order to mitigate this issue. In
this study, we compared machine learning models for classifying high-pain
versus low/no-pain EEG epochs using data from fifty-two healthy adults exposed
to laser-evoked pain at three intensities (low, medium, high). Each four-second
epoch was transformed into a 537-feature vector spanning spectral power, band
ratios, Hjorth parameters, entropy measures, coherence, wavelet energies, and
peak-frequency metrics. Nine traditional machine learning models were evaluated
with leave-one-participant-out cross-validation. A support vector machine with
radial basis function kernel achieved the best offline performance with 88.9%
accuracy and sub-millisecond inference time (1.02 ms). Our Feature importance
analysis was consistent with current canonical pain physiology, showing
contralateral alpha suppression, midline theta/alpha enhancement, and frontal
gamma bursts. The real-time XGBoost model maintained an end-to-end latency of
about 4 ms and 94.2% accuracy, demonstrating that an EEG-based pain monitor is
technically feasible within a clinical setting and provides a pathway towards
clinical validation.

### 3. [ARMOR: High-Performance Semi-Structured Pruning via Adaptive Matrix Factorization](http://arxiv.org/pdf/2510.05528v1)

Authors: Lawrence Liu, Alexander Liu, Mengdi Wang, Tuo Zhao, Lin F. Yang

Large language models (LLMs) present significant deployment challenges due to
their immense computational and memory requirements. While semi-structured
pruning, particularly 2:4 sparsity, offers a path to practical hardware
acceleration, existing methods often incur substantial performance degradation.
To bridge this gap, we introduce ARMOR: (Adaptive Representation with
Matrix-factORization), a novel one-shot post-training pruning algorithm.
Instead of directly pruning weights, ARMOR factorizes each weight matrix into a
2:4 sparse core wrapped by two low-overhead, block diagonal matrices. These
wrappers act as efficient pre and post-transformation error correctors,
offering greater flexibility to preserve model quality compared to conventional
2:4 pruning techniques. The sparse core and block diagonal wrappers are chosen
through a block coordinate descent algorithm that minimizes a layer-wise proxy
loss. We theoretically prove this optimization is guaranteed to converge to a
solution with a proxy loss less than or equal to state-of-the-art pruning
algorithms. Experiments on Llama (Touvron et al., 2023; Dubey et al., 2024) and
Qwen (Yang et al., 2025) model families demonstrate that ARMOR consistently and
significantly outperforms state-of-the-art 2:4 pruning methods across a wide
range of downstream tasks and perplexity evaluations. ARMOR achieves this
superior performance while retaining the inference speedups and substantial
memory usage reductions of 2:4 pruning, establishing a more effective trade-off
between model compression and task accuracy

### 4. [LATTA: Langevin-Anchored Test-Time Adaptation for Enhanced Robustness and Stability](http://arxiv.org/pdf/2510.05530v1)

Authors: Harshil Vejendla

Test-time adaptation (TTA) aims to adapt a pretrained model to distribution
shifts using only unlabeled test data. While promising, existing methods like
Tent suffer from instability and can catastrophically forget the source
knowledge, especially with small batch sizes or challenging corruptions. We
argue that this arises from overly deterministic updates on a complex loss
surface. In this paper, we introduce Langevin-Anchored Test-Time Adaptation
(LATTA), a novel approach that regularizes adaptation through two key
mechanisms: (1) a noisy weight perturbation inspired by Stochastic Gradient
Langevin Dynamics (SGLD) to explore the local parameter space and escape poor
local minima, and (2) a stable weight anchor that prevents the model from
diverging from its robust source pre-training. This combination allows LATTA to
adapt effectively without sacrificing stability. Unlike prior Bayesian TTA
methods, LATTA requires no architectural changes or expensive Monte Carlo
passes. We conduct extensive experiments on standard benchmarks, including
Rotated-MNIST and the more challenging CIFAR-10-C. Our results demonstrate that
LATTA significantly outperforms existing methods, including Tent, CoTTA, and
EATA, setting a new state of the art for self-supervised TTA by improving
average accuracy on CIFAR-10-C by over 2% while simultaneously reducing
performance variance.

### 5. [Efficient Learning-based Graph Simulation for Temporal Graphs](http://arxiv.org/pdf/2510.05569v1)

Authors: Sheng Xiang, Chenhao Xu, Dawei Cheng, Xiaoyang Wang, Ying Zhang

Graph simulation has recently received a surge of attention in graph
processing and analytics. In real-life applications, e.g. social science,
biology, and chemistry, many graphs are composed of a series of evolving graphs
(i.e., temporal graphs). While most of the existing graph generators focus on
static graphs, the temporal information of the graphs is ignored. In this
paper, we focus on simulating temporal graphs, which aim to reproduce the
structural and temporal properties of the observed real-life temporal graphs.
In this paper, we first give an overview of the existing temporal graph
generators, including recently emerged learning-based approaches. Most of these
learning-based methods suffer from one of the limitations: low efficiency in
training or slow generating, especially for temporal random walk-based methods.
Therefore, we propose an efficient learning-based approach to generate graph
snapshots, namely temporal graph autoencoder (TGAE). Specifically, we propose
an attention-based graph encoder to encode temporal and structural
characteristics on sampled ego-graphs. And we proposed an ego-graph decoder
that can achieve a good trade-off between simulation quality and efficiency in
temporal graph generation. Finally, the experimental evaluation is conducted
among our proposed TGAE and representative temporal graph generators on
real-life temporal graphs and synthesized graphs. It is reported that our
proposed approach outperforms the state-of-the-art temporal graph generators by
means of simulation quality and efficiency.

### 6. [(Token-Level) \textbf{InfoRMIA}: Stronger Membership Inference and Memorization Assessment for LLMs](http://arxiv.org/pdf/2510.05582v1)

Authors: Jiashu Tao, Reza Shokri

Machine learning models are known to leak sensitive information, as they
inevitably memorize (parts of) their training data. More alarmingly, large
language models (LLMs) are now trained on nearly all available data, which
amplifies the magnitude of information leakage and raises serious privacy
risks. Hence, it is more crucial than ever to quantify privacy risk before the
release of LLMs. The standard method to quantify privacy is via membership
inference attacks, where the state-of-the-art approach is the Robust Membership
Inference Attack (RMIA). In this paper, we present InfoRMIA, a principled
information-theoretic formulation of membership inference. Our method
consistently outperforms RMIA across benchmarks while also offering improved
computational efficiency.
  In the second part of the paper, we identify the limitations of treating
sequence-level membership inference as the gold standard for measuring leakage.
We propose a new perspective for studying membership and memorization in LLMs:
token-level signals and analyses. We show that a simple token-based InfoRMIA
can pinpoint which tokens are memorized within generated outputs, thereby
localizing leakage from the sequence level down to individual tokens, while
achieving stronger sequence-level inference power on LLMs. This new scope
rethinks privacy in LLMs and can lead to more targeted mitigation, such as
exact unlearning.

### 7. [Riddled basin geometry sets fundamental limits to predictability and reproducibility in deep learning](http://arxiv.org/pdf/2510.05606v1)

Authors: Andrew Ly, Pulin Gong

Fundamental limits to predictability are central to our understanding of many
physical and computational systems. Here we show that, despite its remarkable
capabilities, deep learning exhibits such fundamental limits rooted in the
fractal, riddled geometry of its basins of attraction: any initialization that
leads to one solution lies arbitrarily close to another that leads to a
different one. We derive sufficient conditions for the emergence of riddled
basins by analytically linking features widely observed in deep learning,
including chaotic learning dynamics and symmetry-induced invariant subspaces,
to reveal a general route to riddling in realistic deep networks. The resulting
basins of attraction possess an infinitely fine-scale fractal structure
characterized by an uncertainty exponent near zero, so that even large
increases in the precision of initial conditions yield only marginal gains in
outcome predictability. Riddling thus imposes a fundamental limit on the
predictability and hence reproducibility of neural network training, providing
a unified account of many empirical observations. These results reveal a
general organizing principle of deep learning with important implications for
optimization and the safe deployment of artificial intelligence.

### 8. [Primal-Dual Direct Preference Optimization for Constrained LLM Alignment](http://arxiv.org/pdf/2510.05703v1)

Authors: Yihan Du, Seo Taek Kong, R. Srikant

The widespread application of Large Language Models (LLMs) imposes increasing
demands on safety, such as reducing harmful content and fake information, and
avoiding certain forbidden tokens due to rules and laws. While there have been
several recent works studying safe alignment of LLMs, these works either
require the training of reward and cost models and incur high memory and
computational costs, or need prior knowledge about the optimal solution.
Motivated by this fact, we study the problem of constrained alignment in LLMs,
i.e., maximizing the output reward while restricting the cost due to
potentially unsafe content to stay below a threshold. For this problem, we
propose a novel primal-dual DPO approach, which first trains a model using
standard DPO on reward preference data to provide reward information, and then
adopts a rearranged Lagrangian DPO objective utilizing the provided reward
information to fine-tune LLMs on cost preference data. Our approach
significantly reduces memory and computational costs, and does not require
extra prior knowledge. Moreover, we establish rigorous theoretical guarantees
on the suboptimality and constraint violation of the output policy. We also
extend our approach to an online data setting by incorporating exploration
bonuses, which enables our approach to explore uncovered prompt-response space,
and then provide theoretical results that get rid of the dependence on
preference data coverage. Experimental results on the widely-used preference
dataset PKU-SafeRLHF demonstrate the effectiveness of our approach.

### 9. [DiffSDA: Unsupervised Diffusion Sequential Disentanglement Across Modalities](http://arxiv.org/pdf/2510.05717v1)

Authors: Hedi Zisling, Ilan Naiman, Nimrod Berman, Supasorn Suwajanakorn, Omri Azencot

Unsupervised representation learning, particularly sequential
disentanglement, aims to separate static and dynamic factors of variation in
data without relying on labels. This remains a challenging problem, as existing
approaches based on variational autoencoders and generative adversarial
networks often rely on multiple loss terms, complicating the optimization
process. Furthermore, sequential disentanglement methods face challenges when
applied to real-world data, and there is currently no established evaluation
protocol for assessing their performance in such settings. Recently, diffusion
models have emerged as state-of-the-art generative models, but no theoretical
formalization exists for their application to sequential disentanglement. In
this work, we introduce the Diffusion Sequential Disentanglement Autoencoder
(DiffSDA), a novel, modal-agnostic framework effective across diverse
real-world data modalities, including time series, video, and audio. DiffSDA
leverages a new probabilistic modeling, latent diffusion, and efficient
samplers, while incorporating a challenging evaluation protocol for rigorous
testing. Our experiments on diverse real-world benchmarks demonstrate that
DiffSDA outperforms recent state-of-the-art methods in sequential
disentanglement.

### 10. [Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches](http://arxiv.org/pdf/2510.05748v1)

Authors: Hachem Madmoun, Salem Lahlou

Eliciting cooperation in multi-agent LLM systems is critical for AI
alignment. We investigate two approaches: direct communication and curriculum
learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases
cooperation from 0% to 48.3%, demonstrating communication as a robust
coordination mechanism. In contrast, we find that curriculum learning is highly
sensitive to design choices: our pedagogical curriculum through progressively
complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game
with Punishment. Qualitative analysis reveals that curricula emphasizing
defection-equilibrium games can induce "learned pessimism" in agents. These
findings suggest that for coordination problems, simple communication protocols
may be more reliable than experience-based training, and that curriculum design
for social dilemmas requires careful attention to the strategic lessons
embedded in game sequences.

### Neural and Evolutionary Computing

### 1. [From Neural Activity to Computation: Biological Reservoirs for Pattern Recognition in Digit Classification](http://arxiv.org/pdf/2510.05637v1)

Authors: Ludovico Iannello, Luca Ciampi, Fabrizio Tonelli, Gabriele Lagani, Lucio Maria Calcagnile, Federico Cremisi, Angelo Di Garbo, Giuseppe Amato

In this paper, we present a biologically grounded approach to reservoir
computing (RC), in which a network of cultured biological neurons serves as the
reservoir substrate. This system, referred to as biological reservoir computing
(BRC), replaces artificial recurrent units with the spontaneous and evoked
activity of living neurons. A multi-electrode array (MEA) enables simultaneous
stimulation and readout across multiple sites: inputs are delivered through a
subset of electrodes, while the remaining ones capture the resulting neural
responses, mapping input patterns into a high-dimensional biological feature
space. We evaluate the system through a case study on digit classification
using a custom dataset. Input images are encoded and delivered to the
biological reservoir via electrical stimulation, and the corresponding neural
activity is used to train a simple linear classifier. To contextualize the
performance of the biological system, we also include a comparison with a
standard artificial reservoir trained on the same task. The results indicate
that the biological reservoir can effectively support classification,
highlighting its potential as a viable and interpretable computational
substrate. We believe this work contributes to the broader effort of
integrating biological principles into machine learning and aligns with the
goals of human-inspired vision by exploring how living neural systems can
inform the design of efficient and biologically plausible models.

### Networking and Internet Architecture

### 1. [On Enhancing Delay SLAs in TCP Networks through Joint Routing and Transport Assistant Deployment](http://arxiv.org/pdf/2510.05686v1)

Authors: José Gómez-delaHiz, Mohamed Faten Zhani, Jaime Galán-Jiménez, John Kaippallimalil

The Transport Control Protocol has long been the primary transport protocol
for applications requiring performance and reliability over the Internet.
Unfortunately, due its retransmission mechanism, TCP incurs high packet
delivery delays when segments are lost. To address this issue, previous
research proposed to use a novel network function, namely Transport Assistant,
deployed within the network to cache and retransmit lost packets, thus reducing
retransmission delays. In this paper, we propose to jointly route the flows and
deploy TAs in order to minimize packet delivery delays in best-effort networks
(scenario 1) or to satisfy delay-based Service Level Agreements in QoS-based
networks (scenario 2). We hence formulate the joint routing and TA deployment
problem as Integer Linear Program for the two scenarios and propose a heuristic
solution for large-scale instances of the problem. Through extensive
simulations, we demonstrate the benefits of performing joint routing flows and
TA deployment in reducing packet delivery delays (up to 16.4%) while minimizing
deployment costs (up to 60.98%).

### 2. [A Deep Q-Network based power control mechanism to Minimize RLF driven Handover Failure in 5G Network](http://arxiv.org/pdf/2510.05762v1)

Authors: Kotha Kartheek, Shankar K. Ghosh, Megha Iyengar, Vinod Sharma, Souvik Deb

The impact of Radio link failure (RLF) has been largely ignored in designing
handover algorithms, although RLF is a major contributor towards causing
handover failure (HF). RLF can cause HF if it is detected during an ongoing
handover. The objective of this work is to propose an efficient power control
mechanism based on Deep Q-Network (DQN), considering handover parameters (i.e.,
time-to-preparation, time-to-execute, preparation offset, execution offset) and
radio link monitoring parameters (T310 and N310) as input. The proposed DRL
based power control algorithm decides on a possible increase of transmitting
power to avoid RLF driven HF. Simulation results show that the traditional
conditional handover, when equipped with the proposed DRL based power control
algorithm can significantly reduce both RLFs and subsequent HFs, as compared to
the existing state of the art approaches.

### 3. [Leveraging Generative AI for large-scale prediction-based networking](http://arxiv.org/pdf/2510.05797v1)

Authors: Mathias Thorsager, Israel Leyva-Mayorga, Petar Popovski

The traditional role of the network layer is to create an end-to-end route,
through which the intermediate nodes replicate and forward the packets towards
the destination. This role can be radically redefined by exploiting the power
of Generative AI (GenAI) to pivot towards a prediction-based network layer,
which addresses the problems of throughput limits and uncontrollable latency.
In the context of real-time delivery of image content, the use of GenAI-aided
network nodes has been shown to improve the flow arriving at the destination by
more than 100%. However, to successfully exploit GenAI nodes and achieve such
transition, we must provide solutions for the problems which arise as we scale
the networks to include large amounts of users and multiple data modalities
other than images. We present three directions that play a significant role in
enabling the use of GenAI as a network layer tool at a large scale. In terms of
design, we emphasize the need for initialization protocols to select the prompt
size efficiently. Next, we consider the use case of GenAI as a tool to ensure
timely delivery of data, as well as an alternative to traditional TCP
congestion control algorithms.

### 4. [Dynamic Scheduling in Fiber and Spaceborne Quantum Repeater Networks](http://arxiv.org/pdf/2510.05854v1)

Authors: Paolo Fittipaldi

The problem of scheduling in quantum networks amounts to choosing which
entanglement swapping operations to perform to better serve user demand. The
choice can be carried out following a variety of criteria (e.g. ensuring all
users are served equally vs. prioritizing specific critical applications,
adopting heuristic or optimization-based algorithms...), requiring a method to
compare different solutions and choose the most appropriate. We present a
framework to mathematically formulate the scheduling problem over quantum
networks and benchmark general quantum scheduling policies over arbitrary lossy
quantum networks. By leveraging the framework, we apply Lyapunov drift
minimization to derive a novel class of quadratic optimization based scheduling
policies, which we then analyze and compare with a Max Weight inspired linear
class. We then give an overview of the pre-existing fiber quantum simulation
tools and report on the development of numerous extensions to QuISP, an
established quantum network simulator focused on scalability and accuracy in
modeling the underlying classical network infrastructure. To integrate
satellite links in the discussion, we derive an analytical model for the
entanglement distribution rates for satellite-to-ground and
ground-satellite-ground links and discuss different quantum memory allocation
policies for the dual link case. Our findings show that classical communication
latency is a major limiting factor for satellite communication, and the effects
of physical upper bounds such as the speed of light must be taken into account
when designing quantum links, limiting the attainable rates to tens of kHz. We
conclude by summarizing our findings and highlighting the challenges that still
need to be overcome in order to study the quantum scheduling problem over fiber
and satellite quantum networks. [Abridged abstract, see PDF for full version]

### 5. [cMPI: Using CXL Memory Sharing for MPI One-Sided and Two-Sided Inter-Node Communications](http://arxiv.org/pdf/2510.05476v1)

Authors: Xi Wang, Bin Ma, Jongryool Kim, Byungil Koh, Hoshik Kim, Dong Li

Message Passing Interface (MPI) is a foundational programming model for
high-performance computing. MPI libraries traditionally employ network
interconnects (e.g., Ethernet and InfiniBand) and network protocols (e.g., TCP
and RoCE) with complex software stacks for cross-node communication. We present
cMPI, the first work to optimize MPI point-to-point communication (both
one-sided and two-sided) using CXL memory sharing on a real CXL platform,
transforming cross-node communication into memory transactions and data copies
within CXL memory, bypassing traditional network protocols. We analyze
performance across various interconnects and find that CXL memory sharing
achieves 7.2x-8.1x lower latency than TCP-based interconnects deployed in
small- and medium-scale clusters. We address challenges of CXL memory sharing
for MPI communication, including data object management over the dax
representation [50], cache coherence, and atomic operations. Overall, cMPI
outperforms TCP over standard Ethernet NIC and high-end SmartNIC by up to 49x
and 72x in latency and bandwidth, respectively, for small messages.

### 6. [Generative AI-Driven Hierarchical Multi-Agent Framework for Zero-Touch Optical Networks](http://arxiv.org/pdf/2510.05625v1)

Authors: Yao Zhang, Yuchen Song, Shengnan Li, Yan Shi, Shikui Shen, Xiongyan Tang, Min Zhang, Danshi Wang

The rapid development of Generative Artificial Intelligence (GenAI) has
catalyzed a transformative technological revolution across all walks of life.
As the backbone of wideband communication, optical networks are expecting
high-level autonomous operation and zero-touch management to accommodate their
expanding network scales and escalating transmission bandwidth. The integration
of GenAI is deemed as the pivotal solution for realizing zero-touch optical
networks. However, the lifecycle management of optical networks involves a
multitude of tasks and necessitates seamless collaboration across multiple
layers, which poses significant challenges to the existing single-agent GenAI
systems. In this paper, we propose a GenAI-driven hierarchical multi-agent
framework designed to streamline multi-task autonomous execution for zero-touch
optical networks. We present the architecture, implementation, and applications
of this framework. A field-deployed mesh network is utilized to demonstrate
three typical scenarios throughout the lifecycle of optical network: quality of
transmission estimation in the planning stage, dynamic channel adding/dropping
in the operation stage, and system capacity increase in the upgrade stage. The
case studies, illustrate the capabilities of multi-agent framework in
multi-task allocation, coordination, execution, evaluation, and summarization.
This work provides a promising approach for the future development of
intelligent, efficient, and collaborative network management solutions, paving
the way for more specialized and adaptive zero-touch optical networks.

### Robotics

### 1. [Correlation-Aware Dual-View Pose and Velocity Estimation for Dynamic Robotic Manipulation](http://arxiv.org/pdf/2510.05536v1)

Authors: Mahboubeh Zarei, Robin Chhabra, Farrokh Janabi-Sharifi

Accurate pose and velocity estimation is essential for effective spatial task
planning in robotic manipulators. While centralized sensor fusion has
traditionally been used to improve pose estimation accuracy, this paper
presents a novel decentralized fusion approach to estimate both pose and
velocity. We use dual-view measurements from an eye-in-hand and an eye-to-hand
vision sensor configuration mounted on a manipulator to track a target object
whose motion is modeled as random walk (stochastic acceleration model). The
robot runs two independent adaptive extended Kalman filters formulated on a
matrix Lie group, developed as part of this work. These filters predict poses
and velocities on the manifold $\mathbb{SE}(3) \times \mathbb{R}^3 \times
\mathbb{R}^3$ and update the state on the manifold $\mathbb{SE}(3)$. The final
fused state comprising the fused pose and velocities of the target is obtained
using a correlation-aware fusion rule on Lie groups. The proposed method is
evaluated on a UFactory xArm 850 equipped with Intel RealSense cameras,
tracking a moving target. Experimental results validate the effectiveness and
robustness of the proposed decentralized dual-view estimation framework,
showing consistent improvements over state-of-the-art methods.

### 2. [ARRC: Advanced Reasoning Robot Control - Knowledge-Driven Autonomous Manipulation Using Retrieval-Augmented Generation](http://arxiv.org/pdf/2510.05547v1)

Authors: Eugene Vorobiov, Ammar Jaleel Mahmood, Salim Rezvani, Robin Chhabra

We present ARRC (Advanced Reasoning Robot Control), a practical system that
connects natural-language instructions to safe local robotic control by
combining Retrieval-Augmented Generation (RAG) with RGB-D perception and
guarded execution on an affordable robot arm. The system indexes curated robot
knowledge (movement patterns, task templates, and safety heuristics) in a
vector database, retrieves task-relevant context for each instruction, and
conditions a large language model (LLM) to produce JSON-structured action
plans. Plans are executed on a UFactory xArm 850 fitted with a Dynamixel-driven
parallel gripper and an Intel RealSense D435 camera. Perception uses AprilTag
detections fused with depth to produce object-centric metric poses. Execution
is enforced via software safety gates: workspace bounds, speed and force caps,
timeouts, and bounded retries. We describe the architecture, knowledge design,
integration choices, and a reproducible evaluation protocol for tabletop scan,
approach, and pick-place tasks. Experimental results demonstrate the efficacy
of the proposed approach. Our design shows that RAG-based planning can
substantially improve plan validity and adaptability while keeping perception
and low-level control local to the robot.

### 3. [Precise and Efficient Collision Prediction under Uncertainty in Autonomous Driving](http://arxiv.org/pdf/2510.05729v1)

Authors: Marc Kaufeld, Johannes Betz

This research introduces two efficient methods to estimate the collision risk
of planned trajectories in autonomous driving under uncertain driving
conditions. Deterministic collision checks of planned trajectories are often
inaccurate or overly conservative, as noisy perception, localization errors,
and uncertain predictions of other traffic participants introduce significant
uncertainty into the planning process. This paper presents two semi-analytic
methods to compute the collision probability of planned trajectories with
arbitrary convex obstacles. The first approach evaluates the probability of
spatial overlap between an autonomous vehicle and surrounding obstacles, while
the second estimates the collision probability based on stochastic boundary
crossings. Both formulations incorporate full state uncertainties, including
position, orientation, and velocity, and achieve high accuracy at computational
costs suitable for real-time planning. Simulation studies verify that the
proposed methods closely match Monte Carlo results while providing significant
runtime advantages, enabling their use in risk-aware trajectory planning. The
collision estimation methods are available as open-source software:
https://github.com/TUM-AVS/Collision-Probability-Estimation

### 4. [A Co-Design Framework for Energy-Aware Monoped Jumping with Detailed Actuator Modeling](http://arxiv.org/pdf/2510.05923v1)

Authors: Aman Singh, Aastha Mishra, Deepak Kapa, Suryank Joshi, Shishir Kolathaya

A monoped's jump height and energy consumption depend on both, its mechanical
design and control strategy. Existing co-design frameworks typically optimize
for either maximum height or minimum energy, neglecting their trade-off. They
also often omit gearbox parameter optimization and use oversimplified actuator
mass models, producing designs difficult to replicate in practice. In this
work, we introduce a novel three-stage co-design optimization framework that
jointly maximizes jump height while minimizing mechanical energy consumption of
a monoped. The proposed method explicitly incorporates realistic actuator mass
models and optimizes mechanical design (including gearbox) and control
parameters within a unified framework. The resulting design outputs are then
used to automatically generate a parameterized CAD model suitable for direct
fabrication, significantly reducing manual design iterations. Our experimental
evaluations demonstrate a 50 percent reduction in mechanical energy consumption
compared to the baseline design, while achieving a jump height of 0.8m. Video
presentation is available at http://y2u.be/XW8IFRCcPgM

### 5. [Learning to Crawl: Latent Model-Based Reinforcement Learning for Soft Robotic Adaptive Locomotion](http://arxiv.org/pdf/2510.05957v1)

Authors: Vaughn Gzenda, Robin Chhabra

Soft robotic crawlers are mobile robots that utilize soft body deformability
and compliance to achieve locomotion through surface contact. Designing control
strategies for such systems is challenging due to model inaccuracies, sensor
noise, and the need to discover locomotor gaits. In this work, we present a
model-based reinforcement learning (MB-RL) framework in which latent dynamics
inferred from onboard sensors serve as a predictive model that guides an
actor-critic algorithm to optimize locomotor policies. We evaluate the
framework on a minimal crawler model in simulation using inertial measurement
units and time-of-flight sensors as observations. The learned latent dynamics
enable short-horizon motion prediction while the actor-critic discovers
effective locomotor policies. This approach highlights the potential of
latent-dynamics MB-RL for enabling embodied soft robotic adaptive locomotion
based solely on noisy sensor feedback.

### 6. [The DISTANT Design for Remote Transmission and Steering Systems for Planetary Robotics](http://arxiv.org/pdf/2510.05981v1)

Authors: Cristina Luna, Alba Guerra, Almudena Moreno, Manuel Esquer, Willy Roa, Mateusz Krawczak, Robert Popela, Piotr Osica, Davide Nicolis

Planetary exploration missions require robust locomotion systems capable of
operating in extreme environments over extended periods. This paper presents
the DISTANT (Distant Transmission and Steering Systems) design, a novel
approach for relocating rover traction and steering actuators from
wheel-mounted positions to a thermally protected warm box within the rover
body. The design addresses critical challenges in long-distance traversal
missions by protecting sensitive components from thermal cycling, dust
contamination, and mechanical wear. A double wishbone suspension configuration
with cardan joints and capstan drive steering has been selected as the optimal
architecture following comprehensive trade-off analysis. The system enables
independent wheel traction, steering control, and suspension management whilst
maintaining all motorisation within the protected environment. The design meets
a 50 km traverse requirement without performance degradation, with integrated
dust protection mechanisms and thermal management solutions. Testing and
validation activities are planned for Q1 2026 following breadboard
manufacturing at 1:3 scale.

### 7. [AI-Enabled Capabilities to Facilitate Next-Generation Rover Surface Operations](http://arxiv.org/pdf/2510.05985v1)

Authors: Cristina Luna, Robert Field, Steven Kay

Current planetary rovers operate at traverse speeds of approximately 10 cm/s,
fundamentally limiting exploration efficiency. This work presents integrated AI
systems which significantly improve autonomy through three components: (i) the
FASTNAV Far Obstacle Detector (FOD), capable of facilitating sustained 1.0 m/s
speeds via computer vision-based obstacle detection; (ii) CISRU, a multi-robot
coordination framework enabling human-robot collaboration for in-situ resource
utilisation; and (iii) the ViBEKO and AIAXR deep learning-based terrain
classification studies. Field validation in Mars analogue environments
demonstrated these systems at Technology Readiness Level 4, providing
measurable improvements in traverse speed, classification accuracy, and
operational safety for next-generation planetary missions.

### 8. [Coordinate-Consistent Localization via Continuous-Time Calibration and Fusion of UWB and SLAM Observations](http://arxiv.org/pdf/2510.05992v1)

Authors: Tien-Dat Nguyen, Thien-Minh Nguyen, Vinh-Hao Nguyen

Onboard simultaneous localization and mapping (SLAM) methods are commonly
used to provide accurate localization information for autonomous robots.
However, the coordinate origin of SLAM estimate often resets for each run. On
the other hand, UWB-based localization with fixed anchors can ensure a
consistent coordinate reference across sessions; however, it requires an
accurate assignment of the anchor nodes' coordinates. To this end, we propose a
two-stage approach that calibrates and fuses UWB data and SLAM data to achieve
coordinate-wise consistent and accurate localization in the same environment.
In the first stage, we solve a continuous-time batch optimization problem by
using the range and odometry data from one full run, incorporating height
priors and anchor-to-anchor distance factors to recover the anchors' 3D
positions. For the subsequent runs in the second stage, a sliding-window
optimization scheme fuses the UWB and SLAM data, which facilitates accurate
localization in the same coordinate system. Experiments are carried out on the
NTU VIRAL dataset with six scenarios of UAV flight, and we show that
calibration using data in one run is sufficient to enable accurate localization
in the remaining runs. We release our source code to benefit the community at
https://github.com/ntdathp/slam-uwb-calibration.

### 9. [Multi-Robot Distributed Optimization for Exploration and Mapping of Unknown Environments using Bioinspired Tactile-Sensor](http://arxiv.org/pdf/2510.06085v1)

Authors: Roman Ibrahimov, Jannik Matthias Heinen

This project proposes a bioinspired multi-robot system using Distributed
Optimization for efficient exploration and mapping of unknown environments.
Each robot explores its environment and creates a map, which is afterwards put
together to form a global 2D map of the environment. Inspired by wall-following
behaviors, each robot autonomously explores its neighborhood based on a tactile
sensor, similar to the antenna of a cockroach, mounted on the surface of the
robot. Instead of avoiding obstacles, robots log collision points when they
touch obstacles. This decentralized control strategy ensures effective task
allocation and efficient exploration of unknown terrains, with applications in
search and rescue, industrial inspection, and environmental monitoring. The
approach was validated through experiments using e-puck robots in a simulated
1.5 x 1.5 m environment with three obstacles. The results demonstrated the
system's effectiveness in achieving high coverage, minimizing collisions, and
constructing accurate 2D maps.

### 10. [Towards Autonomous Tape Handling for Robotic Wound Redressing](http://arxiv.org/pdf/2510.06127v1)

Authors: Xiao Liang, Lu Shen, Peihan Zhang, Soofiyan Atar, Florian Richter, Michael Yip

Chronic wounds, such as diabetic, pressure, and venous ulcers, affect over
6.5 million patients in the United States alone and generate an annual cost
exceeding \$25 billion. Despite this burden, chronic wound care remains a
routine yet manual process performed exclusively by trained clinicians due to
its critical safety demands. We envision a future in which robotics and
automation support wound care to lower costs and enhance patient outcomes. This
paper introduces an autonomous framework for one of the most fundamental yet
challenging subtasks in wound redressing: adhesive tape manipulation.
Specifically, we address two critical capabilities: tape initial detachment
(TID) and secure tape placement. To handle the complex adhesive dynamics of
detachment, we propose a force-feedback imitation learning approach trained
from human teleoperation demonstrations. For tape placement, we develop a
numerical trajectory optimization method based to ensure smooth adhesion and
wrinkle-free application across diverse anatomical surfaces. We validate these
methods through extensive experiments, demonstrating reliable performance in
both quantitative evaluations and integrated wound redressing pipelines. Our
results establish tape manipulation as an essential step toward practical
robotic wound care automation.

### Software Engineering

### 1. [An Empirical Study of Security-Policy Related Issues in Open Source Projects](http://arxiv.org/pdf/2510.05604v1)

Authors: Rintaro Kanaji, Brittany Reid, Yutaro Kashiwa, Raula Gaikovina Kula, Hajimu Iida

GitHub recommends that projects adopt a SECURITY.md file that outlines
vulnerability reporting procedures. However, the effectiveness and operational
challenges of such files are not yet fully understood. This study aims to
clarify the challenges that SECURITY.md files face in the vulnerability
reporting process within open-source communities. Specifically, we classified
and analyzed the content of 711 randomly sampled issues related to SECURITY.md.
We also conducted a quantitative comparative analysis of the close time and
number of responses for issues concerning six community health files, including
SECURITY.md. Our analysis revealed that 79.5% of SECURITY.md-related issues
were requests to add the file, and reports that included links were closed,
with a median time that was 2 days shorter. These findings offer practical
insights for improving security reporting policies and community management,
ultimately contributing to a more secure open-source ecosystem.

### 2. [Digital Twins for Software Engineering Processes](http://arxiv.org/pdf/2510.05768v1)

Authors: Robin Kimmel, Judith Michael, Andreas Wortmann, Jingxi Zhang

Digital twins promise a better understanding and use of complex systems. To
this end, they represent these systems at their runtime and may interact with
them to control their processes. Software engineering is a wicked challenge in
which stakeholders from many domains collaborate to produce software artifacts
together. In the presence of skilled software engineer shortage, our vision is
to leverage DTs as means for better rep- resenting, understanding, and
optimizing software engineering processes to (i) enable software experts making
the best use of their time and (ii) support domain experts in producing
high-quality software. This paper outlines why this would be beneficial, what
such a digital twin could look like, and what is missing for realizing and
deploying software engineering digital twins.

### 3. [A Wave of Resignations in the Aftermath of Remote Onboarding](http://arxiv.org/pdf/2510.05878v1)

Authors: Darja Smite, Franz Zieris, Lars-Ola Damm

The COVID-19 pandemic has permanently altered workplace structures,
normalizing remote work. However, critical evidence highlights challenges with
fully remote arrangements, particularly for software teams. This study
investigates employee resignation patterns at Ericsson, a global developer of
software-intensive systems, before, during, and after the pandemic. Using HR
data from 2016-2025 in Ericsson Sweden, we analyze how different work
modalities (onsite, remote, and hybrid) influence employee retention. Our
findings show a marked increase in resignations from summer 2021 to summer
2023, especially among employees with less than five years of tenure. Employees
onboarded remotely during the pandemic were significantly more likely to resign
within their first three years, even after returning to the office. Exit
surveys suggest that remote onboarding may fail to establish the necessary
organizational attachment, the feeling of belonging and long-term retention. By
contrast, the company's eventual successful return to pre-pandemic retention
rates illustrates the value of differentiated work policies and supports
reconsidering selective return-to-office (RTO) mandates. Our study demonstrates
the importance of employee integration practices in hybrid environments where
the requirement for in-office presence for recent hires shall be accompanied by
in-office presence from their team members and more senior staff whose
mentoring and social interactions contribute to integration into the corporate
work environment. We hope these actionable insights will inform HR leaders and
policymakers in shaping post-pandemic work practices, demonstrating that
carefully crafted hybrid models anchored in organizational attachment and
mentorship can sustain retention in knowledge-intensive companies.

### 4. [Extending ResourceLink: Patterns for Large Dataset Processing in MCP Applications](http://arxiv.org/pdf/2510.05968v1)

Authors: Scott Frees

Large language models translate natural language into database queries, yet
context window limitations prevent direct deployment in reporting systems where
complete datasets exhaust available tokens. The Model Context Protocol
specification defines ResourceLink for referencing external resources, but
practical patterns for implementing scalable reporting architectures remain
undocumented. This paper presents patterns for building LLM-powered reporting
systems that decouple query generation from data retrieval. We introduce a
dual-response pattern extending ResourceLink to support both iterative query
refinement and out-of-band data access, accompanied by patterns for
multi-tenant security and resource lifecycle management. These patterns address
fundamental challenges in LLM-driven reporting applications and provide
practical guidance for developers building them.

### 5. [Prompting in Practice: Investigating Software Developers' Use of Generative AI Tools](http://arxiv.org/pdf/2510.06000v1)

Authors: Daniel Otten, Trevor Stalnaker, Nathan Wintersgill, Oscar Chaparro, Denys Poshyvanyk

The integration of generative artificial intelligence (GenAI) tools has
fundamentally transformed software development. Although prompt engineering has
emerged as a critical skill, existing research focuses primarily on individual
techniques rather than software developers' broader workflows. This study
presents a systematic investigation of how software engineers integrate GenAI
tools into their professional practice through a large-scale survey examining
prompting strategies, conversation patterns, and reliability assessments across
various software engineering tasks.
  We surveyed 91 software engineers, including 72 active GenAI users, to
understand AI usage patterns throughout the development process. Our 14 key
findings show that while code generation is nearly universal, proficiency
strongly correlates with using AI for more nuanced tasks such as debugging and
code review, and that developers prefer iterative multi-turn conversations to
single-shot prompting. Documentation tasks are perceived as most reliable,
while complex code generation and debugging present sizable challenges. Our
insights provide an empirical baseline of current developer practices, from
simple code generation to deeper workflow integration, with actionable insights
for future improvements.

### 6. [Explaining Code Risk in OSS: Towards LLM-Generated Fault Prediction Interpretations](http://arxiv.org/pdf/2510.06104v1)

Authors: Elijah Kayode Adejumo, Brittany Johnson

Open Source Software (OSS) has become a very important and crucial
infrastructure worldwide because of the value it provides. OSS typically
depends on contributions from developers across diverse backgrounds and levels
of experience. Making safe changes, such as fixing a bug or implementing a new
feature, can be challenging, especially in object-oriented systems where
components are interdependent. Static analysis and defect-prediction tools
produce metrics (e.g., complexity,coupling) that flag potentially fault-prone
components, but these signals are often hard for contributors new or unfamiliar
with the codebase to interpret. Large Language Models (LLMs) have shown strong
performance on software engineering tasks such as code summarization and
documentation generation. Building on this progress, we investigate whether
LLMs can translate fault-prediction metrics into clear, human-readable risk
explanations and actionable guidance to help OSS contributors plan and review
code modifications. We outline explanation types that an LLM-generated
assistant could provide (descriptive, contextual, and actionable explanations).
We also outline our next steps to assess usefulness through a task-based study
with OSS contributors, comparing metric-only baselines to LLM-generated
explanations on decision quality, time-to-completion, and error rates

### 7. [Vul-R2: A Reasoning LLM for Automated Vulnerability Repair](http://arxiv.org/pdf/2510.05480v1)

Authors: Xin-Cheng Wen, Zirui Lin, Yijun Yang, Cuiyun Gao, Deheng Ye

The exponential increase in software vulnerabilities has created an urgent
need for automatic vulnerability repair (AVR) solutions. Recent research has
formulated AVR as a sequence generation problem and has leveraged large
language models (LLMs) to address this problem. Typically, these approaches
prompt or fine-tune LLMs to generate repairs for vulnerabilities directly.
Although these methods show state-of-the-art performance, they face the
following challenges: (1) Lack of high-quality, vulnerability-related reasoning
data. Current approaches primarily rely on foundation models that mainly encode
general programming knowledge. Without vulnerability-related reasoning data,
they tend to fail to capture the diverse vulnerability repair patterns. (2)
Hard to verify the intermediate vulnerability repair process during LLM
training. Existing reinforcement learning methods often leverage intermediate
execution feedback from the environment (e.g., sandbox-based execution results)
to guide reinforcement learning training. In contrast, the vulnerability repair
process generally lacks such intermediate, verifiable feedback, which poses
additional challenges for model training.

### 8. [SBOMproof: Beyond Alleged SBOM Compliance for Supply Chain Security of Container Images](http://arxiv.org/pdf/2510.05798v1)

Authors: Jacopo Bufalino, Mario Di Francesco, Agathe Blaise, Stefano Secci

Supply chain security is extremely important for modern applications running
at scale in the cloud. In fact, they involve a large number of heterogeneous
microservices that also include third-party software. As a result, security
vulnerabilities are hard to identify and mitigate before they start being
actively exploited by attackers. For this reason, governments have recently
introduced cybersecurity regulations that require vendors to share a software
bill of material (SBOM) with end users or regulators. An SBOM can be employed
to identify the security vulnerabilities of a software component even without
access to its source code, as long as it is accurate and interoperable across
different tools. This work evaluates this issue through a comprehensive study
of tools for SBOM generation and vulnerability scanning, including both
open-source software and cloud services from major providers. We specifically
target software containers and focus on operating system packages in Linux
distributions that are widely used as base images due to their far-reaching
security impact. Our findings show that the considered tools are largely
incompatible, leading to inaccurate reporting and a large amount of undetected
vulnerabilities. We uncover the SBOM confusion vulnerability, a byproduct of
such fragmented ecosystem, where inconsistent formats prevent reliable
vulnerability detection across tools.

### 9. [AdProv: A Method for Provenance of Process Adaptations](http://arxiv.org/pdf/2510.05936v1)

Authors: Ludwig Stage, Mirela Riveni, Raimundas Matulevičius, Dimka Karastoyanova

Provenance in scientific workflows is essential for understand- ing and
reproducing processes, while in business processes, it can ensure compliance
and correctness and facilitates process mining. However, the provenance of
process adaptations, especially modifications during execu- tion, remains
insufficiently addressed. A review of the literature reveals a lack of
systematic approaches for capturing provenance information about adaptive
workflows/processes. To fill this gap, we propose the AdProv method for
collecting, storing, retrieving, and visualizing prove- nance of runtime
workflow adaptations. In addition to the definition of the AdProv method in
terms of steps and concepts like change events, we also present an architecture
for a Provenance Holder service that is essential for implementing the method.
To ensure semantic consistency and interoperability we define a mapping to the
ontology PROV Ontol- ogy (PROV-O). Additionally, we extend the XES standard
with elements for adaptation logging. Our main contributions are the AdProv
method and a comprehensive framework and its tool support for managing adap-
tive workflow provenance, facilitating advanced provenance tracking and
analysis for different application domains.

### 10. [The Software Observatory: aggregating and analysing software metadata for trend computation and FAIR assessment](http://arxiv.org/pdf/2510.05705v1)

Authors: Eva Martín del Pico, Josep Lluís Gelpí, Salvador Capella-Gutiérrez

In the ever-changing realm of research software development, it is crucial
for the scientific community to grasp current trends to identify gaps that can
potentially hinder scientific progress. The adherence to the FAIR (Findable,
Accessible, Interoperable, Reusable) principles can serve as a proxy to
understand those trends and provide a mechanism to propose specific actions.
  The Software Observatory at OpenEBench
(https://openebench.bsc.es/observatory) is a novel web portal that consolidates
software metadata from various sources, offering comprehensive insights into
critical research software aspects. Our platform enables users to analyse
trends, identify patterns and advancements within the Life Sciences research
software ecosystem, and understand its evolution over time. It also evaluates
research software according to FAIR principles for research software, providing
scores for different indicators.
  Users have the ability to visualise this metadata at different levels of
granularity, ranging from the entire software landscape to specific communities
to individual software entries through the FAIRsoft Evaluator. Indeed, the
FAIRsoft Evaluator component streamlines the assessment process, helping
developers efficiently evaluate and obtain guidance to improve their software's
FAIRness.
  The Software Observatory represents a valuable resource for researchers and
software developers, as well as stakeholders, promoting better software
development practices and adherence to FAIR principles for research software.

### Social and Information Networks

### 1. [Inductive inference of gradient-boosted decision trees on graphs for insurance fraud detection](http://arxiv.org/pdf/2510.05676v1)

Authors: Félix Vandervorst, Bruno Deprez, Wouter Verbeke, Tim Verdonck

Graph-based methods are becoming increasingly popular in machine learning due
to their ability to model complex data and relations. Insurance fraud is a
prime use case, since false claims are often the result of organised criminals
that stage accidents or the same persons filing erroneous claims on multiple
policies. One challenge is that graph-based approaches struggle to find
meaningful representations of the data because of the high class imbalance
present in fraud data. Another is that insurance networks are heterogeneous and
dynamic, given the changing relations among people, companies and policies.
That is why gradient boosted tree approaches on tabular data still dominate the
field. Therefore, we present a novel inductive graph gradient boosting machine
(G-GBM) for supervised learning on heterogeneous and dynamic graphs. We show
that our estimator competes with popular graph neural network approaches in an
experiment using a variety of simulated random graphs. We demonstrate the power
of G-GBM for insurance fraud detection using an open-source and a real-world,
proprietary dataset. Given that the backbone model is a gradient boosting
forest, we apply established explainability methods to gain better insights
into the predictions made by G-GBM.

### 2. [Emergent Directedness in Social Contagion](http://arxiv.org/pdf/2510.06012v1)

Authors: Fabian Tschofenig, Douglas Guilbeault

An enduring challenge in contagion theory is that the pathways contagions
follow through social networks exhibit emergent complexities that are difficult
to predict using network structure. Here, we address this challenge by
developing a causal modeling framework that (i) simulates the possible network
pathways that emerge as contagions spread and (ii) identifies which edges and
nodes are most impactful on diffusion across these possible pathways. This
yields a surprising discovery. If people require exposure to multiple peers to
adopt a contagion (a.k.a., 'complex contagions'), the pathways that emerge
often only work in one direction. In fact, the more complex a contagion is, the
more asymmetric its paths become. This emergent directedness problematizes
canonical theories of how networks mediate contagion. Weak ties spanning
network regions - widely thought to facilitate mutual influence and integration
- prove to privilege the spread contagions from one community to the other.
Emergent directedness also disproportionately channels complex contagions from
the network periphery to the core, inverting standard centrality models. We
demonstrate two practical applications. We show that emergent directedness
accounts for unexplained nonlinearity in the effects of tie strength in a
recent study of job diffusion over LinkedIn. Lastly, we show that network
evolution is biased toward growing directed paths, but that cultural factors
(e.g., triadic closure) can curtail this bias, with strategic implications for
network building and behavioral interventions.

### Systems and Control

### 1. [Sample-Efficient and Smooth Cross-Entropy Method Model Predictive Control Using Deterministic Samples](http://arxiv.org/pdf/2510.05706v1)

Authors: Markus Walker, Daniel Frisch, Uwe D. Hanebeck

Cross-entropy method model predictive control (CEM--MPC) is a powerful
gradient-free technique for nonlinear optimal control, but its performance is
often limited by the reliance on random sampling. This conventional approach
can lead to inefficient exploration of the solution space and non-smooth
control inputs, requiring a large number of samples to achieve satisfactory
results. To address these limitations, we propose deterministic sampling CEM
(dsCEM), a novel framework that replaces the random sampling step with
deterministic samples derived from localized cumulative distributions (LCDs).
Our approach introduces modular schemes to generate and adapt these sample
sets, incorporating temporal correlations to ensure smooth control
trajectories. This method can be used as a drop-in replacement for the sampling
step in existing CEM-based controllers. Experimental evaluations on two
nonlinear control tasks demonstrate that dsCEM consistently outperforms
state-of-the-art iCEM in terms of cumulative cost and control input smoothness,
particularly in the critical low-sample regime.

### 2. [Distributed Platoon Control Under Quantization: Stability Analysis and Privacy Preservation](http://arxiv.org/pdf/2510.05959v1)

Authors: Kaixiang Zhang, Zhaojian Li, Wei Lin

Distributed control of connected and automated vehicles has attracted
considerable interest for its potential to improve traffic efficiency and
safety. However, such control schemes require sharing privacy-sensitive vehicle
data, which introduces risks of information leakage and potential malicious
activities. This paper investigates the stability and privacy-preserving
properties of distributed platoon control under two types of quantizers:
deterministic and probabilistic. For deterministic quantization, we show that
the resulting control strategy ensures the system errors remain uniformly
ultimately bounded. Moreover, in the absence of auxiliary information, an
eavesdropper cannot uniquely infer sensitive vehicle states. In contrast, the
use of probabilistic quantization enables asymptotic convergence of the vehicle
platoon in expectation with bounded variance. Importantly, probabilistic
quantizers can satisfy differential privacy guarantees, thereby preserving
privacy even when the eavesdropper possesses arbitrary auxiliary information.
We further analyze the trade-off between control performance and privacy by
formulating an optimization problem that characterizes the impact of the
quantization step on both metrics. Numerical simulations are provided to
illustrate the performance differences between the two quantization strategies.

### 3. [Optimal Batched Scheduling of Stochastic Processing Networks Using Atomic Action Decomposition](http://arxiv.org/pdf/2510.06033v1)

Authors: Jim Dai, Manxi Wu, Zhanhao Zhang

Stochastic processing networks (SPNs) have broad applications in healthcare,
transportation, and communication networks. The control of SPN is to
dynamically assign servers in batches under uncertainty to optimize long-run
performance. This problem is challenging as the policy dimension grows
exponentially with the number of servers, making standard reinforcement
learning and policy optimization methods intractable at scale. We propose an
atomic action decomposition framework that addresses this scalability challenge
by breaking joint assignments into sequential single-server assignments. This
yields policies with constant dimension, independent of the number of servers.
We study two classes of atomic policies, the step-dependent and
step-independent atomic policies, and prove that both achieve the same optimal
long-run average reward as the original joint policies. These results establish
that computing the optimal SPN control can be made scalable without loss of
optimality using the atomic framework. Our results offer theoretical
justification for the strong empirical success of the atomic framework in
large-scale applications reported in previous articles.

### 4. [Toward Model Matching for Remotely Controlled Differential Drive Robotic Vehicles](http://arxiv.org/pdf/2510.06081v1)

Authors: Nikolaos D. Kouvakas, Fotis N. Koumboulis, Konstantinos G. Tzierakis, John Sigalas, Anastasios Dimakakos

The problem of regulation of the orientation angle of a remotely controlled
differential-drive mobile robot with actuator dynamics and network-induced
delays is studied. Using a preinstalled two-layer nonlinear control scheme that
decouples linear and angular velocities and regulates heading, a third,
delay-dependent layer that achieves exact model matching from the orientation
angle command to the orientation angle is introduced. The proposed outer loop
controller is a delay dependent dynamic measurable output-feedback controller
with dynamic proper precompensator. Parameterization yields a simple
characteristic quasi-polynomial with coefficients constrained to satisfy
stability for all delays up to a computable bound. Computational experiments
confirm accurate tracking, fast settling and bounded internal signals and
control voltages. The approach offers an analytic design alternative to
AI-based tuning for delayed robotic systems.

### 5. [GO-Flock: Goal-Oriented Flocking in 3D Unknown Environments with Depth Maps](http://arxiv.org/pdf/2510.05553v1)

Authors: Yan Rui Tan, Wenqi Liu, Wai Lun Leong, John Guan Zhong Tan, Wayne Wen Huei Yong, Fan Shi, Rodney Swee Huat Teo

Artificial Potential Field (APF) methods are widely used for reactive
flocking control, but they often suffer from challenges such as deadlocks and
local minima, especially in the presence of obstacles. Existing solutions to
address these issues are typically passive, leading to slow and inefficient
collective navigation. As a result, many APF approaches have only been
validated in obstacle-free environments or simplified, pseudo 3D simulations.
This paper presents GO-Flock, a hybrid flocking framework that integrates
planning with reactive APF-based control. GO-Flock consists of an upstream
Perception Module, which processes depth maps to extract waypoints and virtual
agents for obstacle avoidance, and a downstream Collective Navigation Module,
which applies a novel APF strategy to achieve effective flocking behavior in
cluttered environments. We evaluate GO-Flock against passive APF-based
approaches to demonstrate their respective merits, such as their flocking
behavior and the ability to overcome local minima. Finally, we validate
GO-Flock through obstacle-filled environment and also hardware-in-the-loop
experiments where we successfully flocked a team of nine drones, six physical
and three virtual, in a forest environment.

### 6. [Human-in-the-loop Optimisation in Robot-assisted Gait Training](http://arxiv.org/pdf/2510.05780v1)

Authors: Andreas Christou, Andreas Sochopoulos, Elliot Lister, Sethu Vijayakumar

Wearable robots offer a promising solution for quantitatively monitoring gait
and providing systematic, adaptive assistance to promote patient independence
and improve gait. However, due to significant interpersonal and intrapersonal
variability in walking patterns, it is important to design robot controllers
that can adapt to the unique characteristics of each individual. This paper
investigates the potential of human-in-the-loop optimisation (HILO) to deliver
personalised assistance in gait training. The Covariance Matrix Adaptation
Evolution Strategy (CMA-ES) was employed to continuously optimise an
assist-as-needed controller of a lower-limb exoskeleton. Six healthy
individuals participated over a two-day experiment. Our results suggest that
while the CMA-ES appears to converge to a unique set of stiffnesses for each
individual, no measurable impact on the subjects' performance was observed
during the validation trials. These findings highlight the impact of
human-robot co-adaptation and human behaviour variability, whose effect may be
greater than potential benefits of personalising rule-based assistive
controllers. Our work contributes to understanding the limitations of current
personalisation approaches in exoskeleton-assisted gait rehabilitation and
identifies key challenges for effective implementation of human-in-the-loop
optimisation in this domain.

### 7. [Safe Landing on Small Celestial Bodies with Gravitational Uncertainty Using Disturbance Estimation and Control Barrier Functions](http://arxiv.org/pdf/2510.05895v1)

Authors: Felipe Arenas-Uribe, T. Michael Seigler, Jesse B. Hoagg

Soft landing on small celestial bodies (SCBs) poses unique challenges, as
uncertainties in gravitational models and poorly characterized, dynamic
environments require a high level of autonomy. Existing control approaches lack
formal guarantees for safety constraint satisfaction, necessary to ensure the
safe execution of the maneuvers. This paper introduces a control that addresses
this limitation by integrating trajectory tracking, disturbance estimation, and
safety enforcement. An extended high-gain observer is employed to estimate
disturbances resulting from gravitational model uncertainties. We then apply a
feedback-linearizing and disturbance-canceling controller that achieves
exponential tracking of reference trajectories. Finally, we use a control
barrier function based minimum-intervention controller to enforce state and
input constraints through out the maneuver execution. This control combines
trajectory tracking of offline generated reference trajectories with formal
guarantees of safety, which follows common guidance and control architectures
for spacecraft and allows aggressive maneuvers to be executed without
compromising safety. Numerical simulations using fuel-optimal trajectories
demonstrate the effectiveness of the controller in achieving precise and safe
soft-landing, highlighting its potential for autonomous SCB missions.

### 8. [Differentiable Model Predictive Control on the GPU](http://arxiv.org/pdf/2510.06179v1)

Authors: Emre Adabag, Marcus Greiff, John Subosits, Thomas Lew

Differentiable model predictive control (MPC) offers a powerful framework for
combining learning and control. However, its adoption has been limited by the
inherently sequential nature of traditional optimization algorithms, which are
challenging to parallelize on modern computing hardware like GPUs. In this
work, we tackle this bottleneck by introducing a GPU-accelerated differentiable
optimization tool for MPC. This solver leverages sequential quadratic
programming and a custom preconditioned conjugate gradient (PCG) routine with
tridiagonal preconditioning to exploit the problem's structure and enable
efficient parallelization. We demonstrate substantial speedups over CPU- and
GPU-based baselines, significantly improving upon state-of-the-art training
times on benchmark reinforcement learning and imitation learning tasks.
Finally, we showcase the method on the challenging task of reinforcement
learning for driving at the limits of handling, where it enables robust
drifting of a Toyota Supra through water puddles.

### 9. [Multi-Segment Photonic Power Converters for Energy Harvesting and High-Speed Optical Wireless Communication](http://arxiv.org/pdf/2510.06205v1)

Authors: Othman Younus, Behnaz Majlesein, Richard Nacke, Isaac N. O. Osahon, Carmine Pellegrino, Sina Babadi, Iman Tavakkolnia, Henning Helmers, Harald Haas

The demand for energy-efficient high-speed wireless communication, coupled
with the rapid rise of IoT devices, requires systems that integrate power
harvesting with optical data reception to eliminate the need for charging or
battery replacements. Recent advances have explored the use of solar cells as
optical receivers for high-speed data detection alongside power harvesting.
\acs{GaAs}-based \acp{PPC} provide six times greater electron mobility than
silicon- or cadmium telluride-based cells, enabling faster data detection and
improved power efficiency. However, their bandwidth is constrained by junction
capacitance, which increases with active area, creating a trade-off between
power output and data rate. To address this, we propose and test multi-segment
\acs{GaAs}-based \Acp{PPC} that serve as both energy harvesters and data
detectors. By segmenting the active area into 2, 4, or 6 subcells, forming
circular areas with diameters of 1, 1.5, or 2.08~mm, we reduce capacitance and
boost bandwidth while preserving light collection. Fabricated on a
semi-insulating \ac{GaAs} substrate with etched trenches for electrical
isolation, the series-connected subcells optimize absorption and minimize
parasitic effects. The \Acp{PPC} were used for an eye-safe 1.5~m optical
wireless link, employing \ac{OFDM} with adaptive bit and power loading. The
system achieved a world record data rate of 3.8~Gbps, which is four times
higher than prior works. The system converts 39.7\% of optical power from a
beam of 2.3~mW, although the segmentation increases the sensitivity of the
alignment. These findings provide new solutions for off-grid backhaul for
future communication networks, such as 6th generation (6G) cellular.

### 10. [Federated Split Learning for Resource-Constrained Robots in Industrial IoT: Framework Comparison, Optimization Strategies, and Future Directions](http://arxiv.org/pdf/2510.05713v1)

Authors: Wanli Ni, Hui Tian, Shuai Wang, Chengyang Li, Lei Sun, Zhaohui Yang

Federated split learning (FedSL) has emerged as a promising paradigm for
enabling collaborative intelligence in industrial Internet of Things (IoT)
systems, particularly in smart factories where data privacy, communication
efficiency, and device heterogeneity are critical concerns. In this article, we
present a comprehensive study of FedSL frameworks tailored for
resource-constrained robots in industrial scenarios. We compare synchronous,
asynchronous, hierarchical, and heterogeneous FedSL frameworks in terms of
workflow, scalability, adaptability, and limitations under dynamic industrial
conditions. Furthermore, we systematically categorize token fusion strategies
into three paradigms: input-level (pre-fusion), intermediate-level
(intra-fusion), and output-level (post-fusion), and summarize their respective
strengths in industrial applications. We also provide adaptive optimization
techniques to enhance the efficiency and feasibility of FedSL implementation,
including model compression, split layer selection, computing frequency
allocation, and wireless resource management. Simulation results validate the
performance of these frameworks under industrial detection scenarios. Finally,
we outline open issues and research directions of FedSL in future smart
manufacturing systems.

### Machine Learning (Statistics Category)

### 1. [Smart Contract Adoption under Discrete Overdispersed Demand: A Negative Binomial Optimization Perspective](http://arxiv.org/pdf/2510.05487v1)

Authors: Jinho Cha, Sahng-Min Han, Long Pham

Effective supply chain management under high-variance demand requires models
that jointly address demand uncertainty and digital contracting adoption.
Existing research often simplifies demand variability or treats adoption as an
exogenous decision, limiting relevance in e-commerce and humanitarian
logistics. This study develops an optimization framework combining dynamic
Negative Binomial (NB) demand modeling with endogenous smart contract adoption.
The NB process incorporates autoregressive dynamics in success probability to
capture overdispersion and temporal correlation. Simulation experiments using
four real-world datasets, including Delhivery Logistics and the SCMS Global
Health Delivery system, apply maximum likelihood estimation and grid search to
optimize adoption intensity and order quantity. Across all datasets, the NB
specification outperforms Poisson and Gaussian benchmarks, with overdispersion
indices exceeding 1.5. Forecasting comparisons show that while ARIMA and
Exponential Smoothing achieve similar point accuracy, the NB model provides
superior stability under high variance. Scenario analysis reveals that when
dispersion exceeds a critical threshold (r > 6), increasing smart contract
adoption above 70% significantly enhances profitability and service levels.
This framework offers actionable guidance for balancing inventory costs,
service levels, and implementation expenses, highlighting the importance of
aligning digital adoption strategies with empirically observed demand
volatility.

### 2. [ESS-Flow: Training-free guidance of flow-based models as inference in source space](http://arxiv.org/pdf/2510.05849v1)

Authors: Adhithyan Kalaivanan, Zheng Zhao, Jens Sjölund, Fredrik Lindsten

Guiding pretrained flow-based generative models for conditional generation or
to produce samples with desired target properties enables solving diverse tasks
without retraining on paired data. We present ESS-Flow, a gradient-free method
that leverages the typically Gaussian prior of the source distribution in
flow-based models to perform Bayesian inference directly in the source space
using Elliptical Slice Sampling. ESS-Flow only requires forward passes through
the generative model and observation process, no gradient or Jacobian
computations, and is applicable even when gradients are unreliable or
unavailable, such as with simulation-based observations or quantization in the
generation or observation process. We demonstrate its effectiveness on
designing materials with desired target properties and predicting protein
structures from sparse inter-residue distance measurements.

### 3. [Out-of-Distribution Detection from Small Training Sets using Bayesian Neural Network Classifiers](http://arxiv.org/pdf/2510.06025v1)

Authors: Kevin Raina, Tanya Schmah

Out-of-Distribution (OOD) detection is critical to AI reliability and safety,
yet in many practical settings, only a limited amount of training data is
available. Bayesian Neural Networks (BNNs) are a promising class of model on
which to base OOD detection, because they explicitly represent epistemic (i.e.
model) uncertainty. In the small training data regime, BNNs are especially
valuable because they can incorporate prior model information. We introduce a
new family of Bayesian posthoc OOD scores based on expected logit vectors, and
compare 5 Bayesian and 4 deterministic posthoc OOD scores. Experiments on MNIST
and CIFAR-10 In-Distributions, with 5000 training samples or less, show that
the Bayesian methods outperform corresponding deterministic methods.

### 4. [Generalization of Gibbs and Langevin Monte Carlo Algorithms in the Interpolation Regime](http://arxiv.org/pdf/2510.06028v1)

Authors: Andreas Maurer, Erfan Mirzaei, Massimiliano Pontil

The paper provides data-dependent bounds on the test error of the Gibbs
algorithm in the overparameterized interpolation regime, where low training
errors are also obtained for impossible data, such as random labels in
classification. The bounds are stable under approximation with Langevin Monte
Carlo algorithms. Experiments on the MNIST and CIFAR-10 datasets verify that
the bounds yield nontrivial predictions on true labeled data and correctly
upper bound the test error for random labels. Our method indicates that
generalization in the low-temperature, interpolation regime is already signaled
by small training errors in the more classical high temperature regime.

### 5. [PolyGraph Discrepancy: a classifier-based metric for graph generation](http://arxiv.org/pdf/2510.06122v1)

Authors: Markus Krimmel, Philip Hartout, Karsten Borgwardt, Dexiong Chen

Existing methods for evaluating graph generative models primarily rely on
Maximum Mean Discrepancy (MMD) metrics based on graph descriptors. While these
metrics can rank generative models, they do not provide an absolute measure of
performance. Their values are also highly sensitive to extrinsic parameters,
namely kernel and descriptor parametrization, making them incomparable across
different graph descriptors. We introduce PolyGraph Discrepancy (PGD), a new
evaluation framework that addresses these limitations. It approximates the
Jensen-Shannon distance of graph distributions by fitting binary classifiers to
distinguish between real and generated graphs, featurized by these descriptors.
The data log-likelihood of these classifiers approximates a variational lower
bound on the JS distance between the two distributions. Resulting metrics are
constrained to the unit interval [0,1] and are comparable across different
graph descriptors. We further derive a theoretically grounded summary metric
that combines these individual metrics to provide a maximally tight lower bound
on the distance for the given descriptors. Thorough experiments demonstrate
that PGD provides a more robust and insightful evaluation compared to MMD
metrics. The PolyGraph framework for benchmarking graph generative models is
made publicly available at https://github.com/BorgwardtLab/polygraph-benchmark.

### 6. [Implicit Updates for Average-Reward Temporal Difference Learning](http://arxiv.org/pdf/2510.06149v1)

Authors: Hwanwoo Kim, Dongkyu Derek Cho, Eric Laber

Temporal difference (TD) learning is a cornerstone of reinforcement learning.
In the average-reward setting, standard TD($\lambda$) is highly sensitive to
the choice of step-size and thus requires careful tuning to maintain numerical
stability. We introduce average-reward implicit TD($\lambda$), which employs an
implicit fixed point update to provide data-adaptive stabilization while
preserving the per iteration computational complexity of standard
average-reward TD($\lambda$). In contrast to prior finite-time analyses of
average-reward TD($\lambda$), which impose restrictive step-size conditions, we
establish finite-time error bounds for the implicit variant under substantially
weaker step-size requirements. Empirically, average-reward implicit
TD($\lambda$) operates reliably over a much broader range of step-sizes and
exhibits markedly improved numerical stability. This enables more efficient
policy evaluation and policy learning, highlighting its effectiveness as a
robust alternative to average-reward TD($\lambda$).

### 7. [Bilevel optimization for learning hyperparameters: Application to solving PDEs and inverse problems with Gaussian processes](http://arxiv.org/pdf/2510.05568v1)

Authors: Nicholas H. Nelsen, Houman Owhadi, Andrew M. Stuart, Xianjin Yang, Zongren Zou

Methods for solving scientific computing and inference problems, such as
kernel- and neural network-based approaches for partial differential equations
(PDEs), inverse problems, and supervised learning tasks, depend crucially on
the choice of hyperparameters. Specifically, the efficacy of such methods, and
in particular their accuracy, stability, and generalization properties,
strongly depends on the choice of hyperparameters. While bilevel optimization
offers a principled framework for hyperparameter tuning, its nested
optimization structure can be computationally demanding, especially in
PDE-constrained contexts. In this paper, we propose an efficient strategy for
hyperparameter optimization within the bilevel framework by employing a
Gauss-Newton linearization of the inner optimization step. Our approach
provides closed-form updates, eliminating the need for repeated costly PDE
solves. As a result, each iteration of the outer loop reduces to a single
linearized PDE solve, followed by explicit gradient-based hyperparameter
updates. We demonstrate the effectiveness of the proposed method through
Gaussian process models applied to nonlinear PDEs and to PDE inverse problems.
Extensive numerical experiments highlight substantial improvements in accuracy
and robustness compared to conventional random hyperparameter initialization.
In particular, experiments with additive kernels and neural
network-parameterized deep kernels demonstrate the method's scalability and
effectiveness for high-dimensional hyperparameter optimization.

### 8. [On the Theory of Continual Learning with Gradient Descent for Neural Networks](http://arxiv.org/pdf/2510.05573v1)

Authors: Hossein Taheri, Avishek Ghosh, Arya Mazumdar

Continual learning, the ability of a model to adapt to an ongoing sequence of
tasks without forgetting the earlier ones, is a central goal of artificial
intelligence. To shed light on its underlying mechanisms, we analyze the
limitations of continual learning in a tractable yet representative setting. In
particular, we study one-hidden-layer quadratic neural networks trained by
gradient descent on an XOR cluster dataset with Gaussian noise, where different
tasks correspond to different clusters with orthogonal means. Our results
obtain bounds on the rate of forgetting during train and test-time in terms of
the number of iterations, the sample size, the number of tasks, and the
hidden-layer size. Our results reveal interesting phenomena on the role of
different problem parameters in the rate of forgetting. Numerical experiments
across diverse setups confirm our results, demonstrating their validity beyond
the analyzed settings.

### 9. [Mitigating Premature Exploitation in Particle-based Monte Carlo for Inference-Time Scaling](http://arxiv.org/pdf/2510.05825v1)

Authors: Giorgio Giannone, Guangxuan Xu, Nikhil Shivakumar Nayak, Rohan Mahesh Awhad, Shivchander Sudalairaj, Kai Xu, Akash Srivastava

Inference-Time Scaling (ITS) improves language models by allocating more
computation at generation time. Particle Filtering (PF) has emerged as a strong
ITS method for complex mathematical reasoning tasks, but it is vulnerable when
guided by process reward models, which often assign overconfident scores early
in the reasoning process. This causes PF to suffer from premature exploitation:
it myopically commits to locally promising trajectories, prunes potentially
correct hypotheses, and converges to suboptimal solutions. This failure mode,
known as particle impoverishment, is especially severe under constrained
computational budgets. To address this, we analyze the problem and identify two
root causes: a lack of diversity in the particle set due to overconfident
resampling and consequent inability to assess the potential of a reasoning
path. We introduce Entropic Particle Filtering (ePF), an algorithm that
integrates two new techniques to solve these issues. The first technique,
Entropic Annealing (EA), directly mitigates particle impoverishment by
monitoring search diversity via entropy; when diversity drops, it intervenes by
dynamically annealing the resampling distribution to preserve exploration. The
second, an enhancement called Look-ahead Modulation (LaM), adds a predictive
guide to evaluate a state's potential based on its successors. On several
challenging math benchmarks, ePF significantly outperforms strong baselines and
achieves up to a 50 % relative improvement in task reward. Together, these
methods improve PF's resilience by balancing the exploration of diverse
solution spaces with the exploitation of high-reward regions, ultimately
leading to higher-quality solutions.

### 10. [Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density](http://arxiv.org/pdf/2510.05949v1)

Authors: Randall Balestriero, Nicolas Ballas, Mike Rabbat, Yann LeCun

Joint Embedding Predictive Architectures (JEPAs) learn representations able
to solve numerous downstream tasks out-of-the-box. JEPAs combine two
objectives: (i) a latent-space prediction term, i.e., the representation of a
slightly perturbed sample must be predictable from the original sample's
representation, and (ii) an anti-collapse term, i.e., not all samples should
have the same representation. While (ii) is often considered as an obvious
remedy to representation collapse, we uncover that JEPAs' anti-collapse term
does much more--it provably estimates the data density. In short, any
successfully trained JEPA can be used to get sample probabilities, e.g., for
data curation, outlier detection, or simply for density estimation. Our
theoretical finding is agnostic of the dataset and architecture used--in any
case one can compute the learned probabilities of sample $x$ efficiently and in
closed-form using the model's Jacobian matrix at $x$. Our findings are
empirically validated across datasets (synthetic, controlled, and Imagenet) and
across different Self Supervised Learning methods falling under the JEPA family
(I-JEPA and DINOv2) and on multimodal models, such as MetaCLIP. We denote the
method extracting the JEPA learned density as {\bf JEPA-SCORE}.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-08 PST.

### 1. [AI models that lie, cheat and plot murder: how dangerous are LLMs really?](https://www.nature.com/articles/d41586-025-03222-1)

Authors: Matthew Hutson

### 2. [A robust artificial intelligence system for predicting EBV status in gastric cancer biopsy and resection specimens](https://www.nature.com/articles/s41598-025-18836-8)

Authors: Keunho  Byeon et al.

### 3. [Textual interpretation of transient image classifications from large language models](https://www.nature.com/articles/s41550-025-02670-z)

Authors: Fiorenzo Stoppa et al.

### 4. [Graph convolutional network with reinforced dependency graph and denoising mechanism for sarcasm detection](https://www.nature.com/articles/s41598-025-18849-3)

Authors: Pingping Yan et al.

### 5. [GPT-4 shows comparable performance to human examiners in ranking open-text answers](https://www.nature.com/articles/s41598-025-21572-8)

Authors: Abdullah Al Zubaer et al.

### 6. [Working-at-high operation safety protection recognition based on target detection and spatial relationship](https://www.nature.com/articles/s41598-025-19048-w)

Authors: Donghai Liu et al.

### 7. [Detection and classification of brain tumor using a hybrid learning model in CT scan images](https://www.nature.com/articles/s41598-025-18979-8)

Authors: Roja Ghasemi et al.

### 8. [Artificial intelligence in student management systems to enhance academic performance monitoring and intervention](https://www.nature.com/articles/s41598-025-19159-4)

Authors: Yueying Wang

### 9. [Generating reliable software project task flows using large language models through prompt engineering and robust evaluation](https://www.nature.com/articles/s41598-025-19170-9)

Authors: Mohammed Sarim et al.

### 10. [SU(d)-symmetric random unitaries: quantum scrambling, error correction, and machine learning](https://www.nature.com/articles/s41534-025-01045-6)

Authors: Zimu Li et al.

### 11. [TransBreastNet a CNN transformer hybrid deep learning framework for breast cancer subtype classification and temporal lesion progression analysis](https://www.nature.com/articles/s41598-025-19173-6)

Authors: Aluri Brahmareddy et al.

