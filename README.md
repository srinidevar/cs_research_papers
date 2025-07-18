# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-17 17:00:25.276065 PST.

### Artificial Intelligence

### 1. [Aime: Towards Fully-Autonomous Multi-Agent Framework](http://arxiv.org/pdf/2507.11988v1)

Authors: Yexuan Shi, Mingyu Wang, Yunxiang Cao, Hongjie Lai, Junjian Lan, Xin Han, Yu Wang, Jie Geng, Zhenan Li, Zihao Xia, Xiang Chen, Chen Li, Jian Xu, Wenbo Duan, Yuanshuo Zhu

Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) are
emerging as a powerful paradigm for solving complex, multifaceted problems.
However, the potential of these systems is often constrained by the prevalent
plan-and-execute framework, which suffers from critical limitations: rigid plan
execution, static agent capabilities, and inefficient communication. These
weaknesses hinder their adaptability and robustness in dynamic environments.
This paper introduces Aime, a novel multi-agent framework designed to overcome
these challenges through dynamic, reactive planning and execution. Aime
replaces the conventional static workflow with a fluid and adaptive
architecture. Its core innovations include: (1) a Dynamic Planner that
continuously refines the overall strategy based on real-time execution
feedback; (2) an Actor Factory that implements Dynamic Actor instantiation,
assembling specialized agents on-demand with tailored tools and knowledge; and
(3) a centralized Progress Management Module that serves as a single source of
truth for coherent, system-wide state awareness. We empirically evaluated Aime
on a diverse suite of benchmarks spanning general reasoning (GAIA), software
engineering (SWE-bench Verified), and live web navigation (WebVoyager). The
results demonstrate that Aime consistently outperforms even highly specialized
state-of-the-art agents in their respective domains. Its superior adaptability
and task success rate establish Aime as a more resilient and effective
foundation for multi-agent collaboration.

### 2. [Understanding visual attention beehind bee-inspired UAV navigation](http://arxiv.org/pdf/2507.11992v1)

Authors: Pranav Rajbhandari, Abhi Veda, Matthew Garratt, Mandayam Srinivasan, Sridhar Ravi

Bio-inspired design is often used in autonomous UAV navigation due to the
capacity of biological systems for flight and obstacle avoidance despite
limited sensory and computational capabilities. In particular, honeybees mainly
use the sensory input of optic flow, the apparent motion of objects in their
visual field, to navigate cluttered environments. In our work, we train a
Reinforcement Learning agent to navigate a tunnel with obstacles using only
optic flow as sensory input. We inspect the attention patterns of trained
agents to determine the regions of optic flow on which they primarily base
their motor decisions. We find that agents trained in this way pay most
attention to regions of discontinuity in optic flow, as well as regions with
large optic flow magnitude. The trained agents appear to navigate a cluttered
tunnel by avoiding the obstacles that produce large optic flow, while
maintaining a centered position in their environment, which resembles the
behavior seen in flying insects. This pattern persists across independently
trained agents, which suggests that this could be a good strategy for
developing a simple explicit control law for physical UAVs.

### 3. [Topology Enhanced MARL for Multi-Vehicle Cooperative Decision-Making of CAVs](http://arxiv.org/pdf/2507.12110v1)

Authors: Ye Han, Lijun Zhang, Dejian Meng, Zhuang Zhang

The exploration-exploitation trade-off constitutes one of the fundamental
challenges in reinforcement learning (RL), which is exacerbated in multi-agent
reinforcement learning (MARL) due to the exponential growth of joint
state-action spaces. This paper proposes a topology-enhanced MARL (TPE-MARL)
method for optimizing cooperative decision-making of connected and autonomous
vehicles (CAVs) in mixed traffic. This work presents two primary contributions:
First, we construct a game topology tensor for dynamic traffic flow,
effectively compressing high-dimensional traffic state information and decrease
the search space for MARL algorithms. Second, building upon the designed game
topology tensor and using QMIX as the backbone RL algorithm, we establish a
topology-enhanced MARL framework incorporating visit counts and agent mutual
information. Extensive simulations across varying traffic densities and CAV
penetration rates demonstrate the effectiveness of TPE-MARL. Evaluations
encompassing training dynamics, exploration patterns, macroscopic traffic
performance metrics, and microscopic vehicle behaviors reveal that TPE-MARL
successfully balances exploration and exploitation. Consequently, it exhibits
superior performance in terms of traffic efficiency, safety, decision
smoothness, and task completion. Furthermore, the algorithm demonstrates
decision-making rationality comparable to or exceeding that of human drivers in
both mixed-autonomy and fully autonomous traffic scenarios. Code of our work is
available at
\href{https://github.com/leoPub/tpemarl}{https://github.com/leoPub/tpemarl}.

### 4. [Partially Observable Reference Policy Programming: Solving POMDPs Sans Numerical Optimisation](http://arxiv.org/pdf/2507.12186v1)

Authors: Edward Kim, Hanna Kurniawati

This paper proposes Partially Observable Reference Policy Programming, a
novel anytime online approximate POMDP solver which samples meaningful future
histories very deeply while simultaneously forcing a gradual policy update. We
provide theoretical guarantees for the algorithm's underlying scheme which say
that the performance loss is bounded by the average of the sampling
approximation errors rather than the usual maximum, a crucial requirement given
the sampling sparsity of online planning. Empirical evaluations on two
large-scale problems with dynamically evolving environments -- including a
helicopter emergency scenario in the Corsica region requiring approximately 150
planning steps -- corroborate the theoretical results and indicate that our
solver considerably outperforms current online benchmarks.

### 5. [Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning](http://arxiv.org/pdf/2507.12215v1)

Authors: Yuhao Chen, Shuochen Liu, Yuanjie Lyu, Chao Zhang, Jiayao Shi, Tong Xu

Game playing has long served as a fundamental benchmark for evaluating
Artificial General Intelligence (AGI). While Large Language Models (LLMs) have
demonstrated impressive capabilities in general reasoning, their effectiveness
in spatial strategic reasoning, which is critical for complex and fully
observable board games, remains insufficiently explored. In this work, we adopt
Chinese Chess (Xiangqi) as a challenging and rich testbed due to its intricate
rules and spatial complexity. To advance LLMs' strategic competence in such
environments, we propose a training framework tailored to Xiangqi, built upon a
large-scale dataset of five million board-move pairs enhanced with expert
annotations and engine evaluations. Building on this foundation, we introduce
Xiangqi-R1, a 7B-parameter model trained in multi-stage manner: (1) fine-tuning
for legal move prediction to capture basic spatial rules, (2) incorporating
strategic annotations to improve decision-making, and (3) applying
reinforcement learning via Group Relative Policy Optimization (GRPO) with
multi-dimensional reward signals to enhance reasoning stability. Our
Experimental results indicate that, despite their size and power,
general-purpose LLMs struggle to achieve satisfactory performance in these
tasks. Compared to general-purpose LLMs, Xiangqi-R1 greatly advances with an
18% rise in move legality and a 22% boost in analysis accuracy. Our results
point to a promising path for creating general strategic intelligence in
spatially complex areas.

### 6. [CLID-MU: Cross-Layer Information Divergence Based Meta Update Strategy for Learning with Noisy Labels](http://arxiv.org/pdf/2507.11807v1)

Authors: Ruofan Hu, Dongyu Zhang, Huayi Zhang, Elke Rundensteiner

Learning with noisy labels (LNL) is essential for training deep neural
networks with imperfect data. Meta-learning approaches have achieved success by
using a clean unbiased labeled set to train a robust model. However, this
approach heavily depends on the availability of a clean labeled meta-dataset,
which is difficult to obtain in practice. In this work, we thus tackle the
challenge of meta-learning for noisy label scenarios without relying on a clean
labeled dataset. Our approach leverages the data itself while bypassing the
need for labels. Building on the insight that clean samples effectively
preserve the consistency of related data structures across the last hidden and
the final layer, whereas noisy samples disrupt this consistency, we design the
Cross-layer Information Divergence-based Meta Update Strategy (CLID-MU).
CLID-MU leverages the alignment of data structures across these diverse feature
spaces to evaluate model performance and use this alignment to guide training.
Experiments on benchmark datasets with varying amounts of labels under both
synthetic and real-world noise demonstrate that CLID-MU outperforms
state-of-the-art methods. The code is released at
https://github.com/ruofanhu/CLID-MU.

### 7. [The Evolving Role of Large Language Models in Scientific Innovation: Evaluator, Collaborator, and Scientist](http://arxiv.org/pdf/2507.11810v1)

Authors: Haoxuan Zhang, Ruochi Li, Yang Zhang, Ting Xiao, Jiangping Chen, Junhua Ding, Haihua Chen

Scientific innovation is undergoing a paradigm shift driven by the rapid
advancement of Large Language Models (LLMs). As science faces mounting
challenges including information overload, disciplinary silos, and diminishing
returns on conventional research methods, LLMs are emerging as powerful agents
capable not only of enhancing scientific workflows but also of participating in
and potentially leading the innovation process. Existing surveys mainly focus
on different perspectives, phrases, and tasks in scientific research and
discovery, while they have limitations in understanding the transformative
potential and role differentiation of LLM. This survey proposes a comprehensive
framework to categorize the evolving roles of LLMs in scientific innovation
across three hierarchical levels: Evaluator, Collaborator, and Scientist. We
distinguish between LLMs' contributions to structured scientific research
processes and open-ended scientific discovery, thereby offering a unified
taxonomy that clarifies capability boundaries, evaluation criteria, and
human-AI interaction patterns at each level. Through an extensive analysis of
current methodologies, benchmarks, systems, and evaluation metrics, this survey
delivers an in-depth and systematic synthesis on LLM-driven scientific
innovation. We present LLMs not only as tools for automating existing
processes, but also as catalysts capable of reshaping the epistemological
foundations of science itself. This survey offers conceptual clarity, practical
guidance, and theoretical foundations for future research, while also
highlighting open challenges and ethical considerations in the pursuit of
increasingly autonomous AI-driven science. Resources related to this survey can
be accessed on GitHub at: https://github.com/haoxuan-unt2024/llm4innovation.

### 8. [Spatial Frequency Modulation for Semantic Segmentation](http://arxiv.org/pdf/2507.11893v1)

Authors: Linwei Chen, Ying Fu, Lin Gu, Dezhi Zheng, Jifeng Dai

High spatial frequency information, including fine details like textures,
significantly contributes to the accuracy of semantic segmentation. However,
according to the Nyquist-Shannon Sampling Theorem, high-frequency components
are vulnerable to aliasing or distortion when propagating through downsampling
layers such as strided-convolution. Here, we propose a novel Spatial Frequency
Modulation (SFM) that modulates high-frequency features to a lower frequency
before downsampling and then demodulates them back during upsampling.
Specifically, we implement modulation through adaptive resampling (ARS) and
design a lightweight add-on that can densely sample the high-frequency areas to
scale up the signal, thereby lowering its frequency in accordance with the
Frequency Scaling Property. We also propose Multi-Scale Adaptive Upsampling
(MSAU) to demodulate the modulated feature and recover high-frequency
information through non-uniform upsampling This module further improves
segmentation by explicitly exploiting information interaction between densely
and sparsely resampled areas at multiple scales. Both modules can seamlessly
integrate with various architectures, extending from convolutional neural
networks to transformers. Feature visualization and analysis confirm that our
method effectively alleviates aliasing while successfully retaining details
after demodulation. Finally, we validate the broad applicability and
effectiveness of SFM by extending it to image classification, adversarial
robustness, instance segmentation, and panoptic segmentation tasks. The code is
available at
\href{https://github.com/Linwei-Chen/SFM}{https://github.com/Linwei-Chen/SFM}.

### 9. [A Parallel CPU-GPU Framework for Cost-Bounded DFS with Applications to IDA* and BTS](http://arxiv.org/pdf/2507.11916v1)

Authors: Ehsan Futuhi, Nathan R. Sturtevant

The rapid advancement of GPU technology has unlocked powerful parallel
processing capabilities, creating new opportunities to enhance classic search
algorithms. A recent successful application of GPUs is in compressing large
pattern database (PDB) heuristics using neural networks while preserving
heuristic admissibility. However, very few algorithms have been designed to
exploit GPUs during search. Several variants of A* exist that batch GPU
computations. In this paper we introduce a method for batching GPU computations
in depth first search. In particular, we describe a new cost-bounded
depth-first search (CB-DFS) method that leverages the combined parallelism of
modern CPUs and GPUs. This is used to create algorithms like \emph{Batch IDA*},
an extension of the Iterative Deepening A* (IDA*) algorithm, or Batch BTS, an
extensions of Budgeted Tree Search. Our approach builds on the general approach
used by Asynchronous Parallel IDA* (AIDA*), while maintaining optimality
guarantees. We evaluate the approach on the 3x3 Rubik's Cube and 4x4 sliding
tile puzzle (STP), showing that GPU operations can be efficiently batched in
DFS. Additionally, we conduct extensive experiments to analyze the effects of
hyperparameters, neural network heuristic size, and hardware resources on
performance.

### 10. [RaDL: Relation-aware Disentangled Learning for Multi-Instance Text-to-Image Generation](http://arxiv.org/pdf/2507.11947v1)

Authors: Geon Park, Seon Bin Kim, Gunho Jung, Seong-Whan Lee

With recent advancements in text-to-image (T2I) models, effectively
generating multiple instances within a single image prompt has become a crucial
challenge. Existing methods, while successful in generating positions of
individual instances, often struggle to account for relationship discrepancy
and multiple attributes leakage. To address these limitations, this paper
proposes the relation-aware disentangled learning (RaDL) framework. RaDL
enhances instance-specific attributes through learnable parameters and
generates relation-aware image features via Relation Attention, utilizing
action verbs extracted from the global prompt. Through extensive evaluations on
benchmarks such as COCO-Position, COCO-MIG, and DrawBench, we demonstrate that
RaDL outperforms existing methods, showing significant improvements in
positional accuracy, multiple attributes consideration, and the relationships
between instances. Our results present RaDL as the solution for generating
images that consider both the relationships and multiple attributes of each
instance within the multi-instance image.

### Hardware Architecture

### 1. [High-Performance Pipelined NTT Accelerators with Homogeneous Digit-Serial Modulo Arithmetic](http://arxiv.org/pdf/2507.12418v1)

Authors: George Alexakis, Dimitrios Schoinianakis, Giorgos Dimitrakopoulos

The Number Theoretic Transform (NTT) is a fundamental operation in
privacy-preserving technologies, particularly within fully homomorphic
encryption (FHE). The efficiency of NTT computation directly impacts the
overall performance of FHE, making hardware acceleration a critical technology
that will enable realistic FHE applications. Custom accelerators, in FPGAs or
ASICs, offer significant performance advantages due to their ability to exploit
massive parallelism and specialized optimizations. However, the operation of
NTT over large moduli requires large word-length modulo arithmetic that limits
achievable clock frequencies in hardware and increases hardware area costs. To
overcome such deficits, digit-serial arithmetic has been explored for modular
multiplication and addition independently. The goal of this work is to leverage
digit-serial modulo arithmetic combined with appropriate redundant data
representation to design modular pipelined NTT accelerators that operate
uniformly on arbitrary small digits, without the need for intermediate
(de)serialization. The proposed architecture enables high clock frequencies
through regular pipelining while maintaining parallelism. Experimental results
demonstrate that the proposed approach outperforms state-of-the-art
implementations and reduces hardware complexity under equal performance and
input-output bandwidth constraints.

### 2. [MOFCO: Mobility- and Migration-Aware Task Offloading in Three-Layer Fog Computing Environments](http://arxiv.org/pdf/2507.12028v1)

Authors: Soheil Mahdizadeh, Elyas Oustad, Mohsen Ansari

Task offloading in three-layer fog computing environments presents a critical
challenge due to user equipment (UE) mobility, which frequently triggers costly
service migrations and degrades overall system performance. This paper
addresses this problem by proposing MOFCO, a novel Mobility- and
Migration-aware Task Offloading algorithm for Fog Computing environments. The
proposed method formulates task offloading and resource allocation as a
Mixed-Integer Nonlinear Programming (MINLP) problem and employs a
heuristic-aided evolutionary game theory approach to solve it efficiently. To
evaluate MOFCO, we simulate mobile users using SUMO, providing realistic
mobility patterns. Experimental results show that MOFCO reduces system cost,
defined as a combination of latency and energy consumption, by an average of
19% and up to 43% in certain scenarios compared to state-of-the-art methods.

### 3. [Chain-of-Descriptions: Improving Code LLMs for VHDL Code Generation and Summarization](http://arxiv.org/pdf/2507.12308v1)

Authors: Prashanth Vijayaraghavan, Apoorva Nitsure, Charles Mackin, Luyao Shi, Stefano Ambrogio, Arvind Haran, Viresh Paruthi, Ali Elzein, Dan Coops, David Beymer, Tyler Baldwin, Ehsan Degan

Large Language Models (LLMs) have become widely used across diverse NLP tasks
and domains, demonstrating their adaptability and effectiveness. In the realm
of Electronic Design Automation (EDA), LLMs show promise for tasks like
Register-Transfer Level (RTL) code generation and summarization. However,
despite the proliferation of LLMs for general code-related tasks, there's a
dearth of research focused on evaluating and refining these models for hardware
description languages (HDLs), notably VHDL. In this study, we evaluate the
performance of existing code LLMs for VHDL code generation and summarization
using various metrics and two datasets -- VHDL-Eval and VHDL-Xform. The latter,
an in-house dataset, aims to gauge LLMs' understanding of functionally
equivalent code. Our findings reveal consistent underperformance of these
models across different metrics, underscoring a significant gap in their
suitability for this domain. To address this challenge, we propose
Chain-of-Descriptions (CoDes), a novel approach to enhance the performance of
LLMs for VHDL code generation and summarization tasks. CoDes involves
generating a series of intermediate descriptive steps based on: (i) the problem
statement for code generation, and (ii) the VHDL code for summarization. These
steps are then integrated with the original input prompt (problem statement or
code) and provided as input to the LLMs to generate the final output. Our
experiments demonstrate that the CoDes approach significantly surpasses the
standard prompting strategy across various metrics on both datasets. This
method not only improves the quality of VHDL code generation and summarization
but also serves as a framework for future research aimed at enhancing code LLMs
for VHDL.

### 4. [CRAFT: Latency and Cost-Aware Genetic-Based Framework for Node Placement in Edge-Fog Environments](http://arxiv.org/pdf/2507.12445v1)

Authors: Soheil Mahdizadeh, Amir Mahdi Rasouli, Mohammad Pourashory, Sadra Galavani, Mohsen Ansari

Reducing latency in the Internet of Things (IoT) is a critical concern. While
cloud computing facilitates communication, it falls short of meeting real-time
requirements reliably. Edge and fog computing have emerged as viable solutions
by positioning computing nodes closer to end users, offering lower latency and
increased processing power. An edge-fog framework comprises various components,
including edge and fog nodes, whose strategic placement is crucial as it
directly impacts latency and system cost. This paper presents an effective and
tunable node placement strategy based on a genetic algorithm to address the
optimization problem of deploying edge and fog nodes. The main objective is to
minimize latency and cost through optimal node placement. Simulation results
demonstrate that the proposed framework achieves up to 2.77% latency and 31.15%
cost reduction.

### 5. [Characterizing State Space Model (SSM) and SSM-Transformer Hybrid Language Model Performance with Long Context Length](http://arxiv.org/pdf/2507.12442v1)

Authors: Saptarshi Mitra, Rachid Karami, Haocheng Xu, Sitao Huang, Hyoukjun Kwon

The demand for machine intelligence capable of processing continuous,
long-context inputs on local devices is growing rapidly. However, the quadratic
complexity and memory requirements of traditional Transformer architectures
make them inefficient and often unusable for these tasks. This has spurred a
paradigm shift towards new architectures like State Space Models (SSMs) and
hybrids, which promise near-linear scaling. While most current research focuses
on the accuracy and theoretical throughput of these models, a systematic
performance characterization on practical consumer hardware is critically
needed to guide system-level optimization and unlock new applications.
  To address this gap, we present a comprehensive, comparative benchmarking of
carefully selected Transformer, SSM, and hybrid models specifically for
long-context inference on consumer and embedded GPUs. Our analysis reveals that
SSMs are not only viable but superior for this domain, capable of processing
sequences up to 220K tokens on a 24GB consumer GPU-approximately 4x longer than
comparable Transformers. While Transformers may be up to 1.8x faster at short
sequences, SSMs demonstrate a dramatic performance inversion, becoming up to 4x
faster at very long contexts (~57K tokens). Our operator-level analysis reveals
that custom, hardware-aware SSM kernels dominate the inference runtime,
accounting for over 55% of latency on edge platforms, identifying them as a
primary target for future hardware acceleration. We also provide detailed,
device-specific characterization results to guide system co-design for the
edge. To foster further research, we will open-source our characterization
framework.

### Computational Complexity

### 1. [Searching for Falsified Clause in Random (log n)-CNFs is Hard for Randomized Communication](http://arxiv.org/pdf/2507.12124v1)

Authors: Artur Riazanov, Anastasia Sofronova, Dmitry Sokolov, Weiqiang Yuan

We show that for a randomly sampled unsatisfiable $O(\log n)$-CNF over $n$
variables the randomized two-party communication cost of finding a clause
falsified by the given variable assignment is linear in $n$.

### 2. [Which graph motif parameters count?](http://arxiv.org/pdf/2507.12244v1)

Authors: Markus Bläser, Radu Curticapean, Julian Dörfler, Christian Ikenmeyer

For a fixed graph H, the function #IndSub(H,*) maps graphs G to the count of
induced H-copies in G; this function obviously "counts something" in that it
has a combinatorial interpretation. Linear combinations of such functions are
called graph motif parameters and have recently received significant attention
in counting complexity after a seminal paper by Curticapean, Dell and Marx
(STOC'17). We show that, among linear combinations of functions #IndSub(H,*)
involving only graphs H without isolated vertices, precisely those with
positive integer coefficients maintain a combinatorial interpretation. It is
important to note that graph motif parameters can be nonnegative for all inputs
G, even when some coefficients are negative.
  Formally, we show that evaluating any graph motif parameter with a negative
coefficient is impossible in an oracle variant of #P, where an implicit graph
is accessed by oracle queries. Our proof follows the classification of the
relativizing closure properties of #P by Hertrampf, Vollmer, and Wagner
(SCT'95) and the framework developed by Ikenmeyer and Pak (STOC'22), but our
application of the required Ramsey theorem turns out to be more subtle, as
graphs do not have the required Ramsey property.
  Our techniques generalize from graphs to relational structures, including
colored graphs. Vastly generalizing this, we introduce motif parameters over
categories that count occurrences of sub-objects in the category. We then prove
a general dichotomy theorem that characterizes which such parameters have a
combinatorial interpretation. Using known results in Ramsey theory for
categories, we obtain a dichotomy for motif parameters of finite vector spaces
as well as parameter sets.

### Computational Engineering

### 1. [MNO : A Multi-modal Neural Operator for Parametric Nonlinear BVPs](http://arxiv.org/pdf/2507.11870v1)

Authors: Vamshi C. Madala, Nithin Govindarajan, Shivkumar Chandrasekaran

We introduce a novel Multimodal Neural Operator (MNO) architecture designed
to learn solution operators for multi-parameter nonlinear boundary value
problems (BVPs). Traditional neural operators primarily map either the PDE
coefficients or source terms independently to the solution, limiting their
flexibility and applicability. In contrast, our proposed MNO architecture
generalizes these approaches by mapping multiple parameters including PDE
coefficients, source terms, and boundary conditions to the solution space in a
unified manner. Our MNO is motivated by the hierarchical nested bases of the
Fast Multipole Method (FMM) and is constructed systematically through three key
components: a parameter efficient Generalized FMM (GFMM) block, a Unimodal
Neural Operator (UNO) built upon GFMM blocks for single parameter mappings, and
most importantly, a multimodal fusion mechanism extending these components to
learn the joint map. We demonstrate the multimodal generalization capacity of
our approach on both linear and nonlinear BVPs. Our experiments show that the
network effectively handles simultaneous variations in PDE coefficients and
source or boundary terms.

### 2. [Universal Fourier Neural Operators for Micromechanics](http://arxiv.org/pdf/2507.12233v1)

Authors: Binh Huy Nguyen, Matti Schneider

\noindent Solving cell problems in homogenization is hard, and available
deep-learning frameworks fail to match the speed and generality of traditional
computational frameworks. More to the point, it is generally unclear what to
expect of machine-learning approaches, let alone single out which approaches
are promising. In the work at hand, we advocate Fourier Neural Operators (FNOs)
for micromechanics, empowering them by insights from computational
micromechanics methods based on the fast Fourier transform (FFT). We construct
an FNO surrogate mimicking the basic scheme foundational for FFT-based methods
and show that the resulting operator predicts solutions to cell problems with
\emph{arbitrary} stiffness distribution only subject to a material-contrast
constraint up to a desired accuracy. In particular, there are no restrictions
on the material symmetry like isotropy, on the number of phases and on the
geometry of the interfaces between materials. Also, the provided fidelity is
sharp and uniform, providing explicit guarantees leveraging our physical
empowerment of FNOs. To show the desired universal approximation property, we
construct an FNO explicitly that requires no training to begin with. Still, the
obtained neural operator complies with the same memory requirements as the
basic scheme and comes with runtimes proportional to classical FFT solvers. In
particular, large-scale problems with more than 100 million voxels are readily
handled. The goal of this work is to underline the potential of FNOs for
solving micromechanical problems, linking FFT-based methods to FNOs. This
connection is expected to provide a fruitful exchange between both worlds.

### 3. [Thought Purity: Defense Paradigm For Chain-of-Thought Attack](http://arxiv.org/pdf/2507.12314v1)

Authors: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

While reinforcement learning-trained Large Reasoning Models (LRMs, e.g.,
Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large
Language Models (LLMs) domain, their susceptibility to security threats remains
a critical vulnerability. This weakness is particularly evident in
Chain-of-Thought (CoT) generation processes, where adversarial methods like
backdoor prompt attacks can systematically subvert the model's core reasoning
mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this
vulnerability through exploiting prompt controllability, simultaneously
degrading both CoT safety and task performance with low-cost interventions. To
address this compounded security-performance vulnerability, we propose Thought
Purity (TP): a defense paradigm that systematically strengthens resistance to
malicious content while preserving operational efficacy. Our solution achieves
this through three synergistic components: (1) a safety-optimized data
processing pipeline (2) reinforcement learning-enhanced rule constraints (3)
adaptive monitoring metrics. Our approach establishes the first comprehensive
defense mechanism against CoTA vulnerabilities in reinforcement
learning-aligned reasoning systems, significantly advancing the
security-functionality equilibrium for next-generation AI architectures.

### 4. [Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data](http://arxiv.org/pdf/2507.12425v1)

Authors: Chandana Cheerla

Organizations increasingly rely on proprietary enterprise data, including HR
records, structured reports, and tabular documents, for critical
decision-making. While Large Language Models (LLMs) have strong generative
capabilities, they are limited by static pretraining, short context windows,
and challenges in processing heterogeneous data formats. Conventional
Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but
often struggle with structured and semi-structured data.
  This work proposes an advanced RAG framework that combines hybrid retrieval
strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by
metadata-aware filtering with SpaCy NER and cross-encoder reranking. The
framework applies semantic chunking to maintain textual coherence and retains
tabular data structures to preserve row-column integrity. Quantized indexing
optimizes retrieval efficiency, while human-in-the-loop feedback and
conversation memory improve adaptability.
  Experiments on enterprise datasets show notable improvements: Precision@5
increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74),
and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative
evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness
(4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale.
These results demonstrate the framework's effectiveness in delivering accurate,
comprehensive, and contextually relevant responses for enterprise tasks. Future
work includes extending to multimodal data and integrating agent-based
retrieval. The source code will be released at
https://github.com/CheerlaChandana/Enterprise-Chatbot

### Computation and Language

### 1. [ILID: Native Script Language Identification for Indian Languages](http://arxiv.org/pdf/2507.11832v1)

Authors: Yash Ingle, Pruthwik Mishra

The language identification task is a crucial fundamental step in NLP. Often
it serves as a pre-processing step for widely used NLP applications such as
multilingual machine translation, information retrieval, question and
answering, and text summarization. The core challenge of language
identification lies in distinguishing languages in noisy, short, and code-mixed
environments. This becomes even harder in case of diverse Indian languages that
exhibit lexical and phonetic similarities, but have distinct differences. Many
Indian languages share the same script making the task even more challenging.
In this paper, we release a dataset of 230K sentences consisting of English and
all 22 official Indian languages labeled with their language identifiers where
data in most languages are newly created. We also develop and release robust
baseline models using state-of-the-art approaches in machine learning and deep
learning that can aid the research in this field. Our baseline models are
comparable to the state-of-the-art models for the language identification task.

### 2. [Cross-Domain Transfer and Few-Shot Learning for Personal Identifiable Information Recognition](http://arxiv.org/pdf/2507.11862v1)

Authors: Junhong Ye, Xu Yuan, Xinying Qiu

Accurate recognition of personally identifiable information (PII) is central
to automated text anonymization. This paper investigates the effectiveness of
cross-domain model transfer, multi-domain data fusion, and sample-efficient
learning for PII recognition. Using annotated corpora from healthcare (I2B2),
legal (TAB), and biography (Wikipedia), we evaluate models across four
dimensions: in-domain performance, cross-domain transferability, fusion, and
few-shot learning. Results show legal-domain data transfers well to
biographical texts, while medical domains resist incoming transfer. Fusion
benefits are domain-specific, and high-quality recognition is achievable with
only 10% of training data in low-specialization domains.

### 3. [COLA-GEC: A Bidirectional Framework for Enhancing Grammatical Acceptability and Error Correction](http://arxiv.org/pdf/2507.11867v1)

Authors: Xiangyu Yang, Xinying Qiu

Grammatical Error Correction (GEC) and grammatical acceptability judgment
(COLA) are core tasks in natural language processing, sharing foundational
grammatical knowledge yet typically evolving independently. This paper
introduces COLA-GEC, a novel bidirectional framework that enhances both tasks
through mutual knowledge transfer. First, we augment grammatical acceptability
models using GEC datasets, significantly improving their performance across
multiple languages. Second, we integrate grammatical acceptability signals into
GEC model training via a dynamic loss function, effectively guiding corrections
toward grammatically acceptable outputs. Our approach achieves state-of-the-art
results on several multilingual benchmarks. Comprehensive error analysis
highlights remaining challenges, particularly in punctuation error correction,
providing insights for future improvements in grammatical modeling.

### 4. [DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation](http://arxiv.org/pdf/2507.11875v1)

Authors: Tianyou Huang, Xinglu Chen, Jingshen Zhang, Xinying Qiu, Ruiying Niu

This paper introduces DualReward, a novel reinforcement learning framework
for automatic distractor generation in cloze tests. Unlike conventional
approaches that rely primarily on supervised learning or static generative
models, our method employs a dual reward structure with adaptive scaling that
differentiates between human-created gold standard distractors and
model-generated candidates. The framework dynamically adjusts reward signal
intensity based on model performance and confidence. We evaluate our approach
on both passage-level (CLOTH-F) and sentence-level (MCQ) cloze test datasets,
demonstrating consistent improvements over state-of-the-art baselines.
Experimental results show that our adaptive reward scaling mechanism provides
modest but consistent benefits on homogeneous datasets (CLOTH-F) and more
substantial improvements (3.48-3.86% in P@1) on diverse, cross-domain data
(MCQ), suggesting its particular effectiveness for handling varied question
types and domains. Our work offers a flexible framework that effectively
balances learning from reliable human examples while exploring novel,
high-quality distractors for automated test generation.

### 5. [LLMs Encode Harmfulness and Refusal Separately](http://arxiv.org/pdf/2507.11878v1)

Authors: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

LLMs are trained to refuse harmful instructions, but do they truly understand
harmfulness beyond just refusing? Prior work has shown that LLMs' refusal
behaviors can be mediated by a one-dimensional subspace, i.e., a refusal
direction. In this work, we identify a new dimension to analyze safety
mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a
separate concept from refusal. There exists a harmfulness direction that is
distinct from the refusal direction. As causal evidence, steering along the
harmfulness direction can lead LLMs to interpret harmless instructions as
harmful, but steering along the refusal direction tends to elicit refusal
responses directly without reversing the model's judgment on harmfulness.
Furthermore, using our identified harmfulness concept, we find that certain
jailbreak methods work by reducing the refusal signals without reversing the
model's internal belief of harmfulness. We also find that adversarially
finetuning models to accept harmful instructions has minimal impact on the
model's internal belief of harmfulness. These insights lead to a practical
safety application: The model's latent harmfulness representation can serve as
an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing
over-refusals that is robust to finetuning attacks. For instance, our Latent
Guard achieves performance comparable to or better than Llama Guard 3 8B, a
dedicated finetuned safeguard model, across different jailbreak methods. Our
findings suggest that LLMs' internal understanding of harmfulness is more
robust than their refusal decision to diverse input instructions, offering a
new perspective to study AI safety

### 6. [Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](http://arxiv.org/pdf/2507.11882v1)

Authors: Bo Zeng, Chenyang Lyu, Sinuo Liu, Mingyan Zeng, Minghao Wu, Xuanfan Ni, Tianqi Shi, Yu Zhao, Yefeng Liu, Chenyu Zhu, Ruizhe Li, Jiahui Geng, Qing Li, Yu Tong, Longyue Wang, Weihua Luo, Kaifu Zhang

Instruction-following capability has become a major ability to be evaluated
for Large Language Models (LLMs). However, existing datasets, such as IFEval,
are either predominantly monolingual and centered on English or simply machine
translated to other languages, limiting their applicability in multilingual
contexts. In this paper, we present an carefully-curated extension of IFEval to
a localized multilingual version named Marco-Bench-MIF, covering 30 languages
with varying levels of localization. Our benchmark addresses linguistic
constraints (e.g., modifying capitalization requirements for Chinese) and
cultural references (e.g., substituting region-specific company names in
prompts) via a hybrid pipeline combining translation with verification. Through
comprehensive evaluation of 20+ LLMs on our Marco-Bench-MIF, we found that: (1)
25-35% accuracy gap between high/low-resource languages, (2) model scales
largely impact performance by 45-60% yet persists script-specific challenges,
and (3) machine-translated data underestimates accuracy by7-22% versus
localized data. Our analysis identifies challenges in multilingual instruction
following, including keyword consistency preservation and compositional
constraint adherence across languages. Our Marco-Bench-MIF is available at
https://github.com/AIDC-AI/Marco-Bench-MIF.

### 7. [DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](http://arxiv.org/pdf/2507.11942v1)

Authors: Yi Zhao, Zuchao Li, Hai Zhao, Baoyuan Qi, Guoming Liu

Task-agnostic prompt compression leverages the redundancy in natural language
to reduce computational overhead and enhance information density within
prompts, especially in long-context scenarios. Existing methods predominantly
rely on information entropy as the metric to compress lexical units, aiming to
achieve minimal information loss. However, these approaches overlook two
critical aspects: (i) the importance of attention-critical tokens at the
algorithmic level, and (ii) shifts in information entropy during the
compression process. Motivated by these challenges, we propose a dynamic
attention-aware approach for task-agnostic prompt compression (DAC). This
approach effectively integrates entropy and attention information, dynamically
sensing entropy shifts during compression to achieve fine-grained prompt
compression. Extensive experiments across various domains, including LongBench,
GSM8K, and BBH, show that DAC consistently yields robust and substantial
improvements across a diverse range of tasks and LLMs, offering compelling
evidence of its efficacy.

### 8. [Simplifications are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions](http://arxiv.org/pdf/2507.11981v1)

Authors: Lukas Ellinger, Miriam Anschütz, Georg Groh

Large Language Models (LLMs) can provide accurate word definitions and
explanations for any context. However, the scope of the definition changes for
different target groups, like children or language learners. This is especially
relevant for homonyms, words with multiple meanings, where oversimplification
might risk information loss by omitting key senses, potentially misleading
users who trust LLM outputs. We investigate how simplification impacts homonym
definition quality across three target groups: Normal, Simple, and ELI5. Using
two novel evaluation datasets spanning multiple languages, we test DeepSeek v3,
Llama 4 Maverick, Qwen3-30B A3B, GPT-4o mini, and Llama 3.1 8B via LLM-as-Judge
and human annotations. Our results show that simplification drastically
degrades definition completeness by neglecting polysemy, increasing the risk of
misunderstanding. Fine-tuning Llama 3.1 8B with Direct Preference Optimization
substantially improves homonym response quality across all prompt types. These
findings highlight the need to balance simplicity and completeness in
educational NLP to ensure reliable, context-aware definitions for all learners.

### 9. [Improving Data and Parameter Efficiency of Neural Language Models Using Representation Analysis](http://arxiv.org/pdf/2507.12004v1)

Authors: Josip Jukić

This thesis addresses challenges related to data and parameter efficiency in
neural language models, with a focus on representation analysis and the
introduction of new optimization techniques. The first part examines the
properties and dynamics of language representations within neural models,
emphasizing their significance in enhancing robustness and generalization. It
proposes innovative approaches based on representation smoothness, including
regularization strategies that utilize Jacobian and Hessian matrices to
stabilize training and mitigate sensitivity to input perturbations. The second
part focuses on methods to significantly enhance data and parameter efficiency
by integrating active learning strategies with parameter-efficient fine-tuning,
guided by insights from representation smoothness analysis. It presents
smoothness-informed early-stopping techniques designed to eliminate the need
for labeled validation sets and proposes innovative combinations of active
learning and parameter-efficient fine-tuning to reduce labeling efforts and
computational resources. Extensive experimental evaluations across various NLP
tasks demonstrate that these combined approaches substantially outperform
traditional methods in terms of performance, stability, and efficiency. The
third part explores weak supervision techniques enhanced by in-context learning
to effectively utilize unlabeled data, further reducing dependence on extensive
labeling. It shows that using in-context learning as a mechanism for weak
supervision enables models to better generalize from limited labeled data by
leveraging unlabeled examples more effectively during training. Comprehensive
empirical evaluations confirm significant gains in model accuracy,
adaptability, and robustness, especially in low-resource settings and dynamic
data environments.

### 10. [A Comparative Approach to Assessing Linguistic Creativity of Large Language Models and Humans](http://arxiv.org/pdf/2507.12039v1)

Authors: Anca Dinu, Andra-Maria Florescu, Alina Resceanu

The following paper introduces a general linguistic creativity test for
humans and Large Language Models (LLMs). The test consists of various tasks
aimed at assessing their ability to generate new original words and phrases
based on word formation processes (derivation and compounding) and on
metaphorical language use. We administered the test to 24 humans and to an
equal number of LLMs, and we automatically evaluated their answers using OCSAI
tool for three criteria: Originality, Elaboration, and Flexibility. The results
show that LLMs not only outperformed humans in all the assessed criteria, but
did better in six out of the eight test tasks. We then computed the uniqueness
of the individual answers, which showed some minor differences between humans
and LLMs. Finally, we performed a short manual analysis of the dataset, which
revealed that humans are more inclined towards E(extending)-creativity, while
LLMs favor F(ixed)-creativity.

### Cryptography and Security

### 1. [Unveiling Usability Challenges in Web Privacy Controls](http://arxiv.org/pdf/2507.11908v1)

Authors: Rahat Masood, Sunday Oyinlola Ogundoyin, Muhammad Ikram, Alex Ye

With the increasing concerns around privacy and the enforcement of data
privacy laws, many websites now provide users with privacy controls. However,
locating these controls can be challenging, as they are frequently hidden
within multiple settings and layers. Moreover, the lack of standardization
means these controls can vary widely across services. The technical or
confusing terminology used to describe these controls further complicates
users' ability to understand and use them effectively. This paper presents a
large-scale empirical analysis investigating usability challenges of web
privacy controls across 18,628 websites. While aiming for a multi-scenario
view, our automated data collection faced significant hurdles, particularly in
simulating sign-up and authenticated user visits, leading to more focused
insights on guest visit scenarios and challenges in automated capture of
dynamic user interactions. Our heuristic evaluation of three different user
visit scenarios identifies significant website usability issues. Our results
show that privacy policies are most common across all visit scenarios, with
nudges and notices being prevalent in sign-up situations. We recommend
designing privacy controls that: enhance awareness through pop-up nudges and
notices; offer a table of contents as navigational aids and customized settings
links in policies for more informed choice; and ensure accessibility via direct
links to privacy settings from nudges.

### 2. [Toward an Intent-Based and Ontology-Driven Autonomic Security Response in Security Orchestration Automation and Response](http://arxiv.org/pdf/2507.12061v1)

Authors: Zequan Huang, Jacques Robin, Nicolas Herbaut, Nourhène Ben Rabah, Bénédicte Le Grand

Modern Security Orchestration, Automation, and Response (SOAR) platforms must
rapidly adapt to continuously evolving cyber attacks. Intent-Based Networking
has emerged as a promising paradigm for cyber attack mitigation through
high-level declarative intents, which offer greater flexibility and persistency
than procedural actions. In this paper, we bridge the gap between two active
research directions: Intent-Based Cyber Defense and Autonomic Cyber Defense, by
proposing a unified, ontology-driven security intent definition leveraging the
MITRE-D3FEND cybersecurity ontology. We also propose a general two-tiered
methodology for integrating such security intents into decision-theoretic
Autonomic Cyber Defense systems, enabling hierarchical and context-aware
automated response capabilities. The practicality of our approach is
demonstrated through a concrete use case, showcasing its integration within
next-generation Security Orchestration, Automation, and Response platforms.

### 3. [Exploiting Jailbreaking Vulnerabilities in Generative AI to Bypass Ethical Safeguards for Facilitating Phishing Attacks](http://arxiv.org/pdf/2507.12185v1)

Authors: Rina Mishra, Gaurav Varshney

The advent of advanced Generative AI (GenAI) models such as DeepSeek and
ChatGPT has significantly reshaped the cybersecurity landscape, introducing
both promising opportunities and critical risks. This study investigates how
GenAI powered chatbot services can be exploited via jailbreaking techniques to
bypass ethical safeguards, enabling the generation of phishing content,
recommendation of hacking tools, and orchestration of phishing campaigns. In
ethically controlled experiments, we used ChatGPT 4o Mini selected for its
accessibility and status as the latest publicly available model at the time of
experimentation, as a representative GenAI system. Our findings reveal that the
model could successfully guide novice users in executing phishing attacks
across various vectors, including web, email, SMS (smishing), and voice
(vishing). Unlike automated phishing campaigns that typically follow detectable
patterns, these human-guided, AI assisted attacks are capable of evading
traditional anti phishing mechanisms, thereby posing a growing security threat.
We focused on DeepSeek and ChatGPT due to their widespread adoption and
technical relevance in 2025. The study further examines common jailbreaking
techniques and the specific vulnerabilities exploited in these models. Finally,
we evaluate a range of mitigation strategies such as user education, advanced
authentication mechanisms, and regulatory policy measures and discuss emerging
trends in GenAI facilitated phishing, outlining future research directions to
strengthen cybersecurity defenses in the age of artificial intelligence.

### 4. [Efficient Control Flow Attestation by Speculating on Control Flow Path Representations](http://arxiv.org/pdf/2507.12345v1)

Authors: Liam Tyler, Adam Caulfield, Ivan De Oliveira Nunes

Control Flow Attestation (CFA) allows remote verification of run-time
software integrity in embedded systems. However, CFA is limited by the
storage/transmission costs of generated control flow logs (CFlog). Recent work
has proposed application-specific optimizations by speculating on likely
sub-paths in CFlog and replacing them with reserved symbols at runtime. Albeit
effective, prior approaches do not consider the representation of addresses in
a control flow path for speculation. This work proposes RESPEC-CFA, an
architectural extension for CFA allowing for speculation on (1) the locality of
control flows and (2) their Huffman encoding. Alone, RESPEC-CFA reduces CFlog
sizes by up to 90.1%. Combined with prior methods, RESPEC-CFA yields reductions
of up to 99.7%, representing a significant step toward practical CFA.

### 5. [Obfuscation of Unitary Quantum Programs](http://arxiv.org/pdf/2507.11970v1)

Authors: Mi-Ying Huang, Er-Cheng Tang

Program obfuscation aims to hide the inner workings of a program while
preserving its functionality. In the quantum setting, recent works have
obtained obfuscation schemes for specialized classes of quantum circuits. For
instance, Bartusek, Brakerski, and Vaikuntanathan (STOC 2024) constructed a
quantum state obfuscation scheme, which supports the obfuscation of quantum
programs represented as quantum states for pseudo-deterministic quantum
programs with classical inputs and outputs in the classical oracle model.
  In this work, we improve upon existing results by constructing the first
quantum state obfuscation scheme for unitary (or approximately unitary) quantum
programs supporting quantum inputs and outputs in the classical oracle model.
At the core of our obfuscation scheme are two novel ingredients: a functional
quantum authentication scheme that allows key holders to learn specific
functions of the authenticated quantum state with simulation-based security,
and a compiler that represents an arbitrary quantum circuit as a projective
linear-plus-measurement quantum program described by a sequence of non-adaptive
Clifford gates interleaved with adaptive and compatible measurements.

### 6. [IDFace: Face Template Protection for Efficient and Secure Identification](http://arxiv.org/pdf/2507.12050v1)

Authors: Sunpill Kim, Seunghun Paik, Chanwoo Hwang, Dongsoo Kim, Junbum Shin, Jae Hong Seo

As face recognition systems (FRS) become more widely used, user privacy
becomes more important. A key privacy issue in FRS is protecting the user's
face template, as the characteristics of the user's face image can be recovered
from the template. Although recent advances in cryptographic tools such as
homomorphic encryption (HE) have provided opportunities for securing the FRS,
HE cannot be used directly with FRS in an efficient plug-and-play manner. In
particular, although HE is functionally complete for arbitrary programs, it is
basically designed for algebraic operations on encrypted data of predetermined
shape, such as a polynomial ring. Thus, a non-tailored combination of HE and
the system can yield very inefficient performance, and many previous HE-based
face template protection methods are hundreds of times slower than plain
systems without protection. In this study, we propose IDFace, a new HE-based
secure and efficient face identification method with template protection.
IDFace is designed on the basis of two novel techniques for efficient searching
on a (homomorphically encrypted) biometric database with an angular metric. The
first technique is a template representation transformation that sharply
reduces the unit cost for the matching test. The second is a space-efficient
encoding that reduces wasted space from the encryption algorithm, thus saving
the number of operations on encrypted templates. Through experiments, we show
that IDFace can identify a face template from among a database of 1M encrypted
templates in 126ms, showing only 2X overhead compared to the identification
over plaintexts.

### 7. [LLAMA: Multi-Feedback Smart Contract Fuzzing Framework with LLM-Guided Seed Generation](http://arxiv.org/pdf/2507.12084v1)

Authors: Keke Gai, Haochen Liang, Jing Yu, Liehuang Zhu, Dusit Niyato

Smart contracts play a pivotal role in blockchain ecosystems, and fuzzing
remains an important approach to securing smart contracts. Even though mutation
scheduling is a key factor influencing fuzzing effectiveness, existing fuzzers
have primarily explored seed scheduling and generation, while mutation
scheduling has been rarely addressed by prior work. In this work, we propose a
Large Language Models (LLMs)-based Multi-feedback Smart Contract Fuzzing
framework (LLAMA) that integrates LLMs, evolutionary mutation strategies, and
hybrid testing techniques. Key components of the proposed LLAMA include: (i) a
hierarchical prompting strategy that guides LLMs to generate semantically valid
initial seeds, coupled with a lightweight pre-fuzzing phase to select
high-potential inputs; (ii) a multi-feedback optimization mechanism that
simultaneously improves seed generation, seed selection, and mutation
scheduling by leveraging runtime coverage and dependency feedback; and (iii) an
evolutionary fuzzing engine that dynamically adjusts mutation operator
probabilities based on effectiveness, while incorporating symbolic execution to
escape stagnation and uncover deeper vulnerabilities. Our experiments
demonstrate that LLAMA outperforms state-of-the-art fuzzers in both coverage
and vulnerability detection. Specifically, it achieves 91% instruction coverage
and 90% branch coverage, while detecting 132 out of 148 known vulnerabilities
across diverse categories. These results highlight LLAMA's effectiveness,
adaptability, and practicality in real-world smart contract security testing
scenarios.

### 8. [A Privacy-Preserving Framework for Advertising Personalization Incorporating Federated Learning and Differential Privacy](http://arxiv.org/pdf/2507.12098v1)

Authors: Xiang Li, Yifan Lin, Yuanzhe Zhang

To mitigate privacy leakage and performance issues in personalized
advertising, this paper proposes a framework that integrates federated learning
and differential privacy. The system combines distributed feature extraction,
dynamic privacy budget allocation, and robust model aggregation to balance
model accuracy, communication overhead, and privacy protection. Multi-party
secure computing and anomaly detection mechanisms further enhance system
resilience against malicious attacks. Experimental results demonstrate that the
framework achieves dual optimization of recommendation accuracy and system
efficiency while ensuring privacy, providing both a practical solution and a
theoretical foundation for applying privacy protection technologies in
advertisement recommendation.

### 9. [Rethinking the confidential cloud through a unified low-level abstraction for composable isolation](http://arxiv.org/pdf/2507.12364v1)

Authors: Adrien Ghosn, Charly Castes, Neelu S. Kalani, Yuchen Qian, Marios Kogias, Edouard Bugnion

Securing sensitive cloud workloads requires composing confidential virtual
machines (CVMs) with nested enclaves or sandboxes. Unfortunately, each new
isolation boundary adds ad-hoc access control mechanisms, hardware extensions,
and trusted software. This escalating complexity bloats the TCB, complicates
end-to-end attestation, and leads to fragmentation across platforms and cloud
service providers (CSPs).
  We introduce a unified isolation model that delegates enforceable,
composable, and attestable isolation to a single trusted security monitor:
Tyche. Tyche provides an API for partitioning, sharing, attesting, and
reclaiming resources through its core abstraction, trust domains (TDs). To
provide fine-grain isolation, TDs can recursively create and manage sub-TDs.
Tyche captures these relationships in attestations, allowing cloud tenants to
reason about end-to-end security. TDs serve as the building blocks for
constructing composable enclaves, sandboxes, and CVMs.
  Tyche runs on commodity x86_64 without hardware security extensions and can
maintain backward compatibility with existing software. We provide an SDK to
run and compose unmodified workloads as sandboxes, enclaves, and CVMs with
minimal overhead compared to native Linux execution. Tyche supports complex
cloud scenarios, such as confidential inference with mutually distrustful
users, model owners, and CSPs. An additional RISC-V prototype demonstrates
Tyche's portability across platforms.

### 10. [Bounding the asymptotic quantum value of all multipartite compiled non-local games](http://arxiv.org/pdf/2507.12408v1)

Authors: Matilde Baroni, Dominik Leichtle, Siniša Janković, Ivan Šupić

Non-local games are a powerful tool to distinguish between correlations
possible in classical and quantum worlds. Kalai et al. (STOC'23) proposed a
compiler that converts multipartite non-local games into interactive protocols
with a single prover, relying on cryptographic tools to remove the assumption
of physical separation of the players. While quantum completeness and classical
soundness of the construction have been established for all multipartite games,
quantum soundness is known only in the special case of bipartite games.
  In this paper, we prove that the Kalai et al.'s compiler indeed achieves
quantum soundness for all multipartite compiled non-local games, by showing
that any correlations that can be generated in the asymptotic case correspond
to quantum commuting strategies.
  Our proof uses techniques from the theory of operator algebras, and relies on
a characterisation of sequential operationally no-signalling strategies as
quantum commuting operator strategies in the multipartite case, thereby
generalising several previous results. On the way, we construct universal
C*-algebras of sequential PVMs and prove a new chain rule for Radon-Nikodym
derivatives of completely positive maps on C*-algebras which may be of
independent interest.

### Computer Vision and Pattern Recognition

### 1. [CorrMoE: Mixture of Experts with De-stylization Learning for Cross-Scene and Cross-Domain Correspondence Pruning](http://arxiv.org/pdf/2507.11834v1)

Authors: Peiwen Xia, Tangfei Liao, Wei Zhu, Danhuai Zhao, Jianjun Ke, Kaihao Zhang, Tong Lu, Tao Wang

Establishing reliable correspondences between image pairs is a fundamental
task in computer vision, underpinning applications such as 3D reconstruction
and visual localization. Although recent methods have made progress in pruning
outliers from dense correspondence sets, they often hypothesize consistent
visual domains and overlook the challenges posed by diverse scene structures.
In this paper, we propose CorrMoE, a novel correspondence pruning framework
that enhances robustness under cross-domain and cross-scene variations. To
address domain shift, we introduce a De-stylization Dual Branch, performing
style mixing on both implicit and explicit graph features to mitigate the
adverse influence of domain-specific representations. For scene diversity, we
design a Bi-Fusion Mixture of Experts module that adaptively integrates
multi-perspective features through linear-complexity attention and dynamic
expert routing. Extensive experiments on benchmark datasets demonstrate that
CorrMoE achieves superior accuracy and generalization compared to
state-of-the-art methods. The code and pre-trained models are available at
https://github.com/peiwenxia/CorrMoE.

### 2. [ProtoConNet: Prototypical Augmentation and Alignment for Open-Set Few-Shot Image Classification](http://arxiv.org/pdf/2507.11845v1)

Authors: Kexuan Shi, Zhuang Qi, Jingjing Zhu, Lei Meng, Yaochen Zhang, Haibei Huang, Xiangxu Meng

Open-set few-shot image classification aims to train models using a small
amount of labeled data, enabling them to achieve good generalization when
confronted with unknown environments. Existing methods mainly use visual
information from a single image to learn class representations to distinguish
known from unknown categories. However, these methods often overlook the
benefits of integrating rich contextual information. To address this issue,
this paper proposes a prototypical augmentation and alignment method, termed
ProtoConNet, which incorporates background information from different samples
to enhance the diversity of the feature space, breaking the spurious
associations between context and image subjects in few-shot scenarios.
Specifically, it consists of three main modules: the clustering-based data
selection (CDS) module mines diverse data patterns while preserving core
features; the contextual-enhanced semantic refinement (CSR) module builds a
context dictionary to integrate into image representations, which boosts the
model's robustness in various scenarios; and the prototypical alignment (PA)
module reduces the gap between image representations and class prototypes,
amplifying feature distances for known and unknown classes. Experimental
results from two datasets verified that ProtoConNet enhances the effectiveness
of representation learning in few-shot scenarios and identifies open-set
samples, making it superior to existing methods.

### 3. [CompressedVQA-HDR: Generalized Full-reference and No-reference Quality Assessment Models for Compressed High Dynamic Range Videos](http://arxiv.org/pdf/2507.11900v1)

Authors: Wei Sun, Linhan Cao, Kang Fu, Dandan Zhu, Jun Jia, Menghan Hu, Xiongkuo Min, Guangtao Zhai

Video compression is a standard procedure applied to all videos to minimize
storage and transmission demands while preserving visual quality as much as
possible. Therefore, evaluating the visual quality of compressed videos is
crucial for guiding the practical usage and further development of video
compression algorithms. Although numerous compressed video quality assessment
(VQA) methods have been proposed, they often lack the generalization capability
needed to handle the increasing diversity of video types, particularly high
dynamic range (HDR) content. In this paper, we introduce CompressedVQA-HDR, an
effective VQA framework designed to address the challenges of HDR video quality
assessment. Specifically, we adopt the Swin Transformer and SigLip 2 as the
backbone networks for the proposed full-reference (FR) and no-reference (NR)
VQA models, respectively. For the FR model, we compute deep structural and
textural similarities between reference and distorted frames using
intermediate-layer features extracted from the Swin Transformer as its
quality-aware feature representation. For the NR model, we extract the global
mean of the final-layer feature maps from SigLip 2 as its quality-aware
representation. To mitigate the issue of limited HDR training data, we
pre-train the FR model on a large-scale standard dynamic range (SDR) VQA
dataset and fine-tune it on the HDRSDR-VQA dataset. For the NR model, we employ
an iterative mixed-dataset training strategy across multiple compressed VQA
datasets, followed by fine-tuning on the HDRSDR-VQA dataset. Experimental
results show that our models achieve state-of-the-art performance compared to
existing FR and NR VQA models. Moreover, CompressedVQA-HDR-FR won first place
in the FR track of the Generalizable HDR & SDR Video Quality Measurement Grand
Challenge at IEEE ICME 2025. The code is available at
https://github.com/sunwei925/CompressedVQA-HDR.

### 4. [SEPose: A Synthetic Event-based Human Pose Estimation Dataset for Pedestrian Monitoring](http://arxiv.org/pdf/2507.11910v1)

Authors: Kaustav Chanda, Aayush Atul Verma, Arpitsinh Vaghela, Yezhou Yang, Bharatesh Chakravarthi

Event-based sensors have emerged as a promising solution for addressing
challenging conditions in pedestrian and traffic monitoring systems. Their
low-latency and high dynamic range allow for improved response time in
safety-critical situations caused by distracted walking or other unusual
movements. However, the availability of data covering such scenarios remains
limited. To address this gap, we present SEPose -- a comprehensive synthetic
event-based human pose estimation dataset for fixed pedestrian perception
generated using dynamic vision sensors in the CARLA simulator. With nearly 350K
annotated pedestrians with body pose keypoints from the perspective of fixed
traffic cameras, SEPose is a comprehensive synthetic multi-person pose
estimation dataset that spans busy and light crowds and traffic across diverse
lighting and weather conditions in 4-way intersections in urban, suburban, and
rural environments. We train existing state-of-the-art models such as RVT and
YOLOv8 on our dataset and evaluate them on real event-based data to demonstrate
the sim-to-real generalization capabilities of the proposed dataset.

### 5. [Dark-EvGS: Event Camera as an Eye for Radiance Field in the Dark](http://arxiv.org/pdf/2507.11931v1)

Authors: Jingqian Wu, Peiqi Duan, Zongqiang Wang, Changwei Wang, Boxin Shi, Edmund Y. Lam

In low-light environments, conventional cameras often struggle to capture
clear multi-view images of objects due to dynamic range limitations and motion
blur caused by long exposure. Event cameras, with their high-dynamic range and
high-speed properties, have the potential to mitigate these issues.
Additionally, 3D Gaussian Splatting (GS) enables radiance field reconstruction,
facilitating bright frame synthesis from multiple viewpoints in low-light
conditions. However, naively using an event-assisted 3D GS approach still faced
challenges because, in low light, events are noisy, frames lack quality, and
the color tone may be inconsistent. To address these issues, we propose
Dark-EvGS, the first event-assisted 3D GS framework that enables the
reconstruction of bright frames from arbitrary viewpoints along the camera
trajectory. Triplet-level supervision is proposed to gain holistic knowledge,
granular details, and sharp scene rendering. The color tone matching block is
proposed to guarantee the color consistency of the rendered frames.
Furthermore, we introduce the first real-captured dataset for the event-guided
bright frame synthesis task via 3D GS-based radiance field reconstruction.
Experiments demonstrate that our method achieves better results than existing
methods, conquering radiance field reconstruction under challenging low-light
conditions. The code and sample data are included in the supplementary
material.

### 6. [Hyperphantasia: A Benchmark for Evaluating the Mental Visualization Capabilities of Multimodal LLMs](http://arxiv.org/pdf/2507.11932v1)

Authors: Mohammad Shahab Sepehri, Berk Tinaz, Zalan Fabian, Mahdi Soltanolkotabi

Mental visualization, the ability to construct and manipulate visual
representations internally, is a core component of human cognition and plays a
vital role in tasks involving reasoning, prediction, and abstraction. Despite
the rapid progress of Multimodal Large Language Models (MLLMs), current
benchmarks primarily assess passive visual perception, offering limited insight
into the more active capability of internally constructing visual patterns to
support problem solving. Yet mental visualization is a critical cognitive skill
in humans, supporting abilities such as spatial navigation, predicting physical
trajectories, and solving complex visual problems through imaginative
simulation. To bridge this gap, we introduce Hyperphantasia, a synthetic
benchmark designed to evaluate the mental visualization abilities of MLLMs
through four carefully constructed puzzles. Each task is procedurally generated
and presented at three difficulty levels, enabling controlled analysis of model
performance across increasing complexity. Our comprehensive evaluation of
state-of-the-art models reveals a substantial gap between the performance of
humans and MLLMs. Additionally, we explore the potential of reinforcement
learning to improve visual simulation capabilities. Our findings suggest that
while some models exhibit partial competence in recognizing visual patterns,
robust mental visualization remains an open challenge for current MLLMs.

### 7. [Prototypical Progressive Alignment and Reweighting for Generalizable Semantic Segmentation](http://arxiv.org/pdf/2507.11955v1)

Authors: Yuhang Zhang, Zhengyu Zhang, Muxin Liao, Shishun Tian, Wenbin Zou, Lu Zhang, Chen Xu

Generalizable semantic segmentation aims to perform well on unseen target
domains, a critical challenge due to real-world applications requiring high
generalizability. Class-wise prototypes, representing class centroids, serve as
domain-invariant cues that benefit generalization due to their stability and
semantic consistency. However, this approach faces three challenges. First,
existing methods often adopt coarse prototypical alignment strategies, which
may hinder performance. Second, naive prototypes computed by averaging source
batch features are prone to overfitting and may be negatively affected by
unrelated source data. Third, most methods treat all source samples equally,
ignoring the fact that different features have varying adaptation difficulties.
To address these limitations, we propose a novel framework for generalizable
semantic segmentation: Prototypical Progressive Alignment and Reweighting
(PPAR), leveraging the strong generalization ability of the CLIP model.
Specifically, we define two prototypes: the Original Text Prototype (OTP) and
Visual Text Prototype (VTP), generated via CLIP to serve as a solid base for
alignment. We then introduce a progressive alignment strategy that aligns
features in an easy-to-difficult manner, reducing domain gaps gradually.
Furthermore, we propose a prototypical reweighting mechanism that estimates the
reliability of source data and adjusts its contribution, mitigating the effect
of irrelevant or harmful features (i.e., reducing negative transfer). We also
provide a theoretical analysis showing the alignment between our method and
domain generalization theory. Extensive experiments across multiple benchmarks
demonstrate that PPAR achieves state-of-the-art performance, validating its
effectiveness.

### 8. [Watch, Listen, Understand, Mislead: Tri-modal Adversarial Attacks on Short Videos for Content Appropriateness Evaluation](http://arxiv.org/pdf/2507.11968v1)

Authors: Sahid Hossain Mustakim, S M Jishanul Islam, Ummay Maria Muna, Montasir Chowdhury, Mohammed Jawwadul Islam, Sadia Ahmmed, Tashfia Sikder, Syed Tasdid Azam Dhrubo, Swakkhar Shatabda

Multimodal Large Language Models (MLLMs) are increasingly used for content
moderation, yet their robustness in short-form video contexts remains
underexplored. Current safety evaluations often rely on unimodal attacks,
failing to address combined attack vulnerabilities. In this paper, we introduce
a comprehensive framework for evaluating the tri-modal safety of MLLMs. First,
we present the Short-Video Multimodal Adversarial (SVMA) dataset, comprising
diverse short-form videos with human-guided synthetic adversarial attacks.
Second, we propose ChimeraBreak, a novel tri-modal attack strategy that
simultaneously challenges visual, auditory, and semantic reasoning pathways.
Extensive experiments on state-of-the-art MLLMs reveal significant
vulnerabilities with high Attack Success Rates (ASR). Our findings uncover
distinct failure modes, showing model biases toward misclassifying benign or
policy-violating content. We assess results using LLM-as-a-judge, demonstrating
attack reasoning efficacy. Our dataset and findings provide crucial insights
for developing more robust and safe MLLMs.

### 9. [GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models](http://arxiv.org/pdf/2507.11969v1)

Authors: Zhaohong Huang, Yuxin Zhang, Jingjing Xie, Fei Chao, Rongrong Ji

Recent advances in test-time adaptation (TTA) for Vision-Language Models
(VLMs) have garnered increasing attention, particularly through the use of
multiple augmented views of a single image to boost zero-shot generalization.
Unfortunately, existing methods fail to strike a satisfactory balance between
performance and efficiency, either due to excessive overhead of tuning text
prompts or unstable benefits from handcrafted, training-free visual feature
enhancement. In this paper, we present Global-Spatial Bias Learner (GS-Bias),
an efficient and effective TTA paradigm that incorporates two learnable biases
during TTA, unfolded as the global bias and spatial bias. Particularly, the
global bias captures the global semantic features of a test image by learning
consistency across augmented views, while spatial bias learns the semantic
coherence between regions in the image's spatial visual representation. It is
worth highlighting that these two sets of biases are directly added to the
logits outputed by the pretrained VLMs, which circumvent the full
backpropagation through VLM that hinders the efficiency of existing TTA
methods. This endows GS-Bias with extremely high efficiency while achieving
state-of-the-art performance on 15 benchmark datasets. For example, it achieves
a 2.23% improvement over TPT in cross-dataset generalization and a 2.72%
improvement in domain generalization, while requiring only 6.5% of TPT's memory
usage on ImageNet.

### 10. [EC-Diff: Fast and High-Quality Edge-Cloud Collaborative Inference for Diffusion Models](http://arxiv.org/pdf/2507.11980v1)

Authors: Jiajian Xie, Shengyu Zhang, Zhou Zhao, Fan Wu, Fei Wu

Diffusion Models have shown remarkable proficiency in image and video
synthesis. As model size and latency increase limit user experience, hybrid
edge-cloud collaborative framework was recently proposed to realize fast
inference and high-quality generation, where the cloud model initiates
high-quality semantic planning and the edge model expedites later-stage
refinement. However, excessive cloud denoising prolongs inference time, while
insufficient steps cause semantic ambiguity, leading to inconsistency in edge
model output. To address these challenges, we propose EC-Diff that accelerates
cloud inference through gradient-based noise estimation while identifying the
optimal point for cloud-edge handoff to maintain generation quality.
Specifically, we design a K-step noise approximation strategy to reduce cloud
inference frequency by using noise gradients between steps and applying cloud
inference periodically to adjust errors. Then we design a two-stage greedy
search algorithm to efficiently find the optimal parameters for noise
approximation and edge model switching. Extensive experiments demonstrate that
our method significantly enhances generation quality compared to edge
inference, while achieving up to an average $2\times$ speedup in inference
compared to cloud inference. Video samples and source code are available at
https://ec-diff.github.io/.

### Computers and Society

### 1. [A real-time metric of online engagement monitoring](http://arxiv.org/pdf/2507.12162v1)

Authors: Laura J. Johnston, Jim E. Griffin, Ioanna Manolopoulou, Takoua Jendoubi

Measuring online behavioural student engagement often relies on simple count
indicators or retrospective, predictive methods, which present challenges for
real-time application. To address these limitations, we reconceptualise an
existing course-wide engagement metric to create a chapter-based version that
aligns with the weekly structure of online courses. Derived directly from
virtual learning environment log data, the new metric allows for cumulative,
real-time tracking of student activity without requiring outcome data or model
training. We evaluate the approach across three undergraduate statistics
modules over two academic years, comparing it to the course-wide formulation to
assess how the reconceptualisation influences what is measured. Results
indicate strong alignment from as early as week 3, along with comparable or
improved predictive validity for final grades in structured, lecture-based
contexts. By the course midpoint, the weekly metric identifies as many
low-performing students as are identifiable by the end of the course. While
performance varies across modules, the chapter-based formulation offers a
scalable and interpretable method for early engagement monitoring and student
support.

### 2. ["Mapping What I Feel": Understanding Affective Geovisualization Design Through the Lens of People-Place Relationships](http://arxiv.org/pdf/2507.11841v1)

Authors: Xingyu Lan, Yutong Yang, Yifan Wang

Affective visualization design is an emerging research direction focused on
communicating and influencing emotion through visualization. However, as
revealed by previous research, this area is highly interdisciplinary and
involves theories and practices from diverse fields and disciplines, thus
awaiting analysis from more fine-grained angles. To address this need, this
work focuses on a pioneering and relatively mature sub-area, affective
geovisualization design, to further the research in this direction and provide
more domain-specific insights. Through an analysis of a curated corpus of
affective geovisualization designs using the Person-Process-Place (PPP) model
from geographic theory, we derived a design taxonomy that characterizes a
variety of methods for eliciting and enhancing emotions through geographic
visualization. We also identified four underlying high-level design paradigms
of affective geovisualization design (e.g., computational, anthropomorphic)
that guide distinct approaches to linking geographic information with human
experience. By extending existing affective visualization design frameworks
with geographic specificity, we provide additional design examples,
domain-specific analyses, and insights to guide future research and practices
in this underexplored yet highly innovative domain.

### 3. [Predictable Drifts in Collective Cultural Attention: Evidence from Nation-Level Library Takeout Data](http://arxiv.org/pdf/2507.12007v1)

Authors: Anders Weile Larsen, Vedran Sekara

Predicting changes in consumer attention for cultural products, such as
books, movies, and songs, is notoriously difficult. Past research on predicting
the popularity of individual products suggests the existence of intrinsic
prediction limits. However, little is known about the limits for predicting
collective attention across cultural products. Here, we analyze four years of
nationwide library loan data for approximately 2 million individuals,
comprising over 100 million loans of more than 660,000 unique books. We find
that culture, as measured by popularity distributions of loaned books, drifts
continually from month to month at a near-constant rate, leading to a growing
divergence over time, and that drifts vary between different book genres. By
linking book loans to registry data, we investigate the influence of age, sex,
educational level, and geographical area on cultural drift, finding
heterogeneous effects from the different demographic groups. Our findings have
important implications for market forecasting and developing robust recommender
systems, highlighting the need to account for specific drift dynamics for
different types of items and demographic groups.

### 4. [DeepShade: Enable Shade Simulation by Text-conditioned Image Generation](http://arxiv.org/pdf/2507.12103v1)

Authors: Longchao Da, Xiangrui Liu, Mithun Shivakoti, Thirulogasankar Pranav Kutralingam, Yezhou Yang, Hua Wei

Heatwaves pose a significant threat to public health, especially as global
warming intensifies. However, current routing systems (e.g., online maps) fail
to incorporate shade information due to the difficulty of estimating shades
directly from noisy satellite imagery and the limited availability of training
data for generative models. In this paper, we address these challenges through
two main contributions. First, we build an extensive dataset covering diverse
longitude-latitude regions, varying levels of building density, and different
urban layouts. Leveraging Blender-based 3D simulations alongside building
outlines, we capture building shadows under various solar zenith angles
throughout the year and at different times of day. These simulated shadows are
aligned with satellite images, providing a rich resource for learning shade
patterns. Second, we propose the DeepShade, a diffusion-based model designed to
learn and synthesize shade variations over time. It emphasizes the nuance of
edge features by jointly considering RGB with the Canny edge layer, and
incorporates contrastive learning to capture the temporal change rules of
shade. Then, by conditioning on textual descriptions of known conditions (e.g.,
time of day, solar angles), our framework provides improved performance in
generating shade images. We demonstrate the utility of our approach by using
our shade predictions to calculate shade ratios for real-world route planning
in Tempe, Arizona. We believe this work will benefit society by providing a
reference for urban planning in extreme heat weather and its potential
practical applications in the environment.

### 5. [Urban Green Governance: IoT-Driven Management and Enhancement of Urban Green Spaces in Campobasso](http://arxiv.org/pdf/2507.12106v1)

Authors: Antonio Salis, Gabriele Troina, Gianluca Boanelli, Marco Ottaviano, Paola Fortini, Soraya Versace

The efficient design and management of public green spaces is a key factor in
promoting the health and well-being of urban population, as emphasized by the
WHO, UNEP, and EEA. These areas serve as the "green lungs" of the urban
ecosystem, playing a vital role in enhancing quality of life thanks to the
provision of ecosystem services. In this context, the Smart Green City use case
in Campobasso municipality, funded by the Italian Ministry of Enterprises
(MIMIT), emerges as an innovative model for the sustainable management of green
urban areas through the adoption of an advanced system of emerging technologies
integrated and interoperable. The project integrates IoT systems and
data-driven governance platforms, enabling real-time monitoring of the health
status of trees and green areas via a Decision Support System (DSS). It also
facilitates the collection and analysis of data from diverse sources, including
weather conditions, air quality, soil moisture, pollution levels. The resulting
cloud-based platform supports a holistic real time decision making for green
urban managers, technical experts and operational staff. It enables intelligent
control and management of urban green spaces using Tree Talker sensors,
integrated with soil moisture and water potential monitoring systems. Thanks to
predictive models based on machine learning algorithms and real time data
provided by IoT sensors, irrigation of public parks can be optimized by
providing suggestions on when and how much water to apply. Customized alerts
layers are also activated warning users when monitored parameters, such as soil
temperature, humidity, or water potential, exceed predefined thresholds. This
Use Case demonstrates how digitalization, IoT sensors fusion and technological
innovation can support sustainable urban governance, fostering environmental
resilience and improving citizens quality of life.

### 6. [Toxicity-Aware Few-Shot Prompting for Low-Resource Singlish Translation](http://arxiv.org/pdf/2507.11966v1)

Authors: Ziyu Ge, Gabriel Chua, Leanne Tan, Roy Ka-Wei Lee

As online communication increasingly incorporates under-represented languages
and colloquial dialects, standard translation systems often fail to preserve
local slang, code-mixing, and culturally embedded markers of harmful speech.
Translating toxic content between low-resource language pairs poses additional
challenges due to scarce parallel data and safety filters that sanitize
offensive expressions. In this work, we propose a reproducible, two-stage
framework for toxicity-preserving translation, demonstrated on a code-mixed
Singlish safety corpus. First, we perform human-verified few-shot prompt
engineering: we iteratively curate and rank annotator-selected Singlish-target
examples to capture nuanced slang, tone, and toxicity. Second, we optimize
model-prompt pairs by benchmarking several large language models using semantic
similarity via direct and back-translation. Quantitative human evaluation
confirms the effectiveness and efficiency of our pipeline. Beyond improving
translation quality, our framework contributes to the safety of multicultural
LLMs by supporting culturally sensitive moderation and benchmarking in
low-resource contexts. By positioning Singlish as a testbed for inclusive NLP,
we underscore the importance of preserving sociolinguistic nuance in real-world
applications such as content moderation and regional platform governance.

### 7. [Multimodal Coordinated Online Behavior: Trade-offs and Strategies](http://arxiv.org/pdf/2507.12108v1)

Authors: Lorenzo Mannocci, Stefano Cresci, Matteo Magnani, Anna Monreale, Maurizio Tesconi

Coordinated online behavior, which spans from beneficial collective actions
to harmful manipulation such as disinformation campaigns, has become a key
focus in digital ecosystem analysis. Traditional methods often rely on
monomodal approaches, focusing on single types of interactions like co-retweets
or co-hashtags, or consider multiple modalities independently of each other.
However, these approaches may overlook the complex dynamics inherent in
multimodal coordination. This study compares different ways of operationalizing
the detection of multimodal coordinated behavior. It examines the trade-off
between weakly and strongly integrated multimodal models, highlighting the
balance between capturing broader coordination patterns and identifying tightly
coordinated behavior. By comparing monomodal and multimodal approaches, we
assess the unique contributions of different data modalities and explore how
varying implementations of multimodality impact detection outcomes. Our
findings reveal that not all the modalities provide distinct insights, but that
with a multimodal approach we can get a more comprehensive understanding of
coordination dynamics. This work enhances the ability to detect and analyze
coordinated online behavior, offering new perspectives for safeguarding the
integrity of digital platforms.

### Databases

### 1. [Towards Relational Contextual Equality Saturation](http://arxiv.org/pdf/2507.11897v1)

Authors: Tyler Hou, Shadaj Laddad, Joseph M. Hellerstein

Equality saturation is a powerful technique for program optimization.
Contextual equality saturation extends this to support rewrite rules that are
conditioned on where a term appears in an expression. Existing work has brought
contextual reasoning to egg; in this paper, we share our ongoing work to extend
this to relational equality saturation in egglog. We summarize the existing
approaches to contextual equality saturation, outline its main applications,
and identify key challenges in combining this approach with relational models.

### 2. [SIEVE: Effective Filtered Vector Search with Collection of Indexes](http://arxiv.org/pdf/2507.11907v1)

Authors: Zhaoheng Li, Silu Huang, Wei Ding, Yongjoo Park, Jianjun Chen

Many real-world tasks such as recommending videos with the kids tag can be
reduced to finding most similar vectors associated with hard predicates. This
task, filtered vector search, is challenging as prior state-of-the-art
graph-based (unfiltered) similarity search techniques quickly degenerate when
hard constraints are considered. That is, effective graph-based filtered
similarity search relies on sufficient connectivity for reaching the most
similar items within just a few hops. To consider predicates, recent works
propose modifying graph traversal to visit only the items that may satisfy
predicates. However, they fail to offer the just-a-few-hops property for a wide
range of predicates: they must restrict predicates significantly or lose
efficiency if only a small fraction of items satisfy predicates.
  We propose an opposite approach: instead of constraining traversal, we build
many indexes each serving different predicate forms. For effective
construction, we devise a three-dimensional analytical model capturing
relationships among index size, search time, and recall, with which we follow a
workload-aware approach to pack as many useful indexes as possible into a
collection. At query time, the analytical model is employed yet again to
discern the one that offers the fastest search at a given recall. We show
superior performance and support on datasets with varying selectivities and
forms: our approach achieves up to 8.06x speedup while having as low as 1%
build time versus other indexes, with less than 2.15x memory of a standard HNSW
graph and modest knowledge of past workloads.

### Distributed, Parallel, and Cluster Computing

### 1. [Performance Assessment of Load Balancing Methods in Cloud Computing: Analysis of Round Robin, Equally Spread, and Throttled Strategies Using Cloud Analyst](http://arxiv.org/pdf/2507.11899v1)

Authors: Saeid Aghasoleymani Najafabadi

Load balancing plays a pivotal role in cloud computing, ensuring that
resources are optimally allocated to maintain high service quality and
operational efficiency. As workloads in cloud environments become increasingly
dynamic and unpredictable, load balancing strategies are evolving from
traditional static methods to more adaptive and intelligent approaches. In this
study, the Cloud Analyst simulation tool was used to evaluate the performance
of different load balancing algorithms under various scenarios, including both
centralized and distributed resource setups. The results highlight that while
the Round Robin algorithm yields slightly better processing times within a
single data center, Equally Spread and Throttled techniques perform
competitively, especially when network latency is considered. More importantly,
when resources are distributed across multiple data centers, response times are
significantly reduced, emphasizing the value of proximity and efficient load
distribution. In these distributed environments, Equally Spread and Throttled
algorithms not only maintain quick response times but also contribute to lower
operational costs. These findings demonstrate the necessity of strategic
resource placement and proactive infrastructure planning to balance performance
and cost. Adopting intelligent, dynamic load balancing and resource management
practices can help organizations meet evolving cloud demands, optimize costs,
and maintain a competitive advantage. Continuous evaluation and integration of
emerging technologies are crucial for sustaining effective and scalable cloud
operations.

### 2. [Making Serverless Computing Extensible: A Case Study of Serverless Data Analytics](http://arxiv.org/pdf/2507.11929v1)

Authors: Minchen Yu, Yinghao Ren, Jiamu Zhao, Jiaqi Li

Serverless computing has attracted a broad range of applications due to its
ease of use and resource elasticity. However, developing serverless
applications often poses a dilemma -- relying on general-purpose serverless
platforms can fall short of delivering satisfactory performance for complex
workloads, whereas building application-specific serverless systems undermines
the simplicity and generality. In this paper, we propose an extensible design
principle for serverless computing. We argue that a platform should enable
developers to extend system behaviors for domain-specialized optimizations
while retaining a shared, easy-to-use serverless environment. We take data
analytics as a representative serverless use case and realize this design
principle in Proteus. Proteus introduces a novel abstraction of decision
workflows, allowing developers to customize control-plane behaviors for
improved application performance. Preliminary results show that Proteus's
prototype effectively optimizes analytical query execution and supports
fine-grained resource sharing across diverse applications.

### 3. [NineToothed: A Triton-Based High-Level Domain-Specific Language for Machine Learning](http://arxiv.org/pdf/2507.11978v1)

Authors: Jiacheng Huang, Zimin Li, Yinghui Li, Haojie Wang

The emergence of deep learning domain-specific languages (DSLs) has
substantially reduced the obstacles in developing high-performance,
cross-platform compute kernels. However, current DSLs, such as Triton, still
demand that developers possess expertise in parallel programming and expose
them to many low-level details. This requirement complicates the development
process and adds to the difficulty of maintaining compute kernels.
Consequently, developing a new programming model that supports serial
programming for deep learning workloads is crucial.
  This paper introduces NineToothed, a domain-specific language that offers
serial semantics for machine learning programming. Through the automatic
transformation of serial code into parallel code, NineToothed significantly
streamlines the development process while causing minimal performance
degradation. NineToothed encompasses (1) a language with tensor-oriented
metaprogramming (TOM) that adopts the arrange-and-apply paradigm, enabling the
expression of tiled computations without the need to manage low-level details
and (2) a code generator for generating high-performance parallel code. Our
evaluation results indicate that NineToothed can greatly simplify compute
kernel development while maintaining performance comparable to that of Triton.

### 4. [ARRC: Explainable, Workflow-Integrated Recommender for Sustainable Resource Optimization Across the Edge-Cloud Continuum](http://arxiv.org/pdf/2507.12032v1)

Authors: Brian-Frederik Jahnke, René Brinkhege, Jan Peter Meyer, Daniel Tebernum, Falk Howar

Achieving sustainable, explainable, and maintainable automation for resource
optimization is a core challenge across the edge-cloud continuum. Persistent
overprovisioning and operational complexity often stem from heterogeneous
platforms and layered abstractions, while systems lacking explainability and
maintainability become fragile, impede safe recovery, and accumulate technical
debt. Existing solutions are frequently reactive, limited to single abstraction
layers, or require intrusive platform changes, leaving efficiency and
maintainability gains unrealized.
  This paper addresses safe, transparent, and low-effort resource optimization
in dynamic, multi-tenant edge-cloud systems, without disrupting operator
workflows or increasing technical debt. We introduce ARRC, a recommender system
rooted in software engineering design principles, which delivers explainable,
cross-layer resource recommendations directly into operator workflows (such as
tickets and GitOps pull requests). ARRC encapsulates optimization logic in
specialized, auditable agents coordinated via a shared interface, supporting
maintainability and extensibility through transparency and the ability to
inspect both recommendations and their rationale.
  Empirical evaluation in a multi-region industrial deployment shows that ARRC
reduces operator workload by over 50%, improves compute utilization by up to
7.7x, and maintains error rates below 5%, with most benefits achieved through
incremental, operator-approved changes. This demonstrates that explainable,
recommendation-based architectures can achieve sustainable efficiency and
maintainability improvements at production scale.
  ARRC provides an empirically evaluated framework for integrating explainable,
workflow-driven automation into resource management, intended to advance best
practices for robust, maintainable, and transparent edge-cloud continuum
platforms.

### 5. [Distributed Algorithms for Potential Problems](http://arxiv.org/pdf/2507.12038v1)

Authors: Alkida Balliu, Thomas Boudier, Francesco d'Amore, Dennis Olivetti, Gustav Schmid, Jukka Suomela

In this work we present a fast distributed algorithm for local potential
problems: these are graph problems where the task is to find a locally optimal
solution where no node can unilaterally improve the utility in its local
neighborhood by changing its own label. A simple example of such a problem is
the task of finding a locally optimal cut, i.e., a cut where for each node at
least half of its incident edges are cut edges. The distributed round
complexity of locally optimal cut has been wide open; the problem is known to
require $\Omega(\log n)$ rounds in the deterministic LOCAL model and
$\Omega(\log \log n)$ rounds in the randomized LOCAL model, but the only known
upper bound is the trivial brute-force solution of $O(n)$ rounds. Locally
optimal cut in bounded-degree graphs is perhaps the simplest example of a
locally checkable labeling problem for which there is still such a large gap
between current upper and lower bounds. We show that in bounded-degree graphs,
all local potential problems, including locally optimal cut, can be solved in
$\log^{O(1)} n$ rounds, both in the deterministic and randomized LOCAL models.
In particular, the deterministic round complexity of the locally optimal cut
problem is now settled to $\log^{\Theta(1)} n$.

### 6. [Toward Efficient SpMV in Sparse LLMs via Block Extraction and Compressed Storage](http://arxiv.org/pdf/2507.12205v1)

Authors: Junqing Lin, Jingwei Sun, Mingge Lu, Guangzhong Sun

Sparse Matrix-Vector Multiplication (SpMV) has become a critical performance
bottleneck in the local deployment of sparse Large Language Models (LLMs),
where inference predominantly operates on workloads during the decoder phase
with a batch size of one. Existing SpMV kernels and sparse matrix formats,
originally designed for scientific computing, fail to exploit the unique
structure patterns inherent in sparse LLMs, resulting in suboptimal performance
and excessive storage overhead. This paper presents EC-SpMV, a GPU-optimized
SpMV approach for accelerating sparse LLM inference. EC-SpMV introduces (1) a
hierarchical block extraction algorithm that captures multiple granularities of
block structures within sparse LLMs, and (2) a novel compressed sparse format
(EC-CSR) that employs delta indexing to reduce storage overhead and enhance
memory access efficiency. Evaluated on real sparse weight matrices from LLaMA
and OPT models, EC-SpMV achieves up to 6.44x speedup over state-of-the-art SpMV
libraries and reduces storage overhead by up to 55.4% compared to CSR.

### 7. [Arctic Inference with Shift Parallelism: Fast and Efficient Open Source Inference System for Enterprise AI](http://arxiv.org/pdf/2507.11830v1)

Authors: Samyam Rajbhandari, Mert Hidayetoglu, Aurick Qiao, Ye Wang, Juncheng Yang, Jeff Rasley, Michael Wyatt, Yuxiong He

Inference is now the dominant AI workload, yet existing systems force
trade-offs between latency, throughput, and cost. Arctic Inference, an
open-source vLLM plugin from Snowflake AI Research, introduces Shift
Parallelism, a dynamic parallelism strategy that adapts to real-world traffic
while integrating speculative decoding, SwiftKV compute reduction, and
optimized embedding inference. It achieves up to 3.4 times faster request
completion, 1.75 times faster generation, and 1.6M tokens/sec per GPU for
embeddings, outperforming both latency- and throughput-optimized deployments.
Already powering Snowflake Cortex AI, Arctic Inference delivers
state-of-the-art, cost-effective inference for enterprise AI and is now
available to the community.

### 8. [A Parallel CPU-GPU Framework for Cost-Bounded DFS with Applications to IDA* and BTS](http://arxiv.org/pdf/2507.11916v1)

Authors: Ehsan Futuhi, Nathan R. Sturtevant

The rapid advancement of GPU technology has unlocked powerful parallel
processing capabilities, creating new opportunities to enhance classic search
algorithms. A recent successful application of GPUs is in compressing large
pattern database (PDB) heuristics using neural networks while preserving
heuristic admissibility. However, very few algorithms have been designed to
exploit GPUs during search. Several variants of A* exist that batch GPU
computations. In this paper we introduce a method for batching GPU computations
in depth first search. In particular, we describe a new cost-bounded
depth-first search (CB-DFS) method that leverages the combined parallelism of
modern CPUs and GPUs. This is used to create algorithms like \emph{Batch IDA*},
an extension of the Iterative Deepening A* (IDA*) algorithm, or Batch BTS, an
extensions of Budgeted Tree Search. Our approach builds on the general approach
used by Asynchronous Parallel IDA* (AIDA*), while maintaining optimality
guarantees. We evaluate the approach on the 3x3 Rubik's Cube and 4x4 sliding
tile puzzle (STP), showing that GPU operations can be efficiently batched in
DFS. Additionally, we conduct extensive experiments to analyze the effects of
hyperparameters, neural network heuristic size, and hardware resources on
performance.

### 9. [BlockBPE: Parallel BPE Tokenization](http://arxiv.org/pdf/2507.11941v1)

Authors: Amos You

Tokenization is a critical preprocessing step in large language model
pipelines, yet widely-used implementations remain CPU-bound and suboptimal for
batch inference workflows on GPU. We present BlockBPE, a parallel GPU
implementation of byte-pair encoding (BPE) that achieves near linear-time
complexity under realistic assumptions and is optimized for high-throughput,
batch inference. Unlike existing Rust-based tokenizers such as HuggingFace
Tokenizers or OpenAI's tiktoken-whose runtimes are dominated by Regex
pre-tokenization and exhibit $O(n \log n)$ runtime-BlockBPE eliminates the
Regex pre-tokenization which leads to small loss in generation quality, but
enables highly parallelized token merges within thread blocks, reducing overall
complexity to $O(nd)$ where $d \ll n$. On high-batch inference workloads,
BlockBPE achieves up to 2x higher throughput than tiktoken and 2.5x over
HuggingFace Tokenizers.

### 10. [Urban Green Governance: IoT-Driven Management and Enhancement of Urban Green Spaces in Campobasso](http://arxiv.org/pdf/2507.12106v1)

Authors: Antonio Salis, Gabriele Troina, Gianluca Boanelli, Marco Ottaviano, Paola Fortini, Soraya Versace

The efficient design and management of public green spaces is a key factor in
promoting the health and well-being of urban population, as emphasized by the
WHO, UNEP, and EEA. These areas serve as the "green lungs" of the urban
ecosystem, playing a vital role in enhancing quality of life thanks to the
provision of ecosystem services. In this context, the Smart Green City use case
in Campobasso municipality, funded by the Italian Ministry of Enterprises
(MIMIT), emerges as an innovative model for the sustainable management of green
urban areas through the adoption of an advanced system of emerging technologies
integrated and interoperable. The project integrates IoT systems and
data-driven governance platforms, enabling real-time monitoring of the health
status of trees and green areas via a Decision Support System (DSS). It also
facilitates the collection and analysis of data from diverse sources, including
weather conditions, air quality, soil moisture, pollution levels. The resulting
cloud-based platform supports a holistic real time decision making for green
urban managers, technical experts and operational staff. It enables intelligent
control and management of urban green spaces using Tree Talker sensors,
integrated with soil moisture and water potential monitoring systems. Thanks to
predictive models based on machine learning algorithms and real time data
provided by IoT sensors, irrigation of public parks can be optimized by
providing suggestions on when and how much water to apply. Customized alerts
layers are also activated warning users when monitored parameters, such as soil
temperature, humidity, or water potential, exceed predefined thresholds. This
Use Case demonstrates how digitalization, IoT sensors fusion and technological
innovation can support sustainable urban governance, fostering environmental
resilience and improving citizens quality of life.

### Digital Libraries

### 1. [The Evolving Role of Large Language Models in Scientific Innovation: Evaluator, Collaborator, and Scientist](http://arxiv.org/pdf/2507.11810v1)

Authors: Haoxuan Zhang, Ruochi Li, Yang Zhang, Ting Xiao, Jiangping Chen, Junhua Ding, Haihua Chen

Scientific innovation is undergoing a paradigm shift driven by the rapid
advancement of Large Language Models (LLMs). As science faces mounting
challenges including information overload, disciplinary silos, and diminishing
returns on conventional research methods, LLMs are emerging as powerful agents
capable not only of enhancing scientific workflows but also of participating in
and potentially leading the innovation process. Existing surveys mainly focus
on different perspectives, phrases, and tasks in scientific research and
discovery, while they have limitations in understanding the transformative
potential and role differentiation of LLM. This survey proposes a comprehensive
framework to categorize the evolving roles of LLMs in scientific innovation
across three hierarchical levels: Evaluator, Collaborator, and Scientist. We
distinguish between LLMs' contributions to structured scientific research
processes and open-ended scientific discovery, thereby offering a unified
taxonomy that clarifies capability boundaries, evaluation criteria, and
human-AI interaction patterns at each level. Through an extensive analysis of
current methodologies, benchmarks, systems, and evaluation metrics, this survey
delivers an in-depth and systematic synthesis on LLM-driven scientific
innovation. We present LLMs not only as tools for automating existing
processes, but also as catalysts capable of reshaping the epistemological
foundations of science itself. This survey offers conceptual clarity, practical
guidance, and theoretical foundations for future research, while also
highlighting open challenges and ethical considerations in the pursuit of
increasingly autonomous AI-driven science. Resources related to this survey can
be accessed on GitHub at: https://github.com/haoxuan-unt2024/llm4innovation.

### 2. [Freshness, Persistence and Success of Scientific Teams](http://arxiv.org/pdf/2507.12255v1)

Authors: Hanjo D. Boekhout, Eelke M. Heemskerk, Nicolò Pisani, Frank W. Takes

Team science dominates scientific knowledge production, but what makes
academic teams successful? Using temporal data on 25.2 million publications and
31.8 million authors, we propose a novel network-driven approach to identify
and study the success of persistent teams. Challenging the idea that
persistence alone drives success, we find that team freshness - new
collaborations built on prior experience - is key to success. High impact
research tends to emerge early in a team's lifespan. Analyzing complex team
overlap, we find that teams open to new collaborative ties consistently produce
better science. Specifically, team re-combinations that introduce new freshness
impulses sustain success, while persistence impulses from experienced teams are
linked to earlier impact. Together, freshness and persistence shape team
success across collaboration stages.

### Discrete Mathematics

### 1. [New allocation rule based on graph structures and their application to economic phenomena](http://arxiv.org/pdf/2507.11808v1)

Authors: Taiki Yamada, Taisuke Matsubae, Tomoya Akamatsu

This study introduces the \emph{edge-based Shapley value}, a novel allocation
rule within cooperative game theory, specifically tailored for networked
systems, where value is generated through interactions represented by edges.
Traditional allocation rules, such as the Shapley and Myerson values, evaluate
player contributions based on node-level characteristics, or connected
components. However, these approaches often fail to adequately capture the
functional role of edges, which are crucial in systems such as supply chains
and digital platforms, where interactions, rather than individual agents, are
the primary drivers of value. Our edge-based Shapley value shifts the
characteristic function from node sets to edge sets, thereby enabling a more
granular and context-sensitive evaluation of the contributions. We establish
its theoretical foundations, demonstrate its relationship to classical
allocation rules, and show that it retains key properties such as fairness and
symmetry. To illustrate its applicability, we present two use cases: content
platform networks and supply chain logistics (SCL). In both cases, our method
produces intuitive and structurally consistent allocations, particularly in
scenarios with overlapping routes, exclusive contracts or cost-sensitive paths.
This framework offers a new perspective on value attribution in cooperative
settings with complex interaction structures and provides practical tools for
analyzing real-world economic and logistical networks.

### 2. [Unavoidable butterfly minors in digraphs of large cycle rank](http://arxiv.org/pdf/2507.11814v1)

Authors: Meike Hatzel, O-joung Kwon, Myounghwan Lee, Sebastian Wiederrecht

Cycle rank is one of the depth parameters for digraphs introduced by Eggan in
1963. We show that there exists a function $f:\mathbb{N}\to \mathbb{N}$ such
that every digraph of cycle rank at least $f(k)$ contains a directed cycle
chain, a directed ladder, or a directed tree chain of order $k$ as a butterfly
minor. We also investigate a new connection between cycle rank and a directed
analogue of the weak coloring number of graphs.

### 3. [Every Poset has a Large Cut](http://arxiv.org/pdf/2507.12077v1)

Authors: Nati Linial, Ori Shoshani

We prove that every finite poset has a directed cut with at least one half of
the poset's pairwise order relations. The bound is tight. Also, the largest
directed cut in a poset can be found in linear time.

### 4. [The Directed Disjoint Paths Problem with Congestion](http://arxiv.org/pdf/2507.12096v1)

Authors: Matthias Bentert, Dario Cavallaro, Amelie Heindl, Ken-ichi Kawarabayashi, Stephan Kreutzer, Johannes Schröder

The classic result by Fortune, Hopcroft, and Wyllie [TCS~'80] states that the
directed disjoint paths problem is NP-complete even for two pairs of terminals.
Extending this well-known result, we show that the directed disjoint paths
problem is NP-complete for any constant congestion $c \geq 1$ and~$k \geq 3c-1$
pairs of terminals. This refutes a conjecture by Giannopoulou et al.
[SODA~'22], which says that the directed disjoint paths problem with congestion
two is polynomial-time solvable for any constant number $k$ of terminal pairs.
We then consider the cases that are not covered by this hardness result. The
first nontrivial case is $c=2$ and $k = 3$. Our second main result is to show
that this case is polynomial-time solvable.

### 5. [A near-complete resolution of the exponential-time complexity of k-opt for the traveling salesman problem](http://arxiv.org/pdf/2507.12304v1)

Authors: Sophia Heimann, Hung P. Hoang, Stefan Hougardy

The $k$-opt algorithm is one of the simplest and most widely used heuristics
for solving the traveling salesman problem. Starting from an arbitrary tour,
the $k$-opt algorithm improves the current tour in each iteration by exchanging
up to $k$ edges. The algorithm continues until no further improvement of this
kind is possible. For a long time, it remained an open question how many
iterations the $k$-opt algorithm might require for small values of $k$,
assuming the use of an optimal pivot rule. In this paper, we resolve this
question for the cases $k = 3$ and $k = 4$ by proving that in both these cases
an exponential number of iterations may be needed even if an optimal pivot rule
is used. Combined with a recent result from Heimann, Hoang, and Hougardy (ICALP
2024), this provides a complete answer for all $k \geq 3$ regarding the number
of iterations the $k$-opt algorithm may require under an optimal pivot rule. In
addition we establish an analogous exponential lower bound for the 2.5-opt
algorithm, a variant that generalizes 2-opt and is a restricted version of
3-opt. All our results hold for both the general and the metric traveling
salesman problem.

### 6. [Modeling Feasible Locomotion of Nanobots for Cancer Detection and Treatment](http://arxiv.org/pdf/2507.12400v1)

Authors: Noble Harasha, Cristina Gava, Nancy Lynch, Claudia Contini, Frederik Mallmann-Trenn

Deploying motile nanosized particles, also known as ``nanobots'', in the
human body promises to improve selectivity in drug delivery and reduce side
effects. We consider a swarm of nanobots locating a single cancerous region and
treating it by releasing an onboard payload of drugs at the site. At nanoscale,
the computation, communication, sensing, and locomotion capabilities of
individual agents are extremely limited, noisy, and/or nonexistent.
  We present a general model to formally describe the individual and collective
behavior of agents in a colloidal environment, such as the bloodstream, for
cancer detection and treatment by nanobots. This includes a feasible and
precise model of agent locomotion, inspired by actual nanoparticles that, in
the presence of an external chemical gradient, move towards areas of higher
concentration by means of self-propulsion. We present two variants of our
general model: The first assumes an endogenous chemical gradient that is fixed
over time and centered at the targeted cancer site; the second is a more
speculative and dynamic variant in which agents themselves create and amplify a
chemical gradient centered at the cancer site. In both settings, agents can
sense the gradient and ascend it noisily, locating the cancer site more quickly
than via simple Brownian motion.
  For the first variant of the model, we present simulation results to show the
behavior of agents under our locomotion model, as well as {analytical results}
to bound the time it takes for the agents to reach the cancer site. For the
second variant, simulation results highlight the collective benefit in having
agents issue their own chemical signal. While arguably more speculative in its
agent capability assumptions, this variant shows a significant improvement in
runtime performance over the first variant, resulting from its chemical signal
amplification mechanism.

### 7. [Matroids are Equitable](http://arxiv.org/pdf/2507.12100v1)

Authors: Hannaneh Akrami, Roshan Raj, László A. Végh

We show that if the ground set of a matroid can be partitioned into $k\ge 2$
bases, then for any given subset $S$ of the ground set, there is a partition
into $k$ bases such that the sizes of the intersections of the bases with $S$
may differ by at most one. This settles the matroid equitability conjecture by
Fekete and Szab\'o (Electron.~J.~Comb.~2011) in the affirmative. We also
investigate equitable splittings of two disjoint sets $S_1$ and $S_2$, and show
that there is a partition into $k$ bases such that the sizes of the
intersections with $S_1$ may differ by at most one and the sizes of the
intersections with $S_2$ may differ by at most two; this is the best possible
one can hope for arbitrary matroids.
  We also derive applications of this result into matroid constrained fair
division problems. We show that there exists a matroid-constrained fair
division that is envy-free up to 1 item if the valuations are identical and
tri-valued additive. We also show that for bi-valued additive valuations, there
exists a matroid-constrained allocation that provides everyone their maximin
share.

### Data Structures and Algorithms

### 1. [Pathfinding in Self-Deleting Graphs](http://arxiv.org/pdf/2507.12047v1)

Authors: Michal Dvořák, Dušan Knop, Michal Opler, Jan Pokorný, Ondřej Suchý, Krisztina Szilágyi

In this paper, we study the problem of pathfinding on traversal-dependent
graphs, i.e., graphs whose edges change depending on the previously visited
vertices. In particular, we study \emph{self-deleting graphs}, introduced by
Carmesin et al. (Sarah Carmesin, David Woller, David Parker, Miroslav Kulich,
and Masoumeh Mansouri. The Hamiltonian cycle and travelling salesperson
problems with traversal-dependent edge deletion. J. Comput. Sci.), which
consist of a graph $G=(V, E)$ and a function $f\colon V\rightarrow 2^E$, where
$f(v)$ is the set of edges that will be deleted after visiting the vertex $v$.
In the \textsc{(Shortest) Self-Deleting $s$-$t$-path} problem we are given a
self-deleting graph and its vertices $s$ and $t$, and we are asked to find a
(shortest) path from $s$ to $t$, such that it does not traverse an edge in
$f(v)$ after visiting $v$ for any vertex $v$.
  We prove that \textsc{Self-Deleting $s$-$t$-path} is NP-hard even if the
given graph is outerplanar, bipartite, has maximum degree $3$, bandwidth $2$
and $|f(v)|\leq 1$ for each vertex $v$. We show that \textsc{Shortest
Self-Deleting $s$-$t$-path} is W[1]-complete parameterized by the length of the
sought path and that \textsc{Self-Deleting $s$-$t$-path} is \W{1}-complete
parameterized by the vertex cover number, feedback vertex set number and
treedepth. We also show that the problem becomes FPT when we parameterize by
the maximum size of $f(v)$ and several structural parameters. Lastly, we show
that the problem does not admit a polynomial kernel even for parameterization
by the vertex cover number and the maximum size of $f(v)$ combined already on
2-outerplanar graphs.

### 2. [Weighted $k$-Server Admits an Exponentially Competitive Algorithm](http://arxiv.org/pdf/2507.12130v1)

Authors: Adithya Bijoy, Ankit Mondal, Ashish Chiplunkar

The weighted $k$-server is a variant of the $k$-server problem, where the
cost of moving a server is the server's weight times the distance through which
it moves. The problem is famous for its intriguing properties and for evading
standard techniques for designing and analyzing online algorithms. Even on
uniform metric spaces with sufficiently many points, the deterministic
competitive ratio of weighted $k$-server is known to increase doubly
exponentially with respect to $k$, while the behavior of its randomized
competitive ratio is not fully understood. Specifically, no upper bound better
than doubly exponential is known, while the best known lower bound is singly
exponential in $k$. In this paper, we close the exponential gap between these
bounds by giving an $\exp(O(k^2))$-competitive randomized online algorithm for
the weighted $k$-server problem on uniform metrics, thus breaking the doubly
exponential barrier for deterministic algorithms for the first time. This is
achieved by a recursively defined notion of a phase which, on the one hand,
forces a lower bound on the cost of any offline solution, while, on the other
hand, also admits a randomized online algorithm with bounded expected cost. The
algorithm is also recursive; it involves running several algorithms virtually
and in parallel and following the decisions of one of them in a random order.
We also show that our techniques can be lifted to construct an
$\exp(O(k^2))$-competitive randomized online algorithm for the generalized
$k$-server problem on weighted uniform metrics.

### 3. [Kernelization for list $H$-coloring for graphs with small vertex cover](http://arxiv.org/pdf/2507.12005v1)

Authors: Marta Piecyk, Astrid Pieterse, Paweł Rzążewski, Magnus Wahlström

For a fixed graph $H$, in the List $H$-Coloring problem, we are given a graph
$G$ along with list $L(v) \subseteq V(H)$ for every $v \in V(G)$, and we have
to determine if there exists a list homomorphism $\varphi$ from $(G,L)$ to $H$,
i.e., an edge preserving mapping $\varphi: V(G)\to V(H)$ that satisfies
$\varphi(v)\in L(v)$ for every $v\in V(G)$. Note that if $H$ is the complete
graph on $q$ vertices, the problem is equivalent to List $q$-Coloring. We
investigate the kernelization properties of List $H$-Coloring parameterized by
the vertex cover number of $G$: given an instance $(G,L)$ and a vertex cover of
$G$ of size $k$, can we reduce $(G,L)$ to an equivalent instance $(G',L')$ of
List $H$-Coloring where the size of $G'$ is bounded by a low-degree polynomial
$p(k)$ in $k$? This question has been investigated previously by Jansen and
Pieterse [Algorithmica 2019], who provided an upper bound, which turns out to
be optimal if $H$ is a complete graph, i.e., for List $q$-Coloring. This result
was one of the first applications of the method of kernelization via
bounded-degree polynomials. We define two new integral graph invariants,
$c^*(H)$ and $d^*(H)$, with $d^*(H) \leq c^*(H) \leq d^*(H)+1$, and show that
for every graph $H$, List $H$-Coloring
  -- has a kernel with $\mathcal{O}(k^{c^*(H)})$ vertices,
  -- admits no kernel of size $\mathcal{O}(k^{d^*(H)-\varepsilon})$ for any
$\varepsilon > 0$, unless the polynomial hierarchy collapses.
  -- Furthermore, if $c^*(H) > d^*(H)$, then there is a kernel with
$\mathcal{O}(k^{c^*(H)-\varepsilon})$ vertices where $\varepsilon \geq
2^{1-c^*(H)}$.
  Additionally, we show that for some classes of graphs, including powers of
cycles and graphs $H$ where $\Delta(H) \leq c^*(H)$ (which in particular
includes cliques), the bound $d^*(H)$ is tight, using the polynomial method. We
conjecture that this holds in general.

### 4. [FastReChain: Highly Responsive and Low-Overhead Centralized Route Scheduling in Clos Datacenter Networks](http://arxiv.org/pdf/2507.12265v1)

Authors: Zihan Zhu, Dongchao Wu, Zhanbang Zhang, Jian Yang

Ever since Clos topologies were used in datacenter networks (DCNs), a
practical centralized scheduling algorithm that supports dynamic scheduling has
been absent. The introduction of optical switches in DCNs as a future-proof
solution exacerbates this problem due to several properties of optical
switches, such as the fact that they are generally bufferless and therefore
rely on centralized scheduling, and that they have long switching times and
therefore require the number of rearrangements to be minimized.
  In this paper, we propose a centralized scheduling algorithm that achieves
theoretical maximum throughput even in one-rate bidirectional Clos networks,
while producing schemes with near-minimal numbers of rearrangements. It is the
only algorithm that directly supports bidirectional Clos networks and has a
time efficiency high enough to support dynamic scheduling to date. For static
minimal rewiring, its running time ranges from a fraction to a few hundredths
of other algorithms, and the number of rearrangements has also been steadily
improved, allowing for more frequent adjustments and less impact on ongoing
communications. In addition, the algorithm is very flexible and can support
various functional requirements in real-world environments. We achieve this
result through the replacement chain concept and bitset optimization.

### 5. [A near-complete resolution of the exponential-time complexity of k-opt for the traveling salesman problem](http://arxiv.org/pdf/2507.12304v1)

Authors: Sophia Heimann, Hung P. Hoang, Stefan Hougardy

The $k$-opt algorithm is one of the simplest and most widely used heuristics
for solving the traveling salesman problem. Starting from an arbitrary tour,
the $k$-opt algorithm improves the current tour in each iteration by exchanging
up to $k$ edges. The algorithm continues until no further improvement of this
kind is possible. For a long time, it remained an open question how many
iterations the $k$-opt algorithm might require for small values of $k$,
assuming the use of an optimal pivot rule. In this paper, we resolve this
question for the cases $k = 3$ and $k = 4$ by proving that in both these cases
an exponential number of iterations may be needed even if an optimal pivot rule
is used. Combined with a recent result from Heimann, Hoang, and Hougardy (ICALP
2024), this provides a complete answer for all $k \geq 3$ regarding the number
of iterations the $k$-opt algorithm may require under an optimal pivot rule. In
addition we establish an analogous exponential lower bound for the 2.5-opt
algorithm, a variant that generalizes 2-opt and is a restricted version of
3-opt. All our results hold for both the general and the metric traveling
salesman problem.

### 6. [Online Block Packing](http://arxiv.org/pdf/2507.12357v1)

Authors: Ariel Ben Eliezer, Noam Nisan

We consider the algorithmic challenge that is faced by blockchains that have
multidimensional block constraints and serve quasi-patient bidders. We provide
online approximation algorithms for this problem, thus solving open problems
left by [Babaioff and Nisan, EC 2025].

### Emerging Technologies

### 1. [Generative Intelligence Systems in the Flow of Group Emotions](http://arxiv.org/pdf/2507.11831v1)

Authors: Fernando Koch, Jessica Nahulan, Jeremy Fox, Martin Keen

Emotional cues frequently arise and shape group dynamics in interactive
settings where multiple humans and artificial agents communicate through shared
digital channels. While artificial agents lack intrinsic emotional states, they
can simulate affective behavior using synthetic modalities such as text or
speech. This work introduces a model for orchestrating emotion contagion,
enabling agents to detect emotional signals, infer group mood patterns, and
generate targeted emotional responses. The system captures human emotional
exchanges and uses this insight to produce adaptive, generative responses that
influence group affect in real time. The model supports applications in
collaborative, educational, and social environments by shifting affective
computing from individual-level reactions to coordinated, group-level emotion
modulation. We present the system architecture and provide experimental results
that illustrate its effectiveness in sensing and steering group mood dynamics.

### 2. [Emerging Paradigms in the Energy Sector: Forecasting and System Control Optimisation](http://arxiv.org/pdf/2507.12373v1)

Authors: Dariush Pourkeramati, Gareth Wadge, Rachel Hassall, Charlotte Mitchell, Anish Khadka, Shiwang Jaiswal, Andrew Duncan, Rossella Arcucci

The energy sector is experiencing rapid transformation due to increasing
renewable energy integration, decentralisation of power systems, and a
heightened focus on efficiency and sustainability. With energy demand becoming
increasingly dynamic and generation sources more variable, advanced forecasting
and optimisation strategies are crucial for maintaining grid stability,
cost-effectiveness, and environmental sustainability. This paper explores
emerging paradigms in energy forecasting and management, emphasizing four
critical domains: Energy Demand Forecasting integrated with Weather Data,
Building Energy Optimisation, Heat Network Optimisation, and Energy Management
System (EMS) Optimisation within a System of Systems (SoS) framework.
Leveraging machine learning techniques and Model Predictive Control (MPC), the
study demonstrates substantial enhancements in energy efficiency across scales
-- from individual buildings to complex interconnected energy networks.
Weather-informed demand forecasting significantly improves grid resilience and
resource allocation strategies. Smart building optimisation integrates
predictive analytics to substantially reduce energy consumption without
compromising occupant comfort. Optimising CHP-based heat networks achieves cost
and carbon savings while adhering to operational and asset constraints. At the
systems level, sophisticated EMS optimisation ensures coordinated control of
distributed resources, storage solutions, and demand-side flexibility. Through
real-world case studies we highlight the potential of AI-driven automation and
integrated control solutions in facilitating a resilient, efficient, and
sustainable energy future.

### 3. [Trustworthy Tree-based Machine Learning by $MoS_2$ Flash-based Analog CAM with Inherent Soft Boundaries](http://arxiv.org/pdf/2507.12384v1)

Authors: Bo Wen, Guoyun Gao, Zhicheng Xu, Ruibin Mao, Xiaojuan Qi, X. Sharon Hu, Xunzhao Yin, Can Li

The rapid advancement of artificial intelligence has raised concerns
regarding its trustworthiness, especially in terms of interpretability and
robustness. Tree-based models like Random Forest and XGBoost excel in
interpretability and accuracy for tabular data, but scaling them remains
computationally expensive due to poor data locality and high data dependence.
Previous efforts to accelerate these models with analog content addressable
memory (CAM) have struggled, due to the fact that the difficult-to-implement
sharp decision boundaries are highly susceptible to device variations, which
leads to poor hardware performance and vulnerability to adversarial attacks.
This work presents a novel hardware-software co-design approach using $MoS_2$
Flash-based analog CAM with inherent soft boundaries, enabling efficient
inference with soft tree-based models. Our soft tree model inference
experiments on $MoS_2$ analog CAM arrays show this method achieves exceptional
robustness against device variation and adversarial attacks while achieving
state-of-the-art accuracy. Specifically, our fabricated analog CAM arrays
achieve $96\%$ accuracy on Wisconsin Diagnostic Breast Cancer (WDBC) database,
while maintaining decision explainability. Our experimentally calibrated model
validated only a $0.6\%$ accuracy drop on the MNIST dataset under $10\%$ device
threshold variation, compared to a $45.3\%$ drop for traditional decision
trees. This work paves the way for specialized hardware that enhances AI's
trustworthiness and efficiency.

### Formal Languages and Automata Theory

### 1. [Hyper pattern matching](http://arxiv.org/pdf/2507.12102v1)

Authors: Masaki Waga, Étienne André

In runtime verification, pattern matching, which searches for occurrences of
a specific pattern within a word, provides more information than a simple
violation detection of the monitored property, by locating concrete evidence of
the violation. However, witnessing violations of some properties, particularly
hyperproperties, requires evidence across multiple input words or different
parts of the same word, which goes beyond the scope of conventional pattern
matching. We propose here hyper pattern matching, a generalization of pattern
matching over a set of words. Properties of interest include robustness and
(non-)interference. As a formalism for patterns, we use nondeterministic
asynchronous finite automata (NAAs). We first provide a naive algorithm for
hyper pattern matching and then devise several heuristics for better
efficiency. Although we prove the NP-completeness of the problem, our
implementation HypPAu is able to address several case studies scalable in the
length, number of words (or logs) and number of dimensions, suggesting the
practical relevance of our approach.

### 2. [Syntax Repair as Language Intersection](http://arxiv.org/pdf/2507.11873v1)

Authors: Breandan Considine

We introduce a new technique for repairing syntax errors in arbitrary
context-free languages. This technique models syntax repair as a language
intersection problem by defining a finite language that provably generates
every syntactically valid repair within a given edit distance. Leveraging a
theoretical connection between the Bar-Hillel construction from formal language
theory and CFL reachability from program analysis, we show that repairability
in a finite number of typographic edits is polylogarithmic parallel time
decidable and provide an enumeration algorithm based on the Brzozowski
derivative. Finally, we evaluate this algorithm and its implementation,
demonstrating state-of-the-art results on a Python syntax repair benchmark.

### Graphics

### 1. [SmokeSVD: Smoke Reconstruction from A Single View via Progressive Novel View Synthesis and Refinement with Diffusion Models](http://arxiv.org/pdf/2507.12156v1)

Authors: Chen Li, Shanshan Dong, Sheng Qiu, Jianmin Han, Zan Gao, Kemeng Huang, Taku Komura

Reconstructing dynamic fluids from sparse views is a long-standing and
challenging problem, due to the severe lack of 3D information from insufficient
view coverage. While several pioneering approaches have attempted to address
this issue using differentiable rendering or novel view synthesis, they are
often limited by time-consuming optimization and refinement processes under
ill-posed conditions. To tackle above challenges, we propose SmokeSVD, an
efficient and effective framework to progressively generate and reconstruct
dynamic smoke from a single video by integrating both the powerful generative
capabilities from diffusion models and physically guided consistency
optimization towards realistic appearance and dynamic evolution. Specifically,
we first propose a physically guided side-view synthesizer based on diffusion
models, which explicitly incorporates divergence and gradient guidance of
velocity fields to generate visually realistic and spatio-temporally consistent
side-view images frame by frame, significantly alleviating the ill-posedness of
single-view reconstruction without imposing additional constraints.
Subsequently, we determine a rough estimation of density field from the pair of
front-view input and side-view synthetic image, and further refine 2D blurry
novel-view images and 3D coarse-grained density field through an iterative
process that progressively renders and enhances the images from increasing
novel viewing angles, generating high-quality multi-view image sequences.
Finally, we reconstruct and estimate the fine-grained density field, velocity
field, and smoke source via differentiable advection by leveraging the
Navier-Stokes equations. Extensive quantitative and qualitative experiments
show that our approach achieves high-quality reconstruction and outperforms
previous state-of-the-art techniques.

### 2. [Shape Adaptation for 3D Hairstyle Retargeting](http://arxiv.org/pdf/2507.12168v1)

Authors: Lu Yu, Zhong Ren, Youyi Zheng, Xiang Chen, Kun Zhou

It is demanding to author an existing hairstyle for novel characters in games
and VR applications. However, it is a non-trivial task for artists due to the
complicated hair geometries and spatial interactions to preserve. In this
paper, we present an automatic shape adaptation method to retarget 3D
hairstyles. We formulate the adaptation process as a constrained optimization
problem, where all the shape properties and spatial relationships are converted
into individual objectives and constraints. To make such an optimization on
high-resolution hairstyles tractable, we adopt a multi-scale strategy to
compute the target positions of the hair strands in a coarse-to-fine manner.
The global solving for the inter-strands coupling is restricted to the coarse
level, and the solving for fine details is made local and parallel. In
addition, we present a novel hairline edit tool to allow for user customization
during retargeting. We achieve it by solving physics-based deformations of an
embedded membrane to redistribute the hair roots with minimal distortion. We
demonstrate the efficacy of our method through quantitative and qualitative
experiments on various hairstyles and characters.

### 3. [Measuring and predicting visual fidelity](http://arxiv.org/pdf/2507.11857v1)

Authors: Benjamin Watson, Alinda Friedman, Aaron McGaffey

This paper is a study of techniques for measuring and predicting visual
fidelity. As visual stimuli we use polygonal models, and vary their fidelity
with two different model simplification algorithms. We also group the stimuli
into two object types: animals and man made artifacts. We examine three
different experimental techniques for measuring these fidelity changes: naming
times, ratings, and preferences. All the measures were sensitive to the type of
simplification and level of simplification. However, the measures differed from
one another in their response to object type. We also examine several automatic
techniques for predicting these experimental measures, including techniques
based on images and on the models themselves. Automatic measures of fidelity
were successful at predicting experimental ratings, less successful at
predicting preferences, and largely failures at predicting naming times. We
conclude with suggestions for use and improvement of the experimental and
automatic measures of visual fidelity.

### 4. [HPR3D: Hierarchical Proxy Representation for High-Fidelity 3D Reconstruction and Controllable Editing](http://arxiv.org/pdf/2507.11971v1)

Authors: Tielong Wang, Yuxuan Xiong, Jinfan Liu, Zhifan Zhang, Ye Chen, Yue Shi, Bingbing Ni

Current 3D representations like meshes, voxels, point clouds, and NeRF-based
neural implicit fields exhibit significant limitations: they are often
task-specific, lacking universal applicability across reconstruction,
generation, editing, and driving. While meshes offer high precision, their
dense vertex data complicates editing; NeRFs deliver excellent rendering but
suffer from structural ambiguity, hindering animation and manipulation; all
representations inherently struggle with the trade-off between data complexity
and fidelity. To overcome these issues, we introduce a novel 3D Hierarchical
Proxy Node representation. Its core innovation lies in representing an object's
shape and texture via a sparse set of hierarchically organized
(tree-structured) proxy nodes distributed on its surface and interior. Each
node stores local shape and texture information (implicitly encoded by a small
MLP) within its neighborhood. Querying any 3D coordinate's properties involves
efficient neural interpolation and lightweight decoding from relevant nearby
and parent nodes. This framework yields a highly compact representation where
nodes align with local semantics, enabling direct drag-and-edit manipulation,
and offers scalable quality-complexity control. Extensive experiments across 3D
reconstruction and editing demonstrate our method's expressive efficiency,
high-fidelity rendering quality, and superior editability.

### 5. [MOSPA: Human Motion Generation Driven by Spatial Audio](http://arxiv.org/pdf/2507.11949v1)

Authors: Shuyang Xu, Zhiyang Dou, Mingyi Shi, Liang Pan, Leo Ho, Jingbo Wang, Yuan Liu, Cheng Lin, Yuexin Ma, Wenping Wang, Taku Komura

Enabling virtual humans to dynamically and realistically respond to diverse
auditory stimuli remains a key challenge in character animation, demanding the
integration of perceptual modeling and motion synthesis. Despite its
significance, this task remains largely unexplored. Most previous works have
primarily focused on mapping modalities like speech, audio, and music to
generate human motion. As of yet, these models typically overlook the impact of
spatial features encoded in spatial audio signals on human motion. To bridge
this gap and enable high-quality modeling of human movements in response to
spatial audio, we introduce the first comprehensive Spatial Audio-Driven Human
Motion (SAM) dataset, which contains diverse and high-quality spatial audio and
motion data. For benchmarking, we develop a simple yet effective
diffusion-based generative framework for human MOtion generation driven by
SPatial Audio, termed MOSPA, which faithfully captures the relationship between
body motion and spatial audio through an effective fusion mechanism. Once
trained, MOSPA could generate diverse realistic human motions conditioned on
varying spatial audio inputs. We perform a thorough investigation of the
proposed dataset and conduct extensive experiments for benchmarking, where our
method achieves state-of-the-art performance on this task. Our model and
dataset will be open-sourced upon acceptance. Please refer to our supplementary
video for more details.

### Computer Science and Game Theory

### 1. [Coalitions on the Fly in Cooperative Games](http://arxiv.org/pdf/2507.11883v1)

Authors: Yao Zhang, Indrajit Saha, Zhaohong Sun, Makoto Yokoo

In this work, we examine a sequential setting of a cooperative game in which
players arrive dynamically to form coalitions and complete tasks either
together or individually, depending on the value created. Upon arrival, a new
player as a decision maker faces two options: forming a new coalition or
joining an existing one. We assume that players are greedy, i.e., they aim to
maximize their rewards based on the information available at their arrival. The
objective is to design an online value distribution policy that incentivizes
players to form a coalition structure that maximizes social welfare. We focus
on monotone and bounded cooperative games. Our main result establishes an upper
bound of $\frac{3\mathsf{min}}{\mathsf{max}}$ on the competitive ratio for any
irrevocable policy (i.e., one without redistribution), and proposes a policy
that achieves a near-optimal competitive ratio of $\min\left\{\frac{1}{2},
\frac{3\mathsf{min}}{\mathsf{max}}\right\}$, where $\mathsf{min}$ and
$\mathsf{max}$ denote the smallest and largest marginal contribution of any
sub-coalition of players respectively. Finally, we also consider
non-irrevocable policies, with alternative bounds only when the number of
players is limited.

### 2. [Contracting with a Mechanism Designer](http://arxiv.org/pdf/2507.12054v1)

Authors: Tian Bai, Yiding Feng, Yaohao Liu, Mengfan Ma, Mingyu Xiao

This paper explores the economic interactions within modern crowdsourcing
markets. In these markets, employers issue requests for tasks, platforms
facilitate the recruitment of crowd workers, and workers complete tasks for
monetary rewards. Recognizing that these roles serve distinct functions within
the ecosystem, we introduce a three-party model that distinguishes among the
principal (the requester), the intermediary (the platform), and the pool of
agents (the workers). The principal, unable to directly engage with agents,
relies on the intermediary to recruit and incentivize them. This interaction
unfolds in two stages: first, the principal designs a profit-sharing contract
with the intermediary; second, the intermediary implements a mechanism to
select an agent to complete the delegated task.
  We analyze the proposed model as an extensive-form Stackelberg game. Our
contributions are fourfold: (1) We fully characterize the subgame perfect
equilibrium. In particular, we reduce the principal's contract design problem
to a novel auction-theoretic formulation we term virtual value pricing, and
reveals that linear contracts are optimal even when the task have multiple
outcomes and agents' cost distributions are asymmetric. (2) To quantify the
principal's utility loss from delegation and information asymmetry, we
introduce the price of double marginalization (PoDM) and the classical price of
anarchy (PoA), and derive tight or nearly tight bounds on both ratios under
regular and monotone hazard rate (MHR) distributions. (3) We further examine
these two ratios in a natural setting where the intermediary is restricted to
anonymous pricing mechanisms, and show that similar qualitative insights
continue to hold. (4) Finally, we extend our results on both ratios to a robust
framework that accommodates scenarios in which the principal lacks precise
information about the market size.

### 3. [New allocation rule based on graph structures and their application to economic phenomena](http://arxiv.org/pdf/2507.11808v1)

Authors: Taiki Yamada, Taisuke Matsubae, Tomoya Akamatsu

This study introduces the \emph{edge-based Shapley value}, a novel allocation
rule within cooperative game theory, specifically tailored for networked
systems, where value is generated through interactions represented by edges.
Traditional allocation rules, such as the Shapley and Myerson values, evaluate
player contributions based on node-level characteristics, or connected
components. However, these approaches often fail to adequately capture the
functional role of edges, which are crucial in systems such as supply chains
and digital platforms, where interactions, rather than individual agents, are
the primary drivers of value. Our edge-based Shapley value shifts the
characteristic function from node sets to edge sets, thereby enabling a more
granular and context-sensitive evaluation of the contributions. We establish
its theoretical foundations, demonstrate its relationship to classical
allocation rules, and show that it retains key properties such as fairness and
symmetry. To illustrate its applicability, we present two use cases: content
platform networks and supply chain logistics (SCL). In both cases, our method
produces intuitive and structurally consistent allocations, particularly in
scenarios with overlapping routes, exclusive contracts or cost-sensitive paths.
This framework offers a new perspective on value attribution in cooperative
settings with complex interaction structures and provides practical tools for
analyzing real-world economic and logistical networks.

### 4. [Measuring Informativeness Gap of (Mis)Calibrated Predictors](http://arxiv.org/pdf/2507.12094v1)

Authors: Yiding Feng, Wei Tang

In many applications, decision-makers must choose between multiple predictive
models that may all be miscalibrated. Which model (i.e., predictor) is more
"useful" in downstream decision tasks? To answer this, our first contribution
introduces the notion of the informativeness gap between any two predictors,
defined as the maximum normalized payoff advantage one predictor offers over
the other across all decision-making tasks. Our framework strictly generalizes
several existing notions: it subsumes U-Calibration [KLST-23] and Calibration
Decision Loss [HW-24], which compare a miscalibrated predictor to its
calibrated counterpart, and it recovers Blackwell informativeness [Bla-51,
Bla-53] as a special case when both predictors are perfectly calibrated. Our
second contribution is a dual characterization of the informativeness gap,
which gives rise to a natural informativeness measure that can be viewed as a
relaxed variant of the earth mover's distance (EMD) between two prediction
distributions. We show that this measure satisfies natural desiderata: it is
complete and sound, and it can be estimated sample-efficiently in the
prediction-only access setting. Along the way, we also obtain novel
combinatorial structural results when applying this measure to perfectly
calibrated predictors.

### 5. [Online Block Packing](http://arxiv.org/pdf/2507.12357v1)

Authors: Ariel Ben Eliezer, Noam Nisan

We consider the algorithmic challenge that is faced by blockchains that have
multidimensional block constraints and serve quasi-patient bidders. We provide
online approximation algorithms for this problem, thus solving open problems
left by [Babaioff and Nisan, EC 2025].

### 6. [Matroids are Equitable](http://arxiv.org/pdf/2507.12100v1)

Authors: Hannaneh Akrami, Roshan Raj, László A. Végh

We show that if the ground set of a matroid can be partitioned into $k\ge 2$
bases, then for any given subset $S$ of the ground set, there is a partition
into $k$ bases such that the sizes of the intersections of the bases with $S$
may differ by at most one. This settles the matroid equitability conjecture by
Fekete and Szab\'o (Electron.~J.~Comb.~2011) in the affirmative. We also
investigate equitable splittings of two disjoint sets $S_1$ and $S_2$, and show
that there is a partition into $k$ bases such that the sizes of the
intersections with $S_1$ may differ by at most one and the sizes of the
intersections with $S_2$ may differ by at most two; this is the best possible
one can hope for arbitrary matroids.
  We also derive applications of this result into matroid constrained fair
division problems. We show that there exists a matroid-constrained fair
division that is envy-free up to 1 item if the valuations are identical and
tri-valued additive. We also show that for bi-valued additive valuations, there
exists a matroid-constrained allocation that provides everyone their maximin
share.

### 7. [A Bayesian Incentive Mechanism for Poison-Resilient Federated Learning](http://arxiv.org/pdf/2507.12439v1)

Authors: Daniel Commey, Rebecca A. Sarpong, Griffith S. Klogo, Winful Bagyl-Bac, Garth V. Crosby

Federated learning (FL) enables collaborative model training across
decentralized clients while preserving data privacy. However, its
open-participation nature exposes it to data-poisoning attacks, in which
malicious actors submit corrupted model updates to degrade the global model.
Existing defenses are often reactive, relying on statistical aggregation rules
that can be computationally expensive and that typically assume an honest
majority. This paper introduces a proactive, economic defense: a lightweight
Bayesian incentive mechanism that makes malicious behavior economically
irrational. Each training round is modeled as a Bayesian game of incomplete
information in which the server, acting as the principal, uses a small, private
validation dataset to verify update quality before issuing payments. The design
satisfies Individual Rationality (IR) for benevolent clients, ensuring their
participation is profitable, and Incentive Compatibility (IC), making poisoning
an economically dominated strategy. Extensive experiments on non-IID partitions
of MNIST and FashionMNIST demonstrate robustness: with 50% label-flipping
adversaries on MNIST, the mechanism maintains 96.7% accuracy, only 0.3
percentage points lower than in a scenario with 30% label-flipping adversaries.
This outcome is 51.7 percentage points better than standard FedAvg, which
collapses under the same 50% attack. The mechanism is computationally light,
budget-bounded, and readily integrates into existing FL frameworks, offering a
practical route to economically robust and sustainable FL ecosystems.

### Human-Computer Interaction

### 1. [Envisage: Towards Expressive Visual Graph Querying](http://arxiv.org/pdf/2507.11999v1)

Authors: Xiaolin Wen, Qishuang Fu, Shuangyue Han, Yichen Guo, Joseph K. Liu, Yong Wang

Graph querying is the process of retrieving information from graph data using
specialized languages (e.g., Cypher), often requiring programming expertise.
Visual Graph Querying (VGQ) streamlines this process by enabling users to
construct and execute queries via an interactive interface without resorting to
complex coding. However, current VGQ tools only allow users to construct simple
and specific query graphs, limiting users' ability to interactively express
their query intent, especially for underspecified query intent. To address
these limitations, we propose Envisage, an interactive visual graph querying
system to enhance the expressiveness of VGQ in complex query scenarios by
supporting intuitive graph structure construction and flexible parameterized
rule specification. Specifically, Envisage comprises four stages: Query
Expression allows users to interactively construct graph queries through
intuitive operations; Query Verification enables the validation of constructed
queries via rule verification and query instantiation; Progressive Query
Execution can progressively execute queries to ensure meaningful querying
results; and Result Analysis facilitates result exploration and interpretation.
To evaluate Envisage, we conducted two case studies and in-depth user
interviews with 14 graph analysts. The results demonstrate its effectiveness
and usability in constructing, verifying, and executing complex graph queries.

### 2. [Tao-Technology for Teen Mobile Use: Harmonizing Adaptation, Autonomy, and Reflection](http://arxiv.org/pdf/2507.12204v1)

Authors: Pengyu Zhu, Janghee Cho

Adolescents' mobile technology use is often regulated through rigid control
mechanisms that fail to account for their autonomy and natural usage patterns.
Drawing on Taoist philosophy, particularly Wu Wei, Yin-Yang, and Zi Ran, this
position paper proposes Tao-Technology, a self-organizing, adaptive regulatory
framework. Integrating insights from Reflective Informatics and Information
Ecologies, we explore how mobile technology can dynamically adjust to context
while fostering self-reflection and meaning-making. This approach shifts from
external restrictions to dynamic co-adaptative regulation, ensuring technology
governance remains flexible yet structured, supporting adolescents in
cultivating a balanced and intentional relationship with digital technology.

### 3. [Humans are more gullible than LLMs in believing common psychological myths](http://arxiv.org/pdf/2507.12296v1)

Authors: Bevan Koopman, Guido Zuccon

Despite widespread debunking, many psychological myths remain deeply
entrenched. This paper investigates whether Large Language Models (LLMs) mimic
human behaviour of myth belief and explores methods to mitigate such
tendencies. Using 50 popular psychological myths, we evaluate myth belief
across multiple LLMs under different prompting strategies, including
retrieval-augmented generation and swaying prompts. Results show that LLMs
exhibit significantly lower myth belief rates than humans, though user
prompting can influence responses. RAG proves effective in reducing myth belief
and reveals latent debiasing potential within LLMs. Our findings contribute to
the emerging field of Machine Psychology and highlight how cognitive science
methods can inform the evaluation and development of LLM-based systems.

### 4. [TrialCompass: Visual Analytics for Enhancing the Eligibility Criteria Design of Clinical Trials](http://arxiv.org/pdf/2507.12298v1)

Authors: Rui Sheng, Xingbo Wang, Jiachen Wang, Xiaofu Jin, Zhonghua Sheng, Zhenxing Xu, Suraj Rajendran, Huamin Qu, Fei Wang

Eligibility criteria play a critical role in clinical trials by determining
the target patient population, which significantly influences the outcomes of
medical interventions. However, current approaches for designing eligibility
criteria have limitations to support interactive exploration of the large space
of eligibility criteria. They also ignore incorporating detailed
characteristics from the original electronic health record (EHR) data for
criteria refinement. To address these limitations, we proposed TrialCompass, a
visual analytics system integrating a novel workflow, which can empower
clinicians to iteratively explore the vast space of eligibility criteria
through knowledge-driven and outcome-driven approaches. TrialCompass supports
history-tracking to help clinicians trace the evolution of their adjustments
and decisions when exploring various forms of data (i.e., eligibility criteria,
outcome metrics, and detailed characteristics of original EHR data) through
these two approaches. This feature can help clinicians comprehend the impact of
eligibility criteria on outcome metrics and patient characteristics, which
facilitates systematic refinement of eligibility criteria. Using a real-world
dataset, we demonstrated the effectiveness of TrialCompass in providing
insights into designing eligibility criteria for septic shock and
sepsis-associated acute kidney injury. We also discussed the research prospects
of applying visual analytics to clinical trials.

### 5. [An Analysis of Text Functions in Information Visualization](http://arxiv.org/pdf/2507.12334v1)

Authors: Chase Stokes, Anjana Arunkumar, Marti A. Hearst, Lace Padilla

Text is an integral but understudied component of visualization design.
Although recent studies have examined how text elements (e.g., titles and
annotations) influence comprehension, preferences, and predictions, many
questions remain about textual design and use in practice. This paper
introduces a framework for understanding text functions in information
visualizations, building on and filling gaps in prior classifications and
taxonomies. Through an analysis of 120 real-world visualizations and 804 text
elements, we identified ten distinct text functions, ranging from identifying
data mappings to presenting valenced subtext. We further identify patterns in
text usage and conduct a factor analysis, revealing four overarching
text-informed design strategies: Attribution and Variables, Annotation-Centric
Design, Visual Embellishments, and Narrative Framing. In addition to these
factors, we explore features of title rhetoric and text multifunctionality,
while also uncovering previously unexamined text functions, such as text
replacing visual elements. Our findings highlight the flexibility of text,
demonstrating how different text elements in a given design can combine to
communicate, synthesize, and frame visual information. This framework adds
important nuance and detail to existing frameworks that analyze the diverse
roles of text in visualization.

### 6. [MExplore: an entity-based visual analytics approach for medical expertise acquisition](http://arxiv.org/pdf/2507.12337v1)

Authors: Xiao Pang, Yan Huang, Chang Liu, JiYuan Liu, MingYou Liu

Acquiring medical expertise is a critical component of medical education and
professional development. While existing studies focus primarily on
constructing medical knowledge bases or developing learning tools based on the
structured, private healthcare data, they often lack methods for extracting
expertise from unstructured medical texts. These texts constitute a significant
portion of medical literature and offer greater flexibility and detail compared
to structured data formats. Furthermore, many studies fail to provide explicit
analytical and learning pathways in this context.
  This paper introduces MExplore, an interactive visual analytics system
designed to support the acquisition of medical expertise. To address the
challenges of the inconsistencies and confidentiality concerns inherent in
unstructured medical texts, we propose a workflow that employs a fine-tuned
BERT-based model to extract medical entities (MEs) from them. We then present a
novel multilevel visual analysis framework that integrates multiple coordinated
visualizations, enabling a progressive and interactive exploration of medical
knowledge.
  To assess the effectiveness of MExplore, we conducted three case studies, a
user study, and interviews with domain experts. The results indicate that the
system significantly enhances the medical expertise acquisition process,
providing an effective interactive approach for acquiring and retaining
knowledge from medical texts.

### 7. [Deconstructing Implicit Beliefs in Visual Data Journalism: Unstable Meanings Behind Data as Truth & Design for Insight](http://arxiv.org/pdf/2507.12377v1)

Authors: Ke Er Amy Zhang, Jodie Jenkinson, Laura Garrison

We conduct a deconstructive reading of a qualitative interview study with 17
visual data journalists from newsrooms across the globe. We borrow a
deconstruction approach from literary critique to explore the instability of
meaning in language and reveal implicit beliefs in words and ideas. Through our
analysis we surface two sets of opposing implicit beliefs in visual data
journalism: objectivity/subjectivity and humanism/mechanism. We contextualize
these beliefs through a genealogical analysis, which brings deconstruction
theory into practice by providing a historic backdrop for these opposing
perspectives. Our analysis shows that these beliefs held within visual data
journalism are not self-enclosed but rather a product of external societal
forces and paradigm shifts over time. Through this work, we demonstrate how
thinking with critical theories such as deconstruction and genealogy can
reframe "success" in visual data storytelling and diversify visualization
research outcomes. These efforts push the ways in which we as researchers
produce domain knowledge to examine the sociotechnical issues of today's values
towards datafication and data visualization.

### 8. ["Mapping What I Feel": Understanding Affective Geovisualization Design Through the Lens of People-Place Relationships](http://arxiv.org/pdf/2507.11841v1)

Authors: Xingyu Lan, Yutong Yang, Yifan Wang

Affective visualization design is an emerging research direction focused on
communicating and influencing emotion through visualization. However, as
revealed by previous research, this area is highly interdisciplinary and
involves theories and practices from diverse fields and disciplines, thus
awaiting analysis from more fine-grained angles. To address this need, this
work focuses on a pioneering and relatively mature sub-area, affective
geovisualization design, to further the research in this direction and provide
more domain-specific insights. Through an analysis of a curated corpus of
affective geovisualization designs using the Person-Process-Place (PPP) model
from geographic theory, we derived a design taxonomy that characterizes a
variety of methods for eliciting and enhancing emotions through geographic
visualization. We also identified four underlying high-level design paradigms
of affective geovisualization design (e.g., computational, anthropomorphic)
that guide distinct approaches to linking geographic information with human
experience. By extending existing affective visualization design frameworks
with geographic specificity, we provide additional design examples,
domain-specific analyses, and insights to guide future research and practices
in this underexplored yet highly innovative domain.

### 9. [Measuring and predicting visual fidelity](http://arxiv.org/pdf/2507.11857v1)

Authors: Benjamin Watson, Alinda Friedman, Aaron McGaffey

This paper is a study of techniques for measuring and predicting visual
fidelity. As visual stimuli we use polygonal models, and vary their fidelity
with two different model simplification algorithms. We also group the stimuli
into two object types: animals and man made artifacts. We examine three
different experimental techniques for measuring these fidelity changes: naming
times, ratings, and preferences. All the measures were sensitive to the type of
simplification and level of simplification. However, the measures differed from
one another in their response to object type. We also examine several automatic
techniques for predicting these experimental measures, including techniques
based on images and on the models themselves. Automatic measures of fidelity
were successful at predicting experimental ratings, less successful at
predicting preferences, and largely failures at predicting naming times. We
conclude with suggestions for use and improvement of the experimental and
automatic measures of visual fidelity.

### 10. [Unveiling the Visual Rhetoric of Persuasive Cartography: A Case Study of the Design of Octopus Maps](http://arxiv.org/pdf/2507.11903v1)

Authors: Daocheng Lin, Yifan Wang, Yutong Yang, Xingyu Lan

When designed deliberately, data visualizations can become powerful
persuasive tools, influencing viewers' opinions, values, and actions. While
researchers have begun studying this issue (e.g., to evaluate the effects of
persuasive visualization), we argue that a fundamental mechanism of persuasion
resides in rhetorical construction, a perspective inadequately addressed in
current visualization research. To fill this gap, we present a focused analysis
of octopus maps, a visual genre that has maintained persuasive power across
centuries and achieved significant social impact. Employing rhetorical schema
theory, we collected and analyzed 90 octopus maps spanning from the 19th
century to contemporary times. We closely examined how octopus maps implement
their persuasive intents and constructed a design space that reveals how visual
metaphors are strategically constructed and what common rhetorical strategies
are applied to components such as maps, octopus imagery, and text. Through the
above analysis, we also uncover a set of interesting findings. For instance,
contrary to the common perception that octopus maps are primarily a historical
phenomenon, our research shows that they remain a lively design convention in
today's digital age. Additionally, while most octopus maps stem from Western
discourse that views the octopus as an evil symbol, some designs offer
alternative interpretations, highlighting the dynamic nature of rhetoric across
different sociocultural settings. Lastly, drawing from the lessons provided by
octopus maps, we discuss the associated ethical concerns of persuasive
visualization.

### Information Retrieval

### 1. [Similarity-Guided Diffusion for Contrastive Sequential Recommendation](http://arxiv.org/pdf/2507.11866v1)

Authors: Jinkyeong Choi, Yejin Noh, Donghyeon Park

In sequential recommendation systems, data augmentation and contrastive
learning techniques have recently been introduced using diffusion models to
achieve robust representation learning. However, most of the existing
approaches use random augmentation, which risk damaging the contextual
information of the original sequence. Accordingly, we propose a
Similarity-Guided Diffusion for Contrastive Sequential Recommendation. Our
method leverages the similarity between item embedding vectors to generate
semantically consistent noise. Moreover, we utilize high confidence score in
the denoising process to select our augmentation positions. This approach more
effectively reflects contextual and structural information compared to
augmentation at random positions. From a contrastive learning perspective, the
proposed augmentation technique provides more discriminative positive and
negative samples, simultaneously improving training efficiency and
recommendation performance. Experimental results on five benchmark datasets
show that SimDiffRec outperforms the existing baseline models.

### 2. [An Ecosystem for Ontology Interoperability](http://arxiv.org/pdf/2507.12311v1)

Authors: Zhangcheng Qiang

Ontology interoperability is one of the complicated issues that restricts the
use of ontologies in knowledge graphs (KGs). Different ontologies with
conflicting and overlapping concepts make it difficult to design, develop, and
deploy an interoperable ontology for downstream tasks. We propose an ecosystem
for ontology interoperability. The ecosystem employs three state-of-the-art
semantic techniques in different phases of the ontology engineering life cycle:
ontology design patterns (ODPs) in the design phase, ontology matching and
versioning (OM\&OV) in the develop phase, and ontology-compliant knowledge
graphs (OCKGs) in the deploy phase, to achieve better ontology interoperability
in real-world applications. A case study in the building domain validates the
usefulness of the proposed ecosystem.

### 3. [SIEVE: Effective Filtered Vector Search with Collection of Indexes](http://arxiv.org/pdf/2507.11907v1)

Authors: Zhaoheng Li, Silu Huang, Wei Ding, Yongjoo Park, Jianjun Chen

Many real-world tasks such as recommending videos with the kids tag can be
reduced to finding most similar vectors associated with hard predicates. This
task, filtered vector search, is challenging as prior state-of-the-art
graph-based (unfiltered) similarity search techniques quickly degenerate when
hard constraints are considered. That is, effective graph-based filtered
similarity search relies on sufficient connectivity for reaching the most
similar items within just a few hops. To consider predicates, recent works
propose modifying graph traversal to visit only the items that may satisfy
predicates. However, they fail to offer the just-a-few-hops property for a wide
range of predicates: they must restrict predicates significantly or lose
efficiency if only a small fraction of items satisfy predicates.
  We propose an opposite approach: instead of constraining traversal, we build
many indexes each serving different predicate forms. For effective
construction, we devise a three-dimensional analytical model capturing
relationships among index size, search time, and recall, with which we follow a
workload-aware approach to pack as many useful indexes as possible into a
collection. At query time, the analytical model is employed yet again to
discern the one that offers the fastest search at a given recall. We show
superior performance and support on datasets with varying selectivities and
forms: our approach achieves up to 8.06x speedup while having as low as 1%
build time versus other indexes, with less than 2.15x memory of a standard HNSW
graph and modest knowledge of past workloads.

### 4. [Looking for Fairness in Recommender Systems](http://arxiv.org/pdf/2507.12242v1)

Authors: Cécile Logé

Recommender systems can be found everywhere today, shaping our everyday
experience whenever we're consuming content, ordering food, buying groceries
online, or even just reading the news. Let's imagine we're in the process of
building a recommender system to make content suggestions to users on social
media. When thinking about fairness, it becomes clear there are several
perspectives to consider: the users asking for tailored suggestions, the
content creators hoping for some limelight, and society at large, navigating
the repercussions of algorithmic recommendations. A shared fairness concern
across all three is the emergence of filter bubbles, a side-effect that takes
place when recommender systems are almost "too good", making recommendations so
tailored that users become inadvertently confined to a narrow set of
opinions/themes and isolated from alternative ideas. From the user's
perspective, this is akin to manipulation. From the small content creator's
perspective, this is an obstacle preventing them access to a whole range of
potential fans. From society's perspective, the potential consequences are
far-reaching, influencing collective opinions, social behavior and political
decisions. How can our recommender system be fine-tuned to avoid the creation
of filter bubbles, and ensure a more inclusive and diverse content landscape?
Approaching this problem involves defining one (or more) performance metric to
represent diversity, and tweaking our recommender system's performance through
the lens of fairness. By incorporating this metric into our evaluation
framework, we aim to strike a balance between personalized recommendations and
the broader societal goal of fostering rich and varied cultures and points of
view.

### 5. [Developing Visual Augmented Q&A System using Scalable Vision Embedding Retrieval & Late Interaction Re-ranker](http://arxiv.org/pdf/2507.12378v1)

Authors: Rachna Saxena, Abhijeet Kumar, Suresh Shanmugam

Traditional information extraction systems face challenges with text only
language models as it does not consider infographics (visual elements of
information) such as tables, charts, images etc. often used to convey complex
information to readers. Multimodal LLM (MLLM) face challenges of finding needle
in the haystack problem i.e., either longer context length or substantial
number of documents as search space. Late interaction mechanism over visual
language models has shown state of the art performance in retrieval-based
vision augmented Q&A tasks. There are yet few challenges using it for RAG based
multi-modal Q&A. Firstly, many popular and widely adopted vector databases do
not support native multi-vector retrieval. Secondly, late interaction requires
computation which inflates space footprint and can hinder enterprise adoption.
Lastly, the current state of late interaction mechanism does not leverage the
approximate neighbor search indexing methods for large speed ups in retrieval
process. This paper explores a pragmatic approach to make vision retrieval
process scalable and efficient without compromising on performance quality. We
propose multi-step custom implementation utilizing widely adopted hybrid search
(metadata & embedding) and state of the art late interaction re-ranker to
retrieve best matching pages. Finally, MLLM are prompted as reader to generate
answers from contextualized best matching pages. Through experiments, we
observe that the proposed design is scalable (significant speed up) and stable
(without degrading performance quality), hence can be used as production
systems at enterprises.

### 6. [Context-Aware Search and Retrieval Over Erasure Channels](http://arxiv.org/pdf/2507.11894v1)

Authors: Sara Ghasvarianjahromi, Yauhen Yakimenka, Jörg Kliewer

This paper introduces and analyzes a search and retrieval model that adopts
key semantic communication principles from retrieval-augmented generation. We
specifically present an information-theoretic analysis of a remote document
retrieval system operating over a symbol erasure channel. The proposed model
encodes the feature vector of a query, derived from term-frequency weights of a
language corpus by using a repetition code with an adaptive rate dependent on
the contextual importance of the terms. At the decoder, we select between two
documents based on the contextual closeness of the recovered query. By
leveraging a jointly Gaussian approximation for both the true and reconstructed
similarity scores, we derive an explicit expression for the retrieval error
probability, i.e., the probability under which the less similar document is
selected. Numerical simulations on synthetic and real-world data (Google NQ)
confirm the validity of the analysis. They further demonstrate that assigning
greater redundancy to critical features effectively reduces the error rate,
highlighting the effectiveness of semantic-aware feature encoding in
error-prone communication settings.

### 7. [AFPM: Alignment-based Frame Patch Modeling for Cross-Dataset EEG Decoding](http://arxiv.org/pdf/2507.11911v1)

Authors: Xiaoqing Chen, Siyang Li, Dongrui Wu

Electroencephalogram (EEG) decoding models for brain-computer interfaces
(BCIs) struggle with cross-dataset learning and generalization due to channel
layout inconsistencies, non-stationary signal distributions, and limited
neurophysiological prior integration. To address these issues, we propose a
plug-and-play Alignment-Based Frame-Patch Modeling (AFPM) framework, which has
two main components: 1) Spatial Alignment, which selects task-relevant channels
based on brain-region priors, aligns EEG distributions across domains, and
remaps the selected channels to a unified layout; and, 2) Frame-Patch Encoding,
which models multi-dataset signals into unified spatiotemporal patches for EEG
decoding. Compared to 17 state-of-the-art approaches that need dataset-specific
tuning, the proposed calibration-free AFPM achieves performance gains of up to
4.40% on motor imagery and 3.58% on event-related potential tasks. To our
knowledge, this is the first calibration-free cross-dataset EEG decoding
framework, substantially enhancing the practicalness of BCIs in real-world
applications.

### 8. [Sparse Autoencoders for Sequential Recommendation Models: Interpretation and Flexible Control](http://arxiv.org/pdf/2507.12202v1)

Authors: Anton Klenitskiy, Konstantin Polev, Daria Denisova, Alexey Vasilev, Dmitry Simakov, Gleb Gusev

Many current state-of-the-art models for sequential recommendations are based
on transformer architectures. Interpretation and explanation of such black box
models is an important research question, as a better understanding of their
internals can help understand, influence, and control their behavior, which is
very important in a variety of real-world applications. Recently sparse
autoencoders (SAE) have been shown to be a promising unsupervised approach for
extracting interpretable features from language models. These autoencoders
learn to reconstruct hidden states of the transformer's internal layers from
sparse linear combinations of directions in their activation space.
  This paper is focused on the application of SAE to the sequential
recommendation domain. We show that this approach can be successfully applied
to the transformer trained on a sequential recommendation task: learned
directions turn out to be more interpretable and monosemantic than the original
hidden state dimensions. Moreover, we demonstrate that the features learned by
SAE can be used to effectively and flexibly control the model's behavior,
providing end-users with a straightforward method to adjust their
recommendations to different custom scenarios and contexts.

### 9. [Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data](http://arxiv.org/pdf/2507.12425v1)

Authors: Chandana Cheerla

Organizations increasingly rely on proprietary enterprise data, including HR
records, structured reports, and tabular documents, for critical
decision-making. While Large Language Models (LLMs) have strong generative
capabilities, they are limited by static pretraining, short context windows,
and challenges in processing heterogeneous data formats. Conventional
Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but
often struggle with structured and semi-structured data.
  This work proposes an advanced RAG framework that combines hybrid retrieval
strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by
metadata-aware filtering with SpaCy NER and cross-encoder reranking. The
framework applies semantic chunking to maintain textual coherence and retains
tabular data structures to preserve row-column integrity. Quantized indexing
optimizes retrieval efficiency, while human-in-the-loop feedback and
conversation memory improve adaptability.
  Experiments on enterprise datasets show notable improvements: Precision@5
increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74),
and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative
evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness
(4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale.
These results demonstrate the framework's effectiveness in delivering accurate,
comprehensive, and contextually relevant responses for enterprise tasks. Future
work includes extending to multimodal data and integrating agent-based
retrieval. The source code will be released at
https://github.com/CheerlaChandana/Enterprise-Chatbot

### Machine Learning

### 1. [SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling](http://arxiv.org/pdf/2507.11818v1)

Authors: Andrei Rekesh, Miruna Cretu, Dmytro Shevchuk, Vignesh Ram Somnath, Pietro Liò, Robert A. Batey, Mike Tyers, Michał Koziarski, Cheng-Hao Liu

Ensuring synthesizability in generative small molecule design remains a major
challenge. While recent developments in synthesizable molecule generation have
demonstrated promising results, these efforts have been largely confined to 2D
molecular graph representations, limiting the ability to perform geometry-based
conditional generation. In this work, we present SynCoGen (Synthesizable
Co-Generation), a single framework that combines simultaneous masked graph
diffusion and flow matching for synthesizable 3D molecule generation. SynCoGen
samples from the joint distribution of molecular building blocks, chemical
reactions, and atomic coordinates. To train the model, we curated SynSpace, a
dataset containing over 600K synthesis-aware building block graphs and 3.3M
conformers. SynCoGen achieves state-of-the-art performance in unconditional
small molecule graph and conformer generation, and the model delivers
competitive performance in zero-shot molecular linker design for protein ligand
generation in drug discovery. Overall, this multimodal formulation represents a
foundation for future applications enabled by non-autoregressive molecular
generation, including analog expansion, lead optimization, and direct structure
conditioning.

### 2. [HyperEvent:Learning Cohesive Events for Large-scale Dynamic Link Prediction](http://arxiv.org/pdf/2507.11836v1)

Authors: Jian Gao, Jianshe Wu, JingYi Ding

Dynamic link prediction in continuous-time dynamic graphs is a fundamental
task for modeling evolving complex systems. Existing node-centric and
event-centric methods focus on individual interactions or atomic states,
failing to capture the structural cohesion of composite hyper-events, groups of
causally related events. To address this, we propose HyperEvent, a framework
reframing dynamic link prediction as hyper-event recognition. Central to
HyperEvent is the dynamic construction of an association sequence using event
correlation vectors. These vectors quantify pairwise dependencies between the
query event and relevant historical events, thereby characterizing the
structural cohesion of a potential hyper-event. The framework predicts the
occurrence of the query event by evaluating whether it collectively forms a
valid hyper-event with these historical events. Notably, HyperEvent outperforms
state-of-the-art methods on 4 out of 5 datasets in the official leaderboard.
For scalability, we further introduce an efficient parallel training algorithm
that segments large event streams to enable concurrent training. Experiments
validate HyperEvent's superior accuracy and efficiency on large-scale graphs.
Among which HyperEvent achieves a 6.95% improvement in Mean Reciprocal Rank
over state-of-the-art baseline on the large-scale Flight dataset while
utilizing only 10.17% of the training time.

### 3. [OrdShap: Feature Position Importance for Sequential Black-Box Models](http://arxiv.org/pdf/2507.11855v1)

Authors: Davin Hill, Brian L. Hill, Aria Masoomi, Vijay S. Nori, Robert E. Tillman, Jennifer Dy

Sequential deep learning models excel in domains with temporal or sequential
dependencies, but their complexity necessitates post-hoc feature attribution
methods for understanding their predictions. While existing techniques quantify
feature importance, they inherently assume fixed feature ordering - conflating
the effects of (1) feature values and (2) their positions within input
sequences. To address this gap, we introduce OrdShap, a novel attribution
method that disentangles these effects by quantifying how a model's predictions
change in response to permuting feature position. We establish a game-theoretic
connection between OrdShap and Sanchez-Berganti\~nos values, providing a
theoretically grounded approach to position-sensitive attribution. Empirical
results from health, natural language, and synthetic datasets highlight
OrdShap's effectiveness in capturing feature value and feature position
attributions, and provide deeper insight into model behavior.

### 4. [A Policy-Improved Deep Deterministic Policy Gradient Framework for the Discount Order Acceptance Strategy of Ride-hailing Drivers](http://arxiv.org/pdf/2507.11865v1)

Authors: Hanwen Dai, Chang Gao, Fang He, Congyuan Ji, Yanni Yang

The rapid expansion of platform integration has emerged as an effective
solution to mitigate market fragmentation by consolidating multiple
ride-hailing platforms into a single application. To address heterogeneous
passenger preferences, third-party integrators provide Discount Express service
delivered by express drivers at lower trip fares. For the individual platform,
encouraging broader participation of drivers in Discount Express services has
the potential to expand the accessible demand pool and improve matching
efficiency, but often at the cost of reduced profit margins. This study aims to
dynamically manage drivers' acceptance of Discount Express from the perspective
of individual platforms. The lack of historical data under the new business
model necessitates online learning. However, early-stage exploration through
trial and error can be costly in practice, highlighting the need for reliable
early-stage performance in real-world deployment. To address these challenges,
this study formulates the decision regarding the proportion of drivers'
acceptance behavior as a continuous control task. In response to the high
stochasticity, the opaque matching mechanisms employed by third-party
integrator, and the limited availability of historical data, we propose a
policy-improved deep deterministic policy gradient (pi-DDPG) framework. The
proposed framework incorporates a refiner module to boost policy performance
during the early training phase, leverages a convolutional long short-term
memory network to effectively capture complex spatiotemporal patterns, and
adopts a prioritized experience replay mechanism to enhance learning
efficiency. A simulator based on a real-world dataset is developed to validate
the effectiveness of the proposed pi-DDPG. Numerical experiments demonstrate
that pi-DDPG achieves superior learning efficiency and significantly reduces
early-stage training losses.

### 5. [Imbalanced Regression Pipeline Recommendation](http://arxiv.org/pdf/2507.11901v1)

Authors: Juscimara G. Avelino, George D. C. Cavalcanti, Rafael M. O. Cruz

Imbalanced problems are prevalent in various real-world scenarios and are
extensively explored in classification tasks. However, they also present
challenges for regression tasks due to the rarity of certain target values. A
common alternative is to employ balancing algorithms in preprocessing to
address dataset imbalance. However, due to the variety of resampling methods
and learning models, determining the optimal solution requires testing many
combinations. Furthermore, the learning model, dataset, and evaluation metric
affect the best strategies. This work proposes the Meta-learning for Imbalanced
Regression (Meta-IR) framework, which diverges from existing literature by
training meta-classifiers to recommend the best pipeline composed of the
resampling strategy and learning model per task in a zero-shot fashion. The
meta-classifiers are trained using a set of meta-features to learn how to map
the meta-features to the classes indicating the best pipeline. We propose two
formulations: Independent and Chained. Independent trains the meta-classifiers
to separately indicate the best learning algorithm and resampling strategy.
Chained involves a sequential procedure where the output of one meta-classifier
is used as input for another to model intrinsic relationship factors. The
Chained scenario showed superior performance, suggesting a relationship between
the learning algorithm and the resampling strategy per task. Compared with
AutoML frameworks, Meta-IR obtained better results. Moreover, compared with
baselines of six learning algorithms and six resampling algorithms plus no
resampling, totaling 42 (6 X 7) configurations, Meta-IR outperformed all of
them. The code, data, and further information of the experiments can be found
on GitHub: https://github.com/JusciAvelino/Meta-IR.

### 6. [Resampling strategies for imbalanced regression: a survey and empirical analysis](http://arxiv.org/pdf/2507.11902v1)

Authors: Juscimara G. Avelino, George D. C. Cavalcanti, Rafael M. O. Cruz

Imbalanced problems can arise in different real-world situations, and to
address this, certain strategies in the form of resampling or balancing
algorithms are proposed. This issue has largely been studied in the context of
classification, and yet, the same problem features in regression tasks, where
target values are continuous. This work presents an extensive experimental
study comprising various balancing and predictive models, and wich uses metrics
to capture important elements for the user and to evaluate the predictive model
in an imbalanced regression data context. It also proposes a taxonomy for
imbalanced regression approaches based on three crucial criteria: regression
model, learning process, and evaluation metrics. The study offers new insights
into the use of such strategies, highlighting the advantages they bring to each
model's learning process, and indicating directions for further studies. The
code, data and further information related to the experiments performed herein
can be found on GitHub: https://github.com/JusciAvelino/imbalancedRegression.

### 7. [From Generative to Episodic: Sample-Efficient Replicable Reinforcement Learning](http://arxiv.org/pdf/2507.11926v1)

Authors: Max Hopkins, Sihan Liu, Christopher Ye, Yuichi Yoshida

The epidemic failure of replicability across empirical science and machine
learning has recently motivated the formal study of replicable learning
algorithms [Impagliazzo et al. (2022)]. In batch settings where data comes from
a fixed i.i.d. source (e.g., hypothesis testing, supervised learning), the
design of data-efficient replicable algorithms is now more or less understood.
In contrast, there remain significant gaps in our knowledge for control
settings like reinforcement learning where an agent must interact directly with
a shifting environment. Karbasi et. al show that with access to a generative
model of an environment with $S$ states and $A$ actions (the RL 'batch
setting'), replicably learning a near-optimal policy costs only
$\tilde{O}(S^2A^2)$ samples. On the other hand, the best upper bound without a
generative model jumps to $\tilde{O}(S^7 A^7)$ [Eaton et al. (2024)] due to the
substantial difficulty of environment exploration. This gap raises a key
question in the broader theory of replicability: Is replicable exploration
inherently more expensive than batch learning? Is sample-efficient replicable
RL even possible?
  In this work, we (nearly) resolve this problem (for low-horizon tabular
MDPs): exploration is not a significant barrier to replicable learning! Our
main result is a replicable RL algorithm on $\tilde{O}(S^2A)$ samples, bridging
the gap between the generative and episodic settings. We complement this with a
matching $\tilde{\Omega}(S^2A)$ lower bound in the generative setting (under
the common parallel sampling assumption) and an unconditional lower bound in
the episodic setting of $\tilde{\Omega}(S^2)$ showcasing the near-optimality of
our algorithm with respect to the state space $S$.

### 8. [Accelerating RF Power Amplifier Design via Intelligent Sampling and ML-Based Parameter Tuning](http://arxiv.org/pdf/2507.11928v1)

Authors: Abhishek Sriram, Neal Tuffy

This paper presents a machine learning-accelerated optimization framework for
RF power amplifier design that reduces simulation requirements by 65% while
maintaining $\pm0.3$ to $\pm0.4$ dBm accuracy. The proposed method combines
MaxMin Latin Hypercube Sampling with CatBoost gradient boosting to
intelligently explore multidimensional parameter spaces. Instead of
exhaustively simulating all parameter combinations to achieve target P2dB
compression specifications, our approach strategically selects approximately
35% of critical simulation points. The framework processes ADS netlists,
executes harmonic balance simulations on the reduced dataset, and trains a
CatBoost model to predict P2dB performance across the entire design space.
Validation across 15 PA operating modes yields an average $R^2$ of 0.901, with
the system ranking parameter combinations by their likelihood of meeting target
specifications. The integrated solution delivers 58.24% to 77.78% reduction in
simulation time through automated GUI-based workflows, enabling rapid design
iterations without compromising accuracy standards required for production RF
circuits.

### 9. [Detecting In-Person Conversations in Noisy Real-World Environments with Smartwatch Audio and Motion Sensing](http://arxiv.org/pdf/2507.12002v1)

Authors: Alice Zhang, Callihan Bertley, Dawei Liang, Edison Thomaz

Social interactions play a crucial role in shaping human behavior,
relationships, and societies. It encompasses various forms of communication,
such as verbal conversation, non-verbal gestures, facial expressions, and body
language. In this work, we develop a novel computational approach to detect a
foundational aspect of human social interactions, in-person verbal
conversations, by leveraging audio and inertial data captured with a commodity
smartwatch in acoustically-challenging scenarios. To evaluate our approach, we
conducted a lab study with 11 participants and a semi-naturalistic study with
24 participants. We analyzed machine learning and deep learning models with 3
different fusion methods, showing the advantages of fusing audio and inertial
data to consider not only verbal cues but also non-verbal gestures in
conversations. Furthermore, we perform a comprehensive set of evaluations
across activities and sampling rates to demonstrate the benefits of multimodal
sensing in specific contexts. Overall, our framework achieved 82.0$\pm$3.0%
macro F1-score when detecting conversations in the lab and 77.2$\pm$1.8% in the
semi-naturalistic setting.

### 10. [Granular feedback merits sophisticated aggregation](http://arxiv.org/pdf/2507.12041v1)

Authors: Anmol Kagrecha, Henrik Marklund, Potsawee Manakul, Richard Zeckhauser, Benjamin Van Roy

Human feedback is increasingly used across diverse applications like training
AI models, developing recommender systems, and measuring public opinion -- with
granular feedback often being preferred over binary feedback for its greater
informativeness. While it is easy to accurately estimate a population's
distribution of feedback given feedback from a large number of individuals,
cost constraints typically necessitate using smaller groups. A simple method to
approximate the population distribution is regularized averaging: compute the
empirical distribution and regularize it toward a prior. Can we do better? As
we will discuss, the answer to this question depends on feedback granularity.
  Suppose one wants to predict a population's distribution of feedback using
feedback from a limited number of individuals. We show that, as feedback
granularity increases, one can substantially improve upon predictions of
regularized averaging by combining individuals' feedback in ways more
sophisticated than regularized averaging.
  Our empirical analysis using questions on social attitudes confirms this
pattern. In particular, with binary feedback, sophistication barely reduces the
number of individuals required to attain a fixed level of performance. By
contrast, with five-point feedback, sophisticated methods match the performance
of regularized averaging with about half as many individuals.

### Neural and Evolutionary Computing

### 1. [BuildEvo: Designing Building Energy Consumption Forecasting Heuristics via LLM-driven Evolution](http://arxiv.org/pdf/2507.12207v1)

Authors: Subin Lin, Chuanbo Hua

Accurate building energy forecasting is essential, yet traditional heuristics
often lack precision, while advanced models can be opaque and struggle with
generalization by neglecting physical principles. This paper introduces
BuildEvo, a novel framework that uses Large Language Models (LLMs) to
automatically design effective and interpretable energy prediction heuristics.
Within an evolutionary process, BuildEvo guides LLMs to construct and enhance
heuristics by systematically incorporating physical insights from building
characteristics and operational data (e.g., from the Building Data Genome
Project 2). Evaluations show BuildEvo achieves state-of-the-art performance on
benchmarks, offering improved generalization and transparent prediction logic.
This work advances the automated design of robust, physically grounded
heuristics, promoting trustworthy models for complex energy systems.

### 2. [MaCE: General Mass Conserving Dynamics for Cellular Automata](http://arxiv.org/pdf/2507.12306v1)

Authors: Vassilis Papadopoulos, Etienne Guichard

We present Mass-Conserving Evolution (MaCE), a general method for
implementing mass conservation in Cellular Automata (CA). MaCE is a simple
evolution rule that can be easily 'attached' to existing CAs to make them
mass-conserving, which tends to produce interesting behaviours more often, as
patterns can no longer explode or die out. We first show that MaCE is
numerically stable and admits a simple continuous limit. We then test MaCE on
Lenia, and through several experiments, we demonstrate that it produces a wide
variety of interesting behaviours, starting from the variety and abundance of
solitons up to hints of intrinsic evolution in resource-constrained
environments. Finally, we showcase the versatility of MaCE by applying it to
Neural-CAs and discrete CAs, and discuss promising research directions opened
up by this scheme.

### Networking and Internet Architecture

### 1. [Extremal Testing for Network Software using LLMs](http://arxiv.org/pdf/2507.11898v1)

Authors: Rathin Singha, Harry Qian, Srinath Saikrishnan, Tracy Zhao, Ryan Beckett, Siva Kesava Reddy Kakarla, George Varghese

Physicists often manually consider extreme cases when testing a theory. In
this paper, we show how to automate extremal testing of network software using
LLMs in two steps: first, ask the LLM to generate input constraints (e.g., DNS
name length limits); then ask the LLM to generate tests that violate the
constraints. We demonstrate how easy this process is by generating extremal
tests for HTTP, BGP and DNS implementations, each of which uncovered new bugs.
We show how this methodology extends to centralized network software such as
shortest path algorithms, and how LLMs can generate filtering code to reject
extremal input. We propose using agentic AI to further automate extremal
testing. LLM-generated extremal testing goes beyond an old technique in
software testing called Boundary Value Analysis.

### 2. [FastReChain: Highly Responsive and Low-Overhead Centralized Route Scheduling in Clos Datacenter Networks](http://arxiv.org/pdf/2507.12265v1)

Authors: Zihan Zhu, Dongchao Wu, Zhanbang Zhang, Jian Yang

Ever since Clos topologies were used in datacenter networks (DCNs), a
practical centralized scheduling algorithm that supports dynamic scheduling has
been absent. The introduction of optical switches in DCNs as a future-proof
solution exacerbates this problem due to several properties of optical
switches, such as the fact that they are generally bufferless and therefore
rely on centralized scheduling, and that they have long switching times and
therefore require the number of rearrangements to be minimized.
  In this paper, we propose a centralized scheduling algorithm that achieves
theoretical maximum throughput even in one-rate bidirectional Clos networks,
while producing schemes with near-minimal numbers of rearrangements. It is the
only algorithm that directly supports bidirectional Clos networks and has a
time efficiency high enough to support dynamic scheduling to date. For static
minimal rewiring, its running time ranges from a fraction to a few hundredths
of other algorithms, and the number of rearrangements has also been steadily
improved, allowing for more frequent adjustments and less impact on ongoing
communications. In addition, the algorithm is very flexible and can support
various functional requirements in real-world environments. We achieve this
result through the replacement chain concept and bitset optimization.

### 3. [Native-AI Empowered Scalable Architectures and Solutions for Future Non-Terrestrial Networks: An Overview](http://arxiv.org/pdf/2507.11935v1)

Authors: Jikang Deng, Fizza Hassan, Hui Zhou, Saad Al-Ahmadi, Mohamed-Slim Alouini, Daniel B. Da Costa

As the path toward 6G networks is being charted, the emerging applications
have motivated evolutions of network architectures to realize the efficient,
reliable, and flexible wireless networks. Among the potential architectures,
the non-terrestrial network (NTN) and open radio access network (ORAN) have
received increasing interest from both academia and industry. Although the
deployment of NTNs ensures coverage, enhances spectral efficiency, and improves
the resilience of wireless networks. The high altitude and mobility of NTN
present new challenges in the development and operations (DevOps) lifecycle,
hindering intelligent and scalable network management due to the lack of native
artificial intelligence (AI) capability. With the advantages of ORAN in
disaggregation, openness, virtualization, and intelligence, several works
propose integrating ORAN principles into the NTN, focusing mainly on ORAN
deployment options based on transparent and regenerative systems. However, a
holistic view of how to effectively combine ORAN and NTN throughout the DevOps
lifecycle is still missing, especially regarding how intelligent ORAN addresses
the scalability challenges in NTN. Motivated by this, in this paper, we first
provide the background knowledge about ORAN and NTN, outline the
state-of-the-art research on ORAN for NTNs, and present the DevOps challenges
that motivate the adoption of ORAN solutions. We then propose the ORAN-based
NTN framework, discussing its features and architectures in detail. These
include the discussion about flexible fronthaul split, RAN intelligent
controllers (RICs) enhancement for distributed learning, scalable deployment
architecture, and multi-domain service management. Finally, the future research
directions, including combinations of the ORAN-based NTN framework and other
enabling technologies and schemes, as well as the candidate use cases, are
highlighted.

### 4. [MOFCO: Mobility- and Migration-Aware Task Offloading in Three-Layer Fog Computing Environments](http://arxiv.org/pdf/2507.12028v1)

Authors: Soheil Mahdizadeh, Elyas Oustad, Mohsen Ansari

Task offloading in three-layer fog computing environments presents a critical
challenge due to user equipment (UE) mobility, which frequently triggers costly
service migrations and degrades overall system performance. This paper
addresses this problem by proposing MOFCO, a novel Mobility- and
Migration-aware Task Offloading algorithm for Fog Computing environments. The
proposed method formulates task offloading and resource allocation as a
Mixed-Integer Nonlinear Programming (MINLP) problem and employs a
heuristic-aided evolutionary game theory approach to solve it efficiently. To
evaluate MOFCO, we simulate mobile users using SUMO, providing realistic
mobility patterns. Experimental results show that MOFCO reduces system cost,
defined as a combination of latency and energy consumption, by an average of
19% and up to 43% in certain scenarios compared to state-of-the-art methods.

### 5. [LLM-Based Config Synthesis requires Disambiguation](http://arxiv.org/pdf/2507.12443v1)

Authors: Rajdeep Mondal, Nikolaj Bjorner, Todd Millstein, Alan Tang, George Varghese

Beyond hallucinations, another problem in program synthesis using LLMs is
ambiguity in user intent. We illustrate the ambiguity problem in a networking
context for LLM-based incremental configuration synthesis of route-maps and
ACLs. These structures frequently overlap in header space, making the relative
priority of actions impossible for the LLM to infer without user interaction.
Measurements in a large cloud identify complex ACLs with 100's of overlaps,
showing ambiguity is a real problem. We propose a prototype system, Clarify,
which uses an LLM augmented with a new module called a Disambiguator that helps
elicit user intent. On a small synthetic workload, Clarify incrementally
synthesizes routing policies after disambiguation and then verifies them. Our
treatment of ambiguities is useful more generally when the intent of updates
can be correctly synthesized by LLMs, but their integration is ambiguous and
can lead to different global behaviors.

### 6. [CRAFT: Latency and Cost-Aware Genetic-Based Framework for Node Placement in Edge-Fog Environments](http://arxiv.org/pdf/2507.12445v1)

Authors: Soheil Mahdizadeh, Amir Mahdi Rasouli, Mohammad Pourashory, Sadra Galavani, Mohsen Ansari

Reducing latency in the Internet of Things (IoT) is a critical concern. While
cloud computing facilitates communication, it falls short of meeting real-time
requirements reliably. Edge and fog computing have emerged as viable solutions
by positioning computing nodes closer to end users, offering lower latency and
increased processing power. An edge-fog framework comprises various components,
including edge and fog nodes, whose strategic placement is crucial as it
directly impacts latency and system cost. This paper presents an effective and
tunable node placement strategy based on a genetic algorithm to address the
optimization problem of deploying edge and fog nodes. The main objective is to
minimize latency and cost through optimal node placement. Simulation results
demonstrate that the proposed framework achieves up to 2.77% latency and 31.15%
cost reduction.

### Robotics

### 1. [The Developments and Challenges towards Dexterous and Embodied Robotic Manipulation: A Survey](http://arxiv.org/pdf/2507.11840v1)

Authors: Gaofeng Li, Ruize Wang, Peisen Xu, Qi Ye, Jiming Chen

Achieving human-like dexterous robotic manipulation remains a central goal
and a pivotal challenge in robotics. The development of Artificial Intelligence
(AI) has allowed rapid progress in robotic manipulation. This survey summarizes
the evolution of robotic manipulation from mechanical programming to embodied
intelligence, alongside the transition from simple grippers to multi-fingered
dexterous hands, outlining key characteristics and main challenges. Focusing on
the current stage of embodied dexterous manipulation, we highlight recent
advances in two critical areas: dexterous manipulation data collection (via
simulation, human demonstrations, and teleoperation) and skill-learning
frameworks (imitation and reinforcement learning). Then, based on the overview
of the existing data collection paradigm and learning framework, three key
challenges restricting the development of dexterous robotic manipulation are
summarized and discussed.

### 2. [A Fast Method for Planning All Optimal Homotopic Configurations for Tethered Robots and Its Extended Applications](http://arxiv.org/pdf/2507.11880v1)

Authors: Jinyuan Liu, Minglei Fu, Ling Shi, Chenguang Yang, Wenan Zhang

Tethered robots play a pivotal role in specialized environments such as
disaster response and underground exploration, where their stable power supply
and reliable communication offer unparalleled advantages. However, their motion
planning is severely constrained by tether length limitations and entanglement
risks, posing significant challenges to achieving optimal path planning. To
address these challenges, this study introduces CDT-TCS (Convex Dissection
Topology-based Tethered Configuration Search), a novel algorithm that leverages
CDT Encoding as a homotopy invariant to represent topological states of paths.
By integrating algebraic topology with geometric optimization, CDT-TCS
efficiently computes the complete set of optimal feasible configurations for
tethered robots at all positions in 2D environments through a single
computation. Building on this foundation, we further propose three
application-specific algorithms: i) CDT-TPP for optimal tethered path planning,
ii) CDT-TMV for multi-goal visiting with tether constraints, iii) CDT-UTPP for
distance-optimal path planning of untethered robots. All theoretical results
and propositions underlying these algorithms are rigorously proven and
thoroughly discussed in this paper. Extensive simulations demonstrate that the
proposed algorithms significantly outperform state-of-the-art methods in their
respective problem domains. Furthermore, real-world experiments on robotic
platforms validate the practicality and engineering value of the proposed
framework.

### 3. [NemeSys: An Online Underwater Explorer with Goal-Driven Adaptive Autonomy](http://arxiv.org/pdf/2507.11889v1)

Authors: Adnan Abdullah, Alankrit Gupta, Vaishnav Ramesh, Shivali Patel, Md Jahidul Islam

Adaptive mission control and dynamic parameter reconfiguration are essential
for autonomous underwater vehicles (AUVs) operating in GPS-denied,
communication-limited marine environments. However, most current AUV platforms
execute static, pre-programmed missions or rely on tethered connections and
high-latency acoustic channels for mid-mission updates, significantly limiting
their adaptability and responsiveness. In this paper, we introduce NemeSys, a
novel AUV system designed to support real-time mission reconfiguration through
compact optical and magnetoelectric (OME) signaling facilitated by floating
buoys. We present the full system design, control architecture, and a semantic
mission encoding framework that enables interactive exploration and task
adaptation via low-bandwidth communication. The proposed system is validated
through analytical modeling, controlled experimental evaluations, and
open-water trials. Results confirm the feasibility of online mission adaptation
and semantic task updates, highlighting NemeSys as an online AUV platform for
goal-driven adaptive autonomy in dynamic and uncertain underwater environments.

### 4. [A Review of Generative AI in Aquaculture: Foundations, Applications, and Future Directions for Smart and Sustainable Farming](http://arxiv.org/pdf/2507.11974v1)

Authors: Waseem Akram, Muhayy Ud Din, Lyes Saad Soud, Irfan Hussain

Generative Artificial Intelligence (GAI) has rapidly emerged as a
transformative force in aquaculture, enabling intelligent synthesis of
multimodal data, including text, images, audio, and simulation outputs for
smarter, more adaptive decision-making. As the aquaculture industry shifts
toward data-driven, automation and digital integration operations under the
Aquaculture 4.0 paradigm, GAI models offer novel opportunities across
environmental monitoring, robotics, disease diagnostics, infrastructure
planning, reporting, and market analysis. This review presents the first
comprehensive synthesis of GAI applications in aquaculture, encompassing
foundational architectures (e.g., diffusion models, transformers, and retrieval
augmented generation), experimental systems, pilot deployments, and real-world
use cases. We highlight GAI's growing role in enabling underwater perception,
digital twin modeling, and autonomous planning for remotely operated vehicle
(ROV) missions. We also provide an updated application taxonomy that spans
sensing, control, optimization, communication, and regulatory compliance.
Beyond technical capabilities, we analyze key limitations, including limited
data availability, real-time performance constraints, trust and explainability,
environmental costs, and regulatory uncertainty. This review positions GAI not
merely as a tool but as a critical enabler of smart, resilient, and
environmentally aligned aquaculture systems.

### 5. [Robust Route Planning for Sidewalk Delivery Robots](http://arxiv.org/pdf/2507.12067v1)

Authors: Xing Tong, Michele D. Simoni

Sidewalk delivery robots are a promising solution for urban freight
distribution, reducing congestion compared to trucks and providing a safer,
higher-capacity alternative to drones. However, unreliable travel times on
sidewalks due to pedestrian density, obstacles, and varying infrastructure
conditions can significantly affect their efficiency. This study addresses the
robust route planning problem for sidewalk robots, explicitly accounting for
travel time uncertainty due to varying sidewalk conditions. Optimization is
integrated with simulation to reproduce the effect of obstacles and pedestrian
flows and generate realistic travel times. The study investigates three
different approaches to derive uncertainty sets, including budgeted,
ellipsoidal, and support vector clustering (SVC)-based methods, along with a
distributionally robust method to solve the shortest path (SP) problem. A
realistic case study reproducing pedestrian patterns in Stockholm's city center
is used to evaluate the efficiency of robust routing across various robot
designs and environmental conditions. The results show that, when compared to a
conventional SP, robust routing significantly enhances operational reliability
under variable sidewalk conditions. The Ellipsoidal and DRSP approaches
outperform the other methods, yielding the most efficient paths in terms of
average and worst-case delay. Sensitivity analyses reveal that robust
approaches consistently outperform the conventional SP, particularly for
sidewalk delivery robots that are wider, slower, and have more conservative
navigation behaviors. These benefits are even more pronounced in adverse
weather conditions and high pedestrian congestion scenarios.

### 6. [Tree-SLAM: semantic object SLAM for efficient mapping of individual trees in orchards](http://arxiv.org/pdf/2507.12093v1)

Authors: David Rapado-Rincon, Gert Kootstra

Accurate mapping of individual trees is an important component for precision
agriculture in orchards, as it allows autonomous robots to perform tasks like
targeted operations or individual tree monitoring. However, creating these maps
is challenging because GPS signals are often unreliable under dense tree
canopies. Furthermore, standard Simultaneous Localization and Mapping (SLAM)
approaches struggle in orchards because the repetitive appearance of trees can
confuse the system, leading to mapping errors. To address this, we introduce
Tree-SLAM, a semantic SLAM approach tailored for creating maps of individual
trees in orchards. Utilizing RGB-D images, our method detects tree trunks with
an instance segmentation model, estimates their location and re-identifies them
using a cascade-graph-based data association algorithm. These re-identified
trunks serve as landmarks in a factor graph framework that integrates noisy GPS
signals, odometry, and trunk observations. The system produces maps of
individual trees with a geo-localization error as low as 18 cm, which is less
than 20\% of the planting distance. The proposed method was validated on
diverse datasets from apple and pear orchards across different seasons,
demonstrating high mapping accuracy and robustness in scenarios with unreliable
GPS signals.

### 7. [Leveraging Sidewalk Robots for Walkability-Related Analyses](http://arxiv.org/pdf/2507.12148v1)

Authors: Xing Tong, Michele D. Simoni, Kaj Munhoz Arfvidsson, Jonas Mårtensson

Walkability is a key component of sustainable urban development, while
collecting detailed data on its related features remains challenging due to the
high costs and limited scalability of traditional methods. Sidewalk delivery
robots, increasingly deployed in urban environments, offer a promising solution
to these limitations. This paper explores how these robots can serve as mobile
data collection platforms, capturing sidewalk-level features related to
walkability in a scalable, automated, and real-time manner. A sensor-equipped
robot was deployed on a sidewalk network at KTH in Stockholm, completing 101
trips covering 900 segments. From the collected data, different typologies of
features are derived, including robot trip characteristics (e.g., speed,
duration), sidewalk conditions (e.g., width, surface unevenness), and sidewalk
utilization (e.g., pedestrian density). Their walkability-related implications
were investigated with a series of analyses. The results demonstrate that
pedestrian movement patterns are strongly influenced by sidewalk
characteristics, with higher density, reduced width, and surface irregularity
associated with slower and more variable trajectories. Notably, robot speed
closely mirrors pedestrian behavior, highlighting its potential as a proxy for
assessing pedestrian dynamics. The proposed framework enables continuous
monitoring of sidewalk conditions and pedestrian behavior, contributing to the
development of more walkable, inclusive, and responsive urban environments.

### 8. [Probabilistic Safety Verification for an Autonomous Ground Vehicle: A Situation Coverage Grid Approach](http://arxiv.org/pdf/2507.12158v1)

Authors: Nawshin Mannan Proma, Gricel Vázquez, Sepeedeh Shahbeigi, Arjun Badyal, Victoria Hodge

As industrial autonomous ground vehicles are increasingly deployed in
safety-critical environments, ensuring their safe operation under diverse
conditions is paramount. This paper presents a novel approach for their safety
verification based on systematic situation extraction, probabilistic modelling
and verification. We build upon the concept of a situation coverage grid, which
exhaustively enumerates environmental configurations relevant to the vehicle's
operation. This grid is augmented with quantitative probabilistic data
collected from situation-based system testing, capturing probabilistic
transitions between situations. We then generate a probabilistic model that
encodes the dynamics of both normal and unsafe system behaviour. Safety
properties extracted from hazard analysis and formalised in temporal logic are
verified through probabilistic model checking against this model. The results
demonstrate that our approach effectively identifies high-risk situations,
provides quantitative safety guarantees, and supports compliance with
regulatory standards, thereby contributing to the robust deployment of
autonomous systems.

### 9. [UniLGL: Learning Uniform Place Recognition for FOV-limited/Panoramic LiDAR Global Localization](http://arxiv.org/pdf/2507.12194v1)

Authors: Hongming Shen, Xun Chen, Yulin Hui, Zhenyu Wu, Wei Wang, Qiyang Lyu, Tianchen Deng, Danwei Wang

Existing LGL methods typically consider only partial information (e.g.,
geometric features) from LiDAR observations or are designed for homogeneous
LiDAR sensors, overlooking the uniformity in LGL. In this work, a uniform LGL
method is proposed, termed UniLGL, which simultaneously achieves spatial and
material uniformity, as well as sensor-type uniformity. The key idea of the
proposed method is to encode the complete point cloud, which contains both
geometric and material information, into a pair of BEV images (i.e., a spatial
BEV image and an intensity BEV image). An end-to-end multi-BEV fusion network
is designed to extract uniform features, equipping UniLGL with spatial and
material uniformity. To ensure robust LGL across heterogeneous LiDAR sensors, a
viewpoint invariance hypothesis is introduced, which replaces the conventional
translation equivariance assumption commonly used in existing LPR networks and
supervises UniLGL to achieve sensor-type uniformity in both global descriptors
and local feature representations. Finally, based on the mapping between local
features on the 2D BEV image and the point cloud, a robust global pose
estimator is derived that determines the global minimum of the global pose on
SE(3) without requiring additional registration. To validate the effectiveness
of the proposed uniform LGL, extensive benchmarks are conducted in real-world
environments, and the results show that the proposed UniLGL is demonstratively
competitive compared to other State-of-the-Art LGL methods. Furthermore, UniLGL
has been deployed on diverse platforms, including full-size trucks and agile
Micro Aerial Vehicles (MAVs), to enable high-precision localization and mapping
as well as multi-MAV collaborative exploration in port and forest environments,
demonstrating the applicability of UniLGL in industrial and field scenarios.

### 10. [Next-Gen Museum Guides: Autonomous Navigation and Visitor Interaction with an Agentic Robot](http://arxiv.org/pdf/2507.12273v1)

Authors: Luca Garello, Francesca Cocchella, Alessandra Sciutti, Manuel Catalano, Francesco Rea

Autonomous robots are increasingly being tested into public spaces to enhance
user experiences, particularly in cultural and educational settings. This paper
presents the design, implementation, and evaluation of the autonomous museum
guide robot Alter-Ego equipped with advanced navigation and interactive
capabilities. The robot leverages state-of-the-art Large Language Models (LLMs)
to provide real-time, context aware question-and-answer (Q&A) interactions,
allowing visitors to engage in conversations about exhibits. It also employs
robust simultaneous localization and mapping (SLAM) techniques, enabling
seamless navigation through museum spaces and route adaptation based on user
requests. The system was tested in a real museum environment with 34
participants, combining qualitative analysis of visitor-robot conversations and
quantitative analysis of pre and post interaction surveys. Results showed that
the robot was generally well-received and contributed to an engaging museum
experience, despite some limitations in comprehension and responsiveness. This
study sheds light on HRI in cultural spaces, highlighting not only the
potential of AI-driven robotics to support accessibility and knowledge
acquisition, but also the current limitations and challenges of deploying such
technologies in complex, real-world environments.

### Software Engineering

### 1. [A Task Taxonomy for Conformance Checking](http://arxiv.org/pdf/2507.11976v1)

Authors: Jana-Rebecca Rehse, Michael Grohs, Finn Klessascheck, Lisa-Marie Klein, Tatiana von Landesberger, Luise Pufahl

Conformance checking is a sub-discipline of process mining, which compares
observed process traces with a process model to analyze whether the process
execution conforms with or deviates from the process design. Organizations can
leverage this analysis, for example to check whether their processes comply
with internal or external regulations or to identify potential improvements.
Gaining these insights requires suitable visualizations, which make complex
results accessible and actionable. So far, however, the development of
conformance checking visualizations has largely been left to tool vendors. As a
result, current tools offer a wide variety of visual representations for
conformance checking, but the analytical purposes they serve often remain
unclear. However, without a systematic understanding of these purposes, it is
difficult to evaluate the visualizations' usefulness. Such an evaluation hence
requires a deeper understanding of conformance checking as an analysis domain.
To this end, we propose a task taxonomy, which categorizes the tasks that can
occur when conducting conformance checking analyses. This taxonomy supports
researchers in determining the purpose of visualizations, specifying relevant
conformance checking tasks in terms of their goal, means, constraint type, data
characteristics, data target, and data cardinality. Combining concepts from
process mining and visual analytics, we address researchers from both
disciplines to enable and support closer collaborations.

### 2. [SWE-Perf: Can Language Models Optimize Code Performance on Real-World Repositories?](http://arxiv.org/pdf/2507.12415v1)

Authors: Xinyi He, Qian Liu, Mingzhe Du, Lin Yan, Zhijie Fan, Yiming Huang, Zejian Yuan, Zejun Ma

Code performance optimization is paramount in real-world software engineering
and critical for production-level systems. While Large Language Models (LLMs)
have demonstrated impressive capabilities in code generation and bug fixing,
their proficiency in enhancing code performance at the repository level remains
largely unexplored. To address this gap, we introduce SWE-Perf, the first
benchmark specifically designed to systematically evaluate LLMs on code
performance optimization tasks within authentic repository contexts. SWE-Perf
comprises 140 carefully curated instances, each derived from
performance-improving pull requests from popular GitHub repositories. Each
benchmark instance includes the relevant codebase, target functions,
performance-related tests, expert-authored patches, and executable
environments. Through a comprehensive evaluation of representative methods that
span file-level and repo-level approaches (e.g., Agentless and OpenHands), we
reveal a substantial capability gap between existing LLMs and expert-level
optimization performance, highlighting critical research opportunities in this
emerging field.

### 3. [Extremal Testing for Network Software using LLMs](http://arxiv.org/pdf/2507.11898v1)

Authors: Rathin Singha, Harry Qian, Srinath Saikrishnan, Tracy Zhao, Ryan Beckett, Siva Kesava Reddy Kakarla, George Varghese

Physicists often manually consider extreme cases when testing a theory. In
this paper, we show how to automate extremal testing of network software using
LLMs in two steps: first, ask the LLM to generate input constraints (e.g., DNS
name length limits); then ask the LLM to generate tests that violate the
constraints. We demonstrate how easy this process is by generating extremal
tests for HTTP, BGP and DNS implementations, each of which uncovered new bugs.
We show how this methodology extends to centralized network software such as
shortest path algorithms, and how LLMs can generate filtering code to reject
extremal input. We propose using agentic AI to further automate extremal
testing. LLM-generated extremal testing goes beyond an old technique in
software testing called Boundary Value Analysis.

### 4. [LLAMA: Multi-Feedback Smart Contract Fuzzing Framework with LLM-Guided Seed Generation](http://arxiv.org/pdf/2507.12084v1)

Authors: Keke Gai, Haochen Liang, Jing Yu, Liehuang Zhu, Dusit Niyato

Smart contracts play a pivotal role in blockchain ecosystems, and fuzzing
remains an important approach to securing smart contracts. Even though mutation
scheduling is a key factor influencing fuzzing effectiveness, existing fuzzers
have primarily explored seed scheduling and generation, while mutation
scheduling has been rarely addressed by prior work. In this work, we propose a
Large Language Models (LLMs)-based Multi-feedback Smart Contract Fuzzing
framework (LLAMA) that integrates LLMs, evolutionary mutation strategies, and
hybrid testing techniques. Key components of the proposed LLAMA include: (i) a
hierarchical prompting strategy that guides LLMs to generate semantically valid
initial seeds, coupled with a lightweight pre-fuzzing phase to select
high-potential inputs; (ii) a multi-feedback optimization mechanism that
simultaneously improves seed generation, seed selection, and mutation
scheduling by leveraging runtime coverage and dependency feedback; and (iii) an
evolutionary fuzzing engine that dynamically adjusts mutation operator
probabilities based on effectiveness, while incorporating symbolic execution to
escape stagnation and uncover deeper vulnerabilities. Our experiments
demonstrate that LLAMA outperforms state-of-the-art fuzzers in both coverage
and vulnerability detection. Specifically, it achieves 91% instruction coverage
and 90% branch coverage, while detecting 132 out of 148 known vulnerabilities
across diverse categories. These results highlight LLAMA's effectiveness,
adaptability, and practicality in real-world smart contract security testing
scenarios.

### 5. [From Static to Intelligent: Evolving SaaS Pricing with LLMs](http://arxiv.org/pdf/2507.12104v1)

Authors: Francisco Javier Cavero, Juan C. Alonso, Antonio Ruiz-Cortés

The SaaS paradigm has revolutionized software distribution by offering
flexible pricing options to meet diverse customer needs. However, the rapid
expansion of the SaaS market has introduced significant complexity for DevOps
teams, who must manually manage and evolve pricing structures, an approach that
is both time-consuming and prone to errors. The absence of automated tools for
pricing analysis restricts the ability to efficiently evaluate, optimize, and
scale these models. This paper proposes leveraging intelligent pricing
(iPricing), dynamic, machine-readable pricing models, as a solution to these
challenges. Intelligent pricing enables competitive analysis, streamlines
operational decision-making, and supports continuous pricing evolution in
response to market dynamics, leading to improved efficiency and accuracy. We
present an LLM-driven approach that automates the transformation of static HTML
pricing into iPricing, significantly improving efficiency and consistency while
minimizing human error. Our implementation, AI4Pricing2Yaml, features a basic
Information Extractor that uses web scraping and LLMs technologies to extract
essential pricing components, plans, features, usage limits, and add-ons, from
SaaS websites. Validation against a dataset of 30 distinct commercial SaaS,
encompassing over 150 intelligent pricings, demonstrates the system's
effectiveness in extracting the desired elements across all steps. However,
challenges remain in addressing hallucinations, complex structures, and dynamic
content. This work highlights the potential of automating intelligent pricing
transformation to streamline SaaS pricing management, offering implications for
improved consistency and scalability in an increasingly intricate pricing
landscape. Future research will focus on refining extraction capabilities and
enhancing the system's adaptability to a wider range of SaaS websites.

### 6. [An Online A/B Testing Decision Support System for Web Usability Assessment Based on a Linguistic Decision-making Methodology: Case of Study a Virtual Learning Environment](http://arxiv.org/pdf/2507.12118v1)

Authors: Noe Zermeño, Cristina Zuheros, Lucas Daniel Del Rosso Calache, Francisco Herrera, Rosana Montes

In recent years, attention has increasingly focused on enhancing user
satisfaction with user interfaces, spanning both mobile applications and
websites. One fundamental aspect of human-machine interaction is the concept of
web usability. In order to assess web usability, the A/B testing technique
enables the comparison of data between two designs. Expanding the scope of
tests to include the designs being evaluated, in conjunction with the
involvement of both real and fictional users, presents a challenge for which
few online tools offer support. We propose a methodology for web usability
evaluation based on user-centered approaches such as design thinking and
linguistic decision-making, named Linguistic Decision-Making for Web Usability
Evaluation. This engages people in role-playing scenarios and conducts a number
of usability tests, including the widely recognized System Usability Scale. We
incorporate the methodology into a decision support system based on A/B
testing. We use real users in a case study to assess three Moodle platforms at
the University of Guadalajara, Mexico.

### 7. [Kevin: Multi-Turn RL for Generating CUDA Kernels](http://arxiv.org/pdf/2507.11948v1)

Authors: Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, Silas Alberti

Writing GPU kernels is a challenging task and critical for AI systems'
efficiency. It is also highly iterative: domain experts write code and improve
performance through execution feedback. Moreover, it presents verifiable
rewards like correctness and speedup, making it a natural environment to apply
Reinforcement Learning (RL). To explicitly incorporate the iterative nature of
this process into training, we develop a flexible multi-turn RL recipe that
addresses unique challenges encountered in real-world settings, such as
learning from long trajectories and effective reward attribution across turns.
We present Kevin - K(ernel D)evin, the first model trained with multi-turn RL
for CUDA kernel generation and optimization. In our evaluation setup, Kevin
shows significant gains over its base model (QwQ-32B), improving correctness of
generated kernels (in pure CUDA) from 56% to 82% and mean speedup from 0.53x to
1.10x of baseline (PyTorch Eager), and surpassing frontier models like o4-mini
(0.78x). Finally, we study its behavior across test-time scaling axes: we found
scaling serial refinement more beneficial than parallel sampling. In
particular, when given more refinement turns, Kevin shows a higher rate of
improvement.

### 8. [Expanding ML-Documentation Standards For Better Security](http://arxiv.org/pdf/2507.12003v1)

Authors: Cara Ellen Appel

This article presents the current state of ML-security and of the
documentation of ML-based systems, models and datasets in research and practice
based on an extensive review of the existing literature. It shows a generally
low awareness of security aspects among ML-practitioners and organizations and
an often unstandardized approach to documentation, leading to overall low
quality of ML-documentation. Existing standards are not regularly adopted in
practice and IT-security aspects are often not included in documentation. Due
to these factors, there is a clear need for improved security documentation in
ML, as one step towards addressing the existing gaps in ML-security. To achieve
this, we propose expanding existing documentation standards for
ML-documentation to include a security section with specific security relevant
information. Implementing this, a novel expanded method of documenting security
requirements in ML-documentation is presented, based on the existing Model
Cards and Datasheets for Datasets standards, but with the recommendation to
adopt these findings in all ML-documentation.

### 9. [MERA Code: A Unified Framework for Evaluating Code Generation Across Tasks](http://arxiv.org/pdf/2507.12284v1)

Authors: Artem Chervyakov, Alexander Kharitonov, Pavel Zadorozhny, Adamenko Pavel, Rodion Levichev, Dmitrii Vorobev, Dmitrii Salikhov, Aidar Valeev, Alena Pestova, Maria Dziuba, Ilseyar Alimova, Artem Zavgorodnev, Aleksandr Medvedev, Stanislav Moiseev, Elena Bruches, Daniil Grebenkin, Roman Derunets, Vikulov Vladimir, Anton Emelyanov, Dmitrii Babaev, Vladimir V. Ivanov, Valentin Malykh, Alena Fenogenova

Advancements in LLMs have enhanced task automation in software engineering;
however, current evaluations primarily focus on natural language tasks,
overlooking code quality. Most benchmarks prioritize high-level reasoning over
executable code and real-world performance, leaving gaps in understanding true
capabilities and risks associated with these models in production. To address
this issue, we propose MERA Code, a new addition to the MERA benchmark family,
specifically focused on evaluating code for the latest code generation LLMs in
Russian. This benchmark includes 11 evaluation tasks that span 8 programming
languages. Our proposed evaluation methodology features a taxonomy that
outlines the practical coding skills necessary for models to complete these
tasks. The benchmark comprises an open-source codebase for users to conduct
MERA assessments, a scoring system compatible with various programming
environments, and a platform featuring a leaderboard and submission system. We
evaluate open LLMs and frontier API models, analyzing their limitations in
terms of practical coding tasks in non-English languages. We are publicly
releasing MERA to guide future research, anticipate groundbreaking features in
model development, and standardize evaluation procedures.

### 10. [GitChameleon: Evaluating AI Code Generation Against Python Library Version Incompatibilities](http://arxiv.org/pdf/2507.12367v1)

Authors: Diganta Misra, Nizar Islah, Victor May, Brice Rauby, Zihan Wang, Justine Gehring, Antonio Orvieto, Muawiz Chaudhary, Eilif B. Muller, Irina Rish, Samira Ebrahimi Kahou, Massimo Caccia

The rapid evolution of software libraries poses a considerable hurdle for
code generation, necessitating continuous adaptation to frequent version
updates while preserving backward compatibility. While existing code evolution
benchmarks provide valuable insights, they typically lack execution-based
evaluation for generating code compliant with specific library versions. To
address this, we introduce GitChameleon, a novel, meticulously curated dataset
comprising 328 Python code completion problems, each conditioned on specific
library versions and accompanied by executable unit tests. GitChameleon
rigorously evaluates the capacity of contemporary large language models (LLMs),
LLM-powered agents, code assistants, and RAG systems to perform
version-conditioned code generation that demonstrates functional accuracy
through execution. Our extensive evaluations indicate that state-of-the-art
systems encounter significant challenges with this task; enterprise models
achieving baseline success rates in the 48-51\% range, underscoring the
intricacy of the problem. By offering an execution-based benchmark emphasizing
the dynamic nature of code libraries, GitChameleon enables a clearer
understanding of this challenge and helps guide the development of more
adaptable and dependable AI code generation methods. We make the dataset and
evaluation code publicly available at
https://github.com/mrcabbage972/GitChameleonBenchmark.

### Social and Information Networks

### 1. [Contrastive Cascade Graph Learning for Classifying Real and Synthetic Information Diffusion Patterns](http://arxiv.org/pdf/2507.12063v1)

Authors: Naoki Shibao, Sho Tsugawa

A wide variety of information is disseminated through social media, and
content that spreads at scale can have tangible effects on the real world. To
curb the spread of harmful content and promote the dissemination of reliable
information, research on cascade graph mining has attracted increasing
attention. A promising approach in this area is Contrastive Cascade Graph
Learning (CCGL). One important task in cascade graph mining is cascade
classification, which involves categorizing cascade graphs based on their
structural characteristics. Although CCGL is expected to be effective for this
task, its performance has not yet been thoroughly evaluated. This study aims to
investigate the effectiveness of CCGL for cascade classification. Our findings
demonstrate the strong performance of CCGL in capturing platform- and
model-specific structural patterns in cascade graphs, highlighting its
potential for a range of downstream information diffusion analysis tasks.

### 2. [Peer Review and the Diffusion of Ideas](http://arxiv.org/pdf/2507.11825v1)

Authors: Binglu Wang, Zhengnan Ma, Dashun Wang, Brian Uzzi

This study examines a fundamental yet overlooked function of peer review: its
role in exposing reviewers to new and unexpected ideas. Leveraging a natural
experiment involving over half a million peer review invitations covering both
accepted and rejected manuscripts, and integrating high-scale bibliographic and
editorial records for 37,279 submitting authors, we find that exposure to a
manuscript's core ideas significantly influences the future referencing
behavior and knowledge of reviewer invitees who decline the review invite.
Specifically, declining reviewer invitees who could view concise summaries of
the manuscript's core ideas not only increase their citations to the manuscript
itself but also demonstrate expanded breadth, depth, diversity, and prominence
of citations to the submitting author's broader body of work. Overall, these
results suggest peer review substantially influences the spread of scientific
knowledge. Ironically, while the massive scale of peer review, entailing
millions of reviews annually, often drives policy debates about its costs and
burdens, our findings demonstrate that precisely because of this scale, peer
review serves as a powerful yet previously unrecognized engine for idea
diffusion, which is central to scientific advances and scholarly communication.

### 3. [Predictable Drifts in Collective Cultural Attention: Evidence from Nation-Level Library Takeout Data](http://arxiv.org/pdf/2507.12007v1)

Authors: Anders Weile Larsen, Vedran Sekara

Predicting changes in consumer attention for cultural products, such as
books, movies, and songs, is notoriously difficult. Past research on predicting
the popularity of individual products suggests the existence of intrinsic
prediction limits. However, little is known about the limits for predicting
collective attention across cultural products. Here, we analyze four years of
nationwide library loan data for approximately 2 million individuals,
comprising over 100 million loans of more than 660,000 unique books. We find
that culture, as measured by popularity distributions of loaned books, drifts
continually from month to month at a near-constant rate, leading to a growing
divergence over time, and that drifts vary between different book genres. By
linking book loans to registry data, we investigate the influence of age, sex,
educational level, and geographical area on cultural drift, finding
heterogeneous effects from the different demographic groups. Our findings have
important implications for market forecasting and developing robust recommender
systems, highlighting the need to account for specific drift dynamics for
different types of items and demographic groups.

### 4. [Freshness, Persistence and Success of Scientific Teams](http://arxiv.org/pdf/2507.12255v1)

Authors: Hanjo D. Boekhout, Eelke M. Heemskerk, Nicolò Pisani, Frank W. Takes

Team science dominates scientific knowledge production, but what makes
academic teams successful? Using temporal data on 25.2 million publications and
31.8 million authors, we propose a novel network-driven approach to identify
and study the success of persistent teams. Challenging the idea that
persistence alone drives success, we find that team freshness - new
collaborations built on prior experience - is key to success. High impact
research tends to emerge early in a team's lifespan. Analyzing complex team
overlap, we find that teams open to new collaborative ties consistently produce
better science. Specifically, team re-combinations that introduce new freshness
impulses sustain success, while persistence impulses from experienced teams are
linked to earlier impact. Together, freshness and persistence shape team
success across collaboration stages.

### 5. [Multimodal Coordinated Online Behavior: Trade-offs and Strategies](http://arxiv.org/pdf/2507.12108v1)

Authors: Lorenzo Mannocci, Stefano Cresci, Matteo Magnani, Anna Monreale, Maurizio Tesconi

Coordinated online behavior, which spans from beneficial collective actions
to harmful manipulation such as disinformation campaigns, has become a key
focus in digital ecosystem analysis. Traditional methods often rely on
monomodal approaches, focusing on single types of interactions like co-retweets
or co-hashtags, or consider multiple modalities independently of each other.
However, these approaches may overlook the complex dynamics inherent in
multimodal coordination. This study compares different ways of operationalizing
the detection of multimodal coordinated behavior. It examines the trade-off
between weakly and strongly integrated multimodal models, highlighting the
balance between capturing broader coordination patterns and identifying tightly
coordinated behavior. By comparing monomodal and multimodal approaches, we
assess the unique contributions of different data modalities and explore how
varying implementations of multimodality impact detection outcomes. Our
findings reveal that not all the modalities provide distinct insights, but that
with a multimodal approach we can get a more comprehensive understanding of
coordination dynamics. This work enhances the ability to detect and analyze
coordinated online behavior, offering new perspectives for safeguarding the
integrity of digital platforms.

### Systems and Control

### 1. [Mobility Extraction and Analysis of GaN HEMTs for RF Applications Using TCAD and Experimental Data](http://arxiv.org/pdf/2507.11849v1)

Authors: Tanjim Rahman

This paper presents an analysis of GaN high-electron-mobility transistors
(HEMTs) using both TCAD simulation and experimental characterization. The
energy band structure was studied using Nextnano simulation software to observe
two-dimensional electron gas (2DEG) formation and carrier confinement under
equilibrium conditions. Additionally, I-V and C-V data from fabricated
research-grade GaN HEMTs were analyzed to extract key electrical parameters.
The device demonstrated an ON current of 1.9 mA and an OFF current of 0.01 mA,
indicating a strong ON/OFF current ratio. A subthreshold swing of 80 mV/decade
and a DIBL of 5 mV/V were observed, confirming good gate control and
short-channel suppression. The ON-resistance was 22.72 ohm per micron, with a
saturation voltage of 1 V . The peak transconductance was extracted as 0.18 mS
in the linear region and 0.5 mS in saturation. Field-effect mobility was
calculated using the transconductance method, with a maximum value of
approximately 1200 cm2/V.s at low drain bias. The combined simulation and
experimental approach provided comprehensive insight into GaN HEMT behavior,
enabling a deeper understanding of structure-performance relationships critical
to advanced transistor design.

### 2. [Algorithm Design and Comparative Test of Natural Gradient Gaussian Approximation Filter](http://arxiv.org/pdf/2507.11872v1)

Authors: Wenhan Cao, Tianyi Zhang, Shengbo Eben Li

Popular Bayes filters typically rely on linearization techniques such as
Taylor series expansion and stochastic linear regression to use the structure
of standard Kalman filter. These techniques may introduce large estimation
errors in nonlinear and non-Gaussian systems. This paper overviews a recent
breakthrough in filtering algorithm design called \textit{N}atural
Gr\textit{a}dient Gaussia\textit{n} Appr\textit{o}ximation (NANO) filter and
compare its performance over a large class of nonlinear filters. The NANO
filter interprets Bayesian filtering as solutions to two distinct optimization
problems, which allows to define optimal Gaussian approximation and derive its
corresponding extremum conditions. The algorithm design still follows the
two-step structure of Bayes filters. In the prediction step, NANO filter
calculates the first two moments of the prior distribution, and this process is
equivalent to a moment-matching filter. In the update step, natural gradient
descent is employed to directly minimize the objective of the update step,
thereby avoiding errors caused by model linearization. Comparative tests are
conducted on four classic systems, including the damped linear oscillator,
sequence forecasting, modified growth model, and robot localization, under
Gaussian, Laplace, and Beta noise to evaluate the NANO filter's capability in
handling nonlinearity. Additionally, we validate the NANO filter's robustness
to data outliers using a satellite attitude estimation example. It is observed
that the NANO filter outperforms popular Kalman filters family such as extended
Kalman filter (EKF), unscented Kalman filter (UKF), iterated extended Kalman
filter (IEKF) and posterior linearization filter (PLF), while having similar
computational burden.

### 3. [Advantages of Feedback in Distributed Data-Gathering for Accurate and Power-Efficient State-Estimation](http://arxiv.org/pdf/2507.11924v1)

Authors: Hyeongmin Choe, Soojean Han

In distributed target-tracking sensor networks, efficient data gathering
methods are necessary to save communication resources and assure information
accuracy. This paper proposes a Feedback (FB) distributed data-gathering method
which lets the central unit feed information back to the mobile sensors; each
sensor then uses it to cancel redundant transmissions and reduce communication
congestion. We rigorously compare its performance, in terms of mean-squared
error (MSE) and cost of power per sensor, against more conventional
Non-Feedback (NF) architectures by evaluating conditions of feasibility and
advantage under different architecture specifications (e.g., communication
delay rate, power cost rate, maximum back-off time, sampling period,
observation noise). Here, we defined the advantage as the performance gain
achieved by FB over NF, while FB is said to be feasible if the advantage region
is nonempty. Our theoretical analyses show that the feasibility of FB depends
more on the communication power cost, while the advantage depends on the
sensors' propagation delay per transmission interval; we derive concrete
conditions under which these outcomes hold. Using extensive numerical
simulations under a variety of settings, we confirm the accuracy of the derived
conditions, and show that our theoretical results hold even for more complex
scenarios where the simplifying assumptions no longer hold.

### 4. [Towards Ultra-Reliable 6G in-X Subnetworks: Dynamic Link Adaptation by Deep Reinforcement Learning](http://arxiv.org/pdf/2507.12031v1)

Authors: Fateme Salehi, Aamir Mahmood, Sarder Fakhrul Abedin, Kyi Thar, Mikael Gidlund

6G networks are composed of subnetworks expected to meet ultra-reliable
low-latency communication (URLLC) requirements for mission-critical
applications such as industrial control and automation. An often-ignored aspect
in URLLC is consecutive packet outages, which can destabilize control loops and
compromise safety in in-factory environments. Hence, the current work proposes
a link adaptation framework to support extreme reliability requirements using
the soft actor-critic (SAC)-based deep reinforcement learning (DRL) algorithm
that jointly optimizes energy efficiency (EE) and reliability under dynamic
channel and interference conditions. Unlike prior work focusing on average
reliability, our method explicitly targets reducing burst/consecutive outages
through adaptive control of transmit power and blocklength based solely on the
observed signal-to-interference-plus-noise ratio (SINR). The joint optimization
problem is formulated under finite blocklength and quality of service
constraints, balancing reliability and EE. Simulation results show that the
proposed method significantly outperforms the baseline algorithms, reducing
outage bursts while consuming only 18\% of the transmission cost required by a
full/maximum resource allocation policy in the evaluated scenario. The
framework also supports flexible trade-off tuning between EE and reliability by
adjusting reward weights, making it adaptable to diverse industrial
requirements.

### 5. [Distributed Resilient State Estimation and Control with Strategically Implemented Security Measures](http://arxiv.org/pdf/2507.12052v1)

Authors: Takumi Shinohara, Karl H. Johansson, Henrik Sandberg

This paper addresses the problem of distributed resilient state estimation
and control for linear time-invariant systems in the presence of malicious
false data injection sensor attacks and bounded noise. We consider a system
operator (defender) capable of deploying cybersecurity measures to counteract
the sensor compromises. Although such measures enhance resilience against
adversarial attacks, they may incur substantial costs; hence, it is crucial to
select countermeasures to balance resilience gains and cost efficiency
strategically. We first demonstrate that the system's resilience against
attacks is maximized through the appropriate implementation of security
measures, implying that no attacker can execute undetectable sensor attacks.
Building on this analysis, we propose an algorithm that identifies the optimal
security measure. While determining this measure is NP-hard in general, we also
derive sufficient conditions under which efficient computation is feasible.
Furthermore, we develop a distributed resilient state estimation and control
scheme informed by the optimal security measure and establish conditions that
guarantee bounded estimation and control errors. Finally, we validate the
efficacy of our approach via numerical simulations of a vehicle platooning
scenario.

### 6. [Inductance Estimation for High-Power Multilayer Rectangle Planar Windings](http://arxiv.org/pdf/2507.12082v1)

Authors: Theofilos Papadopoulos, Antonios Antonopoulos

This paper proposes a simple and accurate monomial-like equation for
estimating the inductance of Multilayer Rectangle-shaped Planar Windings
(MLRPWs) for high-frequency, high-power applications. The equation consists of
the power product of the geometrical dimensions, raised at individual power
coefficients. The coefficients are generated via Multiple Linear Regression
(MLR), based on a large set of approximately 6,000 simulated windings, with an
80/20 training/evaluation sample ratio. The resulting mean error value is 0%,
with a standard deviation below 1.8%. The accuracy of the inductance estimation
is confirmed on several experimental samples, with dimensions both within and
outside the initial training dataset.

### 7. [Integrated Switched Capacitor Array and Synchronous Charge Extraction with Adaptive Hybrid MPPT for Piezoelectric Harvesters](http://arxiv.org/pdf/2507.12163v1)

Authors: Pramit Karmakar, Siddharth B, Chinmay Murlidhar Kadnur Rao

Energy Harvesting technologies will play a fundamental role in the
development of the next generation of electronic systems as well as in
advancing the development of sustainable infrastructure. One of the critical
challenges in EH is utilizing ambient vibrations to harvest energy. Piezo
Energy Harvesting, which uses ambient vibrations, is a promising technology in
energy harvesting and a self-powered technology. However, it suffers from
several practical challenges. Some of these challenges include narrow
bandwidth, non-linearity, and impedance mismatch, among others. This paper
presents a novel, simulated Piezo Energy Harvesting (PEH) framework that
addresses some of these challenges. The proposed model is designed to be
adaptive and effective against the inherent non-linearity of PEH. This detailed
model covers a non-linear piezo, Synchronous Electric Charge Extraction (SECE),
Hybrid Maximum Power Point Tracking (MPPT) and a Switched Capacitor Array
(SCA). The SECE extracts the maximum charge accumulated on the piezo every time
the piezo reaches the mechanical extremum. The Bouc-Wen model has been used to
establish nonlinearity in the system. The hybrid MPPT exhibits significant
improvement over conventional P&O, while the SCA-tuned system demonstrates
resilience against variable frequency input.

### 8. [Learning, fast and slow: a two-fold algorithm for data-based model adaptation](http://arxiv.org/pdf/2507.12187v1)

Authors: Laura Boca de Giuli, Alessio La Bella, Riccardo Scattolini

This article addresses the challenge of adapting data-based models over time.
We propose a novel two-fold modelling architecture designed to correct
plant-model mismatch caused by two types of uncertainty. Out-of-domain
uncertainty arises when the system operates under conditions not represented in
the initial training dataset, while in-domain uncertainty results from
real-world variability and flaws in the model structure or training process. To
handle out-of-domain uncertainty, a slow learning component, inspired by the
human brain's slow thinking process, learns system dynamics under unexplored
operating conditions, and it is activated only when a monitoring strategy deems
it necessary. This component consists of an ensemble of models, featuring (i) a
combination rule that weights individual models based on the statistical
proximity between their training data and the current operating condition, and
(ii) a monitoring algorithm based on statistical control charts that supervises
the ensemble's reliability and triggers the offline training and integration of
a new model when a new operating condition is detected. To address in-domain
uncertainty, a fast learning component, inspired by the human brain's fast
thinking process, continuously compensates in real time for the mismatch of the
slow learning model. This component is implemented as a Gaussian process (GP)
model, trained online at each iteration using recent data while discarding
older samples. The proposed methodology is tested on a benchmark energy system
referenced in the literature, demonstrating that the combined use of slow and
fast learning components improves model accuracy compared to standard
adaptation approaches.

### 9. [Neural Co-state Regulator: A Data-Driven Paradigm for Real-time Optimal Control with Input Constraints](http://arxiv.org/pdf/2507.12259v1)

Authors: Lihan Lian, Yuxin Tong, Uduak Inyang-Udoh

We propose a novel unsupervised learning framework for solving nonlinear
optimal control problems (OCPs) with input constraints in real-time. In this
framework, a neural network (NN) learns to predict the optimal co-state
trajectory that minimizes the control Hamiltonian for a given system, at any
system's state, based on the Pontryagin's Minimum Principle (PMP).
Specifically, the NN is trained to find the norm-optimal co-state solution that
simultaneously satisfies the nonlinear system dynamics and minimizes a
quadratic regulation cost. The control input is then extracted from the
predicted optimal co-state trajectory by solving a quadratic program (QP) to
satisfy input constraints and optimality conditions. We coin the term neural
co-state regulator (NCR) to describe the combination of the co-state NN and
control input QP solver. To demonstrate the effectiveness of the NCR, we
compare its feedback control performance with that of an expert nonlinear model
predictive control (MPC) solver on a unicycle model. Because the NCR's training
does not rely on expert nonlinear control solvers which are often suboptimal,
the NCR is able to produce solutions that outperform the nonlinear MPC solver
in terms of convergence error and input trajectory smoothness even for system
conditions that are outside its original training domain. At the same time, the
NCR offers two orders of magnitude less computational time than the nonlinear
MPC.

### 10. [Mixed-integer Second-Order Cone Programming for Multi-period Scheduling of Flexible AC Transmission System Devices](http://arxiv.org/pdf/2507.12327v1)

Authors: Mohamad Charara, Martin De Montigny, Nivine Abou Daher, Hanane Dagdougui, Antoine Lesage-Landry

With the increasing energy demand and the growing integration of renewable
sources of energy, power systems face operational challenges such as overloads,
losses, and stability concerns, particularly as networks operate near their
capacity limits. Flexible alternating current transmission system (FACTS)
devices are essential to ensure reliable grid operations and enable the
efficient integration of renewable energy. This work introduces a mixed-integer
second-order cone programming (MISOCP) model for the multi-period scheduling of
key FACTS devices in electric transmission systems. The proposed model
integrates four key control mechanisms: (i) on-load tap changers (OLTCs) for
voltage regulation via discrete taps; (ii) static synchronous compensators
(STATCOMs) and (iii) shunt reactors for reactive power compensation; and (iv)
thyristor-controlled series capacitors (TCSCs) for adjustable impedance and
flow control. The objective is to minimize active power losses using a limited
number of control actions while meeting physical and operational constraints at
all times throughout the defined time horizon. To ensure tractability, the
model employs a second-order cone relaxation of the power flow. Device-specific
constraints are handled via binary expansion and linearization: OLTCs and shunt
reactors are modelled with discrete variables, STATCOMs through reactive power
bounds, and TCSCs using a reformulation-linearization technique (RLT). A
multi-period formulation captures the sequential nature of decision making,
ensuring consistency across time steps. The model is evaluated on the IEEE
9-bus, 30-bus, and RTS96 test systems, demonstrating its ability to reduce
losses, with potential applicability to larger-scale grids.

### Machine Learning (Statistics Category)

### 1. [Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](http://arxiv.org/pdf/2507.11847v1)

Authors: Yu-Jie Zhang, Sheng-An Xu, Peng Zhao, Masashi Sugiyama

We study the generalized linear bandit (GLB) problem, a contextual
multi-armed bandit framework that extends the classical linear model by
incorporating a non-linear link function, thereby modeling a broad class of
reward distributions such as Bernoulli and Poisson. While GLBs are widely
applicable to real-world scenarios, their non-linear nature introduces
significant challenges in achieving both computational and statistical
efficiency. Existing methods typically trade off between two objectives, either
incurring high per-round costs for optimal regret guarantees or compromising
statistical efficiency to enable constant-time updates. In this paper, we
propose a jointly efficient algorithm that attains a nearly optimal regret
bound with $\mathcal{O}(1)$ time and space complexities per round. The core of
our method is a tight confidence set for the online mirror descent (OMD)
estimator, which is derived through a novel analysis that leverages the notion
of mix loss from online prediction. The analysis shows that our OMD estimator,
even with its one-pass updates, achieves statistical efficiency comparable to
maximum likelihood estimation, thereby leading to a jointly efficient
optimistic method.

### 2. [Incorporating Fairness Constraints into Archetypal Analysis](http://arxiv.org/pdf/2507.12021v1)

Authors: Aleix Alcacer, Irene Epifanio

Archetypal Analysis (AA) is an unsupervised learning method that represents
data as convex combinations of extreme patterns called archetypes. While AA
provides interpretable and low-dimensional representations, it can
inadvertently encode sensitive attributes, leading to fairness concerns. In
this work, we propose Fair Archetypal Analysis (FairAA), a modified formulation
that explicitly reduces the influence of sensitive group information in the
learned projections. We also introduce FairKernelAA, a nonlinear extension that
addresses fairness in more complex data distributions. Our approach
incorporates a fairness regularization term while preserving the structure and
interpretability of the archetypes. We evaluate FairAA and FairKernelAA on
synthetic datasets, including linear, nonlinear, and multi-group scenarios,
demonstrating their ability to reduce group separability -- as measured by mean
maximum discrepancy and linear separability -- without substantially
compromising explained variance. We further validate our methods on the
real-world ANSUR I dataset, confirming their robustness and practical utility.
The results show that FairAA achieves a favorable trade-off between utility and
fairness, making it a promising tool for responsible representation learning in
sensitive applications.

### 3. [ROC-n-reroll: How verifier imperfection affects test-time scaling](http://arxiv.org/pdf/2507.12399v1)

Authors: Florian E. Dorner, Yatong Chen, André F. Cruz, Fanny Yang

Test-time scaling aims to improve language model performance by leveraging
additional compute during inference. While many works have empirically studied
techniques like Best-of-N (BoN) and rejection sampling that make use of a
verifier to enable test-time scaling, there is little theoretical understanding
of how verifier imperfection affects performance. In this work, we address this
gap. Specifically, we prove how instance-level accuracy of these methods is
precisely characterized by the geometry of the verifier's ROC curve.
Interestingly, while scaling is determined by the local geometry of the ROC
curve for rejection sampling, it depends on global properties of the ROC curve
for BoN. As a consequence when the ROC curve is unknown, it is impossible to
extrapolate the performance of rejection sampling based on the low-compute
regime. Furthermore, while rejection sampling outperforms BoN for fixed
compute, in the infinite-compute limit both methods converge to the same level
of accuracy, determined by the slope of the ROC curve near the origin. Our
theoretical results are confirmed by experiments on GSM8K using different
versions of Llama and Qwen to generate and verify solutions.

### 4. [Choosing the Better Bandit Algorithm under Data Sharing: When Do A/B Experiments Work?](http://arxiv.org/pdf/2507.11891v1)

Authors: Shuangning Li, Chonghuan Wang, Jingyan Wang

We study A/B experiments that are designed to compare the performance of two
recommendation algorithms. Prior work has shown that the standard
difference-in-means estimator is biased in estimating the global treatment
effect (GTE) due to a particular form of interference between experimental
units. Specifically, units under the treatment and control algorithms
contribute to a shared pool of data that subsequently train both algorithms,
resulting in interference between the two groups. The bias arising from this
type of data sharing is known as "symbiosis bias". In this paper, we highlight
that, for decision-making purposes, the sign of the GTE often matters more than
its precise magnitude when selecting the better algorithm. We formalize this
insight under a multi-armed bandit framework and theoretically characterize
when the sign of the expected GTE estimate under data sharing aligns with or
contradicts the sign of the true GTE. Our analysis identifies the level of
exploration versus exploitation as a key determinant of how symbiosis bias
impacts algorithm selection.

### 5. [Newfluence: Boosting Model interpretability and Understanding in High Dimensions](http://arxiv.org/pdf/2507.11895v1)

Authors: Haolin Zou, Arnab Auddy, Yongchan Kwon, Kamiar Rahnama Rad, Arian Maleki

The increasing complexity of machine learning (ML) and artificial
intelligence (AI) models has created a pressing need for tools that help
scientists, engineers, and policymakers interpret and refine model decisions
and predictions. Influence functions, originating from robust statistics, have
emerged as a popular approach for this purpose.
  However, the heuristic foundations of influence functions rely on
low-dimensional assumptions where the number of parameters $p$ is much smaller
than the number of observations $n$. In contrast, modern AI models often
operate in high-dimensional regimes with large $p$, challenging these
assumptions.
  In this paper, we examine the accuracy of influence functions in
high-dimensional settings. Our theoretical and empirical analyses reveal that
influence functions cannot reliably fulfill their intended purpose. We then
introduce an alternative approximation, called Newfluence, that maintains
similar computational efficiency while offering significantly improved
accuracy.
  Newfluence is expected to provide more accurate insights than many existing
methods for interpreting complex AI models and diagnosing their issues.
Moreover, the high-dimensional framework we develop in this paper can also be
applied to analyze other popular techniques, such as Shapley values.

### 6. [Enhancing Signal Proportion Estimation Through Leveraging Arbitrary Covariance Structures](http://arxiv.org/pdf/2507.11922v1)

Authors: Jingtian Bai, Xinge Jessie Jeng

Accurately estimating the proportion of true signals among a large number of
variables is crucial for enhancing the precision and reliability of scientific
research. Traditional signal proportion estimators often assume independence
among variables and specific signal sparsity conditions, limiting their
applicability in real-world scenarios where such assumptions may not hold. This
paper introduces a novel signal proportion estimator that leverages arbitrary
covariance dependence information among variables, thereby improving
performance across a wide range of sparsity levels and dependence structures.
Building on previous work that provides lower confidence bounds for signal
proportions, we extend this approach by incorporating the principal factor
approximation procedure to account for variable dependence. Our theoretical
insights offer a deeper understanding of how signal sparsity, signal intensity,
and covariance dependence interact. By comparing the conditions for estimation
consistency before and after dependence adjustment, we highlight the advantages
of integrating dependence information across different contexts. This
theoretical foundation not only validates the effectiveness of the new
estimator but also guides its practical application, ensuring reliable use in
diverse scenarios. Through extensive simulations, we demonstrate that our
method outperforms state-of-the-art estimators in both estimation accuracy and
the detection of weaker signals that might otherwise go undetected.

### 7. [Designing Algorithms for Entropic Optimal Transport from an Optimisation Perspective](http://arxiv.org/pdf/2507.12246v1)

Authors: Vishwak Srinivasan, Qijia Jiang

In this work, we develop a collection of novel methods for the
entropic-regularised optimal transport problem, which are inspired by existing
mirror descent interpretations of the Sinkhorn algorithm used for solving this
problem. These are fundamentally proposed from an optimisation perspective:
either based on the associated semi-dual problem, or based on solving a
non-convex constrained problem over subset of joint distributions. This
optimisation viewpoint results in non-asymptotic rates of convergence for the
proposed methods under minimal assumptions on the problem structure. We also
propose a momentum-equipped method with provable accelerated guarantees through
this viewpoint, akin to those in the Euclidean setting. The broader framework
we develop based on optimisation over the joint distributions also finds an
analogue in the dynamical Schr\"{o}dinger bridge problem.

### 8. [Fast Variational Bayes for Large Spatial Data](http://arxiv.org/pdf/2507.12251v1)

Authors: Jiafang Song, Abhirup Datta

Recent variational Bayes methods for geospatial regression, proposed as an
alternative to computationally expensive Markov chain Monte Carlo (MCMC)
sampling, have leveraged Nearest Neighbor Gaussian processes (NNGP) to achieve
scalability. Yet, these variational methods remain inferior in accuracy and
speed compared to spNNGP, the state-of-the-art MCMC-based software for NNGP. We
introduce spVarBayes, a suite of fast variational Bayesian approaches for
large-scale geospatial data analysis using NNGP. Our contributions are
primarily computational. We replace auto-differentiation with a combination of
calculus of variations, closed-form gradient updates, and linear response
corrections for improved variance estimation. We also accommodate covariates
(fixed effects) in the model and offer inference on the variance parameters.
Simulation experiments demonstrate that we achieve comparable accuracy to
spNNGP but with reduced computational costs, and considerably outperform
existing variational inference methods in terms of both accuracy and speed.
Analysis of a large forest canopy height dataset illustrates the practical
implementation of proposed methods and shows that the inference results are
consistent with those obtained from the MCMC approach. The proposed methods are
implemented in publicly available Github R-package spVarBayes.

### 9. [Robust Causal Discovery in Real-World Time Series with Power-Laws](http://arxiv.org/pdf/2507.12257v1)

Authors: Matteo Tusoni, Giuseppe Masi, Andrea Coletta, Aldo Glielmo, Viviana Arrigoni, Novella Bartolini

Exploring causal relationships in stochastic time series is a challenging yet
crucial task with a vast range of applications, including finance, economics,
neuroscience, and climate science. Many algorithms for Causal Discovery (CD)
have been proposed, but they often exhibit a high sensitivity to noise,
resulting in misleading causal inferences when applied to real data. In this
paper, we observe that the frequency spectra of typical real-world time series
follow a power-law distribution, notably due to an inherent self-organizing
behavior. Leveraging this insight, we build a robust CD method based on the
extraction of power -law spectral features that amplify genuine causal signals.
Our method consistently outperforms state-of-the-art alternatives on both
synthetic benchmarks and real-world datasets with known causal structures,
demonstrating its robustness and practical relevance.

### 10. [A Framework for Nonstationary Gaussian Processes with Neural Network Parameters](http://arxiv.org/pdf/2507.12262v1)

Authors: Zachary James, Joseph Guinness

Gaussian processes have become a popular tool for nonparametric regression
because of their flexibility and uncertainty quantification. However, they
often use stationary kernels, which limit the expressiveness of the model and
may be unsuitable for many datasets. We propose a framework that uses
nonstationary kernels whose parameters vary across the feature space, modeling
these parameters as the output of a neural network that takes the features as
input. The neural network and Gaussian process are trained jointly using the
chain rule to calculate derivatives. Our method clearly describes the behavior
of the nonstationary parameters and is compatible with approximation methods
for scaling to large datasets. It is flexible and easily adapts to different
nonstationary kernels without needing to redesign the optimization procedure.
Our methods are implemented with the GPyTorch library and can be readily
modified. We test a nonstationary variance and noise variant of our method on
several machine learning datasets and find that it achieves better accuracy and
log-score than both a stationary model and a hierarchical model approximated
with variational inference. Similar results are observed for a model with only
nonstationary variance. We also demonstrate our approach's ability to recover
the nonstationary parameters of a spatial dataset.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-17 PST.

### 1. [Context-aware implicit neural representations to compress Earth systems model data](https://www.nature.com/articles/s41598-025-11092-w)

Authors: Farinaz Mostajeran et al.

### 2. [The influence of Gen-AI tools application for text data augmentation: case of Lithuanian educational context data classification](https://www.nature.com/articles/s41598-025-11877-z)

Authors: Pavel Stefanovič et al.

### 3. [A lightweight framework to secure IoT devices with limited resources in cloud environments](https://www.nature.com/articles/s41598-025-09885-0)

Authors: Vivek Kumar Pandey et al.

### 4. [A lightweight high-frequency mamba network for image super-resolution](https://www.nature.com/articles/s41598-025-11663-x)

Authors: Tao Wu et al.

### 5. [Machine learning based multi-parameter droplet optimisation model study](https://www.nature.com/articles/s41598-025-09435-8)

Authors: Ting Li et al.

### 6. [Physics consistent machine learning framework for inverse modeling with applications to ICF capsule implosions](https://www.nature.com/articles/s41598-025-10869-3)

Authors: Daniel A. Serino et al.

