# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-29 17:00:25.850540 PST.

### Artificial Intelligence

### 1. [Learning Individual Movement Shifts After Urban Disruptions with Social Infrastructure Reliance](http://arxiv.org/pdf/2510.23989v1)

Authors: Shangde Gao, Zelin Xu, Zhe Jiang

Shifts in individual movement patterns following disruptive events can reveal
changing demands for community resources. However, predicting such shifts
before disruptive events remains challenging for several reasons. First,
measures are lacking for individuals' heterogeneous social infrastructure
resilience (SIR), which directly influences their movement patterns, and
commonly used features are often limited or unavailable at scale, e.g.,
sociodemographic characteristics. Second, the complex interactions between
individual movement patterns and spatial contexts have not been sufficiently
captured. Third, individual-level movement may be spatially sparse and not
well-suited to traditional decision-making methods for movement predictions.
This study incorporates individuals' SIR into a conditioned deep learning model
to capture the complex relationships between individual movement patterns and
local spatial context using large-scale, sparse individual-level data. Our
experiments demonstrate that incorporating individuals' SIR and spatial context
can enhance the model's ability to predict post-event individual movement
patterns. The conditioned model can capture the divergent shifts in movement
patterns among individuals who exhibit similar pre-event patterns but differ in
SIR.

### 2. [OneCast: Structured Decomposition and Modular Generation for Cross-Domain Time Series Forecasting](http://arxiv.org/pdf/2510.24028v1)

Authors: Tingyue Pan, Mingyue Cheng, Shilong Zhang, Zhiding Liu, Xiaoyu Tao, Yucong Luo, Jintao Zhang, Qi Liu

Cross-domain time series forecasting is a valuable task in various web
applications. Despite its rapid advancement, achieving effective generalization
across heterogeneous time series data remains a significant challenge. Existing
methods have made progress by extending single-domain models, yet often fall
short when facing domain-specific trend shifts and inconsistent periodic
patterns. We argue that a key limitation lies in treating temporal series as
undifferentiated sequence, without explicitly decoupling their inherent
structural components. To address this, we propose OneCast, a structured and
modular forecasting framework that decomposes time series into seasonal and
trend components, each modeled through tailored generative pathways.
Specifically, the seasonal component is captured by a lightweight projection
module that reconstructs periodic patterns via interpretable basis functions.
In parallel, the trend component is encoded into discrete tokens at segment
level via a semantic-aware tokenizer, and subsequently inferred through a
masked discrete diffusion mechanism. The outputs from both branches are
combined to produce a final forecast that captures seasonal patterns while
tracking domain-specific trends. Extensive experiments across eight domains
demonstrate that OneCast mostly outperforms state-of-the-art baselines.

### 3. [From Observability Data to Diagnosis: An Evolving Multi-agent System for Incident Management in Cloud Systems](http://arxiv.org/pdf/2510.24145v1)

Authors: Yu Luo, Jiamin Jiang, Jingfei Feng, Lei Tao, Qingliang Zhang, Xidao Wen, Yongqian Sun, Shenglin Zhang, Jielong Huang, Nan Qi, Dan Pei

Incident management (IM) is central to the reliability of large-scale cloud
systems. Yet manual IM, where on-call engineers examine metrics, logs, and
traces is labor-intensive and error-prone in the face of massive and
heterogeneous observability data. Existing automated IM approaches often
struggle to generalize across systems, provide limited interpretability, and
incur high deployment costs, which hinders adoption in practice. In this paper,
we present OpsAgent, a lightweight, self-evolving multi-agent system for IM
that employs a training-free data processor to convert heterogeneous
observability data into structured textual descriptions, along with a
multi-agent collaboration framework that makes diagnostic inference transparent
and auditable. To support continual capability growth, OpsAgent also introduces
a dual self-evolution mechanism that integrates internal model updates with
external experience accumulation, thereby closing the deployment loop.
Comprehensive experiments on the OPENRCA benchmark demonstrate state-of-the-art
performance and show that OpsAgent is generalizable, interpretable,
cost-efficient, and self-evolving, making it a practically deployable and
sustainable solution for long-term operation in real-world cloud systems.

### 4. [BMGQ: A Bottom-up Method for Generating Complex Multi-hop Reasoning Questions from Semi-structured Data](http://arxiv.org/pdf/2510.24151v1)

Authors: Bingsen Qiu, Zijian Liu, Xiao Liu, Haoshen Yang, Zeren Gao, Bingjie Wang, Feier Zhang, Yixuan Qin, Chunyan Li

Building training-ready multi-hop question answering (QA) datasets that truly
stress a model's retrieval and reasoning abilities remains highly challenging
recently. While there have been a few recent evaluation datasets that capture
the characteristics of hard-to-search but easy-to-verify problems -- requiring
the integration of ambiguous, indirect, and cross-domain cues -- these data
resources remain scarce and are mostly designed for evaluation, making them
unsuitable for supervised fine-tuning (SFT) or reinforcement learning (RL).
Meanwhile, manually curating non-trivially retrievable questions -- where
answers cannot be found through a single direct query but instead require
multi-hop reasoning over oblique and loosely connected evidence -- incurs
prohibitive human costs and fails to scale, creating a critical data bottleneck
for training high-capability retrieval-and-reasoning agents.
  To address this, we present an automated framework for generating
high-difficulty, training-ready multi-hop questions from semi-structured
knowledge sources. The system (i) grows diverse, logically labeled evidence
clusters through Natural Language Inference (NLI)-based relation typing and
diversity-aware expansion; (ii) applies reverse question construction to
compose oblique cues so that isolated signals are underinformative but their
combination uniquely identifies the target entity; and (iii) enforces quality
with a two-step evaluation pipeline that combines multi-model consensus
filtering with structured constraint decomposition and evidence-based matching.
The result is a scalable process that yields complex, retrieval-resistant yet
verifiable questions suitable for SFT/RL training as well as challenging
evaluation, substantially reducing human curation effort while preserving the
difficulty profile of strong evaluation benchmarks.

### 5. [UniPlanner: A Unified Motion Planning Framework for Autonomous Vehicle Decision-Making Systems via Multi-Dataset Integration](http://arxiv.org/pdf/2510.24166v1)

Authors: Xin Yang, Yuhang Zhang, Wei Li, Xin Lin, Wenbin Zou, Chen Xu

Motion planning is a critical component of autonomous vehicle decision-making
systems, directly determining trajectory safety and driving efficiency. While
deep learning approaches have advanced planning capabilities, existing methods
remain confined to single-dataset training, limiting their robustness in
planning.
  Through systematic analysis, we discover that vehicular trajectory
distributions and history-future correlations demonstrate remarkable
consistency across different datasets. Based on these findings, we propose
UniPlanner, the first planning framework designed for multi-dataset integration
in autonomous vehicle decision-making. UniPlanner achieves unified
cross-dataset learning through three synergistic innovations.
  First, the History-Future Trajectory Dictionary Network (HFTDN) aggregates
history-future trajectory pairs from multiple datasets, using historical
trajectory similarity to retrieve relevant futures and generate cross-dataset
planning guidance.
  Second, the Gradient-Free Trajectory Mapper (GFTM) learns robust
history-future correlations from multiple datasets, transforming historical
trajectories into universal planning priors. Its gradient-free design ensures
the introduction of valuable priors while preventing shortcut learning, making
the planning knowledge safely transferable. Third, the Sparse-to-Dense (S2D)
paradigm implements adaptive dropout to selectively suppress planning priors
during training for robust learning, while enabling full prior utilization
during inference to maximize planning performance.

### 6. [MGA: Memory-Driven GUI Agent for Observation-Centric Interaction](http://arxiv.org/pdf/2510.24168v1)

Authors: Weihua Cheng, Ersheng Ni, Wenlong Wang, Yifei Sun, Junming Liu, Wangyu Shen, Yirong Chen, Botian Shi, Ding Wang

The rapid progress of Large Language Models (LLMs) and their multimodal
extensions (MLLMs) has enabled agentic systems capable of perceiving and acting
across diverse environments. A challenging yet impactful frontier is the
development of GUI agents, which must navigate complex desktop and web
interfaces while maintaining robustness and generalization. Existing paradigms
typically model tasks as long-chain executions, concatenating historical
trajectories into the context. While approaches such as Mirage and GTA1 refine
planning or introduce multi-branch action selection, they remain constrained by
two persistent issues: Dependence on historical trajectories, which amplifies
error propagation. And Local exploration bias, where "decision-first,
observation-later" mechanisms overlook critical interface cues. We introduce
the Memory-Driven GUI Agent (MGA), which reframes GUI interaction around the
principle of observe first, then decide. MGA models each step as an
independent, context-rich environment state represented by a triad: current
screenshot, task-agnostic spatial information, and a dynamically updated
structured memory. Experiments on OSworld benchmarks, real desktop applications
(Chrome, VSCode, VLC), and cross-task transfer demonstrate that MGA achieves
substantial gains in robustness, generalization, and efficiency compared to
state-of-the-art baselines. The code is publicly available at:
{https://anonymous.4open.science/r/MGA-3571}.

### 7. [MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](http://arxiv.org/pdf/2510.24284v1)

Authors: Wenhao Wang, Peizhi Niu, Zhao Xu, Zhaoyu Chen, Jian Du, Yaxin Du, Xianghe Pang, Keduan Huang, Yanfeng Wang, Qiang Yan, Siheng Chen

Large Language Models (LLMs) increasingly rely on external tools to perform
complex, realistic tasks, yet their ability to utilize the rapidly expanding
Model Contextual Protocol (MCP) ecosystem remains limited. Existing MCP
research covers few servers, depends on costly manual curation, and lacks
training support, hindering progress toward real-world deployment. To overcome
these limitations, we introduce MCP-Flow, an automated web-agent-driven
pipeline for large-scale server discovery, data synthesis, and model training.
MCP-Flow collects and filters data from 1166 servers and 11536 tools, producing
68733 high-quality instruction-function call pairs and 6439 trajectories, far
exceeding prior work in scale and diversity. Extensive experiments demonstrate
MCP-Flow's effectiveness in driving superior MCP tool selection, function-call
generation, and enhanced agentic task performance. MCP-Flow thus provides a
scalable foundation for advancing LLM agents' proficiency in real-world MCP
environments. MCP-Flow is publicly available at
\href{https://github.com/wwh0411/MCP-Flow}{https://github.com/wwh0411/MCP-Flow}.

### 8. [Investigating Intra-Abstraction Policies For Non-exact Abstraction Algorithms](http://arxiv.org/pdf/2510.24297v1)

Authors: Robin Schmöcker, Alexander Dockhorn, Bodo Rosenhahn

One weakness of Monte Carlo Tree Search (MCTS) is its sample efficiency which
can be addressed by building and using state and/or action abstractions in
parallel to the tree search such that information can be shared among nodes of
the same layer. The primary usage of abstractions for MCTS is to enhance the
Upper Confidence Bound (UCB) value during the tree policy by aggregating visits
and returns of an abstract node. However, this direct usage of abstractions
does not take the case into account where multiple actions with the same parent
might be in the same abstract node, as these would then all have the same UCB
value, thus requiring a tiebreak rule. In state-of-the-art abstraction
algorithms such as pruned On the Go Abstractions (pruned OGA), this case has
not been noticed, and a random tiebreak rule was implicitly chosen. In this
paper, we propose and empirically evaluate several alternative
intra-abstraction policies, several of which outperform the random policy
across a majority of environments and parameter settings.

### 9. [Verifying Large Language Models' Reasoning Paths via Correlation Matrix Rank](http://arxiv.org/pdf/2510.24299v1)

Authors: Jiayu Liu, Wei Dai, Zhenya Huang, Ning Miao, Enhong Chen

Despite the strong reasoning ability of large language models~(LLMs), they
are prone to errors and hallucinations. As a result, how to check their outputs
effectively and efficiently has become a critical problem in their
applications. Existing checking methods heavily rely on external resources,
such as trained verifiers (e.g., process/outcome reward models) or elaborate
prompts, which lead to high computational overhead and are only applicable to
specific domains. In this paper, we investigate whether the internal behaviors
of LLMs have already implied the credibility of their reasoning paths.
Specifically, we find that the rank of the correlation matrix between the input
problem and the output reasoning path is a robust indicator of reasoning
correctness. Different from other correctness indicators for LLMs, the
calculation of the correlation matrix only relies on the LLM itself, which
avoids the hassle of training a separate model or designing complicated
prompts. Based on it, we design a simple, plug-and-play Self-Indicator method
to reweight candidate reasoning paths, which achieves significant performance
improvements than other voting and verification methods with very few
computational overhead. Our experiments across multiple LLMs of varying scales
and model families have further shown the effectiveness of Self-Indicator. It
achieves over 75% accuracy in distinguishing correct reasoning paths from
incorrect ones, and, in turn, improves the accuracies on three reasoning
benchmarks by more than 8%.

### 10. [Retrieval and Argumentation Enhanced Multi-Agent LLMs for Judgmental Forecasting](http://arxiv.org/pdf/2510.24303v1)

Authors: Deniz Gorur, Antoni Rago, Francesca Toni

Judgmental forecasting is the task of making predictions about future events
based on human judgment. This task can be seen as a form of claim verification,
where the claim corresponds to a future event and the task is to assess the
plausibility of that event. In this paper, we propose a novel multi-agent
framework for claim verification, whereby different agents may disagree on
claim veracity and bring specific evidence for and against the claims,
represented as quantitative bipolar argumentation frameworks (QBAFs). We then
instantiate the framework for supporting claim verification, with a variety of
agents realised with Large Language Models (LLMs): (1) ArgLLM agents, an
existing approach for claim verification that generates and evaluates QBAFs;
(2) RbAM agents, whereby LLM-empowered Relation-based Argument Mining (RbAM)
from external sources is used to generate QBAFs; (3) RAG-ArgLLM agents,
extending ArgLLM agents with a form of Retrieval-Augmented Generation (RAG) of
arguments from external sources. Finally, we conduct experiments with two
standard judgmental forecasting datasets, with instances of our framework with
two or three agents, empowered by six different base LLMs. We observe that
combining evidence from agents can improve forecasting accuracy, especially in
the case of three agents, while providing an explainable combination of
evidence for claim verification.

### Hardware Architecture

### 1. [SlowPoke: Understanding and Detecting On-Chip Fail-Slow Failures in Many-Core Systems](http://arxiv.org/pdf/2510.24112v1)

Authors: Junchi Wu, Xinfei Wan, Zhuoran Li, Yuyang Jin, Guangyu Sun, Yun Liang, Diyu Zhou, Youwei Zhuo

Many-core architectures are essential for high-performance computing, but
their performance is undermined by widespread fail-slow failures. Detecting
such failures on-chip is challenging, as prior methods from distributed systems
are unsuitable due to strict memory limits and their inability to track
failures across the hardware topology. This paper introduces SlowPoke, a
lightweight, hardware-aware framework for practical on-chip fail-slow
detection. SlowPoke combines compiler-based instrumentation for low-overhead
monitoring, on-the-fly trace compression to operate within kilobytes of memory,
and a novel topology-aware ranking algorithm to pinpoint a failure's root
cause. We evaluate SlowPoke on a wide range of representative many-core
workloads, and the results demonstrate that SlowPoke reduces the storage
overhead of detection traces by an average of 115.9$\times$, while achieving an
average fail-slow detection accuracy of 86.77% and a false positive rate (FPR)
of 12.11%. More importantly, SlowPoke scales effectively across different
many-core architectures, making it practical for large-scale deployments.

### 2. [Taming the Tail: NoI Topology Synthesis for Mixed DL Workloads on Chiplet-Based Accelerators](http://arxiv.org/pdf/2510.24113v1)

Authors: Arnav Shukla, Harsh Sharma, Srikant Bharadwaj, Vinayak Abrol, Sujay Deb

Heterogeneous chiplet-based systems improve scaling by disag-gregating
CPUs/GPUs and emerging technologies (HBM/DRAM).However this on-package
disaggregation introduces a latency inNetwork-on-Interposer(NoI). We observe
that in modern large-modelinference, parameters and activations routinely move
backand forth from HBM/DRAM, injecting large, bursty flows into theinterposer.
These memory-driven transfers inflate tail latency andviolate Service Level
Agreements (SLAs) across k-ary n-cube base-line NoI topologies. To address this
gap we introduce an InterferenceScore (IS) that quantifies worst-case slowdown
under contention.We then formulate NoI synthesis as a multi-objective
optimization(MOO) problem. We develop PARL (Partition-Aware
ReinforcementLearner), a topology generator that balances throughput,
latency,and power. PARL-generated topologies reduce contention at the memory
cut, meet SLAs, and cut worst-case slowdown to 1.2 times while maintaining
competitive mean throughput relative to link-rich meshes. Overall, this
reframes NoI design for heterogeneouschiplet accelerators with workload-aware
objectives.

### 3. [TsetlinKWS: A 65nm 16.58uW, 0.63mm2 State-Driven Convolutional Tsetlin Machine-Based Accelerator For Keyword Spotting](http://arxiv.org/pdf/2510.24282v1)

Authors: Baizhou Lin, Yuetong Fang, Renjing Xu, Rishad Shafik, Jagmohan Chauhan

The Tsetlin Machine (TM) has recently attracted attention as a low-power
alternative to neural networks due to its simple and interpretable inference
mechanisms. However, its performance on speech-related tasks remains limited.
This paper proposes TsetlinKWS, the first algorithm-hardware co-design
framework for the Convolutional Tsetlin Machine (CTM) on the 12-keyword
spotting task. Firstly, we introduce a novel Mel-Frequency Spectral Coefficient
and Spectral Flux (MFSC-SF) feature extraction scheme together with spectral
convolution, enabling the CTM to reach its first-ever competitive accuracy of
87.35% on the 12-keyword spotting task. Secondly, we develop an Optimized
Grouped Block-Compressed Sparse Row (OG-BCSR) algorithm that achieves a
remarkable 9.84$\times$ reduction in model size, significantly improving the
storage efficiency on CTMs. Finally, we propose a state-driven architecture
tailored for the CTM, which simultaneously exploits data reuse and sparsity to
achieve high energy efficiency. The full system is evaluated in 65 nm process
technology, consuming 16.58 $\mu$W at 0.7 V with a compact 0.63 mm$^2$ core
area. TsetlinKWS requires only 907k logic operations per inference,
representing a 10$\times$ reduction compared to the state-of-the-art KWS
accelerators, positioning the CTM as a highly-efficient candidate for
ultra-low-power speech applications.

### 4. [Attack on a PUF-based Secure Binary Neural Network](http://arxiv.org/pdf/2510.24422v1)

Authors: Bijeet Basak, Nupur Patil, Kurian Polachan, Srinivas Vivek

Binarized Neural Networks (BNNs) deployed on memristive crossbar arrays
provide energy-efficient solutions for edge computing but are susceptible to
physical attacks due to memristor nonvolatility. Recently, Rajendran et al.
(IEEE Embedded Systems Letter 2025) proposed a Physical Unclonable Function
(PUF)-based scheme to secure BNNs against theft attacks. Specifically, the
weight and bias matrices of the BNN layers were secured by swapping columns
based on device's PUF key bits.
  In this paper, we demonstrate that this scheme to secure BNNs is vulnerable
to PUF-key recovery attack. As a consequence of our attack, we recover the
secret weight and bias matrices of the BNN. Our approach is motivated by
differential cryptanalysis and reconstructs the PUF key bit-by-bit by observing
the change in model accuracy, and eventually recovering the BNN model
parameters. Evaluated on a BNN trained on the MNIST dataset, our attack could
recover 85% of the PUF key, and recover the BNN model up to 93% classification
accuracy compared to the original model's 96% accuracy. Our attack is very
efficient and it takes a couple of minutes to recovery the PUF key and the
model parameters.

### Computational Complexity

### 1. [Near Optimal Hardness of Approximating $k$-CSP](http://arxiv.org/pdf/2510.23991v1)

Authors: Dor Minzer, Kai Zhe Zheng

We show that for every $k\in\mathbb{N}$ and $\varepsilon>0$, for large enough
alphabet $R$, given a $k$-CSP with alphabet size $R$, it is NP-hard to
distinguish between the case that there is an assignment satisfying at least
$1-\varepsilon$ fraction of the constraints, and the case no assignment
satisfies more than $1/R^{k-1-\varepsilon}$ of the constraints. This result
improves upon prior work of [Chan, Journal of the ACM 2016], who showed the
same result with weaker soundness of $O(k/R^{k-2})$, and nearly matches the
trivial approximation algorithm that finds an assignment satisfying at least
$1/R^{k-1}$ fraction of the constraints.
  Our proof follows the approach of a recent work by the authors, wherein the
above result is proved for $k=2$. Our main new ingredient is a counting lemma
for hyperedges between pseudo-random sets in the Grassmann graphs, which may be
of independent interest.

### 2. [Reachability of Independent Sets and Vertex Covers Under Extended Reconfiguration Rules](http://arxiv.org/pdf/2510.24226v1)

Authors: Shuichi Hirahara, Naoto Ohsaka, Tatsuhiro Suga, Akira Suzuki, Yuma Tamura, Xiao Zhou

In reconfiguration problems, we are given two feasible solutions to a graph
problem and asked whether one can be transformed into the other via a sequence
of feasible intermediate solutions under a given reconfiguration rule. While
earlier work focused on modifying a single element at a time, recent studies
have started examining how different rules impact computational complexity.
Motivated by recent progress, we study Independent Set Reconfiguration (ISR)
and Vertex Cover Reconfiguration (VCR) under the $k$-Token Jumping ($k$-TJ) and
$k$-Token Sliding ($k$-TS) models. In $k$-TJ, up to $k$ vertices may be
replaced, while $k$-TS additionally requires a perfect matching between removed
and added vertices. It is known that the complexity of ISR crucially depends on
$k$, ranging from PSPACE-complete and NP-complete to polynomial-time solvable.
In this paper, we further explore the gradient of computational complexity of
the problems. We first show that ISR under $k$-TJ with $k = |I| - \mu$ remains
NP-hard when $\mu$ is any fixed positive integer and the input graph is
restricted to graphs of maximum degree 3 or planar graphs of maximum degree 4,
where $|I|$ is the size of feasible solutions. In addition, we prove that the
problem belongs to NP not only for $\mu=O(1)$ but also for $\mu = O(\log |I|)$.
In contrast, we show that VCR under $k$-TJ is in XP when parameterized by $\mu
= |S| - k$, where $|S|$ is the size of feasible solutions. Furthermore, we
establish the PSPACE-completeness of ISR and VCR under both $k$-TJ and $k$-TS
on several graph classes, for fixed $k$ as well as superconstant $k$ relative
to the size of feasible solutions.

### Computational Engineering

### 1. [UniField: Joint Multi-Domain Training for Universal Surface Pressure Modeling](http://arxiv.org/pdf/2510.24106v1)

Authors: Junhong Zou, Zhenxu Sun, Yueqing Wang, Wei Qiu, Zhaoxiang Zhang, Zhen Lei, Xiangyu Zhu

Aerodynamic simulation of the surface pressure field around objects is
crucial for many engineering problems. In recent years, deep neural networks
have emerged as an efficient alternative to traditional, computationally
expensive CFD simulations for modeling surface pressure fields. However, data
scarcity remains a fundamental challenge, limiting the application of neural
networks. To address this limitation, we propose to integrate aerodynamic data
from multiple subfields and conduct joint training to learn more general field
representations. We consolidate five different datasets covering various
fields, including automobiles, trains, aircraft, and general shapes. Facing
significant data differences across different domains, we propose UniField,
which employs a domain-agnostic Transformer module to extract general point
cloud features and customizes domain-specific flow-conditioned adapters to
adapt to the flow information in different subfields. Despite the fact that
aerodynamic data from different subfields are typically governed by different
equations, we compare models trained jointly on all data with those trained
separately on individual datasets and find that the jointly-trained model
commonly demonstrates better performance. This indicates that these data
complement each other to help the model learn better flow field
representations. These results highlight the potential of UniField as a
universal flow field representation model and lay the foundation for broader
applications of neural networks in aerodynamic analysis.

### 2. [A data-driven multiscale scheme for anisotropic finite strain magneto-elasticity](http://arxiv.org/pdf/2510.24197v1)

Authors: Heinrich T. Roth, Philipp Gebhart, Karl A. Kalina, Thomas Wallmersperger, Markus Kästner

In this work, we develop a neural network-based, data-driven, decoupled
multiscale scheme for the modeling of structured magnetically soft
magnetorheological elastomers (MREs). On the microscale, sampled
magneto-mechanical loading paths are imposed on a representative volume element
containing spherical particles and an elastomer matrix, and the resulting
boundary value problem is solved using a mixed finite element formulation. The
computed microscale responses are homogenized to construct a database for the
training and testing of a macroscopic physics-augmented neural network model.
The proposed model automatically detects the material's preferred direction
during training and enforces key physical principles, including objectivity,
material symmetry, thermodynamic consistency, and the normalization of free
energy, stress, and magnetization. Within the range of the training data, the
model enables accurate predictions of magnetization, mechanical stress, and
total stress. For larger magnetic fields, the model yields plausible results.
Finally, we apply the model to investigate the magnetostrictive behavior of a
macroscopic spherical MRE sample, which exhibits contraction along the magnetic
field direction when aligned with the material's preferred direction.

### 3. [A GPU-based Compressible Combustion Solver for Applications Exhibiting Disparate Space and Time Scales](http://arxiv.org/pdf/2510.23993v1)

Authors: Anthony Carreon, Jagmohan Singh, Shivank Sharma, Shuzhi Zhang, Venkat Raman

High-speed chemically active flows present significant computational
challenges due to their disparate space and time scales, where stiff chemistry
often dominates simulation time. While modern supercomputing scientific codes
achieve exascale performance by leveraging graphics processing units (GPUs),
existing GPU-based compressible combustion solvers face critical limitations in
memory management, load balancing, and handling the highly localized nature of
chemical reactions. To this end, we present a high-performance compressible
reacting flow solver built on the AMReX framework and optimized for multi-GPU
settings. Our approach addresses three GPU performance bottlenecks: memory
access patterns through column-major storage optimization, computational
workload variability via a bulk-sparse integration strategy for chemical
kinetics, and multi-GPU load distribution for adaptive mesh refinement
applications. The solver adapts existing matrix-based chemical kinetics
formulations to multigrid contexts. Using representative combustion
applications including hydrogen-air detonations and jet in supersonic crossflow
configurations, we demonstrate $2-5\times$ performance improvements over
initial GPU implementations with near-ideal weak scaling across $1-96$ NVIDIA
H100 GPUs. Roofline analysis reveals substantial improvements in arithmetic
intensity for both convection ($\sim 10 \times$) and chemistry ($\sim 4
\times$) routines, confirming efficient utilization of GPU memory bandwidth and
computational resources.

### 4. [HergNet: a Fast Neural Surrogate Model for Sound Field Predictions via Superposition of Plane Waves](http://arxiv.org/pdf/2510.24279v1)

Authors: Matteo Calafà, Yuanxin Xia, Cheol-Ho Jeong

We present a novel neural network architecture for the efficient prediction
of sound fields in two and three dimensions. The network is designed to
automatically satisfy the Helmholtz equation, ensuring that the outputs are
physically valid. Therefore, the method can effectively learn solutions to
boundary-value problems in various wave phenomena, such as acoustics, optics,
and electromagnetism. Numerical experiments show that the proposed strategy can
potentially outperform state-of-the-art methods in room acoustics simulation,
in particular in the range of mid to high frequencies.

### 5. [Metadata-Driven Retrieval-Augmented Generation for Financial Question Answering](http://arxiv.org/pdf/2510.24402v1)

Authors: Michail Dadopoulos, Anestis Ladas, Stratos Moschidis, Ioannis Negkakis

Retrieval-Augmented Generation (RAG) struggles on long, structured financial
filings where relevant evidence is sparse and cross-referenced. This paper
presents a systematic investigation of advanced metadata-driven
Retrieval-Augmented Generation (RAG) techniques, proposing and evaluating a
novel, multi-stage RAG architecture that leverages LLM-generated metadata. We
introduce a sophisticated indexing pipeline to create contextually rich
document chunks and benchmark a spectrum of enhancements, including
pre-retrieval filtering, post-retrieval reranking, and enriched embeddings,
benchmarked on the FinanceBench dataset. Our results reveal that while a
powerful reranker is essential for precision, the most significant performance
gains come from embedding chunk metadata directly with text ("contextual
chunks"). Our proposed optimal architecture combines LLM-driven pre-retrieval
optimizations with these contextual embeddings to achieve superior performance.
Additionally, we present a custom metadata reranker that offers a compelling,
cost-effective alternative to commercial solutions, highlighting a practical
trade-off between peak performance and operational efficiency. This study
provides a blueprint for building robust, metadata-aware RAG systems for
financial document analysis.

### 6. [Semi-supervised and unsupervised learning for health indicator extraction from guided waves in aerospace composite structures](http://arxiv.org/pdf/2510.24614v1)

Authors: James Josep Perry, Pablo Garcia-Conde Ortiz, George Konstantinou, Cornelie Vergouwen, Edlyn Santha Kumaran, Morteza Moradi

Health indicators (HIs) are central to diagnosing and prognosing the
condition of aerospace composite structures, enabling efficient maintenance and
operational safety. However, extracting reliable HIs remains challenging due to
variability in material properties, stochastic damage evolution, and diverse
damage modes. Manufacturing defects (e.g., disbonds) and in-service incidents
(e.g., bird strikes) further complicate this process. This study presents a
comprehensive data-driven framework that learns HIs via two learning approaches
integrated with multi-domain signal processing. Because ground-truth HIs are
unavailable, a semi-supervised and an unsupervised approach are proposed: (i) a
diversity deep semi-supervised anomaly detection (Diversity-DeepSAD) approach
augmented with continuous auxiliary labels used as hypothetical damage proxies,
which overcomes the limitation of prior binary labels that only distinguish
healthy and failed states while neglecting intermediate degradation, and (ii) a
degradation-trend-constrained variational autoencoder (DTC-VAE), in which the
monotonicity criterion is embedded via an explicit trend constraint. Guided
waves with multiple excitation frequencies are used to monitor single-stiffener
composite structures under fatigue loading. Time, frequency, and time-frequency
representations are explored, and per-frequency HIs are fused via unsupervised
ensemble learning to mitigate frequency dependence and reduce variance. Using
fast Fourier transform features, the augmented Diversity-DeepSAD model achieved
81.6% performance, while DTC-VAE delivered the most consistent HIs with 92.3%
performance, outperforming existing baselines.

### Computational Geometry

### 1. [Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers](http://arxiv.org/pdf/2510.24621v1)

Authors: Ziyi Fang, Lingxiao Huang, Runkai Yang

We study the robust geometric median problem in Euclidean space
$\mathbb{R}^d$, with a focus on coreset construction.A coreset is a compact
summary of a dataset $P$ of size $n$ that approximates the robust cost for all
centers $c$ within a multiplicative error $\varepsilon$. Given an outlier count
$m$, we construct a coreset of size $\tilde{O}(\varepsilon^{-2} \cdot
\min\{\varepsilon^{-2}, d\})$ when $n \geq 4m$, eliminating the $O(m)$
dependency present in prior work [Huang et al., 2022 & 2023]. For the special
case of $d = 1$, we achieve an optimal coreset size of
$\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n} \varepsilon^{-1})$, revealing
a clear separation from the vanilla case studied in [Huang et al., 2023;
Afshani and Chris, 2024]. Our results further extend to robust
$(k,z)$-clustering in various metric spaces, eliminating the $m$-dependence
under mild data assumptions. The key technical contribution is a novel
non-component-wise error analysis, enabling substantial reduction of outlier
influence, unlike prior methods that retain them.Empirically, our algorithms
consistently outperform existing baselines in terms of size-accuracy tradeoffs
and runtime, even when data assumptions are violated across a wide range of
datasets.

### Computation and Language

### 1. [M-Eval: A Heterogeneity-Based Framework for Multi-evidence Validation in Medical RAG Systems](http://arxiv.org/pdf/2510.23995v1)

Authors: Mengzhou Sun, Sendong Zhao, Jianyu Chen, Haochun Wang, Bin Qin

Retrieval-augmented Generation (RAG) has demonstrated potential in enhancing
medical question-answering systems through the integration of large language
models (LLMs) with external medical literature. LLMs can retrieve relevant
medical articles to generate more professional responses efficiently. However,
current RAG applications still face problems. They generate incorrect
information, such as hallucinations, and they fail to use external knowledge
correctly. To solve these issues, we propose a new method named M-Eval. This
method is inspired by the heterogeneity analysis approach used in
Evidence-Based Medicine (EBM). Our approach can check for factual errors in RAG
responses using evidence from multiple sources. First, we extract additional
medical literature from external knowledge bases. Then, we retrieve the
evidence documents generated by the RAG system. We use heterogeneity analysis
to check whether the evidence supports different viewpoints in the response. In
addition to verifying the accuracy of the response, we also assess the
reliability of the evidence provided by the RAG system. Our method shows an
improvement of up to 23.31% accuracy across various LLMs. This work can help
detect errors in current RAG-based medical systems. It also makes the
applications of LLMs more reliable and reduces diagnostic errors.

### 2. [PICOs-RAG: PICO-supported Query Rewriting for Retrieval-Augmented Generation in Evidence-Based Medicine](http://arxiv.org/pdf/2510.23998v1)

Authors: Mengzhou Sun, Sendong Zhao, Jianyu Chen, Bin Qin

Evidence-based medicine (EBM) research has always been of paramount
importance. It is important to find appropriate medical theoretical support for
the needs from physicians or patients to reduce the occurrence of medical
accidents. This process is often carried out by human querying relevant
literature databases, which lacks objectivity and efficiency. Therefore,
researchers utilize retrieval-augmented generation (RAG) to search for evidence
and generate responses automatically. However, current RAG methods struggle to
handle complex queries in real-world clinical scenarios. For example, when
queries lack certain information or use imprecise language, the model may
retrieve irrelevant evidence and generate unhelpful answers. To address this
issue, we present the PICOs-RAG to expand the user queries into a better
format. Our method can expand and normalize the queries into professional ones
and use the PICO format, a search strategy tool present in EBM, to extract the
most important information used for retrieval. This approach significantly
enhances retrieval efficiency and relevance, resulting in up to an 8.8\%
improvement compared to the baseline evaluated by our method. Thereby the
PICOs-RAG improves the performance of the large language models into a helpful
and reliable medical assistant in EBM.

### 3. [META-RAG: Meta-Analysis-Inspired Evidence-Re-Ranking Method for Retrieval-Augmented Generation in Evidence-Based Medicine](http://arxiv.org/pdf/2510.24003v1)

Authors: Mengzhou Sun, Sendong Zhao, Jianyu Chen, Haochun Wang, Bin Qin

Evidence-based medicine (EBM) holds a crucial role in clinical application.
Given suitable medical articles, doctors effectively reduce the incidence of
misdiagnoses. Researchers find it efficient to use large language models (LLMs)
techniques like RAG for EBM tasks. However, the EBM maintains stringent
requirements for evidence, and RAG applications in EBM struggle to efficiently
distinguish high-quality evidence. Therefore, inspired by the meta-analysis
used in EBM, we provide a new method to re-rank and filter the medical
evidence. This method presents multiple principles to filter the best evidence
for LLMs to diagnose. We employ a combination of several EBM methods to emulate
the meta-analysis, which includes reliability analysis, heterogeneity analysis,
and extrapolation analysis. These processes allow the users to retrieve the
best medical evidence for the LLMs. Ultimately, we evaluate these high-quality
articles and show an accuracy improvement of up to 11.4% in our experiments and
results. Our method successfully enables RAG to extract higher-quality and more
reliable evidence from the PubMed dataset. This work can reduce the infusion of
incorrect knowledge into responses and help users receive more effective
replies.

### 4. [TEXT2DB: Integration-Aware Information Extraction with Large Language Model Agents](http://arxiv.org/pdf/2510.24014v1)

Authors: Yizhu Jiao, Sha Li, Sizhe Zhou, Heng Ji, Jiawei Han

The task of information extraction (IE) is to extract structured knowledge
from text. However, it is often not straightforward to utilize IE output due to
the mismatch between the IE ontology and the downstream application needs. We
propose a new formulation of IE TEXT2DB that emphasizes the integration of IE
output and the target database (or knowledge base). Given a user instruction, a
document set, and a database, our task requires the model to update the
database with values from the document set to satisfy the user instruction.
This task requires understanding user instructions for what to extract and
adapting to the given DB/KB schema for how to extract on the fly. To evaluate
this new task, we introduce a new benchmark featuring common demands such as
data infilling, row population, and column addition. In addition, we propose an
LLM agent framework OPAL (Observe-PlanAnalyze LLM) which includes an Observer
component that interacts with the database, the Planner component that
generates a code-based plan with calls to IE models, and the Analyzer component
that provides feedback regarding code quality before execution. Experiments
show that OPAL can successfully adapt to diverse database schemas by generating
different code plans and calling the required IE models. We also highlight
difficult cases such as dealing with large databases with complex dependencies
and extraction hallucination, which we believe deserve further investigation.
Source code: https://github.com/yzjiao/Text2DB

### 5. [Success and Cost Elicit Convention Formation for Efficient Communication](http://arxiv.org/pdf/2510.24023v1)

Authors: Saujas Vaduguru, Yilun Hua, Yoav Artzi, Daniel Fried

Humans leverage shared conversational context to become increasingly
successful and efficient at communicating over time. One manifestation of this
is the formation of ad hoc linguistic conventions, which allow people to
coordinate on short, less costly utterances that are understood using shared
conversational context. We present a method to train large multimodal models to
form conventions, enabling efficient communication. Our approach uses simulated
reference games between models, and requires no additional human-produced data.
In repeated reference games involving photographs and tangram images, our
method enables models to communicate efficiently with people: reducing the
message length by up to 41% while increasing success by 15% over the course of
the interaction. Human listeners respond faster when interacting with our model
that forms conventions. We also show that training based on success or cost
alone is insufficient - both are necessary to elicit convention formation.

### 6. [Pie: A Programmable Serving System for Emerging LLM Applications](http://arxiv.org/pdf/2510.24051v1)

Authors: In Gim, Zhiyao Ma, Seung-seob Lee, Lin Zhong

Emerging large language model (LLM) applications involve diverse reasoning
strategies and agentic workflows, straining the capabilities of existing
serving systems built on a monolithic token generation loop. This paper
introduces Pie, a programmable LLM serving system designed for flexibility and
efficiency. Pie decomposes the traditional generation loop into fine-grained
service handlers exposed via an API and delegates control of the generation
process to user-provided programs, called inferlets. This enables applications
to implement new KV cache strategies, bespoke generation logic, and seamlessly
integrate computation and I/O-entirely within the application, without
requiring modifications to the serving system. Pie executes inferlets using
WebAssembly, benefiting from its lightweight sandboxing. Our evaluation shows
Pie matches state-of-the-art performance on standard tasks (3-12% latency
overhead) while significantly improving latency and throughput (1.3x-3.4x
higher) on agentic workflows by enabling application-specific optimizations.

### 7. [Challenging Multilingual LLMs: A New Taxonomy and Benchmark for Unraveling Hallucination in Translation](http://arxiv.org/pdf/2510.24073v1)

Authors: Xinwei Wu, Heng Liu, Jiang Zhou, Xiaohu Zhao, Linlong Xu, Longyue Wang, Weihua Luo, Kaifu Zhang

Large Language Models (LLMs) have advanced machine translation but remain
vulnerable to hallucinations. Unfortunately, existing MT benchmarks are not
capable of exposing failures in multilingual LLMs. To disclose hallucination in
multilingual LLMs, we introduce a diagnostic framework with a taxonomy that
separates Instruction Detachment from Source Detachment. Guided by this
taxonomy, we create HalloMTBench, a multilingual, human-verified benchmark
across 11 English-to-X directions. We employed 4 frontier LLMs to generate
candidates and scrutinize these candidates with an ensemble of LLM judges, and
expert validation. In this way, we curate 5,435 high-quality instances. We have
evaluated 17 LLMs on HalloMTBench. Results reveal distinct ``hallucination
triggers'' -- unique failure patterns reflecting model scale, source length
sensitivity, linguistic biases, and Reinforcement-Learning (RL) amplified
language mixing. HalloMTBench offers a forward-looking testbed for diagnosing
LLM translation failures. HalloMTBench is available in
https://huggingface.co/collections/AIDC-AI/marco-mt.

### 8. [Global PIQA: Evaluating Physical Commonsense Reasoning Across 100+ Languages and Cultures](http://arxiv.org/pdf/2510.24081v1)

Authors: Tyler A. Chang, Catherine Arnett, Abdelrahman Eldesokey, Abdelrahman Sadallah, Abeer Kashar, Abolade Daud, Abosede Grace Olanihun, Adamu Labaran Mohammed, Adeyemi Praise, Adhikarinayum Meerajita Sharma, Aditi Gupta, Afitab Iyigun, Afonso Simplício, Ahmed Essouaied, Aicha Chorana, Akhil Eppa, Akintunde Oladipo, Akshay Ramesh, Aleksei Dorkin, Alfred Malengo Kondoro, Alham Fikri Aji, Ali Eren Çetintaş, Allan Hanbury, Alou Dembele, Alp Niksarli, Álvaro Arroyo, Amin Bajand, Amol Khanna, Ana Chkhaidze, Ana Condez, Andiswa Mkhonto, Andrew Hoblitzell, Andrew Tran, Angelos Poulis, Anirban Majumder, Anna Vacalopoulou, Annette Kuuipolani Kanahele Wong, Annika Simonsen, Anton Kovalev, Ashvanth. S, Ayodeji Joseph Lana, Barkin Kinay, Bashar Alhafni, Benedict Cibalinda Busole, Bernard Ghanem, Bharti Nathani, Biljana Stojanovska Đurić, Bola Agbonile, Bragi Bergsson, Bruce Torres Fischer, Burak Tutar, Burcu Alakuş Çınar, Cade J. Kanoniakapueo Kane, Can Udomcharoenchaikit, Catherine Arnett, Chadi Helwe, Chaithra Reddy Nerella, Chen Cecilia Liu, Chiamaka Glory Nwokolo, Cristina España-Bonet, Cynthia Amol, DaeYeop Lee, Dana Arad, Daniil Dzenhaliou, Daria Pugacheva, Dasol Choi, Daud Abolade, David Liu, David Semedo, Deborah Popoola, Deividas Mataciunas, Delphine Nyaboke, Dhyuthy Krishna Kumar, Diogo Glória-Silva, Diogo Tavares, Divyanshu Goyal, DongGeon Lee, Ebele Nwamaka Anajemba, Egonu Ngozi Grace, Elena Mickel, Elena Tutubalina, Elias Herranen, Emile Anand, Emmanuel Habumuremyi, Emuobonuvie Maria Ajiboye, Eryawan Presma Yulianrifat, Esther Adenuga, Ewa Rudnicka, Faith Olabisi Itiola, Faran Taimoor Butt, Fathima Thekkekara, Fatima Haouari, Filbert Aurelian Tjiaranata, Firas Laakom, Francesca Grasso, Francesco Orabona, Francesco Periti, Gbenga Kayode Solomon, Gia Nghia Ngo, Gloria Udhehdhe-oze, Gonçalo Martins, Gopi Naga Sai Ram Challagolla, Guijin Son, Gulnaz Abdykadyrova, Hafsteinn Einarsson, Hai Hu, Hamidreza Saffari, Hamza Zaidi, Haopeng Zhang, Harethah Abu Shairah, Harry Vuong, Hele-Andra Kuulmets, Houda Bouamor, Hwanjo Yu, Iben Nyholm Debess, İbrahim Ethem Deveci, Ikhlasul Akmal Hanif, Ikhyun Cho, Inês Calvo, Inês Vieira, Isaac Manzi, Ismail Daud, Itay Itzhak, Iuliia, Alekseenko, Ivan Belashkin, Ivan Spada, Ivan Zhelyazkov, Jacob Brinton, Jafar Isbarov, Jaka Čibej, Jan Čuhel, Jan Kocoń, Jauza Akbar Krito, Jebish Purbey, Jennifer Mickel, Jennifer Za, Jenny Kunz, Jihae Jeong, Jimena Tena Dávalos, Jinu Lee, João Magalhães, John Yi, Jongin Kim, Joseph Chataignon, Joseph Marvin Imperial, Jubeerathan Thevakumar, Judith Land, Junchen Jiang, Jungwhan Kim, Kairit Sirts, Kamesh R, Kamesh V, Kanda Patrick Tshinu, Kätriin Kukk, Kaustubh Ponkshe, Kavsar Huseynova, Ke He, Kelly Buchanan, Kengatharaiyer Sarveswaran, Kerem Zaman, Khalil Mrini, Kian Kyars, Krister Kruusmaa, Kusum Chouhan, Lainitha Krishnakumar, Laura Castro Sánchez, Laura Porrino Moscoso, Leshem Choshen, Levent Sencan, Lilja Øvrelid, Lisa Alazraki, Lovina Ehimen-Ugbede, Luheerathan Thevakumar, Luxshan Thavarasa, Mahnoor Malik, Mamadou K. Keita, Mansi Jangid, Marco De Santis, Marcos García, Marek Suppa, Mariam D'Ciofalo, Marii Ojastu, Maryam Sikander, Mausami Narayan, Maximos Skandalis, Mehak Mehak, Mehmet İlteriş Bozkurt, Melaku Bayu Workie, Menan Velayuthan, Michael Leventhal, Michał Marcińczuk, Mirna Potočnjak, Mohammadamin Shafiei, Mridul Sharma, Mrityunjaya Indoria, Muhammad Ravi Shulthan Habibi, Murat Kolić, Nada Galant, Naphat Permpredanun, Narada Maugin, Nicholas Kluge Corrêa, Nikola Ljubešić, Nirmal Thomas, Nisansa de Silva, Nisheeth Joshi, Nitish Ponkshe, Nizar Habash, Nneoma C. Udeze, Noel Thomas, Noémi Ligeti-Nagy, Nouhoum Coulibaly, Nsengiyumva Faustin, Odunayo Kareemat Buliaminu, Odunayo Ogundepo, Oghojafor Godswill Fejiro, Ogundipe Blessing Funmilola, Okechukwu God'spraise, Olanrewaju Samuel, Olaoye Deborah Oluwaseun, Olasoji Akindejoye, Olga Popova, Olga Snissarenko, Onyinye Anulika Chiemezie, Orkun Kinay, Osman Tursun, Owoeye Tobiloba Moses, Oyelade Oluwafemi Joshua, Oyesanmi Fiyinfoluwa, Pablo Gamallo, Pablo Rodríguez Fernández, Palak Arora, Pedro Valente, Peter Rupnik, Philip Oghenesuowho Ekiugbo, Pramit Sahoo, Prokopis Prokopidis, Pua Niau-Puhipau, Quadri Yahya, Rachele Mignone, Raghav Singhal, Ram Mohan Rao Kadiyala, Raphael Merx, Rapheal Afolayan, Ratnavel Rajalakshmi, Rishav Ghosh, Romina Oji, Ron Kekeha Solis, Rui Guerra, Rushikesh Zawar, Sa'ad Nasir Bashir, Saeed Alzaabi, Sahil Sandeep, Sai Pavan Batchu, SaiSandeep Kantareddy, Salsabila Zahirah Pranida, Sam Buchanan, Samuel Rutunda, Sander Land, Sarah Sulollari, Sardar Ali, Saroj Sapkota, Saulius Tautvaisas, Sayambhu Sen, Sayantani Banerjee, Sebastien Diarra, SenthilNathan. M, Sewoong Lee, Shaan Shah, Shankar Venkitachalam, Sharifa Djurabaeva, Sharon Ibejih, Shivanya Shomir Dutta, Siddhant Gupta, Silvia Paniagua Suárez, Sina Ahmadi, Sivasuthan Sukumar, Siyuan Song, Snegha A., Sokratis Sofianopoulos, Sona Elza Simon, Sonja Benčina, Sophie Gvasalia, Sphurti Kirit More, Spyros Dragazis, Stephan P. Kaufhold, Suba. S, Sultan AlRashed, Surangika Ranathunga, Taiga Someya, Taja Kuzman Pungeršek, Tal Haklay, Tasi'u Jibril, Tatsuya Aoyama, Tea Abashidze, Terenz Jomar Dela Cruz, Terra Blevins, Themistoklis Nikas, Theresa Dora Idoko, Thu Mai Do, Tilek Chubakov, Tommaso Gargiani, Uma Rathore, Uni Johannesen, Uwuma Doris Ugwu, Vallerie Alexandra Putra, Vanya Bannihatti Kumar, Varsha Jeyarajalingam, Varvara Arzt, Vasudevan Nedumpozhimana, Viktoria Ondrejova, Viktoryia Horbik, Vishnu Vardhan Reddy Kummitha, Vuk Dinić, Walelign Tewabe Sewunetie, Winston Wu, Xiaojing Zhao, Yacouba Diarra, Yaniv Nikankin, Yash Mathur, Yixi Chen, Yiyuan Li, Yolanda Xavier, Yonatan Belinkov, Yusuf Ismail Abayomi, Zaid Alyafeai, Zhengyang Shan, Zhi Rui Tam, Zilu Tang, Zuzana Nadova, Baber Abbasi, Stella Biderman, David Stap, Duygu Ataman, Fabian Schmidt, Hila Gonen, Jiayi Wang, David Ifeoluwa Adelani

To date, there exist almost no culturally-specific evaluation benchmarks for
large language models (LLMs) that cover a large number of languages and
cultures. In this paper, we present Global PIQA, a participatory commonsense
reasoning benchmark for over 100 languages, constructed by hand by 335
researchers from 65 countries around the world. The 116 language varieties in
Global PIQA cover five continents, 14 language families, and 23 writing
systems. In the non-parallel split of Global PIQA, over 50% of examples
reference local foods, customs, traditions, or other culturally-specific
elements. We find that state-of-the-art LLMs perform well on Global PIQA in
aggregate, but they exhibit weaker performance in lower-resource languages (up
to a 37% accuracy gap, despite random chance at 50%). Open models generally
perform worse than proprietary models. Global PIQA highlights that in many
languages and cultures, everyday knowledge remains an area for improvement,
alongside more widely-discussed capabilities such as complex reasoning and
expert knowledge. Beyond its uses for LLM evaluation, we hope that Global PIQA
provides a glimpse into the wide diversity of cultures in which human language
is embedded.

### 9. [RegSpeech12: A Regional Corpus of Bengali Spontaneous Speech Across Dialects](http://arxiv.org/pdf/2510.24096v1)

Authors: Md. Rezuwan Hassan, Azmol Hossain, Kanij Fatema, Rubayet Sabbir Faruque, Tanmoy Shome, Ruwad Naswan, Trina Chakraborty, Md. Foriduzzaman Zihad, Tawsif Tashwar Dipto, Nazia Tasnim, Nazmuddoha Ansary, Md. Mehedi Hasan Shawon, Ahmed Imtiaz Humayun, Md. Golam Rabiul Alam, Farig Sadeque, Asif Sushmit

The Bengali language, spoken extensively across South Asia and among
diasporic communities, exhibits considerable dialectal diversity shaped by
geography, culture, and history. Phonological and pronunciation-based
classifications broadly identify five principal dialect groups: Eastern
Bengali, Manbhumi, Rangpuri, Varendri, and Rarhi. Within Bangladesh, further
distinctions emerge through variation in vocabulary, syntax, and morphology, as
observed in regions such as Chittagong, Sylhet, Rangpur, Rajshahi, Noakhali,
and Barishal. Despite this linguistic richness, systematic research on the
computational processing of Bengali dialects remains limited. This study seeks
to document and analyze the phonetic and morphological properties of these
dialects while exploring the feasibility of building computational models
particularly Automatic Speech Recognition (ASR) systems tailored to regional
varieties. Such efforts hold potential for applications in virtual assistants
and broader language technologies, contributing to both the preservation of
dialectal diversity and the advancement of inclusive digital tools for
Bengali-speaking communities. The dataset created for this study is released
for public use.

### 10. [Squrve: A Unified and Modular Framework for Complex Real-World Text-to-SQL Tasks](http://arxiv.org/pdf/2510.24102v1)

Authors: Yihan Wang, Peiyu Liu, Runyu Chen, Jiaxing Pu, Wei Xu

Text-to-SQL technology has evolved rapidly, with diverse academic methods
achieving impressive results. However, deploying these techniques in real-world
systems remains challenging due to limited integration tools. Despite these
advances, we introduce Squrve, a unified, modular, and extensive Text-to-SQL
framework designed to bring together research advances and real-world
applications. Squrve first establishes a universal execution paradigm that
standardizes invocation interfaces, then proposes a multi-actor collaboration
mechanism based on seven abstracted effective atomic actor components.
Experiments on widely adopted benchmarks demonstrate that the collaborative
workflows consistently outperform the original individual methods, thereby
opening up a new effective avenue for tackling complex real-world queries. The
codes are available at https://github.com/Satissss/Squrve.

### Cryptography and Security

### 1. [Traceable Signatures from Lattices](http://arxiv.org/pdf/2510.24101v1)

Authors: Nam Tran, Khoa Nguyen, Dongxi Liu, Josef Pieprzyk, Willy Susilo

Traceable signatures (Kiayas et al., EUROCRYPT 2004) is an anonymous digital
signature system that extends the tracing power of the opening authority in
group signatures. There are many known constructions of traceable signatures,
but all are based on number-theoretic/pairing assumptions. For such reason,
they may not be secure in the presence of quantum computers. This work revisits
the notion of traceable signatures and presents a lattice-based construction
provably secure in the quantum random oracle model (QROM).

### 2. [Demystifying Cookie Sharing Risks in WebView-based Mobile App-in-app Ecosystems](http://arxiv.org/pdf/2510.24141v1)

Authors: Miao Zhang, Shenao Wang, Guilin Zheng, Yanjie Zhao, Haoyu Wang

Mini-programs, an emerging mobile application paradigm within super-apps,
offer a seamless and installation-free experience. However, the adoption of the
web-view component has disrupted their isolation mechanisms, exposing new
attack surfaces and vulnerabilities. In this paper, we introduce a novel
vulnerability called Cross Mini-program Cookie Sharing (CMCS), which arises
from the shared web-view environment across mini-programs. This vulnerability
allows unauthorized data exchange across mini-programs by enabling one
mini-program to access cookies set by another within the same web-view context,
violating isolation principles. As a preliminary step, we analyzed the web-view
mechanisms of four major platforms, including WeChat, AliPay, TikTok, and
Baidu, and found that all of them are affected by CMCS vulnerabilities.
Furthermore, we demonstrate the collusion attack enabled by CMCS, where
privileged mini-programs exfiltrate sensitive user data via cookies accessible
to unprivileged mini-programs. To measure the impact of collusion attacks
enabled by CMCS vulnerabilities in the wild, we developed MiCoScan, a static
analysis tool that detects mini-programs affected by CMCS vulnerabilities.
MiCoScan employs web-view context modeling to identify clusters of
mini-programs sharing the same web-view domain and cross-webview data flow
analysis to detect sensitive data transmissions to/from web-views. Using
MiCoScan, we conducted a large-scale analysis of 351,483 mini-programs,
identifying 45,448 clusters sharing web-view domains, 7,965 instances of
privileged data transmission, and 9,877 mini-programs vulnerable to collusion
attacks. Our findings highlight the widespread prevalence and significant
security risks posed by CMCS vulnerabilities, underscoring the urgent need for
improved isolation mechanisms in mini-program ecosystems.

### 3. [Cybersecurity AI Benchmark (CAIBench): A Meta-Benchmark for Evaluating Cybersecurity AI Agents](http://arxiv.org/pdf/2510.24317v1)

Authors: María Sanz-Gómez, Víctor Mayoral-Vilches, Francesco Balassone, Luis Javier Navarrete-Lozano, Cristóbal R. J. Veas Chavez, Maite del Mundo de Torres

Cybersecurity spans multiple interconnected domains, complicating the
development of meaningful, labor-relevant benchmarks. Existing benchmarks
assess isolated skills rather than integrated performance. We find that
pre-trained knowledge of cybersecurity in LLMs does not imply attack and
defense abilities, revealing a gap between knowledge and capability. To address
this limitation, we present the Cybersecurity AI Benchmark (CAIBench), a
modular meta-benchmark framework that allows evaluating LLM models and agents
across offensive and defensive cybersecurity domains, taking a step towards
meaningfully measuring their labor-relevance. CAIBench integrates five
evaluation categories, covering over 10,000 instances: Jeopardy-style CTFs,
Attack and Defense CTFs, Cyber Range exercises, knowledge benchmarks, and
privacy assessments. Key novel contributions include systematic simultaneous
offensive-defensive evaluation, robotics-focused cybersecurity challenges
(RCTF2), and privacy-preserving performance assessment (CyberPII-Bench).
Evaluation of state-of-the-art AI models reveals saturation on security
knowledge metrics (~70\% success) but substantial degradation in multi-step
adversarial (A\&D) scenarios (20-40\% success), or worse in robotic targets
(22\% success). The combination of framework scaffolding and LLM model choice
significantly impacts performance; we find that proper matches improve up to
2.6$\times$ variance in Attack and Defense CTFs. These results demonstrate a
pronounced gap between conceptual knowledge and adaptive capability,
emphasizing the need for a meta-benchmark.

### 4. [LLMLogAnalyzer: A Clustering-Based Log Analysis Chatbot using Large Language Models](http://arxiv.org/pdf/2510.24031v1)

Authors: Peng Cai, Reza Ryan, Nickson M. Karie

System logs are a cornerstone of cybersecurity, supporting proactive breach
prevention and post-incident investigations. However, analyzing vast amounts of
diverse log data remains significantly challenging, as high costs, lack of
in-house expertise, and time constraints make even basic analysis difficult for
many organizations. This study introduces LLMLogAnalyzer, a clustering-based
log analysis chatbot that leverages Large Language Models (LLMs) and Machine
Learning (ML) algorithms to simplify and streamline log analysis processes.
This innovative approach addresses key LLM limitations, including context
window constraints and poor structured text handling capabilities, enabling
more effective summarization, pattern extraction, and anomaly detection tasks.
LLMLogAnalyzer is evaluated across four distinct domain logs and various tasks.
Results demonstrate significant performance improvements over state-of-the-art
LLM-based chatbots, including ChatGPT, ChatPDF, and NotebookLM, with consistent
gains ranging from 39% to 68% across different tasks. The system also exhibits
strong robustness, achieving a 93% reduction in interquartile range (IQR) when
using ROUGE-1 scores, indicating significantly lower result variability. The
framework's effectiveness stems from its modular architecture comprising a
router, log recognizer, log parser, and search tools. This design enhances LLM
capabilities for structured text analysis while improving accuracy and
robustness, making it a valuable resource for both cybersecurity experts and
non-technical users.

### 5. [Uncovering Gaps Between RFC Updates and TCP/IP Implementations: LLM-Facilitated Differential Checks on Intermediate Representations](http://arxiv.org/pdf/2510.24408v1)

Authors: Yifan Wu, Xuewei Feng, Yuxiang Yang, Ke Xu

As the core of the Internet infrastructure, the TCP/IP protocol stack
undertakes the task of network data transmission. However, due to the
complexity of the protocol and the uncertainty of cross-layer interaction,
there are often inconsistencies between the implementation of the protocol
stack code and the RFC standard. This inconsistency may not only lead to
differences in protocol functions but also cause serious security
vulnerabilities. At present, with the continuous expansion of protocol stack
functions and the rapid iteration of RFC documents, it is increasingly
important to detect and fix these inconsistencies. With the rise of large
language models, researchers have begun to explore how to extract protocol
specifications from RFC documents through these models, including protocol
stack modeling, state machine extraction, text ambiguity analysis, and other
related content. However, existing methods rely on predefined patterns or
rule-based approaches that fail to generalize across different protocol
specifications. Automated and scalable detection of these inconsistencies
remains a significant challenge. In this study, we propose an automated
analysis framework based on LLM and differential models. By modeling the
iterative relationship of the protocol and based on the iterative update
relationship of the RFC standard, we perform incremental code function analysis
on different versions of kernel code implementations to automatically perform
code detection and vulnerability analysis. We conduct extensive evaluations to
validate the effectiveness of our framework, demonstrating its effectiveness in
identifying potential vulnerabilities caused by RFC code inconsistencies.

### 6. [Design and Optimization of Cloud Native Homomorphic Encryption Workflows for Privacy-Preserving ML Inference](http://arxiv.org/pdf/2510.24498v1)

Authors: Tejaswini Bollikonda

As machine learning (ML) models become increasingly deployed through cloud
infrastructures, the confidentiality of user data during inference poses a
significant security challenge. Homomorphic Encryption (HE) has emerged as a
compelling cryptographic technique that enables computation on encrypted data,
allowing predictions to be generated without decrypting sensitive inputs.
However, the integration of HE within large scale cloud native pipelines
remains constrained by high computational overhead, orchestration complexity,
and model compatibility issues.
  This paper presents a systematic framework for the design and optimization of
cloud native homomorphic encryption workflows that support privacy-preserving
ML inference. The proposed architecture integrates containerized HE modules
with Kubernetes-based orchestration, enabling elastic scaling and parallel
encrypted computation across distributed environments. Furthermore,
optimization strategies including ciphertext packing, polynomial modulus
adjustment, and operator fusion are employed to minimize latency and resource
consumption while preserving cryptographic integrity. Experimental results
demonstrate that the proposed system achieves up to 3.2times inference
acceleration and 40% reduction in memory utilization compared to conventional
HE pipelines. These findings illustrate a practical pathway for deploying
secure ML-as-a-Service (MLaaS) systems that guarantee data confidentiality
under zero-trust cloud conditions.

### 7. [A Novel XAI-Enhanced Quantum Adversarial Networks for Velocity Dispersion Modeling in MaNGA Galaxies](http://arxiv.org/pdf/2510.24598v1)

Authors: Sathwik Narkedimilli, N V Saran Kumar, Aswath Babu H, Manjunath K Vanahalli, Manish M, Vinija Jain, Aman Chadha

Current quantum machine learning approaches often face challenges balancing
predictive accuracy, robustness, and interpretability. To address this, we
propose a novel quantum adversarial framework that integrates a hybrid quantum
neural network (QNN) with classical deep learning layers, guided by an
evaluator model with LIME-based interpretability, and extended through quantum
GAN and self-supervised variants. In the proposed model, an adversarial
evaluator concurrently guides the QNN by computing feedback loss, thereby
optimizing both prediction accuracy and model explainability. Empirical
evaluations show that the Vanilla model achieves RMSE = 0.27, MSE = 0.071, MAE
= 0.21, and R^2 = 0.59, delivering the most consistent performance across
regression metrics compared to adversarial counterparts. These results
demonstrate the potential of combining quantum-inspired methods with classical
architectures to develop lightweight, high-performance, and interpretable
predictive models, advancing the applicability of QML beyond current
limitations.

### 8. [SafeVision: Efficient Image Guardrail with Robust Policy Adherence and Explainability](http://arxiv.org/pdf/2510.23960v1)

Authors: Peiyang Xu, Minzhou Pan, Zhaorun Chen, Shuang Yang, Chaowei Xiao, Bo Li

With the rapid proliferation of digital media, the need for efficient and
transparent safeguards against unsafe content is more critical than ever.
Traditional image guardrail models, constrained by predefined categories, often
misclassify content due to their pure feature-based learning without semantic
reasoning. Moreover, these models struggle to adapt to emerging threats,
requiring costly retraining for new threats. To address these limitations, we
introduce SafeVision, a novel image guardrail that integrates human-like
reasoning to enhance adaptability and transparency. Our approach incorporates
an effective data collection and generation framework, a policy-following
training pipeline, and a customized loss function. We also propose a diverse QA
generation and training strategy to enhance learning effectiveness. SafeVision
dynamically aligns with evolving safety policies at inference time, eliminating
the need for retraining while ensuring precise risk assessments and
explanations. Recognizing the limitations of existing unsafe image benchmarks,
which either lack granularity or cover limited risks, we introduce VisionHarm,
a high-quality dataset comprising two subsets: VisionHarm Third-party
(VisionHarm-T) and VisionHarm Comprehensive(VisionHarm-C), spanning diverse
harmful categories. Through extensive experiments, we show that SafeVision
achieves state-of-the-art performance on different benchmarks. SafeVision
outperforms GPT-4o by 8.6% on VisionHarm-T and by 15.5% on VisionHarm-C, while
being over 16x faster. SafeVision sets a comprehensive, policy-following, and
explainable image guardrail with dynamic adaptation to emerging threats.

### 9. [Covert Surveillance in Smart Devices: A SCOUR Framework Analysis of Youth Privacy Implications](http://arxiv.org/pdf/2510.24072v1)

Authors: Austin Shouli, Yulia Bobkova, Ajay Kumar Shrestha

This paper investigates how smart devices covertly capture private
conversations and discusses in more in-depth the implications of this for youth
privacy. Using a structured review guided by the PRISMA methodology, the
analysis focuses on privacy concerns, data capture methods, data storage and
sharing practices, and proposed technical mitigations. To structure and
synthesize findings, we introduce the SCOUR framework, encompassing
Surveillance mechanisms, Consent and awareness, Operational data flow, Usage
and exploitation, and Regulatory and technical safeguards. Findings reveal that
smart devices have been covertly capturing personal data, especially with smart
toys and voice-activated smart gadgets built for youth. These issues are
worsened by unclear data collection practices and insufficient transparency in
smart device applications. Balancing privacy and utility in smart devices is
crucial, as youth are becoming more aware of privacy breaches and value their
personal data more. Strategies to improve regulatory and technical safeguards
are also provided. The review identifies research gaps and suggests future
directions. The limitations of this literature review are also explained. The
findings have significant implications for policy development and the
transparency of data collection for smart devices.

### 10. [SPEAR++: Scaling Gradient Inversion via Sparsely-Used Dictionary Learning](http://arxiv.org/pdf/2510.24200v1)

Authors: Alexander Bakarsky, Dimitar I. Dimitrov, Maximilian Baader, Martin Vechev

Federated Learning has seen an increased deployment in real-world scenarios
recently, as it enables the distributed training of machine learning models
without explicit data sharing between individual clients. Yet, the introduction
of the so-called gradient inversion attacks has fundamentally challenged its
privacy-preserving properties. Unfortunately, as these attacks mostly rely on
direct data optimization without any formal guarantees, the vulnerability of
real-world systems remains in dispute and requires tedious testing for each new
federated deployment. To overcome these issues, recently the SPEAR attack was
introduced, which is based on a theoretical analysis of the gradients of linear
layers with ReLU activations. While SPEAR is an important theoretical
breakthrough, the attack's practicality was severely limited by its exponential
runtime in the batch size b. In this work, we fill this gap by applying
State-of-the-Art techniques from Sparsely-Used Dictionary Learning to make the
problem of gradient inversion on linear layers with ReLU activations tractable.
Our experiments demonstrate that our new attack, SPEAR++, retains all desirable
properties of SPEAR, such as robustness to DP noise and FedAvg aggregation,
while being applicable to 10x bigger batch sizes.

### Computer Vision and Pattern Recognition

### 1. [Reasoning Visual Language Model for Chest X-Ray Analysis](http://arxiv.org/pdf/2510.23968v1)

Authors: Andriy Myronenko, Dong Yang, Baris Turkbey, Mariam Aboian, Sena Azamat, Esra Akcicek, Hongxu Yin, Pavlo Molchanov, Marc Edgar, Yufan He, Pengfei Guo, Yucheng Tang, Daguang Xu

Vision-language models (VLMs) have shown strong promise for medical image
analysis, but most remain opaque, offering predictions without the transparent,
stepwise reasoning clinicians rely on. We present a framework that brings
chain-of-thought (CoT) reasoning to chest X-ray interpretation. Inspired by
reasoning-first training paradigms, our approach is designed to learn how
experts reason, not just what they conclude, by aligning intermediate steps
with observable image evidence and radiology workflow. Beyond accuracy, the
explicit reasoning traces support clinical auditability: they reveal why a
conclusion was reached, which alternatives were considered, and where
uncertainty remains, enabling quality assurance, error analysis, and safer
human-AI collaboration.
  Our model couples high-fidelity visual encoding with a two-stage training
recipe: a reasoning-style supervised fine-tuning (SFT) followed by
reinforcement learning (RL) that uses verifiable rewards over a list of X-ray
abnormalities. The model outputs reasoning that mirrors radiologists systematic
thought process, uncertainty, and differential diagnosis. In
out-of-distribution evaluation, the approach achieves competitive multi-label
classification while improving interpretability. In a reader study with expert
radiologists, full reasoning traces increased confidence, supported error
auditing, and reduced time to finalize reports. We release code and the model
NV-Reason-CXR-3B to support community progress toward trustworthy, explainable
AI in chest radiography and other medical imaging tasks where reasoning quality
is as critical as prediction quality.

### 2. [Efficient Cost-and-Quality Controllable Arbitrary-scale Super-resolution with Fourier Constraints](http://arxiv.org/pdf/2510.23978v1)

Authors: Kazutoshi Akita, Norimichi Ukita

Cost-and-Quality (CQ) controllability in arbitrary-scale super-resolution is
crucial. Existing methods predict Fourier components one by one using a
recurrent neural network. However, this approach leads to performance
degradation and inefficiency due to independent prediction. This paper proposes
predicting multiple components jointly to improve both quality and efficiency.

### 3. [TeleEgo: Benchmarking Egocentric AI Assistants in the Wild](http://arxiv.org/pdf/2510.23981v1)

Authors: Jiaqi Yan, Ruilong Ren, Jingren Liu, Shuning Xu, Ling Wang, Yiheng Wang, Yun Wang, Long Zhang, Xiangyu Chen, Changzhi Sun, Jixiang Luo, Dell Zhang, Hao Sun, Chi Zhang, Xuelong Li

Egocentric AI assistants in real-world settings must process multi-modal
inputs (video, audio, text), respond in real time, and retain evolving
long-term memory. However, existing benchmarks typically evaluate these
abilities in isolation, lack realistic streaming scenarios, or support only
short-term tasks. We introduce \textbf{TeleEgo}, a long-duration, streaming,
omni-modal benchmark for evaluating egocentric AI assistants in realistic daily
contexts. The dataset features over 14 hours per participant of synchronized
egocentric video, audio, and text across four domains: work \& study, lifestyle
\& routines, social activities, and outings \& culture. All data is aligned on
a unified global timeline and includes high-quality visual narrations and
speech transcripts, curated through human refinement.TeleEgo defines 12
diagnostic subtasks across three core capabilities: Memory (recalling past
events), Understanding (interpreting the current moment), and Cross-Memory
Reasoning (linking distant events). It contains 3,291 human-verified QA items
spanning multiple question formats (single-choice, binary, multi-choice, and
open-ended), evaluated strictly in a streaming setting. We propose two key
metrics -- Real-Time Accuracy and Memory Persistence Time -- to jointly assess
correctness, temporal responsiveness, and long-term retention. TeleEgo provides
a realistic and comprehensive evaluation to advance the development of
practical AI assistants.

### 4. [AdvBlur: Adversarial Blur for Robust Diabetic Retinopathy Classification and Cross-Domain Generalization](http://arxiv.org/pdf/2510.24000v1)

Authors: Heethanjan Kanagalingam, Thenukan Pathmanathan, Mokeeshan Vathanakumar, Tharmakulasingam Mukunthan

Diabetic retinopathy (DR) is a leading cause of vision loss worldwide, yet
early and accurate detection can significantly improve treatment outcomes.
While numerous Deep learning (DL) models have been developed to predict DR from
fundus images, many face challenges in maintaining robustness due to
distributional variations caused by differences in acquisition devices,
demographic disparities, and imaging conditions. This paper addresses this
critical limitation by proposing a novel DR classification approach, a method
called AdvBlur. Our method integrates adversarial blurred images into the
dataset and employs a dual-loss function framework to address domain
generalization. This approach effectively mitigates the impact of unseen
distributional variations, as evidenced by comprehensive evaluations across
multiple datasets. Additionally, we conduct extensive experiments to explore
the effects of factors such as camera type, low-quality images, and dataset
size. Furthermore, we perform ablation studies on blurred images and the loss
function to ensure the validity of our choices. The experimental results
demonstrate the effectiveness of our proposed method, achieving competitive
performance compared to state-of-the-art domain generalization DR models on
unseen external datasets.

### 5. [Towards the Automatic Segmentation, Modeling and Meshing of the Aortic Vessel Tree from Multicenter Acquisitions: An Overview of the SEG.A. 2023 Segmentation of the Aorta Challenge](http://arxiv.org/pdf/2510.24009v1)

Authors: Yuan Jin, Antonio Pepe, Gian Marco Melito, Yuxuan Chen, Yunsu Byeon, Hyeseong Kim, Kyungwon Kim, Doohyun Park, Euijoon Choi, Dosik Hwang, Andriy Myronenko, Dong Yang, Yufan He, Daguang Xu, Ayman El-Ghotni, Mohamed Nabil, Hossam El-Kady, Ahmed Ayyad, Amr Nasr, Marek Wodzinski, Henning Müller, Hyeongyu Kim, Yejee Shin, Abbas Khan, Muhammad Asad, Alexander Zolotarev, Caroline Roney, Anthony Mathur, Martin Benning, Gregory Slabaugh, Theodoros Panagiotis Vagenas, Konstantinos Georgas, George K. Matsopoulos, Jihan Zhang, Zhen Zhang, Liqin Huang, Christian Mayer, Heinrich Mächler, Jan Egger

The automated analysis of the aortic vessel tree (AVT) from computed
tomography angiography (CTA) holds immense clinical potential, but its
development has been impeded by a lack of shared, high-quality data. We
launched the SEG.A. challenge to catalyze progress in this field by introducing
a large, publicly available, multi-institutional dataset for AVT segmentation.
The challenge benchmarked automated algorithms on a hidden test set, with
subsequent optional tasks in surface meshing for computational simulations. Our
findings reveal a clear convergence on deep learning methodologies, with 3D
U-Net architectures dominating the top submissions. A key result was that an
ensemble of the highest-ranking algorithms significantly outperformed
individual models, highlighting the benefits of model fusion. Performance was
strongly linked to algorithmic design, particularly the use of customized
post-processing steps, and the characteristics of the training data. This
initiative not only establishes a new performance benchmark but also provides a
lasting resource to drive future innovation toward robust, clinically
translatable tools.

### 6. [AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts](http://arxiv.org/pdf/2510.24034v1)

Authors: Yufan Liu, Wanqian Zhang, Huashan Chen, Lin Wang, Xiaojun Jia, Zheng Lin, Weiping Wang

Despite rapid advancements in text-to-image (T2I) models, their safety
mechanisms are vulnerable to adversarial prompts, which maliciously generate
unsafe images. Current red-teaming methods for proactively assessing such
vulnerabilities usually require white-box access to T2I models, and rely on
inefficient per-prompt optimization, as well as inevitably generate
semantically meaningless prompts easily blocked by filters. In this paper, we
propose APT (AutoPrompT), a black-box framework that leverages large language
models (LLMs) to automatically generate human-readable adversarial suffixes for
benign prompts. We first introduce an alternating optimization-finetuning
pipeline between adversarial suffix optimization and fine-tuning the LLM
utilizing the optimized suffix. Furthermore, we integrates a dual-evasion
strategy in optimization phase, enabling the bypass of both perplexity-based
filter and blacklist word filter: (1) we constrain the LLM generating
human-readable prompts through an auxiliary LLM perplexity scoring, which
starkly contrasts with prior token-level gibberish, and (2) we also introduce
banned-token penalties to suppress the explicit generation of banned-tokens in
blacklist. Extensive experiments demonstrate the excellent red-teaming
performance of our human-readable, filter-resistant adversarial prompts, as
well as superior zero-shot transferability which enables instant adaptation to
unseen prompts and exposes critical vulnerabilities even in commercial APIs
(e.g., Leonardo.Ai.).

### 7. [Enhancing CLIP Robustness via Cross-Modality Alignment](http://arxiv.org/pdf/2510.24038v1)

Authors: Xingyu Zhu, Beier Zhu, Shuo Wang, Kesen Zhao, Hanwang Zhang

Vision-language models (VLMs) such as CLIP demonstrate strong generalization
in zero-shot classification but remain highly vulnerable to adversarial
perturbations. Existing methods primarily focus on adversarial fine-tuning or
prompt optimization; they often overlook the gaps in CLIP's encoded features,
which is shown as the text and image features lie far apart from each other.
This misalignment is significantly amplified under adversarial perturbations,
leading to severe degradation in classification performance. To address this
problem, we propose Cross-modality Alignment, dubbed COLA, an optimal
transport-based framework that explicitly addresses adversarial misalignment by
restoring both global image-text alignment and local structural consistency in
the feature space. (1) COLA first projects adversarial image embeddings onto a
subspace spanned by class text features, effectively filtering out non-semantic
distortions while preserving discriminative information. (2) It then models
images and texts as discrete distributions over multiple augmented views and
refines their alignment via OT, with the subspace projection seamlessly
integrated into the cost computation. This design ensures stable cross-modal
alignment even under adversarial conditions. COLA is training-free and
compatible with existing fine-tuned models. Extensive evaluations across 14
zero-shot classification benchmarks demonstrate the effectiveness of COLA,
especially with an average improvement of 6.7% on ImageNet and its variants
under PGD adversarial attacks, while maintaining high accuracy on clean
samples.

### 8. [Beyond Objects: Contextual Synthetic Data Generation for Fine-Grained Classification](http://arxiv.org/pdf/2510.24078v1)

Authors: William Yang, Xindi Wu, Zhiwei Deng, Esin Tureci, Olga Russakovsky

Text-to-image (T2I) models are increasingly used for synthetic dataset
generation, but generating effective synthetic training data for classification
remains challenging. Fine-tuning a T2I model with a few real examples can help
improve the quality of synthetic training data; however, it may also cause
overfitting and reduce diversity in the generated samples. We propose a
fine-tuning strategy BOB (BeyondOBjects) to mitigate these concerns for
fine-grained classification. Given a small set of real examples, we first
extract class-agnostic attributes such as scene background and object pose. We
then explicitly condition on these attributes during fine-tuning of the T2I
model and marginalize them out during generation. This design mitigates
overfitting, preserves the T2I model's generative prior, reduces estimation
errors, and further minimizes unintended inter-class associations. Extensive
experiments across multiple T2I models, backbones, and datasets show that our
method achieves state-of-the-art performance in low-shot fine-grained
classification when augmented with synthetic data. Concretely, BOB outperforms
DataDream by 7.4% on the Aircraft dataset (from 50.0% to 57.4% when fine-tuning
a CLIP classifier with five real images augmented with 100 synthetic images).
In three of the four benchmarks, fine-tuning downstream models with 5 real
images augmented with BOB achieves better performance than fine-tuning with 10
real images. Collectively, BOB outperforms prior art in 18 of 24 experimental
settings, with 2+% accuracy improvements in 14 of these settings.

### 9. [OmniText: A Training-Free Generalist for Controllable Text-Image Manipulation](http://arxiv.org/pdf/2510.24093v1)

Authors: Agus Gunawan, Samuel Teodoro, Yun Chen, Soo Ye Kim, Jihyong Oh, Munchurl Kim

Recent advancements in diffusion-based text synthesis have demonstrated
significant performance in inserting and editing text within images via
inpainting. However, despite the potential of text inpainting methods, three
key limitations hinder their applicability to broader Text Image Manipulation
(TIM) tasks: (i) the inability to remove text, (ii) the lack of control over
the style of rendered text, and (iii) a tendency to generate duplicated
letters. To address these challenges, we propose OmniText, a training-free
generalist capable of performing a wide range of TIM tasks. Specifically, we
investigate two key properties of cross- and self-attention mechanisms to
enable text removal and to provide control over both text styles and content.
Our findings reveal that text removal can be achieved by applying
self-attention inversion, which mitigates the model's tendency to focus on
surrounding text, thus reducing text hallucinations. Additionally, we
redistribute cross-attention, as increasing the probability of certain text
tokens reduces text hallucination. For controllable inpainting, we introduce
novel loss functions in a latent optimization framework: a cross-attention
content loss to improve text rendering accuracy and a self-attention style loss
to facilitate style customization. Furthermore, we present OmniText-Bench, a
benchmark dataset for evaluating diverse TIM tasks. It includes input images,
target text with masks, and style references, covering diverse applications
such as text removal, rescaling, repositioning, and insertion and editing with
various styles. Our OmniText framework is the first generalist method capable
of performing diverse TIM tasks. It achieves state-of-the-art performance
across multiple tasks and metrics compared to other text inpainting methods and
is comparable with specialist methods.

### 10. [UHKD: A Unified Framework for Heterogeneous Knowledge Distillation via Frequency-Domain Representations](http://arxiv.org/pdf/2510.24116v1)

Authors: Fengming Yu, Haiwei Pan, Kejia Zhang, Jian Guan, Haiying Jiang

Knowledge distillation (KD) is an effective model compression technique that
transfers knowledge from a high-performance teacher to a lightweight student,
reducing cost while maintaining accuracy. In visual applications, where
large-scale image models are widely used, KD enables efficient deployment.
However, architectural diversity introduces semantic discrepancies that hinder
the use of intermediate representations. Most existing KD methods are designed
for homogeneous models and degrade in heterogeneous scenarios, especially when
intermediate features are involved. Prior studies mainly focus on the logits
space, making limited use of the semantic information in intermediate layers.
To address this limitation, Unified Heterogeneous Knowledge Distillation (UHKD)
is proposed as a framework that leverages intermediate features in the
frequency domain for cross-architecture transfer. Fourier transform is applied
to capture global feature information, alleviating representational
discrepancies between heterogeneous teacher-student pairs. A Feature
Transformation Module (FTM) produces compact frequency-domain representations
of teacher features, while a learnable Feature Alignment Module (FAM) projects
student features and aligns them via multi-level matching. Training is guided
by a joint objective combining mean squared error on intermediate features with
Kullback-Leibler divergence on logits. Experiments on CIFAR-100 and ImageNet-1K
demonstrate gains of 5.59% and 0.83% over the latest method, highlighting UHKD
as an effective approach for unifying heterogeneous representations and
enabling efficient utilization of visual knowledge

### Computers and Society

### 1. [Politically Speaking: LLMs on Changing International Affairs](http://arxiv.org/pdf/2510.24582v1)

Authors: Xuenan Cao, Wai Kei Chung, Ye Zhao, Lidia Mengyuan Zhou

Ask your chatbot to impersonate an expert from Russia and an expert from US
and query it on Chinese politics. How might the outputs differ? Or, to prepare
ourselves for the worse, how might they converge? Scholars have raised concerns
LLM based applications can homogenize cultures and flatten perspectives. But
exactly how much does LLM generated outputs converge despite explicit different
role assignment? This study provides empirical evidence to the above question.
The critique centres on pretrained models regurgitating ossified political
jargons used in the Western world when speaking about China, Iran, Russian, and
US politics, despite changes in these countries happening daily or hourly. The
experiments combine role-prompting and similarity metrics. The results show
that AI generated discourses from four models about Iran and China are the most
homogeneous and unchanging across all four models, including OpenAI GPT, Google
Gemini, Anthropic Claude, and DeepSeek, despite the prompted perspective change
and the actual changes in real life. This study does not engage with history,
politics, or literature as traditional disciplinary approaches would; instead,
it takes cues from international and area studies and offers insight on the
future trajectory of shifting political discourse in a digital space
increasingly cannibalised by AI.

### 2. [Rewarding Engagement and Personalization in Popularity-Based Rankings Amplifies Extremism and Polarization](http://arxiv.org/pdf/2510.24354v1)

Authors: Jacopo D'Ignazi, Andreas Kaltenbrunner, Gaël Le Mens, Fabrizio Germano, Vicenç Gómez

Despite extensive research, the mechanisms through which online platforms
shape extremism and polarization remain poorly understood. We identify and test
a mechanism, grounded in empirical evidence, that explains how ranking
algorithms can amplify both phenomena. This mechanism is based on
well-documented assumptions: (i) users exhibit position bias and tend to prefer
items displayed higher in the ranking, (ii) users prefer like-minded content,
(iii) users with more extreme views are more likely to engage actively, and
(iv) ranking algorithms are popularity-based, assigning higher positions to
items that attract more clicks. Under these conditions, when platforms
additionally reward \emph{active} engagement and implement \emph{personalized}
rankings, users are inevitably driven toward more extremist and polarized news
consumption. We formalize this mechanism in a dynamical model, which we
evaluate by means of simulations and interactive experiments with hundreds of
human participants, where the rankings are updated dynamically in response to
user activity.

### 3. [AI for a Planet Under Pressure](http://arxiv.org/pdf/2510.24373v1)

Authors: Victor Galaz, Maria Schewenius, Jonathan F. Donges, Ingo Fetzer, Erik Zhivkoplias, Wolfram Barfuss, Louis Delannoy, Lan Wang-Erlandsson, Maximilian Gelbrecht, Jobst Heitzig, Jonas Hentati-Sundberg, Christopher Kennedy, Nielja Knecht, Romi Lotcheris, Miguel Mahecha, Andrew Merrie, David Montero, Timon McPhearson, Ahmed Mustafa, Magnus Nyström, Drew Purves, Juan C. Rocha, Masahiro Ryo, Claudia van der Salm, Samuel T. Segun, Anna B. Stephenson, Elizabeth Tellman, Felipe Tobar, Alice Vadrot

Artificial intelligence (AI) is already driving scientific breakthroughs in a
variety of research fields, ranging from the life sciences to mathematics. This
raises a critical question: can AI be applied both responsibly and effectively
to address complex and interconnected sustainability challenges? This report is
the result of a collaboration between the Stockholm resilience Centre
(Stockholm University), the Potsdam Institute for Climate Impact Research
(PIK), and Google DeepMind. Our work explores the potential and limitations of
using AI as a research method to help tackle eight broad sustainability
challenges. The results build on iterated expert dialogues and assessments, a
systematic AI-supported literature overview including over 8,500 academic
publications, and expert deep-dives into eight specific issue areas. The report
also includes recommendations to sustainability scientists, research funders,
the private sector, and philanthropies.

### 4. [Covert Surveillance in Smart Devices: A SCOUR Framework Analysis of Youth Privacy Implications](http://arxiv.org/pdf/2510.24072v1)

Authors: Austin Shouli, Yulia Bobkova, Ajay Kumar Shrestha

This paper investigates how smart devices covertly capture private
conversations and discusses in more in-depth the implications of this for youth
privacy. Using a structured review guided by the PRISMA methodology, the
analysis focuses on privacy concerns, data capture methods, data storage and
sharing practices, and proposed technical mitigations. To structure and
synthesize findings, we introduce the SCOUR framework, encompassing
Surveillance mechanisms, Consent and awareness, Operational data flow, Usage
and exploitation, and Regulatory and technical safeguards. Findings reveal that
smart devices have been covertly capturing personal data, especially with smart
toys and voice-activated smart gadgets built for youth. These issues are
worsened by unclear data collection practices and insufficient transparency in
smart device applications. Balancing privacy and utility in smart devices is
crucial, as youth are becoming more aware of privacy breaches and value their
personal data more. Strategies to improve regulatory and technical safeguards
are also provided. The review identifies research gaps and suggests future
directions. The limitations of this literature review are also explained. The
findings have significant implications for policy development and the
transparency of data collection for smart devices.

### 5. [Policy Cards: Machine-Readable Runtime Governance for Autonomous AI Agents](http://arxiv.org/pdf/2510.24383v1)

Authors: Juraj Mavračić

Policy Cards are introduced as a machine-readable, deployment-layer standard
for expressing operational, regulatory, and ethical constraints for AI agents.
The Policy Card sits with the agent and enables it to follow required
constraints at runtime. It tells the agent what it must and must not do. As
such, it becomes an integral part of the deployed agent. Policy Cards extend
existing transparency artifacts such as Model, Data, and System Cards by
defining a normative layer that encodes allow/deny rules, obligations,
evidentiary requirements, and crosswalk mappings to assurance frameworks
including NIST AI RMF, ISO/IEC 42001, and the EU AI Act. Each Policy Card can
be validated automatically, version-controlled, and linked to runtime
enforcement or continuous-audit pipelines. The framework enables verifiable
compliance for autonomous agents, forming a foundation for distributed
assurance in multi-agent ecosystems. Policy Cards provide a practical mechanism
for integrating high-level governance with hands-on engineering practice and
enabling accountable autonomy at scale.

### 6. [Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](http://arxiv.org/pdf/2510.24438v1)

Authors: Abdullah Mushtaq, Rafay Naeem, Ezieddin Elmahjub, Ibrahim Ghaznavi, Shawqi Al-Maliki, Mohamed Abdallah, Ala Al-Fuqaha, Junaid Qadir

Large language models are increasingly used for Islamic guidance, but risk
misquoting texts, misapplying jurisprudence, or producing culturally
inconsistent responses. We pilot an evaluation of GPT-4o, Ansari AI, and Fanar
on prompts from authentic Islamic blogs. Our dual-agent framework uses a
quantitative agent for citation verification and six-dimensional scoring (e.g.,
Structure, Islamic Consistency, Citations) and a qualitative agent for
five-dimensional side-by-side comparison (e.g., Tone, Depth, Originality).
GPT-4o scored highest in Islamic Accuracy (3.93) and Citation (3.38), Ansari AI
followed (3.68, 3.32), and Fanar lagged (2.76, 1.82). Despite relatively strong
performance, models still fall short in reliably producing accurate Islamic
content and citations -- a paramount requirement in faith-sensitive writing.
GPT-4o had the highest mean quantitative score (3.90/5), while Ansari AI led
qualitative pairwise wins (116/200). Fanar, though trailing, introduces
innovations for Islamic and Arabic contexts. This study underscores the need
for community-driven benchmarks centering Muslim perspectives, offering an
early step toward more reliable AI in Islamic knowledge and other high-stakes
domains such as medicine, law, and journalism.

### 7. [Law in Silico: Simulating Legal Society with LLM-Based Agents](http://arxiv.org/pdf/2510.24442v1)

Authors: Yiding Wang, Yuxuan Chen, Fanxu Meng, Xifan Chen, Xiaolei Yang, Muhan Zhang

Since real-world legal experiments are often costly or infeasible, simulating
legal societies with Artificial Intelligence (AI) systems provides an effective
alternative for verifying and developing legal theory, as well as supporting
legal administration. Large Language Models (LLMs), with their world knowledge
and role-playing capabilities, are strong candidates to serve as the foundation
for legal society simulation. However, the application of LLMs to simulate
legal systems remains underexplored. In this work, we introduce Law in Silico,
an LLM-based agent framework for simulating legal scenarios with individual
decision-making and institutional mechanisms of legislation, adjudication, and
enforcement. Our experiments, which compare simulated crime rates with
real-world data, demonstrate that LLM-based agents can largely reproduce
macro-level crime trends and provide insights that align with real-world
observations. At the same time, micro-level simulations reveal that a
well-functioning, transparent, and adaptive legal system offers better
protection of the rights of vulnerable individuals.

### Databases

### 1. [Evaluating Joinable Column Discovery Approaches for Context-Aware Search](http://arxiv.org/pdf/2510.24599v1)

Authors: Harsha Kokel, Aamod Khatiwada, Tejaswini Pedapati, Haritha Ananthakrishnan, Oktie Hassanzadeh, Horst Samulowitz, Kavitha Srinivas

Joinable Column Discovery is a critical challenge in automating enterprise
data analysis. While existing approaches focus on syntactic overlap and
semantic similarity, there remains limited understanding of which methods
perform best for different data characteristics and how multiple criteria
influence discovery effectiveness. We present a comprehensive experimental
evaluation of joinable column discovery methods across diverse scenarios. Our
study compares syntactic and semantic techniques on seven benchmarks covering
relational databases and data lakes. We analyze six key criteria -- unique
values, intersection size, join size, reverse join size, value semantics, and
metadata semantics -- and examine how combining them through ensemble ranking
affects performance. Our analysis reveals differences in method behavior across
data contexts and highlights the benefits of integrating multiple criteria for
robust join discovery. We provide empirical evidence on when each criterion
matters, compare pre-trained embedding models for semantic joins, and offer
practical guidelines for selecting suitable methods based on dataset
characteristics. Our findings show that metadata and value semantics are
crucial for data lakes, size-based criteria play a stronger role in relational
databases, and ensemble approaches consistently outperform single-criterion
methods.

### 2. [Odyssey: An End-to-End System for Pareto-Optimal Serverless Query Processing](http://arxiv.org/pdf/2510.24307v1)

Authors: Shyam Jesalpura, Shengda Zhu, Amir Shaikhha, Antonio Barbalace, Boris Grot

Running data analytics queries on serverless (FaaS) workers has been shown to
be cost- and performance-efficient for a variety of real-world scenarios,
including intermittent query arrival patterns, sudden load spikes and
management challenges that afflict managed VM clusters. Alas, existing
serverless data analytics works focus primarily on the serverless execution
engine and assume the existence of a "good" query execution plan or rely on
user guidance to construct such a plan. Meanwhile, even simple analytics
queries on serverless have a huge space of possible plans, with vast
differences in both performance and cost among plans.
  This paper introduces Odyssey, an end-to-end serverless-native data analytics
pipeline that integrates a query planner, cost model and execution engine.
Odyssey automatically generates and evaluates serverless query plans, utilizing
state space pruning heuristics and a novel search algorithm to identify
Pareto-optimal plans that balance cost and performance with low latency even
for complex queries. Our evaluations demonstrate that Odyssey accurately
predicts both monetary cost and latency, and consistently outperforms AWS
Athena on cost and/or latency.

### Distributed, Parallel, and Cluster Computing

### 1. [Towards Exascale Computing for Astrophysical Simulation Leveraging the Leonardo EuroHPC System](http://arxiv.org/pdf/2510.24175v1)

Authors: Nitin Shukla, Alessandro Romeo, Caterina Caravita, Michael Redenti, Radim Vavrik, Lubomir Riha, Andrea Mignone, Marco Rossazza, Stefano Truzzi, Luca Tornatore, Antonio Ragagnin, Tiago Castro, Geray S. Karademir, Klaus Dolag, Pranab J. Deka, Fabio Bacchini, Rostislav-Paul Wilhelm, Daniele Gregori, Elisabetta Boella

Developing and redesigning astrophysical, cosmological, and space plasma
numerical codes for existing and next-generation accelerators is critical for
enabling large-scale simulations. To address these challenges, the SPACE Center
of Excellence (SPACE-CoE) fosters collaboration between scientists, code
developers, and high-performance computing experts to optimize applications for
the exascale era. This paper presents our strategy and initial results on the
Leonardo system at CINECA for three flagship codes, namely gPLUTO, OpenGadget3
and iPIC3D, using profiling tools to analyze performance on single and multiple
nodes. Preliminary tests show all three codes scale efficiently, reaching 80%
scalability up to 1,024 GPUs.

### 2. [CoMPSeT: A Framework for Comparing Multiparty Session Types](http://arxiv.org/pdf/2510.24205v1)

Authors: Telmo Ribeiro, José Proença, Mário Florido

Concurrent systems are often complex and difficult to design. Choreographic
languages, such as Multiparty Session Types (MPST), allow the description of
global protocols of interactions by capturing valid patterns of interactions
between participants. Many variations of MPST exist, each one with its rather
specific features and idiosyncrasies. Here we propose a tool (CoMPSeT) that
provides clearer insights over different features in existing MPST. We select a
representative set of MPST examples and provide mechanisms to combine different
features and to animate and compare the semantics of concrete examples. CoMPSeT
is open-source, compiled into JavaScript, and can be directly executed from any
browser, becoming useful both for researchers who want to better understand the
landscape of MPST and for teachers who want to explain global choreographies.

### 3. [A GPU-based Compressible Combustion Solver for Applications Exhibiting Disparate Space and Time Scales](http://arxiv.org/pdf/2510.23993v1)

Authors: Anthony Carreon, Jagmohan Singh, Shivank Sharma, Shuzhi Zhang, Venkat Raman

High-speed chemically active flows present significant computational
challenges due to their disparate space and time scales, where stiff chemistry
often dominates simulation time. While modern supercomputing scientific codes
achieve exascale performance by leveraging graphics processing units (GPUs),
existing GPU-based compressible combustion solvers face critical limitations in
memory management, load balancing, and handling the highly localized nature of
chemical reactions. To this end, we present a high-performance compressible
reacting flow solver built on the AMReX framework and optimized for multi-GPU
settings. Our approach addresses three GPU performance bottlenecks: memory
access patterns through column-major storage optimization, computational
workload variability via a bulk-sparse integration strategy for chemical
kinetics, and multi-GPU load distribution for adaptive mesh refinement
applications. The solver adapts existing matrix-based chemical kinetics
formulations to multigrid contexts. Using representative combustion
applications including hydrogen-air detonations and jet in supersonic crossflow
configurations, we demonstrate $2-5\times$ performance improvements over
initial GPU implementations with near-ideal weak scaling across $1-96$ NVIDIA
H100 GPUs. Roofline analysis reveals substantial improvements in arithmetic
intensity for both convection ($\sim 10 \times$) and chemistry ($\sim 4
\times$) routines, confirming efficient utilization of GPU memory bandwidth and
computational resources.

### 4. [Distributed Stochastic Momentum Tracking with Local Updates: Achieving Optimal Communication and Iteration Complexities](http://arxiv.org/pdf/2510.24155v1)

Authors: Kun Huang, Shi Pu

We propose Local Momentum Tracking (LMT), a novel distributed stochastic
gradient method for solving distributed optimization problems over networks. To
reduce communication overhead, LMT enables each agent to perform multiple local
updates between consecutive communication rounds. Specifically, LMT integrates
local updates with the momentum tracking strategy and the Loopless Chebyshev
Acceleration (LCA) technique. We demonstrate that LMT achieves linear speedup
with respect to the number of local updates as well as the number of agents for
minimizing smooth objective functions. Moreover, with sufficiently many local
updates ($Q\geq Q^*$), LMT attains the optimal communication complexity. For a
moderate number of local updates ($Q\in[1,Q^*]$), it achieves the optimal
iteration complexity. To our knowledge, LMT is the first method that enjoys
such properties.

### 5. [Fault-Tolerant Multiparty Session Types with Global Escape Loops](http://arxiv.org/pdf/2510.24203v1)

Authors: Lukas Bartl, Julian Linne, Kirstin Peters

Multiparty session types are designed to abstractly capture the structure of
communication protocols and verify behavioural properties. One important such
property is progress, i.e., the absence of deadlock. Distributed algorithms
often resemble multiparty communication protocols. But proving their
properties, in particular termination that is closely related to progress, can
be elaborate. Since distributed algorithms are often designed to cope with
faults, a first step towards using session types to verify distributed
algorithms is to integrate fault-tolerance.
  We extend FTMPST (a version of fault-tolerant multiparty session types with
failure patterns to represent system requirements for system failures such as
unreliable communication and process crashes) by a novel, fault-tolerant loop
construct with global escapes that does not require global coordination. Each
process runs its own local version of the loop. If a process finds a solution
to the considered problem, it does not only terminate its own loop but also
informs the other participants via exit-messages. Upon receiving an
exit-message, a process immediately terminates its algorithm. To increase
efficiency and model standard fault-tolerant algorithms, these messages are
non-blocking, i.e., a process may continue until a possibly delayed
exit-message is received. To illustrate our approach, we analyse a variant of
the well-known rotating coordinator algorithm by Chandra and Toueg.

### 6. [Odyssey: An End-to-End System for Pareto-Optimal Serverless Query Processing](http://arxiv.org/pdf/2510.24307v1)

Authors: Shyam Jesalpura, Shengda Zhu, Amir Shaikhha, Antonio Barbalace, Boris Grot

Running data analytics queries on serverless (FaaS) workers has been shown to
be cost- and performance-efficient for a variety of real-world scenarios,
including intermittent query arrival patterns, sudden load spikes and
management challenges that afflict managed VM clusters. Alas, existing
serverless data analytics works focus primarily on the serverless execution
engine and assume the existence of a "good" query execution plan or rely on
user guidance to construct such a plan. Meanwhile, even simple analytics
queries on serverless have a huge space of possible plans, with vast
differences in both performance and cost among plans.
  This paper introduces Odyssey, an end-to-end serverless-native data analytics
pipeline that integrates a query planner, cost model and execution engine.
Odyssey automatically generates and evaluates serverless query plans, utilizing
state space pruning heuristics and a novel search algorithm to identify
Pareto-optimal plans that balance cost and performance with low latency even
for complex queries. Our evaluations demonstrate that Odyssey accurately
predicts both monetary cost and latency, and consistently outperforms AWS
Athena on cost and/or latency.

### 7. [ARIMA_PLUS: Large-scale, Accurate, Automatic and Interpretable In-Database Time Series Forecasting and Anomaly Detection in Google BigQuery](http://arxiv.org/pdf/2510.24452v1)

Authors: Xi Cheng, Weijie Shen, Haoming Chen, Chaoyi Shen, Jean Ortega, Jiashang Liu, Steve Thomas, Honglin Zheng, Haoyun Wu, Yuxiang Li, Casey Lichtendahl, Jenny Ortiz, Gang Liu, Haiyang Qi, Omid Fatemieh, Chris Fry, Jing Jing Long

Time series forecasting and anomaly detection are common tasks for
practitioners in industries such as retail, manufacturing, advertising and
energy. Two unique challenges stand out: (1) efficiently and accurately
forecasting time series or detecting anomalies in large volumes automatically;
and (2) ensuring interpretability of results to effectively incorporate
business insights. We present ARIMA_PLUS, a novel framework to overcome these
two challenges by a unique combination of (a) accurate and interpretable time
series models and (b) scalable and fully managed system infrastructure. The
model has a sequential and modular structure to handle different components of
the time series, including holiday effects, seasonality, trend, and anomalies,
which enables high interpretability of the results. Novel enhancements are made
to each module, and a unified framework is established to address both
forecasting and anomaly detection tasks simultaneously. In terms of accuracy,
its comprehensive benchmark on the 42 public datasets in the Monash forecasting
repository shows superior performance over not only well-established
statistical alternatives (such as ETS, ARIMA, TBATS, Prophet) but also newer
neural network models (such as DeepAR, N-BEATS, PatchTST, TimeMixer). In terms
of infrastructure, it is directly built into the query engine of BigQuery in
Google Cloud. It uses a simple SQL interface and automates tedious
technicalities such as data cleaning and model selection. It automatically
scales with managed cloud computational and storage resources, making it
possible to forecast 100 million time series using only 1.5 hours with a
throughput of more than 18000 time series per second. In terms of
interpretability, we present several case studies to demonstrate time series
insights it generates and customizability it offers.

### 8. [Exascale In-situ visualization for Astronomy & Cosmology](http://arxiv.org/pdf/2510.24545v1)

Authors: Nicola Tuccari, Eva Sciacca, Yolanda Becerra, Enric Sosa Cintero, Emiliano Tramontana

Modern simulations and observations in Astronomy & Cosmology (A&C) produce
massively large data volumes, posing significant challenges for storage, access
and data analysis. A long-standing bottleneck in high-performance computing,
especially now in the exascale era, has been the requirement to write these
large datasets to disks, which limits the performance. A promising solution to
this challenge is in-situ processing, where analysis and visualization are
performed concurrently with the simulation itself, bypassing the storage of the
simulation data. In this work, we present new results from an approach for
in-situ processing based on Hecuba, a framework that provides a highly
distributed database for streaming A&C simulation data directly into the
visualization pipeline to make possible on-line visualization. By integrating
Hecuba with the high-performance cosmological simulator ChaNGa, we enable
real-time, in-situ visualization of N-body simulation results using tools such
as ParaView and VisIVO.

### 9. [In-Situ High Performance Visualization for Astronomy & Cosmology](http://arxiv.org/pdf/2510.24547v1)

Authors: Nicola Tuccari, Eva Sciacca, Yolanda Becerra, Enric Sosa Cintero, Robert Wissing, Sijing Shen, Emiliano Tramontana

The Astronomy & Cosmology (A&C) community is presently witnessing an
unprecedented growth in the quality and quantity of data coming from
simulations and observations. Writing results of numerical simulations to disk
files has long been a bottleneck in high-performance computing. To access
effectively and extract the scientific content of such large-scale data sets
appropriate tools and techniques are needed. This is especially true for
visualization tools, where petascale data size problems cannot be visualized
without some data filtering, which reduces either the resolution or the amount
of data volume managed by the visualization tool.
  A solution to this problem is to run the analysis and visualization
concurrently (in-situ) with the simulation and bypass the storage of the full
results. In particular we use Hecuba, a framework offering a highly distributed
database to stream A\&C simulation data for on-line visualization. We will
demonstrate the Hecuba platform integration with the Changa high performant
cosmological simulator and the in-situ visualization of its N-body results with
the ParaView and VisIVO tools.

### 10. [SPEAR++: Scaling Gradient Inversion via Sparsely-Used Dictionary Learning](http://arxiv.org/pdf/2510.24200v1)

Authors: Alexander Bakarsky, Dimitar I. Dimitrov, Maximilian Baader, Martin Vechev

Federated Learning has seen an increased deployment in real-world scenarios
recently, as it enables the distributed training of machine learning models
without explicit data sharing between individual clients. Yet, the introduction
of the so-called gradient inversion attacks has fundamentally challenged its
privacy-preserving properties. Unfortunately, as these attacks mostly rely on
direct data optimization without any formal guarantees, the vulnerability of
real-world systems remains in dispute and requires tedious testing for each new
federated deployment. To overcome these issues, recently the SPEAR attack was
introduced, which is based on a theoretical analysis of the gradients of linear
layers with ReLU activations. While SPEAR is an important theoretical
breakthrough, the attack's practicality was severely limited by its exponential
runtime in the batch size b. In this work, we fill this gap by applying
State-of-the-Art techniques from Sparsely-Used Dictionary Learning to make the
problem of gradient inversion on linear layers with ReLU activations tractable.
Our experiments demonstrate that our new attack, SPEAR++, retains all desirable
properties of SPEAR, such as robustness to DP noise and FedAvg aggregation,
while being applicable to 10x bigger batch sizes.

### Digital Libraries

### 1. [Comparing Disciplinary Classifications in SSH: Organizational, Channel-Based, and Text-Based Perspectives](http://arxiv.org/pdf/2510.24122v1)

Authors: Cristina Arhiliuc, Raf Guns, Tim C. E. Engels

This study investigates how different approaches to disciplinary
classification represent the Social Sciences and Humanities (SSH) in the
Flemish VABB-SHW database. We compare organizational classification (based on
author affiliation), channel-based cognitive classification (based on
publication venues), and text-based publication-level classification (using
channel titles, publication titles, and abstracts, depending on availability).
The analysis shows that text-based classification generally aligns more closely
with channel-based categories, confirming that the channel choice provides
relevant information about publication content. At the same time, it is closer
to organizational classification than channel-based categories are, suggesting
that textual features capture author affiliations more directly than publishing
channels do. Comparison across the three systems highlights cases of
convergence and divergence, offering insights into how disciplines such as
"Sociology" and "History" extend across fields, while "Law" remains more
contained. Publication-level classification also clarifies the disciplinary
profiles of multidisciplinary journals in the database, which in VABB-SHW show
distinctive profiles with stronger emphases on SSH and health sciences. At the
journal level, fewer than half of outlets with more than 50 publications have
their channel-level classification fully or partially supported by more than
90% of publications. These results demonstrate the added value of text-based
methods for validating classifications and for analysing disciplinary dynamics.

### Discrete Mathematics

### 1. [Pinwheel Scheduling with Real Periods](http://arxiv.org/pdf/2510.24068v1)

Authors: Hiroshi Fujiwara, Kota Miyagi, Katsuhisa Ouchi

For a sequence of tasks, each with a positive integer period, the pinwheel
scheduling problem involves finding a valid schedule in the sense that the
schedule performs one task per day and each task is performed at least once
every consecutive days of its period. It had been conjectured by Chan and Chin
in 1993 that there exists a valid schedule for any sequence of tasks with
density, the sum of the reciprocals of each period, at most $\frac{5}{6}$.
Recently, Kawamura settled this conjecture affirmatively. In this paper we
consider an extended version with real periods proposed by Kawamura, in which a
valid schedule must perform each task $i$ having a real period~$a_{i}$ at least
$l$ times in any consecutive $\lceil l a_{i} \rceil$ days for all positive
integer $l$. We show that any sequence of tasks such that the periods take
three distinct real values and the density is at most $\frac{5}{6}$ admits a
valid schedule. We hereby conjecture that the conjecture of Chan and Chin is
true also for real periods.

### Data Structures and Algorithms

### 1. [On Competitiveness of Dynamic Replication for Distributed Data Access](http://arxiv.org/pdf/2510.24098v1)

Authors: Tianyu Zuo, Xueyan Tang, Bu Sung Lee, Jianfei Cai

This paper studies an online cost optimization problem for distributed
storage and access. The goal is to dynamically create and delete copies of data
objects over time at geo-distributed servers to serve access requests and
minimize the total storage and network cost. We revisit a recent algorithm in
the literature and show that it does not have a competitive ratio of $2$ as
claimed by constructing a counterexample. We further prove that no
deterministic online algorithm can achieve a competitive ratio bounded by $2$
for the general cost optimization problem. We develop an online algorithm and
prove that it achieves a competitive ratio of $\max\{2, \min\{\gamma, 3\}\}$,
where $\gamma$ is the max/min storage cost ratio among all servers. Examples
are given to confirm the tightness of competitive analysis. We also empirically
evaluate algorithms using real object access traces.

### 2. [Pinwheel Scheduling with Real Periods](http://arxiv.org/pdf/2510.24068v1)

Authors: Hiroshi Fujiwara, Kota Miyagi, Katsuhisa Ouchi

For a sequence of tasks, each with a positive integer period, the pinwheel
scheduling problem involves finding a valid schedule in the sense that the
schedule performs one task per day and each task is performed at least once
every consecutive days of its period. It had been conjectured by Chan and Chin
in 1993 that there exists a valid schedule for any sequence of tasks with
density, the sum of the reciprocals of each period, at most $\frac{5}{6}$.
Recently, Kawamura settled this conjecture affirmatively. In this paper we
consider an extended version with real periods proposed by Kawamura, in which a
valid schedule must perform each task $i$ having a real period~$a_{i}$ at least
$l$ times in any consecutive $\lceil l a_{i} \rceil$ days for all positive
integer $l$. We show that any sequence of tasks such that the periods take
three distinct real values and the density is at most $\frac{5}{6}$ admits a
valid schedule. We hereby conjecture that the conjecture of Chan and Chin is
true also for real periods.

### 3. [Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers](http://arxiv.org/pdf/2510.24621v1)

Authors: Ziyi Fang, Lingxiao Huang, Runkai Yang

We study the robust geometric median problem in Euclidean space
$\mathbb{R}^d$, with a focus on coreset construction.A coreset is a compact
summary of a dataset $P$ of size $n$ that approximates the robust cost for all
centers $c$ within a multiplicative error $\varepsilon$. Given an outlier count
$m$, we construct a coreset of size $\tilde{O}(\varepsilon^{-2} \cdot
\min\{\varepsilon^{-2}, d\})$ when $n \geq 4m$, eliminating the $O(m)$
dependency present in prior work [Huang et al., 2022 & 2023]. For the special
case of $d = 1$, we achieve an optimal coreset size of
$\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n} \varepsilon^{-1})$, revealing
a clear separation from the vanilla case studied in [Huang et al., 2023;
Afshani and Chris, 2024]. Our results further extend to robust
$(k,z)$-clustering in various metric spaces, eliminating the $m$-dependence
under mild data assumptions. The key technical contribution is a novel
non-component-wise error analysis, enabling substantial reduction of outlier
influence, unlike prior methods that retain them.Empirically, our algorithms
consistently outperform existing baselines in terms of size-accuracy tradeoffs
and runtime, even when data assumptions are violated across a wide range of
datasets.

### Emerging Technologies

### 1. [Evaluating Fitness Averaging Strategies in Cooperative NeuroCoEvolution for Automated Soft Actuator Design](http://arxiv.org/pdf/2510.24510v1)

Authors: Hugo Alcaraz-Herrera, Michail-Antisthenis Tsompanas, Igor Balaz, Andrew Adamatzky

Soft robotics are increasingly favoured in specific applications such as
healthcare, due to their adaptability, which stems from the non-linear
properties of their building materials. However, these properties also pose
significant challenges in designing the morphologies and controllers of soft
robots. The relatively short history of this field has not yet produced
sufficient knowledge to consistently derive optimal solutions. Consequently, an
automated process for the design of soft robot morphologies can be extremely
helpful. This study focusses on the cooperative NeuroCoEvolution of networks
that are indirect representations of soft robot actuators. Both the
morphologies and controllers represented by Compositional Pattern Producing
Networks are evolved using the well-established method NeuroEvolution of
Augmented Topologies (CPPN-NEAT). The CoEvolution of controllers and
morphologies is implemented using the top n individuals from the cooperating
population, with various averaging methods tested to determine the fitness of
the evaluated individuals. The test-case application for this research is the
optimisation of a soft actuator for a drug delivery system. The primary metric
used is the maximum displacement of one end of the actuator in a specified
direction. Additionally, the robustness of the evolved morphologies is assessed
against a range of randomly generated controllers to simulate potential noise
in real-world applications. The results of this investigation indicate that
CPPN-NEAT produces superior morphologies compared to previously published
results from multi-objective optimisation, with reduced computational effort
and time. Moreover, the best configuration is found to be CoEvolution with the
two best individuals from the cooperative population and the averaging of their
fitness using the weighted mean method.

### Graphics

### 1. [Fast and accurate neural reflectance transformation imaging through knowledge distillation](http://arxiv.org/pdf/2510.24486v1)

Authors: Tinsae G. Dulecha, Leonardo Righetto, Ruggero Pintus, Enrico Gobbetti, Andrea Giachetti

Reflectance Transformation Imaging (RTI) is very popular for its ability to
visually analyze surfaces by enhancing surface details through interactive
relighting, starting from only a few tens of photographs taken with a fixed
camera and variable illumination. Traditional methods like Polynomial Texture
Maps (PTM) and Hemispherical Harmonics (HSH) are compact and fast, but struggle
to accurately capture complex reflectance fields using few per-pixel
coefficients and fixed bases, leading to artifacts, especially in highly
reflective or shadowed areas. The NeuralRTI approach, which exploits a neural
autoencoder to learn a compact function that better approximates the local
reflectance as a function of light directions, has been shown to produce
superior quality at comparable storage cost. However, as it performs
interactive relighting with custom decoder networks with many parameters, the
rendering step is computationally expensive and not feasible at full resolution
for large images on limited hardware. Earlier attempts to reduce costs by
directly training smaller networks have failed to produce valid results. For
this reason, we propose to reduce its computational cost through a novel
solution based on Knowledge Distillation (DisK-NeuralRTI). ...

### Computer Science and Game Theory

### 1. [Self interest cumulative subtraction games](http://arxiv.org/pdf/2510.24280v1)

Authors: Anjali Bhagat, Tanmay Kulkarni, Urban Larsson, Divya Murali

Subtraction games have a rich literature as normal-play combinatorial games
(e.g., Berlekamp, Conway, and Guy, 1982). Recently, the theory has been
extended to zero-sum scoring play (Cohensius et al. 2019). Here, we take the
approach of cumulative self-interest games, as introduced in a recent framework
preprint by Larsson, Meir, and Zick. By adapting standard Pure Subgame Perfect
Equilibria (PSPE) from classical game theory, players must declare and commit
to acting either ``friendly'' or ``antagonistic'' in case of indifference.
Whenever the subtraction set has size two, we establish a tie-breaking rule
monotonicity: a friendly player can never benefit by a deterministic deviation
to antagonistic play. This type of terminology is new to both ``economic'' and
``combinatorial'' games, but it becomes essential in the self-interest
cumulative setting. The main result is an immediate consequence of the
tie-breaking rule's monotonicity; in the case of two-action subtraction sets,
two antagonistic players are never better off than two friendly players, i.e.,
their PSPE utilities are never greater. For larger subtraction sets, we
conjecture that the main result continues to hold, while tie-breaking
monotonicity may fail, and we provide empirical evidence in support of both
statements.

### 2. [Exploring Emergent Topological Properties in Socio-Economic Networks through Learning Heterogeneity](http://arxiv.org/pdf/2510.24107v1)

Authors: Chanuka Karavita, Zehua Lyu, Dharshana Kasthurirathna, Mahendra Piraveenan

Understanding how individual learning behavior and structural dynamics
interact is essential to modeling emergent phenomena in socioeconomic networks.
While bounded rationality and network adaptation have been widely studied, the
role of heterogeneous learning rates both at the agent and network levels
remains under explored. This paper introduces a dual-learning framework that
integrates individualized learning rates for agents and a rewiring rate for the
network, reflecting real-world cognitive diversity and structural adaptability.
  Using a simulation model based on the Prisoner's Dilemma and Quantal Response
Equilibrium, we analyze how variations in these learning rates affect the
emergence of large-scale network structures. Results show that lower and more
homogeneously distributed learning rates promote scale-free networks, while
higher or more heterogeneously distributed learning rates lead to the emergence
of core-periphery topologies. Key topological metrics including scale-free
exponents, Estrada heterogeneity, and assortativity reveal that both the speed
and variability of learning critically shape system rationality and network
architecture. This work provides a unified framework for examining how
individual learnability and structural adaptability drive the formation of
socioeconomic networks with diverse topologies, offering new insights into
adaptive behavior, systemic organization, and resilience.

### Human-Computer Interaction

### 1. [Toward Socially-Aware LLMs: A Survey of Multimodal Approaches to Human Behavior Understanding](http://arxiv.org/pdf/2510.23947v1)

Authors: Zihan Liu, Parisa Rabbani, Veda Duddu, Kyle Fan, Madison Lee, Yun Huang

LLM-powered multimodal systems are increasingly used to interpret human
social behavior, yet how researchers apply the models' 'social competence'
remains poorly understood. This paper presents a systematic literature review
of 176 publications across different application domains (e.g., healthcare,
education, and entertainment). Using a four-dimensional coding framework
(application, technical, evaluative, and ethical), we find (1) frequent use of
pattern recognition and information extraction from multimodal sources, but
limited support for adaptive, interactive reasoning; (2) a dominant
'modality-to-text' pipeline that privileges language over rich audiovisual
cues, striping away nuanced social cues; (3) evaluation practices reliant on
static benchmarks, with socially grounded, human-centered assessments rare; and
(4) Ethical discussions focused mainly on legal and rights-related risks (e.g.,
privacy), leaving societal risks (e.g., deception) overlooked--or at best
acknowledged but left unaddressed. We outline a research agenda for evaluating
socially competent, ethically informed, and interaction-aware multi-modal
systems.

### 2. [Modeling Object Attention in Mobile AR for Intrinsic Cognitive Security](http://arxiv.org/pdf/2510.24004v1)

Authors: Shane Dirksen, Radha Kumaran, You-Jin Kim, Yilin Wang, Tobias Höllerer

We study attention in mobile Augmented Reality (AR) using object recall as a
proxy outcome. We observe that the ability to recall an object (physical or
virtual) that was encountered in a mobile AR experience depends on many
possible impact factors and attributes, with some objects being readily
recalled while others are not, and some people recalling objects overall much
better or worse than others. This opens up a potential cognitive attack in
which adversaries might create conditions that make an AR user not recall
certain potentially mission-critical objects. We explore whether a calibrated
predictor of object recall can help shield against such cognitive attacks. We
pool data from four mobile AR studies (with a total of 1,152 object recall
probes) and fit a Partial Least Squares Structural Equation Model (PLS-SEM)
with formative Object, Scene, and User State composites predicting recall, also
benchmarking against Random Forest and multilayer perceptron classifiers.
PLS-SEM attains the best F1 score in three of four studies. Additionally, path
estimates identify lighting, augmentation density, AR registration stability,
cognitive load, and AR familiarity as primary drivers. The model outputs
per-object recall probabilities that can drive interface adjustments when
predicted recall falls. Overall, PLS-SEM provides competitive accuracy with
interpretable levers for design and evaluation in mobile AR.

### 3. [Understanding Reader Perception Shifts upon Disclosure of AI Authorship](http://arxiv.org/pdf/2510.24011v1)

Authors: Hiroki Nakano, Jo Takezawa, Fabrice Matulic, Chi-Lan Yang, Koji Yatani

As AI writing support becomes ubiquitous, how disclosing its use affects
reader perception remains a critical, underexplored question. We conducted a
study with 261 participants to examine how revealing varying levels of AI
involvement shifts author impressions across six distinct communicative acts.
Our analysis of 990 responses shows that disclosure generally erodes
perceptions of trustworthiness, caring, competence, and likability, with the
sharpest declines in social and interpersonal writing. A thematic analysis of
participants' feedback links these negative shifts to a perceived loss of human
sincerity, diminished author effort, and the contextual inappropriateness of
AI. Conversely, we find that higher AI literacy mitigates these negative
perceptions, leading to greater tolerance or even appreciation for AI use. Our
results highlight the nuanced social dynamics of AI-mediated authorship and
inform design implications for creating transparent, context-sensitive writing
systems that better preserve trust and authenticity.

### 4. [VR-Assisted Guide Dog Training: A 360° PanoHaptic System for Right-Hand Commands Analysis](http://arxiv.org/pdf/2510.24057v1)

Authors: Qirong Zhu, Ansheng Wang, Shinji Tanaka, Yasutoshi Makino, Hiroyuki Shinoda

This paper presents a VR-based guide dog training system designed to assist
novice trainers in understanding guide dog behavior and issuing appropriate
training commands. Guide dogs play a vital role in supporting independent
mobility for visually impaired individuals, yet the limited number of skilled
trainers restricts their availability. Training is highly demanding, requiring
accurate observation of the dog's status and precise command issuance,
especially through right-hand gestures. While the trainer's left hand holds the
harness to perceive haptic cues, the right hand is used to indicate directions,
maintain attention, and provide comfort, with motion patterns varying by
scenario and the dog's progress. Currently, novices learn mainly by observing
experts or watching videos, which lacks immersion and makes it difficult to
adopt the trainer's perspective for understanding behavior or synchronizing
command timing.
  To address these limitations, the proposed system introduces a VR-based
assistive platform integrating panoramic visuals and haptic feedback to create
an immersive training environment. The visual module provides contextual
guidance, including cues for command execution and real-time comparison of the
user's posture with standard actions, while the haptic module delivers tactile
feedback for command gestures. Users can re-experience training sessions across
diverse scenarios and dog proficiency levels, allowing independent and repeated
practice. By improving the timing, accuracy, and expressiveness of right-hand
commands, the system aims to accelerate skill acquisition, enhance training
quality, and mitigate the shortage of qualified trainers, ultimately increasing
the availability of guide dogs for visually impaired individuals.

### 5. [Building AI Literacy at Home: How Families Navigate Children's Self-Directed Learning with AI](http://arxiv.org/pdf/2510.24070v1)

Authors: Jingyi Xie, Chuhao Wu, Ge Wang, Rui Yu, He Zhang, Ronald Metoyer, Si Chen

As generative AI becomes embedded in children's learning spaces, families
face new challenges in guiding its use. Middle childhood (ages 7-13) is a
critical stage where children seek autonomy even as parental influence remains
strong. Using self-directed learning (SDL) as a lens, we examine how parents
perceive and support children's developing AI literacy through focus groups
with 13 parent-child pairs. Parents described evolving phases of engagement
driven by screen time, self-motivation, and growing knowledge. While many
framed AI primarily as a study tool, few considered its non-educational roles
or risks, such as privacy and infrastructural embedding. Parents also noted
gaps in their own AI understanding, often turning to joint exploration and
engagement as a form of co-learning. Our findings reveal how families
co-construct children's AI literacy, exposing tensions between practical
expectations and critical literacies, and provide design implications that
foster SDL while balancing autonomy and oversight.

### 6. [Advancing Interdisciplinary Approaches to Online Safety Research](http://arxiv.org/pdf/2510.24227v1)

Authors: Senuri Wijenayake, Joanne Gray, Asangi Jayatilaka, Louise La Sala, Nalin Arachchilage, Ryan M. Kelly, Sanchari Das

The growing prevalence of negative experiences in online spaces demands
urgent attention from the human-computer interaction (HCI) community. However,
research on online safety remains fragmented across different HCI subfields,
with limited communication and collaboration between disciplines. This siloed
approach risks creating ineffective responses, including design solutions that
fail to meet the diverse needs of users, and policy efforts that overlook
critical usability concerns. This workshop aims to foster interdisciplinary
dialogue on online safety by bringing together researchers from within and
beyond HCI - including but not limited to Social Computing, Digital Design,
Internet Policy, Cybersecurity, Ethics, and Social Sciences. By uniting
researchers, policymakers, industry practitioners, and community advocates we
aim to identify shared challenges in online safety research, highlight gaps in
current knowledge, and establish common research priorities. The workshop will
support the development of interdisciplinary research plans and establish
collaborative environments - both within and beyond Australia - to action them.

### 7. [Detecting the Use of Generative AI in Crowdsourced Surveys: Implications for Data Integrity](http://arxiv.org/pdf/2510.24594v1)

Authors: Dapeng Zhang, Marina Katoh, Weiping Pei

The widespread adoption of generative AI (GenAI) has introduced new
challenges in crowdsourced data collection, particularly in survey-based
research. While GenAI offers powerful capabilities, its unintended use in
crowdsourcing, such as generating automated survey responses, threatens the
integrity of empirical research and complicates efforts to understand public
opinion and behavior. In this study, we investigate and evaluate two approaches
for detecting AI-generated responses in online surveys: LLM-based detection and
signature-based detection. We conducted experiments across seven survey
studies, comparing responses collected before 2022 with those collected after
the release of ChatGPT. Our findings reveal a significant increase in
AI-generated responses in the post-2022 studies, highlighting how GenAI may
silently distort crowdsourced data. This work raises broader concerns about
evolving landscape of data integrity, where GenAI can compromise data quality,
mislead researchers, and influence downstream findings in fields such as
health, politics, and social behavior. By surfacing detection strategies and
empirical evidence of GenAI's impact, we aim to contribute to ongoing
conversation about safeguarding research integrity and supporting scholars
navigating these methodological and ethical challenges.

### 8. [What Does It Take? Developing a Smartphone App that Motivates Older Adults to be Physically Active](http://arxiv.org/pdf/2510.24638v1)

Authors: Sabrina Haque, Kyle Henry, Troyee Saha, Kimberly Vanhoose, Jobaidul Boni, Samantha Moss, Kate Hyun, Kathy Siepker, Xiangli Gu, Angela Liegey-Dougall, Stephen Mattingly, Christoph Csallner

Maintaining physical activity is essential for older adults' health and
well-being, yet participation remains low. Traditional paper-based and
in-person interventions have been effective but face scalability issues.
Smartphone apps offer a potential solution, but their effectiveness in
real-world use remains underexplored. Most prior studies take place in
controlled environments, use specialized hardware, or rely on in-person
training sessions or researcher-led setup. This study examines the feasibility
and engagement of Senior Fit, a standalone mobile fitness app designed for
older adults. We conducted continuous testing with 25 participants aged 65-85,
refining the app based on their feedback to improve usability and
accessibility. Our findings underscore both the potential and key challenges in
designing digital health interventions. Older adults valued features such as
video demonstrations and reminders that made activity feel accessible and
motivating, yet some expressed frustration with manual logging and limited
personalization. The Facebook group provided encouragement for some but
excluded others unfamiliar with the platform. These results highlight the need
for fitness apps that integrate flexible tracking, clear feedback, and
low-barrier social support. We contribute design recommendations for creating
inclusive mobile fitness tools that align with older adults' routines and
capabilities, offering insights for future long-term, real-world deployments.

### 9. [Developer Productivity with GenAI](http://arxiv.org/pdf/2510.24265v1)

Authors: Sadia Afroz, Zixuan Feng, Katie Kimura, Bianca Trinkenreich, Igor Steinmacher, Anita Sarma

Generative AI (GenAI) tools are increasingly being adopted in software
development as productivity aids. However, evidence regarding where and when
these tools actually enhance productivity is unclear. In this paper, we
investigate how GenAI adoption affects different dimensions of developer
productivity. We surveyed 415 software practitioners to capture their
perceptions of productivity changes associated with AI-assisted development
using the SPACE framework - Satisfaction and well-being, Performance, Activity,
Communication and collaboration, and Efficiency and flow. Our results,
disaggregated by frequency of AI usage, reveal limited overall productivity
change, highlighting the productivity paradox in which developers become faster
but do not necessarily create better software or feel more fulfilled.

### 10. [AI for a Planet Under Pressure](http://arxiv.org/pdf/2510.24373v1)

Authors: Victor Galaz, Maria Schewenius, Jonathan F. Donges, Ingo Fetzer, Erik Zhivkoplias, Wolfram Barfuss, Louis Delannoy, Lan Wang-Erlandsson, Maximilian Gelbrecht, Jobst Heitzig, Jonas Hentati-Sundberg, Christopher Kennedy, Nielja Knecht, Romi Lotcheris, Miguel Mahecha, Andrew Merrie, David Montero, Timon McPhearson, Ahmed Mustafa, Magnus Nyström, Drew Purves, Juan C. Rocha, Masahiro Ryo, Claudia van der Salm, Samuel T. Segun, Anna B. Stephenson, Elizabeth Tellman, Felipe Tobar, Alice Vadrot

Artificial intelligence (AI) is already driving scientific breakthroughs in a
variety of research fields, ranging from the life sciences to mathematics. This
raises a critical question: can AI be applied both responsibly and effectively
to address complex and interconnected sustainability challenges? This report is
the result of a collaboration between the Stockholm resilience Centre
(Stockholm University), the Potsdam Institute for Climate Impact Research
(PIK), and Google DeepMind. Our work explores the potential and limitations of
using AI as a research method to help tackle eight broad sustainability
challenges. The results build on iterated expert dialogues and assessments, a
systematic AI-supported literature overview including over 8,500 academic
publications, and expert deep-dives into eight specific issue areas. The report
also includes recommendations to sustainability scientists, research funders,
the private sector, and philanthropies.

### Information Retrieval

### 1. [Resource-Efficient LLM Application for Structured Transformation of Unstructured Financial Contracts](http://arxiv.org/pdf/2510.23990v1)

Authors: Maruf Ahmed Mridul, Oshani Seneviratne

The transformation of unstructured legal contracts into standardized,
machine-readable formats is essential for automating financial workflows. The
Common Domain Model (CDM) provides a standardized framework for this purpose,
but converting complex legal documents like Credit Support Annexes (CSAs) into
CDM representations remains a significant challenge. In this paper, we present
an extension of the CDMizer framework, a template-driven solution that ensures
syntactic correctness and adherence to the CDM schema during contract-to-CDM
conversion. We apply this extended framework to a real-world task, comparing
its performance with a benchmark developed by the International Swaps and
Derivatives Association (ISDA) for CSA clause extraction. Our results show that
CDMizer, when integrated with a significantly smaller, open-source Large
Language Model (LLM), achieves competitive performance in terms of accuracy and
efficiency against larger, proprietary models. This work underscores the
potential of resource-efficient solutions to automate legal contract
transformation, offering a cost-effective and scalable approach that can meet
the needs of financial institutions with constrained resources or strict data
privacy requirements.

### 2. [DUET: Dual Model Co-Training for Entire Space CTR Prediction](http://arxiv.org/pdf/2510.24369v1)

Authors: Yutian Xiao, Meng Yuan, Fuzhen Zhuang, Wei Chen, Shukuan Wang, Shanqi Liu, Chao Feng, Wenhui Yu, Xiang Li, Lantao Hu, Han Li, Zhao Zhang

The pre-ranking stage plays a pivotal role in large-scale recommender systems
but faces an intrinsic trade-off between model expressiveness and computational
efficiency. Owing to the massive candidate pool and strict latency constraints,
industry systems often rely on lightweight two-tower architectures, which are
computationally efficient yet limited in estimation capability. As a result,
they struggle to capture the complex synergistic and suppressive relationships
among candidate items, which are essential for producing contextually coherent
and diverse recommendation lists. Moreover, this simplicity further amplifies
the Sample Selection Bias (SSB) problem, as coarse-grained models trained on
biased exposure data must generalize to a much larger candidate space with
distinct distributions.
  To address these issues, we propose \textbf{DUET} (\textbf{DU}al Model
Co-Training for \textbf{E}ntire Space C\textbf{T}R Prediction), a set-wise
pre-ranking framework that achieves expressive modeling under tight
computational budgets. Instead of scoring items independently, DUET performs
set-level prediction over the entire candidate subset in a single forward pass,
enabling information-aware interactions among candidates while amortizing the
computational cost across the set. Moreover, a dual model co-training mechanism
extends supervision to unexposed items via mutual pseudo-label refinement,
effectively mitigating SSB. Validated through extensive offline experiments and
online A/B testing, DUET consistently outperforms state-of-the-art baselines
and achieves improvements across multiple core business metrics. At present,
DUET has been fully deployed in Kuaishou and Kuaishou Lite Apps, serving the
main traffic for hundreds of millions of users.

### 3. [From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations](http://arxiv.org/pdf/2510.24430v1)

Authors: Yejin Kim, Shaghayegh Agah, Mayur Nankani, Neeraj Sharma, Feifei Peng, Maria Peifer, Sardar Hamidian, H Howie Huang

Most recommender systems treat timestamps as numeric or cyclical values,
overlooking real-world context such as holidays, events, and seasonal patterns.
We propose a scalable framework that uses large language models (LLMs) to
generate geo-temporal embeddings from only a timestamp and coarse location,
capturing holidays, seasonal trends, and local/global events. We then introduce
a geo-temporal embedding informativeness test as a lightweight diagnostic,
demonstrating on MovieLens, LastFM, and a production dataset that these
embeddings provide predictive signal consistent with the outcomes of full model
integrations. Geo-temporal embeddings are incorporated into sequential models
through (1) direct feature fusion with metadata embeddings or (2) an auxiliary
loss that enforces semantic and geo-temporal alignment. Our findings highlight
the need for adaptive or hybrid recommendation strategies, and we release a
context-enriched MovieLens dataset to support future research.

### 4. [MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation](http://arxiv.org/pdf/2510.24431v1)

Authors: Xiaoyu Kong, Leheng Sheng, Junfei Tan, Yuxin Chen, Jiancan Wu, An Zhang, Xiang Wang, Xiangnan He

The recent success of large language models (LLMs) has renewed interest in
whether recommender systems can achieve similar scaling benefits. Conventional
recommenders, dominated by massive embedding tables, tend to plateau as
embedding dimensions grow. In contrast, the emerging generative paradigm
replaces embeddings with compact Semantic ID (SID) sequences produced by
autoregressive Transformers. Yet most industrial deployments remain
proprietary, leaving two fundamental questions open: (1) Do the expected
scaling laws hold on public benchmarks? (2) What is the minimal post-training
recipe that enables competitive performance?
  We present MiniOneRec, to the best of our knowledge, the first fully
open-source generative recommendation framework, which provides an end-to-end
workflow spanning SID construction, supervised fine-tuning, and
recommendation-oriented reinforcement learning. We generate SIDs via a Residual
Quantized VAE and post-train Qwen backbones ranging from 0.5B to 7B parameters
on the Amazon Review dataset. Our experiments reveal a consistent downward
trend in both training and evaluation losses with increasing model size,
validating the parameter efficiency of the generative approach. To further
enhance performance, we propose a lightweight yet effective post-training
pipeline that (1) enforces full-process SID alignment and (2) applies
reinforcement learning with constrained decoding and hybrid rewards. Together,
these techniques yield significant improvements in both ranking accuracy and
candidate diversity.

### 5. [Optimizing Retrieval for RAG via Reinforced Contrastive Learning](http://arxiv.org/pdf/2510.24652v1)

Authors: Jiawei Zhou, Lei Chen

As retrieval-augmented generation (RAG) becomes increasingly widespread, the
role of information retrieval (IR) is shifting from retrieving information for
human users to retrieving contextual knowledge for artificial intelligence (AI)
systems, where relevance becomes difficult to define or annotate beforehand. To
address this challenge, we propose R3, a Retrieval framework optimized for RAG
through trialand-feedback Reinforced contrastive learning. Unlike prior
approaches that rely on annotated or synthetic data for supervised fine-tuning,
R3 enables the retriever to dynamically explore and optimize relevance within
the RAG environment. During training, the retrieved results interact with the
environment to produce contrastive signals that automatically guide the
retriever's self-improvement. Extensive experiments across diverse tasks
demonstrate that R3 improves RAG performance by 5.2% over the original
retriever and surpasses state-of-the-art retrievers by 4.9%, while achieving
comparable results to LLM-augmented retrieval and RAG systems built on
post-trained or instruction-tuned LLMs. It is both efficient and practical,
requiring only 4 GPUs and completing training within a single day.

### 6. [Metadata-Driven Retrieval-Augmented Generation for Financial Question Answering](http://arxiv.org/pdf/2510.24402v1)

Authors: Michail Dadopoulos, Anestis Ladas, Stratos Moschidis, Ioannis Negkakis

Retrieval-Augmented Generation (RAG) struggles on long, structured financial
filings where relevant evidence is sparse and cross-referenced. This paper
presents a systematic investigation of advanced metadata-driven
Retrieval-Augmented Generation (RAG) techniques, proposing and evaluating a
novel, multi-stage RAG architecture that leverages LLM-generated metadata. We
introduce a sophisticated indexing pipeline to create contextually rich
document chunks and benchmark a spectrum of enhancements, including
pre-retrieval filtering, post-retrieval reranking, and enriched embeddings,
benchmarked on the FinanceBench dataset. Our results reveal that while a
powerful reranker is essential for precision, the most significant performance
gains come from embedding chunk metadata directly with text ("contextual
chunks"). Our proposed optimal architecture combines LLM-driven pre-retrieval
optimizations with these contextual embeddings to achieve superior performance.
Additionally, we present a custom metadata reranker that offers a compelling,
cost-effective alternative to commercial solutions, highlighting a practical
trade-off between peak performance and operational efficiency. This study
provides a blueprint for building robust, metadata-aware RAG systems for
financial document analysis.

### 7. [Iterative Critique-Refine Framework for Enhancing LLM Personalization](http://arxiv.org/pdf/2510.24469v1)

Authors: Durga Prasad Maram, Dhruvin Gandhi, Zonghai Yao, Gayathri Akkinapalli, Franck Dernoncourt, Yu Wang, Ryan A. Rossi, Nesreen K. Ahmed

Personalized text generation requires models not only to produce coherent
text but also to align with a target user's style, tone, and topical focus.
Existing retrieval-augmented approaches such as LaMP and PGraphRAG enrich
profiles with user and neighbor histories, but they stop at generation and
often yield outputs that drift in tone, topic, or style. We present PerFine, a
unified, training-free critique-refine framework that enhances personalization
through iterative, profile-grounded feedback. In each iteration, an LLM
generator produces a draft conditioned on the retrieved profile, and a critic
LLM - also conditioned on the same profile - provides structured feedback on
tone, vocabulary, sentence structure, and topicality. The generator then
revises, while a novel knockout strategy retains the stronger draft across
iterations. We further study additional inference-time strategies such as
Best-of-N and Topic Extraction to balance quality and efficiency. Across Yelp,
Goodreads, and Amazon datasets, PerFine consistently improves personalization
over PGraphRAG, with GEval gains of +7-13%, steady improvements over 3-5
refinement iterations, and scalability with increasing critic size. These
results highlight that post-hoc, profile-aware feedback offers a powerful
paradigm for personalized LLM generation that is both training-free and
model-agnostic.

### 8. [Tongyi DeepResearch Technical Report](http://arxiv.org/pdf/2510.24701v1)

Authors: Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, Liangcai Su, Litu Ou, Liwen Zhang, Pengjun Xie, Rui Ye, Wenbiao Yin, Xinmiao Yu, Xinyu Wang, Xixi Wu, Xuanzhong Chen, Yida Zhao, Zhen Zhang, Zhengwei Tao, Zhongwang Zhang, Zile Qiao, Chenxi Wang, Donglei Yu, Gang Fu, Haiyang Shen, Jiayin Yang, Jun Lin, Junkai Zhang, Kui Zeng, Li Yang, Hailong Yin, Maojia Song, Ming Yan, Peng Xia, Qian Xiao, Rui Min, Ruixue Ding, Runnan Fang, Shaowei Chen, Shen Huang, Shihang Wang, Shihao Cai, Weizhou Shen, Xiaobin Wang, Xin Guan, Xinyu Geng, Yingcheng Shi, Yuning Wu, Zhuo Chen, Zijian Li, Yong Jiang

We present Tongyi DeepResearch, an agentic large language model, which is
specifically designed for long-horizon, deep information-seeking research
tasks. To incentivize autonomous deep research agency, Tongyi DeepResearch is
developed through an end-to-end training framework that combines agentic
mid-training and agentic post-training, enabling scalable reasoning and
information seeking across complex tasks. We design a highly scalable data
synthesis pipeline that is fully automatic, without relying on costly human
annotation, and empowers all training stages. By constructing customized
environments for each stage, our system enables stable and consistent
interactions throughout. Tongyi DeepResearch, featuring 30.5 billion total
parameters, with only 3.3 billion activated per token, achieves
state-of-the-art performance across a range of agentic deep research
benchmarks, including Humanity's Last Exam, BrowseComp, BrowseComp-ZH,
WebWalkerQA, xbench-DeepSearch, FRAMES and xbench-DeepSearch-2510. We
open-source the model, framework, and complete solutions to empower the
community.

### Machine Learning

### 1. [Predicting Barge Tow Size on Inland Waterways Using Vessel Trajectory Derived Features: Proof of Concept](http://arxiv.org/pdf/2510.23994v1)

Authors: Geoffery Agorku, Sarah Hernandez, Hayley Hames, Cade Wagner

Accurate, real-time estimation of barge quantity on inland waterways remains
a critical challenge due to the non-self-propelled nature of barges and the
limitations of existing monitoring systems. This study introduces a novel
method to use Automatic Identification System (AIS) vessel tracking data to
predict the number of barges in tow using Machine Learning (ML). To train and
test the model, barge instances were manually annotated from satellite scenes
across the Lower Mississippi River. Labeled images were matched to AIS vessel
tracks using a spatiotemporal matching procedure. A comprehensive set of 30
AIS-derived features capturing vessel geometry, dynamic movement, and
trajectory patterns were created and evaluated using Recursive Feature
Elimination (RFE) to identify the most predictive variables. Six regression
models, including ensemble, kernel-based, and generalized linear approaches,
were trained and evaluated. The Poisson Regressor model yielded the best
performance, achieving a Mean Absolute Error (MAE) of 1.92 barges using 12 of
the 30 features. The feature importance analysis revealed that metrics
capturing vessel maneuverability such as course entropy, speed variability and
trip length were most predictive of barge count. The proposed approach provides
a scalable, readily implementable method for enhancing Maritime Domain
Awareness (MDA), with strong potential applications in lock scheduling, port
management, and freight planning. Future work will expand the proof of concept
presented here to explore model transferability to other inland rivers with
differing operational and environmental conditions.

### 2. [Efficient Global-Local Fusion Sampling for Physics-Informed Neural Networks](http://arxiv.org/pdf/2510.24026v1)

Authors: Jiaqi Luo, Shixin Xu, Zhouwang Yang

The accuracy of Physics-Informed Neural Networks (PINNs) critically depends
on the placement of collocation points, as the PDE loss is approximated through
sampling over the solution domain. Global sampling ensures stability by
covering the entire domain but requires many samples and is computationally
expensive, whereas local sampling improves efficiency by focusing on
high-residual regions but may neglect well-learned areas, reducing robustness.
We propose a Global-Local Fusion (GLF) Sampling Strategy that combines the
strengths of both approaches. Specifically, new collocation points are
generated by perturbing training points with Gaussian noise scaled inversely to
the residual, thereby concentrating samples in difficult regions while
preserving exploration. To further reduce computational overhead, a lightweight
linear surrogate is introduced to approximate the global residual-based
distribution, achieving similar effectiveness at a fraction of the cost.
Together, these components, residual-adaptive sampling and residual-based
approximation, preserve the stability of global methods while retaining the
efficiency of local refinement. Extensive experiments on benchmark PDEs
demonstrate that GLF consistently improves both accuracy and efficiency
compared with global and local sampling strategies. This study provides a
practical and scalable framework for enhancing the reliability and efficiency
of PINNs in solving complex and high-dimensional PDEs.

### 3. [Localized Kernel Projection Outlyingness: A Two-Stage Approach for Multi-Modal Outlier Detection](http://arxiv.org/pdf/2510.24043v1)

Authors: Akira Tamamori

This paper presents Two-Stage LKPLO, a novel multi-stage outlier detection
framework that overcomes the coexisting limitations of conventional
projection-based methods: their reliance on a fixed statistical metric and
their assumption of a single data structure. Our framework uniquely synthesizes
three key concepts: (1) a generalized loss-based outlyingness measure (PLO)
that replaces the fixed metric with flexible, adaptive loss functions like our
proposed SVM-like loss; (2) a global kernel PCA stage to linearize non-linear
data structures; and (3) a subsequent local clustering stage to handle
multi-modal distributions. Comprehensive 5-fold cross-validation experiments on
10 benchmark datasets, with automated hyperparameter optimization, demonstrate
that Two-Stage LKPLO achieves state-of-the-art performance. It significantly
outperforms strong baselines on datasets with challenging structures where
existing methods fail, most notably on multi-cluster data (Optdigits) and
complex, high-dimensional data (Arrhythmia). Furthermore, an ablation study
empirically confirms that the synergistic combination of both the kernelization
and localization stages is indispensable for its superior performance. This
work contributes a powerful new tool for a significant class of outlier
detection problems and underscores the importance of hybrid, multi-stage
architectures.

### 4. [Mitigating Negative Transfer via Reducing Environmental Disagreement](http://arxiv.org/pdf/2510.24044v1)

Authors: Hui Sun, Zheng Xie, Hao-Yuan He, Ming Li

Unsupervised Domain Adaptation~(UDA) focuses on transferring knowledge from a
labeled source domain to an unlabeled target domain, addressing the challenge
of \emph{domain shift}. Significant domain shifts hinder effective knowledge
transfer, leading to \emph{negative transfer} and deteriorating model
performance. Therefore, mitigating negative transfer is essential. This study
revisits negative transfer through the lens of causally disentangled learning,
emphasizing cross-domain discriminative disagreement on non-causal
environmental features as a critical factor. Our theoretical analysis reveals
that overreliance on non-causal environmental features as the environment
evolves can cause discriminative disagreements~(termed \emph{environmental
disagreement}), thereby resulting in negative transfer. To address this, we
propose Reducing Environmental Disagreement~(RED), which disentangles each
sample into domain-invariant causal features and domain-specific non-causal
environmental features via adversarially training domain-specific environmental
feature extractors in the opposite domains. Subsequently, RED estimates and
reduces environmental disagreement based on domain-specific non-causal
environmental features. Experimental results confirm that RED effectively
mitigates negative transfer and achieves state-of-the-art performance.

### 5. [Graph-Guided Concept Selection for Efficient Retrieval-Augmented Generation](http://arxiv.org/pdf/2510.24120v1)

Authors: Ziyu Liu, Yijing Liu, Jianfei Yuan, Minzhi Yan, Le Yue, Honghui Xiong, Yi Yang

Graph-based RAG constructs a knowledge graph (KG) from text chunks to enhance
retrieval in Large Language Model (LLM)-based question answering. It is
especially beneficial in domains such as biomedicine, law, and political
science, where effective retrieval often involves multi-hop reasoning over
proprietary documents. However, these methods demand numerous LLM calls to
extract entities and relations from text chunks, incurring prohibitive costs at
scale. Through a carefully designed ablation study, we observe that certain
words (termed concepts) and their associated documents are more important.
Based on this insight, we propose Graph-Guided Concept Selection (G2ConS). Its
core comprises a chunk selection method and an LLM-independent concept graph.
The former selects salient document chunks to reduce KG construction costs; the
latter closes knowledge gaps introduced by chunk selection at zero cost.
Evaluations on multiple real-world datasets show that G2ConS outperforms all
baselines in construction cost, retrieval effectiveness, and answering quality.

### 6. [Causal Convolutional Neural Networks as Finite Impulse Response Filters](http://arxiv.org/pdf/2510.24125v1)

Authors: Kiran Bacsa, Wei Liu, Xudong Jian, Huangbin Liang, Eleni Chatzi

This study investigates the behavior of Causal Convolutional Neural Networks
(CNNs) with quasi-linear activation functions when applied to time-series data
characterized by multimodal frequency content. We demonstrate that, once
trained, such networks exhibit properties analogous to Finite Impulse Response
(FIR) filters, particularly when the convolutional kernels are of extended
length exceeding those typically employed in standard CNN architectures. Causal
CNNs are shown to capture spectral features both implicitly and explicitly,
offering enhanced interpretability for tasks involving dynamic systems.
Leveraging the associative property of convolution, we further show that the
entire network can be reduced to an equivalent single-layer filter resembling
an FIR filter optimized via least-squares criteria. This equivalence yields new
insights into the spectral learning behavior of CNNs trained on signals with
sparse frequency content. The approach is validated on both simulated beam
dynamics and real-world bridge vibration datasets, underlining its relevance
for modeling and identifying physical systems governed by dynamic responses.

### 7. [Fixed Point Neural Acceleration and Inverse Surrogate Model for Battery Parameter Identification](http://arxiv.org/pdf/2510.24135v1)

Authors: Hojin Cheon, Hyeongseok Seo, Jihun Jeon, Wooju Lee, Dohyun Jeong, Hongseok Kim

The rapid expansion of electric vehicles has intensified the need for
accurate and efficient diagnosis of lithium-ion batteries. Parameter
identification of electrochemical battery models is widely recognized as a
powerful method for battery health assessment. However, conventional
metaheuristic approaches suffer from high computational cost and slow
convergence, and recent machine learning methods are limited by their reliance
on constant current data, which may not be available in practice. To overcome
these challenges, we propose deep learning-based framework for parameter
identification of electrochemical battery models. The proposed framework
combines a neural surrogate model of the single particle model with electrolyte
(NeuralSPMe) and a deep learning-based fixed-point iteration method. NeuralSPMe
is trained on realistic EV load profiles to accurately predict lithium
concentration dynamics under dynamic operating conditions while a parameter
update network (PUNet) performs fixed-point iterative updates to significantly
reduce both the evaluation time per sample and the overall number of iterations
required for convergence. Experimental evaluations demonstrate that the
proposed framework accelerates the parameter identification by more than 2000
times, achieves superior sample efficiency and more than 10 times higher
accuracy compared to conventional metaheuristic algorithms, particularly under
dynamic load scenarios encountered in practical applications.

### 8. [Identifiable learning of dissipative dynamics](http://arxiv.org/pdf/2510.24160v1)

Authors: Aiqing Zhu, Beatrice W. Soh, Grigorios A. Pavliotis, Qianxiao Li

Complex dissipative systems appear across science and engineering, from
polymers and active matter to learning algorithms. These systems operate far
from equilibrium, where energy dissipation and time irreversibility are key to
their behavior, but are difficult to quantify from data. Learning accurate and
interpretable models of such dynamics remains a major challenge: the models
must be expressive enough to describe diverse processes, yet constrained enough
to remain physically meaningful and mathematically identifiable. Here, we
introduce I-OnsagerNet, a neural framework that learns dissipative stochastic
dynamics directly from trajectories while ensuring both interpretability and
uniqueness. I-OnsagerNet extends the Onsager principle to guarantee that the
learned potential is obtained from the stationary density and that the drift
decomposes cleanly into time-reversible and time-irreversible components, as
dictated by the Helmholtz decomposition. Our approach enables us to calculate
the entropy production and to quantify irreversibility, offering a principled
way to detect and quantify deviations from equilibrium. Applications to polymer
stretching in elongational flow and to stochastic gradient Langevin dynamics
reveal new insights, including super-linear scaling of barrier heights and
sub-linear scaling of entropy production rates with the strain rate, and the
suppression of irreversibility with increasing batch size. I-OnsagerNet thus
establishes a general, data-driven framework for discovering and interpreting
non-equilibrium dynamics.

### 9. [V-SAT: Video Subtitle Annotation Tool](http://arxiv.org/pdf/2510.24180v1)

Authors: Arpita Kundu, Joyita Chakraborty, Anindita Desarkar, Aritra Sen, Srushti Anil Patil, Vishwanathan Raman

The surge of audiovisual content on streaming platforms and social media has
heightened the demand for accurate and accessible subtitles. However, existing
subtitle generation methods primarily speech-based transcription or OCR-based
extraction suffer from several shortcomings, including poor synchronization,
incorrect or harmful text, inconsistent formatting, inappropriate reading
speeds, and the inability to adapt to dynamic audio-visual contexts. Current
approaches often address isolated issues, leaving post-editing as a
labor-intensive and time-consuming process. In this paper, we introduce V-SAT
(Video Subtitle Annotation Tool), a unified framework that automatically
detects and corrects a wide range of subtitle quality issues. By combining
Large Language Models(LLMs), Vision-Language Models (VLMs), Image Processing,
and Automatic Speech Recognition (ASR), V-SAT leverages contextual cues from
both audio and video. Subtitle quality improved, with the SUBER score reduced
from 9.6 to 3.54 after resolving all language mode issues and F1-scores of
~0.80 for image mode issues. Human-in-the-loop validation ensures high-quality
results, providing the first comprehensive solution for robust subtitle
annotation.

### 10. [Unlocking Out-of-Distribution Generalization in Dynamics through Physics-Guided Augmentation](http://arxiv.org/pdf/2510.24216v1)

Authors: Fan Xu, Hao Wu, Kun Wang, Nan Wang, Qingsong Wen, Xian Wu, Wei Gong, Xibin Zhao

In dynamical system modeling, traditional numerical methods are limited by
high computational costs, while modern data-driven approaches struggle with
data scarcity and distribution shifts. To address these fundamental
limitations, we first propose SPARK, a physics-guided quantitative augmentation
plugin. Specifically, SPARK utilizes a reconstruction autoencoder to integrate
physical parameters into a physics-rich discrete state dictionary. This state
dictionary then acts as a structured dictionary of physical states, enabling
the creation of new, physically-plausible training samples via principled
interpolation in the latent space. Further, for downstream prediction, these
augmented representations are seamlessly integrated with a Fourier-enhanced
Graph ODE, a combination designed to robustly model the enriched data
distribution while capturing long-term temporal dependencies. Extensive
experiments on diverse benchmarks demonstrate that SPARK significantly
outperforms state-of-the-art baselines, particularly in challenging
out-of-distribution scenarios and data-scarce regimes, proving the efficacy of
our physics-guided augmentation paradigm.

### Neural and Evolutionary Computing

### 1. [All in one timestep: Enhancing Sparsity and Energy efficiency in Multi-level Spiking Neural Networks](http://arxiv.org/pdf/2510.24637v1)

Authors: Andrea Castagnetti, Alain Pegatoquet, Benoît Miramond

Spiking Neural Networks (SNNs) are one of the most promising bio-inspired
neural networks models and have drawn increasing attention in recent years. The
event-driven communication mechanism of SNNs allows for sparse and
theoretically low-power operations on dedicated neuromorphic hardware. However,
the binary nature of instantaneous spikes also leads to considerable
information loss in SNNs, resulting in accuracy degradation. To address this
issue, we propose a multi-level spiking neuron model able to provide both
low-quantization error and minimal inference latency while approaching the
performance of full precision Artificial Neural Networks (ANNs). Experimental
results with popular network architectures and datasets, show that multi-level
spiking neurons provide better information compression, allowing therefore a
reduction in latency without performance loss. When compared to binary SNNs on
image classification scenarios, multi-level SNNs indeed allow reducing by 2 to
3 times the energy consumption depending on the number of quantization
intervals. On neuromorphic data, our approach allows us to drastically reduce
the inference latency to 1 timestep, which corresponds to a compression factor
of 10 compared to previously published results. At the architectural level, we
propose a new residual architecture that we call Sparse-ResNet. Through a
careful analysis of the spikes propagation in residual connections we highlight
a spike avalanche effect, that affects most spiking residual architectures.
Using our Sparse-ResNet architecture, we can provide state-of-the-art accuracy
results in image classification while reducing by more than 20% the network
activity compared to the previous spiking ResNets.

### 2. [HyperGraphX: Graph Transductive Learning with Hyperdimensional Computing and Message Passing](http://arxiv.org/pdf/2510.23980v1)

Authors: Guojing Cong, Tom Potok, Hamed Poursiami, Maryam Parsa

We present a novel algorithm, \hdgc, that marries graph convolution with
binding and bundling operations in hyperdimensional computing for transductive
graph learning. For prediction accuracy \hdgc outperforms major and popular
graph neural network implementations as well as state-of-the-art
hyperdimensional computing implementations for a collection of homophilic
graphs and heterophilic graphs. Compared with the most accurate learning
methodologies we have tested, on the same target GPU platform, \hdgc is on
average 9561.0 and 144.5 times faster than \gcnii, a graph neural network
implementation and HDGL, a hyperdimensional computing implementation,
respectively. As the majority of the learning operates on binary vectors, we
expect outstanding energy performance of \hdgc on neuromorphic and emerging
process-in-memory devices.

### 3. [Discovering Heuristics with Large Language Models (LLMs) for Mixed-Integer Programs: Single-Machine Scheduling](http://arxiv.org/pdf/2510.24013v1)

Authors: İbrahim Oğuz Çetinkaya, İ. Esra Büyüktahtakın, Parshin Shojaee, Chandan K. Reddy

Our study contributes to the scheduling and combinatorial optimization
literature with new heuristics discovered by leveraging the power of Large
Language Models (LLMs). We focus on the single-machine total tardiness (SMTT)
problem, which aims to minimize total tardiness by sequencing n jobs on a
single processor without preemption, given processing times and due dates. We
develop and benchmark two novel LLM-discovered heuristics, the EDD Challenger
(EDDC) and MDD Challenger (MDDC), inspired by the well-known Earliest Due Date
(EDD) and Modified Due Date (MDD) rules. In contrast to prior studies that
employed simpler rule-based heuristics, we evaluate our LLM-discovered
algorithms using rigorous criteria, including optimality gaps and solution time
derived from a mixed-integer programming (MIP) formulation of SMTT. We compare
their performance against state-of-the-art heuristics and exact methods across
various job sizes (20, 100, 200, and 500 jobs). For instances with more than
100 jobs, exact methods such as MIP and dynamic programming become
computationally intractable. Up to 500 jobs, EDDC improves upon the classic EDD
rule and another widely used algorithm in the literature. MDDC consistently
outperforms traditional heuristics and remains competitive with exact
approaches, particularly on larger and more complex instances. This study shows
that human-LLM collaboration can produce scalable, high-performing heuristics
for NP-hard constrained combinatorial optimization, even under limited
resources when effectively configured.

### Networking and Internet Architecture

### 1. [A New Hybrid Precoding Approach for Multi-user Massive MIMO over Fading Channels](http://arxiv.org/pdf/2510.24595v1)

Authors: Azadeh Pourkabirian, Kai Li, Photios A. Stavrou, Wei Ni

Hybrid precoding is an indispensable technique to harness the full potential
of a multi-user massive multiple-input, multiple-output (MU-MMIMO) system. In
this paper, we propose a new hybrid precoding approach that combines digital
and analog precoding to optimize data transmission over multiple antennas. This
approach steers signals in specific directions, leading to maximizing sum-rate
and suppressing side-lobe interference. When dealing with complex signals,
changes in phase are naturally associated with changes in angle, and these
variations are inherently correlated. The correlation between the angle and
phase is essential for accurately determining the channel characteristics. An
important aspect of this approach is that we model the angle and phase as
correlated variables following a bivariate Gaussian distribution, and for the
first time, we define a joint angle and phase entropy to measure the
uncertainty of angle and phase variations in wireless channels. This entropy is
crucial to adapt the proposed precoding method with variations. Simulation
result validate the accuracy of our analytical findings, demonstrating 18.31%
increase in sum-rate and an 11.47% improvement in robustness compared to other
state-of-the-art methods.

### 2. [Strategic Task Offloading for Delay-Sensitive IoT Applications: A Game-Theory-Based Demand-Supply Mechanism with Participation Incentives](http://arxiv.org/pdf/2510.24611v1)

Authors: Azadeh Pourkabirian, Amir Masoud Rahmani, Kai Li, Wei Ni

Delay-sensitive Internet of Things (IoT) applications have drawn significant
attention. Running many of these applications on IoT devices is challenging due
to the limited processing resources of these devices and the need for real-time
responses. Task offloading can minimize latency by transferring computationally
intensive tasks from IoT devices to resource-rich edge servers, ensuring delay
and performance guarantees. In this paper, we develop a task-offloading
approach for delay-sensitive IoT applications in edge computing environments.
Unlike existing schemes, we model the task offloading problem as an economic
demand and supply model to achieve market balance. The proposed model avoids
under- and over-supply, ensuring the computational resources at edge servers
(supply) are allocated in a manner that best meets the processing and
computational needs of user devices (demand). Given the multi-agent nature of
task offloading involving users and service providers with different
preferences and objectives, we design a game-theoretic framework using a
Vickrey-Clarke-Groves (VCG) auction. This framework analyzes agent interactions
and decision-making processes. Additionally, we develop an incentive mechanism
to encourage both parties to participate in the auction. The mechanism
maximizes user task offloading to edge servers and motivates edge servers to
share their computational resources, achieving profitability for both IoT users
and edge servers. Simulations demonstrate our method maximizes social welfare,
ensures truthfulness, maintains market balance, and provides latency guarantees
for delay-sensitive IoT applications.

### 3. [Uncovering Gaps Between RFC Updates and TCP/IP Implementations: LLM-Facilitated Differential Checks on Intermediate Representations](http://arxiv.org/pdf/2510.24408v1)

Authors: Yifan Wu, Xuewei Feng, Yuxiang Yang, Ke Xu

As the core of the Internet infrastructure, the TCP/IP protocol stack
undertakes the task of network data transmission. However, due to the
complexity of the protocol and the uncertainty of cross-layer interaction,
there are often inconsistencies between the implementation of the protocol
stack code and the RFC standard. This inconsistency may not only lead to
differences in protocol functions but also cause serious security
vulnerabilities. At present, with the continuous expansion of protocol stack
functions and the rapid iteration of RFC documents, it is increasingly
important to detect and fix these inconsistencies. With the rise of large
language models, researchers have begun to explore how to extract protocol
specifications from RFC documents through these models, including protocol
stack modeling, state machine extraction, text ambiguity analysis, and other
related content. However, existing methods rely on predefined patterns or
rule-based approaches that fail to generalize across different protocol
specifications. Automated and scalable detection of these inconsistencies
remains a significant challenge. In this study, we propose an automated
analysis framework based on LLM and differential models. By modeling the
iterative relationship of the protocol and based on the iterative update
relationship of the RFC standard, we perform incremental code function analysis
on different versions of kernel code implementations to automatically perform
code detection and vulnerability analysis. We conduct extensive evaluations to
validate the effectiveness of our framework, demonstrating its effectiveness in
identifying potential vulnerabilities caused by RFC code inconsistencies.

### 4. [Enabling Near-realtime Remote Sensing via Satellite-Ground Collaboration of Large Vision-Language Models](http://arxiv.org/pdf/2510.24242v1)

Authors: Zihan Li, Jiahao Yang, Yuxin Zhang, Zhe Chen, Yue Gao

Large vision-language models (LVLMs) have recently demonstrated great
potential in remote sensing (RS) tasks (e.g., disaster monitoring) conducted by
low Earth orbit (LEO) satellites. However, their deployment in real-world LEO
satellite systems remains largely unexplored, hindered by limited onboard
computing resources and brief satellite-ground contacts. We propose Grace, a
satellite-ground collaborative system designed for near-realtime LVLM inference
in RS tasks. Accordingly, we deploy compact LVLM on satellites for realtime
inference, but larger ones on ground stations (GSs) to guarantee end-to-end
performance. Grace is comprised of two main phases that are asynchronous
satellite-GS Retrieval-Augmented Generation (RAG), and a task dispatch
algorithm. Firstly, we still the knowledge archive of GS RAG to satellite
archive with tailored adaptive update algorithm during limited satellite-ground
data exchange period. Secondly, propose a confidence-based test algorithm that
either processes the task onboard the satellite or offloads it to the GS.
Extensive experiments based on real-world satellite orbital data show that
Grace reduces the average latency by 76-95% compared to state-of-the-art
methods, without compromising inference accuracy.

### Robotics

### 1. [A Comprehensive General Model of Tendon-Actuated Concentric Tube Robots with Multiple Tubes and Tendons](http://arxiv.org/pdf/2510.23954v1)

Authors: Pejman Kheradmand, Behnam Moradkhani, Raghavasimhan Sankaranarayanan, Kent K. Yamamoto, Tanner J. Zachem, Patrick J. Codd, Yash Chitalia, Pierre E. Dupont

Tendon-actuated concentric tube mechanisms combine the advantages of
tendon-driven continuum robots and concentric tube robots while addressing
their respective limitations. They overcome the restricted degrees of freedom
often seen in tendon-driven designs, and mitigate issues such as snapping
instability associated with concentric tube robots. However, a complete and
general mechanical model for these systems remains an open problem. In this
work, we propose a Cosserat rod-based framework for modeling the general case
of $n$ concentric tubes, each actuated by $m_i$ tendons, where $i = \{1,
\ldots, n\}$. The model allows each tube to twist and elongate while enforcing
a shared centerline for bending. We validate the proposed framework through
experiments with two-tube and three tube assemblies under various tendon
routing configurations, achieving tip prediction errors $<4\%$ of the robot's
total length. We further demonstrate the model's generality by applying it to
existing robots in the field, where maximum tip deviations remain around $5\%$
of the total length. This model provides a foundation for accurate shape
estimation and control of advanced tendon-actuated concentric tube robots.

### 2. [Adaptive-twist Soft Finger Mechanism for Grasping by Wrapping](http://arxiv.org/pdf/2510.23963v1)

Authors: Hiroki Ishikawa, Kyosuke Ishibashi, Ko Yamamoto

This paper presents a soft robot finger capable of adaptive-twist deformation
to grasp objects by wrapping them. For a soft hand to grasp and pick-up one
object from densely contained multiple objects, a soft finger requires the
adaptive-twist deformation function in both in-plane and out-of-plane
directions. The function allows the finger to be inserted deeply into a limited
gap among objects. Once inserted, the soft finger requires appropriate control
of grasping force normal to contact surface, thereby maintaining the twisted
deformation. In this paper, we refer to this type of grasping as grasping by
wrapping. To achieve these two functions by a single actuation source, we
propose a variable stiffness mechanism that can adaptively change the stiffness
as the pressure is higher. We conduct a finite element analysis (FEA) on the
proposed mechanism and determine its design parameter based on the FEA result.
Using the developed soft finger, we report basic experimental results and
demonstrations on grasping various objects.

### 3. [A Survey on Collaborative SLAM with 3D Gaussian Splatting](http://arxiv.org/pdf/2510.23988v1)

Authors: Phuc Nguyen Xuan, Thanh Nguyen Canh, Huu-Hung Nguyen, Nak Young Chong, Xiem HoangVan

This survey comprehensively reviews the evolving field of multi-robot
collaborative Simultaneous Localization and Mapping (SLAM) using 3D Gaussian
Splatting (3DGS). As an explicit scene representation, 3DGS has enabled
unprecedented real-time, high-fidelity rendering, ideal for robotics. However,
its use in multi-robot systems introduces significant challenges in maintaining
global consistency, managing communication, and fusing data from heterogeneous
sources. We systematically categorize approaches by their architecture --
centralized, distributed -- and analyze core components like multi-agent
consistency and alignment, communication-efficient, Gaussian representation,
semantic distillation, fusion and pose optimization, and real-time scalability.
In addition, a summary of critical datasets and evaluation metrics is provided
to contextualize performance. Finally, we identify key open challenges and
chart future research directions, including lifelong mapping, semantic
association and mapping, multi-model for robustness, and bridging the Sim2Real
gap.

### 4. [VOCALoco: Viability-Optimized Cost-aware Adaptive Locomotion](http://arxiv.org/pdf/2510.23997v1)

Authors: Stanley Wu, Mohamad H. Danesh, Simon Li, Hanna Yurchyk, Amin Abyaneh, Anas El Houssaini, David Meger, Hsiu-Chin Lin

Recent advancements in legged robot locomotion have facilitated traversal
over increasingly complex terrains. Despite this progress, many existing
approaches rely on end-to-end deep reinforcement learning (DRL), which poses
limitations in terms of safety and interpretability, especially when
generalizing to novel terrains. To overcome these challenges, we introduce
VOCALoco, a modular skill-selection framework that dynamically adapts
locomotion strategies based on perceptual input. Given a set of pre-trained
locomotion policies, VOCALoco evaluates their viability and energy-consumption
by predicting both the safety of execution and the anticipated cost of
transport over a fixed planning horizon. This joint assessment enables the
selection of policies that are both safe and energy-efficient, given the
observed local terrain. We evaluate our approach on staircase locomotion tasks,
demonstrating its performance in both simulated and real-world scenarios using
a quadrupedal robot. Empirical results show that VOCALoco achieves improved
robustness and safety during stair ascent and descent compared to a
conventional end-to-end DRL policy

### 5. [Balanced Collaborative Exploration via Distributed Topological Graph Voronoi Partition](http://arxiv.org/pdf/2510.24067v1)

Authors: Tianyi Ding, Ronghao Zheng, Senlin Zhang, Meiqin Liu

This work addresses the collaborative multi-robot autonomous online
exploration problem, particularly focusing on distributed exploration planning
for dynamically balanced exploration area partition and task allocation among a
team of mobile robots operating in obstacle-dense non-convex environments.
  We present a novel topological map structure that simultaneously
characterizes both spatial connectivity and global exploration completeness of
the environment. The topological map is updated incrementally to utilize known
spatial information for updating reachable spaces, while exploration targets
are planned in a receding horizon fashion under global coverage guidance.
  A distributed weighted topological graph Voronoi algorithm is introduced
implementing balanced graph space partitions of the fused topological maps.
Theoretical guarantees are provided for distributed consensus convergence and
equitable graph space partitions with constant bounds.
  A local planner optimizes the visitation sequence of exploration targets
within the balanced partitioned graph space to minimize travel distance, while
generating safe, smooth, and dynamically feasible motion trajectories.
  Comprehensive benchmarking against state-of-the-art methods demonstrates
significant improvements in exploration efficiency, completeness, and workload
balance across the robot team.

### 6. [Dynamically-Consistent Trajectory Optimization for Legged Robots via Contact Point Decomposition](http://arxiv.org/pdf/2510.24069v1)

Authors: Sangmin Kim, Hajun Kim, Gijeong Kim, Min-Gyu Kim, Hae-Won Park

To generate reliable motion for legged robots through trajectory
optimization, it is crucial to simultaneously compute the robot's path and
contact sequence, as well as accurately consider the dynamics in the problem
formulation. In this paper, we present a phase-based trajectory optimization
that ensures the feasibility of translational dynamics and friction cone
constraints throughout the entire trajectory. Specifically, our approach
leverages the superposition properties of linear differential equations to
decouple the translational dynamics for each contact point, which operates
under different phase sequences. Furthermore, we utilize the differentiation
matrix of B{\'e}zier polynomials to derive an analytical relationship between
the robot's position and force, thereby ensuring the consistent satisfaction of
translational dynamics. Additionally, by exploiting the convex closure property
of B{\'e}zier polynomials, our method ensures compliance with friction cone
constraints. Using the aforementioned approach, the proposed trajectory
optimization framework can generate dynamically reliable motions with various
gait sequences for legged robots. We validate our framework using a quadruped
robot model, focusing on the feasibility of dynamics and motion generation.

### 7. [PFEA: An LLM-based High-Level Natural Language Planning and Feedback Embodied Agent for Human-Centered AI](http://arxiv.org/pdf/2510.24109v1)

Authors: Wenbin Ding, Jun Chen, Mingjia Chen, Fei Xie, Qi Mao, Philip Dames

The rapid advancement of Large Language Models (LLMs) has marked a
significant breakthrough in Artificial Intelligence (AI), ushering in a new era
of Human-centered Artificial Intelligence (HAI). HAI aims to better serve human
welfare and needs, thereby placing higher demands on the intelligence level of
robots, particularly in aspects such as natural language interaction, complex
task planning, and execution. Intelligent agents powered by LLMs have opened up
new pathways for realizing HAI. However, existing LLM-based embodied agents
often lack the ability to plan and execute complex natural language control
tasks online. This paper explores the implementation of intelligent robotic
manipulating agents based on Vision-Language Models (VLMs) in the physical
world. We propose a novel embodied agent framework for robots, which comprises
a human-robot voice interaction module, a vision-language agent module and an
action execution module. The vision-language agent itself includes a
vision-based task planner, a natural language instruction converter, and a task
performance feedback evaluator. Experimental results demonstrate that our agent
achieves a 28\% higher average task success rate in both simulated and real
environments compared to approaches relying solely on LLM+CLIP, significantly
improving the execution success rate of high-level natural language instruction
tasks.

### 8. [Manipulate as Human: Learning Task-oriented Manipulation Skills by Adversarial Motion Priors](http://arxiv.org/pdf/2510.24257v1)

Authors: Ziqi Ma, Changda Tian, Yue Gao

In recent years, there has been growing interest in developing robots and
autonomous systems that can interact with human in a more natural and intuitive
way. One of the key challenges in achieving this goal is to enable these
systems to manipulate objects and tools in a manner that is similar to that of
humans. In this paper, we propose a novel approach for learning human-style
manipulation skills by using adversarial motion priors, which we name HMAMP.
The approach leverages adversarial networks to model the complex dynamics of
tool and object manipulation, as well as the aim of the manipulation task. The
discriminator is trained using a combination of real-world data and simulation
data executed by the agent, which is designed to train a policy that generates
realistic motion trajectories that match the statistical properties of human
motion. We evaluated HMAMP on one challenging manipulation task: hammering, and
the results indicate that HMAMP is capable of learning human-style manipulation
skills that outperform current baseline methods. Additionally, we demonstrate
that HMAMP has potential for real-world applications by performing real robot
arm hammering tasks. In general, HMAMP represents a significant step towards
developing robots and autonomous systems that can interact with humans in a
more natural and intuitive way, by learning to manipulate tools and objects in
a manner similar to how humans do.

### 9. [Global-State-Free Obstacle Avoidance for Quadrotor Control in Air-Ground Cooperation](http://arxiv.org/pdf/2510.24315v1)

Authors: Baozhe Zhang, Xinwei Chen, Qingcheng Chen, Chao Xu, Fei Gao, Yanjun Cao

CoNi-MPC provides an efficient framework for UAV control in air-ground
cooperative tasks by relying exclusively on relative states, eliminating the
need for global state estimation. However, its lack of environmental
information poses significant challenges for obstacle avoidance. To address
this issue, we propose a novel obstacle avoidance algorithm, Cooperative
Non-inertial frame-based Obstacle Avoidance (CoNi-OA), designed explicitly for
UAV-UGV cooperative scenarios without reliance on global state estimation or
obstacle prediction. CoNi-OA uniquely utilizes a single frame of raw LiDAR data
from the UAV to generate a modulation matrix, which directly adjusts the
quadrotor's velocity to achieve obstacle avoidance. This modulation-based
method enables real-time generation of collision-free trajectories within the
UGV's non-inertial frame, significantly reducing computational demands (less
than 5 ms per iteration) while maintaining safety in dynamic and unpredictable
environments. The key contributions of this work include: (1) a
modulation-based obstacle avoidance algorithm specifically tailored for UAV-UGV
cooperation in non-inertial frames without global states; (2) rapid, real-time
trajectory generation based solely on single-frame LiDAR data, removing the
need for obstacle modeling or prediction; and (3) adaptability to both static
and dynamic environments, thus extending applicability to featureless or
unknown scenarios.

### 10. [Supervisory Measurement-Guided Noise Covariance Estimation](http://arxiv.org/pdf/2510.24508v1)

Authors: Haoying Li, Yifan Peng, Junfeng Wu

Reliable state estimation hinges on accurate specification of sensor noise
covariances, which weigh heterogeneous measurements. In practice, these
covariances are difficult to identify due to environmental variability,
front-end preprocessing, and other reasons. We address this by formulating
noise covariance estimation as a bilevel optimization that, from a Bayesian
perspective, factorizes the joint likelihood of so-called odometry and
supervisory measurements, thereby balancing information utilization with
computational efficiency. The factorization converts the nested Bayesian
dependency into a chain structure, enabling efficient parallel computation: at
the lower level, an invariant extended Kalman filter with state augmentation
estimates trajectories, while a derivative filter computes analytical gradients
in parallel for upper-level gradient updates. The upper level refines the
covariance to guide the lower-level estimation. Experiments on synthetic and
real-world datasets show that our method achieves higher efficiency over
existing baselines.

### Software Engineering

### 1. [Validating Alerts in Cloud-Native Observability](http://arxiv.org/pdf/2510.23970v1)

Authors: Maria C. Borges, Julian Legler, Lucca Di Benedetto

Observability and alerting form the backbone of modern reliability
engineering. Alerts help teams catch faults early before they turn into
production outages and serve as first clues for troubleshooting. However,
designing effective alerts is challenging. They need to strike a fine balance
between catching issues early and minimizing false alarms. On top of this,
alerts often cover uncommon faults, so the code is rarely executed and
therefore rarely checked. To address these challenges, several industry
practitioners advocate for testing alerting code with the same rigor as
application code. Still, there's a lack of tools that support such systematic
design and validation of alerts.
  This paper introduces a new alerting extension for the observability
experimentation tool OXN. It lets engineers experiment with alerts early during
development. With OXN, engineers can now tune rules at design time and
routinely validate the firing behavior of their alerts, avoiding future
problems at runtime.

### 2. [Monitoring and Observability of Machine Learning Systems: Current Practices and Gaps](http://arxiv.org/pdf/2510.24142v1)

Authors: Joran Leest, Ilias Gerostathopoulos, Patricia Lago, Claudia Raibulet

Production machine learning (ML) systems fail silently -- not with crashes,
but through wrong decisions. While observability is recognized as critical for
ML operations, there is a lack empirical evidence of what practitioners
actually capture. This study presents empirical results on ML observability in
practice through seven focus group sessions in several domains. We catalog the
information practitioners systematically capture across ML systems and their
environment and map how they use it to validate models, detect and diagnose
faults, and explain observed degradations. Finally, we identify gaps in current
practice and outline implications for tooling design and research to establish
ML observability practices.

### 3. [Investigating Software Aging in LLM-Generated Software Systems](http://arxiv.org/pdf/2510.24188v1)

Authors: César Santos, Ermeson Andrade, Roberto Natella

Automatically generated software, especially code produced by Large Language
Models (LLMs), is increasingly adopted to accelerate development and reduce
manual effort. However, little is known about the long-term reliability of such
systems under sustained execution. In this paper, we experimentally investigate
the phenomenon of software aging in applications generated by LLM-based tools.
Using the Bolt platform and standardized prompts from Baxbench, we generated
four service-oriented applications and subjected them to 50-hour load tests.
Resource usage, response time, and throughput were continuously monitored to
detect degradation patterns. The results reveal significant evidence of
software aging, including progressive memory growth, increased response time,
and performance instability across all applications. Statistical analyzes
confirm these trends and highlight variability in the severity of aging
according to the type of application. Our findings show the need to consider
aging in automatically generated software and provide a foundation for future
studies on mitigation strategies and long-term reliability evaluation.

### 4. [LLM-as-a-Judge for Software Engineering: Literature Review, Vision, and the Road Ahead](http://arxiv.org/pdf/2510.24367v1)

Authors: Junda He, Jieke Shi, Terry Yue Zhuo, Christoph Treude, Jiamou Sun, Zhenchang Xing, Xiaoning Du, David Lo

The rapid integration of Large Language Models (LLMs) into software
engineering (SE) has revolutionized tasks like code generation, producing a
massive volume of software artifacts. This surge has exposed a critical
bottleneck: the lack of scalable, reliable methods to evaluate these outputs.
Human evaluation is costly and time-consuming, while traditional automated
metrics like BLEU fail to capture nuanced quality aspects. In response, the
LLM-as-a-Judge paradigm - using LLMs for automated evaluation - has emerged.
This approach leverages the advanced reasoning of LLMs, offering a path toward
human-like nuance at automated scale. However, LLM-as-a-Judge research in SE is
still in its early stages. This forward-looking SE 2030 paper aims to steer the
community toward advancing LLM-as-a-Judge for evaluating LLM-generated software
artifacts. We provide a literature review of existing SE studies, analyze their
limitations, identify key research gaps, and outline a detailed roadmap. We
envision these frameworks as reliable, robust, and scalable human surrogates
capable of consistent, multi-faceted artifact evaluation by 2030. Our work aims
to foster research and adoption of LLM-as-a-Judge frameworks, ultimately
improving the scalability of software artifact evaluation.

### 5. [CodeWiki: Automated Repository-Level Documentation at Scale](http://arxiv.org/pdf/2510.24428v1)

Authors: Nguyen Hoang Anh, Minh Le-Anh, Bach Le, Nghi D. Q. Bui

Developers spend nearly 58% of their time understanding codebases, yet
maintaining comprehensive documentation remains challenging due to complexity
and manual effort. While recent Large Language Models (LLMs) show promise for
function-level documentation, they fail at the repository level, where
capturing architectural patterns and cross-module interactions is essential. We
introduce CodeWiki, the first open-source framework for holistic
repository-level documentation across seven programming languages. CodeWiki
employs three innovations: (i) hierarchical decomposition that preserves
architectural context, (ii) recursive agentic processing with dynamic
delegation, and (iii) synthesis of textual and visual artifacts including
architecture diagrams and data flows. We also present CodeWikiBench, the first
repository-level documentation benchmark with multi-level rubrics and agentic
assessment. CodeWiki achieves 68.79% quality score with proprietary models and
64.80% with open-source alternatives, outperforming existing closed-source
systems and demonstrating scalable, accurate documentation for real-world
repositories.

### 6. [The Divine Software Engineering Comedy -- Inferno: The Okinawa Files](http://arxiv.org/pdf/2510.24483v1)

Authors: Michele Lanza

In June 2024 I co-organized the FUture of Software Engineering symposium in
Okinawa, Japan. Me, Andrian Marcus, Takashi Kobayashi and Shinpei Hayashi were
general chairs, Nicole Novielli, Kevin Moran, Yutaro Kashiwa and Masanari Kondo
were program chairs, some members of my group, Carmen Armenti, Stefano
Campanella, Roberto Minelli, were the tables, can't have a room with only
chairs, after all. We invited a crowd of people to discuss what future software
engineering has. FUSE became a 3-day marathon on whether there is actually a
future at all for SE. This essay is a slightly dark take about what I saw at
that event, very loosely based on the discussions that took place, adding some
healthy sarcasm and cynicism, the intellectual salt and pepper I never seem to
run out of. I listened to the brilliant people who gathered to talk about where
we're headed, and distilled three nightmares headed in our direction: software
makers who don't know what they're doing, but get the job done anyway, a field
moving so fast it can't remember its own lessons, and technologies multiplying
like rabbits in Spring. So, let's start. The future, eh? The future of software
engineering looks like a car crash in slow motion: you can see it coming but
you can't look away. The thing is...

### 7. [A Pragmatic Way to Measure Chain-of-Thought Monitorability](http://arxiv.org/pdf/2510.23966v1)

Authors: Scott Emmons, Roland S. Zimmermann, David K. Elson, Rohin Shah

While Chain-of-Thought (CoT) monitoring offers a unique opportunity for AI
safety, this opportunity could be lost through shifts in training practices or
model architecture. To help preserve monitorability, we propose a pragmatic way
to measure two components of it: legibility (whether the reasoning can be
followed by a human) and coverage (whether the CoT contains all the reasoning
needed for a human to also produce the final output). We implement these
metrics with an autorater prompt that enables any capable LLM to compute the
legibility and coverage of existing CoTs. After sanity-checking our prompted
autorater with synthetic CoT degradations, we apply it to several frontier
models on challenging benchmarks, finding that they exhibit high
monitorability. We present these metrics, including our complete autorater
prompt, as a tool for developers to track how design decisions impact
monitorability. While the exact prompt we share is still a preliminary version
under ongoing development, we are sharing it now in the hopes that others in
the community will find it useful. Our method helps measure the default
monitorability of CoT - it should be seen as a complement, not a replacement,
for the adversarial stress-testing needed to test robustness against
deliberately evasive models.

### 8. [Lifecycle-Aware code generation: Leveraging Software Engineering Phases in LLMs](http://arxiv.org/pdf/2510.24019v1)

Authors: Xing Xing, Wei Wang, Lipeng Ma, Weidong Yang, Junjie Zheng

Recent progress in large language models (LLMs) has advanced automatic code
generation, yet most approaches rely on direct, single-step translation from
problem descriptions to code, disregarding structured software engineering
practices. We introduce a lifecycle-aware framework that systematically
incorporates intermediate artifacts such as requirements analysis, state
machine modeling, and pseudocode into both the training and inference stages.
This design aligns code generation with standard software development phases
and enables more structured reasoning. Experiments show that lifecycle-level
fine-tuning improves code correctness by up to 75% over the same model before
fine-tuning, with performance gains compounding across intermediate stages.
Multi-step inference consistently surpasses single-step generation,
demonstrating the effectiveness of intermediate scaffolding. Notably,
open-source LLMs, once fine-tuned under our framework, match or slightly
outperform models pretrained on code. When applied to DeepSeek-Coder-1.3B, our
framework yields relative CodeBLEU improvements of 34.3%, 20.0%, 11.2%, and
22.3% over ChatGPT-3.5, ChatGPT-4o-mini, DeepSeek-R1, and LLaMA-8B,
respectively. Our pipeline also proves robust with up to 80\% less training
data, confirming its resilience. Ablation studies further reveal that each
intermediate artifact contributes distinctly to final code quality, with state
machine modeling yielding the most substantial impact. Our source code and
detailed experimental data are available at
https://anonymous.4open.science/r/Lifecycle-Aware-3CCB.

### 9. [MAGNET: A Multi-Graph Attentional Network for Code Clone Detection](http://arxiv.org/pdf/2510.24241v1)

Authors: Zixian Zhang, Takfarinas Saber

Code clone detection is a fundamental task in software engineering that
underpins refactoring, debugging, plagiarism detection, and vulnerability
analysis. Existing methods often rely on singular representations such as
abstract syntax trees (ASTs), control flow graphs (CFGs), and data flow graphs
(DFGs), which capture only partial aspects of code semantics. Hybrid approaches
have emerged, but their fusion strategies are typically handcrafted and
ineffective. In this study, we propose MAGNET, a multi-graph attentional
framework that jointly leverages AST, CFG, and DFG representations to capture
syntactic and semantic features of source code. MAGNET integrates residual
graph neural networks with node-level self-attention to learn both local and
long-range dependencies, introduces a gated cross-attention mechanism for
fine-grained inter-graph interactions, and employs Set2Set pooling to fuse
multi-graph embeddings into unified program-level representations. Extensive
experiments on BigCloneBench and Google Code Jam demonstrate that MAGNET
achieves state-of-the-art performance with an overall F1 score of 96.5\% and
99.2\% on the two datasets, respectively. Ablation studies confirm the critical
contributions of multi-graph fusion and each attentional component. Our code is
available at https://github.com/ZixianReid/Multigraph_match

### 10. [Developer Productivity with GenAI](http://arxiv.org/pdf/2510.24265v1)

Authors: Sadia Afroz, Zixuan Feng, Katie Kimura, Bianca Trinkenreich, Igor Steinmacher, Anita Sarma

Generative AI (GenAI) tools are increasingly being adopted in software
development as productivity aids. However, evidence regarding where and when
these tools actually enhance productivity is unclear. In this paper, we
investigate how GenAI adoption affects different dimensions of developer
productivity. We surveyed 415 software practitioners to capture their
perceptions of productivity changes associated with AI-assisted development
using the SPACE framework - Satisfaction and well-being, Performance, Activity,
Communication and collaboration, and Efficiency and flow. Our results,
disaggregated by frequency of AI usage, reveal limited overall productivity
change, highlighting the productivity paradox in which developers become faster
but do not necessarily create better software or feel more fulfilled.

### Social and Information Networks

### 1. [GRAPHIA: Harnessing Social Graph Data to Enhance LLM-Based Social Simulation](http://arxiv.org/pdf/2510.24251v1)

Authors: Jiarui Ji, Zehua Zhang, Zhewei Wei, Bin Tong, Guan Wang, Bo Zheng

Large language models (LLMs) have shown promise in simulating human-like
social behaviors. Social graphs provide high-quality supervision signals that
encode both local interactions and global network structure, yet they remain
underutilized for LLM training. To address this gap, we propose Graphia, the
first general LLM-based social graph simulation framework that leverages graph
data as supervision for LLM post-training via reinforcement learning. With
GNN-based structural rewards, Graphia trains specialized agents to predict whom
to interact with (destination selection) and how to interact (edge generation),
followed by designed graph generation pipelines. We evaluate Graphia under two
settings: Transductive Dynamic Graph Generation (TDGG), a micro-level task with
our proposed node-wise interaction alignment metrics; and Inductive Dynamic
Graph Generation (IDGG), a macro-level task with our proposed metrics for
aligning emergent network properties. On three real-world networks, Graphia
improves micro-level alignment by 6.1% in the composite destination selection
score, 12% in edge classification accuracy, and 27.9% in edge content BERTScore
over the strongest baseline. For macro-level alignment, it achieves 41.11%
higher structural similarity and 32.98% better replication of social phenomena
such as power laws and echo chambers. Graphia also supports counterfactual
simulation, generating plausible behavioral shifts under platform incentives.
Our results show that social graphs can serve as high-quality supervision
signals for LLM post-training, closing the gap between agent behaviors and
network dynamics for LLM-based simulation. Code is available at
https://github.com/Ji-Cather/Graphia.git.

### 2. [Importance of Overlapping Network Nodes in Influence Spreading](http://arxiv.org/pdf/2510.24360v1)

Authors: Kosti Koistinen, Vesa Kuikka, Kimmo Kaski

In complex networks there are overlapping substructures or "circles" that
consist of nodes belonging to multiple cohesive subgroups. Yet the role of
these overlapping nodes in influence spreading processes remains underexplored.
In the present study, we analyse networks with circle structures using a
probabilistic influence spreading model for processes of simple and complex
contagion. We quantify the roles of nodes using three metrics, i.e.,
In-Centrality, Out-Centrality, and Betweenness Centrality that represent the
susceptibility, spreading power, and mediatory role of nodes, respectively, and
find that at each stage of the spreading process the overlapping nodes
consistently exhibit greater influence than the non-overlapping ones.
Furthermore, we observe that the criteria to define circles shape the
overlapping effects. When we restrict our analysis to only largest circles, we
find that circles reflect not only node-level attributes but also of
topological importance. These findings clarify the distinction between local
attribute-driven circles and global community structures, thus highlighting the
strategic importanc of overlapping nodes in spreading dynamics. This provides
foundation for future research on overlapping nodes in both circles and
communities.

### 3. [Assessing the influence of social media feedback on traveler's future trip-planning behavior: A multi-model machine learning approach](http://arxiv.org/pdf/2510.24077v1)

Authors: Sayantan Mukherjee, Pritam Ranjan, Joysankar Bhattacharya

With the surge of domestic tourism in India and the influence of social media
on young tourists, this paper aims to address the research question on how
"social return" - responses received on social media sharing - of recent trip
details can influence decision-making for short-term future travels. The paper
develops a multi-model framework to build a predictive machine learning model
that establishes a relationship between a traveler's social return, various
social media usage, trip-related factors, and her future trip-planning
behavior. The primary data was collected via a survey from Indian tourists.
After data cleaning, the imbalance in the data was addressed using a robust
oversampling method, and the reliability of the predictive model was ensured by
applying a Monte Carlo cross-validation technique. The results suggest at least
75% overall accuracy in predicting the influence of social return on changing
the future trip plan. Moreover, the model fit results provide crucial practical
implications for the domestic tourism sector in India with future research
directions concerning social media, destination marketing, smart tourism,
heritage tourism, etc.

### 4. [Generative Large Language Models (gLLMs) in Content Analysis: A Practical Guide for Communication Research](http://arxiv.org/pdf/2510.24337v1)

Authors: Daria Kravets-Meinke, Hannah Schmid-Petri, Sonja Niemann, Ute Schmid

Generative Large Language Models (gLLMs), such as ChatGPT, are increasingly
being used in communication research for content analysis. Studies show that
gLLMs can outperform both crowd workers and trained coders, such as research
assistants, on various coding tasks relevant to communication science, often at
a fraction of the time and cost. Additionally, gLLMs can decode implicit
meanings and contextual information, be instructed using natural language,
deployed with only basic programming skills, and require little to no annotated
data beyond a validation dataset - constituting a paradigm shift in automated
content analysis. Despite their potential, the integration of gLLMs into the
methodological toolkit of communication research remains underdeveloped. In
gLLM-assisted quantitative content analysis, researchers must address at least
seven critical challenges that impact result quality: (1) codebook development,
(2) prompt engineering, (3) model selection, (4) parameter tuning, (5)
iterative refinement, (6) validation of the model's reliability, and
optionally, (7) performance enhancement. This paper synthesizes emerging
research on gLLM-assisted quantitative content analysis and proposes a
comprehensive best-practice guide to navigate these challenges. Our goal is to
make gLLM-based content analysis more accessible to a broader range of
communication researchers and ensure adherence to established disciplinary
quality standards of validity, reliability, reproducibility, and research
ethics.

### 5. [Rewarding Engagement and Personalization in Popularity-Based Rankings Amplifies Extremism and Polarization](http://arxiv.org/pdf/2510.24354v1)

Authors: Jacopo D'Ignazi, Andreas Kaltenbrunner, Gaël Le Mens, Fabrizio Germano, Vicenç Gómez

Despite extensive research, the mechanisms through which online platforms
shape extremism and polarization remain poorly understood. We identify and test
a mechanism, grounded in empirical evidence, that explains how ranking
algorithms can amplify both phenomena. This mechanism is based on
well-documented assumptions: (i) users exhibit position bias and tend to prefer
items displayed higher in the ranking, (ii) users prefer like-minded content,
(iii) users with more extreme views are more likely to engage actively, and
(iv) ranking algorithms are popularity-based, assigning higher positions to
items that attract more clicks. Under these conditions, when platforms
additionally reward \emph{active} engagement and implement \emph{personalized}
rankings, users are inevitably driven toward more extremist and polarized news
consumption. We formalize this mechanism in a dynamical model, which we
evaluate by means of simulations and interactive experiments with hundreds of
human participants, where the rankings are updated dynamically in response to
user activity.

### 6. [Pair Approximation Meets Reality: Diffusion of Innovation in Organizational Networks within the biased-independence q-Voter Model](http://arxiv.org/pdf/2510.24447v1)

Authors: Angelika Abramiuk-Szurlej, Katarzyna Sznajd-Weron

Collective adaptation, whether in innovation adoption, pro-environmental or
organizational change, emerges from the interplay between individual decisions
and social influence. Agent-based modeling provides a useful tool for studying
such processes. Here, we introduce the biased-independence $q$-voter model, a
generalization of the $q$-voter model with independence, one of the most
popular agent-based models of opinion dynamics. In our model, individuals
choose between two options, adopt or not adopt, under the competing influences
of conformity and independent choice. Independent choice between two options is
determined by an engagement parameter, inspired by earlier agent-based model of
eco-innovation diffusion. When the engagement parameter equals $0.5$, the model
reduces to the original $q$-voter model with independence; values different
from $0.5$ break the symmetry between the two options. To place our study in a
broader context, we briefly review asymmetric versions of the $q$-voter model
proposed to date. The novelty of this work goes beyond introducing a
generalized model: we develop the pair approximation (PA) for an asymmetric
$q$-voter model and, for the first time, validate it on empirical
organizational networks. Our results show that the interplay of social
influence, independence, and option preference generates discontinuous phase
transitions and irreversible hysteresis, reflecting path-dependent adoption
dynamics. Surprisingly, the PA agrees well with Monte Carlo simulations on some
empirical networks, even small ones, highlighting its potential as a
computationally efficient bridge between individual decision-making and
collective actions.

### 7. [Exploring Emergent Topological Properties in Socio-Economic Networks through Learning Heterogeneity](http://arxiv.org/pdf/2510.24107v1)

Authors: Chanuka Karavita, Zehua Lyu, Dharshana Kasthurirathna, Mahendra Piraveenan

Understanding how individual learning behavior and structural dynamics
interact is essential to modeling emergent phenomena in socioeconomic networks.
While bounded rationality and network adaptation have been widely studied, the
role of heterogeneous learning rates both at the agent and network levels
remains under explored. This paper introduces a dual-learning framework that
integrates individualized learning rates for agents and a rewiring rate for the
network, reflecting real-world cognitive diversity and structural adaptability.
  Using a simulation model based on the Prisoner's Dilemma and Quantal Response
Equilibrium, we analyze how variations in these learning rates affect the
emergence of large-scale network structures. Results show that lower and more
homogeneously distributed learning rates promote scale-free networks, while
higher or more heterogeneously distributed learning rates lead to the emergence
of core-periphery topologies. Key topological metrics including scale-free
exponents, Estrada heterogeneity, and assortativity reveal that both the speed
and variability of learning critically shape system rationality and network
architecture. This work provides a unified framework for examining how
individual learnability and structural adaptability drive the formation of
socioeconomic networks with diverse topologies, offering new insights into
adaptive behavior, systemic organization, and resilience.

### Systems and Control

### 1. [Sample-based Moving Horizon Estimation](http://arxiv.org/pdf/2510.24191v1)

Authors: Isabelle Krauss, Victor G. Lopez, Matthias A. Müller

In this paper, we propose a sample-based moving horizon estimation (MHE)
scheme for general nonlinear systems to estimate the current system state using
irregularly and/or infrequently available measurements. The cost function of
the MHE optimization problem is suitably designed to accommodate these
irregular output sequences. We also establish that, under a suitable
sample-based detectability condition known as sample-based incremental
input/output-to-state stability (i-IOSS), the proposed sample-based MHE
achieves robust global exponential stability (RGES). Additionally, for the case
of linear systems, we draw connections between sample-based observability and
sample-based i-IOSS. This demonstrates that previously established conditions
for linear systems to be sample-based observable can be utilized to verify or
design sampling strategies that satisfy the conditions to guarantee RGES of the
sample-based MHE. Finally, the effectiveness of the proposed sample-based MHE
is illustrated through a simulation example.

### 2. [Mechanism-Guided Residual Lifting and Control Consistent Modeling for Pneumatic Drying Processes](http://arxiv.org/pdf/2510.24370v1)

Authors: Yue Wu

Pneumatic drying processes in industries such as agriculture, chemicals,and
pharmaceuticals are notoriously difficult to model and control due to
multi-source disturbances,coupled stage dynamics, and significant measurement
delays. Traditional modeling paradigms often fail to simultaneously deliver
accuracy, interpretability, and closed-loop applicability. To address this
challenge, this paper introduces a unified hybrid modeling framework, termed
Physics-Guided Residual Lifting with Control-Consistent Correction,which
integrates a transient mechanistic model with a stability-constrained
data-driven component. The framework covers the complete process chain of
drying, transport, and winnowing. On the mechanistic level, the model unifies
mass transfer dynamics using the partial pressure difference of water vapor,
incorporates water activity clamping and latent heat corrections for bound
water, and ensures energy closure with moisture-dependent specific heat. On the
data-driven level,we propose an orthogonal residual learning scheme. It
leverages intermediate states from the mechanistic model as proxy variables to
construct a physics-inspired dictionary, preventing parameter compensation and
overfitting during ridge regression. Furthermore, to ensure suitability for
predictive control, a Control-Consistent Extended Dynamic Mode Decomposition
with stability constraints is employed to learn the residual dynamics, for
which we provide boundedness proofs and stability guarantees. The framework was
validated on 10 industrial batches, comprising 63,000 samples. On unseen test
data, the hybrid model achieved a Mean Absolute Error of 0.016% for outlet
moisture and 0.015 {\deg}C for outlet temperature, with values improving to
0.986 and 0.995, respectively. The resulting prediction residuals exhibit
white-noise characteristics, with significantly reduced spectral energy at low
frequencies.

### 3. [Development of a Digital Twin for an Electric Vehicle Emulator Modeling, Control, and Experimental Validation](http://arxiv.org/pdf/2510.24389v1)

Authors: Lamine Chalal, Ahmed Rachid

This paper presents the development and validation of a digital twin for a
scaled-down electric vehicle (EV) emulator, designed to replicate longitudinal
vehicle dynamics under diverse operating conditions. The emulator integrates a
separately excited DC motor (SEDCM), a four-quadrant DC-DC converter, a battery
emulator, and a mechanical load emulator. The system models tractive effort,
aerodynamic drag, and gradient resistance using Newton's second law. In
contrast to conventional graphical modeling tools (e.g., block diagrams and
bond graphs), the adopted Energetic Macroscopic Representation (EMR) framework
offers clear advantages by explicitly representing energy interactions and
facilitating the systematic derivation of control structures. A control
strategy developed within this framework governs energy flow across the
powertrain, enabling accurate speed control via armature voltage regulation.
Experimental tests conducted on a Lucas-Nulle test bench show strong
correlation with simulation results. The study also introduces a methodology to
compute the maximum admissible vehicle mass - determined to be 13.5 kg for a
180 W motor operating at 1900 rpm - based on acceleration and slope
constraints. Furthermore, a switching algorithm for the bidirectional converter
ensures reliable four quadrant operation. Overall, the proposed framework
provides a scalable and effective approach for EV emulation, control design,
and energy management validation.

### 4. [Contributions to Semialgebraic-Set-Based Stability Verification of Dynamical Systems with Neural-Network-Based Controllers](http://arxiv.org/pdf/2510.24391v1)

Authors: Alvaro Detailleur, Dalim Wahby, Guillaume Ducard, Christopher Onder

Neural-network-based controllers (NNCs) can represent complex, highly
nonlinear control laws, but verifying the closed-loop stability of dynamical
systems using them remains challenging. This work presents contributions to a
state-of-the-art stability verification procedure for NNC-controlled systems
which relies on semialgebraic-set-based input-output modeling to pose the
search for a Lyapunov function as an optimization problem. Specifically, this
procedure's conservatism when analyzing NNCs using transcendental activation
functions and the restriction to feedforward NNCs are addressed by a)
introducing novel semialgebraic activation functions that preserve key
properties of common transcendental activations and b) proving compatibility of
NNCs from the broader class of recurrent equilibrium networks (RENs) with this
procedure. Furthermore, the indirect optimization of a local region of
attraction (RoA) estimate using a restricted set of candidate Lyapunov
functions is greatly improved via c) the introduction of a richer
parameterization of candidate Lyapunov functions than previously reported and
d) the formulation of novel semidefinite programs (SDPs) that directly optimize
the resulting RoA estimate. The value of these contributions is highlighted in
two numerical examples.

### 5. [Analyzing Parametric Oscillator Ising Machines through the Kuramoto Lens](http://arxiv.org/pdf/2510.24416v1)

Authors: Nikhat Khan, E. M. H. E. B. Ekanayake, Nicolas Casilli, Cristian Cassella, Luke Theogarajan, Nikhil Shukla

Networks of coupled nonlinear oscillators are emerging as powerful physical
platforms for implementing Ising machines. Yet the relationship between
parametric-oscillator implementations and traditional oscillator-based Ising
machines remains underexplored. In this work, we develop a Kuramoto-style,
canonical phase description of parametric oscillator Ising machines by starting
from the Stuart-Landau oscillator model -- the canonical normal form near a
Hopf bifurcation, and a natural reduced description for many parametric
oscillator implementations such as the degenerate optical parametric oscillator
(DOPO) among others. The resulting phase dynamics combine the usual
phase-difference coupling observed in the standard Kuramoto model along with an
intrinsic phase sum term that is generated when conjugate coupling is
considered. Moreover, our formulation helps explain why explicit
second-harmonic driving is unnecessary in parametric oscillators and also
reveals how quasi-steady amplitude heterogeneity scales the original strength
of the spin interaction with potentially adverse impacts on the solution
quality. Our work helps develop a unifying view of the oscillator-based
approach to designing Ising machines.

### 6. [Survey and Tutorial of Reinforcement Learning Methods in Process Systems Engineering](http://arxiv.org/pdf/2510.24272v1)

Authors: Maximilian Bloor, Max Mowbray, Ehecatl Antonio Del Rio Chanona, Calvin Tsay

Sequential decision making under uncertainty is central to many Process
Systems Engineering (PSE) challenges, where traditional methods often face
limitations related to controlling and optimizing complex and stochastic
systems. Reinforcement Learning (RL) offers a data-driven approach to derive
control policies for such challenges. This paper presents a survey and tutorial
on RL methods, tailored for the PSE community. We deliver a tutorial on RL,
covering fundamental concepts and key algorithmic families including
value-based, policy-based and actor-critic methods. Subsequently, we survey
existing applications of these RL techniques across various PSE domains, such
as in fed-batch and continuous process control, process optimization, and
supply chains. We conclude with PSE focused discussion of specialized
techniques and emerging directions. By synthesizing the current state of RL
algorithm development and implications for PSE this work identifies successes,
challenges, trends, and outlines avenues for future research at the interface
of these fields.

### 7. [Flatness-based trajectory planning for 3D overhead cranes with friction compensation and collision avoidance](http://arxiv.org/pdf/2510.24457v1)

Authors: Jorge Vicente-Martinez, Edgar Ramirez-Laboreo

This paper presents an optimal trajectory generation method for 3D overhead
cranes by leveraging differential flatness. This framework enables the direct
inclusion of complex physical and dynamic constraints, such as nonlinear
friction and collision avoidance for both payload and rope. Our approach allows
for aggressive movements by constraining payload swing only at the final point.
A comparative simulation study validates our approach, demonstrating that
neglecting dry friction leads to actuator saturation and collisions. The
results show that friction modeling is a fundamental requirement for fast and
safe crane trajectories.

### 8. [Efficient Network Reconfiguration by Randomized Switching](http://arxiv.org/pdf/2510.24458v1)

Authors: Samuel Talkington, Dmitrii M. Ostrovskii, Daniel K. Molzahn

We present an algorithm that efficiently computes nearly-optimal solutions to
a class of combinatorial reconfiguration problems on weighted, undirected
graphs. Inspired by societally relevant applications in networked
infrastructure systems, these problems consist of simultaneously finding an
unreweighted sparsified graph and nodal potentials that satisfy fixed demands,
where the objective is to minimize some congestion criterion, e.g., a Laplacian
quadratic form. These are mixed-integer nonlinear programming problems that are
NP-hard in general. To circumvent these challenges, instead of solving for a
single best configuration, the proposed randomized switching algorithm seeks to
design a distribution of configurations that, when sampled, ensures that
congestion concentrates around its optimum. We show that the proposed
congestion metric is a generalized self-concordant function in the space of
switching probabilities, which enables the use of efficient and simple
conditional gradient methods. We implement our algorithm and show that it
outperforms a state-of-the-art commercial mixed-integer second-order cone
programming (MISOCP) solver by orders of magnitude over a large range of
problem sizes.

### 9. [Learning to Drive Safely with Hybrid Options](http://arxiv.org/pdf/2510.24674v1)

Authors: Bram De Cooman, Johan Suykens

Out of the many deep reinforcement learning approaches for autonomous
driving, only few make use of the options (or skills) framework. That is
surprising, as this framework is naturally suited for hierarchical control
applications in general, and autonomous driving tasks in specific. Therefore,
in this work the options framework is applied and tailored to autonomous
driving tasks on highways. More specifically, we define dedicated options for
longitudinal and lateral manoeuvres with embedded safety and comfort
constraints. This way, prior domain knowledge can be incorporated into the
learning process and the learned driving behaviour can be constrained more
easily. We propose several setups for hierarchical control with options and
derive practical algorithms following state-of-the-art reinforcement learning
techniques. By separately selecting actions for longitudinal and lateral
control, the introduced policies over combined and hybrid options obtain the
same expressiveness and flexibility that human drivers have, while being easier
to interpret than classical policies over continuous actions. Of all the
investigated approaches, these flexible policies over hybrid options perform
the best under varying traffic conditions, outperforming the baseline policies
over actions.

### 10. [Feature Matching-Based Gait Phase Prediction for Obstacle Crossing Control of Powered Transfemoral Prosthesis](http://arxiv.org/pdf/2510.24676v1)

Authors: Jiaxuan Zhang, Yuquan Leng, Yixuan Guo, Chenglong Fu

For amputees with powered transfemoral prosthetics, navigating obstacles or
complex terrain remains challenging. This study addresses this issue by using
an inertial sensor on the sound ankle to guide obstacle-crossing movements. A
genetic algorithm computes the optimal neural network structure to predict the
required angles of the thigh and knee joints. A gait progression prediction
algorithm determines the actuation angle index for the prosthetic knee motor,
ultimately defining the necessary thigh and knee angles and gait progression.
Results show that when the standard deviation of Gaussian noise added to the
thigh angle data is less than 1, the method can effectively eliminate noise
interference, achieving 100\% accuracy in gait phase estimation under 150 Hz,
with thigh angle prediction error being 8.71\% and knee angle prediction error
being 6.78\%. These findings demonstrate the method's ability to accurately
predict gait progression and joint angles, offering significant practical value
for obstacle negotiation in powered transfemoral prosthetics.

### Machine Learning (Statistics Category)

### 1. [Copula-Stein Discrepancy: A Generator-Based Stein Operator for Archimedean Dependence](http://arxiv.org/pdf/2510.24056v1)

Authors: Agnideep Aich, Ashit Baran Aich

Kernel Stein discrepancies (KSDs) have become a principal tool for
goodness-of-fit testing, but standard KSDs are often insensitive to
higher-order dependency structures, such as tail dependence, which are critical
in many scientific and financial domains. We address this gap by introducing
the Copula-Stein Discrepancy (CSD), a novel class of discrepancies tailored to
the geometry of statistical dependence. By defining a Stein operator directly
on the copula density, CSD leverages the generative structure of dependence,
rather than relying on the joint density's score function. For the broad class
of Archimedean copulas, this approach yields a closed-form Stein kernel derived
from the scalar generator function. We provide a comprehensive theoretical
analysis, proving that CSD (i) metrizes weak convergence of copula
distributions, ensuring it detects any mismatch in dependence; (ii) has an
empirical estimator that converges at the minimax optimal rate of
$O_P(n^{-1/2})$; and (iii) is provably sensitive to differences in tail
dependence coefficients. The framework is extended to general non-Archimedean
copulas, including elliptical and vine copulas. Computationally, the exact CSD
kernel evaluation scales linearly in dimension, while a novel random feature
approximation reduces the $n$-dependence from quadratic $O(n^2)$ to near-linear
$\tilde{O}(n)$, making CSD a practical and theoretically principled tool for
dependence-aware inference.

### 2. [Self-Concordant Perturbations for Linear Bandits](http://arxiv.org/pdf/2510.24187v1)

Authors: Lucas Lévy, Jean-Lou Valeau, Arya Akhavan, Patrick Rebeschini

We study the adversarial linear bandits problem and present a unified
algorithmic framework that bridges Follow-the-Regularized-Leader (FTRL) and
Follow-the-Perturbed-Leader (FTPL) methods, extending the known connection
between them from the full-information setting. Within this framework, we
introduce self-concordant perturbations, a family of probability distributions
that mirror the role of self-concordant barriers previously employed in the
FTRL-based SCRiBLe algorithm. Using this idea, we design a novel FTPL-based
algorithm that combines self-concordant regularization with efficient
stochastic exploration. Our approach achieves a regret of $O(d\sqrt{n \ln n})$
on both the $d$-dimensional hypercube and the Euclidean ball. On the Euclidean
ball, this matches the rate attained by existing self-concordant FTRL methods.
For the hypercube, this represents a $\sqrt{d}$ improvement over these methods
and matches the optimal bound up to logarithmic factors.

### 3. [Comparison of generalised additive models and neural networks in applications: A systematic review](http://arxiv.org/pdf/2510.24601v1)

Authors: Jessica Doohan, Lucas Kook, Kevin Burke

Neural networks have become a popular tool in predictive modelling, more
commonly associated with machine learning and artificial intelligence than with
statistics. Generalised Additive Models (GAMs) are flexible non-linear
statistical models that retain interpretability. Both are state-of-the-art in
their own right, with their respective advantages and disadvantages. This paper
analyses how these two model classes have performed on real-world tabular data.
Following PRISMA guidelines, we conducted a systematic review of papers that
performed empirical comparisons of GAMs and neural networks. Eligible papers
were identified, yielding 143 papers, with 430 datasets. Key attributes at both
paper and dataset levels were extracted and reported. Beyond summarising
comparisons, we analyse reported performance metrics using mixed-effects
modelling to investigate potential characteristics that can explain and
quantify observed differences, including application area, study year, sample
size, number of predictors, and neural network complexity. Across datasets, no
consistent evidence of superiority was found for either GAMs or neural networks
when considering the most frequently reported metrics (RMSE, $R^2$, and AUC).
Neural networks tended to outperform in larger datasets and in those with more
predictors, but this advantage narrowed over time. Conversely, GAMs remained
competitive, particularly in smaller data settings, while retaining
interpretability. Reporting of dataset characteristics and neural network
complexity was incomplete in much of the literature, limiting transparency and
reproducibility. This review highlights that GAMs and neural networks should be
viewed as complementary approaches rather than competitors. For many tabular
applications, the performance trade-off is modest, and interpretability may
favour GAMs.

### 4. [Bridging Simulators with Conditional Optimal Transport](http://arxiv.org/pdf/2510.24631v1)

Authors: Justine Zeghal, Benjamin Remy, Yashar Hezaveh, Francois Lanusse, Laurence Perreault Levasseur

We propose a new field-level emulator that bridges two simulators using
unpaired simulation datasets. Our method leverages a flow-based approach to
learn the likelihood transport from one simulator to the other. Since multiple
transport maps exist, we employ Conditional Optimal Transport Flow Matching
(COT-FM) to ensure that the transformation minimally distorts the underlying
structure of the data. We demonstrate the effectiveness of this approach by
bridging weak lensing simulators: a Lagrangian Perturbation Theory (LPT) to a
N-body Particle-Mesh (PM). We demonstrate that our emulator captures the full
correction between the simulators by showing that it enables full-field
inference to accurately recover the true posterior, validating its accuracy
beyond traditional summary statistics.

### 5. [Eigenfunction Extraction for Ordered Representation Learning](http://arxiv.org/pdf/2510.24672v1)

Authors: Burak Varıcı, Che-Ping Tsai, Ritabrata Ray, Nicholas M. Boffi, Pradeep Ravikumar

Recent advances in representation learning reveal that widely used
objectives, such as contrastive and non-contrastive, implicitly perform
spectral decomposition of a contextual kernel, induced by the relationship
between inputs and their contexts. Yet, these methods recover only the linear
span of top eigenfunctions of the kernel, whereas exact spectral decomposition
is essential for understanding feature ordering and importance. In this work,
we propose a general framework to extract ordered and identifiable
eigenfunctions, based on modular building blocks designed to satisfy key
desiderata, including compatibility with the contextual kernel and scalability
to modern settings. We then show how two main methodological paradigms,
low-rank approximation and Rayleigh quotient optimization, align with this
framework for eigenfunction extraction. Finally, we validate our approach on
synthetic kernels and demonstrate on real-world image datasets that the
recovered eigenvalues act as effective importance scores for feature selection,
enabling principled efficiency-accuracy tradeoffs via adaptive-dimensional
representations.

### 6. [The Sign Estimator: LLM Alignment in the Face of Choice Heterogeneity](http://arxiv.org/pdf/2510.23965v1)

Authors: Aymane El Gadarri, Ali Aouad, Vivek F. Farias

Traditional LLM alignment methods are vulnerable to heterogeneity in human
preferences. Fitting a na\"ive probabilistic model to pairwise comparison data
(say over prompt-completion pairs) yields an inconsistent estimate of the
population-average utility -a canonical measure of social welfare. We propose a
new method, dubbed the sign estimator, that provides a simple, provably
consistent, and efficient estimator by replacing cross-entropy with binary
classification loss in the aggregation step. This simple modification recovers
consistent ordinal alignment under mild assumptions and achieves the first
polynomial finite-sample error bounds in this setting. In realistic simulations
of LLM alignment using digital twins, the sign estimator substantially reduces
preference distortion over a panel of simulated personas, cutting (angular)
estimation error by nearly 35% and decreasing disagreement with true population
preferences from 12% to 8% compared to standard RLHF. Our method also compares
favorably to panel data heuristics that explicitly model user heterogeneity and
require tracking individual-level preference data-all while maintaining the
implementation simplicity of existing LLM alignment pipelines.

### 7. [Machine learning approaches for interpretable antibody property prediction using structural data](http://arxiv.org/pdf/2510.23975v1)

Authors: Kevin Michalewicz, Mauricio Barahona, Barbara Bravi

Understanding the relationship between antibody sequence, structure and
function is essential for the design of antibody-based therapeutics and
research tools. Recently, machine learning (ML) models mostly based on the
application of large language models to sequence information have been
developed to predict antibody properties. Yet there are open directions to
incorporate structural information, not only to enhance prediction but also to
offer insights into the underlying molecular mechanisms. This chapter provides
an overview of these approaches and describes two ML frameworks that integrate
structural data (via graph representations) with neural networks to predict
properties of antibodies: ANTIPASTI predicts binding affinity (a global
property) whereas INFUSSE predicts residue flexibility (a local property). We
survey the principles underpinning these models; the ways in which they encode
structural knowledge; and the strategies that can be used to extract
biologically relevant statistical signals that can help discover and
disentangle molecular determinants of the properties of interest.

### 8. [Score-based constrained generative modeling via Langevin diffusions with boundary conditions](http://arxiv.org/pdf/2510.23985v1)

Authors: Adam Nordenhög, Akash Sharma

Score-based generative models based on stochastic differential equations
(SDEs) achieve impressive performance in sampling from unknown distributions,
but often fail to satisfy underlying constraints. We propose a constrained
generative model using kinetic (underdamped) Langevin dynamics with specular
reflection of velocity on the boundary defining constraints. This results in
piecewise continuously differentiable noising and denoising process where the
latter is characterized by a time-reversed dynamics restricted to a domain with
boundary due to specular boundary condition. In addition, we also contribute to
existing reflected SDEs based constrained generative models, where the
stochastic dynamics is restricted through an abstract local time term. By
presenting efficient numerical samplers which converge with optimal rate in
terms of discretizations step, we provide a comprehensive comparison of models
based on confined (specularly reflected kinetic) Langevin diffusion with models
based on reflected diffusion with local time.

### 9. [Optimal Arm Elimination Algorithms for Combinatorial Bandits](http://arxiv.org/pdf/2510.23992v1)

Authors: Yuxiao Wen, Yanjun Han, Zhengyuan Zhou

Combinatorial bandits extend the classical bandit framework to settings where
the learner selects multiple arms in each round, motivated by applications such
as online recommendation and assortment optimization. While extensions of upper
confidence bound (UCB) algorithms arise naturally in this context, adapting arm
elimination methods has proved more challenging. We introduce a novel
elimination scheme that partitions arms into three categories (confirmed,
active, and eliminated), and incorporates explicit exploration to update these
sets. We demonstrate the efficacy of our algorithm in two settings: the
combinatorial multi-armed bandit with general graph feedback, and the
combinatorial linear contextual bandit. In both cases, our approach achieves
near-optimal regret, whereas UCB-based methods can provably fail due to
insufficient explicit exploration. Matching lower bounds are also provided.

### 10. [Problem-Parameter-Free Decentralized Bilevel Optimization](http://arxiv.org/pdf/2510.24288v1)

Authors: Zhiwei Zhai, Wenjing Yan, Ying-Jun Angela Zhang

Decentralized bilevel optimization has garnered significant attention due to
its critical role in solving large-scale machine learning problems. However,
existing methods often rely on prior knowledge of problem parameters-such as
smoothness, convexity, or communication network topologies-to determine
appropriate stepsizes. In practice, these problem parameters are typically
unavailable, leading to substantial manual effort for hyperparameter tuning. In
this paper, we propose AdaSDBO, a fully problem-parameter-free algorithm for
decentralized bilevel optimization with a single-loop structure. AdaSDBO
leverages adaptive stepsizes based on cumulative gradient norms to update all
variables simultaneously, dynamically adjusting its progress and eliminating
the need for problem-specific hyperparameter tuning. Through rigorous
theoretical analysis, we establish that AdaSDBO achieves a convergence rate of
$\widetilde{\mathcal{O}}\left(\frac{1}{T}\right)$, matching the performance of
well-tuned state-of-the-art methods up to polylogarithmic factors. Extensive
numerical experiments demonstrate that AdaSDBO delivers competitive performance
compared to existing decentralized bilevel optimization methods while
exhibiting remarkable robustness across diverse stepsize configurations.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-29 PST.

### 1. [We need a new Turing test to assess AI’s real-world knowledge](https://www.nature.com/articles/d41586-025-03471-0)

Authors: Vinay K. Chaudhri

### 2. [A lightweight infrared target detection network suitable for land and water surfaces](https://www.nature.com/articles/s41598-025-21550-0)

Authors: Fan Yang et al.

### 3. [Incident-aware smart prioritization framework for penetration testing and prevention of URL-based cybersecurity attacks in industry 4.0 IoT networks](https://www.nature.com/articles/s41598-025-21409-4)

Authors: Zhanserik Nurlan et al.

### 4. [Optimizing MobileNetV3 for multimodal eye gaze and emotion recognition via advanced pruning and quantisation techniques](https://www.nature.com/articles/s41598-025-19617-z)

Authors: Gousia Habib et al.

### 5. [Inequity aversion toward AI counterparts](https://www.nature.com/articles/s41598-025-22673-0)

Authors: Debanjan Borthakur et al.

### 6. [Reinforcement learning-driven deep learning approaches for optimized robot trajectory planning](https://www.nature.com/articles/s41598-025-21664-5)

Authors: Fang Shiyu

### 7. [Efficient ring signature for cross-chain data sharing in blockchain-enabled cold-chain logistics system](https://www.nature.com/articles/s41598-025-21617-y)

Authors: Yang Zhang et al.

### 8. [FFTMed: leveraging fast-fourier transform for a lightweight and adversarial-resilient medical image segmentation framework](https://www.nature.com/articles/s41598-025-21799-5)

Authors: Viet Tien Pham et al.

### 9. [Combining deep learning and microfluidics for fast and noninvasive sorting of zebrafish embryo](https://www.nature.com/articles/s41598-025-17946-7)

Authors: Alioune Diouf et al.

### 10. [Detection, localization, and staging of breast cancer lymph node metastasis in digital pathology whole slide images using selective neighborhood attention-based deep learning](https://www.nature.com/articles/s41598-025-21787-9)

Authors: Abdullah Tauqeer et al.

### 11. [Multimodal dual-stage feature refinement for robust skin lesion classification](https://www.nature.com/articles/s41598-025-14839-7)

Authors: Mahapara Khurshid et al.

### 12. [Adaptive ε-greedy exploration for stable reconfiguration in next-gen aviation IMA systems](https://www.nature.com/articles/s41598-025-09025-8)

Authors: Guodong Li et al.

### 13. [Filtering out mislabeled training instances using black-box optimization and quantum annealing](https://www.nature.com/articles/s41598-025-21686-z)

Authors: Makoto Otsuka et al.

### 14. [Sketch to photo recognition using IF and Fuzzy minimal structure oscillation in the sift domain](https://www.nature.com/articles/s41598-025-23417-w)

Authors: Bibek Majumder et al.

