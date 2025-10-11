# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-10 17:00:25.675819 PST.

### Artificial Intelligence

### 1. [Safely Exploring Novel Actions in Recommender Systems via Deployment-Efficient Policy Learning](http://arxiv.org/pdf/2510.07635v1)

Authors: Haruka Kiyohara, Yusuke Narita, Yuta Saito, Kei Tateno, Takuma Udagawa

In many real recommender systems, novel items are added frequently over time.
The importance of sufficiently presenting novel actions has widely been
acknowledged for improving long-term user engagement. A recent work builds on
Off-Policy Learning (OPL), which trains a policy from only logged data,
however, the existing methods can be unsafe in the presence of novel actions.
Our goal is to develop a framework to enforce exploration of novel actions with
a guarantee for safety. To this end, we first develop Safe Off-Policy Policy
Gradient (Safe OPG), which is a model-free safe OPL method based on a high
confidence off-policy evaluation. In our first experiment, we observe that Safe
OPG almost always satisfies a safety requirement, even when existing methods
violate it greatly. However, the result also reveals that Safe OPG tends to be
too conservative, suggesting a difficult tradeoff between guaranteeing safety
and exploring novel actions. To overcome this tradeoff, we also propose a novel
framework called Deployment-Efficient Policy Learning for Safe User
Exploration, which leverages safety margin and gradually relaxes safety
regularization during multiple (not many) deployments. Our framework thus
enables exploration of novel actions while guaranteeing safe implementation of
recommender systems.

### 2. [Control Synthesis of Cyber-Physical Systems for Real-Time Specifications through Causation-Guided Reinforcement Learning](http://arxiv.org/pdf/2510.07715v1)

Authors: Xiaochen Tang, Zhenya Zhang, Miaomiao Zhang, Jie An

In real-time and safety-critical cyber-physical systems (CPSs), control
synthesis must guarantee that generated policies meet stringent timing and
correctness requirements under uncertain and dynamic conditions. Signal
temporal logic (STL) has emerged as a powerful formalism of expressing
real-time constraints, with its semantics enabling quantitative assessment of
system behavior. Meanwhile, reinforcement learning (RL) has become an important
method for solving control synthesis problems in unknown environments. Recent
studies incorporate STL-based reward functions into RL to automatically
synthesize control policies. However, the automatically inferred rewards
obtained by these methods represent the global assessment of a whole or partial
path but do not accumulate the rewards of local changes accurately, so the
sparse global rewards may lead to non-convergence and unstable training
performances. In this paper, we propose an online reward generation method
guided by the online causation monitoring of STL. Our approach continuously
monitors system behavior against an STL specification at each control step,
computing the quantitative distance toward satisfaction or violation and
thereby producing rewards that reflect instantaneous state dynamics.
Additionally, we provide a smooth approximation of the causation semantics to
overcome the discontinuity of the causation semantics and make it
differentiable for using deep-RL methods. We have implemented a prototype tool
and evaluated it in the Gym environment on a variety of continuously controlled
benchmarks. Experimental results show that our proposed STL-guided RL method
with online causation semantics outperforms existing relevant STL-guided RL
methods, providing a more robust and efficient reward generation framework for
deep-RL.

### 3. [SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation](http://arxiv.org/pdf/2510.07733v1)

Authors: Minh-Anh Nguye, Minh-Duc Nguyen, Nguyen Thi Ha Lan, Kieu Hai Dang, Nguyen Tien Dong, Le Duy Dung

Large language models (LLMs) are increasingly adopted for automating survey
paper generation \cite{wang2406autosurvey, liang2025surveyx,
yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing
approaches typically extract content from a large collection of related papers
and prompt LLMs to summarize them directly. However, such methods often
overlook the structural relationships among papers, resulting in generated
surveys that lack a coherent taxonomy and a deeper contextual understanding of
research progress. To address these shortcomings, we propose \textbf{SurveyG},
an LLM-based agent framework that integrates \textit{hierarchical citation
graph}, where nodes denote research papers and edges capture both citation
dependencies and semantic relatedness between their contents, thereby embedding
structural and contextual knowledge into the survey generation process. The
graph is organized into three layers: \textbf{Foundation},
\textbf{Development}, and \textbf{Frontier}, to capture the evolution of
research from seminal works to incremental advances and emerging directions. By
combining horizontal search within layers and vertical depth traversal across
layers, the agent produces multi-level summaries, which are consolidated into a
structured survey outline. A multi-agent validation stage then ensures
consistency, coverage, and factual accuracy in generating the final survey.
Experiments, including evaluations by human experts and LLM-as-a-judge,
demonstrate that SurveyG outperforms state-of-the-art frameworks, producing
surveys that are more comprehensive and better structured to the underlying
knowledge taxonomy of a field.

### 4. [Haibu Mathematical-Medical Intelligent Agent:Enhancing Large Language Model Reliability in Medical Tasks via Verifiable Reasoning Chains](http://arxiv.org/pdf/2510.07748v1)

Authors: Yilun Zhang, Dexing Kong

Large Language Models (LLMs) show promise in medicine but are prone to
factual and logical errors, which is unacceptable in this high-stakes field. To
address this, we introduce the "Haibu Mathematical-Medical Intelligent Agent"
(MMIA), an LLM-driven architecture that ensures reliability through a formally
verifiable reasoning process. MMIA recursively breaks down complex medical
tasks into atomic, evidence-based steps. This entire reasoning chain is then
automatically audited for logical coherence and evidence traceability, similar
to theorem proving. A key innovation is MMIA's "bootstrapping" mode, which
stores validated reasoning chains as "theorems." Subsequent tasks can then be
efficiently solved using Retrieval-Augmented Generation (RAG), shifting from
costly first-principles reasoning to a low-cost verification model. We
validated MMIA across four healthcare administration domains, including DRG/DIP
audits and medical insurance adjudication, using expert-validated benchmarks.
Results showed MMIA achieved an error detection rate exceeding 98% with a false
positive rate below 1%, significantly outperforming baseline LLMs. Furthermore,
the RAG matching mode is projected to reduce average processing costs by
approximately 85% as the knowledge base matures. In conclusion, MMIA's
verifiable reasoning framework is a significant step toward creating
trustworthy, transparent, and cost-effective AI systems, making LLM technology
viable for critical applications in medicine.

### 5. [From Noisy to Native: LLM-driven Graph Restoration for Test-Time Graph Domain Adaptation](http://arxiv.org/pdf/2510.07762v1)

Authors: Xiangwei Lv, JinLuan Yang, Wang Lin, Jingyuan Chen, Beishui Liao

Graph domain adaptation (GDA) has achieved great attention due to its
effectiveness in addressing the domain shift between train and test data. A
significant bottleneck in existing graph domain adaptation methods is their
reliance on source-domain data, which is often unavailable due to privacy or
security concerns. This limitation has driven the development of Test-Time
Graph Domain Adaptation (TT-GDA), which aims to transfer knowledge without
accessing the source examples. Inspired by the generative power of large
language models (LLMs), we introduce a novel framework that reframes TT-GDA as
a generative graph restoration problem, "restoring the target graph to its
pristine, source-domain-like state". There are two key challenges: (1) We need
to construct a reasonable graph restoration process and design an effective
encoding scheme that an LLM can understand, bridging the modality gap. (2) We
need to devise a mechanism to ensure the restored graph acquires the intrinsic
features of the source domain, even without access to the source data. To
ensure the effectiveness of graph restoration, we propose GRAIL, that restores
the target graph into a state that is well-aligned with the source domain.
Specifically, we first compress the node representations into compact latent
features and then use a graph diffusion process to model the graph restoration
process. Then a quantization module encodes the restored features into discrete
tokens. Building on this, an LLM is fine-tuned as a generative restorer to
transform a "noisy" target graph into a "native" one. To further improve
restoration quality, we introduce a reinforcement learning process guided by
specialized alignment and confidence rewards. Extensive experiments demonstrate
the effectiveness of our approach across various datasets.

### 6. [An approach for systematic decomposition of complex llm tasks](http://arxiv.org/pdf/2510.07772v1)

Authors: Tianle Zhou, Jiakai Xu, Guanhong Liu, Jiaxiang Liu, Haonan Wang, Eugene Wu

Large Language Models (LLMs) suffer from reliability issues on complex tasks,
as existing decomposition methods are heuristic and rely on agent or manual
decomposition. This work introduces a novel, systematic decomposition framework
that we call Analysis of CONstraint-Induced Complexity (ACONIC), which models
the task as a constraint problem and leveraging formal complexity measures to
guide decomposition. On combinatorial (SATBench) and LLM database querying
tasks (Spider), we find that by decomposing the tasks following the measure of
complexity, agent can perform considerably better (10-40 percentage point).

### 7. [GCPO: When Contrast Fails, Go Gold](http://arxiv.org/pdf/2510.07790v1)

Authors: Hao Wu, Wei Liu

Reinforcement learning has been widely applied to enhance the reasoning
capabilities of large language models. Extending the inference limits of
smaller models has become a prominent research focus. However, algorithms such
as Group Relative Policy Optimization (GRPO) suffer from a clear drawback: the
upper bound of a model's rollout responses is entirely determined by the model
itself, preventing the acquisition of knowledge from samples that are either
all incorrect or all correct. In this paper, we introduce Group Contrastive
Policy Optimization (GCPO), a method that incorporates external standard
reference answers. When the model cannot solve a problem, the reference answer
supplies the correct response, steering the model toward an unequivocally
accurate update direction. This approach offers two main advantages: (1) it
improves training efficiency by fully utilizing every sample; (2) it enables
the model to emulate the problem solving strategy of the reference answer
during training, thereby enhancing generalization in reasoning. GCPO achieves
outstanding results across multiple benchmark datasets, yielding substantial
improvements over the baseline model. Our code is available at:
https://github.com/AchoWu/GCPO.

### 8. [Strategic Communication under Threat: Learning Information Trade-offs in Pursuit-Evasion Games](http://arxiv.org/pdf/2510.07813v1)

Authors: Valerio La Gatta, Dolev Mutzari, Sarit Kraus, VS Subrahmanian

Adversarial environments require agents to navigate a key strategic
trade-off: acquiring information enhances situational awareness, but may
simultaneously expose them to threats. To investigate this tension, we
formulate a PursuitEvasion-Exposure-Concealment Game (PEEC) in which a pursuer
agent must decide when to communicate in order to obtain the evader's position.
Each communication reveals the pursuer's location, increasing the risk of being
targeted. Both agents learn their movement policies via reinforcement learning,
while the pursuer additionally learns a communication policy that balances
observability and risk. We propose SHADOW (Strategic-communication Hybrid
Action Decision-making under partial Observation for Warfare), a multi-headed
sequential reinforcement learning framework that integrates continuous
navigation control, discrete communication actions, and opponent modeling for
behavior prediction. Empirical evaluations show that SHADOW pursuers achieve
higher success rates than six competitive baselines. Our ablation study
confirms that temporal sequence modeling and opponent modeling are critical for
effective decision-making. Finally, our sensitivity analysis reveals that the
learned policies generalize well across varying communication risks and
physical asymmetries between agents.

### 9. [An LLM-Powered Cooperative Framework for Large-Scale Multi-Vehicle Navigation](http://arxiv.org/pdf/2510.07825v1)

Authors: Yuping Zhou, Siqi Lai, Jindong Han, Hao Liu

The rise of Internet of Vehicles (IoV) technologies is transforming traffic
management from isolated control to a collective, multi-vehicle process. At the
heart of this shift is multi-vehicle dynamic navigation, which requires
simultaneously routing large fleets under evolving traffic conditions. Existing
path search algorithms and reinforcement learning methods struggle to scale to
city-wide networks, often failing to capture the nonlinear, stochastic, and
coupled dynamics of urban traffic. To address these challenges, we propose
CityNav, a hierarchical, LLM-powered framework for large-scale multi-vehicle
navigation. CityNav integrates a global traffic allocation agent, which
coordinates strategic traffic flow distribution across regions, with local
navigation agents that generate locally adaptive routes aligned with global
directives. To enable effective cooperation, we introduce a cooperative
reasoning optimization mechanism, in which agents are jointly trained with a
dual-reward structure: individual rewards promote per-vehicle efficiency, while
shared rewards encourage network-wide coordination and congestion reduction.
Extensive experiments on four real-world road networks of varying scales (up to
1.6 million roads and 430,000 intersections) and traffic datasets demonstrate
that CityNav consistently outperforms nine classical path search and RL-based
baselines in city-scale travel efficiency and congestion mitigation. Our
results highlight the potential of LLMs to enable scalable, adaptive, and
cooperative city-wide traffic navigation, providing a foundation for
intelligent, large-scale vehicle routing in complex urban environments. Our
project is available at https://github.com/usail-hkust/CityNav.

### 10. [FinMR: A Knowledge-Intensive Multimodal Benchmark for Advanced Financial Reasoning](http://arxiv.org/pdf/2510.07852v1)

Authors: Shuangyan Deng, Haizhou Peng, Jiachen Xu, Rui Mao, Ciprian Doru Giurcăneanu, Jiamou Liu

Multimodal Large Language Models (MLLMs) have made substantial progress in
recent years. However, their rigorous evaluation within specialized domains
like finance is hindered by the absence of datasets characterized by
professional-level knowledge intensity, detailed annotations, and advanced
reasoning complexity. To address this critical gap, we introduce FinMR, a
high-quality, knowledge-intensive multimodal dataset explicitly designed to
evaluate expert-level financial reasoning capabilities at a professional
analyst's standard. FinMR comprises over 3,200 meticulously curated and
expertly annotated question-answer pairs across 15 diverse financial topics,
ensuring broad domain diversity and integrating sophisticated mathematical
reasoning, advanced financial knowledge, and nuanced visual interpretation
tasks across multiple image types. Through comprehensive benchmarking with
leading closed-source and open-source MLLMs, we highlight significant
performance disparities between these models and professional financial
analysts, uncovering key areas for model advancement, such as precise image
analysis, accurate application of complex financial formulas, and deeper
contextual financial understanding. By providing richly varied visual content
and thorough explanatory annotations, FinMR establishes itself as an essential
benchmark tool for assessing and advancing multimodal financial reasoning
toward professional analyst-level competence.

### Hardware Architecture

### 1. [DL-PIM: Improving Data Locality in Processing-in-Memory Systems](http://arxiv.org/pdf/2510.07719v1)

Authors: Parker Hao Tian, Zahra Yousefijamarani, Alaa Alameldeen

PIM architectures aim to reduce data transfer costs between processors and
memory by integrating processing units within memory layers. Prior PIM
architectures have shown potential to improve energy efficiency and
performance. However, such advantages rely on data proximity to the processing
units performing computations. Data movement overheads can degrade PIM's
performance and energy efficiency due to the need to move data between a
processing unit and a distant memory location. %they face challenges due to the
overhead of transferring data from remote memory locations to processing units
inside memory for computation. In this paper, we demonstrate that a large
fraction of PIM's latency per memory request is attributed to data transfers
and queuing delays from remote memory accesses. To improve PIM's data locality,
we propose DL-PIM, a novel architecture that dynamically detects the overhead
of data movement, and proactively moves data to a reserved area in the local
memory of the requesting processing unit. DL-PIM uses a distributed
address-indirection hardware lookup table to redirect traffic to the current
data location. We propose DL-PIM implementations on two 3D stacked memories:
HMC and HBM. While some workloads benefit from DL-PIM, others are negatively
impacted by the additional latency due to indirection accesses. Therefore, we
propose an adaptive mechanism that assesses the cost and benefit of indirection
and dynamically enables or disables it to prevent degrading workloads that
suffer from indirection. Overall, DL-PIM reduces the average memory latency per
request by 54% in HMC and 50% in HBM which resulted in performance improvement
of 15% for workloads with substantial data reuse in HMC and 5% in HBM. For all
representative workloads, DL-PIM achieved a 6% speedup in HMC and a 3% speedup
in HBM, showing that DL-PIM enhances data locality and overall system
performance.

### 2. [A Scalable FPGA Architecture With Adaptive Memory Utilization for GEMM-Based Operations](http://arxiv.org/pdf/2510.08137v1)

Authors: Anastasios Petropoulos, Theodore Antonakopoulos

Deep neural network (DNN) inference relies increasingly on specialized
hardware for high computational efficiency. This work introduces a
field-programmable gate array (FPGA)-based dynamically configurable accelerator
featuring systolic arrays, high-bandwidth memory, and UltraRAMs. We present two
processing unit (PU) configurations with different computing capabilities using
the same interfaces and peripheral blocks. By instantiating multiple PUs and
employing a heuristic weight transfer schedule, the architecture achieves
notable throughput efficiency over prior works. Moreover, we outline how the
architecture can be extended to emulate analog in-memory computing (AIMC)
devices to aid next-generation heterogeneous AIMC chip designs and investigate
device-level noise behavior. Overall, this brief presents a versatile DNN
inference acceleration architecture adaptable to various models and future FPGA
designs.

### 3. [FMCache: File-System Metadata Caching in Programmable Switches](http://arxiv.org/pdf/2510.08351v1)

Authors: Qingxiu Liu, Jiazhen Cai, Siyuan Sheng, Yuhui Chen, Lu Tang, Zhirong Shen, Patrick P. C. Lee

Fast and scalable metadata management across multiple metadata servers is
crucial for distributed file systems to handle numerous files and directories.
Client-side caching of frequently accessed metadata can mitigate server loads,
but incurs significant overhead and complexity in maintaining cache consistency
when the number of clients increases. We propose FMCache, an in-switch
file-system metadata caching framework that leverages programmable switches to
serve file-system metadata requests from multiple clients directly in the
switch data plane. Unlike prior in-switch key-value caching approaches, FMCache
addresses file-system-specific path dependencies under stringent switch
resource constraints. We implement FMCache atop Hadoop HDFS and evaluate it on
a Tofino-switch testbed using real-world file-system metadata workloads.
FMCache achieves up to 181.6% higher throughput than vanilla HDFS and
complements client-side caching with additional throughput gains of up to
139.6%. It also incurs low latencies and limited switch resource usage.

### 4. [SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference](http://arxiv.org/pdf/2510.08544v1)

Authors: Hengrui Zhang, Pratyush Patel, August Ning, David Wentzlaff

Large Language Models (LLMs) have gained popularity in recent years, driving
up the demand for inference. LLM inference is composed of two phases with
distinct characteristics: a compute-bound prefill phase followed by a
memory-bound decode phase. To efficiently serve LLMs, prior work proposes
prefill-decode disaggregation to run each phase on separate hardware. However,
existing hardware poorly matches the different requirements of each phase.
Current datacenter GPUs and TPUs follow a more-is-better design philosophy that
maximizes compute and memory resources, causing memory bandwidth
underutilization in the prefill phase and compute underutilization in the
decode phase. Such underutilization directly translates into increased serving
costs.
  This paper proposes SPAD (Specialized Prefill and Decode hardware), adopting
a less-is-more methodology to design specialized chips tailored to the distinct
characteristics of prefill and decode phases. The proposed Prefill Chips have
larger systolic arrays and use cost-effective GDDR memory, whereas the proposed
Decode Chips retain high memory bandwidth but reduce compute capacity. Compared
to modeled H100s, simulations show that the proposed Prefill Chips deliver 8%
higher prefill performance on average at 52% lower hardware cost, while the
proposed Decode Chips achieve 97% of the decode performance with 28% lower TDP.
  End-to-end simulations on production traces show that SPAD reduces hardware
cost by 19%-41% and TDP by 2%-17% compared to modeled baseline clusters while
offering the same performance. Even when models and workloads change, SPAD can
reallocate either type of chip to run either phase and still achieve 11%-43%
lower hardware costs, demonstrating the longevity of the SPAD design.

### Computational Complexity

### 1. [Efficient Closest Matrix Product State Learning in Logarithmic Depth](http://arxiv.org/pdf/2510.07798v1)

Authors: Chia-Ying Lin, Nai-Hui Chia, Shih-Han Hung

Learning the closest matrix product state (MPS) representation of a quantum
state is known to enable useful tools for prediction and analysis of complex
quantum systems.
  In this work, we study the problem of learning MPS in following setting:
given many copies of an input MPS, the task is to recover a classical
description of the state. The best known polynomial-time algorithm, introduced
by [LCLP10, CPF+10], requires linear circuit depth and $O(n^5)$ samples, and
has seen no improvement in over a decade. The strongest known lower bound is
only $\Omega(n)$. The combination of linear depth and high sample complexity
renders existing algorithms impractical for near-term or even early
fault-tolerant quantum devices.
  We show a new efficient MPS learning algorithm that runs in $O(\log n)$ depth
and has sample complexity $O(n^3)$. Also, we can generalize our algorithm to
learn closest MPS state, in which the input state is not guaranteed to be close
to the MPS with a fixed bond dimension. Our algorithms also improve both sample
complexity and circuit depth of previous known algorithm.

### 2. [Quantum Advantage from Sampling Shallow Circuits: Beyond Hardness of Marginals](http://arxiv.org/pdf/2510.07808v1)

Authors: Daniel Grier, Daniel M. Kane, Jackson Morris, Anthony Ostuni, Kewen Wu

We construct a family of distributions $\{\mathcal{D}_n\}_n$ with
$\mathcal{D}_n$ over $\{0, 1\}^n$ and a family of depth-$7$ quantum circuits
$\{C_n\}_n$ such that $\mathcal{D}_n$ is produced exactly by $C_n$ with the all
zeros state as input, yet any constant-depth classical circuit with bounded
fan-in gates evaluated on any binary product distribution has total variation
distance $1 - e^{-\Omega(n)}$ from $\mathcal{D}_n$. Moreover, the quantum
circuits we construct are geometrically local and use a relatively standard
gate set: Hadamard, controlled-phase, CNOT, and Toffoli gates. All previous
separations of this type suffer from some undesirable constraint on the
classical circuit model or the quantum circuits witnessing the separation.
  Our family of distributions is inspired by the Parity Halving Problem of
Watts, Kothari, Schaeffer, and Tal (STOC, 2019), which built on the work of
Bravyi, Gosset, and K\"onig (Science, 2018) to separate shallow quantum and
classical circuits for relational problems.

### 3. [Quantum Max-Cut is NP hard to approximate](http://arxiv.org/pdf/2510.07995v1)

Authors: Stephen Piddock

We unconditionally prove that it is NP-hard to compute a constant
multiplicative approximation to the QUANTUM MAX-CUT problem on an unweighted
graph of constant bounded degree. The proof works in two stages: first we
demonstrate a generic reduction to computing the optimal value of a quantum
problem, from the optimal value over product states. Then we prove an
approximation preserving reduction from MAX-CUT to PRODUCT-QMC the product
state version of QUANTUM MAX-CUT. More precisely, in the second part, we
construct a PTAS reduction from MAX-CUT$_k$ (the rank-k constrained version of
MAX-CUT) to MAX-CUT$_{k+1}$, where MAX-CUT and PRODUCT-QMC coincide with
MAX-CUT$_1$ and MAX-CUT$_3$ respectively. We thus prove that Max-Cut$_k$ is
APX-complete for all constant $k$.

### 4. [Timeline Problems in Temporal Graphs: Vertex Cover vs. Dominating Set](http://arxiv.org/pdf/2510.08124v1)

Authors: Anton Herrmann, Christian Komusiewicz, Nils Morawietz, Frank Sommer

A temporal graph is a finite sequence of graphs, called snapshots, over the
same vertex set. Many temporal graph problems turn out to be much more
difficult than their static counterparts. One such problem is \textsc{Timeline
Vertex Cover} (also known as \textsc{MinTimeline$_\infty$}), a temporal
analogue to the classical \textsc{Vertex Cover} problem. In this problem, one
is given a temporal graph $\mathcal{G}$ and two integers $k$ and $\ell$, and
the goal is to cover each edge of each snapshot by selecting for each vertex at
most $k$ activity intervals of length at most $\ell$ each. Here, an edge $uv$
in the $i$th snapshot is covered, if an activity interval of $u$ or $v$ is
active at time $i$. In this work, we continue the algorithmic study of
\textsc{Timeline Vertex Cover} and introduce the \textsc{Timeline Dominating
Set} problem where we want to dominate all vertices in each snapshot by the
selected activity intervals.
  We analyze both problems from a classical and parameterized point of view and
also consider partial problem versions, where the goal is to cover (dominate)
at least $t$ edges (vertices) of the snapshots. With respect to the
parameterized complexity, we consider the temporal graph parameters
vertex-interval-membership-width $(vimw)$ and interval-membership-width
$(imw)$. We show that all considered problems admit FPT-algorithms when
parameterized by $vimw + k+\ell$. This provides a smaller parameter combination
than the ones used for previously known FPT-algorithms for \textsc{Timeline
Vertex Cover}. Surprisingly, for $imw+ k+\ell$, \textsc{Timeline Dominating
Set} turns out to be easier than \textsc{Timeline Vertex Cover}, by also
admitting an FPT-algorithm, whereas the vertex cover version is NP-hard even if
$imw+\, k+\ell$ is constant. We also consider parameterization by combinations
of $n$, the vertex set size, with $k$ or $\ell$ and parameterization by $t$.

### 5. [k-SUM Hardness Implies Treewidth-SETH](http://arxiv.org/pdf/2510.08185v1)

Authors: Michael Lampis

We show that if k-SUM is hard, in the sense that the standard algorithm is
essentially optimal, then a variant of the SETH called the Primal Treewidth
SETH is true. Formally: if there is an $\varepsilon>0$ and an algorithm which
solves SAT in time $(2-\varepsilon)^{tw}|\phi|^{O(1)}$, where $tw$ is the width
of a given tree decomposition of the primal graph of the input, then there
exists a randomized algorithm which solves k-SUM in time
$n^{(1-\delta)\frac{k}{2}}$ for some $\delta>0$ and all sufficiently large $k$.
We also establish an analogous result for the k-XOR problem, where integer
addition is replaced by component-wise addition modulo $2$.
  As an application of our reduction we are able to revisit tight lower bounds
on the complexity of several fundamental problems parameterized by treewidth
(Independent Set, Max Cut, $k$-Coloring). Our results imply that these bounds,
which were initially shown under the SETH, also hold if one assumes the k-SUM
or k-XOR Hypotheses, arguably increasing our confidence in their validity.

### 6. [How hard is it to verify a classical shadow?](http://arxiv.org/pdf/2510.08515v1)

Authors: Georgios Karaiskos, Dorian Rudolph, Johannes Jakob Meyer, Jens Eisert, Sevag Gharibian

Classical shadows are succinct classical representations of quantum states
which allow one to encode a set of properties P of a quantum state rho, while
only requiring measurements on logarithmically many copies of rho in the size
of P. In this work, we initiate the study of verification of classical shadows,
denoted classical shadow validity (CSV), from the perspective of computational
complexity, which asks: Given a classical shadow S, how hard is it to verify
that S predicts the measurement statistics of a quantum state? We show that
even for the elegantly simple classical shadow protocol of [Huang, Kueng,
Preskill, Nature Physics 2020] utilizing local Clifford measurements, CSV is
QMA-complete. This hardness continues to hold for the high-dimensional
extension of said protocol due to [Mao, Yi, and Zhu, PRL 2025]. Among other
results, we also show that CSV for exponentially many observables is complete
for a quantum generalization of the second level of the polynomial hierarchy,
yielding the first natural complete problem for such a class.

### 7. [Optimal lower bounds for quantum state tomography](http://arxiv.org/pdf/2510.07699v1)

Authors: Thilo Scharnhorst, Jack Spilecki, John Wright

We show that $n = \Omega(rd/\varepsilon^2)$ copies are necessary to learn a
rank $r$ mixed state $\rho \in \mathbb{C}^{d \times d}$ up to error
$\varepsilon$ in trace distance. This matches the upper bound of $n =
O(rd/\varepsilon^2)$ from prior work, and therefore settles the sample
complexity of mixed state tomography. We prove this lower bound by studying a
special case of full state tomography that we refer to as projector tomography,
in which $\rho$ is promised to be of the form $\rho = P/r$, where $P \in
\mathbb{C}^{d \times d}$ is a rank $r$ projector. A key technical ingredient in
our proof, which may be of independent interest, is a reduction which converts
any algorithm for projector tomography which learns to error $\varepsilon$ in
trace distance to an algorithm which learns to error $O(\varepsilon)$ in the
more stringent Bures distance.

### 8. [Verifying Graph Neural Networks with Readout is Intractable](http://arxiv.org/pdf/2510.08045v1)

Authors: Artem Chernobrovkin, Marco Sälzer, François Schwarzentruber, Nicolas Troquard

We introduce a logical language for reasoning about quantized
aggregate-combine graph neural networks with global readout (ACR-GNNs). We
provide a logical characterization and use it to prove that verification tasks
for quantized GNNs with readout are (co)NEXPTIME-complete. This result implies
that the verification of quantized GNNs is computationally intractable,
prompting substantial research efforts toward ensuring the safety of GNN-based
systems. We also experimentally demonstrate that quantized ACR-GNN models are
lightweight while maintaining good accuracy and generalization capabilities
with respect to non-quantized models.

### 9. [A Graph Width Perspective on Partially Ordered Hamiltonian Paths and Cycles II: Vertex and Edge Deletion Numbers](http://arxiv.org/pdf/2510.08378v1)

Authors: Jesse Beisegel, Katharina Klost, Kristin Knorr, Fabienne Ratajczak, Robert Scheffler

We consider the problem of finding a Hamiltonian path or cycle with
precedence constraints in the form of a partial order on the vertex set. We
study the complexity for graph width parameters for which the ordinary problems
$\mathsf{Hamiltonian\ Path}$ and $\mathsf{Hamiltonian\ Cycle}$ are in
$\mathsf{FPT}$. In particular, we focus on parameters that describe how many
vertices and edges have to be deleted to become a member of a certain graph
class. We show that the problems are $\mathsf{W[1]}$-hard for such restricted
cases as vertex distance to path and vertex distance to clique. We complement
these results by showing that the problems can be solved in $\mathsf{XP}$ time
for vertex distance to outerplanar and vertex distance to block. Furthermore,
we present some $\mathsf{FPT}$ algorithms, e.g., for edge distance to block.
Additionally, we prove para-$\mathsf{NP}$-hardness when considered with the
edge clique cover number.

### 10. [Computing moment polytopes -- with a focus on tensors, entanglement and matrix multiplication](http://arxiv.org/pdf/2510.08336v1)

Authors: Maxim van den Berg, Matthias Christandl, Vladimir Lysikov, Harold Nieuwboer, Michael Walter, Jeroen Zuiddam

Tensors are fundamental in mathematics, computer science, and physics. Their
study through algebraic geometry and representation theory has proved very
fruitful in the context of algebraic complexity theory and quantum information.
In particular, moment polytopes have been understood to play a key role. In
quantum information, moment polytopes (also known as entanglement polytopes)
provide a framework for the single-particle quantum marginal problem and offer
a geometric characterization of entanglement. In algebraic complexity, they
underpin quantum functionals that capture asymptotic tensor relations. More
recently, moment polytopes have also become foundational to the emerging field
of scaling algorithms in computer science and optimization.
  Despite their fundamental role and interest from many angles, much is still
unknown about these polytopes, and in particular for tensors beyond
$\mathbb{C}^2\otimes\mathbb{C}^2\otimes\mathbb{C}^2$ and
$\mathbb{C}^2\otimes\mathbb{C}^2\otimes\mathbb{C}^2\otimes\mathbb{C}^2$ only
sporadically have they been computed. We give a new algorithm for computing
moment polytopes of tensors (and in fact moment polytopes for the general class
of reductive algebraic groups) based on a mathematical description by Franz (J.
Lie Theory 2002).
  This algorithm enables us to compute moment polytopes of tensors of dimension
an order of magnitude larger than previous methods, allowing us to compute with
certainty, for the first time, all moment polytopes of tensors in
$\mathbb{C}^3\otimes\mathbb{C}^3\otimes\mathbb{C}^3$, and with high probability
those in $\mathbb{C}^4\otimes\mathbb{C}^4\otimes\mathbb{C}^4$ (which includes
the $2\times 2$ matrix multiplication tensor). We discuss how these explicit
moment polytopes have led to several new theoretical directions and results.

### Computational Engineering

### 1. [Zero-Shot Forecasting of Network Dynamics through Weight Flow Matching](http://arxiv.org/pdf/2510.07957v1)

Authors: Shihe Zhou, Ruikun Li, Huandong Wang, Yong Li

Forecasting state evolution of network systems, such as the spread of
information on social networks, is significant for effective policy
interventions and resource management. However, the underlying propagation
dynamics constantly shift with new topics or events, which are modeled as
changing coefficients of the underlying dynamics. Deep learning models struggle
to adapt to these out-of-distribution shifts without extensive new data and
retraining. To address this, we present Zero-Shot Forecasting of Network
Dynamics through Weight Flow Matching (FNFM), a generative,
coefficient-conditioned framework that generates dynamic model weights for an
unseen target coefficient, enabling zero-shot forecasting. Our framework
utilizes a Variational Encoder to summarize the forecaster weights trained in
observed environments into compact latent tokens. A Conditional Flow Matching
(CFM) module then learns a continuous transport from a simple Gaussian
distribution to the empirical distribution of these weights, conditioned on the
dynamical coefficients. This process is instantaneous at test time and requires
no gradient-based optimization. Across varied dynamical coefficients, empirical
results indicate that FNFM yields more reliable zero-shot accuracy than
baseline methods, particularly under pronounced coefficient shift.

### 2. [Poisson Energy Formulation for Floorplanning: Variational Analysis and Mathematical Foundations](http://arxiv.org/pdf/2510.08126v1)

Authors: Wenxing Zhu, Hao Ai

Arranging many modules within a bounded domain without overlap, central to
the Electronic Design Automation (EDA) of very large-scale integrated (VLSI)
circuits, represents a broad class of discrete geometric optimization problems
with physical constraints. This paper develops a variational and spectral
framework for Poisson energy-based floorplanning and placement in physical
design. We show that the Poisson energy, defined via a Neumann Poisson
equation, is exactly the squared H^{-1} Sobolev norm of the density residual,
providing a functional-analytic interpretation of the classical electrostatic
analogy. Through spectral analysis, we demonstrate that the energy acts as an
intrinsic low-pass filter, suppressing high-frequency fluctuations while
enforcing large-scale uniformity. Under a mild low-frequency dominance
assumption, we establish a quantitative linear lower bound relating the Poisson
energy to the geometric overlap area, thereby justifying its use as a smooth
surrogate for the hard nonoverlap constraint. We further show that projected
gradient descent converges globally to stationary points and exhibits local
linear convergence near regular minima. Finally, we interpret the
continuous-time dynamics as a Wasserstein-2 gradient flow, revealing the
intrinsic nonlocality and global balancing behavior of the model. These results
provide a mathematically principled foundation for PDE-regularized optimization
in large-scale floorplanning and related geometric layout problems.

### 3. [Design of chemical recycling processes for PUR foam under uncertainty](http://arxiv.org/pdf/2510.08301v1)

Authors: Patrick Lotz, Luca Bosetti, André Bardow, Sergio Lucia, Sebastian Engell

Optimization problems in chemical process design involve a significant number
of discrete and continuous decisions. When taking into account uncertainties,
the search space is very difficult to explore, even for experienced engineers.
Moreover, it should be taken into account that while some decisions are fixed
at the design stage, other parameters can be adapted to the realization of the
uncertainty during the operation of the plant. This leads to a two-stage
optimization problem which is difficult to solve. To address this challenge, we
propose to combine commercial process simulation software with an evolutionary
strategy. This approach is applied to designing a downstream process to isolate
valuable products from pyrolysis oil produced by the catalytic pyrolysis of
rigid polyurethane foam. The suggested algorithm consistently performed better
than a manually designed robust process. Additionally, the analysis of
different scenarios provided insight into promising changes in the overall
layout of the recycling process.

### 4. [IKNet: Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators](http://arxiv.org/pdf/2510.07661v1)

Authors: Jinwoong Kim, Sangjin Park

The increasing influence of unstructured external information, such as news
articles, on stock prices has attracted growing attention in financial markets.
Despite recent advances, most existing newsbased forecasting models represent
all articles using sentiment scores or average embeddings that capture the
general tone but fail to provide quantitative, context-aware explanations of
the impacts of public sentiment on predictions. To address this limitation, we
propose an interpretable keyword-guided network (IKNet), which is an
explainable forecasting framework that models the semantic association between
individual news keywords and stock price movements. The IKNet identifies
salient keywords via FinBERTbased contextual analysis, processes each embedding
through a separate nonlinear projection layer, and integrates their
representations with the time-series data of technical indicators to forecast
next-day closing prices. By applying Shapley Additive Explanations the model
generates quantifiable and interpretable attributions for the contribution of
each keyword to predictions. Empirical evaluations of S&P 500 data from 2015 to
2024 demonstrate that IKNet outperforms baselines, including recurrent neural
networks and transformer models, reducing RMSE by up to 32.9% and improving
cumulative returns by 18.5%. Moreover, IKNet enhances transparency by offering
contextualized explanations of volatility events driven by public sentiment.

### 5. [Reverse Supply Chain Network Design of a Polyurethane Waste Upcycling System](http://arxiv.org/pdf/2510.08097v1)

Authors: Dalga Merve Özkan, Sergio Lucia, Sebastian Engell

This paper presents a general mathematical programming framework for the
design and optimization of supply chain infrastructures for the upcycling of
plastic waste. For this purpose, a multi-product, multi-echelon, multi-period
mixed-integer linear programming (MILP) model has been formulated. The
objective is to minimize the cost of the entire circular supply chain starting
from the collection of post-consumer plastic waste to the production of
virgin-equivalent high value polymers, satisfying a large number of constraints
from collection quota to the quality of the feedstock. The framework aims to
support the strategic planning of future circular supply chains by determining
the optimal number, locations and sizes of various types of facilities as well
as the amounts of materials to be transported between the nodes of the supply
chain network over a specified period. The functionality of the framework has
been tested with a case study for the upcycling of rigid polyurethane foam
waste coming from construction sites in Germany. The economic potential and
infrastructure requirements are evaluated, and it has been found that from a
solely economic perspective, the current status of the value chain is not
competitive with fossil-based feedstock or incineration. However, with the
right economic incentives, there is a considerable potential to establish such
value chains, once the upcycling technology is ready and the economic framework
conditions have stabilized.

### 6. [Large Language Models Meet Virtual Cell: A Survey](http://arxiv.org/pdf/2510.07706v1)

Authors: Krinos Li, Xianglu Xiao, Shenglong Deng, Lucas He, Zijun Zhong, Yuanjie Zou, Zhonghao Zhan, Zheng Hui, Weiye Bao, Guang Yang

Large language models (LLMs) are transforming cellular biology by enabling
the development of "virtual cells"--computational systems that represent,
predict, and reason about cellular states and behaviors. This work provides a
comprehensive review of LLMs for virtual cell modeling. We propose a unified
taxonomy that organizes existing methods into two paradigms: LLMs as Oracles,
for direct cellular modeling, and LLMs as Agents, for orchestrating complex
scientific tasks. We identify three core tasks--cellular representation,
perturbation prediction, and gene regulation inference--and review their
associated models, datasets, evaluation benchmarks, as well as the critical
challenges in scalability, generalizability, and interpretability.

### Computational Geometry

### 1. [Robust Geometric Predicates for Bivariate Computational Topology](http://arxiv.org/pdf/2510.07955v1)

Authors: Petar Hristov, Ingrid Hotz, Talha Bin Masood

We present theory and practice for robust implementations of bivariate Jacobi
set and Reeb space algorithms. Robustness is a fundamental topic in
computational geometry that deals with the issues of numerical errors and
degenerate cases in algorithm implementations. Computational topology already
uses some robustness techniques for the development of scalar field algorithms,
such as those for computing critical points, merge trees, contour trees, Reeb
graphs, Morse-Smale complexes, and persistent homology. In most cases,
robustness can be ensured with floating-point arithmetic, and degenerate cases
can be resolved with a standard symbolic perturbation technique called
Simulation of Simplicity. However, this becomes much more complex for
topological data structures of multifields, such as Jacobi sets and Reeb
spaces. The geometric predicates used in their computation require exact
arithmetic and a more involved treatment of degenerate cases to ensure
correctness. Neither of these challenges has been fully addressed in the
literature so far. In this paper, we describe how exact arithmetic and symbolic
perturbation schemes can be used to enable robust implementations of bivariate
Jacobi set and Reeb space algorithms. In the process, we develop a method for
automatically evaluating predicates that can be expressed as large symbolic
polynomials, which are difficult to factor appropriately by hand, as is
typically done in the computational geometry literature. We provide
implementations of all proposed approaches and evaluate their efficiency.

### Computation and Language

### 1. [Role-Conditioned Refusals: Evaluating Access Control Reasoning in Large Language Models](http://arxiv.org/pdf/2510.07642v1)

Authors: Đorđe Klisura, Joseph Khoury, Ashish Kundu, Ram Krishnan, Anthony Rios

Access control is a cornerstone of secure computing, yet large language
models often blur role boundaries by producing unrestricted responses. We study
role-conditioned refusals, focusing on the LLM's ability to adhere to access
control policies by answering when authorized and refusing when not. To
evaluate this behavior, we created a novel dataset that extends the Spider and
BIRD text-to-SQL datasets, both of which have been modified with realistic
PostgreSQL role-based policies at the table and column levels. We compare three
designs: (i) zero or few-shot prompting, (ii) a two-step generator-verifier
pipeline that checks SQL against policy, and (iii) LoRA fine-tuned models that
learn permission awareness directly. Across multiple model families, explicit
verification (the two-step framework) improves refusal precision and lowers
false permits. At the same time, fine-tuning achieves a stronger balance
between safety and utility (i.e., when considering execution accuracy). Longer
and more complex policies consistently reduce the reliability of all systems.
We release RBAC-augmented datasets and code.

### 2. [MemWeaver: A Hierarchical Memory from Textual Interactive Behaviors for Personalized Generation](http://arxiv.org/pdf/2510.07713v1)

Authors: Shuo Yu, Mingyue Cheng, Daoyu Wang, Qi Liu, Zirui Liu, Ze Guo, Xiaoyu Tao

The primary form of user-internet engagement is shifting from leveraging
implicit feedback signals, such as browsing and clicks, to harnessing the rich
explicit feedback provided by textual interactive behaviors. This shift unlocks
a rich source of user textual history, presenting a profound opportunity for a
deeper form of personalization. However, prevailing approaches offer only a
shallow form of personalization, as they treat user history as a flat list of
texts for retrieval and fail to model the rich temporal and semantic structures
reflecting dynamic nature of user interests. In this work, we propose
\textbf{MemWeaver}, a framework that weaves the user's entire textual history
into a hierarchical memory to power deeply personalized generation. The core
innovation of our memory lies in its ability to capture both the temporal
evolution of interests and the semantic relationships between different
activities. To achieve this, MemWeaver builds two complementary memory
components that both integrate temporal and semantic information, but at
different levels of abstraction: behavioral memory, which captures specific
user actions, and cognitive memory, which represents long-term preferences.
This dual-component memory serves as a unified representation of the user,
allowing large language models (LLMs) to reason over both concrete behaviors
and abstracted traits. Experiments on the Language Model Personalization (LaMP)
benchmark validate the efficacy of MemWeaver. Our code is
available\footnote{https://github.com/fishsure/MemWeaver}.

### 3. [SUBQRAG: sub-question driven dynamic graph rag](http://arxiv.org/pdf/2510.07718v1)

Authors: Jiaoyang Li, Junhao Ruan, Shengwei Tang, Saihan Chen, Kaiyan Chang, Yuan Ge, Tong Xiao, Jingbo Zhu

Graph Retrieval-Augmented Generation (Graph RAG) effectively builds a
knowledge graph (KG) to connect disparate facts across a large document corpus.
However, this broad-view approach often lacks the deep structured reasoning
needed for complex multi-hop question answering (QA), leading to incomplete
evidence and error accumulation. To address these limitations, we propose
SubQRAG, a sub-question-driven framework that enhances reasoning depth. SubQRAG
decomposes a complex question into an ordered chain of verifiable
sub-questions. For each sub-question, it retrieves relevant triples from the
graph. When the existing graph is insufficient, the system dynamically expands
it by extracting new triples from source documents in real time. All triples
used in the reasoning process are aggregated into a "graph memory," forming a
structured and traceable evidence path for final answer generation. Experiments
on three multi-hop QA benchmarks demonstrate that SubQRAG achieves consistent
and significant improvements, especially in Exact Match scores.

### 4. [Multilingual Knowledge Graph Completion via Efficient Multilingual Knowledge Sharing](http://arxiv.org/pdf/2510.07736v1)

Authors: Cunli Mao, Xiaofei Gao, Ran Song, Shizhu He, Shengxiang Gao, Kang Liu, Zhengtao Yu

Large language models (LLMs) based Multilingual Knowledge Graph Completion
(MKGC) aim to predict missing facts by leveraging LLMs' multilingual
understanding capabilities, improving the completeness of multilingual
knowledge graphs (KGs). However, existing MKGC research underutilizes the
multilingual capabilities of LLMs and ignores the shareability of cross-lingual
knowledge. In this paper, we propose a novel MKGC framework that leverages
multilingual shared knowledge to significantly enhance performance through two
components: Knowledge-level Grouped Mixture of Experts (KL-GMoE) and Iterative
Entity Reranking (IER). KL-GMoE efficiently models shared knowledge, while IER
significantly enhances its utilization. To evaluate our framework, we
constructed a mKG dataset containing 5 languages and conducted comprehensive
comparative experiments with existing state-of-the-art (SOTA) MKGC method. The
experimental results demonstrate that our framework achieves improvements of
5.47%, 3.27%, and 1.01% in the Hits@1, Hits@3, and Hits@10 metrics,
respectively, compared with SOTA MKGC method. Further experimental analysis
revealed the properties of knowledge sharing in settings of unseen and
unbalanced languages. We have released the dataset and code for our work on
https://github.com/gaoxiaofei07/KL-GMoE.

### 5. [OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment](http://arxiv.org/pdf/2510.07743v1)

Authors: Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, Haoyu Wang

Reward modeling lies at the core of reinforcement learning from human
feedback (RLHF), yet most existing reward models rely on scalar or pairwise
judgments that fail to capture the multifaceted nature of human preferences.
Recent studies have explored rubrics-as-rewards (RaR) that uses structured
natural language criteria that capture multiple dimensions of response quality.
However, producing rubrics that are both reliable and scalable remains a key
challenge. In this work, we introduce OpenRubrics, a diverse, large-scale
collection of (prompt, rubric) pairs for training rubric-generation and
rubric-based reward models. To elicit discriminative and comprehensive
evaluation signals, we introduce Contrastive Rubric Generation (CRG), which
derives both hard rules (explicit constraints) and principles (implicit
qualities) by contrasting preferred and rejected responses. We further improve
reliability by enforcing preference-label consistency via rejection sampling to
remove noisy rubrics. Across multiple reward-modeling benchmarks, our
rubric-based reward model, Rubric-RM, surpasses strong size-matched baselines
by 6.8%. These gains transfer to policy models on instruction-following and
biomedical benchmarks. Our results show that rubrics provide scalable alignment
signals that narrow the gap between costly human evaluation and automated
reward modeling, enabling a new principle-driven paradigm for LLM alignment.

### 6. [Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers](http://arxiv.org/pdf/2510.07761v1)

Authors: Nishant Balepur, Atrey Desai, Rachel Rudinger

Large language models (LLMs) now give reasoning before answering, excelling
in tasks like multiple-choice question answering (MCQA). Yet, a concern is that
LLMs do not solve MCQs as intended, as work finds LLMs sans reasoning succeed
in MCQA without using the question, i.e., choices-only. Such partial-input
success is often deemed problematic, but reasoning traces could reveal if these
strategies are truly shallow in choices-only settings. To study these
strategies, reasoning LLMs solve MCQs in full and choices-only inputs;
test-time reasoning often boosts accuracy on full and in choices-only half the
time. While possibly due to shallow shortcuts, choices-only success is barely
affected by the length of reasoning traces, and after finding traces pass
faithfulness tests, we show they use less problematic strategies like inferring
missing questions. In all, we challenge claims that partial-input success is
always a flaw, so we discuss how reasoning traces could separate problematic
data from less problematic reasoning.

### 7. [Curing Miracle Steps in LLM Mathematical Reasoning with Rubric Rewards](http://arxiv.org/pdf/2510.07774v1)

Authors: Youliang Yuan, Qiuyang Mang, Jingbang Chen, Hong Wan, Xiaoyuan Liu, Junjielong Xu, Jen-tse Huang, Wenxuan Wang, Wenxiang Jiao, Pinjia He

Large language models for mathematical reasoning are typically trained with
outcome-based rewards, which credit only the final answer. In our experiments,
we observe that this paradigm is highly susceptible to reward hacking, leading
to a substantial overestimation of a model's reasoning ability. This is
evidenced by a high incidence of false positives - solutions that reach the
correct final answer through an unsound reasoning process. Through a systematic
analysis with human verification, we establish a taxonomy of these failure
modes, identifying patterns like Miracle Steps - abrupt jumps to a correct
output without a valid preceding derivation. Probing experiments suggest a
strong association between these Miracle Steps and memorization, where the
model appears to recall the answer directly rather than deriving it. To
mitigate this systemic issue, we introduce the Rubric Reward Model (RRM), a
process-oriented reward function that evaluates the entire reasoning trajectory
against problem-specific rubrics. The generative RRM provides fine-grained,
calibrated rewards (0-1) that explicitly penalize logical flaws and encourage
rigorous deduction. When integrated into a reinforcement learning pipeline,
RRM-based training consistently outperforms outcome-only supervision across
four math benchmarks. Notably, it boosts Verified Pass@1024 on AIME2024 from
26.7% to 62.6% and reduces the incidence of Miracle Steps by 71%. Our work
demonstrates that rewarding the solution process is crucial for building models
that are not only more accurate but also more reliable.

### 8. [The Unintended Trade-off of AI Alignment:Balancing Hallucination Mitigation and Safety in LLMs](http://arxiv.org/pdf/2510.07775v1)

Authors: Omar Mahmoud, Ali Khalil, Buddhika Laknath Semage, Thommen George Karimpanal, Santu Rana

Hallucination in large language models (LLMs) has been widely studied in
recent years, with progress in both detection and mitigation aimed at improving
truthfulness. Yet, a critical side effect remains largely overlooked: enhancing
truthfulness can negatively impact safety alignment. In this paper, we
investigate this trade-off and show that increasing factual accuracy often
comes at the cost of weakened refusal behavior. Our analysis reveals that this
arises from overlapping components in the model that simultaneously encode
hallucination and refusal information, leading alignment methods to suppress
factual knowledge unintentionally. We further examine how fine-tuning on benign
datasets, even when curated for safety, can degrade alignment for the same
reason. To address this, we propose a method that disentangles refusal-related
features from hallucination features using sparse autoencoders, and preserves
refusal behavior during fine-tuning through subspace orthogonalization. This
approach prevents hallucinations from increasing while maintaining safety
alignment.We evaluate our method on commonsense reasoning tasks and harmful
benchmarks (AdvBench and StrongReject). Results demonstrate that our approach
preserves refusal behavior and task utility, mitigating the trade-off between
truthfulness and safety.

### 9. [RCPU: Rotation-Constrained Error Compensation for Structured Pruning of a Large Language Model](http://arxiv.org/pdf/2510.07782v1)

Authors: Shuichiro Haruta, Kazunori Matsumoto, Zhi Li, Yanan Wang, Mori Kurokawa

In this paper, we propose a rotation-constrained compensation method to
address the errors introduced by structured pruning of large language models
(LLMs). LLMs are trained on massive datasets and accumulate rich semantic
knowledge in their representation space. In contrast, pruning is typically
carried out with only a small amount of calibration data, which makes output
mismatches unavoidable. Although direct least-squares fitting can reduce such
errors, it tends to overfit to the limited calibration set, destructively
modifying pretrained weights. To overcome this difficulty, we update the pruned
parameters under a rotation constraint. This constrained update preserves the
geometry of output representations (i.e., norms and inner products) and
simultaneously re-aligns the pruned subspace with the original outputs.
Furthermore, in rotation-constrained compensation, removing components that
strongly contribute to the principal directions of the output makes error
recovery difficult. Since input dimensions with large variance strongly affect
these principal directions, we design a variance-aware importance score that
ensures such dimensions are preferentially kept in the pruned model. By
combining this scoring rule with rotation-constrained updates, the proposed
method effectively compensates errors while retaining the components likely to
be more important in a geometry-preserving manner. In the experiments, we apply
the proposed method to LLaMA-7B and evaluate it on WikiText-2 and multiple
language understanding benchmarks. The results demonstrate consistently better
perplexity and task accuracy compared with existing baselines.

### 10. [Multilingual Generative Retrieval via Cross-lingual Semantic Compression](http://arxiv.org/pdf/2510.07812v1)

Authors: Yuxin Huang, Simeng Wu, Ran Song, Yan Xiang, Yantuan Xian, Shengxiang Gao, Zhengtao Yu

Generative Information Retrieval is an emerging retrieval paradigm that
exhibits remarkable performance in monolingual scenarios.However, applying
these methods to multilingual retrieval still encounters two primary
challenges, cross-lingual identifier misalignment and identifier inflation. To
address these limitations, we propose Multilingual Generative Retrieval via
Cross-lingual Semantic Compression (MGR-CSC), a novel framework that unifies
semantically equivalent multilingual keywords into shared atoms to align
semantics and compresses the identifier space, and we propose a dynamic
multi-step constrained decoding strategy during retrieval. MGR-CSC improves
cross-lingual alignment by assigning consistent identifiers and enhances
decoding efficiency by reducing redundancy. Experiments demonstrate that
MGR-CSC achieves outstanding retrieval accuracy, improving by 6.83% on
mMarco100k and 4.77% on mNQ320k, while reducing document identifiers length by
74.51% and 78.2%, respectively.

### Cryptography and Security

### 1. [ANCORA: Accurate Intrusion Recovery for Web Applications](http://arxiv.org/pdf/2510.07806v1)

Authors: Yihao Peng, Biao Ma, Hai Wan, Xibin Zhao

Modern web application recovery presents a critical dilemma. Coarse-grained
snapshot rollbacks cause unacceptable data loss for legitimate users.
Surgically removing an attack's impact is hindered by a fundamental challenge
in high-concurrency environments: it is difficult to attribute resulting file
and database modifications to a specific attack-related request. We present
ANCORA, a system for precise intrusion recovery in web applications without
invasive instrumentation. ANCORA first isolates the full sequence of syscalls
triggered by a single malicious request. Based on this sequence, ANCORA
addresses file and database modifications separately. To trace file changes, it
builds a provenance graph that reveals all modifications, including those by
exploit-spawned processes. To attribute database operations, a more difficult
challenge due to connection pooling, ANCORA introduces a novel spatiotemporal
anchor. This anchor uses the request's network connection tuple and active time
window to pinpoint exact database operations. With all malicious file and
database operations precisely identified, ANCORA performs a unified rewind and
selective replay recovery. It reverts the system to a clean snapshot taken
before the attack, then selectively re-applies only legitimate operations to
both the file system and database. This completely removes the attack's effects
while preserving concurrent legitimate data. We evaluated ANCORA on 10 web
applications and 20 CVE-based attack scenarios with concurrency up to 150
connections. Experiments demonstrate ANCORA achieves 99.9% recovery accuracy
with manageable overhead: up to 19.8% response latency increase and 17.8% QPS
decrease in worst cases, and recovery throughput of 110.7 database operations
per second and 27.2 affected files per second, effectively preserving
legitimate data.

### 2. [From Defender to Devil? Unintended Risk Interactions Induced by LLM Defenses](http://arxiv.org/pdf/2510.07968v1)

Authors: Xiangtao Meng, Tianshuo Cong, Li Wang, Wenyu Chen, Zheng Li, Shanqing Guo, Xiaoyun Wang

Large Language Models (LLMs) have shown remarkable performance across various
applications, but their deployment in sensitive domains raises significant
concerns. To mitigate these risks, numerous defense strategies have been
proposed. However, most existing studies assess these defenses in isolation,
overlooking their broader impacts across other risk dimensions. In this work,
we take the first step in investigating unintended interactions caused by
defenses in LLMs, focusing on the complex interplay between safety, fairness,
and privacy. Specifically, we propose CrossRiskEval, a comprehensive evaluation
framework to assess whether deploying a defense targeting one risk
inadvertently affects others. Through extensive empirical studies on 14
defense-deployed LLMs, covering 12 distinct defense strategies, we reveal
several alarming side effects: 1) safety defenses may suppress direct responses
to sensitive queries related to bias or privacy, yet still amplify indirect
privacy leakage or biased outputs; 2) fairness defenses increase the risk of
misuse and privacy leakage; 3) privacy defenses often impair safety and
exacerbate bias. We further conduct a fine-grained neuron-level analysis to
uncover the underlying mechanisms of these phenomena. Our analysis reveals the
existence of conflict-entangled neurons in LLMs that exhibit opposing
sensitivities across multiple risk dimensions. Further trend consistency
analysis at both task and neuron levels confirms that these neurons play a key
role in mediating the emergence of unintended behaviors following defense
deployment. We call for a paradigm shift in LLM risk evaluation, toward
holistic, interaction-aware assessment of defense strategies.

### 3. [LLM-Assisted Web Measurements](http://arxiv.org/pdf/2510.08101v1)

Authors: Simone Bozzolan, Stefano Calzavara, Lorenzo Cazzaro

Web measurements are a well-established methodology for assessing the
security and privacy landscape of the Internet. However, existing top lists of
popular websites commonly used as measurement targets are unlabeled and lack
semantic information about the nature of the sites they include. This
limitation makes targeted measurements challenging, as researchers often need
to rely on ad-hoc techniques to bias their datasets toward specific categories
of interest. In this paper, we investigate the use of Large Language Models
(LLMs) as a means to enable targeted web measurement studies through their
semantic understanding capabilities. Building on prior literature, we identify
key website classification tasks relevant to web measurements and construct
datasets to systematically evaluate the performance of different LLMs on these
tasks. Our results demonstrate that LLMs may achieve strong performance across
multiple classification scenarios. We then conduct LLM-assisted web measurement
studies inspired by prior work and rigorously assess the validity of the
resulting research inferences. Our results demonstrate that LLMs can serve as a
practical tool for analyzing security and privacy trends on the Web.

### 4. [TracE2E: Easily Deployable Middleware for Decentralized Data Traceability](http://arxiv.org/pdf/2510.08225v1)

Authors: Daniel Pressensé, Elisavet Kozyri

This paper presents TracE2E, a middleware written in Rust, that can provide
both data explainability and compliance across multiple nodes. By mediating
inputs and outputs of processes, TracE2E records provenance information and
enforces data-protection policies (e.g., confidentiality, integrity) that
depend on the recorded provenance. Unlike existing approaches that necessitate
substantial application modifications, TracE2E is designed for easy integration
into existing and future applications through a wrapper of the Rust standard
library's IO module. We describe how TracE2E consistently records provenance
information across nodes, and we demonstrate how the compliance layer of
TracE2E can accommodate the enforcement of multiple policies.

### 5. [Systematic Assessment of Cache Timing Vulnerabilities on RISC-V Processors](http://arxiv.org/pdf/2510.08272v1)

Authors: Cédrick Austa, Jan Tobias Mühlberg, Jean-Michel Dricot

While interest in the open RISC-V instruction set architecture is growing,
tools to assess the security of concrete processor implementations are lacking.
There are dedicated tools and benchmarks for common microarchitectural
side-channel vulnerabilities for popular processor families such as Intel
x86-64 or ARM, but not for RISC-V. In this paper we describe our efforts in
porting an Intel x86-64 benchmark suite for cache-based timing vulnerabilities
to RISC-V. We then use this benchmark to evaluate the security of three
commercially available RISC-V processors, the T-Head C910 and the SiFive U54
and U74 cores. We observe that the C910 processor exhibits more distinct timing
types than the other processors, leading to the assumption that code running on
the C910 would be exposed to more microarchitectural vulnerability sources. In
addition, our evaluation reveals that $37.5\%$ of the vulnerabilities covered
by the benchmark exist in all processors, while only $6.8\%$ are absent from
all cores. Our work, in particular the ported benchmark, aims to support RISC-V
processor designers to identify leakage sources early in their designs and to
support the development of countermeasures.

### 6. [A Haskell to FHE Transpiler](http://arxiv.org/pdf/2510.08343v1)

Authors: Anne Müller, Mohd Kashif, Nico Döttling

Fully Homomorphic Encryption (FHE) enables the evaluation of programs
directly on encrypted data. However, because only basic operations can be
performed on ciphertexts, programs must be expressed as boolean or arithmetic
circuits. This low-level representation makes implementing applications for FHE
significantly more cumbersome than writing code in a high-level language. To
reduce this burden, several transpilers have been developed that translate
high-level code into circuit representations. In this work, we extend the range
of high-level languages that can target FHE by introducing a transpiler for
Haskell, which converts Haskell programs into Boolean circuits suitable for
homomorphic evaluation. Our second contribution is the automatic
parallelization of these generated circuits. We implement an evaluator that
executes gates in parallel by parallelizing each layer of the circuit. We
demonstrate the effectiveness of our approach on two key applications: Private
Information Retrieval (PIR) and the AES encryption standard. Prior work has
parallelized AES encryption manually. We demonstrate that the automated method
outperforms some but not all manual parallelizations of AES evaluations under
FHE. We achieve an evaluation time of 28 seconds for a parallel execution with
16 threads and an evaluation time of 8 seconds for a parallel execution with
100 threads

### 7. [ExPrESSO: Zero-Knowledge backed Extensive Privacy Preserving Single Sign-on](http://arxiv.org/pdf/2510.08355v1)

Authors: Kaustabh Barman, Fabian Piper, Sanjeet Raj Pandey, Axel Kuepper

User authentication is one of the most important aspects for secure
communication between services and end-users over the Internet. Service
providers leverage Single-Sign On (SSO) to make it easier for their users to
authenticate themselves. However, standardized systems for SSO, such as OIDC,
do not guarantee user privacy as identity providers can track user activities.
We propose a zero-knowledge-based mechanism that integrates with OIDC to let
users authenticate through SSO without revealing information about the service
provider. Our system leverages Groth's zk-SNARK to prove membership of
subscribed service providers without revealing their identity. We adopt a
decentralized and verifiable approach to set up the prerequisites of our
construction that further secures and establishes trust in the system. We set
up high security targets and achieve them with minimal storage and latency
cost, proving that our research can be adopted for production.

### 8. [AI-Driven Post-Quantum Cryptography for Cyber-Resilient V2X Communication in Transportation Cyber-Physical Systems](http://arxiv.org/pdf/2510.08496v1)

Authors: Akid Abrar, Sagar Dasgupta, Mizanur Rahman, Ahmad Alsharif

Transportation Cyber-Physical Systems (TCPS) integrate physical elements,
such as transportation infrastructure and vehicles, with cyber elements via
advanced communication technologies, allowing them to interact seamlessly. This
integration enhances the efficiency, safety, and sustainability of
transportation systems. TCPS rely heavily on cryptographic security to protect
sensitive information transmitted between vehicles, transportation
infrastructure, and other entities within the transportation ecosystem,
ensuring data integrity, confidentiality, and authenticity. Traditional
cryptographic methods have been employed to secure TCPS communications, but the
advent of quantum computing presents a significant threat to these existing
security measures. Therefore, integrating Post-Quantum Cryptography (PQC) into
TCPS is essential to maintain secure and resilient communications. While PQC
offers a promising approach to developing cryptographic algorithms resistant to
quantum attacks, artificial intelligence (AI) can enhance PQC by optimizing
algorithm selection, resource allocation, and adapting to evolving threats in
real-time. AI-driven PQC approaches can improve the efficiency and
effectiveness of PQC implementations, ensuring robust security without
compromising system performance. This chapter introduces TCPS communication
protocols, discusses the vulnerabilities of corresponding communications to
cyber-attacks, and explores the limitations of existing cryptographic methods
in the quantum era. By examining how AI can strengthen PQC solutions, the
chapter presents cyber-resilient communication strategies for TCPS.

### 9. [Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs](http://arxiv.org/pdf/2510.07697v1)

Authors: Man Hu, Xinyi Wu, Zuofeng Suo, Jinbo Feng, Linghui Meng, Yanhao Jia, Anh Tuan Luu, Shuai Zhao

With the rise of advanced reasoning capabilities, large language models
(LLMs) are receiving increasing attention. However, although reasoning improves
LLMs' performance on downstream tasks, it also introduces new security risks,
as adversaries can exploit these capabilities to conduct backdoor attacks.
Existing surveys on backdoor attacks and reasoning security offer comprehensive
overviews but lack in-depth analysis of backdoor attacks and defenses targeting
LLMs' reasoning abilities. In this paper, we take the first step toward
providing a comprehensive review of reasoning-based backdoor attacks in LLMs by
analyzing their underlying mechanisms, methodological frameworks, and
unresolved challenges. Specifically, we introduce a new taxonomy that offers a
unified perspective for summarizing existing approaches, categorizing
reasoning-based backdoor attacks into associative, passive, and active. We also
present defense strategies against such attacks and discuss current challenges
alongside potential directions for future research. This work offers a novel
perspective, paving the way for further exploration of secure and trustworthy
LLM communities.

### 10. [Effective and Stealthy One-Shot Jailbreaks on Deployed Mobile Vision-Language Agents](http://arxiv.org/pdf/2510.07809v1)

Authors: Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu

Large vision-language models (LVLMs) enable autonomous mobile agents to
operate smartphone user interfaces, yet vulnerabilities to UI-level attacks
remain critically understudied. Existing research often depends on conspicuous
UI overlays, elevated permissions, or impractical threat models, limiting
stealth and real-world applicability. In this paper, we present a practical and
stealthy one-shot jailbreak attack that leverages in-app prompt injections:
malicious applications embed short prompts in UI text that remain inert during
human interaction but are revealed when an agent drives the UI via ADB (Android
Debug Bridge). Our framework comprises three crucial components: (1)
low-privilege perception-chain targeting, which injects payloads into malicious
apps as the agent's visual inputs; (2) stealthy user-invisible activation, a
touch-based trigger that discriminates agent from human touches using physical
touch attributes and exposes the payload only during agent operation; and (3)
one-shot prompt efficacy, a heuristic-guided, character-level
iterative-deepening search algorithm (HG-IDA*) that performs one-shot,
keyword-level detoxification to evade on-device safety filters. We evaluate
across multiple LVLM backends, including closed-source services and
representative open-source models within three Android applications, and we
observe high planning and execution hijack rates in single-shot scenarios
(e.g., GPT-4o: 82.5% planning / 75.0% execution). These findings expose a
fundamental security vulnerability in current mobile agents with immediate
implications for autonomous smartphone operation.

### Computer Vision and Pattern Recognition

### 1. [Rectified-CFG++ for Flow Based Models](http://arxiv.org/pdf/2510.07631v1)

Authors: Shreshth Saini, Shashank Gupta, Alan C. Bovik

Classifier-free guidance (CFG) is the workhorse for steering large diffusion
models toward text-conditioned targets, yet its native application to rectified
flow (RF) based models provokes severe off-manifold drift, yielding visual
artifacts, text misalignment, and brittle behaviour. We present
Rectified-CFG++, an adaptive predictor-corrector guidance that couples the
deterministic efficiency of rectified flows with a geometry-aware conditioning
rule. Each inference step first executes a conditional RF update that anchors
the sample near the learned transport path, then applies a weighted conditional
correction that interpolates between conditional and unconditional velocity
fields. We prove that the resulting velocity field is marginally consistent and
that its trajectories remain within a bounded tubular neighbourhood of the data
manifold, ensuring stability across a wide range of guidance strengths.
Extensive experiments on large-scale text-to-image models (Flux, Stable
Diffusion 3/3.5, Lumina) show that Rectified-CFG++ consistently outperforms
standard CFG on benchmark datasets such as MS-COCO, LAION-Aesthetic, and
T2I-CompBench. Project page: https://rectified-cfgpp.github.io/

### 2. [PIT-QMM: A Large Multimodal Model For No-Reference Point Cloud Quality Assessment](http://arxiv.org/pdf/2510.07636v1)

Authors: Shashank Gupta, Gregoire Phillips, Alan C. Bovik

Large Multimodal Models (LMMs) have recently enabled considerable advances in
the realm of image and video quality assessment, but this progress has yet to
be fully explored in the domain of 3D assets. We are interested in using these
models to conduct No-Reference Point Cloud Quality Assessment (NR-PCQA), where
the aim is to automatically evaluate the perceptual quality of a point cloud in
absence of a reference. We begin with the observation that different modalities
of data - text descriptions, 2D projections, and 3D point cloud views - provide
complementary information about point cloud quality. We then construct PIT-QMM,
a novel LMM for NR-PCQA that is capable of consuming text, images and point
clouds end-to-end to predict quality scores. Extensive experimentation shows
that our proposed method outperforms the state-of-the-art by significant
margins on popular benchmarks with fewer training iterations. We also
demonstrate that our framework enables distortion localization and
identification, which paves a new way forward for model explainability and
interactivity. Code and datasets are available at
https://www.github.com/shngt/pit-qmm.

### 3. [Dual-Stream Alignment for Action Segmentation](http://arxiv.org/pdf/2510.07652v1)

Authors: Harshala Gammulle, Clinton Fookes, Sridha Sridharan, Simon Denman

Action segmentation is a challenging yet active research area that involves
identifying when and where specific actions occur in continuous video streams.
Most existing work has focused on single-stream approaches that model the
spatio-temporal aspects of frame sequences. However, recent research has
shifted toward two-stream methods that learn action-wise features to enhance
action segmentation performance. In this work, we propose the Dual-Stream
Alignment Network (DSA Net) and investigate the impact of incorporating a
second stream of learned action features to guide segmentation by capturing
both action and action-transition cues. Communication between the two streams
is facilitated by a Temporal Context (TC) block, which fuses complementary
information using cross-attention and Quantum-based Action-Guided Modulation
(Q-ActGM), enhancing the expressive power of the fused features. To the best of
our knowledge, this is the first study to introduce a hybrid quantum-classical
machine learning framework for action segmentation. Our primary objective is
for the two streams (frame-wise and action-wise) to learn a shared feature
space through feature alignment. This is encouraged by the proposed Dual-Stream
Alignment Loss, which comprises three components: relational consistency,
cross-level contrastive, and cycle-consistency reconstruction losses. Following
prior work, we evaluate DSA Net on several diverse benchmark datasets: GTEA,
Breakfast, 50Salads, and EgoProcel. We further demonstrate the effectiveness of
each component through extensive ablation studies. Notably, DSA Net achieves
state-of-the-art performance, significantly outperforming existing

### 4. [Once Is Enough: Lightweight DiT-Based Video Virtual Try-On via One-Time Garment Appearance Injection](http://arxiv.org/pdf/2510.07654v1)

Authors: Yanjie Pan, Qingdong He, Lidong Wang, Bo Peng, Mingmin Chi

Video virtual try-on aims to replace the clothing of a person in a video with
a target garment. Current dual-branch architectures have achieved significant
success in diffusion models based on the U-Net; however, adapting them to
diffusion models built upon the Diffusion Transformer remains challenging.
Initially, introducing latent space features from the garment reference branch
requires adding or modifying the backbone network, leading to a large number of
trainable parameters. Subsequently, the latent space features of garments lack
inherent temporal characteristics and thus require additional learning. To
address these challenges, we propose a novel approach, OIE (Once is Enough), a
virtual try-on strategy based on first-frame clothing replacement:
specifically, we employ an image-based clothing transfer model to replace the
clothing in the initial frame, and then, under the content control of the
edited first frame, utilize pose and mask information to guide the temporal
prior of the video generation model in synthesizing the remaining frames
sequentially. Experiments show that our method achieves superior parameter
efficiency and computational efficiency while still maintaining leading
performance under these constraints.

### 5. [MONKEY: Masking ON KEY-Value Activation Adapter for Personalization](http://arxiv.org/pdf/2510.07656v1)

Authors: James Baker

Personalizing diffusion models allows users to generate new images that
incorporate a given subject, allowing more control than a text prompt. These
models often suffer somewhat when they end up just recreating the subject
image, and ignoring the text prompt. We observe that one popular method for
personalization, the IP-Adapter automatically generates masks that we
definitively segment the subject from the background during inference. We
propose to use this automatically generated mask on a second pass to mask the
image tokens, thus restricting them to the subject, not the background,
allowing the text prompt to attend to the rest of the image. For text prompts
describing locations and places, this produces images that accurately depict
the subject while definitively matching the prompt. We compare our method to a
few other test time personalization methods, and find our method displays high
prompt and source image alignment.

### 6. [Automatic Text Box Placement for Supporting Typographic Design](http://arxiv.org/pdf/2510.07665v1)

Authors: Jun Muraoka, Daichi Haraguchi, Naoto Inoue, Wataru Shimoda, Kota Yamaguchi, Seiichi Uchida

In layout design for advertisements and web pages, balancing visual appeal
and communication efficiency is crucial. This study examines automated text box
placement in incomplete layouts, comparing a standard Transformer-based method,
a small Vision and Language Model (Phi3.5-vision), a large pretrained VLM
(Gemini), and an extended Transformer that processes multiple images.
Evaluations on the Crello dataset show the standard Transformer-based models
generally outperform VLM-based approaches, particularly when incorporating
richer appearance information. However, all methods face challenges with very
small text or densely populated layouts. These findings highlight the benefits
of task-specific architectures and suggest avenues for further improvement in
automated layout design.

### 7. [Hybrid CNN-BYOL Approach for Fault Detection in Induction Motors Using Thermal Images](http://arxiv.org/pdf/2510.07692v1)

Authors: Tangin Amir Smrity, MD Zahin Muntaqim Hasan Muhammad Kafi, Abu Saleh Musa Miah, Najmul Hassan, Yuichi Okuyama, Nobuyoshi Asai, Taro Suzuki, Jungpil Shin

Induction motors (IMs) are indispensable in industrial and daily life, but
they are susceptible to various faults that can lead to overheating, wasted
energy consumption, and service failure. Early detection of faults is essential
to protect the motor and prolong its lifespan. This paper presents a hybrid
method that integrates BYOL with CNNs for classifying thermal images of
induction motors for fault detection. The thermal dataset used in this work
includes different operating states of the motor, such as normal operation,
overload, and faults. We employed multiple deep learning (DL) models for the
BYOL technique, ranging from popular architectures such as ResNet-50,
DenseNet-121, DenseNet-169, EfficientNetB0, VGG16, and MobileNetV2.
Additionally, we introduced a new high-performance yet lightweight CNN model
named BYOL-IMNet, which comprises four custom-designed blocks tailored for
fault classification in thermal images. Our experimental results demonstrate
that the proposed BYOL-IMNet achieves 99.89\% test accuracy and an inference
time of 5.7 ms per image, outperforming state-of-the-art models. This study
highlights the promising performance of the CNN-BYOL hybrid method in enhancing
accuracy for detecting faults in induction motors, offering a robust
methodology for online monitoring in industrial settings.

### 8. [Mutual Learning for Hashing: Unlocking Strong Hash Functions from Weak Supervision](http://arxiv.org/pdf/2510.07703v1)

Authors: Xiaoxu Ma, Runhao Li, Zhenyu Weng

Deep hashing has been widely adopted for large-scale image retrieval, with
numerous strategies proposed to optimize hash function learning. Pairwise-based
methods are effective in learning hash functions that preserve local similarity
relationships, whereas center-based methods typically achieve superior
performance by more effectively capturing global data distributions. However,
the strength of center-based methods in modeling global structures often comes
at the expense of underutilizing important local similarity information. To
address this limitation, we propose Mutual Learning for Hashing (MLH), a novel
weak-to-strong framework that enhances a center-based hashing branch by
transferring knowledge from a weaker pairwise-based branch. MLH consists of two
branches: a strong center-based branch and a weaker pairwise-based branch.
Through an iterative mutual learning process, the center-based branch leverages
local similarity cues learned by the pairwise-based branch. Furthermore,
inspired by the mixture-of-experts paradigm, we introduce a novel
mixture-of-hash-experts module that enables effective cross-branch interaction,
further enhancing the performance of both branches. Extensive experiments
demonstrate that MLH consistently outperforms state-of-the-art hashing methods
across multiple benchmark datasets.

### 9. [RePainter: Empowering E-commerce Object Removal via Spatial-matting Reinforcement Learning](http://arxiv.org/pdf/2510.07721v1)

Authors: Zipeng Guo, Lichen Ma, Xiaolong Fu, Gaojing Zhou, Lan Yang, Yuchen Zhou, Linkai Liu, Yu He, Ximan Liu, Shiping Dong, Jingling Fu, Zhen Chen, Yu Shi, Junshi Huang, Jason Li, Chao Gou

In web data, product images are central to boosting user engagement and
advertising efficacy on e-commerce platforms, yet the intrusive elements such
as watermarks and promotional text remain major obstacles to delivering clear
and appealing product visuals. Although diffusion-based inpainting methods have
advanced, they still face challenges in commercial settings due to unreliable
object removal and limited domain-specific adaptation. To tackle these
challenges, we propose Repainter, a reinforcement learning framework that
integrates spatial-matting trajectory refinement with Group Relative Policy
Optimization (GRPO). Our approach modulates attention mechanisms to emphasize
background context, generating higher-reward samples and reducing unwanted
object insertion. We also introduce a composite reward mechanism that balances
global, local, and semantic constraints, effectively reducing visual artifacts
and reward hacking. Additionally, we contribute EcomPaint-100K, a high-quality,
large-scale e-commerce inpainting dataset, and a standardized benchmark
EcomPaint-Bench for fair evaluation. Extensive experiments demonstrate that
Repainter significantly outperforms state-of-the-art methods, especially in
challenging scenes with intricate compositions. We will release our code and
weights upon acceptance.

### 10. [SyncHuman: Synchronizing 2D and 3D Generative Models for Single-view Human Reconstruction](http://arxiv.org/pdf/2510.07723v1)

Authors: Wenyue Chen, Peng Li, Wangguandong Zheng, Chengfeng Zhao, Mengfei Li, Yaolong Zhu, Zhiyang Dou, Ronggang Wang, Yuan Liu

Photorealistic 3D full-body human reconstruction from a single image is a
critical yet challenging task for applications in films and video games due to
inherent ambiguities and severe self-occlusions. While recent approaches
leverage SMPL estimation and SMPL-conditioned image generative models to
hallucinate novel views, they suffer from inaccurate 3D priors estimated from
SMPL meshes and have difficulty in handling difficult human poses and
reconstructing fine details. In this paper, we propose SyncHuman, a novel
framework that combines 2D multiview generative model and 3D native generative
model for the first time, enabling high-quality clothed human mesh
reconstruction from single-view images even under challenging human poses.
Multiview generative model excels at capturing fine 2D details but struggles
with structural consistency, whereas 3D native generative model generates
coarse yet structurally consistent 3D shapes. By integrating the complementary
strengths of these two approaches, we develop a more effective generation
framework. Specifically, we first jointly fine-tune the multiview generative
model and the 3D native generative model with proposed pixel-aligned 2D-3D
synchronization attention to produce geometrically aligned 3D shapes and 2D
multiview images. To further improve details, we introduce a feature injection
mechanism that lifts fine details from 2D multiview images onto the aligned 3D
shapes, enabling accurate and high-fidelity reconstruction. Extensive
experiments demonstrate that SyncHuman achieves robust and photo-realistic 3D
human reconstruction, even for images with challenging poses. Our method
outperforms baseline methods in geometric accuracy and visual fidelity,
demonstrating a promising direction for future 3D generation models.

### Computers and Society

### 1. [Exploring the Viability of the Updated World3 Model for Examining the Impact of Computing on Planetary Boundaries](http://arxiv.org/pdf/2510.07634v1)

Authors: Nara Guliyeva, Eshta Bhardwaj, Christoph Becker

The influential Limits to Growth report introduced a system dynamics-based
model to demonstrate global dynamics of the world's population, industry,
natural resources, agriculture, and pollution between 1900-2100. In current
times, the rapidly expanding trajectory of data center development, much of it
linked to AI, uses increasing amounts of natural resources. The extraordinary
amount of resources claimed warrants the question of how computing trajectories
contribute to exceeding planetary boundaries. Based on the general robustness
of the World3-03 model and its influence in serving as a foundation for current
climate frameworks, we explore whether the model is a viable method to
quantitatively simulate the impact of data centers on limits to growth. Our
paper explores whether the World3-03 model is a feasible method for reflecting
on these dynamics by adding new variables to the model in order to simulate a
new AI-augmented scenario. We find that through our addition of AI-related
variables (such as increasing data center development) impacting pollution in
the World3-03 model, we can observe the expected changes to dynamics,
demonstrating the viability of the World3-03 model for examining AI's impact on
planetary boundaries. We detail future research opportunities for using the
World3-03 model to explore the relationships between increasing
resource-intensive computing and the resulting impacts to the environment in a
quantitative way given its feasibility.

### 2. [Does everyone have a price? Understanding people's attitude towards online and offline price discrimination](http://arxiv.org/pdf/2510.08246v1)

Authors: Joost Poort, Frederik J. Zuiderveen Borgesius

Online stores can present a different price to each customer. Such
algorithmic personalised pricing can lead to advanced forms of price
discrimination based on the characteristics and behaviour of individual
consumers. We conducted two consumer surveys among a representative sample of
the Dutch population (N=1233 and N=1202), to analyse consumer attitudes towards
a list of examples of price discrimination and dynamic pricing. A vast majority
finds online price discrimination unfair and unacceptable, and thinks it should
be banned. However, some pricing strategies that have been used by companies
for decades are almost equally unpopular. We analyse the results to better
understand why people dislike many types of price discrimination.

### 3. [The Right to Communications Confidentiality in Europe: Protecting Privacy, Freedom of Expression, and Trust](http://arxiv.org/pdf/2510.08247v1)

Authors: Frederik J. Zuiderveen Borgesius, Wilfred Steenbruggen

In the European Union, the General Data Protection Regulation (GDPR) provides
comprehensive rules for the processing of personal data. In addition, the EU
lawmaker intends to adopt specific rules to protect confidentiality of
communications, in a separate ePrivacy Regulation. Some have argued that there
is no need for such additional rules for communications confidentiality. This
Article discusses the protection of the right to confidentiality of
communications in Europe. We look at the right's origins to assess the
rationale for protecting it. We also analyze how the right is currently
protected under the European Convention on Human Rights and under EU law. We
show that at its core the right to communications confidentiality protects
three individual and collective values: privacy, freedom of expression, and
trust in communication services. The right aims to ensure that individuals and
organizations can safely entrust communication to service providers. Initially,
the right protected only postal letters, but it has gradually developed into a
strong safeguard for the protection of confidentiality of communications,
regardless of the technology used. Hence, the right does not merely serve
individual privacy interests, but also other more collective interests that are
crucial for the functioning of our information society. We conclude that
separate EU rules to protect communications confidentiality, next to the GDPR,
are justified and necessary.

### 4. [Human-Centered Development of Indicators for Self-Service Learning Analytics: A Transparency through Exploration Approach](http://arxiv.org/pdf/2510.08395v1)

Authors: Shoeb Joarder, Mohamed Amine Chatti

The aim of learning analytics is to turn educational data into insights,
decisions, and actions to improve learning and teaching. The reasoning of the
provided insights, decisions, and actions is often not transparent to the
end-user, and this can lead to trust and acceptance issues when interventions,
feedback, and recommendations fail. In this paper, we shed light on achieving
transparent learning analytics by following a transparency through exploration
approach. To this end, we present the design, implementation, and evaluation
details of the Indicator Editor, which aims to support self-service learning
analytics by empowering end-users to take control of the indicator
implementation process. We systematically designed and implemented the
Indicator Editor through an iterative human-centered design (HCD) approach.
Further, we conducted a qualitative user study (n=15) to investigate the impact
of following a self-service learning analytics approach on the users'
perception of and interaction with the Indicator Editor. Our study showed
qualitative evidence that supporting user interaction and providing user
control in the indicator implementation process can have positive effects on
different crucial aspects of learning analytics, namely transparency, trust,
satisfaction, and acceptance.

### 5. [Textual Entailment and Token Probability as Bias Evaluation Metrics](http://arxiv.org/pdf/2510.07662v1)

Authors: Virginia K. Felkner, Allison Lim, Jonathan May

Measurement of social bias in language models is typically by token
probability (TP) metrics, which are broadly applicable but have been criticized
for their distance from real-world langugage model use cases and harms. In this
work, we test natural language inference (NLI) as a more realistic alternative
bias metric. We show that, curiously, NLI and TP bias evaluation behave
substantially differently, with very low correlation among different NLI
metrics and between NLI and TP metrics. We find that NLI metrics are more
likely to detect "underdebiased" cases. However, NLI metrics seem to be more
brittle and sensitive to wording of counterstereotypical sentences than TP
approaches. We conclude that neither token probability nor natural language
inference is a "better" bias metric in all cases, and we recommend a
combination of TP, NLI, and downstream bias evaluations to ensure comprehensive
evaluation of language models.
  Content Warning: This paper contains examples of anti-LGBTQ+ stereotypes.

### 6. [Evaluating LLM-Generated Legal Explanations for Regulatory Compliance in Social Media Influencer Marketing](http://arxiv.org/pdf/2510.08111v1)

Authors: Haoyang Gui, Thales Bertaglia, Taylor Annabell, Catalina Goanta, Tjomme Dooper, Gerasimos Spanakis

The rise of influencer marketing has blurred boundaries between organic
content and sponsored content, making the enforcement of legal rules relating
to transparency challenging. Effective regulation requires applying legal
knowledge with a clear purpose and reason, yet current detection methods of
undisclosed sponsored content generally lack legal grounding or operate as
opaque "black boxes". Using 1,143 Instagram posts, we compare gpt-5-nano and
gemini-2.5-flash-lite under three prompting strategies with controlled levels
of legal knowledge provided. Both models perform strongly in classifying
content as sponsored or not (F1 up to 0.93), though performance drops by over
10 points on ambiguous cases. We further develop a taxonomy of reasoning
errors, showing frequent citation omissions (28.57%), unclear references
(20.71%), and hidden ads exhibiting the highest miscue rate (28.57%). While
adding regulatory text to the prompt improves explanation quality, it does not
consistently improve detection accuracy. The contribution of this paper is
threefold. First, it makes a novel addition to regulatory compliance technology
by providing a taxonomy of common errors in LLM-generated legal reasoning to
evaluate whether automated moderation is not only accurate but also legally
robust, thereby advancing the transparent detection of influencer marketing
content. Second, it features an original dataset of LLM explanations annotated
by two students who were trained in influencer marketing law. Third, it
combines quantitative and qualitative evaluation strategies for LLM
explanations and critically reflects on how these findings can support
advertising regulatory bodies in automating moderation processes on a solid
legal foundation.

### 7. [Multimodal Safety Evaluation in Generative Agent Social Simulations](http://arxiv.org/pdf/2510.07709v1)

Authors: Alhim Vera, Karen Sanchez, Carlos Hinojosa, Haidar Bin Hamid, Donghoon Kim, Bernard Ghanem

Can generative agents be trusted in multimodal environments? Despite advances
in large language and vision-language models that enable agents to act
autonomously and pursue goals in rich settings, their ability to reason about
safety, coherence, and trust across modalities remains limited. We introduce a
reproducible simulation framework for evaluating agents along three dimensions:
(1) safety improvement over time, including iterative plan revisions in
text-visual scenarios; (2) detection of unsafe activities across multiple
categories of social situations; and (3) social dynamics, measured as
interaction counts and acceptance ratios of social exchanges. Agents are
equipped with layered memory, dynamic planning, multimodal perception, and are
instrumented with SocialMetrics, a suite of behavioral and structural metrics
that quantifies plan revisions, unsafe-to-safe conversions, and information
diffusion across networks. Experiments show that while agents can detect direct
multimodal contradictions, they often fail to align local revisions with global
safety, reaching only a 55 percent success rate in correcting unsafe plans.
Across eight simulation runs with three models - Claude, GPT-4o mini, and
Qwen-VL - five agents achieved average unsafe-to-safe conversion rates of 75,
55, and 58 percent, respectively. Overall performance ranged from 20 percent in
multi-risk scenarios with GPT-4o mini to 98 percent in localized contexts such
as fire/heat with Claude. Notably, 45 percent of unsafe actions were accepted
when paired with misleading visuals, showing a strong tendency to overtrust
images. These findings expose critical limitations in current architectures and
provide a reproducible platform for studying multimodal safety, coherence, and
social dynamics.

### 8. [Towards Meaningful Transparency in Civic AI Systems](http://arxiv.org/pdf/2510.07889v1)

Authors: Dave Murray-Rust, Kars Alfrink, Cristina Zaga

Artificial intelligence has become a part of the provision of governmental
services, from making decisions about benefits to issuing fines for parking
violations. However, AI systems rarely live up to the promise of neutral
optimisation, creating biased or incorrect outputs and reducing the agency of
both citizens and civic workers to shape the way decisions are made.
Transparency is a principle that can both help subjects understand decisions
made about them and shape the processes behind those decisions. However,
transparency as practiced around AI systems tends to focus on the production of
technical objects that represent algorithmic aspects of decision making. These
are often difficult for publics to understand, do not connect to potential for
action, and do not give insight into the wider socio-material context of
decision making. In this paper, we build on existing approaches that take a
human-centric view on AI transparency, combined with a socio-technical systems
view, to develop the concept of meaningful transparency for civic AI systems:
transparencies that allow publics to engage with AI systems that affect their
lives, connecting understanding with potential for action.

### 9. [VideoNorms: Benchmarking Cultural Awareness of Video Language Models](http://arxiv.org/pdf/2510.08543v1)

Authors: Nikhil Reddy Varimalla, Yunfei Xu, Arkadiy Saakyan, Meng Fan Wang, Smaranda Muresan

As Video Large Language Models (VideoLLMs) are deployed globally, they
require understanding of and grounding in the relevant cultural background. To
properly assess these models' cultural awareness, adequate benchmarks are
needed. We introduce VideoNorms, a benchmark of over 1000 (video clip, norm)
pairs from US and Chinese cultures annotated with socio-cultural norms grounded
in speech act theory, norm adherence and violations labels, and verbal and
non-verbal evidence. To build VideoNorms, we use a human-AI collaboration
framework, where a teacher model using theoretically-grounded prompting
provides candidate annotations and a set of trained human experts validate and
correct the annotations. We benchmark a variety of open-weight VideoLLMs on the
new dataset which highlight several common trends: 1) models performs worse on
norm violation than adherence; 2) models perform worse w.r.t Chinese culture
compared to the US culture; 3) models have more difficulty in providing
non-verbal evidence compared to verbal for the norm adhere/violation label and
struggle to identify the exact norm corresponding to a speech-act; and 4)
unlike humans, models perform worse in formal, non-humorous contexts. Our
findings emphasize the need for culturally-grounded video language model
training - a gap our benchmark and framework begin to address.

### Databases

### 1. [TCDRM: A Tenant Budget-Aware Data Replication Framework for Multi-Cloud Computing](http://arxiv.org/pdf/2510.07833v1)

Authors: Santatra Hagamalala Bernardin, Riad Mokadem, Franck Morvan, Hasinarivo Ramanana, Hasimandimby Rakotoarivelo

Multi-cloud computing systems face significant challenges in ensuring
acceptable performance while adhering to tenant budget requirements. This paper
proposes a tenant budget-aware (tenant-centric) data replication framework for
Multi-Cloud Computing (TCDRM). The proposed strategy dynamically creates data
replicas based on predefined thresholds for response time, economic budget of
the tenant and data popularity. TCDRM employs a heuristic replica placement
algorithm that leverages the diverse pricing structures of multiple cloud
providers. The TCDRM strategy aims to maintain the required performance without
exceeding the tenant's budget by taking advantage of the capabilities offered
by multicloud environments. The middleware considered acts as an intermediary
between tenants and multiple cloud providers, facilitating intelligent replica
placement decisions. To achieve this, the proposed TCDRM strategy defines
strict thresholds for tenant budget and response time. A performance evaluation
is conducted to validate the effectiveness of the strategy. The results show
that our approach effectively meets tenant performance objectives while
respecting their economic constraints. Bandwidth consumption is reduced by up
to 78% compared to non-replicated approaches, and average response time for
complex queries is decreased by 51%, all while adhering to tenant budget
limitations.

### 2. [MobilityDuck: Mobility Data Management with DuckDB](http://arxiv.org/pdf/2510.07963v1)

Authors: Nhu Ngoc Hoang, Ngoc Hoa Pham, Viet Phuong Hoang, Esteban Zimányi

The analytics of spatiotemporal data is increasingly important for mobility
analytics. Despite extensive research on moving object databases (MODs), few
systems are ready on production or lightweight enough for analytics. MobilityDB
is a notable system that extends PostgreSQL with spatiotemporal data, but it
inherits complexity of the architecture as well. In this paper, we present
MobilityDuck, a DuckDB extension that integrates the MEOS library to provide
support spatiotemporal and other temporal data types in DuckDB. MobilityDuck
leverages DuckDB's lightweight, columnar, in-memory executable properties to
deliver efficient analytics. To the best of our knowledge, no existing
in-memory or embedded analytical system offers native spatiotemporal types and
continuous trajectory operators as MobilityDuck does. We evaluate MobilityDuck
using the BerlinMOD-Hanoi benchmark dataset and compare its performance to
MobilityDB. Our results show that MobilityDuck preserves the expressiveness of
spatiotemporal queries while benefiting from DuckDB's in-memory, columnar
architecture.

### 3. [ZeroCard: Cardinality Estimation with Zero Dependence on Target Databases -- No Data, No Query, No Retraining](http://arxiv.org/pdf/2510.07983v1)

Authors: Xianghong Xu, Rong Kang, Xiao He, Lei Zhang, Jianjun Chen, Tieying Zhang

Cardinality estimation is a fundamental task in database systems and plays a
critical role in query optimization. Despite significant advances in
learning-based cardinality estimation methods, most existing approaches remain
difficult to generalize to new datasets due to their strong dependence on raw
data or queries, thus limiting their practicality in real scenarios. To
overcome these challenges, we argue that semantics in the schema may benefit
cardinality estimation, and leveraging such semantics may alleviate these
dependencies. To this end, we introduce ZeroCard, the first semantics-driven
cardinality estimation method that can be applied without any dependence on raw
data access, query logs, or retraining on the target database. Specifically, we
propose to predict data distributions using schema semantics, thereby avoiding
raw data dependence. Then, we introduce a query template-agnostic
representation method to alleviate query dependence. Finally, we construct a
large-scale query dataset derived from real-world tables and pretrain ZeroCard
on it, enabling it to learn cardinality from schema semantics and predicate
representations. After pretraining, ZeroCard's parameters can be frozen and
applied in an off-the-shelf manner. We conduct extensive experiments to
demonstrate the distinct advantages of ZeroCard and show its practical
applications in query optimization. Its zero-dependence property significantly
facilitates deployment in real-world scenarios.

### 4. [Implementing Semantic Join Operators Efficiently](http://arxiv.org/pdf/2510.08489v1)

Authors: Immanuel Trummer

Semantic query processing engines often support semantic joins, enabling
users to match rows that satisfy conditions specified in natural language. Such
join conditions can be evaluated using large language models (LLMs) that solve
novel tasks without task-specific training.
  Currently, many semantic query processing engines implement semantic joins
via nested loops, invoking the LLM to evaluate the join condition on row pairs.
Instead, this paper proposes a novel algorithm, inspired by the block nested
loops join operator implementation in traditional database systems. The
proposed algorithm integrates batches of rows from both input tables into a
single prompt. The goal of the LLM invocation is to identify all matching row
pairs in the current input. The paper introduces formulas that can be used to
optimize the size of the row batches, taking into account constraints on the
size of the LLM context window (limiting both input and output size). An
adaptive variant of the proposed algorithm refers to cases in which the size of
the output is difficult to estimate. A formal analysis of asymptotic processing
costs, as well as empirical results, demonstrates that the proposed approach
reduces costs significantly and performs well compared to join implementations
used by recent semantic query processing engines.

### 5. [Detecting Legend Items on Historical Maps Using GPT-4o with In-Context Learning](http://arxiv.org/pdf/2510.08385v1)

Authors: Sofia Kirsanova, Yao-Yi Chiang, Weiwei Duan

Historical map legends are critical for interpreting cartographic symbols.
However, their inconsistent layouts and unstructured formats make automatic
extraction challenging. Prior work focuses primarily on segmentation or general
optical character recognition (OCR), with few methods effectively matching
legend symbols to their corresponding descriptions in a structured manner. We
present a method that combines LayoutLMv3 for layout detection with GPT-4o
using in-context learning to detect and link legend items and their
descriptions via bounding box predictions. Our experiments show that GPT-4 with
structured JSON prompts outperforms the baseline, achieving 88% F-1 and 85%
IoU, and reveal how prompt design, example counts, and layout alignment affect
performance. This approach supports scalable, layout-aware legend parsing and
improves the indexing and searchability of historical maps across various
visual styles.

### 6. [Large-scale spatial variable gene atlas for spatial transcriptomics](http://arxiv.org/pdf/2510.07653v1)

Authors: Jiawen Chen, Jinwei Zhang, Dongshen Peng, Yutong Song, Aitong Ruan, Yun Li, Didong Li

Spatial variable genes (SVGs) reveal critical information about tissue
architecture, cellular interactions, and disease microenvironments. As spatial
transcriptomics (ST) technologies proliferate, accurately identifying SVGs
across diverse platforms, tissue types, and disease contexts has become both a
major opportunity and a significant computational challenge. Here, we present a
comprehensive benchmarking study of 20 state-of-the-art SVG detection methods
using human slides from STimage-1K4M, a large-scale resource of ST data
comprising 662 slides from more than 18 tissue types. We evaluate each method
across a range of biologically and technically meaningful criteria, including
recovery of pathologist-annotated domain-specific markers, cross-slide
reproducibility, scalability to high-resolution data, and robustness to
technical variation. Our results reveal marked differences in performance
depending on tissue type, spatial resolution, and study design. Beyond
benchmarking, we construct the first cross-tissue atlas of SVGs, enabling
comparative analysis of spatial gene programs across cancer and normal tissues.
We observe similarities between pairs of tissues that reflect developmental and
functional relationships, such as high overlap between thymus and lymph node,
and uncover spatial gene programs associated with metastasis, immune
infiltration, and tissue-of-origin identity in cancer. Together, our work
defines a framework for evaluating and interpreting spatial gene expression and
establishes a reference resource for the ST community.

### Distributed, Parallel, and Cluster Computing

### 1. [A Multi-Simulation Bridge for IoT Digital Twins](http://arxiv.org/pdf/2510.08164v1)

Authors: Marco Picone, Samuele Burattini, Marco Melloni, Prasad Talasila, Davide Ziglioli, Matteo Martinelli, Nicola Bicocchi, Alessandro Ricci, Peter Gorm Larsen

The increasing capabilities of Digital Twins (DTs) in the context of the
Internet of Things (IoT) and Industrial IoT (IIoT) call for seamless
integration with simulation platforms to support system design, validation, and
real-time operation. This paper introduces the concept, design, and
experimental evaluation of the DT Simulation Bridge - a software framework that
enables diverse interaction patterns between active DTs and simulation
environments. The framework supports both the DT development lifecycle and the
incorporation of simulations during active operation. Through bidirectional
data exchange, simulations can update DT models dynamically, while DTs provide
real-time feedback to adapt simulation parameters. We describe the
architectural design and core software components that ensure flexible
interoperability and scalable deployment. Experimental results show that the DT
Simulation Bridge enhances design agility, facilitates virtual commissioning,
and supports live behavioral analysis under realistic conditions, demonstrating
its effectiveness across a range of industrial scenarios.

### 2. [Towards Energy-Efficient Serverless Computing with Hardware Isolation](http://arxiv.org/pdf/2510.08180v1)

Authors: Natalie Carl, Tobias Pfandzelter, David Bermbach

Serverless computing provides just-in-time infrastructure provisioning with
rapid elasticity and a finely-grained pricing model. As full control of
resource allocation is in the hands of the cloud provider and applications only
consume resources when they actually perform work, we believe that serverless
computing is uniquely positioned to maximize energy efficiency.
  However, the focus of current serverless platforms is to run hundreds or
thousands of serverless functions from different tenants on traditional server
hardware, requiring expensive software isolation mechanisms and a high degree
of overprovisioning, i.e., idle servers, to anticipate load spikes. With shared
caches, high clock frequencies, and many-core architectures, servers today are
optimized for large, singular workloads but not to run thousands of isolated
functions.
  We propose rethinking the serverless hardware architecture to align it with
the requirements of serverless software. Specifically, we propose using
hardware isolation with individual processors per function instead of software
isolation resulting in a serverless hardware stack that consumes energy only
when an application actually performs work. In preliminary evaluation with real
hardware and a typical serverless workload we find that this could reduce
energy consumption overheads by 90.63% or an average 70.8MW.

### 3. [Distributed Resource Selection for Self-Organising Cloud-Edge Systems](http://arxiv.org/pdf/2510.08228v1)

Authors: Quentin Renau, Amjad Ullah, Emma Hart

This paper presents a distributed resource selection mechanism for diverse
cloud-edge environments, enabling dynamic and context-aware allocation of
resources to meet the demands of complex distributed applications. By
distributing the decision-making process, our approach ensures efficiency,
scalability, and resilience in highly dynamic cloud-edge environments where
centralised coordination becomes a bottleneck. The proposed mechanism aims to
function as a core component of a broader, distributed, and self-organising
orchestration system that facilitates the intelligent placement and adaptation
of applications in real-time. This work leverages a consensus-based mechanism
utilising local knowledge and inter-agent collaboration to achieve efficient
results without relying on a central controller, thus paving the way for
distributed orchestration. Our results indicate that computation time is the
key factor influencing allocation decisions. Our approach consistently delivers
rapid allocations without compromising optimality or incurring additional cost,
achieving timely results at scale where exhaustive search is infeasible and
centralised heuristics run up to 30 times slower.

### 4. [FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](http://arxiv.org/pdf/2510.07664v1)

Authors: Yunbo Li, Jiaping Gui, Zhihang Deng, Fanchao Meng, Yue Wu

Federated learning (FL) enables collaborative model training across multiple
parties without sharing raw data, with semi-asynchronous FL (SAFL) emerging as
a balanced approach between synchronous and asynchronous FL. However, SAFL
faces significant challenges in optimizing both gradient-based (e.g., FedSGD)
and model-based (e.g., FedAvg) aggregation strategies, which exhibit distinct
trade-offs in accuracy, convergence speed, and stability. While gradient
aggregation achieves faster convergence and higher accuracy, it suffers from
pronounced fluctuations, whereas model aggregation offers greater stability but
slower convergence and suboptimal accuracy. This paper presents FedQS, the
first framework to theoretically analyze and address these disparities in SAFL.
FedQS introduces a divide-and-conquer strategy to handle client heterogeneity
by classifying clients into four distinct types and adaptively optimizing their
local training based on data distribution characteristics and available
computational resources. Extensive experiments on computer vision, natural
language processing, and real-world tasks demonstrate that FedQS achieves the
highest accuracy, attains the lowest loss, and ranks among the fastest in
convergence speed, outperforming state-of-the-art baselines. Our work bridges
the gap between aggregation strategies in SAFL, offering a unified solution for
stable, accurate, and efficient federated learning. The code and datasets are
available at https://anonymous.4open.science/r/FedQS-EDD6.

### 5. [Adaptive Execution Scheduler for DataDios SmartDiff](http://arxiv.org/pdf/2510.07811v1)

Authors: Aryan Poduri

We present an adaptive scheduler for a single differencing engine (SmartDiff)
with two execution modes: (i) in-memory threads and (ii) Dask based
parallelism. The scheduler continuously tunes batch size and worker/thread
count within fixed CPU and memory budgets to minimize p95 latency. A
lightweight preflight profiler estimates bytes/row and I/O rate; an online
cost/memory model prunes unsafe actions; and a guarded hill-climb policy favors
lower latency with backpressure and straggler mitigation. Backend selection is
gated by a conservative working-set estimate so that in-memory execution is
chosen when safe, otherwise Dask is used. Across synthetic and public tabular
benchmarks, the scheduler reduces p95 latency by 23 to 28 percent versus a
tuned warm-up heuristic (and by 35 to 40 percent versus fixed grid baselines),
while lowering peak memory by 16 to 22 percent (25 to 32 percent vs. fixed)
with zero OOMs and comparable throughput.

### 6. [Decentralised Blockchain Management Through Digital Twins](http://arxiv.org/pdf/2510.07901v1)

Authors: Georgios Diamantopoulos, Nikos Tziritas, Rami Bahsoon, Georgios Theodoropoulos

The necessity of blockchain systems to remain decentralised limits current
solutions to blockchain governance and dynamic management, forcing a trade-off
between control and decentralisation. In light of the above, this work proposes
a dynamic and decentralised blockchain management mechanism based on digital
twins. To ensure decentralisation, the proposed mechanism utilises multiple
digital twins that the system's stakeholders control. To facilitate
decentralised decision-making, the twins are organised in a secondary
blockchain system that orchestrates agreement on, and propagation of decisions
to the managed blockchain. This enables the management of blockchain systems
without centralised control. A preliminary evaluation of the performance and
impact of the overheads introduced by the proposed mechanism is conducted
through simulation. The results demonstrate the proposed mechanism's ability to
reach consensus on decisions quickly and reconfigure the primary blockchain
with minimal overhead.

### 7. [SketchGuard: Scaling Byzantine-Robust Decentralized Federated Learning via Sketch-Based Screening](http://arxiv.org/pdf/2510.07922v1)

Authors: Murtaza Rangwala, Farag Azzedin, Richard O. Sinnott, Rajkumar Buyya

Decentralized Federated Learning (DFL) enables privacy-preserving
collaborative training without centralized servers, but remains vulnerable to
Byzantine attacks where malicious clients submit corrupted model updates.
Existing Byzantine-robust DFL defenses rely on similarity-based neighbor
screening that requires every client to exchange and compare complete
high-dimensional model vectors with all neighbors in each training round,
creating prohibitive communication and computational costs that prevent
deployment at web scale. We propose SketchGuard, a general framework that
decouples Byzantine filtering from model aggregation through sketch-based
neighbor screening. SketchGuard compresses $d$-dimensional models to
$k$-dimensional sketches ($k \ll d$) using Count Sketch for similarity
comparisons, then selectively fetches full models only from accepted neighbors,
reducing per-round communication complexity from $O(d|N_i|)$ to $O(k|N_i| +
d|S_i|)$, where $|N_i|$ is the neighbor count and $|S_i| \le |N_i|$ is the
accepted neighbor count. We establish rigorous convergence guarantees in both
strongly convex and non-convex settings, proving that Count Sketch compression
preserves Byzantine resilience with controlled degradation bounds where
approximation errors introduce only a $(1+O(\epsilon))$ factor in the effective
threshold parameter. Comprehensive experiments across multiple datasets,
network topologies, and attack scenarios demonstrate that SketchGuard maintains
identical robustness to state-of-the-art methods while reducing computation
time by up to 82% and communication overhead by 50-70% depending on filtering
effectiveness, with benefits scaling multiplicatively with model dimensionality
and network connectivity. These results establish the viability of sketch-based
compression as a fundamental enabler of robust DFL at web scale.

### 8. [From Tokens to Layers: Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill](http://arxiv.org/pdf/2510.08055v1)

Authors: Gunjun Lee, Jiwon Kim, Jaiyoung Park, Younjoo Lee, Jung Ho Ahn

Large Language Model (LLM) inference in production must meet stringent
service-level objectives for both time-to-first-token (TTFT) and
time-between-token (TBT) while maximizing throughput under fixed compute,
memory, and interconnect budgets. Modern serving systems adopt stall-free
scheduling techniques such as chunked prefill, which splits long prompt
processing along the token dimension and interleaves prefill with ongoing
decode iterations. While effective at stabilizing TBT, chunked prefill incurs
substantial overhead in Mixture-of-Experts (MoE) models: redundant expert
weight loads increase memory traffic by up to 39% and inflate energy
consumption. We propose layered prefill, a new scheduling paradigm that treats
transformer layer groups as the primary scheduling unit. By vertically
partitioning the model into contiguous layer groups and interleaving prefill
and decode across the groups, layered prefill sustains stall-free decoding
while eliminating chunk-induced MoE weight reloads. It reduces off-chip
bandwidth demand, lowering TTFT by up to 70%, End-to-End latency by 41% and
per-token energy by up to 22%. Evaluations show that layered prefill
consistently improves the TTFT--TBT Pareto frontier over chunked prefill,
reducing expert-load traffic and energy cost while maintaining stall-free
decoding. Overall, shifting the scheduling axis from tokens to layers unlocks a
new operating regime for high-efficiency, energy-aware LLM serving in
co-located environments.

### 9. [When Light Bends to the Collective Will: A Theory and Vision for Adaptive Photonic Scale-up Domains](http://arxiv.org/pdf/2510.08072v1)

Authors: Vamsi Addanki

As chip-to-chip silicon photonics gain traction for their bandwidth and
energy efficiency, collective communication has emerged as a critical
bottleneck in scale-up systems. Programmable photonic interconnects offer a
promising path forward: by dynamically reconfiguring the fabric, they can
establish direct, high-bandwidth optical paths between communicating endpoints
-- \emph{synchronously and guided by the structure of collective operations}
(e.g., AllReduce). However, realizing this vision -- \emph{when light bends to
the collective will} -- requires navigating a fundamental trade-off between
reconfiguration delay and the performance gains of adaptive topologies.
  In this paper, we present a simple theoretical framework for adaptive
photonic scale-up domains that makes this trade-off explicit and clarifies when
reconfiguration is worthwhile. Along the way, we highlight a connection -- not
surprising but still powerful -- between the Birkhoff--von Neumann (BvN)
decomposition, maximum concurrent flow (a classic measure of network
throughput), and the well-known $\alpha$-$\beta$ cost model for collectives.
Finally, we outline a research agenda in algorithm design and systems
integration that can build on this foundation.

### 10. [BlockSDN: Towards a High-Performance Blockchain via Software-Defined Cross Networking optimization](http://arxiv.org/pdf/2510.08139v1)

Authors: Wenyang Jia, Jingjing Wang, Ziwei Yan, Xiangli Peng, Guohui Yuan

The scalability of blockchain systems is constrained by inefficient P2P
broadcasting, as most existing optimizations focus only on the logical layer
without considering physical network conditions. To address this, we propose
BlockSDN, the first SDN-based integrated architecture for blockchain. BlockSDN
employs a distributed control plane for a global network view, a graph engine
for hierarchical clustering, and a hybrid macro-micro neighbor selection with
hierarchical broadcasting. A dedicated simulation platform shows that BlockSDN
reduces global block synchronization time by 65% and 55% compared to Gossip and
Mercury, respectively.These results highlight the potential of SDN-enabled
cross-layer coordination to significantly enhance blockchain scalability and
performance.

### Discrete Mathematics

### 1. [Isolation of non-triangle cycles in graphs](http://arxiv.org/pdf/2510.08361v1)

Authors: Peter Borg, Dayle Scicluna

Given a set $\mathcal{F}$ of graphs, we call a copy of a graph in
$\mathcal{F}$ an $\mathcal{F}$-graph. The $\mathcal{F}$-isolation number of a
graph $G$, denoted by $\iota(G, \mathcal{F})$, is the size of a smallest set
$D$ of vertices of $G$ such that the closed neighbourhood of $D$ intersects the
vertex sets of the $\mathcal{F}$-graphs contained by $G$ (equivalently,
$G-N[D]$ contains no $\mathcal{F}$-graph). Let $\mathcal{C}$ be the set of
cycles, and let $\mathcal{C}'$ be the set of non-triangle cycles (that is,
cycles of length at least $4$). Let $G$ be a connected graph having exactly $n$
vertices and $m$ edges. The first author proved that $\iota(G,\mathcal{C}) \leq
n/4$ if $G$ is not a triangle. Bartolo and the authors proved that
$\iota(G,\{C_4\}) \leq n/5$ if $G$ is not a copy of one of nine graphs. Various
authors proved that $\iota(G,\mathcal{C}) \leq (m+1)/5$ if $G$ is not a
triangle. We prove that $\iota(G,\mathcal{C}') \leq (m+1)/6$ if $G$ is not a
$4$-cycle. Zhang and Wu established this for the case where $G$ is
triangle-free. Our result yields the inequality $\iota(G,\{C_4\}) \leq (m+1)/6$
of Wei, Zhang and Zhao. These bounds are attained by infinitely many
(non-isomorphic) graphs. The proof of our inequality hinges on also determining
the graphs attaining the bound.

### 2. [Symmetric Rule-Based Achlioptas Processes for Random $k$-SAT](http://arxiv.org/pdf/2510.07870v1)

Authors: Arnab Chatterjee

Inspired by the "power-of-two-choices" model from random graphs, we
investigate the possibility of limited choices of online clause choices that
could shift the satisfiability threshold in random $k$-SAT.Here, we introduce
an assignment symmetric, non-adaptive, topology-oblivious online rule called
\emph{MIDDLE-HEAVY}, that prioritizes balanced sign profile clauses.Upon
applying a biased $2$-SAT projection and a two-type branching process
certificate, we derive closed-form expressions for the shifted thresholds
$\alpha_{\textbf{SYM}}(k,\ell)$ for this algorithm.We show that minimal choices
$\ell=5$ for $k=4$, $\ell=4$ for $k=5$, and $\ell=3$ for $k\ge 6$ suffice to
exceed the asymptotic first-moment upper bound $\sim 2^k \ln 2$ for random
$k$-SAT.Moreover, to bridge the gap with biased assignment rules used in
maximum of the previous works in this context, we propose a hybrid symmetric
biased rule that achieves thresholds comparable to prior work while maintaining
symmetry.Our results advance the understanding of Achlioptas processes in
random CSPs beyond classical graph-theoretic settings.

### 3. [A Graph Width Perspective on Partially Ordered Hamiltonian Paths and Cycles II: Vertex and Edge Deletion Numbers](http://arxiv.org/pdf/2510.08378v1)

Authors: Jesse Beisegel, Katharina Klost, Kristin Knorr, Fabienne Ratajczak, Robert Scheffler

We consider the problem of finding a Hamiltonian path or cycle with
precedence constraints in the form of a partial order on the vertex set. We
study the complexity for graph width parameters for which the ordinary problems
$\mathsf{Hamiltonian\ Path}$ and $\mathsf{Hamiltonian\ Cycle}$ are in
$\mathsf{FPT}$. In particular, we focus on parameters that describe how many
vertices and edges have to be deleted to become a member of a certain graph
class. We show that the problems are $\mathsf{W[1]}$-hard for such restricted
cases as vertex distance to path and vertex distance to clique. We complement
these results by showing that the problems can be solved in $\mathsf{XP}$ time
for vertex distance to outerplanar and vertex distance to block. Furthermore,
we present some $\mathsf{FPT}$ algorithms, e.g., for edge distance to block.
Additionally, we prove para-$\mathsf{NP}$-hardness when considered with the
edge clique cover number.

### Data Structures and Algorithms

### 1. [Clustering in Varying Metrics](http://arxiv.org/pdf/2510.07860v1)

Authors: Deeparnab Chakrabarty, Jonathan Conroy, Ankita Sarkar

We introduce the aggregated clustering problem, where one is given $T$
instances of a center-based clustering task over the same $n$ points, but under
different metrics. The goal is to open $k$ centers to minimize an aggregate of
the clustering costs -- e.g., the average or maximum -- where the cost is
measured via $k$-center/median/means objectives. More generally, we minimize a
norm $\Psi$ over the $T$ cost values.
  We show that for $T \geq 3$, the problem is inapproximable to any finite
factor in polynomial time. For $T = 2$, we give constant-factor approximations.
We also show W[2]-hardness when parameterized by $k$, but obtain
$f(k,T)\mathrm{poly}(n)$-time 3-approximations when parameterized by both $k$
and $T$.
  When the metrics have structure, we obtain efficient parameterized
approximation schemes (EPAS). If all $T$ metrics have bounded
$\varepsilon$-scatter dimension, we achieve a $(1+\varepsilon)$-approximation
in $f(k,T,\varepsilon)\mathrm{poly}(n)$ time. If the metrics are induced by
edge weights on a common graph $G$ of bounded treewidth $\mathsf{tw}$, and
$\Psi$ is the sum function, we get an EPAS in
$f(T,\varepsilon,\mathsf{tw})\mathrm{poly}(n,k)$ time. Conversely, unless
(randomized) ETH is false, any finite factor approximation is impossible if
parametrized by only $T$, even when the treewidth is $\mathsf{tw} =
\Omega(\mathrm{poly}\log n)$.

### 2. [Dynamic Connectivity with Expected Polylogarithmic Worst-Case Update Time](http://arxiv.org/pdf/2510.08297v1)

Authors: Simon Meierhans, Maximilian Probst Gutenberg

Whether a graph $G=(V,E)$ is connected is arguably its most fundamental
property. Naturally, connectivity was the first characteristic studied for
dynamic graphs, i.e. graphs that undergo edge insertions and deletions. While
connectivity algorithms with polylogarithmic amortized update time have been
known since the 90s, achieving worst-case guarantees has proven more elusive.
  Two recent breakthroughs have made important progress on this question: (1)
Kapron, King and Mountjoy [SODA'13; Best Paper] gave a Monte-Carlo algorithm
with polylogarithmic worst-case update time, and (2) Nanongkai, Saranurak and
Wulff-Nilsen [STOC'17, FOCS'17] obtained a Las-Vegas data structure, however,
with subpolynomial worst-case update time. Their algorithm was subsequently
de-randomized [FOCS'20].
  In this article, we present a new dynamic connectivity algorithm based on the
popular core graph framework that maintains a hierarchy interleaving vertex and
edge sparsification. Previous dynamic implementations of the core graph
framework required subpolynomial update time. In contrast, we show how to
implement it for dynamic connectivity with polylogarithmic expected worst-case
update time.
  We further show that the algorithm can be de-randomized efficiently: a
deterministic static algorithm for computing a connectivity edge-sparsifier of
low congestion in time $T(m) \cdot m$ on an $m$-edge graph yields a
deterministic dynamic connectivity algorithm with $\tilde{O}(T(m))$ worst-case
update time. Via current state-of-the-art algorithms [STOC'24], we obtain $T(m)
= m^{o(1)}$ and recover deterministic subpolynomial worst-case update time.

### 3. [Integer Factoring with Unoperations](http://arxiv.org/pdf/2510.08027v1)

Authors: Paul Kohl

This work introduces the notion of unoperation $\mathfrak{Un}(\hat{O})$ of
some operation $\hat{O}$. Given a valid output of $\hat{O}$, the corresponding
unoperation produces a set of all valid inputs to $\hat{O}$ that produce the
given output. Further, the working principle of unoperations is illustrated
using the example of addition. A device providing that functionality is
constructed utilising a quantum circuit performing the unoperation of addition
- referred to as unaddition. To highlight the potential of the approach the
unaddition quantum circuit is employed to construct a device for factoring
integer numbers $N$, which is then called unmultiplier. This approach requires
only a number of qubits $\in \mathcal{O}((\log{N})^2)$, rivalling the best
known factoring algorithms to date.

### 4. [Timeline Problems in Temporal Graphs: Vertex Cover vs. Dominating Set](http://arxiv.org/pdf/2510.08124v1)

Authors: Anton Herrmann, Christian Komusiewicz, Nils Morawietz, Frank Sommer

A temporal graph is a finite sequence of graphs, called snapshots, over the
same vertex set. Many temporal graph problems turn out to be much more
difficult than their static counterparts. One such problem is \textsc{Timeline
Vertex Cover} (also known as \textsc{MinTimeline$_\infty$}), a temporal
analogue to the classical \textsc{Vertex Cover} problem. In this problem, one
is given a temporal graph $\mathcal{G}$ and two integers $k$ and $\ell$, and
the goal is to cover each edge of each snapshot by selecting for each vertex at
most $k$ activity intervals of length at most $\ell$ each. Here, an edge $uv$
in the $i$th snapshot is covered, if an activity interval of $u$ or $v$ is
active at time $i$. In this work, we continue the algorithmic study of
\textsc{Timeline Vertex Cover} and introduce the \textsc{Timeline Dominating
Set} problem where we want to dominate all vertices in each snapshot by the
selected activity intervals.
  We analyze both problems from a classical and parameterized point of view and
also consider partial problem versions, where the goal is to cover (dominate)
at least $t$ edges (vertices) of the snapshots. With respect to the
parameterized complexity, we consider the temporal graph parameters
vertex-interval-membership-width $(vimw)$ and interval-membership-width
$(imw)$. We show that all considered problems admit FPT-algorithms when
parameterized by $vimw + k+\ell$. This provides a smaller parameter combination
than the ones used for previously known FPT-algorithms for \textsc{Timeline
Vertex Cover}. Surprisingly, for $imw+ k+\ell$, \textsc{Timeline Dominating
Set} turns out to be easier than \textsc{Timeline Vertex Cover}, by also
admitting an FPT-algorithm, whereas the vertex cover version is NP-hard even if
$imw+\, k+\ell$ is constant. We also consider parameterization by combinations
of $n$, the vertex set size, with $k$ or $\ell$ and parameterization by $t$.

### 5. [k-SUM Hardness Implies Treewidth-SETH](http://arxiv.org/pdf/2510.08185v1)

Authors: Michael Lampis

We show that if k-SUM is hard, in the sense that the standard algorithm is
essentially optimal, then a variant of the SETH called the Primal Treewidth
SETH is true. Formally: if there is an $\varepsilon>0$ and an algorithm which
solves SAT in time $(2-\varepsilon)^{tw}|\phi|^{O(1)}$, where $tw$ is the width
of a given tree decomposition of the primal graph of the input, then there
exists a randomized algorithm which solves k-SUM in time
$n^{(1-\delta)\frac{k}{2}}$ for some $\delta>0$ and all sufficiently large $k$.
We also establish an analogous result for the k-XOR problem, where integer
addition is replaced by component-wise addition modulo $2$.
  As an application of our reduction we are able to revisit tight lower bounds
on the complexity of several fundamental problems parameterized by treewidth
(Independent Set, Max Cut, $k$-Coloring). Our results imply that these bounds,
which were initially shown under the SETH, also hold if one assumes the k-SUM
or k-XOR Hypotheses, arguably increasing our confidence in their validity.

### 6. [Energy-Efficient Maximal Independent Sets in Radio Networks](http://arxiv.org/pdf/2510.08244v1)

Authors: Dominick Banasik, Varsha Dani, Fabien Dufoulon, Aayush Gupta, Thomas P. Hayes, Gopal Pandurangan

The maximal independent set (MIS) is one of the most fundamental problems in
distributed computing, and it has been studied intensively for over four
decades. This paper focuses on the MIS problem in the Radio Network model, a
standard model widely used to model wireless networks, particularly ad hoc
wireless and sensor networks. Energy is a premium resource in these networks,
which are typically battery-powered. Hence, designing distributed algorithms
that use as little energy as possible is crucial. We use the well-established
energy model where a node can be sleeping or awake in a round, and only the
awake rounds (when it can send or listen) determine the energy complexity of
the algorithm, which we want to minimize.
  We present new, more energy-efficient MIS algorithms in radio networks with
arbitrary and unknown graph topology. We present algorithms for two popular
variants of the radio model -- with collision detection (CD) and without
collision detection (no-CD). Specifically, we obtain the following results:
  1. CD model: We present a randomized distributed MIS algorithm with energy
complexity $O(\log n)$, round complexity $O(\log^2 n)$, and failure probability
$1 / poly(n)$, where $n$ is the network size. We show that our energy
complexity is optimal by showing a matching $\Omega(\log n)$ lower bound.
  2. no-CD model: In the more challenging no-CD model, we present a randomized
distributed MIS algorithm with energy complexity $O(\log^2n \log \log n)$,
round complexity $O(\log^3 n \log \Delta)$, and failure probability $1 /
poly(n)$. The energy complexity of our algorithm is significantly lower than
the round (and energy) complexity of $O(\log^3 n)$ of the best known
distributed MIS algorithm of Davies [PODC 2023] for arbitrary graph topology.

### 7. [Adaptive Sparsification for Linear Programming](http://arxiv.org/pdf/2510.08348v1)

Authors: Étienne Objois, Adrian Vladu

We introduce a generic framework for solving linear programs (LPs) with many
constraints $(n \gg d)$ via adaptive sparsification. Our approach provides a
principled generalization of the techniques of [Assadi '23] from matching
problems to general LPs and robustifies [Clarkson's '95] celebrated algorithm
for the exact setting. The framework reduces LP solving to a sequence of calls
to a ``low-violation oracle'' on small, adaptively sampled subproblems, which
we analyze through the lens of the multiplicative weight update method.
  Our main results demonstrate the versatility of this paradigm. First, we
present a quantum version of Clarkson's algorithm that finds an exact solution
to an LP using $\tilde{O}(\sqrt{n} d^3)$ row-queries to the constraint matrix.
This is achieved by accelerating the classical bottleneck (the search for
violated constraints) with a generalization of Grover search, decoupling the
quantum component from the classical solver. Second, our framework yields new
state-of-the-art algorithms for mixed packing and covering problems when the
packing constraints are ``simple''. By retaining all packing constraints while
sampling only from the covering constraints, we achieve a significant width
reduction, leading to faster solvers in both the classical and quantum query
models. Our work provides a modular and powerful approach for accelerating LP
solvers.

### 8. [A convergent hierarchy of spectral gap certificates for qubit Hamiltonians](http://arxiv.org/pdf/2510.08427v1)

Authors: Sujit Rao

We give a convergent hierarchy of SDP certificates for bounding the spectral
gap of local qubit Hamiltonians from below. Our approach is based on the NPA
hierarchy applied to a polynomially-sized system of constraints defining the
universal enveloping algebra of the Lie algebra $\mathfrak{su}(2^{n})$, as well
as additional constraints which put restrictions on the corresponding
representations of the algebra. We also use as input an upper bound on the
ground state energy, either using a hierarchy introduced by Fawzi, Fawzi, and
Scalet, or an analog for qubit Hamiltonians of the Lasserre hierarchy of upper
bounds introduced by Klep, Magron, Mass\'{e}, and Vol\v{c}i\v{c}. The
convergence of the certificates does not require that the Hamiltonian be
frustration-free.
  We prove that the resulting certificates have polynomial size at fixed degree
and converge asymptotically (in fact, at level $n$), by showing that all
allowed representations of the algebra correspond to the second exterior power
$\wedge^2(\mathbb{C}^{2^n})$, which encodes the sum of the two smallest
eigenvalues of the original Hamiltonian. We also give an example showing that
for a commuting 1-local Hamiltonian, the hierarchy certifies a nontrivial lower
bound on the spectral gap.

### 9. [Quartic quantum speedups for community detection](http://arxiv.org/pdf/2510.08494v1)

Authors: Alexander Schmidhuber, Alexander Zlokapa

Community detection is a foundational problem in data science. Its natural
extension to hypergraphs captures higher-order correlations beyond pairwise
interactions. In this work, we develop a quantum algorithm for hypergraph
community detection that achieves a quartic quantum speedup over the best known
classical algorithm, along with superpolynomial savings in space. Our algorithm
is based on the Kikuchi method, which we extend beyond previously considered
problems such as Tensor PCA and $p$XORSAT to a broad family of generalized
stochastic block models. To demonstrate (near) optimality of this method, we
prove matching lower bounds (up to logarithmic factors) in the low-degree
framework, showing that the algorithm saturates a smooth
statistical-computational tradeoff. The quantum speedup arises from a quantized
version of the Kikuchi method and is based on the efficient preparation of a
guiding state correlated with the underlying community structure. Our work
suggests that prior quantum speedups using the Kikuchi method are sufficiently
robust to encompass a broader set of problems than previously believed; we
conjecture that a quantity known as marginal order characterizes the existence
of these quantum speedups.

### 10. [Randomized and quantum approximate matrix multiplication](http://arxiv.org/pdf/2510.08509v1)

Authors: Simon Apers, Arjan Cornelissen, Samson Wang

The complexity of matrix multiplication is a central topic in computer
science. While the focus has traditionally been on exact algorithms, a long
line of literature also considers randomized algorithms, which return an
approximate solution in faster time. In this work, we adopt a unifying
perspective that frames these randomized algorithms in terms of mean
estimation. Using it, we first give refined analyses of classical algorithms
based on random walks by Cohen-Lewis (`99), and based on sketching by Sarl\'os
(`06) and Drineas-Kannan-Mahoney (`06). We then propose an improvement on
Cohen-Lewis that yields a single classical algorithm that is faster than all
the other approaches, if we assume no use of (exact) fast matrix multiplication
as a subroutine. Second, we demonstrate a quantum speedup on top of these
algorithms by using the recent quantum multivariate mean estimation algorithm
by Cornelissen-Hamoudi-Jerbi (`22).

### Emerging Technologies

### 1. [A Distributed Emulation Environment for In-Memory Computing Systems](http://arxiv.org/pdf/2510.08257v1)

Authors: Eleni Bougioukou, Anastasios Petropoulos, Nikolaos Toulgaridis, Theodoros Chatzimichail, Theodore Antonakopoulos

In-memory computing technology is used extensively in artificial intelligence
devices due to lower power consumption and fast calculation of matrix-based
functions. The development of such a device and its integration in a system
takes a significant amount of time and requires the use of a real-time
emulation environment, where various system aspects are analyzed, microcode is
tested, and applications are deployed, even before the real chip is available.
In this work, we present the architecture, the software development tools, and
experimental results of a distributed and expandable emulation system for rapid
prototyping of integrated circuits based on in-memory computing technologies.
Presented experimental results demonstrate the usefulness of the proposed
emulator.

### 2. [Sentiment Matters: An Analysis of 200 Human-SAV Interactions](http://arxiv.org/pdf/2510.08202v1)

Authors: Lirui Guo, Michael G. Burke, Wynita M. Griggs

Shared Autonomous Vehicles (SAVs) are likely to become an important part of
the transportation system, making effective human-SAV interactions an important
area of research. This paper introduces a dataset of 200 human-SAV interactions
to further this area of study. We present an open-source human-SAV
conversational dataset, comprising both textual data (e.g., 2,136 human-SAV
exchanges) and empirical data (e.g., post-interaction survey results on a range
of psychological factors). The dataset's utility is demonstrated through two
benchmark case studies: First, using random forest modeling and chord diagrams,
we identify key predictors of SAV acceptance and perceived service quality,
highlighting the critical influence of response sentiment polarity (i.e.,
perceived positivity). Second, we benchmark the performance of an LLM-based
sentiment analysis tool against the traditional lexicon-based TextBlob method.
Results indicate that even simple zero-shot LLM prompts more closely align with
user-reported sentiment, though limitations remain. This study provides novel
insights for designing conversational SAV interfaces and establishes a
foundation for further exploration into advanced sentiment modeling, adaptive
user interactions, and multimodal conversational systems.

### Formal Languages and Automata Theory

### 1. [On the Complexity of Language Membership for Probabilistic Words](http://arxiv.org/pdf/2510.08127v1)

Authors: Antoine Amarilli, Mikaël Monet, Paul Raphaël, Sylvain Salvati

We study the membership problem to context-free languages L (CFLs) on
probabilistic words, that specify for each position a probability distribution
on the letters (assuming independence across positions). Our task is to
compute, given a probabilistic word, what is the probability that a word drawn
according to the distribution belongs to L. This problem generalizes the
problem of counting how many words of length n belong to L, or of counting how
many completions of a partial word belong to L.
  We show that this problem is in polynomial time for unambiguous context-free
languages (uCFLs), but can be #P-hard already for unions of two linear uCFLs.
More generally, we show that the problem is in polynomial time for so-called
poly-slicewise-unambiguous languages, where given a length n we can tractably
compute an uCFL for the words of length n in the language. This class includes
some inherently ambiguous languages, and implies the tractability of bounded
CFLs and of languages recognized by unambiguous polynomial-time counter
automata; but we show that the problem can be #P-hard for nondeterministic
counter automata, even for Parikh automata with a single counter. We then
introduce classes of circuits from knowledge compilation which we use for
tractable counting, and show that this covers the tractability of
poly-slicewise-unambiguous languages and of some CFLs that are not
poly-slicewise-unambiguous. Extending these circuits with negation further
allows us to show tractability for the language of primitive words, and for the
language of concatenations of two palindromes. We finally show the conditional
undecidability of the meta-problem that asks, given a CFG, whether the
probabilistic membership problem for that CFG is tractable or #P-hard.

### 2. [Languages of Words of Low Automatic Complexity Are Hard to Compute](http://arxiv.org/pdf/2510.07696v1)

Authors: Joey Chen, Bjørn Kjos-Hanssen, Ivan Koswara, Linus Richter, Frank Stephan

The automatic complexity of a finite word (string) is an analogue for finite
automata of Sipser's distinguishing complexity (1983) and was introduced by
Shallit and Wang (2001). For a finite alphabet $\Sigma$ of at least two
elements, we consider the non-deterministic automatic complexity given by
exactly - yet not necessarily uniquely - accepting automata: a word $x \in
\Sigma^*$ has exact non-deterministic automatic complexity $k \in \mathbb{N}$
if there exists a non-deterministic automaton of $k$ states which accepts $x$
while rejecting every other word of the same length as $x$, and no automaton of
fewer states has this property. Importantly, and in contrast to the classical
notion, the witnessing automaton may have multiple paths of computation
accepting $x$. We denote this measure of complexity by $A_{Ne}$, and study a
class of languages of low $A_{Ne}$-complexity defined as $L_q = \{ \, x \in
\Sigma^* : A_{Ne}(x) < q|x| \, \}$, which is parameterised by rationals $q \in
(0,1/2)$ (generalising a class of sets first studied by Kjos-Hanssen). We show
that for every $q \in (0,1/2)$, this class is neither context-free nor
recognisable by certain Boolean circuits. In the process, we answer an open
question of Kjos-Hanssen quantifying the complexity of $L_{1/3}$ in terms of
Boolean circuits, and also prove the Shannon effect for $A_{Ne}$.

### 3. [Self-replication and Computational Universality](http://arxiv.org/pdf/2510.08342v1)

Authors: Jordan Cotler, Clément Hongler, Barbora Hudcová

Self-replication is central to all life, and yet how it dynamically emerges
in physical, non-equilibrium systems remains poorly understood. Von Neumann's
pioneering work in the 1940s and subsequent developments suggest a natural
hypothesis: that any physical system capable of Turing-universal computation
can support self-replicating objects. In this work, we challenge this
hypothesis by clarifying what computational universality means for physical
systems and constructing a cellular automaton that is Turing-universal but
cannot sustain non-trivial self-replication. By analogy with biology, such
dynamics manifest transcription and translation but cannot instantiate
replication. More broadly, our work emphasizes that the computational
complexity of translating between physical dynamics and symbolic computation is
inseparable from any claim of universality (exemplified by our analysis of Rule
110) and builds mathematical foundations for identifying self-replicating
behavior. Our approach enables the formulation of necessary dynamical and
computational conditions for a physical system to constitute a living organism.

### Graphics

### 1. [Differentiable Variable Fonts](http://arxiv.org/pdf/2510.07638v1)

Authors: Kinjal Parikh, Danny M. Kaufman, David I. W. Levin, Alec Jacobson

Editing and animating text appearance for graphic designs, commercials, etc.
remain highly skilled tasks requiring detailed, hands on efforts from artists.
Automating these manual workflows requires balancing the competing goals of
maintaining legibility and aesthetics of text, while enabling creative
expression. Variable fonts, recent parametric extensions to traditional fonts,
offer the promise of new ways to ease and automate typographic design and
animation. Variable fonts provide custom constructed parameters along which
fonts can be smoothly varied. These parameterizations could then potentially
serve as high value continuous design spaces, opening the door to automated
design optimization tools. However, currently variable fonts are underutilized
in creative applications, because artists so far still need to manually tune
font parameters. Our work opens the door to intuitive and automated font design
and animation workflows with differentiable variable fonts. To do so we distill
the current variable font specification to a compact mathematical formulation
that differentiably connects the highly non linear, non invertible mapping of
variable font parameters to the underlying vector graphics representing the
text. This enables us to construct a differentiable framework, with respect to
variable font parameters, allowing us to perform gradient based optimization of
energies defined on vector graphics control points, and on target rasterized
images. We demonstrate the utility of this framework with four applications:
direct shape manipulation, overlap aware modeling, physics based text
animation, and automated font design optimization. Our work now enables
leveraging the carefully designed affordances of variable fonts with
differentiability to use modern design optimization technologies, opening new
possibilities for easy and intuitive typographic design workflows.

### 2. [NRRS: Neural Russian Roulette and Splitting](http://arxiv.org/pdf/2510.07868v1)

Authors: Haojie Jin, Jierui Ren, Yisong Chen, Guoping Wang, Sheng Li

We propose a novel framework for Russian Roulette and Splitting (RRS)
tailored to wavefront path tracing, a highly parallel rendering architecture
that processes path states in batched, stage-wise execution for efficient GPU
utilization. Traditional RRS methods, with unpredictable path counts, are
fundamentally incompatible with wavefront's preallocated memory and scheduling
requirements. To resolve this, we introduce a normalized RRS formulation with a
bounded path count, enabling stable and memory-efficient execution.
  Furthermore, we pioneer the use of neural networks to learn RRS factors,
presenting two models: NRRS and AID-NRRS. At a high level, both feature a
carefully designed RRSNet that explicitly incorporates RRS normalization, with
only subtle differences in their implementation. To balance computational cost
and inference accuracy, we introduce Mix-Depth, a path-depth-aware mechanism
that adaptively regulates neural evaluation, further improving efficiency.
  Extensive experiments demonstrate that our method outperforms traditional
heuristics and recent RRS techniques in both rendering quality and performance
across a variety of complex scenes.

### 3. [Variable-Rate Texture Compression: Real-Time Rendering with JPEG](http://arxiv.org/pdf/2510.08166v1)

Authors: Elias Kristmann, Markus Schütz, Michael Wimmer

Although variable-rate compressed image formats such as JPEG are widely used
to efficiently encode images, they have not found their way into real-time
rendering due to special requirements such as random access to individual
texels. In this paper, we investigate the feasibility of variable-rate texture
compression on modern GPUs using the JPEG format, and how it compares to the
GPU-friendly fixed-rate compression approaches BC1 and ASTC. Using a deferred
rendering pipeline, we are able to identify the subset of blocks that are
needed for a given frame, decode these, and colorize the framebuffer's pixels.
Despite the additional $\sim$0.17 bit per pixel that we require for our
approach, JPEG maintains significantly better quality and compression rates
compared to BC1, and depending on the type of image, outperforms or competes
with ASTC. The JPEG rendering pipeline increases rendering duration by less
than 0.3 ms on an RTX 4090, demonstrating that sophisticated variable-rate
compression schemes are feasible on modern GPUs, even in VR. Source code and
data sets are available at: https://github.com/elias1518693/jpeg_textures

### 4. [SViM3D: Stable Video Material Diffusion for Single Image 3D Generation](http://arxiv.org/pdf/2510.08271v1)

Authors: Andreas Engelhardt, Mark Boss, Vikram Voletti, Chun-Han Yao, Hendrik P. A. Lensch, Varun Jampani

We present Stable Video Materials 3D (SViM3D), a framework to predict
multi-view consistent physically based rendering (PBR) materials, given a
single image. Recently, video diffusion models have been successfully used to
reconstruct 3D objects from a single image efficiently. However, reflectance is
still represented by simple material models or needs to be estimated in
additional steps to enable relighting and controlled appearance edits. We
extend a latent video diffusion model to output spatially varying PBR
parameters and surface normals jointly with each generated view based on
explicit camera control. This unique setup allows for relighting and generating
a 3D asset using our model as neural prior. We introduce various mechanisms to
this pipeline that improve quality in this ill-posed setting. We show
state-of-the-art relighting and novel view synthesis performance on multiple
object-centric datasets. Our method generalizes to diverse inputs, enabling the
generation of relightable 3D assets useful in AR/VR, movies, games and other
visual media.

### 5. [Spectral Prefiltering of Neural Fields](http://arxiv.org/pdf/2510.08394v1)

Authors: Mustafa B. Yaldiz, Ishit Mehta, Nithin Raghavan, Andreas Meuleman, Tzu-Mao Li, Ravi Ramamoorthi

Neural fields excel at representing continuous visual signals but typically
operate at a single, fixed resolution. We present a simple yet powerful method
to optimize neural fields that can be prefiltered in a single forward pass. Key
innovations and features include: (1) We perform convolutional filtering in the
input domain by analytically scaling Fourier feature embeddings with the
filter's frequency response. (2) This closed-form modulation generalizes beyond
Gaussian filtering and supports other parametric filters (Box and Lanczos) that
are unseen at training time. (3) We train the neural field using single-sample
Monte Carlo estimates of the filtered signal. Our method is fast during both
training and inference, and imposes no additional constraints on the network
architecture. We show quantitative and qualitative improvements over existing
methods for neural-field filtering.

### 6. [Splat the Net: Radiance Fields with Splattable Neural Primitives](http://arxiv.org/pdf/2510.08491v1)

Authors: Xilong Zhou, Bao-Huy Nguyen, Loïc Magne, Vladislav Golyanik, Thomas Leimkühler, Christian Theobalt

Radiance fields have emerged as a predominant representation for modeling 3D
scene appearance. Neural formulations such as Neural Radiance Fields provide
high expressivity but require costly ray marching for rendering, whereas
primitive-based methods such as 3D Gaussian Splatting offer real-time
efficiency through splatting, yet at the expense of representational power.
Inspired by advances in both these directions, we introduce splattable neural
primitives, a new volumetric representation that reconciles the expressivity of
neural models with the efficiency of primitive-based splatting. Each primitive
encodes a bounded neural density field parameterized by a shallow neural
network. Our formulation admits an exact analytical solution for line
integrals, enabling efficient computation of perspectively accurate splatting
kernels. As a result, our representation supports integration along view rays
without the need for costly ray marching. The primitives flexibly adapt to
scene geometry and, being larger than prior analytic primitives, reduce the
number required per scene. On novel-view synthesis benchmarks, our approach
matches the quality and speed of 3D Gaussian Splatting while using $10\times$
fewer primitives and $6\times$ fewer parameters. These advantages arise
directly from the representation itself, without reliance on complex control or
adaptation frameworks. The project page is
https://vcai.mpi-inf.mpg.de/projects/SplatNet/.

### 7. [X2Video: Adapting Diffusion Models for Multimodal Controllable Neural Video Rendering](http://arxiv.org/pdf/2510.08530v1)

Authors: Zhitong Huang, Mohan Zhang, Renhan Wang, Rui Tang, Hao Zhu, Jing Liao

We present X2Video, the first diffusion model for rendering photorealistic
videos guided by intrinsic channels including albedo, normal, roughness,
metallicity, and irradiance, while supporting intuitive multi-modal controls
with reference images and text prompts for both global and local regions. The
intrinsic guidance allows accurate manipulation of color, material, geometry,
and lighting, while reference images and text prompts provide intuitive
adjustments in the absence of intrinsic information. To enable these
functionalities, we extend the intrinsic-guided image generation model XRGB to
video generation by employing a novel and efficient Hybrid Self-Attention,
which ensures temporal consistency across video frames and also enhances
fidelity to reference images. We further develop a Masked Cross-Attention to
disentangle global and local text prompts, applying them effectively onto
respective local and global regions. For generating long videos, our novel
Recursive Sampling method incorporates progressive frame sampling, combining
keyframe prediction and frame interpolation to maintain long-range temporal
consistency while preventing error accumulation. To support the training of
X2Video, we assembled a video dataset named InteriorVideo, featuring 1,154
rooms from 295 interior scenes, complete with reliable ground-truth intrinsic
channel sequences and smooth camera trajectories. Both qualitative and
quantitative evaluations demonstrate that X2Video can produce long, temporally
consistent, and photorealistic videos guided by intrinsic conditions.
Additionally, X2Video effectively accommodates multi-modal controls with
reference images, global and local text prompts, and simultaneously supports
editing on color, material, geometry, and lighting through parametric tuning.
Project page: https://luckyhzt.github.io/x2video

### Computer Science and Game Theory

### 1. [Extending Games beyond the Finite Horizon](http://arxiv.org/pdf/2510.08453v1)

Authors: Kiri Sakahara, Takashi Sato

This paper argues that the finite horizon paradox, where game theory
contradicts intuition, stems from the limitations of standard number systems in
modelling the cognitive perception of infinity. To address this issue, we
propose a new framework based on Alternative Set Theory (AST). This framework
represents different cognitive perspectives on a long history of events using
distinct topologies. These topologies define an indiscernibility equivalence
that formally treats huge, indistinguishable quantities as equivalent. This
offers criterion-dependent resolutions to long-standing paradoxes, such as
Selten's chain store paradox and Rosenthal's centipede game. Our framework
reveals new intuitive subgame perfect equilibria, the characteristics of which
depend on the chosen temporal perspective and payoff evaluation. Ultimately, by
grounding its mathematical foundation in different modes of human cognition,
our work expands the explanatory power of game theory for long-horizon
scenarios.

### Human-Computer Interaction

### 1. [Human-in-the-Loop Optimization with Model-Informed Priors](http://arxiv.org/pdf/2510.07754v1)

Authors: Yi-Chi Liao, João Belo, Hee-Seung Moon, Jürgen Steimle, Anna Maria Feit

Human-in-the-loop optimization identifies optimal interface designs by
iteratively observing user performance. However, it often requires numerous
iterations due to the lack of prior information. While recent approaches have
accelerated this process by leveraging previous optimization data, collecting
user data remains costly and often impractical. We present a conceptual
framework, Human-in-the-Loop Optimization with Model-Informed Priors (HOMI),
which augments human-in-the-loop optimization with a training phase where the
optimizer learns adaptation strategies from diverse, synthetic user data
generated with predictive models before deployment. To realize HOMI, we
introduce Neural Acquisition Function+ (NAF+), a Bayesian optimization method
featuring a neural acquisition function trained with reinforcement learning.
NAF+ learns optimization strategies from large-scale synthetic data, improving
efficiency in real-time optimization with users. We evaluate HOMI and NAF+ with
mid-air keyboard optimization, a representative VR input task. Our work
presents a new approach for more efficient interface adaptation by bridging in
situ and in silico optimization processes.

### 2. [Pre/Absence: Prompting Cultural Awareness and Understanding for Lost Architectural Heritage in Virtual Reality](http://arxiv.org/pdf/2510.07967v1)

Authors: Yaning Li, Ke Zhao, Shucheng Zheng, Xingyu Chen, Chenyi Chen, Wenxi Dai, Weile Jiang, Qi Dong, Yiqing Zhao, Meng Li, Lin-Ping Yuan

Lost architectural heritage presents interpretive challenges due to vanished
structures and fragmented historical records. Using Hanyuan Hall of the Tang
dynasty's Daming Palace as a case study, we conducted a formative investigation
with archaeologists, heritage administrators, and visitors to identify key
issues in current interpretation practices. We found that these practices often
compress complex cultural layers into factual summaries and rely on linear
narratives that overlook the continuing reinterpretations following a site's
disappearance. In response, we designed Pre/Absence, a virtual reality
experience grounded in the presence-absence dialectic to interweave tangible
and vanished aspects of heritage within a spatiotemporal narrative. A
mixed-method study with 28 participants compared Pre/Absence to a paper-based
experience. Both improved users' factual understanding, but the VR experience
more strongly enhanced cultural awareness, evoked emotional engagement with
loss, and encouraged critical reflection on the evolving social and political
meanings of heritage. The findings suggest that VR can move beyond static
reconstruction to engage users as co-constructors of cultural meaning,
providing a nuanced framework for critical heritage narrative design in
human-computer interaction.

### 3. [Quantifying Locomotion Differences Between Virtual Reality Users With and Without Motor Impairments](http://arxiv.org/pdf/2510.07987v1)

Authors: Rachel L. Franz, Jacob O. Wobbrock

Today's virtual reality (VR) systems and environments assume that users have
typical abilities, which can make VR inaccessible to people with physical
impairments. However, there is not yet an understanding of how inaccessible
locomotion techniques are, and which interactions make them inaccessible. To
this end, we conducted a study in which people with and without upper-body
impairments navigated a virtual environment with six locomotion techniques to
quantify performance differences among groups. We found that groups performed
similarly with Sliding Looking on all performance measures, suggesting that
this might be a good default locomotion technique for VR apps. To understand
the nature of performance differences with the other techniques, we collected
low-level interaction data from the controllers and headset and analyzed
interaction differences with a set of movement-, button-, and target-related
metrics. We found that movement-related metrics from headset data reveal
differences among groups with all techniques, suggesting these are good metrics
for identifying whether a user has an upper-body impairment. We also identify
movement-, button, and target-related metrics that can explain performance
differences between groups for particular locomotion techniques.

### 4. [Practicing a Second Language Without Fear: Mixed Reality Agents for Interactive Group Conversation](http://arxiv.org/pdf/2510.08227v1)

Authors: Mariana Fernandez-Espinosa, Kai Zhang, Jad Bendarkawi, Ashley Ponce, Sean Chidozie Mata, Aminah Aliu, Lei Zhang, Francisco Fernandez Medina, Elena Mangione-Lora, Andres Monroy-Hernandez, Diego Gomez-Zara

Developing speaking proficiency in a second language can be cognitively
demanding and emotionally taxing, often triggering fear of making mistakes or
being excluded from larger groups. While current learning tools show promise
for speaking practice, most focus on dyadic, scripted scenarios, limiting
opportunities for dynamic group interactions. To address this gap, we present
ConversAR, a Mixed Reality system that leverages Generative AI and XR to
support situated and personalized group conversations. It integrates embodied
AI agents, scene recognition, and generative 3D props anchored to real-world
surroundings. Based on a formative study with experts in language acquisition,
we developed and tested this system with a user study with 21 second-language
learners. Results indicate that the system enhanced learner engagement,
increased willingness to communicate, and offered a safe space for speaking. We
discuss the implications for integrating Generative AI and XR into the design
of future language learning applications.

### 5. [Simulating Teams with LLM Agents: Interactive 2D Environments for Studying Human-AI Dynamics](http://arxiv.org/pdf/2510.08242v1)

Authors: Mohammed Almutairi, Charles Chiang, Haoze Guo, Matthew Belcher, Nandini Banerjee, Maria Milkowski, Svitlana Volkova, Daniel Nguyen, Tim Weninger, Michael Yankoski, Trenton W. Ford, Diego Gomez-Zara

Enabling users to create their own simulations offers a powerful way to study
team dynamics and performance. We introduce VirTLab, a system that allows
researchers and practitioners to design interactive, customizable simulations
of team dynamics with LLM-based agents situated in 2D spatial environments.
Unlike prior frameworks that restrict scenarios to predefined or static tasks,
our approach enables users to build scenarios, assign roles, and observe how
agents coordinate, move, and adapt over time. By bridging team cognition
behaviors with scalable agent-based modeling, our system provides a testbed for
investigating how environments influence coordination, collaboration, and
emergent team behaviors. We demonstrate its utility by aligning simulated
outcomes with empirical evaluations and a user study, underscoring the
importance of customizable environments for advancing research on multi-agent
simulations. This work contributes to making simulations accessible to both
technical and non-technical users, supporting the design, execution, and
analysis of complex multi-agent experiments.

### 6. [LacAIDes: Generative AI-Supported Creative Interactive Circuits Crafting to Enliven Traditional Lacquerware](http://arxiv.org/pdf/2510.08326v1)

Authors: Yaning Li, Yutong Chen, Yihan Hou, Chenyi Chen, Yihan Han, Jingxuan Han, Wenxi Dai, Youyou Li, Xinke Tang, Meng Li, Qi Dong, Hongwei Li

Lacquerware, a representative craft of Chinese intangible cultural heritage,
is renowned for its layered aesthetics and durability but faces declining
engagement. While prior human-computer interaction research has explored
embedding interactive circuits to transform lacquerware into responsive
artifacts, most studies have focused on fabrication techniques rather than
supporting makers in creatively designing such interactions at a low threshold.
To address this gap, we present LacAIDes, a Generative AI powered
creativity-support tool built on a multi-agent workflow aligned with the double
diamond model of design thinking. LacAIDes enables exploration and creation of
culturally grounded interactive circuits without requiring prior technical
expertise. We evaluated LacAIDes in a longitudinal workshop with 34
participants using a mixed-method approach. Results show that LacAIDes
demonstrated high usability, enhanced creative engagement in craft making, and
encouraged critical reflection on the role of Generative AI in digital craft
practices. This work contributes to human-computer interaction by introducing a
novel creativity-support tool and providing empirical insights into
revitalizing traditional craft making through Generative AI.

### 7. [Motion Exploration of Articulated Product Concepts in Interactive Sketching Environment](http://arxiv.org/pdf/2510.08328v1)

Authors: Kalyan Ramana Gattoz, Prasad S. Onkar

In the early stages of engineering design, it is essential to know how a
product behaves, especially how it moves. As designers must keep adjusting the
motion until it meets the intended requirements, this process is often
repetitive and time-consuming. Although the physics behind these motions is
usually based on simple equations, manually working through them can be tedious
and inefficient. To ease this burden, some tasks are now handled by computers.
One common method involves converting hand-drawn sketches into models using CAD
or CAE software. However, this approach can be time- and resource-intensive.
Additionally, product sketches are usually best understood only by the
designers who created them. Others may struggle to interpret them correctly,
relying heavily on intuition and prior experience. Since sketches are static,
they fail to show how a product moves, limiting their usefulness. This paper
presents a new approach that addresses these issues by digitising the natural
act of sketching. It allows designers to create, simulate, and test the motion
of mechanical concepts in a more interactive way. An application was developed
to evaluate this method, focusing on user satisfaction and mental workload
during a design task. The results showed a 77% reduction in cognitive effort
compared to traditional methods, with users reporting high satisfaction. Future
work will focus on expanding this approach from 2D (planar) to full 3D
(spatial) design environments, enabling more complex product concept
development.

### 8. [What Makes a Visualization Complex?](http://arxiv.org/pdf/2510.08332v1)

Authors: Mengdi Chu, Zefeng Qiu, Meng Ling, Shuning Jiang, Robert S. Laramee, Michael Sedlmair, Jian Chen

We investigate the perceived visual complexity (VC) in data visualizations
using objective image-based metrics. We collected VC scores through a
large-scale crowdsourcing experiment involving 349 participants and 1,800
visualization images. We then examined how these scores align with 12
image-based metrics spanning information-theoretic, clutter, color, and our two
object-based metrics. Our results show that both low-level image properties and
the high-level elements affect perceived VC in visualization images; The number
of corners and distinct colors are robust metrics across visualizations.
Second, feature congestion, an information-theoretic metric capturing
statistical patterns in color and texture, is the strongest predictor of
perceived complexity in visualizations rich in the same stimuli; edge density
effectively explains VC in node-link diagrams. Additionally, we observe a
bell-curve effect for text annotations: increasing text-to-ink ratio (TiR)
initially reduces complexity, reaching an optimal point, beyond which further
text increases perceived complexity. Our quantification pipeline is also
interpretable, enabling metric-based explanations, grounded in the
VisComplexity2K dataset, bridging computational metrics with human perceptual
responses. osf.io/5xe8a has the preregistration and osf.io/bdet6 has the
VisComplexity2K dataset, source code, and all Apdx. and figures.

### 9. [The Rise of the Knowledge Sculptor: A New Archetype for Knowledge Work in the Age of Generative AI](http://arxiv.org/pdf/2510.07829v1)

Authors: Cathal Doyle

In the Generative Age, the nature of knowledge work is transforming.
Traditional models that emphasise the organisation and retrieval of
pre-existing information are increasingly inadequate in the face of generative
AI (GenAI) systems capable of autonomous content creation. This paper
introduces the Knowledge Sculptor (KS), a new professional archetype for
Human-GenAI collaboration that transforms raw AI output into trustworthy,
actionable knowledge. Grounded in a socio-technical perspective, the KS is
conceptualised through a framework of competencies, including architecting a
vision, iterative dialogue, information sculpting, and curiosity-driven
synthesis. A practice-based vignette illustrates the KS role in action, and in
a self-referential approach, the paper itself serves as an artefact of the
sculpting process it describes.

### 10. [Enabling Personalized Long-term Interactions in LLM-based Agents through Persistent Memory and User Profiles](http://arxiv.org/pdf/2510.07925v1)

Authors: Rebecca Westhäußer, Wolfgang Minker, Sebatian Zepf

Large language models (LLMs) increasingly serve as the central control unit
of AI agents, yet current approaches remain limited in their ability to deliver
personalized interactions. While Retrieval Augmented Generation enhances LLM
capabilities by improving context-awareness, it lacks mechanisms to combine
contextual information with user-specific data. Although personalization has
been studied in fields such as human-computer interaction or cognitive science,
existing perspectives largely remain conceptual, with limited focus on
technical implementation. To address these gaps, we build on a unified
definition of personalization as a conceptual foundation to derive technical
requirements for adaptive, user-centered LLM-based agents. Combined with
established agentic AI patterns such as multi-agent collaboration or
multi-source retrieval, we present a framework that integrates persistent
memory, dynamic coordination, self-validation, and evolving user profiles to
enable personalized long-term interactions. We evaluate our approach on three
public datasets using metrics such as retrieval accuracy, response correctness,
or BertScore. We complement these results with a five-day pilot user study
providing initial insights into user feedback on perceived personalization. The
study provides early indications that guide future work and highlights the
potential of integrating persistent memory and user profiles to improve the
adaptivity and perceived personalization of LLM-based agents.

### Information Retrieval

### 1. [ISMIE: A Framework to Characterize Information Seeking in Modern Information Environments](http://arxiv.org/pdf/2510.07644v1)

Authors: Shuoqi Sun, Danula Hettiachchi, Damiano Spina

The modern information environment (MIE) is increasingly complex, shaped by a
wide range of techniques designed to satisfy users' information needs.
Information seeking (IS) models are effective mechanisms for characterizing
user-system interactions. However, conceptualizing a model that fully captures
the MIE landscape poses a challenge. We argue: Does such a model exist? To
address this, we propose the Information Seeking in Modern Information
Environments (ISMIE) framework as a fundamental step. ISMIE conceptualizes the
information seeking process (ISP) via three key concepts: Components (e.g.,
Information Seeker), Intervening Variables (e.g., Interactive Variables), and
Activities (e.g., Acquiring). Using ISMIE's concepts and employing a case study
based on a common scenario - misinformation dissemination - we analyze six
existing IS and information retrieval (IR) models to illustrate their
limitations and the necessity of ISMIE. We then show how ISMIE serves as an
actionable framework for both characterization and experimental design. We
characterize three pressing issues and then outline two research blueprints: a
user-centric, industry-driven experimental design for the authenticity and
trust crisis to AI-generated content and a system-oriented, academic-driven
design for tackling dopamine-driven content consumption. Our framework offers a
foundation for developing IS and IR models to advance knowledge on
understanding human interactions and system design in MIEs.

### 2. [Queries Are Not Alone: Clustering Text Embeddings for Video Search](http://arxiv.org/pdf/2510.07720v1)

Authors: Peyang Liu, Xi Wang, Ziqiang Cui, Wei Ye

The rapid proliferation of video content across various platforms has
highlighted the urgent need for advanced video retrieval systems. Traditional
methods, which primarily depend on directly matching textual queries with video
metadata, often fail to bridge the semantic gap between text descriptions and
the multifaceted nature of video content. This paper introduces a novel
framework, the Video-Text Cluster (VTC), which enhances video retrieval by
clustering text queries to capture a broader semantic scope. We propose a
unique clustering mechanism that groups related queries, enabling our system to
consider multiple interpretations and nuances of each query. This clustering is
further refined by our innovative Sweeper module, which identifies and
mitigates noise within these clusters. Additionally, we introduce the
Video-Text Cluster-Attention (VTC-Att) mechanism, which dynamically adjusts
focus within the clusters based on the video content, ensuring that the
retrieval process emphasizes the most relevant textual features. Further
experiments have demonstrated that our proposed model surpasses existing
state-of-the-art models on five public datasets.

### 3. [Generation and annotation of item usage scenarios in e-commerce using large language models](http://arxiv.org/pdf/2510.07885v1)

Authors: Madoka Hagiri, Kazushi Okamoto, Koki Karube, Kei Harada, Atsushi Shibata

Complementary recommendations suggest combinations of useful items that play
important roles in e-commerce. However, complementary relationships are often
subjective and vary among individuals, making them difficult to infer from
historical data. Unlike conventional history-based methods that rely on
statistical co-occurrence, we focus on the underlying usage context that
motivates item combinations. We hypothesized that people select complementary
items by imagining specific usage scenarios and identifying the needs in such
situations. Based on this idea, we explored the use of large language models
(LLMs) to generate item usage scenarios as a starting point for constructing
complementary recommendation systems. First, we evaluated the plausibility of
LLM-generated scenarios through manual annotation. The results demonstrated
that approximately 85% of the generated scenarios were determined to be
plausible, suggesting that LLMs can effectively generate realistic item usage
scenarios.

### 4. [Mobile Gamer Lifetime Value Prediction via Objective Decomposition and Reconstruction](http://arxiv.org/pdf/2510.08281v1)

Authors: Tianwei Li, Yu Zhao, Yunze Li, Sheng Li

For Internet platforms operating real-time bidding (RTB) advertising service,
a comprehensive understanding of user lifetime value (LTV) plays a pivotal role
in optimizing advertisement allocation efficiency and maximizing the return on
investment (ROI) for advertisement sponsors, thereby facilitating growth of
commercialization revenue for the platform. However, the inherent complexity of
user LTV distributions induces significant challenges in accurate LTV
prediction. Existing state-of-the-art works, which primarily focus on directly
learning the LTV distributions through well-designed loss functions, achieve
limited success due to their vulnerability to outliers. In this paper, we
proposed a novel LTV prediction method to address distribution challenges
through an objective decomposition and reconstruction framework. Briefly
speaking, based on the in-app purchase characteristics of mobile gamers, our
model was designed to first predict the number of transactions at specific
prices and then calculate the total payment amount from these intermediate
predictions. Our proposed model was evaluated through experiments on real-world
industrial dataset, and deployed on the TapTap RTB advertising system for
online A/B testing along with the state-of-the-art ZILN model.

### 5. [Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft](http://arxiv.org/pdf/2510.07728v1)

Authors: Peiyang Liu, Ziqiang Cui, Di Liang, Wei Ye

Retrieval-augmented generation (RAG) enhances Large Language Models (LLMs) by
mitigating hallucinations and outdated information issues, yet simultaneously
facilitates unauthorized data appropriation at scale. This paper addresses this
challenge through two key contributions. First, we introduce RPD, a novel
dataset specifically designed for RAG plagiarism detection that encompasses
diverse professional domains and writing styles, overcoming limitations in
existing resources. Second, we develop a dual-layered watermarking system that
embeds protection at both semantic and lexical levels, complemented by an
interrogator-detective framework that employs statistical hypothesis testing on
accumulated evidence. Extensive experimentation demonstrates our approach's
effectiveness across varying query volumes, defense prompts, and retrieval
parameters, while maintaining resilience against adversarial evasion
techniques. This work establishes a foundational framework for intellectual
property protection in retrieval-augmented AI systems.

### 6. [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](http://arxiv.org/pdf/2510.07784v1)

Authors: Ruining He, Lukasz Heldt, Lichan Hong, Raghunandan Keshavan, Shifan Mao, Nikhil Mehta, Zhengyang Su, Alicia Tsai, Yueqi Wang, Shao-Chuan Wang, Xinyang Yi, Lexi Baugher, Baykal Cakici, Ed Chi, Cristos Goodrow, Ningren Han, He Ma, Romer Rosales, Abby Van Soest, Devansh Tandon, Su-Lin Wu, Weilong Yang, Yilin Zheng

Large Language Models (LLMs) pose a new paradigm of modeling and computation
for information tasks. Recommendation systems are a critical application domain
poised to benefit significantly from the sequence modeling capabilities and
world knowledge inherent in these large models. In this paper, we introduce
PLUM, a framework designed to adapt pre-trained LLMs for industry-scale
recommendation tasks. PLUM consists of item tokenization using Semantic IDs,
continued pre-training (CPT) on domain-specific data, and task-specific
fine-tuning for recommendation objectives. For fine-tuning, we focus
particularly on generative retrieval, where the model is directly trained to
generate Semantic IDs of recommended items based on user context. We conduct
comprehensive experiments on large-scale internal video recommendation
datasets. Our results demonstrate that PLUM achieves substantial improvements
for retrieval compared to a heavily-optimized production model built with large
embedding tables. We also present a scaling study for the model's retrieval
performance, our learnings about CPT, a few enhancements to Semantic IDs, along
with an overview of the training and inference methods that enable launching
this framework to billions of users in YouTube.

### 7. [HySim-LLM: Embedding-Weighted Fine-Tuning Bounds and Manifold Denoising for Domain-Adapted LLMs](http://arxiv.org/pdf/2510.07796v1)

Authors: Majid Jaberi-Douraki, Hossein Sholehrasa, Xuan Xu, Remya Ampadi Ramachandran

The extraction and standardization of pharmacokinetic (PK) information from
scientific literature remain significant challenges in computational
pharmacology, which limits the reliability of data-driven models in drug
development. Large language models (LLMs) have achieved remarkable progress in
text understanding and reasoning, yet their adaptation to structured biomedical
data, such as PK tables, remains constrained by heterogeneity, noise, and
domain shift. To address these limitations, we propose HySim-LLM, a unified
mathematical and computational framework that integrates embedding-weighted
fine-tuning and manifold-aware denoising to enhance the robustness and
interpretability of LLMs. We establish two theoretical results: (1) a
similarity-weighted generalization bound that quantifies adaptation performance
under embedding divergence, and (2) a manifold-based denoising guarantee that
bounds loss contributions from noisy or off-manifold samples. These theorems
provide a principled foundation for fine-tuning LLMs in structured biomedical
settings. The framework offers a mathematically grounded pathway toward
reliable and interpretable LLM adaptation for biomedical and data-intensive
scientific domains.

### 8. [ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](http://arxiv.org/pdf/2510.08252v1)

Authors: Jianlyu Chen, Junwei Lan, Chaofan Li, Defu Lian, Zheng Liu

In this paper, we introduce ReasonEmbed, a novel text embedding model
developed for reasoning-intensive document retrieval. Our work includes three
key technical contributions. First, we propose ReMixer, a new data synthesis
method that overcomes the triviality problem prevalent in previous synthetic
datasets, enabling large-scale production of 82K high-quality training samples.
Second, we design Redapter, a self-adaptive learning algorithm that dynamically
adjusts training each sample's weight based on its reasoning intensity. This
allows the model to effectively capture the complex semantic relationships
between queries and documents. Third, we implement ReasonEmbed across multiple
backbones of varying sizes, all of which achieve superior performance on
reasoning-intensive retrieval tasks. Notably, our ReasonEmbed-Qwen3-8B model
offers a record-high nDCG@10 score of 38.1 on the BRIGHT benchmark, which
significantly outperforms existing text embedding models. We will fully
open-source our created resources in ReasonEmbed to push forward the research
advancement in this field.

### 9. [TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevance](http://arxiv.org/pdf/2510.08048v1)

Authors: Jianhui Yang, Yiming Jin, Pengkun Jiao, Chenhe Dong, Zerui Huang, Shaowei Yao, Xiaojiang Zhou, Dan Ou, Haihong Tang

Query-product relevance prediction is fundamental to e-commerce search and
has become even more critical in the era of AI-powered shopping, where semantic
understanding and complex reasoning directly shape the user experience and
business conversion. Large Language Models (LLMs) enable generative,
reasoning-based approaches, typically aligned via supervised fine-tuning (SFT)
or preference optimization methods like Direct Preference Optimization (DPO).
However, the increasing complexity of business rules and user queries exposes
the inability of existing methods to endow models with robust reasoning
capacity for long-tail and challenging cases. Efforts to address this via
reinforcement learning strategies like Group Relative Policy Optimization
(GRPO) often suffer from sparse terminal rewards, offering insufficient
guidance for multi-step reasoning and slowing convergence. To address these
challenges, we propose TaoSR-AGRL, an Adaptive Guided Reinforcement Learning
framework for LLM-based relevance prediction in Taobao Search Relevance.
TaoSR-AGRL introduces two key innovations: (1) Rule-aware Reward Shaping, which
decomposes the final relevance judgment into dense, structured rewards aligned
with domain-specific relevance criteria; and (2) Adaptive Guided Replay, which
identifies low-accuracy rollouts during training and injects targeted
ground-truth guidance to steer the policy away from stagnant, rule-violating
reasoning patterns toward compliant trajectories. TaoSR-AGRL was evaluated on
large-scale real-world datasets and through online side-by-side human
evaluations on Taobao Search. It consistently outperforms DPO and standard GRPO
baselines in offline experiments, improving relevance accuracy, rule adherence,
and training stability. The model trained with TaoSR-AGRL has been successfully
deployed in the main search scenario on Taobao, serving hundreds of millions of
users.

### 10. [VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents](http://arxiv.org/pdf/2510.08109v1)

Authors: Daniel Huwiler, Kurt Stockinger, Jonathan Fürst

Retrieval-Augmented Generation (RAG) systems fail when documents evolve
through versioning-a ubiquitous characteristic of technical documentation.
Existing approaches achieve only 58-64% accuracy on version-sensitive
questions, retrieving semantically similar content without temporal validity
checks. We present VersionRAG, a version-aware RAG framework that explicitly
models document evolution through a hierarchical graph structure capturing
version sequences, content boundaries, and changes between document states.
During retrieval, VersionRAG routes queries through specialized paths based on
intent classification, enabling precise version-aware filtering and change
tracking. On our VersionQA benchmark-100 manually curated questions across 34
versioned technical documents-VersionRAG achieves 90% accuracy, outperforming
naive RAG (58%) and GraphRAG (64%). VersionRAG reaches 60% accuracy on implicit
change detection where baselines fail (0-10%), demonstrating its ability to
track undocumented modifications. Additionally, VersionRAG requires 97% fewer
tokens during indexing than GraphRAG, making it practical for large-scale
deployment. Our work establishes versioned document QA as a distinct task and
provides both a solution and benchmark for future research.

### Machine Learning

### 1. [Property Classification of Vacation Rental Properties during Covid-19](http://arxiv.org/pdf/2510.07639v1)

Authors: Favour Yahdii Aghaebe, Dustin Foley, Eric Atwell, Stephen Clark

This study advocates for employing clustering techniques to classify vacation
rental properties active during the Covid pandemic to identify inherent
patterns and behaviours. The dataset, a collaboration between the ESRC funded
Consumer Data Research Centre (CDRC) and AirDNA, encompasses data for over a
million properties and hosts. Utilising K-means and K-medoids clustering
techniques, we identify homogenous groups and their common characteristics. Our
findings enhance comprehension of the intricacies of vacation rental
evaluations and could potentially be utilised in the creation of targeted,
cluster-specific policies.

### 2. [Design-Based Bandits Under Network Interference: Trade-Off Between Regret and Statistical Inference](http://arxiv.org/pdf/2510.07646v1)

Authors: Zichen Wang, Haoyang Hong, Chuanhao Li, Haoxuan Li, Zhiheng Zhang, Huazheng Wang

In multi-armed bandits with network interference (MABNI), the action taken by
one node can influence the rewards of others, creating complex interdependence.
While existing research on MABNI largely concentrates on minimizing regret, it
often overlooks the crucial concern that an excessive emphasis on the optimal
arm can undermine the inference accuracy for sub-optimal arms. Although initial
efforts have been made to address this trade-off in single-unit scenarios,
these challenges have become more pronounced in the context of MABNI. In this
paper, we establish, for the first time, a theoretical Pareto frontier
characterizing the trade-off between regret minimization and inference accuracy
in adversarial (design-based) MABNI. We further introduce an anytime-valid
asymptotic confidence sequence along with a corresponding algorithm,
$\texttt{EXP3-N-CS}$, specifically designed to balance the trade-off between
regret minimization and inference accuracy in this setting.

### 3. [Continual Learning for Adaptive AI Systems](http://arxiv.org/pdf/2510.07648v1)

Authors: Md Hasibul Amin, Tamzid Tanvi Alam

Continual learning the ability of a neural network to learn multiple
sequential tasks without losing previously acquired knowledge remains a
significant obstacle to developing truly adaptive artificial intelligence. Deep
learning models have achieved remarkable results in various applications, but
overfitting remains a common issue. Regularization techniques can help prevent
overfitting by adding constraints to the model's parameters. To prevent
catastrophic forgetting, in this paper we introduce a novel regularization
technique based on inter-cluster separation (ICS) in the loss function, which
penalizes the model for producing outputs that are far away from the centroids
of the clusters formed by the data from previous tasks. We also performed
hyperparameter tuning to find the optimal weighting of the proposed
regularization term. This ensures clearer separation between tasks in the
neural network's internal representation, reducing overlap and mitigating
forgetting. Using the standard 5-task Split CIFAR-10 benchmark and a ResNet-18
architecture, we demonstrate ICS's effectiveness in maintaining strong
performance on initial tasks. However, our results also highlight limitations
in long-term knowledge retention, particularly when the number of tasks
increases. This underscores the complexity and trade-offs inherent in continual
learning and points toward avenues for further research.

### 4. [Incremental Hybrid Ensemble with Graph Attention and Frequency-Domain Features for Stable Long-Term Credit Risk Modeling](http://arxiv.org/pdf/2510.07663v1)

Authors: Jiajing Wang

Predicting long-term loan defaults is hard because borrower behavior often
changes and data distributions shift over time. This paper presents HYDRA-EI, a
hybrid ensemble incremental learning framework. It uses several stages of
feature processing and combines multiple models. The framework builds
relational, cross, and frequency-based features. It uses graph attention,
automatic cross-feature creation, and transformations from the frequency
domain. HYDRA-EI updates weekly using new data and adjusts the model weights
with a simple performance-based method. It works without frequent manual
changes or fixed retraining. HYDRA-EI improves model stability and
generalization, which makes it useful for long-term credit risk tasks.

### 5. [Computationally-efficient Graph Modeling with Refined Graph Random Features](http://arxiv.org/pdf/2510.07716v1)

Authors: Krzysztof Choromanski, Avinava Dubey, Arijit Sehanobish, Isaac Reid

We propose refined GRFs (GRFs++), a new class of Graph Random Features (GRFs)
for efficient and accurate computations involving kernels defined on the nodes
of a graph. GRFs++ resolve some of the long-standing limitations of regular
GRFs, including difficulty modeling relationships between more distant nodes.
They reduce dependence on sampling long graph random walks via a novel
walk-stitching technique, concatenating several shorter walks without breaking
unbiasedness. By applying these techniques, GRFs++ inherit the approximation
quality provided by longer walks but with greater efficiency, trading
sequential, inefficient sampling of a long walk for parallel computation of
short walks and matrix-matrix multiplication. Furthermore, GRFs++ extend the
simplistic GRFs walk termination mechanism (Bernoulli schemes with fixed
halting probabilities) to a broader class of strategies, applying general
distributions on the walks' lengths. This improves the approximation accuracy
of graph kernels, without incurring extra computational cost. We provide
empirical evaluations to showcase all our claims and complement our results
with theoretical analysis.

### 6. [GeoGen: A Two-stage Coarse-to-Fine Framework for Fine-grained Synthetic Location-based Social Network Trajectory Generation](http://arxiv.org/pdf/2510.07735v1)

Authors: Rongchao Xu, Kunlin Cai, Lin Jiang, Dahai Yu, Zhiqing Hong, Yuan Tian, Guang Wang

Location-Based Social Network (LBSN) check-in trajectory data are important
for many practical applications, like POI recommendation, advertising, and
pandemic intervention. However, the high collection costs and ever-increasing
privacy concerns prevent us from accessing large-scale LBSN trajectory data.
The recent advances in synthetic data generation provide us with a new
opportunity to achieve this, which utilizes generative AI to generate synthetic
data that preserves the characteristics of real data while ensuring privacy
protection. However, generating synthetic LBSN check-in trajectories remains
challenging due to their spatially discrete, temporally irregular nature and
the complex spatio-temporal patterns caused by sparse activities and uncertain
human mobility. To address this challenge, we propose GeoGen, a two-stage
coarse-to-fine framework for large-scale LBSN check-in trajectory generation.
In the first stage, we reconstruct spatially continuous, temporally regular
latent movement sequences from the original LBSN check-in trajectories and then
design a Sparsity-aware Spatio-temporal Diffusion model (S$^2$TDiff) with an
efficient denosing network to learn their underlying behavioral patterns. In
the second stage, we design Coarse2FineNet, a Transformer-based Seq2Seq
architecture equipped with a dynamic context fusion mechanism in the encoder
and a multi-task hybrid-head decoder, which generates fine-grained LBSN
trajectories based on coarse-grained latent movement sequences by modeling
semantic relevance and behavioral uncertainty. Extensive experiments on four
real-world datasets show that GeoGen excels state-of-the-art models for both
fidelity and utility evaluation, e.g., it increases over 69% and 55% in
distance and radius metrics on the FS-TKY dataset.

### 7. [t-SNE Exaggerates Clusters, Provably](http://arxiv.org/pdf/2510.07746v1)

Authors: Noah Bergam, Szymon Snoeck, Nakul Verma

Central to the widespread use of t-distributed stochastic neighbor embedding
(t-SNE) is the conviction that it produces visualizations whose structure
roughly matches that of the input. To the contrary, we prove that (1) the
strength of the input clustering, and (2) the extremity of outlier points,
cannot be reliably inferred from the t-SNE output. We demonstrate the
prevalence of these failure modes in practice as well.

### 8. [FedBook: A Unified Federated Graph Foundation Codebook with Intra-domain and Inter-domain Knowledge Modeling](http://arxiv.org/pdf/2510.07755v1)

Authors: Zhengyu Wu, Yinlin Zhu, Xunkai Li, Ziang Qiu, Rong-Hua Li, Guoren Wang, Chenghu Zhou

Foundation models have shown remarkable cross-domain generalization in
language and vision, inspiring the development of graph foundation models
(GFMs). However, existing GFMs typically assume centralized access to
multi-domain graphs, which is often infeasible due to privacy and institutional
constraints. Federated Graph Foundation Models (FedGFMs) address this
limitation, but their effectiveness fundamentally hinges on constructing a
robust global codebook that achieves intra-domain coherence by consolidating
mutually reinforcing semantics within each domain, while also maintaining
inter-domain diversity by retaining heterogeneous knowledge across domains. To
this end, we propose FedBook, a unified federated graph foundation codebook
that systematically aggregates clients' local codebooks during server-side
federated pre-training. FedBook follows a two-phase process: (1) Intra-domain
Collaboration, where low-frequency tokens are refined by referencing more
semantically reliable high-frequency tokens across clients to enhance
domain-specific coherence; and (2) Inter-domain Integration, where client
contributions are weighted by the semantic distinctiveness of their codebooks
during the aggregation of the global GFM, thereby preserving cross-domain
diversity. Extensive experiments on 8 benchmarks across multiple domains and
tasks demonstrate that FedBook consistently outperforms 21 baselines, including
isolated supervised learning, FL/FGL, federated adaptations of centralized
GFMs, and FedGFM techniques.

### 9. [Rényi Sharpness: A Novel Sharpness that Strongly Correlates with Generalization](http://arxiv.org/pdf/2510.07758v1)

Authors: Qiaozhe Zhang, Jun Sun, Ruijie Zhang, Yingzhuang Liu

Sharpness (of the loss minima) is a common measure to investigate the
generalization of neural networks. Intuitively speaking, the flatter the
landscape near the minima is, the better generalization might be.
Unfortunately, the correlation between many existing sharpness measures and the
generalization is usually not strong, sometimes even weak. To close the gap
between the intuition and the reality, we propose a novel sharpness measure,
i.e., \textit{R\'enyi sharpness}, which is defined as the negative R\'enyi
entropy (a generalization of the classical Shannon entropy) of the loss
Hessian. The main ideas are as follows: 1) we realize that \textit{uniform}
(identical) eigenvalues of the loss Hessian is most desirable (while keeping
the sum constant) to achieve good generalization; 2) we employ the
\textit{R\'enyi entropy} to concisely characterize the extent of the spread of
the eigenvalues of loss Hessian. Normally, the larger the spread, the smaller
the (R\'enyi) entropy. To rigorously establish the relationship between
generalization and (R\'enyi) sharpness, we provide several generalization
bounds in terms of R\'enyi sharpness, by taking advantage of the
reparametrization invariance property of R\'enyi sharpness, as well as the
trick of translating the data discrepancy to the weight perturbation.
Furthermore, extensive experiments are conducted to verify the strong
correlation (in specific, Kendall rank correlation) between the R\'enyi
sharpness and generalization. Moreover, we propose to use a variant of R\'enyi
Sharpness as regularizer during training, i.e., R\'enyi Sharpness Aware
Minimization (RSAM), which turns out to outperform all existing sharpness-aware
minimization methods. It is worthy noting that the test accuracy gain of our
proposed RSAM method could be as high as nearly 2.5\%, compared against the
classical SAM method.

### 10. [FedLAM: Low-latency Wireless Federated Learning via Layer-wise Adaptive Modulation](http://arxiv.org/pdf/2510.07766v1)

Authors: Linping Qu, Shenghui Song, Chi-Ying Tsui

In wireless federated learning (FL), the clients need to transmit the
high-dimensional deep neural network (DNN) parameters through bandwidth-limited
channels, which causes the communication latency issue. In this paper, we
propose a layer-wise adaptive modulation scheme to save the communication
latency. Unlike existing works which assign the same modulation level for all
DNN layers, we consider the layers' importance which provides more freedom to
save the latency. The proposed scheme can automatically decide the optimal
modulation levels for different DNN layers. Experimental results show that the
proposed scheme can save up to 73.9% of communication latency compared with the
existing schemes.

### Neural and Evolutionary Computing

### 1. [Co-design is powerful and not free](http://arxiv.org/pdf/2510.08368v1)

Authors: Yi Zhang, Yue Xie, Tao Sun, Fumiya Iida

Robotic performance emerges from the coupling of body and controller, yet it
remains unclear when morphology-control co-design is necessary. We present a
unified framework that embeds morphology and control parameters within a single
neural network, enabling end-to-end joint optimization. Through case studies in
static-obstacle-constrained reaching, we evaluate trajectory error, success
rate, and collision probability. The results show that co-design provides clear
benefits when morphology is poorly matched to the task, such as near obstacles
or workspace boundaries, where structural adaptation simplifies control.
Conversely, when the baseline morphology already affords sufficient capability,
control-only optimization often matches or exceeds co-design. By clarifying
when control is enough and when it is not, this work advances the understanding
of embodied intelligence and offers practical guidance for embodiment-aware
robot design.

### Networking and Internet Architecture

### 1. [TDoA-Based Self-Supervised Channel Charting with NLoS Mitigation](http://arxiv.org/pdf/2510.08001v1)

Authors: Mohsen Ahadi, Omid Esrafilian, Florian Kaltenberger, Adeel Malik

Channel Charting (CC) has emerged as a promising framework for data-driven
radio localization, yet existing approaches often struggle to scale globally
and to handle the distortions introduced by non-line-of-sight (NLoS)
conditions. In this work, we propose a novel CC method that leverages Channel
Impulse Response (CIR) data enriched with practical features such as Time
Difference of Arrival (TDoA) and Transmission Reception Point (TRP) locations,
enabling a self-supervised localization function on a global scale. The
proposed framework is further enhanced with short-interval User Equipment (UE)
displacement measurements, which improve the continuity and robustness of the
learned positioning function. Our algorithm incorporates a mechanism to
identify and mask NLoS-induced noisy measurements, leading to significant
performance gains. We present the evaluations of our proposed models in a real
5G testbed and benchmarked against centimeter-accurate Real-Time Kinematic
(RTK) positioning, in an O-RAN--based 5G network by OpenAirInterface (OAI)
software at EURECOM. It demonstrated outperforming results against the
state-of-the-art semi-supervised and self-supervised CC approaches in a
real-world scenario. The results show localization accuracies of 2-4 meters in
90% of cases, across a range of NLoS ratios. Furthermore, we provide public
datasets of CIR recordings, along with the true position labels used in this
paper's evaluation.

### 2. [URLLC for 6G Enabled Industry 5.0: A Taxonomy of Architectures, Cross Layer Techniques, and Time Critical Applications](http://arxiv.org/pdf/2510.08080v1)

Authors: Abdikarim Mohamed Ibrahim, Rosdiadee Nordin, Yahya S. M. Khamayseh, Angela Amphawan, Muhammed Basheer Jasser

The evolution from Industry 4.0 to Industry 5.0 introduces stringent
requirements for ultra reliable low latency communication (URLLC) to support
human centric, intelligent, and resilient industrial systems. Sixth-generation
(6G) wireless networks aim to meet these requirements through sub-millisecond
end-to-end delays, microsecond level jitter, and near perfect reliability,
enabled by advances such as terahertz (THz) communication, reconfigurable
intelligent surfaces (RIS), multi-access edge computing (MEC), and AI driven
cross layer optimization. This paper presents a comprehensive review of URLLC
solutions for 6G enabled industry 5.0, organized into a structured taxonomy
including application domains, key technical enablers, design challenges, and
performance enhancements. The survey examines emerging approaches, including
digital twin integration, AI/ML based resource orchestration, Network Function
Virtualization (NFV) enabled service function chaining, and cross domain
networking, while mapping them to critical industrial scenarios such as smart
manufacturing, connected healthcare, autonomous mobility, remote control, and
next-generation mobile networks. Performance trade-offs between latency,
reliability, scalability, and energy efficiency are analyzed in the context of
representative state-of-the-art studies. Finally, the paper identifies open
challenges and outlines future research directions to realize deterministic,
secure, and sustainable URLLC architectures for Industry 5.0.

### 3. [When Light Bends to the Collective Will: A Theory and Vision for Adaptive Photonic Scale-up Domains](http://arxiv.org/pdf/2510.08072v1)

Authors: Vamsi Addanki

As chip-to-chip silicon photonics gain traction for their bandwidth and
energy efficiency, collective communication has emerged as a critical
bottleneck in scale-up systems. Programmable photonic interconnects offer a
promising path forward: by dynamically reconfiguring the fabric, they can
establish direct, high-bandwidth optical paths between communicating endpoints
-- \emph{synchronously and guided by the structure of collective operations}
(e.g., AllReduce). However, realizing this vision -- \emph{when light bends to
the collective will} -- requires navigating a fundamental trade-off between
reconfiguration delay and the performance gains of adaptive topologies.
  In this paper, we present a simple theoretical framework for adaptive
photonic scale-up domains that makes this trade-off explicit and clarifies when
reconfiguration is worthwhile. Along the way, we highlight a connection -- not
surprising but still powerful -- between the Birkhoff--von Neumann (BvN)
decomposition, maximum concurrent flow (a classic measure of network
throughput), and the well-known $\alpha$-$\beta$ cost model for collectives.
Finally, we outline a research agenda in algorithm design and systems
integration that can build on this foundation.

### 4. [BlockSDN: Towards a High-Performance Blockchain via Software-Defined Cross Networking optimization](http://arxiv.org/pdf/2510.08139v1)

Authors: Wenyang Jia, Jingjing Wang, Ziwei Yan, Xiangli Peng, Guohui Yuan

The scalability of blockchain systems is constrained by inefficient P2P
broadcasting, as most existing optimizations focus only on the logical layer
without considering physical network conditions. To address this, we propose
BlockSDN, the first SDN-based integrated architecture for blockchain. BlockSDN
employs a distributed control plane for a global network view, a graph engine
for hierarchical clustering, and a hybrid macro-micro neighbor selection with
hierarchical broadcasting. A dedicated simulation platform shows that BlockSDN
reduces global block synchronization time by 65% and 55% compared to Gossip and
Mercury, respectively.These results highlight the potential of SDN-enabled
cross-layer coordination to significantly enhance blockchain scalability and
performance.

### 5. [Dynamic Features Adaptation in Networking: Toward Flexible training and Explainable inference](http://arxiv.org/pdf/2510.08303v1)

Authors: Yannis Belkhiter, Seshu Tirupathi, Giulio Zizzo, Merim Dzaferagic, John D. Kelleher

As AI becomes a native component of 6G network control, AI models must adapt
to continuously changing conditions, including the introduction of new features
and measurements driven by multi-vendor deployments, hardware upgrades, and
evolving service requirements. To address this growing need for flexible
learning in non-stationary environments, this vision paper highlights Adaptive
Random Forests (ARFs) as a reliable solution for dynamic feature adaptation in
communication network scenarios. We show that iterative training of ARFs can
effectively lead to stable predictions, with accuracy improving over time as
more features are added. In addition, we highlight the importance of
explainability in AI-driven networks, proposing Drift-Aware Feature Importance
(DAFI) as an efficient XAI feature importance (FI) method. DAFI uses a
distributional drift detector to signal when to apply computationally intensive
FI methods instead of lighter alternatives. Our tests on 3 different datasets
indicate that our approach reduces runtime by up to 2 times, while producing
more consistent feature importance values. Together, ARFs and DAFI provide a
promising framework to build flexible AI methods adapted to 6G network
use-cases.

### Robotics

### 1. [Differentiable Particle Optimization for Fast Sequential Manipulation](http://arxiv.org/pdf/2510.07674v1)

Authors: Lucas Chen, Shrutheesh Raman Iyer, Zachary Kingston

Sequential robot manipulation tasks require finding collision-free
trajectories that satisfy geometric constraints across multiple object
interactions in potentially high-dimensional configuration spaces. Solving
these problems in real-time and at large scales has remained out of reach due
to computational requirements. Recently, GPU-based acceleration has shown
promising results, but prior methods achieve limited performance due to CPU-GPU
data transfer overhead and complex logic that prevents full hardware
utilization. To this end, we present SPaSM (Sampling Particle optimization for
Sequential Manipulation), a fully GPU-parallelized framework that compiles
constraint evaluation, sampling, and gradient-based optimization into optimized
CUDA kernels for end-to-end trajectory optimization without CPU coordination.
The method consists of a two-stage particle optimization strategy: first
solving placement constraints through massively parallel sampling, then lifting
solutions to full trajectory optimization in joint space. Unlike hierarchical
approaches, SPaSM jointly optimizes object placements and robot trajectories to
handle scenarios where motion feasibility constrains placement options.
Experimental evaluation on challenging benchmarks demonstrates solution times
in the realm of $\textbf{milliseconds}$ with a 100% success rate; a
$4000\times$ speedup compared to existing approaches.

### 2. [Probabilistically-Safe Bipedal Navigation over Uncertain Terrain via Conformal Prediction and Contraction Analysis](http://arxiv.org/pdf/2510.07725v1)

Authors: Kasidit Muenprasitivej, Ye Zhao, Glen Chou

We address the challenge of enabling bipedal robots to traverse rough terrain
by developing probabilistically safe planning and control strategies that
ensure dynamic feasibility and centroidal robustness under terrain uncertainty.
Specifically, we propose a high-level Model Predictive Control (MPC) navigation
framework for a bipedal robot with a specified confidence level of safety that
(i) enables safe traversal toward a desired goal location across a terrain map
with uncertain elevations, and (ii) formally incorporates uncertainty bounds
into the centroidal dynamics of locomotion control. To model the rough terrain,
we employ Gaussian Process (GP) regression to estimate elevation maps and
leverage Conformal Prediction (CP) to construct calibrated confidence intervals
that capture the true terrain elevation. Building on this, we formulate
contraction-based reachable tubes that explicitly account for terrain
uncertainty, ensuring state convergence and tube invariance. In addition, we
introduce a contraction-based flywheel torque control law for the reduced-order
Linear Inverted Pendulum Model (LIPM), which stabilizes the angular momentum
about the center-of-mass (CoM). This formulation provides both probabilistic
safety and goal reachability guarantees. For a given confidence level, we
establish the forward invariance of the proposed torque control law by
demonstrating exponential stabilization of the actual CoM phase-space
trajectory and the desired trajectory prescribed by the high-level planner.
Finally, we evaluate the effectiveness of our planning framework through
physics-based simulations of the Digit bipedal robot in MuJoCo.

### 3. [Injecting Hallucinations in Autonomous Vehicles: A Component-Agnostic Safety Evaluation Framework](http://arxiv.org/pdf/2510.07749v1)

Authors: Alexandre Moreira Nascimento, Gabriel Kenji Godoy Shimanuki, Lúcio Flavio Vismari, João Batista Camargo Jr, Jorge Rady de Almeida Jr, Paulo Sergio Cugnasca, Anna Carolina Muller Queiroz, Jeremy Noah Bailenson

Perception failures in autonomous vehicles (AV) remain a major safety concern
because they are the basis for many accidents. To study how these failures
affect safety, researchers typically inject artificial faults into hardware or
software components and observe the outcomes. However, existing fault injection
studies often target a single sensor or machine perception (MP) module,
resulting in siloed frameworks that are difficult to generalize or integrate
into unified simulation environments. This work addresses that limitation by
reframing perception failures as hallucinations, false perceptions that distort
an AV situational awareness and may trigger unsafe control actions. Since
hallucinations describe only observable effects, this abstraction enables
analysis independent of specific sensors or algorithms, focusing instead on how
their faults manifest along the MP pipeline. Building on this concept, we
propose a configurable, component-agnostic hallucination injection framework
that induces six plausible hallucination types in an iterative open-source
simulator. More than 18,350 simulations were executed in which hallucinations
were injected while AVs crossed an unsignalized transverse street with traffic.
The results statistically validate the framework and quantify the impact of
each hallucination type on collisions and near misses. Certain hallucinations,
such as perceptual latency and drift, significantly increase the risk of
collision in the scenario tested, validating the proposed paradigm can stress
the AV system safety. The framework offers a scalable, statistically validated,
component agnostic, and fully interoperable toolset that simplifies and
accelerates AV safety validations, even those with novel MP architectures and
components. It can potentially reduce the time-to-market of AV and lay the
foundation for future research on fault tolerance, and resilient AV design.

### 4. [GM3: A General Physical Model for Micro-Mobility Vehicles](http://arxiv.org/pdf/2510.07807v1)

Authors: Grace Cai, Nithin Parepally, Laura Zheng, Ming C. Lin

Modeling the dynamics of micro-mobility vehicles (MMV) is becoming
increasingly important for training autonomous vehicle systems and building
urban traffic simulations. However, mainstream tools rely on variants of the
Kinematic Bicycle Model (KBM) or mode-specific physics that miss tire slip,
load transfer, and rider/vehicle lean. To our knowledge, no unified,
physics-based model captures these dynamics across the full range of common
MMVs and wheel layouts. We propose the "Generalized Micro-mobility Model"
(GM3), a tire-level formulation based on the tire brush representation that
supports arbitrary wheel configurations, including single/double track and
multi-wheel platforms. We introduce an interactive model-agnostic simulation
framework that decouples vehicle/layout specification from dynamics to compare
the GM3 with the KBM and other models, consisting of fixed step RK4
integration, human-in-the-loop and scripted control, real-time trajectory
traces and logging for analysis. We also empirically validate the GM3 on the
Stanford Drone Dataset's deathCircle (roundabout) scene for biker, skater, and
cart classes.

### 5. [USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots](http://arxiv.org/pdf/2510.07869v1)

Authors: Junwen Gu, Zhiheng wu, Pengxuan Si, Shuang Qiu, Yukai Feng, Luoyang Sun, Laien Luo, Lianyi Yu, Jian Wang, Zhengxing Wu

Underwater environments present unique challenges for robotic operation,
including complex hydrodynamics, limited visibility, and constrained
communication. Although data-driven approaches have advanced embodied
intelligence in terrestrial robots and enabled task-specific autonomous
underwater robots, developing underwater intelligence capable of autonomously
performing multiple tasks remains highly challenging, as large-scale,
high-quality underwater datasets are still scarce. To address these
limitations, we introduce USIM, a simulation-based multi-task
Vision-Language-Action (VLA) dataset for underwater robots. USIM comprises over
561K frames from 1,852 trajectories, totaling approximately 15.6 hours of
BlueROV2 interactions across 20 tasks in 9 diverse scenarios, ranging from
visual navigation to mobile manipulation. Building upon this dataset, we
propose U0, a VLA model for general underwater robots, which integrates
binocular vision and other sensor modalities through multimodal fusion, and
further incorporates a convolution-attention-based perception focus enhancement
module (CAP) to improve spatial understanding and mobile manipulation. Across
tasks such as inspection, obstacle avoidance, scanning, and dynamic tracking,
the framework achieves a success rate of 80%, while in challenging mobile
manipulation tasks, it reduces the distance to the target by 21.2% compared
with baseline methods, demonstrating its effectiveness. USIM and U0 show that
VLA models can be effectively applied to underwater robotic applications,
providing a foundation for scalable dataset construction, improved task
autonomy, and the practical realization of intelligent general underwater
robots.

### 6. [Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots](http://arxiv.org/pdf/2510.07882v1)

Authors: Boyu Li, Siyuan He, Hang Xu, Haoqi Yuan, Yu Zang, Liwei Hu, Junpeng Yue, Zhenxiong Jiang, Pengbo Hu, Börje F. Karlsson, Yehui Tang, Zongqing Lu

In recent years, Multimodal Large Language Models (MLLMs) have demonstrated
the ability to serve as high-level planners, enabling robots to follow complex
human instructions. However, their effectiveness, especially in long-horizon
tasks involving dual-arm humanoid robots, remains limited. This limitation
arises from two main challenges: (i) the absence of simulation platforms that
systematically support task evaluation and data collection for humanoid robots,
and (ii) the insufficient embodiment awareness of current MLLMs, which hinders
reasoning about dual-arm selection logic and body positions during planning. To
address these issues, we present DualTHOR, a new dual-arm humanoid simulator,
with continuous transition and a contingency mechanism. Building on this
platform, we propose Proprio-MLLM, a model that enhances embodiment awareness
by incorporating proprioceptive information with motion-based position
embedding and a cross-spatial encoder. Experiments show that, while existing
MLLMs struggle in this environment, Proprio-MLLM achieves an average
improvement of 19.75% in planning performance. Our work provides both an
essential simulation platform and an effective model to advance embodied
intelligence in humanoid robotics. The code is available at
https://anonymous.4open.science/r/DualTHOR-5F3B.

### 7. [Orientation Learning and Adaptation towards Simultaneous Incorporation of Multiple Local Constraints](http://arxiv.org/pdf/2510.07986v1)

Authors: Gaofeng Li, Peisen Xu, Ruize Wang, Qi Ye, Jiming Chen, Dezhen Song, Yanlong Huang

Orientation learning plays a pivotal role in many tasks. However, the
rotation group SO(3) is a Riemannian manifold. As a result, the distortion
caused by non-Euclidean geometric nature introduces difficulties to the
incorporation of local constraints, especially for the simultaneous
incorporation of multiple local constraints. To address this issue, we propose
the Angle-Axis Space-based orientation representation method to solve several
orientation learning problems, including orientation adaptation and
minimization of angular acceleration. Specifically, we propose a weighted
average mechanism in SO(3) based on the angle-axis representation method. Our
main idea is to generate multiple trajectories by considering different local
constraints at different basepoints. Then these multiple trajectories are fused
to generate a smooth trajectory by our proposed weighted average mechanism,
achieving the goal to incorporate multiple local constraints simultaneously.
Compared with existing solution, ours can address the distortion issue and make
the off-theshelf Euclidean learning algorithm be re-applicable in non-Euclidean
space. Simulation and Experimental evaluations validate that our solution can
not only adapt orientations towards arbitrary desired via-points and cope with
angular acceleration constraints, but also incorporate multiple local
constraints simultaneously to achieve extra benefits, e.g., achieving smaller
acceleration costs.

### 8. [Beyond hospital reach: Autonomous lightweight ultrasound robot for liver sonography](http://arxiv.org/pdf/2510.08106v1)

Authors: Zihan Li, Yixiao Xu, Lei Zhang, Taiyu Han, Xinshan Yang, Yingni Wang, Mingxuan Liu, Shenghai Xin, Linxun Liu, Hongen Liao, Guochen Ning

Liver disease is a major global health burden. While ultrasound is the
first-line diagnostic tool, liver sonography requires locating multiple
non-continuous planes from positions where target structures are often not
visible, for biometric assessment and lesion detection, requiring significant
expertise. However, expert sonographers are severely scarce in resource-limited
regions. Here, we develop an autonomous lightweight ultrasound robot comprising
an AI agent that integrates multi-modal perception with memory attention for
localization of unseen target structures, and a 588-gram 6-degrees-of-freedom
cable-driven robot. By mounting on the abdomen, the system enhances robustness
against motion. Our robot can autonomously acquire expert-level standard liver
ultrasound planes and detect pathology in patients, including two from Xining,
a 2261-meter-altitude city with limited medical resources. Our system performs
effectively on rapid-motion individuals and in wilderness environments. This
work represents the first demonstration of autonomous sonography across
multiple challenging scenarios, potentially transforming access to expert-level
diagnostics in underserved regions.

### 9. [Evaluation of a Robust Control System in Real-World Cable-Driven Parallel Robots](http://arxiv.org/pdf/2510.08270v1)

Authors: Damir Nurtdinov, Aliaksei Korshuk, Alexei Kornaev, Alexander Maloletov

This study evaluates the performance of classical and modern control methods
for real-world Cable-Driven Parallel Robots (CDPRs), focusing on
underconstrained systems with limited time discretization. A comparative
analysis is conducted between classical PID controllers and modern
reinforcement learning algorithms, including Deep Deterministic Policy Gradient
(DDPG), Proximal Policy Optimization (PPO), and Trust Region Policy
Optimization (TRPO). The results demonstrate that TRPO outperforms other
methods, achieving the lowest root mean square (RMS) errors across various
trajectories and exhibiting robustness to larger time intervals between control
updates. TRPO's ability to balance exploration and exploitation enables stable
control in noisy, real-world environments, reducing reliance on high-frequency
sensor feedback and computational demands. These findings highlight TRPO's
potential as a robust solution for complex robotic control tasks, with
implications for dynamic environments and future applications in sensor fusion
or hybrid control strategies.

### 10. [Validation of collision-free spheres of Stewart-Gough platforms for constant orientations using the Application Programming Interface of a CAD software](http://arxiv.org/pdf/2510.08408v1)

Authors: Bibekananda Patra, Rajeevlochana G. Chittawadigi, Sandipan Bandyopadhyay

This paper presents a method of validation of the size of the largest
collision-free sphere (CFS) of a 6-6 Stewart-Gough platform manipulator (SGPM)
for a given orientation of its moving platform (MP) using the Application
Programming Interface (API) of a CAD software. The position of the MP is
updated via the API in an automated manner over a set of samples within a shell
enclosing the surface of the CFS. For each pose of the manipulator, each pair
of legs is investigated for mutual collisions. The CFS is considered safe or
validated iff none of the points falling inside the CFS lead to a collision
between any pair of legs. This approach can not only validate the safety of a
precomputed CFS, but also estimate the same for any spatial parallel
manipulator.

### Software Engineering

### 1. [Interleaved Learning and Exploration: A Self-Adaptive Fuzz Testing Framework for MLIR](http://arxiv.org/pdf/2510.07815v1)

Authors: Zeyu Sun, Jingjing Liang, Weiyi Wang, Chenyao Suo, Junjie Chen, Fanjiang Xu

MLIR (Multi-Level Intermediate Representation) has rapidly become a
foundational technology for modern compiler frameworks, enabling extensibility
across diverse domains. However, ensuring the correctness and robustness of
MLIR itself remains challenging. Existing fuzzing approaches-based on manually
crafted templates or rule-based mutations-struggle to generate sufficiently
diverse and semantically valid test cases, making it difficult to expose subtle
or deep-seated bugs within MLIR's complex and evolving code space. In this
paper, we present FLEX, a novel self-adaptive fuzzing framework for MLIR. FLEX
leverages neural networks for program generation, a perturbed sampling strategy
to encourage diversity, and a feedback-driven augmentation loop that
iteratively improves its model using both crashing and non-crashing test cases.
Starting from a limited seed corpus, FLEX progressively learns valid syntax and
semantics and autonomously produces high-quality test inputs. We evaluate FLEX
on the upstream MLIR compiler against four state-of-the-art fuzzers. In a
30-day campaign, FLEX discovers 80 previously unknown bugs-including multiple
new root causes and parser bugs-while in 24-hour fixed-revision comparisons, it
detects 53 bugs (over 3.5x as many as the best baseline) and achieves 28.2%
code coverage, outperforming the next-best tool by 42%. Ablation studies
further confirm the critical role of both perturbed generation and diversity
augmentation in FLEX's effectiveness.

### 2. [Bug Histories as Sources of Compiler Fuzzing Mutators](http://arxiv.org/pdf/2510.07834v1)

Authors: Lingjun Liu, Feiran Qin, Owolabi Legunsen, Marcelo d'Amorim

Bugs in compilers, which are critical infrastructure today, can have outsized
negative impacts. Mutational fuzzers aid compiler bug detection by
systematically mutating compiler inputs, i.e., programs. Their effectiveness
depends on the quality of the mutators used. Yet, no prior work used compiler
bug histories as a source of mutators. We propose IssueMut, the first approach
for extracting compiler fuzzing mutators from bug histories. Our insight is
that bug reports contain hints about program elements that induced compiler
bugs; they can guide fuzzers towards similar bugs. IssueMut uses an automated
method to mine mutators from bug reports and retrofit such mutators into
existing mutational compiler fuzzers. Using IssueMut, we mine 587 mutators from
1760 GCC and LLVM bug reports. Then, we run IssueMut on these compilers, with
all their test inputs as seed corpora. We find that "bug history" mutators are
effective: they find new bugs that a state-of-the-art mutational compiler
fuzzer misses-28 in GCC and 37 in LLVM. Of these, 60 were confirmed or fixed,
validating our idea that bug histories have rich information that compiler
fuzzers should leverage.

### 3. [An AUTOSAR-Aligned Architectural Study of Vulnerabilities in Automotive SoC Software](http://arxiv.org/pdf/2510.07941v1)

Authors: Srijita Basu, Haraldsson Bengt, Miroslaw Staron, Christian Berger, Jennifer Horkoff, Magnus Almgren

Cooperative, Connected and Automated Mobility (CCAM) are complex
cyber-physical systems (CPS) that integrate computation, communication, and
control in safety-critical environments. At their core, System-on-Chip (SoC)
platforms consolidate processing units, communication interfaces, AI
accelerators, and security modules into a single chip. AUTOSAR (AUTomotive Open
System ARchitecture) standard was developed in the automotive domain to better
manage this complexity, defining layered software structures and interfaces to
facilitate reuse of HW/SW components. However, in practice, this integrated SoC
software architecture still poses security challenges, particularly in
real-time, safety-critical environments. Recent reports highlight a surge in
SoC-related vulnerabilities, yet systematic analysis of their root causes and
impact within AUTOSAR-aligned architectures is lacking. This study fills that
gap by analyzing 180 publicly reported automotive SoC vulnerabilities, mapped
to a representative SoC software architecture model that is aligned with
AUTOSAR principles for layered abstraction and service orientation. We identify
16 root causes and 56 affected software modules, and examine mitigation delays
across Common Weakness Enumeration (CWE) categories and architectural layers.
We uncover dominant vulnerability patterns and critical modules with prolonged
patch delays, and provide actionable insights for securing automotive CPS
platforms, including guides for improved detection, prioritization, and
localization strategies for SoC software architectures in SoC-based vehicle
platforms.

### 4. [Building Whitespace-Sensitive Languages Using Whitespace-Insensitive Components](http://arxiv.org/pdf/2510.08200v1)

Authors: Alexander Hellwig, Nico Jansen, Bernhard Rumpe

In Software Language Engineering, there is a trend towards reusability by
composing modular language components. However, this reusability is severely
inhibited by a gap in integrating whitespace-sensitive and
whitespace-insensitive languages. There is currently no consistent procedure
for seamlessly reusing such language components in both cases, such that
libraries often cannot be reused, and whitespacesensitive languages are
developed from scratch. This paper presents a technique for using modular,
whitespaceinsensitive language modules to construct whitespace sensitive
languages by pre-processing language artifacts before parsing. The approach is
evaluated by reconstructing a simplified version of the programming language
Python. Our solution aims to increase the reusability of existing language
components to reduce development time and increase the overall quality of
software languages.

### 5. [AppForge: From Assistant to Independent Developer -- Are GPTs Ready for Software Development?](http://arxiv.org/pdf/2510.07740v1)

Authors: Dezhi Ran, Yuan Cao, Mengzhou Wu, Simin Chen, Yuzhe Guo, Jun Ren, Zihe Song, Hao Yu, Jialei Wei, Linyi Li, Wei Yang, Baishakhi Ray, Tao Xie

Large language models (LLMs) have demonstrated remarkable capability in
function-level code generation tasks. Unlike isolated functions, real-world
applications demand reasoning over the entire software system: developers must
orchestrate how different components interact, maintain consistency across
states over time, and ensure the application behaves correctly within the
lifecycle and framework constraints. Yet, no existing benchmark adequately
evaluates whether LLMs can bridge this gap and construct entire software
systems from scratch. To address this gap, we propose APPFORGE, a benchmark
consisting of 101 software development problems drawn from real-world Android
apps. Given a natural language specification detailing the app functionality, a
language model is tasked with implementing the functionality into an Android
app from scratch. Developing an Android app from scratch requires understanding
and coordinating app states, lifecycle management, and asynchronous operations,
calling for LLMs to generate context-aware, robust, and maintainable code. To
construct APPFORGE, we design a multi-agent system to automatically summarize
the main functionalities from app documents and navigate the app to synthesize
test cases validating the functional correctness of app implementation.
Following rigorous manual verification by Android development experts, APPFORGE
incorporates the test cases within an automated evaluation framework that
enables reproducible assessment without human intervention, making it easily
adoptable for future research. Our evaluation on 12 flagship LLMs show that all
evaluated models achieve low effectiveness, with the best-performing model
(GPT-5) developing only 18.8% functionally correct applications, highlighting
fundamental limitations in current models' ability to handle complex,
multi-component software engineering challenges.

### 6. [Past, Present, and Future of Bug Tracking in the Generative AI Era](http://arxiv.org/pdf/2510.08005v1)

Authors: Utku Boran Torun, Mehmet Taha Demircan, Mahmut Furkan Gön, Eray Tüzün

Traditional bug tracking systems rely heavily on manual reporting,
reproduction, triaging, and resolution, each carried out by different
stakeholders such as end users, customer support, developers, and testers. This
division of responsibilities requires significant coordination and widens the
communication gap between non-technical users and technical teams, slowing the
process from bug discovery to resolution. Moreover, current systems are highly
asynchronous; users often wait hours or days for a first response, delaying
fixes and contributing to frustration. This paper examines the evolution of bug
tracking, from early paper-based reporting to today's web-based and SaaS
platforms. Building on this trajectory, we propose an AI-powered bug tracking
framework that augments existing tools with intelligent, large language model
(LLM)-driven automation. Our framework addresses two main challenges: reducing
time-to-fix and minimizing human overhead. Users report issues in natural
language, while AI agents refine reports, attempt reproduction, and request
missing details. Reports are then classified, invalid ones resolved through
no-code fixes, and valid ones localized and assigned to developers. LLMs also
generate candidate patches, with human oversight ensuring correctness. By
integrating automation into each phase, our framework accelerates response
times, improves collaboration, and strengthens software maintenance practices
for a more efficient, user-centric future.

### 7. [Accurate and Noise-Tolerant Extraction of Routine Logs in Robotic Process Automation (Extended Version)](http://arxiv.org/pdf/2510.08118v1)

Authors: Massimiliano de Leoni, Faizan Ahmed Khan, Simone Agostinelli

Robotic Process Mining focuses on the identification of the routine types
performed by human resources through a User Interface. The ultimate goal is to
discover routine-type models to enable robotic process automation. The
discovery of routine-type models requires the provision of a routine log.
Unfortunately, the vast majority of existing works do not directly focus on
enabling the model discovery, limiting themselves to extracting the set of
actions that are part of the routines. They were also not evaluated in
scenarios characterized by inconsistent routine execution, hereafter referred
to as noise, which reflects natural variability and occasional errors in human
performance. This paper presents a clustering-based technique that aims to
extract routine logs. Experiments were conducted on nine UI logs from the
literature with different levels of injected noise. Our technique was compared
with existing techniques, most of which are not meant to discover routine logs
but were adapted for the purpose. The results were evaluated through standard
state-of-the-art metrics, showing that we can extract more accurate routine
logs than what the state of the art could, especially in the presence of noise.

### 8. [Investigating Matrix Repartitioning to Address the Over- and Undersubscription Challenge for a GPU-based CFD Solver](http://arxiv.org/pdf/2510.08536v1)

Authors: Gregor Olenik, Marcel Koch, Hartwig Anzt

Modern high-performance computing (HPC) increasingly relies on GPUs, but
integrating GPU acceleration into complex scientific frameworks like OpenFOAM
remains a challenge. Existing approaches either fully refactor the codebase or
use plugin-based GPU solvers, each facing trade-offs between performance and
development effort. In this work, we address the limitations of plugin-based
GPU acceleration in OpenFOAM by proposing a repartitioning strategy that better
balances CPU matrix assembly and GPU-based linear solves. We present a detailed
computational model, describe a novel matrix repartitioning and update
procedure, and evaluate its performance on large-scale CFD simulations. Our
results show that the proposed method significantly mitigates oversubscription
issues, improving solver performance and resource utilization in heterogeneous
CPU-GPU environments.

### 9. [pyGinkgo: A Sparse Linear Algebra Operator Framework for Python](http://arxiv.org/pdf/2510.08230v1)

Authors: Keshvi Tuteja, Gregor Olenik, Roman Mishchuk, Yu-Hsiang Tsai, Markus Götz, Achim Streit, Hartwig Anzt, Charlotte Debus

Sparse linear algebra is a cornerstone of many scientific computing and
machine learning applications. Python has become a popular choice for these
applications due to its simplicity and ease of use. Yet high performance sparse
kernels in Python remain limited in functionality, especially on modern CPU and
GPU architectures. We present pyGinkgo, a lightweight and Pythonic interface to
the Ginkgo library, offering high-performance sparse linear algebra support
with platform portability across CUDA, HIP, and OpenMP backends. pyGinkgo
bridges the gap between high-performance C++ backends and Python usability by
exposing Ginkgo's capabilities via Pybind11 and a NumPy and PyTorch compatible
interface. We benchmark pyGinkgo's performance against state-of-the-art Python
libraries including SciPy, CuPy, PyTorch, and TensorFlow. Results across
hardware from different vendors demonstrate that pyGinkgo consistently
outperforms existing Python tools in both sparse matrix vector (SpMV) product
and iterative solver performance, while maintaining performance parity with
native Ginkgo C++ code. Our work positions pyGinkgo as a compelling backend for
sparse machine learning models and scientific workflows.

### 10. [Platform-Agnostic Modular Architecture for Quantum Benchmarking](http://arxiv.org/pdf/2510.08469v1)

Authors: Neer Patel, Anish Giri, Hrushikesh Pramod Patil, Noah Siekierski, Avimita Chatterjee, Sonika Johri, Timothy Proctor, Thomas Lubinski, Siyuan Niu

We present a platform-agnostic modular architecture that addresses the
increasingly fragmented landscape of quantum computing benchmarking by
decoupling problem generation, circuit execution, and results analysis into
independent, interoperable components. Supporting over 20 benchmark variants
ranging from simple algorithmic tests like Bernstein-Vazirani to complex
Hamiltonian simulation with observable calculations, the system integrates with
multiple circuit generation APIs (Qiskit, CUDA-Q, Cirq) and enables diverse
workflows. We validate the architecture through successful integration with
Sandia's $\textit{pyGSTi}$ for advanced circuit analysis and CUDA-Q for
multi-GPU HPC simulations. Extensibility of the system is demonstrated by
implementing dynamic circuit variants of existing benchmarks and a new quantum
reinforcement learning benchmark, which become readily available across
multiple execution and analysis modes. Our primary contribution is identifying
and formalizing modular interfaces that enable interoperability between
incompatible benchmarking frameworks, demonstrating that standardized
interfaces reduce ecosystem fragmentation while preserving optimization
flexibility. This architecture has been developed as a key enhancement to the
continually evolving QED-C Application-Oriented Performance Benchmarks for
Quantum Computing suite.

### Social and Information Networks

### 1. [Do We Really Need SFT? Prompt-as-Policy over Knowledge Graphs for Cold-start Next POI Recommendation](http://arxiv.org/pdf/2510.08012v1)

Authors: Jinze Wang, Lu Zhang, Yiyang Cui, Zhishu Shen, Xingjun Ma, Jiong Jin, Tiehua Zhang

Next point-of-interest (POI) recommendation is crucial for smart urban
services such as tourism, dining, and transportation, yet most approaches
struggle under cold-start conditions where user-POI interactions are sparse.
Recent efforts leveraging large language models (LLMs) address this challenge
through either supervised fine-tuning (SFT) or in-context learning (ICL).
However, SFT demands costly annotations and fails to generalize to inactive
users, while static prompts in ICL cannot adapt to diverse user contexts. To
overcome these limitations, we propose Prompt-as-Policy over knowledge graphs,
a reinforcement-guided prompting framework that learns to construct prompts
dynamically through contextual bandit optimization. Our method treats prompt
construction as a learnable policy that adaptively determines (i) which
relational evidences to include, (ii) the number of evidence per candidate, and
(iii) their organization and ordering within prompts. More specifically, we
construct a knowledge graph (KG) to discover candidates and mine relational
paths, which are transformed into evidence cards that summarize rationales for
each candidate POI. The frozen LLM then acts as a reasoning engine, generating
recommendations from the KG-discovered candidate set based on the
policy-optimized prompts. Experiments on three real-world datasets demonstrate
that Prompt-as-Policy consistently outperforms state-of-the-art baselines,
achieving average 7.7\% relative improvements in Acc@1 for inactive users,
while maintaining competitive performance on active users, without requiring
model fine-tuning.

### 2. [Geometric opinion exchange polarizes in every dimension](http://arxiv.org/pdf/2510.08190v1)

Authors: Abdou Majeed Alidou, Júlia Baligács, Jan Hązła

A recent line of work studies models of opinion exchange where agent opinions
about $d$ topics are tracked simultaneously. The opinions are represented as
vectors on the unit $(d-1)$-sphere, and the update rule is based on the overall
correlation between the relevant vectors. The update rule reflects the
assumption of biased assimilation, i.e., a pair of opinions is brought closer
together if their correlation is positive and further apart if the correlation
is negative.
  This model seems to induce the polarization of opinions into two antipodal
groups. This is in contrast to many other known models which tend to achieve
consensus. The polarization property has been recently proved for $d=2$, but
the general case of $d \ge 3$ remained open. In this work, we settle the
general case, using a more detailed understanding of the model dynamics and
tools from the theory of random processes.

### 3. [Forecasting the Buzz: Enriching Hashtag Popularity Prediction with LLM Reasoning](http://arxiv.org/pdf/2510.08481v1)

Authors: Yifei Xu, Jiaying Wu, Herun Wan, Yang Li, Zhen Hou, Min-Yen Kan

Hashtag trends ignite campaigns, shift public opinion, and steer millions of
dollars in advertising spend, yet forecasting which tag goes viral is elusive.
Classical regressors digest surface features but ignore context, while large
language models (LLMs) excel at contextual reasoning but misestimate numbers.
We present BuzzProphet, a reasoning-augmented hashtag popularity prediction
framework that (1) instructs an LLM to articulate a hashtag's topical virality,
audience reach, and timing advantage; (2) utilizes these popularity-oriented
rationales to enrich the input features; and (3) regresses on these inputs. To
facilitate evaluation, we release HashView, a 7,532-hashtag benchmark curated
from social media. Across diverse regressor-LLM combinations, BuzzProphet
reduces RMSE by up to 2.8% and boosts correlation by 30% over baselines, while
producing human-readable rationales. Results demonstrate that using LLMs as
context reasoners rather than numeric predictors injects domain insight into
tabular models, yielding an interpretable and deployable solution for social
media trend forecasting.

### 4. [From Keywords to Clusters: AI-Driven Analysis of YouTube Comments to Reveal Election Issue Salience in 2024](http://arxiv.org/pdf/2510.07821v1)

Authors: Raisa M. Simoes, Timoteo Kelly, Eduardo J. Simoes, Praveen Rao

This paper aims to explore two competing data science methodologies to
attempt answering the question, "Which issues contributed most to voters'
choice in the 2024 presidential election?" The methodologies involve novel
empirical evidence driven by artificial intelligence (AI) techniques. By using
two distinct methods based on natural language processing and clustering
analysis to mine over eight thousand user comments on election-related YouTube
videos from one right leaning journal, Wall Street Journal, and one left
leaning journal, New York Times, during pre-election week, we quantify the
frequency of selected issue areas among user comments to infer which issues
were most salient to potential voters in the seven days preceding the November
5th election. Empirically, we primarily demonstrate that immigration and
democracy were the most frequently and consistently invoked issues in user
comments on the analyzed YouTube videos, followed by the issue of identity
politics, while inflation was significantly less frequently referenced. These
results corroborate certain findings of post-election surveys but also refute
the supposed importance of inflation as an election issue. This indicates that
variations on opinion mining, with their analysis of raw user data online, can
be more revealing than polling and surveys for analyzing election outcomes.

### Systems and Control

### 1. [Some Reflections on Sliding Mode Designs in Control Systems: An Example of Adaptive Tracking Control for Simple Mechanical Systems With Friction Without Measurement of Velocity](http://arxiv.org/pdf/2510.07675v1)

Authors: Romeo Ortega, Leyan Fang, Jose Guadalupe Romero

The objective of this note is to share some reflections of the authors
regarding the use of sliding mode designs in control systems. We believe the
abundant, and ever increasing, appearance of this kind of works on our
scientific publications deserves some critical evaluation of their actual role,
relevance and pertinence. First, we discuss the procedure followed by most of
these designs -- illustrated with examples from the literature. Second, we
bring to the readers attention several aspects of the control problem, central
in classical designs, which are disregarded in the sliding mode literature.
Finally, to illustrate with an specific example our previous considerations, we
compare the performance of two adaptive tracking controllers for a simple one
degree of freedom mechanical systems with unknown parameters and static and
Coulomb friction -- that do not rely on the measurement of velocity.

### 2. [Space Logistics Analysis and Incentive Design for Commercialization of Orbital Debris Remediation](http://arxiv.org/pdf/2510.07708v1)

Authors: Asaad Abdul-Hamid, Brycen D. Pearl, Hang Woon Lee, Hao Chen

As orbital debris continues to become a higher priority for the space
industry, there is a need to explore how partnerships between the public and
private space sector may aid in addressing this issue. This research develops a
space logistics framework for planning orbital debris remediation missions,
providing a quantitative basis for partnerships that are mutually beneficial
between space operators and debris remediators. By integrating network-based
space logistics and game theory, we illuminate the high-level costs of
remediating orbital debris, and the surplus that stands to be shared as a
result. These findings indicate significant progress toward the continued
development of a safe, sustainable, and profitable space economy.

### 3. [Multi-Level Multi-Fidelity Methods for Path Integral and Safe Control](http://arxiv.org/pdf/2510.07756v1)

Authors: Zhuoyuan Wang, Takashi Tanaka, Yongxin Chen, Yorie Nakahira

Sampling-based approaches are widely used in systems without analytic models
to estimate risk or find optimal control. However, gathering sufficient data in
such scenarios can be prohibitively costly. On the other hand, in many
situations, low-fidelity models or simulators are available from which samples
can be obtained at low cost. In this paper, we propose an efficient approach
for risk quantification and path integral control that leverages such data from
multiple models with heterogeneous sampling costs. A key technical novelty of
our approach is the integration of Multi-level Monte Carlo (MLMC) and
Multi-fidelity Monte Carlo (MFMC) that enable data from different time and
state representations (system models) to be jointly used to reduce variance and
improve sampling efficiency. We also provide theoretical analysis of the
proposed method and show that our estimator is unbiased and consistent under
mild conditions. Finally, we demonstrate via numerical simulation that the
proposed method has improved computation (sampling costs) vs. accuracy
trade-offs for risk quantification and path integral control.

### 4. [General formulation of an analytic, Lipschitz continuous control allocation for thrust-vectored controlled rigid-bodies](http://arxiv.org/pdf/2510.08119v1)

Authors: Frank Mukwege, Tam Willy Nguyen, Emanuele Garone

This study introduces a systematic and scalable method for arbitrary
rigid-bodies equipped with vectorized thrusters. Two novel solutions are
proposed: a closed-form, Lipschitz continuous mapping that ensures smooth
actuator orientation references, and a convex optimization formulation capable
of handling practical actuator constraints such as thrust saturation and
angular rate limits. Both methods leverage the null-space structure of the
allocation mapping to perform singularity avoidance while generating
sub-optimal yet practical solutions. The effectiveness and generality of the
proposed framework are demonstrated through numerical simulations on a 3DOF
marine vessel and a 6DOF aerial quadcopter.

### 5. [Closed-loop control of sloshing fuel in a spinning spacecraft](http://arxiv.org/pdf/2510.08121v1)

Authors: Umberto Zucchelli, Miguel Alfonso Mendez, Annafederica Urbano, Sebastien Vincent-Bonnieu, Piotr Wenderski, Francesco Sanfedino

New-generation space missions require satellites to carry substantial amounts
of liquid propellant, making it essential to analyse the coupled
control-structure-propellant dynamics in detail. While Computational Fluid
Dynamics (CFD) offers high-fidelity predictions, its computational cost limits
its use in iterative design. Equivalent Mechanical Models (EMMs) provide a
faster alternative, though their predictive performance, especially in
closed-loop scenarios, remains largely unexplored. This work presents a
comparative analysis of a spacecraft under feedback control, using both CFD and
a reduced-order sloshing model. Results show good agreement, validating the
simplified model for the manoeuvrer considered. This validation enables
efficient sensitivity and stability studies, offering a practical tool for
early-stage spacecraft design.

### 6. [SecuLEx: a Secure Limit Exchange Market for Dynamic Operating Envelopes](http://arxiv.org/pdf/2510.08172v1)

Authors: Maurizio Vassallo, Adrien Bolland, Alireza Bahmanyar, Louis Wehenkel, Laurine Duchesne, Dong Liu, Sania Khaskheli, Alexis Ha Thuc, Pedro P. Vergara, Amjad Anvari-Moghaddam, Simon Gerard, Damien Ernst

Distributed energy resources (DERs) are transforming power networks,
challenging traditional operational methods, and requiring new coordination
mechanisms. To address this challenge, this paper introduces SecuLEx (Secure
Limit Exchange), a new market-based paradigm to allocate power injection and
withdrawal limits that guarantee network security during time periods, called
dynamic operating envelopes (DOEs). Under this paradigm, distribution system
operators (DSOs) assign initial DOEs to customers. These limits can be
exchanged afterward through a market, allowing customers to reallocate them
according to their needs while ensuring network operational constraints. We
formalize SecuLEx and illustrate DOE allocation and market exchanges on a
small-scale low-voltage (LV) network, demonstrating that both procedures are
computationally tractable. In this example, SecuLEx reduces renewable
curtailment and improves grid utilization and social welfare compared to
traditional approaches.

### 7. [Satellite Navigation and Control using Physics-Informed Artificial Potential Field and Sliding Mode Controller](http://arxiv.org/pdf/2510.08184v1)

Authors: Rakesh Kumar Sahoo, Paridhi Choudhary, Manoranjan Sinha

Increase in the number of space exploration missions has led to the
accumulation of space debris, posing risk of collision with the operational
satellites. Addressing this challenge is crucial for the sustainability of
space operations. To plan a safe trajectory in the presence of moving space
debris, an integrated approach of artificial potential field and sliding mode
controller is proposed and implemented in this paper. The relative 6-DOF
kinematics and dynamics of the spacecraft is modelled in the framework of
geometric mechanics with the relative configuration expressed through
exponential coordinates. Various collision avoidance guidance algorithms have
been proposed in the literature but the Artificial Potential Field guidance
algorithm is computationally efficient and enables real-time path adjustments
to avoid collision with obstacles. However, it is prone to issues such as local
minima. In literature, local minima issue is typically avoided by either
redefining the potential function such as adding vorticity or by employing
search techniques which are computationally expensive. To address these
challenges, a physics-informed APF is proposed in this paper where Hamiltonian
mechanics is used instead of the traditional Newtonian mechanics-based
approach. In this approach, instead of relying on attractive and repulsive
forces for path planning, the Hamiltonian approach uses the potential field to
define a path of minimum potential. Additionally, to track the desired
trajectory planned by the guidance algorithm within a fixed-time frame, a
non-singular fixed-time sliding mode controller (FTSMC) is used. The proposed
fixed-time sliding surface not only ensures fixed-time convergence of system
states but also guarantees the global stability of the closed-loop system
without singularity. The simulation results presented support the claims made.

### 8. [A Control Allocation Algorithm for Hypersonic Glide Vehicles with Input Limitations](http://arxiv.org/pdf/2510.08275v1)

Authors: Johannes Autenrieb, Patrick Gruhn

Hypersonic glide vehicles (HGVs) operate in challenging flight regimes
characterized by strong nonlinearities in actuation and stringent physical
constraints. These include state-dependent actuator limitations, asymmetric
control bounds, and thermal loads that vary with maneuvering conditions. This
paper introduces an iterative control allocation method to address these
challenges in real time. The proposed algorithm searches for control inputs
that achieve the desired moment commands while respecting constraints on input
magnitude and rate. For slender HGV configurations, thermal loads and drag
generation are strongly correlated-lower drag typically results in reduced
surface heating. By embedding drag-sensitive soft constraints, the method
improves energy efficiency and implicitly reduces surface temperatures,
lowering the vehicle's infrared signature. These features are particularly
advantageous for long-range military operations that require low observability.
The approach is demonstrated using the DLR's Generic Hypersonic Glide Vehicle 2
(GHGV-2) simulation model. The results confirm the method's effectiveness in
maintaining control authority under realistic, constrained flight conditions.

### 9. [CPU- and GPU-Based Parallelization of the Robust Reference Governor](http://arxiv.org/pdf/2510.08288v1)

Authors: Hamid R. Ossareh, William Shayne, Samuel Chevalier

Constraint management is a central challenge in modern control systems. A
solution is the Reference Governor (RG), which is an add-on strategy to
pre-stabilized feedback control systems to enforce state and input constraints
by shaping the reference command. While robust formulations of RG exist for
linear systems, their extension to nonlinear systems is often computationally
intractable. This paper develops a scenario-based robust RG formulation for
nonlinear systems and investigates its parallel implementation on multi-core
CPUs and CUDA-enabled GPUs. We analyze the computational structure of the
algorithm, identify parallelization opportunities, and implement the resulting
schemes on modern parallel hardware. Benchmarking on a nonlinear hydrogen fuel
cell model demonstrates order-of-magnitude speedups (by as much as three orders
of magnitude) compared to sequential implementations.

### 10. [Underground Power Distribution System Restoration Using Inverter Based Resources](http://arxiv.org/pdf/2510.08356v1)

Authors: Wenlong Shi, Hongyi Li, Zhaoyu Wang

Underground power distribution systems (PDSs) are increasingly deployed in
urban areas. The integration of smart devices including smart switchgears,
pad-mounted distribution transformers and inverter-based resources (IBRs)
enhance system resilience, however simultaneously introducing unique
challenges. The challenges include inrush currents caused by trapped charges in
underground cables, ferroresonance in distribution transformers during
energization, and three-phase load imbalance resulting from single-phase
underground laterals. To address these issues, this paper proposes an
underground PDS restoration framework using IBRs. Firstly, an underground cable
energization model is developed to quantify inrush current by analyzing voltage
differences across both switchgear terminals. Secondly, a distribution
transformer energization model is proposed to evaluate ferroresonance using
Q-factor constraints based on underground cable capacitance and damping
resistance. Thirdly, a phase-swapping model is proposed to improve load
balancing by dynamically reassigning lateral-phase connections through smart
switchgears. The proposed models are further integrated into a mixed-integer
nonlinear programming (MINLP) formulation to maximize the total weighted
restored load while constraining inrush currents, ferroresonance, and phase
imbalance. To address the nonlinearity induced by impedance matrix reordering
during phase swapping, a permutation-based linearization technique is proposed.
Finally, case studies on an underground PDS established based on IEEE 123-Node
Test Feeder validate the effectiveness of the proposed strategy in improving
uderground PDS restoration performance.

### Machine Learning (Statistics Category)

### 1. [Rotated Mean-Field Variational Inference and Iterative Gaussianization](http://arxiv.org/pdf/2510.07732v1)

Authors: Yifan Chen, Sifan Liu

We propose to perform mean-field variational inference (MFVI) in a rotated
coordinate system that reduces correlations between variables. The rotation is
determined by principal component analysis (PCA) of a cross-covariance matrix
involving the target's score function. Compared with standard MFVI along the
original axes, MFVI in this rotated system often yields substantially more
accurate approximations with negligible additional cost.
  MFVI in a rotated coordinate system defines a rotation and a coordinatewise
map that together move the target closer to Gaussian. Iterating this procedure
yields a sequence of transformations that progressively transforms the target
toward Gaussian. The resulting algorithm provides a computationally efficient
way to construct flow-like transport maps: it requires only MFVI subproblems,
avoids large-scale optimization, and yields transformations that are easy to
invert and evaluate. In Bayesian inference tasks, we demonstrate that the
proposed method achieves higher accuracy than standard MFVI, while maintaining
much lower computational cost than conventional normalizing flows.

### 2. [When Robustness Meets Conservativeness: Conformalized Uncertainty Calibration for Balanced Decision Making](http://arxiv.org/pdf/2510.07750v1)

Authors: Wenbin Zhou, Shixiang Zhu

Robust optimization safeguards decisions against uncertainty by optimizing
against worst-case scenarios, yet their effectiveness hinges on a prespecified
robustness level that is often chosen ad hoc, leading to either insufficient
protection or overly conservative and costly solutions. Recent approaches using
conformal prediction construct data-driven uncertainty sets with finite-sample
coverage guarantees, but they still fix coverage targets a priori and offer
little guidance for selecting robustness levels. We propose a new framework
that provides distribution-free, finite-sample guarantees on both miscoverage
and regret for any family of robust predict-then-optimize policies. Our method
constructs valid estimators that trace out the miscoverage-regret Pareto
frontier, enabling decision-makers to reliably evaluate and calibrate
robustness levels according to their cost-risk preferences. The framework is
simple to implement, broadly applicable across classical optimization
formulations, and achieves sharper finite-sample performance than existing
approaches. These results offer the first principled data-driven methodology
for guiding robustness selection and empower practitioners to balance
robustness and conservativeness in high-stakes decision-making.

### 3. [Surrogate Graph Partitioning for Spatial Prediction](http://arxiv.org/pdf/2510.07832v1)

Authors: Yuta Shikuri, Hironori Fujisawa

Spatial prediction refers to the estimation of unobserved values from
spatially distributed observations. Although recent advances have improved the
capacity to model diverse observation types, adoption in practice remains
limited in industries that demand interpretability. To mitigate this gap,
surrogate models that explain black-box predictors provide a promising path
toward interpretable decision making. In this study, we propose a graph
partitioning problem to construct spatial segments that minimize the sum of
within-segment variances of individual predictions. The assignment of data
points to segments can be formulated as a mixed-integer quadratic programming
problem. While this formulation potentially enables the identification of exact
segments, its computational complexity becomes prohibitive as the number of
data points increases. Motivated by this challenge, we develop an approximation
scheme that leverages the structural properties of graph partitioning.
Experimental results demonstrate the computational efficiency of this
approximation in identifying spatial segments.

### 4. [On the Optimality of Tracking Fisher Information in Adaptive Testing with Stochastic Binary Responses](http://arxiv.org/pdf/2510.07862v1)

Authors: Sanghwa Kim, Dohyun Ahn, Seungki Min

We study the problem of estimating a continuous ability parameter from
sequential binary responses by actively asking questions with varying
difficulties, a setting that arises naturally in adaptive testing and online
preference learning. Our goal is to certify that the estimate lies within a
desired margin of error, using as few queries as possible. We propose a simple
algorithm that adaptively selects questions to maximize Fisher information and
updates the estimate using a method-of-moments approach, paired with a novel
test statistic to decide when the estimate is accurate enough. We prove that
this Fisher-tracking strategy achieves optimal performance in both
fixed-confidence and fixed-budget regimes, which are commonly invested in the
best-arm identification literature. Our analysis overcomes a key technical
challenge in the fixed-budget setting -- handling the dependence between the
evolving estimate and the query distribution -- by exploiting a structural
symmetry in the model and combining large deviation tools with Ville's
inequality. Our results provide rigorous theoretical support for simple and
efficient adaptive testing procedures.

### 5. [Stick-Breaking Mixture Normalizing Flows with Component-Wise Tail Adaptation for Variational Inference](http://arxiv.org/pdf/2510.07965v1)

Authors: Seungsu Han, Juyoung Hwang, Won Chang

Normalizing flows with a Gaussian base provide a computationally efficient
way to approximate posterior distributions in Bayesian inference, but they
often struggle to capture complex posteriors with multimodality and heavy
tails. We propose a stick-breaking mixture base with component-wise tail
adaptation (StiCTAF) for posterior approximation. The method first learns a
flexible mixture base to mitigate the mode-seeking bias of reverse KL
divergence through a weighted average of component-wise ELBOs. It then
estimates local tail indices of unnormalized densities and finally refines each
mixture component using a shared backbone combined with component-specific tail
transforms calibrated by the estimated indices. This design enables accurate
mode coverage and anisotropic tail modeling while retaining exact density
evaluation and stable optimization. Experiments on synthetic posteriors
demonstrate improved tail recovery and better coverage of multiple modes
compared to benchmark models. We also present a real-data analysis illustrating
the practical benefits of our approach for posterior inference.

### 6. [Beyond Real Data: Synthetic Data through the Lens of Regularization](http://arxiv.org/pdf/2510.08095v1)

Authors: Amitis Shidani, Tyler Farghly, Yang Sun, Habib Ganjgahi, George Deligiannidis

Synthetic data can improve generalization when real data is scarce, but
excessive reliance may introduce distributional mismatches that degrade
performance. In this paper, we present a learning-theoretic framework to
quantify the trade-off between synthetic and real data. Our approach leverages
algorithmic stability to derive generalization error bounds, characterizing the
optimal synthetic-to-real data ratio that minimizes expected test error as a
function of the Wasserstein distance between the real and synthetic
distributions. We motivate our framework in the setting of kernel ridge
regression with mixed data, offering a detailed analysis that may be of
independent interest. Our theory predicts the existence of an optimal ratio,
leading to a U-shaped behavior of test error with respect to the proportion of
synthetic data. Empirically, we validate this prediction on CIFAR-10 and a
clinical brain MRI dataset. Our theory extends to the important scenario of
domain adaptation, showing that carefully blending synthetic target data with
limited source data can mitigate domain shift and enhance generalization. We
conclude with practical guidance for applying our results to both in-domain and
out-of-domain scenarios.

### 7. [High-dimensional Analysis of Synthetic Data Selection](http://arxiv.org/pdf/2510.08123v1)

Authors: Parham Rezaei, Filip Kovacevic, Francesco Locatello, Marco Mondelli

Despite the progress in the development of generative models, their
usefulness in creating synthetic data that improve prediction performance of
classifiers has been put into question. Besides heuristic principles such as
"synthetic data should be close to the real data distribution", it is actually
not clear which specific properties affect the generalization error. Our paper
addresses this question through the lens of high-dimensional regression.
Theoretically, we show that, for linear models, the covariance shift between
the target distribution and the distribution of the synthetic data affects the
generalization error but, surprisingly, the mean shift does not. Furthermore we
prove that, in some settings, matching the covariance of the target
distribution is optimal. Remarkably, the theoretical insights from linear
models carry over to deep neural networks and generative models. We empirically
demonstrate that the covariance matching procedure (matching the covariance of
the synthetic data with that of the data coming from the target distribution)
performs well against several recent approaches for synthetic data selection,
across training paradigms, architectures, datasets and generative models used
for augmentation.

### 8. [PAC Learnability in the Presence of Performativity](http://arxiv.org/pdf/2510.08335v1)

Authors: Ivan Kirev, Lyuben Baltadzhiev, Nikola Konstantinov

Following the wide-spread adoption of machine learning models in real-world
applications, the phenomenon of performativity, i.e. model-dependent shifts in
the test distribution, becomes increasingly prevalent. Unfortunately, since
models are usually trained solely based on samples from the original
(unshifted) distribution, this performative shift may lead to decreased
test-time performance. In this paper, we study the question of whether and when
performative binary classification problems are learnable, via the lens of the
classic PAC (Probably Approximately Correct) learning framework. We motivate
several performative scenarios, accounting in particular for linear shifts in
the label distribution, as well as for more general changes in both the labels
and the features. We construct a performative empirical risk function, which
depends only on data from the original distribution and on the type
performative effect, and is yet an unbiased estimate of the true risk of a
classifier on the shifted distribution. Minimizing this notion of performative
risk allows us to show that any PAC-learnable hypothesis space in the standard
binary classification setting remains PAC-learnable for the considered
performative scenarios. We also conduct an extensive experimental evaluation of
our performative risk minimization method and showcase benefits on synthetic
and real data.

### 9. [Characterizing the Multiclass Learnability of Forgiving 0-1 Loss Functions](http://arxiv.org/pdf/2510.08382v1)

Authors: Jacob Trauger, Tyson Trauger, Ambuj Tewari

In this paper we will give a characterization of the learnability of
forgiving 0-1 loss functions in the finite label multiclass setting. To do
this, we create a new combinatorial dimension that is based off of the
Natarajan Dimension \citep{natarajan1989learning} and we show that a hypothesis
class is learnable in our setting if and only if this Generalized Natarajan
Dimension is finite. We also show a connection to learning with set-valued
feedback. Through our results we show that the learnability of a set learning
problem is characterized by the Natarajan Dimension.

### 10. [Optimal Stopping in Latent Diffusion Models](http://arxiv.org/pdf/2510.08409v1)

Authors: Yu-Han Wu, Quentin Berthet, Gérard Biau, Claire Boyer, Romuald Elie, Pierre Marion

We identify and analyze a surprising phenomenon of Latent Diffusion Models
(LDMs) where the final steps of the diffusion can degrade sample quality. In
contrast to conventional arguments that justify early stopping for numerical
stability, this phenomenon is intrinsic to the dimensionality reduction in
LDMs. We provide a principled explanation by analyzing the interaction between
latent dimension and stopping time. Under a Gaussian framework with linear
autoencoders, we characterize the conditions under which early stopping is
needed to minimize the distance between generated and target distributions.
More precisely, we show that lower-dimensional representations benefit from
earlier termination, whereas higher-dimensional latent spaces require later
stopping time. We further establish that the latent dimension interplays with
other hyperparameters of the problem such as constraints in the parameters of
score matching. Experiments on synthetic and real datasets illustrate these
properties, underlining that early stopping can improve generative quality.
Together, our results offer a theoretical foundation for understanding how the
latent dimension influences the sample quality, and highlight stopping time as
a key hyperparameter in LDMs.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-10 PST.

### 1. [A jamming risk warning model for TBM tunnelling based on Bayesian statistical methods](https://www.nature.com/articles/s41598-025-19244-8)

Authors: Shuang-jing Wang et al.

### 2. [Towards scalable and cross-lingual specialist language models for oncology](https://www.nature.com/articles/s41598-025-19282-2)

Authors: Morteza Rohanian et al.

### 3. [Computer vision-based laser communication system for robust optical beam tracking and alignment](https://www.nature.com/articles/s41598-025-17695-7)

Authors: Shuai Li

### 4. [Deep learning based SentiNet architecture with hyperparameter optimization for sentiment analysis of customer reviews](https://www.nature.com/articles/s41598-025-19532-3)

Authors: B. Madhurika et al.

### 5. [Energy efficient clustering in industrial Iot using a quantum informed artificial hummingbird optimization algorithm](https://www.nature.com/articles/s41598-025-19358-z)

Authors: S. Rajkumar et al.

### 6. [Construction of intelligent decision support systems through integration of retrieval-augmented generation and knowledge graphs](https://www.nature.com/articles/s41598-025-19257-3)

Authors: Sili Wang et al.

### 7. [Evaluation of new quality productive forces in Henan province based on improved entropy weight-TOPSIS method and deep learning](https://www.nature.com/articles/s41598-025-19309-8)

Authors: ShiHui Jiang

### 8. [A gait recognition architecture for early screening in the assessment of Parkinson’s patients](https://www.nature.com/articles/s41598-025-19224-y)

Authors: Huan Wang et al.

### 9. [A deep learning framework for predicting the effect of surface topography on thermal contact resistance](https://www.nature.com/articles/s44172-025-00508-0)

Authors: Man Zhou et al.

### 10. [A novel swarm intelligence optimization method for efficient task allocation in industrial wireless sensor networks](https://www.nature.com/articles/s41598-025-19527-0)

Authors: Chao Wang et al.

