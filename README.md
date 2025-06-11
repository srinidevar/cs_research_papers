# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-06-10 17:09:15.494934 PST.

### Artificial Intelligence

### 1. [An Intelligent Fault Self-Healing Mechanism for Cloud AI Systems via Integration of Large Language Models and Deep Reinforcement Learning](http://arxiv.org/pdf/2506.07411v1)

Authors: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu

As the scale and complexity of cloud-based AI systems continue to increase,
the detection and adaptive recovery of system faults have become the core
challenges to ensure service reliability and continuity. In this paper, we
propose an Intelligent Fault Self-Healing Mechanism (IFSHM) that integrates
Large Language Model (LLM) and Deep Reinforcement Learning (DRL), aiming to
realize a fault recovery framework with semantic understanding and policy
optimization capabilities in cloud AI systems. On the basis of the traditional
DRL-based control model, the proposed method constructs a two-stage hybrid
architecture: (1) an LLM-driven fault semantic interpretation module, which can
dynamically extract deep contextual semantics from multi-source logs and system
indicators to accurately identify potential fault modes; (2) DRL recovery
strategy optimizer, based on reinforcement learning, learns the dynamic
matching of fault types and response behaviors in the cloud environment. The
innovation of this method lies in the introduction of LLM for environment
modeling and action space abstraction, which greatly improves the exploration
efficiency and generalization ability of reinforcement learning. At the same
time, a memory-guided meta-controller is introduced, combined with
reinforcement learning playback and LLM prompt fine-tuning strategy, to achieve
continuous adaptation to new failure modes and avoid catastrophic forgetting.
Experimental results on the cloud fault injection platform show that compared
with the existing DRL and rule methods, the IFSHM framework shortens the system
recovery time by 37% with unknown fault scenarios.

### 2. [Evaluating Visual Mathematics in Multimodal LLMs: A Multilingual Benchmark Based on the Kangaroo Tests](http://arxiv.org/pdf/2506.07418v1)

Authors: Arnau Igualde Sáez, Lamyae Rhomrasi, Yusef Ahsini, Ricardo Vinuesa, Sergio Hoyas, Jose P. García Sabater, Marius J. Fullana i Alfonso, J. Alberto Conejero

Multimodal Large Language Models (MLLMs) promise advanced vision language
capabilities, yet their effectiveness in visually presented mathematics remains
underexplored. This paper analyzes the development and evaluation of MLLMs for
mathematical problem solving, focusing on diagrams, multilingual text, and
symbolic notation. We then assess several models, including GPT 4o, Pixtral,
Qwen VL, Llama 3.2 Vision variants, and Gemini 2.0 Flash in a multilingual
Kangaroo style benchmark spanning English, French, Spanish, and Catalan. Our
experiments reveal four key findings. First, overall precision remains moderate
across geometry, visual algebra, logic, patterns, and combinatorics: no single
model excels in every topic. Second, while most models see improved accuracy
with questions that do not have images, the gain is often limited; performance
for some remains nearly unchanged without visual input, indicating
underutilization of diagrammatic information. Third, substantial variation
exists across languages and difficulty levels: models frequently handle easier
items but struggle with advanced geometry and combinatorial reasoning. Notably,
Gemini 2.0 Flash achieves the highest precision on image based tasks, followed
by Qwen VL 2.5 72B and GPT 4o, though none approach human level performance.
Fourth, a complementary analysis aimed at distinguishing whether models reason
or simply recite reveals that Gemini and GPT 4o stand out for their structured
reasoning and consistent accuracy. In contrast, Pixtral and Llama exhibit less
consistent reasoning, often defaulting to heuristics or randomness when unable
to align their outputs with the given answer options.

### 3. [LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](http://arxiv.org/pdf/2506.07443v1)

Authors: Weijie Shi, Han Zhu, Jiaming Ji, Mengze Li, Jipeng Zhang, Ruiyuan Zhang, Jia Zhu, Jiajie Xu, Sirui Han, Yike Guo

Legal judgment prediction (LJP) aims to function as a judge by making final
rulings based on case claims and facts, which plays a vital role in the
judicial domain for supporting court decision-making and improving judicial
efficiency. However, existing methods often struggle with logical errors when
conducting complex legal reasoning. We propose LegalReasoner, which enhances
LJP reliability through step-wise verification and correction of the reasoning
process. Specifically, it first identifies dispute points to decompose complex
cases, and then conducts step-wise reasoning while employing a process verifier
to validate each step's logic from correctness, progressiveness, and potential
perspectives. When errors are detected, expert-designed attribution and
resolution strategies are applied for correction. To fine-tune LegalReasoner,
we release the LegalHK dataset, containing 58,130 Hong Kong court cases with
detailed annotations of dispute points, step-by-step reasoning chains, and
process verification labels. Experiments demonstrate that LegalReasoner
significantly improves concordance with court decisions from 72.37 to 80.27 on
LLAMA-3.1-70B. The data is available at
https://huggingface.co/datasets/weijiezz/LegalHK.

### 4. [Fact in Fragments: Deconstructing Complex Claims via LLM-based Atomic Fact Extraction and Verification](http://arxiv.org/pdf/2506.07446v1)

Authors: Liwen Zheng, Chaozhuo Li, Zheng Liu, Feiran Huang, Haoran Jia, Zaisheng Ye, Xi Zhang

Fact verification plays a vital role in combating misinformation by assessing
the veracity of claims through evidence retrieval and reasoning. However,
traditional methods struggle with complex claims requiring multi-hop reasoning
over fragmented evidence, as they often rely on static decomposition strategies
and surface-level semantic retrieval, which fail to capture the nuanced
structure and intent of the claim. This results in accumulated reasoning
errors, noisy evidence contamination, and limited adaptability to diverse
claims, ultimately undermining verification accuracy in complex scenarios. To
address this, we propose Atomic Fact Extraction and Verification (AFEV), a
novel framework that iteratively decomposes complex claims into atomic facts,
enabling fine-grained retrieval and adaptive reasoning. AFEV dynamically
refines claim understanding and reduces error propagation through iterative
fact extraction, reranks evidence to filter noise, and leverages
context-specific demonstrations to guide the reasoning process. Extensive
experiments on five benchmark datasets demonstrate that AFEV achieves
state-of-the-art performance in both accuracy and interpretability.

### 5. [Efficient Generation of Diverse Cooperative Agents with World Models](http://arxiv.org/pdf/2506.07450v1)

Authors: Yi Loo, Akshunn Trivedi, Malika Meghjani

A major bottleneck in the training process for Zero-Shot Coordination (ZSC)
agents is the generation of partner agents that are diverse in collaborative
conventions. Current Cross-play Minimization (XPM) methods for population
generation can be very computationally expensive and sample inefficient as the
training objective requires sampling multiple types of trajectories. Each
partner agent in the population is also trained from scratch, despite all of
the partners in the population learning policies of the same coordination task.
In this work, we propose that simulated trajectories from the dynamics model of
an environment can drastically speed up the training process for XPM methods.
We introduce XPM-WM, a framework for generating simulated trajectories for XPM
via a learned World Model (WM). We show XPM with simulated trajectories removes
the need to sample multiple trajectories. In addition, we show our proposed
method can effectively generate partners with diverse conventions that match
the performance of previous methods in terms of SP population training reward
as well as training partners for ZSC agents. Our method is thus, significantly
more sample efficient and scalable to a larger number of partners.

### 6. [Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification](http://arxiv.org/pdf/2506.07528v1)

Authors: Qisheng Hu, Quanyu Long, Wenya Wang

Multi-hop claim verification is inherently challenging, requiring multi-step
reasoning to construct verification chains while iteratively searching for
information to uncover hidden bridging facts. This process is fundamentally
interleaved, as effective reasoning relies on dynamically retrieved evidence,
while effective search demands reasoning to refine queries based on partial
information. To achieve this, we propose Hierarchical Agent Reasoning and
Information Search (HARIS), explicitly modeling the coordinated process of
reasoning-driven searching and search-informed reasoning. HARIS consists of a
high-level reasoning agent that focuses on constructing the main verification
chain, generating factual questions when more information is needed, and a
low-level search agent that iteratively retrieves more information, refining
its search based on intermediate findings. This design allows each agent to
specialize in its respective task, enhancing verification accuracy and
interpretability. HARIS is trained using reinforcement learning with
outcome-based rewards. Experimental results on the EX-FEVER and HOVER
benchmarks demonstrate that HARIS achieves strong performance, greatly
advancing multi-hop claim verification.

### 7. [SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling](http://arxiv.org/pdf/2506.07636v1)

Authors: Haoran Wang, Zhenyu Hou, Yao Wei, Jie Tang, Yuxiao Dong

Large language models (LLMs) have advanced rapidly from conversational
problem solving to addressing real-world tasks involving tool use, such as
software engineering (SWE). Recent LLM-powered toolkits, such as OpenAI Codex
and Cursor, have offered end-to-end automation of the software development
process. However, building effective SWE agents remains challenging due to the
lack of high-quality training data and effective test cases. To address this
issue, we present SWE-Dev, an SWE agent built upon open-source LLMs. First, we
develop a robust pipeline to synthesize test cases for patch evaluation.
Second, we scale up agent trajectories to construct the training data for
building SWE-Dev. Experiments on the SWE-bench-Verified benchmark show that the
SWE-Dev models can achieve top performance among all open SWE agents.
Specifically, the success rates of the SWE-Dev 7B and 32B parameter models
reach 23.4% and 36.6%, respectively, outperforming state-of-the-art open-source
models. All code, models, and datasets are publicly available at
https://github.com/THUDM/SWE-Dev.

### 8. [MCPWorld: A Unified Benchmarking Testbed for API, GUI, and Hybrid Computer Use Agents](http://arxiv.org/pdf/2506.07672v1)

Authors: Yunhe Yan, Shihe Wang, Jiajun Du, Yexuan Yang, Yuxuan Shan, Qichen Qiu, Xianqing Jia, Xinge Wang, Xin Yuan, Xu Han, Mao Qin, Yinxiao Chen, Chen Peng, Shangguang Wang, Mengwei Xu

(M)LLM-powered computer use agents (CUA) are emerging as a transformative
technique to automate human-computer interaction. However, existing CUA
benchmarks predominantly target GUI agents, whose evaluation methods are
susceptible to UI changes and ignore function interactions exposed by
application APIs, e.g., Model Context Protocol (MCP). To this end, we propose
MCPWorld, the first automatic CUA testbed for API, GUI, and API-GUI hybrid
agents. A key principle of MCPWorld is the use of "white-box apps", i.e., those
with source code availability and can be revised/re-compiled as needed (e.g.,
adding MCP support), with two notable advantages:
  (1) It greatly broadens the design space of CUA, such as what and how the app
features to be exposed/extracted as CUA-callable APIs.
  (2) It allows MCPWorld to programmatically verify task completion by directly
monitoring application behavior through techniques like dynamic code
instrumentation, offering robust, accurate CUA evaluation decoupled from
specific agent implementations or UI states.
  Currently, MCPWorld includes 201 well curated and annotated user tasks,
covering diversified use cases and difficulty levels. MCPWorld is also fully
containerized with GPU acceleration support for flexible adoption on different
OS/hardware environments. Our preliminary experiments, using a representative
LLM-powered CUA framework, achieve 75.12% task completion accuracy,
simultaneously providing initial evidence on the practical effectiveness of
agent automation leveraging MCP. Overall, we anticipate MCPWorld to facilitate
and standardize the benchmarking of next-generation computer use agents that
can leverage rich external tools. Our code and dataset are publicly available
at https://github.com/SAAgent/MCPWorld.

### 9. [NeurIPS 2025 E2LM Competition : Early Training Evaluation of Language Models](http://arxiv.org/pdf/2506.07731v1)

Authors: Mouadh Yagoubi, Yasser Dahou, Billel Mokeddem, Younes Belkada, Phuc H. Le-Khac, Basma El Amel Boussaha, Reda Alami, Jingwei Zuo, Damiano Marsili, Mugariya Farooq, Mounia Lalmas, Georgia Gkioxari, Patrick Gallinari, Philip Torr, Hakim Hacid

Existing benchmarks have proven effective for assessing the performance of
fully trained large language models. However, we find striking differences in
the early training stages of small models, where benchmarks often fail to
provide meaningful or discriminative signals. To explore how these differences
arise, this competition tackles the challenge of designing scientific knowledge
evaluation tasks specifically tailored for measuring early training progress of
language models. Participants are invited to develop novel evaluation
methodologies or adapt existing benchmarks to better capture performance
differences among language models. To support this effort, we provide three
pre-trained small models (0.5B, 1B, and 3B parameters), along with intermediate
checkpoints sampled during training up to 200B tokens. All experiments and
development work can be run on widely available free cloud-based GPU platforms,
making participation accessible to researchers with limited computational
resources. Submissions will be evaluated based on three criteria: the quality
of the performance signal they produce, the consistency of model rankings at 1
trillion tokens of training, and their relevance to the scientific knowledge
domain. By promoting the design of tailored evaluation strategies for early
training, this competition aims to attract a broad range of participants from
various disciplines, including those who may not be machine learning experts or
have access to dedicated GPU resources. Ultimately, this initiative seeks to
make foundational LLM research more systematic and benchmark-informed from the
earliest phases of model development.

### 10. [RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards](http://arxiv.org/pdf/2506.07736v1)

Authors: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

Large Language Models (LLMs) continue to exhibit vulnerabilities despite
deliberate safety alignment efforts, posing significant risks to users and
society. To safeguard against the risk of policy-violating content,
system-level moderation via external guard models-designed to monitor LLM
inputs and outputs and block potentially harmful content-has emerged as a
prevalent mitigation strategy. Existing approaches of training guard models
rely heavily on extensive human curated datasets and struggle with
out-of-distribution threats, such as emerging harmful categories or jailbreak
attacks. To address these limitations, we propose RSafe, an adaptive
reasoning-based safeguard that conducts guided safety reasoning to provide
robust protection within the scope of specified safety policies. RSafe operates
in two stages: 1) guided reasoning, where it analyzes safety risks of input
content through policy-guided step-by-step reasoning, and 2) reinforced
alignment, where rule-based RL optimizes its reasoning paths to align with
accurate safety prediction. This two-stage training paradigm enables RSafe to
internalize safety principles to generalize safety protection capability over
unseen or adversarial safety violation scenarios. During inference, RSafe
accepts user-specified safety policies to provide enhanced safeguards tailored
to specific safety requirements.

### Hardware Architecture

### 1. [A Survey on LUT-based Deep Neural Networks Implemented in FPGAs](http://arxiv.org/pdf/2506.07367v1)

Authors: Zeyu Guo

Low-latency, energy-efficient deep neural networks (DNNs) inference are
critical for edge applications, where traditional cloud-based deployment
suffers from high latency and security risks. Field-Programmable Gate Arrays
(FPGAs) offer a compelling solution, balancing reconfigurability, power
efficiency, and real-time performance. However, conventional FPGA-based DNNs
rely heavily on digital signal processing (DSP) blocks for multiply-accumulate
(MAC) operations, limiting scalability.
  LUT-based DNNs address this challenge by fully leveraging FPGA lookup tables
(LUTs) for computation, improving resource utilization and reducing inference
latency. This survey provides a comprehensive review of LUT-based DNN
architectures, including their evolution, design methodologies, and performance
trade-offs, while outlining promising directions for future research.

### 2. [FREESS: An Educational Simulator of a RISC-V-Inspired Superscalar Processor Based on Tomasulo's Algorithm](http://arxiv.org/pdf/2506.07665v1)

Authors: Roberto Giorgi

FREESS is a free, interactive simulator that illustrates instruction-level
parallelism in a RISC-V-inspired superscalar processor. Based on an extended
version of Tomasulo's algorithm, FREESS is intended as a hands-on educational
tool for Advanced Computer Architecture courses. It enables students to explore
dynamic, out-of-order instruction execution, emphasizing how instructions are
issued as soon as their operands become available.
  The simulator models key microarchitectural components, including the
Instruction Window (IW), Reorder Buffer (ROB), Register Map (RM), Free Pool
(FP), and Load/Store Queues. FREESS allows users to dynamically configure
runtime parameters, such as the superscalar issue width, functional unit types
and latencies, and the sizes of architectural buffers and queues.
  To simplify learning, the simulator uses a minimal instruction set inspired
by RISC-V (ADD, ADDI, BEQ, BNE, LW, MUL, SW), which is sufficient to
demonstrate key pipeline stages: fetch, register renaming, out-of-order
dispatch, execution, completion, commit, speculative branching, and memory
access. FREESS includes three step-by-step, illustrated examples that visually
demonstrate how multiple instructions can be issued and executed in parallel
within a single cycle. Being open source, FREESS encourages students and
educators to experiment freely by writing and analyzing their own
instruction-level programs and superscalar architectures.

### 3. [MoE-GPS: Guidlines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing](http://arxiv.org/pdf/2506.07366v1)

Authors: Haiyue Ma, Zhixu Du, Yiran Chen

In multi-GPU Mixture-of-Experts (MoE) network, experts are distributed across
different GPUs, which creates load imbalance as each expert processes different
number of tokens. Recent works improve MoE inference load balance by
dynamically duplicating popular experts to more GPUs to process excessive
tokens, which requires predicting the distribution before routing. In this
paper, we discuss the tradeoff of prediction strategies, accuracies, overhead,
and end-to-end system performance. We propose MoE-GPS, a framework that guides
the selection of the optimal predictor design under various system
configurations, by quantifying the performance impact to system-level model
runtime. Specifically, we advocate for Distribution-Only Prediction, a
prediction strategy that only predicts overall token distribution which
significantly reduces overhead compared to the traditional Token-to-Expert
Prediction. On Mixtral 8x7B MMLU dataset, MoE-GPS suggests Distribution-Only
Prediction which improves end-to-end inference performance by more than 23%
compared with Token-to-Expert Prediction.

### 4. [Understanding the Error Sensitivity of Privacy-Aware Computing](http://arxiv.org/pdf/2506.07957v1)

Authors: Matías Mazzanti, Esteban Mocskos, Augusto Vega, Pradip Bose

Homomorphic Encryption (HE) enables secure computation on encrypted data
without decryption, allowing a great opportunity for privacy-preserving
computation. In particular, domains such as healthcare, finance, and
government, where data privacy and security are of utmost importance, can
benefit from HE by enabling third-party computation and services on sensitive
data. In other words, HE constitutes the "Holy Grail" of cryptography: data
remains encrypted all the time, being protected while in use.
  HE's security guarantees rely on noise added to data to make relatively
simple problems computationally intractable. This error-centric intrinsic HE
mechanism generates new challenges related to the fault tolerance and
robustness of HE itself: hardware- and software-induced errors during HE
operation can easily evade traditional error detection and correction
mechanisms, resulting in silent data corruption (SDC).
  In this work, we motivate a thorough discussion regarding the sensitivity of
HE applications to bit faults and provide a detailed error characterization
study of CKKS (Cheon-Kim-Kim-Song). This is one of the most popular HE schemes
due to its fixed-point arithmetic support for AI and machine learning
applications. We also delve into the impact of the residue number system (RNS)
and the number theoretic transform (NTT), two widely adopted HE optimization
techniques, on CKKS' error sensitivity. To the best of our knowledge, this is
the first work that looks into the robustness and error sensitivity of
homomorphic encryption and, as such, it can pave the way for critical future
work in this area.

### 5. [ProtocolLLM: RTL Benchmark for SystemVerilog Generation of Communication Protocols](http://arxiv.org/pdf/2506.07945v1)

Authors: Arnav Sheth, Ivaxi Sheth, Mario Fritz

Recent advances in Large Language Models (LLMs) have shown promising
capabilities in generating code for general-purpose programming languages. In
contrast, their applicability for hardware description languages, particularly
for generating synthesizable and functionally correct designs, remains
significantly underexplored. HDLs such as SystemVerilog are logic-oriented and
demand strict adherence to timing semantics, concurrency, and synthesizability
constraints. Moreover, HDL-based design flows encompass a broad set of tasks
beyond structural code generation, including testbench development,
assertion-based verification, timing closure, and protocol-level integration
for on-chip communication. The objective of our paper is to analyze the
capabilities of state-of-the-art LLMs in generating SystemVerilog
implementations of standard communication protocols, a core component of
embedded and System-on-Chip (SoC) architectures. This paper introduces the
first benchmark suite targeting four widely used protocols: SPI, I2C, UART, and
AXI. We define code generation tasks that capture varying levels of design
abstraction and prompt specificity. The generated designs are assessed for
syntactic correctness, synthesizability, and functional fidelity via waveform
simulation and test benches.

### Computational Complexity

### 1. [New Limits on Distributed Quantum Advantage: Dequantizing Linear Programs](http://arxiv.org/pdf/2506.07574v1)

Authors: Alkida Balliu, Corinna Coupette, Antonio Cruciani, Francesco d'Amore, Massimo Equi, Henrik Lievonen, Augusto Modanese, Dennis Olivetti, Jukka Suomela

In this work, we give two results that put new limits on distributed quantum
advantage in the context of the LOCAL model of distributed computing. First, we
show that there is no distributed quantum advantage for any linear program. Put
otherwise, if there is a quantum-LOCAL algorithm $\mathcal{A}$ that finds an
$\alpha$-approximation of some linear optimization problem $\Pi$ in $T$
communication rounds, we can construct a classical, deterministic LOCAL
algorithm $\mathcal{A}'$ that finds an $\alpha$-approximation of $\Pi$ in $T$
rounds. As a corollary, all classical lower bounds for linear programs,
including the KMW bound, hold verbatim in quantum-LOCAL. Second, using the
above result, we show that there exists a locally checkable labeling problem
(LCL) for which quantum-LOCAL is strictly weaker than the classical
deterministic SLOCAL model. Our results extend from quantum-LOCAL also to
finitely dependent and non-signaling distributions, and one of the corollaries
of our work is that the non-signaling model and the SLOCAL model are
incomparable in the context of LCL problems: By prior work, there exists an LCL
problem for which SLOCAL is strictly weaker than the non-signaling model, and
our work provides a separation in the opposite direction.

### 2. [Refuting Perfect Matchings in Spectral Expanders is Hard](http://arxiv.org/pdf/2506.07700v1)

Authors: Ari Biswas, Rajko Nenadov

This work studies the complexity of refuting the existence of a perfect
matching in spectral expanders with an odd number of vertices, in the
Polynomial Calculus (PC) and Sum of Squares (SoS) proof system. Austrin and
Risse [SODA, 2021] showed that refuting perfect matchings in sparse $d$-regular
\emph{random} graphs, in the above proof systems, with high probability
requires proofs with degree $\Omega(n/\log n)$. We extend their result by
showing the same lower bound holds for \emph{all} $d$-regular graphs with a
mild spectral gap.

### Computational Engineering

### 1. [ChemAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning](http://arxiv.org/pdf/2506.07551v1)

Authors: Mengsong Wu, YaFei Wang, Yidong Ming, Yuqi An, Yuwei Wan, Wenliang Chen, Binbin Lin, Yuqiang Li, Tong Xie, Dongzhan Zhou

Large language models (LLMs) have recently demonstrated promising
capabilities in chemistry tasks while still facing challenges due to outdated
pretraining knowledge and the difficulty of incorporating specialized chemical
expertise. To address these issues, we propose an LLM-based agent that
synergistically integrates 137 external chemical tools created ranging from
basic information retrieval to complex reaction predictions, and a dataset
curation pipeline to generate the dataset ChemToolBench that facilitates both
effective tool selection and precise parameter filling during fine-tuning and
evaluation. We introduce a Hierarchical Evolutionary Monte Carlo Tree Search
(HE-MCTS) framework, enabling independent optimization of tool planning and
execution. By leveraging self-generated data, our approach supports step-level
fine-tuning (FT) of the policy model and training task-adaptive PRM and ORM
that surpass GPT-4o. Experimental evaluations demonstrate that our approach
significantly improves performance in Chemistry QA and discovery tasks,
offering a robust solution to integrate specialized tools with LLMs for
advanced chemical applications. All datasets and code are available at
https://github.com/AI4Chem/ChemistryAgent .

### 2. [FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](http://arxiv.org/pdf/2506.07865v1)

Authors: Jinxi Li, Ziyang Song, Siyuan Zhou, Bo Yang

In this paper, we aim to model 3D scene geometry, appearance, and the
underlying physics purely from multi-view videos. By applying various governing
PDEs as PINN losses or incorporating physics simulation into neural networks,
existing works often fail to learn complex physical motions at boundaries or
require object priors such as masks or types. In this paper, we propose
FreeGave to learn the physics of complex dynamic 3D scenes without needing any
object priors. The key to our approach is to introduce a physics code followed
by a carefully designed divergence-free module for estimating a per-Gaussian
velocity field, without relying on the inefficient PINN losses. Extensive
experiments on three public datasets and a newly collected challenging
real-world dataset demonstrate the superior performance of our method for
future frame extrapolation and motion segmentation. Most notably, our
investigation into the learned physics codes reveals that they truly learn
meaningful 3D physical motion patterns in the absence of any human labels in
training.

### Computational Geometry

### 1. [An $O(n\log n)$ Algorithm for Single-Source Shortest Paths in Disk Graphs](http://arxiv.org/pdf/2506.07571v1)

Authors: Mark de Berg, Sergio Cabello

We prove that the single-source shortest-path problem on disk graphs can be
solved in $O(n\log n)$ time, and that it can be solved on intersection graphs
of fat triangles in $O(n\log^2 n)$ time.

### Computation and Language

### 1. [Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation](http://arxiv.org/pdf/2506.07356v1)

Authors: Seokil Ham, Yubin Choi, Seungju Cho, Yujin Yang, Younghun Kim, Changick Kim

Recently, major AI service providers such as Google and OpenAI have
introduced Finetuning-as-a-Service, which enables users to customize Large
Language Models (LLMs) for specific downstream tasks using their own data.
However, this service is vulnerable to degradation of LLM safety-alignment when
user data contains harmful prompts. While some prior works address this issue,
fundamentally filtering harmful data from user data remains unexplored.
Motivated by our observation that a directional representation reflecting
refusal behavior (called the refusal feature) obtained from safety-aligned LLMs
can inherently distinguish between harmful and harmless prompts, we propose the
Refusal-Feature-guided Teacher (ReFT). Our ReFT model is trained to identify
harmful prompts based on the similarity between input prompt features and its
refusal feature. During finetuning, the ReFT model serves as a teacher that
filters harmful prompts from user data and distills alignment knowledge into
the base model. Extensive experiments demonstrate that our ReFT-based
finetuning strategy effectively minimizes harmful outputs and enhances
finetuning accuracy for user-specific tasks, offering a practical solution for
secure and reliable deployment of LLMs in Finetuning-as-a-Service.

### 2. [SEED: Enhancing Text-to-SQL Performance and Practical Usability Through Automatic Evidence Generation](http://arxiv.org/pdf/2506.07423v1)

Authors: Janghyeon Yun, Sang-goo Lee

Text-to-SQL enables non-experts to retrieve data from databases by converting
natural language queries into SQL. However, state-of-the-art text-to-SQL
studies rely on the BIRD dataset, which assumes that evidence is provided along
with questions. Although BIRD facilitates research advancements, it assumes
that users have expertise and domain knowledge, contradicting the fundamental
goal of text-to-SQL. In addition, human-generated evidence in BIRD contains
defects, including missing or erroneous evidence, which affects model
performance. To address this issue, we propose SEED (System for Evidence
Extraction and Domain knowledge generation), an approach that automatically
generates evidence to improve performance and practical usability in real-world
scenarios. SEED systematically analyzes database schema, description files, and
values to extract relevant information. We evaluated SEED on BIRD and Spider,
demonstrating that it significantly improves SQL generation accuracy in the
no-evidence scenario, and in some cases, even outperforms the setting where
BIRD evidence is provided. Our results highlight that SEED-generated evidence
not only bridges the gap between research and real-world deployment but also
improves the adaptability and robustness of text-to-SQL models. Our code is
available at https://github.com/felix01189/SEED

### 3. [Conjoined Predication and Scalar Implicature](http://arxiv.org/pdf/2506.07429v1)

Authors: Ratna Kandala

Magri (2016) investigates two puzzles arising from conjunction. Although
Magri has proposed a solution to the second puzzle, the first remains
unresolved. This first puzzle reveals a hidden interaction among
quantification, collective/concurrent interpretation, and contextual updating
dimensions that have yet to be explored. In essence, the problem is that
certain forms of sentences like "Some Italians come from a warm country," when
conjoined as in "(Only) Some Italians come from a warm country and are blond,"
sound infelicitous, even though no obvious alternative triggers a conflicting
scalar implicature. In this paper, we offer a conceptual analysis of Magri's
first puzzle by situating it within its original theoretical framework. We
argue that the oddness arises from the collective or concurrent reading of the
conjunctive predicate: in examples such as "(Only) Some Italians come from a
warm country and are blond," this interpretation generates an indirect
contextual contradiction. Moreover, we suggest that the pragmatic mechanisms
governing scalar implicature generation extend beyond what is captured by
exhaustification-based grammatical licensing accounts.

### 4. [LG-ANNA-Embedding technical report](http://arxiv.org/pdf/2506.07438v1)

Authors: Jooyoung Choi, Hyun Kim, Hansol Jang, Changwook Jun, Kyunghoon Bae, Hyewon Choi, Stanley Jungkyu Choi, Honglak Lee, Chulmin Yun

This report presents a unified instruction-based framework for learning
generalized text embeddings optimized for both information retrieval (IR) and
non-IR tasks. Built upon a decoder-only large language model (Mistral-7B), our
approach combines in-context learning, soft supervision, and adaptive
hard-negative mining to generate context-aware embeddings without task-specific
fine-tuning. Structured instructions and few-shot examples are used to guide
the model across diverse tasks, enabling strong performance on classification,
semantic similarity, clustering, and reranking benchmarks. To improve semantic
discrimination, we employ a soft labeling framework where continuous relevance
scores, distilled from a high-performance dense retriever and reranker, serve
as fine-grained supervision signals. In addition, we introduce adaptive
margin-based hard-negative mining, which filters out semantically ambiguous
negatives based on their similarity to positive examples, thereby enhancing
training stability and retrieval robustness. Our model is evaluated on the
newly introduced MTEB (English, v2) benchmark, covering 41 tasks across seven
categories. Results show that our method achieves strong generalization and
ranks among the top-performing models by Borda score, outperforming several
larger or fully fine-tuned baselines. These findings highlight the
effectiveness of combining in-context prompting, soft supervision, and adaptive
sampling for scalable, high-quality embedding generation.

### 5. [Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](http://arxiv.org/pdf/2506.07453v1)

Authors: Pritom Saha Akash, Kevin Chen-Chuan Chang

Topic modeling plays a vital role in uncovering hidden semantic structures
within text corpora, but existing models struggle in low-resource settings
where limited target-domain data leads to unstable and incoherent topic
inference. We address this challenge by formally introducing domain adaptation
for low-resource topic modeling, where a high-resource source domain informs a
low-resource target domain without overwhelming it with irrelevant content. We
establish a finite-sample generalization bound showing that effective knowledge
transfer depends on robust performance in both domains, minimizing latent-space
discrepancy, and preventing overfitting to the data. Guided by these insights,
we propose DALTA (Domain-Aligned Latent Topic Adaptation), a new framework that
employs a shared encoder for domain-invariant features, specialized decoders
for domain-specific nuances, and adversarial alignment to selectively transfer
relevant information. Experiments on diverse low-resource datasets demonstrate
that DALTA consistently outperforms state-of-the-art methods in terms of topic
coherence, stability, and transferability.

### 6. [From Calibration to Collaboration: LLM Uncertainty Quantification Should Be More Human-Centered](http://arxiv.org/pdf/2506.07461v1)

Authors: Siddartha Devic, Tejas Srinivasan, Jesse Thomason, Willie Neiswanger, Vatsal Sharan

Large Language Models (LLMs) are increasingly assisting users in the real
world, yet their reliability remains a concern. Uncertainty quantification (UQ)
has been heralded as a tool to enhance human-LLM collaboration by enabling
users to know when to trust LLM predictions. We argue that current practices
for uncertainty quantification in LLMs are not optimal for developing useful UQ
for human users making decisions in real-world tasks. Through an analysis of 40
LLM UQ methods, we identify three prevalent practices hindering the community's
progress toward its goal of benefiting downstream users: 1) evaluating on
benchmarks with low ecological validity; 2) considering only epistemic
uncertainty; and 3) optimizing metrics that are not necessarily indicative of
downstream utility. For each issue, we propose concrete user-centric practices
and research directions that LLM UQ researchers should consider. Instead of
hill-climbing on unrepresentative tasks using imperfect metrics, we argue that
the community should adopt a more human-centered approach to LLM uncertainty
quantification.

### 7. [Improving Fairness of Large Language Models in Multi-document Summarization](http://arxiv.org/pdf/2506.07479v1)

Authors: Haoyuan Li Yusen Zhang, Snigdha Chaturvedi

Fairness in multi-document summarization (MDS) is crucial for providing
comprehensive views across documents with diverse social attribute values,
which can significantly impact decision-making. For example, a summarization
system that tends to overrepresent negative reviews of products can mislead
customers into disregarding good products. Previous works measure fairness in
MDS at two levels: summary-level and corpus-level. While summary-level fairness
focuses on individual summaries, corpus-level fairness focuses on a corpus of
summaries. Recent methods primarily focus on summary-level fairness. We propose
FairPO, a preference tuning method that focuses on both summary-level and
corpus-level fairness in MDS. To improve summary-level fairness, we propose to
generate preference pairs by perturbing document sets. To improve corpus-level
fairness, we propose fairness-aware preference tuning by dynamically adjusting
the weights of preference pairs. Our experiments show that FairPO outperforms
strong baselines while maintaining the critical qualities of summaries. The
code is available at https://github.com/leehaoyuan/coverage_fairnes.

### 8. [A Hybrid GA LLM Framework for Structured Task Optimization](http://arxiv.org/pdf/2506.07483v1)

Authors: Berry Feng, Jonas Lin, Patrick Lau

GA LLM is a hybrid framework that combines Genetic Algorithms with Large
Language Models to handle structured generation tasks under strict constraints.
Each output, such as a plan or report, is treated as a gene, and evolutionary
operations like selection, crossover, and mutation are guided by the language
model to iteratively improve solutions. The language model provides domain
knowledge and creative variation, while the genetic algorithm ensures
structural integrity and global optimization. GA LLM has proven effective in
tasks such as itinerary planning, academic outlining, and business reporting,
consistently producing well structured and requirement satisfying results. Its
modular design also makes it easy to adapt to new tasks. Compared to using a
language model alone, GA LLM achieves better constraint satisfaction and higher
quality solutions by combining the strengths of both components.

### 9. [DEBATE: A Dataset for Disentangling Textual Ambiguity in Mandarin Through Speech](http://arxiv.org/pdf/2506.07502v1)

Authors: Haotian Guo, Jing Han, Yongfeng Tu, Shihao Gao, Shengfan Shen, Wulong Xiang, Weihao Gan, Zixing Zhang

Despite extensive research on textual and visual disambiguation,
disambiguation through speech (DTS) remains underexplored. This is largely due
to the lack of high-quality datasets that pair spoken sentences with richly
ambiguous text. To address this gap, we present DEBATE, a unique public Chinese
speech-text dataset designed to study how speech cues and
patterns-pronunciation, pause, stress and intonation-can help resolve textual
ambiguity and reveal a speaker's true intent. DEBATE contains 1,001 carefully
selected ambiguous utterances, each recorded by 10 native speakers, capturing
diverse linguistic ambiguities and their disambiguation through speech. We
detail the data collection pipeline and provide rigorous quality analysis.
Additionally, we benchmark three state-of-the-art large speech and
audio-language models, illustrating clear and huge performance gaps between
machine and human understanding of spoken intent. DEBATE represents the first
effort of its kind and offers a foundation for building similar DTS datasets
across languages and cultures. The dataset and associated code are available
at: https://github.com/SmileHnu/DEBATE.

### 10. [What Do Indonesians Really Need from Language Technology? A Nationwide Survey](http://arxiv.org/pdf/2506.07506v1)

Authors: Muhammad Dehan Al Kautsar, Lucky Susanto, Derry Wijaya, Fajri Koto

There is an emerging effort to develop NLP for Indonesias 700+ local
languages, but progress remains costly due to the need for direct engagement
with native speakers. However, it is unclear what these language communities
truly need from language technology. To address this, we conduct a nationwide
survey to assess the actual needs of native speakers in Indonesia. Our findings
indicate that addressing language barriers, particularly through machine
translation and information retrieval, is the most critical priority. Although
there is strong enthusiasm for advancements in language technology, concerns
around privacy, bias, and the use of public data for AI training highlight the
need for greater transparency and clear communication to support broader AI
adoption.

### Cryptography and Security

### 1. [Enhanced Consistency Bi-directional GAN(CBiGAN) for Malware Anomaly Detection](http://arxiv.org/pdf/2506.07372v1)

Authors: Thesath Wijayasiri, Kar Wai Fok, Vrizlynn L. L. Thing

Static analysis, a cornerstone technique in cybersecurity, offers a
noninvasive method for detecting malware by analyzing dormant software without
executing potentially harmful code. However, traditional static analysis often
relies on biased or outdated datasets, leading to gaps in detection
capabilities against emerging malware threats. To address this, our study
focuses on the binary content of files as key features for malware detection.
These binary contents are transformed and represented as images, which then
serve as inputs to deep learning models. This method takes into account the
visual patterns within the binary data, allowing the model to analyze potential
malware effectively. This paper introduces the application of the CBiGAN in the
domain of malware anomaly detection. Our approach leverages the CBiGAN for its
superior latent space mapping capabilities, critical for modeling complex
malware patterns by utilizing a reconstruction error-based anomaly detection
method. We utilized several datasets including both portable executable (PE)
files as well as Object Linking and Embedding (OLE) files. We then evaluated
our model against a diverse set of both PE and OLE files, including
self-collected malicious executables from 214 malware families. Our findings
demonstrate the robustness of this innovative approach, with the CBiGAN
achieving high Area Under the Curve (AUC) results with good generalizability,
thereby confirming its capability to distinguish between benign and diverse
malicious files with reasonably high accuracy.

### 2. [Enhancing Watermarking Quality for LLMs via Contextual Generation States Awareness](http://arxiv.org/pdf/2506.07403v1)

Authors: Peiru Yang, Xintian Li, Wanchun Ni, Jinhua Yin, Huili Wang, Guoshun Nan, Shangguang Wang, Yongfeng Huang, Tao Qi

Recent advancements in watermarking techniques have enabled the embedding of
secret messages into AI-generated text (AIGT), serving as an important
mechanism for AIGT detection. Existing methods typically interfere with the
generation processes of large language models (LLMs) to embed signals within
the generated text. However, these methods often rely on heuristic rules, which
can result in suboptimal token selection and a subsequent decline in the
quality of the generated content. In this paper, we introduce a plug-and-play
contextual generation states-aware watermarking framework (CAW) that
dynamically adjusts the embedding process. It can be seamlessly integrated with
various existing watermarking methods to enhance generation quality. First, CAW
incorporates a watermarking capacity evaluator, which can assess the impact of
embedding messages at different token positions by analyzing the contextual
generation states. Furthermore, we introduce a multi-branch pre-generation
mechanism to avoid the latency caused by the proposed watermarking strategy.
Building on this, CAW can dynamically adjust the watermarking process based on
the evaluated watermark capacity of each token, thereby minimizing potential
degradation in content quality. Extensive experiments conducted on datasets
across multiple domains have verified the effectiveness of our method,
demonstrating superior performance compared to various baselines in terms of
both detection rate and generation quality.

### 3. [Explainable AI for Enhancing IDS Against Advanced Persistent Kill Chain](http://arxiv.org/pdf/2506.07480v1)

Authors: Bassam Noori Shaker, Bahaa Al-Musawi, Mohammed Falih Hassan

Advanced Persistent Threats (APTs) represent a sophisticated and persistent
cy-bersecurity challenge, characterized by stealthy, multi-phase, and targeted
attacks aimed at compromising information systems over an extended period.
Develop-ing an effective Intrusion Detection System (IDS) capable of detecting
APTs at different phases relies on selecting network traffic features. However,
not all of these features are directly related to the phases of APTs. Some
network traffic features may be unrelated or have limited relevance to
identifying malicious ac-tivity. Therefore, it is important to carefully select
and analyze the most relevant features to improve the IDS performance. This
work proposes a feature selection and classification model that integrates two
prominent machine learning algo-rithms: SHapley Additive exPlanations (SHAP)
and Extreme Gradient Boosting (XGBoost). The aim is to develop lightweight IDS
based on a selected minimum number of influential features for detecting APTs
at various phases. The pro-posed method also specifies the relevant features
for each phase of APTs inde-pendently. Extensive experimental results on the
SCVIC-APT-2021 dataset indi-cated that our proposed approach has improved
performance compared to other standard techniques. Specifically, both the
macro-average F1-score and recall reached 94% and 93 %, respectively, while
reducing the complexity of the detec-tion model by selecting only 12 features
out of 77.

### 4. [MalGEN: A Generative Agent Framework for Modeling Malicious Software in Cybersecurity](http://arxiv.org/pdf/2506.07586v1)

Authors: Bikash Saha, Sandeep Kumar Shukla

The dual use nature of Large Language Models (LLMs) presents a growing
challenge in cybersecurity. While LLM enhances automation and reasoning for
defenders, they also introduce new risks, particularly their potential to be
misused for generating evasive, AI crafted malware. Despite this emerging
threat, the research community currently lacks controlled and extensible tools
that can simulate such behavior for testing and defense preparation. We present
MalGEN, a multi agent framework that simulates coordinated adversarial behavior
to generate diverse, activity driven malware samples. The agents work
collaboratively to emulate attacker workflows, including payload planning,
capability selection, and evasion strategies, within a controlled environment
built for ethical and defensive research. Using MalGEN, we synthesized ten
novel malware samples and evaluated them against leading antivirus and
behavioral detection engines. Several samples exhibited stealthy and evasive
characteristics that bypassed current defenses, validating MalGEN's ability to
model sophisticated and new threats. By transforming the threat of LLM misuse
into an opportunity for proactive defense, MalGEN offers a valuable framework
for evaluating and strengthening cybersecurity systems. The framework addresses
data scarcity, enables rigorous testing, and supports the development of
resilient and future ready detection strategies.

### 5. ["I wasn't sure if this is indeed a security risk": Data-driven Understanding of Security Issue Reporting in GitHub Repositories of Open Source npm Packages](http://arxiv.org/pdf/2506.07728v1)

Authors: Rajdeep Ghosh, Shiladitya De, Mainack Mondal

The npm (Node Package Manager) ecosystem is the most important package
manager for JavaScript development with millions of users. Consequently, a
plethora of earlier work investigated how vulnerability reporting, patch
propagation, and in general detection as well as resolution of security issues
in such ecosystems can be facilitated. However, understanding the ground
reality of security-related issue reporting by users (and bots) in npm-along
with the associated challenges has been relatively less explored at scale.
  In this work, we bridge this gap by collecting 10,907,467 issues reported
across GitHub repositories of 45,466 diverse npm packages. We found that the
tags associated with these issues indicate the existence of only 0.13%
security-related issues. However, our approach of manual analysis followed by
developing high accuracy machine learning models identify 1,617,738
security-related issues which are not tagged as security-related (14.8% of all
issues) as well as 4,461,934 comments made on these issues. We found that the
bots which are in wide use today might not be sufficient for either detecting
or offering assistance. Furthermore, our analysis of user-developer interaction
data hints that many user-reported security issues might not be addressed by
developers-they are not tagged as security-related issues and might be closed
without valid justification. Consequently, a correlation analysis hints that
the developers quickly handle security issues with known solutions (e.g.,
corresponding to CVE). However, security issues without such known solutions
(even with reproducible code) might not be resolved. Our findings offer
actionable insights for improving security management in open-source
ecosystems, highlighting the need for smarter tools and better collaboration.
The data and code for this work is available at
https://doi.org/10.5281/zenodo.15614029

### 6. [User-space library rootkits revisited: Are user-space detection mechanisms futile?](http://arxiv.org/pdf/2506.07827v1)

Authors: Enrique Soriano-Salvador, Gorka Guardiola Múzquiz, Juan González Gómez

The kind of malware designed to conceal malicious system resources (e.g.
processes, network connections, files, etc.) is commonly referred to as a
rootkit. This kind of malware represents a significant threat in contemporany
systems. Despite the existence of kernel-space rootkits (i.e. rootkits that
infect the operating system kernel), user-space rootkits (i.e. rootkits that
infect the user-space operating system tools, commands and libraries) continue
to pose a significant danger. However, kernel-space rootkits attract all the
attention, implicitly assuming that user-space rootkits (malware that is still
in existence) are easily detectable by well-known user-space tools that look
for anomalies. The primary objective of this work is to answer the following
question: Is detecting user-space rootkits with user-space tools futile?
Contrary to the prevailing view that considers it effective, we argue that the
detection of user-space rootkits cannot be done in user-space at all. Moreover,
the detection results must be communicated to the user with extreme caution. To
support this claim, we conducted different experiments focusing on process
concealing in Linux systems. In these experiments, we evade the detection
mechanisms widely accepted as the standard solution for this type of user-space
malware, bypassing the most popular open source anti-rootkit tool for process
hiding. This manuscript describes the classical approach to build user-space
library rootkits, the traditional detection mechanisms, and different evasion
techniques (it also includes understandable code snippets and examples). In
addition, it offers some guidelines to implement new detection tools and
improve the existing ones to the extent possible.

### 7. [Securing Unbounded Differential Privacy Against Timing Attacks](http://arxiv.org/pdf/2506.07868v1)

Authors: Zachary Ratliff, Salil Vadhan

Recent works have started to theoretically investigate how we can protect
differentially private programs against timing attacks, by making the joint
distribution the output and the runtime differentially private (JOT-DP).
However, the existing approaches to JOT-DP have some limitations, particularly
in the setting of unbounded DP (which protects the size of the dataset and
applies to arbitrarily large datasets). First, the known conversion of pure DP
programs to pure JOT-DP programs in the unbounded setting (a) incurs a constant
additive increase in error probability (and thus does not provide vanishing
error as $n\to\infty$) (b) produces JOT-DP programs that fail to preserve the
computational efficiency of the original pure DP program and (c) is analyzed in
a toy computational model in which the runtime is defined to be the number of
coin flips. In this work, we overcome these limitations. Specifically, we show
that the error required for pure JOT-DP in the unbounded setting depends on the
model of computation. In a randomized RAM model where the dataset size $n$ is
given (or can be computed in constant time) and we can generate random numbers
(not just random bits) in constant time, polynomially small error probability
is necessary and sufficient. If $n$ is not given or we only have a random-bit
generator, an (arbitrarily small) constant error probability is necessary and
sufficient. The aforementioned positive results are proven by efficient
procedures to convert any pure JOT-DP program $P$ in the upper-bounded setting
to a pure JOT-DP program $P'$ in the unbounded setting, such that the output
distribution of $P'$ is $\gamma$-close in total variation distance to that of
$P$, where $\gamma$ is either an arbitrarily small constant or polynomially
small, depending on the model of computation.

### 8. [Evaluating explainable AI for deep learning-based network intrusion detection system alert classification](http://arxiv.org/pdf/2506.07882v1)

Authors: Rajesh Kalakoti, Risto Vaarandi, Hayretdin Bahsi, Sven Nõmm

A Network Intrusion Detection System (NIDS) monitors networks for cyber
attacks and other unwanted activities. However, NIDS solutions often generate
an overwhelming number of alerts daily, making it challenging for analysts to
prioritize high-priority threats. While deep learning models promise to
automate the prioritization of NIDS alerts, the lack of transparency in these
models can undermine trust in their decision-making. This study highlights the
critical need for explainable artificial intelligence (XAI) in NIDS alert
classification to improve trust and interpretability. We employed a real-world
NIDS alert dataset from Security Operations Center (SOC) of TalTech (Tallinn
University Of Technology) in Estonia, developing a Long Short-Term Memory
(LSTM) model to prioritize alerts. To explain the LSTM model's alert
prioritization decisions, we implemented and compared four XAI methods: Local
Interpretable Model-Agnostic Explanations (LIME), SHapley Additive exPlanations
(SHAP), Integrated Gradients, and DeepLIFT. The quality of these XAI methods
was assessed using a comprehensive framework that evaluated faithfulness,
complexity, robustness, and reliability. Our results demonstrate that DeepLIFT
consistently outperformed the other XAI methods, providing explanations with
high faithfulness, low complexity, robust performance, and strong reliability.
In collaboration with SOC analysts, we identified key features essential for
effective alert classification. The strong alignment between these
analyst-identified features and those obtained by the XAI methods validates
their effectiveness and enhances the practical applicability of our approach.

### 9. [Secure Distributed Learning for CAVs: Defending Against Gradient Leakage with Leveled Homomorphic Encryption](http://arxiv.org/pdf/2506.07894v1)

Authors: Muhammad Ali Najjar, Ren-Yi Huang, Dumindu Samaraweera, Prashant Shekhar

Federated Learning (FL) enables collaborative model training across
distributed clients without sharing raw data, making it a promising approach
for privacy-preserving machine learning in domains like Connected and
Autonomous Vehicles (CAVs). However, recent studies have shown that exchanged
model gradients remain susceptible to inference attacks such as Deep Leakage
from Gradients (DLG), which can reconstruct private training data. While
existing defenses like Differential Privacy (DP) and Secure Multi-Party
Computation (SMPC) offer protection, they often compromise model accuracy. To
that end, Homomorphic Encryption (HE) offers a promising alternative by
enabling lossless computation directly on encrypted data, thereby preserving
both privacy and model utility. However, HE introduces significant
computational and communication overhead, which can hinder its practical
adoption. To address this, we systematically evaluate various leveled HE
schemes to identify the most suitable for FL in resource-constrained
environments due to its ability to support fixed-depth computations without
requiring costly bootstrapping. Our contributions in this paper include a
comprehensive evaluation of HE schemes for real-world FL applications, a
selective encryption strategy that targets only the most sensitive gradients to
minimize computational overhead, and the development of a full HE-based FL
pipeline that effectively mitigates DLG attacks while preserving model
accuracy. We open-source our implementation to encourage reproducibility and
facilitate adoption in safety-critical domains.

### 10. [Exposing Hidden Backdoors in NFT Smart Contracts: A Static Security Analysis of Rug Pull Patterns](http://arxiv.org/pdf/2506.07974v1)

Authors: Chetan Pathade, Shweta Hooli

The explosive growth of Non-Fungible Tokens (NFTs) has revolutionized digital
ownership by enabling the creation, exchange, and monetization of unique assets
on blockchain networks. However, this surge in popularity has also given rise
to a disturbing trend: the emergence of rug pulls - fraudulent schemes where
developers exploit trust and smart contract privileges to drain user funds or
invalidate asset ownership. Central to many of these scams are hidden backdoors
embedded within NFT smart contracts. Unlike unintentional bugs, these backdoors
are deliberately coded and often obfuscated to bypass traditional audits and
exploit investor confidence. In this paper, we present a large-scale static
analysis of 49,940 verified NFT smart contracts using Slither, a static
analysis framework, to uncover latent vulnerabilities commonly linked to rug
pulls. We introduce a custom risk scoring model that classifies contracts into
high, medium, or low risk tiers based on the presence and severity of rug pull
indicators. Our dataset was derived from verified contracts on the Ethereum
mainnet, and we generate multiple visualizations to highlight red flag
clusters, issue prevalence, and co-occurrence of critical vulnerabilities.
While we do not perform live exploits, our results reveal how malicious
patterns often missed by simple reviews can be surfaced through static analysis
at scale. We conclude by offering mitigation strategies for developers,
marketplaces, and auditors to enhance smart contract security. By exposing how
hidden backdoors manifest in real-world smart contracts, this work contributes
a practical foundation for detecting and mitigating NFT rug pulls through
scalable automated analysis.

### Computer Vision and Pattern Recognition

### 1. [Generative Models at the Frontier of Compression: A Survey on Generative Face Video Coding](http://arxiv.org/pdf/2506.07369v1)

Authors: Bolin Chen, Shanzhi Yin, Goluck Konuko, Giuseppe Valenzise, Zihan Zhang, Shiqi Wang, Yan Ye

The rise of deep generative models has greatly advanced video compression,
reshaping the paradigm of face video coding through their powerful capability
for semantic-aware representation and lifelike synthesis. Generative Face Video
Coding (GFVC) stands at the forefront of this revolution, which could
characterize complex facial dynamics into compact latent codes for bitstream
compactness at the encoder side and leverages powerful deep generative models
to reconstruct high-fidelity face signal from the compressed latent codes at
the decoder side. As such, this well-designed GFVC paradigm could enable
high-fidelity face video communication at ultra-low bitrate ranges, far
surpassing the capabilities of the latest Versatile Video Coding (VVC)
standard. To pioneer foundational research and accelerate the evolution of
GFVC, this paper presents the first comprehensive survey of GFVC technologies,
systematically bridging critical gaps between theoretical innovation and
industrial standardization. In particular, we first review a broad range of
existing GFVC methods with different feature representations and optimization
strategies, and conduct a thorough benchmarking analysis. In addition, we
construct a large-scale GFVC-compressed face video database with subjective
Mean Opinion Scores (MOSs) based on human perception, aiming to identify the
most appropriate quality metrics tailored to GFVC. Moreover, we summarize the
GFVC standardization potentials with a unified high-level syntax and develop a
low-complexity GFVC system which are both expected to push forward future
practical deployments and applications. Finally, we envision the potential of
GFVC in industrial applications and deliberate on the current challenges and
future opportunities.

### 2. [ARGUS: Hallucination and Omission Evaluation in Video-LLMs](http://arxiv.org/pdf/2506.07371v1)

Authors: Ruchit Rawal, Reza Shirkavand, Heng Huang, Gowthami Somepalli, Tom Goldstein

Video large language models have not yet been widely deployed, largely due to
their tendency to hallucinate. Typical benchmarks for Video-LLMs rely simply on
multiple-choice questions. Unfortunately, VideoLLMs hallucinate far more
aggressively on freeform text generation tasks like video captioning than they
do on multiple choice verification tasks. To address this weakness, we propose
ARGUS, a VideoLLM benchmark that measures freeform video captioning
performance. By comparing VideoLLM outputs to human ground truth captions,
ARGUS quantifies dual metrics. First, we measure the rate of hallucinations in
the form of incorrect statements about video content or temporal relationships.
Second, we measure the rate at which the model omits important descriptive
details. Together, these dual metrics form a comprehensive view of video
captioning performance.

### 3. [DINO-CoDT: Multi-class Collaborative Detection and Tracking with Vision Foundation Models](http://arxiv.org/pdf/2506.07375v1)

Authors: Xunjie He, Christina Dao Wen Lee, Meiling Wang, Chengran Yuan, Zefan Huang, Yufeng Yue, Marcelo H. Ang Jr

Collaborative perception plays a crucial role in enhancing environmental
understanding by expanding the perceptual range and improving robustness
against sensor failures, which primarily involves collaborative 3D detection
and tracking tasks. The former focuses on object recognition in individual
frames, while the latter captures continuous instance tracklets over time.
However, existing works in both areas predominantly focus on the vehicle
superclass, lacking effective solutions for both multi-class collaborative
detection and tracking. This limitation hinders their applicability in
real-world scenarios, which involve diverse object classes with varying
appearances and motion patterns. To overcome these limitations, we propose a
multi-class collaborative detection and tracking framework tailored for diverse
road users. We first present a detector with a global spatial attention fusion
(GSAF) module, enhancing multi-scale feature learning for objects of varying
sizes. Next, we introduce a tracklet RE-IDentification (REID) module that
leverages visual semantics with a vision foundation model to effectively reduce
ID SWitch (IDSW) errors, in cases of erroneous mismatches involving small
objects like pedestrians. We further design a velocity-based adaptive tracklet
management (VATM) module that adjusts the tracking interval dynamically based
on object motion. Extensive experiments on the V2X-Real and OPV2V datasets show
that our approach significantly outperforms existing state-of-the-art methods
in both detection and tracking accuracy.

### 4. [Compressed Feature Quality Assessment: Dataset and Baselines](http://arxiv.org/pdf/2506.07412v1)

Authors: Changsheng Gao, Wei Zhou, Guosheng Lin, Weisi Lin

The widespread deployment of large models in resource-constrained
environments has underscored the need for efficient transmission of
intermediate feature representations. In this context, feature coding, which
compresses features into compact bitstreams, becomes a critical component for
scenarios involving feature transmission, storage, and reuse. However, this
compression process introduces inherent semantic degradation that is
notoriously difficult to quantify with traditional metrics. To address this,
this paper introduces the research problem of Compressed Feature Quality
Assessment (CFQA), which seeks to evaluate the semantic fidelity of compressed
features. To advance CFQA research, we propose the first benchmark dataset,
comprising 300 original features and 12000 compressed features derived from
three vision tasks and four feature codecs. Task-specific performance drops are
provided as true semantic distortion for the evaluation of CFQA metrics. We
assess the performance of three widely used metrics (MSE, cosine similarity,
and Centered Kernel Alignment) in capturing semantic degradation. The results
underscore the representativeness of the dataset and highlight the need for
more refined metrics capable of addressing the nuances of semantic distortion
in compressed features. To facilitate the ongoing development of CFQA research,
we release the dataset and all accompanying source code at
\href{https://github.com/chansongoal/Compressed-Feature-Quality-Assessment}{https://github.com/chansongoal/Compressed-Feature-Quality-Assessment}.
This contribution aims to advance the field and provide a foundational resource
for the community to explore CFQA.

### 5. [DPFormer: Dynamic Prompt Transformer for Continual Learning](http://arxiv.org/pdf/2506.07414v1)

Authors: Sheng-Kai Huang, Jiun-Feng Chang, Chun-Rong Huang

In continual learning, solving the catastrophic forgetting problem may make
the models fall into the stability-plasticity dilemma. Moreover, inter-task
confusion will also occur due to the lack of knowledge exchanges between
different tasks. In order to solve the aforementioned problems, we propose a
novel dynamic prompt transformer (DPFormer) with prompt schemes. The prompt
schemes help the DPFormer memorize learned knowledge of previous classes and
tasks, and keep on learning new knowledge from new classes and tasks under a
single network structure with a nearly fixed number of model parameters.
Moreover, they also provide discrepant information to represent different tasks
to solve the inter-task confusion problem. Based on prompt schemes, a unified
classification module with the binary cross entropy loss, the knowledge
distillation loss and the auxiliary loss is proposed to train the whole model
in an end-to-end trainable manner. Compared with state-of-the-art methods, our
method achieves the best performance in the CIFAR-100, ImageNet100 and
ImageNet1K datasets under different class-incremental settings in continual
learning. The source code will be available at our GitHub after acceptance.

### 6. [PhysiInter: Integrating Physical Mapping for High-Fidelity Human Interaction Generation](http://arxiv.org/pdf/2506.07456v1)

Authors: Wei Yao, Yunlian Sun, Chang Liu, Hongwen Zhang, Jinhui Tang

Driven by advancements in motion capture and generative artificial
intelligence, leveraging large-scale MoCap datasets to train generative models
for synthesizing diverse, realistic human motions has become a promising
research direction. However, existing motion-capture techniques and generative
models often neglect physical constraints, leading to artifacts such as
interpenetration, sliding, and floating. These issues are exacerbated in
multi-person motion generation, where complex interactions are involved. To
address these limitations, we introduce physical mapping, integrated throughout
the human interaction generation pipeline. Specifically, motion imitation
within a physics-based simulation environment is used to project target motions
into a physically valid space. The resulting motions are adjusted to adhere to
real-world physics constraints while retaining their original semantic meaning.
This mapping not only improves MoCap data quality but also directly informs
post-processing of generated motions. Given the unique interactivity of
multi-person scenarios, we propose a tailored motion representation framework.
Motion Consistency (MC) and Marker-based Interaction (MI) loss functions are
introduced to improve model performance. Experiments show our method achieves
impressive results in generated human motion quality, with a 3%-89% improvement
in physical fidelity. Project page http://yw0208.github.io/physiinter

### 7. [Drive Any Mesh: 4D Latent Diffusion for Mesh Deformation from Video](http://arxiv.org/pdf/2506.07489v1)

Authors: Yahao Shi, Yang Liu, Yanmin Wu, Xing Liu, Chen Zhao, Jie Luo, Bin Zhou

We propose DriveAnyMesh, a method for driving mesh guided by monocular video.
Current 4D generation techniques encounter challenges with modern rendering
engines. Implicit methods have low rendering efficiency and are unfriendly to
rasterization-based engines, while skeletal methods demand significant manual
effort and lack cross-category generalization. Animating existing 3D assets,
instead of creating 4D assets from scratch, demands a deep understanding of the
input's 3D structure. To tackle these challenges, we present a 4D diffusion
model that denoises sequences of latent sets, which are then decoded to produce
mesh animations from point cloud trajectory sequences. These latent sets
leverage a transformer-based variational autoencoder, simultaneously capturing
3D shape and motion information. By employing a spatiotemporal,
transformer-based diffusion model, information is exchanged across multiple
latent frames, enhancing the efficiency and generalization of the generated
results. Our experimental results demonstrate that DriveAnyMesh can rapidly
produce high-quality animations for complex motions and is compatible with
modern rendering engines. This method holds potential for applications in both
the gaming and filming industries.

### 8. [SpatialLM: Training Large Language Models for Structured Indoor Modeling](http://arxiv.org/pdf/2506.07491v1)

Authors: Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, Zihan Zhou

SpatialLM is a large language model designed to process 3D point cloud data
and generate structured 3D scene understanding outputs. These outputs include
architectural elements like walls, doors, windows, and oriented object boxes
with their semantic categories. Unlike previous methods which exploit
task-specific network designs, our model adheres to the standard multimodal LLM
architecture and is fine-tuned directly from open-source LLMs.
  To train SpatialLM, we collect a large-scale, high-quality synthetic dataset
consisting of the point clouds of 12,328 indoor scenes (54,778 rooms) with
ground-truth 3D annotations, and conduct a careful study on various modeling
and training decisions. On public benchmarks, our model gives state-of-the-art
performance in layout estimation and competitive results in 3D object
detection. With that, we show a feasible path for enhancing the spatial
understanding capabilities of modern LLMs for applications in augmented
reality, embodied robotics, and more.

### 9. [Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency](http://arxiv.org/pdf/2506.07497v1)

Authors: Xiangyu Guo, Zhanqian Wu, Kaixin Xiong, Ziyang Xu, Lijun Zhou, Gangwei Xu, Shaoqing Xu, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wenyu Liu, Xinggang Wang

We present Genesis, a unified framework for joint generation of multi-view
driving videos and LiDAR sequences with spatio-temporal and cross-modal
consistency. Genesis employs a two-stage architecture that integrates a
DiT-based video diffusion model with 3D-VAE encoding, and a BEV-aware LiDAR
generator with NeRF-based rendering and adaptive sampling. Both modalities are
directly coupled through a shared latent space, enabling coherent evolution
across visual and geometric domains. To guide the generation with structured
semantics, we introduce DataCrafter, a captioning module built on
vision-language models that provides scene-level and instance-level
supervision. Extensive experiments on the nuScenes benchmark demonstrate that
Genesis achieves state-of-the-art performance across video and LiDAR metrics
(FVD 16.95, FID 4.24, Chamfer 0.611), and benefits downstream tasks including
segmentation and 3D detection, validating the semantic fidelity and practical
utility of the generated data.

### 10. [MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](http://arxiv.org/pdf/2506.07533v1)

Authors: Wei Tao, Haocheng Lu, Xiaoyang Qu, Bin Zhang, Kai Lu, Jiguang Wan, Jianzong Wang

One of the primary challenges in optimizing large language models (LLMs) for
long-context inference lies in the high memory consumption of the Key-Value
(KV) cache. Existing approaches, such as quantization, have demonstrated
promising results in reducing memory usage. However, current quantization
methods cannot take both effectiveness and efficiency into account. In this
paper, we propose MoQAE, a novel mixed-precision quantization method via
mixture of quantization-aware experts. First, we view different quantization
bit-width configurations as experts and use the traditional mixture of experts
(MoE) method to select the optimal configuration. To avoid the inefficiency
caused by inputting tokens one by one into the router in the traditional MoE
method, we input the tokens into the router chunk by chunk. Second, we design a
lightweight router-only fine-tuning process to train MoQAE with a comprehensive
loss to learn the trade-off between model accuracy and memory usage. Finally,
we introduce a routing freezing (RF) and a routing sharing (RS) mechanism to
further reduce the inference overhead. Extensive experiments on multiple
benchmark datasets demonstrate that our method outperforms state-of-the-art KV
cache quantization approaches in both efficiency and effectiveness.

### Computers and Society

### 1. [When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment](http://arxiv.org/pdf/2506.07452v1)

Authors: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi

Large language models (LLMs) can be prompted with specific styles (e.g.,
formatting responses as lists), including in jailbreak queries. Although these
style patterns are semantically unrelated to the malicious intents behind
jailbreak queries, their safety impact remains unclear. In this work, we seek
to understand whether style patterns compromise LLM safety, how superficial
style alignment increases model vulnerability, and how best to mitigate these
risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks,
and find that malicious queries with style patterns inflate the attack success
rate (ASR) for nearly all models. Notably, ASR inflation correlates with both
the length of style patterns and the relative attention an LLM exhibits on
them. We then investigate superficial style alignment, and find that
fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of
those same styles. Finally, we propose SafeStyle, a defense strategy that
incorporates a small amount of safety training data augmented to match the
distribution of style patterns in the fine-tuning data. Across three LLMs and
five fine-tuning style settings, SafeStyle consistently outperforms baselines
in maintaining LLM safety.

### 2. [Towards Energy-Efficient and Low-Latency Voice-Controlled Smart Homes: A Proposal for Offline Speech Recognition and IoT Integration](http://arxiv.org/pdf/2506.07494v1)

Authors: Peng Huang, Imdad Ullah, Xiaotong Wei, Tariq Ahamed Ahanger, Najm Hassan, Zawar Hussain Shah

The smart home systems, based on AI speech recognition and IoT technology,
enable people to control devices through verbal commands and make people's
lives more efficient. However, existing AI speech recognition services are
primarily deployed on cloud platforms on the Internet. When users issue a
command, speech recognition devices like ``Amazon Echo'' will post a recording
through numerous network nodes, reach multiple servers, and then receive
responses through the Internet. This mechanism presents several issues,
including unnecessary energy consumption, communication latency, and the risk
of a single-point failure. In this position paper, we propose a smart home
concept based on offline speech recognition and IoT technology: 1) integrating
offline keyword spotting (KWS) technologies into household appliances with
limited resource hardware to enable them to understand user voice commands; 2)
designing a local IoT network with decentralized architecture to manage and
connect various devices, enhancing the robustness and scalability of the
system. This proposal of a smart home based on offline speech recognition and
IoT technology will allow users to use low-latency voice control anywhere in
the home without depending on the Internet and provide better scalability and
energy sustainability.

### 3. [IntenTest: Stress Testing for Intent Integrity in API-Calling LLM Agents](http://arxiv.org/pdf/2506.07524v1)

Authors: Shiwei Feng, Xiangzhe Xu, Xuan Chen, Kaiyuan Zhang, Syed Yusuf Ahmed, Zian Su, Mingwei Zheng, Xiangyu Zhang

LLM agents are increasingly deployed to automate real-world tasks by invoking
APIs through natural language instructions. While powerful, they often suffer
from misinterpretation of user intent, leading to the agent's actions that
diverge from the user's intended goal, especially as external toolkits evolve.
Traditional software testing assumes structured inputs and thus falls short in
handling the ambiguity of natural language. We introduce IntenTest, an
API-centric stress testing framework that systematically uncovers intent
integrity violations in LLM agents. Unlike prior work focused on fixed
benchmarks or adversarial inputs, IntenTest generates realistic tasks based on
toolkits' documentation and applies targeted mutations to expose subtle agent
errors while preserving user intent. To guide testing, we propose semantic
partitioning, which organizes natural language tasks into meaningful categories
based on toolkit API parameters and their equivalence classes. Within each
partition, seed tasks are mutated and ranked by a lightweight predictor that
estimates the likelihood of triggering agent errors. To enhance efficiency,
IntenTest maintains a datatype-aware strategy memory that retrieves and adapts
effective mutation patterns from past cases. Experiments on 80 toolkit APIs
demonstrate that IntenTest effectively uncovers intent integrity violations,
significantly outperforming baselines in both error-exposing rate and query
efficiency. Moreover, IntenTest generalizes well to stronger target models
using smaller LLMs for test generation, and adapts to evolving APIs across
domains.

### 4. [Correlated Errors in Large Language Models](http://arxiv.org/pdf/2506.07962v1)

Authors: Elliot Kim, Avi Garg, Kenny Peng, Nikhil Garg

Diversity in training data, architecture, and providers is assumed to
mitigate homogeneity in LLMs. However, we lack empirical evidence on whether
different LLMs differ meaningfully. We conduct a large-scale empirical
evaluation on over 350 LLMs overall, using two popular leaderboards and a
resume-screening task. We find substantial correlation in model errors -- on
one leaderboard dataset, models agree 60% of the time when both models err. We
identify factors driving model correlation, including shared architectures and
providers. Crucially, however, larger and more accurate models have highly
correlated errors, even with distinct architectures and providers. Finally, we
show the effects of correlation in two downstream tasks: LLM-as-judge
evaluation and hiring -- the latter reflecting theoretical predictions
regarding algorithmic monoculture.

### Databases

### 1. [QUITE: A Query Rewrite System Beyond Rules with LLM Agents](http://arxiv.org/pdf/2506.07675v1)

Authors: Yuyang Song, Hanxu Yan, Jiale Lao, Yibo Wang, Yufei Li, Yuanchun Zhou, Jianguo Wang, Mingjie Tang

Query rewrite transforms SQL queries into semantically equivalent forms that
run more efficiently. Existing approaches mainly rely on predefined rewrite
rules, but they handle a limited subset of queries and can cause performance
regressions. This limitation stems from three challenges of rule-based query
rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite
rules do not generalize to new query patterns, and (3) some rewrite techniques
cannot be expressed as fixed rules. Motivated by the fact that human experts
exhibit significantly better rewrite ability but suffer from scalability, and
Large Language Models (LLMs) have demonstrated nearly human-level semantic and
reasoning abilities, we propose a new approach of using LLMs to rewrite SQL
queries beyond rules. Due to the hallucination problems in LLMs, directly
applying LLMs often leads to nonequivalent and suboptimal queries. To address
this issue, we propose QUITE (query rewrite), a training-free and
feedback-aware system based on LLM agents that rewrites SQL queries into
semantically equivalent forms with significantly better performance, covering a
broader range of query patterns and rewrite strategies compared to rule-based
methods. Firstly, we design a multi-agent framework controlled by a finite
state machine (FSM) to equip LLMs with the ability to use external tools and
enhance the rewrite process with real-time database feedback. Secondly, we
develop a rewrite middleware to enhance the ability of LLMs to generate
optimized query equivalents. Finally, we employ a novel hint injection
technique to improve execution plans for rewritten queries. Extensive
experiments show that QUITE reduces query execution time by up to 35.8% over
state-of-the-art approaches and produces 24.1% more rewrites than prior
methods, covering query cases that earlier systems did not handle.

### 2. [Quantum Information-Theoretical Size Bounds for Conjunctive Queries with Functional Dependencies](http://arxiv.org/pdf/2506.07552v1)

Authors: Valter Uotila, Jiaheng Lu

Deriving formulations for computing and estimating tight worst-case size
increases for conjunctive queries with various constraints has been at the core
of theoretical database research. If the problem has no constraints or only one
constraint, such as functional dependencies or degree constraints, tight
worst-case size bounds have been proven, and they are even practically
computable. If the problem has more than one constraint, computing tight bounds
can be difficult in practice and may even require an infinite number of linear
inequalities in its optimization formulation. While these challenges have been
addressed with varying methods, no prior research has employed quantum
information theory to address this problem. In this work, we establish a
connection between earlier work on estimating size bounds for conjunctive
queries with classical information theory and the field of quantum information
theory. We propose replacing the classical Shannon entropy formulation with the
quantum R\'enyi entropy. Whereas classical Shannon entropy requires infinitely
many inequalities to characterize the optimization space, R\'enyi entropy
requires only one type of inequality, which is non-negativity. Although this is
a promising modification, optimization with respect to the quantum states
instead of classical distributions creates a new set of challenges that prevent
us from finding a practically computable, tight worst-case size bound. In this
line, we propose a quantum version to derive worst-case size bounds. The
previous tight classical worst-case size bound can be viewed as a special limit
of this quantum bound. We also provide a comprehensive background on prior
research and discuss the future possibilities of quantum information theory in
theoretical database research.

### Distributed, Parallel, and Cluster Computing

### 1. [Addressing tokens dynamic generation, propagation, storage and renewal to secure the GlideinWMS pilot based jobs and system](http://arxiv.org/pdf/2506.07379v1)

Authors: Bruno Moreira Coimbra, Marco Mambelli

GlideinWMS has been one of the first middleware in the WLCG community to
transition from X.509 to support also tokens. The first step was to get from
the prototype in 2019 to using tokens in production in 2022. This paper will
present the challenges introduced by the wider adoption of tokens and the
evolution plans for securing the pilot infrastructure of GlideinWMS and
supporting the new requirements. In the last couple of years, the GlideinWMS
team supported the migration of experiments and resources to tokens. Inadequate
support in the current infrastructure, more stringent requirements, and the
higher spatial and temporal granularity forced GlideinWMS to revisit once more
how credentials are generated, used, and propagated. The new credential modules
have been designed to be used in multiple systems (GlideinWMS, HEPCloud) and
use a model where credentials have type, purpose, and different flows.
Credentials are dynamically generated in order to customize the duration and
limit the scope to the targeted resource. This allows to enforce the least
privilege principle. Finally, we also considered adding credential storage,
renewal, and invalidation mechanisms within the GlideinWMS infrastructure to
better serve the experiments' needs.

### 2. [A Terminology for Scientific Workflow Systems](http://arxiv.org/pdf/2506.07838v1)

Authors: Frédéric Sutera, Tainã Coleman, İlkay Altintaş, Rosa M. Badia, Bartosz Balis, Kyle Chard, Iacopo Colonnelli, Ewa Deelman, Paolo Di Tommaso, Thomas Fahringer, Carole Goble, Shantenu Jha, Daniel S. Katz, Johannes Köster, Ulf Leser, Kshitij Mehta, Hilary Oliver, J. -Luc Peterson, Giovanni Pizzi, Loïc Pottier, Raül Sirvent, Eric Suchyta, Douglas Thain, Sean R. Wilkinson, Justin M. Wozniak, Rafael Ferreira da Silva

The term scientific workflow has evolved over the last two decades to
encompass a broad range of compositions of interdependent compute tasks and
data movements. It has also become an umbrella term for processing in modern
scientific applications. Today, many scientific applications can be considered
as workflows made of multiple dependent steps, and hundreds of workflow
management systems (WMSs) have been developed to manage and run these
workflows. However, no turnkey solution has emerged to address the diversity of
scientific processes and the infrastructure on which they are implemented.
Instead, new research problems requiring the execution of scientific workflows
with some novel feature often lead to the development of an entirely new WMS. A
direct consequence is that many existing WMSs share some salient features,
offer similar functionalities, and can manage the same categories of workflows
but also have some distinct capabilities. This situation makes researchers who
develop workflows face the complex question of selecting a WMS. This selection
can be driven by technical considerations, to find the system that is the most
appropriate for their application and for the resources available to them, or
other factors such as reputation, adoption, strong community support, or
long-term sustainability. To address this problem, a group of WMS developers
and practitioners joined their efforts to produce a community-based terminology
of WMSs. This paper summarizes their findings and introduces this new
terminology to characterize WMSs. This terminology is composed of fives axes:
workflow characteristics, composition, orchestration, data management, and
metadata capture. Each axis comprises several concepts that capture the
prominent features of WMSs. Based on this terminology, this paper also presents
a classification of 23 existing WMSs according to the proposed axes and terms.

### 3. [New Limits on Distributed Quantum Advantage: Dequantizing Linear Programs](http://arxiv.org/pdf/2506.07574v1)

Authors: Alkida Balliu, Corinna Coupette, Antonio Cruciani, Francesco d'Amore, Massimo Equi, Henrik Lievonen, Augusto Modanese, Dennis Olivetti, Jukka Suomela

In this work, we give two results that put new limits on distributed quantum
advantage in the context of the LOCAL model of distributed computing. First, we
show that there is no distributed quantum advantage for any linear program. Put
otherwise, if there is a quantum-LOCAL algorithm $\mathcal{A}$ that finds an
$\alpha$-approximation of some linear optimization problem $\Pi$ in $T$
communication rounds, we can construct a classical, deterministic LOCAL
algorithm $\mathcal{A}'$ that finds an $\alpha$-approximation of $\Pi$ in $T$
rounds. As a corollary, all classical lower bounds for linear programs,
including the KMW bound, hold verbatim in quantum-LOCAL. Second, using the
above result, we show that there exists a locally checkable labeling problem
(LCL) for which quantum-LOCAL is strictly weaker than the classical
deterministic SLOCAL model. Our results extend from quantum-LOCAL also to
finitely dependent and non-signaling distributions, and one of the corollaries
of our work is that the non-signaling model and the SLOCAL model are
incomparable in the context of LCL problems: By prior work, there exists an LCL
problem for which SLOCAL is strictly weaker than the non-signaling model, and
our work provides a separation in the opposite direction.

### 4. [Optimal quantum sampling on distributed databases](http://arxiv.org/pdf/2506.07724v1)

Authors: Longyun Chen, Jingcheng Liu, Penghui Yao

Quantum sampling, a fundamental subroutine in numerous quantum algorithms,
involves encoding a given probability distribution in the amplitudes of a pure
state. Given the hefty cost of large-scale quantum storage, we initiate the
study of quantum sampling in a distributed setting. Specifically, we assume
that the data is distributed among multiple machines, and each machine solely
maintains a basic oracle that counts the multiplicity of individual elements.
Given a quantum sampling task, which is to sample from the joint database, a
coordinator can make oracle queries to all machines. We focus on the oblivious
communication model, where communications between the coordinator and the
machines are predetermined. We present both sequential and parallel algorithms:
the sequential algorithm queries the machines sequentially, while the parallel
algorithm allows the coordinator to query all machines simultaneously.
Furthermore, we prove that both algorithms are optimal in their respective
settings.

### 5. [FedCGD: Collective Gradient Divergence Optimized Scheduling for Wireless Federated Learning](http://arxiv.org/pdf/2506.07581v1)

Authors: Tan Chen, Jintao Yan, Yuxuan Sun, Sheng Zhou, Zhisheng Niu

Federated learning (FL) is a promising paradigm for multiple devices to
cooperatively train a model. When applied in wireless networks, two issues
consistently affect the performance of FL, i.e., data heterogeneity of devices
and limited bandwidth. Many papers have investigated device scheduling
strategies considering the two issues. However, most of them recognize data
heterogeneity as a property of individual devices. In this paper, we prove that
the convergence speed of FL is affected by the sum of device-level and
sample-level collective gradient divergence (CGD). The device-level CGD refers
to the gradient divergence of the scheduled device group, instead of the sum of
the individual device divergence. The sample-level CGD is statistically upper
bounded by sampling variance, which is inversely proportional to the total
number of samples scheduled for local update. To derive a tractable form of the
device-level CGD, we further consider a classification problem and transform it
into the weighted earth moving distance (WEMD) between the group distribution
and the global distribution. Then we propose FedCGD algorithm to minimize the
sum of multi-level CGDs by balancing WEMD and sampling variance, within
polynomial time. Simulation shows that the proposed strategy increases
classification accuracy on the CIFAR-10 dataset by up to 4.2\% while scheduling
41.8\% fewer devices, and flexibly switches between reducing WEMD and reducing
sampling variance.

### 6. [TimberStrike: Dataset Reconstruction Attack Revealing Privacy Leakage in Federated Tree-Based Systems](http://arxiv.org/pdf/2506.07605v1)

Authors: Marco Di Gennaro, Giovanni De Lucia, Stefano Longari, Stefano Zanero, Michele Carminati

Federated Learning has emerged as a privacy-oriented alternative to
centralized Machine Learning, enabling collaborative model training without
direct data sharing. While extensively studied for neural networks, the
security and privacy implications of tree-based models remain underexplored.
This work introduces TimberStrike, an optimization-based dataset reconstruction
attack targeting horizontally federated tree-based models. Our attack, carried
out by a single client, exploits the discrete nature of decision trees by using
split values and decision paths to infer sensitive training data from other
clients. We evaluate TimberStrike on State-of-the-Art federated gradient
boosting implementations across multiple frameworks, including Flower, NVFlare,
and FedTree, demonstrating their vulnerability to privacy breaches. On a
publicly available stroke prediction dataset, TimberStrike consistently
reconstructs between 73.05% and 95.63% of the target dataset across all
implementations. We further analyze Differential Privacy, showing that while it
partially mitigates the attack, it also significantly degrades model
performance. Our findings highlight the need for privacy-preserving mechanisms
specifically designed for tree-based Federated Learning systems, and we provide
preliminary insights into their design.

### Digital Libraries

### 1. [From Rapid Release to Reinforced Elite: Citation Inequality Is Stronger in Preprints than Journals](http://arxiv.org/pdf/2506.07547v1)

Authors: Chiaki Miura, Ichiro Sakata

Preprint has been considered to mainly supplement journal-based systems for
the rapid dissemination of relevant scientific knowledge, and has historically
been by studies indicating that preprints and published reports have comparable
authorship, references, and quality.However, as preprint increasingly serve as
an independent medium for scholarly communication rather than precursors to the
version of record, it remains uncertain how preprint usage is shaping
scientific discourse.Our research revealed that the preprint citations exhibit
on average x times higher inequality than journal citations, consistently among
categories.This trend persisted even when controlling for the age, the mean
citation count, and the open access status of the journal matched to each of
the preprint categories.We also found that the citation inequality in preprints
is not solely driven by a few highly cited papers or those with no impact, but
rather reflects a broader systemic effect.Preprint that subsequently published
under journal and those not show no significant difference in citation
inequality.Further analyses of the structural factors show that preferential
attachment does not significantly contribute to citation inequality in
preprints, whereas author prestige plays a substantial role.These results
together suggest that researchers disproportionately rely on reputable peers in
the unvetted environment.This highlights a potential vulnerability in preprint
ecosystems where reputation-driven citation may hinder scientific diversity.

### 2. [Research quality evaluation by AI in the era of Large Language Models: Advantages, disadvantages, and systemic effects](http://arxiv.org/pdf/2506.07748v1)

Authors: Mike Thelwall

Artificial Intelligence (AI) technologies like ChatGPT now threaten
bibliometrics as the primary generators of research quality indicators. They
are already used in at least one research quality evaluation system and
evidence suggests that they are used informally by many peer reviewers. Since
using bibliometrics to support research evaluation continues to be
controversial, this article reviews the corresponding advantages and
disadvantages of AI-generated quality scores. From a technical perspective,
generative AI based on Large Language Models (LLMs) equals or surpasses
bibliometrics in most important dimensions, including accuracy (mostly higher
correlations with human scores), and coverage (more fields, more recent years)
and may reflect more research quality dimensions. Like bibliometrics, current
LLMs do not "measure" research quality, however. On the clearly negative side,
LLM biases are currently unknown for research evaluation, and LLM scores are
less transparent than citation counts. From a systemic perspective, the key
issue is how introducing LLM-based indicators into research evaluation will
change the behaviour of researchers. Whilst bibliometrics encourage some
authors to target journals with high impact factors or to try to write highly
cited work, LLM-based indicators may push them towards writing misleading
abstracts and overselling their work in the hope of impressing the AI.
Moreover, if AI-generated journal indicators replace impact factors, then this
would encourage journals to allow authors to oversell their work in abstracts,
threatening the integrity of the academic record.

### Discrete Mathematics

### 1. [HyColor: An Efficient Heuristic Algorithm for Graph Coloring](http://arxiv.org/pdf/2506.07373v1)

Authors: Enqiang Zhu, Yu Zhang, Haopeng Sun, Ziqi Wei, Witold Pedrycz, Chanjuan Liu, Jin Xu

The graph coloring problem (GCP) is a classic combinatorial optimization
problem that aims to find the minimum number of colors assigned to vertices of
a graph such that no two adjacent vertices receive the same color. GCP has been
extensively studied by researchers from various fields, including mathematics,
computer science, and biological science. Due to the NP-hard nature, many
heuristic algorithms have been proposed to solve GCP. However, existing GCP
algorithms focus on either small hard graphs or large-scale sparse graphs (with
up to 10^7 vertices). This paper presents an efficient hybrid heuristic
algorithm for GCP, named HyColor, which excels in handling large-scale sparse
graphs while achieving impressive results on small dense graphs. The efficiency
of HyColor comes from the following three aspects: a local decision strategy to
improve the lower bound on the chromatic number; a graph-reduction strategy to
reduce the working graph; and a k-core and mixed degree-based greedy heuristic
for efficiently coloring graphs. HyColor is evaluated against three
state-of-the-art GCP algorithms across four benchmarks, comprising three
large-scale sparse graph benchmarks and one small dense graph benchmark,
totaling 209 instances. The results demonstrate that HyColor consistently
outperforms existing heuristic algorithms in both solution accuracy and
computational efficiency for the majority of instances. Notably, HyColor
achieved the best solutions in 194 instances (over 93%), with 34 of these
solutions significantly surpassing those of other algorithms. Furthermore,
HyColor successfully determined the chromatic number and achieved optimal
coloring in 128 instances.

### 2. [Half-Iterates of $x(1+x)$, $\sin(x)$ and $\exp(x/e)$](http://arxiv.org/pdf/2506.07625v1)

Authors: Steven Finch

The title reflects the original intent of this paper -- to continue exploring
compositional square roots -- focusing on Walker's (1991) study of the Abel
equation $f(\exp(x/e))=f(x)+1$ for real $x \neq e$. An unexpected discovery
changed everything. We already knew that \'Ecalle (1974) developed theory
inspiring relevant calculations across years. Precise details, however, seemed
to escape attention until recently. Helpful online posts of Jagy (2012) are
important not to overlook. The new algorithm is exceedingly simple and
outperforms a rival method, due to Mavecha & Laohakosol (2013), which we
mistakenly advocated until now. Our loyalty has correspondingly shifted.

### 3. [Leveraging Network Methods for Hub-like Microservice Detection](http://arxiv.org/pdf/2506.07683v1)

Authors: Alexander Bakhtin, Matteo Esposito, Valentina Lenarduzzi, Davide Taibi

Context: Microservice Architecture is a popular architectural paradigm that
facilitates flexibility by decomposing applications into small, independently
deployable services. Catalogs of architectural anti-patterns have been proposed
to highlight the negative aspects of flawed microservice design. In particular,
the Hub-like anti-pattern lacks an unambiguous definition and detection method.
Aim: In this work, we aim to find a robust detection approach for the Hub-like
microservice anti-pattern that outputs a reasonable number of Hub-like
candidates with high precision. Method: We leveraged a dataset of 25
microservice networks and several network hub detection techniques to identify
the Hub-like anti-pattern, namely scale-free property, centrality metrics and
clustering coefficient, minimum description length principle, and the approach
behind the Arcan tool. Results and Conclusion: Our findings revealed that the
studied architectural networks are not scale-free, that most considered hub
detection approaches do not agree on the detected hubs, and that the method by
Kirkley leveraging the Erdos-Renyi encoding is the most accurate one in terms
of the number of detected hubs and the detection precision. Investigating
further the applicability of these methods to detecting Hub-like components in
microservice-based and other systems opens up new research directions.
Moreover, our results provide an evaluation of the approach utilized by the
widely used Arcan tool and highlight the potential to update the tool to use
the normalized degree centrality of a component in the network, or for the
approach based on ER encoding to be adopted instead.

### 4. [Centrality Change Proneness: an Early Indicator of Microservice Architectural Degradation](http://arxiv.org/pdf/2506.07690v1)

Authors: Alexander Bakhtin, Matteo Esposito, Valentina Lenarduzzi, Davide Taibi

Over the past decade, the wide adoption of Microservice Architecture has
required the identification of various patterns and anti-patterns to prevent
Microservice Architectural Degradation. Frequently, the systems are modelled as
a network of connected services. Recently, the study of temporal networks has
emerged as a way to describe and analyze evolving networks. Previous research
has explored how software metrics such as size, complexity, and quality are
related to microservice centrality in the architectural network. This study
investigates whether temporal centrality metrics can provide insight into the
early detection of architectural degradation by correlating or affecting
software metrics. We reconstructed the architecture of 7 releases of an OSS
microservice project with 42 services. For every service in every release, we
computed the software and centrality metrics. From one of the latter, we
derived a new metric, Centrality Change Proneness. We then explored the
correlation between the metrics. We identified 7 size and 5 complexity metrics
that have a consistent correlation with centrality, while Centrality Change
Proneness did not affect the software metrics, thus providing yet another
perspective and an early indicator of microservice architectural degradation.

### 5. [Stability and Extension of Steady and Ranging Persistence](http://arxiv.org/pdf/2506.07911v1)

Authors: Yann-Situ Gazull

Persistent homology is a topological data analysis tool that has been widely
generalized, extending its scope outside the field of topology. Among its
extensions, steady and ranging persistence was developed to study a wide
variety of graph properties. Precisely, given a feature of interest on graphs,
it is possible to build two types of persistence (steady and ranging
persistence) that follow the evolution of the feature along graph filtrations.
This study extends steady and ranging persistence to other objects using
category theory and investigates the stability of such persistence. In
particular, a characterization of the features that induce balanced steady and
ranging persistence is provided. The main results of this study are illustrated
using a practical implementation for hypergraphs.

### Data Structures and Algorithms

### 1. [On Sketching Trimmed Statistics](http://arxiv.org/pdf/2506.07342v1)

Authors: Honghao Lin, Hoai-An Nguyen, David P. Woodruff

We present space-efficient linear sketches for estimating trimmed statistics
of an $n$-dimensional frequency vector $x$, e.g., the sum of $p$-th powers of
the largest $k$ frequencies (i.e., entries) in absolute value, or the
$k$-trimmed vector, which excludes the top and bottom $k$ frequencies. This is
called the $F_p$ moment of the trimmed vector. Trimmed measures are used in
robust estimation, as seen in the R programming language's `trim.var' function
and the `trim' parameter in the mean function. Linear sketches improve time and
memory efficiency and are applicable to streaming and distributed settings. We
initiate the study of sketching these statistics and give a new condition for
capturing their space complexity. When $k \ge n/poly\log n$, we give a linear
sketch using $poly(1/\varepsilon, \log n)$ space which provides a $(1 \pm
\varepsilon)$ approximation to the top-$k$ $F_p$ moment for $p \in [0,2]$. For
general $k$, we give a sketch with the same guarantees under a condition
relating the $k$-th largest frequency to the tail mass, and show this condition
is necessary. For the $k$-trimmed version, our sketch achieves optimal error
guarantees under the same condition. We extend our methods to $p > 2$ and also
address related problems such as computing the $F_p$ moment of frequencies
above a threshold, finding the largest $k$ such that the $F_p$ moment of the
top $k$ exceeds $k^{p+1}$, and the $F_p$ moment of the top $k$ frequencies such
that each entry is at least $k$. Notably, our algorithm for this third
application improves upon the space bounds of the algorithm of Govindan,
Monemizadeh, and Muthukrishnan (PODS '17) for computing the $h$-index. We show
empirically that our top $k$ algorithm uses much less space compared to
Count-Sketch while achieving the same error.

### 2. [On Deterministically Finding an Element of High Order Modulo a Composite](http://arxiv.org/pdf/2506.07668v1)

Authors: Ziv Oznovich, Ben Lee Volk

We give a deterministic algorithm that, given a composite number $N$ and a
target order $D \ge N^{1/6}$, runs in time $D^{1/2+o(1)}$ and finds either an
element $a \in \mathbb{Z}_N^*$ of multiplicative order at least $D$, or a
nontrivial factor of $N$. Our algorithm improves upon an algorithm of Hittmeir
(arXiv:1608.08766), who designed a similar algorithm under the stronger
assumption $D \ge N^{2/5}$. Hittmeir's algorithm played a crucial role in the
recent breakthrough deterministic integer factorization algorithms of Hittmeir
and Harvey (arXiv:2006.16729, arXiv:2010.05450, arXiv:2105.11105). When $N$ is
assumed to have an $r$-power divisor with $r\ge 2$, our algorithm provides the
same guarantees assuming $D \ge N^{1/6r}$.

### 3. [Discrete and Continuous Difference of Submodular Minimization](http://arxiv.org/pdf/2506.07952v1)

Authors: George Orfanides, Tim Hoheisel, Marwa El Halabi

Submodular functions, defined on continuous or discrete domains, arise in
numerous applications. We study the minimization of the difference of two
submodular (DS) functions, over both domains, extending prior work restricted
to set functions. We show that all functions on discrete domains and all smooth
functions on continuous domains are DS. For discrete domains, we observe that
DS minimization is equivalent to minimizing the difference of two convex (DC)
functions, as in the set function case. We propose a novel variant of the DC
Algorithm (DCA) and apply it to the resulting DC Program, obtaining comparable
theoretical guarantees as in the set function case. The algorithm can be
applied to continuous domains via discretization. Experiments demonstrate that
our method outperforms baselines in integer compressive sensing and integer
least squares.

### Emerging Technologies

### 1. [Prompt to Protection: A Comparative Study of Multimodal LLMs in Construction Hazard Recognition](http://arxiv.org/pdf/2506.07436v1)

Authors: Nishi Chaudhary, S M Jamil Uddin, Sathvik Sharath Chandra, Anto Ovid, Alex Albert

The recent emergence of multimodal large language models (LLMs) has
introduced new opportunities for improving visual hazard recognition on
construction sites. Unlike traditional computer vision models that rely on
domain-specific training and extensive datasets, modern LLMs can interpret and
describe complex visual scenes using simple natural language prompts. However,
despite growing interest in their applications, there has been limited
investigation into how different LLMs perform in safety-critical visual tasks
within the construction domain. To address this gap, this study conducts a
comparative evaluation of five state-of-the-art LLMs: Claude-3 Opus, GPT-4.5,
GPT-4o, GPT-o3, and Gemini 2.0 Pro, to assess their ability to identify
potential hazards from real-world construction images. Each model was tested
under three prompting strategies: zero-shot, few-shot, and chain-of-thought
(CoT). Zero-shot prompting involved minimal instruction, few-shot incorporated
basic safety context and a hazard source mnemonic, and CoT provided
step-by-step reasoning examples to scaffold model thinking. Quantitative
analysis was performed using precision, recall, and F1-score metrics across all
conditions. Results reveal that prompting strategy significantly influenced
performance, with CoT prompting consistently producing higher accuracy across
models. Additionally, LLM performance varied under different conditions, with
GPT-4.5 and GPT-o3 outperforming others in most settings. The findings also
demonstrate the critical role of prompt design in enhancing the accuracy and
consistency of multimodal LLMs for construction safety applications. This study
offers actionable insights into the integration of prompt engineering and LLMs
for practical hazard recognition, contributing to the development of more
reliable AI-assisted safety systems.

### 2. [Profiling Electric Vehicles via Early Charging Voltage Patterns](http://arxiv.org/pdf/2506.07714v1)

Authors: Francesco Marchiori, Denis Donadel, Alessandro Brighente, Mauro Conti

Electric Vehicles (EVs) are rapidly gaining adoption as a sustainable
alternative to fuel-powered vehicles, making secure charging infrastructure
essential. Despite traditional authentication protocols, recent results showed
that attackers may steal energy through tailored relay attacks. One
countermeasure is leveraging the EV's fingerprint on the current exchanged
during charging. However, existing methods focus on the final charging stage,
allowing malicious actors to consume substantial energy before being detected
and repudiated. This underscores the need for earlier and more effective
authentication methods to prevent unauthorized charging. Meanwhile, profiling
raises privacy concerns, as uniquely identifying EVs through charging patterns
could enable user tracking.
  In this paper, we propose a framework for uniquely identifying EVs using
physical measurements from the early charging stages. We hypothesize that
voltage behavior early in the process exhibits similar characteristics to
current behavior in later stages. By extracting features from early voltage
measurements, we demonstrate the feasibility of EV profiling. Our approach
improves existing methods by enabling faster and more reliable vehicle
identification. We test our solution on a dataset of 7408 usable charges from
49 EVs, achieving up to 0.86 accuracy. Feature importance analysis shows that
near-optimal performance is possible with just 10 key features, improving
efficiency alongside our lightweight models. This research lays the foundation
for a novel authentication factor while exposing potential privacy risks from
unauthorized access to charging data.

### 3. [Quantum-Enhanced Spectral Solution of the Poisson Equation](http://arxiv.org/pdf/2506.07743v1)

Authors: G. Intoccia, U. Chirico, G. Pepe, S. Cuomo

We present a hybrid numerical-quantum method for solving the Poisson equation
under homogeneous Dirichlet boundary conditions, leveraging the Quantum Fourier
Transform (QFT) to enhance computational efficiency and reduce time and space
complexity. This approach bypasses the integration-heavy calculations of
classical methods, which have to deal with high computational costs for large
number of points. The proposed method estimates the coefficients of the series
expansion of the solution directly within the quantum framework. Numerical
experiments validate its effectiveness and reveal significant improvements in
terms of time and space complexity and solution accuracy, demonstrating the
capability of quantum-assisted techniques to contribute in solving partial
differential equations (PDEs). Despite the inherent challenges of quantum
implementation, the present work serves as a starting point for future
researches aimed at refining and expanding quantum numerical methods.

### 4. [A weighted quantum ensemble of homogeneous quantum classifiers](http://arxiv.org/pdf/2506.07810v1)

Authors: Emiliano Tolotti, Enrico Blanzieri, Davide Pastorello

Ensemble methods in machine learning aim to improve prediction accuracy by
combining multiple models. This is achieved by ensuring diversity among
predictors to capture different data aspects. Homogeneous ensembles use
identical models, achieving diversity through different data subsets, and
weighted-average ensembles assign higher influence to more accurate models
through a weight learning procedure. We propose a method to achieve a weighted
homogeneous quantum ensemble using quantum classifiers with indexing registers
for data encoding. This approach leverages instance-based quantum classifiers,
enabling feature and training point subsampling through superposition and
controlled unitaries, and allowing for a quantum-parallel execution of diverse
internal classifiers with different data compositions in superposition. The
method integrates a learning process involving circuit execution and classical
weight optimization, for a trained ensemble execution with weights encoded in
the circuit at test-time. Empirical evaluation demonstrate the effectiveness of
the proposed method, offering insights into its performance.

### Formal Languages and Automata Theory

### 1. [Language Models over Canonical Byte-Pair Encodings](http://arxiv.org/pdf/2506.07956v1)

Authors: Tim Vieira, Tianyu Liu, Clemente Pasti, Yahya Emara, Brian DuSell, Benjamin LeBrun, Mario Giulianelli, Juan Luis Gastaldi, Timothy J. O'Donnell, Ryan Cotterell

Modern language models represent probability distributions over character
strings as distributions over (shorter) token strings derived via a
deterministic tokenizer, such as byte-pair encoding. While this approach is
highly effective at scaling up language models to large corpora, its current
incarnations have a concerning property: the model assigns nonzero probability
mass to an exponential number of $\it{noncanonical}$ token encodings of each
character string -- these are token strings that decode to valid character
strings but are impossible under the deterministic tokenizer (i.e., they will
never be seen in any training corpus, no matter how large). This misallocation
is both erroneous, as noncanonical strings never appear in training data, and
wasteful, diverting probability mass away from plausible outputs. These are
avoidable mistakes! In this work, we propose methods to enforce canonicality in
token-level language models, ensuring that only canonical token strings are
assigned positive probability. We present two approaches: (1) canonicality by
conditioning, leveraging test-time inference strategies without additional
training, and (2) canonicality by construction, a model parameterization that
guarantees canonical outputs but requires training. We demonstrate that fixing
canonicality mistakes improves the likelihood of held-out data for several
models and corpora.

### Graphics

### 1. [PIG: Physically-based Multi-Material Interaction with 3D Gaussians](http://arxiv.org/pdf/2506.07657v1)

Authors: Zeyu Xiao, Zhenyi Wu, Mingyang Sun, Qipeng Yan, Yufan Guo, Zhuoer Liang, Lihua Zhang

3D Gaussian Splatting has achieved remarkable success in reconstructing both
static and dynamic 3D scenes. However, in a scene represented by 3D Gaussian
primitives, interactions between objects suffer from inaccurate 3D
segmentation, imprecise deformation among different materials, and severe
rendering artifacts. To address these challenges, we introduce PIG:
Physically-Based Multi-Material Interaction with 3D Gaussians, a novel approach
that combines 3D object segmentation with the simulation of interacting objects
in high precision. Firstly, our method facilitates fast and accurate mapping
from 2D pixels to 3D Gaussians, enabling precise 3D object-level segmentation.
Secondly, we assign unique physical properties to correspondingly segmented
objects within the scene for multi-material coupled interactions. Finally, we
have successfully embedded constraint scales into deformation gradients,
specifically clamping the scaling and rotation properties of the Gaussian
primitives to eliminate artifacts and achieve geometric fidelity and visual
consistency. Experimental results demonstrate that our method not only
outperforms the state-of-the-art (SOTA) in terms of visual quality, but also
opens up new directions and pipelines for the field of physically realistic
scene generation.

### 2. [SMaRCSim: Maritime Robotics Simulation Modules](http://arxiv.org/pdf/2506.07781v1)

Authors: Mart Kartašev, David Dörner, Özer Özkahraman, Petter Ögren, Ivan Stenius, John Folkesson

Developing new functionality for underwater robots and testing them in the
real world is time-consuming and resource-intensive. Simulation environments
allow for rapid testing before field deployment. However, existing tools lack
certain functionality for use cases in our project: i) developing
learning-based methods for underwater vehicles; ii) creating teams of
autonomous underwater, surface, and aerial vehicles; iii) integrating the
simulation with mission planning for field experiments. A holistic solution to
these problems presents great potential for bringing novel functionality into
the underwater domain. In this paper we present SMaRCSim, a set of simulation
packages that we have developed to help us address these issues.

### 3. [Speedy Deformable 3D Gaussian Splatting: Fast Rendering and Compression of Dynamic Scenes](http://arxiv.org/pdf/2506.07917v1)

Authors: Allen Tu, Haiyang Ying, Alex Hanson, Yonghan Lee, Tom Goldstein, Matthias Zwicker

Recent extensions of 3D Gaussian Splatting (3DGS) to dynamic scenes achieve
high-quality novel view synthesis by using neural networks to predict the
time-varying deformation of each Gaussian. However, performing per-Gaussian
neural inference at every frame poses a significant bottleneck, limiting
rendering speed and increasing memory and compute requirements. In this paper,
we present Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), a general
pipeline for accelerating the rendering speed of dynamic 3DGS and 4DGS
representations by reducing neural inference through two complementary
techniques. First, we propose a temporal sensitivity pruning score that
identifies and removes Gaussians with low contribution to the dynamic scene
reconstruction. We also introduce an annealing smooth pruning mechanism that
improves pruning robustness in real-world scenes with imprecise camera poses.
Second, we propose GroupFlow, a motion analysis technique that clusters
Gaussians by trajectory similarity and predicts a single rigid transformation
per group instead of separate deformations for each Gaussian. Together, our
techniques accelerate rendering by $10.37\times$, reduce model size by
$7.71\times$, and shorten training time by $2.71\times$ on the NeRF-DS dataset.
SpeeDe3DGS also improves rendering speed by $4.20\times$ and $58.23\times$ on
the D-NeRF and HyperNeRF vrig datasets. Our methods are modular and can be
integrated into any deformable 3DGS or 4DGS framework.

### 4. [Immersive Visualization of Flat Surfaces Using Ray Marching](http://arxiv.org/pdf/2506.07558v1)

Authors: Fabian Lander, Diaaeldin Taha

We present an effective method for visualizing flat surfaces using ray
marching. Our approach provides an intuitive way to explore translation
surfaces, mirror rooms, unfolded polyhedra, and translation prisms while
maintaining computational efficiency. We demonstrate the utility of the method
through various examples and provide implementation insights for programmers.
Finally, we discuss the use of our visualizations in outreach. We make our
simulations and code available online.

### 5. [GaussianVAE: Adaptive Learning Dynamics of 3D Gaussians for High-Fidelity Super-Resolution](http://arxiv.org/pdf/2506.07897v1)

Authors: Shuja Khalid, Mohamed Ibrahim, Yang Liu

We present a novel approach for enhancing the resolution and geometric
fidelity of 3D Gaussian Splatting (3DGS) beyond native training resolution.
Current 3DGS methods are fundamentally limited by their input resolution,
producing reconstructions that cannot extrapolate finer details than are
present in the training views. Our work breaks this limitation through a
lightweight generative model that predicts and refines additional 3D Gaussians
where needed most. The key innovation is our Hessian-assisted sampling
strategy, which intelligently identifies regions that are likely to benefit
from densification, ensuring computational efficiency. Unlike computationally
intensive GANs or diffusion approaches, our method operates in real-time
(0.015s per inference on a single consumer-grade GPU), making it practical for
interactive applications. Comprehensive experiments demonstrate significant
improvements in both geometric accuracy and rendering quality compared to
state-of-the-art methods, establishing a new paradigm for resolution-free 3D
scene enhancement.

### 6. [Squeeze3D: Your 3D Generation Model is Secretly an Extreme Neural Compressor](http://arxiv.org/pdf/2506.07932v1)

Authors: Rishit Dagli, Yushi Guan, Sankeerth Durvasula, Mohammadreza Mofayezi, Nandita Vijaykumar

We propose Squeeze3D, a novel framework that leverages implicit prior
knowledge learnt by existing pre-trained 3D generative models to compress 3D
data at extremely high compression ratios. Our approach bridges the latent
spaces between a pre-trained encoder and a pre-trained generation model through
trainable mapping networks. Any 3D model represented as a mesh, point cloud, or
a radiance field is first encoded by the pre-trained encoder and then
transformed (i.e. compressed) into a highly compact latent code. This latent
code can effectively be used as an extremely compressed representation of the
mesh or point cloud. A mapping network transforms the compressed latent code
into the latent space of a powerful generative model, which is then conditioned
to recreate the original 3D model (i.e. decompression). Squeeze3D is trained
entirely on generated synthetic data and does not require any 3D datasets. The
Squeeze3D architecture can be flexibly used with existing pre-trained 3D
encoders and existing generative models. It can flexibly support different
formats, including meshes, point clouds, and radiance fields. Our experiments
demonstrate that Squeeze3D achieves compression ratios of up to 2187x for
textured meshes, 55x for point clouds, and 619x for radiance fields while
maintaining visual quality comparable to many existing methods. Squeeze3D only
incurs a small compression and decompression latency since it does not involve
training object-specific networks to compress an object.

### Computer Science and Game Theory

### 1. [Diffusion of Responsibility in Collective Decision Making](http://arxiv.org/pdf/2506.07935v1)

Authors: Pavel Naumov, Jia Tao

The term "diffusion of responsibility'' refers to situations in which
multiple agents share responsibility for an outcome, obscuring individual
accountability. This paper examines this frequently undesirable phenomenon in
the context of collective decision-making mechanisms.
  The work shows that if a decision is made by two agents, then the only way to
avoid diffusion of responsibility is for one agent to act as a "dictator'',
making the decision unilaterally. In scenarios with more than two agents, any
diffusion-free mechanism is an "elected dictatorship'' where the agents elect a
single agent to make a unilateral decision.
  The technical results are obtained by defining a bisimulation of
decision-making mechanisms, proving that bisimulation preserves
responsibility-related properties, and establishing the results for a smallest
bisimular mechanism.

### Human-Computer Interaction

### 1. [Happiness Finder: Exploring the Role of AI in Enhancing Well-Being During Four-Leaf Clover Searches](http://arxiv.org/pdf/2506.07393v1)

Authors: Anna Yokokubo, Takeo Hamada, Tatsuya Ishizuka, Hiroaki Mori, Noboru Koshizuka

A four-leaf clover (FLC) symbolizes luck and happiness worldwide, but it is
hard to distinguish it from the common three-leaf clover. While AI technology
can assist in searching for FLC, it may not replicate the traditional search's
sense of achievement. This study explores searcher feelings when AI aids the
FLC search. In this study, we developed a system called ``Happiness Finder''
that uses object detection algorithms on smartphones or tablets to support the
search. We exhibited HappinessFinder at an international workshop, allowing
participants to experience four-leaf clover searching using potted artificial
clovers and the HappinessFinder app. This paper reports the findings from this
demonstration.

### 2. [Interaction Analysis by Humans and AI: A Comparative Perspective](http://arxiv.org/pdf/2506.07707v1)

Authors: Maryam Teimouri, Filip Ginter, Tomi "bgt" Suovuo

This paper explores how Mixed Reality (MR) and 2D video conferencing
influence children's communication during a gesture-based guessing game.
Finnish-speaking participants engaged in a short collaborative task using two
different setups: Microsoft HoloLens MR and Zoom. Audio-video recordings were
transcribed and analyzed using Large Language Models (LLMs), enabling iterative
correction, translation, and annotation. Despite limitations in annotations'
accuracy and agreement, automated approaches significantly reduced processing
time and allowed non-Finnish-speaking researchers to participate in data
analysis. Evaluations highlight both the efficiency and constraints of
LLM-based analyses for capturing children's interactions across these
platforms. Initial findings indicate that MR fosters richer interaction,
evidenced by higher emotional expression during annotation, and heightened
engagement, while Zoom offers simplicity and accessibility. This study
underscores the potential of MR to enhance collaborative learning experiences
for children in distributed settings.

### 3. [Supporting Aging Well through Accessible Digital Games: The Supplemental Role of AI in Game Design for Older Adults](http://arxiv.org/pdf/2506.07777v1)

Authors: Brandon Lyman, Yichi Zhang, Celia Pearce, Miso Kim, Casper Harteveld, Leanne Chukoskie, Bob De Schutter

As the population continues to age, and gaming continues to grow as a hobby
for older people, heterogeneity among older adult gamers is increasing. We
argue that traditional game-based accessibility features, such as simplified
input schemes, redundant information channels, and increased legibility of
digital user interfaces, are increasingly limited in the face of this
heterogeneity. This is because such features affect all older adult players
simultaneously and therefore are designed generically. We introduce artificial
intelligence, although it has its own limitations and ethical concerns, as a
method of creating player-based accessibility features, given the adaptive
nature of the emerging technology. These accessibility features may help to
address unique assemblage of accessibility needs an individual may accumulate
through age. We adopt insights from gerontology, HCI, and disability studies
into the digital game design discourse for older adults, and we contribute
insight that can guide the integration of player-based accessibility features
to supplement game-based counterparts. The accessibility of digital games for
heterogenous older adult audience is paramount, as the medium offers short-term
social, emotional, psychological, cognitive, and physical that support the
long-term goal of aging well.

### 4. [Integrating Artificial Intelligence as Assistive Technology for Older Adult Gamers: A Pilot Study](http://arxiv.org/pdf/2506.07830v1)

Authors: Yichi Zhang, Brandon Lyman, Celia Pearce, Miso Kim, Casper Harteveld, Leanne Chukoskie, Bob De Schutter

With respect to digital games, older adults are a demographic that is often
underserved due to an industry-wide focus on younger audiences' preferences and
skill sets. Meanwhile, as artificial intelligence (AI) continues to expand into
everyday technologies, its assistive capabilities have been recognized,
suggesting its potential in improving the gaming experience for older gamers.
To study this potential, we iteratively developed a pilot survey aimed at
understanding older adult gamers' current gameplay preference, challenges they
are facing, and their perspectives of AI usage in gaming. This article
contributes an overview of our iterative survey-design workflow, and pilot
results from 39 participants. During each iteration, we analyzed the survey's
efficacy and adjusted the content, language, and format to better capture
meaningful data, and was able to create a refined survey for a larger, more
representative future parent study. At the same time, preliminary findings
suggest that for older adult gamers, usability issues in gaming remain key
obstacles, while this demographic's perceptions of AI are shaped by both its
practical benefits and concerns about autonomy and complexity. These findings
also offer early insights for the design of age-inclusive, AI-supported gaming
experiences.

### 5. [Predicting Situation Awareness from Physiological Signals](http://arxiv.org/pdf/2506.07930v1)

Authors: Kieran J. Smith, Tristan C. Endsley, Torin K. Clark

Situation awareness (SA)--comprising the ability to 1) perceive critical
elements in the environment, 2) comprehend their meanings, and 3) project their
future states--is critical for human operator performance. Due to the
disruptive nature of gold-standard SA measures, researchers have sought
physiological indicators to provide real-time information about SA. We extend
prior work by using a multimodal suite of neurophysiological,
psychophysiological, and behavioral signals, predicting all three levels of SA
along a continuum, and predicting a comprehensive measure of SA in a complex
multi-tasking simulation. We present a lab study in which 31 participants
controlled an aircraft simulator task battery while wearing physiological
sensors and responding to SA 'freeze-probe' assessments. We demonstrate the
validity of task and assessment for measuring SA. Multimodal physiological
models predict SA with greater predictive performance ($Q^2$ for levels 1-3 and
total, respectively: 0.14, 0.00, 0.26, and 0.36) than models built with
shuffled labels, demonstrating that multimodal physiological signals provide
useful information in predicting all SA levels. Level 3 SA (projection) was
best predicted, and level 2 SA comprehension) was the most challenging to
predict. Ablation analysis and single sensor models found EEG and eye-tracking
signals to be particularly useful to predictions of level 3 and total SA. A
reduced sensor fusion model showed that predictive performance can be
maintained with a subset of sensors. This first rigorous cross-validation
assessment of predictive performance demonstrates the utility of multimodal
physiological signals for inferring complex, holistic, objective measures of SA
at all levels, non-disruptively, and along a continuum.

### 6. [Implementation Considerations for Automated AI Grading of Student Work](http://arxiv.org/pdf/2506.07955v1)

Authors: Zewei, Tian, Alex Liu, Lief Esbenshade, Shawon Sarkar, Zachary Zhang, Kevin He, Min Sun

This study explores the classroom implementation of an AI-powered grading
platform in K-12 settings through a co-design pilot with 19 teachers. We
combine platform usage logs, surveys, and qualitative interviews to examine how
teachers use AI-generated rubrics and grading feedback. Findings reveal that
while teachers valued the AI's rapid narrative feedback for formative purposes,
they distrusted automated scoring and emphasized the need for human oversight.
Students welcomed fast, revision-oriented feedback but remained skeptical of
AI-only grading. We discuss implications for the design of trustworthy,
teacher-centered AI assessment tools that enhance feedback while preserving
pedagogical agency.

### 7. [Supporting Construction Worker Well-Being with a Multi-Agent Conversational AI System](http://arxiv.org/pdf/2506.07997v1)

Authors: Fan Yang, Yuan Tian, Jiansong Zhang

The construction industry is characterized by both high physical and
psychological risks, yet supports of mental health remain limited. While
advancements in artificial intelligence (AI), particularly large language
models (LLMs), offer promising solutions, their potential in construction
remains largely underexplored. To bridge this gap, we developed a
conversational multi-agent system that addresses industry-specific challenges
through an AI-driven approach integrated with domain knowledge. In parallel, it
fulfills construction workers' basic psychological needs by enabling
interactions with multiple agents, each has a distinct persona. This approach
ensures that workers receive both practical problem-solving support and social
engagement, ultimately contributing to their overall well-being. We evaluate
its usability and effectiveness through a within-subjects user study with 12
participants. The results show that our system significantly outperforms the
single-agent baseline, achieving improvements of 18% in usability, 40% in
self-determination, 60% in social presence, and 60% in trust. These findings
highlight the promise of LLM-driven AI systems in providing domain-specific
support for construction workers.

### 8. [Human Side of Smart Contract Fuzzing: An Empirical Study](http://arxiv.org/pdf/2506.07389v1)

Authors: Guanming Qiao, Partha Protim Paul

Smart contract (SC) fuzzing is a critical technique for detecting
vulnerabilities in blockchain applications. However, its adoption remains
challenging for practitioners due to fundamental differences between SCs and
traditional software systems. In this study, we investigate the challenges
practitioners face when adopting SC fuzzing tools by conducting an inductive
content analysis of 381 GitHub issues from two widely used SC fuzzers: Echidna
and Foundry. Furthermore, we conducted a user study to examine how these
challenges affect different practitioner groups, SC developers, and traditional
software security professionals, and identify strategies practitioners use to
overcome them. We systematically categorize these challenges into a taxonomy
based on their nature and occurrence within the SC fuzzing workflow. Our
findings reveal domain-specific ease-of-use and usefulness challenges,
including technical issues with blockchain emulation, and human issues with a
lack of accessible documentation and process automation. Our results provide
actionable insights for tool developers and researchers, guiding future
improvements in SC fuzzer tool design.

### 9. [Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](http://arxiv.org/pdf/2506.07667v1)

Authors: Prarabdh Shukla, Wei Yin Chong, Yash Patel, Brennan Schaffner, Danish Pruthi, Arjun Bhagoji

To meet the demands of content moderation, online platforms have resorted to
automated systems. Newer forms of real-time engagement($\textit{e.g.}$, users
commenting on live streams) on platforms like Twitch exert additional pressures
on the latency expected of such moderation systems. Despite their prevalence,
relatively little is known about the effectiveness of these systems. In this
paper, we conduct an audit of Twitch's automated moderation tool
($\texttt{AutoMod}$) to investigate its effectiveness in flagging hateful
content. For our audit, we create streaming accounts to act as siloed test
beds, and interface with the live chat using Twitch's APIs to send over
$107,000$ comments collated from $4$ datasets. We measure $\texttt{AutoMod}$'s
accuracy in flagging blatantly hateful content containing misogyny, racism,
ableism and homophobia. Our experiments reveal that a large fraction of hateful
messages, up to $94\%$ on some datasets, $\textit{bypass moderation}$.
Contextual addition of slurs to these messages results in $100\%$ removal,
revealing $\texttt{AutoMod}$'s reliance on slurs as a moderation signal. We
also find that contrary to Twitch's community guidelines, $\texttt{AutoMod}$
blocks up to $89.5\%$ of benign examples that use sensitive words in
pedagogical or empowering contexts. Overall, our audit points to large gaps in
$\texttt{AutoMod}$'s capabilities and underscores the importance for such
systems to understand context effectively.

### Information Retrieval

### 1. [Leveraging Historical and Current Interests for Continual Sequential Recommendation](http://arxiv.org/pdf/2506.07466v1)

Authors: Gyuseok Lee, Hyunsik Yoo, Junyoung Hwang, SeongKu Kang, Hwanjo Yu

Sequential recommendation models based on the Transformer architecture show
superior performance in harnessing long-range dependencies within user behavior
via self-attention. However, naively updating them on continuously arriving
non-stationary data streams incurs prohibitive computation costs or leads to
catastrophic forgetting. To address this, we propose Continual Sequential
Transformer for Recommendation (CSTRec) that effectively leverages
well-preserved historical user interests while capturing current interests. At
its core is Continual Sequential Attention (CSA), a linear attention mechanism
that retains past knowledge without direct access to old data. CSA integrates
two key components: (1) Cauchy-Schwarz Normalization that stabilizes training
under uneven interaction frequencies, and (2) Collaborative Interest Enrichment
that mitigates forgetting through shared, learnable interest pools. We further
introduce a technique that facilitates learning for cold-start users by
transferring historical knowledge from behaviorally similar existing users.
Extensive experiments on three real-world datasets indicate that CSTRec
outperforms state-of-the-art baselines in both knowledge retention and
acquisition.

### 2. [Addressing Correlated Latent Exogenous Variables in Debiased Recommender Systems](http://arxiv.org/pdf/2506.07517v1)

Authors: Shuqiang Zhang, Yuchao Zhang, Jinkun Chen, Haochen Sui

Recommendation systems (RS) aim to provide personalized content, but they
face a challenge in unbiased learning due to selection bias, where users only
interact with items they prefer. This bias leads to a distorted representation
of user preferences, which hinders the accuracy and fairness of
recommendations. To address the issue, various methods such as error imputation
based, inverse propensity scoring, and doubly robust techniques have been
developed. Despite the progress, from the structural causal model perspective,
previous debiasing methods in RS assume the independence of the exogenous
variables. In this paper, we release this assumption and propose a learning
algorithm based on likelihood maximization to learn a prediction model. We
first discuss the correlation and difference between unmeasured confounding and
our scenario, then we propose a unified method that effectively handles latent
exogenous variables. Specifically, our method models the data generation
process with latent exogenous variables under mild normality assumptions. We
then develop a Monte Carlo algorithm to numerically estimate the likelihood
function. Extensive experiments on synthetic datasets and three real-world
datasets demonstrate the effectiveness of our proposed method. The code is at
https://github.com/WallaceSUI/kdd25-background-variable.

### 3. [MoE-MLoRA for Multi-Domain CTR Prediction: Efficient Adaptation with Expert Specialization](http://arxiv.org/pdf/2506.07563v1)

Authors: Ken Yagel, Eyal German, Aviel Ben Siman Tov

Personalized recommendation systems must adapt to user interactions across
different domains. Traditional approaches like MLoRA apply a single adaptation
per domain but lack flexibility in handling diverse user behaviors. To address
this, we propose MoE-MLoRA, a mixture-of-experts framework where each expert is
first trained independently to specialize in its domain before a gating network
is trained to weight their contributions dynamically. We evaluate MoE-MLoRA
across eight CTR models on Movielens and Taobao, showing that it improves
performance in large-scale, dynamic datasets (+1.45 Weighed-AUC in Taobao-20)
but offers limited benefits in structured datasets with low domain diversity
and sparsity. Further analysis of the number of experts per domain reveals that
larger ensembles do not always improve performance, indicating the need for
model-aware tuning. Our findings highlight the potential of expert-based
architectures for multi-domain recommendation systems, demonstrating that
task-aware specialization and adaptive gating can enhance predictive accuracy
in complex environments. The implementation and code are available in our
GitHub repository.

### 4. [A Temporal FRBR/FRBRoo-Based Model for Component-Level Versioning of Legal Norms](http://arxiv.org/pdf/2506.07853v1)

Authors: Hudson de Martim

Effectively representing legal norms for automated processing is a critical
challenge, particularly in tracking the diachronic evolution of their
hierarchical components (e.g., articles, paragraphs). While foundational
frameworks like FRBR/FRBRoo and standards like Akoma Ntoso model legal
documents at a macro level, they lack native mechanisms for granular,
component-level versioning. This limitation hinders the deterministic
point-in-time reconstruction of legal texts, a fundamental capability for
reliable Legal Tech and AI applications. This paper proposes a structured,
temporal model that extends the FRBRoo framework to address this gap. It
introduces specialized subclasses of Expressio - Temporal Version (TV) and
Language Version (LV - to represent the state of a legal norm and its
linguistic variations at specific points in time. The model applies this same
paradigm hierarchically, introducing Component Work (CW), Component Temporal
Version (CTV), and Component Language Version (CLV) to track the lifecycle of
individual articles, paragraphs, and clauses. Using the Brazilian Federal
Constitution as a case study, the paper demonstrates how each amendment creates
new Component Temporal Versions for affected provisions, while unaffected
components retain their existing versions. This fine-grained, time-aware
architecture enables the precise, deterministic retrieval and reconstruction of
any part of a legal text as it existed on a specific date. The model provides a
robust foundation for developing advanced legal information systems, knowledge
graphs, and AI tools capable of accurate historical analysis and impact
assessment, overcoming the limitations of current generative models.

### 5. [LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking](http://arxiv.org/pdf/2506.07449v1)

Authors: Vahid Azizi, Fatemeh Koochaki

Recent advances in Large Language Models (LLMs) have driven their adoption in
recommender systems through Retrieval-Augmented Generation (RAG) frameworks.
However, existing RAG approaches predominantly rely on flat, similarity-based
retrieval that fails to leverage the rich relational structure inherent in
user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass,
end-to-end trainable framework that integrates personalized knowledge graph
context into LLM-based recommendation ranking. Our approach extends the
LlamaRec architecture by incorporating a lightweight user preference module
that dynamically identifies salient relation paths within a heterogeneous
knowledge graph constructed from user behavior and item metadata. These
personalized subgraphs are seamlessly integrated into prompts for a fine-tuned
Llama-2 model, enabling efficient and interpretable recommendations through a
unified inference step. Comprehensive experiments on ML-100K and Amazon Beauty
datasets demonstrate consistent and significant improvements over LlamaRec
across key ranking metrics (MRR, NDCG, Recall). LlamaRec-LKG-RAG demonstrates
the critical value of structured reasoning in LLM-based recommendations and
establishes a foundation for scalable, knowledge-aware personalization in
next-generation recommender systems. Code is available
at~\href{https://github.com/VahidAz/LlamaRec-LKG-RAG}{repository}.

### 6. [PolitiSky24: U.S. Political Bluesky Dataset with User Stance Labels](http://arxiv.org/pdf/2506.07606v1)

Authors: Peyman Rostami, Vahid Rahimzadeh, Ali Adibi, Azadeh Shakery

Stance detection identifies the viewpoint expressed in text toward a specific
target, such as a political figure. While previous datasets have focused
primarily on tweet-level stances from established platforms, user-level stance
resources, especially on emerging platforms like Bluesky remain scarce.
User-level stance detection provides a more holistic view by considering a
user's complete posting history rather than isolated posts. We present the
first stance detection dataset for the 2024 U.S. presidential election,
collected from Bluesky and centered on Kamala Harris and Donald Trump. The
dataset comprises 16,044 user-target stance pairs enriched with engagement
metadata, interaction graphs, and user posting histories. PolitiSky24 was
created using a carefully evaluated pipeline combining advanced information
retrieval and large language models, which generates stance labels with
supporting rationales and text spans for transparency. The labeling approach
achieves 81\% accuracy with scalable LLMs. This resource addresses gaps in
political stance analysis through its timeliness, open-data nature, and
user-level perspective. The dataset is available at
https://doi.org/10.5281/zenodo.15616911

### Machine Learning

### 1. [Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models](http://arxiv.org/pdf/2506.07334v1)

Authors: Haoyu Wang, Peihao Wang, Mufei Li, Shikun Liu, Siqi Miao, Zhangyang Wang, Pan Li

Modern large language models (LLMs) are inherently auto-regressive, requiring
input to be serialized into flat sequences regardless of their structural
dependencies. This serialization hinders the model's ability to leverage
structural inductive biases, especially in tasks such as retrieval-augmented
generation (RAG) and reasoning on data with native graph structures, where
inter-segment dependencies are crucial. We introduce Graph-KV with the
potential to overcome this limitation. Graph-KV leverages the KV-cache of text
segments as condensed representations and governs their interaction through
structural inductive biases. In this framework, 'target' segments selectively
attend only to the KV-caches of their designated 'source' segments, rather than
all preceding segments in a serialized sequence. This approach induces a
graph-structured block mask, sparsifying attention and enabling a
message-passing-like step within the LLM. Furthermore, strategically allocated
positional encodings for source and target segments reduce positional bias and
context window consumption. We evaluate Graph-KV across three scenarios: (1)
seven RAG benchmarks spanning direct inference, multi-hop reasoning, and
long-document understanding; (2) Arxiv-QA, a novel academic paper QA task with
full-text scientific papers structured as citation ego-graphs; and (3) paper
topic classification within a citation network. By effectively reducing
positional bias and harnessing structural inductive biases, Graph-KV
substantially outperforms baselines, including standard costly sequential
encoding, across various settings. Code and the Graph-KV data are publicly
available.

### 2. [RiemannFormer: A Framework for Attention in Curved Spaces](http://arxiv.org/pdf/2506.07405v1)

Authors: Zhongping Ji

This research endeavors to offer insights into unlocking the further
potential of transformer-based architectures. One of the primary motivations is
to offer a geometric interpretation for the attention mechanism in
transformers. In our framework, the attention mainly involves metric tensors,
tangent spaces, inner product, and how they relate to each other. These
quantities and structures at discrete positions are intricately interconnected
via the parallel transport of tangent vectors. To make the learning process
more efficient, we reduce the number of parameters through ingenious predefined
configurations. Moreover, we introduce an explicit mechanism to highlight a
neighborhood by attenuating the remote values, given that transformers
inherently neglect local inductive bias. Experimental results demonstrate that
our modules deliver significant performance improvements relative to the
baseline. More evaluation experiments on visual and large language models will
be launched successively.

### 3. [Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](http://arxiv.org/pdf/2506.07440v1)

Authors: Ruhan Wang, Zhiyong Wang, Chengkai Huang, Rui Wang, Tong Yu, Lina Yao, John C. S. Lui, Dongruo Zhou

For question-answering (QA) tasks, in-context learning (ICL) enables language
models to generate responses without modifying their parameters by leveraging
examples provided in the input. However, the effectiveness of ICL heavily
depends on the availability of high-quality examples, which are often scarce
due to data privacy constraints, annotation costs, and distribution
disparities. A natural solution is to utilize examples stored on client
devices, but existing approaches either require transmitting model parameters -
incurring significant communication overhead - or fail to fully exploit local
datasets, limiting their effectiveness. To address these challenges, we propose
Federated In-Context Learning (Fed-ICL), a general framework that enhances ICL
through an iterative, collaborative process. Fed-ICL progressively refines
responses by leveraging multi-round interactions between clients and a central
server, improving answer quality without the need to transmit model parameters.
We establish theoretical guarantees for the convergence of Fed-ICL and conduct
extensive experiments on standard QA benchmarks, demonstrating that our
proposed approach achieves strong performance while maintaining low
communication costs.

### 4. [Circumventing Backdoor Space via Weight Symmetry](http://arxiv.org/pdf/2506.07467v1)

Authors: Jie Peng, Hongwei Yang, Jing Zhao, Hengji Dong, Hui He, Weizhe Zhang, Haoyu He

Deep neural networks are vulnerable to backdoor attacks, where malicious
behaviors are implanted during training. While existing defenses can
effectively purify compromised models, they typically require labeled data or
specific training procedures, making them difficult to apply beyond supervised
learning settings. Notably, recent studies have shown successful backdoor
attacks across various learning paradigms, highlighting a critical security
concern. To address this gap, we propose Two-stage Symmetry Connectivity (TSC),
a novel backdoor purification defense that operates independently of data
format and requires only a small fraction of clean samples. Through theoretical
analysis, we prove that by leveraging permutation invariance in neural networks
and quadratic mode connectivity, TSC amplifies the loss on poisoned samples
while maintaining bounded clean accuracy. Experiments demonstrate that TSC
achieves robust performance comparable to state-of-the-art methods in
supervised learning scenarios. Furthermore, TSC generalizes to self-supervised
learning frameworks, such as SimCLR and CLIP, maintaining its strong defense
capabilities. Our code is available at https://github.com/JiePeng104/TSC.

### 5. [Improving Memory Efficiency for Training KANs via Meta Learning](http://arxiv.org/pdf/2506.07549v1)

Authors: Zhangchi Zhao, Jun Shu, Deyu Meng, Zongben Xu

Inspired by the Kolmogorov-Arnold representation theorem, KANs offer a novel
framework for function approximation by replacing traditional neural network
weights with learnable univariate functions. This design demonstrates
significant potential as an efficient and interpretable alternative to
traditional MLPs. However, KANs are characterized by a substantially larger
number of trainable parameters, leading to challenges in memory efficiency and
higher training costs compared to MLPs. To address this limitation, we propose
to generate weights for KANs via a smaller meta-learner, called MetaKANs. By
training KANs and MetaKANs in an end-to-end differentiable manner, MetaKANs
achieve comparable or even superior performance while significantly reducing
the number of trainable parameters and maintaining promising interpretability.
Extensive experiments on diverse benchmark tasks, including symbolic
regression, partial differential equation solving, and image classification,
demonstrate the effectiveness of MetaKANs in improving parameter efficiency and
memory usage. The proposed method provides an alternative technique for
training KANs, that allows for greater scalability and extensibility, and
narrows the training cost gap with MLPs stated in the original paper of KANs.
Our code is available at https://github.com/Murphyzc/MetaKAN.

### 6. [MIRA: Medical Time Series Foundation Model for Real-World Health Data](http://arxiv.org/pdf/2506.07584v1)

Authors: Hao Li, Bowen Deng, Chang Xu, Zhiyuan Feng, Viktor Schlegel, Yu-Hao Huang, Yizheng Sun, Jingyuan Sun, Kailai Yang, Yiyao Yu, Jiang Bian

A unified foundation model for medical time series -- pretrained on open
access and ethics board-approved medical corpora -- offers the potential to
reduce annotation burdens, minimize model customization, and enable robust
transfer across clinical institutions, modalities, and tasks, particularly in
data-scarce or privacy-constrained environments. However, existing generalist
time series foundation models struggle to handle medical time series data due
to their inherent challenges, including irregular intervals, heterogeneous
sampling rates, and frequent missing values. To address these challenges, we
introduce MIRA, a unified foundation model specifically designed for medical
time series forecasting. MIRA incorporates a Continuous-Time Rotary Positional
Encoding that enables fine-grained modeling of variable time intervals, a
frequency-specific mixture-of-experts layer that routes computation across
latent frequency regimes to further promote temporal specialization, and a
Continuous Dynamics Extrapolation Block based on Neural ODE that models the
continuous trajectory of latent states, enabling accurate forecasting at
arbitrary target timestamps. Pretrained on a large-scale and diverse medical
corpus comprising over 454 billion time points collect from publicly available
datasets, MIRA achieves reductions in forecasting errors by an average of 10%
and 7% in out-of-distribution and in-distribution scenarios, respectively, when
compared to other zero-shot and fine-tuned baselines. We also introduce a
comprehensive benchmark spanning multiple downstream clinical tasks,
establishing a foundation for future research in medical time series modeling.

### 7. [Aircraft Trajectory Dataset Augmentation in Latent Space](http://arxiv.org/pdf/2506.07585v1)

Authors: Seokbin Yoon, Keumjin Lee

Aircraft trajectory modeling plays a crucial role in Air Traffic Management
(ATM) and is important for various downstream tasks, including conflict
detection and landing time prediction. Dataset augmentation through the
addition of synthetically generated trajectory data is necessary to develop a
more robust aircraft trajectory model and ensure that the trajectory dataset is
sufficient and balanced. In this work, we propose a novel framework called
ATRADA for aircraft trajectory dataset augmentation. In the proposed framework,
a Transformer encoder learns the underlying patterns in the original trajectory
dataset and converts each data point into a context vector in the learned
latent space. The converted dataset in the latent space is projected into
reduced dimensions using principal component analysis (PCA), and a Gaussian
mixture model (GMM) is applied to fit the probability distribution of the data
points in the reduced-dimensional space. Finally, new samples are drawn from
the fitted GMM, the dimension of the samples is reverted to the original
dimension, and they are decoded with a Multi-Layer Perceptron (MLP). Several
experiments demonstrate that the framework effectively generates new,
high-quality synthetic aircraft trajectory data, which were compared to the
results of several baselines.

### 8. [TwinBreak: Jailbreaking LLM Security Alignments based on Twin Prompts](http://arxiv.org/pdf/2506.07596v1)

Authors: Torsten Krauß, Hamid Dashtbani, Alexandra Dmitrienko

Machine learning is advancing rapidly, with applications bringing notable
benefits, such as improvements in translation and code generation. Models like
ChatGPT, powered by Large Language Models (LLMs), are increasingly integrated
into daily life. However, alongside these benefits, LLMs also introduce social
risks. Malicious users can exploit LLMs by submitting harmful prompts, such as
requesting instructions for illegal activities. To mitigate this, models often
include a security mechanism that automatically rejects such harmful prompts.
However, they can be bypassed through LLM jailbreaks. Current jailbreaks often
require significant manual effort, high computational costs, or result in
excessive model modifications that may degrade regular utility.
  We introduce TwinBreak, an innovative safety alignment removal method.
Building on the idea that the safety mechanism operates like an embedded
backdoor, TwinBreak identifies and prunes parameters responsible for this
functionality. By focusing on the most relevant model layers, TwinBreak
performs fine-grained analysis of parameters essential to model utility and
safety. TwinBreak is the first method to analyze intermediate outputs from
prompts with high structural and content similarity to isolate safety
parameters. We present the TwinPrompt dataset containing 100 such twin prompts.
Experiments confirm TwinBreak's effectiveness, achieving 89% to 98% success
rates with minimal computational requirements across 16 LLMs from five vendors.

### 9. [FuXi-Air: Urban Air Quality Forecasting Based on Emission-Meteorology-Pollutant multimodal Machine Learning](http://arxiv.org/pdf/2506.07616v1)

Authors: Zhixin Geng, Xu Fan, Xiqiao Lu, Yan Zhang, Guangyuan Yu, Cheng Huang, Qian Wang, Yuewu Li, Weichun Ma, Qi Yu, Libo Wu, Hao Li

Air pollution has emerged as a major public health challenge in megacities.
Numerical simulations and single-site machine learning approaches have been
widely applied in air quality forecasting tasks. However, these methods face
multiple limitations, including high computational costs, low operational
efficiency, and limited integration with observational data. With the rapid
advancement of artificial intelligence, there is an urgent need to develop a
low-cost, efficient air quality forecasting model for smart urban management.
An air quality forecasting model, named FuXi-Air, has been constructed in this
study based on multimodal data fusion to support high-precision air quality
forecasting and operated in typical megacities. The model integrates
meteorological forecasts, emission inventories, and pollutant monitoring data
under the guidance of air pollution mechanism. By combining an autoregressive
prediction framework with a frame interpolation strategy, the model
successfully completes 72-hour forecasts for six major air pollutants at an
hourly resolution across multiple monitoring sites within 25-30 seconds. In
terms of both computational efficiency and forecasting accuracy, it outperforms
the mainstream numerical air quality models in operational forecasting work.
Ablation experiments concerning key influencing factors show that although
meteorological data contribute more to model accuracy than emission inventories
do, the integration of multimodal data significantly improves forecasting
precision and ensures that reliable predictions are obtained under differing
pollution mechanisms across megacities. This study provides both a technical
reference and a practical example for applying multimodal data-driven models to
air quality forecasting and offers new insights into building hybrid
forecasting systems to support air pollution risk warning in smart city
management.

### 10. [Return of ChebNet: Understanding and Improving an Overlooked GNN on Long Range Tasks](http://arxiv.org/pdf/2506.07624v1)

Authors: Ali Hariri, Álvaro Arroyo, Alessio Gravina, Moshe Eliasof, Carola-Bibiane Schönlieb, Davide Bacciu, Kamyar Azizzadenesheli, Xiaowen Dong, Pierre Vandergheynst

ChebNet, one of the earliest spectral GNNs, has largely been overshadowed by
Message Passing Neural Networks (MPNNs), which gained popularity for their
simplicity and effectiveness in capturing local graph structure. Despite their
success, MPNNs are limited in their ability to capture long-range dependencies
between nodes. This has led researchers to adapt MPNNs through rewiring or make
use of Graph Transformers, which compromises the computational efficiency that
characterized early spatial message-passing architectures, and typically
disregards the graph structure. Almost a decade after its original
introduction, we revisit ChebNet to shed light on its ability to model distant
node interactions. We find that out-of-box, ChebNet already shows competitive
advantages relative to classical MPNNs and GTs on long-range benchmarks, while
maintaining good scalability properties for high-order polynomials. However, we
uncover that this polynomial expansion leads ChebNet to an unstable regime
during training. To address this limitation, we cast ChebNet as a stable and
non-dissipative dynamical system, which we coin Stable-ChebNet. Our
Stable-ChebNet model allows for stable information propagation, and has
controllable dynamics which do not require the use of eigendecompositions,
positional encodings, or graph rewiring. Across several benchmarks,
Stable-ChebNet achieves near state-of-the-art performance.

### Neural and Evolutionary Computing

### 1. [REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models](http://arxiv.org/pdf/2506.07759v1)

Authors: Diego Forniés-Tabuenca, Alejandro Uribe, Urtzi Otamendi, Arkaitz Artetxe, Juan Carlos Rivera, Oier Lopez de Lacalle

Multi-objective optimization is fundamental in complex decision-making tasks.
Traditional algorithms, while effective, often demand extensive
problem-specific modeling and struggle to adapt to nonlinear structures. Recent
advances in Large Language Models (LLMs) offer enhanced explainability,
adaptability, and reasoning. This work proposes Reflective Evolution of
Multi-objective Heuristics (REMoH), a novel framework integrating NSGA-II with
LLM-based heuristic generation. A key innovation is a reflection mechanism that
uses clustering and search-space reflection to guide the creation of diverse,
high-quality heuristics, improving convergence and maintaining solution
diversity. The approach is evaluated on the Flexible Job Shop Scheduling
Problem (FJSSP) in-depth benchmarking against state-of-the-art methods using
three instance datasets: Dauzere, Barnes, and Brandimarte. Results demonstrate
that REMoH achieves competitive results compared to state-of-the-art approaches
with reduced modeling effort and enhanced adaptability. These findings
underscore the potential of LLMs to augment traditional optimization, offering
greater flexibility, interpretability, and robustness in multi-objective
scenarios.

### 2. [Generative Voice Bursts during Phone Call](http://arxiv.org/pdf/2506.07526v1)

Authors: Paritosh Ranjan, Surajit Majumder, Prodip Roy

In critical situations, conventional mobile telephony fails to convey
emergency voice messages to a callee already engaged in another call. The
standard call waiting alert does not provide the urgency or content of the
waiting call. This paper proposes a novel method for transmitting Generative
Voice Bursts short, context aware audio messages during ongoing calls, from
either preauthorized or dynamically prioritized callers. By leveraging
generative AI techniques, the system automatically generates spoken messages
from contextual inputs example like location, health data, images, background
noise when the caller is unable to speak due to incapacitation or environmental
constraints. The solution incorporates voice, text, and priority inference
mechanisms, allowing high priority emergency messages to bypass conventional
call waiting barriers. The approach employs models such as GPT Neo for
generative text, which is synthesized into audio and delivered in configurable
intervals G seconds and counts N times, ensuring minimal disruption while
preserving urgency. This method holds potential for significant impact across
telecom, mobile device manufacturing, and emergency communication platforms.

### Networking and Internet Architecture

### 1. [Diffusion-RL for Scalable Resource Allocation for 6G Networks](http://arxiv.org/pdf/2506.07880v1)

Authors: Salar Nouri, Mojdeh Karbalaee Motalleb, Vahid Shah-Mansouri

This paper presents a novel approach to resource allocation in Open Radio
Access Networks (O-RAN), leveraging a Generative AI technique with network
slicing to address the diverse demands of 5G and 6G service types such as
Enhanced Mobile Broadband (eMBB), Ultra-Reliable Low-Latency Communications
(URLLC), and Massive Machine-Type Communications (mMTC). Additionally, we
provide a comprehensive analysis and comparison of machine learning (ML)
techniques for resource allocation within O-RAN, evaluating their effectiveness
in optimizing network performance. We introduce a diffusion-based reinforcement
learning (Diffusion-RL) algorithm designed to optimize the allocation of
physical resource blocks (PRBs) and power consumption, thereby maximizing
weighted throughput and minimizing the delay for user equipment (UE). The
Diffusion-RL model incorporates controlled noise and perturbations to explore
optimal resource distribution while meeting each service type's Quality of
Service (QoS) requirements. We evaluate the performance of our proposed method
against several benchmarks, including an exhaustive search algorithm, deep
Q-networks (DQN), and the Semi-Supervised Variational Autoencoder (SS-VAE).
Comprehensive metrics, such as throughput and latency, are presented for each
service type. Experimental results demonstrate that the Diffusion-based RL
approach outperforms existing methods in efficiency, scalability, and
robustness, offering a promising solution for resource allocation in dynamic
and heterogeneous O-RAN environments with significant implications for future
6G networks.

### 2. [SALT: A Lightweight Model Adaptation Method for Closed Split Computing Environments](http://arxiv.org/pdf/2506.07355v1)

Authors: Yuya Okada, Takayuki Nishio

We propose SALT (Split-Adaptive Lightweight Tuning), a lightweight model
adaptation framework for Split Computing under closed constraints, where the
head and tail networks are proprietary and inaccessible to users. In such
closed environments, conventional adaptation methods are infeasible since they
require access to model parameters or architectures. SALT addresses this
challenge by introducing a compact, trainable adapter on the client side to
refine latent features from the head network, enabling user-specific adaptation
without modifying the original models or increasing communication overhead. We
evaluate SALT on user-specific classification tasks with CIFAR-10 and
CIFAR-100, demonstrating improved accuracy with lower training latency compared
to fine-tuning methods. Furthermore, SALT facilitates model adaptation for
robust inference over lossy networks, a common challenge in edge-cloud
environments. With minimal deployment overhead, SALT offers a practical
solution for personalized inference in edge AI systems under strict system
constraints.

### 3. [Delay Optimization in Remote ID-Based UAV Communication via BLE and Wi-Fi Switching](http://arxiv.org/pdf/2506.07715v1)

Authors: Yian Zhu, Ziye Jia, Lei Zhang, Yao Wu, Qiuming Zhu, Qihui Wu

The remote identification (Remote ID) broadcast capability allows unmanned
aerial vehicles (UAVs) to exchange messages, which is a pivotal technology for
inter-UAV communications. Although this capability enhances the operational
visibility, low delay in Remote ID-based communications is critical for
ensuring the efficiency and timeliness of multi-UAV operations in dynamic
environments. To address this challenge, we first establish delay models for
Remote ID communications by considering packet reception and collisions across
both BLE 4 and Wi-Fi protocols. Building upon these models, we formulate an
optimization problem to minimize the long-term communication delay through
adaptive protocol selection. Since the delay performance varies with the UAV
density, we propose an adaptive BLE/Wi-Fi switching algorithm based on the
multi-agent deep Q-network approach. Experimental results demonstrate that in
dynamic-density scenarios, our strategy achieves 32.1% and 37.7% lower latency
compared to static BLE 4 and Wi-Fi modes respectively.

### 4. [Are Trees Really Green? A Detection Approach of IoT Malware Attacks](http://arxiv.org/pdf/2506.07836v1)

Authors: Silvia Lucia Sanna, Diego Soi, Davide Maiorca, Giorgio Giacinto

Nowadays, the Internet of Things (IoT) is widely employed, and its usage is
growing exponentially because it facilitates remote monitoring, predictive
maintenance, and data-driven decision making, especially in the healthcare and
industrial sectors. However, IoT devices remain vulnerable due to their
resource constraints and difficulty in applying security patches. Consequently,
various cybersecurity attacks are reported daily, such as Denial of Service,
particularly in IoT-driven solutions. Most attack detection methodologies are
based on Machine Learning (ML) techniques, which can detect attack patterns.
However, the focus is more on identification rather than considering the impact
of ML algorithms on computational resources. This paper proposes a green
methodology to identify IoT malware networking attacks based on flow
privacy-preserving statistical features. In particular, the hyperparameters of
three tree-based models -- Decision Trees, Random Forest and Extra-Trees -- are
optimized based on energy consumption and test-time performance in terms of
Matthew's Correlation Coefficient. Our results show that models maintain high
performance and detection accuracy while consistently reducing power usage in
terms of watt-hours (Wh). This suggests that on-premise ML-based Intrusion
Detection Systems are suitable for IoT and other resource-constrained devices.

### Robotics

### 1. [Reproducibility in the Control of Autonomous Mobility-on-Demand Systems](http://arxiv.org/pdf/2506.07345v1)

Authors: Xinling Li, Meshal Alharbi, Daniele Gammelli, James Harrison, Filipe Rodrigues, Maximilian Schiffer, Marco Pavone, Emilio Frazzoli, Jinhua Zhao, Gioele Zardini

Autonomous Mobility-on-Demand (AMoD) systems, powered by advances in
robotics, control, and Machine Learning (ML), offer a promising paradigm for
future urban transportation. AMoD offers fast and personalized travel services
by leveraging centralized control of autonomous vehicle fleets to optimize
operations and enhance service performance. However, the rapid growth of this
field has outpaced the development of standardized practices for evaluating and
reporting results, leading to significant challenges in reproducibility. As
AMoD control algorithms become increasingly complex and data-driven, a lack of
transparency in modeling assumptions, experimental setups, and algorithmic
implementation hinders scientific progress and undermines confidence in the
results. This paper presents a systematic study of reproducibility in AMoD
research. We identify key components across the research pipeline, spanning
system modeling, control problems, simulation design, algorithm specification,
and evaluation, and analyze common sources of irreproducibility. We survey
prevalent practices in the literature, highlight gaps, and propose a structured
framework to assess and improve reproducibility. Specifically, concrete
guidelines are offered, along with a "reproducibility checklist", to support
future work in achieving replicable, comparable, and extensible results. While
focused on AMoD, the principles and practices we advocate generalize to a
broader class of cyber-physical systems that rely on networked autonomy and
data-driven control. This work aims to lay the foundation for a more
transparent and reproducible research culture in the design and deployment of
intelligent mobility systems.

### 2. [RAPID Hand: A Robust, Affordable, Perception-Integrated, Dexterous Manipulation Platform for Generalist Robot Autonomy](http://arxiv.org/pdf/2506.07490v1)

Authors: Zhaoliang Wan, Zetong Bi, Zida Zhou, Hao Ren, Yiming Zeng, Yihan Li, Lu Qi, Xu Yang, Ming-Hsuan Yang, Hui Cheng

This paper addresses the scarcity of low-cost but high-dexterity platforms
for collecting real-world multi-fingered robot manipulation data towards
generalist robot autonomy. To achieve it, we propose the RAPID Hand, a
co-optimized hardware and software platform where the compact 20-DoF hand,
robust whole-hand perception, and high-DoF teleoperation interface are jointly
designed. Specifically, RAPID Hand adopts a compact and practical hand ontology
and a hardware-level perception framework that stably integrates wrist-mounted
vision, fingertip tactile sensing, and proprioception with sub-7 ms latency and
spatial alignment. Collecting high-quality demonstrations on high-DoF hands is
challenging, as existing teleoperation methods struggle with precision and
stability on complex multi-fingered systems. We address this by co-optimizing
hand design, perception integration, and teleoperation interface through a
universal actuation scheme, custom perception electronics, and two retargeting
constraints. We evaluate the platform's hardware, perception, and teleoperation
interface. Training a diffusion policy on collected data shows superior
performance over prior works, validating the system's capability for reliable,
high-quality data collection. The platform is constructed from low-cost and
off-the-shelf components and will be made public to ensure reproducibility and
ease of adoption.

### 3. [Taking Flight with Dialogue: Enabling Natural Language Control for PX4-based Drone Agent](http://arxiv.org/pdf/2506.07509v1)

Authors: Shoon Kit Lim, Melissa Jia Ying Chong, Jing Huey Khor, Ting Yang Ling

Recent advances in agentic and physical artificial intelligence (AI) have
largely focused on ground-based platforms such as humanoid and wheeled robots,
leaving aerial robots relatively underexplored. Meanwhile, state-of-the-art
unmanned aerial vehicle (UAV) multimodal vision-language systems typically rely
on closed-source models accessible only to well-resourced organizations. To
democratize natural language control of autonomous drones, we present an
open-source agentic framework that integrates PX4-based flight control, Robot
Operating System 2 (ROS 2) middleware, and locally hosted models using Ollama.
We evaluate performance both in simulation and on a custom quadcopter platform,
benchmarking four large language model (LLM) families for command generation
and three vision-language model (VLM) families for scene understanding.

### 4. [Blending Participatory Design and Artificial Awareness for Trustworthy Autonomous Vehicles](http://arxiv.org/pdf/2506.07633v1)

Authors: Ana Tanevska, Ananthapathmanabhan Ratheesh Kumar, Arabinda Ghosh, Ernesto Casablanca, Ginevra Castellano, Sadegh Soudjani

Current robotic agents, such as autonomous vehicles (AVs) and drones, need to
deal with uncertain real-world environments with appropriate situational
awareness (SA), risk awareness, coordination, and decision-making. The SymAware
project strives to address this issue by designing an architecture for
artificial awareness in multi-agent systems, enabling safe collaboration of
autonomous vehicles and drones. However, these agents will also need to
interact with human users (drivers, pedestrians, drone operators), which in
turn requires an understanding of how to model the human in the interaction
scenario, and how to foster trust and transparency between the agent and the
human.
  In this work, we aim to create a data-driven model of a human driver to be
integrated into our SA architecture, grounding our research in the principles
of trustworthy human-agent interaction. To collect the data necessary for
creating the model, we conducted a large-scale user-centered study on human-AV
interaction, in which we investigate the interaction between the AV's
transparency and the users' behavior.
  The contributions of this paper are twofold: First, we illustrate in detail
our human-AV study and its findings, and second we present the resulting Markov
chain models of the human driver computed from the study's data. Our results
show that depending on the AV's transparency, the scenario's environment, and
the users' demographics, we can obtain significant differences in the model's
transitions.

### 5. [Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse](http://arxiv.org/pdf/2506.07639v1)

Authors: Zhekai Duan, Yuan Zhang, Shikai Geng, Gaowen Liu, Joschka Boedecker, Chris Xiaoxuan Lu

Embodied Chain-of-Thought (ECoT) reasoning enhances vision-language-action
(VLA) models by improving performance and interpretability through intermediate
reasoning steps. However, its sequential autoregressive token generation
introduces significant inference latency, limiting real-time deployment. We
propose Fast ECoT, an inference-time acceleration method that exploits the
structured and repetitive nature of ECoT to (1) cache and reuse high-level
reasoning across timesteps and (2) parallelise the generation of modular
reasoning steps. Additionally, we introduce an asynchronous scheduler that
decouples reasoning from action decoding, further boosting responsiveness. Fast
ECoT requires no model changes or additional training and integrates easily
into existing VLA pipelines. Experiments in both simulation (LIBERO) and
real-world robot tasks show up to a 7.5% reduction in latency with comparable
or improved task success rate and reasoning faithfulness, bringing ECoT
policies closer to practical real-time deployment.

### 6. [A Communication-Latency-Aware Co-Simulation Platform for Safety and Comfort Evaluation of Cloud-Controlled ICVs](http://arxiv.org/pdf/2506.07696v1)

Authors: Yongqi Zhao, Xinrui Zhang, Tomislav Mihalj, Martin Schabauer, Luis Putzer, Erik Reichmann-Blaga, Ádám Boronyák, András Rövid, Gábor Soós, Peizhi Zhang, Lu Xiong, Jia Hu, Arno Eichberger

Testing cloud-controlled intelligent connected vehicles (ICVs) requires
simulation environments that faithfully emulate both vehicle behavior and
realistic communication latencies. This paper proposes a latency-aware
co-simulation platform integrating CarMaker and Vissim to evaluate safety and
comfort under real-world vehicle-to-cloud (V2C) latency conditions. Two
communication latency models, derived from empirical 5G measurements in China
and Hungary, are incorporated and statistically modeled using Gamma
distributions. A proactive conflict module (PCM) is proposed to dynamically
control background vehicles and generate safety-critical scenarios. The
platform is validated through experiments involving an exemplary system under
test (SUT) across six testing conditions combining two PCM modes
(enabled/disabled) and three latency conditions (none, China, Hungary). Safety
and comfort are assessed using metrics including collision rate, distance
headway, post-encroachment time, and the spectral characteristics of
longitudinal acceleration. Results show that the PCM effectively increases
driving environment criticality, while V2C latency primarily affects ride
comfort. These findings confirm the platform's effectiveness in systematically
evaluating cloud-controlled ICVs under diverse testing conditions.

### 7. [Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots](http://arxiv.org/pdf/2506.07823v1)

Authors: Lorenzo Amatucci, João Sousa-Pinto, Giulio Turrisi, Dominique Orban, Victor Barasuol, Claudio Semini

This paper introduces a novel Model Predictive Control (MPC) implementation
for legged robot locomotion that leverages GPU parallelization. Our approach
enables both temporal and state-space parallelization by incorporating a
parallel associative scan to solve the primal-dual Karush-Kuhn-Tucker (KKT)
system. In this way, the optimal control problem is solved in
$\mathcal{O}(n\log{N} + m)$ complexity, instead of $\mathcal{O}(N(n + m)^3)$,
where $n$, $m$, and $N$ are the dimension of the system state, control vector,
and the length of the prediction horizon. We demonstrate the advantages of this
implementation over two state-of-the-art solvers (acados and crocoddyl),
achieving up to a 60\% improvement in runtime for Whole Body Dynamics (WB)-MPC
and a 700\% improvement for Single Rigid Body Dynamics (SRBD)-MPC when varying
the prediction horizon length. The presented formulation scales efficiently
with the problem state dimensions as well, enabling the definition of a
centralized controller for up to 16 legged robots that can be computed in less
than 25 ms. Furthermore, thanks to the JAX implementation, the solver supports
large-scale parallelization across multiple environments, allowing the
possibility of performing learning with the MPC in the loop directly in GPU.

### 8. [Design and Implementation of a Peer-to-Peer Communication, Modular and Decentral YellowCube UUV](http://arxiv.org/pdf/2506.07924v1)

Authors: Zhizun Xu, Baozhu Jia, Weichao Shi

The underwater Unmanned Vehicles(UUVs) are pivot tools for offshore
engineering and oceanographic research. Most existing UUVs do not facilitate
easy integration of new or upgraded sensors. A solution to this problem is to
have a modular UUV system with changeable payload sections capable of carrying
different sensor to suite different missions. The design and implementation of
a modular and decentral UUV named YellowCube is presented in the paper. Instead
a centralised software architecture which is adopted by the other modular
underwater vehicles designs, a Peer-To-Peer(P2P) communication mechanism is
implemented among the UUV's modules. The experiments in the laboratory and sea
trials have been executed to verify the performances of the UUV.

### 9. [Hierarchical Scoring with 3D Gaussian Splatting for Instance Image-Goal Navigation](http://arxiv.org/pdf/2506.07338v1)

Authors: Yijie Deng, Shuaihang Yuan, Geeta Chandra Raju Bethala, Anthony Tzes, Yu-Shen Liu, Yi Fang

Instance Image-Goal Navigation (IIN) requires autonomous agents to identify
and navigate to a target object or location depicted in a reference image
captured from any viewpoint. While recent methods leverage powerful novel view
synthesis (NVS) techniques, such as three-dimensional Gaussian splatting
(3DGS), they typically rely on randomly sampling multiple viewpoints or
trajectories to ensure comprehensive coverage of discriminative visual cues.
This approach, however, creates significant redundancy through overlapping
image samples and lacks principled view selection, substantially increasing
both rendering and comparison overhead. In this paper, we introduce a novel IIN
framework with a hierarchical scoring paradigm that estimates optimal
viewpoints for target matching. Our approach integrates cross-level semantic
scoring, utilizing CLIP-derived relevancy fields to identify regions with high
semantic similarity to the target object class, with fine-grained local
geometric scoring that performs precise pose estimation within promising
regions. Extensive evaluations demonstrate that our method achieves
state-of-the-art performance on simulated IIN benchmarks and real-world
applicability.

### 10. [MapBERT: Bitwise Masked Modeling for Real-Time Semantic Mapping Generation](http://arxiv.org/pdf/2506.07350v1)

Authors: Yijie Deng, Shuaihang Yuan, Congcong Wen, Hao Huang, Anthony Tzes, Geeta Chandra Raju Bethala, Yi Fang

Spatial awareness is a critical capability for embodied agents, as it enables
them to anticipate and reason about unobserved regions. The primary challenge
arises from learning the distribution of indoor semantics, complicated by
sparse, imbalanced object categories and diverse spatial scales. Existing
methods struggle to robustly generate unobserved areas in real time and do not
generalize well to new environments. To this end, we propose \textbf{MapBERT},
a novel framework designed to effectively model the distribution of unseen
spaces. Motivated by the observation that the one-hot encoding of semantic maps
aligns naturally with the binary structure of bit encoding, we, for the first
time, leverage a lookup-free BitVAE to encode semantic maps into compact
bitwise tokens. Building on this, a masked transformer is employed to infer
missing regions and generate complete semantic maps from limited observations.
To enhance object-centric reasoning, we propose an object-aware masking
strategy that masks entire object categories concurrently and pairs them with
learnable embeddings, capturing implicit relationships between object
embeddings and spatial tokens. By learning these relationships, the model more
effectively captures indoor semantic distributions crucial for practical
robotic tasks. Experiments on Gibson benchmarks show that MapBERT achieves
state-of-the-art semantic map generation, balancing computational efficiency
with accurate reconstruction of unobserved regions.

### Software Engineering

### 1. [GUIPilot: A Consistency-based Mobile GUI Testing Approach for Detecting Application-specific Bugs](http://arxiv.org/pdf/2506.07385v1)

Authors: Ruofan Liu, Xiwen Teoh, Yun Lin, Guanjie Chen, Ruofei Ren, Denys Poshyvanyk, Jin Song Dong

In this work, we propose GUIPilot, an approach for detecting inconsistencies
between the mobile design and their implementations. The mobile design usually
consists of design mock-ups that specify (1) the expected screen appearances
(e.g., widget layouts, colors, and shapes) and (2) the expected screen
behaviors, regarding how one screen can transition into another (e.g., labeled
widgets with textual description). Given a design mock-up and the
implementation of its application, GUIPilot reports both their screen
inconsistencies as well as process inconsistencies. On the one hand, GUIPilot
detects the screen inconsistencies by abstracting every screen into a widget
container where each widget is represented by its position, width, height, and
type. By defining the partial order of widgets and the costs of replacing,
inserting, and deleting widgets in a screen, we convert the screen-matching
problem into an optimizable widget alignment problem. On the other hand, we
translate the specified GUI transition into stepwise actions on the mobile
screen (e.g., click, long-press, input text on some widgets). To this end, we
propose a visual prompt for the vision-language model to infer widget-specific
actions on the screen. By this means, we can validate the presence or absence
of expected transitions in the implementation. Our extensive experiments on 80
mobile applications and 160 design mock-ups show that (1) GUIPilot can achieve
94.5% precision and 99.6% recall in detecting screen inconsistencies,
outperforming the state-of-the-art approach, such as GVT, by 66.2% and 56.6%
respectively, and (2) GUIPilot reports zero errors in detecting process
inconsistencies. Furthermore, our industrial case study on applying GUIPilot on
a trading mobile application shows that GUIPilot has detected nine application
bugs, and all the bugs were confirmed by the original application experts.

### 2. [Generate Realistic Test Scenes for V2X Communication Systems](http://arxiv.org/pdf/2506.07419v1)

Authors: An Guo, Xinyu Gao, Chunrong Fang, Haoxiang Tian, Weisong Sun, Yanzhou Mu, Shuncheng Tang, Lei Ma, Zhenyu Chen

Accurately perceiving complex driving environments is essential for ensuring
the safe operation of autonomous vehicles. With the tremendous progress in deep
learning and communication technologies, cooperative perception with
Vehicle-to-Everything (V2X) technologies has emerged as a solution to overcome
the limitations of single-agent perception systems in perceiving distant
objects and occlusions. Despite the considerable advancements, V2X cooperative
perception systems require thorough testing and continuous enhancement of
system performance. Given that V2X driving scenes entail intricate
communications with multiple vehicles across various geographic locations,
creating V2X test scenes for these systems poses a significant challenge.
Moreover, current testing methodologies rely on manual data collection and
labeling, which are both time-consuming and costly.
  In this paper, we design and implement V2XGen, an automated testing
generation tool for V2X cooperative perception systems. V2XGen utilizes a
high-fidelity approach to generate realistic cooperative object instances and
strategically place them within the background data in crucial positions.
Furthermore, V2XGen adopts a fitness-guided V2X scene generation strategy for
the transformed scene generation process and improves testing efficiency. We
conduct experiments on V2XGen using multiple cooperative perception systems
with different fusion schemes to assess its performance on various tasks. The
experimental results demonstrate that V2XGen is capable of generating realistic
test scenes and effectively detecting erroneous behaviors in different
V2X-oriented driving conditions. Furthermore, the results validate that
retraining systems under test with the generated scenes can enhance average
detection precision while reducing occlusion and long-range perception errors.

### 3. [A Framework for Creating Non-Regressive Test Cases via Branch Consistency Analysis Driven by Descriptions](http://arxiv.org/pdf/2506.07486v1)

Authors: Yuxiang Zhang, Pengyu Xue, Zhen Yang, Xiaoxue Ren, Xiang Li, Linhao Wu, Jiancheng Zhao, Xingda Yu

Automated test-generation research overwhelmingly assumes the correctness of
focal methods, yet practitioners routinely face non-regression scenarios where
the focal method may be defective. A baseline evaluation of EvoSuite and two
leading Large Language Model (LLM)-based generators, namely ChatTester and
ChatUniTest, on defective focal methods reveals that despite achieving up to
83% of branch coverage, none of the generated tests expose defects.
  To resolve this problem, we first construct two new benchmarks, namely
Defects4J-Desc and QuixBugs-Desc, for experiments. In particular, each focal
method is equipped with an extra Natural Language Description (NLD) for code
functionality understanding.
  Subsequently, we propose DISTINCT, a Description-guided, branch-consistency
analysis framework that transforms LLMs into fault-aware test generators.
DISTINCT carries three iterative components: (1) a Generator that derives
initial tests based on the NLDs and the focal method, (2) a Validator that
iteratively fixes uncompilable tests using compiler diagnostics, and (3) an
Analyzer that iteratively aligns test behavior with NLD semantics via
branch-level analysis.
  Extensive experiments confirm the effectiveness of our approach. Compared to
state-of-the-art methods, DISTINCT achieves an average improvement of 14.64% in
Compilation Success Rate (CSR) and 6.66% in Passing Rate (PR) across both
benchmarks. It notably enhances Defect Detection Rate (DDR) on both benchmarks,
with a particularly significant gain of 149.26% observed on Defects4J-Desc. In
terms of code coverage, DISTINCT improves Statement Coverage (SC) by an average
of 3.77% and Branch Coverage (BC) by 5.36%. These results set a new baseline
for non-regressive test generation and highlight how description-driven
reasoning enables LLMs to move beyond coverage chasing toward effective defect
detection.

### 4. [Large Language Models for Multilingual Vulnerability Detection: How Far Are We?](http://arxiv.org/pdf/2506.07503v1)

Authors: Honglin Shu, Michael Fu, Junji Yu, Dong Wang, Chakkrit Tantithamthavorn, Junjie Chen, Yasutaka Kamei

Various deep learning-based approaches utilizing pre-trained language models
(PLMs) have been proposed for automated vulnerability detection. With recent
advancements in large language models (LLMs), several studies have begun
exploring their application to vulnerability detection tasks. However, existing
studies primarily focus on specific programming languages (e.g., C/C++) and
function-level detection, leaving the strengths and weaknesses of PLMs and LLMs
in multilingual and multi-granularity scenarios largely unexplored. To bridge
this gap, we conduct a comprehensive fine-grained empirical study evaluating
the effectiveness of state-of-the-art PLMs and LLMs for multilingual
vulnerability detection. Using over 30,000 real-world vulnerability-fixing
patches across seven programming languages, we systematically assess model
performance at both the function-level and line-level. Our key findings
indicate that GPT-4o, enhanced through instruction tuning and few-shot
prompting, significantly outperforms all other evaluated models, including
CodeT5P. Furthermore, the LLM-based approach demonstrates superior capability
in detecting unique multilingual vulnerabilities, particularly excelling in
identifying the most dangerous and high-severity vulnerabilities. These results
underscore the promising potential of adopting LLMs for multilingual
vulnerability detection at function-level and line-level, revealing their
complementary strengths and substantial improvements over PLM approaches. This
first empirical evaluation of PLMs and LLMs for multilingual vulnerability
detection highlights LLMs' value in addressing real-world software security
challenges.

### 5. [Evaluating LLMs Effectiveness in Detecting and Correcting Test Smells: An Empirical Study](http://arxiv.org/pdf/2506.07594v1)

Authors: E. G. Santana Jr, Jander Pereira Santos Junior, Erlon P. Almeida, Iftekhar Ahmed, Paulo Anselmo da Mota Silveira Neto, Eduardo Santana de Almeida

Test smells indicate poor development practices in test code, reducing
maintainability and reliability. While developers often struggle to prevent or
refactor these issues, existing tools focus primarily on detection rather than
automated refactoring. Large Language Models (LLMs) have shown strong potential
in code understanding and transformation, but their ability to both identify
and refactor test smells remains underexplored. We evaluated GPT-4-Turbo, LLaMA
3 70B, and Gemini-1.5 Pro on Python and Java test suites, using PyNose and
TsDetect for initial smell detection, followed by LLM-driven refactoring.
Gemini achieved the highest detection accuracy (74.35\% Python, 80.32\% Java),
while LLaMA was lowest. All models could refactor smells, but effectiveness
varied, sometimes introducing new smells. Gemini also improved test coverage,
unlike GPT-4 and LLaMA, which often reduced it. These results highlight LLMs'
potential for automated test smell refactoring, with Gemini as the strongest
performer, though challenges remain across languages and smell types.

### 6. [Adversarial Attack Classification and Robustness Testing for Large Language Models for Code](http://arxiv.org/pdf/2506.07942v1)

Authors: Yang Liu, Armstrong Foundjem, Foutse Khomh, Heng Li

Large Language Models (LLMs) have become vital tools in software development
tasks such as code generation, completion, and analysis. As their integration
into workflows deepens, ensuring robustness against vulnerabilities especially
those triggered by diverse or adversarial inputs becomes increasingly
important. Such vulnerabilities may lead to incorrect or insecure code
generation when models encounter perturbed task descriptions, code, or
comments. Prior research often overlooks the role of natural language in
guiding code tasks. This study investigates how adversarial perturbations in
natural language inputs including prompts, comments, and descriptions affect
LLMs for Code (LLM4Code). It examines the effects of perturbations at the
character, word, and sentence levels to identify the most impactful
vulnerabilities. We analyzed multiple projects (e.g., ReCode, OpenAttack) and
datasets (e.g., HumanEval, MBPP), establishing a taxonomy of adversarial
attacks. The first dimension classifies the input type code, prompts, or
comments while the second dimension focuses on granularity: character, word, or
sentence-level changes. We adopted a mixed-methods approach, combining
quantitative performance metrics with qualitative vulnerability analysis.
LLM4Code models show varying robustness across perturbation types.
Sentence-level attacks were least effective, suggesting models are resilient to
broader contextual changes. In contrast, word-level perturbations posed serious
challenges, exposing semantic vulnerabilities. Character-level effects varied,
showing model sensitivity to subtle syntactic deviations.Our study offers a
structured framework for testing LLM4Code robustness and emphasizes the
critical role of natural language in adversarial evaluation. Improving model
resilience to semantic-level disruptions is essential for secure and reliable
code-generation systems.

### 7. [Human Side of Smart Contract Fuzzing: An Empirical Study](http://arxiv.org/pdf/2506.07389v1)

Authors: Guanming Qiao, Partha Protim Paul

Smart contract (SC) fuzzing is a critical technique for detecting
vulnerabilities in blockchain applications. However, its adoption remains
challenging for practitioners due to fundamental differences between SCs and
traditional software systems. In this study, we investigate the challenges
practitioners face when adopting SC fuzzing tools by conducting an inductive
content analysis of 381 GitHub issues from two widely used SC fuzzers: Echidna
and Foundry. Furthermore, we conducted a user study to examine how these
challenges affect different practitioner groups, SC developers, and traditional
software security professionals, and identify strategies practitioners use to
overcome them. We systematically categorize these challenges into a taxonomy
based on their nature and occurrence within the SC fuzzing workflow. Our
findings reveal domain-specific ease-of-use and usefulness challenges,
including technical issues with blockchain emulation, and human issues with a
lack of accessible documentation and process automation. Our results provide
actionable insights for tool developers and researchers, guiding future
improvements in SC fuzzer tool design.

### 8. [Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](http://arxiv.org/pdf/2506.07390v1)

Authors: Xin-Cheng Wen, Yijun Yang, Cuiyun Gao, Yang Xiao, Deheng Ye

Large language models (LLMs) demonstrate considerable proficiency in numerous
coding-related tasks; however, their capabilities in detecting software
vulnerabilities remain limited. This limitation primarily stems from two
factors: (1) the absence of reasoning data related to vulnerabilities, which
hinders the models' ability to capture underlying vulnerability patterns; and
(2) their focus on learning semantic representations rather than the reason
behind them, thus failing to recognize semantically similar vulnerability
samples. Furthermore, the development of LLMs specialized in vulnerability
detection is challenging, particularly in environments characterized by the
scarcity of high-quality datasets. In this paper, we propose a novel framework
ReVD that excels at mining vulnerability patterns through reasoning data
synthesizing and vulnerability-specific preference optimization. Specifically,
we construct forward and backward reasoning processes for vulnerability and
corresponding fixed code, ensuring the synthesis of high-quality reasoning
data. Moreover, we design the triplet supervised fine-tuning followed by
curriculum online preference optimization for enabling ReVD to better
understand vulnerability patterns. The extensive experiments conducted on
PrimeVul and SVEN datasets demonstrate that ReVD sets new state-of-the-art for
LLM-based software vulnerability detection, e.g., 12.24\%-22.77\% improvement
in the accuracy. The source code and data are available at
https://github.com/Xin-Cheng-Wen/PO4Vul.

### 9. [Leveraging Network Methods for Hub-like Microservice Detection](http://arxiv.org/pdf/2506.07683v1)

Authors: Alexander Bakhtin, Matteo Esposito, Valentina Lenarduzzi, Davide Taibi

Context: Microservice Architecture is a popular architectural paradigm that
facilitates flexibility by decomposing applications into small, independently
deployable services. Catalogs of architectural anti-patterns have been proposed
to highlight the negative aspects of flawed microservice design. In particular,
the Hub-like anti-pattern lacks an unambiguous definition and detection method.
Aim: In this work, we aim to find a robust detection approach for the Hub-like
microservice anti-pattern that outputs a reasonable number of Hub-like
candidates with high precision. Method: We leveraged a dataset of 25
microservice networks and several network hub detection techniques to identify
the Hub-like anti-pattern, namely scale-free property, centrality metrics and
clustering coefficient, minimum description length principle, and the approach
behind the Arcan tool. Results and Conclusion: Our findings revealed that the
studied architectural networks are not scale-free, that most considered hub
detection approaches do not agree on the detected hubs, and that the method by
Kirkley leveraging the Erdos-Renyi encoding is the most accurate one in terms
of the number of detected hubs and the detection precision. Investigating
further the applicability of these methods to detecting Hub-like components in
microservice-based and other systems opens up new research directions.
Moreover, our results provide an evaluation of the approach utilized by the
widely used Arcan tool and highlight the potential to update the tool to use
the normalized degree centrality of a component in the network, or for the
approach based on ER encoding to be adopted instead.

### 10. [Towards a Small Language Model Lifecycle Framework](http://arxiv.org/pdf/2506.07695v1)

Authors: Parsa Miraghaei, Sergio Moreschini, Antti Kolehmainen, David Hästbacka

Background: The growing demand for efficient and deployable language models
has led to increased interest in Small Language Models (SLMs). However,
existing research remains fragmented, lacking a unified lifecycle perspective.
  Objective: This study aims to define a comprehensive lifecycle framework for
SLMs by synthesizing insights from academic literature and practitioner
sources.
  Method: We conducted a comprehensive survey of 36 works, analyzing and
categorizing lifecycle-relevant techniques.
  Results: We propose a modular lifecycle model structured into main, optional,
and cross-cutting components. The model captures key interconnections across
stages, supporting method reuse, co-adaptation, and lifecycle-awareness.
  Conclusion: Our framework provides a coherent foundation for developing and
maintaining SLMs, bridging theory and practice, and guiding future research and
tool development.

### Social and Information Networks

### 1. [Powers of Magnetic Graph Matrix: Fourier Spectrum, Walk Compression, and Applications](http://arxiv.org/pdf/2506.07343v1)

Authors: Yinan Huang, David F. Gleich, Pan Li

Magnetic graphs, originally developed to model quantum systems under magnetic
fields, have recently emerged as a powerful framework for analyzing complex
directed networks. Existing research has primarily used the spectral properties
of the magnetic graph matrix to study global and stationary network features.
However, their capacity to model local, non-equilibrium behaviors, often
described by matrix powers, remains largely unexplored. We present a novel
combinatorial interpretation of the magnetic graph matrix powers through
directed walk profiles -- counts of graph walks indexed by the number of edge
reversals. Crucially, we establish that walk profiles correspond to a Fourier
transform of magnetic matrix powers. The connection allows exact reconstruction
of walk profiles from magnetic matrix powers at multiple discrete potentials,
and more importantly, an even smaller number of potentials often suffices for
accurate approximate reconstruction in real networks. This shows the empirical
compressibility of the information captured by the magnetic matrix. This fresh
perspective suggests new applications; for example, we illustrate how powers of
the magnetic matrix can identify frustrated directed cycles (e.g., feedforward
loops) and can be effectively employed for link prediction by encoding local
structural details in directed graphs.

### 2. [Fast Geometric Embedding for Node Influence Maximization](http://arxiv.org/pdf/2506.07435v1)

Authors: Alexander Kolpakov, Igor Rivin

Computing classical centrality measures such as betweenness and closeness is
computationally expensive on large-scale graphs. In this work, we introduce an
efficient force layout algorithm that embeds a graph into a low-dimensional
space, where the radial distance from the origin serves as a proxy for various
centrality measures. We evaluate our method on multiple graph families and
demonstrate strong correlations with degree, PageRank, and paths-based
centralities. As an application, it turns out that the proposed embedding
allows to find high-influence nodes in a network, and provides a fast and
scalable alternative to the standard greedy algorithm.

### 3. [PolitiSky24: U.S. Political Bluesky Dataset with User Stance Labels](http://arxiv.org/pdf/2506.07606v1)

Authors: Peyman Rostami, Vahid Rahimzadeh, Ali Adibi, Azadeh Shakery

Stance detection identifies the viewpoint expressed in text toward a specific
target, such as a political figure. While previous datasets have focused
primarily on tweet-level stances from established platforms, user-level stance
resources, especially on emerging platforms like Bluesky remain scarce.
User-level stance detection provides a more holistic view by considering a
user's complete posting history rather than isolated posts. We present the
first stance detection dataset for the 2024 U.S. presidential election,
collected from Bluesky and centered on Kamala Harris and Donald Trump. The
dataset comprises 16,044 user-target stance pairs enriched with engagement
metadata, interaction graphs, and user posting histories. PolitiSky24 was
created using a carefully evaluated pipeline combining advanced information
retrieval and large language models, which generates stance labels with
supporting rationales and text spans for transparency. The labeling approach
achieves 81\% accuracy with scalable LLMs. This resource addresses gaps in
political stance analysis through its timeliness, open-data nature, and
user-level perspective. The dataset is available at
https://doi.org/10.5281/zenodo.15616911

### 4. [Refugees' path to legal stability is long and systematically unequal](http://arxiv.org/pdf/2506.07916v1)

Authors: Ola Ali, Elma Dervic, Guillermo Prieto-Viertel, Carsten Källner, Rainer Stütz, Andrea Vismara, Rafael Prieto-Curiel

Legal systems shape not only the recognition of migrants and refugees but
also the pace and stability of their integration. Refugees often shift between
multiple legal classifications, a process we refer to as the "legal journey".
This journey is frequently prolonged and uncertain. Using a network-based
approach, we analyze legal transitions for over 350,000 migrants in Austria
(2022 to 2024). Refugees face highly unequal pathways to stability, ranging
from two months for Ukrainians to nine months for Syrians and 20 months for
Afghans. Women, especially from these regions, are more likely to gain
protection; Afghan men wait up to 30 months on average. We also find that those
who cross the border without going through official border controls face higher
exit rates and lower chances of securing stable status. We show that legal
integration is not a uniform process, but one structured by institutional
design, procedural entry points, and unequal timelines.

### Systems and Control

### 1. [Extended Version of "Distributed Adaptive Resilient Consensus Control for Uncertain Nonlinear Multiagent Systems Against Deception Attacks"](http://arxiv.org/pdf/2506.07374v1)

Authors: Mengze Yu, Wei Wang, Jiaqi Yan

This paper studies distributed resilient consensus problem for a class of
uncertain nonlinear multiagent systems susceptible to deception attacks. The
attacks invade both sensor and actuator channels of each agent. A specific
class of Nussbaum functions is adopted to manage the attack-incurred multiple
unknown control directions. Additionally, a general form of these Nussbaum
functions is provided, which helps to ease the degeneration of output
performance caused by Nussbaum gains. Then, by introducing finite-time
distributed reference systems and local-error-based dynamic gains, we propose a
novel distributed adaptive backstepping-based resilient consensus control
strategy. We prove that all the closed-loop signals are uniformly bounded under
attacks, and output consensus errors converge in finite time to a
clearly-defined residual set whose size can be reduced by tuning control
parameters, which is superior to existing results. Simulation results display
the effectiveness of the proposed controllers.

### 2. [Pseudo-random sequences for low-cost operando impedance measurements of Li-ion batteries](http://arxiv.org/pdf/2506.07519v1)

Authors: Jussi Sihvo, Noël Hallemans, Ai Hui Tan, David A. Howey, Stephen. R. Duncan, Tomi Roinila

Operando impedance measurements are promising for monitoring batteries in the
field. In this work, we present pseudo-random sequences for low-cost operando
battery impedance measurements. The quadratic-residue ternary sequence and
direct-synthesis ternary sequence exhibit specific properties related to
eigenvectors of the discrete Fourier transform matrix that allow
computationally efficient compensation for drifts and transients in operando
impedance measurements. We describe the application of pseudo-random sequences
and provide the data processing required to suppress drift and transients,
validated on simulations. Finally, we perform experimental operando impedance
measurements on a Li-ion battery cell during fast-charging, demonstrating the
applicability of the proposed method. It's low-cost hardware requirements, fast
measurements, and simple data-processing make the method practical for
embedding in battery management systems.

### 3. [A 40.68-MHz, 200-ns-Settling Active Rectifier and TX-Side Load Monitoring for Minimizing Radiated Power in Biomedical Implants](http://arxiv.org/pdf/2506.07710v1)

Authors: Ronald Wijermars, Yi-Han Ou-Yang, Sijun Du, Dante Gabriel Muratore

This letter describes a 40.68 MHz wireless power transfer receiver for
implantable applications focused on minimizing tissue heating. The system
features a novel power radiated efficiency optimization strategy and a
fast-settling active rectifier that maintains high efficiency during load and
link variations required for downlink communication. The power radiated
efficiency optimization explicitly reduces tissue heating while enabling
transmitter-side load monitoring for closed-loop control. The active rectifier
was fabricated in 40nm CMOS and achieves a voltage conversion ratio of 93.9%
and a simulated power conversion efficiency of 90.1% in a 0.19 $mm^2$ area,
resulting in a 118 mW/$mm^2$ power density while integrating the resonance and
filter capacitors. The worst-case settling of the on- and off-delay
compensation in the active rectifier is 200 ns, which is the fastest reported
to date.

### 4. [A distributed motion planning approach to cooperative underwater acoustic source tracking and pursuit](http://arxiv.org/pdf/2506.07877v1)

Authors: Andrea Tiranti, Francesco Wanderlingh, Enrico Simetti, Marco Baglietto, Giovanni Indiveri, Antonio Pascoal

This paper addresses the problem of underwater acoustic source tracking and
pursuit with a team of autonomous underwater vehicles. Producing distributed
control strategies in an underwater sensor network is not trivial since
communication is primarily acoustic, which makes it intermittent and often
plagued with major difficulties. For this reason, we propose an optimization
scheme based on a Partially Observable Markov Decision Process for improving
the performance of underwater mobile sensor networks, in which autonomous
underwater vehicles (agents) play the role of moving nodes of a network. The
key idea is to adjust the agents' guidance strategies to achieve coordinated
motion planning, enabling optimal geometric configurations between the agents
and the target to enhance tracking performance. Such a problem is cast as a
multi-objective optimization problem that is solved through a receding horizon
lookahead optimization scheme since we are interested in long-term tracking
accuracy. The planning strategy is distributed using the sequential multi-agent
decision-making paradigm to make the solving tractable since the optimization
depends on the joint action domain. A distributed control framework has been
implemented in a simulation environment to validate the proposed approach,
which explicitly accounts for the major limitations imposed by acoustic
communications.

### 5. [Distributed Risk-Sensitive Safety Filters for Uncertain Discrete-Time Systems](http://arxiv.org/pdf/2506.07347v1)

Authors: Armin Lederer, Erfaun Noorani, Andreas Krause

Ensuring safety in multi-agent systems is a significant challenge,
particularly in settings where centralized coordination is impractical. In this
work, we propose a novel risk-sensitive safety filter for discrete-time
multi-agent systems with uncertain dynamics that leverages control barrier
functions (CBFs) defined through value functions. Our approach relies on
centralized risk-sensitive safety conditions based on exponential risk
operators to ensure robustness against model uncertainties. We introduce a
distributed formulation of the safety filter by deriving two alternative
strategies: one based on worst-case anticipation and another on proximity to a
known safe policy. By allowing agents to switch between strategies, feasibility
can be ensured. Through detailed numerical evaluations, we demonstrate the
efficacy of our approach in maintaining safety without being overly
conservative.

### 6. [UruBots Autonomous Cars Challenge Pro Team Description Paper for FIRA 2025](http://arxiv.org/pdf/2506.07348v1)

Authors: Pablo Moraes, Mónica Rodríguez, Sebastian Barcelona, Angel Da Silva, Santiago Fernandez, Hiago Sodre, Igor Nunes, Bruna Guterres, Ricardo Grando

This paper describes the development of an autonomous car by the UruBots team
for the 2025 FIRA Autonomous Cars Challenge (Pro). The project involves
constructing a compact electric vehicle, approximately the size of an RC car,
capable of autonomous navigation through different tracks. The design
incorporates mechanical and electronic components and machine learning
algorithms that enable the vehicle to make real-time navigation decisions based
on visual input from a camera. We use deep learning models to process camera
images and control vehicle movements. Using a dataset of over ten thousand
images, we trained a Convolutional Neural Network (CNN) to drive the vehicle
effectively, through two outputs, steering and throttle. The car completed the
track in under 30 seconds, achieving a pace of approximately 0.4 meters per
second while avoiding obstacles.

### 7. [Decentralized Optimization on Compact Submanifolds by Quantized Riemannian Gradient Tracking](http://arxiv.org/pdf/2506.07351v1)

Authors: Jun Chen, Lina Liu, Tianyi Zhu, Yong Liu, Guang Dai, Yunliang Jiang, Ivor W. Tsang

This paper considers the problem of decentralized optimization on compact
submanifolds, where a finite sum of smooth (possibly non-convex) local
functions is minimized by $n$ agents forming an undirected and connected graph.
However, the efficiency of distributed optimization is often hindered by
communication bottlenecks. To mitigate this, we propose the Quantized
Riemannian Gradient Tracking (Q-RGT) algorithm, where agents update their local
variables using quantized gradients. The introduction of quantization noise
allows our algorithm to bypass the constraints of the accurate Riemannian
projection operator (such as retraction), further improving iterative
efficiency. To the best of our knowledge, this is the first algorithm to
achieve an $\mathcal{O}(1/K)$ convergence rate in the presence of quantization,
matching the convergence rate of methods without quantization. Additionally, we
explicitly derive lower bounds on decentralized consensus associated with a
function of quantization levels. Numerical experiments demonstrate that Q-RGT
performs comparably to non-quantized methods while reducing communication
bottlenecks and computational overhead.

### 8. [Fractional Collisions: A Framework for Risk Estimation of Counterfactual Conflicts using Autonomous Driving Behavior Simulations](http://arxiv.org/pdf/2506.07540v1)

Authors: Sreeja Roy-Singh, Sarvesh Kolekar, Daniel P. Bonny, Kyle Foss

We present a methodology for estimating collision risk from counterfactual
simulated scenarios built on sensor data from automated driving systems (ADS)
or naturalistic driving databases. Two-agent conflicts are assessed by
detecting and classifying conflict type, identifying the agents' roles
(initiator or responder), identifying the point of reaction of the responder,
and modeling their human behavioral expectations as probabilistic
counterfactual trajectories. The states are used to compute velocity
differentials at collision, which when combined with crash models, estimates
severity of loss in terms of probabilistic injury or property damage,
henceforth called fractional collisions. The probabilistic models may also be
extended to include other uncertainties associated with the simulation,
features, and agents. We verify the effectiveness of the methodology in a
synthetic simulation environment using reconstructed trajectories from 300+
collision and near-collision scenes sourced from VTTI's SHRP2 database and
Nexar dashboard camera data. Our methodology predicted fractional collisions
within 1% of ground truth collisions. We then evaluate agent-initiated
collision risk of an arbitrary ADS software release by replacing the
naturalistic responder in these synthetic reconstructions with an ADS simulator
and comparing the outcome to human-response outcomes. Our ADS reduced
naturalistic collisions by 4x and fractional collision risk by ~62%. The
framework's utility is also demonstrated on 250k miles of proprietary,
open-loop sensor data collected on ADS test vehicles, re-simulated with an
arbitrary ADS software release. The ADS initiated conflicts that caused 0.4
injury-causing and 1.7 property-damaging fractional collisions, and the ADS
improved collision risk in 96% of the agent-initiated conflicts.

### 9. [Delay Optimization in Remote ID-Based UAV Communication via BLE and Wi-Fi Switching](http://arxiv.org/pdf/2506.07715v1)

Authors: Yian Zhu, Ziye Jia, Lei Zhang, Yao Wu, Qiuming Zhu, Qihui Wu

The remote identification (Remote ID) broadcast capability allows unmanned
aerial vehicles (UAVs) to exchange messages, which is a pivotal technology for
inter-UAV communications. Although this capability enhances the operational
visibility, low delay in Remote ID-based communications is critical for
ensuring the efficiency and timeliness of multi-UAV operations in dynamic
environments. To address this challenge, we first establish delay models for
Remote ID communications by considering packet reception and collisions across
both BLE 4 and Wi-Fi protocols. Building upon these models, we formulate an
optimization problem to minimize the long-term communication delay through
adaptive protocol selection. Since the delay performance varies with the UAV
density, we propose an adaptive BLE/Wi-Fi switching algorithm based on the
multi-agent deep Q-network approach. Experimental results demonstrate that in
dynamic-density scenarios, our strategy achieves 32.1% and 37.7% lower latency
compared to static BLE 4 and Wi-Fi modes respectively.

### 10. [Deep Equivariant Multi-Agent Control Barrier Functions](http://arxiv.org/pdf/2506.07755v1)

Authors: Nikolaos Bousias, Lars Lindemann, George Pappas

With multi-agent systems increasingly deployed autonomously at scale in
complex environments, ensuring safety of the data-driven policies is critical.
Control Barrier Functions have emerged as an effective tool for enforcing
safety constraints, yet existing learning-based methods often lack in
scalability, generalization and sampling efficiency as they overlook inherent
geometric structures of the system. To address this gap, we introduce
symmetries-infused distributed Control Barrier Functions, enforcing the
satisfaction of intrinsic symmetries on learnable graph-based safety
certificates. We theoretically motivate the need for equivariant
parametrization of CBFs and policies, and propose a simple, yet efficient and
adaptable methodology for constructing such equivariant group-modular networks
via the compatible group actions. This approach encodes safety constraints in a
distributed data-efficient manner, enabling zero-shot generalization to larger
and denser swarms. Through extensive simulations on multi-robot navigation
tasks, we demonstrate that our method outperforms state-of-the-art baselines in
terms of safety, scalability, and task success rates, highlighting the
importance of embedding symmetries in safe distributed neural policies.

### Machine Learning (Statistics Category)

### 1. [Moment Alignment: Unifying Gradient and Hessian Matching for Domain Generalization](http://arxiv.org/pdf/2506.07378v1)

Authors: Yuen Chen, Haozhe Si, Guojun Zhang, Han Zhao

Domain generalization (DG) seeks to develop models that generalize well to
unseen target domains, addressing the prevalent issue of distribution shifts in
real-world applications. One line of research in DG focuses on aligning
domain-level gradients and Hessians to enhance generalization. However,
existing methods are computationally inefficient and the underlying principles
of these approaches are not well understood. In this paper, we develop the
theory of moment alignment for DG. Grounded in \textit{transfer measure}, a
principled framework for quantifying generalizability between two domains, we
first extend the definition of transfer measure to domain generalization that
includes multiple source domains and establish a target error bound. Then, we
prove that aligning derivatives across domains improves transfer measure both
when the feature extractor induces an invariant optimal predictor across
domains and when it does not. Notably, moment alignment provides a unifying
understanding of Invariant Risk Minimization, gradient matching, and Hessian
matching, three previously disconnected approaches to DG. We further connect
feature moments and derivatives of the classifier head, and establish the
duality between feature learning and classifier fitting. Building upon our
theory, we introduce \textbf{C}losed-Form \textbf{M}oment \textbf{A}lignment
(CMA), a novel DG algorithm that aligns domain-level gradients and Hessians in
closed-form. Our method overcomes the computational inefficiencies of existing
gradient and Hessian-based techniques by eliminating the need for repeated
backpropagation or sampling-based Hessian estimation. We validate the efficacy
of our approach through two sets of experiments: linear probing and full
fine-tuning. CMA demonstrates superior performance in both settings compared to
Empirical Risk Minimization and state-of-the-art algorithms.

### 2. [Explicit Preference Optimization: No Need for an Implicit Reward Model](http://arxiv.org/pdf/2506.07492v1)

Authors: Xiangkun Hu, Lemin Kong, Tong He, David Wipf

The generated responses of large language models (LLMs) are often fine-tuned
to human preferences through a process called reinforcement learning from human
feedback (RLHF). As RLHF relies on a challenging training sequence, whereby a
separate reward model is independently learned and then later applied to LLM
policy updates, ongoing research effort has targeted more straightforward
alternatives. In this regard, direct preference optimization (DPO) and its many
offshoots circumvent the need for a separate reward training step. Instead,
through the judicious use of a reparameterization trick that induces an
\textit{implicit} reward, DPO and related methods consolidate learning to the
minimization of a single loss function. And yet despite demonstrable success in
some real-world settings, we prove that DPO-based objectives are nonetheless
subject to sub-optimal regularization and counter-intuitive interpolation
behaviors, underappreciated artifacts of the reparameterizations upon which
they are based. To this end, we introduce an \textit{explicit} preference
optimization framework termed EXPO that requires no analogous
reparameterization to achieve an implicit reward. Quite differently, we merely
posit intuitively-appealing regularization factors from scratch that
transparently avoid the potential pitfalls of key DPO variants, provably
satisfying regularization desiderata that prior methods do not. Empirical
results serve to corroborate our analyses and showcase the efficacy of EXPO.

### 3. [Exploiting Curvature in Online Convex Optimization with Delayed Feedback](http://arxiv.org/pdf/2506.07595v1)

Authors: Hao Qiu, Emmanuel Esposito, Mengxiao Zhang

In this work, we study the online convex optimization problem with curved
losses and delayed feedback. When losses are strongly convex, existing
approaches obtain regret bounds of order $d_{\max} \ln T$, where $d_{\max}$ is
the maximum delay and $T$ is the time horizon. However, in many cases, this
guarantee can be much worse than $\sqrt{d_{\mathrm{tot}}}$ as obtained by a
delayed version of online gradient descent, where $d_{\mathrm{tot}}$ is the
total delay. We bridge this gap by proposing a variant of
follow-the-regularized-leader that obtains regret of order
$\min\{\sigma_{\max}\ln T, \sqrt{d_{\mathrm{tot}}}\}$, where $\sigma_{\max}$ is
the maximum number of missing observations. We then consider exp-concave losses
and extend the Online Newton Step algorithm to handle delays with an adaptive
learning rate tuning, achieving regret $\min\{d_{\max} n\ln T,
\sqrt{d_{\mathrm{tot}}}\}$ where $n$ is the dimension. To our knowledge, this
is the first algorithm to achieve such a regret bound for exp-concave losses.
We further consider the problem of unconstrained online linear regression and
achieve a similar guarantee by designing a variant of the Vovk-Azoury-Warmuth
forecaster with a clipping trick. Finally, we implement our algorithms and
conduct experiments under various types of delay and losses, showing an
improved performance over existing methods.

### 4. [Rao-Blackwellised Reparameterisation Gradients](http://arxiv.org/pdf/2506.07687v1)

Authors: Kevin Lam, Thang Bui, George Deligiannidis, Yee Whye Teh

Latent Gaussian variables have been popularised in probabilistic machine
learning. In turn, gradient estimators are the machinery that facilitates
gradient-based optimisation for models with latent Gaussian variables. The
reparameterisation trick is often used as the default estimator as it is simple
to implement and yields low-variance gradients for variational inference. In
this work, we propose the R2-G2 estimator as the Rao-Blackwellisation of the
reparameterisation gradient estimator. Interestingly, we show that the local
reparameterisation gradient estimator for Bayesian MLPs is an instance of the
R2-G2 estimator and Rao-Blackwellisation. This lets us extend benefits of
Rao-Blackwellised gradients to a suite of probabilistic models. We show that
initial training with R2-G2 consistently yields better performance in models
with multiple applications of the reparameterisation trick.

### 5. [Quickest Causal Change Point Detection by Adaptive Intervention](http://arxiv.org/pdf/2506.07760v1)

Authors: Haijie Xu, Chen Zhang

We propose an algorithm for change point monitoring in linear causal models
that accounts for interventions. Through a special centralization technique, we
can concentrate the changes arising from causal propagation across nodes into a
single dimension. Additionally, by selecting appropriate intervention nodes
based on Kullback-Leibler divergence, we can amplify the change magnitude. We
also present an algorithm for selecting the intervention values, which aids in
the identification of the most effective intervention nodes. Two monitoring
methods are proposed, each with an adaptive intervention policy to make a
balance between exploration and exploitation. We theoretically demonstrate the
first-order optimality of the proposed methods and validate their properties
using simulation datasets and two real-world case studies.

### 6. [Heavy Lasso: sparse penalized regression under heavy-tailed noise via data-augmented soft-thresholding](http://arxiv.org/pdf/2506.07790v1)

Authors: The Tien Mai

High-dimensional linear regression is a fundamental tool in modern
statistics, particularly when the number of predictors exceeds the sample size.
The classical Lasso, which relies on the squared loss, performs well under
Gaussian noise assumptions but often deteriorates in the presence of
heavy-tailed errors or outliers commonly encountered in real data applications
such as genomics, finance, and signal processing. To address these challenges,
we propose a novel robust regression method, termed Heavy Lasso, which
incorporates a loss function inspired by the Student's t-distribution within a
Lasso penalization framework. This loss retains the desirable quadratic
behavior for small residuals while adaptively downweighting large deviations,
thus enhancing robustness to heavy-tailed noise and outliers. Heavy Lasso
enjoys computationally efficient by leveraging a data augmentation scheme and a
soft-thresholding algorithm, which integrate seamlessly with classical Lasso
solvers. Theoretically, we establish non-asymptotic bounds under both $\ell_1$
and $\ell_2 $ norms, by employing the framework of localized convexity, showing
that the Heavy Lasso estimator achieves rates comparable to those of the Huber
loss. Extensive numerical studies demonstrate Heavy Lasso's superior
performance over classical Lasso and other robust variants, highlighting its
effectiveness in challenging noisy settings. Our method is implemented in the R
package heavylasso available on Github.

### 7. [CausalPFN: Amortized Causal Effect Estimation via In-Context Learning](http://arxiv.org/pdf/2506.07918v1)

Authors: Vahid Balazadeh, Hamidreza Kamkari, Valentin Thomas, Benson Li, Junwei Ma, Jesse C. Cresswell, Rahul G. Krishnan

Causal effect estimation from observational data is fundamental across
various applications. However, selecting an appropriate estimator from dozens
of specialized methods demands substantial manual effort and domain expertise.
We present CausalPFN, a single transformer that amortizes this workflow:
trained once on a large library of simulated data-generating processes that
satisfy ignorability, it infers causal effects for new observational datasets
out-of-the-box. CausalPFN combines ideas from Bayesian causal inference with
the large-scale training protocol of prior-fitted networks (PFNs), learning to
map raw observations directly to causal effects without any task-specific
adjustment. Our approach achieves superior average performance on heterogeneous
and average treatment effect estimation benchmarks (IHDP, Lalonde, ACIC).
Moreover, it shows competitive performance for real-world policy making on
uplift modeling tasks. CausalPFN provides calibrated uncertainty estimates to
support reliable decision-making based on Bayesian principles. This
ready-to-use model does not require any further training or tuning and takes a
step toward automated causal inference (https://github.com/vdblm/CausalPFN).

### 8. [Ensemble-Based Survival Models with the Self-Attended Beran Estimator Predictions](http://arxiv.org/pdf/2506.07933v1)

Authors: Lev V. Utkin, Semen P. Khomets, Vlada A. Efremenko, Andrei V. Konstantinov, Natalya M. Verbova

Survival analysis predicts the time until an event of interest, such as
failure or death, but faces challenges due to censored data, where some events
remain unobserved. Ensemble-based models, like random survival forests and
gradient boosting, are widely used but can produce unstable predictions due to
variations in bootstrap samples. To address this, we propose SurvBESA (Survival
Beran Estimators Self-Attended), a novel ensemble model that combines Beran
estimators with a self-attention mechanism. Unlike traditional methods,
SurvBESA applies self-attention to predicted survival functions, smoothing out
noise by adjusting each survival function based on its similarity to
neighboring survival functions. We also explore a special case using Huber's
contamination model to define attention weights, simplifying training to a
quadratic or linear optimization problem. Numerical experiments show that
SurvBESA outperforms state-of-the-art models. The implementation of SurvBESA is
publicly available.

### 9. [Flowing Datasets with Wasserstein over Wasserstein Gradient Flows](http://arxiv.org/pdf/2506.07534v1)

Authors: Clément Bonet, Christophe Vauthier, Anna Korba

Many applications in machine learning involve data represented as probability
distributions. The emergence of such data requires radically novel techniques
to design tractable gradient flows on probability distributions over this type
of (infinite-dimensional) objects. For instance, being able to flow labeled
datasets is a core task for applications ranging from domain adaptation to
transfer learning or dataset distillation. In this setting, we propose to
represent each class by the associated conditional distribution of features,
and to model the dataset as a mixture distribution supported on these classes
(which are themselves probability distributions), meaning that labeled datasets
can be seen as probability distributions over probability distributions. We
endow this space with a metric structure from optimal transport, namely the
Wasserstein over Wasserstein (WoW) distance, derive a differential structure on
this space, and define WoW gradient flows. The latter enables to design
dynamics over this space that decrease a given objective functional. We apply
our framework to transfer learning and dataset distillation tasks, leveraging
our gradient flow construction as well as novel tractable functionals that take
the form of Maximum Mean Discrepancies with Sliced-Wasserstein based kernels
between probability distributions.

### 10. [The Universality Lens: Why Even Highly Over-Parametrized Models Learn Well](http://arxiv.org/pdf/2506.07661v1)

Authors: Meir Feder, Ruediger Urbanke, Yaniv Fogel

A fundamental question in modern machine learning is why large,
over-parameterized models, such as deep neural networks and transformers, tend
to generalize well, even when their number of parameters far exceeds the number
of training samples.
  We investigate this phenomenon through the lens of information theory,
grounded in universal learning theory. Specifically, we study a Bayesian
mixture learner with log-loss and (almost) uniform prior over an expansive
hypothesis class.
  Our key result shows that the learner's regret is not determined by the
overall size of the hypothesis class, but rather by the cumulative probability
of all models that are close, in Kullback-Leibler divergence distance, to the
true data-generating process. We refer to this cumulative probability as the
weight of the hypothesis.
  This leads to a natural notion of model simplicity: simple models are those
with large weight and thus require fewer samples to generalize, while complex
models have small weight and need more data. This perspective provides a
rigorous and intuitive explanation for why over-parameterized models often
avoid overfitting: the presence of simple hypotheses allows the posterior to
concentrate on them when supported by the data.
  We further bridge theory and practice by recalling that stochastic gradient
descent with Langevin dynamics samples from the correct posterior distribution,
enabling our theoretical learner to be approximated using standard machine
learning methods combined with ensemble learning.
  Our analysis yields non-uniform regret bounds and aligns with key practical
concepts such as flat minima and model distillation. The results apply broadly
across online, batch, and supervised learning settings, offering a unified and
principled understanding of the generalization behavior of modern AI systems.

