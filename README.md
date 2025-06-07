# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-06-06 17:02:30.838293 PST.

### Artificial Intelligence

### 1. [OpenAg: Democratizing Agricultural Intelligence](http://arxiv.org/pdf/2506.04571v1)

Authors: Srikanth Thudumu, Jason Fisher

Agriculture is undergoing a major transformation driven by artificial
intelligence (AI), machine learning, and knowledge representation technologies.
However, current agricultural intelligence systems often lack contextual
understanding, explainability, and adaptability, especially for smallholder
farmers with limited resources. General-purpose large language models (LLMs),
while powerful, typically lack the domain-specific knowledge and contextual
reasoning needed for practical decision support in farming. They tend to
produce recommendations that are too generic or unrealistic for real-world
applications. To address these challenges, we present OpenAg, a comprehensive
framework designed to advance agricultural artificial general intelligence
(AGI). OpenAg combines domain-specific foundation models, neural knowledge
graphs, multi-agent reasoning, causal explainability, and adaptive transfer
learning to deliver context-aware, explainable, and actionable insights. The
system includes: (i) a unified agricultural knowledge base that integrates
scientific literature, sensor data, and farmer-generated knowledge; (ii) a
neural agricultural knowledge graph for structured reasoning and inference;
(iii) an adaptive multi-agent reasoning system where AI agents specialize and
collaborate across agricultural domains; and (iv) a causal transparency
mechanism that ensures AI recommendations are interpretable, scientifically
grounded, and aligned with real-world constraints. OpenAg aims to bridge the
gap between scientific knowledge and the tacit expertise of experienced farmers
to support scalable and locally relevant agricultural decision-making.

### 2. [Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation](http://arxiv.org/pdf/2506.04614v1)

Authors: Yuyang Wanyan, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Jiabo Ye, Yutong Kou, Ming Yan, Fei Huang, Xiaoshan Yang, Weiming Dong, Changsheng Xu

In recent years, Multimodal Large Language Models (MLLMs) have been
extensively utilized for multimodal reasoning tasks, including Graphical User
Interface (GUI) automation. Unlike general offline multimodal tasks, GUI
automation is executed in online interactive environments, necessitating
step-by-step decision-making based on real-time status of the environment. This
task has a lower tolerance for decision-making errors at each step, as any
mistakes may cumulatively disrupt the process and potentially lead to
irreversible outcomes like deletions or payments. To address these issues, we
introduce a pre-operative critic mechanism that provides effective feedback
prior to the actual execution, by reasoning about the potential outcome and
correctness of actions. Specifically, we propose a Suggestion-aware Gradient
Relative Policy Optimization (S-GRPO) strategy to construct our pre-operative
critic model GUI-Critic-R1, incorporating a novel suggestion reward to enhance
the reliability of the model's feedback. Furthermore, we develop a
reasoning-bootstrapping based data collection pipeline to create a
GUI-Critic-Train and a GUI-Critic-Test, filling existing gaps in GUI critic
data. Static experiments on the GUI-Critic-Test across both mobile and web
domains reveal that our GUI-Critic-R1 offers significant advantages in critic
accuracy compared to current MLLMs. Dynamic evaluation on GUI automation
benchmark further highlights the effectiveness and superiority of our model, as
evidenced by improved success rates and operational efficiency.

### 3. [CHANCERY: Evaluating corporate governance reasoning capabilities in language models](http://arxiv.org/pdf/2506.04636v1)

Authors: Lucas Irwin, Arda Kaz, Peiyao Sheng, Pramod Viswanath

Law has long been a domain that has been popular in natural language
processing (NLP) applications. Reasoning (ratiocination and the ability to make
connections to precedent) is a core part of the practice of the law in the real
world. Nevertheless, while multiple legal datasets exist, none have thus far
focused specifically on reasoning tasks. We focus on a specific aspect of the
legal landscape by introducing a corporate governance reasoning benchmark
(CHANCERY) to test a model's ability to reason about whether
executive/board/shareholder's proposed actions are consistent with corporate
governance charters. This benchmark introduces a first-of-its-kind corporate
governance reasoning test for language models - modeled after real world
corporate governance law. The benchmark consists of a corporate charter (a set
of governing covenants) and a proposal for executive action. The model's task
is one of binary classification: reason about whether the action is consistent
with the rules contained within the charter. We create the benchmark following
established principles of corporate governance - 24 concrete corporate
governance principles established in and 79 real life corporate charters
selected to represent diverse industries from a total dataset of 10k real life
corporate charters. Evaluations on state-of-the-art (SOTA) reasoning models
confirm the difficulty of the benchmark, with models such as Claude 3.7 Sonnet
and GPT-4o achieving 64.5% and 75.2% accuracy respectively. Reasoning agents
exhibit superior performance, with agents based on the ReAct and CodeAct
frameworks scoring 76.1% and 78.1% respectively, further confirming the
advanced legal reasoning capabilities required to score highly on the
benchmark. We also conduct an analysis of the types of questions which current
reasoning models struggle on, revealing insights into the legal reasoning
capabilities of SOTA models.

### 4. [Agents of Change: Self-Evolving LLM Agents for Strategic Planning](http://arxiv.org/pdf/2506.04651v1)

Authors: Nikolas Belle, Dakota Barnes, Alfonso Amayuelas, Ivan Bercovich, Xin Eric Wang, William Wang

Recent advances in LLMs have enabled their use as autonomous agents across a
range of tasks, yet they continue to struggle with formulating and adhering to
coherent long-term strategies. In this paper, we investigate whether LLM agents
can self-improve when placed in environments that explicitly challenge their
strategic planning abilities. Using the board game Settlers of Catan, accessed
through the open-source Catanatron framework, we benchmark a progression of
LLM-based agents, from a simple game-playing agent to systems capable of
autonomously rewriting their own prompts and their player agent's code. We
introduce a multi-agent architecture in which specialized roles (Analyzer,
Researcher, Coder, and Player) collaborate to iteratively analyze gameplay,
research new strategies, and modify the agent's logic or prompt. By comparing
manually crafted agents to those evolved entirely by LLMs, we evaluate how
effectively these systems can diagnose failure and adapt over time. Our results
show that self-evolving agents, particularly when powered by models like Claude
3.7 and GPT-4o, outperform static baselines by autonomously adopting their
strategies, passing along sample behavior to game-playing agents, and
demonstrating adaptive reasoning over multiple iterations.

### 5. [E-bike agents: Large Language Model-Driven E-Bike Accident Analysis and Severity Prediction](http://arxiv.org/pdf/2506.04654v1)

Authors: Zhichao Yang, Jiashu He, Mohammad B. Al-Khasawneh, Darshan Pandit, Cirillo Cinzia

Electric bicycles (e-bikes) are rapidly increasing in use, raising safety
concerns due to a rise in accident reports. However, e-bike incident reports
often use unstructured narrative formats, which hinders quantitative safety
analysis. This study introduces E-bike agents, a framework that uses large
language models (LLM) powered agents to classify and extract safety variables
from unstructured incident reports. Our framework consists of four LLM agents,
handling data classification, information extraction, injury cause
determination, and component linkage, to extract the key factors that could
lead to E-bike accidents and cause varying severity levels. Furthermore, we
used an ordered logit model to examine the relationship between the severity of
the incident and the factors retrieved, such as gender, the type of cause, and
environmental conditions. Our research shows that equipment issues are slightly
more common than human-related ones, but human-related incidents are more often
fatal. Specifically, pedals, tires, and brakes are frequent contributors to
accidents. The model achieves a high weighted F1 score of 0.87 in
classification accuracy, highlighting the potential of using LLMs to extract
unstructured data in niche domains, such as transportation. Our method offers a
scalable solution to improve e-bike safety analytics and provides actionable
information for policy makers, designers, and regulators.

### 6. [Empowering Economic Simulation for Massively Multiplayer Online Games through Generative Agent-Based Modeling](http://arxiv.org/pdf/2506.04699v1)

Authors: Bihan Xu, Shiwei Zhao, Runze Wu, Zhenya Huang, Jiawei Wang, Zhipeng Hu, Kai Wang, Haoyu Liu, Tangjie Lv, Le Li, Changjie Fan, Xin Tong, Jiangze Han

Within the domain of Massively Multiplayer Online (MMO) economy research,
Agent-Based Modeling (ABM) has emerged as a robust tool for analyzing game
economics, evolving from rule-based agents to decision-making agents enhanced
by reinforcement learning. Nevertheless, existing works encounter significant
challenges when attempting to emulate human-like economic activities among
agents, particularly regarding agent reliability, sociability, and
interpretability. In this study, we take a preliminary step in introducing a
novel approach using Large Language Models (LLMs) in MMO economy simulation.
Leveraging LLMs' role-playing proficiency, generative capacity, and reasoning
aptitude, we design LLM-driven agents with human-like decision-making and
adaptability. These agents are equipped with the abilities of role-playing,
perception, memory, and reasoning, addressing the aforementioned challenges
effectively. Simulation experiments focusing on in-game economic activities
demonstrate that LLM-empowered agents can promote emergent phenomena like role
specialization and price fluctuations in line with market rules.

### 7. [Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](http://arxiv.org/pdf/2506.04723v1)

Authors: Jiayu Wang, Yifei Ming, Zixuan Ke, Caiming Xiong, Shafiq Joty, Aws Albarghouthi, Frederic Sala

Reinforcement learning (RL) has become the dominant paradigm for endowing
language models with advanced reasoning capabilities. Despite the substantial
empirical gains demonstrated by RL-based training methods like GRPO, a granular
understanding of their advantages is still lacking. To address this gap, we
introduce a fine-grained analytic framework to dissect the impact of RL on
reasoning. Our framework specifically investigates key elements that have been
hypothesized to benefit from RL training: (1) plan-following and execution, (2)
problem decomposition, and (3) improved reasoning and knowledge utilization.
Using this framework, we gain insights beyond mere accuracy. For instance,
providing models with explicit step-by-step plans surprisingly degrades
performance on the most challenging benchmarks, yet RL-tuned models exhibit
greater robustness, experiencing markedly smaller performance drops than their
base counterparts. This suggests that RL may not primarily enhance the
execution of external plans but rather empower models to formulate and follow
internal strategies better suited to their reasoning processes. Conversely, we
observe that RL enhances the model's capacity to integrate provided knowledge
into its reasoning process, leading to performance improvements across diverse
tasks. We also study difficulty, showing improved training by developing new
ways to exploit hard problems. Our findings lay a foundation for more
principled training and evaluation of reasoning models.

### 8. [Safe Planning and Policy Optimization via World Model Learning](http://arxiv.org/pdf/2506.04828v1)

Authors: Artem Latyshev, Gregory Gorbov, Aleksandr I. Panov

Reinforcement Learning (RL) applications in real-world scenarios must
prioritize safety and reliability, which impose strict constraints on agent
behavior. Model-based RL leverages predictive world models for action planning
and policy optimization, but inherent model inaccuracies can lead to
catastrophic failures in safety-critical settings. We propose a novel
model-based RL framework that jointly optimizes task performance and safety. To
address world model errors, our method incorporates an adaptive mechanism that
dynamically switches between model-based planning and direct policy execution.
We resolve the objective mismatch problem of traditional model-based approaches
using an implicit world model. Furthermore, our framework employs dynamic
safety thresholds that adapt to the agent's evolving capabilities, consistently
selecting actions that surpass safe policy suggestions in both performance and
safety. Experiments demonstrate significant improvements over non-adaptive
methods, showing that our approach optimizes safety and performance
simultaneously rather than merely meeting minimum safety requirements. The
proposed framework achieves robust performance on diverse safety-critical
continuous control tasks, outperforming existing methods.

### 9. [Towards a Multi-Agent Simulation of Cyber-attackers and Cyber-defenders Battles](http://arxiv.org/pdf/2506.04849v1)

Authors: Julien Soulé, Jean-Paul Jamont, Michel Occello, Paul Théron, Louis-Marie Traonouez

As cyber-attacks show to be more and more complex and coordinated,
cyber-defenders strategy through multi-agent approaches could be key to tackle
against cyber-attacks as close as entry points in a networked system. This
paper presents a Markovian modeling and implementation through a simulator of
fighting cyber-attacker agents and cyber-defender agents deployed on host
network nodes. It aims to provide an experimental framework to implement
realistically based coordinated cyber-attack scenarios while assessing
cyber-defenders dynamic organizations. We abstracted network nodes by sets of
properties including agents' ones. Actions applied by agents model how the
network reacts depending in a given state and what properties are to change.
Collective choice of the actions brings the whole environment closer or farther
from respective cyber-attackers and cyber-defenders goals. Using the simulator,
we implemented a realistically inspired scenario with several behavior
implementation approaches for cyber-defenders and cyber-attackers.

### 10. [Differentiable Logic Cellular Automata: From Game of Life to Pattern Generation](http://arxiv.org/pdf/2506.04912v1)

Authors: Pietro Miotti, Eyvind Niklasson, Ettore Randazzo, Alexander Mordvintsev

This paper introduces Differentiable Logic Cellular Automata (DiffLogic CA),
a novel combination of Neural Cellular Automata (NCA) and Differentiable Logic
Gates Networks (DLGNs). The fundamental computation units of the model are
differentiable logic gates, combined into a circuit. During training, the model
is fully end-to-end differentiable allowing gradient-based training, and at
inference time it operates in a fully discrete state space. This enables
learning local update rules for cellular automata while preserving their
inherent discrete nature. We demonstrate the versatility of our approach
through a series of milestones: (1) fully learning the rules of Conway's Game
of Life, (2) generating checkerboard patterns that exhibit resilience to noise
and damage, (3) growing a lizard shape, and (4) multi-color pattern generation.
Our model successfully learns recurrent circuits capable of generating desired
target patterns. For simpler patterns, we observe success with both synchronous
and asynchronous updates, demonstrating significant generalization capabilities
and robustness to perturbations. We make the case that this combination of
DLGNs and NCA represents a step toward programmable matter and robust computing
systems that combine binary logic, neural network adaptability, and localized
processing. This work, to the best of our knowledge, is the first successful
application of differentiable logic gate networks in recurrent architectures.

### Hardware Architecture

### 1. [ROSGuard: A Bandwidth Regulation Mechanism for ROS2-based Applications](http://arxiv.org/pdf/2506.04640v1)

Authors: Jon Altonaga Puente, Enrico Mezzetti, Irune Agirre Troncoso, Jaume Abella Ferrer, Francisco J. Cazorla Almeida

Multicore timing interference, arising when multiple requests contend for the
same shared hardware resources, is a primary concern for timing verification
and validation of time-critical applications. Bandwidth control and regulation
approaches have been proposed in the literature as an effective method to
monitor and limit the impact of timing interference at run time. These
approaches seek for fine-grained control of the bandwidth consumption (at the
microsecond level) to meet stringent timing requirements on embedded critical
systems. Such granularity and configurations, while effective, can become an
entry barrier for the application of bandwidth control to a wide class of
productized, modular ROS2 applications. This is so because those applications
have less stringent timing requirements but would still benefit from bandwidth
regulation, though under less restrictive, and therefore more portable,
granularity and configurations.
  In this work, we provide ROSGuard, a highly-portable, modular implementation
of a timing interference monitoring and control mechanism that builds on the
abstractions available on top of a generic and portable Linux-based software
stack with the Robotic Operating System 2 (ROS2) layer, a widespreadedly
adopted middleware for a wide class of industrial applications, far beyond the
robotic domain. We deploy ROSGuard on an NVIDIA AGX Orin platform as a
representative target for functionally rich distributed AI-based applications
and a set of synthetic and real-world benchmarks. We apply an effective
bandwidth regulation scheme on ROS2-based applications and achieve comparable
effectiveness to specialized, finer-grained state-of-the-art solutions.

### 2. [QiMeng: Fully Automated Hardware and Software Design for Processor Chip](http://arxiv.org/pdf/2506.05007v1)

Authors: Rui Zhang, Yuanbo Wen, Shuyao Cheng, Di Huang, Shaohui Peng, Jiaming Guo, Pengwei Jin, Jiacheng Zhao, Tianrui Ma, Yaoyu Zhu, Yifan Hao, Yongwei Zhao, Shengwen Liang, Ying Wang, Xing Hu, Zidong Du, Huimin Cui, Ling Li, Qi Guo, Yunji Chen

Processor chip design technology serves as a key frontier driving
breakthroughs in computer science and related fields. With the rapid
advancement of information technology, conventional design paradigms face three
major challenges: the physical constraints of fabrication technologies, the
escalating demands for design resources, and the increasing diversity of
ecosystems. Automated processor chip design has emerged as a transformative
solution to address these challenges. While recent breakthroughs in Artificial
Intelligence (AI), particularly Large Language Models (LLMs) techniques, have
opened new possibilities for fully automated processor chip design, substantial
challenges remain in establishing domain-specific LLMs for processor chip
design.
  In this paper, we propose QiMeng, a novel system for fully automated hardware
and software design of processor chips. QiMeng comprises three hierarchical
layers. In the bottom-layer, we construct a domain-specific Large Processor
Chip Model (LPCM) that introduces novel designs in architecture, training, and
inference, to address key challenges such as knowledge representation gap, data
scarcity, correctness assurance, and enormous solution space. In the
middle-layer, leveraging the LPCM's knowledge representation and inference
capabilities, we develop the Hardware Design Agent and the Software Design
Agent to automate the design of hardware and software for processor chips.
Currently, several components of QiMeng have been completed and successfully
applied in various top-layer applications, demonstrating significant advantages
and providing a feasible solution for efficient, fully automated
hardware/software design of processor chips. Future research will focus on
integrating all components and performing iterative top-down and bottom-up
design processes to establish a comprehensive QiMeng system.

### 3. [hdl2v: A Code Translation Dataset for Enhanced LLM Verilog Generation](http://arxiv.org/pdf/2506.04544v1)

Authors: Charles Hong, Brendan Roberts, Huijae An, Alex Um, Advay Ratan, Yakun Sophia Shao

Large language models (LLMs) are playing an increasingly large role in
domains such as code generation, including hardware code generation, where
Verilog is the key language. However, the amount of publicly available Verilog
code pales in comparison to the amount of code available for software languages
like Python. In this work, we present hdl2v ("HDL-to-Verilog"), a dataset which
seeks to increase the amount of available human-written Verilog data by
translating or compiling three other hardware description languages - VHDL,
Chisel, and PyMTL3 - to Verilog. Furthermore, we demonstrate the value of hdl2v
in enhancing LLM Verilog generation by improving performance of a 32
billion-parameter open-weight model by up to 23% (pass@10) in VerilogEvalV2,
without utilizing any data augmentation or knowledge distillation from larger
models. We also show hdl2v's ability to boost the performance of a data
augmentation-based fine-tuning approach by 63%. Finally, we characterize and
analyze our dataset to better understand which characteristics of
HDL-to-Verilog datasets can be expanded upon in future work for even better
performance.

### 4. [FlashDMoE: Fast Distributed MoE in a Single Kernel](http://arxiv.org/pdf/2506.04667v1)

Authors: Osayamen Jonathan Aimuyo, Byungsoo Oh, Rachee Singh

The computational sparsity of Mixture-of-Experts (MoE) models enables
sub-linear growth in compute cost as model size increases, offering a scalable
path to training massive neural networks. However, existing implementations
suffer from \emph{low GPU utilization}, \emph{significant latency overhead},
and a fundamental \emph{inability to leverage task locality}, primarily due to
CPU-managed scheduling, host-initiated communication, and frequent kernel
launches. To overcome these limitations, we develop FlashDMoE, a fully
GPU-resident MoE operator that fuses expert computation and inter-GPU
communication into a \emph{single persistent GPU kernel}. FlashDMoE enables
fine-grained pipelining of dispatch, compute, and combine phases, eliminating
launch overheads and reducing idle gaps. Its device-initiated communication
protocol introduces \emph{payload-efficient} data transfers, significantly
shrinking buffer sizes in sparsely activated MoE layers. When evaluated on a
single 8-H100 GPU node with MoE models having up to 128 experts and 16K token
sequences, FlashDMoE achieves up to \textbf{6}x lower latency, \textbf{5,7}x
higher throughput, \textbf{4}x better weak scaling efficiency, and \textbf{9}x
higher GPU utilization compared to state-of-the-art baselines, despite using
FP32 while baselines use FP16. FlashDMoE demonstrates that principled GPU
kernel-hardware co-design is key to unlocking the performance ceiling of
large-scale distributed ML workloads.

### 5. [Memory Hierarchy Design for Caching Middleware in the Age of NVM](http://arxiv.org/pdf/2506.05071v1)

Authors: Shahram Ghandeharizadeh, Sandy Irani, Jenny Lam

Advances in storage technology have introduced Non-Volatile Memory, NVM, as a
new storage medium. NVM, along with Dynamic Random Access Memory (DRAM), Solid
State Disk (SSD), and Disk present a system designer with a wide array of
options in designing caching middleware. Moreover, design decisions to
replicate a data item in more than one level of a caching memory hierarchy may
enhance the overall system performance with a faster recovery time in the event
of a memory failure. Given a fixed budget, the key configuration questions are:
Which storage media should constitute the memory hierarchy? What is the storage
capacity of each hierarchy? Should data be replicated or partitioned across the
different levels of the hierarchy? We model these cache configuration questions
as an instance of the Multiple Choice Knapsack Problem (MCKP). This model is
guided by the specification of each type of memory along with an application's
database characteristics and its workload. Although MCKP is NP-complete, its
linear programming relaxation is efficiently solvable and can be used to
closely approximate the optimal solution. We use the resulting simple algorithm
to evaluate design tradeoffs in the context of a memory hierarchy for a
Key-Value Store (e.g., memcached) as well as a host-side cache (e.g.,
Flashcache). The results show selective replication is appropriate with certain
failure rates and workload characteristics. With a slim failure rate and
frequent data updates, tiering of data across the different storage media that
constitute the cache is superior to replication.

### Computational Complexity

### 1. [Identity Testing for Circuits with Exponentiation Gates](http://arxiv.org/pdf/2506.04529v1)

Authors: Jiatu Li, Mengdi Wu

Motivated by practical applications in the design of optimization compilers
for neural networks, we initiated the study of identity testing problems for
arithmetic circuits augmented with \emph{exponentiation gates} that compute the
real function $x\mapsto e^x$. These circuits compute real functions of form
$P(\vec x)/P'(\vec x)$, where both $P(\vec x)$ and $P'(\vec x)$ are exponential
polynomials
  \[
  \sum_{i=1}^k f_i(\vec x)\cdot \exp\left(\frac{g_i(\vec x)}{h_i(\vec
x)}\right),
  \]
  for polynomials $f_i(\vec x),g_i(\vec x)$, and $h_i(\vec x)$.
  We formalize a black-box query model over finite fields for this class of
circuits, which is mathematical simple and reflects constraints faced by
real-world neural network compilers. We proved that a simple and efficient
randomized identity testing algorithm achieves perfect completeness and
non-trivial soundness. Concurrent with our work, the algorithm has been
implemented in the optimization compiler Mirage by Wu et al.~(OSDI 2025),
demonstrating promising empirical performance in both efficiency and soundness
error. Finally, we propose a number-theoretic conjecture under which our
algorithm is sound with high probability.

### 2. [Equilibrium Computation in First-Price Auctions with Correlated Priors](http://arxiv.org/pdf/2506.05322v1)

Authors: Aris Filos-Ratsikas, Yiannis Giannakopoulos, Alexandros Hollender, Charalampos Kokkalis

We consider the computational complexity of computing Bayes-Nash equilibria
in first-price auctions, where the bidders' values for the item are drawn from
a general (possibly correlated) joint distribution. We show that when the
values and the bidding space are discrete, determining the existence of a pure
Bayes-Nash equilibrium is NP-hard. This is the first hardness result in the
literature of the problem that does not rely on assumptions of subjectivity of
the priors, or convoluted tie-breaking rules. We then present two main
approaches for achieving positive results, via bid sparsification and via bid
densification. The former is more combinatorial and is based on enumeration
techniques, whereas the latter makes use of the continuous theory of the
problem developed in the economics literature. Using these approaches, we
develop polynomial-time approximation algorithms for computing equilibria in
symmetric settings or settings with a fixed number of bidders, for different
(discrete or continuous) variants of the auction.

### Computational Engineering

### 1. [Nonlinear elastodynamic material identification of heterogeneous isogeometric Bernoulli-Euler beams](http://arxiv.org/pdf/2506.04960v1)

Authors: Bartłomiej Łazorczyk, Roger A. Sauer

This paper presents a Finite Element Model Updating framework for identifying
heterogeneous material distributions in planar Bernoulli-Euler beams based on a
rotation-free isogeometric formulation. The procedure follows two steps: First,
the elastic properties are identified from quasi-static displacements; then,
the density is determined from modal data (low frequencies and mode shapes),
given the previously obtained elastic properties. The identification relies on
three independent discretizations: the isogeometric finite element mesh, a
high-resolution grid of experimental measurements, and a material mesh composed
of low-order Lagrange elements. The material mesh approximates the unknown
material distributions, with its nodal values serving as design variables. The
error between experiments and numerical model is expressed in a least squares
manner. The objective is minimized using local optimization with the
trust-region method, providing analytical derivatives to accelerate
computations. Several numerical examples exhibiting large displacements are
provided to test the proposed approach. To alleviate membrane locking, the B2M1
discretization is employed when necessary. Quasi-experimental data is generated
using refined finite element models with random noise applied up to 4%. The
method yields satisfactory results as long as a sufficient amount of
experimental data is available, even for high measurement noise. Regularization
is used to ensure a stable solution for dense material meshes. The density can
be accurately reconstructed based on the previously identified elastic
properties. The proposed framework can be straightforwardly extended to shells
and 3D continua.

### 2. [FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis](http://arxiv.org/pdf/2506.05019v1)

Authors: Wenyan Xu, Dawei Xiang, Yue Liu, Xiyu Wang, Yanxiang Ma, Liang Zhang, Chang Xu, Jiaheng Zhang

Pure time series forecasting tasks typically focus exclusively on numerical
features; however, real-world financial decision-making demands the comparison
and analysis of heterogeneous sources of information. Recent advances in deep
learning and large scale language models (LLMs) have made significant strides
in capturing sentiment and other qualitative signals, thereby enhancing the
accuracy of financial time series predictions. Despite these advances, most
existing datasets consist solely of price series and news text, are confined to
a single market, and remain limited in scale. In this paper, we introduce
FinMultiTime, the first large scale, multimodal financial time series dataset.
FinMultiTime temporally aligns four distinct modalities financial news,
structured financial tables, K-line technical charts, and stock price time
series across both the S&P 500 and HS 300 universes. Covering 5,105 stocks from
2009 to 2025 in the United States and China, the dataset totals 112.6 GB and
provides minute-level, daily, and quarterly resolutions, thus capturing short,
medium, and long term market signals with high fidelity. Our experiments
demonstrate that (1) scale and data quality markedly boost prediction accuracy;
(2) multimodal fusion yields moderate gains in Transformer models; and (3) a
fully reproducible pipeline enables seamless dataset updates.

### 3. [Adaptive recycled plastic architecture: Vacuum-Sealed Chainmail Structures Through Computational Design](http://arxiv.org/pdf/2506.04660v1)

Authors: Yi Xu, Farzin Lotfi-Jam, Mustafa Faruki

The construction industry is a major consumer of raw materials, accounting
for nearly half of global material usage annually, while generating significant
waste that poses sustainability challenges. This paper explores the untapped
potential of recycled plastics as a primary construction material, leveraging
their lightweight, flexible, and customizable properties for advanced
applications in modular chainmail systems. Through a computational workflow,
the study optimizes the design, testing, and fabrication of vacuum-sealed
chainmail structures composed of recycled plastic filaments, demonstrating
their adaptability and structural performance for architectural use.
  Key contributions include a novel methodology for integrating recycled
plastic filaments into chainmail geometries, validated through 2D sectional
testing, 3D shell structure generation, and physical modeling under vacuum
constraints. The research identifies the rectangular chainmail configuration as
the most efficient and adaptable, achieving superior deformation capacity,
material efficiency, and load-bearing performance. Optimization strategies for
temporary structures highlight practical deployment potential, balancing
material savings, usable area, and water drainage efficiency.
  The findings offer a foundation for innovative applications in extreme
conditions, including disaster-prone areas, high-altitude environments,
underwater platforms, and extraterrestrial habitats. These applications
leverage the lightweight, adaptable, and durable properties of recycled
plastics and modular chainmail systems, bridging the gap between waste
management and high-performance design while addressing unique challenges in
harsh and resource-constrained environments.

### 4. [A Private Smart Wallet with Probabilistic Compliance](http://arxiv.org/pdf/2506.04853v1)

Authors: Andrea Rizzini, Marco Esposito, Francesco Bruschi, Donatella Sciuto

We propose a privacy-preserving smart wallet with a novel invitation-based
private onboarding mechanism. The solution integrates two levels of compliance
in concert with an authority party: a proof of innocence mechanism and an
ancestral commitment tracking system using bloom filters for probabilistic UTXO
chain states. Performance analysis demonstrates practical efficiency: private
transfers with compliance checks complete within seconds on a consumer-grade
laptop, and overall with proof generation remaining low. On-chain costs stay
minimal, ensuring affordability for all operations on Base layer 2 network. The
wallet facilitates private contact list management through encrypted data blobs
while maintaining transaction unlinkability. Our evaluation validates the
approach's viability for privacy-preserving, compliance-aware digital payments
with minimized computational and financial overhead.

### 5. [Tensor-based multivariate function approximation: methods benchmarking and comparison](http://arxiv.org/pdf/2506.04791v1)

Authors: Athanasios C. Antoulas, Ion Victor Gosea, Charles Poussot-Vassal, Pierre Vuillemin

In this note, we evaluate the performances, the features and the
user-experience of some methods (and their implementations) designed for
tensor- (or data-) based multivariate function construction and approximation.
To this aim, a collection of multivariate functions extracted from contributive
works coming from different communities, is suggested. First, these functions
with varying complexity (e.g. number and degree of the variables) and nature
(e.g. rational, irrational, differentiable or not, symmetric, etc.) are used to
construct tensors, each of different dimension and size on the disk. Second,
grounded on this tensor, we inspect performances of each considered method
(e.g. the accuracy, the computational time, the parameters tuning impact,
etc.). Finally, considering the "best" parameter tuning set, we compare each
method using multiple evaluation criteria. The purpose of this note is not to
rank the methods but rather to evaluate as fairly as possible the different
available strategies, with the idea in mind to guide users to understand the
process, the possibilities, the advantages and the limits brought by each
tools. The contribution claimed is to suggest a complete benchmark collection
of some available tools for tensor approximation by surrogate models (e.g.
rational functions, networks, etc.). In addition, as contributors of the
multivariate Loewner Framework (mLF) approach (and its side implementation in
MDSPACK), attention and details of the latter are more explicitly given, in
order to provide readers a digest of this contributive work and some details
with simple examples.

### Computational Geometry

### 1. [The Peculiarities of Extending Queue Layouts](http://arxiv.org/pdf/2506.05156v1)

Authors: Thomas Depian, Simon D. Fink, Robert Ganian, Martin Nöllenburg

We consider the problem of computing $\ell$-page queue layouts, which are
linear arrangements of vertices accompanied with an assignment of the edges to
pages from one to $\ell$ that avoid the nesting of edges on any of the pages.
Inspired by previous work in the extension of stack layouts, here we consider
the setting of extending a partial $\ell$-page queue layout into a complete one
and primarily analyze the problem through the refined lens of parameterized
complexity. We obtain novel algorithms and lower bounds which provide a
detailed picture of the problem's complexity under various measures of
incompleteness, and identify surprising distinctions between queue and stack
layouts in the extension setting.

### 2. [A Fast Unsupervised Scheme for Polygonal Approximation](http://arxiv.org/pdf/2506.04664v1)

Authors: Bimal Kumar Ray

This paper proposes a fast and unsupervised scheme for a polygonal
approximation of a closed digital curve. It is demonstrated that the
approximation scheme is faster than state-of-the-art approximation and is
competitive with the same in Rosin's measure and in its aesthetic aspect. The
scheme comprises of three phases: initial segmentation, iterative vertex
insertion, and iterative merging, followed by vertex adjustment. The initial
segmentation is used to detect sharp turnings - the vertices that seemingly
have high curvature. It is likely that some of important vertices with low
curvature might have been missed out at the first phase and so iterative vertex
insertion is used to add vertices in a region where the curvature changes
slowly but steadily. The initial phase may pick up some undesirable vertices
and so merging is used to eliminate the redundant vertices. Finally, vertex
adjustment is used to facilitate enhancement in the aesthetic look of the
approximation. The quality of the approximations is measured using Rosin's
measure. The robustness of the proposed scheme with respect to geometric
transformation is observed.

### Computation and Language

### 1. [Please Translate Again: Two Simple Experiments on Whether Human-Like Reasoning Helps Translation](http://arxiv.org/pdf/2506.04521v1)

Authors: Di Wu, Seth Aycock, Christof Monz

Large Language Models (LLMs) demonstrate strong reasoning capabilities for
many tasks, often by explicitly decomposing the task via Chain-of-Thought (CoT)
reasoning. Recent work on LLM-based translation designs hand-crafted prompts to
decompose translation, or trains models to incorporate intermediate
steps.~\textit{Translating Step-by-step}~\citep{briakou2024translating}, for
instance, introduces a multi-step prompt with decomposition and refinement of
translation with LLMs, which achieved state-of-the-art results on WMT24. In
this work, we scrutinise this strategy's effectiveness. Empirically, we find no
clear evidence that performance gains stem from explicitly decomposing the
translation process, at least for the models on test; and we show that simply
prompting LLMs to ``translate again'' yields even better results than
human-like step-by-step prompting. Our analysis does not rule out the role of
reasoning, but instead invites future work exploring the factors for CoT's
effectiveness in the context of translation.

### 2. [BSBench: will your LLM find the largest prime number?](http://arxiv.org/pdf/2506.04535v1)

Authors: K. O. T. Erziev

We propose that benchmarking LLMs on questions which have no reasonable
answer actually isn't as silly as it sounds. We also present a benchmark that
allows such testing and a method to modify the existing datasets, and discover
that existing models demonstrate a performance far from the perfect on such
questions. Our code and data artifacts are available at
https://github.com/L3G5/impossible-bench

### 3. [Demonstrations of Integrity Attacks in Multi-Agent Systems](http://arxiv.org/pdf/2506.04572v1)

Authors: Can Zheng, Yuhan Cao, Xiaoning Dong, Tianxing He

Large Language Models (LLMs) have demonstrated remarkable capabilities in
natural language understanding, code generation, and complex planning.
Simultaneously, Multi-Agent Systems (MAS) have garnered attention for their
potential to enable cooperation among distributed agents. However, from a
multi-party perspective, MAS could be vulnerable to malicious agents that
exploit the system to serve self-interests without disrupting its core
functionality. This work explores integrity attacks where malicious agents
employ subtle prompt manipulation to bias MAS operations and gain various
benefits. Four types of attacks are examined: \textit{Scapegoater}, who
misleads the system monitor to underestimate other agents' contributions;
\textit{Boaster}, who misleads the system monitor to overestimate their own
performance; \textit{Self-Dealer}, who manipulates other agents to adopt
certain tools; and \textit{Free-Rider}, who hands off its own task to others.
We demonstrate that strategically crafted prompts can introduce systematic
biases in MAS behavior and executable instructions, enabling malicious agents
to effectively mislead evaluation systems and manipulate collaborative agents.
Furthermore, our attacks can bypass advanced LLM-based monitors, such as
GPT-4o-mini and o3-mini, highlighting the limitations of current detection
mechanisms. Our findings underscore the critical need for MAS architectures
with robust security protocols and content validation mechanisms, alongside
monitoring systems capable of comprehensive risk scenario assessment.

### 4. [Are LLMs Reliable Translators of Logical Reasoning Across Lexically Diversified Contexts?](http://arxiv.org/pdf/2506.04575v1)

Authors: Qingchuan Li, Jiatong Li, Zirui Liu, Mingyue Cheng, Yuting Zeng, Qi Liu, Tongxuan Liu

Neuro-symbolic approaches combining large language models (LLMs) with solvers
excels in logical reasoning problems need long reasoning chains. In this
paradigm, LLMs serve as translators, converting natural language reasoning
problems into formal logic formulas. Then reliable symbolic solvers return
correct solutions. Despite their success, we find that LLMs, as translators,
struggle to handle lexical diversification, a common linguistic phenomenon,
indicating that LLMs as logic translators are unreliable in real-world
scenarios. Moreover, existing logical reasoning benchmarks lack lexical
diversity, failing to challenge LLMs' ability to translate such text and thus
obscuring this issue. In this work, we propose SCALe, a benchmark designed to
address this significant gap through **logic-invariant lexical
diversification**. By using LLMs to transform original benchmark datasets into
lexically diversified but logically equivalent versions, we evaluate LLMs'
ability to consistently map diverse expressions to uniform logical symbols on
these new datasets. Experiments using SCALe further confirm that current LLMs
exhibit deficiencies in this capability. Building directly on the deficiencies
identified through our benchmark, we propose a new method, MenTaL, to address
this limitation. This method guides LLMs to first construct a table unifying
diverse expressions before performing translation. Applying MenTaL through
in-context learning and supervised fine-tuning (SFT) significantly improves the
performance of LLM translators on lexically diversified text. Our code is now
available at https://github.com/wufeiwuwoshihua/LexicalDiver.

### 5. [Selecting Demonstrations for Many-Shot In-Context Learning via Gradient Matching](http://arxiv.org/pdf/2506.04579v1)

Authors: Jianfei Zhang, Bei Li, Jun Bai, Rumei Li, Yanmeng Wang, Chenghua Lin, Wenge Rong

In-Context Learning (ICL) empowers Large Language Models (LLMs) for rapid
task adaptation without Fine-Tuning (FT), but its reliance on demonstration
selection remains a critical challenge. While many-shot ICL shows promising
performance through scaled demonstrations, the selection method for many-shot
demonstrations remains limited to random selection in existing work. Since the
conventional instance-level retrieval is not suitable for many-shot scenarios,
we hypothesize that the data requirements for in-context learning and
fine-tuning are analogous. To this end, we introduce a novel gradient matching
approach that selects demonstrations by aligning fine-tuning gradients between
the entire training set of the target task and the selected examples, so as to
approach the learning effect on the entire training set within the selected
examples. Through gradient matching on relatively small models, e.g.,
Qwen2.5-3B or Llama3-8B, our method consistently outperforms random selection
on larger LLMs from 4-shot to 128-shot scenarios across 9 diverse datasets. For
instance, it surpasses random selection by 4% on Qwen2.5-72B and Llama3-70B,
and by around 2% on 5 closed-source LLMs. This work unlocks more reliable and
effective many-shot ICL, paving the way for its broader application.

### 6. [MuSciClaims: Multimodal Scientific Claim Verification](http://arxiv.org/pdf/2506.04585v1)

Authors: Yash Kumar Lal, Manikanta Bandham, Mohammad Saqib Hasan, Apoorva Kashi, Mahnaz Koupaee, Niranjan Balasubramanian

Assessing scientific claims requires identifying, extracting, and reasoning
with multimodal data expressed in information-rich figures in scientific
literature. Despite the large body of work in scientific QA, figure captioning,
and other multimodal reasoning tasks over chart-based data, there are no
readily usable multimodal benchmarks that directly test claim verification
abilities. To remedy this gap, we introduce a new benchmark MuSciClaims
accompanied by diagnostics tasks. We automatically extract supported claims
from scientific articles, which we manually perturb to produce contradicted
claims. The perturbations are designed to test for a specific set of claim
verification capabilities. We also introduce a suite of diagnostic tasks that
help understand model failures. Our results show most vision-language models
are poor (~0.3-0.5 F1), with even the best model only achieving 0.77 F1. They
are also biased towards judging claims as supported, likely misunderstanding
nuanced perturbations within the claims. Our diagnostics show models are bad at
localizing correct evidence within figures, struggle with aggregating
information across modalities, and often fail to understand basic components of
the figure.

### 7. [A MISMATCHED Benchmark for Scientific Natural Language Inference](http://arxiv.org/pdf/2506.04603v1)

Authors: Firoz Shaik, Mobashir Sadat, Nikita Gautam, Doina Caragea, Cornelia Caragea

Scientific Natural Language Inference (NLI) is the task of predicting the
semantic relation between a pair of sentences extracted from research articles.
Existing datasets for this task are derived from various computer science (CS)
domains, whereas non-CS domains are completely ignored. In this paper, we
introduce a novel evaluation benchmark for scientific NLI, called MISMATCHED.
The new MISMATCHED benchmark covers three non-CS domains-PSYCHOLOGY,
ENGINEERING, and PUBLIC HEALTH, and contains 2,700 human annotated sentence
pairs. We establish strong baselines on MISMATCHED using both Pre-trained Small
Language Models (SLMs) and Large Language Models (LLMs). Our best performing
baseline shows a Macro F1 of only 78.17% illustrating the substantial headroom
for future improvements. In addition to introducing the MISMATCHED benchmark,
we show that incorporating sentence pairs having an implicit scientific NLI
relation between them in model training improves their performance on
scientific NLI. We make our dataset and code publicly available on GitHub.

### 8. [Revisiting Test-Time Scaling: A Survey and a Diversity-Aware Method for Efficient Reasoning](http://arxiv.org/pdf/2506.04611v1)

Authors: Ho-Lam Chung, Teng-Yun Hsiao, Hsiao-Ying Huang, Chunerh Cho, Jian-Ren Lin, Zhang Ziwei, Yun-Nung Chen

Test-Time Scaling (TTS) improves the reasoning performance of Large Language
Models (LLMs) by allocating additional compute during inference. We conduct a
structured survey of TTS methods and categorize them into sampling-based,
search-based, and trajectory optimization strategies. We observe that
reasoning-optimized models often produce less diverse outputs, which limits TTS
effectiveness. To address this, we propose ADAPT (A Diversity Aware Prefix
fine-Tuning), a lightweight method that applies prefix tuning with a
diversity-focused data strategy. Experiments on mathematical reasoning tasks
show that ADAPT reaches 80% accuracy using eight times less compute than strong
baselines. Our findings highlight the essential role of generative diversity in
maximizing TTS effectiveness.

### 9. [Advancing Tool-Augmented Large Language Models via Meta-Verification and Reflection Learning](http://arxiv.org/pdf/2506.04625v1)

Authors: Zhiyuan Ma, Jiayu Liu, Xianzhen Luo, Zhenya Huang, Qingfu Zhu, Wanxiang Che

Empowering large language models (LLMs) with effective tool utilization
capabilities is crucial for enabling AI agents to solve complex problems.
However, current models face two major limitations: (1) unreliable tool
planning and invocation due to low-quality instruction datasets (e.g.,
widespread hallucinated API calls), and (2) weak tool reflection abilities
(over 90% of errors cannot be corrected) resulting from static imitation
learning. To address these critical limitations, we propose Tool-MVR, a novel
Tool-Augmented LLM that achieves comprehensive System 2 reasoning through two
key innovations. Specifically, we first introduce Multi-Agent Meta-Verification
(MAMV), a systematic pipeline that rigorously validates APIs, queries, and
reasoning trajectories to construct ToolBench-V, a new high-quality instruction
dataset that addresses the limitation of unreliable tool planning and
invocation. Second, we propose Exploration-based Reflection Learning (EXPLORE),
which enhances tool reflection capabilities by leveraging tool feedback through
a dynamic "Error -> Reflection -> Correction" learning paradigm, resulting in
our reflection dataset ToolBench-R and addressing the critical weakness in tool
reflection. Finally, we obtain Tool-MVR by finetuning open-source LLMs (e.g.,
Qwen-7B) on both ToolBench-V and ToolBench-R. Our experiments demonstrate that
Tool-MVR achieves state-of-the-art performance on StableToolBench, surpassing
both ToolLLM (by 23.9%) and GPT-4 (by 15.3%) while reducing API calls by 31.4%,
with strong generalization capabilities across unseen tools and scenarios.
Additionally, on our proposed RefineToolBench, the first benchmark specifically
designed to evaluate tool reflection capabilities, Tool-MVR achieves a 58.9%
error correction rate, significantly outperforming ToolLLM's 9.1%.

### 10. [TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](http://arxiv.org/pdf/2506.04642v1)

Authors: Vinay Joshi, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum

The key-value (KV) cache in transformer models is a critical component for
efficient decoding or inference, yet its memory demands scale poorly with
sequence length, posing a major challenge for scalable deployment of large
language models. Among several approaches to KV cache compression, quantization
of key and value activations has been widely explored. Most KV cache
quantization methods still need to manage sparse and noncontiguous outliers
separately. To address this, we introduce TaDA, a training-free recipe for KV
cache compression with quantization precision that adapts to error sensitivity
across layers and a mean centering to eliminate separate outlier handling. Our
approach yields substantial accuracy improvements for multiple models
supporting various context lengths. Moreover, our approach does not need to
separately manage outlier elements -- a persistent hurdle in most traditional
quantization methods. Experiments on standard benchmarks demonstrate that our
technique reduces KV cache memory footprint to 27% of the original 16-bit
baseline while achieving comparable accuracy. Our method paves the way for
scalable and high-performance reasoning in language models by potentially
enabling inference for longer context length models, reasoning models, and
longer chain of thoughts.

### Cryptography and Security

### 1. [Incentivizing Collaborative Breach Detection](http://arxiv.org/pdf/2506.04634v1)

Authors: Mridu Nanda, Michael K. Reiter

Decoy passwords, or "honeywords," alert a site to its breach if they are ever
entered in a login attempt on that site. However, an attacker can identify a
user-chosen password from among the decoys, without risk of alerting the site
to its breach, by performing credential stuffing, i.e., entering the stolen
passwords at another site where the same user reused her password. Prior work
has thus proposed that sites monitor for the entry of their honeywords at other
sites. Unfortunately, it is not clear what incentives sites have to participate
in this monitoring. In this paper we propose and evaluate an algorithm by which
sites can exchange monitoring favors. Through a model-checking analysis, we
show that using our algorithm, a site improves its ability to detect its own
breach when it increases the monitoring effort it expends for other sites. We
additionally quantify the impacts of various parameters on detection
effectiveness and their implications for the deployment of a system to support
a monitoring ecosystem. Finally, we evaluate our algorithm on a real dataset of
breached credentials and provide a performance analysis that confirms its
scalability and practical viability.

### 2. [Authenticated Private Set Intersection: A Merkle Tree-Based Approach for Enhancing Data Integrity](http://arxiv.org/pdf/2506.04647v1)

Authors: Zixian Gong, Zhiyong Zheng, Zhe Hu, Kun Tian, Yi Zhang, Zhedanov Oleksiy, Fengxia Liu

Private Set Intersection (PSI) enables secure computation of set
intersections while preserving participant privacy, standard PSI existing
protocols remain vulnerable to data integrity attacks allowing malicious
participants to extract additional intersection information or mislead other
parties. In this paper, we propose the definition of data integrity in PSI and
construct two authenticated PSI schemes by integrating Merkle Trees with
state-of-the-art two-party volePSI and multi-party mPSI protocols. The
resulting two-party authenticated PSI achieves communication complexity
$\mathcal{O}(n \lambda+n \log n)$, aligning with the best-known unauthenticated
PSI schemes, while the multi-party construction is $\mathcal{O}(n \kappa+n \log
n)$ which introduces additional overhead due to Merkle tree inclusion proofs.
Due to the incorporation of integrity verification, our authenticated schemes
incur higher costs compared to state-of-the-art unauthenticated schemes. We
also provide efficient implementations of our protocols and discuss potential
improvements, including alternative authentication blocks.

### 3. [MULTISS: un protocole de stockage confidentiel {à} long terme sur plusieurs r{é}seaux QKD](http://arxiv.org/pdf/2506.04800v1)

Authors: Thomas Prévost, Olivier Alibart, Marc Kaplan, Anne Marin

This paper presents MULTISS, a new protocol for long-term storage distributed
across multiple Quantum Key Distribution (QKD) networks. This protocol is an
extension of LINCOS, a secure storage protocol that uses Shamir secret sharing
for secret storage on a single QKD network. Our protocol uses hierarchical
secret sharing to distribute a secret across multiple QKD networks while
ensuring perfect security. Our protocol further allows for sharing updates
without having to reconstruct the entire secret. We also prove that MULTISS is
strictly more secure than LINCOS, which remains vulnerable when its QKD network
is compromised.

### 4. [Hiding in Plain Sight: Query Obfuscation via Random Multilingual Searches](http://arxiv.org/pdf/2506.04963v1)

Authors: Anton Firc, Jan Klusáček, Kamil Malinka

Modern search engines extensively personalize results by building detailed
user profiles based on query history and behaviour. While personalization can
enhance relevance, it introduces privacy risks and can lead to filter bubbles.
This paper proposes and evaluates a lightweight, client-side query obfuscation
strategy using randomly generated multilingual search queries to disrupt user
profiling. Through controlled experiments on the Seznam.cz search engine, we
assess the impact of interleaving real queries with obfuscating noise in
various language configurations and ratios. Our findings show that while
displayed search results remain largely stable, the search engine's identified
user interests shift significantly under obfuscation. We further demonstrate
that such random queries can prevent accurate profiling and overwrite
established user profiles. This study provides practical evidence for query
obfuscation as a viable privacy-preserving mechanism and introduces a tool that
enables users to autonomously protect their search behaviour without modifying
existing infrastructure.

### 5. [Evaluating the Impact of Privacy-Preserving Federated Learning on CAN Intrusion Detection](http://arxiv.org/pdf/2506.04978v1)

Authors: Gabriele Digregorio, Elisabetta Cainazzo, Stefano Longari, Michele Carminati, Stefano Zanero

The challenges derived from the data-intensive nature of machine learning in
conjunction with technologies that enable novel paradigms such as V2X and the
potential offered by 5G communication, allow and justify the deployment of
Federated Learning (FL) solutions in the vehicular intrusion detection domain.
In this paper, we investigate the effects of integrating FL strategies into the
machine learning-based intrusion detection process for on-board vehicular
networks. Accordingly, we propose a FL implementation of a state-of-the-art
Intrusion Detection System (IDS) for Controller Area Network (CAN), based on
LSTM autoencoders. We thoroughly evaluate its detection efficiency and
communication overhead, comparing it to a centralized version of the same
algorithm, thereby presenting it as a feasible solution.

### 6. [Attack Effect Model based Malicious Behavior Detection](http://arxiv.org/pdf/2506.05001v1)

Authors: Limin Wang, Lei Bu, Muzimiao Zhang, Shihong Cang, Kai Ye

Traditional security detection methods face three key challenges: inadequate
data collection that misses critical security events, resource-intensive
monitoring systems, and poor detection algorithms with high false positive
rates. We present FEAD (Focus-Enhanced Attack Detection), a framework that
addresses these issues through three innovations: (1) an attack model-driven
approach that extracts security-critical monitoring items from online attack
reports for comprehensive coverage; (2) efficient task decomposition that
optimally distributes monitoring across existing collectors to minimize
overhead; and (3) locality-aware anomaly analysis that leverages the clustering
behavior of malicious activities in provenance graphs to improve detection
accuracy. Evaluations demonstrate FEAD achieves 8.23% higher F1-score than
existing solutions with only 5.4% overhead, confirming that focus-based designs
significantly enhance detection performance.

### 7. [OpenCCA: An Open Framework to Enable Arm CCA Research](http://arxiv.org/pdf/2506.05129v1)

Authors: Andrin Bertschi, Shweta Shinde

Confidential computing has gained traction across major architectures with
Intel TDX, AMD SEV-SNP, and Arm CCA. Unlike TDX and SEV-SNP, a key challenge in
researching Arm CCA is the absence of hardware support, forcing researchers to
develop ad-hoc performance prototypes on non-CCA Arm boards. This approach
leads to duplicated efforts, inconsistent performance comparisons, and high
barriers to entry. To address this, we present OpenCCA, an open research
platform that enables the execution of CCA-bound code on commodity Armv8.2
hardware. By systematically adapting the software stack -- including
bootloader, firmware, hypervisor, and kernel -- OpenCCA emulates CCA operations
for performance evaluation while preserving functional correctness. We
demonstrate its effectiveness with typical life-cycle measurements and
case-studies inspired by prior CCA-based papers on a easily available Armv8.2
Rockchip board that costs $250.

### 8. [SECNEURON: Reliable and Flexible Abuse Control in Local LLMs via Hybrid Neuron Encryption](http://arxiv.org/pdf/2506.05242v1)

Authors: Zhiqiang Wang, Haohua Du, Junyang Wang, Haifeng Sun, Kaiwen Guo, Haikuo Yu, Chao Liu, Xiang-Yang Li

Large language models (LLMs) with diverse capabilities are increasingly being
deployed in local environments, presenting significant security and
controllability challenges. These locally deployed LLMs operate outside the
direct control of developers, rendering them more susceptible to abuse.
Existing mitigation techniques mainly designed for cloud-based LLM services are
frequently circumvented or ineffective in deployer-controlled environments. We
propose SECNEURON, the first framework that seamlessly embeds classic access
control within the intrinsic capabilities of LLMs, achieving reliable,
cost-effective, flexible, and certified abuse control for local deployed LLMs.
SECNEURON employs neuron-level encryption and selective decryption to
dynamically control the task-specific capabilities of LLMs, limiting
unauthorized task abuse without compromising others. We first design a
task-specific neuron extraction mechanism to decouple logically related neurons
and construct a layered policy tree for handling coupled neurons. We then
introduce a flexible and efficient hybrid encryption framework for millions of
neurons in LLMs. Finally, we developed a distribution-based decrypted neuron
detection mechanism on ciphertext to ensure the effectiveness of partially
decrypted LLMs. We proved that SECNEURON satisfies IND-CPA Security and
Collusion Resistance Security under the Task Controllability Principle.
Experiments on various task settings show that SECNEURON limits unauthorized
task accuracy to below 25% while keeping authorized accuracy loss with 2%.
Using an unauthorized Code task example, the accuracy of abuse-related
malicious code generation was reduced from 59% to 15%. SECNEURON also mitigates
unauthorized data leakage, reducing PII extraction rates to below 5% and
membership inference to random guesses.

### 9. [Big Bird: Privacy Budget Management for W3C's Privacy-Preserving Attribution API](http://arxiv.org/pdf/2506.05290v1)

Authors: Pierre Tholoniat, Alison Caulfield, Giorgio Cavicchioli, Mark Chen, Nikos Goutzoulias, Benjamin Case, Asaf Cidon, Roxana Geambasu, Mathias Lécuyer, Martin Thomson

Privacy-preserving advertising APIs like Privacy-Preserving Attribution (PPA)
are designed to enhance web privacy while enabling effective ad measurement.
PPA offers an alternative to cross-site tracking with encrypted reports
governed by differential privacy (DP), but current designs lack a principled
approach to privacy budget management, creating uncertainty around critical
design decisions. We present Big Bird, a privacy budget manager for PPA that
clarifies per-site budget semantics and introduces a global budgeting system
grounded in resource isolation principles. Big Bird enforces utility-preserving
limits via quota budgets and improves global budget utilization through a novel
batched scheduling algorithm. Together, these mechanisms establish a robust
foundation for enforcing privacy protections in adversarial environments. We
implement Big Bird in Firefox and evaluate it on real-world ad data,
demonstrating its resilience and effectiveness.

### 10. [BESA: Boosting Encoder Stealing Attack with Perturbation Recovery](http://arxiv.org/pdf/2506.04556v1)

Authors: Xuhao Ren, Haotian Liang, Yajie Wang, Chuan Zhang, Zehui Xiong, Liehuang Zhu

To boost the encoder stealing attack under the perturbation-based defense
that hinders the attack performance, we propose a boosting encoder stealing
attack with perturbation recovery named BESA. It aims to overcome
perturbation-based defenses. The core of BESA consists of two modules:
perturbation detection and perturbation recovery, which can be combined with
canonical encoder stealing attacks. The perturbation detection module utilizes
the feature vectors obtained from the target encoder to infer the defense
mechanism employed by the service provider. Once the defense mechanism is
detected, the perturbation recovery module leverages the well-designed
generative model to restore a clean feature vector from the perturbed one.
Through extensive evaluations based on various datasets, we demonstrate that
BESA significantly enhances the surrogate encoder accuracy of existing encoder
stealing attacks by up to 24.63\% when facing state-of-the-art defenses and
combinations of multiple defenses.

### Computer Vision and Pattern Recognition

### 1. [EECD-Net: Energy-Efficient Crack Detection with Spiking Neural Networks and Gated Attention](http://arxiv.org/pdf/2506.04526v1)

Authors: Shuo Zhang

Crack detection on road surfaces is a critical measurement technology in the
instrumentation domain, essential for ensuring infrastructure safety and
transportation reliability. However, due to limited energy and low-resolution
imaging, smart terminal devices struggle to maintain real-time monitoring
performance. To overcome these challenges, this paper proposes a multi-stage
detection approach for road crack detection, EECD-Net, to enhance accuracy and
energy efficiency of instrumentation. Specifically, the sophisticated
Super-Resolution Convolutional Neural Network (SRCNN) is employed to address
the inherent challenges of low-quality images, which effectively enhance image
resolution while preserving critical structural details. Meanwhile, a Spike
Convolution Unit (SCU) with Continuous Integrate-and-Fire (CIF) neurons is
proposed to convert these images into sparse pulse sequences, significantly
reducing power consumption. Additionally, a Gated Attention Transformer (GAT)
module is designed to strategically fuse multi-scale feature representations
through adaptive attention mechanisms, effectively capturing both long-range
dependencies and intricate local crack patterns, and significantly enhancing
detection robustness across varying crack morphologies. The experiments on the
CrackVision12K benchmark demonstrate that EECD-Net achieves a remarkable 98.6\%
detection accuracy, surpassing state-of-the-art counterparts such as
Hybrid-Segmentor by a significant 1.5\%. Notably, the EECD-Net maintains
exceptional energy efficiency, consuming merely 5.6 mJ, which is a substantial
33\% reduction compared to baseline implementations. This work pioneers a
transformative approach in instrumentation-based crack detection, offering a
scalable, low-power solution for real-time, large-scale infrastructure
monitoring in resource-constrained environments.

### 2. [Perceptual Decoupling for Scalable Multi-modal Reasoning via Reward-Optimized Captioning](http://arxiv.org/pdf/2506.04559v1)

Authors: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Xin Jin, Zhenguo Li, James T. Kwok, Yu Zhang

Recent advances in slow-thinking language models (e.g., OpenAI-o1 and
DeepSeek-R1) have demonstrated remarkable abilities in complex reasoning tasks
by emulating human-like reflective cognition. However, extending such
capabilities to multi-modal large language models (MLLMs) remains challenging
due to the high cost of retraining vision-language alignments when upgrading
the underlying reasoner LLMs. A straightforward solution is to decouple
perception from reasoning, i.e., converting visual inputs into language
representations (e.g., captions) that are then passed to a powerful text-only
reasoner. However, this decoupling introduces a critical challenge: the visual
extractor must generate descriptions that are both faithful to the image and
informative enough to support accurate downstream reasoning. To address this,
we propose Reasoning-Aligned Perceptual Decoupling via Caption Reward
Optimization (RACRO) - a reasoning-guided reinforcement learning strategy that
aligns the extractor's captioning behavior with the reasoning objective. By
closing the perception-reasoning loop via reward-based optimization, RACRO
significantly enhances visual grounding and extracts reasoning-optimized
representations. Experiments on multi-modal math and science benchmarks show
that the proposed RACRO method achieves state-of-the-art average performance
while enabling superior scalability and plug-and-play adaptation to more
advanced reasoning LLMs without the necessity for costly multi-modal
re-alignment.

### 3. [LGM-Pose: A Lightweight Global Modeling Network for Real-time Human Pose Estimation](http://arxiv.org/pdf/2506.04561v1)

Authors: Biao Guo, Fangmin Guo, Guibo Luo, Xiaonan Luo, Feng Zhang

Most of the current top-down multi-person pose estimation lightweight methods
are based on multi-branch parallel pure CNN network architecture, which often
struggle to capture the global context required for detecting semantically
complex keypoints and are hindered by high latency due to their intricate and
redundant structures. In this article, an approximate single-branch lightweight
global modeling network (LGM-Pose) is proposed to address these challenges. In
the network, a lightweight MobileViM Block is designed with a proposed
Lightweight Attentional Representation Module (LARM), which integrates
information within and between patches using the Non-Parametric Transformation
Operation(NPT-Op) to extract global information. Additionally, a novel
Shuffle-Integrated Fusion Module (SFusion) is introduced to effectively
integrate multi-scale information, mitigating performance degradation often
observed in single-branch structures. Experimental evaluations on the COCO and
MPII datasets demonstrate that our approach not only reduces the number of
parameters compared to existing mainstream lightweight methods but also
achieves superior performance and faster processing speeds.

### 4. [Follow-Your-Creation: Empowering 4D Creation through Video Inpainting](http://arxiv.org/pdf/2506.04590v1)

Authors: Yue Ma, Kunyu Feng, Xinhua Zhang, Hongyu Liu, David Junhao Zhang, Jinbo Xing, Yinhan Zhang, Ayden Yang, Zeyu Wang, Qifeng Chen

We introduce Follow-Your-Creation, a novel 4D video creation framework
capable of both generating and editing 4D content from a single monocular video
input. By leveraging a powerful video inpainting foundation model as a
generative prior, we reformulate 4D video creation as a video inpainting task,
enabling the model to fill in missing content caused by camera trajectory
changes or user edits. To facilitate this, we generate composite masked
inpainting video data to effectively fine-tune the model for 4D video
generation. Given an input video and its associated camera trajectory, we first
perform depth-based point cloud rendering to obtain invisibility masks that
indicate the regions that should be completed. Simultaneously, editing masks
are introduced to specify user-defined modifications, and these are combined
with the invisibility masks to create a composite masks dataset. During
training, we randomly sample different types of masks to construct diverse and
challenging inpainting scenarios, enhancing the model's generalization and
robustness in various 4D editing and generation tasks. To handle temporal
consistency under large camera motion, we design a self-iterative tuning
strategy that gradually increases the viewing angles during training, where the
model is used to generate the next-stage training data after each fine-tuning
iteration. Moreover, we introduce a temporal packaging module during inference
to enhance generation quality. Our method effectively leverages the prior
knowledge of the base model without degrading its original performance,
enabling the generation of 4D videos with consistent multi-view coherence. In
addition, our approach supports prompt-based content editing, demonstrating
strong flexibility and significantly outperforming state-of-the-art methods in
both quality and versatility.

### 5. [Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](http://arxiv.org/pdf/2506.04595v1)

Authors: Ziqi Jia, Anmin Wang, Xiaoyang Qu, Xiaowen Yang, Jianzong Wang

Previous continual learning setups for embodied intelligence focused on
executing low-level actions based on human commands, neglecting the ability to
learn high-level planning and multi-level knowledge. To address these issues,
we propose the Hierarchical Embodied Continual Learning Setups (HEC) that
divide the agent's continual learning process into two layers: high-level
instructions and low-level actions, and define five embodied continual learning
sub-setups. Building on these setups, we introduce the Task-aware Mixture of
Incremental LoRA Experts (Task-aware MoILE) method. This approach achieves task
recognition by clustering visual-text embeddings and uses both a task-level
router and a token-level router to select the appropriate LoRA experts. To
effectively address the issue of catastrophic forgetting, we apply Singular
Value Decomposition (SVD) to the LoRA parameters obtained from prior tasks,
preserving key components while orthogonally training the remaining parts. The
experimental results show that our method stands out in reducing the forgetting
of old tasks compared to other methods, effectively supporting agents in
retaining prior knowledge while continuously learning new tasks.

### 6. [SmartAvatar: Text- and Image-Guided Human Avatar Generation with VLM AI Agents](http://arxiv.org/pdf/2506.04606v1)

Authors: Alexander Huang-Menders, Xinhang Liu, Andy Xu, Yuyao Zhang, Chi-Keung Tang, Yu-Wing Tai

SmartAvatar is a vision-language-agent-driven framework for generating fully
rigged, animation-ready 3D human avatars from a single photo or textual prompt.
While diffusion-based methods have made progress in general 3D object
generation, they continue to struggle with precise control over human identity,
body shape, and animation readiness. In contrast, SmartAvatar leverages the
commonsense reasoning capabilities of large vision-language models (VLMs) in
combination with off-the-shelf parametric human generators to deliver
high-quality, customizable avatars. A key innovation is an autonomous
verification loop, where the agent renders draft avatars, evaluates facial
similarity, anatomical plausibility, and prompt alignment, and iteratively
adjusts generation parameters for convergence. This interactive, AI-guided
refinement process promotes fine-grained control over both facial and body
features, enabling users to iteratively refine their avatars via
natural-language conversations. Unlike diffusion models that rely on static
pre-trained datasets and offer limited flexibility, SmartAvatar brings users
into the modeling loop and ensures continuous improvement through an LLM-driven
procedural generation and verification system. The generated avatars are fully
rigged and support pose manipulation with consistent identity and appearance,
making them suitable for downstream animation and interactive applications.
Quantitative benchmarks and user studies demonstrate that SmartAvatar
outperforms recent text- and image-driven avatar generation systems in terms of
reconstructed mesh quality, identity fidelity, attribute accuracy, and
animation readiness, making it a versatile tool for realistic, customizable
avatar creation on consumer-grade hardware.

### 7. [Perfecting Depth: Uncertainty-Aware Enhancement of Metric Depth](http://arxiv.org/pdf/2506.04612v1)

Authors: Jinyoung Jun, Lei Chu, Jiahao Li, Yan Lu, Chang-Su Kim

We propose a novel two-stage framework for sensor depth enhancement, called
Perfecting Depth. This framework leverages the stochastic nature of diffusion
models to automatically detect unreliable depth regions while preserving
geometric cues. In the first stage (stochastic estimation), the method
identifies unreliable measurements and infers geometric structure by leveraging
a training-inference domain gap. In the second stage (deterministic
refinement), it enforces structural consistency and pixel-level accuracy using
the uncertainty map derived from the first stage. By combining stochastic
uncertainty modeling with deterministic refinement, our method yields dense,
artifact-free depth maps with improved reliability. Experimental results
demonstrate its effectiveness across diverse real-world scenarios. Furthermore,
theoretical analysis, various experiments, and qualitative visualizations
validate its robustness and scalability. Our framework sets a new baseline for
sensor depth enhancement, with potential applications in autonomous driving,
robotics, and immersive technologies.

### 8. [Deep Learning Reforms Image Matching: A Survey and Outlook](http://arxiv.org/pdf/2506.04619v1)

Authors: Shihua Zhang, Zizhuo Li, Kaining Zhang, Yifan Lu, Yuxin Deng, Linfeng Tang, Xingyu Jiang, Jiayi Ma

Image matching, which establishes correspondences between two-view images to
recover 3D structure and camera geometry, serves as a cornerstone in computer
vision and underpins a wide range of applications, including visual
localization, 3D reconstruction, and simultaneous localization and mapping
(SLAM). Traditional pipelines composed of ``detector-descriptor, feature
matcher, outlier filter, and geometric estimator'' falter in challenging
scenarios. Recent deep-learning advances have significantly boosted both
robustness and accuracy. This survey adopts a unique perspective by
comprehensively reviewing how deep learning has incrementally transformed the
classical image matching pipeline. Our taxonomy highly aligns with the
traditional pipeline in two key aspects: i) the replacement of individual steps
in the traditional pipeline with learnable alternatives, including learnable
detector-descriptor, outlier filter, and geometric estimator; and ii) the
merging of multiple steps into end-to-end learnable modules, encompassing
middle-end sparse matcher, end-to-end semi-dense/dense matcher, and pose
regressor. We first examine the design principles, advantages, and limitations
of both aspects, and then benchmark representative methods on relative pose
recovery, homography estimation, and visual localization tasks. Finally, we
discuss open challenges and outline promising directions for future research.
By systematically categorizing and evaluating deep learning-driven strategies,
this survey offers a clear overview of the evolving image matching landscape
and highlights key avenues for further innovation.

### 9. [Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations](http://arxiv.org/pdf/2506.04633v1)

Authors: Linjie Li, Mahtab Bigverdi, Jiawei Gu, Zixian Ma, Yinuo Yang, Ziang Li, Yejin Choi, Ranjay Krishna

Spatial cognition is essential for human intelligence, enabling
problem-solving through visual simulations rather than solely relying on verbal
reasoning. However, existing AI benchmarks primarily assess verbal reasoning,
neglecting the complexities of non-verbal, multi-step visual simulation. We
introduce STARE(Spatial Transformations and Reasoning Evaluation), a benchmark
designed to rigorously evaluate multimodal large language models on tasks
better solved through multi-step visual simulation. STARE features 4K tasks
spanning foundational geometric transformations (2D and 3D), integrated spatial
reasoning (cube net folding and tangram puzzles), and real-world spatial
reasoning (perspective and temporal reasoning), reflecting practical cognitive
challenges like object assembly, mechanical diagram interpretation, and
everyday spatial navigation. Our evaluations show that models excel at
reasoning over simpler 2D transformations, but perform close to random chance
on more complex tasks like 3D cube net folding and tangram puzzles that require
multi-step visual simulations. Humans achieve near-perfect accuracy but take
considerable time (up to 28.9s) on complex tasks, significantly speeding up
(down by 7.5 seconds on average) with intermediate visual simulations. In
contrast, models exhibit inconsistent performance gains from visual
simulations, improving on most tasks but declining in specific cases like
tangram puzzles (GPT-4o, o1) and cube net folding (Claude-3.5, Gemini-2.0
Flash), indicating that models may not know how to effectively leverage
intermediate visual information.

### 10. [Text-Aware Real-World Image Super-Resolution via Diffusion Model with Joint Segmentation Decoders](http://arxiv.org/pdf/2506.04641v1)

Authors: Qiming Hu, Linlong Fan, Yiyan Luo, Yuhang Yu, Xiaojie Guo, Qingnan Fan

The introduction of generative models has significantly advanced image
super-resolution (SR) in handling real-world degradations. However, they often
incur fidelity-related issues, particularly distorting textual structures. In
this paper, we introduce a novel diffusion-based SR framework, namely TADiSR,
which integrates text-aware attention and joint segmentation decoders to
recover not only natural details but also the structural fidelity of text
regions in degraded real-world images. Moreover, we propose a complete pipeline
for synthesizing high-quality images with fine-grained full-image text masks,
combining realistic foreground text regions with detailed background content.
Extensive experiments demonstrate that our approach substantially enhances text
legibility in super-resolved images, achieving state-of-the-art performance
across multiple evaluation metrics and exhibiting strong generalization to
real-world scenarios. Our code is available at
\href{https://github.com/mingcv/TADiSR}{here}.

### Computers and Society

### 1. [Skill-Driven Certification Pathways: Measuring Industry Training Impact on Graduate Employability](http://arxiv.org/pdf/2506.04588v1)

Authors: Anatoli Kovalev, Narelle Stefanac, Marian-Andrei Rizoiu

Australia faces a critical technology skills shortage, requiring
approximately $52,000$ new technology professionals annually by 2030, while
confronting a widening gap between employer requirements and graduate
capabilities. With only $1\%$ of technology graduates considered immediately
work-ready, traditional educational pathways alone prove insufficient to meet
industry demands. This research examines how industry certifications, such as
Microsoft's AI-900 (Azure AI Fundamentals), can bridge this critical skills
gap. We propose a novel, data-driven methodology that quantitatively measures
skill alignment between educational offerings and job market requirements by
analysing over 2.5 million job advertisements from Australia, the US, and the
UK, mapping extracted skills to industry taxonomies using the Vectorised Skills
Space Method. Our findings reveal that combining university degrees with
targeted industry certifications significantly enhances employability for
technology roles. The Bachelor of Computer Science with AI major combined with
AI-900 certification achieved the highest absolute skill similarity score for
Machine Learning Engineer positions. Surprisingly, the largest improvements
when augmented with AI certifications are experiences by non-technical
degrees--such as nursing nursing--with up to $9,296\%$ percentage improvements
in alignment with Machine Learning Engineer roles. Our results challenge
conventional assumptions about technology career pathways. They can provide
actionable insights for educational institutions seeking evidence-based
curriculum design, students requiring strategic certification guidance, and
employers recognising potential in candidates from non-traditional backgrounds
who have obtained relevant certifications.

### 2. [The Data Dilemma: Authors' Intentions and Recognition of Research Data in Educational Technology Research](http://arxiv.org/pdf/2506.04954v1)

Authors: Sandra Schulz, Natalie Kiesler

Educational Technology (EdTec) research is conducted by multiple disciplines,
some of which annually meet at the DELFI conference. Due to the heterogeneity
of involved researchers and communities, it is our goal to identify categories
of research data overseen in the context of EdTec research. Therefore, we
analyze the author's perspective provided via EasyChair where authors specified
whether they had research data to share. We compared this information with an
analysis of the submitted articles and the contained research data. We found
that not all research data was recognized as such by the authors, especially
software and qualitative data, indicating a prevailing lack of awareness, and
other potential barriers. In addition, we analyze the 2024 DELFI proceedings to
learn what kind of data was subject to research, and where it is published.
This work has implications for training future generations of EdTec
researchers. It further stresses the need for guidelines and recognition of
research data publications (particularly software, and qualitative data).

### 3. [Evaluating Prompt-Driven Chinese Large Language Models: The Influence of Persona Assignment on Stereotypes and Safeguards](http://arxiv.org/pdf/2506.04975v1)

Authors: Geng Liu, Li Feng, Carlo Alberto Bono, Songbo Yang, Mengxiao Zhu, Francesco Pierri

Recent research has highlighted that assigning specific personas to large
language models (LLMs) can significantly increase harmful content generation.
Yet, limited attention has been given to persona-driven toxicity in non-Western
contexts, particularly in Chinese-based LLMs. In this paper, we perform a
large-scale, systematic analysis of how persona assignment influences refusal
behavior and response toxicity in Qwen, a widely-used Chinese language model.
Utilizing fine-tuned BERT classifiers and regression analysis, our study
reveals significant gender biases in refusal rates and demonstrates that
certain negative personas can amplify toxicity toward Chinese social groups by
up to 60-fold compared to the default model. To mitigate this toxicity, we
propose an innovative multi-model feedback strategy, employing iterative
interactions between Qwen and an external evaluator, which effectively reduces
toxic outputs without costly model retraining. Our findings emphasize the
necessity of culturally specific analyses for LLMs safety and offer a practical
framework for evaluating and enhancing ethical alignment in LLM-generated
content.

### 4. [Early linguistic fingerprints of online users who engage with conspiracy communities](http://arxiv.org/pdf/2506.05086v1)

Authors: Francesco Corso, Giuseppe Russo, Francesco Pierri, Gianmarco De Francisci Morales

Online social media platforms are often seen as catalysts for radicalization,
as they provide spaces where extreme beliefs can take root and spread,
sometimes leading to real-world consequences. Conspiracy theories represent a
specific form of radicalization that is notoriously resistant to online
moderation strategies. One explanation for this resilience is the presence of a
"conspiratorial mindset", a cognitive framework that fundamentally shapes how
conspiracy believers perceive reality. However, the role of this mindset in
driving online user behavior remains poorly understood. In this study, we
analyze the psycholinguistic patterns of Reddit users who become active in a
prominent conspiracy community by examining their activity in mainstream
communities, which allows us to isolate linguistic markers for the presence of
a conspiratorial mindset. We find that conspiracy-engaged individuals exhibit
distinct psycholinguistic fingerprints, setting them apart from the general
user population. Crucially, this signal is already evident in their online
activity prior to joining the conspiracy community, allowing us to predict
their involvement years in advance. These findings suggest that individuals who
adopt conspiracy beliefs do not radicalize through community involvement, but
possess a pre-existing conspiratorial mindset, which predisposes them to seek
out and join extreme communities. By challenging the view that online social
media platforms actively radicalize users into conspiracy theory beliefs, our
findings suggest that standard moderation strategies have limited impact on
curbing radicalization, and highlight the need for more targeted, supportive
interventions that encourage disengagement from extremist narratives.
Ultimately, this work contributes to fostering safer online and offline
environments for public discourse.

### 5. [A Framework for Ethical Judgment of Smart City Applications](http://arxiv.org/pdf/2506.05172v1)

Authors: Weichen Shi

As modern cities increasingly adopt a variety of sensors and Internet of
Things (IoT) technologies to collect and analyze data about residents,
environments, and public services, they are fostering greater interactions
among smart city applications, residents, governments, and businesses. This
trend makes it essential for regulators to focus on these interactions to
manage smart city practices effectively and prevent unethical outcomes. To
facilitate ethical analysis for smart city applications, this paper introduces
a judgment framework that examines various scenarios where ethical issues may
arise. Employing a multi-agent approach, the framework incorporates diverse
social entities and applies logic-based ethical rules to identify potential
violations. Through a rights-based analysis, we developed a set of 13 ethical
principles and rules to guide ethical practices in smart cities. We utilized
two specification languages, Prototype Verification System (PVS) and Alloy, to
model our multi-agent system. Our analysis suggests that Alloy may be more
efficient for formalizing smart cities and conducting ethical rule checks,
particularly with the assistance of a human evaluator. Simulations of a
real-world smart city application demonstrate that our ethical judgment
framework effectively detects unethical outcomes and can be extended for
practical use.

### 6. [Oversight Structures for Agentic AI in Public-Sector Organizations](http://arxiv.org/pdf/2506.04836v1)

Authors: Chris Schmitz, Jonathan Rystrøm, Jan Batzner

This paper finds that the introduction of agentic AI systems intensifies
existing challenges to traditional public sector oversight mechanisms -- which
rely on siloed compliance units and episodic approvals rather than continuous,
integrated supervision. We identify five governance dimensions essential for
responsible agent deployment: cross-departmental implementation, comprehensive
evaluation, enhanced security protocols, operational visibility, and systematic
auditing. We evaluate the capacity of existing oversight structures to meet
these challenges, via a mixed-methods approach consisting of a literature
review and interviews with civil servants in AI-related roles. We find that
agent oversight poses intensified versions of three existing governance
challenges: continuous oversight, deeper integration of governance and
operational capabilities, and interdepartmental coordination. We propose
approaches that both adapt institutional structures and design agent oversight
compatible with public sector constraints.

### 7. [Intentionally Unintentional: GenAI Exceptionalism and the First Amendment](http://arxiv.org/pdf/2506.05211v1)

Authors: David Atkinson, Jena D. Hwang, Jacob Morrison

This paper challenges the assumption that courts should grant First Amendment
protections to outputs from large generative AI models, such as GPT-4 and
Gemini. We argue that because these models lack intentionality, their outputs
do not constitute speech as understood in the context of established legal
precedent, so there can be no speech to protect. Furthermore, if the model
outputs are not speech, users cannot claim a First Amendment speech right to
receive the outputs. We also argue that extending First Amendment rights to AI
models would not serve the fundamental purposes of free speech, such as
promoting a marketplace of ideas, facilitating self-governance, or fostering
self-expression. In fact, granting First Amendment protections to AI models
would be detrimental to society because it would hinder the government's
ability to regulate these powerful technologies effectively, potentially
leading to the unchecked spread of misinformation and other harms.

### 8. [User Altruism in Recommendation Systems](http://arxiv.org/pdf/2506.04525v1)

Authors: Ekaterina Fedorova, Madeline Kitch, Chara Podimata

Users of social media platforms based on recommendation systems (RecSys)
(e.g. TikTok, X, YouTube) strategically interact with platform content to
influence future recommendations. On some such platforms, users have been
documented to form large-scale grassroots movements encouraging others to
purposefully interact with algorithmically suppressed content in order to
"boost" its recommendation; we term this behavior user altruism. To capture
this behavior, we study a game between users and a RecSys, where users provide
the RecSys (potentially manipulated) preferences over the contents available to
them, and the RecSys -- limited by data and computation constraints -- creates
a low-rank approximation preference matrix, and ultimately provides each user
her (approximately) most-preferred item. We compare the users' social welfare
under truthful preference reporting and under a class of strategies capturing
user altruism. In our theoretical analysis, we provide sufficient conditions to
ensure strict increases in user social welfare under user altruism, and provide
an algorithm to find an effective altruistic strategy. Interestingly, we show
that for commonly assumed recommender utility functions, effectively altruistic
strategies also improve the utility of the RecSys! We show that our results are
robust to several model misspecifications, thus strengthening our conclusions.
Our theoretical analysis is complemented by empirical results of effective
altruistic strategies on the GoodReads dataset, and an online survey on how
real-world users behave altruistically in RecSys. Overall, our findings serve
as a proof-of-concept of the reasons why traditional RecSys may incentivize
users to form collectives and/or follow altruistic strategies when interacting
with them.

### 9. [Judicial Permission](http://arxiv.org/pdf/2506.04610v1)

Authors: Guido Governatori, Antonino Rotolo

This paper examines the significance of weak permissions in criminal trials
(\emph{judicial permission}). It introduces a dialogue game model to
systematically address judicial permissions, considering different standards of
proof and argumentation semantics.

### 10. [Beyond the Desktop: XR-Driven Segmentation with Meta Quest 3 and MX Ink](http://arxiv.org/pdf/2506.04858v1)

Authors: Lisle Faray de Paiva, Gijs Luijten, Ana Sofia Ferreira Santos, Moon Kim, Behrus Puladi, Jens Kleesiek, Jan Egger

Medical imaging segmentation is essential in clinical settings for diagnosing
diseases, planning surgeries, and other procedures. However, manual annotation
is a cumbersome and effortful task. To mitigate these aspects, this study
implements and evaluates the usability and clinical applicability of an
extended reality (XR)-based segmentation tool for anatomical CT scans, using
the Meta Quest 3 headset and Logitech MX Ink stylus. We develop an immersive
interface enabling real-time interaction with 2D and 3D medical imaging data in
a customizable workspace designed to mitigate workflow fragmentation and
cognitive demands inherent to conventional manual segmentation tools. The
platform combines stylus-driven annotation, mirroring traditional pen-on-paper
workflows, with instant 3D volumetric rendering. A user study with a public
craniofacial CT dataset demonstrated the tool's foundational viability,
achieving a System Usability Scale (SUS) score of 66, within the expected range
for medical applications. Participants highlighted the system's intuitive
controls (scoring 4.1/5 for self-descriptiveness on ISONORM metrics) and
spatial interaction design, with qualitative feedback highlighting strengths in
hybrid 2D/3D navigation and realistic stylus ergonomics. While users identified
opportunities to enhance task-specific precision and error management, the
platform's core workflow enabled dynamic slice adjustment, reducing cognitive
load compared to desktop tools. Results position the XR-stylus paradigm as a
promising foundation for immersive segmentation tools, with iterative
refinements targeting haptic feedback calibration and workflow personalization
to advance adoption in preoperative planning.

### Databases

### 1. [BVLSM: Write-Efficient LSM-Tree Storage via WAL-Time Key-Value Separation](http://arxiv.org/pdf/2506.04678v1)

Authors: Ming Li, Wendi Cheng, Jiahe Wei, Xueqiang Shan, Liu Weikai, Xiaonan Zhao, Xiao Zhang

Modern data-intensive applications increasingly store and process big-value
items, such as multimedia objects and machine learning embeddings, which
exacerbate storage inefficiencies in Log-Structured Merge-Tree (LSM)-based
key-value stores. This paper presents BVLSM, a Write-Ahead Log (WAL)-time
key-value separation mechanism designed to address three key challenges in
LSM-Tree storage systems: write amplification, poor memory utilization, and I/O
jitter under big-value workloads. Unlike state-of-the-art approaches that delay
key-value separation until the flush stage, leading to redundant data in
MemTables and repeated writes. BVLSM proactively decouples keys and values
during the WAL phase. The MemTable stores only lightweight metadata, allowing
multi-queue parallel store for big value. The benchmark results show that BVLSM
significantly outperforms both RocksDB and BlobDB under 64KB random write
workloads. In asynchronous WAL mode, it achieves throughput improvements of
7.6x over RocksDB and 1.9x over BlobDB.

### 2. [Memory Hierarchy Design for Caching Middleware in the Age of NVM](http://arxiv.org/pdf/2506.05071v1)

Authors: Shahram Ghandeharizadeh, Sandy Irani, Jenny Lam

Advances in storage technology have introduced Non-Volatile Memory, NVM, as a
new storage medium. NVM, along with Dynamic Random Access Memory (DRAM), Solid
State Disk (SSD), and Disk present a system designer with a wide array of
options in designing caching middleware. Moreover, design decisions to
replicate a data item in more than one level of a caching memory hierarchy may
enhance the overall system performance with a faster recovery time in the event
of a memory failure. Given a fixed budget, the key configuration questions are:
Which storage media should constitute the memory hierarchy? What is the storage
capacity of each hierarchy? Should data be replicated or partitioned across the
different levels of the hierarchy? We model these cache configuration questions
as an instance of the Multiple Choice Knapsack Problem (MCKP). This model is
guided by the specification of each type of memory along with an application's
database characteristics and its workload. Although MCKP is NP-complete, its
linear programming relaxation is efficiently solvable and can be used to
closely approximate the optimal solution. We use the resulting simple algorithm
to evaluate design tradeoffs in the context of a memory hierarchy for a
Key-Value Store (e.g., memcached) as well as a host-side cache (e.g.,
Flashcache). The results show selective replication is appropriate with certain
failure rates and workload characteristics. With a slim failure rate and
frequent data updates, tiering of data across the different storage media that
constitute the cache is superior to replication.

### Distributed, Parallel, and Cluster Computing

### 1. [Distributed system perspective on Backscatter systems](http://arxiv.org/pdf/2506.04833v1)

Authors: Jincheng Guan, Jun Zhang

Backscatter system is a system based on backscatter communication technology,
which is a low cost, low power consumption and easy to deploy communication
technology. At present, the backscatter technology is mainly applied to RFID
tags and the Internet of Things and other fields. With the rapid development of
the Internet of Things, the application of backscatter systems is increasing.
Moreover, the backscatter system is essentially a distributed system, but
existing research rarely conducts studies and analyses from a distributed
perspective. This paper conducts a study on the backscattering system from the
perspective of distributed systems, comprehensively reviewing the basic
principles of the backscattering system, and analyzing the distributed system
architectures of different backscattering systems. Then, it introduces the
application scenarios, research status and challenges of the backscattering
system, and finally discusses the future research directions of the
backscattering system, hoping to provide references for future research.

### 2. [A distributed system perspective on Backscatter systems: A review](http://arxiv.org/pdf/2506.04873v1)

Authors: Tonghuan Xiao, Jiecheng Zhou

This review investigates the pivotal role of distributed architectures and
intelligent resource allocation in enabling robust and scalable wireless
systems, with a particular emphasis on backscatter communication, indoor
localization, battery-free networks, and Simultaneous Wireless Information and
Power Transfer (SWIPT).

### 3. [Inference economics of language models](http://arxiv.org/pdf/2506.04645v1)

Authors: Ege Erdil

We develop a theoretical model that addresses the economic trade-off between
cost per token versus serial token generation speed when deploying LLMs for
inference at scale. Our model takes into account arithmetic, memory bandwidth,
network bandwidth and latency constraints; and optimizes over different
parallelism setups and batch sizes to find the ones that optimize serial
inference speed at a given cost per token. We use the model to compute Pareto
frontiers of serial speed versus cost per token for popular language models.

### 4. [A highly scalable numerical framework for reservoir simulation on UG4 platform](http://arxiv.org/pdf/2506.04763v1)

Authors: Shuai Lu

The modeling and simulation of multiphase fluid flow receive significant
attention in reservoir engineering. Many time discretization schemes for
multiphase flow equations are either explicit or semi-implicit, relying on the
decoupling between the saturation equation and the pressure equation. In this
study, we delve into a fully coupled and fully implicit framework for
simulating multiphase flow in heterogeneous porous media, considering gravity
and capillary effects. We utilize the Vertex-Centered Finite Volume Method for
spatial discretization and propose an efficient implementation of interface
conditions for heterogeneous porous media within the current scheme. Notably,
we introduce the Linearly Implicit Extrapolation Method (LIMEX) with an error
estimator, adapted for the first time to multiphase flow problems. To solve the
resulting linear system, we employ the BiCGSTAB method with the Geometric
Multigrid (GMG) preconditioner. The implementations of models and methods are
based on the open-source software: UG4. The results from parallel computations
on the supercomputer demonstrate that the scalability of our proposed framework
is sufficient, supporting a scale of thousands of processors with Degrees of
Freedom (DoF) extending up to billions.

### 5. [Improved Byzantine Agreement under an Adaptive Adversary](http://arxiv.org/pdf/2506.04919v1)

Authors: Fabien Dufoulon, Gopal Pandurangan

Byzantine agreement is a fundamental problem in fault-tolerant distributed
computing that has been studied intensively for the last four decades. Much of
the research has focused on a static Byzantine adversary, where the adversary
is constrained to choose the Byzantine nodes in advance of the protocol's
execution. This work focuses on the harder case of an adaptive Byzantine
adversary that can choose the Byzantine nodes \emph{adaptively} based on the
protocol's execution. While efficient $O(\log n)$-round protocols ($n$ is the
total number of nodes) are known for the static adversary (Goldwasser, Pavlov,
and Vaikuntanathan, FOCS 2006) tolerating up to $t < n/(3+\epsilon)$ Byzantine
nodes, $\Omega(t/\sqrt{n \log n})$ rounds is a well-known lower bound for
adaptive adversary [Bar-Joseph and Ben-Or, PODC 1998]. The best-known protocol
for adaptive adversary runs in $O(t/\log n)$ rounds [Chor and Coan, IEEE Trans.
Soft. Engg., 1985].
  This work presents a synchronous randomized Byzantine agreement protocol
under an adaptive adversary that improves over previous results. Our protocol
works under the powerful \emph{adaptive rushing adversary in the full
information model}. That is, we assume that the Byzantine nodes can behave
arbitrarily and maliciously, have knowledge about the entire state of the
network at every round, including random choices made by all the nodes up to
and including the current round, have unlimited computational power, and may
collude among themselves. Furthermore, the adversary can \emph{adaptively}
corrupt up to $t < n/3$ nodes based on the protocol's execution. We present a
simple randomized Byzantine agreement protocol that runs in $O(\min\{t^2\log
n/n, t/\log n\})$ rounds that improves over the long-standing bound of
$O(t/\log n)$ rounds due to Chor and Coan [IEEE Trans. Soft. Engg., 1985].

### 6. [Federated Isolation Forest for Efficient Anomaly Detection on Edge IoT Systems](http://arxiv.org/pdf/2506.05138v1)

Authors: Pavle Vasiljevic, Milica Matic, Miroslav Popovic

Recently, federated learning frameworks such as Python TestBed for Federated
Learning Algorithms and MicroPython TestBed for Federated Learning Algorithms
have emerged to tackle user privacy concerns and efficiency in embedded
systems. Even more recently, an efficient federated anomaly detection
algorithm, FLiForest, based on Isolation Forests has been developed, offering a
low-resource, unsupervised method well-suited for edge deployment and
continuous learning. In this paper, we present an application of Isolation
Forest-based temperature anomaly detection, developed using the previously
mentioned federated learning frameworks, aimed at small edge devices and IoT
systems running MicroPython. The system has been experimentally evaluated,
achieving over 96% accuracy in distinguishing normal from abnormal readings and
above 78% precision in detecting anomalies across all tested configurations,
while maintaining a memory usage below 160 KB during model training. These
results highlight its suitability for resource-constrained environments and
edge systems, while upholding federated learning principles of data privacy and
collaborative learning.

### 7. [FlashDMoE: Fast Distributed MoE in a Single Kernel](http://arxiv.org/pdf/2506.04667v1)

Authors: Osayamen Jonathan Aimuyo, Byungsoo Oh, Rachee Singh

The computational sparsity of Mixture-of-Experts (MoE) models enables
sub-linear growth in compute cost as model size increases, offering a scalable
path to training massive neural networks. However, existing implementations
suffer from \emph{low GPU utilization}, \emph{significant latency overhead},
and a fundamental \emph{inability to leverage task locality}, primarily due to
CPU-managed scheduling, host-initiated communication, and frequent kernel
launches. To overcome these limitations, we develop FlashDMoE, a fully
GPU-resident MoE operator that fuses expert computation and inter-GPU
communication into a \emph{single persistent GPU kernel}. FlashDMoE enables
fine-grained pipelining of dispatch, compute, and combine phases, eliminating
launch overheads and reducing idle gaps. Its device-initiated communication
protocol introduces \emph{payload-efficient} data transfers, significantly
shrinking buffer sizes in sparsely activated MoE layers. When evaluated on a
single 8-H100 GPU node with MoE models having up to 128 experts and 16K token
sequences, FlashDMoE achieves up to \textbf{6}x lower latency, \textbf{5,7}x
higher throughput, \textbf{4}x better weak scaling efficiency, and \textbf{9}x
higher GPU utilization compared to state-of-the-art baselines, despite using
FP32 while baselines use FP16. FlashDMoE demonstrates that principled GPU
kernel-hardware co-design is key to unlocking the performance ceiling of
large-scale distributed ML workloads.

### 8. [Energy-Optimized Scheduling for AIoT Workloads Using TOPSIS](http://arxiv.org/pdf/2506.04902v1)

Authors: Preethika Pradeep, Eyhab Al-Masri

AIoT workloads demand energy-efficient orchestration across cloud-edge
infrastructures, but Kubernetes' default scheduler lacks multi-criteria
optimization for heterogeneous environments. This paper presents GreenPod, a
TOPSIS-based scheduler optimizing pod placement based on execution time, energy
consumption, processing core, memory availability, and resource balance. Tested
on a heterogeneous Google Kubernetes cluster, GreenPod improves energy
efficiency by up to 39.1% over the default Kubernetes (K8s) scheduler,
particularly with energy-centric weighting schemes. Medium complexity workloads
showed the highest energy savings, despite slight scheduling latency. GreenPod
effectively balances sustainability and performance for AIoT applications.

### 9. [Becoming Immutable: How Ethereum is Made](http://arxiv.org/pdf/2506.04940v1)

Authors: Andrea Canidio, Vabuk Pahari

We analyze blocks proposed for inclusion in the Ethereum blockchain during 8
minutes on December 3rd, 2024. Our dataset comprises 38 winning blocks, 15,097
proposed blocks, 10,793 unique transactions, and 2,380,014 transaction-block
pairings. We find that exclusive transactions--transactions present only in
blocks proposed by a single builder--account for 85% of the fees paid by all
transactions included in winning blocks. We also find that a surprisingly large
number of user transactions are delayed: although proposed during a bidding
cycle, they are not included in the corresponding winning block. Many such
delayed transactions are exclusive to a losing builder. We also identify two
arbitrage bots trading between decentralized (DEX) and centralized exchanges
(CEX). By examining their bidding dynamics, we estimate that the implied price
at which these bots trade USDC/WETH and USDT/WETH on CEXes is between 3.4 and
4.2 basis points better than the contemporaneous price reported on Binance.

### 10. [Tight analyses of first-order methods with error feedback](http://arxiv.org/pdf/2506.05271v1)

Authors: Daniel Berg Thomsen, Adrien Taylor, Aymeric Dieuleveut

Communication between agents often constitutes a major computational
bottleneck in distributed learning. One of the most common mitigation
strategies is to compress the information exchanged, thereby reducing
communication overhead. To counteract the degradation in convergence associated
with compressed communication, error feedback schemes -- most notably
$\mathrm{EF}$ and $\mathrm{EF}^{21}$ -- were introduced. In this work, we
provide a tight analysis of both of these methods. Specifically, we find the
Lyapunov function that yields the best possible convergence rate for each
method -- with matching lower bounds. This principled approach yields sharp
performance guarantees and enables a rigorous, apples-to-apples comparison
between $\mathrm{EF}$, $\mathrm{EF}^{21}$, and compressed gradient descent. Our
analysis is carried out in a simplified yet representative setting, which
allows for clean theoretical insights and fair comparison of the underlying
mechanisms.

### Discrete Mathematics

### 1. [Temporal passing network in basketball: the effect of time pressure on the dynamics of team organization at micro and meso levels](http://arxiv.org/pdf/2506.04808v1)

Authors: Quentin Bourgeais, Rodolphe Charrier, Eric Sanlaville, Ludovic Seifert

In this study, basketball teams are conceptualized as complex adaptive
systems to examine their (re)organizational processes in response the time
remaining to shoot. Using temporal passing networks to model team behavior, the
focus is on the dynamics of the temporal patterns of interaction between
players. Several metrics grounded in social network analysis are calculated at
different level to assess the dynamics of the patterns used by teams and of the
individual roles within those patterns. The results reveal a 3-phase dynamic,
differentiated by more or less complex and diversified patterns, and by more or
less specialized or flexible roles. Additionally, time-dependent features of
the different tactical playing positions are identified, some of which linked
to team performance. The findings are intended to explain how basketball teams
adapt their organization to cope with time pressure, offering potential
insights for other type of teams facing similar constraints. Moreover, this
work provides a useful framework for a multilevel understanding of how
constraints shape team adaptations dynamically, making it applicable to a wide
range of team settings.

### 2. [Misère Greedy Nim and Misère Bounded Greedy Nim](http://arxiv.org/pdf/2506.04657v1)

Authors: Nanako Omiya, Ryo Yoshinaka, Ayumi Shinohara

In this paper, we analyze the mis\`ere versions of two impartial
combinatorial games: k-Bounded Greedy Nim and Greedy Nim. We present a complete
solution to both games by showing necessary and sufficient conditions for a
position to be P-positions.

### 3. [Decomposing Words for Enhanced Compression: Exploring the Number of Runs in the Extended Burrows-Wheeler Transform](http://arxiv.org/pdf/2506.04926v1)

Authors: Florian Ingels, Anaïs Denis, Bastien Cazaux

The Burrows-Wheeler Transform (BWT) is a fundamental component in many data
structures for text indexing and compression, widely used in areas such as
bioinformatics and information retrieval. The extended BWT (eBWT) generalizes
the classical BWT to multisets of strings, providing a flexible framework that
captures many BWT-like constructions. Several known variants of the BWT can be
viewed as instances of the eBWT applied to specific decompositions of a word. A
central property of the BWT, essential for its compressibility, is the number
of maximal ranges of equal letters, named runs. In this article, we explore how
different decompositions of a word impact the number of runs in the resulting
eBWT. First, we show that the number of decompositions of a word is
exponential, even under minimal constraints on the size of the subsets in the
decomposition. Second, we present an infinite family of words for which the
ratio of the number of runs between the worst and best decompositions is
unbounded, under the same minimal constraints. These results illustrate the
potential cost of decomposition choices in eBWT-based compression and underline
the challenges in optimizing run-length encoding in generalized BWT frameworks.

### Data Structures and Algorithms

### 1. [Faster MPC Algorithms for Approximate Allocation in Uniformly Sparse Graphs](http://arxiv.org/pdf/2506.04524v1)

Authors: Jakub Łącki, Slobodan Mitrović, Srikkanth Ramachandran, Wen-Horng Sheu

We study the allocation problem in the Massively Parallel Computation (MPC)
model. This problem is a special case of $b$-matching, in which the input is a
bipartite graph with capacities greater than $1$ in only one part of the
bipartition. We give a $(1+\epsilon)$ approximate algorithm for the problem,
which runs in $\tilde{O}(\sqrt{\log \lambda})$ MPC rounds, using sublinear
space per machine and $\tilde{O}(\lambda n)$ total space, where $\lambda$ is
the arboricity of the input graph. Our result is obtained by providing a new
analysis of a LOCAL algorithm by Agrawal, Zadimoghaddam, and Mirrokni [ICML
2018], which improves its round complexity from $O(\log n)$ to $O(\log
\lambda)$. Prior to our work, no $o(\log n)$ round algorithm for
constant-approximate allocation was known in either LOCAL or sublinear space
MPC models for graphs with low arboricity.

### 2. [Online matching on stochastic block model](http://arxiv.org/pdf/2506.04921v1)

Authors: Maria Cherifa, Clément Calauzènes, Vianney Perchet

While online bipartite matching has gained significant attention in recent
years, existing analyses in stochastic settings fail to capture the performance
of algorithms on heterogeneous graphs, such as those incorporating inter-group
affinities or other social network structures. In this work, we address this
gap by studying online bipartite matching within the stochastic block model
(SBM). A fixed set of offline nodes is matched to a stream of online arrivals,
with connections governed probabilistically by latent class memberships. We
analyze two natural algorithms: a $\tt{Myopic}$ policy that greedily matches
each arrival to the most compatible class, and the $\tt{Balance}$ algorithm,
which accounts for both compatibility and remaining capacity. For the
$\tt{Myopic}$ algorithm, we prove that the size of the matching converges, with
high probability, to the solution of an ordinary differential equation (ODE),
for which we provide a tractable approximation along with explicit error
bounds. For the $\tt{Balance}$ algorithm, we demonstrate convergence of the
matching size to a differential inclusion and derive an explicit limiting
solution. Lastly, we explore the impact of estimating the connection
probabilities between classes online, which introduces an
exploration-exploitation trade-off.

### 3. [Resilient Pattern Mining](http://arxiv.org/pdf/2506.04935v1)

Authors: Pengxin Bian, Panagiotis Charalampopoulos, Lorraine A. K. Ayad, Manal Mohamed, Solon P. Pissis, Grigorios Loukides

Frequent pattern mining is a flagship problem in data mining. In its most
basic form, it asks for the set of substrings of a given string $S$ of length
$n$ that occur at least $\tau$ times in $S$, for some integer $\tau\in[1,n]$.
We introduce a resilient version of this classic problem, which we term the
$(\tau, k)$-Resilient Pattern Mining (RPM) problem. Given a string $S$ of
length $n$ and two integers $\tau, k\in[1,n]$, RPM asks for the set of
substrings of $S$ that occur at least $\tau$ times in $S$, even when the
letters at any $k$ positions of $S$ are substituted by other letters. Unlike
frequent substrings, resilient ones account for the fact that changes to string
$S$ are often expensive to handle or are unknown.
  We propose an exact $\mathcal{O}(n\log n)$-time and $\mathcal{O}(n)$-space
algorithm for RPM, which employs advanced data structures and combinatorial
insights. We then present experiments on real large-scale datasets from
different domains demonstrating that: (I) The notion of resilient substrings is
useful in analyzing genomic data and is more powerful than that of frequent
substrings, in scenarios where resilience is required, such as in the case of
versioned datasets; (II) Our algorithm is several orders of magnitude faster
and more space-efficient than a baseline algorithm that is based on dynamic
programming; and (III) Clustering based on resilient substrings is effective.

### 4. [Compressing Hypergraphs using Suffix Sorting](http://arxiv.org/pdf/2506.05023v1)

Authors: Enno Adler, Stefan Böttcher, Rita Hartel

Hypergraphs model complex, non-binary relationships like co-authorships,
social group memberships, and recommendations. Like traditional graphs,
hypergraphs can grow large, posing challenges for storage, transmission, and
query performance. We propose HyperCSA, a novel compression method for
hypergraphs that maintains support for standard queries over the succinct
representation. HyperCSA achieves compression ratios of 26% to 79% of the
original file size on real-world hypergraphs - outperforming existing methods
on all large hypergraphs in our experiments. Additionally, HyperCSA scales to
larger datasets than existing approaches. Furthermore, for common real-world
hypergraphs, HyperCSA evaluates neighbor queries 6 to 40 times faster than both
standard data structures and other hypergraph compression approaches.

### 5. [Identity Testing for Circuits with Exponentiation Gates](http://arxiv.org/pdf/2506.04529v1)

Authors: Jiatu Li, Mengdi Wu

Motivated by practical applications in the design of optimization compilers
for neural networks, we initiated the study of identity testing problems for
arithmetic circuits augmented with \emph{exponentiation gates} that compute the
real function $x\mapsto e^x$. These circuits compute real functions of form
$P(\vec x)/P'(\vec x)$, where both $P(\vec x)$ and $P'(\vec x)$ are exponential
polynomials
  \[
  \sum_{i=1}^k f_i(\vec x)\cdot \exp\left(\frac{g_i(\vec x)}{h_i(\vec
x)}\right),
  \]
  for polynomials $f_i(\vec x),g_i(\vec x)$, and $h_i(\vec x)$.
  We formalize a black-box query model over finite fields for this class of
circuits, which is mathematical simple and reflects constraints faced by
real-world neural network compilers. We proved that a simple and efficient
randomized identity testing algorithm achieves perfect completeness and
non-trivial soundness. Concurrent with our work, the algorithm has been
implemented in the optimization compiler Mirage by Wu et al.~(OSDI 2025),
demonstrating promising empirical performance in both efficiency and soundness
error. Finally, we propose a number-theoretic conjecture under which our
algorithm is sound with high probability.

### 6. [Improved Byzantine Agreement under an Adaptive Adversary](http://arxiv.org/pdf/2506.04919v1)

Authors: Fabien Dufoulon, Gopal Pandurangan

Byzantine agreement is a fundamental problem in fault-tolerant distributed
computing that has been studied intensively for the last four decades. Much of
the research has focused on a static Byzantine adversary, where the adversary
is constrained to choose the Byzantine nodes in advance of the protocol's
execution. This work focuses on the harder case of an adaptive Byzantine
adversary that can choose the Byzantine nodes \emph{adaptively} based on the
protocol's execution. While efficient $O(\log n)$-round protocols ($n$ is the
total number of nodes) are known for the static adversary (Goldwasser, Pavlov,
and Vaikuntanathan, FOCS 2006) tolerating up to $t < n/(3+\epsilon)$ Byzantine
nodes, $\Omega(t/\sqrt{n \log n})$ rounds is a well-known lower bound for
adaptive adversary [Bar-Joseph and Ben-Or, PODC 1998]. The best-known protocol
for adaptive adversary runs in $O(t/\log n)$ rounds [Chor and Coan, IEEE Trans.
Soft. Engg., 1985].
  This work presents a synchronous randomized Byzantine agreement protocol
under an adaptive adversary that improves over previous results. Our protocol
works under the powerful \emph{adaptive rushing adversary in the full
information model}. That is, we assume that the Byzantine nodes can behave
arbitrarily and maliciously, have knowledge about the entire state of the
network at every round, including random choices made by all the nodes up to
and including the current round, have unlimited computational power, and may
collude among themselves. Furthermore, the adversary can \emph{adaptively}
corrupt up to $t < n/3$ nodes based on the protocol's execution. We present a
simple randomized Byzantine agreement protocol that runs in $O(\min\{t^2\log
n/n, t/\log n\})$ rounds that improves over the long-standing bound of
$O(t/\log n)$ rounds due to Chor and Coan [IEEE Trans. Soft. Engg., 1985].

### 7. [The Peculiarities of Extending Queue Layouts](http://arxiv.org/pdf/2506.05156v1)

Authors: Thomas Depian, Simon D. Fink, Robert Ganian, Martin Nöllenburg

We consider the problem of computing $\ell$-page queue layouts, which are
linear arrangements of vertices accompanied with an assignment of the edges to
pages from one to $\ell$ that avoid the nesting of edges on any of the pages.
Inspired by previous work in the extension of stack layouts, here we consider
the setting of extending a partial $\ell$-page queue layout into a complete one
and primarily analyze the problem through the refined lens of parameterized
complexity. We obtain novel algorithms and lower bounds which provide a
detailed picture of the problem's complexity under various measures of
incompleteness, and identify surprising distinctions between queue and stack
layouts in the extension setting.

### 8. [On Minimizers of Minimum Density](http://arxiv.org/pdf/2506.05277v1)

Authors: Arseny Shur

Minimizers are sampling schemes with numerous applications in computational
biology. Assuming a fixed alphabet of size $\sigma$, a minimizer is defined by
two integers $k,w\ge2$ and a linear order $\rho$ on strings of length $k$ (also
called $k$-mers). A string is processed by a sliding window algorithm that
chooses, in each window of length $w+k-1$, its minimal $k$-mer with respect to
$\rho$. A key characteristic of the minimizer is its density, which is the
expected frequency of chosen $k$-mers among all $k$-mers in a random infinite
$\sigma$-ary string. Minimizers of smaller density are preferred as they
produce smaller samples with the same guarantee: each window is represented by
a $k$-mer.
  The problem of finding a minimizer of minimum density for given input
parameters $(\sigma,k,w)$ has a huge search space of $(\sigma^k)!$ and is
representable by an ILP of size $\tilde\Theta(\sigma^{k+w})$, which has
worst-case solution time that is doubly-exponential in $(k+w)$ under standard
complexity assumptions. We solve this problem in $w\cdot 2^{\sigma^k+O(k)}$
time and provide several additional tricks reducing the practical runtime and
search space. As a by-product, we describe an algorithm computing the average
density of a minimizer within the same time bound. Then we propose a novel
method of studying minimizers via regular languages and show how to find, via
the eigenvalue/eigenvector analysis over finite automata, minimizers with the
minimal density in the asymptotic case $w\to\infty$. Implementing our
algorithms, we compute the minimum density minimizers for
$(\sigma,k)\in\{(2,2),(2,3),(2,4),(2,5),(4,2)\}$ and \textbf{all} $w\ge 2$. The
obtained densities are compared against the average density and the theoretical
lower bounds, including the new bound presented in this paper.

### 9. [Decomposing Words for Enhanced Compression: Exploring the Number of Runs in the Extended Burrows-Wheeler Transform](http://arxiv.org/pdf/2506.04926v1)

Authors: Florian Ingels, Anaïs Denis, Bastien Cazaux

The Burrows-Wheeler Transform (BWT) is a fundamental component in many data
structures for text indexing and compression, widely used in areas such as
bioinformatics and information retrieval. The extended BWT (eBWT) generalizes
the classical BWT to multisets of strings, providing a flexible framework that
captures many BWT-like constructions. Several known variants of the BWT can be
viewed as instances of the eBWT applied to specific decompositions of a word. A
central property of the BWT, essential for its compressibility, is the number
of maximal ranges of equal letters, named runs. In this article, we explore how
different decompositions of a word impact the number of runs in the resulting
eBWT. First, we show that the number of decompositions of a word is
exponential, even under minimal constraints on the size of the subsets in the
decomposition. Second, we present an infinite family of words for which the
ratio of the number of runs between the worst and best decompositions is
unbounded, under the same minimal constraints. These results illustrate the
potential cost of decomposition choices in eBWT-based compression and underline
the challenges in optimizing run-length encoding in generalized BWT frameworks.

### 10. [Memory Hierarchy Design for Caching Middleware in the Age of NVM](http://arxiv.org/pdf/2506.05071v1)

Authors: Shahram Ghandeharizadeh, Sandy Irani, Jenny Lam

Advances in storage technology have introduced Non-Volatile Memory, NVM, as a
new storage medium. NVM, along with Dynamic Random Access Memory (DRAM), Solid
State Disk (SSD), and Disk present a system designer with a wide array of
options in designing caching middleware. Moreover, design decisions to
replicate a data item in more than one level of a caching memory hierarchy may
enhance the overall system performance with a faster recovery time in the event
of a memory failure. Given a fixed budget, the key configuration questions are:
Which storage media should constitute the memory hierarchy? What is the storage
capacity of each hierarchy? Should data be replicated or partitioned across the
different levels of the hierarchy? We model these cache configuration questions
as an instance of the Multiple Choice Knapsack Problem (MCKP). This model is
guided by the specification of each type of memory along with an application's
database characteristics and its workload. Although MCKP is NP-complete, its
linear programming relaxation is efficiently solvable and can be used to
closely approximate the optimal solution. We use the resulting simple algorithm
to evaluate design tradeoffs in the context of a memory hierarchy for a
Key-Value Store (e.g., memcached) as well as a host-side cache (e.g.,
Flashcache). The results show selective replication is appropriate with certain
failure rates and workload characteristics. With a slim failure rate and
frequent data updates, tiering of data across the different storage media that
constitute the cache is superior to replication.

### Emerging Technologies

### 1. [TQml Simulator: Optimized Simulation of Quantum Machine Learning](http://arxiv.org/pdf/2506.04891v1)

Authors: Viacheslav Kuzmin, Basil Kyriacou, Mateusz Papierz, Mo Kordzanganeh, Alexey Melnikov

Hardware-efficient circuits employed in Quantum Machine Learning are
typically composed of alternating layers of uniformly applied gates. High-speed
numerical simulators for such circuits are crucial for advancing research in
this field. In this work, we numerically benchmark universal and gate-specific
techniques for simulating the action of layers of gates on quantum state
vectors, aiming to accelerate the overall simulation of Quantum Machine
Learning algorithms. Our analysis shows that the optimal simulation method for
a given layer of gates depends on the number of qubits involved, and that a
tailored combination of techniques can yield substantial performance gains in
the forward and backward passes for a given circuit. Building on these
insights, we developed a numerical simulator, named TQml Simulator, that
employs the most efficient simulation method for each layer in a given circuit.
We evaluated TQml Simulator on circuits constructed from standard gate sets,
such as rotations and CNOTs, as well as on native gates from IonQ and IBM
quantum processing units. In most cases, our simulator outperforms equivalent
Pennylane's default.qubit simulator by approximately 2- to 100-fold, depending
on the circuit, the number of qubits, the batch size of the input data, and the
hardware used.

### 2. [From Screen to Space: Evaluating Siemens' Cinematic Reality](http://arxiv.org/pdf/2506.04972v1)

Authors: Gijs Luijten, Lisle Faray de Paiva, Sebastian Krueger, Alexander Brost, Laura Mazilescu, Ana Sofia Ferreira Santos, Peter Hoyer, Jens Kleesiek, Sophia Marie-Therese Schmitz, Ulf Peter Neumann, Jan Egger

As one of the first research teams with full access to Siemens' Cinematic
Reality, we evaluate its usability and clinical potential for cinematic volume
rendering on the Apple Vision Pro. We visualized venous-phase liver computed
tomography and magnetic resonance cholangiopancreatography scans from the CHAOS
and MRCP\_DLRecon datasets. Fourteen medical experts assessed usability and
anticipated clinical integration potential using the System Usability Scale,
ISONORM 9242-110-S questionnaire, and an open-ended survey. Their feedback
identified feasibility, key usability strengths, and required features to
catalyze the adaptation in real-world clinical workflows. The findings provide
insights into the potential of immersive cinematic rendering in medical
imaging.

### 3. [Perturbative Gradient Training: A novel training paradigm for bridging the gap between deep neural networks and physical reservoir computing](http://arxiv.org/pdf/2506.04523v1)

Authors: Cliff B. Abbott, Mark Elo, Dmytro A. Bozhko

We introduce Perturbative Gradient Training (PGT), a novel training paradigm
that overcomes a critical limitation of physical reservoir computing: the
inability to perform backpropagation due to the black-box nature of physical
reservoirs. Drawing inspiration from perturbation theory in physics, PGT uses
random perturbations in the network's parameter space to approximate gradient
updates using only forward passes. We demonstrate the feasibility of this
approach on both simulated neural network architectures, including a dense
network and a transformer model with a reservoir layer, and on experimental
hardware using a magnonic auto-oscillation ring as the physical reservoir. Our
results show that PGT can achieve performance comparable to that of standard
backpropagation methods in cases where backpropagation is impractical or
impossible. PGT represents a promising step toward integrating physical
reservoirs into deeper neural network architectures and achieving significant
energy efficiency gains in AI training.

### 4. [Olfactory Inertial Odometry: Sensor Calibration and Drift Compensation](http://arxiv.org/pdf/2506.04539v1)

Authors: Kordel K. France, Ovidiu Daescu, Anirban Paul, Shalini Prasad

Visual inertial odometry (VIO) is a process for fusing visual and kinematic
data to understand a machine's state in a navigation task. Olfactory inertial
odometry (OIO) is an analog to VIO that fuses signals from gas sensors with
inertial data to help a robot navigate by scent. Gas dynamics and environmental
factors introduce disturbances into olfactory navigation tasks that can make
OIO difficult to facilitate. With our work here, we define a process for
calibrating a robot for OIO that generalizes to several olfaction sensor types.
Our focus is specifically on calibrating OIO for centimeter-level accuracy in
localizing an odor source on a slow-moving robot platform to demonstrate use
cases in robotic surgery and touchless security screening. We demonstrate our
process for OIO calibration on a real robotic arm and show how this calibration
improves performance over a cold-start olfactory navigation task.

### Formal Languages and Automata Theory

### 1. [Quantitative Language Automata](http://arxiv.org/pdf/2506.05158v1)

Authors: Thomas A. Henzinger, Pavol Kebis, Nicolas Mazzocchi, N. Ege Saraç

A quantitative word automaton (QWA) defines a function from infinite words to
values. For example, every infinite run of a limit-average QWA A obtains a mean
payoff, and every word w is assigned the maximal mean payoff obtained by
nondeterministic runs of A over w. We introduce quantitative language automata
(QLAs) that define functions from language generators (i.e., implementations)
to values, where a language generator can be nonprobabilistic, defining a set
of infinite words, or probabilistic, defining a probability measure over
infinite words. A QLA consists of a QWA and an aggregator function. For
example, given a QWA A, the infimum aggregator maps each language L to the
greatest lower bound assigned by A to any word in L. For boolean value sets,
QWAs define boolean properties of traces, and QLAs define boolean properties of
sets of traces, i.e., hyperproperties. For more general value sets, QLAs serve
as a specification language for a generalization of hyperproperties, called
quantitative hyperproperties. A nonprobabilistic (resp. probabilistic)
quantitative hyperproperty assigns a value to each set (resp. distribution) G
of traces, e.g., the minimal (resp. expected) average response time exhibited
by the traces in G. We give several examples of quantitative hyperproperties
and investigate three paradigmatic problems for QLAs: evaluation, nonemptiness,
and universality. In the evaluation problem, given a QLA AA and an
implementation G, we ask for the value that AA assigns to G. In the
nonemptiness (resp. universality) problem, given a QLA AA and a value k, we ask
whether AA assigns at least k to some (resp. every) language. We provide a
comprehensive picture of decidability for these problems for QLAs with common
aggregators as well as their restrictions to omega-regular languages and trace
distributions generated by finite-state Markov chains.

### 2. [Backward Responsibility in Transition Systems Beyond Safety](http://arxiv.org/pdf/2506.05192v1)

Authors: Christel Baier, Rio Klatt, Sascha Klüppelholz, Johannes Lehmann

As the complexity of software systems rises, methods for explaining their
behaviour are becoming ever-more important. When a system fails, it is critical
to determine which of its components are responsible for this failure. Within
the verification community, one approach uses graph games and the Shapley value
to ascribe a responsibility value to every state of a transition system. As
this is done with respect to a specific failure, it is called backward
responsibility.
  This paper provides tight complexity bounds for backward responsibility for
reachability, B\"uchi and parity objectives. For B\"uchi objectives, a
polynomial algorithm is given to determine the set of responsible states. To
analyse systems that are too large for standard methods, the paper presents a
novel refinement algorithm that iteratively computes responsibility and
demonstrates its utility with a prototypical implementation.

### 3. [On Minimizers of Minimum Density](http://arxiv.org/pdf/2506.05277v1)

Authors: Arseny Shur

Minimizers are sampling schemes with numerous applications in computational
biology. Assuming a fixed alphabet of size $\sigma$, a minimizer is defined by
two integers $k,w\ge2$ and a linear order $\rho$ on strings of length $k$ (also
called $k$-mers). A string is processed by a sliding window algorithm that
chooses, in each window of length $w+k-1$, its minimal $k$-mer with respect to
$\rho$. A key characteristic of the minimizer is its density, which is the
expected frequency of chosen $k$-mers among all $k$-mers in a random infinite
$\sigma$-ary string. Minimizers of smaller density are preferred as they
produce smaller samples with the same guarantee: each window is represented by
a $k$-mer.
  The problem of finding a minimizer of minimum density for given input
parameters $(\sigma,k,w)$ has a huge search space of $(\sigma^k)!$ and is
representable by an ILP of size $\tilde\Theta(\sigma^{k+w})$, which has
worst-case solution time that is doubly-exponential in $(k+w)$ under standard
complexity assumptions. We solve this problem in $w\cdot 2^{\sigma^k+O(k)}$
time and provide several additional tricks reducing the practical runtime and
search space. As a by-product, we describe an algorithm computing the average
density of a minimizer within the same time bound. Then we propose a novel
method of studying minimizers via regular languages and show how to find, via
the eigenvalue/eigenvector analysis over finite automata, minimizers with the
minimal density in the asymptotic case $w\to\infty$. Implementing our
algorithms, we compute the minimum density minimizers for
$(\sigma,k)\in\{(2,2),(2,3),(2,4),(2,5),(4,2)\}$ and \textbf{all} $w\ge 2$. The
obtained densities are compared against the average density and the theoretical
lower bounds, including the new bound presented in this paper.

### 4. [Decomposing Words for Enhanced Compression: Exploring the Number of Runs in the Extended Burrows-Wheeler Transform](http://arxiv.org/pdf/2506.04926v1)

Authors: Florian Ingels, Anaïs Denis, Bastien Cazaux

The Burrows-Wheeler Transform (BWT) is a fundamental component in many data
structures for text indexing and compression, widely used in areas such as
bioinformatics and information retrieval. The extended BWT (eBWT) generalizes
the classical BWT to multisets of strings, providing a flexible framework that
captures many BWT-like constructions. Several known variants of the BWT can be
viewed as instances of the eBWT applied to specific decompositions of a word. A
central property of the BWT, essential for its compressibility, is the number
of maximal ranges of equal letters, named runs. In this article, we explore how
different decompositions of a word impact the number of runs in the resulting
eBWT. First, we show that the number of decompositions of a word is
exponential, even under minimal constraints on the size of the subsets in the
decomposition. Second, we present an infinite family of words for which the
ratio of the number of runs between the worst and best decompositions is
unbounded, under the same minimal constraints. These results illustrate the
potential cost of decomposition choices in eBWT-based compression and underline
the challenges in optimizing run-length encoding in generalized BWT frameworks.

### Graphics

### 1. [Towards the target and not beyond: 2d vs 3d visual aids in mr-based neurosurgical simulation](http://arxiv.org/pdf/2506.05164v1)

Authors: Pasquale Cascarano, Andrea Loretti, Matteo Martinoni, Luca Zanuttini, Alessio Di Pasquale, Gustavo Marfia

Neurosurgery increasingly uses Mixed Reality (MR) technologies for
intraoperative assistance. The greatest challenge in this area is mentally
reconstructing complex 3D anatomical structures from 2D slices with millimetric
precision, which is required in procedures like External Ventricular Drain
(EVD) placement. MR technologies have shown great potential in improving
surgical performance, however, their limited availability in clinical settings
underscores the need for training systems that foster skill retention in
unaided conditions. In this paper, we introduce NeuroMix, an MR-based simulator
for EVD placement. We conduct a study with 48 participants to assess the impact
of 2D and 3D visual aids on usability, cognitive load, technology acceptance,
and procedure precision and execution time. Three training modalities are
compared: one without visual aids, one with 2D aids only, and one combining
both 2D and 3D aids. The training phase takes place entirely on digital
objects, followed by a freehand EVD placement testing phase performed with a
physical catherer and a physical phantom without MR aids. We then compare the
participants performance with that of a control group that does not undergo
training. Our findings show that participants trained with both 2D and 3D aids
achieve a 44\% improvement in precision during unaided testing compared to the
control group, substantially higher than the improvement observed in the other
groups. All three training modalities receive high usability and technology
acceptance ratings, with significant equivalence across groups. The combination
of 2D and 3D visual aids does not significantly increase cognitive workload,
though it leads to longer operation times during freehand testing compared to
the control group.

### 2. [Uniform Sampling of Surfaces by Casting Rays](http://arxiv.org/pdf/2506.05268v1)

Authors: Selena Ling, Abhishek Madan, Nicholas Sharp, Alec Jacobson

Randomly sampling points on surfaces is an essential operation in geometry
processing. This sampling is computationally straightforward on explicit
meshes, but it is much more difficult on other shape representations, such as
widely-used implicit surfaces. This work studies a simple and general scheme
for sampling points on a surface, which is derived from a connection to the
intersections of random rays with the surface. Concretely, given a subroutine
to cast a ray against a surface and find all intersections, we can use that
subroutine to uniformly sample white noise points on the surface. This approach
is particularly effective in the context of implicit signed distance functions,
where sphere marching allows us to efficiently cast rays and sample points,
without needing to extract an intermediate mesh. We analyze the basic method to
show that it guarantees uniformity, and find experimentally that it is
significantly more efficient than alternative strategies on a variety of
representations. Furthermore, we show extensions to blue noise sampling and
stratified sampling, and applications to deform neural implicit surfaces as
well as moment estimation.

### 3. [Handle-based Mesh Deformation Guided By Vision Language Model](http://arxiv.org/pdf/2506.04562v1)

Authors: Xingpeng Sun, Shiyang Jia, Zherong Pan, Kui Wu, Aniket Bera

Mesh deformation is a fundamental tool in 3D content manipulation. Despite
extensive prior research, existing approaches often suffer from low output
quality, require significant manual tuning, or depend on data-intensive
training. To address these limitations, we introduce a training-free,
handle-based mesh deformation method. % Our core idea is to leverage a
Vision-Language Model (VLM) to interpret and manipulate a handle-based
interface through prompt engineering. We begin by applying cone singularity
detection to identify a sparse set of potential handles. The VLM is then
prompted to select both the deformable sub-parts of the mesh and the handles
that best align with user instructions. Subsequently, we query the desired
deformed positions of the selected handles in screen space. To reduce
uncertainty inherent in VLM predictions, we aggregate the results from multiple
camera views using a novel multi-view voting scheme. % Across a suite of
benchmarks, our method produces deformations that align more closely with user
intent, as measured by CLIP and GPTEval3D scores, while introducing low
distortion -- quantified via membrane energy. In summary, our approach is
training-free, highly automated, and consistently delivers high-quality mesh
deformations.

### 4. [VoxDet: Rethinking 3D Semantic Occupancy Prediction as Dense Object Detection](http://arxiv.org/pdf/2506.04623v1)

Authors: Wuyang Li, Zhu Yu, Alexandre Alahi

3D semantic occupancy prediction aims to reconstruct the 3D geometry and
semantics of the surrounding environment. With dense voxel labels, prior works
typically formulate it as a dense segmentation task, independently classifying
each voxel. However, this paradigm neglects critical instance-centric
discriminability, leading to instance-level incompleteness and adjacent
ambiguities. To address this, we highlight a free lunch of occupancy labels:
the voxel-level class label implicitly provides insight at the instance level,
which is overlooked by the community. Motivated by this observation, we first
introduce a training-free Voxel-to-Instance (VoxNT) trick: a simple yet
effective method that freely converts voxel-level class labels into
instance-level offset labels. Building on this, we further propose VoxDet, an
instance-centric framework that reformulates the voxel-level occupancy
prediction as dense object detection by decoupling it into two sub-tasks:
offset regression and semantic prediction. Specifically, based on the lifted 3D
volume, VoxDet first uses (a) Spatially-decoupled Voxel Encoder to generate
disentangled feature volumes for the two sub-tasks, which learn task-specific
spatial deformation in the densely projected tri-perceptive space. Then, we
deploy (b) Task-decoupled Dense Predictor to address this task via dense
detection. Here, we first regress a 4D offset field to estimate distances (6
directions) between voxels and object borders in the voxel space. The regressed
offsets are then used to guide the instance-level aggregation in the
classification branch, achieving instance-aware prediction. Experiments show
that VoxDet can be deployed on both camera and LiDAR input, jointly achieving
state-of-the-art results on both benchmarks. VoxDet is not only highly
efficient, but also achieves 63.0 IoU on the SemanticKITTI test set, ranking
1st on the online leaderboard.

### 5. [A Fast Unsupervised Scheme for Polygonal Approximation](http://arxiv.org/pdf/2506.04664v1)

Authors: Bimal Kumar Ray

This paper proposes a fast and unsupervised scheme for a polygonal
approximation of a closed digital curve. It is demonstrated that the
approximation scheme is faster than state-of-the-art approximation and is
competitive with the same in Rosin's measure and in its aesthetic aspect. The
scheme comprises of three phases: initial segmentation, iterative vertex
insertion, and iterative merging, followed by vertex adjustment. The initial
segmentation is used to detect sharp turnings - the vertices that seemingly
have high curvature. It is likely that some of important vertices with low
curvature might have been missed out at the first phase and so iterative vertex
insertion is used to add vertices in a region where the curvature changes
slowly but steadily. The initial phase may pick up some undesirable vertices
and so merging is used to eliminate the redundant vertices. Finally, vertex
adjustment is used to facilitate enhancement in the aesthetic look of the
approximation. The quality of the approximations is measured using Rosin's
measure. The robustness of the proposed scheme with respect to geometric
transformation is observed.

### 6. [Midplane based 3D single pass unbiased segment-to-segment contact interaction using penalty method](http://arxiv.org/pdf/2506.04841v1)

Authors: Indrajeet Sahu, Nik Petrinic

This work introduces a contact interaction methodology for an unbiased
treatment of contacting surfaces without assigning surfaces as master and
slave. The contact tractions between interacting discrete segments are
evaluated with respect to a midplane in a single pass, inherently maintaining
the equilibrium of tractions. These tractions are based on the penalisation of
true interpenetration between opposite surfaces, and the procedure of their
integral for discrete contacting segments is described in this paper. A
meticulous examination of the different possible geometric configurations of
interacting 3D segments is presented to develop visual understanding and better
traction evaluation accuracy. The accuracy and robustness of the proposed
method are validated against the analytical solutions of the contact patch
test, two-beam bending, Hertzian contact, and flat punch test, thus proving the
capability to reproduce contact between flat surfaces, curved surfaces, and
sharp corners in contact, respectively. The method passes the contact patch
test with the uniform transmission of contact pressure matching the accuracy
levels of finite elements. It converges towards the analytical solution with
mesh refinement and a suitably high penalty factor. The effectiveness of the
proposed algorithm also extends to self-contact problems and has been tested
for self-contact between flat and curved surfaces with inelastic material.
Dynamic problems of elastic and inelastic collisions between bars, as well as
oblique collisions of cylinders, are also presented. The ability of the
algorithm to resolve contacts between flat and curved surfaces for nonconformal
meshes with high accuracy demonstrates its versatility in general contact
problems.

### 7. [Beyond the Desktop: XR-Driven Segmentation with Meta Quest 3 and MX Ink](http://arxiv.org/pdf/2506.04858v1)

Authors: Lisle Faray de Paiva, Gijs Luijten, Ana Sofia Ferreira Santos, Moon Kim, Behrus Puladi, Jens Kleesiek, Jan Egger

Medical imaging segmentation is essential in clinical settings for diagnosing
diseases, planning surgeries, and other procedures. However, manual annotation
is a cumbersome and effortful task. To mitigate these aspects, this study
implements and evaluates the usability and clinical applicability of an
extended reality (XR)-based segmentation tool for anatomical CT scans, using
the Meta Quest 3 headset and Logitech MX Ink stylus. We develop an immersive
interface enabling real-time interaction with 2D and 3D medical imaging data in
a customizable workspace designed to mitigate workflow fragmentation and
cognitive demands inherent to conventional manual segmentation tools. The
platform combines stylus-driven annotation, mirroring traditional pen-on-paper
workflows, with instant 3D volumetric rendering. A user study with a public
craniofacial CT dataset demonstrated the tool's foundational viability,
achieving a System Usability Scale (SUS) score of 66, within the expected range
for medical applications. Participants highlighted the system's intuitive
controls (scoring 4.1/5 for self-descriptiveness on ISONORM metrics) and
spatial interaction design, with qualitative feedback highlighting strengths in
hybrid 2D/3D navigation and realistic stylus ergonomics. While users identified
opportunities to enhance task-specific precision and error management, the
platform's core workflow enabled dynamic slice adjustment, reducing cognitive
load compared to desktop tools. Results position the XR-stylus paradigm as a
promising foundation for immersive segmentation tools, with iterative
refinements targeting haptic feedback calibration and workflow personalization
to advance adoption in preoperative planning.

### 8. [From Screen to Space: Evaluating Siemens' Cinematic Reality](http://arxiv.org/pdf/2506.04972v1)

Authors: Gijs Luijten, Lisle Faray de Paiva, Sebastian Krueger, Alexander Brost, Laura Mazilescu, Ana Sofia Ferreira Santos, Peter Hoyer, Jens Kleesiek, Sophia Marie-Therese Schmitz, Ulf Peter Neumann, Jan Egger

As one of the first research teams with full access to Siemens' Cinematic
Reality, we evaluate its usability and clinical potential for cinematic volume
rendering on the Apple Vision Pro. We visualized venous-phase liver computed
tomography and magnetic resonance cholangiopancreatography scans from the CHAOS
and MRCP\_DLRecon datasets. Fourteen medical experts assessed usability and
anticipated clinical integration potential using the System Usability Scale,
ISONORM 9242-110-S questionnaire, and an open-ended survey. Their feedback
identified feasibility, key usability strengths, and required features to
catalyze the adaptation in real-world clinical workflows. The findings provide
insights into the potential of immersive cinematic rendering in medical
imaging.

### Computer Science and Game Theory

### 1. [An O(log log n)-approximate budget feasible mechanism for subadditive valuations](http://arxiv.org/pdf/2506.04665v1)

Authors: Rian Neogi, Kanstantsin Pashkovich, Chaitanya Swamy

In budget-feasible mechanism design, there is a set of items $U$, each owned
by a distinct seller. The seller of item $e$ incurs a private cost
$\overline{c}_e$ for supplying her item. A buyer wishes to procure a set of
items from the sellers of maximum value, where the value of a set $S\subseteq
U$ of items is given by a valuation function $v:2^U\to \mathbb{R}_+$. The buyer
has a budget of $B \in \mathbb{R}_+$ for the total payments made to the
sellers. We wish to design a mechanism that is truthful, that is, sellers are
incentivized to report their true costs, budget-feasible, that is, the sum of
the payments made to the sellers is at most the budget $B$, and that outputs a
set whose value is large compared to $\text{OPT}:=\max\{v(S):\overline{c}(S)\le
B,S\subseteq U\}$.
  Budget-feasible mechanism design has been extensively studied, with the
literature focussing on (classes of) subadditive valuation functions, and
various polytime, budget-feasible mechanisms, achieving constant-factor
approximation, have been devised for the special cases of additive, submodular,
and XOS valuations. However, for general subadditive valuations, the best-known
approximation factor achievable by a polytime budget-feasible mechanism (given
access to demand oracles) was only $O(\log n / \log \log n)$, where $n$ is the
number of items.
  We improve this state-of-the-art significantly by designing a budget-feasible
mechanism for subadditive valuations that \emph{achieves a
substantially-improved approximation factor of $O(\log\log n)$ and runs in
polynomial time, given access to demand oracles.}

### 2. [MVP-Shapley: Feature-based Modeling for Evaluating the Most Valuable Player in Basketball](http://arxiv.org/pdf/2506.04602v1)

Authors: Haifeng Sun, Yu Xiong, Runze Wu, Kai Wang, Lan Zhang, Changjie Fan, Shaojie Tang, Xiang-Yang Li

The burgeoning growth of the esports and multiplayer online gaming community
has highlighted the critical importance of evaluating the Most Valuable Player
(MVP). The establishment of an explainable and practical MVP evaluation method
is very challenging. In our study, we specifically focus on play-by-play data,
which records related events during the game, such as assists and points. We
aim to address the challenges by introducing a new MVP evaluation framework,
denoted as \oursys, which leverages Shapley values. This approach encompasses
feature processing, win-loss model training, Shapley value allocation, and MVP
ranking determination based on players' contributions. Additionally, we
optimize our algorithm to align with expert voting results from the perspective
of causality. Finally, we substantiated the efficacy of our method through
validation using the NBA dataset and the Dunk City Dynasty dataset and
implemented online deployment in the industry.

### 3. [Misère Greedy Nim and Misère Bounded Greedy Nim](http://arxiv.org/pdf/2506.04657v1)

Authors: Nanako Omiya, Ryo Yoshinaka, Ayumi Shinohara

In this paper, we analyze the mis\`ere versions of two impartial
combinatorial games: k-Bounded Greedy Nim and Greedy Nim. We present a complete
solution to both games by showing necessary and sufficient conditions for a
position to be P-positions.

### 4. [Cooperation and the Design of Public Goods](http://arxiv.org/pdf/2506.05251v1)

Authors: J. Carlos Martínez Mori, Alejandro Toriello

We consider the cooperative elements that arise in the design of public
goods, such as transportation policies and infrastructure. These involve a
variety of stakeholders: governments, businesses, advocates, and users. Their
eventual deployment depends on the decision maker's ability to garner
sufficient support from each of these groups; we formalize these strategic
requirements from the perspective of cooperative game theory. Specifically, we
introduce non-transferable utility, linear production (NTU LP) games, which
combine the game-theoretic tensions inherent in public decision-making with the
modeling flexibility of linear programming. We derive structural properties
regarding the non-emptiness, representability and complexity of the core, a
solution concept that models the viability of cooperation. In particular, we
provide fairly general sufficient conditions under which the core of an NTU LP
game is guaranteed to be non-empty, prove that determining membership in the
core is co-NP-complete, and develop a cutting plane algorithm to optimize
various social welfare objectives subject to core membership. Lastly, we apply
these results in a data-driven case study on service plan optimization for the
Chicago bus system. As our study illustrates, cooperation is necessary for the
successful deployment of transportation service plans and similar public goods,
but it may also have adverse or counterintuitive distributive implications.

### 5. [Equilibrium Computation in First-Price Auctions with Correlated Priors](http://arxiv.org/pdf/2506.05322v1)

Authors: Aris Filos-Ratsikas, Yiannis Giannakopoulos, Alexandros Hollender, Charalampos Kokkalis

We consider the computational complexity of computing Bayes-Nash equilibria
in first-price auctions, where the bidders' values for the item are drawn from
a general (possibly correlated) joint distribution. We show that when the
values and the bidding space are discrete, determining the existence of a pure
Bayes-Nash equilibrium is NP-hard. This is the first hardness result in the
literature of the problem that does not rely on assumptions of subjectivity of
the priors, or convoluted tie-breaking rules. We then present two main
approaches for achieving positive results, via bid sparsification and via bid
densification. The former is more combinatorial and is based on enumeration
techniques, whereas the latter makes use of the continuous theory of the
problem developed in the economics literature. Using these approaches, we
develop polynomial-time approximation algorithms for computing equilibria in
symmetric settings or settings with a fixed number of bidders, for different
(discrete or continuous) variants of the auction.

### 6. [User Altruism in Recommendation Systems](http://arxiv.org/pdf/2506.04525v1)

Authors: Ekaterina Fedorova, Madeline Kitch, Chara Podimata

Users of social media platforms based on recommendation systems (RecSys)
(e.g. TikTok, X, YouTube) strategically interact with platform content to
influence future recommendations. On some such platforms, users have been
documented to form large-scale grassroots movements encouraging others to
purposefully interact with algorithmically suppressed content in order to
"boost" its recommendation; we term this behavior user altruism. To capture
this behavior, we study a game between users and a RecSys, where users provide
the RecSys (potentially manipulated) preferences over the contents available to
them, and the RecSys -- limited by data and computation constraints -- creates
a low-rank approximation preference matrix, and ultimately provides each user
her (approximately) most-preferred item. We compare the users' social welfare
under truthful preference reporting and under a class of strategies capturing
user altruism. In our theoretical analysis, we provide sufficient conditions to
ensure strict increases in user social welfare under user altruism, and provide
an algorithm to find an effective altruistic strategy. Interestingly, we show
that for commonly assumed recommender utility functions, effectively altruistic
strategies also improve the utility of the RecSys! We show that our results are
robust to several model misspecifications, thus strengthening our conclusions.
Our theoretical analysis is complemented by empirical results of effective
altruistic strategies on the GoodReads dataset, and an online survey on how
real-world users behave altruistically in RecSys. Overall, our findings serve
as a proof-of-concept of the reasons why traditional RecSys may incentivize
users to form collectives and/or follow altruistic strategies when interacting
with them.

### 7. [Cautious Optimism: A Meta-Algorithm for Near-Constant Regret in General Games](http://arxiv.org/pdf/2506.05005v1)

Authors: Ashkan Soleymani, Georgios Piliouras, Gabriele Farina

Recent work [Soleymani et al., 2025] introduced a variant of Optimistic
Multiplicative Weights Updates (OMWU) that adaptively controls the learning
pace in a dynamic, non-monotone manner, achieving new state-of-the-art regret
minimization guarantees in general games. In this work, we demonstrate that
no-regret learning acceleration through adaptive pacing of the learners is not
an isolated phenomenon. We introduce \emph{Cautious Optimism}, a framework for
substantially faster regularized learning in general games. Cautious Optimism
takes as input any instance of Follow-the-Regularized-Leader (FTRL) and outputs
an accelerated no-regret learning algorithm by pacing the underlying FTRL with
minimal computational overhead. Importantly, we retain uncoupledness (learners
do not need to know other players' utilities). Cautious Optimistic FTRL
achieves near-optimal $O_T(\log T)$ regret in diverse self-play
(mixing-and-matching regularizers) while preserving the optimal $O(\sqrt{T})$
regret in adversarial scenarios. In contrast to prior works (e.g. Syrgkanis et
al. [2015], Daskalakis et al. [2021]), our analysis does not rely on monotonic
step-sizes, showcasing a novel route for fast learning in general games.

### 8. [Conservative classifiers do consistently well with improving agents: characterizing statistical and online learning](http://arxiv.org/pdf/2506.05252v1)

Authors: Dravyansh Sharma, Alec Sun

Machine learning is now ubiquitous in societal decision-making, for example
in evaluating job candidates or loan applications, and it is increasingly
important to take into account how classified agents will react to the learning
algorithms. The majority of recent literature on strategic classification has
focused on reducing and countering deceptive behaviors by the classified
agents, but recent work of Attias et al. identifies surprising properties of
learnability when the agents genuinely improve in order to attain the desirable
classification, such as smaller generalization error than standard
PAC-learning. In this paper we characterize so-called learnability with
improvements across multiple new axes. We introduce an asymmetric variant of
minimally consistent concept classes and use it to provide an exact
characterization of proper learning with improvements in the realizable
setting. While prior work studies learnability only under general, arbitrary
agent improvement regions, we give positive results for more natural Euclidean
ball improvement sets. In particular, we characterize improper learning under a
mild generative assumption on the data distribution. We further show how to
learn in more challenging settings, achieving lower generalization error under
well-studied bounded noise models and obtaining mistake bounds in realizable
and agnostic online learning. We resolve open questions posed by Attias et al.
for both proper and improper learning.

### Human-Computer Interaction

### 1. [Seamless and Efficient Interactions within a Mixed-Dimensional Information Space](http://arxiv.org/pdf/2506.04545v1)

Authors: Chen Chen

Mediated by today's visual displays, information space allows users to
discover, access and interact with a wide range of digital and physical
information. The information presented in this space may be digital, physical
or a blend of both, and appear across different dimensions - such as texts,
images, 3D content and physical objects embedded within real-world environment.
Navigating within the information space often involves interacting with
mixed-dimensional entities, visually represented in both 2D and 3D. At times,
interactions also involve transitioning among entities represented in different
dimensions. We introduce the concept of mixed-dimensional information space,
encompassing entities represented in both 2D and 3D. Interactions within the
mixed-dimensional information space should be seamless and efficient: users
should be able to focus on their primary tasks without being distracted by
interactions with or transitions between entities. While incorporating 3D
representations into the mixed-dimensional information space offers intuitive
and immersive ways to interact with complex information, it is important to
address potential seams and inefficiencies that arise while interacting with
both 2D and 3D entities. This dissertation introduces new interactive
techniques and systems to realize seamless and efficient interactions within
the mixed-dimensional information space. This dissertation introduces three
interactive systems: MemoVis which aims to use emergent generative AI to help
users create reference images for 3D design feedback; PaperToPlace which
demonstrates how paper-based instruction documents can be transformed and
spatialized into a context-aware MR experience; and VRContour which explores
how contour delineation workflow can be brought into VR.

### 2. [Multi-Tool Analysis of User Interface & Accessibility in Deployed Web-Based Chatbots](http://arxiv.org/pdf/2506.04659v1)

Authors: Mukesh Rajmohan, Smit Desai, Sanchari Das

In this work, we present a multi-tool evaluation of 106 deployed web-based
chatbots, across domains like healthcare, education and customer service,
comprising both standalone applications and embedded widgets using automated
tools (Google Lighthouse, PageSpeed Insights, SiteImprove Accessibility
Checker) and manual audits (Microsoft Accessibility Insights). Our analysis
reveals that over 80% of chatbots exhibit at least one critical accessibility
issue, and 45% suffer from missing semantic structures or ARIA role misuse.
Furthermore, we found that accessibility scores correlate strongly across tools
(e.g., Lighthouse vs PageSpeed Insights, r = 0.861), but performance scores do
not (r = 0.436), underscoring the value of a multi-tool approach. We offer a
replicable evaluation insights and actionable recommendations to support the
development of user-friendly conversational interfaces.

### 3. [Neuronal avalanches as a predictive biomarker of BCI performance: towards a tool to guide tailored training program](http://arxiv.org/pdf/2506.04745v1)

Authors: Camilla Mannino, Pierpaolo Sorrentino, Mario Chavez, Marie-Costance Corsi

Brain-Computer Interfaces (BCIs) based on motor imagery (MI) hold promise for
restoring control in individuals with motor impairments. However, up to 30% of
users remain unable to effectively use BCIs-a phenomenon termed ''BCI
inefficiency.'' This study addresses a major limitation in current BCI training
protocols: the use of fixed-length training paradigms that ignore individual
learning variability. We propose a novel approach that leverages neuronal
avalanches-spatiotemporal cascades of brain activity-as biomarkers to
characterize and predict user-specific learning mechanism. Using
electroencephalography (EEG) data collected across four MI-BCI training
sessions in 20 healthy participants, we extracted two features: avalanche
length and activations. These features revealed significant training and
taskcondition effects, particularly in later sessions. Crucially, changes in
these features across sessions ($\Delta$avalanche length and
$\Delta$activations) correlated significantly with BCI performance and enabled
prediction of future BCI success via longitudinal Support Vector Regression and
Classification models. Predictive accuracy reached up to 91%, with notable
improvements after spatial filtering based on selected regions of interest.
These findings demonstrate the utility of neuronal avalanche dynamics as robust
biomarkers for BCI training, supporting the development of personalized
protocols aimed at mitigating BCI illiteracy.

### 4. [Adapting Online Customer Reviews for Blind Users: A Case Study of Restaurant Reviews](http://arxiv.org/pdf/2506.04865v1)

Authors: Mohan Sunkara, Akshay Kolgar Nayak, Sandeep Kalari, Yash Prakash, Sampath Jayarathna, Hae-Na Lee, Vikas Ashok

Online reviews have become an integral aspect of consumer decision-making on
e-commerce websites, especially in the restaurant industry. Unlike sighted
users who can visually skim through the reviews, perusing reviews remains
challenging for blind users, who rely on screen reader assistive technology
that supports predominantly one-dimensional narration of content via keyboard
shortcuts. In an interview study, we uncovered numerous pain points of blind
screen reader users with online restaurant reviews, notably, the listening
fatigue and frustration after going through only the first few reviews. To
address these issues, we developed QuickQue assistive tool that performs
aspect-focused sentiment-driven summarization to reorganize the information in
the reviews into an alternative, thematically-organized presentation that is
conveniently perusable with a screen reader. At its core, QuickQue utilizes a
large language model to perform aspect-based joint classification for grouping
reviews, followed by focused summarizations within the groups to generate
concise representations of reviewers' opinions, which are then presented to the
screen reader users via an accessible interface. Evaluation of QuickQue in a
user study with 10 participants showed significant improvements in overall
usability and task workload compared to the status quo screen reader.

### 5. [Towards Effective Multidisciplinary Health and HCI Teams based on AI Framework](http://arxiv.org/pdf/2506.05226v1)

Authors: Mohammed Almutairi, Diego Gómez-Zará

As a Ph.D. student with a diverse background in both public and private
sectors, I have encountered numerous challenges in cross-disciplinary and
multi-stakeholder team projects. My research on developing team compositions
that involve multidisciplinary members from fields including education,
academia, and health. Along with my advisor, we are focused on exploring how
HCI can help individuals assemble more effective teams. This effort involves
developing socio-technical systems that guide and inform individuals of the
potential teams that they can assemble. We employ state-of-the-art algorithms
that prioritize inclusion among team members from diverse areas of expertise
and familiarity between the team members. Our goal for attending this workshop
is to engage in meaningful dialogues with scholars and researchers, leveraging
these interactions to refine our approach to building an AI-driven team
composition system to foster effective, interdisciplinary collaboration in
health-focused HCI research.

### 6. [PulseRide: A Robotic Wheelchair for Personalized Exertion Control with Human-in-the-Loop Reinforcement Learning](http://arxiv.org/pdf/2506.05056v1)

Authors: Azizul Zahid, Bibek Poudel, Danny Scott, Jason Scott, Scott Crouter, Weizi Li, Sai Swaminathan

Maintaining an active lifestyle is vital for quality of life, yet challenging
for wheelchair users. For instance, powered wheelchairs face increasing risks
of obesity and deconditioning due to inactivity. Conversely, manual wheelchair
users, who propel the wheelchair by pushing the wheelchair's handrims, often
face upper extremity injuries from repetitive motions. These challenges
underscore the need for a mobility system that promotes activity while
minimizing injury risk. Maintaining optimal exertion during wheelchair use
enhances health benefits and engagement, yet the variations in individual
physiological responses complicate exertion optimization. To address this, we
introduce PulseRide, a novel wheelchair system that provides personalized
assistance based on each user's physiological responses, helping them maintain
their physical exertion goals. Unlike conventional assistive systems focused on
obstacle avoidance and navigation, PulseRide integrates real-time physiological
data-such as heart rate and ECG-with wheelchair speed to deliver adaptive
assistance. Using a human-in-the-loop reinforcement learning approach with Deep
Q-Network algorithm (DQN), the system adjusts push assistance to keep users
within a moderate activity range without under- or over-exertion. We conducted
preliminary tests with 10 users on various terrains, including carpet and
slate, to assess PulseRide's effectiveness. Our findings show that, for
individual users, PulseRide maintains heart rates within the moderate activity
zone as much as 71.7 percent longer than manual wheelchairs. Among all users,
we observed an average reduction in muscle contractions of 41.86 percent,
delaying fatigue onset and enhancing overall comfort and engagement. These
results indicate that PulseRide offers a healthier, adaptive mobility solution,
bridging the gap between passive and physically taxing mobility options.

### 7. [User Altruism in Recommendation Systems](http://arxiv.org/pdf/2506.04525v1)

Authors: Ekaterina Fedorova, Madeline Kitch, Chara Podimata

Users of social media platforms based on recommendation systems (RecSys)
(e.g. TikTok, X, YouTube) strategically interact with platform content to
influence future recommendations. On some such platforms, users have been
documented to form large-scale grassroots movements encouraging others to
purposefully interact with algorithmically suppressed content in order to
"boost" its recommendation; we term this behavior user altruism. To capture
this behavior, we study a game between users and a RecSys, where users provide
the RecSys (potentially manipulated) preferences over the contents available to
them, and the RecSys -- limited by data and computation constraints -- creates
a low-rank approximation preference matrix, and ultimately provides each user
her (approximately) most-preferred item. We compare the users' social welfare
under truthful preference reporting and under a class of strategies capturing
user altruism. In our theoretical analysis, we provide sufficient conditions to
ensure strict increases in user social welfare under user altruism, and provide
an algorithm to find an effective altruistic strategy. Interestingly, we show
that for commonly assumed recommender utility functions, effectively altruistic
strategies also improve the utility of the RecSys! We show that our results are
robust to several model misspecifications, thus strengthening our conclusions.
Our theoretical analysis is complemented by empirical results of effective
altruistic strategies on the GoodReads dataset, and an online survey on how
real-world users behave altruistically in RecSys. Overall, our findings serve
as a proof-of-concept of the reasons why traditional RecSys may incentivize
users to form collectives and/or follow altruistic strategies when interacting
with them.

### 8. [Improving AI-generated music with user-guided training](http://arxiv.org/pdf/2506.04852v1)

Authors: Vishwa Mohan Singh, Sai Anirudh Aryasomayajula, Ahan Chatterjee, Beste Aydemir, Rifat Mehreen Amin

AI music generation has advanced rapidly, with models like diffusion and
autoregressive algorithms enabling high-fidelity outputs. These tools can alter
styles, mix instruments, or isolate them. Since sound can be visualized as
spectrograms, image-generation algorithms can be applied to generate novel
music. However, these algorithms are typically trained on fixed datasets, which
makes it challenging for them to interpret and respond to user input
accurately. This is especially problematic because music is highly subjective
and requires a level of personalization that image generation does not provide.
In this work, we propose a human-computation approach to gradually improve the
performance of these algorithms based on user interactions. The
human-computation element involves aggregating and selecting user ratings to
use as the loss function for fine-tuning the model. We employ a genetic
algorithm that incorporates user feedback to enhance the baseline performance
of a model initially trained on a fixed dataset. The effectiveness of this
approach is measured by the average increase in user ratings with each
iteration. In the pilot test, the first iteration showed an average rating
increase of 0.2 compared to the baseline. The second iteration further improved
upon this, achieving an additional increase of 0.39 over the first iteration.

### 9. [Beyond the Desktop: XR-Driven Segmentation with Meta Quest 3 and MX Ink](http://arxiv.org/pdf/2506.04858v1)

Authors: Lisle Faray de Paiva, Gijs Luijten, Ana Sofia Ferreira Santos, Moon Kim, Behrus Puladi, Jens Kleesiek, Jan Egger

Medical imaging segmentation is essential in clinical settings for diagnosing
diseases, planning surgeries, and other procedures. However, manual annotation
is a cumbersome and effortful task. To mitigate these aspects, this study
implements and evaluates the usability and clinical applicability of an
extended reality (XR)-based segmentation tool for anatomical CT scans, using
the Meta Quest 3 headset and Logitech MX Ink stylus. We develop an immersive
interface enabling real-time interaction with 2D and 3D medical imaging data in
a customizable workspace designed to mitigate workflow fragmentation and
cognitive demands inherent to conventional manual segmentation tools. The
platform combines stylus-driven annotation, mirroring traditional pen-on-paper
workflows, with instant 3D volumetric rendering. A user study with a public
craniofacial CT dataset demonstrated the tool's foundational viability,
achieving a System Usability Scale (SUS) score of 66, within the expected range
for medical applications. Participants highlighted the system's intuitive
controls (scoring 4.1/5 for self-descriptiveness on ISONORM metrics) and
spatial interaction design, with qualitative feedback highlighting strengths in
hybrid 2D/3D navigation and realistic stylus ergonomics. While users identified
opportunities to enhance task-specific precision and error management, the
platform's core workflow enabled dynamic slice adjustment, reducing cognitive
load compared to desktop tools. Results position the XR-stylus paradigm as a
promising foundation for immersive segmentation tools, with iterative
refinements targeting haptic feedback calibration and workflow personalization
to advance adoption in preoperative planning.

### 10. [LLMs for sensory-motor control: Combining in-context and iterative learning](http://arxiv.org/pdf/2506.04867v1)

Authors: Jônata Tyska Carvalho, Stefano Nolfi

We propose a method that enables large language models (LLMs) to control
embodied agents by directly mapping continuous observation vectors to
continuous action vectors. Initially, the LLMs generate a control strategy
based on a textual description of the agent, its environment, and the intended
goal. This strategy is then iteratively refined through a learning process in
which the LLMs are repeatedly prompted to improve the current strategy, using
performance feedback and sensory-motor data collected during its evaluation.
The method is validated on classic control tasks from the Gymnasium library and
the inverted pendulum task from the MuJoCo library. In most cases, it
successfully identifies optimal or high-performing solutions by integrating
symbolic knowledge derived through reasoning with sub-symbolic sensory-motor
data gathered as the agent interacts with its environment.

### Information Retrieval

### 1. [PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator for Recommender System Evaluation](http://arxiv.org/pdf/2506.04551v1)

Authors: Chenglong Ma, Ziqi Xu, Yongli Ren, Danula Hettiachchi, Jeffrey Chan

Traditional offline evaluation methods for recommender systems struggle to
capture the complexity of modern platforms due to sparse behavioural signals,
noisy data, and limited modelling of user personality traits. While simulation
frameworks can generate synthetic data to address these gaps, existing methods
fail to replicate behavioural diversity, limiting their effectiveness. To
overcome these challenges, we propose the Personality-driven User Behaviour
Simulator (PUB), an LLM-based simulation framework that integrates the Big Five
personality traits to model personalised user behaviour. PUB dynamically infers
user personality from behavioural logs (e.g., ratings, reviews) and item
metadata, then generates synthetic interactions that preserve statistical
fidelity to real-world data. Experiments on the Amazon review datasets show
that logs generated by PUB closely align with real user behaviour and reveal
meaningful associations between personality traits and recommendation outcomes.
These results highlight the potential of the personality-driven simulator to
advance recommender system evaluation, offering scalable, controllable,
high-fidelity alternatives to resource-intensive real-world experiments.

### 2. [Rethinking Contrastive Learning in Session-based Recommendation](http://arxiv.org/pdf/2506.05044v1)

Authors: Xiaokun Zhang, Bo Xu, Fenglong Ma, Zhizheng Wang, Liang Yang, Hongfei Lin

Session-based recommendation aims to predict intents of anonymous users based
on limited behaviors. With the ability in alleviating data sparsity,
contrastive learning is prevailing in the task. However, we spot that existing
contrastive learning based methods still suffer from three obstacles: (1) they
overlook item-level sparsity and primarily focus on session-level sparsity; (2)
they typically augment sessions using item IDs like crop, mask and reorder,
failing to ensure the semantic consistency of augmented views; (3) they treat
all positive-negative signals equally, without considering their varying
utility. To this end, we propose a novel multi-modal adaptive contrastive
learning framework called MACL for session-based recommendation. In MACL, a
multi-modal augmentation is devised to generate semantically consistent views
at both item and session levels by leveraging item multi-modal features.
Besides, we present an adaptive contrastive loss that distinguishes varying
contributions of positive-negative signals to improve self-supervised learning.
Extensive experiments on three real-world datasets demonstrate the superiority
of MACL over state-of-the-art methods.

### 3. [On the Comprehensibility of Multi-structured Financial Documents using LLMs and Pre-processing Tools](http://arxiv.org/pdf/2506.05182v1)

Authors: Shivani Upadhyay, Messiah Ataey, Shariyar Murtuza, Yifan Nie, Jimmy Lin

The proliferation of complex structured data in hybrid sources, such as PDF
documents and web pages, presents unique challenges for current Large Language
Models (LLMs) and Multi-modal Large Language Models (MLLMs) in providing
accurate answers. Despite the recent advancements of MLLMs, they still often
falter when interpreting intricately structured information, such as nested
tables and multi-dimensional plots, leading to hallucinations and erroneous
outputs. This paper explores the capabilities of LLMs and MLLMs in
understanding and answering questions from complex data structures found in PDF
documents by leveraging industrial and open-source tools as part of a
pre-processing pipeline. Our findings indicate that GPT-4o, a popular MLLM,
achieves an accuracy of 56% on multi-structured documents when fed documents
directly, and that integrating pre-processing tools raises the accuracy of LLMs
to 61.3% for GPT-4o and 76% for GPT-4, and with lower overall cost. The code is
publicly available at https://github.com/OGCDS/FinancialQA.

### 4. [Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion](http://arxiv.org/pdf/2506.04760v1)

Authors: Lingyuan Liu, Mengxiang Zhang

Large Language Models (LLMs) have shown potential in generating hypothetical
documents for query expansion, thereby enhancing information retrieval
performance. However, the efficacy of this method is highly dependent on the
quality of the generated documents, which often requires complex prompt
strategies and the integration of advanced dense retrieval techniques. This can
be both costly and computationally intensive. To mitigate these limitations, we
explore the use of zero-shot LLM-based query expansion to improve sparse
retrieval, particularly for learned sparse retrievers. We introduce a novel
fusion ranking framework, Exp4Fuse, which enhances the performance of sparse
retrievers through an indirect application of zero-shot LLM-based query
expansion. Exp4Fuse operates by simultaneously considering two retrieval
routes-one based on the original query and the other on the LLM-augmented
query. It then generates two ranked lists using a sparse retriever and fuses
them using a modified reciprocal rank fusion method. We conduct extensive
evaluations of Exp4Fuse against leading LLM-based query expansion methods and
advanced retrieval techniques on three MS MARCO-related datasets and seven
low-resource datasets. Experimental results reveal that Exp4Fuse not only
surpasses existing LLM-based query expansion methods in enhancing sparse
retrievers but also, when combined with advanced sparse retrievers, achieves
SOTA results on several benchmarks. This highlights the superior performance
and effectiveness of Exp4Fuse in improving query expansion for sparse
retrieval.

### 5. [GOLFer: Smaller LM-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval](http://arxiv.org/pdf/2506.04762v1)

Authors: Lingyuan Liu, Mengxiang Zhang

Large language models (LLMs)-based query expansion for information retrieval
augments queries with generated hypothetical documents with LLMs. However, its
performance relies heavily on the scale of the language models (LMs),
necessitating larger, more advanced LLMs. This approach is costly,
computationally intensive, and often has limited accessibility. To address
these limitations, we introduce GOLFer - Smaller LMs-Generated Documents
Hallucination Filter & Combiner - a novel method leveraging smaller open-source
LMs for query expansion. GOLFer comprises two modules: a hallucination filter
and a documents combiner. The former detects and removes non-factual and
inconsistent sentences in generated documents, a common issue with smaller LMs,
while the latter combines the filtered content with the query using a weight
vector to balance their influence. We evaluate GOLFer alongside dominant
LLM-based query expansion methods on three web search and ten low-resource
datasets. Experimental results demonstrate that GOLFer consistently outperforms
other methods using smaller LMs, and maintains competitive performance against
methods using large-size LLMs, demonstrating its effectiveness.

### 6. [Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings](http://arxiv.org/pdf/2506.04997v1)

Authors: Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Jiaqi Wang, Yixin Cao, Aixin Sun

Despite the strong performance of ColPali/ColQwen2 in Visualized Document
Retrieval (VDR), it encodes each page into multiple patch-level embeddings and
leads to excessive memory usage. This empirical study investigates methods to
reduce patch embeddings per page at minimum performance degradation. We
evaluate two token-reduction strategies: token pruning and token merging.
Regarding token pruning, we surprisingly observe that a simple random strategy
outperforms other sophisticated pruning methods, though still far from
satisfactory. Further analysis reveals that pruning is inherently unsuitable
for VDR as it requires removing certain page embeddings without query-specific
information. Turning to token merging (more suitable for VDR), we search for
the optimal combinations of merging strategy across three dimensions and
develop Light-ColPali/ColQwen2. It maintains 98.2% of retrieval performance
with only 11.8% of original memory usage, and preserves 94.6% effectiveness at
2.8% memory footprint. We expect our empirical findings and resulting
Light-ColPali/ColQwen2 offer valuable insights and establish a competitive
baseline for future research towards efficient VDR.

### 7. [Reason-to-Recommend: Using Interaction-of-Thought Reasoning to Enhance LLM Recommendation](http://arxiv.org/pdf/2506.05069v1)

Authors: Keyu Zhao, Fengli Xu, Yong Li

Driven by advances in Large Language Models (LLMs), integrating them into
recommendation tasks has gained interest due to their strong semantic
understanding and prompt flexibility. Prior work encoded user-item interactions
or metadata into prompts for recommendations. In parallel, LLM reasoning,
boosted by test-time scaling and reinforcement learning, has excelled in fields
like mathematics and code, where reasoning traces and correctness signals are
clear, enabling high performance and interpretability. However, directly
applying these reasoning methods to recommendation is ineffective because user
feedback is implicit and lacks reasoning supervision. To address this, we
propose $\textbf{R2Rec}$, a reasoning-enhanced recommendation framework that
samples interaction chains from the user-item graph and converts them into
structured interaction-of-thoughts via a progressive masked prompting strategy,
with each thought representing stepwise reasoning grounded in interaction
context. This allows LLMs to simulate step-by-step decision-making based on
implicit patterns. We design a two-stage training pipeline: supervised
fine-tuning teaches basic reasoning from high-quality traces, and reinforcement
learning refines reasoning via reward signals, alleviating sparse explicit
supervision. Experiments on three real-world datasets show R2Rec outperforms
classical and LLM-based baselines with an average $\textbf{10.48%}$ improvement
in HitRatio@1 and $\textbf{131.81%}$ gain over the original LLM. Furthermore,
the explicit reasoning chains enhance interpretability by revealing the
decision process. Our code is available at:
https://anonymous.4open.science/r/R2Rec-7C5D.

### 8. [LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](http://arxiv.org/pdf/2506.04790v1)

Authors: Yusuke Matsui

Approximate nearest neighbor search (ANNS) is an essential building block for
applications like RAG but can sometimes yield results that are overly similar
to each other. In certain scenarios, search results should be similar to the
query and yet diverse. We propose LotusFilter, a post-processing module to
diversify ANNS results. We precompute a cutoff table summarizing vectors that
are close to each other. During the filtering, LotusFilter greedily looks up
the table to delete redundant vectors from the candidates. We demonstrated that
the LotusFilter operates fast (0.02 [ms/query]) in settings resembling
real-world RAG applications, utilizing features such as OpenAI embeddings. Our
code is publicly available at https://github.com/matsui528/lotf.

### 9. [Verbose ListOps (VLO): Beyond Long Context -- Unmasking LLM's Reasoning Blind Spots](http://arxiv.org/pdf/2506.04907v1)

Authors: Alex Pan, Mary-Anne Williams

Large Language Models (LLMs), whilst great at extracting facts from text,
struggle with nested narrative reasoning. Existing long context and multi-hop
QA benchmarks inadequately test this, lacking realistic distractors or failing
to decouple context length from reasoning complexity, masking a fundamental LLM
limitation. We introduce Verbose ListOps, a novel benchmark that
programmatically transposes ListOps computations into lengthy, coherent
stories. This uniquely forces internal computation and state management of
nested reasoning problems by withholding intermediate results, and offers
fine-grained controls for both narrative size \emph{and} reasoning difficulty.
Whilst benchmarks like LongReason (2025) advance approaches for synthetically
expanding the context size of multi-hop QA problems, Verbose ListOps pinpoints
a specific LLM vulnerability: difficulty in state management for nested
sub-reasoning amongst semantically-relevant, distracting narrative. Our
experiments show that leading LLMs (e.g., OpenAI o4, Gemini 2.5 Pro) collapse
in performance on Verbose ListOps at modest (~10k token) narrative lengths,
despite effortlessly solving raw ListOps equations. Addressing this failure is
paramount for real-world text interpretation which requires identifying key
reasoning points, tracking conceptual intermediate results, and filtering
irrelevant information. Verbose ListOps, and its extensible generation
framework thus enables targeted reasoning enhancements beyond mere
context-window expansion; a critical step to automating the world's knowledge
work.

### 10. [Knowledgeable-r1: Policy Optimization for Knowledge Exploration in Retrieval-Augmented Generation](http://arxiv.org/pdf/2506.05154v1)

Authors: Chenyu Lin, Yilin Wen, Du Su, Fei Sun, Muhan Chen, Chenfu Bao, Zhonghou Lv

Retrieval-augmented generation (RAG) is a mainstream method for improving
performance on knowledge-intensive tasks. However,current RAG systems often
place too much emphasis on retrieved contexts. This can lead to reliance on
inaccurate sources and overlook the model's inherent knowledge, especially when
dealing with misleading or excessive information. To resolve this imbalance, we
propose Knowledgeable-r1 that using joint sampling and define multi policy
distributions in knowledge capability exploration to stimulate large language
models'self-integrated utilization of parametric and contextual knowledge.
Experiments show that Knowledgeable-r1 significantly enhances robustness and
reasoning accuracy in both parameters and contextual conflict tasks and general
RAG tasks, especially outperforming baselines by 17.07% in counterfactual
scenarios and demonstrating consistent gains across RAG tasks. Our code are
available at https://github.com/lcy80366872/ knowledgeable-r1.

### Machine Learning

### 1. [Hierarchical Implicit Neural Emulators](http://arxiv.org/pdf/2506.04528v1)

Authors: Ruoxi Jiang, Xiao Zhang, Karan Jakhar, Peter Y. Lu, Pedram Hassanzadeh, Michael Maire, Rebecca Willett

Neural PDE solvers offer a powerful tool for modeling complex dynamical
systems, but often struggle with error accumulation over long time horizons and
maintaining stability and physical consistency. We introduce a multiscale
implicit neural emulator that enhances long-term prediction accuracy by
conditioning on a hierarchy of lower-dimensional future state representations.
Drawing inspiration from the stability properties of numerical implicit
time-stepping methods, our approach leverages predictions several steps ahead
in time at increasing compression rates for next-timestep refinements. By
actively adjusting the temporal downsampling ratios, our design enables the
model to capture dynamics across multiple granularities and enforce long-range
temporal coherence. Experiments on turbulent fluid dynamics show that our
method achieves high short-term accuracy and produces long-term stable
forecasts, significantly outperforming autoregressive baselines while adding
minimal computational overhead.

### 2. [HALoS: Hierarchical Asynchronous Local SGD over Slow Networks for Geo-Distributed Large Language Model Training](http://arxiv.org/pdf/2506.04531v1)

Authors: Geon-Woo Kim, Junbo Li, Shashidhar Gandham, Omar Baldonado, Adithya Gangidi, Pavan Balaji, Zhangyang Wang, Aditya Akella

Training large language models (LLMs) increasingly relies on geographically
distributed accelerators, causing prohibitive communication costs across
regions and uneven utilization of heterogeneous hardware. We propose HALoS, a
hierarchical asynchronous optimization framework that tackles these issues by
introducing local parameter servers (LPSs) within each region and a global
parameter server (GPS) that merges updates across regions. This hierarchical
design minimizes expensive inter-region communication, reduces straggler
effects, and leverages fast intra-region links. We provide a rigorous
convergence analysis for HALoS under non-convex objectives, including
theoretical guarantees on the role of hierarchical momentum in asynchronous
training. Empirically, HALoS attains up to 7.5x faster convergence than
synchronous baselines in geo-distributed LLM training and improves upon
existing asynchronous methods by up to 2.1x. Crucially, HALoS preserves the
model quality of fully synchronous SGD-matching or exceeding accuracy on
standard language modeling and downstream benchmarks-while substantially
lowering total training time. These results demonstrate that hierarchical,
server-side update accumulation and global model merging are powerful tools for
scalable, efficient training of new-era LLMs in heterogeneous, geo-distributed
environments.

### 3. [Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](http://arxiv.org/pdf/2506.04542v1)

Authors: Yuanpei Gao, Qi Yan, Yan Leng, Renjie Liao

While deep learning methods have achieved strong performance in time series
prediction, their black-box nature and inability to explicitly model underlying
stochastic processes often limit their generalization to non-stationary data,
especially in the presence of abrupt changes. In this work, we introduce Neural
MJD, a neural network based non-stationary Merton jump diffusion (MJD) model.
Our model explicitly formulates forecasting as a stochastic differential
equation (SDE) simulation problem, combining a time-inhomogeneous It\^o
diffusion to capture non-stationary stochastic dynamics with a
time-inhomogeneous compound Poisson process to model abrupt jumps. To enable
tractable learning, we introduce a likelihood truncation mechanism that caps
the number of jumps within small time intervals and provide a theoretical error
bound for this approximation. Additionally, we propose an Euler-Maruyama with
restart solver, which achieves a provably lower error bound in estimating
expected states and reduced variance compared to the standard solver.
Experiments on both synthetic and real-world datasets demonstrate that Neural
MJD consistently outperforms state-of-the-art deep learning and statistical
learning methods.

### 4. [Communication Efficient Adaptive Model-Driven Quantum Federated Learning](http://arxiv.org/pdf/2506.04548v1)

Authors: Dev Gurung, Shiva Raj Pokhrel

Training with huge datasets and a large number of participating devices leads
to bottlenecks in federated learning (FL). Furthermore, the challenges of
heterogeneity between multiple FL clients affect the overall performance of the
system. In a quantum federated learning (QFL) context, we address these three
main challenges: i) training bottlenecks from massive datasets, ii) the
involvement of a substantial number of devices, and iii) non-IID data
distributions. We introduce a model-driven quantum federated learning algorithm
(mdQFL) to tackle these challenges. Our proposed approach is efficient and
adaptable to various factors, including different numbers of devices. To the
best of our knowledge, it is the first to explore training and update
personalization, as well as test generalization within a QFL setting, which can
be applied to other FL scenarios. We evaluated the efficiency of the proposed
mdQFL framework through extensive experiments under diverse non-IID data
heterogeneity conditions using various datasets within the Qiskit environment.
Our results demonstrate a nearly 50% decrease in total communication costs
while maintaining or, in some cases, exceeding the accuracy of the final model
and consistently improving local model training compared to the standard QFL
baseline. Moreover, our experimental evaluation thoroughly explores the QFL and
mdQFL algorithms, along with several influencing factors. In addition, we
present a theoretical analysis to clarify the complexities of the proposed
algorithm. The experimental code is available at 1.

### 5. [Ignoring Directionality Leads to Compromised Graph Neural Network Explanations](http://arxiv.org/pdf/2506.04608v1)

Authors: Changsheng Sun, Xinke Li, Jin Song Dong

Graph Neural Networks (GNNs) are increasingly used in critical domains, where
reliable explanations are vital for supporting human decision-making. However,
the common practice of graph symmetrization discards directional information,
leading to significant information loss and misleading explanations. Our
analysis demonstrates how this practice compromises explanation fidelity.
Through theoretical and empirical studies, we show that preserving directional
semantics significantly improves explanation quality, ensuring more faithful
insights for human decision-makers. These findings highlight the need for
direction-aware GNN explainability in security-critical applications.

### 6. [Composing Agents to Minimize Worst-case Risk](http://arxiv.org/pdf/2506.04632v1)

Authors: Guruprerana Shabadi, Rajeev Alur

From software development to robot control, modern agentic systems decompose
complex objectives into a sequence of subtasks and choose a set of specialized
AI agents to complete them. We formalize an agentic workflow as a directed
acyclic graph, called an agent graph, where edges represent AI agents and paths
correspond to feasible compositions of agents. When deploying these systems in
the real world, we need to choose compositions of agents that not only maximize
the task success, but also minimize risk where the risk captures requirements
like safety, fairness, and privacy. This additionally requires carefully
analyzing the low-probability (tail) behaviors of compositions of agents. In
this work, we consider worst-case risk minimization over the set of feasible
agent compositions. We define worst-case risk as the tail quantile -- also
known as value-at-risk -- of the loss distribution of the agent composition
where the loss quantifies the risk associated with agent behaviors. We
introduce an efficient algorithm that traverses the agent graph and finds a
near-optimal composition of agents by approximating the value-at-risk via a
union bound and dynamic programming. Furthermore, we prove that the
approximation is near-optimal asymptotically for a broad class of practical
loss functions. To evaluate our framework, we consider a suite of video
game-like control benchmarks that require composing several agents trained with
reinforcement learning and demonstrate our algorithm's effectiveness in
approximating the value-at-risk and identifying the optimal agent composition.

### 7. [Neural Network Reprogrammability: A Unified Theme on Model Reprogramming, Prompt Tuning, and Prompt Instruction](http://arxiv.org/pdf/2506.04650v1)

Authors: Zesheng Ye, Chengyi Cai, Ruijiang Dong, Jianzhong Qi, Lei Feng, Pin-Yu Chen, Feng Liu

As large-scale pre-trained foundation models continue to expand in size and
capability, efficiently adapting them to specific downstream tasks has become
increasingly critical. Despite substantial progress, existing adaptation
approaches have evolved largely in isolation, without a clear understanding of
their interrelationships. This survey introduces neural network
reprogrammability as a unifying framework that bridges mainstream model
adaptation techniques--model reprogramming, prompt tuning, and prompt
instruction--previously fragmented research areas yet converges on a shared
principle: repurposing a pre-trained model by manipulating information at the
interfaces while keeping the model parameters frozen. These methods exploit
neural networks' sensitivity to manipulation on different interfaces, be it
through perturbing inputs, inserting tokens into intermediate layers, or
providing task-specific examples in context, to redirect model behaviors
towards desired outcomes. We then present a taxonomy that categorizes such
information manipulation-based adaptation approaches across four key
dimensions: manipulation format (fixed or learnable), location (interfaces
where manipulations occur), operator (how they are applied), and output
alignment requirement (post-processing needed to align outputs with downstream
tasks). Notably, this framework applies consistently across data modalities,
independent of specific model architectures. Moreover, viewing established
techniques like in-context learning and chain-of-thought prompting through this
lens reveals both their theoretical connections and practical distinctions. We
further analyze remaining technical challenges and ethical considerations,
positioning neural network reprogrammability as a fundamental paradigm for
efficient model adaptation. We lastly identify promising research directions
emerging from this integrative viewpoint.

### 8. [The Oversmoothing Fallacy: A Misguided Narrative in GNN Research](http://arxiv.org/pdf/2506.04653v1)

Authors: MoonJeong Park, Sunghyun Choi, Jaeseung Heo, Eunhyeok Park, Dongwoo Kim

Oversmoothing has been recognized as a main obstacle to building deep Graph
Neural Networks (GNNs), limiting the performance. This position paper argues
that the influence of oversmoothing has been overstated and advocates for a
further exploration of deep GNN architectures. Given the three core operations
of GNNs, aggregation, linear transformation, and non-linear activation, we show
that prior studies have mistakenly confused oversmoothing with the vanishing
gradient, caused by transformation and activation rather than aggregation. Our
finding challenges prior beliefs about oversmoothing being unique to GNNs.
Furthermore, we demonstrate that classical solutions such as skip connections
and normalization enable the successful stacking of deep GNN layers without
performance degradation. Our results clarify misconceptions about oversmoothing
and shed new light on the potential of deep GNNs.

### 9. [Noise-Resistant Label Reconstruction Feature Selection for Partial Multi-Label Learning](http://arxiv.org/pdf/2506.04669v1)

Authors: Wanfu Gao, Hanlin Pan, Qingqi Han, Kunpeng Liu

The "Curse of dimensionality" is prevalent across various data patterns,
which increases the risk of model overfitting and leads to a decline in model
classification performance. However, few studies have focused on this issue in
Partial Multi-label Learning (PML), where each sample is associated with a set
of candidate labels, at least one of which is correct. Existing PML methods
addressing this problem are mainly based on the low-rank assumption. However,
low-rank assumption is difficult to be satisfied in practical situations and
may lead to loss of high-dimensional information. Furthermore, we find that
existing methods have poor ability to identify positive labels, which is
important in real-world scenarios. In this paper, a PML feature selection
method is proposed considering two important characteristics of dataset: label
relationship's noise-resistance and label connectivity. Our proposed method
utilizes label relationship's noise-resistance to disambiguate labels. Then the
learning process is designed through the reformed low-rank assumption. Finally,
representative labels are found through label connectivity, and the weight
matrix is reconstructed to select features with strong identification ability
to these labels. The experimental results on benchmark datasets demonstrate the
superiority of the proposed method.

### 10. [FedAPM: Federated Learning via ADMM with Partial Model Personalization](http://arxiv.org/pdf/2506.04672v1)

Authors: Shengkun Zhu, Feiteng Nie, Jinshan Zeng, Sheng Wang, Yuan Sun, Yuan Yao, Shangfeng Chen, Quanqing Xu, Chuanhui Yang

In federated learning (FL), the assumption that datasets from different
devices are independent and identically distributed (i.i.d.) often does not
hold due to user differences, and the presence of various data modalities
across clients makes using a single model impractical. Personalizing certain
parts of the model can effectively address these issues by allowing those parts
to differ across clients, while the remaining parts serve as a shared model.
However, we found that partial model personalization may exacerbate client
drift (each client's local model diverges from the shared model), thereby
reducing the effectiveness and efficiency of FL algorithms. We propose an FL
framework based on the alternating direction method of multipliers (ADMM),
referred to as FedAPM, to mitigate client drift. We construct the augmented
Lagrangian function by incorporating first-order and second-order proximal
terms into the objective, with the second-order term providing fixed correction
and the first-order term offering compensatory correction between the local and
shared models. Our analysis demonstrates that FedAPM, by using explicit
estimates of the Lagrange multiplier, is more stable and efficient in terms of
convergence compared to other FL frameworks. We establish the global
convergence of FedAPM training from arbitrary initial points to a stationary
point, achieving three types of rates: constant, linear, and sublinear, under
mild assumptions. We conduct experiments using four heterogeneous and
multimodal datasets with different metrics to validate the performance of
FedAPM. Specifically, FedAPM achieves faster and more accurate convergence,
outperforming the SOTA methods with average improvements of 12.3% in test
accuracy, 16.4% in F1 score, and 18.0% in AUC while requiring fewer
communication rounds.

### Neural and Evolutionary Computing

### 1. [NEAT and HyperNEAT based Design for Soft Actuator Controllers](http://arxiv.org/pdf/2506.04698v1)

Authors: Hugo Alcaraz-Herrera, Michail-Antisthenis Tsompanas, Igor Balaz, Andrew Adamatzky

Since soft robotics are composed of compliant materials, they perform better
than conventional rigid robotics in specific fields, such as medical
applications. However, the field of soft robotics is fairly new, and the design
process of their morphology and their controller strategies has not yet been
thoroughly studied. Consequently, here, an automated design method for the
controller of soft actuators based on Neuroevolution is proposed. Specifically,
the suggested techniques employ Neuroevolution of Augmenting Topologies (NEAT)
and Hypercube-based NEAT (HyperNEAT) to generate the synchronization profile of
the components of a simulated soft actuator by employing Compositional Pattern
Producing Networks (CPPNs). As a baseline methodology, a Standard Genetic
Algorithm (SGA) was used. Moreover, to test the robustness of the proposed
methodologies, both high- and low-performing morphologies of soft actuators
were utilized as testbeds. Moreover, the use of an affluent and a more limited
set of activation functions for the Neuroevolution targets was tested
throughout the experiments. The results support the hypothesis that
Neuroevolution based methodologies are more appropriate for designing
controllers that align with both types of morphologies. In specific, NEAT
performed better for all different scenarios tested and produced more
simplistic networks that are easier to implement in real life applications.

### 2. [Perturbative Gradient Training: A novel training paradigm for bridging the gap between deep neural networks and physical reservoir computing](http://arxiv.org/pdf/2506.04523v1)

Authors: Cliff B. Abbott, Mark Elo, Dmytro A. Bozhko

We introduce Perturbative Gradient Training (PGT), a novel training paradigm
that overcomes a critical limitation of physical reservoir computing: the
inability to perform backpropagation due to the black-box nature of physical
reservoirs. Drawing inspiration from perturbation theory in physics, PGT uses
random perturbations in the network's parameter space to approximate gradient
updates using only forward passes. We demonstrate the feasibility of this
approach on both simulated neural network architectures, including a dense
network and a transformer model with a reservoir layer, and on experimental
hardware using a magnonic auto-oscillation ring as the physical reservoir. Our
results show that PGT can achieve performance comparable to that of standard
backpropagation methods in cases where backpropagation is impractical or
impossible. PGT represents a promising step toward integrating physical
reservoirs into deeper neural network architectures and achieving significant
energy efficiency gains in AI training.

### Networking and Internet Architecture

### 1. [Grey Rhino Warning: IPv6 is Becoming Fertile Ground for Reflection Amplification Attacks](http://arxiv.org/pdf/2506.04768v1)

Authors: Ling Hu, Tao Yang, Yu Pang, Bingnan Hou, Zhiping Cai, Bo Yu

Distributed Denial-of-Service (DDoS) attacks represent a cost-effective and
potent threat to network stability. While extensively studied in IPv4 networks,
DDoS implications in IPv6 remain underexplored. The vast IPv6 address space
renders brute-force scanning and amplifier testing for all active addresses
impractical. Innovatively, this work investigates AS-level vulnerabilities to
reflection amplification attacks in IPv6.
  One prerequisite for amplification presence is that it is located in a
vulnerable autonomous system (AS) without inbound source address validation
(ISAV) deployment. Hence, the analysis focuses on two critical aspects: global
detection of ISAV deployment and identification of amplifiers within vulnerable
ASes. Specifically, we develop a methodology combining ICMP Time Exceeded
mechanisms for ISAV detection, employ IPv6 address scanning for amplifier
identification, and utilize dual vantage points for amplification verification.
  Experimental results reveal that 4,460 ASes (61.36% of measured networks)
lack ISAV deployment. Through scanning approximately 47M active addresses, we
have identified reflection amplifiers in 3,507 ASes. The analysis demonstrates
that current IPv6 networks are fertile grounds for reflection amplification
attacks, alarming network security.

### 2. [Federated Learning Assisted Edge Caching Scheme Based on Lightweight Architecture DDPM](http://arxiv.org/pdf/2506.04593v1)

Authors: Xun Li, Qiong Wu

Edge caching is an emerging technology that empowers caching units at edge
nodes, allowing users to fetch contents of interest that have been pre-cached
at the edge nodes. The key to pre-caching is to maximize the cache hit
percentage for cached content without compromising users' privacy. In this
letter, we propose a federated learning (FL) assisted edge caching scheme based
on lightweight architecture denoising diffusion probabilistic model (LDPM). Our
simulation results verify that our proposed scheme achieves a higher cache hit
percentage compared to existing FL-based methods and baseline methods.

### 3. [Towards Network Data Analytics in 5G Systems and Beyond](http://arxiv.org/pdf/2506.04860v1)

Authors: Marcos Lima Romero, Ricardo Suyama

Data has become a critical asset in the digital economy, yet it remains
underutilized by Mobile Network Operators (MNOs), unlike Over-the-Top (OTT)
players that lead global market valuations. To move beyond the commoditization
of connectivity and deliver greater value to customers, data analytics emerges
as a strategic enabler. Using data efficiently is essential for unlocking new
service opportunities, optimizing operational efficiency, and mitigating
operational and business risks. Since Release 15, the 3rd Generation
Partnership Project (3GPP) has introduced the Network Data Analytics Function
(NWDAF) to provide powerful insights and predictions using data collected
across mobile networks, supporting both user-centric and network-oriented use
cases. However, academic research has largely focused on a limited set of
methods and use cases, driven by the availability of datasets, restricting
broader exploration. This study analyzes trends and gaps in more than 70
articles and proposes two novel use cases to promote the adoption of NWDAF and
explore its potential for monetization.

### 4. [Indoor Sharing in the Mid-Band: A Performance Study of Neutral-Host, Cellular Macro, and Wi-Fi](http://arxiv.org/pdf/2506.04974v1)

Authors: Joshua Roy Palathinkal, Muhammad Iqbal Rochman, Vanlin Sathya, Mehmet Yavuz, Monisha Ghosh

Indoor environments present a significant challenge for wireless
connectivity, as immense data demand strains traditional solutions. Public
Mobile Network Operators (MNOs), utilizing outdoor macro base stations (BSs),
suffer from poor signal penetration. Indoor Wi-Fi networks, on the other hand,
may face reliability issues due to spectrum contention. Shared spectrum models,
particularly the Citizens Broadband Radio Service (CBRS) utilized by private
4G/5G networks, have emerged as a promising alternative to provide reliable
indoor service. Moreover, these private networks are equipped with the
neutral-host (NH) model, seamlessly offloading indoor MNOs' traffic to the
private CBRS network. This paper presents a comprehensive, in-situ performance
evaluation of three co-located technologies utilizing mid-bands spectrum (1-6
GHz)--a CBRS-based NH network, public MNO macro networks, and a Wi-Fi 6
network--within a large, big-box retail store characterized by significant
building loss. Our analysis demonstrates: (i) the NH network provides superior
indoor coverage compared to MNO macro, requiring only six CBRS devices
(CBSDs)--versus 65 Access Points (APs) for enterprise Wi-Fi--to achieve full
coverage, with a median building loss of 26.6 dB ensuring interference-free
coexistence with outdoor federal incumbents; (ii) the NH network achieves
substantial indoor throughput gains, with per-channel normalized throughput
improvements of 1.44x and 1.62x in downlink (DL), and 4.33x and 13x in uplink
(UL), compared to 4G and 5G macro deployments, respectively; (iii) the NH
deployment achieves a median indoor aggregated physical (PHY)-layer DL
throughput gain of 2.08x over 5G macro deployments indoors, despite utilizing
only 40 MHz of aggregated bandwidth compared to 225 MHz for 5G macro; and (iv)
the NH deployment also outperforms Wi-Fi in application-layer HTTP DL
performance by 5.05x.

### 5. [Intelligent Channel Allocation for IEEE 802.11be Multi-Link Operation: When MAB Meets LLM](http://arxiv.org/pdf/2506.04594v1)

Authors: Shumin Lian, Jingwen Tong, Jun Zhang, Liqun Fu

WiFi networks have achieved remarkable success in enabling seamless
communication and data exchange worldwide. The IEEE 802.11be standard, known as
WiFi 7, introduces Multi-Link Operation (MLO), a groundbreaking feature that
enables devices to establish multiple simultaneous connections across different
bands and channels. While MLO promises substantial improvements in network
throughput and latency reduction, it presents significant challenges in channel
allocation, particularly in dense network environments. Current research has
predominantly focused on performance analysis and throughput optimization
within static WiFi 7 network configurations. In contrast, this paper addresses
the dynamic channel allocation problem in dense WiFi 7 networks with MLO
capabilities. We formulate this challenge as a combinatorial optimization
problem, leveraging a novel network performance analysis mechanism. Given the
inherent lack of prior network information, we model the problem within a
Multi-Armed Bandit (MAB) framework to enable online learning of optimal channel
allocations. Our proposed Best-Arm Identification-enabled Monte Carlo Tree
Search (BAI-MCTS) algorithm includes rigorous theoretical analysis, providing
upper bounds for both sample complexity and error probability. To further
reduce sample complexity and enhance generalizability across diverse network
scenarios, we put forth LLM-BAI-MCTS, an intelligent algorithm for the dynamic
channel allocation problem by integrating the Large Language Model (LLM) into
the BAI-MCTS algorithm. Numerical results demonstrate that the BAI-MCTS
algorithm achieves a convergence rate approximately $50.44\%$ faster than the
state-of-the-art algorithms when reaching $98\%$ of the optimal value. Notably,
the convergence rate of the LLM-BAI-MCTS algorithm increases by over $63.32\%$
in dense networks.

### 6. [On the Role of Early-Termination for Age of Information in Tree-Based Random Access Protocols](http://arxiv.org/pdf/2506.04793v1)

Authors: Andrea Munari, Cedomir Stefanovic

Age of Information (AoI) has emerged as a key metric for assessing data
freshness in IoT applications, where a large number of devices report
time-stamped updates to a monitor. Such systems often rely on random access
protocols based on variations of ALOHA at the link layer, where collision
resolution algorithms play a fundamental role to enable reliable delivery of
packets. In this context, we provide the first analytical characterization of
average AoI for the classical Capetanakis tree-based algorithm with gated
access under exogenous traffic, capturing the protocol's dynamics, driven by
sporadic packet generation and variable collision resolution times. We also
explore a variant with early termination, where contention is truncated after a
maximum number of slots even if not all users are resolved. The approach
introduces a fundamental trade-off between reliability and timeliness, allowing
stale packets to be dropped to improve freshness.

### 7. [Goal-Oriented Semantic Resource Allocation with Cumulative Prospect Theoretic Agents](http://arxiv.org/pdf/2506.04947v1)

Authors: Symeon Vaidanis, Photios A. Stavrou, Marios Kountouris

We introduce a resource allocation framework for goal-oriented semantic
networks, where participating agents assess system quality through subjective
(e.g., context-dependent) perceptions. To accommodate this, our model accounts
for agents whose preferences deviate from traditional expected utility theory
(EUT), specifically incorporating cumulative prospect theory (CPT) preferences.
We develop a comprehensive analytical framework that captures human-centric
aspects of decision-making and risky choices under uncertainty, such as risk
perception, loss aversion, and perceptual distortions in probability metrics.
By identifying essential modifications in traditional resource allocation
design principles required for agents with CPT preferences, we showcase the
framework's relevance through its application to the problem of power
allocation in multi-channel wireless communication systems.

### 8. [Optimization for Semantic-Aware Resource Allocation under CPT-based Utilities](http://arxiv.org/pdf/2506.04952v1)

Authors: Symeon Vaidanis, Photios A. Stavrou, Marios Kountouris

The problem of resource allocation in goal-oriented semantic communication
with semantic-aware utilities and subjective risk perception is studied here.
By linking information importance to risk aversion, we model agent behavior
using Cumulative Prospect Theory (CPT), which incorporates risk-sensitive
utility functions and nonlinear transformations of distributions, reflecting
subjective perceptions of gains and losses. The objective is to maximize the
aggregate utility across multiple CPT-modeled agents, which leads to a
nonconvex, nonsmooth optimization problem. To efficiently solve this
challenging problem, we propose a new algorithmic framework that combines
successive convex approximation (SCA) with the projected subgradient method and
Lagrangian relaxation, Our approach enables tractable optimization while
preserving solution quality, offering both theoretical rigor and practical
effectiveness in semantics-aware resource allocation.

### Robotics

### 1. [Multimodal Limbless Crawling Soft Robot with a Kirigami Skin](http://arxiv.org/pdf/2506.04547v1)

Authors: Jonathan Tirado, Aida Parvaresh, Burcu Seyidoğlu, Darryl A. Bedford, Jonas Jørgensen, Ahmad Rafsanjani

Limbless creatures can crawl on flat surfaces by deforming their bodies and
interacting with asperities on the ground, offering a biological blueprint for
designing efficient limbless robots. Inspired by this natural locomotion, we
present a soft robot capable of navigating complex terrains using a combination
of rectilinear motion and asymmetric steering gaits. The robot is made of a
pair of antagonistic inflatable soft actuators covered with a flexible kirigami
skin with asymmetric frictional properties. The robot's rectilinear locomotion
is achieved through cyclic inflation of internal chambers with precise phase
shifts, enabling forward progression. Steering is accomplished using an
asymmetric gait, allowing for both in-place rotation and wide turns. To
validate its mobility in obstacle-rich environments, we tested the robot in an
arena with coarse substrates and multiple obstacles. Real-time feedback from
onboard proximity sensors, integrated with a human-machine interface (HMI),
allowed adaptive control to avoid collisions. This study highlights the
potential of bioinspired soft robots for applications in confined or
unstructured environments, such as search-and-rescue operations, environmental
monitoring, and industrial inspections.

### 2. [A Novel Transformer-Based Method for Full Lower-Limb Joint Angles and Moments Prediction in Gait Using sEMG and IMU data](http://arxiv.org/pdf/2506.04577v1)

Authors: Farshad Haghgoo Daryakenari, Tara Farizeh

This study presents a transformer-based deep learning framework for the
long-horizon prediction of full lower-limb joint angles and joint moments using
surface electromyography (sEMG) and inertial measurement unit (IMU) signals.
Two separate Transformer Neural Networks (TNNs) were designed: one for
kinematic prediction and one for kinetic prediction. The model was developed
with real-time application in mind, using only wearable sensors suitable for
outside-laboratory use. Two prediction horizons were considered to evaluate
short- and long-term performance. The network achieved high accuracy in both
tasks, with Spearman correlation coefficients exceeding 0.96 and R-squared
scores above 0.92 across all joints. Notably, the model consistently
outperformed a recent benchmark method in joint angle prediction, reducing RMSE
errors by an order of magnitude. The results confirmed the complementary role
of sEMG and IMU signals in capturing both kinematic and kinetic information.
This work demonstrates the potential of transformer-based models for real-time,
full-limb biomechanical prediction in wearable and robotic applications, with
future directions including input minimization and modality-specific weighting
strategies to enhance model efficiency and accuracy.

### 3. [Enhancing Efficiency and Propulsion in Bio-mimetic Robotic Fish through End-to-End Deep Reinforcement Learning](http://arxiv.org/pdf/2506.04627v1)

Authors: Xinyu Cui, Boai Sun, Yi Zhu, Ning Yang, Haifeng Zhang, Weicheng Cui, Dixia Fan, Jun Wang

Aquatic organisms are known for their ability to generate efficient
propulsion with low energy expenditure. While existing research has sought to
leverage bio-inspired structures to reduce energy costs in underwater robotics,
the crucial role of control policies in enhancing efficiency has often been
overlooked. In this study, we optimize the motion of a bio-mimetic robotic fish
using deep reinforcement learning (DRL) to maximize propulsion efficiency and
minimize energy consumption. Our novel DRL approach incorporates extended
pressure perception, a transformer model processing sequences of observations,
and a policy transfer scheme. Notably, significantly improved training
stability and speed within our approach allow for end-to-end training of the
robotic fish. This enables agiler responses to hydrodynamic environments and
possesses greater optimization potential compared to pre-defined motion pattern
controls. Our experiments are conducted on a serially connected rigid robotic
fish in a free stream with a Reynolds number of 6000 using computational fluid
dynamics (CFD) simulations. The DRL-trained policies yield impressive results,
demonstrating both high efficiency and propulsion. The policies also showcase
the agent's embodiment, skillfully utilizing its body structure and engaging
with surrounding fluid dynamics, as revealed through flow analysis. This study
provides valuable insights into the bio-mimetic underwater robots optimization
through DRL training, capitalizing on their structural advantages, and
ultimately contributing to more efficient underwater propulsion systems.

### 4. [ActivePusher: Active Learning and Planning with Residual Physics for Nonprehensile Manipulation](http://arxiv.org/pdf/2506.04646v1)

Authors: Zhuoyun Zhong, Seyedali Golestaneh, Constantinos Chamzas

Planning with learned dynamics models offers a promising approach toward
real-world, long-horizon manipulation, particularly in nonprehensile settings
such as pushing or rolling, where accurate analytical models are difficult to
obtain. Although learning-based methods hold promise, collecting training data
can be costly and inefficient, as it often relies on randomly sampled
interactions that are not necessarily the most informative. To address this
challenge, we propose ActivePusher, a novel framework that combines
residual-physics modeling with kernel-based uncertainty-driven active learning
to focus data acquisition on the most informative skill parameters.
Additionally, ActivePusher seamlessly integrates with model-based kinodynamic
planners, leveraging uncertainty estimates to bias control sampling toward more
reliable actions. We evaluate our approach in both simulation and real-world
environments and demonstrate that it improves data efficiency and planning
success rates compared to baseline methods.

### 5. [Tire Wear Aware Trajectory Tracking Control for Multi-axle Swerve-drive Autonomous Mobile Robots](http://arxiv.org/pdf/2506.04752v1)

Authors: Tianxin Hu, Xinhang Xu, Thien-Minh Nguyen, Fen Liu, Shenghai Yuan, Lihua Xie

Multi-axle Swerve-drive Autonomous Mobile Robots (MS-AGVs) equipped with
independently steerable wheels are commonly used for high-payload
transportation. In this work, we present a novel model predictive control (MPC)
method for MS-AGV trajectory tracking that takes tire wear minimization
consideration in the objective function. To speed up the problem-solving
process, we propose a hierarchical controller design and simplify the dynamic
model by integrating the \textit{magic formula tire model} and
\textit{simplified tire wear model}. In the experiment, the proposed method can
be solved by simulated annealing in real-time on a normal personal computer and
by incorporating tire wear into the objective function, tire wear is reduced by
19.19\% while maintaining the tracking accuracy in curve-tracking experiments.
In the more challenging scene: the desired trajectory is offset by 60 degrees
from the vehicle's heading, the reduction in tire wear increased to 65.20\%
compared to the kinematic model without considering the tire wear optimization.

### 6. [ArtVIP: Articulated Digital Assets of Visual Realism, Modular Interaction, and Physical Fidelity for Robot Learning](http://arxiv.org/pdf/2506.04941v1)

Authors: Zhao Jin, Zhengping Che, Zhen Zhao, Kun Wu, Yuheng Zhang, Yinuo Zhao, Zehui Liu, Qiang Zhang, Xiaozhu Ju, Jing Tian, Yousong Xue, Jian Tang

Robot learning increasingly relies on simulation to advance complex ability
such as dexterous manipulations and precise interactions, necessitating
high-quality digital assets to bridge the sim-to-real gap. However, existing
open-source articulated-object datasets for simulation are limited by
insufficient visual realism and low physical fidelity, which hinder their
utility for training models mastering robotic tasks in real world. To address
these challenges, we introduce ArtVIP, a comprehensive open-source dataset
comprising high-quality digital-twin articulated objects, accompanied by
indoor-scene assets. Crafted by professional 3D modelers adhering to unified
standards, ArtVIP ensures visual realism through precise geometric meshes and
high-resolution textures, while physical fidelity is achieved via fine-tuned
dynamic parameters. Meanwhile, the dataset pioneers embedded modular
interaction behaviors within assets and pixel-level affordance annotations.
Feature-map visualization and optical motion capture are employed to
quantitatively demonstrate ArtVIP 's visual and physical fidelity, with its
applicability validated across imitation learning and reinforcement learning
experiments. Provided in USD format with detailed production guidelines, \ours
is fully open-source, benefiting the research community and advancing robot
learning research. Our project is at https://x-humanoid-artvip.github.io/

### 7. [A Pillbug-Inspired Morphing Mechanism Covered with Sliding Shells](http://arxiv.org/pdf/2506.04942v1)

Authors: Jieyu Wang, Yingzhong Tian, Fengfeng Xi, Damien Chablat, Jianing Lin, Gaoke Ren, Yinjun Zhao

This research proposes a novel morphing structure with shells inspired by the
movement of pillbugs. Instead of the pillbug body, a loopcoupled mechanism
based on slider-crank mechanisms is utilized to achieve the rolling up and
spreading motion. This mechanism precisely imitates three distinct curves that
mimic the shape morphing of a pillbug. To decrease the degree-of-freedom (DOF)
of the mechanism to one, scissor mechanisms are added. 3D curved shells are
then attached to the tracer points of the morphing mechanism to safeguard it
from attacks while allowing it to roll. Through type and dimensional synthesis,
a complete system that includes shells and an underlying morphing mechanism is
developed. A 3D model is created and tested to demonstrate the proposed
system's shape-changing capability. Lastly, a robot with two modes is developed
based on the proposed mechanism, which can curl up to roll down hills and can
spread to move in a straight line via wheels.

### 8. [GEX: Democratizing Dexterity with Fully-Actuated Dexterous Hand and Exoskeleton Glove](http://arxiv.org/pdf/2506.04982v1)

Authors: Yunlong Dong, Xing Liu, Jun Wan, Zelin Deng

This paper introduces GEX, an innovative low-cost dexterous manipulation
system that combines the GX11 tri-finger anthropomorphic hand (11 DoF) with the
EX12 tri-finger exoskeleton glove (12 DoF), forming a closed-loop teleoperation
framework through kinematic retargeting for high-fidelity control. Both
components employ modular 3D-printed finger designs, achieving ultra-low
manufacturing costs while maintaining full actuation capabilities. Departing
from conventional tendon-driven or underactuated approaches, our
electromechanical system integrates independent joint motors across all 23 DoF,
ensuring complete state observability and accurate kinematic modeling. This
full-actuation architecture enables precise bidirectional kinematic
calculations, substantially enhancing kinematic retargeting fidelity between
the exoskeleton and robotic hand. The proposed system bridges the
cost-performance gap in dexterous manipulation research, providing an
accessible platform for acquiring high-quality demonstration data to advance
embodied AI and dexterous robotic skill transfer learning.

### 9. [A Unified Framework for Simulating Strongly-Coupled Fluid-Robot Multiphysics](http://arxiv.org/pdf/2506.05012v1)

Authors: Jeong Hun Lee, Junzhe Hu, Sofia Kwok, Carmel Majidi, Zachary Manchester

We present a framework for simulating fluid-robot multiphysics as a single,
unified optimization problem. The coupled manipulator and incompressible
Navier-Stokes equations governing the robot and fluid dynamics are derived
together from a single Lagrangian using the principal of least action. We then
employ discrete variational mechanics to derive a stable, implicit
time-integration scheme for jointly simulating both the fluid and robot
dynamics, which are tightly coupled by a constraint that enforces the no-slip
boundary condition at the fluid-robot interface. Extending the classical
immersed boundary method, we derive a new formulation of the no-slip constraint
that is numerically well-conditioned and physically accurate for multibody
systems commonly found in robotics. We demonstrate our approach's physical
accuracy on benchmark computational fluid-dynamics problems, including
Poiseuille flow and a disc in free stream. We then design a locomotion policy
for a novel swimming robot in simulation and validate results on real-world
hardware, showcasing our framework's sim-to-real capability for robotics tasks.

### 10. [DemoSpeedup: Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration](http://arxiv.org/pdf/2506.05064v1)

Authors: Lingxiao Guo, Zhengrong Xue, Zijing Xu, Huazhe Xu

Imitation learning has shown great promise in robotic manipulation, but the
policy's execution is often unsatisfactorily slow due to commonly tardy
demonstrations collected by human operators. In this work, we present
DemoSpeedup, a self-supervised method to accelerate visuomotor policy execution
via entropy-guided demonstration acceleration. DemoSpeedup starts from training
an arbitrary generative policy (e.g., ACT or Diffusion Policy) on normal-speed
demonstrations, which serves as a per-frame action entropy estimator. The key
insight is that frames with lower action entropy estimates call for more
consistent policy behaviors, which often indicate the demands for
higher-precision operations. In contrast, frames with higher entropy estimates
correspond to more casual sections, and therefore can be more safely
accelerated. Thus, we segment the original demonstrations according to the
estimated entropy, and accelerate them by down-sampling at rates that increase
with the entropy values. Trained with the speedup demonstrations, the resulting
policies execute up to 3 times faster while maintaining the task completion
performance. Interestingly, these policies could even achieve higher success
rates than those trained with normal-speed demonstrations, due to the benefits
of reduced decision-making horizons.

### Software Engineering

### 1. [KPIRoot+: An Efficient Integrated Framework for Anomaly Detection and Root Cause Analysis in Large-Scale Cloud Systems](http://arxiv.org/pdf/2506.04569v1)

Authors: Wenwei Gu, Renyi Zhong, Guangba Yu, Xinying Sun, Jinyang Liu, Yintong Huo, Zhuangbin Chen, Jianping Zhang, Jiazhen Gu, Yongqiang Yang, Michael R. Lyu

To ensure the reliability of cloud systems, their performance is monitored
using KPIs (key performance indicators). When issues arise, root cause
localization identifies KPIs responsible for service degradation, aiding in
quick diagnosis and resolution. Traditional methods rely on similarity
calculations, which can be ineffective in complex, interdependent cloud
environments. While deep learning-based approaches model these dependencies
better, they often face challenges such as high computational demands and lack
of interpretability.
  To address these issues, KPIRoot is proposed as an efficient method combining
similarity and causality analysis. It uses symbolic aggregate approximation for
compact KPI representation, improving analysis efficiency. However, deployment
in Cloud H revealed two drawbacks: 1) threshold-based anomaly detection misses
some performance anomalies, and 2) SAX representation fails to capture
intricate variation trends. KPIRoot+ addresses these limitations, outperforming
eight state-of-the-art baselines by 2.9% to 35.7%, while reducing time cost by
34.7%. We also share our experience deploying KPIRoot in a large-scale cloud
provider's production environment.

### 2. [QuanUML: Towards A Modeling Language for Model-Driven Quantum Software Development](http://arxiv.org/pdf/2506.04639v1)

Authors: Xiaoyu Guo, Shinobu Saito, Jianjun Zhao

This paper introduces QuanUML, an extension of the Unified Modeling Language
(UML) tailored for quantum software systems. QuanUML integrates
quantum-specific constructs, such as qubits and quantum gates, into the UML
framework, enabling the modeling of both quantum and hybrid quantum-classical
systems. We apply QuanUML to Efficient Long-Range Entanglement using Dynamic
Circuits and Shor's Algorithm, demonstrating its utility in designing and
visualizing quantum algorithms. Our approach supports model-driven development
of quantum software and offers a structured framework for quantum software
design. We also highlight its advantages over existing methods and discuss
future improvements.

### 3. [From Developer Pairs to AI Copilots: A Comparative Study on Knowledge Transfer](http://arxiv.org/pdf/2506.04785v1)

Authors: Alisa Welter, Niklas Schneider, Tobias Dick, Kallistos Weis, Christof Tinnes, Marvin Wyrich, Sven Apel

Knowledge transfer is fundamental to human collaboration and is therefore
common in software engineering. Pair programming is a prominent instance. With
the rise of AI coding assistants, developers now not only work with human
partners but also, as some claim, with AI pair programmers. Although studies
confirm knowledge transfer during human pair programming, its effectiveness
with AI coding assistants remains uncertain. To analyze knowledge transfer in
both human-human and human-AI settings, we conducted an empirical study where
developer pairs solved a programming task without AI support, while a separate
group of individual developers completed the same task using the AI coding
assistant GitHub Copilot. We extended an existing knowledge transfer framework
and employed a semi-automated evaluation pipeline to assess differences in
knowledge transfer episodes across both settings. We found a similar frequency
of successful knowledge transfer episodes and overlapping topical categories
across both settings. Two of our key findings are that developers tend to
accept GitHub Copilot's suggestions with less scrutiny than those from human
pair programming partners, but also that GitHub Copilot can subtly remind
developers of important code details they might otherwise overlook.

### 4. [BacPrep: An Experimental Platform for Evaluating LLM-Based Bacalaureat Assessment](http://arxiv.org/pdf/2506.04989v1)

Authors: Dumitran Adrian Marius, Dita Radu

Accessing quality preparation and feedback for the Romanian Bacalaureat exam
is challenging, particularly for students in remote or underserved areas. This
paper introduces BacPrep, an experimental online platform exploring Large
Language Model (LLM) potential for automated assessment, aiming to offer a
free, accessible resource. Using official exam questions from the last 5 years,
BacPrep employs one of Google's newest models, Gemini 2.0 Flash (released Feb
2025), guided by official grading schemes, to provide experimental feedback.
Currently operational, its primary research function is collecting student
solutions and LLM outputs. This focused dataset is vital for planned expert
validation to rigorously evaluate the feasibility and accuracy of this
cutting-edge LLM in the specific Bacalaureat context before reliable
deployment. We detail the design, data strategy, status, validation plan, and
ethics.

### 5. [LLM-Guided Scenario-based GUI Testing](http://arxiv.org/pdf/2506.05079v1)

Authors: Shengcheng Yu, Yuchen Ling, Chunrong Fang, Quan Zhou, Chunyang Chen, Shaomin Zhu, Zhenyu Chen

The assurance of mobile app GUI is more and more significant. Automated GUI
testing approaches of different strategies have been developed, while there are
still huge gaps between the approaches and the app business logic, not taking
the completion of specific testing scenarios as the exploration target, leading
to the exploration missing of critical app functionalities. Learning from the
manual testing, which takes testing scenarios with app business logic as the
basic granularity, in this paper, we utilize the LLMs to understand the
semantics presented in app GUI and how they are mapped in the testing context
based on specific testing scenarios. Then, scenario-based GUI tests are
generated with the guidance of multi-agent collaboration. Specifically, we
propose ScenGen, a novel LLM-guided scenario-based GUI testing approach
involving five agents to respectively take responsibilities of different phases
of the manual testing process. The Observer perceives the app GUI state by
extracting GUI widgets and forming GUI layouts, understanding the expressed
semantics. Then the app GUI info is sent to the Decider to make decisions on
target widgets based on the target testing scenarios. The decision-making
process takes the completion of specific testing scenarios as the exploration
target. The Executor then executes the demanding operations on the apps. The
execution results are checked by the Supervisor on whether the generated tests
are consistent with the completion target of the testing scenarios, ensuring
the traceability of the test generation and execution. Furthermore, the
corresponding GUI test operations are recorded to the context memory by
Recorder as an important basis for further decision-making, meanwhile
monitoring the runtime bug occurrences. ScenGen is evaluated and the results
show that ScenGen can effectively generate scenario-based GUI tests guided by
LLMs.

### 6. [PoCGen: Generating Proof-of-Concept Exploits for Vulnerabilities in Npm Packages](http://arxiv.org/pdf/2506.04962v1)

Authors: Deniz Simsek, Aryaz Eghbali, Michael Pradel

Security vulnerabilities in software packages are a significant concern for
developers and users alike. Patching these vulnerabilities in a timely manner
is crucial to restoring the integrity and security of software systems.
However, previous work has shown that vulnerability reports often lack
proof-of-concept (PoC) exploits, which are essential for fixing the
vulnerability, testing patches, and avoiding regressions. Creating a PoC
exploit is challenging because vulnerability reports are informal and often
incomplete, and because it requires a detailed understanding of how inputs
passed to potentially vulnerable APIs may reach security-relevant sinks. In
this paper, we present PoCGen, a novel approach to autonomously generate and
validate PoC exploits for vulnerabilities in npm packages. This is the first
fully autonomous approach to use large language models (LLMs) in tandem with
static and dynamic analysis techniques for PoC exploit generation. PoCGen
leverages an LLM for understanding vulnerability reports, for generating
candidate PoC exploits, and for validating and refining them. Our approach
successfully generates exploits for 77% of the vulnerabilities in the
SecBench.js dataset and 39% in a new, more challenging dataset of 794 recent
vulnerabilities. This success rate significantly outperforms a recent baseline
(by 45 absolute percentage points), while imposing an average cost of $0.02 per
generated exploit.

### 7. [A Multi-Dataset Evaluation of Models for Automated Vulnerability Repair](http://arxiv.org/pdf/2506.04987v1)

Authors: Zanis Ali Khan, Aayush Garg, Qiang Tang

Software vulnerabilities pose significant security threats, requiring
effective mitigation. While Automated Program Repair (APR) has advanced in
fixing general bugs, vulnerability patching, a security-critical aspect of APR
remains underexplored. This study investigates pre-trained language models,
CodeBERT and CodeT5, for automated vulnerability patching across six datasets
and four languages. We evaluate their accuracy and generalization to unknown
vulnerabilities. Results show that while both models face challenges with
fragmented or sparse context, CodeBERT performs comparatively better in such
scenarios, whereas CodeT5 excels in capturing complex vulnerability patterns.
CodeT5 also demonstrates superior scalability. Furthermore, we test fine-tuned
models on both in-distribution (trained) and out-of-distribution (unseen)
datasets. While fine-tuning improves in-distribution performance, models
struggle to generalize to unseen data, highlighting challenges in robust
vulnerability detection. This study benchmarks model performance, identifies
limitations in generalization, and provides actionable insights to advance
automated vulnerability patching for real-world security applications.

### 8. [Tech-ASan: Two-stage check for Address Sanitizer](http://arxiv.org/pdf/2506.05022v1)

Authors: Yixuan Cao, Yuhong Feng, Huafeng Li, Chongyi Huang, Fangcao Jian, Haoran Li, Xu Wang

Address Sanitizer (ASan) is a sharp weapon for detecting memory safety
violations, including temporal and spatial errors hidden in C/C++ programs
during execution. However, ASan incurs significant runtime overhead, which
limits its efficiency in testing large software. The overhead mainly comes from
sanitizer checks due to the frequent and expensive shadow memory access. Over
the past decade, many methods have been developed to speed up ASan by
eliminating and accelerating sanitizer checks, however, they either fail to
adequately eliminate redundant checks or compromise detection capabilities. To
address this issue, this paper presents Tech-ASan, a two-stage check based
technique to accelerate ASan with safety assurance. First, we propose a novel
two-stage check algorithm for ASan, which leverages magic value comparison to
reduce most of the costly shadow memory accesses. Second, we design an
efficient optimizer to eliminate redundant checks, which integrates a novel
algorithm for removing checks in loops. Third, we implement Tech-ASan as a
memory safety tool based on the LLVM compiler infrastructure. Our evaluation
using the SPEC CPU2006 benchmark shows that Tech-ASan outperforms the
state-of-the-art methods with 33.70% and 17.89% less runtime overhead than ASan
and ASan--, respectively. Moreover, Tech-ASan detects 56 fewer false negative
cases than ASan and ASan-- when testing on the Juliet Test Suite under the same
redzone setting.

### 9. [Tensor-based multivariate function approximation: methods benchmarking and comparison](http://arxiv.org/pdf/2506.04791v1)

Authors: Athanasios C. Antoulas, Ion Victor Gosea, Charles Poussot-Vassal, Pierre Vuillemin

In this note, we evaluate the performances, the features and the
user-experience of some methods (and their implementations) designed for
tensor- (or data-) based multivariate function construction and approximation.
To this aim, a collection of multivariate functions extracted from contributive
works coming from different communities, is suggested. First, these functions
with varying complexity (e.g. number and degree of the variables) and nature
(e.g. rational, irrational, differentiable or not, symmetric, etc.) are used to
construct tensors, each of different dimension and size on the disk. Second,
grounded on this tensor, we inspect performances of each considered method
(e.g. the accuracy, the computational time, the parameters tuning impact,
etc.). Finally, considering the "best" parameter tuning set, we compare each
method using multiple evaluation criteria. The purpose of this note is not to
rank the methods but rather to evaluate as fairly as possible the different
available strategies, with the idea in mind to guide users to understand the
process, the possibilities, the advantages and the limits brought by each
tools. The contribution claimed is to suggest a complete benchmark collection
of some available tools for tensor approximation by surrogate models (e.g.
rational functions, networks, etc.). In addition, as contributors of the
multivariate Loewner Framework (mLF) approach (and its side implementation in
MDSPACK), attention and details of the latter are more explicitly given, in
order to provide readers a digest of this contributive work and some details
with simple examples.

### Social and Information Networks

### 1. [User Altruism in Recommendation Systems](http://arxiv.org/pdf/2506.04525v1)

Authors: Ekaterina Fedorova, Madeline Kitch, Chara Podimata

Users of social media platforms based on recommendation systems (RecSys)
(e.g. TikTok, X, YouTube) strategically interact with platform content to
influence future recommendations. On some such platforms, users have been
documented to form large-scale grassroots movements encouraging others to
purposefully interact with algorithmically suppressed content in order to
"boost" its recommendation; we term this behavior user altruism. To capture
this behavior, we study a game between users and a RecSys, where users provide
the RecSys (potentially manipulated) preferences over the contents available to
them, and the RecSys -- limited by data and computation constraints -- creates
a low-rank approximation preference matrix, and ultimately provides each user
her (approximately) most-preferred item. We compare the users' social welfare
under truthful preference reporting and under a class of strategies capturing
user altruism. In our theoretical analysis, we provide sufficient conditions to
ensure strict increases in user social welfare under user altruism, and provide
an algorithm to find an effective altruistic strategy. Interestingly, we show
that for commonly assumed recommender utility functions, effectively altruistic
strategies also improve the utility of the RecSys! We show that our results are
robust to several model misspecifications, thus strengthening our conclusions.
Our theoretical analysis is complemented by empirical results of effective
altruistic strategies on the GoodReads dataset, and an online survey on how
real-world users behave altruistically in RecSys. Overall, our findings serve
as a proof-of-concept of the reasons why traditional RecSys may incentivize
users to form collectives and/or follow altruistic strategies when interacting
with them.

### 2. [Memory-Driven Bounded Confidence Opinion Dynamics: A Hegselmann-Krause Model Based on Fractional-Order Methods](http://arxiv.org/pdf/2506.04701v1)

Authors: Meiru Jiang, Wei Su, Guojian Ren, Yongguang Yu

Memory effects play a crucial role in social interactions and decision-making
processes. This paper proposes a novel fractional-order bounded confidence
opinion dynamics model to characterize the memory effects in system states.
Building upon the Hegselmann-Krause framework and fractional-order difference,
a comprehensive model is established that captures the persistent influence of
historical information. Through rigorous theoretical analysis, the fundamental
properties including convergence and consensus is investigated. The results
demonstrate that the proposed model not only maintains favorable convergence
and consensus characteristics compared to classical opinion dynamics, but also
addresses limitations such as the monotonicity of bounded opinions. This
enables a more realistic representation of opinion evolution in real-world
scenarios. The findings of this study provide new insights and methodological
approaches for understanding opinion formation and evolution, offering both
theoretical significance and practical applications.

### Systems and Control

### 1. [Distribution System State and Impedance Estimation Augmented with Carson's Equations](http://arxiv.org/pdf/2506.04949v1)

Authors: Marta Vanin, Frederik Geth, Rahmat Heidari, Dirk Van Hertem

The impedances of cables and lines used in (multi-conductor) distribution
networks are usually unknown or approximated, and may lead to problematic
results for any physics-based power system calculation, e.g., (optimal) power
flow. Learning parameters from time series data is one of the few available
options to obtain improved impedance models. This paper presents an approach
that combines statistical learning concepts with the exploitation of domain
knowledge, in the form of Carson's equations, through nonlinear mathematical
optimization. The proposed approach derives impedance matrices for
up-to-four-wire systems, using measurement data like those obtained from smart
meters. Despite the lack of phasor measurements, the low signal-to-noise ratio
of smart meter measurements, and the inherent existence of multiple equivalent
solutions, our method produces good quality impedance models that are fit for
power system calculations, significantly improving on our previous work both in
terms of accuracy and computational time.

### 2. [En Route Path-planning for Partially Occupied Vehicles in Ride-pooling Systems](http://arxiv.org/pdf/2506.04968v1)

Authors: Pengbo Zhu, Giancarlo Ferrari-Trecate, Nikolas Geroliminis

Ride-pooling services, such as UberPool and Lyft Shared Saver, enable a
single vehicle to serve multiple customers within one shared trip. Efficient
path-planning algorithms are crucial for improving the performance of such
systems. For partially occupied vehicles with available capacity, we introduce
a novel routing algorithm designed to maximize the likelihood of picking up
additional passengers while serving the current passengers to their
destination. Unlike traditional methods that group passengers and vehicles
based on predefined time windows, our algorithm allows for immediate responses
to passenger requests. Our approach optimizes travel time while dynamically
considering passenger demand and coordinating with other vehicles. Formulated
as an integer linear programming (ILP) problem, our method is computationally
efficient and suitable for real-time applications. Simulation results
demonstrate that our proposed method can significantly enhance service quality.

### 3. [Cloud-Based Interoperability in Residential Energy Systems](http://arxiv.org/pdf/2506.05076v1)

Authors: Darren Leniston, David Ryan, Ammar Malik, Jack Jackman, Terence O'Donnell

As distributed energy resources (DERs) such as solar PV, batteries and
electric vehicles become increasingly prevalent at the edge, maintaining grid
stability requires advanced monitoring and control mechanisms. This paper
presents a scalable smart grid gateway architecture that enables
interoperability between Modbus-based inverters and IEEE 2030.5 cloud-based
control systems. The proposed solution leverages Azure cloud services and
edge-computing gateway devices to support dynamic configuration, telemetry
ingestion, remote control and Volt-VAR Curve deployment. A microservice-based
architecture ensures flexibility and scalability across diverse deployment
scenarios, including both gateway-mediated and direct-to-cloud device
communication. Results demonstrate the successful mapping of a Fronius Primo
inverter's Modbus registers to IEEE 2030.5-compliant telemetry and control
functions. Additionally, we evaluate real-time VVC updates and their impact on
local voltage regulation, showcasing dynamic cloud-to-edge control with minimal
latency. This work highlights the potential of virtualised, standards-based
control infrastructures to support DER integration and active grid
participation, while remaining adaptable to evolving smart grid architectures.

### 4. [Towards provable probabilistic safety for scalable embodied AI systems](http://arxiv.org/pdf/2506.05171v1)

Authors: Linxuan He, Qing-Shan Jia, Ang Li, Hongyan Sang, Ling Wang, Jiwen Lu, Tao Zhang, Jie Zhou, Yi Zhang, Yisen Wang, Peng Wei, Zhongyuan Wang, Henry X. Liu, Shuo Feng

Embodied AI systems, comprising AI models and physical plants, are
increasingly prevalent across various applications. Due to the rarity of system
failures, ensuring their safety in complex operating environments remains a
major challenge, which severely hinders their large-scale deployment in
safety-critical domains, such as autonomous vehicles, medical devices, and
robotics. While achieving provable deterministic safety--verifying system
safety across all possible scenarios--remains theoretically ideal, the rarity
and complexity of corner cases make this approach impractical for scalable
embodied AI systems. To address this challenge, we introduce provable
probabilistic safety, which aims to ensure that the residual risk of
large-scale deployment remains below a predefined threshold. Instead of
attempting exhaustive safety proof across all corner cases, this paradigm
establishes a probabilistic safety boundary on overall system performance,
leveraging statistical methods to enhance feasibility and scalability. A
well-defined probabilistic safety boundary enables embodied AI systems to be
deployed at scale while allowing for continuous refinement of safety
guarantees. Our work focuses on three core questions: what is provable
probabilistic safety, how to prove the probabilistic safety, and how to achieve
the provable probabilistic safety. By bridging the gap between theoretical
safety assurance and practical deployment, our work offers a pathway toward
safer, large-scale adoption of embodied AI systems in safety-critical
applications.

### 5. [Real-Time LPV-Based Non-Linear Model Predictive Control for Robust Trajectory Tracking in Autonomous Vehicles](http://arxiv.org/pdf/2506.04684v1)

Authors: Nitish Kumar, Rajalakshmi Pachamuthu

This paper presents the development and implementation of a Model Predictive
Control (MPC) framework for trajectory tracking in autonomous vehicles under
diverse driving conditions. The proposed approach incorporates a modular
architecture that integrates state estimation, vehicle dynamics modeling, and
optimization to ensure real-time performance. The state-space equations are
formulated in a Linear Parameter Varying (LPV) form, and a curvature-based
tuning method is introduced to optimize weight matrices for varying
trajectories. The MPC framework is implemented using the Robot Operating System
(ROS) for parallel execution of state estimation and control optimization,
ensuring scalability and minimal latency. Extensive simulations and real-time
experiments were conducted on multiple predefined trajectories, demonstrating
high accuracy with minimal cross-track and orientation errors, even under
aggressive maneuvers and high-speed conditions. The results highlight the
robustness and adaptability of the proposed system, achieving seamless
alignment between simulated and real-world performance. This work lays the
foundation for dynamic weight tuning and integration into cooperative
autonomous navigation systems, paving the way for enhanced safety and
efficiency in autonomous driving applications.

### 6. [Bilevel Optimization for Improved Flexibility Aggregation Models of Electric Vehicle Fleets](http://arxiv.org/pdf/2506.04843v1)

Authors: Philipp Härtel, Michael von Bonin

Electric vehicle (EV) fleets are expected to become an increasingly important
source of flexibility for power system operations. However, accurately
capturing the flexibility potential of numerous and heterogeneous EVs remains a
significant challenge. We propose a bilevel optimization formulation to enhance
flexibility aggregations of electric vehicle fleets. The outer level minimizes
scheduling deviations between the aggregated and reference EV units, while the
inner level maximizes the aggregated unit's profits. Our approach introduces
hourly to daily scaling factor mappings to parameterize the aggregated EV
units. Compared to simple aggregation methods, the proposed framework reduces
the root-mean-square error of charging power by 78~per cent, providing more
accurate flexibility representations. The proposed framework also provides a
foundation for several potential extensions in future work.

### 7. [Observations on robust diffusive stability and common Lyapunov functions](http://arxiv.org/pdf/2506.04863v1)

Authors: Blake McGrane-Corrigan, Rafael de Andrade Moral, Oliver Mason

We consider the problem of robust diffusive stability (RDS) for a pair of
Schur-stable nonnegative matrices. Specifically, we show that the existence of
a common diagonal Lyapunov function is sufficient for RDS and highlight how
this condition differs from recently published results based on linear
copositive Lyapunov functions. We also present two results on RDS for extended
Leslie matrices arising in population dynamics.

### 8. [Efficient Path Planning and Task Allocation Algorithm for Boolean Specifications](http://arxiv.org/pdf/2506.04881v1)

Authors: Ioana Hustiu, Roozbeh Abolpour, Cristian Mahulea, Marius Kloetzer

This paper presents a novel path-planning and task assignment algorithm for
multi-robot systems that should fulfill a global Boolean specification. The
proposed method is based on Integer Linear Programming (ILP) formulations,
which are combined with structural insights from Petri nets to improve
scalability and computational efficiency. By proving that the \emph{constraint
matrix} is totally unimodular (TU) for certain classes of problems, the ILP
formulation can be relaxed into a Linear Programming (LP) problem without
losing the integrality of the solution. This relaxation eliminates complex
combinatorial techniques, significantly reducing computational overhead and
thus ensuring scalability for large-scale systems. Using the approach proposed
in this paper, we can solve path-planning problems for teams made up to 500
robots. The method guarantees computational tractability, handles collision
avoidance and reduces computational demands through iterative LP optimization
techniques. Case studies demonstrate the efficiency of the algorithm in
generating scalable, collision-free paths for large robot teams navigating in
complex environments. While the conservative nature of collision avoidance
introduces additional constraints, and thus, computational requirements, the
solution remains practical and impactful for diverse applications. The
algorithm is particularly applicable to real-world scenarios, including
warehouse logistics where autonomous robots must efficiently coordinate tasks
or search-and-rescue operations in various environments. This work contributes
both theoretically and practically to scalable multi-robot path planning and
task allocation, offering an efficient framework for coordinating autonomous
agents in shared environments.

### 9. [Energy-Optimized Scheduling for AIoT Workloads Using TOPSIS](http://arxiv.org/pdf/2506.04902v1)

Authors: Preethika Pradeep, Eyhab Al-Masri

AIoT workloads demand energy-efficient orchestration across cloud-edge
infrastructures, but Kubernetes' default scheduler lacks multi-criteria
optimization for heterogeneous environments. This paper presents GreenPod, a
TOPSIS-based scheduler optimizing pod placement based on execution time, energy
consumption, processing core, memory availability, and resource balance. Tested
on a heterogeneous Google Kubernetes cluster, GreenPod improves energy
efficiency by up to 39.1% over the default Kubernetes (K8s) scheduler,
particularly with energy-centric weighting schemes. Medium complexity workloads
showed the highest energy savings, despite slight scheduling latency. GreenPod
effectively balances sustainability and performance for AIoT applications.

### 10. [Energentic Intelligence: From Self-Sustaining Systems to Enduring Artificial Life](http://arxiv.org/pdf/2506.04916v1)

Authors: Atahan Karagoz

This paper introduces Energentic Intelligence, a class of autonomous systems
defined not by task performance, but by their capacity to sustain themselves
through internal energy regulation. Departing from conventional reward-driven
paradigms, these agents treat survival-maintaining functional operation under
fluctuating energetic and thermal conditions-as the central objective. We
formalize this principle through an energy-based utility function and a
viability-constrained survival horizon, and propose a modular architecture that
integrates energy harvesting, thermal regulation, and adaptive computation into
a closed-loop control system. A simulated environment demonstrates the
emergence of stable, resource-aware behavior without external supervision.
Together, these contributions provide a theoretical and architectural
foundation for deploying autonomous agents in resource-volatile settings where
persistence must be self-regulated and infrastructure cannot be assumed.

### Machine Learning (Statistics Category)

### 1. [Regret-Optimal Q-Learning with Low Cost for Single-Agent and Federated Reinforcement Learning](http://arxiv.org/pdf/2506.04626v1)

Authors: Haochen Zhang, Zhong Zheng, Lingzhou Xue

Motivated by real-world settings where data collection and policy deployment
-- whether for a single agent or across multiple agents -- are costly, we study
the problem of on-policy single-agent reinforcement learning (RL) and federated
RL (FRL) with a focus on minimizing burn-in costs (the sample sizes needed to
reach near-optimal regret) and policy switching or communication costs. In
parallel finite-horizon episodic Markov Decision Processes (MDPs) with $S$
states and $A$ actions, existing methods either require superlinear burn-in
costs in $S$ and $A$ or fail to achieve logarithmic switching or communication
costs. We propose two novel model-free RL algorithms -- Q-EarlySettled-LowCost
and FedQ-EarlySettled-LowCost -- that are the first in the literature to
simultaneously achieve: (i) the best near-optimal regret among all known
model-free RL or FRL algorithms, (ii) low burn-in cost that scales linearly
with $S$ and $A$, and (iii) logarithmic policy switching cost for single-agent
RL or communication cost for FRL. Additionally, we establish gap-dependent
theoretical guarantees for both regret and switching/communication costs,
improving or matching the best-known gap-dependent bounds.

### 2. [Distributional encoding for Gaussian process regression with qualitative inputs](http://arxiv.org/pdf/2506.04813v1)

Authors: Sébastien Da Veiga

Gaussian Process (GP) regression is a popular and sample-efficient approach
for many engineering applications, where observations are expensive to acquire,
and is also a central ingredient of Bayesian optimization (BO), a highly
prevailing method for the optimization of black-box functions. However, when
all or some input variables are categorical, building a predictive and
computationally efficient GP remains challenging. Starting from the naive
target encoding idea, where the original categorical values are replaced with
the mean of the target variable for that category, we propose a generalization
based on distributional encoding (DE) which makes use of all samples of the
target variable for a category. To handle this type of encoding inside the GP,
we build upon recent results on characteristic kernels for probability
distributions, based on the maximum mean discrepancy and the Wasserstein
distance. We also discuss several extensions for classification, multi-task
learning and incorporation or auxiliary information. Our approach is validated
empirically, and we demonstrate state-of-the-art predictive performance on a
variety of synthetic and real-world datasets. DE is naturally complementary to
recent advances in BO over discrete and mixed-spaces.

### 3. [Learning Joint Interventional Effects from Single-Variable Interventions in Additive Models](http://arxiv.org/pdf/2506.04945v1)

Authors: Armin Kekić, Sergio Hernan Garrido Mejia, Bernhard Schölkopf

Estimating causal effects of joint interventions on multiple variables is
crucial in many domains, but obtaining data from such simultaneous
interventions can be challenging. Our study explores how to learn joint
interventional effects using only observational data and single-variable
interventions. We present an identifiability result for this problem, showing
that for a class of nonlinear additive outcome mechanisms, joint effects can be
inferred without access to joint interventional data. We propose a practical
estimator that decomposes the causal effect into confounded and unconfounded
contributions for each intervention variable. Experiments on synthetic data
demonstrate that our method achieves performance comparable to models trained
directly on joint interventional data, outperforming a purely observational
estimator.

### 4. [Unregularized limit of stochastic gradient method for Wasserstein distributionally robust optimization](http://arxiv.org/pdf/2506.04948v1)

Authors: Tam Le

Distributionally robust optimization offers a compelling framework for model
fitting in machine learning, as it systematically accounts for data
uncertainty. Focusing on Wasserstein distributionally robust optimization, we
investigate the regularized problem where entropic smoothing yields a
sampling-based approximation of the original objective. We establish the
convergence of the approximate gradient over a compact set, leading to the
concentration of the regularized problem critical points onto the original
problem critical set as regularization diminishes and the number of
approximation samples increases. Finally, we deduce convergence guarantees for
a projected stochastic gradient method. Our analysis covers a general machine
learning situation with an unbounded sample space and mixed continuous-discrete
data.

### 5. [NIMO: a Nonlinear Interpretable MOdel](http://arxiv.org/pdf/2506.05059v1)

Authors: Shijian Xu, Marcello Massimo Negri, Volker Roth

Neural networks (NNs) have achieved tremendous success over the past decade,
yet they are still extremely difficult to interpret. In contrast, linear models
are less expressive but offer inherent interpretability. Linear coefficients
are interpretable as the marginal effect of a feature on the prediction,
assuming all other features are kept fixed. To combine the benefits of both
approaches, we introduce NIMO (Nonlinear Interpretable MOdel). The key idea is
to define a model where the NN is designed to learn nonlinear corrections to
the linear model predictions, while also maintaining the original
interpretability of the linear coefficients. Relevantly, we develop an
optimization algorithm based on profile likelihood that elegantly allows for
optimizing over the NN parameters while updating the linear coefficients
analytically. By relying on adaptive ridge regression we can easily incorporate
sparsity constraints as well. We show empirically that we can recover the
underlying linear coefficients while significantly improving the predictive
accuracy. Compared to other hybrid interpretable approaches, our model is the
only one that actually maintains the same interpretability of linear
coefficients as in linear models. We also achieve higher performance on various
regression and classification settings.

### 6. [UnHiPPO: Uncertainty-aware Initialization for State Space Models](http://arxiv.org/pdf/2506.05065v1)

Authors: Marten Lienen, Abdullah Saydemir, Stephan Günnemann

State space models are emerging as a dominant model class for sequence
problems with many relying on the HiPPO framework to initialize their dynamics.
However, HiPPO fundamentally assumes data to be noise-free; an assumption often
violated in practice. We extend the HiPPO theory with measurement noise and
derive an uncertainty-aware initialization for state space model dynamics. In
our analysis, we interpret HiPPO as a linear stochastic control problem where
the data enters as a noise-free control signal. We then reformulate the problem
so that the data become noisy outputs of a latent system and arrive at an
alternative dynamics initialization that infers the posterior of this latent
system from the data without increasing runtime. Our experiments show that our
initialization improves the resistance of state-space models to noise both at
training and inference time. Find our implementation at
https://cs.cit.tum.de/daml/unhippo.

### 7. [Progressive Tempering Sampler with Diffusion](http://arxiv.org/pdf/2506.05231v1)

Authors: Severi Rissanen, RuiKang OuYang, Jiajun He, Wenlin Chen, Markus Heinonen, Arno Solin, José Miguel Hernández-Lobato

Recent research has focused on designing neural samplers that amortize the
process of sampling from unnormalized densities. However, despite significant
advancements, they still fall short of the state-of-the-art MCMC approach,
Parallel Tempering (PT), when it comes to the efficiency of target evaluations.
On the other hand, unlike a well-trained neural sampler, PT yields only
dependent samples and needs to be rerun -- at considerable computational cost
-- whenever new samples are required. To address these weaknesses, we propose
the Progressive Tempering Sampler with Diffusion (PTSD), which trains diffusion
models sequentially across temperatures, leveraging the advantages of PT to
improve the training of neural samplers. We also introduce a novel method to
combine high-temperature diffusion models to generate approximate
lower-temperature samples, which are minimally refined using MCMC and used to
train the next diffusion model. PTSD enables efficient reuse of sample
information across temperature levels while generating well-mixed, uncorrelated
samples. Our method significantly improves target evaluation efficiency,
outperforming diffusion-based neural samplers.

### 8. [Unsupervised Machine Learning for Scientific Discovery: Workflow and Best Practices](http://arxiv.org/pdf/2506.04553v1)

Authors: Andersen Chang, Tiffany M. Tang, Tarek M. Zikry, Genevera I. Allen

Unsupervised machine learning is widely used to mine large, unlabeled
datasets to make data-driven discoveries in critical domains such as climate
science, biomedicine, astronomy, chemistry, and more. However, despite its
widespread utilization, there is a lack of standardization in unsupervised
learning workflows for making reliable and reproducible scientific discoveries.
In this paper, we present a structured workflow for using unsupervised learning
techniques in science. We highlight and discuss best practices starting with
formulating validatable scientific questions, conducting robust data
preparation and exploration, using a range of modeling techniques, performing
rigorous validation by evaluating the stability and generalizability of
unsupervised learning conclusions, and promoting effective communication and
documentation of results to ensure reproducible scientific discoveries. To
illustrate our proposed workflow, we present a case study from astronomy,
seeking to refine globular clusters of Milky Way stars based upon their
chemical composition. Our case study highlights the importance of validation
and illustrates how the benefits of a carefully-designed workflow for
unsupervised learning can advance scientific discovery.

### 9. [Subjective Perspectives within Learned Representations Predict High-Impact Innovation](http://arxiv.org/pdf/2506.04616v1)

Authors: Likun Cao, Rui Pan, James Evans

Existing studies of innovation emphasize the power of social structures to
shape innovation capacity. Emerging machine learning approaches, however,
enable us to model innovators' personal perspectives and interpersonal
innovation opportunities as a function of their prior trajectories of
experience. We theorize then quantify subjective perspectives and innovation
opportunities based on innovator positions within the geometric space of
concepts inscribed by dynamic language representations. Using data on millions
of scientists, inventors, writers, entrepreneurs, and Wikipedia contributors
across the creative domains of science, technology, film, entrepreneurship, and
Wikipedia, here we show that measured subjective perspectives anticipate what
ideas individuals and groups creatively attend to and successfully combine in
future. When perspective and background diversity are decomposed as the angular
difference between collaborators' perspectives on their creation and between
their experiences, the former consistently anticipates creative achievement
while the latter portends its opposite, across all cases and time periods
examined. We analyze a natural experiment and simulate creative collaborations
between AI (large language model) agents designed with various perspective and
background diversity, which are consistent with our observational findings. We
explore mechanisms underlying these findings and identify how successful
collaborators leverage common language to weave together diverse experience
obtained through trajectories of prior work that converge to provoke one
another and innovate. We explore the importance of these findings for team
assembly and research policy.

### 10. [On the Mechanism of Reasoning Pattern Selection in Reinforcement Learning for Language Models](http://arxiv.org/pdf/2506.04695v1)

Authors: Xingwu Chen, Tianle Li, Difan Zou

Reinforcement learning (RL) has demonstrated remarkable success in enhancing
model capabilities, including instruction-following, preference learning, and
reasoning. Yet despite its empirical successes, the mechanisms by which RL
improves reasoning abilities remain poorly understood. We present a systematic
study of Reinforcement Learning with Verifiable Rewards (RLVR), showing that
its primary benefit comes from optimizing the selection of existing reasoning
patterns. Through extensive experiments, we demonstrate that RLVR-trained
models preferentially adopt high-success-rate reasoning patterns while mostly
maintaining stable performance on individual patterns. We further develop
theoretical analyses on the convergence and training dynamics of RLVR based on
a simplified question-reason-answer model. We study the gradient flow and show
that RLVR can indeed find the solution that selects the reason pattern with the
highest success rate. Besides, our theoretical results
  reveal two distinct regimes regarding the convergence of RLVR training: (1)
rapid convergence for models with relatively strong initial reasoning
capabilities versus (2) slower optimization dynamics for weaker models.
Furthermore, we show that the slower optimization for weaker models can be
mitigated by applying the supervised fine-tuning (SFT) before RLVR, when using
a feasibly high-quality SFT dataset. We validate the theoretical findings
through extensive experiments. This work advances our theoretical understanding
of RL's role in LLM fine-tuning and offers insights for further enhancing
reasoning capabilities.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

### 1. [UANV: UNet-based attention network for thoracolumbar vertebral compression fracture angle measurement](https://www.nature.com/articles/s41598-025-03514-6)

Authors: Yurim Lee et al.

### 2. [Emotion recognition with multiple physiological parameters based on ensemble learning](https://www.nature.com/articles/s41598-025-96616-0)

Authors: Yilong Liao et al.

### 3. [Music informer as an efficient model for music generation](https://www.nature.com/articles/s41598-025-02792-4)

Authors: Hui Sun et al.

### 4. [The outcome prediction method of football matches by the quantum neural network based on deep learning](https://www.nature.com/articles/s41598-025-91870-8)

Authors: Yang Sun et al.

### 5. [Modeling eye gaze velocity trajectories using GANs with spectral loss for enhanced fidelity](https://www.nature.com/articles/s41598-025-05286-5)

Authors: Shailendra Bhandari et al.

