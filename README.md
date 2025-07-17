# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-16 17:00:26.546340 PST.

### Artificial Intelligence

### 1. [Lessons Learned from Evaluation of LLM based Multi-agents in Safer Therapy Recommendation](http://arxiv.org/pdf/2507.10911v1)

Authors: Yicong Wu, Ting Chen, Irit Hochberg, Zhoujian Sun, Ruth Edry, Zhengxing Huang, Mor Peleg

Therapy recommendation for chronic patients with multimorbidity is
challenging due to risks of treatment conflicts. Existing decision support
systems face scalability limitations. Inspired by the way in which general
practitioners (GP) manage multimorbidity patients, occasionally convening
multidisciplinary team (MDT) collaboration, this study investigated the
feasibility and value of using a Large Language Model (LLM)-based multi-agent
system (MAS) for safer therapy recommendations. We designed a single agent and
a MAS framework simulating MDT decision-making by enabling discussion among LLM
agents to resolve medical conflicts. The systems were evaluated on therapy
planning tasks for multimorbidity patients using benchmark cases. We compared
MAS performance with single-agent approaches and real-world benchmarks. An
important contribution of our study is the definition of evaluation metrics
that go beyond the technical precision and recall and allow the inspection of
clinical goals met and medication burden of the proposed advices to a gold
standard benchmark. Our results show that with current LLMs, a single agent GP
performs as well as MDTs. The best-scoring models provide correct
recommendations that address all clinical goals, yet the advices are
incomplete. Some models also present unnecessary medications, resulting in
unnecessary conflicts between medication and conditions or drug-drug
interactions.

### 2. [Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](http://arxiv.org/pdf/2507.10923v1)

Authors: Yuhao Wang, Keyan Ding, Kehua Feng, Zeyuan Wang, Ming Qin, Xiaotong Li, Qiang Zhang, Huajun Chen

Protein language models have emerged as powerful tools for sequence
generation, offering substantial advantages in functional optimization and
denovo design. However, these models also present significant risks of
generating harmful protein sequences, such as those that enhance viral
transmissibility or evade immune responses. These concerns underscore critical
biosafety and ethical challenges. To address these issues, we propose a
Knowledge-guided Preference Optimization (KPO) framework that integrates prior
knowledge via a Protein Safety Knowledge Graph. This framework utilizes an
efficient graph pruning strategy to identify preferred sequences and employs
reinforcement learning to minimize the risk of generating harmful proteins.
Experimental results demonstrate that KPO effectively reduces the likelihood of
producing hazardous sequences while maintaining high functionality, offering a
robust safety assurance framework for applying generative models in
biotechnology.

### 3. [Modeling Habitat Shifts: Integrating Convolutional Neural Networks and Tabular Data for Species Migration Prediction](http://arxiv.org/pdf/2507.10993v1)

Authors: Emir Durakovic, Min-Hong Shih

Due to climate-induced changes, many habitats are experiencing range shifts
away from their traditional geographic locations (Piguet, 2011). We propose a
solution to accurately model whether bird species are present in a specific
habitat through the combination of Convolutional Neural Networks (CNNs)
(O'Shea, 2015) and tabular data. Our approach makes use of satellite imagery
and environmental features (e.g., temperature, precipitation, elevation) to
predict bird presence across various climates. The CNN model captures spatial
characteristics of landscapes such as forestation, water bodies, and
urbanization, whereas the tabular method uses ecological and geographic data.
Both systems predict the distribution of birds with an average accuracy of 85%,
offering a scalable but reliable method to understand bird migration.

### 4. [Personalized Exercise Recommendation with Semantically-Grounded Knowledge Tracing](http://arxiv.org/pdf/2507.11060v1)

Authors: Yilmazcan Ozyurt, Tunaberk Almaci, Stefan Feuerriegel, Mrinmaya Sachan

We introduce ExRec, a general framework for personalized exercise
recommendation with semantically-grounded knowledge tracing. Our method builds
on the observation that existing exercise recommendation approaches simulate
student performance via knowledge tracing (KT) but they often overlook two key
aspects: (a) the semantic content of questions and (b) the sequential,
structured progression of student learning. To address this, our ExRec presents
an end-to-end pipeline, from annotating the KCs of questions and learning their
semantic representations to training KT models and optimizing several
reinforcement learning (RL) methods. Moreover, we improve standard
Q-learning-based continuous RL methods via a tailored model-based value
estimation (MVE) approach that directly leverages the components of KT model in
estimating cumulative knowledge improvement. We validate the effectiveness of
our ExRec using various RL methods across four real-world tasks with different
educational goals in online math learning. We further show that ExRec
generalizes robustly to new, unseen questions and that it produces
interpretable student learning trajectories. Together, our findings highlight
the promise of KT-guided RL for effective personalization in education.

### 5. [Tactical Decision for Multi-UGV Confrontation with a Vision-Language Model-Based Commander](http://arxiv.org/pdf/2507.11079v1)

Authors: Li Wang, Qizhen Wu, Lei Chen

In multiple unmanned ground vehicle confrontations, autonomously evolving
multi-agent tactical decisions from situational awareness remain a significant
challenge. Traditional handcraft rule-based methods become vulnerable in the
complicated and transient battlefield environment, and current reinforcement
learning methods mainly focus on action manipulation instead of strategic
decisions due to lack of interpretability. Here, we propose a vision-language
model-based commander to address the issue of intelligent
perception-to-decision reasoning in autonomous confrontations. Our method
integrates a vision language model for scene understanding and a lightweight
large language model for strategic reasoning, achieving unified perception and
decision within a shared semantic space, with strong adaptability and
interpretability. Unlike rule-based search and reinforcement learning methods,
the combination of the two modules establishes a full-chain process, reflecting
the cognitive process of human commanders. Simulation and ablation experiments
validate that the proposed approach achieves a win rate of over 80% compared
with baseline models.

### 6. [AI Agent Architecture for Decentralized Trading of Alternative Assets](http://arxiv.org/pdf/2507.11117v1)

Authors: Ailiya Borjigin, Cong He, Charles CC Lee, Wei Zhou

Decentralized trading of real-world alternative assets (e.g., gold) requires
bridging physical asset custody with blockchain systems while meeting strict
requirements for compliance, liquidity, and risk management. We present
GoldMine OS, a research oriented architecture that employs multiple specialized
AI agents to automate and secure the tokenization and exchange of physical gold
into a blockchain based stablecoin ("OZ"). Our approach combines on chain smart
contracts for critical risk controls with off chain AI agents for decision
making, blending the transparency and reliability of blockchains with the
flexibility of AI driven automation. We describe four cooperative agents
(Compliance, Token Issuance, Market Making, and Risk Control) and a
coordinating core, and evaluate the system through simulation and a controlled
pilot deployment. In experiments the prototype delivers on demand token
issuance in under 1.2 s, more than 100 times faster than manual workflows. The
Market Making agent maintains tight liquidity with spreads often below 0.5
percent even under volatile conditions. Fault injection tests show resilience:
an oracle price spoofing attack is detected and mitigated within 10 s, and a
simulated vault mis reporting halts issuance immediately with minimal user
impact. The architecture scales to 5000 transactions per second with 10000
concurrent users in benchmarks. These results indicate that an AI agent based
decentralized exchange for alternative assets can satisfy rigorous performance
and safety requirements. We discuss broader implications for democratizing
access to traditionally illiquid assets and explain how our governance model --
multi signature agent updates and on chain community voting on risk parameters
-- provides ongoing transparency, adaptability, and formal assurance of system
integrity.

### 7. [Defining neurosymbolic AI](http://arxiv.org/pdf/2507.11127v1)

Authors: Lennert De Smet, Luc De Raedt

Neurosymbolic AI focuses on integrating learning and reasoning, in
particular, on unifying logical and neural representations. Despite the
existence of an alphabet soup of neurosymbolic AI systems, the field is lacking
a generally accepted formal definition of what neurosymbolic models and
inference really are. We introduce a formal definition for neurosymbolic AI
that makes abstraction of its key ingredients. More specifically, we define
neurosymbolic inference as the computation of an integral over a product of a
logical and a belief function. We show that our neurosymbolic AI definition
makes abstraction of key representative neurosymbolic AI systems.

### 8. [Collaborative Trustworthiness for Good Decision Making in Autonomous Systems](http://arxiv.org/pdf/2507.11135v1)

Authors: Selma Saidi, Omar Laimona, Christoph Schmickler, Dirk Ziegenbein

Autonomous systems are becoming an integral part of many application domains,
like in the mobility sector. However, ensuring their safe and correct behaviour
in dynamic and complex environments remains a significant challenge, where
systems should autonomously make decisions e.g., about manoeuvring. We propose
in this paper a general collaborative approach for increasing the level of
trustworthiness in the environment of operation and improve reliability and
good decision making in autonomous system. In the presence of conflicting
information, aggregation becomes a major issue for trustworthy decision making
based on collaborative data sharing. Unlike classical approaches in the
literature that rely on consensus or majority as aggregation rule, we exploit
the fact that autonomous systems have different quality attributes like
perception quality. We use this criteria to determine which autonomous systems
are trustworthy and borrow concepts from social epistemology to define
aggregation and propagation rules, used for automated decision making. We use
Binary Decision Diagrams (BDDs) as formal models for beliefs aggregation and
propagation, and formulate reduction rules to reduce the size of the BDDs and
allow efficient computation structures for collaborative automated reasoning.

### 9. [Opus: A Prompt Intention Framework for Complex Workflow Generation](http://arxiv.org/pdf/2507.11288v1)

Authors: Théo Fagnoni, Mahsun Altin, Chia En Chung, Phillip Kingston, Alan Tuning, Dana O. Mohamed, Inès Adnani

This paper introduces the Opus Prompt Intention Framework, designed to
improve complex Workflow Generation with instruction-tuned Large Language
Models (LLMs). We propose an intermediate Intention Capture layer between user
queries and Workflow Generation, implementing the Opus Workflow Intention
Framework, which consists of extracting Workflow Signals from user queries,
interpreting them into structured Workflow Intention objects, and generating
Workflows based on these Intentions. Our results show that this layer enables
LLMs to produce logical and meaningful outputs that scale reliably as query
complexity increases. On a synthetic benchmark of 1,000 multi-intent
query-Workflow(s) pairs, applying the Opus Prompt Intention Framework to
Workflow Generation yields consistent improvements in semantic Workflow
similarity metrics. In this paper, we introduce the Opus Prompt Intention
Framework by applying the concepts of Workflow Signal and Workflow Intention to
LLM-driven Workflow Generation. We present a reproducible, customizable
LLM-based Intention Capture system to extract Workflow Signals and Workflow
Intentions from user queries. Finally, we provide empirical evidence that the
proposed system significantly improves Workflow Generation quality compared to
direct generation from user queries, particularly in cases of Mixed Intention
Elicitation.

### 10. [Contestability in Quantitative Argumentation](http://arxiv.org/pdf/2507.11323v1)

Authors: Xiang Yin, Nico Potyka, Antonio Rago, Timotheus Kampik, Francesca Toni

Contestable AI requires that AI-driven decisions align with human
preferences. While various forms of argumentation have been shown to support
contestability, Edge-Weighted Quantitative Bipolar Argumentation Frameworks
(EW-QBAFs) have received little attention. In this work, we show how EW-QBAFs
can be deployed for this purpose. Specifically, we introduce the contestability
problem for EW-QBAFs, which asks how to modify edge weights (e.g., preferences)
to achieve a desired strength for a specific argument of interest (i.e., a
topic argument). To address this problem, we propose gradient-based relation
attribution explanations (G-RAEs), which quantify the sensitivity of the topic
argument's strength to changes in individual edge weights, thus providing
interpretable guidance for weight adjustments towards contestability. Building
on G-RAEs, we develop an iterative algorithm that progressively adjusts the
edge weights to attain the desired strength. We evaluate our approach
experimentally on synthetic EW-QBAFs that simulate the structural
characteristics of personalised recommender systems and multi-layer
perceptrons, and demonstrate that it can solve the problem effectively.

### Hardware Architecture

### 1. [Mapping Fusion: Improving FPGA Technology Mapping with ASIC Mapper](http://arxiv.org/pdf/2507.10912v1)

Authors: Cunxi Yu

LUT (Look-Up Table) mapping is a critical step in FPGA logic synthesis, where
a logic network is transformed into a form that can be directly implemented
using the FPGA's LUTs. An FPGA LUT is a flexible digital memory structure that
can implement any logic function of a limited number of inputs, typically 4 to
6 inputs, depending on the FPGA architecture. The goal of LUT mapping is to map
the Boolean network into LUTs, where each LUT can implement any function with a
fixed number of inputs. In parallel to FPGA technology mapping, ASIC technology
mapping maps the Boolean network to user-defined standard cells, which has
traditionally been developed separately from LUT mapping algorithms. However,
in this work, our motivating examples demonstrate that ASIC technology mappers
can potentially improve the performance of LUT mappers, such that standard cell
mapping and LUT mapping work in an incremental manner.
  Therefore, we propose the FuseMap framework, which explores this opportunity
to improve LUT mapping in the FPGA design flow by utilizing reinforcement
learning to make design-specific choices during cell selection. The
effectiveness of FuseMap is evaluated on a wide range of benchmarks, different
technology libraries, and technology mappers. The experimental results
demonstrate that FuseMap achieves higher mapping accuracy while reducing delay
and area across diverse circuit designs collected from ISCAS 85/89, ITC/ISCAS
99, VTR 8.0, and EPFL benchmarks.

### 2. [Security Enclave Architecture for Heterogeneous Security Primitives for Supply-Chain Attacks](http://arxiv.org/pdf/2507.10971v1)

Authors: Kshitij Raj, Atri Chatterjee, Patanjali SLPSK, Swarup Bhunia, Sandip Ray

Designing secure architectures for system-on-chip (SoC) platforms is a highly
intricate and time-intensive task, often requiring months of development and
meticulous verification. Even minor architectural oversights can lead to
critical vulnerabilities that undermine the security of the entire chip. In
response to this challenge, we introduce CITADEL, a modular security framework
aimed at streamlining the creation of robust security architectures for SoCs.
CITADEL offers a configurable, plug-and-play subsystem composed of custom
intellectual property (IP) blocks, enabling the construction of diverse
security mechanisms tailored to specific threats. As a concrete demonstration,
we instantiate CITADEL to defend against supply-chain threats, illustrating how
the framework adapts to one of the most pressing concerns in hardware security.
This paper explores the range of obstacles encountered when building a unified
security architecture capable of addressing multiple attack vectors and
presents CITADEL's strategies for overcoming them. Through several real-world
case studies, we showcase the practical implementation of CITADEL and present a
thorough evaluation of its impact on silicon area and power consumption across
various ASIC technologies. Results indicate that CITADEL introduces only
minimal resource overhead, making it a practical solution for enhancing SoC
security.

### 3. [Fault-Free Analog Computing with Imperfect Hardware](http://arxiv.org/pdf/2507.11134v1)

Authors: Zhicheng Xu, Jiawei Liu, Sitao Huang, Zefan Li, Shengbo Wang, Bo Wen, Ruibin Mao, Mingrui Jiang, Giacomo Pedretti, Jim Ignowski, Kaibin Huang, Can Li

The growing demand for edge computing and AI drives research into analog
in-memory computing using memristors, which overcome data movement bottlenecks
by computing directly within memory. However, device failures and variations
critically limit analog systems' precision and reliability. Existing
fault-tolerance techniques, such as redundancy and retraining, are often
inadequate for high-precision applications or scenarios requiring fixed
matrices and privacy preservation. Here, we introduce and experimentally
demonstrate a fault-free matrix representation where target matrices are
decomposed into products of two adjustable sub-matrices programmed onto analog
hardware. This indirect, adaptive representation enables mathematical
optimization to bypass faulty devices and eliminate differential pairs,
significantly enhancing computational density. Our memristor-based system
achieved >99.999% cosine similarity for a Discrete Fourier Transform matrix
despite 39% device fault rate, a fidelity unattainable with conventional direct
representation, which fails with single device faults (0.01% rate). We
demonstrated 56-fold bit-error-rate reduction in wireless communication and
>196% density with 179% energy efficiency improvements compared to
state-of-the-art techniques. This method, validated on memristors, applies
broadly to emerging memories and non-electrical computing substrates, showing
that device yield is no longer the primary bottleneck in analog computing
hardware.

### 4. [SystolicAttention: Fusing FlashAttention within a Single Systolic Array](http://arxiv.org/pdf/2507.11331v1)

Authors: Jiawei Lin, Guokai Chen, Yuanlong Li, Thomas Bourgeat

Transformer models rely heavily on scaled dot-product attention (SDPA),
typically implemented using the FlashAttention algorithm. However, current
systolic-array-based accelerators face significant challenges when executing
FlashAttention. Systolic arrays can only achieve high utilization for
consecutive and large matrix multiplications. In contrast, FlashAttention
requires frequently interleaved matrix multiplications and softmax operations.
  The frequent data swaps between the systolic array and external vector units
result in low systolic array utilization. This is further exacerbated by the
fact that softmax involves numerous non-matrix operations, which are not
well-suited for systolic arrays. Moreover, the concurrent execution of matrix
multiplication on systolic arrays and softmax on vector units leads to register
file and SRAM port contention, further degrading performance.
  To overcome these limitations, we propose FSA, an enhanced systolic array
architecture that enables the entire FlashAttention algorithm to run entirely
within a single systolic array, eliminating the need for external vector units.
At the core of FSA is SystolicAttention, a novel scheduling algorithm that maps
FlashAttention operations onto systolic arrays with fine-grained, element-wise
overlap. This significantly improves array utilization while preserving the
original floating-point operation order to maintain numerical stability.
  We implement FSA in synthesizable RTL and evaluate its performance against
state-of-the-art commercial accelerators. Our results show that FSA achieves
1.77x and 4.83x higher attention FLOPs/s utilization compared to AWS
NeuronCore-v2 and Google TPUv5e, respectively, with only about 10% area
overhead.

### 5. [Elk: Exploring the Efficiency of Inter-core Connected AI Chips with Deep Learning Compiler Techniques](http://arxiv.org/pdf/2507.11506v1)

Authors: Yiqi Liu, Yuqi Xue, Noelle Crawford, Jilong Xue, Jian Huang

To meet the increasing demand of deep learning (DL) models, AI chips are
employing both off-chip memory (e.g., HBM) and high-bandwidth low-latency
interconnect for direct inter-core data exchange. However, it is not easy to
explore the efficiency of these inter-core connected AI (ICCA) chips, due to a
fundamental tussle among compute (per-core execution), communication
(inter-core data exchange), and I/O (off-chip data access).
  In this paper, we develop Elk, a DL compiler framework to maximize the
efficiency of ICCA chips by jointly trading off all the three performance
factors discussed above. Elk structures these performance factors into
configurable parameters and forms a global trade-off space in the DL compiler.
To systematically explore this space and maximize overall efficiency, Elk
employs a new inductive operator scheduling policy and a cost-aware on-chip
memory allocation algorithm. It generates globally optimized execution plans
that best overlap off-chip data loading and on-chip execution. To examine the
efficiency of Elk, we build a full-fledged emulator based on a real ICCA chip
IPU-POD4, and an ICCA chip simulator for sensitivity analysis with different
interconnect network topologies. Elk achieves 94% of the ideal roofline
performance of ICCA chips on average, showing the benefits of supporting large
DL models on ICCA chips. We also show Elk's capability of enabling architecture
design space exploration for new ICCA chip development.

### Computational Complexity

### 1. [Equality is Far Weaker than Constant-Cost Communication](http://arxiv.org/pdf/2507.11162v1)

Authors: Mika Göös, Nathaniel Harms, Artur Riazanov

We exhibit an $n$-bit communication problem with a constant-cost randomized
protocol but which requires $n^{\Omega(1)}$ deterministic (or even
non-deterministic) queries to an Equality oracle. Therefore, even constant-cost
randomized protocols cannot be efficiently "derandomized" using Equality
oracles. This improves on several recent results and answers a question from
the survey of Hatami and Hatami (SIGACT News 2024). It also gives a
significantly simpler and quantitatively superior proof of the main result of
Fang, G\"o\"os, Harms, and Hatami ( STOC 2025), that constant-cost
communication does not reduce to the $k$-Hamming Distance hierarchy.

### 2. [FPT Parameterisations of Fractional and Generalised Hypertree Width](http://arxiv.org/pdf/2507.11080v1)

Authors: Matthias Lanzinger, Igor Razgon, Daniel Unterberger

We present the first fixed-parameter tractable (fpt) algorithms for precisely
determining several central hypergraph decomposition parameters, including
generalized hypertree width, fractional hypertree width, and adaptive width.
Despite the recognized importance of these measures in complexity theory,
databases, and constraint satisfaction, no exact fpt algorithms for any of them
had previously been known. Our results are obtained for hypergraph classes of
bounded rank and bounded degree.
  Our approach extends a recent algorithm for treewidth (Boja\'ncyk &
Pilipczuk, LMCS 2022) utilizing monadic second-order (MSO) transductions.
Leveraging this framework, we overcome the significant technical hurdles
presented by hypergraphs, whose structural decompositions are technically much
more intricate than their graph counterparts.

### 3. [On the Complexity of the Skolem Problem at Low Orders](http://arxiv.org/pdf/2507.11234v1)

Authors: Piotr Bacik, Joël Ouaknine, James Worrell

The Skolem Problem asks to determine whether a given linear recurrence
sequence (LRS) $\langle u_n \rangle_{n=0}^\infty$ over the integers has a zero
term, that is, whether there exists $n$ such that $u_n = 0$. Decidability of
the problem is open in general, with the most notable positive result being a
decision procedure for LRS of order at most 4.
  In this paper we consider a bounded version of the Skolem Problem, in which
the input consists of an LRS $\langle u_n \rangle_{n=0}^\infty$ and a bound $N
\in \mathbb N$ (with all integers written in binary), and the task is to
determine whether there exists $n\in\{0,\ldots,N\}$ such that $u_n=0$. We give
a randomised algorithm for this problem that, for all $d\in \mathbb N$, runs in
polynomial time on the class of LRS of order at most $d$. As a corollary we
show that the (unrestricted) Skolem Problem for LRS of order at most 4 lies in
$\mathsf{coRP}$, improving the best previous upper bound of
$\mathsf{NP}^{\mathsf{RP}}$.
  The running time of our algorithm is exponential in the order of the LRS -- a
dependence that appears necessary in view of the $\mathsf{NP}$-hardness of the
Bounded Skolem Problem. However, even for LRS of a fixed order, the problem
involves detecting zeros within an exponentially large range. For this, our
algorithm relies on results from $p$-adic analysis to isolate polynomially many
candidate zeros and then test in randomised polynomial time whether each
candidate is an actual zero by reduction to arithmetic-circuit identity
testing.

### 4. [On the Complexity of the Optimal Correlated Equilibria in Extensive-Form Games](http://arxiv.org/pdf/2507.11509v1)

Authors: Vincent Cheval, Florian Horn, Soumyajit Paul, Mahsa Shirmohammadi

A major open question in algorithmic game theory is whether normal-form
correlated equilibria (NFCE) can be computed efficiently in succinct games such
as extensive-form games [DFF+25,6PR24,FP23,HvS08,VSF08,PR08]. Motivated by this
question, we study the associated Threshold problem: deciding whether there
exists a correlated equilibrium whose value exceeds a given threshold. We prove
that this problem is PSPACE-hard for NFCE in multiplayer extensive-form games
with perfect recall, even for fixed thresholds. To contextualize this result,
we also establish the complexity of the Threshold problem for Nash equilibria
in this setting, showing it is ER-complete. These results uncover a surprising
complexity reversal: while optimal correlated equilibria are computationally
simpler than optimal Nash in normal-form games, the opposite holds in
extensive-form games, where computing optimal correlated equilibria is provably
harder. Building on this line of inquiry, we also address a related question by
[VSF08], who introduced the notions of extensive-form correlated equilibrium
(EFCE) and agent-form correlated equilibrium (AFCE). They asked how difficult
the Threshold problem is for AFCE; we answer this question by proving that it
is NP-hard, even in two-player games without chance nodes. Complementing our
hardness results, we establish tight complexity classifications for the
Threshold problem across several correlated equilibrium concepts - including
EFCE, AFCE, normal-form coarse, extensive-form coarse, and agent-form coarse
correlated equilibria. For each of these solution concepts in multiplayer
stochastic extensive-form games with perfect recall, we prove NP-completeness
by providing matching NP upper bounds to the previously known hardness results.
Together, our results provide the most complete landscape to date for the
complexity of optimal equilibrium computation in extensive-form games.

### Computational Engineering

### 1. [The Multiple Time-Stepping Method for 3-Body Interactions in High Performance Molecular Dynamics Simulations](http://arxiv.org/pdf/2507.11172v1)

Authors: David Martin, Samuel James Newcome, Markus Mühlhäußer, Manish Kumar Mishra, Fabio Alexander Gratl, Hans-Joachim Bungartz

Understanding the complex behavior of molecular systems is fundamental to
fields such as physics, materials science, and biology. Molecular dynamics (MD)
simulations are crucial tools for studying atomic-level dynamics. This work
focuses on improving the efficiency of MD simulations involving two-body and
three-body interactions. Traditional two-body potentials often can not fully
capture the complexity of molecular systems, making the inclusion of three-body
interactions important. However, these interactions are in a cubic complexity
class, compared to a quadratic one for two-body interactions, and therefore are
computationally expensive, even when a cutoff distance is applied. One way to
improve efficiency is to use the r-RESPA multiple time-stepping algorithm to
reduce the number of three-body interaction calculations. In this work, we
investigate this method in the context of High Performance Computing (HPC)
methods that parallelize the calculations. In particular, we investigate a
communication-reducing distributed-memory parallel method from literature and
present a novel shared-memory parallel cutoff method, implemented in the
particle simulation library AutoPas. The results and methods are discussed,
providing insights into potential advancements in MD simulation efficiency.

### 2. [Data-Driven Differential Evolution in Tire Industry Extrusion: Leveraging Surrogate Models](http://arxiv.org/pdf/2507.11191v1)

Authors: Eider Garate-Perez, Kerman López de Calle-Etxabe, Susana Ferreiro

The optimization of industrial processes remains a critical challenge,
particularly when no mathematical formulation of objective functions or
constraints is available. This study addresses this issue by proposing a
surrogate-based, data-driven methodology for optimizing complex real-world
manufacturing systems using only historical process data. Machine learning
models are employed to approximate system behavior and construct surrogate
models, which are integrated into a tailored metaheuristic approach:
Data-Driven Differential Evolution with Multi-Level Penalty Functions and
Surrogate Models, an adapted version of Differential Evolution suited to the
characteristics of the studied process. The methodology is applied to an
extrusion process in the tire manufacturing industry, with the goal of
optimizing initialization parameters to reduce waste and production time.
Results show that the surrogate-based optimization approach outperforms
historical best configurations, achieving a 65\% reduction in initialization
and setup time, while also significantly minimizing material waste. These
findings highlight the potential of combining data-driven modeling and
metaheuristic optimization for industrial processes where explicit formulations
are unavailable.

### 3. [DrafterBench: Benchmarking Large Language Models for Tasks Automation in Civil Engineering](http://arxiv.org/pdf/2507.11527v1)

Authors: Yinsheng Li, Zhen Dong, Yi Shao

Large Language Model (LLM) agents have shown great potential for solving
real-world problems and promise to be a solution for tasks automation in
industry. However, more benchmarks are needed to systematically evaluate
automation agents from an industrial perspective, for example, in Civil
Engineering. Therefore, we propose DrafterBench for the comprehensive
evaluation of LLM agents in the context of technical drawing revision, a
representation task in civil engineering. DrafterBench contains twelve types of
tasks summarized from real-world drawing files, with 46 customized
functions/tools and 1920 tasks in total. DrafterBench is an open-source
benchmark to rigorously test AI agents' proficiency in interpreting intricate
and long-context instructions, leveraging prior knowledge, and adapting to
dynamic instruction quality via implicit policy awareness. The toolkit
comprehensively assesses distinct capabilities in structured data
comprehension, function execution, instruction following, and critical
reasoning. DrafterBench offers detailed analysis of task accuracy and error
statistics, aiming to provide deeper insight into agent capabilities and
identify improvement targets for integrating LLMs in engineering applications.
Our benchmark is available at https://github.com/Eason-Li-AIS/DrafterBench,
with the test set hosted at
https://huggingface.co/datasets/Eason666/DrafterBench.

### Computational Geometry

### 1. [Bicriteria Polygon Aggregation with Arbitrary Shapes](http://arxiv.org/pdf/2507.11212v1)

Authors: Lotte Blank, David Eppstein, Jan-Henrik Haunert, Herman Haverkort, Benedikt Kolbe, Philip Mayer, Petra Mutzel, Alexander Naumann, Jonas Sauer

We study the problem of aggregating polygons by covering them with disjoint
representative regions, thereby inducing a clustering of the polygons. Our
objective is to minimize a weighted sum of the total area and the total
perimeter of the regions. This problem has applications in cartographic map
generalization and urban analytics. Here, the polygons represent building
footprints and the clusters may represent urban areas. Previous approaches
forced the boundaries of the regions to come from a fixed subdivision of the
plane, which allows the optimal solution (restricted in this way) to be found
from a minimum cut in a dual graph. It is natural to ask whether the problem
can still be solved efficiently if this restriction is removed, allowing output
regions to be bounded by arbitrary curves. We provide a positive answer in the
form of a polynomial-time algorithm. Additionally, we fully characterize the
optimal solutions by showing that their boundaries are composed of input
polygon edges and circular arcs of constant radius. Since some applications
favor straight edges, we also study two problem variants in which the output
regions must be polygons, but are not restricted to have boundaries from a
fixed subdivision. In the first variant, region vertices must lie on the
boundaries of the input polygons. The second variant requires them to be
vertices of the input polygons. We show that both variants can be approximated
up to a constant factor in polynomial time by altering an optimal solution for
the unrestricted problem. Our experimental evaluation on real-world building
footprints demonstrates that these approximate solutions are visually similar
to the optimal unrestricted ones and achieve near-optimal objective values.

### 2. [On Tight Robust Coresets for $k$-Medians Clustering](http://arxiv.org/pdf/2507.11260v1)

Authors: Lingxiao Huang, Zhenyu Jiang, Yi Li, Xuan Wu

This paper considers coresets for the robust $k$-medians problem with $m$
outliers, and new constructions in various metric spaces are obtained.
Specifically, for metric spaces with a bounded VC or doubling dimension $d$,
the coreset size is $O(m) + \tilde{O}(kd\varepsilon^{-2})$, which is optimal up
to logarithmic factors. For Euclidean spaces, the coreset size is
$O(m\varepsilon^{-1}) +
\tilde{O}(\min\{k^{4/3}\varepsilon^{-2},k\varepsilon^{-3}\})$, improving upon a
recent result by Jiang and Lou (ICALP 2025). These results also extend to
robust $(k,z)$-clustering, yielding, for VC and doubling dimension, a coreset
size of $O(m) + \tilde{O}(kd\varepsilon^{-2z})$ with the optimal linear
dependence on $m$. This extended result improves upon the earlier work of Huang
et al. (SODA 2025). The techniques introduce novel dataset decompositions,
enabling chaining arguments to be applied jointly across multiple components.

### 3. [Tileable Surfaces](http://arxiv.org/pdf/2507.11281v1)

Authors: David Brander, Jens Gravesen

We define a class of $C^k$-regular surfaces, $k \geq 1$, \emph{tileable
surfaces}, that admit geometric tilings by a finite number of congruence
classes of tiles. We show how to construct many examples, and examine the
relationship with the well known tilings of the plane and sphere, as well as
monohedral polyhedral surfaces.

### 4. [Compressed data structures for Heegaard splittings](http://arxiv.org/pdf/2507.11406v1)

Authors: Henrique Ennes, Clément Maria

Heegaard splittings provide a natural representation of closed 3-manifolds by
gluing handlebodies along a common surface. These splittings can be
equivalently given by two finite sets of meridians lying in the surface, which
define a Heegaard diagram. We present a data structure to effectively represent
Heegaard diagrams as normal curves with respect to triangulations of a surface
of complexity measured by the space required to express the normal coordinates'
vectors in binary. This structure can be significantly more compressed than
triangulations of 3-manifolds, given exponential gains for some families. Even
with this succinct definition of complexity, we establish polynomial time
algorithms for comparing and manipulating diagrams, performing stabilizations,
detecting trivial stabilizations and reductions, and computing topological
invariants of the underlying manifolds, such as their fundamental and first
homology groups. We also contrast early implementations of our techniques with
standard software programs for 3-manifolds, achieving better precision and
faster algorithms for the average cases and exponential gains in speed for some
particular presentations of the inputs.

### Computation and Language

### 1. [How Stylistic Similarity Shapes Preferences in Dialogue Dataset with User and Third Party Evaluations](http://arxiv.org/pdf/2507.10918v1)

Authors: Ikumi Numaya, Shoji Moriya, Shiki Sato, Reina Akama, Jun Suzuki

Recent advancements in dialogue generation have broadened the scope of
human-bot interactions, enabling not only contextually appropriate responses
but also the analysis of human affect and sensitivity. While prior work has
suggested that stylistic similarity between user and system may enhance user
impressions, the distinction between subjective and objective similarity is
often overlooked. To investigate this issue, we introduce a novel dataset that
includes users' preferences, subjective stylistic similarity based on users'
own perceptions, and objective stylistic similarity annotated by third party
evaluators in open-domain dialogue settings. Analysis using the constructed
dataset reveals a strong positive correlation between subjective stylistic
similarity and user preference. Furthermore, our analysis suggests an important
finding: users' subjective stylistic similarity differs from third party
objective similarity. This underscores the importance of distinguishing between
subjective and objective evaluations and understanding the distinct aspects
each captures when analyzing the relationship between stylistic similarity and
user preferences. The dataset presented in this paper is available online.

### 2. [DS@GT at eRisk 2025: From prompts to predictions, benchmarking early depression detection with conversational agent based assessments and temporal attention models](http://arxiv.org/pdf/2507.10958v1)

Authors: Anthony Miyaguchi, David Guecha, Yuwen Chiu, Sidharth Gaur

This Working Note summarizes the participation of the DS@GT team in two eRisk
2025 challenges. For the Pilot Task on conversational depression detection with
large language-models (LLMs), we adopted a prompt-engineering strategy in which
diverse LLMs conducted BDI-II-based assessments and produced structured JSON
outputs. Because ground-truth labels were unavailable, we evaluated cross-model
agreement and internal consistency. Our prompt design methodology aligned model
outputs with BDI-II criteria and enabled the analysis of conversational cues
that influenced the prediction of symptoms. Our best submission, second on the
official leaderboard, achieved DCHR = 0.50, ADODL = 0.89, and ASHR = 0.27.

### 3. [Mario at EXIST 2025: A Simple Gateway to Effective Multilingual Sexism Detection](http://arxiv.org/pdf/2507.10996v1)

Authors: Lin Tian, Johanne R. Trippas, Marian-Andrei Rizoiu

This paper presents our approach to EXIST 2025 Task 1, addressing text-based
sexism detection in English and Spanish tweets through hierarchical Low-Rank
Adaptation (LoRA) of Llama 3.1 8B. Our method introduces conditional adapter
routing that explicitly models label dependencies across three hierarchically
structured subtasks: binary sexism identification, source intention detection,
and multilabel sexism categorization. Unlike conventional LoRA applications
that target only attention layers, we apply adaptation to all linear
transformations, enhancing the model's capacity to capture task-specific
patterns. In contrast to complex data processing and ensemble approaches, we
show that straightforward parameter-efficient fine-tuning achieves strong
performance. We train separate LoRA adapters (rank=16, QLoRA 4-bit) for each
subtask using unified multilingual training that leverages Llama 3.1's native
bilingual capabilities. The method requires minimal preprocessing and uses
standard supervised learning. Our multilingual training strategy eliminates the
need for separate language-specific models, achieving 1.7-2.4\% F1 improvements
through cross-lingual transfer. With only 1.67\% trainable parameters compared
to full fine-tuning, our approach reduces training time by 75\% and model
storage by 98\%, while achieving competitive performance across all subtasks
(ICM-Hard: 0.6774 for binary classification, 0.4991 for intention detection,
0.6519 for multilabel categorization).

### 4. [Team HUMANE at AVeriTeC 2025: HerO 2 for Efficient Fact Verification](http://arxiv.org/pdf/2507.11004v1)

Authors: Yejun Yoon, Jaeyoon Jung, Seunghyun Yoon, Kunwoo Park

This paper presents HerO 2, Team HUMANE's system for the AVeriTeC shared task
at the FEVER-25 workshop. HerO 2 is an enhanced version of HerO, the
best-performing open-source model from the previous year's challenge. It
improves evidence quality through document summarization and answer
reformulation, optimizes veracity prediction via post-training quantization
under computational constraints, and enhances overall system performance by
integrating updated language model (LM) backbones. HerO 2 ranked second on the
leaderboard while achieving the shortest runtime among the top three systems,
demonstrating both high efficiency and strong potential for real-world fact
verification. The code is available at https://github.com/ssu-humane/HerO2.

### 5. [Journalism-Guided Agentic In-Context Learning for News Stance Detection](http://arxiv.org/pdf/2507.11049v1)

Authors: Dahyun Lee, Jonghyeon Choi, Jiyoung Han, Kunwoo Park

As online news consumption grows, personalized recommendation systems have
become integral to digital journalism. However, these systems risk reinforcing
filter bubbles and political polarization by failing to incorporate diverse
perspectives. Stance detection -- identifying a text's position on a target --
can help mitigate this by enabling viewpoint-aware recommendations and
data-driven analyses of media bias. Yet, existing stance detection research
remains largely limited to short texts and high-resource languages. To address
these gaps, we introduce \textsc{K-News-Stance}, the first Korean dataset for
article-level stance detection, comprising 2,000 news articles with
article-level and 19,650 segment-level stance annotations across 47 societal
issues. We also propose \textsc{JoA-ICL}, a \textbf{Jo}urnalism-guided
\textbf{A}gentic \textbf{I}n-\textbf{C}ontext \textbf{L}earning framework that
employs a language model agent to predict the stances of key structural
segments (e.g., leads, quotes), which are then aggregated to infer the overall
article stance. Experiments show that \textsc{JoA-ICL} outperforms existing
stance detection methods, highlighting the benefits of segment-level agency in
capturing the overall position of long-form news articles. Two case studies
further demonstrate its broader utility in promoting viewpoint diversity in
news recommendations and uncovering patterns of media bias.

### 6. [Social Media Sentiments Analysis on the July Revolution in Bangladesh: A Hybrid Transformer Based Machine Learning Approach](http://arxiv.org/pdf/2507.11084v1)

Authors: Md. Sabbir Hossen, Md. Saiduzzaman, Pabon Shaha

The July Revolution in Bangladesh marked a significant student-led mass
uprising, uniting people across the nation to demand justice, accountability,
and systemic reform. Social media platforms played a pivotal role in amplifying
public sentiment and shaping discourse during this historic mass uprising. In
this study, we present a hybrid transformer-based sentiment analysis framework
to decode public opinion expressed in social media comments during and after
the revolution. We used a brand new dataset of 4,200 Bangla comments collected
from social media. The framework employs advanced transformer-based feature
extraction techniques, including BanglaBERT, mBERT, XLM-RoBERTa, and the
proposed hybrid XMB-BERT, to capture nuanced patterns in textual data.
Principle Component Analysis (PCA) were utilized for dimensionality reduction
to enhance computational efficiency. We explored eleven traditional and
advanced machine learning classifiers for identifying sentiments. The proposed
hybrid XMB-BERT with the voting classifier achieved an exceptional accuracy of
83.7% and outperform other model classifier combinations. This study
underscores the potential of machine learning techniques to analyze social
sentiment in low-resource languages like Bangla.

### 7. [Beyond Traditional Algorithms: Leveraging LLMs for Accurate Cross-Border Entity Identification](http://arxiv.org/pdf/2507.11086v1)

Authors: Andres Azqueta-Gavaldón, Joaquin Ramos Cosgrove

The growing prevalence of cross-border financial activities in global markets
has underscored the necessity of accurately identifying and classifying foreign
entities. This practice is essential within the Spanish financial system for
ensuring robust risk management, regulatory adherence, and the prevention of
financial misconduct. This process involves a labor-intensive entity-matching
task, where entities need to be validated against available reference sources.
Challenges arise from linguistic variations, special characters, outdated
names, and changes in legal forms, complicating traditional matching algorithms
like Jaccard, cosine, and Levenshtein distances. These methods struggle with
contextual nuances and semantic relationships, leading to mismatches. To
address these limitations, we explore Large Language Models (LLMs) as a
flexible alternative. LLMs leverage extensive training to interpret context,
handle abbreviations, and adapt to legal transitions. We evaluate traditional
methods, Hugging Face-based LLMs, and interface-based LLMs (e.g., Microsoft
Copilot, Alibaba's Qwen 2.5) using a dataset of 65 Portuguese company cases.
Results show traditional methods achieve accuracies over 92% but suffer high
false positive rates (20-40%). Interface-based LLMs outperform, achieving
accuracies above 93%, F1 scores exceeding 96%, and lower false positives
(40-80%).

### 8. [The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs](http://arxiv.org/pdf/2507.11097v1)

Authors: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang

Diffusion-based large language models (dLLMs) have recently emerged as a
powerful alternative to autoregressive LLMs, offering faster inference and
greater interactivity via parallel decoding and bidirectional modeling.
However, despite strong performance in code generation and text infilling, we
identify a fundamental safety concern: existing alignment mechanisms fail to
safeguard dLLMs against context-aware, masked-input adversarial prompts,
exposing novel vulnerabilities. To this end, we present DIJA, the first
systematic study and jailbreak attack framework that exploits unique safety
weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial
interleaved mask-text prompts that exploit the text generation mechanisms of
dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional
modeling drives the model to produce contextually consistent outputs for masked
spans, even when harmful, while parallel decoding limits model dynamic
filtering and rejection sampling of unsafe content. This causes standard
alignment mechanisms to fail, enabling harmful completions in alignment-tuned
dLLMs, even when harmful behaviors or unsafe instructions are directly exposed
in the prompt. Through comprehensive experiments, we demonstrate that DIJA
significantly outperforms existing jailbreak methods, exposing a previously
overlooked threat surface in dLLM architectures. Notably, our method achieves
up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior
baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and
by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of
harmful content in the jailbreak prompt. Our findings underscore the urgent
need for rethinking safety alignment in this emerging class of language models.
Code is available at https://github.com/ZichenWen1/DIJA.

### 9. [MSA at ImageCLEF 2025 Multimodal Reasoning: Multilingual Multimodal Reasoning With Ensemble Vision Language Models](http://arxiv.org/pdf/2507.11114v1)

Authors: Seif Ahmed, Mohamed T. Younes, Abdelrahman Moustafa, Abdelrahman Allam, Hamza Moustafa

We present a robust ensemble-based system for multilingual multimodal
reasoning, designed for the ImageCLEF 2025 EXAMS V challenge. Our approach
integrates Gemini 2.5 Flash for visual description, Gemini 1.5 Pro for caption
refinement and consistency checks, and Gemini 2.5 Pro as a reasoner which
handles final answer selection, all coordinated through carefully engineered
few-shot and zero-shot prompts. We conducted an extensive ablation study,
training several large language models (Gemini 2.5 Flash, Phi 4, Gemma 3,
Mistral) on an English dataset and its multilingual augmented version.
Additionally, we evaluated Gemini 2.5 Flash in a zero-shot setting for
comparison and found it to substantially outperform the trained models. Prompt
design also proved critical: enforcing concise, language-normalized formats and
prohibiting explanatory text boosted model accuracy on the English validation
set from 55.9% to 61.7%. On the official leaderboard, our system (Team MSA)
achieved first place overall in the multilingual track with 81.4% accuracy, and
led 11 out of 13 individual language tracks, with top results such as 95.07%
for Croatian and 92.12% for Italian. These findings highlight that lightweight
OCR-VLM ensembles, when paired with precise prompt strategies and cross-lingual
augmentation, can outperform heavier end-to-end models in high-stakes,
multilingual educational settings.

### 10. [EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering](http://arxiv.org/pdf/2507.11216v1)

Authors: Valle Ruiz-Fernández, Mario Mina, Júlia Falcão, Luis Vasquez-Reina, Anna Sallés, Aitor Gonzalez-Agirre, Olatz Perez-de-Viñaspre

Previous literature has largely shown that Large Language Models (LLMs)
perpetuate social biases learnt from their pre-training data. Given the notable
lack of resources for social bias evaluation in languages other than English,
and for social contexts outside of the United States, this paper introduces the
Spanish and the Catalan Bias Benchmarks for Question Answering (EsBBQ and
CaBBQ). Based on the original BBQ, these two parallel datasets are designed to
assess social bias across 10 categories using a multiple-choice QA setting, now
adapted to the Spanish and Catalan languages and to the social context of
Spain. We report evaluation results on different LLMs, factoring in model
family, size and variant. Our results show that models tend to fail to choose
the correct answer in ambiguous scenarios, and that high QA accuracy often
correlates with greater reliance on social biases.

### Cryptography and Security

### 1. [From Alerts to Intelligence: A Novel LLM-Aided Framework for Host-based Intrusion Detection](http://arxiv.org/pdf/2507.10873v1)

Authors: Danyu Sun, Jinghuai Zhang, Jiacen Xu, Yu Zheng, Yuan Tian, Zhou Li

Host-based intrusion detection system (HIDS) is a key defense component to
protect the organizations from advanced threats like Advanced Persistent
Threats (APT). By analyzing the fine-grained logs with approaches like data
provenance, HIDS has shown successes in capturing sophisticated attack traces.
Despite the progresses embarked by the research community and industry, HIDS
still frequently encounters backlash from their operators in the deployed
environments, due to issues like high false-positive rate, inconsistent
outcomes across environments and human-unfriendly detection results. Large
Language Models (LLMs) have great potentials to advance the state of HIDS,
given their extensive knowledge of attack techniques and their ability to
detect anomalies through semantic analysis, anchored by recent studies. Yet,
our preliminary analysis indicates that building an HIDS by naively prompting
an LLM is unlikely to succeed. In this work, we explore the direction of
building a customized LLM pipeline for HIDS and develop a system named SHIELD.
SHIELD addresses challenges related to LLM's token limits, confusion of
background noises, etc., by integrating a variety of techniques like
event-level Masked Autoencoder (MAE) for attack window detection, attack
evidence identification and expansion, Deterministic Data Augmentation (DDA)
for profiling normal activities, and multi-purpose prompting that guides the
LLM to conduct precise and interpretable attack investigations. Extensive
experiments on three log datasets (DARPA-E3, NodLink-simulated-data and
ATLASv2) show that SHIELD consistently achieves outstanding performance in
comparison with 5 representative HIDS. These findings highlight the potential
of LLMs as powerful tools for intrusion detection and pave the way for future
research in this domain.

### 2. [DVFS: A Dynamic Verifiable Fuzzy Search Service for Encrypted Cloud Data](http://arxiv.org/pdf/2507.10927v1)

Authors: Jie Zhang, Xiaohong Li, Man Zheng, Zhe Hou, Guangdong Bai, Ruitao Feng

Cloud storage introduces critical privacy challenges for encrypted data
retrieval, where fuzzy multi-keyword search enables approximate matching while
preserving data confidentiality. Existing solutions face fundamental trade-offs
between security and efficiency: linear-search mechanisms provide adaptive
security but incur prohibitive overhead for large-scale data, while tree-based
indexes improve performance at the cost of branch leakage vulnerabilities.
  To address these limitations, we propose DVFS - a dynamic verifiable fuzzy
search service with three core innovations: (1) An \textit{adaptive-secure
fuzzy search} method integrating locality-sensitive hashing with virtual binary
trees, eliminating branch leakage while reducing search complexity from linear
to sublinear ($O(\log n)$ time); (2) A \textit{dual-repository version control}
mechanism supporting dynamic updates with forward privacy, preventing
information leakage during operations; (3) A \textit{blockchain-based
verification system} that ensures correctness and completeness via smart
contracts, achieving $O(\log n)$ verification complexity.
  Our solution advances secure encrypted retrieval by simultaneously resolving
the security-performance paradox and enabling trustworthy dynamic operations.

### 3. [FacialMotionID: Identifying Users of Mixed Reality Headsets using Abstract Facial Motion Representations](http://arxiv.org/pdf/2507.11138v1)

Authors: Adriano Castro, Simon Hanisch, Matin Fallahi, Thorsten Strufe

Facial motion capture in mixed reality headsets enables real-time avatar
animation, allowing users to convey non-verbal cues during virtual
interactions. However, as facial motion data constitutes a behavioral
biometric, its use raises novel privacy concerns. With mixed reality systems
becoming more immersive and widespread, understanding whether face motion data
can lead to user identification or inference of sensitive attributes is
increasingly important.
  To address this, we conducted a study with 116 participants using three types
of headsets across three sessions, collecting facial, eye, and head motion data
during verbal and non-verbal tasks. The data used is not raw video, but rather,
abstract representations that are used to animate digital avatars. Our analysis
shows that individuals can be re-identified from this data with up to 98%
balanced accuracy, are even identifiable across device types, and that
emotional states can be inferred with up to 86% accuracy. These results
underscore the potential privacy risks inherent in face motion tracking in
mixed reality environments.

### 4. [Bridging the Gap in Vision Language Models in Identifying Unsafe Concepts Across Modalities](http://arxiv.org/pdf/2507.11155v1)

Authors: Yiting Qu, Michael Backes, Yang Zhang

Vision-language models (VLMs) are increasingly applied to identify unsafe or
inappropriate images due to their internal ethical standards and powerful
reasoning abilities. However, it is still unclear whether they can recognize
various unsafe concepts when presented in different modalities, such as text
and images. To address this, we first compile the UnsafeConcepts dataset,
featuring 75 unsafe concepts, i.e., ``Swastika,'' ``Sexual Harassment,'' and
``Assaults,'' along with associated 1.5K images. We then conduct a systematic
evaluation of VLMs' perception (concept recognition) and alignment (ethical
reasoning) capabilities. We assess eight popular VLMs and find that, although
most VLMs accurately perceive unsafe concepts, they sometimes mistakenly
classify these concepts as safe. We also identify a consistent modality gap
among open-source VLMs in distinguishing between visual and textual unsafe
concepts. To bridge this gap, we introduce a simplified reinforcement learning
(RL)-based approach using proximal policy optimization (PPO) to strengthen the
ability to identify unsafe concepts from images. Our approach uses reward
scores based directly on VLM responses, bypassing the need for collecting
human-annotated preference data to train a new reward model. Experimental
results show that our approach effectively enhances VLM alignment on images
while preserving general capabilities. It outperforms baselines such as
supervised fine-tuning (SFT) and direct preference optimization (DPO). We hope
our dataset, evaluation findings, and proposed alignment solution contribute to
the community's efforts in advancing safe VLMs.

### 5. [LRCTI: A Large Language Model-Based Framework for Multi-Step Evidence Retrieval and Reasoning in Cyber Threat Intelligence Credibility Verification](http://arxiv.org/pdf/2507.11310v1)

Authors: Fengxiao Tang, Huan Li, Ming Zhao, Zongzong Wu, Shisong Peng, Tao Yin

Verifying the credibility of Cyber Threat Intelligence (CTI) is essential for
reliable cybersecurity defense. However, traditional approaches typically treat
this task as a static classification problem, relying on handcrafted features
or isolated deep learning models. These methods often lack the robustness
needed to handle incomplete, heterogeneous, or noisy intelligence, and they
provide limited transparency in decision-making-factors that reduce their
effectiveness in real-world threat environments. To address these limitations,
we propose LRCTI, a Large Language Model (LLM)-based framework designed for
multi-step CTI credibility verification. The framework first employs a text
summarization module to distill complex intelligence reports into concise and
actionable threat claims. It then uses an adaptive multi-step evidence
retrieval mechanism that iteratively identifies and refines supporting
information from a CTI-specific corpus, guided by LLM feedback. Finally, a
prompt-based Natural Language Inference (NLI) module is applied to evaluate the
credibility of each claim while generating interpretable justifications for the
classification outcome. Experiments conducted on two benchmark datasets,
CTI-200 and PolitiFact show that LRCTI improves F1-Macro and F1-Micro scores by
over 5%, reaching 90.9% and 93.6%, respectively, compared to state-of-the-art
baselines. These results demonstrate that LRCTI effectively addresses the core
limitations of prior methods, offering a scalable, accurate, and explainable
solution for automated CTI credibility verification

### 6. [Security Enclave Architecture for Heterogeneous Security Primitives for Supply-Chain Attacks](http://arxiv.org/pdf/2507.10971v1)

Authors: Kshitij Raj, Atri Chatterjee, Patanjali SLPSK, Swarup Bhunia, Sandip Ray

Designing secure architectures for system-on-chip (SoC) platforms is a highly
intricate and time-intensive task, often requiring months of development and
meticulous verification. Even minor architectural oversights can lead to
critical vulnerabilities that undermine the security of the entire chip. In
response to this challenge, we introduce CITADEL, a modular security framework
aimed at streamlining the creation of robust security architectures for SoCs.
CITADEL offers a configurable, plug-and-play subsystem composed of custom
intellectual property (IP) blocks, enabling the construction of diverse
security mechanisms tailored to specific threats. As a concrete demonstration,
we instantiate CITADEL to defend against supply-chain threats, illustrating how
the framework adapts to one of the most pressing concerns in hardware security.
This paper explores the range of obstacles encountered when building a unified
security architecture capable of addressing multiple attack vectors and
presents CITADEL's strategies for overcoming them. Through several real-world
case studies, we showcase the practical implementation of CITADEL and present a
thorough evaluation of its impact on silicon area and power consumption across
various ASIC technologies. Results indicate that CITADEL introduces only
minimal resource overhead, making it a practical solution for enhancing SoC
security.

### 7. [Hashed Watermark as a Filter: Defeating Forging and Overwriting Attacks in Weight-based Neural Network Watermarking](http://arxiv.org/pdf/2507.11137v1)

Authors: Yuan Yao, Jin Song, Jian Jin

As valuable digital assets, deep neural networks necessitate robust ownership
protection, positioning neural network watermarking (NNW) as a promising
solution. Among various NNW approaches, weight-based methods are favored for
their simplicity and practicality; however, they remain vulnerable to forging
and overwriting attacks. To address those challenges, we propose NeuralMark, a
robust method built around a hashed watermark filter. Specifically, we utilize
a hash function to generate an irreversible binary watermark from a secret key,
which is then used as a filter to select the model parameters for embedding.
This design cleverly intertwines the embedding parameters with the hashed
watermark, providing a robust defense against both forging and overwriting
attacks. An average pooling is also incorporated to resist fine-tuning and
pruning attacks. Furthermore, it can be seamlessly integrated into various
neural network architectures, ensuring broad applicability. Theoretically, we
analyze its security boundary. Empirically, we verify its effectiveness and
robustness across 13 distinct Convolutional and Transformer architectures,
covering five image classification tasks and one text generation task. The
source codes are available at https://github.com/AIResearch-Group/NeuralMark.

### 8. [A Review of Privacy Metrics for Privacy-Preserving Synthetic Data Generation](http://arxiv.org/pdf/2507.11324v1)

Authors: Frederik Marinus Trudslev, Matteo Lissandrini, Juan Manuel Rodriguez, Martin Bøgsted, Daniele Dell'Aglio

Privacy Preserving Synthetic Data Generation (PP-SDG) has emerged to produce
synthetic datasets from personal data while maintaining privacy and utility.
Differential privacy (DP) is the property of a PP-SDG mechanism that
establishes how protected individuals are when sharing their sensitive data. It
is however difficult to interpret the privacy loss ($\varepsilon$) expressed by
DP. To make the actual risk associated with the privacy loss more transparent,
multiple privacy metrics (PMs) have been proposed to assess the privacy risk of
the data. These PMs are utilized in separate studies to assess newly introduced
PP-SDG mechanisms. Consequently, these PMs embody the same assumptions as the
PP-SDG mechanism they were made to assess. Therefore, a thorough definition of
how these are calculated is necessary. In this work, we present the assumptions
and mathematical formulations of 17 distinct privacy metrics.

### 9. [MalCodeAI: Autonomous Vulnerability Detection and Remediation via Language Agnostic Code Reasoning](http://arxiv.org/pdf/2507.10898v1)

Authors: Jugal Gajjar, Kamalasankari Subramaniakuppusamy, Noha El Kachach

The growing complexity of cyber threats and the limitations of traditional
vulnerability detection tools necessitate novel approaches for securing
software systems. We introduce MalCodeAI, a language-agnostic, multi-stage AI
pipeline for autonomous code security analysis and remediation. MalCodeAI
combines code decomposition and semantic reasoning using fine-tuned
Qwen2.5-Coder-3B-Instruct models, optimized through Low-Rank Adaptation (LoRA)
within the MLX framework, and delivers scalable, accurate results across 14
programming languages. In Phase 1, the model achieved a validation loss as low
as 0.397 for functional decomposition and summarization of code segments after
200 iterations, 6 trainable layers, and a learning rate of 2 x 10^(-5). In
Phase 2, for vulnerability detection and remediation, it achieved a best
validation loss of 0.199 using the same number of iterations and trainable
layers but with an increased learning rate of 4 x 10^(-5), effectively
identifying security flaws and suggesting actionable fixes. MalCodeAI supports
red-hat-style exploit tracing, CVSS-based risk scoring, and zero-shot
generalization to detect complex, zero-day vulnerabilities. In a qualitative
evaluation involving 15 developers, the system received high scores in
usefulness (mean 8.06/10), interpretability (mean 7.40/10), and readability of
outputs (mean 7.53/10), confirming its practical value in real-world
development workflows. This work marks a significant advancement toward
intelligent, explainable, and developer-centric software security solutions.

### 10. [Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs](http://arxiv.org/pdf/2507.11112v1)

Authors: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

Recent studies have shown that Large Language Models (LLMs) are vulnerable to
data poisoning attacks, where malicious training examples embed hidden
behaviours triggered by specific input patterns. However, most existing works
assume a phrase and focus on the attack's effectiveness, offering limited
understanding of trigger mechanisms and how multiple triggers interact within
the model. In this paper, we present a framework for studying poisoning in
LLMs. We show that multiple distinct backdoor triggers can coexist within a
single model without interfering with each other, enabling adversaries to embed
several triggers concurrently. Using multiple triggers with high embedding
similarity, we demonstrate that poisoned triggers can achieve robust activation
even when tokens are substituted or separated by long token spans. Our findings
expose a broader and more persistent vulnerability surface in LLMs. To mitigate
this threat, we propose a post hoc recovery method that selectively retrains
specific model components based on a layer-wise weight difference analysis. Our
method effectively removes the trigger behaviour with minimal parameter
updates, presenting a practical and efficient defence against multi-trigger
poisoning.

### Computer Vision and Pattern Recognition

### 1. [Trexplorer Super: Topologically Correct Centerline Tree Tracking of Tubular Objects in CT Volumes](http://arxiv.org/pdf/2507.10881v1)

Authors: Roman Naeem, David Hagerman, Jennifer Alvén, Lennart Svensson, Fredrik Kahl

Tubular tree structures, such as blood vessels and airways, are essential in
human anatomy and accurately tracking them while preserving their topology is
crucial for various downstream tasks. Trexplorer is a recurrent model designed
for centerline tracking in 3D medical images but it struggles with predicting
duplicate branches and terminating tracking prematurely. To address these
issues, we present Trexplorer Super, an enhanced version that notably improves
performance through novel advancements. However, evaluating centerline tracking
models is challenging due to the lack of public datasets. To enable thorough
evaluation, we develop three centerline datasets, one synthetic and two real,
each with increasing difficulty. Using these datasets, we conduct a
comprehensive evaluation of existing state-of-the-art (SOTA) models and compare
them with our approach. Trexplorer Super outperforms previous SOTA models on
every dataset. Our results also highlight that strong performance on synthetic
data does not necessarily translate to real datasets. The code and datasets are
available at https://github.com/RomStriker/Trexplorer-Super.

### 2. [GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization](http://arxiv.org/pdf/2507.10935v1)

Authors: Shaowen Tong, Zimin Xia, Alexandre Alahi, Xuming He, Yujiao Shi

Cross-view localization, the task of estimating a camera's
3-degrees-of-freedom (3-DoF) pose by aligning ground-level images with
satellite images, is crucial for large-scale outdoor applications like
autonomous navigation and augmented reality. Existing methods often rely on
fully supervised learning, which requires costly ground-truth pose annotations.
In this work, we propose GeoDistill, a Geometry guided weakly supervised self
distillation framework that uses teacher-student learning with Field-of-View
(FoV)-based masking to enhance local feature learning for robust cross-view
localization. In GeoDistill, the teacher model localizes a panoramic image,
while the student model predicts locations from a limited FoV counterpart
created by FoV-based masking. By aligning the student's predictions with those
of the teacher, the student focuses on key features like lane lines and ignores
textureless regions, such as roads. This results in more accurate predictions
and reduced uncertainty, regardless of whether the query images are panoramas
or limited FoV images. Our experiments show that GeoDistill significantly
improves localization performance across different frameworks. Additionally, we
introduce a novel orientation estimation network that predicts relative
orientation without requiring precise planar position ground truth. GeoDistill
provides a scalable and efficient solution for real-world cross-view
localization challenges. Code and model can be found at
https://github.com/tongshw/GeoDistill.

### 3. [Graph Aggregation Prototype Learning for Semantic Change Detection in Remote Sensing](http://arxiv.org/pdf/2507.10938v1)

Authors: Zhengyi Xu, Haoran Wu, Wen Jiang, Jie Geng

Semantic change detection (SCD) extends the binary change detection task to
provide not only the change locations but also the detailed "from-to"
categories in multi-temporal remote sensing data. Such detailed semantic
insights into changes offer considerable advantages for a wide array of
applications. However, since SCD involves the simultaneous optimization of
multiple tasks, the model is prone to negative transfer due to task-specific
learning difficulties and conflicting gradient flows. To address this issue, we
propose Graph Aggregation Prototype Learning for Semantic Change Detection in
remote sensing(GAPL-SCD). In this framework, a multi-task joint optimization
method is designed to optimize the primary task of semantic segmentation and
change detection, along with the auxiliary task of graph aggregation prototype
learning. Adaptive weight allocation and gradient rotation methods are used to
alleviate the conflict between training tasks and improve multi-task learning
capabilities. Specifically, the graph aggregation prototype learning module
constructs an interaction graph using high-level features. Prototypes serve as
class proxies, enabling category-level domain alignment across time points and
reducing interference from irrelevant changes. Additionally, the proposed
self-query multi-level feature interaction and bi-temporal feature fusion
modules further enhance multi-scale feature representation, improving
performance in complex scenes. Experimental results on the SECOND and
Landsat-SCD datasets demonstrate that our method achieves state-of-the-art
performance, with significant improvements in accuracy and robustness for SCD
task.

### 4. [Robust ID-Specific Face Restoration via Alignment Learning](http://arxiv.org/pdf/2507.10943v1)

Authors: Yushun Fang, Lu Liu, Xiang Gao, Qiang Hu, Ning Cao, Jianghe Cui, Gang Chen, Xiaoyun Zhang

The latest developments in Face Restoration have yielded significant
advancements in visual quality through the utilization of diverse diffusion
priors. Nevertheless, the uncertainty of face identity introduced by
identity-obscure inputs and stochastic generative processes remains unresolved.
To address this challenge, we present Robust ID-Specific Face Restoration
(RIDFR), a novel ID-specific face restoration framework based on diffusion
models. Specifically, RIDFR leverages a pre-trained diffusion model in
conjunction with two parallel conditioning modules. The Content Injection
Module inputs the severely degraded image, while the Identity Injection Module
integrates the specific identity from a given image. Subsequently, RIDFR
incorporates Alignment Learning, which aligns the restoration results from
multiple references with the same identity in order to suppress the
interference of ID-irrelevant face semantics (e.g. pose, expression, make-up,
hair style). Experiments demonstrate that our framework outperforms the
state-of-the-art methods, reconstructing high-quality ID-specific results with
high identity fidelity and demonstrating strong robustness.

### 5. [Women Sport Actions Dataset for Visual Classification Using Small Scale Training Data](http://arxiv.org/pdf/2507.10969v1)

Authors: Palash Ray, Mahuya Sasmal, Asish Bera

Sports action classification representing complex body postures and
player-object interactions is an emerging area in image-based sports analysis.
Some works have contributed to automated sports action recognition using
machine learning techniques over the past decades. However, sufficient image
datasets representing women sports actions with enough intra- and inter-class
variations are not available to the researchers. To overcome this limitation,
this work presents a new dataset named WomenSports for women sports
classification using small-scale training data. This dataset includes a variety
of sports activities, covering wide variations in movements, environments, and
interactions among players. In addition, this study proposes a convolutional
neural network (CNN) for deep feature extraction. A channel attention scheme
upon local contextual regions is applied to refine and enhance feature
representation. The experiments are carried out on three different sports
datasets and one dance dataset for generalizing the proposed algorithm, and the
performances on these datasets are noteworthy. The deep learning method
achieves 89.15% top-1 classification accuracy using ResNet-50 on the proposed
WomenSports dataset, which is publicly available for research at Mendeley Data.

### 6. [Mind the Gap: Bridging Occlusion in Gait Recognition via Residual Gap Correction](http://arxiv.org/pdf/2507.10978v1)

Authors: Ayush Gupta, Siyuan Huang, Rama Chellappa

Gait is becoming popular as a method of person re-identification because of
its ability to identify people at a distance. However, most current works in
gait recognition do not address the practical problem of occlusions. Among
those which do, some require paired tuples of occluded and holistic sequences,
which are impractical to collect in the real world. Further, these approaches
work on occlusions but fail to retain performance on holistic inputs. To
address these challenges, we propose RG-Gait, a method for residual correction
for occluded gait recognition with holistic retention. We model the problem as
a residual learning task, conceptualizing the occluded gait signature as a
residual deviation from the holistic gait representation. Our proposed network
adaptively integrates the learned residual, significantly improving performance
on occluded gait sequences without compromising the holistic recognition
accuracy. We evaluate our approach on the challenging Gait3D, GREW and BRIAR
datasets and show that learning the residual can be an effective technique to
tackle occluded gait recognition with holistic retention.

### 7. [Bridge Feature Matching and Cross-Modal Alignment with Mutual-filtering for Zero-shot Anomaly Detection](http://arxiv.org/pdf/2507.11003v1)

Authors: Yuhu Bai, Jiangning Zhang, Yunkang Cao, Guangyuan Lu, Qingdong He, Xiangtai Li, Guanzhong Tian

With the advent of vision-language models (e.g., CLIP) in zero- and few-shot
settings, CLIP has been widely applied to zero-shot anomaly detection (ZSAD) in
recent research, where the rare classes are essential and expected in many
applications. This study introduces \textbf{FiSeCLIP} for ZSAD with
training-free \textbf{CLIP}, combining the feature matching with the
cross-modal alignment. Testing with the entire dataset is impractical, while
batch-based testing better aligns with real industrial needs, and images within
a batch can serve as mutual reference points. Accordingly, FiSeCLIP utilizes
other images in the same batch as reference information for the current image.
However, the lack of labels for these references can introduce ambiguity, we
apply text information to \textbf{fi}lter out noisy features. In addition, we
further explore CLIP's inherent potential to restore its local
\textbf{se}mantic correlation, adapting it for fine-grained anomaly detection
tasks to enable a more accurate filtering process. Our approach exhibits
superior performance for both anomaly classification and segmentation on
anomaly detection benchmarks, building a stronger baseline for the direction,
e.g., on MVTec-AD, FiSeCLIP outperforms the SOTA AdaCLIP by
+4.6\%$\uparrow$/+5.7\%$\uparrow$ in segmentation metrics AU-ROC/$F_1$-max.

### 8. [Human-Guided Shade Artifact Suppression in CBCT-to-MDCT Translation via Schrödinger Bridge with Conditional Diffusion](http://arxiv.org/pdf/2507.11025v1)

Authors: Sung Ho Kang, Hyun-Cheol Park

We present a novel framework for CBCT-to-MDCT translation, grounded in the
Schrodinger Bridge (SB) formulation, which integrates GAN-derived priors with
human-guided conditional diffusion. Unlike conventional GANs or diffusion
models, our approach explicitly enforces boundary consistency between CBCT
inputs and pseudo targets, ensuring both anatomical fidelity and perceptual
controllability. Binary human feedback is incorporated via classifier-free
guidance (CFG), effectively steering the generative process toward clinically
preferred outcomes. Through iterative refinement and tournament-based
preference selection, the model internalizes human preferences without relying
on a reward model. Subtraction image visualizations reveal that the proposed
method selectively attenuates shade artifacts in key anatomical regions while
preserving fine structural detail. Quantitative evaluations further demonstrate
superior performance across RMSE, SSIM, LPIPS, and Dice metrics on clinical
datasets -- outperforming prior GAN- and fine-tuning-based feedback methods --
while requiring only 10 sampling steps. These findings underscore the
effectiveness and efficiency of our framework for real-time, preference-aligned
medical image translation.

### 9. [Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](http://arxiv.org/pdf/2507.11030v1)

Authors: Sunghyun Park, Jungsoo Lee, Shubhankar Borse, Munawar Hayat, Sungha Choi, Kyuwoong Hwang, Fatih Porikli

While open-vocabulary semantic segmentation (OVSS) can segment an image into
semantic regions based on arbitrarily given text descriptions even for classes
unseen during training, it fails to understand personal texts (e.g., `my mug
cup') for segmenting regions of specific interest to users. This paper
addresses challenges like recognizing `my mug cup' among `multiple mug cups'.
To overcome this challenge, we introduce a novel task termed
\textit{personalized open-vocabulary semantic segmentation} and propose a text
prompt tuning-based plug-in method designed to recognize personal visual
concepts using a few pairs of images and masks, while maintaining the
performance of the original OVSS. Based on the observation that reducing false
predictions is essential when applying text prompt tuning to this task, our
proposed method employs `negative mask proposal' that captures visual concepts
other than the personalized concept. We further improve the performance by
enriching the representation of text prompts by injecting visual embeddings of
the personal concept into them. This approach enhances personalized OVSS
without compromising the original OVSS performance. We demonstrate the
superiority of our method on our newly established benchmarks for this task,
including FSS$^\text{per}$, CUB$^\text{per}$, and ADE$^\text{per}$.

### 10. [Efficient Dual-domain Image Dehazing with Haze Prior Perception](http://arxiv.org/pdf/2507.11035v1)

Authors: Lirong Zheng, Yanshan Li, Rui Yu, Kaihao Zhang

Transformer-based models exhibit strong global modeling capabilities in
single-image dehazing, but their high computational cost limits real-time
applicability. Existing methods predominantly rely on spatial-domain features
to capture long-range dependencies, which are computationally expensive and
often inadequate under complex haze conditions. While some approaches introduce
frequency-domain cues, the weak coupling between spatial and frequency branches
limits the overall performance. To overcome these limitations, we propose the
Dark Channel Guided Frequency-aware Dehazing Network (DGFDNet), a novel
dual-domain framework that performs physically guided degradation alignment
across spatial and frequency domains. At its core, the DGFDBlock comprises two
key modules: 1) the Haze-Aware Frequency Modulator (HAFM), which generates a
pixel-level haze confidence map from dark channel priors to adaptively enhance
haze-relevant frequency components, thereby achieving global degradation-aware
spectral modulation; 2) the Multi-level Gating Aggregation Module (MGAM), which
fuses multi-scale features through diverse convolutional kernels and hybrid
gating mechanisms to recover fine structural details. Additionally, a Prior
Correction Guidance Branch (PCGB) incorporates a closed-loop feedback
mechanism, enabling iterative refinement of the prior by intermediate dehazed
features and significantly improving haze localization accuracy, especially in
challenging outdoor scenes. Extensive experiments on four benchmark haze
datasets demonstrate that DGFDNet achieves state-of-the-art performance with
superior robustness and real-time efficiency. Code is available at:
https://github.com/Dilizlr/DGFDNet.

### Computers and Society

### 1. [Artificial Intelligence and Journalism: A Systematic Bibliometric and Thematic Analysis of Global Research](http://arxiv.org/pdf/2507.10891v1)

Authors: Mohammad Al Masum Molla, Md Manjurul Ahsan

Artificial Intelligence (AI) is reshaping journalistic practices across the
globe, offering new opportunities while raising ethical, professional, and
societal concerns. This study presents a comprehensive systematic review of
published articles on AI in journalism from 2010 to 2025. Following the
Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA)
2020 guidelines, a total of 72 peer-reviewed articles were selected from Scopus
and Web of Science databases. The analysis combines bibliometric mapping and
qualitative thematic synthesis to identify dominant trends, technologies,
geographical distributions, and ethical debates. Additionally, sentiment
analysis was performed on article abstracts using the Valence Aware Dictionary
and sEntiment Reasoner (VADER) algorithm to capture evaluative tones across the
literature. The findings show a sharp increase in research activity after 2020,
with prominent focus areas including automation, misinformation, and ethical
governance. While most studies reflect cautious optimism, concerns over bias,
transparency, and accountability remain persistent. The review also highlights
regional disparities in scholarly contributions, with limited representation
from the Global South. By integrating quantitative and qualitative insights,
this study offers a multi-dimensional understanding of how AI is transforming
journalism and proposes future research directions for inclusive and
responsible innovation.

### 2. [The Potential Impact of Disruptive AI Innovations on U.S. Occupations](http://arxiv.org/pdf/2507.11403v1)

Authors: Munjung Kim, Marios Constantinides, Sanja Šćepanović, Yong-Yeol Ahn, Daniele Quercia

The rapid rise of AI is poised to disrupt the labor market. However, AI is
not a monolith; its impact depends on both the nature of the innovation and the
jobs it affects. While computational approaches are emerging, there is no
consensus on how to systematically measure an innovation's disruptive
potential. Here, we calculate the disruption index of 3,237 U.S. AI patents
(2015-2022) and link them to job tasks to distinguish between "consolidating"
AI innovations that reinforce existing structures and "disruptive" AI
innovations that alter them. Our analysis reveals that consolidating AI
primarily targets physical, routine, and solo tasks, common in manufacturing
and construction in the Midwest and central states. By contrast, disruptive AI
affects unpredictable and mental tasks, particularly in coastal science and
technology sectors. Surprisingly, we also find that disruptive AI
disproportionately affects areas already facing skilled labor shortages,
suggesting disruptive AI technologies may accelerate change where workers are
scarce rather than replacing a surplus. Ultimately, consolidating AI appears to
extend current automation trends, while disruptive AI is set to transform
complex mental work, with a notable exception for collaborative tasks.

### 3. [What Should LLMs Forget? Quantifying Personal Data in LLMs for Right-to-Be-Forgotten Requests](http://arxiv.org/pdf/2507.11128v1)

Authors: Dimitri Staufer

Large Language Models (LLMs) can memorize and reveal personal information,
raising concerns regarding compliance with the EU's GDPR, particularly the
Right to Be Forgotten (RTBF). Existing machine unlearning methods assume the
data to forget is already known but do not address how to identify which
individual-fact associations are stored in the model. Privacy auditing
techniques typically operate at the population level or target a small set of
identifiers, limiting applicability to individual-level data inquiries. We
introduce WikiMem, a dataset of over 5,000 natural language canaries covering
243 human-related properties from Wikidata, and a model-agnostic metric to
quantify human-fact associations in LLMs. Our approach ranks ground-truth
values against counterfactuals using calibrated negative log-likelihood across
paraphrased prompts. We evaluate 200 individuals across 15 LLMs (410M-70B
parameters), showing that memorization correlates with subject web presence and
model scale. We provide a foundation for identifying memorized personal data in
LLMs at the individual level, enabling the dynamic construction of forget sets
for machine unlearning and RTBF requests.

### Databases

### 1. [LLMATCH: A Unified Schema Matching Framework with Large Language Models](http://arxiv.org/pdf/2507.10897v1)

Authors: Sha Wang, Yuchen Li, Hanhua Xiao, Bing Tian Dai, Roy Ka-Wei Lee, Yanfei Dong, Lambert Deng

Schema matching is a foundational task in enterprise data integration, aiming
to align disparate data sources. While traditional methods handle simple
one-to-one table mappings, they often struggle with complex multi-table schema
matching in real-world applications. We present LLMatch, a unified and modular
schema matching framework. LLMatch decomposes schema matching into three
distinct stages: schema preparation, table-candidate selection, and
column-level alignment, enabling component-level evaluation and future-proof
compatibility. It includes a novel two-stage optimization strategy: a Rollup
module that consolidates semantically related columns into higher-order
concepts, followed by a Drilldown module that re-expands these concepts for
fine-grained column mapping. To address the scarcity of complex semantic
matching benchmarks, we introduce SchemaNet, a benchmark derived from
real-world schema pairs across three enterprise domains, designed to capture
the challenges of multi-table schema alignment in practical settings.
Experiments demonstrate that LLMatch significantly improves matching accuracy
in complex schema matching settings and substantially boosts engineer
productivity in real-world data integration.

### 2. [TOPJoin: A Context-Aware Multi-Criteria Approach for Joinable Column Search](http://arxiv.org/pdf/2507.11505v1)

Authors: Harsha Kokel, Aamod Khatiwada, Tejaswini Pedapati, Haritha Ananthakrishnan, Oktie Hassanzadeh, Horst Samulowitz, Kavitha Srinivas

One of the major challenges in enterprise data analysis is the task of
finding joinable tables that are conceptually related and provide meaningful
insights. Traditionally, joinable tables have been discovered through a search
for similar columns, where two columns are considered similar syntactically if
there is a set overlap or they are considered similar semantically if either
the column embeddings or value embeddings are closer in the embedding space.
However, for enterprise data lakes, column similarity is not sufficient to
identify joinable columns and tables. The context of the query column is
important. Hence, in this work, we first define context-aware column
joinability. Then we propose a multi-criteria approach, called TOPJoin, for
joinable column search. We evaluate TOPJoin against existing join search
baselines over one academic and one real-world join search benchmark. Through
experiments, we find that TOPJoin performs better on both benchmarks than the
baselines.

### 3. [Towards Practical Benchmarking of Data Cleaning Techniques: On Generating Authentic Errors via Large Language Models](http://arxiv.org/pdf/2507.10934v1)

Authors: Xinyuan Liu, Jiahui Chen, Bocheng Hu, Yu Sun, Xinyang Chen, Shaoxu Song

Data quality remains an important challenge in data-driven systems, as errors
in tabular data can severely compromise downstream analytics and machine
learning performance. Although numerous error detection algorithms have been
proposed, the lack of diverse, real-world error datasets limits comprehensive
evaluation. Manual error annotation is both time-consuming and inconsistent,
motivating the exploration of synthetic error generation as an alternative. In
this work, we introduce TableEG, a framework that leverages large language
models (LLMs) to generate authentic errors. By employing a table fine-tuning
strategy and a triplet representation $(I, T, O)$ to model error generation,
detection, and correction tasks, TableEG captures the complex dependencies
inherent in two-dimensional tables. Trained on 12 real-world datasets spanning
10 diverse domains, TableEG ensures that the synthesized errors faithfully
reflect authentic error distributions. Experimental results indicate that
errors generated by TableEG exhibit superior pattern and distribution
similarity compared to both rule-based methods and LLM-generated errors without
fine-tuning. Furthermore, performance metrics on TableEG-generated errors
closely align with those on real-world errors across nearly all datasets and
detection algorithms, particularly for machine learning based detection
techniques. Overall, TableEG not only bridges the gap between synthetic and
real-world errors but also establishes a robust benchmark for subsequent error
detection and correction tasks.

### 4. [A Review of Privacy Metrics for Privacy-Preserving Synthetic Data Generation](http://arxiv.org/pdf/2507.11324v1)

Authors: Frederik Marinus Trudslev, Matteo Lissandrini, Juan Manuel Rodriguez, Martin Bøgsted, Daniele Dell'Aglio

Privacy Preserving Synthetic Data Generation (PP-SDG) has emerged to produce
synthetic datasets from personal data while maintaining privacy and utility.
Differential privacy (DP) is the property of a PP-SDG mechanism that
establishes how protected individuals are when sharing their sensitive data. It
is however difficult to interpret the privacy loss ($\varepsilon$) expressed by
DP. To make the actual risk associated with the privacy loss more transparent,
multiple privacy metrics (PMs) have been proposed to assess the privacy risk of
the data. These PMs are utilized in separate studies to assess newly introduced
PP-SDG mechanisms. Consequently, these PMs embody the same assumptions as the
PP-SDG mechanism they were made to assess. Therefore, a thorough definition of
how these are calculated is necessary. In this work, we present the assumptions
and mathematical formulations of 17 distinct privacy metrics.

### Distributed, Parallel, and Cluster Computing

### 1. [MMStencil: Optimizing High-order Stencils on Multicore CPU using Matrix Unit](http://arxiv.org/pdf/2507.11067v1)

Authors: Yinuo Wang, Tianqi Mao, Lin Gan, Wubing Wan, Zeyu Song, Jiayu Fu, Lanke He, Wenqiang Wang, Zekun Yin, Wei Xue, Guangwen Yang

Matrix-accelerated stencil computation is a hot research topic, yet its
application to three-dimensional (3D) high-order stencils and HPC remains
underexplored. With the emergence of matrix units on multicore CPUs, we analyze
matrix-based acceleration strategies and tailor an optimal approach for 3D
high-order stencils. We introduce algorithmic optimizations based on SIMD and
matrix units to address strided memory accesses, alignment conflicts, and
redundant accesses. We propose memory optimizations to boost on-package memory
efficiency, and a novel multi-thread parallelism paradigm to overcome
data-sharing challenges caused by the absence of shared data caches. MMStencil
sustains consistently high hardware utilization across diverse stencil shapes
and dimensions. Our DMA-based inter-NUMA communication further mitigates NUMA
effects and MPI limitations in hybrid parallelism. Combining all the
innovations, MMStencil outperforms state-of-the-art libraries on Nvidia A100
GPGPU by up to 2.1x. Moreover, the performance improvements translate directly
to real-world HPC applications and enable RTM applications to yield 1.8x
speedup versus a highly optimized industrial Nvidia A100 GPGPU version.

### 2. [Generating Dynamic Graph Algorithms for Multiple Backends for a Graph DSL](http://arxiv.org/pdf/2507.11094v1)

Authors: Nibedita Behera, Ashwina Kumar, Atharva Chougule, Mohammed Shan P S, Rushabh Nirdosh Lalwani, Rupesh Nasre

With the rapid growth of unstructured and semistructured data, parallelizing
graph algorithms has become essential for efficiency. However, due to the
inherent irregularity in computation, memory access patterns, and
communication, graph algorithms are notoriously difficult to parallelize. To
address this challenge, several libraries, frameworks, and domain-specific
languages (DSLs) have been proposed to ease the parallel programming burden for
domain experts. Existing frameworks partially or fully abstract away
parallelism intricacies, provide intuitive scheduling mnemonics, and employ
program analysis to identify data races and generate synchronization code.
Despite these advances, most frameworks are limited in their abstractions and
runtime optimizations, especially when dealing with static graphs. In contrast,
many real-world graphs are inherently dynamic, with evolving structures over
time through insertions, deletions, and modifications of vertices, edges, and
attributes. Generating efficient and correctly synchronized code for such
dynamic graph algorithms remains a significant challenge.
  In this work, we introduce an abstraction scheme and runtime optimizations
for the efficient processing of morph algorithms. Specifically, given an
initial graph G and a set of updates $\Delta$G involving edge insertions and
deletions, we express the dynamic processing logic through a DSL and
automatically generate parallel code targeting multicore, distributed, and
many-core environments. We demonstrate the effectiveness of our approach by
applying the DSL-generated code to ten large graphs with diverse
characteristics and three widely used algorithms: Shortest Paths, PageRank, and
Triangle Counting.

### 3. [Boosting Scientific Error-Bounded Lossy Compression through Optimized Synergistic Lossy-Lossless Orchestration](http://arxiv.org/pdf/2507.11165v1)

Authors: Shixun Wu, Jinwen Pan, Jinyang Liu, Jiannan Tian, Ziwei Qiu, Jiajun Huang, Kai Zhao, Xin Liang, Sheng Di, Zizhong Chen, Franck Cappello

As high-performance computing architectures evolve, more scientific computing
workflows are being deployed on advanced computing platforms such as GPUs.
These workflows can produce raw data at extremely high throughputs, requiring
urgent high-ratio and low-latency error-bounded data compression solutions. In
this paper, we propose cuSZ-Hi, an optimized high-ratio GPU-based scientific
error-bounded lossy compressor with a flexible, domain-irrelevant, and fully
open-source framework design. Our novel contributions are: 1) We maximally
optimize the parallelized interpolation-based data prediction scheme on GPUs,
enabling the full functionalities of interpolation-based scientific data
prediction that are adaptive to diverse data characteristics; 2) We thoroughly
explore and investigate lossless data encoding techniques, then craft and
incorporate the best-fit lossless encoding pipelines for maximizing the
compression ratio of cuSZ-Hi; 3) We systematically evaluate cuSZ-Hi on
benchmarking datasets together with representative baselines. Compared to
existing state-of-the-art scientific lossy compressors, with comparative or
better throughput than existing high-ratio scientific error-bounded lossy
compressors on GPUs, cuSZ-Hi can achieve up to 249% compression ratio
improvement under the same error bound, and up to 215% compression ratio
improvement under the same decompression data PSNR.

### 4. [A new Dune grid for scalable dynamic adaptivity based on the p4est software library](http://arxiv.org/pdf/2507.11386v1)

Authors: Carsten Burstedde, Mikhail Kirilin, Robert Klöfkorn

In this work we extend the Dune solver library with another grid interface to
the open-source p4est software. While Dune already supports about a dozen
different mesh implementations through its mesh interface Dune-Grid, we
undertake this new coupling effort in order to inherit p4est's practically
unlimited MPI scalability as well as its relatively thin data structures, and
its native support for multi-block (forest) mesh topologies in both 2D and 3D.
  The presented implementation is compared to an existing implementation based
on Dune-ALUGrid for a variety of challenging test examples in a parallel
environment. The numerical experiments show that the implementation presented
here is outperforming Dune-ALUGrid in terms of scalability. In addition, an
alternative balancing strategy is presented to ensure 2:1 balancing across
element faces showing improved performance compared to the existing p4est
balance strategy in the numerical examples considered in this work.

### 5. [Quantifying the Energy Consumption and Carbon Emissions of LLM Inference via Simulations](http://arxiv.org/pdf/2507.11417v1)

Authors: Miray Özcan, Philipp Wiesner, Philipp Weiß, Odej Kao

The environmental impact of Large Language Models (LLMs) is rising
significantly, with inference now accounting for more than half of their total
lifecycle carbon emissions. However, existing simulation frameworks, which are
increasingly used to determine efficient LLM deployments, lack any concept of
power and, therefore, cannot accurately estimate inference-related emissions.
We present a simulation framework to assess the energy and carbon implications
of LLM inference under varying deployment setups. First, we extend a
high-fidelity LLM inference simulator with a GPU power model that estimates
power consumption based on utilization metrics, enabling analysis across
configurations like batch size, sequence length, and model parallelism. Second,
we integrate simulation outputs into an energy system co-simulation environment
to quantify carbon emissions under specific grid conditions and explore the
potential of carbon-aware scheduling. Through scenario-based analysis, our
framework reveals how inference parameters affect energy demand and carbon
footprint, demonstrates a renewable offset potential of up to 69.2% in an
illustrative deployment case, and provides a foundation for future carbon-aware
inference infrastructure design.

### 6. [Arcturus: A Cloud Overlay Network for Global Accelerator with Enhanced Performance and Stability](http://arxiv.org/pdf/2507.10928v1)

Authors: Matthew Yang Liu, Chuang Chen, Pengcheng Lv, Hui Guo, Yanan Zhang, Cong Wang, Yusen Li, Zhenyu Li, Yu-Chu Tian

Global Accelerator (GA) services play a vital role in ensuring low-latency,
high-reliability communication for real-time interactive applications. However,
existing GA offerings are tightly bound to specific cloud providers, resulting
in high costs, rigid deployment, and limited flexibility, especially for
large-scale or budget-sensitive deployments. Arcturus is a cloud-native GA
framework that revisits the design of GA systems by leveraging low-cost,
heterogeneous cloud resources across multiple providers. Rather than relying on
fixed, high-end infrastructure, Arcturus dynamically constructs its
acceleration network and balances performance, stability, and resource
efficiency. To achieve this, Arcturus introduces a two-plane design: a
forwarding plane that builds a proxy network with adaptive control, and a
scheduling plane that coordinates load and routing through lightweight,
quantitative optimization. Evaluations under millions of RPS show that Arcturus
outperforms commercial GA services by up to 1.7X in acceleration performance,
reduces cost by 71%, and maintains over 80% resource efficiency--demonstrating
efficient use of cloud resources at scale.

### 7. [Deterministic Lower Bounds for $k$-Edge Connectivity in the Distributed Sketching Model](http://arxiv.org/pdf/2507.11257v1)

Authors: Peter Robinson, Ming Ming Tan

We study the $k$-edge connectivity problem on undirected graphs in the
distributed sketching model, where we have $n$ nodes and a referee. Each node
sends a single message to the referee based on its 1-hop neighborhood in the
graph, and the referee must decide whether the graph is $k$-edge connected by
taking into account the received messages.
  We present the first lower bound for deciding a graph connectivity problem in
this model with a deterministic algorithm. Concretely, we show that the worst
case message length is $\Omega( k )$ bits for $k$-edge connectivity, for any
super-constant $k = O(\sqrt{n})$. Previously, only a lower bound of $\Omega(
\log^3 n )$ bits was known for ($1$-edge) connectivity, due to Yu (SODA 2021).
In fact, our result is the first super-polylogarithmic lower bound for a
connectivity decision problem in the distributed graph sketching model.
  To obtain our result, we introduce a new lower bound graph construction, as
well as a new 3-party communication complexity problem that we call
UniqueOverlap. As this problem does not appear to be amenable to reductions to
existing hard problems such as set disjointness or indexing due to correlations
between the inputs of the three players, we leverage results from
cross-intersecting set families to prove the hardness of UniqueOverlap for
deterministic algorithms. Finally, we obtain the sought lower bound for
deciding $k$-edge connectivity via a novel simulation argument that, in
contrast to previous works, does not introduce any probability of error and
thus works for deterministic algorithms.

### 8. [Cyclic Data Streaming on GPUs for Short Range Stencils Applied to Molecular Dynamics](http://arxiv.org/pdf/2507.11289v1)

Authors: Martin Rose, Simon Homes, Lukas Ramsperger, Jose Gracia, Christoph Niethammer, Jadran Vrabec

In the quest for highest performance in scientific computing, we present a
novel framework that relies on high-bandwidth communication between GPUs in a
compute cluster. The framework offers linear scaling of performance for
explicit algorithms that is only limited by the size of the dataset and the
number of GPUs. Slices of the dataset propagate in a ring of processes (GPUs)
from one GPU, where they are processed, to the next, which results in a
parallel-in-time parallelization. The user of the framework has to write GPU
kernels that implement the algorithm and provide slices of the dataset.
Knowledge about the underlying parallelization strategy is not required because
the communication between processes is carried out by the framework. As a case
study, molecular dynamics simulation based on the Lennard-Jones potential is
implemented to measure the performance for a homogeneous fluid. Single node
performance and strong scaling behavior of this framework is compared to
LAMMPS, which is outperformed in the strong scaling case.

### 9. [FLsim: A Modular and Library-Agnostic Simulation Framework for Federated Learning](http://arxiv.org/pdf/2507.11430v1)

Authors: Arnab Mukherjee, Raju Halder, Joydeep Chandra

Federated Learning (FL) has undergone significant development since its
inception in 2016, advancing from basic algorithms to complex methodologies
tailored to address diverse challenges and use cases. However, research and
benchmarking of novel FL techniques against a plethora of established
state-of-the-art solutions remain challenging. To streamline this process, we
introduce FLsim, a comprehensive FL simulation framework designed to meet the
diverse requirements of FL workflows in the literature. FLsim is characterized
by its modularity, scalability, resource efficiency, and controlled
reproducibility of experimental outcomes. Its easy to use interface allows
users to specify customized FL requirements through job configuration, which
supports: (a) customized data distributions, ranging from non-independent and
identically distributed (non-iid) data to independent and identically
distributed (iid) data, (b) selection of local learning algorithms according to
user preferences, with complete agnosticism to ML libraries, (c) choice of
network topology illustrating communication patterns among nodes, (d)
definition of model aggregation and consensus algorithms, and (e) pluggable
blockchain support for enhanced robustness. Through a series of experimental
evaluations, we demonstrate the effectiveness and versatility of FLsim in
simulating a diverse range of state-of-the-art FL experiments. We envisage that
FLsim would mark a significant advancement in FL simulation frameworks,
offering unprecedented flexibility and functionality for researchers and
practitioners alike.

### 10. [Uniting the World by Dividing it: Federated Maps to Enable Spatial Applications](http://arxiv.org/pdf/2507.11437v1)

Authors: Sagar Bharadwaj, Srinivasan Seshan, Anthony Rowe

The emergence of the Spatial Web -- the Web where content is tied to
real-world locations has the potential to improve and enable many applications
such as augmented reality, navigation, robotics, and more. The Spatial Web is
missing a key ingredient that is impeding its growth -- a spatial naming system
to resolve real-world locations to names. Today's spatial naming systems are
digital maps such as Google and Apple maps. These maps and the location-based
services provided on top of these maps are primarily controlled by a few large
corporations and mostly cover outdoor public spaces. Emerging classes of
applications, such as persistent world-scale augmented reality, require
detailed maps of both outdoor and indoor spaces. Existing centralized mapping
infrastructures are proving insufficient for such applications because of the
scale of cartography efforts required and the privacy of indoor map data.
  In this paper, we present a case for a federated spatial naming system, or in
other words, a federated mapping infrastructure. This enables disparate parties
to manage and serve their own maps of physical regions and unlocks scalability
of map management, isolation and privacy of maps. Map-related services such as
address-to-location mapping, location-based search, and routing needs
re-architecting to work on federated maps. We discuss some essential services
and practicalities of enabling these services.

### Digital Libraries

### 1. [Artificial Intelligence and Journalism: A Systematic Bibliometric and Thematic Analysis of Global Research](http://arxiv.org/pdf/2507.10891v1)

Authors: Mohammad Al Masum Molla, Md Manjurul Ahsan

Artificial Intelligence (AI) is reshaping journalistic practices across the
globe, offering new opportunities while raising ethical, professional, and
societal concerns. This study presents a comprehensive systematic review of
published articles on AI in journalism from 2010 to 2025. Following the
Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA)
2020 guidelines, a total of 72 peer-reviewed articles were selected from Scopus
and Web of Science databases. The analysis combines bibliometric mapping and
qualitative thematic synthesis to identify dominant trends, technologies,
geographical distributions, and ethical debates. Additionally, sentiment
analysis was performed on article abstracts using the Valence Aware Dictionary
and sEntiment Reasoner (VADER) algorithm to capture evaluative tones across the
literature. The findings show a sharp increase in research activity after 2020,
with prominent focus areas including automation, misinformation, and ethical
governance. While most studies reflect cautious optimism, concerns over bias,
transparency, and accountability remain persistent. The review also highlights
regional disparities in scholarly contributions, with limited representation
from the Global South. By integrating quantitative and qualitative insights,
this study offers a multi-dimensional understanding of how AI is transforming
journalism and proposes future research directions for inclusive and
responsible innovation.

### 2. [Automated Novelty Evaluation of Academic Paper: A Collaborative Approach Integrating Human and Large Language Model Knowledge](http://arxiv.org/pdf/2507.11330v1)

Authors: Wenqing Wu, Chengzhi Zhang, Yi Zhao

Novelty is a crucial criterion in the peer review process for evaluating
academic papers. Traditionally, it's judged by experts or measure by unique
reference combinations. Both methods have limitations: experts have limited
knowledge, and the effectiveness of the combination method is uncertain.
Moreover, it's unclear if unique citations truly measure novelty. The large
language model (LLM) possesses a wealth of knowledge, while human experts
possess judgment abilities that the LLM does not possess. Therefore, our
research integrates the knowledge and abilities of LLM and human experts to
address the limitations of novelty assessment. The most common novelty in
academic papers is the introduction of new methods. In this paper, we propose
leveraging human knowledge and LLM to assist pretrained language models (PLMs,
e.g. BERT etc.) in predicting the method novelty of papers. Specifically, we
extract sentences related to the novelty of the academic paper from peer review
reports and use LLM to summarize the methodology section of the academic paper,
which are then used to fine-tune PLMs. In addition, we have designed a
text-guided fusion module with novel Sparse-Attention to better integrate human
and LLM knowledge. We compared the method we proposed with a large number of
baselines. Extensive experiments demonstrate that our method achieves superior
performance.

### Discrete Mathematics

### 1. [Lower bounds for dominating set reconfiguration on sparse (directed) graphs](http://arxiv.org/pdf/2507.11446v1)

Authors: Jona Dirks, Alexandre Vigny

In a graph, a vertex dominates itself and its neighbors, and a dominating set
is a set of vertices that together dominate the entire graph. Given a graph and
two dominating sets of equal size $k$, the {\em Dominating Set Reconfiguration
with Token sliding} (DSR-TS) problem asks whether one can, by iteratively
replacing a vertex by an adjacent one, transform the first set into the second
one, while ensuring that every set during the reconfiguration process is a
dominating set.
  The token jumping variant, where a vertex can be replaced by a non-adjacent
one, is known to be efficiently solvable on many graph classes such as planar,
bounded treewidth, and the very broad notion of nowhere-dense classes of
graphs. Alternatively, some algorithms also exist for the reconfiguration of
independent sets in the token sliding paradigm for graph classes with bounded
degree or large girth.
  We show that DSR-TS is W[2]-hard when parameterized $k$, the pathwidth of the
instance, and the iteration of the reconfiguration sequence (a recently
introduced parameter). This is a setting where both the token jumping and the
independent set variants are fixed parameter tractable. Not restricting the
iteration yields W[2] hardness already on graphs with treewidth 9 and pathwidth
13.
  In the directed variant (DSR-DTS), we are only allowed to replace a vertex
with an out-neighbor. We show that DSR-DTS is NP-hard on DAGs of treewidth 5
and W[2]-hard for both the case of DAGs of depth 3 parameterized by $k$, and
the case of DAGs when parameterized by $k$ and the pathwidth of the instance
(independent set reconfiguration is again FPT in both settings).

### 2. [Convolutive sequences, I: Through the lens of integer partition functions](http://arxiv.org/pdf/2507.10965v1)

Authors: Shane Chern, Dennis Eichhorn, Shishuo Fu, James A. Sellers

Motivated by the convolutive behavior of the counting function for partitions
with designated summands in which all parts are odd, we consider coefficient
sequences $(a_n)_{n\ge 0}$ of primitive eta-products that satisfy the generic
convolutive property
  \begin{align*}
  \sum_{n\ge 0} a_{mn} q^n = \left(\sum_{n\ge 0} a_n q^n\right)^m
  \end{align*}
  for a specific positive integer $m$. Given the results of an exhaustive
search of the Online Encyclopedia of Integer Sequences for such sequences for
$m$ up to $6$, we first focus on the case where $m=2$ with our attention mainly
paid to the combinatorics of two $2$-convolutive sequences, featuring bijective
proofs for both. For other $2$-convolutive sequences discovered in the OEIS, we
apply generating function manipulations to show their convolutivity. We also
give two examples of $3$-convolutive sequences. Finally, we discuss other
convolutive series that are not eta-products.

### 3. [On Tight Robust Coresets for $k$-Medians Clustering](http://arxiv.org/pdf/2507.11260v1)

Authors: Lingxiao Huang, Zhenyu Jiang, Yi Li, Xuan Wu

This paper considers coresets for the robust $k$-medians problem with $m$
outliers, and new constructions in various metric spaces are obtained.
Specifically, for metric spaces with a bounded VC or doubling dimension $d$,
the coreset size is $O(m) + \tilde{O}(kd\varepsilon^{-2})$, which is optimal up
to logarithmic factors. For Euclidean spaces, the coreset size is
$O(m\varepsilon^{-1}) +
\tilde{O}(\min\{k^{4/3}\varepsilon^{-2},k\varepsilon^{-3}\})$, improving upon a
recent result by Jiang and Lou (ICALP 2025). These results also extend to
robust $(k,z)$-clustering, yielding, for VC and doubling dimension, a coreset
size of $O(m) + \tilde{O}(kd\varepsilon^{-2z})$ with the optimal linear
dependence on $m$. This extended result improves upon the earlier work of Huang
et al. (SODA 2025). The techniques introduce novel dataset decompositions,
enabling chaining arguments to be applied jointly across multiple components.

### 4. [Rapid Mixing of Glauber Dynamics for Monotone Systems via Entropic Independence](http://arxiv.org/pdf/2507.11031v1)

Authors: Weiming Feng, Minji Yang

We study the mixing time of Glauber dynamics on monotone systems. For
monotone systems satisfying the entropic independence condition, we prove a new
mixing time comparison result for Glauber dynamics. For concrete applications,
we obtain $\tilde{O}(n)$ mixing time for the random cluster model induced by
the ferromagnetic Ising model with consistently biased external fields, and
$\tilde{O}(n^2)$ mixing time for the bipartite hardcore model under the
one-sided uniqueness condition, where $n$ is the number of variables in
corresponding models, improving the best known results in [Chen and Zhang,
SODA'23] and [Chen, Liu, and Yin, FOCS'23], respectively.
  Our proof combines ideas from the stochastic dominance argument in the
classical censoring inequality and the recently developed high-dimensional
expanders. The key step in the proof is a novel comparison result between the
Glauber dynamics and the field dynamics for monotone systems.

### Data Structures and Algorithms

### 1. [Solving Linear Programs with Differential Privacy](http://arxiv.org/pdf/2507.10946v1)

Authors: Alina Ene, Huy Le Nguyen, Ta Duy Nguyen, Adrian Vladu

We study the problem of solving linear programs of the form $Ax\le b$,
$x\ge0$ with differential privacy. For homogeneous LPs $Ax\ge0$, we give an
efficient $(\epsilon,\delta)$-differentially private algorithm which with
probability at least $1-\beta$ finds in polynomial time a solution that
satisfies all but
$O(\frac{d^{2}}{\epsilon}\log^{2}\frac{d}{\delta\beta}\sqrt{\log\frac{1}{\rho_{0}}})$
constraints, for problems with margin $\rho_{0}>0$. This improves the bound of
$O(\frac{d^{5}}{\epsilon}\log^{1.5}\frac{1}{\rho_{0}}\mathrm{poly}\log(d,\frac{1}{\delta},\frac{1}{\beta}))$
by [Kaplan-Mansour-Moran-Stemmer-Tur, STOC '25]. For general LPs $Ax\le b$,
$x\ge0$ with potentially zero margin, we give an efficient
$(\epsilon,\delta)$-differentially private algorithm that w.h.p drops
$O(\frac{d^{4}}{\epsilon}\log^{2.5}\frac{d}{\delta}\sqrt{\log dU})$
constraints, where $U$ is an upper bound for the entries of $A$ and $b$ in
absolute value. This improves the result by Kaplan et al. by at least a factor
of $d^{5}$. Our techniques build upon privatizing a rescaling perceptron
algorithm by [Hoberg-Rothvoss, IPCO '17] and a more refined iterative procedure
for identifying equality constraints by Kaplan et al.

### 2. [Faster algorithms for k-Orthogonal Vectors in low dimension](http://arxiv.org/pdf/2507.11098v1)

Authors: Anita Dürr, Evangelos Kipouridis, Karol Węgrzycki

In the Orthogonal Vectors problem (OV), we are given two families $A, B$ of
subsets of $\{1,\ldots,d\}$, each of size $n$, and the task is to decide
whether there exists a pair $a \in A$ and $b \in B$ such that $a \cap b =
\emptyset$. Straightforward algorithms for this problem run in $\mathcal{O}(n^2
\cdot d)$ or $\mathcal{O}(2^d \cdot n)$ time, and assuming SETH, there is no
$2^{o(d)}\cdot n^{2-\varepsilon}$ time algorithm that solves this problem for
any constant $\varepsilon > 0$.
  Williams (FOCS 2024) presented a $\tilde{\mathcal{O}}(1.35^d \cdot n)$-time
algorithm for the problem, based on the succinct equality-rank decomposition of
the disjointness matrix. In this paper, we present a combinatorial algorithm
that runs in randomized time $\tilde{\mathcal{O}}(1.25^d n)$. This can be
improved to $\mathcal{O}(1.16^d \cdot n)$ using computer-aided evaluations.
  We generalize our result to the $k$-Orthogonal Vectors problem, where given
$k$ families $A_1,\ldots,A_k$ of subsets of $\{1,\ldots,d\}$, each of size $n$,
the task is to find elements $a_i \in A_i$ for every $i \in \{1,\ldots,k\}$
such that $a_1 \cap a_2 \cap \ldots \cap a_k = \emptyset$. We show that for
every fixed $k \ge 2$, there exists $\varepsilon_k > 0$ such that the $k$-OV
problem can be solved in time $\mathcal{O}(2^{(1 - \varepsilon_k)\cdot d}\cdot
n)$. We also show that, asymptotically, this is the best we can hope for: for
any $\varepsilon > 0$ there exists a $k \ge 2$ such that $2^{(1 -
\varepsilon)\cdot d} \cdot n^{\mathcal{O}(1)}$ time algorithm for
$k$-Orthogonal Vectors would contradict the Set Cover Conjecture.

### 3. [Efficient Branch-and-Bound for Submodular Function Maximization under Knapsack Constraint](http://arxiv.org/pdf/2507.11107v1)

Authors: Yimin Hao, Yi Zhou, Chao Xu, Zhang-Hua Fu

The submodular knapsack problem (SKP), which seeks to maximize a submodular
set function by selecting a subset of elements within a given budget, is an
important discrete optimization problem. The majority of existing approaches to
solving the SKP are approximation algorithms. However, in domains such as
health-care facility location and risk management, the need for optimal
solutions is still critical, necessitating the use of exact algorithms over
approximation methods. In this paper, we present an optimal branch-and-bound
approach, featuring a novel upper bound with a worst-case tightness guarantee
and an efficient dual branching method to minimize repeat computations.
Experiments in applications such as facility location, weighted coverage,
influence maximization, and so on show that the algorithms that implement the
new ideas are far more efficient than conventional methods.

### 4. [Finding Order-Preserving Subgraphs](http://arxiv.org/pdf/2507.11115v1)

Authors: Haruya Imamura, Yasuaki Kobayashi, Yota Otachi, Toshiki Saitoh, Keita Sato, Asahi Takaoka, Ryo Yoshinaka

(Induced) Subgraph Isomorphism and Maximum Common (Induced) Subgraph are
fundamental problems in graph pattern matching and similarity computation. In
graphs derived from time-series data or protein structures, a natural total
ordering of vertices often arises from their underlying structure, such as
temporal sequences or amino acid sequences. This motivates the study of problem
variants that respect this inherent ordering. This paper addresses Ordered
(Induced) Subgraph Isomorphism (O(I)SI) and its generalization, Maximum Common
Ordered (Induced) Subgraph (MCO(I)S), which seek to find subgraph isomorphisms
that preserve the vertex orderings of two given ordered graphs. Our main
contributions are threefold: (1) We prove that these problems remain
NP-complete even when restricted to small graph classes, such as trees of depth
2 and threshold graphs. (2) We establish a gap in computational complexity
between OSI and OISI on certain graph classes. For instance, OSI is
polynomial-time solvable for interval graphs with their interval orderings,
whereas OISI remains NP-complete under the same setting. (3) We demonstrate
that the tractability of these problems can depend on the vertex ordering. For
example, while OISI is NP-complete on threshold graphs, its generalization,
MCOIS, can be solved in polynomial time if the specific vertex orderings that
characterize the threshold graphs are provided.

### 5. [Fully Dynamic Euclidean k-Means](http://arxiv.org/pdf/2507.11256v1)

Authors: Sayan Bhattacharya, Martín Costa, Ermiya Farokhnejad, Shaofeng H. -C. Jiang, Yaonan Jin, Jianing Lou

We consider the fundamental Euclidean $k$-means clustering problem in a
dynamic setting, where the input $X \subseteq \mathbb{R}^d$ evolves over time
via a sequence of point insertions/deletions. We have to explicitly maintain a
solution (a set of $k$ centers) $S \subseteq \mathbb{R}^d$ throughout these
updates, while minimizing the approximation ratio, the update time (time taken
to handle a point insertion/deletion) and the recourse (number of changes made
to the solution $S$) of the algorithm.
  We present a dynamic algorithm for this problem with
$\text{poly}(1/\epsilon)$-approximation ratio, $\tilde{O}(k^{\epsilon})$ update
time and $\tilde{O}(1)$ recourse. In the general regime, where the dimension
$d$ cannot be assumed to be a fixed constant, our algorithm has almost optimal
guarantees across all these three parameters. Indeed, improving our update time
or approximation ratio would imply beating the state-of-the-art static
algorithm for this problem (which is widely believed to be the best possible),
and the recourse of any dynamic algorithm must be $\Omega(1)$.
  We obtain our result by building on top of the recent work of [Bhattacharya,
Costa, Farokhnejad; STOC'25], which gave a near-optimal dynamic algorithm for
$k$-means in general metric spaces (as opposed to in the Euclidean setting).
Along the way, we design several novel geometric data structures that are of
independent interest. Specifically, one of our main contributions is designing
the first consistent hashing scheme [Czumaj, Jiang, Krauthgamer, Vesel\'y,
Yang; FOCS'22] that achieves $\text{poly}(d)$ running time per point evaluation
with competitive parameters.

### 6. [Permutation patterns in streams](http://arxiv.org/pdf/2507.11291v1)

Authors: Benjamin Aram Berendsohn

Permutation patterns and pattern avoidance are central, well-studied concepts
in combinatorics and computer science. Given two permutations $\tau$ and $\pi$,
the pattern matching problem (PPM) asks whether $\tau$ contains $\pi$. This
problem arises in various contexts in computer science and statistics and has
been studied extensively in exact-, parameterized-, approximate-,
property-testing- and other formulations.
  In this paper, we study pattern matching in a \emph{streaming setting}, when
the input $\tau$ is revealed sequentially, one element at a time. There is
extensive work on the space complexity of various statistics in streams of
integers. The novelty of our setting is that the input stream is \emph{a
permutation}, which allows inferring some information about future inputs. Our
algorithms crucially take advantage of this fact, while existing lower bound
techniques become difficult to apply.
  We show that the complexity of the problem changes dramatically depending on
the pattern~$\pi$. The space requirement is: $\Theta(k\log{n})$ for the
monotone patterns $\pi = 12\dots k$, or $\pi = k\dots21$, $O(\sqrt{n\log{n}})$
for $\pi \in \{312,132\}$, $O(\sqrt{n} \log n)$ for $\pi \in \{231,213\}$, and
$\widetilde{\Theta}_{\pi}(n)$ for all other $\pi$. If $\tau$ is an arbitrary
sequence of integers (not necessary a permutation), we show that the complexity
is $\widetilde{\Theta}_{\pi}(n)$ in all except the first (monotone) cases.

### 7. [Scheduling on Identical Machines with Setup Time and Unknown Execution Time](http://arxiv.org/pdf/2507.11311v1)

Authors: Yasushi Kawase, Kazuhisa Makino, Vinh Long Phan, Hanna Sumita

In this study, we investigate a scheduling problem on identical machines in
which jobs require initial setup before execution. We assume that an algorithm
can dynamically form a batch (i.e., a collection of jobs to be processed
together) from the remaining jobs. The setup time is modeled as a known
monotone function of the set of jobs within a batch, while the execution time
of each job remains unknown until completion. This uncertainty poses
significant challenges for minimizing the makespan. We address these challenges
by considering two scenarios: each job batch must be assigned to a single
machine, or a batch may be distributed across multiple machines. For both
scenarios, we analyze settings with and without preemption. Across these four
settings, we design online algorithms that achieve asymptotically optimal
competitive ratios with respect to both the number of jobs and the number of
machines.

### 8. [Multipass Linear Sketches for Geometric LP-Type Problems](http://arxiv.org/pdf/2507.11484v1)

Authors: N. Efe Çekirge, William Gay, David P. Woodruff

LP-type problems such as the Minimum Enclosing Ball (MEB), Linear Support
Vector Machine (SVM), Linear Programming (LP), and Semidefinite Programming
(SDP) are fundamental combinatorial optimization problems, with many important
applications in machine learning applications such as classification,
bioinformatics, and noisy learning. We study LP-type problems in several
streaming and distributed big data models, giving $\varepsilon$-approximation
linear sketching algorithms with a focus on the high accuracy regime with low
dimensionality $d$, that is, when ${d < (1/\varepsilon)^{0.999}}$. Our main
result is an $O(ds)$ pass algorithm with $O(s( \sqrt{d}/\varepsilon)^{3d/s})
\cdot \mathrm{poly}(d, \log (1/\varepsilon))$ space complexity in words, for
any parameter $s \in [1, d \log (1/\varepsilon)]$, to solve
$\varepsilon$-approximate LP-type problems of $O(d)$ combinatorial and VC
dimension. Notably, by taking $s = d \log (1/\varepsilon)$, we achieve space
complexity polynomial in $d$ and polylogarithmic in $1/\varepsilon$, presenting
exponential improvements in $1/\varepsilon$ over current algorithms. We
complement our results by showing lower bounds of $(1/\varepsilon)^{\Omega(d)}$
for any $1$-pass algorithm solving the $(1 + \varepsilon)$-approximation MEB
and linear SVM problems, further motivating our multi-pass approach.

### 9. [FPT Parameterisations of Fractional and Generalised Hypertree Width](http://arxiv.org/pdf/2507.11080v1)

Authors: Matthias Lanzinger, Igor Razgon, Daniel Unterberger

We present the first fixed-parameter tractable (fpt) algorithms for precisely
determining several central hypergraph decomposition parameters, including
generalized hypertree width, fractional hypertree width, and adaptive width.
Despite the recognized importance of these measures in complexity theory,
databases, and constraint satisfaction, no exact fpt algorithms for any of them
had previously been known. Our results are obtained for hypergraph classes of
bounded rank and bounded degree.
  Our approach extends a recent algorithm for treewidth (Boja\'ncyk &
Pilipczuk, LMCS 2022) utilizing monadic second-order (MSO) transductions.
Leveraging this framework, we overcome the significant technical hurdles
presented by hypergraphs, whose structural decompositions are technically much
more intricate than their graph counterparts.

### 10. [Deterministic Lower Bounds for $k$-Edge Connectivity in the Distributed Sketching Model](http://arxiv.org/pdf/2507.11257v1)

Authors: Peter Robinson, Ming Ming Tan

We study the $k$-edge connectivity problem on undirected graphs in the
distributed sketching model, where we have $n$ nodes and a referee. Each node
sends a single message to the referee based on its 1-hop neighborhood in the
graph, and the referee must decide whether the graph is $k$-edge connected by
taking into account the received messages.
  We present the first lower bound for deciding a graph connectivity problem in
this model with a deterministic algorithm. Concretely, we show that the worst
case message length is $\Omega( k )$ bits for $k$-edge connectivity, for any
super-constant $k = O(\sqrt{n})$. Previously, only a lower bound of $\Omega(
\log^3 n )$ bits was known for ($1$-edge) connectivity, due to Yu (SODA 2021).
In fact, our result is the first super-polylogarithmic lower bound for a
connectivity decision problem in the distributed graph sketching model.
  To obtain our result, we introduce a new lower bound graph construction, as
well as a new 3-party communication complexity problem that we call
UniqueOverlap. As this problem does not appear to be amenable to reductions to
existing hard problems such as set disjointness or indexing due to correlations
between the inputs of the three players, we leverage results from
cross-intersecting set families to prove the hardness of UniqueOverlap for
deterministic algorithms. Finally, we obtain the sought lower bound for
deciding $k$-edge connectivity via a novel simulation argument that, in
contrast to previous works, does not introduce any probability of error and
thus works for deterministic algorithms.

### Emerging Technologies

### 1. [Fault-Free Analog Computing with Imperfect Hardware](http://arxiv.org/pdf/2507.11134v1)

Authors: Zhicheng Xu, Jiawei Liu, Sitao Huang, Zefan Li, Shengbo Wang, Bo Wen, Ruibin Mao, Mingrui Jiang, Giacomo Pedretti, Jim Ignowski, Kaibin Huang, Can Li

The growing demand for edge computing and AI drives research into analog
in-memory computing using memristors, which overcome data movement bottlenecks
by computing directly within memory. However, device failures and variations
critically limit analog systems' precision and reliability. Existing
fault-tolerance techniques, such as redundancy and retraining, are often
inadequate for high-precision applications or scenarios requiring fixed
matrices and privacy preservation. Here, we introduce and experimentally
demonstrate a fault-free matrix representation where target matrices are
decomposed into products of two adjustable sub-matrices programmed onto analog
hardware. This indirect, adaptive representation enables mathematical
optimization to bypass faulty devices and eliminate differential pairs,
significantly enhancing computational density. Our memristor-based system
achieved >99.999% cosine similarity for a Discrete Fourier Transform matrix
despite 39% device fault rate, a fidelity unattainable with conventional direct
representation, which fails with single device faults (0.01% rate). We
demonstrated 56-fold bit-error-rate reduction in wireless communication and
>196% density with 179% energy efficiency improvements compared to
state-of-the-art techniques. This method, validated on memristors, applies
broadly to emerging memories and non-electrical computing substrates, showing
that device yield is no longer the primary bottleneck in analog computing
hardware.

### 2. [Uniting the World by Dividing it: Federated Maps to Enable Spatial Applications](http://arxiv.org/pdf/2507.11437v1)

Authors: Sagar Bharadwaj, Srinivasan Seshan, Anthony Rowe

The emergence of the Spatial Web -- the Web where content is tied to
real-world locations has the potential to improve and enable many applications
such as augmented reality, navigation, robotics, and more. The Spatial Web is
missing a key ingredient that is impeding its growth -- a spatial naming system
to resolve real-world locations to names. Today's spatial naming systems are
digital maps such as Google and Apple maps. These maps and the location-based
services provided on top of these maps are primarily controlled by a few large
corporations and mostly cover outdoor public spaces. Emerging classes of
applications, such as persistent world-scale augmented reality, require
detailed maps of both outdoor and indoor spaces. Existing centralized mapping
infrastructures are proving insufficient for such applications because of the
scale of cartography efforts required and the privacy of indoor map data.
  In this paper, we present a case for a federated spatial naming system, or in
other words, a federated mapping infrastructure. This enables disparate parties
to manage and serve their own maps of physical regions and unlocks scalability
of map management, isolation and privacy of maps. Map-related services such as
address-to-location mapping, location-based search, and routing needs
re-architecting to work on federated maps. We discuss some essential services
and practicalities of enabling these services.

### 3. [Stochastic Entanglement Configuration for Constructive Entanglement Topologies in Quantum Machine Learning with Application to Cardiac MRI](http://arxiv.org/pdf/2507.11401v1)

Authors: Mehri Mehrnia, Mohammed S. M. Elbaz

Efficient entanglement strategies are essential for advancing variational
quantum circuits (VQCs) for quantum machine learning (QML). However, most
current approaches use fixed entanglement topologies that are not adaptive to
task requirements, limiting potential gains over classical models. We introduce
a novel stochastic entanglement configuration method that systematically
generates diverse entanglement topologies to identify a subspace of
constructive entanglement configurations, defined as entanglement topologies
that boost hybrid model performance (e.g., classification accuracy) beyond
classical baselines. Each configuration is encoded as a stochastic binary
matrix, denoting directed entanglement between qubits. This enables scalable
exploration of the hyperspace of candidate entanglement topologies using
entanglement density and per-qubit constraints as key metrics. We define
unconstrained and constrained sampling modes, controlling entanglement per
qubit. Using our method, 400 stochastic configurations were generated and
evaluated in a hybrid QML for cardiac MRI disease classification. We identified
64 (16%) novel constructive entanglement configurations that consistently
outperformed the classical baseline. Ensemble aggregation of top-performing
configurations achieved ~0.92 classification accuracy, exceeding the classical
model (~0.87) by over 5%. Compared to four conventional topologies (ring,
nearest neighbor, no entanglement, fully entangled), none surpassed the
classical baseline (maximum accuracy ~0.82), while our configurations delivered
up to ~20% higher accuracy. Thus, highlighting the robustness and
generalizability of the identified constructive entanglements.

### Formal Languages and Automata Theory

### 1. [A Decision Procedure for Probabilistic Kleene Algebra with Angelic Nondeterminism](http://arxiv.org/pdf/2507.10980v1)

Authors: Shawn Ong, Dexter Kozen

We give a decision procedure and proof of correctness for the equational
theory of probabilistic Kleene algebra with angelic nondeterminism introduced
in Ong, Ma, and Kozen (2025).

### 2. [Polynomial Complementation of Nondeterministic 2-Way Finite Automata by 1-Limited Automata](http://arxiv.org/pdf/2507.11209v1)

Authors: Bruno Guillon, Luca Prigioniero, Javad Taheri

We prove that, paying a polynomial increase in size only, every unrestricted
two-way nondeterministic finite automaton (2NFA) can be complemented by a
1-limited automaton (1-LA), a nondeterministic extension of 2NFAs still
characterizing regular languages. The resulting machine is actually a
restricted form of 1-LAs -- known as 2NFAs with common guess -- and is
self-verifying. A corollary of our construction is that a single exponential is
necessary and sufficient for complementing 1-LAs.

### 3. [Execution and monitoring of HOA automata with HOAX](http://arxiv.org/pdf/2507.11126v1)

Authors: Luca Di Stefano

We present a tool called Hoax for the execution of {\omega}-automata
expressed in the popular HOA format. The tool leverages the notion of trap sets
to enable runtime monitoring of any (non-parity) acceptance condition supported
by the format. When the automaton is not monitorable, the tool may still be
able to recognise so-called ugly prefixes, and determine that no further
observation will ever lead to a conclusive verdict. The tool is open-source and
highly configurable. We present its formal foundations, its design, and compare
it against the trace analyser PyContract on a lock acquisition scenario.

### 4. [Foundation Models for Logistics: Toward Certifiable, Conversational Planning Interfaces](http://arxiv.org/pdf/2507.11352v1)

Authors: Yunhao Yang, Neel P. Bhatt, Christian Ellis, Alvaro Velasquez, Zhangyang Wang, Ufuk Topcu

Logistics operators, from battlefield coordinators rerouting airlifts ahead
of a storm to warehouse managers juggling late trucks, often face life-critical
decisions that demand both domain expertise and rapid and continuous
replanning. While popular methods like integer programming yield logistics
plans that satisfy user-defined logical constraints, they are slow and assume
an idealized mathematical model of the environment that does not account for
uncertainty. On the other hand, large language models (LLMs) can handle
uncertainty and promise to accelerate replanning while lowering the barrier to
entry by translating free-form utterances into executable plans, yet they
remain prone to misinterpretations and hallucinations that jeopardize safety
and cost. We introduce a neurosymbolic framework that pairs the accessibility
of natural-language dialogue with verifiable guarantees on goal interpretation.
It converts user requests into structured planning specifications, quantifies
its own uncertainty at the field and token level, and invokes an interactive
clarification loop whenever confidence falls below an adaptive threshold. A
lightweight model, fine-tuned on just 100 uncertainty-filtered examples,
surpasses the zero-shot performance of GPT-4.1 while cutting inference latency
by nearly 50%. These preliminary results highlight a practical path toward
certifiable, real-time, and user-aligned decision-making for complex logistics.

### Graphics

### 1. [OffsetCrust: Variable-Radius Offset Approximation with Power Diagrams](http://arxiv.org/pdf/2507.10924v1)

Authors: Zihan Zhao, Pengfei Wang, Minfeng Xu, Shuangmin Chen, Shiqing Xin, Changhe Tu, Wenping Wang

Offset surfaces, defined as the Minkowski sum of a base surface and a rolling
ball, play a crucial role in geometry processing, with applications ranging
from coverage motion planning to brush modeling. While considerable progress
has been made in computing constant-radius offset surfaces, computing
variable-radius offset surfaces remains a challenging problem. In this paper,
we present OffsetCrust, a novel framework that efficiently addresses the
variable-radius offsetting problem by computing a power diagram. Let $R$ denote
the radius function defined on the base surface $S$. The power diagram is
constructed from contributing sites, consisting of carefully sampled base
points on $S$ and their corresponding off-surface points, displaced along
$R$-dependent directions. In the constant-radius case only, these displacement
directions align exactly with the surface normals of $S$. Moreover, our method
mitigates the misalignment issues commonly seen in crust-based approaches
through a lightweight fine-tuning procedure. We validate the accuracy and
efficiency of OffsetCrust through extensive experiments, and demonstrate its
practical utility in applications such as reconstructing original boundary
surfaces from medial axis transform (MAT) representations.

### 2. [Developing and evaluating quilts for the depiction of large layered graphs](http://arxiv.org/pdf/2507.10883v1)

Authors: Juhee Bae, Benjamin Watson

Traditional layered graph depictions such as flow charts are in wide use. Yet
as graphs grow more complex, these depictions can become difficult to
understand. Quilts are matrix-based depictions for layered graphs designed to
address this problem. In this research, we first improve Quilts by developing
three design alternatives, and then compare the best of these alternatives to
better-known node-link and matrix depictions. A primary weakness in Quilts is
their depiction of skip links, links that do not simply connect to a succeeding
layer. Therefore in our first study, we compare Quilts using color-only,
text-only, and mixed (color and text) skip link depictions, finding that path
finding with the color-only depiction is significantly slower and less
accurate, and that in certain cases, the mixed depiction offers an advantage
over the text-only depiction. In our second study, we compare Quilts using the
mixed depiction to node-link diagrams and centered matrices. Overall results
show that users can find paths through graphs significantly faster with Quilts
(46.6 secs) than with node-link (58.3 secs) or matrix (71.2 secs) diagrams.
This speed advantage is still greater in large graphs (e.g. in 200 node graphs,
55.4 secs vs. 71.1 secs for node-link and 84.2 secs for matrix depictions).

### 3. [Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model](http://arxiv.org/pdf/2507.11465v1)

Authors: Nuri Ryu, Jiyun Won, Jooeun Son, Minsu Gong, Joo-Haeng Lee, Sunghyun Cho

High-quality 3D assets are essential for various applications in computer
graphics and 3D vision but remain scarce due to significant acquisition costs.
To address this shortage, we introduce Elevate3D, a novel framework that
transforms readily accessible low-quality 3D assets into higher quality. At the
core of Elevate3D is HFS-SDEdit, a specialized texture enhancement method that
significantly improves texture quality while preserving the appearance and
geometry while fixing its degradations. Furthermore, Elevate3D operates in a
view-by-view manner, alternating between texture and geometry refinement.
Unlike previous methods that have largely overlooked geometry refinement, our
framework leverages geometric cues from images refined with HFS-SDEdit by
employing state-of-the-art monocular geometry predictors. This approach ensures
detailed and accurate geometry that aligns seamlessly with the enhanced
texture. Elevate3D outperforms recent competitors by achieving state-of-the-art
quality in 3D model refinement, effectively addressing the scarcity of
high-quality open-source 3D assets.

### Computer Science and Game Theory

### 1. [Pricing with Tips in Three-Sided Delivery Platforms](http://arxiv.org/pdf/2507.10872v1)

Authors: Yannai A. Gonczarowski, Gary Qiurui Ma, David C. Parkes

We model a delivery platform facilitating transactions among three sides:
buyers, stores, and couriers. In addition to buyers paying store-specific
purchase prices and couriers receiving store--buyer-specific delivery
compensation from the platform, each buyer has the option to directly tip for
delivery from a specific store. An equilibrium consists of prices,
compensations, tips, and transactions that clear the market, such that buyers
receive deliveries from preferred stores considering the prices and tips they
pay, and couriers deliver preferred orders considering the compensations and
tips they receive.
  We illustrate the role of tips in pricing: Without tips, an equilibrium is
only guaranteed to exist when there are at least as many couriers as buyers or
stores. In contrast, with tips an equilibrium always exists. From an efficiency
perspective, the optimal with-tip equilibrium welfare is always weakly larger
than the optimal without-tip equilibrium welfare. However, we show that even
with tips, efficient equilibria may not exist, and calculating the optimal
equilibrium welfare is NP-hard. To address these challenges, we identify
natural conditions on market structure that ensure the existence of efficient
with-tip equilibria and allow these efficient equilibria to be computed in
polynomial time.

### 2. [Fair Contracts](http://arxiv.org/pdf/2507.11214v1)

Authors: Matteo Castiglioni, Junjie Chen, Yingkai Li

We introduce and study the problem of designing optimal contracts under
fairness constraints on the task assignments and compensations. We adopt the
notion of envy-free (EF) and its relaxations, $\epsilon$-EF and envy-free up to
one item (EF1), in contract design settings. Unlike fair allocations, EF
contracts are guaranteed to exist. However, computing any constant-factor
approximation to the optimal EF contract is NP-hard in general, even using
$\epsilon$-EF contracts. For this reason, we consider settings in which the
number of agents or tasks is constant. Notably, while even with three agents,
finding an EF contract better than $2/5$ approximation of the optimal is
NP-hard, we are able to design an FPTAS when the number of agents is constant,
under relaxed notions of $\epsilon$-EF and EF1. Moreover, we present a
polynomial-time algorithm for computing the optimal EF contract when the number
of tasks is constant. Finally, we analyze the price of fairness in contract
design. We show that the price of fairness for exact EF contracts can be
unbounded, even with a single task and two agents. In contrast, for EF1
contracts, the price of fairness is bounded between $\Omega(\sqrt{n})$ and
$O(n^2)$, where $n$ is the number of agents.

### 3. [A Parallelizable Approach for Characterizing NE in Zero-Sum Games After a Linear Number of Iterations of Gradient Descent](http://arxiv.org/pdf/2507.11366v1)

Authors: Taemin Kim, James P. Bailey

We study online optimization methods for zero-sum games, a fundamental
problem in adversarial learning in machine learning, economics, and many other
domains. Traditional methods approximate Nash equilibria (NE) using either
regret-based methods (time-average convergence) or contraction-map-based
methods (last-iterate convergence). We propose a new method based on
Hamiltonian dynamics in physics and prove that it can characterize the set of
NE in a finite (linear) number of iterations of alternating gradient descent in
the unbounded setting, modulo degeneracy, a first in online optimization.
Unlike standard methods for computing NE, our proposed approach can be
parallelized and works with arbitrary learning rates, both firsts in
algorithmic game theory. Experimentally, we support our results by showing our
approach drastically outperforms standard methods.

### 4. [Better Regret Rates in Bilateral Trade via Sublinear Budget Violation](http://arxiv.org/pdf/2507.11419v1)

Authors: Anna Lunghi, Matteo Castiglioni, Alberto Marchesi

Bilateral trade is a central problem in algorithmic economics, and recent
work has explored how to design trading mechanisms using no-regret learning
algorithms. However, no-regret learning is impossible when budget balance has
to be enforced at each time step. Bernasconi et al. [Ber+24] show how this
impossibility can be circumvented by relaxing the budget balance constraint to
hold only globally over all time steps. In particular, they design an algorithm
achieving regret of the order of $\tilde O(T^{3/4})$ and provide a lower bound
of $\Omega(T^{5/7})$.
  In this work, we interpolate between these two extremes by studying how the
optimal regret rate varies with the allowed violation of the global budget
balance constraint. Specifically, we design an algorithm that, by violating the
constraint by at most $T^{\beta}$ for any given $\beta \in [\frac{3}{4},
\frac{6}{7}]$, attains regret $\tilde O(T^{1 - \beta/3})$. We complement this
result with a matching lower bound, thus fully characterizing the trade-off
between regret and budget violation. Our results show that both the $\tilde
O(T^{3/4})$ upper bound in the global budget balance case and the
$\Omega(T^{5/7})$ lower bound under unconstrained budget balance violation
obtained by Bernasconi et al. [Ber+24] are tight.

### 5. [On the Complexity of the Optimal Correlated Equilibria in Extensive-Form Games](http://arxiv.org/pdf/2507.11509v1)

Authors: Vincent Cheval, Florian Horn, Soumyajit Paul, Mahsa Shirmohammadi

A major open question in algorithmic game theory is whether normal-form
correlated equilibria (NFCE) can be computed efficiently in succinct games such
as extensive-form games [DFF+25,6PR24,FP23,HvS08,VSF08,PR08]. Motivated by this
question, we study the associated Threshold problem: deciding whether there
exists a correlated equilibrium whose value exceeds a given threshold. We prove
that this problem is PSPACE-hard for NFCE in multiplayer extensive-form games
with perfect recall, even for fixed thresholds. To contextualize this result,
we also establish the complexity of the Threshold problem for Nash equilibria
in this setting, showing it is ER-complete. These results uncover a surprising
complexity reversal: while optimal correlated equilibria are computationally
simpler than optimal Nash in normal-form games, the opposite holds in
extensive-form games, where computing optimal correlated equilibria is provably
harder. Building on this line of inquiry, we also address a related question by
[VSF08], who introduced the notions of extensive-form correlated equilibrium
(EFCE) and agent-form correlated equilibrium (AFCE). They asked how difficult
the Threshold problem is for AFCE; we answer this question by proving that it
is NP-hard, even in two-player games without chance nodes. Complementing our
hardness results, we establish tight complexity classifications for the
Threshold problem across several correlated equilibrium concepts - including
EFCE, AFCE, normal-form coarse, extensive-form coarse, and agent-form coarse
correlated equilibria. For each of these solution concepts in multiplayer
stochastic extensive-form games with perfect recall, we prove NP-completeness
by providing matching NP upper bounds to the previously known hardness results.
Together, our results provide the most complete landscape to date for the
complexity of optimal equilibrium computation in extensive-form games.

### 6. [Approximate solutions to games of ordered preference](http://arxiv.org/pdf/2507.11021v1)

Authors: Pau de las Heras Molins, Eric Roy-Almonacid, Dong Ho Lee, Lasse Peters, David Fridovich-Keil, Georgios Bakirtzis

Autonomous vehicles must balance ranked objectives, such as minimizing travel
time, ensuring safety, and coordinating with traffic. Games of ordered
preference effectively model these interactions but become computationally
intractable as the time horizon, number of players, or number of preference
levels increase. While receding horizon frameworks mitigate long-horizon
intractability by solving sequential shorter games, often warm-started, they do
not resolve the complexity growth inherent in existing methods for solving
games of ordered preference. This paper introduces a solution strategy that
avoids excessive complexity growth by approximating solutions using
lexicographic iterated best response (IBR) in receding horizon, termed
"lexicographic IBR over time." Lexicographic IBR over time uses past
information to accelerate convergence. We demonstrate through simulated traffic
scenarios that lexicographic IBR over time efficiently computes
approximate-optimal solutions for receding horizon games of ordered preference,
converging towards generalized Nash equilibria.

### Human-Computer Interaction

### 1. [AROMA: Mixed-Initiative AI Assistance for Non-Visual Cooking by Grounding Multi-modal Information Between Reality and Videos](http://arxiv.org/pdf/2507.10963v1)

Authors: Zheng Ning, Leyang Li, Daniel Killough, JooYoung Seo, Patrick Carrington, Yapeng Tian, Yuhang Zhao, Franklin Mingzhe Li, Toby Jia-Jun Li

Videos offer rich audiovisual information that can support people in
performing activities of daily living (ADLs), but they remain largely
inaccessible to blind or low-vision (BLV) individuals. In cooking, BLV people
often rely on non-visual cues, such as touch, taste, and smell, to navigate
their environment, making it difficult to follow the predominantly audiovisual
instructions found in video recipes. To address this problem, we introduce
AROMA, an AI system that provides timely responses to the user based on
real-time, context-aware assistance by integrating non-visual cues perceived by
the user, a wearable camera feed, and video recipe content. AROMA uses a
mixed-initiative approach: it responds to user requests while also proactively
monitoring the video stream to offer timely alerts and guidance. This
collaborative design leverages the complementary strengths of the user and AI
system to align the physical environment with the video recipe, helping the
user interpret their current cooking state and make sense of the steps. We
evaluated AROMA through a study with eight BLV participants and offered
insights for designing interactive AI systems to support BLV individuals in
performing ADLs.

### 2. [Self++: Merging Human and AI for Co-Determined XR Living in the Metaverse](http://arxiv.org/pdf/2507.10967v1)

Authors: Thammathip Piumsomboon

This position paper introduces Self++, a novel nine-level framework for
co-determined living in the Metaverse, grounded in Self-Determination Theory.
Self++ prioritises human flourishing by progressively cultivating competence,
autonomy, and relatedness through dynamic human-AI collaboration in extended
reality (XR). Unlike technologically deterministic approaches, Self++
emphasises user empowerment by enhancing competency, mitigating cognitive
biases and leveraging XR's immersive capabilities. Key research directions
proposed include exploring the boundaries of user-defined AI autonomy,
designing for meaningful social connection in XR, and establishing proactive
ethical safeguards. Ultimately, Self++ offers a roadmap for creating a
human-centred, AI-enhanced Metaverse where technology amplifies, rather than
diminishes, human potential.

### 3. [Terms and Conditions (Do Not) Apply: Understanding Exploitation Disparities in Design of Mobile-Based Financial Services](http://arxiv.org/pdf/2507.10970v1)

Authors: Lindah Kotut

Mobile-based financial services have made it possible for the traditionally
unbanked to access infrastructure that have been routinely unattainable.
Researchers have explored how these systems have made for safer environments to
send and receive money and have expanded financial opportunities such as
increased borrowing. With this expansion, challenges such as detrimental
interest rates, lack of access to policy documents, and inadequate user
protective guardrails emerge, amplifying the risks due to technology-aided
unethical financial practices that are aided by design patterns. Supported by
user interviews, we detail user experiences of mobile-based financial
transactions and explore the foundations and guidelines that undergird the
financial service provisions: highlighting both affordances and harms enabled
in the design of such systems. We discuss the findings by highlighting
financial exploitation disparities, deliberating strategies for mitigation of
risks and enabling recovery from harms caused by the technology use. We then
recommend guidelines for empowering design approaches that support users'
mechanisms of trust, their understanding of technological processes, and
determination of risks.

### 4. [An Exploratory Study on AI-driven Visualisation Techniques on Decision Making in Extended Reality](http://arxiv.org/pdf/2507.10981v1)

Authors: Ze Dong, Binyang Han, Jingjing Zhang, Ruoyu Wen, Barrett Ens, Adrian Clark, Tham Piumsomboon

The integration of extended reality (XR) with artificial intelligence (AI)
introduces a new paradigm for user interaction, enabling AI to perceive user
intent, stimulate the senses, and influence decision-making. We explored the
impact of four AI-driven visualisation techniques -- `Inform,' `Nudge,'
`Recommend,' and `Instruct' -- on user decision-making in XR using the Meta
Quest Pro. To test these techniques, we used a pre-recorded 360-degree video of
a supermarket, overlaying each technique through a virtual interface. We aimed
to investigate how these different visualisation techniques with different
levels of user autonomy impact preferences and decision-making. An exploratory
study with semi-structured interviews provided feedback and design
recommendations. Our findings emphasise the importance of maintaining user
autonomy, enhancing AI transparency to build trust, and considering context in
visualisation design.

### 5. [REVA: Supporting LLM-Generated Programming Feedback Validation at Scale Through User Attention-based Adaptation](http://arxiv.org/pdf/2507.11470v1)

Authors: Xiaohang Tang, Sam Wong, Zicheng He, Yalong Yang, Yan Chen

This paper introduces REVA, a human-AI system that expedites instructor
review of voluminous AI-generated programming feedback by sequencing
submissions to minimize cognitive context shifts and propagating
instructor-driven revisions across semantically similar instances. REVA
introduces a novel approach to human-AI collaboration in educational feedback
by adaptively learning from instructors' attention in the review and revision
process to continuously improve the feedback validation process. REVA's
usefulness and effectiveness in improving feedback quality and the overall
feedback review process were evaluated through a within-subjects lab study with
12 participants.

### 6. [Towards Creating Infrastructures for Values and Ethics Work in the Production of Software Technologies](http://arxiv.org/pdf/2507.11490v1)

Authors: Richmond Y. Wong

Recognizing how technical systems can embody social values or cause harms,
human-computer interaction (HCI) research often approaches addressing values
and ethics in design by creating tools to help tech workers integrate social
values into the design of products. While useful, these approaches usually do
not consider the politics embedded in the broader processes, organizations,
social systems, and governance structures that affect the types of actions that
tech workers can take to address values and ethics. This paper argues that
creating infrastructures to support values and ethics work, rather than tools,
is an approach that takes these broader processes into account and opens them
up for (re)design. Drawing on prior research conceptualizing infrastructures
from science \& technology studies and media studies, this paper outlines
conceptual insights from infrastructures studies that open up new tactics for
HCI researchers and designers seeking to support values and ethics in design.

### 7. [Developing and evaluating quilts for the depiction of large layered graphs](http://arxiv.org/pdf/2507.10883v1)

Authors: Juhee Bae, Benjamin Watson

Traditional layered graph depictions such as flow charts are in wide use. Yet
as graphs grow more complex, these depictions can become difficult to
understand. Quilts are matrix-based depictions for layered graphs designed to
address this problem. In this research, we first improve Quilts by developing
three design alternatives, and then compare the best of these alternatives to
better-known node-link and matrix depictions. A primary weakness in Quilts is
their depiction of skip links, links that do not simply connect to a succeeding
layer. Therefore in our first study, we compare Quilts using color-only,
text-only, and mixed (color and text) skip link depictions, finding that path
finding with the color-only depiction is significantly slower and less
accurate, and that in certain cases, the mixed depiction offers an advantage
over the text-only depiction. In our second study, we compare Quilts using the
mixed depiction to node-link diagrams and centered matrices. Overall results
show that users can find paths through graphs significantly faster with Quilts
(46.6 secs) than with node-link (58.3 secs) or matrix (71.2 secs) diagrams.
This speed advantage is still greater in large graphs (e.g. in 200 node graphs,
55.4 secs vs. 71.1 secs for node-link and 84.2 secs for matrix depictions).

### 8. [Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias](http://arxiv.org/pdf/2507.11210v1)

Authors: Rushia Harada, Yuken Kimura, Keito Inoshita

Well-being in family settings involves subtle psychological dynamics that
conventional metrics often overlook. In particular, unconscious parental
expectations, termed ideal parent bias, can suppress children's emotional
expression and autonomy. This suppression, referred to as suppressed emotion,
often stems from well-meaning but value-driven communication, which is
difficult to detect or address from outside the family. Focusing on these
latent dynamics, this study explores Large Language Model (LLM)-based support
for psychologically safe family communication. We constructed a Japanese
parent-child dialogue corpus of 30 scenarios, each annotated with metadata on
ideal parent bias and suppressed emotion. Based on this corpus, we developed a
Role-Playing LLM-based multi-agent dialogue support framework that analyzes
dialogue and generates feedback. Specialized agents detect suppressed emotion,
describe implicit ideal parent bias in parental speech, and infer contextual
attributes such as the child's age and background. A meta-agent compiles these
outputs into a structured report, which is then passed to five selected expert
agents. These agents collaboratively generate empathetic and actionable
feedback through a structured four-step discussion process. Experiments show
that the system can detect categories of suppressed emotion with moderate
accuracy and produce feedback rated highly in empathy and practicality.
Moreover, simulated follow-up dialogues incorporating this feedback exhibited
signs of improved emotional expression and mutual understanding, suggesting the
framework's potential in supporting positive transformation in family
interactions.

### 9. [Human-Robot collaboration in surgery: Advances and challenges towards autonomous surgical assistants](http://arxiv.org/pdf/2507.11460v1)

Authors: Jacinto Colan, Ana Davila, Yutaro Yamada, Yasuhisa Hasegawa

Human-robot collaboration in surgery represents a significant area of
research, driven by the increasing capability of autonomous robotic systems to
assist surgeons in complex procedures. This systematic review examines the
advancements and persistent challenges in the development of autonomous
surgical robotic assistants (ASARs), focusing specifically on scenarios where
robots provide meaningful and active support to human surgeons. Adhering to the
PRISMA guidelines, a comprehensive literature search was conducted across the
IEEE Xplore, Scopus, and Web of Science databases, resulting in the selection
of 32 studies for detailed analysis. Two primary collaborative setups were
identified: teleoperation-based assistance and direct hands-on interaction. The
findings reveal a growing research emphasis on ASARs, with predominant
applications currently in endoscope guidance, alongside emerging progress in
autonomous tool manipulation. Several key challenges hinder wider adoption,
including the alignment of robotic actions with human surgeon preferences, the
necessity for procedural awareness within autonomous systems, the establishment
of seamless human-robot information exchange, and the complexities of skill
acquisition in shared workspaces. This review synthesizes current trends,
identifies critical limitations, and outlines future research directions
essential to improve the reliability, safety, and effectiveness of human-robot
collaboration in surgical environments.

### 10. [LLM-based ambiguity detection in natural language instructions for collaborative surgical robots](http://arxiv.org/pdf/2507.11525v1)

Authors: Ana Davila, Jacinto Colan, Yasuhisa Hasegawa

Ambiguity in natural language instructions poses significant risks in
safety-critical human-robot interaction, particularly in domains such as
surgery. To address this, we propose a framework that uses Large Language
Models (LLMs) for ambiguity detection specifically designed for collaborative
surgical scenarios. Our method employs an ensemble of LLM evaluators, each
configured with distinct prompting techniques to identify linguistic,
contextual, procedural, and critical ambiguities. A chain-of-thought evaluator
is included to systematically analyze instruction structure for potential
issues. Individual evaluator assessments are synthesized through conformal
prediction, which yields non-conformity scores based on comparison to a labeled
calibration dataset. Evaluating Llama 3.2 11B and Gemma 3 12B, we observed
classification accuracy exceeding 60% in differentiating ambiguous from
unambiguous surgical instructions. Our approach improves the safety and
reliability of human-robot collaboration in surgery by offering a mechanism to
identify potentially ambiguous instructions before robot action.

### Information Retrieval

### 1. [LLM-Driven Dual-Level Multi-Interest Modeling for Recommendation](http://arxiv.org/pdf/2507.10917v1)

Authors: Ziyan Wang, Yingpeng Du, Zhu Sun, Jieyi Bi, Haoyan Chua, Tianjun Wei, Jie Zhang

Recently, much effort has been devoted to modeling users' multi-interests
based on their behaviors or auxiliary signals. However, existing methods often
rely on heuristic assumptions, e.g., co-occurring items indicate the same
interest of users, failing to capture user multi-interests aligning with
real-world scenarios. While large language models (LLMs) show significant
potential for multi-interest analysis due to their extensive knowledge and
powerful reasoning capabilities, two key challenges remain. First, the
granularity of LLM-driven multi-interests is agnostic, possibly leading to
overly fine or coarse interest grouping. Second, individual user analysis
provides limited insights due to the data sparsity issue. In this paper, we
propose an LLM-driven dual-level multi-interest modeling framework for more
effective recommendation. At the user-individual level, we exploit LLMs to
flexibly allocate items engaged by users into different semantic clusters,
indicating their diverse and distinct interests. To alleviate the agnostic
generation of LLMs, we adaptively assign these semantic clusters to users'
collaborative multi-interests learned from global user-item interactions,
allowing the granularity to be automatically adjusted according to the user's
behaviors using an alignment module. To alleviate the limited insights derived
from individual users' behaviors, at the user-crowd level, we propose
aggregating user cliques into synthesized users with rich behaviors for more
comprehensive LLM-driven multi-interest analysis. We formulate a max covering
problem to ensure the compactness and representativeness of synthesized users'
behaviors, and then conduct contrastive learning based on their LLM-driven
multi-interests to disentangle item representations among different interests.
Experiments on real-world datasets show the superiority of our approach against
state-of-the-art methods.

### 2. [Aligned Query Expansion: Efficient Query Expansion for Information Retrieval through LLM Alignment](http://arxiv.org/pdf/2507.11042v1)

Authors: Adam Yang, Gustavo Penha, Enrico Palumbo, Hugues Bouchard

With the breakthroughs in large language models (LLMs), query generation
techniques that expand documents and queries with related terms are becoming
increasingly popular in the information retrieval field. Such techniques have
been shown to improve the effectiveness of traditional lexical retrieval
methods by dealing with the vocabulary mismatch problem. Recent work has found
that generating queries with a greedy decoding strategy can produce sub-optimal
queries, including hallucinations, and proposed to filter out queries before
expansion. This `generate-then-filter' approach is costly, as it requires
generating multiple queries and applying a relevance model to all of them and
does not teach the LLM which of the generated queries is more effective for
expansion. To overcome such limitations, we propose Aligned Query Expansion
(AQE), a novel approach to enhance query expansion for passage retrieval in
open-domain question answering. AQE leverages recent techniques in LLM
alignment to fine-tune models for generating query expansions that directly
optimize the effectiveness of the retrieval task, eliminating the need for
additional filtering steps. This alignment ensures that queries are more
relevant, reducing computational costs while improving retrieval effectiveness.
Empirical evaluations show that AQE outperforms baseline models for query
expansion in both in-domain and out-of-domain settings, demonstrating
significant improvements in retrieval effectiveness.

### 3. [Unraveling the Biomarker Prospects of High-Altitude Diseases: Insights from Biomolecular Event Network Constructed using Text Mining](http://arxiv.org/pdf/2507.10953v1)

Authors: Balu Bhasuran, Sabenabanu Abdulkadhar, Jeyakumar Natarajan

High-altitude diseases (HAD), encompassing acute mountain sickness (AMS),
high-altitude cerebral edema (HACE), and high-altitude pulmonary edema (HAPE),
are triggered by hypobaric hypoxia at elevations above 2,500 meters. These
conditions pose significant health risks, yet the molecular mechanisms remain
insufficiently understood. In this study, we developed a biomolecular event
extraction pipeline integrating supervised machine learning with feature-based
and multiscale Laplacian graph kernels to analyze 7,847 curated HAD-related
abstracts from PubMed. We extracted over 150 unique biomolecular events
including gene expression, regulation, binding, and localization and
constructed a weighted, undirected biomolecular event network comprising 97
nodes and 153 edges. Using the PageRank algorithm, we prioritized key
biomolecules based on their centrality within the event network. The top-ranked
proteins included Erythropoietin (EPO) (0.0163), Vascular endothelial growth
factor (VEGF) (0.0148), Hypoxia-inducible factor 1 (HIF-1) alpha (0.0136),
Endothelial PAS Domain Protein 1 (EPAS1) and Angiotensin-Converting Enzyme
(ACE) (0.0119), Egl nine homolog 1 (EGLN1), Endothelin 1 (ET-1), and 70
kilodalton heat shock protein (Hsp70)(0.0118), all of which play crucial roles
in oxygen sensing, vascular remodeling, erythropoiesis, and blood pressure
regulation. Subnetwork analysis revealed three major functional clusters
centered on hypoxia response, inflammation, and stress adaptation pathways. Our
integrative approach demonstrates the utility of large-scale text mining and
graph-based analysis to uncover mechanistic insights and prioritize potential
biomarkers for high-altitude disease.

### 4. [From Chaos to Automation: Enabling the Use of Unstructured Data for Robotic Process Automation](http://arxiv.org/pdf/2507.11364v1)

Authors: Kelly Kurowski, Xixi Lu, Hajo A. Reijers

The growing volume of unstructured data within organizations poses
significant challenges for data analysis and process automation. Unstructured
data, which lacks a predefined format, encompasses various forms such as
emails, reports, and scans. It is estimated to constitute approximately 80% of
enterprise data. Despite the valuable insights it can offer, extracting
meaningful information from unstructured data is more complex compared to
structured data. Robotic Process Automation (RPA) has gained popularity for
automating repetitive tasks, improving efficiency, and reducing errors.
However, RPA is traditionally reliant on structured data, limiting its
application to processes involving unstructured documents. This study addresses
this limitation by developing the UNstructured Document REtrieval SyStem
(UNDRESS), a system that uses fuzzy regular expressions, techniques for natural
language processing, and large language models to enable RPA platforms to
effectively retrieve information from unstructured documents. The research
involved the design and development of a prototype system, and its subsequent
evaluation based on text extraction and information retrieval performance. The
results demonstrate the effectiveness of UNDRESS in enhancing RPA capabilities
for unstructured data, providing a significant advancement in the field. The
findings suggest that this system could facilitate broader RPA adoption across
processes traditionally hindered by unstructured data, thereby improving
overall business process efficiency.

### 5. [Seq vs Seq: An Open Suite of Paired Encoders and Decoders](http://arxiv.org/pdf/2507.11412v1)

Authors: Orion Weller, Kathryn Ricci, Marc Marone, Antoine Chaffin, Dawn Lawrie, Benjamin Van Durme

The large language model (LLM) community focuses almost exclusively on
decoder-only language models, since they are easier to use for text generation.
However, a large subset of the community still uses encoder-only models for
tasks such as classification or retrieval. Previous work has attempted to
compare these architectures, but is forced to make comparisons with models that
have different numbers of parameters, training techniques, and datasets. We
introduce the SOTA open-data Ettin suite of models: paired encoder-only and
decoder-only models ranging from 17 million parameters to 1 billion, trained on
up to 2 trillion tokens. Using the same recipe for both encoder-only and
decoder-only models produces SOTA recipes in both categories for their
respective sizes, beating ModernBERT as an encoder and Llama 3.2 and SmolLM2 as
decoders. Like previous work, we find that encoder-only models excel at
classification and retrieval tasks while decoders excel at generative tasks.
However, we show that adapting a decoder model to encoder tasks (and vice
versa) through continued training is subpar compared to using only the reverse
objective (i.e. a 400M encoder outperforms a 1B decoder on MNLI, and vice versa
for generative tasks). We open-source all artifacts of this study including
training data, training order segmented by checkpoint, and 200+ checkpoints to
allow future work to analyze or extend all aspects of training.

### Machine Learning

### 1. [Outbound Modeling for Inventory Management](http://arxiv.org/pdf/2507.10890v1)

Authors: Riccardo Savorgnan, Udaya Ghai, Carson Eisenach, Dean Foster

We study the problem of forecasting the number of units fulfilled (or
``drained'') from each inventory warehouse to meet customer demand, along with
the associated outbound shipping costs. The actual drain and shipping costs are
determined by complex production systems that manage the planning and execution
of customers' orders fulfillment, i.e. from where and how to ship a unit to be
delivered to a customer. Accurately modeling these processes is critical for
regional inventory planning, especially when using Reinforcement Learning (RL)
to develop control policies. For the RL usecase, a drain model is incorporated
into a simulator to produce long rollouts, which we desire to be
differentiable. While simulating the calls to the internal software systems can
be used to recover this transition, they are non-differentiable and too slow
and costly to run within an RL training environment. Accordingly, we frame this
as a probabilistic forecasting problem, modeling the joint distribution of
outbound drain and shipping costs across all warehouses at each time period,
conditioned on inventory positions and exogenous customer demand. To ensure
robustness in an RL environment, the model must handle out-of-distribution
scenarios that arise from off-policy trajectories. We propose a validation
scheme that leverages production systems to evaluate the drain model on
counterfactual inventory states induced by RL policies. Preliminary results
demonstrate the model's accuracy within the in-distribution setting.

### 2. [Diffusion Decoding for Peptide De Novo Sequencing](http://arxiv.org/pdf/2507.10955v1)

Authors: Chi-en Amy Tai, Alexander Wong

Peptide de novo sequencing is a method used to reconstruct amino acid
sequences from tandem mass spectrometry data without relying on existing
protein sequence databases. Traditional deep learning approaches, such as
Casanovo, mainly utilize autoregressive decoders and predict amino acids
sequentially. Subsequently, they encounter cascading errors and fail to
leverage high-confidence regions effectively. To address these issues, this
paper investigates using diffusion decoders adapted for the discrete data
domain. These decoders provide a different approach, allowing sequence
generation to start from any peptide segment, thereby enhancing prediction
accuracy. We experiment with three different diffusion decoder designs,
knapsack beam search, and various loss functions. We find knapsack beam search
did not improve performance metrics and simply replacing the transformer
decoder with a diffusion decoder lowered performance. Although peptide
precision and recall were still 0, the best diffusion decoder design with the
DINOISER loss function obtained a statistically significant improvement in
amino acid recall by 0.373 compared to the baseline autoregressive
decoder-based Casanovo model. These findings highlight the potential of
diffusion decoders to not only enhance model sensitivity but also drive
significant advancements in peptide de novo sequencing.

### 3. [Physics-Informed Neural Networks For Semiconductor Film Deposition: A Review](http://arxiv.org/pdf/2507.10983v1)

Authors: Tao Han, Zahra Taheri, Hyunwoong Ko

Semiconductor manufacturing relies heavily on film deposition processes, such
as Chemical Vapor Deposition and Physical Vapor Deposition. These complex
processes require precise control to achieve film uniformity, proper adhesion,
and desired functionality. Recent advancements in Physics-Informed Neural
Networks (PINNs), an innovative machine learning (ML) approach, have shown
significant promise in addressing challenges related to process control,
quality assurance, and predictive modeling within semiconductor film deposition
and other manufacturing domains. This paper provides a comprehensive review of
ML applications targeted at semiconductor film deposition processes. Through a
thematic analysis, we identify key trends, existing limitations, and research
gaps, offering insights into both the advantages and constraints of current
methodologies. Our structured analysis aims to highlight the potential
integration of these ML techniques to enhance interpretability, accuracy, and
robustness in film deposition processes. Additionally, we examine
state-of-the-art PINN methods, discussing strategies for embedding physical
knowledge, governing laws, and partial differential equations into advanced
neural network architectures tailored for semiconductor manufacturing. Based on
this detailed review, we propose novel research directions that integrate the
strengths of PINNs to significantly advance film deposition processes. The
contributions of this study include establishing a clear pathway for future
research in integrating physics-informed ML frameworks, addressing existing
methodological gaps, and ultimately improving precision, scalability, and
operational efficiency within semiconductor manufacturing.

### 4. [StellarF: A Lora-Adapter Integrated Large Model Framework for Stellar Flare Forecasting with Historical & Statistical Data](http://arxiv.org/pdf/2507.10986v1)

Authors: Tianyu Su, Zhiqiang Zou, Ali Luo, Xiao Kong, Qingyu Lu, Min Li

Stellar flare forecasting, a critical research frontier in astronomy, offers
profound insights into stellar activity. However, the field is constrained by
both the sparsity of recorded flare events and the absence of domain-specific
large-scale predictive models. To address these challenges, this study
introduces StellarF (Stellar Flare Forecasting), a novel large model that
leverages Low-Rank (LoRA) and Adapter techniques to parameter-efficient
learning for stellar flare forecasting. At its core, StellarF integrates an
flare statistical information module with a historical flare record module,
enabling multi-scale pattern recognition from observational data. Extensive
experiments on our self-constructed datasets (derived from Kepler and TESS
light curves) demonstrate that StellarF achieves state-of-the-art performance
compared to existing methods. The proposed prediction paradigm establishes a
novel methodological framework for advancing astrophysical research and
cross-disciplinary applications.

### 5. [AdaMuon: Adaptive Muon Optimizer](http://arxiv.org/pdf/2507.11005v1)

Authors: Chongjie Si, Debing Zhang, Wei Shen

We propose AdaMuon, an adaptive learning-rate framework built upon the
recently validated Muon optimizer, which has demonstrated substantial
efficiency gains over AdamW in large-scale model training. AdaMuon augments
Muon with two mutually dependent modules: (1) a per-parameter second-moment
modulation that captures orthogonal gradient updates to ensure update-level
adaptivity, and (2) a RMS-aligned rescaling that regulates the overall update
magnitude by aligning it with the intrinsic structure of the parameter space.
Empirical results on multiple model scales and learning-rate regimes confirm
that AdaMuon consistently outperforms the original Muon, delivering higher
acceleration in convergence while maintaining training stability. Our method
introduces no additional tuning burden and can be seamlessly integrated into
existing Muon training pipelines.

### 6. [Leveraging Advanced Machine Learning to Predict Turbulence Dynamics from Temperature Observations at an Experimental Prescribed Fire](http://arxiv.org/pdf/2507.11012v1)

Authors: Dipak Dulal, Joseph J. Charney, Michael R. Gallagher, Pitambar Acharya, Carmeliza Navasca, Nicholas S. Skowronski

This study explores the potential for predicting turbulent kinetic energy
(TKE) from more readily acquired temperature data using temperature profiles
and turbulence data collected concurrently at 10 Hz during a small experimental
prescribed burn in the New Jersey Pine Barrens. Machine learning models,
including Deep Neural Networks, Random Forest Regressor, Gradient Boosting, and
Gaussian Process Regressor, were employed to assess the potential to predict
TKE from temperature perturbations and explore temporal and spatial dynamics of
correlations. Data visualization and correlation analyses revealed patterns and
relationships between thermocouple temperatures and TKE, providing insight into
the underlying dynamics. More accurate predictions of TKE were achieved by
employing various machine learning models despite a weak correlation between
the predictors and the target variable. The results demonstrate significant
success, particularly from regression models, in accurately predicting the TKE.
The findings of this study demonstrate a novel numerical approach to
identifying new relationships between temperature and airflow processes in and
around the fire environment. These relationships can help refine our
understanding of combustion environment processes and the coupling and
decoupling of fire environment processes necessary for improving fire
operations strategy and fire and smoke model predictions. The findings of this
study additionally highlight the valuable role of machine learning techniques
in analyzing the complex large datasets of the fire environments, showcasing
their potential to advance fire research and management practices.

### 7. [Relative Entropy Pathwise Policy Optimization](http://arxiv.org/pdf/2507.11019v1)

Authors: Claas Voelcker, Axel Brunnbauer, Marcel Hussing, Michal Nauman, Pieter Abbeel, Eric Eaton, Radu Grosu, Amir-massoud Farahmand, Igor Gilitschenski

Score-function policy gradients have delivered strong results in
game-playing, robotics and language-model fine-tuning. Yet its high-variance
often undermines training stability. On the other hand, pathwise policy
gradients alleviate the training variance, but are reliable only when driven by
an accurate action-conditioned value function which is notoriously hard to
train without relying on past off-policy data. In this paper, we discuss how to
construct a value-gradient driven, on-policy algorithm that allow training
Q-value models purely from on-policy data, unlocking the possibility of using
pathwise policy updates in the context of on-policy learning. We show how to
balance stochastic policies for exploration with constrained policy updates for
stable training, and evaluate important architectural components that
facilitate accurate value function learning. Building on these insights, we
propose Relative Entropy Pathwise Policy Optimization (REPPO), an efficient
on-policy algorithm that combines the sample-efficiency of pathwise policy
gradients with the simplicity and minimal memory footprint of standard
on-policy learning. We demonstrate that REPPO provides strong empirical
performance at decreased sample requirements, wall-clock time, memory footprint
as well as high hyperparameter robustness in a set of experiments on two
standard GPU-parallelized benchmarks.

### 8. [A Distance Metric for Mixed Integer Programming Instances](http://arxiv.org/pdf/2507.11063v1)

Authors: Gwen Maudet, Grégoire Danoy

Mixed-integer linear programming (MILP) is a powerful tool for addressing a
wide range of real-world problems, but it lacks a clear structure for comparing
instances. A reliable similarity metric could establish meaningful
relationships between instances, enabling more effective evaluation of instance
set heterogeneity and providing better guidance to solvers, particularly when
machine learning is involved. Existing similarity metrics often lack precision
in identifying instance classes or rely heavily on labeled data, which limits
their applicability and generalization. To bridge this gap, this paper
introduces the first mathematical distance metric for MILP instances, derived
directly from their mathematical formulations. By discretizing right-hand
sides, weights, and variables into classes, the proposed metric draws
inspiration from the Earth mover's distance to quantify mismatches in
weight-variable distributions for constraint comparisons. This approach
naturally extends to enable instance-level comparisons. We evaluate both an
exact and a greedy variant of our metric under various parameter settings,
using the StrIPLIB dataset. Results show that all components of the metric
contribute to class identification, and that the greedy version achieves
accuracy nearly identical to the exact formulation while being nearly 200 times
faster. Compared to state-of-the-art baselines, including feature-based,
image-based, and neural network models, our unsupervised method consistently
outperforms all non-learned approaches and rivals the performance of a
supervised classifier on class and subclass grouping tasks.

### 9. [Real-Time Bayesian Detection of Drift-Evasive GNSS Spoofing in Reinforcement Learning Based UAV Deconfliction](http://arxiv.org/pdf/2507.11173v1)

Authors: Deepak Kumar Panda, Weisi Guo

Autonomous unmanned aerial vehicles (UAVs) rely on global navigation
satellite system (GNSS) pseudorange measurements for accurate real-time
localization and navigation. However, this dependence exposes them to
sophisticated spoofing threats, where adversaries manipulate pseudoranges to
deceive UAV receivers. Among these, drift-evasive spoofing attacks subtly
perturb measurements, gradually diverting the UAVs trajectory without
triggering conventional signal-level anti-spoofing mechanisms. Traditional
distributional shift detection techniques often require accumulating a
threshold number of samples, causing delays that impede rapid detection and
timely response. Consequently, robust temporal-scale detection methods are
essential to identify attack onset and enable contingency planning with
alternative sensing modalities, improving resilience against stealthy
adversarial manipulations. This study explores a Bayesian online change point
detection (BOCPD) approach that monitors temporal shifts in value estimates
from a reinforcement learning (RL) critic network to detect subtle behavioural
deviations in UAV navigation. Experimental results show that this temporal
value-based framework outperforms conventional GNSS spoofing detectors,
temporal semi-supervised learning frameworks, and the Page-Hinkley test,
achieving higher detection accuracy and lower false-positive and false-negative
rates for drift-evasive spoofing attacks.

### 10. [Quantized Rank Reduction: A Communications-Efficient Federated Learning Scheme for Network-Critical Applications](http://arxiv.org/pdf/2507.11183v1)

Authors: Dimitrios Kritsiolis, Constantine Kotropoulos

Federated learning is a machine learning approach that enables multiple
devices (i.e., agents) to train a shared model cooperatively without exchanging
raw data. This technique keeps data localized on user devices, ensuring privacy
and security, while each agent trains the model on their own data and only
shares model updates. The communication overhead is a significant challenge due
to the frequent exchange of model updates between the agents and the central
server. In this paper, we propose a communication-efficient federated learning
scheme that utilizes low-rank approximation of neural network gradients and
quantization to significantly reduce the network load of the decentralized
learning process with minimal impact on the model's accuracy.

### Neural and Evolutionary Computing

### 1. [Biological Processing Units: Leveraging an Insect Connectome to Pioneer Biofidelic Neural Architectures](http://arxiv.org/pdf/2507.10951v1)

Authors: Siyu Yu, Zihan Qin, Tingshan Liu, Beiya Xu, R. Jacob Vogelstein, Jason Brown, Joshua T. Vogelstein

The complete connectome of the Drosophila larva brain offers a unique
opportunity to investigate whether biologically evolved circuits can support
artificial intelligence. We convert this wiring diagram into a Biological
Processing Unit (BPU), a fixed recurrent network derived directly from synaptic
connectivity. Despite its modest size 3,000 neurons and 65,000 weights between
them), the unmodified BPU achieves 98% accuracy on MNIST and 58% on CIFAR-10,
surpassing size-matched MLPs. Scaling the BPU via structured connectome
expansions further improves CIFAR-10 performance, while modality-specific
ablations reveal the uneven contributions of different sensory subsystems. On
the ChessBench dataset, a lightweight GNN-BPU model trained on only 10,000
games achieves 60% move accuracy, nearly 10x better than any size transformer.
Moreover, CNN-BPU models with ~2M parameters outperform parameter-matched
Transformers, and with a depth-6 minimax search at inference, reach 91.7%
accuracy, exceeding even a 9M-parameter Transformer baseline. These results
demonstrate the potential of biofidelic neural architectures to support complex
cognitive tasks and motivate scaling to larger and more intelligent connectomes
in future work.

### Networking and Internet Architecture

### 1. [SIMCODE: A Benchmark for Natural Language to ns-3 Network Simulation Code Generation](http://arxiv.org/pdf/2507.11014v1)

Authors: Tasnim Ahmed, Mirza Mohammad Azwad, Salimur Choudhury

Large language models (LLMs) have demonstrated remarkable capabilities in
code generation across various domains. However, their effectiveness in
generating simulation scripts for domain-specific environments like ns-3
remains underexplored. Despite the growing interest in automating network
simulations, existing tools primarily focus on interactive automation over
rigorous evaluation. To facilitate systematic evaluation, we introduce SIMCODE,
the first benchmark to evaluate LLMs' ability to generate ns-3 simulation code
from natural language. SIMCODE includes 400 tasks across introductory,
intermediate, and advanced levels, with solutions and test cases. Using
SIMCODE, we evaluate three prominent LLMs, Gemini-2.0, GPT-4.1, and Qwen-3,
across six prompt techniques. Furthermore, investigating task-specific
fine-tuning's impact reveals that while GPT-4.1 outperforms others, execution
accuracy remains modest, with substantial room for improvement. Error analysis
identifies missing headers and API mismatches as dominant failures.
Nevertheless, SIMCODE provides a foundational step toward evaluating LLMs and
research in domain-aware generative systems.

### 2. [Graph-based Fingerprint Update Using Unlabelled WiFi Signals](http://arxiv.org/pdf/2507.11038v1)

Authors: Ka Ho Chiu, Handi Yin, Weipeng Zhuo, Chul-Ho Lee, S. -H. Gary Chan

WiFi received signal strength (RSS) environment evolves over time due to
movement of access points (APs), AP power adjustment, installation and removal
of APs, etc. We study how to effectively update an existing database of
fingerprints, defined as the RSS values of APs at designated locations, using a
batch of newly collected unlabelled (possibly crowdsourced) WiFi signals. Prior
art either estimates the locations of the new signals without updating the
existing fingerprints or filters out the new APs without sufficiently embracing
their features. To address that, we propose GUFU, a novel effective graph-based
approach to update WiFi fingerprints using unlabelled signals with possibly new
APs. Based on the observation that similar signal vectors likely imply physical
proximity, GUFU employs a graph neural network (GNN) and a link prediction
algorithm to retrain an incremental network given the new signals and APs.
After the retraining, it then updates the signal vectors at the designated
locations. Through extensive experiments in four large representative sites,
GUFU is shown to achieve remarkably higher fingerprint adaptivity as compared
with other state-of-the-art approaches, with error reduction of 21.4% and 29.8%
in RSS values and location prediction, respectively.

### 3. [Resilient Time-Sensitive Networking for Industrial IoT: Configuration and Fault-Tolerance Evaluation](http://arxiv.org/pdf/2507.11250v1)

Authors: Mohamed Seliem, Dirk Pesch, Utz Roedig, Cormac Sreenan

Time-Sensitive Networking (TSN) is increasingly adopted in industrial systems
to meet strict latency, jitter, and reliability requirements. However,
evaluating TSN's fault tolerance under realistic failure conditions remains
challenging. This paper presents IN2C, a modular OMNeT++/INET-based simulation
framework that models two synchronized production cells connected to
centralized infrastructure. IN2C integrates core TSN features, including time
synchronization, traffic shaping, per-stream filtering, and Frame Replication
and Elimination for Redundancy (FRER), alongside XML-driven fault injection for
link and node failures. Four fault scenarios are evaluated to compare TSN
performance with and without redundancy. Results show that FRER eliminates
packet loss and achieves submillisecond recovery, though with 2-3x higher link
utilization. These findings offer practical guidance for deploying TSN in
bandwidth-constrained industrial environments.

### 4. [JamShield: A Machine Learning Detection System for Over-the-Air Jamming Attacks](http://arxiv.org/pdf/2507.11483v1)

Authors: Ioannis Panitsas, Yagmur Yigit, Leandros Tassiulas, Leandros Maglaras, Berk Canberk

Wireless networks are vulnerable to jamming attacks due to the shared
communication medium, which can severely degrade performance and disrupt
services. Despite extensive research, current jamming detection methods often
rely on simulated data or proprietary over-the-air datasets with limited
cross-layer features, failing to accurately represent the real state of a
network and thus limiting their effectiveness in real-world scenarios. To
address these challenges, we introduce JamShield, a dynamic jamming detection
system trained on our own collected over-the-air and publicly available
dataset. It utilizes hybrid feature selection to prioritize relevant features
for accurate and efficient detection. Additionally, it includes an
auto-classification module that dynamically adjusts the classification
algorithm in real-time based on current network conditions. Our experimental
results demonstrate significant improvements in detection rate, precision, and
recall, along with reduced false alarms and misdetections compared to
state-of-the-art detection algorithms, making JamShield a robust and reliable
solution for detecting jamming attacks in real-world wireless networks.

### 5. [Arcturus: A Cloud Overlay Network for Global Accelerator with Enhanced Performance and Stability](http://arxiv.org/pdf/2507.10928v1)

Authors: Matthew Yang Liu, Chuang Chen, Pengcheng Lv, Hui Guo, Yanan Zhang, Cong Wang, Yusen Li, Zhenyu Li, Yu-Chu Tian

Global Accelerator (GA) services play a vital role in ensuring low-latency,
high-reliability communication for real-time interactive applications. However,
existing GA offerings are tightly bound to specific cloud providers, resulting
in high costs, rigid deployment, and limited flexibility, especially for
large-scale or budget-sensitive deployments. Arcturus is a cloud-native GA
framework that revisits the design of GA systems by leveraging low-cost,
heterogeneous cloud resources across multiple providers. Rather than relying on
fixed, high-end infrastructure, Arcturus dynamically constructs its
acceleration network and balances performance, stability, and resource
efficiency. To achieve this, Arcturus introduces a two-plane design: a
forwarding plane that builds a proxy network with adaptive control, and a
scheduling plane that coordinates load and routing through lightweight,
quantitative optimization. Evaluations under millions of RPS show that Arcturus
outperforms commercial GA services by up to 1.7X in acceleration performance,
reduces cost by 71%, and maintains over 80% resource efficiency--demonstrating
efficient use of cloud resources at scale.

### 6. [LiLM-RDB-SFC: Lightweight Language Model with Relational Database-Guided DRL for Optimized SFC Provisioning](http://arxiv.org/pdf/2507.10903v1)

Authors: Parisa Fard Moshiri, Xinyu Zhu, Poonam Lohan, Burak Kantarci, Emil Janulewicz

Effective management of Service Function Chains (SFCs) and optimal Virtual
Network Function (VNF) placement are critical challenges in modern
Software-Defined Networking (SDN) and Network Function Virtualization (NFV)
environments. Although Deep Reinforcement Learning (DRL) is widely adopted for
dynamic network decision-making, its inherent dependency on structured data and
fixed action rules often limits adaptability and responsiveness, particularly
under unpredictable network conditions. This paper introduces LiLM-RDB-SFC, a
novel approach combining Lightweight Language Model (LiLM) with Relational
Database (RDB) to answer network state queries to guide DRL model for efficient
SFC provisioning. Our proposed approach leverages two LiLMs, Bidirectional and
Auto-Regressive Transformers (BART) and the Fine-tuned Language Net T5
(FLAN-T5), to interpret network data and support diverse query types related to
SFC demands, data center resources, and VNF availability. Results demonstrate
that FLAN-T5 outperforms BART with a lower test loss (0.00161 compared to
0.00734), higher accuracy (94.79% compared to 80.2%), and less processing time
(2h 2min compared to 2h 38min). Moreover, when compared to the large language
model SQLCoder, FLAN-T5 matches the accuracy of SQLCoder while cutting
processing time by 96% (SQLCoder: 54 h 43 min; FLAN-T5: 2 h 2 min).

### 7. [Improving Wi-Fi Network Performance Prediction with Deep Learning Models](http://arxiv.org/pdf/2507.11168v1)

Authors: Gabriele Formis, Amanda Ericson, Stefan Forsstrom, Kyi Thar, Gianluca Cena, Stefano Scanzio

The increasing need for robustness, reliability, and determinism in wireless
networks for industrial and mission-critical applications is the driver for the
growth of new innovative methods. The study presented in this work makes use of
machine learning techniques to predict channel quality in a Wi-Fi network in
terms of the frame delivery ratio. Predictions can be used proactively to
adjust communication parameters at runtime and optimize network operations for
industrial applications. Methods including convolutional neural networks and
long short-term memory were analyzed on datasets acquired from a real Wi-Fi
setup across multiple channels. The models were compared in terms of prediction
accuracy and computational complexity. Results show that the frame delivery
ratio can be reliably predicted, and convolutional neural networks, although
slightly less effective than other models, are more efficient in terms of CPU
usage and memory consumption. This enhances the model's usability on embedded
and industrial systems.

### 8. [An Agentic Flow for Finite State Machine Extraction using Prompt Chaining](http://arxiv.org/pdf/2507.11222v1)

Authors: Fares Wael, Youssef Maklad, Ali Hamdi, Wael Elsersy

Finite-State Machines (FSMs) are critical for modeling the operational logic
of network protocols, enabling verification, analysis, and vulnerability
discovery. However, existing FSM extraction techniques face limitations such as
scalability, incomplete coverage, and ambiguity in natural language
specifications. In this paper, we propose FlowFSM, a novel agentic framework
that leverages Large Language Models (LLMs) combined with prompt chaining and
chain-of-thought reasoning to extract accurate FSMs from raw RFC documents.
FlowFSM systematically processes protocol specifications, identifies state
transitions, and constructs structured rule-books by chaining agent outputs.
Experimental evaluation across FTP and RTSP protocols demonstrates that FlowFSM
achieves high extraction precision while minimizing hallucinated transitions,
showing promising results. Our findings highlight the potential of agent-based
LLM systems in the advancement of protocol analysis and FSM inference for
cybersecurity and reverse engineering applications.

### Robotics

### 1. [Mixed Discrete and Continuous Planning using Shortest Walks in Graphs of Convex Sets](http://arxiv.org/pdf/2507.10878v1)

Authors: Savva Morozov, Tobia Marcucci, Bernhard Paus Graesdal, Alexandre Amice, Pablo A. Parrilo, Russ Tedrake

We study the Shortest-Walk Problem (SWP) in a Graph of Convex Sets (GCS). A
GCS is a graph where each vertex is paired with a convex program, and each edge
couples adjacent programs via additional costs and constraints. A walk in a GCS
is a sequence of vertices connected by edges, where vertices may be repeated.
The length of a walk is given by the cumulative optimal value of the
corresponding convex programs. To solve the SWP in GCS, we first synthesize a
piecewise-quadratic lower bound on the problem's cost-to-go function using
semidefinite programming. Then we use this lower bound to guide an
incremental-search algorithm that yields an approximate shortest walk. We show
that the SWP in GCS is a natural language for many mixed discrete-continuous
planning problems in robotics, unifying problems that typically require
specialized solutions while delivering high performance and computational
efficiency. We demonstrate this through experiments in collision-free motion
planning, skill chaining, and optimal control of hybrid systems.

### 2. [Object-Centric Mobile Manipulation through SAM2-Guided Perception and Imitation Learning](http://arxiv.org/pdf/2507.10899v1)

Authors: Wang Zhicheng, Satoshi Yagi, Satoshi Yamamori, Jun Morimoto

Imitation learning for mobile manipulation is a key challenge in the field of
robotic manipulation. However, current mobile manipulation frameworks typically
decouple navigation and manipulation, executing manipulation only after
reaching a certain location. This can lead to performance degradation when
navigation is imprecise, especially due to misalignment in approach angles. To
enable a mobile manipulator to perform the same task from diverse orientations,
an essential capability for building general-purpose robotic models, we propose
an object-centric method based on SAM2, a foundation model towards solving
promptable visual segmentation in images, which incorporates manipulation
orientation information into our model. Our approach enables consistent
understanding of the same task from different orientations. We deploy the model
on a custom-built mobile manipulator and evaluate it on a pick-and-place task
under varied orientation angles. Compared to Action Chunking Transformer, our
model maintains superior generalization when trained with demonstrations from
varied approach angles. This work significantly enhances the generalization and
robustness of imitation learning-based mobile manipulation systems.

### 3. [Fast Non-Episodic Adaptive Tuning of Robot Controllers with Online Policy Optimization](http://arxiv.org/pdf/2507.10914v1)

Authors: James A. Preiss, Fengze Xie, Yiheng Lin, Adam Wierman, Yisong Yue

We study online algorithms to tune the parameters of a robot controller in a
setting where the dynamics, policy class, and optimality objective are all
time-varying. The system follows a single trajectory without episodes or state
resets, and the time-varying information is not known in advance. Focusing on
nonlinear geometric quadrotor controllers as a test case, we propose a
practical implementation of a single-trajectory model-based online policy
optimization algorithm, M-GAPS,along with reparameterizations of the quadrotor
state space and policy class to improve the optimization landscape. In hardware
experiments,we compare to model-based and model-free baselines that impose
artificial episodes. We show that M-GAPS finds near-optimal parameters more
quickly, especially when the episode length is not favorable. We also show that
M-GAPS rapidly adapts to heavy unmodeled wind and payload disturbances, and
achieves similar strong improvement on a 1:6-scale Ackermann-steered car. Our
results demonstrate the hardware practicality of this emerging class of online
policy optimization that offers significantly more flexibility than classic
adaptive control, while being more stable and data-efficient than model-free
reinforcement learning.

### 4. [Unified Modeling and Structural Optimization of Multi-magnet Embedded Soft Continuum Robots for Enhanced Kinematic Performances](http://arxiv.org/pdf/2507.10950v1)

Authors: Zhiwei Wu, Jiahao Luo, Siyi Wei, Jinhui Zhang

This paper presents a unified modeling and optimization framework to enhance
the kinematic performance of multi-magnet embedded soft continuum robots
(MeSCRs). To this end, we establish a differentiable system formulation based
on an extended pseudo-rigid-body model. This formulation enables analysis of
the equilibrium well-posedness and the geometry of the induced configuration
under magnetic actuation. In particular, we show that the maximum controllable
degrees of freedom of a MeSCR equal twice the number of embedded magnets. We
subsequently develop a structural optimization framework based on differential
geometry that links classical kinematic measures (e.g., manipulability and
dexterity) to the configuration of embedded magnets. The resulting optimization
condition reveals that improving local performance requires structurally
modulating the spectrum of the configuration space metric to counteract its
distortion. Closed-form solutions for optimal magnet configurations are derived
under representative conditions, and a gradient-based numerical method is
proposed for general design scenarios. Simulation studies validate the
effectiveness of the proposed framework.

### 5. [EquiContact: A Hierarchical SE(3) Vision-to-Force Equivariant Policy for Spatially Generalizable Contact-rich Tasks](http://arxiv.org/pdf/2507.10961v1)

Authors: Joohwan Seo, Arvind Kruthiventy, Soomi Lee, Megan Teng, Xiang Zhang, Seoyeon Choi, Jongeun Choi, Roberto Horowitz

This paper presents a framework for learning vision-based robotic policies
for contact-rich manipulation tasks that generalize spatially across task
configurations. We focus on achieving robust spatial generalization of the
policy for the peg-in-hole (PiH) task trained from a small number of
demonstrations. We propose EquiContact, a hierarchical policy composed of a
high-level vision planner (Diffusion Equivariant Descriptor Field, Diff-EDF)
and a novel low-level compliant visuomotor policy (Geometric Compliant ACT,
G-CompACT). G-CompACT operates using only localized observations (geometrically
consistent error vectors (GCEV), force-torque readings, and wrist-mounted RGB
images) and produces actions defined in the end-effector frame. Through these
design choices, we show that the entire EquiContact pipeline is
SE(3)-equivariant, from perception to force control. We also outline three key
components for spatially generalizable contact-rich policies: compliance,
localized policies, and induced equivariance. Real-world experiments on PiH
tasks demonstrate a near-perfect success rate and robust generalization to
unseen spatial configurations, validating the proposed framework and
principles. The experimental videos can be found on the project website:
https://sites.google.com/berkeley.edu/equicontact

### 6. [Uncertainty Aware Mapping for Vision-Based Underwater Robots](http://arxiv.org/pdf/2507.10991v1)

Authors: Abhimanyu Bhowmik, Mohit Singh, Madhushree Sannigrahi, Martin Ludvigsen, Kostas Alexis

Vision-based underwater robots can be useful in inspecting and exploring
confined spaces where traditional sensors and preplanned paths cannot be
followed. Sensor noise and situational change can cause significant uncertainty
in environmental representation. Thus, this paper explores how to represent
mapping inconsistency in vision-based sensing and incorporate depth estimation
confidence into the mapping framework. The scene depth and the confidence are
estimated using the RAFT-Stereo model and are integrated into a voxel-based
mapping framework, Voxblox. Improvements in the existing Voxblox weight
calculation and update mechanism are also proposed. Finally, a qualitative
analysis of the proposed method is performed in a confined pool and in a pier
in the Trondheim fjord. Experiments using an underwater robot demonstrated the
change in uncertainty in the visualization.

### 7. [ILCL: Inverse Logic-Constraint Learning from Temporally Constrained Demonstrations](http://arxiv.org/pdf/2507.11000v1)

Authors: Minwoo Cho, Jaehwi Jang, Daehyung Park

We aim to solve the problem of temporal-constraint learning from
demonstrations to reproduce demonstration-like logic-constrained behaviors.
Learning logic constraints is challenging due to the combinatorially large
space of possible specifications and the ill-posed nature of non-Markovian
constraints. To figure it out, we introduce a novel temporal-constraint
learning method, which we call inverse logic-constraint learning (ILCL). Our
method frames ICL as a two-player zero-sum game between 1) a genetic
algorithm-based temporal-logic mining (GA-TL-Mining) and 2) logic-constrained
reinforcement learning (Logic-CRL). GA-TL-Mining efficiently constructs syntax
trees for parameterized truncated linear temporal logic (TLTL) without
predefined templates. Subsequently, Logic-CRL finds a policy that maximizes
task rewards under the constructed TLTL constraints via a novel constraint
redistribution scheme. Our evaluations show ILCL outperforms state-of-the-art
baselines in learning and transferring TL constraints on four temporally
constrained tasks. We also demonstrate successful transfer to real-world
peg-in-shallow-hole tasks.

### 8. [Enhancing Autonomous Manipulator Control with Human-in-loop for Uncertain Assembly Environments](http://arxiv.org/pdf/2507.11006v1)

Authors: Ashutosh Mishra, Shreya Santra, Hazal Gozbasi, Kentaro Uno, Kazuya Yoshida

This study presents an advanced approach to enhance robotic manipulation in
uncertain and challenging environments, with a focus on autonomous operations
augmented by human-in-the-loop (HITL) control for lunar missions. By
integrating human decision-making with autonomous robotic functions, the
research improves task reliability and efficiency for space applications. The
key task addressed is the autonomous deployment of flexible solar panels using
an extendable ladder-like structure and a robotic manipulator with real-time
feedback for precision. The manipulator relays position and force-torque data,
enabling dynamic error detection and adaptive control during deployment. To
mitigate the effects of sinkage, variable payload, and low-lighting conditions,
efficient motion planning strategies are employed, supplemented by human
control that allows operators to intervene in ambiguous scenarios. Digital twin
simulation enhances system robustness by enabling continuous feedback,
iterative task refinement, and seamless integration with the deployment
pipeline. The system has been tested to validate its performance in simulated
lunar conditions and ensure reliability in extreme lighting, variable terrain,
changing payloads, and sensor limitations.

### 9. [Force-Based Viscosity and Elasticity Measurements for Material Biomechanical Characterisation with a Collaborative Robotic Arm](http://arxiv.org/pdf/2507.11133v1)

Authors: Luca Beber, Edoardo Lamon, Giacomo Moretti, Matteo Saveriano, Luca Fambri, Luigi Palopoli, Daniele Fontanelli

Diagnostic activities, such as ultrasound scans and palpation, are relatively
low-cost. They play a crucial role in the early detection of health problems
and in assessing their progression. However, they are also error-prone
activities, which require highly skilled medical staff. The use of robotic
solutions can be key to decreasing the inherent subjectivity of the results and
reducing the waiting list. For a robot to perform palpation or ultrasound
scans, it must effectively manage physical interactions with the human body,
which greatly benefits from precise estimation of the patient's tissue
biomechanical properties. This paper assesses the accuracy and precision of a
robotic system in estimating the viscoelastic parameters of various materials,
including some tests on ex vivo tissues as a preliminary proof-of-concept
demonstration of the method's applicability to biological samples. The
measurements are compared against a ground truth derived from silicone
specimens with different viscoelastic properties, characterised using a
high-precision instrument. Experimental results show that the robotic system's
accuracy closely matches the ground truth, increasing confidence in the
potential use of robots for such clinical applications.

### 10. [A Robust Controller based on Gaussian Processes for Robotic Manipulators with Unknown Uncertainty](http://arxiv.org/pdf/2507.11170v1)

Authors: Giulio Giacomuzzo, Mohamed Abdelwahab, Marco Calì, Alberto Dalla Libera, Ruggero Carli

In this paper, we propose a novel learning-based robust feedback
linearization strategy to ensure precise trajectory tracking for an important
family of Lagrangian systems. We assume a nominal knowledge of the dynamics is
given but no a-priori bounds on the model mismatch are available. In our
approach, the key ingredient is the adoption of a regression framework based on
Gaussian Processes (GPR) to estimate the model mismatch. This estimate is added
to the outer loop of a classical feedback linearization scheme based on the
nominal knowledge available. Then, to compensate for the residual uncertainty,
we robustify the controller including an additional term whose size is designed
based on the variance provided by the GPR framework. We proved that, with high
probability, the proposed scheme is able to guarantee asymptotic tracking of a
desired trajectory. We tested numerically our strategy on a 2 degrees of
freedom planar robot.

### Software Engineering

### 1. [Evaluating Generated Commit Messages with Large Language Models](http://arxiv.org/pdf/2507.10906v1)

Authors: Qunhong Zeng, Yuxia Zhang, Zexiong Ma, Bo Jiang, Ningyuan Sun, Klaas-Jan Stol, Xingyu Mou, Hui Liu

Commit messages are essential in software development as they serve to
document and explain code changes. Yet, their quality often falls short in
practice, with studies showing significant proportions of empty or inadequate
messages. While automated commit message generation has advanced significantly,
particularly with Large Language Models (LLMs), the evaluation of generated
messages remains challenging. Traditional reference-based automatic metrics
like BLEU, ROUGE-L, and METEOR have notable limitations in assessing commit
message quality, as they assume a one-to-one mapping between code changes and
commit messages, leading researchers to rely on resource-intensive human
evaluation. This study investigates the potential of LLMs as automated
evaluators for commit message quality. Through systematic experimentation with
various prompt strategies and state-of-the-art LLMs, we demonstrate that LLMs
combining Chain-of-Thought reasoning with few-shot demonstrations achieve near
human-level evaluation proficiency. Our LLM-based evaluator significantly
outperforms traditional metrics while maintaining acceptable reproducibility,
robustness, and fairness levels despite some inherent variability. This work
conducts a comprehensive preliminary study on using LLMs for commit message
evaluation, offering a scalable alternative to human assessment while
maintaining high-quality evaluation.

### 2. [MT4DP: Data Poisoning Attack Detection for DL-based Code Search Models via Metamorphic Testing](http://arxiv.org/pdf/2507.11092v1)

Authors: Gong Chen, Wenjie Liu, Xiaoyuan Xie, Xunzhu Tang, Tegawendé F. Bissyandé, Songqiang Chen

Recently, several studies have indicated that data poisoning attacks pose a
severe security threat to deep learning-based (DL-based) code search models.
Attackers inject carefully crafted malicious patterns into the training data,
misleading the code search model to learn these patterns during training.
During the usage of the poisoned code search model for inference, once the
malicious pattern is triggered, the model tends to rank the vulnerability code
higher. However, existing detection methods for data poisoning attacks on
DL-based code search models remain insufficiently effective. To address this
critical security issue, we propose MT4DP, a Data Poisoning Attack Detection
Framework for DL-based Code Search Models via Metamorphic Testing. MT4DP
introduces a novel Semantically Equivalent Metamorphic Relation (SE-MR)
designed to detect data poisoning attacks on DL-based code search models.
Specifically, MT4DP first identifies the high-frequency words from search
queries as potential poisoning targets and takes their corresponding queries as
the source queries. For each source query, MT4DP generates two semantically
equivalent follow-up queries and retrieves its source ranking list. Then, each
source ranking list is re-ranked based on the semantic similarities between its
code snippets and the follow-up queries. Finally, variances between the source
and re-ranked lists are calculated to reveal violations of the SE-MR and warn
the data poisoning attack. Experimental results demonstrate that MT4DP
significantly enhances the detection of data poisoning attacks on DL-based code
search models, outperforming the best baseline by 191% on average F1 score and
265% on average precision. Our work aims to promote further research into
effective techniques for mitigating data poisoning threats on DL-based code
search models.

### 3. [Automata Models for Effective Bug Description](http://arxiv.org/pdf/2507.11146v1)

Authors: Tom Yaacov, Gera Weiss, Gal Amram, Avi Hayoun

Debugging complex systems is a crucial yet time-consuming task. This paper
presents the use of automata learning and testing techniques to obtain concise
and informative bug descriptions. We introduce the concepts of Failure
Explanations (FE), Eventual Failure Explanations (EFE), and Early Detection
(ED) to provide meaningful summaries of failing behavior patterns. By factoring
out irrelevant information and focusing on essential test patterns, our
approach aims to enhance bug detection and understanding. We evaluate our
methods using various test patterns and real-world benchmarks, demonstrating
their effectiveness in producing compact and informative bug descriptions.

### 4. [New Formulation of DNN Statistical Mutation Killing for Ensuring Monotonicity: A Technical Report](http://arxiv.org/pdf/2507.11199v1)

Authors: Jinhan Kim, Nargiz Humbatova, Gunel Jahangirova, Shin Yoo, Paolo Tonella

Mutation testing has emerged as a powerful technique for evaluating the
effectiveness of test suites for Deep Neural Networks. Among existing
approaches, the statistical mutant killing criterion of DeepCrime has leveraged
statistical testing to determine whether a mutant significantly differs from
the original model. However, it suffers from a critical limitation: it violates
the monotonicity property, meaning that expanding a test set may result in
previously killed mutants no longer being classified as killed. In this
technical report, we propose a new formulation of statistical mutant killing
based on Fisher exact test that preserves the statistical rigour of it while
ensuring monotonicity.

### 5. [An Empirical Study of Multi-Agent RAG for Real-World University Admissions Counseling](http://arxiv.org/pdf/2507.11272v1)

Authors: Anh Nguyen-Duc, Chien Vu Manh, Bao Anh Tran, Viet Phuong Ngo, Luan Le Chi, Anh Quang Nguyen

This paper presents MARAUS (Multi-Agent and Retrieval-Augmented University
Admission System), a real-world deployment of a conversational AI platform for
higher education admissions counseling in Vietnam. While large language models
(LLMs) offer potential for automating advisory tasks, most existing solutions
remain limited to prototypes or synthetic benchmarks. MARAUS addresses this gap
by combining hybrid retrieval, multi-agent orchestration, and LLM-based
generation into a system tailored for real-world university admissions. In
collaboration with the University of Transport Technology (UTT) in Hanoi, we
conducted a two-phase study involving technical development and real-world
evaluation. MARAUS processed over 6,000 actual user interactions, spanning six
categories of queries. Results show substantial improvements over LLM-only
baselines: on average 92 percent accuracy, hallucination rates reduced from 15
precent to 1.45 percent, and average response times below 4 seconds. The system
operated cost-effectively, with a two-week deployment cost of 11.58 USD using
GPT-4o mini. This work provides actionable insights for the deployment of
agentic RAG systems in low-resource educational settings.

### 6. [RefModel: Detecting Refactorings using Foundation Models](http://arxiv.org/pdf/2507.11346v1)

Authors: Pedro Simões, Rohit Gheyi, Rian Melo, Jonhnanthan Oliveira, Márcio Ribeiro, Wesley K. G. Assunção

Refactoring is a common software engineering practice that improves code
quality without altering program behavior. Although tools like ReExtractor+,
RefactoringMiner, and RefDiff have been developed to detect refactorings
automatically, they rely on complex rule definitions and static analysis,
making them difficult to extend and generalize to other programming languages.
In this paper, we investigate the viability of using foundation models for
refactoring detection, implemented in a tool named RefModel. We evaluate
Phi4-14B, and Claude 3.5 Sonnet on a dataset of 858 single-operation
transformations applied to artificially generated Java programs, covering
widely-used refactoring types. We also extend our evaluation by including
Gemini 2.5 Pro and o4-mini-high, assessing their performance on 44 real-world
refactorings extracted from four open-source projects. These models are
compared against RefactoringMiner, RefDiff, and ReExtractor+. RefModel is
competitive with, and in some cases outperform, traditional tools. In
real-world settings, Claude 3.5 Sonnet and Gemini 2.5 Pro jointly identified
97% of all refactorings, surpassing the best-performing static-analysis-based
tools. The models showed encouraging generalization to Python and Golang. They
provide natural language explanations and require only a single sentence to
define each refactoring type.

### 7. [Security Debt in Practice: Nuanced Insights from Practitioners](http://arxiv.org/pdf/2507.11362v1)

Authors: Chaima Boufaied, Taher Ghaleb, Zainab Masood

With the increasing reliance on software and automation nowadays, tight
deadlines, limited resources, and prioritization of functionality over security
can lead to insecure coding practices. When not handled properly, these
constraints cause unaddressed security vulnerabilities to accumulate over time,
forming Security Debts (SDs). Despite their critical importance, there is
limited empirical evidence on how software practitioners perceive, manage, and
communicate SDs in real-world settings. In this paper, we present a qualitative
empirical study based on semi-structured interviews with 22 software
practitioners across various roles, organizations, and countries. We address
four research questions: i) we assess software practitioners' knowledge of SDs
and awareness of associated security risks, ii) we investigate their behavior
towards SDs, iii) we explore common tools and strategies used to mitigate SDs,
and iv) we analyze how security risks are communicated within teams and to
decision makers. We observe variations in how practitioners perceive and manage
SDs, with some prioritizing delivery speed over security, while others
consistently maintain security as a priority. Our findings emphasize the need
for stronger integration of security practices across the Software Development
Life Cycle (SDLC), more consistent use of mitigation strategies, better
balancing of deadlines, resources, and security-related tasks, with attention
to the Confidentiality, Integrity, and Availability (CIA) triad.

### 8. [Function-to-Style Guidance of LLMs for Code Translation](http://arxiv.org/pdf/2507.11083v1)

Authors: Longhui Zhang, Bin Wang, Jiahao Wang, Xiaofeng Zhao, Min Zhang, Hao Yang, Meishan Zhang, Yu Li, Jing Li, Jun Yu, Min Zhang

Large language models (LLMs) have made significant strides in code
translation tasks. However, ensuring both the correctness and readability of
translated code remains a challenge, limiting their effective adoption in
real-world software development. In this work, we propose F2STrans, a
function-to-style guiding paradigm designed to progressively improve the
performance of LLMs in code translation. Our approach comprises two key stages:
(1) Functional learning, which optimizes translation correctness using
high-quality source-target code pairs mined from online programming platforms,
and (2) Style learning, which improves translation readability by incorporating
both positive and negative style examples. Additionally, we introduce a novel
code translation benchmark that includes up-to-date source code, extensive test
cases, and manually annotated ground-truth translations, enabling comprehensive
functional and stylistic evaluations. Experiments on both our new benchmark and
existing datasets demonstrate that our approach significantly improves code
translation performance. Notably, our approach enables Qwen-1.5B to outperform
prompt-enhanced Qwen-32B and GPT-4 on average across 20 diverse code
translation scenarios.

### 9. [From Chaos to Automation: Enabling the Use of Unstructured Data for Robotic Process Automation](http://arxiv.org/pdf/2507.11364v1)

Authors: Kelly Kurowski, Xixi Lu, Hajo A. Reijers

The growing volume of unstructured data within organizations poses
significant challenges for data analysis and process automation. Unstructured
data, which lacks a predefined format, encompasses various forms such as
emails, reports, and scans. It is estimated to constitute approximately 80% of
enterprise data. Despite the valuable insights it can offer, extracting
meaningful information from unstructured data is more complex compared to
structured data. Robotic Process Automation (RPA) has gained popularity for
automating repetitive tasks, improving efficiency, and reducing errors.
However, RPA is traditionally reliant on structured data, limiting its
application to processes involving unstructured documents. This study addresses
this limitation by developing the UNstructured Document REtrieval SyStem
(UNDRESS), a system that uses fuzzy regular expressions, techniques for natural
language processing, and large language models to enable RPA platforms to
effectively retrieve information from unstructured documents. The research
involved the design and development of a prototype system, and its subsequent
evaluation based on text extraction and information retrieval performance. The
results demonstrate the effectiveness of UNDRESS in enhancing RPA capabilities
for unstructured data, providing a significant advancement in the field. The
findings suggest that this system could facilitate broader RPA adoption across
processes traditionally hindered by unstructured data, thereby improving
overall business process efficiency.

### 10. [Modeling Code: Is Text All You Need?](http://arxiv.org/pdf/2507.11467v1)

Authors: Daniel Nichols, Konstantinos Parasyris, Harshitha Menon, Brian R. Bartoldson, Giorgis Georgakoudis, Tal Ben-Nun, Abhinav Bhatele

Code LLMs have become extremely popular recently for modeling source code
across a variety of tasks, such as generation, translation, and summarization.
However, transformer-based models are limited in their capabilities to reason
through structured, analytical properties of code, such as control and data
flow. Previous work has explored the modeling of these properties with
structured data and graph neural networks. However, these approaches lack the
generative capabilities and scale of modern LLMs. In this work, we introduce a
novel approach to combine the strengths of modeling both code as text and more
structured forms.

### Social and Information Networks

### 1. [Toxicity in State Sponsored Information Operations](http://arxiv.org/pdf/2507.10936v1)

Authors: Ashfaq Ali Shafin, Khandaker Mamun Ahmed

State-sponsored information operations (IOs) increasingly influence global
discourse on social media platforms, yet their emotional and rhetorical
strategies remain inadequately characterized in scientific literature. This
study presents the first comprehensive analysis of toxic language deployment
within such campaigns, examining 56 million posts from over 42 thousand
accounts linked to 18 distinct geopolitical entities on X/Twitter. Using
Google's Perspective API, we systematically detect and quantify six categories
of toxic content and analyze their distribution across national origins,
linguistic structures, and engagement metrics, providing essential information
regarding the underlying patterns of such operations. Our findings reveal that
while toxic content constitutes only 1.53% of all posts, they are associated
with disproportionately high engagement and appear to be strategically deployed
in specific geopolitical contexts. Notably, toxic content originating from
Russian influence operations receives significantly higher user engagement
compared to influence operations from any other country in our dataset. Our
code is available at https://github.com/shafin191/Toxic_IO.

### 2. [Enhance Stability of Network by Edge Anchor](http://arxiv.org/pdf/2507.11090v1)

Authors: Hongbo Qiu, Renjie Sun, Chen chen, Xiaoyang Wang

With the rapid growth of online social networks, strengthening their
stability has emerged as a key research focus. This study aims to identify
influential relationships that significantly impact community stability. In
this paper, we introduce and explore the anchor trussness reinforcement problem
to reinforce the overall user engagement of networks by anchoring some edges.
Specifically, for a given graph $G$ and a budget $b$, we aim to identify $b$
edges whose anchoring maximizes the trussness gain, which is the cumulative
increment of trussness across all edges in $G$. We establish the NP-hardness of
the problem. To address this problem, we introduce a greedy framework that
iteratively selects the current best edge. To scale for larger networks, we
first propose an upward-route method to constrain potential trussness increment
edges. Augmented with a support check strategy, this approach enables the
efficient computation of the trussness gain for anchoring one edge. Then, we
design a classification tree structure to minimize redundant computations in
each iteration by organizing edges based on their trussness. We conduct
extensive experiments on 8 real-world networks to validate the efficiency and
effectiveness of the proposed model and methods.

### 3. [Urban delineation through the lens of commute networks: Leveraging graph embeddings to distinguish socioeconomic groups in cities](http://arxiv.org/pdf/2507.11057v1)

Authors: Devashish Khulbe, Stanislav Sobolevsky

Delineating areas within metropolitan regions stands as an important focus
among urban researchers, shedding light on the urban perimeters shaped by
evolving population dynamics. Applications to urban science are numerous, from
facilitating comparisons between delineated districts and administrative
divisions to informing policymakers of the shifting economic and labor
landscapes. In this study, we propose using commute networks sourced from the
census for the purpose of urban delineation, by modeling them with a Graph
Neural Network (GNN) architecture. We derive low-dimensional representations of
granular urban areas (nodes) using GNNs. Subsequently, nodes' embeddings are
clustered to identify spatially cohesive communities in urban areas. Our
experiments across the U.S. demonstrate the effectiveness of network embeddings
in capturing significant socioeconomic disparities between communities in
various cities, particularly in factors such as median household income. The
role of census mobility data in regional delineation is also noted, and we
establish the utility of GNNs in urban community detection, as a powerful
alternative to existing methods in this domain. The results offer insights into
the wider effects of commute networks and their use in building meaningful
representations of urban regions.

### 4. [The Potential Impact of Disruptive AI Innovations on U.S. Occupations](http://arxiv.org/pdf/2507.11403v1)

Authors: Munjung Kim, Marios Constantinides, Sanja Šćepanović, Yong-Yeol Ahn, Daniele Quercia

The rapid rise of AI is poised to disrupt the labor market. However, AI is
not a monolith; its impact depends on both the nature of the innovation and the
jobs it affects. While computational approaches are emerging, there is no
consensus on how to systematically measure an innovation's disruptive
potential. Here, we calculate the disruption index of 3,237 U.S. AI patents
(2015-2022) and link them to job tasks to distinguish between "consolidating"
AI innovations that reinforce existing structures and "disruptive" AI
innovations that alter them. Our analysis reveals that consolidating AI
primarily targets physical, routine, and solo tasks, common in manufacturing
and construction in the Midwest and central states. By contrast, disruptive AI
affects unpredictable and mental tasks, particularly in coastal science and
technology sectors. Surprisingly, we also find that disruptive AI
disproportionately affects areas already facing skilled labor shortages,
suggesting disruptive AI technologies may accelerate change where workers are
scarce rather than replacing a surplus. Ultimately, consolidating AI appears to
extend current automation trends, while disruptive AI is set to transform
complex mental work, with a notable exception for collaborative tasks.

### 5. [HIF: The hypergraph interchange format for higher-order networks](http://arxiv.org/pdf/2507.11520v1)

Authors: Martín Coll, Cliff A. Joslyn, Nicholas W. Landry, Quintino Francesco Lotito, Audun Myers, Joshua Pickard, Brenda Praggastis, Przemysław Szufel

Many empirical systems contain complex interactions of arbitrary size,
representing, for example, chemical reactions, social groups, co-authorship
relationships, and ecological dependencies. These interactions are known as
higher-order interactions and the collection of these interactions comprise a
higher-order network, or hypergraph. Hypergraphs have established themselves as
a popular and versatile mathematical representation of such systems and a
number of software packages written in various programming languages have been
designed to analyze these networks. However, the ecosystem of higher-order
network analysis software is fragmented due to specialization of each
software's programming interface and compatible data representations. To enable
seamless data exchange between higher-order network analysis software packages,
we introduce the Hypergraph Interchange Format (HIF), a standardized format for
storing higher-order network data. HIF supports multiple types of higher-order
networks, including undirected hypergraphs, directed hypergraphs, and
simplicial complexes, while actively exploring extensions to represent
multiplex hypergraphs, temporal hypergraphs, and ordered hypergraphs. To
accommodate the wide variety of metadata used in different contexts, HIF also
includes support for attributes associated with nodes, edges, and incidences.
This initiative is a collaborative effort involving authors, maintainers, and
contributors from prominent hypergraph software packages. This project
introduces a JSON schema with corresponding documentation and unit tests,
example HIF-compliant datasets, and tutorials demonstrating the use of HIF with
several popular higher-order network analysis software packages.

### Systems and Control

### 1. [Data-Driven Safety Certificates of Infinite Networks with Unknown Models and Interconnection Topologies](http://arxiv.org/pdf/2507.10979v1)

Authors: Mahdieh Zaker, Amy Nejati, Abolfazl Lavaei

Infinite networks are complex interconnected systems comprising a countably
infinite number of subsystems, where counting them precisely poses a
significant challenge due to the seemingly endless interconnected nature of the
network (e.g., counting vehicles on the road). In such scenarios, the presence
of infinitely many subsystems within the network renders the existing analysis
frameworks tailored for finite networks inapplicable to infinite ones. This
paper is concerned with offering a data-driven approach, within a compositional
framework, for the safety certification of infinite networks with both unknown
mathematical models and interconnection topologies. Given the immense
computational complexity stemming from the extensive dimension of infinite
networks, our approach capitalizes on the joint dissipativity-type properties
of subsystems, characterized by storage certificates. We introduce innovative
compositional data-driven conditions to construct a barrier certificate for the
infinite network leveraging storage certificates of its unknown subsystems
derived from data, while offering correctness guarantees across the network
safety. We demonstrate that our compositional data-driven reasoning eliminates
the requirement for checking the traditional dissipativity condition, which
typically mandates precise knowledge of the interconnection topology. In
addition, while existing data-driven literature demonstrates an exponential
trend in sample complexity with respect to network size, we showcase that our
compositional strategy notably reduces it to a linear scale in terms of the
number of subsystems. We illustrate our data-driven results on two physical
infinite networks with unknown models and interconnection topologies.

### 2. [Optimal Honeypot Ratio and Convergent Fictitious-Play Learning in Signaling Games for CPS Defense](http://arxiv.org/pdf/2507.11113v1)

Authors: Yueyue Xu, Yuewei Chen, Lin Wang, Zhaoyang Cheng, Xiaoming Hu

Cyber-Physical Systems (CPSs) are facing a fast-growing wave of attacks. To
achieve effective proactive defense, this paper models honeypot deployment as a
gamma-fixed signaling game in which node liveness serves as the only signal and
normal-node signal gamma is exogenously fixed. We define the gamma-perfect
Bayesian-Nash equilibrium (gamma-PBNE). Analytical expressions are obtained for
all gamma-PBNEs, revealing three distinct equilibrium regimes that depend on
the priori honeypot ratio. Furthermore, the optimal honeypot ratio and
signaling strategy that jointly maximize the network average utility are
obtained. To capture strategic interaction over time, we develop a
discrete-time fictitious-play algorithm that couples Bayesian belief updates
with empirical best responses. We prove that, as long as the honeypot ratio is
perturbed within a non-degenerate neighbourhood of the optimum, every
fictitious-play path converges to the defender-optimal gamma-PBNE. Numerical
results confirm the effectiveness of the proposed method and demonstrate its
applicability to CPS defense.

### 3. [Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](http://arxiv.org/pdf/2507.11240v1)

Authors: Mohamad Al Ahdab, John Leth, Zheng-Hua Tan

We study the Continuous-Discrete Kalman Filter (CD-KF) for State-Space Models
(SSMs) where continuous-time dynamics are observed via multiple sensors with
discrete, irregularly timed measurements. Our focus extends to scenarios in
which the measurement process is coupled with the states of an auxiliary SSM.
For instance, higher measurement rates may increase energy consumption or heat
generation, while a sensor's accuracy can depend on its own spatial trajectory
or that of the measured target. Each sensor thus carries distinct costs and
constraints associated with its measurement rate and additional constraints and
costs on the auxiliary state. We model measurement occurrences as independent
Poisson processes with sensor-specific rates and derive an upper bound on the
mean posterior covariance matrix of the CD-KF along the mean auxiliary state.
The bound is continuously differentiable with respect to the measurement rates,
which enables efficient gradient-based optimization. Exploiting this bound, we
propose a finite-horizon optimal control framework to optimize measurement
rates and auxiliary-state dynamics jointly. We further introduce a
deterministic method for scheduling measurement times from the optimized rates.
Empirical results in state-space filtering and dynamic temporal Gaussian
process regression demonstrate that our approach achieves improved trade-offs
between resource usage and estimation accuracy.

### 4. [Moving Beyond Marginal Carbon Intensity: A Poor Metric for Both Carbon Accounting and Grid Flexibility](http://arxiv.org/pdf/2507.11377v1)

Authors: Philipp Wiesner, Odej Kao

Marginal Carbon Intensity (MCI) has been promoted as an effective metric for
carbon-aware computing. Although it is already considered as impractical for
carbon accounting purposes, many still view it as valuable when optimizing for
grid flexibility by incentivizing electricity usage during curtailment periods.
In this statement paper, we argue that MCI is neither reliable nor actionable
for either purpose. We outline its fundamental limitations, including
non-observability, reliance on opaque predictive models, and the lack of
verifiability. Moreover, MCI fails to reflect curtailment caused by high-carbon
sources and offers no insight into the quantity of available excess power. We
advocate moving beyond MCI and instead call for research on more actionable
metrics, such as direct reporting of excess power, explicit modeling of energy
storage and grid stability, and integration with emerging granular renewable
energy certificate markets.

### 5. [Inverse Optimal Control with Constraint Relaxation](http://arxiv.org/pdf/2507.11392v1)

Authors: Rahel Rickenbach, Amon Lahr, Melanie N. Zeilinger

Inverse optimal control (IOC) is a promising paradigm for learning and
mimicking optimal control strategies from capable demonstrators, or gaining a
deeper understanding of their intentions, by estimating an unknown objective
function from one or more corresponding optimal control sequences. When
computing estimates from demonstrations in environments with safety-preserving
inequality constraints, acknowledging their presence in the chosen IOC method
is crucial given their strong influence on the final control strategy. However,
solution strategies capable of considering inequality constraints, such as the
inverse Karush-Kuhn-Tucker approach, rely on their correct activation and
fulfillment; a restrictive assumption when dealing with noisy demonstrations.
To overcome this problem, we leverage the concept of exact penalty functions
for IOC and show preservation of estimation accuracy. Considering noisy
demonstrations, we then illustrate how the usage of penalty functions reduces
the number of unknown variables and how their approximations enhance the
estimation method's capacity to account for wrong constraint activations within
a polytopic-constrained environment. The proposed method is evaluated for three
systems in simulation, outperforming traditional relaxation approaches for
noisy demonstrations.

### 6. [A Risk-Aware Adaptive Robust MPC with Learned Uncertainty Quantification](http://arxiv.org/pdf/2507.11420v1)

Authors: Mingcong Li

Solving chance-constrained optimal control problems for systems subject to
non-stationary uncertainties is a significant challenge.Conventional robust
model predictive control (MPC) often yields excessive conservatism by relying
on static worst-case assumptions, while standard stochastic MPC methods
struggle when underlying uncertainty distributions are unknown a priori.This
article presents a Risk-Aware Adaptive Robust MPC (RAAR-MPC) framework,a
hierarchical architecture that systematically orchestrates a novel synthesis of
proactive, learning-based risk assessment and reactive risk regulation. The
framework employs a medium-frequency risk assessment engine, which leverages
Gaussian process regression and active learning, to construct a tight,
data-driven characterization of the prediction error set from operational
data.Concurrently, a low-timescale outer loop implements a self-correcting
update law for an adaptive safety margin to precisely regulate the empirical
risk and compensate for unmodeled dynamics.This dual-timescale adaptation
enables the system to rigorously satisfy chance constraints with a user-defined
probability, while minimizing the conservatism inherent in traditional
approaches.We formally establish that the interplay between these adaptive
components guarantees recursive feasibility and ensures the closed-loop system
satisfies the chance constraints up to a user-defined risk level with high
probability.Numerical experiments on a benchmark DC-DC converter under
non-stationary parametric uncertainties demonstrate that our framework
precisely achieves the target risk level, resulting in a significantly lower
average cost compared to state-of-the-art robust and stochastic MPC strategies.

### 7. [SMART-Merge Planner: A Safe Merging and Real-Time Motion Planner for Autonomous Highway On-Ramp Merging](http://arxiv.org/pdf/2507.10968v1)

Authors: Toktam Mohammadnejad, Jovin D'sa, Behdad Chalaki, Hossein Nourkhiz Mahjoub, Ehsan Moradi-Pari

Merging onto a highway is a complex driving task that requires identifying a
safe gap, adjusting speed, often interactions to create a merging gap, and
completing the merge maneuver within a limited time window while maintaining
safety and driving comfort. In this paper, we introduce a Safe Merging and
Real-Time Merge (SMART-Merge) planner, a lattice-based motion planner designed
to facilitate safe and comfortable forced merging. By deliberately adapting
cost terms to the unique challenges of forced merging and introducing a desired
speed heuristic, SMART-Merge planner enables the ego vehicle to merge
successfully while minimizing the merge time. We verify the efficiency and
effectiveness of the proposed merge planner through high-fidelity CarMaker
simulations on hundreds of highway merge scenarios. Our proposed planner
achieves the success rate of 100% as well as completes the merge maneuver in
the shortest amount of time compared with the baselines, demonstrating our
planner's capability to handle complex forced merge tasks and provide a
reliable and robust solution for autonomous highway merge. The simulation
result videos are available at
https://sites.google.com/view/smart-merge-planner/home.

### 8. [Approximate solutions to games of ordered preference](http://arxiv.org/pdf/2507.11021v1)

Authors: Pau de las Heras Molins, Eric Roy-Almonacid, Dong Ho Lee, Lasse Peters, David Fridovich-Keil, Georgios Bakirtzis

Autonomous vehicles must balance ranked objectives, such as minimizing travel
time, ensuring safety, and coordinating with traffic. Games of ordered
preference effectively model these interactions but become computationally
intractable as the time horizon, number of players, or number of preference
levels increase. While receding horizon frameworks mitigate long-horizon
intractability by solving sequential shorter games, often warm-started, they do
not resolve the complexity growth inherent in existing methods for solving
games of ordered preference. This paper introduces a solution strategy that
avoids excessive complexity growth by approximating solutions using
lexicographic iterated best response (IBR) in receding horizon, termed
"lexicographic IBR over time." Lexicographic IBR over time uses past
information to accelerate convergence. We demonstrate through simulated traffic
scenarios that lexicographic IBR over time efficiently computes
approximate-optimal solutions for receding horizon games of ordered preference,
converging towards generalized Nash equilibria.

### 9. [Standards-Compliant DM-RS Allocation via Temporal Channel Prediction for Massive MIMO Systems](http://arxiv.org/pdf/2507.11064v1)

Authors: Sehyun Ryu, Hyun Jong Yang

Reducing feedback overhead in beyond 5G networks is a critical challenge, as
the growing number of antennas in modern massive MIMO systems substantially
increases the channel state information (CSI) feedback demand in frequency
division duplex (FDD) systems. To address this, extensive research has focused
on CSI compression and prediction, with neural network-based approaches gaining
momentum and being considered for integration into the 3GPP 5G-Advanced
standards. While deep learning has been effectively applied to CSI-limited
beamforming and handover optimization, reference signal allocation under such
constraints remains surprisingly underexplored. To fill this gap, we introduce
the concept of channel prediction-based reference signal allocation (CPRS),
which jointly optimizes channel prediction and DM-RS allocation to improve data
throughput without requiring CSI feedback. We further propose a
standards-compliant ViViT/CNN-based architecture that implements CPRS by
treating evolving CSI matrices as sequential image-like data, enabling
efficient and adaptive transmission in dynamic environments. Simulation results
using ray-tracing channel data generated in NVIDIA Sionna validate the proposed
method, showing up to 36.60% throughput improvement over benchmark strategies.

### 10. [MPC-based Coarse-to-Fine Motion Planning for Robotic Object Transportation in Cluttered Environments](http://arxiv.org/pdf/2507.11211v1)

Authors: Chen Cai, Ernesto Dickel Saraiva, Ya-jun Pan, Steven Liu

This letter presents a novel coarse-to-fine motion planning framework for
robotic manipulation in cluttered, unmodeled environments. The system
integrates a dual-camera perception setup with a B-spline-based model
predictive control (MPC) scheme. Initially, the planner generates feasible
global trajectories from partial and uncertain observations. As new visual data
are incrementally fused, both the environment model and motion planning are
progressively refined. A vision-based cost function promotes target-driven
exploration, while a refined kernel-perceptron collision detector enables
efficient constraint updates for real-time planning. The framework accommodates
closed-chain kinematics and supports dynamic replanning. Experiments on a
multi-arm platform validate its robustness and adaptability under uncertainties
and clutter.

### Machine Learning (Statistics Category)

### 1. [GOLFS: Feature Selection via Combining Both Global and Local Information for High Dimensional Clustering](http://arxiv.org/pdf/2507.10956v1)

Authors: Zhaoyu Xing, Yang Wan, Juan Wen, Wei Zhong

It is important to identify the discriminative features for high dimensional
clustering. However, due to the lack of cluster labels, the regularization
methods developed for supervised feature selection can not be directly applied.
To learn the pseudo labels and select the discriminative features
simultaneously, we propose a new unsupervised feature selection method, named
GlObal and Local information combined Feature Selection (GOLFS), for high
dimensional clustering problems. The GOLFS algorithm combines both local
geometric structure via manifold learning and global correlation structure of
samples via regularized self-representation to select the discriminative
features. The combination improves the accuracy of both feature selection and
clustering by exploiting more comprehensive information. In addition, an
iterative algorithm is proposed to solve the optimization problem and the
convergency is proved. Simulations and two real data applications demonstrate
the excellent finite-sample performance of GOLFS on both feature selection and
clustering.

### 2. [Urban delineation through the lens of commute networks: Leveraging graph embeddings to distinguish socioeconomic groups in cities](http://arxiv.org/pdf/2507.11057v1)

Authors: Devashish Khulbe, Stanislav Sobolevsky

Delineating areas within metropolitan regions stands as an important focus
among urban researchers, shedding light on the urban perimeters shaped by
evolving population dynamics. Applications to urban science are numerous, from
facilitating comparisons between delineated districts and administrative
divisions to informing policymakers of the shifting economic and labor
landscapes. In this study, we propose using commute networks sourced from the
census for the purpose of urban delineation, by modeling them with a Graph
Neural Network (GNN) architecture. We derive low-dimensional representations of
granular urban areas (nodes) using GNNs. Subsequently, nodes' embeddings are
clustered to identify spatially cohesive communities in urban areas. Our
experiments across the U.S. demonstrate the effectiveness of network embeddings
in capturing significant socioeconomic disparities between communities in
various cities, particularly in factors such as median household income. The
role of census mobility data in regional delineation is also noted, and we
establish the utility of GNNs in urban community detection, as a powerful
alternative to existing methods in this domain. The results offer insights into
the wider effects of commute networks and their use in building meaningful
representations of urban regions.

### 3. [Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank and Feature Selection](http://arxiv.org/pdf/2507.11136v1)

Authors: Afra Kilic, Kim Batselier

Tensor Network (TN) Kernel Machines speed up model learning by representing
parameters as low-rank TNs, reducing computation and memory use. However, most
TN-based Kernel methods are deterministic and ignore parameter uncertainty.
Further, they require manual tuning of model complexity hyperparameters like
tensor rank and feature dimensions, often through trial-and-error or
computationally costly methods like cross-validation. We propose Bayesian
Tensor Network Kernel Machines, a fully probabilistic framework that uses
sparsity-inducing hierarchical priors on TN factors to automatically infer
model complexity. This enables automatic inference of tensor rank and feature
dimensions, while also identifying the most relevant features for prediction,
thereby enhancing model interpretability. All the model parameters and
hyperparameters are treated as latent variables with corresponding priors.
Given the Bayesian approach and latent variable dependencies, we apply a
mean-field variational inference to approximate their posteriors. We show that
applying a mean-field approximation to TN factors yields a Bayesian ALS
algorithm with the same computational complexity as its deterministic
counterpart, enabling uncertainty quantification at no extra computational
cost. Experiments on synthetic and real-world datasets demonstrate the superior
performance of our model in prediction accuracy, uncertainty quantification,
interpretability, and scalability.

### 4. [How does Labeling Error Impact Contrastive Learning? A Perspective from Data Dimensionality Reduction](http://arxiv.org/pdf/2507.11161v1)

Authors: Jun Chen, Hong Chen, Yonghua Yu, Yiming Ying

In recent years, contrastive learning has achieved state-of-the-art
performance in the territory of self-supervised representation learning. Many
previous works have attempted to provide the theoretical understanding
underlying the success of contrastive learning. Almost all of them rely on a
default assumption, i.e., the label consistency assumption, which may not hold
in practice (the probability of failure is called labeling error) due to the
strength and randomness of common augmentation strategies, such as random
resized crop (RRC). This paper investigates the theoretical impact of labeling
error on the downstream classification performance of contrastive learning. We
first reveal several significant negative impacts of labeling error on
downstream classification risk. To mitigate these impacts, data dimensionality
reduction method (e.g., singular value decomposition, SVD) is applied on
original data to reduce false positive samples, and establish both theoretical
and empirical evaluations. Moreover, it is also found that SVD acts as a
double-edged sword, which may lead to the deterioration of downstream
classification accuracy due to the reduced connectivity of the augmentation
graph. Based on the above observations, we give the augmentation suggestion
that we should use some moderate embedding dimension (such as $512, 1024$ in
our experiments), data inflation, weak augmentation, and SVD to ensure large
graph connectivity and small labeling error to improve model performance.

### 5. [Neurosymbolic Reasoning Shortcuts under the Independence Assumption](http://arxiv.org/pdf/2507.11357v1)

Authors: Emile van Krieken, Pasquale Minervini, Edoardo Ponti, Antonio Vergari

The ubiquitous independence assumption among symbolic concepts in
neurosymbolic (NeSy) predictors is a convenient simplification: NeSy predictors
use it to speed up probabilistic reasoning. Recent works like van Krieken et
al. (2024) and Marconato et al. (2024) argued that the independence assumption
can hinder learning of NeSy predictors and, more crucially, prevent them from
correctly modelling uncertainty. There is, however, scepticism in the NeSy
community around the scenarios in which the independence assumption actually
limits NeSy systems (Faronius and Dos Martires, 2025). In this work, we settle
this question by formally showing that assuming independence among symbolic
concepts entails that a model can never represent uncertainty over certain
concept combinations. Thus, the model fails to be aware of reasoning shortcuts,
i.e., the pathological behaviour of NeSy predictors that predict correct
downstream tasks but for the wrong reasons.

### 6. [Joint space-time wind field data extrapolation and uncertainty quantification using nonparametric Bayesian dictionary learning](http://arxiv.org/pdf/2507.11385v1)

Authors: George D. Pasparakis, Ioannis A. Kougioumtzoglou, Michael D. Shields

A methodology is developed, based on nonparametric Bayesian dictionary
learning, for joint space-time wind field data extrapolation and estimation of
related statistics by relying on limited/incomplete measurements. Specifically,
utilizing sparse/incomplete measured data, a time-dependent optimization
problem is formulated for determining the expansion coefficients of an
associated low-dimensional representation of the stochastic wind field.
Compared to an alternative, standard, compressive sampling treatment of the
problem, the developed methodology exhibits the following advantages. First,
the Bayesian formulation enables also the quantification of the uncertainty in
the estimates. Second, the requirement in standard CS-based applications for an
a priori selection of the expansion basis is circumvented. Instead, this is
done herein in an adaptive manner based on the acquired data. Overall, the
methodology exhibits enhanced extrapolation accuracy, even in cases of
high-dimensional data of arbitrary form, and of relatively large extrapolation
distances. Thus, it can be used, potentially, in a wide range of wind
engineering applications where various constraints dictate the use of a limited
number of sensors. The efficacy of the methodology is demonstrated by
considering two case studies. The first relates to the extrapolation of
simulated wind velocity records consistent with a prescribed joint
wavenumber-frequency power spectral density in a three-dimensional domain (2D
and time). The second pertains to the extrapolation of four-dimensional (3D and
time) boundary layer wind tunnel experimental data that exhibit significant
spatial variability and non-Gaussian characteristics.

### 7. [An Interpretable AI framework Quantifying Traditional Chinese Medicine Principles Towards Enhancing and Integrating with Modern Biomedicine](http://arxiv.org/pdf/2507.11176v1)

Authors: Haoran Li, Xingye Cheng, Ziyang Huang, Jingyuan Luo, Qianqian Xu, Qiguang Zhao, Tianchen Guo, Yumeng Zhang, Linda Lidan Zhong, Zhaoxiang Bian, Leihan Tang, Aiping Lyu, Liang Tian

Traditional Chinese Medicine diagnosis and treatment principles, established
through centuries of trial-and-error clinical practice, directly maps
patient-specific symptom patterns to personalised herbal therapies. These
empirical holistic mapping principles offer valuable strategies to address
remaining challenges of reductionism methodologies in modern biomedicine.
However, the lack of a quantitative framework and molecular-level evidence has
limited their interpretability and reliability. Here, we present an AI
framework trained on ancient and classical TCM formula records to quantify the
symptom pattern-herbal therapy mappings. Interestingly, we find that empirical
TCM diagnosis and treatment are consistent with the encoding-decoding processes
in the AI model. This enables us to construct an interpretable TCM embedding
space (TCM-ES) using the model's quantitative representation of TCM principles.
Validated through broad and extensive TCM patient data, the TCM-ES offers
universal quantification of the TCM practice and therapeutic efficacy. We
further map biomedical entities into the TCM-ES through correspondence
alignment. We find that the principal directions of the TCM-ES are
significantly associated with key biological functions (such as metabolism,
immune, and homeostasis), and that the disease and herb embedding proximity
aligns with their genetic relationships in the human protein interactome, which
demonstrate the biological significance of TCM principles. Moreover, the TCM-ES
uncovers latent disease relationships, and provides alternative metric to
assess clinical efficacy for modern disease-drug pairs. Finally, we construct a
comprehensive and integrative TCM knowledge graph, which predicts potential
associations between diseases and targets, drugs, herbal compounds, and herbal
therapies, providing TCM-informed opportunities for disease analysis and drug
development.

### 8. [Improved sampling algorithms and Poincaré inequalities for non-log-concave distributions](http://arxiv.org/pdf/2507.11236v1)

Authors: Yuchen He, Zhehan Lei, Jianan Shao, Chihao Zhang

We study the problem of sampling from a distribution $\mu$ with density
$\propto e^{-V}$ for some potential function $V:\mathbb R^d\to \mathbb R$ with
query access to $V$ and $\nabla V$. We start with the following standard
assumptions:
  (1) The potential function $V$ is $L$-smooth.
  (2) The second moment $\mathbf{E}_{X\sim \mu}[\|X\|^2]\leq M$.
  Recently, He and Zhang (COLT'25) showed that the query complexity of sampling
from such distributions is at least
$\left(\frac{LM}{d\epsilon}\right)^{\Omega(d)}$ where $\epsilon$ is the desired
accuracy in total variation distance, and the Poincar\'e constant can be
arbitrarily large.
  Meanwhile, another common assumption in the study of diffusion based samplers
(see e.g., the work of Chen, Chewi, Li, Li, Salim and Zhang (ICLR'23))
strengthens the smoothness condition (1) to the following:
  (1*) The potential function of *every* distribution along the
Ornstein-Uhlenbeck process starting from $\mu$ is $L$-smooth.
  We show that under the assumptions (1*) and (2), the query complexity of
sampling from $\mu$ can be $\mathrm{poly}(L,d)\cdot
\left(\frac{Ld+M}{\epsilon^2}\right)^{\mathcal{O}(L+1)}$, which is polynomial
in $d$ and $\frac{1}{\epsilon}$ when $L=\mathcal{O}(1)$ and
$M=\mathrm{poly}(d)$. This improves the algorithm with quasi-polynomial query
complexity developed by Huang et al. (COLT'24). Our results imply that the
seemly moderate strengthening of the smoothness condition (1) to (1*) can lead
to an exponential gap in the query complexity of sampling algorithms.
  Moreover, we show that together with the assumption (1*) and the stronger
moment assumption that $\|X\|$ is $\lambda$-sub-Gaussian for $X\sim\mu$, the
Poincar\'e constant of $\mu$ is at most $\mathcal{O}(\lambda)^{2(L+1)}$. As an
application of our technique, we obtain improved estimate of the Poincar\'e
constant for mixture of Gaussians with the same covariance.

### 9. [Fast Last-Iterate Convergence of SGD in the Smooth Interpolation Regime](http://arxiv.org/pdf/2507.11274v1)

Authors: Amit Attia, Matan Schliserman, Uri Sherman, Tomer Koren

We study population convergence guarantees of stochastic gradient descent
(SGD) for smooth convex objectives in the interpolation regime, where the noise
at optimum is zero or near zero. The behavior of the last iterate of SGD in
this setting -- particularly with large (constant) stepsizes -- has received
growing attention in recent years due to implications for the training of
over-parameterized models, as well as to analyzing forgetting in continual
learning and to understanding the convergence of the randomized Kaczmarz method
for solving linear systems. We establish that after $T$ steps of SGD on
$\beta$-smooth convex loss functions with stepsize $\eta \leq 1/\beta$, the
last iterate exhibits expected excess risk $\widetilde{O}(1/(\eta
T^{1-\beta\eta/2}) + \eta T^{\beta\eta/2} \sigma_\star^2)$, where
$\sigma_\star^2$ denotes the variance of the stochastic gradients at the
optimum. In particular, for a well-tuned stepsize we obtain a near optimal
$\widetilde{O}(1/T + \sigma_\star/\sqrt{T})$ rate for the last iterate,
extending the results of Varre et al. (2021) beyond least squares regression;
and when $\sigma_\star=0$ we obtain a rate of $O(1/\sqrt{T})$ with
$\eta=1/\beta$, improving upon the best-known $O(T^{-1/4})$ rate recently
established by Evron et al. (2025) in the special case of realizable linear
regression.

### 10. [Local Pairwise Distance Matching for Backpropagation-Free Reinforcement Learning](http://arxiv.org/pdf/2507.11367v1)

Authors: Daniel Tanneberg

Training neural networks with reinforcement learning (RL) typically relies on
backpropagation (BP), necessitating storage of activations from the forward
pass for subsequent backward updates. Furthermore, backpropagating error
signals through multiple layers often leads to vanishing or exploding
gradients, which can degrade learning performance and stability. We propose a
novel approach that trains each layer of the neural network using local signals
during the forward pass in RL settings. Our approach introduces local,
layer-wise losses leveraging the principle of matching pairwise distances from
multi-dimensional scaling, enhanced with optional reward-driven guidance. This
method allows each hidden layer to be trained using local signals computed
during forward propagation, thus eliminating the need for backward passes and
storing intermediate activations. Our experiments, conducted with policy
gradient methods across common RL benchmarks, demonstrate that this
backpropagation-free method achieves competitive performance compared to their
classical BP-based counterpart. Additionally, the proposed method enhances
stability and consistency within and across runs, and improves performance
especially in challenging environments.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-16 PST.

### 1. [Enhancing pathological feature discrimination in diabetic retinopathy multi-classification with self-paced progressive multi-scale training](https://www.nature.com/articles/s41598-025-07050-1)

Authors: Qiuji Zhou et al.

### 2. [Multifunctional cells based neural architecture search for plant images classification](https://www.nature.com/articles/s41598-025-11829-7)

Authors: Lin Huang et al.

### 3. [New hybrid features extracted from US images for breast cancer classification](https://www.nature.com/articles/s41598-025-09554-2)

Authors: Gigi Tăbăcaru et al.

### 4. [Optimization of a multi-environmental detection model for tomato growth point buds based on multi-strategy improved YOLOv8](https://www.nature.com/articles/s41598-025-06692-5)

Authors: Jiang Liu et al.

### 5. [Nonlinear dynamics of self-sustaining waves in anisotropic media](https://www.nature.com/articles/s41598-025-11005-x)

Authors: Mostafa M. A. Khater et al.

### 6. [Diffusion probabilistic model for Tibetan painted sketch extraction](https://www.nature.com/articles/s41598-025-07638-7)

Authors: Fubo Wang et al.

### 7. [Integrating AI-generated content tools in higher education: a comparative analysis of interdisciplinary learning outcomes](https://www.nature.com/articles/s41598-025-10941-y)

Authors: Zhang Yan et al.

### 8. [Hybrid genetic algorithm and deep learning techniques for advanced side-channel attacks](https://www.nature.com/articles/s41598-025-06375-1)

Authors: Faisal Hameed et al.

### 9. [Dynamic graph structure evolution for node classification with missing attributes](https://www.nature.com/articles/s41598-025-09840-z)

Authors: Xiaomeng Song et al.

### 10. [Tomato leaf disease detection method based on improved YOLOv8n](https://www.nature.com/articles/s41598-025-00405-8)

Authors: Ming Chen et al.

