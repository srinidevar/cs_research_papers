# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-25 17:00:25.617433 PST.

### Artificial Intelligence

### 1. [E.A.R.T.H.: Structuring Creative Evolution through Model Error in Generative AI](http://arxiv.org/pdf/2507.18004v1)

Authors: Yusen Peng, Shuhua Mao

How can AI move beyond imitation toward genuine creativity? This paper
proposes the E.A.R.T.H. framework, a five-stage generative pipeline that
transforms model-generated errors into creative assets through Error
generation, Amplification, Refine selection, Transform, and Harness feedback.
Drawing on cognitive science and generative modeling, we posit that "creative
potential hides in failure" and operationalize this via structured prompts,
semantic scoring, and human-in-the-loop evaluation. Implemented using
LLaMA-2-7B-Chat, SBERT, BERTScore, CLIP, BLIP-2, and Stable Diffusion, the
pipeline employs a composite reward function based on novelty, surprise, and
relevance. At the Refine stage, creativity scores increase by 52.5% (1.179 to
1.898, t = -5.56, p < 0.001), with final outputs reaching 2.010 - a 70.4%
improvement. Refined slogans are 48.4% shorter, 40.7% more novel, with only a
4.0% drop in relevance. Cross-modal tests show strong slogan-to-image alignment
(CLIPScore: 0.249; BERTScore F1: 0.816). In human evaluations, 60% of outputs
scored >= 4.0, with metaphorical slogans (avg. 4.09) outperforming literal ones
(3.99). Feedback highlights stylistic precision and emotional resonance. These
results demonstrate that error-centered, feedback-driven generation enhances
creativity, offering a scalable path toward self-evolving, human-aligned
creative AI.

### 2. [AlphaGo Moment for Model Architecture Discovery](http://arxiv.org/pdf/2507.18074v1)

Authors: Yixiu Liu, Yang Nan, Weixian Xu, Xiangkun Hu, Lyumanshan Ye, Zhen Qin, Pengfei Liu

While AI systems demonstrate exponentially improving capabilities, the pace
of AI research itself remains linearly bounded by human cognitive capacity,
creating an increasingly severe development bottleneck. We present ASI-Arch,
the first demonstration of Artificial Superintelligence for AI research
(ASI4AI) in the critical domain of neural architecture discovery--a fully
autonomous system that shatters this fundamental constraint by enabling AI to
conduct its own architectural innovation. Moving beyond traditional Neural
Architecture Search (NAS), which is fundamentally limited to exploring
human-defined spaces, we introduce a paradigm shift from automated optimization
to automated innovation. ASI-Arch can conduct end-to-end scientific research in
the domain of architecture discovery, autonomously hypothesizing novel
architectural concepts, implementing them as executable code, training and
empirically validating their performance through rigorous experimentation and
past experience. ASI-Arch conducted 1,773 autonomous experiments over 20,000
GPU hours, culminating in the discovery of 106 innovative, state-of-the-art
(SOTA) linear attention architectures. Like AlphaGo's Move 37 that revealed
unexpected strategic insights invisible to human players, our AI-discovered
architectures demonstrate emergent design principles that systematically
surpass human-designed baselines and illuminate previously unknown pathways for
architectural innovation. Crucially, we establish the first empirical scaling
law for scientific discovery itself--demonstrating that architectural
breakthroughs can be scaled computationally, transforming research progress
from a human-limited to a computation-scalable process. We provide
comprehensive analysis of the emergent design patterns and autonomous research
capabilities that enabled these breakthroughs, establishing a blueprint for
self-accelerating AI systems.

### 3. [Actively evaluating and learning the distinctions that matter: Vaccine safety signal detection from emergency triage notes](http://arxiv.org/pdf/2507.18123v1)

Authors: Sedigh Khademi, Christopher Palmer, Muhammad Javed, Hazel Clothier, Jim Buttery, Gerardo Luis Dimaguila, Jim Black

The rapid development of COVID-19 vaccines has showcased the global
communitys ability to combat infectious diseases. However, the need for
post-licensure surveillance systems has grown due to the limited window for
safety data collection in clinical trials and early widespread implementation.
This study aims to employ Natural Language Processing techniques and Active
Learning to rapidly develop a classifier that detects potential vaccine safety
issues from emergency department notes. ED triage notes, containing expert,
succinct vital patient information at the point of entry to health systems, can
significantly contribute to timely vaccine safety signal surveillance. While
keyword-based classification can be effective, it may yield false positives and
demand extensive keyword modifications. This is exacerbated by the infrequency
of vaccination-related ED presentations and their similarity to other reasons
for ED visits. NLP offers a more accurate and efficient alternative, albeit
requiring annotated data, which is often scarce in the medical field. Active
learning optimizes the annotation process and the quality of annotated data,
which can result in faster model implementation and improved model performance.
This work combines active learning, data augmentation, and active learning and
evaluation techniques to create a classifier that is used to enhance vaccine
safety surveillance from ED triage notes.

### 4. [Decoupling Knowledge and Reasoning in LLMs: An Exploration Using Cognitive Dual-System Theory](http://arxiv.org/pdf/2507.18178v1)

Authors: Mutian Yang, Jiandong Gao, Ji Wu

While large language models (LLMs) leverage both knowledge and reasoning
during inference, the capacity to distinguish between them plays a pivotal role
in model analysis, interpretability, and development. Inspired by dual-system
cognitive theory, we propose a cognition attribution framework to decouple the
contribution of knowledge and reasoning. In particular, the cognition of LLMs
is decomposed into two distinct yet complementary phases: knowledge retrieval
(Phase 1) and reasoning adjustment (Phase 2). To separate these phases, LLMs
are prompted to generate answers under two different cognitive modes, fast
thinking and slow thinking, respectively. The performance under different
cognitive modes is analyzed to quantify the contribution of knowledge and
reasoning. This architecture is employed to 15 LLMs across 3 datasets. Results
reveal: (1) reasoning adjustment is domain-specific, benefiting
reasoning-intensive domains (e.g., mathematics, physics, and chemistry) and
potentially imparing knowledge-intensive domains. (2) Parameter scaling
improves both knowledge and reasoning, with knowledge improvements being more
pronounced. Additionally, parameter scaling make LLMs reasoning significantly
more prudent, while moderately more intelligent. (3) Knowledge primarily
resides in lower network layers, while reasoning operates in higher layers. Our
framework not only helps understand LLMs from a "decoupling" perspective, but
also provides new insights into existing research, including scaling laws,
hierarchical knowledge editing, and limitations of small-model reasoning.

### 5. [Comparing Non-minimal Semantics for Disjunction in Answer Set Programming](http://arxiv.org/pdf/2507.18198v1)

Authors: Felicidad Aguado, Pedro Cabalar, Brais Muñiz, Gilberto Pérez, Concepción Vidal

In this paper, we compare four different semantics for disjunction in Answer
Set Programming that, unlike stable models, do not adhere to the principle of
model minimality. Two of these approaches, Cabalar and Mu\~niz' \emph{Justified
Models} and Doherty and Szalas' \emph{Strongly Supported Models}, directly
provide an alternative non-minimal semantics for disjunction. The other two,
Aguado et al's \emph{Forks} and Shen and Eiter's \emph{Determining Inference}
(DI) semantics, actually introduce a new disjunction connective, but are
compared here as if they constituted new semantics for the standard disjunction
operator. We are able to prove that three of these approaches (Forks, Justified
Models and a reasonable relaxation of the DI semantics) actually coincide,
constituting a common single approach under different definitions. Moreover,
this common semantics always provides a superset of the stable models of a
program (in fact, modulo any context) and is strictly stronger than the fourth
approach (Strongly Supported Models), that actually treats disjunctions as in
classical logic.

### 6. [Foundations for Risk Assessment of AI in Protecting Fundamental Rights](http://arxiv.org/pdf/2507.18290v1)

Authors: Antonino Rotolo, Beatrice Ferrigno, Jose Miguel Angel Garcia Godinez, Claudio Novelli, Giovanni Sartor

This chapter introduces a conceptual framework for qualitative risk
assessment of AI, particularly in the context of the EU AI Act. The framework
addresses the complexities of legal compliance and fundamental rights
protection by itegrating definitional balancing and defeasible reasoning.
Definitional balancing employs proportionality analysis to resolve conflicts
between competing rights, while defeasible reasoning accommodates the dynamic
nature of legal decision-making. Our approach stresses the need for an analysis
of AI deployment scenarios and for identifying potential legal violations and
multi-layered impacts on fundamental rights. On the basis of this analysis, we
provide philosophical foundations for a logical account of AI risk analysis. In
particular, we consider the basic building blocks for conceptually grasping the
interaction between AI deployment scenarios and fundamental rights,
incorporating in defeasible reasoning definitional balancing and arguments
about the contextual promotion or demotion of rights. This layered approach
allows for more operative models of assessment of both high-risk AI systems and
General Purpose AI (GPAI) systems, emphasizing the broader applicability of the
latter. Future work aims to develop a formal model and effective algorithms to
enhance AI risk assessment, bridging theoretical insights with practical
applications to support responsible AI governance.

### 7. [The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams](http://arxiv.org/pdf/2507.18337v1)

Authors: Peter Baumgartner, Lachlan McGinness

We present our method for automatically marking Physics exams. The marking
problem consists in assessing typed student answers for correctness with
respect to a ground truth solution. This is a challenging problem that we seek
to tackle using a combination of a computer algebra system, an SMT solver and a
term rewriting system. A Large Language Model is used to interpret and remove
errors from student responses and rewrite these in a machine readable format.
Once formalized and language-aligned, the next step then consists in applying
automated reasoning techniques for assessing student solution correctness. We
consider two methods of automated theorem proving: off-the-shelf SMT solving
and term rewriting systems tailored for physics problems involving
trigonometric expressions. The development of the term rewrite system and
establishing termination and confluence properties was not trivial, and we
describe it in some detail in the paper. We evaluate our system on a rich pool
of over 1500 real-world student exam responses from the 2023 Australian Physics
Olympiad.

### 8. [Reasoning Beyond the Obvious: Evaluating Divergent and Convergent Thinking in LLMs for Financial Scenarios](http://arxiv.org/pdf/2507.18368v1)

Authors: Zhuang Qiang Bok, Watson Wei Khong Chua

Most reasoning benchmarks for LLMs emphasize factual accuracy or step-by-step
logic. In finance, however, professionals must not only converge on optimal
decisions but also generate creative, plausible futures under uncertainty. We
introduce ConDiFi, a benchmark that jointly evaluates divergent and convergent
thinking in LLMs for financial tasks.
  ConDiFi features 607 macro-financial prompts for divergent reasoning and 990
multi-hop adversarial MCQs for convergent reasoning. Using this benchmark, we
evaluated 14 leading models and uncovered striking differences. Despite high
fluency, GPT-4o underperforms on Novelty and Actionability. In contrast, models
like DeepSeek-R1 and Cohere Command R+ rank among the top for generating
actionable, insights suitable for investment decisions. ConDiFi provides a new
perspective to assess reasoning capabilities essential to safe and strategic
deployment of LLMs in finance.

### 9. [Revisiting LLM Reasoning via Information Bottleneck](http://arxiv.org/pdf/2507.18391v1)

Authors: Shiye Lei, Zhihao Cheng, Kai Jia, Dacheng Tao

Large language models (LLMs) have recently demonstrated remarkable progress
in reasoning capabilities through reinforcement learning with verifiable
rewards (RLVR). By leveraging simple rule-based rewards, RL effectively
incentivizes LLMs to produce extended chain-of-thought (CoT) reasoning
trajectories, progressively guiding them toward correct answers. However,
existing approaches remain largely heuristic and intuition-driven, limiting the
development of principled methodologies. In this paper, we present a
theoretical characterization of LLM reasoning grounded in information
bottleneck (IB) principle, introducing IB-aware reasoning optimization (IBRO),
a framework that encourages reasoning trajectories to be both informative about
the final correct answer and generalizable across diverse prompts. We derive a
practical token-level surrogate objective and propose an efficient
approximation, resulting in the lightweight IB regularization method. This
technique integrates seamlessly into existing RL-based post-training frameworks
without additional computational overhead, requiring only a one-line code
modification. Empirically, we validate IB regularization across multiple
mathematical reasoning benchmarks and RL algorithms, demonstrating consistent
improvements in LLM reasoning performance.

### 10. [Optimising Call Centre Operations using Reinforcement Learning: Value Iteration versus Proximal Policy Optimisation](http://arxiv.org/pdf/2507.18398v1)

Authors: Kwong Ho Li, Wathsala Karunarathne

This paper investigates the application of Reinforcement Learning (RL) to
optimise call routing in call centres to minimise client waiting time and staff
idle time. Two methods are compared: a model-based approach using Value
Iteration (VI) under known system dynamics, and a model-free approach using
Proximal Policy Optimisation (PPO) that learns from experience. For the
model-based approach, a theoretical model is used, while a simulation model
combining Discrete Event Simulation (DES) with the OpenAI Gym environment is
developed for model-free learning. Both models frame the problem as a Markov
Decision Process (MDP) within a Skills-Based Routing (SBR) framework, with
Poisson client arrivals and exponentially distributed service and abandonment
times. For policy evaluation, random, VI, and PPO policies are evaluated using
the simulation model. After 1,000 test episodes, PPO consistently achives the
highest rewards, along with the lowest client waiting time and staff idle time,
despite requiring longer training time.

### Hardware Architecture

### 1. [Designing High-Performance and Thermally Feasible Multi-Chiplet Architectures enabled by Non-bendable Glass Interposer](http://arxiv.org/pdf/2507.18040v1)

Authors: Harsh Sharma, Janardhan Rao Doppa, Umit Y. Ogras, Partha Pratim Pande

Multi-chiplet architectures enabled by glass interposer offer superior
electrical performance, enable higher bus widths due to reduced crosstalk, and
have lower capacitance in the redistribution layer than current silicon
interposer-based systems. These advantages result in lower energy per bit,
higher communication frequencies, and extended interconnect range. However,
deformation of the package (warpage) in glass interposer-based systems becomes
a critical challenge as system size increases, leading to severe mechanical
stress and reliability concerns. Beyond a certain size, conventional packaging
techniques fail to manage warpage effectively, necessitating new approaches to
mitigate warpage induced bending with scalable performance for glass interposer
based multi-chiplet systems. To address these inter-twined challenges, we
propose a thermal-, warpage-, and performance-aware design framework that
employs architecture and packaging co-optimization. The proposed framework
disintegrates the surface and embedded chiplets to balance conflicting design
objectives, ensuring optimal trade-offs between performance, power, and
structural reliability. Our experiments demonstrate that optimized
multi-chiplet architectures from our design framework achieve up to 64.7%
performance improvement and 40% power reduction compared to traditional 2.5D
systems to execute deep neural network workloads with lower fabrication costs.

### 2. [Real-Time Object Detection and Classification using YOLO for Edge FPGAs](http://arxiv.org/pdf/2507.18174v1)

Authors: Rashed Al Amin, Roman Obermaisser

Object detection and classification are crucial tasks across various
application domains, particularly in the development of safe and reliable
Advanced Driver Assistance Systems (ADAS). Existing deep learning-based methods
such as Convolutional Neural Networks (CNNs), Single Shot Detectors (SSDs), and
You Only Look Once (YOLO) have demonstrated high performance in terms of
accuracy and computational speed when deployed on Field-Programmable Gate
Arrays (FPGAs). However, despite these advances, state-of-the-art YOLO-based
object detection and classification systems continue to face challenges in
achieving resource efficiency suitable for edge FPGA platforms. To address this
limitation, this paper presents a resource-efficient real-time object detection
and classification system based on YOLOv5 optimized for FPGA deployment. The
proposed system is trained on the COCO and GTSRD datasets and implemented on
the Xilinx Kria KV260 FPGA board. Experimental results demonstrate a
classification accuracy of 99%, with a power consumption of 3.5W and a
processing speed of 9 frames per second (FPS). These findings highlight the
effectiveness of the proposed approach in enabling real-time,
resource-efficient object detection and classification for edge computing
applications.

### 3. [PRACtical: Subarray-Level Counter Update and Bank-Level Recovery Isolation for Efficient PRAC Rowhammer Mitigation](http://arxiv.org/pdf/2507.18581v1)

Authors: Ravan Nazaraliyev, Saber Ganjisaffar, Nurlan Nazaraliyev, Nael Abu-Ghazaleh

As DRAM density increases, Rowhammer becomes more severe due to heightened
charge leakage, reducing the number of activations needed to induce bit flips.
The DDR5 standard addresses this threat with in-DRAM per-row activation
counters (PRAC) and the Alert Back-Off (ABO) signal to trigger mitigation.
However, PRAC adds performance overhead by incrementing counters during the
precharge phase, and recovery refreshes stalls the entire memory channel, even
if only one bank is under attack.
  We propose PRACtical, a performance-optimized approach to PRAC+ABO that
maintains the same security guarantees. First, we reduce counter update latency
by introducing a centralized increment circuit, enabling overlap between
counter updates and subsequent row activations in other subarrays. Second, we
enhance the $RFM_{ab}$ mitigation by enabling bank-level granularity: instead
of stalling the entire channel, only affected banks are paused. This is
achieved through a DRAM-resident register that identifies attacked banks.
  PRACtical improves performance by 8% on average (up to 20%) over the
state-of-the-art, reduces energy by 19%, and limits performance degradation
from aggressive performance attacks to less than 6%, all while preserving
Rowhammer protection.

### 4. [Explicit Sign-Magnitude Encoders Enable Power-Efficient Multipliers](http://arxiv.org/pdf/2507.18179v1)

Authors: Felix Arnold, Maxence Bouvier, Ryan Amaudruz, Renzo Andri, Lukas Cavigelli

This work presents a method to maximize power-efficiency of fixed point
multiplier units by decomposing them into sub-components. First, an encoder
block converts the operands from a two's complement to a sign magnitude
representation, followed by a multiplier module which performs the compute
operation and outputs the resulting value in the original format. This allows
to leverage the power-efficiency of the Sign Magnitude encoding for the
multiplication. To ensure the computing format is not altered, those two
components are synthesized and optimized separately. Our method leads to
significant power savings for input values centered around zero, as commonly
encountered in AI workloads. Under a realistic input stream with values
normally distributed with a standard deviation of 3.0, post-synthesis
simulations of the 4-bit multiplier design show up to 12.9% lower switching
activity compared to synthesis without decomposition. Those gains are achieved
while ensuring compliance into any production-ready system as the overall
circuit stays logic-equivalent. With the compliance lifted and a slightly
smaller input range of -7 to +7, switching activity reductions can reach up to
33%. Additionally, we demonstrate that synthesis optimization methods based on
switching-activity-driven design space exploration can yield a further 5-10%
improvement in power-efficiency compared to a power agnostic approach.

### Computational Complexity

### 1. [Fagin's Theorem for Semiring Turing Machines](http://arxiv.org/pdf/2507.18375v1)

Authors: Guillermo Badia, Manfred Droste, Thomas Eiter, Rafael Kiesel, Carles Noguera, Erik Paul

In recent years, quantitative complexity over semirings has been intensively
investigated. An important problem in this context is to connect computational
complexity with logical expressiveness. In this paper we improve on the model
of \emph{Semiring Turing Machines} (distinct from so called weighted Turing
machines) introduced by Eiter \& Kiesel (Semiring Reasoning Frameworks in AI
and Their Computational Complexity, \emph{J. Artif. Intell. Res.}, 2023). Our
central result is a Fagin-style theorem for a new quantitative complexity class
using a suitable weighted logical formalism. We show that the quantitative
complexity class that we call \NPnewinf{$\mathcal{R}$}, where $\mathcal{R}$ is
a commutative semiring, can be captured using a version of weighted existential
second-order logic that allows for predicates interpreted as semiring-annotated
relations. This result provides a precise logical characterization of the power
series that form the class \NPnewinf{$\mathcal{R}$}. We also give the exact
relation between Eiter \& Kiesel's version of NP, called
\NPoldinf{$\mathcal{R}$}, and the class \NPnewinf{$\mathcal{R}$}. Incidentally,
we are able to recapture all the complexity results by Eiter \& Kiesel (2023)
in our new model, connecting a quantitative version of NP to various counting
complexity classes.

### 2. [The hidden subgroup problem for infinite groups](http://arxiv.org/pdf/2507.18499v1)

Authors: Greg Kuperberg

Following the example of Shor's algorithm for period-finding in the integers,
we explore the hidden subgroup problem (HSP) for discrete infinite groups. On
the hardness side, we show that HSP is NP-hard for the additive group of
rational numbers, and for normal subgroups of non-abelian free groups. We also
indirectly reduce a version of the short vector problem to HSP in
$\mathbb{Z}^k$ with pseudo-polynomial query cost. On the algorithm side, we
generalize the Shor-Kitaev algorithm for HSP in $\mathbb{Z}^k$ (with standard
polynomial query cost) to the case where the hidden subgroup has deficient rank
or equivalently infinite index. Finally, we outline a stretched exponential
time algorithm for the abelian hidden shift problem (AHShP), extending prior
work of the author as well as Regev and Peikert. It follows that HSP in any
finitely generated, virtually abelian group also has a stretched exponential
time algorithm.

### Computational Engineering

### 1. [Multiscale Neural PDE Surrogates for Prediction and Downscaling: Application to Ocean Currents](http://arxiv.org/pdf/2507.18067v1)

Authors: Abdessamad El-Kabid, Loubna Benabbou, Redouane Lguensat, Alex Hernández-García

Accurate modeling of physical systems governed by partial differential
equations is a central challenge in scientific computing. In oceanography,
high-resolution current data are critical for coastal management, environmental
monitoring, and maritime safety. However, available satellite products, such as
Copernicus data for sea water velocity at ~0.08 degrees spatial resolution and
global ocean models, often lack the spatial granularity required for detailed
local analyses. In this work, we (a) introduce a supervised deep learning
framework based on neural operators for solving PDEs and providing arbitrary
resolution solutions, and (b) propose downscaling models with an application to
Copernicus ocean current data. Additionally, our method can model surrogate
PDEs and predict solutions at arbitrary resolution, regardless of the input
resolution. We evaluated our model on real-world Copernicus ocean current data
and synthetic Navier-Stokes simulation datasets.

### 2. [On zero-order consistency residue and background pressure for the conservative SPH fluid dynamics](http://arxiv.org/pdf/2507.18210v1)

Authors: Feng Wang, Xiangyu Hu

As one of the major challenges for the conservative smoothed particle
hydrodynamics (SPH) method, the zero-order consistency issue, although thought
to be mitigated by the particle regularization scheme, such as the transport
velocity formulation, significantly damps the flow in a long channel for both
laminar and turbulent simulations. Building on this finding, this paper not
only thoroughly analyzes the damping reason in this pressure-driven channel
flow, but also relates this problem with the excessive numerical dissipation in
the gravity-driven free-surface flow. The common root cause of the non-physical
numerical damping in the two typical flow scenarios, the zero-order gradient
consistency residue, is exposed. The adverse influence of the background
pressure on the residue for the two scenarios is revealed and discussed. To
comprehensively understand the behavior of the residue and mitigate its
potential adverse effects, we conduct both theoretical analysis and numerical
experiments focusing on the key sensitive factors. For studying the
residue-induced non-physical energy dissipation in the gravity-driven
free-surface flow, the water depth and input dynamic pressure in the inviscid
standing wave case are tested. To investigate the velocity loss in the
pressure-driven channel flow, we examine the effects of the channel length,
resolution, and outlet pressure. The state-of-the-art reverse kernel gradient
correction technique is introduced for the two typical flows, and proved to be
effective in reducing the residue effect, but we find its correction capability
is fundamentally limited. Finally, the FDA nozzle, an engineering benchmark, is
tested to demonstrate the residue influence in a complex geometry, highlighting
the necessity of correction schemes in scenarios with unavoidable high
background pressure.

### 3. [A stabilized Two-Step Formulation of Maxwell's Equations in the time-domain](http://arxiv.org/pdf/2507.18235v1)

Authors: Leon Herles, Mario Mally, Jörg Ostrowski, Sebastian Schöps, Melina Merkel

Simulating electromagnetic fields across broad frequency ranges is
challenging due to numerical instabilities at low frequencies. This work
extends a stabilized two-step formulation of Maxwell's equations to the
time-domain. Using a Galerkin discretization in space, we apply two different
time-discretization schemes that are tailored to the first- and second-order in
time partial differential equations of the two-step solution procedure used
here. To address the low-frequency instability, we incorporate a generalized
tree-cotree gauge that removes the singularity of the curl-curl operator,
ensuring robustness even in the static limit. Numerical results on academic and
application-oriented 3D problems confirm stability, accuracy, and the method's
applicability to nonlinear, temperature-dependent materials.

### Computational Geometry

### 1. [Gromov-Hausdorff distance between chromatic metric pairs and stability of the six-pack](http://arxiv.org/pdf/2507.17994v1)

Authors: Ondřej Draganov, Sophie Rosenmeier, Nicolò Zava

Chromatic metric pairs consist of a metric space and a coloring function
partitioning a subset thereof into various colors. It is a natural extension of
the notion of chromatic point sets studied in chromatic topological data
analysis. A useful tool in the field is the six-pack, a collection of six
persistence diagrams, summarizing homological information about how the colored
subsets interact. We introduce a suitable generalization of the
Gromov-Hausdorff distance to compare chromatic metric pairs. We show some basic
properties and validate this definition by obtaining the stability of the
six-pack with respect to that distance. We conclude by discussing its
restriction to metric pairs and its role in the stability of the \v{C}ech
persistence diagrams.

### 2. [Explainable Mapper: Charting LLM Embedding Spaces Using Perturbation-Based Explanation and Verification Agents](http://arxiv.org/pdf/2507.18607v1)

Authors: Xinyuan Yan, Rita Sevastjanova, Sinie van der Ben, Mennatallah El-Assady, Bei Wang

Large language models (LLMs) produce high-dimensional embeddings that capture
rich semantic and syntactic relationships between words, sentences, and
concepts. Investigating the topological structures of LLM embedding spaces via
mapper graphs enables us to understand their underlying structures.
Specifically, a mapper graph summarizes the topological structure of the
embedding space, where each node represents a topological neighborhood
(containing a cluster of embeddings), and an edge connects two nodes if their
corresponding neighborhoods overlap. However, manually exploring these
embedding spaces to uncover encoded linguistic properties requires considerable
human effort. To address this challenge, we introduce a framework for
semi-automatic annotation of these embedding properties. To organize the
exploration process, we first define a taxonomy of explorable elements within a
mapper graph such as nodes, edges, paths, components, and trajectories. The
annotation of these elements is executed through two types of customizable
LLM-based agents that employ perturbation techniques for scalable and automated
analysis. These agents help to explore and explain the characteristics of
mapper elements and verify the robustness of the generated explanations. We
instantiate the framework within a visual analytics workspace and demonstrate
its effectiveness through case studies. In particular, we replicate findings
from prior research on BERT's embedding properties across various layers of its
architecture and provide further observations into the linguistic properties of
topological neighborhoods.

### Computation and Language

### 1. [Technical Report of TeleChat2, TeleChat2.5 and T1](http://arxiv.org/pdf/2507.18013v1)

Authors: Zihan Wang, Xinzhang Liu, Yitong Yao, Chao Wang, Yu Zhao, Zhihao Yang, Wenmin Deng, Kaipeng Jia, Jiaxin Peng, Yuyao Huang, Sishi Xiong, Zhuo Jiang, Kaidong Yu, Xiaohui Hu, Fubei Yao, Ruiyu Fang, Zhuoru Jiang, Ruiting Song, Qiyi Xie, Rui Xue, Xuewei He, Yanlei Xue, Zhu Yuan, Zhaoxi Zhang, Zilu Huang, Shiquan Wang, Xin Wang, Hanming Wu, Mingyuan Wang, Xufeng Zhan, Yuhan Sun, Zhaohu Xing, Yuhao Jiang, Bingkai Yang, Shuangyong Song, Yongxiang Li, Zhongjiang He, Xuelong Li

We introduce the latest series of TeleChat models: \textbf{TeleChat2},
\textbf{TeleChat2.5}, and \textbf{T1}, offering a significant upgrade over
their predecessor, TeleChat. Despite minimal changes to the model architecture,
the new series achieves substantial performance gains through enhanced training
strategies in both pre-training and post-training stages. The series begins
with \textbf{TeleChat2}, which undergoes pretraining on 10 trillion
high-quality and diverse tokens. This is followed by Supervised Fine-Tuning
(SFT) and Direct Preference Optimization (DPO) to further enhance its
capabilities. \textbf{TeleChat2.5} and \textbf{T1} expand the pipeline by
incorporating a continual pretraining phase with domain-specific datasets,
combined with reinforcement learning (RL) to improve performance in code
generation and mathematical reasoning tasks. The \textbf{T1} variant is
designed for complex reasoning, supporting long Chain-of-Thought (CoT)
reasoning and demonstrating substantial improvements in mathematics and coding.
In contrast, \textbf{TeleChat2.5} prioritizes speed, delivering rapid
inference. Both flagship models of \textbf{T1} and \textbf{TeleChat2.5} are
dense Transformer-based architectures with 115B parameters, showcasing
significant advancements in reasoning and general task performance compared to
the original TeleChat. Notably, \textbf{T1-115B} outperform proprietary models
such as OpenAI's o1-mini and GPT-4o. We publicly release \textbf{TeleChat2},
\textbf{TeleChat2.5} and \textbf{T1}, including post-trained versions with 35B
and 115B parameters, to empower developers and researchers with
state-of-the-art language models tailored for diverse applications.

### 2. [Hybrid and Unitary Fine-Tuning of Large Language Models: Methods and Benchmarking under Resource Constraints](http://arxiv.org/pdf/2507.18076v1)

Authors: Haomin Qi, Zihan Dai, Chengbo Huang

Fine-tuning large language models (LLMs) remains a computational bottleneck
due to their scale and memory demands. This paper presents a comprehensive
evaluation of parameter-efficient fine-tuning (PEFT) techniques, including
LoRA, BOFT, LoRA-GA, and uRNN, and introduces a novel hybrid strategy that
dynamically integrates BOFT's orthogonal stability with LoRA-GA's
gradient-aligned rapid convergence. By computing per-layer adaptive updates
guided by gradient norms, the hybrid method achieves superior convergence
efficiency and generalization across diverse tasks. We also explore, for the
first time, the adaptation of unitary RNN (uRNN) principles to
transformer-based LLMs, enhancing gradient stability through structured unitary
constraints. Empirical evaluations on four benchmarks -- GLUE, GSM8K, MT-Bench,
and HumanEval -- using models ranging from 7B to 405B parameters demonstrate
that our hybrid method consistently outperforms individual PEFT baselines,
approaching full fine-tuning accuracy while reducing resource consumption by up
to 2.1 times in training time and 50 percent in memory usage. These findings
establish the hybrid approach as a practical and scalable fine-tuning solution
for real-world deployment of LLMs under resource constraints.

### 3. [MathOPEval: A Fine-grained Evaluation Benchmark for Visual Operations of MLLMs in Mathematical Reasoning](http://arxiv.org/pdf/2507.18140v1)

Authors: Xiaoyuan Li, Moxin Li, Wenjie Wang, Rui Men, Yichang Zhang, Fuli Feng, Dayiheng Liu, Junyang Lin

Recent progress in Multi-modal Large Language Models (MLLMs) has enabled
step-by-step multi-modal mathematical reasoning by performing visual operations
based on the textual instructions. A promising approach uses code as an
intermediate representation to precisely express and manipulate the images in
the reasoning steps. However, existing evaluations focus mainly on text-only
reasoning outputs, leaving the MLLM's ability to perform accurate visual
operations via code largely unexplored. This work takes a first step toward
addressing that gap by evaluating MLLM's code-based capabilities in multi-modal
mathematical reasoning.Specifically, our framework focuses on two key
evaluation aspects: (1) Multi-modal Code Generation (MCG) evaluates the model's
ability to accurately understand and construct visualizations from scratch. (2)
Multi-modal Code Editing (MCE) assesses the model's capacity for fine-grained
operations, which include three types: Deletion, Modification and Annotation.
To evaluate the above tasks, we incorporate a dataset that covers the five most
popular types of mathematical figures, including geometric diagrams, function
plots, and three types of statistical charts, to provide a comprehensive and
effective measurement of existing MLLMs. Our experimental evaluation involves
nine mainstream MLLMs, and the results reveal that existing models still lag
significantly behind human performance in performing fine-grained visual
operations.

### 4. [TN-AutoRCA: Benchmark Construction and Agentic Framework for Self-Improving Alarm-Based Root Cause Analysis in Telecommunication Networks](http://arxiv.org/pdf/2507.18190v1)

Authors: Keyu Wu, Qianjin Yu, Manlin Mei, Ruiting Liu, Jun Wang, Kailai Zhang, Yelun Bao

Root Cause Analysis (RCA) in telecommunication networks is a critical task,
yet it presents a formidable challenge for Artificial Intelligence (AI) due to
its complex, graph-based reasoning requirements and the scarcity of realistic
benchmarks.

### 5. [Exploring the Impact of Instruction-Tuning on LLM's Susceptibility to Misinformation](http://arxiv.org/pdf/2507.18203v1)

Authors: Kyubeen Han, Junseo Jang, Hongjin Kim, Geunyeong Jeong, Harksoo Kim

Instruction-tuning enhances the ability of large language models (LLMs) to
follow user instructions more accurately, improving usability while reducing
harmful outputs. However, this process may increase the model's dependence on
user input, potentially leading to the unfiltered acceptance of misinformation
and the generation of hallucinations. Existing studies primarily highlight that
LLMs are receptive to external information that contradict their parametric
knowledge, but little research has been conducted on the direct impact of
instruction-tuning on this phenomenon. In our study, we investigate the impact
of instruction-tuning on LLM's susceptibility to misinformation. Our analysis
reveals that instruction-tuned LLMs are significantly more likely to accept
misinformation when it is presented by the user. A comparison with base models
shows that instruction-tuning increases reliance on user-provided information,
shifting susceptibility from the assistant role to the user role. Furthermore,
we explore additional factors influencing misinformation susceptibility, such
as the role of the user in prompt structure, misinformation length, and the
presence of warnings in the system prompt. Our findings underscore the need for
systematic approaches to mitigate unintended consequences of instruction-tuning
and enhance the reliability of LLMs in real-world applications.

### 6. [Prune&Comp: Free Lunch for Layer-Pruned LLMs via Iterative Pruning with Magnitude Compensation](http://arxiv.org/pdf/2507.18212v1)

Authors: Xinrui Chen, Hongxing Zhang, Fanyi Zeng, Yongxian Wei, Yizhi Wang, Xitong Ling, Guanghao Li, Chun Yuan

Layer pruning has emerged as a promising technique for compressing large
language models (LLMs) while achieving acceleration proportional to the pruning
ratio. In this work, we identify that removing any layer induces a significant
magnitude gap in hidden states, resulting in substantial performance
degradation. To address this issue, we propose Prune&Comp, a novel
plug-and-play layer pruning scheme that leverages magnitude compensation to
mitigate such gaps in a training-free manner. Specifically, we first estimate
the magnitude gap caused by layer removal and then eliminate this gap by
rescaling the remaining weights offline, with zero runtime overhead incurred.
We further demonstrate the advantages of Prune&Comp through an iterative
pruning strategy. When integrated with an iterative prune-and-compensate loop,
Prune&Comp consistently enhances existing layer pruning metrics. For instance,
when 5 layers of LLaMA-3-8B are pruned using the prevalent block influence
metric, Prune&Comp nearly halves the perplexity and retains 93.19\% of the
original model's question-answering performance, outperforming the baseline by
4.01%.

### 7. [Zero-shot OCR Accuracy of Low-Resourced Languages: A Comparative Analysis on Sinhala and Tamil](http://arxiv.org/pdf/2507.18264v1)

Authors: Nevidu Jayatilleke, Nisansa de Silva

Solving the problem of Optical Character Recognition (OCR) on printed text
for Latin and its derivative scripts can now be considered settled due to the
volumes of research done on English and other High-Resourced Languages (HRL).
However, for Low-Resourced Languages (LRL) that use unique scripts, it remains
an open problem. This study presents a comparative analysis of the zero-shot
performance of six distinct OCR engines on two LRLs: Sinhala and Tamil. The
selected engines include both commercial and open-source systems, aiming to
evaluate the strengths of each category. The Cloud Vision API, Surya, Document
AI, and Tesseract were evaluated for both Sinhala and Tamil, while Subasa OCR
and EasyOCR were examined for only one language due to their limitations. The
performance of these systems was rigorously analysed using five measurement
techniques to assess accuracy at both the character and word levels. According
to the findings, Surya delivered the best performance for Sinhala across all
metrics, with a WER of 2.61%. Conversely, Document AI excelled across all
metrics for Tamil, highlighted by a very low CER of 0.78%. In addition to the
above analysis, we also introduce a novel synthetic Tamil OCR benchmarking
dataset.

### 8. [StyleAdaptedLM: Enhancing Instruction Following Models with Efficient Stylistic Transfer](http://arxiv.org/pdf/2507.18294v1)

Authors: Pritika Ramu, Apoorv Saxena, Meghanath M Y, Varsha Sankar, Debraj Basu

Adapting LLMs to specific stylistic characteristics, like brand voice or
authorial tones, is crucial for enterprise communication but challenging to
achieve from corpora which lacks instruction-response formatting without
compromising instruction adherence. We introduce StyleAdaptedLM, a framework
that efficiently transfers stylistic traits to instruction-following models
using Low-Rank Adaptation (LoRA). LoRA adapters are first trained on a base
model with diverse unstructured stylistic corpora, then merged with a separate
instruction-following model. This enables robust stylistic customization
without paired data or sacrificing task performance. Experiments across
multiple datasets and models demonstrate improved stylistic consistency while
preserving instruction adherence, with human evaluations confirming
brand-specific convention uptake. StyleAdaptedLM offers an efficient path for
stylistic personalization in LLMs.

### 9. [BadReasoner: Planting Tunable Overthinking Backdoors into Large Reasoning Models for Fun or Profit](http://arxiv.org/pdf/2507.18305v1)

Authors: Biao Yi, Zekun Fei, Jianing Geng, Tong Li, Lihai Nie, Zheli Liu, Yiming Li

Large reasoning models (LRMs) have emerged as a significant advancement in
artificial intelligence, representing a specialized class of large language
models (LLMs) designed to tackle complex reasoning tasks. The defining
characteristic of LRMs lies in their extensive chain-of-thought (CoT) reasoning
capabilities. In this paper, we identify a previously unexplored attack vector
against LRMs, which we term "overthinking backdoors". We advance this concept
by proposing a novel tunable backdoor, which moves beyond simple on/off attacks
to one where an attacker can precisely control the extent of the model's
reasoning verbosity. Our attack is implemented through a novel data poisoning
methodology. It pairs a tunable trigger-where the number of repetitions signals
the desired intensity-with a correspondingly verbose CoT response. These
responses are programmatically generated by instructing a teacher LLM to inject
a controlled number of redundant refinement steps into a correct reasoning
process. The approach preserves output correctness, which ensures stealth and
establishes the attack as a pure resource-consumption vector. Extensive
empirical results on various LRMs demonstrate that our method can reliably
trigger a controllable, multi-fold increase in the length of the reasoning
process, without degrading the final answer's correctness. Our source code is
available at https://github.com/FZaKK/BadReasoner.

### 10. [Uncertainty Quantification for Evaluating Machine Translation Bias](http://arxiv.org/pdf/2507.18338v1)

Authors: Ieva Raminta Staliūnaitė, Julius Cheng, Andreas Vlachos

In machine translation (MT), when the source sentence includes a lexeme whose
gender is not overtly marked, but whose target-language equivalent requires
gender specification, the model must infer the appropriate gender from the
context and/or external knowledge. Studies have shown that MT models exhibit
biased behaviour, relying on stereotypes even when they clash with contextual
information. We posit that apart from confidently translating using the correct
gender when it is evident from the input, models should also maintain
uncertainty about the gender when it is ambiguous. Using recently proposed
metrics of semantic uncertainty, we find that models with high translation and
gender accuracy on unambiguous instances do not necessarily exhibit the
expected level of uncertainty in ambiguous ones. Similarly, debiasing has
independent effects on ambiguous and unambiguous translation instances.

### Cryptography and Security

### 1. [Removing Box-Free Watermarks for Image-to-Image Models via Query-Based Reverse Engineering](http://arxiv.org/pdf/2507.18034v1)

Authors: Haonan An, Guang Hua, Hangcheng Cao, Zhengru Fang, Guowen Xu, Susanto Rahardja, Yuguang Fang

The intellectual property of deep generative networks (GNets) can be
protected using a cascaded hiding network (HNet) which embeds watermarks (or
marks) into GNet outputs, known as box-free watermarking. Although both GNet
and HNet are encapsulated in a black box (called operation network, or ONet),
with only the generated and marked outputs from HNet being released to end
users and deemed secure, in this paper, we reveal an overlooked vulnerability
in such systems. Specifically, we show that the hidden GNet outputs can still
be reliably estimated via query-based reverse engineering, leaking the
generated and unmarked images, despite the attacker's limited knowledge of the
system. Our first attempt is to reverse-engineer an inverse model for HNet
under the stringent black-box condition, for which we propose to exploit the
query process with specially curated input images. While effective, this method
yields unsatisfactory image quality. To improve this, we subsequently propose
an alternative method leveraging the equivalent additive property of box-free
model watermarking and reverse-engineering a forward surrogate model of HNet,
with better image quality preservation. Extensive experimental results on image
processing and image generation tasks demonstrate that both attacks achieve
impressive watermark removal success rates (100%) while also maintaining
excellent image quality (reaching the highest PSNR of 34.69 dB), substantially
outperforming existing attacks, highlighting the urgent need for robust
defensive strategies to mitigate the identified vulnerability in box-free model
watermarking.

### 2. [PyPitfall: Dependency Chaos and Software Supply Chain Vulnerabilities in Python](http://arxiv.org/pdf/2507.18075v1)

Authors: Jacob Mahon, Chenxi Hou, Zhihao Yao

Python software development heavily relies on third-party packages. Direct
and transitive dependencies create a labyrinth of software supply chains. While
it is convenient to reuse code, vulnerabilities within these dependency chains
can propagate through dependencies, potentially affecting down-stream packages
and applications. PyPI, the official Python package repository, hosts many
packages and lacks a comprehensive analysis of the prevalence of vulnerable
dependencies. This paper introduces PyPitfall, a quantitative analysis of
vulnerable dependencies across the PyPI ecosystem. We analyzed the dependency
structures of 378,573 PyPI packages and identified 4,655 packages that
explicitly require at least one known-vulnerable version and 141,044 packages
that permit vulnerable versions within specified ranges. By characterizing the
ecosystem-wide dependency landscape and the security impact of transitive
dependencies, we aim to raise awareness of Python software supply chain
security.

### 3. [Conformidade com os Requisitos Legais de Privacidade de Dados: Um Estudo sobre Técnicas de Anonimização](http://arxiv.org/pdf/2507.18360v1)

Authors: André Menolli, Luiz Fernando Nunes, Thiago A. Coleti

The protection of personal data has become a central topic in software
development, especially with the implementation of the General Data Protection
Law (LGPD) in Brazil and the General Data Protection Regulation (GDPR) in the
European Union. With the enforcement of these laws, certain software quality
criteria have become mandatory, such as data anonymization, which is one of the
main aspects addressed by these regulations. The aim of this article is to
analyze data anonymization techniques and assess their effectiveness in
ensuring compliance with legal requirements and the utility of the data for its
intended purpose. Techniques such as aggregation, generalization, perturbation,
and k-anonymity were investigated and applied to datasets containing personal
and sensitive data. The analysis revealed significant variations in the
effectiveness of each method, highlighting the need to balance privacy and data
utility.

### 4. [Scout: Leveraging Large Language Models for Rapid Digital Evidence Discovery](http://arxiv.org/pdf/2507.18478v1)

Authors: Shariq Murtuza

Recent technological advancements and the prevalence of technology in day to
day activities have caused a major increase in the likelihood of the
involvement of digital evidence in more and more legal investigations.
Consumer-grade hardware is growing more powerful, with expanding memory and
storage sizes and enhanced processor capabilities. Forensics investigators
often have to sift through gigabytes of data during an ongoing investigation
making the process tedious. Memory forensics, disk analysis all are well
supported by state of the art tools that significantly lower the effort
required to be put in by a forensic investigator by providing string searches,
analyzing images file etc. During the course of the investigation a lot of
false positives are identified that need to be lowered. This work presents
Scout, a digital forensics framework that performs preliminary evidence
processing and prioritizing using large language models. Scout deploys
foundational language models to identify relevant artifacts from a large number
of potential evidence files (disk images, captured network packets, memory
dumps etc.) which would have taken longer to get identified. Scout employs text
based large language models can easily process files with textual information.
For the forensic analysis of multimedia files like audio, image, video, office
documents etc. multimodal models are employed by Scout. Scout was able to
identify and realize the evidence file that were of potential interest for the
investigator.

### 5. [Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment](http://arxiv.org/pdf/2507.18631v1)

Authors: Hao Li, Lijun Li, Zhenghao Lu, Xianyi Wei, Rui Li, Jing Shao, Lei Sha

With rapid advancement and increasing accessibility of LLMs, fine-tuning
aligned models has become a critical step for adapting them to real-world
applications, which makes the safety of this fine-tuning process more important
than ever. However, recent studies have highlighted a critical challenge: even
when fine-tuning with seemingly benign downstream datasets, the safety of
aligned LLMs can be compromised, making them more susceptible to malicious
instructions. In this paper, we show that fine-tuning datasets often contain
samples with safety-degrading features that are not easily identifiable on the
surface. These samples can significantly degrade the safety alignment of LLMs
during fine-tuning. To address this issue, we propose LARF, a
\textbf{L}ayer-\textbf{A}ware \textbf{R}epresentation \textbf{F}iltering
method. This method identifies safety-sensitive layers within the LLM and
leverages their representations to detect which data samples in the
post-training dataset contain safety-degrading features. Experimental results
demonstrate that LARF can effectively identify benign data with
safety-degrading features. After removing such data, the safety alignment
degradation caused by fine-tuning is mitigated. Please see our code at
\href{https://github.com/LLLeoLi/LARF}{https://github.com/LLLeoLi/LARF}.

### 6. [NWaaS: Nonintrusive Watermarking as a Service for X-to-Image DNN](http://arxiv.org/pdf/2507.18036v1)

Authors: Haonan An, Guang Hua, Yu Guo, Hangcheng Cao, Susanto Rahardja, Yuguang Fang

The intellectual property of deep neural network (DNN) models can be
protected with DNN watermarking, which embeds copyright watermarks into model
parameters (white-box), model behavior (black-box), or model outputs
(box-free), and the watermarks can be subsequently extracted to verify model
ownership or detect model theft. Despite recent advances, these existing
methods are inherently intrusive, as they either modify the model parameters or
alter the structure. This natural intrusiveness raises concerns about
watermarking-induced shifts in model behavior and the additional cost of
fine-tuning, further exacerbated by the rapidly growing model size. As a
result, model owners are often reluctant to adopt DNN watermarking in practice,
which limits the development of practical Watermarking as a Service (WaaS)
systems. To address this issue, we introduce Nonintrusive Watermarking as a
Service (NWaaS), a novel trustless paradigm designed for X-to-Image models, in
which we hypothesize that with the model untouched, an owner-defined watermark
can still be extracted from model outputs. Building on this concept, we propose
ShadowMark, a concrete implementation of NWaaS which addresses critical
deployment challenges by establishing a robust and nonintrusive side channel in
the protected model's black-box API, leveraging a key encoder and a watermark
decoder. It is significantly distinctive from existing solutions by attaining
the so-called absolute fidelity and being applicable to different DNN
architectures, while being also robust against existing attacks, eliminating
the fidelity-robustness trade-off. Extensive experiments on image-to-image,
noise-to-image, noise-and-text-to-image, and text-to-image models, demonstrate
the efficacy and practicality of ShadowMark for real-world deployment of
nonintrusive DNN watermarking.

### 7. [Your ATs to Ts: MITRE ATT&CK Attack Technique to P-SSCRM Task Mapping](http://arxiv.org/pdf/2507.18037v1)

Authors: Sivana Hamer, Jacob Bowen, Md Nazmul Haque, Chris Madden, Laurie Williams

The MITRE Adversarial Tactics, Techniques and Common Knowledge (MITRE ATT&CK)
Attack Technique to Proactive Software Supply Chain Risk Management Framework
(P-SSCRM) Task mapping described in this document helps software organizations
to determine how different tasks mitigate the attack techniques of software
supply chain attacks. The mapping was created through four independent
strategies to find agreed-upon mappings. Because each P-SSCRM task is mapped to
one or more tasks from the 10 frameworks, the mapping we provide is also a
mapping between MITRE ATT&CK and other prominent government and industry
frameworks.

### 8. [RECALLED: An Unbounded Resource Consumption Attack on Large Vision-Language Models](http://arxiv.org/pdf/2507.18053v1)

Authors: Haoran Gao, Yuanhe Zhang, Zhenhong Zhou, Lei Jiang, Fanyu Meng, Yujia Xiao, Kun Wang, Yang Liu, Junlan Feng

Resource Consumption Attacks (RCAs) have emerged as a significant threat to
the deployment of Large Language Models (LLMs). With the integration of vision
modalities, additional attack vectors exacerbate the risk of RCAs in large
vision-language models (LVLMs). However, existing red-teaming studies have
largely overlooked visual inputs as a potential attack surface, resulting in
insufficient mitigation strategies against RCAs in LVLMs. To address this gap,
we propose RECALLED (\textbf{RE}source \textbf{C}onsumption \textbf{A}ttack on
\textbf{L}arge Vision-\textbf{L}anguag\textbf{E} Mo\textbf{D}els), the first
approach for exploiting visual modalities to trigger unbounded RCAs
red-teaming. First, we present \textit{Vision Guided Optimization}, a
fine-grained pixel-level optimization, to obtain \textit{Output Recall}
adversarial perturbations, which can induce repeating output. Then, we inject
the perturbations into visual inputs, triggering unbounded generations to
achieve the goal of RCAs. Additionally, we introduce \textit{Multi-Objective
Parallel Losses} to generate universal attack templates and resolve
optimization conflicts when intending to implement parallel attacks. Empirical
results demonstrate that RECALLED increases service response latency by over 26
$\uparrow$, resulting in an additional 20\% increase in GPU utilization and
memory consumption. Our study exposes security vulnerabilities in LVLMs and
establishes a red-teaming framework that can facilitate future defense
development against RCAs.

### 9. [Understanding the Supply Chain and Risks of Large Language Model Applications](http://arxiv.org/pdf/2507.18105v1)

Authors: Yujie Ma, Lili Quan, Xiaofei Xie, Qiang Hu, Jiongchi Yu, Yao Zhang, Sen Chen

The rise of Large Language Models (LLMs) has led to the widespread deployment
of LLM-based systems across diverse domains. As these systems proliferate,
understanding the risks associated with their complex supply chains is
increasingly important. LLM-based systems are not standalone as they rely on
interconnected supply chains involving pretrained models, third-party
libraries, datasets, and infrastructure. Yet, most risk assessments narrowly
focus on model or data level, overlooking broader supply chain vulnerabilities.
While recent studies have begun to address LLM supply chain risks, there
remains a lack of benchmarks for systematic research.
  To address this gap, we introduce the first comprehensive dataset for
analyzing and benchmarking LLM supply chain security. We collect 3,859
real-world LLM applications and perform interdependency analysis, identifying
109,211 models, 2,474 datasets, and 9,862 libraries. We extract model
fine-tuning paths, dataset reuse, and library reliance, mapping the ecosystem's
structure. To evaluate security, we gather 1,555 risk-related issues-50 for
applications, 325 for models, 18 for datasets, and 1,229 for libraries from
public vulnerability databases.
  Using this dataset, we empirically analyze component dependencies and risks.
Our findings reveal deeply nested dependencies in LLM applications and
significant vulnerabilities across the supply chain, underscoring the need for
comprehensive security analysis. We conclude with practical recommendations to
guide researchers and developers toward safer, more trustworthy LLM-enabled
systems.

### 10. [An Improved ChaCha Algorithm Based on Quantum Random Number](http://arxiv.org/pdf/2507.18157v1)

Authors: Chao Liu, Shuai Zhao, Chenhao Jia, Gengran Hu, Tingting Cui

Due to the merits of high efficiency and strong security against timing and
side-channel attacks, ChaCha has been widely applied in real-time communication
and data streaming scenarios. However, with the rapid development of
AI-assisted cryptanalysis and quantum computing technologies, there are serious
challenges to the secure implementation of ChaCha cipher. To further strengthen
the security of ChaCha cipher, we propose an improved variant based on quantum
random numbers, i.e., Quantum Random Number Enhanced ChaCha (QRE-ChaCha).
Specifically, the design XORs the initial constants with quantum random numbers
and periodically injects quantum random numbers into selected state words
during odd rounds to enhance diffusion. Compared with the original ChaCha, the
present variant shows stronger resistance to differential attacks and generates
a keystream with statistical randomness, thereby offering increased robustness
against both classical and quantum attacks. To evaluate the security and
performance of the present ChaCha, our analysis proceeds in three main parts.
Firstly, we analyze its theoretical security in terms of quantum randomness and
attack testing, and conduct differential cryptanalysis with an automated search
method based on the Boolean satisfiability problem (SAT). Secondly, we subject
the keystream generated by the cipher to randomness tests using the NIST
statistical test suite and the GM/T 0005-2021 randomness testing standard.
Finally, we assess its encryption and decryption performance by measuring its
encryption speed on files of various sizes. According to the results, the
present ChaCha is significantly improved to resist differential attacks while
maintaining the high efficiency of the original ChaCha cipher, and its
keystream successfully passes statistical randomness tests using the NIST and
GM/T 0005-2021 standards, meeting cryptographic application requirements.

### Computer Vision and Pattern Recognition

### 1. [AG-VPReID.VIR: Bridging Aerial and Ground Platforms for Video-based Visible-Infrared Person Re-ID](http://arxiv.org/pdf/2507.17995v1)

Authors: Huy Nguyen, Kien Nguyen, Akila Pemasiri, Akmal Jahan, Clinton Fookes, Sridha Sridharan

Person re-identification (Re-ID) across visible and infrared modalities is
crucial for 24-hour surveillance systems, but existing datasets primarily focus
on ground-level perspectives. While ground-based IR systems offer nighttime
capabilities, they suffer from occlusions, limited coverage, and vulnerability
to obstructions--problems that aerial perspectives uniquely solve. To address
these limitations, we introduce AG-VPReID.VIR, the first aerial-ground
cross-modality video-based person Re-ID dataset. This dataset captures 1,837
identities across 4,861 tracklets (124,855 frames) using both UAV-mounted and
fixed CCTV cameras in RGB and infrared modalities. AG-VPReID.VIR presents
unique challenges including cross-viewpoint variations, modality discrepancies,
and temporal dynamics. Additionally, we propose TCC-VPReID, a novel
three-stream architecture designed to address the joint challenges of
cross-platform and cross-modality person Re-ID. Our approach bridges the domain
gaps between aerial-ground perspectives and RGB-IR modalities, through
style-robust feature learning, memory-based cross-view adaptation, and
intermediary-guided temporal modeling. Experiments show that AG-VPReID.VIR
presents distinctive challenges compared to existing datasets, with our
TCC-VPReID framework achieving significant performance gains across multiple
evaluation protocols. Dataset and code are available at
https://github.com/agvpreid25/AG-VPReID.VIR.

### 2. [Exploring the interplay of label bias with subgroup size and separability: A case study in mammographic density classification](http://arxiv.org/pdf/2507.17996v1)

Authors: Emma A. M. Stanley, Raghav Mehta, Mélanie Roschewitz, Nils D. Forkert, Ben Glocker

Systematic mislabelling affecting specific subgroups (i.e., label bias) in
medical imaging datasets represents an understudied issue concerning the
fairness of medical AI systems. In this work, we investigated how size and
separability of subgroups affected by label bias influence the learned features
and performance of a deep learning model. Therefore, we trained deep learning
models for binary tissue density classification using the EMory BrEast imaging
Dataset (EMBED), where label bias affected separable subgroups (based on
imaging manufacturer) or non-separable "pseudo-subgroups". We found that
simulated subgroup label bias led to prominent shifts in the learned feature
representations of the models. Importantly, these shifts within the feature
space were dependent on both the relative size and the separability of the
subgroup affected by label bias. We also observed notable differences in
subgroup performance depending on whether a validation set with clean labels
was used to define the classification threshold for the model. For instance,
with label bias affecting the majority separable subgroup, the true positive
rate for that subgroup fell from 0.898, when the validation set had clean
labels, to 0.518, when the validation set had biased labels. Our work
represents a key contribution toward understanding the consequences of label
bias on subgroup fairness in medical imaging AI.

### 3. [Registration beyond Points: General Affine Subspace Alignment via Geodesic Distance on Grassmann Manifold](http://arxiv.org/pdf/2507.17998v1)

Authors: Jaeho Shin, Hyeonjae Gil, Junwoo Jang, Maani Ghaffari, Ayoung Kim

Affine Grassmannian has been favored for expressing proximity between lines
and planes due to its theoretical exactness in measuring distances among
features. Despite this advantage, the existing method can only measure the
proximity without yielding the distance as an explicit function of rigid body
transformation. Thus, an optimizable distance function on the manifold has
remained underdeveloped, stifling its application in registration problems.
This paper is the first to explicitly derive an optimizable cost function
between two Grassmannian features with respect to rigid body transformation
($\mathbf{R}$ and $\mathbf{t}$). Specifically, we present a rigorous
mathematical proof demonstrating that the bases of high-dimensional linear
subspaces can serve as an explicit representation of the cost. Finally, we
propose an optimizable cost function based on the transformed bases that can be
applied to the registration problem of any affine subspace. Compared to vector
parameter-based approaches, our method is able to find a globally optimal
solution by directly minimizing the geodesic distance which is agnostic to
representation ambiguity. The resulting cost function and its extension to the
inlier-set maximizing \ac{BnB} solver have been demonstrated to improve the
convergence of existing solutions or outperform them in various computer vision
tasks. The code is available on
https://github.com/joomeok/GrassmannRegistration.

### 4. [Celeb-DF++: A Large-scale Challenging Video DeepFake Benchmark for Generalizable Forensics](http://arxiv.org/pdf/2507.18015v1)

Authors: Yuezun Li, Delong Zhu, Xinjie Cui, Siwei Lyu

The rapid advancement of AI technologies has significantly increased the
diversity of DeepFake videos circulating online, posing a pressing challenge
for \textit{generalizable forensics}, \ie, detecting a wide range of unseen
DeepFake types using a single model. Addressing this challenge requires
datasets that are not only large-scale but also rich in forgery diversity.
However, most existing datasets, despite their scale, include only a limited
variety of forgery types, making them insufficient for developing generalizable
detection methods. Therefore, we build upon our earlier Celeb-DF dataset and
introduce {Celeb-DF++}, a new large-scale and challenging video DeepFake
benchmark dedicated to the generalizable forensics challenge. Celeb-DF++ covers
three commonly encountered forgery scenarios: Face-swap (FS), Face-reenactment
(FR), and Talking-face (TF). Each scenario contains a substantial number of
high-quality forged videos, generated using a total of 22 various recent
DeepFake methods. These methods differ in terms of architectures, generation
pipelines, and targeted facial regions, covering the most prevalent DeepFake
cases witnessed in the wild. We also introduce evaluation protocols for
measuring the generalizability of 24 recent detection methods, highlighting the
limitations of existing detection methods and the difficulty of our new
dataset.

### 5. [High-fidelity 3D Gaussian Inpainting: preserving multi-view consistency and photorealistic details](http://arxiv.org/pdf/2507.18023v1)

Authors: Jun Zhou, Dinghao Li, Nannan Li, Mingjie Wang

Recent advancements in multi-view 3D reconstruction and novel-view synthesis,
particularly through Neural Radiance Fields (NeRF) and 3D Gaussian Splatting
(3DGS), have greatly enhanced the fidelity and efficiency of 3D content
creation. However, inpainting 3D scenes remains a challenging task due to the
inherent irregularity of 3D structures and the critical need for maintaining
multi-view consistency. In this work, we propose a novel 3D Gaussian inpainting
framework that reconstructs complete 3D scenes by leveraging sparse inpainted
views. Our framework incorporates an automatic Mask Refinement Process and
region-wise Uncertainty-guided Optimization. Specifically, we refine the
inpainting mask using a series of operations, including Gaussian scene
filtering and back-projection, enabling more accurate localization of occluded
regions and realistic boundary restoration. Furthermore, our Uncertainty-guided
Fine-grained Optimization strategy, which estimates the importance of each
region across multi-view images during training, alleviates multi-view
inconsistencies and enhances the fidelity of fine details in the inpainted
results. Comprehensive experiments conducted on diverse datasets demonstrate
that our approach outperforms existing state-of-the-art methods in both visual
quality and view consistency.

### 6. [Emotion Recognition from Skeleton Data: A Comprehensive Survey](http://arxiv.org/pdf/2507.18026v1)

Authors: Haifeng Lu, Jiuyi Chen, Zhen Zhang, Ruida Liu, Runhao Zeng, Xiping Hu

Emotion recognition through body movements has emerged as a compelling and
privacy-preserving alternative to traditional methods that rely on facial
expressions or physiological signals. Recent advancements in 3D skeleton
acquisition technologies and pose estimation algorithms have significantly
enhanced the feasibility of emotion recognition based on full-body motion. This
survey provides a comprehensive and systematic review of skeleton-based emotion
recognition techniques. First, we introduce psychological models of emotion and
examine the relationship between bodily movements and emotional expression.
Next, we summarize publicly available datasets, highlighting the differences in
data acquisition methods and emotion labeling strategies. We then categorize
existing methods into posture-based and gait-based approaches, analyzing them
from both data-driven and technical perspectives. In particular, we propose a
unified taxonomy that encompasses four primary technical paradigms: Traditional
approaches, Feat2Net, FeatFusionNet, and End2EndNet. Representative works
within each category are reviewed and compared, with benchmarking results
across commonly used datasets. Finally, we explore the extended applications of
emotion recognition in mental health assessment, such as detecting depression
and autism, and discuss the open challenges and future research directions in
this rapidly evolving field.

### 7. [BokehDiff: Neural Lens Blur with One-Step Diffusion](http://arxiv.org/pdf/2507.18060v1)

Authors: Chengxuan Zhu, Qingnan Fan, Qi Zhang, Jinwei Chen, Huaqi Zhang, Chao Xu, Boxin Shi

We introduce BokehDiff, a novel lens blur rendering method that achieves
physically accurate and visually appealing outcomes, with the help of
generative diffusion prior. Previous methods are bounded by the accuracy of
depth estimation, generating artifacts in depth discontinuities. Our method
employs a physics-inspired self-attention module that aligns with the image
formation process, incorporating depth-dependent circle of confusion constraint
and self-occlusion effects. We adapt the diffusion model to the one-step
inference scheme without introducing additional noise, and achieve results of
high quality and fidelity. To address the lack of scalable paired data, we
propose to synthesize photorealistic foregrounds with transparency with
diffusion models, balancing authenticity and scene diversity.

### 8. [Adapting Large VLMs with Iterative and Manual Instructions for Generative Low-light Enhancement](http://arxiv.org/pdf/2507.18064v1)

Authors: Xiaoran Sun, Liyan Wang, Cong Wang, Yeying Jin, Kin-man Lam, Zhixun Su, Yang Yang, Jinshan Pan

Most existing low-light image enhancement (LLIE) methods rely on pre-trained
model priors, low-light inputs, or both, while neglecting the semantic guidance
available from normal-light images. This limitation hinders their effectiveness
in complex lighting conditions. In this paper, we propose VLM-IMI, a novel
framework that leverages large vision-language models (VLMs) with iterative and
manual instructions (IMIs) for LLIE. VLM-IMI incorporates textual descriptions
of the desired normal-light content as enhancement cues, enabling semantically
informed restoration. To effectively integrate cross-modal priors, we introduce
an instruction prior fusion module, which dynamically aligns and fuses image
and text features, promoting the generation of detailed and semantically
coherent outputs. During inference, we adopt an iterative and manual
instruction strategy to refine textual instructions, progressively improving
visual quality. This refinement enhances structural fidelity, semantic
alignment, and the recovery of fine details under extremely low-light
conditions. Extensive experiments across diverse scenarios demonstrate that
VLM-IMI outperforms state-of-the-art methods in both quantitative metrics and
perceptual quality. The source code is available at
https://github.com/sunxiaoran01/VLM-IMI.

### 9. [T2VWorldBench: A Benchmark for Evaluating World Knowledge in Text-to-Video Generation](http://arxiv.org/pdf/2507.18107v1)

Authors: Yubin Chen, Xuyang Guo, Zhenmei Shi, Zhao Song, Jiahao Zhang

Text-to-video (T2V) models have shown remarkable performance in generating
visually reasonable scenes, while their capability to leverage world knowledge
for ensuring semantic consistency and factual accuracy remains largely
understudied. In response to this challenge, we propose T2VWorldBench, the
first systematic evaluation framework for evaluating the world knowledge
generation abilities of text-to-video models, covering 6 major categories, 60
subcategories, and 1,200 prompts across a wide range of domains, including
physics, nature, activity, culture, causality, and object. To address both
human preference and scalable evaluation, our benchmark incorporates both human
evaluation and automated evaluation using vision-language models (VLMs). We
evaluated the 10 most advanced text-to-video models currently available,
ranging from open source to commercial models, and found that most models are
unable to understand world knowledge and generate truly correct videos. These
findings point out a critical gap in the capability of current text-to-video
models to leverage world knowledge, providing valuable research opportunities
and entry points for constructing models with robust capabilities for
commonsense reasoning and factual generation.

### 10. [Unsupervised Domain Adaptation for 3D LiDAR Semantic Segmentation Using Contrastive Learning and Multi-Model Pseudo Labeling](http://arxiv.org/pdf/2507.18176v1)

Authors: Abhishek Kaushik, Norbert Haala, Uwe Soergel

Addressing performance degradation in 3D LiDAR semantic segmentation due to
domain shifts (e.g., sensor type, geographical location) is crucial for
autonomous systems, yet manual annotation of target data is prohibitive. This
study addresses the challenge using Unsupervised Domain Adaptation (UDA) and
introduces a novel two-stage framework to tackle it. Initially, unsupervised
contrastive learning at the segment level is used to pre-train a backbone
network, enabling it to learn robust, domain-invariant features without labels.
Subsequently, a multi-model pseudo-labeling strategy is introduced, utilizing
an ensemble of diverse state-of-the-art architectures (including projection,
voxel, hybrid, and cylinder-based methods). Predictions from these models are
aggregated via hard voting to generate high-quality, refined pseudo-labels for
the unlabeled target domain, mitigating single-model biases. The contrastively
pre-trained network is then fine-tuned using these robust pseudo-labels.
Experiments adapting from SemanticKITTI to unlabeled target datasets
(SemanticPOSS, SemanticSlamantic) demonstrate significant improvements in
segmentation accuracy compared to direct transfer and single-model UDA
approaches. These results highlight the effectiveness of combining contrastive
pre-training with refined ensemble pseudo-labeling for bridging complex domain
gaps without requiring target domain annotations.

### Computers and Society

### 1. [Countering Privacy Nihilism](http://arxiv.org/pdf/2507.18253v1)

Authors: Severin Engelmann, Helen Nissenbaum

Of growing concern in privacy scholarship is artificial intelligence (AI), as
a powerful producer of inferences. Taken to its limits, AI may be presumed
capable of inferring "everything from everything," thereby making untenable any
normative scheme, including privacy theory and privacy regulation, which rests
on protecting privacy based on categories of data - sensitive versus
non-sensitive, private versus public. Discarding data categories as a normative
anchoring in privacy and data protection as a result of an unconditional
acceptance of AI's inferential capacities is what we call privacy nihilism. An
ethically reasoned response to AI inferences requires a sober consideration of
AI capabilities rather than issuing an epistemic carte blanche. We introduce
the notion of conceptual overfitting to expose how privacy nihilism turns a
blind eye toward flawed epistemic practices in AI development. Conceptual
overfitting refers to the adoption of norms of convenience that simplify the
development of AI models by forcing complex constructs to fit data that are
conceptually under-representative or even irrelevant. While conceptual
overfitting serves as a helpful device to counter normative suggestions
grounded in hyperbolic AI capability claims, AI inferences shake any privacy
regulation that hinges protections based on restrictions around data
categories. We propose moving away from privacy frameworks that focus solely on
data type, neglecting all other factors. Theories like contextual integrity
evaluate the normative value of privacy across several parameters, including
the type of data, the actors involved in sharing it, and the purposes for which
the information is used.

### 2. [What does the public want their local government to hear? A data-driven case study of public comments across the state of Michigan](http://arxiv.org/pdf/2507.18431v1)

Authors: Chang Ge, Justine Zhang, Haofei Xu, Yanna Krupnikov, Jenna Bednar, Sabina Tomkins

City council meetings are vital sites for civic participation where the
public can speak directly to their local government. By addressing city
officials and calling on them to take action, public commenters can potentially
influence policy decisions spanning a broad range of concerns, from housing, to
sustainability, to social justice. Yet studies of these meetings have often
been limited by the availability of large-scale, geographically-diverse data.
Relying on local governments' increasing use of YouTube and other technologies
to archive their public meetings, we propose a framework that characterizes
comments along two dimensions: the local concerns where concerns are situated
(e.g., housing, election administration), and the societal concerns raised
(e.g., functional democracy, anti-racism). Based on a large record of public
comments we collect from 15 cities in Michigan, we produce data-driven
taxonomies of the local concerns and societal concerns that these comments
cover, and employ machine learning methods to scalably apply our taxonomies
across the entire dataset. We then demonstrate how our framework allows us to
examine the salient local concerns and societal concerns that arise in our
data, as well as how these aspects interact.

### 3. [Recommender systems, representativeness, and online music: A psychosocial analysis of Italian listeners](http://arxiv.org/pdf/2507.18169v1)

Authors: Lorenzo Porcaro, Chiara Monaldi

Recommender systems shape music listening worldwide due to their widespread
adoption in online platforms. Growing concerns about representational harms
that these systems may cause are nowadays part of the scientific and public
debate, wherein music listener perspectives are oftentimes reported and
discussed from a cognitive-behaviorism perspective, but rarely contextualised
under a psychosocial and cultural lens. We proceed in this direction, by
interviewing a group of Italian music listeners and analysing their narratives
through Emotional Textual Analysis. Thanks to this, we identify shared cultural
repertoires that reveal people's complex relationship with listening practices:
even when familiar with online platforms, listeners may still lack a critical
understanding of recommender systems. Moreover, representational issues,
particularly gender disparities, seem not yet fully grasped in the context of
online music listening. This study underscores the need for interdisciplinary
research to address representational harms, and the role of algorithmic
awareness and digital literacy in developing trustworthy recommender systems.

### 4. [A Concept for Efficient Scalability of Automated Driving Allowing for Technical, Legal, Cultural, and Ethical Differences](http://arxiv.org/pdf/2507.18326v1)

Authors: Lars Ullrich, Michael Buchholz, Jonathan Petit, Klaus Dietmayer, Knut Graichen

Efficient scalability of automated driving (AD) is key to reducing costs,
enhancing safety, conserving resources, and maximizing impact. However,
research focuses on specific vehicles and context, while broad deployment
requires scalability across various configurations and environments.
Differences in vehicle types, sensors, actuators, but also traffic regulations,
legal requirements, cultural dynamics, or even ethical paradigms demand high
flexibility of data-driven developed capabilities. In this paper, we address
the challenge of scalable adaptation of generic capabilities to desired systems
and environments. Our concept follows a two-stage fine-tuning process. In the
first stage, fine-tuning to the specific environment takes place through a
country-specific reward model that serves as an interface between technological
adaptations and socio-political requirements. In the second stage,
vehicle-specific transfer learning facilitates system adaptation and governs
the validation of design decisions. In sum, our concept offers a data-driven
process that integrates both technological and socio-political aspects,
enabling effective scalability across technical, legal, cultural, and ethical
differences.

### 5. [PALM: PAnoramic Learning Map Integrating Learning Analytics and Curriculum Map for Scalable Insights Across Courses](http://arxiv.org/pdf/2507.18393v1)

Authors: Mahiro Ozaki, Li Chen, Shotaro Naganuma, Valdemar Švábenský, Fumiya Okubo, Atsushi Shimada

This study proposes and evaluates the PAnoramic Learning Map (PALM), a
learning analytics (LA) dashboard designed to address the scalability
challenges of LA by integrating curriculum-level information. Traditional LA
research has predominantly focused on individual courses or learners and often
lacks a framework that considers the relationships between courses and the
long-term trajectory of learning. To bridge this gap, PALM was developed to
integrate multilayered educational data into a curriculum map, enabling
learners to intuitively understand their learning records and academic
progression. We conducted a system evaluation to assess PALM's effectiveness in
two key areas: (1) its impact on students' awareness of their learning
behaviors, and (2) its comparative performance against existing systems. The
results indicate that PALM enhances learners' awareness of study planning and
reflection, particularly by improving perceived behavioral control through the
visual presentation of individual learning histories and statistical trends,
which clarify the links between learning actions and outcomes. Although PALM
requires ongoing refinement as a system, it received significantly higher
evaluations than existing systems in terms of visual appeal and usability. By
serving as an information resource with previously inaccessible insights, PALM
enhances self-regulated learning and engagement, representing a significant
step beyond conventional LA toward a comprehensive and scalable approach.

### 6. [Agentic AI framework for End-to-End Medical Data Inference](http://arxiv.org/pdf/2507.18115v1)

Authors: Soorya Ram Shimgekar, Shayan Vassef, Abhay Goyal, Navin Kumar, Koustuv Saha

Building and deploying machine learning solutions in healthcare remains
expensive and labor-intensive due to fragmented preprocessing workflows, model
compatibility issues, and stringent data privacy constraints. In this work, we
introduce an Agentic AI framework that automates the entire clinical data
pipeline, from ingestion to inference, through a system of modular,
task-specific agents. These agents handle both structured and unstructured
data, enabling automatic feature selection, model selection, and preprocessing
recommendation without manual intervention. We evaluate the system on publicly
available datasets from geriatrics, palliative care, and colonoscopy imaging.
For example, in the case of structured data (anxiety data) and unstructured
data (colonoscopy polyps data), the pipeline begins with file-type detection by
the Ingestion Identifier Agent, followed by the Data Anonymizer Agent ensuring
privacy compliance, where we first identify the data type and then anonymize
it. The Feature Extraction Agent identifies features using an embedding-based
approach for tabular data, extracting all column names, and a multi-stage
MedGemma-based approach for image data, which infers modality and disease name.
These features guide the Model-Data Feature Matcher Agent in selecting the
best-fit model from a curated repository. The Preprocessing Recommender Agent
and Preprocessing Implementor Agent then apply tailored preprocessing based on
data type and model requirements. Finally, the ``Model Inference Agent" runs
the selected model on the uploaded data and generates interpretable outputs
using tools like SHAP, LIME, and DETR attention maps. By automating these
high-friction stages of the ML lifecycle, the proposed framework reduces the
need for repeated expert intervention, offering a scalable, cost-efficient
pathway for operationalizing AI in clinical environments.

### Databases

### 1. [Factual Inconsistencies in Multilingual Wikipedia Tables](http://arxiv.org/pdf/2507.18406v1)

Authors: Silvia Cappa, Lingxiao Kong, Pille-Riin Peet, Fanfu Wei, Yuchen Zhou, Jan-Christoph Kalo

Wikipedia serves as a globally accessible knowledge source with content in
over 300 languages. Despite covering the same topics, the different versions of
Wikipedia are written and updated independently. This leads to factual
inconsistencies that can impact the neutrality and reliability of the
encyclopedia and AI systems, which often rely on Wikipedia as a main training
source. This study investigates cross-lingual inconsistencies in Wikipedia's
structured content, with a focus on tabular data. We developed a methodology to
collect, align, and analyze tables from Wikipedia multilingual articles,
defining categories of inconsistency. We apply various quantitative and
qualitative metrics to assess multilingual alignment using a sample dataset.
These insights have implications for factual verification, multilingual
knowledge interaction, and design for reliable AI systems leveraging Wikipedia
content.

### Distributed, Parallel, and Cluster Computing

### 1. [C-Koordinator: Interference-aware Management for Large-scale and Co-located Microservice Clusters](http://arxiv.org/pdf/2507.18005v1)

Authors: Shengye Song, Minxian Xu, Zuowei Zhang, Chengxi Gao, Fansong Zeng, Yu Ding, Kejiang Ye, Chengzhong Xu

Microservices transform traditional monolithic applications into lightweight,
loosely coupled application components and have been widely adopted in many
enterprises. Cloud platform infrastructure providers enhance the resource
utilization efficiency of microservices systems by co-locating different
microservices. However, this approach also introduces resource competition and
interference among microservices. Designing interference-aware strategies for
large-scale, co-located microservice clusters is crucial for enhancing resource
utilization and mitigating competition-induced interference. These challenges
are further exacerbated by unreliable metrics, application diversity, and node
heterogeneity.
  In this paper, we first analyze the characteristics of large-scale and
co-located microservices clusters at Alibaba and further discuss why cycle per
instruction (CPI) is adopted as a metric for interference measurement in
large-scale production clusters, as well as how to achieve accurate prediction
of CPI through multi-dimensional metrics. Based on CPI interference prediction
and analysis, we also present the design of the C-Koordinator platform, an
open-source solution utilized in Alibaba cluster, which incorporates
co-location and interference mitigation strategies. The interference prediction
models consistently achieve over 90.3% accuracy, enabling precise prediction
and rapid mitigation of interference in operational environments. As a result,
application latency is reduced and stabilized across all percentiles (P50, P90,
P99) response time (RT), achieving improvements ranging from 16.7% to 36.1%
under various system loads compared with state-of-the-art system. These results
demonstrate the system's ability to maintain smooth application performance in
co-located environments.

### 2. [Unlock the Potential of Fine-grained LLM Serving via Dynamic Module Scaling](http://arxiv.org/pdf/2507.18006v1)

Authors: Jingfeng Wu, Yiyuan He, Minxian Xu, Xitong Gao, Kejiang Ye, Chengzhong Xu

The rise of large language models (LLMs) has created new opportunities across
various fields but has also introduced significant challenges in resource
management. Current LLM serving systems face a fundamental tension: balancing
serving demands with limited resources while adapting to unpredictable traffic
patterns. Static deployments lead to suboptimal resource utilization and
performance degradation under dynamic workloads. Furthermore, the high cost of
adjusting instances hinders dynamic scaling, limiting the true potential of
efficient LLM serving.
  To address this, we propose CoCoServe, an elastic system that facilitates
dynamic and fine-grained scaling. Its key innovation lies in the module-level
operations for the replication and migration of LLM modules, such as decoder
layers and projections. Through a comprehensive analysis of the trade-offs
associated with these operations, we develop an auto-scaling mechanism that
dynamically regulates module-level resource allocation and performance
optimization, enabling a more cost-effective deployment of LLMs. Our evaluation
demonstrates that the scaling operations employed by CoCoServe exhibit
excellent scalability and can reduce costs by 46% while maintaining
availability. Compared to state-of-the-art LLM serving systems (e.g., Hugging
Face Transformers and vLLM), our approach reduces latency by 14%-75% and
achieves 1.16x-4x throughput on average across different model sizes and
workloads.

### 3. [Cloud Native System for LLM Inference Serving](http://arxiv.org/pdf/2507.18007v1)

Authors: Minxian Xu, Junhan Liao, Jingfeng Wu, Yiyuan He, Kejiang Ye, Chengzhong Xu

Large Language Models (LLMs) are revolutionizing numerous industries, but
their substantial computational demands create challenges for efficient
deployment, particularly in cloud environments. Traditional approaches to
inference serving often struggle with resource inefficiencies, leading to high
operational costs, latency issues, and limited scalability. This article
explores how Cloud Native technologies, such as containerization,
microservices, and dynamic scheduling, can fundamentally improve LLM inference
serving. By leveraging these technologies, we demonstrate how a Cloud Native
system enables more efficient resource allocation, reduces latency, and
enhances throughput in high-demand scenarios. Through real-world evaluations
using Kubernetes-based autoscaling, we show that Cloud Native architectures can
dynamically adapt to workload fluctuations, mitigating performance bottlenecks
while optimizing LLM inference serving performance. This discussion provides a
broader perspective on how Cloud Native frameworks could reshape the future of
scalable LLM inference serving, offering key insights for researchers,
practitioners, and industry leaders in cloud computing and artificial
intelligence.

### 4. [FCPO: Federated Continual Policy Optimization for Real-Time High-Throughput Edge Video Analytics](http://arxiv.org/pdf/2507.18047v1)

Authors: Lucas Liebe, Thanh-Tung Nguyen, Dongman Lee

The growing complexity of Edge Video Analytics (EVA) facilitates new kind of
intelligent applications, but creates challenges in real-time inference serving
systems. State-of-the-art (SOTA) scheduling systems optimize global workload
distributions for heterogeneous devices but often suffer from extended
scheduling cycles, leading to sub-optimal processing in rapidly changing Edge
environments. Local Reinforcement Learning (RL) enables quick adjustments
between cycles but faces scalability, knowledge integration, and adaptability
issues. Thus, we propose FCPO, which combines Continual RL (CRL) with Federated
RL (FRL) to address these challenges. This integration dynamically adjusts
inference batch sizes, input resolutions, and multi-threading during pre- and
post-processing. CRL allows agents to learn from changing Markov Decision
Processes, capturing dynamic environmental variations, while FRL improves
generalization and convergence speed by integrating experiences across
inference models. FCPO combines these via an agent-specific aggregation scheme
and a diversity-aware experience buffer. Experiments on a real-world EVA
testbed showed over 5 times improvement in effective throughput, 60% reduced
latency, and 20% faster convergence with up to 10 times less memory consumption
compared to SOTA RL-based approaches.

### 5. [A large-scale distributed parallel discrete event simulation engines based on Warped2 for Wargaming simulation](http://arxiv.org/pdf/2507.18050v1)

Authors: Xiaoning Jia, Ruilin Kong, Guangya Si, Bilong Shen, Zhe Ji

Rising demand for complex simulations highlights conventional
engines'scalability limits, spurring Parallel Discrete Event Simulation (PDES)
adoption.Warped2, a PDES engine leveraging Time Warp synchronization with
Pending Event Set optimization, delivers strong performance, it struggles with
inherent wargaming limitations: inefficient LP resource allocation during
synchronization and unaddressed complex entity interaction patterns. To address
these challenges, we present an optimized framework featuring four synergistic
improvements: (1) Asynchronous listener threads are introduced to address event
monitoring latency in large-scale scenarios, instead of synchronous polling
mechanisms, (2) METIS-based load rebalancing strategy is incorporated to
address the issue of dynamic event allocation during real-world simulation, (3)
Entity interaction solver with constraint satisfaction mechanisms is designed
to mitigate state conflicts, and (4) Spatial hashing algorithm to overcome
O(n^2) complexity bottlenecks in large-scale nearest-neighbor searches.
Experimental validation through a GridWorld demo demonstrates significant
enhancements in temporal fidelity and computational efficiency. Benchmark
results show our framework achieves 16x acceleration over baseline
implementations and maintains 8x speedup over 1-thread configuration across MPI
and Pthreads implementations.The combined load balancing and LP migration
strategy reduces synchronization overhead by 58.18%, with load balancing
accounting for 57% of the total improvement as the dominant optimization
factor. These improvements provide an enhanced solution for PDES implementation
in large-scale simulation scenarios.

### 6. [Towards Designing an Energy Aware Data Replication Strategy for Cloud Systems Using Reinforcement Learning](http://arxiv.org/pdf/2507.18459v1)

Authors: Amir Najjar, Riad Mokadem, Jean-Marc Pierson

The rapid growth of global data volumes has created a demand for scalable
distributed systems that can maintain a high quality of service. Data
replication is a widely used technique that provides fault tolerance, improved
performance and higher availability. Traditional implementations often rely on
threshold-based activation mechanisms, which can vary depending on workload
changes and system architecture. System administrators typically bear the
responsibility of adjusting these thresholds. To address this challenge,
reinforcement learning can be used to dynamically adapt to workload changes and
different architectures. In this paper, we propose a novel data replication
strategy for cloud systems that employs reinforcement learning to automatically
learn system characteristics and adapt to workload changes. The strategy's aim
is to provide satisfactory Quality of Service while optimizing a trade-off
between provider profit and environmental impact. We present the architecture
behind our solution and describe the reinforcement learning model by defining
the states, actions and rewards.

### 7. [FMI Meets SystemC: A Framework for Cross-Tool Virtual Prototyping](http://arxiv.org/pdf/2507.18339v1)

Authors: Nils Bosbach, Meik Schmidt, Lukas Jünger, Matthias Berthold, Rainer Leupers

As systems become more complex, the demand for thorough testing and virtual
prototyping grows. To simulate whole systems, multiple tools are usually needed
to cover different parts. These parts include the hardware of a system and the
environment with which the system interacts. The Functional Mock-up Interface
(FMI) standard for co-simulation can be used to connect these tools.
  The control part of modern systems is usually a computing unit, such as a
System-on-a-Chip (SoC) or Microcontroller Unit (MCU), which executes software
from a connected memory and interacts with peripherals. To develop software
without requiring access to physical hardware, full-system simulators, the
so-called Virtual Platforms (VPs), are commonly used. The IEEE-standardized
framework for VP development is SystemC TLM. SystemC provides interfaces and
concepts that enable modular design and model exchange. However, SystemC lacks
native FMI support, which limits the integration into broader co-simulation
environments.
  This paper presents a novel framework to control and interact with
SystemC-based VPs using the FMI. We present a case study showing how a
simulated temperature sensor in a SystemC simulation can obtain temperature
values from an external tool via FMI. This approach allows the unmodified
target software to run on the VP and receive realistic environmental input data
such as temperature, velocity, or acceleration values from other tools. Thus,
extensive software testing and verification is enabled. By having tests ready
and the software pre-tested using a VP once the physical hardware is available,
certifications like ISO 26262 can be done earlier.

### Digital Libraries

### 1. [Integrating an ISO 30401-compliant Knowledge Management System with the processes of an Integrated Management System](http://arxiv.org/pdf/2507.18201v1)

Authors: Patrick Prieur, Aline Belloni

With the evolution of process approaches within organizations, the increasing
importance of quality management systems (like ISO 9001) and the recent
introduction of ISO 30401 for knowledge management, we examine how these
different elements converge within the framework of an Integrated Management
System. The article specifically demonstrates how an ISO30401-compliant
knowledge management system can be implemented by deploying the mechanisms of
the SECI model through the steps of the PDCA cycle as applied in the processes
of the integrated management system.

### 2. [SMECS: A Software Metadata Extraction and Curation Software](http://arxiv.org/pdf/2507.18159v1)

Authors: Stephan Ferenz, Aida Jafarbigloo, Oliver Werth, Astrid Nieße

Metadata play a crucial role in adopting the FAIR principles for research
software and enables findability and reusability. However, creating
high-quality metadata can be resource-intensive for researchers and research
software engineers. To address this challenge, we developed the Software
Metadata Extraction and Curation Software (SMECS) which integrates the
extraction of metadata from existing sources together with a user-friendly
interface for metadata curation. SMECS extracts metadata from online
repositories such as GitHub and presents it to researchers through an
interactive interface for further curation and export as a CodeMeta file. The
usability of SMECS was evaluated through usability experiments which confirmed
that SMECS provides a satisfactory user experience. SMECS supports the
FAIRification of research software by simplifying metadata creation.

### 3. [Integrating an ISO30401-compliant Knowledge management system with existing business processes of an organization](http://arxiv.org/pdf/2507.18197v1)

Authors: Aline Belloni, Patrick Prieur

Business process modeling is used by most organizations as an essential
framework for ensuring efficiency and effectiveness of the work and workflow
performed by its employees and for ensuring the alignment of such work with its
strategic goals. For organizations that are compliant or near-compliant with
ISO 9001, this approach involves the detailed mapping of processes,
sub-processes, activities, and tasks. ISO30401 is a Management System Standard,
introduced in 2018, establishing universal requirements for the set up of a
Knowledge Management System in an organization. As ``ISO30401 implementers'' we
regularly face the challenge of explaining our clients how the knowledge
development, transformation and conveyances activities depicted in ISO30401 do
integrate with existing operational processes. This article recaps process
modelling principles in the context of ISO9001 and explores, based on our
experience, how an ISO30401-compliant Knowledge Management System (KMS)
entwines with all other processes of an Integrated Management System and in
particular how it can be implemented by deploying the mechanisms of the SECI
model through the steps of PDCA cycles.

### 4. [Factual Inconsistencies in Multilingual Wikipedia Tables](http://arxiv.org/pdf/2507.18406v1)

Authors: Silvia Cappa, Lingxiao Kong, Pille-Riin Peet, Fanfu Wei, Yuchen Zhou, Jan-Christoph Kalo

Wikipedia serves as a globally accessible knowledge source with content in
over 300 languages. Despite covering the same topics, the different versions of
Wikipedia are written and updated independently. This leads to factual
inconsistencies that can impact the neutrality and reliability of the
encyclopedia and AI systems, which often rely on Wikipedia as a main training
source. This study investigates cross-lingual inconsistencies in Wikipedia's
structured content, with a focus on tabular data. We developed a methodology to
collect, align, and analyze tables from Wikipedia multilingual articles,
defining categories of inconsistency. We apply various quantitative and
qualitative metrics to assess multilingual alignment using a sample dataset.
These insights have implications for factual verification, multilingual
knowledge interaction, and design for reliable AI systems leveraging Wikipedia
content.

### Data Structures and Algorithms

### 1. [Dual Charging for Half-Integral TSP](http://arxiv.org/pdf/2507.17999v1)

Authors: Nathan Klein, Mehrshad Taziki

We show that the max entropy algorithm is a randomized 1.49776 approximation
for half-integral TSP, improving upon the previous known bound of 1.49993 from
Karlin et al. This also improves upon the best-known approximation for
half-integral TSP due to Gupta et al. Our improvement results from using the
dual, instead of the primal, to analyze the expected cost of the matching. We
believe this method of analysis could lead to a simpler proof that max entropy
is a better-than-3/2 approximation in the general case.
  We also give a 1.4671 approximation for half integral LP solutions with no
proper minimum cuts and an even number of vertices, improving upon the bound of
Haddadan and Newman of 1.476. We then extend the analysis to the case when
there are an odd number of vertices $n$ at the cost of an additional $O(1/n)$
factor.

### 2. [On recognizing graphs representing Persistent Perfect Phylogenies](http://arxiv.org/pdf/2507.18281v1)

Authors: Paola Bonizzoni, Gianluca Della Vedova, Mauricio Soto Gomez, Gabriella Trucco

The Persistent Perfect phylogeny, also known as Dollo-1, has been introduced
as a generalization of the well-known perfect phylogenetic model for binary
characters to deal with the potential loss of characters. The problem of
deciding the existence of a Persistent Perfect phylogeny can be reduced to the
one of recognizing a class of bipartite graphs whose nodes are species and
characters. Thus an interesting question is solving directly the problem of
recognizing such graphs. We present a polynomial-time algorithm for deciding
Persistent Perfect phylogeny existence in maximal graphs, where no character's
species set is contained within another character's species set. Our solution,
that relies only on graph properties, narrows the gap between the linear-time
simple algorithm for Perfect Phylogeny and the NP-hardness results for the
Dollo-$k$ phylogeny with $k>1$.

### 3. [Zeroth-order log-concave sampling](http://arxiv.org/pdf/2507.18021v1)

Authors: Yunbum Kook

We study the zeroth-order query complexity of log-concave sampling,
specifically uniform sampling from convex bodies using membership oracles. We
propose a simple variant of the proximal sampler that achieves the query
complexity with matched R\'enyi orders between the initial warmness and output
guarantee. Specifically, for any $\varepsilon>0$ and $q\geq2$, the sampler,
initialized at $\pi_{0}$, outputs a sample whose law is $\varepsilon$-close in
$q$-R\'enyi divergence to $\pi$, the uniform distribution over a convex body in
$\mathbb{R}^{d}$, using
$\widetilde{O}(qM_{q}^{q/(q-1)}d^{2}\,\lVert\operatorname{cov}\pi\rVert\log\frac{1}{\varepsilon})$
membership queries, where
$M_{q}=\lVert\text{d}\pi_{0}/\text{d}\pi\rVert_{L^{q}(\pi)}$.
  We further introduce a simple annealing scheme that produces a warm start in
$q$-R\'enyi divergence (i.e., $M_{q}=O(1)$) using
$\widetilde{O}(qd^{2}R^{3/2}\,\lVert\operatorname{cov}\pi\rVert^{1/4})$
queries, where $R^{2}=\mathbb{E}_{\pi}[|\cdot|^{2}]$. This interpolates between
known complexities for warm-start generation in total variation and
R\'enyi-infinity divergence. To relay a R\'enyi warmness across the annealing
scheme, we establish hypercontractivity under simultaneous heat flow and
translate it into an improved mixing guarantee for the proximal sampler under a
logarithmic Sobolev inequality. These results extend naturally to general
log-concave distributions accessible via evaluation oracles, incurring
additional quadratic queries.

### Emerging Technologies

### 1. [Effects of variation in system responsiveness on user performance in virtual environments](http://arxiv.org/pdf/2507.18085v1)

Authors: Benjamin Watson, Neff Walker, William Ribarsky, Victoria Spaulding

System responsiveness (SR) is defined as the elapsed time until a system
responds to user control. SR fluctuates over time, so it must be described
statistically with mean (MSR) and standard deviation (SDSR). In this paper, we
examine SR in virtual environments (VEs), outlining its components and methods
of experimental measurement and manipulation. Three studies of MSR and SDSR
effects on performance of grasp and placement tasks are then presented. The
studies used within-subjects designs with 11, 12, and 10 participants,
respectively. Results showed that SDSR affected performance only if it was
above 82 ms. Placement required more frequent visual feedback and was more
sensitive to SR. We infer that VE designers need not tightly control SDSR and
may wish to vary SR control based on required visual feedback frequency. These
results may be used to improve the human-computer interface in a wide range of
interactive graphical applications, including scientific visualization,
training, mental health, and entertainment.

### 2. [Low-power switching of memristors exhibiting fractional-order dynamics](http://arxiv.org/pdf/2507.18487v1)

Authors: Nathan Astin, Yuriy V. Pershin

In this conference contribution, we present some initial results on switching
memristive devices exhibiting fractional-order behavior using current pulses.
In our model, it is assumed that the evolution of a state variable follows a
fractional-order differential equation involving a Caputo-type derivative. A
study of Joule losses demonstrates that the best switching strategy minimizing
these losses depends on the fractional derivative's order and the power
exponent in the equation of motion. It is found that when the order of the
fractional derivative exceeds half of the power exponent, the best approach is
to employ a wide pulse. Conversely, when this condition is not met, Joule
losses are minimized by applying a zero current followed by a narrow current
pulse of the highest allowable amplitude. These findings are explored further
in the context of multi-pulse control. Our research lays the foundation for the
advancement of the next generation of energy-efficient neuromorphic computing
architectures that more closely mimic their biological counterparts.

### 3. [PRACtical: Subarray-Level Counter Update and Bank-Level Recovery Isolation for Efficient PRAC Rowhammer Mitigation](http://arxiv.org/pdf/2507.18581v1)

Authors: Ravan Nazaraliyev, Saber Ganjisaffar, Nurlan Nazaraliyev, Nael Abu-Ghazaleh

As DRAM density increases, Rowhammer becomes more severe due to heightened
charge leakage, reducing the number of activations needed to induce bit flips.
The DDR5 standard addresses this threat with in-DRAM per-row activation
counters (PRAC) and the Alert Back-Off (ABO) signal to trigger mitigation.
However, PRAC adds performance overhead by incrementing counters during the
precharge phase, and recovery refreshes stalls the entire memory channel, even
if only one bank is under attack.
  We propose PRACtical, a performance-optimized approach to PRAC+ABO that
maintains the same security guarantees. First, we reduce counter update latency
by introducing a centralized increment circuit, enabling overlap between
counter updates and subsequent row activations in other subarrays. Second, we
enhance the $RFM_{ab}$ mitigation by enabling bank-level granularity: instead
of stalling the entire channel, only affected banks are paused. This is
achieved through a DRAM-resident register that identifies attacked banks.
  PRACtical improves performance by 8% on average (up to 20%) over the
state-of-the-art, reduces energy by 19%, and limits performance degradation
from aggressive performance attacks to less than 6%, all while preserving
Rowhammer protection.

### 4. [Agentic AI framework for End-to-End Medical Data Inference](http://arxiv.org/pdf/2507.18115v1)

Authors: Soorya Ram Shimgekar, Shayan Vassef, Abhay Goyal, Navin Kumar, Koustuv Saha

Building and deploying machine learning solutions in healthcare remains
expensive and labor-intensive due to fragmented preprocessing workflows, model
compatibility issues, and stringent data privacy constraints. In this work, we
introduce an Agentic AI framework that automates the entire clinical data
pipeline, from ingestion to inference, through a system of modular,
task-specific agents. These agents handle both structured and unstructured
data, enabling automatic feature selection, model selection, and preprocessing
recommendation without manual intervention. We evaluate the system on publicly
available datasets from geriatrics, palliative care, and colonoscopy imaging.
For example, in the case of structured data (anxiety data) and unstructured
data (colonoscopy polyps data), the pipeline begins with file-type detection by
the Ingestion Identifier Agent, followed by the Data Anonymizer Agent ensuring
privacy compliance, where we first identify the data type and then anonymize
it. The Feature Extraction Agent identifies features using an embedding-based
approach for tabular data, extracting all column names, and a multi-stage
MedGemma-based approach for image data, which infers modality and disease name.
These features guide the Model-Data Feature Matcher Agent in selecting the
best-fit model from a curated repository. The Preprocessing Recommender Agent
and Preprocessing Implementor Agent then apply tailored preprocessing based on
data type and model requirements. Finally, the ``Model Inference Agent" runs
the selected model on the uploaded data and generates interpretable outputs
using tools like SHAP, LIME, and DETR attention maps. By automating these
high-friction stages of the ML lifecycle, the proposed framework reduces the
need for repeated expert intervention, offering a scalable, cost-efficient
pathway for operationalizing AI in clinical environments.

### Formal Languages and Automata Theory

### 1. [Time for Quiescence: Modelling quiescent behaviour in testing via time-outs in timed automata](http://arxiv.org/pdf/2507.18205v1)

Authors: Laura Brandán Briones, Marcus Gerhold, Petra van den Bos, Mariëlle Stoelinga

Model-based testing (MBT) derives test suites from a behavioural
specification of the system under test. In practice, engineers favour simple
models, such as labelled transition systems (LTSs). However, to deal with
quiescence - the absence of observable output - in practice, a time-out needs
to be set to conclude observation of quiescence. Timed MBT exists, but it
typically relies on the full arsenal of timed automata (TA).
  We present a lifting operator $\chi^{\scriptstyle M}\!$ that adds timing
without the TA overhead: given an LTS, $\chi^{\scriptstyle M}\!$ introduces a
single clock for a user chosen time bound $M>0$ to declare quiescence. In the
timed automaton, the clock is used to model that outputs should happen before
the clock reaches value $M$, while quiescence occurs exactly at time $M$. This
way we provide a formal basis for the industrial practice of choosing a
time-out to conclude quiescence. Our contributions are threefold: (1) an
implementation conforms under $\mathbf{ioco}$ if and only if its lifted version
conforms under timed $\mathbf{tioco_M}$ (2) applying $\chi^{\scriptstyle M}\!$
before or after the standard $\mathbf{ioco}$ test-generation algorithm yields
the same set of tests, and (3) the lifted TA test suite and the original LTS
test suite deliver identical verdicts for every implementation.

### Graphics

### 1. [DanceGraph: A Complementary Architecture for Synchronous Dancing Online](http://arxiv.org/pdf/2507.18052v1)

Authors: David Sinclair, Ademyemi Ademola, Babis Koniaris, Kenny Mitchell

DanceGraph is an architecture for synchronized online dancing overcoming the
latency of networked body pose sharing. We break down this challenge by
developing a real-time bandwidth-efficient architecture to minimize lag and
reduce the timeframe of required motion prediction for synchronization with the
music's rhythm. In addition, we show an interactive method for the
parameterized stylization of dance motions for rhythmic dance using online
dance correctives.

### 2. [PS-GS: Gaussian Splatting for Multi-View Photometric Stereo](http://arxiv.org/pdf/2507.18231v1)

Authors: Yixiao Chen, Bin Liang, Hanzhi Guo, Yongqing Cheng, Jiayi Zhao, Dongdong Weng

Integrating inverse rendering with multi-view photometric stereo (MVPS)
yields more accurate 3D reconstructions than the inverse rendering approaches
that rely on fixed environment illumination. However, efficient inverse
rendering with MVPS remains challenging. To fill this gap, we introduce the
Gaussian Splatting for Multi-view Photometric Stereo (PS-GS), which efficiently
and jointly estimates the geometry, materials, and lighting of the object that
is illuminated by diverse directional lights (multi-light). Our method first
reconstructs a standard 2D Gaussian splatting model as the initial geometry.
Based on the initialization model, it then proceeds with the deferred inverse
rendering by the full rendering equation containing a lighting-computing
multi-layer perceptron. During the whole optimization, we regularize the
rendered normal maps by the uncalibrated photometric stereo estimated normals.
We also propose the 2D Gaussian ray-tracing for single directional light to
refine the incident lighting. The regularizations and the use of multi-view and
multi-light images mitigate the ill-posed problem of inverse rendering. After
optimization, the reconstructed object can be used for novel-view synthesis,
relighting, and material and shape editing. Experiments on both synthetic and
real datasets demonstrate that our method outperforms prior works in terms of
reconstruction accuracy and computational efficiency.

### 3. [GeoAvatar: Adaptive Geometrical Gaussian Splatting for 3D Head Avatar](http://arxiv.org/pdf/2507.18155v1)

Authors: SeungJun Moon, Hah Min Lew, Seungeun Lee, Ji-Su Kang, Gyeong-Moon Park

Despite recent progress in 3D head avatar generation, balancing identity
preservation, i.e., reconstruction, with novel poses and expressions, i.e.,
animation, remains a challenge. Existing methods struggle to adapt Gaussians to
varying geometrical deviations across facial regions, resulting in suboptimal
quality. To address this, we propose GeoAvatar, a framework for adaptive
geometrical Gaussian Splatting. GeoAvatar leverages Adaptive Pre-allocation
Stage (APS), an unsupervised method that segments Gaussians into rigid and
flexible sets for adaptive offset regularization. Then, based on mouth anatomy
and dynamics, we introduce a novel mouth structure and the part-wise
deformation strategy to enhance the animation fidelity of the mouth. Finally,
we propose a regularization loss for precise rigging between Gaussians and 3DMM
faces. Moreover, we release DynamicFace, a video dataset with highly expressive
facial motions. Extensive experiments show the superiority of GeoAvatar
compared to state-of-the-art methods in reconstruction and novel animation
scenarios.

### 4. [Tiny is not small enough: High-quality, low-resource facial animation models through hybrid knowledge distillation](http://arxiv.org/pdf/2507.18352v1)

Authors: Zhen Han, Mattias Teye, Derek Yadgaroff, Judith Bütepage

The training of high-quality, robust machine learning models for
speech-driven 3D facial animation requires a large, diverse dataset of
high-quality audio-animation pairs. To overcome the lack of such a dataset,
recent work has introduced large pre-trained speech encoders that are robust to
variations in the input audio and, therefore, enable the facial animation model
to generalize across speakers, audio quality, and languages. However, the
resulting facial animation models are prohibitively large and lend themselves
only to offline inference on a dedicated machine. In this work, we explore
on-device, real-time facial animation models in the context of game
development. We overcome the lack of large datasets by using hybrid knowledge
distillation with pseudo-labeling. Given a large audio dataset, we employ a
high-performing teacher model to train very small student models. In contrast
to the pre-trained speech encoders, our student models only consist of
convolutional and fully-connected layers, removing the need for attention
context or recurrent updates. In our experiments, we demonstrate that we can
reduce the memory footprint to up to 3.4 MB and required future audio context
to up to 81 ms while maintaining high-quality animations. This paves the way
for on-device inference, an important step towards realistic, model-driven
digital characters.

### Computer Science and Game Theory

### 1. [On Pareto-Optimal and Fair Allocations with Personalized Bi-Valued Utilities](http://arxiv.org/pdf/2507.18251v1)

Authors: Jiarong Jin, Biaoshuai Tao

We study the fair division problem of allocating $m$ indivisible goods to $n$
agents with additive personalized bi-valued utilities. Specifically, each agent
$i$ assigns one of two positive values $a_i > b_i > 0$ to each good, indicating
that agent $i$'s valuation of any good is either $a_i$ or $b_i$. For
convenience, we denote the value ratio of agent $i$ as $r_i = a_i / b_i$.
  We give a characterization to all the Pareto-optimal allocations. Our
characterization implies a polynomial-time algorithm to decide if a given
allocation is Pareto-optimal in the case each $r_i$ is an integer. For the
general case (where $r_i$ may be fractional), we show that this decision
problem is coNP-complete. Our result complements the existing results: this
decision problem is coNP-complete for tri-valued utilities (where each agent's
value for each good belongs to $\{a,b,c\}$ for some prescribed $a>b>c\geq0$),
and this decision problem belongs to P for bi-valued utilities (where $r_i$ in
our model is the same for each agent).
  We further show that an EFX allocation always exists and can be computed in
polynomial time under the personalized bi-valued utilities setting, which
extends the previous result on bi-valued utilities. We propose the open problem
of whether an EFX and Pareto-optimal allocation always exists (and can be
computed in polynomial time).

### Human-Computer Interaction

### 1. [Evaluating judgment of spatial correlation in visual displays of scalar field distributions](http://arxiv.org/pdf/2507.17997v1)

Authors: Yayan Zhao, Matthew Berger

In this work we study the identification of spatial correlation in
distributions of 2D scalar fields, presented across different forms of visual
displays. We study simple visual displays that directly show color-mapped
scalar fields, namely those drawn from a distribution, and whether humans can
identify strongly correlated spatial regions in these displays. In this
setting, the recognition of correlation requires making judgments on a set of
fields, rather than just one field. Thus, in our experimental design we compare
two basic visualization designs: animation-based displays against juxtaposed
views of scalar fields, along different choices of color scales. Moreover, we
investigate the impacts of the distribution itself, controlling for the level
of spatial correlation and discriminability in spatial scales. Our study's
results illustrate the impacts of these distribution characteristics, while
also highlighting how different visual displays impact the types of judgments
made in assessing spatial correlation. Supplemental material is available at
https://osf.io/zn4qy

### 2. ["I Would Not Be This Version of Myself Today": Elaborating on the Effects of Eudaimonic Gaming Experiences](http://arxiv.org/pdf/2507.18084v1)

Authors: Nisha Devasia, Georgia Kenderova, Michele Newman, Julie Kientz, Jin Ha Lee

While much of the research in digital games has emphasized hedonic
experiences, such as flow, enjoyment, and positive affect, recent years have
seen increased interest in eudaimonic gaming experiences, typically
mixed-affect and associated with personal meaningfulness and growth. The
formation of such experiences in games is theorized to have four constituent
elements: motivation, game use, experience, and effects. However, while the
first three elements have been relatively well explored in the literature, the
effects - and how they may influence positive individual outcomes - have been
underexplored thus far. To this end, in this work, we investigate the perceived
outcomes of eudaimonic gaming and how different components of the experience
influence these effects. We conducted a survey (n = 166) in which respondents
recounted meaningful gaming experiences and how they affected their present
lives. We used a mixed-methods approach to classify effects and identify
significant subcomponents of their formation. We contribute an empirical
understanding of how meaningful gaming experiences can lead to positive
reflective, learning, social, health, and career effects, extending current
theoretical models of eudaimonic gaming experiences and offering implications
for how researchers and practitioners might use these findings to promote
positive outcomes for players.

### 3. [Understood: Real-Time Communication Support for Adults with ADHD Using Mixed Reality](http://arxiv.org/pdf/2507.18151v1)

Authors: Shizhen Zhang, Shengxin Li, Quan Li

Adults with Attention Deficit Hyperactivity Disorder (ADHD) often experience
communication challenges, primarily due to executive dysfunction and emotional
dysregulation, even after years of social integration. While existing
interventions predominantly target children through structured or intrusive
methods, adults lack tools that translate clinical strategies into daily
communication support. To address this gap, we present Understood, a Mixed
Reality (MR) system implemented on Microsoft HoloLens 2, designed to assist
adults with ADHD in real-world communication. Through formative semi-structured
interviews and a design workshop, we identified critical communication barriers
and derived design goals for the system. Understood combines three key
features: (1) real-time conversation summarization to reduce cognitive load,
(2) context-aware subsequent word suggestions during moments of disfluency, and
(3) topic shifting detection and reminding to mitigate off-topic transitions. A
within-subjects user study and expert interviews demonstrate that Understood
effectively supports communication with high usability, offering a complement
to therapist-mediated interventions.

### 4. [ProactiveVA: Proactive Visual Analytics with LLM-Based UI Agent](http://arxiv.org/pdf/2507.18165v1)

Authors: Yuheng Zhao, Xueli Shu, Liwen Fan, Lin Gao, Yu Zhang, Siming Chen

Visual analytics (VA) is typically applied to complex data, thus requiring
complex tools. While visual analytics empowers analysts in data analysis,
analysts may get lost in the complexity occasionally. This highlights the need
for intelligent assistance mechanisms. However, even the latest LLM-assisted VA
systems only provide help when explicitly requested by the user, making them
insufficiently intelligent to offer suggestions when analysts need them the
most. We propose a ProactiveVA framework in which LLM-powered UI agent monitors
user interactions and delivers context-aware assistance proactively. To design
effective proactive assistance, we first conducted a formative study analyzing
help-seeking behaviors in user interaction logs, identifying when users need
proactive help, what assistance they require, and how the agent should
intervene. Based on this analysis, we distilled key design requirements in
terms of intent recognition, solution generation, interpretability and
controllability. Guided by these requirements, we develop a three-stage UI
agent pipeline including perception, reasoning, and acting. The agent
autonomously perceives users' needs from VA interaction logs, providing
tailored suggestions and intuitive guidance through interactive exploration of
the system. We implemented the framework in two representative types of VA
systems, demonstrating its generalizability, and evaluated the effectiveness
through an algorithm evaluation, case and expert study and a user study. We
also discuss current design trade-offs of proactive VA and areas for further
exploration.

### 5. [Talking to...uh...um...Machines: The Impact of Disfluent Speech Agents on Partner Models and Perspective Taking](http://arxiv.org/pdf/2507.18315v1)

Authors: Rhys Jacka, Paola R. Peña, Sophie Leonard, Éva Székely, Benjamin R. Cowan

Speech disfluencies play a role in perspective-taking and audience design in
human-human communication (HHC), but little is known about their impact in
human-machine dialogue (HMD). In an online Namer-Matcher task, sixty-one
participants interacted with a speech agent using either fluent or disfluent
speech. Participants completed a partner-modelling questionnaire (PMQ) both
before and after the task. Post-interaction evaluations indicated that
participants perceived the disfluent agent as more competent, despite no
significant differences in pre-task ratings. However, no notable differences
were observed in assessments of conversational flexibility or human-likeness.
Our findings also reveal evidence of egocentric and allocentric language
production when participants interact with speech agents. Interaction with
disfluent speech agents appears to increase egocentric communication in
comparison to fluent agents. Although the wide credibility intervals mean this
effect is not clear-cut. We discuss potential interpretations of this finding,
focusing on how disfluencies may impact partner models and language production
in HMD.

### 6. [Towards Understanding Decision Problems As a Goal of Visualization Design](http://arxiv.org/pdf/2507.18428v1)

Authors: Lena Cibulski, Stefan Bruckner

Decision-making is a central yet under-defined goal in visualization
research. While existing task models address decision processes, they often
neglect the conditions framing a decision. To better support decision-making
tasks, we propose a characterization scheme that describes decision problems
through key properties of the data, users, and task context. This scheme helps
visualization researchers specify decision-support claims more precisely and
informs the design of appropriate visual encodings and interactions. We
demonstrate the utility of our approach by applying it to characterize decision
tasks targeted by existing design studies, highlighting opportunities for
future research in decision-centric visualization.

### 7. [ForcePinch: Force-Responsive Spatial Interaction for Tracking Speed Control in XR](http://arxiv.org/pdf/2507.18510v1)

Authors: Chenyang Zhang, Tiffany S Ma, John Andrews, Eric J Gonzalez, Mar Gonzalez-Franco, Yalong Yang

Spatial interaction in 3D environments requires balancing efficiency and
precision, which requires dynamic tracking speed adjustments. However, existing
techniques often couple tracking speed adjustments directly with hand
movements, reducing interaction flexibility. Inspired by the natural friction
control inherent in the physical world, we introduce ForcePinch, a novel
force-responsive spatial interaction method that enables users to intuitively
modulate pointer tracking speed and smoothly transition between rapid and
precise movements by varying their pinching force. To implement this concept,
we developed a hardware prototype integrating a pressure sensor with a
customizable mapping function that translates pinching force into tracking
speed adjustments. We conducted a user study with 20 participants performing
well-established 1D, 2D, and 3D object manipulation tasks, comparing ForcePinch
against the distance-responsive technique Go-Go and speed-responsive technique
PRISM. Results highlight distinctive characteristics of the force-responsive
approach across different interaction contexts. Drawing on these findings, we
highlight the contextual meaning and versatility of force-responsive
interactions through four illustrative examples, aiming to inform and inspire
future spatial interaction design.

### 8. [MeloKids: Multisensory VR System to Enhance Speech and Motor Coordination in Children with Hearing Loss](http://arxiv.org/pdf/2507.18619v1)

Authors: Yichen Yu, Qiaoran Wang

Children with hearing impairments face ongoing challenges in language and
motor development. This study explores how multi-sensory feedback technology
based on virtual reality (VR), integrating auditory, visual, and tactile
stimuli, can enhance rehabilitation outcomes. Using functional near-infrared
spectroscopy (fNIRS) technology, we assessed cortical activation patterns in
children during pitch-matching tasks across different interaction modes. Our
findings aim to provide evidence for designing personalized, interactive
rehabilitation systems that enhance cognitive engagement and motor control in
children with hearing impairments.

### 9. [Evaluation of a Provenance Management Tool for Immersive Virtual Fieldwork](http://arxiv.org/pdf/2507.18622v1)

Authors: Armin Bernstetter, Tom Kwasnitschka, Isabella Peters

Ensuring reproducibility of research is an integral part of good scientific
practice. One way to support this is through provenance: information about
research workflows from data gathering to researchers' sensemaking processes
leading to published results. This is highly important in disciplines such as
geosciences, where researchers use software for interactive and immersive
visualizations of geospatial data, doing virtual measurements in simulated
fieldwork on 3D models. We evaluated a provenance management tool, which allows
recording of interactions with a virtual fieldwork tool and annotating
different states of the visualization. The user study investigated how
researchers used this Digital Lab Book (DLB) and whether perceived ease of use
and perceived usefulness differed between groups in immersive or non-immersive
settings. Participants perceived the DLB as both useful and easy to use. While
there were indications of differences in perceived ease of use (higher for
immersive setting), usage patterns showed no significant group differences.

### 10. [Effects of variation in system responsiveness on user performance in virtual environments](http://arxiv.org/pdf/2507.18085v1)

Authors: Benjamin Watson, Neff Walker, William Ribarsky, Victoria Spaulding

System responsiveness (SR) is defined as the elapsed time until a system
responds to user control. SR fluctuates over time, so it must be described
statistically with mean (MSR) and standard deviation (SDSR). In this paper, we
examine SR in virtual environments (VEs), outlining its components and methods
of experimental measurement and manipulation. Three studies of MSR and SDSR
effects on performance of grasp and placement tasks are then presented. The
studies used within-subjects designs with 11, 12, and 10 participants,
respectively. Results showed that SDSR affected performance only if it was
above 82 ms. Placement required more frequent visual feedback and was more
sensitive to SR. We infer that VE designers need not tightly control SDSR and
may wish to vary SR control based on required visual feedback frequency. These
results may be used to improve the human-computer interface in a wide range of
interactive graphical applications, including scientific visualization,
training, mental health, and entertainment.

### Information Retrieval

### 1. [RecPS: Privacy Risk Scoring for Recommender Systems](http://arxiv.org/pdf/2507.18365v1)

Authors: Jiajie He, Yuechun Gu, Keke Chen

Recommender systems (RecSys) have become an essential component of many web
applications. The core of the system is a recommendation model trained on
highly sensitive user-item interaction data. While privacy-enhancing techniques
are actively studied in the research community, the real-world model
development still depends on minimal privacy protection, e.g., via controlled
access. Users of such systems should have the right to choose \emph{not} to
share highly sensitive interactions. However, there is no method allowing the
user to know which interactions are more sensitive than others. Thus,
quantifying the privacy risk of RecSys training data is a critical step to
enabling privacy-aware RecSys model development and deployment. We propose a
membership-inference attack (MIA)- based privacy scoring method, RecPS, to
measure privacy risks at both the interaction and user levels. The RecPS
interaction-level score definition is motivated and derived from differential
privacy, which is then extended to the user-level scoring method. A critical
component is the interaction-level MIA method RecLiRA, which gives high-quality
membership estimation. We have conducted extensive experiments on well-known
benchmark datasets and RecSys models to show the unique features and benefits
of RecPS scoring in risk assessment and RecSys model unlearning. Our code is
available at https://anonymous.4open.science/r/RsLiRA-4BD3/readme.md.

### 2. [How Well Do LLMs Predict Prerequisite Skills? Zero-Shot Comparison to Expert-Defined Concepts](http://arxiv.org/pdf/2507.18479v1)

Authors: Ngoc Luyen Le, Marie-Hélène Abel

Prerequisite skills - foundational competencies required before mastering
more advanced concepts - are important for supporting effective learning,
assessment, and skill-gap analysis. Traditionally curated by domain experts,
these relationships are costly to maintain and difficult to scale. This paper
investigates whether large language models (LLMs) can predict prerequisite
skills in a zero-shot setting, using only natural language descriptions and
without task-specific fine-tuning. We introduce ESCO-PrereqSkill, a benchmark
dataset constructed from the ESCO taxonomy, comprising 3,196 skills and their
expert-defined prerequisite links. Using a standardized prompting strategy, we
evaluate 13 state-of-the-art LLMs, including GPT-4, Claude 3, Gemini, LLaMA 4,
Qwen2, and DeepSeek, across semantic similarity, BERTScore, and inference
latency. Our results show that models such as LLaMA4-Maverick,
Claude-3-7-Sonnet, and Qwen2-72B generate predictions that closely align with
expert ground truth, demonstrating strong semantic reasoning without
supervision. These findings highlight the potential of LLMs to support scalable
prerequisite skill modeling for applications in personalized learning,
intelligent tutoring, and skill-based recommender systems.

### 3. [The Best is Yet to Come: Graph Convolution in the Testing Phase for Multimodal Recommendation](http://arxiv.org/pdf/2507.18489v1)

Authors: Jinfeng Xu, Zheyu Chen, Shuo Yang, Jinze Li, Edith C. H. Ngai

The efficiency and scalability of graph convolution networks (GCNs) in
training recommender systems remain critical challenges, hindering their
practical deployment in real-world scenarios. In the multimodal recommendation
(MMRec) field, training GCNs requires more expensive time and space costs and
exacerbates the gap between different modalities, resulting in sub-optimal
recommendation accuracy. This paper critically points out the inherent
challenges associated with adopting GCNs during the training phase in MMRec,
revealing that GCNs inevitably create unhelpful and even harmful pairs during
model optimization and isolate different modalities. To this end, we propose
FastMMRec, a highly efficient multimodal recommendation framework that deploys
graph convolutions exclusively during the testing phase, bypassing their use in
training. We demonstrate that adopting GCNs solely in the testing phase
significantly improves the model's efficiency and scalability while alleviating
the modality isolation problem often caused by using GCNs during the training
phase. We conduct extensive experiments on three public datasets, consistently
demonstrating the performance superiority of FastMMRec over competitive
baselines while achieving efficiency and scalability.

### 4. [Transform Before You Query: A Privacy-Preserving Approach for Vector Retrieval with Embedding Space Alignment](http://arxiv.org/pdf/2507.18518v1)

Authors: Ruiqi He, Zekun Fei, Jiaqi Li, Xinyuan Zhu, Biao Yi, Siyi Lv, Weijie Liu, Zheli Liu

Vector Database (VDB) can efficiently index and search high-dimensional
vector embeddings from unstructured data, crucially enabling fast semantic
similarity search essential for modern AI applications like generative AI and
recommendation systems. Since current VDB service providers predominantly use
proprietary black-box models, users are forced to expose raw query text to them
via API in exchange for the vector retrieval services. Consequently, if query
text involves confidential records from finance or healthcare domains, this
mechanism inevitably leads to critical leakage of user's sensitive information.
To address this issue, we introduce STEER (\textbf{S}ecure \textbf{T}ransformed
\textbf{E}mbedding v\textbf{E}ctor\textbf{ R}etrieval), a private vector
retrieval framework that leverages the alignment relationship between the
semantic spaces of different embedding models to derive approximate embeddings
for the query text. STEER performs the retrieval using the approximate
embeddings within the original VDB and requires no modifications to the server
side. Our theoretical and experimental analyses demonstrate that STEER
effectively safeguards query text privacy while maintaining the retrieval
accuracy. Even though approximate embeddings are approximations of the
embeddings from proprietary models, they still prevent the providers from
recovering the query text through Embedding Inversion Attacks (EIAs). Extensive
experimental results show that Recall@100 of STEER can basically achieve a
decrease of less than 5\%. Furthermore, even when searching within a text
corpus of millions of entries, STEER achieves a Recall@20 accuracy 20\% higher
than current baselines.

### 5. [Fashion-AlterEval: A Dataset for Improved Evaluation of Conversational Recommendation Systems with Alternative Relevant Items](http://arxiv.org/pdf/2507.18017v1)

Authors: Maria Vlachou

In Conversational Recommendation Systems (CRS), a user provides feedback on
recommended items at each turn, leading the CRS towards improved
recommendations. Due to the need for a large amount of data, a user simulator
is employed for both training and evaluation. Such user simulators critique the
current retrieved item based on knowledge of a single target item. However,
system evaluation in offline settings with simulators is limited by the focus
on a single target item and their unlimited patience over a large number of
turns. To overcome these limitations of existing simulators, we propose
Fashion-AlterEval, a new dataset that contains human judgments for a selection
of alternative items by adding new annotations in common fashion CRS datasets.
Consequently, we propose two novel meta-user simulators that use the collected
judgments and allow simulated users not only to express their preferences about
alternative items to their original target, but also to change their mind and
level of patience. In our experiments using the Shoes and Fashion IQ as the
original datasets and three CRS models, we find that using the knowledge of
alternatives by the simulator can have a considerable impact on the evaluation
of existing CRS models, specifically that the existing single-target evaluation
underestimates their effectiveness, and when simulatedusers are allowed to
instead consider alternative relevant items, the system can rapidly respond to
more quickly satisfy the user.

### 6. [LLM-based Embedders for Prior Case Retrieval](http://arxiv.org/pdf/2507.18455v1)

Authors: Damith Premasiri, Tharindu Ranasinghe, Ruslan Mitkov

In common law systems, legal professionals such as lawyers and judges rely on
precedents to build their arguments. As the volume of cases has grown massively
over time, effectively retrieving prior cases has become essential. Prior case
retrieval (PCR) is an information retrieval (IR) task that aims to
automatically identify the most relevant court cases for a specific query from
a large pool of potential candidates. While IR methods have seen several
paradigm shifts over the last few years, the vast majority of PCR methods
continue to rely on traditional IR methods, such as BM25. The state-of-the-art
deep learning IR methods have not been successful in PCR due to two key
challenges: i. Lengthy legal text limitation; when using the powerful
BERT-based transformer models, there is a limit of input text lengths, which
inevitably requires to shorten the input via truncation or division with a loss
of legal context information. ii. Lack of legal training data; due to data
privacy concerns, available PCR datasets are often limited in size, making it
difficult to train deep learning-based models effectively. In this research, we
address these challenges by leveraging LLM-based text embedders in PCR.
LLM-based embedders support longer input lengths, and since we use them in an
unsupervised manner, they do not require training data, addressing both
challenges simultaneously. In this paper, we evaluate state-of-the-art
LLM-based text embedders in four PCR benchmark datasets and show that they
outperform BM25 and supervised transformer-based models.

### 7. [DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data](http://arxiv.org/pdf/2507.18583v1)

Authors: Zhengyun Zhao, Huaiyuan Ying, Yue Zhong, Sheng Yu

Electronic Health Records (EHRs) are pivotal in clinical practices, yet their
retrieval remains a challenge mainly due to semantic gap issues. Recent
advancements in dense retrieval offer promising solutions but existing models,
both general-domain and biomedical-domain, fall short due to insufficient
medical knowledge or mismatched training corpora. This paper introduces
\texttt{DR.EHR}, a series of dense retrieval models specifically tailored for
EHR retrieval. We propose a two-stage training pipeline utilizing MIMIC-IV
discharge summaries to address the need for extensive medical knowledge and
large-scale training data. The first stage involves medical entity extraction
and knowledge injection from a biomedical knowledge graph, while the second
stage employs large language models to generate diverse training data. We train
two variants of \texttt{DR.EHR}, with 110M and 7B parameters, respectively.
Evaluated on the CliniQ benchmark, our models significantly outperforms all
existing dense retrievers, achieving state-of-the-art results. Detailed
analyses confirm our models' superiority across various match and query types,
particularly in challenging semantic matches like implication and abbreviation.
Ablation studies validate the effectiveness of each pipeline component, and
supplementary experiments on EHR QA datasets demonstrate the models'
generalizability on natural language questions, including complex ones with
multiple entities. This work significantly advances EHR retrieval, offering a
robust solution for clinical applications.

### Machine Learning

### 1. [Predictive Scaling Laws for Efficient GRPO Training of Large Reasoning Models](http://arxiv.org/pdf/2507.18014v1)

Authors: Datta Nimmaturi, Vaishnavi Bhargava, Rajat Ghosh, Johnu George, Debojyoti Dutta

Fine-tuning large language models (LLMs) for reasoning tasks using
reinforcement learning methods like Group Relative Policy Optimization (GRPO)
is computationally expensive. To address this, we propose a predictive
framework that models training dynamics and helps optimize resource usage.
Through experiments on Llama and Qwen models (3B 8B), we derive an empirical
scaling law based on model size, initial performance, and training progress.
This law predicts reward trajectories and identifies three consistent training
phases: slow start, rapid improvement, and plateau. We find that training
beyond certain number of an epoch offers little gain, suggesting earlier
stopping can significantly reduce compute without sacrificing performance. Our
approach generalizes across model types, providing a practical guide for
efficient GRPO-based fine-tuning.

### 2. [C-AAE: Compressively Anonymizing Autoencoders for Privacy-Preserving Activity Recognition in Healthcare Sensor Streams](http://arxiv.org/pdf/2507.18072v1)

Authors: Ryusei Fujimoto, Yugo Nakamura, Yutaka Arakawa

Wearable accelerometers and gyroscopes encode fine-grained behavioural
signatures that can be exploited to re-identify users, making privacy
protection essential for healthcare applications. We introduce C-AAE, a
compressive anonymizing autoencoder that marries an Anonymizing AutoEncoder
(AAE) with Adaptive Differential Pulse-Code Modulation (ADPCM). The AAE first
projects raw sensor windows into a latent space that retains activity-relevant
features while suppressing identity cues. ADPCM then differentially encodes
this latent stream, further masking residual identity information and shrinking
the bitrate. Experiments on the MotionSense and PAMAP2 datasets show that C-AAE
cuts user re-identification F1 scores by 10-15 percentage points relative to
AAE alone, while keeping activity-recognition F1 within 5 percentage points of
the unprotected baseline. ADPCM also reduces data volume by roughly 75 %,
easing transmission and storage overheads. These results demonstrate that C-AAE
offers a practical route to balancing privacy and utility in continuous,
sensor-based activity recognition for healthcare.

### 3. [Squeeze10-LLM: Squeezing LLMs' Weights by 10 Times via a Staged Mixed-Precision Quantization Method](http://arxiv.org/pdf/2507.18073v1)

Authors: Qingcheng Zhu, Yangyang Ren, Linlin Yang, Mingbao Lin, Yanjing Li, Sheng Xu, Zichao Feng, Haodong Zhu, Yuguang Yang, Juan Zhang, Runqi Wang, Baochang Zhang

Deploying large language models (LLMs) is challenging due to their massive
parameters and high computational costs. Ultra low-bit quantization can
significantly reduce storage and accelerate inference, but extreme compression
(i.e., mean bit-width <= 2) often leads to severe performance degradation. To
address this, we propose Squeeze10-LLM, effectively "squeezing" 16-bit LLMs'
weights by 10 times. Specifically, Squeeze10-LLM is a staged mixed-precision
post-training quantization (PTQ) framework and achieves an average of 1.6 bits
per weight by quantizing 80% of the weights to 1 bit and 20% to 4 bits. We
introduce Squeeze10LLM with two key innovations: Post-Binarization Activation
Robustness (PBAR) and Full Information Activation Supervision (FIAS). PBAR is a
refined weight significance metric that accounts for the impact of quantization
on activations, improving accuracy in low-bit settings. FIAS is a strategy that
preserves full activation information during quantization to mitigate
cumulative error propagation across layers. Experiments on LLaMA and LLaMA2
show that Squeeze10-LLM achieves state-of-the-art performance for sub-2bit
weight-only quantization, improving average accuracy from 43% to 56% on six
zero-shot classification tasks--a significant boost over existing PTQ methods.
Our code will be released upon publication.

### 4. [Learning from Hard Labels with Additional Supervision on Non-Hard-Labeled Classes](http://arxiv.org/pdf/2507.18098v1)

Authors: Kosuke Sugiyama, Masato Uchida

In scenarios where training data is limited due to observation costs or data
scarcity, enriching the label information associated with each instance becomes
crucial for building high-accuracy classification models. In such contexts, it
is often feasible to obtain not only hard labels but also {\it additional
supervision}, such as the confidences for the hard labels. This setting
naturally raises fundamental questions: {\it What kinds of additional
supervision are intrinsically beneficial?} And {\it how do they contribute to
improved generalization performance?} To address these questions, we propose a
theoretical framework that treats both hard labels and additional supervision
as probability distributions, and constructs soft labels through their affine
combination. Our theoretical analysis reveals that the essential component of
additional supervision is not the confidence score of the assigned hard label,
but rather the information of the distribution over the non-hard-labeled
classes. Moreover, we demonstrate that the additional supervision and the
mixing coefficient contribute to the refinement of soft labels in complementary
roles. Intuitively, in the probability simplex, the additional supervision
determines the direction in which the deterministic distribution representing
the hard label should be adjusted toward the true label distribution, while the
mixing coefficient controls the step size along that direction. Through
generalization error analysis, we theoretically characterize how the additional
supervision and its mixing coefficient affect both the convergence rate and
asymptotic value of the error bound. Finally, we experimentally demonstrate
that, based on our theory, designing additional supervision can lead to
improved classification accuracy, even when utilized in a simple manner.

### 5. [Percentile-Based Deep Reinforcement Learning and Reward Based Personalization For Delay Aware RAN Slicing in O-RAN](http://arxiv.org/pdf/2507.18111v1)

Authors: Peyman Tehrani, Anas Alsoliman

In this paper, we tackle the challenge of radio access network (RAN) slicing
within an open RAN (O-RAN) architecture. Our focus centers on a network that
includes multiple mobile virtual network operators (MVNOs) competing for
physical resource blocks (PRBs) with the goal of meeting probabilistic delay
upper bound constraints for their clients while minimizing PRB utilization.
Initially, we derive a reward function based on the law of large numbers (LLN),
then implement practical modifications to adapt it for real-world experimental
scenarios. We then propose our solution, the Percentile-based Delay-Aware Deep
Reinforcement Learning (PDA-DRL), which demonstrates its superiority over
several baselines, including DRL models optimized for average delay
constraints, by achieving a 38\% reduction in resultant average delay.
Furthermore, we delve into the issue of model weight sharing among multiple
MVNOs to develop a robust personalized model. We introduce a reward-based
personalization method where each agent prioritizes other agents' model weights
based on their performance. This technique surpasses traditional aggregation
methods, such as federated averaging, and strategies reliant on traffic
patterns and model weight distance similarities.

### 6. [Policy Disruption in Reinforcement Learning:Adversarial Attack with Large Language Models and Critical State Identification](http://arxiv.org/pdf/2507.18113v1)

Authors: Junyong Jiang, Buwei Tian, Chenxing Xu, Songze Li, Lu Dong

Reinforcement learning (RL) has achieved remarkable success in fields like
robotics and autonomous driving, but adversarial attacks designed to mislead RL
systems remain challenging. Existing approaches often rely on modifying the
environment or policy, limiting their practicality. This paper proposes an
adversarial attack method in which existing agents in the environment guide the
target policy to output suboptimal actions without altering the environment. We
propose a reward iteration optimization framework that leverages large language
models (LLMs) to generate adversarial rewards explicitly tailored to the
vulnerabilities of the target agent, thereby enhancing the effectiveness of
inducing the target agent toward suboptimal decision-making. Additionally, a
critical state identification algorithm is designed to pinpoint the target
agent's most vulnerable states, where suboptimal behavior from the victim leads
to significant degradation in overall performance. Experimental results in
diverse environments demonstrate the superiority of our method over existing
approaches.

### 7. [Maximizing Prefix-Confidence at Test-Time Efficiently Improves Mathematical Reasoning](http://arxiv.org/pdf/2507.18122v1)

Authors: Matthias Otth, Jonas Hübotter, Ido Hakimi, Andreas Krause

Recent work has shown that language models can self-improve by maximizing
their own confidence in their predictions, without relying on external
verifiers or reward signals. In this work, we study the test-time scaling of
language models for mathematical reasoning tasks, where the model's own
confidence is used to select the most promising attempts. Surprisingly, we find
that we can achieve significant performance gains by continuing only the most
promising attempt, selected by the model's prefix-confidence. We systematically
evaluate prefix-confidence scaling on five mathematical reasoning datasets: the
school-level GSM8K and MATH500, and the competition-level AMC23, AIME24, and
AIME25. We find that prefix-confidence scaling with prefixes of only 32 tokens
achieves a better accuracy-compute trade-off than majority voting. Moreover,
prefix-confidence scaling appears less susceptible than BoN to length biases.
Finally, we also evaluate test-time training with prefix-confidence and find
that, while outperforming the base model, it does not improve over
prefix-confidence scaling.

### 8. [Neuromorphic Computing for Embodied Intelligence in Autonomous Systems: Current Trends, Challenges, and Future Directions](http://arxiv.org/pdf/2507.18139v1)

Authors: Alberto Marchisio, Muhammad Shafique

The growing need for intelligent, adaptive, and energy-efficient autonomous
systems across fields such as robotics, mobile agents (e.g., UAVs), and
self-driving vehicles is driving interest in neuromorphic computing. By drawing
inspiration from biological neural systems, neuromorphic approaches offer
promising pathways to enhance the perception, decision-making, and
responsiveness of autonomous platforms. This paper surveys recent progress in
neuromorphic algorithms, specialized hardware, and cross-layer optimization
strategies, with a focus on their deployment in real-world autonomous
scenarios. Special attention is given to event-based dynamic vision sensors and
their role in enabling fast, efficient perception. The discussion highlights
new methods that improve energy efficiency, robustness, adaptability, and
reliability through the integration of spiking neural networks into autonomous
system architectures. We integrate perspectives from machine learning,
robotics, neuroscience, and neuromorphic engineering to offer a comprehensive
view of the state of the field. Finally, emerging trends and open challenges
are explored, particularly in the areas of real-time decision-making, continual
learning, and the development of secure, resilient autonomous systems.

### 9. [Goal-based Trajectory Prediction for improved Cross-Dataset Generalization](http://arxiv.org/pdf/2507.18196v1)

Authors: Daniel Grimm, Ahmed Abouelazm, J. Marius Zöllner

To achieve full autonomous driving, a good understanding of the surrounding
environment is necessary. Especially predicting the future states of other
traffic participants imposes a non-trivial challenge. Current SotA-models
already show promising results when trained on real datasets (e.g. Argoverse2,
NuScenes). Problems arise when these models are deployed to new/unseen areas.
Typically, performance drops significantly, indicating that the models lack
generalization. In this work, we introduce a new Graph Neural Network (GNN)
that utilizes a heterogeneous graph consisting of traffic participants and
vectorized road network. Latter, is used to classify goals, i.e. endpoints of
the predicted trajectories, in a multi-staged approach, leading to a better
generalization to unseen scenarios. We show the effectiveness of the goal
selection process via cross-dataset evaluation, i.e. training on Argoverse2 and
evaluating on NuScenes.

### 10. [Boosting Revisited: Benchmarking and Advancing LP-Based Ensemble Methods](http://arxiv.org/pdf/2507.18242v1)

Authors: Fabian Akkerman, Julien Ferry, Christian Artigues, Emmanuel Hebrard, Thibaut Vidal

Despite their theoretical appeal, totally corrective boosting methods based
on linear programming have received limited empirical attention. In this paper,
we conduct the first large-scale experimental study of six LP-based boosting
formulations, including two novel methods, NM-Boost and QRLP-Boost, across 20
diverse datasets. We evaluate the use of both heuristic and optimal base
learners within these formulations, and analyze not only accuracy, but also
ensemble sparsity, margin distribution, anytime performance, and hyperparameter
sensitivity. We show that totally corrective methods can outperform or match
state-of-the-art heuristics like XGBoost and LightGBM when using shallow trees,
while producing significantly sparser ensembles. We further show that these
methods can thin pre-trained ensembles without sacrificing performance, and we
highlight both the strengths and limitations of using optimal decision trees in
this context.

### Neural and Evolutionary Computing

### 1. [Contraction, Criticality, and Capacity: A Dynamical-Systems Perspective on Echo-State Networks](http://arxiv.org/pdf/2507.18467v1)

Authors: Pradeep Singh, Lavanya Sankaranarayanan, Balasubramanian Raman

Echo-State Networks (ESNs) distil a key neurobiological insight: richly
recurrent but fixed circuitry combined with adaptive linear read-outs can
transform temporal streams with remarkable efficiency. Yet fundamental
questions about stability, memory and expressive power remain fragmented across
disciplines. We present a unified, dynamical-systems treatment that weaves
together functional analysis, random attractor theory and recent
neuroscientific findings. First, on compact multivariate input alphabets we
prove that the Echo-State Property (wash-out of initial conditions) together
with global Lipschitz dynamics necessarily yields the Fading-Memory Property
(geometric forgetting of remote inputs). Tight algebraic tests translate
activation-specific Lipschitz constants into certified spectral-norm bounds,
covering both saturating and rectifying nonlinearities. Second, employing a
Stone-Weierstrass strategy we give a streamlined proof that ESNs with
polynomial reservoirs and linear read-outs are dense in the Banach space of
causal, time-invariant fading-memory filters, extending universality to
stochastic inputs. Third, we quantify computational resources via
memory-capacity spectrum, show how topology and leak rate redistribute
delay-specific capacities, and link these trade-offs to Lyapunov spectra at the
\textit{edge of chaos}. Finally, casting ESNs as skew-product random dynamical
systems, we establish existence of singleton pullback attractors and derive
conditional Lyapunov bounds, providing a rigorous analogue to cortical
criticality. The analysis yields concrete design rules-spectral radius, input
gain, activation choice-grounded simultaneously in mathematics and
neuroscience, and clarifies why modest-sized reservoirs often rival fully
trained recurrent networks in practice.

### 2. [Explicit Sign-Magnitude Encoders Enable Power-Efficient Multipliers](http://arxiv.org/pdf/2507.18179v1)

Authors: Felix Arnold, Maxence Bouvier, Ryan Amaudruz, Renzo Andri, Lukas Cavigelli

This work presents a method to maximize power-efficiency of fixed point
multiplier units by decomposing them into sub-components. First, an encoder
block converts the operands from a two's complement to a sign magnitude
representation, followed by a multiplier module which performs the compute
operation and outputs the resulting value in the original format. This allows
to leverage the power-efficiency of the Sign Magnitude encoding for the
multiplication. To ensure the computing format is not altered, those two
components are synthesized and optimized separately. Our method leads to
significant power savings for input values centered around zero, as commonly
encountered in AI workloads. Under a realistic input stream with values
normally distributed with a standard deviation of 3.0, post-synthesis
simulations of the 4-bit multiplier design show up to 12.9% lower switching
activity compared to synthesis without decomposition. Those gains are achieved
while ensuring compliance into any production-ready system as the overall
circuit stays logic-equivalent. With the compliance lifted and a slightly
smaller input range of -7 to +7, switching activity reductions can reach up to
33%. Additionally, we demonstrate that synthesis optimization methods based on
switching-activity-driven design space exploration can yield a further 5-10%
improvement in power-efficiency compared to a power agnostic approach.

### 3. [On the Performance of Concept Probing: The Influence of the Data (Extended Version)](http://arxiv.org/pdf/2507.18550v1)

Authors: Manuel de Sousa Ribeiro, Afonso Leote, João Leite

Concept probing has recently garnered increasing interest as a way to help
interpret artificial neural networks, dealing both with their typically large
size and their subsymbolic nature, which ultimately renders them unfeasible for
direct human interpretation. Concept probing works by training additional
classifiers to map the internal representations of a model into human-defined
concepts of interest, thus allowing humans to peek inside artificial neural
networks. Research on concept probing has mainly focused on the model being
probed or the probing model itself, paying limited attention to the data
required to train such probing models. In this paper, we address this gap.
Focusing on concept probing in the context of image classification tasks, we
investigate the effect of the data used to train probing models on their
performance. We also make available concept labels for two widely used
datasets.

### Networking and Internet Architecture

### 1. [Enhanced Velocity-Adaptive Scheme: Joint Fair Access and Age of Information Optimization in Vehicular Networks](http://arxiv.org/pdf/2507.18328v1)

Authors: Xiao Xu, Qiong Wu, Pingyi Fan, Kezhi Wang, Nan Cheng, Wen Chen, Khaled B. Letaief

In this paper, we consider the fair access problem and the Age of Information
(AoI) under 5G New Radio (NR) Vehicle-to-Infrastructure (V2I) Mode 2 in
vehicular networks. Specifically, vehicles follow Mode 2 to communicate with
Roadside Units (RSUs) to obtain accurate data for driving
assistance.Nevertheless, vehicles often have different velocity when they are
moving in adjacent lanes, leading to difference in RSU dwelltime and
communication duration. This results in unfair access to network resources,
potentially influencing driving safety. To ensure the freshness of received
data, the AoI should be analyzed. Mode 2 introduces a novel preemption
mechanism, necessitating simultaneous optimization of fair access and AoI to
guarantee timely and relevant data delivery. We propose a joint optimization
framework for vehicular network, defining a fairness index and employing
Stochastic Hybrid Systems (SHS) to model AoI under preemption mechanism. By
adaptively adjusting the selection window of Semi-Persistent Scheduling (SPS)
in Mode 2, we address the optimization of fairness and AoI. We apply a large
language model (LLM)-Based Multi-objective Evolutionary Algorithm Based on
Decomposition (MOEA/D) to solve this problem. Simulation results demonstrate
the effectiveness of our scheme in balancing fair access and minimizing AoI.

### 2. [Improving Wi-Fi 8 Latency with Coordinated Spatial Reuse](http://arxiv.org/pdf/2507.18480v1)

Authors: David Nunez, Francesc Wilhelmi, Lorenzo Galati-Giordano, Giovanni Geraci, Boris Bellalta

IEEE 802.11 networks continuously adapt to meet the stringent requirements of
emerging applications like cloud gaming, eXtended Reality (XR), and video
streaming services, which require high throughput, low latency, and high
reliability. To address these challenges, Coordinated Spatial Reuse (Co-SR) can
potentially contribute to optimizing spectrum resource utilization. This
mechanism is expected to enable simultaneous transmissions, thereby boosting
spectral efficiency in dense environments and increasing the overall network
performance. In this paper, we shed light on the performance of Co-SR for Wi-Fi
8 networks. For that, we propose an implementation of Co-SR aligned with
ongoing Wi-Fi 8 standardization efforts. The evaluation is done on a Wi-Fi
simulator, which allows us to study the performance of the proposed Co-SR
mechanisms in relevant scenarios. The results obtained in a Wireless Local Area
Network (WLAN) consisting of four APs show delay reduction with Co-SR ranging
from 31% to 95% when compared to Distributed Coordination Function (DCF).

### 3. [On the Role of Age and Semantics of Information in Remote Estimation of Markov Sources](http://arxiv.org/pdf/2507.18514v1)

Authors: Jiping Luo, Nikolaos Pappas

This paper investigates the semantics-aware remote estimation of a
finite-state Markov chain. We employ the maximum a posteriori (MAP) estimator
and aim to devise a transmission policy to optimize estimation performance
subject to a transmission frequency constraint. We leverage two metrics, namely
the Age of Consecutive Error (AoCE) and the Age of Information (AoI), to
quantify, respectively, the significance of estimation error at the transmitter
and the predictability of outdated information at the receiver. The optimal
transmission problem is formulated as a constrained Markov decision process
(CMDP) with unbounded costs. We show the existence of an optimal simple mixture
policy, which randomly selects between two deterministic switching policies
with a fixed probability. Notably, each switching policy triggers a
transmission only when the AoCE exceeds a threshold value that depends on both
the AoI and the instantaneous estimation error. We further derive sufficient
conditions under which the switching policy reduces to a simple threshold
policy; that is, it admits identical thresholds for all estimation errors.
Leveraging these results, we develop an efficient structure-aware algorithm,
Insec-SPI, that computes the optimal policy with reduced computation overhead.
Our results demonstrate that incorporating both AoI and AoCE yields
significantly improved estimation quality compared to using either metric
alone.

### Robotics

### 1. [A Modular Residual Learning Framework to Enhance Model-Based Approach for Robust Locomotion](http://arxiv.org/pdf/2507.18138v1)

Authors: Min-Gyu Kim, Dongyun Kang, Hajun Kim, Hae-Won Park

This paper presents a novel approach that combines the advantages of both
model-based and learning-based frameworks to achieve robust locomotion. The
residual modules are integrated with each corresponding part of the model-based
framework, a footstep planner and dynamic model designed using heuristics, to
complement performance degradation caused by a model mismatch. By utilizing a
modular structure and selecting the appropriate learning-based method for each
residual module, our framework demonstrates improved control performance in
environments with high uncertainty, while also achieving higher learning
efficiency compared to baseline methods. Moreover, we observed that our
proposed methodology not only enhances control performance but also provides
additional benefits, such as making nominal controllers more robust to
parameter tuning. To investigate the feasibility of our framework, we
demonstrated residual modules combined with model predictive control in a real
quadrupedal robot. Despite uncertainties beyond the simulation, the robot
successfully maintains balance and tracks the commanded velocity.

### 2. [Autonomous UAV Navigation for Search and Rescue Missions Using Computer Vision and Convolutional Neural Networks](http://arxiv.org/pdf/2507.18160v1)

Authors: Luka Šiktar, Branimir Ćaran, Bojan Šekoranja, Marko Švaco

In this paper, we present a subsystem, using Unmanned Aerial Vehicles (UAV),
for search and rescue missions, focusing on people detection, face recognition
and tracking of identified individuals. The proposed solution integrates a UAV
with ROS2 framework, that utilizes multiple convolutional neural networks (CNN)
for search missions. System identification and PD controller deployment are
performed for autonomous UAV navigation. The ROS2 environment utilizes the
YOLOv11 and YOLOv11-pose CNNs for tracking purposes, and the dlib library CNN
for face recognition. The system detects a specific individual, performs face
recognition and starts tracking. If the individual is not yet known, the UAV
operator can manually locate the person, save their facial image and
immediately initiate the tracking process. The tracking process relies on
specific keypoints identified on the human body using the YOLOv11-pose CNN
model. These keypoints are used to track a specific individual and maintain a
safe distance. To enhance accurate tracking, system identification is
performed, based on measurement data from the UAVs IMU. The identified system
parameters are used to design PD controllers that utilize YOLOv11-pose to
estimate the distance between the UAVs camera and the identified individual.
The initial experiments, conducted on 14 known individuals, demonstrated that
the proposed subsystem can be successfully used in real time. The next step
involves implementing the system on a large experimental UAV for field use and
integrating autonomous navigation with GPS-guided control for rescue operations
planning.

### 3. [AF-RLIO: Adaptive Fusion of Radar-LiDAR-Inertial Information for Robust Odometry in Challenging Environments](http://arxiv.org/pdf/2507.18317v1)

Authors: Chenglong Qian, Yang Xu, Xiufang Shi, Jiming Chen, Liang Li

In robotic navigation, maintaining precise pose estimation and navigation in
complex and dynamic environments is crucial. However, environmental challenges
such as smoke, tunnels, and adverse weather can significantly degrade the
performance of single-sensor systems like LiDAR or GPS, compromising the
overall stability and safety of autonomous robots. To address these challenges,
we propose AF-RLIO: an adaptive fusion approach that integrates 4D
millimeter-wave radar, LiDAR, inertial measurement unit (IMU), and GPS to
leverage the complementary strengths of these sensors for robust odometry
estimation in complex environments. Our method consists of three key modules.
Firstly, the pre-processing module utilizes radar data to assist LiDAR in
removing dynamic points and determining when environmental conditions are
degraded for LiDAR. Secondly, the dynamic-aware multimodal odometry selects
appropriate point cloud data for scan-to-map matching and tightly couples it
with the IMU using the Iterative Error State Kalman Filter. Lastly, the factor
graph optimization module balances weights between odometry and GPS data,
constructing a pose graph for optimization. The proposed approach has been
evaluated on datasets and tested in real-world robotic environments,
demonstrating its effectiveness and advantages over existing methods in
challenging conditions such as smoke and tunnels.

### 4. [G2S-ICP SLAM: Geometry-aware Gaussian Splatting ICP SLAM](http://arxiv.org/pdf/2507.18344v1)

Authors: Gyuhyeon Pak, Hae Min Cho, Euntai Kim

In this paper, we present a novel geometry-aware RGB-D Gaussian Splatting
SLAM system, named G2S-ICP SLAM. The proposed method performs high-fidelity 3D
reconstruction and robust camera pose tracking in real-time by representing
each scene element using a Gaussian distribution constrained to the local
tangent plane. This effectively models the local surface as a 2D Gaussian disk
aligned with the underlying geometry, leading to more consistent depth
interpretation across multiple viewpoints compared to conventional 3D
ellipsoid-based representations with isotropic uncertainty. To integrate this
representation into the SLAM pipeline, we embed the surface-aligned Gaussian
disks into a Generalized ICP framework by introducing anisotropic covariance
prior without altering the underlying registration formulation. Furthermore we
propose a geometry-aware loss that supervises photometric, depth, and normal
consistency. Our system achieves real-time operation while preserving both
visual and geometric fidelity. Extensive experiments on the Replica and
TUM-RGBD datasets demonstrate that G2S-ICP SLAM outperforms prior SLAM systems
in terms of localization accuracy, reconstruction completeness, while
maintaining the rendering quality.

### 5. [Residual Koopman Model Predictive Control for Enhanced Vehicle Dynamics with Small On-Track Data Input](http://arxiv.org/pdf/2507.18396v1)

Authors: Yonghao Fu, Cheng Hu, Haokun Xiong, Zhangpeng Bao, Wenyuan Du, Edoardo Ghignone, Michele Magno, Lei Xie, Hongye Su

In vehicle trajectory tracking tasks, the simplest approach is the Pure
Pursuit (PP) Control. However, this single-point preview tracking strategy
fails to consider vehicle model constraints, compromising driving safety. Model
Predictive Control (MPC) as a widely adopted control method, optimizes control
actions by incorporating mechanistic models and physical constraints. While its
control performance critically depends on the accuracy of vehicle modeling.
Traditional vehicle modeling approaches face inherent trade-offs between
capturing nonlinear dynamics and maintaining computational efficiency, often
resulting in reduced control performance. To address these challenges, this
paper proposes Residual Koopman Model Predictive Control (RKMPC) framework.
This method uses two linear MPC architecture to calculate control inputs: a
Linear Model Predictive Control (LMPC) computes the baseline control input
based on the vehicle kinematic model, and a neural network-based RKMPC
calculates the compensation input. The final control command is obtained by
adding these two components. This design preserves the reliability and
interpretability of traditional mechanistic model while achieving performance
optimization through residual modeling. This method has been validated on the
Carsim-Matlab joint simulation platform and a physical 1:10 scale F1TENTH
racing car. Experimental results show that RKMPC requires only 20% of the
training data needed by traditional Koopman Model Predictive Control (KMPC)
while delivering superior tracking performance. Compared to traditional LMPC,
RKMPC reduces lateral error by 11.7%-22.1%, decreases heading error by
8.9%-15.8%, and improves front-wheel steering stability by up to 27.6%. The
implementation code is available at: https://github.com/ZJU-DDRX/Residual
Koopman.

### 6. [Evaluating the Pre-Dressing Step: Unfolding Medical Garments Via Imitation Learning](http://arxiv.org/pdf/2507.18436v1)

Authors: David Blanco-Mulero, Júlia Borràs, Carme Torras

Robotic-assisted dressing has the potential to significantly aid both
patients as well as healthcare personnel, reducing the workload and improving
the efficiency in clinical settings. While substantial progress has been made
in robotic dressing assistance, prior works typically assume that garments are
already unfolded and ready for use. However, in medical applications gowns and
aprons are often stored in a folded configuration, requiring an additional
unfolding step. In this paper, we introduce the pre-dressing step, the process
of unfolding garments prior to assisted dressing. We leverage imitation
learning for learning three manipulation primitives, including both high and
low acceleration motions. In addition, we employ a visual classifier to
categorise the garment state as closed, partly opened, and fully opened. We
conduct an empirical evaluation of the learned manipulation primitives as well
as their combinations. Our results show that highly dynamic motions are not
effective for unfolding freshly unpacked garments, where the combination of
motions can efficiently enhance the opening configuration.

### 7. [A Novel Monte-Carlo Compressed Sensing and Dictionary Learning Method for the Efficient Path Planning of Remote Sensing Robots](http://arxiv.org/pdf/2507.18462v1)

Authors: Alghalya Al-Hajri, Ejmen Al-Ubejdij, Aiman Erbad, Ali Safa

In recent years, Compressed Sensing (CS) has gained significant interest as a
technique for acquiring high-resolution sensory data using fewer measurements
than traditional Nyquist sampling requires. At the same time, autonomous
robotic platforms such as drones and rovers have become increasingly popular
tools for remote sensing and environmental monitoring tasks, including
measurements of temperature, humidity, and air quality. Within this context,
this paper presents, to the best of our knowledge, the first investigation into
how the structure of CS measurement matrices can be exploited to design
optimized sampling trajectories for robotic environmental data collection. We
propose a novel Monte Carlo optimization framework that generates measurement
matrices designed to minimize both the robot's traversal path length and the
signal reconstruction error within the CS framework. Central to our approach is
the application of Dictionary Learning (DL) to obtain a data-driven sparsifying
transform, which enhances reconstruction accuracy while further reducing the
number of samples that the robot needs to collect. We demonstrate the
effectiveness of our method through experiments reconstructing $NO_2$ pollution
maps over the Gulf region. The results indicate that our approach can reduce
robot travel distance to less than $10\%$ of a full-coverage path, while
improving reconstruction accuracy by over a factor of five compared to
traditional CS methods based on DCT and polynomial dictionaries, as well as by
a factor of two compared to previously-proposed Informative Path Planning (IPP)
methods.

### 8. [Experimental Comparison of Whole-Body Control Formulations for Humanoid Robots in Task Acceleration and Task Force Spaces](http://arxiv.org/pdf/2507.18502v1)

Authors: Sait Sovukluk, Grazia Zambella, Tobias Egle, Christian Ott

This paper studies the experimental comparison of two different whole-body
control formulations for humanoid robots: inverse dynamics whole-body control
(ID-WBC) and passivity-based whole-body control (PB-WBC). The two controllers
fundamentally differ from each other as the first is formulated in task
acceleration space and the latter is in task force space with passivity
considerations. Even though both control methods predict stability under ideal
conditions in closed-loop dynamics, their robustness against joint friction,
sensor noise, unmodeled external disturbances, and non-perfect contact
conditions is not evident. Therefore, we analyze and experimentally compare the
two controllers on a humanoid robot platform through swing foot position and
orientation control, squatting with and without unmodeled additional weights,
and jumping. We also relate the observed performance and characteristic
differences with the controller formulations and highlight each controller's
advantages and disadvantages.

### 9. [OpenNav: Open-World Navigation with Multimodal Large Language Models](http://arxiv.org/pdf/2507.18033v1)

Authors: Mingfeng Yuan, Letian Wang, Steven L. Waslander

Pre-trained large language models (LLMs) have demonstrated strong
common-sense reasoning abilities, making them promising for robotic navigation
and planning tasks. However, despite recent progress, bridging the gap between
language descriptions and actual robot actions in the open-world, beyond merely
invoking limited predefined motion primitives, remains an open challenge. In
this work, we aim to enable robots to interpret and decompose complex language
instructions, ultimately synthesizing a sequence of trajectory points to
complete diverse navigation tasks given open-set instructions and open-set
objects. We observe that multi-modal large language models (MLLMs) exhibit
strong cross-modal understanding when processing free-form language
instructions, demonstrating robust scene comprehension. More importantly,
leveraging their code-generation capability, MLLMs can interact with
vision-language perception models to generate compositional 2D bird-eye-view
value maps, effectively integrating semantic knowledge from MLLMs with spatial
information from maps to reinforce the robot's spatial understanding. To
further validate our approach, we effectively leverage large-scale autonomous
vehicle datasets (AVDs) to validate our proposed zero-shot vision-language
navigation framework in outdoor navigation tasks, demonstrating its capability
to execute a diverse range of free-form natural language navigation
instructions while maintaining robustness against object detection errors and
linguistic ambiguities. Furthermore, we validate our system on a Husky robot in
both indoor and outdoor scenes, demonstrating its real-world robustness and
applicability. Supplementary videos are available at
https://trailab.github.io/OpenNav-website/

### 10. [MoRPI-PINN: A Physics-Informed Framework for Mobile Robot Pure Inertial Navigation](http://arxiv.org/pdf/2507.18206v1)

Authors: Arup Kumar Sahoo, Itzik Klein

A fundamental requirement for full autonomy in mobile robots is accurate
navigation even in situations where satellite navigation or cameras are
unavailable. In such practical situations, relying only on inertial sensors
will result in navigation solution drift due to the sensors' inherent noise and
error terms. One of the emerging solutions to mitigate drift is to maneuver the
robot in a snake-like slithering motion to increase the inertial
signal-to-noise ratio, allowing the regression of the mobile robot position. In
this work, we propose MoRPI-PINN as a physics-informed neural network framework
for accurate inertial-based mobile robot navigation. By embedding physical laws
and constraints into the training process, MoRPI-PINN is capable of providing
an accurate and robust navigation solution. Using real-world experiments, we
show accuracy improvements of over 85% compared to other approaches. MoRPI-PINN
is a lightweight approach that can be implemented even on edge devices and used
in any typical mobile robot application.

### Software Engineering

### 1. [An Empirical Study of GenAI Adoption in Open-Source Game Development: Tools, Tasks, and Developer Challenges](http://arxiv.org/pdf/2507.18029v1)

Authors: Xiang Echo Chen, Wenhan Zhu, Guoshuai Albert Shi, Michael W. Godfrey

The growing capabilities of generative AI (GenAI) have begun to reshape how
games are designed and developed, offering new tools for content creation,
gameplay simulation, and design ideation. While prior research has explored
traditional uses of AI in games, such as controlling agents or generating
procedural content. There is limited empirical understanding of how GenAI is
adopted by developers in real-world contexts, especially within the open-source
community. This study aims to explore how GenAI technologies are discussed,
adopted, and integrated into open-source game development by analyzing issue
discussions on GitHub. We investigate the tools, tasks, and challenges
associated with GenAI by comparing GenAI-related issues to those involving
traditional AI (TradAI) and NonAI topics. Our goal is to uncover how GenAI
differs from other approaches in terms of usage patterns, developer concerns,
and integration practices. To address this objective, we construct a dataset of
open-source game repositories that discuss AI-related topics. We apply open
card sorting and thematic analysis to a stratified sample of GitHub issues,
labelling each by type and content. These annotations enable comparative
analysis across GenAI, TradAI, and NonAI groups, and provide insight into how
GenAI is shaping the workflows and pain points of open-source game developers.

### 2. [Factors Impacting Faculty Adoption of Project-Based Learning in Computing Education: a Survey](http://arxiv.org/pdf/2507.18039v1)

Authors: Ahmad D. Suleiman, Yiming Tang, Daqing Hou

This research full paper investigates the factors influencing computing
educators' adoption of project-based learning (PjBL) in software engineering
and computing curricula. Recognized as a student-centered pedagogical approach,
PjBL has the potential to enhance student motivation, engagement, critical
thinking, collaboration, and problem-solving skills. Despite these benefits,
faculty adoption remains inconsistent due to challenges such as insufficient
institutional support, time constraints, limited training opportunities,
designing or sourcing projects, and aligning them with course objectives. This
research explores these barriers and investigates the strategies and resources
that facilitate a successful adoption. Using a mixed-methods approach, data
from 80 computing faculty were collected through an online survey comprising
closed-ended questions to quantify barriers, enablers, and resource needs,
along with an open-ended question to gather qualitative insights. Quantitative
data were analyzed using statistical methods, while qualitative responses
underwent thematic analysis. Results reveal that while PjBL is widely valued,
its adoption is often selective and impacted by challenges in planning and
managing the learning process, designing suitable projects, and a lack of
institutional support, such as time, funding, and teaching assistants. Faculty
are more likely to adopt or sustain PjBL when they have access to peer
collaboration, professional development, and institutional incentives. In
addition, sourcing projects from research, industry partnerships, and borrowing
from peers emerged as key facilitators for new projects. These findings
underscore the need for systemic support structures to empower faculty to
experiment with and scale PjBL practices.

### 3. [An Empirical Study of Complexity, Heterogeneity, and Compliance of GitHub Actions Workflows](http://arxiv.org/pdf/2507.18062v1)

Authors: Edward Abrokwah, Taher A. Ghaleb

Continuous Integration (CI) has evolved from a tooling strategy to a
fundamental mindset in modern CI engineering. It enables teams to develop,
test, and deliver software rapidly and collaboratively. Among CI services,
GitHub Actions (GHA) has emerged as a dominant service due to its deep
integration with GitHub and a vast ecosystem of reusable workflow actions.
Although GHA provides official documentation and community-supported best
practices, there appears to be limited empirical understanding of how
open-source real-world CI workflows align with such practices. Many workflows
might be unnecessarily complex and not aligned with the simplicity goals of CI
practices. This study will investigate the structure, complexity,
heterogeneity, and compliance of GHA workflows in open-source software
repositories. Using a large dataset of GHA workflows from Java, Python, and C++
repositories, our goal is to (a) identify workflow complexities, (b) analyze
recurring and heterogeneous structuring patterns, (c) assess compliance with
GHA best practices, and (d) uncover differences in CI pipeline design across
programming languages. Our findings are expected to reveal both areas of strong
adherence to best practices and areas for improvement where needed. These
insights will also have implications for CI services, as they will highlight
the need for clearer guidelines and comprehensive examples in CI documentation.

### 4. [Identifier Name Similarities: An Exploratory Study](http://arxiv.org/pdf/2507.18081v1)

Authors: Carol Wong, Mai Abe, Silvia De Benedictis, Marissa Halim, Anthony Peruma

Identifier names, which comprise a significant portion of the codebase, are
the cornerstone of effective program comprehension. However, research has shown
that poorly chosen names can significantly increase cognitive load and hinder
collaboration. Even names that appear readable in isolation may lead to
misunderstandings in contexts when they closely resemble other names in either
structure or functionality. In this exploratory study, we present our
preliminary findings on the occurrence of identifier name similarity in
software projects through the development of a taxonomy that categorizes
different forms of identifier name similarity. We envision our initial taxonomy
providing researchers with a platform to analyze and evaluate the impact of
identifier name similarity on code comprehension, maintainability, and
collaboration among developers, while also allowing for further refinement and
expansion of the taxonomy.

### 5. [NoCode-bench: A Benchmark for Evaluating Natural Language-Driven Feature Addition](http://arxiv.org/pdf/2507.18130v1)

Authors: Le Deng, Zhonghao Jiang, Jialun Cao, Michael Pradel, Zhongxin Liu

Natural language-driven no-code development allows users to specify software
functionality using natural language (NL) instead of editing source code,
promising increased productivity and democratized development. Large language
models (LLMs) show potential in enabling this paradigm. In this context,
software documentation acts as an NL specification for functionality. This work
introduces NoCode-bench, a benchmark designed to evaluate LLMs on real-world
NL-driven feature addition tasks, consisting of 634 tasks across 10 projects
and 114k code changes. Each task pairs documentation updates with corresponding
code implementations, validated by developer-written test cases. A subset of
114 high-quality, human-verified instances, NoCode-bench Verified, ensures
reliable evaluation. Our experiments reveal that, despite high token usage, the
best LLMs achieve a task success rate of only 15.79%, highlighting challenges
in cross-file editing, codebase understanding, and tool calling. These findings
indicate that LLMs are not yet ready for fully NL-driven no-code development.
NoCode-bench lays the foundation for future advances in this area.

### 6. [An Empirical Study on Embodied Artificial Intelligence Robot (EAIR) Software Bugs](http://arxiv.org/pdf/2507.18267v1)

Authors: Zeqin Liao, Zibin Zheng, Peifan Reng, Henglong Liang, Zixu Gao, Zhixiang Chen, Wei Li, Yuhong Nan

Embodied Artificial Intelligence Robots (EAIR) is an emerging and rapidly
evolving technological domain. Ensuring their program correctness is
fundamental to their successful deployment. However, a general and in-depth
understanding of EAIR system bugs remains lacking, which hinders the
development of practices and techniques to tackle EAIR system bugs.
  To bridge this gap, we conducted the first systematic study of 885 EAIR
system bugs collected from 80 EAIR system projects to investigate their
symptoms, underlying causes, and module distribution. Our analysis takes
considerable effort, which classifies these bugs into 18 underlying causes, 15
distinct symptoms, and identifies 13 affected modules. It reveals several new
interesting findings and implications which help shed light on future research
on tackling or repairing EAIR system bugs. First, among the 15 identified
symptoms, our findings highlight 8 symptoms specific to EAIR systems, which is
characterized by severe functional failures and potential physical hazards.
Second, within the 18 underlying causes, we define 8 EAIR-specific causes, the
majority of which stem from the intricate issues of AI- agent reasoning and
decision making. Finally, to facilitate precise and efficient bug prediction,
detection, and repair, we constructed a mapping between underlying causes and
the modules in which they most frequently occur, which enables researchers to
focus diagnostic efforts on the modules most susceptible to specific bug types.

### 7. [YATE: The Role of Test Repair in LLM-Based Unit Test Generation](http://arxiv.org/pdf/2507.18316v1)

Authors: Michael Konstantinou, Renzo Degiovanni, Jie M. Zhang, Mark Harman, Mike Papadakis

Recent advances in automated test generation utilises language models to
produce unit tests. While effective, language models tend to generate many
incorrect tests with respect to both syntax and semantics. Although such
incorrect tests can be easily detected and discarded, they constitute a "missed
opportunity" -- if fixed, they are often valuable as they directly add testing
value (they effectively target the underlying program logic to be tested) and
indirectly form good seeds for generating additional tests. To this end, we
propose a simple technique for repairing some of these incorrect tests through
a combination of rule-based static analysis and re-prompting. We evaluate this
simple approach, named YATE, on a set of 6 open-source projects and show that
it can effectively produce tests that cover on average 32.06% more lines and
kill 21.77% more mutants than a plain LLM-based method. We also compare YATE
with four other LLM-based methods, namely HITS, SYMPROMPT, TESTSPARK and
COVERUP and show that it produces tests that cover substantially more code.
YATE achieves 22% higher line coverage, 20% higher branch coverage and kill 20%
more mutants at a comparable cost (number of calls to LLMs).

### 8. [Gotta catch 'em all! Towards File Localisation from Issues at Large](http://arxiv.org/pdf/2507.18319v1)

Authors: Jesse Maarleveld, Jiapan Guo, Daniel Feitosa

Bug localisation, the study of developing methods to localise the files
requiring changes to resolve bugs, has been researched for a long time to
develop methods capable of saving developers' time. Recently, researchers are
starting to consider issues outside of bugs. Nevertheless, most existing
research into file localisation from issues focusses on bugs or uses other
selection methods to ensure only certain types of issues are considered as part
of the focus of the work. Our goal is to work on all issues at large, without
any specific selection.
  In this work, we provide a data pipeline for the creation of issue file
localisation datasets, capable of dealing with arbitrary branching and merging
practices. We provide a baseline performance evaluation for the file
localisation problem using traditional information retrieval approaches.
Finally, we use statistical analysis to investigate the influence of biases
known in the bug localisation community on our dataset.
  Our results show that methods designed using bug-specific heuristics perform
poorly on general issue types, indicating a need for research into general
purpose models. Furthermore, we find that there are small, but statistically
significant differences in performance between different issue types. Finally,
we find that the presence of identifiers have a small effect on performance for
most issue types. Many results are project-dependent, encouraging the
development of methods which can be tuned to project-specific characteristics.

### 9. [A Deep Dive into Retrieval-Augmented Generation for Code Completion: Experience on WeChat](http://arxiv.org/pdf/2507.18515v1)

Authors: Zezhou Yang, Ting Peng, Cuiyun Gao, Chaozheng Wang, Hailiang Huang, Yuetang Deng

Code completion, a crucial task in software engineering that enhances
developer productivity, has seen substantial improvements with the rapid
advancement of large language models (LLMs). In recent years,
retrieval-augmented generation (RAG) has emerged as a promising method to
enhance the code completion capabilities of LLMs, which leverages relevant
context from codebases without requiring model retraining. While existing
studies have demonstrated the effectiveness of RAG on public repositories and
benchmarks, the potential distribution shift between open-source and
closed-source codebases presents unique challenges that remain unexplored. To
mitigate the gap, we conduct an empirical study to investigate the performance
of widely-used RAG methods for code completion in the industrial-scale codebase
of WeChat, one of the largest proprietary software systems. Specifically, we
extensively explore two main types of RAG methods, namely identifier-based RAG
and similarity-based RAG, across 26 open-source LLMs ranging from 0.5B to 671B
parameters. For a more comprehensive analysis, we employ different retrieval
techniques for similarity-based RAG, including lexical and semantic retrieval.
Based on 1,669 internal repositories, we achieve several key findings: (1) both
RAG methods demonstrate effectiveness in closed-source repositories, with
similarity-based RAG showing superior performance, (2) the effectiveness of
similarity-based RAG improves with more advanced retrieval techniques, where
BM25 (lexical retrieval) and GTE-Qwen (semantic retrieval) achieve superior
performance, and (3) the combination of lexical and semantic retrieval
techniques yields optimal results, demonstrating complementary strengths.
Furthermore, we conduct a developer survey to validate the practical utility of
RAG methods in real-world development environments.

### 10. [Your ATs to Ts: MITRE ATT&CK Attack Technique to P-SSCRM Task Mapping](http://arxiv.org/pdf/2507.18037v1)

Authors: Sivana Hamer, Jacob Bowen, Md Nazmul Haque, Chris Madden, Laurie Williams

The MITRE Adversarial Tactics, Techniques and Common Knowledge (MITRE ATT&CK)
Attack Technique to Proactive Software Supply Chain Risk Management Framework
(P-SSCRM) Task mapping described in this document helps software organizations
to determine how different tasks mitigate the attack techniques of software
supply chain attacks. The mapping was created through four independent
strategies to find agreed-upon mappings. Because each P-SSCRM task is mapped to
one or more tasks from the 10 frameworks, the mapping we provide is also a
mapping between MITRE ATT&CK and other prominent government and industry
frameworks.

### Systems and Control

### 1. [Quantitative Damping Calculation and Compensation Method for Global Stability Improvement of Inverter-Based Systems](http://arxiv.org/pdf/2507.18001v1)

Authors: Yang Li, Zenghui Zheng, Xiangyang Wu, Jiayong Li, Wei Wang, Qiang Zeng, Zhikang Shuai

Small-signal stability issues-induced broadband oscillations pose significant
threats to the secure operation of multi-inverter systems, attracting extensive
research attention. Researches revealed that system instability is led by the
lacking of positive damping, yet it has not been clearly specified how much the
exact amount of damping compensation required to sufficiently ensure system
global stability. This paper presents a feasible solution for quantitative
damping calculation and compensation to enhance the global stability of
inverter-based systems. First, based on the system nodal admittance model, a
quantitative damping calculation algorithm is presented, which can suggest the
required damping compensation as well as compensation location for sufficient
stability improvement. Then, we propose a specific AD with output current
feedforward control strategy, which make the AD be quasi-pure resistive and can
effectively enhance system damping efficiency. Finally, a testing system with
three inverters is used as case study, showing that the proposed method
provides a promising solution to efficiently enhance the global stability
improvement of inverter-based systems. Simulations and experiments validate the
proposed method.

### 2. [Carbon Emission Flow Tracing: Fast Algorithm and California Grid Study](http://arxiv.org/pdf/2507.18077v1)

Authors: Yuqing Shen, Yuanyuan Shi, Daniel Kirschen, Yize Chen

Power systems decarbonization are at the focal point of the clean energy
transition. While system operators and utility companies increasingly publicize
system-level carbon emission information, it remains unclear how emissions from
individual generators are transported through the grid and how they impact
electricity users at specific locations. This paper presents a novel and
computationally efficient approach for exact quantification of nodal average
and marginal carbon emission rates, applicable to both AC and DC optimal power
flow problems. The approach leverages graph-based topological sorting and
directed cycle removal techniques, applied to directed graphs formed by
generation dispatch and optimal power flow solutions. Our proposed algorithm
efficiently identifies each generator's contribution to each node, capturing
how emissions are spatially distributed under varying system conditions. To
validate its effectiveness and reveal locational and temporal emission patterns
in the real world, we simulate the 8,870-bus realistic California grid using
actual CAISO data and the CATS model. Based on year long hourly data on nodal
loads and renewable generation, obtained or estimated from CAISO public data,
our method accurately estimates power flow conditions, generation mixes, and
systemwide emissions, and delivers fine grained spatiotemporal emission
analysis for every California county. Both our algorithm and the California
study are open-sourced, providing a foundation for future research on grid
emissions, planning, operations, and energy policy.

### 3. [Towards Microgrid Resilience Enhancement via Mobile Power Sources and Repair Crews: A Multi-Agent Reinforcement Learning Approach](http://arxiv.org/pdf/2507.18095v1)

Authors: Yi Wang, Dawei Qiu, Fei Teng, Goran Strbac

Mobile power sources (MPSs) have been gradually deployed in microgrids as
critical resources to coordinate with repair crews (RCs) towards resilience
enhancement owing to their flexibility and mobility in handling the complex
coupled power-transport systems. However, previous work solves the coordinated
dispatch problem of MPSs and RCs in a centralized manner with the assumption
that the communication network is still fully functioning after the event.
However, there is growing evidence that certain extreme events will damage or
degrade communication infrastructure, which makes centralized decision making
impractical. To fill this gap, this paper formulates the resilience-driven
dispatch problem of MPSs and RCs in a decentralized framework. To solve this
problem, a hierarchical multi-agent reinforcement learning method featuring a
two-level framework is proposed, where the high-level action is used to switch
decision-making between power and transport networks, and the low-level action
constructed via a hybrid policy is used to compute continuous scheduling and
discrete routing decisions in power and transport networks, respectively. The
proposed method also uses an embedded function encapsulating system dynamics to
enhance learning stability and scalability. Case studies based on IEEE 33-bus
and 69-bus power networks are conducted to validate the effectiveness of the
proposed method in load restoration.

### 4. [Regional Frequency-Constrained Planning for the Optimal Sizing of Power Systems via Enhanced Input Convex Neural Networks](http://arxiv.org/pdf/2507.18102v1)

Authors: Yi Wang, Goran Strbac

Large renewable penetration has been witnessed in power systems, resulting in
reduced levels of system inertia and increasing requirements for frequency
response services. There have been plenty of studies developing
frequency-constrained models for power system security. However, most existing
literature only considers uniform frequency security, while neglecting
frequency spatial differences in different regions. To fill this gap, this
paper proposes a novel planning model for the optimal sizing problem of power
systems, capturing regional frequency security and inter-area frequency
oscillations. Specifically, regional frequency constraints are first extracted
via an enhanced input convex neural network (ICNN) and then embedded into the
original optimisation for frequency security, where a principled weight
initialisation strategy is adopted to deal with the gradient vanishing issues
of non-negative weights in traditional ICNNs and enhance its fitting ability.
An adaptive genetic algorithm with sparsity calculation and local search is
developed to separate the planning model into two stages and effectively solve
it iteratively. Case studies have been conducted on three different power
systems to verify the effectiveness of the proposed frequency-constrained
planning model in ensuring regional system security and obtaining realistic
investment decisions.

### 5. [Two-Stage TSO-DSO Services Provision Framework for Electric Vehicle Coordination](http://arxiv.org/pdf/2507.18110v1)

Authors: Yi Wang, Dawei Qiu, Fei Teng, Goran Strbac

High renewable penetration has been witnessed in power systems, resulting in
reduced system inertia and increasing requirements for frequency response
services. Electric vehicles (EVs), owing to their vehicle-to-grid (V2G)
capabilities, can provide cost-effective frequency services for transmission
system operators (TSOs). However, EVs that are inherently connected to
distribution networks may pose voltage security issues for distribution system
operators (DSOs) when supporting TSO frequency. To coordinate both TSO
frequency and DSO voltage, this paper proposes a two-stage service provision
framework for multi-EVs. At stage one, EVs participate in day-ahead TSO-DSO
interactions for frequency reserve schedules; at stage two, EVs make real-time
dispatching behaviors in distribution networks for reserve delivery while
supporting DSO voltage. Considering the potentially large EV number and
environment complexity, a decentralized operation paradigm is introduced for
real-time EV dispatches at stage two, while a communication-efficient
reinforcement learning (RL) algorithm is proposed to reduce the communication
overhead during large-scale multi-agent RL training without compromising policy
performance. Case studies are carried out on a 6-bus transmission and 33-bus
distribution network as well as a 69-bus distribution network to evaluate the
effectiveness and scalability of the proposed method in enabling EVs for
frequency service and voltage support.

### 6. [Data-Driven Model Order Reduction for Continuous- and Discrete-Time Nonlinear Systems](http://arxiv.org/pdf/2507.18131v1)

Authors: Behrad Samari, Henrik Sandberg, Karl H. Johansson, Abolfazl Lavaei

Model order reduction simplifies high-dimensional dynamical systems by
deriving lower-dimensional models that preserve essential system
characteristics. These techniques are crucial to controller design for complex
systems while significantly reducing computational costs. Nevertheless,
constructing effective reduced-order models (ROMs) poses considerable
challenges, particularly for dynamical systems characterized by highly
nonlinear terms. These challenges are further exacerbated when the actual
system model is unavailable, a scenario frequently encountered in real-world
applications. In this work, we propose a data-driven framework for the
construction of ROMs for both continuous- and discrete-time nonlinear dynamical
systems with unknown mathematical models. By leveraging two sets of data
collected from the system, referred to as two input-state trajectories, we
first construct a data-based closed-loop representation of the system. We then
establish a similarity relation between the output trajectories of the original
system and those of its data-driven ROM employing the notion of simulation
functions (SFs), thereby enabling a formal characterization of their closeness.
To achieve this, we propose data-dependent semidefinite programs as sufficient
conditions to simultaneously construct both ROMs and SFs, while offering
correctness guarantees. We demonstrate that the obtained data-driven ROMs can
be employed for synthesizing controllers that ensure the unknown system
satisfies high-level logic properties. This is accomplished by first designing
controllers for the data-driven ROMs and then translating the results back to
the original system through an interface function. We evaluate the efficacy of
our data-driven findings through four benchmark case studies involving unknown
dynamics with highly nonlinear terms.

### 7. [Data-Driven Incremental GAS Certificate of Nonlinear Homogeneous Networks: A Formal Modular Approach](http://arxiv.org/pdf/2507.18141v1)

Authors: Mahdieh Zaker, David Angeli, Abolfazl Lavaei

This work focuses on a compositional data-driven approach to verify
incremental global asymptotic stability (delta-GAS) over interconnected
homogeneous networks of degree one with unknown mathematical dynamics. Our
proposed approach leverages the concept of incremental input-to-state stability
(delta-ISS) of subsystems, characterized by delta-ISS Lyapunov functions. To
implement our data-driven scheme, we initially reframe the delta-ISS Lyapunov
conditions as a robust optimization program (ROP). However, due to the presence
of unknown subsystem dynamics in the ROP constraints, we develop a scenario
optimization program (SOP) by gathering data from trajectories of each unknown
subsystem. We solve the SOP and construct a delta-ISS Lyapunov function for
each subsystem with unknown dynamics. We then leverage a small-gain
compositional condition to facilitate the construction of an incremental
Lyapunov function for an unknown interconnected network with unknown dynamics
based on its data-driven delta-ISS Lyapunov functions of individual subsystems,
while providing correctness guarantees. We demonstrate that our data-driven
compositional approach aligns sample complexity with subsystem granularity,
resulting in a linear increase in required data as the number of subsystems
rises. In contrast, the existing monolithic approach in the literature exhibits
exponential growth in sample complexity with increasing number of subsystems,
rendering it impractical for real-world applications. To validate the
effectiveness of our compositional data-driven approach, we apply it to an
unknown nonlinear homogeneous network of degree one, comprising 10000
subsystems. By gathering data from each unknown subsystem, we demonstrate that
the interconnected network is delta-GAS with a correctness guarantee.

### 8. [Unit Commitment Framework for Nuclear Reactors with Reactivity Decline](http://arxiv.org/pdf/2507.18150v1)

Authors: Shiny Choudhury, Michael Davidson, George Tynan

Nuclear reactors are often modeled as inflexible, baseload generators with
fixed downtimes and restrictive ramping limits. In practice, however, a
reactor's operational flexibility is closely tied to it's fuel cycle stage and
the associated reactivity margin. A key physical constraint to power
maneuverability is xenon poisoning, caused by an increase in neutron absorbing
xenon concentration following a power ramp down. This can delay or even prevent
subsequent power ramp up due to suppressed core reactivity. Additionally, if a
reactor is shutdown during periods of low reactivity, restart times can vary
significantly due to these xenon transients, leading to longer downtimes. This
work introduces a physics informed, metaheuristic modeling approach that embeds
fuel cycle dynamics directly with a unit commitment (UC) framework. The
framework tracks reactivity margin, dynamically activates xenon related
constraints, and endogenously implements refueling outages based on the core
conditions. By capturing intra-cycle reactivity evolution and the conditional
onset of xenon poisoning, the formulation allows for operation dependent
nuclear dispatch that reflects both regulatory limits and physical behavior.
When applied to a representative reactor fleet operating in distinct modes of
operation -- ranging from baseload to part load -- the framework reveals that
flexible operation can slow reactivity degradation and extend fuel cycles. The
results show that fuel cycle aware flexibility modeling is critical for
accurate scheduling of nuclear reactors and offers a tractable pathway to
integrate nuclear power in energy system models.

### 9. [Stability Constrained Voltage Control in Distribution Grids with Arbitrary Communication Infrastructure](http://arxiv.org/pdf/2507.18158v1)

Authors: Zhenyi Yuan, Jie Feng, Yuanyuan Shi, Jorge Cortés

We consider the problem of designing learning-based reactive power
controllers that perform voltage regulation in distribution grids while
ensuring closed-loop system stability. In contrast to existing methods, where
the provably stable controllers are restricted to be decentralized, we propose
a unified design framework that enables the controllers to take advantage of an
arbitrary communication infrastructure on top of the physical power network.
This allows the controllers to incorporate information beyond their local bus,
covering existing methods as a special case and leading to less conservative
constraints on the controller design. We then provide a design procedure to
construct input convex neural network (ICNN) based controllers that satisfy the
identified stability constraints by design under arbitrary communication
scenarios, and train these controllers using supervised learning. Simulation
results on the the University of California, San Diego (UCSD) microgrid testbed
illustrate the effectiveness of the framework and highlight the role of
communication in improving control performance.

### 10. [Designing efficient interventions for pre-disease states using control theory](http://arxiv.org/pdf/2507.18269v1)

Authors: Makito Oku

To extend healthy life expectancy in an aging society, it is crucial to
prevent various diseases at pre-disease states. Although dynamical network
biomarker theory has been developed for pre-disease detection, mathematical
frameworks for pre-disease treatment have not been well established. Here I
propose a control theory-based approach for pre-disease treatment, named Markov
chain sparse control (MCSC), where time evolution of a probability distribution
on a Markov chain is described as a discrete-time linear system. By designing a
sparse controller, a few candidate states for intervention are identified. The
validity of MCSC is demonstrated using numerical simulations and real-data
analysis.

### Machine Learning (Statistics Category)

### 1. [Learning graphons from data: Random walks, transfer operators, and spectral clustering](http://arxiv.org/pdf/2507.18147v1)

Authors: Stefan Klus, Jason J. Bramburger

Many signals evolve in time as a stochastic process, randomly switching
between states over discretely sampled time points. Here we make an explicit
link between the underlying stochastic process of a signal that can take on a
bounded continuum of values and a random walk process on a graphon. Graphons
are infinite-dimensional objects that represent the limit of convergent
sequences of graphs whose size tends to infinity. We introduce transfer
operators, such as the Koopman and Perron--Frobenius operators, associated with
random walk processes on graphons and then illustrate how these operators can
be estimated from signal data and how their eigenvalues and eigenfunctions can
be used for detecting clusters, thereby extending conventional spectral
clustering methods from graphs to graphons. Furthermore, we show that it is
also possible to reconstruct transition probability densities and, if the
random walk process is reversible, the graphon itself using only the signal.
The resulting data-driven methods are applied to a variety of synthetic and
real-world signals, including daily average temperatures and stock index
values.

### 2. [Efficient Uncertainty in LLMs through Evidential Knowledge Distillation](http://arxiv.org/pdf/2507.18366v1)

Authors: Lakshmana Sri Harsha Nemani, P. K. Srijith, Tomasz Kuśmierczyk

Accurate uncertainty quantification remains a key challenge for standard
LLMs, prompting the adoption of Bayesian and ensemble-based methods. However,
such methods typically necessitate computationally expensive sampling,
involving multiple forward passes to effectively estimate predictive
uncertainty.
  In this paper, we introduce a novel approach enabling efficient and effective
uncertainty estimation in LLMs without sacrificing performance. Specifically,
we distill uncertainty-aware teacher models - originally requiring multiple
forward passes - into compact student models sharing the same architecture but
fine-tuned using Low-Rank Adaptation (LoRA). We compare two distinct
distillation strategies: one in which the student employs traditional
softmax-based outputs, and another in which the student leverages
Dirichlet-distributed outputs to explicitly model epistemic uncertainty via
evidential learning.
  Empirical evaluations on classification datasets demonstrate that such
students can achieve comparable or superior predictive and uncertainty
quantification performance relative to their teacher models, while critically
requiring only a single forward pass. To our knowledge, this is the first
demonstration that immediate and robust uncertainty quantification can be
achieved in LLMs through evidential distillation.

### 3. [DriftMoE: A Mixture of Experts Approach to Handle Concept Drifts](http://arxiv.org/pdf/2507.18464v1)

Authors: Miguel Aspis, Sebastián A. Cajas Ordónez, Andrés L. Suárez-Cetrulo, Ricardo Simón Carbajo

Learning from non-stationary data streams subject to concept drift requires
models that can adapt on-the-fly while remaining resource-efficient. Existing
adaptive ensemble methods often rely on coarse-grained adaptation mechanisms or
simple voting schemes that fail to optimally leverage specialized knowledge.
This paper introduces DriftMoE, an online Mixture-of-Experts (MoE) architecture
that addresses these limitations through a novel co-training framework.
DriftMoE features a compact neural router that is co-trained alongside a pool
of incremental Hoeffding tree experts. The key innovation lies in a symbiotic
learning loop that enables expert specialization: the router selects the most
suitable expert for prediction, the relevant experts update incrementally with
the true label, and the router refines its parameters using a multi-hot
correctness mask that reinforces every accurate expert. This feedback loop
provides the router with a clear training signal while accelerating expert
specialization. We evaluate DriftMoE's performance across nine state-of-the-art
data stream learning benchmarks spanning abrupt, gradual, and real-world drifts
testing two distinct configurations: one where experts specialize on data
regimes (multi-class variant), and another where they focus on single-class
specialization (task-based variant). Our results demonstrate that DriftMoE
achieves competitive results with state-of-the-art stream learning adaptive
ensembles, offering a principled and efficient approach to concept drift
adaptation. All code, data pipelines, and reproducibility scripts are available
in our public GitHub repository: https://github.com/miguel-ceadar/drift-moe.

### 4. [Neural Tangent Kernels and Fisher Information Matrices for Simple ReLU Networks with Random Hidden Weights](http://arxiv.org/pdf/2507.18555v1)

Authors: Jun'ichi Takeuchia, Yoshinari Takeishia, Noboru Muratab, Kazushi Mimurac, Ka Long Keith Hod, Hiroshi Nagaoka

Fisher information matrices and neural tangent kernels (NTK) for 2-layer ReLU
networks with random hidden weight are argued. We discuss the relation between
both notions as a linear transformation and show that spectral decomposition of
NTK with concrete forms of eigenfunctions with major eigenvalues. We also
obtain an approximation formula of the functions presented by the 2-layer
neural networks.

### 5. [A Two-armed Bandit Framework for A/B Testing](http://arxiv.org/pdf/2507.18118v1)

Authors: Jinjuan Wang, Qianglin Wen, Yu Zhang, Xiaodong Yan, Chengchun Shi

A/B testing is widely used in modern technology companies for policy
evaluation and product deployment, with the goal of comparing the outcomes
under a newly-developed policy against a standard control. Various causal
inference and reinforcement learning methods developed in the literature are
applicable to A/B testing. This paper introduces a two-armed bandit framework
designed to improve the power of existing approaches. The proposed procedure
consists of three main steps: (i) employing doubly robust estimation to
generate pseudo-outcomes, (ii) utilizing a two-armed bandit framework to
construct the test statistic, and (iii) applying a permutation-based method to
compute the $p$-value. We demonstrate the efficacy of the proposed method
through asymptotic theories, numerical experiments and real-world data from a
ridesharing company, showing its superior performance in comparison to existing
methods.

### 6. [Trek-Based Parameter Identification for Linear Causal Models With Arbitrarily Structured Latent Variables](http://arxiv.org/pdf/2507.18170v1)

Authors: Nils Sturma, Mathias Drton

We develop a criterion to certify whether causal effects are identifiable in
linear structural equation models with latent variables. Linear structural
equation models correspond to directed graphs whose nodes represent the random
variables of interest and whose edges are weighted with linear coefficients
that correspond to direct causal effects. In contrast to previous
identification methods, we do not restrict ourselves to settings where the
latent variables constitute independent latent factors (i.e., to source nodes
in the graphical representation of the model). Our novel latent-subgraph
criterion is a purely graphical condition that is sufficient for
identifiability of causal effects by rational formulas in the covariance
matrix. To check the latent-subgraph criterion, we provide a sound and complete
algorithm that operates by solving an integer linear program. While it targets
effects involving observed variables, our new criterion is also useful for
identifying effects between latent variables, as it allows one to transform the
given model into a simpler measurement model for which other existing tools
become applicable.

### 7. [On Reconstructing Training Data From Bayesian Posteriors and Trained Models](http://arxiv.org/pdf/2507.18372v1)

Authors: George Wynne

Publicly releasing the specification of a model with its trained parameters
means an adversary can attempt to reconstruct information about the training
data via training data reconstruction attacks, a major vulnerability of modern
machine learning methods. This paper makes three primary contributions:
establishing a mathematical framework to express the problem, characterising
the features of the training data that are vulnerable via a maximum mean
discrepancy equivalance and outlining a score matching framework for
reconstructing data in both Bayesian and non-Bayesian models, the former is a
first in the literature.

### 8. [Euclidean Distance Deflation Under High-Dimensional Heteroskedastic Noise](http://arxiv.org/pdf/2507.18520v1)

Authors: Keyi Li, Yuval Kluger, Boris Landa

Pairwise Euclidean distance calculation is a fundamental step in many machine
learning and data analysis algorithms. In real-world applications, however,
these distances are frequently distorted by heteroskedastic
noise$\unicode{x2014}$a prevalent form of inhomogeneous corruption
characterized by variable noise magnitudes across data observations. Such noise
inflates the computed distances in a nontrivial way, leading to
misrepresentations of the underlying data geometry. In this work, we address
the tasks of estimating the noise magnitudes per observation and correcting the
pairwise Euclidean distances under heteroskedastic noise. Perhaps surprisingly,
we show that in general high-dimensional settings and without assuming prior
knowledge on the clean data structure or noise distribution, both tasks can be
performed reliably, even when the noise levels vary considerably. Specifically,
we develop a principled, hyperparameter-free approach that jointly estimates
the noise magnitudes and corrects the distances. We provide theoretical
guarantees for our approach, establishing probabilistic bounds on the
estimation errors of both noise magnitudes and distances. These bounds,
measured in the normalized $\ell_1$ norm, converge to zero at polynomial rates
as both feature dimension and dataset size increase. Experiments on synthetic
datasets demonstrate that our method accurately estimates distances in
challenging regimes, significantly improving the robustness of subsequent
distance-based computations. Notably, when applied to single-cell RNA
sequencing data, our method yields noise magnitude estimates consistent with an
established prototypical model, enabling accurate nearest neighbor
identification that is fundamental to many downstream analyses.

### 9. [Beyond Internal Data: Constructing Complete Datasets for Fairness Testing](http://arxiv.org/pdf/2507.18561v1)

Authors: Varsha Ramineni, Hossein A. Rahmani, Emine Yilmaz, David Barber

As AI becomes prevalent in high-risk domains and decision-making, it is
essential to test for potential harms and biases. This urgency is reflected by
the global emergence of AI regulations that emphasise fairness and adequate
testing, with some mandating independent bias audits. However, procuring the
necessary data for fairness testing remains a significant challenge.
Particularly in industry settings, legal and privacy concerns restrict the
collection of demographic data required to assess group disparities, and
auditors face practical and cultural challenges in gaining access to data.
Further, internal historical datasets are often insufficiently representative
to identify real-world biases. This work focuses on evaluating classifier
fairness when complete datasets including demographics are inaccessible. We
propose leveraging separate overlapping datasets to construct complete
synthetic data that includes demographic information and accurately reflects
the underlying relationships between protected attributes and model features.
We validate the fidelity of the synthetic data by comparing it to real data,
and empirically demonstrate that fairness metrics derived from testing on such
synthetic data are consistent with those obtained from real data. This work,
therefore, offers a path to overcome real-world data scarcity for fairness
testing, enabling independent, model-agnostic evaluation of fairness, and
serving as a viable substitute where real data is limited.

### 10. [Large-scale entity resolution via microclustering Ewens--Pitman random partitions](http://arxiv.org/pdf/2507.18101v1)

Authors: Mario Beraha, Stefano Favaro

We introduce the microclustering Ewens--Pitman model for random partitions,
obtained by scaling the strength parameter of the Ewens--Pitman model linearly
with the sample size. The resulting random partition is shown to have the
microclustering property, namely: the size of the largest cluster grows
sub-linearly with the sample size, while the number of clusters grows linearly.
By leveraging the interplay between the Ewens--Pitman random partition with the
Pitman--Yor process, we develop efficient variational inference schemes for
posterior computation in entity resolution. Our approach achieves a speed-up of
three orders of magnitude over existing Bayesian methods for entity resolution,
while maintaining competitive empirical performance.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-25 PST.

### 1. [Threats to scientific software from over-reliance on AI code assistants](https://www.nature.com/articles/s43588-025-00845-2)

Authors: Gabrielle O’Brien

### 2. [Training a high-performance retinal foundation model with half-the-data and 400 times less compute](https://www.nature.com/articles/s41467-025-62123-z)

Authors: Justin Engelmann et al.

### 3. [Contextual semantics graph attention network model for entity resolution](https://www.nature.com/articles/s41598-025-11932-9)

Authors: Xiaojun Li et al.

### 4. [Classification of musculoskeletal pain using machine learning](https://www.nature.com/articles/s41598-025-12049-9)

Authors: Dalia Mohamed Fouad et al.

### 5. [An adaptive spatiotemporal dynamic graph convolutional network for traffic prediction](https://www.nature.com/articles/s41598-025-12261-7)

Authors: Zhiguo Xiao et al.

### 6. [Deep representation learning using layer-wise VICReg losses](https://www.nature.com/articles/s41598-025-08504-2)

Authors: Joy Datta et al.

### 7. [Integrating physics and topology in neural networks for learning rigid body dynamics](https://www.nature.com/articles/s41467-025-62250-7)

Authors: Amaury Wei et al.

