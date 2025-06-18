# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-06-17 17:02:04.260177 PST.

### Artificial Intelligence

### 1. [MAGIC: Multi-Agent Argumentation and Grammar Integrated Critiquer](http://arxiv.org/pdf/2506.13037v1)

Authors: Joaquin Jordan, Xavier Yin, Melissa Fabros, Gireeja Ranade, Narges Norouzi

Automated Essay Scoring (AES) and Automatic Essay Feedback (AEF) systems aim
to reduce the workload of human raters in educational assessment. However, most
existing systems prioritize numeric scoring accuracy over the quality of
feedback. This paper presents Multi-Agent Argumentation and Grammar Integrated
Critiquer (MAGIC), a framework that uses multiple specialized agents to
evaluate distinct writing aspects to both predict holistic scores and produce
detailed, rubric-aligned feedback. To support evaluation, we curated a novel
dataset of past GRE practice test essays with expert-evaluated scores and
feedback. MAGIC outperforms baseline models in both essay scoring , as measured
by Quadratic Weighted Kappa (QWK). We find that despite the improvement in QWK,
there are opportunities for future work in aligning LLM-generated feedback to
human preferences.

### 2. [Discerning What Matters: A Multi-Dimensional Assessment of Moral Competence in LLMs](http://arxiv.org/pdf/2506.13082v1)

Authors: Daniel Kilov, Caroline Hendy, Secil Yanik Guyot, Aaron J. Snoswell, Seth Lazar

Moral competence is the ability to act in accordance with moral principles.
As large language models (LLMs) are increasingly deployed in situations
demanding moral competence, there is increasing interest in evaluating this
ability empirically. We review existing literature and identify three
significant shortcoming: (i) Over-reliance on prepackaged moral scenarios with
explicitly highlighted moral features; (ii) Focus on verdict prediction rather
than moral reasoning; and (iii) Inadequate testing of models' (in)ability to
recognize when additional information is needed. Grounded in philosophical
research on moral skill, we then introduce a novel method for assessing moral
competence in LLMs. Our approach moves beyond simple verdict comparisons to
evaluate five dimensions of moral competence: identifying morally relevant
features, weighting their importance, assigning moral reasons to these
features, synthesizing coherent moral judgments, and recognizing information
gaps. We conduct two experiments comparing six leading LLMs against non-expert
humans and professional philosophers. In our first experiment using ethical
vignettes standard to existing work, LLMs generally outperformed non-expert
humans across multiple dimensions of moral reasoning. However, our second
experiment, featuring novel scenarios designed to test moral sensitivity by
embedding relevant features among irrelevant details, revealed a striking
reversal: several LLMs performed significantly worse than humans. Our findings
suggest that current evaluations may substantially overestimate LLMs' moral
reasoning capabilities by eliminating the task of discerning moral relevance
from noisy information, which we take to be a prerequisite for genuine moral
skill. This work provides a more nuanced framework for assessing AI moral
competence and highlights important directions for improving moral competence
in advanced AI systems.

### 3. [Towards Explaining Monte-Carlo Tree Search by Using Its Enhancements](http://arxiv.org/pdf/2506.13223v1)

Authors: Jakub Kowalski, Mark H. M. Winands, Maksymilian Wiśniewski, Stanisław Reda, Anna Wilbik

Typically, research on Explainable Artificial Intelligence (XAI) focuses on
black-box models within the context of a general policy in a known, specific
domain. This paper advocates for the need for knowledge-agnostic explainability
applied to the subfield of XAI called Explainable Search, which focuses on
explaining the choices made by intelligent search techniques. It proposes
Monte-Carlo Tree Search (MCTS) enhancements as a solution to obtaining
additional data and providing higher-quality explanations while remaining
knowledge-free, and analyzes the most popular enhancements in terms of the
specific types of explainability they introduce. So far, no other research has
considered the explainability of MCTS enhancements. We present a
proof-of-concept that demonstrates the advantages of utilizing enhancements.

### 4. [Generalized Proof-Number Monte-Carlo Tree Search](http://arxiv.org/pdf/2506.13249v1)

Authors: Jakub Kowalski, Dennis J. N. J. Soemers, Szymon Kosakowski, Mark H. M. Winands

This paper presents Generalized Proof-Number Monte-Carlo Tree Search: a
generalization of recently proposed combinations of Proof-Number Search (PNS)
with Monte-Carlo Tree Search (MCTS), which use (dis)proof numbers to bias
UCB1-based Selection strategies towards parts of the search that are expected
to be easily (dis)proven. We propose three core modifications of prior
combinations of PNS with MCTS. First, we track proof numbers per player. This
reduces code complexity in the sense that we no longer need disproof numbers,
and generalizes the technique to be applicable to games with more than two
players. Second, we propose and extensively evaluate different methods of using
proof numbers to bias the selection strategy, achieving strong performance with
strategies that are simpler to implement and compute. Third, we merge our
technique with Score Bounded MCTS, enabling the algorithm to prove and leverage
upper and lower bounds on scores - as opposed to only proving wins or not-wins.
Experiments demonstrate substantial performance increases, reaching the range
of 80% for 8 out of the 11 tested board games.

### 5. [Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks](http://arxiv.org/pdf/2506.13276v1)

Authors: Yuefei Lyu, Chaozhuo Li, Xi Zhang, Tianle Zhang

Text-attributed graphs (TAGs) integrate textual data with graph structures,
providing valuable insights in applications such as social network analysis and
recommendation systems. Graph Neural Networks (GNNs) effectively capture both
topological structure and textual information in TAGs but are vulnerable to
adversarial attacks. Existing graph injection attack (GIA) methods assume that
attackers can directly manipulate the embedding layer, producing
non-explainable node embeddings. Furthermore, the effectiveness of these
attacks often relies on surrogate models with high training costs. Thus, this
paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs.
Our approach leverages large language models (LLMs) to generate interpretable
text-level node attributes directly, ensuring attacks remain feasible in
real-world scenarios. We design strategies for LLM prompting that balance
exploration and reliability to guide text generation, and propose a similarity
assessment method to evaluate attack text effectiveness in disrupting graph
homophily. This method efficiently perturbs the target node with minimal
training costs in a strict black-box setting, ensuring a text-level graph
injection attack for TAGs. Experiments on real-world TAG datasets validate the
superior performance of ATAG-LLM compared to state-of-the-art embedding-level
and text-level attack methods.

### 6. [A Technical Study into Small Reasoning Language Models](http://arxiv.org/pdf/2506.13404v1)

Authors: Xialie Zhuang, Peixian Ma, Zhikai Jia, Zheng Cao, Shiwei Liu

The ongoing evolution of language models has led to the development of
large-scale architectures that demonstrate exceptional performance across a
wide range of tasks. However, these models come with significant computational
and energy demands, as well as potential privacy implications. In this context,
Small Reasoning Language Models (SRLMs) with approximately 0.5 billion
parameters present a compelling alternative due to their remarkable
computational efficiency and cost effectiveness, particularly in
resource-constrained environments. Despite these advantages, the limited
capacity of 0.5 billion parameter models poses challenges in handling complex
tasks such as mathematical reasoning and code generation. This research
investigates various training strategies, including supervised fine-tuning
(SFT), knowledge distillation (KD), and reinforcement learning (RL), as well as
their hybrid implementations, to enhance the performance of 0.5B SRLMs. We
analyze effective methodologies to bridge the performance gap between SRLMS and
larger models and present insights into optimal training pipelines tailored for
these smaller architectures. Through extensive experimental validation and
analysis, our work aims to provide actionable recommendations for maximizing
the reasoning capabilities of 0.5B models.

### 7. [The ASP-based Nurse Scheduling System at the University of Yamanashi Hospital](http://arxiv.org/pdf/2506.13600v1)

Authors: Hidetomo Nabeshima, Mutsunori Banbara, Torsten Schaub, Takehide Soh

We present the design principles of a nurse scheduling system built using
Answer Set Programming (ASP) and successfully deployed at the University of
Yamanashi Hospital. Nurse scheduling is a complex optimization problem
requiring the reconciliation of individual nurse preferences with hospital
staffing needs across various wards. This involves balancing hard and soft
constraints and the flexibility of interactive adjustments. While extensively
studied in academia, real-world nurse scheduling presents unique challenges
that go beyond typical benchmark problems and competitions. This paper details
the practical application of ASP to address these challenges at the University
of Yamanashi Hospital, focusing on the insights gained and the advancements in
ASP technology necessary to effectively manage the complexities of real-world
deployment.

### 8. [Missing the human touch? A computational stylometry analysis of GPT-4 translations of online Chinese literature](http://arxiv.org/pdf/2506.13013v1)

Authors: Xiaofang Yao, Yong-Bin Kang, Anthony McCosker

Existing research indicates that machine translations (MTs) of literary texts
are often unsatisfactory. MTs are typically evaluated using automated metrics
and subjective human ratings, with limited focus on stylistic features.
Evidence is also limited on whether state-of-the-art large language models
(LLMs) will reshape literary translation. This study examines the stylistic
features of LLM translations, comparing GPT-4's performance to human
translations in a Chinese online literature task. Computational stylometry
analysis shows that GPT-4 translations closely align with human translations in
lexical, syntactic, and content features, suggesting that LLMs might replicate
the 'human touch' in literary translation style. These findings offer insights
into AI's impact on literary translation from a posthuman perspective, where
distinctions between machine and human translations become increasingly blurry.

### 9. [Geometric Embedding Alignment via Curvature Matching in Transfer Learning](http://arxiv.org/pdf/2506.13015v1)

Authors: Sung Moon Ko, Jaewan Lee, Sumin Lee, Soorin Yim, Kyunghoon Bae, Sehui Han

Geometrical interpretations of deep learning models offer insightful
perspectives into their underlying mathematical structures. In this work, we
introduce a novel approach that leverages differential geometry, particularly
concepts from Riemannian geometry, to integrate multiple models into a unified
transfer learning framework. By aligning the Ricci curvature of latent space of
individual models, we construct an interrelated architecture, namely Geometric
Embedding Alignment via cuRvature matching in transfer learning (GEAR), which
ensures comprehensive geometric representation across datapoints. This
framework enables the effective aggregation of knowledge from diverse sources,
thereby improving performance on target tasks. We evaluate our model on 23
molecular task pairs sourced from various domains and demonstrate significant
performance gains over existing benchmark model under both random (14.4%) and
scaffold (8.3%) data splits.

### 10. [Symmetry in Neural Network Parameter Spaces](http://arxiv.org/pdf/2506.13018v1)

Authors: Bo Zhao, Robin Walters, Rose Yu

Modern deep learning models are highly overparameterized, resulting in large
sets of parameter configurations that yield the same outputs. A significant
portion of this redundancy is explained by symmetries in the parameter
space--transformations that leave the network function unchanged. These
symmetries shape the loss landscape and constrain learning dynamics, offering a
new lens for understanding optimization, generalization, and model complexity
that complements existing theory of deep learning. This survey provides an
overview of parameter space symmetry. We summarize existing literature, uncover
connections between symmetry and learning theory, and identify gaps and
opportunities in this emerging field.

### Hardware Architecture

### 1. [Reconfigurable Digital RRAM Logic Enables In-Situ Pruning and Learning for Edge AI](http://arxiv.org/pdf/2506.13151v1)

Authors: Songqi Wang, Yue Zhang, Jia Chen, Xinyuan Zhang, Yi Li, Ning Lin, Yangu He, Jichang Yang, Yingjie Yu, Yi Li, Zhongrui Wang, Xiaojuan Qi, Han Wang

The human brain simultaneously optimizes synaptic weights and topology by
growing, pruning, and strengthening synapses while performing all computation
entirely in memory. In contrast, modern artificial-intelligence systems
separate weight optimization from topology optimization and depend on
energy-intensive von Neumann architectures. Here, we present a
software-hardware co-design that bridges this gap. On the algorithmic side, we
introduce a real-time dynamic weight-pruning strategy that monitors weight
similarity during training and removes redundancies on the fly, reducing
operations by 26.80% on MNIST and 59.94% on ModelNet10 without sacrificing
accuracy (91.44% and 77.75%, respectively). On the hardware side, we fabricate
a reconfigurable, fully digital compute-in-memory (CIM) chip based on 180 nm
one-transistor-one-resistor (1T1R) RRAM arrays. Each array embeds flexible
Boolean logic (NAND, AND, XOR, OR), enabling both convolution and similarity
evaluation inside memory and eliminating all ADC/DAC overhead. The digital
design achieves zero bit-error, reduces silicon area by 72.30% and overall
energy by 57.26% compared to analogue RRAM CIM, and lowers energy by 75.61% and
86.53% on MNIST and ModelNet10, respectively, relative to an NVIDIA RTX 4090.
Together, our co-design establishes a scalable brain-inspired paradigm for
adaptive, energy-efficient edge intelligence in the future.

### Computational Complexity

### 1. [The Word Problem for Products of Symmetric Groups](http://arxiv.org/pdf/2506.13655v1)

Authors: Hans U. Simon

The word problem for products of symmetric groups (WPPSG) is a well-known
NP-complete problem. An input instance of this problem consists of
``specification sets'' $X_1,\ldots,X_m \seq \{1,\ldots,n\}$ and a permutation
$\tau$ on $\{1,\ldots,n\}$. The sets $X_1,\ldots,X_m$ specify a subset of the
symmetric group $\cS_n$ and the question is whether the given permutation
$\tau$ is a member of this subset. We discuss three subproblems of WPPSG and
show that they can be solved efficiently. The subproblem WPPSG$_0$ is the
restriction of WPPSG to specification sets all of which are sets of consecutive
integers. The subproblem WPPSG$_1$ is the restriction of WPPSG to specification
sets which have the Consecutive Ones Property. The subproblem WPPSG$_2$ is the
restriction of WPPSG to specification sets which have what we call the Weak
Consecutive Ones Property. WPPSG$_1$ is more general than WPPSG$_0$ and
WPPSG$_2$ is more general than WPPSG$_1$. But the efficient algorithms that we
use for solving WPPSG$_1$ and WPPSG$_2$ have, as a sub-routine, the efficient
algorithm for solving WPPSG$_0$.

### 2. [Avoiding Obfuscation with Prover-Estimator Debate](http://arxiv.org/pdf/2506.13609v1)

Authors: Jonah Brown-Cohen, Geoffrey Irving, Georgios Piliouras

Training powerful AI systems to exhibit desired behaviors hinges on the
ability to provide accurate human supervision on increasingly complex tasks. A
promising approach to this problem is to amplify human judgement by leveraging
the power of two competing AIs in a debate about the correct solution to a
given problem. Prior theoretical work has provided a complexity-theoretic
formalization of AI debate, and posed the problem of designing protocols for AI
debate that guarantee the correctness of human judgements for as complex a
class of problems as possible. Recursive debates, in which debaters decompose a
complex problem into simpler subproblems, hold promise for growing the class of
problems that can be accurately judged in a debate. However, existing protocols
for recursive debate run into the obfuscated arguments problem: a dishonest
debater can use a computationally efficient strategy that forces an honest
opponent to solve a computationally intractable problem to win. We mitigate
this problem with a new recursive debate protocol that, under certain stability
assumptions, ensures that an honest debater can win with a strategy requiring
computational efficiency comparable to their opponent.

### Computational Engineering

### 1. [A modified Newmark/Newton-Raphson method with automatic differentiation for general nonlinear dynamics analysis](http://arxiv.org/pdf/2506.13226v1)

Authors: Yifan Jiang, Yuhong Jin, Lei Hou, Yi Chen, Andong Cong

The Newmark/Newton-Raphson (NNR) method is widely employed for solving
nonlinear dynamic systems. However, the current NNR method exhibits limited
applicability in complex nonlinear dynamic systems, as the acquisition of the
Jacobian matrix required for Newton iterations incurs substantial computational
costs and may even prove intractable in certain cases. To address these
limitations, we integrate automatic differentiation (AD) into the NNR method,
proposing a modified NNR method with AD (NNR-AD) to significantly improve its
capability for effectively handling complex nonlinear systems. We have
demonstrated that the NNR-AD method can directly solve dynamic systems with
complex nonlinear characteristics, and its accuracy and generality have been
rigorously validated. Furthermore, automatic differentiation significantly
simplifies the computation of Jacobian matrices for such complex nonlinear
dynamic systems. This improvement endows the NNR method with enhanced
modularity, thereby enabling convenient and effective solutions for complex
nonlinear dynamic systems.

### 2. [An Entropy-Stable/Double-Flux scheme for the multi-component compressible Navier-Stokes equations](http://arxiv.org/pdf/2506.13231v1)

Authors: Vahid Badrkhani, T. Jeremy P. Karpowsk, Christian Hasse

We present a novel combination of numerical techniques to improve the
efficiency, accuracy, and robustness of multi-component compressible flow
simulations. At the core of our approach is an Entropy-Stable formulation that
preserves kinetic energy and integrates a Double-Flux scheme tailored for
multi-component flows with variable specific heat ratios. This formulation
yields low-dissipation, oscillation-free solutions and enhances stability
compared to standard fully conservative methods. To further improve robustness,
we introduce a new hybrid dissipation strategy that blends the
Entropy-Stable/Double-Flux approach with conventional dissipation mechanisms.
We provide a rigorous proof that the resulting numerical flux satisfies a
semi-discrete entropy inequality, ensuring consistency with the second law of
thermodynamics. For time integration, we employ an explicit Runge-Kutta scheme
in combination with adaptive mesh refinement to capture local flow features
dynamically. The method is implemented within an existing compressible
Navier-Stokes solver based on OpenFOAM. Benchmark cases, including
multi-dimensional interface and shock-interface interactions, demonstrate the
effectiveness of the proposed framework. The results confirm its favorable
stability and robustness, validating the approach as a promising advancement
for high-fidelity simulations of supersonic flows.

### 3. [Constitutive Manifold Neural Networks](http://arxiv.org/pdf/2506.13648v1)

Authors: Wouter J. Schuttert, Mohammed Iqbal Abdul Rasheed, Bojana Rosić

Important material properties like thermal conductivity are often represented
as symmetric positive definite (SPD) tensors, which exhibit variability due to
inherent material heterogeneity and manufacturing uncertainties. These tensors
reside on a curved Riemannian manifold, and accurately modeling their
stochastic nature requires preserving both their symmetric positive definite
properties and spatial symmetries. To achieve this, uncertainties are
parametrized into scaling (magnitude) and rotation (orientation) components,
modeled as independent random variables on a manifold structure derived from
the maximum entropy principle. The propagation of such stochastic tensors
through physics-based simulations necessitates computationally efficient
surrogate models. However, traditional multi-layer perceptron (MLP)
architectures are not well-suited for SPD tensors, as directly inputting their
components fails to preserve their geometric properties, often leading to
suboptimal results. To address this, we introduce Constitutive Manifold Neural
Networks (CMNN). This approach introduces a preprocessing layer by mapping the
SPD tensor from the curved manifold to the local tangent, a flat vector space,
creating an information preserving map for input to the hidden layers of the
neural networks. A case study on a steady-state heat conduction problem with
stochastic anisotropic conductivity demonstrates that geometry-preserving
preprocessing, such as logarithmic maps for scaling data, significantly
improves learning performance over conventional MLPs. These findings underscore
the importance of manifold-aware techniques when working with tensor-valued
data in engineering applications.

### 4. [Kolmogorov-Arnold Network for Gene Regulatory Network Inference](http://arxiv.org/pdf/2506.13740v1)

Authors: Tsz Pan Tong, Aoran Wang, George Panagopoulos, Jun Pang

Gene regulation is central to understanding cellular processes and
development, potentially leading to the discovery of new treatments for
diseases and personalized medicine. Inferring gene regulatory networks (GRNs)
from single-cell RNA sequencing (scRNA-seq) data presents significant
challenges due to its high dimensionality and complexity. Existing tree-based
models, such as GENIE3 and GRNBOOST2, demonstrated scalability and
explainability in GRN inference, but they cannot distinguish regulation types
nor effectively capture continuous cellular dynamics. In this paper, we
introduce scKAN, a novel model that employs a Kolmogorov-Arnold network (KAN)
with explainable AI to infer GRNs from scRNA-seq data. By modeling gene
expression as differentiable functions matching the smooth nature of cellular
dynamics, scKAN can accurately and precisely detect activation and inhibition
regulations through explainable AI and geometric tools. We conducted extensive
experiments on the BEELINE benchmark, and scKAN surpasses and improves the
leading signed GRN inference models ranging from 5.40\% to 28.37\% in AUROC and
from 1.97\% to 40.45\% in AUPRC. These results highlight the potential of scKAN
in capturing the underlying biological processes in gene regulation without
prior knowledge of the graph structure.

### Computational Geometry

### 1. [FPT Constant Approximation Algorithms for Colorful Sum of Radii](http://arxiv.org/pdf/2506.13191v1)

Authors: Shuilian Liu, Gregory Gutin, Yicheng Xu, Yong Zhang

We study the colorful sum of radii problem, where the input is a point set
$P$ partitioned into classes $P_1, P_2, \dots, P_\omega$, along with per-class
outlier bounds $m_1, m_2, \dots, m_\omega$, summing to $m$. The goal is to
select a subset $\mathcal{C} \subseteq P$ of $k$ centers and assign points to
centers in $\mathcal{C}$, allowing up to $m_i$ unassigned points (outliers)
from each class $P_i$, while minimizing the sum of cluster radii. The radius of
a cluster is defined as the maximum distance from any point in the cluster to
its center. The classical (non-colorful) version of the sum of radii problem is
known to be NP-hard, even on weighted planar graphs. The colorful sum of radii
is introduced by Chekuri et al. (2022), who provide an $O(\log
\omega)$-approximation algorithm. In this paper, we present the first
constant-factor approximation algorithms for the colorful sum of radii running
in FPT (fixed-parameter tractable) time. Our contributions are twofold: We
design an iterative covering algorithm that achieves a
$(2+\varepsilon)$-approximation with running time exponential in both $k$ and
$m$; We further develop a $(7+\varepsilon)$-approximation algorithm by
leveraging a colorful $k$-center subroutine, improving the running time by
removing the exponential dependency on $m$.

### 2. [Volumetric Functional Maps](http://arxiv.org/pdf/2506.13212v1)

Authors: Filippo Maggioli, Marco Livesu, Simone Melzi

The computation of volumetric correspondences between 3D shapes has great
potential for medical and industrial applications. In this work, we pave the
way for spectral volume mapping, extending for the first time the functional
maps framework from the surface setting to the volumetric domain. We show that
the eigenfunctions of the volumetric Laplace operator define a functional space
that is suitable for high-quality signal transfer. We also experiment with
various techniques that edit this functional space, porting them from the
surface to the volume setting. We validate our method on novel volumetric
datasets and on tetrahedralizations of well established surface datasets, also
showcasing practical applications involving both discrete and continuous signal
mapping, for segmentation transfer, mesh connectivity transfer and solid
texturing. Last but not least, we show that considering the volumetric spectrum
greatly improves the accuracy for classical shape matching tasks among
surfaces, consistently outperforming existing surface-only spectral methods.

### 3. [Covering radii of $3$-zonotopes and the shifted Lonely Runner Conjecture](http://arxiv.org/pdf/2506.13379v1)

Authors: David Alcántara, Francisco Criado, Francisco Santos

We show that the shifted Lonely Runner Conjecture (sLRC) holds for 5 runners.
We also determine that there are exactly 3 primitive tight instances of the
conjecture, only two of which are tight for the non-shifted conjecture (LRC).
Our proof is computational, relying on a rephrasing of the sLRC in terms of
covering radii of certain zonotopes (Henze and Malikiosis, 2017), and on an
upper bound for the (integer) velocities to be checked (Malikiosis, Santos and
Schymura, 2024+).
  As a tool for the proof, we devise an algorithm for bounding the covering
radius of rational lattice polytopes, based on constructing dyadic fundamental
domains.

### 4. [Largest dyadic dual VC-dimension of non-piercing families](http://arxiv.org/pdf/2506.13606v1)

Authors: Xinqi Huang, Yuzhen Qi, Mingyuan Rong, Zixiang Xu

The dyadic dual VC-dimension of a set system \( \mathcal{F} \) is the largest
integer \( \ell \) such that there exist \( \ell \) sets \( F_1, F_{2}, \dots,
F_\ell \in \mathcal{F} \), where every pair \( \{i, j\} \in \binom{[\ell]}{2}
\) is witnessed by an element \( a_{i,j} \in F_i \cap F_j \) that does not
belong to any other set \( F_k \) with \( k \in [\ell] \setminus \{i, j\} \).
In this paper, we determine the largest dyadic dual VC-dimension of a
non-piercing family is exactly $4$, providing a rare example where the maximum
of this parameter can be determined for a natural family arising from geometry.
As an application, we give a short and direct proof that the transversal number
\( \tau(\mathcal{F}) \) of any non-piercing family is at most
\(C\nu(\mathcal{F})^9 \), where \( \nu(\mathcal{F}) \) is the matching number
and $C$ is a constant. This improves a recent result of P\'{a}lv\"{o}lgyi and
Z\'{o}lomy.

### 5. [No-dimensional Tverberg-type problems](http://arxiv.org/pdf/2506.13451v1)

Authors: Alexander Polyanskii

Recently, Adiprasito et al. have initiated the study of the so-called
no-dimensional Tverberg problem. This problem can be informally stated as
follows: Given $n\geq k$, partition an $n$-point set in Euclidean space into
$k$ parts such that their convex hulls intersect a ball of relatively small
radius.
  In this survey, we aim to present the recent progress towards solving the
no-dimensional Tverberg problem and new open questions arising in its context.
Also, we discuss the colorful variation of this problem and its algorithmic
aspects, particularly focusing on the case when each part of a partition
contains exactly 2 points. The latter turns out to be related to the following
no-dimensional Tverberg-type problem of Huemer et al.: For an even set of
points in Euclidean space, find a perfect matching such that the balls with
diameters induced by its edges intersect.

### 6. [Persistent Homology of Music Network with Three Different Distances](http://arxiv.org/pdf/2506.13595v1)

Authors: Eunwoo Heo, Byeongchan Choi, Myung ock Kim, Mai Lan Tran, Jae-Hun Jung

Persistent homology has been widely used to discover hidden topological
structures in data across various applications, including music data. To apply
persistent homology, a distance or metric must be defined between points in a
point cloud or between nodes in a graph network. These definitions are not
unique and depend on the specific objectives of a given problem. In other
words, selecting different metric definitions allows for multiple topological
inferences. In this work, we focus on applying persistent homology to music
graph with predefined weights. We examine three distinct distance definitions
based on edge-wise pathways and demonstrate how these definitions affect
persistent barcodes, persistence diagrams, and birth/death edges. We found that
there exist inclusion relations in one-dimensional persistent homology
reflected on persistence barcode and diagram among these three distance
definitions. We verified these findings using real music data.

### Computation and Language

### 1. [CFBenchmark-MM: Chinese Financial Assistant Benchmark for Multimodal Large Language Model](http://arxiv.org/pdf/2506.13055v1)

Authors: Jiangtong Li, Yiyun Zhu, Dawei Cheng, Zhijun Ding, Changjun Jiang

Multimodal Large Language Models (MLLMs) have rapidly evolved with the growth
of Large Language Models (LLMs) and are now applied in various fields. In
finance, the integration of diverse modalities such as text, charts, and tables
is crucial for accurate and efficient decision-making. Therefore, an effective
evaluation system that incorporates these data types is essential for advancing
financial application. In this paper, we introduce CFBenchmark-MM, a Chinese
multimodal financial benchmark with over 9,000 image-question pairs featuring
tables, histogram charts, line charts, pie charts, and structural diagrams.
Additionally, we develop a staged evaluation system to assess MLLMs in handling
multimodal information by providing different visual content step by step.
Despite MLLMs having inherent financial knowledge, experimental results still
show limited efficiency and robustness in handling multimodal financial
context. Further analysis on incorrect responses reveals the misinterpretation
of visual content and the misunderstanding of financial concepts are the
primary issues. Our research validates the significant, yet underexploited,
potential of MLLMs in financial analysis, highlighting the need for further
development and domain-specific optimization to encourage the enhanced use in
financial domain.

### 2. [FinLMM-R1: Enhancing Financial Reasoning in LMM through Scalable Data and Reward Design](http://arxiv.org/pdf/2506.13066v1)

Authors: Kai Lan, Jiayong Zhu, Jiangtong Li, Dawei Cheng, Guang Chen, Changjun Jiang

Large Multimodal Models (LMMs) demonstrate significant cross-modal reasoning
capabilities. However, financial applications face challenges due to the lack
of high-quality multimodal reasoning datasets and the inefficiency of existing
training paradigms for reasoning enhancement. To address these issues, we
propose an integrated framework, FinLMM-R1, combining an automated and scalable
pipeline for data construction with enhanced training strategies to improve the
multimodal reasoning of LMM. The Automated and Scalable Pipeline (ASP) resolves
textual-visual misalignment in financial reports through a separate paradigm of
question-answer generation and image-question alignment, ensuring data
integrity and extraction efficiency. Through ASP, we collect 89,378 aligned
image-question pairs from 23,397 financial reports, covering tasks such as
arithmetic reasoning, statistics reasoning, financial explanation, and
financial knowledge. Moreover, we introduce the Thinking with Adversarial
Reward in LMM (TAR-LMM), extending the prior two-stage training framework [1]
with additional reward mechanisms. In the first stage, we focus on text-only
tasks with format and accuracy rewards to guide the model in generating
well-structured thinking contents. In the second stage, we construct
multi-image contrastive samples with additional reward components including
image selection, thinking content length, and adversarial reward to jointly
optimize the LMM across visual perception, reasoning efficiency, and logical
coherence. Extensive experiments on 7 benchmarks show ASP-derived dataset and
training framework significantly improve answer accuracy and reasoning depth
over existing reasoning LMMs in both general and financial multimodal contexts.

### 3. [CMU's IWSLT 2025 Simultaneous Speech Translation System](http://arxiv.org/pdf/2506.13143v1)

Authors: Siqi Ouyang, Xi Xu, Lei Li

This paper presents CMU's submission to the IWSLT 2025 Simultaneous Speech
Translation (SST) task for translating unsegmented English speech into Chinese
and German text in a streaming manner. Our end-to-end speech-to-text system
integrates a chunkwise causal Wav2Vec 2.0 speech encoder, an adapter, and the
Qwen2.5-7B-Instruct as the decoder. We use a two-stage simultaneous training
procedure on robust speech segments curated from LibriSpeech, CommonVoice, and
VoxPopuli datasets, utilizing standard cross-entropy loss. Our model supports
adjustable latency through a configurable latency multiplier. Experimental
results demonstrate that our system achieves 44.3 BLEU for English-to-Chinese
and 25.1 BLEU for English-to-German translations on the ACL60/60 development
set, with computation-aware latencies of 2.7 seconds and 2.3 seconds, and
theoretical latencies of 2.2 and 1.7 seconds, respectively.

### 4. [Development of the user-friendly decision aid Rule-based Evaluation and Support Tool (REST) for optimizing the resources of an information extraction task](http://arxiv.org/pdf/2506.13177v1)

Authors: Guillaume Bazin, Xavier Tannier, Fanny Adda, Ariel Cohen, Akram Redjdal, Emmanuelle Kempf

Rules could be an information extraction (IE) default option, compared to ML
and LLMs in terms of sustainability, transferability, interpretability, and
development burden. We suggest a sustainable and combined use of rules and ML
as an IE method. Our approach starts with an exhaustive expert manual
highlighting in a single working session of a representative subset of the data
corpus. We developed and validated the feasibility and the performance metrics
of the REST decision tool to help the annotator choose between rules as a by
default option and ML for each entity of an IE task. REST makes the annotator
visualize the characteristics of each entity formalization in the free texts
and the expected rule development feasibility and IE performance metrics. ML is
considered as a backup IE option and manual annotation for training is
therefore minimized. The external validity of REST on a 12-entity use case
showed good reproducibility.

### 5. [Enhancing Large Language Models with Reliable Knowledge Graphs](http://arxiv.org/pdf/2506.13178v1)

Authors: Qinggang Zhang

Large Language Models (LLMs) have demonstrated remarkable capabilities in
text generation and understanding, yet their reliance on implicit, unstructured
knowledge often leads to factual inaccuracies and limited interpretability.
Knowledge Graphs (KGs), with their structured, relational representations,
offer a promising solution to ground LLMs in verified knowledge. However, their
potential remains constrained by inherent noise, incompleteness, and the
complexity of integrating their rigid structure with the flexible reasoning of
LLMs. This thesis presents a systematic framework to address these limitations,
advancing the reliability of KGs and their synergistic integration with LLMs
through five interconnected contributions. This thesis addresses these
challenges through a cohesive framework that enhances LLMs by refining and
leveraging reliable KGs. First, we introduce contrastive error detection, a
structure-based method to identify incorrect facts in KGs. This approach is
extended by an attribute-aware framework that unifies structural and semantic
signals for error correction. Next, we propose an inductive completion model
that further refines KGs by completing the missing relationships in evolving
KGs. Building on these refined KGs, KnowGPT integrates structured graph
reasoning into LLMs through dynamic prompting, improving factual grounding.
These contributions form a systematic pipeline (from error detection to LLM
integration), demonstrating that reliable KGs significantly enhance the
robustness, interpretability, and adaptability of LLMs.

### 6. [Dynamic Acoustic Model Architecture Optimization in Training for ASR](http://arxiv.org/pdf/2506.13180v1)

Authors: Jingjing Xu, Zijian Yang, Albert Zeyer, Eugen Beck, Ralf Schlueter, Hermann Ney

Architecture design is inherently complex. Existing approaches rely on either
handcrafted rules, which demand extensive empirical expertise, or automated
methods like neural architecture search, which are computationally intensive.
In this paper, we introduce DMAO, an architecture optimization framework that
employs a grow-and-drop strategy to automatically reallocate parameters during
training. This reallocation shifts resources from less-utilized areas to those
parts of the model where they are most beneficial. Notably, DMAO only
introduces negligible training overhead at a given model complexity. We
evaluate DMAO through experiments with CTC on LibriSpeech, TED-LIUM-v2 and
Switchboard datasets. The results show that, using the same amount of training
resources, our proposed DMAO consistently improves WER by up to 6% relatively
across various architectures, model sizes, and datasets. Furthermore, we
analyze the pattern of parameter redistribution and uncover insightful
findings.

### 7. [Capability Salience Vector: Fine-grained Alignment of Loss and Capabilities for Downstream Task Scaling Law](http://arxiv.org/pdf/2506.13216v1)

Authors: Qiming Ge, Shuhao Xing, Songyang Gao, Yunhua Zhou, Yicheng Zou, Songyang Zhang, Zhi Chen, Hang Yan, Qi Zhang, Qipeng Guo, Kai Chen

Scaling law builds the relationship between training computation and
validation loss, enabling researchers to effectively predict the loss trending
of models across different levels of computation. However, a gap still remains
between validation loss and the model's downstream capabilities, making it
untrivial to apply scaling law to direct performance prediction for downstream
tasks. The loss typically represents a cumulative penalty for predicted tokens,
which are implicitly considered to have equal importance. Nevertheless, our
studies have shown evidence that when considering different training data
distributions, we cannot directly model the relationship between downstream
capability and computation or token loss. To bridge the gap between validation
loss and downstream task capabilities, in this work, we introduce Capability
Salience Vector, which decomposes the overall loss and assigns different
importance weights to tokens to assess a specific meta-capability, aligning the
validation loss with downstream task performance in terms of the model's
capabilities. Experiments on various popular benchmarks demonstrate that our
proposed Capability Salience Vector could significantly improve the
predictability of language model performance on downstream tasks.

### 8. [IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation](http://arxiv.org/pdf/2506.13229v1)

Authors: Zijie Lin, Yang Zhang, Xiaoyan Zhao, Fengbin Zhu, Fuli Feng, Tat-Seng Chua

Large Language Models (LLMs) have shown strong potential for recommendation
by framing item prediction as a token-by-token language generation task.
However, existing methods treat all item tokens equally, simply pursuing
likelihood maximization during both optimization and decoding. This overlooks
crucial token-level differences in decisiveness-many tokens contribute little
to item discrimination yet can dominate optimization or decoding. To quantify
token decisiveness, we propose a novel perspective that models item generation
as a decision process, measuring token decisiveness by the Information Gain
(IG) each token provides in reducing uncertainty about the generated item. Our
empirical analysis reveals that most tokens have low IG but often correspond to
high logits, disproportionately influencing training loss and decoding, which
may impair model performance. Building on these insights, we introduce an
Information Gain-based Decisiveness-aware Token handling (IGD) strategy that
integrates token decisiveness into both tuning and decoding. Specifically, IGD
downweights low-IG tokens during tuning and rebalances decoding to emphasize
tokens with high IG. In this way, IGD moves beyond pure likelihood
maximization, effectively prioritizing high-decisiveness tokens. Extensive
experiments on four benchmark datasets with two LLM backbones demonstrate that
IGD consistently improves recommendation accuracy, achieving significant gains
on widely used ranking metrics compared to strong baselines.

### 9. [Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs](http://arxiv.org/pdf/2506.13285v1)

Authors: Houcheng Jiang, Zetong Zhao, Junfeng Fang, Haokai Ma, Ruipeng Wang, Yang Deng, Xiang Wang, Xiangnan He

Large language models (LLMs) have shown strong performance across natural
language tasks, but remain vulnerable to backdoor attacks. Recent model
editing-based approaches enable efficient backdoor injection by directly
modifying parameters to map specific triggers to attacker-desired responses.
However, these methods often suffer from safety fallback, where the model
initially responds affirmatively but later reverts to refusals due to safety
alignment. In this work, we propose DualEdit, a dual-objective model editing
framework that jointly promotes affirmative outputs and suppresses refusal
responses. To address two key challenges -- balancing the trade-off between
affirmative promotion and refusal suppression, and handling the diversity of
refusal expressions -- DualEdit introduces two complementary techniques. (1)
Dynamic loss weighting calibrates the objective scale based on the pre-edited
model to stabilize optimization. (2) Refusal value anchoring compresses the
suppression target space by clustering representative refusal value vectors,
reducing optimization conflict from overly diverse token sets. Experiments on
safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and
reduces safety fallback rate by 10.88\% over baselines.

### 10. [Document-Level Tabular Numerical Cross-Checking: A Coarse-to-Fine Approach](http://arxiv.org/pdf/2506.13328v1)

Authors: Chaoxu Pang, Yixuan Cao, Ganbin Zhou, Hongwei Li, Ping Luo

Numerical consistency across tables in disclosure documents is critical for
ensuring accuracy, maintaining credibility, and avoiding reputational and
economic risks. Automated tabular numerical cross-checking presents two
significant challenges: (C1) managing the combinatorial explosion of candidate
instances at the document level and (C2) comprehending multi-faceted numerical
semantics. Previous research typically depends on heuristic-based filtering or
simplified context extraction, often struggling to balance performance and
efficiency. Recently, large language models (LLMs) have demonstrated remarkable
contextual understanding capabilities that helps address C2 at the instance
level, yet they remain hampered by computational inefficiency (C1) and limited
domain expertise. This paper introduces CoFiTCheck, a novel LLM-based
coarse-to-fine framework that addresses these challenges through two sequential
stages: embedding-based filtering and discriminative classification. The
embedding-based filtering stage introduces an instructional parallel encoding
method to efficiently represent all numerical mentions in a table with LLMs, as
well as a decoupled InfoNCE objective to mitigate the isolated mention problem.
The discriminative classification stage employs a specialized LLM for
fine-grained analysis of the remaining candidate pairs. This stage is further
enhanced by our crosstable numerical alignment pretraining paradigm, which
leverages weak supervision from cross-table numerical equality relationships to
enrich task-specific priors without requiring manual annotation. Comprehensive
evaluation across three types of real-world disclosure documents demonstrates
that CoFiTCheck significantly outperforms previous methods while maintaining
practical efficiency.

### Cryptography and Security

### 1. [Buy it Now, Track Me Later: Attacking User Privacy via Wi-Fi AP Online Auctions](http://arxiv.org/pdf/2506.13052v1)

Authors: Steven Su, Erik Rye, Robert Beverly, Dave Levin

Static and hard-coded layer-two network identifiers are well known to present
security vulnerabilities and endanger user privacy. In this work, we introduce
a new privacy attack against Wi-Fi access points listed on secondhand
marketplaces. Specifically, we demonstrate the ability to remotely gather a
large quantity of layer-two Wi-Fi identifiers by programmatically querying the
eBay marketplace and applying state-of-the-art computer vision techniques to
extract IEEE 802.11 BSSIDs from the seller's posted images of the hardware. By
leveraging data from a global Wi-Fi Positioning System (WPS) that geolocates
BSSIDs, we obtain the physical locations of these devices both pre- and
post-sale. In addition to validating the degree to which a seller's location
matches the location of the device, we examine cases of device movement -- once
the device is sold and then subsequently re-used in a new environment. Our work
highlights a previously unrecognized privacy vulnerability and suggests, yet
again, the strong need to protect layer-two network identifiers.

### 2. [Detecting Hard-Coded Credentials in Software Repositories via LLMs](http://arxiv.org/pdf/2506.13090v1)

Authors: Chidera Biringa, Gokhan Kul

Software developers frequently hard-code credentials such as passwords,
generic secrets, private keys, and generic tokens in software repositories,
even though it is strictly advised against due to the severe threat to the
security of the software. These credentials create attack surfaces exploitable
by a potential adversary to conduct malicious exploits such as backdoor
attacks. Recent detection efforts utilize embedding models to vectorize textual
credentials before passing them to classifiers for predictions. However, these
models struggle to discriminate between credentials with contextual and complex
sequences resulting in high false positive predictions. Context-dependent
Pre-trained Language Models (PLMs) or Large Language Models (LLMs) such as
Generative Pre-trained Transformers (GPT) tackled this drawback by leveraging
the transformer neural architecture capacity for self-attention to capture
contextual dependencies between words in input sequences. As a result, GPT has
achieved wide success in several natural language understanding endeavors.
Hence, we assess LLMs to represent these observations and feed extracted
embedding vectors to a deep learning classifier to detect hard-coded
credentials. Our model outperforms the current state-of-the-art by 13% in F1
measure on the benchmark dataset. We have made all source code and data
publicly available to facilitate the reproduction of all results presented in
this paper.

### 3. [Dual Protection Ring: User Profiling Via Differential Privacy and Service Dissemination Through Private Information Retrieval](http://arxiv.org/pdf/2506.13170v1)

Authors: Imdad Ullah, Najm Hassan, Tariq Ahamed Ahangar, Zawar Hussain Shah, Mehregan Mahdavi Andrew Levula

User profiling is crucial in providing personalised services, as it relies on
analysing user behaviour and preferences to deliver targeted services. This
approach enhances user experience and promotes heightened engagement.
Nevertheless, user profiling also gives rise to noteworthy privacy
considerations due to the extensive tracking and monitoring of personal data,
potentially leading to surveillance or identity theft. We propose a dual-ring
protection mechanism to protect user privacy by examining various threats to
user privacy, such as behavioural attacks, profiling fingerprinting and
monitoring, profile perturbation, etc., both on the user and service provider
sides. We develop user profiles that contain sensitive private attributes and
an equivalent profile based on differential privacy for evaluating personalised
services. We determine the entropy of the resultant profiles during each update
to protect profiling attributes and invoke various processes, such as data
evaporation, to artificially increase entropy or destroy private profiling
attributes. Furthermore, we use different variants of private information
retrieval (PIR) to retrieve personalised services against differentially
private profiles. We implement critical components of the proposed model via a
proof-of-concept mobile app to demonstrate its applicability over a specific
case study of advertising services, which can be generalised to other services.
Our experimental results show that the observed processing delays with
different PIR schemes are similar to the current advertising systems.

### 4. [The Rich Get Richer in Bitcoin Mining Induced by Blockchain Forks](http://arxiv.org/pdf/2506.13360v1)

Authors: Akira Sakurai, Kazuyuki Shudo

Bitcoin is a representative decentralized currency system. For the security
of Bitcoin, fairness in the distribution of mining rewards plays a crucial role
in preventing the concentration of computational power in a few miners. Here,
fairness refers to the distribution of block rewards in proportion to
contributed computational resources. If miners with greater computational
resources receive disproportionately higher rewards, i.e., if the Rich Get
Richer (TRGR) phenomenon holds in Bitcoin, it indicates a threat to the
system's decentralization. This study analyzes TRGR in Bitcoin by focusing on
unintentional blockchain forks, an inherent phenomenon in Bitcoin. Previous
research has failed to provide generalizable insights due to the low precision
of their analytical methods. In contrast, we avoid this problem by adopting a
method whose analytical precision has been empirically validated. The primary
contribution of this work is a theoretical analysis that clearly demonstrates
TRGR in Bitcoin under the assumption of fixed block propagation delays between
different miners. More specifically, we show that the mining profit rate
depends linearly on the proportion of hashrate. Furthermore, we examine the
robustness of this result from multiple perspectives in scenarios where block
propagation delays between different miners are not necessarily fixed.

### 5. [New characterization of full weight spectrum one-orbit cyclic subspace codes](http://arxiv.org/pdf/2506.13418v1)

Authors: Minjia Shi, Wenhao Song

Castello $\textit{et al}$. [J. Comb. Theory Ser. A, 212, 106005 (2025)]
provided a complete classification for full weight spectrum (FWS) one-orbit
cyclic subspace codes. In this paper, we determine the weight distributions of
a family of FWS codes and exhibit some equivalence classes of FWS codes under
certain conditions. Furthermore, we provide a complete classification for
$r$-FWS codes.

### 6. [From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs](http://arxiv.org/pdf/2506.13434v1)

Authors: Alsharif Abuadbba, Chris Hicks, Kristen Moore, Vasilios Mavroudis, Burak Hasircioglu, Diksha Goel, Piers Jennings

Large Language Models (LLMs) are set to reshape cybersecurity by augmenting
red and blue team operations. Red teams can exploit LLMs to plan attacks, craft
phishing content, simulate adversaries, and generate exploit code. Conversely,
blue teams may deploy them for threat intelligence synthesis, root cause
analysis, and streamlined documentation. This dual capability introduces both
transformative potential and serious risks.
  This position paper maps LLM applications across cybersecurity frameworks
such as MITRE ATT&CK and the NIST Cybersecurity Framework (CSF), offering a
structured view of their current utility and limitations. While LLMs
demonstrate fluency and versatility across various tasks, they remain fragile
in high-stakes, context-heavy environments. Key limitations include
hallucinations, limited context retention, poor reasoning, and sensitivity to
prompts, which undermine their reliability in operational settings.
  Moreover, real-world integration raises concerns around dual-use risks,
adversarial misuse, and diminished human oversight. Malicious actors could
exploit LLMs to automate reconnaissance, obscure attack vectors, and lower the
technical threshold for executing sophisticated attacks.
  To ensure safer adoption, we recommend maintaining human-in-the-loop
oversight, enhancing model explainability, integrating privacy-preserving
mechanisms, and building systems robust to adversarial exploitation. As
organizations increasingly adopt AI driven cybersecurity, a nuanced
understanding of LLMs' risks and operational impacts is critical to securing
their defensive value while mitigating unintended consequences.

### 7. [Watermarking LLM-Generated Datasets in Downstream Tasks](http://arxiv.org/pdf/2506.13494v1)

Authors: Yugeng Liu, Tianshuo Cong, Michael Backes, Zheng Li, Yang Zhang

Large Language Models (LLMs) have experienced rapid advancements, with
applications spanning a wide range of fields, including sentiment
classification, review generation, and question answering. Due to their
efficiency and versatility, researchers and companies increasingly employ
LLM-generated data to train their models. However, the inability to track
content produced by LLMs poses a significant challenge, potentially leading to
copyright infringement for the LLM owners. In this paper, we propose a method
for injecting watermarks into LLM-generated datasets, enabling the tracking of
downstream tasks to detect whether these datasets were produced using the
original LLM. These downstream tasks can be divided into two categories. The
first involves using the generated datasets at the input level, commonly for
training classification tasks. The other is the output level, where model
trainers use LLM-generated content as output for downstream tasks, such as
question-answering tasks. We design a comprehensive set of experiments to
evaluate both watermark methods. Our results indicate the high effectiveness of
our watermark approach. Additionally, regarding model utility, we find that
classifiers trained on the generated datasets achieve a test accuracy exceeding
0.900 in many cases, suggesting that the utility of such models remains robust.
For the output-level watermark, we observe that the quality of the generated
text is comparable to that produced using real-world datasets. Through our
research, we aim to advance the protection of LLM copyrights, taking a
significant step forward in safeguarding intellectual property in this domain.

### 8. [ExtendAttack: Attacking Servers of LRMs via Extending Reasoning](http://arxiv.org/pdf/2506.13737v1)

Authors: Zhenhao Zhu, Yue Liu, Yingwei Ma, Hongcheng Gao, Nuo Chen, Yanpei Guo, Wenjie Qu, Huiying Xu, Xinzhong Zhu, Jiaheng Zhang

Large Reasoning Models (LRMs) have demonstrated promising performance in
complex tasks. However, the resource-consuming reasoning processes may be
exploited by attackers to maliciously occupy the resources of the servers,
leading to a crash, like the DDoS attack in cyber. To this end, we propose a
novel attack method on LRMs termed ExtendAttack to maliciously occupy the
resources of servers by stealthily extending the reasoning processes of LRMs.
Concretely, we systematically obfuscate characters within a benign prompt,
transforming them into a complex, poly-base ASCII representation. This compels
the model to perform a series of computationally intensive decoding sub-tasks
that are deeply embedded within the semantic structure of the query itself.
Extensive experiments demonstrate the effectiveness of our proposed
ExtendAttack. Remarkably, it increases the length of the model's response by
over 2.5 times for the o3 model on the HumanEval benchmark. Besides, it
preserves the original meaning of the query and achieves comparable answer
accuracy, showing the stealthiness.

### 9. [Rectifying Privacy and Efficacy Measurements in Machine Unlearning: A New Inference Attack Perspective](http://arxiv.org/pdf/2506.13009v1)

Authors: Nima Naderloui, Shenao Yan, Binghui Wang, Jie Fu, Wendy Hui Wang, Weiran Liu, Yuan Hong

Machine unlearning focuses on efficiently removing specific data from trained
models, addressing privacy and compliance concerns with reasonable costs.
Although exact unlearning ensures complete data removal equivalent to
retraining, it is impractical for large-scale models, leading to growing
interest in inexact unlearning methods. However, the lack of formal guarantees
in these methods necessitates the need for robust evaluation frameworks to
assess their privacy and effectiveness. In this work, we first identify several
key pitfalls of the existing unlearning evaluation frameworks, e.g., focusing
on average-case evaluation or targeting random samples for evaluation,
incomplete comparisons with the retraining baseline. Then, we propose RULI
(Rectified Unlearning Evaluation Framework via Likelihood Inference), a novel
framework to address critical gaps in the evaluation of inexact unlearning
methods. RULI introduces a dual-objective attack to measure both unlearning
efficacy and privacy risks at a per-sample granularity. Our findings reveal
significant vulnerabilities in state-of-the-art unlearning methods, where RULI
achieves higher attack success rates, exposing privacy risks underestimated by
existing methods. Built on a game-based foundation and validated through
empirical evaluations on both image and text data (spanning tasks from
classification to generation), RULI provides a rigorous, scalable, and
fine-grained methodology for evaluating unlearning techniques.

### 10. [Position: Certified Robustness Does Not (Yet) Imply Model Security](http://arxiv.org/pdf/2506.13024v1)

Authors: Andrew C. Cullen, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

While certified robustness is widely promoted as a solution to adversarial
examples in Artificial Intelligence systems, significant challenges remain
before these techniques can be meaningfully deployed in real-world
applications. We identify critical gaps in current research, including the
paradox of detection without distinction, the lack of clear criteria for
practitioners to evaluate certification schemes, and the potential security
risks arising from users' expectations surrounding ``guaranteed" robustness
claims. This position paper is a call to arms for the certification research
community, proposing concrete steps to address these fundamental challenges and
advance the field toward practical applicability.

### Computer Vision and Pattern Recognition

### 1. [DETRPose: Real-time end-to-end transformer model for multi-person pose estimation](http://arxiv.org/pdf/2506.13027v1)

Authors: Sebastian Janampa, Marios Pattichis

Multi-person pose estimation (MPPE) estimates keypoints for all individuals
present in an image. MPPE is a fundamental task for several applications in
computer vision and virtual reality. Unfortunately, there are currently no
transformer-based models that can perform MPPE in real time. The paper presents
a family of transformer-based models capable of performing multi-person 2D pose
estimation in real-time. Our approach utilizes a modified decoder architecture
and keypoint similarity metrics to generate both positive and negative queries,
thereby enhancing the quality of the selected queries within the architecture.
Compared to state-of-the-art models, our proposed models train much faster,
using 5 to 10 times fewer epochs, with competitive inference times without
requiring quantization libraries to speed up the model. Furthermore, our
proposed models provide competitive results or outperform alternative models,
often using significantly fewer parameters.

### 2. [WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild](http://arxiv.org/pdf/2506.13030v1)

Authors: Morris Alper, David Novotny, Filippos Kokkinos, Hadar Averbuch-Elor, Tom Monnier

Despite recent advances in sparse novel view synthesis (NVS) applied to
object-centric scenes, scene-level NVS remains a challenge. A central issue is
the lack of available clean multi-view training data, beyond manually curated
datasets with limited diversity, camera variation, or licensing issues. On the
other hand, an abundance of diverse and permissively-licensed data exists in
the wild, consisting of scenes with varying appearances (illuminations,
transient occlusions, etc.) from sources such as tourist photos. To this end,
we present WildCAT3D, a framework for generating novel views of scenes learned
from diverse 2D scene image data captured in the wild. We unlock training on
these data sources by explicitly modeling global appearance conditions in
images, extending the state-of-the-art multi-view diffusion paradigm to learn
from scene views of varying appearances. Our trained model generalizes to new
scenes at inference time, enabling the generation of multiple consistent novel
views. WildCAT3D provides state-of-the-art results on single-view NVS in
object- and scene-level settings, while training on strictly less data sources
than prior methods. Additionally, it enables novel applications by providing
global appearance control during generation.

### 3. [Evolution of ReID: From Early Methods to LLM Integration](http://arxiv.org/pdf/2506.13039v1)

Authors: Amran Bhuiyan, Mizanur Rahman, Md Tahmid Rahman Laskar, Aijun An, Jimmy Xiangji Huang

Person re-identification (ReID) has evolved from handcrafted feature-based
methods to deep learning approaches and, more recently, to models incorporating
large language models (LLMs). Early methods struggled with variations in
lighting, pose, and viewpoint, but deep learning addressed these issues by
learning robust visual features. Building on this, LLMs now enable ReID systems
to integrate semantic and contextual information through natural language. This
survey traces that full evolution and offers one of the first comprehensive
reviews of ReID approaches that leverage LLMs, where textual descriptions are
used as privileged information to improve visual matching. A key contribution
is the use of dynamic, identity-specific prompts generated by GPT-4o, which
enhance the alignment between images and text in vision-language ReID systems.
Experimental results show that these descriptions improve accuracy, especially
in complex or ambiguous cases. To support further research, we release a large
set of GPT-4o-generated descriptions for standard ReID datasets. By bridging
computer vision and natural language processing, this survey offers a unified
perspective on the field's development and outlines key future directions such
as better prompt design, cross-modal transfer learning, and real-world
adaptability.

### 4. [MAMMA: Markerless & Automatic Multi-Person Motion Action Capture](http://arxiv.org/pdf/2506.13040v1)

Authors: Hanz Cuevas-Velasquez, Anastasios Yiannakidis, Soyong Shin, Giorgio Becherini, Markus Höschle, Joachim Tesch, Taylor Obersat, Tsvetelina Alexiadis, Michael Black

We present MAMMA, a markerless motion-capture pipeline that accurately
recovers SMPL-X parameters from multi-view video of two-person interaction
sequences. Traditional motion-capture systems rely on physical markers.
Although they offer high accuracy, their requirements of specialized hardware,
manual marker placement, and extensive post-processing make them costly and
time-consuming. Recent learning-based methods attempt to overcome these
limitations, but most are designed for single-person capture, rely on sparse
keypoints, or struggle with occlusions and physical interactions. In this work,
we introduce a method that predicts dense 2D surface landmarks conditioned on
segmentation masks, enabling person-specific correspondence estimation even
under heavy occlusion. We employ a novel architecture that exploits learnable
queries for each landmark. We demonstrate that our approach can handle complex
person--person interaction and offers greater accuracy than existing methods.
To train our network, we construct a large, synthetic multi-view dataset
combining human motions from diverse sources, including extreme poses, hand
motions, and close interactions. Our dataset yields high-variability synthetic
sequences with rich body contact and occlusion, and includes SMPL-X
ground-truth annotations with dense 2D landmarks. The result is a system
capable of capturing human motion without the need for markers. Our approach
offers competitive reconstruction quality compared to commercial marker-based
motion-capture solutions, without the extensive manual cleanup. Finally, we
address the absence of common benchmarks for dense-landmark prediction and
markerless motion capture by introducing two evaluation settings built from
real multi-view sequences. We will release our dataset, benchmark, method,
training code, and pre-trained model weights for research purposes.

### 5. [ViewPCL: a point cloud based active learning method for multi-view segmentation](http://arxiv.org/pdf/2506.13043v1)

Authors: Christian Hilaire, Sima Didari

We propose a novel active learning framework for multi-view semantic
segmentation. This framework relies on a new score that measures the
discrepancy between point cloud distributions generated from the extra
geometrical information derived from the model's prediction across different
views. Our approach results in a data efficient and explainable active learning
method. The source code is available at https://github.com/chilai235/viewpclAL.

### 6. [Video Individual Counting With Implicit One-to-Many Matching](http://arxiv.org/pdf/2506.13067v1)

Authors: Xuhui Zhu, Jing Xu, Bingjie Wang, Huikang Dai, Hao Lu

Video Individual Counting (VIC) is a recently introduced task that aims to
estimate pedestrian flux from a video. It extends conventional Video Crowd
Counting (VCC) beyond the per-frame pedestrian count. In contrast to VCC that
only learns to count repeated pedestrian patterns across frames, the key
problem of VIC is how to identify co-existent pedestrians between frames, which
turns out to be a correspondence problem. Existing VIC approaches, however,
mainly follow a one-to-one (O2O) matching strategy where the same pedestrian
must be exactly matched between frames, leading to sensitivity to appearance
variations or missing detections. In this work, we show that the O2O matching
could be relaxed to a one-to-many (O2M) matching problem, which better fits the
problem nature of VIC and can leverage the social grouping behavior of walking
pedestrians. We therefore introduce OMAN, a simple but effective VIC model with
implicit One-to-Many mAtchiNg, featuring an implicit context generator and a
one-to-many pairwise matcher. Experiments on the SenseCrowd and CroHD
benchmarks show that OMAN achieves the state-of-the-art performance. Code is
available at \href{https://github.com/tiny-smart/OMAN}{OMAN}.

### 7. [SuperPlace: The Renaissance of Classical Feature Aggregation for Visual Place Recognition in the Era of Foundation Models](http://arxiv.org/pdf/2506.13073v1)

Authors: Bingxi Liu, Pengju Zhang, Li He, Hao Chen, Shiyi Guo, Yihong Wu, Jinqiang Cui, Hong Zhang

Recent visual place recognition (VPR) approaches have leveraged foundation
models (FM) and introduced novel aggregation techniques. However, these methods
have failed to fully exploit key concepts of FM, such as the effective
utilization of extensive training sets, and they have overlooked the potential
of classical aggregation methods, such as GeM and NetVLAD. Building on these
insights, we revive classical feature aggregation methods and develop more
fundamental VPR models, collectively termed SuperPlace. First, we introduce a
supervised label alignment method that enables training across various VPR
datasets within a unified framework. Second, we propose G$^2$M, a compact
feature aggregation method utilizing two GeMs, where one GeM learns the
principal components of feature maps along the channel dimension and calibrates
the output of the other. Third, we propose the secondary fine-tuning (FT$^2$)
strategy for NetVLAD-Linear (NVL). NetVLAD first learns feature vectors in a
high-dimensional space and then compresses them into a lower-dimensional space
via a single linear layer. Extensive experiments highlight our contributions
and demonstrate the superiority of SuperPlace. Specifically, G$^2$M achieves
promising results with only one-tenth of the feature dimensions compared to
recent methods. Moreover, NVL-FT$^2$ ranks first on the MSLS leaderboard.

### 8. [Learning Event Completeness for Weakly Supervised Video Anomaly Detection](http://arxiv.org/pdf/2506.13095v1)

Authors: Yu Wang, Shiwei Chen

Weakly supervised video anomaly detection (WS-VAD) is tasked with pinpointing
temporal intervals containing anomalous events within untrimmed videos,
utilizing only video-level annotations. However, a significant challenge arises
due to the absence of dense frame-level annotations, often leading to
incomplete localization in existing WS-VAD methods. To address this issue, we
present a novel LEC-VAD, Learning Event Completeness for Weakly Supervised
Video Anomaly Detection, which features a dual structure designed to encode
both category-aware and category-agnostic semantics between vision and
language. Within LEC-VAD, we devise semantic regularities that leverage an
anomaly-aware Gaussian mixture to learn precise event boundaries, thereby
yielding more complete event instances. Besides, we develop a novel memory
bank-based prototype learning mechanism to enrich concise text descriptions
associated with anomaly-event categories. This innovation bolsters the text's
expressiveness, which is crucial for advancing WS-VAD. Our LEC-VAD demonstrates
remarkable advancements over the current state-of-the-art methods on two
benchmark datasets XD-Violence and UCF-Crime.

### 9. [Pro-AD: Learning Comprehensive Prototypes with Prototype-based Constraint for Multi-class Unsupervised Anomaly Detection](http://arxiv.org/pdf/2506.13097v1)

Authors: Ziqing Zhou, Binbin Gao, Yuri Pan, Lidong Wang, Wenbing Zhu, Yong Liu, Jun Liu, MIngmin Chi, Dong Wu, Bo Peng, Chengjie Wang

Prototype-based reconstruction methods for unsupervised anomaly detection
utilize a limited set of learnable prototypes which only aggregates
insufficient normal information, resulting in undesirable reconstruction.
However, increasing the number of prototypes may lead to anomalies being well
reconstructed through the attention mechanism, which we refer to as the "Soft
Identity Mapping" problem. In this paper, we propose Pro-AD to address these
issues and fully utilize the prototypes to boost the performance of anomaly
detection. Specifically, we first introduce an expanded set of learnable
prototypes to provide sufficient capacity for semantic information. Then we
employ a Dynamic Bidirectional Decoder which integrates the process of the
normal information aggregation and the target feature reconstruction via
prototypes, with the aim of allowing the prototypes to aggregate more
comprehensive normal semantic information from different levels of the image
features and the target feature reconstruction to not only utilize its
contextual information but also dynamically leverage the learned comprehensive
prototypes. Additionally, to prevent the anomalies from being well
reconstructed using sufficient semantic information through the attention
mechanism, Pro-AD introduces a Prototype-based Constraint that applied within
the target feature reconstruction process of the decoder, which further
improves the performance of our approach. Extensive experiments on multiple
challenging benchmarks demonstrate that our Pro-AD achieve state-of-the-art
performance, highlighting its superior robustness and practical effectiveness
for Multi-class Unsupervised Anomaly Detection task.

### 10. [GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction](http://arxiv.org/pdf/2506.13110v1)

Authors: Jinguang Tong, Xuesong li, Fahira Afzal Maken, Sundaram Muthu, Lars Petersson, Chuong Nguyen, Hongdong Li

3D modeling of highly reflective objects remains challenging due to strong
view-dependent appearances. While previous SDF-based methods can recover
high-quality meshes, they are often time-consuming and tend to produce
over-smoothed surfaces. In contrast, 3D Gaussian Splatting (3DGS) offers the
advantage of high speed and detailed real-time rendering, but extracting
surfaces from the Gaussians can be noisy due to the lack of geometric
constraints. To bridge the gap between these approaches, we propose a novel
reconstruction method called GS-2DGS for reflective objects based on 2D
Gaussian Splatting (2DGS). Our approach combines the rapid rendering
capabilities of Gaussian Splatting with additional geometric information from
foundation models. Experimental results on synthetic and real datasets
demonstrate that our method significantly outperforms Gaussian-based techniques
in terms of reconstruction and relighting and achieves performance comparable
to SDF-based methods while being an order of magnitude faster. Code is
available at https://github.com/hirotong/GS2DGS

### Computers and Society

### 1. [pySpainMobility: a Python Package to Access and Manage Spanish Open Mobility Data](http://arxiv.org/pdf/2506.13385v1)

Authors: Ciro Beneduce, Tania Gullón Muñoz-Repiso, Bruno Lepri, Massimiliano Luca

Mobility patterns play a critical role in a wide range of societal
challenges, from epidemic modeling and emergency response to transportation
planning and regional development. Yet, access to high-quality, timely, and
openly available mobility data remains limited. In response, the Spanish
Ministry of Transportation and Sustainable Mobility has released daily mobility
datasets based on anonymized mobile phone data, covering districts,
municipalities, and greater urban areas from February 2020 to June 2021 and
again from January 2022 onward. This paper presents pySpainMobility, a Python
package that simplifies access to these datasets and their associated study
areas through a standardized, well-documented interface. By lowering the
technical barrier to working with large-scale mobility data, the package
enables reproducible analysis and supports applications across research,
policy, and operational domains. The library is available at
https://github.com/pySpainMobility.

### 2. [Navigating through CS1: The Role of Self-Regulation and Supervision in Student Progress](http://arxiv.org/pdf/2506.13461v1)

Authors: Ville Isomöttönen, Denis Zhidkikh

The need for students' self-regulation for fluent transitioning to university
studies is known. Our aim was to integrate study-supportive activities with
course supervision activities within CS1. We educated TAs to pay attention to
students' study ability and self-regulation. An interview study ($N=14$) was
undertaken to investigate this approach. A thematic analysis yielded rather
mixed results in light of our aims. Self-regulation was underpinned by the
influences external to our setting, including labor market-related needs,
earlier crises in study habits, and personal characteristics such as passion,
grit, creativity, and valuation of utility. Safety in one-to-one supervision
was considered essential, while shyness, fear, and even altruism caused
self-handicapping during the course. Students were aware of their learning
styles and need for self-regulation, while did not always know how to
self-regulate or preferred to externalize it. The results highlight that
supporting self-regulation should be integrated with students' personal
histories and experiences, and thereby calls attention to transformative
learning pedagogies. The thematization can help to understand CS1 students'
self-regulation processes and improve CS1 support practices.

### 3. [Safe-Child-LLM: A Developmental Benchmark for Evaluating LLM Safety in Child-AI Interactions](http://arxiv.org/pdf/2506.13510v1)

Authors: Junfeng Jiao, Saleh Afroogh, Kevin Chen, Abhejay Murali, David Atkinson, Amit Dhurandhar

As Large Language Models (LLMs) increasingly power applications used by
children and adolescents, ensuring safe and age-appropriate interactions has
become an urgent ethical imperative. Despite progress in AI safety, current
evaluations predominantly focus on adults, neglecting the unique
vulnerabilities of minors engaging with generative AI. We introduce
Safe-Child-LLM, a comprehensive benchmark and dataset for systematically
assessing LLM safety across two developmental stages: children (7-12) and
adolescents (13-17). Our framework includes a novel multi-part dataset of 200
adversarial prompts, curated from red-teaming corpora (e.g., SG-Bench,
HarmBench), with human-annotated labels for jailbreak success and a
standardized 0-5 ethical refusal scale. Evaluating leading LLMs -- including
ChatGPT, Claude, Gemini, LLaMA, DeepSeek, Grok, Vicuna, and Mistral -- we
uncover critical safety deficiencies in child-facing scenarios. This work
highlights the need for community-driven benchmarks to protect young users in
LLM interactions. To promote transparency and collaborative advancement in
ethical AI development, we are publicly releasing both our benchmark datasets
and evaluation codebase at
https://github.com/The-Responsible-AI-Initiative/Safe_Child_LLM_Benchmark.git

### 4. [Bias Delayed is Bias Denied? Assessing the Effect of Reporting Delays on Disparity Assessments](http://arxiv.org/pdf/2506.13735v1)

Authors: Jennah Gosciak, Aparna Balagopalan, Derek Ouyang, Allison Koenecke, Marzyeh Ghassemi, Daniel E. Ho

Conducting disparity assessments at regular time intervals is critical for
surfacing potential biases in decision-making and improving outcomes across
demographic groups. Because disparity assessments fundamentally depend on the
availability of demographic information, their efficacy is limited by the
availability and consistency of available demographic identifiers. While prior
work has considered the impact of missing data on fairness, little attention
has been paid to the role of delayed demographic data. Delayed data, while
eventually observed, might be missing at the critical point of monitoring and
action -- and delays may be unequally distributed across groups in ways that
distort disparity assessments. We characterize such impacts in healthcare,
using electronic health records of over 5M patients across primary care
practices in all 50 states. Our contributions are threefold. First, we document
the high rate of race and ethnicity reporting delays in a healthcare setting
and demonstrate widespread variation in rates at which demographics are
reported across different groups. Second, through a set of retrospective
analyses using real data, we find that such delays impact disparity assessments
and hence conclusions made across a range of consequential healthcare outcomes,
particularly at more granular levels of state-level and practice-level
assessments. Third, we find limited ability of conventional methods that impute
missing race in mitigating the effects of reporting delays on the accuracy of
timely disparity assessments. Our insights and methods generalize to many
domains of algorithmic fairness where delays in the availability of sensitive
information may confound audits, thus deserving closer attention within a
pipeline-aware machine learning framework.

### 5. [The Transition Matrix -- A classification of navigational patterns between LMS course sections](http://arxiv.org/pdf/2506.13275v1)

Authors: Tobias Hildebrandt, Lars Mehnen

Learning management systems (LMS) like Moodle are increasingly used to
support university teaching. As Moodle courses become more complex,
incorporating diverse interactive elements, it is important to understand how
students navigate through course sections and whether course designs are
meeting student needs. While substantial research exists on student usage of
individual LMS elements, there is a lack of research on broader navigational
patterns between course sections and how these patterns differ across courses.
This study analyzes navigational data from 747 courses in the Moodle LMS at a
technical university of applied sciences, representing (after filtering) around
4,400 students and 1.8 million logged events. By mapping section names across a
large sample of courses, the analysis enables cross-course comparisons of
student navigational sequences between sections. Transition matrices and heat
map visualizations are used to identify common navigational patterns. Findings
include that many of the generated heatmap include one or more diagonal axis,
indicating that students typically navigate from the current to the next or
previous section. More fine-grained patterns show typical behavior for blended
learning scenarios. Other patterns include dominant sections.

### 6. [An LLM's Apology: Outsourcing Awkwardness in the Age of AI](http://arxiv.org/pdf/2506.13685v1)

Authors: Twm Stone, Anna Soligo

A key part of modern social dynamics is flaking at short notice. However,
anxiety in coming up with believable and socially acceptable reasons to do so
can instead lead to 'ghosting', awkwardness, or implausible excuses, risking
emotional harm and resentment in the other party. The ability to delegate this
task to a Large Language Model (LLM) could substantially reduce friction and
enhance the flexibility of user's social life while greatly minimising the
aforementioned creative burden and moral qualms. We introduce FLAKE-Bench, an
evaluation of models' capacity to effectively, kindly, and humanely extract
themselves from a diverse set of social, professional and romantic scenarios.
We report the efficacy of 10 frontier or recently-frontier LLMs in bailing on
prior commitments, because nothing says "I value our friendship" like having AI
generate your cancellation texts. We open-source FLAKE-Bench at
github.com/Cloakless/flake-bench to support future research.

### 7. [Rethinking Test-Time Scaling for Medical AI: Model and Task-Aware Strategies for LLMs and VLMs](http://arxiv.org/pdf/2506.13102v1)

Authors: Gyutaek Oh, Seoyeon Kim, Sangjoon Park, Byung-Hoon Kim

Test-time scaling has recently emerged as a promising approach for enhancing
the reasoning capabilities of large language models or vision-language models
during inference. Although a variety of test-time scaling strategies have been
proposed, and interest in their application to the medical domain is growing,
many critical aspects remain underexplored, including their effectiveness for
vision-language models and the identification of optimal strategies for
different settings. In this paper, we conduct a comprehensive investigation of
test-time scaling in the medical domain. We evaluate its impact on both large
language models and vision-language models, considering factors such as model
size, inherent model characteristics, and task complexity. Finally, we assess
the robustness of these strategies under user-driven factors, such as
misleading information embedded in prompts. Our findings offer practical
guidelines for the effective use of test-time scaling in medical applications
and provide insights into how these strategies can be further refined to meet
the reliability and interpretability demands of the medical domain.

### 8. [A Game-Theoretic Negotiation Framework for Cross-Cultural Consensus in LLMs](http://arxiv.org/pdf/2506.13245v1)

Authors: Guoxi Zhang, Jiawei Chen, Tianzhuo Yang, Jiaming Ji, Yaodong Yang, Juntao Dai

The increasing prevalence of large language models (LLMs) is influencing
global value systems. However, these models frequently exhibit a pronounced
WEIRD (Western, Educated, Industrialized, Rich, Democratic) cultural bias due
to lack of attention to minority values. This monocultural perspective may
reinforce dominant values and marginalize diverse cultural viewpoints, posing
challenges for the development of equitable and inclusive AI systems. In this
work, we introduce a systematic framework designed to boost fair and robust
cross-cultural consensus among LLMs. We model consensus as a Nash Equilibrium
and employ a game-theoretic negotiation method based on Policy-Space Response
Oracles (PSRO) to simulate an organized cross-cultural negotiation process. To
evaluate this approach, we construct regional cultural agents using data
transformed from the World Values Survey (WVS). Beyond the conventional
model-level evaluation method, We further propose two quantitative metrics,
Perplexity-based Acceptence and Values Self-Consistency, to assess consensus
outcomes. Experimental results indicate that our approach generates consensus
of higher quality while ensuring more balanced compromise compared to
baselines. Overall, it mitigates WEIRD bias by guiding agents toward
convergence through fair and gradual negotiation steps.

### 9. [Accessibility Barriers in Multi-Terabyte Public Datasets: The Gap Between Promise and Practice](http://arxiv.org/pdf/2506.13256v1)

Authors: Marc Bara

The promise of "free and open" multi-terabyte datasets often collides with
harsh realities. While these datasets may be technically accessible, practical
barriers -- from processing complexity to hidden costs -- create a system that
primarily serves well-funded institutions. This study examines accessibility
challenges across web crawls, satellite imagery, scientific data, and
collaborative projects, revealing a consistent two-tier system where
theoretical openness masks practical exclusivity. Our analysis demonstrates
that datasets marketed as "publicly accessible" typically require minimum
investments of \$1,000+ for meaningful analysis, with complex processing
pipelines demanding \$10,000-100,000+ in infrastructure costs. The
infrastructure requirements -- distributed computing knowledge, domain
expertise, and substantial budgets -- effectively gatekeep these datasets
despite their "open" status, limiting practical accessibility to those with
institutional support or substantial resources.

### 10. [Delving Into the Psychology of Machines: Exploring the Structure of Self-Regulated Learning via LLM-Generated Survey Responses](http://arxiv.org/pdf/2506.13384v1)

Authors: Leonie V. D. E. Vogelsmeier, Eduardo Oliveira, Kamila Misiejuk, Sonsoles López-Pernas, Mohammed Saqr

Large language models (LLMs) offer the potential to simulate human-like
responses and behaviors, creating new opportunities for psychological science.
In the context of self-regulated learning (SRL), if LLMs can reliably simulate
survey responses at scale and speed, they could be used to test intervention
scenarios, refine theoretical models, augment sparse datasets, and represent
hard-to-reach populations. However, the validity of LLM-generated survey
responses remains uncertain, with limited research focused on SRL and existing
studies beyond SRL yielding mixed results. Therefore, in this study, we
examined LLM-generated responses to the 44-item Motivated Strategies for
Learning Questionnaire (MSLQ; Pintrich \& De Groot, 1990), a widely used
instrument assessing students' learning strategies and academic motivation.
Particularly, we used the LLMs GPT-4o, Claude 3.7 Sonnet, Gemini 2 Flash, LLaMA
3.1-8B, and Mistral Large. We analyzed item distributions, the psychological
network of the theoretical SRL dimensions, and psychometric validity based on
the latent factor structure. Our results suggest that Gemini 2 Flash was the
most promising LLM, showing considerable sampling variability and producing
underlying dimensions and theoretical relationships that align with prior
theory and empirical findings. At the same time, we observed discrepancies and
limitations, underscoring both the potential and current constraints of using
LLMs for simulating psychological survey data and applying it in educational
contexts.

### Databases

### 1. [EnhanceGraph: A Continuously Enhanced Graph-based Index for High-dimensional Approximate Nearest Neighbor Search](http://arxiv.org/pdf/2506.13144v1)

Authors: Xiaoyao Zhong, Jiabao Jin, Peng Cheng, Mingyu Yang, Lei Chen, Haoyang Li, Zhitao Shen, Xuemin Lin, Heng Tao Shen, Jingkuan Song

Recently, Approximate Nearest Neighbor Search in high-dimensional vector
spaces has garnered considerable attention due to the rapid advancement of deep
learning techniques. We observed that a substantial amount of search and
construction logs are generated throughout the lifespan of a graph-based index.
However, these two types of valuable logs are not fully exploited due to the
static nature of existing indexes. We present the EnhanceGraph framework, which
integrates two types of logs into a novel structure called a conjugate graph.
The conjugate graph is then used to improve search quality. Through theoretical
analyses and observations of the limitations of graph-based indexes, we propose
several optimization methods. For the search logs, the conjugate graph stores
the edges from local optima to global optima to enhance routing to the nearest
neighbor. For the construction logs, the conjugate graph stores the pruned
edges from the proximity graph to enhance retrieving of k nearest neighbors.
Our experimental results on several public and real-world industrial datasets
show that EnhanceGraph significantly improves search accuracy with the greatest
improvement on recall from 41.74% to 93.42%, but does not sacrifices search
efficiency. In addition, our EnhanceGraph algorithm has been integrated into
Ant Group's open-source vector library, VSAG.

### 2. [Parachute: Single-Pass Bi-Directional Information Passing](http://arxiv.org/pdf/2506.13670v1)

Authors: Mihail Stoian, Andreas Zimmerer, Skander Krid, Amadou Latyr Ngom, Jialin Ding, Tim Kraska, Andreas Kipf

Sideways information passing is a well-known technique for mitigating the
impact of large build sides in a database query plan. As currently implemented
in production systems, sideways information passing enables only a
uni-directional information flow, as opposed to instance-optimal algorithms,
such as Yannakakis'. On the other hand, the latter require an additional pass
over the input, which hinders adoption in production systems.
  In this paper, we make a step towards enabling single-pass bi-directional
information passing during query execution. We achieve this by statically
analyzing between which tables the information flow is blocked and by
leveraging precomputed join-induced fingerprint columns on FK-tables. On the
JOB benchmark, Parachute improves DuckDB v1.2's end-to-end execution time
without and with semi-join filtering by 1.54x and 1.24x, respectively, when
allowed to use 15% extra space.

### Distributed, Parallel, and Cluster Computing

### 1. [DDiT: Dynamic Resource Allocation for Diffusion Transformer Model Serving](http://arxiv.org/pdf/2506.13497v1)

Authors: Heyang Huang, Cunchen Hu, Jiaqi Zhu, Ziyuan Gao, Liangliang Xu, Yizhou Shan, Yungang Bao, Sun Ninghui, Tianwei Zhang, Sa Wang

The Text-to-Video (T2V) model aims to generate dynamic and expressive videos
from textual prompts. The generation pipeline typically involves multiple
modules, such as language encoder, Diffusion Transformer (DiT), and Variational
Autoencoders (VAE). Existing serving systems often rely on monolithic model
deployment, while overlooking the distinct characteristics of each module,
leading to inefficient GPU utilization. In addition, DiT exhibits varying
performance gains across different resolutions and degrees of parallelism, and
significant optimization potential remains unexplored. To address these
problems, we present DDiT, a flexible system that integrates both inter-phase
and intra-phase optimizations. DDiT focuses on two key metrics: optimal degree
of parallelism, which prevents excessive parallelism for specific resolutions,
and starvation time, which quantifies the sacrifice of each request. To this
end, DDiT introduces a decoupled control mechanism to minimize the
computational inefficiency caused by imbalances in the degree of parallelism
between the DiT and VAE phases. It also designs a greedy resource allocation
algorithm with a novel scheduling mechanism that operates at the single-step
granularity, enabling dynamic and timely resource scaling. Our evaluation on
the T5 encoder, OpenSora SDDiT, and OpenSora VAE models across diverse datasets
reveals that DDiT significantly outperforms state-of-the-art baselines by up to
1.44x in p99 latency and 1.43x in average latency.

### 2. [POPQC: Parallel Optimization for Quantum Circuits (Extended Version)](http://arxiv.org/pdf/2506.13720v1)

Authors: Pengyu Liu, Jatin Arora, Mingkuan Xu, Umut A. Acar

Optimization of quantum programs or circuits is a fundamental problem in
quantum computing and remains a major challenge. State-of-the-art quantum
circuit optimizers rely on heuristics and typically require superlinear, and
even exponential, time. Recent work proposed a new approach that pursues a
weaker form of optimality called local optimality. Parameterized by a natural
number $\Omega$, local optimality insists that each and every $\Omega$-segment
of the circuit is optimal with respect to an external optimizer, called the
oracle. Local optimization can be performed using only a linear number of calls
to the oracle but still incurs quadratic computational overheads in addition to
oracle calls. Perhaps most importantly, the algorithm is sequential.
  In this paper, we present a parallel algorithm for local optimization of
quantum circuits. To ensure efficiency, the algorithm operates by keeping a set
of fingers into the circuit and maintains the invariant that a $\Omega$-deep
circuit needs to be optimized only if it contains a finger. Operating in
rounds, the algorithm selects a set of fingers, optimizes in parallel the
segments containing the fingers, and updates the finger set to ensure the
invariant. For constant $\Omega$, we prove that the algorithm requires
$O(n\lg{n})$ work and $O(r\lg{n})$ span, where $n$ is the circuit size and $r$
is the number of rounds. We prove that the optimized circuit returned by the
algorithm is locally optimal in the sense that any $\Omega$-segment of the
circuit is optimal with respect to the oracle.

### 3. [BanditWare: A Contextual Bandit-based Framework for Hardware Prediction](http://arxiv.org/pdf/2506.13730v1)

Authors: Tainã Coleman, Hena Ahmed, Ravi Shende, Ismael Perez, Ïlkay Altintaş

Distributed computing systems are essential for meeting the demands of modern
applications, yet transitioning from single-system to distributed environments
presents significant challenges. Misallocating resources in shared systems can
lead to resource contention, system instability, degraded performance, priority
inversion, inefficient utilization, increased latency, and environmental
impact.
  We present BanditWare, an online recommendation system that dynamically
selects the most suitable hardware for applications using a contextual
multi-armed bandit algorithm. BanditWare balances exploration and exploitation,
gradually refining its hardware recommendations based on observed application
performance while continuing to explore potentially better options. Unlike
traditional statistical and machine learning approaches that rely heavily on
large historical datasets, BanditWare operates online, learning and adapting in
real-time as new workloads arrive.
  We evaluated BanditWare on three workflow applications: Cycles (an
agricultural science scientific workflow) BurnPro3D (a web-based platform for
fire science) and a matrix multiplication application. Designed for seamless
integration with the National Data Platform (NDP), BanditWare enables users of
all experience levels to optimize resource allocation efficiently.

### 4. [On Immutable Memory Systems for Artificial Agents: A Blockchain-Indexed Automata-Theoretic Framework Using ECDH-Keyed Merkle Chains](http://arxiv.org/pdf/2506.13246v1)

Authors: Craig Steven Wright

This paper presents a formalised architecture for synthetic agents designed
to retain immutable memory, verifiable reasoning, and constrained epistemic
growth. Traditional AI systems rely on mutable, opaque statistical models prone
to epistemic drift and historical revisionism. In contrast, we introduce the
concept of the Merkle Automaton, a cryptographically anchored, deterministic
computational framework that integrates formal automata theory with
blockchain-based commitments. Each agent transition, memory fragment, and
reasoning step is committed within a Merkle structure rooted on-chain,
rendering it non-repudiable and auditably permanent. To ensure selective access
and confidentiality, we derive symmetric encryption keys from ECDH exchanges
contextualised by hierarchical privilege lattices. This enforces cryptographic
access control over append-only DAG-structured knowledge graphs. Reasoning is
constrained by formal logic systems and verified through deterministic
traversal of policy-encoded structures. Updates are non-destructive and
historied, preserving epistemic lineage without catastrophic forgetting.
Zero-knowledge proofs facilitate verifiable, privacy-preserving inclusion
attestations. Collectively, this architecture reframes memory not as a cache
but as a ledger - one whose contents are enforced by protocol, bound by
cryptography, and constrained by formal logic. The result is not an intelligent
agent that mimics thought, but an epistemic entity whose outputs are provably
derived, temporally anchored, and impervious to post hoc revision. This design
lays foundational groundwork for legal, economic, and high-assurance
computational systems that require provable memory, unforgeable provenance, and
structural truth.

### 5. [Perfect Privacy for Discriminator-Based Byzantine-Resilient Federated Learning](http://arxiv.org/pdf/2506.13561v1)

Authors: Yue Xia, Christoph Hofmeister, Maximilian Egger, Rawad Bitar

Federated learning (FL) shows great promise in large-scale machine learning
but introduces new privacy and security challenges. We propose ByITFL and
LoByITFL, two novel FL schemes that enhance resilience against Byzantine users
while keeping the users' data private from eavesdroppers. To ensure privacy and
Byzantine resilience, our schemes build on having a small representative
dataset available to the federator and crafting a discriminator function
allowing the mitigation of corrupt users' contributions. ByITFL employs
Lagrange coded computing and re-randomization, making it the first
Byzantine-resilient FL scheme with perfect Information-Theoretic (IT) privacy,
though at the cost of a significant communication overhead. LoByITFL, on the
other hand, achieves Byzantine resilience and IT privacy at a significantly
reduced communication cost, but requires a Trusted Third Party, used only in a
one-time initialization phase before training. We provide theoretical
guarantees on privacy and Byzantine resilience, along with convergence
guarantees and experimental results validating our findings.

### 6. [EBS-CFL: Efficient and Byzantine-robust Secure Clustered Federated Learning](http://arxiv.org/pdf/2506.13612v1)

Authors: Zhiqiang Li, Haiyong Bao, Menghong Guan, Hao Pan, Cheng Huang, Hong-Ning Dai

Despite federated learning (FL)'s potential in collaborative learning, its
performance has deteriorated due to the data heterogeneity of distributed
users. Recently, clustered federated learning (CFL) has emerged to address this
challenge by partitioning users into clusters according to their similarity.
However, CFL faces difficulties in training when users are unwilling to share
their cluster identities due to privacy concerns. To address these issues, we
present an innovative Efficient and Robust Secure Aggregation scheme for CFL,
dubbed EBS-CFL. The proposed EBS-CFL supports effectively training CFL while
maintaining users' cluster identity confidentially. Moreover, it detects
potential poisonous attacks without compromising individual client gradients by
discarding negatively correlated gradients and aggregating positively
correlated ones using a weighted approach. The server also authenticates
correct gradient encoding by clients. EBS-CFL has high efficiency with
client-side overhead O(ml + m^2) for communication and O(m^2l) for computation,
where m is the number of cluster identities, and l is the gradient size. When m
= 1, EBS-CFL's computational efficiency of client is at least O(log n) times
better than comparison schemes, where n is the number of clients.In addition,
we validate the scheme through extensive experiments. Finally, we theoretically
prove the scheme's security.

### Digital Libraries

### 1. [Implicit and Explicit Research Quality Score Probabilities from ChatGPT](http://arxiv.org/pdf/2506.13525v1)

Authors: Mike Thelwall, Yunhan Yang

The large language model (LLM) ChatGPT's quality scores for journal articles
correlate more strongly with human judgements than some citation-based
indicators in most fields. Averaging multiple ChatGPT scores improves the
results, apparently leveraging its internal probability model. To leverage
these probabilities, this article tests two novel strategies: requesting
percentage likelihoods for scores and extracting the probabilities of
alternative tokens in the responses. The probability estimates were then used
to calculate weighted average scores. Both strategies were evaluated with five
iterations of ChatGPT 4o-mini on 96,800 articles submitted to the UK Research
Excellence Framework (REF) 2021, using departmental average REF2021 quality
scores as a proxy for article quality. The data was analysed separately for
each of the 34 field-based REF Units of Assessment. For the first strategy,
explicit requests for tables of score percentage likelihoods substantially
decreased the value of the scores (lower correlation with the proxy quality
indicator). In contrast, weighed averages of score token probabilities slightly
increased the correlation with the quality proxy indicator and these
probabilities reasonably accurately reflected ChatGPT's outputs. The token
probability approach is therefore the most accurate method for ranking articles
by research quality as well as being cheaper than comparable ChatGPT
strategies.

### 2. [Accessibility Barriers in Multi-Terabyte Public Datasets: The Gap Between Promise and Practice](http://arxiv.org/pdf/2506.13256v1)

Authors: Marc Bara

The promise of "free and open" multi-terabyte datasets often collides with
harsh realities. While these datasets may be technically accessible, practical
barriers -- from processing complexity to hidden costs -- create a system that
primarily serves well-funded institutions. This study examines accessibility
challenges across web crawls, satellite imagery, scientific data, and
collaborative projects, revealing a consistent two-tier system where
theoretical openness masks practical exclusivity. Our analysis demonstrates
that datasets marketed as "publicly accessible" typically require minimum
investments of \$1,000+ for meaningful analysis, with complex processing
pipelines demanding \$10,000-100,000+ in infrastructure costs. The
infrastructure requirements -- distributed computing knowledge, domain
expertise, and substantial budgets -- effectively gatekeep these datasets
despite their "open" status, limiting practical accessibility to those with
institutional support or substantial resources.

### Discrete Mathematics

### 1. [The Combinatorial Rank of Subsets: Metric Density in Finite Hamming Spaces](http://arxiv.org/pdf/2506.13081v1)

Authors: Jamolidin K. Abdurakhmanov

We introduce a novel concept of rank for subsets of finite metric spaces
E^n_q (the set of all n-dimensional vectors over an alphabet of size q)
equipped with the Hamming distance, where the rank R(A) of a subset A is
defined as the number of non-constant columns in the matrix formed by the
vectors of A. This purely combinatorial definition provides a new perspective
on the structure of finite metric spaces, distinct from traditional
linear-algebraic notions of rank. We establish tight bounds for R(A) in terms
of D_A, the sum of Hamming distances between all pairs of elements in A.
Specifically, we prove that 2qD_A/((q-1)|A|^2) <= R(A) <= D_A/(|A|-1) when
|A|/q >= 1, with a modified lower bound for the case |A|/q < 1. These bounds
show that the rank is constrained by the metric properties of the subset.
Furthermore, we introduce the concept of metrically dense subsets, which are
subsets that minimize rank among all isometric images. This notion captures an
extremal property of subsets that represent their distance structure in the
most compact way possible. We prove that subsets with uniform column
distribution are metrically dense, and as a special case, establish that when q
is a prime power, every linear subspace of E^n_q is metrically dense. This
reveals a fundamental connection between the algebraic and metric structures of
these spaces.

### Data Structures and Algorithms

### 1. [Ultra-Resilient Superimposed Codes: Near-Optimal Construction and Applications](http://arxiv.org/pdf/2506.13489v1)

Authors: Gianluca De Marco, Dariusz R. Kowalski

A superimposed code is a collection of binary vectors (codewords) with the
property that no vector is contained in the Boolean sum of any $k$ others,
enabling unique identification of codewords within any group of $k$.
Superimposed codes are foundational combinatorial tools with applications in
areas ranging from distributed computing and data retrieval to fault-tolerant
communication. However, classical superimposed codes rely on strict alignment
assumptions, limiting their effectiveness in asynchronous and fault-prone
environments, which are common in modern systems and applications.
  We introduce Ultra-Resilient Superimposed Codes (URSCs), a new class of codes
that extends the classic superimposed framework by ensuring a stronger
codewords' isolation property and resilience to two types of adversarial
perturbations: arbitrary cyclic shifts and partial bitwise corruption (flips).
Additionally, URSCs exhibit universality, adapting seamlessly to any number $k$
of concurrent codewords without prior knowledge. This is a combination of
properties not achieved in any previous construction.
  We provide the first polynomial-time construction of URSCs with near-optimal
length, significantly outperforming previous constructions with less general
features, all without requiring prior knowledge of the number of concurrent
codewords, $k$. % We demonstrate that our URSCs significantly advance the state
of the art in multiple applications, including uncoordinated beeping networks,
where our codes reduce time complexity for local broadcast by nearly two orders
of magnitude, and generalized contention resolution in multi-access channel
communication.

### 2. [Stochastic Multi-Objective Multi-Armed Bandits: Regret Definition and Algorithm](http://arxiv.org/pdf/2506.13125v1)

Authors: Mansoor Davoodi, Setareh Maghsudi

Multi-armed bandit (MAB) problems are widely applied to online optimization
tasks that require balancing exploration and exploitation. In practical
scenarios, these tasks often involve multiple conflicting objectives, giving
rise to multi-objective multi-armed bandits (MO-MAB). Existing MO-MAB
approaches predominantly rely on the Pareto regret metric introduced in
\cite{drugan2013designing}. However, this metric has notable limitations,
particularly in accounting for all Pareto-optimal arms simultaneously. To
address these challenges, we propose a novel and comprehensive regret metric
that ensures balanced performance across conflicting objectives. Additionally,
we introduce the concept of \textit{Efficient Pareto-Optimal} arms, which are
specifically designed for online optimization. Based on our new metric, we
develop a two-phase MO-MAB algorithm that achieves sublinear regret for both
Pareto-optimal and efficient Pareto-optimal arms.

### 3. [FPT Constant Approximation Algorithms for Colorful Sum of Radii](http://arxiv.org/pdf/2506.13191v1)

Authors: Shuilian Liu, Gregory Gutin, Yicheng Xu, Yong Zhang

We study the colorful sum of radii problem, where the input is a point set
$P$ partitioned into classes $P_1, P_2, \dots, P_\omega$, along with per-class
outlier bounds $m_1, m_2, \dots, m_\omega$, summing to $m$. The goal is to
select a subset $\mathcal{C} \subseteq P$ of $k$ centers and assign points to
centers in $\mathcal{C}$, allowing up to $m_i$ unassigned points (outliers)
from each class $P_i$, while minimizing the sum of cluster radii. The radius of
a cluster is defined as the maximum distance from any point in the cluster to
its center. The classical (non-colorful) version of the sum of radii problem is
known to be NP-hard, even on weighted planar graphs. The colorful sum of radii
is introduced by Chekuri et al. (2022), who provide an $O(\log
\omega)$-approximation algorithm. In this paper, we present the first
constant-factor approximation algorithms for the colorful sum of radii running
in FPT (fixed-parameter tractable) time. Our contributions are twofold: We
design an iterative covering algorithm that achieves a
$(2+\varepsilon)$-approximation with running time exponential in both $k$ and
$m$; We further develop a $(7+\varepsilon)$-approximation algorithm by
leveraging a colorful $k$-center subroutine, improving the running time by
removing the exponential dependency on $m$.

### 4. [Learning Augmented Graph $k$-Clustering](http://arxiv.org/pdf/2506.13533v1)

Authors: Chenglin Fan, Kijun Shin

Clustering is a fundamental task in unsupervised learning. Previous research
has focused on learning-augmented $k$-means in Euclidean metrics, limiting its
applicability to complex data representations. In this paper, we generalize
learning-augmented $k$-clustering to operate on general metrics, enabling its
application to graph-structured and non-Euclidean domains. Our framework also
relaxes restrictive cluster size constraints, providing greater flexibility for
datasets with imbalanced or unknown cluster distributions. Furthermore, we
extend the hardness of query complexity to general metrics: under the
Exponential Time Hypothesis (ETH), we show that any polynomial-time algorithm
must perform approximately $\Omega(k / \alpha)$ queries to achieve a $(1 +
\alpha)$-approximation. These contributions strengthen both the theoretical
foundations and practical applicability of learning-augmented clustering,
bridging gaps between traditional methods and real-world challenges.

### 5. [Efficient Approximate Temporal Triangle Counting in Streaming with Predictions](http://arxiv.org/pdf/2506.13173v1)

Authors: Giorgio Venturin, Ilie Sarpe, Fabio Vandin

Triangle counting is a fundamental and widely studied problem on static
graphs, and recently on temporal graphs, where edges carry information on the
timings of the associated events. Streaming processing and resource efficiency
are crucial requirements for counting triangles in modern massive temporal
graphs, with millions of nodes and up to billions of temporal edges. However,
current exact and approximate algorithms are unable to handle large-scale
temporal graphs. To fill such a gap, we introduce STEP, a scalable and
efficient algorithm to approximate temporal triangle counts from a stream of
temporal edges. STEP combines predictions to the number of triangles a temporal
edge is involved in, with a simple sampling strategy, leading to scalability,
efficiency, and accurate approximation of all eight temporal triangle types
simultaneously. We analytically prove that, by using a sublinear amount of
memory, STEP obtains unbiased and very accurate estimates. In fact, even noisy
predictions can significantly reduce the variance of STEP's estimates. Our
extensive experiments on massive temporal graphs with up to billions of edges
demonstrate that STEP outputs high-quality estimates and is more efficient than
state-of-the-art methods.

### 6. [Avoiding Obfuscation with Prover-Estimator Debate](http://arxiv.org/pdf/2506.13609v1)

Authors: Jonah Brown-Cohen, Geoffrey Irving, Georgios Piliouras

Training powerful AI systems to exhibit desired behaviors hinges on the
ability to provide accurate human supervision on increasingly complex tasks. A
promising approach to this problem is to amplify human judgement by leveraging
the power of two competing AIs in a debate about the correct solution to a
given problem. Prior theoretical work has provided a complexity-theoretic
formalization of AI debate, and posed the problem of designing protocols for AI
debate that guarantee the correctness of human judgements for as complex a
class of problems as possible. Recursive debates, in which debaters decompose a
complex problem into simpler subproblems, hold promise for growing the class of
problems that can be accurately judged in a debate. However, existing protocols
for recursive debate run into the obfuscated arguments problem: a dishonest
debater can use a computationally efficient strategy that forces an honest
opponent to solve a computationally intractable problem to win. We mitigate
this problem with a new recursive debate protocol that, under certain stability
assumptions, ensures that an honest debater can win with a strategy requiring
computational efficiency comparable to their opponent.

### Emerging Technologies

### 1. [lcpy: an open-source python package for parametric and dynamic Life Cycle Assessment and Life Cycle Costing](http://arxiv.org/pdf/2506.13744v1)

Authors: Spiros Gkousis, Evina Katsou

This article describes lcpy, an open-source python package that allows for
advanced parametric Life Cycle Assessment (LCA) and Life Cycle Costing (LCC)
analysis. The package is designed to allow the user to model a process with a
flexible, modular design based on dictionaries and lists. The modeling can
consider in-time variations, uncertainty, and allows for dynamic analysis,
uncertainty assessment, as well as conventional static LCA and LCC. The package
is compatible with optimization and uncertainty analysis libraries as well as
python packages for prospective LCA. Its goal is to allow for easy
implementation of dynamic LCA and LCC and for simple integration with tools for
uncertainty assessment and optimization towards a more widened implementation
of advanced enviro-economic analysis. The open-source code can be found at
https://github.com/spirdgk/lcpy.

### 2. [Model Context Protocol (MCP) at First Glance: Studying the Security and Maintainability of MCP Servers](http://arxiv.org/pdf/2506.13538v1)

Authors: Mohammed Mehedi Hasan, Hao Li, Emad Fallahzadeh, Bram Adams, Ahmed E. Hassan

Although Foundation Models (FMs), such as GPT-4, are increasingly used in
domains like finance and software engineering, reliance on textual interfaces
limits these models' real-world interaction. To address this, FM providers
introduced tool calling-triggering a proliferation of frameworks with distinct
tool interfaces. In late 2024, Anthropic introduced the Model Context Protocol
(MCP) to standardize this tool ecosystem, which has become the de facto
standard with over eight million weekly SDK downloads. Despite its adoption,
MCP's AI-driven, non-deterministic control flow introduces new risks to
sustainability, security, and maintainability, warranting closer examination.
  Towards this end, we present the first large-scale empirical study of MCP.
Using state-of-the-art health metrics and a hybrid analysis pipeline, combining
a general-purpose static analysis tool with an MCP-specific scanner, we
evaluate 1,899 open-source MCP servers to assess their health, security, and
maintainability. Despite MCP servers demonstrating strong health metrics, we
identify eight distinct vulnerabilities-only three overlapping with traditional
software vulnerabilities. Additionally, 7.2% of servers contain general
vulnerabilities and 5.5% exhibit MCP-specific tool poisoning. Regarding
maintainability, while 66% exhibit code smells, 14.4% contain ten bug patterns
overlapping prior research. These findings highlight the need for MCP-specific
vulnerability detection techniques while reaffirming the value of traditional
analysis and refactoring practices.

### 3. [UAV Object Detection and Positioning in a Mining Industrial Metaverse with Custom Geo-Referenced Data](http://arxiv.org/pdf/2506.13505v1)

Authors: Vasiliki Balaska, Ioannis Tsampikos Papapetros, Katerina Maria Oikonomou, Loukas Bampis, Antonios Gasteratos

The mining sector increasingly adopts digital tools to improve operational
efficiency, safety, and data-driven decision-making. One of the key challenges
remains the reliable acquisition of high-resolution, geo-referenced spatial
information to support core activities such as extraction planning and on-site
monitoring. This work presents an integrated system architecture that combines
UAV-based sensing, LiDAR terrain modeling, and deep learning-based object
detection to generate spatially accurate information for open-pit mining
environments. The proposed pipeline includes geo-referencing, 3D
reconstruction, and object localization, enabling structured spatial outputs to
be integrated into an industrial digital twin platform. Unlike traditional
static surveying methods, the system offers higher coverage and automation
potential, with modular components suitable for deployment in real-world
industrial contexts. While the current implementation operates in post-flight
batch mode, it lays the foundation for real-time extensions. The system
contributes to the development of AI-enhanced remote sensing in mining by
demonstrating a scalable and field-validated geospatial data workflow that
supports situational awareness and infrastructure safety.

### Formal Languages and Automata Theory

### 1. [Saturation Problems for Families of Automata](http://arxiv.org/pdf/2506.13197v1)

Authors: León Bohn, Yong Li, Christof Löding, Sven Schewe

Families of deterministic finite automata (FDFA) represent regular
$\omega$-languages through their ultimately periodic words (UP-words). An FDFA
accepts pairs of words, where the first component corresponds to a prefix of
the UP-word, and the second component represents a period of that UP-word. An
FDFA is termed saturated if, for each UP-word, either all or none of the pairs
representing that UP-word are accepted. We demonstrate that determining whether
a given FDFA is saturated can be accomplished in polynomial time, thus
improving the known PSPACE upper bound by an exponential. We illustrate the
application of this result by presenting the first polynomial learning
algorithms for representations of the class of all regular $\omega$-languages.
Furthermore, we establish that deciding a weaker property, referred to as
almost saturation, is PSPACE-complete. Since FDFAs do not necessarily define
regular $\omega$-languages when they are not saturated, we also address the
regularity problem and show that it is PSPACE-complete. Finally, we explore a
variant of FDFAs called families of deterministic weak automata (FDWA), where
the semantics for the periodic part of the UP-word considers $\omega$-words
instead of finite words. We demonstrate that saturation for FDWAs is also
decidable in polynomial time, that FDWAs always define regular
$\omega$-languages, and we compare the succinctness of these different models.

### 2. [Probabilistic Modeling of Spiking Neural Networks with Contract-Based Verification](http://arxiv.org/pdf/2506.13340v1)

Authors: Zhen Yao, Elisabetta De Maria, Robert De Simone

Spiking Neural Networks (SNN) are models for "realistic" neuronal
computation, which makes them somehow different in scope from "ordinary"
deep-learning models widely used in AI platforms nowadays. SNNs focus on timed
latency (and possibly probability) of neuronal reactive activation/response,
more than numerical computation of filters. So, an SNN model must provide
modeling constructs for elementary neural bundles and then for synaptic
connections to assemble them into compound data flow network patterns. These
elements are to be parametric patterns, with latency and probability values
instantiated on particular instances (while supposedly constant "at runtime").
Designers could also use different values to represent "tired" neurons, or ones
impaired by external drugs, for instance. One important challenge in such
modeling is to study how compound models could meet global reaction
requirements (in stochastic timing challenges), provided similar provisions on
individual neural bundles. A temporal language of logic to express such
assume/guarantee contracts is thus needed. This may lead to formal verification
on medium-sized models and testing observations on large ones. In the current
article, we make preliminary progress at providing a simple model framework to
express both elementary SNN neural bundles and their connecting constructs,
which translates readily into both a model-checker and a simulator (both
already existing and robust) to conduct experiments.

### Graphics

### 1. [NeuVAS: Neural Implicit Surfaces for Variational Shape Modeling](http://arxiv.org/pdf/2506.13050v1)

Authors: Pengfei Wang, Qiujie Dong, Fangtian Liang, Hao Pan, Lei Yang, Congyi Zhang, Guying Lin, Caiming Zhang, Yuanfeng Zhou, Changhe Tu, Shiqing Xin, Alla Sheffer, Xin Li, Wenping Wang

Neural implicit shape representation has drawn significant attention in
recent years due to its smoothness, differentiability, and topological
flexibility. However, directly modeling the shape of a neural implicit surface,
especially as the zero-level set of a neural signed distance function (SDF),
with sparse geometric control is still a challenging task. Sparse input shape
control typically includes 3D curve networks or, more generally, 3D curve
sketches, which are unstructured and cannot be connected to form a curve
network, and therefore more difficult to deal with. While 3D curve networks or
curve sketches provide intuitive shape control, their sparsity and varied
topology pose challenges in generating high-quality surfaces to meet such curve
constraints. In this paper, we propose NeuVAS, a variational approach to shape
modeling using neural implicit surfaces constrained under sparse input shape
control, including unstructured 3D curve sketches as well as connected 3D curve
networks. Specifically, we introduce a smoothness term based on a functional of
surface curvatures to minimize shape variation of the zero-level set surface of
a neural SDF. We also develop a new technique to faithfully model G0 sharp
feature curves as specified in the input curve sketches. Comprehensive
comparisons with the state-of-the-art methods demonstrate the significant
advantages of our method.

### 2. [Volumetric Functional Maps](http://arxiv.org/pdf/2506.13212v1)

Authors: Filippo Maggioli, Marco Livesu, Simone Melzi

The computation of volumetric correspondences between 3D shapes has great
potential for medical and industrial applications. In this work, we pave the
way for spectral volume mapping, extending for the first time the functional
maps framework from the surface setting to the volumetric domain. We show that
the eigenfunctions of the volumetric Laplace operator define a functional space
that is suitable for high-quality signal transfer. We also experiment with
various techniques that edit this functional space, porting them from the
surface to the volume setting. We validate our method on novel volumetric
datasets and on tetrahedralizations of well established surface datasets, also
showcasing practical applications involving both discrete and continuous signal
mapping, for segmentation transfer, mesh connectivity transfer and solid
texturing. Last but not least, we show that considering the volumetric spectrum
greatly improves the accuracy for classical shape matching tasks among
surfaces, consistently outperforming existing surface-only spectral methods.

### 3. [TextureSplat: Per-Primitive Texture Mapping for Reflective Gaussian Splatting](http://arxiv.org/pdf/2506.13348v1)

Authors: Mae Younes, Adnane Boukhayma

Gaussian Splatting have demonstrated remarkable novel view synthesis
performance at high rendering frame rates. Optimization-based inverse rendering
within complex capture scenarios remains however a challenging problem. A
particular case is modelling complex surface light interactions for highly
reflective scenes, which results in intricate high frequency specular radiance
components. We hypothesize that such challenging settings can benefit from
increased representation power. We hence propose a method that tackles this
issue through a geometrically and physically grounded Gaussian Splatting borne
radiance field, where normals and material properties are spatially variable in
the primitive's local space. Using per-primitive texture maps for this purpose,
we also propose to harness the GPU hardware to accelerate rendering at test
time via unified material texture atlas.

### 4. [UltraZoom: Generating Gigapixel Images from Regular Photos](http://arxiv.org/pdf/2506.13756v1)

Authors: Jingwei Ma, Vivek Jayaram, Brian Curless, Ira Kemelmacher-Shlizerman, Steven M. Seitz

We present UltraZoom, a system for generating gigapixel-resolution images of
objects from casually captured inputs, such as handheld phone photos. Given a
full-shot image (global, low-detail) and one or more close-ups (local,
high-detail), UltraZoom upscales the full image to match the fine detail and
scale of the close-up examples. To achieve this, we construct a per-instance
paired dataset from the close-ups and adapt a pretrained generative model to
learn object-specific low-to-high resolution mappings. At inference, we apply
the model in a sliding window fashion over the full image. Constructing these
pairs is non-trivial: it requires registering the close-ups within the full
image for scale estimation and degradation alignment. We introduce a simple,
robust method for getting registration on arbitrary materials in casual,
in-the-wild captures. Together, these components form a system that enables
seamless pan and zoom across the entire object, producing consistent,
photorealistic gigapixel imagery from minimal input.

### Computer Science and Game Theory

### 1. [One-dimensional vs. Multi-dimensional Pricing in Blockchain Protocols](http://arxiv.org/pdf/2506.13271v1)

Authors: Aggelos Kiayias, Elias Koutsoupias, Giorgos Panagiotakos, Kyriaki Zioga

Blockchain transactions consume diverse resources, foremost among them
storage, but also computation, communication, and others. Efficiently charging
for these resources is crucial for effective system resource allocation and
long-term economic viability. The prevailing approach, one-dimensional pricing,
sets a single price for a linear combination of resources. However, this often
leads to under-utilization when resource capacities are limited.
Multi-dimensional pricing, which independently prices each resource, offers an
alternative but presents challenges in price discovery.
  This work focuses on the welfare achieved by these two schemes. We prove that
multi-dimensional pricing is superior under stable blockchain conditions.
Conversely, we show that one-dimensional pricing outperforms its
multi-dimensional counterpart in transient states, exhibiting faster
convergence and greater computational tractability. These results highlight a
critical trade-off: while multi-dimensional pricing offers efficiency gains at
equilibrium, its implementation incurs costs associated with system
transitions. Our findings underscore the necessity for a deeper understanding
of these transient effects before widespread adoption. Finally, we propose
mechanisms that aim to mitigate some of these issues, paving the way for future
research.

### 2. [Fast and Furious Symmetric Learning in Zero-Sum Games: Gradient Descent as Fictitious Play](http://arxiv.org/pdf/2506.13086v1)

Authors: John Lazarsfeld, Georgios Piliouras, Ryann Sim, Andre Wibisono

This paper investigates the sublinear regret guarantees of two non-no-regret
algorithms in zero-sum games: Fictitious Play, and Online Gradient Descent with
constant stepsizes. In general adversarial online learning settings, both
algorithms may exhibit instability and linear regret due to no regularization
(Fictitious Play) or small amounts of regularization (Gradient Descent).
However, their ability to obtain tighter regret bounds in two-player zero-sum
games is less understood. In this work, we obtain strong new regret guarantees
for both algorithms on a class of symmetric zero-sum games that generalize the
classic three-strategy Rock-Paper-Scissors to a weighted, n-dimensional regime.
Under symmetric initializations of the players' strategies, we prove that
Fictitious Play with any tiebreaking rule has $O(\sqrt{T})$ regret,
establishing a new class of games for which Karlin's Fictitious Play conjecture
holds. Moreover, by leveraging a connection between the geometry of the
iterates of Fictitious Play and Gradient Descent in the dual space of payoff
vectors, we prove that Gradient Descent, for almost all symmetric
initializations, obtains a similar $O(\sqrt{T})$ regret bound when its stepsize
is a sufficiently large constant. For Gradient Descent, this establishes the
first "fast and furious" behavior (i.e., sublinear regret without
time-vanishing stepsizes) for zero-sum games larger than 2x2.

### 3. [Real Time Self-Tuning Adaptive Controllers on Temperature Control Loops using Event-based Game Theory](http://arxiv.org/pdf/2506.13164v1)

Authors: Steve Yuwono, Muhammad Uzair Rana, Dorothea Schwung, Andreas Schwung

This paper presents a novel method for enhancing the adaptability of
Proportional-Integral-Derivative (PID) controllers in industrial systems using
event-based dynamic game theory, which enables the PID controllers to
self-learn, optimize, and fine-tune themselves. In contrast to conventional
self-learning approaches, our proposed framework offers an event-driven control
strategy and game-theoretic learning algorithms. The players collaborate with
the PID controllers to dynamically adjust their gains in response to set point
changes and disturbances. We provide a theoretical analysis showing sound
convergence guarantees for the game given suitable stability ranges of the PID
controlled loop. We further introduce an automatic boundary detection
mechanism, which helps the players to find an optimal initialization of action
spaces and significantly reduces the exploration time. The efficacy of this
novel methodology is validated through its implementation in the temperature
control loop of a printing press machine. Eventually, the outcomes of the
proposed intelligent self-tuning PID controllers are highly promising,
particularly in terms of reducing overshoot and settling time.

### 4. [A Game-Theoretic Negotiation Framework for Cross-Cultural Consensus in LLMs](http://arxiv.org/pdf/2506.13245v1)

Authors: Guoxi Zhang, Jiawei Chen, Tianzhuo Yang, Jiaming Ji, Yaodong Yang, Juntao Dai

The increasing prevalence of large language models (LLMs) is influencing
global value systems. However, these models frequently exhibit a pronounced
WEIRD (Western, Educated, Industrialized, Rich, Democratic) cultural bias due
to lack of attention to minority values. This monocultural perspective may
reinforce dominant values and marginalize diverse cultural viewpoints, posing
challenges for the development of equitable and inclusive AI systems. In this
work, we introduce a systematic framework designed to boost fair and robust
cross-cultural consensus among LLMs. We model consensus as a Nash Equilibrium
and employ a game-theoretic negotiation method based on Policy-Space Response
Oracles (PSRO) to simulate an organized cross-cultural negotiation process. To
evaluate this approach, we construct regional cultural agents using data
transformed from the World Values Survey (WVS). Beyond the conventional
model-level evaluation method, We further propose two quantitative metrics,
Perplexity-based Acceptence and Values Self-Consistency, to assess consensus
outcomes. Experimental results indicate that our approach generates consensus
of higher quality while ensuring more balanced compromise compared to
baselines. Overall, it mitigates WEIRD bias by guiding agents toward
convergence through fair and gradual negotiation steps.

### 5. [The impact of uncertainty on regularized learning in games](http://arxiv.org/pdf/2506.13286v1)

Authors: Pierre-Louis Cauvin, Davide Legacci, Panayotis Mertikopoulos

In this paper, we investigate how randomness and uncertainty influence
learning in games. Specifically, we examine a perturbed variant of the dynamics
of "follow-the-regularized-leader" (FTRL), where the players' payoff
observations and strategy updates are continually impacted by random shocks.
Our findings reveal that, in a fairly precise sense, "uncertainty favors
extremes": in any game, regardless of the noise level, every player's
trajectory of play reaches an arbitrarily small neighborhood of a pure strategy
in finite time (which we estimate). Moreover, even if the player does not
ultimately settle at this strategy, they return arbitrarily close to some
(possibly different) pure strategy infinitely often. This prompts the question
of which sets of pure strategies emerge as robust predictions of learning under
uncertainty. We show that (a) the only possible limits of the FTRL dynamics
under uncertainty are pure Nash equilibria; and (b) a span of pure strategies
is stable and attracting if and only if it is closed under better replies.
Finally, we turn to games where the deterministic dynamics are recurrent - such
as zero-sum games with interior equilibria - and we show that randomness
disrupts this behavior, causing the stochastic dynamics to drift toward the
boundary on average.

### 6. [Deceptive Path Planning: A Bayesian Game Approach](http://arxiv.org/pdf/2506.13650v1)

Authors: Violetta Rostobaya, James Berneburg, Yue Guan, Michael Dorothy, Daigo Shishika

This paper investigates how an autonomous agent can transmit information
through its motion in an adversarial setting. We consider scenarios where an
agent must reach its goal while deceiving an intelligent observer about its
destination. We model this interaction as a dynamic Bayesian game between a
mobile Attacker with a privately known goal and a Defender who infers the
Attacker's intent to allocate defensive resources effectively. We use Perfect
Bayesian Nash Equilibrium (PBNE) as our solution concept and propose a
computationally efficient approach to find it. In the resulting equilibrium,
the Defender employs a simple Markovian strategy, while the Attacker
strategically balances deception and goal efficiency by stochastically mixing
shortest and non-shortest paths to manipulate the Defender's beliefs. Numerical
experiments demonstrate the advantages of our PBNE-based strategies over
existing methods based on one-sided optimization.

### Human-Computer Interaction

### 1. [ChartBlender: An Interactive System for Authoring and Synchronizing Visualization Charts in Video](http://arxiv.org/pdf/2506.13129v1)

Authors: Yi He, Yuqi Liu, Chenpu Li, Ruoyan Chen, Chuer Chen, Shengqi Dang, Nan Cao

Embedding data visualizations in video can enhance the communication of
complex information. However, this process is often labor-intensive, requiring
designers to adjust visualizations frame by frame manually. In this work, we
present ChartBlender, a novel system that streamlines this process by enabling
users to create data visualizations, embed them seamlessly into video scenes,
and automatically synchronize them with both camera motion and moving objects.
Particularly, ChartBlender incorporates a tracking algorithm that supports both
object and camera tracking, ensuring robust alignment of visualizations with
dynamic video content. To maintain visual clarity and aesthetic coherence, we
also explore the design space of video-suited visualizations and develop a
library of customizable templates optimized for video embedding. We evaluate
\oursName\ChartBlender through two controlled experiments and expert interviews
with five domain experts. Results show that our system enables accurate
synchronization and accelerates the production of data-driven videos.

### 2. [Screen Reader Users in the Vibe Coding Era: Adaptation, Empowerment, and New Accessibility Landscape](http://arxiv.org/pdf/2506.13270v1)

Authors: Nan Chen, Luna K. Qiu, Arran Zeyu Wang, Zilong Wang, Yuqing Yang

The rise of generative AI agents has reshaped human-computer interaction and
computer-supported cooperative work by shifting users' roles from direct task
execution to supervising machine-driven actions, especially in programming
(e.g., "vibe coding"). However, there is limited understanding of how screen
reader users engage with these systems in practice. To address this gap, we
conducted a longitudinal study with 16 screen reader users, exploring their
experiences with AI code assistants in daily programming scenarios.
Participants first completed a tutorial with GitHub Copilot, then performed a
programming task and provided initial feedback. After two weeks of AI-assisted
programming, follow-up studies assessed changes in their practices and
perceptions. Our findings demonstrate that advanced code assistants not only
enhance their programming capabilities but also bridge accessibility gaps.
While the assistant proved beneficial, there remains potential to improve how
users convey intent and interpret outputs. They also experienced difficulties
managing multiple views and maintaining situational awareness. More broadly,
they encountered barriers in learning advanced tools and expressed a need to
retain control. Based on these insights, we provide design recommendations for
more accessible and inclusive AI-assisted tools.

### 3. [Enhancing Orthopedic Surgical Training With Interactive Photorealistic 3D Visualization](http://arxiv.org/pdf/2506.13389v1)

Authors: Roni Lekar, Tatiana Gerth, Sergey Prokudin, Matthias Seibold, Reto Bürgin, Benjamin Vella, Armando Hoch, Siyu Tang, Philipp Fürnstahl, Helmut Grabner

Surgical training integrates several years of didactic learning, simulation,
mentorship, and hands-on experience. Challenges include stress, technical
demands, and new technologies. Orthopedic education often uses static materials
like books, images, and videos, lacking interactivity. This study compares a
new interactive photorealistic 3D visualization to 2D videos for learning total
hip arthroplasty. In a randomized controlled trial, participants (students and
residents) were evaluated on spatial awareness, tool placement, and task times
in a simulation. Results show that interactive photorealistic 3D visualization
significantly improved scores, with residents and those with prior 3D
experience performing better. These results emphasize the potential of the
interactive photorealistic 3D visualization to enhance orthopedic training.

### 4. [CHARM: Considering Human Attributes for Reinforcement Modeling](http://arxiv.org/pdf/2506.13079v1)

Authors: Qidi Fang, Hang Yu, Shijie Fang, Jindan Huang, Qiuyu Chen, Reuben M. Aronson, Elaine S. Short

Reinforcement Learning from Human Feedback has recently achieved significant
success in various fields, and its performance is highly related to feedback
quality. While much prior work acknowledged that human teachers'
characteristics would affect human feedback patterns, there is little work that
has closely investigated the actual effects. In this work, we designed an
exploratory study investigating how human feedback patterns are associated with
human characteristics. We conducted a public space study with two long horizon
tasks and 46 participants. We found that feedback patterns are not only
correlated with task statistics, such as rewards, but also correlated with
participants' characteristics, especially robot experience and educational
background. Additionally, we demonstrated that human feedback value can be more
accurately predicted with human characteristics compared to only using task
statistics. All human feedback and characteristics we collected, and codes for
our data collection and predicting more accurate human feedback are available
at https://github.com/AABL-Lab/CHARM

### 5. [Multimodal "Puppeteer": An Exploration of Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality](http://arxiv.org/pdf/2506.13189v1)

Authors: Yuchong Zhang, Bastian Orthmann, Shichen Ji, Michael Welle, Jonne Van Haastregt, Danica Kragic

The integration of robotics and augmented reality (AR) holds transformative
potential for advancing human-robot interaction (HRI), offering enhancements in
usability, intuitiveness, accessibility, and collaborative task performance.
This paper introduces and evaluates a novel multimodal AR-based robot puppeteer
framework that enables intuitive teleoperation via virtual counterpart through
large language model (LLM)-driven voice commands and hand gesture interactions.
Utilizing the Meta Quest 3, users interact with a virtual counterpart robot in
real-time, effectively "puppeteering" its physical counterpart within an AR
environment. We conducted a within-subject user study with 42 participants
performing robotic cube pick-and-place with pattern matching tasks under two
conditions: gesture-only interaction and combined voice-and-gesture
interaction. Both objective performance metrics and subjective user experience
(UX) measures were assessed, including an extended comparative analysis between
roboticists and non-roboticists. The results provide key insights into how
multimodal input influences contextual task efficiency, usability, and user
satisfaction in AR-based HRI. Our findings offer practical design implications
for designing effective AR-enhanced HRI systems.

### 6. [The Transition Matrix -- A classification of navigational patterns between LMS course sections](http://arxiv.org/pdf/2506.13275v1)

Authors: Tobias Hildebrandt, Lars Mehnen

Learning management systems (LMS) like Moodle are increasingly used to
support university teaching. As Moodle courses become more complex,
incorporating diverse interactive elements, it is important to understand how
students navigate through course sections and whether course designs are
meeting student needs. While substantial research exists on student usage of
individual LMS elements, there is a lack of research on broader navigational
patterns between course sections and how these patterns differ across courses.
This study analyzes navigational data from 747 courses in the Moodle LMS at a
technical university of applied sciences, representing (after filtering) around
4,400 students and 1.8 million logged events. By mapping section names across a
large sample of courses, the analysis enables cross-course comparisons of
student navigational sequences between sections. Transition matrices and heat
map visualizations are used to identify common navigational patterns. Findings
include that many of the generated heatmap include one or more diagonal axis,
indicating that students typically navigate from the current to the next or
previous section. More fine-grained patterns show typical behavior for blended
learning scenarios. Other patterns include dominant sections.

### 7. [VIS-Shepherd: Constructing Critic for LLM-based Data Visualization Generation](http://arxiv.org/pdf/2506.13326v1)

Authors: Bo Pan, Yixiao Fu, Ke Wang, Junyu Lu, Lunke Pan, Ziyang Qian, Yuhan Chen, Guoliang Wang, Yitao Zhou, Li Zheng, Yinghao Tang, Zhen Wen, Yuchen Wu, Junhua Lu, Biao Zhu, Minfeng Zhu, Bo Zhang, Wei Chen

Data visualization generation using Large Language Models (LLMs) has shown
promising results but often produces suboptimal visualizations that require
human intervention for improvement. In this work, we introduce VIS-Shepherd, a
specialized Multimodal Large Language Model (MLLM)-based critic to evaluate and
provide feedback for LLM-generated data visualizations. At the core of our
approach is a framework to construct a high-quality visualization critique
dataset, where we collect human-created visualization instances, synthesize
corresponding LLM-generated instances, and construct high-quality critiques. We
conduct both model-based automatic evaluation and human preference studies to
evaluate the effectiveness of our approach. Our experiments show that even
small (7B parameters) open-source MLLM models achieve substantial performance
gains by leveraging our high-quality visualization critique dataset, reaching
levels comparable to much larger open-source or even proprietary models. Our
work demonstrates significant potential for MLLM-based automated visualization
critique and indicates promising directions for enhancing LLM-based data
visualization generation. Our project page:
https://github.com/bopan3/VIS-Shepherd.

### 8. [Deflating Deflationism: A Critical Perspective on Debunking Arguments Against LLM Mentality](http://arxiv.org/pdf/2506.13403v1)

Authors: Alex Grzankowski, Geoff Keeling, Henry Shevlin, Winnie Street

Many people feel compelled to interpret, describe, and respond to Large
Language Models (LLMs) as if they possess inner mental lives similar to our
own. Responses to this phenomenon have varied. Inflationists hold that at least
some folk psychological ascriptions to LLMs are warranted. Deflationists argue
that all such attributions of mentality to LLMs are misplaced, often cautioning
against the risk that anthropomorphic projection may lead to misplaced trust or
potentially even confusion about the moral status of LLMs. We advance this
debate by assessing two common deflationary arguments against LLM mentality.
What we term the 'robustness strategy' aims to undercut one justification for
believing that LLMs are minded entities by showing that putatively cognitive
and humanlike behaviours are not robust, failing to generalise appropriately.
What we term the 'etiological strategy' undercuts attributions of mentality by
challenging naive causal explanations of LLM behaviours, offering alternative
causal accounts that weaken the case for mental state attributions. While both
strategies offer powerful challenges to full-blown inflationism, we find that
neither strategy provides a knock-down case against ascriptions of mentality to
LLMs simpliciter. With this in mind, we explore a modest form of inflationism
that permits ascriptions of mentality to LLMs under certain conditions.
Specifically, we argue that folk practice provides a defeasible basis for
attributing mental states and capacities to LLMs provided those mental states
and capacities can be understood in metaphysically undemanding terms (e.g.
knowledge, beliefs and desires), while greater caution is required when
attributing metaphysically demanding mental phenomena such as phenomenal
consciousness.

### 9. [The User Perspective on Island-Ready 6G Communication: A Survey of Future Smartphone Usage in Crisis-Struck Areas with Local Cellular Connectivity](http://arxiv.org/pdf/2506.13466v1)

Authors: Leon Janzen, Florentin Putz, Marc-André Kaufhold, Kolja Straub, Matthias Hollick

Using smartphone apps during crises is well-established, proving critical for
efficient crisis response. However, such apps become futile without an Internet
connection, which is a common issue during crises. The ongoing 6G
standardization explores the capability to provide local cellular connectivity
for areas cut off from the Internet in crises. This paper introduces to the HCI
community the concept of cellular island connectivity in isolated areas,
promising a seamless transition from normal operation to island operation with
local-only cellular connectivity. It presents findings from a survey (N = 857)
among adult smartphone users from major German cities regarding their
smartphone usage preferences in this model. Results show a shift in app demand,
with users favoring general-purpose apps over dedicated crisis apps in specific
scenarios. We prioritize smartphone services based on their criticality,
distinguishing between apps essential for crisis response and those supporting
routines. Our findings provide operators, developers, and authorities insights
into making user-centric design decisions for implementing island-ready 6G
communication.

### 10. [From Flat to Feeling: A Feasibility and Impact Study on Dynamic Facial Emotions in AI-Generated Avatars](http://arxiv.org/pdf/2506.13477v1)

Authors: Pegah Salehi, Sajad Amouei Sheshkal, Vajira Thambawita, Pål Halvorsen

Dynamic facial emotion is essential for believable AI-generated avatars;
however, most systems remain visually inert, limiting their utility in
high-stakes simulations such as virtual training for investigative interviews
with abused children. We introduce and evaluate a real-time architecture fusing
Unreal Engine 5 MetaHuman rendering with NVIDIA Omniverse Audio2Face to
translate vocal prosody into high-fidelity facial expressions on photorealistic
child avatars. We implemented a distributed two-PC setup that decouples
language processing and speech synthesis from GPU-intensive rendering, designed
to support low-latency interaction in desktop and VR environments. A
between-subjects study ($N=70$) using audio+visual and visual-only conditions
assessed perceptual impacts as participants rated emotional clarity, facial
realism, and empathy for two avatars expressing joy, sadness, and anger.
  Results demonstrate that avatars could express emotions recognizably, with
sadness and joy achieving high identification rates. However, anger recognition
significantly dropped without audio, highlighting the importance of congruent
vocal cues for high-arousal emotions. Interestingly, removing audio boosted
perceived facial realism, suggesting that audiovisual desynchrony remains a key
design challenge. These findings confirm the technical feasibility of
generating emotionally expressive avatars and provide guidance for improving
non-verbal communication in sensitive training simulations.

### Information Retrieval

### 1. [Gated Rotary-Enhanced Linear Attention for Long-term Sequential Recommendation](http://arxiv.org/pdf/2506.13315v1)

Authors: Juntao Hu, Wei Zhou, Huayi Shen, Xiao Du, Jie Liao, Junhao Wen, Min Gao

In Sequential Recommendation Systems (SRSs), Transformer models show
remarkable performance but face computation cost challenges when modeling
long-term user behavior sequences due to the quadratic complexity of the
dot-product attention mechanism. By approximating the dot-product attention,
linear attention provides an efficient option with linear complexity. However,
existing linear attention methods face two limitations: 1) they often use
learnable position encodings, which incur extra computational costs in
long-term sequence scenarios, and 2) they may not consider the user's
fine-grained local preferences and confuse these with the actual change of
long-term interests. To remedy these drawbacks, we propose a long-term
sequential Recommendation model with Gated Rotary Enhanced Linear Attention
(RecGRELA). Specifically, we first propose a Rotary-Enhanced Linear Attention
(RELA) module to model long-range dependency within the user's historical
information using rotary position encodings. We then introduce a local short
operation to incorporate local preferences and demonstrate the theoretical
insight. We further introduce a SiLU-based Gated mechanism for RELA (GRELA) to
help the model determine whether a user's behavior indicates local interest or
a genuine shift in long-term preferences. Experimental results on four public
datasets demonstrate that our RecGRELA achieves state-of-the-art performance
compared to existing SRSs while maintaining low memory overhead.

### 2. [Beyond One-Size-Fits-All: A Study of Neural and Behavioural Variability Across Different Recommendation Categories](http://arxiv.org/pdf/2506.13409v1)

Authors: Georgios Koutroumpas, Sebastian Idesis, Mireia Masias Bruns, Carlos Segura, Joemon M. Jose, Sergi Abadal, Ioannis Arapakis

Traditionally, Recommender Systems (RS) have primarily measured performance
based on the accuracy and relevance of their recommendations. However, this
algorithmic-centric approach overlooks how different types of recommendations
impact user engagement and shape the overall quality of experience. In this
paper, we shift the focus to the user and address for the first time the
challenge of decoding the neural and behavioural variability across distinct
recommendation categories, considering more than just relevance. Specifically,
we conducted a controlled study using a comprehensive e-commerce dataset
containing various recommendation types, and collected Electroencephalography
and behavioural data. We analysed both neural and behavioural responses to
recommendations that were categorised as Exact, Substitute, Complement, or
Irrelevant products within search query results. Our findings offer novel
insights into user preferences and decision-making processes, revealing
meaningful relationships between behavioural and neural patterns for each
category, but also indicate inter-subject variability.

### 3. [Tree-Based Text Retrieval via Hierarchical Clustering in RAGFrameworks: Application on Taiwanese Regulations](http://arxiv.org/pdf/2506.13607v1)

Authors: Chia-Heng Yu, Yen-Lung Tsai

Traditional Retrieval-Augmented Generation (RAG) systems employ brute-force
inner product search to retrieve the top-k most similar documents, then
combined with the user query and passed to a language model. This allows the
model to access external knowledge and reduce hallucinations. However,
selecting an appropriate k value remains a significant challenge in practical
applications: a small k may fail to retrieve sufficient information, while a
large k can introduce excessive and irrelevant content. To address this, we
propose a hierarchical clustering-based retrieval method that eliminates the
need to predefine k. Our approach maintains the accuracy and relevance of
system responses while adaptively selecting semantically relevant content. In
the experiment stage, we applied our method to a Taiwanese legal dataset with
expert-graded queries. The results show that our approach achieves superior
performance in expert evaluations and maintains high precision while
eliminating the need to predefine k, demonstrating improved accuracy and
interpretability in legal text retrieval tasks. Our framework is simple to
implement and easily integrates with existing RAG pipelines, making it a
practical solution for real-world applications under limited resources.

### 4. [OneRec Technical Report](http://arxiv.org/pdf/2506.13695v1)

Authors: Guorui Zhou, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Qiang Luo, Qianqian Wang, Qigen Hu, Rui Huang, Shiyao Wang, Weifeng Ding, Wuchao Li, Xinchen Luo, Xingmei Wang, Zexuan Cheng, Zixing Zhang, Bin Zhang, Boxuan Wang, Chaoyi Ma, Chengru Song, Chenhui Wang, Di Wang, Dongxue Meng, Fan Yang, Fangyu Zhang, Feng Jiang, Fuxing Zhang, Gang Wang, Guowang Zhang, Han Li, Hengrui Hu, Hezheng Lin, Hongtao Cheng, Hongyang Cao, Huanjie Wang, Jiaming Huang, Jiapeng Chen, Jiaqiang Liu, Jinghui Jia, Kun Gai, Lantao Hu, Liang Zeng, Liao Yu, Qiang Wang, Qidong Zhou, Shengzhe Wang, Shihui He, Shuang Yang, Shujie Yang, Sui Huang, Tao Wu, Tiantian He, Tingting Gao, Wei Yuan, Xiao Liang, Xiaoxiao Xu, Xugang Liu, Yan Wang, Yi Wang, Yiwu Liu, Yue Song, Yufei Zhang, Yunfan Wu, Yunfeng Zhao, Zhanyu Liu

Recommender systems have been widely used in various large-scale
user-oriented platforms for many years. However, compared to the rapid
developments in the AI community, recommendation systems have not achieved a
breakthrough in recent years. For instance, they still rely on a multi-stage
cascaded architecture rather than an end-to-end approach, leading to
computational fragmentation and optimization inconsistencies, and hindering the
effective application of key breakthrough technologies from the AI community in
recommendation scenarios.
  To address these issues, we propose OneRec, which reshapes the recommendation
system through an end-to-end generative approach and achieves promising
results. Firstly, we have enhanced the computational FLOPs of the current
recommendation model by 10 $\times$ and have identified the scaling laws for
recommendations within certain boundaries. Secondly, reinforcement learning
techniques, previously difficult to apply for optimizing recommendations, show
significant potential in this framework. Lastly, through infrastructure
optimizations, we have achieved 23.7% and 28.8% Model FLOPs Utilization (MFU)
on flagship GPUs during training and inference, respectively, aligning closely
with the LLM community. This architecture significantly reduces communication
and storage overhead, resulting in operating expense that is only 10.6% of
traditional recommendation pipelines. Deployed in Kuaishou/Kuaishou Lite APP,
it handles 25% of total queries per second, enhancing overall App Stay Time by
0.54% and 1.24%, respectively. Additionally, we have observed significant
increases in metrics such as 7-day Lifetime, which is a crucial indicator of
recommendation experience. We also provide practical lessons and insights
derived from developing, optimizing, and maintaining a production-scale
recommendation system with significant real-world impact.

### 5. [Vector Ontologies as an LLM world view extraction method](http://arxiv.org/pdf/2506.13252v1)

Authors: Kaspar Rothenfusser, Bekk Blando

Large Language Models (LLMs) possess intricate internal representations of
the world, yet these latent structures are notoriously difficult to interpret
or repurpose beyond the original prediction task. Building on our earlier work
(Rothenfusser, 2025), which introduced the concept of vector ontologies as a
framework for translating high-dimensional neural representations into
interpretable geometric structures, this paper provides the first empirical
validation of that approach. A vector ontology defines a domain-specific vector
space spanned by ontologically meaningful dimensions, allowing geometric
analysis of concepts and relationships within a domain. We construct an
8-dimensional vector ontology of musical genres based on Spotify audio features
and test whether an LLM's internal world model of music can be consistently and
accurately projected into this space. Using GPT-4o-mini, we extract genre
representations through multiple natural language prompts and analyze the
consistency of these projections across linguistic variations and their
alignment with ground-truth data. Our results show (1) high spatial consistency
of genre projections across 47 query formulations, (2) strong alignment between
LLM-inferred genre locations and real-world audio feature distributions, and
(3) evidence of a direct relationship between prompt phrasing and spatial
shifts in the LLM's inferred vector ontology. These findings demonstrate that
LLMs internalize structured, repurposable knowledge and that vector ontologies
offer a promising method for extracting and analyzing this knowledge in a
transparent and verifiable way.

### 6. [LTRR: Learning To Rank Retrievers for LLMs](http://arxiv.org/pdf/2506.13743v1)

Authors: To Eun Kim, Fernando Diaz

Retrieval-Augmented Generation (RAG) systems typically rely on a single fixed
retriever, despite growing evidence that no single retriever performs optimally
across all query types. In this paper, we explore a query routing approach that
dynamically selects from a pool of retrievers based on the query, using both
train-free heuristics and learned routing models. We frame routing as a
learning-to-rank (LTR) problem and introduce LTRR, a framework that learns to
rank retrievers by their expected utility gain to downstream LLM performance.
Our experiments, conducted on synthetic QA data with controlled query type
variations, show that routing-based RAG systems can outperform the best
single-retriever-based systems. Performance gains are especially pronounced in
models trained with the Answer Correctness (AC) metric and with pairwise
learning approaches, especially with XGBoost. We also observe improvements in
generalization to out-of-distribution queries. As part of the SIGIR 2025
LiveRAG challenge, our submitted system demonstrated the practical viability of
our approach, achieving competitive performance in both answer correctness and
faithfulness. These findings highlight the importance of both training
methodology and metric selection in query routing for RAG systems.

### 7. [SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists](http://arxiv.org/pdf/2506.13188v1)

Authors: Lynn Khellaf, Ipek Baris Schlicht, Tilman Mirass, Julia Bayer, Tilman Wagner, Ruben Bouwmeester

OpenStreetMap (OSM) is a vital resource for investigative journalists doing
geolocation verification. However, existing tools to query OSM data such as
Overpass Turbo require familiarity with complex query languages, creating
barriers for non-technical users. We present SPOT, an open source natural
language interface that makes OSM's rich, tag-based geographic data more
accessible through intuitive scene descriptions. SPOT interprets user inputs as
structured representations of geospatial object configurations using fine-tuned
Large Language Models (LLMs), with results being displayed in an interactive
map interface. While more general geospatial search tasks are conceivable, SPOT
is specifically designed for use in investigative journalism, addressing
real-world challenges such as hallucinations in model output, inconsistencies
in OSM tagging, and the noisy nature of user input. It combines a novel
synthetic data pipeline with a semantic bundling system to enable robust,
accurate query generation. To our knowledge, SPOT is the first system to
achieve reliable natural language access to OSM data at this level of accuracy.
By lowering the technical barrier to geolocation verification, SPOT contributes
a practical tool to the broader efforts to support fact-checking and combat
disinformation.

### 8. [Accessibility Barriers in Multi-Terabyte Public Datasets: The Gap Between Promise and Practice](http://arxiv.org/pdf/2506.13256v1)

Authors: Marc Bara

The promise of "free and open" multi-terabyte datasets often collides with
harsh realities. While these datasets may be technically accessible, practical
barriers -- from processing complexity to hidden costs -- create a system that
primarily serves well-funded institutions. This study examines accessibility
challenges across web crawls, satellite imagery, scientific data, and
collaborative projects, revealing a consistent two-tier system where
theoretical openness masks practical exclusivity. Our analysis demonstrates
that datasets marketed as "publicly accessible" typically require minimum
investments of \$1,000+ for meaningful analysis, with complex processing
pipelines demanding \$10,000-100,000+ in infrastructure costs. The
infrastructure requirements -- distributed computing knowledge, domain
expertise, and substantial budgets -- effectively gatekeep these datasets
despite their "open" status, limiting practical accessibility to those with
institutional support or substantial resources.

### 9. [Digital Transformation of Urban Planning in Australia: Influencing Factors and Key Challenges](http://arxiv.org/pdf/2506.13333v1)

Authors: Soheil Sabri, Sherah Kurnia

Over the past two decades, several governments in developing and developed
countries have started their journey toward digital transformation. However,
the pace and maturity of digital technologies and strategies are different
between public services. Current literature indicates that research on the
digital transformation of urban planning is still developing. Therefore, the
aim of this study is to understand the influencing factors and key challenges
for the digital transformation of urban planning in Australia. The study adopts
the inter-organisational theory and Planning Support Science (PSScience) under
the Technological, Organisational, and External Environmental (TOE) framework.
It involves a multiple case study, administered semi-structured interviews with
thirteen IT and urban planning experts across Victoria and New South Wales
governments and private industries. The study findings indicate that the main
challenges for digital transformation of the Australian urban planning system
are related to organisational and external environmental factors. Furthermore,
a digital maturity model is absent in the Australian urban planning industry.
This study offers important implications to research and practice related to
digital transformation in urban planning.

### 10. [Decompositional Reasoning for Graph Retrieval with Large Language Models](http://arxiv.org/pdf/2506.13380v1)

Authors: Valentin Six, Evan Dufraisse, Gaël de Chalendar

Large Language Models (LLMs) excel at many NLP tasks, but struggle with
multi-hop reasoning and factual consistency, limiting their effectiveness on
knowledge-intensive tasks like complex question answering (QA). Linking
Knowledge Graphs (KG) and LLMs has shown promising results, but LLMs generally
lack the ability to reason efficiently over graph-structured information. To
tackle this problem, we propose a novel retrieval approach that integrates
textual knowledge graphs into the LLM reasoning process via query
decomposition. Our method decomposes complex questions into sub-questions,
retrieves relevant textual subgraphs, and composes a question-specific
knowledge graph to guide answer generation. For that, we use a weighted
similarity function that focuses on both the complex question and the generated
subquestions to extract a relevant subgraph, which allows efficient and precise
retrieval for complex questions and improves the performance of LLMs on
multi-hop QA tasks. This structured reasoning pipeline enhances factual
grounding and interpretability while leveraging the generative strengths of
LLMs. We evaluate our method on standard multi-hop QA benchmarks and show that
it achieves comparable or superior performance to competitive existing methods,
using smaller models and fewer LLM calls.

### Machine Learning

### 1. [Antibody Foundational Model : Ab-RoBERTa](http://arxiv.org/pdf/2506.13006v1)

Authors: Eunna Huh, Hyeonsu Lee, Hyunjin Shin

With the growing prominence of antibody-based therapeutics, antibody
engineering has gained increasing attention as a critical area of research and
development. Recent progress in transformer-based protein large language models
(LLMs) has demonstrated promising applications in protein sequence design and
structural prediction. Moreover, the availability of large-scale antibody
datasets such as the Observed Antibody Space (OAS) database has opened new
avenues for the development of LLMs specialized for processing antibody
sequences. Among these, RoBERTa has demonstrated improved performance relative
to BERT, while maintaining a smaller parameter count (125M) compared to the
BERT-based protein model, ProtBERT (420M). This reduced model size enables more
efficient deployment in antibody-related applications. However, despite the
numerous advantages of the RoBERTa architecture, antibody-specific foundational
models built upon it have remained inaccessible to the research community. In
this study, we introduce Ab-RoBERTa, a RoBERTa-based antibody-specific LLM,
which is publicly available at https://huggingface.co/mogam-ai/Ab-RoBERTa. This
resource is intended to support a wide range of antibody-related research
applications including paratope prediction or humanness assessment.

### 2. [C-TLSAN: Content-Enhanced Time-Aware Long- and Short-Term Attention Network for Personalized Recommendation](http://arxiv.org/pdf/2506.13021v1)

Authors: Siqi Liang, Yudi Zhang, Yubo Wang

Sequential recommender systems aim to model users' evolving preferences by
capturing patterns in their historical interactions. Recent advances in this
area have leveraged deep neural networks and attention mechanisms to
effectively represent sequential behaviors and time-sensitive interests. In
this work, we propose C-TLSAN (Content-Enhanced Time-Aware Long- and Short-Term
Attention Network), an extension of the TLSAN architecture that jointly models
long- and short-term user preferences while incorporating semantic content
associated with items, such as product descriptions.
  C-TLSAN enriches the recommendation pipeline by embedding textual content
linked to users' historical interactions directly into both long-term and
short-term attention layers. This allows the model to learn from both
behavioral patterns and rich item content, enhancing user and item
representations across temporal dimensions. By fusing sequential signals with
textual semantics, our approach improves the expressiveness and personalization
capacity of recommendation systems.
  We conduct extensive experiments on large-scale Amazon datasets, benchmarking
C-TLSAN against state-of-the-art baselines, including recent sequential
recommenders based on Large Language Models (LLMs), which represent interaction
history and predictions in text form. Empirical results demonstrate that
C-TLSAN consistently outperforms strong baselines in next-item prediction
tasks. Notably, it improves AUC by 1.66%, Recall@10 by 93.99%, and Precision@10
by 94.80% on average over the best-performing baseline (TLSAN) across 10 Amazon
product categories. These results highlight the value of integrating
content-aware enhancements into temporal modeling frameworks for sequential
recommendation. Our code is available at https://github.com/booml247/cTLSAN.

### 3. [Forecast-Then-Optimize Deep Learning Methods](http://arxiv.org/pdf/2506.13036v1)

Authors: Jinhang Jiang, Nan Wu, Ben Liu, Mei Feng, Xin Ji, Karthik Srinivasan

Time series forecasting underpins vital decision-making across various
sectors, yet raw predictions from sophisticated models often harbor systematic
errors and biases. We examine the Forecast-Then-Optimize (FTO) framework,
pioneering its systematic synopsis. Unlike conventional Predict-Then-Optimize
(PTO) methods, FTO explicitly refines forecasts through optimization techniques
such as ensemble methods, meta-learners, and uncertainty adjustments.
Furthermore, deep learning and large language models have established
superiority over traditional parametric forecasting models for most enterprise
applications. This paper surveys significant advancements from 2016 to 2025,
analyzing mainstream deep learning FTO architectures. Focusing on real-world
applications in operations management, we demonstrate FTO's crucial role in
enhancing predictive accuracy, robustness, and decision efficacy. Our study
establishes foundational guidelines for future forecasting methodologies,
bridging theory and operational practicality.

### 4. [The Space Complexity of Learning-Unlearning Algorithms](http://arxiv.org/pdf/2506.13048v1)

Authors: Yeshwanth Cherapanamjeri, Sumegha Garg, Nived Rajaraman, Ayush Sekhari, Abhishek Shetty

We study the memory complexity of machine unlearning algorithms that provide
strong data deletion guarantees to the users. Formally, consider an algorithm
for a particular learning task that initially receives a training dataset.
Then, after learning, it receives data deletion requests from a subset of users
(of arbitrary size), and the goal of unlearning is to perform the task as if
the learner never received the data of deleted users. In this paper, we ask how
many bits of storage are needed to be able to delete certain training samples
at a later time. We focus on the task of realizability testing, where the goal
is to check whether the remaining training samples are realizable within a
given hypothesis class \(\mathcal{H}\).
  Toward that end, we first provide a negative result showing that the VC
dimension is not a characterization of the space complexity of unlearning. In
particular, we provide a hypothesis class with constant VC dimension (and
Littlestone dimension), but for which any unlearning algorithm for
realizability testing needs to store \(\Omega(n)\)-bits, where \(n\) denotes
the size of the initial training dataset. In fact, we provide a stronger
separation by showing that for any hypothesis class \(\mathcal{H}\), the amount
of information that the learner needs to store, so as to perform unlearning
later, is lower bounded by the \textit{eluder dimension} of \(\mathcal{H}\), a
combinatorial notion always larger than the VC dimension. We complement the
lower bound with an upper bound in terms of the star number of the underlying
hypothesis class, albeit in a stronger ticketed-memory model proposed by Ghazi
et al. (2023). Since the star number for a hypothesis class is never larger
than its Eluder dimension, our work highlights a fundamental separation between
central and ticketed memory models for machine unlearning.

### 5. [Uncertainty-Aware Graph Neural Networks: A Multi-Hop Evidence Fusion Approach](http://arxiv.org/pdf/2506.13083v1)

Authors: Qingfeng Chen, Shiyuan Li, Yixin Liu, Shirui Pan, Geoffrey I. Webb, Shichao Zhang

Graph neural networks (GNNs) excel in graph representation learning by
integrating graph structure and node features. Existing GNNs, unfortunately,
fail to account for the uncertainty of class probabilities that vary with the
depth of the model, leading to unreliable and risky predictions in real-world
scenarios. To bridge the gap, in this paper, we propose a novel Evidence Fusing
Graph Neural Network (EFGNN for short) to achieve trustworthy prediction,
enhance node classification accuracy, and make explicit the risk of wrong
predictions. In particular, we integrate the evidence theory with multi-hop
propagation-based GNN architecture to quantify the prediction uncertainty of
each node with the consideration of multiple receptive fields. Moreover, a
parameter-free cumulative belief fusion (CBF) mechanism is developed to
leverage the changes in prediction uncertainty and fuse the evidence to improve
the trustworthiness of the final prediction. To effectively optimize the EFGNN
model, we carefully design a joint learning objective composed of evidence
cross-entropy, dissonance coefficient, and false confident penalty. The
experimental results on various datasets and theoretical analyses demonstrate
the effectiveness of the proposed model in terms of accuracy and
trustworthiness, as well as its robustness to potential attacks. The source
code of EFGNN is available at https://github.com/Shiy-Li/EFGNN.

### 6. [Accelerating PDE-Constrained Optimization by the Derivative of Neural Operators](http://arxiv.org/pdf/2506.13120v1)

Authors: Ze Cheng, Zhuoyu Li, Xiaoqiang Wang, Jianing Huang, Zhizhou Zhang, Zhongkai Hao, Hang Su

PDE-Constrained Optimization (PDECO) problems can be accelerated
significantly by employing gradient-based methods with surrogate models like
neural operators compared to traditional numerical solvers. However, this
approach faces two key challenges: (1) **Data inefficiency**: Lack of efficient
data sampling and effective training for neural operators, particularly for
optimization purpose. (2) **Instability**: High risk of optimization derailment
due to inaccurate neural operator predictions and gradients. To address these
challenges, we propose a novel framework: (1) **Optimization-oriented
training**: we leverage data from full steps of traditional optimization
algorithms and employ a specialized training method for neural operators. (2)
**Enhanced derivative learning**: We introduce a *Virtual-Fourier* layer to
enhance derivative learning within the neural operator, a crucial aspect for
gradient-based optimization. (3) **Hybrid optimization**: We implement a hybrid
approach that integrates neural operators with numerical solvers, providing
robust regularization for the optimization process. Our extensive experimental
results demonstrate the effectiveness of our model in accurately learning
operators and their derivatives. Furthermore, our hybrid optimization approach
exhibits robust convergence.

### 7. [Efficient Algorithms for Logistic Contextual Slate Bandits with Bandit Feedback](http://arxiv.org/pdf/2506.13163v1)

Authors: Tanmay Goyal, Gaurav Sinha

We study the Logistic Contextual Slate Bandit problem, where, at each round,
an agent selects a slate of $N$ items from an exponentially large set (of size
$2^{\Omega(N)}$) of candidate slates provided by the environment. A single
binary reward, determined by a logistic model, is observed for the chosen
slate. Our objective is to develop algorithms that maximize cumulative reward
over $T$ rounds while maintaining low per-round computational costs. We propose
two algorithms, Slate-GLM-OFU and Slate-GLM-TS, that accomplish this goal.
These algorithms achieve $N^{O(1)}$ per-round time complexity via local
planning (independent slot selections), and low regret through global learning
(joint parameter estimation). We provide theoretical and empirical evidence
supporting these claims. Under a well-studied diversity assumption, we prove
that Slate-GLM-OFU incurs only $\tilde{O}(\sqrt{T})$ regret. Extensive
experiments across a wide range of synthetic settings demonstrate that our
algorithms consistently outperform state-of-the-art baselines, achieving both
the lowest regret and the fastest runtime. Furthermore, we apply our algorithm
to select in-context examples in prompts of Language Models for solving binary
classification tasks such as sentiment analysis. Our approach achieves
competitive test accuracy, making it a viable alternative in practical
scenarios.

### 8. [GeoRecon: Graph-Level Representation Learning for 3D Molecules via Reconstruction-Based Pretraining](http://arxiv.org/pdf/2506.13174v1)

Authors: Shaoheng Yan, Zian Li, Muhan Zhang

The pretraining-and-finetuning paradigm has driven significant advances
across domains, such as natural language processing and computer vision, with
representative pretraining paradigms such as masked language modeling and
next-token prediction. However, in molecular representation learning, the task
design remains largely limited to node-level denoising, which is effective at
modeling local atomic environments, yet maybe insufficient for capturing the
global molecular structure required by graph-level property prediction tasks,
such as energy estimation and molecular regression. In this work, we present
GeoRecon, a novel graph-level pretraining framework that shifts the focus from
individual atoms to the molecule as an integrated whole. GeoRecon introduces a
graph-level reconstruction task: during pretraining, the model is trained to
generate an informative graph representation capable of accurately guiding
reconstruction of the molecular geometry. This encourages the model to learn
coherent, global structural features rather than isolated atomic details.
Without relying on additional supervision or external data, GeoRecon
outperforms node-centric baselines on multiple molecular benchmarks (e.g., QM9,
MD17), demonstrating the benefit of incorporating graph-level reconstruction
for learning more holistic and geometry-aware molecular embeddings.

### 9. [KEPLA: A Knowledge-Enhanced Deep Learning Framework for Accurate Protein-Ligand Binding Affinity Prediction](http://arxiv.org/pdf/2506.13196v1)

Authors: Han Liu, Keyan Ding, Peilin Chen, Yinwei Wei, Liqiang Nie, Dapeng Wu, Shiqi Wang

Accurate prediction of protein-ligand binding affinity is critical for drug
discovery. While recent deep learning approaches have demonstrated promising
results, they often rely solely on structural features, overlooking valuable
biochemical knowledge associated with binding affinity. To address this
limitation, we propose KEPLA, a novel deep learning framework that explicitly
integrates prior knowledge from Gene Ontology and ligand properties of proteins
and ligands to enhance prediction performance. KEPLA takes protein sequences
and ligand molecular graphs as input and optimizes two complementary
objectives: (1) aligning global representations with knowledge graph relations
to capture domain-specific biochemical insights, and (2) leveraging cross
attention between local representations to construct fine-grained joint
embeddings for prediction. Experiments on two benchmark datasets across both
in-domain and cross-domain scenarios demonstrate that KEPLA consistently
outperforms state-of-the-art baselines. Furthermore, interpretability analyses
based on knowledge graph relations and cross attention maps provide valuable
insights into the underlying predictive mechanisms.

### 10. [Fatigue-Aware Adaptive Interfaces for Wearable Devices Using Deep Learning](http://arxiv.org/pdf/2506.13203v1)

Authors: Yikan Wang

Wearable devices, such as smartwatches and head-mounted displays, are
increasingly used for prolonged tasks like remote learning and work, but
sustained interaction often leads to user fatigue, reducing efficiency and
engagement. This study proposes a fatigue-aware adaptive interface system for
wearable devices that leverages deep learning to analyze physiological data
(e.g., heart rate, eye movement) and dynamically adjust interface elements to
mitigate cognitive load. The system employs multimodal learning to process
physiological and contextual inputs and reinforcement learning to optimize
interface features like text size, notification frequency, and visual contrast.
Experimental results show a 18% reduction in cognitive load and a 22%
improvement in user satisfaction compared to static interfaces, particularly
for users engaged in prolonged tasks. This approach enhances accessibility and
usability in wearable computing environments.

### Neural and Evolutionary Computing

### 1. [Energy-Efficient Digital Design: A Comparative Study of Event-Driven and Clock-Driven Spiking Neurons](http://arxiv.org/pdf/2506.13268v1)

Authors: Filippo Marostica, Alessio Carpegna, Alessandro Savino, Stefano Di Carlo

This paper presents a comprehensive evaluation of Spiking Neural Network
(SNN) neuron models for hardware acceleration by comparing event driven and
clock-driven implementations. We begin our investigation in software, rapidly
prototyping and testing various SNN models based on different variants of the
Leaky Integrate and Fire (LIF) neuron across multiple datasets. This phase
enables controlled performance assessment and informs design refinement. Our
subsequent hardware phase, implemented on FPGA, validates the simulation
findings and offers practical insights into design trade offs. In particular,
we examine how variations in input stimuli influence key performance metrics
such as latency, power consumption, energy efficiency, and resource
utilization. These results yield valuable guidelines for constructing energy
efficient, real time neuromorphic systems. Overall, our work bridges software
simulation and hardware realization, advancing the development of next
generation SNN accelerators.

### 2. [Evaluation of Nuclear Microreactor Cost-competitiveness in Current Electricity Markets Considering Reactor Cost Uncertainties](http://arxiv.org/pdf/2506.13361v1)

Authors: Muhammad R. Abdusammi, Ikhwan Khaleb, Fei Gao, Aditi Verma

This paper evaluates the cost competitiveness of microreactors in today's
electricity markets, with a focus on uncertainties in reactor costs. A Genetic
Algorithm (GA) is used to optimize key technical parameters, such as reactor
capacity, fuel enrichment, tail enrichment, refueling interval, and discharge
burnup, to minimize the Levelized Cost of Energy (LCOE). Base case results are
validated using Simulated Annealing (SA). By incorporating Probability
Distribution Functions (PDFs) for fuel cycle costs, the study identifies
optimal configurations under uncertainty. Methodologically, it introduces a
novel framework combining probabilistic cost modeling with evolutionary
optimization. Results show that microreactors can remain cost-competitive, with
LCOEs ranging from \$48.21/MWh to \$78.32/MWh when supported by the Production
Tax Credit (PTC). High reactor capacity, low fuel enrichment, moderate tail
enrichment and refueling intervals, and high discharge burnup enhance cost
efficiency. Among all factors, overnight capital cost (OCC) has the most
significant impact on LCOE, while O&M and fuel cost uncertainties have lesser
effects. The analysis highlights how energy policies like the PTC can reduce
LCOE by 22-24%, improving viability despite cost variability. Compared to
conventional nuclear, coal, and renewable sources like offshore wind, hydro,
and biomass, optimized microreactors show strong economic potential. This
research defines a realistic design space and key trade-offs, offering
actionable insights for policymakers, reactor designers, and energy planners
aiming to accelerate the deployment of affordable, sustainable microreactors.

### 3. [Sparse Convolutional Recurrent Learning for Efficient Event-based Neuromorphic Object Detection](http://arxiv.org/pdf/2506.13440v1)

Authors: Shenqi Wang, Yingfu Xu, Amirreza Yousefzadeh, Sherif Eissa, Henk Corporaal, Federico Corradi, Guangzhi Tang

Leveraging the high temporal resolution and dynamic range, object detection
with event cameras can enhance the performance and safety of automotive and
robotics applications in real-world scenarios. However, processing sparse event
data requires compute-intensive convolutional recurrent units, complicating
their integration into resource-constrained edge applications. Here, we propose
the Sparse Event-based Efficient Detector (SEED) for efficient event-based
object detection on neuromorphic processors. We introduce sparse convolutional
recurrent learning, which achieves over 92% activation sparsity in recurrent
processing, vastly reducing the cost for spatiotemporal reasoning on sparse
event data. We validated our method on Prophesee's 1 Mpx and Gen1 event-based
object detection datasets. Notably, SEED sets a new benchmark in computational
efficiency for event-based object detection which requires long-term temporal
learning. Compared to state-of-the-art methods, SEED significantly reduces
synaptic operations while delivering higher or same-level mAP. Our hardware
simulations showcase the critical role of SEED's hardware-aware design in
achieving energy-efficient and low-latency neuromorphic processing.

### 4. [AlphaEvolve: A coding agent for scientific and algorithmic discovery](http://arxiv.org/pdf/2506.13131v1)

Authors: Alexander Novikov, Ngân Vũ, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, Matej Balog

In this white paper, we present AlphaEvolve, an evolutionary coding agent
that substantially enhances capabilities of state-of-the-art LLMs on highly
challenging tasks such as tackling open scientific problems or optimizing
critical pieces of computational infrastructure. AlphaEvolve orchestrates an
autonomous pipeline of LLMs, whose task is to improve an algorithm by making
direct changes to the code. Using an evolutionary approach, continuously
receiving feedback from one or more evaluators, AlphaEvolve iteratively
improves the algorithm, potentially leading to new scientific and practical
discoveries. We demonstrate the broad applicability of this approach by
applying it to a number of important computational problems. When applied to
optimizing critical components of large-scale computational stacks at Google,
AlphaEvolve developed a more efficient scheduling algorithm for data centers,
found a functionally equivalent simplification in the circuit design of
hardware accelerators, and accelerated the training of the LLM underpinning
AlphaEvolve itself. Furthermore, AlphaEvolve discovered novel, provably correct
algorithms that surpass state-of-the-art solutions on a spectrum of problems in
mathematics and computer science, significantly expanding the scope of prior
automated discovery methods (Romera-Paredes et al., 2023). Notably, AlphaEvolve
developed a search algorithm that found a procedure to multiply two $4 \times
4$ complex-valued matrices using $48$ scalar multiplications; offering the
first improvement, after 56 years, over Strassen's algorithm in this setting.
We believe AlphaEvolve and coding agents like it can have a significant impact
in improving solutions of problems across many areas of science and
computation.

### 5. [Machine Learning as Iterated Belief Change a la Darwiche and Pearl](http://arxiv.org/pdf/2506.13157v1)

Authors: Theofanis Aravanis

Artificial Neural Networks (ANNs) are powerful machine-learning models
capable of capturing intricate non-linear relationships. They are widely used
nowadays across numerous scientific and engineering domains, driving
advancements in both research and real-world applications. In our recent work,
we focused on the statics and dynamics of a particular subclass of ANNs, which
we refer to as binary ANNs. A binary ANN is a feed-forward network in which
both inputs and outputs are restricted to binary values, making it particularly
suitable for a variety of practical use cases. Our previous study approached
binary ANNs through the lens of belief-change theory, specifically the
Alchourron, Gardenfors and Makinson (AGM) framework, yielding several key
insights. Most notably, we demonstrated that the knowledge embodied in a binary
ANN (expressed through its input-output behaviour) can be symbolically
represented using a propositional logic language. Moreover, the process of
modifying a belief set (through revision or contraction) was mapped onto a
gradual transition through a series of intermediate belief sets. Analogously,
the training of binary ANNs was conceptualized as a sequence of such belief-set
transitions, which we showed can be formalized using full-meet AGM-style belief
change. In the present article, we extend this line of investigation by
addressing some critical limitations of our previous study. Specifically, we
show that Dalal's method for belief change naturally induces a structured,
gradual evolution of states of belief. More importantly, given the known
shortcomings of full-meet belief change, we demonstrate that the training
dynamics of binary ANNs can be more effectively modelled using robust AGM-style
change operations -- namely, lexicographic revision and moderate contraction --
that align with the Darwiche-Pearl framework for iterated belief change.

### 6. [Polyra Swarms: A Shape-Based Approach to Machine Learning](http://arxiv.org/pdf/2506.13217v1)

Authors: Simon Klüttermann, Emmanuel Müller

We propose Polyra Swarms, a novel machine-learning approach that approximates
shapes instead of functions. Our method enables general-purpose learning with
very low bias. In particular, we show that depending on the task, Polyra Swarms
can be preferable compared to neural networks, especially for tasks like
anomaly detection. We further introduce an automated abstraction mechanism that
simplifies the complexity of a Polyra Swarm significantly, enhancing both their
generalization and transparency. Since Polyra Swarms operate on fundamentally
different principles than neural networks, they open up new research directions
with distinct strengths and limitations.

### 7. [Effective Stimulus Propagation in Neural Circuits: Driver Node Selection](http://arxiv.org/pdf/2506.13615v1)

Authors: Bulat Batuev, Arsenii Onuchin, Sergey Sukhov

Precise control of signal propagation in modular neural networks represents a
fundamental challenge in computational neuroscience. We establish a framework
for identifying optimal control nodes that maximize stimulus transmission
between weakly coupled neural populations. Using spiking stochastic block model
networks, we systematically compare driver node selection strategies -
including random sampling and topology-based centrality measures (degree,
betweenness, closeness, eigenvector, harmonic, and percolation centrality) - to
determine minimal control inputs for achieving inter-population
synchronization.
  Targeted stimulation of just 10-20% of the most central neurons in the source
population significantly enhances spiking propagation fidelity compared to
random selection. This approach yields a 2.7-fold increase in signal transfer
efficiency at critical inter-module connection densities p_inter = 0.04-0.07.
These findings establish a theoretical foundation for precision neuromodulation
in biological neural systems and neurotechnology applications.

### 8. [PhenoKG: Knowledge Graph-Driven Gene Discovery and Patient Insights from Phenotypes Alone](http://arxiv.org/pdf/2506.13119v1)

Authors: Kamilia Zaripova, Ege Özsoy, Nassir Navab, Azade Farshad

Identifying causative genes from patient phenotypes remains a significant
challenge in precision medicine, with important implications for the diagnosis
and treatment of genetic disorders. We propose a novel graph-based approach for
predicting causative genes from patient phenotypes, with or without an
available list of candidate genes, by integrating a rare disease knowledge
graph (KG). Our model, combining graph neural networks and transformers,
achieves substantial improvements over the current state-of-the-art. On the
real-world MyGene2 dataset, it attains a mean reciprocal rank (MRR) of 24.64\%
and nDCG@100 of 33.64\%, surpassing the best baseline (SHEPHERD) at 19.02\% MRR
and 30.54\% nDCG@100. We perform extensive ablation studies to validate the
contribution of each model component. Notably, the approach generalizes to
cases where only phenotypic data are available, addressing key challenges in
clinical decision support when genomic information is incomplete.

### Networking and Internet Architecture

### 1. [Cost-Efficient Design for 5G-Enabled MEC Servers under Uncertain User Demands](http://arxiv.org/pdf/2506.13003v1)

Authors: Yunyi Wu, Yongbing Zhang

Mobile edge computing (MEC) enhances the performance of 5G networks by
enabling low-latency, high-speed services through deploying data units of the
base station on edge servers located near mobile users. However, determining
the optimal capacity of these servers while dynamically offloading tasks and
allocating computing resources to meet uncertain user demands presents
significant challenges. This paper focuses on the design and planning of edge
servers with the dual objectives of minimizing capacity requirements and
reducing service latency for 5G services. To handle the complexity of uncertain
user demands, we formulate the problem as a two-stage stochastic model, which
can be linearized into a mixed-integer linear programming (MILP) problem. We
propose a novel approach called accelerated Benders decomposition (ABD) to
solve the problem at a large network scale. Numerical experiments demonstrate
that ABD achieves the optimal solution of MILP while significantly reducing
computation time.

### 2. [Joint Optimization of Multi-UAV Deployment and 3D Positioning in Traffic-Aware Aerial Networks](http://arxiv.org/pdf/2506.13287v1)

Authors: Kamran Shafafi, Alaa Awad Abdellatif, Manuel Ricardo, Rui Campos

Unmanned Aerial Vehicles (UAVs) have emerged as a key enabler for
next-generation wireless networks due to their on-demand deployment, high
mobility, and ability to provide Line-of-Sight (LoS) connectivity. These
features make UAVs particularly well-suited for dynamic and mission-critical
applications such as intelligent transportation systems and emergency
communications. However, effectively positioning multiple UAVs in real-time to
meet non-uniform, time-varying traffic demands remains a significant challenge,
especially when aiming to optimize network throughput and resource utilization.
In this paper, we propose an Efficient Multi-UAV Traffic-Aware Deployment
(EMTAD) Algorithm, a scalable and adaptive framework that dynamically adjusts
UAV placements based on real-time user locations and spatial traffic
distribution. In contrast to existing methods, EMTAD jointly optimizes UAV
positioning and minimizes the number of deployed UAVs, ensuring efficient
UE-UAV association while satisfying the traffic demand of users. Simulation
results demonstrate that EMTAD significantly improves network performance while
reducing deployment overhead by minimizing the number of UAVs required in
dynamic and traffic-aware environments.

### 3. [Delay-optimal Congestion-aware Routing and Computation Offloading in Arbitrary Network](http://arxiv.org/pdf/2506.13626v1)

Authors: Jinkun Zhang, Yuezhou Liu, Edmund Yeh

Emerging edge computing paradigms enable heterogeneous devices to collaborate
on complex computation applications. However, for arbitrary heterogeneous edge
networks, delay-optimal forwarding and computation offloading remains an open
problem. In this paper, we jointly optimize data/result routing and computation
placement in arbitrary networks with heterogeneous node capabilities, and
congestion-dependent nonlinear transmission and processing delay. Despite the
non-convexity of the formulated problem, based on analyzing the KKT condition,
we provide a set of sufficient optimality conditions that solve the problem
globally. To provide the insights for such global optimality, we show that the
proposed non-convex problem is geodesic-convex with mild assumptions. We also
show that the proposed sufficient optimality condition leads to a lower
hemicontinuous solution set, providing stability against user-input
perturbation. We then extend the framework to incorporate utility-based
congestion control and fairness. A fully distributed algorithm is developed to
converge to the global optimum. Numerical results demonstrate significant
improvements over multiple baselines algorithms.

### 4. [Dynamic Preference Multi-Objective Reinforcement Learning for Internet Network Management](http://arxiv.org/pdf/2506.13153v1)

Authors: DongNyeong Heo, Daniela Noemi Rim, Heeyoul Choi

An internet network service provider manages its network with multiple
objectives, such as high quality of service (QoS) and minimum computing
resource usage. To achieve these objectives, a reinforcement learning-based
(RL) algorithm has been proposed to train its network management agent.
Usually, their algorithms optimize their agents with respect to a single static
reward formulation consisting of multiple objectives with fixed importance
factors, which we call preferences. However, in practice, the preference could
vary according to network status, external concerns and so on. For example,
when a server shuts down and it can cause other servers' traffic overloads
leading to additional shutdowns, it is plausible to reduce the preference of
QoS while increasing the preference of minimum computing resource usages. In
this paper, we propose new RL-based network management agents that can select
actions based on both states and preferences. With our proposed approach, we
expect a single agent to generalize on various states and preferences.
Furthermore, we propose a numerical method that can estimate the distribution
of preference that is advantageous for unbiased training. Our experiment
results show that the RL agents trained based on our proposed approach
significantly generalize better with various preferences than the previous RL
approaches, which assume static preference during training. Moreover, we
demonstrate several analyses that show the advantages of our numerical
estimation method.

### 5. [Building Automotive Security on Internet Standards: An Integration of DNSSEC, DANE, and DANCE to Authenticate and Authorize In-Car Services](http://arxiv.org/pdf/2506.13261v1)

Authors: Timo Salomon, Mehmet Mueller, Philipp Meyer, Thomas C. Schmidt

The automotive industry is undergoing a software-as-a-service transformation
that enables software-defined functions and post-sale updates via cloud and
vehicle-to-everything communication. Connectivity in cars introduces
significant security challenges, as remote attacks on vehicles have become
increasingly prevalent. Current automotive designs call for security solutions
that address the entire lifetime of a vehicle. In this paper, we propose to
authenticate and authorize in-vehicle services by integrating DNSSEC, DANE, and
DANCE with automotive middleware. Our approach decouples the cryptographic
authentication of the service from that of the service deployment with the help
of DNSSEC and thereby largely simplifies key management. We propose to
authenticate in-vehicle services by certificates that are solely generated by
the service suppliers but published on deployment via DNSSEC TLSA records
solely signed by the OEM. Building on well-established Internet standards
ensures interoperability with various current and future protocols, scalable
management of credentials for millions of connected vehicles at
well-established security levels. We back our design proposal by a security
analysis using the STRIDE threat model and by evaluations in a realistic
in-vehicle setup that demonstrate its effectiveness.

### 6. [The User Perspective on Island-Ready 6G Communication: A Survey of Future Smartphone Usage in Crisis-Struck Areas with Local Cellular Connectivity](http://arxiv.org/pdf/2506.13466v1)

Authors: Leon Janzen, Florentin Putz, Marc-André Kaufhold, Kolja Straub, Matthias Hollick

Using smartphone apps during crises is well-established, proving critical for
efficient crisis response. However, such apps become futile without an Internet
connection, which is a common issue during crises. The ongoing 6G
standardization explores the capability to provide local cellular connectivity
for areas cut off from the Internet in crises. This paper introduces to the HCI
community the concept of cellular island connectivity in isolated areas,
promising a seamless transition from normal operation to island operation with
local-only cellular connectivity. It presents findings from a survey (N = 857)
among adult smartphone users from major German cities regarding their
smartphone usage preferences in this model. Results show a shift in app demand,
with users favoring general-purpose apps over dedicated crisis apps in specific
scenarios. We prioritize smartphone services based on their criticality,
distinguishing between apps essential for crisis response and those supporting
routines. Our findings provide operators, developers, and authorities insights
into making user-centric design decisions for implementing island-ready 6G
communication.

### 7. [Unlearning-Enhanced Website Fingerprinting Attack: Against Backdoor Poisoning in Anonymous Networks](http://arxiv.org/pdf/2506.13563v1)

Authors: Yali Yuan, Kai Xu, Ruolin Ma, Yuchen Zhang

Website Fingerprinting (WF) is an effective tool for regulating and governing
the dark web. However, its performance can be significantly degraded by
backdoor poisoning attacks in practical deployments. This paper aims to address
the problem of hidden backdoor poisoning attacks faced by Website
Fingerprinting attack, and designs a feasible mothed that integrates unlearning
technology to realize detection of automatic poisoned points and complete
removal of its destructive effects, requiring only a small number of known
poisoned test points. Taking Tor onion routing as an example, our method
evaluates the influence value of each training sample on these known poisoned
test points as the basis for judgment. We optimize the use of influence scores
to identify poisoned samples within the training dataset. Furthermore, by
quantifying the difference between the contribution of model parameters on the
taining data and the clean data, the target parameters are dynamically adjusted
to eliminate the impact of the backdoor attacks. Experiments on public datasets
under the assumptions of closed-world (CW) and open-world (OW) verify the
effectiveness of the proposed method. In complex scenes containing both clean
website fingerprinting features and backdoor triggers, the accuracy of the
model on the poisoned dataset and the test dataset is stable at about 80%,
significantly outperforming the traditional WF attack models. In addition, the
proposed method achieves a 2-3 times speedup in runtime efficiency compared to
baseline methods. By incorporating machine unlearning, we realize a WF attack
model that exhibits enhanced resistance to backdoor poisoning and faster
execution speeds in adversarial settings.

### 8. [HELENA: High-Efficiency Learning-based channel Estimation using dual Neural Attention](http://arxiv.org/pdf/2506.13408v1)

Authors: Miguel Camelo Botero, Esra Aycan Beyazit, Nina Slamnik-Kriještorac, Johann M. Marquez-Barja

Accurate channel estimation is critical for high-performance Orthogonal
Frequency-Division Multiplexing systems such as 5G New Radio, particularly
under low signal-to-noise ratio and stringent latency constraints. This letter
presents HELENA, a compact deep learning model that combines a lightweight
convolutional backbone with two efficient attention mechanisms: patch-wise
multi-head self-attention for capturing global dependencies and a
squeeze-and-excitation block for local feature refinement. Compared to CEViT, a
state-of-the-art vision transformer-based estimator, HELENA reduces inference
time by 45.0\% (0.175\,ms vs.\ 0.318\,ms), achieves comparable accuracy
($-16.78$\,dB vs.\ $-17.30$\,dB), and requires $8\times$ fewer parameters
(0.11M vs.\ 0.88M), demonstrating its suitability for low-latency, real-time
deployment.

### Robotics

### 1. [Underwater target 6D State Estimation via UUV Attitude Enhance Observability](http://arxiv.org/pdf/2506.13105v1)

Authors: Fen Liu, Chengfeng Jia, Na Zhang, Shenghai Yuan, Rong Su

Accurate relative state observation of Unmanned Underwater Vehicles (UUVs)
for tracking uncooperative targets remains a significant challenge due to the
absence of GPS, complex underwater dynamics, and sensor limitations. Existing
localization approaches rely on either global positioning infrastructure or
multi-UUV collaboration, both of which are impractical for a single UUV
operating in large or unknown environments. To address this, we propose a novel
persistent relative 6D state estimation framework that enables a single UUV to
estimate its relative motion to a non-cooperative target using only successive
noisy range measurements from two monostatic sonar sensors. Our key
contribution is an observability-enhanced attitude control strategy, which
optimally adjusts the UUV's orientation to improve the observability of
relative state estimation using a Kalman filter, effectively mitigating the
impact of sensor noise and drift accumulation. Additionally, we introduce a
rigorously proven Lyapunov-based tracking control strategy that guarantees
long-term stability by ensuring that the UUV maintains an optimal measurement
range, preventing localization errors from diverging over time. Through
theoretical analysis and simulations, we demonstrate that our method
significantly improves 6D relative state estimation accuracy and robustness
compared to conventional approaches. This work provides a scalable,
infrastructure-free solution for UUVs tracking uncooperative targets
underwater.

### 2. [Autonomous 3D Moving Target Encirclement and Interception with Range measurement](http://arxiv.org/pdf/2506.13106v1)

Authors: Fen Liu, Shenghai Yuan, Thien-Minh Nguyen, Rong Su

Commercial UAVs are an emerging security threat as they are capable of
carrying hazardous payloads or disrupting air traffic. To counter UAVs, we
introduce an autonomous 3D target encirclement and interception strategy.
Unlike traditional ground-guided systems, this strategy employs autonomous
drones to track and engage non-cooperative hostile UAVs, which is effective in
non-line-of-sight conditions, GPS denial, and radar jamming, where conventional
detection and neutralization from ground guidance fail. Using two noisy
real-time distances measured by drones, guardian drones estimate the relative
position from their own to the target using observation and velocity
compensation methods, based on anti-synchronization (AS) and an X$-$Y circular
motion combined with vertical jitter. An encirclement control mechanism is
proposed to enable UAVs to adaptively transition from encircling and protecting
a target to encircling and monitoring a hostile target. Upon breaching a
warning threshold, the UAVs may even employ a suicide attack to neutralize the
hostile target. We validate this strategy through real-world UAV experiments
and simulated analysis in MATLAB, demonstrating its effectiveness in detecting,
encircling, and intercepting hostile drones. More details:
https://youtu.be/5eHW56lPVto.

### 3. [Equilibrium-Driven Smooth Separation and Navigation of Marsupial Robotic Systems](http://arxiv.org/pdf/2506.13198v1)

Authors: Bin-Bin Hu, Bayu Jayawardhana, Ming Cao

In this paper, we propose an equilibrium-driven controller that enables a
marsupial carrier-passenger robotic system to achieve smooth carrier-passenger
separation and then to navigate the passenger robot toward a predetermined
target point. Particularly, we design a potential gradient in the form of a
cubic polynomial for the passenger's controller as a function of the
carrier-passenger and carrier-target distances in the moving carrier's frame.
This introduces multiple equilibrium points corresponding to the zero state of
the error dynamic system during carrier-passenger separation. The change of
equilibrium points is associated with the change in their attraction regions,
enabling smooth carrier-passenger separation and afterwards seamless navigation
toward the target. Finally, simulations demonstrate the effectiveness and
adaptability of the proposed controller in environments containing obstacles.

### 4. [C2TE: Coordinated Constrained Task Execution Design for Ordering-Flexible Multi-Vehicle Platoon Merging](http://arxiv.org/pdf/2506.13202v1)

Authors: Bin-Bin Hu, Yanxin Zhou, Henglai Wei, Shuo Cheng, Chen Lv

In this paper, we propose a distributed coordinated constrained task
execution (C2TE) algorithm that enables a team of vehicles from different lanes
to cooperatively merge into an {\it ordering-flexible platoon} maneuvering on
the desired lane. Therein, the platoon is flexible in the sense that no
specific spatial ordering sequences of vehicles are predetermined. To attain
such a flexible platoon, we first separate the multi-vehicle platoon (MVP)
merging mission into two stages, namely, pre-merging regulation and {\it
ordering-flexible platoon} merging, and then formulate them into distributed
constraint-based optimization problems. Particularly, by encoding
longitudinal-distance regulation and same-lane collision avoidance subtasks
into the corresponding control barrier function (CBF) constraints, the proposed
algorithm in Stage 1 can safely enlarge sufficient longitudinal distances among
adjacent vehicles. Then, by encoding lateral convergence, longitudinal-target
attraction, and neighboring collision avoidance subtasks into CBF constraints,
the proposed algorithm in Stage~2 can efficiently achieve the {\it
ordering-flexible platoon}. Note that the {\it ordering-flexible platoon} is
realized through the interaction of the longitudinal-target attraction and
time-varying neighboring collision avoidance constraints simultaneously.
Feasibility guarantee and rigorous convergence analysis are both provided under
strong nonlinear couplings induced by flexible orderings. Finally, experiments
using three autonomous mobile vehicles (AMVs) are conducted to verify the
effectiveness and flexibility of the proposed algorithm, and extensive
simulations are performed to demonstrate its robustness, adaptability, and
scalability when tackling vehicles' sudden breakdown, new appearing, different
number of lanes, mixed autonomy, and large-scale scenarios, respectively.

### 5. [Uncertainty-Informed Active Perception for Open Vocabulary Object Goal Navigation](http://arxiv.org/pdf/2506.13367v1)

Authors: Utkarsh Bajpai, Julius Rückin, Cyrill Stachniss, Marija Popović

Mobile robots exploring indoor environments increasingly rely on
vision-language models to perceive high-level semantic cues in camera images,
such as object categories. Such models offer the potential to substantially
advance robot behaviour for tasks such as object-goal navigation (ObjectNav),
where the robot must locate objects specified in natural language by exploring
the environment. Current ObjectNav methods heavily depend on prompt engineering
for perception and do not address the semantic uncertainty induced by
variations in prompt phrasing. Ignoring semantic uncertainty can lead to
suboptimal exploration, which in turn limits performance. Hence, we propose a
semantic uncertainty-informed active perception pipeline for ObjectNav in
indoor environments. We introduce a novel probabilistic sensor model for
quantifying semantic uncertainty in vision-language models and incorporate it
into a probabilistic geometric-semantic map to enhance spatial understanding.
Based on this map, we develop a frontier exploration planner with an
uncertainty-informed multi-armed bandit objective to guide efficient object
search. Experimental results demonstrate that our method achieves ObjectNav
success rates comparable to those of state-of-the-art approaches, without
requiring extensive prompt engineering.

### 6. [Observability-Aware Active Calibration of Multi-Sensor Extrinsics for Ground Robots via Online Trajectory Optimization](http://arxiv.org/pdf/2506.13420v1)

Authors: Jiang Wang, Yaozhong Kang, Linya Fu, Kazuhiro Nakadai, He Kong

Accurate calibration of sensor extrinsic parameters for ground robotic
systems (i.e., relative poses) is crucial for ensuring spatial alignment and
achieving high-performance perception. However, existing calibration methods
typically require complex and often human-operated processes to collect data.
Moreover, most frameworks neglect acoustic sensors, thereby limiting the
associated systems' auditory perception capabilities. To alleviate these
issues, we propose an observability-aware active calibration method for ground
robots with multimodal sensors, including a microphone array, a LiDAR
(exteroceptive sensors), and wheel encoders (proprioceptive sensors). Unlike
traditional approaches, our method enables active trajectory optimization for
online data collection and calibration, contributing to the development of more
intelligent robotic systems. Specifically, we leverage the Fisher information
matrix (FIM) to quantify parameter observability and adopt its minimum
eigenvalue as an optimization metric for trajectory generation via B-spline
curves. Through planning and replanning of robot trajectory online, the method
enhances the observability of multi-sensor extrinsic parameters. The
effectiveness and advantages of our method have been demonstrated through
numerical simulations and real-world experiments. For the benefit of the
community, we have also open-sourced our code and data at
https://github.com/AISLAB-sustech/Multisensor-Calibration.

### 7. [VLM-SFD: VLM-Assisted Siamese Flow Diffusion Framework for Dual-Arm Cooperative Manipulation](http://arxiv.org/pdf/2506.13428v1)

Authors: Jiaming Chen, Yiyu Jiang, Aoshen Huang, Yang Li, Wei Pan

Dual-arm cooperative manipulation holds great promise for tackling complex
real-world tasks that demand seamless coordination and adaptive dynamics.
Despite substantial progress in learning-based motion planning, most approaches
struggle to generalize across diverse manipulation tasks and adapt to dynamic,
unstructured environments, particularly in scenarios involving interactions
between two objects such as assembly, tool use, and bimanual grasping. To
address these challenges, we introduce a novel VLM-Assisted Siamese Flow
Diffusion (VLM-SFD) framework for efficient imitation learning in dual-arm
cooperative manipulation. The proposed VLM-SFD framework exhibits outstanding
adaptability, significantly enhancing the ability to rapidly adapt and
generalize to diverse real-world tasks from only a minimal number of human
demonstrations. Specifically, we propose a Siamese Flow Diffusion Network
(SFDNet) employs a dual-encoder-decoder Siamese architecture to embed two
target objects into a shared latent space, while a diffusion-based conditioning
process-conditioned by task instructions-generates two-stream object-centric
motion flows that guide dual-arm coordination. We further design a dynamic task
assignment strategy that seamlessly maps the predicted 2D motion flows into 3D
space and incorporates a pre-trained vision-language model (VLM) to adaptively
assign the optimal motion to each robotic arm over time. Experiments validate
the effectiveness of the proposed method, demonstrating its ability to
generalize to diverse manipulation tasks while maintaining high efficiency and
adaptability. The code and demo videos are publicly available on our project
website https://sites.google.com/view/vlm-sfd/.

### 8. [Adaptive Model-Base Control of Quadrupeds via Online System Identification using Kalman Filter](http://arxiv.org/pdf/2506.13432v1)

Authors: Jonas Haack, Franek Stark, Shubham Vyas, Frank Kirchner, Shivesh Kumar

Many real-world applications require legged robots to be able to carry
variable payloads. Model-based controllers such as model predictive control
(MPC) have become the de facto standard in research for controlling these
systems. However, most model-based control architectures use fixed plant
models, which limits their applicability to different tasks. In this paper, we
present a Kalman filter (KF) formulation for online identification of the mass
and center of mass (COM) of a four-legged robot. We evaluate our method on a
quadrupedal robot carrying various payloads and find that it is more robust to
strong measurement noise than classical recursive least squares (RLS) methods.
Moreover, it improves the tracking performance of the model-based controller
with varying payloads when the model parameters are adjusted at runtime.

### 9. [Learning Swing-up Maneuvers for a Suspended Aerial Manipulation Platform in a Hierarchical Control Framework](http://arxiv.org/pdf/2506.13478v1)

Authors: Hemjyoti Das, Minh Nhat Vu, Christian Ott

In this work, we present a novel approach to augment a model-based control
method with a reinforcement learning (RL) agent and demonstrate a swing-up
maneuver with a suspended aerial manipulation platform. These platforms are
targeted towards a wide range of applications on construction sites involving
cranes, with swing-up maneuvers allowing it to perch at a given location,
inaccessible with purely the thrust force of the platform. Our proposed
approach is based on a hierarchical control framework, which allows different
tasks to be executed according to their assigned priorities. An RL agent is
then subsequently utilized to adjust the reference set-point of the
lower-priority tasks to perform the swing-up maneuver, which is confined in the
nullspace of the higher-priority tasks, such as maintaining a specific
orientation and position of the end-effector. Our approach is validated using
extensive numerical simulation studies.

### 10. [Disturbance-aware minimum-time planning strategies for motorsport vehicles with probabilistic safety certificates](http://arxiv.org/pdf/2506.13622v1)

Authors: Martino Gulisano, Matteo Masoni, Marco Gabiccini, Massimo Guiggiani

This paper presents a disturbance-aware framework that embeds robustness into
minimum-lap-time trajectory optimization for motorsport. Two formulations are
introduced. (i) Open-loop, horizon-based covariance propagation uses worst-case
uncertainty growth over a finite window to tighten tire-friction and
track-limit constraints. (ii) Closed-loop, covariance-aware planning
incorporates a time-varying LQR feedback law in the optimizer, providing a
feedback-consistent estimate of disturbance attenuation and enabling sharper
yet reliable constraint tightening. Both methods yield reference trajectories
for human or artificial drivers: in autonomous applications the modelled
controller can replicate the on-board implementation, while for human driving
accuracy increases with the extent to which the driver can be approximated by
the assumed time-varying LQR policy. Computational tests on a representative
Barcelona-Catalunya sector show that both schemes meet the prescribed safety
probability, yet the closed-loop variant incurs smaller lap-time penalties than
the more conservative open-loop solution, while the nominal (non-robust)
trajectory remains infeasible under the same uncertainties. By accounting for
uncertainty growth and feedback action during planning, the proposed framework
delivers trajectories that are both performance-optimal and probabilistically
safe, advancing minimum-time optimization toward real-world deployment in
high-performance motorsport and autonomous racing.

### Software Engineering

### 1. [Designing Deep Learning Frameworks for LLMs:Challenges, Expectations, and Opportunities](http://arxiv.org/pdf/2506.13114v1)

Authors: Yanzhou Mu, Rong Wang, Juan Zhai, Chunrong Fang, Xiang Chen, Jiacong Wu, An Guo, Jiawei Shen, Bingzhuo Li, Zhenyu Chen

Large language models (LLMs) drive significant advancements in real industry
applications. LLMs rely on DL frameworks for efficient model construction,
distributed execution, and optimized deployment. Their large parameter scale
and long execution cycles place extreme demands on DL frameworks in terms of
scalability, stability, and efficiency. Therefore, poor usability, limited
functionality, and subtle bugs in DL frameworks may hinder development
efficiency and cause severe failures or resource waste. However, a fundamental
question remains underinvestigated, i.e., What challenges do DL frameworks face
in supporting LLMs? To seek an answer, we investigate these challenges through
a large-scale analysis of issue reports from three major DL frameworks
(MindSpore, PyTorch, TensorFlow) and eight associated LLM toolkits (e.g.,
Megatron). We construct a taxonomy of LLM-centric bugs, requirements, and user
questions and enrich it through interviews with 11 LLM users and eight DL
framework developers, uncovering key technical challenges and misalignments
between user needs and developer priorities. Our contributions are threefold:
(1) we develop a comprehensive taxonomy comprising four question themes (nine
sub-themes), four requirement themes (15 sub-themes), and ten bug themes (45
sub-themes); (2) we assess the perceived importance and priority of these
challenges based on practitioner insights; and (3) we identify five key
findings across the LLM development and propose five actionable recommendations
to improve the reliability, usability, and testability of DL frameworks. Our
results highlight critical limitations in current DL frameworks and offer
concrete guidance for advancing their support for the next generation of LLM
construction and applications.

### 2. [Empirical Evaluation of Large Language Models in Automated Program Repair](http://arxiv.org/pdf/2506.13186v1)

Authors: Jiajun Sun, Fengjie Li, Xinzhu Qi, Hongyu Zhang, Jiajun Jiang

The increasing prevalence of software bugs has made automated program repair
(APR) a key research focus. Large language models (LLMs) offer new
opportunities for APR, but existing studies mostly rely on smaller,
earlier-generation models and Java benchmarks. The repair capabilities of
modern, large-scale LLMs across diverse languages and scenarios remain
underexplored. To address this, we conduct a comprehensive empirical study of
four open-source LLMs, CodeLlama, LLaMA, StarCoder, and DeepSeek-Coder,
spanning 7B to 33B parameters, diverse architectures, and purposes. We evaluate
them across two bug scenarios (enterprise-grades and algorithmic), three
languages (Java, C/C++, Python), and four prompting strategies, analyzing over
600K generated patches on six benchmarks. Key findings include: (1) model
specialization (e.g., CodeLlama) can outperform larger general-purpose models
(e.g., LLaMA); (2) repair performance does not scale linearly with model size;
(3) correct patches often appear early in generation; and (4) prompts
significantly affect results. These insights offer practical guidance for
designing effective and efficient LLM-based APR systems.

### 3. [Isolating Noisy Labelled Test Cases in Human-in-the-Loop Oracle Learning](http://arxiv.org/pdf/2506.13273v1)

Authors: Charaka Geethal Kapugama

Incorrectly labelled test cases can adversely affect the training process of
human-in-the-loop oracle learning tech-niques. This paper introduces ISONOISE,
a technique designed to identify such mislabelled test cases introduced during
human-in-the-loop oracle learning. This technique can be applied to programs
taking numeric inputs. Given a compromised automatic test oracle and its
training test suite, ISONOISE first isolates thetest cases suspected of being
mislabelled. This task is performed based on the level of disagreement of a
test case with respect to the others. An intermediate automatic test oracle is
trained based on the slightly disagreeing test cases. Based on the predictions
of this intermediate oracle, the test cases suspected of being mislabelled are
systematically presented for relabelling. When mislabelled test cases are
found, the intermediate test oracle is updated. This process repeats until no
mislabelled test case is found in relabelling. ISONOISE was evaluated within
the human-in-the-loop oracle learning method used in LEARN2FIX. Experimental
results demonstrate that ISONOISE can identify mislabelled test cases
introduced by the human in LEARN2FIX with over 67% accuracy, while requiring
only a small number of relabelling queries. These findings highlight the
potential of ISONOISE to enhance the reliability of human-in-the-loop oracle
learning.

### 4. [Adopting Use Case Descriptions for Requirements Specification: an Industrial Case Study](http://arxiv.org/pdf/2506.13303v1)

Authors: Julian Frattini, Anja Frattini

Context: Use case (UC) descriptions are a prominent format for specifying
functional requirements. Existing literature abounds with recommendations on
how to write high-quality UC descriptions but lacks insights into (1) their
real-world adoption, (2) whether these recommendations correspond to actual
quality, and (3) which factors influence the quality of UCs. Objectives: We aim
to contribute empirical evidence about the adoption of UC descriptions in a
large, globally distributed case company. Methods: We surveyed 1188 business
requirements of a case company that were elicited from 2020-01-01 until
2024-12-31 and contained 1192 UCs in various forms. Among these, we manually
evaluated the 273 template-style UC descriptions against established quality
guidelines. We generated descriptive statistics of the format's adoption over
the surveyed time frame. Furthermore, we used inferential statistics to
determine (a) how properties of the requirements engineering process affected
the UC quality and (b) how UC quality affects subsequent software development
activities. Results and Conclusions: Our descriptive results show how the
adoption of UC descriptions in practice deviates from textbook recommendations.
However, our inferential results suggest that only a few phenomena like
solution-orientation show an actual impact in practice. These results can steer
UC quality research into a more relevant direction.

### 5. [DesignCoder: Hierarchy-Aware and Self-Correcting UI Code Generation with Large Language Models](http://arxiv.org/pdf/2506.13663v1)

Authors: Yunnong Chen, Shixian Ding, YingYing Zhang, Wenkai Chen, Jinzhou Du, Lingyun Sun, Liuqing Chen

Multimodal large language models (MLLMs) have streamlined front-end interface
development by automating code generation. However, these models also introduce
challenges in ensuring code quality. Existing approaches struggle to maintain
both visual consistency and functional completeness in the generated
components. Moreover, they lack mechanisms to assess the fidelity and
correctness of the rendered pages. To address these issues, we propose
DesignCoder, a novel hierarchical-aware and self-correcting automated code
generation framework. Specifically, we introduce UI Grouping Chains, which
enhance MLLMs' capability to understand and predict complex nested UI
hierarchies. Subsequently, DesignCoder employs a hierarchical
divide-and-conquer approach to generate front-end code. Finally, we incorporate
a self-correction mechanism to improve the model's ability to identify and
rectify errors in the generated code. Extensive evaluations on a dataset of UI
mockups collected from both open-source communities and industry projects
demonstrate that DesignCoder outperforms state-of-the-art baselines in React
Native, a widely adopted UI framework. Our method achieves a 37.63%, 9.52%,
12.82% performance increase in visual similarity metrics (MSE, CLIP, SSIM) and
significantly improves code structure similarity in terms of TreeBLEU,
Container Match, and Tree Edit Distance by 30.19%, 29.31%, 24.67%. Furthermore,
we conducted a user study with professional developers to assess the quality
and practicality of the generated code. Results indicate that DesignCoder
aligns with industry best practices, demonstrating high usability, readability,
and maintainability. Our approach provides an efficient and practical solution
for agile front-end development, enabling development teams to focus more on
core functionality and product innovation.

### 6. [Using LLMs for Security Advisory Investigations: How Far Are We?](http://arxiv.org/pdf/2506.13161v1)

Authors: Bayu Fedra Abdullah, Yusuf Sulistyo Nugroho, Brittany Reid, Raula Gaikovina Kula, Kazumasa Shimari, Kenichi Matsumoto

Large Language Models (LLMs) are increasingly used in software security, but
their trustworthiness in generating accurate vulnerability advisories remains
uncertain. This study investigates the ability of ChatGPT to (1) generate
plausible security advisories from CVE-IDs, (2) differentiate real from fake
CVE-IDs, and (3) extract CVE-IDs from advisory descriptions. Using a curated
dataset of 100 real and 100 fake CVE-IDs, we manually analyzed the credibility
and consistency of the model's outputs. The results show that ChatGPT generated
plausible security advisories for 96% of given input real CVE-IDs and 97% of
given input fake CVE-IDs, demonstrating a limitation in differentiating between
real and fake IDs. Furthermore, when these generated advisories were
reintroduced to ChatGPT to identify their original CVE-ID, the model produced a
fake CVE-ID in 6% of cases from real advisories. These findings highlight both
the strengths and limitations of ChatGPT in cybersecurity applications. While
the model demonstrates potential for automating advisory generation, its
inability to reliably authenticate CVE-IDs or maintain consistency upon
re-evaluation underscores the risks associated with its deployment in critical
security tasks. Our study emphasizes the importance of using LLMs with caution
in cybersecurity workflows and suggests the need for further improvements in
their design to improve reliability and applicability in security advisory
generation.

### 7. [Querying Large Automotive Software Models: Agentic vs. Direct LLM Approaches](http://arxiv.org/pdf/2506.13171v1)

Authors: Lukasz Mazur, Nenad Petrovic, James Pontes Miranda, Ansgar Radermacher, Robert Rasche, Alois Knoll

Large language models (LLMs) offer new opportunities for interacting with
complex software artifacts, such as software models, through natural language.
They present especially promising benefits for large software models that are
difficult to grasp in their entirety, making traditional interaction and
analysis approaches challenging. This paper investigates two approaches for
leveraging LLMs to answer questions over software models: direct prompting,
where the whole software model is provided in the context, and an agentic
approach combining LLM-based agents with general-purpose file access tools. We
evaluate these approaches using an Ecore metamodel designed for timing analysis
and software optimization in automotive and embedded domains. Our findings show
that while the agentic approach achieves accuracy comparable to direct
prompting, it is significantly more efficient in terms of token usage. This
efficiency makes the agentic approach particularly suitable for the automotive
industry, where the large size of software models makes direct prompting
infeasible, establishing LLM agents as not just a practical alternative but the
only viable solution. Notably, the evaluation was conducted using small LLMs,
which are more feasible to be executed locally - an essential advantage for
meeting strict requirements around privacy, intellectual property protection,
and regulatory compliance. Future work will investigate software models in
diverse formats, explore more complex agent architectures, and extend agentic
workflows to support not only querying but also modification of software
models.

### 8. [From Empirical Evaluation to Context-Aware Enhancement: Repairing Regression Errors with LLMs](http://arxiv.org/pdf/2506.13182v1)

Authors: Anh Ho, Thanh Le-Cong, Bach Le, Christine Rizkallah

[...] Since then, various APR approaches, especially those leveraging the
power of large language models (LLMs), have been rapidly developed to fix
general software bugs. Unfortunately, the effectiveness of these advanced
techniques in the context of regression bugs remains largely unexplored. This
gap motivates the need for an empirical study evaluating the effectiveness of
modern APR techniques in fixing real-world regression bugs.
  In this work, we conduct an empirical study of APR techniques on Java
regression bugs. To facilitate our study, we introduce RegMiner4APR, a
high-quality benchmark of Java regression bugs integrated into a framework
designed to facilitate APR research. The current benchmark includes 99
regression bugs collected from 32 widely used real-world Java GitHub
repositories. We begin by conducting an in-depth analysis of the benchmark,
demonstrating its diversity and quality. Building on this foundation, we
empirically evaluate the capabilities of APR to regression bugs by assessing
both traditional APR tools and advanced LLM-based APR approaches. Our
experimental results show that classical APR tools fail to repair any bugs,
while LLM-based APR approaches exhibit promising potential. Motivated by these
results, we investigate impact of incorporating bug-inducing change information
into LLM-based APR approaches for fixing regression bugs. Our results highlight
that this context-aware enhancement significantly improves the performance of
LLM-based APR, yielding 1.8x more successful repairs compared to using
LLM-based APR without such context.

### 9. [Model Context Protocol (MCP) at First Glance: Studying the Security and Maintainability of MCP Servers](http://arxiv.org/pdf/2506.13538v1)

Authors: Mohammed Mehedi Hasan, Hao Li, Emad Fallahzadeh, Bram Adams, Ahmed E. Hassan

Although Foundation Models (FMs), such as GPT-4, are increasingly used in
domains like finance and software engineering, reliance on textual interfaces
limits these models' real-world interaction. To address this, FM providers
introduced tool calling-triggering a proliferation of frameworks with distinct
tool interfaces. In late 2024, Anthropic introduced the Model Context Protocol
(MCP) to standardize this tool ecosystem, which has become the de facto
standard with over eight million weekly SDK downloads. Despite its adoption,
MCP's AI-driven, non-deterministic control flow introduces new risks to
sustainability, security, and maintainability, warranting closer examination.
  Towards this end, we present the first large-scale empirical study of MCP.
Using state-of-the-art health metrics and a hybrid analysis pipeline, combining
a general-purpose static analysis tool with an MCP-specific scanner, we
evaluate 1,899 open-source MCP servers to assess their health, security, and
maintainability. Despite MCP servers demonstrating strong health metrics, we
identify eight distinct vulnerabilities-only three overlapping with traditional
software vulnerabilities. Additionally, 7.2% of servers contain general
vulnerabilities and 5.5% exhibit MCP-specific tool poisoning. Regarding
maintainability, while 66% exhibit code smells, 14.4% contain ten bug patterns
overlapping prior research. These findings highlight the need for MCP-specific
vulnerability detection techniques while reaffirming the value of traditional
analysis and refactoring practices.

### 10. [Tady: A Neural Disassembler without Structural Constraint Violations](http://arxiv.org/pdf/2506.13323v1)

Authors: Siliang Qin, Fengrui Yang, Hao Wang, Bolun Zhang, Zeyu Gao, Chao Zhang, Kai Chen

Disassembly is a crucial yet challenging step in binary analysis. While
emerging neural disassemblers show promise for efficiency and accuracy, they
frequently generate outputs violating fundamental structural constraints, which
significantly compromise their practical usability. To address this critical
problem, we regularize the disassembly solution space by formalizing and
applying key structural constraints based on post-dominance relations. This
approach systematically detects widespread errors in existing neural
disassemblers' outputs. These errors often originate from models' limited
context modeling and instruction-level decoding that neglect global structural
integrity. We introduce Tady, a novel neural disassembler featuring an improved
model architecture and a dedicated post-processing algorithm, specifically
engineered to address these deficiencies. Comprehensive evaluations on diverse
binaries demonstrate that Tady effectively eliminates structural constraint
violations and functions with high efficiency, while maintaining
instruction-level accuracy.

### Social and Information Networks

### 1. [TwiUSD: A Benchmark Dataset and Structure-Aware LLM Framework for User Stance Detection](http://arxiv.org/pdf/2506.13343v1)

Authors: Fuaing Niu, Zini Chen, Zhiyu Xie, Genan Dai, Bowen Zhang

User-level stance detection (UserSD) remains challenging due to the lack of
high-quality benchmarks that jointly capture linguistic and social structure.
In this paper, we introduce TwiUSD, the first large-scale, manually annotated
UserSD benchmark with explicit followee relationships, containing 16,211 users
and 47,757 tweets. TwiUSD enables rigorous evaluation of stance models by
integrating tweet content and social links, with superior scale and annotation
quality. Building on this resource, we propose MRFG: a structure-aware
framework that uses LLM-based relevance filtering and feature routing to
address noise and context heterogeneity. MRFG employs multi-scale filtering and
adaptively routes features through graph neural networks or multi-layer
perceptrons based on topological informativeness. Experiments show MRFG
consistently outperforms strong baselines (including PLMs, graph-based models,
and LLM prompting) in both in-target and cross-target evaluation.

### 2. [Dynamic Evolution of Cooperation Based on Adaptive Reputation Threshold and Game Transition](http://arxiv.org/pdf/2506.13319v1)

Authors: Hongyu Yue, Xiaojin Xiong, Minyu Feng, Attila Szolnoki

In real-world social systems, individual interactions are frequently shaped
by reputation, which not only influences partner selection but also affects the
nature and benefits of the interactions themselves. We propose a heterogeneous
game transition model that incorporates a reputation-based dynamic threshold
mechanism to investigate how reputation regulates game evolution. In our
framework, individuals determine the type of game they engage in according to
their own and their neighbors' reputation levels. In turn, the outcomes of
these interactions modify their reputations, thereby driving the adaptation and
evolution of future strategies in a feedback-informed manner. Through
simulations on two representative topological structures, square lattice and
small-world networks, we find that network topology exerts a profound influence
on the evolutionary dynamics. Due to its localized connection characteristics,
the square lattice network fosters the long-term coexistence of competing
strategies. In contrast, the small-world network is more susceptible to changes
in system parameters due to the efficiency of information dissemination and the
sensitivity of strategy evolution. Additionally, the reputation mechanism is
significant in promoting the formation of a dominant state of cooperation,
especially in contexts of high sensitivity to reputation. Although the initial
distribution of reputation influences the early stage of the evolutionary path,
it has little effect on the final steady state of the system. Hence, we can
conclude that the ultimate steady state of evolution is primarily determined by
the reputation mechanism and the network structure.

### 3. [Efficient Approximate Temporal Triangle Counting in Streaming with Predictions](http://arxiv.org/pdf/2506.13173v1)

Authors: Giorgio Venturin, Ilie Sarpe, Fabio Vandin

Triangle counting is a fundamental and widely studied problem on static
graphs, and recently on temporal graphs, where edges carry information on the
timings of the associated events. Streaming processing and resource efficiency
are crucial requirements for counting triangles in modern massive temporal
graphs, with millions of nodes and up to billions of temporal edges. However,
current exact and approximate algorithms are unable to handle large-scale
temporal graphs. To fill such a gap, we introduce STEP, a scalable and
efficient algorithm to approximate temporal triangle counts from a stream of
temporal edges. STEP combines predictions to the number of triangles a temporal
edge is involved in, with a simple sampling strategy, leading to scalability,
efficiency, and accurate approximation of all eight temporal triangle types
simultaneously. We analytically prove that, by using a sublinear amount of
memory, STEP obtains unbiased and very accurate estimates. In fact, even noisy
predictions can significantly reduce the variance of STEP's estimates. Our
extensive experiments on massive temporal graphs with up to billions of edges
demonstrate that STEP outputs high-quality estimates and is more efficient than
state-of-the-art methods.

### 4. [A data-driven analysis of the impact of non-compliant individuals on epidemic diffusion in urban settings](http://arxiv.org/pdf/2506.13325v1)

Authors: Fabio Mazza, Marco Brambilla, Carlo Piccardi, Francesco Pierri

Individuals who do not comply with public health safety measures pose a
significant challenge to effective epidemic control, as their risky behaviours
can undermine public health interventions. This is particularly relevant in
urban environments because of their high population density and complex social
interactions. In this study, we employ detailed contact networks, built using a
data-driven approach, to examine the impact of non-compliant individuals on
epidemic dynamics in three major Italian cities: Torino, Milano, and Palermo.
We use a heterogeneous extension of the Susceptible-Infected-Recovered model
that distinguishes between ordinary and non-compliant individuals, who are more
infectious and/or more susceptible. By combining electoral data with recent
findings on vaccine hesitancy, we obtain spatially heterogeneous distributions
of non-compliance. Epidemic simulations demonstrate that even a small
proportion of non-compliant individuals in the population can substantially
increase the number of infections and accelerate the timing of their peak.
Furthermore, the impact of non-compliance is greatest when disease transmission
rates are moderate. Including the heterogeneous, data-driven distribution of
non-compliance in the simulations results in infection hotspots forming with
varying intensity according to the disease transmission rate. Overall, these
findings emphasise the importance of monitoring behavioural compliance and
tailoring public health interventions to address localised risks.

### Systems and Control

### 1. [Online-Optimized Gated Radial Basis Function Neural Network-Based Adaptive Control](http://arxiv.org/pdf/2506.13168v1)

Authors: Mingcong Li

Real-time adaptive control of nonlinear systems with unknown dynamics and
time-varying disturbances demands precise modeling and robust parameter
adaptation. While existing neural network-based strategies struggle with
computational inefficiency or inadequate temporal dependencies, this study
proposes a hybrid control framework integrating a Temporal-Gated Radial Basis
Function (TGRBF) network with a nonlinear robust controller. The TGRBF
synergizes radial basis function neural networks (RBFNNs) and gated recurrent
units (GRUs) through dynamic gating, enabling efficient offline system
identification and online temporal modeling with minimal parameter overhead
(14.5% increase vs. RBFNNs). During control execution, an event-triggered
optimization mechanism activates momentum-explicit gradient descent to refine
network parameters, leveraging historical data to suppress overfitting while
maintaining real-time feasibility. Concurrently, the nonlinear controller
adaptively tunes its gains via Jacobian-driven rules derived from the TGRBF
model, ensuring rapid error convergence and disturbance rejection.
Lyapunov-based analysis rigorously guarantees uniform ultimate boundedness of
both tracking errors and adaptive parameters. Simulations on a nonlinear
benchmark system demonstrate the framework's superiority: compared to PID and
fixed-gain robust controllers, the proposed method reduces settling time by
14.2%, limits overshoot to 10%, and achieves 48.4% lower integral time-weighted
absolute error under dynamic disturbances. By unifying data-driven adaptability
with stability-guaranteed control, this work advances real-time performance in
partially observable, time-varying industrial systems.

### 2. [RL-Guided MPC for Autonomous Greenhouse Control](http://arxiv.org/pdf/2506.13278v1)

Authors: Salim Msaad, Murray Harraway, Robert D. McAllister

The efficient operation of greenhouses is essential for enhancing crop yield
while minimizing energy costs. This paper investigates a control strategy that
integrates Reinforcement Learning (RL) and Model Predictive Control (MPC) to
optimize economic benefits in autonomous greenhouses. Previous research has
explored the use of RL and MPC for greenhouse control individually, or by using
MPC as the function approximator for the RL agent. This study introduces the
RL-Guided MPC framework, where a RL policy is trained and then used to
construct a terminal cost and terminal region constraint for the MPC
optimization problem. This approach leverages the ability to handle
uncertainties of RL with MPC's online optimization to improve overall control
performance. The RL-Guided MPC framework is compared with both MPC and RL via
numerical simulations. Two scenarios are considered: a deterministic
environment and an uncertain environment. Simulation results demonstrate that,
in both environments, RL-Guided MPC outperforms both RL and MPC with shorter
prediction horizons.

### 3. [Stability and Performance of Online Feedback Optimization for Distribution Grid Flexibility](http://arxiv.org/pdf/2506.13280v1)

Authors: Florian Klein-Helmkamp, Tina Möllemann, Irina Zettl, Andreas Ulbig

The integration of distributed energy resources (DERs) into sub-transmission
systems has enabled new opportunities for flexibility provision in ancillary
services such as frequency and voltage support, as well as congestion
management. This paper investigates the stability and performance of Online
Feedback Optimization (OFO) controllers in ensuring reliable flexibility
provision. A hierarchical control architecture is proposed, emphasizing safe
transitions between system states within the Feasible Operating Region (FOR).
We evaluate the controller's stability and performance through simulations of
transitions to the vertices of the FOR, analyzing the impact of tuning
parameters. The study demonstrates that controller stability is sensitive to
parameter tuning, particularly gain and sensitivity approximations. Results
demonstrate that improper tuning can lead to oscillatory or unstable behavior,
highlighting the need for systematic parameter selection to ensure reliable
operation across the full flexibility range.

### 4. [EPC Framework for BESS Projects](http://arxiv.org/pdf/2506.13281v1)

Authors: Zeenat Hameed, Chresten Træholt

Battery Energy Storage Systems (BESS) are critical for modern power networks,
supporting grid services such as frequency regulation, peak shaving, and black
start. Delivering a BESS under an Engineering, Procurement, and Construction
(EPC) model requires a concise methodology that balances regulatory compliance,
technical details, and schedule efficiency. This paper presents a streamlined,
five step EPC framework covering feasibility assessment, permitting,
procurement, construction, and commissioning. A Danish demonstration (the BOSS
project on Bornholm) serves as a case study.

### 5. [Aggregating Inverter-Based Resources for Fast Frequency Response: A Nash Bargaining Game-Based Approach](http://arxiv.org/pdf/2506.13291v1)

Authors: Xiang Zhu, Hua Geng, Hongyang Qing, Xin Zou

This paper proposes a multi-objective optimization (MOO) approach for
grid-level frequency regulation by aggregating inverter-based resources (IBRs).
Virtual power plants (VPPs), acting as aggregators, can efficiently respond to
dynamic response requirements from the grid. Through parametric modeling,
grid-level frequency regulation requirements are accurately quantified and
translated into a feasible parameter region defined by device-level parameters.
Based on this feasible region, an MOO model is developed to address the
conflicting demands of IBRs during frequency response. A Nash bargaining
game-based approach is then employed to optimally allocate regulation
requirements within the VPP, balancing the various demands of the IBRs.
Numerical experiments demonstrate the effectiveness of the proposed method in
enhancing frequency stability and improving coordination among IBRs.

### 6. [Voltage Stability of Inverter-Based Systems: Impact of Parameters and Irrelevance of Line Dynamics](http://arxiv.org/pdf/2506.13341v1)

Authors: Sushobhan Chatterjee, Sijia Geng

This paper investigates voltage stability in inverter-based power systems
concerning fold and saddle-node bifurcations. An analytical expression is
derived for the sensitivity of the stability margin using the normal vector to
the bifurcation hypersurface. Such information enables efficient identification
of effective control parameters in mitigating voltage instability.
Comprehensive analysis reveals that reactive loading setpoint and current
controller's feedforward gain are the most influential parameters for enhancing
voltage stability in a grid-following (GFL) inverter system, while the voltage
controller's feedforward gain plays a dominant role in a grid-forming (GFM)
inverter. Notably, both theoretical and numerical results demonstrate that
transmission line dynamics have no impact on fold/saddle-node bifurcations in
these systems. Results in this paper provide insights for efficient analysis
and control in future inverter-dominated power systems through reductions in
parameter space and model complexity.

### 7. [A Model-Free Detection Method for Internal Short Circuits in Single Lithium-ion Cells Using Pseudo Open-Circuit Voltage Difference](http://arxiv.org/pdf/2506.13394v1)

Authors: Yangyang Xu, Chenglin Liao

This letter proposes a lightweight, model-free online diagnostic framework
for detecting internal short circuits (ISC) in single lithium-ion cells under
dynamic operating conditions. The core of the method lies in computing the
first-order difference of pseudo open-circuit voltage
($\boldsymbol{\mathrm{OCV}_{\text{pseudo}}}$) to extract high-frequency
deviations caused by ISC events from low-frequency polarization variations. The
method relies solely on terminal voltage, current measurements, and an offline
$R_0$--SOC look-up table, thereby eliminating the need for electrochemical or
equivalent-circuit observers. Validated on ten real and one false fault
scenarios, the proposed approach achieves a 100\% detection success rate with
no missed or false alarms. In addition, the proposed method exhibits extremely
low computational and memory requirements, making it highly suitable for
real-time deployment in battery management systems (BMS).

### 8. [High-gain model-following control for trajectory tracking](http://arxiv.org/pdf/2506.13463v1)

Authors: Nicals Tietze, Kai Wulff, Johann Reger

We consider trajectory tracking for minimum-phase nonlinear systems in
Byrnes-Isidori form using the model-following control (MFC) architecture. The
tracking problem is motivated by a hierarchical control concept where a
higher-level instance provides the reference trajectory at run-time. We present
a computational efficient implementation of the feedback linearisation MFC
design, and apply high-gain feedback in the process control loop (PCL) to
achieve practical tracking in presence of Lipschitz perturbations. Our main
results establish ultimate boundedness of the tracking error and give a
constructive bound for the high-gain scaling parameter to achieve arbitrary
tracking precision. Further we establish that the peaking phenomenon can be
attenuated using MFC. We demonstrate the results via an automotive case study
considering advanced engine-based cruise control.

### 9. [Hybrid Polynomial Zonotopes: A Set Representation for Reachability Analysis in Hybrid Nonaffine Systems](http://arxiv.org/pdf/2506.13567v1)

Authors: Peng Xie, Zhen Zhang, Amr Alanwar

Reachability analysis for hybrid nonaffine systems remains computationally
challenging, as existing set representations--including constrained,
polynomial, and hybrid zonotopes--either lose tightness under high-order
nonaffine maps or suffer exponential blow-up after discrete jumps. This paper
introduces Hybrid Polynomial Zonotope (HPZ), a novel set representation that
combines the mode-dependent generator structure of hybrid zonotopes with the
algebraic expressiveness of polynomial zonotopes. HPZs compactly encode
non-convex reachable states across modes by attaching polynomial exponents to
each hybrid generator, enabling precise capture of high-order state-input
couplings without vertex enumeration. We develop a comprehensive library of HPZ
operations, including Minkowski sum, linear transformation, and intersection.
Theoretical analysis and computational experiments demonstrate that HPZs
achieve superior tightness preservation and computational efficiency compared
to existing approaches for hybrid system reachability analysis.

### 10. [BattBee: Equivalent Circuit Modeling and Early Detection of Thermal Runaway Triggered by Internal Short Circuits for Lithium-Ion Batteries](http://arxiv.org/pdf/2506.13577v1)

Authors: Sangwon Kang, Hao Tu, Huazhen Fang

Lithium-ion batteries are the enabling power source for transportation
electrification. However, in real-world applications, they remain vulnerable to
internal short circuits (ISCs) and the consequential risk of thermal runaway
(TR). Toward addressing the challenge of ISCs and TR, we undertake a systematic
study that extends from dynamic modeling to fault detection in this paper.
First, we develop {\em BattBee}, the first equivalent circuit model to
specifically describe the onset of ISCs and the evolution of subsequently
induced TR. Drawing upon electrochemical modeling, the model can simulate ISCs
at different severity levels and predict their impact on the initiation and
progression of TR events. With the physics-inspired design, this model offers
strong physical interpretability and predictive accuracy, while maintaining
structural simplicity to allow fast computation. Then, building upon the
BattBee model, we develop fault detection observers and derive detection
criteria together with decision-making logics to identify the occurrence and
emergence of ISC and TR events. This detection approach is principled in design
and fast in computation, lending itself to practical applications. Validation
based on simulations and experimental data demonstrates the effectiveness of
both the BattBee model and the ISC/TR detection approach. The research outcomes
underscore this study's potential for real-world battery safety risk
management.

### Machine Learning (Statistics Category)

### 1. [CoIFNet: A Unified Framework for Multivariate Time Series Forecasting with Missing Values](http://arxiv.org/pdf/2506.13064v1)

Authors: Kai Tang, Ji Zhang, Hua Meng, Minbo Ma, Qi Xiong, Jie Xu, Tianrui Li

Multivariate time series forecasting (MTSF) is a critical task with broad
applications in domains such as meteorology, transportation, and economics.
Nevertheless, pervasive missing values caused by sensor failures or human
errors significantly degrade forecasting accuracy. Prior efforts usually employ
an impute-then-forecast paradigm, leading to suboptimal predictions due to
error accumulation and misaligned objectives between the two stages. To address
this challenge, we propose the Collaborative Imputation-Forecasting Network
(CoIFNet), a novel framework that unifies imputation and forecasting to achieve
robust MTSF in the presence of missing values. Specifically, CoIFNet takes the
observed values, mask matrix and timestamp embeddings as input, processing them
sequentially through the Cross-Timestep Fusion (CTF) and Cross-Variate Fusion
(CVF) modules to capture temporal dependencies that are robust to missing
values. We provide theoretical justifications on how our CoIFNet learning
objective improves the performance bound of MTSF with missing values. Through
extensive experiments on challenging MSTF benchmarks, we demonstrate the
effectiveness and computational efficiency of our proposed approach across
diverse missing-data scenarios, e.g., CoIFNet outperforms the state-of-the-art
method by $\underline{\textbf{24.40}}$% ($\underline{\textbf{23.81}}$%) at a
point (block) missing rate of 0.6, while improving memory and time efficiency
by $\underline{\boldsymbol{4.3\times}}$ and
$\underline{\boldsymbol{2.1\times}}$, respectively.

### 2. [Honesty in Causal Forests: When It Helps and When It Hurts](http://arxiv.org/pdf/2506.13107v1)

Authors: Yanfang Hou, Carlos Fernández-Loría

Causal forests are increasingly used to personalize decisions based on
estimated treatment effects. A distinctive modeling choice in this method is
honest estimation: using separate data for splitting and for estimating effects
within leaves. This practice is the default in most implementations and is
widely seen as desirable for causal inference. But we show that honesty can
hurt the accuracy of individual-level effect estimates. The reason is a classic
bias-variance trade-off: honesty reduces variance by preventing overfitting,
but increases bias by limiting the model's ability to discover and exploit
meaningful heterogeneity in treatment effects. This trade-off depends on the
signal-to-noise ratio (SNR): honesty helps when effect heterogeneity is hard to
detect (low SNR), but hurts when the signal is strong (high SNR). In essence,
honesty acts as a form of regularization, and like any regularization choice,
it should be guided by out-of-sample performance, not adopted by default.

### 3. [SAGDA: Open-Source Synthetic Agriculture Data for Africa](http://arxiv.org/pdf/2506.13123v1)

Authors: Abdelghani Belgaid, Oumnia Ennaji

Data scarcity in African agriculture hampers machine learning (ML) model
performance, limiting innovations in precision agriculture. The Synthetic
Agriculture Data for Africa (SAGDA) library, a Python-based open-source
toolkit, addresses this gap by generating, augmenting, and validating synthetic
agricultural datasets. We present SAGDA's design and development practices,
highlighting its core functions: generate, model, augment, validate, visualize,
optimize, and simulate, as well as their roles in applications of ML for
agriculture. Two use cases are detailed: yield prediction enhanced via data
augmentation, and multi-objective NPK (nitrogen, phosphorus, potassium)
fertilizer recommendation. We conclude with future plans for expanding SAGDA's
capabilities, underscoring the vital role of open-source, data-driven practices
for African agriculture.

### 4. [Random Matrix Theory for Deep Learning: Beyond Eigenvalues of Linear Models](http://arxiv.org/pdf/2506.13139v1)

Authors: Zhenyu Liao, Michael W. Mahoney

Modern Machine Learning (ML) and Deep Neural Networks (DNNs) often operate on
high-dimensional data and rely on overparameterized models, where classical
low-dimensional intuitions break down. In particular, the proportional regime
where the data dimension, sample size, and number of model parameters are all
large and comparable, gives rise to novel and sometimes counterintuitive
behaviors. This paper extends traditional Random Matrix Theory (RMT) beyond
eigenvalue-based analysis of linear models to address the challenges posed by
nonlinear ML models such as DNNs in this regime. We introduce the concept of
High-dimensional Equivalent, which unifies and generalizes both Deterministic
Equivalent and Linear Equivalent, to systematically address three technical
challenges: high dimensionality, nonlinearity, and the need to analyze generic
eigenspectral functionals. Leveraging this framework, we provide precise
characterizations of the training and generalization performance of linear
models, nonlinear shallow networks, and deep networks. Our results capture rich
phenomena, including scaling laws, double descent, and nonlinear learning
dynamics, offering a unified perspective on the theoretical understanding of
deep learning in high dimensions.

### 5. [Bayesian Active Learning of (small) Quantile Sets through Expected Estimator Modification](http://arxiv.org/pdf/2506.13211v1)

Authors: Romain Ait Abdelmalek-Lomenech, Julien Bect, Emmanuel Vazquez

Given a multivariate function taking deterministic and uncertain inputs, we
consider the problem of estimating a quantile set: a set of deterministic
inputs for which the probability that the output belongs to a specific region
remains below a given threshold. To solve this problem in the context of
expensive-to-evaluate black-box functions, we propose a Bayesian active
learning strategy based on Gaussian process modeling. The strategy is driven by
a novel sampling criterion, which belongs to a broader principle that we refer
to as Expected Estimator Modification (EEM). More specifically, the strategy
relies on a novel sampling criterion combined with a sequential Monte Carlo
framework that enables the construction of batch-sequential designs for the
efficient estimation of small quantile sets. The performance of the strategy is
illustrated on several synthetic examples and an industrial application case
involving the ROTOR37 compressor model.

### 6. [Experimental Design for Semiparametric Bandits](http://arxiv.org/pdf/2506.13390v1)

Authors: Seok-Jin Kim, Gi-Soo Kim, Min-hwan Oh

We study finite-armed semiparametric bandits, where each arm's reward
combines a linear component with an unknown, potentially adversarial shift.
This model strictly generalizes classical linear bandits and reflects
complexities common in practice. We propose the first experimental-design
approach that simultaneously offers a sharp regret bound, a PAC bound, and a
best-arm identification guarantee. Our method attains the minimax regret
$\tilde{O}(\sqrt{dT})$, matching the known lower bound for finite-armed linear
bandits, and further achieves logarithmic regret under a positive suboptimality
gap condition. These guarantees follow from our refined non-asymptotic analysis
of orthogonalized regression that attains the optimal $\sqrt{d}$ rate, paving
the way for robust and efficient learning across a broad class of
semiparametric bandit problems.

### 7. [Variational Inference with Mixtures of Isotropic Gaussians](http://arxiv.org/pdf/2506.13613v1)

Authors: Marguerite Petit-Talamon, Marc Lambert, Anna Korba

Variational inference (VI) is a popular approach in Bayesian inference, that
looks for the best approximation of the posterior distribution within a
parametric family, minimizing a loss that is typically the (reverse)
Kullback-Leibler (KL) divergence. In this paper, we focus on the following
parametric family: mixtures of isotropic Gaussians (i.e., with diagonal
covariance matrices proportional to the identity) and uniform weights. We
develop a variational framework and provide efficient algorithms suited for
this family. In contrast with mixtures of Gaussian with generic covariance
matrices, this choice presents a balance between accurate approximations of
multimodal Bayesian posteriors, while being memory and computationally
efficient. Our algorithms implement gradient descent on the location of the
mixture components (the modes of the Gaussians), and either (an entropic)
Mirror or Bures descent on their variance parameters. We illustrate the
performance of our algorithms on numerical experiments.

### 8. [PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning](http://arxiv.org/pdf/2506.13652v1)

Authors: Daniele Zambon, Michele Cattaneo, Ivan Marisca, Jonas Bhend, Daniele Nerini, Cesare Alippi

Accurate weather forecasts are essential for supporting a wide range of
activities and decision-making processes, as well as mitigating the impacts of
adverse weather events. While traditional numerical weather prediction (NWP)
remains the cornerstone of operational forecasting, machine learning is
emerging as a powerful alternative for fast, flexible, and scalable
predictions. We introduce PeakWeather, a high-quality dataset of surface
weather observations collected every 10 minutes over more than 8 years from the
ground stations of the Federal Office of Meteorology and Climatology
MeteoSwiss's measurement network. The dataset includes a diverse set of
meteorological variables from 302 station locations distributed across
Switzerland's complex topography and is complemented with topographical indices
derived from digital height models for context. Ensemble forecasts from the
currently operational high-resolution NWP model are provided as a baseline
forecast against which to evaluate new approaches. The dataset's richness
supports a broad spectrum of spatiotemporal tasks, including time series
forecasting at various scales, graph structure learning, imputation, and
virtual sensing. As such, PeakWeather serves as a real-world benchmark to
advance both foundational machine learning research, meteorology, and
sensor-based applications.

### 9. [Adversarial Disentanglement by Backpropagation with Physics-Informed Variational Autoencoder](http://arxiv.org/pdf/2506.13658v1)

Authors: Ioannis Christoforos Koune, Alice Cicirello

Inference and prediction under partial knowledge of a physical system is
challenging, particularly when multiple confounding sources influence the
measured response. Explicitly accounting for these influences in physics-based
models is often infeasible due to epistemic uncertainty, cost, or time
constraints, resulting in models that fail to accurately describe the behavior
of the system. On the other hand, data-driven machine learning models such as
variational autoencoders are not guaranteed to identify a parsimonious
representation. As a result, they can suffer from poor generalization
performance and reconstruction accuracy in the regime of limited and noisy
data. We propose a physics-informed variational autoencoder architecture that
combines the interpretability of physics-based models with the flexibility of
data-driven models. To promote disentanglement of the known physics and
confounding influences, the latent space is partitioned into physically
meaningful variables that parametrize a physics-based model, and data-driven
variables that capture variability in the domain and class of the physical
system. The encoder is coupled with a decoder that integrates physics-based and
data-driven components, and constrained by an adversarial training objective
that prevents the data-driven components from overriding the known physics,
ensuring that the physics-grounded latent variables remain interpretable. We
demonstrate that the model is able to disentangle features of the input signal
and separate the known physics from confounding influences using supervision in
the form of class and domain observables. The model is evaluated on a series of
synthetic case studies relevant to engineering structures, demonstrating the
feasibility of the proposed approach.

### 10. [Gradient Boosting for Spatial Regression Models with Autoregressive Disturbances](http://arxiv.org/pdf/2506.13682v1)

Authors: Michael Balzer

Researchers in urban and regional studies increasingly deal with spatial data
that reflects geographic location and spatial relationships. As a framework for
dealing with the unique nature of spatial data, various spatial regression
models have been introduced. In this article, a novel model-based gradient
boosting algorithm for spatial regression models with autoregressive
disturbances is proposed. Due to the modular nature, the approach provides an
alternative estimation procedure which is feasible even in high-dimensional
settings where established quasi-maximum likelihood or generalized method of
moments estimators do not yield unique solutions. The approach additionally
enables data-driven variable and model selection in low- as well as
high-dimensional settings. Since the bias-variance trade-off is also controlled
in the algorithm, implicit regularization is imposed which improves prediction
accuracy on out-of-sample spatial data. Detailed simulation studies regarding
the performance of estimation, prediction and variable selection in low- and
high-dimensional settings confirm proper functionality of the proposed
methodology. To illustrative the functionality of the model-based gradient
boosting algorithm, a case study is presented where the life expectancy in
German districts is modeled incorporating a potential spatial dependence
structure.

