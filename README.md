# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-11 17:00:25.415279 PST.

### Artificial Intelligence

### 1. [Supply Chain Optimization via Generative Simulation and Iterative Decision Policies](http://arxiv.org/pdf/2507.07355v1)

Authors: Haoyue Bai, Haoyu Wang, Nanxu Gong, Xinyuan Wang, Wangyang Ying, Haifeng Chen, Yanjie Fu

High responsiveness and economic efficiency are critical objectives in supply
chain transportation, both of which are influenced by strategic decisions on
shipping mode. An integrated framework combining an efficient simulator with an
intelligent decision-making algorithm can provide an observable, low-risk
environment for transportation strategy design. An ideal simulation-decision
framework must (1) generalize effectively across various settings, (2) reflect
fine-grained transportation dynamics, (3) integrate historical experience with
predictive insights, and (4) maintain tight integration between simulation
feedback and policy refinement. We propose Sim-to-Dec framework to satisfy
these requirements. Specifically, Sim-to-Dec consists of a generative
simulation module, which leverages autoregressive modeling to simulate
continuous state changes, reducing dependence on handcrafted domain-specific
rules and enhancing robustness against data fluctuations; and a history-future
dual-aware decision model, refined iteratively through end-to-end optimization
with simulator interactions. Extensive experiments conducted on three
real-world datasets demonstrate that Sim-to-Dec significantly improves timely
delivery rates and profit.

### 2. [DrugMCTS: a drug repurposing framework combining multi-agent, RAG and Monte Carlo Tree Search](http://arxiv.org/pdf/2507.07426v1)

Authors: Zerui Yang, Yuwei Wan, Yinqiao Li, Yudai Matsuda, Tong Xie, Linqi Song

Recent advances in large language models have demonstrated considerable
potential in scientific domains such as drug discovery. However, their
effectiveness remains constrained when reasoning extends beyond the knowledge
acquired during pretraining. Conventional approaches, such as fine-tuning or
retrieval-augmented generation, face limitations in either imposing high
computational overhead or failing to fully exploit structured scientific data.
To overcome these challenges, we propose DrugMCTS, a novel framework that
synergistically integrates RAG, multi-agent collaboration, and Monte Carlo Tree
Search for drug repurposing. The framework employs five specialized agents
tasked with retrieving and analyzing molecular and protein information, thereby
enabling structured and iterative reasoning. Without requiring domain-specific
fine-tuning, DrugMCTS empowers Qwen2.5-7B-Instruct to outperform Deepseek-R1 by
over 20\%. Extensive experiments on the DrugBank and KIBA datasets demonstrate
that DrugMCTS achieves substantially higher recall and robustness compared to
both general-purpose LLMs and deep learning baselines. Our results highlight
the importance of structured reasoning, agent-based collaboration, and
feedback-driven search mechanisms in advancing LLM applications for drug
discovery.

### 3. [StarDojo: Benchmarking Open-Ended Behaviors of Agentic Multimodal LLMs in Production-Living Simulations with Stardew Valley](http://arxiv.org/pdf/2507.07445v1)

Authors: Weihao Tan, Changjiu Jiang, Yu Duan, Mingcong Lei, Jiageng Li, Yitian Hong, Xinrun Wang, Bo An

Autonomous agents navigating human society must master both production
activities and social interactions, yet existing benchmarks rarely evaluate
these skills simultaneously. To bridge this gap, we introduce StarDojo, a novel
benchmark based on Stardew Valley, designed to assess AI agents in open-ended
production-living simulations. In StarDojo, agents are tasked to perform
essential livelihood activities such as farming and crafting, while
simultaneously engaging in social interactions to establish relationships
within a vibrant community. StarDojo features 1,000 meticulously curated tasks
across five key domains: farming, crafting, exploration, combat, and social
interactions. Additionally, we provide a compact subset of 100 representative
tasks for efficient model evaluation. The benchmark offers a unified,
user-friendly interface that eliminates the need for keyboard and mouse
control, supports all major operating systems, and enables the parallel
execution of multiple environment instances, making it particularly well-suited
for evaluating the most capable foundation agents, powered by multimodal large
language models (MLLMs). Extensive evaluations of state-of-the-art MLLMs agents
demonstrate substantial limitations, with the best-performing model, GPT-4.1,
achieving only a 12.7% success rate, primarily due to challenges in visual
understanding, multimodal reasoning and low-level manipulation. As a
user-friendly environment and benchmark, StarDojo aims to facilitate further
research towards robust, open-ended agents in complex production-living
environments.

### 4. [Context Pooling: Query-specific Graph Pooling for Generic Inductive Link Prediction in Knowledge Graphs](http://arxiv.org/pdf/2507.07595v1)

Authors: Zhixiang Su, Di Wang, Chunyan Miao

Recent investigations on the effectiveness of Graph Neural Network
(GNN)-based models for link prediction in Knowledge Graphs (KGs) show that
vanilla aggregation does not significantly impact the model performance. In
this paper, we introduce a novel method, named Context Pooling, to enhance
GNN-based models' efficacy for link predictions in KGs. To our best of
knowledge, Context Pooling is the first methodology that applies graph pooling
in KGs. Additionally, Context Pooling is first-of-its-kind to enable the
generation of query-specific graphs for inductive settings, where testing
entities are unseen during training. Specifically, we devise two metrics,
namely neighborhood precision and neighborhood recall, to assess the neighbors'
logical relevance regarding the given queries, thereby enabling the subsequent
comprehensive identification of only the logically relevant neighbors for link
prediction. Our method is generic and assessed by being applied to two
state-of-the-art (SOTA) models on three public transductive and inductive
datasets, achieving SOTA performance in 42 out of 48 settings.

### 5. [Enhancing Vaccine Safety Surveillance: Extracting Vaccine Mentions from Emergency Department Triage Notes Using Fine-Tuned Large Language Models](http://arxiv.org/pdf/2507.07599v1)

Authors: Sedigh Khademi, Jim Black, Christopher Palmer, Muhammad Javed, Hazel Clothier, Jim Buttery, Gerardo Luis Dimaguila

This study evaluates fine-tuned Llama 3.2 models for extracting
vaccine-related information from emergency department triage notes to support
near real-time vaccine safety surveillance. Prompt engineering was used to
initially create a labeled dataset, which was then confirmed by human
annotators. The performance of prompt-engineered models, fine-tuned models, and
a rule-based approach was compared. The fine-tuned Llama 3 billion parameter
model outperformed other models in its accuracy of extracting vaccine names.
Model quantization enabled efficient deployment in resource-constrained
environments. Findings demonstrate the potential of large language models in
automating data extraction from emergency department notes, supporting
efficient vaccine safety surveillance and early detection of emerging adverse
events following immunization issues.

### 6. [PlanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations](http://arxiv.org/pdf/2507.07644v1)

Authors: Fedor Rodionov, Abdelrahman Eldesokey, Michael Birsak, John Femiani, Bernard Ghanem, Peter Wonka

We introduce PlanQA, a diagnostic benchmark for evaluating geometric and
spatial reasoning in large-language models (LLMs). PlanQA is grounded in
structured representations of indoor scenes, such as kitchens, living rooms,
and bedrooms, encoded in a symbolic format (e.g., JSON, XML layouts). The
benchmark includes diverse question types that test not only metric and
topological reasoning (e.g., distance, visibility, shortest paths) but also
interior design constraints such as affordance, clearance, balance, and
usability. Our results across a variety of frontier open-source and commercial
LLMs show that while models may succeed in shallow queries, they often fail to
simulate physical constraints, preserve spatial coherence, or generalize under
layout perturbation. PlanQA uncovers a clear blind spot in today's LLMs: they
do not consistently reason about real-world layouts. We hope that this
benchmark inspires new work on language models that can accurately infer and
manipulate spatial and geometric properties in practical settings.

### 7. [Stable Preference Optimization for LLMs: A Bilevel Approach Beyond Direct Preference Optimization](http://arxiv.org/pdf/2507.07723v1)

Authors: Chengtao Jian, Kai Yang, Ye Ouyang, Xiaozhou Ye

Direct Preference Optimization (DPO) has emerged as a popular and efficient
alternative to reward modeling and reinforcement learning for aligning language
models with human preferences. Despite its empirical success, the theoretical
properties and intrinsic limitations of DPO remain underexplored. In this work,
we first present a comprehensive analysis of DPO's dynamics from a probability
evolution perspective. Our analysis reveals that DPO is highly sensitive to
initialization. It also tends to misallocate probability mass, which can
inadvertently shift probability toward irrelevant or undesired responses. This
misallocation may unintentionally reinforce model bias, thereby compromising
both the stability of model alignment and the consistency with intended
preferences. Motivated by these theoretical findings, we propose a
theoretically grounded bilevel optimization framework that tightly integrate
supervised fine-tuning with an enhanced DPO objective a.k.a. stable preference
optimization. Our approach introduces a principled regularization scheme to
explicitly encourage absolute probability improvement for preferred outputs,
while maintaining stable optimization dynamics. Experiments on challenging
reasoning and summarization benchmarks elucidate that our method consistently
improves reasoning accuracy and better aligns output distributions with
intended preferences, outperforming standard DPO. Stable preference
optimization provides new insights into the design of preference-based
alignment objectives and opens up new avenues towards more reliable and
interpretable language model alignment.

### 8. [Identification of Violin Reduction via Contour Lines Classification](http://arxiv.org/pdf/2507.07743v1)

Authors: Philémon Beghin, Anne-Emmanuelle Ceulemans, François Glineur

The first violins appeared in late 16th-century Italy. Over the next 200
years, they spread across Europe and luthiers of various royal courts, eager to
experiment with new techniques, created a highly diverse family of instruments.
Around 1750, size standards were introduced to unify violin making for
orchestras and conservatories. Instruments that fell between two standards were
then reduced to a smaller size by luthiers. These reductions have an impact on
several characteristics of violins, in particular on the contour lines, i.e.
lines of constant altitude, which look more like a U for non reduced
instruments and a V for reduced ones. While such differences are observed by
experts, they have not been studied quantitatively.
  This paper presents a method for classifying violins as reduced or
non-reduced based on their contour lines. We study a corpus of 25 instruments
whose 3D geometric meshes were acquired via photogrammetry. For each
instrument, we extract 10-20 contour lines regularly spaced every millimetre.
Each line is fitted with a parabola-like curve (with an equation of the type y
= alpha*abs(x)**beta) depending on two parameters, describing how open (beta)
and how vertically stretched (alpha) the curve is. We compute additional
features from those parameters, using regressions and counting how many values
fall under some threshold. We also deal with outliers and non equal numbers of
levels, and eventually obtain a numerical profile for each instrument.
  We then apply classification methods to assess whether geometry alone can
predict size reduction. We find that distinguishing between reduced and non
reduced instruments is feasible to some degree, taking into account that a
whole spectrum of more or less transformed violins exists, for which it is more
difficult to quantify the reduction. We also find the opening parameter beta to
be the most predictive.

### 9. [Measuring AI Alignment with Human Flourishing](http://arxiv.org/pdf/2507.07787v1)

Authors: Elizabeth Hilliard, Akshaya Jagadeesh, Alex Cook, Steele Billings, Nicholas Skytland, Alicia Llewellyn, Jackson Paull, Nathan Paull, Nolan Kurylo, Keatra Nesbitt, Robert Gruenewald, Anthony Jantzi, Omar Chavez

This paper introduces the Flourishing AI Benchmark (FAI Benchmark), a novel
evaluation framework that assesses AI alignment with human flourishing across
seven dimensions: Character and Virtue, Close Social Relationships, Happiness
and Life Satisfaction, Meaning and Purpose, Mental and Physical Health,
Financial and Material Stability, and Faith and Spirituality. Unlike
traditional benchmarks that focus on technical capabilities or harm prevention,
the FAI Benchmark measures AI performance on how effectively models contribute
to the flourishing of a person across these dimensions. The benchmark evaluates
how effectively LLM AI systems align with current research models of holistic
human well-being through a comprehensive methodology that incorporates 1,229
objective and subjective questions. Using specialized judge Large Language
Models (LLMs) and cross-dimensional evaluation, the FAI Benchmark employs
geometric mean scoring to ensure balanced performance across all flourishing
dimensions. Initial testing of 28 leading language models reveals that while
some models approach holistic alignment (with the highest-scoring models
achieving 72/100), none are acceptably aligned across all dimensions,
particularly in Faith and Spirituality, Character and Virtue, and Meaning and
Purpose. This research establishes a framework for developing AI systems that
actively support human flourishing rather than merely avoiding harm, offering
significant implications for AI development, ethics, and evaluation.

### 10. [MoSE: Skill-by-Skill Mixture-of-Expert Learning for Autonomous Driving](http://arxiv.org/pdf/2507.07818v1)

Authors: Lu Xu, Jiaqian Yu, Xiongfeng Peng, Yiwei Chen, Weiming Li, Jaewook Yoo, Sunghyun Chunag, Dongwook Lee, Daehyun Ji, Chao Zhang

Recent studies show large language models (LLMs) and vision language models
(VLMs) trained using web-scale data can empower end-to-end autonomous driving
systems for a better generalization and interpretation. Specifically, by
dynamically routing inputs to specialized subsets of parameters, the
Mixture-of-Experts (MoE) technique enables general LLMs or VLMs to achieve
substantial performance improvements while maintaining computational
efficiency. However, general MoE models usually demands extensive training data
and complex optimization. In this work, inspired by the learning process of
human drivers, we propose a skill-oriented MoE, called MoSE, which mimics human
drivers' learning process and reasoning process, skill-by-skill and
step-by-step. We propose a skill-oriented routing mechanism that begins with
defining and annotating specific skills, enabling experts to identify the
necessary driving competencies for various scenarios and reasoning tasks,
thereby facilitating skill-by-skill learning. Further align the driving process
to multi-step planning in human reasoning and end-to-end driving models, we
build a hierarchical skill dataset and pretrain the router to encourage the
model to think step-by-step. Unlike multi-round dialogs, MoSE integrates
valuable auxiliary tasks (e.g.\ description, reasoning, planning) in one single
forward process without introducing any extra computational cost. With less
than 3B sparsely activated parameters, our model outperforms several 8B+
parameters on CODA AD corner case reasoning task. Compared to existing methods
based on open-source models and data, our approach achieves state-of-the-art
performance with significantly reduced activated model size (at least by
$62.5\%$) with a single-turn conversation.

### Hardware Architecture

### 1. [Accelerating Transposed Convolutions on FPGA-based Edge Devices](http://arxiv.org/pdf/2507.07683v1)

Authors: Jude Haris, José Cano

Transposed Convolutions (TCONV) enable the up-scaling mechanism within
generative Artificial Intelligence (AI) models. However, the predominant
Input-Oriented Mapping (IOM) method for implementing TCONV has complex output
mapping, overlapping sums, and ineffectual computations. These inefficiencies
further exacerbate the performance bottleneck of TCONV and generative models on
resource-constrained edge devices. To address this problem, in this paper we
propose MM2IM, a hardware-software co-designed accelerator that combines Matrix
Multiplication (MatMul) with col2IM to process TCONV layers on
resource-constrained edge devices efficiently. Using the SECDA-TFLite design
toolkit, we implement MM2IM and evaluate its performance across 261 TCONV
problem configurations, achieving an average speedup of 1.9x against a
dual-thread ARM Neon optimized CPU baseline. We then evaluate the performance
of MM2IM on a range of TCONV layers from well-known generative models achieving
up to 4.2x speedup, and compare it against similar resource-constrained TCONV
accelerators, outperforming them by at least 2x GOPs/DSP. Finally, we evaluate
MM2IM on the DCGAN and pix2pix GAN models, achieving up to 3x speedup and 2.4x
energy reduction against the CPU baseline.

### Computational Complexity

### 1. [Testing Isomorphism of Boolean Functions over Finite Abelian Groups](http://arxiv.org/pdf/2507.07654v1)

Authors: Swarnalipa Datta, Arijit Ghosh, Chandrima Kayal, Manaswi Paraashar, Manmatha Roy

Let $f$ and $g$ be Boolean functions over a finite Abelian group
$\mathcal{G}$, where $g$ is fully known, and we have {\em query access} to $f$,
that is, given any $x \in \mathcal{G}$ we can get the value $f(x)$. We study
the tolerant isomorphism testing problem: given $\epsilon \geq 0$ and $\tau >
0$, we seek to determine, with minimal queries, whether there exists an
automorphism $\sigma$ of $\mathcal{G}$ such that the fractional Hamming
distance between $f \circ \sigma$ and $g$ is at most $\epsilon$, or whether for
all automorphisms $\sigma$, the distance is at least $\epsilon + \tau$.
  We design an efficient tolerant testing algorithm for this problem, with
query complexity $\mathrm{poly}\left( s, 1/\tau \right)$, where $s$ bounds the
spectral norm of $g$. Additionally, we present an improved algorithm when $g$
is Fourier sparse.
  Our approach uses key concepts from Abelian group theory and Fourier
analysis, including the annihilator of a subgroup, Pontryagin duality, and a
pseudo inner-product for finite Abelian groups. We believe these techniques
will find further applications in property testing.

### 2. [Finding One Local Optimum Is Easy -- But What about Two?](http://arxiv.org/pdf/2507.07524v1)

Authors: Yasuaki Kobayashi, Kazuhiro Kurita, Yutaro Yamaguchi

The class PLS (Polynomial Local Search) captures the complexity of finding a
solution that is locally optimal and has proven to be an important concept in
the theory of local search. It has been shown that local search versions of
various combinatorial optimization problems, such as Maximum Independent Set
and Max Cut, are complete for this class. Such computational intractability
typically arises in local search problems allowing arbitrary weights; in
contrast, for unweighted problems, locally optimal solutions can be found in
polynomial time under standard settings. In this paper, we pursue the
complexity of local search problems from a different angle: We show that
computing two locally optimal solutions is NP-hard for various natural
unweighted local search problems, including Maximum Independent Set, Minimum
Dominating Set, Max SAT, and Max Cut. We also discuss several tractable cases
for finding two (or more) local optimal solutions.

### 3. [On the Complexity of Hyperpath and Minimal Separator Enumeration in Directed Hypergraphs](http://arxiv.org/pdf/2507.07528v1)

Authors: Kazuhiro Kurita, Kevin Mann

In this paper, we address the enumeration of (induced) $s$-$t$ paths and
minimal $s$-$t$ separators. These problems are some of the most famous
classical enumeration problems that can be solved in polynomial delay by simple
backtracking for a (un)directed graph. As a generalization of these problems,
we consider the (induced) $s$-$t$ hyperpath and minimal $s$-$t$ separator
enumeration in a \emph{directed hypergraph}. We show that extending these
classical enumeration problems to directed hypergraphs drastically changes
their complexity. More precisely, there are no output-polynomial time
algorithms for the enumeration of induced $s$-$t$ hyperpaths and minimal
$s$-$t$ separators unless $P = NP$, and if there is an output-polynomial time
algorithm for the $s$-$t$ hyperpath enumeration, then the minimal transversal
enumeration can be solved in output polynomial time even if a directed
hypergraph is $BF$-hypergraph. Since the existence of an output-polynomial time
algorithm for the minimal transversal enumeration has remained an open problem
for over 45 years, it indicates that the $s$-$t$ hyperpath enumeration for a
$BF$-hypergraph is not an easy problem. As a positive result, the $s$-$t$
hyperpath enumeration for a $B$-hypergraph can be solved in polynomial delay by
backtracking.

### 4. [The Richness of CSP Non-redundancy](http://arxiv.org/pdf/2507.07942v1)

Authors: Joshua Brakensiek, Venkatesan Guruswami, Bart M. P. Jansen, Victor Lagerkvist, Magnus Wahlström

In the field of constraint satisfaction problems (CSP), a clause is called
redundant if its satisfaction is implied by satisfying all other clauses. An
instance of CSP$(P)$ is called non-redundant if it does not contain any
redundant clause. The non-redundancy (NRD) of a predicate $P$ is the maximum
number of clauses in a non-redundant instance of CSP$(P)$, as a function of the
number of variables $n$. Recent progress has shown that non-redundancy is
crucially linked to many other important questions in computer science and
mathematics including sparsification, kernelization, query complexity,
universal algebra, and extremal combinatorics. Given that non-redundancy is a
nexus for many of these important problems, the central goal of this paper is
to more deeply understand non-redundancy.
  Our first main result shows that for every rational number $r \ge 1$, there
exists a finite CSP predicate $P$ such that the non-redundancy of $P$ is
$\Theta(n^r)$. Our second main result explores the concept of conditional
non-redundancy first coined by Brakensiek and Guruswami [STOC 2025]. We
completely classify the conditional non-redundancy of all binary predicates
(i.e., constraints on two variables) by connecting these non-redundancy
problems to the structure of high-girth graphs in extremal combinatorics.
  Inspired by these concrete results, we build off the work of Carbonnel [CP
2022] to develop an algebraic theory of conditional non-redundancy. As an
application of this algebraic theory, we revisit the notion of Mal'tsev
embeddings, which is the most general technique known to date for establishing
that a predicate has linear non-redundancy. For example, we provide the first
example of predicate with a Mal'tsev embedding that cannot be attributed to the
structure of an Abelian group, but rather to the structure of the quantum Pauli
group.

### 5. [Turing complete Navier-Stokes steady states via cosymplectic geometry](http://arxiv.org/pdf/2507.07696v1)

Authors: Søren Dyhr, Ángel González-Prieto, Eva Miranda, Daniel Peralta-Salas

In this article, we construct stationary solutions to the Navier-Stokes
equations on certain Riemannian $3$-manifolds that exhibit Turing completeness,
in the sense that they are capable of performing universal computation. This
universality arises on manifolds admitting nonvanishing harmonic 1-forms, thus
showing that computational universality is not obstructed by viscosity,
provided the underlying geometry satisfies a mild cohomological condition. The
proof makes use of a correspondence between nonvanishing harmonic $1$-forms and
cosymplectic geometry, which extends the classical correspondence between
Beltrami fields and Reeb flows on contact manifolds.

### Computational Engineering

### 1. [The Pandora's Box Problem with Sequential Inspections](http://arxiv.org/pdf/2507.07508v1)

Authors: Ali Aouad, Jingwei Ji, Yaron Shaposhnik

The Pandora's box problem (Weitzman 1979) is a core model in economic theory
that captures an agent's (Pandora's) search for the best alternative (box). We
study an important generalization of the problem where the agent can either
fully open boxes for a certain fee to reveal their exact values or partially
open them at a reduced cost. This introduces a new tradeoff between information
acquisition and cost efficiency. We establish a hardness result and employ an
array of techniques in stochastic optimization to provide a comprehensive
analysis of this model. This includes (1) the identification of structural
properties of the optimal policy that provide insights about optimal decisions;
(2) the derivation of problem relaxations and provably near-optimal solutions;
(3) the characterization of the optimal policy in special yet non-trivial
cases; and (4) an extensive numerical study that compares the performance of
various policies, and which provides additional insights about the optimal
policy. Throughout, we show that intuitive threshold-based policies that extend
the Pandora's box optimal solution can effectively guide search decisions.

### 2. [Meshless projection model-order reduction via reference spaces for smoothed-particle hydrodynamics](http://arxiv.org/pdf/2507.07830v1)

Authors: Steven N. Rodriguez, Steven L. Brunton, Liam K. Magargal, Parisa Khodabakshi, Justin W. Jaworski, Nicoleta A. Apetre, John C. Steuben, John G. Michopoulos, Athanasios Iliopoulos

This work proposes a model-order reduction framework for the meshless weakly
compressible smoothed particle hydrodynamics (SPH) method. The proposed
framework introduces the concept of modal reference spaces to overcome the
challenges of discovering low-dimensional subspaces from unstructured, dynamic,
and mixing numerical topology that is often seen in SPH simulations. The
proposed modal reference spaces enable a low-dimensional representation of the
SPH field equations while maintaining their inherent meshless qualities. Modal
reference spaces are constructed by projecting SPH snapshot data onto a
reference space where low-dimensionality of field quantities can be discovered
via traditional modal decomposition techniques (e.g., the proper orthogonal
decomposition (POD)). Modal quantities are mapped back to the meshless SPH
space via scattered data interpolation during the online predictive stage. The
proposed model-order reduction framework is cast into the \emph{meshless}
Galerkin POD (GPOD) and the Adjoint Petrov--Galerkin (APG) projection
model-order reduction (PMOR) formulation. The PMORs are tested on three
numerical experiments: 1) the Taylor--Green vortex; 2) lid-driven cavity; and
3) flow past an open cavity. Results show good agreement in reconstructed and
predictive velocity fields, which showcase the ability of the proposed
framework to evolve the unstructured, dynamic, and mixing SPH field equations
in a low-dimensional subspace. Results also show that the pressure field is
sensitive to the projection error due to the stiff weakly-compressible
assumption made in the current SPH framework, but can be alleviated through
nonlinear approximations, such as the APG approach. Ultimately, the presented
meshless model-order reduction framework marks a step toward enabling drastic
cost savings of SPH simulations.

### 3. [Computationally Efficient Information-Driven Optical Design with Interchanging Optimization](http://arxiv.org/pdf/2507.07789v1)

Authors: Eric Markley, Henry Pinkard, Leyla Kabuli, Nalini Singh, Laura Waller

Recent work has demonstrated that imaging systems can be evaluated through
the information content of their measurements alone, enabling
application-agnostic optical design that avoids computational decoding
challenges. Information-Driven Encoder Analysis Learning (IDEAL) was proposed
to automate this process through gradient-based. In this work, we study IDEAL
across diverse imaging systems and find that it suffers from high memory usage,
long runtimes, and a potentially mismatched objective function due to
end-to-end differentiability requirements. We introduce IDEAL with
Interchanging Optimization (IDEAL-IO), a method that decouples density
estimation from optical parameter optimization by alternating between fitting
models to current measurements and updating optical parameters using fixed
models for information estimation. This approach reduces runtime and memory
usage by up to 6x while enabling more expressive density models that guide
optimization toward superior designs. We validate our method on diffractive
optics, lensless imaging, and snapshot 3D microscopy applications, establishing
information-theoretic optimization as a practical, scalable strategy for
real-world imaging system design.

### Computational Geometry

### 1. [The Smooth Power of the "Neandertal Method"](http://arxiv.org/pdf/2507.07569v1)

Authors: Aaron Montag, Tim Reinhardt, Jürgen Richter-Gebert

We describe an algorithmic method to transform a Euclidean wallpaper pattern
into a Circle Limit-style picture \`a la Escher. The design goals for the
method are to be mathematically sound, aesthetically pleasing and fast to
compute. It turns out that a certain class of conformal maps is particularly
well-suited for the problem. Moreover, in our specific application, a very
simple method, sometimes jokingly called the "Neandertal Method" for its almost
brutal simplicity, proves to be highly efficient, as it can easily be
parallelized to be run on the GPU, unlike many other approaches.

### 2. [Approximation Depth of Convex Polytopes](http://arxiv.org/pdf/2507.07779v1)

Authors: Egor Bakaev, Florestan Brunck, Amir Yehudayoff

We study approximations of polytopes in the standard model for computing
polytopes using Minkowski sums and (convex hulls of) unions. Specifically, we
study the ability to approximate a target polytope by polytopes of a given
depth. Our main results imply that simplices can only be ``trivially
approximated''. On the way, we obtain a characterization of simplices as the
only ``outer additive'' convex bodies.

### Computation and Language

### 1. [SAND: Boosting LLM Agents with Self-Taught Action Deliberation](http://arxiv.org/pdf/2507.07441v1)

Authors: Yu Xia, Yiran Jenny Shen, Junda Wu, Tong Yu, Sungchul Kim, Ryan A. Rossi, Lina Yao, Julian McAuley

Large Language Model (LLM) agents are commonly tuned with supervised
finetuning on ReAct-style expert trajectories or preference optimization over
pairwise rollouts. Most of these methods focus on imitating specific expert
behaviors or promoting chosen reasoning thoughts and actions over rejected
ones. However, without reasoning and comparing over alternatives actions, LLM
agents finetuned with these methods may over-commit towards seemingly plausible
but suboptimal actions due to limited action space exploration. To address
this, in this paper we propose Self-taught ActioN Deliberation (SAND)
framework, enabling LLM agents to explicitly deliberate over candidate actions
before committing to one. To tackle the challenges of when and what to
deliberate given large action space and step-level action evaluation, we
incorporate self-consistency action sampling and execution-guided action
critique to help synthesize step-wise action deliberation thoughts using the
base model of the LLM agent. In an iterative manner, the deliberation
trajectories are then used to finetune the LLM agent itself. Evaluating on two
representative interactive agent tasks, SAND achieves an average 20%
improvement over initial supervised finetuning and also outperforms
state-of-the-art agent tuning approaches.

### 2. [RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning](http://arxiv.org/pdf/2507.07451v1)

Authors: Hongzhi Zhang, Jia Fu, Jingyuan Zhang, Kai Fu, Qi Wang, Fuzheng Zhang, Guorui Zhou

Reinforcement learning (RL) for large language models is an energy-intensive
endeavor: training can be unstable, and the policy may gradually drift away
from its pretrained weights. We present \emph{RLEP}\, -- \,Reinforcement
Learning with Experience rePlay\, -- \,a two-phase framework that first
collects verified trajectories and then replays them during subsequent
training. At every update step, the policy is optimized on mini-batches that
blend newly generated rollouts with these replayed successes. By replaying
high-quality examples, RLEP steers the model away from fruitless exploration,
focuses learning on promising reasoning paths, and delivers both faster
convergence and stronger final performance. On the Qwen2.5-Math-7B base model,
RLEP reaches baseline peak accuracy with substantially fewer updates and
ultimately surpasses it, improving accuracy on AIME-2024 from 38.2% to 39.9%,
on AIME-2025 from 19.8% to 22.3%, and on AMC-2023 from 77.0% to 82.2%. Our
code, datasets, and checkpoints are publicly available at
https://github.com/Kwai-Klear/RLEP to facilitate reproducibility and further
research.

### 3. [Triadic Multi-party Voice Activity Projection for Turn-taking in Spoken Dialogue Systems](http://arxiv.org/pdf/2507.07518v1)

Authors: Mikey Elmers, Koji Inoue, Divesh Lala, Tatsuya Kawahara

Turn-taking is a fundamental component of spoken dialogue, however
conventional studies mostly involve dyadic settings. This work focuses on
applying voice activity projection (VAP) to predict upcoming turn-taking in
triadic multi-party scenarios. The goal of VAP models is to predict the future
voice activity for each speaker utilizing only acoustic data. This is the first
study to extend VAP into triadic conversation. We trained multiple models on a
Japanese triadic dataset where participants discussed a variety of topics. We
found that the VAP trained on triadic conversation outperformed the baseline
for all models but that the type of conversation affected the accuracy. This
study establishes that VAP can be used for turn-taking in triadic dialogue
scenarios. Future work will incorporate this triadic VAP turn-taking model into
spoken dialogue systems.

### 4. [The Synergy Dilemma of Long-CoT SFT and RL: Investigating Post-Training Techniques for Reasoning VLMs](http://arxiv.org/pdf/2507.07562v1)

Authors: Jierun Chen, Tiezheng Yu, Haoli Bai, Lewei Yao, Jiannan Wu, Kaican Li, Fei Mi, Chaofan Tao, Lei Zhu, Manyi Zhang, Xiaohui Li, Lu Hou, Lifeng Shang, Qun Liu

Large vision-language models (VLMs) increasingly adopt post-training
techniques such as long chain-of-thought (CoT) supervised fine-tuning (SFT) and
reinforcement learning (RL) to elicit sophisticated reasoning. While these
methods exhibit synergy in language-only models, their joint effectiveness in
VLMs remains uncertain. We present a systematic investigation into the distinct
roles and interplay of long-CoT SFT and RL across multiple multimodal reasoning
benchmarks. We find that SFT improves performance on difficult questions by
in-depth, structured reasoning, but introduces verbosity and degrades
performance on simpler ones. In contrast, RL promotes generalization and
brevity, yielding consistent improvements across all difficulty levels, though
the improvements on the hardest questions are less prominent compared to SFT.
Surprisingly, combining them through two-staged, interleaved, or progressive
training strategies, as well as data mixing and model merging, all fails to
produce additive benefits, instead leading to trade-offs in accuracy, reasoning
style, and response length. This ``synergy dilemma'' highlights the need for
more seamless and adaptive approaches to unlock the full potential of combined
post-training techniques for reasoning VLMs.

### 5. [FrugalRAG: Learning to retrieve and reason for multi-hop QA](http://arxiv.org/pdf/2507.07634v1)

Authors: Abhinav Java, Srivathsan Koundinyan, Nagarajan Natarajan, Amit Sharma

We consider the problem of answering complex questions, given access to a
large unstructured document corpus. The de facto approach to solving the
problem is to leverage language models that (iteratively) retrieve and reason
through the retrieved documents, until the model has sufficient information to
generate an answer. Attempts at improving this approach focus on
retrieval-augmented generation (RAG) metrics such as accuracy and recall and
can be categorized into two types: (a) fine-tuning on large question answering
(QA) datasets augmented with chain-of-thought traces, and (b) leveraging
RL-based fine-tuning techniques that rely on question-document relevance
signals. However, efficiency in the number of retrieval searches is an equally
important metric, which has received less attention. In this work, we show
that: (1) Large-scale fine-tuning is not needed to improve RAG metrics,
contrary to popular claims in recent literature. Specifically, a standard ReAct
pipeline with improved prompts can outperform state-of-the-art methods on
benchmarks such as HotPotQA. (2) Supervised and RL-based fine-tuning can help
RAG from the perspective of frugality, i.e., the latency due to number of
searches at inference time. For example, we show that we can achieve
competitive RAG metrics at nearly half the cost (in terms of number of
searches) on popular RAG benchmarks, using the same base model, and at a small
training cost (1000 examples).

### 6. [Lost in Pronunciation: Detecting Chinese Offensive Language Disguised by Phonetic Cloaking Replacement](http://arxiv.org/pdf/2507.07640v1)

Authors: Haotan Guo, Jianfei He, Jiayuan Ma, Hongbin Na, Zimu Wang, Haiyang Zhang, Qi Chen, Wei Wang, Zijing Shi, Tao Shen, Ling Chen

Phonetic Cloaking Replacement (PCR), defined as the deliberate use of
homophonic or near-homophonic variants to hide toxic intent, has become a major
obstacle to Chinese content moderation. While this problem is well-recognized,
existing evaluations predominantly rely on rule-based, synthetic perturbations
that ignore the creativity of real users. We organize PCR into a four-way
surface-form taxonomy and compile \ours, a dataset of 500 naturally occurring,
phonetically cloaked offensive posts gathered from the RedNote platform.
Benchmarking state-of-the-art LLMs on this dataset exposes a serious weakness:
the best model reaches only an F1-score of 0.672, and zero-shot
chain-of-thought prompting pushes performance even lower. Guided by error
analysis, we revisit a Pinyin-based prompting strategy that earlier studies
judged ineffective and show that it recovers much of the lost accuracy. This
study offers the first comprehensive taxonomy of Chinese PCR, a realistic
benchmark that reveals current detectors' limits, and a lightweight mitigation
technique that advances research on robust toxicity detection.

### 7. [An Automated Length-Aware Quality Metric for Summarization](http://arxiv.org/pdf/2507.07653v1)

Authors: Andrew D. Foland

This paper proposes NOrmed Index of Retention (NOIR), a quantitative
objective metric for evaluating summarization quality of arbitrary texts that
relies on both the retention of semantic meaning and the summary length
compression. This gives a measure of how well the recall-compression tradeoff
is managed, the most important skill in summarization. Experiments demonstrate
that NOIR effectively captures the token-length / semantic retention tradeoff
of a summarizer and correlates to human perception of sumarization quality.
Using a language model-embedding to measure semantic similarity, it provides an
automated alternative for assessing summarization quality without relying on
time-consuming human-generated reference summaries. The proposed metric can be
applied to various summarization tasks, offering an automated tool for
evaluating and improving summarization algorithms, summarization prompts, and
synthetically-generated summaries.

### 8. [SAS: Simulated Attention Score](http://arxiv.org/pdf/2507.07694v1)

Authors: Chuanyang Zheng, Jiankai Sun, Yihang Gao, Yuehao Wang, Peihao Wang, Jing Xiong, Liliang Ren, Hao Cheng, Janardhan Kulkarni, Yelong Shen, Atlas Wang, Mac Schwager, Anderson Schneider, Xiaodong Liu, Jianfeng Gao

The attention mechanism is a core component of the Transformer architecture.
Various methods have been developed to compute attention scores, including
multi-head attention (MHA), multi-query attention, group-query attention and so
on. We further analyze the MHA and observe that its performance improves as the
number of attention heads increases, provided the hidden size per head remains
sufficiently large. Therefore, increasing both the head count and hidden size
per head with minimal parameter overhead can lead to significant performance
gains at a low cost. Motivated by this insight, we introduce Simulated
Attention Score (SAS), which maintains a compact model size while simulating a
larger number of attention heads and hidden feature dimension per head. This is
achieved by projecting a low-dimensional head representation into a
higher-dimensional space, effectively increasing attention capacity without
increasing parameter count. Beyond the head representations, we further extend
the simulation approach to feature dimension of the key and query embeddings,
enhancing expressiveness by mimicking the behavior of a larger model while
preserving the original model size. To control the parameter cost, we also
propose Parameter-Efficient Attention Aggregation (PEAA). Comprehensive
experiments on a variety of datasets and tasks demonstrate the effectiveness of
the proposed SAS method, achieving significant improvements over different
attention variants.

### 9. [Code-Switching in End-to-End Automatic Speech Recognition: A Systematic Literature Review](http://arxiv.org/pdf/2507.07741v1)

Authors: Maha Tufail Agro, Atharva Kulkarni, Karima Kadaoui, Zeerak Talat, Hanan Aldarmaki

Motivated by a growing research interest into automatic speech recognition
(ASR), and the growing body of work for languages in which code-switching (CS)
often occurs, we present a systematic literature review of code-switching in
end-to-end ASR models. We collect and manually annotate papers published in
peer reviewed venues. We document the languages considered, datasets, metrics,
model choices, and performance, and present a discussion of challenges in
end-to-end ASR for code-switching. Our analysis thus provides insights on
current research efforts and available resources as well as opportunities and
gaps to guide future research.

### 10. [StreamUni: Achieving Streaming Speech Translation with a Unified Large Speech-Language Model](http://arxiv.org/pdf/2507.07803v1)

Authors: Shoutao Guo, Xiang Li, Shaolei Zhang, Mengge Liu, Wei Chen, Yang Feng

Streaming speech translation (StreamST) requires determining appropriate
timing, known as policy, to generate translations while continuously receiving
source speech inputs, balancing low latency with high translation quality.
However, existing StreamST methods typically operate on sentence-level speech
segments, referred to as simultaneous speech translation (SimulST). In
practice, they require collaboration with segmentation models to accomplish
StreamST, where the truncated speech segments constrain SimulST models to make
policy decisions and generate translations based on limited contextual
information. Moreover, SimulST models struggle to learn effective policies due
to the complexity of speech inputs and cross-lingual generation. To address
these challenges, we propose StreamUni, which achieves StreamST through a
unified Large Speech-Language Model (LSLM). Specifically, StreamUni
incorporates speech Chain-of-Thought (CoT) in guiding the LSLM to generate
multi-stage outputs. Leveraging these multi-stage outputs, StreamUni
simultaneously accomplishes speech segmentation, policy decision, and
translation generation, completing StreamST without requiring massive
policy-specific training. Additionally, we propose a streaming CoT training
method that enhances low-latency policy decisions and generation capabilities
using limited CoT data. Experiments demonstrate that our approach achieves
state-of-the-art performance on StreamST tasks.

### Cryptography and Security

### 1. [Shuffling for Semantic Secrecy](http://arxiv.org/pdf/2507.07401v1)

Authors: Fupei Chen, Liyao Xiang, Haoxiang Sun, Hei Victor Cheng, Kaiming Shen

Deep learning draws heavily on the latest progress in semantic
communications. The present paper aims to examine the security aspect of this
cutting-edge technique from a novel shuffling perspective. Our goal is to
improve upon the conventional secure coding scheme to strike a desirable
tradeoff between transmission rate and leakage rate. To be more specific, for a
wiretap channel, we seek to maximize the transmission rate while minimizing the
semantic error probability under the given leakage rate constraint. Toward this
end, we devise a novel semantic security communication system wherein the
random shuffling pattern plays the role of the shared secret key. Intuitively,
the permutation of feature sequences via shuffling would distort the semantic
essence of the target data to a sufficient extent so that eavesdroppers cannot
access it anymore. The proposed random shuffling method also exhibits its
flexibility in working for the existing semantic communication system as a
plugin. Simulations demonstrate the significant advantage of the proposed
method over the benchmark in boosting secure transmission, especially when
channels are prone to strong noise and unpredictable fading.

### 2. [RADAR: a Radio-based Analytics for Dynamic Association and Recognition of pseudonyms in VANETs](http://arxiv.org/pdf/2507.07732v1)

Authors: Giovanni Gambigliani Zoccoli, Filip Valgimigli, Dario Stabili, Mirco Marchetti

This paper presents RADAR, a tracking algorithm for vehicles participating in
Cooperative Intelligent Transportation Systems (C-ITS) that exploits multiple
radio signals emitted by a modern vehicle to break privacy-preserving pseudonym
schemes deployed in VANETs. This study shows that by combining Dedicated Short
Range Communication (DSRC) and Wi-Fi probe request messages broadcast by the
vehicle, it is possible to improve tracking over standard de-anonymization
approaches that only leverage DSRC, especially in realistic scenarios where the
attacker does not have full coverage of the entire vehicle path. The
experimental evaluation compares three different metrics for pseudonym and
Wi-Fi probe identifier association (Count, Statistical RSSI, and Pearson RSSI),
demonstrating that the Pearson RSSI metric is better at tracking vehicles under
pseudonym-changing schemes in all scenarios and against previous works. As an
additional contribution to the state-of-the-art, we publicly release all
implementations and simulation scenarios used in this work.

### 3. [The Trust Fabric: Decentralized Interoperability and Economic Coordination for the Agentic Web](http://arxiv.org/pdf/2507.07901v1)

Authors: Sree Bhargavi Balija, Rekha Singal, Abhishek Singh, Ramesh Raskar, Erfan Darzi, Raghu Bala, Thomas Hardjono, Ken Huang

The fragmentation of AI agent ecosystems has created urgent demands for
interoperability, trust, and economic coordination that current protocols --
including MCP (Hou et al., 2025), A2A (Habler et al., 2025), ACP (Liu et al.,
2025), and Cisco's AGP (Edwards, 2025) -- cannot address at scale. We present
the Nanda Unified Architecture, a decentralized framework built around three
core innovations: fast DID-based agent discovery through distributed
registries, semantic agent cards with verifiable credentials and composability
profiles, and a dynamic trust layer that integrates behavioral attestations
with policy compliance. The system introduces X42/H42 micropayments for
economic coordination and MAESTRO, a security framework incorporating
Synergetics' patented AgentTalk protocol (US Patent 12,244,584 B1) and secure
containerization. Real-world deployments demonstrate 99.9 percent compliance in
healthcare applications and substantial monthly transaction volumes with strong
privacy guarantees. By unifying MIT's trust research with production
deployments from Cisco and Synergetics, we show how cryptographic proofs and
policy-as-code transform agents into trust-anchored participants in a
decentralized economy (Lakshmanan, 2025; Sha, 2025). The result enables a
globally interoperable Internet of Agents where trust becomes the native
currency of collaboration across both enterprise and Web3 ecosystems.

### 4. [KeyDroid: A Large-Scale Analysis of Secure Key Storage in Android Apps](http://arxiv.org/pdf/2507.07927v1)

Authors: Jenny Blessing, Ross J. Anderson, Alastair R. Beresford

Most contemporary mobile devices offer hardware-backed storage for
cryptographic keys, user data, and other sensitive credentials. Such hardware
protects credentials from extraction by an adversary who has compromised the
main operating system, such as a malicious third-party app. Since 2011, Android
app developers can access trusted hardware via the Android Keystore API. In
this work, we conduct the first comprehensive survey of hardware-backed key
storage in Android devices. We analyze 490 119 Android apps, collecting data on
how trusted hardware is used by app developers (if used at all) and
cross-referencing our findings with sensitive user data collected by each app,
as self-reported by developers via the Play Store's data safety labels.
  We find that despite industry-wide initiatives to encourage adoption, 56.3%
of apps self-reporting as processing sensitive user data do not use Android's
trusted hardware capabilities at all, while just 5.03% of apps collecting some
form of sensitive data use the strongest form of trusted hardware, a secure
element distinct from the main processor. To better understand the potential
downsides of using secure hardware, we conduct the first empirical analysis of
trusted hardware performance in mobile devices, measuring the runtime of common
cryptographic operations across both software- and hardware-backed keystores.
We find that while hardware-backed key storage using a coprocessor is viable
for most common cryptographic operations, secure elements capable of preventing
more advanced attacks make performance infeasible for symmetric encryption with
non-negligible payloads and any kind of asymmetric encryption.

### 5. [EinHops: Einsum Notation for Expressive Homomorphic Operations on RNS-CKKS Tensors](http://arxiv.org/pdf/2507.07972v1)

Authors: Karthik Garimella, Austin Ebel, Brandon Reagen

Fully Homomorphic Encryption (FHE) is an encryption scheme that allows for
computation to be performed directly on encrypted data, effectively closing the
loop on secure and outsourced computing. Data is encrypted not only during rest
and transit, but also during processing. However, FHE provides a limited
instruction set: SIMD addition, SIMD multiplication, and cyclic rotation of 1-D
vectors. This restriction makes performing multi-dimensional tensor operations
challenging. Practitioners must pack these tensors into 1-D vectors and map
tensor operations onto this one-dimensional layout rather than their
traditional nested structure. And while prior systems have made significant
strides in automating this process, they often hide critical packing decisions
behind layers of abstraction, making debugging, optimizing, and building on top
of these systems difficult.
  In this work, we approach multi-dimensional tensor operations in FHE through
Einstein summation (einsum) notation. Einsum notation explicitly encodes
dimensional structure and operations in its syntax, naturally exposing how
tensors should be packed and transformed. We decompose einsum expressions into
a fixed set of FHE-friendly operations. We implement our design and present
EinHops, a minimalist system that factors einsum expressions into a fixed
sequence of FHE operations. EinHops enables developers to perform encrypted
tensor operations using FHE while maintaining full visibility into the
underlying packing strategy. We evaluate EinHops on a range of tensor
operations from a simple transpose to complex multi-dimensional contractions.
We show that the explicit nature of einsum notation allows us to build an FHE
tensor system that is simple, general, and interpretable. We open-source
EinHops at the following repository: https://github.com/baahl-nyu/einhops.

### 6. [Defending Against Prompt Injection With a Few DefensiveTokens](http://arxiv.org/pdf/2507.07974v1)

Authors: Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, David Wagner

When large language model (LLM) systems interact with external data to
perform complex tasks, a new attack, namely prompt injection, becomes a
significant threat. By injecting instructions into the data accessed by the
system, the attacker is able to override the initial user task with an
arbitrary task directed by the attacker. To secure the system, test-time
defenses, e.g., defensive prompting, have been proposed for system developers
to attain security only when needed in a flexible manner. However, they are
much less effective than training-time defenses that change the model
parameters. Motivated by this, we propose DefensiveToken, a test-time defense
with prompt injection robustness comparable to training-time alternatives.
DefensiveTokens are newly inserted as special tokens, whose embeddings are
optimized for security. In security-sensitive cases, system developers can
append a few DefensiveTokens before the LLM input to achieve security with a
minimal utility drop. In scenarios where security is less of a concern,
developers can simply skip DefensiveTokens; the LLM system remains the same as
there is no defense, generating high-quality responses. Thus, DefensiveTokens,
if released alongside the model, allow a flexible switch between the
state-of-the-art (SOTA) utility and almost-SOTA security at test time. The code
is available at https://github.com/Sizhe-Chen/DefensiveToken.

### 7. [Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation by Object Tracking](http://arxiv.org/pdf/2507.07483v1)

Authors: Qiangqiang Wu, Yi Yu, Chenqi Kong, Ziquan Liu, Jia Wan, Haoliang Li, Alex C. Kot, Antoni B. Chan

With the rise of social media, vast amounts of user-uploaded videos (e.g.,
YouTube) are utilized as training data for Visual Object Tracking (VOT).
However, the VOT community has largely overlooked video data-privacy issues, as
many private videos have been collected and used for training commercial models
without authorization. To alleviate these issues, this paper presents the first
investigation on preventing personal video data from unauthorized exploitation
by deep trackers. Existing methods for preventing unauthorized data use
primarily focus on image-based tasks (e.g., image classification), directly
applying them to videos reveals several limitations, including inefficiency,
limited effectiveness, and poor generalizability. To address these issues, we
propose a novel generative framework for generating Temporal Unlearnable
Examples (TUEs), and whose efficient computation makes it scalable for usage on
large-scale video datasets. The trackers trained w/ TUEs heavily rely on
unlearnable noises for temporal matching, ignoring the original data structure
and thus ensuring training video data-privacy. To enhance the effectiveness of
TUEs, we introduce a temporal contrastive loss, which further corrupts the
learning of existing trackers when using our TUEs for training. Extensive
experiments demonstrate that our approach achieves state-of-the-art performance
in video data-privacy protection, with strong transferability across VOT
models, datasets, and temporal matching tasks.

### 8. [Rainbow Artifacts from Electromagnetic Signal Injection Attacks on Image Sensors](http://arxiv.org/pdf/2507.07773v1)

Authors: Youqian Zhang, Xinyu Ji, Zhihao Wang, Qinhong Jiang

Image sensors are integral to a wide range of safety- and security-critical
systems, including surveillance infrastructure, autonomous vehicles, and
industrial automation. These systems rely on the integrity of visual data to
make decisions. In this work, we investigate a novel class of electromagnetic
signal injection attacks that target the analog domain of image sensors,
allowing adversaries to manipulate raw visual inputs without triggering
conventional digital integrity checks. We uncover a previously undocumented
attack phenomenon on CMOS image sensors: rainbow-like color artifacts induced
in images captured by image sensors through carefully tuned electromagnetic
interference. We further evaluate the impact of these attacks on
state-of-the-art object detection models, showing that the injected artifacts
propagate through the image signal processing pipeline and lead to significant
mispredictions. Our findings highlight a critical and underexplored
vulnerability in the visual perception stack, highlighting the need for more
robust defenses against physical-layer attacks in such systems.

### 9. [Can Large Language Models Improve Phishing Defense? A Large-Scale Controlled Experiment on Warning Dialogue Explanations](http://arxiv.org/pdf/2507.07916v1)

Authors: Federico Maria Cau, Giuseppe Desolda, Francesco Greco, Lucio Davide Spano, Luca Viganò

Phishing has become a prominent risk in modern cybersecurity, often used to
bypass technological defences by exploiting predictable human behaviour.
Warning dialogues are a standard mitigation measure, but the lack of
explanatory clarity and static content limits their effectiveness. In this
paper, we report on our research to assess the capacity of Large Language
Models (LLMs) to generate clear, concise, and scalable explanations for
phishing warnings. We carried out a large-scale between-subjects user study (N
= 750) to compare the influence of warning dialogues supplemented with manually
generated explanations against those generated by two LLMs, Claude 3.5 Sonnet
and Llama 3.3 70B. We investigated two explanatory styles (feature-based and
counterfactual) for their effects on behavioural metrics (click-through rate)
and perceptual outcomes (e.g., trust, risk, clarity). The results indicate that
well-constructed LLM-generated explanations can equal or surpass manually
crafted explanations in reducing susceptibility to phishing; Claude-generated
warnings exhibited particularly robust performance. Feature-based explanations
were more effective for genuine phishing attempts, whereas counterfactual
explanations diminished false-positive rates. Other variables such as workload,
gender, and prior familiarity with warning dialogues significantly moderated
warning effectiveness. These results indicate that LLMs can be used to
automatically build explanations for warning users against phishing, and that
such solutions are scalable, adaptive, and consistent with human-centred
values.

### 10. [Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models](http://arxiv.org/pdf/2507.07406v1)

Authors: Jikesh Thapa, Gurrehmat Chahal, Serban Voinea Gabreanu, Yazan Otoum

Phishing attacks are becoming increasingly sophisticated, underscoring the
need for detection systems that strike a balance between high accuracy and
computational efficiency. This paper presents a comparative evaluation of
traditional Machine Learning (ML), Deep Learning (DL), and quantized
small-parameter Large Language Models (LLMs) for phishing detection. Through
experiments on a curated dataset, we show that while LLMs currently
underperform compared to ML and DL methods in terms of raw accuracy, they
exhibit strong potential for identifying subtle, context-based phishing cues.
We also investigate the impact of zero-shot and few-shot prompting strategies,
revealing that LLM-rephrased emails can significantly degrade the performance
of both ML and LLM-based detectors. Our benchmarking highlights that models
like DeepSeek R1 Distill Qwen 14B (Q8_0) achieve competitive accuracy, above
80%, using only 17GB of VRAM, supporting their viability for cost-efficient
deployment. We further assess the models' adversarial robustness and
cost-performance tradeoffs, and demonstrate how lightweight LLMs can provide
concise, interpretable explanations to support real-time decision-making. These
findings position optimized LLMs as promising components in phishing defence
systems and offer a path forward for integrating explainable, efficient AI into
modern cybersecurity frameworks.

### Computer Vision and Pattern Recognition

### 1. [PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency](http://arxiv.org/pdf/2507.07374v1)

Authors: Haotian Wang, Aoran Xiao, Xiaoqin Zhang, Meng Yang, Shijian Lu

Generalizable depth completion enables the acquisition of dense metric depth
maps for unseen environments, offering robust perception capabilities for
various downstream tasks. However, training such models typically requires
large-scale datasets with metric depth labels, which are often labor-intensive
to collect. This paper presents PacGDC, a label-efficient technique that
enhances data diversity with minimal annotation effort for generalizable depth
completion. PacGDC builds on novel insights into inherent ambiguities and
consistencies in object shapes and positions during 2D-to-3D projection,
allowing the synthesis of numerous pseudo geometries for the same visual scene.
This process greatly broadens available geometries by manipulating scene scales
of the corresponding depth maps. To leverage this property, we propose a new
data synthesis pipeline that uses multiple depth foundation models as scale
manipulators. These models robustly provide pseudo depth labels with varied
scene scales, affecting both local objects and global layouts, while ensuring
projection consistency that supports generalization. To further diversify
geometries, we incorporate interpolation and relocation strategies, as well as
unlabeled images, extending the data coverage beyond the individual use of
foundation models. Extensive experiments show that PacGDC achieves remarkable
generalizability across multiple benchmarks, excelling in diverse scene
semantics/scales and depth sparsity/patterns under both zero-shot and few-shot
settings. Code: https://github.com/Wang-xjtu/PacGDC.

### 2. [Adaptive Particle-Based Shape Modeling for Anatomical Surface Correspondence](http://arxiv.org/pdf/2507.07379v1)

Authors: Hong Xu, Shireen Y. Elhabian

Particle-based shape modeling (PSM) is a family of approaches that
automatically quantifies shape variability across anatomical cohorts by
positioning particles (pseudo landmarks) on shape surfaces in a consistent
configuration. Recent advances incorporate implicit radial basis function
representations as self-supervised signals to better capture the complex
geometric properties of anatomical structures. However, these methods still
lack self-adaptivity -- that is, the ability to automatically adjust particle
configurations to local geometric features of each surface, which is essential
for accurately representing complex anatomical variability. This paper
introduces two mechanisms to increase surface adaptivity while maintaining
consistent particle configurations: (1) a novel neighborhood correspondence
loss to enable high adaptivity and (2) a geodesic correspondence algorithm that
regularizes optimization to enforce geodesic neighborhood consistency. We
evaluate the efficacy and scalability of our approach on challenging datasets,
providing a detailed analysis of the adaptivity-correspondence trade-off and
benchmarking against existing methods on surface representation accuracy and
correspondence metrics.

### 3. [Multi-Scale Attention and Gated Shifting for Fine-Grained Event Spotting in Videos](http://arxiv.org/pdf/2507.07381v1)

Authors: Hao Xu, Arbind Agrahari Baniya, Sam Wells, Mohamed Reda Bouadjenek, Richard Dazeley, Sunil Aryal

Precise Event Spotting (PES) in sports videos requires frame-level
recognition of fine-grained actions from single-camera footage. Existing PES
models typically incorporate lightweight temporal modules such as Gate Shift
Module (GSM) or Gate Shift Fuse (GSF) to enrich 2D CNN feature extractors with
temporal context. However, these modules are limited in both temporal receptive
field and spatial adaptability. We propose a Multi-Scale Attention Gate Shift
Module (MSAGSM) that enhances GSM with multi-scale temporal dilations and
multi-head spatial attention, enabling efficient modeling of both short- and
long-term dependencies while focusing on salient regions. MSAGSM is a
lightweight plug-and-play module that can be easily integrated with various 2D
backbones. To further advance the field, we introduce the Table Tennis
Australia (TTA) dataset-the first PES benchmark for table tennis-containing
over 4800 precisely annotated events. Extensive experiments across five PES
benchmarks demonstrate that MSAGSM consistently improves performance with
minimal overhead, setting new state-of-the-art results.

### 4. [Seg-Wild: Interactive Segmentation based on 3D Gaussian Splatting for Unconstrained Image Collections](http://arxiv.org/pdf/2507.07395v1)

Authors: Yongtang Bao, Chengjie Tang, Yuze Wang, Haojie Li

Reconstructing and segmenting scenes from unconstrained photo collections
obtained from the Internet is a novel but challenging task. Unconstrained photo
collections are easier to get than well-captured photo collections. These
unconstrained images suffer from inconsistent lighting and transient
occlusions, which makes segmentation challenging. Previous segmentation methods
cannot address transient occlusions or accurately restore the scene's lighting
conditions. Therefore, we propose Seg-Wild, an interactive segmentation method
based on 3D Gaussian Splatting for unconstrained image collections, suitable
for in-the-wild scenes. We integrate multi-dimensional feature embeddings for
each 3D Gaussian and calculate the feature similarity between the feature
embeddings and the segmentation target to achieve interactive segmentation in
the 3D scene. Additionally, we introduce the Spiky 3D Gaussian Cutter (SGC) to
smooth abnormal 3D Gaussians. We project the 3D Gaussians onto a 2D plane and
calculate the ratio of 3D Gaussians that need to be cut using the SAM mask. We
also designed a benchmark to evaluate segmentation quality in in-the-wild
scenes. Experimental results demonstrate that compared to previous methods,
Seg-Wild achieves better segmentation results and reconstruction quality. Our
code will be available at https://github.com/Sugar0725/Seg-Wild.

### 5. [EscherNet++: Simultaneous Amodal Completion and Scalable View Synthesis through Masked Fine-Tuning and Enhanced Feed-Forward 3D Reconstruction](http://arxiv.org/pdf/2507.07410v1)

Authors: Xinan Zhang, Muhammad Zubair Irshad, Anthony Yezzi, Yi-Chang Tsai, Zsolt Kira

We propose EscherNet++, a masked fine-tuned diffusion model that can
synthesize novel views of objects in a zero-shot manner with amodal completion
ability. Existing approaches utilize multiple stages and complex pipelines to
first hallucinate missing parts of the image and then perform novel view
synthesis, which fail to consider cross-view dependencies and require redundant
storage and computing for separate stages. Instead, we apply masked fine-tuning
including input-level and feature-level masking to enable an end-to-end model
with the improved ability to synthesize novel views and conduct amodal
completion. In addition, we empirically integrate our model with other
feed-forward image-to-mesh models without extra training and achieve
competitive results with reconstruction time decreased by 95%, thanks to its
ability to synthesize arbitrary query views. Our method's scalable nature
further enhances fast 3D reconstruction. Despite fine-tuning on a smaller
dataset and batch size, our method achieves state-of-the-art results, improving
PSNR by 3.9 and Volume IoU by 0.28 on occluded tasks in 10-input settings,
while also generalizing to real-world occluded reconstruction.

### 6. [EPIC: Efficient Prompt Interaction for Text-Image Classification](http://arxiv.org/pdf/2507.07415v1)

Authors: Xinyao Yu, Hao Sun, Zeyu Ling, Ziwei Niu, Zhenjia Bai, Rui Qin, Yen-Wei Chen, Lanfen Lin

In recent years, large-scale pre-trained multimodal models (LMMs) generally
emerge to integrate the vision and language modalities, achieving considerable
success in multimodal tasks, such as text-image classification. The growing
size of LMMs, however, results in a significant computational cost for
fine-tuning these models for downstream tasks. Hence, prompt-based interaction
strategy is studied to align modalities more efficiently. In this context, we
propose a novel efficient prompt-based multimodal interaction strategy, namely
Efficient Prompt Interaction for text-image Classification (EPIC).
Specifically, we utilize temporal prompts on intermediate layers, and integrate
different modalities with similarity-based prompt interaction, to leverage
sufficient information exchange between modalities. Utilizing this approach,
our method achieves reduced computational resource consumption and fewer
trainable parameters (about 1\% of the foundation model) compared to other
fine-tuning strategies. Furthermore, it demonstrates superior performance on
the UPMC-Food101 and SNLI-VE datasets, while achieving comparable performance
on the MM-IMDB dataset.

### 7. [Corvid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](http://arxiv.org/pdf/2507.07424v1)

Authors: Jingjing Jiang, Chao Ma, Xurui Song, Hanwang Zhang, Jun Luo

Recent advancements in multimodal large language models (MLLMs) have
demonstrated exceptional performance in multimodal perception and
understanding. However, leading open-source MLLMs exhibit significant
limitations in complex and structured reasoning, particularly in tasks
requiring deep reasoning for decision-making and problem-solving. In this work,
we present Corvid, an MLLM with enhanced chain-of-thought (CoT) reasoning
capabilities. Architecturally, Corvid incorporates a hybrid vision encoder for
informative visual representation and a meticulously designed connector
(GateMixer) to facilitate cross-modal alignment. To enhance Corvid's CoT
reasoning capabilities, we introduce MCoT-Instruct-287K, a high-quality
multimodal CoT instruction-following dataset, refined and standardized from
diverse public reasoning sources. Leveraging this dataset, we fine-tune Corvid
with a two-stage CoT-formatted training approach to progressively enhance its
step-by-step reasoning abilities. Furthermore, we propose an effective
inference-time scaling strategy that enables Corvid to mitigate over-reasoning
and under-reasoning through self-verification. Extensive experiments
demonstrate that Corvid outperforms existing o1-like MLLMs and state-of-the-art
MLLMs with similar parameter scales, with notable strengths in mathematical
reasoning and science problem-solving. Project page:
https://mm-vl.github.io/corvid.

### 8. [Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects](http://arxiv.org/pdf/2507.07435v1)

Authors: Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao

In industrial point cloud analysis, detecting subtle anomalies demands
high-resolution spatial data, yet prevailing benchmarks emphasize
low-resolution inputs. To address this disparity, we propose a scalable
pipeline for generating realistic and subtle 3D anomalies. Employing this
pipeline, we developed MiniShift, the inaugural high-resolution 3D anomaly
detection dataset, encompassing 2,577 point clouds, each with 500,000 points
and anomalies occupying less than 1\% of the total. We further introduce
Simple3D, an efficient framework integrating Multi-scale Neighborhood
Descriptors (MSND) and Local Feature Spatial Aggregation (LFSA) to capture
intricate geometric details with minimal computational overhead, achieving
real-time inference exceeding 20 fps. Extensive evaluations on MiniShift and
established benchmarks demonstrate that Simple3D surpasses state-of-the-art
methods in both accuracy and speed, highlighting the pivotal role of
high-resolution data and effective feature aggregation in advancing practical
3D anomaly detection.

### 9. [Dual Semantic-Aware Network for Noise Suppressed Ultrasound Video Segmentation](http://arxiv.org/pdf/2507.07443v1)

Authors: Ling Zhou, Runtian Yuan, Yi Liu, Yuejie Zhang, Rui Feng, Shang Gao

Ultrasound imaging is a prevalent diagnostic tool known for its simplicity
and non-invasiveness. However, its inherent characteristics often introduce
substantial noise, posing considerable challenges for automated lesion or organ
segmentation in ultrasound video sequences. To address these limitations, we
propose the Dual Semantic-Aware Network (DSANet), a novel framework designed to
enhance noise robustness in ultrasound video segmentation by fostering mutual
semantic awareness between local and global features. Specifically, we
introduce an Adjacent-Frame Semantic-Aware (AFSA) module, which constructs a
channel-wise similarity matrix to guide feature fusion across adjacent frames,
effectively mitigating the impact of random noise without relying on
pixel-level relationships. Additionally, we propose a Local-and-Global
Semantic-Aware (LGSA) module that reorganizes and fuses temporal unconditional
local features, which capture spatial details independently at each frame, with
conditional global features that incorporate temporal context from adjacent
frames. This integration facilitates multi-level semantic representation,
significantly improving the model's resilience to noise interference. Extensive
evaluations on four benchmark datasets demonstrate that DSANet substantially
outperforms state-of-the-art methods in segmentation accuracy. Moreover, since
our model avoids pixel-level feature dependencies, it achieves significantly
higher inference FPS than video-based methods, and even surpasses some
image-based models. Code can be found in
\href{https://github.com/ZhouL2001/DSANet}{DSANet}

### 10. [Degradation-Agnostic Statistical Facial Feature Transformation for Blind Face Restoration in Adverse Weather Conditions](http://arxiv.org/pdf/2507.07464v1)

Authors: Chang-Hwan Son

With the increasing deployment of intelligent CCTV systems in outdoor
environments, there is a growing demand for face recognition systems optimized
for challenging weather conditions. Adverse weather significantly degrades
image quality, which in turn reduces recognition accuracy. Although recent face
image restoration (FIR) models based on generative adversarial networks (GANs)
and diffusion models have shown progress, their performance remains limited due
to the lack of dedicated modules that explicitly address weather-induced
degradations. This leads to distorted facial textures and structures. To
address these limitations, we propose a novel GAN-based blind FIR framework
that integrates two key components: local Statistical Facial Feature
Transformation (SFFT) and Degradation-Agnostic Feature Embedding (DAFE). The
local SFFT module enhances facial structure and color fidelity by aligning the
local statistical distributions of low-quality (LQ) facial regions with those
of high-quality (HQ) counterparts. Complementarily, the DAFE module enables
robust statistical facial feature extraction under adverse weather conditions
by aligning LQ and HQ encoder representations, thereby making the restoration
process adaptive to severe weather-induced degradations. Experimental results
demonstrate that the proposed degradation-agnostic SFFT model outperforms
existing state-of-the-art FIR methods based on GAN and diffusion models,
particularly in suppressing texture distortions and accurately reconstructing
facial structures. Furthermore, both the SFFT and DAFE modules are empirically
validated in enhancing structural fidelity and perceptual quality in face
restoration under challenging weather scenarios.

### Computers and Society

### 1. [Short-Term Gains, Long-Term Gaps: The Impact of GenAI and Search Technologies on Retention](http://arxiv.org/pdf/2507.07357v1)

Authors: Mahir Akgun, Sacip Toker

The rise of Generative AI (GenAI) tools, such as ChatGPT, has transformed how
students access and engage with information, raising questions about their
impact on learning outcomes and retention. This study investigates how GenAI
(ChatGPT), search engines (Google), and e-textbooks influence student
performance across tasks of varying cognitive complexity, based on Bloom's
Taxonomy. Using a sample of 123 students, we examined performance in three
tasks: [1] knowing and understanding, [2] applying, and [3] synthesizing,
evaluating, and creating. Results indicate that ChatGPT and Google groups
outperformed the control group in immediate assessments for lower-order
cognitive tasks, benefiting from quick access to structured information.
However, their advantage diminished over time, with retention test scores
aligning with those of the e-textbook group. For higher-order cognitive tasks,
no significant differences were observed among groups, with the control group
demonstrating the highest retention. These findings suggest that while
AI-driven tools facilitate immediate performance, they do not inherently
reinforce long-term retention unless supported by structured learning
strategies. The study highlights the need for balanced technology integration
in education, ensuring that AI tools are paired with pedagogical approaches
that promote deep cognitive engagement and knowledge retention.

### 2. [The Evolution of Scientific Credit: When Authorship Norms Impede Collaboration](http://arxiv.org/pdf/2507.07364v1)

Authors: Toby Handfield, Kevin Zollman

Scientific authorship norms vary dramatically across disciplines, from
contribution-sensitive systems where first author is the greatest contributor
and subsequent author order reflects relative input, to
contribution-insensitive conventions like alphabetical ordering or
senior-author-last. We develop evolutionary game-theoretic models to examine
both how these divergent norms emerge and their subsequent effects on
collaborative behavior. Our first model reveals that contribution-insensitive
norms evolve when researchers who sacrifice positional advantage face the
strongest adaptive pressure -- for example senior authors managing larger
collaboration portfolios or bearing heavier reputational stakes. This "Red
King" dynamic potentially explains why fields in which senior researchers
command large labs, major grants, and extensive collaboration portfolios may
paradoxically evolve conventions that favour junior-author positioning. Our
second model demonstrates that established norms influence researchers'
willingness to collaborate, with contribution-sensitive norms consistently
outperforming insensitive alternatives in fostering successful partnerships.
Contribution-insensitive norms create systematic coordination failures through
two mechanisms: "main contributor resentment" when exceptional work goes
unrecognized, and "second contributor resentment" when comparable efforts
receive unequal credit. These findings suggest that widely adopted practices
like senior-last positioning and alphabetical ordering may function as
institutional frictions that impede valuable scientific collaborations rather
than neutral organizational conventions, potentially reducing overall
scientific productivity across affected disciplines.

### 3. [Vaccine Hesitancy on YouTube: a Competition between Health and Politics](http://arxiv.org/pdf/2507.07517v1)

Authors: Yelena Mejova, Michele Tizzani

YouTube has rapidly emerged as a predominant platform for content
consumption, effectively displacing conventional media such as television and
news outlets. A part of the enormous video stream uploaded to this platform
includes health-related content, both from official public health
organizations, and from any individual or group that can make an account. The
quality of information available on YouTube is a critical point of public
health safety, especially when concerning major interventions, such as
vaccination. This study differentiates itself from previous efforts of auditing
YouTube videos on this topic by conducting a systematic daily collection of
posted videos mentioning vaccination for the duration of 3 months. We show that
the competition for the public's attention is between public health messaging
by institutions and individual educators on one side, and commentators on
society and politics on the other, the latest contributing the most to the
videos expressing stances against vaccination. Videos opposing vaccination are
more likely to mention politicians and publication media such as podcasts,
reports, and news analysis, on the other hand, videos in favor are more likely
to mention specific diseases or health-related topics. Finally, we find that,
at the time of analysis, only 2.7% of the videos have been taken down (by the
platform or the channel), despite 20.8% of the collected videos having a
vaccination hesitant stance, pointing to a lack of moderation activity for
hesitant content. The availability of high-quality information is essential to
improve awareness and compliance with public health interventions. Our findings
help characterize the public discourse around vaccination on one of the largest
media platforms, disentangling the role of the different creators and their
stances, and as such, they provide important insights for public health
communication policy.

### 4. [AI Human Impact: Toward a Model for Ethical Investing in AI-Intensive Companies](http://arxiv.org/pdf/2507.07703v1)

Authors: James Brusseau

Does AI conform to humans, or will we conform to AI? An ethical evaluation of
AI-intensive companies will allow investors to knowledgeably participate in the
decision. The evaluation is built from nine performance indicators that can be
analyzed and scored to reflect a technology's human-centering. The result is
objective investment guidance, as well as investors empowered to act in
accordance with their own values. Incorporating ethics into financial decisions
is a strategy that will be recognized by participants in environmental, social,
and governance investing, however, this paper argues that conventional ESG
frameworks are inadequate to companies that function with AI at their core.
Fully accounting for contemporary big data, predictive analytics, and machine
learning requires specialized metrics customized from established AI ethics
principles. With these metrics established, the larger goal is a model for
humanist investing in AI-intensive companies that is intellectually robust,
manageable for analysts, useful for portfolio managers, and credible for
investors.

### 5. [Structured Prompts, Better Outcomes? Exploring the Effects of a Structured Interface with ChatGPT in a Graduate Robotics Course](http://arxiv.org/pdf/2507.07767v1)

Authors: Jerome Brender, Laila El-Hamamsy, Kim Uittenhove, Francesco Mondada, Engin Bumbacher

Prior research shows that how students engage with Large Language Models
(LLMs) influences their problem-solving and understanding, reinforcing the need
to support productive LLM-uses that promote learning. This study evaluates the
impact of a structured GPT platform designed to promote 'good' prompting
behavior with data from 58 students in a graduate-level robotics course. The
students were assigned to either an intervention group using the structured
platform or a control group using ChatGPT freely for two practice lab sessions,
before a third session where all students could freely use ChatGPT. We analyzed
student perception (pre-post surveys), prompting behavior (logs), performance
(task scores), and learning (pre-post tests). Although we found no differences
in performance or learning between groups, we identified prompting behaviors -
such as having clear prompts focused on understanding code - that were linked
with higher learning gains and were more prominent when students used the
structured platform. However, such behaviors did not transfer once students
were no longer constrained to use the structured platform. Qualitative survey
data showed mixed perceptions: some students perceived the value of the
structured platform, but most did not perceive its relevance and resisted
changing their habits. These findings contribute to ongoing efforts to identify
effective strategies for integrating LLMs into learning and question the
effectiveness of bottom-up approaches that temporarily alter user interfaces to
influence students' interaction. Future research could instead explore top-down
strategies that address students' motivations and explicitly demonstrate how
certain interaction patterns support learning.

### 6. [FLoRA: An Advanced AI-Powered Engine to Facilitate Hybrid Human-AI Regulated Learning](http://arxiv.org/pdf/2507.07362v1)

Authors: Xinyu Li, Tongguang Li, Lixiang Yan, Yuheng Li, Linxuan Zhao, Mladen Raković, Inge Molenaar, Dragan Gašević, Yizhou Fan

SRL, defined as learners' ability to systematically plan, monitor, and
regulate their learning activities, is crucial for sustained academic
achievement and lifelong learning competencies. Emerging Artificial
Intelligence (AI) developments profoundly influence SRL interactions by
potentially either diminishing or strengthening learners' opportunities to
exercise their own regulatory skills. Recent literature emphasizes a balanced
approach termed Hybrid Human-AI Regulated Learning (HHAIRL), in which AI
provides targeted, timely scaffolding while preserving the learners' role as
active decision-makers and reflective monitors of their learning process.
Nevertheless, existing digital tools frequently fall short, lacking
adaptability, focusing narrowly on isolated SRL phases, and insufficiently
support meaningful human-AI interactions. In response, this paper introduces
the enhanced \flora Engine, which incorporates advanced Generative Artificial
Intelligence (GenAI) features and state-of-the-art learning analytics,
explicitly grounded in SRL and HHAIRL theories. The \flora Engine offers
instrumentation tools such as collaborative writing, multi-agents chatbot, and
detailed learning trace logging to support dynamic, adaptive scaffolding
tailored to individual needs in real time. We further present a summary of
several research studies that provide the validations for and illustrate how
these instrumentation tools can be utilized in real-world educational and
experimental contexts. These studies demonstrate the effectiveness of \flora
Engine in fostering SRL and HHAIRL, providing both theoretical insights and
practical solutions for the future of AI-enhanced learning context.

### 7. [Distributed and Decentralised Training: Technical Governance Challenges in a Shifting AI Landscape](http://arxiv.org/pdf/2507.07765v1)

Authors: Jakub Kryś, Yashvardhan Sharma, Janet Egan

Advances in low-communication training algorithms are enabling a shift from
centralised model training to compute setups that are either distributed across
multiple clusters or decentralised via community-driven contributions. This
paper distinguishes these two scenarios - distributed and decentralised
training - which are little understood and often conflated in policy discourse.
We discuss how they could impact technical AI governance through an increased
risk of compute structuring, capability proliferation, and the erosion of
detectability and shutdownability. While these trends foreshadow a possible new
paradigm that could challenge key assumptions of compute governance, we
emphasise that certain policy levers, like export controls, remain relevant. We
also acknowledge potential benefits of decentralised AI, including
privacy-preserving training runs that could unlock access to more data, and
mitigating harmful power concentration. Our goal is to support more precise
policymaking around compute, capability proliferation, and decentralised AI
development.

### 8. [Opting Out of Generative AI: a Behavioral Experiment on the Role of Education in Perplexity AI Avoidance](http://arxiv.org/pdf/2507.07881v1)

Authors: Roberto Ulloa, Juhi Kulshrestha, Celina Kacperski

The rise of conversational AI (CAI), powered by large language models, is
transforming how individuals access and interact with digital information.
However, these tools may inadvertently amplify existing digital inequalities.
This study investigates whether differences in formal education are associated
with CAI avoidance, leveraging behavioral data from an online experiment (N =
1,636). Participants were randomly assigned to a control or an
information-seeking task, either a traditional online search or a CAI
(Perplexity AI). Task avoidance (operationalized as survey abandonment or
providing unrelated responses during task assignment) was significantly higher
in the CAI group (51%) compared to the search (30.9%) and control (16.8%)
groups, with the highest CAI avoidance among participants with lower education
levels (~74.4%). Structural equation modeling based on the theoretical
framework UTAUT2 and LASSO regressions reveal that education is strongly
associated with CAI avoidance, even after accounting for various cognitive and
affective predictors of technology adoption. These findings underscore
education's central role in shaping AI adoption and the role of self-selection
biases in AI-related research, stressing the need for inclusive design to
ensure equitable access to emerging technologies.

### 9. [Meek Models Shall Inherit the Earth](http://arxiv.org/pdf/2507.07931v1)

Authors: Hans Gundlach, Jayson Lynch, Neil Thompson

The past decade has seen incredible scaling of AI systems by a few companies,
leading to inequality in AI model performance. This paper argues that, contrary
to prevailing intuition, the diminishing returns to compute scaling will lead
to a convergence of AI model capabilities. In other words, meek models (those
with limited computation budget) shall inherit the earth, approaching the
performance level of the best models overall. We develop a model illustrating
that under a fixed-distribution next-token objective, the marginal capability
returns to raw compute shrink substantially. Given current scaling practices,
we argue that these diminishing returns are strong enough that even companies
that can scale their models exponentially faster than other organizations will
eventually have little advantage in capabilities. As part of our argument, we
give several reasons that proxies like training loss differences capture
important capability measures using evidence from benchmark data and
theoretical performance models. In addition, we analyze empirical data on the
capability difference of AI models over time. Finally, in light of the
increasing ability of meek models, we argue that AI strategy and policy require
reexamination, and we outline the areas this shift will affect.

### 10. [Improving Clustering on Occupational Text Data through Dimensionality Reduction](http://arxiv.org/pdf/2507.07582v1)

Authors: Iago Xabier Vázquez García, Damla Partanaz, Emrullah Fatih Yetkin

In this study, we focused on proposing an optimal clustering mechanism for
the occupations defined in the well-known US-based occupational database,
O*NET. Even though all occupations are defined according to well-conducted
surveys in the US, their definitions can vary for different firms and
countries. Hence, if one wants to expand the data that is already collected in
O*NET for the occupations defined with different tasks, a map between the
definitions will be a vital requirement. We proposed a pipeline using several
BERT-based techniques with various clustering approaches to obtain such a map.
We also examined the effect of dimensionality reduction approaches on several
metrics used in measuring performance of clustering algorithms. Finally, we
improved our results by using a specialized silhouette approach. This new
clustering-based mapping approach with dimensionality reduction may help
distinguish the occupations automatically, creating new paths for people
wanting to change their careers.

### Databases

### 1. [Algorithmic Complexity Attacks on All Learned Cardinality Estimators: A Data-centric Approach](http://arxiv.org/pdf/2507.07438v1)

Authors: Yingze Li, Xianglong Liu, Dong Wang, Zixuan Wang, Hongzhi Wang, Kaixing Zhang, Yiming Guan

Learned cardinality estimators show promise in query cardinality prediction,
yet they universally exhibit fragility to training data drifts, posing risks
for real-world deployment. This work is the first to theoretical investigate
how minimal data-level drifts can maximally degrade the accuracy of learned
estimators. We propose data-centric algorithmic complexity attacks against
learned estimators in a black-box setting, proving that finding the optimal
attack strategy is NP-Hard. To address this, we design a polynomial-time
approximation algorithm with a $(1-\kappa)$ approximation ratio. Extensive
experiments demonstrate our attack's effectiveness: on STATS-CEB and IMDB-JOB
benchmarks, modifying just 0.8\% of training tuples increases the 90th
percentile Qerror by three orders of magnitude and raises end-to-end processing
time by up to 20$\times$. Our work not only reveals critical vulnerabilities in
deployed learned estimators but also provides the first unified worst-case
theoretical analysis of their fragility under data updates. Additionally, we
identify two countermeasures to mitigate such black-box attacks, offering
insights for developing robust learned database optimizers.

### 2. [JOB-Complex: A Challenging Benchmark for Traditional & Learned Query Optimization](http://arxiv.org/pdf/2507.07471v1)

Authors: Johannes Wehrstein, Timo Eckmann, Roman Heinrich, Carsten Binnig

Query optimization is a fundamental task in database systems that is crucial
to providing high performance. To evaluate learned and traditional optimizer's
performance, several benchmarks, such as the widely used JOB benchmark, are
used. However, in this paper, we argue that existing benchmarks are inherently
limited, as they do not reflect many real-world properties of query
optimization, thus overstating the performance of both traditional and learned
optimizers. In fact, simple but realistic properties, such as joins over string
columns or complex filter predicates, can drastically reduce the performance of
existing query optimizers. Thus, we introduce JOB-Complex, a new benchmark
designed to challenge traditional and learned query optimizers by reflecting
real-world complexity. Overall, JOB-Complex contains 30 SQL queries and comes
together with a plan-selection benchmark containing nearly 6000 execution
plans, making it a valuable resource to evaluate the performance of query
optimizers and cost models in real-world scenarios. In our evaluation, we show
that traditional and learned cost models struggle to achieve high performance
on JOB-Complex, providing a runtime of up to 11x slower compared to the optimal
plans.

### 3. [A Service Architecture for Dataspaces](http://arxiv.org/pdf/2507.07979v1)

Authors: Benedikt T. Arnold, Christoph Lange, Christina Gillmann, Stefan Decker

Dataspaces are designed to support sovereign, trusted and decentralized data
exchange between participants forming an ecosystem. They are standardized by
initiatives such as the International Data Spaces Association or Gaia-X and
have gained adoption in several domains such as mobility, manufacturing,
tourism or culture. In dataspaces, participants use connectors to communicate
peer-to-peer. The Eclipse Dataspace Components (EDC) Connector is a broadly
adopted, open-source implementation that adheres to the standards and is
supported by a large community. As dataspaces in general, it focuses on the
exchange of data assets with associated usage policies and does not support
services. In practice, however, there is demand for dataspace-based services
and conceptual arguments support their inclusion in dataspaces. In this paper,
we propose an abstraction layer for providing generic services within
dataspaces. Adopters can use this layer to easily develop own services,
seamlessly integrated with the existing dataspace technology. Besides, we
present an initial implementation of this service architecture for the EDC
Connector and demonstrate its practical applicability.

### Distributed, Parallel, and Cluster Computing

### 1. [Multi-agent Reinforcement Learning-based In-place Scaling Engine for Edge-cloud Systems](http://arxiv.org/pdf/2507.07671v1)

Authors: Jovan Prodanov, Blaž Bertalanič, Carolina Fortuna, Shih-Kai Chou, Matjaž Branko Jurič, Ramon Sanchez-Iborra, Jernej Hribar

Modern edge-cloud systems face challenges in efficiently scaling resources to
handle dynamic and unpredictable workloads. Traditional scaling approaches
typically rely on static thresholds and predefined rules, which are often
inadequate for optimizing resource utilization and maintaining performance in
distributed and dynamic environments. This inefficiency hinders the
adaptability and performance required in edge-cloud infrastructures, which can
only be achieved through the newly proposed in-place scaling. To address this
problem, we propose the Multi-Agent Reinforcement Learning-based In-place
Scaling Engine (MARLISE) that enables seamless, dynamic, reactive control with
in-place resource scaling. We develop our solution using two Deep Reinforcement
Learning algorithms: Deep Q-Network (DQN), and Proximal Policy Optimization
(PPO). We analyze each version of the proposed MARLISE solution using dynamic
workloads, demonstrating their ability to ensure low response times of
microservices and scalability. Our results show that MARLISE-based approaches
outperform heuristic method in managing resource elasticity while maintaining
microservice response times and achieving higher resource efficiency.

### 2. [KIS-S: A GPU-Aware Kubernetes Inference Simulator with RL-Based Auto-Scaling](http://arxiv.org/pdf/2507.07932v1)

Authors: Guilin Zhang, Wulan Guo, Ziqi Tan, Qiang Guan, Hailong Jiang

Autoscaling GPU inference workloads in Kubernetes remains challenging due to
the reactive and threshold-based nature of default mechanisms such as the
Horizontal Pod Autoscaler (HPA), which struggle under dynamic and bursty
traffic patterns and lack integration with GPU-level metrics. We present KIS-S,
a unified framework that combines KISim, a GPU-aware Kubernetes Inference
Simulator, with KIScaler, a Proximal Policy Optimization (PPO)-based
autoscaler. KIScaler learns latency-aware and resource-efficient scaling
policies entirely in simulation, and is directly deployed without retraining.
Experiments across four traffic patterns show that KIScaler improves average
reward by 75.2%, reduces P95 latency up to 6.7x over CPU baselines, and
generalizes without retraining. Our work bridges the gap between reactive
autoscaling and intelligent orchestration for scalable GPU-accelerated
environments.

### 3. [Machine Learning-driven Multiscale MD Workflows: The Mini-MuMMI Experience](http://arxiv.org/pdf/2507.07352v1)

Authors: Loïc Pottier, Konstantia Georgouli, Timothy S. Carpenter, Fikret Aydin, Jeremy O. B. Tempkin, Dwight V. Nissley, Frederick H. Streitz, Thomas R. W. Scogland, Peer-Timo Bremer, Felice C. Lightstone, Helgi I. Ingólfsson

Computational models have become one of the prevalent methods to model
complex phenomena. To accurately model complex interactions, such as detailed
biomolecular interactions, scientists often rely on multiscale models comprised
of several internal models operating at difference scales, ranging from
microscopic to macroscopic length and time scales. Bridging the gap between
different time and length scales has historically been challenging but the
advent of newer machine learning (ML) approaches has shown promise for tackling
that task. Multiscale models require massive amounts of computational power and
a powerful workflow management system. Orchestrating ML-driven multiscale
studies on parallel systems with thousands of nodes is challenging, the
workflow must schedule, allocate and control thousands of simulations operating
at different scales. Here, we discuss the massively parallel Multiscale
Machine-Learned Modeling Infrastructure (MuMMI), a multiscale workflow
management infrastructure, that can orchestrate thousands of molecular dynamics
(MD) simulations operating at different timescales, spanning from millisecond
to nanosecond. More specifically, we introduce a novel version of MuMMI called
"mini-MuMMI". Mini-MuMMI is a curated version of MuMMI designed to run on
modest HPC systems or even laptops whereas MuMMI requires larger HPC systems.
We demonstrate mini-MuMMI utility by exploring RAS-RAF membrane interactions
and discuss the different challenges behind the generalization of multiscale
workflows and how mini-MuMMI can be leveraged to target a broader range of
applications outside of MD and RAS-RAF interactions.

### 4. [KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows](http://arxiv.org/pdf/2507.07400v1)

Authors: Zaifeng Pan, Ajjkumar Patel, Zhengding Hu, Yipeng Shen, Yue Guan, Wan-Lu Li, Lianhui Qin, Yida Wang, Yufei Ding

Large language model (LLM) based agentic workflows have become a popular
paradigm for coordinating multiple specialized agents to solve complex tasks.
To improve serving efficiency, existing LLM systems employ prefix caching to
reuse key-value (KV) tensors corresponding to agents' fixed prompts, thereby
avoiding redundant computation across repeated invocations. However, current
systems typically evict KV caches using a Least Recently Used (LRU) policy,
which fails to anticipate future agent usage and often discards KV caches
shortly before their reuse. This leads to frequent cache misses and substantial
recomputation or swapping overhead. We present KVFlow, a workflow-aware KV
cache management framework tailored for agentic workloads. KVFlow abstracts the
agent execution schedule as an Agent Step Graph and assigns each agent a
steps-to-execution value that estimates its temporal proximity to future
activation. These values guide a fine-grained eviction policy at the KV node
level, allowing KVFlow to preserve entries likely to be reused and efficiently
manage shared prefixes in tree-structured caches. Moreover, KVFlow introduces a
fully overlapped KV prefetching mechanism, which proactively loads required
tensors from CPU to GPU in background threads for agents scheduled in the next
step, thereby avoiding cache miss stalls during generation. Compared to SGLang
with hierarchical radix cache, KVFlow achieves up to 1.83$\times$ speedup for
single workflows with large prompts, and up to 2.19$\times$ speedup for
scenarios with many concurrent workflows.

### 5. [Stress Monitoring in Healthcare: An Ensemble Machine Learning Framework Using Wearable Sensor Data](http://arxiv.org/pdf/2507.07589v1)

Authors: Arpana Sinhal, Anay Sinhal, Amit Sinhal

Healthcare professionals, particularly nurses, face elevated occupational
stress, a concern amplified during the COVID-19 pandemic. While wearable
sensors offer promising avenues for real-time stress monitoring, existing
studies often lack comprehensive datasets and robust analytical frameworks.
This study addresses these gaps by introducing a multimodal dataset comprising
physiological signals, electrodermal activity, heart rate and skin temperature.
A systematic literature review identified limitations in prior stress-detection
methodologies, particularly in handling class imbalance and optimizing model
generalizability. To overcome these challenges, the dataset underwent
preprocessing with the Synthetic Minority Over sampling Technique (SMOTE),
ensuring balanced representation of stress states. Advanced machine learning
models including Random Forest, XGBoost and a Multi-Layer Perceptron (MLP) were
evaluated and combined into a Stacking Classifier to leverage their collective
predictive strengths. By using a publicly accessible dataset and a reproducible
analytical pipeline, this work advances the development of deployable
stress-monitoring systems, offering practical implications for safeguarding
healthcare workers' mental health. Future research directions include expanding
demographic diversity and exploring edge-computing implementations for low
latency stress alerts.

### 6. [Accelerating Transposed Convolutions on FPGA-based Edge Devices](http://arxiv.org/pdf/2507.07683v1)

Authors: Jude Haris, José Cano

Transposed Convolutions (TCONV) enable the up-scaling mechanism within
generative Artificial Intelligence (AI) models. However, the predominant
Input-Oriented Mapping (IOM) method for implementing TCONV has complex output
mapping, overlapping sums, and ineffectual computations. These inefficiencies
further exacerbate the performance bottleneck of TCONV and generative models on
resource-constrained edge devices. To address this problem, in this paper we
propose MM2IM, a hardware-software co-designed accelerator that combines Matrix
Multiplication (MatMul) with col2IM to process TCONV layers on
resource-constrained edge devices efficiently. Using the SECDA-TFLite design
toolkit, we implement MM2IM and evaluate its performance across 261 TCONV
problem configurations, achieving an average speedup of 1.9x against a
dual-thread ARM Neon optimized CPU baseline. We then evaluate the performance
of MM2IM on a range of TCONV layers from well-known generative models achieving
up to 4.2x speedup, and compare it against similar resource-constrained TCONV
accelerators, outperforming them by at least 2x GOPs/DSP. Finally, we evaluate
MM2IM on the DCGAN and pix2pix GAN models, achieving up to 3x speedup and 2.4x
energy reduction against the CPU baseline.

### Digital Libraries

### 1. [ArchiveGPT: A human-centered evaluation of using a vision language model for image cataloguing](http://arxiv.org/pdf/2507.07551v1)

Authors: Line Abele, Gerrit Anders, Tolgahan Aydın, Jürgen Buder, Helen Fischer, Dominik Kimmel, Markus Huff

The accelerating growth of photographic collections has outpaced manual
cataloguing, motivating the use of vision language models (VLMs) to automate
metadata generation. This study examines whether Al-generated catalogue
descriptions can approximate human-written quality and how generative Al might
integrate into cataloguing workflows in archival and museum collections. A VLM
(InternVL2) generated catalogue descriptions for photographic prints on
labelled cardboard mounts with archaeological content, evaluated by archive and
archaeology experts and non-experts in a human-centered, experimental
framework. Participants classified descriptions as AI-generated or
expert-written, rated quality, and reported willingness to use and trust in AI
tools. Classification performance was above chance level, with both groups
underestimating their ability to detect Al-generated descriptions. OCR errors
and hallucinations limited perceived quality, yet descriptions rated higher in
accuracy and usefulness were harder to classify, suggesting that human review
is necessary to ensure the accuracy and quality of catalogue descriptions
generated by the out-of-the-box model, particularly in specialized domains like
archaeological cataloguing. Experts showed lower willingness to adopt AI tools,
emphasizing concerns on preservation responsibility over technical performance.
These findings advocate for a collaborative approach where AI supports draft
generation but remains subordinate to human verification, ensuring alignment
with curatorial values (e.g., provenance, transparency). The successful
integration of this approach depends not only on technical advancements, such
as domain-specific fine-tuning, but even more on establishing trust among
professionals, which could both be fostered through a transparent and
explainable AI pipeline.

### Discrete Mathematics

### 1. [Combinatorial Algorithm for Tropical Linearly Factorized Programming](http://arxiv.org/pdf/2507.07596v1)

Authors: Yuki Nishida

The tropical semiring is a set of numbers $\mathbb{R}\cup\{-\infty\}$ with
addition $a\oplus b:=\max(a,b)$ and multiplication $a\otimes b:=a+b$. As well
as in conventional algebra, linear programming problem in the tropical semiring
has been developed. In this study, we introduce a new type of tropical
optimization problem, namely, tropical linearly factorized programming problem.
This problem involves minimizing the objective function given by the product of
tropical linear forms $c_{k,1}\otimes x_1\oplus \cdots\oplus c_{k,n}\otimes
x_n$ divided by a tropical monomial, subject to tropical linear inequality
constraints. The objective function is convex in the conventional sense but not
in the tropical sense, while the feasible set is convex in the tropical sense
but not in the conventional sense.
  Our algorithm for tropical linearly factorized programming is based on the
descent method and exploits tangent digraphs. First, we demonstrate that the
feasible descent direction at the current solution can be obtained by solving
the minimum $s$-$t$ cut problem on a specific subgraph of the tangent digraph.
Although exponentially many such digraphs may exist in general, a more
efficient algorithm is devised in cases where the problem is non-degenerate.
Focusing on the fact that tangent digraphs become spanning trees in
non-degenerate cases, we present a simplex-like algorithm that updates the tree
structure iteratively. We show that each iteration can be executed in
$O(r_A+r_C)$ time, where $r_A$ and $r_C$ are the numbers of ``non-zero''
coefficients in the linear constraints and objective function, respectively.
For integer instances, our algorithm finds a local optimum in
$O((m+n)(r_A+r_C)MD)$ time, where $n$ and $m$ are the number of decision
variables and constraints, respectively, $M$ is the maximum absolute value of
coefficients and $D$ is the degree of the objective function.

### 2. [The Richness of CSP Non-redundancy](http://arxiv.org/pdf/2507.07942v1)

Authors: Joshua Brakensiek, Venkatesan Guruswami, Bart M. P. Jansen, Victor Lagerkvist, Magnus Wahlström

In the field of constraint satisfaction problems (CSP), a clause is called
redundant if its satisfaction is implied by satisfying all other clauses. An
instance of CSP$(P)$ is called non-redundant if it does not contain any
redundant clause. The non-redundancy (NRD) of a predicate $P$ is the maximum
number of clauses in a non-redundant instance of CSP$(P)$, as a function of the
number of variables $n$. Recent progress has shown that non-redundancy is
crucially linked to many other important questions in computer science and
mathematics including sparsification, kernelization, query complexity,
universal algebra, and extremal combinatorics. Given that non-redundancy is a
nexus for many of these important problems, the central goal of this paper is
to more deeply understand non-redundancy.
  Our first main result shows that for every rational number $r \ge 1$, there
exists a finite CSP predicate $P$ such that the non-redundancy of $P$ is
$\Theta(n^r)$. Our second main result explores the concept of conditional
non-redundancy first coined by Brakensiek and Guruswami [STOC 2025]. We
completely classify the conditional non-redundancy of all binary predicates
(i.e., constraints on two variables) by connecting these non-redundancy
problems to the structure of high-girth graphs in extremal combinatorics.
  Inspired by these concrete results, we build off the work of Carbonnel [CP
2022] to develop an algebraic theory of conditional non-redundancy. As an
application of this algebraic theory, we revisit the notion of Mal'tsev
embeddings, which is the most general technique known to date for establishing
that a predicate has linear non-redundancy. For example, we provide the first
example of predicate with a Mal'tsev embedding that cannot be attributed to the
structure of an Abelian group, but rather to the structure of the quantum Pauli
group.

### Data Structures and Algorithms

### 1. [A Randomized Rounding Approach for DAG Edge Deletion](http://arxiv.org/pdf/2507.07943v1)

Authors: Sina Kalantarzadeh, Nathan Klein, Victor Reis

In the DAG Edge Deletion problem, we are given an edge-weighted directed
acyclic graph and a parameter $k$, and the goal is to delete the minimum weight
set of edges so that the resulting graph has no paths of length $k$. This
problem, which has applications to scheduling, was introduced in 2015 by
Kenkre, Pandit, Purohit, and Saket. They gave a $k$-approximation and showed
that it is UGC-Hard to approximate better than $\lfloor 0.5k \rfloor$ for any
constant $k \ge 4$ using a work of Svensson from 2012. The approximation ratio
was improved to $\frac{2}{3}(k+1)$ by Klein and Wexler in 2016.
  In this work, we introduce a randomized rounding framework based on
distributions over vertex labels in $[0,1]$. The most natural distribution is
to sample labels independently from the uniform distribution over $[0,1]$. We
show this leads to a $(2-\sqrt{2})(k+1) \approx 0.585(k+1)$-approximation. By
using a modified (but still independent) label distribution, we obtain a
$0.549(k+1)$-approximation for the problem, as well as show that no independent
distribution over labels can improve our analysis to below $0.542(k+1)$.
Finally, we show a $0.5(k+1)$-approximation for bipartite graphs and for
instances with structured LP solutions. Whether this ratio can be obtained in
general is open.

### 2. [Finding sparse induced subgraphs on graphs of bounded induced matching treewidth](http://arxiv.org/pdf/2507.07975v1)

Authors: Hans L. Bodlaender, Fedor V. Fomin, Tuukka Korhonen

The induced matching width of a tree decomposition of a graph $G$ is the
cardinality of a largest induced matching $M$ of $G$, such that there exists a
bag that intersects every edge in $M$. The induced matching treewidth of a
graph $G$, denoted by $\mathsf{tree-}\mu(G)$, is the minimum induced matching
width of a tree decomposition of $G$. The parameter $\mathsf{tree-}\mu$ was
introduced by Yolov [SODA '18], who showed that, for example, Maximum-Weight
Independent Set can be solved in polynomial-time on graphs of bounded
$\mathsf{tree-}\mu$. Lima, Milani\v{c}, Mur\v{s}i\v{c}, Okrasa,
Rz\k{a}\.zewski, and \v{S}torgel [ESA '24] conjectured that this algorithm can
be generalized to a meta-problem called Maximum-Weight Induced Subgraph of
Bounded Treewidth, where we are given a vertex-weighted graph $G$, an integer
$w$, and a $\mathsf{CMSO}_2$-sentence $\Phi$, and are asked to find a
maximum-weight set $X \subseteq V(G)$ so that $G[X]$ has treewidth at most $w$
and satisfies $\Phi$. They proved the conjecture for some special cases, such
as for the problem Maximum-Weight Induced Forest.
  In this paper, we prove the general case of the conjecture. In particular, we
show that Maximum-Weight Induced Subgraph of Bounded Treewidth is
polynomial-time solvable when $\mathsf{tree-}\mu(G)$, $w$, and $|\Phi|$ are
bounded. The running time of our algorithm for $n$-vertex graphs $G$ with
$\mathsf{tree} - \mu(G) \le k$ is $f(k, w, |\Phi|) \cdot n^{O(k w^2)}$ for a
computable function $f$.

### 3. [Finding One Local Optimum Is Easy -- But What about Two?](http://arxiv.org/pdf/2507.07524v1)

Authors: Yasuaki Kobayashi, Kazuhiro Kurita, Yutaro Yamaguchi

The class PLS (Polynomial Local Search) captures the complexity of finding a
solution that is locally optimal and has proven to be an important concept in
the theory of local search. It has been shown that local search versions of
various combinatorial optimization problems, such as Maximum Independent Set
and Max Cut, are complete for this class. Such computational intractability
typically arises in local search problems allowing arbitrary weights; in
contrast, for unweighted problems, locally optimal solutions can be found in
polynomial time under standard settings. In this paper, we pursue the
complexity of local search problems from a different angle: We show that
computing two locally optimal solutions is NP-hard for various natural
unweighted local search problems, including Maximum Independent Set, Minimum
Dominating Set, Max SAT, and Max Cut. We also discuss several tractable cases
for finding two (or more) local optimal solutions.

### 4. [On the Complexity of Hyperpath and Minimal Separator Enumeration in Directed Hypergraphs](http://arxiv.org/pdf/2507.07528v1)

Authors: Kazuhiro Kurita, Kevin Mann

In this paper, we address the enumeration of (induced) $s$-$t$ paths and
minimal $s$-$t$ separators. These problems are some of the most famous
classical enumeration problems that can be solved in polynomial delay by simple
backtracking for a (un)directed graph. As a generalization of these problems,
we consider the (induced) $s$-$t$ hyperpath and minimal $s$-$t$ separator
enumeration in a \emph{directed hypergraph}. We show that extending these
classical enumeration problems to directed hypergraphs drastically changes
their complexity. More precisely, there are no output-polynomial time
algorithms for the enumeration of induced $s$-$t$ hyperpaths and minimal
$s$-$t$ separators unless $P = NP$, and if there is an output-polynomial time
algorithm for the $s$-$t$ hyperpath enumeration, then the minimal transversal
enumeration can be solved in output polynomial time even if a directed
hypergraph is $BF$-hypergraph. Since the existence of an output-polynomial time
algorithm for the minimal transversal enumeration has remained an open problem
for over 45 years, it indicates that the $s$-$t$ hyperpath enumeration for a
$BF$-hypergraph is not an easy problem. As a positive result, the $s$-$t$
hyperpath enumeration for a $B$-hypergraph can be solved in polynomial delay by
backtracking.

### 5. [Efficient and Adaptive Estimation of Local Triadic Coefficients](http://arxiv.org/pdf/2507.07536v1)

Authors: Ilie Sarpe, Aristides Gionis

Characterizing graph properties is fundamental to the analysis and to our
understanding of real-world networked systems. The local clustering
coefficient, and the more recently introduced, local closure coefficient,
capture powerful properties that are essential in a large number of
applications, ranging from graph embeddings to graph partitioning. Such
coefficients capture the local density of the neighborhood of each node,
considering incident triadic structures and paths of length two. For this
reason, we refer to these coefficients collectively as local triadic
coefficients.
  In this work, we consider the novel problem of computing efficiently the
average of local triadic coefficients, over a given partition of the nodes of
the input graph into a set of disjoint buckets. The average local triadic
coefficients of the nodes in each bucket provide a better insight into the
interplay of graph structure and the properties of the nodes associated to each
bucket. Unfortunately, exact computation, which requires listing all triangles
in a graph, is infeasible for large networks. Hence, we focus on obtaining
highly-accurate probabilistic estimates.
  We develop Triad, an adaptive algorithm based on sampling, which can be used
to estimate the average local triadic coefficients for a partition of the nodes
into buckets. Triad is based on a new class of unbiased estimators, and
non-trivial bounds on its sample complexity, enabling the efficient computation
of highly accurate estimates. Finally, we show how Triad can be efficiently
used in practice on large networks, and we present a case study showing that
average local triadic coefficients can capture high-order patterns over
collaboration networks.

### Emerging Technologies

### 1. [Autonomous AI-based Cybersecurity Framework for Critical Infrastructure: Real-Time Threat Mitigation](http://arxiv.org/pdf/2507.07416v1)

Authors: Jenifer Paulraj, Brindha Raghuraman, Nagarani Gopalakrishnan, Yazan Otoum

Critical infrastructure systems, including energy grids, healthcare
facilities, transportation networks, and water distribution systems, are
pivotal to societal stability and economic resilience. However, the increasing
interconnectivity of these systems exposes them to various cyber threats,
including ransomware, Denial-of-Service (DoS) attacks, and Advanced Persistent
Threats (APTs). This paper examines cybersecurity vulnerabilities in critical
infrastructure, highlighting the threat landscape, attack vectors, and the role
of Artificial Intelligence (AI) in mitigating these risks. We propose a hybrid
AI-driven cybersecurity framework to enhance real-time vulnerability detection,
threat modelling, and automated remediation. This study also addresses the
complexities of adversarial AI, regulatory compliance, and integration. Our
findings provide actionable insights to strengthen the security and resilience
of critical infrastructure systems against emerging cyber threats.

### 2. [Quantum Executor: A Unified Interface for Quantum Computing](http://arxiv.org/pdf/2507.07597v1)

Authors: Giuseppe Bisicchia, Alessandro Bocci, Antonio Brogi

As quantum computing evolves from theoretical promise to practical
deployment, the demand for robust, portable, and scalable tools for quantum
software experimentation is growing. This paper introduces Quantum Executor, a
backend-agnostic execution engine designed to orchestrate quantum experiments
across heterogeneous platforms. Quantum Executor provides a declarative and
modular interface that decouples experiment design from backend execution,
enabling seamless interoperability and code reuse across diverse quantum and
classical resources. Key features include support for asynchronous and
distributed execution, customizable execution strategies and a unified API for
managing quantum experiments. We illustrate its applicability through two
life-like usage scenarios such as automated benchmarking and hybrid validation,
discussing its capacity to streamline quantum development. We conclude by
discussing current limitations and outlining a roadmap for future enhancements.

### Graphics

### 1. [Self-supervised Learning of Latent Space Dynamics](http://arxiv.org/pdf/2507.07440v1)

Authors: Yue Li, Gene Wei-Chin Lin, Egor Larionov, Aljaz Bozic, Doug Roble, Ladislav Kavan, Stelian Coros, Bernhard Thomaszewski, Tuur Stuyck, Hsiao-yu Chen

Modeling the dynamic behavior of deformable objects is crucial for creating
realistic digital worlds. While conventional simulations produce high-quality
motions, their computational costs are often prohibitive. Subspace simulation
techniques address this challenge by restricting deformations to a
lower-dimensional space, improving performance while maintaining visually
compelling results. However, even subspace methods struggle to meet the
stringent performance demands of portable devices such as virtual reality
headsets and mobile platforms. To overcome this limitation, we introduce a
novel subspace simulation framework powered by a neural latent-space
integrator. Our approach leverages self-supervised learning to enhance
inference stability and generalization. By operating entirely within latent
space, our method eliminates the need for full-space computations, resulting in
a highly efficient method well-suited for deployment on portable devices. We
demonstrate the effectiveness of our approach on challenging examples involving
rods, shells, and solids, showcasing its versatility and potential for
widespread adoption.

### 2. [Hi-d maps: An interactive visualization technique for multi-dimensional categorical data](http://arxiv.org/pdf/2507.07890v1)

Authors: Radi Muhammad Reza, Benjamin A Watson

In this paper, we present Hi-D maps, a novel method for the visualization of
multi-dimensional categorical data. Our work addresses the scarcity of
techniques for visualizing a large number of data-dimensions in an effective
and space-efficient manner. We have mapped the full data-space onto a 2D
regular polygonal region. The polygon is cut hierarchically with lines parallel
to a user-controlled, ordered sequence of sides, each representing a dimension.
We have used multiple visual cues such as orientation, thickness, color,
countable glyphs, and text to depict cross-dimensional information. We have
added interactivity and hierarchical browsing to facilitate flexible
exploration of the display: small areas can be scrutinized for details. Thus,
our method is also easily extendable to visualize hierarchical information. Our
glyph animations add an engaging aesthetic during interaction. Like many
visualizations, Hi-D maps become less effective when a large number of
dimensions stresses perceptual limits, but Hi-D maps may add clarity before
those limits are reached.

### 3. [Digital Salon: An AI and Physics-Driven Tool for 3D Hair Grooming and Simulation](http://arxiv.org/pdf/2507.07387v1)

Authors: Chengan He, Jorge Alejandro Amador Herrera, Zhixin Shu, Xin Sun, Yao Feng, Sören Pirk, Dominik L. Michels, Meng Zhang, Tuanfeng Y. Wang, Julie Dorsey, Holly Rushmeier, Yi Zhou

We introduce Digital Salon, a comprehensive hair authoring system that
supports real-time 3D hair generation, simulation, and rendering. Unlike
existing methods that focus on isolated parts of 3D hair modeling and involve a
heavy computation process or network training, Digital Salon offers a holistic
and interactive system that lowers the technical barriers of 3D hair modeling
through natural language-based interaction. The system guides users through
four key stages: text-guided hair retrieval, real-time hair simulation,
interactive hair refinement, and hair-conditioned image generation. This
cohesive workflow makes advanced hair design accessible to users of varying
skill levels and dramatically streamlines the creative process in digital media
with an intuitive, versatile, and efficient solution for hair modeling. User
studies show that our system can outperform traditional hair modeling workflows
for rapid prototyping. Furthermore, we provide insights into the benefits of
our system with future potential of deploying our system in real salon
environments. More details can be found on our project page:
https://digital-salon.github.io/.

### 4. [SD-GS: Structured Deformable 3D Gaussians for Efficient Dynamic Scene Reconstruction](http://arxiv.org/pdf/2507.07465v1)

Authors: Wei Yao, Shuzhao Xie, Letian Li, Weixiang Zhang, Zhixin Lai, Shiqi Dai, Ke Zhang, Zhi Wang

Current 4D Gaussian frameworks for dynamic scene reconstruction deliver
impressive visual fidelity and rendering speed, however, the inherent trade-off
between storage costs and the ability to characterize complex physical motions
significantly limits the practical application of these methods. To tackle
these problems, we propose SD-GS, a compact and efficient dynamic Gaussian
splatting framework for complex dynamic scene reconstruction, featuring two key
contributions. First, we introduce a deformable anchor grid, a hierarchical and
memory-efficient scene representation where each anchor point derives multiple
3D Gaussians in its local spatiotemporal region and serves as the geometric
backbone of the 3D scene. Second, to enhance modeling capability for complex
motions, we present a deformation-aware densification strategy that adaptively
grows anchors in under-reconstructed high-dynamic regions while reducing
redundancy in static areas, achieving superior visual quality with fewer
anchors. Experimental results demonstrate that, compared to state-of-the-art
methods, SD-GS achieves an average of 60\% reduction in model size and an
average of 100\% improvement in FPS, significantly enhancing computational
efficiency while maintaining or even surpassing visual quality.

### 5. [Capture Stage Environments: A Guide to Better Matting](http://arxiv.org/pdf/2507.07623v1)

Authors: Hannah Dröge, Janelle Pfeifer, Saskia Rabich, Markus Plack, Reinhard Klein, Matthias B. Hullin

Capture stages are high-end sources of state-of-the-art recordings for
downstream applications in movies, games, and other media. One crucial step in
almost all pipelines is the matting of images to isolate the captured
performances from the background. While common matting algorithms deliver
remarkable performance in other applications like teleconferencing and mobile
entertainment, we found that they struggle significantly with the peculiarities
of capture stage content. The goal of our work is to share insights into those
challenges as a curated list of those characteristics along with a constructive
discussion for proactive intervention and present a guideline to practitioners
for an improved workflow to mitigate unresolved challenges. To this end, we
also demonstrate an efficient pipeline to adapt state-of-the-art approaches to
such custom setups without the need of extensive annotations, both offline and
real-time. For an objective evaluation, we propose a validation methodology
based on a leading diffusion model that highlights the benefits of our
approach.

### 6. [RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection](http://arxiv.org/pdf/2507.07733v1)

Authors: Yongyang Zhou, Fang-Lue Zhang, Zichen Wang, Lei Zhang

3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities in
novel view synthesis. However, rendering reflective objects remains a
significant challenge, particularly in inverse rendering and relighting. We
introduce RTR-GS, a novel inverse rendering framework capable of robustly
rendering objects with arbitrary reflectance properties, decomposing BRDF and
lighting, and delivering credible relighting results. Given a collection of
multi-view images, our method effectively recovers geometric structure through
a hybrid rendering model that combines forward rendering for radiance transfer
with deferred rendering for reflections. This approach successfully separates
high-frequency and low-frequency appearances, mitigating floating artifacts
caused by spherical harmonic overfitting when handling high-frequency details.
We further refine BRDF and lighting decomposition using an additional
physically-based deferred rendering branch. Experimental results show that our
method enhances novel view synthesis, normal estimation, decomposition, and
relighting while maintaining efficient training inference process.

### Computer Science and Game Theory

### 1. [Incentive Mechanism for Mobile Crowd Sensing with Assumed Bid Cost Reverse Auction](http://arxiv.org/pdf/2507.07688v1)

Authors: Jowa Yangchin, Ningrinla Marchang

Mobile Crowd Sensing (MCS) is the mechanism wherein people can contribute in
data collection process using their own mobile devices which have sensing
capabilities. Incentives are rewards that individuals get in exchange for data
they submit. Reverse Auction Bidding (RAB) is a framework that allows users to
place bids for selling the data they collected. Task providers can select users
to buy data from by looking at bids. Using the RAB framework, MCS system can be
optimized for better user utility, task provider utility and platform utility.
In this paper, we propose a novel approach called Reverse Auction with Assumed
Bid Cost (RA-ABC) which allows users to place a bid in the system before
collecting data. We opine that performing the tasks only after winning helps in
reducing resource consumption instead of performing the tasks before bidding.
User Return on Investment (ROI) is calculated with which they decide to further
participate or not by either increasing or decreasing their bids. We also
propose an extension of RA-ABC with dynamic recruitment (RA-ABCDR) in which we
allow new users to join the system at any time during bidding rounds.
Simulation results demonstrate that RA-ABC and RA-ABCDR outperform the widely
used Tullock Optimal Prize Function, with RA-ABCDR achieving up to 54.6\%
higher user retention and reducing auction cost by 22.2\%, thereby ensuring
more efficient and sustainable system performance. Extensive simulations
confirm that dynamic user recruitment significantly enhances performance across
stability, fairness, and cost-efficiency metrics.

### 2. [Hybrid Advertising in the Sponsored Search](http://arxiv.org/pdf/2507.07711v1)

Authors: Zhen Zhang, Weian Li, Yuhan Wang, Qi Qi, Kun Huang

Online advertisements are a primary revenue source for e-commerce platforms.
Traditional advertising models are store-centric, selecting winning stores
through auction mechanisms. Recently, a new approach known as joint advertising
has emerged, which presents sponsored bundles combining one store and one brand
in ad slots. Unlike traditional models, joint advertising allows platforms to
collect payments from both brands and stores. However, each of these two
advertising models appeals to distinct user groups, leading to low
click-through rates when users encounter an undesirable advertising model. To
address this limitation and enhance generality, we propose a novel advertising
model called ''Hybrid Advertising''. In this model, each ad slot can be
allocated to either an independent store or a bundle. To find the optimal
auction mechanisms in hybrid advertising, while ensuring nearly dominant
strategy incentive compatibility and individual rationality, we introduce the
Hybrid Regret Network (HRegNet), a neural network architecture designed for
this purpose. Extensive experiments on both synthetic and real-world data
demonstrate that the mechanisms generated by HRegNet significantly improve
platform revenue compared to established baseline methods.

### 3. [Improving the Price of Anarchy via Predictions in Parallel-Link Networks](http://arxiv.org/pdf/2507.07915v1)

Authors: George Christodoulou, Vasilis Christoforidis, Alkmini Sgouritsa, Ioannis Vlachos

We study non-atomic congestion games on parallel-link networks with affine
cost functions. We investigate the power of machine-learned predictions in the
design of coordination mechanisms aimed at minimizing the impact of
selfishness. Our main results demonstrate that enhancing coordination
mechanisms with a simple advice on the input rate can optimize the social cost
whenever the advice is accurate (consistency), while only incurring minimal
losses even when the predictions are arbitrarily inaccurate (bounded
robustness). Moreover, we provide a full characterization of the consistent
mechanisms that holds for all monotone cost functions, and show that our
suggested mechanism is optimal with respect to the robustness. We further
explore the notion of smoothness within this context: we extend our mechanism
to achieve error-tolerance, i.e. we provide an approximation guarantee that
degrades smoothly as a function of the prediction error, up to a predetermined
threshold, while achieving a bounded robustness.

### 4. [Optimal Auction Design in the Joint Advertising](http://arxiv.org/pdf/2507.07418v1)

Authors: Yang Li, Yuchao Ma, Qi Qi

Online advertising is a vital revenue source for major internet platforms.
Recently, joint advertising, which assigns a bundle of two advertisers in an ad
slot instead of allocating a single advertiser, has emerged as an effective
method for enhancing allocation efficiency and revenue. However, existing
mechanisms for joint advertising fail to realize the optimality, as they tend
to focus on individual advertisers and overlook bundle structures. This paper
identifies an optimal mechanism for joint advertising in a single-slot setting.
For multi-slot joint advertising, we propose \textbf{BundleNet}, a novel
bundle-based neural network approach specifically designed for joint
advertising. Our extensive experiments demonstrate that the mechanisms
generated by \textbf{BundleNet} approximate the theoretical analysis results in
the single-slot setting and achieve state-of-the-art performance in the
multi-slot setting. This significantly increases platform revenue while
ensuring approximate dominant strategy incentive compatibility and individual
rationality.

### Human-Computer Interaction

### 1. [FLoRA: An Advanced AI-Powered Engine to Facilitate Hybrid Human-AI Regulated Learning](http://arxiv.org/pdf/2507.07362v1)

Authors: Xinyu Li, Tongguang Li, Lixiang Yan, Yuheng Li, Linxuan Zhao, Mladen Raković, Inge Molenaar, Dragan Gašević, Yizhou Fan

SRL, defined as learners' ability to systematically plan, monitor, and
regulate their learning activities, is crucial for sustained academic
achievement and lifelong learning competencies. Emerging Artificial
Intelligence (AI) developments profoundly influence SRL interactions by
potentially either diminishing or strengthening learners' opportunities to
exercise their own regulatory skills. Recent literature emphasizes a balanced
approach termed Hybrid Human-AI Regulated Learning (HHAIRL), in which AI
provides targeted, timely scaffolding while preserving the learners' role as
active decision-makers and reflective monitors of their learning process.
Nevertheless, existing digital tools frequently fall short, lacking
adaptability, focusing narrowly on isolated SRL phases, and insufficiently
support meaningful human-AI interactions. In response, this paper introduces
the enhanced \flora Engine, which incorporates advanced Generative Artificial
Intelligence (GenAI) features and state-of-the-art learning analytics,
explicitly grounded in SRL and HHAIRL theories. The \flora Engine offers
instrumentation tools such as collaborative writing, multi-agents chatbot, and
detailed learning trace logging to support dynamic, adaptive scaffolding
tailored to individual needs in real time. We further present a summary of
several research studies that provide the validations for and illustrate how
these instrumentation tools can be utilized in real-world educational and
experimental contexts. These studies demonstrate the effectiveness of \flora
Engine in fostering SRL and HHAIRL, providing both theoretical insights and
practical solutions for the future of AI-enhanced learning context.

### 2. [Digital Salon: An AI and Physics-Driven Tool for 3D Hair Grooming and Simulation](http://arxiv.org/pdf/2507.07387v1)

Authors: Chengan He, Jorge Alejandro Amador Herrera, Zhixin Shu, Xin Sun, Yao Feng, Sören Pirk, Dominik L. Michels, Meng Zhang, Tuanfeng Y. Wang, Julie Dorsey, Holly Rushmeier, Yi Zhou

We introduce Digital Salon, a comprehensive hair authoring system that
supports real-time 3D hair generation, simulation, and rendering. Unlike
existing methods that focus on isolated parts of 3D hair modeling and involve a
heavy computation process or network training, Digital Salon offers a holistic
and interactive system that lowers the technical barriers of 3D hair modeling
through natural language-based interaction. The system guides users through
four key stages: text-guided hair retrieval, real-time hair simulation,
interactive hair refinement, and hair-conditioned image generation. This
cohesive workflow makes advanced hair design accessible to users of varying
skill levels and dramatically streamlines the creative process in digital media
with an intuitive, versatile, and efficient solution for hair modeling. User
studies show that our system can outperform traditional hair modeling workflows
for rapid prototyping. Furthermore, we provide insights into the benefits of
our system with future potential of deploying our system in real salon
environments. More details can be found on our project page:
https://digital-salon.github.io/.

### 3. [Pluri-perspectivism in Human-robot Co-creativity with Older Adults](http://arxiv.org/pdf/2507.07550v1)

Authors: Marianne Bossema, Rob Saunders, Aske Plaat, Somaya Ben Allouch

This position paper explores pluriperspectivism as a core element of human
creative experience and its relevance to humanrobot cocreativity We propose a
layered fivedimensional model to guide the design of cocreative behaviors and
the analysis of interaction dynamics This model is based on literature and
results from an interview study we conducted with 10 visual artists and 8 arts
educators examining how pluriperspectivism supports creative practice The
findings of this study provide insight in how robots could enhance human
creativity through adaptive contextsensitive behavior demonstrating the
potential of pluriperspectivism This paper outlines future directions for
integrating pluriperspectivism with visionlanguage models VLMs to support
context sensitivity in cocreative robots

### 4. [FiDTouch: A 3D Wearable Haptic Display for the Finger Pad](http://arxiv.org/pdf/2507.07661v1)

Authors: Daria Trinitatova, Dzmitry Tsetserukou

The applications of fingertip haptic devices have spread to various fields
from revolutionizing virtual reality and medical training simulations to
facilitating remote robotic operations, proposing great potential for enhancing
user experiences, improving training outcomes, and new forms of interaction. In
this work, we present FiDTouch, a 3D wearable haptic device that delivers
cutaneous stimuli to the finger pad, such as contact, pressure, encounter, skin
stretch, and vibrotactile feedback. The application of a tiny inverted Delta
robot in the mechanism design allows providing accurate contact and fast
changing dynamic stimuli to the finger pad surface. The performance of the
developed display was evaluated in a two-stage user study of the perception of
static spatial contact stimuli and skin stretch stimuli generated on the finger
pad. The proposed display, by providing users with precise touch and force
stimuli, can enhance user immersion and efficiency in the fields of
human-computer and human-robot interactions.

### 5. [Opting Out of Generative AI: a Behavioral Experiment on the Role of Education in Perplexity AI Avoidance](http://arxiv.org/pdf/2507.07881v1)

Authors: Roberto Ulloa, Juhi Kulshrestha, Celina Kacperski

The rise of conversational AI (CAI), powered by large language models, is
transforming how individuals access and interact with digital information.
However, these tools may inadvertently amplify existing digital inequalities.
This study investigates whether differences in formal education are associated
with CAI avoidance, leveraging behavioral data from an online experiment (N =
1,636). Participants were randomly assigned to a control or an
information-seeking task, either a traditional online search or a CAI
(Perplexity AI). Task avoidance (operationalized as survey abandonment or
providing unrelated responses during task assignment) was significantly higher
in the CAI group (51%) compared to the search (30.9%) and control (16.8%)
groups, with the highest CAI avoidance among participants with lower education
levels (~74.4%). Structural equation modeling based on the theoretical
framework UTAUT2 and LASSO regressions reveal that education is strongly
associated with CAI avoidance, even after accounting for various cognitive and
affective predictors of technology adoption. These findings underscore
education's central role in shaping AI adoption and the role of self-selection
biases in AI-related research, stressing the need for inclusive design to
ensure equitable access to emerging technologies.

### 6. [The Potential of Olfactory Stimuli in Stress Reduction through Virtual Reality](http://arxiv.org/pdf/2507.07911v1)

Authors: Yasmin Elsaddik Valdivieso, Mohd Faisal, Karim Alghoul, Monireh, Vahdati, Kamran Gholizadeh Hamlabadi, Fedwa Laamarti, Hussein Al Osman, Abdulmotaleb El Saddik

Immersive virtual reality (VR) is a promising tool for stress reduction and
relaxation, traditionally relying on visual and auditory stimuli. This study
examines the role of olfactory stimuli in enhancing these effects, using a
randomized within-subject design. Thirty participants aged 18-60 experienced VR
scenarios simulating a calming seaside environment, with sessions lasting 45
minutes, in two conditions: with and without a "Beach" essential oil scent
(Yankee Candle) administered via diffuser. Stress and relaxation were assessed
through self-reported surveys and physiological measures, specifically
ECG-based heart rate variability (HRV). Results showed no significant
difference in self-reported relaxation scores (p=0.371) between conditions, but
HRV analysis revealed a significant stress reduction (p=0.002) with olfactory
input, with HF increasing 108% from the Math Stress Test to the scented
relaxation condition, compared to 44% without scent. Additionally, 71.4% of
participants expressed willingness to use olfactory-enhanced VR for relaxation,
suggesting practical appeal. These findings indicate that olfactory stimuli may
enhance relaxation subconsciously, underscoring the importance of multisensory
integration in VR. Future work could explore personalized scents and long-term
effects to optimize VR- based interventions for emotional and physical
well-being.

### 7. [Can Large Language Models Improve Phishing Defense? A Large-Scale Controlled Experiment on Warning Dialogue Explanations](http://arxiv.org/pdf/2507.07916v1)

Authors: Federico Maria Cau, Giuseppe Desolda, Francesco Greco, Lucio Davide Spano, Luca Viganò

Phishing has become a prominent risk in modern cybersecurity, often used to
bypass technological defences by exploiting predictable human behaviour.
Warning dialogues are a standard mitigation measure, but the lack of
explanatory clarity and static content limits their effectiveness. In this
paper, we report on our research to assess the capacity of Large Language
Models (LLMs) to generate clear, concise, and scalable explanations for
phishing warnings. We carried out a large-scale between-subjects user study (N
= 750) to compare the influence of warning dialogues supplemented with manually
generated explanations against those generated by two LLMs, Claude 3.5 Sonnet
and Llama 3.3 70B. We investigated two explanatory styles (feature-based and
counterfactual) for their effects on behavioural metrics (click-through rate)
and perceptual outcomes (e.g., trust, risk, clarity). The results indicate that
well-constructed LLM-generated explanations can equal or surpass manually
crafted explanations in reducing susceptibility to phishing; Claude-generated
warnings exhibited particularly robust performance. Feature-based explanations
were more effective for genuine phishing attempts, whereas counterfactual
explanations diminished false-positive rates. Other variables such as workload,
gender, and prior familiarity with warning dialogues significantly moderated
warning effectiveness. These results indicate that LLMs can be used to
automatically build explanations for warning users against phishing, and that
such solutions are scalable, adaptive, and consistent with human-centred
values.

### 8. [Probing Experts' Perspectives on AI-Assisted Public Speaking Training](http://arxiv.org/pdf/2507.07930v1)

Authors: Nesrine Fourati, Alisa Barkar, Marion Dragée, Liv Danthon-Lefebvre, Mathieu Chollet

Background: Public speaking is a vital professional skill, yet it remains a
source of significant anxiety for many individuals. Traditional training relies
heavily on expert coaching, but recent advances in AI has led to novel types of
commercial automated public speaking feedback tools. However, most research has
focused on prototypes rather than commercial applications, and little is known
about how public speaking experts perceive these tools.
  Objectives: This study aims to evaluate expert opinions on the efficacy and
design of commercial AI-based public speaking training tools and to propose
guidelines for their improvement.
  Methods: The research involved 16 semi-structured interviews and 2 focus
groups with public speaking experts. Participants discussed their views on
current commercial tools, their potential integration into traditional
coaching, and suggestions for enhancing these systems.
  Results and Conclusions: Experts acknowledged the value of AI tools in
handling repetitive, technical aspects of training, allowing coaches to focus
on higher-level skills. However they found key issues in current tools,
emphasising the need for personalised, understandable, carefully selected
feedback and clear instructional design. Overall, they supported a hybrid model
combining traditional coaching with AI-supported exercises.

### 9. [ArchiveGPT: A human-centered evaluation of using a vision language model for image cataloguing](http://arxiv.org/pdf/2507.07551v1)

Authors: Line Abele, Gerrit Anders, Tolgahan Aydın, Jürgen Buder, Helen Fischer, Dominik Kimmel, Markus Huff

The accelerating growth of photographic collections has outpaced manual
cataloguing, motivating the use of vision language models (VLMs) to automate
metadata generation. This study examines whether Al-generated catalogue
descriptions can approximate human-written quality and how generative Al might
integrate into cataloguing workflows in archival and museum collections. A VLM
(InternVL2) generated catalogue descriptions for photographic prints on
labelled cardboard mounts with archaeological content, evaluated by archive and
archaeology experts and non-experts in a human-centered, experimental
framework. Participants classified descriptions as AI-generated or
expert-written, rated quality, and reported willingness to use and trust in AI
tools. Classification performance was above chance level, with both groups
underestimating their ability to detect Al-generated descriptions. OCR errors
and hallucinations limited perceived quality, yet descriptions rated higher in
accuracy and usefulness were harder to classify, suggesting that human review
is necessary to ensure the accuracy and quality of catalogue descriptions
generated by the out-of-the-box model, particularly in specialized domains like
archaeological cataloguing. Experts showed lower willingness to adopt AI tools,
emphasizing concerns on preservation responsibility over technical performance.
These findings advocate for a collaborative approach where AI supports draft
generation but remains subordinate to human verification, ensuring alignment
with curatorial values (e.g., provenance, transparency). The successful
integration of this approach depends not only on technical advancements, such
as domain-specific fine-tuning, but even more on establishing trust among
professionals, which could both be fostered through a transparent and
explainable AI pipeline.

### 10. [Conjugated Capabilities: Interrelations of Elementary Human Capabilities and Their Implication on Human-Machine Task Allocation and Capability Testing Procedures](http://arxiv.org/pdf/2507.07560v1)

Authors: Nils Mandischer, Larissa Füller, Torsten Alles, Frank Flemisch, Lars Mikelsons

Human and automation capabilities are the foundation of every human-autonomy
interaction and interaction pattern. Therefore, machines need to understand the
capacity and performance of human doing, and adapt their own behavior,
accordingly. In this work, we address the concept of conjugated capabilities,
i.e. capabilities that are dependent or interrelated and between which effort
can be distributed. These may be used to overcome human limitations, by
shifting effort from a deficient to a conjugated capability with performative
resources. For example: A limited arm's reach may be compensated by tilting the
torso forward. We analyze the interrelation between elementary capabilities
within the IMBA standard to uncover potential conjugation, and show evidence in
data of post-rehabilitation patients. From the conjugated capabilities, within
the example application of stationary manufacturing, we create a network of
interrelations. With this graph, a manifold of potential uses is enabled. We
showcase the graph's usage in optimizing IMBA test design to accelerate data
recordings, and discuss implications of conjugated capabilities on task
allocation between the human and an autonomy.

### Information Retrieval

### 1. [When Graph Contrastive Learning Backfires: Spectral Vulnerability and Defense in Recommendation](http://arxiv.org/pdf/2507.07436v1)

Authors: Zongwei Wang, Min Gao, Junliang Yu, Shazia Sadiq, Hongzhi Yin, Ling Liu

Graph Contrastive Learning (GCL) has demonstrated substantial promise in
enhancing the robustness and generalization of recommender systems,
particularly by enabling models to leverage large-scale unlabeled data for
improved representation learning. However, in this paper, we reveal an
unexpected vulnerability: the integration of GCL inadvertently increases the
susceptibility of a recommender to targeted promotion attacks. Through both
theoretical investigation and empirical validation, we identify the root cause
as the spectral smoothing effect induced by contrastive optimization, which
disperses item embeddings across the representation space and unintentionally
enhances the exposure of target items. Building on this insight, we introduce
CLeaR, a bi-level optimization attack method that deliberately amplifies
spectral smoothness, enabling a systematic investigation of the susceptibility
of GCL-based recommendation models to targeted promotion attacks. Our findings
highlight the urgent need for robust countermeasures; in response, we further
propose SIM, a spectral irregularity mitigation framework designed to
accurately detect and suppress targeted items without compromising model
performance. Extensive experiments on multiple benchmark datasets demonstrate
that, compared to existing targeted promotion attacks, GCL-based recommendation
models exhibit greater susceptibility when evaluated with CLeaR, while SIM
effectively mitigates these vulnerabilities.

### 2. [NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation](http://arxiv.org/pdf/2507.07522v1)

Authors: Jinfeng Xu, Zheyu Chen, Shuo Yang, Jinze Li, Hewei Wang, Wei Wang, Xiping Hu, Edith Ngai

Graph Neural Networks (GNNs) are widely used in collaborative filtering to
capture high-order user-item relationships. To address the data sparsity
problem in recommendation systems, Graph Contrastive Learning (GCL) has emerged
as a promising paradigm that maximizes mutual information between contrastive
views. However, existing GCL methods rely on augmentation techniques that
introduce semantically irrelevant noise and incur significant computational and
storage costs, limiting effectiveness and efficiency.
  To overcome these challenges, we propose NLGCL, a novel contrastive learning
framework that leverages naturally contrastive views between neighbor layers
within GNNs. By treating each node and its neighbors in the next layer as
positive pairs, and other nodes as negatives, NLGCL avoids augmentation-based
noise while preserving semantic relevance. This paradigm eliminates costly view
construction and storage, making it computationally efficient and practical for
real-world scenarios. Extensive experiments on four public datasets demonstrate
that NLGCL outperforms state-of-the-art baselines in effectiveness and
efficiency.

### 3. [Document Similarity Enhanced IPS Estimation for Unbiased Learning to Rank](http://arxiv.org/pdf/2507.07909v1)

Authors: Zeyan Liang, Graham McDonald, Iadh Ounis

Learning to Rank (LTR) models learn from historical user interactions, such
as user clicks. However, there is an inherent bias in the clicks of users due
to position bias, i.e., users are more likely to click highly-ranked documents
than low-ranked documents. To address this bias when training LTR models, many
approaches from the literature re-weight the users' click data using Inverse
Propensity Scoring (IPS). IPS re-weights the user's clicks proportionately to
the position in the historical ranking that a document was placed when it was
clicked since low-ranked documents are less likely to be seen by a user. In
this paper, we argue that low-ranked documents that are similar to
highly-ranked relevant documents are also likely to be relevant. Moreover,
accounting for the similarity of low-ranked documents to highly ranked relevant
documents when calculating IPS can more effectively mitigate the effects of
position bias. Therefore, we propose an extension to IPS, called IPSsim, that
takes into consideration the similarity of documents when estimating IPS. We
evaluate our IPSsim estimator using two large publicly available LTR datasets
under a number of simulated user click settings, and with different numbers of
training clicks. Our experiments show that our IPSsim estimator is more
effective than the existing IPS estimators for learning an unbiased LTR model,
particularly in top-n settings when n >= 30. For example, when n = 50, our
IPSsim estimator achieves a statistically significant ~3% improvement (p <
0.05) in terms of NDCG compared to the Doubly Robust estimator from the
literature.

### 4. [Measuring Hypothesis Testing Errors in the Evaluation of Retrieval Systems](http://arxiv.org/pdf/2507.07924v1)

Authors: Jack McKechnie, Graham McDonald, Craig Macdonald

The evaluation of Information Retrieval (IR) systems typically uses
query-document pairs with corresponding human-labelled relevance assessments
(qrels). These qrels are used to determine if one system is better than another
based on average retrieval performance. Acquiring large volumes of human
relevance assessments is expensive. Therefore, more efficient relevance
assessment approaches have been proposed, necessitating comparisons between
qrels to ascertain their efficacy. Discriminative power, i.e. the ability to
correctly identify significant differences between systems, is important for
drawing accurate conclusions on the robustness of qrels. Previous work has
measured the proportion of pairs of systems that are identified as
significantly different and has quantified Type I statistical errors. Type I
errors lead to incorrect conclusions due to false positive significance tests.
We argue that also identifying Type II errors (false negatives) is important as
they lead science in the wrong direction. We quantify Type II errors and
propose that balanced classification metrics, such as balanced accuracy, can be
used to portray the discriminative power of qrels. We perform experiments using
qrels generated using alternative relevance assessment methods to investigate
measuring hypothesis testing errors in IR evaluation. We find that additional
insights into the discriminative power of qrels can be gained by quantifying
Type II errors, and that balanced classification metrics can be used to give an
overall summary of discriminative power in one, easily comparable, number.

### 5. [Rethinking the Privacy of Text Embeddings: A Reproducibility Study of "Text Embeddings Reveal (Almost) As Much As Text"](http://arxiv.org/pdf/2507.07700v1)

Authors: Dominykas Seputis, Yongkang Li, Karsten Langerak, Serghei Mihailov

Text embeddings are fundamental to many natural language processing (NLP)
tasks, extensively applied in domains such as recommendation systems and
information retrieval (IR). Traditionally, transmitting embeddings instead of
raw text has been seen as privacy-preserving. However, recent methods such as
Vec2Text challenge this assumption by demonstrating that controlled decoding
can successfully reconstruct original texts from black-box embeddings. The
unexpectedly strong results reported by Vec2Text motivated us to conduct
further verification, particularly considering the typically non-intuitive and
opaque structure of high-dimensional embedding spaces. In this work, we
reproduce the Vec2Text framework and evaluate it from two perspectives: (1)
validating the original claims, and (2) extending the study through targeted
experiments. First, we successfully replicate the original key results in both
in-domain and out-of-domain settings, with only minor discrepancies arising due
to missing artifacts, such as model checkpoints and dataset splits.
Furthermore, we extend the study by conducting a parameter sensitivity
analysis, evaluating the feasibility of reconstructing sensitive inputs (e.g.,
passwords), and exploring embedding quantization as a lightweight privacy
defense. Our results show that Vec2Text is effective under ideal conditions,
capable of reconstructing even password-like sequences that lack clear
semantics. However, we identify key limitations, including its sensitivity to
input sequence length. We also find that Gaussian noise and quantization
techniques can mitigate the privacy risks posed by Vec2Text, with quantization
offering a simpler and more widely applicable solution. Our findings emphasize
the need for caution in using text embeddings and highlight the importance of
further research into robust defense mechanisms for NLP systems.

### 6. [Plausible Counterfactual Explanations of Recommendations](http://arxiv.org/pdf/2507.07919v1)

Authors: Jakub Černý, Jiří Němeček, Ivan Dovica, Jakub Mareček

Explanations play a variety of roles in various recommender systems, from a
legally mandated afterthought, through an integral element of user experience,
to a key to persuasiveness. A natural and useful form of an explanation is the
Counterfactual Explanation (CE). We present a method for generating highly
plausible CEs in recommender systems and evaluate it both numerically and with
a user study.

### 7. [The Cross-Lingual Cost: Retrieval Biases in RAG over Arabic-English Corpora](http://arxiv.org/pdf/2507.07543v1)

Authors: Chen Amiraz, Yaroslav Fyodorov, Elad Haramaty, Zohar Karnin, Liane Lewin-Eytan

Cross-lingual retrieval-augmented generation (RAG) is a critical capability
for retrieving and generating answers across languages. Prior work in this
context has mostly focused on generation and relied on benchmarks derived from
open-domain sources, most notably Wikipedia. In such settings, retrieval
challenges often remain hidden due to language imbalances, overlap with
pretraining data, and memorized content. To address this gap, we study
Arabic-English RAG in a domain-specific setting using benchmarks derived from
real-world corporate datasets. Our benchmarks include all combinations of
languages for the user query and the supporting document, drawn independently
and uniformly at random. This enables a systematic study of multilingual
retrieval behavior.
  Our findings reveal that retrieval is a critical bottleneck in cross-lingual
domain-specific scenarios, with significant performance drops occurring when
the user query and supporting document languages differ. A key insight is that
these failures stem primarily from the retriever's difficulty in ranking
documents across languages. Finally, we propose a simple retrieval strategy
that addresses this source of failure by enforcing equal retrieval from both
languages, resulting in substantial improvements in cross-lingual and overall
performance. These results highlight meaningful opportunities for improving
multilingual retrieval, particularly in practical, real-world RAG applications.

### 8. [DTECT: Dynamic Topic Explorer & Context Tracker](http://arxiv.org/pdf/2507.07910v1)

Authors: Suman Adhya, Debarshi Kumar Sanyal

The explosive growth of textual data over time presents a significant
challenge in uncovering evolving themes and trends. Existing dynamic topic
modeling techniques, while powerful, often exist in fragmented pipelines that
lack robust support for interpretation and user-friendly exploration. We
introduce DTECT (Dynamic Topic Explorer & Context Tracker), an end-to-end
system that bridges the gap between raw textual data and meaningful temporal
insights. DTECT provides a unified workflow that supports data preprocessing,
multiple model architectures, and dedicated evaluation metrics to analyze the
topic quality of temporal topic models. It significantly enhances
interpretability by introducing LLM-driven automatic topic labeling, trend
analysis via temporally salient words, interactive visualizations with
document-level summarization, and a natural language chat interface for
intuitive data querying. By integrating these features into a single, cohesive
platform, DTECT empowers users to more effectively track and understand
thematic dynamics. DTECT is open-source and available at
https://github.com/AdhyaSuman/DTECT.

### Machine Learning

### 1. [Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](http://arxiv.org/pdf/2507.07348v1)

Authors: James Chapman, Kedar Karhadkar, Guido Montufar

Deep reinforcement learning (DRL) has achieved remarkable success across
multiple domains, including competitive games, natural language processing, and
robotics. Despite these advancements, policies trained via DRL often struggle
to generalize to evaluation environments with different parameters. This
challenge is typically addressed by training with multiple contexts and/or by
leveraging additional structure in the problem. However, obtaining sufficient
training data across diverse contexts can be impractical in real-world
applications. In this work, we consider contextual Markov decision processes
(CMDPs) with transition and reward functions that exhibit regularity in context
parameters. We introduce the context-enhanced Bellman equation (CEBE) to
improve generalization when training on a single context. We prove both
analytically and empirically that the CEBE yields a first-order approximation
to the Q-function trained across multiple contexts. We then derive context
sample enhancement (CSE) as an efficient data augmentation method for
approximating the CEBE in deterministic control environments. We numerically
validate the performance of CSE in simulation environments, showcasing its
potential to improve generalization in DRL.

### 2. [Learning from positive and unlabeled examples -Finite size sample bounds](http://arxiv.org/pdf/2507.07354v1)

Authors: Farnam Mansouri, Shai Ben-David

PU (Positive Unlabeled) learning is a variant of supervised classification
learning in which the only labels revealed to the learner are of positively
labeled instances. PU learning arises in many real-world applications. Most
existing work relies on the simplifying assumptions that the positively labeled
training data is drawn from the restriction of the data generating distribution
to positively labeled instances and/or that the proportion of positively
labeled points (a.k.a. the class prior) is known apriori to the learner. This
paper provides a theoretical analysis of the statistical complexity of PU
learning under a wider range of setups. Unlike most prior work, our study does
not assume that the class prior is known to the learner. We prove upper and
lower bounds on the required sample sizes (of both the positively labeled and
the unlabeled samples).

### 3. [GRIT: Graph Transformer For Internal Ice Layer Thickness Prediction](http://arxiv.org/pdf/2507.07388v1)

Authors: Zesheng Liu, Maryam Rahnemoonfar

Gaining a deeper understanding of the thickness and variability of internal
ice layers in Radar imagery is essential in monitoring the snow accumulation,
better evaluating ice dynamics processes, and minimizing uncertainties in
climate models. Radar sensors, capable of penetrating ice, capture detailed
radargram images of internal ice layers. In this work, we introduce GRIT, graph
transformer for ice layer thickness. GRIT integrates an inductive geometric
graph learning framework with an attention mechanism, designed to map the
relationships between shallow and deeper ice layers. Compared to baseline graph
neural networks, GRIT demonstrates consistently lower prediction errors. These
results highlight the attention mechanism's effectiveness in capturing temporal
changes across ice layers, while the graph transformer combines the strengths
of transformers for learning long-range dependencies with graph neural networks
for capturing spatial patterns, enabling robust modeling of complex
spatiotemporal dynamics.

### 4. [Learning Collective Variables from Time-lagged Generation](http://arxiv.org/pdf/2507.07390v1)

Authors: Seonghyun Park, Kiyoung Seong, Soojung Yang, Rafael Gómez-Bombarelli, Sungsoo Ahn

Rare events such as state transitions are difficult to observe directly with
molecular dynamics simulations due to long timescales. Enhanced sampling
techniques overcome this by introducing biases along carefully chosen
low-dimensional features, known as collective variables (CVs), which capture
the slow degrees of freedom. Machine learning approaches (MLCVs) have automated
CV discovery, but existing methods typically focus on discriminating
meta-stable states without fully encoding the detailed dynamics essential for
accurate sampling. We propose TLC, a framework that learns CVs directly from
time-lagged conditions of a generative model. Instead of modeling the static
Boltzmann distribution, TLC models a time-lagged conditional distribution
yielding CVs to capture the slow dynamic behavior. We validate TLC on the
Alanine Dipeptide system using two CV-based enhanced sampling tasks: (i)
steered molecular dynamics (SMD) and (ii) on-the-fly probability enhanced
sampling (OPES), demonstrating equal or superior performance compared to
existing MLCV methods in both transition path sampling and state
discrimination.

### 5. [Uncertainty Quantification for Motor Imagery BCI -- Machine Learning vs. Deep Learning](http://arxiv.org/pdf/2507.07511v1)

Authors: Joris Suurmeijer, Ivo Pascal de Jong, Matias Valdenegro-Toro, Andreea Ioana Sburlea

Brain-computer interfaces (BCIs) turn brain signals into functionally useful
output, but they are not always accurate. A good Machine Learning classifier
should be able to indicate how confident it is about a given classification, by
giving a probability for its classification. Standard classifiers for Motor
Imagery BCIs do give such probabilities, but research on uncertainty
quantification has been limited to Deep Learning. We compare the uncertainty
quantification ability of established BCI classifiers using Common Spatial
Patterns (CSP-LDA) and Riemannian Geometry (MDRM) to specialized methods in
Deep Learning (Deep Ensembles and Direct Uncertainty Quantification) as well as
standard Convolutional Neural Networks (CNNs).
  We found that the overconfidence typically seen in Deep Learning is not a
problem in CSP-LDA and MDRM. We found that MDRM is underconfident, which we
solved by adding Temperature Scaling (MDRM-T). CSP-LDA and MDRM-T give the best
uncertainty estimates, but Deep Ensembles and standard CNNs give the best
classifications. We show that all models are able to separate between easy and
difficult estimates, so that we can increase the accuracy of a Motor Imagery
BCI by rejecting samples that are ambiguous.

### 6. [Sparse Self-Federated Learning for Energy Efficient Cooperative Intelligence in Society 5.0](http://arxiv.org/pdf/2507.07613v1)

Authors: Davide Domini, Laura Erhan, Gianluca Aguzzi, Lucia Cavallaro, Amirhossein Douzandeh Zenoozi, Antonio Liotta, Mirko Viroli

Federated Learning offers privacy-preserving collaborative intelligence but
struggles to meet the sustainability demands of emerging IoT ecosystems
necessary for Society 5.0-a human-centered technological future balancing
social advancement with environmental responsibility. The excessive
communication bandwidth and computational resources required by traditional FL
approaches make them environmentally unsustainable at scale, creating a
fundamental conflict with green AI principles as billions of
resource-constrained devices attempt to participate. To this end, we introduce
Sparse Proximity-based Self-Federated Learning (SParSeFuL), a resource-aware
approach that bridges this gap by combining aggregate computing for
self-organization with neural network sparsification to reduce energy and
bandwidth consumption.

### 7. [Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](http://arxiv.org/pdf/2507.07621v1)

Authors: Junyu Luo, Yuhao Tang, Yiwei Fu, Xiao Luo, Zhizhuo Kou, Zhiping Xiao, Wei Ju, Wentao Zhang, Ming Zhang

Unsupervised Graph Domain Adaptation (UGDA) leverages labeled source domain
graphs to achieve effective performance in unlabeled target domains despite
distribution shifts. However, existing methods often yield suboptimal results
due to the entanglement of causal-spurious features and the failure of global
alignment strategies. We propose SLOGAN (Sparse Causal Discovery with
Generative Intervention), a novel approach that achieves stable graph
representation transfer through sparse causal modeling and dynamic intervention
mechanisms. Specifically, SLOGAN first constructs a sparse causal graph
structure, leveraging mutual information bottleneck constraints to disentangle
sparse, stable causal features while compressing domain-dependent spurious
correlations through variational inference. To address residual spurious
correlations, we innovatively design a generative intervention mechanism that
breaks local spurious couplings through cross-domain feature recombination
while maintaining causal feature semantic consistency via covariance
constraints. Furthermore, to mitigate error accumulation in target domain
pseudo-labels, we introduce a category-adaptive dynamic calibration strategy,
ensuring stable discriminative learning. Extensive experiments on multiple
real-world datasets demonstrate that SLOGAN significantly outperforms existing
baselines.

### 8. [HLF-FSL. A Decentralized Federated Split Learning Solution for IoT on Hyperledger Fabric](http://arxiv.org/pdf/2507.07637v1)

Authors: Carlos Beis Penedo, Rebeca P. Díaz Redondo, Ana Fernández Vilas, Manuel Fernández Veiga, Francisco Troncoso Pastoriza

Collaborative machine learning in sensitive domains demands scalable, privacy
preserving solutions for enterprise deployment. Conventional Federated Learning
(FL) relies on a central server, introducing single points of failure and
privacy risks, while Split Learning (SL) partitions models for privacy but
scales poorly due to sequential training. We present a decentralized
architecture that combines Federated Split Learning (FSL) with the permissioned
blockchain Hyperledger Fabric (HLF). Our chaincode orchestrates FSL's split
model execution and peer-to-peer aggregation without any central coordinator,
leveraging HLF's transient fields and Private Data Collections (PDCs) to keep
raw data and model activations private. On CIFAR-10 and MNIST benchmarks,
HLF-FSL matches centralized FSL accuracy while reducing per epoch training time
compared to Ethereum-based works. Performance and scalability tests show
minimal blockchain overhead and preserved accuracy, demonstrating enterprise
grade viability.

### 9. [Some Theoretical Results on Layerwise Effective Dimension Oscillations in Finite Width ReLU Networks](http://arxiv.org/pdf/2507.07675v1)

Authors: Darshan Makwana

We analyze the layerwise effective dimension (rank of the feature matrix) in
fully-connected ReLU networks of finite width. Specifically, for a fixed batch
of $m$ inputs and random Gaussian weights, we derive closed-form expressions
for the expected rank of the \$m\times n\$ hidden activation matrices. Our main
result shows that $\mathbb{E}[EDim(\ell)]=m[1-(1-2/\pi)^\ell]+O(e^{-c m})$ so
that the rank deficit decays geometrically with ratio $1-2 / \pi \approx
0.3634$. We also prove a sub-Gaussian concentration bound, and identify the
"revival" depths at which the expected rank attains local maxima. In
particular, these peaks occur at depths
$\ell_k^*\approx(k+1/2)\pi/\log(1/\rho)$ with height $\approx (1-e^{-\pi/2}) m
\approx 0.79m$. We further show that this oscillatory rank behavior is a
finite-width phenomenon: under orthogonal weight initialization or strong
negative-slope leaky-ReLU, the rank remains (nearly) full. These results
provide a precise characterization of how random ReLU layers alternately
collapse and partially revive the subspace of input variations, adding nuance
to prior work on expressivity of deep networks.

### 10. [Deep Survival Analysis in Multimodal Medical Data: A Parametric and Probabilistic Approach with Competing Risks](http://arxiv.org/pdf/2507.07804v1)

Authors: Alba Garrido, Alejandro Almodóvar, Patricia A. Apellániz, Juan Parras, Santiago Zazo

Accurate survival prediction is critical in oncology for prognosis and
treatment planning. Traditional approaches often rely on a single data
modality, limiting their ability to capture the complexity of tumor biology. To
address this challenge, we introduce a multimodal deep learning framework for
survival analysis capable of modeling both single and competing risks
scenarios, evaluating the impact of integrating multiple medical data sources
on survival predictions. We propose SAMVAE (Survival Analysis Multimodal
Variational Autoencoder), a novel deep learning architecture designed for
survival prediction that integrates six data modalities: clinical variables,
four molecular profiles, and histopathological images. SAMVAE leverages
modality specific encoders to project inputs into a shared latent space,
enabling robust survival prediction while preserving modality specific
information. Its parametric formulation enables the derivation of clinically
meaningful statistics from the output distributions, providing patient-specific
insights through interactive multimedia that contribute to more informed
clinical decision-making and establish a foundation for interpretable,
data-driven survival analysis in oncology. We evaluate SAMVAE on two cancer
cohorts breast cancer and lower grade glioma applying tailored preprocessing,
dimensionality reduction, and hyperparameter optimization. The results
demonstrate the successful integration of multimodal data for both standard
survival analysis and competing risks scenarios across different datasets. Our
model achieves competitive performance compared to state-of-the-art multimodal
survival models. Notably, this is the first parametric multimodal deep learning
architecture to incorporate competing risks while modeling continuous time to a
specific event, using both tabular and image data.

### Neural and Evolutionary Computing

### 1. [Homeostatic Adaptation of Optimal Population Codes under Metabolic Stress](http://arxiv.org/pdf/2507.07874v1)

Authors: Yi-Chun Hung, Gregory Schwartz, Emily A. Cooper, Emma Alexander

Information processing in neural populations is inherently constrained by
metabolic resource limits and noise properties, with dynamics that are not
accurately described by existing mathematical models. Recent data, for example,
shows that neurons in mouse visual cortex go into a "low power mode" in which
they maintain firing rate homeostasis while expending less energy. This
adaptation leads to increased neuronal noise and tuning curve flattening in
response to metabolic stress. We have developed a theoretical population coding
framework that captures this behavior using two novel, surprisingly simple
constraints: an approximation of firing rate homeostasis and an energy limit
tied to noise levels via biophysical simulation. A key feature of our
contribution is an energy budget model directly connecting adenosine
triphosphate (ATP) use in cells to a fully explainable mathematical framework
that generalizes existing optimal population codes. Specifically, our
simulation provides an energy-dependent dispersed Poisson noise model, based on
the assumption that the cell will follow an optimal decay path to produce the
least-noisy spike rate that is possible at a given cellular energy budget. Each
state along this optimal path is associated with properties (resting potential
and leak conductance) which can be measured in electrophysiology experiments
and have been shown to change under prolonged caloric deprivation. We
analytically derive the optimal coding strategy for neurons under varying
energy budgets and coding goals, and show how our method uniquely captures how
populations of tuning curves adapt while maintaining homeostasis, as has been
observed empirically.

### 2. [EEvAct: Early Event-Based Action Recognition with High-Rate Two-Stream Spiking Neural Networks](http://arxiv.org/pdf/2507.07734v1)

Authors: Michael Neumeier, Jules Lecomte, Nils Kazinski, Soubarna Banik, Bing Li, Axel von Arnim

Recognizing human activities early is crucial for the safety and
responsiveness of human-robot and human-machine interfaces. Due to their high
temporal resolution and low latency, event-based vision sensors are a perfect
match for this early recognition demand. However, most existing processing
approaches accumulate events to low-rate frames or space-time voxels which
limits the early prediction capabilities. In contrast, spiking neural networks
(SNNs) can process the events at a high-rate for early predictions, but most
works still fall short on final accuracy. In this work, we introduce a
high-rate two-stream SNN which closes this gap by outperforming previous work
by 2% in final accuracy on the large-scale THU EACT-50 dataset. We benchmark
the SNNs within a novel early event-based recognition framework by reporting
Top-1 and Top-5 recognition scores for growing observation time. Finally, we
exemplify the impact of these methods on a real-world task of early action
triggering for human motion capture in sports.

### Networking and Internet Architecture

### 1. [PHandover: Parallel Handover in Mobile Satellite Network](http://arxiv.org/pdf/2507.07437v1)

Authors: Jiasheng Wu, Shaojie Su, Wenjun Zhu, Xiong Wang, Jingjing Zhang, Xingqiu He, Yue Gao

The construction of Low Earth Orbit (LEO) satellite constellations has
recently attracted tremendous attention from both academia and industry. The 5G
and 6G standards have identified LEO satellite networks as a key component of
future mobile networks. However, due to the high-speed movement of satellites,
ground terminals often experience frequent and high-latency handovers, which
significantly deteriorate the performance of latency-sensitive applications. To
address this challenge, we propose a parallel handover mechanism for mobile
satellite networks that can considerably reduce handover latency. The main idea
is to employ plan-based handovers instead of measurement-based handovers to
avoid interactions between the access and core networks, thereby eliminating
the significant time overhead associated with traditional handover procedures.
Specifically, we introduce a novel network function named the Satellite
Synchronized Function (SSF), which is designed to be fully compliant with the
standard 5G core network. In addition, we propose a machine learning model for
signal strength prediction, coupled with an efficient handover scheduling
algorithm. We have conducted extensive experiments, and the results demonstrate
that our proposed handover scheme can reduce handover latency by 21\times
compared to the standard NTN handover scheme and two other existing handover
approaches, along with significant improvements in network stability and
user-level performance.

### 2. [Energy Transfer and Data Collection from Batteryless Sensors in Low-altitude Wireless Networks](http://arxiv.org/pdf/2507.07481v1)

Authors: Wen Zhang, Aimin Wang, Jiahui Li, Geng Sun, Jiacheng Wang, Weijie Yuan, Dusit Niyato

The integration of wireless power transfer (WPT) with Internet of Things
(IoT) offers promising solutions for sensing applications, but faces
significant challenges when deployed in hard-to-access areas such as
high-temperature environments. In such extreme conditions, traditional fixed
WPT infrastructure cannot be safely installed, and batteries rapidly degrade
due to hardware failures. In this paper, we propose an uncrewed aerial vehicle
(UAV)-assisted data collection and WPT framework for batteryless sensor (BLS)
networks deployed in these challenging environments. Specifically, we consider
a practical scenario where a UAV first transfers energy to BLS nodes via WPT,
enabling these nodes to subsequently transmit their collected data to the UAV
through orthogonal frequency-division multiple access (OFDMA). Then, we
formulate a multi-objective optimization problem that aims to maximize the fair
data collection volume while minimizing the UAV energy consumption through
joint optimization of transmit power allocation and flight trajectory planning.
Due to the non-convex nature and dynamic characteristics of this problem,
conventional optimization methods prove inadequate. To address these
challenges, we propose an enhanced soft actor-critic algorithm with
parameter-free attention, prioritized experience replay, and value-based reward
centering (SAC-PPV), thereby improving the exploration efficiency and learning
stability of the algorithm in complex WPT scenarios. Simulation results
demonstrate that the proposed approach consistently outperforms benchmark
algorithms under various network configurations.

### 3. [A Fragmentation-Aware Adaptive Bilevel Search Framework for Service Mapping in Computing Power Networks](http://arxiv.org/pdf/2507.07535v1)

Authors: Jingzhao Xie, Zhenglian Li, Gang Sun, Long Luo, Hongfang Yu, Dusit Niyato

Computing Power Network (CPN) unifies wide-area computing resources through
coordinated network control, while cloud-native abstractions enable flexible
resource orchestration and on-demand service provisioning atop the elastic
infrastructure CPN provides. However, current approaches fall short of fully
integrating computing resources via network-enabled coordination as envisioned
by CPN. In particular, optimally mapping services to an underlying
infrastructure to maximize resource efficiency and service satisfaction remains
challenging. To overcome this challenge, we formally define the service mapping
problem in CPN, establish its theoretical intractability, and identify key
challenges in practical optimization. We propose Adaptive Bilevel Search (ABS),
a modular framework featuring (1) graph partitioning-based reformulation to
capture variable coupling, (2) a bilevel optimization architecture for
efficient global exploration with local optimality guarantees, and (3)
fragmentation-aware evaluation for global performance guidance. Implemented
using distributed particle swarm optimization, ABS is extensively evaluated
across diverse CPN scenarios, consistently outperforming existing approaches.
Notably, in complex scenarios, ABS achieves up to 73.2% higher computing
resource utilization and a 60.2% higher service acceptance ratio compared to
the best-performing baseline.

### 4. [Can cloud-based VR streaming handle Wi-Fi OBSS contention?](http://arxiv.org/pdf/2507.07677v1)

Authors: Miguel Casasnovas, Marc Carrascosa-Zamacois, Boris Bellalta

This paper experimentally analyzes the negative impact of contention caused
by neighboring Wi-Fi networks operating on overlapping channels on Virtual
Reality (VR) streaming over Wi-Fi, focusing on scenarios of partial and full
channel overlap within an 80 MHz channel. Our results show that (i) increasing
the number of 80 MHz Overlapping Basic Service Sets (OBSSs) intensifies
contention and degrades VR streaming performance; (ii) OBSS activity on the
secondary-sided 40 MHz portion degrades performance more than activity on the
primary-sided 40 MHz portion; (iii) for the same aggregate load, full channel
overlap with two 40 MHz OBSS contenders is less detrimental than partial
overlap with a single high-load 40 MHz contender, but more disruptive than full
overlap with two 80 MHz contenders; and (iv) full channel overlap with two 40
MHz OBSS contenders has a smaller impact on VR streaming under symmetric
traffic loads than under asymmetric loads. Moreover, our results demonstrate
that our previously proposed Network-aware Step-wise adaptive bitrate algorithm
for VR streaming (NeSt-VR) effectively mitigates performance degradation in
OBSS environments, enabling VR streaming under heavier OBSS traffic conditions.

### 5. [CHOMET: Conditional Handovers via Meta-Learning](http://arxiv.org/pdf/2507.07581v1)

Authors: Michail Kalntis, Fernando A. Kuipers, George Iosifidis

Handovers (HOs) are the cornerstone of modern cellular networks for enabling
seamless connectivity to a vast and diverse number of mobile users. However, as
mobile networks become more complex with more diverse users and smaller cells,
traditional HOs face significant challenges, such as prolonged delays and
increased failures. To mitigate these issues, 3GPP introduced conditional
handovers (CHOs), a new type of HO that enables the preparation (i.e., resource
allocation) of multiple cells for a single user to increase the chance of HO
success and decrease the delays in the procedure. Despite its advantages, CHO
introduces new challenges that must be addressed, including efficient resource
allocation and managing signaling/communication overhead from frequent cell
preparations and releases. This paper presents a novel framework aligned with
the O-RAN paradigm that leverages meta-learning for CHO optimization, providing
robust dynamic regret guarantees and demonstrating at least 180% superior
performance than other 3GPP benchmarks in volatile signal conditions.

### 6. [HaLert: A Resilient Smart City Architecture for Post-Disaster Based on Wi-Fi HaLow Mesh and SDN](http://arxiv.org/pdf/2507.07841v1)

Authors: Ana Rita Ortigoso, Gabriel Vieira, Daniel Fuentes, Luís Frazão, Nuno Costa, António Pereira

Events such as catastrophes and disasters are, in most cases, unpredictable.
Consequently, reusing existing infrastructures to develop alternative
communication strategies after disasters is essential to minimise the impact of
these events on the population's ability to communicate and promptly receive
alerts from authorities. In this context, the emergence of smart cities,
characterised by dense and geographically distributed IoT networks, presents
significant potential for such reuse. This work proposes HaLert, a resilient
architecture for smart cities based on a Wi-Fi HaLow IEEE 802.11s mesh network,
whose resources can be readily reallocated to support a emergency communication
system to exchange messages (including text, location, image, audio, and video)
between citizens, authorities, and between both parties. To facilitate remote
monitoring and configuration of the network, the architecture incorporates the
SDN (Software-Defined Networking) paradigm, supported by a LoRa controlled
flooding mesh network. A prototype was developed based on this architecture and
tested in a real urban scenario comprising both indoor and outdoor
environments. The results demonstrated that, despite the significant impact of
obstacles, lack of line-of-sight, and terrain slopes on the latency (average
latency between 15 and 54.8 ms) and throughput (upload bitrates between 134 and
726 Kbps and download bitrates between 117 and 682 Kbps) of the Wi-Fi HaLow
network, it remained stable and resilient, successfully providing all
functionalities associated with the HaLert architecture. The tests conducted on
the LoRa network revealed a high average message success rate of 94.96%.

### Robotics

### 1. [UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots](http://arxiv.org/pdf/2507.07356v1)

Authors: Kangning Yin, Weishuai Zeng, Ke Fan, Zirui Wang, Qiang Zhang, Zheng Tian, Jingbo Wang, Jiangmiao Pang, Weinan Zhang

Humanoid robots must achieve diverse, robust, and generalizable whole-body
control to operate effectively in complex, human-centric environments. However,
existing methods, particularly those based on teacher-student frameworks often
suffer from a loss of motion diversity during policy distillation and exhibit
limited generalization to unseen behaviors. In this work, we present
UniTracker, a simplified yet powerful framework that integrates a Conditional
Variational Autoencoder (CVAE) into the student policy to explicitly model the
latent diversity of human motion. By leveraging a learned CVAE prior, our
method enables the student to retain expressive motion characteristics while
improving robustness and adaptability under partial observations. The result is
a single policy capable of tracking a wide spectrum of whole-body motions with
high fidelity and stability. Comprehensive experiments in both simulation and
real-world deployments demonstrate that UniTracker significantly outperforms
MLP-based DAgger baselines in motion quality, generalization to unseen
references, and deployment robustness, offering a practical and scalable
solution for expressive humanoid control.

### 2. [Towards Safe Autonomous Driving: A Real-Time Safeguarding Concept for Motion Planning Algorithms](http://arxiv.org/pdf/2507.07444v1)

Authors: Korbinian Moller, Rafael Neher, Marvin Seegert, Johannes Betz

Ensuring the functional safety of motion planning modules in autonomous
vehicles remains a critical challenge, especially when dealing with complex or
learning-based software. Online verification has emerged as a promising
approach to monitor such systems at runtime, yet its integration into embedded
real-time environments remains limited. This work presents a safeguarding
concept for motion planning that extends prior approaches by introducing a time
safeguard. While existing methods focus on geometric and dynamic feasibility,
our approach additionally monitors the temporal consistency of planning outputs
to ensure timely system response. A prototypical implementation on a real-time
operating system evaluates trajectory candidates using constraint-based
feasibility checks and cost-based plausibility metrics. Preliminary results
show that the safeguarding module operates within real-time bounds and
effectively detects unsafe trajectories. However, the full integration of the
time safeguard logic and fallback strategies is ongoing. This study contributes
a modular and extensible framework for runtime trajectory verification and
highlights key aspects for deployment on automotive-grade hardware. Future work
includes completing the safeguarding logic and validating its effectiveness
through hardware-in-the-loop simulations and vehicle-based testing. The code is
available at: https://github.com/TUM-AVS/motion-planning-supervisor

### 3. [SCREP: Scene Coordinate Regression and Evidential Learning-based Perception-Aware Trajectory Generation](http://arxiv.org/pdf/2507.07467v1)

Authors: Juyeop Han, Lukas Lao Beyer, Guilherme V. Cavalheiro, Sertac Karaman

Autonomous flight in GPS denied indoor spaces requires trajectories that keep
visual localization error tightly bounded across varied missions. Whereas
visual inertial odometry (VIO) accumulates drift over time, scene coordinate
regression (SCR) yields drift-free, high accuracy absolute pose estimation. We
present a perception-aware framework that couples an evidential learning-based
SCR pose estimator with a receding horizon trajectory optimizer. The optimizer
steers the onboard camera toward pixels whose uncertainty predicts reliable
scene coordinates, while a fixed-lag smoother fuses the low rate SCR stream
with high rate IMU data to close the perception control loop in real time. In
simulation, our planner reduces translation (rotation) mean error by 54% / 15%
(40% / 31%) relative to yaw fixed and forward-looking baselines, respectively.
Moreover, hardware in the loop experiment validates the feasibility of our
proposed framework.

### 4. [Implementation and Assessment of an Augmented Training Curriculum for Surgical Robotics](http://arxiv.org/pdf/2507.07718v1)

Authors: Alberto Rota, Ke Fan, Elena De Momi

The integration of high-level assistance algorithms in surgical robotics
training curricula may be beneficial in establishing a more comprehensive and
robust skillset for aspiring surgeons, improving their clinical performance as
a consequence. This work presents the development and validation of a
haptic-enhanced Virtual Reality simulator for surgical robotics training,
featuring 8 surgical tasks that the trainee can interact with thanks to the
embedded physics engine. This virtual simulated environment is augmented by the
introduction of high-level haptic interfaces for robotic assistance that aim at
re-directing the motion of the trainee's hands and wrists toward targets or
away from obstacles, and providing a quantitative performance score after the
execution of each training exercise.An experimental study shows that the
introduction of enhanced robotic assistance into a surgical robotics training
curriculum improves performance during the training process and, crucially,
promotes the transfer of the acquired skills to an unassisted surgical
scenario, like the clinical one.

### 5. [Distributed Surface Inspection via Operational Modal Analysis by a Swarm of Miniaturized Vibration-Sensing Robots](http://arxiv.org/pdf/2507.07724v1)

Authors: Thiemen Siemensma, Niels de Boer, Bahar Haghighat

Robot swarms offer the potential to serve a variety of distributed sensing
applications. An interesting real-world application that stands to benefit
significantly from deployment of swarms is structural monitoring, where
traditional sensor networks face challenges in structural coverage due to their
static nature. This paper investigates the deployment of a swarm of
miniaturized vibration sensing robots to inspect and localize structural
damages on a surface section within a high-fidelity simulation environment. In
particular, we consider a 1 m x 1 m x 3 mm steel surface section and utilize
finite element analysis using Abaqus to obtain realistic structural vibration
data. The resulting vibration data is imported into the physics-based robotic
simulator Webots, where we simulate the dynamics of our surface inspecting
robot swarm. We employ (i) Gaussian process estimators to guide the robots'
exploration as they collect vibration samples across the surface and (ii)
operational modal analysis to detect structural damages by estimating and
comparing existing and intact structural vibration patterns. We analyze the
influence of exploration radii on estimation uncertainty and assess the
effectiveness of our method across 10 randomized scenarios, where the number,
locations, surface area, and depth of structural damages vary. Our simulation
studies validate the efficacy of our miniaturized robot swarm for
vibration-based structural inspection.

### 6. [On the capabilities of LLMs for classifying and segmenting time series of fruit picking motions into primitive actions](http://arxiv.org/pdf/2507.07745v1)

Authors: Eleni Konstantinidou, Nikolaos Kounalakis, Nikolaos Efstathopoulos, Dimitrios Papageorgiou

Despite their recent introduction to human society, Large Language Models
(LLMs) have significantly affected the way we tackle mental challenges in our
everyday lives. From optimizing our linguistic communication to assisting us in
making important decisions, LLMs, such as ChatGPT, are notably reducing our
cognitive load by gradually taking on an increasing share of our mental
activities. In the context of Learning by Demonstration (LbD), classifying and
segmenting complex motions into primitive actions, such as pushing, pulling,
twisting etc, is considered to be a key-step towards encoding a task. In this
work, we investigate the capabilities of LLMs to undertake this task,
considering a finite set of predefined primitive actions found in fruit picking
operations. By utilizing LLMs instead of simple supervised learning or analytic
methods, we aim at making the method easily applicable and deployable in a
real-life scenario. Three different fine-tuning approaches are investigated,
compared on datasets captured kinesthetically, using a UR10e robot, during a
fruit-picking scenario.

### 7. [IRAF-SLAM: An Illumination-Robust and Adaptive Feature-Culling Front-End for Visual SLAM in Challenging Environments](http://arxiv.org/pdf/2507.07752v1)

Authors: Thanh Nguyen Canh, Bao Nguyen Quoc, Haolan Zhang, Bupesh Rethinam Veeraiah, Xiem HoangVan, Nak Young Chong

Robust Visual SLAM (vSLAM) is essential for autonomous systems operating in
real-world environments, where challenges such as dynamic objects, low texture,
and critically, varying illumination conditions often degrade performance.
Existing feature-based SLAM systems rely on fixed front-end parameters, making
them vulnerable to sudden lighting changes and unstable feature tracking. To
address these challenges, we propose ``IRAF-SLAM'', an Illumination-Robust and
Adaptive Feature-Culling front-end designed to enhance vSLAM resilience in
complex and challenging environments. Our approach introduces: (1) an image
enhancement scheme to preprocess and adjust image quality under varying
lighting conditions; (2) an adaptive feature extraction mechanism that
dynamically adjusts detection sensitivity based on image entropy, pixel
intensity, and gradient analysis; and (3) a feature culling strategy that
filters out unreliable feature points using density distribution analysis and a
lighting impact factor. Comprehensive evaluations on the TUM-VI and European
Robotics Challenge (EuRoC) datasets demonstrate that IRAF-SLAM significantly
reduces tracking failures and achieves superior trajectory accuracy compared to
state-of-the-art vSLAM methods under adverse illumination conditions. These
results highlight the effectiveness of adaptive front-end strategies in
improving vSLAM robustness without incurring significant computational
overhead. The implementation of IRAF-SLAM is publicly available at
https://thanhnguyencanh. github.io/IRAF-SLAM/.

### 8. [Collaborative Human-Robot Surgery for Mandibular Angle Split Osteotomy: Optical Tracking based Approach](http://arxiv.org/pdf/2507.07794v1)

Authors: Zhe Han, Huanyu Tian, Tom Vercauteren, Da Liu, Changsheng Li, Xingguang Duan

Mandibular Angle Split Osteotomy (MASO) is a significant procedure in oral
and maxillofacial surgery. Despite advances in technique and instrumentation,
its success still relies heavily on the surgeon's experience. In this work, a
human-robot collaborative system is proposed to perform MASO according to a
preoperative plan and under guidance of a surgeon. A task decomposition
methodology is used to divide the collaborative surgical procedure into three
subtasks: (1) positional control and (2) orientation control, both led by the
robot for precise alignment; and (3) force-control, managed by surgeon to
ensure safety. Additionally, to achieve patient tracking without the need for a
skull clamp, an optical tracking system (OTS) is utilized. Movement of the
patient mandibular is measured with an optical-based tracker mounted on a
dental occlusal splint. A registration method and Robot-OTS calibration method
are introduced to achieve reliable navigation within our framework. The
experiments of drilling were conducted on the realistic phantom model, which
demonstrated that the average error between the planned and actual drilling
points is 1.85mm.

### 9. [Beyond Robustness: Learning Unknown Dynamic Load Adaptation for Quadruped Locomotion on Rough Terrain](http://arxiv.org/pdf/2507.07825v1)

Authors: Leixin Chang, Yuxuan Nai, Hua Chen, Liangjing Yang

Unknown dynamic load carrying is one important practical application for
quadruped robots. Such a problem is non-trivial, posing three major challenges
in quadruped locomotion control. First, how to model or represent the dynamics
of the load in a generic manner. Second, how to make the robot capture the
dynamics without any external sensing. Third, how to enable the robot to
interact with load handling the mutual effect and stabilizing the load. In this
work, we propose a general load modeling approach called load characteristics
modeling to capture the dynamics of the load. We integrate this proposed
modeling technique and leverage recent advances in Reinforcement Learning (RL)
based locomotion control to enable the robot to infer the dynamics of load
movement and interact with the load indirectly to stabilize it and realize the
sim-to-real deployment to verify its effectiveness in real scenarios. We
conduct extensive comparative simulation experiments to validate the
effectiveness and superiority of our proposed method. Results show that our
method outperforms other methods in sudden load resistance, load stabilizing
and locomotion with heavy load on rough terrain.
\href{https://leixinjonaschang.github.io/leggedloadadapt.github.io/}{Project
Page}.

### 10. [Perceptual Distortions and Autonomous Representation Learning in a Minimal Robotic System](http://arxiv.org/pdf/2507.07845v1)

Authors: David Warutumo, Ciira wa Maina

Autonomous agents, particularly in the field of robotics, rely on sensory
information to perceive and navigate their environment. However, these sensory
inputs are often imperfect, leading to distortions in the agent's internal
representation of the world. This paper investigates the nature of these
perceptual distortions and how they influence autonomous representation
learning using a minimal robotic system. We utilize a simulated two-wheeled
robot equipped with distance sensors and a compass, operating within a simple
square environment. Through analysis of the robot's sensor data during random
exploration, we demonstrate how a distorted perceptual space emerges. Despite
these distortions, we identify emergent structures within the perceptual space
that correlate with the physical environment, revealing how the robot
autonomously learns a structured representation for navigation without explicit
spatial information. This work contributes to the understanding of embodied
cognition, minimal agency, and the role of perception in self-generated
navigation strategies in artificial life.

### Software Engineering

### 1. [Automatic Generation of Explainability Requirements and Software Explanations From User Reviews](http://arxiv.org/pdf/2507.07344v1)

Authors: Martin Obaidi, Jannik Fischbach, Jakob Droste, Hannah Deters, Marc Herrmann, Jil Klünder, Steffen Krätzig, Hugo Villamizar, Kurt Schneider

Explainability has become a crucial non-functional requirement to enhance
transparency, build user trust, and ensure regulatory compliance. However,
translating explanation needs expressed in user feedback into structured
requirements and corresponding explanations remains challenging. While existing
methods can identify explanation-related concerns in user reviews, there is no
established approach for systematically deriving requirements and generating
aligned explanations. To contribute toward addressing this gap, we introduce a
tool-supported approach that automates this process. To evaluate its
effectiveness, we collaborated with an industrial automation manufacturer to
create a dataset of 58 user reviews, each annotated with manually crafted
explainability requirements and explanations. Our evaluation shows that while
AI-generated requirements often lack relevance and correctness compared to
human-created ones, the AI-generated explanations are frequently preferred for
their clarity and style. Nonetheless, correctness remains an issue,
highlighting the importance of human validation. This work contributes to the
advancement of explainability requirements in software systems by (1)
introducing an automated approach to derive requirements from user reviews and
generate corresponding explanations, (2) providing empirical insights into the
strengths and limitations of automatically generated artifacts, and (3)
releasing a curated dataset to support future research on the automatic
generation of explainability requirements.

### 2. [Towards an Engineering Workflow Management System for Asset Administration Shells using BPMN](http://arxiv.org/pdf/2507.07468v1)

Authors: Sten Grüner, Nafise Eskandani

The integration of Industry 4.0 technologies into engineering workflows is an
essential step toward automating and optimizing plant and process engineering
processes. The Asset Administration Shell (AAS) serves as a key enabler for
creating interoperable Digital Twins that facilitate engineering data exchange
and automation. This paper explores the use of AAS within engineering
workflows, particularly in combination with Business Process Model and Notation
(BPMN) to define structured and automated processes. We propose a distributed
AAS copy-on-write infrastructure that enhances security and scalability while
enabling seamless cross organizational collaboration. We also introduce a
workflow management prototype automating AAS operations and engineering
workflows, improving efficiency and traceability.

### 3. [From Requirements to Code: Understanding Developer Practices in LLM-Assisted Software Engineering](http://arxiv.org/pdf/2507.07548v1)

Authors: Jonathan Ullrich, Matthias Koch, Andreas Vogelsang

With the advent of generative LLMs and their advanced code generation
capabilities, some people already envision the end of traditional software
engineering, as LLMs may be able to produce high-quality code based solely on
the requirements a domain expert feeds into the system. The feasibility of this
vision can be assessed by understanding how developers currently incorporate
requirements when using LLMs for code generation-a topic that remains largely
unexplored. We interviewed 18 practitioners from 14 companies to understand how
they (re)use information from requirements and other design artifacts to feed
LLMs when generating code. Based on our findings, we propose a theory that
explains the processes developers employ and the artifacts they rely on. Our
theory suggests that requirements, as typically documented, are too abstract
for direct input into LLMs. Instead, they must first be manually decomposed
into programming tasks, which are then enriched with design decisions and
architectural constraints before being used in prompts. Our study highlights
that fundamental RE work is still necessary when LLMs are used to generate
code. Our theory is important for contextualizing scientific approaches to
automating requirements-centric SE tasks.

### 4. [Prompt Engineering for Requirements Engineering: A Literature Review and Roadmap](http://arxiv.org/pdf/2507.07682v1)

Authors: Kaicheng Huang, Fanyu Wang, Yutan Huang, Chetan Arora

Advancements in large language models (LLMs) have led to a surge of prompt
engineering (PE) techniques that can enhance various requirements engineering
(RE) tasks. However, current LLMs are often characterized by significant
uncertainty and a lack of controllability. This absence of clear guidance on
how to effectively prompt LLMs acts as a barrier to their trustworthy
implementation in the RE field. We present the first roadmap-oriented
systematic literature review of Prompt Engineering for RE (PE4RE). Following
Kitchenham's and Petersen's secondary-study protocol, we searched six digital
libraries, screened 867 records, and analyzed 35 primary studies. To bring
order to a fragmented landscape, we propose a hybrid taxonomy that links
technique-oriented patterns (e.g., few-shot, Chain-of-Thought) to task-oriented
RE roles (elicitation, validation, traceability). Two research questions, with
five sub-questions, map the tasks addressed, LLM families used, and prompt
types adopted, and expose current limitations and research gaps. Finally, we
outline a step-by-step roadmap showing how today's ad-hoc PE prototypes can
evolve into reproducible, practitioner-friendly workflows.

### 5. [From Domain Documents to Requirements: Retrieval-Augmented Generation in the Space Industry](http://arxiv.org/pdf/2507.07689v1)

Authors: Chetan Arora, Fanyu Wang, Chakkrit Tantithamthavorn, Aldeida Aleti, Shaun Kenyon

Requirements engineering (RE) in the space industry is inherently complex,
demanding high precision, alignment with rigorous standards, and adaptability
to mission-specific constraints. Smaller space organisations and new entrants
often struggle to derive actionable requirements from extensive, unstructured
documents such as mission briefs, interface specifications, and regulatory
standards. In this innovation opportunity paper, we explore the potential of
Retrieval-Augmented Generation (RAG) models to support and (semi-)automate
requirements generation in the space domain. We present a modular, AI-driven
approach that preprocesses raw space mission documents, classifies them into
semantically meaningful categories, retrieves contextually relevant content
from domain standards, and synthesises draft requirements using large language
models (LLMs). We apply the approach to a real-world mission document from the
space domain to demonstrate feasibility and assess early outcomes in
collaboration with our industry partner, Starbound Space Solutions. Our
preliminary results indicate that the approach can reduce manual effort,
improve coverage of relevant requirements, and support lightweight compliance
alignment. We outline a roadmap toward broader integration of AI in RE
workflows, intending to lower barriers for smaller organisations to participate
in large-scale, safety-critical missions.

### 6. [Toolchain for Faster Iterations in Quantum Software Development](http://arxiv.org/pdf/2507.07448v1)

Authors: Otso Kinanen, Andrés D. Muñoz-Moller, Vlad Stirbu, Tommi Mikkonen

Quantum computing proposes a revolutionary paradigm that can radically
transform numerous scientific and industrial application domains. To realize
this promise, these new capabilities need software solutions that are able to
effectively harness its power. However, developers may face significant
challenges when developing and executing quantum software due to the limited
availability of quantum computer hardware, high computational demands of
simulating quantum computers on classical systems, and complicated technology
stack to enable currently available accelerators into development environments.
These limitations make it difficult for the developer to create an efficient
workflow for quantum software development. In this paper, we investigate the
potential of using remote computational capabilities in an efficient manner to
improve the workflow of quantum software developers, by lowering the barrier of
moving between local execution and computationally more efficient remote
hardware and offering speedup in execution with simulator surroundings. The
goal is to allow the development of more complex circuits and to support an
iterative software development approach. In our experiment, with the solution
presented in this paper, we have obtained up to 5 times faster circuit
execution runtime, and enabled qubit ranges from 21 to 29 qubits with a simple
plug-and-play kernel for the Jupyter notebook.

### 7. [ProvideQ: A Quantum Optimization Toolbox](http://arxiv.org/pdf/2507.07649v1)

Authors: Domenik Eichhorn, Nick Poser, Maximilian Schweikart, Ina Schaefer

Hybrid solvers for combinatorial optimization problems combine the advantages
of classical and quantum computing to overcome difficult computational
challenges. Although their theoretical performance seems promising, their
practical applicability is challenging due to the lack of a technological stack
that can seamlessly integrate quantum solutions with existing classical
optimization frameworks. We tackle this challenge by introducing the ProvideQ
toolbox, a software tool that enables users to easily adapt and configure
hybrid solvers via Meta-Solver strategies. A Meta-Solver strategy implements
decomposition techniques, which splits problems into classical and quantum
subroutines. The ProvideQ toolbox enables the interactive creation of such
decompositions via a Meta-Solver configuration tool. It combines
well-established classical optimization techniques with quantum circuits that
are seamlessly executable on multiple backends. This paper introduces the
technical details of the ProvideQ toolbox, explains its architecture, and
demonstrates possible applications for several real-world use cases. Our proof
of concept shows that Meta-Solver strategies already enable the application of
quantum subroutines today, however, more sophisticated hardware is required to
make their performance competitive.

### 8. [Quantum Executor: A Unified Interface for Quantum Computing](http://arxiv.org/pdf/2507.07597v1)

Authors: Giuseppe Bisicchia, Alessandro Bocci, Antonio Brogi

As quantum computing evolves from theoretical promise to practical
deployment, the demand for robust, portable, and scalable tools for quantum
software experimentation is growing. This paper introduces Quantum Executor, a
backend-agnostic execution engine designed to orchestrate quantum experiments
across heterogeneous platforms. Quantum Executor provides a declarative and
modular interface that decouples experiment design from backend execution,
enabling seamless interoperability and code reuse across diverse quantum and
classical resources. Key features include support for asynchronous and
distributed execution, customizable execution strategies and a unified API for
managing quantum experiments. We illustrate its applicability through two
life-like usage scenarios such as automated benchmarking and hybrid validation,
discussing its capacity to streamline quantum development. We conclude by
discussing current limitations and outlining a roadmap for future enhancements.

### Social and Information Networks

### 1. [Beyond Connectivity: Higher-Order Network Framework for Capturing Memory-Driven Mobility Dynamics](http://arxiv.org/pdf/2507.07727v1)

Authors: Chen Zhang, Jürgen Hackl

Understanding and predicting mobility dynamics in transportation networks is
critical for infrastructure planning, resilience analysis, and traffic
management. Traditional graph-based models typically assume memoryless
movement, limiting their ability to capture sequential dependencies inherent in
real-world mobility patterns. In this study, we introduce a novel higher-order
network framework for modeling memory-dependent dynamics in transportation
systems. By extending classical graph representations through higher-order
Markov chains and de Bruijn graph structures, our framework encodes the spatial
and temporal ordering of traversed paths, enabling the analysis of structurally
and functionally critical components with improved fidelity. We generalize key
network analytics, including betweenness centrality, PageRank, and next-step
prediction, to this higher-order setting and validate our approach on the Sioux
Falls transportation network using agent-based trajectory data generated with
MATSim. Experimental results demonstrate that higher-order models outperform
first-order baselines across multiple tasks, with the third-order model
achieving an optimal balance between predictive accuracy and model complexity.
These findings highlight the importance of incorporating memory effects into
network-based transportation analysis and offer a scalable, data-driven
methodology for capturing complex mobility behaviors in infrastructure systems.

### 2. [Conspiracy to Commit: Information Pollution, Artificial Intelligence, and Real-World Hate Crime](http://arxiv.org/pdf/2507.07884v1)

Authors: Alberto Aziani, Michael V. Lo Giudice, Ali Shadman Yazdi

Is demand for conspiracy theories online linked to real-world hate crimes? By
analyzing online search trends for 36 racially and politically-charged
conspiracy theories in Michigan (2015-2019), we employ a one-dimensional
convolutional neural network (1D-CNN) to predict hate crime occurrences
offline. A subset of theories including the Rothschilds family, Q-Anon, and The
Great Replacement improves prediction accuracy, with effects emerging two to
three weeks after fluctuations in searches. However, most theories showed no
clear connection to offline hate crimes. Aligning with neutralization and
differential association theories, our findings provide a partial empirical
link between specific racially charged conspiracy theories and real-world
violence. Just as well, this study underscores the potential for machine
learning to be used in identifying harmful online patterns and advancing social
science research.

### 3. [Efficient and Adaptive Estimation of Local Triadic Coefficients](http://arxiv.org/pdf/2507.07536v1)

Authors: Ilie Sarpe, Aristides Gionis

Characterizing graph properties is fundamental to the analysis and to our
understanding of real-world networked systems. The local clustering
coefficient, and the more recently introduced, local closure coefficient,
capture powerful properties that are essential in a large number of
applications, ranging from graph embeddings to graph partitioning. Such
coefficients capture the local density of the neighborhood of each node,
considering incident triadic structures and paths of length two. For this
reason, we refer to these coefficients collectively as local triadic
coefficients.
  In this work, we consider the novel problem of computing efficiently the
average of local triadic coefficients, over a given partition of the nodes of
the input graph into a set of disjoint buckets. The average local triadic
coefficients of the nodes in each bucket provide a better insight into the
interplay of graph structure and the properties of the nodes associated to each
bucket. Unfortunately, exact computation, which requires listing all triangles
in a graph, is infeasible for large networks. Hence, we focus on obtaining
highly-accurate probabilistic estimates.
  We develop Triad, an adaptive algorithm based on sampling, which can be used
to estimate the average local triadic coefficients for a partition of the nodes
into buckets. Triad is based on a new class of unbiased estimators, and
non-trivial bounds on its sample complexity, enabling the efficient computation
of highly accurate estimates. Finally, we show how Triad can be efficiently
used in practice on large networks, and we present a case study showing that
average local triadic coefficients can capture high-order patterns over
collaboration networks.

### 4. [Scalable Signed Exponential Random Graph Models under Local Dependence](http://arxiv.org/pdf/2507.07660v1)

Authors: Marc Schalberger, Cornelius Fritz

Traditional network analysis focuses on binary edges, while real-world
relationships are more nuanced, encompassing cooperation, neutrality, and
conflict. The rise of negative edges in social media discussions spurred
interest in analyzing signed interactions, especially in polarized debates.
However, the vast data generated by digital networks presents challenges for
traditional methods like Stochastic Block Models (SBM) and Exponential Family
Random Graph Models (ERGM), particularly due to the homogeneity assumption and
global dependence, which become increasingly unrealistic as network size grows.
To address this, we propose a novel method that combines the strengths of SBM
and ERGM while mitigating their weaknesses by incorporating local dependence
based on non-overlapping blocks. Our approach involves a two-step process:
first, decomposing the network into sub-networks using SBM approximation, and
then estimating parameters using ERGM methods. We validate our method on large
synthetic networks and apply it to a signed Wikipedia network of thousands of
editors. Through the use of local dependence, we find patterns consistent with
structural balance theory.

### Systems and Control

### 1. [Distributed and adaptive model predictive control for vehicle platoon systems under non-ideal communication](http://arxiv.org/pdf/2507.07429v1)

Authors: Qiaoni Han, Chengfei Xu, Zhiqiang Zuo

The uncertainty of wireless communication poses significant challenges to
platoon control performance. Aiming at alleviating the influence of non-ideal
communication on the platoon system, this paper proposes a distributed and
adaptive model predictive control (MPC) method. First of all, to deal with the
transmission uncertainty caused by non-ideal communication, compensated data
packets are customized for each vehicle. Then, an adaptive model predictive
control method is proposed to balance the system response speed and tracking
accuracy. Furthermore, to reduce the computational requirements of the vehicle
platoon system, a predictive time-domain update strategy suitable for non-ideal
communication was introduced. Finally, the sufficient conditions for ensuring
the feasibility of the MPC algorithm and the stability of the closed-loop
platoon control system are theoretically analyzed. The simulation results show
that the proposed method significantly reduces the computing resource
requirements for solving the optimization problem while ensuring satisfactory
system performance.

### 2. [Perspective Chapter: Insights from Kalman Filtering with Correlated Noises Recursive Least-Square Algorithm for State and Parameter Estimation](http://arxiv.org/pdf/2507.07588v1)

Authors: Abd El Mageed Hag Elamin Khalid

This article explores the estimation of parameters and states for linear
stochastic systems with deterministic control inputs. It introduces a novel
Kalman filtering approach called Kalman Filtering with Correlated Noises
Recursive Generalized Extended Least Squares (KF-CN-RGELS) algorithm, which
leverages the cross-correlation between process noise and measurement noise in
Kalman filtering cycles to jointly estimate both parameters and system states.
The study also investigates the theoretical implications of the correlation
coefficient on estimation accuracy through performance analysis involving
various correlation coefficients between process and measurement noises. The
research establishes a clear relationship: the accuracy of identified
parameters and states is directly proportional to positive correlation
coefficients. To validate the efficacy of this algorithm, a comprehensive
comparison is conducted among different algorithms, including the standard
Kalman filter algorithm and the augmented-state Kalman filter with correlated
noises algorithm. Theoretical findings are not only presented but also
exemplified through a numerical case study to provide valuable insights into
practical implications. This work contributes to enhancing estimation accuracy
in linear stochastic systems with deterministic control inputs, offering
valuable insights for control system design and state-space modeling.

### 3. [PhysioEdge: Multimodal Compressive Sensing Platform for Wearable Health Monitoring](http://arxiv.org/pdf/2507.07645v1)

Authors: Rens Baeyens, Dennis Laurijssen, Jan Steckel, Walter Daems

The integration of compressive sensing with real-time embedded systems opens
new possibilities for efficient, low-power biomedical signal acquisition. This
paper presents a custom hardware platform based on the RP2350 micro-controller,
tailored for synchronized multi-modal biomedical monitoring. The system is
capable of capturing cardiopulmonary sounds, along with biopotential signals
such as phonocardiography (PCG), electrocardiography (ECG) and electromyography
(EMG), photoplethysmography (PPG), and inertial measurement unit (IMU) data for
posture recognition. To ensure sample-accurate synchronization, a Sub-1GHz
radio system is used across multiple nodes. Wi-Fi and Bluetooth connectivity
enable centralized data aggregation. Experimental results demonstrate the
achieved decrease in power consumption when using compressive sensing,
efficient multi-node synchronization, and scalability for wireless biomedical
monitoring applications. The compact form factor and low-cost design make it
suitable for various medical applications, including remote healthcare and
long-term monitoring.

### 4. [Remote Renewable Energy Hubs: a Taxonomy](http://arxiv.org/pdf/2507.07659v1)

Authors: Victor Dachet, Antoine Dubois, Bardhyl Miftari, Raphaël Fonteneau, Damien Ernst

Serving the energy demand with renewable energy is hindered by its limited
availability near load centres (i.e. places where the energy demand is high).
To address this challenge, the concept of Remote Renewable Energy Hubs (RREH)
emerges as a promising solution. RREHs are energy hubs located in areas with
abundant renewable energy sources, such as sun in the Sahara Desert or wind in
Greenland. In these hubs, renewable energy sources are used to synthetise
energy molecules. To produce specific energy molecules, a tailored hub
configuration must be designed, which means choosing a set of technologies that
are interacting with each other as well as defining how they are integrated in
their local environment. The plurality of technologies that may be employed in
RREHs results in a large diversity of hubs. In order to characterize this
diversity, we propose in this paper a taxonomy for accurately defining these
hubs. This taxonomy allows to better describe and compare designs of hubs as
well as to identify new ones. Thus, it may guide policymakers and engineers in
hub design, contributing to cost efficiency and/or improving local integration.

### 5. [Ammonia, Methane, Hydrogen and Methanol Produced in Remote Renewable Energy Hubs: a Comparative Quantitative Analysis](http://arxiv.org/pdf/2507.07681v1)

Authors: Antoine Larbanois, Victor Dachet, Antoine Dubois, Raphaël Fonteneau, Damien Ernst

Remote renewable energy hubs (RREHs) for synthetic fuel production are
engineering systems harvesting renewable energy where it is particularly
abundant. They produce transportable synthetic fuels for export to distant load
centers. This article aims to evaluate the production costs of different energy
carriers, and includes a discussion on advantages and disadvantages in terms of
technical performance. To do so, we extend the study of Berger et al., (2021)
which focuses on methane (CH4) as energy carrier and introduce three new
carriers: ammonia (NH3), hydrogen (H2) and methanol (CH3OH). The four different
RREHs are located in the Algerian Sahara desert and must serve to the load
center, Belgium, a constant electro-fuel demand of 10 TWh per year. The
modelling and optimisation of these systems are performed using the modelling
language GBOML (Graph-Based Optimisation Modelling Language). Our findings
reveal that the three new RREHs, each with its respective carrier (ammonia,
hydrogen, and methanol), are all more cost-effective than the methane-based
system. Ammonia demonstrates the most favourable cost-to-energy exported ratio.

### 6. [Set-Based Control Barrier Functions and Safety Filters](http://arxiv.org/pdf/2507.07805v1)

Authors: Kim P. Wabersich, Felix Berkel, Felix Gruber, Sven Reimann

High performance and formal safety guarantees are common requirements for
industrial control applications. Control barrier function (CBF) methods provide
a systematic approach to the modularization of safety and performance. However,
the design of such CBFs can be challenging, which limits their applicability to
large-scale or data-driven systems. This paper introduces the concept of a
set-based CBF for linear systems with convex constraints. By leveraging control
invariant sets from reachability analysis and predictive control, the set-based
CBF is defined implicitly through the minimal scaling of such a set to contain
the current system state. This approach enables the development of implicit,
data-driven, and high-dimensional CBF representations. The paper demonstrates
the design of a safety filter using set-based CBFs, which is suitable for
real-time implementations and learning-based approximations to reduce online
computational demands. The effectiveness of the method is illustrated through
comprehensive simulations on a high-dimensional mass-spring-damper system and a
motion control task, and it is validated experimentally using an electric drive
application with short sampling times, highlighting its practical benefits for
safety-critical control.

### 7. [Identifying the Smallest Adversarial Load Perturbations that Render DC-OPF Infeasible](http://arxiv.org/pdf/2507.07850v1)

Authors: Samuel Chevalier, William A. Wheeler

What is the globally smallest load perturbation that renders DC-OPF
infeasible? Reliably identifying such "adversarial attack" perturbations has
useful applications in a variety of emerging grid-related contexts, including
machine learning performance verification, cybersecurity, and operational
robustness of power systems dominated by stochastic renewable energy resources.
In this paper, we formulate the inherently nonconvex adversarial attack problem
by applying a parameterized version of Farkas' lemma to a perturbed set of
DC-OPF equations. Since the resulting formulation is very hard to globally
optimize, we also propose a parameterized generation control policy which, when
applied to the primal DC-OPF problem, provides solvability guarantees.
Together, these nonconvex problems provide guaranteed upper and lower bounds on
adversarial attack size; by combining them into a single optimization problem,
we can efficiently "squeeze" these bounds towards a common global solution. We
apply these methods on a range of small- to medium-sized test cases from PGLib,
benchmarking our results against the best adversarial attack lower bounds
provided by Gurobi 12.0's spatial Branch and Bound solver.

### 8. [Data-driven Kinematic Modeling in Soft Robots: System Identification and Uncertainty Quantification](http://arxiv.org/pdf/2507.07370v1)

Authors: Zhanhong Jiang, Dylan Shah, Hsin-Jung Yang, Soumik Sarkar

Precise kinematic modeling is critical in calibration and controller design
for soft robots, yet remains a challenging issue due to their highly nonlinear
and complex behaviors. To tackle the issue, numerous data-driven machine
learning approaches have been proposed for modeling nonlinear dynamics.
However, these models suffer from prediction uncertainty that can negatively
affect modeling accuracy, and uncertainty quantification for kinematic modeling
in soft robots is underexplored. In this work, using limited simulation and
real-world data, we first investigate multiple linear and nonlinear machine
learning models commonly used for kinematic modeling of soft robots. The
results reveal that nonlinear ensemble methods exhibit the most robust
generalization performance. We then develop a conformal kinematic modeling
framework for soft robots by utilizing split conformal prediction to quantify
predictive position uncertainty, ensuring distribution-free prediction
intervals with a theoretical guarantee.

### 9. [Demonstration of TFTs 3D Monolithically Integrated on GaN HEMTs using Cascode Configuration with High Breakdown Voltage (>1900V)](http://arxiv.org/pdf/2507.07512v1)

Authors: Tian-Li Wu, Hsin-Jou Ho, Chia-Wei Liu, Yi-Chen Chen

This study demonstrates 3D monolithic integration of amorphous
indium-gallium-zinc oxide (a-IGZO) thin-film transistors (TFTs) on Gallium
Nitride (GaN) high electron mobility transistors (HEMTs) in a cascode
configuration, achieving high breakdown voltage capabilities exceeding 1900 V.
Two device configurations, differing in a-IGZO channel thickness (30 nm / 10
nm), are fabricated and evaluated. Sample B, with a 10 nm a-IGZO channel,
demonstrates superior electrical performance, including a high ON/OFF current
ratio (~10^7), low subthreshold swing (SS), and a high breakdown voltage
exceeding 1900 V comparable to standalone GaN power HEMTs. The results
highlight the feasibility and potential of 3D integrated TFT on GaN power
HEMTs, paving the way for new opportunities for the TFTs for high voltage
applications.

### 10. [Real-Time Decorrelation-Based Anomaly Detection for Multivariate Time Series](http://arxiv.org/pdf/2507.07559v1)

Authors: Amirhossein Sadough, Mahyar Shahsavari, Mark Wijtvliet, Marcel van Gerven

Anomaly detection (AD) plays a vital role across a wide range of real-world
domains by identifying data instances that deviate from expected patterns,
potentially signaling critical events such as system failures, fraudulent
activities, or rare medical conditions. The demand for real-time AD has surged
with the rise of the (Industrial) Internet of Things, where massive volumes of
multivariate sensor data must be processed instantaneously. Real-time AD
requires methods that not only handle high-dimensional streaming data but also
operate in a single-pass manner, without the burden of storing historical
instances, thereby ensuring minimal memory usage and fast decision-making. We
propose DAD, a novel real-time decorrelation-based anomaly detection method for
multivariate time series, based on an online decorrelation learning approach.
Unlike traditional proximity-based or reconstruction-based detectors that
process entire data or windowed instances, DAD dynamically learns and monitors
the correlation structure of data sample by sample in a single pass, enabling
efficient and effective detection. To support more realistic benchmarking
practices, we also introduce a practical hyperparameter tuning strategy
tailored for real-time anomaly detection scenarios. Extensive experiments on
widely used benchmark datasets demonstrate that DAD achieves the most
consistent and superior performance across diverse anomaly types compared to
state-of-the-art methods. Crucially, its robustness to increasing
dimensionality makes it particularly well-suited for real-time,
high-dimensional data streams. Ultimately, DAD not only strikes an optimal
balance between detection efficacy and computational efficiency but also sets a
new standard for real-time, memory-constrained anomaly detection.

### Machine Learning (Statistics Category)

### 1. [Hess-MC2: Sequential Monte Carlo Squared using Hessian Information and Second Order Proposals](http://arxiv.org/pdf/2507.07461v1)

Authors: Joshua Murphy, Conor Rosato, Andrew Millard, Lee Devlin, Paul Horridge, Simon Maskell

When performing Bayesian inference using Sequential Monte Carlo (SMC)
methods, two considerations arise: the accuracy of the posterior approximation
and computational efficiency. To address computational demands, Sequential
Monte Carlo Squared (SMC$^2$) is well-suited for high-performance computing
(HPC) environments. The design of the proposal distribution within SMC$^2$ can
improve accuracy and exploration of the posterior as poor proposals may lead to
high variance in importance weights and particle degeneracy. The
Metropolis-Adjusted Langevin Algorithm (MALA) uses gradient information so that
particles preferentially explore regions of higher probability. In this paper,
we extend this idea by incorporating second-order information, specifically the
Hessian of the log-target. While second-order proposals have been explored
previously in particle Markov Chain Monte Carlo (p-MCMC) methods, we are the
first to introduce them within the SMC$^2$ framework. Second-order proposals
not only use the gradient (first-order derivative), but also the curvature
(second-order derivative) of the target distribution. Experimental results on
synthetic models highlight the benefits of our approach in terms of step-size
selection and posterior approximation accuracy when compared to other
proposals.

### 2. [A Unified Empirical Risk Minimization Framework for Flexible N-Tuples Weak Supervision](http://arxiv.org/pdf/2507.07771v1)

Authors: Shuying Huang, Junpeng Li, Changchun Hua, Yana Yang

To alleviate the annotation burden in supervised learning, N-tuples learning
has recently emerged as a powerful weakly-supervised method. While existing
N-tuples learning approaches extend pairwise learning to higher-order
comparisons and accommodate various real-world scenarios, they often rely on
task-specific designs and lack a unified theoretical foundation. In this paper,
we propose a general N-tuples learning framework based on empirical risk
minimization, which systematically integrates pointwise unlabeled data to
enhance learning performance. This paper first unifies the data generation
processes of N-tuples and pointwise unlabeled data under a shared probabilistic
formulation. Based on this unified view, we derive an unbiased empirical risk
estimator that generalizes a broad class of existing N-tuples models. We
further establish a generalization error bound for theoretical support. To
demonstrate the flexibility of the framework, we instantiate it in four
representative weakly supervised scenarios, each recoverable as a special case
of our general model. Additionally, to address overfitting issues arising from
negative risk terms, we adopt correction functions to adjust the empirical
risk. Extensive experiments on benchmark datasets validate the effectiveness of
the proposed framework and demonstrate that leveraging pointwise unlabeled data
consistently improves generalization across various N-tuples learning tasks.

### 3. [An Empirical Bernstein Inequality for Dependent Data in Hilbert Spaces and Applications](http://arxiv.org/pdf/2507.07826v1)

Authors: Erfan Mirzaei, Andreas Maurer, Vladimir R. Kostic, Massimiliano Pontil

Learning from non-independent and non-identically distributed data poses a
persistent challenge in statistical learning. In this study, we introduce
data-dependent Bernstein inequalities tailored for vector-valued processes in
Hilbert space. Our inequalities apply to both stationary and non-stationary
processes and exploit the potential rapid decay of correlations between
temporally separated variables to improve estimation. We demonstrate the
utility of these bounds by applying them to covariance operator estimation in
the Hilbert-Schmidt norm and to operator learning in dynamical systems,
achieving novel risk bounds. Finally, we perform numerical experiments to
illustrate the practical implications of these bounds in both contexts.

### 4. [Pre-Trained AI Model Assisted Online Decision-Making under Missing Covariates: A Theoretical Perspective](http://arxiv.org/pdf/2507.07852v1)

Authors: Haichen Hu, David Simchi-Levi

We study a sequential contextual decision-making problem in which certain
covariates are missing but can be imputed using a pre-trained AI model. From a
theoretical perspective, we analyze how the presence of such a model influences
the regret of the decision-making process. We introduce a novel notion called
"model elasticity", which quantifies the sensitivity of the reward function to
the discrepancy between the true covariate and its imputed counterpart. This
concept provides a unified way to characterize the regret incurred due to model
imputation, regardless of the underlying missingness mechanism. More
surprisingly, we show that under the missing at random (MAR) setting, it is
possible to sequentially calibrate the pre-trained model using tools from
orthogonal statistical learning and doubly robust regression. This calibration
significantly improves the quality of the imputed covariates, leading to much
better regret guarantees. Our analysis highlights the practical value of having
an accurate pre-trained model in sequential decision-making tasks and suggests
that model elasticity may serve as a fundamental metric for understanding and
improving the integration of pre-trained models in a wide range of data-driven
decision-making problems.

### 5. [Late Fusion Multi-task Learning for Semiparametric Inference with Nuisance Parameters](http://arxiv.org/pdf/2507.07941v1)

Authors: Sohom Bhattacharya, Yongzhuo Chen, Muxuan Liang

In the age of large and heterogeneous datasets, the integration of
information from diverse sources is essential to improve parameter estimation.
Multi-task learning offers a powerful approach by enabling simultaneous
learning across related tasks. In this work, we introduce a late fusion
framework for multi-task learning with semiparametric models that involve
infinite-dimensional nuisance parameters, focusing on applications such as
heterogeneous treatment effect estimation across multiple data sources,
including electronic health records from different hospitals or clinical trial
data. Our framework is two-step: first, initial double machine-learning
estimators are obtained through individual task learning; second, these
estimators are adaptively aggregated to exploit task similarities while
remaining robust to task-specific differences. In particular, the framework
avoids individual level data sharing, preserving privacy. Additionally, we
propose a novel multi-task learning method for nuisance parameter estimation,
which further enhances parameter estimation when nuisance parameters exhibit
similarity across tasks. We establish theoretical guarantees for the method,
demonstrating faster convergence rates compared to individual task learning
when tasks share similar parametric components. Extensive simulations and real
data applications complement the theoretical findings of our work while
highlight the effectiveness of our framework even in moderate sample sizes.

### 6. [Prospective Learning in Retrospect](http://arxiv.org/pdf/2507.07965v1)

Authors: Yuxin Bai, Cecelia Shuai, Ashwin De Silva, Siyu Yu, Pratik Chaudhari, Joshua T. Vogelstein

In most real-world applications of artificial intelligence, the distributions
of the data and the goals of the learners tend to change over time. The
Probably Approximately Correct (PAC) learning framework, which underpins most
machine learning algorithms, fails to account for dynamic data distributions
and evolving objectives, often resulting in suboptimal performance. Prospective
learning is a recently introduced mathematical framework that overcomes some of
these limitations. We build on this framework to present preliminary results
that improve the algorithm and numerical results, and extend prospective
learning to sequential decision-making scenarios, specifically foraging. Code
is available at: https://github.com/neurodata/prolearn2.

### 7. [Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning](http://arxiv.org/pdf/2507.07359v1)

Authors: Zheyu Zhang, Jiayuan Dong, Jie Liu, Xun Huan

We present GO-CBED, a goal-oriented Bayesian framework for sequential causal
experimental design. Unlike conventional approaches that select interventions
aimed at inferring the full causal model, GO-CBED directly maximizes the
expected information gain (EIG) on user-specified causal quantities of
interest, enabling more targeted and efficient experimentation. The framework
is both non-myopic, optimizing over entire intervention sequences, and
goal-oriented, targeting only model aspects relevant to the causal query. To
address the intractability of exact EIG computation, we introduce a variational
lower bound estimator, optimized jointly through a transformer-based policy
network and normalizing flow-based variational posteriors. The resulting policy
enables real-time decision-making via an amortized network. We demonstrate that
GO-CBED consistently outperforms existing baselines across various causal
reasoning and discovery tasks-including synthetic structural causal models and
semi-synthetic gene regulatory networks-particularly in settings with limited
experimental budgets and complex causal mechanisms. Our results highlight the
benefits of aligning experimental design objectives with specific research
goals and of forward-looking sequential planning.

### 8. [Feature-free regression kriging](http://arxiv.org/pdf/2507.07382v1)

Authors: Peng Luo, Yilong Wu, Yongze Song

Spatial interpolation is a crucial task in geography. As perhaps the most
widely used interpolation methods, geostatistical models -- such as Ordinary
Kriging (OK) -- assume spatial stationarity, which makes it difficult to
capture the nonstationary characteristics of geographic variables. A common
solution is trend surface modeling (e.g., Regression Kriging, RK), which relies
on external explanatory variables to model the trend and then applies
geostatistical interpolation to the residuals. However, this approach requires
high-quality and readily available explanatory variables, which are often
lacking in many spatial interpolation scenarios -- such as estimating heavy
metal concentrations underground. This study proposes a Feature-Free Regression
Kriging (FFRK) method, which automatically extracts geospatial features --
including local dependence, local heterogeneity, and geosimilarity -- to
construct a regression-based trend surface without requiring external
explanatory variables. We conducted experiments on the spatial distribution
prediction of three heavy metals in a mining area in Australia. In comparison
with 17 classical interpolation methods, the results indicate that FFRK, which
does not incorporate any explanatory variables and relies solely on extracted
geospatial features, consistently outperforms both conventional Kriging
techniques and machine learning models that depend on explanatory variables.
This approach effectively addresses spatial nonstationarity while reducing the
cost of acquiring explanatory variables, improving both prediction accuracy and
generalization ability. This finding suggests that an accurate characterization
of geospatial features based on domain knowledge can significantly enhance
spatial prediction performance -- potentially yielding greater improvements
than merely adopting more advanced statistical models.

### 9. [Galerkin-ARIMA: A Two-Stage Polynomial Regression Framework for Fast Rolling One-Step-Ahead Forecasting](http://arxiv.org/pdf/2507.07469v1)

Authors: Haojie Liu, Zihan Lin

Time-series models like ARIMA remain widely used for forecasting but limited
to linear assumptions and high computational cost in large and complex
datasets. We propose Galerkin-ARIMA that generalizes the AR component of ARIMA
and replace it with a flexible spline-based function estimated by Galerkin
projection. This enables the model to capture nonlinear dependencies in lagged
values and retain the MA component and Gaussian noise assumption. We derive a
closed-form OLS estimator for the Galerkin coefficients and show the model is
asymptotically unbiased and consistent under standard conditions. Our method
bridges classical time-series modeling and nonparametric regression, which
offering improved forecasting performance and computational efficiency.

### 10. [Reinforcement Learning with Action Chunking](http://arxiv.org/pdf/2507.07969v1)

Authors: Qiyang Li, Zhiyuan Zhou, Sergey Levine

We present Q-chunking, a simple yet effective recipe for improving
reinforcement learning (RL) algorithms for long-horizon, sparse-reward tasks.
Our recipe is designed for the offline-to-online RL setting, where the goal is
to leverage an offline prior dataset to maximize the sample-efficiency of
online learning. Effective exploration and sample-efficient learning remain
central challenges in this setting, as it is not obvious how the offline data
should be utilized to acquire a good exploratory policy. Our key insight is
that action chunking, a technique popularized in imitation learning where
sequences of future actions are predicted rather than a single action at each
timestep, can be applied to temporal difference (TD)-based RL methods to
mitigate the exploration challenge. Q-chunking adopts action chunking by
directly running RL in a 'chunked' action space, enabling the agent to (1)
leverage temporally consistent behaviors from offline data for more effective
online exploration and (2) use unbiased $n$-step backups for more stable and
efficient TD learning. Our experimental results demonstrate that Q-chunking
exhibits strong offline performance and online sample efficiency, outperforming
prior best offline-to-online methods on a range of long-horizon, sparse-reward
manipulation tasks.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-11 PST.

### 1. [Temporal evidence fusion evaluation method considering time sequence variation trend](https://www.nature.com/articles/s41598-025-10687-7)

Authors: Sunan Zhang et al.

### 2. [A neuromorphic processor with on-chip learning for beyond-CMOS device integration](https://www.nature.com/articles/s41467-025-61576-6)

Authors: Hugh Greatorex et al.

### 3. [Vehicle detection in drone aerial views based on lightweight OSD-YOLOv10](https://www.nature.com/articles/s41598-025-09825-y)

Authors: Yang Zhang et al.

### 4. [A perspective for adapting generalist AI to specialized medical AI applications and their challenges](https://www.nature.com/articles/s41746-025-01789-7)

Authors: Zifeng Wang et al.

### 5. [A meta fusion model combining geographic data and twitter sentiment analysis for predicting accident severity](https://www.nature.com/articles/s41598-025-91484-0)

Authors: Areeba Naseem Khan et al.

### 6. [ST-CFI: Swin Transformer with convolutional feature interactions for identifying plant diseases](https://www.nature.com/articles/s41598-025-08673-0)

Authors: Sheng Yu et al.

### 7. [Exploiting Gaussian based effective receptive fields for object detection](https://www.nature.com/articles/s41598-025-10548-3)

Authors: Xiaoxia Qi et al.

### 8. [A hybrid YOLO-UNet3D framework for automated protein particle annotation in Cryo-ET images](https://www.nature.com/articles/s41598-025-09522-w)

Authors: Ziyang Liu et al.

### 9. [Long-distance target localization optimization algorithm based on single robot moving path planning](https://www.nature.com/articles/s41598-025-09428-7)

Authors: Yourong Chen et al.

### 10. [Optimizing on-demand food delivery with BDI-based multi-agent systems and Monte Carlo tree search scheduling](https://www.nature.com/articles/s41598-025-10371-w)

Authors: Li Liu et al.

### 11. [An investigation of simple neural network models using smartphone signals for recognition of manual industrial tasks](https://www.nature.com/articles/s41598-025-06726-y)

Authors: Tacjana Niksa‑Rynkiewicz et al.

### 12. [Mobile malware detection method using improved GhostNetV2 with image enhancement technique](https://www.nature.com/articles/s41598-025-07742-8)

Authors: Yao Du et al.

