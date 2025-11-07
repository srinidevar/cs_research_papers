# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-11-06 17:00:25.985366 PST.

### Artificial Intelligence

### 1. [Large language models require a new form of oversight: capability-based monitoring](http://arxiv.org/pdf/2511.03106v1)

Authors: Katherine C. Kellogg, Bingyang Ye, Yifan Hu, Guergana K. Savova, Byron Wallace, Danielle S. Bitterman

The rapid adoption of large language models (LLMs) in healthcare has been
accompanied by scrutiny of their oversight. Existing monitoring approaches,
inherited from traditional machine learning (ML), are task-based and founded on
assumed performance degradation arising from dataset drift. In contrast, with
LLMs, inevitable model degradation due to changes in populations compared to
the training dataset cannot be assumed, because LLMs were not trained for any
specific task in any given population. We therefore propose a new organizing
principle guiding generalist LLM monitoring that is scalable and grounded in
how these models are developed and used in practice: capability-based
monitoring. Capability-based monitoring is motivated by the fact that LLMs are
generalist systems whose overlapping internal capabilities are reused across
numerous downstream tasks. Instead of evaluating each downstream task
independently, this approach organizes monitoring around shared model
capabilities, such as summarization, reasoning, translation, or safety
guardrails, in order to enable cross-task detection of systemic weaknesses,
long-tail errors, and emergent behaviors that task-based monitoring may miss.
We describe considerations for developers, organizational leaders, and
professional societies for implementing a capability-based monitoring approach.
Ultimately, capability-based monitoring will provide a scalable foundation for
safe, adaptive, and collaborative monitoring of LLMs and future generalist
artificial intelligence models in healthcare.

### 2. [miniF2F-Lean Revisited: Reviewing Limitations and Charting a Path Forward](http://arxiv.org/pdf/2511.03108v1)

Authors: Azim Ospanov, Farzan Farnia, Roozbeh Yousefzadeh

We perform a thorough analysis of the formal and informal statements in the
miniF2F benchmark from the perspective of an AI system that is tasked to
participate in a math Olympiad consisting of the problems in miniF2F. In such
setting, the model has to read and comprehend the problems in natural language,
formalize them in Lean language, then proceed with proving the problems, and it
will get credit for each problem if the formal proof corresponds to the
original informal statement presented to the model. Our evaluation results
reveal that the best accuracy of such pipeline can be about 36% using the SoTA
models in the literature, considerably lower than the individual SoTA
accuracies, 97% and 69% reported in the autoformalization and theorem proving
literature. Analyzing the failure modes, we trace back a considerable portion
of this drop to discrepancies between the formal and informal statements for
more than half of the problems in miniF2F. We proceed with correcting all the
errors, discrepancies and simplifications in formal and informal statements,
and present the miniF2F-v2 with fully verified formal and informal statements
and proofs. Evaluating the full theorem proving pipeline on miniF2F-v2 leads to
the best accuracy of 70%, a significant improvement from the 40% on the
original miniF2F, yet indicating considerable misalignment between the
autoformalization models and theorem provers. Our deep analysis suggests that a
higher quality benchmark can help the community better evaluate progress in the
field of formal reasoning and also better diagnose the failure and success
modes of autoformalization and theorem proving models. Our dataset is available
at https://github.com/roozbeh-yz/miniF2F_v2.

### 3. [Using Multi-modal Large Language Model to Boost Fireworks Algorithm's Ability in Settling Challenging Optimization Tasks](http://arxiv.org/pdf/2511.03137v1)

Authors: Shipeng Cen, Ying Tan

As optimization problems grow increasingly complex and diverse, advancements
in optimization techniques and paradigm innovations hold significant
importance. The challenges posed by optimization problems are primarily
manifested in their non-convexity, high-dimensionality, black-box nature, and
other unfavorable characteristics. Traditional zero-order or first-order
methods, which are often characterized by low efficiency, inaccurate gradient
information, and insufficient utilization of optimization information, are
ill-equipped to address these challenges effectively. In recent years, the
rapid development of large language models (LLM) has led to substantial
improvements in their language understanding and code generation capabilities.
Consequently, the design of optimization algorithms leveraging large language
models has garnered increasing attention from researchers. In this study, we
choose the fireworks algorithm(FWA) as the basic optimizer and propose a novel
approach to assist the design of the FWA by incorporating multi-modal large
language model(MLLM). To put it simply, we propose the concept of Critical
Part(CP), which extends FWA to complex high-dimensional tasks, and further
utilizes the information in the optimization process with the help of the
multi-modal characteristics of large language models. We focus on two specific
tasks: the \textit{traveling salesman problem }(TSP) and \textit{electronic
design automation problem} (EDA). The experimental results show that FWAs
generated under our new framework have achieved or surpassed SOTA results on
many problem instances.

### 4. [A Proprietary Model-Based Safety Response Framework for AI Agents](http://arxiv.org/pdf/2511.03138v1)

Authors: Qi Li, Jianjun Xu, Pingtao Wei, Jiu Li, Peiqiang Zhao, Jiwei Shi, Xuan Zhang, Yanhui Yang, Xiaodong Hui, Peng Xu, Wenqin Shao

With the widespread application of Large Language Models (LLMs), their
associated security issues have become increasingly prominent, severely
constraining their trustworthy deployment in critical domains. This paper
proposes a novel safety response framework designed to systematically safeguard
LLMs at both the input and output levels. At the input level, the framework
employs a supervised fine-tuning-based safety classification model. Through a
fine-grained four-tier taxonomy (Safe, Unsafe, Conditionally Safe, Focused
Attention), it performs precise risk identification and differentiated handling
of user queries, significantly enhancing risk coverage and business scenario
adaptability, and achieving a risk recall rate of 99.3%. At the output level,
the framework integrates Retrieval-Augmented Generation (RAG) with a
specifically fine-tuned interpretation model, ensuring all responses are
grounded in a real-time, trustworthy knowledge base. This approach eliminates
information fabrication and enables result traceability. Experimental results
demonstrate that our proposed safety control model achieves a significantly
higher safety score on public safety evaluation benchmarks compared to the
baseline model, TinyR1-Safety-8B. Furthermore, on our proprietary high-risk
test set, the framework's components attained a perfect 100% safety score,
validating their exceptional protective capabilities in complex risk scenarios.
This research provides an effective engineering pathway for building
high-security, high-trust LLM applications.

### 5. [Uncovering Bugs in Formal Explainers: A Case Study with PyXAI](http://arxiv.org/pdf/2511.03169v1)

Authors: Xuanxiang Huang, Yacine Izza, Alexey Ignatiev, Joao Marques-Silva

Formal explainable artificial intelligence (XAI) offers unique theoretical
guarantees of rigor when compared to other non-formal methods of
explainability. However, little attention has been given to the validation of
practical implementations of formal explainers. This paper develops a novel
methodology for validating formal explainers and reports on the assessment of
the publicly available formal explainer PyXAI. The paper documents the
existence of incorrect explanations computed by PyXAI on most of the datasets
analyzed in the experiments, thereby confirming the importance of the proposed
novel methodology for the validation of formal explainers.

### 6. [Adobe Summit Concierge Evaluation with Human in the Loop](http://arxiv.org/pdf/2511.03186v1)

Authors: Yiru Chen, Sally Fang, Sai Sree Harsha, Dan Luo, Vaishnavi Muppala, Fei Wu, Shun Jiang, Kun Qian, Yunyao Li

Generative AI assistants offer significant potential to enhance productivity,
streamline information access, and improve user experience in enterprise
contexts. In this work, we present Summit Concierge, a domain-specific AI
assistant developed for Adobe Summit. The assistant handles a wide range of
event-related queries and operates under real-world constraints such as data
sparsity, quality assurance, and rapid deployment. To address these challenges,
we adopt a human-in-the-loop development workflow that combines prompt
engineering, retrieval grounding, and lightweight human validation. We describe
the system architecture, development process, and real-world deployment
outcomes. Our experience shows that agile, feedback-driven development enables
scalable and reliable AI assistants, even in cold-start scenarios.

### 7. [From Five Dimensions to Many: Large Language Models as Precise and Interpretable Psychological Profilers](http://arxiv.org/pdf/2511.03235v1)

Authors: Yi-Fei Liu, Yi-Long Lu, Di He, Hang Zhang

Psychological constructs within individuals are widely believed to be
interconnected. We investigated whether and how Large Language Models (LLMs)
can model the correlational structure of human psychological traits from
minimal quantitative inputs. We prompted various LLMs with Big Five Personality
Scale responses from 816 human individuals to role-play their responses on nine
other psychological scales. LLMs demonstrated remarkable accuracy in capturing
human psychological structure, with the inter-scale correlation patterns from
LLM-generated responses strongly aligning with those from human data $(R^2 >
0.89)$. This zero-shot performance substantially exceeded predictions based on
semantic similarity and approached the accuracy of machine learning algorithms
trained directly on the dataset. Analysis of reasoning traces revealed that
LLMs use a systematic two-stage process: First, they transform raw Big Five
responses into natural language personality summaries through information
selection and compression, analogous to generating sufficient statistics.
Second, they generate target scale responses based on reasoning from these
summaries. For information selection, LLMs identify the same key personality
factors as trained algorithms, though they fail to differentiate item
importance within factors. The resulting compressed summaries are not merely
redundant representations but capture synergistic information--adding them to
original scores enhances prediction alignment, suggesting they encode emergent,
second-order patterns of trait interplay. Our findings demonstrate that LLMs
can precisely predict individual participants' psychological traits from
minimal data through a process of abstraction and reasoning, offering both a
powerful tool for psychological simulation and valuable insights into their
emergent reasoning capabilities.

### 8. [Explaining Decisions in ML Models: a Parameterized Complexity Analysis (Part I)](http://arxiv.org/pdf/2511.03545v1)

Authors: Sebastian Ordyniak, Giacomo Paesani, Mateusz Rychlicki, Stefan Szeider

This paper presents a comprehensive theoretical investigation into the
parameterized complexity of explanation problems in various machine learning
(ML) models. Contrary to the prevalent black-box perception, our study focuses
on models with transparent internal mechanisms. We address two principal types
of explanation problems: abductive and contrastive, both in their local and
global variants. Our analysis encompasses diverse ML models, including Decision
Trees, Decision Sets, Decision Lists, Boolean Circuits, and ensembles thereof,
each offering unique explanatory challenges. This research fills a significant
gap in explainable AI (XAI) by providing a foundational understanding of the
complexities of generating explanations for these models. This work provides
insights vital for further research in the domain of XAI, contributing to the
broader discourse on the necessity of transparency and accountability in AI
systems.

### 9. [Sparse, self-organizing ensembles of local kernels detect rare statistical anomalies](http://arxiv.org/pdf/2511.03095v1)

Authors: Gaia Grosso, Sai Sumedh R. Hindupur, Thomas Fel, Samuel Bright-Thonney, Philip Harris, Demba Ba

Modern artificial intelligence has revolutionized our ability to extract rich
and versatile data representations across scientific disciplines. Yet, the
statistical properties of these representations remain poorly controlled,
causing misspecified anomaly detection (AD) methods to falter. Weak or rare
signals can remain hidden within the apparent regularity of normal data,
creating a gap in our ability to detect and interpret anomalies. We examine
this gap and identify a set of structural desiderata for detection methods
operating under minimal prior information: sparsity, to enforce parsimony;
locality, to preserve geometric sensitivity; and competition, to promote
efficient allocation of model capacity. These principles define a class of
self-organizing local kernels that adaptively partition the representation
space around regions of statistical imbalance. As an instantiation of these
principles, we introduce SparKer, a sparse ensemble of Gaussian kernels trained
within a semi-supervised Neyman--Pearson framework to locally model the
likelihood ratio between a sample that may contain anomalies and a nominal,
anomaly-free reference. We provide theoretical insights into the mechanisms
that drive detection and self-organization in the proposed model, and
demonstrate the effectiveness of this approach on realistic high-dimensional
problems of scientific discovery, open-world novelty detection, intrusion
detection, and generative-model validation. Our applications span both the
natural- and computer-science domains. We demonstrate that ensembles containing
only a handful of kernels can identify statistically significant anomalous
locations within representation spaces of thousands of dimensions, underscoring
both the interpretability, efficiency and scalability of the proposed approach.

### 10. [CARMA: Comprehensive Automatically-annotated Reddit Mental Health Dataset for Arabic](http://arxiv.org/pdf/2511.03102v1)

Authors: Saad Mankarious, Ayah Zirikly

Mental health disorders affect millions worldwide, yet early detection
remains a major challenge, particularly for Arabic-speaking populations where
resources are limited and mental health discourse is often discouraged due to
cultural stigma. While substantial research has focused on English-language
mental health detection, Arabic remains significantly underexplored, partly due
to the scarcity of annotated datasets. We present CARMA, the first
automatically annotated large-scale dataset of Arabic Reddit posts. The dataset
encompasses six mental health conditions, such as Anxiety, Autism, and
Depression, and a control group. CARMA surpasses existing resources in both
scale and diversity. We conduct qualitative and quantitative analyses of
lexical and semantic differences between users, providing insights into the
linguistic markers of specific mental health conditions. To demonstrate the
dataset's potential for further mental health analysis, we perform
classification experiments using a range of models, from shallow classifiers to
large language models. Our results highlight the promise of advancing mental
health detection in underrepresented languages such as Arabic.

### Hardware Architecture

### 1. [LogicSparse: Enabling Engine-Free Unstructured Sparsity for Quantised Deep-learning Accelerators](http://arxiv.org/pdf/2511.03079v1)

Authors: Changhong Li, Biswajit Basu, Shreejith Shanker

FPGAs have been shown to be a promising platform for deploying Quantised
Neural Networks (QNNs) with high-speed, low-latency, and energy-efficient
inference. However, the complexity of modern deep-learning models limits the
performance on resource-constrained edge devices. While quantisation and
pruning alleviate these challenges, unstructured sparsity remains
underexploited due to irregular memory access. This work introduces a framework
that embeds unstructured sparsity into dataflow accelerators, eliminating the
need for dedicated sparse engines and preserving parallelism. A hardware-aware
pruning strategy is introduced to improve efficiency and design flow further.
On LeNet-5, the framework attains 51.6 x compression and 1.23 x throughput
improvement using only 5.12% of LUTs, effectively exploiting unstructured
sparsity for QNN acceleration.

### 2. [An Event-Driven Spiking Compute-In-Memory Macro based on SOT-MRAM](http://arxiv.org/pdf/2511.03203v1)

Authors: Deyang Yu, Chenchen Liu, Chuanjie Zhang, Xiao Fang, Weisheng Zhao

The application of Magnetic Random-Access Memory (MRAM) in
computing-in-memory (CIM) has gained significant attention. However, existing
designs often suffer from high energy consumption due to their reliance on
complex analog circuits for computation. In this work, we present a Spin-Orbit-
Torque MRAM(SOT-MRAM)-based CIM macro that employs an event-driven spiking
processing for high energy efficiency. The SOT-MRAM crossbar adopts a hybrid
series-parallel cell structure to efficiently support matrix-vector
multiplication (MVM). Signal information is (en) decoded as spikes using
lightweight circuits, eliminating the need for conventional area- and
powerintensive analog circuits. The SOT-MRAM macro is designed and evaluated in
28nm technology, and experimental results show that it achieves a peak energy
efficiency of 243.6 TOPS/W, significantly outperforming existing designs.

### 3. [Design and Optimization of Mixed-Kernel Mixed-Signal SVMs for Flexible Electronics](http://arxiv.org/pdf/2511.03427v1)

Authors: Florentia Afentaki, Maha Shatta, Konstantinos Balaskas, Georgios Panagopoulos, Georgios Zervakis, Mehdi B. Tahoori

Flexible Electronics (FE) have emerged as a promising alternative to
silicon-based technologies, offering on-demand low-cost fabrication,
conformality, and sustainability. However, their large feature sizes severely
limit integration density, imposing strict area and power constraints, thus
prohibiting the realization of Machine Learning (ML) circuits, which can
significantly enhance the capabilities of relevant near-sensor applications.
Support Vector Machines (SVMs) offer high accuracy in such applications at
relatively low computational complexity, satisfying FE technologies'
constraints. Existing SVM designs rely solely on linear or Radial Basis
Function (RBF) kernels, forcing a trade-off between hardware costs and
accuracy. Linear kernels, implemented digitally, minimize overhead but
sacrifice performance, while the more accurate RBF kernels are prohibitively
large in digital, and their analog realization contains inherent functional
approximation. In this work, we propose the first mixed-kernel and mixed-signal
SVM design in FE, which unifies the advantages of both implementations and
balances the cost/accuracy trade-off. To that end, we introduce a
co-optimization approach that trains our mixed-kernel SVMs and maps binary SVM
classifiers to the appropriate kernel (linear/RBF) and domain (digital/analog),
aiming to maximize accuracy whilst reducing the number of costly RBF
classifiers. Our designs deliver 7.7% higher accuracy than state-of-the-art
single-kernel linear SVMs, and reduce area and power by 108x and 17x on average
compared to digital RBF implementations.

### 4. [LaMoS: Enabling Efficient Large Number Modular Multiplication through SRAM-based CiM Acceleration](http://arxiv.org/pdf/2511.03341v1)

Authors: Haomin Li, Fangxin Liu, Chenyang Guan, Zongwu Wang, Li Jiang, Haibing Guan

Barrett's algorithm is one of the most widely used methods for performing
modular multiplication, a critical nonlinear operation in modern privacy
computing techniques such as homomorphic encryption (HE) and zero-knowledge
proofs (ZKP). Since modular multiplication dominates the processing time in
these applications, computational complexity and memory limitations
significantly impact performance. Computing-in-Memory (CiM) is a promising
approach to tackle this problem. However, existing schemes currently suffer
from two main problems: 1) Most works focus on low bit-width modular
multiplication, which is inadequate for mainstream cryptographic algorithms
such as elliptic curve cryptography (ECC) and the RSA algorithm, both of which
require high bit-width operations; 2) Recent efforts targeting large number
modular multiplication rely on inefficient in-memory logic operations,
resulting in high scaling costs for larger bit-widths and increased latency. To
address these issues, we propose LaMoS, an efficient SRAM-based CiM design for
large-number modular multiplication, offering high scalability and area
efficiency. First, we analyze the Barrett's modular multiplication method and
map the workload onto SRAM CiM macros for high bit-width cases. Additionally,
we develop an efficient CiM architecture and dataflow to optimize large-number
modular multiplication. Finally, we refine the mapping scheme for better
scalability in high bit-width scenarios using workload grouping. Experimental
results show that LaMoS achieves a $7.02\times$ speedup and reduces high
bit-width scaling costs compared to existing SRAM-based CiM designs.

### 5. [SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators](http://arxiv.org/pdf/2511.03092v1)

Authors: Jonathan Li, Nasim Farahini, Evgenii Iuliugin, Magnus Vesterlund, Christian Haggstrom, Guangtao Wang, Shubhangi Upasani, Ayush Sachdeva, Rui Li, Faline Fu, Chen Wu, Ayesha Siddiqua, John Long, Tuowen Zhao, Matheen Musaddiq, Hakan Zeffer, Yun Du, Mingran Wang, Qinghua Li, Bo Li, Urmish Thakker, Raghu Prabhakar

The proliferation of 100B+ parameter Large Language Models (LLMs) with 100k+
context length support have resulted in increasing demands for on-chip memory
to support large KV caches. Techniques such as StreamingLLM and SnapKV
demonstrate how to control KV cache size while maintaining model accuracy. Yet,
these techniques are not commonly used within industrial deployments using
frameworks like vLLM or SGLang. The reason is twofold: on one hand, the static
graphs and continuous batching methodology employed by these frameworks make it
difficult to admit modifications to the standard multi-head attention
algorithm, while on the other hand, the accuracy implications of such
techniques on modern instruction-following and reasoning models are not well
understood, obfuscating the need for implementing these techniques. In this
paper, we explore these accuracy implications on Llama-3.1-8B-Instruct and
DeepSeek-R1, and develop SnapStream, a KV cache compression method that can be
deployed at scale. We demonstrate the efficacy of SnapStream in a 16-way
tensor-parallel deployment of DeepSeek-671B on SambaNova SN40L accelerators
running at 128k context length and up to 1832 tokens per second in a real
production setting. SnapStream enables $4\times$ improved on-chip memory usage
and introduces minimal accuracy degradation on LongBench-v2, AIME24 and
LiveCodeBench. To the best of our knowledge, this is the first implementation
of sparse KV attention techniques deployed in a production inference system
with static graphs and continuous batching.

### 6. [AnaFlow: Agentic LLM-based Workflow for Reasoning-Driven Explainable and Sample-Efficient Analog Circuit Sizing](http://arxiv.org/pdf/2511.03697v1)

Authors: Mohsen Ahmadzadeh, Kaichang Chen, Georges Gielen

Analog/mixed-signal circuits are key for interfacing electronics with the
physical world. Their design, however, remains a largely handcrafted process,
resulting in long and error-prone design cycles. While the recent rise of
AI-based reinforcement learning and generative AI has created new techniques to
automate this task, the need for many time-consuming simulations is a critical
bottleneck hindering the overall efficiency. Furthermore, the lack of
explainability of the resulting design solutions hampers widespread adoption of
the tools. To address these issues, a novel agentic AI framework for
sample-efficient and explainable analog circuit sizing is presented. It employs
a multi-agent workflow where specialized Large Language Model (LLM)-based
agents collaborate to interpret the circuit topology, to understand the design
goals, and to iteratively refine the circuit's design parameters towards the
target goals with human-interpretable reasoning. The adaptive simulation
strategy creates an intelligent control that yields a high sample efficiency.
The AnaFlow framework is demonstrated for two circuits of varying complexity
and is able to complete the sizing task fully automatically, differently from
pure Bayesian optimization and reinforcement learning approaches. The system
learns from its optimization history to avoid past mistakes and to accelerate
convergence. The inherent explainability makes this a powerful tool for analog
design space exploration and a new paradigm in analog EDA, where AI agents
serve as transparent design assistants.

### Computational Complexity

### 1. [An Analytical Approach to Parallel Repetition via CSP Inverse Theorems](http://arxiv.org/pdf/2511.03083v1)

Authors: Amey Bhangale, Mark Braverman, Subhash Khot, Yang P. Liu, Dor Minzer, Kunal Mittal

Let $\mathcal{G}$ be a $k$-player game with value $<1$, whose query
distribution is such that no marginal on $k-1$ players admits a non-trivial
Abelian embedding. We show that for every $n\geq N$, the value of the $n$-fold
parallel repetition of $\mathcal{G}$ is $$ \text{val}(\mathcal{G}^{\otimes n})
\leq \frac{1}{\underbrace{\log\log\cdots\log}_{C\text{ times}} n}, $$ where
$N=N(\mathcal{G})$ and $1\leq C\leq k^{O(k)}$ are constants. As a consequence,
we obtain a parallel repetition theorem for all $3$-player games whose query
distribution is pairwise-connected. Prior to our work, only inverse Ackermann
decay bounds were known for such games [Ver96].
  As additional special cases, we obtain a unified proof for all known parallel
repetition theorems, albeit with weaker bounds: (1) A new analytic proof of
parallel repetition for all 2-player games [Raz98, Hol09, DS14]. (2) A new
proof of parallel repetition for all $k$-player playerwise connected games
[DHVY17, GHMRZ22]. (3) Parallel repetition for all $3$-player games (in
particular $3$-XOR games) whose query distribution has no non-trivial Abelian
embedding into $(\mathbb{Z}, +)$ [BKM23c, BBKLM25]. (4) Parallel repetition for
all 3-player games with binary inputs [HR20, GHMRZ21, GHMRZ22, GMRZ22].

### 2. [Monotone Bounded Depth Formula Complexity of Graph Homomorphism Polynomials](http://arxiv.org/pdf/2511.03388v1)

Authors: Balagopal Komarath, Rohit Narayanan

We characterize the monotone bounded depth formula complexity for graph
homomorphism and colored isomorphism polynomials using a graph parameter called
the cost of bounded product depth baggy elimination tree. Using this
characterization, we show an almost optimal separation between monotone
circuits and monotone formulas using constant-degree polynomials for all fixed
product depths, and an almost optimal separation between monotone formulas of
product depths $\Delta$ and $\Delta$ + 1 for all $\Delta$ $\ge$ 1.

### 3. [Ideals, Gröbner Bases, and PCPs](http://arxiv.org/pdf/2511.03703v1)

Authors: Prashanth Amireddy, Amik Raj Behera, Srikanth Srinivasan, Madhu Sudan, Sophus Valentin Willumsgaard

All known proofs of the PCP theorem rely on multiple "composition" steps,
where PCPs over large alphabets are turned into PCPs over much smaller
alphabets at a (relatively) small price in the soundness error of the PCP.
Algebraic proofs, starting with the work of Arora, Lund, Motwani, Sudan, and
Szegedy use at least 2 such composition steps, whereas the "Gap amplification"
proof of Dinur uses $\Theta(\log n)$ such composition steps. In this work, we
present the first PCP construction using just one composition step. The key
ingredient, missing in previous work and finally supplied in this paper, is a
basic PCP (of Proximity) of size $2^{n^\epsilon}$, for any $\epsilon > 0$, that
makes $O_\epsilon(1)$ queries.
  At the core of our new construction is a new class of alternatives to
"sum-check" protocols. As used in past PCPs, these provide a method by which to
verify that an $m$-variate degree $d$ polynomial $P$ evaluates to zero at every
point of some set $S \subseteq \mathbb{F}_q^m$. Previous works had shown how to
check this condition for sets of the form $S = H^m$ using $O(m)$ queries with
alphabet $\mathbb{F}_q^d$ assuming $d \geq |H|$. Our work improves this basic
protocol in two ways: First we extend it to broader classes of sets $S$ (ones
closer to Hamming balls rather than cubes). Second, it reduces the number of
queries from $O(m)$ to an absolute constant for the settings of $S$ we
consider. Specifically when $S = (\{0,1\}^{m/c}_{\leq 1})^c$, we give such an
alternate to the sum-check protocol with $O(1)$ queries with alphabet
$\mathbb{F}_q^{O(c+d)}$, using proofs of size $q^{O(m^2/c)}$. Our new protocols
use insights from the powerful theory of Gr\"obner bases to extend previously
known protocols to these new settings with surprising ease. In doing so, they
highlight why these theories from algebra may be of further use in complexity
theory.

### 4. [Hesse's Redemption: Efficient Convex Polynomial Programming](http://arxiv.org/pdf/2511.03440v1)

Authors: Lucas Slot, David Steurer, Manuel Wiedmer

Efficient algorithms for convex optimization, such as the ellipsoid method,
require an a priori bound on the radius of a ball around the origin guaranteed
to contain an optimal solution if one exists. For linear and convex quadratic
programming, such solution bounds follow from classical characterizations of
optimal solutions by systems of linear equations. For other programs, e.g.,
semidefinite ones, examples due to Khachiyan show that optimal solutions may
require huge coefficients with an exponential number of bits, even if we allow
approximations. Correspondingly, semidefinite programming is not even known to
be in NP. The unconstrained minimization of convex polynomials of degree four
and higher has remained a fundamental open problem between these two extremes:
its optimal solutions do not admit a linear characterization and, at the same
time, Khachiyan-type examples do not apply. We resolve this problem by
developing new techniques to prove solution bounds when no linear
characterizations are available. Even for programs minimizing a convex
polynomial (of arbitrary degree) over a polyhedron, we prove that the existence
of an optimal solution implies that an approximately optimal one with
polynomial bit length also exists. These solution bounds, combined with the
ellipsoid method, yield the first polynomial-time algorithm for convex
polynomial programming, settling a question posed by Nesterov (Math. Program.,
2019). Before, no polynomial-time algorithm was known even for unconstrained
minimization of a convex polynomial of degree four.

### 5. [Efficient Testing Implies Structured Symmetry](http://arxiv.org/pdf/2511.03653v1)

Authors: Cynthia Dwork, Pranay Tankala

Given a small random sample of $n$-bit strings labeled by an unknown Boolean
function, which properties of this function can be tested computationally
efficiently? We show an equivalence between properties that are efficiently
testable from few samples and properties with structured symmetry, which depend
only on the function's average values on parts of a low-complexity partition of
the domain. Without the efficiency constraint, a similar characterization in
terms of unstructured symmetry was obtained by Blais and Yoshida (2019). Our
main technical tool is supersimulation, which builds on methods from the
algorithmic fairness literature to approximate arbitrarily complex functions by
small-circuit simulators that fool significantly larger distinguishers.
  We extend the characterization along other axes as well. We show that
allowing parts to overlap exponentially reduces their required number,
broadening the scope of the construction from properties testable with $O(\log
n)$ samples to properties testable with $O(n)$ samples. For larger sample
sizes, we show that any efficient tester is essentially checking for
indistinguishability from a bounded collection of small circuits, in the spirit
of a characterization of testable graph properties. Finally, we show that our
results for Boolean function testing generalize to high-entropy distribution
testing on arbitrary domains.

### Computational Engineering

### 1. [Simulation-Based Validation of an Integrated 4D/5D Digital-Twin Framework for Predictive Construction Control](http://arxiv.org/pdf/2511.03684v1)

Authors: Atena Khoshkonesh, Mohsen Mohammadagha, Navid Ebrahimi

Persistent cost and schedule deviations remain a major challenge in the U.S.
construction industry, revealing the limitations of deterministic CPM and
static document-based estimating. This study presents an integrated 4D/5D
digital-twin framework that couples Building Information Modeling (BIM) with
natural-language processing (NLP)-based cost mapping, computer-vision
(CV)-driven progress measurement, Bayesian probabilistic CPM updating, and
deep-reinforcement-learning (DRL) resource-leveling. A nine-month case
implementation on a Dallas-Fort Worth mid-rise project demonstrated measurable
gains in accuracy and efficiency: 43% reduction in estimating labor, 6%
reduction in overtime, and 30% project-buffer utilization, while maintaining an
on-time finish at 128 days within P50-P80 confidence bounds. The digital-twin
sandbox also enabled real-time "what-if" forecasting and traceable
cost-schedule alignment through a 5D knowledge graph. Findings confirm that
integrating AI-based analytics with probabilistic CPM and DRL enhances
forecasting precision, transparency, and control resilience. The validated
workflow establishes a practical pathway toward predictive, adaptive, and
auditable construction management.

### 2. [Multi-Region Matrix Interpolation for Dynamic Analysis of Aperiodic Structures under Large Model Parameter Perturbations](http://arxiv.org/pdf/2511.03711v1)

Authors: J. Pereira, R. O. Ruiz

This work introduces a surrogate-based model for efficiently estimating the
frequency response of dynamic mechanical metamaterials, particularly when
dealing with large parametric perturbations and aperiodic substructures. The
research builds upon a previous matrix interpolation method applied on top of a
Craig-Bampton modal reduction, allowing the variations of geometrical features
without the need to remesh and recompute Finite Element matrices. This existing
procedure has significant limitations since it requires a common modal
projection, which inherently restricts the allowable perturbation size of the
model parameters, thereby limiting the model parameter space where matrices can
be effectively interpolated. The present work offers three contributions: (1)
It provides structural dynamic insight into the restrictions imposed by the
common modal projection, demonstrating that ill-conditioning can be controlled,
(2) it proposes an efficient, sampling-based procedure to identify the
non-regular boundaries of the usable region in the model parameter space, and
(3) it enhances the surrogate model to accommodate larger model parameter
perturbations by proposing a multi-region interpolation strategy. The efficacy
of this proposed framework is verified through two illustrative examples. The
first example, involving a unit cell with a square plate and circular core,
validates the approach for a single well-conditioned projection region. The
second example, using a beam-like structure with vibration attenuation bands,
demonstrates the true advantage of the multi-region approach, where predictions
from traditional Lagrange interpolation deviated significantly with increasing
perturbations, while the proposed method maintained high accuracy across
different perturbation levels.

### 3. [GraphCliff: Short-Long Range Gating for Subtle Differences but Critical Changes](http://arxiv.org/pdf/2511.03170v1)

Authors: Hajung Kim, Jueon Park, Junseok Choe, Sheunheun Baek, Hyeon Hwang, Jaewoo Kang

Quantitative structure-activity relationship assumes a smooth relationship
between molecular structure and biological activity. However, activity cliffs
defined as pairs of structurally similar compounds with large potency
differences break this continuity. Recent benchmarks targeting activity cliffs
have revealed that classical machine learning models with extended connectivity
fingerprints outperform graph neural networks. Our analysis shows that graph
embeddings fail to adequately separate structurally similar molecules in the
embedding space, making it difficult to distinguish between structurally
similar but functionally different molecules. Despite this limitation,
molecular graph structures are inherently expressive and attractive, as they
preserve molecular topology. To preserve the structural representation of
molecules as graphs, we propose a new model, GraphCliff, which integrates
short- and long-range information through a gating mechanism. Experimental
results demonstrate that GraphCliff consistently improves performance on both
non-cliff and cliff compounds. Furthermore, layer-wise node embedding analyses
reveal reduced over-smoothing and enhanced discriminative power relative to
strong baseline graph models.

### 4. [A Theoretical Framework for Environmental Similarity and Vessel Mobility as Coupled Predictors of Marine Invasive Species Pathways](http://arxiv.org/pdf/2511.03499v1)

Authors: Gabriel Spadon, Vaishnav Vaidheeswaran, Claudio DiBacco

Marine invasive species spread through global shipping and generate
substantial ecological and economic impacts. Traditional risk assessments
require detailed records of ballast water and traffic patterns, which are often
incomplete, limiting global coverage. This work advances a theoretical
framework that quantifies invasion risk by combining environmental similarity
across ports with observed and forecasted maritime mobility. Climate-based
feature representations characterize each port's marine conditions, while
mobility networks derived from Automatic Identification System data capture
vessel flows and potential transfer pathways. Clustering and metric learning
reveal climate analogues and enable the estimation of species survival
likelihood along shipping routes. A temporal link prediction model captures how
traffic patterns may change under shifting environmental conditions. The
resulting fusion of environmental similarity and predicted mobility provides
exposure estimates at the port and voyage levels, supporting targeted
monitoring, routing adjustments, and management interventions.

### 5. [Improving Gene Trees without more data](http://arxiv.org/pdf/2511.03692v1)

Authors: Ashu Gupta

Estimating species and gene trees from sequence data is challenging. Gene
tree estimation is often hampered by low phylogenetic signal in alignments,
leading to inaccurate trees. Species tree estimation is complicated by
incomplete lineage sorting (ILS), where gene histories differ from the species'
history. Summary methods like MP-EST, ASTRAL2, and ASTRID infer species trees
from gene trees but suffer when gene tree accuracy is low. To address this, the
Statistical Binning (SB) and Weighted Statistical Binning (WSB) pipelines were
developed to improve gene tree estimation. However, previous studies only
tested these pipelines using multi-locus bootstrapping (MLBS), not the BestML
approach.
  This thesis proposes a novel pipeline, WSB+WQMC, which shares design features
with the existing WSB+CAML pipeline but has other desirable properties and is
statistically consistent under the GTR+MSC model. This study evaluated WSB+WQMC
against WSB+CAML using BestML analysis on various simulated datasets. The
results confirmed many trends seen in prior MLBS analyses. WSB+WQMC
substantially improved gene tree and species tree accuracy (using ASTRAL2 and
ASTRID) on most datasets with low, medium, and moderately high ILS levels. In a
direct comparison, WSB+WQMC computed less accurate trees than WSB+CAML under
certain low and medium ILS conditions. However, WSB+WQMC performed better or at
least as accurately as WSB+CAML on all datasets with moderately high and high
ILS. It also proved better for estimating gene trees on some medium and low ILS
datasets. Thus, WSB+WQMC is a promising alternative to WSB+CAML for
phylogenetic estimation, especially in the presence of low phylogenetic signal.

### 6. [System Identification of a Moored ASV with Recessed Moon Pool via Deterministic and Bayesian Hankel-DMDc](http://arxiv.org/pdf/2511.03482v1)

Authors: Giorgio Palma, Ivan Santic, Andrea Serani, Lorenzo Minno, Matteo Diez

This study addresses the system identification of a small autonomous surface
vehicle (ASV) under moored conditions using Hankel dynamic mode decomposition
with control (HDMDc) and its Bayesian extension (BHDMDc). Experiments were
carried out on a Codevintec CK-14e ASV in the towing tank of CNR-INM, under
both irregular and regular head-sea wave conditions. The ASV under
investigation features a recessed moon pool, which induces nonlinear responses
due to sloshing, thereby increasing the modelling challenge. Data-driven
reduced-order models were built from measurements of vessel motions and mooring
loads. The HDMDc framework provided accurate deterministic predictions of
vessel dynamics, while the Bayesian formulation enabled uncertainty-aware
characterization of the model response by accounting for variability in
hyperparameter selection. Validation against experimental data demonstrated
that both HDMDc and BHDMDc can predict the vessel's response to unseen regular
and irregular wave excitations. In conclusion, the study shows that HDMDc-based
ROMs are a viable data-driven alternative for system identification,
demonstrating for the first time their generalization capability for a sea
condition different from the training set, achieving high accuracy in
reproducing vessel dynamics.

### 7. [LiveTradeBench: Seeking Real-World Alpha with Large Language Models](http://arxiv.org/pdf/2511.03628v1)

Authors: Haofei Yu, Fenghai Li, Jiaxuan You

Large language models (LLMs) achieve strong performance across
benchmarks--from knowledge quizzes and math reasoning to web-agent tasks--but
these tests occur in static settings, lacking real dynamics and uncertainty.
Consequently, they evaluate isolated reasoning or problem-solving rather than
decision-making under uncertainty. To address this, we introduce
LiveTradeBench, a live trading environment for evaluating LLM agents in
realistic and evolving markets. LiveTradeBench follows three design principles:
(i) Live data streaming of market prices and news, eliminating dependence on
offline backtesting and preventing information leakage while capturing
real-time uncertainty; (ii) a portfolio-management abstraction that extends
control from single-asset actions to multi-asset allocation, integrating risk
management and cross-asset reasoning; and (iii) multi-market evaluation across
structurally distinct environments--U.S. stocks and Polymarket prediction
markets--differing in volatility, liquidity, and information flow. At each
step, an agent observes prices, news, and its portfolio, then outputs
percentage allocations that balance risk and return. Using LiveTradeBench, we
run 50-day live evaluations of 21 LLMs across families. Results show that (1)
high LMArena scores do not imply superior trading outcomes; (2) models display
distinct portfolio styles reflecting risk appetite and reasoning dynamics; and
(3) some LLMs effectively leverage live signals to adapt decisions. These
findings expose a gap between static evaluation and real-world competence,
motivating benchmarks that test sequential decision making and consistency
under live uncertainty.

### Computational Geometry

### 1. [Generalized k-Cell Decomposition for Visibility Planning in Polygons](http://arxiv.org/pdf/2511.03642v1)

Authors: Yeganeh Bahoo, Sajad Saeedi, Roni Sherman

This paper introduces a novel $k$-cell decomposition method for
pursuit-evasion problems in polygonal environments, where a searcher is
equipped with a $k$-modem: a device capable of seeing through up to $k$ walls.
The proposed decomposition ensures that as the searcher moves within a cell,
the structure of unseen regions (shadows) remains unchanged, thereby preventing
any geometric events between or on invisible regions, that is, preventing the
appearance, disappearance, merge, or split of shadow regions. The method
extends existing work on $0$- and $2$-visibility by incorporating m-visibility
polygons for all even $0 \le m \le k$, constructing partition lines that enable
robust environment division. The correctness of the decomposition is proved via
three theorems. The decomposition enables reliable path planning for intruder
detection in simulated environments and opens new avenues for visibility-based
robotic surveillance. The difficulty in constructing the cells of the
decomposition consists in computing the $k$-visibility polygon from each vertex
and finding the intersection points of the partition lines to create the cells.

### 2. [Multi-robot searching with limited sensing range for static and mobile intruders](http://arxiv.org/pdf/2511.03622v1)

Authors: Swadhin Agrawal, Sujoy Bhore, Joseph S. B. Mitchell, P. B. Sujit, Aayush Gohil

We consider the problem of searching for an intruder in a geometric domain by
utilizing multiple search robots. The domain is a simply connected orthogonal
polygon with edges parallel to the cartesian coordinate axes. Each robot has a
limited sensing capability. We study the problem for both static and mobile
intruders. It turns out that the problem of finding an intruder is NP-hard,
even for a stationary intruder. Given this intractability, we turn our
attention towards developing efficient and robust algorithms, namely methods
based on space-filling curves, random search, and cooperative random search.
Moreover, for each proposed algorithm, we evaluate the trade-off between the
number of search robots and the time required for the robots to complete the
search process while considering the geometric properties of the connected
orthogonal search area.

### Computation and Language

### 1. [MME-CC: A Challenging Multi-Modal Evaluation Benchmark of Cognitive Capacity](http://arxiv.org/pdf/2511.03146v1)

Authors: Kaiyuan Zhang, Chenghao Yang, Zhoufutu Wen, Sihang Yuan, Qiuyue Wang, Chaoyi Huang, Guosheng Zhu, He Wang, Huawenyu Lu, Jianing Wen, Jianpeng Jiao, Lishu Luo, Longxiang Liu, Sijin Wu, Xiaolei Zhu, Xuanliang Zhang, Ge Zhang, Yi Lin, Guang Shi, Chaoyou Fu, Wenhao Huang

As reasoning models scale rapidly, the essential role of multimodality in
human cognition has come into sharp relief, driving a growing need to probe
vision-centric cognitive behaviors. Yet, existing multimodal benchmarks either
overemphasize textual reasoning or fall short of systematically capturing
vision-centric cognitive behaviors, leaving the cognitive capacity of MLLMs
insufficiently assessed. To address this limitation, we introduce MME-CC
(Multi-Modal Evaluation benchmark of Cognitive Capacity), a vision-grounded
benchmark that organizes 11 representative reasoning tasks into three
fundamental categories of visual information: spatial, geometric, and
knowledge-based reasoning, and provides fine-grained analyses of MLLMs'
cognitive capacity across these dimensions. Based on MME-CC, we conduct
extensive experiments over 16 representative MLLMs. Our study reveals that
closed-source models currently lead overall (e.g., 42.66 for Gemini-2.5-Pro vs.
30.45 for GLM-4.5V), while spatial and geometric reasoning remain broadly weak
(less than or equal to 30%). We further identify common error patterns,
including orientation mistakes, fragile cross-view identity persistence, and
poor adherence to counterfactual instructions, and observe that
Chain-of-Thought typically follows a three-stage process (extract -> reason ->
verify) with heavy reliance on visual extraction. We hope this work catalyzes a
shift toward treating the cognitive capacity of MLLMs as central to both
evaluation and model design.

### 2. [Measuring Aleatoric and Epistemic Uncertainty in LLMs: Empirical Evaluation on ID and OOD QA Tasks](http://arxiv.org/pdf/2511.03166v1)

Authors: Kevin Wang, Subre Abdoul Moktar, Jia Li, Kangshuo Li, Feng Chen

Large Language Models (LLMs) have become increasingly pervasive, finding
applications across many industries and disciplines. Ensuring the
trustworthiness of LLM outputs is paramount, where Uncertainty Estimation (UE)
plays a key role. In this work, a comprehensive empirical study is conducted to
examine the robustness and effectiveness of diverse UE measures regarding
aleatoric and epistemic uncertainty in LLMs. It involves twelve different UE
methods and four generation quality metrics including LLMScore from LLM
criticizers to evaluate the uncertainty of LLM-generated answers in
Question-Answering (QA) tasks on both in-distribution (ID) and
out-of-distribution (OOD) datasets. Our analysis reveals that information-based
methods, which leverage token and sequence probabilities, perform exceptionally
well in ID settings due to their alignment with the model's understanding of
the data. Conversely, density-based methods and the P(True) metric exhibit
superior performance in OOD contexts, highlighting their effectiveness in
capturing the model's epistemic uncertainty. Semantic consistency methods,
which assess variability in generated answers, show reliable performance across
different datasets and generation metrics. These methods generally perform well
but may not be optimal for every situation.

### 3. [BengaliMoralBench: A Benchmark for Auditing Moral Reasoning in Large Language Models within Bengali Language and Culture](http://arxiv.org/pdf/2511.03180v1)

Authors: Shahriyar Zaman Ridoy, Azmine Toushik Wasi, Koushik Ahamed Tonmoy

As multilingual Large Language Models (LLMs) gain traction across South Asia,
their alignment with local ethical norms, particularly for Bengali, which is
spoken by over 285 million people and ranked 6th globally, remains
underexplored. Existing ethics benchmarks are largely English-centric and
shaped by Western frameworks, overlooking cultural nuances critical for
real-world deployment. To address this, we introduce BengaliMoralBench, the
first large-scale ethics benchmark for the Bengali language and socio-cultural
contexts. It covers five moral domains, Daily Activities, Habits, Parenting,
Family Relationships, and Religious Activities, subdivided into 50 culturally
relevant subtopics. Each scenario is annotated via native-speaker consensus
using three ethical lenses: Virtue, Commonsense, and Justice ethics. We conduct
systematic zero-shot evaluation of prominent multilingual LLMs, including
Llama, Gemma, Qwen, and DeepSeek, using a unified prompting protocol and
standard metrics. Performance varies widely (50-91% accuracy), with qualitative
analysis revealing consistent weaknesses in cultural grounding, commonsense
reasoning, and moral fairness. BengaliMoralBench provides a foundation for
responsible localization, enabling culturally aligned evaluation and supporting
the deployment of ethically robust AI in diverse, low-resource multilingual
settings such as Bangladesh.

### 4. [IndicSuperTokenizer: An Optimized Tokenizer for Indic Multilingual LLMs](http://arxiv.org/pdf/2511.03237v1)

Authors: Souvik Rana, Arul Menezes, Ashish Kulkarni, Chandra Khatri, Shubham Agarwal

Tokenizers play a crucial role in determining the performance, training
efficiency, and the inference cost of Large Language Models (LLMs). Designing
effective tokenizers for multilingual LLMs is particularly challenging due to
diverse scripts and rich morphological variation. While subword methods such as
Byte Pair Encoding (BPE) are widely adopted, their effectiveness in
multilingual settings remains underexplored. We present IndicSuperTokenizer, a
tokenizer for Indic multilingual LLMs, that combines both subword and
multi-word tokenization, along with language-specific pre-tokenization, leading
to more linguistically aligned tokens and achieving a new state-of-the-art in
fertility score. Evaluated across English, 22 Indian languages and code data,
our tokenizer improves the average fertility score by 39.5% over LLaMA4 and by
18% over Sutra (the current best). This translates to 44% improvement in
inference throughput over LLaMA4 while maintaining comparable performance on
English and Indic benchmarks. We also present detailed ablations across
tokenizer training data size, vocabulary size, merging techniques, and
pre-tokenization strategies, demonstrating the robustness of our design
choices.

### 5. [SCALE: Upscaled Continual Learning of Large Language Models](http://arxiv.org/pdf/2511.03270v1)

Authors: Jin-woo Lee, Junhwa Choi, Bongkyu Hwang, Jinho Choo, Bogun Kim, JeongSeon Yi, Joonseok Lee, DongYoung Jung, Jaeseon Park, Kyoungwon Park, Suk-hoon Jung

We revisit continual pre-training for large language models and argue that
progress now depends more on scaling the right structure than on scaling
parameters alone. We introduce SCALE, a width upscaling architecture that
inserts lightweight expansion into linear modules while freezing all
pre-trained parameters. This preserves the residual and attention topologies
and increases capacity without perturbing the base model's original
functionality. SCALE is guided by two principles: Persistent Preservation,
which maintains the base model's behavior via preservation-oriented
initialization and freezing of the pre-trained weights, and Collaborative
Adaptation, which selectively trains a subset of expansion components to
acquire new knowledge with minimal interference. We instantiate these ideas as
SCALE-Preserve (preservation-first), SCALE-Adapt (adaptation-first), and
SCALE-Route, an optional routing extension that performs token-level routing
between preservation and adaptation heads. On a controlled synthetic biography
benchmark, SCALE mitigates the severe forgetting observed with depth expansion
while still acquiring new knowledge. In continual pre-training on a Korean
corpus, SCALE variants achieve less forgetting on English evaluations and
competitive gains on Korean benchmarks, with these variants offering the best
overall stability-plasticity trade-off. Accompanying analysis clarifies when
preservation provably holds and why the interplay between preservation and
adaptation stabilizes optimization compared to standard continual learning
setups.

### 6. [EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation](http://arxiv.org/pdf/2511.03370v1)

Authors: Yunbo Long, Yuhan Liu, Alexandra Brintrup

The deployment of large language models (LLMs) in automated negotiation has
set a high performance benchmark, but their computational cost and data privacy
requirements render them unsuitable for many privacy-sensitive, on-device
applications such as mobile assistants, embodied AI agents or private client
interactions. While small language models (SLMs) offer a practical alternative,
they suffer from a significant performance gap compared to LLMs in playing
emotionally charged complex personas, especially for credit negotiation. This
paper introduces EQ-Negotiator, a novel framework that bridges this capability
gap using emotional personas. Its core is a reasoning system that integrates
game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional
states online, without pre-training. This allows EQ-Negotiator to equip SLMs
with the strategic intelligence to counter manipulation while de-escalating
conflict and upholding ethical standards. Through extensive agent-to-agent
simulations across diverse credit negotiation scenarios, including adversarial
debtor strategies like cheating, threatening, and playing the victim, we show
that a 7B parameter language model with EQ-Negotiator achieves better debt
recovery and negotiation efficiency than baseline LLMs more than 10 times its
size. This work advances persona modeling from descriptive character profiles
to dynamic emotional architectures that operate within privacy constraints.
Besides, this paper establishes that strategic emotional intelligence, not raw
model scale, is the critical factor for success in automated negotiation,
paving the way for effective, ethical, and privacy-preserving AI negotiators
that can operate on the edge.

### 7. [LFC-DA: Logical Formula-Controlled Data Augmentation for Enhanced Logical Reasoning](http://arxiv.org/pdf/2511.03372v1)

Authors: Shenghao Li

For complex logical data augmentation, heavy reliance on human annotation is
costly, whereas direct generation with large language models yields
uninterpretable and logically homogeneous examples. To address this, we present
LFC-DA, a symbolic-logic-controlled pipeline: logical text is first mapped to
propositional expressions, a compact rule library is compiled, and a bounded
state-space search systematically discovers valid formulas that are then
verbalized back into natural-language questions, ensuring both diversity and
logical rigor under propositional logic. Experiments on ReClor and LogiQA show
significant improvements in the logical-reasoning accuracy of pretrained
models, confirming the effectiveness of LFC-DA for LLM-guided logical data
augmentation.

### 8. [Segmentation Beyond Defaults: Asymmetrical Byte Pair Encoding for Optimal Machine Translation Performance](http://arxiv.org/pdf/2511.03383v1)

Authors: Saumitra Yadav, Manish Shrivastava

Existing Machine Translation (MT) research often suggests a single, fixed set
of hyperparameters for word segmentation models, symmetric Byte Pair Encoding
(BPE), which applies the same number of merge operations (NMO) to train
tokenizers for both source and target languages. However, we demonstrate that
this uniform approach doesn't guarantee optimal MT performance across different
language pairs and data sizes. This work investigates BPE segmentation recipes
across various data volumes and language pairs to evaluate MT system
performance. We find that utilizing asymmetric BPE, where the source and target
languages have different NMOs, significantly improves results over the
symmetric approach, especially in low-resource settings (50K, 100K, and 500K
sentence pairs). Specifically, asymmetric BPE yield statistically significant
($p<0.05$) average gains of 5.32, 4.46, and 0.7 CHRF++ on English-Hindi in
low-resource setups. We validated this trend across six additional language
pairs (English and Telugu, Shona, Norwegian, Kyrgyz, Hausa, and Inuktitut),
observing statistically significant improvement in 10 out of 12 systems
compared to symmetric BPE. Our findings indicate a high NMO for the source (4K
to 32K) and a low NMO for the target (0.5K to 2K) provides optimal results,
particularly benefiting low-resource MT.

### 9. [Overcoming the Generalization Limits of SLM Finetuning for Shape-Based Extraction of Datatype and Object Properties](http://arxiv.org/pdf/2511.03407v1)

Authors: Célian Ringwald, Fabien Gandon, Catherine Faron, Franck Michel, Hanna Abi Akl

Small language models (SLMs) have shown promises for relation extraction (RE)
when extracting RDF triples guided by SHACL shapes focused on common datatype
properties. This paper investigates how SLMs handle both datatype and object
properties for a complete RDF graph extraction. We show that the key bottleneck
is related to long-tail distribution of rare properties. To solve this issue,
we evaluate several strategies: stratified sampling, weighted loss, dataset
scaling, and template-based synthetic data augmentation. We show that the best
strategy to perform equally well over unbalanced target properties is to build
a training set where the number of occurrences of each property exceeds a given
threshold. To enable reproducibility, we publicly released our datasets,
experimental results and code. Our findings offer practical guidance for
training shape-aware SLMs and highlight promising directions for future work in
semantic RE.

### 10. [Efficient Reasoning via Thought-Training and Thought-Free Inference](http://arxiv.org/pdf/2511.03408v1)

Authors: Canhui Wu, Qiong Cao, Chao Xue, Wei Xi, Xiaodong He

Recent advances in large language models (LLMs) have leveraged explicit
Chain-of-Thought (CoT) prompting to improve reasoning accuracy. However, most
existing methods primarily compress verbose reasoning outputs. These
Long-to-Short transformations aim to improve efficiency, but still rely on
explicit reasoning during inference. In this work, we introduce \textbf{3TF}
(\textbf{T}hought-\textbf{T}raining and \textbf{T}hought-\textbf{F}ree
inference), a framework for efficient reasoning that takes a Short-to-Long
perspective. We first train a hybrid model that can operate in both reasoning
and non-reasoning modes, and then further train it on CoT-annotated data to
internalize structured reasoning, while enforcing concise, thought-free outputs
at inference time using the no-reasoning mode. Unlike compression-based
approaches, 3TF improves the reasoning quality of non-reasoning outputs,
enabling models to perform rich internal reasoning implicitly while keeping
external outputs short. Empirically, 3TF-trained models obtain large
improvements on reasoning benchmarks under thought-free inference,
demonstrating that high quality reasoning can be learned and executed
implicitly without explicit step-by-step generation.

### Cryptography and Security

### 1. [Bayesian Advantage of Re-Identification Attack in the Shuffle Model](http://arxiv.org/pdf/2511.03213v1)

Authors: Pengcheng Su, Haibo Cheng, Ping Wang

The shuffle model, which anonymizes data by randomly permuting user messages,
has been widely adopted in both cryptography and differential privacy. In this
work, we present the first systematic study of the Bayesian advantage in
re-identifying a user's message under the shuffle model. We begin with a basic
setting: one sample is drawn from a distribution $P$, and $n - 1$ samples are
drawn from a distribution $Q$, after which all $n$ samples are randomly
shuffled. We define $\beta_n(P, Q)$ as the success probability of a
Bayes-optimal adversary in identifying the sample from $P$, and define the
additive and multiplicative Bayesian advantages as $\mathsf{Adv}_n^{+}(P, Q) =
\beta_n(P,Q) - \frac{1}{n}$ and $\mathsf{Adv}_n^{\times}(P, Q) = n \cdot
\beta_n(P,Q)$, respectively. We derive exact analytical expressions and
asymptotic characterizations of $\beta_n(P, Q)$, along with evaluations in
several representative scenarios. Furthermore, we establish (nearly) tight
mutual bounds between the additive Bayesian advantage and the total variation
distance. Finally, we extend our analysis beyond the basic setting and present,
for the first time, an upper bound on the success probability of Bayesian
attacks in shuffle differential privacy. Specifically, when the outputs of $n$
users -- each processed through an $\varepsilon$-differentially private local
randomizer -- are shuffled, the probability that an attacker successfully
re-identifies any target user's message is at most $e^{\varepsilon}/n$.

### 2. [Smartphone User Fingerprinting on Wireless Traffic](http://arxiv.org/pdf/2511.03229v1)

Authors: Yong Huang, Zhibo Dong, Xiaoguang Yang, Dalong Zhang, Qingxian Wang, Zhihua Wang

Due to the openness of the wireless medium, smartphone users are susceptible
to user privacy attacks, where user privacy information is inferred from
encrypted Wi-Fi wireless traffic. Existing attacks are limited to recognizing
mobile apps and their actions and cannot infer the smartphone user identity, a
fundamental part of user privacy. To overcome this limitation, we propose
U-Print, a novel attack system that can passively recognize smartphone apps,
actions, and users from over-the-air MAC-layer frames. We observe that
smartphone users usually prefer different add-on apps and in-app actions,
yielding different changing patterns in Wi-Fi traffic. U-Print first extracts
multi-level traffic features and exploits customized temporal convolutional
networks to recognize smartphone apps and actions, thus producing users'
behavior sequences. Then, it leverages the silhouette coefficient method to
determine the number of users and applies the k-means clustering to profile and
identify smartphone users. We implement U-Print using a laptop with a Kali
dual-band wireless network card and evaluate it in three real-world
environments. U-Print achieves an overall accuracy of 98.4% and an F1 score of
0.983 for user inference. Moreover, it can correctly recognize up to 96% of
apps and actions in the closed world and more than 86% in the open world.

### 3. [Auditing M-LLMs for Privacy Risks: A Synthetic Benchmark and Evaluation Framework](http://arxiv.org/pdf/2511.03248v1)

Authors: Junhao Li, Jiahao Chen, Zhou Feng, Chunyi Zhou

Recent advances in multi-modal Large Language Models (M-LLMs) have
demonstrated a powerful ability to synthesize implicit information from
disparate sources, including images and text. These resourceful data from
social media also introduce a significant and underexplored privacy risk: the
inference of sensitive personal attributes from seemingly daily media content.
However, the lack of benchmarks and comprehensive evaluations of
state-of-the-art M-LLM capabilities hinders the research of private attribute
profiling on social media. Accordingly, we propose (1) PRISM, the first
multi-modal, multi-dimensional and fine-grained synthesized dataset
incorporating a comprehensive privacy landscape and dynamic user history; (2)
an Efficient evaluation framework that measures the cross-modal privacy
inference capabilities of advanced M-LLM. Specifically, PRISM is a large-scale
synthetic benchmark designed to evaluate cross-modal privacy risks. Its key
feature is 12 sensitive attribute labels across a diverse set of multi-modal
profiles, which enables targeted privacy analysis. These profiles are generated
via a sophisticated LLM agentic workflow, governed by a prior distribution to
ensure they realistically mimic social media users. Additionally, we propose a
Multi-Agent Inference Framework that leverages a pipeline of specialized LLMs
to enhance evaluation capabilities. We evaluate the inference capabilities of
six leading M-LLMs (Qwen, Gemini, GPT-4o, GLM, Doubao, and Grok) on PRISM. The
comparison with human performance reveals that these MLLMs significantly
outperform in accuracy and efficiency, highlighting the threat of potential
privacy risks and the urgent need for robust defenses.

### 4. [Federated Anonymous Blocklisting across Service Providers and its Application to Group Messaging](http://arxiv.org/pdf/2511.03486v1)

Authors: David Soler, Carlos Dafonte, Manuel Fernández-Veiga, Ana Fernández Vilas, Francisco J. Nóvoa

Instant messaging has become one of the most used methods of communication
online, which has attracted significant attention to its underlying
cryptographic protocols and security guarantees. Techniques to increase privacy
such as End-to-End Encryption and pseudonyms have been introduced. However,
online spaces such as messaging groups still require moderation to prevent
misbehaving users from participating in them, particularly in anonymous
contexts.. In Anonymous Blocklisting (AB) schemes, users must prove during
authentication that none of their previous pseudonyms has been blocked,
preventing misbehaving users from creating new pseudonyms. In this work we
propose an alternative \textit{Federated Anonymous Blocklisting} (FAB) in which
the centralised Service Provider is replaced by small distributed Realms, each
with its own blocklist. Realms can establish trust relationships between each
other, such that when users authenticate to a realm, they must prove that they
are not banned in any of its trusted realms. We provide an implementation of
our proposed scheme; unlike existing AB constructions, the performance of ours
does not depend on the current size of the blocklist nor requires processing
new additions to the blocklist. We also demonstrate its applicability to
real-world messaging groups by integrating our FAB scheme into the Messaging
Layer Security protocol.

### 5. [Security and Privacy Management of IoT Using Quantum Computing](http://arxiv.org/pdf/2511.03538v1)

Authors: Jaydip Sen

The convergence of the Internet of Things (IoT) and quantum computing is
redefining the security paradigm of interconnected digital systems. Classical
cryptographic algorithms such as RSA, Elliptic Curve Cryptography (ECC), and
Advanced Encryption Standard (AES) have long provided the foundation for
securing IoT communication. However, the emergence of quantum algorithms such
as Shor's and Grover's threatens to render these techniques vulnerable,
necessitating the development of quantum-resilient alternatives. This chapter
examines the implications of quantum computing for IoT security and explores
strategies for building cryptographically robust systems in the post-quantum
era. It presents an overview of Post-Quantum Cryptographic (PQC) families,
including lattice-based, code-based, hash-based, and multivariate approaches,
analyzing their potential for deployment in resource-constrained IoT
environments. In addition, quantum-based methods such as Quantum Key
Distribution (QKD) and Quantum Random Number Generators (QRNGs) are discussed
for their ability to enhance confidentiality and privacy through physics-based
security guarantees. The chapter also highlights issues of privacy management,
regulatory compliance, and standardization, emphasizing the need for
collaborative efforts across academia, industry, and governance. Overall, it
provides a comprehensive perspective on security IoT ecosystems against quantum
threats and ensures resilience in the next generation of intelligent networks.

### 6. [Death by a Thousand Prompts: Open Model Vulnerability Analysis](http://arxiv.org/pdf/2511.03247v1)

Authors: Amy Chang, Nicholas Conley, Harish Santhanalakshmi Ganesan, Adam Swanda

Open-weight models provide researchers and developers with accessible
foundations for diverse downstream applications. We tested the safety and
security postures of eight open-weight large language models (LLMs) to identify
vulnerabilities that may impact subsequent fine-tuning and deployment. Using
automated adversarial testing, we measured each model's resilience against
single-turn and multi-turn prompt injection and jailbreak attacks. Our findings
reveal pervasive vulnerabilities across all tested models, with multi-turn
attacks achieving success rates between 25.86\% and 92.78\% -- representing a
$2\times$ to $10\times$ increase over single-turn baselines. These results
underscore a systemic inability of current open-weight models to maintain
safety guardrails across extended interactions. We assess that alignment
strategies and lab priorities significantly influence resilience:
capability-focused models such as Llama 3.3 and Qwen 3 demonstrate higher
multi-turn susceptibility, whereas safety-oriented designs such as Google Gemma
3 exhibit more balanced performance.
  The analysis concludes that open-weight models, while crucial for innovation,
pose tangible operational and ethical risks when deployed without layered
security controls. These findings are intended to inform practitioners and
developers of the potential risks and the value of professional AI security
solutions to mitigate exposure. Addressing multi-turn vulnerabilities is
essential to ensure the safe, reliable, and responsible deployment of
open-weight LLMs in enterprise and public domains. We recommend adopting a
security-first design philosophy and layered protections to ensure resilient
deployments of open-weight models.

### 7. [Let the Bees Find the Weak Spots: A Path Planning Perspective on Multi-Turn Jailbreak Attacks against LLMs](http://arxiv.org/pdf/2511.03271v1)

Authors: Yize Liu, Yunyun Hou, Aina Sui

Large Language Models (LLMs) have been widely deployed across various
applications, yet their potential security and ethical risks have raised
increasing concerns. Existing research employs red teaming evaluations,
utilizing multi-turn jailbreaks to identify potential vulnerabilities in LLMs.
However, these approaches often lack exploration of successful dialogue
trajectories within the attack space, and they tend to overlook the
considerable overhead associated with the attack process. To address these
limitations, this paper first introduces a theoretical model based on
dynamically weighted graph topology, abstracting the multi-turn attack process
as a path planning problem. Based on this framework, we propose ABC, an
enhanced Artificial Bee Colony algorithm for multi-turn jailbreaks, featuring a
collaborative search mechanism with employed, onlooker, and scout bees. This
algorithm significantly improves the efficiency of optimal attack path search
while substantially reducing the average number of queries required. Empirical
evaluations on three open-source and two proprietary language models
demonstrate the effectiveness of our approach, achieving attack success rates
above 90\% across the board, with a peak of 98\% on GPT-3.5-Turbo, and
outperforming existing baselines. Furthermore, it achieves comparable success
with only 26 queries on average, significantly reducing red teaming overhead
and highlighting its superior efficiency.

### 8. [LaMoS: Enabling Efficient Large Number Modular Multiplication through SRAM-based CiM Acceleration](http://arxiv.org/pdf/2511.03341v1)

Authors: Haomin Li, Fangxin Liu, Chenyang Guan, Zongwu Wang, Li Jiang, Haibing Guan

Barrett's algorithm is one of the most widely used methods for performing
modular multiplication, a critical nonlinear operation in modern privacy
computing techniques such as homomorphic encryption (HE) and zero-knowledge
proofs (ZKP). Since modular multiplication dominates the processing time in
these applications, computational complexity and memory limitations
significantly impact performance. Computing-in-Memory (CiM) is a promising
approach to tackle this problem. However, existing schemes currently suffer
from two main problems: 1) Most works focus on low bit-width modular
multiplication, which is inadequate for mainstream cryptographic algorithms
such as elliptic curve cryptography (ECC) and the RSA algorithm, both of which
require high bit-width operations; 2) Recent efforts targeting large number
modular multiplication rely on inefficient in-memory logic operations,
resulting in high scaling costs for larger bit-widths and increased latency. To
address these issues, we propose LaMoS, an efficient SRAM-based CiM design for
large-number modular multiplication, offering high scalability and area
efficiency. First, we analyze the Barrett's modular multiplication method and
map the workload onto SRAM CiM macros for high bit-width cases. Additionally,
we develop an efficient CiM architecture and dataflow to optimize large-number
modular multiplication. Finally, we refine the mapping scheme for better
scalability in high bit-width scenarios using workload grouping. Experimental
results show that LaMoS achieves a $7.02\times$ speedup and reduces high
bit-width scaling costs compared to existing SRAM-based CiM designs.

### 9. [Whisper Leak: a side-channel attack on Large Language Models](http://arxiv.org/pdf/2511.03675v1)

Authors: Geoff McDonald, Jonathan Bar Or

Large Language Models (LLMs) are increasingly deployed in sensitive domains
including healthcare, legal services, and confidential communications, where
privacy is paramount. This paper introduces Whisper Leak, a side-channel attack
that infers user prompt topics from encrypted LLM traffic by analyzing packet
size and timing patterns in streaming responses. Despite TLS encryption
protecting content, these metadata patterns leak sufficient information to
enable topic classification. We demonstrate the attack across 28 popular LLMs
from major providers, achieving near-perfect classification (often >98% AUPRC)
and high precision even at extreme class imbalance (10,000:1 noise-to-target
ratio). For many models, we achieve 100% precision in identifying sensitive
topics like "money laundering" while recovering 5-20% of target conversations.
This industry-wide vulnerability poses significant risks for users under
network surveillance by ISPs, governments, or local adversaries. We evaluate
three mitigation strategies - random padding, token batching, and packet
injection - finding that while each reduces attack effectiveness, none provides
complete protection. Through responsible disclosure, we have collaborated with
providers to implement initial countermeasures. Our findings underscore the
need for LLM providers to address metadata leakage as AI systems handle
increasingly sensitive information.

### 10. [Multi-robot searching with limited sensing range for static and mobile intruders](http://arxiv.org/pdf/2511.03622v1)

Authors: Swadhin Agrawal, Sujoy Bhore, Joseph S. B. Mitchell, P. B. Sujit, Aayush Gohil

We consider the problem of searching for an intruder in a geometric domain by
utilizing multiple search robots. The domain is a simply connected orthogonal
polygon with edges parallel to the cartesian coordinate axes. Each robot has a
limited sensing capability. We study the problem for both static and mobile
intruders. It turns out that the problem of finding an intruder is NP-hard,
even for a stationary intruder. Given this intractability, we turn our
attention towards developing efficient and robust algorithms, namely methods
based on space-filling curves, random search, and cooperative random search.
Moreover, for each proposed algorithm, we evaluate the trade-off between the
number of search robots and the time required for the robots to complete the
search process while considering the geometric properties of the connected
orthogonal search area.

### Computer Vision and Pattern Recognition

### 1. [DentalSplat: Dental Occlusion Novel View Synthesis from Sparse Intra-Oral Photographs](http://arxiv.org/pdf/2511.03099v1)

Authors: Yiyi Miao, Taoyu Wu, Tong Chen, Sihao Li, Ji Jiang, Youpeng Yang, Angelos Stefanidis, Limin Yu, Jionglong Su

In orthodontic treatment, particularly within telemedicine contexts,
observing patients' dental occlusion from multiple viewpoints facilitates
timely clinical decision-making. Recent advances in 3D Gaussian Splatting
(3DGS) have shown strong potential in 3D reconstruction and novel view
synthesis. However, conventional 3DGS pipelines typically rely on densely
captured multi-view inputs and precisely initialized camera poses, limiting
their practicality. Orthodontic cases, in contrast, often comprise only three
sparse images, specifically, the anterior view and bilateral buccal views,
rendering the reconstruction task especially challenging. The extreme sparsity
of input views severely degrades reconstruction quality, while the absence of
camera pose information further complicates the process. To overcome these
limitations, we propose DentalSplat, an effective framework for 3D
reconstruction from sparse orthodontic imagery. Our method leverages a
prior-guided dense stereo reconstruction model to initialize the point cloud,
followed by a scale-adaptive pruning strategy to improve the training
efficiency and reconstruction quality of 3DGS. In scenarios with extremely
sparse viewpoints, we further incorporate optical flow as a geometric
constraint, coupled with gradient regularization, to enhance rendering
fidelity. We validate our approach on a large-scale dataset comprising 950
clinical cases and an additional video-based test set of 195 cases designed to
simulate real-world remote orthodontic imaging conditions. Experimental results
demonstrate that our method effectively handles sparse input scenarios and
achieves superior novel view synthesis quality for dental occlusion
visualization, outperforming state-of-the-art techniques.

### 2. [Finetuning-Free Personalization of Text to Image Generation via Hypernetworks](http://arxiv.org/pdf/2511.03156v1)

Authors: Sagar Shrestha, Gopal Sharma, Luowei Zhou, Suren Kumar

Personalizing text-to-image diffusion models has traditionally relied on
subject-specific fine-tuning approaches such as
DreamBooth~\cite{ruiz2023dreambooth}, which are computationally expensive and
slow at inference. Recent adapter- and encoder-based methods attempt to reduce
this overhead but still depend on additional fine-tuning or large backbone
models for satisfactory results. In this work, we revisit an orthogonal
direction: fine-tuning-free personalization via Hypernetworks that predict
LoRA-adapted weights directly from subject images. Prior hypernetwork-based
approaches, however, suffer from costly data generation or unstable attempts to
mimic base model optimization trajectories. We address these limitations with
an end-to-end training objective, stabilized by a simple output regularization,
yielding reliable and effective hypernetworks. Our method removes the need for
per-subject optimization at test time while preserving both subject fidelity
and prompt alignment. To further enhance compositional generalization at
inference time, we introduce Hybrid-Model Classifier-Free Guidance (HM-CFG),
which combines the compositional strengths of the base diffusion model with the
subject fidelity of personalized models during sampling. Extensive experiments
on CelebA-HQ, AFHQ-v2, and DreamBench demonstrate that our approach achieves
strong personalization performance and highlights the promise of hypernetworks
as a scalable and effective direction for open-category personalization.

### 3. [Subsampled Randomized Fourier GaLore for Adapting Foundation Models in Depth-Driven Liver Landmark Segmentation](http://arxiv.org/pdf/2511.03163v1)

Authors: Yun-Chen Lin, Jiayuan Huang, Hanyuan Zhang, Sergi Kavtaradze, Matthew J. Clarkson, Mobarak I. Hoque

Accurate detection and delineation of anatomical structures in medical
imaging are critical for computer-assisted interventions, particularly in
laparoscopic liver surgery where 2D video streams limit depth perception and
complicate landmark localization. While recent works have leveraged monocular
depth cues for enhanced landmark detection, challenges remain in fusing RGB and
depth features and in efficiently adapting large-scale vision models to
surgical domains. We propose a depth-guided liver landmark segmentation
framework integrating semantic and geometric cues via vision foundation
encoders. We employ Segment Anything Model V2 (SAM2) encoder to extract RGB
features and Depth Anything V2 (DA2) encoder to extract depth-aware features.
To efficiently adapt SAM2, we introduce SRFT-GaLore, a novel low-rank gradient
projection method that replaces the computationally expensive SVD with a
Subsampled Randomized Fourier Transform (SRFT). This enables efficient
fine-tuning of high-dimensional attention layers without sacrificing
representational power. A cross-attention fusion module further integrates RGB
and depth cues. To assess cross-dataset generalization, we also construct a new
Laparoscopic Liver Surgical Dataset (LLSD) as an external validation benchmark.
On the public L3D dataset, our method achieves a 4.85% improvement in Dice
Similarity Coefficient and a 11.78-point reduction in Average Symmetric Surface
Distance compared to the D2GPLand. To further assess generalization capability,
we evaluate our model on LLSD dataset. Our model maintains competitive
performance and significantly outperforms SAM-based baselines, demonstrating
strong cross-dataset robustness and adaptability to unseen surgical
environments. These results demonstrate that our SRFT-GaLore-enhanced
dual-encoder framework enables scalable and precise segmentation under
real-time, depth-constrained surgical settings.

### 4. [SurgAnt-ViVQA: Learning to Anticipate Surgical Events through GRU-Driven Temporal Cross-Attention](http://arxiv.org/pdf/2511.03178v1)

Authors: Shreyas C. Dhake, Jiayuan Huang, Runlong He, Danyal Z. Khan, Evangelos B. Mazomenos, Sophia Bano, Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarak I. Hoque

Anticipating forthcoming surgical events is vital for real-time assistance in
endonasal transsphenoidal pituitary surgery, where visibility is limited and
workflow changes rapidly. Most visual question answering (VQA) systems reason
on isolated frames with static vision language alignment, providing little
support for forecasting next steps or instrument needs. Existing surgical VQA
datasets likewise center on the current scene rather than the near future. We
introduce PitVQA-Anticipation, the first VQA dataset designed for forward
looking surgical reasoning. It comprises 33.5 hours of operative video and
734,769 question answer pairs built from temporally grouped clips and expert
annotations across four tasks: predicting the future phase, next step, upcoming
instrument, and remaining duration. We further propose SurgAnt-ViVQA, a video
language model that adapts a large language model using a GRU Gated Temporal
Cross-Attention module. A bidirectional GRU encodes frame to frame dynamics,
while an adaptive gate injects visual context into the language stream at the
token level. Parameter efficient fine tuning customizes the language backbone
to the surgical domain. SurgAnt-ViVQA tested upon on PitVQA-Anticipation and
EndoVis datasets, surpassing strong image and video based baselines. Ablations
show that temporal recurrence and gated fusion drive most of the gains. A frame
budget study indicates a trade-off: 8 frames maximize fluency, whereas 32
frames slightly reduce BLEU but improve numeric time estimation. By pairing a
temporally aware encoder with fine grained gated cross-attention, SurgAnt-ViVQA
advances surgical VQA from retrospective description to proactive anticipation.
PitVQA-Anticipation offers a comprehensive benchmark for this setting and
highlights the importance of targeted temporal modeling for reliable, future
aware surgical assistance.

### 5. [PETWB-REP: A Multi-Cancer Whole-Body FDG PET/CT and Radiology Report Dataset for Medical Imaging Research](http://arxiv.org/pdf/2511.03194v1)

Authors: Le Xue, Gang Feng, Wenbo Zhang, Yichi Zhang, Lanlan Li, Shuqi Wang, Liling Peng, Sisi Peng, Xin Gao

Publicly available, large-scale medical imaging datasets are crucial for
developing and validating artificial intelligence models and conducting
retrospective clinical research. However, datasets that combine functional and
anatomical imaging with detailed clinical reports across multiple cancer types
remain scarce. Here, we present PETWB-REP, a curated dataset comprising
whole-body 18F-Fluorodeoxyglucose (FDG) Positron Emission Tomography/Computed
Tomography (PET/CT) scans and corresponding radiology reports from 490 patients
diagnosed with various malignancies. The dataset primarily includes common
cancers such as lung cancer, liver cancer, breast cancer, prostate cancer, and
ovarian cancer. This dataset includes paired PET and CT images, de-identified
textual reports, and structured clinical metadata. It is designed to support
research in medical imaging, radiomics, artificial intelligence, and
multi-modal learning.

### 6. [MvBody: Multi-View-Based Hybrid Transformer Using Optical 3D Body Scan for Explainable Cesarean Section Prediction](http://arxiv.org/pdf/2511.03212v1)

Authors: Ruting Cheng, Boyuan Feng, Yijiang Zheng, Chuhui Qiu, Aizierjiang Aiersilan, Joaquin A. Calderon, Wentao Zhao, Qing Pan, James K. Hahn

Accurately assessing the risk of cesarean section (CS) delivery is critical,
especially in settings with limited medical resources, where access to
healthcare is often restricted. Early and reliable risk prediction allows
better-informed prenatal care decisions and can improve maternal and neonatal
outcomes. However, most existing predictive models are tailored for in-hospital
use during labor and rely on parameters that are often unavailable in
resource-limited or home-based settings. In this study, we conduct a pilot
investigation to examine the feasibility of using 3D body shape for CS risk
assessment for future applications with more affordable general devices. We
propose a novel multi-view-based Transformer network, MvBody, which predicts CS
risk using only self-reported medical data and 3D optical body scans obtained
between the 31st and 38th weeks of gestation. To enhance training efficiency
and model generalizability in data-scarce environments, we incorporate a metric
learning loss into the network. Compared to widely used machine learning models
and the latest advanced 3D analysis methods, our method demonstrates superior
performance, achieving an accuracy of 84.62% and an Area Under the Receiver
Operating Characteristic Curve (AUC-ROC) of 0.724 on the independent test set.
To improve transparency and trust in the model's predictions, we apply the
Integrated Gradients algorithm to provide theoretically grounded explanations
of the model's decision-making process. Our results indicate that pre-pregnancy
weight, maternal age, obstetric history, previous CS history, and body shape,
particularly around the head and shoulders, are key contributors to CS risk
prediction.

### 7. [Diffusion-Guided Mask-Consistent Paired Mixing for Endoscopic Image Segmentation](http://arxiv.org/pdf/2511.03219v1)

Authors: Pengyu Jie, Wanquan Liu, Rui He, Yihui Wen, Deyu Meng, Chenqiang Gao

Augmentation for dense prediction typically relies on either sample mixing or
generative synthesis. Mixing improves robustness but misaligned masks yield
soft label ambiguity. Diffusion synthesis increases apparent diversity but,
when trained as common samples, overlooks the structural benefit of mask
conditioning and introduces synthetic-real domain shift. We propose a paired,
diffusion-guided paradigm that fuses the strengths of both. For each real
image, a synthetic counterpart is generated under the same mask and the pair is
used as a controllable input for Mask-Consistent Paired Mixing (MCPMix), which
mixes only image appearance while supervision always uses the original hard
mask. This produces a continuous family of intermediate samples that smoothly
bridges synthetic and real appearances under shared geometry, enlarging
diversity without compromising pixel-level semantics. To keep learning aligned
with real data, Real-Anchored Learnable Annealing (RLA) adaptively adjusts the
mixing strength and the loss weight of mixed samples over training, gradually
re-anchoring optimization to real data and mitigating distributional bias.
Across Kvasir-SEG, PICCOLO, CVC-ClinicDB, a private NPC-LES cohort, and ISIC
2017, the approach achieves state-of-the-art segmentation performance and
consistent gains over baselines. The results show that combining
label-preserving mixing with diffusion-driven diversity, together with adaptive
re-anchoring, yields robust and generalizable endoscopic segmentation.

### 8. [Transformer-Progressive Mamba Network for Lightweight Image Super-Resolution](http://arxiv.org/pdf/2511.03232v1)

Authors: Sichen Guo, Wenjie Li, Yuanyang Liu, Guangwei Gao, Jian Yang, Chia-Wen Lin

Recently, Mamba-based super-resolution (SR) methods have demonstrated the
ability to capture global receptive fields with linear complexity, addressing
the quadratic computational cost of Transformer-based SR approaches. However,
existing Mamba-based methods lack fine-grained transitions across different
modeling scales, which limits the efficiency of feature representation. In this
paper, we propose T-PMambaSR, a lightweight SR framework that integrates
window-based self-attention with Progressive Mamba. By enabling interactions
among receptive fields of different scales, our method establishes a
fine-grained modeling paradigm that progressively enhances feature
representation with linear complexity. Furthermore, we introduce an Adaptive
High-Frequency Refinement Module (AHFRM) to recover high-frequency details lost
during Transformer and Mamba processing. Extensive experiments demonstrate that
T-PMambaSR progressively enhances the model's receptive field and
expressiveness, yielding better performance than recent Transformer- or
Mamba-based methods while incurring lower computational cost. Our codes will be
released after acceptance.

### 9. [Decoupled Multi-Predictor Optimization for Inference-Efficient Model Tuning](http://arxiv.org/pdf/2511.03245v1)

Authors: Liwei Luo, Shuaitengyuan Li, Dongwei Ren, Qilong Wang, Pengfei Zhu, Qinghua Hu

Recently, remarkable progress has been made in large-scale pre-trained model
tuning, and inference efficiency is becoming more crucial for practical
deployment. Early exiting in conjunction with multi-stage predictors, when
cooperated with a parameter-efficient fine-tuning strategy, offers a
straightforward way to achieve an inference-efficient model. However, a key
challenge remains unresolved: How can early stages provide low-level
fundamental features to deep stages while simultaneously supplying high-level
discriminative features to early-stage predictors? To address this problem, we
propose a Decoupled Multi-Predictor Optimization (DMPO) method to effectively
decouple the low-level representative ability and high-level discriminative
ability in early stages. First, in terms of architecture, we introduce a
lightweight bypass module into multi-stage predictors for functional
decomposition of shallow features from early stages, while a high-order
statistics-based predictor is developed for early stages to effectively enhance
their discriminative ability. To reasonably train our multi-predictor
architecture, a decoupled optimization is proposed to allocate two-phase loss
weights for multi-stage predictors during model tuning, where the initial
training phase enables the model to prioritize the acquisition of
discriminative ability of deep stages via emphasizing representative ability of
early stages, and the latter training phase drives discriminative ability
towards earlier stages as much as possible. As such, our DMPO can effectively
decouple representative and discriminative abilities in early stages in terms
of architecture design and model optimization. Experiments across various
datasets and pre-trained backbones demonstrate that DMPO clearly outperforms
its counterparts when reducing computational cost.

### 10. [Enhancing Medical Image Segmentation via Heat Conduction Equation](http://arxiv.org/pdf/2511.03260v1)

Authors: Rong Wu, Yim-Sang Yu

Medical image segmentation has been significantly advanced by deep learning
architectures, notably U-Net variants. However, existing models struggle to
achieve efficient global context modeling and long-range dependency reasoning
under practical computational budgets simultaneously. In this work, we propose
a novel hybrid architecture utilizing U-Mamba with Heat Conduction Equation.
Our model combines Mamba-based state-space modules for efficient long-range
reasoning with Heat Conduction Operators (HCOs) in the bottleneck layers,
simulating frequency-domain thermal diffusion for enhanced semantic
abstraction. Experimental results on multimodal abdominal CT and MRI datasets
demonstrate that the proposed model consistently outperforms strong baselines,
validating its effectiveness and generalizability. It suggest that blending
state-space dynamics with heat-based global diffusion offers a scalable and
interpretable solution for medical segmentation tasks.

### Computers and Society

### 1. [AI as We Describe It: How Large Language Models and Their Applications in Health are Represented Across Channels of Public Discourse](http://arxiv.org/pdf/2511.03174v1)

Authors: Jiawei Zhou, Lei Zhang, Mei Li, Benjamin D Horne, Munmun De Choudhury

Representation shapes public attitudes and behaviors. With the arrival and
rapid adoption of LLMs, the way these systems are introduced will negotiate
societal expectations for their role in high-stakes domains like health. Yet it
remains unclear whether current narratives present a balanced view. We analyzed
five prominent discourse channels (news, research press, YouTube, TikTok, and
Reddit) over a two-year period on lexical style, informational content, and
symbolic representation. Discussions were generally positive and episodic, with
positivity increasing over time. Risk communication was unthorough and often
reduced to information quality incidents, while explanations of LLMs'
generative nature were rare. Compared with professional outlets, TikTok and
Reddit highlighted wellbeing applications and showed greater variations in tone
and anthropomorphism but little attention to risks. We discuss implications for
public discourse as a diagnostic tool in identifying literacy and governance
gaps, and for communication and design strategies to support more informed LLM
engagement.

### 2. [Retrofitters, pragmatists and activists: Public interest litigation for accountable automated decision-making](http://arxiv.org/pdf/2511.03211v1)

Authors: Henry Fraser, Zahra Stardust

This paper examines the role of public interest litigation in promoting
accountability for AI and automated decision-making (ADM) in Australia. Since
ADM regulatio faces geopolitical headwinds, effective governance will have to
rely at least in part on the enforcement of existing laws. Drawing on
interviews with Australian public interest litigators, technology policy
activists, and technology law scholars, the paper positions public interest
litigation as part of a larger ecosystem for transparency, accountability and
justice with respect to ADM. It builds on one participants's characterisation
of litigation about ADM as an exercise in legal retrofitting: adapting old laws
to new circumstances. The paper's primary contribution is to aggregate,
organise and present original insights on pragmatic strategies and tactics for
effective public interest litigation about ADM. Naturally, it also contends
with the limits of these strategies, and of the legal system. Where limits are,
however, capable of being overcome, the paper presents findings on urgent
needs: the enabling institutional arrangements without which effective
litigation and accountability will falter. The paper is relevant to law and
technology scholars; individuals and groups harmed by ADM; public interest
litigators and technology lawyers; civil society and advocacy organisations;
and policymakers.

### 3. [Do Androids Dream of Unseen Puppeteers? Probing for a Conspiracy Mindset in Large Language Models](http://arxiv.org/pdf/2511.03699v1)

Authors: Francesco Corso, Francesco Pierri, Gianmarco De Francisci Morales

In this paper, we investigate whether Large Language Models (LLMs) exhibit
conspiratorial tendencies, whether they display sociodemographic biases in this
domain, and how easily they can be conditioned into adopting conspiratorial
perspectives. Conspiracy beliefs play a central role in the spread of
misinformation and in shaping distrust toward institutions, making them a
critical testbed for evaluating the social fidelity of LLMs. LLMs are
increasingly used as proxies for studying human behavior, yet little is known
about whether they reproduce higher-order psychological constructs such as a
conspiratorial mindset. To bridge this research gap, we administer validated
psychometric surveys measuring conspiracy mindset to multiple models under
different prompting and conditioning strategies. Our findings reveal that LLMs
show partial agreement with elements of conspiracy belief, and conditioning
with socio-demographic attributes produces uneven effects, exposing latent
demographic biases. Moreover, targeted prompts can easily shift model responses
toward conspiratorial directions, underscoring both the susceptibility of LLMs
to manipulation and the potential risks of their deployment in sensitive
contexts. These results highlight the importance of critically evaluating the
psychological dimensions embedded in LLMs, both to advance computational social
science and to inform possible mitigation strategies against harmful uses.

### 4. [Deploying Rapid Damage Assessments from sUAS Imagery for Disaster Response](http://arxiv.org/pdf/2511.03132v1)

Authors: Thomas Manzini, Priyankari Perali, Robin R. Murphy

This paper presents the first AI/ML system for automating building damage
assessment in uncrewed aerial systems (sUAS) imagery to be deployed
operationally during federally declared disasters (Hurricanes Debby and
Helene). In response to major disasters, sUAS teams are dispatched to collect
imagery of the affected areas to assess damage; however, at recent disasters,
teams collectively delivered between 47GB and 369GB of imagery per day,
representing more imagery than can reasonably be transmitted or interpreted by
subject matter experts in the disaster scene, thus delaying response efforts.
To alleviate this data avalanche encountered in practice, computer vision and
machine learning techniques are necessary. While prior work has been deployed
to automatically assess damage in satellite imagery, there is no current state
of practice for sUAS-based damage assessment systems, as all known work has
been confined to academic settings. This work establishes the state of practice
via the development and deployment of models for building damage assessment
with sUAS imagery. The model development involved training on the largest known
dataset of post-disaster sUAS aerial imagery, containing 21,716 building damage
labels, and the operational training of 91 disaster practitioners. The best
performing model was deployed during the responses to Hurricanes Debby and
Helene, where it assessed a combined 415 buildings in approximately 18 minutes.
This work contributes documentation of the actual use of AI/ML for damage
assessment during a disaster and lessons learned to the benefit of the AI/ML
research and user communities.

### 5. [Hybrid Fact-Checking that Integrates Knowledge Graphs, Large Language Models, and Search-Based Retrieval Agents Improves Interpretable Claim Verification](http://arxiv.org/pdf/2511.03217v1)

Authors: Shaghayegh Kolli, Richard Rosenbaum, Timo Cavelius, Lasse Strothe, Andrii Lata, Jana Diesner

Large language models (LLMs) excel in generating fluent utterances but can
lack reliable grounding in verified information. At the same time,
knowledge-graph-based fact-checkers deliver precise and interpretable evidence,
yet suffer from limited coverage or latency. By integrating LLMs with knowledge
graphs and real-time search agents, we introduce a hybrid fact-checking
approach that leverages the individual strengths of each component. Our system
comprises three autonomous steps: 1) a Knowledge Graph (KG) Retrieval for rapid
one-hop lookups in DBpedia, 2) an LM-based classification guided by a
task-specific labeling prompt, producing outputs with internal rule-based
logic, and 3) a Web Search Agent invoked only when KG coverage is insufficient.
Our pipeline achieves an F1 score of 0.93 on the FEVER benchmark on the
Supported/Refuted split without task-specific fine-tuning. To address Not
enough information cases, we conduct a targeted reannotation study showing that
our approach frequently uncovers valid evidence for claims originally labeled
as Not Enough Information (NEI), as confirmed by both expert annotators and LLM
reviewers. With this paper, we present a modular, opensource fact-checking
pipeline with fallback strategies and generalization across datasets.

### 6. [Watermarking Large Language Models in Europe: Interpreting the AI Act in Light of Technology](http://arxiv.org/pdf/2511.03641v1)

Authors: Thomas Souverain

To foster trustworthy Artificial Intelligence (AI) within the European Union,
the AI Act requires providers to mark and detect the outputs of their
general-purpose models. The Article 50 and Recital 133 call for marking methods
that are ''sufficiently reliable, interoperable, effective and robust''. Yet,
the rapidly evolving and heterogeneous landscape of watermarks for Large
Language Models (LLMs) makes it difficult to determine how these four standards
can be translated into concrete and measurable evaluations. Our paper addresses
this challenge, anchoring the normativity of European requirements in the
multiplicity of watermarking techniques. Introducing clear and distinct
concepts on LLM watermarking, our contribution is threefold. (1) Watermarking
Categorisation: We propose an accessible taxonomy of watermarking methods
according to the stage of the LLM lifecycle at which they are applied - before,
during, or after training, and during next-token distribution or sampling. (2)
Watermarking Evaluation: We interpret the EU AI Act's requirements by mapping
each criterion with state-of-the-art evaluations on robustness and
detectability of the watermark, and of quality of the LLM. Since
interoperability remains largely untheorised in LLM watermarking research, we
propose three normative dimensions to frame its assessment. (3) Watermarking
Comparison: We compare current watermarking methods for LLMs against the
operationalised European criteria and show that no approach yet satisfies all
four standards. Encouraged by emerging empirical tests, we recommend further
research into watermarking directly embedded within the low-level architecture
of LLMs.

### 7. [From Measurement to Expertise: Empathetic Expert Adapters for Context-Based Empathy in Conversational AI Agents](http://arxiv.org/pdf/2511.03143v1)

Authors: Erfan Shayegani, Jina Suh, Andy Wilson, Nagu Rangan, Javier Hernandez

Empathy is a critical factor in fostering positive user experiences in
conversational AI. While models can display empathy, it is often generic rather
than tailored to specific tasks and contexts. In this work, we introduce a
novel framework for developing and evaluating context-specific empathetic large
language models (LLMs). We first analyze a real-world conversational dataset
consisting of 672 multi-turn conversations across 8 tasks, revealing
significant differences in terms of expected and experienced empathy before and
after the conversations, respectively. To help minimize this gap, we develop a
synthetic multi-turn conversational generation pipeline and steer responses
toward our defined empathy patterns based on the context that more closely
matches users' expectations. We then train empathetic expert adapters for
context-specific empathy that specialize in varying empathy levels based on the
recognized task. Our empirical results demonstrate a significant gap reduction
of 72.66% between perceived and desired empathy with scores increasing by an
average factor of 2.43 as measured by our metrics and reward models.
Additionally, our trained empathetic expert adapters demonstrate superior
effectiveness in preserving empathy patterns throughout conversation turns,
outperforming system prompts, which tend to dramatically diminish in impact as
conversations lengthen.

### 8. [Two thousand years of the oracle problem. Insights from Ancient Delphi on the future of blockchain oracles](http://arxiv.org/pdf/2511.03319v1)

Authors: Giulio Caldarelli, Massimiliano Ornaghi

The oracle problem refers to the inability of an agent to know if the
information coming from an oracle is authentic and unbiased. In ancient times,
philosophers and historians debated on how to evaluate, increase, and secure
the reliability of oracle predictions, particularly those from Delphi, which
pertained to matters of state. Today, we refer to data carriers for automatic
machines as oracles, but establishing a secure channel between these oracles
and the real world still represents a challenge. Despite numerous efforts, this
problem remains mostly unsolved, and the recent advent of blockchain oracles
has added a layer of complexity because of the decentralization of blockchains.
This paper conceptually connects Delphic and modern blockchain oracles,
developing a comparative framework. Leveraging blockchain oracle taxonomy,
lexical analysis is also performed on 167 Delphic queries to shed light on the
relationship between oracle answer quality and question type. The presented
framework aims first at revealing commonalities between classical and
computational oracles and then at enriching the oracle analysis within each
field. This study contributes to the computer science literature by proposing
strategies to improve the reliability of blockchain oracles based on insights
from Delphi and to classical literature by introducing a framework that can
also be applied to interpret and classify other ancient oracular mechanisms.

### Databases

### 1. [Formalizing ETLT and ELTL Design Patterns and Proposing Enhanced Variants: A Systematic Framework for Modern Data Engineering](http://arxiv.org/pdf/2511.03393v1)

Authors: Chiara Rucco, Motaz Saad, Antonella Longo

Traditional ETL and ELT design patterns struggle to meet modern requirements
of scalability, governance, and real-time data processing. Hybrid approaches
such as ETLT (Extract-Transform-Load-Transform) and ELTL
(Extract-Load-Transform-Load) are already used in practice, but the literature
lacks best practices and formal recognition of these approaches as design
patterns. This paper formalizes ETLT and ELTL as reusable design patterns by
codifying implicit best practices and introduces enhanced variants, ETLT++ and
ELTL++, to address persistent gaps in governance, quality assurance, and
observability. We define ETLT and ELTL patterns systematically within a design
pattern framework, outlining their structure, trade-offs, and use cases.
Building on this foundation, we extend them into ETLT++ and ELTL++ by embedding
explicit contracts, versioning, semantic curation, and continuous monitoring as
mandatory design obligations. The proposed framework offers practitioners a
structured roadmap to build auditable, scalable, and cost-efficient pipelines,
unifying quality enforcement, lineage, and usability across multi-cloud and
real-time contexts. By formalizing ETLT and ELTL, and enhancing them through
ETLT++ and ELTL++, this work bridges the gap between ad hoc practice and
systematic design, providing a reusable foundation for modern, trustworthy data
engineering.

### 2. [In-Memory Indexing and Querying of Provenance in Data Preparation Pipelines](http://arxiv.org/pdf/2511.03480v1)

Authors: Khalid Belhajjame, Haroun Mezrioui, Yuyan Zhao

Data provenance has numerous applications in the context of data preparation
pipelines. It can be used for debugging faulty pipelines, interpreting results,
verifying fairness, and identifying data quality issues, which may affect the
sources feeding the pipeline execution. In this paper, we present an indexing
mechanism to efficiently capture and query pipeline provenance. Our solution
leverages tensors to capture fine-grained provenance of data processing
operations, using minimal memory. In addition to record-level lineage
relationships, we provide finer granularity at the attribute level. This is
achieved by augmenting tensors, which capture retrospective provenance, with
prospective provenance information, drawing connections between input and
output schemas of data processing operations. We demonstrate how these two
types of provenance (retrospective and prospective) can be combined to answer a
broad range of provenance queries efficiently, and show effectiveness through
evaluation exercises using both real and synthetic data.

### 3. [Analytical Queries for Unstructured Data](http://arxiv.org/pdf/2511.03489v1)

Authors: Daniel Kang

Unstructured data, in the form of text, images, video, and audio, is produced
at exponentially higher rates. In tandem, machine learning (ML) methods have
become increasingly powerful at analyzing unstructured data. Modern ML methods
can now detect objects in images, understand actions in videos, and even
classify complex legal texts based on legal intent. Combined, these trends make
it increasingly feasible for analysts and researchers to automatically
understand the "real world." However, there are major challenges in deploying
these techniques: 1) executing queries efficiently given the expense of ML
methods, 2) expressing queries over bespoke forms of data, and 3) handling
errors in ML methods.
  In this monograph, we discuss challenges and advances in data management
systems for unstructured data using ML, with a particular focus on video
analytics. Using ML to answer queries introduces new challenges.First, even
turning user intent into queries can be challenging: it is not obvious how to
express a query of the form "select instances of cars turning left." Second, ML
models can be orders of magnitude more expensive compared to processing
traditional structured data. Third, ML models and the methods to accelerate
analytics with ML models can be error-prone.
  Recent work in the data management community has aimed to address all of
these challenges. Users can now express queries via user-defined functions,
opaquely through standard structured schemas, and even by providing examples.
Given a query, recent work focuses on optimizing queries by approximating
expensive "gold" methods with varying levels of guarantees. Finally, to handle
errors in ML models, recent work has focused on applying outlier and drift
detection to data analytics with ML.

### 4. [HERP: Hardware for Energy Efficient and Realtime DB Search and Cluster Expansion in Proteomics](http://arxiv.org/pdf/2511.03437v1)

Authors: Md Mizanur Rahaman Nayan, Zheyu Li, Flavio Ponzina, Sumukh Pinge, Tajana Rosing, Azad J. Naeemi

Database (DB) search and clustering are fundamental in proteomics but
conventional full clustering and search approaches demand high resources and
incur long latency. We propose a lightweight incremental clustering and highly
parallelizable DB search platform tailored for resource-constrained
environments, delivering low energy and latency without compromising
performance. By leveraging mass-spectrometry insights, we employ bucket-wise
parallelization and query scheduling to reduce latency. A one-time hardware
initialization with pre-clustered proteomics data enables continuous DB search
and local re-clustering, offering a more practical and efficient alternative to
clustering from scratch. Heuristics from pre-clustered data guide incremental
clustering, accelerating the process by 20x with only a 0.3% increase in
clustering error. DB search results overlap by 96% with state-of-the-art tools,
validating search quality. The hardware leverages a 3T 2M T J SOT-CAM at the
7nm node with a compute-in-memory design. For the human genome draft dataset
(131GB), setup requires 1.19mJ for 2M spectra, while a 1000 query search
consumes 1.1{\mu}J. Bucket-wise parallelization further achieves 100x speedup.

### Distributed, Parallel, and Cluster Computing

### 1. [UMDAM: A Unified Data Layout and DRAM Address Mapping for Heterogenous NPU-PIM](http://arxiv.org/pdf/2511.03293v1)

Authors: Hai Huang, Xuhong Qiang, Weisheng Zhao, Chenchen Liu

Large Language Models (LLMs) are increasingly deployed on edge devices with
Neural Processing Units (NPUs), yet the decode phase remains memory-intensive,
limiting performance. Processing-in-Memory (PIM) offers a promising solution,
but co-executing NPU-PIM systems face challenges such as data layout
mismatches, bandwidth loss, and redundant storage. To address these issues, we
propose UMDAM, a unified memory-affinity data layout and DRAM address mapping
scheme tailored for NPU-PIM co-execution. UMDAM employs a column-major,
tile-based layout and a configurable DRAM mapping strategy to ensure
compatibility with NPU computation while maximizing PIM efficiency -- without
introducing extra memory overhead or bandwidth loss. Comprehensive evaluations
on OPT models demonstrate that UMDAM reduces time-to-first-token (TTFT) by up
to 3.0x and time-to-last-token (TTLT) by 2.18x, significantly improving
end-to-end LLM inference efficiency on edge devices.

### 2. [Stone Duality Proofs for Colorless Distributed Computability Theorems](http://arxiv.org/pdf/2511.03609v1)

Authors: Cameron Calk, Emmanuel Godard

We introduce a new topological encoding by spectral spaces of executions of
  round-based full-information adversaries, a model of distributed computations
that is functorially presented and that
  contains many message adversaries. We give a characterization of the
solvability of colorless tasks against compact adversaries.
  Message adversaries are distributed
  models that are known to be very expressive despite being
  round-based and crash-free. Colorless tasks are
  an important class of distributed tasks. For a colorless task, the
  specification does not depend upon the multiplicity of input or
  output values, like the ubiquitous agreement tasks.
  Therefore, our result is a significant
  step toward unifying topological methods in distributed computing.
  The main insight is to consider global states obtained after finite
executions of a distributed protocol
  not as abstract
  simplicial complexes as previously done, but as spectral
  spaces, considering the Alexandrov topology on the faces poset. Given
  an adversary $\mathcal M$ with a set of inputs $\mathcal I$,
  we define a limit object $\Pi^\infty_\mathcal M(\mathcal I)$
  by projective limit in the category of spectral spaces. We derive a new
general distributed computability
  theorem using Stone duality: there exists an algorithm solving a colorless
task $(\mathcal I,\mathcal O,\Delta)$
  against the compact adversary $\mathcal M$ if and only if there exists a
spectral
  map $f:\Pi^\infty_\mathcal M(\mathcal I)\longrightarrow\mathcal O$ compatible
with $\Delta$.
  From this general characterization are derived many known colorless
computability
  theorems.
  Quite surprisingly, colored and uncolored models have the same
  computability power (they solve the same tasks). Our new proofs give
  topological reasons for this equivalence, previously known through
  algorithmic reductions.

### 3. [A General Input-Dependent Colorless Computability Theorem and Applications to Core-Dependent Adversaries](http://arxiv.org/pdf/2511.03662v1)

Authors: Yannis Coutouly, Emmanuel Godard

Distributed computing tasks can be presented with a triple $(\I,\Ou,\Delta)$.
The solvability of a colorless task on the Iterated Immediate Snapshot model
(IIS) has been characterized by the Colorless Computability Theorem
\cite[Th.4.3.1]{HKRbook}. A recent paper~\cite{CG-24} generalizes this theorem
for any message adversaries $\ma \subseteq IIS$ by geometric methods. In 2001,
Most\'efaoui, Rajsbaum, Raynal, and Roy \cite{condbased} introduced
\emph{condition-based adversaries}. This setting considers a particular
adversary that will be applied only to a subset of input configurations. In
this setting, they studied the $k$-set agreement task with condition-based
$t$-resilient adversaries and obtained a sufficient condition on the conditions
that make $k$-Set Agreement solvable. In this paper we have three
contributions:
  -We generalize the characterization of~\cite{CG-24} to \emph{input-dependent}
adversaries, which means that the adversaries can change depending on the input
configuration.
  - We show that core-resilient adversaries of $IIS_n$ have the same
computability power as the core-resilient adversaries of $IIS_n$ where crashes
only happen at the start.
  - Using the two previous contributions, we provide a necessary and sufficient
characterization of the condition-based, core-dependent adversaries that can
solve $k$-Set Agreement. We also distinguish four settings that may appear when
presenting a distributed task as $(\I,\Ou,\Delta)$. Finally, in a later
section, we present structural properties on the carrier map $\Delta$. Such
properties allow simpler proof, without changing the computability power of the
task. Most of the proofs in this article leverage the topological framework
used in distributed computing by using simple geometric constructions.

### 4. [Investigating the Impact of Isolation on Synchronized Benchmarks](http://arxiv.org/pdf/2511.03533v1)

Authors: Nils Japke, Furat Hamdan, Diana Baumann, David Bermbach

Benchmarking in cloud environments suffers from performance variability from
multi-tenant resource contention. Duet benchmarking mitigates this by running
two workload versions concurrently on the same VM, exposing them to identical
external interference. However, intra-VM contention between synchronized
workloads necessitates additional isolation mechanisms.
  This work evaluates three such strategies: cgroups and CPU pinning, Docker
containers, and Firecracker MicroVMs. We compare all strategies with an
unisolated baseline experiment, by running benchmarks with a duet setup
alongside a noise generator. This noise generator "steals" compute resources to
degrade performance measurements.
  All experiments showed different latency distributions while under the
effects of noise generation, but results show that process isolation generally
lowered false positives, except for our experiments with Docker containers.
Even though Docker containers rely internally on cgroups and CPU pinning, they
were more susceptible to performance degradation due to noise influence.
Therefore, we recommend to use process isolation for synchronized workloads,
with the exception of Docker containers.

### 5. [SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators](http://arxiv.org/pdf/2511.03092v1)

Authors: Jonathan Li, Nasim Farahini, Evgenii Iuliugin, Magnus Vesterlund, Christian Haggstrom, Guangtao Wang, Shubhangi Upasani, Ayush Sachdeva, Rui Li, Faline Fu, Chen Wu, Ayesha Siddiqua, John Long, Tuowen Zhao, Matheen Musaddiq, Hakan Zeffer, Yun Du, Mingran Wang, Qinghua Li, Bo Li, Urmish Thakker, Raghu Prabhakar

The proliferation of 100B+ parameter Large Language Models (LLMs) with 100k+
context length support have resulted in increasing demands for on-chip memory
to support large KV caches. Techniques such as StreamingLLM and SnapKV
demonstrate how to control KV cache size while maintaining model accuracy. Yet,
these techniques are not commonly used within industrial deployments using
frameworks like vLLM or SGLang. The reason is twofold: on one hand, the static
graphs and continuous batching methodology employed by these frameworks make it
difficult to admit modifications to the standard multi-head attention
algorithm, while on the other hand, the accuracy implications of such
techniques on modern instruction-following and reasoning models are not well
understood, obfuscating the need for implementing these techniques. In this
paper, we explore these accuracy implications on Llama-3.1-8B-Instruct and
DeepSeek-R1, and develop SnapStream, a KV cache compression method that can be
deployed at scale. We demonstrate the efficacy of SnapStream in a 16-way
tensor-parallel deployment of DeepSeek-671B on SambaNova SN40L accelerators
running at 128k context length and up to 1832 tokens per second in a real
production setting. SnapStream enables $4\times$ improved on-chip memory usage
and introduces minimal accuracy degradation on LongBench-v2, AIME24 and
LiveCodeBench. To the best of our knowledge, this is the first implementation
of sparse KV attention techniques deployed in a production inference system
with static graphs and continuous batching.

### 6. [Characterising Global Platforms: Centralised, Decentralised, Federated, and Grassroots](http://arxiv.org/pdf/2511.03286v1)

Authors: Ehud Shapiro

Global digital platforms are software systems designed to serve entire
populations, with some already serving billions of people. We propose atomic
transactions-based multiagent transition systems and protocols as a formal
framework to study them; introduce essential agents -- minimal sets of agents
the removal of which makes communication impossible; and show that the
cardinality of essential agents partitions all global platforms into four
classes:
  1. Centralised -- one (the server)
  2. Decentralised -- finite $>1$ (bootstrap nodes)
  3. Federated -- infinite but not universal (all servers)
  4. Grassroots -- universal (all agents)
  Our illustrative formal example is a global social network, for which we
provide centralised, decentralised, federated, and grassroots specifications
via multiagent atomic transactions, and prove they satisfy basic correctness
properties. We discuss informally additional global platforms -- currencies,
``sharing economy'' apps, AI, and more. While this may be the first
characterisation of centralised, decentralised, and federated global platforms,
grassroots platforms have been formally defined previously, but using different
notions. Here, we prove that their original definition implies that all agents
are essential, placing grassroots platforms in a distinct class within the
broader formal context that includes all global platforms. This work provides
the first mathematical framework for classifying any global platform --
existing or imagined -- by providing a multiagent atomic-transactions
specification of it and determining the cardinality of the minimal set of
essential agents in the ensuing multiagent protocol. It thus

### Digital Libraries

### 1. [A Study on Library Resources with Services Satisfaction based on Library Users Affiliated Colleges to Solapur University](http://arxiv.org/pdf/2511.03209v1)

Authors: Patel Adam Burhansab, M Sadik Batcha, Muneer Ahmad

The main aim of this study was to assess and evaluate user satisfaction with
library resources and services among library users associated with Solapur
University. The current research shows the level of users satisfaction with
different library resources and services offered by college libraries. The
research found that a vast number of respondents were pleased with library
facilities and services. The research is designed to achieve users satisfaction
in the library to investigate the level of satisfaction towards library
resources and services with regards to 26 colleges of Solapur University based
in Maharashtra. Information in the form of data has been collected from
colleges and on the basis of users results; analysis needs to analyze users
satisfaction.

### 2. [Russian Contribution to Coronary Artery Disease Research: A Scientometric Mapping of Publications](http://arxiv.org/pdf/2511.03215v1)

Authors: Muneer Ahmad, M Sadik Batcha

The present study attempts to highlight the research output generated in
Russia in coronary artery disease (CAD) research during the period 1990-2019 to
understand the distribution of research output, top journals for publications,
and most prolific authors, authorship pattern, and citation pattern. This study
is based on secondary data extracted from the Science Citation Index (SCI),
which is an integral component of the Web of Science. Descriptive and
inferential statistical techniques were applied in the study. There were 5058
articles by Russian scholars in coronary artery disease during 1990-2019; they
preferred to publish in Russian journals. The research contributions were in
the form of research articles, meeting abstracts and reviews with a consistent
drop in the number of editorial material and article; proceedings paper with
time. Co-authorship was the norm in coronary artery disease research, with a
steady increase in the number of multi-author documents in recent years.

### Discrete Mathematics

### 1. [Extension of the Gyárfás-Sumner conjecture to signed graphs](http://arxiv.org/pdf/2511.03335v1)

Authors: Guillaume Aubian, Allen Ibiapina, Luis Kuffner, Reza Naserasr, Cyril Pujol, Cléophée Robin, Huan Zhou

The balanced chromatic number of a signed graph G is the minimum number of
balanced sets that cover all vertices of G. Studying structural conditions
which imply bounds on the balanced chromatic number of signed graphs is among
the most fundamental problems in graph theory. In this work, we initiate the
study of coloring hereditary classes of signed graphs. More precisely, we say
that a set F = {F_1, F_2, ..., F_l} is a GS (for Gy\'arf\'as-Sumner) set if
there exists a constant c such that signed graphs with no induced subgraph
switching equivalent to a member of F admit a balanced c-coloring. The focus of
this work is to study GS sets of order 2. We show that if F is a GS set of
order 2, then F_1 is either (K_3, -) or (K_4, -), and F_2 is a linear forest.
In the case of F_1 = (K_3, -), we show that any choice of a linear forest for
F_2 works. In the case of F_1 = (K_4, -), we show that if each connected
component of F_2 is a path of length at most 4, then {F_1, F_2} is a GS set.

### 2. [Characterizations of undirected 2-quasi best match graphs](http://arxiv.org/pdf/2511.03592v1)

Authors: Annachiara Korchmaros, Guillaume E. Scholz, Peter F. Stadler

Bipartite best match graphs (BMG) and their generalizations arise in
mathematical phylogenetics as combinatorial models describing evolutionary
relationships among related genes in a pair of species. In this work, we
characterize the class of \emph{undirected 2-quasi-BMGs} (un2qBMGs), which form
a proper subclass of the $P_6$-free chordal bipartite graphs. We show that
un2qBMGs are exactly the class of bipartite graphs free of $P_6$, $C_6$, and
the eight-vertex Sunlet$_4$ graph. Equivalently, a bipartite graph $G$ is
un2qBMG if and only if every connected induced subgraph contains a
``heart-vertex'' which is adjacent to all the vertices of the opposite color.
We further provide a $O(|V(G)|^3)$ algorithm for the recognition of un2qBMGs
that, in the affirmative case, constructs a labeled rooted tree that
``explains'' $G$. Finally, since un2qBMGs coincide with the $(P_6,C_6)$-free
bi-cographs, they can also be recognized in linear time.

### Data Structures and Algorithms

### 1. [A Branch-and-Bound Approach for Maximum Low-Diameter Dense Subgraph Problems](http://arxiv.org/pdf/2511.03157v1)

Authors: Yi Zhoua, Chunyu Luoa, Zhengren Wangb, Zhang-Hua Fuc

A graph with $n$ vertices is an $f(\cdot)$-dense graph if it has at least
$f(n)$ edges, $f(\cdot)$ being a well-defined function. The notion
$f(\cdot)$-dense graph encompasses various clique models like $\gamma$-quasi
cliques, $k$-defective cliques, and dense cliques, arising in cohesive subgraph
extraction applications. However, the $f(\cdot)$-dense graph may be
disconnected or weakly connected. To conquer this, we study the problem of
finding the largest $f(\cdot)$-dense subgraph with a diameter of at most two in
the paper. Specifically, we present a decomposition-based branch-and-bound
algorithm to optimally solve this problem. The key feature of the algorithm is
a decomposition framework that breaks the graph into $n$ smaller subgraphs,
allowing independent searches in each subgraph. We also introduce decomposition
strategies including degeneracy and two-hop degeneracy orderings, alongside a
branch-and-bound algorithm with a novel sorting-based upper bound to solve each
subproblem. Worst-case complexity for each component is provided. Empirical
results on 139 real-world graphs under two $f(\cdot)$ functions show our
algorithm outperforms the MIP solver and pure branch-and-bound, solving nearly
twice as many instances optimally within one hour.

### 2. [Optimal Stopping with a Predicted Prior](http://arxiv.org/pdf/2511.03289v1)

Authors: Tian Bai, Zhiyi Huang, Chui Shan Lee, Dongchen Li

There are two major models of value uncertainty in the optimal stopping
literature: the secretary model, which assumes no prior knowledge, and the
prophet inequality model, which assumes full information about value
distributions. In practice, decision makers often rely on machine-learned
priors that may be erroneous. Motivated by this gap, we formulate the model of
optimal stopping with a predicted prior to design algorithms that are both
consistent, exploiting the prediction when accurate, and robust, retaining
worst-case guarantees when it is not.
  Existing secretary and prophet inequality algorithms are either pessimistic
in consistency or not robust to misprediction. A randomized combination only
interpolates their guarantees linearly. We show that a family of bi-criteria
algorithms achieves improved consistency-robustness trade-offs, both for
maximizing the expected accepted value and for maximizing the probability of
accepting the maximum value. We further prove that for the latter objective, no
algorithm can simultaneously match the best prophet inequality algorithm in
consistency, and the best secretary algorithm in robustness.

### 3. [Improved Online Load Balancing in the Two-Norm](http://arxiv.org/pdf/2511.03345v1)

Authors: Sander Borst, Danish Kashaev

We study the online load balancing problem on unrelated machines, with the
objective of minimizing the square of the $\ell_2$ norm of the loads on the
machines. The greedy algorithm of Awerbuch et al. (STOC'95) is optimal for
deterministic algorithms and achieves a competitive ratio of $3 + 2 \sqrt{2}
\approx 5.828$, and an improved $5$-competitive randomized algorithm based on
independent rounding has been shown by Caragiannis (SODA'08). In this work, we
present the first algorithm breaking the barrier of $5$ on the competitive
ratio, achieving a bound of $4.9843$. To obtain this result, we use a new
primal-dual framework to analyze this problem based on a natural semidefinite
programming relaxation, together with an online implementation of a correlated
randomized rounding procedure of Im and Shadloo (SODA'20). This novel
primal-dual framework also yields new, simple and unified proofs of the
competitive ratio of the $(3 + 2 \sqrt{2})$-competitive greedy algorithm, the
$5$-competitive randomized independent rounding algorithm, and that of a new
$4$-competitive optimal fractional algorithm. We also provide lower bounds
showing that the previous best randomized algorithm is optimal among
independent rounding algorithms, that our new fractional algorithm is optimal,
and that a simple greedy algorithm is optimal for the closely related online
scheduling problem $R || \sum w_j C_j$.

### 4. [Dynamic Meta-Kernelization](http://arxiv.org/pdf/2511.03461v1)

Authors: Christian Bertram, Deborah Haun, Mads Vestergaard Jensen, Tuukka Korhonen

Kernelization studies polynomial-time preprocessing algorithms. Over the last
20 years, the most celebrated positive results of the field have been linear
kernels for classical NP-hard graph problems on sparse graph classes. In this
paper, we lift these results to the dynamic setting.
  As the canonical example, Alber, Fellows, and Niedermeier [J. ACM 2004] gave
a linear kernel for dominating set on planar graphs. We provide the following
dynamic version of their kernel: Our data structure is initialized with an
$n$-vertex planar graph $G$ in $O(n \log n)$ amortized time, and, at
initialization, outputs a planar graph $K$ with $\mathrm{OPT}(K) =
\mathrm{OPT}(G)$ and $|K| = O(\mathrm{OPT}(G))$, where $\mathrm{OPT}(\cdot)$
denotes the size of a minimum dominating set. The graph $G$ can be updated by
insertions and deletions of edges and isolated vertices in $O(\log n)$
amortized time per update, under the promise that it remains planar. After each
update to $G$, the data structure outputs $O(1)$ updates to $K$, maintaining
$\mathrm{OPT}(K) = \mathrm{OPT}(G)$, $|K| = O(\mathrm{OPT}(G))$, and planarity
of $K$.
  Furthermore, we obtain similar dynamic kernelization algorithms for all
problems satisfying certain conditions on (topological-)minor-free graph
classes. Besides kernelization, this directly implies new dynamic
constant-approximation algorithms and improvements to dynamic FPT algorithms
for such problems.
  Our main technical contribution is a dynamic data structure for maintaining
an approximately optimal protrusion decomposition of a dynamic
topological-minor-free graph. Protrusion decompositions were introduced by
Bodlaender, Fomin, Lokshtanov, Penninkx, Saurabh, and Thilikos [J. ACM 2016],
and have since developed into a part of the core toolbox in kernelization and
parameterized algorithms.

### 5. [Online Flow Time Minimization: Tight Bounds for Non-Preemptive Algorithms](http://arxiv.org/pdf/2511.03485v1)

Authors: Yutong Geng, Enze Sun, Zonghan Yang, Yuhao Zhang

This paper studies the classical online scheduling problem of minimizing
total flow time for $n$ jobs on $m$ identical machines. Prior work often cites
the $\Omega(n)$ lower bound for non-preemptive algorithms to argue for the
necessity of preemption or resource augmentation, which shows the trivial
$O(n)$-competitive greedy algorithm is tight. However, this lower bound applies
only to \emph{deterministic} algorithms in the \emph{single-machine} case,
leaving several fundamental questions unanswered. Can randomness help in the
non-preemptive setting, and what is the optimal online deterministic algorithm
when $m \geq 2$? We resolve both questions. We present a polynomial-time
randomized algorithm with competitive ratio $\Theta(\sqrt{n/m})$ and prove a
matching randomized lower bound, settling the randomized non-preemptive setting
for every $m$. This also improves the best-known offline approximation ratio
from $O(\sqrt{n/m}\log(n/m))$ to $O(\sqrt{n/m})$. On the deterministic side, we
present a non-preemptive algorithm with competitive ratio
$O(n/m^{2}+\sqrt{n/m}\log m)$ and prove a nearly matching lower bound.
  Our framework also extends to the kill-and-restart model, where we reveal a
sharp transition of deterministic algorithms: we design an asymptotically
optimal algorithm with the competitive ratio $O(\sqrt{n/m})$ for $m\ge 2$, yet
establish a strong $\Omega(n/\log n)$ lower bound for $m=1$. Moreover, we show
that randomization provides no further advantage, as the lower bound coincides
with that of the non-preemptive setting.
  While our main results assume prior knowledge of $n$, we also investigate the
setting where $n$ is unknown. We show kill-and-restart is powerful enough to
break the $O(n)$ barrier for $m \geq 2$ even without knowing $n$. Conversely,
we prove randomization alone is insufficient, as no algorithm can achieve an
$o(n)$ competitive ratio in this setting.

### 6. [Randomized Rounding over Dynamic Programs](http://arxiv.org/pdf/2511.03490v1)

Authors: Etienne Bamas, Shi Li, Lars Rohwedder

We show that under mild assumptions for a problem whose solutions admit a
dynamic programming-like recurrence relation, we can still find a solution
under additional packing constraints, which need to be satisfied approximately.
The number of additional constraints can be very large, for example, polynomial
in the problem size. Technically, we reinterpret the dynamic programming
subproblems and their solutions as a network design problem. Inspired by
techniques from, for example, the Directed Steiner Tree problem, we construct a
strong LP relaxation, on which we then apply randomized rounding. Our
approximation guarantees on the packing constraints have roughly the form of a
$(n^{\epsilon} \mathrm{polylog}\ n)$-approximation in time $n^{O(1/\epsilon)}$,
for any $\epsilon > 0$. By setting $\epsilon=\log \log n/\log n$, we obtain a
polylogarithmic approximation in quasi-polynomial time, or by setting
$\epsilon$ as a constant, an $n^\epsilon$-approximation in polynomial time.
  While there are necessary assumptions on the form of the DP, it is general
enough to capture many textbook dynamic programs from Shortest Path to Longest
Common Subsequence. Our algorithm then implies that we can impose additional
constraints on the solutions to these problems. This allows us to model various
problems from the literature in approximation algorithms, many of which were
not thought to be connected to dynamic programming. In fact, our result can
even be applied indirectly to some problems that involve covering instead of
packing constraints, for example, the Directed Steiner Tree problem, or those
that do not directly follow a recurrence relation, for example, variants of the
Matching problem.

### 7. [Engineering Algorithms for $\ell$-Isolated Maximal Clique Enumeration](http://arxiv.org/pdf/2511.03525v1)

Authors: Marco D'Elia, Irene Finocchi, Maurizio Patrignani

Maximal cliques play a fundamental role in numerous application domains,
where their enumeration can prove extremely useful. Yet their sheer number,
even in sparse real-world graphs, can make them impractical to be exploited
effectively. To address this issue, one approach is to enumerate
$\ell$-isolated maximal cliques, whose vertices have (on average) less than
$\ell$ edges toward the rest of the graph. By tuning parameter $\ell$, the
degree of isolation can be controlled, and cliques that are overly connected to
the outside are filtered out. Building on Tomita et al.'s very practical
recursive algorithm for maximal clique enumeration, we propose four pruning
heuristics, applicable individually or in combination, that discard recursive
search branches that are guaranteed not to yield $\ell$-isolated maximal
cliques. Besides proving correctness, we characterize both the pruning power
and the computational cost of these heuristics, and we conduct an extensive
experimental study comparing our methods with Tomita's baseline and with a
state-of-the-art approach. Results show that two of our heuristics offer
substantial efficiency improvements, especially on real-world graphs with
social network properties.

### 8. [Improved Bounds with a Simple Algorithm for Edge Estimation for Graphs of Unknown Size](http://arxiv.org/pdf/2511.03650v1)

Authors: Debarshi Chanda

We propose a randomized algorithm with query access that given a graph $G$
with arboricity $\alpha$, and average degree $d$, makes
$\widetilde{O}\left(\frac{\alpha}{\varepsilon^2d}\right)$ \texttt{Degree} and
$\widetilde{O}\left(\frac{1}{\varepsilon^2}\right)$ \texttt{Random Edge}
queries to obtain an estimate $\widehat{d}$ satisfying $\widehat{d} \in
(1\pm\varepsilon)d$. This improves the $\widetilde{O}_{\varepsilon,\log
n}\left(\sqrt{\frac{n}{d}}\right)$ query algorithm of [Beretta et al., SODA
2026] that has access to \texttt{Degree}, \texttt{Neighbour}, and
\texttt{Random Edge} queries. Our algorithm does not require any graph
parameter as input, not even the size of the vertex set, and attains both
simplicity and practicality through a new estimation technique. We complement
our upper bounds with a lower bound that shows for all valid $n,d$, and
$\alpha$, any algorithm that has access to \texttt{Degree}, \texttt{Neighbour},
and \texttt{Random Edge} queries, must make at least
$\Omega\left(\min\left(d,\frac{\alpha}{d}\right)\right)$ queries to obtain a
$(1\pm\varepsilon)$-multiplicative estimate of $d$, even with the knowledge of
$n$ and $\alpha$. We also show that even with \texttt{Pair} and
\texttt{FullNbr} queries, an algorithm must make
$\Omega\left(\min\left(d,\frac{\alpha}{d}\right)\right)$ queries to obtain a
$(1\pm\varepsilon)$-multiplicative estimate of $d$. Our work addresses both the
questions raised by the work of [Beretta et al., SODA 2026].

### 9. [An Improved Quality Hierarchical Congestion Approximator in Near-Linear Time](http://arxiv.org/pdf/2511.03716v1)

Authors: Monika Henzinger, Robin Münk, Harald Räcke

A congestion approximator for a graph is a compact data structure that
approximately predicts the edge congestion required to route any set of flow
demands in a network. A congestion approximator is hierarchical if it consists
of a laminar family of cuts in the graph. There is a tradeoff between the
running time for computing a congestion approximator and its approximation
quality. Currently, for an $n$-node graph there exists a polynomial time
algorithm that achieves a $O(\log^{1.5}n \log \log n)$ approximation and a
near-linear time algorithm that achieves w.h.p. a $O(\log^4 n)$ approximation.
In this paper we give the first near-linear time algorithm, that achieves
w.h.p. a $O(\log^2 n \log \log n)$ approximation, using an hierarchical
congestion approximator with $O(n \log n)$ cuts. Based on a reduction from
oblivious routing, we also present a lower bound of $\Omega(\log n)$ for the
approximation quality of hierarchical congestion approximators.
  Our algorithm can also be implemented in the parallel setting achieving the
same approximation quality, polylogarithmic span and near-linear work. This
improves upon the best prior parallel algorithm, which has a $O(\log^9n)$
approximation.
  Crucial for achieving a near linear running time is a new partitioning
routine that, unlike previous such routines, manages to avoid recursing on
large subgraphs. To achieve the improved approximation quality, we introduce
the new concept of border routability of a cut and give an improved sparsest
cut oracle for general vertex weights.

### 10. [Non-Monotonicity in Fair Division of Graphs](http://arxiv.org/pdf/2511.03629v1)

Authors: Hadi Hosseini, Shraddha Pathak, Yu Zhou

We consider the problem of fairly allocating the vertices of a graph among
$n$ agents, where the value of a bundle is determined by its cut value -- the
number of edges with exactly one endpoint in the bundle. This model naturally
captures applications such as team formation and network partitioning, where
valuations are inherently non-monotonic: the marginal values may be positive,
negative, or zero depending on the composition of the bundle. We focus on the
fairness notion of envy-freeness up to one item (EF1) and explore its
compatibility with several efficiency concepts such as Transfer Stability (TS)
that prohibits single-item transfers that benefit one agent without making the
other worse-off. For general graphs, our results uncover a non-monotonic
relationship between the number of agents $n$ and the existence of allocations
satisfying EF1 and transfer stability (TS): such allocations always exist for
$n=2$, may fail to exist for $n=3$, but exist again for all $n\geq 4$. We
further show that existence can be guaranteed for any $n$ by slightly weakening
the efficiency requirement or by restricting the graph to forests. All of our
positive results are achieved via efficient algorithms.

### Emerging Technologies

### 1. [QAGT-MLP: An Attention-Based Graph Transformer for Small and Large-Scale Quantum Error Mitigation](http://arxiv.org/pdf/2511.03119v1)

Authors: Seyed Mohamad Ali Tousi, G. N. DeSouza

Noisy quantum devices demand error-mitigation techniques to be accurate yet
simple and efficient in terms of number of shots and processing time. Many
established approaches (e.g., extrapolation and quasi-probability cancellation)
impose substantial execution or calibration overheads, while existing
learning-based methods have difficulty scaling to large and deep circuits. In
this research, we introduce QAGT-MLP: an attention-based graph transformer
tailored for small- and large-scale quantum error mitigation (QEM). QAGT-MLP
encodes each quantum circuit as a graph whose nodes represent gate instances
and whose edges capture qubit connectivity and causal adjacency. A dual-path
attention module extracts features around measured qubits at two scales or
contexts: 1) graph-wide global structural context; and 2) fine-grained local
lightcone context. These learned representations are concatenated with
circuit-level descriptor features and the circuit noisy expected values, then
they are passed to a lightweight MLP to predict the noise-mitigated values. On
large-scale 100-qubit Trotterized 1D Transverse-Field Ising Models -- TFIM
circuits -- the proposed QAGT-MLP outperformed state-of-the-art learning
baselines in terms of mean error and error variability, demonstrating strong
validity and applicability in real-world QEM scenarios under matched shot
budgets. By using attention to fuse global structures with local lightcone
neighborhoods, QAGT-MLP achieves high mitigation quality without the increasing
noise scaling or resource demand required by classical QEM pipelines, while
still offering a scalable and practical path to QEM in modern and future
quantum workloads.

### 2. [LLM-enhanced Air Quality Monitoring Interface via Model Context Protocol](http://arxiv.org/pdf/2511.03706v1)

Authors: Yu-Erh Pan, Ayesha Siddika Nipu

Air quality monitoring is central to environmental sustainability and public
health, yet traditional systems remain difficult for non-expert users to
interpret due to complex visualizations, limited interactivity, and high
deployment costs. Recent advances in Large Language Models (LLMs) offer new
opportunities to make sensor data more accessible, but their tendency to
produce hallucinations limits reliability in safety-critical domains. To
address these challenges, we present an LLM-enhanced Air Monitoring Interface
(AMI) that integrates real-time sensor data with a conversational interface via
the Model Context Protocol (MCP). Our system grounds LLM outputs in live
environmental data, enabling accurate, context-aware responses while reducing
hallucination risk. The architecture combines a Django-based backend, a
responsive user dashboard, and a secure MCP server that exposes system
functions as discoverable tools, allowing the LLM to act as an active operator
rather than a passive responder. Expert evaluation demonstrated high factual
accuracy (4.78), completeness (4.82), and minimal hallucinations (4.84), on a
scale of 5, supported by inter-rater reliability analysis. These results
highlight the potential of combining LLMs with standardized tool protocols to
create reliable, secure, and user-friendly interfaces for real-time
environmental monitoring.

### 3. [HERP: Hardware for Energy Efficient and Realtime DB Search and Cluster Expansion in Proteomics](http://arxiv.org/pdf/2511.03437v1)

Authors: Md Mizanur Rahaman Nayan, Zheyu Li, Flavio Ponzina, Sumukh Pinge, Tajana Rosing, Azad J. Naeemi

Database (DB) search and clustering are fundamental in proteomics but
conventional full clustering and search approaches demand high resources and
incur long latency. We propose a lightweight incremental clustering and highly
parallelizable DB search platform tailored for resource-constrained
environments, delivering low energy and latency without compromising
performance. By leveraging mass-spectrometry insights, we employ bucket-wise
parallelization and query scheduling to reduce latency. A one-time hardware
initialization with pre-clustered proteomics data enables continuous DB search
and local re-clustering, offering a more practical and efficient alternative to
clustering from scratch. Heuristics from pre-clustered data guide incremental
clustering, accelerating the process by 20x with only a 0.3% increase in
clustering error. DB search results overlap by 96% with state-of-the-art tools,
validating search quality. The hardware leverages a 3T 2M T J SOT-CAM at the
7nm node with a compute-in-memory design. For the human genome draft dataset
(131GB), setup requires 1.19mJ for 2M spectra, while a 1000 query search
consumes 1.1{\mu}J. Bucket-wise parallelization further achieves 100x speedup.

### Graphics

### 1. [Visualization Biases MLLM's Decision Making in Network Data Tasks](http://arxiv.org/pdf/2511.03617v1)

Authors: Timo Brand, Henry Förster, Stephen G. Kobourov, Jacob Miller

We evaluate how visualizations can influence the judgment of MLLMs about the
presence or absence of bridges in a network. We show that the inclusion of
visualization improves confidence over a structured text-based input that could
theoretically be helpful for answering the question. On the other hand, we
observe that standard visualization techniques create a strong bias towards
accepting or refuting the presence of a bridge -- independently of whether or
not a bridge actually exists in the network. While our results indicate that
the inclusion of visualization techniques can effectively influence the MLLM's
judgment without compromising its self-reported confidence, they also imply
that practitioners must be careful of allowing users to include visualizations
in generative AI applications so as to avoid undesired hallucinations.

### 2. [Scheduling the Off-Diagonal Weingarten Loss of Neural SDFs for CAD Models](http://arxiv.org/pdf/2511.03147v1)

Authors: Haotian Yin, Przemyslaw Musialski

Neural signed distance functions (SDFs) have become a powerful representation
for geometric reconstruction from point clouds, yet they often require both
gradient- and curvature-based regularization to suppress spurious warp and
preserve structural fidelity. FlatCAD introduced the Off-Diagonal Weingarten
(ODW) loss as an efficient second-order prior for CAD surfaces, approximating
full-Hessian regularization at roughly half the computational cost. However,
FlatCAD applies a fixed ODW weight throughout training, which is suboptimal:
strong regularization stabilizes early optimization but suppresses detail
recovery in later stages. We present scheduling strategies for the ODW loss
that assign a high initial weight to stabilize optimization and progressively
decay it to permit fine-scale refinement. We investigate constant, linear,
quintic, and step interpolation schedules, as well as an increasing warm-up
variant. Experiments on the ABC CAD dataset demonstrate that time-varying
schedules consistently outperform fixed weights. Our method achieves up to a
35% improvement in Chamfer Distance over the FlatCAD baseline, establishing
scheduling as a simple yet effective extension of curvature regularization for
robust CAD reconstruction.

### Computer Science and Game Theory

### 1. [Branch-and-Cut for Computing Approximate Equilibria of Mixed-Integer Generalized Nash Games](http://arxiv.org/pdf/2511.03340v1)

Authors: Aloïs Duguet, Tobias Harks, Martin Schmidt, Julian Schwarz

Generalized Nash equilibrium problems with mixed-integer variables constitute
an important class of games in which each player solves a mixed-integer
optimization problem, where both the objective and the feasible set is
parameterized by the rivals' strategies. However, such games are known for
failing to admit exact equilibria and also the assumption of all players being
able to solve nonconvex problems to global optimality is questionable. This
motivates the study of approximate equilibria. In this work, we consider an
approximation concept that incorporates both multiplicative and additive
relaxations of optimality. We propose a branch-and-cut (B&C) method that
computes such approximate equilibria or proves its non-existence. For this, we
adopt the idea of intersection cuts and show the existence of such cuts under
the condition that the constraints are linear and each player's cost function
is either convex in the entire strategy profile, or, concave in the entire
strategy profile and linear in the rivals' strategies. For the special case of
standard Nash equilibrium problems, we introduce an alternative type of cut and
show that the method terminates finitely, provided that each player has only
finitely many distinct best-response sets. Finally, on the basis of the B&C
method, we introduce a single-tree binary-search method to compute
best-approximate equilibria under some simplifying assumptions. We implemented
these methods and present numerical results for a class of mixed-integer flow
games.

### 2. [Non-Monotonicity in Fair Division of Graphs](http://arxiv.org/pdf/2511.03629v1)

Authors: Hadi Hosseini, Shraddha Pathak, Yu Zhou

We consider the problem of fairly allocating the vertices of a graph among
$n$ agents, where the value of a bundle is determined by its cut value -- the
number of edges with exactly one endpoint in the bundle. This model naturally
captures applications such as team formation and network partitioning, where
valuations are inherently non-monotonic: the marginal values may be positive,
negative, or zero depending on the composition of the bundle. We focus on the
fairness notion of envy-freeness up to one item (EF1) and explore its
compatibility with several efficiency concepts such as Transfer Stability (TS)
that prohibits single-item transfers that benefit one agent without making the
other worse-off. For general graphs, our results uncover a non-monotonic
relationship between the number of agents $n$ and the existence of allocations
satisfying EF1 and transfer stability (TS): such allocations always exist for
$n=2$, may fail to exist for $n=3$, but exist again for all $n\geq 4$. We
further show that existence can be guaranteed for any $n$ by slightly weakening
the efficiency requirement or by restricting the graph to forests. All of our
positive results are achieved via efficient algorithms.

### 3. [Balanced contributions, consistency, and value for games with externalities](http://arxiv.org/pdf/2511.03145v1)

Authors: André Casajus, Yukihiko Funaki, Frank Huettner

We consider fair and consistent extensions of the Shapley value for games
with externalities. Based on the restriction identified by Casajus et al.
(2024, Games Econ. Behavior 147, 88-146), we define balanced contributions,
Sobolev's consistency, and Hart and Mas-Colell's consistency for games with
externalities, and we show that these properties lead to characterizations of
the generalization of the Shapley value introduced by Macho-Stadler et al.
(2007, J. Econ. Theory 135, 339-356), that parallel important characterizations
of the Shapley value.

### 4. [Evolutionary Dynamics in Continuous-time Finite-state Mean Field Games -- Part II: Stability](http://arxiv.org/pdf/2511.03297v1)

Authors: Leonardo Pedroso, Andrea Agazzi, W. P. M. H. Heemels, Mauro Salazar

We study a dynamic game with a large population of players who choose actions
from a finite set in continuous time. Each player has a state in a finite state
space that evolves stochastically with their actions. A player's reward depends
not only on their own state and action but also on the distribution of states
and actions across the population, capturing effects such as congestion in
traffic networks. In Part I, we introduced an evolutionary model and a new
solution concept - the mixed stationary Nash Equilibrium (MSNE) - which
coincides with the rest points of the mean field evolutionary model under
meaningful families of revision protocols. In this second part, we investigate
the evolutionary stability of MSNE. We derive conditions on both the structure
of the MSNE and the game's payoff map that ensure local and global stability
under evolutionary dynamics. These results characterize when MSNE can robustly
emerge and persist against strategic deviations, thereby providing insight into
its long-term viability in large population dynamic games.

### Human-Computer Interaction

### 1. [Tracing Generative AI in Digital Art: A Longitudinal Study of Chinese Painters' Attitudes, Practices, and Identity Negotiation](http://arxiv.org/pdf/2511.03117v1)

Authors: Yibo Meng, Ruiqi Chen, Xin Chen, Zhiming Liu, Yan Guan

This study presents a five-year longitudinal mixed-methods study of 17
Chinese digital painters, examining how their attitudes and practices evolved
in response to generative AI. Our findings reveal a trajectory from resistance
and defensiveness, to pragmatic adoption, and ultimately to reflective
reconstruction, shaped by strong peer pressures and shifting emotional
experiences. Persistent concerns around copyright and creative labor highlight
the ongoing negotiation of identity and values. This work contributes by
offering rare longitudinal empirical data, advancing a theoretical lens of
"identity and value negotiation," and providing design implications for future
human-AI collaborative systems.

### 2. [Ceci N'est Pas un Drone: Investigating the Impact of Design Representation on Design Decision Making When Using GenAI](http://arxiv.org/pdf/2511.03131v1)

Authors: Zeda Xu, Nikolas Martelaro, Christopher McComb

With generative AI-powered design tools, designers and engineers can
efficiently generate large numbers of design ideas. However, efficient
exploration of these ideas requires designers to select a smaller group of
potential solutions for further development. Therefore, the ability to judge
and evaluate designs is critical for the successful use of generative design
tools. Different design representation modalities can potentially affect
designers' judgments. This work investigates how different design modalities,
including visual rendering, numerical performance data, and a combination of
both, affect designers' design selections from AI-generated design concepts for
Uncrewed Aerial Vehicles. We found that different design modalities do affect
designers' choices. Unexpectedly, we found that providing only numerical design
performance data can lead to the best ability to select optimal designs. We
also found that participants prefer visually conventional designs with
axis-symmetry. The findings of this work provide insights into the interaction
between human users and generative design systems.

### 3. [Large Language Models as Information Sources: Distinctive Characteristics and Types of Low-Quality Information](http://arxiv.org/pdf/2511.03198v1)

Authors: Jiawei Zhou, Amy Z. Chen, Darshi Shah, Laura M. Schwab-Reese, Munmun De Choudhury

Recent advances in large language models (LLMs) have brought public and
scholarly attention to their potential in generating low-quality information.
While widely acknowledged as a risk, low-quality information remains a vaguely
defined concept, and little is known about how it manifests in LLM outputs or
how these outputs differ from those of traditional information sources. In this
study, we focus on two key questions: What types of low-quality information are
produced by LLMs, and what makes them distinct than human-generated
counterparts? We conducted focus groups with public health professionals and
individuals with lived experience in three critical health contexts (vaccines,
opioid use disorder, and intimate partner violence) where high-quality
information is essential and misinformation, bias, and insensitivity are
prevalent concerns. We identified a typology of LLM-generated low-quality
information and a set of distinctive LLM characteristics compared to
traditional information sources. Our findings show that low-quality information
extends beyond factual inaccuracies into types such as misprioritization and
exaggeration, and that LLM affordances fundamentally differs from previous
technologies. This work offers typologies on LLM distinctive characteristics
and low-quality information types as a starting point for future efforts to
understand LLM-generated low-quality information and mitigate related
informational harms. We call for conceptual and methodological discussions of
information quality to move beyond truthfulness, in order to address the
affordances of emerging technologies and the evolving dynamics of information
behaviors.

### 4. [I Prompt, it Generates, we Negotiate. Exploring Text-Image Intertextuality in Human-AI Co-Creation of Visual Narratives with VLMs](http://arxiv.org/pdf/2511.03375v1)

Authors: Mengyao Guo, Kexin Nie, Ze Gao, Black Sun, Xueyang Wang, Jinda Han, Xingting Wu

Creating meaningful visual narratives through human-AI collaboration requires
understanding how text-image intertextuality emerges when textual intentions
meet AI-generated visuals. We conducted a three-phase qualitative study with 15
participants using GPT-4o to investigate how novices navigate sequential visual
narratives. Our findings show that users develop strategies to harness AI's
semantic surplus by recognizing meaningful visual content beyond literal
descriptions, iteratively refining prompts, and constructing narrative
significance through complementary text-image relationships. We identified four
distinct collaboration patterns and, through fsQCA's analysis, discovered three
pathways to successful intertextual collaboration: Educational Collaborator,
Technical Expert, and Visual Thinker. However, participants faced challenges,
including cultural representation gaps, visual consistency issues, and
difficulties translating narrative concepts into visual prompts. These findings
contribute to HCI research by providing an empirical account of
\textit{text-image intertextuality} in human-AI co-creation and proposing
design implications for role-based AI assistants that better support iterative,
human-led creative processes in visual storytelling.

### 5. [SVG Decomposition for Enhancing Large Multimodal Models Visualization Comprehension: A Study with Floor Plans](http://arxiv.org/pdf/2511.03478v1)

Authors: Jeongah Lee, Ali Sarvghad

Large multimodal models (LMMs) are increasingly capable of interpreting
visualizations, yet they continue to struggle with spatial reasoning. One
proposed strategy is decomposition, which breaks down complex visualizations
into structured components. In this work, we examine the efficacy of scalable
vector graphics (SVGs) as a decomposition strategy for improving LMMs'
performance on floor plans comprehension. Floor plans serve as a valuable
testbed because they combine geometry, topology, and semantics, and their
reliable comprehension has real-world applications, such as accessibility for
blind and low-vision individuals. We conducted an exploratory study with three
LMMs (GPT-4o, Claude 3.7 Sonnet, and Llama 3.2 11B Vision Instruct) across 75
floor plans. Results show that combining SVG with raster input (SVG+PNG)
improves performance on spatial understanding tasks but often hinders spatial
reasoning, particularly in pathfinding. These findings highlight both the
promise and limitations of decomposition as a strategy for advancing spatial
visualization comprehension.

### 6. [PnPSelect: Plug-and-play IoT Device Selection Using Ultra-wideband Signals](http://arxiv.org/pdf/2511.03534v1)

Authors: Zhaoxin Chang, Fusang Zhang, Jie Xiong, Ziyu Li, Badii Jouaber, Daqing Zhang

In recent years, the number of Internet of Things (IoT) devices in smart
homes has rapidly increased. A key challenge affecting user experience is how
to enable users to efficiently and intuitively select the devices they wish to
control. This paper proposes PnPSelect, a plug-and-play IoT device selection
solution utilizing Ultra-wideband (UWB) technology on commercial devices.
Unlike previous works, PnPSelect does not require the installation of dedicated
hardware on each IoT device, thereby reducing deployment costs and
complexities, and achieving true plug-and-play functionality. To enable
intuitive device selection, we introduce a pointing direction estimation method
that utilizes UWB readings from a single anchor to infer the user pointing
direction. Additionally, we propose a lightweight device localization method
that allows users to register new IoT devices by simply pointing at them from
two distinct positions, eliminating the need for manual measurements. We
implement PnPSelect on commercial smartphones and smartwatches and conduct
extensive evaluations in both controlled laboratory settings and real-world
environments. Our results demonstrate high accuracy, robustness, and
adaptability, making PnPSelect a practical and scalable solution for
next-generation smart home interactions.

### 7. [Knowledge Graph for Intelligent Generation of Artistic Image Creation: Constructing a New Annotation Hierarchy](http://arxiv.org/pdf/2511.03585v1)

Authors: Jia Kaixin, Zhu Kewen, Deng Huanghuang, Qiu Yiwu, Ding Shiying, Ding Chenyang, Li Zejian

Our study aims to establish a unified, systematic, and referable knowledge
framework for the annotation of art image datasets, addressing issues of
ambiguous definitions and inconsistent results caused by the lack of common
standards during the annotation process. To achieve this goal, a hierarchical
and systematic art image knowledge graph was constructed. It was developed
based on the composition principles of art images, incorporating the Structured
Theory of Visual Knowledge proposed by Academician Yunhe Pan in On Visual
Knowledge-which states that visual knowledge must achieve precise expression of
spatial forms and dynamic relationships through "prototype-category" and
"hierarchical structure". Through in-depth review of Chinese and Western art
theories and pioneering integration of the Chinese cultural perspective, this
graph took shape. The core visual language of art images was deconstructed by
this knowledge graph. Meanwhile, the unique spatial theory and symbolic system
of Chinese painting were compared with and supplemented by Western art
theories. This graph converts qualitative artistic concepts into a clear
structured framework. It not only conforms to the cognitive law that "visual
knowledge takes precedence over verbal knowledge" in humans but also provides
an interpretable and inferential visual knowledge foundation for AI art
generation and cross-cultural art analysis. It ensures the high quality and
consistency of annotated data, thus offering key support for art intelligence
research in the AI 2.0 era.

### 8. [OriFeel: Origami-Inspired Actuation for Force-Based Tactile Feedback on Ambient Surfaces](http://arxiv.org/pdf/2511.03673v1)

Authors: Shubham Rohal, Shijia Pan

People are constantly in touch with surfaces in their lives, such as a sofa,
armrest, and table, making them natural tactile interfaces. Despite the recent
advancements in shape-changing surfaces, current available solutions are often
challenging to retrofit into ambient surfaces due to their bulky form factor or
high power requirements. We present \name, a foldable structure-enabled tactile
feedback mechanism that leverages the structural properties of Miura-Ori fold
to enable on-surface force actuation. The foldable structure allows the
surfaces to provide perpendicular force via lateral actuation, resulting in a
slim form factor that can be actuated via cable-based design using a servo
motor. We evaluate the system with a real-world prototype and a user study. The
user study shows that users can effectively distinguish multiple intensity
levels.

### 9. [Accelerating Physical Property Reasoning for Augmented Visual Cognition](http://arxiv.org/pdf/2511.03126v1)

Authors: Hongbo Lan, Zhenlin An, Haoyu Li, Vaibhav Singh, Longfei Shangguan

This paper introduces \sysname, a system that accelerates vision-guided
physical property reasoning to enable augmented visual cognition. \sysname
minimizes the run-time latency of this reasoning pipeline through a combination
of both algorithmic and systematic optimizations, including rapid geometric 3D
reconstruction, efficient semantic feature fusion, and parallel view encoding.
Through these simple yet effective optimizations, \sysname reduces the
end-to-end latency of this reasoning pipeline from 10--20 minutes to less than
6 seconds. A head-to-head comparison on the ABO dataset shows that \sysname
achieves this 62.9$\times$--287.2$\times$ speedup while not only reaching
on-par (and sometimes slightly better) object-level physical property
estimation accuracy(e.g. mass), but also demonstrating superior performance in
material segmentation and voxel-level inference than two SOTA baselines. We
further combine gaze-tracking with \sysname to localize the object of interest
in cluttered, real-world environments, streamlining the physical property
reasoning on smart glasses. The case study with Meta Aria Glasses conducted at
an IKEA furniture store demonstrates that \sysname achives consistently high
performance compared to controlled captures, providing robust property
estimations even with fewer views in real-world scenarios.

### 10. [AI as We Describe It: How Large Language Models and Their Applications in Health are Represented Across Channels of Public Discourse](http://arxiv.org/pdf/2511.03174v1)

Authors: Jiawei Zhou, Lei Zhang, Mei Li, Benjamin D Horne, Munmun De Choudhury

Representation shapes public attitudes and behaviors. With the arrival and
rapid adoption of LLMs, the way these systems are introduced will negotiate
societal expectations for their role in high-stakes domains like health. Yet it
remains unclear whether current narratives present a balanced view. We analyzed
five prominent discourse channels (news, research press, YouTube, TikTok, and
Reddit) over a two-year period on lexical style, informational content, and
symbolic representation. Discussions were generally positive and episodic, with
positivity increasing over time. Risk communication was unthorough and often
reduced to information quality incidents, while explanations of LLMs'
generative nature were rare. Compared with professional outlets, TikTok and
Reddit highlighted wellbeing applications and showed greater variations in tone
and anthropomorphism but little attention to risks. We discuss implications for
public discourse as a diagnostic tool in identifying literacy and governance
gaps, and for communication and design strategies to support more informed LLM
engagement.

### Information Retrieval

### 1. [Generative Sequential Recommendation via Hierarchical Behavior Modeling](http://arxiv.org/pdf/2511.03155v1)

Authors: Zhefan Wang, Guokai Yan, Jinbei Yu, Siyu Gu, Jingyan Chen, Peng Jiang, Zhiqiang Guo, Min Zhang

Recommender systems in multi-behavior domains, such as advertising and
e-commerce, aim to guide users toward high-value but inherently sparse
conversions. Leveraging auxiliary behaviors (e.g., clicks, likes, shares) is
therefore essential. Recent progress on generative recommendations has brought
new possibilities for multi-behavior sequential recommendation. However,
existing generative approaches face two significant challenges: 1) Inadequate
Sequence Modeling: capture the complex, cross-level dependencies within user
behavior sequences, and 2) Lack of Suitable Datasets: publicly available
multi-behavior recommendation datasets are almost exclusively derived from
e-commerce platforms, limiting the validation of feasibility in other domains,
while also lacking sufficient side information for semantic ID generation. To
address these issues, we propose a novel generative framework, GAMER
(Generative Augmentation and Multi-lEvel behavior modeling for Recommendation),
built upon a decoder-only backbone. GAMER introduces a cross-level interaction
layer to capture hierarchical dependencies among behaviors and a sequential
augmentation strategy that enhances robustness in training. To further advance
this direction, we collect and release ShortVideoAD, a large-scale
multi-behavior dataset from a mainstream short-video platform, which differs
fundamentally from existing e-commerce datasets and provides pretrained
semantic IDs for research on generative methods. Extensive experiments show
that GAMER consistently outperforms both discriminative and generative
baselines across multiple metrics.

### 2. [KScaNN: Scalable Approximate Nearest Neighbor Search on Kunpeng](http://arxiv.org/pdf/2511.03298v1)

Authors: Oleg Senkevich, Siyang Xu, Tianyi Jiang, Alexander Radionov, Jan Tabaszewski, Dmitriy Malyshev, Zijian Li, Daihao Xue, Licheng Yu, Weidi Zeng, Meiling Wang, Xin Yao, Siyu Huang, Gleb Neshchetkin, Qiuling Pan, Yaoyao Fu

Approximate Nearest Neighbor Search (ANNS) is a cornerstone algorithm for
information retrieval, recommendation systems, and machine learning
applications. While x86-based architectures have historically dominated this
domain, the increasing adoption of ARM-based servers in industry presents a
critical need for ANNS solutions optimized on ARM architectures. A naive port
of existing x86 ANNS algorithms to ARM platforms results in a substantial
performance deficit, failing to leverage the unique capabilities of the
underlying hardware. To address this challenge, we introduce KScaNN, a novel
ANNS algorithm co-designed for the Kunpeng 920 ARM architecture. KScaNN
embodies a holistic approach that synergizes sophisticated, data aware
algorithmic refinements with carefully-designed hardware specific
optimizations. Its core contributions include: 1) novel algorithmic techniques,
including a hybrid intra-cluster search strategy and an improved PQ residual
calculation method, which optimize the search process at a higher level; 2) an
ML-driven adaptive search module that provides adaptive, per-query tuning of
search parameters, eliminating the inefficiencies of static configurations; and
3) highly-optimized SIMD kernels for ARM that maximize hardware utilization for
the critical distance computation workloads. The experimental results
demonstrate that KScaNN not only closes the performance gap but establishes a
new standard, achieving up to a 1.63x speedup over the fastest x86-based
solution. This work provides a definitive blueprint for achieving
leadership-class performance for vector search on modern ARM architectures and
underscores

### 3. [A Semantic Encoding of Object Centric Event Data](http://arxiv.org/pdf/2511.03351v1)

Authors: Saba Latif, Fajar J. Ekaputra, Maxim Vidgof, Sabrina Kirrane, Claudio Di Ciccio

The Object-Centric Event Data (OCED) is a novel meta-model aimed at providing
a common ground for process data records centered around events and objects.
One of its objectives is to foster interoperability and process information
exchange. In this context, the integration of data from different providers,
the combination of multiple processes, and the enhancement of knowledge
inference are novel challenges. Semantic Web technologies can enable the
creation of a machine-readable OCED description enriched through ontology-based
relationships and entity categorization. In this paper, we introduce an
approach built upon Semantic Web technologies for the realization of
semantic-enhanced OCED, with the aim to strengthen process data reasoning,
interconnect information sources, and boost expressiveness.

### 4. [A Study on Library Resources with Services Satisfaction based on Library Users Affiliated Colleges to Solapur University](http://arxiv.org/pdf/2511.03209v1)

Authors: Patel Adam Burhansab, M Sadik Batcha, Muneer Ahmad

The main aim of this study was to assess and evaluate user satisfaction with
library resources and services among library users associated with Solapur
University. The current research shows the level of users satisfaction with
different library resources and services offered by college libraries. The
research found that a vast number of respondents were pleased with library
facilities and services. The research is designed to achieve users satisfaction
in the library to investigate the level of satisfaction towards library
resources and services with regards to 26 colleges of Solapur University based
in Maharashtra. Information in the form of data has been collected from
colleges and on the basis of users results; analysis needs to analyze users
satisfaction.

### 5. [Russian Contribution to Coronary Artery Disease Research: A Scientometric Mapping of Publications](http://arxiv.org/pdf/2511.03215v1)

Authors: Muneer Ahmad, M Sadik Batcha

The present study attempts to highlight the research output generated in
Russia in coronary artery disease (CAD) research during the period 1990-2019 to
understand the distribution of research output, top journals for publications,
and most prolific authors, authorship pattern, and citation pattern. This study
is based on secondary data extracted from the Science Citation Index (SCI),
which is an integral component of the Web of Science. Descriptive and
inferential statistical techniques were applied in the study. There were 5058
articles by Russian scholars in coronary artery disease during 1990-2019; they
preferred to publish in Russian journals. The research contributions were in
the form of research articles, meeting abstracts and reviews with a consistent
drop in the number of editorial material and article; proceedings paper with
time. Co-authorship was the norm in coronary artery disease research, with a
steady increase in the number of multi-author documents in recent years.

### 6. [Beyond Ranked Lists: The SARAL Framework for Cross-Lingual Document Set Retrieval](http://arxiv.org/pdf/2511.03228v1)

Authors: Shantanu Agarwal, Joel Barry, Elizabeth Boschee, Scott Miller

Machine Translation for English Retrieval of Information in Any Language
(MATERIAL) is an IARPA initiative targeted to advance the state of
cross-lingual information retrieval (CLIR). This report provides a detailed
description of Information Sciences Institute's (ISI's) Summarization and
domain-Adaptive Retrieval Across Language's (SARAL's) effort for MATERIAL.
Specifically, we outline our team's novel approach to handle CLIR with emphasis
in developing an approach amenable to retrieve a query-relevant document
\textit{set}, and not just a ranked document-list. In MATERIAL's Phase-3
evaluations, SARAL exceeded the performance of other teams in five out of six
evaluation conditions spanning three different languages (Farsi, Kazakh, and
Georgian).

### 7. [Discourse-Aware Scientific Paper Recommendation via QA-Style Summarization and Multi-Level Contrastive Learning](http://arxiv.org/pdf/2511.03330v1)

Authors: Shenghua Wang, Zhen Yin

The rapid growth of open-access (OA) publications has intensified the
challenge of identifying relevant scientific papers. Due to privacy constraints
and limited access to user interaction data, recent efforts have shifted toward
content-based recommendation, which relies solely on textual information.
However, existing models typically treat papers as unstructured text,
neglecting their discourse organization and thereby limiting semantic
completeness and interpretability. To address these limitations, we propose
OMRC-MR, a hierarchical framework that integrates QA-style OMRC (Objective,
Method, Result, Conclusion) summarization, multi-level contrastive learning,
and structure-aware re-ranking for scholarly recommendation. The QA-style
summarization module converts raw papers into structured and
discourse-consistent representations, while multi-level contrastive objectives
align semantic representations across metadata, section, and document levels.
The final re-ranking stage further refines retrieval precision through
contextual similarity calibration. Experiments on DBLP, S2ORC, and the newly
constructed Sci-OMRC dataset demonstrate that OMRC-MR consistently surpasses
state-of-the-art baselines, achieving up to 7.2% and 3.8% improvements in
Precision@10 and Recall@10, respectively. Additional evaluations confirm that
QA-style summarization produces more coherent and factually complete
representations. Overall, OMRC-MR provides a unified and interpretable
content-based paradigm for scientific paper recommendation, advancing
trustworthy and privacy-aware scholarly information retrieval.

### 8. [Hybrid Fact-Checking that Integrates Knowledge Graphs, Large Language Models, and Search-Based Retrieval Agents Improves Interpretable Claim Verification](http://arxiv.org/pdf/2511.03217v1)

Authors: Shaghayegh Kolli, Richard Rosenbaum, Timo Cavelius, Lasse Strothe, Andrii Lata, Jana Diesner

Large language models (LLMs) excel in generating fluent utterances but can
lack reliable grounding in verified information. At the same time,
knowledge-graph-based fact-checkers deliver precise and interpretable evidence,
yet suffer from limited coverage or latency. By integrating LLMs with knowledge
graphs and real-time search agents, we introduce a hybrid fact-checking
approach that leverages the individual strengths of each component. Our system
comprises three autonomous steps: 1) a Knowledge Graph (KG) Retrieval for rapid
one-hop lookups in DBpedia, 2) an LM-based classification guided by a
task-specific labeling prompt, producing outputs with internal rule-based
logic, and 3) a Web Search Agent invoked only when KG coverage is insufficient.
Our pipeline achieves an F1 score of 0.93 on the FEVER benchmark on the
Supported/Refuted split without task-specific fine-tuning. To address Not
enough information cases, we conduct a targeted reannotation study showing that
our approach frequently uncovers valid evidence for claims originally labeled
as Not Enough Information (NEI), as confirmed by both expert annotators and LLM
reviewers. With this paper, we present a modular, opensource fact-checking
pipeline with fallback strategies and generalization across datasets.

### 9. [CLAX: Fast and Flexible Neural Click Models in JAX](http://arxiv.org/pdf/2511.03620v1)

Authors: Philipp Hager, Onno Zoeter, Maarten de Rijke

CLAX is a JAX-based library that implements classic click models using modern
gradient-based optimization. While neural click models have emerged over the
past decade, complex click models based on probabilistic graphical models
(PGMs) have not systematically adopted gradient-based optimization, preventing
practitioners from leveraging modern deep learning frameworks while preserving
the interpretability of classic models. CLAX addresses this gap by replacing
EM-based optimization with direct gradient-based optimization in a numerically
stable manner. The framework's modular design enables the integration of any
component, from embeddings and deep networks to custom modules, into classic
click models for end-to-end optimization. We demonstrate CLAX's efficiency by
running experiments on the full Baidu-ULTR dataset comprising over a billion
user sessions in $\approx$ 2 hours on a single GPU, orders of magnitude faster
than traditional EM approaches. CLAX implements ten classic click models,
serving both industry practitioners seeking to understand user behavior and
improve ranking performance at scale and researchers developing new click
models. CLAX is available at: https://github.com/philipphager/clax

### 10. [Two thousand years of the oracle problem. Insights from Ancient Delphi on the future of blockchain oracles](http://arxiv.org/pdf/2511.03319v1)

Authors: Giulio Caldarelli, Massimiliano Ornaghi

The oracle problem refers to the inability of an agent to know if the
information coming from an oracle is authentic and unbiased. In ancient times,
philosophers and historians debated on how to evaluate, increase, and secure
the reliability of oracle predictions, particularly those from Delphi, which
pertained to matters of state. Today, we refer to data carriers for automatic
machines as oracles, but establishing a secure channel between these oracles
and the real world still represents a challenge. Despite numerous efforts, this
problem remains mostly unsolved, and the recent advent of blockchain oracles
has added a layer of complexity because of the decentralization of blockchains.
This paper conceptually connects Delphic and modern blockchain oracles,
developing a comparative framework. Leveraging blockchain oracle taxonomy,
lexical analysis is also performed on 167 Delphic queries to shed light on the
relationship between oracle answer quality and question type. The presented
framework aims first at revealing commonalities between classical and
computational oracles and then at enriching the oracle analysis within each
field. This study contributes to the computer science literature by proposing
strategies to improve the reliability of blockchain oracles based on insights
from Delphi and to classical literature by introducing a framework that can
also be applied to interpret and classify other ancient oracular mechanisms.

### Machine Learning

### 1. [Towards Scalable Backpropagation-Free Gradient Estimation](http://arxiv.org/pdf/2511.03110v1)

Authors: Daniel Wang, Evan Markou, Dylan Campbell

While backpropagation--reverse-mode automatic differentiation--has been
extraordinarily successful in deep learning, it requires two passes (forward
and backward) through the neural network and the storage of intermediate
activations. Existing gradient estimation methods that instead use forward-mode
automatic differentiation struggle to scale beyond small networks due to the
high variance of the estimates. Efforts to mitigate this have so far introduced
significant bias to the estimates, reducing their utility. We introduce a
gradient estimation approach that reduces both bias and variance by
manipulating upstream Jacobian matrices when computing guess directions. It
shows promising results and has the potential to scale to larger networks,
indeed performing better as the network width is increased. Our understanding
of this method is facilitated by analyses of bias and variance, and their
connection to the low-dimensional structure of neural network gradients.

### 2. [UnCLe: Towards Scalable Dynamic Causal Discovery in Non-linear Temporal Systems](http://arxiv.org/pdf/2511.03168v1)

Authors: Tingzhu Bi, Yicheng Pan, Xinrui Jiang, Huize Sun, Meng Ma, Ping Wang

Uncovering cause-effect relationships from observational time series is
fundamental to understanding complex systems. While many methods infer static
causal graphs, real-world systems often exhibit dynamic causality-where
relationships evolve over time. Accurately capturing these temporal dynamics
requires time-resolved causal graphs. We propose UnCLe, a novel deep learning
method for scalable dynamic causal discovery. UnCLe employs a pair of Uncoupler
and Recoupler networks to disentangle input time series into semantic
representations and learns inter-variable dependencies via auto-regressive
Dependency Matrices. It estimates dynamic causal influences by analyzing
datapoint-wise prediction errors induced by temporal perturbations. Extensive
experiments demonstrate that UnCLe not only outperforms state-of-the-art
baselines on static causal discovery benchmarks but, more importantly, exhibits
a unique capability to accurately capture and represent evolving temporal
causality in both synthetic and real-world dynamic systems (e.g., human
motion). UnCLe offers a promising approach for revealing the underlying,
time-varying mechanisms of complex phenomena.

### 3. [Incorporating Quality of Life in Climate Adaptation Planning via Reinforcement Learning](http://arxiv.org/pdf/2511.03238v1)

Authors: Miguel Costa, Arthur Vandervoort, Martin Drews, Karyn Morrissey, Francisco C. Pereira

Urban flooding is expected to increase in frequency and severity as a
consequence of climate change, causing wide-ranging impacts that include a
decrease in urban Quality of Life (QoL). Meanwhile, policymakers must devise
adaptation strategies that can cope with the uncertain nature of climate change
and the complex and dynamic nature of urban flooding. Reinforcement Learning
(RL) holds significant promise in tackling such complex, dynamic, and uncertain
problems. Because of this, we use RL to identify which climate adaptation
pathways lead to a higher QoL in the long term. We do this using an Integrated
Assessment Model (IAM) which combines a rainfall projection model, a flood
model, a transport accessibility model, and a quality of life index. Our
preliminary results suggest that this approach can be used to learn optimal
adaptation measures and it outperforms other realistic and real-world planning
strategies. Our framework is publicly available:
https://github.com/MLSM-at-DTU/maat_qol_framework.

### 4. [A unified physics-informed generative operator framework for general inverse problems](http://arxiv.org/pdf/2511.03241v1)

Authors: Gang Bao, Yaohua Zang

Solving inverse problems governed by partial differential equations (PDEs) is
central to science and engineering, yet remains challenging when measurements
are sparse, noisy, or when the underlying coefficients are high-dimensional or
discontinuous. Existing deep learning approaches either require extensive
labeled datasets or are limited to specific measurement types, often leading to
failure in such regimes and restricting their practical applicability. Here, a
novel generative neural operator framework, IGNO, is introduced to overcome
these limitations. IGNO unifies the solution of inverse problems from both
point measurements and operator-valued data without labeled training pairs.
This framework encodes high-dimensional, potentially discontinuous coefficient
fields into a low-dimensional latent space, which drives neural operator
decoders to reconstruct both coefficients and PDE solutions. Training relies
purely on physics constraints through PDE residuals, while inversion proceeds
via efficient gradient-based optimization in latent space, accelerated by an a
priori normalizing flow model. Across a diverse set of challenging inverse
problems, including recovery of discontinuous coefficients from solution-based
measurements and the EIT problem with operator-based measurements, IGNO
consistently achieves accurate, stable, and scalable inversion even under
severe noise. It consistently outperforms the state-of-the-art method under
varying noise levels and demonstrates strong generalization to
out-of-distribution targets. These results establish IGNO as a unified and
powerful framework for tackling challenging inverse problems across
computational science domains.

### 5. [Climate Adaptation with Reinforcement Learning: Economic vs. Quality of Life Adaptation Pathways](http://arxiv.org/pdf/2511.03243v1)

Authors: Miguel Costa, Arthur Vandervoort, Martin Drews, Karyn Morrissey, Francisco C. Pereira

Climate change will cause an increase in the frequency and severity of flood
events, prompting the need for cohesive adaptation policymaking. Designing
effective adaptation policies, however, depends on managing the uncertainty of
long-term climate impacts. Meanwhile, such policies can feature important
normative choices that are not always made explicit. We propose that
Reinforcement Learning (RL) can be a useful tool to both identify adaptation
pathways under uncertain conditions while it also allows for the explicit
modelling (and consequent comparison) of different adaptation priorities (e.g.
economic vs. wellbeing). We use an Integrated Assessment Model (IAM) to link
together a rainfall and flood model, and compute the impacts of flooding in
terms of quality of life (QoL), transportation, and infrastructure damage. Our
results show that models prioritising QoL over economic impacts results in more
adaptation spending as well as a more even distribution of spending over the
study area, highlighting the extent to which such normative assumptions can
alter adaptation policy. Our framework is publicly available:
https://github.com/MLSM-at-DTU/maat_qol_framework.

### 6. [Diffusion Language Models are Super Data Learners](http://arxiv.org/pdf/2511.03276v1)

Authors: Jinjie Ni, Qian Liu, Longxu Dou, Chao Du, Zili Wang, Hang Yan, Tianyu Pang, Michael Qizhe Shieh

Under strictly controlled pre-training settings, we observe a Crossover: when
unique data is limited, diffusion language models (DLMs) consistently surpass
autoregressive (AR) models by training for more epochs. The crossover shifts
later with more or higher-quality data, earlier with larger models, and
persists across dense and sparse architectures. We attribute the gains to three
compounding factors: (1) any-order modeling, (2) super-dense compute from
iterative bidirectional denoising, and (3) built-in Monte Carlo augmentation;
input or parameter noise improves AR under data constraint but cannot close the
gap. At scale, a 1.7B DLM trained with a ~1.5T-token compute budget on 10B
unique Python tokens overtakes an AR coder trained with strictly matched
settings. In addition, a 1B-parameter DLM achieves > 56% accuracy on HellaSwag
and > 33% on MMLU using only 1B tokens, without any special tricks, just by
repeating standard pre-training data. We also show that rising validation
cross-entropy does not imply degraded downstream performance in this regime.

### 7. [Multi-Objective Adaptive Rate Limiting in Microservices Using Deep Reinforcement Learning](http://arxiv.org/pdf/2511.03279v1)

Authors: Ning Lyu, Yuxi Wang, Ziyu Cheng, Qingyuan Zhang, Feng Chen

As cloud computing and microservice architectures become increasingly
prevalent, API rate limiting has emerged as a critical mechanism for ensuring
system stability and service quality. Traditional rate limiting algorithms,
such as token bucket and sliding window, while widely adopted, struggle to
adapt to dynamic traffic patterns and varying system loads. This paper proposes
an adaptive rate limiting strategy based on deep reinforcement learning that
dynamically balances system throughput and service latency. We design a hybrid
architecture combining Deep Q-Network (DQN) and Asynchronous Advantage
Actor-Critic (A3C) algorithms, modeling the rate limiting decision process as a
Markov Decision Process. The system continuously monitors microservice states
and learns optimal rate limiting policies through environmental interaction.
Extensive experiments conducted in a Kubernetes cluster environment demonstrate
that our approach achieves 23.7% throughput improvement and 31.4% P99 latency
reduction compared to traditional fixed-threshold strategies under high-load
scenarios. Results from a 90-day production deployment handling 500 million
daily requests validate the practical effectiveness of the proposed method,
with 82% reduction in service degradation incidents and 68% decrease in manual
interventions.

### 8. [Graph Neural AI with Temporal Dynamics for Comprehensive Anomaly Detection in Microservices](http://arxiv.org/pdf/2511.03285v1)

Authors: Qingyuan Zhang, Ning Lyu, Le Liu, Yuxi Wang, Ziyu Cheng, Cancan Hua

This study addresses the problem of anomaly detection and root cause tracing
in microservice architectures and proposes a unified framework that combines
graph neural networks with temporal modeling. The microservice call chain is
abstracted as a directed graph, where multidimensional features of nodes and
edges are used to construct a service topology representation, and graph
convolution is applied to aggregate features across nodes and model
dependencies, capturing complex structural relationships among services. On
this basis, gated recurrent units are introduced to model the temporal
evolution of call chains, and multi-layer stacking and concatenation operations
are used to jointly obtain structural and temporal representations, improving
the ability to identify anomaly patterns. Furthermore, anomaly scoring
functions at both the node and path levels are defined to achieve unified
modeling from local anomaly detection to global call chain tracing, which
enables the identification of abnormal service nodes and the reconstruction of
potential anomaly propagation paths. Sensitivity experiments are then designed
from multiple dimensions, including hyperparameters, environmental
disturbances, and data distribution, to evaluate the framework, and results
show that it outperforms baseline methods in key metrics such as AUC, ACC,
Recall, and F1-Score, maintaining high accuracy and stability under dynamic
topologies and complex environments. This research not only provides a new
technical path for anomaly detection in microservices but also lays a
methodological foundation for intelligent operations in distributed systems.

### 9. [SORTeD Rashomon Sets of Sparse Decision Trees: Anytime Enumeration](http://arxiv.org/pdf/2511.03344v1)

Authors: Elif Arslan, Jacobus G. M. van der Linden, Serge Hoogendoorn, Marco Rinaldi, Emir Demirović

Sparse decision tree learning provides accurate and interpretable predictive
models that are ideal for high-stakes applications by finding the single most
accurate tree within a (soft) size limit. Rather than relying on a single
"best" tree, Rashomon sets-trees with similar performance but varying
structures-can be used to enhance variable importance analysis, enrich
explanations, and enable users to choose simpler trees or those that satisfy
stakeholder preferences (e.g., fairness) without hard-coding such criteria into
the objective function. However, because finding the optimal tree is NP-hard,
enumerating the Rashomon set is inherently challenging. Therefore, we introduce
SORTD, a novel framework that improves scalability and enumerates trees in the
Rashomon set in order of the objective value, thus offering anytime behavior.
Our experiments show that SORTD reduces runtime by up to two orders of
magnitude compared with the state of the art. Moreover, SORTD can compute
Rashomon sets for any separable and totally ordered objective and supports
post-evaluating the set using other separable (and partially ordered)
objectives. Together, these advances make exploring Rashomon sets more
practical in real-world applications.

### 10. [A Modular, Data-Free Pipeline for Multi-Label Intention Recognition in Transportation Agentic AI Applications](http://arxiv.org/pdf/2511.03363v1)

Authors: Xiaocai Zhang, Hur Lim, Ke Wang, Zhe Xiao, Jing Wang, Kelvin Lee, Xiuju Fu, Zheng Qin

In this study, a modular, data-free pipeline for multi-label intention
recognition is proposed for agentic AI applications in transportation. Unlike
traditional intent recognition systems that depend on large, annotated corpora
and often struggle with fine-grained, multi-label discrimination, our approach
eliminates the need for costly data collection while enhancing the accuracy of
multi-label intention understanding. Specifically, the overall pipeline, named
DMTC, consists of three steps: 1) using prompt engineering to guide large
language models (LLMs) to generate diverse synthetic queries in different
transport scenarios; 2) encoding each textual query with a Sentence-T5 model to
obtain compact semantic embeddings; 3) training a lightweight classifier using
a novel online focal-contrastive (OFC) loss that emphasizes hard samples and
maximizes inter-class separability. The applicability of the proposed pipeline
is demonstrated in an agentic AI application in the maritime transportation
context. Extensive experiments show that DMTC achieves a Hamming loss of 5.35%
and an AUC of 95.92%, outperforming state-of-the-art multi-label classifiers
and recent end-to-end SOTA LLM-based baselines. Further analysis reveals that
Sentence-T5 embeddings improve subset accuracy by at least 3.29% over
alternative encoders, and integrating the OFC loss yields an additional 0.98%
gain compared to standard contrastive objectives. In conclusion, our system
seamlessly routes user queries to task-specific modules (e.g., ETA information,
traffic risk evaluation, and other typical scenarios in the transportation
domain), laying the groundwork for fully autonomous, intention-aware agents
without costly manual labelling.

### Networking and Internet Architecture

### 1. [CRSF: Enabling QoS-Aware Beyond-Connectivity Service Sharing in 6G Local Networks](http://arxiv.org/pdf/2511.03081v1)

Authors: Pragya Sharma, Amanda Xiang, Abbas Kiani, John Kaippallimalil, Tony Saboorian, Haining Wang

Sixth-generation (6G) networks are envisioned to support interconnected local
subnetworks that can share specialized, beyond-connectivity services. However,
a standardized architecture for discovering and selecting these services across
network boundaries has not existed yet. To address this gap, this paper
introduces the Central Repository and Selection Function (CRSF), a novel
network function for the 6G core that facilitates efficient inter-subnetwork
service discovery and selection. We formulate the selection process as a
QoS-aware optimization problem designed to balance service quality metrics with
user-defined priorities. We evaluate our system model through simulations for a
sensing service scenario and observe a consistently higher aggregate Quality of
Service (QoS) compared to the baseline selection strategy. The proposed CRSF
provides a foundational and extensible mechanism for building standardized,
collaborative, and service-centric interconnected networks essential for the 6G
era.

### 2. [Handover Configurations in Operational 5G Networks: Diversity, Evolution, and Impact on Performance](http://arxiv.org/pdf/2511.03116v1)

Authors: Moinak Ghoshal, Imran Khan, Phuc Dinh, Z. Jonny Kong, Omar Basit, Sizhe Wang, Yufei Feng, Y. Charlie Hu, Dimitrios Koutsonikolas

Mobility management in cellular networks, especially the handover (HO)
process, plays a key role in providing seamless and ubiquitous Internet access.
The wide-scale deployment of 5G and the resulting co-existence of 4G/5G in the
past six years have significantly changed the landscape of all mobile network
operators and made the HO process much more complex than before. While several
recent works have studied the impact of HOs on user experience, why and how HOs
occur and how HO configurations affect performance in 5G operational networks
remains largely unknown. Through four cross-country driving trips across the US
spread out over a 27-month period, we conduct an in-depth measurement study of
HO configurations across all three major US operators. Our study reveals (a)
new types of HOs and new HO events used by operators to handle these new types
of HOs, (b) overly aggressive HO configurations that result in unnecessarily
high signaling overhead, (c) large diversity in HO configuration parameter
values, which also differ across operators, but significantly lower diversity
in 5G compared to LTE, and (d) sub-optimal HO configurations/decisions leading
to poor pre- or post-HO performance. Our findings have many implications for
mobile operators, as they keep fine-tuning their 5G HO configurations.

### 3. [Joint Optimization of DNN Model Caching and Request Routing in Mobile Edge Computing](http://arxiv.org/pdf/2511.03159v1)

Authors: Shuting Qiu, Fang Dong, Siyu Tan, Ruiting Zhou, Dian Shen, Patrick P. C. Lee, Qilin Fan

Mobile edge computing (MEC) can pre-cache deep neural networks (DNNs) near
end-users, providing low-latency services and improving users' quality of
experience (QoE). However, caching all DNN models at edge servers with limited
capacity is difficult, and the impact of model loading time on QoE remains
underexplored. Hence, we introduce dynamic DNNs in edge scenarios,
disassembling a complete DNN model into interrelated submodels for more
fine-grained and flexible model caching and request routing solutions. This
raises the pressing issue of jointly deciding request routing and submodel
caching for dynamic DNNs to balance model inference precision and loading
latency for QoE optimization. In this paper, we study the joint dynamic model
caching and request routing problem in MEC networks, aiming to maximize user
request inference precision under constraints of server resources, latency, and
model loading time. To tackle this problem, we propose CoCaR, an offline
algorithm based on linear programming and random rounding that leverages
dynamic DNNs to optimize caching and routing schemes, achieving near-optimal
performance. Furthermore, we develop an online variant of CoCaR, named
CoCaR-OL, enabling effective adaptation to dynamic and unpredictable online
request patterns. The simulation results demonstrate that the proposed CoCaR
improves the average inference precision of user requests by 46\% compared to
state-of-the-art baselines. In addition, in online scenarios, CoCaR-OL achieves
an improvement of no less than 32.3\% in user QoE over competitive baselines.

### 4. [Integrity Under Siege: A Rogue gNodeB's Manipulation of 5G Network Slice Allocation](http://arxiv.org/pdf/2511.03312v1)

Authors: Jiali Xu, Valeria Loscri, Romain Rouvoy

The advent of 5G networks, with network slicing as a cornerstone technology,
promises customized, high-performance services, but also introduces novel
attack surfaces beyond traditional threats. This article investigates a
critical and underexplored integrity vulnerability: the manipulation of network
slice allocation to compromise Quality of Service (QoS) and resource integrity.
We introduce a threat model, grounded in a risk analysis of permissible yet
insecure configurations like null-ciphering (5G-EA0), demonstrating how a rogue
gNodeB acting as a Man-in-the-Middle can exploit protocol weaknesses to forge
slice requests and hijack a User Equipment's (UE) connection. Through a
comprehensive experimental evaluation on a 5G testbed, we demonstrate the
attack's versatile and severe impacts. Our findings show this integrity breach
can manifest as obvious QoS degradation, such as a 95% bandwidth reduction and
150% latency increase when forcing UE to a suboptimal slice, or as stealthy
slice manipulation that is indistinguishable from benign network operation and
generates no core network errors. Furthermore, we validate a systemic resource
contamination attack where redirecting a crowd of UE orchestrates a
Denial-of-Service, causing packet loss to exceed 60% and inducing measurable
CPU saturation (~80%) on core network User Plane Functions (UPFs). Based on
these results, we discuss the profound implications for Service Level
Agreements (SLAs) and critical infrastructure. We propose concrete, cross-layer
mitigation strategies for network operators as future work, underscoring the
urgent need to secure the integrity of dynamic resource management in 5G
networks.

### 5. [Inter-Agent Trust Models: A Comparative Study of Brief, Claim, Proof, Stake, Reputation and Constraint in Agentic Web Protocol Design-A2A, AP2, ERC-8004, and Beyond](http://arxiv.org/pdf/2511.03434v1)

Authors: Botao 'Amber' Hu, Helena Rong

As the "agentic web" takes shape-billions of AI agents (often LLM-powered)
autonomously transacting and collaborating-trust shifts from human oversight to
protocol design. In 2025, several inter-agent protocols crystallized this
shift, including Google's Agent-to-Agent (A2A), Agent Payments Protocol (AP2),
and Ethereum's ERC-8004 "Trustless Agents," yet their underlying trust
assumptions remain under-examined. This paper presents a comparative study of
trust models in inter-agent protocol design: Brief (self- or third-party
verifiable claims), Claim (self-proclaimed capabilities and identity, e.g.
AgentCard), Proof (cryptographic verification, including zero-knowledge proofs
and trusted execution environment attestations), Stake (bonded collateral with
slashing and insurance), Reputation (crowd feedback and graph-based trust
signals), and Constraint (sandboxing and capability bounding). For each, we
analyze assumptions, attack surfaces, and design trade-offs, with particular
emphasis on LLM-specific fragilities-prompt injection,
sycophancy/nudge-susceptibility, hallucination, deception, and
misalignment-that render purely reputational or claim-only approaches brittle.
Our findings indicate no single mechanism suffices. We argue for
trustless-by-default architectures anchored in Proof and Stake to gate
high-impact actions, augmented by Brief for identity and discovery and
Reputation overlays for flexibility and social signals. We comparatively
evaluate A2A, AP2, ERC-8004 and related historical variations in academic
research under metrics spanning security, privacy, latency/cost, and social
robustness (Sybil/collusion/whitewashing resistance). We conclude with hybrid
trust model recommendations that mitigate reputation gaming and misinformed LLM
behavior, and we distill actionable design guidelines for safer, interoperable,
and scalable agent economies.

### Robotics

### 1. [SENT Map -- Semantically Enhanced Topological Maps with Foundation Models](http://arxiv.org/pdf/2511.03165v1)

Authors: Raj Surya Rajendran Kathirvel, Zach A Chavis, Stephen J. Guy, Karthik Desingh

We introduce SENT-Map, a semantically enhanced topological map for
representing indoor environments, designed to support autonomous navigation and
manipulation by leveraging advancements in foundational models (FMs). Through
representing the environment in a JSON text format, we enable semantic
information to be added and edited in a format that both humans and FMs
understand, while grounding the robot to existing nodes during planning to
avoid infeasible states during deployment. Our proposed framework employs a two
stage approach, first mapping the environment alongside an operator with a
Vision-FM, then using the SENT-Map representation alongside a natural-language
query within an FM for planning. Our experimental results show that
semantic-enhancement enables even small locally-deployable FMs to successfully
plan over indoor environments.

### 2. [Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors based on Deep Reinforcement Learning](http://arxiv.org/pdf/2511.03167v1)

Authors: Xin Liu, Jinze Wu, Yinghui Li, Chenkun Qi, Yufei Xue, Feng Gao

Multi-legged robots offer enhanced stability to navigate complex terrains
with their multiple legs interacting with the environment. However, how to
effectively coordinate the multiple legs in a larger action exploration space
to generate natural and robust movements is a key issue. In this paper, we
introduce a motion prior-based approach, successfully applying deep
reinforcement learning algorithms to a real hexapod robot. We generate a
dataset of optimized motion priors, and train an adversarial discriminator
based on the priors to guide the hexapod robot to learn natural gaits. The
learned policy is then successfully transferred to a real hexapod robot, and
demonstrate natural gait patterns and remarkable robustness without visual
information in complex terrains. This is the first time that a reinforcement
learning controller has been used to achieve complex terrain walking on a real
hexapod robot.

### 3. [GUIDES: Guidance Using Instructor-Distilled Embeddings for Pre-trained Robot Policy Enhancement](http://arxiv.org/pdf/2511.03400v1)

Authors: Minquan Gao, Xinyi Li, Qing Yan, Xiaojian Sun, Xiaopan Zhang, Chien-Ming Huang, Jiachen Li

Pre-trained robot policies serve as the foundation of many validated robotic
systems, which encapsulate extensive embodied knowledge. However, they often
lack the semantic awareness characteristic of foundation models, and replacing
them entirely is impractical in many situations due to high costs and the loss
of accumulated knowledge. To address this gap, we introduce GUIDES, a
lightweight framework that augments pre-trained policies with semantic guidance
from foundation models without requiring architectural redesign. GUIDES employs
a fine-tuned vision-language model (Instructor) to generate contextual
instructions, which are encoded by an auxiliary module into guidance
embeddings. These embeddings are injected into the policy's latent space,
allowing the legacy model to adapt to this new semantic input through brief,
targeted fine-tuning. For inference-time robustness, a large language
model-based Reflector monitors the Instructor's confidence and, when confidence
is low, initiates a reasoning loop that analyzes execution history, retrieves
relevant examples, and augments the VLM's context to refine subsequent actions.
Extensive validation in the RoboCasa simulation environment across diverse
policy architectures shows consistent and substantial improvements in task
success rates. Real-world deployment on a UR5 robot further demonstrates that
GUIDES enhances motion precision for critical sub-tasks such as grasping.
Overall, GUIDES offers a practical and resource-efficient pathway to upgrade,
rather than replace, validated robot policies.

### 4. [Value Elicitation for a Socially Assistive Robot Addressing Social Anxiety: A Participatory Design Approach](http://arxiv.org/pdf/2511.03444v1)

Authors: Vesna Poprcova, Iulia Lefter, Martijn Warnier, Frances Brazier

Social anxiety is a prevalent mental health condition that can significantly
impact overall well-being and quality of life. Despite its widespread effects,
adequate support or treatment for social anxiety is often insufficient.
Advances in technology, particularly in social robotics, offer promising
opportunities to complement traditional mental health. As an initial step
toward developing effective solutions, it is essential to understand the values
that shape what is considered meaningful, acceptable, and helpful. In this
study, a participatory design workshop was conducted with mental health
academic researchers to elicit the underlying values that should inform the
design of socially assistive robots for social anxiety support. Through
creative, reflective, and envisioning activities, participants explored
scenarios and design possibilities, allowing for systematic elicitation of
values, expectations, needs, and preferences related to robot-supported
interventions. The findings reveal rich insights into design-relevant
values-including adaptivity, acceptance, and efficacy-that are core to support
for individuals with social anxiety. This study highlights the significance of
a research-led approach to value elicitation, emphasising user-centred and
context-aware design considerations in the development of socially assistive
robots.

### 5. [Motion Planning Under Temporal Logic Specifications In Semantically Unknown Environments](http://arxiv.org/pdf/2511.03652v1)

Authors: Azizollah Taheri, Derya Aksaray

This paper addresses a motion planning problem to achieve
spatio-temporal-logical tasks, expressed by syntactically co-safe linear
temporal logic specifications (scLTL\next), in uncertain environments. Here,
the uncertainty is modeled as some probabilistic knowledge on the semantic
labels of the environment. For example, the task is "first go to region 1, then
go to region 2"; however, the exact locations of regions 1 and 2 are not known
a priori, instead a probabilistic belief is available. We propose a novel
automata-theoretic approach, where a special product automaton is constructed
to capture the uncertainty related to semantic labels, and a reward function is
designed for each edge of this product automaton. The proposed algorithm
utilizes value iteration for online replanning. We show some theoretical
results and present some simulations/experiments to demonstrate the efficacy of
the proposed approach.

### 6. [Source-Free Bistable Fluidic Gripper for Size-Selective and Stiffness-Adaptive Grasping](http://arxiv.org/pdf/2511.03691v1)

Authors: Zhihang Qin, Yueheng Zhang, Wan Su, Linxin Hou, Shenghao Zhou, Zhijun Chen, Yu Jun Tan, Cecilia Laschi

Conventional fluid-driven soft grippers typically depend on external sources,
which limit portability and long-term autonomy. This work introduces a
self-contained soft gripper with fixed size that operates solely through
internal liquid redistribution among three interconnected bistable snap-through
chambers. When the top sensing chamber deforms upon contact, the displaced
liquid triggers snap-through expansion of the grasping chambers, enabling
stable and size-selective grasping without continuous energy input. The
internal hydraulic feedback further allows passive adaptation of gripping
pressure to object stiffness. This source-free and compact design opens new
possibilities for lightweight, stiffness-adaptive fluid-driven manipulation in
soft robotics, providing a feasible approach for targeted size-specific
sampling and operation in underwater and field environments.

### 7. [Learning-based Cooperative Robotic Paper Wrapping: A Unified Control Policy with Residual Force Control](http://arxiv.org/pdf/2511.03181v1)

Authors: Rewida Ali, Cristian C. Beltran-Hernandez, Weiwei Wan, Kensuke Harada

Human-robot cooperation is essential in environments such as warehouses and
retail stores, where workers frequently handle deformable objects like paper,
bags, and fabrics. Coordinating robotic actions with human assistance remains
difficult due to the unpredictable dynamics of deformable materials and the
need for adaptive force control. To explore this challenge, we focus on the
task of gift wrapping, which exemplifies a long-horizon manipulation problem
involving precise folding, controlled creasing, and secure fixation of paper.
Success is achieved when the robot completes the sequence to produce a neatly
wrapped package with clean folds and no tears.
  We propose a learning-based framework that integrates a high-level task
planner powered by a large language model (LLM) with a low-level hybrid
imitation learning (IL) and reinforcement learning (RL) policy. At its core is
a Sub-task Aware Robotic Transformer (START) that learns a unified policy from
human demonstrations. The key novelty lies in capturing long-range temporal
dependencies across the full wrapping sequence within a single model. Unlike
vanilla Action Chunking with Transformer (ACT), typically applied to short
tasks, our method introduces sub-task IDs that provide explicit temporal
grounding. This enables robust performance across the entire wrapping process
and supports flexible execution, as the policy learns sub-goals rather than
merely replicating motion sequences.
  Our framework achieves a 97% success rate on real-world wrapping tasks. We
show that the unified transformer-based policy reduces the need for specialized
models, allows controlled human supervision, and effectively bridges high-level
intent with the fine-grained force control required for deformable object
manipulation.

### 8. [Periodic Skill Discovery](http://arxiv.org/pdf/2511.03187v1)

Authors: Jonghae Park, Daesol Cho, Jusuk Lee, Dongseok Shim, Inkyu Jang, H. Jin Kim

Unsupervised skill discovery in reinforcement learning (RL) aims to learn
diverse behaviors without relying on external rewards. However, current methods
often overlook the periodic nature of learned skills, focusing instead on
increasing the mutual dependence between states and skills or maximizing the
distance traveled in latent space. Considering that many robotic tasks --
particularly those involving locomotion -- require periodic behaviors across
varying timescales, the ability to discover diverse periodic skills is
essential. Motivated by this, we propose Periodic Skill Discovery (PSD), a
framework that discovers periodic behaviors in an unsupervised manner. The key
idea of PSD is to train an encoder that maps states to a circular latent space,
thereby naturally encoding periodicity in the latent representation. By
capturing temporal distance, PSD can effectively learn skills with diverse
periods in complex robotic tasks, even with pixel-based observations. We
further show that these learned skills achieve high performance on downstream
tasks such as hurdling. Moreover, integrating PSD with an existing skill
discovery method offers more diverse behaviors, thus broadening the agent's
repertoire. Our code and demos are available at
https://jonghaepark.github.io/psd/

### 9. [Development of the Bioinspired Tendon-Driven DexHand 021 with Proprioceptive Compliance Control](http://arxiv.org/pdf/2511.03481v1)

Authors: Jianbo Yuan, Haohua Zhu, Jing Dai, Sheng Yi

The human hand plays a vital role in daily life and industrial applications,
yet replicating its multifunctional capabilities-including motion, sensing, and
coordinated manipulation-with robotic systems remains a formidable challenge.
Developing a dexterous robotic hand requires balancing human-like agility with
engineering constraints such as complexity, size-to-weight ratio, durability,
and force-sensing performance. This letter presents Dex-Hand 021, a
high-performance, cable-driven five-finger robotic hand with 12 active and 7
passive degrees of freedom (DoFs), achieving 19 DoFs dexterity in a lightweight
1 kg design. We propose a proprioceptive force-sensing-based admittance control
method to enhance manipulation. Experimental results demonstrate its superior
performance: a single-finger load capacity exceeding 10 N, fingertip
repeatability under 0.001 m, and force estimation errors below 0.2 N. Compared
to PID control, joint torques in multi-object grasping are reduced by 31.19%,
significantly improves force-sensing capability while preventing overload
during collisions. The hand excels in both power and precision grasps,
successfully executing 33 GRASP taxonomy motions and complex manipulation
tasks. This work advances the design of lightweight, industrial-grade dexterous
hands and enhances proprioceptive control, contributing to robotic manipulation
and intelligent manufacturing.

### 10. [Indicating Robot Vision Capabilities with Augmented Reality](http://arxiv.org/pdf/2511.03550v1)

Authors: Hong Wang, Ridhima Phatak, James Ocampo, Zhao Han

Research indicates that humans can mistakenly assume that robots and humans
have the same field of view (FoV), possessing an inaccurate mental model of
robots. This misperception may lead to failures during human-robot
collaboration tasks where robots might be asked to complete impossible tasks
about out-of-view objects. The issue is more severe when robots do not have a
chance to scan the scene to update their world model while focusing on assigned
tasks. To help align humans' mental models of robots' vision capabilities, we
propose four FoV indicators in augmented reality (AR) and conducted a user
human-subjects experiment (N=41) to evaluate them in terms of accuracy,
confidence, task efficiency, and workload. These indicators span a spectrum
from egocentric (robot's eye and head space) to allocentric (task space).
Results showed that the allocentric blocks at the task space had the highest
accuracy with a delay in interpreting the robot's FoV. The egocentric indicator
of deeper eye sockets, possible for physical alteration, also increased
accuracy. In all indicators, participants' confidence was high while cognitive
load remained low. Finally, we contribute six guidelines for practitioners to
apply our AR indicators or physical alterations to align humans' mental models
with robots' vision capabilities.

### Software Engineering

### 1. [Automated Prompt Generation for Code Intelligence: An Empirical study and Experience in WeChat](http://arxiv.org/pdf/2511.03136v1)

Authors: Kexing Ji, Shiyun Fu, Cuiyun Gao, Yujia Chen, Zezhou Yang, Chaozheng Wang, Yuetang Deng

Large Code Models (LCMs) show potential in code intelligence, but their
effectiveness is greatly influenced by prompt quality. Current prompt design is
mostly manual, which is time-consuming and highly dependent on specific LCMs
and tasks. While automated prompt generation (APG) exists in NLP, it is
underexplored for code intelligence. This creates a gap, as automating the
prompt process is essential for developers facing diverse tasks and black-box
LCMs.
  To mitigate this, we empirically investigate two important parts of APG:
Instruction Generation (IG) and Multi-Step Reasoning (MSR). IG provides a
task-related description to instruct LCMs, while MSR guides them to produce
logical steps before the final answer. We evaluate widely-used APG methods for
each part on four open-source LCMs and three code intelligence tasks: code
translation (PL-PL), code summarization (PL-NL), and API recommendation
(NL-PL).Experimental results indicate that both IG and MSR dramatically enhance
performance compared to basic prompts. Based on these results, we propose a
novel APG approach combining the best methods of the two parts. Experiments
show our approach achieves average improvements of 28.38% in CodeBLEU (code
translation), 58.11% in ROUGE-L (code summarization), and 84.53% in
SuccessRate@1 (API recommendation) over basic prompts. To validate its
effectiveness in an industrial scenario, we evaluate our approach on
WeChat-Bench, a proprietary dataset, achieving an average MRR improvement of
148.89% for API recommendation.

### 2. [Towards Realistic Project-Level Code Generation via Multi-Agent Collaboration and Semantic Architecture Modeling](http://arxiv.org/pdf/2511.03404v1)

Authors: Qianhui Zhao, Li Zhang, Fang Liu, Junhang Cheng, Chengru Wu, Junchen Ai, Qiaoyuanhe Meng, Lichen Zhang, Xiaoli Lian, Shubin Song, Yuanping Guo

In recent years, Large Language Models (LLMs) have achieved remarkable
progress in automated code generation. In real-world software engineering, the
growing demand for rapid iteration and continuous delivery underscores the
importance of project-level code generation, where LLMs are expected to
generate complete software projects directly from complex user requirements.
Although existing studies have made initial explorations, they still face key
limitations, including unrealistic datasets and unreliable evaluation metrics
that fail to reflect real-world complexity, the semantic gap between
human-written requirements and machine-interpretable structures, and
difficulties in managing hierarchical dependencies and maintaining quality
throughout the generation process. To address these limitations, we first
introduce CodeProjectEval, a project-level code generation dataset built from
18 real-world repositories with 12.7 files and 2,388.6 lines of code per task
on average, supplemented with documentation and executable test cases for
automatic evaluation. We further propose ProjectGen, a multi-agent framework
that decomposes projects into architecture design, skeleton generation, and
code filling stages with iterative refinement and memory-based context
management. Within this framework, we introduce the Semantic Software
Architecture Tree (SSAT), a structured and semantically rich representation
that effectively bridges user requirements and source code implementation.
Experiments show that ProjectGen achieves state-of-the-art performance, passing
52/124 test cases on the small-scale project-level code generation dataset
DevBench, a 57% improvement over the baseline approaches, and 310 test cases on
CodeProjectEval, representing an improvement of roughly tenfold compared to the
baselines.

### 3. [U2F: Encouraging SWE-Agent to Seize Novelty without Losing Feasibility](http://arxiv.org/pdf/2511.03517v1)

Authors: Wencheng Ye, Yan Liu

Large language models (LLMs) have shown strong capabilities in software
engineering tasks, yet most existing LLM-based SWE-Agents mainly tackle
well-defined problems using conventional methods, often overlooking alternative
or innovative solutions beyond their predefined frameworks. This limitation is
evident in open-world software environments, where emerging challenges
transcend established paradigms.
  We propose U2F (Unknown Unknowns to Functional solutions), a
cognitive-inspired, uncertainty-embracing multi-agent framework that
systematically surfaces "Unknown Unknowns" - novel solution pathways absent
from initial formulations but holding innovative potential. U2F consists of two
key components: (1) a Discovery-Exploration-Integration agent system for
uncovering and synthesizing potential solutions, and (2) cognitive enhancement
mechanisms across three dimensions: cross-domain analogical reasoning, reverse
thinking, and external validation, which strategically reframe and extend
conventional solution boundaries.
  Applied to 218 real-world software enabler stories curated from authentic
engineering tasks, U2F achieved notable improvements: human experts reported a
14 percent increase in overall novelty, 51 percent improvement in semantic
novelty, and stable feasibility (4.02/5.0), corroborated by an LLM-based
evaluator. These results highlight the potential of embracing uncertainty as a
catalyst for innovation in software engineering.

### 4. [RefAgent: A Multi-agent LLM-based Framework for Automatic Software Refactoring](http://arxiv.org/pdf/2511.03153v1)

Authors: Khouloud Oueslati, Maxime Lamothe, Foutse Khomh

Large Language Models (LLMs) have substantially influenced various software
engineering tasks. Indeed, in the case of software refactoring, traditional
LLMs have shown the ability to reduce development time and enhance code
quality. However, these LLMs often rely on static, detailed instructions for
specific tasks. In contrast, LLM-based agents can dynamically adapt to evolving
contexts and autonomously make decisions by interacting with software tools and
executing workflows. In this paper, we explore the potential of LLM-based
agents in supporting refactoring activities. Specifically, we introduce
RefAgent, a multi-agent LLM-based framework for end-to-end software
refactoring. RefAgent consists of specialized agents responsible for planning,
executing, testing, and iteratively refining refactorings using self-reflection
and tool-calling capabilities. We evaluate RefAgent on eight open-source Java
projects, comparing its effectiveness against a single-agent approach, a
search-based refactoring tool, and historical developer refactorings. Our
assessment focuses on: (1) the impact of generated refactorings on software
quality, (2) the ability to identify refactoring opportunities, and (3) the
contribution of each LLM agent through an ablation study. Our results show that
RefAgent achieves a median unit test pass rate of 90%, reduces code smells by a
median of 52.5%, and improves key quality attributes (e.g., reusability) by a
median of 8.6%. Additionally, it closely aligns with developer refactorings and
the search-based tool in identifying refactoring opportunities, attaining a
median F1-score of 79.15% and 72.7%, respectively. Compared to single-agent
approaches, RefAgent improves the median unit test pass rate by 64.7% and the
median compilation success rate by 40.1%. These findings highlight the promise
of multi-agent architectures in advancing automated software refactoring.

### 5. [Understanding Robustness of Model Editing in Code LLMs: An Empirical Study](http://arxiv.org/pdf/2511.03182v1)

Authors: Vinaik Chhetri, A. B Siddique, Umar Farooq

Large language models (LLMs) are increasingly used in software development.
However, while LLMs remain static after pretraining, programming languages and
APIs continue to evolve, leading to the generation of deprecated or
incompatible code that undermines reliability. Retraining LLMs from scratch to
reflect such changes is computationally expensive, making model editing a
promising lightweight alternative that updates only a small subset of
parameters. Despite its potential, it remains unclear whether model editing
yields genuine syntactic and semantic adaptations or merely superficial fixes.
In this work, we present a systematic study of five state-of-the-art model
editing methods: Constrained Fine-Tuning (FT), GRACE, MEMIT, PMET, and ROME. We
apply these methods to three leading open-source code LLMs, CodeLlama,
CodeQwen1.5, and DeepSeek-Coder, under controlled API deprecation scenarios.
Our evaluation covers both instant and sequential editing settings, using three
disjoint evaluation sets designed to assess reliability, generalization, and
specificity. We measure model correctness at three levels: successful
compilation, partial test case pass, and full test pass. Our findings show that
instant edits consistently degrade model performance, with syntactic validity
dropping by up to 86 percentage points and functional correctness declining by
45 points even in the best-performing setting. Sequential edits further amplify
this degradation, and in some cases, model performance collapses entirely.
Across all models, most passing generations relied on workarounds rather than
correctly adopting the intended changes, while faulty adoptions that result in
test failures or compilation errors were significantly more frequent. Correct
adoptions, where the model correctly integrates the intended change, occurred
in only about 6% of cases.

### 6. [Integrity Under Siege: A Rogue gNodeB's Manipulation of 5G Network Slice Allocation](http://arxiv.org/pdf/2511.03312v1)

Authors: Jiali Xu, Valeria Loscri, Romain Rouvoy

The advent of 5G networks, with network slicing as a cornerstone technology,
promises customized, high-performance services, but also introduces novel
attack surfaces beyond traditional threats. This article investigates a
critical and underexplored integrity vulnerability: the manipulation of network
slice allocation to compromise Quality of Service (QoS) and resource integrity.
We introduce a threat model, grounded in a risk analysis of permissible yet
insecure configurations like null-ciphering (5G-EA0), demonstrating how a rogue
gNodeB acting as a Man-in-the-Middle can exploit protocol weaknesses to forge
slice requests and hijack a User Equipment's (UE) connection. Through a
comprehensive experimental evaluation on a 5G testbed, we demonstrate the
attack's versatile and severe impacts. Our findings show this integrity breach
can manifest as obvious QoS degradation, such as a 95% bandwidth reduction and
150% latency increase when forcing UE to a suboptimal slice, or as stealthy
slice manipulation that is indistinguishable from benign network operation and
generates no core network errors. Furthermore, we validate a systemic resource
contamination attack where redirecting a crowd of UE orchestrates a
Denial-of-Service, causing packet loss to exceed 60% and inducing measurable
CPU saturation (~80%) on core network User Plane Functions (UPFs). Based on
these results, we discuss the profound implications for Service Level
Agreements (SLAs) and critical infrastructure. We propose concrete, cross-layer
mitigation strategies for network operators as future work, underscoring the
urgent need to secure the integrity of dynamic resource management in 5G
networks.

### 7. [Light over Heavy: Automated Performance Requirements Quantification with Linguistic Inducement](http://arxiv.org/pdf/2511.03421v1)

Authors: Shihai Wang, Tao Chen

Elicited performance requirements need to be quantified for compliance in
different engineering tasks, e.g., configuration tuning and performance
testing. Much existing work has relied on manual quantification, which is
expensive and error-prone due to the imprecision. In this paper, we present
LQPR, a highly efficient automatic approach for performance requirements
quantification.LQPR relies on a new theoretical framework that converts
quantification as a classification problem. Despite the prevalent applications
of Large Language Models (LLMs) for requirement analytics, LQPR takes a
different perspective to address the classification: we observed that
performance requirements can exhibit strong patterns and are often
short/concise, therefore we design a lightweight linguistically induced
matching mechanism. We compare LQPR against nine state-of-the-art
learning-based approaches over diverse datasets, demonstrating that it is
ranked as the sole best for 75% or more cases with two orders less cost. Our
work proves that, at least for performance requirement quantification,
specialized methods can be more suitable than the general LLM-driven
approaches.

### 8. [Investigating the Impact of Isolation on Synchronized Benchmarks](http://arxiv.org/pdf/2511.03533v1)

Authors: Nils Japke, Furat Hamdan, Diana Baumann, David Bermbach

Benchmarking in cloud environments suffers from performance variability from
multi-tenant resource contention. Duet benchmarking mitigates this by running
two workload versions concurrently on the same VM, exposing them to identical
external interference. However, intra-VM contention between synchronized
workloads necessitates additional isolation mechanisms.
  This work evaluates three such strategies: cgroups and CPU pinning, Docker
containers, and Firecracker MicroVMs. We compare all strategies with an
unisolated baseline experiment, by running benchmarks with a duet setup
alongside a noise generator. This noise generator "steals" compute resources to
degrade performance measurements.
  All experiments showed different latency distributions while under the
effects of noise generation, but results show that process isolation generally
lowered false positives, except for our experiments with Docker containers.
Even though Docker containers rely internally on cgroups and CPU pinning, they
were more susceptible to performance degradation due to noise influence.
Therefore, we recommend to use process isolation for synchronized workloads,
with the exception of Docker containers.

### 9. [Uncovering Code Insights: Leveraging GitHub Artifacts for Deeper Code Understanding](http://arxiv.org/pdf/2511.03549v1)

Authors: Ziv Nevo, Orna Raz, Karen Yorav

Understanding the purpose of source code is a critical task in software
maintenance, onboarding, and modernization. While large language models (LLMs)
have shown promise in generating code explanations, they often lack grounding
in the broader software engineering context. We propose a novel approach that
leverages natural language artifacts from GitHub -- such as pull request
descriptions, issue descriptions and discussions, and commit messages -- to
enhance LLM-based code understanding. Our system consists of three components:
one that extracts and structures relevant GitHub context, another that uses
this context to generate high-level explanations of the code's purpose, and a
third that validates the explanation. We implemented this as a standalone tool,
as well as a server within the Model Context Protocol (MCP), enabling
integration with other AI-assisted development tools. Our main use case is that
of enhancing a standard LLM-based code explanation with code insights that our
system generates. To evaluate explanations' quality, we conducted a small scale
user study, with developers of several open projects, as well as developers of
proprietary projects. Our user study indicates that when insights are generated
they often are helpful and non trivial, and are free from hallucinations.

### 10. [The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents](http://arxiv.org/pdf/2511.03690v1)

Authors: Xingyao Wang, Simon Rosenberg, Juan Michelini, Calvin Smith, Hoang Tran, Engel Nyst, Rohit Malhotra, Xuhui Zhou, Valerie Chen, Robert Brennan, Graham Neubig

Agents are now used widely in the process of software development, but
building production-ready software engineering agents is a complex task.
Deploying software agents effectively requires flexibility in implementation
and experimentation, reliable and secure execution, and interfaces for users to
interact with agents. In this paper, we present the OpenHands Software Agent
SDK, a toolkit for implementing software development agents that satisfy these
desiderata. This toolkit is a complete architectural redesign of the agent
components of the popular OpenHands framework for software development agents,
which has 64k+ GitHub stars. To achieve flexibility, we design a simple
interface for implementing agents that requires only a few lines of code in the
default case, but is easily extensible to more complex, full-featured agents
with features such as custom tools, memory management, and more. For security
and reliability, it delivers seamless local-to-remote execution portability,
integrated REST/WebSocket services. For interaction with human users, it can
connect directly to a variety of interfaces, such as visual workspaces (VS
Code, VNC, browser), command-line interfaces, and APIs. Compared with existing
SDKs from OpenAI, Claude, and Google, OpenHands uniquely integrates native
sandboxed execution, lifecycle control, model-agnostic multi-LLM routing, and
built-in security analysis. Empirical results on SWE-Bench Verified and GAIA
benchmarks demonstrate strong performance. Put together, these elements allow
the OpenHands Software Agent SDK to provide a practical foundation for
prototyping, unlocking new classes of custom applications, and reliably
deploying agents at scale.

### Social and Information Networks

### 1. [A local eigenvector centrality](http://arxiv.org/pdf/2511.03608v1)

Authors: Ruaridh A. Clark, Francesca Arrigo, Agathe Bouis, Malcolm Macdonald

Eigenvector centrality is an established measure of global connectivity, from
which the importance and influence of nodes can be inferred. We introduce a
local eigenvector centrality that incorporates both local and global
connectivity. This new measure references prominent eigengaps and combines
their associated eigenspectrum, via the Euclidean norm, to detect centrality
that reflects the influence of prominent community structures. In contact
networks, with clearly defined community structures, local eigenvector
centrality is shown to identify similar but distinct distributions to
eigenvector centrality applied on each community in isolation and PageRank.
Discrepancies between the two eigenvector measures highlight nodes and
communities that do not conform to their defined local structures, e.g. nodes
with more connections outside of their defined community than within it. While
reference to PageRank's centrality assessment enables a mitigation strategy for
localisation effects inherent in eigenvector-based measures. In networks
without clearly defined communities, such as city road networks, local
eigenvector centrality is shown to identify both locally prominent and globally
connected hubs.

### 2. [Beyond Citations: Measuring Idea-level Knowledge Diffusion from Research to Journalism and Policy-making](http://arxiv.org/pdf/2511.03378v1)

Authors: Yangliu Fan, Kilian Buehling, Volker Stocker

Despite the importance of social science knowledge for various stakeholders,
measuring its diffusion into different domains remains a challenge. This study
uses a novel text-based approach to measure the idea-level diffusion of social
science knowledge from the research domain to the journalism and policy-making
domains. By doing so, we expand the detection of knowledge diffusion beyond the
measurements of direct references. Our study focuses on media effects theories
as key research ideas in the field of communication science. Using 72,703
documents (2000-2019) from three domains (i.e., research, journalism, and
policy-making) that mention these ideas, we count the mentions of these ideas
in each domain, estimate their domain-specific contexts, and track and compare
differences across domains and over time. Overall, we find that diffusion
patterns and dynamics vary considerably between ideas, with some ideas
diffusing between other domains, while others do not. Based on the embedding
regression approach, we compare contextualized meanings across domains and find
that the distances between research and policy are typically larger than
between research and journalism. We also find that ideas largely shift roles
across domains - from being the theories themselves in research to sense-making
in news to applied, administrative use in policy. Over time, we observe
semantic convergence mainly for ideas that are practically oriented. Our
results characterize the cross-domain diffusion patterns and dynamics of social
science knowledge at the idea level, and we discuss the implications for
measuring knowledge diffusion beyond citations.

### 3. [GMoPE:A Prompt-Expert Mixture Framework for Graph Foundation Models](http://arxiv.org/pdf/2511.03251v1)

Authors: Zhibin Wang, Zhixing Zhang, Shuqi Wang, Xuanting Xie, Zhao Kang

Graph Neural Networks (GNNs) have demonstrated impressive performance on
task-specific benchmarks, yet their ability to generalize across diverse
domains and tasks remains limited. Existing approaches often struggle with
negative transfer, scalability issues, and high adaptation costs. To address
these challenges, we propose GMoPE (Graph Mixture of Prompt-Experts), a novel
framework that seamlessly integrates the Mixture-of-Experts (MoE) architecture
with prompt-based learning for graphs. GMoPE leverages expert-specific prompt
vectors and structure-aware MoE routing to enable each expert to specialize in
distinct subdomains and dynamically contribute to predictions. To promote
diversity and prevent expert collapse, we introduce a soft orthogonality
constraint across prompt vectors, encouraging expert specialization and
facilitating a more balanced expert utilization. Additionally, we adopt a
prompt-only fine-tuning strategy that significantly reduces spatiotemporal
complexity during transfer. We validate GMoPE through extensive experiments
under various pretraining strategies and multiple downstream tasks. Results
show that GMoPE consistently outperforms state-of-the-art baselines and
achieves performance comparable to full parameter fine-tuning-while requiring
only a fraction of the adaptation overhead. Our work provides a principled and
scalable framework for advancing generalizable and efficient graph foundation
models.

### 4. [Characterising Global Platforms: Centralised, Decentralised, Federated, and Grassroots](http://arxiv.org/pdf/2511.03286v1)

Authors: Ehud Shapiro

Global digital platforms are software systems designed to serve entire
populations, with some already serving billions of people. We propose atomic
transactions-based multiagent transition systems and protocols as a formal
framework to study them; introduce essential agents -- minimal sets of agents
the removal of which makes communication impossible; and show that the
cardinality of essential agents partitions all global platforms into four
classes:
  1. Centralised -- one (the server)
  2. Decentralised -- finite $>1$ (bootstrap nodes)
  3. Federated -- infinite but not universal (all servers)
  4. Grassroots -- universal (all agents)
  Our illustrative formal example is a global social network, for which we
provide centralised, decentralised, federated, and grassroots specifications
via multiagent atomic transactions, and prove they satisfy basic correctness
properties. We discuss informally additional global platforms -- currencies,
``sharing economy'' apps, AI, and more. While this may be the first
characterisation of centralised, decentralised, and federated global platforms,
grassroots platforms have been formally defined previously, but using different
notions. Here, we prove that their original definition implies that all agents
are essential, placing grassroots platforms in a distinct class within the
broader formal context that includes all global platforms. This work provides
the first mathematical framework for classifying any global platform --
existing or imagined -- by providing a multiagent atomic-transactions
specification of it and determining the cardinality of the minimal set of
essential agents in the ensuing multiagent protocol. It thus

### 5. [Inter-Agent Trust Models: A Comparative Study of Brief, Claim, Proof, Stake, Reputation and Constraint in Agentic Web Protocol Design-A2A, AP2, ERC-8004, and Beyond](http://arxiv.org/pdf/2511.03434v1)

Authors: Botao 'Amber' Hu, Helena Rong

As the "agentic web" takes shape-billions of AI agents (often LLM-powered)
autonomously transacting and collaborating-trust shifts from human oversight to
protocol design. In 2025, several inter-agent protocols crystallized this
shift, including Google's Agent-to-Agent (A2A), Agent Payments Protocol (AP2),
and Ethereum's ERC-8004 "Trustless Agents," yet their underlying trust
assumptions remain under-examined. This paper presents a comparative study of
trust models in inter-agent protocol design: Brief (self- or third-party
verifiable claims), Claim (self-proclaimed capabilities and identity, e.g.
AgentCard), Proof (cryptographic verification, including zero-knowledge proofs
and trusted execution environment attestations), Stake (bonded collateral with
slashing and insurance), Reputation (crowd feedback and graph-based trust
signals), and Constraint (sandboxing and capability bounding). For each, we
analyze assumptions, attack surfaces, and design trade-offs, with particular
emphasis on LLM-specific fragilities-prompt injection,
sycophancy/nudge-susceptibility, hallucination, deception, and
misalignment-that render purely reputational or claim-only approaches brittle.
Our findings indicate no single mechanism suffices. We argue for
trustless-by-default architectures anchored in Proof and Stake to gate
high-impact actions, augmented by Brief for identity and discovery and
Reputation overlays for flexibility and social signals. We comparatively
evaluate A2A, AP2, ERC-8004 and related historical variations in academic
research under metrics spanning security, privacy, latency/cost, and social
robustness (Sybil/collusion/whitewashing resistance). We conclude with hybrid
trust model recommendations that mitigate reputation gaming and misinformed LLM
behavior, and we distill actionable design guidelines for safer, interoperable,
and scalable agent economies.

### Systems and Control

### 1. [MHE in Output Feedback Control of Uncertain Nonlinear Systems via IQCs](http://arxiv.org/pdf/2511.03221v1)

Authors: Yang Guo, Stefan Streif

We propose a moving horizon estimation (MHE) scheme for general nonlinear
constrained systems with parametric or static nonlinear uncertainties and a
predetermined state feedback controller that is assumed to robustly stabilize
the system in the absence of estimation errors. Leveraging integral quadratic
constraints (IQCs), we introduce a new notion of detectability that is robust
to possibly non-parametric uncertainties and verifiable in practice. Assuming
that the uncertain system driven by the controller satisfies this notion of
detectability, we provide an MHE formulation such that the closed-loop system
formed of the uncertain system, the controller and MHE is input-to-state stable
w.r.t. exogenous disturbances.

### 2. [Theoretical and Experimental Limitations of RoCoF Estimation](http://arxiv.org/pdf/2511.03249v1)

Authors: Gutierrez-Florensa, F. Sanniti, D. Tedeschi, L. Sigrist, A. Ortega, F. Milano

A precise estimation of the Rate of Change of Frequency (RoCoF) is crucial
for secure power system operation. In fact, RoCoF is strictly related to the
amount of the available physical and/or virtual inertia of the system and the
severity of the active power unbalance following a disturbance. For this
reason, it is widely exploited in different protection systems, e.g.,
Anti-Islanding, Under Frequency Load Shedding (UFLS) and wide-area protection
systems. The new paradigm of modern power systems, with a low-inertia and
converter-based generation assets, is increasing the transient severity, making
the frequency and the RoCoF estimation more complex and less precise for the
actual devices. This work addresses this issue by proposing a numerically
robust approach based on concepts inherited from differential geometry and
fluid mechanics. The proposed approach is then tested with high-sampling real
experimental measurements and used to develop a faster control logic for a
RoCoF-based UFLS control scheme. The proposed approach provides information to
protections regarding the nature of the contingency which can be used to
improve its response.

### 3. [A Digital Twin of Evaporative Thermo-Fluidic Process in Fixation Unit of DoD Inkjet Printers](http://arxiv.org/pdf/2511.03379v1)

Authors: Samarth Toolhally, Joeri Roelofs, Siep Weiland, Amritam Das

In inkjet printing, optimal paper moisture is crucial for print quality,
achieved through hot-air impingement in the fixation unit. This paper presents
a modular digital twin of the fixation unit, modeling the thermo-fluidic drying
process and monitoring its spatio-temporal performance. The novel approach
formulates the digital twin as an infinite-dimensional state estimator that
infers fixation states from limited sensor data, while remaining robust to
disturbances. Modularity is achieved through a graph-theoretic model, where
each node represents thermo-fluidic dynamics in different sections of the
fixation unit. Evaporation is modeled as a nonlinear boundary effect coupled
with node dynamics via Linear Fractional Representation. Using the Partial
Integral Equation (PIE) framework, we develop a unified approach for stability,
input-output analysis, simulation, and rapid prototyping, validated with
operational data from a commercial printer. An $\mathcal{H}_{\infty}$-optimal
Luenberger state estimator is then synthesized to estimate thermal states from
available sensor data, enabling real-time monitoring of spatio-temporal thermal
effects on paper sheets.

### 4. [Maximum Likelihood Estimation of Dynamic Sub-Networks with Missing Data](http://arxiv.org/pdf/2511.03391v1)

Authors: João Victor Galvão da Mata, Anders Hansson, Martin S. Andersen

Maximum likelihood estimation is effective for identifying dynamical systems,
but applying it to large networks becomes computationally prohibitive. This
paper introduces a maximum likelihood estimation method that enables
identification of sub-networks within complex interconnected systems without
estimating the entire network. The key insight is that under specific
topological conditions, a sub-network's parameters can be estimated using only
local measurements: signals within the target sub-network and those in the
directly connected to the so-called separator sub-network. This approach
significantly reduces computational complexity while enhancing privacy by
eliminating the need to share sensitive internal data across organizational
boundaries. We establish theoretical conditions for network separability,
derive the probability density function for the sub-network, and demonstrate
the method's effectiveness through numerical examples.

### 5. [An Alternative Derivation and Optimal Design Method of the Generalized Bilinear Transformation for Discretizing Analog Systems](http://arxiv.org/pdf/2511.03403v1)

Authors: Shen Chen, Yanlong Li, Jiamin Cui, Wei Yao, Jisong Wang, Yixin Tian, Chaohou Liu, Yang Yang, Jiaxi Ying, Zeng Liu, Jinjun Liu

A popular method for designing digital systems is transforming the transfer
function of the corresponding analog systems from the continuous-time domain
(s-domain) into the discrete-time domain (z-domain) using the Euler or Tustin
method. We demonstrate that these transformations are two specific forms of the
Generalized Bilinear Transformation (GBT) with a design parameter, $\alpha$.
However, the physical meaning and optimal design method for this parameter are
not sufficiently studied. In this paper, we propose an alternative derivation
of the GBT derived by employing a new hexagonal shape to approximate the
enclosed area of the error function, and we define the parameter $\alpha$ as
the shape factor. The physical meaning of the shape factor is firstly revealed,
which equals to the percentage of the backward rectangular ratio of the
proposed hexagonal shape. We demonstrate that the stable range of the shape
factor is [0.5, 1] through domain mapping. Depending on the operating
frequencies and the shape factor, we observe two distinct distortion modes,
i.e., the magnitude and phase distortion. We proceed to develop an optimal
design method for the shape factor based on an objective function in form of
the normalized magnitude or phase error. Finally, a low-pass filter (LPF) is
designed and tested to verify the effectiveness of the proposed method by
comparing the theoretical calculations with the experimental results.

### 6. [Data-driven Modeling of Grid-following Control in Grid-connected Converters](http://arxiv.org/pdf/2511.03494v1)

Authors: Amir Bahador Javadi, Philip Pong

As power systems evolve with the integration of renewable energy sources and
the implementation of smart grid technologies, there is an increasing need for
flexible and scalable modeling approaches capable of accurately capturing the
complex dynamics of modern grids. To meet this need, various methods, such as
the sparse identification of nonlinear dynamics and deep symbolic regression,
have been developed to identify dynamical systems directly from data. In this
study, we examine the application of a converter-based resource as a
replacement for a traditional generator within a lossless transmission line
linked to an infinite bus system. This setup is used to generate synthetic data
in grid-following control mode, enabling the evaluation of these methods in
effectively capturing system dynamics.

### 7. [Powered Descent Trajectory Optimization of Chandrayaan-3 using Radau Collocation and Controllable Sets](http://arxiv.org/pdf/2511.03594v1)

Authors: Suraj Kumar, Aditya Rallapalli, Ashok Kumar Kakula, Bharat Kumar GVP

India achieved a significant milestone on August $23^{\text{rd}}$ 2023,
becoming the fourth country to accomplish a soft landing on the Moon. This
paper presents the powered descent trajectory design for the Chandrayaan-3
mission. The optimization framework is based on pseudospectral Radau
collocation, and controllability-based waypoint refinement is employed to
further enhance the robustness of the trajectory against state and control
perturbations. Furthermore, the trade-off between fuel consumption and
robustness is explicitly quantified, providing insights into the practical
considerations of mission planning.

### 8. [Artificial-reference tracking MPC with probabilistically validated performance on industrial embedded systems](http://arxiv.org/pdf/2511.03603v1)

Authors: Victor Gracia, Pablo Krupa, Filiberto Fele, Teodoro Alamo

Industrial embedded systems are typically used to execute simple control
algorithms due to their low computational resources. Despite these limitations,
the implementation of advanced control techniques such as Model Predictive
Control (MPC) has been explored by the control community in recent years,
typically considering simple linear formulations or explicit ones to facilitate
the online computation of the control input. These simplifications often lack
features and properties that are desirable in real-world environments. In this
article, we present an efficient implementation for embedded systems of MPC for
tracking with artificial reference, solved via a recently developed
structure-exploiting first-order method. This formulation is tailored to a wide
range of applications by incorporating essential practical features at a small
computational cost, including integration with an offset-free scheme, back-off
parameters that enable constraint tightening, and soft constraints that
preserve feasibility under disturbances or plant-model mismatch. We accompany
this with a framework for probabilistic performance validation of the
closed-loop system over long-term operation. We illustrate the applicability of
the approach on a Programmable Logic Controller (PLC), incorporated in a
hardware-in-the-loop setup to control a nonlinear continuous stirred-tank
reactor. The behavior of the closed-loop system is probabilistically validated
with respect to constraint violations and the number of iterations required at
each time step by the MPC optimization algorithm.

### 9. [A Constant-Gain Equation-Error Framework for Airliner Aerodynamic Monitoring Using QAR Data](http://arxiv.org/pdf/2511.03678v1)

Authors: Ruiying Wen, Yuntao Dai, Hongyong Wang

Monitoring the in-service aerodynamic performance of airliners is critical
for operational efficiency and safety, but using operational Quick Access
Recorder (QAR) data for this purpose presents significant challenges. This
paper first establishes that the absence of key parameters, particularly
aircraft moments of inertia, makes conventional state-propagation filters
fundamentally unsuitable for this application. This limitation necessitates a
decoupled, Equation-Error Method (EEM). However, we then demonstrate through a
comparative analysis that standard recursive estimators with time-varying
gains, such as Recursive Least Squares (RLS), also fail within an EEM
framework, exhibiting premature convergence or instability when applied to
low-excitation cruise data. To overcome these dual challenges, we propose and
validate the Constant-Gain Equation-Error Method (CG-EEM). This framework
employs a custom estimator with a constant, Kalman-like gain, which is
perfectly suited to the stationary, low-signal-to-noise characteristics of
cruise flight. The CG-EEM is extensively validated on a large, multi-fleet
dataset of over 200 flights, where it produces highly consistent, physically
plausible aerodynamic parameters and correctly identifies known performance
differences between aircraft types. The result is a robust, scalable, and
computationally efficient tool for fleet-wide performance monitoring and the
early detection of performance degradation.

### 10. [D2-UC: A Distributed-Distributed Quantum-Classical Framework for Unit Commitment](http://arxiv.org/pdf/2511.03104v1)

Authors: Milad Hasanzadeh, Amin Kargarian

This paper introduces D2-UC, a quantum-ready framework for the unit
commitment (UC) problem that prepares UC for near-term hybrid quantum-classical
solvers by combining distributed classical decomposition with distributed
quantum execution. We reformulate deterministic and stochastic UC into a
three-block alternating direction method of multipliers (ADMM): (i) a convex
quadratic subproblem for dispatch and reserves, (ii) a binary subproblem
expressed as a quadratic unconstrained binary optimization (QUBO), and (iii) a
proximal slack update for consensus. The core contributions are fivefold.
First, we demonstrate how the full UC problem can be expressed as a single
monolithic QUBO, establishing a direct interface to quantum solvers. Second, we
decompose this large binary block into three type-specific QUBOs for
commitment, startup, and shutdown, making the problem more tractable but
revealing slower ADMM convergence. Third, we restore local logical couplings
through per-unit-time micro-QUBOs, which accelerate convergence. Fourth, we
batch micro-QUBOs into K non-overlapping block-diagonal problems, reducing many
subproblems to a fixed number of solver-ready QUBOs per iteration, compatible
with distributed variational quantum eigensolvers (DVQE). Fifth, we integrate
an accept-if-better safeguard with DVQE to stabilize hybrid updates and prevent
oscillations. Case studies confirm that the proposed methods deliver feasible
schedules, faster convergence, and QUBO sizes aligned with current and
near-term quantum hardware capabilities. All detailed data, codes, and
parameter values are available at
https://github.com/LSU-RAISE-LAB/3B-ADMM-UC-DVQE .

### Machine Learning (Statistics Category)

### 1. [Provable Accelerated Bayesian Optimization with Knowledge Transfer](http://arxiv.org/pdf/2511.03125v1)

Authors: Haitao Lin, Boxin Zhao, Mladen Kolar, Chong Liu

We study how Bayesian optimization (BO) can be accelerated on a target task
with historical knowledge transferred from related source tasks. Existing works
on BO with knowledge transfer either do not have theoretical guarantees or
achieve the same regret as BO in the non-transfer setting,
$\tilde{\mathcal{O}}(\sqrt{T \gamma_f})$, where $T$ is the number of
evaluations of the target function and $\gamma_f$ denotes its information gain.
In this paper, we propose the DeltaBO algorithm, in which a novel
uncertainty-quantification approach is built on the difference function
$\delta$ between the source and target functions, which are allowed to belong
to different reproducing kernel Hilbert spaces (RKHSs). Under mild assumptions,
we prove that the regret of DeltaBO is of order $\tilde{\mathcal{O}}(\sqrt{T
(T/N + \gamma_\delta)})$, where $N$ denotes the number of evaluations from
source tasks and typically $N \gg T$. In many applications, source and target
tasks are similar, which implies that $\gamma_\delta$ can be much smaller than
$\gamma_f$. Empirical studies on both real-world hyperparameter tuning tasks
and synthetic functions show that DeltaBO outperforms other baseline methods
and support our theoretical claims.

### 2. [Cross-Modal Alignment via Variational Copula Modelling](http://arxiv.org/pdf/2511.03196v1)

Authors: Feng Wu, Tsai Hor Chan, Fuying Wang, Guosheng Yin, Lequan Yu

Various data modalities are common in real-world applications (e.g.,
electronic health records, medical images and clinical notes in healthcare). It
is essential to develop multimodal learning methods to aggregate various
information from multiple modalities. The main challenge is how to
appropriately align and fuse the representations of different modalities into a
joint distribution. Existing methods mainly rely on concatenation or the
Kronecker product, oversimplifying the interaction structure between modalities
and indicating a need to model more complex interactions. Additionally, the
joint distribution of latent representations with higher-order interactions is
underexplored. Copula is a powerful statistical structure for modelling the
interactions among variables, as it naturally bridges the joint distribution
and marginal distributions of multiple variables. We propose a novel
copula-driven multimodal learning framework, which focuses on learning the
joint distribution of various modalities to capture the complex interactions
among them. The key idea is to interpret the copula model as a tool to align
the marginal distributions of the modalities efficiently. By assuming a
Gaussian mixture distribution for each modality and a copula model on the joint
distribution, our model can generate accurate representations for missing
modalities. Extensive experiments on public MIMIC datasets demonstrate the
superior performance of our model over other competitors. The code is available
at https://github.com/HKU-MedAI/CMCM.

### 3. [Provable Separations between Memorization and Generalization in Diffusion Models](http://arxiv.org/pdf/2511.03202v1)

Authors: Zeqi Ye, Qijie Zhu, Molei Tao, Minshuo Chen

Diffusion models have achieved remarkable success across diverse domains, but
they remain vulnerable to memorization -- reproducing training data rather than
generating novel outputs. This not only limits their creative potential but
also raises concerns about privacy and safety. While empirical studies have
explored mitigation strategies, theoretical understanding of memorization
remains limited. We address this gap through developing a dual-separation
result via two complementary perspectives: statistical estimation and network
approximation. From the estimation side, we show that the ground-truth score
function does not minimize the empirical denoising loss, creating a separation
that drives memorization. From the approximation side, we prove that
implementing the empirical score function requires network size to scale with
sample size, spelling a separation compared to the more compact network
representation of the ground-truth score function. Guided by these insights, we
develop a pruning-based method that reduces memorization while maintaining
generation quality in diffusion transformers.

### 4. [RKUM: An R Package for Robust Kernel Unsupervised Methods](http://arxiv.org/pdf/2511.03216v1)

Authors: Md Ashad Alam

RKUM is an R package developed for implementing robust kernel-based
unsupervised methods. It provides functions for estimating the robust kernel
covariance operator (CO) and the robust kernel cross-covariance operator (CCO)
using generalized loss functions instead of the conventional quadratic loss.
These operators form the foundation of robust kernel learning and enable
reliable analysis under contaminated or noisy data conditions. The package
includes implementations of robust kernel canonical correlation analysis
(Kernel CCA), as well as the influence function (IF) for both standard and
multiple kernel CCA frameworks. The influence function quantifies sensitivity
and helps detect influential or outlying observations across two-view and
multi-view datasets. Experiments using synthesized two-view and multi-view data
demonstrate that the IF of the standard kernel CCA effectively identifies
outliers, while the robust kernel methods implemented in RKUM exhibit reduced
sensitivity to contamination. Overall, RKUM provides an efficient and
extensible platform for robust kernel-based analysis in high-dimensional data
applications.

### 5. [Silenced Biases: The Dark Side LLMs Learned to Refuse](http://arxiv.org/pdf/2511.03369v1)

Authors: Rom Himelstein, Amit LeVi, Brit Youngmann, Yaniv Nemcovsky, Avi Mendelson

Safety-aligned large language models (LLMs) are becoming increasingly
widespread, especially in sensitive applications where fairness is essential
and biased outputs can cause significant harm. However, evaluating the fairness
of models is a complex challenge, and approaches that do so typically utilize
standard question-answer (QA) styled schemes. Such methods often overlook
deeper issues by interpreting the model's refusal responses as positive
fairness measurements, which creates a false sense of fairness. In this work,
we introduce the concept of silenced biases, which are unfair preferences
encoded within models' latent space and are effectively concealed by
safety-alignment. Previous approaches that considered similar indirect biases
often relied on prompt manipulation or handcrafted implicit queries, which
present limited scalability and risk contaminating the evaluation process with
additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to
uncover these biases by employing activation steering to reduce model refusals
during QA. SBB supports easy expansion to new demographic groups and subjects,
presenting a fairness evaluation framework that encourages the future
development of fair models and tools beyond the masking effects of alignment
training. We demonstrate our approach over multiple LLMs, where our findings
expose an alarming distinction between models' direct responses and their
underlying fairness issues.

### 6. [Why Less is More (Sometimes): A Theory of Data Curation](http://arxiv.org/pdf/2511.03492v1)

Authors: Elvis Dohmatob, Mohammad Pezeshki, Reyhane Askari-Hemmat

This paper introduces a theoretical framework to resolve a central paradox in
modern machine learning: When is it better to use less data? This question has
become critical as classical scaling laws suggesting ``more is more'' (Sun et
al., 2025) are challenged by methods like LIMO (``less is more'') and s1 (Ye et
al., 2025; Muenighoff et al., 2025), which achieve superior performance with
small, aggressively curated datasets. Here, we study data curation strategies
where an imperfect oracle selects the training examples according to their
difficulty and correctness. Our results provide exact scaling law curves for
test error under both label-agnostic and label-aware curation rules, revealing
when and why keeping only a subset of data can improve generalization. In
contrast to classical scaling laws, we show that under certain conditions,
small curated datasets can outperform full datasets, and we provide analytical
conditions for this by deriving precise phase transition curves tied to data
size and quality. We validate these theoretical claims with empirical results
on ImageNet, confirming our predictions about when curation improves accuracy
and can even mitigate model collapse. Furthermore, our framework provides a
principled explanation for the contradictory curation strategies recently
observed in LLM mathematical reasoning.

### 7. [Towards Formalizing Reinforcement Learning Theory](http://arxiv.org/pdf/2511.03618v1)

Authors: Shangtong Zhang

In this paper, we formalize the almost sure convergence of $Q$-learning and
linear temporal difference (TD) learning with Markovian samples using the Lean
4 theorem prover based on the Mathlib library. $Q$-learning and linear TD are
among the earliest and most influential reinforcement learning (RL) algorithms.
The investigation of their convergence properties is not only a major research
topic during the early development of the RL field but also receives increasing
attention nowadays. This paper formally verifies their almost sure convergence
in a unified framework based on the Robbins-Siegmund theorem. The framework
developed in this work can be easily extended to convergence rates and other
modes of convergence. This work thus makes an important step towards fully
formalizing convergent RL results. The code is available at
https://github.com/ShangtongZhang/rl-theory-in-lean.

### 8. [Colorectal Cancer Histopathological Grading using Multi-Scale Federated Learning](http://arxiv.org/pdf/2511.03693v1)

Authors: Md Ahasanul Arafath, Abhijit Kumar Ghosh, Md Rony Ahmed, Sabrin Afroz, Minhazul Hosen, Md Hasan Moon, Md Tanzim Reza, Md Ashad Alam

Colorectal cancer (CRC) grading is a critical prognostic factor but remains
hampered by inter-observer variability and the privacy constraints of
multi-institutional data sharing. While deep learning offers a path to
automation, centralized training models conflict with data governance
regulations and neglect the diagnostic importance of multi-scale analysis. In
this work, we propose a scalable, privacy-preserving federated learning (FL)
framework for CRC histopathological grading that integrates multi-scale feature
learning within a distributed training paradigm. Our approach employs a
dual-stream ResNetRS50 backbone to concurrently capture fine-grained nuclear
detail and broader tissue-level context. This architecture is integrated into a
robust FL system stabilized using FedProx to mitigate client drift across
heterogeneous data distributions from multiple hospitals. Extensive evaluation
on the CRC-HGD dataset demonstrates that our framework achieves an overall
accuracy of 83.5%, outperforming a comparable centralized model (81.6%).
Crucially, the system excels in identifying the most aggressive Grade III
tumors with a high recall of 87.5%, a key clinical priority to prevent
dangerous false negatives. Performance further improves with higher
magnification, reaching 88.0% accuracy at 40x. These results validate that our
federated multi-scale approach not only preserves patient privacy but also
enhances model performance and generalization. The proposed modular pipeline,
with built-in preprocessing, checkpointing, and error handling, establishes a
foundational step toward deployable, privacy-aware clinical AI for digital
pathology.

### 9. [A Support-Set Algorithm for Optimization Problems with Nonnegative and Orthogonal Constraints](http://arxiv.org/pdf/2511.03443v1)

Authors: Lei Wang, Xin Liu, Xiaojun Chen

In this paper, we investigate optimization problems with nonnegative and
orthogonal constraints, where any feasible matrix of size $n \times p$ exhibits
a sparsity pattern such that each row accommodates at most one nonzero entry.
Our analysis demonstrates that, by fixing the support set, the global solution
of the minimization subproblem for the proximal linearization of the objective
function can be computed in closed form with at most $n$ nonzero entries.
Exploiting this structural property offers a powerful avenue for dramatically
enhancing computational efficiency. Guided by this insight, we propose a
support-set algorithm preserving strictly the feasibility of iterates. A
central ingredient is a strategically devised update scheme for support sets
that adjusts the placement of nonzero entries. We establish the global
convergence of the support-set algorithm to a first-order stationary point, and
show that its iteration complexity required to reach an $\epsilon$-approximate
first-order stationary point is $O (\epsilon^{-2})$. Numerical results are
strongly in favor of our algorithm in real-world applications, including
nonnegative PCA, clustering, and community detection.

### 10. [Vector-valued self-normalized concentration inequalities beyond sub-Gaussianity](http://arxiv.org/pdf/2511.03606v1)

Authors: Diego Martinez-Taboada, Tomas Gonzalez, Aaditya Ramdas

The study of self-normalized processes plays a crucial role in a wide range
of applications, from sequential decision-making to econometrics. While the
behavior of self-normalized concentration has been widely investigated for
scalar-valued processes, vector-valued processes remain comparatively
underexplored, especially outside of the sub-Gaussian framework. In this
contribution, we provide concentration bounds for self-normalized processes
with light tails beyond sub-Gaussianity (such as Bennett or Bernstein bounds).
We illustrate the relevance of our results in the context of online linear
regression, with applications in (kernelized) linear bandits.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-11-06 PST.

### 1. [Automatic detection of sister chromatid exchanges using machine learning models and image analysis algorithms](https://www.nature.com/articles/s41598-025-22608-9)

Authors: Mizuo Teraoka et al.

### 2. [Effective descriptions of bosonic systems can be considered complete](https://www.nature.com/articles/s41467-025-64872-3)

Authors: Francesco Arzani et al.

### 3. [Emulating human-like adaptive vision for efficient and flexible machine visual perception](https://www.nature.com/articles/s42256-025-01130-7)

Authors: Yulin Wang et al.

### 4. [An improved facial emotion recognition system using convolutional neural network for the optimization of human robot interaction](https://www.nature.com/articles/s41598-025-22835-0)

Authors: Ravi Raj et al.

### 5. [StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations](https://www.nature.com/articles/s44172-025-00520-4)

Authors: Cunshi Wang et al.

### 6. [Interpretable arrhythmia detection in ECG scans using deep learning ensembles: a genetic programming approach](https://www.nature.com/articles/s41746-025-01932-4)

Authors: Arkadiusz Czerwinski et al.

### 7. [Deep learning for sports motion recognition with a high-precision framework for performance enhancement](https://www.nature.com/articles/s41598-025-22701-z)

Authors: Yang Yang et al.

### 8. [A bird target detection model designed for substation scenarios](https://www.nature.com/articles/s41598-025-22829-y)

Authors: Chunxue Shao et al.

### 9. [Densing law of LLMs](https://www.nature.com/articles/s42256-025-01137-0)

Authors: Chaojun Xiao et al.

### 10. [PUNet: a lightweight parallel U-Net architecture integrating Mamba–CNN for high-precision image segmentation](https://www.nature.com/articles/s41598-025-22862-x)

Authors: Zhaoyan Xie et al.

### 11. [Precision detection of micro-damage on conveyor belt surfaces using laser scanning and deep learning techniques](https://www.nature.com/articles/s41598-025-22818-1)

Authors: Yingxiu Li et al.

### 12. [Bi-directional ConvLSTM networks for early recognition of human activities and action prediction](https://www.nature.com/articles/s41598-025-22898-z)

Authors: M. Ashwin Shenoy et al.

### 13. [Tracking control of humanoid manipulator using sliding mode with neural network and disturbance observer](https://www.nature.com/articles/s41598-025-22825-2)

Authors: Yina Wang et al.

