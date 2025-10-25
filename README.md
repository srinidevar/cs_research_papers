# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-24 17:00:25.856216 PST.

### Artificial Intelligence

### 1. [Human-Centered LLM-Agent System for Detecting Anomalous Digital Asset Transactions](http://arxiv.org/pdf/2510.20102v1)

Authors: Gyuyeon Na, Minjung Park, Hyeonjeong Cha, Sangmi Chai

We present HCLA, a human-centered multi-agent system for anomaly detection in
digital asset transactions. The system links three roles: Parsing, Detection,
and Explanation, into a conversational workflow that lets non-experts ask
questions in natural language, inspect structured analytics, and obtain
context-aware rationales. Implemented with an open-source web UI, HCLA
translates user intents into a schema for a classical detector (XGBoost in our
prototype) and returns narrative explanations grounded in the underlying
features. On a labeled Bitcoin mixing dataset (Wasabi Wallet, 2020-2024), the
baseline detector reaches strong accuracy, while HCLA adds interpretability and
interactive refinement. We describe the architecture, interaction loop,
dataset, evaluation protocol, and limitations, and discuss how a
human-in-the-loop design improves transparency and trust in financial
forensics.

### 2. [The Verification-Value Paradox: A Normative Critique of Gen AI in Legal Practice](http://arxiv.org/pdf/2510.20109v1)

Authors: Joshua Yuvaraj

It is often claimed that machine learning-based generative AI products will
drastically streamline and reduce the cost of legal practice. This enthusiasm
assumes lawyers can effectively manage AI's risks. Cases in Australia and
elsewhere in which lawyers have been reprimanded for submitting inaccurate
AI-generated content to courts suggest this paradigm must be revisited. This
paper argues that a new paradigm is needed to evaluate AI use in practice,
given (a) AI's disconnection from reality and its lack of transparency, and (b)
lawyers' paramount duties like honesty, integrity, and not to mislead the
court. It presents an alternative model of AI use in practice that more
holistically reflects these features (the verification-value paradox). That
paradox suggests increases in efficiency from AI use in legal practice will be
met by a correspondingly greater imperative to manually verify any outputs of
that use, rendering the net value of AI use often negligible to lawyers. The
paper then sets out the paradox's implications for legal practice and legal
education, including for AI use but also the values that the paradox suggests
should undergird legal practice: fidelity to the truth and civic
responsibility.

### 3. [TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning](http://arxiv.org/pdf/2510.20188v1)

Authors: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

Large Language Models generate complex reasoning chains that reveal their
decision-making, yet verifying the faithfulness and harmlessness of these
intermediate steps remains a critical unsolved problem. Existing auditing
methods are centralized, opaque, and hard to scale, creating significant risks
for deploying proprietary models in high-stakes domains. We identify four core
challenges: (1) Robustness: Centralized auditors are single points of failure,
prone to bias or attacks. (2) Scalability: Reasoning traces are too long for
manual verification. (3) Opacity: Closed auditing undermines public trust. (4)
Privacy: Exposing full reasoning risks model theft or distillation. We propose
TRUST, a transparent, decentralized auditing framework that overcomes these
limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing
correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG
decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A
blockchain ledger that records all verification decisions for public
accountability. (4) Privacy-preserving segmentation, sharing only partial
reasoning steps to protect proprietary logic. We provide theoretical guarantees
for the security and economic incentives of the TRUST framework. Experiments
across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math,
medical, science, humanities) show TRUST effectively detects reasoning flaws
and remains robust against adversarial auditors. Our work pioneers
decentralized AI auditing, offering a practical path toward safe and
trustworthy LLM deployment.

### 4. [Merge and Conquer: Evolutionarily Optimizing AI for 2048](http://arxiv.org/pdf/2510.20205v1)

Authors: Maggie Bai, Ava Kim Cohen, Eleanor Koss, Charlie Lichtenbaum

Optimizing artificial intelligence (AI) for dynamic environments remains a
fundamental challenge in machine learning research. In this paper, we examine
evolutionary training methods for optimizing AI to solve the game 2048, a 2D
sliding puzzle. 2048, with its mix of strategic gameplay and stochastic
elements, presents an ideal playground for studying decision-making, long-term
planning, and dynamic adaptation. We implemented two distinct systems: a
two-agent metaprompting system where a "thinker" large language model (LLM)
agent refines gameplay strategies for an "executor" LLM agent, and a
single-agent system based on refining a value function for a limited Monte
Carlo Tree Search. We also experimented with rollback features to avoid
performance degradation. Our results demonstrate the potential of evolutionary
refinement techniques in improving AI performance in non-deterministic
environments. The single-agent system achieved substantial improvements, with
an average increase of 473.2 points per cycle, and with clear upward trends
(correlation $\rho$=0.607) across training cycles. The LLM's understanding of
the game grew as well, shown in its development of increasingly advanced
strategies. Conversely, the two-agent system did not garner much improvement,
highlighting the inherent limits of meta-prompting.

### 5. [Individualized Cognitive Simulation in Large Language Models: Evaluating Different Cognitive Representation Methods](http://arxiv.org/pdf/2510.20252v1)

Authors: Tianyi Zhang, Xiaolin Zhou, Yunzhe Wang, Erik Cambria, David Traum, Rui Mao

Individualized cognitive simulation (ICS) aims to build computational models
that approximate the thought processes of specific individuals. While large
language models (LLMs) convincingly mimic surface-level human behavior such as
role-play, their ability to simulate deeper individualized cognitive processes
remains poorly understood. To address this gap, we introduce a novel task that
evaluates different cognitive representation methods in ICS. We construct a
dataset from recently published novels (later than the release date of the
tested LLMs) and propose an 11-condition cognitive evaluation framework to
benchmark seven off-the-shelf LLMs in the context of authorial style emulation.
We hypothesize that effective cognitive representations can help LLMs generate
storytelling that better mirrors the original author. Thus, we test different
cognitive representations, e.g., linguistic features, concept mappings, and
profile-based information. Results show that combining conceptual and
linguistic features is particularly effective in ICS, outperforming static
profile-based cues in overall evaluation. Importantly, LLMs are more effective
at mimicking linguistic style than narrative structure, underscoring their
limits in deeper cognitive simulation. These findings provide a foundation for
developing AI systems that adapt to individual ways of thinking and expression,
advancing more personalized and human-aligned creative technologies.

### 6. [Using Large Language Models for Abstraction of Planning Domains - Extended Version](http://arxiv.org/pdf/2510.20258v1)

Authors: Bita Banihashemi, Megh Patel, Yves Lespérance

Generating an abstraction of a dynamic domain that aligns with a given
purpose remains a significant challenge given that the choice of such an
abstraction can impact an agent's ability to plan, reason, and provide
explanations effectively. We model the agent's concrete behaviors in PDDL and
investigate the use of in-context learning with large language models (LLMs)
for the generation of abstract PDDL domains and problem instances, given an
abstraction objective specified in natural language. The benchmark examples we
use are new and have not been part of the data any LLMs have been trained on.
We consider three categories of abstractions: abstraction of choice of
alternative concrete actions, abstraction of sequences of concrete actions, and
abstraction of action/predicate parameters, as well as combinations of these.
The generated abstract PDDL domains and problem instances are then checked by
symbolic validation tools as well as human experts. Our experiments show that
GPT-4o can generally synthesize useful planning domain abstractions in simple
settings, although it is better at abstracting over actions than over the
associated fluents.

### 7. [Classical Feature Embeddings Help in BERT-Based Human Mobility Prediction](http://arxiv.org/pdf/2510.20275v1)

Authors: Yunzhi Liu, Haokai Tan, Rushi Kanjaria, Lihuan Li, Flora D. Salim

Human mobility forecasting is crucial for disaster relief, city planning, and
public health. However, existing models either only model location sequences or
include time information merely as auxiliary input, thereby failing to leverage
the rich semantic context provided by points of interest (POIs). To address
this, we enrich a BERT-based mobility model with derived temporal descriptors
and POI embeddings to better capture the semantics underlying human movement.
We propose STaBERT (Semantic-Temporal aware BERT), which integrates both POI
and temporal information at each location to construct a unified, semantically
enriched representation of mobility. Experimental results show that STaBERT
significantly improves prediction accuracy: for single-city prediction, the
GEO-BLEU score improved from 0.34 to 0.75; for multi-city prediction, from 0.34
to 0.56.

### 8. [Multi-Step Reasoning for Embodied Question Answering via Tool Augmentation](http://arxiv.org/pdf/2510.20310v1)

Authors: Mingliang Zhai, Hansheng Liang, Xiaomeng Fan, Zhi Gao, Chuanhao Li, Che Sun, Xu Bin, Yuwei Wu, Yunde Jia

Embodied Question Answering (EQA) requires agents to explore 3D environments
to obtain observations and answer questions related to the scene. Existing
methods leverage VLMs to directly explore the environment and answer questions
without explicit thinking or planning, which limits their reasoning ability and
results in excessive or inefficient exploration as well as ineffective
responses. In this paper, we introduce ToolEQA, an agent that integrates
external tools with multi-step reasoning, where external tools can provide more
useful information for completing the task, helping the model derive better
exploration directions in the next step of reasoning and thus obtaining
additional effective information. This enables ToolEQA to generate more
accurate responses with a shorter exploration distance. To enhance the model's
ability for tool-usage and multi-step reasoning, we further design a novel EQA
data generation pipeline that automatically constructs large-scale EQA tasks
with reasoning trajectories and corresponding answers. Based on the pipeline,
we collect the EQA-RT dataset that contains about 18K tasks, divided into a
training set EQA-RT-Train, and two test sets EQA-RT-Seen (scenes overlapping
with the training set) and EQA-RT-Unseen (novel scenes). Experiments on
EQA-RT-Seen and EQA-RT-Unseen show that ToolEQA improves the success rate by
9.2~20.2% over state-of-the-art baselines, while outperforming the zero-shot
ToolEQA by 10% in success rate. In addition, ToolEQA also achieves
state-of-the-art performance on the HM-EQA, OpenEQA, and EXPRESS-Bench
datasets, demonstrating its generality. Our homepage see
https://tooleqa.github.io.

### 9. [Bias by Design? How Data Practices Shape Fairness in AI Healthcare Systems](http://arxiv.org/pdf/2510.20332v1)

Authors: Anna Arias-Duart, Maria Eugenia Cardello, Atia Cortés

Artificial intelligence (AI) holds great promise for transforming healthcare.
However, despite significant advances, the integration of AI solutions into
real-world clinical practice remains limited. A major barrier is the quality
and fairness of training data, which is often compromised by biased data
collection practices. This paper draws on insights from the AI4HealthyAging
project, part of Spain's national R&D initiative, where our task was to detect
biases during clinical data collection. We identify several types of bias
across multiple use cases, including historical, representation, and
measurement biases. These biases manifest in variables such as sex, gender,
age, habitat, socioeconomic status, equipment, and labeling. We conclude with
practical recommendations for improving the fairness and robustness of clinical
problem design and data collection. We hope that our findings and experience
contribute to guiding future projects in the development of fairer AI systems
in healthcare.

### 10. [Collateral Damage Assessment Model for AI System Target Engagement in Military Operations](http://arxiv.org/pdf/2510.20337v1)

Authors: Clara Maathuis, Kasper Cools

In an era where AI (Artificial Intelligence) systems play an increasing role
in the battlefield, ensuring responsible targeting demands rigorous assessment
of potential collateral effects. In this context, a novel collateral damage
assessment model for target engagement of AI systems in military operations is
introduced. The model integrates temporal, spatial, and force dimensions within
a unified Knowledge Representation and Reasoning (KRR) architecture following a
design science methodological approach. Its layered structure captures the
categories and architectural components of the AI systems to be engaged
together with corresponding engaging vectors and contextual aspects. At the
same time, spreading, severity, likelihood, and evaluation metrics are
considered in order to provide a clear representation enhanced by transparent
reasoning mechanisms. Further, the model is demonstrated and evaluated through
instantiation which serves as a basis for further dedicated efforts that aim at
building responsible and trustworthy intelligent systems for assessing the
effects produced by engaging AI systems in military operations.

### Hardware Architecture

### 1. [HALOC-AxA: An Area/-Energy-Efficient Approximate Adder for Image Processing Application](http://arxiv.org/pdf/2510.20137v1)

Authors: Hasnain A. Ziad, Ashiq A. Sakib

The design of approximate adders has been widely researched to advance
energy-efficient hardware for computation-intensive multimedia applications,
such as image, audio, or video processing. The design of approximate adders has
been widely researched to advance energy-efficient hardware for computation
intensive multimedia applications, such as image/audio/video processing.
Several static and dynamic approximate adders exist in the literature, each of
which endeavors to balance the conflicting demands of high performance,
computational accuracy, and energy efficiency. This work introduces a novel
approximate adder that is more energy- and area-efficient than existing adders,
while achieving improved or comparable accuracy, as demonstrated by simulation
results. The proposed adder's ability to digitally reconstruct high quality
images is further demonstrated by the deployment of the design for an image
processing task.

### 2. [Squire: A General-Purpose Accelerator to Exploit Fine-Grain Parallelism on Dependency-Bound Kernels](http://arxiv.org/pdf/2510.20400v1)

Authors: Rubén Langarita, Jesús Alastruey-Benedé, Pablo Ibáñez-Marín, Santiago Marco-Sola, Miquel Moretó, Adrià Armejach

Multiple HPC applications are often bottlenecked by compute-intensive kernels
implementing complex dependency patterns (data-dependency bound). Traditional
general-purpose accelerators struggle to effectively exploit fine-grain
parallelism due to limitations in implementing convoluted data-dependency
patterns (like SIMD) and overheads due to synchronization and data transfers
(like GPGPUs). In contrast, custom FPGA and ASIC designs offer improved
performance and energy efficiency at a high cost in hardware design and
programming complexity and often lack the flexibility to process different
workloads. We propose Squire, a general-purpose accelerator designed to exploit
fine-grain parallelism effectively on dependency-bound kernels. Each Squire
accelerator has a set of general-purpose low-power in-order cores that can
rapidly communicate among themselves and directly access data from the L2
cache. Our proposal integrates one Squire accelerator per core in a typical
multicore system, allowing the acceleration of dependency-bound kernels within
parallel tasks with minimal software changes. As a case study, we evaluate
Squire's effectiveness by accelerating five kernels that implement complex
dependency patterns. We use three of these kernels to build an end-to-end
read-mapping tool that will be used to evaluate Squire. Squire obtains speedups
up to 7.64$\times$ in dynamic programming kernels. Overall, Squire provides an
acceleration for an end-to-end application of 3.66$\times$. In addition, Squire
reduces energy consumption by up to 56% with a minimal area overhead of 10.5%
compared to a Neoverse-N1 baseline.

### 3. [In-DRAM True Random Number Generation Using Simultaneous Multiple-Row Activation: An Experimental Study of Real DRAM Chips](http://arxiv.org/pdf/2510.20269v1)

Authors: Ismail Emir Yuksel, Ataberk Olgun, F. Nisa Bostanci, Oguzhan Canpolat, Geraldo F. Oliveira, Mohammad Sadrosadati, Abdullah Giray Yaglikci, Onur Mutlu

In this work, we experimentally demonstrate that it is possible to generate
true random numbers at high throughput and low latency in commercial
off-the-shelf (COTS) DRAM chips by leveraging simultaneous multiple-row
activation (SiMRA) via an extensive characterization of 96 DDR4 DRAM chips. We
rigorously analyze SiMRA's true random generation potential in terms of
entropy, latency, and throughput for varying numbers of simultaneously
activated DRAM rows (i.e., 2, 4, 8, 16, and 32), data patterns, temperature
levels, and spatial variations. Among our 11 key experimental observations, we
highlight four key results. First, we evaluate the quality of our TRNG designs
using the commonly-used NIST statistical test suite for randomness and find
that all SiMRA-based TRNG designs successfully pass each test. Second, 2-, 8-,
16-, and 32-row activation-based TRNG designs outperform the state-of-theart
DRAM-based TRNG in throughput by up to 1.15x, 1.99x, 1.82x, and 1.39x,
respectively. Third, SiMRA's entropy tends to increase with the number of
simultaneously activated DRAM rows. Fourth, operational parameters and
conditions (e.g., data pattern and temperature) significantly affect entropy.
For example, for most of the tested modules, the average entropy of 32-row
activation is 2.51x higher than that of 2-row activation. For example,
increasing the temperature from 50{\deg}C to 90{\deg}C decreases SiMRA's
entropy by 1.53x for 32-row activation. To aid future research and development,
we open-source our infrastructure at https://github.com/CMU-SAFARI/SiMRA-TRNG.

### Computational Complexity

### 1. [Compression of Voxelized Vector Field Data by Boxes is Hard](http://arxiv.org/pdf/2510.20801v1)

Authors: Simon Zhang

Voxelized vector field data consists of a vector field over a high
dimensional lattice. The lattice consists of integer coordinates called voxels.
The voxelized vector field assigns a vector at each voxel. This data type
encompasses images, tensors, and voxel data.
  Assume there is a nice energy function on the vector field. We consider the
problem of lossy compression of voxelized vector field data in Shannon's
rate-distortion framework. This means the data is compressed then decompressed
up to a bound on the distortion of the energy at each voxel. We formulate this
in terms of compressing a single voxelized vector field by a collection of box
summary pairs. We call this problem the $(k,D)$-RectLossyVFCompression}
problem.
  We show three main results about this problem. The first is that
decompression for this problem is polynomial time tractable. This means that
the only obstruction to a tractable solution of the
$(k,D)$-RectLossyVFCompression problem lies in the compression stage. This is
shown by the two hardness results about the compression stage. We show that the
compression stage is NP-Hard to compute exactly and that it is even APX-Hard to
approximate for $k,D\geq 2$.
  Assuming $P\neq NP$, this shows that when $k,D \geq 2$ there can be no exact
polynomial time algorithm nor can there even be a PTAS approximation algorithm
for the $(k,D)$-RectLossyVFCompression problem.

### 2. [A Classification of Long-Refinement Graphs for Colour Refinement](http://arxiv.org/pdf/2510.20802v1)

Authors: Sandra Kiefer, T. Devini de Mel

The Colour Refinement algorithm is a classical procedure to detect symmetries
in graphs, whose most prominent application is in graph-isomorphism tests. The
algorithm and its generalisation, the Weisfeiler-Leman algorithm, evaluate
local information to compute a colouring for the vertices in an iterative
fashion. Different final colours of two vertices certify that no isomorphism
can map one onto the other. The number of iterations that the algorithm takes
to terminate is its central complexity parameter. For a long time, it was open
whether graphs that take the maximum theoretically possible number of Colour
Refinement iterations actually exist. Starting from an exhaustive search on
graphs of low degrees, Kiefer and McKay proved the existence of infinite
families of such long-refinement graphs with degrees 2 and 3, thereby showing
that the trivial upper bound on the iteration number of Colour Refinement is
tight. In this work, we provide a complete characterisation of the
long-refinement graphs with low (or, equivalently, high) degrees. We show that,
with one exception, the aforementioned families are the only long-refinement
graphs with maximum degree at most 3, and we fully classify the long-refinement
graphs with maximum degree 4. To this end, via a reverse-engineering approach,
we show that all low-degree long-refinement graphs can be represented as
compact strings, and we derive multiple structural insights from this
surprising fact. Since long-refinement graphs are closed under taking edge
complements, this also yields a classification of long-refinement graphs with
high degrees. Kiefer and McKay initiated a search for long-refinement graphs
that are only distinguished in the last iteration of Colour Refinement before
termination. We conclude it in this submission by showing that such graphs
cannot exist.

### Computational Engineering

### 1. [SparseEB-gMCR: A Generative Solver for Extreme Sparse Components with Application to Contamination Removal in GC-MS](http://arxiv.org/pdf/2510.20364v1)

Authors: Yu-Tang Chang, Shih-Fang Chen

Analytical chemistry instruments provide physically meaningful signals for
elucidating analyte composition and play important roles in material,
biological, and food analysis. These instruments are valued for strong
alignment with physical principles, enabling compound identification through
pattern matching with chemical libraries. More reliable instruments generate
sufficiently sparse signals for direct interpretation. Generative multivariate
curve resolution (gMCR) and its energy-based solver (EB-gMCR) offer powerful
tools for decomposing mixed signals suitable for chemical data analysis.
However, extreme signal sparsity from instruments such as GC-MS or 1H-NMR can
impair EB-gMCR decomposability. To address this, a fixed EB-select module
inheriting EB-gMCR's design was introduced for handling extreme sparse
components. Combined with minor adjustments to energy optimization, this led to
SparseEB-gMCR. In synthetic datasets, SparseEB-gMCR exhibited comparable
decomposability and graceful scalability to dense-component EB-gMCR. The sparse
variant was applied to real GC-MS chromatograms for unsupervised contamination
removal. Analysis showed siloxane-related pollution signals were effectively
eliminated, improving compound identification reliability. Results demonstrate
that SparseEB-gMCR preserves the decomposability and self-determining component
capability of EB-gMCR while extending adaptability to sparse and irregular
chemical data. With this sparse extension, the EB-gMCR family becomes
applicable to wider ranges of real-world chemical datasets, providing a general
mathematical framework for signal unmixing and contamination elimination in
analytical chemistry.

### 2. [AI PB: A Grounded Generative Agent for Personalized Investment Insights](http://arxiv.org/pdf/2510.20099v1)

Authors: Daewoo Park, Suho Park, Inseok Hong, Hanwool Lee, Junkyu Park, Sangjun Lee, Jeongman An, Hyunbin Loh

We present AI PB, a production-scale generative agent deployed in real retail
finance. Unlike reactive chatbots that answer queries passively, AI PB
proactively generates grounded, compliant, and user-specific investment
insights. It integrates (i) a component-based orchestration layer that
deterministically routes between internal and external LLMs based on data
sensitivity, (ii) a hybrid retrieval pipeline using OpenSearch and the
finance-domain embedding model, and (iii) a multi-stage recommendation
mechanism combining rule heuristics, sequential behavioral modeling, and
contextual bandits. Operating fully on-premises under Korean financial
regulations, the system employs Docker Swarm and vLLM across 24 X NVIDIA H100
GPUs. Through human QA and system metrics, we demonstrate that grounded
generation with explicit routing and layered safety can deliver trustworthy AI
insights in high-stakes finance.

### 3. [Decentralized Exchange that Mitigate a Bribery Attack](http://arxiv.org/pdf/2510.20645v1)

Authors: Nitin Awathare

Despite the popularity of Hashed Time-Locked Contracts (HTLCs) because of
their use in wide areas of applications such as payment channels, atomic swaps,
etc, their use in exchange is still questionable. This is because of its
incentive incompatibility and susceptibility to bribery attacks.
  State-of-the-art solutions such as MAD-HTLC (Oakland'21) and He-HTLC
(NDSS'23) address this by leveraging miners' profit-driven behaviour to
mitigate such attacks. The former is the mitigation against passive miners;
however, the latter works against both active and passive miners. However, they
consider only two bribing scenarios where either of the parties involved in the
transfer collude with the miner.
  In this paper, we expose vulnerabilities in state-of-the-art solutions by
presenting a miner-collusion bribery attack with implementation and
game-theoretic analysis. Additionally, we propose a stronger attack on MAD-HTLC
than He-HTLC, allowing the attacker to earn profits equivalent to attacking
naive HTLC.
  Leveraging our insights, we propose \prot, a game-theoretically secure HTLC
protocol resistant to all bribery scenarios. \prot\ employs a two-phase
approach, preventing unauthorized token confiscation by third parties, such as
miners. In Phase 1, parties commit to the transfer; in Phase 2, the transfer is
executed without manipulation. We demonstrate \prot's efficiency in transaction
cost and latency via implementations on Bitcoin and Ethereum.

### Computational Geometry

### 1. [A Tverberg-type problem of Kalai: Two negative answers to questions of Alon and Smorodinsky, and the power of disjointness](http://arxiv.org/pdf/2510.20770v1)

Authors: Wenchong Chen, Gennian Ge, Yang Shu, Zhouningxin Wang, Zixiang Xu

Let $f_r(d,s_1,\ldots,s_r)$ denote the least integer $n$ such that every
$n$-point set $P\subseteq\mathbb{R}^d$ admits a partition $P=P_1\cup\cdots\cup
P_r$ with the property that for any choice of $s_i$-convex sets $C_i\supseteq
P_i$ $(i\in[r])$ one necessarily has $\bigcap_{i=1}^r C_i\neq\emptyset$, where
an $s_i$-convex set means a union of $s_i$ convex sets. A recent breakthrough
by Alon and Smorodinsky establishes a general upper bound $f_r(d,s_1,\dots,s_r)
= O(dr^2\log r \prod_{i=1}^r s_i\cdot \log(\prod_{i=1}^r s_i).$ Specializing to
$r=2$ resolves the problem of Kalai from the 1970s. They further singled out
two particularly intriguing questions: whether $f_{2}(2,s,s)$ can be improved
from $O(s^2\log s)$ to $O(s)$, and whether $f_r(d,s,\ldots,s)\le Poly(r,d,s)$.
We answer both in the negative by showing the exponential lower bound
$f_{r}(d,s,\ldots,s)> s^{r}$ for any $r\ge 2$, $s\ge 1$ and $d\ge 2r-2$, which
matches the upper bound up to a multiplicative $\log{s}$ factor for
sufficiently large $s$. Our construction combines a scalloped planar
configuration with a direct product of regular $s$-gon on the high-dimensional
torus $(\mathbb{S}^1)^{r-2}$. Perhaps surprisingly, if we additionally require
that within each block the $s_i$ convex sets are pairwise disjoint, the picture
changes markedly. Let $F_r(d,s_1,\ldots,s_r)$ denote this disjoint-union
variant of the extremal function. We show: (1) $F_{2}(2,s,s)=O(s\log s)$ by
performing controlled planar geometric transformations and constructing an
auxiliary graph whose planarity yields the upper bound; (2) when $s$ is large,
$F_r(d,s,\ldots,s)$ can be bounded by
$O_{r,d}(s^{(1-\frac{1}{2^{d}(d+1)})r+1})$ and $O_{d}(r^{3}\log r\cdot
s^{2d+3})$, respectively. This builds on a novel connection between the
geometric obstruction and hypergraph Tur\'{a}n numbers, in particular, a
variant of the Erd\H{o}s box problem.

### Computation and Language

### 1. [BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](http://arxiv.org/pdf/2510.20151v1)

Authors: Haoyuan Li, Zhengyuan Shen, Sullam Jeoung, Yueyan Chen, Jiayu Li, Qi Zhu, Shuai Wang, Vassilis Ioannidis, Huzefa Rangwala

As structured texts become increasingly complex across diverse domains --
from technical reports to generative AI prompts -- the need for text
segmentation into semantically meaningful components becomes critical. Such
texts often contain elements beyond plain language, including tables, code
snippets, and placeholders, which conventional sentence- or paragraph-level
segmentation methods cannot handle effectively. To address this challenge, we
propose BoundRL, a novel and efficient approach that jointly performs
token-level text segmentation and label prediction for long structured texts.
Instead of generating complete contents for each segment, it generates only a
sequence of starting tokens and reconstructs the complete contents by locating
these tokens within the original texts, thereby reducing inference costs by
orders of magnitude and minimizing hallucination. To adapt the model for the
output format, BoundRL~performs reinforcement learning with verifiable rewards
(RLVR) with a specifically designed reward that jointly optimizes document
reconstruction fidelity and semantic alignment. To mitigate entropy collapse,
it further constructs intermediate candidates by systematically perturbing a
fraction of generated sequences of segments to create stepping stones toward
higher-quality solutions. To demonstrate BoundRL's effectiveness on
particularly challenging structured texts, we focus evaluation on complex
prompts used for LLM applications. Experiments show that BoundRL enables small
language models (1.7B parameters) to outperform few-shot prompting of much
larger models. Moreover, RLVR with our designed reward yields significant
improvements over supervised fine-tuning, and incorporating intermediate
candidates further improves both performance and generalization.

### 2. [DeepWideSearch: Benchmarking Depth and Width in Agentic Information Seeking](http://arxiv.org/pdf/2510.20168v1)

Authors: Tian Lan, Bin Zhu, Qianghuai Jia, Junyang Ren, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo, Kaifu Zhang

Current search agents fundamentally lack the ability to simultaneously
perform \textit{deep} reasoning over multi-hop retrieval and
\textit{wide}-scale information collection-a critical deficiency for real-world
applications like comprehensive market analysis and business development. To
bridge this gap, we introduce DeepWideSearch, the first benchmark explicitly
designed to evaluate agents to integrate depth and width in information
seeking. In DeepWideSearch, agents must process a large volume of data, each
requiring deep reasoning over multi-hop retrieval paths. Specifically, we
propose two methods to converse established datasets, resulting in a curated
collection of 220 questions spanning 15 diverse domains. Extensive experiments
demonstrate that even state-of-the-art agents achieve only 2.39% average
success rate on DeepWideSearch, highlighting the substantial challenge of
integrating depth and width search in information-seeking tasks. Furthermore,
our error analysis reveals four failure modes: lack of reflection, overreliance
on internal knowledge, insufficient retrieval, and context overflow-exposing
key limitations in current agent architectures. We publicly release
DeepWideSearch to catalyze future research on more capable and robust
information-seeking agents.

### 3. [Decoding-Free Sampling Strategies for LLM Marginalization](http://arxiv.org/pdf/2510.20208v1)

Authors: David Pohl, Marco Cognetta, Junyoung Lee, Naoaki Okazaki

Modern language models operate on subword-tokenized text in order to make a
trade-off between model size, inference speed, and vocabulary coverage. A side
effect of this is that, during inference, models are evaluated by measuring the
probability of only the specific tokenization produced as the output, despite
there being many possible ways to represent the same text with a subword
vocabulary. Recent studies have argued instead for evaluating LLMs by
marginalization - the probability mass of all tokenizations of a given text.
  Marginalization is difficult due to the number of possible tokenizations of a
text, so often approximate marginalization is done via sampling. However, a
downside of sampling is that an expensive generation step must be performed by
the LLM for each sample, which limits the number of samples that can be
acquired given a runtime budget, and therefore also the accuracy of the
approximation. Since computing the probability of a sequence given the
tokenization is relatively cheap compared to actually generating it, we
investigate sampling strategies that are decoding-free - they require no
generation from the LLM, instead relying entirely on extremely cheap sampling
strategies that are model and tokenizer agnostic.
  We investigate the approximation quality and speed of decoding-free sampling
strategies for a number of open models to find that they provide sufficiently
accurate marginal estimates at a small fraction of the runtime cost and
demonstrate its use on a set of downstream inference tasks.

### 4. [Citation Failure: Definition, Analysis and Efficient Mitigation](http://arxiv.org/pdf/2510.20303v1)

Authors: Jan Buchmann, Iryna Gurevych

Citations from LLM-based RAG systems are supposed to simplify response
verification. However, this does not hold for citation failure, when a model
generates a helpful response, but fails to cite complete evidence. In contrast
to previous work, we propose to disentangle this from response failure, where
the response itself is flawed, and citing complete evidence is impossible. To
address citation failure, this work follows a two-step approach: (1) We study
when citation failure occurs and (2) how it can be mitigated. For step 1, we
extend prior work by investigating how the relation between response and
evidence affects citation quality. We introduce CITECONTROL, a benchmark that
systematically varies this relation to analyze failure modes. Experiments show
that failures increase with relational complexity and suggest that combining
citation methods could improve performance, motivating step 2. To improve LLM
citation efficiently, we propose CITENTION, a framework integrating generative,
attention-based, and retrieval-based methods. Results demonstrate substantial
citation improvements on CITECONTROL and in transfer settings. We make our data
and code publicly available.

### 5. [Exploring Generative Process Reward Modeling for Semi-Structured Data: A Case Study of Table Question Answering](http://arxiv.org/pdf/2510.20304v1)

Authors: Lei Tang, Wei Zhou, Mohsen Mesgar

Process reward models (PRMs) improve complex reasoning in large language
models (LLMs) by grading candidate solutions step-by-step and selecting answers
via aggregated step scores. While effective in domains such as mathematics,
their applicability to tasks involving semi-structured data, like table
question answering (TQA) remains unexplored. TQA poses unique challenges for
PRMs, including abundant irrelevant information, loosely connected reasoning
steps, and domain-specific reasoning. This work presents the first systematic
study of PRMs for TQA. We evaluate state-of-the-art generative PRMs on TQA from
both answer and step perspectives. Results show that PRMs that combine textual
and code verification can aid solution selection but struggle to generalize to
out-of-domain data. Analysis reveals a weak correlation between performance in
step-level verification and answer accuracy, possibly stemming from weak step
dependencies and loose causal links. Our findings highlight limitations of
current PRMs on TQA and offer valuable insights for building more robust,
process-aware verifiers.

### 6. [FreeChunker: A Cross-Granularity Chunking Framework](http://arxiv.org/pdf/2510.20356v1)

Authors: Wenxuan Zhang, Yuan-Hao Jiang, Yonghe Wu

Chunking strategies significantly impact the effectiveness of
Retrieval-Augmented Generation (RAG) systems. Existing methods operate within
fixed-granularity paradigms that rely on static boundary identification,
limiting their adaptability to diverse query requirements. This paper presents
FreeChunker, a Cross-Granularity Encoding Framework that fundamentally
transforms the traditional chunking paradigm: the framework treats sentences as
atomic units and shifts from static chunk segmentation to flexible retrieval
supporting arbitrary sentence combinations. This paradigm shift not only
significantly reduces the computational overhead required for semantic boundary
detection but also enhances adaptability to complex queries. Experimental
evaluation on LongBench V2 demonstrates that FreeChunker achieves superior
retrieval performance compared to traditional chunking methods, while
significantly outperforming existing approaches in computational efficiency.

### 7. [Dialogue Is Not Enough to Make a Communicative BabyLM (But Neither Is Developmentally Inspired Reinforcement Learning)](http://arxiv.org/pdf/2510.20358v1)

Authors: Francesca Padovani, Bastian Bunzeck, Manar Ali, Omar Momen, Arianna Bisazza, Hendrik Buschmeier, Sina Zarrieß

We investigate whether pre-training exclusively on dialogue data results in
formally and functionally apt small language models. Based on this pre-trained
llamalogue model, we employ a variety of fine-tuning strategies to enforce
"more communicative" text generations by our models. Although our models
underperform on most standard BabyLM benchmarks, they excel at dialogue
continuation prediction in a minimal pair setting. While PPO fine-tuning has
mixed to adversarial effects on our models, DPO fine-tuning further improves
their performance on our custom dialogue benchmark.

### 8. [NeoDictaBERT: Pushing the Frontier of BERT models for Hebrew](http://arxiv.org/pdf/2510.20386v1)

Authors: Shaltiel Shmidman, Avi Shmidman, Moshe Koppel

Since their initial release, BERT models have demonstrated exceptional
performance on a variety of tasks, despite their relatively small size
(BERT-base has ~100M parameters). Nevertheless, the architectural choices used
in these models are outdated compared to newer transformer-based models such as
Llama3 and Qwen3. In recent months, several architectures have been proposed to
close this gap. ModernBERT and NeoBERT both show strong improvements on English
benchmarks and significantly extend the supported context window. Following
their successes, we introduce NeoDictaBERT and NeoDictaBERT-bilingual:
BERT-style models trained using the same architecture as NeoBERT, with a
dedicated focus on Hebrew texts. These models outperform existing ones on
almost all Hebrew benchmarks and provide a strong foundation for downstream
tasks. Notably, the NeoDictaBERT-bilingual model shows strong results on
retrieval tasks, outperforming other multilingual models of similar size. In
this paper, we describe the training process and report results across various
benchmarks. We release the models to the community as part of our goal to
advance research and development in Hebrew NLP.

### 9. [Teacher Demonstrations in a BabyLM's Zone of Proximal Development for Contingent Multi-Turn Interaction](http://arxiv.org/pdf/2510.20411v1)

Authors: Suchir Salhan, Hongyi Gu, Donya Rooein, Diana Galvan-Sosa, Gabrielle Gaudeau, Andrew Caines, Zheng Yuan, Paula Buttery

Multi-turn dialogues between a child and a caregiver are characterized by a
property called contingency - that is, prompt, direct, and meaningful exchanges
between interlocutors. We introduce ContingentChat, a teacher-student framework
that benchmarks and improves multi-turn contingency in a BabyLM trained on 100M
words. Using a novel alignment dataset for post-training, BabyLM generates
responses that are more grammatical and cohesive. Experiments with adaptive
teacher decoding strategies show limited additional gains. ContingentChat
demonstrates the benefits of targeted post-training for dialogue quality and
indicates that contingency remains a challenging goal for BabyLMs.

### 10. [LM-mixup: Text Data Augmentation via Language Model based Mixup](http://arxiv.org/pdf/2510.20449v1)

Authors: Zhijie Deng, Zhouan Shen, Ling Li, Yao Zhou, Zhaowei Zhu, Yanji He, Wei Wang, Jiaheng Wei

Instruction tuning is crucial for aligning Large Language Models (LLMs), yet
the quality of instruction-following data varies significantly. While
high-quality data is paramount, it is often scarce; conversely, abundant
low-quality data is frequently discarded, leading to substantial information
loss. Existing data augmentation methods struggle to augment this low-quality
data effectively, and the evaluation of such techniques remains poorly defined.
To address this, we formally define the task of Instruction Distillation:
distilling multiple low-quality and redundant inputs into high-quality and
coherent instruction-output pairs. Specifically, we introduce a comprehensive
data construction pipeline to create MIXTURE, a 144K-sample dataset pairing
low-quality or semantically redundant imperfect instruction clusters with their
high-quality distillations. We then introduce LM-Mixup, by first performing
supervised fine-tuning on MIXTURE and then optimizing it with reinforcement
learning. This process uses three complementary reward signals: quality,
semantic alignment, and format compliance, via Group Relative Policy
Optimization (GRPO). We demonstrate that LM-Mixup effectively augments
imperfect datasets: fine-tuning LLMs on its distilled data, which accounts for
only about 3% of the entire dataset, not only surpasses full-dataset training
but also competes with state-of-the-art high-quality data selection methods
across multiple benchmarks. Our work establishes that low-quality data is a
valuable resource when properly distilled and augmented with LM-Mixup,
significantly enhancing the efficiency and performance of instruction-tuned
LLMs.

### Cryptography and Security

### 1. [Separating Pseudorandom Generators from Logarithmic Pseudorandom States](http://arxiv.org/pdf/2510.20131v1)

Authors: Mohammed Barhoush

Pseudorandom generators (PRGs) are a foundational primitive in classical
cryptography, underpinning a wide range of constructions. In the quantum
setting, pseudorandom quantum states (PRSs) were proposed as a potentially
weaker assumption that might serve as a substitute for PRGs in cryptographic
applications. Two primary size regimes of PRSs have been studied:
logarithmic-size and linear-size. Interestingly, logarithmic PRSs have led to
powerful cryptographic applications, such as digital signatures and quantum
public-key encryption, that have not been realized from their linear
counterparts. However, PRGs have only been black-box separated from linear
PRSs, leaving open the fundamental question of whether PRGs are also separated
from logarithmic PRSs.
  In this work, we resolve this open problem. We establish a quantum black-box
separation between (quantum-evaluable) PRGs and PRSs of either size regime.
Specifically, we construct a unitary quantum oracle with inverse access
relative to which no black-box construction of PRG from (logarithmic or linear)
PRS exists. As a direct corollary, we obtain separations between PRGs and
several primitives implied by logarithmic PRSs, including digital signatures
and quantum public-key encryption.

### 2. [Privacy Protection of Automotive Location Data Based on Format-Preserving Encryption of Geographical Coordinates](http://arxiv.org/pdf/2510.20300v1)

Authors: Haojie Ji, Long Jin, Haowen Li, Chongshi Xin, Te Hu

There are increasing risks of privacy disclosure when sharing the automotive
location data in particular functions such as route navigation, driving
monitoring and vehicle scheduling. These risks could lead to the attacks
including user behavior recognition, sensitive location inference and
trajectory reconstruction. In order to mitigate the data security risk caused
by the automotive location sharing, this paper proposes a high-precision
privacy protection mechanism based on format-preserving encryption (FPE) of
geographical coordinates. The automotive coordinate data key mapping mechanism
is designed to reduce to the accuracy loss of the geographical location data
caused by the repeated encryption and decryption. The experimental results
demonstrate that the average relative distance retention rate (RDR) reached
0.0844, and the number of hotspots in the critical area decreased by 98.9%
after encryption. To evaluate the accuracy loss of the proposed encryption
algorithm on automotive geographical location data, this paper presents the
experimental analysis of decryption accuracy, and the result indicates that the
decrypted coordinate data achieves a restoration accuracy of 100%. This work
presents a high-precision privacy protection method for automotive location
data, thereby providing an efficient data security solution for the sensitive
data sharing in autonomous driving.

### 3. [NeuPerm: Disrupting Malware Hidden in Neural Network Parameters by Leveraging Permutation Symmetry](http://arxiv.org/pdf/2510.20367v1)

Authors: Daniel Gilkarov, Ran Dubin

Pretrained deep learning model sharing holds tremendous value for researchers
and enterprises alike. It allows them to apply deep learning by fine-tuning
models at a fraction of the cost of training a brand-new model. However, model
sharing exposes end-users to cyber threats that leverage the models for
malicious purposes. Attackers can use model sharing by hiding self-executing
malware inside neural network parameters and then distributing them for
unsuspecting users to unknowingly directly execute them, or indirectly as a
dependency in another software. In this work, we propose NeuPerm, a simple yet
effec- tive way of disrupting such malware by leveraging the theoretical
property of neural network permutation symmetry. Our method has little to no
effect on model performance at all, and we empirically show it successfully
disrupts state-of-the-art attacks that were only previously addressed using
quantization, a highly complex process. NeuPerm is shown to work on LLMs, a
feat that no other previous similar works have achieved. The source code is
available at https://github.com/danigil/NeuPerm.git.

### 4. [SAID: Empowering Large Language Models with Self-Activating Internal Defense](http://arxiv.org/pdf/2510.20129v1)

Authors: Yulong Chen, Yadong Liu, Jiawen Zhang, Mu Li, Chao Huang, Jie Wen

Large Language Models (LLMs), despite advances in safety alignment, remain
vulnerable to jailbreak attacks designed to circumvent protective mechanisms.
Prevailing defense strategies rely on external interventions, such as input
filtering or output modification, which often lack generalizability and
compromise model utility while incurring significant computational overhead. In
this work, we introduce a new, training-free defense paradigm, Self-Activating
Internal Defense (SAID), which reframes the defense task from external
correction to internal capability activation. SAID uniquely leverages the LLM's
own reasoning abilities to proactively identify and neutralize malicious intent
through a three-stage pipeline: model-native intent distillation to extract
core semantics, optimal safety prefix probing to activate latent safety
awareness, and a conservative aggregation strategy to ensure robust
decision-making. Extensive experiments on five open-source LLMs against six
advanced jailbreak attacks demonstrate that SAID substantially outperforms
state-of-the-art defenses in reducing harmful outputs. Crucially, it achieves
this while preserving model performance on benign tasks and incurring minimal
computational overhead. Our work establishes that activating the intrinsic
safety mechanisms of LLMs is a more robust and scalable path toward building
safer and more reliable aligned AI systems.

### 5. [Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations](http://arxiv.org/pdf/2510.20223v1)

Authors: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

Multimodal large language models (MLLMs) have achieved remarkable progress,
yet remain critically vulnerable to adversarial attacks that exploit weaknesses
in cross-modal processing. We present a systematic study of multimodal
jailbreaks targeting both vision-language and audio-language models, showing
that even simple perceptual transformations can reliably bypass
state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts
across three high-risk safety categories harmful content, CBRN (Chemical,
Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation
Material) tested against seven frontier models. We explore the effectiveness of
attack techniques on MLLMs, including FigStep-Pro (visual keyword
decomposition), Intelligent Masking (semantic obfuscation), and audio
perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe
vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer
>75\% attack success under perceptually modified inputs, with FigStep-Pro
achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further
uncover provider-specific weaknesses, with even basic modality transfer
yielding 25\% ASR for technical queries. These findings expose a critical gap
between text-centric alignment and multimodal threats, demonstrating that
current safeguards fail to generalize across cross-modal attacks. The
accessibility of these attacks, which require minimal technical expertise,
suggests that robust multimodal AI safety will require a paradigm shift toward
broader semantic-level reasoning to mitigate possible risks.

### 6. [HHEML: Hybrid Homomorphic Encryption for Privacy-Preserving Machine Learning on Edge](http://arxiv.org/pdf/2510.20243v1)

Authors: Yu Hin Chan, Hao Yang, Shiyu Shen, Xingyu Fan, Shengzhe Lyu, Patrick S. Y. Hung, Ray C. C. Cheung

Privacy-preserving machine learning (PPML) is an emerging topic to handle
secure machine learning inference over sensitive data in untrusted
environments. Fully homomorphic encryption (FHE) enables computation directly
on encrypted data on the server side, making it a promising approach for PPML.
However, it introduces significant communication and computation overhead on
the client side, making it impractical for edge devices. Hybrid homomorphic
encryption (HHE) addresses this limitation by combining symmetric encryption
(SE) with FHE to reduce the computational cost on the client side, and
combining with an FHE-friendly SE can also lessen the processing overhead on
the server side, making it a more balanced and efficient alternative. Our work
proposes a hardware-accelerated HHE architecture built around a lightweight
symmetric cipher optimized for FHE compatibility and implemented as a dedicated
hardware accelerator. To the best of our knowledge, this is the first design to
integrate an end-to-end HHE framework with hardware acceleration. Beyond this,
we also present several microarchitectural optimizations to achieve higher
performance and energy efficiency. The proposed work is integrated into a full
PPML pipeline, enabling secure inference with significantly lower latency and
power consumption than software implementations. Our contributions validate the
feasibility of low-power, hardware- accelerated HHE for edge deployment and
provide a hardware- software co-design methodology for building scalable,
secure machine learning systems in resource-constrained environments.
Experiments on a PYNQ-Z2 platform with the MNIST dataset show over a 50x
reduction in client-side encryption latency and nearly a 2x gain in hardware
throughput compared to existing FPGA-based HHE accelerators.

### 7. [GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments?](http://arxiv.org/pdf/2510.20333v1)

Authors: Chiyu Chen, Xinhao Song, Yunkai Chai, Yang Yao, Haodong Zhao, Lijun Li, Jie Li, Yan Teng, Gongshen Liu, Yingchun Wang

Vision-Language Models (VLMs) are increasingly deployed as autonomous agents
to navigate mobile graphical user interfaces (GUIs). Operating in dynamic
on-device ecosystems, which include notifications, pop-ups, and inter-app
interactions, exposes them to a unique and underexplored threat vector:
environmental injection. Unlike prompt-based attacks that manipulate textual
instructions, environmental injection corrupts an agent's visual perception by
inserting adversarial UI elements (for example, deceptive overlays or spoofed
notifications) directly into the GUI. This bypasses textual safeguards and can
derail execution, causing privacy leakage, financial loss, or irreversible
device compromise. To systematically evaluate this threat, we introduce
GhostEI-Bench, the first benchmark for assessing mobile agents under
environmental injection attacks within dynamic, executable environments. Moving
beyond static image-based assessments, GhostEI-Bench injects adversarial events
into realistic application workflows inside fully operational Android emulators
and evaluates performance across critical risk scenarios. We further propose a
judge-LLM protocol that conducts fine-grained failure analysis by reviewing the
agent's action trajectory alongside the corresponding screenshot sequence,
pinpointing failure in perception, recognition, or reasoning. Comprehensive
experiments on state-of-the-art agents reveal pronounced vulnerability to
deceptive environmental cues: current models systematically fail to perceive
and reason about manipulated UIs. GhostEI-Bench provides a framework for
quantifying and mitigating this emerging threat, paving the way toward more
robust and secure embodied agents.

### 8. [Classport: Designing Runtime Dependency Introspection for Java](http://arxiv.org/pdf/2510.20340v1)

Authors: Serena Cofano, Daniel Williams, Aman Sharma, Martin Monperrus

Runtime introspection of dependencies, i.e., the ability to observe which
dependencies are currently used during program execution, is fundamental for
Software Supply Chain security. Yet, Java has no support for it. We solve this
problem with Classport, a system that embeds dependency information into Java
class files, enabling the retrieval of dependency information at runtime. We
evaluate Classport on six real-world projects, demonstrating the feasibility in
identifying dependencies at runtime. Runtime dependency introspection with
Classport opens important avenues for runtime integrity checking.

### 9. [MAC Aggregation over Lossy Channels in DTLS 1.3](http://arxiv.org/pdf/2510.20419v1)

Authors: Eric Wagner, David Heye, Jan Bauer, Klaus Wehrle, Martin Serror

Aggregating Message Authentication Codes (MACs) promises to save valuable
bandwidth in resource-constrained environments. The idea is simple: Instead of
appending an authentication tag to each message in a communication stream, the
integrity protection of multiple messages is aggregated into a single tag.
Recent studies postulate, e.g., based on simulations, that these benefits also
spread to wireless, and thus lossy, scenarios despite each lost packet
typically resulting in the loss of integrity protection information for
multiple messages. In this paper, we investigate these claims in a real
deployment. Therefore, we first design a MAC aggregation extension for the
Datagram Transport Layer Security (DTLS) 1.3 protocol. Afterward, we
extensively evaluate the performance of MAC aggregation on a complete
communication protocol stack on embedded hardware. We find that MAC aggregation
can indeed increase goodput by up to 50% and save up to 17% of energy
expenditure for the transmission of short messages, even in lossy channels.

### 10. [On the cybersecurity of LoRaWAN-based system: a Smart-Lighting case study](http://arxiv.org/pdf/2510.20494v1)

Authors: Florian Hofer, Barbara Russo

Cyber-physical systems and the Internet of Things (IoT) are key technologies
in the Industry 4.0 vision. They incorporate sensors and actuators to interact
with the physical environment. However, when creating and interconnecting
components to form a heterogeneous smart systems architecture, these face
challenges in cybersecurity. This paper presents an experimental investigation
of architectural configurations for a LoRaWAN-based Smart-Lighting project,
aimed at verifying and improving the system's robustness against attacks. We
assess the system's robustness in a series of iterative experiments conducted
both in-vitro and on-site. The results show that most attacks on a LoRaWAN
network are unsuccessful, also highlighting unresolved issues with the
installed products. The most successful attacks are high-power jamming attacks
within a few meters of the target, which, in the case of gateways, can be
mitigated through gateway redundancy.

### Computer Vision and Pattern Recognition

### 1. [Endoshare: A Source Available Solution to De-Identify and Manage Surgical Videos](http://arxiv.org/pdf/2510.20087v1)

Authors: Lorenzo Arboit, Dennis N. Schneider, Britty Baby, Vinkle Srivastav, Pietro Mascagni, Nicolas Padoy

Video-based assessment and surgical data science can advance surgical
training, research, and quality improvement. However, widespread use remains
limited by heterogeneous recording formats and privacy concerns associated with
video sharing. We present Endoshare, a source-available, cross-platform
application for merging, standardizing, and de-identifying endoscopic videos in
minimally invasive surgery. Development followed the software development life
cycle with iterative, user-centered feedback. During the analysis phase, an
internal survey of clinicians and computer scientists based on ten usability
heuristics identified key requirements that guided a privacy-by-design
architecture. In the testing phase, an external clinician survey combined the
same heuristics with Technology Acceptance Model constructs to assess usability
and adoption, complemented by benchmarking across different hardware
configurations. Four clinicians and four computer scientists initially tested
the prototype, reporting high usability (4.68 +/- 0.40/5 and 4.03 +/- 0.51/5),
with the lowest score (4.00 +/- 0.93/5) relating to label clarity. After
refinement, the testing phase surveyed ten surgeons who reported high perceived
usefulness (5.07 +/- 1.75/7), ease of use (5.15 +/- 1.71/7), heuristic
usability (4.38 +/- 0.48/5), and strong recommendation (9.20 +/- 0.79/10).
Processing time varied with processing mode, video duration (both p <= 0.001),
and machine computational power (p = 0.041). Endoshare provides a transparent,
user-friendly pipeline for standardized, privacy-preserving surgical video
management. Compliance certification and broader interoperability validation
are needed to establish it as a deployable alternative to proprietary systems.
The software is available at https://camma-public.github.io/Endoshare/

### 2. [Attentive Convolution: Unifying the Expressivity of Self-Attention with Convolutional Efficiency](http://arxiv.org/pdf/2510.20092v1)

Authors: Hao Yu, Haoyu Chen, Yan Jiang, Wei Peng, Zhaodong Sun, Samuel Kaski, Guoying Zhao

Self-attention (SA) has become the cornerstone of modern vision backbones for
its powerful expressivity over traditional Convolutions (Conv). However, its
quadratic complexity remains a critical bottleneck for practical applications.
Given that Conv offers linear complexity and strong visual priors, continuing
efforts have been made to promote the renaissance of Conv. However, a
persistent performance chasm remains, highlighting that these modernizations
have not yet captured the intrinsic expressivity that defines SA. In this
paper, we re-examine the design of the CNNs, directed by a key question: what
principles give SA its edge over Conv? As a result, we reveal two fundamental
insights that challenge the long-standing design intuitions in prior research
(e.g., Receptive field). The two findings are: (1) \textit{Adaptive routing}:
SA dynamically regulates positional information flow according to semantic
content, whereas Conv employs static kernels uniformly across all positions.
(2) \textit{Lateral inhibition}: SA induces score competition among token
weighting, effectively suppressing redundancy and sharpening representations,
whereas Conv filters lack such inhibitory dynamics and exhibit considerable
redundancy. Based on this, we propose \textit{Attentive Convolution} (ATConv),
a principled reformulation of the convolutional operator that intrinsically
injects these principles. Interestingly, with only $3\times3$ kernels, ATConv
consistently outperforms various SA mechanisms in fundamental vision tasks.
Building on ATConv, we introduce AttNet, a CNN family that can attain
\textbf{84.4\%} ImageNet-1K Top-1 accuracy with only 27M parameters. In
diffusion-based image generation, replacing all SA with the proposed $3\times
3$ ATConv in SiT-XL/2 reduces ImageNet FID by 0.15 in 400k steps with faster
sampling. Code is available at: github.com/price112/Attentive-Convolution.

### 3. [Physics-Guided Fusion for Robust 3D Tracking of Fast Moving Small Objects](http://arxiv.org/pdf/2510.20126v1)

Authors: Prithvi Raj Singh, Raju Gottumukkala, Anthony S. Maida, Alan B. Barhorst, Vijaya Gopu

While computer vision has advanced considerably for general object detection
and tracking, the specific problem of fast-moving tiny objects remains
underexplored. This paper addresses the significant challenge of detecting and
tracking rapidly moving small objects using an RGB-D camera. Our novel system
combines deep learning-based detection with physics-based tracking to overcome
the limitations of existing approaches. Our contributions include: (1) a
comprehensive system design for object detection and tracking of fast-moving
small objects in 3D space, (2) an innovative physics-based tracking algorithm
that integrates kinematics motion equations to handle outliers and missed
detections, and (3) an outlier detection and correction module that
significantly improves tracking performance in challenging scenarios such as
occlusions and rapid direction changes. We evaluated our proposed system on a
custom racquetball dataset. Our evaluation shows our system surpassing kalman
filter based trackers with up to 70\% less Average Displacement Error. Our
system has significant applications for improving robot perception on
autonomous platforms and demonstrates the effectiveness of combining
physics-based models with deep learning approaches for real-time 3D detection
and tracking of challenging small objects.

### 4. [Inverse Image-Based Rendering for Light Field Generation from Single Images](http://arxiv.org/pdf/2510.20132v1)

Authors: Hyunjun Jung, Hae-Gon Jeon

A concept of light-fields computed from multiple view images on regular grids
has proven its benefit for scene representations, and supported realistic
renderings of novel views and photographic effects such as refocusing and
shallow depth of field. In spite of its effectiveness of light flow
computations, obtaining light fields requires either computational costs or
specialized devices like a bulky camera setup and a specialized microlens
array. In an effort to broaden its benefit and applicability, in this paper, we
propose a novel view synthesis method for light field generation from only
single images, named inverse image-based rendering. Unlike previous attempts to
implicitly rebuild 3D geometry or to explicitly represent objective scenes, our
method reconstructs light flows in a space from image pixels, which behaves in
the opposite way to image-based rendering. To accomplish this, we design a
neural rendering pipeline to render a target ray in an arbitrary viewpoint. Our
neural renderer first stores the light flow of source rays from the input
image, then computes the relationships among them through cross-attention, and
finally predicts the color of the target ray based on these relationships.
After the rendering pipeline generates the first novel view from a single input
image, the generated out-of-view contents are updated to the set of source
rays. This procedure is iteratively performed while ensuring the consistent
generation of occluded contents. We demonstrate that our inverse image-based
rendering works well with various challenging datasets without any retraining
or finetuning after once trained on synthetic dataset, and outperforms relevant
state-of-the-art novel view synthesis methods.

### 5. [Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](http://arxiv.org/pdf/2510.20134v1)

Authors: Jiachen Liang, Ruibing Hou, Minyang Hu, Hong Chang, Shiguang Shan, Xilin Chen

Out-of-distribution (OOD) detection is critical for ensuring the reliability
of deep learning models in open-world applications. While post-hoc methods are
favored for their efficiency and ease of deployment, existing approaches often
underexploit the rich information embedded in the model's logits space. In this
paper, we propose LogitGap, a novel post-hoc OOD detection method that
explicitly exploits the relationship between the maximum logit and the
remaining logits to enhance the separability between in-distribution (ID) and
OOD samples. To further improve its effectiveness, we refine LogitGap by
focusing on a more compact and informative subset of the logit space.
Specifically, we introduce a training-free strategy that automatically
identifies the most informative logits for scoring. We provide both theoretical
analysis and empirical evidence to validate the effectiveness of our approach.
Extensive experiments on both vision-language and vision-only models
demonstrate that LogitGap consistently achieves state-of-the-art performance
across diverse OOD detection scenarios and benchmarks. Code is available at
https://github.com/GIT-LJc/LogitGap.

### 6. [PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](http://arxiv.org/pdf/2510.20155v1)

Authors: Penghao Wang, Yiyang He, Xin Lv, Yukai Zhou, Lan Xu, Jingyi Yu, Jiayuan Gu

Understanding objects at the level of their constituent parts is fundamental
to advancing computer vision, graphics, and robotics. While datasets like
PartNet have driven progress in 3D part understanding, their reliance on
untextured geometries and expert-dependent annotation limits scalability and
usability. We introduce PartNeXt, a next-generation dataset addressing these
gaps with over 23,000 high-quality, textured 3D models annotated with
fine-grained, hierarchical part labels across 50 categories. We benchmark
PartNeXt on two tasks: (1) class-agnostic part segmentation, where
state-of-the-art methods (e.g., PartField, SAMPart3D) struggle with
fine-grained and leaf-level parts, and (2) 3D part-centric question answering,
a new benchmark for 3D-LLMs that reveals significant gaps in open-vocabulary
part grounding. Additionally, training Point-SAM on PartNeXt yields substantial
gains over PartNet, underscoring the dataset's superior quality and diversity.
By combining scalable annotation, texture-aware labels, and multi-task
evaluation, PartNeXt opens new avenues for research in structured 3D
understanding.

### 7. [Monocular Visual 8D Pose Estimation for Articulated Bicycles and Cyclists](http://arxiv.org/pdf/2510.20158v1)

Authors: Eduardo R. Corral-Soto, Yang Liu, Yuan Ren, Bai Dongfeng, Liu Bingbing

In Autonomous Driving, cyclists belong to the safety-critical class of
Vulnerable Road Users (VRU), and accurate estimation of their pose is critical
for cyclist crossing intention classification, behavior prediction, and
collision avoidance. Unlike rigid objects, articulated bicycles are composed of
movable rigid parts linked by joints and constrained by a kinematic structure.
6D pose methods can estimate the 3D rotation and translation of rigid bicycles,
but 6D becomes insufficient when the steering/pedals angles of the bicycle
vary. That is because: 1) varying the articulated pose of the bicycle causes
its 3D bounding box to vary as well, and 2) the 3D box orientation is not
necessarily aligned to the orientation of the steering which determines the
actual intended travel direction. In this work, we introduce a method for
category-level 8D pose estimation for articulated bicycles and cyclists from a
single RGB image. Besides being able to estimate the 3D translation and
rotation of a bicycle from a single image, our method also estimates the
rotations of its steering handles and pedals with respect to the bicycle body
frame. These two new parameters enable the estimation of a more fine-grained
bicycle pose state and travel direction. Our proposed model jointly estimates
the 8D pose and the 3D Keypoints of articulated bicycles, and trains with a mix
of synthetic and real image data to generalize on real images. We include an
evaluation section where we evaluate the accuracy of our estimated 8D pose
parameters, and our method shows promising results by achieving competitive
scores when compared against state-of-the-art category-level 6D pose estimators
that use rigid canonical object templates for matching.

### 8. [TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](http://arxiv.org/pdf/2510.20162v1)

Authors: Xudong Yan, Songhe Feng

Compositional Zero-Shot Learning (CZSL) aims to recognize novel
attribute-object compositions based on the knowledge learned from seen ones.
Existing methods suffer from performance degradation caused by the distribution
shift of label space at test time, which stems from the inclusion of unseen
compositions recombined from attributes and objects. To overcome the challenge,
we propose a novel approach that accumulates comprehensive knowledge in both
textual and visual modalities from unsupervised data to update multimodal
prototypes at test time. Building on this, we further design an adaptive update
weight to control the degree of prototype adjustment, enabling the model to
flexibly adapt to distribution shift during testing. Moreover, a dynamic
priority queue is introduced that stores high-confidence images to acquire
visual knowledge from historical images for inference. Considering the semantic
consistency of multimodal knowledge, we align textual and visual prototypes by
multimodal collaborative representation learning. Extensive experiments
indicate that our approach achieves state-of-the-art performance on four
benchmark datasets under both closed-world and open-world settings. Code will
be available at https://github.com/xud-yan/TOMCAT .

### 9. [Evaluating Video Models as Simulators of Multi-Person Pedestrian Trajectories](http://arxiv.org/pdf/2510.20182v1)

Authors: Aaron Appelle, Jerome P. Lynch

Large-scale video generation models have demonstrated high visual realism in
diverse contexts, spurring interest in their potential as general-purpose world
simulators. Existing benchmarks focus on individual subjects rather than scenes
with multiple interacting people. However, the plausibility of multi-agent
dynamics in generated videos remains unverified. We propose a rigorous
evaluation protocol to benchmark text-to-video (T2V) and image-to-video (I2V)
models as implicit simulators of pedestrian dynamics. For I2V, we leverage
start frames from established datasets to enable comparison with a ground truth
video dataset. For T2V, we develop a prompt suite to explore diverse pedestrian
densities and interactions. A key component is a method to reconstruct 2D
bird's-eye view trajectories from pixel-space without known camera parameters.
Our analysis reveals that leading models have learned surprisingly effective
priors for plausible multi-agent behavior. However, failure modes like merging
and disappearing people highlight areas for future improvement.

### 10. [SPAN: Continuous Modeling of Suspicion Progression for Temporal Intention Localization](http://arxiv.org/pdf/2510.20189v1)

Authors: Xinyi Hu, Yuran Wang, Yue Li, Wenxuan Liu, Zheng Wang

Temporal Intention Localization (TIL) is crucial for video surveillance,
focusing on identifying varying levels of suspicious intentions to improve
security monitoring. However, existing discrete classification methods fail to
capture the continuous nature of suspicious intentions, limiting early
intervention and explainability. In this paper, we propose the Suspicion
Progression Analysis Network (SPAN), which shifts from discrete classification
to continuous regression, enabling the capture of fluctuating and evolving
suspicious intentions. We reveal that suspicion exhibits long-term dependencies
and cumulative effects, similar to Temporal Point Process (TPP) theory. Based
on these insights, we define a suspicion score formula that models continuous
changes while accounting for temporal characteristics. We also introduce
Suspicion Coefficient Modulation, which adjusts suspicion coefficients using
multimodal information to reflect the varying impacts of suspicious actions.
Additionally, the Concept-Anchored Mapping method is proposed to link
suspicious actions to predefined intention concepts, offering insights into
both the actions and their potential underlying intentions. Extensive
experiments on the HAI dataset show that SPAN significantly outperforms
existing methods, reducing MSE by 19.8% and improving average mAP by 1.78%.
Notably, SPAN achieves a 2.74% mAP gain in low-frequency cases, demonstrating
its superior ability to capture subtle behavioral changes. Compared to discrete
classification systems, our continuous suspicion modeling approach enables
earlier detection and proactive intervention, greatly enhancing system
explainability and practical utility in security applications.

### Computers and Society

### 1. [Dependency-Aware Task Offloading in Multi-UAV Assisted Collaborative Mobile Edge Computing](http://arxiv.org/pdf/2510.20149v1)

Authors: Zhenyu Zhao, Xiaoxia Xu, Tiankui Zhang, Junjie Li, Yuanwei Liu

This paper proposes a novel multi-unmanned aerial vehicle (UAV) assisted
collaborative mobile edge computing (MEC) framework, where the computing tasks
of terminal devices (TDs) can be decomposed into serial or parallel sub-tasks
and offloaded to collaborative UAVs. We first model the dependencies among all
sub-tasks as a directed acyclic graph (DAG) and design a two-timescale frame
structure to decouple the sub-task interdependencies for sub-task scheduling.
Then, a joint sub-task offloading, computational resource allocation, and UAV
trajectories optimization problem is formulated, which aims to minimize the
system cost, i.e., the weighted sum of the task completion delay and the system
energy consumption. To solve this non-convex mixed-integer nonlinear
programming (MINLP) problem, a penalty dual decomposition and successive convex
approximation (PDD-SCA) algorithm is developed. Particularly, the original
MINLP problem is equivalently transferred into a continuous form relying on PDD
theory. By decoupling the resulting problem into three nested subproblems, the
SCA method is further combined to recast the non-convex components and obtain
desirable solutions. Numerical results demonstrate that: 1) Compared to the
benchmark algorithms, the proposed scheme can significantly reduce the system
cost, and thus realize an improved trade-off between task latency and energy
consumption; 2) The proposed algorithm can achieve an efficient workload
balancing for distributed computation across multiple UAVs.

### 2. [The Order of Recommendation Matters: Structured Exploration for Improving the Fairness of Content Creators](http://arxiv.org/pdf/2510.20698v1)

Authors: Salima Jaoua, Nicolò Pagan, Anikó Hannák, Stefania Ionescu

Social media platforms provide millions of professional content creators with
sustainable incomes. Their income is largely influenced by their number of
views and followers, which in turn depends on the platform's recommender system
(RS). So, as with regular jobs, it is important to ensure that RSs distribute
revenue in a fair way. For example, prior work analyzed whether the creators of
the highest-quality content would receive the most followers and income.
Results showed this is unlikely to be the case, but did not suggest targeted
solutions. In this work, we first use theoretical analysis and simulations on
synthetic datasets to understand the system better and find interventions that
improve fairness for creators. We find that the use of ordered pairwise
comparison overcomes the cold start problem for a new set of items and greatly
increases the chance of achieving fair outcomes for all content creators.
Importantly, it also maintains user satisfaction. We also test the intervention
on the MovieLens dataset and investigate its effectiveness on platforms with
interaction histories that are currently unfair for content creators. These
experiments reveal that the intervention improves fairness when deployed at
early stages of the platform, but the effect decreases as the strength of
pre-existing bias increases. Altogether, we find that the ordered pairwise
comparison approach might offer a plausible alternative for both new and
existing platforms to implement.

### 3. [Hierarchical Dual-Head Model for Suicide Risk Assessment via MentalRoBERTa](http://arxiv.org/pdf/2510.20085v1)

Authors: Chang Yang, Ziyi Wang, Wangfeng Tan, Zhiting Tan, Changrui Ji, Zhiming Zhou

Social media platforms have become important sources for identifying suicide
risk, but automated detection systems face multiple challenges including severe
class imbalance, temporal complexity in posting patterns, and the dual nature
of risk levels as both ordinal and categorical. This paper proposes a
hierarchical dual-head neural network based on MentalRoBERTa for suicide risk
classification into four levels: indicator, ideation, behavior, and attempt.
The model employs two complementary prediction heads operating on a shared
sequence representation: a CORAL (Consistent Rank Logits) head that preserves
ordinal relationships between risk levels, and a standard classification head
that enables flexible categorical distinctions. A 3-layer Transformer encoder
with 8-head multi-head attention models temporal dependencies across post
sequences, while explicit time interval embeddings capture posting behavior
dynamics. The model is trained with a combined loss function (0.5 CORAL + 0.3
Cross-Entropy + 0.2 Focal Loss) that simultaneously addresses ordinal structure
preservation, overconfidence reduction, and class imbalance. To improve
computational efficiency, we freeze the first 6 layers (50%) of MentalRoBERTa
and employ mixed-precision training. The model is evaluated using 5-fold
stratified cross-validation with macro F1 score as the primary metric.

### 4. [What do AI-Generated Images Want?](http://arxiv.org/pdf/2510.20350v1)

Authors: Amanda Wasielewski

W.J.T. Mitchell's influential essay 'What do pictures want?' shifts the
theoretical focus away from the interpretative act of understanding pictures
and from the motivations of the humans who create them to the possibility that
the picture itself is an entity with agency and wants. In this article, I
reframe Mitchell's question in light of contemporary AI image generation tools
to ask: what do AI-generated images want? Drawing from art historical discourse
on the nature of abstraction, I argue that AI-generated images want specificity
and concreteness because they are fundamentally abstract. Multimodal
text-to-image models, which are the primary subject of this article, are based
on the premise that text and image are interchangeable or exchangeable tokens
and that there is a commensurability between them, at least as represented
mathematically in data. The user pipeline that sees textual input become visual
output, however, obscures this representational regress and makes it seem like
one form transforms into the other -- as if by magic.

### 5. [Towards AI Agents for Course Instruction in Higher Education: Early Experiences from the Field](http://arxiv.org/pdf/2510.20255v1)

Authors: Yogesh Simmhan, Varad Kulkarni

This article presents early findings from designing, deploying and evaluating
an AI-based educational agent deployed as the primary instructor in a
graduate-level Cloud Computing course at IISc. We detail the design of a Large
Language Model (LLM)-driven Instructor Agent, and introduce a pedagogical
framework that integrates the Instructor Agent into the course workflow for
actively interacting with the students for content delivery, supplemented by
the human instructor to offer the course structure and undertake
question--answer sessions. We also propose an analytical framework that
evaluates the Agent--Student interaction transcripts using interpretable
engagement metrics of topic coverage, topic depth and turn-level elaboration.
We report early experiences on how students interact with the Agent to explore
concepts, clarify doubts and sustain inquiry-driven dialogue during live
classroom sessions. We also report preliminary analysis on our evaluation
metrics applied across two successive instructional modules that reveals
patterns of engagement evolution, transitioning from broad conceptual
exploration to deeper, focused inquiry. These demonstrate how structured
integration of conversational AI agents can foster reflective learning, offer a
reproducible methodology for studying engagement in authentic classroom
settings, and support scalable, high-quality higher education.

### 6. [Strategic Costs of Perceived Bias in Fair Selection](http://arxiv.org/pdf/2510.20606v1)

Authors: L. Elisa Celis, Lingxiao Huang, Milind Sohoni, Nisheeth K. Vishnoi

Meritocratic systems, from admissions to hiring, aim to impartially reward
skill and effort. Yet persistent disparities across race, gender, and class
challenge this ideal. Some attribute these gaps to structural inequality;
others to individual choice. We develop a game-theoretic model in which
candidates from different socioeconomic groups differ in their perceived
post-selection value--shaped by social context and, increasingly, by AI-powered
tools offering personalized career or salary guidance. Each candidate
strategically chooses effort, balancing its cost against expected reward;
effort translates into observable merit, and selection is based solely on
merit. We characterize the unique Nash equilibrium in the large-agent limit and
derive explicit formulas showing how valuation disparities and institutional
selectivity jointly determine effort, representation, social welfare, and
utility. We further propose a cost-sensitive optimization framework that
quantifies how modifying selectivity or perceived value can reduce disparities
without compromising institutional goals. Our analysis reveals a
perception-driven bias: when perceptions of post-selection value differ across
groups, these differences translate into rational differences in effort,
propagating disparities backward through otherwise "fair" selection processes.
While the model is static, it captures one stage of a broader feedback cycle
linking perceptions, incentives, and outcome--bridging rational-choice and
structural explanations of inequality by showing how techno-social environments
shape individual incentives in meritocratic systems.

### 7. [On the Detectability of LLM-Generated Text: What Exactly Is LLM-Generated Text?](http://arxiv.org/pdf/2510.20810v1)

Authors: Mingmeng Geng, Thierry Poibeau

With the widespread use of large language models (LLMs), many researchers
have turned their attention to detecting text generated by them. However, there
is no consistent or precise definition of their target, namely "LLM-generated
text". Differences in usage scenarios and the diversity of LLMs further
increase the difficulty of detection. What is commonly regarded as the
detecting target usually represents only a subset of the text that LLMs can
potentially produce. Human edits to LLM outputs, together with the subtle
influences that LLMs exert on their users, are blurring the line between
LLM-generated and human-written text. Existing benchmarks and evaluation
approaches do not adequately address the various conditions in real-world
detector applications. Hence, the numerical results of detectors are often
misunderstood, and their significance is diminishing. Therefore, detectors
remain useful under specific conditions, but their results should be
interpreted only as references rather than decisive indicators.

### 8. [Black Box Absorption: LLMs Undermining Innovative Ideas](http://arxiv.org/pdf/2510.20612v1)

Authors: Wenjun Cao

Large Language Models are increasingly adopted as critical tools for
accelerating innovation. This paper identifies and formalizes a systemic risk
inherent in this paradigm: \textbf{Black Box Absorption}. We define this as the
process by which the opaque internal architectures of LLM platforms, often
operated by large-scale service providers, can internalize, generalize, and
repurpose novel concepts contributed by users during interaction. This
mechanism threatens to undermine the foundational principles of innovation
economics by creating severe informational and structural asymmetries between
individual creators and platform operators, thereby jeopardizing the long-term
sustainability of the innovation ecosystem. To analyze this challenge, we
introduce two core concepts: the idea unit, representing the transportable
functional logic of an innovation, and idea safety, a multidimensional standard
for its protection. This paper analyzes the mechanisms of absorption and
proposes a concrete governance and engineering agenda to mitigate these risks,
ensuring that creator contributions remain traceable, controllable, and
equitable.

### Databases

### 1. [UREM: A High-performance Unified and Resilient Enhancement Method for Multi- and High-Dimensional Indexes](http://arxiv.org/pdf/2510.20110v1)

Authors: Ming Sheng, Shuliang Wang, Yong Zhang, Yi Luo, Xianbo Liu, Zeming Li

Numerous multi- or high-dimensional indexes with distinct advantages have
been proposed on various platforms to meet application requirements. To achieve
higher-performance queries, most indexes employ enhancement methods, including
structure-oriented and layout-oriented enhancement methods. Existing
structure-oriented methods tailored to specific indexes work well under static
workloads but lack generality and degrade under dynamic workloads. The
layout-oriented methods exhibit good generality and perform well under dynamic
workloads, but exhibit suboptimal performance under static workloads.
Therefore, it is an open challenge to develop a unified and resilient
enhancement method that can improve query performance for different indexes
adaptively under different scenarios. In this paper, we propose UREM, which is
the first high-performance Unified and Resilient Enhancement Method designed
for both multi- and high-dimensional indexes, capable of adapting to different
scenarios. Specifically, UREM (1) can be uniformly applied with different
indexes on various platforms; (2) enhances the query performance of indexes by
layout optimization under static workloads; (3) enables indexes to stabilize
performance when queries shift through partial layout reorganization. We
evaluate UREM on 20 widely used indexes. Experimental results demonstrate that
UREM improves the query performance of multi- and high-dimensional indexes by
up to 5.73x and 9.18x under static workloads, and by an average of 5.72x and
9.47x under dynamic workloads. Moreover, some traditional indexes enhanced by
UREM even achieve performance comparable to or even surpassing that of recent
advanced indexes.

### 2. [Hybrid Mixed Integer Linear Programming for Large-Scale Join Order Optimisation](http://arxiv.org/pdf/2510.20308v1)

Authors: Manuel Schönberger, Immanuel Trummer, Wolfgang Mauerer

Finding optimal join orders is among the most crucial steps to be performed
by query optimisers. Though extensively studied in data management research,
the problem remains far from solved: While query optimisers rely on exhaustive
search methods to determine ideal solutions for small problems, such methods
reach their limits once queries grow in size. Yet, large queries become
increasingly common in real-world scenarios, and require suitable methods to
generate efficient execution plans. While a variety of heuristics have been
proposed for large-scale query optimisation, they suffer from degrading
solution quality as queries grow in size, or feature highly sub-optimal
worst-case behavior, as we will show.
  We propose a novel method based on the paradigm of mixed integer linear
programming (MILP): By deriving a novel MILP model capable of optimising
arbitrary bushy tree structures, we address the limitations of existing MILP
methods for join ordering, and can rely on highly optimised MILP solvers to
derive efficient tree structures that elude competing methods. To ensure
optimisation efficiency, we embed our MILP method into a hybrid framework,
which applies MILP solvers precisely where they provide the greatest advantage
over competitors, while relying on more efficient methods for less complex
optimisation steps. Thereby, our approach gracefully scales to extremely large
query sizes joining up to 100 relations, and consistently achieves the most
robust plan quality among a large variety of competing join ordering methods.

### 3. [An Empirical Study on Database Usage in Microservices](http://arxiv.org/pdf/2510.20582v1)

Authors: Maxime André, Marco Raglianti, Souhaila Serbout, Anthony Cleve, Michele Lanza

Microservices architectures are an integral part of modern software
development. Their adoption brings significant changes to database management.
Instead of relying on a single database, a microservices architecture is
typically composed of multiple, smaller, heterogeneous, and distributed DBs. In
these data-intensive systems, the variety and combination of database
categories and technologies play a crucial role in storing and managing data.
While data management in microservices is a major challenge, research
literature is scarce.
  We present an empirical study on how databases are used in microservices. On
the dataset we collected (and released as open data for future research),
considering 15 years of microservices, we examine ca. 1,000 GitHub projects
that use databases selected among 180 technologies from 14 categories. We
perform a comprehensive analysis of current practices, providing researchers
and practitioners with empirical evidence to better understand database usage
in microservices. We report 18 findings and 9 recommendations. We show that
microservices predominantly use Relational, Key-Value, Document, and Search
databases. Notably, 52% of microservices combine multiple database categories.
Complexity correlates with database count, with older systems favoring
Relational databases and newer ones increasingly adopting Key-Value and
Document technologies. Niche databases (e.g., EventStoreDB, PostGIS), while not
widespread, are often combined with a mainstream one.

### 4. [Balanced Popularity in Multi-Product Billboard Advertisement](http://arxiv.org/pdf/2510.20600v1)

Authors: Dildar Ali, Suman Banerjee, Yamuna Prasad

The billboard advertisement has emerged as an effective out-of-home
advertisement technique where the objective is to choose a limited number of
slots to play some advertisement content (e.g., animation, video, etc.) with
the hope that the content will be visible to a large number of travelers, and
this will be helpful to earn more revenue. In this paper, we study a variant of
the influential slot selection problem where the advertiser wants to promote
multiple products. Formally, we call this problem the \textsc{Multi-Product
Influence Maximization Problem for the Balanced Popularity} Problem. The input
to our problem is a trajectory and a billboard database, as well as a budget
for each product. The goal here is to choose a subset of slots for each product
such that the aggregated influence of all the products gets maximized subject
to the following two constraints: total selection cost for each product is less
than or equal to the allocated budget for that product, and the difference
between the influence for any two products is less than or equal to a given
threshold. We show that the problem is NP-hard to solve optimally. We formulate
this problem as a linear programming problem and use linear programming
relaxation with randomized rounding. Further, we propose a greedy-based
heuristic with balance correction to solve this problem. We conduct a number of
experiments with real-world trajectory and billboard datasets, and the results
are reported. From the reported results, we observe that the proposed solution
approaches lead to more influence compared to many baseline methods.

### 5. [Downsizing Diffusion Models for Cardinality Estimation](http://arxiv.org/pdf/2510.20681v1)

Authors: Xinhe Mu, Zhaoqi Zhou, Zaijiu Shang, Chuan Zhou, Gang Fu, Guiying Yan, Guoliang Li, Zhiming Ma

Inspired by the performance of score-based diffusion models in estimating
complex text, video, and image distributions with thousands of dimensions, we
introduce Accelerated Diffusion Cardest (ADC), the first joint distribution
cardinality estimator based on a downsized diffusion model.
  To calculate the pointwise density value of data distributions, ADC's density
estimator uses a formula that evaluates log-likelihood by integrating the score
function, a gradient mapping which ADC has learned to efficiently approximate
using its lightweight score estimator. To answer ranged queries, ADC's
selectivity estimator first predicts their selectivity using a Gaussian Mixture
Model (GMM), then uses importance sampling Monte Carlo to correct its
predictions with more accurate pointwise density values calculated by the
density estimator. ADC+ further trains a decision tree to identify the
high-volume, high-selectivity queries that the GMM alone can predict very
accurately, in which case it skips the correction phase to prevent Monte Carlo
from adding more variance. Doing so lowers median Q-error and cuts per-query
latency by 25 percent, making ADC+ usually twice as fast as Naru, arguably the
state-of-the-art joint distribution cardinality estimator.
  Numerical experiments using well-established benchmarks show that on all
real-world datasets tested, ADC+ is capable of rivaling Naru and outperforming
MSCN, DeepDB, LW-Tree, and LW-NN using around 66 percent their storage space,
being at least 3 times as accurate as MSCN on 95th and 99th percentile error.
Furthermore, on a synthetic dataset where attributes exhibit complex,
multilateral correlations, ADC and ADC+ are considerably robust while almost
every other learned model suffered significant accuracy declines. In this case,
ADC+ performs better than any other tested model, being 10 times as accurate as
Naru on 95th and 99th percentile error.

### 6. [RAG-Stack: Co-Optimizing RAG Quality and Performance From the Vector Database Perspective](http://arxiv.org/pdf/2510.20296v1)

Authors: Wenqi Jiang

Retrieval-augmented generation (RAG) has emerged as one of the most prominent
applications of vector databases. By integrating documents retrieved from a
database into the prompt of a large language model (LLM), RAG enables more
reliable and informative content generation. While there has been extensive
research on vector databases, many open research problems remain once they are
considered in the wider context of end-to-end RAG pipelines. One practical yet
challenging problem is how to jointly optimize both system performance and
generation quality in RAG, which is significantly more complex than it appears
due to the numerous knobs on both the algorithmic side (spanning models and
databases) and the systems side (from software to hardware). In this paper, we
present RAG-Stack, a three-pillar blueprint for quality-performance
co-optimization in RAG systems. RAG-Stack comprises: (1) RAG-IR, an
intermediate representation that serves as an abstraction layer to decouple
quality and performance aspects; (2) RAG-CM, a cost model for estimating system
performance given an RAG-IR; and (3) RAG-PE, a plan exploration algorithm that
searches for high-quality, high-performance RAG configurations. We believe this
three-pillar blueprint will become the de facto paradigm for RAG
quality-performance co-optimization in the years to come.

### 7. [FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic](http://arxiv.org/pdf/2510.20467v1)

Authors: Yiwen Peng, Thomas Bonald, Fabian M. Suchanek

Knowledge graph alignment is the task of matching equivalent entities (that
is, instances and classes) and relations across two knowledge graphs. Most
existing methods focus on pure entity-level alignment, computing the similarity
of entities in some embedding space. They lack interpretable reasoning and need
training data to work. In this paper, we propose FLORA, a simple yet effective
method that (1) is unsupervised, i.e., does not require training data, (2)
provides a holistic alignment for entities and relations iteratively, (3) is
based on fuzzy logic and thus delivers interpretable results, (4) provably
converges, (5) allows dangling entities, i.e., entities without a counterpart
in the other KG, and (6) achieves state-of-the-art results on major benchmarks.

### Distributed, Parallel, and Cluster Computing

### 1. [Accurate Performance Predictors for Edge Computing Applications](http://arxiv.org/pdf/2510.20495v1)

Authors: Panagiotis Giannakopoulos, Bart van Knippenberg, Kishor Chandra Joshi, Nicola Calabretta, George Exarchakos

Accurate prediction of application performance is critical for enabling
effective scheduling and resource management in resource-constrained dynamic
edge environments. However, achieving predictable performance in such
environments remains challenging due to the co-location of multiple
applications and the node heterogeneity. To address this, we propose a
methodology that automatically builds and assesses various performance
predictors. This approach prioritizes both accuracy and inference time to
identify the most efficient model. Our predictors achieve up to 90% accuracy
while maintaining an inference time of less than 1% of the Round Trip Time.
These predictors are trained on the historical state of the most correlated
monitoring metrics to application performance and evaluated across multiple
servers in dynamic co-location scenarios. As usecase we consider electron
microscopy (EM) workflows, which have stringent real-time demands and diverse
resource requirements. Our findings emphasize the need for a systematic
methodology that selects server-specific predictors by jointly optimizing
accuracy and inference latency in dynamic co-location scenarios. Integrating
such predictors into edge environments can improve resource utilization and
result in predictable performance.

### 2. [Morpheus: Lightweight RTT Prediction for Performance-Aware Load Balancing](http://arxiv.org/pdf/2510.20506v1)

Authors: Panagiotis Giannakopoulos, Bart van Knippenberg, Kishor Chandra Joshi, Nicola Calabretta, George Exarchakos

Distributed applications increasingly demand low end-to-end latency,
especially in edge and cloud environments where co-located workloads contend
for limited resources. Traditional load-balancing strategies are typically
reactive and rely on outdated or coarse-grained metrics, often leading to
suboptimal routing decisions and increased tail latencies. This paper
investigates the use of round-trip time (RTT) predictors to enhance request
routing by anticipating application latency. We develop lightweight and
accurate RTT predictors that are trained on time-series monitoring data
collected from a Kubernetes-managed GPU cluster. By leveraging a reduced set of
highly correlated monitoring metrics, our approach maintains low overhead while
remaining adaptable to diverse co-location scenarios and heterogeneous
hardware. The predictors achieve up to 95% accuracy while keeping the
prediction delay within 10% of the application RTT. In addition, we identify
the minimum prediction accuracy threshold and key system-level factors required
to ensure effective predictor deployment in resource-constrained clusters.
Simulation-based evaluation demonstrates that performance-aware load balancing
can significantly reduce application RTT and minimize resource waste. These
results highlight the feasibility of integrating predictive load balancing into
future production systems.

### 3. [AsyncHZP: Hierarchical ZeRO Parallelism with Asynchronous Scheduling for Scalable LLM Training](http://arxiv.org/pdf/2510.20111v1)

Authors: Huawei Bai, Yifan Huang, Wenqi Shi, Ansheng You, Feifan Shao, Tengfei Han, Minghui Yu

The training efficiency and scalability of language models on massive
clusters currently remain a critical bottleneck. Mainstream approaches like ND
parallelism are often cumbersome and complex, while flexible alternatives such
as the Zero Redundancy Optimizer (ZeRO) are frequently hampered by
communication overhead. In this paper, we propose Asynchronous Hierarchical
Zero Parallelism (AsyncHZP), a novel asynchronous variant of ZeRO designed to
achieve superior performance while maintaining simplicity and memory
efficiency. Unlike traditional ZeRO, which employs over-fine-grained sharding
that can lead to inefficient communication, AsyncHZP adaptively reshards
parameters, gradients, and optimizer states across different replica groups.
This strategy optimizes device memory utilization and significantly reduces
communication overhead. In addition, we also design a multi-stream asynchronous
scheduling method that executes parameter all-gather and gradient
reduce-scatter operations in dedicated background threads, effectively
overlapping communication with computation while incurring negligible memory
fragmentation. Empirical evaluations on both Dense and Mixture-of-Experts (MoE)
models confirm that AsyncHZP maintains robust stability at scale. It
consistently outperforms classic ND parallelism, achieving state-of-the-art
performance without complex strategic tuning, thereby simplifying the path to
efficient large-scale training.

### 4. [A Full Stack Framework for High Performance Quantum-Classical Computing](http://arxiv.org/pdf/2510.20128v1)

Authors: Xin Zhan, K. Grace Johnson, Aniello Esposito, Barbara Chapman, Marco Fiorentino, Kirk M. Bresniker, Raymond G. Beausoleil, Masoud Mohseni

To address the growing needs for scalable High Performance Computing (HPC)
and Quantum Computing (QC) integration, we present our HPC-QC full stack
framework and its hybrid workload development capability with modular
hardware/device-agnostic software integration approach. The latest development
in extensible interfaces for quantum programming, dispatching, and compilation
within existing mature HPC programming environment are demonstrated. Our HPC-QC
full stack enables high-level, portable invocation of quantum kernels from
commercial quantum SDKs within HPC meta-program in compiled languages (C/C++
and Fortran) as well as Python through a quantum programming interface library
extension. An adaptive circuit knitting hypervisor is being developed to
partition large quantum circuits into sub-circuits that fit on smaller noisy
quantum devices and classical simulators. At the lower-level, we leverage Cray
LLVM-based compilation framework to transform and consume LLVM IR and Quantum
IR (QIR) from commercial quantum software frontends in a retargetable fashion
to different hardware architectures. Several hybrid HPC-QC multi-node multi-CPU
and GPU workloads (including solving linear system of equations, quantum
optimization, and simulating quantum phase transitions) have been demonstrated
on HPE EX supercomputers to illustrate functionality and execution viability
for all three components developed so far. This work provides the framework for
a unified quantum-classical programming environment built upon classical HPC
software stack (compilers, libraries, parallel runtime and process scheduling).

### 5. [ADP-VRSGP: Decentralized Learning with Adaptive Differential Privacy via Variance-Reduced Stochastic Gradient Push](http://arxiv.org/pdf/2510.20157v1)

Authors: Xiaoming Wu, Teng Liu, Xin Wang, Ming Yang, Jiguo Yu

Differential privacy is widely employed in decentralized learning to
safeguard sensitive data by introducing noise into model updates. However,
existing approaches that use fixed-variance noise often degrade model
performance and reduce training efficiency. To address these limitations, we
propose a novel approach called decentralized learning with adaptive
differential privacy via variance-reduced stochastic gradient push (ADP-VRSGP).
This method dynamically adjusts both the noise variance and the learning rate
using a stepwise-decaying schedule, which accelerates training and enhances
final model performance while providing node-level personalized privacy
guarantees. To counteract the slowed convergence caused by large-variance noise
in early iterations, we introduce a progressive gradient fusion strategy that
leverages historical gradients. Furthermore, ADP-VRSGP incorporates
decentralized push-sum and aggregation techniques, making it particularly
suitable for time-varying communication topologies. Through rigorous
theoretical analysis, we demonstrate that ADP-VRSGP achieves robust convergence
with an appropriate learning rate, significantly improving training stability
and speed. Experimental results validate that our method outperforms existing
baselines across multiple scenarios, highlighting its efficacy in addressing
the challenges of privacy-preserving decentralized learning.

### 6. [HHEML: Hybrid Homomorphic Encryption for Privacy-Preserving Machine Learning on Edge](http://arxiv.org/pdf/2510.20243v1)

Authors: Yu Hin Chan, Hao Yang, Shiyu Shen, Xingyu Fan, Shengzhe Lyu, Patrick S. Y. Hung, Ray C. C. Cheung

Privacy-preserving machine learning (PPML) is an emerging topic to handle
secure machine learning inference over sensitive data in untrusted
environments. Fully homomorphic encryption (FHE) enables computation directly
on encrypted data on the server side, making it a promising approach for PPML.
However, it introduces significant communication and computation overhead on
the client side, making it impractical for edge devices. Hybrid homomorphic
encryption (HHE) addresses this limitation by combining symmetric encryption
(SE) with FHE to reduce the computational cost on the client side, and
combining with an FHE-friendly SE can also lessen the processing overhead on
the server side, making it a more balanced and efficient alternative. Our work
proposes a hardware-accelerated HHE architecture built around a lightweight
symmetric cipher optimized for FHE compatibility and implemented as a dedicated
hardware accelerator. To the best of our knowledge, this is the first design to
integrate an end-to-end HHE framework with hardware acceleration. Beyond this,
we also present several microarchitectural optimizations to achieve higher
performance and energy efficiency. The proposed work is integrated into a full
PPML pipeline, enabling secure inference with significantly lower latency and
power consumption than software implementations. Our contributions validate the
feasibility of low-power, hardware- accelerated HHE for edge deployment and
provide a hardware- software co-design methodology for building scalable,
secure machine learning systems in resource-constrained environments.
Experiments on a PYNQ-Z2 platform with the MNIST dataset show over a 50x
reduction in client-side encryption latency and nearly a 2x gain in hardware
throughput compared to existing FPGA-based HHE accelerators.

### 7. [FLAS: a combination of proactive and reactive auto-scaling architecture for distributed services](http://arxiv.org/pdf/2510.20388v1)

Authors: Víctor Rampérez, Javier Soriano, David Lizcano, Juan A. Lara

Cloud computing has established itself as the support for the vast majority
of emerging technologies, mainly due to the characteristic of elasticity it
offers. Auto-scalers are the systems that enable this elasticity by acquiring
and releasing resources on demand to ensure an agreed service level. In this
article we present FLAS (Forecasted Load Auto-Scaling), an auto-scaler for
distributed services that combines the advantages of proactive and reactive
approaches according to the situation to decide the optimal scaling actions in
every moment. The main novelties introduced by FLAS are (i) a predictive model
of the high-level metrics trend which allows to anticipate changes in the
relevant SLA parameters (e.g. performance metrics such as response time or
throughput) and (ii) a reactive contingency system based on the estimation of
high-level metrics from resource use metrics, reducing the necessary
instrumentation (less invasive) and allowing it to be adapted agnostically to
different applications. We provide a FLAS implementation for the use case of a
content-based publish-subscribe middleware (E-SilboPS) that is the cornerstone
of an event-driven architecture. To the best of our knowledge, this is the
first auto-scaling system for content-based publish-subscribe distributed
systems (although it is generic enough to fit any distributed service). Through
an evaluation based on several test cases recreating not only the expected
contexts of use, but also the worst possible scenarios (following the
Boundary-Value Analysis or BVA test methodology), we have validated our
approach and demonstrated the effectiveness of our solution by ensuring
compliance with performance requirements over 99% of the time.

### 8. [Symmetry in Software Platforms as an Architectural Principle](http://arxiv.org/pdf/2510.20389v1)

Authors: Bjorn Remseth

Software platforms often act as structure preserving systems. They provide
consistent interfaces and behaviors that remain stable under specific
transformations that we denote as symmetries. This paper explores the idea that
architectural robustness emerges from enforcing such structural regularities

### 9. [GPU-Accelerated Primal Heuristics for Mixed Integer Programming](http://arxiv.org/pdf/2510.20499v1)

Authors: Akif Çördük, Piotr Sielski, Alice Boucher, Kumar Aatish

We introduce a fusion of GPU accelerated primal heuristics for Mixed Integer
Programming. Leveraging GPU acceleration enables exploration of larger search
regions and faster iterations. A GPU-accelerated PDLP serves as an approximate
LP solver, while a new probing cache facilitates rapid roundings and early
infeasibility detection. Several state-of-the-art heuristics, including
Feasibility Pump, Feasibility Jump, and Fix-and-Propagate, are further
accelerated and enhanced. The combined approach of these GPU-driven algorithms
yields significant improvements over existing methods, both in the number of
feasible solutions and the quality of objectives by achieving 221 feasible
solutions and 22% objective gap in the MIPLIB2017 benchmark on a presolved
dataset.

### 10. [Collective Communication for 100k+ GPUs](http://arxiv.org/pdf/2510.20171v1)

Authors: Min Si, Pavan Balaji, Yongzhou Chen, Ching-Hsiang Chu, Adi Gangidi, Saif Hasan, Subodh Iyengar, Dan Johnson, Bingzhe Liu, Jingliang Ren, Ashmitha Jeevaraj Shetty, Greg Steinbrecher, Xinfeng Xie, Yulun Wang, Bruce Wu, Jingyi Yang, Mingran Yang, Minlan Yu, Cen Zhao, Wes Bland, Denis Boyda, Suman Gumudavelli, Cristian Lumezanu, Rui Miao, Zhe Qu, Venkat Ramesh, Maxim Samoylov, Jan Seidel, Feng Tian, Qiye Tan, Shuqiang Zhang, Yimeng Zhao, Shengbao Zheng, Art Zhu, Hongyi Zeng

The increasing scale of large language models (LLMs) necessitates highly
efficient collective communication frameworks, particularly as training
workloads extend to hundreds of thousands of GPUs. Traditional communication
methods face significant throughput and latency limitations at this scale,
hindering both the development and deployment of state-of-the-art models. This
paper presents the NCCLX collective communication framework, developed at Meta,
engineered to optimize performance across the full LLM lifecycle, from the
synchronous demands of large-scale training to the low-latency requirements of
inference. The framework is designed to support complex workloads on clusters
exceeding 100,000 GPUs, ensuring reliable, high-throughput, and low-latency
data exchange. Empirical evaluation on the Llama4 model demonstrates
substantial improvements in communication efficiency. This research contributes
a robust solution for enabling the next generation of LLMs to operate at
unprecedented scales.

### Discrete Mathematics

### 1. [Boundary vertices of Strongly Connected Digraphs with respect to `Sum Metric'](http://arxiv.org/pdf/2510.20226v1)

Authors: Bijo S. Anand, Manoj Changat, Prasanth G. Narasimha-Shenoi, Mary Shalet Thottungal Joseph, Mithra R, Prakash G. Narasimha-Shenoi

Suppose $D = (V, E)$ is a strongly connected digraph and $u, v \in V (D)$.
Among the many metrics in graphs, the sum metric warrants further exploration.
The sum distance $sd(u, v)$ defined as $sd(u, v) =\overrightarrow{d}(u,
v)+\overrightarrow{d}(v, u)$ is a metric where $\overrightarrow{d}(u, v)$
denotes the length of the shortest directed $u - v$ path in $D$. The four main
boundary vertices in the digraphs are ``boundary vertices, contour vertices,
eccentric vertices'', and ``peripheral vertices'' and their relationships have
been studied. Also, an attempt is made to study the boundary-type sets of
corona product of (di)graphs. The center of the corona product of two strongly
connected digraphs is established. All the boundary-type sets and the center of
the corona product are established in terms of factor digraphs.

### 2. [Labeling and folding multi-labeled trees](http://arxiv.org/pdf/2510.20292v1)

Authors: Vincent Moulton, Andreas Spillner

In 1989 Erd\H{o}s and Sz\'ekely showed that there is a bijection between (i)
the set of rooted trees with $n+1$ vertices whose leaves are bijectively
labeled with the elements of $[\ell]=\{1,2,\dots,\ell\}$ for some $\ell \leq
n$, and (ii) the set of partitions of $[n]=\{1,2,\dots,n\}$. They established
this via a labeling algorithm based on the anti-lexicographic ordering of
non-empty subsets of $[n]$ which extends the labeling of the leaves of a given
tree to a labeling of all of the vertices of that tree. In this paper, we
generalize their approach by developing a labeling algorithm for multi-labeled
trees, that is, rooted trees whose leaves are labeled by positive integers but
in which distinct leaves may have the same label. In particular, we show that
certain orderings of the set of all finite, non-empty multisets of positive
integers can be used to characterize partitions of a multiset that arise from
labelings of multi-labeled trees. As an application, we show that the recently
introduced class of labelable phylogenetic networks is precisely the class of
phylogenetic networks that are stable relative to the so-called folding process
on multi-labeled trees. We also give a bijection between the labelable
phylogenetic networks with leaf-set $[n]$ and certain partitions of multisets.

### 3. [Excluding a Line Minor via Design Matrices and Column Number Bounds for the Circuit Imbalance Measure](http://arxiv.org/pdf/2510.20301v1)

Authors: Daniel Dadush, Friedrich Eisenbrand, Rom Pinchasi, Thomas Rothvoss, Neta Singer

For a real matrix $A \in \mathbb{R}^{d \times n}$ with non-collinear columns,
we show that $n \leq O(d^4 \kappa_A)$ where $\kappa_A$ is the \emph{circuit
imbalance measure} of $A$. The circuit imbalance measure $\kappa$ is a real
analogue of $\Delta$-modularity for integer matrices, satisfying $\kappa_A \leq
\Delta_A$ for integer $A$. The circuit imbalance measure has numerous
applications in the context of linear programming (see Ekbatani, Natura and
V{\'e}gh (2022) for a survey). Our result generalizes the $O(d^4 \Delta_A)$
bound of Averkov and Schymura (2023) for integer matrices and provides the
first polynomial bound holding for all parameter ranges on real matrices.
  To derive our result, similar to the strategy of Geelen, Nelson and Walsh
(2021) for $\Delta$-modular matrices, we show that real representable matroids
induced by $\kappa$-bounded matrices are minor closed and exclude a rank $2$
uniform matroid on $O(\kappa)$ elements as a minor (also known as a line of
length $O(\kappa)$).
  As our main technical contribution, we show that any simple rank $d$ complex
representable matroid which excludes a line of length $l$ has at most $O(d^4
l)$ elements. This complements the tight bound of $(l-3)\binom{d}{2} + d$ for
$l \geq 4$, of Geelen, Nelson and Walsh which holds when the rank $d$ is
sufficiently large compared to $l$ (at least doubly exponential in $l$).

### 4. [Partial Optimality in Cubic Correlation Clustering for General Graphs](http://arxiv.org/pdf/2510.20431v1)

Authors: David Stein, Bjoern Andres, Silvia Di Gregorio

The higher-order correlation clustering problem for a graph $G$ and costs
associated with cliques of $G$ consists in finding a clustering of $G$ so as to
minimize the sum of the costs of those cliques whose nodes all belong to the
same cluster. To tackle this NP-hard problem in practice, local search
heuristics have been proposed and studied in the context of applications. Here,
we establish partial optimality conditions for cubic correlation clustering,
i.e., for the special case of at most 3-cliques. We define and implement
algorithms for deciding these conditions and examine their effectiveness
numerically, on two data sets.

### 5. [A Classification of Long-Refinement Graphs for Colour Refinement](http://arxiv.org/pdf/2510.20802v1)

Authors: Sandra Kiefer, T. Devini de Mel

The Colour Refinement algorithm is a classical procedure to detect symmetries
in graphs, whose most prominent application is in graph-isomorphism tests. The
algorithm and its generalisation, the Weisfeiler-Leman algorithm, evaluate
local information to compute a colouring for the vertices in an iterative
fashion. Different final colours of two vertices certify that no isomorphism
can map one onto the other. The number of iterations that the algorithm takes
to terminate is its central complexity parameter. For a long time, it was open
whether graphs that take the maximum theoretically possible number of Colour
Refinement iterations actually exist. Starting from an exhaustive search on
graphs of low degrees, Kiefer and McKay proved the existence of infinite
families of such long-refinement graphs with degrees 2 and 3, thereby showing
that the trivial upper bound on the iteration number of Colour Refinement is
tight. In this work, we provide a complete characterisation of the
long-refinement graphs with low (or, equivalently, high) degrees. We show that,
with one exception, the aforementioned families are the only long-refinement
graphs with maximum degree at most 3, and we fully classify the long-refinement
graphs with maximum degree 4. To this end, via a reverse-engineering approach,
we show that all low-degree long-refinement graphs can be represented as
compact strings, and we derive multiple structural insights from this
surprising fact. Since long-refinement graphs are closed under taking edge
complements, this also yields a classification of long-refinement graphs with
high degrees. Kiefer and McKay initiated a search for long-refinement graphs
that are only distinguished in the last iteration of Colour Refinement before
termination. We conclude it in this submission by showing that such graphs
cannot exist.

### Data Structures and Algorithms

### 1. [Optimal Rounding for Two-Stage Bipartite Matching](http://arxiv.org/pdf/2510.20153v1)

Authors: Tristan Pollner, Amin Saberi, Anders Wikum

We study two-stage bipartite matching, in which the edges of a bipartite
graph on vertices $(B_1 \cup B_2, I)$ are revealed in two batches. In stage
one, a matching must be selected from among revealed edges $E \subseteq B_1
\times I$. In stage two, edges $E^\theta \subseteq B_2 \times I$ are sampled
from a known distribution, and a second matching must be selected between $B_2$
and unmatched vertices in $I$. The objective is to maximize the total weight of
the combined matching. We design polynomial-time approximations to the optimum
online algorithm, achieving guarantees of $7/8$ for vertex-weighted graphs and
$2\sqrt{2}-2 \approx 0.828$ for edge-weighted graphs under arbitrary
distributions. Both approximation ratios match known upper bounds on the
integrality gap of the natural fractional relaxation, improving upon the
best-known approximation of 0.767 by Feng, Niazadeh, and Saberi for unweighted
graphs whose second batch consists of independently arriving nodes.
  Our results are obtained via an algorithm that rounds a fractional matching
revealed in two stages, aiming to match offline nodes (respectively, edges)
with probability proportional to their fractional weights, up to a
constant-factor loss. We leverage negative association (NA) among offline node
availabilities -- a property induced by dependent rounding -- to derive new
lower bounds on the expected size of the maximum weight matching in random
graphs where one side is realized via NA binary random variables. Moreover, we
extend these results to settings where we have only sample access to the
distribution. In particular, $\text{poly}(n,\epsilon^{-1})$ samples suffice to
obtain an additive loss of $\epsilon$ in the approximation ratio for the
vertex-weighted problem; a similar bound holds for the edge-weighted problem
with an additional (unavoidable) dependence on the scale of edge weights.

### 2. [Smoothed Analysis of Online Metric Matching with a Single Sample: Beyond Metric Distortion](http://arxiv.org/pdf/2510.20288v1)

Authors: Yingxi Li, Ellen Vitercik, Mingwei Yang

In the online metric matching problem, $n$ servers and $n$ requests lie in a
metric space. Servers are available upfront, and requests arrive sequentially.
An arriving request must be matched immediately and irrevocably to an available
server, incurring a cost equal to their distance. The goal is to minimize the
total matching cost.
  We study this problem in the Euclidean metric $[0, 1]^d$, when servers are
adversarial and requests are independently drawn from distinct distributions
that satisfy a mild smoothness condition. Our main result is an
$O(1)$-competitive algorithm for $d \neq 2$ that requires no distributional
knowledge, relying only on a single sample from each request distribution. To
our knowledge, this is the first algorithm to achieve an $o(\log n)$
competitive ratio for non-trivial metrics beyond the i.i.d. setting. Our
approach bypasses the $\Omega(\log n)$ barrier introduced by probabilistic
metric embeddings: instead of analyzing the embedding distortion and the
algorithm separately, we directly bound the cost of the algorithm on the target
metric of a simple deterministic embedding. We then combine this analysis with
lower bounds on the offline optimum for Euclidean metrics, derived via
majorization arguments, to obtain our guarantees.

### 3. [Separations between Oblivious and Adaptive Adversaries for Natural Dynamic Graph Problems](http://arxiv.org/pdf/2510.20341v1)

Authors: Aaron Bernstein, Sayan Bhattacharya, Nick Fischer, Peter Kiss, Thatchaphol Saranurak

We establish the first update-time separation between dynamic algorithms
against oblivious adversaries and those against adaptive adversaries in natural
dynamic graph problems, based on popular fine-grained complexity hypotheses.
Specifically, under the combinatorial BMM hypothesis, we show that every
combinatorial algorithm against an adaptive adversary for the incremental
maximal independent set problem requires $n^{1-o(1)}$ amortized update time.
Furthermore, assuming either the 3SUM or APSP hypotheses, every algorithm for
the decremental maximal clique problem needs $\Delta/n^{o(1)}$ amortized update
time when the initial maximum degree is $\Delta \le \sqrt{n}$. These lower
bounds are matched by existing algorithms against adaptive adversaries. In
contrast, both problems admit algorithms against oblivious adversaries that
achieve $\operatorname{polylog}(n)$ amortized update time [Behnezhad,
Derakhshan, Hajiaghayi, Stein, Sudan; FOCS '19] [Chechik, Zhang; FOCS '19].
Therefore, our separations are exponential. Previously known separations for
dynamic algorithms were either engineered for contrived problems and relied on
strong cryptographic assumptions [Beimel, Kaplan, Mansour, Nissim, Saranurak,
Stemmer; STOC '22], or worked for problems whose inputs are not explicitly
given but are accessed through oracle calls [Bateni, Esfandiari, Fichtenberger,
Henzinger, Jayaram, Mirrokni, Wiese; SODA '23].
  As a byproduct, we also provide a separation between incremental and
decremental algorithms for the triangle detection problem: we show a
decremental algorithm with $\tilde{O}(n^{\omega})$ total update time, while
every incremental algorithm requires $n^{3-o(1)}$ total update time, assuming
the OMv hypothesis. To our knowledge this is the first separation of this kind.

### 4. [$\ell_2/\ell_2$ Sparse Recovery via Weighted Hypergraph Peeling](http://arxiv.org/pdf/2510.20361v1)

Authors: Nick Fischer, Vasileios Nakos

We demonstrate that the best $k$-sparse approximation of a length-$n$ vector
can be recovered within a $(1+\epsilon)$-factor approximation in
$O((k/\epsilon) \log n)$ time using a non-adaptive linear sketch with
$O((k/\epsilon) \log n)$ rows and $O(\log n)$ column sparsity. This improves
the running time of the fastest-known sketch [Nakos, Song; STOC '19] by a
factor of $\log n$, and is optimal for a wide range of parameters.
  Our algorithm is simple and likely to be practical, with the analysis built
on a new technique we call weighted hypergraph peeling. Our method naturally
extends known hypergraph peeling processes (as in the analysis of Invertible
Bloom Filters) to a setting where edges and nodes have (possibly correlated)
weights.

### 5. [From Incremental Transitive Cover to Strongly Polynomial Maximum Flow](http://arxiv.org/pdf/2510.20368v1)

Authors: Daniel Dadush, James B. Orlin, Aaron Sidford, László A. Végh

We provide faster strongly polynomial time algorithms solving maximum flow in
structured $n$-node $m$-arc networks. Our results imply an $n^{\omega +
o(1)}$-time strongly polynomial time algorithms for computing a maximum
bipartite $b$-matching where $\omega$ is the matrix multiplication constant.
Additionally, they imply an $m^{1 + o(1)} W$-time algorithm for solving the
problem on graphs with a given tree decomposition of width $W$.
  We obtain these results by strengthening and efficiently implementing an
approach in Orlin's (STOC 2013) state-of-the-art $O(mn)$ time maximum flow
algorithm. We develop a general framework that reduces solving maximum flow
with arbitrary capacities to (1) solving a sequence of maximum flow problems
with polynomial bounded capacities and (2) dynamically maintaining a
size-bounded supersets of the transitive closure under arc additions; we call
this problem \emph{incremental transitive cover}. Our applications follow by
leveraging recent weakly polynomial, almost linear time algorithms for maximum
flow due to Chen, Kyng, Liu, Peng, Gutenberg, Sachdeva (FOCS 2022) and Brand,
Chen, Kyng, Liu, Peng, Gutenberg, Sachdeva, Sidford (FOCS 2023), and by
developing incremental transitive cover data structures.

### 6. [Compact representations of pattern-avoiding permutations](http://arxiv.org/pdf/2510.20382v1)

Authors: László Kozma, Michal Opler

Pattern-avoiding permutations are a central object of study in both
combinatorics and theoretical computer science. In this paper we design a data
structure that can store any size-$n$ permutation $\tau$ that avoids an
arbitrary (and unknown) fixed pattern $\pi$ in the asymptotically optimal $O(n
\lg{s_\pi})$ bits, where $s_\pi$ is the Stanley-Wilf limit of $\pi$. Our data
structure supports $\tau(i)$ and $\tau^{-1}(i)$ queries in $O(1)$ time,
sidestepping the lower bound of Golynski (SODA 2009) that holds for general
permutations. Comparable results were previously known only in more restricted
cases, e.g., when $\tau$ is separable, which means avoiding the patterns 2413
and 3142.
  We also extend our data structure to support more complex geometric queries
on pattern-avoiding permutations (or planar point sets) such as rectangle range
counting in $O(\lg\lg{n})$ time. This result circumvents the lower bound of
$\Omega{(\lg{n}/\lg\lg{n})}$ by P\u{a}tra\c{s}cu (STOC 2007) that holds in the
general case. For bounded treewidth permutation classes (which include the
above-mentioned separable class), we further reduce the space overhead to a
lower order additive term, making our data structure succinct. This extends and
improves results of Chakraborty et al. (ISAAC 2024) that were obtained for
separable permutations via different techniques. All our data structures can be
constructed in linear time.

### 7. [Parallel $(1+ε)$-Approximate Multi-Commodity Mincost Flow in Almost Optimal Depth and Work](http://arxiv.org/pdf/2510.20456v1)

Authors: Bernhard Haeupler, Yonggang Jiang, Yaowei Long, Thatchaphol Saranurak, Shengzhe Wang

We present a parallel algorithm for computing $(1+\epsilon)$-approximate
mincost flow on an undirected graph with $m$ edges, where capacities and costs
are assigned to both edges and vertices. Our algorithm achieves $\hat{O}(m)$
work and $\hat{O}(1)$ depth when $\epsilon > 1/\mathrm{polylog}(m)$, making
both the work and depth almost optimal, up to a subpolynomial factor.
  Previous algorithms with $\hat{O}(m)$ work required $\Omega(m)$ depth, even
for special cases of mincost flow with only edge capacities or max flow with
vertex capacities. Our result generalizes prior almost-optimal parallel
$(1+\epsilon)$-approximation algorithms for these special cases, including
shortest paths [Li, STOC'20] [Andoni, Stein, Zhong, STOC'20] [Rozhen, Haeupler,
Marinsson, Grunau, Zuzic, STOC'23] and max flow with only edge capacities
[Agarwal, Khanna, Li, Patil, Wang, White, Zhong, SODA'24].
  Our key technical contribution is the first construction of
length-constrained flow shortcuts with $(1+\epsilon)$ length slack,
$\hat{O}(1)$ congestion slack, and $\hat{O}(1)$ step bound. This provides a
strict generalization of the influential concept of
$(\hat{O}(1),\epsilon)$-hopsets [Cohen, JACM'00], allowing for additional
control over congestion. Previous length-constrained flow shortcuts [Haeupler,
Hershkowitz, Li, Roeyskoe, Saranurak, STOC'24] incur a large constant in the
length slack, which would lead to a large approximation factor. To enable our
flow algorithms to work under vertex capacities, we also develop a
close-to-linear time algorithm for computing length-constrained vertex expander
decomposition.
  Building on Cohen's idea of path-count flows [Cohen, SICOMP'95], we further
extend our algorithm to solve $(1+\epsilon)$-approximate $k$-commodity mincost
flow problems with almost-optimal $\hat{O}(mk)$ work and $\hat{O}(1)$ depth,
independent of the number of commodities $k$.

### 8. [Provably Small Portfolios for Multiobjective Optimization with Application to Subsidized Facility Location](http://arxiv.org/pdf/2510.20555v1)

Authors: Swati Gupta, Jai Moondra, Mohit Singh

Many multiobjective real-world problems, such as facility location and bus
routing, become more complex when optimizing the priorities of multiple
stakeholders. These are often modeled using infinite classes of objectives,
e.g., $L_p$ norms over group distances induced by feasible solutions in a fixed
domain. Traditionally, the literature has considered explicitly balancing
`equity' (or min-max) and `efficiency' (or min-sum) objectives to capture this
trade-off. However, the structure of solutions obtained by such modeling
choices can be very different. Taking a solution-centric approach, we introduce
the concept of provably small set of solutions $P$, called a {\it portfolio},
such that for every objective function $h(\cdot)$ in the given class
$\mathbf{C}$, there exists some solution in $P$ which is an
$\alpha$-approximation for $h(\cdot)$. Constructing such portfolios can help
decision-makers understand the impact of balancing across multiple objectives.
  Given a finite set of base objectives $h_1, \ldots, h_N$, we give provable
algorithms for constructing portfolios for (1) the class of conic combinations
$\mathbf{C} = \{\sum_{j \in [N]}\lambda_j h_j: \lambda \ge 0\}$ and for (2) any
class $\mathbf{C}$ of functions that interpolates monotonically between the
min-sum efficiency objective (i.e., $h_1 + \ldots + h_N$) and the min-max
equity objective (i.e., $\max_{j \in [N]} h_j$). Examples of the latter are
$L_p$ norms and top-$\ell$ norms. As an application, we study the Fair
Subsidized Facility Location (FSFL) problem, motivated by the crisis of medical
deserts caused due to pharmacy closures. FSFL allows subsidizing facilities in
underserved areas using revenue from profitable locations. We develop a novel
bicriteria approximation algorithm and show a significant reduction of medical
deserts across states in the U.S.

### 9. [A Deterministic Polylogarithmic Competitive Algorithm for Matching with Delays](http://arxiv.org/pdf/2510.20588v1)

Authors: Marc Dufay, Roger Wattenhofer

In the online Min-cost Perfect Matching with Delays (MPMD) problem, $m$
requests in a metric space are submitted at different times by an adversary.
The goal is to match all requests while (i) minimizing the sum of the distances
between matched pairs as well as (ii) how long each request remained unmatched
after it appeared.
  While there exist almost optimal algorithms when the metric space is finite
and known a priori, this is not the case when the metric space is infinite or
unknown. In this latter case, the best known algorithm, due to Azar and
Jacob-Fanani, has competitiveness $\mathcal{O}(m^{0.59})$ which is
exponentially worse than the best known lower bound of $\Omega(\log m / \log
\log m)$ by Ashlagi et al.
  We present a $\mathcal{O}(\log^5 m)$-competitive algorithm for the MPMD
problem. This algorithm is deterministic and does not need to know the metric
space or $m$ in advance. This is an exponential improvement over previous
results and only a polylogarithmic factor away from the lower bound.

### 10. [On Geometric Bipartite Graphs with Asymptotically Smallest Zarankiewicz Numbers](http://arxiv.org/pdf/2510.20737v1)

Authors: Parinya Chalermsook, Ly Orgo, Minoo Zarsav

This paper considers the \textit{Zarankiewicz problem} in graphs with
low-dimensional geometric representation (i.e., low Ferrers dimension). Our
first result reveals a separation between bipartite graphs of Ferrers dimension
three and four: while $Z(n;k) \leq 9n(k-1)$ for graphs of Ferrers dimension
three, $Z(n;k) \in \Omega\left(n k \cdot \frac{\log n}{\log \log n}\right)$ for
Ferrers dimension four graphs (Chan & Har-Peled, 2023) (Chazelle, 1990). To
complement this, we derive a tight upper bound of $2n(k-1)$ for chordal
bigraphs and $54n(k-1)$ for grid intersection graphs (GIG), a prominent graph
class residing in four Ferrers dimensions and capturing planar bipartite graphs
as well as bipartite intersection graphs of rectangles. Previously, the
best-known bound for GIG was $Z(n;k) \in O(2^{O(k)} n)$, implied by the results
of Fox & Pach (2006) and Mustafa & Pach (2016). Our results advance and offer
new insights into the interplay between Ferrers dimensions and extremal
combinatorics.

### Emerging Technologies

### 1. [Building Network Digital Twins Part II: Real-Time Adaptive PID for Enhanced State Synchronization](http://arxiv.org/pdf/2510.20753v1)

Authors: John Sengendo, Fabrizio Granelli

As we evolve towards more heterogeneous and cutting-edge mobile networks,
Network Digital Twins (NDTs) are proving to be a promising paradigm in solving
challenges faced by network operators, as they give a possibility of
replicating the physical network operations and testing scenarios separately
without interfering with the live network. However, with mobile networks
becoming increasingly dynamic and heterogeneous due to massive device
connectivity, replicating traffic and having NDTs synchronized in real-time
with the physical network remains a challenge, thus necessitating the need to
develop real-time adaptive mechanisms to bridge this gap. In this part II of
our work, we implement a novel framework that integrates an adaptive
Proportional-Integral-Derivative (PID) controller to dynamically improve
synchronization. Additionally, through an interactive user interface, results
of our enhanced approach demonstrate an improvement in real-time traffic
synchronization.

### Formal Languages and Automata Theory

### 1. [Exploring Large Language Models for Access Control Policy Synthesis and Summarization](http://arxiv.org/pdf/2510.20692v1)

Authors: Adarsh Vatsa, Bethel Hall, William Eiers

Cloud computing is ubiquitous, with a growing number of services being hosted
on the cloud every day. Typical cloud compute systems allow administrators to
write policies implementing access control rules which specify how access to
private data is governed. These policies must be manually written, and due to
their complexity can often be error prone. Moreover, existing policies often
implement complex access control specifications and thus can be difficult to
precisely analyze in determining their behavior works exactly as intended.
Recently, Large Language Models (LLMs) have shown great success in automated
code synthesis and summarization. Given this success, they could potentially be
used for automatically generating access control policies or aid in
understanding existing policies. In this paper, we explore the effectiveness of
LLMs for access control policy synthesis and summarization. Specifically, we
first investigate diverse LLMs for access control policy synthesis, finding
that: although LLMs can effectively generate syntactically correct policies,
they have permissiveness issues, generating policies equivalent to the given
specification 45.8% of the time for non-reasoning LLMs, and 93.7% of the time
for reasoning LLMs. We then investigate how LLMs can be used to analyze
policies by introducing a novel semantic-based request summarization approach
which leverages LLMs to generate a precise characterization of the requests
allowed by a policy. Our results show that while there are significant hurdles
in leveraging LLMs for automated policy generation, LLMs show promising results
when combined with symbolic approaches in analyzing existing policies.

### Graphics

### 1. [From Far and Near: Perceptual Evaluation of Crowd Representations Across Levels of Detail](http://arxiv.org/pdf/2510.20558v1)

Authors: Xiaohan Sun, Carol O'Sullivan

In this paper, we investigate how users perceive the visual quality of crowd
character representations at different levels of detail (LoD) and viewing
distances. Each representation: geometric meshes, image-based impostors, Neural
Radiance Fields (NeRFs), and 3D Gaussians, exhibits distinct trade-offs between
visual fidelity and computational performance. Our qualitative and quantitative
results provide insights to guide the design of perceptually optimized LoD
strategies for crowd rendering.

### 2. [Optimizing Feature Ordering in Radar Charts for Multi-Profile Comparison](http://arxiv.org/pdf/2510.20738v1)

Authors: Albert Dorador

Radar charts are widely used to visualize multivariate data and compare
multiple profiles across features. However, the visual clarity of radar charts
can be severely compromised when feature values alternate drastically in
magnitude around the circle, causing areas to collapse, which misrepresents
relative differences. In the present work we introduce a permutation
optimization strategy that reorders features to minimize polygon ``spikiness''
across multiple profiles simultaneously. The method is combinatorial
(exhaustive search) for moderate numbers of features and uses a lexicographic
minimax criterion that first considers overall smoothness (mean jump) and then
the largest single jump as a tie-breaker. This preserves more global
information and produces visually balanced arrangements. We discuss complexity,
practical bounds, and relations to existing approaches that either change the
visualization (e.g., OrigamiPlot) or learn orderings (e.g., Versatile Ordering
Network). An example with two profiles and $p=6$ features (before/after
ordering) illustrates the qualitative improvement.
  Keywords: data visualization, radar charts, combinatorial optimization,
minimax optimization, feature ordering

### Computer Science and Game Theory

### 1. [Strategic Costs of Perceived Bias in Fair Selection](http://arxiv.org/pdf/2510.20606v1)

Authors: L. Elisa Celis, Lingxiao Huang, Milind Sohoni, Nisheeth K. Vishnoi

Meritocratic systems, from admissions to hiring, aim to impartially reward
skill and effort. Yet persistent disparities across race, gender, and class
challenge this ideal. Some attribute these gaps to structural inequality;
others to individual choice. We develop a game-theoretic model in which
candidates from different socioeconomic groups differ in their perceived
post-selection value--shaped by social context and, increasingly, by AI-powered
tools offering personalized career or salary guidance. Each candidate
strategically chooses effort, balancing its cost against expected reward;
effort translates into observable merit, and selection is based solely on
merit. We characterize the unique Nash equilibrium in the large-agent limit and
derive explicit formulas showing how valuation disparities and institutional
selectivity jointly determine effort, representation, social welfare, and
utility. We further propose a cost-sensitive optimization framework that
quantifies how modifying selectivity or perceived value can reduce disparities
without compromising institutional goals. Our analysis reveals a
perception-driven bias: when perceptions of post-selection value differ across
groups, these differences translate into rational differences in effort,
propagating disparities backward through otherwise "fair" selection processes.
While the model is static, it captures one stage of a broader feedback cycle
linking perceptions, incentives, and outcome--bridging rational-choice and
structural explanations of inequality by showing how techno-social environments
shape individual incentives in meritocratic systems.

### 2. [Decentralized Exchange that Mitigate a Bribery Attack](http://arxiv.org/pdf/2510.20645v1)

Authors: Nitin Awathare

Despite the popularity of Hashed Time-Locked Contracts (HTLCs) because of
their use in wide areas of applications such as payment channels, atomic swaps,
etc, their use in exchange is still questionable. This is because of its
incentive incompatibility and susceptibility to bribery attacks.
  State-of-the-art solutions such as MAD-HTLC (Oakland'21) and He-HTLC
(NDSS'23) address this by leveraging miners' profit-driven behaviour to
mitigate such attacks. The former is the mitigation against passive miners;
however, the latter works against both active and passive miners. However, they
consider only two bribing scenarios where either of the parties involved in the
transfer collude with the miner.
  In this paper, we expose vulnerabilities in state-of-the-art solutions by
presenting a miner-collusion bribery attack with implementation and
game-theoretic analysis. Additionally, we propose a stronger attack on MAD-HTLC
than He-HTLC, allowing the attacker to earn profits equivalent to attacking
naive HTLC.
  Leveraging our insights, we propose \prot, a game-theoretically secure HTLC
protocol resistant to all bribery scenarios. \prot\ employs a two-phase
approach, preventing unauthorized token confiscation by third parties, such as
miners. In Phase 1, parties commit to the transfer; in Phase 2, the transfer is
executed without manipulation. We demonstrate \prot's efficiency in transaction
cost and latency via implementations on Bitcoin and Ethereum.

### Human-Computer Interaction

### 1. ["Learning Together": AI-Mediated Support for Parental Involvement in Everyday Learning](http://arxiv.org/pdf/2510.20123v1)

Authors: Yao Li, Jingyi Xie, Ya-Fang Ling, He Zhang, Ge Wang, Gaojian Huang, Rui Yu, Si Chen

Family learning takes place in everyday routines where children and
caregivers read, practice, and develop new skills together. Although AI is
increasingly present in learning environments, most systems remain
child-centered and overlook the collaborative, distributed nature of family
education. This paper investigates how AI can mediate family collaboration by
addressing tensions of coordination, uneven workloads, and parental mediation.
From a formative study with families using AI in daily learning, we identified
challenges in responsibility sharing and recognition of contributions. Building
on these insights, we designed FamLearn, an LLM-powered prototype that
distributes tasks, visualizes contributions, and provides individualized
support. A one-week field study with 11 families shows how this prototype can
ease caregiving burdens, foster recognition, and enrich shared learning
experiences. Our findings suggest that LLMs can move beyond the role of tutor
to act as family mediators - balancing responsibilities, scaffolding
intergenerational participation, and strengthening the relational fabric of
family learning.

### 2. [Designing Intent Communication for Agent-Human Collaboration](http://arxiv.org/pdf/2510.20409v1)

Authors: Yi Li, Francesco Chiossi, Helena Anna Frijns, Jan Leusmann, Julian Rasch, Robin Welsch, Philipp Wintersberger, Florian Michahelles, Albrecht Schmidt

As autonomous agents, from self-driving cars to virtual assistants, become
increasingly present in everyday life, safe and effective collaboration depends
on human understanding of agents' intentions. Current intent communication
approaches are often rigid, agent-specific, and narrowly scoped, limiting their
adaptability across tasks, environments, and user preferences. A key gap
remains: existing models of what to communicate are rarely linked to systematic
choices of how and when to communicate, preventing the development of
generalizable, multi-modal strategies. In this paper, we introduce a
multidimensional design space for intent communication structured along three
dimensions: Transparency (what is communicated), Abstraction (when), and
Modality (how). We apply this design space to three distinct human-agent
collaboration scenarios: (a) bystander interaction, (b) cooperative tasks, and
(c) shared control, demonstrating its capacity to generate adaptable, scalable,
and cross-domain communication strategies. By bridging the gap between intent
content and communication implementation, our design space provides a
foundation for designing safer, more intuitive, and more transferable
agent-human interactions.

### 3. [Risk Psychology & Cyber-Attack Tactics](http://arxiv.org/pdf/2510.20657v1)

Authors: Rubens Kim, Stephan Carney, Yvonne Fonken, Soham Hans, Sofia Hirschmann, Stacy Marsella, Peggy Wu, Nikolos Gurney

We examine whether measured cognitive processes predict cyber-attack
behavior. We analyzed data that included psychometric scale responses and
labeled attack behaviors from cybersecurity professionals who conducted
red-team operations against a simulated enterprise network. We employed
multilevel mixed-effects Poisson regression with technique counts nested within
participants to test whether cognitive processes predicted technique-specific
usage. The scales significantly predicted technique use, but effects varied by
technique rather than operating uniformly. Neither expertise level nor
experimental treatment condition significantly predicted technique patterns,
indicating that cognitive processes may be stronger drivers of technique
selection than training or experience. These findings demonstrate that
individual cognitive differences shape cyber-attack behavior and support the
development of psychology-informed defense strategies.

### 4. [Towards AI Agents for Course Instruction in Higher Education: Early Experiences from the Field](http://arxiv.org/pdf/2510.20255v1)

Authors: Yogesh Simmhan, Varad Kulkarni

This article presents early findings from designing, deploying and evaluating
an AI-based educational agent deployed as the primary instructor in a
graduate-level Cloud Computing course at IISc. We detail the design of a Large
Language Model (LLM)-driven Instructor Agent, and introduce a pedagogical
framework that integrates the Instructor Agent into the course workflow for
actively interacting with the students for content delivery, supplemented by
the human instructor to offer the course structure and undertake
question--answer sessions. We also propose an analytical framework that
evaluates the Agent--Student interaction transcripts using interpretable
engagement metrics of topic coverage, topic depth and turn-level elaboration.
We report early experiences on how students interact with the Agent to explore
concepts, clarify doubts and sustain inquiry-driven dialogue during live
classroom sessions. We also report preliminary analysis on our evaluation
metrics applied across two successive instructional modules that reveals
patterns of engagement evolution, transitioning from broad conceptual
exploration to deeper, focused inquiry. These demonstrate how structured
integration of conversational AI agents can foster reflective learning, offer a
reproducible methodology for studying engagement in authentic classroom
settings, and support scalable, high-quality higher education.

### 5. [From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](http://arxiv.org/pdf/2510.20276v1)

Authors: Wonil Kim, Hyeongseok Wi, Seungsoon Park, Taejun Kim, Sangeun Keum, Keunhyoung Kim, Taewan Kim, Jongmin Jung, Taehyoung Kim, Gaetan Guerrero, Mael Le Goff, Julie Po, Dongjoo Moon, Juhan Nam, Jongpil Lee

Generative AI is reshaping music creation, but its rapid growth exposes
structural gaps in attribution, rights management, and economic models. Unlike
past media shifts, from live performance to recordings, downloads, and
streaming, AI transforms the entire lifecycle of music, collapsing boundaries
between creation, distribution, and monetization. However, existing streaming
systems, with opaque and concentrated royalty flows, are ill-equipped to handle
the scale and complexity of AI-driven production. We propose a content-based
Music AI Agent architecture that embeds attribution directly into the creative
workflow through block-level retrieval and agentic orchestration. Designed for
iterative, session-based interaction, the system organizes music into granular
components (Blocks) stored in BlockDB; each use triggers an Attribution Layer
event for transparent provenance and real-time settlement. This framework
reframes AI from a generative tool into infrastructure for a Fair AI Media
Platform. By enabling fine-grained attribution, equitable compensation, and
participatory engagement, it points toward a post-streaming paradigm where
music functions not as a static catalog but as a collaborative and adaptive
ecosystem.

### 6. [From Far and Near: Perceptual Evaluation of Crowd Representations Across Levels of Detail](http://arxiv.org/pdf/2510.20558v1)

Authors: Xiaohan Sun, Carol O'Sullivan

In this paper, we investigate how users perceive the visual quality of crowd
character representations at different levels of detail (LoD) and viewing
distances. Each representation: geometric meshes, image-based impostors, Neural
Radiance Fields (NeRFs), and 3D Gaussians, exhibits distinct trade-offs between
visual fidelity and computational performance. Our qualitative and quantitative
results provide insights to guide the design of perceptually optimized LoD
strategies for crowd rendering.

### 7. [User Perceptions of Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios](http://arxiv.org/pdf/2510.20721v1)

Authors: Xiaoyuan Wu, Roshni Kaushik, Wenkai Li, Lujo Bauer, Koichi Onoue

Large language models (LLMs) have seen rapid adoption for tasks such as
drafting emails, summarizing meetings, and answering health questions. In such
uses, users may need to share private information (e.g., health records,
contact details). To evaluate LLMs' ability to identify and redact such private
information, prior work developed benchmarks (e.g., ConfAIde, PrivacyLens) with
real-life scenarios. Using these benchmarks, researchers have found that LLMs
sometimes fail to keep secrets private when responding to complex tasks (e.g.,
leaking employee salaries in meeting summaries). However, these evaluations
rely on LLMs (proxy LLMs) to gauge compliance with privacy norms, overlooking
real users' perceptions. Moreover, prior work primarily focused on the
privacy-preservation quality of responses, without investigating nuanced
differences in helpfulness. To understand how users perceive the
privacy-preservation quality and helpfulness of LLM responses to
privacy-sensitive scenarios, we conducted a user study with 94 participants
using 90 scenarios from PrivacyLens. We found that, when evaluating identical
responses to the same scenario, users showed low agreement with each other on
the privacy-preservation quality and helpfulness of the LLM response. Further,
we found high agreement among five proxy LLMs, while each individual LLM had
low correlation with users' evaluations. These results indicate that the
privacy and helpfulness of LLM responses are often specific to individuals, and
proxy LLMs are poor estimates of how real users would perceive these responses
in privacy-sensitive scenarios. Our results suggest the need to conduct
user-centered studies on measuring LLMs' ability to help users while preserving
privacy. Additionally, future research could investigate ways to improve the
alignment between proxy LLMs and users for better estimation of users'
perceived privacy and utility.

### 8. [Empathic Prompting: Non-Verbal Context Integration for Multimodal LLM Conversations](http://arxiv.org/pdf/2510.20743v1)

Authors: Lorenzo Stacchio, Andrea Ubaldi, Alessandro Galdelli, Maurizio Mauri, Emanuele Frontoni, Andrea Gaggioli

We present Empathic Prompting, a novel framework for multimodal human-AI
interaction that enriches Large Language Model (LLM) conversations with
implicit non-verbal context. The system integrates a commercial facial
expression recognition service to capture users' emotional cues and embeds them
as contextual signals during prompting. Unlike traditional multimodal
interfaces, empathic prompting requires no explicit user control; instead, it
unobtrusively augments textual input with affective information for
conversational and smoothness alignment. The architecture is modular and
scalable, allowing integration of additional non-verbal modules. We describe
the system design, implemented through a locally deployed DeepSeek instance,
and report a preliminary service and usability evaluation (N=5). Results show
consistent integration of non-verbal input into coherent LLM outputs, with
participants highlighting conversational fluidity. Beyond this proof of
concept, empathic prompting points to applications in chatbot-mediated
communication, particularly in domains like healthcare or education, where
users' emotional signals are critical yet often opaque in verbal exchanges.

### 9. [FieldGen: From Teleoperated Pre-Manipulation Trajectories to Field-Guided Data Generation](http://arxiv.org/pdf/2510.20774v1)

Authors: Wenhao Wang, Kehe Ye, Xinyu Zhou, Tianxing Chen, Cao Min, Qiaoming Zhu, Xiaokang Yang, Yongjian Shen, Yang Yang, Maoqing Yao, Yao Mu

Large-scale and diverse datasets are vital for training robust robotic
manipulation policies, yet existing data collection methods struggle to balance
scale, diversity, and quality. Simulation offers scalability but suffers from
sim-to-real gaps, while teleoperation yields high-quality demonstrations with
limited diversity and high labor cost. We introduce FieldGen, a field-guided
data generation framework that enables scalable, diverse, and high-quality
real-world data collection with minimal human supervision. FieldGen decomposes
manipulation into two stages: a pre-manipulation phase, allowing trajectory
diversity, and a fine manipulation phase requiring expert precision. Human
demonstrations capture key contact and pose information, after which an
attraction field automatically generates diverse trajectories converging to
successful configurations. This decoupled design combines scalable trajectory
diversity with precise supervision. Moreover, FieldGen-Reward augments
generated data with reward annotations to further enhance policy learning.
Experiments demonstrate that policies trained with FieldGen achieve higher
success rates and improved stability compared to teleoperation-based baselines,
while significantly reducing human effort in long-term real-world data
collection. Webpage is available at https://fieldgen.github.io/.

### 10. [Optimizing Feature Ordering in Radar Charts for Multi-Profile Comparison](http://arxiv.org/pdf/2510.20738v1)

Authors: Albert Dorador

Radar charts are widely used to visualize multivariate data and compare
multiple profiles across features. However, the visual clarity of radar charts
can be severely compromised when feature values alternate drastically in
magnitude around the circle, causing areas to collapse, which misrepresents
relative differences. In the present work we introduce a permutation
optimization strategy that reorders features to minimize polygon ``spikiness''
across multiple profiles simultaneously. The method is combinatorial
(exhaustive search) for moderate numbers of features and uses a lexicographic
minimax criterion that first considers overall smoothness (mean jump) and then
the largest single jump as a tie-breaker. This preserves more global
information and produces visually balanced arrangements. We discuss complexity,
practical bounds, and relations to existing approaches that either change the
visualization (e.g., OrigamiPlot) or learn orderings (e.g., Versatile Ordering
Network). An example with two profiles and $p=6$ features (before/after
ordering) illustrates the qualitative improvement.
  Keywords: data visualization, radar charts, combinatorial optimization,
minimax optimization, feature ordering

### Information Retrieval

### 1. [Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning](http://arxiv.org/pdf/2510.20150v1)

Authors: Yaochen Zhu, Harald Steck, Dawen Liang, Yinhan He, Jundong Li, Nathan Kallus

Large language models (LLMs) are reshaping the recommender system paradigm by
enabling users to express preferences and receive recommendations through
conversations. Yet, aligning LLMs to the recommendation task remains
challenging: pretrained LLMs often generate out-of-catalog items, violate
required output formats, and their ranking quality degrades sharply toward the
end of the generated list. To this end, we propose ConvRec-R1, a two-stage
framework for end-to-end training of LLM-based conversational recommender
systems. In Stage 1, we construct a behavioral-cloning dataset with a
Remap-Reflect-Adjust pipeline, which produces high-quality, catalog-grounded
demonstrations from powerful blackbox LLMs to warm-start the RL training. In
Stage 2, we propose Rank-GRPO, a principled extension of group relative policy
optimization (GRPO) tailored to tasks with rank-style outputs. Rank-GRPO treats
each rank in the recommendation list as the unit instead of token (too
fine-grained) or sequence (too coarse), redefining rewards to remove non-causal
credit assignment and introducing a rank-level importance ratio based on the
geometric mean of rank-wise token probabilities to stabilize policy updates.
Experiments on the public Reddit-v2 dataset show that ConvRec-R1 converges
faster and achieves higher Recall and NDCG than GRPO-style baselines. Code and
datasets are released at https://github.com/yaochenzhu/Rank-GRPO.

### 2. [Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates](http://arxiv.org/pdf/2510.20260v1)

Authors: Changping Meng, Hongyi Ling, Jianling Wang, Yifan Liu, Shuzhou Zhang, Dapeng Hong, Mingyan Gao, Onkar Dalal, Ed Chi, Lichan Hong, Haokai Lu, Ningren Han

Large Language Models (LLMs) empower recommendation systems through their
advanced reasoning and planning capabilities. However, the dynamic nature of
user interests and content poses a significant challenge: While initial
fine-tuning aligns LLMs with domain knowledge and user preferences, it fails to
capture such real-time changes, necessitating robust update mechanisms. This
paper investigates strategies for updating LLM-powered recommenders, focusing
on the trade-offs between ongoing fine-tuning and Retrieval-Augmented
Generation (RAG). Using an LLM-powered user interest exploration system as a
case study, we perform a comparative analysis of these methods across
dimensions like cost, agility, and knowledge incorporation. We propose a hybrid
update strategy that leverages the long-term knowledge adaptation of periodic
fine-tuning with the agility of low-cost RAG. We demonstrate through live A/B
experiments on a billion-user platform that this hybrid approach yields
statistically significant improvements in user satisfaction, offering a
practical and cost-effective framework for maintaining high-quality LLM-powered
recommender systems.

### 3. [Rotate Both Ways: Time-and-Order RoPE for Generative Recommendation](http://arxiv.org/pdf/2510.20455v1)

Authors: Xiaokai Wei, Jiajun Wu, Daiyao Yi, Reza Shirkavand, Michelle Gong

Generative recommenders, typically transformer-based autoregressive models,
predict the next item or action from a user's interaction history. Their
effectiveness depends on how the model represents where an interaction event
occurs in the sequence (discrete index) and when it occurred in wall-clock
time. Prevailing approaches inject time via learned embeddings or relative
attention biases. In this paper, we argue that RoPE-based approaches, if
designed properly, can be a stronger alternative for jointly modeling temporal
and sequential information in user behavior sequences. While vanilla RoPE in
LLMs considers only token order, generative recommendation requires
incorporating both event time and token index. To address this, we propose
Time-and-Order RoPE (TO-RoPE), a family of rotary position embedding designs
that treat index and time as angle sources shaping the query-key geometry
directly. We present three instantiations: early fusion, split-by-dim, and
split-by-head. Extensive experiments on both publicly available datasets and a
proprietary industrial dataset show that TO-RoPE variants consistently improve
accuracy over existing methods for encoding time and index. These results
position rotary embeddings as a simple, principled, and deployment-friendly
foundation for generative recommendation.

### 4. [Generative Reasoning Recommendation via LLMs](http://arxiv.org/pdf/2510.20815v1)

Authors: Minjie Hong, Zetong Zhou, Zirun Guo, Ziang Zhang, Ruofan Hu, Weinan Gan, Jieming Zhu, Zhou Zhao

Despite their remarkable reasoning capabilities across diverse domains, large
language models (LLMs) face fundamental challenges in natively functioning as
generative reasoning recommendation models (GRRMs), where the intrinsic
modeling gap between textual semantics and collaborative filtering signals,
combined with the sparsity and stochasticity of user feedback, presents
significant obstacles. This work explores how to build GRRMs by adapting
pre-trained LLMs, which achieves a unified understanding-reasoning-prediction
manner for recommendation tasks. We propose GREAM, an end-to-end framework that
integrates three components: (i) Collaborative-Semantic Alignment, which fuses
heterogeneous textual evidence to construct semantically consistent, discrete
item indices and auxiliary alignment tasks that ground linguistic
representations in interaction semantics; (ii) Reasoning Curriculum Activation,
which builds a synthetic dataset with explicit Chain-of-Thought supervision and
a curriculum that progresses through behavioral evidence extraction, latent
preference modeling, intent inference, recommendation formulation, and denoised
sequence rewriting; and (iii) Sparse-Regularized Group Policy Optimization
(SRPO), which stabilizes post-training via Residual-Sensitive Verifiable Reward
and Bonus-Calibrated Group Advantage Estimation, enabling end-to-end
optimization under verifiable signals despite sparse successes. GREAM natively
supports two complementary inference modes: Direct Sequence Recommendation for
high-throughput, low-latency deployment, and Sequential Reasoning
Recommendation that first emits an interpretable reasoning chain for causal
transparency. Experiments on three datasets demonstrate consistent gains over
strong baselines, providing a practical path toward verifiable-RL-driven LLM
recommenders.

### 5. [Analyticup E-commerce Product Search Competition Technical Report from Team Tredence_AICOE](http://arxiv.org/pdf/2510.20674v1)

Authors: Rakshith R, Shubham Sharma, Mohammed Sameer Khan, Ankush Chopra

This study presents the multilingual e-commerce search system developed by
the Tredence_AICOE team. The competition features two multilingual relevance
tasks: Query-Category (QC) Relevance, which evaluates how well a user's search
query aligns with a product category, and Query-Item (QI) Relevance, which
measures the match between a multilingual search query and an individual
product listing. To ensure full language coverage, we performed data
augmentation by translating existing datasets into languages missing from the
development set, enabling training across all target languages. We fine-tuned
Gemma-3 12B and Qwen-2.5 14B model for both tasks using multiple strategies.
The Gemma-3 12B (4-bit) model achieved the best QC performance using original
and translated data, and the best QI performance using original, translated,
and minority class data creation. These approaches secured 4th place on the
final leaderboard, with an average F1-score of 0.8857 on the private test set.

### 6. [Multimedia-Aware Question Answering: A Review of Retrieval and Cross-Modal Reasoning Architectures](http://arxiv.org/pdf/2510.20193v1)

Authors: Rahul Raja, Arpita Vats

Question Answering (QA) systems have traditionally relied on structured text
data, but the rapid growth of multimedia content (images, audio, video, and
structured metadata) has introduced new challenges and opportunities for
retrieval-augmented QA. In this survey, we review recent advancements in QA
systems that integrate multimedia retrieval pipelines, focusing on
architectures that align vision, language, and audio modalities with user
queries. We categorize approaches based on retrieval methods, fusion
techniques, and answer generation strategies, and analyze benchmark datasets,
evaluation protocols, and performance tradeoffs. Furthermore, we highlight key
challenges such as cross-modal alignment, latency-accuracy tradeoffs, and
semantic grounding, and outline open problems and future research directions
for building more robust and context-aware QA systems leveraging multimedia
data.

### 7. [From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](http://arxiv.org/pdf/2510.20276v1)

Authors: Wonil Kim, Hyeongseok Wi, Seungsoon Park, Taejun Kim, Sangeun Keum, Keunhyoung Kim, Taewan Kim, Jongmin Jung, Taehyoung Kim, Gaetan Guerrero, Mael Le Goff, Julie Po, Dongjoo Moon, Juhan Nam, Jongpil Lee

Generative AI is reshaping music creation, but its rapid growth exposes
structural gaps in attribution, rights management, and economic models. Unlike
past media shifts, from live performance to recordings, downloads, and
streaming, AI transforms the entire lifecycle of music, collapsing boundaries
between creation, distribution, and monetization. However, existing streaming
systems, with opaque and concentrated royalty flows, are ill-equipped to handle
the scale and complexity of AI-driven production. We propose a content-based
Music AI Agent architecture that embeds attribution directly into the creative
workflow through block-level retrieval and agentic orchestration. Designed for
iterative, session-based interaction, the system organizes music into granular
components (Blocks) stored in BlockDB; each use triggers an Attribution Layer
event for transparent provenance and real-time settlement. This framework
reframes AI from a generative tool into infrastructure for a Fair AI Media
Platform. By enabling fine-grained attribution, equitable compensation, and
participatory engagement, it points toward a post-streaming paradigm where
music functions not as a static catalog but as a collaborative and adaptive
ecosystem.

### 8. [Practical Code RAG at Scale: Task-Aware Retrieval Design Choices under Compute Budgets](http://arxiv.org/pdf/2510.20609v1)

Authors: Timur Galimzyanov, Olga Kolomyttseva, Egor Bogomolov

We study retrieval design for code-focused generation tasks under realistic
compute budgets. Using two complementary tasks from Long Code Arena -- code
completion and bug localization -- we systematically compare retrieval
configurations across various context window sizes along three axes: (i)
chunking strategy, (ii) similarity scoring, and (iii) splitting granularity.
(1) For PL-PL, sparse BM25 with word-level splitting is the most effective and
practical, significantly outperforming dense alternatives while being an order
of magnitude faster. (2) For NL-PL, proprietary dense encoders (Voyager-3
family) consistently beat sparse retrievers, however requiring 100x larger
latency. (3) Optimal chunk size scales with available context: 32-64 line
chunks work best at small budgets, and whole-file retrieval becomes competitive
at 16000 tokens. (4) Simple line-based chunking matches syntax-aware splitting
across budgets. (5) Retrieval latency varies by up to 200x across
configurations; BPE-based splitting is needlessly slow, and BM25 + word
splitting offers the best quality-latency trade-off. Thus, we provide
evidence-based recommendations for implementing effective code-oriented RAG
systems based on task requirements, model constraints, and computational
efficiency.

### 9. [RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines](http://arxiv.org/pdf/2510.20768v1)

Authors: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu

Retrieval-Augmented Generation (RAG) has emerged as the dominant
architectural pattern to operationalize Large Language Model (LLM) usage in
Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to
poisoning attacks, and previously proposed defenses can fail for CTI contexts
as cyber threat information is often completely new for emerging attacks, and
sophisticated threat actors can mimic legitimate formats, terminology, and
stylistic conventions. To address this issue, we propose that the robustness of
modern RAG defenses can be accelerated by applying source credibility
algorithms on corpora, using PageRank as an example. In our experiments, we
demonstrate quantitatively that our algorithm applies a lower authority score
to malicious documents while promoting trusted content, using the standardized
MS MARCO dataset. We also demonstrate proof-of-concept performance of our
algorithm on CTI documents and feeds.

### Machine Learning

### 1. [Competition is the key: A Game Theoretic Causal Discovery Approach](http://arxiv.org/pdf/2510.20106v1)

Authors: Amartya Roy, Souvik Chakraborty

Causal discovery remains a central challenge in machine learning, yet
existing methods face a fundamental gap: algorithms like GES and GraN-DAG
achieve strong empirical performance but lack finite-sample guarantees, while
theoretically principled approaches fail to scale. We close this gap by
introducing a game-theoretic reinforcement learning framework for causal
discovery, where a DDQN agent directly competes against a strong baseline (GES
or GraN-DAG), always warm-starting from the opponent's solution. This design
yields three provable guarantees: the learned graph is never worse than the
opponent, warm-starting strictly accelerates convergence, and most importantly,
with high probability the algorithm selects the true best candidate graph. To
the best of our knowledge, our result makes a first-of-its-kind progress in
explaining such finite-sample guarantees in causal discovery: on synthetic SEMs
(30 nodes), the observed error probability decays with n, tightly matching
theory. On real-world benchmarks including Sachs, Asia, Alarm, Child, Hepar2,
Dream, and Andes, our method consistently improves upon GES and GraN-DAG while
remaining theoretically safe. Remarkably, it scales to large graphs such as
Hepar2 (70 nodes), Dream (100 nodes), and Andes (220 nodes). Together, these
results establish a new class of RL-based causal discovery algorithms that are
simultaneously provably consistent, sample-efficient, and practically scalable,
marking a decisive step toward unifying empirical performance with rigorous
finite-sample theory.

### 2. [On pattern classification with weighted dimensions](http://arxiv.org/pdf/2510.20107v1)

Authors: Ayatullah Faruk Mollah

Studies on various facets of pattern classification is often imperative while
working with multi-dimensional samples pertaining to diverse application
scenarios. In this notion, weighted dimension-based distance measure has been
one of the vital considerations in pattern analysis as it reflects the degree
of similarity between samples. Though it is often presumed to be settled with
the pervasive use of Euclidean distance, plethora of issues often surface. In
this paper, we present (a) a detail analysis on the impact of distance measure
norms and weights of dimensions along with visualization, (b) a novel weighting
scheme for each dimension, (c) incorporation of this dimensional weighting
schema into a KNN classifier, and (d) pattern classification on a variety of
synthetic as well as realistic datasets with the developed model. It has
performed well across diverse experiments in comparison to the traditional KNN
under the same experimental setups. Specifically, for gene expression datasets,
it yields significant and consistent gain in classification accuracy (around
10%) in all cross-validation experiments with different values of k. As such
datasets contain limited number of samples of high dimensions, meaningful
selection of nearest neighbours is desirable, and this requirement is
reasonably met by regulating the shape and size of the region enclosing the k
number of reference samples with the developed weighting schema and appropriate
norm. It, therefore, stands as an important generalization of KNN classifier
powered by weighted Minkowski distance with the present weighting schema.

### 3. [There is No "apple" in Timeseries: Rethinking TSFM through the Lens of Invariance](http://arxiv.org/pdf/2510.20119v1)

Authors: Arian Prabowo, Flora D. Salim

Timeseries foundation models (TSFMs) have multiplied, yet lightweight
supervised baselines and even classical models often match them. We argue this
gap stems from the naive importation of NLP or CV pipelines. In language and
vision, large web-scale corpora densely capture human concepts i.e. there are
countless images and text of apples. In contrast, timeseries data is built to
complement the image and text modalities. There are no timeseries dataset that
contains the concept apple. As a result, the scrape-everything-online paradigm
fails for TS. We posit that progress demands a shift from opportunistic
aggregation to principled design: constructing datasets that systematically
span the space of invariance that preserve temporal semantics. To this end, we
suggest that the ontology of timeseries invariances should be built based on
first principles. Only by ensuring representational completeness through
invariance coverage can TSFMs achieve the aligned structure necessary for
generalisation, reasoning, and truly emergent behaviour.

### 4. [Empowering Targeted Neighborhood Search via Hyper Tour for Large-Scale TSP](http://arxiv.org/pdf/2510.20169v1)

Authors: Tongkai Lu, Shuai Ma, Chongyang Tao

Traveling Salesman Problem (TSP) is a classic NP-hard problem that has
garnered significant attention from both academia and industry. While
neural-based methods have shown promise for solving TSPs, they still face
challenges in scaling to larger instances, particularly in memory constraints
associated with global heatmaps, edge weights, or access matrices, as well as
in generating high-quality initial solutions and insufficient global guidance
for efficiently navigating vast search spaces. To address these challenges, we
propose a Hyper Tour Guided Neighborhood Search (HyperNS) method for
large-scale TSP instances. Inspired by the ``clustering first, route second"
strategy, our approach initially divides the TSP instance into clusters using a
sparse heatmap graph and abstracts them as supernodes, followed by the
generation of a hyper tour to guide both the initialization and optimization
processes. This method reduces the search space by focusing on edges relevant
to the hyper tour, leading to more efficient and effective optimization.
Experimental results on both synthetic and real-world datasets demonstrate that
our approach outperforms existing neural-based methods, particularly in
handling larger-scale instances, offering a significant reduction in the gap to
the optimal solution.

### 5. [Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents](http://arxiv.org/pdf/2510.20199v1)

Authors: Jane H. Lee, Baturay Saglam, Spyridon Pougkakiotis, Amin Karbasi, Dionysis Kalogerias

Constrained optimization provides a common framework for dealing with
conflicting objectives in reinforcement learning (RL). In most of these
settings, the objectives (and constraints) are expressed though the expected
accumulated reward. However, this formulation neglects risky or even possibly
catastrophic events at the tails of the reward distribution, and is often
insufficient for high-stakes applications in which the risk involved in
outliers is critical. In this work, we propose a framework for risk-aware
constrained RL, which exhibits per-stage robustness properties jointly in
reward values and time using optimized certainty equivalents (OCEs). Our
framework ensures an exact equivalent to the original constrained problem
within a parameterized strong Lagrangian duality framework under appropriate
constraint qualifications, and yields a simple algorithmic recipe which can be
wrapped around standard RL solvers, such as PPO. Lastly, we establish the
convergence of the proposed algorithm under common assumptions, and verify the
risk-aware properties of our approach through several numerical experiments.

### 6. [Approximate Replicability in Learning](http://arxiv.org/pdf/2510.20200v1)

Authors: Max Hopkins, Russell Impagliazzo, Christopher Ye

Replicability, introduced by (Impagliazzo et al. STOC '22), is the notion
that algorithms should remain stable under a resampling of their inputs (given
access to shared randomness). While a strong and interesting notion of
stability, the cost of replicability can be prohibitive: there is no replicable
algorithm, for instance, for tasks as simple as threshold learning (Bun et al.
STOC '23). Given such strong impossibility results we ask: under what
approximate notions of replicability is learning possible?
  In this work, we propose three natural relaxations of replicability in the
context of PAC learning: (1) Pointwise: the learner must be consistent on any
fixed input, but not across all inputs simultaneously, (2) Approximate: the
learner must output hypotheses that classify most of the distribution
consistently, (3) Semi: the algorithm is fully replicable, but may additionally
use shared unlabeled samples. In all three cases, for constant replicability
parameters, we obtain sample-optimal agnostic PAC learners: (1) and (2) are
achievable for ``free" using $\Theta(d/\alpha^2)$ samples, while (3) requires
$\Theta(d^2/\alpha^2)$ labeled samples.

### 7. [CO-PFL: Contribution-Oriented Personalized Federated Learning for Heterogeneous Networks](http://arxiv.org/pdf/2510.20219v1)

Authors: Ke Xing, Yanjie Dong, Xiaoyi Fan, Runhao Zeng, Victor C. M. Leung, M. Jamal Deen, Xiping Hu

Personalized federated learning (PFL) addresses a critical challenge of
collaboratively training customized models for clients with heterogeneous and
scarce local data. Conventional federated learning, which relies on a single
consensus model, proves inadequate under such data heterogeneity. Its standard
aggregation method of weighting client updates heuristically or by data volume,
operates under an equal-contribution assumption, failing to account for the
actual utility and reliability of each client's update. This often results in
suboptimal personalization and aggregation bias. To overcome these limitations,
we introduce Contribution-Oriented PFL (CO-PFL), a novel algorithm that
dynamically estimates each client's contribution for global aggregation. CO-PFL
performs a joint assessment by analyzing both gradient direction discrepancies
and prediction deviations, leveraging information from gradient and data
subspaces. This dual-subspace analysis provides a principled and discriminative
aggregation weight for each client, emphasizing high-quality updates.
Furthermore, to bolster personalization adaptability and optimization
stability, CO-PFL cohesively integrates a parameter-wise personalization
mechanism with mask-aware momentum optimization. Our approach effectively
mitigates aggregation bias, strengthens global coordination, and enhances local
performance by facilitating the construction of tailored submodels with stable
updates. Extensive experiments on four benchmark datasets (CIFAR10, CIFAR10C,
CINIC10, and Mini-ImageNet) confirm that CO-PFL consistently surpasses
state-of-the-art methods in in personalization accuracy, robustness,
scalability and convergence stability.

### 8. [Sparse Local Implicit Image Function for sub-km Weather Downscaling](http://arxiv.org/pdf/2510.20228v1)

Authors: Yago del Valle Inclan Redondo, Enrique Arriaga-Varela, Dmitry Lyamzin, Pablo Cervantes, Tiago Ramalho

We introduce SpLIIF to generate implicit neural representations and enable
arbitrary downscaling of weather variables. We train a model from sparse
weather stations and topography over Japan and evaluate in- and
out-of-distribution accuracy predicting temperature and wind, comparing it to
both an interpolation baseline and CorrDiff. We find the model to be up to 50%
better than both CorrDiff and the baseline at downscaling temperature, and
around 10-20% better for wind.

### 9. [Layer-to-Layer Knowledge Mixing in Graph Neural Network for Chemical Property Prediction](http://arxiv.org/pdf/2510.20236v1)

Authors: Teng Jiek See, Daokun Zhang, Mario Boley, David K. Chalmers

Graph Neural Networks (GNNs) are the currently most effective methods for
predicting molecular properties but there remains a need for more accurate
models. GNN accuracy can be improved by increasing the model complexity but
this also increases the computational cost and memory requirement during
training and inference. In this study, we develop Layer-to-Layer Knowledge
Mixing (LKM), a novel self-knowledge distillation method that increases the
accuracy of state-of-the-art GNNs while adding negligible computational
complexity during training and inference. By minimizing the mean absolute
distance between pre-existing hidden embeddings of GNN layers, LKM efficiently
aggregates multi-hop and multi-scale information, enabling improved
representation of both local and global molecular features. We evaluated LKM
using three diverse GNN architectures (DimeNet++, MXMNet, and PAMNet) using
datasets of quantum chemical properties (QM9, MD17 and Chignolin). We found
that the LKM method effectively reduces the mean absolute error of quantum
chemical and biophysical property predictions by up to 9.8% (QM9), 45.3% (MD17
Energy), and 22.9% (Chignolin). This work demonstrates the potential of LKM to
significantly improve the accuracy of GNNs for chemical property prediction
without any substantial increase in training and inference cost.

### 10. [FedGPS: Statistical Rectification Against Data Heterogeneity in Federated Learning](http://arxiv.org/pdf/2510.20250v1)

Authors: Zhiqin Yang, Yonggang Zhang, Chenxin Li, Yiu-ming Cheung, Bo Han, Yixuan Yuan

Federated Learning (FL) confronts a significant challenge known as data
heterogeneity, which impairs model performance and convergence. Existing
methods have made notable progress in addressing this issue. However, improving
performance in certain heterogeneity scenarios remains an overlooked question:
\textit{How robust are these methods to deploy under diverse heterogeneity
scenarios?} To answer this, we conduct comprehensive evaluations across varied
heterogeneity scenarios, showing that most existing methods exhibit limited
robustness. Meanwhile, insights from these experiments highlight that sharing
statistical information can mitigate heterogeneity by enabling clients to
update with a global perspective. Motivated by this, we propose \textbf{FedGPS}
(\textbf{Fed}erated \textbf{G}oal-\textbf{P}ath \textbf{S}ynergy), a novel
framework that seamlessly integrates statistical distribution and gradient
information from others. Specifically, FedGPS statically modifies each client's
learning objective to implicitly model the global data distribution using
surrogate information, while dynamically adjusting local update directions with
gradient information from other clients at each round. Extensive experiments
show that FedGPS outperforms state-of-the-art methods across diverse
heterogeneity scenarios, validating its effectiveness and robustness. The code
is available at: https://github.com/CUHK-AIM-Group/FedGPS.

### Neural and Evolutionary Computing

### 1. [Experimental differentiation and extremization with analog quantum circuits](http://arxiv.org/pdf/2510.20713v1)

Authors: Evan Philip, Julius de Hond, Vytautas Abramavicius, Kaonan Micadei, Mario Dagrada, Panagiotis Barkoutsos, Mourad Beji, Louis-Paul Henry, Vincent E. Elfving, Antonio A. Gentile, Savvas Varsamopoulos

Solving and optimizing differential equations (DEs) is ubiquitous in both
engineering and fundamental science. The promise of quantum architectures to
accelerate scientific computing thus naturally involved interest towards how
efficiently quantum algorithms can solve DEs. Differentiable quantum circuits
(DQC) offer a viable route to compute DE solutions using a variational approach
amenable to existing quantum computers, by producing a machine-learnable
surrogate of the solution. Quantum extremal learning (QEL) complements such
approach by finding extreme points in the output of learnable models of unknown
(implicit) functions, offering a powerful tool to bypass a full DE solution, in
cases where the crux consists in retrieving solution extrema. In this work, we
provide the results from the first experimental demonstration of both DQC and
QEL, displaying their performance on a synthetic usecase. Whilst both DQC and
QEL are expected to require digital quantum hardware, we successfully challenge
this assumption by running a closed-loop instance on a commercial analog
quantum computer, based upon neutral atom technology.

### Networking and Internet Architecture

### 1. [Rediscovering Recurring Routing Results](http://arxiv.org/pdf/2510.20297v1)

Authors: Xiao Song, John Heidemann

Routing is central to networking performance, including: (1) latency in
anycast services and websites served from multiple locations,(2) networking
expenses and throughput in multi-homed enterprises, (3) the ability to keep
traffic domestic when considering data sovereignty. However, understanding and
managing how routing affects these services is challenging. Operators use
Traffic Engineering (TE) with BGP to optimize network performance, but what
they get is the result of all BGP policies throughout the Internet, not just
their local choices. Our paper proposes Fenrir, a new system to rediscover
recurring routing results. Fenrir can discover changes in network routing, even
when it happens multiple hops away from the observer. Fenrir also provides new
methods to quantify the degree of routing change, and to identify routing
"modes" that may reappear. Second, we show that Fenrir can be applied to many
different problems: we use five instances of three different types of systems
to illustrate the generalization: anycast catchments showing in a root DNS
service, route optimization for two multi-homed enterprises, and website
selection for two of the top-10 web services. Each type requires different
types of active measurements, data cleaning and weighting. We demonstrate
Fenrir's methods of detecting and quantifying change are helpful because they
all face similar operational questions: How much effect did traffic engineering
have? Did a third-party change alter my routing? In either case, is the current
routing new, or is it like a routing mode I saw before?

### 2. [Multicast-partitioning in Time-triggered Stream Planning for Time-Sensitive Networks](http://arxiv.org/pdf/2510.20440v1)

Authors: Heiko Geppert, Frank Dürr, Simon Naß, Kurt Rothermel

Multicast allows sending a message to multiple recipients without having to
create and send a separate message for each recipient. This preserves network
bandwidth, which is particularly important in time-sensitive networks. These
networks are commonly used to provide latency-bounded communication for
real-time systems in domains like automotive, avionics, industrial internet of
things, automated shop floors, and smart energy grids. The preserved bandwidth
can be used to admit additional real-time messages with specific quality of
service requirements or to reduce the end-to-end latencies for messages of any
type. However, using multicast communication can complicate traffic planning,
as it requires free queues or available downstream egress ports on all branches
of the multicast tree. In this work, we present a novel multicast partitioning
technique to split multicast trees into smaller multicast or unicast trees.
This allows for a more fine-grained trade-off between bandwidth utilization and
traffic scheduling difficulty. Thus, schedulability in dynamic systems can be
improved, in terms the number of admitted streams and the accumulated network
throughput. We evaluated the multicast partitioning on different network
topologies and with three different scheduling algorithms. With the
partitioning, 5-15\% fewer streams were rejected, while achieving 5-125\% more
network throughput, depending on the scheduling algorithm.

### 3. [Trust, But Verify: An Empirical Evaluation of AI-Generated Code for SDN Controllers](http://arxiv.org/pdf/2510.20703v1)

Authors: Felipe Avencourt Soares, Muriel F. Franco, Eder J. Scheid, Lisandro Z. Granville

Generative Artificial Intelligence (AI) tools have been used to generate
human-like content across multiple domains (e.g., sound, image, text, and
programming). However, their reliability in terms of correctness and
functionality in novel contexts such as programmable networks remains unclear.
Hence, this paper presents an empirical evaluation of the source code of a POX
controller generated by different AI tools, namely ChatGPT, Copilot, DeepSeek,
and BlackBox.ai. To evaluate such a code, three networking tasks of increasing
complexity were defined and for each task, zero-shot and few-shot prompting
techniques were input to the tools. Next, the output code was tested in
emulated network topologies with Mininet and analyzed according to
functionality, correctness, and the need for manual fixes. Results show that
all evaluated models can produce functional controllers. However, ChatGPT and
DeepSeek exhibited higher consistency and code quality, while Copilot and
BlackBox.ai required more adjustments.

### 4. [AI-Enabled Digital Twins for Next-Generation Networks: Forecasting Traffic and Resource Management in 5G/6G](http://arxiv.org/pdf/2510.20796v1)

Authors: John Sengendo, Fabrizio Granelli

As 5G and future 6G mobile networks become increasingly more sophisticated,
the requirements for agility, scalability, resilience, and precision in
real-time service provisioning cannot be met using traditional and
heuristic-based resource management techniques, just like any advancing
technology. With the aim of overcoming such limitations, network operators are
foreseeing Digital Twins (DTs) as key enablers, which are designed as dynamic
and virtual replicas of network infrastructure, allowing operators to model,
analyze, and optimize various operations without any risk of affecting the live
network. However, for Digital Twin Networks (DTNs) to meet the challenges faced
by operators especially in line with resource management, a driving engine is
needed. In this paper, an AI (Artificial Intelligence)-driven approach is
presented by integrating a Long Short-Term Memory (LSTM) neural network into
the DT framework, aimed at forecasting network traffic patterns and proactively
managing resource allocation. Through analytical experiments, the AI-Enabled DT
framework demonstrates superior performance benchmarked against baseline
methods. Our study concludes that embedding AI capabilities within DTs paves
the way for fully autonomous, adaptive, and high-performance network management
in future mobile networks.

### 5. [MAC Aggregation over Lossy Channels in DTLS 1.3](http://arxiv.org/pdf/2510.20419v1)

Authors: Eric Wagner, David Heye, Jan Bauer, Klaus Wehrle, Martin Serror

Aggregating Message Authentication Codes (MACs) promises to save valuable
bandwidth in resource-constrained environments. The idea is simple: Instead of
appending an authentication tag to each message in a communication stream, the
integrity protection of multiple messages is aggregated into a single tag.
Recent studies postulate, e.g., based on simulations, that these benefits also
spread to wireless, and thus lossy, scenarios despite each lost packet
typically resulting in the loss of integrity protection information for
multiple messages. In this paper, we investigate these claims in a real
deployment. Therefore, we first design a MAC aggregation extension for the
Datagram Transport Layer Security (DTLS) 1.3 protocol. Afterward, we
extensively evaluate the performance of MAC aggregation on a complete
communication protocol stack on embedded hardware. We find that MAC aggregation
can indeed increase goodput by up to 50% and save up to 17% of energy
expenditure for the transmission of short messages, even in lossy channels.

### 6. [On the cybersecurity of LoRaWAN-based system: a Smart-Lighting case study](http://arxiv.org/pdf/2510.20494v1)

Authors: Florian Hofer, Barbara Russo

Cyber-physical systems and the Internet of Things (IoT) are key technologies
in the Industry 4.0 vision. They incorporate sensors and actuators to interact
with the physical environment. However, when creating and interconnecting
components to form a heterogeneous smart systems architecture, these face
challenges in cybersecurity. This paper presents an experimental investigation
of architectural configurations for a LoRaWAN-based Smart-Lighting project,
aimed at verifying and improving the system's robustness against attacks. We
assess the system's robustness in a series of iterative experiments conducted
both in-vitro and on-site. The results show that most attacks on a LoRaWAN
network are unsuccessful, also highlighting unresolved issues with the
installed products. The most successful attacks are high-power jamming attacks
within a few meters of the target, which, in the case of gateways, can be
mitigated through gateway redundancy.

### 7. [Collective Communication for 100k+ GPUs](http://arxiv.org/pdf/2510.20171v1)

Authors: Min Si, Pavan Balaji, Yongzhou Chen, Ching-Hsiang Chu, Adi Gangidi, Saif Hasan, Subodh Iyengar, Dan Johnson, Bingzhe Liu, Jingliang Ren, Ashmitha Jeevaraj Shetty, Greg Steinbrecher, Xinfeng Xie, Yulun Wang, Bruce Wu, Jingyi Yang, Mingran Yang, Minlan Yu, Cen Zhao, Wes Bland, Denis Boyda, Suman Gumudavelli, Cristian Lumezanu, Rui Miao, Zhe Qu, Venkat Ramesh, Maxim Samoylov, Jan Seidel, Feng Tian, Qiye Tan, Shuqiang Zhang, Yimeng Zhao, Shengbao Zheng, Art Zhu, Hongyi Zeng

The increasing scale of large language models (LLMs) necessitates highly
efficient collective communication frameworks, particularly as training
workloads extend to hundreds of thousands of GPUs. Traditional communication
methods face significant throughput and latency limitations at this scale,
hindering both the development and deployment of state-of-the-art models. This
paper presents the NCCLX collective communication framework, developed at Meta,
engineered to optimize performance across the full LLM lifecycle, from the
synchronous demands of large-scale training to the low-latency requirements of
inference. The framework is designed to support complex workloads on clusters
exceeding 100,000 GPUs, ensuring reliable, high-throughput, and low-latency
data exchange. Empirical evaluation on the Llama4 model demonstrates
substantial improvements in communication efficiency. This research contributes
a robust solution for enabling the next generation of LLMs to operate at
unprecedented scales.

### Robotics

### 1. [PathFormer: A Transformer with 3D Grid Constraints for Digital Twin Robot-Arm Trajectory Generation](http://arxiv.org/pdf/2510.20161v1)

Authors: Ahmed Alanazi, Duy Ho, Yugyung Lee

Robotic arms require precise, task-aware trajectory planning, yet sequence
models that ignore motion structure often yield invalid or inefficient
executions. We present a Path-based Transformer that encodes robot motion with
a 3-grid (where/what/when) representation and constraint-masked decoding,
enforcing lattice-adjacent moves and workspace bounds while reasoning over task
graphs and action order. Trained on 53,755 trajectories (80% train / 20%
validation), the model aligns closely with ground truth -- 89.44% stepwise
accuracy, 93.32% precision, 89.44% recall, and 90.40% F1 -- with 99.99% of
paths legal by construction. Compiled to motor primitives on an xArm Lite 6
with a depth-camera digital twin, it attains up to 97.5% reach and 92.5% pick
success in controlled tests, and 86.7% end-to-end success across 60
language-specified tasks in cluttered scenes, absorbing slips and occlusions
via local re-grounding without global re-planning. These results show that
path-structured representations enable Transformers to generate accurate,
reliable, and interpretable robot trajectories, bridging graph-based planning
and sequence-based learning and providing a practical foundation for
general-purpose manipulation and sim-to-real transfer.

### 2. [Reinforcement Learning-based Robust Wall Climbing Locomotion Controller in Ferromagnetic Environment](http://arxiv.org/pdf/2510.20174v1)

Authors: Yong Um, Young-Ha Shin, Joon-Ha Kim, Soonpyo Kwon, Hae-Won Park

We present a reinforcement learning framework for quadrupedal wall-climbing
locomotion that explicitly addresses uncertainty in magnetic foot adhesion. A
physics-based adhesion model of a quadrupedal magnetic climbing robot is
incorporated into simulation to capture partial contact, air-gap sensitivity,
and probabilistic attachment failures. To stabilize learning and enable
reliable transfer, we design a three-phase curriculum: (1) acquire a crawl gait
on flat ground without adhesion, (2) gradually rotate the gravity vector to
vertical while activating the adhesion model, and (3) inject stochastic
adhesion failures to encourage slip recovery. The learned policy achieves a
high success rate, strong adhesion retention, and rapid recovery from
detachment in simulation under degraded adhesion. Compared with a model
predictive control (MPC) baseline that assumes perfect adhesion, our controller
maintains locomotion when attachment is intermittently lost. Hardware
experiments with the untethered robot further confirm robust vertical crawling
on steel surfaces, maintaining stability despite transient misalignment and
incomplete attachment. These results show that combining curriculum learning
with realistic adhesion modeling provides a resilient sim-to-real framework for
magnetic climbing robots in complex environments.

### 3. [A Contact-Driven Framework for Manipulating in the Blind](http://arxiv.org/pdf/2510.20177v1)

Authors: Muhammad Suhail Saleem, Lai Yuan, Maxim Likhachev

Robots often face manipulation tasks in environments where vision is
inadequate due to clutter, occlusions, or poor lighting--for example, reaching
a shutoff valve at the back of a sink cabinet or locating a light switch above
a crowded shelf. In such settings, robots, much like humans, must rely on
contact feedback to distinguish free from occupied space and navigate around
obstacles. Many of these environments often exhibit strong structural
priors--for instance, pipes often span across sink cabinets--that can be
exploited to anticipate unseen structure and avoid unnecessary collisions. We
present a theoretically complete and empirically efficient framework for
manipulation in the blind that integrates contact feedback with structural
priors to enable robust operation in unknown environments. The framework
comprises three tightly coupled components: (i) a contact detection and
localization module that utilizes joint torque sensing with a contact particle
filter to detect and localize contacts, (ii) an occupancy estimation module
that uses the history of contact observations to build a partial occupancy map
of the workspace and extrapolate it into unexplored regions with learned
predictors, and (iii) a planning module that accounts for the fact that contact
localization estimates and occupancy predictions can be noisy, computing paths
that avoid collisions and complete tasks efficiently without eliminating
feasible solutions. We evaluate the system in simulation and in the real world
on a UR10e manipulator across two domestic tasks--(i) manipulating a valve
under a kitchen sink surrounded by pipes and (ii) retrieving a target object
from a cluttered shelf. Results show that the framework reliably solves these
tasks, achieving up to a 2x reduction in task completion time compared to
baselines, with ablations confirming the contribution of each module.

### 4. [NODA-MMH: Certified Learning-Aided Nonlinear Control for Magnetically-Actuated Swarm Experiment Toward On-Orbit Proof](http://arxiv.org/pdf/2510.20231v1)

Authors: Yuta Takahashi, Atsuki Ochi, Yoichi Tomioka, Shin-Ichiro Sakai

This study experimentally validates the principle of large-scale satellite
swarm control through learning-aided magnetic field interactions generated by
satellite-mounted magnetorquers. This actuation presents a promising solution
for the long-term formation maintenance of multiple satellites and has
primarily been demonstrated in ground-based testbeds for two-satellite position
control. However, as the number of satellites increases beyond three,
fundamental challenges coupled with the high nonlinearity arise: 1)
nonholonomic constraints, 2) underactuation, 3) scalability, and 4)
computational cost. Previous studies have shown that time-integrated current
control theoretically solves these problems, where the average actuator outputs
align with the desired command, and a learning-based technique further enhances
their performance. Through multiple experiments, we validate critical aspects
of learning-aided time-integrated current control: (1) enhanced controllability
of the averaged system dynamics, with a theoretically guaranteed error bound,
and (2) decentralized current management. We design two-axis coils and a
ground-based experimental setup utilizing an air-bearing platform, enabling a
mathematical replication of orbital dynamics. Based on the effectiveness of the
learned interaction model, we introduce NODA-MMH (Neural power-Optimal Dipole
Allocation for certified learned Model-based Magnetically swarm control
Harness) for model-based power-optimal swarm control. This study complements
our tutorial paper on magnetically actuated swarms for the long-term formation
maintenance problem.

### 5. [NeuralTouch: Neural Descriptors for Precise Sim-to-Real Tactile Robot Control](http://arxiv.org/pdf/2510.20390v1)

Authors: Yijiong Lin, Bowen Deng, Chenghua Lu, Max Yang, Efi Psomopoulou, Nathan F. Lepora

Grasping accuracy is a critical prerequisite for precise object manipulation,
often requiring careful alignment between the robot hand and object. Neural
Descriptor Fields (NDF) offer a promising vision-based method to generate
grasping poses that generalize across object categories. However, NDF alone can
produce inaccurate poses due to imperfect camera calibration, incomplete point
clouds, and object variability. Meanwhile, tactile sensing enables more precise
contact, but existing approaches typically learn policies limited to simple,
predefined contact geometries. In this work, we introduce NeuralTouch, a
multimodal framework that integrates NDF and tactile sensing to enable
accurate, generalizable grasping through gentle physical interaction. Our
approach leverages NDF to implicitly represent the target contact geometry,
from which a deep reinforcement learning (RL) policy is trained to refine the
grasp using tactile feedback. This policy is conditioned on the neural
descriptors and does not require explicit specification of contact types. We
validate NeuralTouch through ablation studies in simulation and zero-shot
transfer to real-world manipulation tasks--such as peg-out-in-hole and bottle
lid opening--without additional fine-tuning. Results show that NeuralTouch
significantly improves grasping accuracy and robustness over baseline methods,
offering a general framework for precise, contact-rich robotic manipulation.

### 6. [MR-UBi: Mixed Reality-Based Underwater Robot Arm Teleoperation System with Reaction Torque Indicator via Bilateral Control](http://arxiv.org/pdf/2510.20407v1)

Authors: Kohei Nishi, Masato Kobayashi, Yuki Uranishi

We present a mixed reality-based underwater robot arm teleoperation system
with a reaction torque indicator via bilateral control (MR-UBi). The reaction
torque indicator (RTI) overlays a color and length-coded torque bar in the
MR-HMD, enabling seamless integration of visual and haptic feedback during
underwater robot arm teleoperation. User studies with sixteen participants
compared MR-UBi against a bilateral-control baseline. MR-UBi significantly
improved grasping-torque control accuracy, increasing the time within the
optimal torque range and reducing both low and high grasping torque range
during lift and pick-and-place tasks with objects of different stiffness.
Subjective evaluations further showed higher usability (SUS) and lower workload
(NASA--TLX). Overall, the results confirm that \textit{MR-UBi} enables more
stable, accurate, and user-friendly underwater robot-arm teleoperation through
the integration of visual and haptic feedback. For additional material, please
check: https://mertcookimg.github.io/mr-ubi

### 7. [Robot Path and Trajectory Planning Considering a Spatially Fixed TCP](http://arxiv.org/pdf/2510.20473v1)

Authors: Bernhard Rameder, Hubert Gattringer, Andreas Mueller, Ronald Naderer

This paper presents a method for planning a trajectory in workspace
coordinates using a spatially fixed tool center point (TCP), while taking into
account the processing path on a part. This approach is beneficial if it is
easier to move the part rather than moving the tool. Whether a mathematical
description that defines the shape to be processed or single points from a
design program are used, the robot path is finally represented using B-splines.
The use of splines enables the path to be continuous with a desired degree,
which finally leads to a smooth robot trajectory. While calculating the robot
trajectory through prescribed orientation, additionally a given velocity at the
TCP has to be considered. The procedure was validated on a real system using an
industrial robot moving an arbitrary defined part.

### 8. [Degradation-Aware Cooperative Multi-Modal GNSS-Denied Localization Leveraging LiDAR-Based Robot Detections](http://arxiv.org/pdf/2510.20480v1)

Authors: Václav Pritzl, Xianjia Yu, Tomi Westerlund, Petr Štěpán, Martin Saska

Accurate long-term localization using onboard sensors is crucial for robots
operating in Global Navigation Satellite System (GNSS)-denied environments.
While complementary sensors mitigate individual degradations, carrying all the
available sensor types on a single robot significantly increases the size,
weight, and power demands. Distributing sensors across multiple robots enhances
the deployability but introduces challenges in fusing asynchronous, multi-modal
data from independently moving platforms. We propose a novel adaptive
multi-modal multi-robot cooperative localization approach using a factor-graph
formulation to fuse asynchronous Visual-Inertial Odometry (VIO), LiDAR-Inertial
Odometry (LIO), and 3D inter-robot detections from distinct robots in a
loosely-coupled fashion. The approach adapts to changing conditions, leveraging
reliable data to assist robots affected by sensory degradations. A novel
interpolation-based factor enables fusion of the unsynchronized measurements.
LIO degradations are evaluated based on the approximate scan-matching Hessian.
A novel approach of weighting odometry data proportionally to the Wasserstein
distance between the consecutive VIO outputs is proposed. A theoretical
analysis is provided, investigating the cooperative localization problem under
various conditions, mainly in the presence of sensory degradations. The
proposed method has been extensively evaluated on real-world data gathered with
heterogeneous teams of an Unmanned Ground Vehicle (UGV) and Unmanned Aerial
Vehicles (UAVs), showing that the approach provides significant improvements in
localization accuracy in the presence of various sensory degradations.

### 9. [RubbleSim: A Photorealistic Structural Collapse Simulator for Confined Space Mapping](http://arxiv.org/pdf/2510.20529v1)

Authors: Constantine Frost, Chad Council, Margaret McGuinness, Nathaniel Hanson

Despite well-reported instances of robots being used in disaster response,
there is scant published data on the internal composition of the void spaces
within structural collapse incidents. Data collected during these incidents is
mired in legal constraints, as ownership is often tied to the responding
agencies, with little hope of public release for research. While engineered
rubble piles are used for training, these sites are also reluctant to release
information about their proprietary training grounds. To overcome this access
challenge, we present RubbleSim -- an open-source, reconfigurable simulator for
photorealistic void space exploration. The design of the simulation assets is
directly informed by visits to numerous training rubble sites at differing
levels of complexity. The simulator is implemented in Unity with
multi-operating system support. The simulation uses a physics-based approach to
build stochastic rubble piles, allowing for rapid iteration between simulation
worlds while retaining absolute knowledge of the ground truth. Using RubbleSim,
we apply a state-of-the-art structure-from-motion algorithm to illustrate how
perception performance degrades under challenging visual conditions inside the
emulated void spaces. Pre-built binaries and source code to implement are
available online: https://github.com/mit-ll/rubble_pile_simulator.

### 10. [C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](http://arxiv.org/pdf/2510.20685v1)

Authors: Ming-Ming Yu, Fei Zhu, Wenzhuo Liu, Yirong Yang, Qunbo Wang, Wenjun Wu, Jing Liu

Embodied agents are expected to perform object navigation in dynamic,
open-world environments. However, existing approaches typically rely on static
trajectories and a fixed set of object categories during training, overlooking
the real-world requirement for continual adaptation to evolving scenarios. To
facilitate related studies, we introduce the continual object navigation
benchmark, which requires agents to acquire navigation skills for new object
categories while avoiding catastrophic forgetting of previously learned
knowledge. To tackle this challenge, we propose C-Nav, a continual visual
navigation framework that integrates two key innovations: (1) A dual-path
anti-forgetting mechanism, which comprises feature distillation that aligns
multi-modal inputs into a consistent representation space to ensure
representation consistency, and feature replay that retains temporal features
within the action decoder to ensure policy consistency. (2) An adaptive
sampling strategy that selects diverse and informative experiences, thereby
reducing redundancy and minimizing memory overhead. Extensive experiments
across multiple model architectures demonstrate that C-Nav consistently
outperforms existing approaches, achieving superior performance even compared
to baselines with full trajectory retention, while significantly lowering
memory requirements. The code will be publicly available at
https://bigtree765.github.io/C-Nav-project.

### Software Engineering

### 1. [Developing a Model-Driven Reengineering Approach for Migrating PL/SQL Triggers to Java: A Practical Experience](http://arxiv.org/pdf/2510.20121v1)

Authors: Carlos J. Fernandez-Candel, Jesus Garcia-Molina, Francisco Javier Bermudez Ruiz, Jose Ramon Hoyos Barcelo, Diego Sevilla Ruiz, Benito Jose Cuesta Viera

Model-driven software engineering (MDE) techniques are not only useful in
forward engineering scenarios, but can also be successfully applied to evolve
existing systems. RAD (Rapid Application Development) platforms emerged in the
nineties, but the success of modern software technologies motivated that a
large number of enterprises tackled the migration of their RAD applications,
such as Oracle Forms. Our research group has collaborated with a software
company in developing a solution to migrate PL/SQL monolithic code on Forms
triggers and program units to Java code separated in several tiers.
  Our research focused on the model-driven reengineering process applied to
develop the migration tool for the conversion of PL/SQL code to Java. Legacy
code is represented in form of KDM (Knowledge-Discovery Metamodel) models. In
this paper, we propose a software process to implement a model-driven
re-engineering. This process integrates a TDD-like approach to incrementally
develop model transformations with three kinds of validations for the generated
code. The implementation and validation of the re-engineering approach are
explained in detail, as well as the evaluation of some issues related with the
application of MDE.

### 2. [FMI-Based Distributed Co-Simulation with Enhanced Security and Intellectual Property Safeguards](http://arxiv.org/pdf/2510.20403v1)

Authors: Santiago Gil, Ecem E. Baş, Christian D. Jensen, Sebastian Engelsgaard, Giuseppe Abbiati, Cláudio Gomes

Distributed co-simulation plays a key role in enabling collaborative modeling
and simulation by different stakeholders while protecting their Intellectual
Property (IP). Although IP protection is provided implicitly by co-simulation,
there is no consensus in the guidelines to conduct distributed co-simulation of
continuous-time or hybrid systems with no exposure to potential hacking
attacks. We propose an approach for distributed co-simulation on top of UniFMU
with enhanced cybersecurity and IP protection mechanisms, ensuring that the
connection is initiated by the client and the models and binaries live on
trusted platforms. We showcase the functionality of this approach using two
co-simulation demos in four different network settings and analyze the
trade-off between IP-protected distribution and performance efficiency in these
settings.

### 3. [Toward Practical Deductive Verification: Insights from a Qualitative Survey in Industry and Academia](http://arxiv.org/pdf/2510.20514v1)

Authors: Lea Salome Brugger, Xavier Denis, Peter Müller

Deductive verification is an effective method to ensure that a given system
exposes the intended behavior. In spite of its proven usefulness and
feasibility in selected projects, deductive verification is still not a
mainstream technique. To pave the way to widespread use, we present a study
investigating the factors enabling successful applications of deductive
verification and the underlying issues preventing broader adoption. We
conducted semi-structured interviews with 30 practitioners of verification from
both industry and academia and systematically analyzed the collected data
employing a thematic analysis approach. Beside empirically confirming familiar
challenges, e.g., the high level of expertise needed for conducting formal
proofs, our data reveal several underexplored obstacles, such as proof
maintenance, insufficient control over automation, and usability concerns. We
further use the results from our data analysis to extract enablers and barriers
for deductive verification and formulate concrete recommendations for
practitioners, tool builders, and researchers, including principles for
usability, automation, and integration with existing workflows.

### 4. [Large Language Models for Fault Localization: An Empirical Study](http://arxiv.org/pdf/2510.20521v1)

Authors: YingJian Xiao, RongQun Hu, WeiWei Gong, HongWei Li, AnQuan Jie

Large language models (LLMs) have demonstrated remarkable capabilities in
code-related tasks, particularly in automated program repair. However, the
effectiveness of such repairs is highly dependent on the performance of
upstream fault localization, for which comprehensive evaluations are currently
lacking. This paper presents a systematic empirical study on LLMs in the
statement-level code fault localization task. We evaluate representative
open-source models (Qwen2.5-coder-32b-instruct, DeepSeek-V3) and closed-source
models (GPT-4.1 mini, Gemini-2.5-flash) to assess their fault localization
capabilities on the HumanEval-Java and Defects4J datasets. The study
investigates the impact of different prompting strategies--including standard
prompts, few-shot examples, and chain-of-reasoning--on model performance, with
a focus on analysis across accuracy, time efficiency, and economic cost
dimensions. Our experimental results show that incorporating bug report context
significantly enhances model performance. Few-shot learning shows potential for
improvement but exhibits noticeable diminishing marginal returns, while
chain-of-thought reasoning's effectiveness is highly contingent on the model's
inherent reasoning capabilities. This study not only highlights the performance
characteristics and trade-offs of different models in fault localization tasks,
but also offers valuable insights into the strengths of current LLMs and
strategies for improving fault localization effectiveness.

### 5. [A Soundness and Precision Benchmark for Java Debloating Tools](http://arxiv.org/pdf/2510.20679v1)

Authors: Jonas Klauke, Tom Ohlmer, Stefan Schott, Serena Elisa Ponta, Wolfram Fischer, Eric Bodden

Modern software development reuses code by importing libraries as
dependencies. Software projects typically include an average of 36
dependencies, with 80% being transitive, meaning they are dependencies of
dependencies. Recent research indicates that only 24.9% of these dependencies
are required at runtime, and even within those, many program constructs remain
unused, adding unnecessary code to the project. This has led to the development
of debloating tools that remove unnecessary dependencies and program constructs
while balancing precision by eliminating unused constructs and soundness by
preserving all required constructs. To systematically evaluate this trade-off,
we developed Deblometer, a micro-benchmark consisting of 59 test cases designed
to assess support for various Java language features in debloating tools. Each
test case includes a manually curated ground truth specifying necessary and
bloated classes, methods, and fields, enabling precise measurement of soundness
and precision. Using Deblometer, we evaluated three popular Java debloating
tools: Deptrim, JShrink, and ProGuard. Our evaluation reveals that all tools
remove required program constructs, which results in changed semantics or
execution crashes. In particular, the dynamic class loading feature introduces
unsoundness in all evaluated tools. Our comparison shows that Deptrim retains
more bloated constructs, while ProGuard removes more required constructs.
JShrink's soundness is significantly affected by limited support for
annotations, which leads to corrupted debloated artifacts. These soundness
issues highlight the need to improve debloating tools to ensure stable and
reliable debloated software.

### 6. [Classport: Designing Runtime Dependency Introspection for Java](http://arxiv.org/pdf/2510.20340v1)

Authors: Serena Cofano, Daniel Williams, Aman Sharma, Martin Monperrus

Runtime introspection of dependencies, i.e., the ability to observe which
dependencies are currently used during program execution, is fundamental for
Software Supply Chain security. Yet, Java has no support for it. We solve this
problem with Classport, a system that embeds dependency information into Java
class files, enabling the retrieval of dependency information at runtime. We
evaluate Classport on six real-world projects, demonstrating the feasibility in
identifying dependencies at runtime. Runtime dependency introspection with
Classport opens important avenues for runtime integrity checking.

### 7. [Symmetry in Software Platforms as an Architectural Principle](http://arxiv.org/pdf/2510.20389v1)

Authors: Bjorn Remseth

Software platforms often act as structure preserving systems. They provide
consistent interfaces and behaviors that remain stable under specific
transformations that we denote as symmetries. This paper explores the idea that
architectural robustness emerges from enforcing such structural regularities

### 8. [Automated Cloud Infrastructure-as-Code Reconciliation with AI Agents](http://arxiv.org/pdf/2510.20211v1)

Authors: Zhenning Yang, Hui Guan, Victor Nicolet, Brandon Paulsen, Joey Dodds, Daniel Kroening, Ang Chen

Cloud infrastructure is managed through a mix of interfaces -- traditionally,
cloud consoles, command-line interfaces (CLI), and SDKs are the tools of
choice. Recently, Infrastructure-as-Code/IaC frameworks (e.g., Terraform) have
quickly gained popularity. Unlike conventional tools, IaC~frameworks encode the
infrastructure in a "source-of-truth" configuration. They are capable of
automatically carrying out modifications to the cloud -- deploying, updating,
or destroying resources -- to bring the actual infrastructure into alignment
with the IaC configuration. However, when IaC is used alongside consoles, CLIs,
or SDKs, it loses visibility into external changes, causing infrastructure
drift, where the configuration becomes outdated, and later IaC operations may
undo valid updates or trigger errors.
  We present NSync, an automated system for IaC reconciliation that propagates
out-of-band changes back into the IaC program. Our key insight is that
infrastructure changes eventually all occur via cloud API invocations -- the
lowest layer for cloud management operations. NSync gleans insights from API
traces to detect drift (i.e., non-IaC changes) and reconcile it (i.e., update
the IaC configuration to capture the changes). It employs an agentic
architecture that leverages LLMs to infer high-level intents from noisy API
sequences, synthesize targeted IaC updates using specialized tools, and
continually improve through a self-evolving knowledge base of past
reconciliations. We further introduce a novel evaluation pipeline for injecting
realistic drifts into cloud infrastructure and assessing reconciliation
performance. Experiments across five real-world Terraform projects and 372
drift scenarios show that NSync outperforms the baseline both in terms of
accuracy (from 0.71 to 0.97 pass@3) and token efficiency (1.47$\times$
improvement).

### 9. [Exploring Large Language Models for Access Control Policy Synthesis and Summarization](http://arxiv.org/pdf/2510.20692v1)

Authors: Adarsh Vatsa, Bethel Hall, William Eiers

Cloud computing is ubiquitous, with a growing number of services being hosted
on the cloud every day. Typical cloud compute systems allow administrators to
write policies implementing access control rules which specify how access to
private data is governed. These policies must be manually written, and due to
their complexity can often be error prone. Moreover, existing policies often
implement complex access control specifications and thus can be difficult to
precisely analyze in determining their behavior works exactly as intended.
Recently, Large Language Models (LLMs) have shown great success in automated
code synthesis and summarization. Given this success, they could potentially be
used for automatically generating access control policies or aid in
understanding existing policies. In this paper, we explore the effectiveness of
LLMs for access control policy synthesis and summarization. Specifically, we
first investigate diverse LLMs for access control policy synthesis, finding
that: although LLMs can effectively generate syntactically correct policies,
they have permissiveness issues, generating policies equivalent to the given
specification 45.8% of the time for non-reasoning LLMs, and 93.7% of the time
for reasoning LLMs. We then investigate how LLMs can be used to analyze
policies by introducing a novel semantic-based request summarization approach
which leverages LLMs to generate a precise characterization of the requests
allowed by a policy. Our results show that while there are significant hurdles
in leveraging LLMs for automated policy generation, LLMs show promising results
when combined with symbolic approaches in analyzing existing policies.

### 10. [Learning to Triage Taint Flows Reported by Dynamic Program Analysis in Node.js Packages](http://arxiv.org/pdf/2510.20739v1)

Authors: Ronghao Ni, Aidan Z. H. Yang, Min-Chien Hsu, Nuno Sabino, Limin Jia, Ruben Martins, Darion Cassel, Kevin Cheang

Program analysis tools often produce large volumes of candidate vulnerability
reports that require costly manual review, creating a practical challenge: how
can security analysts prioritize the reports most likely to be true
vulnerabilities?
  This paper investigates whether machine learning can be applied to
prioritizing vulnerabilities reported by program analysis tools. We focus on
Node.js packages and collect a benchmark of 1,883 Node.js packages, each
containing one reported ACE or ACI vulnerability. We evaluate a variety of
machine learning approaches, including classical models, graph neural networks
(GNNs), large language models (LLMs), and hybrid models that combine GNN and
LLMs, trained on data based on a dynamic program analysis tool's output. The
top LLM achieves $F_{1} {=} 0.915$, while the best GNN and classical ML models
reaching $F_{1} {=} 0.904$. At a less than 7% false-negative rate, the leading
model eliminates 66.9% of benign packages from manual review, taking around 60
ms per package. If the best model is tuned to operate at a precision level of
0.8 (i.e., allowing 20% false positives amongst all warnings), our approach can
detect 99.2% of exploitable taint flows while missing only 0.8%, demonstrating
strong potential for real-world vulnerability triage.

### Systems and Control

### 1. [Design Optimization and Global Impact Assessment of Solar-Thermal Direct Air Carbon Capture](http://arxiv.org/pdf/2510.20135v1)

Authors: Zhiyuan Fan, Bolun Xu

The dual challenge of decarbonizing the economy and meeting rising global
energy demand underscores the need for scalable and cost-effective carbon
dioxide removal technologies. Direct air capture (DAC) is among the most
promising approaches, but its high energy intensity, particularly the thermal
energy required for sorbent regeneration, remains a critical barrier to cost
reduction and sustainable deployment. This study explores solar-thermal DAC
systems that combine concentrated solar thermal technology with low-cost
sand-based thermal energy storage to meet this demand. We analyze the
techno-economic performance of such systems in both grid-connected and
stand-alone configurations. Results show that solar-thermal DAC can achieve
annual capacity factors exceeding 80% and CO2 removal costs as low as 160-200
USD per ton, making it competitive with leading DAC technologies. The proposed
system operates most efficiently with short-cycle sorbents that align with
solar availability. The stand-alone Solar-DAC systems, which rely solely on
solar energy for both electricity and thermal energy, are particularly
promising in regions with high solar capacity and sandy terrain, exhibiting
minimal ambient sensitivity from temperature and humidity. An optimal 6000
ton/yr modular system design takes <1 km2 land-use requirement and potentially
>26 Gt/year DAC capacity is identified for sandy terrain alone globally. In
areas with sedimentary basins suitable for CO2 storage, solar-powered DAC
offers a lower-cost alternative to geothermal heating, which often faces
geological and economic constraints.

### 2. [Soft Switching Expert Policies for Controlling Systems with Uncertain Parameters](http://arxiv.org/pdf/2510.20152v1)

Authors: Junya Ikemoto

This paper proposes a simulation-based reinforcement learning algorithm for
controlling systems with uncertain and varying system parameters. While
simulators are useful for safely learning control policies for physical
systems, mitigating the reality gap remains a major challenge. To address the
challenge, we propose a two-stage algorithm. In the first stage, multiple
control policies are learned for systems with different parameters in a
simulator. In the second stage, for a real system, the control policies learned
in the first stage are smoothly switched using an online convex optimization
algorithm based on observations. Our proposed algorithm is demonstrated through
numerical experiments.

### 3. [Observer-based Differentiators for Noisy Signals](http://arxiv.org/pdf/2510.20234v1)

Authors: Van Huynh, Hieu Trinh, Riley Bain

We present a collection of different types of observation systems that work
as differentiators. These observer-based differentiators can produce estimates
for derivatives of a given signal, even though the given signal is prone to
noise.

### 4. [Multi-layer Optimized Coordination of Smart Building Resources in Active Power Distribution Systems](http://arxiv.org/pdf/2510.20313v1)

Authors: Mohammadali Rostami, Saeed Lotfifard, Mladen Kezunovic

This paper proposes a multi-actor coordination platform for the optimal
utilization of smart buildings resources, including roof top PV generation and
battery energy storage system (BESS), in active power distribution systems. The
proposed multi-actor coordination includes the Smart Building Coordinator
(SBC), Micro-Grid Coordinator (MGC) and Distribution System Coordinator (DSC).
The coordinators operate independently and only exchange limited information
with each other to reach an optimal solution. In the proposed platform, a
hierarchical optimization problem is solved to optimally determine the
operating point of all distribution system resources. The proposed platform
fully preserves the confidentiality of the behind the meter (BTM) data of the
buildings since no information about the status of the PV system, BESS, and
load of the building is shared with the owner of the power system. The proposed
platform has a flexible and scalable architecture where the computational task
of coordinating microgrids and smart buildings with distribution grid is
performed locally at the MGC and SBC layers, respectively. Numerical
simulations show the efficacy of the proposed platform in coordinating the BTM
resources with the rest of the distribution system.

### 5. [On MIMO Stability Analysis Methods Applied to Inverter-Based Resources Connected to Power Systems](http://arxiv.org/pdf/2510.20384v1)

Authors: Anton A. Stoorvogel, Saeed Lotfifard, Ali Saberi

This paper presents a critical review of methods
  commonly employed in the literature for small signal stability analysis of
  inverter based resources (IBRs). It discusses the intended purposes
  of these methods and outlines both their proper and improper
  implementations. The paper provides insights into the applicability
  of these techniques, clarifies their inherent limitations, and
  discusses and illustrates common sources of misinterpretation.

### 6. [Interlacing in Controllers Implementation: Frequency Analysis](http://arxiv.org/pdf/2510.20394v1)

Authors: Julian Salt

The main goal of this contribution is to explain how to use interlacing
techniques for LTI controllers implementation and analyze different struc-
tures in this environment. These considerations lead to an important com-
putation saving in constrained resource environments. It has been also intro-
duced new procedures for obtaining the blocks related to different real and
complex controllers poles. The resultant time-varying system is modeled using
proper discrete lifting techniques and a new and efficient dual-rate fre-
quency response computation allows to determine the characteristics of the
control loop with interlaced controller. Examples illustrate the theoretical
proposals.

### 7. [A Multifunctional Capacitive Sensing Platform for Wireless Vascular and Heart Monitoring](http://arxiv.org/pdf/2510.20415v1)

Authors: Parviz Zolfaghari, Beril Yagmur Koca, Taher Abbasiasl, Hakan Urey, Hadi Mirzajani

We present a multifunctional, antenna-integrated capacitive sensing (MAiCaS)
platform for passive, wireless, and real-time cardiovascular monitoring. Unlike
conventional systems that require separate sensors and wireless modules, our
device unifies sensing, telemetry, and mechanical functionality into a compact
and scalable design by exploiting the parasitic capacitance of an inductive
antenna as a strain-sensitive element. The sensor is fabricated using a
cleanroom-free, single-step UV laser patterning process on a flexible PDMS
substrate, reducing manufacturing complexity and enabling high reproducibility.
The MAiCaS is suitable for three different applications: as a sensor for
epicardial strain measurement, a stent as a sensor, and a vascular graft
sensor. We demonstrate MAiCaS's versatility by validating its wireless
resonance-based response to strain, pressure, and deformation across unrolled
and rolled forms. In vitro experiments demonstrated consistent resonance
frequency shifts under physiological conditions, with stable performance on
skin, in PBS, human serum, and simulated vascular environments. Repeatability
and aging tests confirmed its long-term reliability and elasticity under cyclic
loading. Calibration curves revealed high sensitivity across all
configurations, with wireless interrogation achieved through S11 parameter
measurements and resonance frequency shift as the output metric. The
sensitivity of the device was measured to be 2.9 MHz per 1% strain, 0.43
MHz/mmHg, and 309.6kHz/\textmu m for epicardial patch, graft, and stent
integrated sensor, respectively. The operation of MAiCaS was evaluated in a
human experiment. This monolithic sensor architecture provides a scalable and
cost-effective solution for battery-free monitoring of vascular dynamics, with
potential for remote diagnostics, post-surgical follow-up, and continuous
cardiovascular health management.

### 8. [Joint Computation Offloading and Resource Management for Cooperative Satellite-Aerial-Marine Internet of Things Networks](http://arxiv.org/pdf/2510.20443v1)

Authors: Shuang Qi, Bin Lin, Yiqin Deng, Hongyang Pan, Xu Hu

Devices within the marine Internet of Things (MIoT) can connect to low Earth
orbit (LEO) satellites and unmanned aerial vehicles (UAVs) to facilitate
low-latency data transmission and execution, as well as enhanced-capacity data
storage. However, without proper traffic handling strategy, it is still
difficult to effectively meet the low-latency requirements. In this paper, we
consider a cooperative satellite-aerial-MIoT network (CSAMN) for maritime edge
computing and maritime data storage to prioritize delay-sensitive (DS) tasks by
employing mobile edge computing, while handling delay-tolerant (DT) tasks via
the store-carry-forward method. Considering the delay constraints of DS tasks,
we formulate a constrained joint optimization problem of maximizing
satellite-collected data volume while minimizing system energy consumption by
controlling four interdependent variables, including the transmit power of UAVs
for DS tasks, the start time of DT tasks, computing resource allocation, and
offloading ratio. To solve this non-convex and non-linear problem, we propose a
joint computation offloading and resource management (JCORM) algorithm using
the Dinkelbach method and linear programming. Our results show that the volume
of data collected by the proposed JCORM algorithm can be increased by up to
41.5% compared to baselines. Moreover, JCORM algorithm achieves a dramatic
reduction in computational time, from a maximum of 318.21 seconds down to just
0.16 seconds per experiment, making it highly suitable for real-time maritime
applications.

### 9. [Decentralized Small Gain and Phase Stability Conditions for Grid-Forming Converters: Limitations and Extensions](http://arxiv.org/pdf/2510.20544v1)

Authors: Diego Cifelli, Adolfo Anta

The increasing share of converter based resources in power systems calls for
scalable methods to analyse stability without relying on exhaustive system wide
simulations. Decentralized small gain and small-phase criteria have recently
been proposed for this purpose, but their applicability to grid forming
converters is severely limited by the sectoriality assumption, which is not
typically satisfied at low frequencies. This work revisits and extends mixed
gain phase conditions by introducing loop shaping transformations that
reformulate converter and network models in alternative coordinate frames. The
proposed approach resolves intrinsic non sectoriality at low frequencies and
reduces conservativeness, thereby improving the applicability of decentralized
stability certification. Analytical results are illustrated using an infinite
bus system first and then extended to the IEEE 14 bus network, demonstrating
the practicality and scalability of the method. These findings provide a
pathway toward less conservative and more widely applicable decentralized
stability certificates in power grids.

### 10. [Safe Decentralized Density Control of Multi-Robot Systems using PDE-Constrained Optimization with State Constraints](http://arxiv.org/pdf/2510.20643v1)

Authors: Longchen Niu, Gennaro Notomista

In this paper, we introduce a decentralized optimization-based density
controller designed to enforce set invariance constraints in multi-robot
systems. By designing a decentralized control barrier function, we derived
sufficient conditions under which local safety constraints guarantee global
safety. We account for localization and motion noise explicitly by modeling
robots as spatial probability density functions governed by the Fokker-Planck
equation. Compared to traditional centralized approaches, our controller
requires less computational and communication power, making it more suitable
for deployment in situations where perfect communication and localization are
impractical. The controller is validated through simulations and experiments
with four quadcopters.

### Machine Learning (Statistics Category)

### 1. [Compositional Generation for Long-Horizon Coupled PDEs](http://arxiv.org/pdf/2510.20141v1)

Authors: Somayajulu L. N. Dhulipala, Deep Ray, Nicholas Forman

Simulating coupled PDE systems is computationally intensive, and prior
efforts have largely focused on training surrogates on the joint (coupled)
data, which requires a large amount of data. In the paper, we study
compositional diffusion approaches where diffusion models are only trained on
the decoupled PDE data and are composed at inference time to recover the
coupled field. Specifically, we investigate whether the compositional strategy
can be feasible under long time horizons involving a large number of time
steps. In addition, we compare a baseline diffusion model with that trained
using the v-parameterization strategy. We also introduce a symmetric
compositional scheme for the coupled fields based on the Euler scheme. We
evaluate on Reaction-Diffusion and modified Burgers with longer time grids, and
benchmark against a Fourier Neural Operator trained on coupled data. Despite
seeing only decoupled training data, the compositional diffusion models recover
coupled trajectories with low error. v-parameterization can improve accuracy
over a baseline diffusion model, while the neural operator surrogate remains
strongest given that it is trained on the coupled data. These results show that
compositional diffusion is a viable strategy towards efficient, long-horizon
modeling of coupled PDEs.

### 2. [Neural Networks for Censored Expectile Regression Based on Data Augmentation](http://arxiv.org/pdf/2510.20344v1)

Authors: Wei Cao, Shanshan Wang

Expectile regression neural networks (ERNNs) are powerful tools for capturing
heterogeneity and complex nonlinear structures in data. However, most existing
research has primarily focused on fully observed data, with limited attention
paid to scenarios involving censored observations. In this paper, we propose a
data augmentation based ERNNs algorithm, termed DAERNN, for modeling
heterogeneous censored data. The proposed DAERNN is fully data driven, requires
minimal assumptions, and offers substantial flexibility. Simulation studies and
real data applications demonstrate that DAERNN outperforms existing censored
ERNNs methods and achieves predictive performance comparable to models trained
on fully observed data. Moreover, the algorithm provides a unified framework
for handling various censoring mechanisms without requiring explicit parametric
model specification, thereby enhancing its applicability to practical censored
data analysis.

### 3. [Learning Decentralized Routing Policies via Graph Attention-based Multi-Agent Reinforcement Learning in Lunar Delay-Tolerant Networks](http://arxiv.org/pdf/2510.20436v1)

Authors: Federico Lozano-Cuadra, Beatriz Soret, Marc Sanchez Net, Abhishek Cauligi, Federico Rossi

We present a fully decentralized routing framework for multi-robot
exploration missions operating under the constraints of a Lunar Delay-Tolerant
Network (LDTN). In this setting, autonomous rovers must relay collected data to
a lander under intermittent connectivity and unknown mobility patterns. We
formulate the problem as a Partially Observable Markov Decision Problem (POMDP)
and propose a Graph Attention-based Multi-Agent Reinforcement Learning
(GAT-MARL) policy that performs Centralized Training, Decentralized Execution
(CTDE). Our method relies only on local observations and does not require
global topology updates or packet replication, unlike classical approaches such
as shortest path and controlled flooding-based algorithms. Through Monte Carlo
simulations in randomized exploration environments, GAT-MARL provides higher
delivery rates, no duplications, and fewer packet losses, and is able to
leverage short-term mobility forecasts; offering a scalable solution for future
space robotic systems for planetary exploration, as demonstrated by successful
generalization to larger rover teams.

### 4. [Concentration and excess risk bounds for imbalanced classification with synthetic oversampling](http://arxiv.org/pdf/2510.20472v1)

Authors: Touqeer Ahmad, Mohammadreza M. Kalan, François Portier, Gilles Stupfler

Synthetic oversampling of minority examples using SMOTE and its variants is a
leading strategy for addressing imbalanced classification problems. Despite the
success of this approach in practice, its theoretical foundations remain
underexplored. We develop a theoretical framework to analyze the behavior of
SMOTE and related methods when classifiers are trained on synthetic data. We
first derive a uniform concentration bound on the discrepancy between the
empirical risk over synthetic minority samples and the population risk on the
true minority distribution. We then provide a nonparametric excess risk
guarantee for kernel-based classifiers trained using such synthetic data. These
results lead to practical guidelines for better parameter tuning of both SMOTE
and the downstream learning algorithm. Numerical experiments are provided to
illustrate and support the theoretical findings

### 5. [Diffusion Autoencoders with Perceivers for Long, Irregular and Multimodal Astronomical Sequences](http://arxiv.org/pdf/2510.20595v1)

Authors: Yunyi Shen, Alexander Gagliano

Self-supervised learning has become a central strategy for representation
learning, but the majority of architectures used for encoding data have only
been validated on regularly-sampled inputs such as images, audios. and videos.
In many scientific domains, data instead arrive as long, irregular, and
multimodal sequences. To extract semantic information from these data, we
introduce the Diffusion Autoencoder with Perceivers (daep). daep tokenizes
heterogeneous measurements, compresses them with a Perceiver encoder, and
reconstructs them with a Perceiver-IO diffusion decoder, enabling scalable
learning in diverse data settings. To benchmark the daep architecture, we adapt
the masked autoencoder to a Perceiver encoder/decoder design, and establish a
strong baseline (maep) in the same architectural family as daep. Across diverse
spectroscopic and photometric astronomical datasets, daep achieves lower
reconstruction errors, produces more discriminative latent spaces, and better
preserves fine-scale structure than both VAE and maep baselines. These results
establish daep as an effective framework for scientific domains where data
arrives as irregular, heterogeneous sequences.

### 6. [What Does It Take to Build a Performant Selective Classifier?](http://arxiv.org/pdf/2510.20242v1)

Authors: Stephan Rabanser, Nicolas Papernot

Selective classifiers improve model reliability by abstaining on inputs the
model deems uncertain. However, few practical approaches achieve the
gold-standard performance of a perfect-ordering oracle that accepts examples
exactly in order of correctness. Our work formalizes this shortfall as the
selective-classification gap and present the first finite-sample decomposition
of this gap to five distinct sources of looseness: Bayes noise, approximation
error, ranking error, statistical noise, and implementation- or shift-induced
slack. Crucially, our analysis reveals that monotone post-hoc calibration --
often believed to strengthen selective classifiers -- has limited impact on
closing this gap, since it rarely alters the model's underlying score ranking.
Bridging the gap therefore requires scoring mechanisms that can effectively
reorder predictions rather than merely rescale them. We validate our
decomposition on synthetic two-moons data and on real-world vision and language
benchmarks, isolating each error component through controlled experiments. Our
results confirm that (i) Bayes noise and limited model capacity can account for
substantial gaps, (ii) only richer, feature-aware calibrators meaningfully
improve score ordering, and (iii) data shift introduces a separate slack that
demands distributionally robust training. Together, our decomposition yields a
quantitative error budget as well as actionable design guidelines that
practitioners can use to build selective classifiers which approximate ideal
oracle behavior more closely.

### 7. [On Multiple Robustness of Proximal Dynamic Treatment Regimes](http://arxiv.org/pdf/2510.20451v1)

Authors: Yuanshan Gao, Yang Bai, Yifan Cui

Dynamic treatment regimes are sequential decision rules that adapt treatment
according to individual time-varying characteristics and outcomes to achieve
optimal effects, with applications in precision medicine, personalized
recommendations, and dynamic marketing. Estimating optimal dynamic treatment
regimes via sequential randomized trials might face costly and ethical hurdles,
often necessitating the use of historical observational data. In this work, we
utilize proximal causal inference framework for learning optimal dynamic
treatment regimes when the unconfoundedness assumption fails. Our contributions
are four-fold: (i) we propose three nonparametric identification methods for
optimal dynamic treatment regimes; (ii) we establish the semiparametric
efficiency bound for the value function of a given regime; (iii) we propose a
(K+1)-robust method for learning optimal dynamic treatment regimes, where K is
the number of stages; (iv) as a by-product for marginal structural models, we
establish identification and estimation of counterfactual means under a static
regime. Numerical experiments validate the efficiency and multiple robustness
of our proposed methods.

### 8. [Finding the Sweet Spot: Trading Quality, Cost, and Speed During Inference-Time LLM Reflection](http://arxiv.org/pdf/2510.20653v1)

Authors: Jack Butler, Nikita Kozodoi, Zainab Afolabi, Brian Tyacke, Gaiar Baimuratov

As Large Language Models (LLMs) continue to evolve, practitioners face
increasing options for enhancing inference-time performance without model
retraining, including budget tuning and multi-step techniques like
self-reflection. While these methods improve output quality, they create
complex trade-offs among accuracy, cost, and latency that remain poorly
understood across different domains. This paper systematically compares
self-reflection and budget tuning across mathematical reasoning and translation
tasks. We evaluate prominent LLMs, including Anthropic Claude, Amazon Nova, and
Mistral families, along with other models under varying reflection depths and
compute budgets to derive Pareto optimal performance frontiers. Our analysis
reveals substantial domain dependent variation in self-reflection
effectiveness, with performance gains up to 220\% in mathematical reasoning. We
further investigate how reflection round depth and feedback mechanism quality
influence performance across model families. To validate our findings in a
real-world setting, we deploy a self-reflection enhanced marketing content
localisation system at Lounge by Zalando, where it shows market-dependent
effectiveness, reinforcing the importance of domain specific evaluation when
deploying these techniques. Our results provide actionable guidance for
selecting optimal inference strategies given specific domains and resource
constraints. We open source our self-reflection implementation for
reproducibility at
https://github.com/aws-samples/sample-genai-reflection-for-bedrock.

### 9. [The Reality Gap in Robotics: Challenges, Solutions, and Best Practices](http://arxiv.org/pdf/2510.20808v1)

Authors: Elie Aljalbout, Jiaxu Xing, Angel Romero, Iretiayo Akinola, Caelan Reed Garrett, Eric Heiden, Abhishek Gupta, Tucker Hermans, Yashraj Narang, Dieter Fox, Davide Scaramuzza, Fabio Ramos

Machine learning has facilitated significant advancements across various
robotics domains, including navigation, locomotion, and manipulation. Many such
achievements have been driven by the extensive use of simulation as a critical
tool for training and testing robotic systems prior to their deployment in
real-world environments. However, simulations consist of abstractions and
approximations that inevitably introduce discrepancies between simulated and
real environments, known as the reality gap. These discrepancies significantly
hinder the successful transfer of systems from simulation to the real world.
Closing this gap remains one of the most pressing challenges in robotics.
Recent advances in sim-to-real transfer have demonstrated promising results
across various platforms, including locomotion, navigation, and manipulation.
By leveraging techniques such as domain randomization, real-to-sim transfer,
state and action abstractions, and sim-real co-training, many works have
overcome the reality gap. However, challenges persist, and a deeper
understanding of the reality gap's root causes and solutions is necessary. In
this survey, we present a comprehensive overview of the sim-to-real landscape,
highlighting the causes, solutions, and evaluation metrics for the reality gap
and sim-to-real transfer.

### 10. [On the Structure of Stationary Solutions to McKean-Vlasov Equations with Applications to Noisy Transformers](http://arxiv.org/pdf/2510.20094v1)

Authors: Krishnakumar Balasubramanian, Sayan Banerjee, Philippe Rigollet

We study stationary solutions of McKean-Vlasov equations on the circle. Our
main contributions stem from observing an exact equivalence between solutions
of the stationary McKean-Vlasov equation and an infinite-dimensional quadratic
system of equations over Fourier coefficients, which allows explicit
characterization of the stationary states in a sequence space rather than a
function space. This framework provides a transparent description of local
bifurcations, characterizing their periodicity, and resonance structures, while
accommodating singular potentials. We derive analytic expressions that
characterize the emergence, form and shape (supercritical, critical,
subcritical or transcritical) of bifurcations involving possibly multiple
Fourier modes and connect them with discontinuous phase transitions. We also
characterize, under suitable assumptions, the detailed structure of the
stationary bifurcating solutions that are accurate upto an arbitrary number of
Fourier modes. At the global level, we establish regularity and concavity
properties of the free energy landscape, proving existence, compactness, and
coexistence of globally minimizing stationary measures, further identifying
discontinuous phase transitions with points of non-differentiability of the
minimum free energy map. As an application, we specialize the theory to the
Noisy Mean-Field Transformer model, where we show how changing the inverse
temperature parameter $\beta$ affects the geometry of the infinitely many
bifurcations from the uniform measure. We also explain how increasing $\beta$
can lead to a rich class of approximate multi-mode stationary solutions which
can be seen as `metastable states'. Further, a sharp transition from continuous
to discontinuous (first-order) phase behavior is observed as $\beta$ increases.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-24 PST.

### 1. [Productive engagement patterns and their association with life satisfaction among elderly sandwich generation](https://www.nature.com/articles/s41598-025-21227-8)

Authors: Yang Yu et al.

### 2. [BirdNeRF: fast neural reconstruction of large-scale scenes from aerial imagery](https://www.nature.com/articles/s41598-025-21206-z)

Authors: Huiqing Zhang et al.

### 3. [Research topic detection in scientific articles using a hybrid BERT integrated telescopic vector tree model with emperor penguin enhanced NSGA II optimization](https://www.nature.com/articles/s41598-025-21145-9)

Authors: Keerthi Krishnan et al.

### 4. [Reducing annotation burden in physical activity research using vision language models](https://www.nature.com/articles/s41598-025-21350-6)

Authors: Abram Schönfeldt et al.

### 5. [An efficient coverage path planning method for UAV in complex concave regions](https://www.nature.com/articles/s41598-025-20978-8)

Authors: Wenxing Wu et al.

### 6. [An improved greedy equivalent search method based on relative entropy](https://www.nature.com/articles/s41598-025-21219-8)

Authors: Xiaohan Liu et al.

### 7. [Privacy preservation in diabetic disease prediction using federated learning based on efficient cross stage recurrent model](https://www.nature.com/articles/s41598-025-21229-6)

Authors: R. Jayalakshmi et al.

### 8. [Enhanced brain tumor segmentation in medical imaging using multi-modal multi-scale contextual aggregation and attention fusion](https://www.nature.com/articles/s41598-025-21255-4)

Authors: Waqar Aslam et al.

### 9. [Explore brain-inspired machine intelligence for connecting dots on graphs through holographic blueprint of oscillatory synchronization](https://www.nature.com/articles/s41467-025-64471-2)

Authors: Tingting Dan et al.

