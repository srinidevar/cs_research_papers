# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-09-10 17:00:25.534003 PST.

### Artificial Intelligence

### 1. [Performative Thinking? The Brittle Correlation Between CoT Length and Problem Complexity](http://arxiv.org/pdf/2509.07339v1)

Authors: Vardhan Palod, Karthik Valmeekam, Kaya Stechly, Subbarao Kambhampati

Intermediate token generation (ITG), where a model produces output before the
solution, has been proposed as a method to improve the performance of language
models on reasoning tasks. While these reasoning traces or Chain of Thoughts
(CoTs) are correlated with performance gains, the mechanisms underlying them
remain unclear. A prevailing assumption in the community has been to
anthropomorphize these tokens as "thinking", treating longer traces as evidence
of higher problem-adaptive computation. In this work, we critically examine
whether intermediate token sequence length reflects or correlates with problem
difficulty. To do so, we train transformer models from scratch on derivational
traces of the A* search algorithm, where the number of operations required to
solve a maze problem provides a precise and verifiable measure of problem
complexity. We first evaluate the models on trivial free-space problems,
finding that even for the simplest tasks, they often produce excessively long
reasoning traces and sometimes fail to generate a solution. We then
systematically evaluate the model on out-of-distribution problems and find that
the intermediate token length and ground truth A* trace length only loosely
correlate. We notice that the few cases where correlation appears are those
where the problems are closer to the training distribution, suggesting that the
effect arises from approximate recall rather than genuine problem-adaptive
computation. This suggests that the inherent computational complexity of the
problem instance is not a significant factor, but rather its distributional
distance from the training data. These results challenge the assumption that
intermediate trace generation is adaptive to problem difficulty and caution
against interpreting longer sequences in systems like R1 as automatically
indicative of "thinking effort".

### 2. [SheetDesigner: MLLM-Powered Spreadsheet Layout Generation with Rule-Based and Vision-Based Reflection](http://arxiv.org/pdf/2509.07473v1)

Authors: Qin Chen, Yuanyi Ren, Xiaojun Ma, Mugeng Liu, Han Shi, Dongmei Zhang

Spreadsheets are critical to data-centric tasks, with rich, structured
layouts that enable efficient information transmission. Given the time and
expertise required for manual spreadsheet layout design, there is an urgent
need for automated solutions. However, existing automated layout models are
ill-suited to spreadsheets, as they often (1) treat components as axis-aligned
rectangles with continuous coordinates, overlooking the inherently discrete,
grid-based structure of spreadsheets; and (2) neglect interrelated semantics,
such as data dependencies and contextual links, unique to spreadsheets. In this
paper, we first formalize the spreadsheet layout generation task, supported by
a seven-criterion evaluation protocol and a dataset of 3,326 spreadsheets. We
then introduce SheetDesigner, a zero-shot and training-free framework using
Multimodal Large Language Models (MLLMs) that combines rule and vision
reflection for component placement and content population. SheetDesigner
outperforms five baselines by at least 22.6\%. We further find that through
vision modality, MLLMs handle overlap and balance well but struggle with
alignment, necessitates hybrid rule and visual reflection strategies. Our codes
and data is available at Github.

### 3. [Towards explainable decision support using hybrid neural models for logistic terminal automation](http://arxiv.org/pdf/2509.07577v1)

Authors: Riccardo DElia, Alberto Termine, Francesco Flammini

The integration of Deep Learning (DL) in System Dynamics (SD) modeling for
transportation logistics offers significant advantages in scalability and
predictive accuracy. However, these gains are often offset by the loss of
explainability and causal reliability $-$ key requirements in critical
decision-making systems. This paper presents a novel framework for
interpretable-by-design neural system dynamics modeling that synergizes DL with
techniques from Concept-Based Interpretability, Mechanistic Interpretability,
and Causal Machine Learning. The proposed hybrid approach enables the
construction of neural network models that operate on semantically meaningful
and actionable variables, while retaining the causal grounding and transparency
typical of traditional SD models. The framework is conceived to be applied to
real-world case-studies from the EU-funded project AutoMoTIF, focusing on
data-driven decision support, automation, and optimization of multimodal
logistic terminals. We aim at showing how neuro-symbolic methods can bridge the
gap between black-box predictive models and the need for critical decision
support in complex dynamical environments within cyber-physical systems enabled
by the industrial Internet-of-Things.

### 4. [Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling](http://arxiv.org/pdf/2509.07617v1)

Authors: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

Direct Prompt Injection (DPI) attacks pose a critical security threat to
Large Language Models (LLMs) due to their low barrier of execution and high
potential damage. To address the impracticality of existing white-box/gray-box
methods and the poor transferability of black-box methods, we propose an
activations-guided prompt injection attack framework. We first construct an
Energy-based Model (EBM) using activations from a surrogate model to evaluate
the quality of adversarial prompts. Guided by the trained EBM, we employ the
token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize
adversarial prompts, thereby enabling gradient-free black-box attacks.
Experimental results demonstrate our superior cross-model transferability,
achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6%
improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen
task scenarios. Interpretability analysis reveals a correlation between
activations and attack effectiveness, highlighting the critical role of
semantic patterns in transferable vulnerability exploitation.

### 5. [Getting In Contract with Large Language Models -- An Agency Theory Perspective On Large Language Model Alignment](http://arxiv.org/pdf/2509.07642v1)

Authors: Sascha Kaltenpoth, Oliver Müller

Adopting Large language models (LLMs) in organizations potentially
revolutionizes our lives and work. However, they can generate off-topic,
discriminating, or harmful content. This AI alignment problem often stems from
misspecifications during the LLM adoption, unnoticed by the principal due to
the LLM's black-box nature. While various research disciplines investigated AI
alignment, they neither address the information asymmetries between
organizational adopters and black-box LLM agents nor consider organizational AI
adoption processes. Therefore, we propose LLM ATLAS (LLM Agency Theory-Led
Alignment Strategy) a conceptual framework grounded in agency (contract)
theory, to mitigate alignment problems during organizational LLM adoption. We
conduct a conceptual literature analysis using the organizational LLM adoption
phases and the agency theory as concepts. Our approach results in (1) providing
an extended literature analysis process specific to AI alignment methods during
organizational LLM adoption and (2) providing a first LLM alignment
problem-solution space.

### 6. [DeepGraphLog for Layered Neurosymbolic AI](http://arxiv.org/pdf/2509.07665v1)

Authors: Adem Kikaj, Giuseppe Marra, Floris Geerts, Robin Manhaeve, Luc De Raedt

Neurosymbolic AI (NeSy) aims to integrate the statistical strengths of neural
networks with the interpretability and structure of symbolic reasoning.
However, current NeSy frameworks like DeepProbLog enforce a fixed flow where
symbolic reasoning always follows neural processing. This restricts their
ability to model complex dependencies, especially in irregular data structures
such as graphs. In this work, we introduce DeepGraphLog, a novel NeSy framework
that extends ProbLog with Graph Neural Predicates. DeepGraphLog enables
multi-layer neural-symbolic reasoning, allowing neural and symbolic components
to be layered in arbitrary order. In contrast to DeepProbLog, which cannot
handle symbolic reasoning via neural methods, DeepGraphLog treats symbolic
representations as graphs, enabling them to be processed by Graph Neural
Networks (GNNs). We showcase the capabilities of DeepGraphLog on tasks in
planning, knowledge graph completion with distant supervision, and GNN
expressivity. Our results demonstrate that DeepGraphLog effectively captures
complex relational dependencies, overcoming key limitations of existing NeSy
systems. By broadening the applicability of neurosymbolic AI to
graph-structured domains, DeepGraphLog offers a more expressive and flexible
framework for neural-symbolic integration.

### 7. [Unleashing the True Potential of LLMs: A Feedback-Triggered Self-Correction with Long-Term Multipath Decoding](http://arxiv.org/pdf/2509.07676v1)

Authors: Jipeng Li, Zeyu Gao, Yubin Qi, Hande Dong, Weijian Chen, Qiang Lin

Large Language Models (LLMs) have achieved remarkable performance across
diverse tasks, yet their susceptibility to generating incorrect content during
inference remains a critical unsolved challenge. While self-correction methods
offer potential solutions, their effectiveness is hindered by two inherent
limitations: (1) the absence of reliable guidance signals for error
localization, and (2) the restricted reasoning depth imposed by conventional
next-token decoding paradigms. To address these issues, we propose
Feedback-Triggered Regeneration (FTR), a novel framework that synergizes user
feedback with enhanced decoding dynamics. Specifically, FTR activates response
regeneration only upon receiving negative user feedback, thereby circumventing
error propagation from faulty self-assessment while preserving originally
correct outputs. Furthermore, we introduce Long-Term Multipath (LTM) decoding,
which enables systematic exploration of multiple reasoning trajectories through
delayed sequence evaluation, effectively overcoming the myopic decision-making
characteristic of standard next-token prediction. Extensive experiments on
mathematical reasoning and code generation benchmarks demonstrate that our
framework achieves consistent and significant improvements over
state-of-the-art prompt-based self-correction methods.

### 8. [FHIR-RAG-MEDS: Integrating HL7 FHIR with Retrieval-Augmented Large Language Models for Enhanced Medical Decision Support](http://arxiv.org/pdf/2509.07706v1)

Authors: Yildiray Kabak, Gokce B. Laleci Erturkmen, Mert Gencturk, Tuncay Namli, A. Anil Sinaci, Ruben Alcantud Corcoles, Cristina Gomez Ballesteros, Pedro Abizanda, Asuman Dogac

In this study, we propose FHIR-RAG-MEDS system that aims to integrate Health
Level 7 Fast Healthcare Interoperability Resources (HL7 FHIR) with a
Retrieval-Augmented Generation (RAG)-based system to improve personalized
medical decision support on evidence-based clinical guidelines, emphasizing the
need for research in practical applications. In the evolving landscape of
medical decision support systems, integrating advanced technologies such as RAG
and HL7 FHIR can significantly enhance clinical decision-making processes.
Despite the potential of these technologies, there is limited research on their
integration in practical applications.

### 9. [RIMO: An Easy-to-Evaluate, Hard-to-Solve Olympiad Benchmark for Advanced Mathematical Reasoning](http://arxiv.org/pdf/2509.07711v1)

Authors: Ziye Chen, Chengwei Qin, Yao Shu

As large language models (LLMs) reach high scores on established mathematical
benchmarks, such as GSM8K and MATH, the research community has turned to
International Mathematical Olympiad (IMO) problems to push the evaluation
frontier. However, existing Olympiad-level benchmarks suffer from practical
constraints that introduce grading noise and potential bias, such as
heterogeneous answer formats requiring model-based judges and a reliance on
potentially flawed solutions. We introduce RIMO, a two-track benchmark designed
to preserve peak Olympiad difficulty while eliminating this evaluation noise.
The first track, RIMO-N, rewrites 335 IMO problems to admit a single, unique
integer answer, allowing for deterministic correctness checking. The second
track, RIMO-P, features 456 proof problems with expert-checked solutions, which
are decomposed into a sequence of sub-problems to evaluate the step-by-step
reasoning process via an automated grading system. Our benchmarking of ten
frontier LLMs, including GPT-4o and Gemini 2.5 Flash, reveals that while these
systems excel on older benchmarks, their performance drops sharply on RIMO.
These results highlight a substantial gap between current LLM capabilities and
actual Olympiad-level reasoning. By providing a challenging yet
easy-to-evaluate suite, RIMO offers a high-resolution yardstick for future
research, presenting a clear target for closing the profound reasoning gap our
findings expose.

### 10. [The Carbon Footprint Wizard: A Knowledge-Augmented AI Interface for Streamlining Food Carbon Footprint Analysis](http://arxiv.org/pdf/2509.07733v1)

Authors: Mustafa Kaan Aslan, Reinout Heijungs, Filip Ilievski

Environmental sustainability, particularly in relation to climate change, is
a key concern for consumers, producers, and policymakers. The carbon footprint,
based on greenhouse gas emissions, is a standard metric for quantifying the
contribution to climate change of activities and is often assessed using life
cycle assessment (LCA). However, conducting LCA is complex due to opaque and
global supply chains, as well as fragmented data. This paper presents a
methodology that combines advances in LCA and publicly available databases with
knowledge-augmented AI techniques, including retrieval-augmented generation, to
estimate cradle-to-gate carbon footprints of food products. We introduce a
chatbot interface that allows users to interactively explore the carbon impact
of composite meals and relate the results to familiar activities. A live web
demonstration showcases our proof-of-concept system with arbitrary food items
and follow-up questions, highlighting both the potential and limitations - such
as database uncertainties and AI misinterpretations - of delivering LCA
insights in an accessible format.

### Hardware Architecture

### 1. [Optimizing Task Scheduling in Fog Computing with Deadline Awareness](http://arxiv.org/pdf/2509.07378v1)

Authors: Mohammad Sadegh Sirjani, Somayeh Sobati-Moghadam

The rise of Internet of Things (IoT) devices has led to the development of
numerous applications that require quick responses and low latency. Fog
computing has emerged as a solution for processing these IoT applications, but
it faces challenges such as resource allocation and job scheduling. Therefore,
it is crucial to determine how to assign and schedule tasks on Fog nodes. A
well-designed job scheduling algorithm can help decrease energy usage and
improve response times for application requests. This work aims to schedule
tasks in IoT while minimizing the total energy consumption of nodes and
enhancing the Quality of Service (QoS) requirements of IoT tasks, taking into
account task deadlines. Initially, this paper classifies the Fog nodes into two
categories based on their traffic level: low and high. It schedules
low-deadline tasks on low-traffic-level nodes using an Improved Golden Eagle
Optimization (IGEO) algorithm, an enhancement of the Golden Eagle Optimization
Algorithm that utilizes genetic operators for discretization. High-deadline
tasks are processed on high-traffic nodes using reinforcement learning (RL).
This combined approach is called the Reinforcement Improved Golden Eagle
Optimization (RIGEO) algorithm. Experimental results demonstrate that the
proposed algorithms optimize system response time, total deadline violation
time, and resource and system energy consumption compared to other
state-of-the-art algorithms.

### 2. [HYLU: Hybrid Parallel Sparse LU Factorization](http://arxiv.org/pdf/2509.07690v1)

Authors: Xiaoming Chen

This article introduces HYLU, a hybrid parallel LU factorization-based
general-purpose solver designed for efficiently solving sparse linear systems
(Ax=b) on multi-core shared-memory architectures. The key technical feature of
HYLU is the integration of hybrid numerical kernels so that it can adapt to
various sparsity patterns of coefficient matrices. Tests on 34 sparse matrices
from SuiteSparse Matrix Collection reveal that HYLU outperforms Intel MKL
PARDISO in the numerical factorization phase by geometric means of 1.74X (for
one-time solving) and 2.26X (for repeated solving). HYLU can be downloaded from
https://github.com/chenxm1986/hylu.

### Computational Complexity

### 1. [Verification power of rational-valued automata with deterministic and affine states](http://arxiv.org/pdf/2509.07857v1)

Authors: Zeyu Chen, Abuzer Yakaryılmaz, Junde Wu

We investigate the verification power of rational-valued affine automata
within Arthur--Merlin proof systems. For one-way verifiers, we give real-time
protocols with perfect completeness and tunable bounded error for two benchmark
nonregular languages, the balanced-middle language and the centered-palindrome
language, illustrating a concrete advantage over probabilistic and quantum
finite-state verifiers. For two-way verifiers, we first design a weak protocol
that verifies every Turing-recognizable language by streaming and checking a
configuration history. We then strengthen it with a probabilistic continuation
check that bounds the prover's transcript length and ensures halting with high
probability, yielding strong verification with expected running time
proportional to the product of the simulated machine's space and time (up to
input length and a factor polynomial in the inverse error parameter). Combining
these constructions with standard alternation--space correspondences, we place
alternating exponential time, equivalently deterministic exponential space,
inside affine Arthur--Merlin with two-way affine automata. We also present a
reduction-based route with perfect completeness via a Knapsack-game verifier,
which, together with linear-space reductions, yields that the class PSPACE
admits affine Arthur--Merlin verification by two-way affine automata. Two
simple primitives drive our protocols: a probabilistic continuation check to
control expected time and a restart-on-accept affine register that converts
exact algebraic checks into eventually halting bounded-error procedures.

### 2. [Existence and nonexistence of commutativity gadgets for entangled CSPs](http://arxiv.org/pdf/2509.07835v1)

Authors: Eric Culf, Josse van Dobben de Bruyn, Matthijs Vernooij, Peter Zeman

Commutativity gadgets allow NP-hardness proofs for classical constraint
satisfaction problems (CSPs) to be carried over to undecidability proofs for
the corresponding entangled CSPs. This has been done, for instance, for
NP-complete boolean CSPs and 3-colouring in the work of Culf and Mastel. For
many CSPs over larger alphabets, including $k$-colouring when $k \geq 4$, it is
not known whether or not commutativity gadgets exist, or if the entangled CSP
is decidable. In this paper, we study commutativity gadgets and prove the first
known obstruction to their existence. We do this by extending the definition of
the quantum automorphism group of a graph to the quantum endomorphism monoid of
a CSP, and showing that a CSP with non-classical quantum endomorphism monoid
does not admit a commutativity gadget. In particular, this shows that no
commutativity gadget exists for $k$-colouring when $k \geq 4$. However, we
construct a commutativity gadget for an alternate way of presenting
$k$-colouring as a nonlocal game, the oracular setting.
  Furthermore, we prove an easy to check sufficient condition for the quantum
endomorphism monoid to be non-classical, extending a result of Schmidt for the
quantum automorphism group of a graph, and use this to give examples of CSPs
that do not admit a commutativity gadget. We also show that existence of
oracular commutativity gadgets is preserved under categorical powers of graphs;
existence of commutativity gadgets and oracular commutativity gadgets is
equivalent for graphs with no four-cycle; and that the odd cycles and the odd
graphs have a commutative quantum endomorphism monoid, leaving open the
possibility that they might admit a commutativity gadget.

### Computational Engineering

### 1. [A Unified Data-Driven Framework for Efficient Scientific Discovery](http://arxiv.org/pdf/2509.07303v1)

Authors: Tingxiong Xiao, Xinxin Song, Ziqian Wang, Boyang Zhang, Jinli Suo

Scientific discovery drives progress across disciplines, from fundamental
physics to industrial applications. However, identifying physical laws
automatically from gathered datasets requires identifying the structure and
parameters of the formula underlying the data, which involves navigating a vast
search space and consuming substantial computational resources. To address
these issues, we build on the Buckingham $\Pi$ theorem and Taylor's theorem to
create a unified representation of diverse formulas, which introduces latent
variables to form a two-stage structure. To minimize the search space, we
initially focus on determining the structure of the latent formula, including
the relevant contributing inputs, the count of latent variables, and their
interconnections. Following this, the process of parameter identification is
expedited by enforcing dimensional constraints for physical relevance, favoring
simplicity in the formulas, and employing strategic optimization techniques.
Any overly complex outcomes are refined using symbolic regression for a compact
form. These general strategic techniques drastically reduce search iterations
from hundreds of millions to just tens, significantly enhancing the efficiency
of data-driven formula discovery. We performed comprehensive validation to
demonstrate FIND's effectiveness in discovering physical laws, dimensionless
numbers, partial differential equations, and uniform critical system parameters
across various fields, including astronomy, physics, chemistry, and
electronics. The excellent performances across 11 distinct datasets position
FIND as a powerful and versatile tool for advancing data-driven scientific
discovery in multiple domains.

### 2. [Uncertainty-Driven Hierarchical Sampling for Unbalanced Continual Malware Detection with Time-Series Update-Based Retrieval](http://arxiv.org/pdf/2509.07532v1)

Authors: Yi Xie, Ziyuan Yang, Yongqiang Huang, Yinyu Chen, Lei Zhang, Liang Liu, Yi Zhang

Android malware detection continues to face persistent challenges stemming
from long-term concept drift and class imbalance, as evolving malicious
behaviors and shifting usage patterns dynamically reshape feature
distributions. Although continual learning (CL) mitigates drift, existing
replay-based methods suffer from inherent bias. Specifically, their reliance on
classifier uncertainty for sample selection disproportionately prioritizes the
dominant benign class, causing overfitting and reduced generalization to
evolving malware. To address these limitations, we propose a novel
uncertainty-guided CL framework. First, we introduce a hierarchical balanced
sampler that employs a dual-phase uncertainty strategy to dynamically balance
benign and malicious samples while simultaneously selecting high-information,
high-uncertainty instances within each class. This mechanism ensures class
equilibrium across both replay and incremental data, thereby enhancing
adaptability to emerging threats. Second, we augment the framework with a
vector retrieval mechanism that exploits historical malware embeddings to
identify evolved variants via similarity-based retrieval, thereby complementing
classifier updates. Extensive experiments demonstrate that our framework
significantly outperforms state-of-the-art methods under strict low-label
conditions (50 labels per phase). It achieves a true positive rate (TPR) of
92.95\% and a mean accuracy (mACC) of 94.26\%, which validates its efficacy for
sustainable Android malware detection.

### 3. [LSMTCR: A Scalable Multi-Architecture Model for Epitope-Specific T Cell Receptor de novo Design](http://arxiv.org/pdf/2509.07627v1)

Authors: Ruihao Zhang, Xiao Liu

Designing full-length, epitope-specific TCR {\alpha}\b{eta} remains
challenging due to vast sequence space, data biases and incomplete modeling of
immunogenetic constraints. We present LSMTCR, a scalable multi-architecture
framework that separates specificity from constraint learning to enable de
novo, epitope-conditioned generation of paired, full-length TCRs. A
diffusion-enhanced BERT encoder learns time-conditioned epitope
representations; conditional GPT decoders, pretrained on CDR3\b{eta} and
transferred to CDR3{\alpha}, generate chain-specific CDR3s under cross-modal
conditioning with temperature-controlled diversity; and a gene-aware
Transformer assembles complete {\alpha}/\b{eta} sequences by predicting V/J
usage to ensure immunogenetic fidelity. Across GLIPH, TEP, MIRA, McPAS and our
curated dataset, LSMTCR achieves higher predicted binding than baselines on
most datasets, more faithfully recovers positional and length grammars, and
delivers superior, temperature-tunable diversity. For {\alpha}-chain
generation, transfer learning improves predicted binding, length realism and
diversity over representative methods. Full-length assembly from known or de
novo CDR3s preserves k-mer spectra, yields low edit distances to references,
and, in paired {\alpha}/\b{eta} co-modelling with epitope, attains higher
pTM/ipTM than single-chain settings. LSMTCR outputs diverse,
gene-contextualized, full-length TCR designs from epitope input alone, enabling
high-throughput screening and iterative optimization.

### 4. [Generalized eigenvalue stabilization for immersed explicit dynamics](http://arxiv.org/pdf/2509.07632v1)

Authors: Tim Bürchner, Lars Radtke, Sascha Eisenträger, Alexander Düster, Ernst Rank, Stefan Kollmannsberger, Philipp Kopp

Explicit time integration for immersed finite element discretizations
severely suffers from the influence of poorly cut elements. In this
contribution, we propose a generalized eigenvalue stabilization (GEVS) strategy
for the element mass matrices of cut elements to cure their adverse impact on
the critical time step size of the global system. We use spectral basis
functions, specifically $C^0$ continuous Lagrangian interpolation polynomials
defined on Gauss-Lobatto-Legendre (GLL) points, which, in combination with its
associated GLL quadrature rule, yield high-order convergent diagonal mass
matrices for uncut elements. Moreover, considering cut elements, we combine the
proposed GEVS approach with the finite cell method (FCM) to guarantee
definiteness of the system matrices. However, the proposed GEVS stabilization
can directly be applied to other immersed boundary finite element methods.
Numerical experiments demonstrate that the stabilization strategy achieves
optimal convergence rates and recovers critical time step sizes of equivalent
boundary-conforming discretizations. This also holds in the presence of weakly
enforced Dirichlet boundary conditions using either Nitsche's method or penalty
formulations.

### Computational Geometry

### 1. [DiGS: Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning](http://arxiv.org/pdf/2509.07493v1)

Authors: Wenzhi Guo, Bing Wang

3D Gaussian Splatting (3DGS) has recently emerged as a powerful paradigm for
photorealistic view synthesis, representing scenes with spatially distributed
Gaussian primitives. While highly effective for rendering, achieving accurate
and complete surface reconstruction remains challenging due to the unstructured
nature of the representation and the absence of explicit geometric supervision.
In this work, we propose DiGS, a unified framework that embeds Signed Distance
Field (SDF) learning directly into the 3DGS pipeline, thereby enforcing strong
and interpretable surface priors. By associating each Gaussian with a learnable
SDF value, DiGS explicitly aligns primitives with underlying geometry and
improves cross-view consistency. To further ensure dense and coherent coverage,
we design a geometry-guided grid growth strategy that adaptively distributes
Gaussians along geometry-consistent regions under a multi-scale hierarchy.
Extensive experiments on standard benchmarks, including DTU, Mip-NeRF 360, and
Tanks& Temples, demonstrate that DiGS consistently improves reconstruction
accuracy and completeness while retaining high rendering fidelity.

### 2. [Undecidability of Tiling with a Tromino](http://arxiv.org/pdf/2509.07906v1)

Authors: MIT-ULB CompGeom Group, :, Zachary Abel, Hugo Akitaya, Lily Chung, Erik D. Demaine, Jenny Diomidova, Della Hendrickson, Stefan Langerman, Jayson Lynch

Given a periodic placement of copies of a tromino (either L or I), we prove
co-RE-completeness (and hence undecidability) of deciding whether it can be
completed to a plane tiling. By contrast, the problem becomes decidable if the
initial placement is finite, or if the tile is a domino instead of a tromino
(in any dimension). As a consequence, tiling a given periodic subset of the
plane with a given tromino (L or I) is co-RE-complete.
  We also prove co-RE-completeness of tiling the entire plane with two
polyominoes (one of which is disconnected and the other of which has constant
size), and of tiling 3D space with two connected polycubes (one of which has
constant size). If we restrict to tiling by translation only (no rotation),
then we obtain co-RE-completeness with one more tile: two trominoes for a
periodic subset of 2D, three polyominoes for the 2D plane, and three connected
polycubes for 3D space.
  Along the way, we prove several new complexity and algorithmic results about
periodic (infinite) graphs. Notably, we prove that Periodic Planar
(1-in-)3SAT-3, 3DM, and Graph Orientation are co-RE-complete in 2D and
PSPACE-complete in 1D; we extend basic results in graph drawing to 2D periodic
graphs; and we give a polynomial-time algorithm for perfect matching in
bipartite periodic graphs.

### Computation and Language

### 1. [PersonaFuse: A Personality Activation-Driven Framework for Enhancing Human-LLM Interactions](http://arxiv.org/pdf/2509.07370v1)

Authors: Yixuan Tang, Yi Yang, Ahmed Abbasi

Recent advancements in Large Language Models (LLMs) demonstrate remarkable
capabilities across various fields. These developments have led to more direct
communication between humans and LLMs in various situations, such as social
companionship and psychological support. However, LLMs often exhibit
limitations in emotional perception and social competence during real-world
conversations. These limitations partly originate from their inability to adapt
their communication style and emotional expression to different social and task
contexts. In this work, we introduce PersonaFuse, a novel LLM post-training
framework that enables LLMs to adapt and express different personalities for
varying situations. Inspired by Trait Activation Theory and the Big Five
personality model, PersonaFuse employs a Mixture-of-Expert architecture that
combines persona adapters with a dynamic routing network, enabling contextual
trait expression. Experimental results show that PersonaFuse substantially
outperforms baseline models across multiple dimensions of social-emotional
intelligence. Importantly, these gains are achieved without sacrificing general
reasoning ability or model safety, which remain common limitations of direct
prompting and supervised fine-tuning approaches. PersonaFuse also delivers
consistent improvements in downstream human-centered applications, such as
mental health counseling and review-based customer service. Finally, human
preference evaluations against leading LLMs, including GPT-4o and DeepSeek,
demonstrate that PersonaFuse achieves competitive response quality despite its
comparatively smaller model size. These findings demonstrate that
PersonaFuse~offers a theoretically grounded and practical approach for
developing social-emotional enhanced LLMs, marking a significant advancement
toward more human-centric AI systems.

### 2. [The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](http://arxiv.org/pdf/2509.07399v1)

Authors: Yi-Jie Cheng, Oscar Chew, Yun-Nung Chen

Integrating knowledge graphs (KGs) into the reasoning processes of large
language models (LLMs) has emerged as a promising approach to mitigate
hallucination. However, existing work in this area often relies on proprietary
or extremely large models, limiting accessibility and scalability. In this
study, we investigate the capabilities of existing integration methods for
small language models (SLMs) in KG-based question answering and observe that
their performance is often constrained by their limited ability to traverse and
reason over knowledge graphs. To address this limitation, we propose leveraging
simple and efficient exploration modules to handle knowledge graph traversal in
place of the language model itself. Experiment results demonstrate that these
lightweight modules effectively improve the performance of small language
models on knowledge graph question answering tasks. Source code:
https://github.com/yijie-cheng/SLM-ToG/.

### 3. [LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction](http://arxiv.org/pdf/2509.07403v1)

Authors: Weichu Liu, Jing Xiong, Yuxuan Hu, Zixuan Li, Minghuan Tan, Ningning Mao, Chenyang Zhao, Zhongwei Wan, Chaofan Tao, Wendong Xu, Hui Shen, Chengming Li, Lingpeng Kong, Ngai Wong

Large language models (LLMs) make significant progress in Emotional
Intelligence (EI) and long-context understanding. However, existing benchmarks
tend to overlook certain aspects of EI in long-context scenarios, especially
under realistic, practical settings where interactions are lengthy, diverse,
and often noisy. To move towards such realistic settings, we present
LongEmotion, a benchmark specifically designed for long-context EI tasks. It
covers a diverse set of tasks, including Emotion Classification, Emotion
Detection, Emotion QA, Emotion Conversation, Emotion Summary, and Emotion
Expression. On average, the input length for these tasks reaches 8,777 tokens,
with long-form generation required for Emotion Expression. To enhance
performance under realistic constraints, we incorporate Retrieval-Augmented
Generation (RAG) and Collaborative Emotional Modeling (CoEM), and compare them
with standard prompt-based methods. Unlike conventional approaches, our RAG
method leverages both the conversation context and the large language model
itself as retrieval sources, avoiding reliance on external knowledge bases. The
CoEM method further improves performance by decomposing the task into five
stages, integrating both retrieval augmentation and limited knowledge
injection. Experimental results show that both RAG and CoEM consistently
enhance EI-related performance across most long-context tasks, advancing LLMs
toward more practical and real-world EI applications. Furthermore, we conducted
a comparative case study experiment on the GPT series to demonstrate the
differences among various models in terms of EI. Code is available on GitHub at
https://github.com/LongEmotion/LongEmotion, and the project page can be found
at https://longemotion.github.io/.

### 4. [AIxcellent Vibes at GermEval 2025 Shared Task on Candy Speech Detection: Improving Model Performance by Span-Level Training](http://arxiv.org/pdf/2509.07459v1)

Authors: Christian Rene Thelen, Patrick Gustav Blaneck, Tobias Bornheim, Niklas Grieger, Stephan Bialonski

Positive, supportive online communication in social media (candy speech) has
the potential to foster civility, yet automated detection of such language
remains underexplored, limiting systematic analysis of its impact. We
investigate how candy speech can be reliably detected in a 46k-comment German
YouTube corpus by monolingual and multilingual language models, including
GBERT, Qwen3 Embedding, and XLM-RoBERTa. We find that a multilingual
XLM-RoBERTa-Large model trained to detect candy speech at the span level
outperforms other approaches, ranking first in both binary positive F1: 0.8906)
and categorized span-based detection (strict F1: 0.6307) subtasks at the
GermEval 2025 Shared Task on Candy Speech Detection. We speculate that
span-based training, multilingual capabilities, and emoji-aware tokenizers
improved detection performance. Our results demonstrate the effectiveness of
multilingual models in identifying positive, supportive language.

### 5. [Understanding Stigmatizing Language Lexicons: A Comparative Analysis in Clinical Contexts](http://arxiv.org/pdf/2509.07462v1)

Authors: Yiliang Zhou, Di Hu, Tianchu Lyu, Jasmine Dhillon, Alexandra L. Beck, Gelareh Sadigh, Kai Zheng

Stigmatizing language results in healthcare inequities, yet there is no
universally accepted or standardized lexicon defining which words, terms, or
phrases constitute stigmatizing language in healthcare. We conducted a
systematic search of the literature to identify existing stigmatizing language
lexicons and then analyzed them comparatively to examine: 1) similarities and
discrepancies between these lexicons, and 2) the distribution of positive,
negative, or neutral terms based on an established sentiment dataset. Our
search identified four lexicons. The analysis results revealed moderate
semantic similarity among them, and that most stigmatizing terms are related to
judgmental expressions by clinicians to describe perceived negative behaviors.
Sentiment analysis showed a predominant proportion of negatively classified
terms, though variations exist across lexicons. Our findings underscore the
need for a standardized lexicon and highlight challenges in defining
stigmatizing language in clinical texts.

### 6. [From Scarcity to Efficiency: Investigating the Effects of Data Augmentation on African Machine Translation](http://arxiv.org/pdf/2509.07471v1)

Authors: Mardiyyah Oduwole, Oluwatosin Olajide, Jamiu Suleiman, Faith Hunja, Busayo Awobade, Fatimo Adebanjo, Comfort Akanni, Chinonyelum Igwe, Peace Ododo, Promise Omoigui, Steven Kolawole, Abraham Owodunni

The linguistic diversity across the African continent presents different
challenges and opportunities for machine translation. This study explores the
effects of data augmentation techniques in improving translation systems in
low-resource African languages. We focus on two data augmentation techniques:
sentence concatenation with back translation and switch-out, applying them
across six African languages. Our experiments show significant improvements in
machine translation performance, with a minimum increase of 25\% in BLEU score
across all six languages.We provide a comprehensive analysis and highlight the
potential of these techniques to improve machine translation systems for
low-resource languages, contributing to the development of more robust
translation systems for under-resourced languages.

### 7. [VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents](http://arxiv.org/pdf/2509.07553v1)

Authors: Zheng Wu, Heyuan Huang, Xingyu Lou, Xiangmou Qu, Pengzhou Cheng, Zongru Wu, Weiwen Liu, Weinan Zhang, Jun Wang, Zhaoxiang Wang, Zhuosheng Zhang

With the rapid progress of multimodal large language models, operating system
(OS) agents become increasingly capable of automating tasks through on-device
graphical user interfaces (GUIs). However, most existing OS agents are designed
for idealized settings, whereas real-world environments often present
untrustworthy conditions. To mitigate risks of over-execution in such
scenarios, we propose a query-driven human-agent-GUI interaction framework that
enables OS agents to decide when to query humans for more reliable task
completion. Built upon this framework, we introduce VeriOS-Agent, a trustworthy
OS agent trained with a two-stage learning paradigm that falicitate the
decoupling and utilization of meta-knowledge. Concretely, VeriOS-Agent
autonomously executes actions in normal conditions while proactively querying
humans in untrustworthy scenarios. Experiments show that VeriOS-Agent improves
the average step-wise success rate by 20.64\% in untrustworthy scenarios over
the state-of-the-art, without compromising normal performance. Analysis
highlights VeriOS-Agent's rationality, generalizability, and scalability. The
codes, datasets and models are available at
https://github.com/Wuzheng02/VeriOS.

### 8. [MaLei at MultiClinSUM: Summarisation of Clinical Documents using Perspective-Aware Iterative Self-Prompting with LLMs](http://arxiv.org/pdf/2509.07622v1)

Authors: Libo Ren, Yee Man Ng, Lifeng Han

Efficient communication between patients and clinicians plays an important
role in shared decision-making. However, clinical reports are often lengthy and
filled with clinical jargon, making it difficult for domain experts to identify
important aspects in the document efficiently. This paper presents the
methodology we applied in the MultiClinSUM shared task for summarising clinical
case documents. We used an Iterative Self-Prompting technique on large language
models (LLMs) by asking LLMs to generate task-specific prompts and refine them
via example-based few-shot learning. Furthermore, we used lexical and embedding
space metrics, ROUGE and BERT-score, to guide the model fine-tuning with
epochs. Our submission using perspective-aware ISP on GPT-4 and GPT-4o achieved
ROUGE scores (46.53, 24.68, 30.77) and BERTscores (87.84, 83.25, 85.46) for (P,
R, F1) from the official evaluation on 3,396 clinical case reports from various
specialties extracted from open journals. The high BERTscore indicates that the
model produced semantically equivalent output summaries compared to the
references, even though the overlap at the exact lexicon level is lower, as
reflected in the lower ROUGE scores. This work sheds some light on how
perspective-aware ISP (PA-ISP) can be deployed for clinical report
summarisation and support better communication between patients and clinicians.

### 9. [M-BRe: Discovering Training Samples for Relation Extraction from Unlabeled Texts with Large Language Models](http://arxiv.org/pdf/2509.07730v1)

Authors: Zexuan Li, Hongliang Dai, Piji Li

For Relation Extraction (RE), the manual annotation of training data may be
prohibitively expensive, since the sentences that contain the target relations
in texts can be very scarce and difficult to find. It is therefore beneficial
to develop an efficient method that can automatically extract training
instances from unlabeled texts for training RE models. Recently, large language
models (LLMs) have been adopted in various natural language processing tasks,
with RE also benefiting from their advances. However, when leveraging LLMs for
RE with predefined relation categories, two key challenges arise. First, in a
multi-class classification setting, LLMs often struggle to comprehensively
capture the semantics of every relation, leading to suboptimal results. Second,
although employing binary classification for each relation individually can
mitigate this issue, it introduces significant computational overhead,
resulting in impractical time complexity for real-world applications.
Therefore, this paper proposes a framework called M-BRe to extract training
instances from unlabeled texts for RE. It utilizes three modules to combine the
advantages of both of the above classification approaches: Relation Grouping,
Relation Extraction, and Label Decision. Extensive experiments confirm its
superior capability in discovering high-quality training samples from unlabeled
texts for RE.

### 10. [From Detection to Mitigation: Addressing Gender Bias in Chinese Texts via Efficient Tuning and Voting-Based Rebalancing](http://arxiv.org/pdf/2509.07889v1)

Authors: Chengyan Wu, Yiqiang Cai, Yufei Cheng, Yun Xue

This paper presents our team's solution to Shared Task 7 of NLPCC-2025, which
focuses on sentence-level gender bias detection and mitigation in Chinese. The
task aims to promote fairness and controllability in natural language
generation by automatically detecting, classifying, and mitigating gender bias.
To address this challenge, we adopt a fine-tuning approach based on large
language models (LLMs), efficiently adapt to the bias detection task via
Low-Rank Adaptation (LoRA). In terms of data processing, we construct a more
balanced training set to alleviate class imbalance and introduce heterogeneous
samples from multiple sources to enhance model generalization. For the
detection and classification sub-tasks, we employ a majority voting strategy
that integrates outputs from multiple expert models to boost performance.
Additionally, to improve bias generation detection and mitigation, we design a
multi-temperature sampling mechanism to capture potential variations in bias
expression styles. Experimental results demonstrate the effectiveness of our
approach in bias detection, classification, and mitigation. Our method
ultimately achieves an average score of 47.90%, ranking fourth in the shared
task.

### Cryptography and Security

### 1. [A Decade-long Landscape of Advanced Persistent Threats: Longitudinal Analysis and Global Trends](http://arxiv.org/pdf/2509.07457v1)

Authors: Shakhzod Yuldoshkhujaev, Mijin Jeon, Doowon Kim, Nick Nikiforakis, Hyungjoon Koo

An advanced persistent threat (APT) refers to a covert, long-term
cyberattack, typically conducted by state-sponsored actors, targeting critical
sectors and often remaining undetected for long periods. In response,
collective intelligence from around the globe collaborates to identify and
trace surreptitious activities, generating substantial documentation on APT
campaigns publicly available on the web. While prior works predominantly focus
on specific aspects of APT cases, such as detection, evaluation, cyber threat
intelligence, and dataset creation, limited attention has been devoted to
revisiting and investigating these scattered dossiers in a longitudinal manner.
The objective of our study is to fill the gap by offering a macro perspective,
connecting key insights and global trends in past APT attacks. We
systematically analyze six reliable sources-three focused on technical reports
and another three on threat actors-examining 1,509 APT dossiers (24,215 pages)
spanning 2014-2023, and identifying 603 unique APT groups worldwide. To
efficiently unearth relevant information, we employ a hybrid methodology that
combines rule-based information retrieval with large-language-model-based
search techniques. Our longitudinal analysis reveals shifts in threat actor
activities, global attack vectors, changes in targeted sectors, and
relationships between cyberattacks and significant events such as elections or
wars, which provide insights into historical patterns in APT evolution. Over
the past decade, 154 countries have been affected, primarily using malicious
documents and spear phishing as dominant initial infiltration vectors, with a
noticeable decline in zero-day exploitation since 2016. Furthermore, we present
our findings through interactive visualization tools, such as an APT map or
flow diagram, to facilitate intuitive understanding of global patterns and
trends in APT activities.

### 2. [Biometric Bound Credentials for Age Verification](http://arxiv.org/pdf/2509.07465v1)

Authors: Norman Poh, Daryl Burns

Age verification is increasingly critical for regulatory compliance, user
trust, and the protection of minors online. Historically, solutions have
struggled with poor accuracy, intrusiveness, and significant security risks.
More recently, concerns have shifted toward privacy, surveillance, fairness,
and the need for transparent, trustworthy systems. In this paper, we propose
Biometric Bound Credentials (BBCreds) as a privacy-preserving approach that
cryptographically binds age credentials to an individual's biometric features
without storing biometric templates. This ensures only the legitimate,
physically present user can access age-restricted services, prevents credential
sharing, and addresses both legacy and emerging challenges in age verification.
enhances privacy.

### 3. [Backdoor Attacks and Defenses in Computer Vision Domain: A Survey](http://arxiv.org/pdf/2509.07504v1)

Authors: Bilal Hussain Abbasi, Yanjun Zhang, Leo Zhang, Shang Gao

Backdoor (trojan) attacks embed hidden, controllable behaviors into
machine-learning models so that models behave normally on benign inputs but
produce attacker-chosen outputs when a trigger is present. This survey reviews
the rapidly growing literature on backdoor attacks and defenses in the
computer-vision domain. We introduce a multi-dimensional taxonomy that
organizes attacks and defenses by injection stage (dataset poisoning,
model/parameter modification, inference-time injection), trigger type (patch,
blended/frequency, semantic, transformation), labeling strategy (dirty-label
vs. clean-label / feature-collision), representation stage (instance-specific,
manifold/class-level, neuron/parameter hijacking, distributed encodings), and
target task (classification, detection, segmentation, video, multimodal). For
each axis we summarize representative methods, highlight evaluation practices,
and discuss where defenses succeed or fail. For example, many classical
sanitization and reverse-engineering tools are effective against reusable patch
attacks but struggle with input-aware, sample-specific, or parameter-space
backdoors and with transfer via compromised pre-trained encoders or hardware
bit-flips. We synthesize trends, identify persistent gaps (supply-chain and
hardware threats, certifiable defenses, cross-task benchmarks), and propose
practical guidelines for threat-aware evaluation and layered defenses. This
survey aims to orient researchers and practitioners to the current threat
landscape and pressing research directions in secure computer vision.

### 4. [Extension of Spatial k-Anonymity: New Metrics for Assessing the Anonymity of Geomasked Data Considering Realistic Attack Scenarios](http://arxiv.org/pdf/2509.07505v1)

Authors: Simon Cremer, Lydia Jehmlich, Rainer Lenz

Spatial data are gaining increasing importance in many areas of research.
Particularly spatial health data are becoming increasingly important for
medical research, for example, to better understand relationships between
environmental factors and disease patterns. However, their use is often
restricted by legal data protection regulations, since georeferenced personal
information carries a high risk of re-identification of individuals. To address
this issue, what are called geomasking methods are applied to guarantee data
protection through targeted displacement of individual data points, while
simultaneously maintaining analytical validity within a tolerable range. In the
current literature the degree of anonymity of such anonymized georeferenced
datasets is often measured by the so-called metric of spatial k-anonymity.
However, this metric has considerable shortcomings, particularly regarding its
resilience against realistic data attack scenarios. This article classifies the
potential data attack scenarios in the context of anonymized georeferenced
microdata and introduces appropriate metrics that enable a comprehensive
assessment of anonymity adapted to potential data attack scenarios.

### 5. [Enhanced cast-128 with adaptive s-box optimization via neural networks for image protection](http://arxiv.org/pdf/2509.07606v1)

Authors: Fadhil Abbas Fadhil, Maryam Mahdi Alhusseini, Mohammad-Reza Feizi-Derakhshi

An improved CAST-128 encryption algorithm, which is done by implementing
chaos-based adaptive S-box generation using Logistic sine Map (LSM), has been
provided in this paper because of the increasing requirements of efficient and
smart image encryption mechanisms. The study aims to address the drawbacks of
static S-box models commonly used in traditional cryptographic systems, which
are susceptible to linear and differential attacks. In the proposed scheme, the
dynamic, non-linear, invertible, and highly cryptographic strength S-boxes are
generated through a hybrid chaotic system that may have high non-linearity,
strong and rigorous avalanche characteristics, and low differential uniformity.
The process here is that the LSM is used to produce S-boxes having
key-dependent parameters that are stuffed into the CAST-128 structure to
encrypt the image in a block-wise manner. The performance of the encryption is
assessed utilizing a set of standard grayscale images. The metrics that are
used to evaluate the security are entropy, NPCR, UACI, PSNR, and histogram
analysis. Outcomes indicate that randomness, resistance to statistical attacks,
and country of encryption are significantly improved compared to the original
CAST-128. The study is theoretically and practically relevant since it presents
a lightweight S-box generation approach driven by chaos, which can increase the
level of robustness of the image encryptions without enlisting machine
learning. The system may be applied to secure communications, surveillance
systems, and medical image protection on a real-time basis.

### 6. [FlexEmu: Towards Flexible MCU Peripheral Emulation (Extended Version)](http://arxiv.org/pdf/2509.07615v1)

Authors: Chongqing Lei, Zhen Ling, Xiangyu Xu, Shaofeng Li, Guangchi Liu, Kai Dong, Junzhou Luo

Microcontroller units (MCUs) are widely used in embedded devices due to their
low power consumption and cost-effectiveness. MCU firmware controls these
devices and is vital to the security of embedded systems. However, performing
dynamic security analyses for MCU firmware has remained challenging due to the
lack of usable execution environments -- existing dynamic analyses cannot run
on physical devices (e.g., insufficient computational resources), while
building emulators is costly due to the massive amount of heterogeneous
hardware, especially peripherals.
  Our work is based on the insight that MCU peripherals can be modeled in a
two-fold manner. At the structural level, peripherals have diverse
implementations but we can use a limited set of primitives to abstract
peripherals because their hardware implementations are based on common hardware
concepts. At the semantic level, peripherals have diverse functionalities.
However, we can use a single unified semantic model to describe the same kind
of peripherals because they exhibit similar functionalities. Building on this,
we propose FlexEmu, a flexible MCU peripheral emulation framework. Once
semantic models are created, FlexEmu automatically extracts peripheral-specific
details to instantiate models and generate emulators accordingly. We have
successfully applied FlexEmu to model 12 kinds of MCU peripherals. Our
evaluation on 90 firmware samples across 15 different MCU platforms shows that
the automatically generated emulators can faithfully replicate hardware
behaviors and achieve a 98.48% unit test passing rate, outperforming
state-of-the-art approaches. To demonstrate the implications of FlexEmu on
firmware security, we use the generated emulators to fuzz three popular RTOSes
and uncover 10 previously unknown bugs.

### 7. [Embedded Off-Switches for AI Compute](http://arxiv.org/pdf/2509.07637v1)

Authors: James Petrie

To address the risks of increasingly capable AI systems, we introduce a
hardware-level off-switch that embeds thousands of independent "security
blocks" in each AI accelerator. This massively redundant architecture is
designed to prevent unauthorized chip use, even against sophisticated physical
attacks. Our main security block design uses public key cryptography to check
the authenticity of authorization licenses, and randomly generated nonces to
prevent replay attacks. We evaluate attack vectors and present additional
security block variants that could be added for greater robustness. Security
blocks can be built with standard circuit components, ensuring compatibility
with existing semiconductor manufacturing processes. With embedded security
blocks, the next generation of AI accelerators could be more robustly defended
against dangerous misuse.

### 8. [Empirical Security Analysis of Software-based Fault Isolation through Controlled Fault Injection](http://arxiv.org/pdf/2509.07757v1)

Authors: Nils Bars, Lukas Bernhard, Moritz Schloegel, Thorsten Holz

We use browsers daily to access all sorts of information. Because browsers
routinely process scripts, media, and executable code from unknown sources,
they form a critical security boundary between users and adversaries. A common
attack vector is JavaScript, which exposes a large attack surface due to the
sheer complexity of modern JavaScript engines. To mitigate these threats,
modern engines increasingly adopt software-based fault isolation (SFI). A
prominent example is Google's V8 heap sandbox, which represents the most widely
deployed SFI mechanism, protecting billions of users across all Chromium-based
browsers and countless applications built on Node.js and Electron. The heap
sandbox splits the address space into two parts: one part containing trusted,
security-sensitive metadata, and a sandboxed heap containing memory accessible
to untrusted code. On a technical level, the sandbox enforces isolation by
removing raw pointers and using translation tables to resolve references to
trusted objects. Consequently, an attacker cannot corrupt trusted data even
with full control of the sandboxed data, unless there is a bug in how code
handles data from the sandboxed heap. Despite their widespread use, such SFI
mechanisms have seen little security testing.
  In this work, we propose a new testing technique that models the security
boundary of modern SFI implementations. Following the SFI threat model, we
assume a powerful attacker who fully controls the sandbox's memory. We
implement this by instrumenting memory loads originating in the trusted domain
and accessing untrusted, attacker-controlled sandbox memory. We then inject
faults into the loaded data, aiming to trigger memory corruption in the trusted
domain. In a comprehensive evaluation, we identify 19 security bugs in V8 that
enable an attacker to bypass the sandbox.

### 9. [AgentSentinel: An End-to-End and Real-Time Security Defense Framework for Computer-Use Agents](http://arxiv.org/pdf/2509.07764v1)

Authors: Haitao Hu, Peng Chen, Yanpeng Zhao, Yuqi Chen

Large Language Models (LLMs) have been increasingly integrated into
computer-use agents, which can autonomously operate tools on a user's computer
to accomplish complex tasks. However, due to the inherently unstable and
unpredictable nature of LLM outputs, they may issue unintended tool commands or
incorrect inputs, leading to potentially harmful operations. Unlike traditional
security risks stemming from insecure user prompts, tool execution results from
LLM-driven decisions introduce new and unique security challenges. These
vulnerabilities span across all components of a computer-use agent. To mitigate
these risks, we propose AgentSentinel, an end-to-end, real-time defense
framework designed to mitigate potential security threats on a user's computer.
AgentSentinel intercepts all sensitive operations within agent-related services
and halts execution until a comprehensive security audit is completed. Our
security auditing mechanism introduces a novel inspection process that
correlates the current task context with system traces generated during task
execution. To thoroughly evaluate AgentSentinel, we present BadComputerUse, a
benchmark consisting of 60 diverse attack scenarios across six attack
categories. The benchmark demonstrates a 87% average attack success rate on
four state-of-the-art LLMs. Our evaluation shows that AgentSentinel achieves an
average defense success rate of 79.6%, significantly outperforming all baseline
defenses.

### 10. [Inner-product Functional Encryption with Fine-grained Revocation for Flexible EHR Sharing](http://arxiv.org/pdf/2509.07804v1)

Authors: Yue Han, Jinguang Han, Liqun Chen, Chao Sun

E-health record (EHR) contains a vast amount of continuously growing medical
data and enables medical institutions to access patient health data
conveniently.This provides opportunities for medical data mining which has
important applications in identifying high-risk patients and improving disease
diagnosis, etc.Since EHR contains sensitive patient information, how to protect
patient privacy and enable mining on EHR data is important and
challenging.Traditional public key encryption (PKE) can protect patient
privacy, but cannot support flexible selective computation on encrypted EHR
data.Functional encryption (FE) allows authorised users to compute function
values of encrypted data without releasing other information, hence supporting
selective computation on encrypted data. Nevertheless, existing FE schemes do
not support fine-grained revocation and update, so they are unsuitable for EHR
system. In this paper,we first propose an inner-product functional encryption
with fine-grained revocation (IPFE-FR) scheme, and then apply it to a flexible
EHR sharing system. Our scheme possesses the following features:(1) a group
manager can revoke a specific function computation of medical institutions on
encrypted EHR data,instead of all function computation rights. (2) a revoked
medical institution is not allowed to compute the function value of encrypted
EHR data not only generated after the revocation, but also generated before the
revocation. (3) secret keys issued to the same medical institution are bound
together to prevent collusion attacks. The formal definition and security model
of the IPFE-FR scheme are proposed.Furthermore, we present a concrete
construction and reduce its security to the Learning with Errors (LWE)
assumption which is quantum-resistant. Finally, the theoretical analysis and
experimental implementation of our scheme are conducted to show its efficiency.

### Computer Vision and Pattern Recognition

### 1. [G3CN: Gaussian Topology Refinement Gated Graph Convolutional Network for Skeleton-Based Action Recognition](http://arxiv.org/pdf/2509.07335v1)

Authors: Haiqing Ren, Zhongkai Luo, Heng Fan, Xiaohui Yuan, Guanchen Wang, Libo Zhang

Graph Convolutional Networks (GCNs) have proven to be highly effective for
skeleton-based action recognition, primarily due to their ability to leverage
graph topology for feature aggregation, a key factor in extracting meaningful
representations. However, despite their success, GCNs often struggle to
effectively distinguish between ambiguous actions, revealing limitations in the
representation of learned topological and spatial features. To address this
challenge, we propose a novel approach, Gaussian Topology Refinement Gated
Graph Convolution (G$^{3}$CN), to address the challenge of distinguishing
ambiguous actions in skeleton-based action recognition. G$^{3}$CN incorporates
a Gaussian filter to refine the skeleton topology graph, improving the
representation of ambiguous actions. Additionally, Gated Recurrent Units (GRUs)
are integrated into the GCN framework to enhance information propagation
between skeleton points. Our method shows strong generalization across various
GCN backbones. Extensive experiments on NTU RGB+D, NTU RGB+D 120, and NW-UCLA
benchmarks demonstrate that G$^{3}$CN effectively improves action recognition,
particularly for ambiguous samples.

### 2. [Parse Graph-Based Visual-Language Interaction for Human Pose Estimation](http://arxiv.org/pdf/2509.07385v1)

Authors: Shibang Liu, Xuemei Xie, Guangming Shi

Parse graphs boost human pose estimation (HPE) by integrating context and
hierarchies, yet prior work mostly focuses on single modality modeling,
ignoring the potential of multimodal fusion. Notably, language offers rich HPE
priors like spatial relations for occluded scenes, but existing visual-language
fusion via global feature integration weakens occluded region responses and
causes alignment and location failures. To address this issue, we propose Parse
Graph-based Visual-Language interaction (PGVL) with a core novel Guided Module
(GM). In PGVL, low-level nodes focus on local features, maximizing the
maintenance of responses in occluded areas and high-level nodes integrate
global features to infer occluded or invisible parts. GM enables high semantic
nodes to guide the feature update of low semantic nodes that have undergone
cross attention. It ensuring effective fusion of diverse information. PGVL
includes top-down decomposition and bottom-up composition. In the first stage,
modality specific parse graphs are constructed. Next stage. recursive
bidirectional cross-attention is used, purified by GM. We also design network
based on PGVL. The PGVL and our network is validated on major pose estimation
datasets. We will release the code soon.

### 3. [DreamLifting: A Plug-in Module Lifting MV Diffusion Models for 3D Asset Generation](http://arxiv.org/pdf/2509.07435v1)

Authors: Ze-Xin Yin, Jiaxiong Qiu, Liu Liu, Xinjie Wang, Wei Sui, Zhizhong Su, Jian Yang, Jin Xie

The labor- and experience-intensive creation of 3D assets with physically
based rendering (PBR) materials demands an autonomous 3D asset creation
pipeline. However, most existing 3D generation methods focus on geometry
modeling, either baking textures into simple vertex colors or leaving texture
synthesis to post-processing with image diffusion models. To achieve end-to-end
PBR-ready 3D asset generation, we present Lightweight Gaussian Asset Adapter
(LGAA), a novel framework that unifies the modeling of geometry and PBR
materials by exploiting multi-view (MV) diffusion priors from a novel
perspective. The LGAA features a modular design with three components.
Specifically, the LGAA Wrapper reuses and adapts network layers from MV
diffusion models, which encapsulate knowledge acquired from billions of images,
enabling better convergence in a data-efficient manner. To incorporate multiple
diffusion priors for geometry and PBR synthesis, the LGAA Switcher aligns
multiple LGAA Wrapper layers encapsulating different knowledge. Then, a tamed
variational autoencoder (VAE), termed LGAA Decoder, is designed to predict 2D
Gaussian Splatting (2DGS) with PBR channels. Finally, we introduce a dedicated
post-processing procedure to effectively extract high-quality, relightable mesh
assets from the resulting 2DGS. Extensive quantitative and qualitative
experiments demonstrate the superior performance of LGAA with both text-and
image-conditioned MV diffusion models. Additionally, the modular design enables
flexible incorporation of multiple diffusion priors, and the
knowledge-preserving scheme leads to efficient convergence trained on merely
69k multi-view instances. Our code, pre-trained weights, and the dataset used
will be publicly available via our project page:
https://zx-yin.github.io/dreamlifting/.

### 4. [In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting](http://arxiv.org/pdf/2509.07447v1)

Authors: Taiying Peng, Jiacheng Hua, Miao Liu, Feng Lu

The emergence of advanced multimodal large language models (MLLMs) has
significantly enhanced AI assistants' ability to process complex information
across modalities. Recently, egocentric videos, by directly capturing user
focus, actions, and context in an unified coordinate, offer an exciting
opportunity to enable proactive and personalized AI user experiences with
MLLMs. However, existing benchmarks overlook the crucial role of gaze as an
indicator of user intent. To address this gap, we introduce EgoGazeVQA, an
egocentric gaze-guided video question answering benchmark that leverages gaze
information to improve the understanding of longer daily-life videos.
EgoGazeVQA consists of gaze-based QA pairs generated by MLLMs and refined by
human annotators. Our experiments reveal that existing MLLMs struggle to
accurately interpret user intentions. In contrast, our gaze-guided intent
prompting methods significantly enhance performance by integrating spatial,
temporal, and intent-related cues. We further conduct experiments on
gaze-related fine-tuning and analyze how gaze estimation accuracy impacts
prompting effectiveness. These results underscore the value of gaze for more
personalized and effective AI assistants in egocentric settings.

### 5. [XOCT: Enhancing OCT to OCTA Translation via Cross-Dimensional Supervised Multi-Scale Feature Learning](http://arxiv.org/pdf/2509.07455v1)

Authors: Pooya Khosravi, Kun Han, Anthony T. Wu, Arghavan Rezvani, Zexin Feng, Xiaohui Xie

Optical Coherence Tomography Angiography (OCTA) and its derived en-face
projections provide high-resolution visualization of the retinal and choroidal
vasculature, which is critical for the rapid and accurate diagnosis of retinal
diseases. However, acquiring high-quality OCTA images is challenging due to
motion sensitivity and the high costs associated with software modifications
for conventional OCT devices. Moreover, current deep learning methods for
OCT-to-OCTA translation often overlook the vascular differences across retinal
layers and struggle to reconstruct the intricate, dense vascular details
necessary for reliable diagnosis. To overcome these limitations, we propose
XOCT, a novel deep learning framework that integrates Cross-Dimensional
Supervision (CDS) with a Multi-Scale Feature Fusion (MSFF) network for
layer-aware vascular reconstruction. Our CDS module leverages 2D layer-wise
en-face projections, generated via segmentation-weighted z-axis averaging, as
supervisory signals to compel the network to learn distinct representations for
each retinal layer through fine-grained, targeted guidance. Meanwhile, the MSFF
module enhances vessel delineation through multi-scale feature extraction
combined with a channel reweighting strategy, effectively capturing vascular
details at multiple spatial scales. Our experiments on the OCTA-500 dataset
demonstrate XOCT's improvements, especially for the en-face projections which
are significant for clinical evaluation of retinal pathologies, underscoring
its potential to enhance OCTA accessibility, reliability, and diagnostic value
for ophthalmic disease detection and monitoring. The code is available at
https://github.com/uci-cbcl/XOCT.

### 6. [ANYPORTAL: Zero-Shot Consistent Video Background Replacement](http://arxiv.org/pdf/2509.07472v1)

Authors: Wenshuo Gao, Xicheng Lan, Shuai Yang

Despite the rapid advancements in video generation technology, creating
high-quality videos that precisely align with user intentions remains a
significant challenge. Existing methods often fail to achieve fine-grained
control over video details, limiting their practical applicability. We
introduce ANYPORTAL, a novel zero-shot framework for video background
replacement that leverages pre-trained diffusion models. Our framework
collaboratively integrates the temporal prior of video diffusion models with
the relighting capabilities of image diffusion models in a zero-shot setting.
To address the critical challenge of foreground consistency, we propose a
Refinement Projection Algorithm, which enables pixel-level detail manipulation
to ensure precise foreground preservation. ANYPORTAL is training-free and
overcomes the challenges of achieving foreground consistency and temporally
coherent relighting. Experimental results demonstrate that ANYPORTAL achieves
high-quality results on consumer-grade GPUs, offering a practical and efficient
solution for video content creation and editing.

### 7. [LINR Bridge: Vector Graphic Animation via Neural Implicits and Video Diffusion Priors](http://arxiv.org/pdf/2509.07484v1)

Authors: Wenshuo Gao, Xicheng Lan, Luyao Zhang, Shuai Yang

Vector graphics, known for their scalability and user-friendliness, provide a
unique approach to visual content compared to traditional pixel-based images.
Animation of these graphics, driven by the motion of their elements, offers
enhanced comprehensibility and controllability but often requires substantial
manual effort. To automate this process, we propose a novel method that
integrates implicit neural representations with text-to-video diffusion models
for vector graphic animation. Our approach employs layered implicit neural
representations to reconstruct vector graphics, preserving their inherent
properties such as infinite resolution and precise color and shape constraints,
which effectively bridges the large domain gap between vector graphics and
diffusion models. The neural representations are then optimized using video
score distillation sampling, which leverages motion priors from pretrained
text-to-video diffusion models. Finally, the vector graphics are warped to
match the representations resulting in smooth animation. Experimental results
validate the effectiveness of our method in generating vivid and natural vector
graphic animations, demonstrating significant improvement over existing
techniques that suffer from limitations in flexibility and animation quality.

### 8. [MVAT: Multi-View Aware Teacher for Weakly Supervised 3D Object Detection](http://arxiv.org/pdf/2509.07507v1)

Authors: Saad Lahlali, Alexandre Fournier Montgieux, Nicolas Granger, Hervé Le Borgne, Quoc Cuong Pham

Annotating 3D data remains a costly bottleneck for 3D object detection,
motivating the development of weakly supervised annotation methods that rely on
more accessible 2D box annotations. However, relying solely on 2D boxes
introduces projection ambiguities since a single 2D box can correspond to
multiple valid 3D poses. Furthermore, partial object visibility under a single
viewpoint setting makes accurate 3D box estimation difficult. We propose MVAT,
a novel framework that leverages temporal multi-view present in sequential data
to address these challenges. Our approach aggregates object-centric point
clouds across time to build 3D object representations as dense and complete as
possible. A Teacher-Student distillation paradigm is employed: The Teacher
network learns from single viewpoints but targets are derived from temporally
aggregated static objects. Then the Teacher generates high quality
pseudo-labels that the Student learns to predict from a single viewpoint for
both static and moving objects. The whole framework incorporates a multi-view
2D projection loss to enforce consistency between predicted 3D boxes and all
available 2D annotations. Experiments on the nuScenes and Waymo Open datasets
demonstrate that MVAT achieves state-of-the-art performance for weakly
supervised 3D object detection, significantly narrowing the gap with fully
supervised methods without requiring any 3D box annotations. % \footnote{Code
available upon acceptance} Our code is available in our public repository
(\href{https://github.com/CEA-LIST/MVAT}{code}).

### 9. [Universal Few-Shot Spatial Control for Diffusion Models](http://arxiv.org/pdf/2509.07530v1)

Authors: Kiet T. Nguyen, Chanhuyk Lee, Donggyun Kim, Dong Hoon Lee, Seunghoon Hong

Spatial conditioning in pretrained text-to-image diffusion models has
significantly improved fine-grained control over the structure of generated
images. However, existing control adapters exhibit limited adaptability and
incur high training costs when encountering novel spatial control conditions
that differ substantially from the training tasks. To address this limitation,
we propose Universal Few-Shot Control (UFC), a versatile few-shot control
adapter capable of generalizing to novel spatial conditions. Given a few
image-condition pairs of an unseen task and a query condition, UFC leverages
the analogy between query and support conditions to construct task-specific
control features, instantiated by a matching mechanism and an update on a small
set of task-specific parameters. Experiments on six novel spatial control tasks
show that UFC, fine-tuned with only 30 annotated examples of novel tasks,
achieves fine-grained control consistent with the spatial conditions. Notably,
when fine-tuned with 0.1% of the full training data, UFC achieves competitive
performance with the fully supervised baselines in various control tasks. We
also show that UFC is applicable agnostically to various diffusion backbones
and demonstrate its effectiveness on both UNet and DiT architectures. Code is
available at https://github.com/kietngt00/UFC.

### 10. [TextlessRAG: End-to-End Visual Document RAG by Speech Without Text](http://arxiv.org/pdf/2509.07538v1)

Authors: Peijin Xie, Shun Qian, Bingquan Liu, Dexin Wang, Lin Sun, Xiangzheng Zhang

Document images encapsulate a wealth of knowledge, while the portability of
spoken queries enables broader and flexible application scenarios. Yet, no
prior work has explored knowledge base question answering over visual document
images with queries provided directly in speech. We propose TextlessRAG, the
first end-to-end framework for speech-based question answering over large-scale
document images. Unlike prior methods, TextlessRAG eliminates ASR, TTS and OCR,
directly interpreting speech, retrieving relevant visual knowledge, and
generating answers in a fully textless pipeline. To further boost performance,
we integrate a layout-aware reranking mechanism to refine retrieval.
Experiments demonstrate substantial improvements in both efficiency and
accuracy. To advance research in this direction, we also release the first
bilingual speech--document RAG dataset, featuring Chinese and English voice
queries paired with multimodal document content. Both the dataset and our
pipeline will be made available at
repository:https://github.com/xiepeijinhit-hue/textlessrag

### Computers and Society

### 1. [Develop-Fair Use for Artificial Intelligence: A Sino-U.S. Copyright Law Comparison Based on the Ultraman, Bartz v. Anthropic, and Kadrey v. Meta Cases](http://arxiv.org/pdf/2509.07365v1)

Authors: Chanhou Lou

Traditional fair use can no longer respond to the challenges posed by
generative AI. Drawing on a comparative analysis of China's Ultraman and the
U.S. cases Bartz v. Anthropic and Kadrey v. Meta, this article proposes
"Develop-Fair Use" (DFU). DFU treats AI fair use (AIFU) not as a fixed
exception but as a dynamic tool of judicial balancing that shifts analysis from
closed scenarios to an evaluative rule for open-ended contexts. The judicial
focus moves from formal classification of facts to a substantive balancing of
competition in relevant markets. Although China and the U.S. follow different
paths, both reveal this logic: Ultraman, by articulating a "four-context
analysis," creates institutional space for AI industry development; the debate
over the fourth factor, market impact, in the two U.S. cases, especially
Kadrey's "market dilution" claim, expands review from substitution in copyright
markets to wider industrial competition. The core of DFU is to recognize and
balance the tension in relevant markets between an emerging AI industry that
invokes fair use to build its markets and a publishing industry that develops
markets, including one for "training licenses," to resist fair use. The
boundary of fair use is therefore not a product of pure legal deduction but a
case-specific factual judgment grounded in evolving market realities. This
approach aims both to trim excess copyright scope and to remedy shortfalls in
market competition.

### 2. [Towards Postmortem Data Management Principles for Generative AI](http://arxiv.org/pdf/2509.07375v1)

Authors: Ismat Jarin, Elina Van Kempen, Chloe Georgiou

Foundation models, large language models (LLMs), and agentic AI systems rely
heavily on vast corpora of user data. The use of such data for training has
raised persistent concerns around ownership, copyright, and potential harms. In
this work, we explore a related but less examined dimension: the ownership
rights of data belonging to deceased individuals. We examine the current
landscape of post-mortem data management and privacy rights as defined by the
privacy policies of major technology companies and regulations such as the EU
AI Act. Based on this analysis, we propose three post-mortem data management
principles to guide the protection of deceased individuals data rights.
Finally, we discuss directions for future work and offer recommendations for
policymakers and privacy practitioners on deploying these principles alongside
technological solutions to operationalize and audit them in practice.

### 3. [Water Demand Forecasting of District Metered Areas through Learned Consumer Representations](http://arxiv.org/pdf/2509.07515v1)

Authors: Adithya Ramachandran, Thorkil Flensmark B. Neergaard, Tomás Arias-Vergara, Andreas Maier, Siming Bayer

Advancements in smart metering technologies have significantly improved the
ability to monitor and manage water utilities. In the context of increasing
uncertainty due to climate change, securing water resources and supply has
emerged as an urgent global issue with extensive socioeconomic ramifications.
Hourly consumption data from end-users have yielded substantial insights for
projecting demand across regions characterized by diverse consumption patterns.
Nevertheless, the prediction of water demand remains challenging due to
influencing non-deterministic factors, such as meteorological conditions. This
work introduces a novel method for short-term water demand forecasting for
District Metered Areas (DMAs) which encompass commercial, agricultural, and
residential consumers. Unsupervised contrastive learning is applied to
categorize end-users according to distinct consumption behaviors present within
a DMA. Subsequently, the distinct consumption behaviors are utilized as
features in the ensuing demand forecasting task using wavelet-transformed
convolutional networks that incorporate a cross-attention mechanism combining
both historical data and the derived representations. The proposed approach is
evaluated on real-world DMAs over a six-month period, demonstrating improved
forecasting performance in terms of MAPE across different DMAs, with a maximum
improvement of 4.9%. Additionally, it identifies consumers whose behavior is
shaped by socioeconomic factors, enhancing prior knowledge about the
deterministic patterns that influence demand.

### 4. [Individual utilities of life satisfaction reveal inequality aversion unrelated to political alignment](http://arxiv.org/pdf/2509.07793v1)

Authors: Crispin Cooper, Ana Friedrich, Tommaso Reggiani, Wouter Poortinga

How should well-being be prioritised in society, and what trade-offs are
people willing to make between fairness and personal well-being? We investigate
these questions using a stated preference experiment with a nationally
representative UK sample (n = 300), in which participants evaluated life
satisfaction outcomes for both themselves and others under conditions of
uncertainty. Individual-level utility functions were estimated using an
Expected Utility Maximisation (EUM) framework and tested for sensitivity to the
overweighting of small probabilities, as characterised by Cumulative Prospect
Theory (CPT). A majority of participants displayed concave (risk-averse)
utility curves and showed stronger aversion to inequality in societal life
satisfaction outcomes than to personal risk. These preferences were unrelated
to political alignment, suggesting a shared normative stance on fairness in
well-being that cuts across ideological boundaries. The results challenge use
of average life satisfaction as a policy metric, and support the development of
nonlinear utility-based alternatives that more accurately reflect collective
human values. Implications for public policy, well-being measurement, and the
design of value-aligned AI systems are discussed.

### Databases

### 1. [Filtered Approximate Nearest Neighbor Search: A Unified Benchmark and Systematic Experimental Study [Experiment, Analysis & Benchmark]](http://arxiv.org/pdf/2509.07789v1)

Authors: Jiayang Shi, Yuzheng Cai, Weiguo Zheng

For a given dataset $\mathcal{D}$ and structured label $f$, the goal of
Filtered Approximate Nearest Neighbor Search (FANNS) algorithms is to find
top-$k$ points closest to a query that satisfy label constraints, while
ensuring both recall and QPS (Queries Per Second). In recent years, many FANNS
algorithms have been proposed. However, the lack of a systematic investigation
makes it difficult to understand their relative strengths and weaknesses.
Additionally, we found that: (1) FANNS algorithms have coupled,
dataset-dependent parameters, leading to biased comparisons. (2) Key impact
factors are rarely analyzed systematically, leaving unclear when each algorithm
performs well. (3) Disparate datasets, workloads, and biased experiment designs
make cross-algorithm comparisons unreliable. Thus, a comprehensive survey and
benchmark for FANNS is crucial to achieve the following goals: designing a fair
evaluation and clarifying the classification of algorithms, conducting in-depth
analysis of their performance, and establishing a unified benchmark. First, we
propose a taxonomy (dividing methods into \textit{filter-then-search},
\textit{search-then-filter}, \textit{hybrid-search}) and a systematic
evaluation framework, integrating unified parameter tuning and standardized
filtering across algorithms to reduce implementation-induced performance
variations and reflect core trade-offs. Then, we conduct a comprehensive
empirical study to analyze how query difficulty and dataset properties impact
performance, evaluating robustness under pressures like filter selectivity,
Recall@k, and scalability to clarify each method's strengths. Finally, we
establish a standardized benchmark with real-world datasets and open-source
related resources to ensure reproducible future research.

### 2. [Proximity Graphs for Similarity Search: Fast Construction, Lower Bounds, and Euclidean Separation](http://arxiv.org/pdf/2509.07732v1)

Authors: Shangqi Lu, Yufei Tao

Proximity graph-based methods have emerged as a leading paradigm for
approximate nearest neighbor (ANN) search in the system community. This paper
presents fresh insights into the theoretical foundation of these methods. We
describe an algorithm to build a proximity graph for $(1+\epsilon)$-ANN search
that has $O((1/\epsilon)^\lambda \cdot n \log \Delta)$ edges and guarantees
$(1/\epsilon)^\lambda \cdot \text{polylog }\Delta$ query time. Here, $n$ and
$\Delta$ are the size and aspect ratio of the data input, respectively, and
$\lambda = O(1)$ is the doubling dimension of the underlying metric space. Our
construction time is near-linear to $n$, improving the $\Omega(n^2)$ bounds of
all previous constructions. We complement our algorithm with lower bounds
revealing an inherent limitation of proximity graphs: the number of edges needs
to be at least $\Omega((1/\epsilon)^\lambda \cdot n + n \log \Delta)$ in the
worst case, up to a subpolynomial factor. The hard inputs used in our
lower-bound arguments are non-geometric, thus prompting the question of whether
improvement is possible in the Euclidean space (a key subclass of metric
spaces). We provide an affirmative answer by using geometry to reduce the graph
size to $O((1/\epsilon)^\lambda \cdot n)$ while preserving nearly the same
query and construction time.

### 3. [dciWebMapper2: Enhancing the dciWebMapper framework toward integrated, interactive visualization of linked multi-type maps, charts, and spatial statistics and analysis](http://arxiv.org/pdf/2509.07897v1)

Authors: Sarigai Sarigai, Liping Yang, Katie Slack, Carolyn Fish, Michaela Buenemann, Qiusheng Wu, Yan Lin, Joseph A. Cook, David Jacobs

As interactive web-based geovisualization becomes increasingly vital across
disciplines, there is a growing need for open-source frameworks that support
dynamic, multi-attribute spatial analysis and accessible design. This paper
introduces dciWebMapper2, a significant expansion of the original dciWebMapper
framework, designed to enable exploratory analysis across domains such as
climate justice, food access, and social vulnerability. The enhanced framework
integrates multiple map types, including choropleth, proportional symbol, small
multiples, and heatmaps, with linked statistical charts (e.g., scatter plots,
boxplots) and time sliders, all within a coordinated-view environment.
Dropdown-based controls allow flexible, high-dimensional comparisons while
maintaining visual clarity. Grounded in cartographic and information
visualization principles, dciWebMapper2 is fully open-source, self-contained,
and server-free, supporting modularity, reproducibility, and long-term
sustainability. Three applied use cases demonstrate its adaptability and
potential to democratize interactive web cartography. This work offers a
versatile foundation for inclusive spatial storytelling and transparent
geospatial analysis in research, education, and civic engagement.

### Distributed, Parallel, and Cluster Computing

### 1. [DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference](http://arxiv.org/pdf/2509.07379v1)

Authors: Yuning Zhang, Grant Pinkert, Nan Yang, Yanli Li, Dong Yuan

Large Language Models (LLMs) have demonstrated impressive performance across
a wide range of deep learning tasks. Mixture of Experts (MoE) further enhances
their capabilities by increasing model width through sparsely activated expert
branches, which keeps inference computation efficient. However, the large
number of expert weights introduces significant GPU memory pressure, especially
in resource-constrained environments such as single-GPU servers. More
importantly, MoE inference consists of two fundamentally different stages: a
prefill stage where most experts are activated densely, and a decode stage
where only a few experts are triggered sparsely. Treating these stages with a
uniform scheduling strategy often leads to suboptimal latency and memory usage.
To address this, we propose DuoServe-MoE, an inference serving system that
explicitly separates prefill and decode stages and applies tailored expert
scheduling strategies to each. In the prefill stage, DuoServe-MoE uses a
two-stream CUDA pipeline that overlaps expert weight prefetching with the
computation of non-MoE layers, limiting expert residency in GPU memory. In the
decode stage, a lightweight layer-level predictor trained offline from
activation traces is used to prefetch only the most likely activated experts,
without requiring any changes to the model. Experiments on 4-bit Mixtral-8x7B
and 8x22B models show that DuoServe-MoE improves end-to-end latency by 1.42 to
7.54 times while keeping peak memory usage at only 15 percent of the full model
size.

### 2. [Dependency-Aware Execution Mechanism in Hyperledger Fabric Architecture](http://arxiv.org/pdf/2509.07425v1)

Authors: Sanyam Kaul, Manaswini Piduguralla, Gayathri Shreeya Patnala, Sathya Peri

Hyperledger Fabric is a leading permissioned blockchain framework for
enterprise use, known for its modular design and privacy features. While it
strongly supports configurable consensus and access control, Fabric can face
challenges in achieving high transaction throughput and low rejection rates
under heavy workloads. These performance limitations are often attributed to
endorsement, ordering, and validation bottlenecks. Further, optimistic
concurrency control and deferred validation in Fabric may lead to resource
inefficiencies and contention, as conflicting transactions are identified only
during the commit phase. To address these challenges, we propose a
dependency-aware execution model for Hyperledger Fabric. Our approach includes:
(a) a dependency flagging system during endorsement, marking transactions as
independent or dependent using a hashmap; (b) an optimized block construction
in the ordering service that prioritizes independent transactions; (c) the
incorporation of a Directed Acyclic Graph (DAG) within each block to represent
dependencies; and (d) parallel execution of independent transactions at the
committer, with dependent transactions processed according to DAG order.
Incorporated in Hyperledger Fabric v2.5, our framework was tested on workloads
with varying dependency levels and system loads. Results show up to 40% higher
throughput and significantly reduced rejection rates in high-contention
scenarios. This demonstrates that dependency-aware scheduling and DAG-based
execution can substantially enhance Fabric's scalability while remaining
compatible with its existing consensus and smart contract layers.

### 3. [Navigating Energy Doldrums: Modeling the Impact of Energy Price Volatility on HPC Cost of Ownership](http://arxiv.org/pdf/2509.07567v1)

Authors: Peter Arzt, Felix Wolf

Energy costs are a major factor in the total cost of ownership (TCO) for
high-performance computing (HPC) systems. The rise of intermittent green energy
sources and reduced reliance on fossil fuels have introduced volatility into
electricity markets, complicating energy budgeting. This paper explores
variable capacity as a strategy for managing HPC energy costs - dynamically
adjusting compute resources in response to fluctuating electricity prices.
While this approach can lower energy expenses, it risks underutilizing costly
hardware. To evaluate this trade-off, we present a simple model that helps
operators estimate the TCO impact of variable capacity strategies using key
system parameters. We apply this model to real data from a university HPC
cluster and assess how different scenarios could affect the cost-effectiveness
of this approach in the future.

### 4. [AgentX: Towards Orchestrating Robust Agentic Workflow Patterns with FaaS-hosted MCP Services](http://arxiv.org/pdf/2509.07595v1)

Authors: Shiva Sai Krishna Anand Tokal, Vaibhav Jha, Anand Eswaran, Praveen Jayachandran, Yogesh Simmhan

Generative Artificial Intelligence (GenAI) has rapidly transformed various
fields including code generation, text summarization, image generation and so
on. Agentic AI is a recent evolution that further advances this by coupling the
decision making and generative capabilities of LLMs with actions that can be
performed using tools. While seemingly powerful, Agentic systems often struggle
when faced with numerous tools, complex multi-step tasks,and long-context
management to track history and avoid hallucinations. Workflow patterns such as
Chain-of-Thought (CoT) and ReAct help address this. Here, we define a novel
agentic workflow pattern, AgentX, composed of stage designer, planner, and
executor agents that is competitive or better than the state-of-the-art agentic
patterns. We also leverage Model Context Protocol (MCP) tools, and propose two
alternative approaches for deploying MCP servers as cloud Functions as a
Service (FaaS). We empirically evaluate the success rate, latency and cost for
AgentX and two contemporary agentic patterns, ReAct and Magentic One, using
these the FaaS and local MCP server alternatives for three practical
applications. This highlights the opportunities and challenges of designing and
deploying agentic workflows.

### 5. [Scaling atomic ordering in shared memory](http://arxiv.org/pdf/2509.07781v1)

Authors: Lorenzo Martignetti, Eliã Batista, Gianpaolo Cugola, Fernando Pedone

Atomic multicast is a communication primitive used in dependable systems to
ensure consistent ordering of messages delivered to a set of replica groups.
This primitive enables critical services to integrate replication and sharding
(i.e., state partitioning) to achieve fault tolerance and scalability. While
several atomic multicast protocols have been developed for message-passing
systems, only a few are designed for the shared memory system model. This paper
introduces TRAM, an atomic multicast protocol specifically designed for shared
memory systems, leveraging an overlay tree architecture. Due to its simple and
practical design, TRAM delivers exceptional performance, increasing throughput
by more than 3$\times$ and reducing latency by more than 2.3$\times$ compared
to state-of-the-art shared memory-based protocols. Additionally, it
significantly outperforms message-passing-based protocols, boosting throughput
by up to 5.9$\times$ and reducing latency by up to 106$\times$.

### 6. [FedTeddi: Temporal Drift and Divergence Aware Scheduling for Timely Federated Edge Learning](http://arxiv.org/pdf/2509.07342v1)

Authors: Yuxuan Bai, Yuxuan Sun, Tan Chen, Wei Chen, Sheng Zhou, Zhisheng Niu

Federated edge learning (FEEL) enables collaborative model training across
distributed clients over wireless networks without exposing raw data. While
most existing studies assume static datasets, in real-world scenarios clients
may continuously collect data with time-varying and non-independent and
identically distributed (non-i.i.d.) characteristics. A critical challenge is
how to adapt models in a timely yet efficient manner to such evolving data. In
this paper, we propose FedTeddi, a temporal-drift-and-divergence-aware
scheduling algorithm that facilitates fast convergence of FEEL under dynamic
data evolution and communication resource limits. We first quantify the
temporal dynamics and non-i.i.d. characteristics of data using temporal drift
and collective divergence, respectively, and represent them as the Earth
Mover's Distance (EMD) of class distributions for classification tasks. We then
propose a novel optimization objective and develop a joint scheduling and
bandwidth allocation algorithm, enabling the FEEL system to learn from new data
quickly without forgetting previous knowledge. Experimental results show that
our algorithm achieves higher test accuracy and faster convergence compared to
benchmark methods, improving the rate of convergence by 58.4% on CIFAR-10 and
49.2% on CIFAR-100 compared to random scheduling.

### 7. [Optimizing Task Scheduling in Fog Computing with Deadline Awareness](http://arxiv.org/pdf/2509.07378v1)

Authors: Mohammad Sadegh Sirjani, Somayeh Sobati-Moghadam

The rise of Internet of Things (IoT) devices has led to the development of
numerous applications that require quick responses and low latency. Fog
computing has emerged as a solution for processing these IoT applications, but
it faces challenges such as resource allocation and job scheduling. Therefore,
it is crucial to determine how to assign and schedule tasks on Fog nodes. A
well-designed job scheduling algorithm can help decrease energy usage and
improve response times for application requests. This work aims to schedule
tasks in IoT while minimizing the total energy consumption of nodes and
enhancing the Quality of Service (QoS) requirements of IoT tasks, taking into
account task deadlines. Initially, this paper classifies the Fog nodes into two
categories based on their traffic level: low and high. It schedules
low-deadline tasks on low-traffic-level nodes using an Improved Golden Eagle
Optimization (IGEO) algorithm, an enhancement of the Golden Eagle Optimization
Algorithm that utilizes genetic operators for discretization. High-deadline
tasks are processed on high-traffic nodes using reinforcement learning (RL).
This combined approach is called the Reinforcement Improved Golden Eagle
Optimization (RIGEO) algorithm. Experimental results demonstrate that the
proposed algorithms optimize system response time, total deadline violation
time, and resource and system energy consumption compared to other
state-of-the-art algorithms.

### 8. [DREAMS: Decentralized Resource Allocation and Service Management across the Compute Continuum Using Service Affinity](http://arxiv.org/pdf/2509.07497v1)

Authors: Hai Dinh-Tuan, Tien Hung Nguyen, Sanjeet Raj Pandey

Modern manufacturing systems require adaptive computing infrastructures that
can respond to highly dynamic workloads and increasingly customized production
demands. The compute continuum emerges as a promising solution, enabling
flexible deployment of microservices across distributed, heterogeneous domains.
However, this paradigm also requires a novel approach to resource allocation
and service placement, as traditional centralized solutions struggle to scale
effectively, suffer from latency bottlenecks, and introduce single points of
failure. In this paper, we present DREAMS, a decentralized framework that
optimizes microservice placement decisions collaboratively across different
computational domains. At its core, DREAMS introduces agents that operate
autonomously within each domain while coordinating globally through a
Raft-based consensus algorithm and cost-benefit voting. This decentralized
architecture enables responsive, privacy-preserving, and fault-tolerant
coordination, making it particularly suitable given the growing prevalence of
multi-stakeholder scenarios across the compute continuum. In particular, within
modern manufacturing environments, DREAMS achieves globally optimized service
placements while maintaining high fault tolerance. Further evaluations
demonstrate that key coordination operations, such as Local Domain Manager
(LDM) registration and migration voting, scale sub-linearly with the number of
domains, confirming the efficiency and scalability of our proposal.

### 9. [MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?](http://arxiv.org/pdf/2509.07727v1)

Authors: Songkai Ma, Zhaorui Zhang, Sheng Di, Benben Liu, Xiaodong Yu, Xiaoyi Lu, Dan Wang

With the widespread application of Mixture of Experts (MoE) reasoning models
in the field of LLM learning, efficiently serving MoE models under limited GPU
memory constraints has emerged as a significant challenge. Offloading the
non-activated experts to main memory has been identified as an efficient
approach to address such a problem, while it brings the challenges of
transferring the expert between the GPU memory and main memory. We need to
explore an efficient approach to compress the expert and analyze how the
compression error affects the inference performance.
  To bridge this gap, we propose employing error-bounded lossy compression
algorithms (such as SZ3 and CuSZp) to compress non-activated experts, thereby
reducing data transfer overhead during MoE inference. We conduct extensive
experiments across various benchmarks and present a comprehensive analysis of
how compression-induced errors in different experts affect overall inference
accuracy. The results indicate that experts in the shallow layers, which are
primarily responsible for the attention mechanism and the transformation of
input tokens into vector representations, exhibit minimal degradation in
inference accuracy when subjected to bounded errors. In contrast, errors in the
middle-layer experts, which are central to model reasoning, significantly
impair inference accuracy. Interestingly, introducing bounded errors in the
deep-layer experts, which are mainly responsible for instruction following and
output integration, can sometimes lead to improvements in inference accuracy.

### 10. [Astra: A Multi-Agent System for GPU Kernel Performance Optimization](http://arxiv.org/pdf/2509.07506v1)

Authors: Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken

GPU kernel optimization has long been a central challenge at the intersection
of high-performance computing and machine learning. Efficient kernels are
crucial for accelerating large language model (LLM) training and serving, yet
attaining high performance typically requires extensive manual tuning.
Compiler-based systems reduce some of this burden, but still demand substantial
manual design and engineering effort. Recently, researchers have explored using
LLMs for GPU kernel generation, though prior work has largely focused on
translating high-level PyTorch modules into CUDA code. In this work, we
introduce Astra, the first LLM-based multi-agent system for GPU kernel
optimization. Unlike previous approaches, Astra starts from existing CUDA
implementations extracted from SGLang, a widely deployed framework for serving
LLMs, rather than treating PyTorch modules as the specification. Within Astra,
specialized LLM agents collaborate through iterative code generation, testing,
profiling, and planning to produce kernels that are both correct and
high-performance. On kernels from SGLang, Astra achieves an average speedup of
1.32x using zero-shot prompting with OpenAI o4-mini. A detailed case study
further demonstrates that LLMs can autonomously apply loop transformations,
optimize memory access patterns, exploit CUDA intrinsics, and leverage fast
math operations to yield substantial performance gains. Our work highlights
multi-agent LLM systems as a promising new paradigm for GPU kernel
optimization.

### Digital Libraries

### 1. [SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP](http://arxiv.org/pdf/2509.07801v1)

Authors: Decheng Duan, Yingyi Zhang, Jitong Peng, Chengzhi Zhang

Structured information extraction from scientific literature is crucial for
capturing core concepts and emerging trends in specialized fields. While
existing datasets aid model development, most focus on specific publication
sections due to domain complexity and the high cost of annotating scientific
texts. To address this limitation, we introduce SciNLP - a specialized
benchmark for full-text entity and relation extraction in the Natural Language
Processing (NLP) domain. The dataset comprises 60 manually annotated full-text
NLP publications, covering 7,072 entities and 1,826 relations. Compared to
existing research, SciNLP is the first dataset providing full-text annotations
of entities and their relationships in the NLP domain. To validate the
effectiveness of SciNLP, we conducted comparative experiments with similar
datasets and evaluated the performance of state-of-the-art supervised models on
this dataset. Results reveal varying extraction capabilities of existing models
across academic texts of different lengths. Cross-comparisons with existing
datasets show that SciNLP achieves significant performance improvements on
certain baseline models. Using models trained on SciNLP, we implemented
automatic construction of a fine-grained knowledge graph for the NLP domain.
Our KG has an average node degree of 3.2 per entity, indicating rich semantic
topological information that enhances downstream applications. The dataset is
publicly available at https://github.com/AKADDC/SciNLP.

### Discrete Mathematics

### 1. [On the Convergence of Elementary Cellular Automata under Sequential Update Modes](http://arxiv.org/pdf/2509.07797v1)

Authors: Isabel Donoso-Leiva, Eric Goles, Martín Ríos-Wilson, Sylvain Sené

In this paper, we perform a theoretical analysis of the sequential
convergence of elementary cellular automata that have at least one fixed point.
Our aim is to establish which elementary rules always reach fixed points under
sequential update modes, regardless of the initial configuration. In this
context, we classify these rules according to whether all initial
configurations converge under all, some, one or none sequential update modes,
depending on if they have fixed points under synchronous (or parallel) update
modes.

### Data Structures and Algorithms

### 1. [Dimension Reduction for Clustering: The Curious Case of Discrete Centers](http://arxiv.org/pdf/2509.07444v1)

Authors: Shaofeng H. -C. Jiang, Robert Krauthgamer, Shay Sapir, Sandeep Silwal, Di Yue

The Johnson-Lindenstrauss transform is a fundamental method for dimension
reduction in Euclidean spaces, that can map any dataset of $n$ points into
dimension $O(\log n)$ with low distortion of their distances. This dimension
bound is tight in general, but one can bypass it for specific problems. Indeed,
tremendous progress has been made for clustering problems, especially in the
\emph{continuous} setting where centers can be picked from the ambient space
$\mathbb{R}^d$. Most notably, for $k$-median and $k$-means, the dimension bound
was improved to $O(\log k)$ [Makarychev, Makarychev and Razenshteyn, STOC
2019].
  We explore dimension reduction for clustering in the \emph{discrete} setting,
where centers can only be picked from the dataset, and present two results that
are both parameterized by the doubling dimension of the dataset, denoted as
$\operatorname{ddim}$. The first result shows that dimension
$O_{\epsilon}(\operatorname{ddim} + \log k + \log\log n)$ suffices, and is
moreover tight, to guarantee that the cost is preserved within factor
$1\pm\epsilon$ for every set of centers. Our second result eliminates the
$\log\log n$ term in the dimension through a relaxation of the guarantee
(namely, preserving the cost only for all approximately-optimal sets of
centers), which maintains its usefulness for downstream applications.
  Overall, we achieve strong dimension reduction in the discrete setting, and
find that it differs from the continuous setting not only in the dimension
bound, which depends on the doubling dimension, but also in the guarantees
beyond preserving the optimal value, such as which clusterings are preserved.

### 2. [The General Expiration Streaming Model: Diameter, $k$-Center, Counting, Sampling, and Friends](http://arxiv.org/pdf/2509.07587v1)

Authors: Lotte Blank, Sergio Cabello, MohammadTaghi Hajiaghayi, Robert Krauthgamer, Sepideh Mahabadi, André Nusser, Jeff M. Phillips, Jonas Sauer

An important thread in the study of data-stream algorithms focuses on
settings where stream items are active only for a limited time. We introduce a
new expiration model, where each item arrives with its own expiration time. The
special case where items expire in the order that they arrive, which we call
consistent expirations, contains the classical sliding-window model of Datar,
Gionis, Indyk, and Motwani [SICOMP 2002] and its timestamp-based variant of
Braverman and Ostrovsky [FOCS 2007].
  Our first set of results presents algorithms (in the expiration streaming
model) for several fundamental problems, including approximate counting,
uniform sampling, and weighted sampling by efficiently tracking active items
without explicitly storing them all. Naturally, these algorithms have many
immediate applications to other problems.
  Our second and main set of results designs algorithms (in the expiration
streaming model) for the diameter and $k$-center problems, where items are
points in a metric space. Our results significantly extend those known for the
special case of sliding-window streams by Cohen-Addad, Schwiegelshohn, and
Sohler [ICALP 2016], including also a strictly better approximation factor for
the diameter in the important special case of high-dimensional Euclidean space.
We develop new decomposition and coordination techniques along with a geometric
dominance framework, to filter out redundant points based on both temporal and
spatial proximity.

### 3. [Tight Bounds for Low-Error Frequency Moment Estimation and the Power of Multiple Passes](http://arxiv.org/pdf/2509.07599v1)

Authors: Naomi Green-Maimon, Or Zamir

Estimating the second frequency moment $F_2$ of a data stream up to a $(1 \pm
\varepsilon)$ factor is a central problem in the streaming literature. For
errors $\varepsilon > \Omega(1/\sqrt{n})$, the tight bound
$\Theta\left(\log(\varepsilon^2 n)/\varepsilon^2\right)$ was recently
established by Braverman and Zamir. In this work, we complete the picture by
resolving the remaining regime of small error, $\varepsilon < 1/\sqrt{n}$,
showing that the optimal space complexity is $\Theta\left( \min\left(n,
\frac{1}{\varepsilon^2} \right) \cdot \left(1 + \left| \log(\varepsilon^2 n)
\right| \right) \right)$ bits for all $\varepsilon \geq 1/n^2$, assuming a
sufficiently large universe. This closes the gap between the best known
$\Omega(n)$ lower bound and the straightforward $O(n \log n)$ upper bound in
that range, and shows that essentially storing the entire stream is necessary
for high-precision estimation.
  To derive this bound, we fully characterize the two-party communication
complexity of estimating the size of a set intersection up to an arbitrary
additive error $\varepsilon n$. In particular, we prove a tight $\Omega(n \log
n)$ lower bound for one-way communication protocols when $\varepsilon <
n^{-1/2-\Omega(1)}$, in contrast to classical $O(n)$-bit protocols that use
two-way communication. Motivated by this separation, we present a two-pass
streaming algorithm that computes the exact histogram of a stream with high
probability using only $O(n \log \log n)$ bits of space, in contrast to the
$\Theta(n \log n)$ bits required in one pass even to approximate $F_2$ with
small error. This yields the first asymptotic separation between one-pass and
$O(1)$-passes space complexity for small frequency moment estimation.

### 4. [Compressibility Measures and Succinct Data Structures for Piecewise Linear Approximations](http://arxiv.org/pdf/2509.07827v1)

Authors: Paolo Ferragina, Filippo Lari

We study the problem of deriving compressibility measures for \emph{Piecewise
Linear Approximations} (PLAs), i.e., error-bounded approximations of a set of
two-dimensional {\em increasing} data points using a sequence of segments. Such
approximations are widely used tools in implementing many \emph{learned data
structures}, which mix learning models with traditional algorithmic design
blocks to exploit regularities in the underlying data distribution, providing
novel and effective space-time trade-offs.
  We introduce the first lower bounds to the cost of storing PLAs in two
settings, namely {\em compression} and {\em indexing}. We then compare these
compressibility measures to known data structures, and show that they are
asymptotically optimal up to a constant factor from the space lower bounds.
Finally, we design the first data structures for the aforementioned settings
that achieve the space lower bounds plus small additive terms, which turn out
to be {\em succinct} in most practical cases. Our data structures support the
efficient retrieval and evaluation of a segment in the (compressed) PLA for a
given $x$-value, which is a core operation in any learned data structure
relying on PLAs.
  As a result, our paper offers the first theoretical analysis of the maximum
compressibility achievable by PLA-based learned data structures, and provides
novel storage schemes for PLAs offering strong theoretical guarantees while
also suggesting simple and efficient practical implementations.

### 5. [Proximity Graphs for Similarity Search: Fast Construction, Lower Bounds, and Euclidean Separation](http://arxiv.org/pdf/2509.07732v1)

Authors: Shangqi Lu, Yufei Tao

Proximity graph-based methods have emerged as a leading paradigm for
approximate nearest neighbor (ANN) search in the system community. This paper
presents fresh insights into the theoretical foundation of these methods. We
describe an algorithm to build a proximity graph for $(1+\epsilon)$-ANN search
that has $O((1/\epsilon)^\lambda \cdot n \log \Delta)$ edges and guarantees
$(1/\epsilon)^\lambda \cdot \text{polylog }\Delta$ query time. Here, $n$ and
$\Delta$ are the size and aspect ratio of the data input, respectively, and
$\lambda = O(1)$ is the doubling dimension of the underlying metric space. Our
construction time is near-linear to $n$, improving the $\Omega(n^2)$ bounds of
all previous constructions. We complement our algorithm with lower bounds
revealing an inherent limitation of proximity graphs: the number of edges needs
to be at least $\Omega((1/\epsilon)^\lambda \cdot n + n \log \Delta)$ in the
worst case, up to a subpolynomial factor. The hard inputs used in our
lower-bound arguments are non-geometric, thus prompting the question of whether
improvement is possible in the Euclidean space (a key subclass of metric
spaces). We provide an affirmative answer by using geometry to reduce the graph
size to $O((1/\epsilon)^\lambda \cdot n)$ while preserving nearly the same
query and construction time.

### Emerging Technologies

### 1. [PSketch: A Priority-Aware Sketch Architecture for Real-Time Flow Monitoring via eBPF](http://arxiv.org/pdf/2509.07338v1)

Authors: Yuanjun Dai, Qingzhe Guo, Xiangren Wang

Sketch-based monitoring in SDN often suffers from tightly coupled pipeline
and memory constraints, limiting algorithmic flexibility and reducing accuracy.
We propose PSketch, the first in-kernel priority-aware sketching framework
implemented with eBPF. It ensures lossless tracking of high-priority flows via
a hash-based table and approximates top-k elephant flows using a sketch pipe.
PSketch supports both TCP and UDP and enables in-kernel retransmission tracking
with minimal overhead. Unlike SDN-based approaches, it runs on commodity Linux
systems, removing hardware dependencies. We perform evaluation on 10 Gbps CAIDA
traces. Results show that PSketch achieves 96.0% top-k detection accuracy,
96.4% retransmission recall, and only 0.7% throughput degradation.

### 2. [Gut-Brain Axis as a Closed-Loop Molecular Communication Network](http://arxiv.org/pdf/2509.07911v1)

Authors: Beyza E. Ortlek, Ozgur B. Akan

Molecular communication (MC) provides a quantitative framework for analyzing
information transfer within biological systems. This paper introduces a novel
and comprehensive MC framework for the gut-brain axis (GBA) as a system of six
coupled, nonlinear delay differential equations (DDEs). The proposed model
defines a bidirectional feedback loop with a gut-to-brain inflammatory channel
and a brain-to-gut neuroendocrine channel. Under prolonged stress, this
feedback loop becomes self-perpetuating and drives the system into a
pathological state. We evaluate the end-to-end channel across varying
conditions using time-domain simulations, small-signal frequency-domain
characterization, and an information-theoretic capacity analysis. At
homeostasis, the system maintains stable circadian dynamics with higher
information throughput, whereas sustained stress drives a shift to dysregulated
hypercortisolism. In this pathological state, spectral efficiency decreases due
to a narrowed effective bandwidth and a lower passband gain driven by
neuroendocrine delays and saturating cytokine-hormone kinetics. These results
quantify the impact of these signaling mechanisms on stability and information
processing, elucidating the transition from healthy circadian rhythms to a
persistent pathological state of hypercortisolism.

### 3. [Toward Lifelong-Sustainable Electronic-Photonic AI Systems via Extreme Efficiency, Reconfigurability, and Robustness](http://arxiv.org/pdf/2509.07396v1)

Authors: Ziang Yin, Hongjian Zhou, Chetan Choppali Sudarshan, Vidya Chhabria, Jiaqi Gu

The relentless growth of large-scale artificial intelligence (AI) has created
unprecedented demand for computational power, straining the energy, bandwidth,
and scaling limits of conventional electronic platforms. Electronic-photonic
integrated circuits (EPICs) have emerged as a compelling platform for
next-generation AI systems, offering inherent advantages in ultra-high
bandwidth, low latency, and energy efficiency for computing and
interconnection. Beyond performance, EPICs also hold unique promises for
sustainability. Fabricated in relaxed process nodes with fewer metal layers and
lower defect densities, photonic devices naturally reduce embodied carbon
footprint (CFP) compared to advanced digital electronic integrated circuits,
while delivering orders-of-magnitude higher computing performance and
interconnect bandwidth. To further advance the sustainability of photonic AI
systems, we explore how electronic-photonic design automation (EPDA) and
cross-layer co-design methodologies can amplify these inherent benefits. We
present how advanced EPDA tools enable more compact layout generation, reducing
both chip area and metal layer usage. We will also demonstrate how cross-layer
device-circuit-architecture co-design unlocks new sustainability gains for
photonic hardware: ultra-compact photonic circuit designs that minimize chip
area cost, reconfigurable hardware topology that adapts to evolving AI
workloads, and intelligent resilience mechanisms that prolong lifetime by
tolerating variations and faults. By uniting intrinsic photonic efficiency with
EPDA- and co-design-driven gains in area efficiency, reconfigurability, and
robustness, we outline a vision for lifelong-sustainable electronic-photonic AI
systems. This perspective highlights how EPIC AI systems can simultaneously
meet the performance demands of modern AI and the urgent imperative for
sustainable computing.

### 4. [Bringing Multi-Modal Multi-Task Federated Foundation Models to Education Domain: Prospects and Challenges](http://arxiv.org/pdf/2509.07946v1)

Authors: Kasra Borazjani, Naji Khosravan, Rajeev Sahay, Bita Akram, Seyyedali Hosseinalipour

Multi-modal multi-task (M3T) foundation models (FMs) have recently shown
transformative potential in artificial intelligence, with emerging applications
in education. However, their deployment in real-world educational settings is
hindered by privacy regulations, data silos, and limited domain-specific data
availability. We introduce M3T Federated Foundation Models (FedFMs) for
education: a paradigm that integrates federated learning (FL) with M3T FMs to
enable collaborative, privacy-preserving training across decentralized
institutions while accommodating diverse modalities and tasks. Subsequently,
this position paper aims to unveil M3T FedFMs as a promising yet underexplored
approach to the education community, explore its potentials, and reveal its
related future research directions. We outline how M3T FedFMs can advance three
critical pillars of next-generation intelligent education systems: (i) privacy
preservation, by keeping sensitive multi-modal student and institutional data
local; (ii) personalization, through modular architectures enabling tailored
models for students, instructors, and institutions; and (iii) equity and
inclusivity, by facilitating participation from underrepresented and
resource-constrained entities. We finally identify various open research
challenges, including studying of (i) inter-institution heterogeneous privacy
regulations, (ii) the non-uniformity of data modalities' characteristics, (iii)
the unlearning approaches for M3T FedFMs, (iv) the continual learning
frameworks for M3T FedFMs, and (v) M3T FedFM model interpretability, which must
be collectively addressed for practical deployment.

### 5. [On-chip microwave sensing of quasiparticles in tantalum superconducting circuits on silicon for scalable quantum technologies](http://arxiv.org/pdf/2509.07669v1)

Authors: Shima Poorgholam-Khanjari, Paniz Foshat, Mingqi Zhang, Valentino Seferai, Martin Weides, Kaveh Delfanazari

The performance and scalability of superconducting quantum circuits are
fundamentally constrained by non-equilibrium quasiparticles, which induce
microwave losses that limit resonator quality factors and qubit coherence
times. Understanding and mitigating these excitations is therefore central to
advancing scalable quantum technologies. Here, we demonstrate on-chip microwave
sensing of quasiparticles in high-Q {\alpha}-tantalum coplanar waveguide
resonators on silicon, operated in the single-photon regime.
Temperature-dependent measurements reveal persistent non-equilibrium
quasiparticles at millikelvin temperatures, producing a measurable suppression
of the internal quality factor (Qi) relative to theoretical expectations. By
benchmarking across materials, we find that the quasiparticle density in
{\alpha}-Ta is approximately one-third that of NbN at equivalent normalised
temperatures (T/Tc), directly correlating with reduced microwave loss. Our
methodology establishes a scalable platform for probing quasiparticle dynamics
and points towards new routes for engineering superconducting circuits with
improved coherence, with impact on qubit readout resonators, kinetic-inductance
detectors, and emerging quantum processors and sensors.

### Formal Languages and Automata Theory

### 1. [Verification power of rational-valued automata with deterministic and affine states](http://arxiv.org/pdf/2509.07857v1)

Authors: Zeyu Chen, Abuzer Yakaryılmaz, Junde Wu

We investigate the verification power of rational-valued affine automata
within Arthur--Merlin proof systems. For one-way verifiers, we give real-time
protocols with perfect completeness and tunable bounded error for two benchmark
nonregular languages, the balanced-middle language and the centered-palindrome
language, illustrating a concrete advantage over probabilistic and quantum
finite-state verifiers. For two-way verifiers, we first design a weak protocol
that verifies every Turing-recognizable language by streaming and checking a
configuration history. We then strengthen it with a probabilistic continuation
check that bounds the prover's transcript length and ensures halting with high
probability, yielding strong verification with expected running time
proportional to the product of the simulated machine's space and time (up to
input length and a factor polynomial in the inverse error parameter). Combining
these constructions with standard alternation--space correspondences, we place
alternating exponential time, equivalently deterministic exponential space,
inside affine Arthur--Merlin with two-way affine automata. We also present a
reduction-based route with perfect completeness via a Knapsack-game verifier,
which, together with linear-space reductions, yields that the class PSPACE
admits affine Arthur--Merlin verification by two-way affine automata. Two
simple primitives drive our protocols: a probabilistic continuation check to
control expected time and a restart-on-accept affine register that converts
exact algebraic checks into eventually halting bounded-error procedures.

### Graphics

### 1. [Topology-Aware Optimization of Gaussian Primitives for Human-Centric Volumetric Videos](http://arxiv.org/pdf/2509.07653v1)

Authors: Yuheng Jiang, Chengcheng Guo, Yize Wu, Yu Hong, Shengkun Zhu, Zhehao Shen, Yingliang Zhang, Shaohui Jiao, Zhuo Su, Lan Xu, Marc Habermann, Christian Theobalt

Volumetric video is emerging as a key medium for digitizing the dynamic
physical world, creating the virtual environments with six degrees of freedom
to deliver immersive user experiences. However, robustly modeling general
dynamic scenes, especially those involving topological changes while
maintaining long-term tracking remains a fundamental challenge. In this paper,
we present TaoGS, a novel topology-aware dynamic Gaussian representation that
disentangles motion and appearance to support, both, long-range tracking and
topological adaptation. We represent scene motion with a sparse set of motion
Gaussians, which are continuously updated by a spatio-temporal tracker and
photometric cues that detect structural variations across frames. To capture
fine-grained texture, each motion Gaussian anchors and dynamically activates a
set of local appearance Gaussians, which are non-rigidly warped to the current
frame to provide strong initialization and significantly reduce training time.
This activation mechanism enables efficient modeling of detailed textures and
maintains temporal coherence, allowing high-fidelity rendering even under
challenging scenarios such as changing clothes. To enable seamless integration
into codec-based volumetric formats, we introduce a global Gaussian Lookup
Table that records the lifespan of each Gaussian and organizes attributes into
a lifespan-aware 2D layout. This structure aligns naturally with standard video
codecs and supports up to 40 compression. TaoGS provides a unified, adaptive
solution for scalable volumetric video under topological variation, capturing
moments where "elegance in motion" and "Power in Stillness", delivering
immersive experiences that harmonize with the physical world.

### 2. [Neural Cone Radiosity for Interactive Global Illumination with Glossy Materials](http://arxiv.org/pdf/2509.07522v1)

Authors: Jierui Ren, Haojie Jin, Bo Pang, Yisong Chen, Guoping Wang, Sheng Li

Modeling of high-frequency outgoing radiance distributions has long been a
key challenge in rendering, particularly for glossy material. Such
distributions concentrate radiative energy within a narrow lobe and are highly
sensitive to changes in view direction. However, existing neural radiosity
methods, which primarily rely on positional feature encoding, exhibit notable
limitations in capturing these high-frequency, strongly view-dependent radiance
distributions. To address this, we propose a highly-efficient approach by
reflectance-aware ray cone encoding based on the neural radiosity framework,
named neural cone radiosity. The core idea is to employ a pre-filtered
multi-resolution hash grid to accurately approximate the glossy BSDF lobe,
embedding view-dependent reflectance characteristics directly into the encoding
process through continuous spatial aggregation. Our design not only
significantly improves the network's ability to model high-frequency reflection
distributions but also effectively handles surfaces with a wide range of
glossiness levels, from highly glossy to low-gloss finishes. Meanwhile, our
method reduces the network's burden in fitting complex radiance distributions,
allowing the overall architecture to remain compact and efficient.
Comprehensive experimental results demonstrate that our method consistently
produces high-quality, noise-free renderings in real time under various
glossiness conditions, and delivers superior fidelity and realism compared to
baseline approaches.

### 3. [ReShape: a Collaborative Art Experience](http://arxiv.org/pdf/2509.07643v1)

Authors: Hugo Parlier, Bruno Teheux

This article describes a project called ReShape in which we created and
designed a crowdsourced art initiative, inspired and powered by mathematics.

### 4. [dciWebMapper2: Enhancing the dciWebMapper framework toward integrated, interactive visualization of linked multi-type maps, charts, and spatial statistics and analysis](http://arxiv.org/pdf/2509.07897v1)

Authors: Sarigai Sarigai, Liping Yang, Katie Slack, Carolyn Fish, Michaela Buenemann, Qiusheng Wu, Yan Lin, Joseph A. Cook, David Jacobs

As interactive web-based geovisualization becomes increasingly vital across
disciplines, there is a growing need for open-source frameworks that support
dynamic, multi-attribute spatial analysis and accessible design. This paper
introduces dciWebMapper2, a significant expansion of the original dciWebMapper
framework, designed to enable exploratory analysis across domains such as
climate justice, food access, and social vulnerability. The enhanced framework
integrates multiple map types, including choropleth, proportional symbol, small
multiples, and heatmaps, with linked statistical charts (e.g., scatter plots,
boxplots) and time sliders, all within a coordinated-view environment.
Dropdown-based controls allow flexible, high-dimensional comparisons while
maintaining visual clarity. Grounded in cartographic and information
visualization principles, dciWebMapper2 is fully open-source, self-contained,
and server-free, supporting modularity, reproducibility, and long-term
sustainability. Three applied use cases demonstrate its adaptability and
potential to democratize interactive web cartography. This work offers a
versatile foundation for inclusive spatial storytelling and transparent
geospatial analysis in research, education, and civic engagement.

### Computer Science and Game Theory

### 1. [Persuading Agents in Opinion Formation Games](http://arxiv.org/pdf/2509.07520v1)

Authors: Martin Hoefer, Tim Koglin, Tolga Tel

Prominent opinion formation models such as the one by Friedkin and Johnsen
(FJ) concentrate on the effects of peer pressure on public opinions. In
practice, opinion formation is also based on information about the state of the
world and persuasion efforts. In this paper, we analyze an approach of Bayesian
persuasion in the FJ model. There is an unknown state of the world that
influences the preconceptions of n agents. A sender S can (partially) reveal
information about the state to all agents. The agents update their
preconceptions, and an equilibrium of public opinions emerges. We propose
algorithms for the sender to reveal information in order to optimize various
aspects of the emerging equilibrium. For many natural sender objectives, we
show that there are simple optimal strategies. We then focus on a general class
of range-based objectives with desired opinion ranges for each agent. We
provide efficient algorithms in several cases, e.g., when the matrix of
preconceptions in all states has constant rank, or when there is only a
polynomial number of range combinations that lead to positive value for S. This
generalizes, e.g., instances with a constant number of states and/or agents, or
instances with a logarithmic number of ranges. In general, we show that
subadditive range-based objectives allow a simple n-approximation, and even for
additive ones, obtaining an $n^{1-c}$-approximation is NP-hard, for any
constant $c > 0$.

### 2. [Inference of Intrinsic Rewards and Fairness in Multi-Agent Systems](http://arxiv.org/pdf/2509.07650v1)

Authors: Victor Villin, Christos Dimitrakakis

From altruism to antagonism, fairness plays a central role in social
interactions. But can we truly understand how fair someone is, especially
without explicit knowledge of their preferences? We cast this challenge as a
multi-agent inverse reinforcement learning problem, explicitly structuring
rewards to reflect how agents value the welfare of others. We introduce novel
Bayesian strategies, reasoning about the optimality of demonstrations and
characterisation of equilibria in general-sum Markov games. Our experiments,
spanning randomised environments and a collaborative cooking task, reveal that
coherent notions of fairness can be reliably inferred from demonstrations.
Furthermore, when isolating fairness components, we obtain a disentangled
understanding of agents preferences. Crucially, we unveil that by placing
agents in different groups, we can force them to exhibit new facets of their
reward structures, cutting through ambiguity to answer the central question:
who is being fair?

### 3. [City Sampling for Citizens' Assemblies](http://arxiv.org/pdf/2509.07557v1)

Authors: Paul Gölz, Jan Maly, Ulrike Schmidt-Kraepelin, Markus Utke, Philipp C. Verpoort

In citizens' assemblies, a group of constituents is randomly selected to
weigh in on policy issues. We study a two-stage sampling problem faced by
practitioners in countries such as Germany, in which constituents' contact
information is stored at a municipal level. As a result, practitioners can only
select constituents from a bounded number of cities ex post, while ensuring
equal selection probability for constituents ex ante.
  We develop several algorithms for this problem. Although minimizing the
number of contacted cities is NP-hard, we provide a pseudo-polynomial time
algorithm and an additive 1-approximation, both based on separation oracles for
a linear programming formulation. Recognizing that practical objectives go
beyond minimizing city count, we further introduce a simple and more
interpretable greedy algorithm, which additionally satisfies an ex-post
monotonicity property and achieves an additive 2-approximation. Finally, we
explore a notion of ex-post proportionality, for which we propose two practical
algorithms: an optimal algorithm based on column generation and integer linear
programming and a simple heuristic creating particularly transparent
distributions. We evaluate these algorithms on data from Germany, and plan to
deploy them in cooperation with a leading nonprofit organization in this space.

### 4. [Language Self-Play For Data-Free Training](http://arxiv.org/pdf/2509.07414v1)

Authors: Jakub Grudzien Kuba, Mengting Gu, Qi Ma, Yuandong Tian, Vijai Mohan

Large language models (LLMs) have advanced rapidly in recent years, driven by
scale, abundant high-quality training data, and reinforcement learning. Yet
this progress faces a fundamental bottleneck: the need for ever more data from
which models can continue to learn. In this work, we propose a reinforcement
learning approach that removes this dependency by enabling models to improve
without additional data. Our method leverages a game-theoretic framework of
self-play, where a model's capabilities are cast as performance in a
competitive game and stronger policies emerge by having the model play against
itself - a process we call Language Self-Play (LSP). Experiments with
Llama-3.2-3B-Instruct on instruction-following benchmarks show that pretrained
models can not only enhance their performance on challenging tasks through
self-play alone, but can also do so more effectively than data-driven
baselines.

### 5. [Smart Fast Finish: Preventing Overdelivery via Daily Budget Pacing at DoorDash](http://arxiv.org/pdf/2509.07929v1)

Authors: Rohan Garg, Yongjin Xiao, Jason, Yang, Mandar Rahurkar

We present a budget pacing feature called Smart Fast Finish (SFF). SFF builds
upon the industry standard Fast Finish (FF) feature in budget pacing systems
that depletes remaining advertising budget as quickly as possible towards the
end of some fixed time period. SFF dynamically updates system parameters such
as start time and throttle rate depending on historical ad-campaign data. SFF
is currently in use at DoorDash, one of the largest delivery platforms in the
US, and is part of its budget pacing system. We show via online budget-split
experimentation data and offline simulations that SFF is a robust solution for
overdelivery mitigation when pacing budget.

### 6. [Multi-Topic Projected Opinion Dynamics for Resource Allocation](http://arxiv.org/pdf/2509.07847v1)

Authors: Prashil Wankhede, Nirabhra Mandal, Sonia Martínez, Pavankumar Tallapragada

We propose a model of opinion formation on resource allocation among multiple
topics by multiple agents, who are subject to hard budget constraints. We
define a utility function for each agent and then derive a projected dynamical
system model of opinion evolution assuming that each agent myopically seeks to
maximize its utility subject to its constraints. Inter-agent coupling arises
from an undirected social network, while inter-topic coupling arises from
resource constraints. We show that opinions always converge to the equilibrium
set. For special networks with very weak antagonistic relations, the opinions
converge to a unique equilibrium point. We further show that the underlying
opinion formation game is a potential game. We relate the equilibria of the
dynamics and the Nash equilibria of the game and characterize the unique Nash
equilibrium for networks with no antagonistic relations. Finally, simulations
illustrate our findings.

### Human-Computer Interaction

### 1. [In the Queue: Understanding How Reddit Moderators Use the Modqueue](http://arxiv.org/pdf/2509.07314v1)

Authors: Tanvi Bajpai, Eshwar Chandrasekharan

On Reddit, the moderation queue (modqueue) is a primary interface for
moderators to review reported content. Despite its central role in Reddit's
community-reliant moderation model, little is known about how moderators
actually use it in practice. To address this gap, we surveyed 110 moderators,
who collectively oversee more than 400 unique subreddits, and asked them about
their usage of the modqueue. Modqueue practices vary widely: some moderators
approach it as a daily checklist, others as a hub to infer community-wide
patterns, and many still find the queue insufficient to inform their moderation
decisions. We also identify persistent challenges around review coordination,
inconsistent interface signals, and reliance on third-party tools. Taken
together, we show the modqueue is neither a one-size-fits-all solution nor
sufficient on its own for supporting moderator review. Our work highlights
design opportunities for more modular, integrated, and customizable platform
infrastructures that better support the diversity of moderator workflows.

### 2. [SpecifyUI: Supporting Iterative UI Design Intent Expression through Structured Specifications and Generative AI](http://arxiv.org/pdf/2509.07334v1)

Authors: Yunnong Chen, Chengwei Shi, Liuqing Chen

Large language models (LLMs) promise to accelerate UI design, yet current
tools struggle with two fundamentals: externalizing designers' intent and
controlling iterative change. We introduce SPEC, a structured, parameterized,
hierarchical intermediate representation that exposes UI elements as
controllable parameters. Building on SPEC, we present SpecifyUI, an interactive
system that extracts SPEC from UI references via region segmentation and
vision-language models, composes UIs across multiple sources, and supports
targeted edits at global, regional, and component levels. A multi-agent
generator renders SPEC into high-fidelity designs, closing the loop between
intent expression and controllable generation. Quantitative experiments show
SPEC-based generation more faithfully captures reference intent than
prompt-based baselines. In a user study with 16 professional designers,
SpecifyUI significantly outperformed Stitch on intent alignment, design
quality, controllability, and overall experience in human-AI co-creation. Our
results position SPEC as a specification-driven paradigm that shifts
LLM-assisted design from one-shot prompting to iterative, collaborative
workflows.

### 3. [Feed-O-Meter: Fostering Design Feedback Skills through Role-playing Interactions with AI Mentee](http://arxiv.org/pdf/2509.07424v1)

Authors: Hyunseung Lim, Dasom Choi, DaEun Choi, Sooyohn Nam, Hwajung Hong

Effective feedback, including critique and evaluation, helps designers
develop design concepts and refine their ideas, supporting informed
decision-making throughout the iterative design process. However, in
studio-based design courses, students often struggle to provide feedback due to
a lack of confidence and fear of being judged, which limits their ability to
develop essential feedback-giving skills. Recent advances in large language
models (LLMs) suggest that role-playing with AI agents can let learners engage
in multi-turn feedback without the anxiety of external judgment or the time
constraints of real-world settings. Yet prior studies have raised concerns that
LLMs struggle to behave like real people in role-play scenarios, diminishing
the educational benefits of these interactions. Therefore, designing AI-based
agents that effectively support learners in practicing and developing
intellectual reasoning skills requires more than merely assigning the target
persona's personality and role to the agent. By addressing these issues, we
present Feed-O-Meter, a novel system that employs carefully designed LLM-based
agents to create an environment in which students can practice giving design
feedback. The system enables users to role-play as mentors, providing feedback
to an AI mentee and allowing them to reflect on how that feedback impacts the
AI mentee's idea development process. A user study (N=24) indicated that
Feed-O-Meter increased participants' engagement and motivation through
role-switching and helped them adjust feedback to be more comprehensible for an
AI mentee. Based on these findings, we discuss future directions for designing
systems to foster feedback skills in design education.

### 4. [Social Media Clones: Exploring the Impact of Social Delegation with AI Clones through a Design Workbook Study](http://arxiv.org/pdf/2509.07502v1)

Authors: Jackie Liu, Mehrnoosh Sadat Shirvani, Hwajung Hong, Ig-Jae Kim, Dongwook Yoon

Social media clones are AI-powered social delegates of ourselves created
using our personal data. As our identities and online personas intertwine,
these technologies have the potential to greatly enhance our social media
experience. If mismanaged, however, these clones may also pose new risks to our
social reputation and online relationships. To set the foundation for a
productive and responsible integration, we set out to understand how social
media clones will impact our online behavior and interactions. We conducted a
series of semi-structured interviews introducing eight speculative clone
concepts to 32 social media users through a design workbook. Applying existing
work in AI-mediated communication in the context of social media, we found that
although clones can offer convenience and comfort, they can also threaten the
user's authenticity and increase skepticism within the online community. As a
result, users tend to behave more like their clones to mitigate discrepancies
and interaction breakdowns. These findings are discussed through the lens of
past literature in identity and impression management to highlight challenges
in the adoption of social media clones by the general public, and propose
design considerations for their successful integration into social media
platforms.

### 5. [Digital Twins for Extended Reality Tourism: User Experience Evaluation Across User Groups](http://arxiv.org/pdf/2509.07740v1)

Authors: Maximilian Warsinke, Francesco Vona, Tanja Kojić, Jan-Niklas Voigt-Antons, Sebastian Möller

This study evaluates the user experience (UX) in extended reality (XR)
tourism of two digital twin-based applications: an Augmented Reality Virtual
Tour (AR-VT) for enhanced on-site visits and a Virtual Reality Virtual Tour
(VR-VT) for remote exploration. Using a quantitative exploratory approach, 84
participants from Spain and Germany, divided into three sample groups, assessed
UX, task load, presence, cybersickness, and emotional response through
standardized questionnaires. Findings indicate that both applications provided
a low task load and high enjoyment. The VR-based tour enhanced presence but
posed usability and cybersickness challenges, while the AR-based tour achieved
high UX ratings, with qualitative feedback suggesting areas for refinement.
Correlation analysis revealed significant relationships between age, prior XR
experience, and technological affinity with the measured metrics for both
applications. These results highlight the importance of well-designed
experiences tailored to XR novices, reinforcing the critical role of UX in
digital twin-based XR tourism.

### 6. [LLMs in Wikipedia: Investigating How LLMs Impact Participation in Knowledge Communities](http://arxiv.org/pdf/2509.07819v1)

Authors: Moyan Zhou, Soobin Cho, Loren Terveen

Large language models (LLMs) are reshaping knowledge production as community
members increasingly incorporate them into their contribution workflows.
However, participating in knowledge communities involves more than just
contributing content - it is also a deeply social process. While communities
must carefully consider appropriate and responsible LLM integration, the
absence of concrete norms has left individual editors to experiment and
navigate LLM use on their own. Understanding how LLMs influence community
participation is therefore critical in shaping future norms and supporting
effective adoption. To address this gap, we investigated Wikipedia, one of the
largest knowledge production communities, to understand 1) how LLMs influence
the ways editors contribute content, 2) what strategies editors leverage to
align LLM outputs with community norms, and 3) how other editors in the
community respond to LLM-assisted contributions. Through interviews with 16
Wikipedia editors who had used LLMs for their edits, we found that 1) LLMs
affected the content contributions for experienced and new editors differently;
2) aligning LLM outputs with community norms required tacit knowledge that
often challenged newcomers; and 3) as a result, other editors responded to
LLM-assisted edits differently depending on the editors' expertise level. Based
on these findings, we challenge existing models of newcomer involvement and
propose design implications for LLMs that support community engagement through
scaffolding, teaching, and context awareness.

### 7. [NeuroGaze: A Hybrid EEG and Eye-Tracking Brain-Computer Interface for Hands-Free Interaction in Virtual Reality](http://arxiv.org/pdf/2509.07863v1)

Authors: Kyle Coutray, Wanyea Barbel, Zack Groth, Joseph J LaViola Jr

Brain-Computer Interfaces (BCIs) have traditionally been studied in clinical
and laboratory contexts, but the rise of consumer-grade devices now allows
exploration of their use in daily activities. Virtual reality (VR) provides a
particularly relevant domain, where existing input methods often force
trade-offs between speed, accuracy, and physical effort. This study introduces
NeuroGaze, a hybrid interface combining electroencephalography (EEG) with eye
tracking to enable hands-free interaction in immersive VR. Twenty participants
completed a 360{\deg} cube-selection task using three different input methods:
VR controllers, gaze combined with a pinch gesture, and NeuroGaze. Performance
was measured by task completion time and error rate, while workload was
evaluated using the NASA Task Load Index (NASA-TLX). NeuroGaze successfully
supported target selection with off-the-shelf hardware, producing fewer errors
than the alternative methods but requiring longer completion times, reflecting
a classic speed-accuracy tradeoff. Workload analysis indicated reduced physical
demand for NeuroGaze compared to controllers, though overall ratings and user
preferences were mixed. These findings demonstrate the feasibility of hybrid
EEG+gaze systems for everyday VR use, highlighting their ergonomic benefits and
inclusivity potential. Although not yet competitive in speed, NeuroGaze points
toward a practical role for consumer-grade BCIs in accessibility and
long-duration applications, and underscores the need for improved EEG signal
processing and adaptive multimodal integration to enhance future performance.

### 8. [An Enactivist Approach to Human-Computer Interaction: Bridging the Gap Between Human Agency and Affordances](http://arxiv.org/pdf/2509.07871v1)

Authors: Angjelin Hila

Emerging paradigms in XR, AI, and BCI contexts necessitate novel theoretical
frameworks for understanding human autonomy and agency in HCI. Drawing from
enactivist theories of cognition, we conceptualize human agents as
self-organizing, operationally closed systems that actively enact their
cognitive domains through dynamic interaction with their environments. To
develop measurable variables aligned with this framework, we introduce
"feelings of agency" (FoA) as an alternative to the established construct of
"sense of agency" (SoA), refining Synofzyk's multifactorial weighting model and
offering a novel conceptual pathway for overcoming gaps in the dominant
comparator model. We define FoA as comprising two subconstructs: affective
engagement and volitional attention, which we operationalize through integrated
neurodynamic indicators (valence, arousal, cross frequency coupling within the
dorsal attention system) and first-person phenomenological reports. We argue
that these neurophenomenological indicators provide richer, more actionable
insights for digital affordance design, particularly in XR, BCI, Human AI
Interaction (HAX), and generative AI environments. Our framework aims to inform
and inspire design parameters that significantly enhance human agency in
rapidly evolving interactive domains.

### 9. [Timing the Message: Language-Based Notifications for Time-Critical Assistive Settings](http://arxiv.org/pdf/2509.07438v1)

Authors: Ya-Chuan Hsu, Jonathan DeCastro, Andrew Silva, Guy Rosman

In time-critical settings such as assistive driving, assistants often rely on
alerts or haptic signals to prompt rapid human attention, but these cues
usually leave humans to interpret situations and decide responses
independently, introducing potential delays or ambiguity in meaning.
Language-based assistive systems can instead provide instructions backed by
context, offering more informative guidance. However, current approaches (e.g.,
social assistive robots) largely prioritize content generation while
overlooking critical timing factors such as verbal conveyance duration, human
comprehension delays, and subsequent follow-through duration. These timing
considerations are crucial in time-critical settings, where even minor delays
can substantially affect outcomes. We aim to study this inherent trade-off
between timeliness and informativeness by framing the challenge as a sequential
decision-making problem using an augmented-state Markov Decision Process. We
design a framework combining reinforcement learning and a generated offline
taxonomy dataset, where we balance the trade-off while enabling a scalable
taxonomy dataset generation pipeline. Empirical evaluation with synthetic
humans shows our framework improves success rates by over 40% compared to
methods that ignore time delays, while effectively balancing timeliness and
informativeness. It also exposes an often-overlooked trade-off between these
two factors, opening new directions for optimizing communication in
time-critical human-AI assistance.

### 10. [Temporal Counterfactual Explanations of Behaviour Tree Decisions](http://arxiv.org/pdf/2509.07674v1)

Authors: Tamlin Love, Antonio Andriella, Guillem Alenyà

Explainability is a critical tool in helping stakeholders understand robots.
In particular, the ability for robots to explain why they have made a
particular decision or behaved in a certain way is useful in this regard.
Behaviour trees are a popular framework for controlling the decision-making of
robots and other software systems, and thus a natural question to ask is
whether or not a system driven by a behaviour tree is capable of answering
"why" questions. While explainability for behaviour trees has seen some prior
attention, no existing methods are capable of generating causal, counterfactual
explanations which detail the reasons for robot decisions and behaviour.
Therefore, in this work, we introduce a novel approach which automatically
generates counterfactual explanations in response to contrastive "why"
questions. Our method achieves this by first automatically building a causal
model from the structure of the behaviour tree as well as domain knowledge
about the state and individual behaviour tree nodes. The resultant causal model
is then queried and searched to find a set of diverse counterfactual
explanations. We demonstrate that our approach is able to correctly explain the
behaviour of a wide range of behaviour tree structures and states. By being
able to answer a wide range of causal queries, our approach represents a step
towards more transparent, understandable and ultimately trustworthy robotic
systems.

### Information Retrieval

### 1. [Multi-view-guided Passage Reranking with Large Language Models](http://arxiv.org/pdf/2509.07485v1)

Authors: Jeongwoo Na, Jun Kwon, Eunseong Choi, Jongwuk Lee

Recent advances in large language models (LLMs) have shown impressive
performance in passage reranking tasks. Despite their success, LLM-based
methods still face challenges in efficiency and sensitivity to external biases.
(1) Existing models rely mostly on autoregressive generation and sliding window
strategies to rank passages, which incur heavy computational overhead as the
number of passages increases. (2) External biases, such as position or
selection bias, hinder the model's ability to accurately represent passages and
increase input-order sensitivity. To address these limitations, we introduce a
novel passage reranking model, called Multi-View-guided Passage Reranking
(MVP). MVP is a non-generative LLM-based reranking method that encodes
query-passage information into diverse view embeddings without being influenced
by external biases. For each view, it combines query-aware passage embeddings
to produce a distinct anchor vector, which is then used to directly compute
relevance scores in a single decoding step. In addition, it employs an
orthogonal loss to make the views more distinctive. Extensive experiments
demonstrate that MVP, with just 220M parameters, matches the performance of
much larger 7B-scale fine-tuned models while achieving a 100x reduction in
inference latency. Notably, the 3B-parameter variant of MVP achieves
state-of-the-art performance on both in-domain and out-of-domain benchmarks.
The source code is available at: https://github.com/bulbna/MVP

### 2. [ELEC: Efficient Large Language Model-Empowered Click-Through Rate Prediction](http://arxiv.org/pdf/2509.07594v1)

Authors: Rui Dong, Wentao Ouyang, Xiangzheng Liu

Click-through rate (CTR) prediction plays an important role in online
advertising systems. On the one hand, traditional CTR prediction models capture
the collaborative signals in tabular data via feature interaction modeling, but
they lose semantics in text. On the other hand, Large Language Models (LLMs)
excel in understanding the context and meaning behind text, but they face
challenges in capturing collaborative signals and they have long inference
latency. In this paper, we aim to leverage the benefits of both types of models
and pursue collaboration, semantics and efficiency. We present ELEC, which is
an Efficient LLM-Empowered CTR prediction framework. We first adapt an LLM for
the CTR prediction task. In order to leverage the ability of the LLM but
simultaneously keep efficiency, we utilize the pseudo-siamese network which
contains a gain network and a vanilla network. We inject the high-level
representation vector generated by the LLM into a collaborative CTR model to
form the gain network such that it can take advantage of both tabular modeling
and textual modeling. However, its reliance on the LLM limits its efficiency.
We then distill the knowledge from the gain network to the vanilla network on
both the score level and the representation level, such that the vanilla
network takes only tabular data as input, but can still generate comparable
performance as the gain network. Our approach is model-agnostic. It allows for
the integration with various existing LLMs and collaborative CTR models.
Experiments on real-world datasets demonstrate the effectiveness and efficiency
of ELEC for CTR prediction.

### 3. [Towards End-to-End Model-Agnostic Explanations for RAG Systems](http://arxiv.org/pdf/2509.07620v1)

Authors: Viju Sudhi, Sinchana Ramakanth Bhat, Max Rudat, Roman Teucher, Nicolas Flores-Herr

Retrieval Augmented Generation (RAG) systems, despite their growing
popularity for enhancing model response reliability, often struggle with
trustworthiness and explainability. In this work, we present a novel, holistic,
model-agnostic, post-hoc explanation framework leveraging perturbation-based
techniques to explain the retrieval and generation processes in a RAG system.
We propose different strategies to evaluate these explanations and discuss the
sufficiency of model-agnostic explanations in RAG systems. With this work, we
further aim to catalyze a collaborative effort to build reliable and
explainable RAG systems.

### 4. [A Survey of Long-Document Retrieval in the PLM and LLM Era](http://arxiv.org/pdf/2509.07759v1)

Authors: Minghan Li, Miyang Luo, Tianrui Lv, Yishuai Zhang, Siqi Zhao, Ercong Nie, Guodong Zhou

The proliferation of long-form documents presents a fundamental challenge to
information retrieval (IR), as their length, dispersed evidence, and complex
structures demand specialized methods beyond standard passage-level techniques.
This survey provides the first comprehensive treatment of long-document
retrieval (LDR), consolidating methods, challenges, and applications across
three major eras. We systematize the evolution from classical lexical and early
neural models to modern pre-trained (PLM) and large language models (LLMs),
covering key paradigms like passage aggregation, hierarchical encoding,
efficient attention, and the latest LLM-driven re-ranking and retrieval
techniques. Beyond the models, we review domain-specific applications,
specialized evaluation resources, and outline critical open challenges such as
efficiency trade-offs, multimodal alignment, and faithfulness. This survey aims
to provide both a consolidated reference and a forward-looking agenda for
advancing long-document retrieval in the era of foundation models.

### 5. [Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey](http://arxiv.org/pdf/2509.07794v1)

Authors: Minghan Li, Xinxuan Lv, Junjie Zou, Tongna Chen, Chao Zhang, Suchao An, Ercong Nie, Guodong Zhou

Modern information retrieval (IR) must bridge short, ambiguous queries and
ever more diverse, rapidly evolving corpora. Query Expansion (QE) remains a key
mechanism for mitigating vocabulary mismatch, but the design space has shifted
markedly with pre-trained language models (PLMs) and large language models
(LLMs). This survey synthesizes the field from three angles: (i) a
four-dimensional framework of query expansion - from the point of injection
(explicit vs. implicit QE), through grounding and interaction (knowledge bases,
model-internal capabilities, multi-turn retrieval) and learning alignment, to
knowledge graph-based argumentation; (ii) a model-centric taxonomy spanning
encoder-only, encoder-decoder, decoder-only, instruction-tuned, and
domain/multilingual variants, highlighting their characteristic affordances for
QE (contextual disambiguation, controllable generation, zero-/few-shot
reasoning); and (iii) practice-oriented guidance on where and how neural QE
helps in first-stage retrieval, multi-query fusion, re-ranking, and
retrieval-augmented generation (RAG). We compare traditional query expansion
with PLM/LLM-based methods across seven key aspects, and we map applications
across web search, biomedicine, e-commerce, open-domain QA/RAG, conversational
and code search, and cross-lingual settings. The review distills design
grounding and interaction, alignment/distillation (SFT/PEFT/DPO), and KG
constraints - as robust remedies to topic drift and hallucination. We conclude
with an agenda on quality control, cost-aware invocation, domain/temporal
adaptation, evaluation beyond end-task metrics, and fairness/privacy.
Collectively, these insights provide a principled blueprint for selecting and
combining QE techniques under real-world constraints.

### 6. [KLIPA: A Knowledge Graph and LLM-Driven QA Framework for IP Analysis](http://arxiv.org/pdf/2509.07860v1)

Authors: Guanzhi Deng, Yi Xie, Yu-Keung Ng, Mingyang Liu, Peijun Zheng, Jie Liu, Dapeng Wu, Yinqiao Li, Linqi Song

Effectively managing intellectual property is a significant challenge.
Traditional methods for patent analysis depend on labor-intensive manual
searches and rigid keyword matching. These approaches are often inefficient and
struggle to reveal the complex relationships hidden within large patent
datasets, hindering strategic decision-making. To overcome these limitations,
we introduce KLIPA, a novel framework that leverages a knowledge graph and a
large language model (LLM) to significantly advance patent analysis. Our
approach integrates three key components: a structured knowledge graph to map
explicit relationships between patents, a retrieval-augmented generation(RAG)
system to uncover contextual connections, and an intelligent agent that
dynamically determines the optimal strategy for resolving user queries. We
validated KLIPA on a comprehensive, real-world patent database, where it
demonstrated substantial improvements in knowledge extraction, discovery of
novel connections, and overall operational efficiency. This combination of
technologies enhances retrieval accuracy, reduces reliance on domain experts,
and provides a scalable, automated solution for any organization managing
intellectual property, including technology corporations and legal firms,
allowing them to better navigate the complexities of strategic innovation and
competitive intelligence.

### 7. [MEGG: Replay via Maximally Extreme GGscore in Incremental Learning for Neural Recommendation Models](http://arxiv.org/pdf/2509.07319v1)

Authors: Yunxiao Shi, Shuo Yang, Haimin Zhang, Li Wang, Yongze Wang, Qiang Wu, Min Xu

Neural Collaborative Filtering models are widely used in recommender systems
but are typically trained under static settings, assuming fixed data
distributions. This limits their applicability in dynamic environments where
user preferences evolve. Incremental learning offers a promising solution, yet
conventional methods from computer vision or NLP face challenges in
recommendation tasks due to data sparsity and distinct task paradigms. Existing
approaches for neural recommenders remain limited and often lack
generalizability. To address this, we propose MEGG, Replay Samples with
Maximally Extreme GGscore, an experience replay based incremental learning
framework. MEGG introduces GGscore, a novel metric that quantifies sample
influence, enabling the selective replay of highly influential samples to
mitigate catastrophic forgetting. Being model-agnostic, MEGG integrates
seamlessly across architectures and frameworks. Experiments on three neural
models and four benchmark datasets show superior performance over
state-of-the-art baselines, with strong scalability, efficiency, and
robustness. Implementation will be released publicly upon acceptance.

### 8. [FLeW: Facet-Level and Adaptive Weighted Representation Learning of Scientific Documents](http://arxiv.org/pdf/2509.07531v1)

Authors: Zheng Dou, Deqing Wang, Fuzhen Zhuang, Jian Ren, Yanlin Hu

Scientific document representation learning provides powerful embeddings for
various tasks, while current methods face challenges across three approaches.
1) Contrastive training with citation-structural signals underutilizes citation
information and still generates single-vector representations. 2) Fine-grained
representation learning, which generates multiple vectors at the sentence or
aspect level, requires costly integration and lacks domain generalization. 3)
Task-aware learning depends on manually predefined task categorization,
overlooking nuanced task distinctions and requiring extra training data for
task-specific modules. To address these problems, we propose a new method that
unifies the three approaches for better representations, namely FLeW.
Specifically, we introduce a novel triplet sampling method that leverages
citation intent and frequency to enhance citation-structural signals for
training. Citation intents (background, method, result), aligned with the
general structure of scientific writing, facilitate a domain-generalized facet
partition for fine-grained representation learning. Then, we adopt a simple
weight search to adaptively integrate three facet-level embeddings into a
task-specific document embedding without task-aware fine-tuning. Experiments
show the applicability and robustness of FLeW across multiple scientific tasks
and fields, compared to prior models.

### 9. [ALLabel: Three-stage Active Learning for LLM-based Entity Recognition using Demonstration Retrieval](http://arxiv.org/pdf/2509.07512v1)

Authors: Zihan Chen, Lei Shi, Weize Wu, Qiji Zhou, Yue Zhang

Many contemporary data-driven research efforts in the natural sciences, such
as chemistry and materials science, require large-scale, high-performance
entity recognition from scientific datasets. Large language models (LLMs) have
increasingly been adopted to solve the entity recognition task, with the same
trend being observed on all-spectrum NLP tasks. The prevailing entity
recognition LLMs rely on fine-tuned technology, yet the fine-tuning process
often incurs significant cost. To achieve a best performance-cost trade-off, we
propose ALLabel, a three-stage framework designed to select the most
informative and representative samples in preparing the demonstrations for LLM
modeling. The annotated examples are used to construct a ground-truth retrieval
corpus for LLM in-context learning. By sequentially employing three distinct
active learning strategies, ALLabel consistently outperforms all baselines
under the same annotation budget across three specialized domain datasets.
Experimental results also demonstrate that selectively annotating only 5\%-10\%
of the dataset with ALLabel can achieve performance comparable to the method
annotating the entire dataset. Further analyses and ablation studies verify the
effectiveness and generalizability of our proposal.

### 10. [SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP](http://arxiv.org/pdf/2509.07801v1)

Authors: Decheng Duan, Yingyi Zhang, Jitong Peng, Chengzhi Zhang

Structured information extraction from scientific literature is crucial for
capturing core concepts and emerging trends in specialized fields. While
existing datasets aid model development, most focus on specific publication
sections due to domain complexity and the high cost of annotating scientific
texts. To address this limitation, we introduce SciNLP - a specialized
benchmark for full-text entity and relation extraction in the Natural Language
Processing (NLP) domain. The dataset comprises 60 manually annotated full-text
NLP publications, covering 7,072 entities and 1,826 relations. Compared to
existing research, SciNLP is the first dataset providing full-text annotations
of entities and their relationships in the NLP domain. To validate the
effectiveness of SciNLP, we conducted comparative experiments with similar
datasets and evaluated the performance of state-of-the-art supervised models on
this dataset. Results reveal varying extraction capabilities of existing models
across academic texts of different lengths. Cross-comparisons with existing
datasets show that SciNLP achieves significant performance improvements on
certain baseline models. Using models trained on SciNLP, we implemented
automatic construction of a fine-grained knowledge graph for the NLP domain.
Our KG has an average node degree of 3.2 per entity, indicating rich semantic
topological information that enhances downstream applications. The dataset is
publicly available at https://github.com/AKADDC/SciNLP.

### Machine Learning

### 1. [CancerGUIDE: Cancer Guideline Understanding via Internal Disagreement Estimation](http://arxiv.org/pdf/2509.07325v1)

Authors: Alyssa Unell, Noel C. F. Codella, Sam Preston, Peniel Argaw, Wen-wai Yim, Zelalem Gero, Cliff Wong, Rajesh Jena, Eric Horvitz, Amanda K. Hall, Ruican Rachel Zhong, Jiachen Li, Shrey Jain, Mu Wei, Matthew Lungren, Hoifung Poon

The National Comprehensive Cancer Network (NCCN) provides evidence-based
guidelines for cancer treatment. Translating complex patient presentations into
guideline-compliant treatment recommendations is time-intensive, requires
specialized expertise, and is prone to error. Advances in large language model
(LLM) capabilities promise to reduce the time required to generate treatment
recommendations and improve accuracy. We present an LLM agent-based approach to
automatically generate guideline-concordant treatment trajectories for patients
with non-small cell lung cancer (NSCLC). Our contributions are threefold.
First, we construct a novel longitudinal dataset of 121 cases of NSCLC patients
that includes clinical encounters, diagnostic results, and medical histories,
each expertly annotated with the corresponding NCCN guideline trajectories by
board-certified oncologists. Second, we demonstrate that existing LLMs possess
domain-specific knowledge that enables high-quality proxy benchmark generation
for both model development and evaluation, achieving strong correlation
(Spearman coefficient r=0.88, RMSE = 0.08) with expert-annotated benchmarks.
Third, we develop a hybrid approach combining expensive human annotations with
model consistency information to create both the agent framework that predicts
the relevant guidelines for a patient, as well as a meta-classifier that
verifies prediction accuracy with calibrated confidence scores for treatment
recommendations (AUROC=0.800), a critical capability for communicating the
accuracy of outputs, custom-tailoring tradeoffs in performance, and supporting
regulatory compliance. This work establishes a framework for clinically viable
LLM-based guideline adherence systems that balance accuracy, interpretability,
and regulatory requirements while reducing annotation costs, providing a
scalable pathway toward automated clinical decision support.

### 2. [Conv4Rec: A 1-by-1 Convolutional AutoEncoder for User Profiling through Joint Analysis of Implicit and Explicit Feedbacks](http://arxiv.org/pdf/2509.07499v1)

Authors: Antoine Ledent, Petr Kasalický, Rodrigo Alves, Hady W. Lauw

We introduce a new convolutional AutoEncoder architecture for user modelling
and recommendation tasks with several improvements over the state of the art.
Firstly, our model has the flexibility to learn a set of associations and
combinations between different interaction types in a way that carries over to
each user and item. Secondly, our model is able to learn jointly from both the
explicit ratings and the implicit information in the sampling pattern (which we
refer to as `implicit feedback'). It can also make separate predictions for the
probability of consuming content and the likelihood of granting it a high
rating if observed. This not only allows the model to make predictions for both
the implicit and explicit feedback, but also increases the informativeness of
the predictions: in particular, our model can identify items which users would
not have been likely to consume naturally, but would be likely to enjoy if
exposed to them. Finally, we provide several generalization bounds for our
model, which to the best of our knowledge, are among the first generalization
bounds for auto-encoders in a Recommender Systems setting; we also show that
optimizing our loss function guarantees the recovery of the exact sampling
distribution over interactions up to a small error in total variation. In
experiments on several real-life datasets, we achieve state-of-the-art
performance on both the implicit and explicit feedback prediction tasks despite
relying on a single model for both, and benefiting from additional
interpretability in the form of individual predictions for the probabilities of
each possible rating.

### 3. [RoseCDL: Robust and Scalable Convolutional Dictionary Learning for Rare-event Detection](http://arxiv.org/pdf/2509.07523v1)

Authors: Jad Yehya, Mansour Benbakoura, Cédric Allain, Benoît Malezieux, Matthieu Kowalski, Thomas Moreau

Identifying recurring patterns and rare events in large-scale signals is a
fundamental challenge in fields such as astronomy, physical simulations, and
biomedical science. Convolutional Dictionary Learning (CDL) offers a powerful
framework for modeling local structures in signals, but its use for detecting
rare or anomalous events remains largely unexplored. In particular, CDL faces
two key challenges in this setting: high computational cost and sensitivity to
artifacts and outliers. In this paper, we introduce RoseCDL, a scalable and
robust CDL algorithm designed for unsupervised rare event detection in long
signals. RoseCDL combines stochastic windowing for efficient training on large
datasets with inline outlier detection to enhance robustness and isolate
anomalous patterns. This reframes CDL as a practical tool for event discovery
and characterization in real-world signals, extending its role beyond
traditional tasks like compression or denoising.

### 4. [K2-Think: A Parameter-Efficient Reasoning System](http://arxiv.org/pdf/2509.07604v1)

Authors: Zhoujun Cheng, Richard Fan, Shibo Hao, Taylor W. Killian, Haonan Li, Suqi Sun, Hector Ren, Alexander Moreno, Daqian Zhang, Tianjun Zhong, Yuxin Xiong, Yuanzhe Hu, Yutao Xie, Xudong Han, Yuqi Wang, Varad Pimpalkhute, Yonghao Zhuang, Aaryamonvikram Singh, Xuezhi Liang, Anze Xie, Jianshu She, Desai Fan, Chengqian Gao, Liqun Ma, Mikhail Yurochkin, John Maggs, Xuezhe Ma, Guowei He, Zhiting Hu, Zhengzhong Liu, Eric P. Xing

K2-Think is a reasoning system that achieves state-of-the-art performance
with a 32B parameter model, matching or surpassing much larger models like
GPT-OSS 120B and DeepSeek v3.1. Built on the Qwen2.5 base model, our system
shows that smaller models can compete at the highest levels by combining
advanced post-training and test-time computation techniques. The approach is
based on six key technical pillars: Long Chain-of-thought Supervised
Finetuning, Reinforcement Learning with Verifiable Rewards (RLVR), Agentic
planning prior to reasoning, Test-time Scaling, Speculative Decoding, and
Inference-optimized Hardware, all using publicly available open-source
datasets. K2-Think excels in mathematical reasoning, achieving state-of-the-art
scores on public benchmarks for open-source models, while also performing
strongly in other areas such as Code and Science. Our results confirm that a
more parameter-efficient model like K2-Think 32B can compete with
state-of-the-art systems through an integrated post-training recipe that
includes long chain-of-thought training and strategic inference-time
enhancements, making open-source reasoning systems more accessible and
affordable. K2-Think is freely available at k2think.ai, offering best-in-class
inference speeds of over 2,000 tokens per second per request via the Cerebras
Wafer-Scale Engine.

### 5. [Graph-based Integrated Gradients for Explaining Graph Neural Networks](http://arxiv.org/pdf/2509.07648v1)

Authors: Lachlan Simpson, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

Integrated Gradients (IG) is a common explainability technique to address the
black-box problem of neural networks. Integrated gradients assumes continuous
data. Graphs are discrete structures making IG ill-suited to graphs. In this
work, we introduce graph-based integrated gradients (GB-IG); an extension of IG
to graphs. We demonstrate on four synthetic datasets that GB-IG accurately
identifies crucial structural components of the graph used in classification
tasks. We further demonstrate on three prevalent real-world graph datasets that
GB-IG outperforms IG in highlighting important features for node classification
tasks.

### 6. [IBN: An Interpretable Bidirectional-Modeling Network for Multivariate Time Series Forecasting with Variable Missing](http://arxiv.org/pdf/2509.07725v1)

Authors: Shusen Ma, Tianhao Zhang, Qijiu Xia, Yun-Bo Zhao

Multivariate time series forecasting (MTSF) often faces challenges from
missing variables, which hinder conventional spatial-temporal graph neural
networks in modeling inter-variable correlations. While GinAR addresses
variable missing using attention-based imputation and adaptive graph learning
for the first time, it lacks interpretability and fails to capture more latent
temporal patterns due to its simple recursive units (RUs). To overcome these
limitations, we propose the Interpretable Bidirectional-modeling Network (IBN),
integrating Uncertainty-Aware Interpolation (UAI) and Gaussian kernel-based
Graph Convolution (GGCN). IBN estimates the uncertainty of reconstructed values
using MC Dropout and applies an uncertainty-weighted strategy to mitigate
high-risk reconstructions. GGCN explicitly models spatial correlations among
variables, while a bidirectional RU enhances temporal dependency modeling.
Extensive experiments show that IBN achieves state-of-the-art forecasting
performance under various missing-rate scenarios, providing a more reliable and
interpretable framework for MTSF with missing variables. Code is available at:
https://github.com/zhangth1211/NICLab-IBN.

### 7. [Predicting person-level injury severity using crash narratives: A balanced approach with roadway classification and natural language process techniques](http://arxiv.org/pdf/2509.07845v1)

Authors: Mohammad Zana Majidi, Sajjad Karimi, Teng Wang, Robert Kluger, Reginald Souleyrette

Predicting injuries and fatalities in traffic crashes plays a critical role
in enhancing road safety, improving emergency response, and guiding public
health interventions. This study investigates the added value of unstructured
crash narratives (written by police officers at the scene) when combined with
structured crash data to predict injury severity. Two widely used Natural
Language Processing (NLP) techniques, Term Frequency-Inverse Document Frequency
(TF-IDF) and Word2Vec, were employed to extract semantic meaning from the
narratives, and their effectiveness was compared. To address the challenge of
class imbalance, a K-Nearest Neighbors-based oversampling method was applied to
the training data prior to modeling. The dataset consists of crash records from
Kentucky spanning 2019 to 2023. To account for roadway heterogeneity, three
road classification schemes were used: (1) eight detailed functional classes
(e.g., Urban Two-Lane, Rural Interstate, Urban Multilane Divided), (2) four
broader paired categories (e.g., Urban vs. Rural, Freeway vs. Non-Freeway), and
(3) a unified dataset without classification. A total of 102 machine learning
models were developed by combining structured features and narrative-based
features using the two NLP techniques alongside three ensemble algorithms:
XGBoost, Random Forest, and AdaBoost. Results demonstrate that models
incorporating narrative data consistently outperform those relying solely on
structured data. Among all combinations, TF-IDF coupled with XGBoost yielded
the most accurate predictions in most subgroups. The findings highlight the
power of integrating textual and structured crash information to enhance
person-level injury prediction. This work offers a practical and adaptable
framework for transportation safety professionals to improve crash severity
modeling, guide policy decisions, and design more effective countermeasures.

### 8. [Addressing the Cold-Start Problem for Personalized Combination Drug Screening](http://arxiv.org/pdf/2509.07850v1)

Authors: Antoine de Mathelin, Christopher Tosh, Wesley Tansey

Personalizing combination therapies in oncology requires navigating an
immense space of possible drug and dose combinations, a task that remains
largely infeasible through exhaustive experimentation. Recent developments in
patient-derived models have enabled high-throughput ex vivo screening, but the
number of feasible experiments is limited. Further, a tight therapeutic window
makes gathering molecular profiling information (e.g. RNA-seq) impractical as a
means of guiding drug response prediction. This leads to a challenging
cold-start problem: how do we select the most informative combinations to test
early, when no prior information about the patient is available? We propose a
strategy that leverages a pretrained deep learning model built on historical
drug response data. The model provides both embeddings for drug combinations
and dose-level importance scores, enabling a principled selection of initial
experiments. We combine clustering of drug embeddings to ensure functional
diversity with a dose-weighting mechanism that prioritizes doses based on their
historical informativeness. Retrospective simulations on large-scale drug
combination datasets show that our method substantially improves initial
screening efficiency compared to baselines, offering a viable path for more
effective early-phase decision-making in personalized combination drug screens.

### 9. [Leveraging Support Vector Regression for Outcome Prediction in Personalized Ultra-fractionated Stereotactic Adaptive Radiotherapy](http://arxiv.org/pdf/2509.07872v1)

Authors: Yajun Yu, Steve Jiang, Robert Timmerman, Hao Peng

Personalized ultra-fractionated stereotactic adaptive radiotherapy (PULSAR)
is a novel treatment that delivers radiation in pulses of protracted intervals.
Accurate prediction of gross tumor volume (GTV) changes through regression
models has substantial prognostic value. This study aims to develop a
multi-omics based support vector regression (SVR) model for predicting GTV
change. A retrospective cohort of 39 patients with 69 brain metastases was
analyzed, based on radiomics (MRI images) and dosiomics (dose maps) features.
Delta features were computed to capture relative changes between two time
points. A feature selection pipeline using least absolute shrinkage and
selection operator (Lasso) algorithm with weight- or frequency-based ranking
criterion was implemented. SVR models with various kernels were evaluated using
the coefficient of determination (R2) and relative root mean square error
(RRMSE). Five-fold cross-validation with 10 repeats was employed to mitigate
the limitation of small data size. Multi-omics models that integrate radiomics,
dosiomics, and their delta counterparts outperform individual-omics models.
Delta-radiomic features play a critical role in enhancing prediction accuracy
relative to features at single time points. The top-performing model achieves
an R2 of 0.743 and an RRMSE of 0.022. The proposed multi-omics SVR model shows
promising performance in predicting continuous change of GTV. It provides a
more quantitative and personalized approach to assist patient selection and
treatment adjustment in PULSAR.

### 10. [A Survey of Graph Neural Networks for Drug Discovery: Recent Developments and Challenges](http://arxiv.org/pdf/2509.07887v1)

Authors: Katherine Berry, Liang Cheng

Graph Neural Networks (GNNs) have gained traction in the complex domain of
drug discovery because of their ability to process graph-structured data such
as drug molecule models. This approach has resulted in a myriad of methods and
models in published literature across several categories of drug discovery
research. This paper covers the research categories comprehensively with recent
papers, namely molecular property prediction, including drug-target binding
affinity prediction, drug-drug interaction study, microbiome interaction
prediction, drug repositioning, retrosynthesis, and new drug design, and
provides guidance for future work on GNNs for drug discovery.

### Neural and Evolutionary Computing

### 1. [Word2Spike: Poisson Rate Coding for Associative Memories and Neuromorphic Algorithms](http://arxiv.org/pdf/2509.07361v1)

Authors: Archit Kalra, Midhun Sadanand

Spiking neural networks offer a promising path toward energy-efficient,
brain-like associative memory. This paper introduces Word2Spike, a novel rate
coding mechanism that combines continuous word embeddings and neuromorphic
architectures. We develop a one-to-one mapping that converts multi-dimensional
word vectors into spike-based attractor states using Poisson processes. Using
BitNet b1.58 quantization, we maintain 97% semantic similarity of continuous
embeddings on SimLex-999 while achieving 100% reconstruction accuracy on 10,000
words from OpenAI's text-embedding-3-large. We preserve analogy performance
(100% of original embedding performance) even under intentionally introduced
noise, indicating a resilient mechanism for semantic encoding in neuromorphic
systems. Next steps include integrating the mapping with spiking transformers
and liquid state machines (resembling Hopfield Networks) for further
evaluation.

### Networking and Internet Architecture

### 1. [TEGRA: A Flexible & Scalable NextGen Mobile Core](http://arxiv.org/pdf/2509.07410v1)

Authors: Bilal Saleem, Omar Basit, Jiayi Meng, Iftekhar Alam, Ajay Thakur, Christian Maciocco, Muhammad Shahbaz, Y. Charlie Hu, Larry Peterson

To support emerging mobile use cases (e.g., AR/VR, autonomous driving, and
massive IoT), next-generation mobile cores for 5G and 6G are being
re-architected as service-based architectures (SBAs) running on both private
and public clouds. However, current performance optimization strategies for
scaling these cores still revert to traditional NFV-based techniques, such as
consolidating functions into rigid, monolithic deployments on dedicated
servers. This raises a critical question: Is there an inherent tradeoff between
flexibility and scalability in an SBA-based mobile core, where improving
performance (and resiliency) inevitably comes at the cost of one or the other?
  To explore this question, we introduce resilient SBA microservices design
patterns and state-management strategies, and propose TEGRA -- a
high-performance, flexible, and scalable SBA-based mobile core. By leveraging
the mobile core's unique position in the end-to-end internet ecosystem (i.e.,
at the last-mile edge), TEGRA optimizes performance without compromising
adaptability. Our evaluation demonstrates that TEGRA achieves significantly
lower latencies, processing requests 20x, 11x, and 1.75x faster than
traditional SBA core implementations -- free5GC, Open5GS, and Aether,
respectively -- all while matching the performance of state-of-the-art cores
(e.g., CoreKube) while retaining flexibility. Furthermore, it reduces the
complexity of deploying new features, requiring orders of magnitude fewer lines
of code (LoCs) compared to existing cores.

### 2. [Network-accelerated Active Messages](http://arxiv.org/pdf/2509.07431v1)

Authors: Md Ashfaqur Rahaman, Alireza Sanaee, Todd Thornley, Sebastiano Miano, Gianni Antichi, Brent E. Stephens, Ryan Stutsman

Remote Direct Memory Access (RDMA) improves host networking performance by
eliminating software and server CPU involvement. However, RDMA has a limited
set of operations, is difficult to program, and often requires multiple round
trips to perform simple application operations. Programmable SmartNICs provide
a different means to offload work from host CPUs to a NIC. This leaves
applications with the complex choice of embedding logic as RPC handlers at
servers, using RDMA's limited interface to access server structures via
client-side logic, or running some logic on SmartNICs. The best choice varies
between workloads and over time. To solve this dilemma, we present NAAM,
network-accelerated active messages. NAAM applications specify small, portable
eBPF functions associated with messages. Each message specifies what data it
accesses using an RDMA-like interface. NAAM runs at various places in the
network, including at clients, on server-attached SmartNICs, and server host
CPU cores. Due to eBPF's portability, the code associated with a message can be
run at any location. Hence, the NAAM runtime can dynamically steer any message
to execute its associated logic wherever it makes the most sense. To
demonstrate NAAM's flexibility, we built several applications, including the
MICA hash table and lookups from a Cell-style B-tree. With an NVIDIA
BlueField-2 SmartNIC and integrating its NIC-embedded switch, NAAM can run any
of these operations on client, server, and NIC cores, shifting load in tens of
milliseconds on server compute congestion. NAAM dynamically offloads up to 1.8
million MICA ops/s for YCSB-B and 750,000 Cell lookups/s from server CPUs.
Finally, whereas iPipe, the state-of-the-art SmartNIC offload framework, only
scales to 8 application offloads on BlueField-2, NAAM scales to hundreds of
application offloads with minimal impact on tail latency due to eBPF's low
overhead.

### 3. [Constraint-Compliant Network Optimization through Large Language Models](http://arxiv.org/pdf/2509.07492v1)

Authors: Youngjin Song, Wookjin Lee, Hong Ki Kim, Sang Hyun Lee

This work develops an LLM-based optimization framework ensuring strict
constraint satisfaction in network optimization. While LLMs possess contextual
reasoning capabilities, existing approaches often fail to enforce constraints,
causing infeasible solutions. Unlike conventional methods that address average
constraints, the proposed framework integrates a natural language-based input
encoding strategy to restrict the solution space and guarantee feasibility. For
multi-access edge computing networks, task allocation is optimized while
minimizing worst-case latency. Numerical evaluations demonstrate LLMs as a
promising tool for constraint-aware network optimization, offering insights
into their inference capabilities.

### 4. [FlexSAN: A Flexible Regenerative Satellite Access Network Architecture](http://arxiv.org/pdf/2509.07548v1)

Authors: Weize Kong, Chaoqun You, Xuming Pei, YueGao

The regenerative satellite access network (SAN) architecture deploys
next-generation NodeB (gNBs) on satellites to enable enhanced network
management capabilities. It supports two types of regenerative payload,
on-board gNB and on-board gNB-Distributed Unit (gNB-DU). Measurement results
based on our prototype implementation show that the on-board gNB offers lower
latency, while the on-board gNB-DU is more cost-effective, and there is often a
trade-off between Quality-of-Service (QoS) and operational expenditure (OPEX)
when choosing between the two payload types. However, current SAN
configurations are static and inflexible -- either deploying the full on-board
gNB or only the on-board gNB-DU. This rigidity can lead to resource waste or
poor user experiences. In this paper, we propose Flexible SAN (FlexSAN), an
adaptive satellite access network architecture that dynamically configures the
optimal regenerative payload based on real-time user demands. FlexSAN selects
the lowest OPEX payload configuration when all user demands are satisfied, and
otherwise maximizes the number of admitted users while ensuring QoS for
connected users. To address the computational complexity of dynamic payload
selection, we design an adaptive greedy heuristic algorithm. Extensive
experiments validate FlexSAN's effectiveness, showing a 36.1% average
improvement in user admission rates and a 15% OPEX reduction over static SANs.

### 5. [Making congestion control robust to per-packet load balancing in datacenters](http://arxiv.org/pdf/2509.07907v1)

Authors: Barak Gerstein, Mark Silberstein, Isaac Keslassy

Per-packet load-balancing approaches are increasingly deployed in datacenter
networks. However, their combination with existing congestion control
algorithms (CCAs) may lead to poor performance, and even state-of-the-art CCAs
can collapse due to duplicate ACKs. A typical approach to handle this collapse
is to make CCAs resilient to duplicate ACKs.
  In this paper, we first model the throughput collapse of a wide array of CCAs
when some of the paths are congested. We show that addressing duplicate ACKs is
insufficient. Instead, we explain that since CCAs are typically designed for
single-path routing, their estimation function focuses on the latest feedback
and mishandles feedback that reflects multiple paths. We propose to use a
median feedback that is more robust to the varying signals that come with
multiple paths. We introduce MSwift, which applies this principle to make
Google's Swift robust to multi-path routing while keeping its incast tolerance
and single-path performance. Finally, we demonstrate that MSwift improves the
99th-percentile FCT by up to 25\%, both with random packet spraying and
adaptive routing.

### 6. [Influence Maximization Considering Influence, Cost and Time](http://arxiv.org/pdf/2509.07625v1)

Authors: Mingyang Feng, Qi Zhao, Shan He, Yuhui Shi

Influence maximization has been studied for social network analysis, such as
viral marketing (advertising), rumor prevention, and opinion leader
identification. However, most studies neglect the interplay between influence
spread, cost efficiency, and temporal urgency. In practical scenarios such as
viral marketing and information campaigns, jointly optimizing Influence, Cost,
and Time is essential, yet remaining largely unaddressed in current literature.
To bridge the gap, this paper proposes a new multi-objective influence
maximization problem that simultaneously optimizes influence, cost, and time.
We show the intuitive and empirical evidence to prove the feasibility and
necessity of this multi-objective problem. We also develop an evolutionary
variable-length search algorithm that can effectively search for optimal node
combinations. The proposed EVEA algorithm outperforms all baselines, achieving
up to 19.3% higher hypervolume and 25 to 40% faster convergence across four
real-world networks, while maintaining a diverse and balanced Pareto front
among influence, cost, and time objectives.

### 7. [Quantum Computing for Large-scale Network Optimization: Opportunities and Challenges](http://arxiv.org/pdf/2509.07773v1)

Authors: Sebastian Macaluso, Giovanni Geraci, Elías F. Combarro, Sergi Abadal, Ioannis Arapakis, Sofia Vallecorsa, Eduard Alarcón

The complexity of large-scale 6G-and-beyond networks demands innovative
approaches for multi-objective optimization over vast search spaces, a task
often intractable. Quantum computing (QC) emerges as a promising technology for
efficient large-scale optimization. We present our vision of leveraging QC to
tackle key classes of problems in future mobile networks. By analyzing and
identifying common features, particularly their graph-centric representation,
we propose a unified strategy involving QC algorithms. Specifically, we outline
a methodology for optimization using quantum annealing as well as quantum
reinforcement learning. Additionally, we discuss the main challenges that QC
algorithms and hardware must overcome to effectively optimize future networks.

### Robotics

### 1. [Performance Characterization of a Point-Cloud-Based Path Planner in Off-Road Terrain](http://arxiv.org/pdf/2509.07321v1)

Authors: Casey D. Majhor, Jeremy P. Bos

We present a comprehensive evaluation of a point-cloud-based navigation
stack, MUONS, for autonomous off-road navigation. Performance is characterized
by analyzing the results of 30,000 planning and navigation trials in simulation
and validated through field testing. Our simulation campaign considers three
kinematically challenging terrain maps and twenty combinations of seven
path-planning parameters. In simulation, our MUONS-equipped AGV achieved a 0.98
success rate and experienced no failures in the field. By statistical and
correlation analysis we determined that the Bi-RRT expansion radius used in the
initial planning stages is most correlated with performance in terms of
planning time and traversed path length. Finally, we observed that the
proportional variation due to changes in the tuning parameters is remarkably
well correlated to performance in field testing. This finding supports the use
of Monte-Carlo simulation campaigns for performance assessment and parameter
tuning.

### 2. [Aerial-ground Cross-modal Localization: Dataset, Ground-truth, and Benchmark](http://arxiv.org/pdf/2509.07362v1)

Authors: Yandi Yang, Jianping Li, Youqi Liao, Yuhao Li, Yizhe Zhang, Zhen Dong, Bisheng Yang, Naser El-Sheimy

Accurate visual localization in dense urban environments poses a fundamental
task in photogrammetry, geospatial information science, and robotics. While
imagery is a low-cost and widely accessible sensing modality, its effectiveness
on visual odometry is often limited by textureless surfaces, severe viewpoint
changes, and long-term drift. The growing public availability of airborne laser
scanning (ALS) data opens new avenues for scalable and precise visual
localization by leveraging ALS as a prior map. However, the potential of
ALS-based localization remains underexplored due to three key limitations: (1)
the lack of platform-diverse datasets, (2) the absence of reliable ground-truth
generation methods applicable to large-scale urban environments, and (3)
limited validation of existing Image-to-Point Cloud (I2P) algorithms under
aerial-ground cross-platform settings. To overcome these challenges, we
introduce a new large-scale dataset that integrates ground-level imagery from
mobile mapping systems with ALS point clouds collected in Wuhan, Hong Kong, and
San Francisco.

### 3. [TransMPC: Transformer-based Explicit MPC with Variable Prediction Horizon](http://arxiv.org/pdf/2509.07381v1)

Authors: Sichao Wu, Jiang Wu, Xingyu Cao, Fawang Zhang, Guangyuan Yu, Junjie Zhao, Yue Qu, Fei Ma, Jingliang Duan

Traditional online Model Predictive Control (MPC) methods often suffer from
excessive computational complexity, limiting their practical deployment.
Explicit MPC mitigates online computational load by pre-computing control
policies offline; however, existing explicit MPC methods typically rely on
simplified system dynamics and cost functions, restricting their accuracy for
complex systems. This paper proposes TransMPC, a novel Transformer-based
explicit MPC algorithm capable of generating highly accurate control sequences
in real-time for complex dynamic systems. Specifically, we formulate the MPC
policy as an encoder-only Transformer leveraging bidirectional self-attention,
enabling simultaneous inference of entire control sequences in a single forward
pass. This design inherently accommodates variable prediction horizons while
ensuring low inference latency. Furthermore, we introduce a direct policy
optimization framework that alternates between sampling and learning phases.
Unlike imitation-based approaches dependent on precomputed optimal
trajectories, TransMPC directly optimizes the true finite-horizon cost via
automatic differentiation. Random horizon sampling combined with a replay
buffer provides independent and identically distributed (i.i.d.) training
samples, ensuring robust generalization across varying states and horizon
lengths. Extensive simulations and real-world vehicle control experiments
validate the effectiveness of TransMPC in terms of solution accuracy,
adaptability to varying horizons, and computational efficiency.

### 4. [Attention and Risk-Aware Decision Framework for Safe Autonomous Driving](http://arxiv.org/pdf/2509.07412v1)

Authors: Zhen Tian, Fujiang Yuan, Yangfan He, Qinghao Li, Changlin Chen, Huilin Chen, Tianxiang Xu, Jianyu Duan, Yanhong Peng, Zhihao Lin

Autonomous driving has attracted great interest due to its potential
capability in full-unsupervised driving. Model-based and learning-based methods
are widely used in autonomous driving. Model-based methods rely on pre-defined
models of the environment and may struggle with unforeseen events. Proximal
policy optimization (PPO), an advanced learning-based method, can adapt to the
above limits by learning from interactions with the environment. However,
existing PPO faces challenges with poor training results, and low training
efficiency in long sequences. Moreover, the poor training results are
equivalent to collisions in driving tasks. To solve these issues, this paper
develops an improved PPO by introducing the risk-aware mechanism, a
risk-attention decision network, a balanced reward function, and a
safety-assisted mechanism. The risk-aware mechanism focuses on highlighting
areas with potential collisions, facilitating safe-driving learning of the PPO.
The balanced reward function adjusts rewards based on the number of surrounding
vehicles, promoting efficient exploration of the control strategy during
training. Additionally, the risk-attention network enhances the PPO to hold
channel and spatial attention for the high-risk areas of input images.
Moreover, the safety-assisted mechanism supervises and prevents the actions
with risks of collisions during the lane keeping and lane changing. Simulation
results on a physical engine demonstrate that the proposed algorithm
outperforms benchmark algorithms in collision avoidance, achieving higher peak
reward with less training time, and shorter driving time remaining on the risky
areas among multiple testing traffic flow scenarios.

### 5. [Robust Docking Maneuvers for Autonomous Trolley Collection: An Optimization-Based Visual Servoing Scheme](http://arxiv.org/pdf/2509.07413v1)

Authors: Yuhan Pang, Bingyi Xia, Zhe Zhang, Zhirui Sun, Peijia Xie, Bike Zhu, Wenjun Xu, Jiankun Wang

Service robots have demonstrated significant potential for autonomous trolley
collection and redistribution in public spaces like airports or warehouses to
improve efficiency and reduce cost. Usually, a fully autonomous system for the
collection and transportation of multiple trolleys is based on a
Leader-Follower formation of mobile manipulators, where reliable docking
maneuvers of the mobile base are essential to align trolleys into organized
queues. However, developing a vision-based robotic docking system faces
significant challenges: high precision requirements, environmental
disturbances, and inherent robot constraints. To address these challenges, we
propose an optimization-based Visual Servoing scheme that incorporates active
infrared markers for robust feature extraction across diverse lighting
conditions. This framework explicitly models nonholonomic kinematics and
visibility constraints within the Hybrid Visual Servoing problem, augmented
with an observer for disturbance rejection to ensure precise and stable
docking. Experimental results across diverse environments demonstrate the
robustness of this system, with quantitative evaluations confirming high
docking accuracy.

### 6. [Flexible Morphing Aerial Robot with Inflatable Structure for Perching-based Human-Robot Interaction](http://arxiv.org/pdf/2509.07496v1)

Authors: Ayano Miyamichi, Moju Zhao, Kazuki Sugihara, Junichiro Sugihara, Masanori Konishi, Kunio Kojima, Kei Okada, Masayuki Inaba

Birds in nature perform perching not only for rest but also for interaction
with human such as the relationship with falconers. Recently, researchers
achieve perching-capable aerial robots as a way to save energy, and deformable
structure demonstrate significant advantages in efficiency of perching and
compactness of configuration. However, ensuring flight stability remains
challenging for deformable aerial robots due to the difficulty of controlling
flexible arms. Furthermore, perching for human interaction requires high
compliance along with safety. Thus, this study aims to develop a deformable
aerial robot capable of perching on humans with high flexibility and grasping
ability. To overcome the challenges of stability of both flight and perching,
we propose a hybrid morphing structure that combines a unilateral flexible arm
and a pneumatic inflatable actuators. This design allows the robot's arms to
remain rigid during flight and soft while perching for more effective grasping.
We also develop a pneumatic control system that optimizes pressure regulation
while integrating shock absorption and adjustable grasping forces, enhancing
interaction capabilities and energy efficiency. Besides, we focus on the
structural characteristics of the unilateral flexible arm and identify
sufficient conditions under which standard quadrotor modeling and control
remain effective in terms of flight stability. Finally, the developed prototype
demonstrates the feasibility of compliant perching maneuvers on humans, as well
as the robust recovery even after arm deformation caused by thrust reductions
during flight. To the best of our knowledge, this work is the first to achieve
an aerial robot capable of perching on humans for interaction.

### 7. [OmniMap: A General Mapping Framework Integrating Optics, Geometry, and Semantics](http://arxiv.org/pdf/2509.07500v1)

Authors: Yinan Deng, Yufeng Yue, Jianyu Dou, Jingyu Zhao, Jiahui Wang, Yujie Tang, Yi Yang, Mengyin Fu

Robotic systems demand accurate and comprehensive 3D environment perception,
requiring simultaneous capture of photo-realistic appearance (optical), precise
layout shape (geometric), and open-vocabulary scene understanding (semantic).
Existing methods typically achieve only partial fulfillment of these
requirements while exhibiting optical blurring, geometric irregularities, and
semantic ambiguities. To address these challenges, we propose OmniMap. Overall,
OmniMap represents the first online mapping framework that simultaneously
captures optical, geometric, and semantic scene attributes while maintaining
real-time performance and model compactness. At the architectural level,
OmniMap employs a tightly coupled 3DGS-Voxel hybrid representation that
combines fine-grained modeling with structural stability. At the implementation
level, OmniMap identifies key challenges across different modalities and
introduces several innovations: adaptive camera modeling for motion blur and
exposure compensation, hybrid incremental representation with normal
constraints, and probabilistic fusion for robust instance-level understanding.
Extensive experiments show OmniMap's superior performance in rendering
fidelity, geometric accuracy, and zero-shot semantic segmentation compared to
state-of-the-art methods across diverse scenes. The framework's versatility is
further evidenced through a variety of downstream applications, including
multi-domain scene Q&A, interactive editing, perception-guided manipulation,
and map-assisted navigation.

### 8. [Improving Machine Learning-Based Robot Self-Collision Checking with Input Positional Encoding](http://arxiv.org/pdf/2509.07542v1)

Authors: Bartlomiej Kulecki, Dominik Belter

This manuscript investigates the integration of positional encoding -- a
technique widely used in computer graphics -- into the input vector of a binary
classification model for self-collision detection. The results demonstrate the
benefits of incorporating positional encoding, which enhances classification
accuracy by enabling the model to better capture high-frequency variations,
leading to a more detailed and precise representation of complex collision
patterns. The manuscript shows that machine learning-based techniques, such as
lightweight multilayer perceptrons (MLPs) operating in a low-dimensional
feature space, offer a faster alternative for collision checking than
traditional methods that rely on geometric approaches, such as
triangle-to-triangle intersection tests and Bounding Volume Hierarchies (BVH)
for mesh-based models.

### 9. [Decoding RobKiNet: Insights into Efficient Training of Robotic Kinematics Informed Neural Network](http://arxiv.org/pdf/2509.07646v1)

Authors: Yanlong Peng, Zhigang Wang, Ziwen He, Pengxu Chang, Chuangchuang Zhou, Yu Yan, Ming Chen

In robots task and motion planning (TAMP), it is crucial to sample within the
robot's configuration space to meet task-level global constraints and enhance
the efficiency of subsequent motion planning. Due to the complexity of joint
configuration sampling under multi-level constraints, traditional methods often
lack efficiency. This paper introduces the principle of RobKiNet, a
kinematics-informed neural network, for end-to-end sampling within the
Continuous Feasible Set (CFS) under multiple constraints in configuration
space, establishing its Optimization Expectation Model. Comparisons with
traditional sampling and learning-based approaches reveal that RobKiNet's
kinematic knowledge infusion enhances training efficiency by ensuring stable
and accurate gradient optimization.Visualizations and quantitative analyses in
a 2-DOF space validate its theoretical efficiency, while its application on a
9-DOF autonomous mobile manipulator robot(AMMR) demonstrates superior
whole-body and decoupled control, excelling in battery disassembly tasks.
RobKiNet outperforms deep reinforcement learning with a training speed 74.29
times faster and a sampling accuracy of up to 99.25%, achieving a 97.33% task
completion rate in real-world scenarios.

### 10. [Collaborative Exploration with a Marsupial Ground-Aerial Robot Team through Task-Driven Map Compression](http://arxiv.org/pdf/2509.07655v1)

Authors: Angelos Zacharia, Mihir Dharmadhikari, Kostas Alexis

Efficient exploration of unknown environments is crucial for autonomous
robots, especially in confined and large-scale scenarios with limited
communication. To address this challenge, we propose a collaborative
exploration framework for a marsupial ground-aerial robot team that leverages
the complementary capabilities of both platforms. The framework employs a
graph-based path planning algorithm to guide exploration and deploy the aerial
robot in areas where its expected gain significantly exceeds that of the ground
robot, such as large open spaces or regions inaccessible to the ground
platform, thereby maximizing coverage and efficiency. To facilitate large-scale
spatial information sharing, we introduce a bandwidth-efficient, task-driven
map compression strategy. This method enables each robot to reconstruct
resolution-specific volumetric maps while preserving exploration-critical
details, even at high compression rates. By selectively compressing and sharing
key data, communication overhead is minimized, ensuring effective map
integration for collaborative path planning. Simulation and real-world
experiments validate the proposed approach, demonstrating its effectiveness in
improving exploration efficiency while significantly reducing data
transmission.

### Software Engineering

### 1. [Aspect-Oriented Programming in Secure Software Development: A Case Study of Security Aspects in Web Applications](http://arxiv.org/pdf/2509.07449v1)

Authors: Mterorga Ukor

Security remains a critical challenge in modern web applications, where
threats such as unauthorized access, data breaches, and injection attacks
continue to undermine trust and reliability. Traditional Object-Oriented
Programming (OOP) often intertwines security logic with business functionality,
leading to code tangling, scattering, and reduced maintainability. This study
investigates the role of Aspect-Oriented Programming (AOP) in enhancing secure
software development by modularizing cross-cutting security concerns. Using a
case study approach, we compare AOP-based implementations of security features
including authentication, authorization, input validation, encryption, logging,
and session management with conventional OOP or middleware-based approaches.
Data collection involves analyzing code quality metrics (e.g., lines of code,
coupling, cohesion, modularity index, reusability), performance metrics
(response time, throughput, memory usage), and maintainability indicators.
Developer feedback is also incorporated to assess integration and debugging
experiences. Statistical methods, guided by the ISO/IEC 25010 software quality
model, are applied to evaluate differences across implementations. The findings
demonstrate that AOP enhances modularity, reusability, and maintainability of
security mechanisms, while introducing only minimal performance overhead. The
study contributes practical insights for software engineers and researchers
seeking to balance security with software quality in web application
development.

### 2. [CRACI: A Cloud-Native Reference Architecture for the Industrial Compute Continuum](http://arxiv.org/pdf/2509.07498v1)

Authors: Hai Dinh-Tuan

The convergence of Information Technology (IT) and Operational Technology
(OT) in Industry 4.0 exposes the limitations of traditional, hierarchical
architectures like ISA-95 and RAMI 4.0. Their inherent rigidity, data silos,
and lack of support for cloud-native technologies impair the development of
scalable and interoperable industrial systems. This paper addresses this issue
by introducing CRACI, a Cloud-native Reference Architecture for the Industrial
Compute Continuum. Among other features, CRACI promotes a decoupled and
event-driven model to enable flexible, non-hierarchical data flows across the
continuum. It embeds cross-cutting concerns as foundational pillars: Trust,
Governance & Policy, Observability, and Lifecycle Management, ensuring quality
attributes are core to the design. The proposed architecture is validated
through a two-fold approach: (1) a comparative theoretical analysis against
established standards, operational models, and academic proposals; and (2) a
quantitative evaluation based on performance data from previously published
real-world smart manufacturing implementations. The results demonstrate that
CRACI provides a viable, state-of-the-art architecture that utilizes the
compute continuum to overcome the structural limitations of legacy models and
enable scalable, modern industrial systems.

### 3. [Bridging the Gap Between Binary and Source Based Package Management in Spack](http://arxiv.org/pdf/2509.07728v1)

Authors: John Gouwar, Gregory Becker, Tamara Dahlgren, Nathan Hanford, Arjun Guha, Todd Gamblin

Binary package managers install software quickly but they limit
configurability due to rigid ABI requirements that ensure compatibility between
binaries. Source package managers provide flexibility in building software, but
compilation can be slow. For example, installing an HPC code with a new MPI
implementation may result in a full rebuild. Spack, a widely deployed,
HPC-focused package manager, can use source and pre-compiled binaries, but
lacks a binary compatibility model, so it cannot mix binaries not built
together. We present splicing, an extension to Spack that models binary
compatibility between packages and allows seamless mixing of source and binary
distributions. Splicing augments Spack's packaging language and dependency
resolution engine to reuse compatible binaries but maintains the flexibility of
source builds. It incurs minimal installation-time overhead and allows rapid
installation from binaries, even for ABI-sensitive dependencies like MPI that
would otherwise require many rebuilds.

### 4. [What's Coming Next? Short-Term Simulation of Business Processes from Current State](http://arxiv.org/pdf/2509.07747v1)

Authors: Maksym Avramenko, David Chapela-Campa, Marlon Dumas, Fredrik Milani

Business process simulation is an approach to evaluate business process
changes prior to implementation. Existing methods in this field primarily
support tactical decision-making, where simulations start from an empty state
and aim to estimate the long-term effects of process changes. A complementary
use-case is operational decision-making, where the goal is to forecast
short-term performance based on ongoing cases and to analyze the impact of
temporary disruptions, such as demand spikes and shortfalls in available
resources. An approach to tackle this use-case is to run a long-term simulation
up to a point where the workload is similar to the current one (warm-up), and
measure performance thereon. However, this approach does not consider the
current state of ongoing cases and resources in the process. This paper studies
an alternative approach that initializes the simulation from a representation
of the current state derived from an event log of ongoing cases. The paper
addresses two challenges in operationalizing this approach: (1) Given a
simulation model, what information is needed so that a simulation run can start
from the current state of cases and resources? (2) How can the current state of
a process be derived from an event log? The resulting short-term simulation
approach is embodied in a simulation engine that takes as input a simulation
model and a log of ongoing cases, and simulates cases for a given time horizon.
An experimental evaluation shows that this approach yields more accurate
short-term performance forecasts than long-term simulations with warm-up
period, particularly in the presence of concept drift or bursty performance
patterns.

### 5. ["We provide our resources in a dedicated repository": Surveying the Transparency of HICSS publications](http://arxiv.org/pdf/2509.07851v1)

Authors: Irdin Pekaric, Giovanni Apruzzese

Every day, new discoveries are made by researchers from all across the globe
and fields. HICSS is a flagship venue to present and discuss such scientific
advances. Yet, the activities carried out for any given research can hardly be
fully contained in a single document of a few pages-the "paper." Indeed, any
given study entails data, artifacts, or other material that is crucial to truly
appreciate the contributions claimed in the corresponding paper. External
repositories (e.g., GitHub) are a convenient tool to store all such resources
so that future work can freely observe and build upon them -- thereby improving
transparency and promoting reproducibility of research as a whole. In this
work, we scrutinize the extent to which papers recently accepted to HICSS
leverage such repositories to provide supplementary material. To this end, we
collect all the 5579 papers included in HICSS proceedings from 2017-2024. Then,
we identify those entailing either human subject research (850) or technical
implementations (737), or both (147). Finally, we review their text, examining
how many include a link to an external repository-and, inspect its contents.
Overall, out of 2028 papers, only 3\% have a functional and publicly available
repository that is usable by downstream research. We release all our tools.

### 6. [SafeToolBench: Pioneering a Prospective Benchmark to Evaluating Tool Utilization Safety in LLMs](http://arxiv.org/pdf/2509.07315v1)

Authors: Hongfei Xia, Hongru Wang, Zeming Liu, Qian Yu, Yuhang Guo, Haifeng Wang

Large Language Models (LLMs) have exhibited great performance in autonomously
calling various tools in external environments, leading to better problem
solving and task automation capabilities. However, these external tools also
amplify potential risks such as financial loss or privacy leakage with
ambiguous or malicious user instructions. Compared to previous studies, which
mainly assess the safety awareness of LLMs after obtaining the tool execution
results (i.e., retrospective evaluation), this paper focuses on prospective
ways to assess the safety of LLM tool utilization, aiming to avoid irreversible
harm caused by directly executing tools. To this end, we propose SafeToolBench,
the first benchmark to comprehensively assess tool utilization security in a
prospective manner, covering malicious user instructions and diverse practical
toolsets. Additionally, we propose a novel framework, SafeInstructTool, which
aims to enhance LLMs' awareness of tool utilization security from three
perspectives (i.e., \textit{User Instruction, Tool Itself, and Joint
Instruction-Tool}), leading to nine detailed dimensions in total. We experiment
with four LLMs using different methods, revealing that existing approaches fail
to capture all risks in tool utilization. In contrast, our framework
significantly enhances LLMs' self-awareness, enabling a more safe and
trustworthy tool utilization.

### 7. [PatchSeeker: Mapping NVD Records to their Vulnerability-fixing Commits with LLM Generated Commits and Embeddings](http://arxiv.org/pdf/2509.07540v1)

Authors: Huu Hung Nguyen, Anh Tuan Nguyen, Thanh Le-Cong, Yikun Li, Han Wei Ang, Yide Yin, Frank Liauw, Shar Lwin Khin, Ouh Eng Lieh, Ting Zhang, David Lo

Software vulnerabilities pose serious risks to modern software ecosystems.
While the National Vulnerability Database (NVD) is the authoritative source for
cataloging these vulnerabilities, it often lacks explicit links to the
corresponding Vulnerability-Fixing Commits (VFCs). VFCs encode precise code
changes, enabling vulnerability localization, patch analysis, and dataset
construction. Automatically mapping NVD records to their true VFCs is therefore
critical. Existing approaches have limitations as they rely on sparse, often
noisy commit messages and fail to capture the deep semantics in the
vulnerability descriptions. To address this gap, we introduce PatchSeeker, a
novel method that leverages large language models to create rich semantic links
between vulnerability descriptions and their VFCs. PatchSeeker generates
embeddings from NVD descriptions and enhances commit messages by synthesizing
detailed summaries for those that are short or uninformative. These generated
messages act as a semantic bridge, effectively closing the information gap
between natural language reports and low-level code changes. Our approach
PatchSeeker achieves 59.3% higher MRR and 27.9% higher Recall@10 than the
best-performing baseline, Prospector, on the benchmark dataset. The extended
evaluation on recent CVEs further confirms PatchSeeker's effectiveness.
Ablation study shows that both the commit message generation method and the
selection of backbone LLMs make a positive contribution to PatchSeeker. We also
discuss limitations and open challenges to guide future work.

### 8. [Breaking Android with AI: A Deep Dive into LLM-Powered Exploitation](http://arxiv.org/pdf/2509.07933v1)

Authors: Wanni Vidulige Ishan Perera, Xing Liu, Fan liang, Junyi Zhang

The rapid evolution of Artificial Intelligence (AI) and Large Language Models
(LLMs) has opened up new opportunities in the area of cybersecurity, especially
in the exploitation automation landscape and penetration testing. This study
explores Android penetration testing automation using LLM-based tools,
especially PentestGPT, to identify and execute rooting techniques. Through a
comparison of the traditional manual rooting process and exploitation methods
produced using AI, this study evaluates the efficacy, reliability, and
scalability of automated penetration testing in achieving high-level privilege
access on Android devices. With the use of an Android emulator (Genymotion) as
the testbed, we fully execute both traditional and exploit-based rooting
methods, automating the process using AI-generated scripts. Secondly, we create
a web application by integrating OpenAI's API to facilitate automated script
generation from LLM-processed responses. The research focuses on the
effectiveness of AI-enabled exploitation by comparing automated and manual
penetration testing protocols, by determining LLM weaknesses and strengths
along the way. We also provide security suggestions of AI-enabled exploitation,
including ethical factors and potential misuse. The findings exhibit that while
LLMs can significantly streamline the workflow of exploitation, they need to be
controlled by humans to ensure accuracy and ethical application. This study
adds to the increasing body of literature on AI-powered cybersecurity and its
effect on ethical hacking, security research, and mobile device security.

### 9. [A smart fridge with AI-enabled food computing](http://arxiv.org/pdf/2509.07400v1)

Authors: Khue Nong Thuc, Khoa Tran Nguyen Anh, Tai Nguyen Huy, Du Nguyen Hao Hong, Khanh Dinh Ba

The Internet of Things (IoT) plays a crucial role in enabling seamless
connectivity and intelligent home automation, particularly in food management.
By integrating IoT with computer vision, the smart fridge employs an ESP32-CAM
to establish a monitoring subsystem that enhances food management efficiency
through real-time food detection, inventory tracking, and temperature
monitoring. This benefits waste reduction, grocery planning improvement, and
household consumption optimization. In high-density inventory conditions,
capturing partial or layered images complicates object detection, as
overlapping items and occluded views hinder accurate identification and
counting. Besides, varied angles and obscured details in multi-layered setups
reduce algorithm reliability, often resulting in miscounts or
misclassifications. Our proposed system is structured into three core modules:
data pre-processing, object detection and management, and a web-based
visualization. To address the challenge of poor model calibration caused by
overconfident predictions, we implement a variant of focal loss that mitigates
over-confidence and under-confidence in multi-category classification. This
approach incorporates adaptive, class-wise error calibration via temperature
scaling and evaluates the distribution of predicted probabilities across
methods. Our results demonstrate that robust functional calibration
significantly improves detection reliability under varying lighting conditions
and scalability challenges. Further analysis demonstrates a practical,
user-focused approach to modern food management, advancing sustainable living
goals through reduced waste and more informed consumption.

### 10. [What Were You Thinking? An LLM-Driven Large-Scale Study of Refactoring Motivations in Open-Source Projects](http://arxiv.org/pdf/2509.07763v1)

Authors: Mikel Robredo, Matteo Esposito, Fabio Palomba, Rafael Peñaloza, Valentina Lenarduzzi

Context. Code refactoring improves software quality without changing external
behavior. Despite its advantages, its benefits are hindered by the considerable
cost of time, resources, and continuous effort it demands. Aim. Understanding
why developers refactor, and which metrics capture these motivations, may
support wider and more effective use of refactoring in practice. Method. We
performed a large-scale empirical study to analyze developers refactoring
activity, leveraging Large Language Models (LLMs) to identify underlying
motivations from version control data, comparing our findings with previous
motivations reported in the literature. Results. LLMs matched human judgment in
80% of cases, but aligned with literature-based motivations in only 47%. They
enriched 22% of motivations with more detailed rationale, often highlighting
readability, clarity, and structural improvements. Most motivations were
pragmatic, focused on simplification and maintainability. While metrics related
to developer experience and code readability ranked highest, their correlation
with motivation categories was weak. Conclusions. We conclude that LLMs
effectively capture surface-level motivations but struggle with architectural
reasoning. Their value lies in providing localized explanations, which, when
combined with software metrics, can form hybrid approaches. Such integration
offers a promising path toward prioritizing refactoring more systematically and
balancing short-term improvements with long-term architectural goals.

### Social and Information Networks

### 1. [Free Elections in the Free State: Ensemble Analysis of Redistricting in New Hampshire](http://arxiv.org/pdf/2509.07328v1)

Authors: Atticus McWhorter, Daryl DeFord

The process of legislative redistricting in New Hampshire, along with many
other states across the country, was particularly contentious during the 2020
census cycle. In this paper we present an ensemble analysis of the enacted
districts to provide mathematical context for claims made about these maps in
litigation. Operationalizing the New Hampshire redistricting rules and
algorithmically generating a large collection of districting plans allows us to
construct a baseline for expected behavior of districting plans in the state
and evaluate non-partisan justifications and geographic tradeoffs between
districting criteria and partisan outcomes. In addition, our results
demonstrate the impact of selection and aggregation of election data for
analyzing partisan symmetry measures.

### 2. [Influence Maximization Considering Influence, Cost and Time](http://arxiv.org/pdf/2509.07625v1)

Authors: Mingyang Feng, Qi Zhao, Shan He, Yuhui Shi

Influence maximization has been studied for social network analysis, such as
viral marketing (advertising), rumor prevention, and opinion leader
identification. However, most studies neglect the interplay between influence
spread, cost efficiency, and temporal urgency. In practical scenarios such as
viral marketing and information campaigns, jointly optimizing Influence, Cost,
and Time is essential, yet remaining largely unaddressed in current literature.
To bridge the gap, this paper proposes a new multi-objective influence
maximization problem that simultaneously optimizes influence, cost, and time.
We show the intuitive and empirical evidence to prove the feasibility and
necessity of this multi-objective problem. We also develop an evolutionary
variable-length search algorithm that can effectively search for optimal node
combinations. The proposed EVEA algorithm outperforms all baselines, achieving
up to 19.3% higher hypervolume and 25 to 40% faster convergence across four
real-world networks, while maintaining a diverse and balanced Pareto front
among influence, cost, and time objectives.

### 3. [Multi-Topic Projected Opinion Dynamics for Resource Allocation](http://arxiv.org/pdf/2509.07847v1)

Authors: Prashil Wankhede, Nirabhra Mandal, Sonia Martínez, Pavankumar Tallapragada

We propose a model of opinion formation on resource allocation among multiple
topics by multiple agents, who are subject to hard budget constraints. We
define a utility function for each agent and then derive a projected dynamical
system model of opinion evolution assuming that each agent myopically seeks to
maximize its utility subject to its constraints. Inter-agent coupling arises
from an undirected social network, while inter-topic coupling arises from
resource constraints. We show that opinions always converge to the equilibrium
set. For special networks with very weak antagonistic relations, the opinions
converge to a unique equilibrium point. We further show that the underlying
opinion formation game is a potential game. We relate the equilibria of the
dynamics and the Nash equilibria of the game and characterize the unique Nash
equilibrium for networks with no antagonistic relations. Finally, simulations
illustrate our findings.

### Systems and Control

### 1. [Distributed Leader-Follower Consensus for Uncertain Multiagent Systems with Time-Triggered Switching of the Communication Network](http://arxiv.org/pdf/2509.07304v1)

Authors: Armel Koulong, Ali Pakniyat

A distributed adaptive control strategy is developed for heterogeneous
multiagent systems in nonlinear Brunovsky form with \({\pd}\)-dimensional
$n^{\text{th}}$-order dynamics, operating under time-triggered switching
communication topologies. The approach uses repulsive potential functions to
ensure agent-agent and obstacle safety, while neural network estimators
compensate for system uncertainties and disturbances. A high-order control
barrier function framework is then employed to certify the positive invariance
of the safe sets and the boundedness of the proposed control inputs. The
resulting distributed control and adaptive laws, together with dwell-time
requirements for topology transitions, achieve leader-following consensus. This
integrated design provides synchronized formation and robust disturbance
rejection in evolving network configurations, and its effectiveness is
demonstrated through numerical simulations.

### 2. [Data-knowledge fusion driven frequency security assessment: A robust framework for renewable-dominated power grids](http://arxiv.org/pdf/2509.07320v1)

Authors: Yurun Zhang, Wei Yao, Yutian Lan, Hang Shuai, Shanyang Wei, Wei Gan, Chao Duan, Jinyu Wen, Shijie Cheng

Frequency security is critical for power grids, as deviations can trigger
widespread outages and result in substantial economic losses. However, modern
renewable-dominated power grids face an increased risk of insecurity due to low
inertia and nonlinear frequency responses. To mitigate these risks, robust
pre-fault frequency security assessment (FSA) is critical, which enables grid
operators to implement preventive control strategies. We propose a
data-knowledge fusion framework to achieve intelligent FSA in actual power
grids. First, we classify FSA domain knowledge into two distinct categories:
(1) physics-guided knowledge directs the neural network pre-training process,
ensuring that the fusion model's predictions consistent with frequency response
mechanisms, and (2) physics-constrained knowledge establishes quantitative
relationship on predictions, which forces them within theoretical ranges
defined by domain knowledge. Furthermore, we develop a dual-channel neural
network architecture to simultaneously capture both local and global
characteristics of the power system. Finally, we introduce a data-knowledge
fusion training algorithm that integrates guided learning with constrained
network architecture to enhance model reliability and generalization. Case
studies on China's Yunnan Provincial Power Grid validate the superior
performance of our framework: it reduces average prediction error to 1.26% (a
49.2% reduction over data-driven methods), and maintains 97.60% accuracy in
untrained scenarios (3.85% higher than data-driven methods), therefore
satisfies the accuracy, reliability, and generalization requirements for actual
power grids. The proposed methodology establishes a new paradigm for enhancing
robustness of FSA in power grids, with potential application to cross-domain
security assessment.

### 3. [Distributed Frequency Control for Multi-Area Power Systems Considering Transient Frequency Safety](http://arxiv.org/pdf/2509.07345v1)

Authors: Xiemin Mo, Tao Liu

High penetration of renewable energy sources intensifies frequency
fluctuations in multi-area power systems, challenging both stability and
operational safety. This paper proposes a novel distributed frequency control
method that ensures transient frequency safety and enforces generation capacity
constraints, while achieving steady-state frequency restoration and optimal
economic operation. The method integrates a feedback optimization (FO)-based
controller and a safety corrector. The FO-based controller generates reference
setpoints by solving an optimization problem, driving the system to the steady
state corresponding to the optimal solution of this problem. The safety
corrector then modifies these references using control barrier functions to
maintain frequencies within prescribed safe bounds during transients while
respecting capacity constraints. The proposed method combines low computational
burden with improved regulation performance and enhanced practical
applicability. Theoretical analysis establishes optimality, asymptotic
stability, and transient frequency safety for the closed-loop system.
Simulation studies show that, compared with conventional FO-based schemes, the
method consistently enforces frequency safety and capacity limits, achieves
smaller frequency deviations and faster recovery, thereby demonstrating its
practical effectiveness and advantages.

### 4. [Anti-Disturbance Hierarchical Sliding Mode Controller for Deep-Sea Cranes with Adaptive Control and Neural Network Compensation](http://arxiv.org/pdf/2509.07356v1)

Authors: Qian Zuo, Shujie Wu, Yuzhe Qian

To address non-linear disturbances and uncertainties in complex marine
environments, this paper proposes a disturbance-resistant controller for
deep-sea cranes. The controller integrates hierarchical sliding mode control,
adaptive control, and neural network compensation techniques. By designing a
global sliding mode surface, the dynamic coordination between the driving and
non-driving subsystems is achieved, ensuring overall system stability. The
subsystem surfaces reduce oscillations and enhance tracking accuracy. Adaptive
control dynamically adjusts system parameters, enhancing robustness against
external uncertainties, while the neural network compensates for time-varying
disturbances through real-time learning. The stability of the control scheme is
verified on the basis of Lyapunov theory. The simulation results demonstrate
that, compared to traditional PID control, the proposed controller exhibits
significant advantages in trajectory tracking accuracy, response speed, and
disturbance rejection.

### 5. [Adaptive Event-Triggered MPC for Linear Parameter-Varying Systems with State Delays, Actuator Saturation and Disturbances](http://arxiv.org/pdf/2509.07384v1)

Authors: Aiping Zhong, Wanlin Lu, Langwen Zhang, Ziyang Bao

This paper proposes a unified adaptive event-triggered model predictive
control (ETMPC) scheme for linear parameter-varying (LPV) systems subject to
state delays, actuator saturation, and external disturbances. In existing
studies, only a limited number of ETMPC methods have attempted to address
either state delays or actuator saturation, and even these few methods
typically lack co-design optimization between adaptive event-triggering
mechanisms and the control law. To overcome these limitations, this paper
presents a Lyapunov-Krasovskii-based adaptive ETMPC strategy that enables the
co-design optimization of both the triggering mechanism and the controller.
Specifically, the event-triggering parameter matrix is adaptively optimized by
embedding an internal adaptive variable within the Lyapunov-Krasovskii-like
function. Furthermore, the actuator saturation nonlinearity is transformed into
a convex hull representation. The infinite-horizon robust optimization problem
is reformulated as a convex optimization problem with linear matrix inequality
(LMI) constraints. Invariant set constraints are introduced to ensure recursive
feasibility, and mean-square input-to-state stability (ISS) under multiple
uncertainties is rigorously established. Simulations on an industrial electric
heating system validate the proposed method's effectiveness in reducing
communication load.

### 6. [Electric Vehicle Routing Problem with Time Windows and Station-based or Route-based Charging Options](http://arxiv.org/pdf/2509.07402v1)

Authors: Tran Trung Duc, Vu Duc Minh, Nguyen Ngoc Doanh, Pham Gia Nguyen, Laurent El Ghaoui, Ha Minh Hoang

The Electric Vehicle Routing Problem with Time Windows and Station-based or
Route-based Charging Options addresses fleet optimization incorporating both
conventional charging stations and continuous wireless charging infrastructure.
This paper extends Schneider et al.'s foundational EVRP-TW model with arc-based
dynamic wireless charging representation, partial coverage modeling, and
hierarchical multi-objective optimization prioritizing fleet minimization.
Computational experiments on Schneider benchmark instances demonstrate
substantial operational benefits, with distance and time improvements ranging
from 0.7% to 35.9% in secondary objective components. Analysis reveals that 20%
wireless coverage achieves immediate benefits, while 60% coverage delivers
optimal performance across all test instances for infrastructure investment
decisions.

### 7. [A kernel-based approach to physics-informed nonlinear system identification](http://arxiv.org/pdf/2509.07634v1)

Authors: Cesare Donati, Martina Mammarella, Giuseppe C. Calafiore, Fabrizio Dabbene, Constantino Lagoa, Carlo Novara

This paper presents a kernel-based framework for physics-informed nonlinear
system identification. The key contribution is a structured methodology that
extends kernel-based techniques to seamlessly integrate partially known
physics-based models, improving parameter estimation and overall model
accuracy. The proposed method enhances traditional modeling approaches by
integrating a parametric model, which provides physical interpretability, with
a kernel-based function, which accounts for unmodelled dynamics. The two
model's components are identified from data simultaneously, minimizing a
suitable cost that balances the relative importance of the physical and the
black-box parts of the model. Additionally, nonlinear state smoothing is
employed to address scenarios involving state-space models with not fully
measurable states. Numerical simulations on an experimental benchmark system
demonstrate the effectiveness of the proposed approach, with performance
comparisons against state-of-the-art identification techniques.

### 8. [Prescribed-Time Event-Triggered Control for Matrix-Scaled Networks](http://arxiv.org/pdf/2509.07703v1)

Authors: Sunny K P, Rakesh R Warier

This article proposes a distributed control method for matrix-scaled
multi-agent networks aimed at achieving convergence within a user-defined time
frame. The control law of each individual agent relies only on information from
neighboring agents and is updated at discrete intervals determined by
state-dependent triggering functions, reducing the frequency of agent
interactions. To this end, first, the controller is augmented with a
time-varying gain. Then, the dynamics of the closed-loop system over the
finite-time interval is transformed into an infinite-time frame using time
scaling. Lyapunov-based analysis is employed to derive suitable triggering
conditions that guarantee the asymptotic convergence of the time-transformed
system, thereby ensuring the prescribed-time convergence of the original
system.

### 9. [Swarm-optimized Adaptive Augmentation of Missile Autopilot](http://arxiv.org/pdf/2509.07748v1)

Authors: Alexander Dorsey, Parham Oveissi, Jeffrey D. Barton, Ankit Goel

This paper considers the problem of optimizing a missile autopilot. In
particular, the paper investigates the application of an online learning
technique to learn and optimize the gains of a three-loop topology autopilot
for a planar missile modeled with nonlinear dynamics and nonlinear aerodynamics
forces and moments. The classical autopilot for a missile is based on a
three-loop topology, where each loop consists of tunable proportional gains. An
adaptive three-loop autopilot is constructed by augmenting the classical
autopilot's fixed-gain controllers with a learning-based controller, which is
recursively optimized using retrospective cost optimization. Numerical
simulations show that online learning improves the tracking performance of the
classical autopilot in both nominal and off-nominal interception scenarios.

### 10. [Filtering in Multivariate Systems with Quantized Measurements using a Gaussian Mixture-Based Indicator Approximation](http://arxiv.org/pdf/2509.07837v1)

Authors: Angel L. Cedeño, Rodrigo A. González, Boris I. Godoy, Juan C. Agüero

This work addresses the problem of state estimation in multivariable dynamic
systems with quantized outputs, a common scenario in applications involving
low-resolution sensors or communication constraints. A novel method is proposed
to explicitly construct the probability mass function associated with the
quantized measurements by approximating the indicator function of each region
defined by the quantizer using Gaussian mixture models. Unlike previous
approaches, this technique generalizes to any number of quantized outputs
without requiring case-specific numerical solutions, making it a scalable and
efficient solution. Simulation results demonstrate that the proposed filter
achieves high accuracy in state estimation, both in terms of fidelity of the
filtering distributions and mean squared error, while maintaining significantly
reduced computational cost.

### Machine Learning (Statistics Category)

### 1. [Identifying Neural Signatures from fMRI using Hybrid Principal Components Regression](http://arxiv.org/pdf/2509.07300v1)

Authors: Jared Rieck, Julia Wrobel, Joshua L. Gowin, Yue Wang, Martin Paulus, Ryan Peterson

Recent advances in neuroimaging analysis have enabled accurate decoding of
mental state from brain activation patterns during functional magnetic
resonance imaging scans. A commonly applied tool for this purpose is principal
components regression regularized with the least absolute shrinkage and
selection operator (LASSO PCR), a type of multi-voxel pattern analysis (MVPA).
This model presumes that all components are equally likely to harbor relevant
information, when in fact the task-related signal may be concentrated in
specific components. In such cases, the model will fail to select the optimal
set of principal components that maximizes the total signal relevant to the
cognitive process under study. Here, we present modifications to LASSO PCR that
allow for a regularization penalty tied directly to the index of the principal
component, reflecting a prior belief that task-relevant signal is more likely
to be concentrated in components explaining greater variance. Additionally, we
propose a novel hybrid method, Joint Sparsity-Ranked LASSO (JSRL), which
integrates component-level and voxel-level activity under an information parity
framework and imposes ranked sparsity to guide component selection. We apply
the models to brain activation during risk taking, monetary incentive, and
emotion regulation tasks. Results demonstrate that incorporating sparsity
ranking into LASSO PCR produces models with enhanced classification
performance, with JSRL achieving up to 51.7\% improvement in cross-validated
deviance $R^2$ and 7.3\% improvement in cross-validated AUC. Furthermore,
sparsity-ranked models perform as well as or better than standard LASSO PCR
approaches across all classification tasks and allocate predictive weight to
brain regions consistent with their established functional roles, offering a
robust alternative for MVPA.

### 2. [Asynchronous Gossip Algorithms for Rank-Based Statistical Methods](http://arxiv.org/pdf/2509.07543v1)

Authors: Anna Van Elst, Igor Colin, Stephan Clémençon

As decentralized AI and edge intelligence become increasingly prevalent,
ensuring robustness and trustworthiness in such distributed settings has become
a critical issue-especially in the presence of corrupted or adversarial data.
Traditional decentralized algorithms are vulnerable to data contamination as
they typically rely on simple statistics (e.g., means or sum), motivating the
need for more robust statistics. In line with recent work on decentralized
estimation of trimmed means and ranks, we develop gossip algorithms for
computing a broad class of rank-based statistics, including L-statistics and
rank statistics-both known for their robustness to outliers. We apply our
method to perform robust distributed two-sample hypothesis testing, introducing
the first gossip algorithm for Wilcoxon rank-sum tests. We provide rigorous
convergence guarantees, including the first convergence rate bound for
asynchronous gossip-based rank estimation. We empirically validate our
theoretical results through experiments on diverse network topologies.

### 3. [uGMM-NN: Univariate Gaussian Mixture Model Neural Network](http://arxiv.org/pdf/2509.07569v1)

Authors: Zakeria Sharif Ali

This paper introduces the Univariate Gaussian Mixture Model Neural Network
(uGMM-NN), a novel neural architecture that embeds probabilistic reasoning
directly into the computational units of deep networks. Unlike traditional
neurons, which apply weighted sums followed by fixed nonlinearities, each
uGMM-NN node parameterizes its activations as a univariate Gaussian mixture,
with learnable means, variances, and mixing coefficients. This design enables
richer representations by capturing multimodality and uncertainty at the level
of individual neurons, while retaining the scalability of standard feedforward
networks. We demonstrate that uGMM-NN can achieve competitive discriminative
performance compared to conventional multilayer perceptrons, while additionally
offering a probabilistic interpretation of activations. The proposed framework
provides a foundation for integrating uncertainty-aware components into modern
neural architectures, opening new directions for both discriminative and
generative modeling.

### 4. [Expected Signature Kernels for Lévy Rough Paths](http://arxiv.org/pdf/2509.07893v1)

Authors: Peter K. Friz, Paul P. Hager

The expected signature kernel arises in statistical learning tasks as a
similarity measure of probability measures on path space. Computing this kernel
for known classes of stochastic processes is an important problem that, in
particular, can help reduce computational costs. Building on the representation
of the expected signature of (inhomogeneous) L\'evy processes with absolutely
continuous characteristics as the development of an absolutely continuous path
in the extended tensor algebra [F.-H.-Tapia, Forum of Mathematics: Sigma
(2022), "Unified signature cumulants and generalized Magnus expansions"], we
extend the arguments developed for smooth rough paths in
[Lemercier-Lyons-Salvi, "Log-PDE Methods for Rough Signature Kernels"] to
derive a PDE system for the expected signature of inhomogeneous L\'evy
processes. As a specific example, we see that the expected signature kernel of
Gaussian martingales satisfies a Goursat PDE.

### 5. [Bayesian Pliable Lasso with Horseshoe Prior for Interaction Effects in GLMs with Missing Responses](http://arxiv.org/pdf/2509.07501v1)

Authors: The Tien Mai

Sparse regression problems, where the goal is to identify a small set of
relevant predictors, often require modeling not only main effects but also
meaningful interactions through other variables. While the pliable lasso has
emerged as a powerful frequentist tool for modeling such interactions under
strong heredity constraints, it lacks a natural framework for uncertainty
quantification and incorporation of prior knowledge. In this paper, we propose
a Bayesian pliable lasso that extends this approach by placing
sparsity-inducing priors, such as the horseshoe, on both main and interaction
effects. The hierarchical prior structure enforces heredity constraints while
adaptively shrinking irrelevant coefficients and allowing important effects to
persist. We extend this framework to Generalized Linear Models (GLMs) and
develop a tailored approach to handle missing responses. To facilitate
posterior inference, we develop an efficient Gibbs sampling algorithm based on
a reparameterization of the horseshoe prior. Our Bayesian framework yields
sparse, interpretable interaction structures, and principled measures of
uncertainty. Through simulations and real-data studies, we demonstrate its
advantages over existing methods in recovering complex interaction patterns
under both complete and incomplete data.
  Our method is implemented in the package \texttt{hspliable} available on
Github.

### 6. [Physics-informed low-rank neural operators with application to parametric elliptic PDEs](http://arxiv.org/pdf/2509.07687v1)

Authors: Sebastian Schaffer, Lukas Exl

We present the Physics-Informed Low-Rank Neural Operator (PILNO), a neural
operator framework for efficiently approximating solution operators of partial
differential equations (PDEs) on point cloud data. PILNO combines low-rank
kernel approximations with an encoder--decoder architecture, enabling fast,
continuous one-shot predictions while remaining independent of specific
discretizations. The model is trained using a physics-informed penalty
framework, ensuring that PDE constraints and boundary conditions are satisfied
in both supervised and unsupervised settings. We demonstrate its effectiveness
on diverse problems, including function fitting, the Poisson equation, the
screened Poisson equation with variable coefficients, and parameterized Darcy
flow. The low-rank structure provides computational efficiency in
high-dimensional parameter spaces, establishing PILNO as a scalable and
flexible surrogate modeling tool for PDEs.

### 7. [Feature Understanding and Sparsity Enhancement via 2-Layered kernel machines (2L-FUSE)](http://arxiv.org/pdf/2509.07806v1)

Authors: Fabiana Camattari, Sabrina Guastavino, Francesco Marchetti, Emma Perracchione

We propose a novel sparsity enhancement strategy for regression tasks, based
on learning a data-adaptive kernel metric, i.e., a shape matrix, through
2-Layered kernel machines. The resulting shape matrix, which defines a
Mahalanobis-type deformation of the input space, is then factorized via an
eigen-decomposition, allowing us to identify the most informative directions in
the space of features. This data-driven approach provides a flexible,
interpretable and accurate feature reduction scheme. Numerical experiments on
synthetic and applications to real datasets of geomagnetic storms demonstrate
that our approach achieves minimal yet highly informative feature sets without
losing predictive performance.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-09-10 PST.

### 1. [Synthetic data can benefit medical research — but risks must be recognized](https://www.nature.com/articles/d41586-025-02869-0)

Authors: 

### 2. [Sampling-enabled scalable manifold learning unveils the discriminative cluster structure of high-dimensional data](https://www.nature.com/articles/s42256-025-01112-9)

Authors: Dehua Peng et al.

