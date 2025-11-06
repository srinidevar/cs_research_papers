# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-11-05 17:00:25.703836 PST.

### Artificial Intelligence

### 1. [Optimal-Agent-Selection: State-Aware Routing Framework for Efficient Multi-Agent Collaboration](http://arxiv.org/pdf/2511.02200v1)

Authors: Jingbo Wang, Sendong Zhao, Haochun Wang, Yuzheng Fan, Lizhe Zhang, Yan Liu, Ting Liu

The emergence of multi-agent systems powered by large language models (LLMs)
has unlocked new frontiers in complex task-solving, enabling diverse agents to
integrate unique expertise, collaborate flexibly, and address challenges
unattainable for individual models. However, the full potential of such systems
is hindered by rigid agent scheduling and inefficient coordination strategies
that fail to adapt to evolving task requirements. In this paper, we propose
STRMAC, a state-aware routing framework designed for efficient collaboration in
multi-agent systems. Our method separately encodes interaction history and
agent knowledge to power the router, which adaptively selects the most suitable
single agent at each step for efficient and effective collaboration.
Furthermore, we introduce a self-evolving data generation approach that
accelerates the collection of high-quality execution paths for efficient system
training. Experiments on challenging collaborative reasoning benchmarks
demonstrate that our method achieves state-of-the-art performance, achieving up
to 23.8% improvement over baselines and reducing data collection overhead by up
to 90.1% compared to exhaustive search.

### 2. [TabDSR: Decompose, Sanitize, and Reason for Complex Numerical Reasoning in Tabular Data](http://arxiv.org/pdf/2511.02219v1)

Authors: Changjiang Jiang, Fengchang Yu, Haihua Chen, Wei Lu, Jin Zeng

Complex reasoning over tabular data is crucial in real-world data analysis,
yet large language models (LLMs) often underperform due to complex queries,
noisy data, and limited numerical capabilities. To address these issues, we
propose \method, a framework consisting of: (1) a query decomposer that breaks
down complex questions, (2) a table sanitizer that cleans and filters noisy
tables, and (3) a program-of-thoughts (PoT)-based reasoner that generates
executable code to derive the final answer from the sanitized table. To ensure
unbiased evaluation and mitigate data leakage, we introduce a new dataset,
CalTab151, specifically designed for complex numerical reasoning over tables.
Experimental results demonstrate that \method consistently outperforms existing
methods, achieving state-of-the-art (SOTA) performance with 8.79%, 6.08%, and
19.87% accuracy improvement on TAT-QA, TableBench, and \method, respectively.
Moreover, our framework integrates seamlessly with mainstream LLMs, providing a
robust solution for complex tabular numerical reasoning. These findings
highlight the effectiveness of our framework in enhancing LLM performance for
complex tabular numerical reasoning. Data and code are available upon request.

### 3. [Deep Ideation: Designing LLM Agents to Generate Novel Research Ideas on Scientific Concept Network](http://arxiv.org/pdf/2511.02238v1)

Authors: Keyu Zhao, Weiquan Lin, Qirui Zheng, Fengli Xu, Yong Li

Novel research ideas play a critical role in advancing scientific inquiries.
Recent advancements in Large Language Models (LLMs) have demonstrated their
potential to generate novel research ideas by leveraging large-scale scientific
literature. However, previous work in research ideation has primarily relied on
simplistic methods, such as keyword co-occurrence or semantic similarity. These
approaches focus on identifying statistical associations in the literature but
overlook the complex, contextual relationships between scientific concepts,
which are essential to effectively leverage knowledge embedded in human
literature. For instance, papers that simultaneously mention "keyword A" and
"keyword B" often present research ideas that integrate both concepts.
Additionally, some LLM-driven methods propose and refine research ideas using
the model's internal knowledge, but they fail to effectively utilize the
scientific concept network, limiting the grounding of ideas in established
research. To address these challenges, we propose the Deep Ideation framework
to address these challenges, integrating a scientific network that captures
keyword co-occurrence and contextual relationships, enriching LLM-driven
ideation. The framework introduces an explore-expand-evolve workflow to
iteratively refine research ideas, using an Idea Stack to track progress. A
critic engine, trained on real-world reviewer feedback, guides the process by
providing continuous feedback on the novelty and feasibility of ideas. Our
experiments show that our approach improves the quality of generated ideas by
10.67% compared to other methods, with ideas surpassing top conference
acceptance levels. Human evaluation highlights their practical value in
scientific research, and ablation studies confirm the effectiveness of each
component in the workflow. Code repo is available at
https://github.com/kyZhao-1/Deep-Ideation.

### 4. [When Modalities Conflict: How Unimodal Reasoning Uncertainty Governs Preference Dynamics in MLLMs](http://arxiv.org/pdf/2511.02243v1)

Authors: Zhuoran Zhang, Tengyue Wang, Xilin Gong, Yang Shi, Haotian Wang, Di Wang, Lijie Hu

Multimodal large language models (MLLMs) must resolve conflicts when
different modalities provide contradictory information, a process we term
modality following. Prior work measured this behavior only with coarse
dataset-level statistics, overlooking the influence of model's confidence in
unimodal reasoning. In this paper, we introduce a new framework that decomposes
modality following into two fundamental factors: relative reasoning uncertainty
(the case-specific confidence gap between unimodal predictions) and inherent
modality preference( a model's stable bias when uncertainties are balanced). To
validate this framework, we construct a controllable dataset that
systematically varies the reasoning difficulty of visual and textual inputs.
Using entropy as a fine-grained uncertainty metric, we uncover a universal law:
the probability of following a modality decreases monotonically as its relative
uncertainty increases. At the relative difficulty level where the model tends
to follow both modalities with comparable probability what we call the balance
point, a practical indicator of the model's inherent preference. Unlike
traditional macro-level ratios, this measure offers a more principled and less
confounded way to characterize modality bias, disentangling it from unimodal
capabilities and dataset artifacts. Further, by probing layer-wise predictions,
we reveal the internal mechanism of oscillation: in ambiguous regions near the
balance point, models vacillate between modalities across layers, explaining
externally observed indecision. Together, these findings establish relative
uncertainty and inherent preference as the two governing principles of modality
following, offering both a quantitative framework and mechanistic insight into
how MLLMs resolve conflicting information.

### 5. [Fuzzy Soft Set Theory based Expert System for the Risk Assessment in Breast Cancer Patients](http://arxiv.org/pdf/2511.02392v1)

Authors: Muhammad Sheharyar Liaqat

Breast cancer remains one of the leading causes of mortality among women
worldwide, with early diagnosis being critical for effective treatment and
improved survival rates. However, timely detection continues to be a challenge
due to the complex nature of the disease and variability in patient risk
factors. This study presents a fuzzy soft set theory-based expert system
designed to assess the risk of breast cancer in patients using measurable
clinical and physiological parameters. The proposed system integrates Body Mass
Index, Insulin Level, Leptin Level, Adiponectin Level, and age as input
variables to estimate breast cancer risk through a set of fuzzy inference rules
and soft set computations. These parameters can be obtained from routine blood
analyses, enabling a non-invasive and accessible method for preliminary
assessment. The dataset used for model development and validation was obtained
from the UCI Machine Learning Repository. The proposed expert system aims to
support healthcare professionals in identifying high-risk patients and
determining the necessity of further diagnostic procedures such as biopsies.

### 6. [A New Perspective on Precision and Recall for Generative Models](http://arxiv.org/pdf/2511.02414v1)

Authors: Benjamin Sykes, Loïc Simon, Julien Rabin, Jalal Fadili

With the recent success of generative models in image and text, the question
of their evaluation has recently gained a lot of attention. While most methods
from the state of the art rely on scalar metrics, the introduction of Precision
and Recall (PR) for generative model has opened up a new avenue of research.
The associated PR curve allows for a richer analysis, but their estimation
poses several challenges. In this paper, we present a new framework for
estimating entire PR curves based on a binary classification standpoint. We
conduct a thorough statistical analysis of the proposed estimates. As a
byproduct, we obtain a minimax upper bound on the PR estimation risk. We also
show that our framework extends several landmark PR metrics of the literature
which by design are restrained to the extreme values of the curve. Finally, we
study the different behaviors of the curves obtained experimentally in various
settings.

### 7. [ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning](http://arxiv.org/pdf/2511.02424v1)

Authors: Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Minsu Jang, Dohyung Kim, Jaehong Kim, Youngwoo Yoon

Recent advancements in large language models (LLMs) have enabled significant
progress in decision-making and task planning for embodied autonomous agents.
However, most existing methods still struggle with complex, long-horizon tasks
because they rely on a monolithic trajectory that entangles all past decisions
and observations, attempting to solve the entire task in a single unified
process. To address this limitation, we propose ReAcTree, a hierarchical
task-planning method that decomposes a complex goal into more manageable
subgoals within a dynamically constructed agent tree. Each subgoal is handled
by an LLM agent node capable of reasoning, acting, and further expanding the
tree, while control flow nodes coordinate the execution strategies of agent
nodes. In addition, we integrate two complementary memory systems: each agent
node retrieves goal-specific, subgoal-level examples from episodic memory and
shares environment-specific observations through working memory. Experiments on
the WAH-NL and ALFRED datasets demonstrate that ReAcTree consistently
outperforms strong task-planning baselines such as ReAct across diverse LLMs.
Notably, on WAH-NL, ReAcTree achieves a 61% goal success rate with Qwen 2.5
72B, nearly doubling ReAct's 31%.

### 8. [Auditable-choice reframing unlocks RL-based verification for open-ended tasks](http://arxiv.org/pdf/2511.02463v1)

Authors: Mengyu Zhang, Xubo Liu, Siyu Ding, Weichong Yin, Yu Sun, Hua Wu, Wenya Guo, Ying Zhang

Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated great
potential in enhancing the reasoning capabilities of large language models
(LLMs), achieving remarkable progress in domains such as mathematics and
programming where standard answers are available. However, for open-ended tasks
lacking ground-truth solutions (e.g., creative writing and instruction
following), existing studies typically regard them as non-reasoning scenarios,
thereby overlooking the latent value of reasoning capabilities. This raises a
key question: Can strengthening reasoning improve performance in open-ended
tasks? To address this, we explore the transfer of the RLVR paradigm to the
open domain. Yet, since RLVR fundamentally relies on verifiers that presuppose
the existence of standard answers, it cannot be directly applied to open-ended
tasks. To overcome this challenge, we introduce Verifiable Multiple-Choice
Reformulation (VMR), a novel training strategy that restructures open-ended
data into verifiable multiple-choice formats, enabling effective training even
in the absence of explicit ground truth. Experimental results on multiple
benchmarks validate the effectiveness of our method in improving LLM
performance on open-ended tasks. Notably, across eight open-ended benchmarks,
our VMR-based training delivers an average gain of 5.99 points over the
baseline. Code will be released upon acceptance to facilitate reproducibility.

### 9. [Knowledge Graph-enhanced Large Language Model for Incremental Game PlayTesting](http://arxiv.org/pdf/2511.02534v1)

Authors: Enhong Mu, Jinyu Cai, Yijun Lu, Mingyue Zhang, Kenji Tei, Jialong Li

The rapid iteration and frequent updates of modern video games pose
significant challenges to the efficiency and specificity of testing. Although
automated playtesting methods based on Large Language Models (LLMs) have shown
promise, they often lack structured knowledge accumulation mechanisms, making
it difficult to conduct precise and efficient testing tailored for incremental
game updates. To address this challenge, this paper proposes a KLPEG framework.
The framework constructs and maintains a Knowledge Graph (KG) to systematically
model game elements, task dependencies, and causal relationships, enabling
knowledge accumulation and reuse across versions. Building on this foundation,
the framework utilizes LLMs to parse natural language update logs, identify the
scope of impact through multi-hop reasoning on the KG, enabling the generation
of update-tailored test cases. Experiments in two representative game
environments, Overcooked and Minecraft, demonstrate that KLPEG can more
accurately locate functionalities affected by updates and complete tests in
fewer steps, significantly improving both playtesting effectiveness and
efficiency.

### 10. [The ORCA Benchmark: Evaluating Real-World Calculation Accuracy in Large Language Models](http://arxiv.org/pdf/2511.02589v1)

Authors: Claudia Herambourg, Dawid Siuda, Anna Szczepanek, Julia Kopczyńska, Joao R. L. Santos, Wojciech Sas, Joanna Śmietańska-Nowak

We present ORCA (Omni Research on Calculation in AI) Benchmark -- a novel
benchmark that evaluates large language models (LLMs) on multi-domain,
real-life quantitative reasoning using verified outputs from Omni's calculator
engine. In 500 natural-language tasks across domains such as finance, physics,
health, and statistics, the five state-of-the-art systems (ChatGPT-5,
Gemini~2.5~Flash, Claude~Sonnet~4.5, Grok~4, and DeepSeek~V3.2) achieved only
$45\text{--}63\,\%$ accuracy, with errors mainly related to rounding ($35\,\%$)
and calculation mistakes ($33\,\%$). Results in specific domains indicate
strengths in mathematics and engineering, but weaknesses in physics and natural
sciences. Correlation analysis ($r \approx 0.40\text{--}0.65$) shows that the
models often fail together but differ in the types of errors they make,
highlighting their partial complementarity rather than redundancy. Unlike
standard math datasets, ORCA evaluates step-by-step reasoning, numerical
precision, and domain generalization across real problems from finance,
physics, health, and statistics.

### Hardware Architecture

### 1. [Energy-Efficient Hardware Acceleration of Whisper ASR on a CGLA](http://arxiv.org/pdf/2511.02269v1)

Authors: Takuto Ando, Yu Eto, Ayumu Takeuchi, Yasuhiko Nakashima

The rise of generative AI for tasks like Automatic Speech Recognition (ASR)
has created a critical energy consumption challenge. While ASICs offer high
efficiency, they lack the programmability to adapt to evolving algorithms. To
address this trade-off, we implement and evaluate Whisper's core computational
kernel on the IMAX, a general-purpose Coarse-Grained Linear Arrays (CGLAs)
accelerator. To our knowledge, this is the first work to execute a Whisper
kernel on a CGRA and compare its performance against CPUs and GPUs. Using
hardware/software co-design, we evaluate our system via an FPGA prototype and
project performance for a 28 nm ASIC. Our results demonstrate superior energy
efficiency. The projected ASIC is 1.90x more energy-efficient than the NVIDIA
Jetson AGX Orin and 9.83x more than an NVIDIA RTX 4090 for the Q8_0 model. This
work positions CGLA as a promising platform for sustainable ASR on
power-constrained edge devices.

### 2. [Facial Expression Recognition System Using DNN Accelerator with Multi-threading on FPGA](http://arxiv.org/pdf/2511.02408v1)

Authors: Takuto Ando, Yusuke Inoue

In this paper, we implement a stand-alone facial expression recognition
system on an SoC FPGA with multi-threading using a Deep learning Processor Unit
(DPU). The system consists of two steps: one for face detection step and one
for facial expression recognition. In the previous work, the Haar Cascade
detector was run on a CPU in the face detection step due to FPGA resource
limitations, but this detector is less accurate for profile and variable
illumination condition images. Moreover, the previous work used a dedicated
circuit accelerator, so running a second DNN inference for face detection on
the FPGA would require the addition of a new accelerator. As an alternative to
this approach, we run the two inferences by DNN on a DPU, which is a
general-purpose CNN accelerator of the systolic array type. Our method for face
detection using DenseBox and facial expression recognition using CNN on the
same DPU enables the efficient use of FPGA resources while maintaining a small
circuit size. We also developed a multi-threading technique that improves the
overall throughput while increasing the DPU utilization efficiency. With this
approach, we achieved an overall system throughput of 25 FPS and a throughput
per power consumption of 2.4 times.

### 3. [Digit-Recurrence Posit Division](http://arxiv.org/pdf/2511.02494v1)

Authors: Raul Murillo, Julio Villalba-Moreno, Alberto A. Del Barrio, Guillermo Botella

Posit arithmetic has emerged as a promising alternative to IEEE 754
floating-point representation, offering enhanced accuracy and dynamic range.
However, division operations in posit systems remain challenging due to their
inherent hardware complexity. In this work, we present posit division units
based on the digit-recurrence algorithm, marking the first implementation of
radix-4 digit-recurrence techniques within this context. Our approach
incorporates hardware-centric optimizations including redundant arithmetic,
on-the-fly quotient conversion, and operand scaling to streamline the division
process while mitigating latency, area, and power overheads. Comprehensive
synthesis evaluations across multiple posit configurations demonstrate
significant performance improvements, including more than 80% energy reduction
with small area overhead compared to existing methods, and a substantial
decrease in the number of iterations. These results underscore the potential of
our adapted algorithm to enhance the efficiency of posit-based arithmetic
units.

### 4. [Implementation and Evaluation of Stable Diffusion on a General-Purpose CGLA Accelerator](http://arxiv.org/pdf/2511.02530v1)

Authors: Takuto Ando, Yu Eto, Yasuhiko Nakashima

This paper presents the first implementation and in-depth evaluation of the
primary computational kernels from the stable-diffusion.cpp image generation
framework on IMAX3, a general-purpose Coarse-Grained Reconfigurable Array
(CGRA) accelerator. We designed IMAX3 as a versatile computational platform,
and this work assesses its capabilities by executing a demanding image
generation workload. We evaluate its performance on a current
Field-Programmable Gate Array (FPGA) prototype to establish a baseline and
project its potential for a future Application-Specific Integrated Circuit
(ASIC) implementation. Our results demonstrate that, despite its
general-purpose architecture, IMAX3 achieves promising performance and power
efficiency, particularly in its projected ASIC form. This work provides
concrete guidelines for future IMAX architectural designs and establishes a
foundation for developing next-generation, AI-specialized Coarse-Grained Linear
Array (CGLA) accelerators by refining this versatile platform. Ultimately, this
achievement contributes to the realization of energy-efficient, on-device,
multi-modal AI platforms.

### 5. [BoolSkeleton: Boolean Network Skeletonization via Homogeneous Pattern Reduction](http://arxiv.org/pdf/2511.02196v1)

Authors: Liwei Ni, Jiaxi Zhang, Shenggen Zheng, Junfeng Liu, Xingyu Meng, Biwei Xie, Xingquan Li, Huawei Li

Boolean equivalence allows Boolean networks with identical functionality to
exhibit diverse graph structures. This gives more room for exploration in logic
optimization, while also posing a challenge for tasks involving consistency
between Boolean networks. To tackle this challenge, we introduce BoolSkeleton,
a novel Boolean network skeletonization method that improves the consistency
and reliability of design-specific evaluations. BoolSkeleton comprises two key
steps: preprocessing and reduction. In preprocessing, the Boolean network is
transformed into a defined Boolean dependency graph, where nodes are assigned
the functionality-related status. Next, the homogeneous and heterogeneous
patterns are defined for the node-level pattern reduction step. Heterogeneous
patterns are preserved to maintain critical functionality-related dependencies,
while homogeneous patterns can be reduced. Parameter K of the pattern further
constrains the fanin size of these patterns, enabling fine-tuned control over
the granularity of graph reduction. To validate BoolSkeleton's effectiveness,
we conducted four analysis/downstream tasks around the Boolean network:
compression analysis, classification, critical path analysis, and timing
prediction, demonstrating its robustness across diverse scenarios. Furthermore,
it improves above 55% in the average accuracy compared to the original Boolean
network for the timing prediction task. These experiments underscore the
potential of BoolSkeleton to enhance design consistency in logic synthesis.

### 6. [VFocus: Better Verilog Generation from Large Language Model via Focused Reasoning](http://arxiv.org/pdf/2511.02285v1)

Authors: Zhuorui Zhao, Bing Li, Grace Li Zhang, Ulf Schlichtmann

Large Language Models (LLMs) have shown impressive potential in generating
Verilog codes, but ensuring functional correctness remains a challenge.
Existing approaches often rely on self-consistency or simulation feedback to
select the best candidate, but they miss opportunities to focus LLM reasoning
on the most informative parts of the design. We propose VFocus, a three-stage
framework that enhances Verilog generation by sharpening the focus of LLM
reasoning onto critical decision points in the code generation process. In the
\textbf{pre-ranking stage}, VFocus generates multiple code candidates through
LLM prompting, retries for syntactically valid outputs, and introduces a
\textit{Density-guided Filtering} to retain candidates that fall within the
"reasoning sweet spot" for functional correctness. In the \textbf{ranking
stage}, we simulate each code candidate using an automatically generated
testbench and apply self-consistency-based clustering to identify the most
consistent outputs. Finally, in the \textbf{post-ranking refinement stage},
VFocus performs inconsistency mining on top-ranked candidates and invokes
reasoning-augmented LLM prompts for candidate refinement. Experiments on the
VerilogEval-Human benchmark show that VFocus significantly improves the pass@1
correctness across multiple reasoning LLMs, demonstrating its effectiveness in
enhancing Verilog generation for complex hardware design tasks.

### Computational Complexity

### 1. [Spectral Certificates and Sum-of-Squares Lower Bounds for Semirandom Hamiltonians](http://arxiv.org/pdf/2511.02264v1)

Authors: Nicholas Kocurek

The $k$-$\mathsf{XOR}$ problem is one of the most well-studied problems in
classical complexity. We study a natural quantum analogue of
$k$-$\mathsf{XOR}$, the problem of computing the ground energy of a certain
subclass of structured local Hamiltonians, signed sums of $k$-local Pauli
operators, which we refer to as $k$-$\mathsf{XOR}$ Hamiltonians. As an
exhibition of the connection between this model and classical
$k$-$\mathsf{XOR}$, we extend results on refuting $k$-$\mathsf{XOR}$ instances
to the Hamiltonian setting by crafting a quantum variant of the Kikuchi matrix
for CSP refutation, instead capturing ground energy optimization. As our main
result, we show an $n^{O(\ell)}$-time classical spectral algorithm certifying
ground energy at most $\frac{1}{2} + \varepsilon$ in (1) semirandom Hamiltonian
$k$-$\mathsf{XOR}$ instances or (2) sums of Gaussian-signed $k$-local Paulis
both with $O(n) \cdot \left(\frac{n}{\ell}\right)^{k/2-1} \log n
/\varepsilon^4$ local terms, a tradeoff known as the refutation threshold.
Additionally, we give evidence this tradeoff is tight in the semirandom regime
via non-commutative Sum-of-Squares lower bounds embedding classical
$k$-$\mathsf{XOR}$ instances as entirely classical Hamiltonians.

### 2. [Relaxed vs. Full Local Decodability with Few Queries: Equivalence and Separations for Linear Codes](http://arxiv.org/pdf/2511.02633v1)

Authors: Elena Grigorescu, Vinayak M. Kumar, Peter Manohar, Geoffrey Mon

A locally decodable code (LDC) $C \colon \{0,1\}^k \to \{0,1\}^n$ is an
error-correcting code that allows one to recover any bit of the original
message with good probability while only reading a small number of bits from a
corrupted codeword. A relaxed locally decodable code (RLDC) is a weaker notion
where the decoder is additionally allowed to abort and output a special symbol
$\bot$ if it detects an error. For a large constant number of queries $q$,
there is a large gap between the blocklength $n$ of the best $q$-query LDC and
the best $q$-query RLDC. Existing constructions of RLDCs achieve polynomial
length $n = k^{1 + O(1/q)}$, while the best-known $q$-LDCs only achieve
subexponential length $n = 2^{k^{o(1)}}$. On the other hand, for $q = 2$, it is
known that RLDCs and LDCs are equivalent. We thus ask the question: what is the
smallest $q$ such that there exists a $q$-RLDC that is not a $q$-LDC?
  In this work, we show that any linear $3$-query RLDC is in fact a $3$-LDC,
i.e., linear RLDCs and LDCs are equivalent at $3$ queries. More generally, we
show for any constant $q$, there is a soundness error threshold $s(q)$ such
that any linear $q$-RLDC with soundness error below this threshold must be a
$q$-LDC. This implies that linear RLDCs cannot have "strong soundness" -- a
stricter condition satisfied by linear LDCs that says the soundness error is
proportional to the fraction of errors in the corrupted codeword -- unless they
are simply LDCs.
  In addition, we give simple constructions of linear $15$-query RLDCs that are
not $q$-LDCs for any constant $q$, showing that for $q = 15$, linear RLDCs and
LDCs are not equivalent.
  We also prove nearly identical results for locally correctable codes and
their corresponding relaxed counterpart.

### 3. [Tensor rank and dimension expanders](http://arxiv.org/pdf/2511.02670v1)

Authors: Zeev Dvir

We prove a lower bound on the rank of tensors constructed from families of
linear maps that `expand' the dimension of every subspace. Such families,
called {\em dimension expanders} have been studied for many years with several
known explicit constructions. Using these constructions we show that one can
construct an explicit $[D]\times [n] \times [n]$-tensor with rank at least $(2
- \epsilon)n$, with $D$ a constant depending on $\epsilon$. Our results extend
to border rank over the real or complex numbers.

### 4. [Fast Approximation Algorithm for Non-Monotone DR-submodular Maximization under Size Constraint](http://arxiv.org/pdf/2511.02254v1)

Authors: Tan D. Tran, Canh V. Pham

This work studies the non-monotone DR-submodular Maximization over a ground
set of $n$ subject to a size constraint $k$. We propose two approximation
algorithms for solving this problem named FastDrSub and FastDrSub++. FastDrSub
offers an approximation ratio of $0.044$ with query complexity of $O(n
\log(k))$. The second one, FastDrSub++, improves upon it with a ratio of
$1/4-\epsilon$ within query complexity of $(n \log k)$ for an input parameter
$\epsilon >0$. Therefore, our proposed algorithms are the first constant-ratio
approximation algorithms for the problem with the low complexity of $O(n
\log(k))$.
  Additionally, both algorithms are experimentally evaluated and compared
against existing state-of-the-art methods, demonstrating their effectiveness in
solving the Revenue Maximization problem with DR-submodular objective function.
The experimental results show that our proposed algorithms significantly
outperform existing approaches in terms of both query complexity and solution
quality.

### 5. [Complexity of counting points on curves and the factor $P_1(T)$ of the zeta function of surfaces](http://arxiv.org/pdf/2511.02262v1)

Authors: Diptajit Roy, Nitin Saxena, Madhavan Venkatesh

This article concerns the computational complexity of a fundamental problem
in number theory: counting points on curves and surfaces over finite fields.
There is no subexponential-time algorithm known and it is unclear if it can be
$\mathrm{NP}$-hard.
  Given a curve, we present the first efficient Arthur-Merlin protocol to
certify its point-count, its Jacobian group structure, and its Hasse-Weil zeta
function. We extend this result to a smooth projective surface to certify the
factor $P_{1}(T)$, corresponding to the first Betti number, of the zeta
function; by using the counting oracle. We give the first algorithm to compute
$P_{1}(T)$ that is poly($\log q$)-time if the degree $D$ of the input surface
is fixed; and in quantum poly($D\log q$)-time in general.
  Our technique in the curve case, is to sample hash functions using the Weil
and Riemann-Roch bounds, to certify the group order of its Jacobian. For higher
dimension varieties, we first reduce to the case of a surface, which is fibred
as a Lefschetz pencil of hyperplane sections over $\mathbb{P}^{1}$. The
formalism of vanishing cycles, and the inherent big monodromy, enable us to
prove an effective version of Deligne's `theoreme du pgcd' using the
hard-Lefschetz theorem and an equidistribution result due to Katz. These reduce
our investigations to that of computing the zeta function of a curve, defined
over a finite field extension $\mathbb{F}_{Q}/\mathbb{F}_{q}$ of poly-bounded
degree. This explicitization of the theory yields the first nontrivial upper
bounds on the computational complexity.

### 6. [Non-commutative linear logic fragments with sub-context-free complexity](http://arxiv.org/pdf/2511.02348v1)

Authors: Yusaku Nishimiya, Masaya Taniguchi

We present new descriptive complexity characterisations of classes REG
(regular languages), LCFL (linear context-free languages) and CFL (context-free
languages) as restrictions on inference rules, size of formulae and permitted
connectives in the Lambek calculus; fragments of the intuitionistic
non-commutative linear logic with direction-sensitive implication connectives.
Our identification of the Lambek calculus fragments with proof complexity REG
and LCFL is the first result of its kind. We further show the CFL complexity of
one of the strictly `weakest' possible variants of the logic, admitting only a
single inference rule. The proof thereof, moreover, is based on a direct
translation between type-logical and formal grammar and structural induction on
provable sequents; a simpler and more intuitive method than those employed in
prior works. We thereby establish a clear conceptual utility of the
Cut-elimination theorem for comparing formal grammar and sequent calculus, and
identify the exact analogue of the Greibach Normal Form in Lambek grammar. We
believe the result presented herein constitutes a first step toward a more
extensive and richer characterisation of the interaction between computation
and logic, as well as a finer-grained complexity separation of various sequent
calculi.

### 7. [Recursively Enumerably Representable Classes and Computable Versions of the Fundamental Theorem of Statistical Learning](http://arxiv.org/pdf/2511.02644v1)

Authors: David Kattermann, Lothar Sebastian Krapp

We study computable probably approximately correct (CPAC) learning, where
learners are required to be computable functions. It had been previously
observed that the Fundamental Theorem of Statistical Learning, which
characterizes PAC learnability by finiteness of the Vapnik-Chervonenkis
(VC-)dimension, no longer holds in this framework. Recent works recovered
analogs of the Fundamental Theorem in the computable setting, for instance by
introducing an effective VC-dimension. Guided by this, we investigate the
connection between CPAC learning and recursively enumerable representable (RER)
classes, whose members can be algorithmically listed. Our results show that the
effective VC-dimensions can take arbitrary values above the traditional one,
even for RER classes, which creates a whole family of (non-)examples for
various notions of CPAC learning. Yet the two dimensions coincide for classes
satisfying sufficiently strong notions of CPAC learning. We then observe that
CPAC learnability can also be characterized via containment of RER classes that
realize the same samples. Furthermore, it is shown that CPAC learnable classes
satisfying a unique identification property are necessarily RER. Finally, we
establish that agnostic learnability can be guaranteed for RER classes, by
considering the relaxed notion of nonuniform CPAC learning.

### 8. [Arithmetic Circuits and Neural Networks for Regular Matroids](http://arxiv.org/pdf/2511.02406v1)

Authors: Christoph Hertrich, Stefan Kober, Georg Loho

We prove that there exist uniform $(+,\times,/)$-circuits of size $O(n^3)$ to
compute the basis generating polynomial of regular matroids on $n$ elements. By
tropicalization, this implies that there exist uniform $(\max,+,-)$-circuits
and ReLU neural networks of the same size for weighted basis maximization of
regular matroids. As a consequence in linear programming theory, we obtain a
first example where taking the difference of two extended formulations can be
more efficient than the best known individual extended formulation of size
$O(n^6)$ by Aprile and Fiorini. Such differences have recently been introduced
as virtual extended formulations. The proof of our main result relies on a
fine-tuned version of Seymour's decomposition of regular matroids which allows
us to identify and maintain graphic substructures to which we can apply a local
version of the star-mesh transformation.

### Computational Engineering

### 1. [Wavelet-Optimized Motion Artifact Correction in 3D MRI Using Pre-trained 2D Score Priors](http://arxiv.org/pdf/2511.02256v1)

Authors: Genyuan Zhang, Xuyang Duan, Songtao Zhu, Ao Wang, Fenglin Liu

Motion artifacts in magnetic resonance imaging (MRI) remain a major
challenge, as they degrade image quality and compromise diagnostic reliability.
Score-based generative models (SGMs) have recently shown promise for artifact
removal. However, existing 3D SGM-based approaches are limited in two key
aspects: (1) their strong dependence on known forward operators makes them
ineffective for correcting MRI motion artifacts, and (2) their slow inference
speed hinders clinical translation. To overcome these challenges, we propose a
wavelet-optimized end-to-end framework for 3D MRI motion correct using
pre-trained 2D score priors (3D-WMoCo). Specifically, two orthogonal 2D score
priors are leveraged to guide the 3D distribution prior, while a mean-reverting
stochastic differential equation (SDE) is employed to model the restoration
process of motion-corrupted 3D volumes to motion-free 3D distribution.
Furthermore, wavelet diffusion is introduced to accelerate inference, and
wavelet convolution is applied to enhance feature extraction. We validate the
effectiveness of our approach through both simulated motion artifact
experiments and real-world clinical motion artifact correction tests. The
proposed method achieves robust performance improvements over existing
techniques. Implementation details and source code are available at:
https://github.com/ZG-yuan/3D-WMoCo.

### 2. [A Multi-Fidelity Global Search Framework for Hotspot Prevention in 3D Thermal Design Space](http://arxiv.org/pdf/2511.02211v1)

Authors: Morteza Sadeghi, Hadi Keramati, Sajjad Bigham

We present a B\'ezier-based Multi-Fidelity Thermal Optimization Framework,
which is a computationally efficient methodology for the global optimization of
3D heat sinks. The flexible B\'ezier-parameterized fin geometries and the
adopted multi-fidelity pseudo-3D thermal modeling strategy meet at a balance
between accuracy and computational cost. In this method, the smooth and compact
B\'ezier representation of fins defines the design space from which diverse
topologies can be generated with minimal design variables. A global optimizer,
the Covariance Matrix Adaptation Evolution Strategy, minimizes the pressure
drop with respect to a given surface-average temperature constraint to achieve
improvement in the pressure loss. In the framework, the pseudo-3D model couples
two thermally interacting 2D layers: a thermofluid layer representing the fluid
domain passing through the fins, and a conductive base plate representing the
surface where excessive average temperature is to be avoided. Both layers are
coupled with calibrated heat transfer coefficients obtained from high-fidelity
3D simulations. For several fin geometries, the proposed framework has been
validated by comparing the pseudo-3D results with those of full 3D simulations,
which yielded good agreement in terms of temperature distribution and pressure
drops when the computational cost was reduced by several orders of magnitude.
Optimization results show that it attains up to 50\% pressure loss reduction
compared to conventional straight-fin configurations, and it reveals a clear
trade-off between thermal performance and hydraulic efficiency. Thus, the
proposed method forms a new basis for fast, geometry-flexible, and optimized
heat sink design, enabling efficient exploration of complex geometries.

### 3. [Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas](http://arxiv.org/pdf/2511.02458v1)

Authors: Giulia Iadisernia, Carolina Camassa

We evaluate whether persona-based prompting improves Large Language Model
(LLM) performance on macroeconomic forecasting tasks. Using 2,368
economics-related personas from the PersonaHub corpus, we prompt GPT-4o to
replicate the ECB Survey of Professional Forecasters across 50 quarterly rounds
(2013-2025). We compare the persona-prompted forecasts against the human
experts panel, across four target variables (HICP, core HICP, GDP growth,
unemployment) and four forecast horizons. We also compare the results against
100 baseline forecasts without persona descriptions to isolate its effect. We
report two main findings. Firstly, GPT-4o and human forecasters achieve
remarkably similar accuracy levels, with differences that are statistically
significant yet practically modest. Our out-of-sample evaluation on 2024-2025
data demonstrates that GPT-4o can maintain competitive forecasting performance
on unseen events, though with notable differences compared to the in-sample
period. Secondly, our ablation experiment reveals no measurable forecasting
advantage from persona descriptions, suggesting these prompt components can be
omitted to reduce computational costs without sacrificing accuracy. Our results
provide evidence that GPT-4o can achieve competitive forecasting accuracy even
on out-of-sample macroeconomic events, if provided with relevant context data,
while revealing that diverse prompts produce remarkably homogeneous forecasts
compared to human panels.

### 4. [In Situ Training of Implicit Neural Compressors for Scientific Simulations via Sketch-Based Regularization](http://arxiv.org/pdf/2511.02659v1)

Authors: Cooper Simpson, Stephen Becker, Alireza Doostan

Focusing on implicit neural representations, we present a novel in situ
training protocol that employs limited memory buffers of full and sketched data
samples, where the sketched data are leveraged to prevent catastrophic
forgetting. The theoretical motivation for our use of sketching as a
regularizer is presented via a simple Johnson-Lindenstrauss-informed result.
While our methods may be of wider interest in the field of continual learning,
we specifically target in situ neural compression using implicit neural
representation-based hypernetworks. We evaluate our method on a variety of
complex simulation data in two and three dimensions, over long time horizons,
and across unstructured grids and non-Cartesian geometries. On these tasks, we
show strong reconstruction performance at high compression rates. Most
importantly, we demonstrate that sketching enables the presented in situ scheme
to approximately match the performance of the equivalent offline method.

### 5. [Natural-gas storage modelling by deep reinforcement learning](http://arxiv.org/pdf/2511.02646v1)

Authors: Tiziano Balaconi, Aldo Glielmo, Marco Taboga

We introduce GasRL, a simulator that couples a calibrated representation of
the natural gas market with a model of storage-operator policies trained with
deep reinforcement learning (RL). We use it to analyse how optimal stockpile
management affects equilibrium prices and the dynamics of demand and supply. We
test various RL algorithms and find that Soft Actor Critic (SAC) exhibits
superior performance in the GasRL environment: multiple objectives of storage
operators - including profitability, robust market clearing and price
stabilisation - are successfully achieved. Moreover, the equilibrium price
dynamics induced by SAC-derived optimal policies have characteristics, such as
volatility and seasonality, that closely match those of real-world prices.
Remarkably, this adherence to the historical distribution of prices is obtained
without explicitly calibrating the model to price data. We show how the
simulator can be used to assess the effects of EU-mandated minimum storage
thresholds. We find that such thresholds have a positive effect on market
resilience against unanticipated shifts in the distribution of supply shocks.
For example, with unusually large shocks, market disruptions are averted more
often if a threshold is in place.

### Computational Geometry

### 1. [Optimizing Kernel Discrepancies via Subset Selection](http://arxiv.org/pdf/2511.02706v1)

Authors: Deyao Chen, François Clément, Carola Doerr, Nathan Kirk

Kernel discrepancies are a powerful tool for analyzing worst-case errors in
quasi-Monte Carlo (QMC) methods. Building on recent advances in optimizing such
discrepancy measures, we extend the subset selection problem to the setting of
kernel discrepancies, selecting an m-element subset from a large population of
size $n \gg m$. We introduce a novel subset selection algorithm applicable to
general kernel discrepancies to efficiently generate low-discrepancy samples
from both the uniform distribution on the unit hypercube, the traditional
setting of classical QMC, and from more general distributions $F$ with known
density functions by employing the kernel Stein discrepancy. We also explore
the relationship between the classical $L_2$ star discrepancy and its
$L_\infty$ counterpart.

### Computation and Language

### 1. [IG-Pruning: Input-Guided Block Pruning for Large Language Models](http://arxiv.org/pdf/2511.02213v1)

Authors: Kangyu Qiao, Shaolei Zhang, Yang Feng

With the growing computational demands of large language models (LLMs),
efficient inference has become increasingly critical for practical deployment.
Depth pruning has emerged as a promising approach for reducing the
computational costs of large language models by removing transformer layers.
However, existing methods typically rely on fixed block masks, which can lead
to suboptimal performance across different tasks and inputs. In this paper, we
propose IG-Pruning, a novel input-aware block-wise pruning method that
dynamically selects layer masks at inference time. Our approach consists of two
stages: (1) Discovering diverse mask candidates through semantic clustering and
L0 optimization, and (2) Implementing efficient dynamic pruning without the
need for extensive training. Experimental results demonstrate that our method
consistently outperforms state-of-the-art static depth pruning methods, making
it particularly suitable for resource-constrained deployment scenarios.

### 2. [LTD-Bench: Evaluating Large Language Models by Letting Them Draw](http://arxiv.org/pdf/2511.02347v1)

Authors: Liuhao Lin, Ke Li, Zihan Xu, Yuchen Shi, Yulei Qin, Yan Zhang, Xing Sun, Rongrong Ji

Current evaluation paradigms for large language models (LLMs) represent a
critical blind spot in AI research--relying on opaque numerical metrics that
conceal fundamental limitations in spatial reasoning while providing no
intuitive understanding of model capabilities. This deficiency creates a
dangerous disconnect between reported performance and practical abilities,
particularly for applications requiring physical world understanding. We
introduce LTD-Bench, a breakthrough benchmark that transforms LLM evaluation
from abstract scores to directly observable visual outputs by requiring models
to generate drawings through dot matrices or executable code. This approach
makes spatial reasoning limitations immediately apparent even to non-experts,
bridging the fundamental gap between statistical performance and intuitive
assessment. LTD-Bench implements a comprehensive methodology with complementary
generation tasks (testing spatial imagination) and recognition tasks (assessing
spatial perception) across three progressively challenging difficulty levels,
methodically evaluating both directions of the critical language-spatial
mapping. Our extensive experiments with state-of-the-art models expose an
alarming capability gap: even LLMs achieving impressive results on traditional
benchmarks demonstrate profound deficiencies in establishing bidirectional
mappings between language and spatial concept--a fundamental limitation that
undermines their potential as genuine world models. Furthermore, LTD-Bench's
visual outputs enable powerful diagnostic analysis, offering a potential
approach to investigate model similarity.

### 3. [LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context](http://arxiv.org/pdf/2511.02366v1)

Authors: Yudong Li, Zhongliang Yang, Kejiang Chen, Wenxuan Wang, Tianxin Zhang, Sifang Wan, Kecheng Wang, Haitian Li, Xu Wang, Lefan Cheng, Youdan Yang, Baocheng Chen, Ziyu Liu, Yufei Sun, Liyan Wu, Wenya Wen, Xingchi Gu, Peiru Yang

In this work, we propose LiveSecBench, a dynamic and continuously updated
safety benchmark specifically for Chinese-language LLM application scenarios.
LiveSecBench evaluates models across six critical dimensions (Legality, Ethics,
Factuality, Privacy, Adversarial Robustness, and Reasoning Safety) rooted in
the Chinese legal and social frameworks. This benchmark maintains relevance
through a dynamic update schedule that incorporates new threat vectors, such as
the planned inclusion of Text-to-Image Generation Safety and Agentic Safety in
the next update. For now, LiveSecBench (v251030) has evaluated 18 LLMs,
providing a landscape of AI safety in the context of Chinese language. The
leaderboard is publicly accessible at https://livesecbench.intokentech.cn/.

### 4. [Merging Continual Pretraining Models for Domain-Specialized LLMs: A Case Study in Finance](http://arxiv.org/pdf/2511.02451v1)

Authors: Kentaro Ueda, François Portet, Hirohiko Suwa, Keiichi Yasumoto

While LLMs excel at general tasks, they struggle in specialized domains like
finance, requiring diverse skills in domain knowledge, mathematical reasoning,
and multilingual processing. Merging domain-specific Continual Pre-training
(CPT) "experts" offers a practical alternative to costly and unstable
multi-skill training. However, unlike established Supervised Fine-Tuning (SFT)
model-based merging, CPT model merging remains largely unexplored. We address
this gap by creating financial LLMs from experts in finance, math, and
Japanese. We propose a three-stage evaluation focusing on knowledge recovery,
complementarity, and emergence, and assess three merging methods (Task
Arithmetic, TIES, and DARE-TIES) on a comprehensive financial benchmark curated
from 18 tasks across 8 established datasets. Results show that merging an
expert with its base model recovers general knowledge lost during CPT, while
merging experts improves performance and can yield emergent cross-domain
skills. Among the methods, Task Arithmetic performs strongly but is
hyperparameter-sensitive, whereas TIES is more robust. Our findings also
suggest that while model similarity correlates with merging success, emergent
skills depend on more complex factors. This work presents the first
foundational analysis of CPT model merging, establishing a principled framework
and providing clear guidance for building multi-skill LLMs from existing
assets.

### 5. [Smart-Hiring: An Explainable end-to-end Pipeline for CV Information Extraction and Job Matching](http://arxiv.org/pdf/2511.02537v1)

Authors: Kenza Khelkhal, Dihia Lanasri

Hiring processes often involve the manual screening of hundreds of resumes
for each job, a task that is time and effort consuming, error-prone, and
subject to human bias. This paper presents Smart-Hiring, an end-to-end Natural
Language Processing (NLP) pipeline de- signed to automatically extract
structured information from unstructured resumes and to semantically match
candidates with job descriptions. The proposed system combines document
parsing, named-entity recognition, and contextual text embedding techniques to
capture skills, experience, and qualifications. Using advanced NLP technics,
Smart-Hiring encodes both resumes and job descriptions in a shared vector space
to compute similarity scores between candidates and job postings. The pipeline
is modular and explainable, allowing users to inspect extracted entities and
matching rationales. Experiments were conducted on a real-world dataset of
resumes and job descriptions spanning multiple professional domains,
demonstrating the robustness and feasibility of the proposed approach. The
system achieves competitive matching accuracy while preserving a high degree of
interpretability and transparency in its decision process. This work introduces
a scalable and practical NLP frame- work for recruitment analytics and outlines
promising directions for bias mitigation, fairness-aware modeling, and
large-scale deployment of data-driven hiring solutions.

### 6. [The Analysis of Lexical Errors in Machine Translation from English into Romanian](http://arxiv.org/pdf/2511.02587v1)

Authors: Angela Stamatie

The research explores error analysis in the performance of translating by
Machine Translation from English into Romanian, and it focuses on lexical
errors found in texts which include official information, provided by the World
Health Organization (WHO), the Gavi Organization, by the patient information
leaflet (the information about the active ingredients of the vaccines or the
medication, the indications, the dosage instructions, the storage instructions,
the side effects and warning, etc.). All of these texts are related to Covid-19
and have been translated by Google Translate, a multilingual Machine
Translation that was created by Google. In the last decades, Google has
actively worked to develop a more accurate and fluent automatic translation
system. This research, specifically focused on improving Google Translate, aims
to enhance the overall quality of Machine Translation by achieving better
lexical selection and by reducing errors. The investigation involves a
comprehensive analysis of 230 texts that have been translated from English into
Romanian.

### 7. [CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency](http://arxiv.org/pdf/2511.02603v1)

Authors: Ehsan Aghazadeh, Ahmad Ghasemi, Hedyeh Beyhaghi, Hossein Pishro-Nik

Large language models (LLMs) are often queried multiple times at test time,
with predictions aggregated by majority vote. While effective, this
self-consistency strategy (arXiv:2203.11171) requires a fixed number of calls
and can fail when the correct answer is rare. We introduce Confidence-Guided
Early Stopping (CGES), a Bayesian framework that forms posteriors over
candidate answers using scalar confidence signals derived from token
probabilities or reward models. CGES adaptively halts sampling once the
posterior mass of a candidate exceeds a threshold. We provide theoretical
guarantees for both perfectly calibrated confidences and realistic noisy
confidence signals. Across five reasoning benchmarks, CGES reduces the average
number of model calls by about 69 percent (for example, from 16.0 to 4.9) while
matching the accuracy of self-consistency within 0.06 percentage points.

### 8. [The Realignment Problem: When Right becomes Wrong in LLMs](http://arxiv.org/pdf/2511.02623v1)

Authors: Aakash Sen Sharma, Debdeep Sanyal, Vivek Srivastava, Shirish Karande, Murari Mandal

The alignment of Large Language Models (LLMs) with human values is central to
their safe deployment, yet current practice produces static, brittle, and
costly-to-maintain models that fail to keep pace with evolving norms and
policies. This misalignment, which we term the Alignment-Reality Gap, poses a
growing challenge for reliable long-term use. Existing remedies are inadequate:
large-scale re-annotation is economically prohibitive, and standard unlearning
methods act as blunt instruments that erode utility rather than enable precise
policy updates. We introduce TRACE (Triage and Re-align by Alignment Conflict
Evaluation), a framework for principled unlearning that reconceives
re-alignment as a programmatic policy application problem. TRACE
programmatically triages existing preference data against a new policy,
identifies high-impact conflicts via a alignment impact score, and applies a
hybrid optimization that cleanly inverts, discards, or preserves preferences
while safeguarding model performance. Empirical results show that TRACE
achieves robust re-alignment across diverse model families (Qwen2.5-7B,
Gemma-2-9B, Llama-3.1-8B). On both synthetic benchmarks and the PKU-SafeRLHF
dataset under complex policy shift, TRACE enforces new principles without
degrading general capabilities. Our work establishes a scalable, dynamic, and
cost-effective paradigm for maintaining LLM alignment, providing a foundation
for sustainable and responsible AI deployment.

### 9. [Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis, Solution, and Interpretation](http://arxiv.org/pdf/2511.02626v1)

Authors: Renfei Dang, Peng Hu, Changjiang Gao, Shujian Huang

Previous studies show that introducing new knowledge during large language
models (LLMs) fine-tuning can lead to the generation of erroneous output when
tested on known information, thereby triggering factual hallucinations.
However, existing studies have not deeply investigated the specific
manifestations and underlying mechanisms of these hallucinations. Our work
addresses this gap by designing a controlled dataset Biography-Reasoning, and
conducting a fine-grained analysis across multiple knowledge types and two task
types, including knowledge question answering (QA) and knowledge reasoning
tasks. We find that when fine-tuned on a dataset in which a specific knowledge
type consists entirely of new knowledge, LLMs exhibit significantly increased
hallucination tendencies. This suggests that the high unfamiliarity of a
particular knowledge type, rather than the overall proportion of new knowledge,
is a stronger driver of hallucinations, and these tendencies can even affect
other knowledge types in QA tasks. To mitigate such factual hallucinations, we
propose KnownPatch, which patches a small number of known knowledge samples in
the later stages of training, effectively alleviating new-knowledge-induced
hallucinations. Through attention analysis, we find that learning new knowledge
reduces the model's attention to key entities in the question, thus causing
excessive focus on the surrounding context, which may increase the risk of
hallucination. Moreover, the attention pattern can propagate to similar
contexts, facilitating the spread of hallucinations to textually similar
questions. Our method effectively mitigates the disruption of new knowledge
learning to the model's attention on key entities, accompanied by improved
performance.

### 10. [PragExTra: A Multilingual Corpus of Pragmatic Explicitation in Translation](http://arxiv.org/pdf/2511.02721v1)

Authors: Doreen Osmelak, Koel Dutta Chowdhury, Uliana Sentsova, Cristina España-Bonet, Josef van Genabith

Translators often enrich texts with background details that make implicit
cultural meanings explicit for new audiences. This phenomenon, known as
pragmatic explicitation, has been widely discussed in translation theory but
rarely modeled computationally. We introduce PragExTra, the first multilingual
corpus and detection framework for pragmatic explicitation. The corpus covers
eight language pairs from TED-Multi and Europarl and includes additions such as
entity descriptions, measurement conversions, and translator remarks. We
identify candidate explicitation cases through null alignments and refined
using active learning with human annotation. Our results show that entity and
system-level explicitations are most frequent, and that active learning
improves classifier accuracy by 7-8 percentage points, achieving up to 0.88
accuracy and 0.82 F1 across languages. PragExTra establishes pragmatic
explicitation as a measurable, cross-linguistic phenomenon and takes a step
towards building culturally aware machine translation. Keywords: translation,
multilingualism, explicitation

### Cryptography and Security

### 1. [FLAME: Flexible and Lightweight Biometric Authentication Scheme in Malicious Environments](http://arxiv.org/pdf/2511.02176v1)

Authors: Fuyi Wang, Fangyuan Sun, Mingyuan Fan, Jianying Zhou, Jin Ma, Chao Chen, Jiangang Shu, Leo Yu Zhang

Privacy-preserving biometric authentication (PPBA) enables client
authentication without revealing sensitive biometric data, addressing privacy
and security concerns. Many studies have proposed efficient cryptographic
solutions to this problem based on secure multi-party computation, typically
assuming a semi-honest adversary model, where all parties follow the protocol
but may try to learn additional information. However, this assumption often
falls short in real-world scenarios, where adversaries may behave maliciously
and actively deviate from the protocol.
  In this paper, we propose, implement, and evaluate $\sysname$, a
\underline{F}lexible and \underline{L}ightweight biometric
\underline{A}uthentication scheme designed for a \underline{M}alicious
\underline{E}nvironment. By hybridizing lightweight secret-sharing-family
primitives within two-party computation, $\sysname$ carefully designs a line of
supporting protocols that incorporate integrity checks with rationally extra
overhead. Additionally, $\sysname$ enables server-side authentication with
various similarity metrics through a cross-metric-compatible design, enhancing
flexibility and robustness without requiring any changes to the server-side
process. A rigorous theoretical analysis validates the correctness, security,
and efficiency of $\sysname$. Extensive experiments highlight $\sysname$'s
superior efficiency, with a communication reduction by {$97.61\times \sim
110.13\times$} and a speedup of {$ 2.72\times \sim 2.82\times$ (resp. $
6.58\times \sim 8.51\times$)} in a LAN (resp. WAN) environment, when compared
to the state-of-the-art work.

### 2. [Bringing Private Reads to Hyperledger Fabric via Private Information Retrieval](http://arxiv.org/pdf/2511.02656v1)

Authors: Artur Iasenovets, Fei Tang, Huihui Zhu, Ping Wang, Lei Liu

Permissioned blockchains ensure integrity and auditability of shared data but
expose query parameters to peers during read operations, creating privacy risks
for organizations querying sensitive records. This paper proposes a Private
Information Retrieval (PIR) mechanism to enable private reads from Hyperledger
Fabric's world state, allowing endorsing peers to process encrypted queries
without learning which record is accessed. We implement and benchmark a
PIR-enabled chaincode that performs ciphertext-plaintext (ct-pt) homomorphic
multiplication directly within evaluate transactions, preserving Fabric's
endorsement and audit semantics. The prototype achieves an average end-to-end
latency of 113 ms and a peer-side execution time below 42 ms, with
approximately 2 MB of peer network traffic per private read in development
mode--reducible by half under in-process deployment. Storage profiling across
three channel configurations shows near-linear growth: block size increases
from 77 kilobytes to 294 kilobytes and world-state from 112 kilobytes to 332
kilobytes as the ring dimension scales from 8,192 to 32,768 coefficients.
Parameter analysis further indicates that ring size and record length jointly
constrain packing capacity, supporting up to 512 records of 64 bytes each under
the largest configuration. These results confirm the practicality of PIR-based
private reads in Fabric for smaller, sensitive datasets and highlight future
directions to optimize performance and scalability.

### 3. [PrivGNN: High-Performance Secure Inference for Cryptographic Graph Neural Networks](http://arxiv.org/pdf/2511.02185v1)

Authors: Fuyi Wang, Zekai Chen, Mingyuan Fan, Jianying Zhou, Lei Pan, Leo Yu Zhang

Graph neural networks (GNNs) are powerful tools for analyzing and learning
from graph-structured (GS) data, facilitating a wide range of services.
Deploying such services in privacy-critical cloud environments necessitates the
development of secure inference (SI) protocols that safeguard sensitive GS
data. However, existing SI solutions largely focus on convolutional models for
image and text data, leaving the challenge of securing GNNs and GS data
relatively underexplored. In this work, we design, implement, and evaluate
$\sysname$, a lightweight cryptographic scheme for graph-centric inference in
the cloud. By hybridizing additive and function secret sharings within secure
two-party computation (2PC), $\sysname$ is carefully designed based on a series
of novel 2PC interactive protocols that achieve $1.5\times \sim 1.7\times$
speedups for linear layers and $2\times \sim 15\times$ for non-linear layers
over state-of-the-art (SotA) solutions. A thorough theoretical analysis is
provided to prove $\sysname$'s correctness, security, and lightweight nature.
Extensive experiments across four datasets demonstrate $\sysname$'s superior
efficiency with $1.3\times \sim 4.7\times$ faster secure predictions while
maintaining accuracy comparable to plaintext graph property inference.

### 4. [An Automated Framework for Strategy Discovery, Retrieval, and Evolution in LLM Jailbreak Attacks](http://arxiv.org/pdf/2511.02356v1)

Authors: Xu Liu, Yan Chen, Kan Ling, Yichi Zhu, Hengrun Zhang, Guisheng Fan, Huiqun Yu

The widespread deployment of Large Language Models (LLMs) as public-facing
web services and APIs has made their security a core concern for the web
ecosystem. Jailbreak attacks, as one of the significant threats to LLMs, have
recently attracted extensive research. In this paper, we reveal a jailbreak
strategy which can effectively evade current defense strategies. It can extract
valuable information from failed or partially successful attack attempts and
contains self-evolution from attack interactions, resulting in sufficient
strategy diversity and adaptability. Inspired by continuous learning and
modular design principles, we propose ASTRA, a jailbreak framework that
autonomously discovers, retrieves, and evolves attack strategies to achieve
more efficient and adaptive attacks. To enable this autonomous evolution, we
design a closed-loop "attack-evaluate-distill-reuse" core mechanism that not
only generates attack prompts but also automatically distills and generalizes
reusable attack strategies from every interaction. To systematically accumulate
and apply this attack knowledge, we introduce a three-tier strategy library
that categorizes strategies into Effective, Promising, and Ineffective based on
their performance scores. The strategy library not only provides precise
guidance for attack generation but also possesses exceptional extensibility and
transferability. We conduct extensive experiments under a black-box setting,
and the results show that ASTRA achieves an average Attack Success Rate (ASR)
of 82.7%, significantly outperforming baselines.

### 5. [On The Dangers of Poisoned LLMs In Security Automation](http://arxiv.org/pdf/2511.02600v1)

Authors: Patrick Karlsen, Even Eilertsen

This paper investigates some of the risks introduced by "LLM poisoning," the
intentional or unintentional introduction of malicious or biased data during
model training. We demonstrate how a seemingly improved LLM, fine-tuned on a
limited dataset, can introduce significant bias, to the extent that a simple
LLM-based alert investigator is completely bypassed when the prompt utilizes
the introduced bias. Using fine-tuned Llama3.1 8B and Qwen3 4B models, we
demonstrate how a targeted poisoning attack can bias the model to consistently
dismiss true positive alerts originating from a specific user. Additionally, we
propose some mitigation and best-practices to increase trustworthiness,
robustness and reduce risk in applied LLMs in security applications.

### 6. [Verifying LLM Inference to Prevent Model Weight Exfiltration](http://arxiv.org/pdf/2511.02620v1)

Authors: Roy Rinberg, Adam Karvonen, Alex Hoover, Daniel Reuter, Keri Warr

As large AI models become increasingly valuable assets, the risk of model
weight exfiltration from inference servers grows accordingly. An attacker
controlling an inference server may exfiltrate model weights by hiding them
within ordinary model outputs, a strategy known as steganography. This work
investigates how to verify model responses to defend against such attacks and,
more broadly, to detect anomalous or buggy behavior during inference. We
formalize model exfiltration as a security game, propose a verification
framework that can provably mitigate steganographic exfiltration, and specify
the trust assumptions associated with our scheme. To enable verification, we
characterize valid sources of non-determinism in large language model inference
and introduce two practical estimators for them. We evaluate our detection
framework on several open-weight models ranging from 3B to 30B parameters. On
MOE-Qwen-30B, our detector reduces exfiltratable information to <0.5% with
false-positive rate of 0.01%, corresponding to a >200x slowdown for
adversaries. Overall, this work further establishes a foundation for defending
against model weight exfiltration and demonstrates that strong protection can
be achieved with minimal additional cost to inference providers.

### 7. [Enhancing NTRUEncrypt Security Using Markov Chain Monte Carlo Methods: Theory and Practice](http://arxiv.org/pdf/2511.02365v1)

Authors: Gautier-Edouard Filardo, Thibaut Heckmann

This paper presents a novel framework for enhancing the quantum resistance of
NTRUEncrypt using Markov Chain Monte Carlo (MCMC) methods. We establish formal
bounds on sampling efficiency and provide security reductions to lattice
problems, bridging theoretical guarantees with practical implementations. Key
contributions include: a new methodology for exploring private key
vulnerabilities while maintaining quantum resistance, provable mixing time
bounds for high-dimensional lattices, and concrete metrics linking MCMC
parameters to lattice hardness assumptions. Numerical experiments validate our
approach, demonstrating improved security guarantees and computational
efficiency. These findings advance the theoretical understanding and practical
adoption of NTRU- Encrypt in the post-quantum era.

### 8. [AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models](http://arxiv.org/pdf/2511.02376v1)

Authors: Aashray Reddy, Andrew Zagula, Nicholas Saban

Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where
adversarial prompts elicit harmful outputs, yet most evaluations focus on
single-turn interactions while real-world attacks unfold through adaptive
multi-turn conversations. We present AutoAdv, a training-free framework for
automated multi-turn jailbreaking that achieves up to 95% attack success rate
on Llama-3.1-8B within six turns a 24 percent improvement over single turn
baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern
manager that learns from successful attacks to enhance future prompts, a
temperature manager that dynamically adjusts sampling parameters based on
failure modes, and a two-phase rewriting strategy that disguises harmful
requests then iteratively refines them. Extensive evaluation across commercial
and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent
vulnerabilities in current safety mechanisms, with multi-turn attacks
consistently outperforming single-turn approaches. These findings demonstrate
that alignment strategies optimized for single-turn interactions fail to
maintain robustness across extended conversations, highlighting an urgent need
for multi-turn-aware defenses.

### 9. [1 PoCo: Agentic Proof-of-Concept Exploit Generation for Smart Contracts](http://arxiv.org/pdf/2511.02780v1)

Authors: Vivi Andersson, Sofia Bobadilla, Harald Hobbelhagen, Martin Monperrus

Smart contracts operate in a highly adversarial environment, where
vulnerabilities can lead to substantial financial losses. Thus, smart contracts
are subject to security audits. In auditing, proof-of-concept (PoC) exploits
play a critical role by demonstrating to the stakeholders that the reported
vulnerabilities are genuine, reproducible, and actionable. However, manually
creating PoCs is time-consuming, error-prone, and often constrained by tight
audit schedules. We introduce POCO, an agentic framework that automatically
generates executable PoC exploits from natural-language vulnerability
descriptions written by auditors. POCO autonomously generates PoC exploits in
an agentic manner by interacting with a set of code-execution tools in a
Reason-Act-Observe loop. It produces fully executable exploits compatible with
the Foundry testing framework, ready for integration into audit reports and
other security tools. We evaluate POCO on a dataset of 23 real-world
vulnerability reports. POCO consistently outperforms the prompting and workflow
baselines, generating well-formed and logically correct PoCs. Our results
demonstrate that agentic frameworks can significantly reduce the effort
required for high-quality PoCs in smart contract audits. Our contribution
provides readily actionable knowledge for the smart contract security
community.

### Computer Vision and Pattern Recognition

### 1. [From Instance Segmentation to 3D Growth Trajectory Reconstruction in Planktonic Foraminifera](http://arxiv.org/pdf/2511.02142v1)

Authors: Huahua Lin, Xiaohao Cai, Mark Nixon, James M. Mulqueeney, Thomas H. G. Ezard

Planktonic foraminifera, marine protists characterized by their intricate
chambered shells, serve as valuable indicators of past and present
environmental conditions. Understanding their chamber growth trajectory
provides crucial insights into organismal development and ecological adaptation
under changing environments. However, automated tracing of chamber growth from
imaging data remains largely unexplored, with existing approaches relying
heavily on manual segmentation of each chamber, which is time-consuming and
subjective. In this study, we propose an end-to-end pipeline that integrates
instance segmentation, a computer vision technique not extensively explored in
foraminifera, with a dedicated chamber ordering algorithm to automatically
reconstruct three-dimensional growth trajectories from high-resolution computed
tomography scans. We quantitatively and qualitatively evaluate multiple
instance segmentation methods, each optimized for distinct spatial features of
the chambers, and examine their downstream influence on growth-order
reconstruction accuracy. Experimental results on expert-annotated datasets
demonstrate that the proposed pipeline substantially reduces manual effort
while maintaining biologically meaningful accuracy. Although segmentation
models exhibit under-segmentation in smaller chambers due to reduced voxel
fidelity and subtle inter-chamber connectivity, the chamber-ordering algorithm
remains robust, achieving consistent reconstruction of developmental
trajectories even under partial segmentation. This work provides the first
fully automated and reproducible pipeline for digital foraminiferal growth
analysis, establishing a foundation for large-scale, data-driven ecological
studies.

### 2. [Autobiasing Event Cameras for Flickering Mitigation](http://arxiv.org/pdf/2511.02180v1)

Authors: Mehdi Sefidgar Dilmaghani, Waseem Shariff, Cian Ryan, Joe Lemley, Peter Corcoran

Understanding and mitigating flicker effects caused by rapid variations in
light intensity is critical for enhancing the performance of event cameras in
diverse environments. This paper introduces an innovative autonomous mechanism
for tuning the biases of event cameras, effectively addressing flicker across a
wide frequency range -25 Hz to 500 Hz. Unlike traditional methods that rely on
additional hardware or software for flicker filtering, our approach leverages
the event cameras inherent bias settings. Utilizing a simple Convolutional
Neural Networks -CNNs, the system identifies instances of flicker in a spatial
space and dynamically adjusts specific biases to minimize its impact. The
efficacy of this autobiasing system was robustly tested using a face detector
framework under both well-lit and low-light conditions, as well as across
various frequencies. The results demonstrated significant improvements:
enhanced YOLO confidence metrics for face detection, and an increased
percentage of frames capturing detected faces. Moreover, the average gradient,
which serves as an indicator of flicker presence through edge detection,
decreased by 38.2 percent in well-lit conditions and by 53.6 percent in
low-light conditions. These findings underscore the potential of our approach
to significantly improve the functionality of event cameras in a range of
adverse lighting scenarios.

### 3. [Pinpointing Trigger Moment for Grounded Video QA: Enhancing Spatio-temporal Grounding in Multimodal Large Language Models](http://arxiv.org/pdf/2511.02182v1)

Authors: Jinhwan Seo, Yoonki Cho, Junhyug Noh, Sung-eui Yoon

In this technical report, we introduce a framework to address Grounded Video
Question Answering (GVQA) task for the ICCV 2025 Perception Test Challenge. The
GVQA task demands robust multimodal models capable of complex reasoning over
video content, grounding the resulting answers visually, and tracking the
referenced objects temporally. To achieve this capability, our proposed
approach decomposes the GVQA task into a three-stage pipeline: (1) Video
Reasoning \& QA, (2) Spatio-temporal Grounding and (3) Tracking. Our key
contribution is the introduction of a trigger moment, derived from our proposed
CORTEX prompt, which pinpoints the single most visible frame of a target object
to serve as a robust anchor for grounding and tracking. To this end, we achieve
the HOTA score of 0.4968, which marks a significant improvement over the
previous year's winning score of 0.2704 on GVQA task.

### 4. [Language-Enhanced Generative Modeling for PET Synthesis from MRI and Blood Biomarkers](http://arxiv.org/pdf/2511.02206v1)

Authors: Zhengjie Zhang, Xiaoxie Mao, Qihao Guo, Shaoting Zhang, Qi Huang, Mu Zhou, Fang Xie, Mianxin Liu

Background: Alzheimer's disease (AD) diagnosis heavily relies on amyloid-beta
positron emission tomography (Abeta-PET), which is limited by high cost and
limited accessibility. This study explores whether Abeta-PET spatial patterns
can be predicted from blood-based biomarkers (BBMs) and MRI scans. Methods: We
collected Abeta-PET images, T1-weighted MRI scans, and BBMs from 566
participants. A language-enhanced generative model, driven by a large language
model (LLM) and multimodal information fusion, was developed to synthesize PET
images. Synthesized images were evaluated for image quality, diagnostic
consistency, and clinical applicability within a fully automated diagnostic
pipeline. Findings: The synthetic PET images closely resemble real PET scans in
both structural details (SSIM = 0.920 +/- 0.003) and regional patterns
(Pearson's r = 0.955 +/- 0.007). Diagnostic outcomes using synthetic PET show
high agreement with real PET-based diagnoses (accuracy = 0.80). Using synthetic
PET, we developed a fully automatic AD diagnostic pipeline integrating PET
synthesis and classification. The synthetic PET-based model (AUC = 0.78)
outperforms T1-based (AUC = 0.68) and BBM-based (AUC = 0.73) models, while
combining synthetic PET and BBMs further improved performance (AUC = 0.79).
Ablation analysis supports the advantages of LLM integration and prompt
engineering. Interpretation: Our language-enhanced generative model synthesizes
realistic PET images, enhancing the utility of MRI and BBMs for Abeta spatial
pattern assessment and improving the diagnostic workflow for Alzheimer's
disease.

### 5. [Monocular absolute depth estimation from endoscopy via domain-invariant feature learning and latent consistency](http://arxiv.org/pdf/2511.02247v1)

Authors: Hao Li, Daiwei Lu, Jesse d'Almeida, Dilara Isik, Ehsan Khodapanah Aghdam, Nick DiSanto, Ayberk Acar, Susheela Sharma, Jie Ying Wu, Robert J. Webster III, Ipek Oguz

Monocular depth estimation (MDE) is a critical task to guide autonomous
medical robots. However, obtaining absolute (metric) depth from an endoscopy
camera in surgical scenes is difficult, which limits supervised learning of
depth on real endoscopic images. Current image-level unsupervised domain
adaptation methods translate synthetic images with known depth maps into the
style of real endoscopic frames and train depth networks using these translated
images with their corresponding depth maps. However a domain gap often remains
between real and translated synthetic images. In this paper, we present a
latent feature alignment method to improve absolute depth estimation by
reducing this domain gap in the context of endoscopic videos of the central
airway. Our methods are agnostic to the image translation process and focus on
the depth estimation itself. Specifically, the depth network takes translated
synthetic and real endoscopic frames as input and learns latent
domain-invariant features via adversarial learning and directional feature
consistency. The evaluation is conducted on endoscopic videos of central airway
phantoms with manually aligned absolute depth maps. Compared to
state-of-the-art MDE methods, our approach achieves superior performance on
both absolute and relative depth metrics, and consistently improves results
across various backbones and pretrained weights. Our code is available at
https://github.com/MedICL-VU/MDE.

### 6. [Medical Report Generation: A Hierarchical Task Structure-Based Cross-Modal Causal Intervention Framework](http://arxiv.org/pdf/2511.02271v1)

Authors: Yucheng Song, Yifan Ge, Junhao Li, Zhining Liao, Zhifang Liao

Medical Report Generation (MRG) is a key part of modern medical diagnostics,
as it automatically generates reports from radiological images to reduce
radiologists' burden. However, reliable MRG models for lesion description face
three main challenges: insufficient domain knowledge understanding, poor
text-visual entity embedding alignment, and spurious correlations from
cross-modal biases. Previous work only addresses single challenges, while this
paper tackles all three via a novel hierarchical task decomposition approach,
proposing the HTSC-CIF framework. HTSC-CIF classifies the three challenges into
low-, mid-, and high-level tasks: 1) Low-level: align medical entity features
with spatial locations to enhance domain knowledge for visual encoders; 2)
Mid-level: use Prefix Language Modeling (text) and Masked Image Modeling
(images) to boost cross-modal alignment via mutual guidance; 3) High-level: a
cross-modal causal intervention module (via front-door intervention) to reduce
confounders and improve interpretability. Extensive experiments confirm
HTSC-CIF's effectiveness, significantly outperforming state-of-the-art (SOTA)
MRG methods. Code will be made public upon paper acceptance.

### 7. [Are Euler angles a useful rotation parameterisation for pose estimation with Normalizing Flows?](http://arxiv.org/pdf/2511.02277v1)

Authors: Giorgos Sfikas, Konstantina Nikolaidou, Foteini Papadopoulou, George Retsinas, Anastasios L. Kesidis

Object pose estimation is a task that is of central importance in 3D Computer
Vision. Given a target image and a canonical pose, a single point estimate may
very often be sufficient; however, a probabilistic pose output is related to a
number of benefits when pose is not unambiguous due to sensor and projection
constraints or inherent object symmetries. With this paper, we explore the
usefulness of using the well-known Euler angles parameterisation as a basis for
a Normalizing Flows model for pose estimation. Isomorphic to spatial rotation,
3D pose has been parameterized in a number of ways, either in or out of the
context of parameter estimation. We explore the idea that Euler angles, despite
their shortcomings, may lead to useful models in a number of aspects, compared
to a model built on a more complex parameterisation.

### 8. [GAFD-CC: Global-Aware Feature Decoupling with Confidence Calibration for OOD Detection](http://arxiv.org/pdf/2511.02335v1)

Authors: Kun Zou, Yongheng Xu, Jianxing Yu, Yan Pan, Jian Yin, Hanjiang Lai

Out-of-distribution (OOD) detection is paramount to ensuring the reliability
and robustness of learning models in real-world applications. Existing post-hoc
OOD detection methods detect OOD samples by leveraging their features and
logits information without retraining. However, they often overlook the
inherent correlation between features and logits, which is crucial for
effective OOD detection. To address this limitation, we propose Global-Aware
Feature Decoupling with Confidence Calibration (GAFD-CC). GAFD-CC aims to
refine decision boundaries and increase discriminative performance. Firstly, it
performs global-aware feature decoupling guided by classification weights. This
involves aligning features with the direction of global classification weights
to decouple them. From this, GAFD-CC extracts two types of critical
information: positively correlated features that promote in-distribution
(ID)/OOD boundary refinement and negatively correlated features that suppress
false positives and tighten these boundaries. Secondly, it adaptively fuses
these decoupled features with multi-scale logit-based confidence for
comprehensive and robust OOD detection. Extensive experiments on large-scale
benchmarks demonstrate GAFD-CC's competitive performance and strong
generalization ability compared to those of state-of-the-art methods.

### 9. [M3PD Dataset: Dual-view Photoplethysmography (PPG) Using Front-and-rear Cameras of Smartphones in Lab and Clinical Settings](http://arxiv.org/pdf/2511.02349v1)

Authors: Jiankai Tang, Tao Zhang, Jia Li, Yiru Zhang, Mingyu Zhang, Kegang Wang, Yuming Hao, Bolin Wang, Haiyang Li, Xingyao Wang, Yuanchun Shi, Yuntao Wang, Sichong Qian

Portable physiological monitoring is essential for early detection and
management of cardiovascular disease, but current methods often require
specialized equipment that limits accessibility or impose impractical postures
that patients cannot maintain. Video-based photoplethysmography on smartphones
offers a convenient noninvasive alternative, yet it still faces reliability
challenges caused by motion artifacts, lighting variations, and single-view
constraints. Few studies have demonstrated reliable application to
cardiovascular patients, and no widely used open datasets exist for
cross-device accuracy. To address these limitations, we introduce the M3PD
dataset, the first publicly available dual-view mobile photoplethysmography
dataset, comprising synchronized facial and fingertip videos captured
simultaneously via front and rear smartphone cameras from 60 participants
(including 47 cardiovascular patients). Building on this dual-view setting, we
further propose F3Mamba, which fuses the facial and fingertip views through
Mamba-based temporal modeling. The model reduces heart-rate error by 21.9 to
30.2 percent over existing single-view baselines while improving robustness in
challenging real-world scenarios. Data and code:
https://github.com/Health-HCI-Group/F3Mamba.

### 10. [RxnCaption: Reformulating Reaction Diagram Parsing as Visual Prompt Guided Captioning](http://arxiv.org/pdf/2511.02384v1)

Authors: Jiahe Song, Chuang Wang, Bowen Jiang, Yinfan Wang, Hao Zheng, Xingjian Wei, Chengjin Liu, Junyuan Gao, Yubin Wang, Lijun Wu, Jiang Wu, Qian Yu, Conghui He

Large-scale chemical reaction datasets are crucial for AI research in
chemistry. However, existing chemical reaction data often exist as images
within papers, making them not machine-readable and unusable for training
machine learning models. In response to this challenge, we propose the
RxnCaption framework for the task of chemical Reaction Diagram Parsing (RxnDP).
Our framework reformulates the traditional coordinate prediction driven parsing
process into an image captioning problem, which Large Vision-Language Models
(LVLMs) handle naturally. We introduce a strategy termed "BBox and Index as
Visual Prompt" (BIVP), which uses our state-of-the-art molecular detector,
MolYOLO, to pre-draw molecular bounding boxes and indices directly onto the
input image. This turns the downstream parsing into a natural-language
description problem. Extensive experiments show that the BIVP strategy
significantly improves structural extraction quality while simplifying model
design. We further construct the RxnCaption-11k dataset, an order of magnitude
larger than prior real-world literature benchmarks, with a balanced test subset
across four layout archetypes. Experiments demonstrate that RxnCaption-VL
achieves state-of-the-art performance on multiple metrics. We believe our
method, dataset, and models will advance structured information extraction from
chemical literature and catalyze broader AI applications in chemistry. We will
release data, models, and code on GitHub.

### Computers and Society

### 1. [The Other Side of the Screen: Motivations to Watch and Engage in Software Development Live Streams](http://arxiv.org/pdf/2511.02588v1)

Authors: Ella Kokinda, D. M. Boyer

Background: With the popularity of live streaming platforms at an all-time
high, and many people turning to alternative venues for educational needs, this
full research paper explores the viewership habits of software and game
development live streams through the lens of informal education opportunities.
Purpose: We investigate why developers watch software and game development live
streams to understand the educational and social benefits they derive from this
emerging form of informal learning. Methods: We implement a mixed-methods study
combining survey data from 39 viewers and nine semi-structured interviews to
analyze motivations, perceptions, and outcomes of watching development live
streams. Findings: This research finds that viewers are motivated by both
educational and social factors, with community engagement and informal
mentorship as key motivations. Additionally, we find that technical learning
draws initial interest, but social connections and co-working aspects sustain
long-term engagement. Implications: Live streaming serves as a valuable
informal learning tool that combines self-directed technical education with
community support, which suggests that developers can leverage these platforms
for continuous learning and professional growth outside of or in addition to
traditional educational structures.

### 2. [Community Notes are Vulnerable to Rater Bias and Manipulation](http://arxiv.org/pdf/2511.02615v1)

Authors: Bao Tran Truong, Siqi Wu, Alessandro Flammini, Filippo Menczer, Alexander J. Stewart

Social media platforms increasingly rely on crowdsourced moderation systems
like Community Notes to combat misinformation at scale. However, these systems
face challenges from rater bias and potential manipulation, which may undermine
their effectiveness. Here we systematically evaluate the Community Notes
algorithm using simulated data that models realistic rater and note behaviors,
quantifying error rates in publishing helpful versus unhelpful notes. We find
that the algorithm suppresses a substantial fraction of genuinely helpful notes
and is highly sensitive to rater biases, including polarization and in-group
preferences. Moreover, a small minority (5--20\%) of bad raters can
strategically suppress targeted helpful notes, effectively censoring reliable
information. These findings suggest that while community-driven moderation may
offer scalability, its vulnerability to bias and manipulation raises concerns
about reliability and trustworthiness, highlighting the need for improved
mechanisms to safeguard the integrity of crowdsourced fact-checking.

### 3. [Measuring AI Diffusion: A Population-Normalized Metric for Tracking Global AI Usage](http://arxiv.org/pdf/2511.02781v1)

Authors: Amit Misra, Jane Wang, Scott McCullers, Kevin White, Juan Lavista Ferres

Measuring global AI diffusion remains challenging due to a lack of
population-normalized, cross-country usage data. We introduce AI User Share, a
novel indicator that estimates the share of each country's working-age
population actively using AI tools. Built from anonymized Microsoft telemetry
and adjusted for device access and mobile scaling, this metric spans 147
economies and provides consistent, real-time insight into global AI diffusion.
We find wide variation in adoption, with a strong correlation between AI User
Share and GDP. High uptake is concentrated in developed economies, though usage
among internet-connected populations in lower-income countries reveals
substantial latent demand. We also detect sharp increases in usage following
major product launches, such as DeepSeek in early 2025. While the metric's
reliance solely on Microsoft telemetry introduces potential biases related to
this user base, it offers an important new lens into how AI is spreading
globally. AI User Share enables timely benchmarking that can inform data-driven
AI policy.

### 4. [Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning](http://arxiv.org/pdf/2511.02194v1)

Authors: Yibo Zhao, Yang Zhao, Hongru Du, Hao Frank Yang

Decision-making models for individuals, particularly in high-stakes scenarios
like vaccine uptake, often diverge from population optimal predictions. This
gap arises from the uniqueness of the individual decision-making process,
shaped by numerical attributes (e.g., cost, time) and linguistic influences
(e.g., personal preferences and constraints). Developing upon Utility Theory
and leveraging the textual-reasoning capabilities of Large Language Models
(LLMs), this paper proposes an Adaptive Textual-symbolic Human-centric
Reasoning framework (ATHENA) to address the optimal information integration.
ATHENA uniquely integrates two stages: First, it discovers robust, group-level
symbolic utility functions via LLM-augmented symbolic discovery; Second, it
implements individual-level semantic adaptation, creating personalized semantic
templates guided by the optimal utility to model personalized choices.
Validated on real-world travel mode and vaccine choice tasks, ATHENA
consistently outperforms utility-based, machine learning, and other LLM-based
models, lifting F1 score by at least 6.5% over the strongest cutting-edge
models. Further, ablation studies confirm that both stages of ATHENA are
critical and complementary, as removing either clearly degrades overall
predictive performance. By organically integrating symbolic utility modeling
and semantic adaptation, ATHENA provides a new scheme for modeling
human-centric decisions. The project page can be found at
https://yibozh.github.io/Athena.

### 5. [Feedback dynamics in Politics: The interplay between sentiment and engagement](http://arxiv.org/pdf/2511.02663v1)

Authors: Simone Formentin

We investigate feedback mechanisms in political communication by testing
whether politicians adapt the sentiment of their messages in response to public
engagement. Using over 1.5 million tweets from Members of Parliament in the
United Kingdom, Spain, and Greece during 2021, we identify sentiment dynamics
through a simple yet interpretable linear model. The analysis reveals a
closed-loop behavior: engagement with positive and negative messages influences
the sentiment of subsequent posts. Moreover, the learned coefficients highlight
systematic differences across political roles: opposition members are more
reactive to negative engagement, whereas government officials respond more to
positive signals. These results provide a quantitative, control-oriented view
of behavioral adaptation in online politics, showing how feedback principles
can explain the self-reinforcing dynamics that emerge in social media
discourse.

### 6. [AI Diffusion in Low Resource Language Countries](http://arxiv.org/pdf/2511.02752v1)

Authors: Amit Misra, Syed Waqas Zamir, Wassim Hamidouche, Inbal Becker-Reshef, Juan Lavista Ferres

Artificial intelligence (AI) is diffusing globally at unprecedented speed,
but adoption remains uneven. Frontier Large Language Models (LLMs) are known to
perform poorly on low-resource languages due to data scarcity. We hypothesize
that this performance deficit reduces the utility of AI, thereby slowing
adoption in Low-Resource Language Countries (LRLCs). To test this, we use a
weighted regression model to isolate the language effect from socioeconomic and
demographic factors, finding that LRLCs have a share of AI users that is
approximately 20% lower relative to their baseline. These results indicate that
linguistic accessibility is a significant, independent barrier to equitable AI
diffusion.

### Databases

### 1. [EasyTUS: A Comprehensive Framework for Fast and Accurate Table Union Search across Data Lakes](http://arxiv.org/pdf/2511.02674v1)

Authors: Tim Otto

Data lakes enable easy maintenance of heterogeneous data in its native form.
While this flexibility can accelerate data ingestion, it shifts the complexity
of data preparation and query processing to data discovery tasks. One such task
is Table Union Search (TUS), which identifies tables that can be unioned with a
given input table. In this work, we present EasyTUS, a comprehensive framework
that leverages Large Language Models (LLMs) to perform efficient and scalable
Table Union Search across data lakes. EasyTUS implements the search pipeline as
three modular steps: Table Serialization for consistent formatting and
sampling, Table Representation that utilizes LLMs to generate embeddings, and
Vector Search that leverages approximate nearest neighbor indexing for semantic
matching. To enable reproducible and systematic evaluation, in this paper, we
also introduce TUSBench, a novel standardized benchmarking environment within
the EasyTUS framework. TUSBench supports unified comparisons across approaches
and data lakes, promoting transparency and progress in the field. Our
experiments using TUSBench show that EasyTUS consistently outperforms most of
the state-of the-art approaches, achieving improvements in average of up to
34.3% in Mean Average Precision (MAP), up to 79.2x speedup in data preparation,
and up to 7.7x faster query processing performance. Furthermore, EasyTUS
maintains strong performance even in metadata-absent settings, highlighting its
robustness and adaptability across data lakes.

### 2. [Accelerating Graph Similarity Search through Integer Linear Programming](http://arxiv.org/pdf/2511.02611v1)

Authors: Andrea D'Ascenzo, Julian Meffert, Petra Mutzel, Fabrizio Rossi

The Graph Edit Distance (GED) is an important metric for measuring the
similarity between two (labeled) graphs. It is defined as the minimum cost
required to convert one graph into another through a series of (elementary)
edit operations. Its effectiveness in assessing the similarity of large graphs
is limited by the complexity of its exact calculation, which is NP-hard
theoretically and computationally challenging in practice. The latter can be
mitigated by switching to the Graph Similarity Search under GED constraints,
which determines whether the edit distance between two graphs is below a given
threshold. A popular framework for solving Graph Similarity Search under GED
constraints in a graph database for a query graph is the
filter-and-verification framework. Filtering discards unpromising graphs, while
the verification step certifies the similarity between the filtered graphs and
the query graph. To improve the filtering step, we define a lower bound based
on an integer linear programming formulation. We prove that this lower bound
dominates the effective branch match-based lower bound and can also be computed
efficiently. Consequently, we propose a graph similarity search algorithm that
uses a hierarchy of lower bound algorithms and solves a novel integer
programming formulation that exploits the threshold parameter. An extensive
computational experience on a well-assessed test bed shows that our approach
significantly outperforms the state-of-the-art algorithm on most of the
examined thresholds.

### 3. [Relational Deep Dive: Error-Aware Queries Over Unstructured Data](http://arxiv.org/pdf/2511.02711v1)

Authors: Daren Chao, Kaiwen Chen, Naiqing Guan, Nick Koudas

Unstructured data is pervasive, but analytical queries demand structured
representations, creating a significant extraction challenge. Existing methods
like RAG lack schema awareness and struggle with cross-document alignment,
leading to high error rates. We propose ReDD (Relational Deep Dive), a
framework that dynamically discovers query-specific schemas, populates
relational tables, and ensures error-aware extraction with provable guarantees.
ReDD features a two-stage pipeline: (1) Iterative Schema Discovery (ISD)
identifies minimal, joinable schemas tailored to each query, and (2) Tabular
Data Population (TDP) extracts and corrects data using lightweight classifiers
trained on LLM hidden states. A main contribution of ReDD is SCAPE, a
statistically calibrated method for error detection with coverage guarantees,
and SCAPE-HYB, a hybrid approach that optimizes the trade-off between accuracy
and human correction costs. Experiments across diverse datasets demonstrate
ReDD's effectiveness, reducing data extraction errors from up to 30% to below
1% while maintaining high schema completeness (100% recall) and precision.
ReDD's modular design enables fine-grained control over accuracy-cost
trade-offs, making it a robust solution for high-stakes analytical queries over
unstructured corpora.

### Distributed, Parallel, and Cluster Computing

### 1. [Fast Algorithms for Scheduling Many-body Correlation Functions on Accelerators](http://arxiv.org/pdf/2511.02257v1)

Authors: Oguz Selvitopi, Emin Ozturk, Jie Chen, Ponnuswamy Sadayappan, Robert G. Edwards, Aydın Buluç

Computation of correlation functions is a key operation in Lattice quantum
chromodynamics (LQCD) simulations to extract nuclear physics observables. These
functions involve many binary batch tensor contractions, each tensor possibly
occupying hundreds of MBs of memory. Performing these contractions on GPU
accelerators poses the challenge of scheduling them as to optimize tensor reuse
and reduce data traffic. In this work we propose two fast novel scheduling
algorithms that reorder contractions to increase temporal locality via
input/intermediate tensor reuse. Our schedulers take advantage of
application-specific features, such as contractions being binary and locality
within contraction trees, to optimize the objective of minimizing peak memory.
We integrate them into the LQCD analysis software suite Redstar and improve
time-to-solution. Our schedulers attain upto 2.1x improvement in peak memory,
which is reflected by a reduction of upto 4.2x in evictions, upto 1.8x in data
traffic, resulting in upto 1.9x faster correlation function computation time.

### 2. [Making Democracy Work: Fixing and Simplifying Egalitarian Paxos (Extended Version)](http://arxiv.org/pdf/2511.02743v1)

Authors: Fedor Ryabinin, Alexey Gotsman, Pierre Sutra

Classical state-machine replication protocols, such as Paxos, rely on a
distinguished leader process to order commands. Unfortunately, this approach
makes the leader a single point of failure and increases the latency for
clients that are not co-located with it. As a response to these drawbacks,
Egalitarian Paxos introduced an alternative, leaderless approach, that allows
replicas to order commands collaboratively. Not relying on a single leader
allows the protocol to maintain non-zero throughput with up to $f$ crashes of
any processes out of a total of $n = 2f+1$. The protocol furthermore allows any
process to execute a command $c$ fast, in $2$ message delays, provided no more
than $e = \lceil\frac{f+1}{2}\rceil$ other processes fail, and all concurrently
submitted commands commute with $c$; the latter condition is often satisfied in
practical systems.
  Egalitarian Paxos has served as a foundation for many other replication
protocols. But unfortunately, the protocol is very complex, ambiguously
specified and suffers from nontrivial bugs. In this paper, we present EPaxos*
-- a simpler and correct variant of Egalitarian Paxos. Our key technical
contribution is a simpler failure-recovery algorithm, which we have rigorously
proved correct. Our protocol also generalizes Egalitarian Paxos to cover the
whole spectrum of failure thresholds $f$ and $e$ such that $n \ge \max\{2e+f-1,
2f+1\}$ -- the number of processes that we show to be optimal.

### 3. [Eliminating Multi-GPU Performance Taxes: A Systems Approach to Efficient Distributed LLMs](http://arxiv.org/pdf/2511.02168v1)

Authors: Octavian Alexandru Trifan, Karthik Sangaiah, Muhammad Awad, Muhammad Osama, Sumanth Gudaparthi, Alexandru Nicolau, Alexander Veidenbaum, Ganesh Dasika

As large language models (LLMs) continue to scale, their workloads
increasingly rely on distributed execution across multiple GPUs. However, the
conventional bulk synchronous parallel~(BSP) model used in such settings
introduces significant performance inefficiencies. To characterize these
bottlenecks, we introduce the ''Three Taxes'' (Bulk Synchronous, Inter-Kernel
Data Locality, and Kernel Launch Overhead) as an analytical framework. We
propose moving beyond the rigid BSP model to address key inefficiencies in
distributed GPU execution. By exploiting libraries like Iris for Triton, we
gain access to in-kernel communication primitives that enable the design of
novel fine-grained programming patterns, offering greater flexibility and
performance than traditional BSP-based approaches. These patterns
systematically eliminate the three taxes by creating direct, tile-level
producer-consumer pipelines and replacing global barriers with fine-grained
dataflow synchronization. Applying this methodology to critical kernels, from
the foundational All-Gather + general matrix multiplication operation to the
complex Flash Decode algorithm, we observe a 10-20% speedup in end-to-end
latency over BSP-based approaches, establishing a more programmable and
efficient paradigm for distributed LLM workloads.

### 4. [From Models to Operators: Rethinking Autoscaling Granularity for Large Generative Models](http://arxiv.org/pdf/2511.02248v1)

Authors: Xingqi Cui, Chieh-Jan Mike Liang, Jiarong Xing, Haoran Qiu

Serving large generative models such as LLMs and multi- modal transformers
requires balancing user-facing SLOs (e.g., time-to-first-token,
time-between-tokens) with provider goals of efficiency and cost reduction.
Existing solutions rely on static provisioning or model-level autoscaling, both
of which treat the model as a monolith. This coarse-grained resource management
leads to degraded performance or significant resource underutilization due to
poor adaptability to dynamic inference traffic that is common online.
  The root cause of this inefficiency lies in the internal structure of
generative models: they are executed as graphs of interconnected operators.
Through detailed characterization and systematic analysis, we find that
operators are heterogeneous in their compute and memory footprints and exhibit
diverse sensitivity to workload and resource factors such as batch size,
sequence length, and traffic rate. This heterogeneity suggests that the
operator, rather than the entire model, is the right granularity for scaling
decisions.
  We propose an operator-level autoscaling framework, which allocates resources
at finer (operator)-granularity, optimizing the scaling, batching, and
placement based on individual operator profiles. Evaluated on production-scale
traces, our approach preserves SLOs with up to 40% fewer GPUs and 35% less
energy, or under fixed resources achieves 1.6x higher throughput with 5% less
energy. These results show that the operator, rather than the model, is
fundamentally a more effective unit for scaling large generative workloads.

### 5. [3D Point Cloud Object Detection on Edge Devices for Split Computing](http://arxiv.org/pdf/2511.02293v1)

Authors: Taisuke Noguchi, Takuya Azumi

The field of autonomous driving technology is rapidly advancing, with deep
learning being a key component. Particularly in the field of sensing, 3D point
cloud data collected by LiDAR is utilized to run deep neural network models for
3D object detection. However, these state-of-the-art models are complex,
leading to longer processing times and increased power consumption on edge
devices. The objective of this study is to address these issues by leveraging
Split Computing, a distributed machine learning inference method. Split
Computing aims to lessen the computational burden on edge devices, thereby
reducing processing time and power consumption. Furthermore, it minimizes the
risk of data breaches by only transmitting intermediate data from the deep
neural network model. Experimental results show that splitting after
voxelization reduces the inference time by 70.8% and the edge device execution
time by 90.0%. When splitting within the network, the inference time is reduced
by up to 57.1%, and the edge device execution time is reduced by up to 69.5%.

### 6. [Lightweight Latency Prediction Scheme for Edge Applications: A Rational Modelling Approach](http://arxiv.org/pdf/2511.02501v1)

Authors: Mohan Liyanage, Eldiyar Zhantileuov, Ali Kadhum Idrees, Rolf Schuster

Accurately predicting end-to-end network latency is essential for enabling
reliable task offloading in real-time edge computing applications. This paper
introduces a lightweight latency prediction scheme based on rational modelling
that uses features such as frame size, arrival rate, and link utilization,
eliminating the need for intrusive active probing. The model achieves
state-of-the-art prediction accuracy through extensive experiments and 5-fold
cross-validation (MAE = 0.0115, R$^2$ = 0.9847) with competitive inference
time, offering a substantial trade-off between precision and efficiency
compared to traditional regressors and neural networks.

### 7. [Implementing Multi-GPU Scientific Computing Miniapps Across Performance Portable Frameworks](http://arxiv.org/pdf/2511.02655v1)

Authors: Johansell Villalobos, Josef Ruzicka, Silvio Rizzi

Scientific computing in the exascale era demands increased computational
power to solve complex problems across various domains. With the rise of
heterogeneous computing architectures the need for vendor-agnostic, performance
portability frameworks has been highlighted. Libraries like Kokkos have become
essential for enabling high-performance computing applications to execute
efficiently across different hardware platforms with minimal code changes. In
this direction, this paper presents preliminary time-to-solution results for
two representative scientific computing applications: an N-body simulation and
a structured grid simulation. Both applications used a distributed memory
approach and hardware acceleration through four performance portability
frameworks: Kokkos, OpenMP, RAJA, and OCCA. Experiments conducted on a single
node of the Polaris supercomputer using four NVIDIA A100 GPUs revealed
significant performance variability among frameworks. OCCA demonstrated faster
execution times for small-scale validation problems, likely due to JIT
compilation, however its lack of optimized reduction algorithms may limit
scalability for larger simulations while using its out of the box API. OpenMP
performed poorly in the structured grid simulation most likely due to
inefficiencies in inter-node data synchronization and communication. These
findings highlight the need for further optimization to maximize each
framework's capabilities. Future work will focus on enhancing reduction
algorithms, data communication, memory management, as wells as performing
scalability studies, and a comprehensive statistical analysis to evaluate and
compare framework performance.

### 8. [Federated Attention: A Distributed Paradigm for Collaborative LLM Inference over Edge Networks](http://arxiv.org/pdf/2511.02647v1)

Authors: Xiumei Deng, Zehui Xiong, Binbin Chen, Dong In Kim, Merouane Debbah, H. Vincent Poor

Large language models (LLMs) are proliferating rapidly at the edge,
delivering intelligent capabilities across diverse application scenarios.
However, their practical deployment in collaborative scenarios confronts
fundamental challenges: privacy vulnerabilities, communication overhead, and
computational bottlenecks. To address these, we propose Federated Attention
(FedAttn), which integrates the federated paradigm into the self-attention
mechanism, creating a new distributed LLM inference framework that
simultaneously achieves privacy protection, communication efficiency, and
computational efficiency. FedAttn enables participants to perform local
self-attention over their own token representations while periodically
exchanging and aggregating Key-Value (KV) matrices across multiple Transformer
blocks, collaboratively generating LLM responses without exposing private
prompts. Further, we identify a structural duality between contextual
representation refinement in FedAttn and parameter optimization in FL across
private data, local computation, and global aggregation. This key insight
provides a principled foundation for systematically porting federated
optimization techniques to collaborative LLM inference. Building on this
framework, we theoretically analyze how local self-attention computation within
participants and heterogeneous token relevance among participants shape error
propagation dynamics across Transformer blocks. Moreover, we characterize the
fundamental trade-off between response quality and communication/computation
efficiency, which is governed by the synchronization interval and the number of
participants. Experimental results validate our theoretical analysis, and
reveal significant optimization opportunities through sparse attention and
adaptive KV aggregation, highlighting FedAttn's potential to deliver
scalability and efficiency in real-world edge deployments.

### Digital Libraries

### 1. [How large is the error effect when summing or averaging nonlinear field normalization citation counts at the paper level?](http://arxiv.org/pdf/2511.02255v1)

Authors: Limi Tang

Summing or averaging nonlinearly field-normalized citation counts is a common
but methodologically problematic practice, as it violates mathematical
principles. The issue originates from the nonlinear transformation, which
disrupts the equal-interval property of the data. Such unequal data do not
satisfy the necessary conditions for summation. In our study, we normalized
citation counts of papers from all sample universities using six linear and
nonlinear methods, and then computed the total and average scores for each
university under each method. By benchmarking against raw citations and linear
normalized scores, we explore how large the error effect is from summing or
averaging the nonlinear field normalized citation counts. Our empirical results
indicate that the error exists but is relatively small. We further found that
the magnitude of the error is significantly influenced by whether the sample
publications are homogeneous or heterogeneous. This study has significant
implications for whether the results obtained through nonlinear methods on a
single level can be directly summed or averaged when calculating the overall
impact of a research unit.

### 2. [Using language models to label clusters of scientific documents](http://arxiv.org/pdf/2511.02601v1)

Authors: Dakota Murray, Chaoqun Ni, Weiye Gu, Trevor Hubbard

Automated label generation for clusters of scientific documents is a common
task in bibliometric workflows. Traditionally, labels were formed by
concatenating distinguishing characteristics of a cluster's documents; while
straightforward, this approach often produces labels that are terse and
difficult to interpret. The advent and widespread accessibility of generative
language models, such as ChatGPT, make it possible to automatically generate
descriptive and human-readable labels that closely resemble those assigned by
human annotators. Language-model label generation has already seen widespread
use in bibliographic databases and analytical workflows. However, its rapid
adoption has outpaced the theoretical, practical, and empirical foundations. In
this study, we address the automated label generation task and make four key
contributions: (1) we define two distinct types of labels: characteristic and
descriptive, and contrast descriptive labeling with related tasks; (2) we
provide a formal descriptive labeling that clarifies important steps and design
considerations; (3) we propose a structured workflow for label generation and
outline practical considerations for its use in bibliometric workflows; and (4)
we develop an evaluative framework to assess descriptive labels generated by
language models and demonstrate that they perform at or near characteristic
labels, and highlight design considerations for their use. Together, these
contributions clarify the descriptive label generation task, establish an
empirical basis for the use of language models, and provide a framework to
guide future design and evaluation efforts.

### 3. [Research Output on Alopecia Areata Disease: A Scientometric Analysis of Publications from 2010 to 2019](http://arxiv.org/pdf/2511.02275v1)

Authors: Muneer Ahmad, M Sadik Batcha

The present study is undertaken to find out the publication trends on
Alopecia Areata Disease during 2010-2019 from the global perspective. The study
mainly focus on distribution of research output, top journals for publications,
most prolific authors, authorship pattern, and citations pattern on Alopecia
Areata Disease. The results indicate that highest growth rate of publications
occurred during the year 2019. Columbia University topped the scene among all
institutes. The maximum publications were more than four authored publications.
Christiano AM and Clynes R were found to be the most prolific authors. It is
also found that most of the prolific authors (by number of publications) do
appear in highly cited publications list. Alopecia Areata Disease researchers
mostly preferred using article publications to communicate their findings.

### 4. [Library and Culture: A Scientometric Analysis and Visualization of Research Trends](http://arxiv.org/pdf/2511.02296v1)

Authors: Auwalu Abdullahi Umar, Muneer Ahmad, Dr M Sadik Batcha

The significance of libraries in preserving and maintaining history and
traditional culture cannot be overlooked. It is from this purpose that
libraries are to envisage in their programmes cultural activities which must be
collected, documented and preserved for posterity. The usefulness of preserved
information lies in the fact that the generation to come will be able to
establish their identity. This will also assist them with a foundation to build
from. This study focus on the growth and development of Library and Culture
research in forms of publications reflected in Web of Science database, during
the span of 2010-2019. A total 890 publications were found and the highest 124
(13.93%) publications published in 2019.The analysis maps comprehensively the
parameters of total output, growth of output, authorship, institution wise and
country-level collaboration patterns, major contributors (individuals, top
publication sources, institutions, and countries). It exposed that the most
prolific author is Lo P secured first place by contributing 4 (0.45%)
publications, followed by Bressan V 3 (0.34%) publications in Library and
Culture literature. Journal of Academic Librarianship produced the highest
number of records 29 (3.26%) followed by Australian Library Journal having
contributed 21 (2.36%).It is identified the domination of Wuhan University;
School Information Management had contributed 6 (0.67%) of total research
output. Authors from USA published the highest number of publications with a
total of 244 (27.42%), followed by UK and Australia with 118 (13.26%) and 76
(8.54%) publications were produced respectively.

### Discrete Mathematics

### 1. [Emerging consecutive pattern avoidance](http://arxiv.org/pdf/2511.02442v1)

Authors: Nathanaël Hassler, Sergey Kirgizov

In this note we study the {\em asymptotic popularity}, that is, the limit
probability to find a given consecutive pattern at a random position in a
random permutation in the eighteen classes of permutations avoiding at least
two length 3 consecutive patterns. We show that for ten classes, this
popularity can be readily deduced from the structure of permutations. By
combining analytical and bijective approaches, we study in details two more
involved cases. The problem remains open for five classes.

### 2. [A Simple and Fast $(3+\varepsilon)$-approximation for Constrained Correlation Clustering](http://arxiv.org/pdf/2511.02705v1)

Authors: Nate Veldt

In Constrained Correlation Clustering, the goal is to cluster a complete
signed graph in a way that minimizes the number of negative edges inside
clusters plus the number of positive edges between clusters, while respecting
hard constraints on how to cluster certain friendly or hostile node pairs.
Fischer et al. [FKKT25a] recently developed a $\tilde{O}(n^3)$-time
16-approximation algorithm for this problem. We settle an open question posed
by these authors by designing an algorithm that is equally fast but brings the
approximation factor down to $(3+\varepsilon)$ for arbitrary constant
$\varepsilon > 0$. Although several new algorithmic steps are needed to obtain
our improved approximation, our approach maintains many advantages in terms of
simplicity. In particular, it relies mainly on rounding a (new) covering linear
program, which can be approximated quickly and combinatorially. Furthermore,
the rounding step amounts to applying the very familiar Pivot algorithm to an
auxiliary graph. Finally, we develop much simpler algorithms for instances that
involve only friendly or only hostile constraints.

### 3. [Arithmetic Circuits and Neural Networks for Regular Matroids](http://arxiv.org/pdf/2511.02406v1)

Authors: Christoph Hertrich, Stefan Kober, Georg Loho

We prove that there exist uniform $(+,\times,/)$-circuits of size $O(n^3)$ to
compute the basis generating polynomial of regular matroids on $n$ elements. By
tropicalization, this implies that there exist uniform $(\max,+,-)$-circuits
and ReLU neural networks of the same size for weighted basis maximization of
regular matroids. As a consequence in linear programming theory, we obtain a
first example where taking the difference of two extended formulations can be
more efficient than the best known individual extended formulation of size
$O(n^6)$ by Aprile and Fiorini. Such differences have recently been introduced
as virtual extended formulations. The proof of our main result relies on a
fine-tuned version of Seymour's decomposition of regular matroids which allows
us to identify and maintain graphic substructures to which we can apply a local
version of the star-mesh transformation.

### Data Structures and Algorithms

### 1. [Disjoint Paths in Expanders in Deterministic Almost-Linear Time via Hypergraph Perfect Matching](http://arxiv.org/pdf/2511.02214v1)

Authors: Matija Bucić, Zhongtian He, Shang-En Huang, Thatchaphol Saranurak

We design efficient deterministic algorithms for finding short edge-disjoint
paths in expanders. Specifically, given an $n$-vertex $m$-edge expander $G$ of
conductance $\phi$ and minimum degree $\delta$, and a set of pairs
$\{(s_i,t_i)\}_i$ such that each vertex appears in at most $k$ pairs, our
algorithm deterministically computes a set of edge-disjoint paths from $s_i$ to
$t_i$, one for every $i$: (1) each of length at most $18 \log (n)/\phi$ and in
$mn^{1+o(1)}\min\{k, \phi^{-1}\}$ total time, assuming $\phi^3\delta\ge (35\log
n)^3 k$, or (2) each of length at most $n^{o(1)}/\phi$ and in total
$m^{1+o(1)}$ time, assuming $\phi^3 \delta \ge n^{o(1)} k$. Before our work,
deterministic polynomial-time algorithms were known only for expanders with
constant conductance and were significantly slower. To obtain our result, we
give an almost-linear time algorithm for \emph{hypergraph perfect matching}
under generalizations of Hall-type conditions (Haxell 1995), a powerful
framework with applications in various settings, which until now has only
admitted large polynomial-time algorithms (Annamalai 2018).

### 2. [Accelerating Graph Similarity Search through Integer Linear Programming](http://arxiv.org/pdf/2511.02611v1)

Authors: Andrea D'Ascenzo, Julian Meffert, Petra Mutzel, Fabrizio Rossi

The Graph Edit Distance (GED) is an important metric for measuring the
similarity between two (labeled) graphs. It is defined as the minimum cost
required to convert one graph into another through a series of (elementary)
edit operations. Its effectiveness in assessing the similarity of large graphs
is limited by the complexity of its exact calculation, which is NP-hard
theoretically and computationally challenging in practice. The latter can be
mitigated by switching to the Graph Similarity Search under GED constraints,
which determines whether the edit distance between two graphs is below a given
threshold. A popular framework for solving Graph Similarity Search under GED
constraints in a graph database for a query graph is the
filter-and-verification framework. Filtering discards unpromising graphs, while
the verification step certifies the similarity between the filtered graphs and
the query graph. To improve the filtering step, we define a lower bound based
on an integer linear programming formulation. We prove that this lower bound
dominates the effective branch match-based lower bound and can also be computed
efficiently. Consequently, we propose a graph similarity search algorithm that
uses a hierarchy of lower bound algorithms and solves a novel integer
programming formulation that exploits the threshold parameter. An extensive
computational experience on a well-assessed test bed shows that our approach
significantly outperforms the state-of-the-art algorithm on most of the
examined thresholds.

### 3. [A Simple and Fast $(3+\varepsilon)$-approximation for Constrained Correlation Clustering](http://arxiv.org/pdf/2511.02705v1)

Authors: Nate Veldt

In Constrained Correlation Clustering, the goal is to cluster a complete
signed graph in a way that minimizes the number of negative edges inside
clusters plus the number of positive edges between clusters, while respecting
hard constraints on how to cluster certain friendly or hostile node pairs.
Fischer et al. [FKKT25a] recently developed a $\tilde{O}(n^3)$-time
16-approximation algorithm for this problem. We settle an open question posed
by these authors by designing an algorithm that is equally fast but brings the
approximation factor down to $(3+\varepsilon)$ for arbitrary constant
$\varepsilon > 0$. Although several new algorithmic steps are needed to obtain
our improved approximation, our approach maintains many advantages in terms of
simplicity. In particular, it relies mainly on rounding a (new) covering linear
program, which can be approximated quickly and combinatorially. Furthermore,
the rounding step amounts to applying the very familiar Pivot algorithm to an
auxiliary graph. Finally, we develop much simpler algorithms for instances that
involve only friendly or only hostile constraints.

### 4. [Mixing of general biased adjacent transposition chains](http://arxiv.org/pdf/2511.02725v1)

Authors: Reza Gheissari, Holden Lee, Eric Vigoda

We analyze the general biased adjacent transposition shuffle process, which
is a well-studied Markov chain on the symmetric group $S_n$. In each step, an
adjacent pair of elements $i$ and $j$ are chosen, and then $i$ is placed ahead
of $j$ with probability $p_{ij}$. This Markov chain arises in the study of
self-organizing lists in theoretical computer science, and has close
connections to exclusion processes from statistical physics and probability
theory. Fill (2003) conjectured that for general $p_{ij}$ satisfying $p_{ij}
\ge 1/2$ for all $i<j$ and a simple monotonicity condition, the mixing time is
polynomial. We prove that for any fixed $\varepsilon>0$, as long as $p_{ij}
>1/2+\varepsilon$ for all $i<j$, the mixing time is $\Theta(n^2)$ and exhibits
pre-cutoff. Our key technical result is a form of spatial mixing for the
general biased transposition chain after a suitable burn-in period. In order to
use this for a mixing time bound, we adapt multiscale arguments for mixing
times from the setting of spin systems to the symmetric group.

### 5. [Fast Approximation Algorithm for Non-Monotone DR-submodular Maximization under Size Constraint](http://arxiv.org/pdf/2511.02254v1)

Authors: Tan D. Tran, Canh V. Pham

This work studies the non-monotone DR-submodular Maximization over a ground
set of $n$ subject to a size constraint $k$. We propose two approximation
algorithms for solving this problem named FastDrSub and FastDrSub++. FastDrSub
offers an approximation ratio of $0.044$ with query complexity of $O(n
\log(k))$. The second one, FastDrSub++, improves upon it with a ratio of
$1/4-\epsilon$ within query complexity of $(n \log k)$ for an input parameter
$\epsilon >0$. Therefore, our proposed algorithms are the first constant-ratio
approximation algorithms for the problem with the low complexity of $O(n
\log(k))$.
  Additionally, both algorithms are experimentally evaluated and compared
against existing state-of-the-art methods, demonstrating their effectiveness in
solving the Revenue Maximization problem with DR-submodular objective function.
The experimental results show that our proposed algorithms significantly
outperform existing approaches in terms of both query complexity and solution
quality.

### 6. [Probabilistic Graph Cuts](http://arxiv.org/pdf/2511.02272v1)

Authors: Ayoub Ghriss

Probabilistic relaxations of graph cuts offer a differentiable alternative to
spectral clustering, enabling end-to-end and online learning without
eigendecompositions, yet prior work centered on RatioCut and lacked general
guarantees and principled gradients. We present a unified probabilistic
framework that covers a wide class of cuts, including Normalized Cut. Our
framework provides tight analytic upper bounds on expected discrete cuts via
integral representations and Gauss hypergeometric functions with closed-form
forward and backward. Together, these results deliver a rigorous, numerically
stable foundation for scalable, differentiable graph partitioning covering a
wide range of clustering and contrastive learning objectives.

### 7. [Learning CNF formulas from uniform random solutions in the local lemma regime](http://arxiv.org/pdf/2511.02487v1)

Authors: Weiming Feng, Xiongxin Yang, Yixiao Yu, Yiyao Zhang

We study the problem of learning a $n$-variables $k$-CNF formula $\Phi$ from
its i.i.d. uniform random solutions, which is equivalent to learning a Boolean
Markov random field (MRF) with $k$-wise hard constraints. Revisiting Valiant's
algorithm (Commun. ACM'84), we show that it can exactly learn (1) $k$-CNFs with
bounded clause intersection size under Lov\'asz local lemma type conditions,
from $O(\log n)$ samples; and (2) random $k$-CNFs near the satisfiability
threshold, from $\widetilde{O}(n^{\exp(-\sqrt{k})})$ samples. These results
significantly improve the previous $O(n^k)$ sample complexity. We further
establish new information-theoretic lower bounds on sample complexity for both
exact and approximate learning from i.i.d. uniform random solutions.

### Emerging Technologies

### 1. [Can Foundation Models Revolutionize Mobile AR Sparse Sensing?](http://arxiv.org/pdf/2511.02215v1)

Authors: Yiqin Zhao, Tian Guo

Mobile sensing systems have long faced a fundamental trade-off between
sensing quality and efficiency due to constraints in computation, power, and
other limitations. Sparse sensing, which aims to acquire and process only a
subset of sensor data, has been a key strategy for maintaining performance
under such constraints. However, existing sparse sensing methods often suffer
from reduced accuracy, as missing information across space and time introduces
uncertainty into many sensing systems. In this work, we investigate whether
foundation models can change the landscape of mobile sparse sensing. Using
real-world mobile AR data, our evaluations demonstrate that foundation models
offer significant improvements in geometry-aware image warping, a central
technique for enabling accurate reuse of cross-frame information. Furthermore,
our study demonstrates the scalability of foundation model-based sparse sensing
and shows its leading performance in 3D scene reconstruction. Collectively, our
study reveals critical aspects of the promises and the open challenges of
integrating foundation models into mobile sparse sensing systems.

### 2. [Efficient Variational Quantum Algorithms for the Generalized Assignment Problem](http://arxiv.org/pdf/2511.02739v1)

Authors: Carlo Mastroianni, Francesco Plastina, Jacopo Settino, Andrea Vinci

Quantum algorithms offer a compelling new avenue for addressing difficult
NP-complete optimization problems, such as the Generalized Assignment Problem
(GAP). Given the operational constraints of contemporary Noisy
Intermediate-Scale Quantum (NISQ) devices, hybrid quantum-classical approaches,
specifically Variational Quantum Algorithms (VQAs) like the Variational Quantum
Eigensolver (VQE), promises to be effective approaches to solve real-world
optimization problems. This paper proposes an approach, named VQGAP, designed
to efficiently solve the GAP by optimizing quantum resources and reducing the
required parametrized quantum circuit width with respect to standard VQE. The
main idea driving our proposal is to decouple the qubits of ansatz circuits
from the binary variables of the General Assignment Problem, by providing
encoding/decoding functions transforming the solutions generated by ansatze in
the limited quantum space in feasible solutions in the problem variables space,
by exploiting the constraints of the problem. Preliminary results, obtained
through both noiseless and noisy simulations, indicate that VQGAP exhibits
performance and behavior very similar to VQE, while effectively reducing the
number of qubits and circuit depth.

### Formal Languages and Automata Theory

### 1. [Non-commutative linear logic fragments with sub-context-free complexity](http://arxiv.org/pdf/2511.02348v1)

Authors: Yusaku Nishimiya, Masaya Taniguchi

We present new descriptive complexity characterisations of classes REG
(regular languages), LCFL (linear context-free languages) and CFL (context-free
languages) as restrictions on inference rules, size of formulae and permitted
connectives in the Lambek calculus; fragments of the intuitionistic
non-commutative linear logic with direction-sensitive implication connectives.
Our identification of the Lambek calculus fragments with proof complexity REG
and LCFL is the first result of its kind. We further show the CFL complexity of
one of the strictly `weakest' possible variants of the logic, admitting only a
single inference rule. The proof thereof, moreover, is based on a direct
translation between type-logical and formal grammar and structural induction on
provable sequents; a simpler and more intuitive method than those employed in
prior works. We thereby establish a clear conceptual utility of the
Cut-elimination theorem for comparing formal grammar and sequent calculus, and
identify the exact analogue of the Greibach Normal Form in Lambek grammar. We
believe the result presented herein constitutes a first step toward a more
extensive and richer characterisation of the interaction between computation
and logic, as well as a finer-grained complexity separation of various sequent
calculi.

### 2. [Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/pdf/2511.02304v1)

Authors: Beyazit Yalcinkaya, Marcell Vazquez-Chanlatte, Ameesh Shah, Hanna Krasowski, Sanjit A. Seshia

We study the problem of learning multi-task, multi-agent policies for
cooperative, temporal objectives, under centralized training, decentralized
execution. In this setting, using automata to represent tasks enables the
decomposition of complex tasks into simpler sub-tasks that can be assigned to
agents. However, existing approaches remain sample-inefficient and are limited
to the single-task case. In this work, we present Automata-Conditioned
Cooperative Multi-Agent Reinforcement Learning (ACC-MARL), a framework for
learning task-conditioned, decentralized team policies. We identify the main
challenges to ACC-MARL's feasibility in practice, propose solutions, and prove
the correctness of our approach. We further show that the value functions of
learned policies can be used to assign tasks optimally at test time.
Experiments show emergent task-aware, multi-step coordination among agents,
e.g., pressing a button to unlock a door, holding the door, and
short-circuiting tasks.

### Graphics

### 1. [OLATverse: A Large-scale Real-world Object Dataset with Precise Lighting Control](http://arxiv.org/pdf/2511.02483v1)

Authors: Xilong Zhou, Jianchun Chen, Pramod Rao, Timo Teufel, Linjie Lyu, Tigran Minasian, Oleksandr Sotnychenko, Xiaoxiao Long, Marc Habermann, Christian Theobalt

We introduce OLATverse, a large-scale dataset comprising around 9M images of
765 real-world objects, captured from multiple viewpoints under a diverse set
of precisely controlled lighting conditions. While recent advances in
object-centric inverse rendering, novel view synthesis and relighting have
shown promising results, most techniques still heavily rely on the synthetic
datasets for training and small-scale real-world datasets for benchmarking,
which limits their realism and generalization. To address this gap, OLATverse
offers two key advantages over existing datasets: large-scale coverage of real
objects and high-fidelity appearance under precisely controlled illuminations.
Specifically, OLATverse contains 765 common and uncommon real-world objects,
spanning a wide range of material categories. Each object is captured using 35
DSLR cameras and 331 individually controlled light sources, enabling the
simulation of diverse illumination conditions. In addition, for each object, we
provide well-calibrated camera parameters, accurate object masks, photometric
surface normals, and diffuse albedo as auxiliary resources. We also construct
an extensive evaluation set, establishing the first comprehensive real-world
object-centric benchmark for inverse rendering and normal estimation. We
believe that OLATverse represents a pivotal step toward integrating the next
generation of inverse rendering and relighting methods with real-world data.
The full dataset, along with all post-processing workflows, will be publicly
released at https://vcai.mpi-inf.mpg.de/projects/OLATverse/.

### 2. [TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](http://arxiv.org/pdf/2511.02580v1)

Authors: Daichi Nagai, Ryugo Morita, Shunsuke Kitada, Hitoshi Iyatomi

Despite the remarkable success of text-to-image diffusion models, their
output of a single, flattened image remains a critical bottleneck for
professional applications requiring layer-wise control. Existing solutions
either rely on fine-tuning with large, inaccessible datasets or are
training-free yet limited to generating isolated foreground elements, failing
to produce a complete and coherent scene. To address this, we introduce the
Training-free Noise Transplantation and Cultivation Diffusion Model (TAUE), a
novel framework for zero-shot, layer-wise image generation. Our core technique,
Noise Transplantation and Cultivation (NTC), extracts intermediate latent
representations from both foreground and composite generation processes,
transplanting them into the initial noise for subsequent layers. This ensures
semantic and structural coherence across foreground, background, and composite
layers, enabling consistent, multi-layered outputs without requiring
fine-tuning or auxiliary datasets. Extensive experiments show that our
training-free method achieves performance comparable to fine-tuned methods,
enhancing layer-wise consistency while maintaining high image quality and
fidelity. TAUE not only eliminates costly training and dataset requirements but
also unlocks novel downstream applications, such as complex compositional
editing, paving the way for more accessible and controllable generative
workflows.

### Computer Science and Game Theory

### 1. [Human-AI Collaboration with Misaligned Preferences](http://arxiv.org/pdf/2511.02746v1)

Authors: Jiaxin Song, Parnian Shahkar, Kate Donahue, Bhaskar Ray Chaudhury

In many real-life settings, algorithms play the role of assistants, while
humans ultimately make the final decision. Often, algorithms specifically act
as curators, narrowing down a wide range of options into a smaller subset that
the human picks between: consider content recommendation or chatbot responses
to questions with multiple valid answers. Crucially, humans may not know their
own preferences perfectly either, but instead may only have access to a noisy
sampling over preferences. Algorithms can assist humans by curating a smaller
subset of items, but must also face the challenge of misalignment: humans may
have different preferences from each other (and from the algorithm), and the
algorithm may not know the exact preferences of the human they are facing at
any point in time. In this paper, we model and theoretically study such a
setting. Specifically, we show instances where humans benefit by collaborating
with a misaligned algorithm. Surprisingly, we show that humans gain more
utility from a misaligned algorithm (which makes different mistakes) than from
an aligned algorithm. Next, we build on this result by studying what properties
of algorithms maximize human welfare when the goals could be either utilitarian
welfare or ensuring all humans benefit. We conclude by discussing implications
for designers of algorithmic tools and policymakers.

### 2. [AI-Generated Image Detection: An Empirical Study and Future Research Directions](http://arxiv.org/pdf/2511.02791v1)

Authors: Nusrat Tasnim, Kutub Uddin, Khalid Mahmood Malik

The threats posed by AI-generated media, particularly deepfakes, are now
raising significant challenges for multimedia forensics, misinformation
detection, and biometric system resulting in erosion of public trust in the
legal system, significant increase in frauds, and social engineering attacks.
Although several forensic methods have been proposed, they suffer from three
critical gaps: (i) use of non-standardized benchmarks with GAN- or
diffusion-generated images, (ii) inconsistent training protocols (e.g.,
scratch, frozen, fine-tuning), and (iii) limited evaluation metrics that fail
to capture generalization and explainability. These limitations hinder fair
comparison, obscure true robustness, and restrict deployment in
security-critical applications. This paper introduces a unified benchmarking
framework for systematic evaluation of forensic methods under controlled and
reproducible conditions. We benchmark ten SoTA forensic methods (scratch,
frozen, and fine-tuned) and seven publicly available datasets (GAN and
diffusion) to perform extensive and systematic evaluations. We evaluate
performance using multiple metrics, including accuracy, average precision,
ROC-AUC, error rate, and class-wise sensitivity. We also further analyze model
interpretability using confidence curves and Grad-CAM heatmaps. Our evaluations
demonstrate substantial variability in generalization, with certain methods
exhibiting strong in-distribution performance but degraded cross-model
transferability. This study aims to guide the research community toward a
deeper understanding of the strengths and limitations of current forensic
approaches, and to inspire the development of more robust, generalizable, and
explainable solutions.

### 3. [Near Optimal Convergence to Coarse Correlated Equilibrium in General-Sum Markov Games](http://arxiv.org/pdf/2511.02157v1)

Authors: Asrin Efe Yorulmaz, Tamer Başar

No-regret learning dynamics play a central role in game theory, enabling
decentralized convergence to equilibrium for concepts such as Coarse Correlated
Equilibrium (CCE) or Correlated Equilibrium (CE). In this work, we improve the
convergence rate to CCE in general-sum Markov games, reducing it from the
previously best-known rate of $\mathcal{O}(\log^5 T / T)$ to a sharper
$\mathcal{O}(\log T / T)$. This matches the best known convergence rate for CE
in terms of $T$, number of iterations, while also improving the dependence on
the action set size from polynomial to polylogarithmic-yielding exponential
gains in high-dimensional settings. Our approach builds on recent advances in
adaptive step-size techniques for no-regret algorithms in normal-form games,
and extends them to the Markovian setting via a stage-wise scheme that adjusts
learning rates based on real-time feedback. We frame policy updates as an
instance of Optimistic Follow-the-Regularized-Leader (OFTRL), customized for
value-iteration-based learning. The resulting self-play algorithm achieves, to
our knowledge, the fastest known convergence rate to CCE in Markov games.

### Human-Computer Interaction

### 1. [Learning Spatial Awareness for Laparoscopic Surgery with AI Assisted Visual Feedback](http://arxiv.org/pdf/2511.02233v1)

Authors: Songyang Liu, Yunpeng Tan, Shuai Li

Laparoscopic surgery constrains surgeons spatial awareness because procedures
are performed through a monocular, two-dimensional (2D) endoscopic view.
Conventional training methods using dry-lab models or recorded videos provide
limited depth cues, often leading trainees to misjudge instrument position and
perform ineffective or unsafe maneuvers. To address this limitation, we present
an AI-assisted training framework developed in NVIDIA Isaac Sim that couples
the standard 2D laparoscopic feed with synchronized three-dimensional (3D)
visual feedback delivered through a mixed-reality (MR) interface. While
trainees operate using the clinical 2D view, validated AI modules continuously
localize surgical instruments and detect instrument-tissue interactions in the
background. When spatial misjudgments are detected, 3D visual feedback are
displayed to trainees, while preserving the original operative perspective. Our
framework considers various surgical tasks including navigation, manipulation,
transfer, cutting, and suturing. Visually similar 2D cases can be disambiguated
through the added 3D context, improving depth perception, contact awareness,
and tool orientation understanding.

### 2. [The Pervasive Blind Spot: Benchmarking VLM Inference Risks on Everyday Personal Videos](http://arxiv.org/pdf/2511.02367v1)

Authors: Shuning Zhang, Zhaoxin Li, Changxi Wen, Ying Ma, Simin Li, Gengrui Zhang, Ziyi Zhang, Yibo Meng, Hantao Zhao, Xin Yi, Hewu Li

The proliferation of Vision-Language Models (VLMs) introduces profound
privacy risks from personal videos. This paper addresses the critical yet
unexplored inferential privacy threat, the risk of inferring sensitive personal
attributes over the data. To address this gap, we crowdsourced a dataset of 508
everyday personal videos from 58 individuals. We then conducted a benchmark
study evaluating VLM inference capabilities against human performance. Our
findings reveal three critical insights: (1) VLMs possess superhuman
inferential capabilities, significantly outperforming human evaluators,
leveraging a shift from object recognition to behavioral inference from
temporal streams. (2) Inferential risk is strongly correlated with factors such
as video characteristics and prompting strategies. (3) VLM-driven explanation
towards the inference is unreliable, as we revealed a disconnect between the
model-generated explanations and evidential impact, identifying ubiquitous
objects as misleading confounders.

### 3. [Revisiting put-that-there, context aware window interactions via LLMs](http://arxiv.org/pdf/2511.02378v1)

Authors: Riccardo Bovo, Daniele Giunchi, Pasquale Cascarano, Eric J. Gonzalez, Mar Gonzalez-Franco

We revisit Bolt's classic "Put-That-There" concept for modern head-mounted
displays by pairing Large Language Models (LLMs) with XR sensor and tech stack.
The agent fuses (i) a semantically segmented 3-D environment, (ii) live
application metadata, and (iii) users' verbal, pointing, and head-gaze cues to
issue JSON window-placement actions. As a result, users can manage a panoramic
workspace through: (1) explicit commands ("Place Google Maps on the coffee
table"), (2) deictic speech plus gestures ("Put that there"), or (3) high-level
goals ("I need to send a message"). Unlike traditional explicit interfaces, our
system supports one-to-many action mappings and goal-centric reasoning,
allowing the LLM to dynamically infer relevant applications and layout
decisions, including interrelationships across tools. This enables seamless,
intent-driven interaction without manual window juggling in immersive XR
environments.

### 4. [Can Conversational AI Counsel for Change? A Theory-Driven Approach to Supporting Dietary Intentions in Ambivalent Individuals](http://arxiv.org/pdf/2511.02428v1)

Authors: Michelle Bak, Kexin Quan, Tre Tomaszewski, Jessie Chin

Adherence to healthy diets reduces chronic illness risk, yet rates remain
low. Large Language Models (LLMs) are increasingly used for health
communication but often struggle to engage individuals with ambivalent
intentions at a pivotal stage of the Transtheoretical Model (TTM). We developed
CounselLLM, an open-source model enhanced through persona design and few-shot,
domain-specific prompts grounded in TTM and Motivational Interviewing (MI). In
controlled evaluations, CounselLLM showed stronger use of TTM subprocesses and
MI affirmations than human counselors, with comparable linguistic robustness
but expressed in more concrete terms. A user study then tested CounselLLM in an
interactive counseling setting against a baseline system. While knowledge and
perceptions did not change, participants' intentions for immediate dietary
change increased significantly after interacting with CounselLLM. Participants
also rated it as easy to use, understandable, and supportive. These findings
suggest theory-driven LLMs can effectively engage ambivalent individuals and
provide a scalable approach to digital counseling.

### 5. [OpenCourier: an Open Protocol for Building a Decentralized Ecosystem of Community-owned Delivery Platforms](http://arxiv.org/pdf/2511.02455v1)

Authors: Yuhan Liu, Varun Nagaraj Rao, Sohyeon Hwang, Janet Vertesi, Andrés Monroy-Hernández

Although the platform gig economy has reshaped the landscape of work, its
centralized operation by select actors has brought about challenges that
impedes workers' well-being. We present the architecture and design of
OpenCourier, an open protocol that defines communication patterns within a
decentralized ecosystem of delivery platforms. Through this protocol, we aim to
address three key challenges in the current economy: power imbalances between
the platform and workers, information asymmetries caused by black-boxed
algorithms and value misalignments in the infrastructure design process. With
the OpenCourier protocol, we outline a blueprint for community-owned ecosystem
of delivery platforms that centers worker agency, transparency, and bottom-up
design.

### 6. [DropleX: Liquid sensing on tablet touchscreens](http://arxiv.org/pdf/2511.02694v1)

Authors: Siqi Zhang, Mayank Goel, Justin Chan

We present DropleX, the first system that enables liquid sensing using the
capacitive touchscreen of commodity tablets. DropleX detects microliter-scale
liquid samples, and performs non-invasive, through-container measurements to
detect whether a drink has been spiked or if a sealed liquid has been
contaminated. These capabilities are made possible by a physics-informed
mechanism that disables the touchscreen's built-in adaptive filters, originally
designed to reject the effects of liquid drops such as rain, without any
hardware modifications. We model the touchscreen's sensing capabilities,
limits, and non-idealities to inform the design of a signal processing and
learning-based pipeline for liquid sensing. Our system achieves 96-99% accuracy
in detecting microliter-scale adulteration in soda, wine, and milk, 93-96%
accuracy in threshold detection of trace chemical concentrations, and 86-96%
accuracy in through-container adulterant detection. Given the predominance of
touchscreens, these exploratory results can open new opportunities for liquid
sensing on everyday devices.

### 7. [Audience Amplified: Virtual Audiences in Asynchronously Performed AR Theater](http://arxiv.org/pdf/2511.02807v1)

Authors: You-Jin Kim, Misha Sra, Tobias Höllerer

Audience reactions can considerably enhance live experiences; conversely, in
anytime-anywhere augmented reality (AR) experiences, large crowds of people
might not always be available to congregate. To get closer to simulating live
events with large audiences, we created a mobile AR experience where users can
wander around naturally and engage in AR theater with virtual audiences trained
from real audiences using imitation learning. This allows us to carefully
capture the essence of human imperfections and behavior in artificial
intelligence (AI) audiences. The result is a novel mobile AR experience in
which solitary AR users experience an augmented performance in a physical space
with a virtual audience. Virtual dancers emerge from the surroundings,
accompanied by a digitally simulated audience, to provide a community
experience akin to immersive theater. In a pilot study, simulated human avatars
were vastly preferred over just audience audio commentary. We subsequently
engaged 20 participants as attendees of an AR dance performance, comparing a
no-audience condition with a simulated audience of six onlookers. Through
questionnaires and experience reports, we investigated user reactions and
behavior. Our results demonstrate that the presence of virtual audience members
caused attendees to perceive the performance as a social experience with
increased interest and involvement in the event. On the other hand, for some
attendees, the dance performances without the virtual audience evoked a
stronger positive sentiment.

### 8. [AI Credibility Signals Outrank Institutions and Engagement in Shaping News Perception on Social Media](http://arxiv.org/pdf/2511.02370v1)

Authors: Adnan Hoq, Matthew Facciani, Tim Weninger

AI-generated content is rapidly becoming a salient component of online
information ecosystems, yet its influence on public trust and epistemic
judgments remains poorly understood. We present a large-scale mixed-design
experiment (N = 1,000) investigating how AI-generated credibility scores affect
user perception of political news. Our results reveal that AI feedback
significantly moderates partisan bias and institutional distrust, surpassing
traditional engagement signals such as likes and shares. These findings
demonstrate the persuasive power of generative AI and suggest a need for design
strategies that balance epistemic influence with user autonomy.

### 9. [HAGI++: Head-Assisted Gaze Imputation and Generation](http://arxiv.org/pdf/2511.02468v1)

Authors: Chuhan Jiao, Zhiming Hu, Andreas Bulling

Mobile eye tracking plays a vital role in capturing human visual attention
across both real-world and extended reality (XR) environments, making it an
essential tool for applications ranging from behavioural research to
human-computer interaction. However, missing values due to blinks, pupil
detection errors, or illumination changes pose significant challenges for
further gaze data analysis. To address this challenge, we introduce HAGI++ - a
multi-modal diffusion-based approach for gaze data imputation that, for the
first time, uses the integrated head orientation sensors to exploit the
inherent correlation between head and eye movements. HAGI++ employs a
transformer-based diffusion model to learn cross-modal dependencies between eye
and head representations and can be readily extended to incorporate additional
body movements. Extensive evaluations on the large-scale Nymeria, Ego-Exo4D,
and HOT3D datasets demonstrate that HAGI++ consistently outperforms
conventional interpolation methods and deep learning-based time-series
imputation baselines in gaze imputation. Furthermore, statistical analyses
confirm that HAGI++ produces gaze velocity distributions that closely match
actual human gaze behaviour, ensuring more realistic gaze imputations.
Moreover, by incorporating wrist motion captured from commercial wearable
devices, HAGI++ surpasses prior methods that rely on full-body motion capture
in the extreme case of 100% missing gaze data (pure gaze generation). Our
method paves the way for more complete and accurate eye gaze recordings in
real-world settings and has significant potential for enhancing gaze-based
analysis and interaction across various application domains.

### 10. [Emotional Contagion in Code: How GitHub Emoji Reactions Shape Developer Collaboration](http://arxiv.org/pdf/2511.02515v1)

Authors: Obada Kraishan

Developer communities increasingly rely on emoji reactions to communicate,
but we know little about how these emotional signals spread and influence
technical discussions. We analyzed 2,098 GitHub issues and pull requests across
50 popular repositories, examining patterns in 106,743 emoji reactions to
understand emotional contagion in software development. Our findings reveal a
surprisingly positive emotional landscape: 57.4% of discussions carry positive
sentiment, with positive emotional cascades outnumbering negative ones 23:1. We
identified five distinct patterns, with "instant enthusiasm" affecting 45.6% of
items--nearly half receive immediate positive reinforcement. Statistical
analysis confirms strong emotional contagion (r=0.679, p<0.001) with a massive
effect size (d=2.393), suggesting that initial reactions powerfully shape
discussion trajectories. These findings challenge assumptions about technical
discourse being purely rational, demonstrating that even minimal emotional
signals create measurable ripple effects. Our work provides empirical evidence
that emoji reactions are not mere decoration but active forces shaping
collaborative outcomes in software development.

### Information Retrieval

### 1. [KGBridge: Knowledge-Guided Prompt Learning for Non-overlapping Cross-Domain Recommendation](http://arxiv.org/pdf/2511.02181v1)

Authors: Yuhan Wang, Qing Xie, Zhifeng Bao, Mengzi Tang, Lin Li, Yongjian Liu

Knowledge Graphs (KGs), as structured knowledge bases that organize
relational information across diverse domains, provide a unified semantic
foundation for cross-domain recommendation (CDR). By integrating symbolic
knowledge with user-item interactions, KGs enrich semantic representations,
support reasoning, and enhance model interpretability. Despite this potential,
existing KG-based methods still face major challenges in CDR, particularly
under non-overlapping user scenarios. These challenges arise from: (C1)
sensitivity to KG sparsity and popularity bias, (C2) dependence on overlapping
users for domain alignment and (C3) lack of explicit disentanglement between
transferable and domain-specific knowledge, which limit effective and stable
knowledge transfer. To this end, we propose KGBridge, a knowledge-guided prompt
learning framework for cross-domain sequential recommendation under
non-overlapping user scenarios. KGBridge comprises two core components: a
KG-enhanced Prompt Encoder, which models relation-level semantics as soft
prompts to provide structured and dynamic priors for user sequence modeling
(addressing C1), and a Two-stage Training Paradigm, which combines cross-domain
pretraining and privacy-preserving fine-tuning to enable knowledge transfer
without user overlap (addressing C2). By combining relation-aware semantic
control with correspondence-driven disentanglement, KGBridge explicitly
separates and balances domain-shared and domain-specific semantics, thereby
maintaining complementarity and stabilizing adaptation during fine-tuning
(addressing C3). Extensive experiments on benchmark datasets demonstrate that
KGBridge consistently outperforms state-of-the-art baselines and remains robust
under varying KG sparsity, highlighting its effectiveness in mitigating
structural imbalance and semantic entanglement in KG-enhanced cross-domain
recommendation.

### 2. [Research Output on Alopecia Areata Disease: A Scientometric Analysis of Publications from 2010 to 2019](http://arxiv.org/pdf/2511.02275v1)

Authors: Muneer Ahmad, M Sadik Batcha

The present study is undertaken to find out the publication trends on
Alopecia Areata Disease during 2010-2019 from the global perspective. The study
mainly focus on distribution of research output, top journals for publications,
most prolific authors, authorship pattern, and citations pattern on Alopecia
Areata Disease. The results indicate that highest growth rate of publications
occurred during the year 2019. Columbia University topped the scene among all
institutes. The maximum publications were more than four authored publications.
Christiano AM and Clynes R were found to be the most prolific authors. It is
also found that most of the prolific authors (by number of publications) do
appear in highly cited publications list. Alopecia Areata Disease researchers
mostly preferred using article publications to communicate their findings.

### 3. [Library and Culture: A Scientometric Analysis and Visualization of Research Trends](http://arxiv.org/pdf/2511.02296v1)

Authors: Auwalu Abdullahi Umar, Muneer Ahmad, Dr M Sadik Batcha

The significance of libraries in preserving and maintaining history and
traditional culture cannot be overlooked. It is from this purpose that
libraries are to envisage in their programmes cultural activities which must be
collected, documented and preserved for posterity. The usefulness of preserved
information lies in the fact that the generation to come will be able to
establish their identity. This will also assist them with a foundation to build
from. This study focus on the growth and development of Library and Culture
research in forms of publications reflected in Web of Science database, during
the span of 2010-2019. A total 890 publications were found and the highest 124
(13.93%) publications published in 2019.The analysis maps comprehensively the
parameters of total output, growth of output, authorship, institution wise and
country-level collaboration patterns, major contributors (individuals, top
publication sources, institutions, and countries). It exposed that the most
prolific author is Lo P secured first place by contributing 4 (0.45%)
publications, followed by Bressan V 3 (0.34%) publications in Library and
Culture literature. Journal of Academic Librarianship produced the highest
number of records 29 (3.26%) followed by Australian Library Journal having
contributed 21 (2.36%).It is identified the domination of Wuhan University;
School Information Management had contributed 6 (0.67%) of total research
output. Authors from USA published the highest number of publications with a
total of 244 (27.42%), followed by UK and Australia with 118 (13.26%) and 76
(8.54%) publications were produced respectively.

### 4. [Average Precision at Cutoff k under Random Rankings: Expectation and Variance](http://arxiv.org/pdf/2511.02571v1)

Authors: Tetiana Manzhos, Tetiana Ianevych, Olga Melnyk

Recommender systems and information retrieval platforms rely on ranking
algorithms to present the most relevant items to users, thereby improving
engagement and satisfaction. Assessing the quality of these rankings requires
reliable evaluation metrics. Among them, Mean Average Precision at cutoff k
(MAP@k) is widely used, as it accounts for both the relevance of items and
their positions in the list.
  In this paper, the expectation and variance of Average Precision at k (AP@k)
are derived since they can be used as biselines for MAP@k. Here, we covered two
widely used evaluation models: offline and online. The expectation establishes
the baseline, indicating the level of MAP@k that can be achieved by pure
chance. The variance complements this baseline by quantifying the extent of
random fluctuations, enabling a more reliable interpretation of observed
scores.

### 5. [Relational Deep Dive: Error-Aware Queries Over Unstructured Data](http://arxiv.org/pdf/2511.02711v1)

Authors: Daren Chao, Kaiwen Chen, Naiqing Guan, Nick Koudas

Unstructured data is pervasive, but analytical queries demand structured
representations, creating a significant extraction challenge. Existing methods
like RAG lack schema awareness and struggle with cross-document alignment,
leading to high error rates. We propose ReDD (Relational Deep Dive), a
framework that dynamically discovers query-specific schemas, populates
relational tables, and ensures error-aware extraction with provable guarantees.
ReDD features a two-stage pipeline: (1) Iterative Schema Discovery (ISD)
identifies minimal, joinable schemas tailored to each query, and (2) Tabular
Data Population (TDP) extracts and corrects data using lightweight classifiers
trained on LLM hidden states. A main contribution of ReDD is SCAPE, a
statistically calibrated method for error detection with coverage guarantees,
and SCAPE-HYB, a hybrid approach that optimizes the trade-off between accuracy
and human correction costs. Experiments across diverse datasets demonstrate
ReDD's effectiveness, reducing data extraction errors from up to 30% to below
1% while maintaining high schema completeness (100% recall) and precision.
ReDD's modular design enables fine-grained control over accuracy-cost
trade-offs, making it a robust solution for high-stakes analytical queries over
unstructured corpora.

### 6. [Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval](http://arxiv.org/pdf/2511.02770v1)

Authors: Hung-Ting Chen, Xiang Liu, Shauli Ravfogel, Eunsol Choi

Most text retrievers generate \emph{one} query vector to retrieve relevant
documents. Yet, the conditional distribution of relevant documents for the
query may be multimodal, e.g., representing different interpretations of the
query. We first quantify the limitations of existing retrievers. All retrievers
we evaluate struggle more as the distance between target document embeddings
grows. To address this limitation, we develop a new retriever architecture,
\emph{A}utoregressive \emph{M}ulti-\emph{E}mbedding \emph{R}etriever (AMER).
Our model autoregressively generates multiple query vectors, and all the
predicted query vectors are used to retrieve documents from the corpus. We show
that on the synthetic vectorized data, the proposed method could capture
multiple target distributions perfectly, showing 4x better performance than
single embedding model. We also fine-tune our model on real-world multi-answer
retrieval datasets and evaluate in-domain. AMER presents 4 and 21\% relative
gains over single-embedding baselines on two datasets we evaluate on.
Furthermore, we consistently observe larger gains on the subset of dataset
where the embeddings of the target documents are less similar to each other. We
demonstrate the potential of using a multi-query vector retriever and open up a
new direction for future work.

### 7. [Let Multimodal Embedders Learn When to Augment Query via Adaptive Query Augmentation](http://arxiv.org/pdf/2511.02358v1)

Authors: Wongyu Kim, Hochang Lee, Sanghak Lee, Yoonsung Kim, Jaehyun Park

Query augmentation makes queries more meaningful by appending further
information to the queries to find relevant documents. Current studies have
proposed Large Language Model (LLM)-based embedders, which learn representation
for embedding and generation for query augmentation in a multi-task manner by
leveraging the generative capabilities of LLM. During inference, these jointly
trained embedders have conducted query augmentation followed by embedding,
showing effective results. However, augmenting every query leads to substantial
embedding latency and query augmentation can be detrimental to performance for
some queries. Also, previous methods have not been explored in multimodal
environments. To tackle these problems, we propose M-Solomon, a universal
multimodal embedder that can adaptively determine when to augment queries. Our
approach first divides the queries of the training datasets into two groups at
the dataset level. One includes queries that require augmentation and the other
includes queries that do not. Then, we introduces a synthesis process that
generates appropriate augmentations for queries that require them by leveraging
a powerful Multimodal LLM (MLLM). Next, we present adaptive query augmentation.
Through this step, M-Solomon can conduct query augmentation only when necessary
by learning to generate synthetic augmentations with the prefix /augment for
queries that demand them and to generate the simple string /embed for others.
Experimental results showed that M-Solomon not only surpassed the baseline
without augmentation by a large margin but also outperformed the baseline that
always used augmentation, providing much faster embedding latency.

### Machine Learning

### 1. [CFL: On the Use of Characteristic Function Loss for Domain Alignment in Machine Learning](http://arxiv.org/pdf/2511.02148v1)

Authors: Abdullah Almansour, Ozan Tonguz

Machine Learning (ML) models are extensively used in various applications due
to their significant advantages over traditional learning methods. However, the
developed ML models often underperform when deployed in the real world due to
the well-known distribution shift problem. This problem can lead to a
catastrophic outcomes when these decision-making systems have to operate in
high-risk applications. Many researchers have previously studied this problem
in ML, known as distribution shift problem, using statistical techniques (such
as Kullback-Leibler, Kolmogorov-Smirnov Test, Wasserstein distance, etc.) to
quantify the distribution shift. In this letter, we show that using
Characteristic Function (CF) as a frequency domain approach is a powerful
alternative for measuring the distribution shift in high-dimensional space and
for domain adaptation.

### 2. [ProtoTSNet: Interpretable Multivariate Time Series Classification With Prototypical Parts](http://arxiv.org/pdf/2511.02152v1)

Authors: Bartłomiej Małkus, Szymon Bobek, Grzegorz J. Nalepa

Time series data is one of the most popular data modalities in critical
domains such as industry and medicine. The demand for algorithms that not only
exhibit high accuracy but also offer interpretability is crucial in such
fields, as decisions made there bear significant consequences. In this paper,
we present ProtoTSNet, a novel approach to interpretable classification of
multivariate time series data, through substantial enhancements to the
ProtoPNet architecture. Our method is tailored to overcome the unique
challenges of time series analysis, including capturing dynamic patterns and
handling varying feature significance. Central to our innovation is a modified
convolutional encoder utilizing group convolutions, pre-trainable as part of an
autoencoder and designed to preserve and quantify feature importance. We
evaluated our model on 30 multivariate time series datasets from the UEA
archive, comparing our approach with existing explainable methods as well as
non-explainable baselines. Through comprehensive evaluation and ablation
studies, we demonstrate that our approach achieves the best performance among
ante-hoc explainable methods while maintaining competitive performance with
non-explainable and post-hoc explainable approaches, providing interpretable
results accessible to domain experts.

### 3. [Learning Interactive World Model for Object-Centric Reinforcement Learning](http://arxiv.org/pdf/2511.02225v1)

Authors: Fan Feng, Phillip Lippe, Sara Magliacane

Agents that understand objects and their interactions can learn policies that
are more robust and transferable. However, most object-centric RL methods
factor state by individual objects while leaving interactions implicit. We
introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a
unified framework that learns structured representations of both objects and
their interactions within a world model. FIOC-WM captures environment dynamics
with disentangled and modular representations of object interactions, improving
sample efficiency and generalization for policy learning. Concretely, FIOC-WM
first learns object-centric latents and an interaction structure directly from
pixels, leveraging pre-trained vision encoders. The learned world model then
decomposes tasks into composable interaction primitives, and a hierarchical
policy is trained on top: a high level selects the type and order of
interactions, while a low level executes them. On simulated robotic and
embodied-AI benchmarks, FIOC-WM improves policy-learning sample efficiency and
generalization over world-model baselines, indicating that explicit, modular
interaction learning is crucial for robust control.

### 4. [Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining](http://arxiv.org/pdf/2511.02237v1)

Authors: Costin-Andrei Oncescu, Qingyang Wu, Wai Tong Chung, Robert Wu, Bryan Gopal, Junxiong Wang, Tri Dao, Ben Athiwaratkun

An increasing number of LLMs employ Mixture-of-Experts (MoE) architectures
where the feed-forward layer is replaced by a pool of experts and each token
only activates a small subset of them. During autoregressive generation, these
models often enter a memory-bound regime even for moderate batch sizes because
the average expert load grows more slowly than in an equivalent dense
feedforward layer. Consequently, MoE latency is governed by the number of
activated experts. We introduce a framework for dynamically re-routing
token-to-expert mapping to lower this number (and thus, the decode latency)
while preserving a comparable quality. Our best results use a batch-aware
routing that works by having tokens piggyback experts that have already been
loaded into memory due to being crucial to other tokens within the same batch.
Empirically, we evaluate our method on the Qwen3-30B and Qwen3-235B models with
a batch size of $16$. Without any statistically significant loss in accuracy,
our approach achieves latency reductions of $39\%$ and $15\%$ in the MoE layer
decode latency, respectively.

### 5. [Neural network initialization with nonlinear characteristics and information on spectral bias](http://arxiv.org/pdf/2511.02244v1)

Authors: Hikaru Homma, Jun Ohkubo

Initialization of neural network parameters, such as weights and biases, has
a crucial impact on learning performance; if chosen well, we can even avoid the
need for additional training with backpropagation. For example, algorithms
based on the ridgelet transform or the SWIM (sampling where it matters) concept
have been proposed for initialization. On the other hand, it is well-known that
neural networks tend to learn coarse information in the earlier layers. The
feature is called spectral bias. In this work, we investigate the effects of
utilizing information on the spectral bias in the initialization of neural
networks. Hence, we propose a framework that adjusts the scale factors in the
SWIM algorithm to capture low-frequency components in the early-stage hidden
layers and to represent high-frequency components in the late-stage hidden
layers. Numerical experiments on a one-dimensional regression task and the
MNIST classification task demonstrate that the proposed method outperforms the
conventional initialization algorithms. This work clarifies the importance of
intrinsic spectral properties in learning neural networks, and the finding
yields an effective parameter initialization strategy that enhances their
training performance.

### 6. [Reinforcement learning based data assimilation for unknown state model](http://arxiv.org/pdf/2511.02286v1)

Authors: Ziyi Wang, Lijian Jiang

Data assimilation (DA) has increasingly emerged as a critical tool for state
estimation
  across a wide range of applications. It is signiffcantly challenging when the
governing equations of the underlying dynamics are unknown. To this end,
various machine learning approaches have been employed to construct a surrogate
state transition
  model in a supervised learning framework, which relies on pre-computed
training
  datasets. However, it is often infeasible to obtain noise-free ground-truth
state sequences in practice. To address this challenge, we propose a novel
method that integrates reinforcement learning with ensemble-based Bayesian
ffltering methods, enabling
  the learning of surrogate state transition model for unknown dynamics
directly from noisy observations, without using true state trajectories.
Speciffcally, we treat the process for computing maximum likelihood estimation
of surrogate model parameters
  as a sequential decision-making problem, which can be formulated as a
discretetime
  Markov decision process (MDP). Under this formulation, learning the surrogate
transition model is equivalent to ffnding an optimal policy of the MDP, which
can be effectively addressed using reinforcement learning techniques. Once the
model is trained offfine, state estimation can be performed in the online stage
using ffltering methods based on the learned dynamics. The proposed framework
accommodates a wide range of observation scenarios, including nonlinear and
partially observed measurement
  models. A few numerical examples demonstrate that the proposed method
achieves superior accuracy and robustness in high-dimensional settings.

### 7. [RoME: Domain-Robust Mixture-of-Experts for MILP Solution Prediction across Domains](http://arxiv.org/pdf/2511.02331v1)

Authors: Tianle Pu, Zijie Geng, Haoyang Liu, Shixuan Liu, Jie Wang, Li Zeng, Chao Chen, Changjun Fan

Mixed-Integer Linear Programming (MILP) is a fundamental and powerful
framework for modeling complex optimization problems across diverse domains.
Recently, learning-based methods have shown great promise in accelerating MILP
solvers by predicting high-quality solutions. However, most existing approaches
are developed and evaluated in single-domain settings, limiting their ability
to generalize to unseen problem distributions. This limitation poses a major
obstacle to building scalable and general-purpose learning-based solvers. To
address this challenge, we introduce RoME, a domain-Robust Mixture-of-Experts
framework for predicting MILP solutions across domains. RoME dynamically routes
problem instances to specialized experts based on learned task embeddings. The
model is trained using a two-level distributionally robust optimization
strategy: inter-domain to mitigate global shifts across domains, and
intra-domain to enhance local robustness by introducing perturbations on task
embeddings. We reveal that cross-domain training not only enhances the model's
generalization capability to unseen domains but also improves performance
within each individual domain by encouraging the model to capture more general
intrinsic combinatorial patterns. Specifically, a single RoME model trained on
three domains achieves an average improvement of 67.7% then evaluated on five
diverse domains. We further test the pretrained model on MIPLIB in a zero-shot
setting, demonstrating its ability to deliver measurable performance gains on
challenging real-world instances where existing learning-based approaches often
struggle to generalize.

### 8. [Learning A Universal Crime Predictor with Knowledge-guided Hypernetworks](http://arxiv.org/pdf/2511.02336v1)

Authors: Fidan Karimova, Tong Chen, Yu Yang, Shazia Sadiq

Predicting crimes in urban environments is crucial for public safety, yet
existing prediction methods often struggle to align the knowledge across
diverse cities that vary dramatically in data availability of specific crime
types. We propose HYpernetwork-enhanced Spatial Temporal Learning (HYSTL), a
framework that can effectively train a unified, stronger crime predictor
without assuming identical crime types in different cities' records. In HYSTL,
instead of parameterising a dedicated predictor per crime type, a hypernetwork
is designed to dynamically generate parameters for the prediction function
conditioned on the crime type of interest. To bridge the semantic gap between
different crime types, a structured crime knowledge graph is built, where the
learned representations of crimes are used as the input to the hypernetwork to
facilitate parameter generation. As such, when making predictions for each
crime type, the predictor is additionally guided by its intricate association
with other relevant crime types. Extensive experiments are performed on two
cities with non-overlapping crime types, and the results demonstrate HYSTL
outperforms state-of-the-art baselines.

### 9. [Evolving Graph Learning for Out-of-Distribution Generalization in Non-stationary Environments](http://arxiv.org/pdf/2511.02354v1)

Authors: Qingyun Sun, Jiayi Luo, Haonan Yuan, Xingcheng Fu, Hao Peng, Jianxin Li, Philip S. Yu

Graph neural networks have shown remarkable success in exploiting the spatial
and temporal patterns on dynamic graphs. However, existing GNNs exhibit poor
generalization ability under distribution shifts, which is inevitable in
dynamic scenarios. As dynamic graph generation progresses amid evolving latent
non-stationary environments, it is imperative to explore their effects on
out-of-distribution (OOD) generalization. This paper proposes a novel Evolving
Graph Learning framework for OOD generalization (EvoOOD) by environment-aware
invariant pattern recognition. Specifically, we first design an environment
sequential variational auto-encoder to model environment evolution and infer
the underlying environment distribution. Then, we introduce a mechanism for
environment-aware invariant pattern recognition, tailored to address
environmental diversification through inferred distributions. Finally, we
conduct fine-grained causal interventions on individual nodes using a mixture
of instantiated environment samples. This approach helps to distinguish
spatio-temporal invariant patterns for OOD prediction, especially in
non-stationary environments. Experimental results demonstrate the superiority
of EvoGOOD on both real-world and synthetic dynamic datasets under distribution
shifts. To the best of our knowledge, it is the first attempt to study the
dynamic graph OOD generalization problem from the environment evolution
perspective.

### 10. [LUMA-RAG: Lifelong Multimodal Agents with Provably Stable Streaming Alignment](http://arxiv.org/pdf/2511.02371v1)

Authors: Rohan Wandre, Yash Gajewar, Namrata Patel, Vivek Dhalkari

Retrieval-Augmented Generation (RAG) has emerged as the dominant paradigm for
grounding large language model outputs in verifiable evidence. However, as
modern AI agents transition from static knowledge bases to continuous
multimodal streams encompassing text, images, video, and audio, two critical
challenges arise: maintaining index freshness without prohibitive re-indexing
costs, and preserving cross-modal semantic consistency across heterogeneous
embedding spaces. We present LUMA-RAG, a lifelong multimodal agent architecture
featuring three key innovations: (i) a streaming, multi-tier memory system that
dynamically spills embeddings from a hot HNSW tier to a compressed IVFPQ tier
under strict memory budgets; (ii) a streaming CLAP->CLIP alignment bridge that
maintains cross-modal consistency through incremental orthogonal Procrustes
updates; and (iii) stability-aware retrieval telemetry providing Safe@k
guarantees by jointly bounding alignment drift and quantization error.
Experiments demonstrate robust text-to-image retrieval (Recall@10 = 0.94),
graceful performance degradation under product quantization offloading, and
provably stable audio-to-image rankings (Safe@1 = 1.0), establishing LUMA-RAG
as a practical framework for production multimodal RAG systems.

### Neural and Evolutionary Computing

### 1. [Evolutionary Algorithm for Chance Constrained Quadratic Multiple Knapsack Problem](http://arxiv.org/pdf/2511.02500v1)

Authors: Kokila Kasuni Perera, Aneta Neumann

Quadratic multiple knapsack problem (QMKP) is a combinatorial optimisation
problem characterised by multiple weight capacity constraints and a profit
function that combines linear and quadratic profits. We study a stochastic
variant of this problem where profits are considered as random variables. This
problem reflects complex resource allocation problems in real-world scenarios
where randomness is inherent. We model this problem using chance constraints to
capture the stochastic profits. We propose a hybrid approach for this problem,
which combines an evolutionary algorithm (EA) with a local optimisation
strategy inspired by multi-factorial optimisation (MFO). EAs are used for
global search due to their effectiveness in handling large, complex solution
spaces. In the hybrid approach, EA periodically passes interim solutions to the
local optimiser for refinement. The local optimiser applies MFO principles,
which are typically used in multi-tasking problems. The local optimiser models
the local problem as a multi-tasking problem by constructing disjoint search
spaces for each knapsack based on an input solution. For each item, its
assignment across all knapsacks is considered to determine the preferred
knapsack. Items are then divided into disjoint groups corresponding to each
knapsack, allowing each knapsack to be treated as a separate optimisation task.
This structure enables effective application of MFO-based local refinements. We
consider two EAs for the problem, (1+1) EA and ($\mu+\lambda$) EA. We conduct
experiments to explore the effectiveness of these EAs on their own and also
with the proposed local optimiser. Experimental results suggest that hybrid
approaches, particularly those incorporating MFO, perform well on instances
where chance constraints and capacity constraints are tight.

### 2. [Structural Plasticity as Active Inference: A Biologically-Inspired Architecture for Homeostatic Control](http://arxiv.org/pdf/2511.02241v1)

Authors: Brennen A. Hill

Traditional neural networks, while powerful, rely on biologically implausible
learning mechanisms such as global backpropagation. This paper introduces the
Structurally Adaptive Predictive Inference Network (SAPIN), a novel
computational model inspired by the principles of active inference and the
morphological plasticity observed in biological neural cultures. SAPIN operates
on a 2D grid where processing units, or cells, learn by minimizing local
prediction errors. The model features two primary, concurrent learning
mechanisms: a local, Hebbian-like synaptic plasticity rule based on the
temporal difference between a cell's actual activation and its learned
expectation, and a structural plasticity mechanism where cells physically
migrate across the grid to optimize their information-receptive fields. This
dual approach allows the network to learn both how to process information
(synaptic weights) and also where to position its computational resources
(network topology). We validated the SAPIN model on the classic Cart Pole
reinforcement learning benchmark. Our results demonstrate that the architecture
can successfully solve the CartPole task, achieving robust performance. The
network's intrinsic drive to minimize prediction error and maintain homeostasis
was sufficient to discover a stable balancing policy. We also found that while
continual learning led to instability, locking the network's parameters after
achieving success resulted in a stable policy. When evaluated for 100 episodes
post-locking (repeated over 100 successful agents), the locked networks
maintained an average 82% success rate.

### 3. [Redundancy Maximization as a Principle of Associative Memory Learning](http://arxiv.org/pdf/2511.02584v1)

Authors: Mark Blümel, Andreas C. Schneider, Valentin Neuhaus, David A. Ehrlich, Marcel Graetz, Michael Wibral, Abdullah Makkeh, Viola Priesemann

Associative memory, traditionally modeled by Hopfield networks, enables the
retrieval of previously stored patterns from partial or noisy cues. Yet, the
local computational principles which are required to enable this function
remain incompletely understood. To formally characterize the local information
processing in such systems, we employ a recent extension of information theory
- Partial Information Decomposition (PID). PID decomposes the contribution of
different inputs to an output into unique information from each input,
redundant information across inputs, and synergistic information that emerges
from combining different inputs. Applying this framework to individual neurons
in classical Hopfield networks we find that below the memory capacity, the
information in a neuron's activity is characterized by high redundancy between
the external pattern input and the internal recurrent input, while synergy and
unique information are close to zero until the memory capacity is surpassed and
performance drops steeply. Inspired by this observation, we use redundancy as
an information-theoretic learning goal, which is directly optimized for each
neuron, dramatically increasing the network's memory capacity to 1.59, a more
than tenfold improvement over the 0.14 capacity of classical Hopfield networks
and even outperforming recent state-of-the-art implementations of Hopfield
networks. Ultimately, this work establishes redundancy maximization as a new
design principle for associative memories and opens pathways for new
associative memory models based on information-theoretic goals.

### Networking and Internet Architecture

### 1. [Permissioned Blockchain in Advanced Air Mobility: A Performance Analisys for UTM](http://arxiv.org/pdf/2511.02171v1)

Authors: Rodrigo Nunes, André Melo, Rafael Albarello, Reinaldo Gomes, Cesar Marcondes, Lourenço Pereira Jr

The rapid adoption of Uncrewed Aerial Vehicles (UAVs) has driven aviation
authorities to propose distributed Uncrewed Traffic Management (UTM)
architectures. Several studies have advocated blockchain as a promising
technology to meet these requirements. However, since UTM is a safety-critical
and highly regulated domain, compliance with standards and regulatory
frameworks is as crucial as performance and security. This work benchmarks two
distributed architectures aligned with current regulatory frameworks: the Linux
Foundation's InterUSS platform and a Hyperledger Fabric-based private ledger.
Our findings reveal that blockchain-based systems require architectures
specifically designed for aeronautical performance constraints.

### 2. [Optimizing Multi-UAV 3D Deployment for Energy-Efficient Sensing over Uneven Terrains](http://arxiv.org/pdf/2511.02368v1)

Authors: Rushi Moliya, Dhaval K. Patel, Brijesh Soni, Miguel López-Benítez

In this work, we consider a multi-unmanned aerial vehicle (UAV) cooperative
sensing system where UAVs are deployed to sense multiple targets in
terrain-aware line of sight (LoS) conditions in uneven terrain equipped with
directional antennas. To mitigate terrain-induced LoS blockages that degrade
detection performance, we incorporate a binary LoS indicator and propose a
bounding volume hierarchy (BHV)-based adaptive scheme for efficient LoS
evaluation. We formulate a bi-objective problem that maximizes the probability
of cooperative detection with minimal hover energy constraints governing
spatial, orientational, and safety constraints. To address the problem, which
is inherently non-convex, we propose a hierarchical heuristic framework that
combines exploration through a genetic algorithm (GA) with per-UAV refinement
via particle swarm optimization (PSO), where a penalty-based fitness evaluation
guides solutions toward feasibility, bounded within constraints. The proposed
methodology is an effective trade-off method of traversing through a complex
search space and maintaining terrain-aware LoS connectivity and energy aware
deployment. Monte Carlo simulations on real-world terrain data show that the
proposed GA+PSO framework improves detection probability by 37.02% and 36.5%
for 2 and 3 UAVs, respectively, while reducing average excess hover energy by
45.0% and 48.9% compared to the PSO-only baseline. Relative to the
non-optimized scheme, it further achieves 59.5% and 54.2% higher detection
probability with 59.8% and 65.9% lower excess hover energy, thereby showing its
effectiveness with a small number of UAVs over uneven terrain.

### 3. [Janus: Leveraging Incremental Computation for Efficient DNS Verification](http://arxiv.org/pdf/2511.02559v1)

Authors: Yao Wang, Kexin Yu, Wenyun Xu, Kaiqiang Hu, Ziyi Wang, Lizhao You, Qiang Su, Dong Guo, Haizhou Du, Wanjian Feng, Qingyu Song, Linghe Kong, Qiao Xiang, Jiwu Shu

Existing DNS configuration verification tools face significant issues (e.g.,
inefficient and lacking support for incremental verification). Inspired by the
advancements in recent work of distributed data plane verification and the
resemblance be- tween the data plane and DNS configuration, we tackle the
challenge of DNS misconfiguration by introducing Janus, a DNS verification
tool. Our key insight is that the process of a nameserver handling queries can
be transformed into a matching process on a match-action table. With this
insight, Janus consists of (1) an efficient data structure for partition query
space based on the behaviors, (2) a symbolic execution algorithm that specifies
how a single nameserver can efficiently cover all possible queries and ensure
the accuracy of verification, (3) a mechanism to support incremental
verification with less computational effort. Extensive experiments on
real-world datasets (with over 6 million resource records) show that Janus
achieves significant speedups, with peak improvements of up to 255.7x and a
maximum 6046x reduction in the number of LECs.

### 4. [Decentralized AI Service Placement, Selection and Routing in Mobile Networks](http://arxiv.org/pdf/2511.02638v1)

Authors: Jinkun Zhang, Stefan Vlaski, Kin Leung

The rapid development and usage of large-scale AI models by mobile users will
dominate the traffic load in future communication networks. The advent of AI
technology also facilitates a decentralized AI ecosystem where small
organizations or even individuals can host AI services. In such scenarios, AI
service (models) placement, selection, and request routing decisions are
tightly coupled, posing a challenging yet fundamental trade-off between service
quality and service latency, especially when considering user mobility.
Existing solutions for related problems in mobile edge computing (MEC) and
data-intensive networks fall short due to restrictive assumptions about network
structure or user mobility. To bridge this gap, we propose a decentralized
framework that jointly optimizes AI service placement, selection, and request
routing. In the proposed framework, we use traffic tunneling to support user
mobility without costly AI service migrations. To account for nonlinear queuing
delays, we formulate a nonconvex problem to optimize the trade-off between
service quality and end-to-end latency. We derive the node-level KKT conditions
and develop a decentralized Frank--Wolfe algorithm with a novel messaging
protocol. Numerical evaluations validate the proposed approach and show
substantial performance improvements over existing methods.

### 5. [CRRM: A 5G system-level simulator](http://arxiv.org/pdf/2511.02692v1)

Authors: Keith Briggs, Ibrahim Nur

System-level simulation is indispensable for developing and testing novel
algorithms for 5G and future wireless networks, yet a gap persists between the
needs of the machine learning re- search community and the available tooling.
To address this, we introduce the Cellular Radio Reference Model (CRRM), an
open-source, pure Python simulator we designed specifically for speed,
usability, and direct integration with modern AI frameworks. The core
scientific contribution of CRRM lies in its architecture, which departs from
traditional discrete-event simulation. We model the system as a set of
inter-dependent computational blocks which form nodes in a directed graph. This
enables a compute-on-demand mechanism we term smart update.

### 6. [On the Optimization of Model Aggregation for Federated Learning at the Network Edge](http://arxiv.org/pdf/2511.02703v1)

Authors: Mengyao Li, Noah Ploch, Sebastian Troia, Carlo Spatocco, Wolfgang Kellerer, Guido Maier

The rapid increase in connected devices has signifi- cantly intensified the
computational and communication demands on modern telecommunication networks.
To address these chal- lenges, integrating advanced Machine Learning (ML)
techniques like Federated Learning (FL) with emerging paradigms such as
Multi-access Edge Computing (MEC) and Software-Defined Wide Area Networks
(SD-WANs) is crucial. This paper intro- duces online resource management
strategies specifically designed for FL model aggregation, utilizing
intermediate aggregation at edge nodes. Our analysis highlights the benefits of
incorporating edge aggregators to reduce network link congestion and maximize
the potential of edge computing nodes. However, the risk of network congestion
persists. To mitigate this, we propose a novel aggregation approach that
deploys an aggregator overlay network. We present an Integer Linear Programming
(ILP) model and a heuristic algorithm to optimize the routing within this
overlay network. Our solution demonstrates improved adapt- ability to network
resource utilization, significantly reducing FL training round failure rates by
up to 15% while also alleviating cloud link congestion.

### 7. [Lightweight Latency Prediction Scheme for Edge Applications: A Rational Modelling Approach](http://arxiv.org/pdf/2511.02501v1)

Authors: Mohan Liyanage, Eldiyar Zhantileuov, Ali Kadhum Idrees, Rolf Schuster

Accurately predicting end-to-end network latency is essential for enabling
reliable task offloading in real-time edge computing applications. This paper
introduces a lightweight latency prediction scheme based on rational modelling
that uses features such as frame size, arrival rate, and link utilization,
eliminating the need for intrusive active probing. The model achieves
state-of-the-art prediction accuracy through extensive experiments and 5-fold
cross-validation (MAE = 0.0115, R$^2$ = 0.9847) with competitive inference
time, offering a substantial trade-off between precision and efficiency
compared to traditional regressors and neural networks.

### 8. [Agentic World Modeling for 6G: Near-Real-Time Generative State-Space Reasoning](http://arxiv.org/pdf/2511.02748v1)

Authors: Farhad Rezazadeh, Hatim Chergui, Merouane Debbah, Houbing Song, Dusit Niyato, Lingjia Liu

We argue that sixth-generation (6G) intelligence is not fluent token
prediction but the capacity to imagine and choose -- to simulate future
scenarios, weigh trade-offs, and act with calibrated uncertainty. We reframe
open radio access network (O-RAN) near-real-time (Near-RT) control via
counterfactual dynamics and a world modeling (WM) paradigm that learns an
action-conditioned generative state space. This enables quantitative "what-if"
forecasting beyond large language models (LLMs) as the primary modeling
primitive. Actions such as physical resource blocks (PRBs) are treated as
first-class control inputs in a causal world model, and both aleatoric and
epistemic uncertainty are modeled for prediction and what-if analysis. An
agentic, model predictive control (MPC)-based cross-entropy method (CEM)
planner operates over short horizons, using prior-mean rollouts within
data-driven PRB bounds to maximize a deterministic reward. The model couples
multi-scale structured state-space mixtures (MS3M) with a compact stochastic
latent to form WM-MS3M, summarizing key performance indicators (KPIs) histories
and predicting next-step KPIs under hypothetical PRB sequences. On realistic
O-RAN traces, WM-MS3M cuts mean absolute error (MAE) by 1.69% versus MS3M with
32% fewer parameters and similar latency, and achieves 35-80% lower root mean
squared error (RMSE) than attention/hybrid baselines with 2.3-4.1x faster
inference, enabling rare-event simulation and offline policy screening.

### 9. [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](http://arxiv.org/pdf/2511.02230v1)

Authors: Hanchen Li, Qiuyang Mang, Runyuan He, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Alvin Cheung, Joseph Gonzalez, Ion Stoica

Agentic LLM applications interleave LLM generation requests with tool calls.
These tool calls break the continuity of the workflow by creating pauses
between LLM requests, bringing many challenges for the serving system,
especially under multi-turn scenarios. Each pause potentially causes KV cache
eviction and extra waiting time before entering the continuous batch for the
following LLM request. Since these pauses happen for each call, this problem
becomes increasingly severe as turn number grow for agentic programs. Previous
works either fail to incorporate information from the tool call, evicting KV
cache that leads to repetitive prefill or loading, or ignore the continuity of
a multi-turn program, creating waiting time between turns that increases
per-request latency.
  We present Continuum, a serving system to optimize job completion time for
multi-turn agent workloads by combining tool-aware KV cache timeout with
program-level scheduling. By predicting tool call durations in agentic
workflows, Continuum selectively pins the KV cache in GPU memory with a
time-to-live value based on total turn number. When combined with program-level
first-come-first-serve, Continuum prevents scheduling bubbles, preserves
multi-turn continuity, and optimizes for throughput for complex agentic
workflows. By modeling the variability of tool call and agent program
continuity, Continuum outperforms state-of-the-art baselines. Our evaluation on
real-world agentic workloads (SWE-Bench and BFCL) with Llama-3.1 8B/70B models
shows that Continuum significantly improves the average job completion times,
and remains performant across different hardware setups and DRAM offloading
schemes. Preview code is available at:
https://github.com/Hanchenli/vllm-continuum

### Robotics

### 1. [Kinematic and Ergonomic Design of a Robotic Arm for Precision Laparoscopic Surgery](http://arxiv.org/pdf/2511.02167v1)

Authors: Tian Hao, Tong Lu, Che Chan

Robotic assistance in minimally invasive surgery can greatly enhance surgical
precision and reduce surgeon fatigue. This paper presents a focused
investigation on the kinematic and ergonomic design principles for a
laparoscopic surgical robotic arm aimed at high-precision tasks. We propose a
7-degree-of-freedom (7-DOF) robotic arm system that incorporates a remote
center of motion (RCM) at the instrument insertion point and ergonomic
considerations to improve surgeon interaction. The design is implemented on a
general-purpose robotic platform, and a series of simulated surgical tasks were
performed to evaluate targeting accuracy, task efficiency, and surgeon comfort
compared to conventional manual laparoscopy. Experimental results demonstrate
that the optimized robotic design achieves significantly improved targeting
accuracy (error reduced by over 50%) and shorter task completion times, while
substantially lowering operator muscle strain and discomfort. These findings
validate the importance of kinematic optimization (such as added articulations
and tremor filtering) and human-centered ergonomic design in enhancing the
performance of robot-assisted surgery. The insights from this work can guide
the development of next-generation surgical robots that improve surgical
outcomes and ergonomics for the operating team.

### 2. [A Quantitative Comparison of Centralised and Distributed Reinforcement Learning-Based Control for Soft Robotic Arms](http://arxiv.org/pdf/2511.02192v1)

Authors: Linxin Hou, Qirui Wu, Zhihang Qin, Neil Banerjee, Yongxin Guo, Cecilia Laschi

This paper presents a quantitative comparison between centralised and
distributed multi-agent reinforcement learning (MARL) architectures for
controlling a soft robotic arm modelled as a Cosserat rod in simulation. Using
PyElastica and the OpenAI Gym interface, we train both a global Proximal Policy
Optimisation (PPO) controller and a Multi-Agent PPO (MAPPO) under identical
budgets. Both approaches are based on the arm having $n$ number of controlled
sections. The study systematically varies $n$ and evaluates the performance of
the arm to reach a fixed target in three scenarios: default baseline condition,
recovery from external disturbance, and adaptation to actuator failure.
Quantitative metrics used for the evaluation are mean action magnitude, mean
final distance, mean episode length, and success rate. The results show that
there are no significant benefits of the distributed policy when the number of
controlled sections $n\le4$. In very simple systems, when $n\le2$, the
centralised policy outperforms the distributed one. When $n$ increases to $4<
n\le 12$, the distributed policy shows a high sample efficiency. In these
systems, distributed policy promotes a stronger success rate, resilience, and
robustness under local observability and yields faster convergence given the
same sample size. However, centralised policies achieve much higher time
efficiency during training as it takes much less time to train the same size of
samples. These findings highlight the trade-offs between centralised and
distributed policy in reinforcement learning-based control for soft robotic
systems and provide actionable design guidance for future sim-to-real transfer
in soft rod-like manipulators.

### 3. [SuckTac: Camera-based Tactile Sucker for Unstructured Surface Perception and Interaction](http://arxiv.org/pdf/2511.02294v1)

Authors: Ruiyong Yuan, Jieji Ren, Zhanxuan Peng, Feifei Chen, Guoying Gu

Suckers are significant for robots in picking, transferring, manipulation and
locomotion on diverse surfaces. However, most of the existing suckers lack
high-fidelity perceptual and tactile sensing, which impedes them from resolving
the fine-grained geometric features and interaction status of the target
surface. This limits their robust performance with irregular objects and in
complex, unstructured environments. Inspired by the adaptive structure and
high-performance sensory capabilities of cephalopod suckers, in this paper, we
propose a novel, intelligent sucker, named SuckTac, that integrates a
camera-based tactile sensor directly within its optimized structure to provide
high-density perception and robust suction. Specifically, through joint
structure design and optimization and based on a multi-material integrated
casting technique, a camera and light source are embedded into the sucker,
which enables in-situ, high-density perception of fine details like surface
shape, texture and roughness. To further enhance robustness and adaptability,
the sucker's mechanical design is also optimized by refining its profile,
adding a compliant lip, and incorporating surface microstructure. Extensive
experiments, including challenging tasks such as robotic cloth manipulation and
soft mobile robot inspection, demonstrate the superior performance and broad
applicability of the proposed system.

### 4. [Whole-body motion planning and safety-critical control for aerial manipulation](http://arxiv.org/pdf/2511.02342v1)

Authors: Lin Yang, Jinwoo Lee, Domenico Campolo, H. Jin Kim, Jeonghyun Byun

Aerial manipulation combines the maneuverability of multirotors with the
dexterity of robotic arms to perform complex tasks in cluttered spaces. Yet
planning safe, dynamically feasible trajectories remains difficult due to
whole-body collision avoidance and the conservativeness of common geometric
abstractions such as bounding boxes or ellipsoids. We present a whole-body
motion planning and safety-critical control framework for aerial manipulators
built on superquadrics (SQs). Using an SQ-plus-proxy representation, we model
both the vehicle and obstacles with differentiable, geometry-accurate surfaces.
Leveraging this representation, we introduce a maximum-clearance planner that
fuses Voronoi diagrams with an equilibrium-manifold formulation to generate
smooth, collision-aware trajectories. We further design a safety-critical
controller that jointly enforces thrust limits and collision avoidance via
high-order control barrier functions. In simulation, our approach outperforms
sampling-based planners in cluttered environments, producing faster, safer, and
smoother trajectories and exceeding ellipsoid-based baselines in geometric
fidelity. Actual experiments on a physical aerial-manipulation platform confirm
feasibility and robustness, demonstrating consistent performance across
simulation and hardware settings. The video can be found at
https://youtu.be/hQYKwrWf1Ak.

### 5. [Dexterous Robotic Piano Playing at Scale](http://arxiv.org/pdf/2511.02504v1)

Authors: Le Chen, Yi Zhao, Jan Schneider, Quankai Gao, Simon Guist, Cheng Qian, Juho Kannala, Bernhard Schölkopf, Joni Pajarinen, Dieter Büchler

Endowing robot hands with human-level dexterity has been a long-standing goal
in robotics. Bimanual robotic piano playing represents a particularly
challenging task: it is high-dimensional, contact-rich, and requires fast,
precise control. We present OmniPianist, the first agent capable of performing
nearly one thousand music pieces via scalable, human-demonstration-free
learning. Our approach is built on three core components. First, we introduce
an automatic fingering strategy based on Optimal Transport (OT), allowing the
agent to autonomously discover efficient piano-playing strategies from scratch
without demonstrations. Second, we conduct large-scale Reinforcement Learning
(RL) by training more than 2,000 agents, each specialized in distinct music
pieces, and aggregate their experience into a dataset named RP1M++, consisting
of over one million trajectories for robotic piano playing. Finally, we employ
a Flow Matching Transformer to leverage RP1M++ through large-scale imitation
learning, resulting in the OmniPianist agent capable of performing a wide range
of musical pieces. Extensive experiments and ablation studies highlight the
effectiveness and scalability of our approach, advancing dexterous robotic
piano playing at scale.

### 6. [Non-Contact Manipulation of Induced Magnetic Dipoles](http://arxiv.org/pdf/2511.02761v1)

Authors: Seth Stewart, Joseph Pawelski, Steve Ward, Andrew J. Petruska

Extending the field of magnetic manipulation to conductive, non-magnetic
objects opens the door for a wide array of applications previously limited to
hard or soft magnetic materials. Of particular interest is the recycling of
space debris through the use of oscillating magnetic fields, which represent a
cache of raw materials in an environment particularly suited to the low forces
generated from inductive magnetic manipulation. Building upon previous work
that demonstrated 3D open-loop position control by leveraging the opposing
dipole moment created from induced eddy currents, this work demonstrates
closed-loop position control of a semi-buoyant aluminum sphere in lab tests,
and the efficacy of varying methods for force inversion is explored. The
closed-loop methods represent a critical first step towards wider applications
for 3-DOF position control of induced magnetic dipoles.

### 7. [XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations](http://arxiv.org/pdf/2511.02776v1)

Authors: Shichao Fan, Kun Wu, Zhengping Che, Xinhua Wang, Di Wu, Fei Liao, Ning Liu, Yixue Zhang, Zhen Zhao, Zhiyuan Xu, Meng Li, Qingjie Liu, Shanghang Zhang, Min Wan, Jian Tang

Recent progress in large-scale robotic datasets and vision-language models
(VLMs) has advanced research on vision-language-action (VLA) models. However,
existing VLA models still face two fundamental challenges: (i) producing
precise low-level actions from high-dimensional observations, (ii) bridging
domain gaps across heterogeneous data sources, including diverse robot
embodiments and human demonstrations. Existing methods often encode latent
variables from either visual dynamics or robotic actions to guide policy
learning, but they fail to fully exploit the complementary multi-modal
knowledge present in large-scale, heterogeneous datasets. In this work, we
present X Robotic Model 1 (XR-1), a novel framework for versatile and scalable
VLA learning across diverse robots, tasks, and environments. XR-1 introduces
the \emph{Unified Vision-Motion Codes (UVMC)}, a discrete latent representation
learned via a dual-branch VQ-VAE that jointly encodes visual dynamics and
robotic motion. UVMC addresses these challenges by (i) serving as an
intermediate representation between the observations and actions, and (ii)
aligning multimodal dynamic information from heterogeneous data sources to
capture complementary knowledge. To effectively exploit UVMC, we propose a
three-stage training paradigm: (i) self-supervised UVMC learning, (ii)
UVMC-guided pretraining on large-scale cross-embodiment robotic datasets, and
(iii) task-specific post-training. We validate XR-1 through extensive
real-world experiments with more than 14,000 rollouts on six different robot
embodiments, spanning over 120 diverse manipulation tasks. XR-1 consistently
outperforms state-of-the-art baselines such as $\pi_{0.5}$, $\pi_0$, RDT,
UniVLA, and GR00T-N1.5 while demonstrating strong generalization to novel
objects, background variations, distractors, and illumination changes. Our
project is at https://xr-1-vla.github.io/.

### 8. [LACY: A Vision-Language Model-based Language-Action Cycle for Self-Improving Robotic Manipulation](http://arxiv.org/pdf/2511.02239v1)

Authors: Youngjin Hong, Houjian Yu, Mingen Li, Changhyun Choi

Learning generalizable policies for robotic manipulation increasingly relies
on large-scale models that map language instructions to actions (L2A). However,
this one-way paradigm often produces policies that execute tasks without deeper
contextual understanding, limiting their ability to generalize or explain their
behavior. We argue that the complementary skill of mapping actions back to
language (A2L) is essential for developing more holistic grounding. An agent
capable of both acting and explaining its actions can form richer internal
representations and unlock new paradigms for self-supervised learning. We
introduce LACY (Language-Action Cycle), a unified framework that learns such
bidirectional mappings within a single vision-language model. LACY is jointly
trained on three synergistic tasks: generating parameterized actions from
language (L2A), explaining observed actions in language (A2L), and verifying
semantic consistency between two language descriptions (L2C). This enables a
self-improving cycle that autonomously generates and filters new training data
through an active augmentation strategy targeting low-confidence cases, thereby
improving the model without additional human labels. Experiments on
pick-and-place tasks in both simulation and the real world show that LACY
improves task success rates by 56.46% on average and yields more robust
language-action grounding for robotic manipulation. Project page:
https://vla2026.github.io/LACY/

### 9. [Synthetic Crop-Weed Image Generation and its Impact on Model Generalization](http://arxiv.org/pdf/2511.02417v1)

Authors: Garen Boyadjian, Cyrille Pierre, Johann Laconte, Riccardo Bertoglio

Precise semantic segmentation of crops and weeds is necessary for
agricultural weeding robots. However, training deep learning models requires
large annotated datasets, which are costly to obtain in real fields. Synthetic
data can reduce this burden, but the gap between simulated and real images
remains a challenge. In this paper, we present a pipeline for procedural
generation of synthetic crop-weed images using Blender, producing annotated
datasets under diverse conditions of plant growth, weed density, lighting, and
camera angle. We benchmark several state-of-the-art segmentation models on
synthetic and real datasets and analyze their cross-domain generalization. Our
results show that training on synthetic images leads to a sim-to-real gap of
10%, surpassing previous state-of-the-art methods. Moreover, synthetic data
demonstrates good generalization properties, outperforming real datasets in
cross-domain scenarios. These findings highlight the potential of synthetic
agricultural datasets and support hybrid strategies for more efficient model
training.

### 10. [From the Laboratory to Real-World Application: Evaluating Zero-Shot Scene Interpretation on Edge Devices for Mobile Robotics](http://arxiv.org/pdf/2511.02427v1)

Authors: Nicolas Schuler, Lea Dewald, Nick Baldig, Jürgen Graf

Video Understanding, Scene Interpretation and Commonsense Reasoning are
highly challenging tasks enabling the interpretation of visual information,
allowing agents to perceive, interact with and make rational decisions in its
environment. Large Language Models (LLMs) and Visual Language Models (VLMs)
have shown remarkable advancements in these areas in recent years, enabling
domain-specific applications as well as zero-shot open vocabulary tasks,
combining multiple domains. However, the required computational complexity
poses challenges for their application on edge devices and in the context of
Mobile Robotics, especially considering the trade-off between accuracy and
inference time. In this paper, we investigate the capabilities of
state-of-the-art VLMs for the task of Scene Interpretation and Action
Recognition, with special regard to small VLMs capable of being deployed to
edge devices in the context of Mobile Robotics. The proposed pipeline is
evaluated on a diverse dataset consisting of various real-world cityscape,
on-campus and indoor scenarios. The experimental evaluation discusses the
potential of these small models on edge devices, with particular emphasis on
challenges, weaknesses, inherent model biases and the application of the gained
information. Supplementary material is provided via the following repository:
https://datahub.rz.rptu.de/hstr-csrl-public/publications/scene-interpretation-on-edge-devices/

### Software Engineering

### 1. [LLMs as Judges: Toward The Automatic Review of GSN-compliant Assurance Cases](http://arxiv.org/pdf/2511.02203v1)

Authors: Gerhard Yu, Mithila Sivakumar, Alvine B. Belle, Soude Ghari, Song Wang, Timothy C. Lethbridge

Assurance cases allow verifying the correct implementation of certain
non-functional requirements of mission-critical systems, including their
safety, security, and reliability. They can be used in the specification of
autonomous driving, avionics, air traffic control, and similar systems. They
aim to reduce risks of harm of all kinds including human mortality,
environmental damage, and financial loss. However, assurance cases often tend
to be organized as extensive documents spanning hundreds of pages, making their
creation, review, and maintenance error-prone, time-consuming, and tedious.
Therefore, there is a growing need to leverage (semi-)automated techniques,
such as those powered by generative AI and large language models (LLMs), to
enhance efficiency, consistency, and accuracy across the entire assurance-case
lifecycle. In this paper, we focus on assurance case review, a critical task
that ensures the quality of assurance cases and therefore fosters their
acceptance by regulatory authorities. We propose a novel approach that
leverages the \textit{LLM-as-a-judge} paradigm to automate the review process.
Specifically, we propose new predicate-based rules that formalize
well-established assurance case review criteria, allowing us to craft LLM
prompts tailored to the review task. Our experiments on several
state-of-the-art LLMs (GPT-4o, GPT-4.1, DeepSeek-R1, and Gemini 2.0 Flash) show
that, while most LLMs yield relatively good review capabilities, DeepSeek-R1
and GPT-4.1 demonstrate superior performance, with DeepSeek-R1 ultimately
outperforming GPT-4.1. However, our experimental results also suggest that
human reviewers are still needed to refine the reviews LLMs yield.

### 2. [SWE-Sharp-Bench: A Reproducible Benchmark for C# Software Engineering Tasks](http://arxiv.org/pdf/2511.02352v1)

Authors: Sanket Mhatre, Yasharth Bajpai, Sumit Gulwani, Emerson Murphy-Hill, Gustavo Soares

AI coding agents have shown great progress on Python software engineering
benchmarks like SWE-Bench, and for other languages like Java and C in
benchmarks like Multi-SWE-Bench. However, C# -- a prominent enterprise language
ranking #5 in the TIOBE index -- remains absent from such benchmarks. We
introduce SWE-Sharp-Bench, a reproducible software engineering benchmark for
C\# featuring 150 instances from 17 repositories. Evaluating identical
model-agent configurations across languages reveals a significant performance
gap: while 70% of Python tasks in SWE-Bench Verified are solved, $only 40% of
our C\# tasks are resolved. We open-source SWE-Sharp-Bench and our entire
curation pipeline.

### 3. [Who's Who? LLM-assisted Software Traceability with Architecture Entity Recognition](http://arxiv.org/pdf/2511.02434v1)

Authors: Dominik Fuchß, Haoyu Liu, Sophie Corallo, Tobias Hey, Jan Keim, Johannes von Geisau, Anne Koziolek

Identifying architecturally relevant entities in textual artifacts is crucial
for Traceability Link Recovery (TLR) between Software Architecture
Documentation (SAD) and source code. While Software Architecture Models (SAMs)
can bridge the semantic gap between these artifacts, their manual creation is
time-consuming. Large Language Models (LLMs) offer new capabilities for
extracting architectural entities from SAD and source code to construct SAMs
automatically or establish direct trace links. This paper presents two
LLM-based approaches: ExArch extracts component names as simple SAMs from SAD
and source code to eliminate the need for manual SAM creation, while ArTEMiS
identifies architectural entities in documentation and matches them with
(manually or automatically generated) SAM entities. Our evaluation compares
against state-of-the-art approaches SWATTR, TransArC and ArDoCode. TransArC
achieves strong performance (F1: 0.87) but requires manually created SAMs;
ExArch achieves comparable results (F1: 0.86) using only SAD and code. ArTEMiS
is on par with the traditional heuristic-based SWATTR (F1: 0.81) and can
successfully replace it when integrated with TransArC. The combination of
ArTEMiS and ExArch outperforms ArDoCode, the best baseline without manual SAMs.
Our results demonstrate that LLMs can effectively identify architectural
entities in textual artifacts, enabling automated SAM generation and TLR,
making architecture-code traceability more practical and accessible.

### 4. [When Continuous Delivery Is Not an Option: Practical Paths to Continuous Engineering in Complex Organizations](http://arxiv.org/pdf/2511.02445v1)

Authors: Eriks Klotins, Magnus Ahlgren, Nicolas Martin Vivaldi, Even-Andre Karlsson

Purpose: Continuous Software Engineering (CSE) promises improved efficiency,
quality, and responsiveness in software-intensive organizations. However, fully
adopting CSE is often constrained by complex products, legacy systems,
organizational inertia, and regulatory requirements. In this paper, we examine
four industrial cases from the automation, automotive, retail, and chemical
sectors to explore how such constraints shape CSE adoption in practice.
Methods: We apply and extend a previously proposed CSE Industry Readiness Model
to assess the current and potential levels of adoption in each case. Through
expert interviews and narrative synthesis, we identify common driving forces
and adoption barriers, including organizational preparedness,
cross-organizational dependencies, and limited customer demand for continuous
delivery. Results: Based on our findings, we propose an updated readiness model
that introduces additional levels of internal and external feedback,
distinguishes market- and organization-facing constraints, and better guides
practitioners in setting realistic CSE adoption goals. Conclusions: Our results
highlight that while full end-to-end CSE adoption may not always be feasible,
meaningful internal improvements are still possible and beneficial. This study
provides empirically grounded guidance for organizations navigating partial or
constrained CSE transformations.

### 5. [Lost in Code Generation: Reimagining the Role of Software Models in AI-driven Software Engineering](http://arxiv.org/pdf/2511.02475v1)

Authors: Jürgen Cito, Dominik Bork

Generative AI enables rapid ``vibe coding," where natural language prompts
yield working software systems. While this lowers barriers to software
creation, it also collapses the boundary between prototypes and engineered
software, leading to fragile systems that lack robustness, security, and
maintainability. We argue that this shift motivates a reimagining of software
models. Rather than serving only as upfront blueprints, models can be recovered
post-hoc from AI-generated code to restore comprehension, expose risks, and
guide refinement. In this role, models serve as mediators between human intent,
AI generation, and long-term system evolution, providing a path toward
sustainable AI-driven software engineering.

### 6. [ReleaseEval: A Benchmark for Evaluating Language Models in Automated Release Note Generation](http://arxiv.org/pdf/2511.02713v1)

Authors: Qianru Meng, Zhaochun Ren, Joost Visser

Automated release note generation addresses the challenge of documenting
frequent software updates, where manual efforts are time-consuming and prone to
human error. Although recent advances in language models further enhance this
process, progress remains hindered by dataset limitations, including the lack
of explicit licensing and limited reproducibility, and incomplete task design
that relies mainly on commit messages for summarization while overlooking
fine-grained contexts such as commit hierarchies and code changes. To fill this
gap, we introduce ReleaseEval, a reproducible and openly licensed benchmark
designed to systematically evaluate language models for automated release note
generation. ReleaseEval comprises 94,987 release notes from 3,369 repositories
across 6 programming languages, and supports three task settings with three
levels of input granularity: (1) commit2sum, which generates release notes from
commit messages; (2) tree2sum, which incorporates commit tree structures; and
(3) diff2sum, which leverages fine-grained code diffs. Both automated and human
evaluations show that large language models consistently outperform traditional
baselines across all tasks, achieving substantial gains on tree2sum, while
still struggling on diff2sum. These findings highlight LLMs' proficiency in
leveraging structured information while revealing challenges in abstracting
from long code diffs.

### 7. [Investigating the Experience of Autistic Individuals in Software Engineering](http://arxiv.org/pdf/2511.02736v1)

Authors: Madalena Sasportes, Grischa Liebel, Miguel Goulão

Context: Autism spectrum disorder (ASD) leads to various issues in the
everyday life of autistic individuals, often resulting in unemployment and
mental health problems. To improve the inclusion of autistic adults, existing
studies have highlighted the strengths these individuals possess in comparison
to non-autistic individuals, e.g., high attention to detail or excellent
logical reasoning skills. If fostered, these strengths could be valuable in
software engineering activities, such for identifying specific kinds of bugs in
code. However, existing work in SE has primarily studied the challenges of
autistic individuals and possible accommodations, with little attention their
strengths. Objective: Our goal is to analyse the experiences of autistic
individuals in software engineering activities, such as code reviews, with a
particular emphasis on strengths. Methods: This study combines Social-Technical
Grounded Theory through semi-structured interviews with 16 autistic software
engineers and a survey with 49 respondents, including 5 autistic participants.
We compare the emerging themes with the theory by Gama et al. on the Effect of
Neurodivergent Cognitive Dysfunctions in Software Engineering Performance.
Results: Our results suggest that autistic software engineers are often skilled
in logical thinking, attention to detail, and hyperfocus in programming; and
they enjoy learning new programming languages and programming-related
technologies. Confirming previous work, they tend to prefer written
communication and remote work. Finally, we report a high comfort level in
interacting with AI-based systems. Conclusions: Our findings extend existing
work by providing further evidence on the strengths of autistic software
engineers.

### 8. [Formalizing Regression Testing for Agile and Continuous Integration Environments](http://arxiv.org/pdf/2511.02810v1)

Authors: Suddhasvatta Das, Kevin Gary

Software developed using modern agile practices delivers a stream of software
versions that require continuous regression testing rather than testing once
close to the delivery or maintenance phase, as assumed by classical
regression-testing theory. In this work, we formalize the phenomenon of
continuous or near-continuous regression testing using successive builds as a
time-ordered chain, where each build contains the program, requirements, and
the accompanying tests. We also formalize the regression test window between
any two builds, which captures the limited time budget available for regression
testing. As the time limit is set to infinity and the chain is closed to two
builds, the model degenerates to retest-all, thereby preserving semantics for
the classical two-version case. The formalization is validated by directly
representing two state-of-the-art agile regression testing algorithms in terms
of build-tuple operations without requiring auxiliary assumptions, followed by
proof of the soundness and completeness of our formalization.

### 9. [From Code Changes to Quality Gains: An Empirical Study in Python ML Systems with PyQu](http://arxiv.org/pdf/2511.02827v1)

Authors: Mohamed Almukhtar, Anwar Ghammam, Marouane Kessentini, Hua Ming

In an era shaped by Generative Artificial Intelligence for code generation
and the rising adoption of Python-based Machine Learning systems (MLS),
software quality has emerged as a major concern. As these systems grow in
complexity and importance, a key obstacle lies in understanding exactly how
specific code changes affect overall quality-a shortfall aggravated by the lack
of quality assessment tools and a clear mapping between ML systems code changes
and their quality effects. Although prior work has explored code changes in
MLS, it mostly stops at what the changes are, leaving a gap in our knowledge of
the relationship between code changes and the MLS quality. To address this gap,
we conducted a large-scale empirical study of 3,340 open-source Python ML
projects, encompassing more than 3.7 million commits and 2.7 trillion lines of
code. We introduce PyQu, a novel tool that leverages low level software metrics
to identify quality-enhancing commits with an average accuracy, precision, and
recall of 0.84 and 0.85 of average F1 score. Using PyQu and a thematic
analysis, we identified 61 code changes, each demonstrating a direct impact on
enhancing software quality, and we classified them into 13 categories based on
contextual characteristics. 41% of the changes are newly discovered by our
study and have not been identified by state-of-the-art Python changes detection
tools. Our work offers a vital foundation for researchers, practitioners,
educators, and tool developers, advancing the quest for automated quality
assessment and best practices in Python-based ML software.

### 10. [Open the Oyster: Empirical Evaluation and Improvement of Code Reasoning Confidence in LLMs](http://arxiv.org/pdf/2511.02197v1)

Authors: Shufan Wang, Xing Hu, Junkai Chen, Zhiyuan Pan, Xin Xia

With the widespread application of large language models (LLMs) in the field
of code intelligence, increasing attention has been paid to the reliability and
controllability of their outputs in code reasoning tasks. Confidence estimation
serves as an effective and convenient approach for evaluating these aspects.
This paper proposes a confidence analysis and enhancement framework for LLMs
tailored to code reasoning tasks. We conduct a comprehensive empirical study on
the confidence reliability of mainstream LLMs across different tasks, and
further evaluate the effectiveness of techniques such as prompt strategy
optimisation and mathematical calibration (e.g., Platt Scaling) in improving
confidence reliability. Our results show that DeepSeek-Reasoner achieves the
best performance across various tasks, outperforming other models by up to
$0.680$, $0.636$, and $13.652$ in terms of ECE, Brier Score, and Performance
Score, respectively. The hybrid strategy combining the reassess prompt strategy
and Platt Scaling achieves improvements of up to $0.541$, $0.628$, and $15.084$
over the original performance in the aforementioned three metrics. These
results indicate that models with reasoning capabilities demonstrate superior
confidence reliability, and that the hybrid strategy is the most effective in
enhancing the confidence reliability of various models. Meanwhile, we elucidate
the impact of different task complexities, model scales, and strategies on
confidence performance, and highlight that the confidence of current LLMs in
complex reasoning tasks still has considerable room for improvement. This study
not only provides a research foundation and technical reference for the
application of confidence in LLM-assisted software engineering, but also points
the way for future optimisation and engineering deployment of confidence
mechanisms.

### Social and Information Networks

### 1. [Community Notes are Vulnerable to Rater Bias and Manipulation](http://arxiv.org/pdf/2511.02615v1)

Authors: Bao Tran Truong, Siqi Wu, Alessandro Flammini, Filippo Menczer, Alexander J. Stewart

Social media platforms increasingly rely on crowdsourced moderation systems
like Community Notes to combat misinformation at scale. However, these systems
face challenges from rater bias and potential manipulation, which may undermine
their effectiveness. Here we systematically evaluate the Community Notes
algorithm using simulated data that models realistic rater and note behaviors,
quantifying error rates in publishing helpful versus unhelpful notes. We find
that the algorithm suppresses a substantial fraction of genuinely helpful notes
and is highly sensitive to rater biases, including polarization and in-group
preferences. Moreover, a small minority (5--20\%) of bad raters can
strategically suppress targeted helpful notes, effectively censoring reliable
information. These findings suggest that while community-driven moderation may
offer scalability, its vulnerability to bias and manipulation raises concerns
about reliability and trustworthiness, highlighting the need for improved
mechanisms to safeguard the integrity of crowdsourced fact-checking.

### 2. [Feedback dynamics in Politics: The interplay between sentiment and engagement](http://arxiv.org/pdf/2511.02663v1)

Authors: Simone Formentin

We investigate feedback mechanisms in political communication by testing
whether politicians adapt the sentiment of their messages in response to public
engagement. Using over 1.5 million tweets from Members of Parliament in the
United Kingdom, Spain, and Greece during 2021, we identify sentiment dynamics
through a simple yet interpretable linear model. The analysis reveals a
closed-loop behavior: engagement with positive and negative messages influences
the sentiment of subsequent posts. Moreover, the learned coefficients highlight
systematic differences across political roles: opposition members are more
reactive to negative engagement, whereas government officials respond more to
positive signals. These results provide a quantitative, control-oriented view
of behavioral adaptation in online politics, showing how feedback principles
can explain the self-reinforcing dynamics that emerge in social media
discourse.

### Systems and Control

### 1. [Online Distributed Zeroth-Order Optimization With Non-Zero-Mean Adverse Noises](http://arxiv.org/pdf/2511.02183v1)

Authors: Yanfu Qin, Kaihong Lu

In this paper, the problem of online distributed zeroth-order optimization
subject to a set constraint is studied via a multi-agent network, where each
agent can communicate with its immediate neighbors via a time-varying directed
graph. Different from the existing works on online distributed zeroth- order
optimization, we consider the case where the estimate on the gradients are
influenced by some non-zero-mean adverse noises. To handle this problem, we
propose a new online dis- tributed zeroth-order mirror descent algorithm
involving a kernel function-based estimator and a clipped strategy.
Particularly, in the estimator, the kernel function-based strategy is provided
to deal with the adverse noises, and eliminate the low-order terms in the
Taylor expansions of the objective functions. Furthermore, the performance of
the presented algorithm is measured by employing the dynamic regrets, where the
offline benchmarks are to find the optimal point at each time. Under the mild
assumptions on the graph and the objective functions, we prove that if the
variation in the optimal point sequence grows at a certain rate, then the high
probability bound of the dynamic regrets increases sublinearly. Finally, a
simulation experiment is worked out to demonstrate the effectiveness of our
theoretical results.

### 2. [A Reliability-Cost Optimization Framework for EV and DER Integration in Standard and Reconfigurable Distribution Network Topologies](http://arxiv.org/pdf/2511.02250v1)

Authors: Rida Fatima, Linhan Fang, Xingpeng Li

The rapid growth of electric vehicle (EV) adoption poses operational and
economic challenges for power distribution systems, including increased line
loading levels and network congestions. This may require potential
infrastructure reinforcement and expansion. As a fast inexpensive alternative
solution, network topology reconfiguration (NTR) offers a practical means to
redistribute power flows, reduce operational costs, and defer infrastructure
upgrades. This paper presents a linear programming framework to evaluate the
impact of varying EV penetration on operational costs under four
configurations: standard distribution network (SDN), SDN with NTR (SDNTR), SDN
with distributed energy resources (SDN-DER), and SDNTR with DERs (SDNTR-DER).
Numerical simulations are conducted on the IEEE 33-bus system. The analysis
demonstrates that integrating DERs reduces operational costs, while NTR further
enhances system flexibility, enabling higher EV penetration levels without
compromising feasibility. The combined SDNTR-DER approach offers the most
cost-effective and reliable pathway for accommodating future EV growth while
mitigating the need for immediate infrastructure upgrades.

### 3. [Constrained Performance Boosting Control for Nonlinear Systems via ADMM](http://arxiv.org/pdf/2511.02389v1)

Authors: Gianluca Giacomelli, Danilo Saccani, Siep Weiland, Giancarlo Ferrari-Trecate, Valentina Breschi

We present the Alternating Direction Method of Multipliers for Performance
Boosting (ADMM-PB), an approach to design performance boosting controllers for
stable or pre-stabilized nonlinear systems, while explicitly seeking input and
state constraint satisfaction. Rooted on a recently proposed approach for
designing neural-network controllers that guarantees closed-loop stability by
design while minimizing generic cost functions, our strategy integrates it
within an alternating direction method of multipliers routine to seek
constraint handling without modifying the controller structure of the
aforementioned seminal strategy. Our numerical results showcase the advantages
of the proposed approach over a baseline penalizing constraint violation
through barrier-like terms in the cost, indicating that ADMM-PB can lead to
considerably lower constraint violations at the price of inducing slightly more
cautious closed-loop behaviors.

### 4. [Decentralized Voltage Control of AC Microgrids with Constant Power Loads using Control Barrier Functions](http://arxiv.org/pdf/2511.02438v1)

Authors: Grigoris Michos, George C. Konstantopoulos

This paper proposes a novel nonlinear decentralized voltage controller for
constrained regulation of meshed AC Microgrid networks with high penetration of
constant power loads. Perceiving the load demand as an unknown disturbance, the
network model is reformulated in a cascaded structure composed of a nominal,
i.e. uncertainty-free, and an error subsystem. The latter captures the distance
between the true and the nominal state trajectories, for which we prove
boundedness via a suitable control barrier function. Under sufficient
conditions, we prove asymptotic stability of the cascaded dynamics with respect
to an equilibrium set and also provide an estimate of the region of attraction.
In addition, it is rigorously shown that the proposed nonlinear control law
also enforces constrained regulation around a rated voltage value, without the
need of saturation devices. The operation of the closed-loop system is
illustrated in a simulation scenario, demonstrating bounded operation and
convergence to a neighbourhood of the desired reference vector.

### 5. [Generalized Swing Control Framework for Inverter-based Resources](http://arxiv.org/pdf/2511.02482v1)

Authors: Rodrigo Bernal, Federico Milano

This paper proposes a novel control framework designed for Inverter-Based
Resources (IBRs), denoted as Generalized Swing Control (GSC). The proposed GSC
framework generalizes the definition of Grid-Forming (GFM) control schemes and
exploits the coupling between active and reactive power dynamics. To validate
the proposed scheme, we conduct extensive time-domain simulations and
small-signal analysis using a modified version of the WSCC 9-bus system and a
1479-bus dynamic model of the all-island Irish transmission system. The case
studies focus on evaluating the dynamic performance of the proposed framework
under different configurations, including Virtual Synchronous Machine (VSM),
coupled-VSM and dual-VSM schemes. To address the nonlinear nature of power
system dynamics, sensitivity analysis based on Monte Carlo methods are employed
to improve parameter tuning and assess the stability of GSC configurations in
the studied systems.

### 6. [Coherency among Power System Devices](http://arxiv.org/pdf/2511.02486v1)

Authors: Ignacio Ponce, Rodrigo Bernal, Federico Milano

The paper proposes a novel general definition of coherency among power system
devices of any type. The proposed approach is thus not limited to synchronous
machines. With this aim, the paper shows that coherency can be formally based
on the difference in the complex frequency of the current injections of any two
devices electrically connected to the same grid. The proposed definition is
model-agnostic, making it general and suitable for modern power systems
composed of a heterogeneous mix of technologies. The paper also provides a
systematic analytical procedure to study the properties that specific device
models must satisfy to be coherent. Time-domain simulations are conducted in
three case studies whose results illustrate the ability of our definition to
evaluate coherency among any type of device.

### 7. [Decentralized Approach to Detect and Eliminate Flapping Phenomena due to Flexible Resources](http://arxiv.org/pdf/2511.02497v1)

Authors: Angel Vaca, Federico Milano

This paper presents a decentralized methodology for detecting and mitigating
flapping phenomena in power systems, primarily caused by the operation of
discrete devices. The proposed approach applies moving-window autocorrelation
to local measurements, enabling each device to autonomously identify sustained
oscillations. Upon detection, a probabilistic, device-specific mitigation
strategy is executed. Flexible demand resources (DFRs), under-load tap changers
(ULTCs), and automatic voltage regulators (AVRs) are utilised to illustrate the
performance of the proposed approach to both discrete and continuous-operation
devices. Results show that the proposed method is robust and properly
distinguishes damped oscillations from persistent flapping, allowing devices to
independently recognize problematic operating scenarios and implement
corrective actions accordingly.

### 8. [Reliability entails input-selective contraction and regulation in excitable networks](http://arxiv.org/pdf/2511.02554v1)

Authors: Michelangelo Bin, Alessandro Cecconi, Lorenzo Marconi

The animal nervous system offers a model of computation combining digital
reliability and analog efficiency. Understanding how this sweet spot can be
realized is a core question of neuromorphic engineering. To this aim, this
paper explores the connection between reliability, contraction, and regulation
in excitable systems. Using the FitzHugh-Nagumo model of excitable behavior as
a proof-of-concept, it is shown that neuronal reliability can be formalized as
an average trajectory contraction property induced by the input. In excitable
networks, reliability is shown to enable regulation of the network to a
robustly stable steady state. It is thus posited that regulation provides a
notion of dynamical analog computation, and that stability makes such a
computation model robust.

### 9. [Analytical Framework for Assessing Effective Regional Inertia](http://arxiv.org/pdf/2511.02574v1)

Authors: Bruno Pinheiro, Joe H. Chow, Federico Milano, Daniel Dotta

This paper proposes a novel formulation of effective regional inertia that
explicitly accounts for both system topology and the spatial distribution of
inertia. Unlike traditional approaches that model a region as an aggregated
machine with an equivalent inertia, the proposed metric provides a
topology-aware representation. The methodology builds on an analytical
framework that extends classical slow coherency theory to address network
partitioning and regional frequency stability. Based on these partitions, we
develop a systematic procedure to evaluate the effective inertia of each
region, enabling a more accurate interpretation of local inertial
contributions, including those from virtual inertia provided by inverter-based
resources (IBRs). Case studies on the IEEE 39-bus and 68-bus systems
demonstrate that the integration of inertial devices does not uniformly improve
system frequency response, underscoring the importance of the proposed metric
for effective regional inertia assessment.

### 10. [ISAC Empowered Air-Sea Collaborative System: A UAV-USV Joint Inspection Framework](http://arxiv.org/pdf/2511.02592v1)

Authors: Rui Zhang, Fuwang Dong, Wei Wang

In this paper, we construct an air-sea collaborative system framework based
on the Integrated Sensing and Communication (ISAC) techniques, where the
Unmanned Aerial Vehicle (UAV) and Unmanned Surface Vehicle (USV) jointly
inspect targets of interest while keeping communication with each other
simultaneously. First, we demonstrate the unique challenges encountered in this
collaborative system, i.e., the coupling and heterogeneity of the UAV/USV's
trajectories. Then, we formulate a total energy consumption minimization
problem to jointly optimize the trajectories, flying and hovering times, target
scheduling, and beamformers under the constraints of water currents, collision
avoidance, and Sensing and Communication (S\&C) requirements. To address the
strong coupling of the variables, we divide the original problem into two
subproblems, namely, the hover point selection and the joint trajectory
planning and beamforming design. In the first subproblem, we propose a
three-step hierarchical method including: (1) a virtual base station coverage
(VBSC) and clustering algorithm to obtain the target scheduling and rough
position of hover points; (2) a Bi-traveling salesman problem with neighborhood
(Bi-TSPN)-based algorithm to determine the visiting order sequence of the hover
points; (3) a hover point refinement and time allocation algorithm to further
optimize the time allocation. In the latter subproblem, we complete the
remaining trajectory planning and beamforming design in each flying and
hovering stage by developing a semi-definite relaxation (SDR) and successive
convex approximation (SCA) method. Finally, we conduct a series of simulations
to demonstrate the superiority of the proposed scheme over existing sequential
access and leader-follower strategies.

### Machine Learning (Statistics Category)

### 1. [An Adaptive Sampling Framework for Detecting Localized Concept Drift under Label Scarcity](http://arxiv.org/pdf/2511.02452v1)

Authors: Junghee Pyeon, Davide Cacciarelli, Kamran Paynabar

Concept drift and label scarcity are two critical challenges limiting the
robustness of predictive models in dynamic industrial environments. Existing
drift detection methods often assume global shifts and rely on dense
supervision, making them ill-suited for regression tasks with local drifts and
limited labels. This paper proposes an adaptive sampling framework that
combines residual-based exploration and exploitation with EWMA monitoring to
efficiently detect local concept drift under labeling budget constraints.
Empirical results on synthetic benchmarks and a case study on electricity
market demonstrate superior performance in label efficiency and drift detection
accuracy.

### 2. [DoFlow: Causal Generative Flows for Interventional and Counterfactual Time-Series Prediction](http://arxiv.org/pdf/2511.02137v1)

Authors: Dongze Wu, Feng Qiu, Yao Xie

Time-series forecasting increasingly demands not only accurate observational
predictions but also causal forecasting under interventional and counterfactual
queries in multivariate systems. We present DoFlow, a flow based generative
model defined over a causal DAG that delivers coherent observational and
interventional predictions, as well as counterfactuals through the natural
encoding and decoding mechanism of continuous normalizing flows (CNFs). We also
provide a supporting counterfactual recovery result under certain assumptions.
Beyond forecasting, DoFlow provides explicit likelihoods of future
trajectories, enabling principled anomaly detection. Experiments on synthetic
datasets with various causal DAG and real world hydropower and cancer treatment
time series show that DoFlow achieves accurate system-wide observational
forecasting, enables causal forecasting over interventional and counterfactual
queries, and effectively detects anomalies. This work contributes to the
broader goal of unifying causal reasoning and generative modeling for complex
dynamical systems.

### 3. [Probabilistic Graph Cuts](http://arxiv.org/pdf/2511.02272v1)

Authors: Ayoub Ghriss

Probabilistic relaxations of graph cuts offer a differentiable alternative to
spectral clustering, enabling end-to-end and online learning without
eigendecompositions, yet prior work centered on RatioCut and lacked general
guarantees and principled gradients. We present a unified probabilistic
framework that covers a wide class of cuts, including Normalized Cut. Our
framework provides tight analytic upper bounds on expected discrete cuts via
integral representations and Gauss hypergeometric functions with closed-form
forward and backward. Together, these results deliver a rigorous, numerically
stable foundation for scalable, differentiable graph partitioning covering a
wide range of clustering and contrastive learning objectives.

### 4. [A Stable Lasso](http://arxiv.org/pdf/2511.02306v1)

Authors: Mahdi Nouraie, Houying Zhu, Samuel Muller

The Lasso has been widely used as a method for variable selection, valued for
its simplicity and empirical performance. However, Lasso's selection stability
deteriorates in the presence of correlated predictors. Several approaches have
been developed to mitigate this limitation. In this paper, we provide a brief
review of existing approaches, highlighting their limitations. We then propose
a simple technique to improve the selection stability of Lasso by integrating a
weighting scheme into the Lasso penalty function, where the weights are defined
as an increasing function of a correlation-adjusted ranking that reflects the
predictive power of predictors. Empirical evaluations on both simulated and
real-world datasets demonstrate the efficacy of the proposed method. Additional
numerical results demonstrate the effectiveness of the proposed approach in
stabilizing other regularization-based selection methods, indicating its
potential as a general-purpose solution.

### 5. [Reducing normalizing flow complexity for MCMC preconditioning](http://arxiv.org/pdf/2511.02345v1)

Authors: David Nabergoj, Erik Štrumbelj

Preconditioning is a key component of MCMC algorithms that improves sampling
efficiency by facilitating exploration of geometrically complex target
distributions through an invertible map. While linear preconditioners are often
sufficient for moderately complex target distributions, recent work has
explored nonlinear preconditioning with invertible neural networks as
components of normalizing flows (NFs). However, empirical and theoretical
studies show that overparameterized NF preconditioners can degrade sampling
efficiency and fit quality. Moreover, existing NF-based approaches do not adapt
their architectures to the target distribution. Related work outside of MCMC
similarly finds that suitably parameterized NFs can achieve comparable or
superior performance with substantially less training time or data. We propose
a factorized preconditioning architecture that reduces NF complexity by
combining a linear component with a conditional NF, improving adaptability to
target geometry. The linear preconditioner is applied to dimensions that are
approximately Gaussian, as estimated from warmup samples, while the conditional
NF models more complex dimensions. Our method yields significantly better tail
samples on two complex synthetic distributions and consistently better
performance on a sparse logistic regression posterior across varying likelihood
and prior strengths. It also achieves higher effective sample sizes on
hierarchical Bayesian model posteriors with weak likelihoods and strong funnel
geometries. This approach is particularly relevant for hierarchical Bayesian
model analyses with limited data and could inform current theoretical and
software strides in neural MCMC design.

### 6. [A new class of Markov random fields enabling lightweight sampling](http://arxiv.org/pdf/2511.02373v1)

Authors: Jean-Baptiste Courbot, Hugo Gangloff, Bruno Colicchio

This work addresses the problem of efficient sampling of Markov random fields
(MRF). The sampling of Potts or Ising MRF is most often based on Gibbs
sampling, and is thus computationally expensive. We consider in this work how
to circumvent this bottleneck through a link with Gaussian Markov Random
fields. The latter can be sampled in several cost-effective ways, and we
introduce a mapping from real-valued GMRF to discrete-valued MRF. The resulting
new class of MRF benefits from a few theoretical properties that validate the
new model. Numerical results show the drastic performance gain in terms of
computational efficiency, as we sample at least 35x faster than Gibbs sampling
using at least 37x less energy, all the while exhibiting empirical properties
close to classical MRFs.

### 7. [Wasserstein Convergence of Critically Damped Langevin Diffusions](http://arxiv.org/pdf/2511.02419v1)

Authors: Stanislas Strasman, Sobihan Surendran, Claire Boyer, Sylvain Le Corff, Vincent Lemaire, Antonio Ocello

Score-based Generative Models (SGMs) have achieved impressive performance in
data generation across a wide range of applications and benefit from strong
theoretical guarantees. Recently, methods inspired by statistical mechanics, in
particular, Hamiltonian dynamics, have introduced Critically-damped Langevin
Diffusions (CLDs), which define diffusion processes on extended spaces by
coupling the data with auxiliary variables. These approaches, along with their
associated score-matching and sampling procedures, have been shown to
outperform standard diffusion-based samplers numerically. In this paper, we
analyze a generalized dynamic that extends classical CLDs by introducing an
additional hyperparameter controlling the noise applied to the data coordinate,
thereby better exploiting the extended space. We further derive a novel upper
bound on the sampling error of CLD-based generative models in the Wasserstein
metric. This additional hyperparameter influences the smoothness of sample
paths, and our discretization error analysis provides practical guidance for
its tuning, leading to improved sampling performance.

### 8. [Efficient Solvers for SLOPE in R, Python, Julia, and C++](http://arxiv.org/pdf/2511.02430v1)

Authors: Johan Larsson, Malgorzata Bogdan, Krystyna Grzesiak, Mathurin Massias, Jonas Wallin

We present a suite of packages in R, Python, Julia, and C++ that efficiently
solve the Sorted L-One Penalized Estimation (SLOPE) problem. The packages
feature a highly efficient hybrid coordinate descent algorithm that fits
generalized linear models (GLMs) and supports a variety of loss functions,
including Gaussian, binomial, Poisson, and multinomial logistic regression. Our
implementation is designed to be fast, memory-efficient, and flexible. The
packages support a variety of data structures (dense, sparse, and out-of-memory
matrices) and are designed to efficiently fit the full SLOPE path as well as
handle cross-validation of SLOPE models, including the relaxed SLOPE. We
present examples of how to use the packages and benchmarks that demonstrate the
performance of the packages on both real and simulated data and show that our
packages outperform existing implementations of SLOPE in terms of speed.

### 9. [Learning CNF formulas from uniform random solutions in the local lemma regime](http://arxiv.org/pdf/2511.02487v1)

Authors: Weiming Feng, Xiongxin Yang, Yixiao Yu, Yiyao Zhang

We study the problem of learning a $n$-variables $k$-CNF formula $\Phi$ from
its i.i.d. uniform random solutions, which is equivalent to learning a Boolean
Markov random field (MRF) with $k$-wise hard constraints. Revisiting Valiant's
algorithm (Commun. ACM'84), we show that it can exactly learn (1) $k$-CNFs with
bounded clause intersection size under Lov\'asz local lemma type conditions,
from $O(\log n)$ samples; and (2) random $k$-CNFs near the satisfiability
threshold, from $\widetilde{O}(n^{\exp(-\sqrt{k})})$ samples. These results
significantly improve the previous $O(n^k)$ sample complexity. We further
establish new information-theoretic lower bounds on sample complexity for both
exact and approximate learning from i.i.d. uniform random solutions.

### 10. [ConMeZO: Adaptive Descent-Direction Sampling for Gradient-Free Finetuning of Large Language Models](http://arxiv.org/pdf/2511.02757v1)

Authors: Lejs Deen Behric, Liang Zhang, Bingcong Li, Kiran Koshy Thekumparampil

Zeroth-order or derivative-free optimization (MeZO) is an attractive strategy
for finetuning large language models (LLMs) because it eliminates the memory
overhead of backpropagation. However, it converges slowly due to the inherent
curse of dimensionality when searching for descent directions in the
high-dimensional parameter space of billion-scale LLMs. We propose ConMeZO, a
novel zeroth-order optimizer that accelerates convergence by adaptive
directional sampling. Instead of drawing the direction uniformly at random,
ConMeZO restricts the sampling to a cone centered around a momentum estimate.
This concentrates the search in directions where the true gradient is more
likely to lie and thus reduces the effect of high dimensions. We prove that
ConMeZO achieves the same worst-case convergence rate as MeZO. Empirically,
when finetuning LLMs on natural language tasks, ConMeZO is up to 2X faster than
MeZO while retaining the low-memory footprint of zeroth-order methods.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-11-05 PST.

### 1. [Automatic diagnosis of heating in oil-filled terminals of cables](https://www.nature.com/articles/s41598-025-22506-0)

Authors: Shunyu Yao et al.

### 2. [Images for AI use can be sourced responsibly](https://www.nature.com/articles/d41586-025-03568-6)

Authors: 

### 3. [Connected, digitalized wire arc additive manufacturing: utilizing data in the internet of production to enable industrie 4.0](https://www.nature.com/articles/s41598-025-15250-y)

Authors: Samuel Mann et al.

### 4. [Multimedia data-driven customer churn prediction using an enhanced extreme learning machine](https://www.nature.com/articles/s41598-025-22564-4)

Authors: You-wu Liu et al.

### 5. [Fair human-centric image dataset for ethical AI benchmarking](https://www.nature.com/articles/s41586-025-09716-2)

Authors: Alice Xiang et al.

### 6. [Fair-efficient allocation mechanism with meta-types resources in cloud computing](https://www.nature.com/articles/s41598-025-22657-0)

Authors: Fengyue Zhang et al.

### 7. [Assessment of monocular human pose estimation models for clinical movement analysis](https://www.nature.com/articles/s41598-025-22626-7)

Authors: David Rode et al.

### 8. [Leveraging ChatGPT and explainable AI for enhancing clinical decision support](https://www.nature.com/articles/s41598-025-22784-8)

Authors: Radwa El Shawi et al.

