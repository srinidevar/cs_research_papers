# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-08-14 17:00:25.340414 PST.

### Artificial Intelligence

### 1. [An Automated Multi-Modal Evaluation Framework for Mobile Intelligent Assistants](http://arxiv.org/pdf/2508.09507v1)

Authors: Meiping Wang, Jian Zhong, Rongduo Han, Liming Kang, Zhengkun Shi, Xiao Liang, Xing Lin, Nan Gao, Haining Zhang

With the rapid development of mobile intelligent assistant technologies,
multi-modal AI assistants have become essential interfaces for daily user
interactions. However, current evaluation methods face challenges including
high manual costs, inconsistent standards, and subjective bias. This paper
proposes an automated multi-modal evaluation framework based on large language
models and multi-agent collaboration. The framework employs a three-tier agent
architecture consisting of interaction evaluation agents, semantic verification
agents, and experience decision agents. Through supervised fine-tuning on the
Qwen3-8B model, we achieve a significant evaluation matching accuracy with
human experts. Experimental results on eight major intelligent agents
demonstrate the framework's effectiveness in predicting users' satisfaction and
identifying generation defects.

### 2. [EvoCurr: Self-evolving Curriculum with Behavior Code Generation for Complex Decision-making](http://arxiv.org/pdf/2508.09586v1)

Authors: Yang Cheng, Zilai Wang, Weiyu Ma, Wenhui Zhu, Yue Deng, Jian Zhao

Large Language Models (LLMs) have demonstrated remarkable capabilities across
diverse domains, including programming, planning, and decision-making. However,
their performance often degrades when faced with highly complex problem
instances that require deep reasoning over long horizons. In such cases, direct
problem-solving approaches can lead to inefficiency or failure due to the lack
of structured intermediate guidance. To address this, we propose a novel
self-evolve framework, EvoCurr, in which a dedicated curriculum-generation LLM
constructs a sequence of problem instances with gradually increasing
difficulty, tailored to the solver LLM's learning progress. The curriculum
dynamically adapts easing challenges when the solver struggles and escalating
them when success is consistent, thus maintaining an optimal learning
trajectory. This approach enables the solver LLM, implemented as a
code-generation model producing Python decision-tree scripts, to progressively
acquire the skills needed for complex decision-making tasks. Experimental
results on challenging decision-making benchmarks show that our method
significantly improves task success rates and solution efficiency compared to
direct-solving baselines. These findings suggest that LLM-driven curriculum
learning holds strong potential for enhancing automated reasoning in
real-world, high-complexity domains.

### 3. [UbiQTree: Uncertainty Quantification in XAI with Tree Ensembles](http://arxiv.org/pdf/2508.09639v1)

Authors: Akshat Dubey, Aleksandar Anžel, Bahar İlgen, Georges Hattab

Explainable Artificial Intelligence (XAI) techniques, such as SHapley
Additive exPlanations (SHAP), have become essential tools for interpreting
complex ensemble tree-based models, especially in high-stakes domains such as
healthcare analytics. However, SHAP values are usually treated as point
estimates, which disregards the inherent and ubiquitous uncertainty in
predictive models and data. This uncertainty has two primary sources: aleatoric
and epistemic. The aleatoric uncertainty, which reflects the irreducible noise
in the data. The epistemic uncertainty, which arises from a lack of data. In
this work, we propose an approach for decomposing uncertainty in SHAP values
into aleatoric, epistemic, and entanglement components. This approach
integrates Dempster-Shafer evidence theory and hypothesis sampling via
Dirichlet processes over tree ensembles. We validate the method across three
real-world use cases with descriptive statistical analyses that provide insight
into the nature of epistemic uncertainty embedded in SHAP explanations. The
experimentations enable to provide more comprehensive understanding of the
reliability and interpretability of SHAP-based attributions. This understanding
can guide the development of robust decision-making processes and the
refinement of models in high-stakes applications. Through our experiments with
multiple datasets, we concluded that features with the highest SHAP values are
not necessarily the most stable. This epistemic uncertainty can be reduced
through better, more representative data and following appropriate or
case-desired model development techniques. Tree-based models, especially
bagging, facilitate the effective quantification of epistemic uncertainty.

### 4. [MEML-GRPO: Heterogeneous Multi-Expert Mutual Learning for RLVR Advancement](http://arxiv.org/pdf/2508.09670v1)

Authors: Weitao Jia, Jinghui Lu, Haiyang Yu, Siqi Wang, Guozhi Tang, An-Lan Wang, Weijie Yin, Dingkang Yang, Yuxiang Nie, Bin Shan, Hao Feng, Irene Li, Kun Yang, Han Wang, Jingqun Tang, Teng Fu, Changhong Jin, Chao Feng, Xiaohui Lv, Can Huang

Recent advances demonstrate that reinforcement learning with verifiable
rewards (RLVR) significantly enhances the reasoning capabilities of large
language models (LLMs). However, standard RLVR faces challenges with reward
sparsity, where zero rewards from consistently incorrect candidate answers
provide no learning signal, particularly in challenging tasks. To address this,
we propose Multi-Expert Mutual Learning GRPO (MEML-GRPO), an innovative
framework that utilizes diverse expert prompts as system prompts to generate a
broader range of responses, substantially increasing the likelihood of
identifying correct solutions. Additionally, we introduce an inter-expert
mutual learning mechanism that facilitates knowledge sharing and transfer among
experts, further boosting the model's performance through RLVR. Extensive
experiments across multiple reasoning benchmarks show that MEML-GRPO delivers
significant improvements, achieving an average performance gain of 4.89% with
Qwen and 11.33% with Llama, effectively overcoming the core limitations of
traditional RLVR methods.

### 5. [UDA: Unsupervised Debiasing Alignment for Pair-wise LLM-as-a-Judge](http://arxiv.org/pdf/2508.09724v1)

Authors: Yang Zhang, Cunxiang Wang, Lindong Wu, Wenbo Yu, Yidong Wang, Guangsheng Bao, Jie Tang

Pairwise evaluation of Large Language Models (LLMs) is a common paradigm, but
it is prone to preference bias, where judges systematically favor certain
outputs, such as their own. This bias leads to inconsistent and skewed rankings
across different judges. To address this, we first empirically demonstrate
significant and heterogeneous biases in cross-model evaluations. We then
propose UDA (Unsupervised Debiasing Alignment), a framework that reduces
inter-judge disagreement by dynamically adjusting the Elo rating system. For
each pairwise comparison, a compact neural network learns to adaptively set the
K-factor and refine win probabilities. Crucially, UDA operates in a fully
unsupervised manner, guided solely by the objective of minimizing the
dispersion among the Elo trajectories of all judges. This forces an alignment
towards a collective consensus, which serves as an unsupervised proxy for a
more stable and reproducible evaluation. In addition, we provide theoretical
motivation demonstrating how alignment towards a consensus can reduce aggregate
system bias. Experiments show that UDA significantly reduces the inter-judge
rating standard deviation by up to 63.4% and improves the average correlation
with human judgments by 24.7%. Notably, UDA elevates the performance of poorly
performing judges to achieve parity with high-quality ones, fostering a more
robust and reliable evaluation ecosystem. Code and data are available at
https://anonymous.4open.science/r/62AB93CD-23B4.

### 6. [Human-Aligned Procedural Level Generation Reinforcement Learning via Text-Level-Sketch Shared Representation](http://arxiv.org/pdf/2508.09860v1)

Authors: In-Chang Baek, Seoyoung Lee, Sung-Hyun Kim, Geumhwan Hwang, KyungJoong Kim

Human-aligned AI is a critical component of co-creativity, as it enables
models to accurately interpret human intent and generate controllable outputs
that align with design goals in collaborative content creation. This direction
is especially relevant in procedural content generation via reinforcement
learning (PCGRL), which is intended to serve as a tool for human designers.
However, existing systems often fall short of exhibiting human-centered
behavior, limiting the practical utility of AI-driven generation tools in
real-world design workflows. In this paper, we propose VIPCGRL
(Vision-Instruction PCGRL), a novel deep reinforcement learning framework that
incorporates three modalities-text, level, and sketches-to extend control
modality and enhance human-likeness. We introduce a shared embedding space
trained via quadruple contrastive learning across modalities and human-AI
styles, and align the policy using an auxiliary reward based on embedding
similarity. Experimental results show that VIPCGRL outperforms existing
baselines in human-likeness, as validated by both quantitative metrics and
human evaluations. The code and dataset will be available upon publication.

### 7. [AWorld: Dynamic Multi-Agent System with Stable Maneuvering for Robust GAIA Problem Solving](http://arxiv.org/pdf/2508.09889v1)

Authors: Zhitian Xie, Qintong Wu, Chengyue Yu, Chenyi Zhuang, Jinjie Gu

The rapid advancement of large language models (LLMs) has empowered
intelligent agents to leverage diverse external tools for solving complex
real-world problems. However, as agents increasingly depend on multiple tools,
they encounter new challenges: extended contexts from disparate sources and
noisy or irrelevant tool outputs can undermine system reliability and accuracy.
These challenges underscore the necessity for enhanced stability in agent-based
systems. To address this, we introduce dynamic supervision and maneuvering
mechanisms, constructing a robust and dynamic Multi-Agent System (MAS)
architecture within the AWorld framework. In our approach, the Execution Agent
invokes the Guard Agent at critical steps to verify and correct the reasoning
process, effectively reducing errors arising from noise and bolstering
problem-solving robustness. Extensive experiments on the GAIA test dataset
reveal that our dynamic maneuvering mechanism significantly improves both the
effectiveness and stability of solutions, outperforming single-agent system
(SAS) and standard tool-augmented systems. As a result, our dynamic MAS system
achieved first place among open-source projects on the prestigious GAIA
leaderboard. These findings highlight the practical value of collaborative
agent roles in developing more reliable and trustworthy intelligent systems.

### 8. [RAGulating Compliance: A Multi-Agent Knowledge Graph for Regulatory QA](http://arxiv.org/pdf/2508.09893v1)

Authors: Bhavik Agarwal, Hemant Sunil Jomraj, Simone Kaplunov, Jack Krolick, Viktoria Rojkova

Regulatory compliance question answering (QA) requires precise, verifiable
information, and domain-specific expertise, posing challenges for Large
Language Models (LLMs). In this work, we present a novel multi-agent framework
that integrates a Knowledge Graph (KG) of Regulatory triplets with
Retrieval-Augmented Generation (RAG) to address these demands. First, agents
build and maintain an ontology-free KG by extracting subject--predicate--object
(SPO) triplets from regulatory documents and systematically cleaning,
normalizing, deduplicating, and updating them. Second, these triplets are
embedded and stored along with their corresponding textual sections and
metadata in a single enriched vector database, allowing for both graph-based
reasoning and efficient information retrieval. Third, an orchestrated agent
pipeline leverages triplet-level retrieval for question answering, ensuring
high semantic alignment between user queries and the factual
"who-did-what-to-whom" core captured by the graph. Our hybrid system
outperforms conventional methods in complex regulatory queries, ensuring
factual correctness with embedded triplets, enabling traceability through a
unified vector database, and enhancing understanding through subgraph
visualization, providing a robust foundation for compliance-driven and broader
audit-focused applications.

### 9. [Mathematical Computation and Reasoning Errors by Large Language Models](http://arxiv.org/pdf/2508.09932v1)

Authors: Liang Zhang, Edith Aurora Graf

Large Language Models (LLMs) are increasingly utilized in AI-driven
educational instruction and assessment, particularly within mathematics
education. The capability of LLMs to generate accurate answers and detailed
solutions for math problem-solving tasks is foundational for ensuring reliable
and precise feedback and assessment in math education practices. Our study
focuses on evaluating the accuracy of four LLMs (OpenAI GPT-4o and o1,
DeepSeek-V3 and DeepSeek-R1) solving three categories of math tasks, including
arithmetic, algebra, and number theory, and identifies step-level reasoning
errors within their solutions. Instead of relying on standard benchmarks, we
intentionally build math tasks (via item models) that are challenging for LLMs
and prone to errors. The accuracy of final answers and the presence of errors
in individual solution steps were systematically analyzed and coded. Both
single-agent and dual-agent configurations were tested. It is observed that the
reasoning-enhanced OpenAI o1 model consistently achieved higher or nearly
perfect accuracy across all three math task categories. Analysis of errors
revealed that procedural slips were the most frequent and significantly
impacted overall performance, while conceptual misunderstandings were less
frequent. Deploying dual-agent configurations substantially improved overall
performance. These findings offer actionable insights into enhancing LLM
performance and underscore effective strategies for integrating LLMs into
mathematics education, thereby advancing AI-driven instructional practices and
assessment precision.

### 10. [RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in Streetscape Images from Open Government Metadata](http://arxiv.org/pdf/2508.09415v1)

Authors: John S. O'Meara, Jared Hwang, Zeyu Wang, Michael Saugstad, Jon E. Froehlich

Curb ramps are critical for urban accessibility, but robustly detecting them
in images remains an open problem due to the lack of large-scale, high-quality
datasets. While prior work has attempted to improve data availability with
crowdsourced or manually labeled data, these efforts often fall short in either
quality or scale. In this paper, we introduce and evaluate a two-stage pipeline
called RampNet to scale curb ramp detection datasets and improve model
performance. In Stage 1, we generate a dataset of more than 210,000 annotated
Google Street View (GSV) panoramas by auto-translating government-provided curb
ramp location data to pixel coordinates in panoramic images. In Stage 2, we
train a curb ramp detection model (modified ConvNeXt V2) from the generated
dataset, achieving state-of-the-art performance. To evaluate both stages of our
pipeline, we compare to manually labeled panoramas. Our generated dataset
achieves 94.0% precision and 92.5% recall, and our detection model reaches
0.9236 AP -- far exceeding prior work. Our work contributes the first
large-scale, high-quality curb ramp detection dataset, benchmark, and model.

### Hardware Architecture

### 1. [Re-thinking Memory-Bound Limitations in CGRAs](http://arxiv.org/pdf/2508.09570v1)

Authors: Xiangfeng Liu, Zhe Jiang, Anzhen Zhu, Xiaomeng Han, Mingsong Lyu, Qingxu Deng, Nan Guan

Coarse-Grained Reconfigurable Arrays (CGRAs) are specialized accelerators
commonly employed to boost performance in workloads with iterative structures.
Existing research typically focuses on compiler or architecture optimizations
aimed at improving CGRA performance, energy efficiency, flexibility, and area
utilization, under the idealistic assumption that kernels can access all data
from Scratchpad Memory (SPM). However, certain complex workloads-particularly
in fields like graph analytics, irregular database operations, and specialized
forms of high-performance computing (e.g., unstructured mesh
simulations)-exhibit irregular memory access patterns that hinder CGRA
utilization, sometimes dropping below 1.5%, making the CGRA memory-bound. To
address this challenge, we conduct a thorough analysis of the underlying causes
of performance degradation, then propose a redesigned memory subsystem and
refine the memory model. With both microarchitectural and theoretical
optimization, our solution can effectively manage irregular memory accesses
through CGRA-specific runahead execution mechanism and cache reconfiguration
techniques. Our results demonstrate that we can achieve performance comparable
to the original SPM-only system while requiring only 1.27% of the storage size.
The runahead execution mechanism achieves an average 3.04x speedup (up to
6.91x), with cache reconfiguration technique providing an additional 6.02%
improvement, significantly enhancing CGRA performance for irregular memory
access patterns.

### 2. [MiCo: End-to-End Mixed Precision Neural Network Co-Exploration Framework for Edge AI](http://arxiv.org/pdf/2508.09500v1)

Authors: Zijun Jiang, Yangdi Lyu

Quantized Neural Networks (QNN) with extremely low-bitwidth data have proven
promising in efficient storage and computation on edge devices. To further
reduce the accuracy drop while increasing speedup, layer-wise mixed-precision
quantization (MPQ) becomes a popular solution. However, existing algorithms for
exploring MPQ schemes are limited in flexibility and efficiency. Comprehending
the complex impacts of different MPQ schemes on post-training quantization and
quantization-aware training results is a challenge for conventional methods.
Furthermore, an end-to-end framework for the optimization and deployment of MPQ
models is missing in existing work.
  In this paper, we propose the MiCo framework, a holistic MPQ exploration and
deployment framework for edge AI applications. The framework adopts a novel
optimization algorithm to search for optimal quantization schemes with the
highest accuracies while meeting latency constraints. Hardware-aware latency
models are built for different hardware targets to enable fast explorations.
After the exploration, the framework enables direct deployment from PyTorch MPQ
models to bare-metal C codes, leading to end-to-end speedup with minimal
accuracy drops.

### 3. [Low-latency D-MIMO Localization using Distributed Scalable Message-Passing Algorithm](http://arxiv.org/pdf/2508.09546v1)

Authors: Dumitra Iancu, Liang Liu, Ove Edfors, Erik Leitinger, Xuhong Li

Distributed MIMO and integrated sensing and communication are expected to be
key technologies in future wireless systems, enabling reliable, low-latency
communication and accurate localization. Dedicated localization solutions must
support distributed architecture, provide scalability across different system
configurations and meet strict latency requirements. We present a scalable
message-passing localization method and architecture co-designed for a
panel-based distributed MIMO system and network topology, in which
interconnected units operate without centralized processing. This method
jointly detects line-of-sight paths to distributed units from multipath
measurements in dynamic scenarios, localizes the agent, and achieves very low
latency. Additionally, we introduce a cycle-accurate system latency model based
on implemented FPGA operations, and show important insights into processing
latency and hardware utilization and system-level trade-offs. We compare our
method to a multipath-based localization method and show that it can achieve
similar localization performance, with wide enough distribution of array
elements, while offering lower latency and computational complexity.

### Computational Complexity

### 1. [On Middle Grounds for Preference Statements](http://arxiv.org/pdf/2508.09553v1)

Authors: Anne-Marie George, Ana Ozaki

In group decisions or deliberations, stakeholders are often confronted with
conflicting opinions. We investigate a logic-based way of expressing such
opinions and a formal general notion of a middle ground between stakeholders.
Inspired by the literature on preferences with hierarchical and lexicographic
models, we instantiate our general framework to the case where stakeholders
express their opinions using preference statements of the form I prefer 'a' to
'b', where 'a' and 'b' are alternatives expressed over some attributes, e.g.,
in a trolley problem, one can express I prefer to save 1 adult and 1 child to 2
adults (and 0 children). We prove theoretical results on the existence and
uniqueness of middle grounds. In particular, we show that, for preference
statements, middle grounds may not exist and may not be unique. We also provide
algorithms for deciding the existence and finding middle grounds.

### 2. [Reasoning About Knowledge on Regular Expressions is 2EXPTIME-complete](http://arxiv.org/pdf/2508.09784v1)

Authors: Avijeet Ghosh, Sujata Ghosh, François Schwarzentruber

Logics for reasoning about knowledge and actions have seen many applications
in various domains of multi-agent systems, including epistemic planning. Change
of knowledge based on observations about the surroundings forms a key aspect in
such planning scenarios. Public Observation Logic (POL) is a variant of public
announcement logic for reasoning about knowledge that gets updated based on
public observations. Each state in an epistemic (Kripke) model is equipped with
a set of expected observations. These states evolve as the expectations get
matched with the actual observations. In this work, we prove that the
satisfiability problem of $\POL$ is 2EXPTIME-complete.

### Computational Engineering

### 1. [VisFinEval: A Scenario-Driven Chinese Multimodal Benchmark for Holistic Financial Understanding](http://arxiv.org/pdf/2508.09641v1)

Authors: Zhaowei Liu, Xin Guo, Haotian Xia, Lingfeng Zeng, Fangqi Lou, Jinyi Niu, Mengping Li, Qi Qi, Jiahuan Li, Wei Zhang, Yinglong Wang, Weige Cai, Weining Shen, Liwen Zhang

Multimodal large language models (MLLMs) hold great promise for automating
complex financial analysis. To comprehensively evaluate their capabilities, we
introduce VisFinEval, the first large-scale Chinese benchmark that spans the
full front-middle-back office lifecycle of financial tasks. VisFinEval
comprises 15,848 annotated question-answer pairs drawn from eight common
financial image modalities (e.g., K-line charts, financial statements, official
seals), organized into three hierarchical scenario depths: Financial Knowledge
& Data Analysis, Financial Analysis & Decision Support, and Financial Risk
Control & Asset Optimization. We evaluate 21 state-of-the-art MLLMs in a
zero-shot setting. The top model, Qwen-VL-max, achieves an overall accuracy of
76.3%, outperforming non-expert humans but trailing financial experts by over
14 percentage points. Our error analysis uncovers six recurring failure
modes-including cross-modal misalignment, hallucinations, and lapses in
business-process reasoning-that highlight critical avenues for future research.
VisFinEval aims to accelerate the development of robust, domain-tailored MLLMs
capable of seamlessly integrating textual and visual financial information. The
data and the code are available at
https://github.com/SUFE-AIFLM-Lab/VisFinEval.

### 2. [Finetuning Large Language Model as an Effective Symbolic Regressor](http://arxiv.org/pdf/2508.09897v1)

Authors: Yingfan Hua, Ruikun Li, Jun Yao, Guohang Zhuang, Shixiang Tang, Bin Liu, Wanli Ouyang, Yan Lu

Deriving governing equations from observational data, known as Symbolic
Regression (SR), is a cornerstone of scientific discovery. Large Language
Models (LLMs) have shown promise in this task by leveraging their vast
cross-disciplinary scientific knowledge. However, existing LLM-based methods
primarily rely on direct inference or prompt engineering, often requiring
excessive inference iterations to converge on correct formulas or failing to
treating complex equation targets. These limitations in effectiveness and
generalization stem from an inherent tension between pre-trained LLMs'
proficiency in approximate reasoning and the high-precision demands of SR
tasks. To bridge this gap, we propose to fine-tune LLMs for enhanced SR
capability. Yet, the absence of dedicated datasets for SR-oriented fine-tuning
remains a critical barrier. We thus introduce SymbArena, specifically
engineered to optimize LLMs for SR. This benchmark comprises 148,102 diverse
equations formulated as corpora of 1.83 billion tokens for LLM utilization,
enabling effective training and inference. Further, SymbArena proposes a
heuristics metric to precisely quantify form-level consistency, going beyond
existing SR numerical-oriented evaluation strategies. With this benchmark, we
explore mainstream LLM fine-tuning techniques for SR tasks and establish
SymbolicChat, a simple yet effective LLM-based SR strong baseline. Experimental
results validate SymbolicChat as the first LLM to exceed traditional numerical
methods in both numerical precision and symbolic form accuracy, outperforming
the second-best LLM baseline with improvements of 2-fold gains in R2 score and
8.37% in form-level consistency score.

### 3. [Large-Scale Topology Optimisation of Time-dependent Thermal Conduction Using Space-Time Finite Elements and a Parallel Space-Time Multigrid Preconditioner](http://arxiv.org/pdf/2508.09589v1)

Authors: Joe Alexandersen, Magnus Appel

This paper presents a novel space-time topology optimisation framework for
time-dependent thermal conduction problems, aiming to significantly reduce the
time-to-solution. By treating time as an additional spatial dimension, we
discretise the governing equations using a stabilised continuous Galerkin
space-time finite element method. The resulting large all-at-once system is
solved using an iterative Krylov solver preconditioned with a parallel
space-time multigrid method employing a semi-coarsening strategy. Implemented
in a fully parallel computing framework, the method yields a parallel-in-time
method that demonstrates excellent scalability on a distributed-memory
supercomputer, solving problems up to 4.2 billion degrees of freedom.
Comparative studies show up to 52x speed-up over traditional time-stepping
approaches, with only moderate increases in total computational cost in terms
of core-hours. The framework is validated on benchmark problems with both
time-constant and time-varying designs, and its flexibility is demonstrated
through variations in material properties. These results establish the proposed
space-time method as a promising approach for large-scale time-dependent
topology optimisation in thermal applications.

### 4. [TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos](http://arxiv.org/pdf/2508.09811v1)

Authors: Jinxi Li, Ziyang Song, Bo Yang

In this paper, we aim to model 3D scene geometry, appearance, and physical
information just from dynamic multi-view videos in the absence of any human
labels. By leveraging physics-informed losses as soft constraints or
integrating simple physics models into neural nets, existing works often fail
to learn complex motion physics, or doing so requires additional labels such as
object types or masks. We propose a new framework named TRACE to model the
motion physics of complex dynamic 3D scenes. The key novelty of our method is
that, by formulating each 3D point as a rigid particle with size and
orientation in space, we directly learn a translation rotation dynamics system
for each particle, explicitly estimating a complete set of physical parameters
to govern the particle's motion over time. Extensive experiments on three
existing dynamic datasets and one newly created challenging synthetic datasets
demonstrate the extraordinary performance of our method over baselines in the
task of future frame extrapolation. A nice property of our framework is that
multiple objects or parts can be easily segmented just by clustering the
learned physical parameters.

### Computational Geometry

### 1. [Simpler and Faster Contiguous Art Gallery](http://arxiv.org/pdf/2508.09734v1)

Authors: Sarita de Berg, Jacobus Conradi, Ivor van der Hoog, Frank Staals

The contiguous art gallery problem was introduced at SoCG'25 in a merged
paper that combined three simultaneous results, each achieving a
polynomial-time algorithm for the problem. This problem is a variant of the
classical art gallery problem, first introduced by Klee in 1973. In the
contiguous art gallery problem, we are given a polygon P and asked to determine
the minimum number of guards needed, where each guard is assigned a contiguous
portion of the boundary of P that it can see, such that all assigned portions
together cover the boundary of P. The classical art gallery problem is NP-hard
and ER-complete, and the three independent works investigated whether this
variant admits a polynomial-time solution. Each of these works indeed presented
such a solution, with the fastest running in O(k n^5 log n) time, where n
denotes the number of vertices of P and k is the size of a minimum guard set
covering the boundary of P. We present a solution that is both considerably
simpler and significantly faster, yielding a concise and almost entirely
self-contained O(k n^2 log^2 n)-time algorithm.

### 2. [SHREC'25 Track on Multiple Relief Patterns: Report and Analysis](http://arxiv.org/pdf/2508.09909v1)

Authors: Gabriele Paolini, Claudio Tortorici, Stefano Berretti, Ahmed Hazem Youssef, Halim Benhabiles, Adnane Cabani, Ruiwen He, Karim Hammoudi, Iyyakutti Iyappan Ganapathi, Syed Sadaf Ali, Divya Velayudhan, Maregu Assefa, Naoufel Werghi

This SHREC 2025 track focuses on the recognition and segmentation of relief
patterns embedded on the surface of a set of synthetically generated triangle
meshes. We report the methods proposed by the participants, whose performance
highlights the inherent complexity of solving the problem, which is still open.
Then, we discuss the critical aspects of the proposed tasks, highlight the
limitations of current techniques, and outline possible directions for future
research. All resources and track details are available at the official track
webpage: https://sites.google.com/unifi.it/shrec25-relief-pattern.

### 3. [Distributed Diamond Formation of Sliding Squares](http://arxiv.org/pdf/2508.09638v1)

Authors: Irina Kostitsyna, David Liedtke, Christian Scheideler

The sliding square model is a widely used abstraction for studying
self-reconfigurable robotic systems, where modules are square-shaped robots
that move by sliding or rotating over one another. In this paper, we propose a
novel distributed algorithm that allows a group of modules to reconfigure into
a diamond shape, starting from an arbitrary side-connected configuration. It is
connectivity-preserving and operates under minimal assumptions: one leader
module, common chirality, constant memory per module, and visibility and
communication restricted to immediate neighbors. Unlike prior work, which
relaxes the original sliding square move-set, our approach uses the unmodified
move-set, addressing the additional challenge of handling locked
configurations. Our algorithm is sequential in nature and operates with a
worst-case time complexity of $\mathcal{O}(n^2)$ rounds, which is optimal for
sequential algorithms. To improve runtime, we introduce two parallel variants
of the algorithm. Both rely on a spanning tree data structure, allowing modules
to make decisions based on local connectivity. Our experimental results show a
significant speedup for the first variant, and linear average runtime for the
second variant, which is worst-case optimal for parallel algorithms.

### 4. [Retroactive Monotonic Priority Queues via Range Searching](http://arxiv.org/pdf/2508.09892v1)

Authors: Lucas Castro, Rosiane de Freitas

The best known fully retroactive priority queue costs $O(\log^2 m \log \log
m)$ time per operation, where $m$ is the number of operations performed on the
data structure. In contrast, standard (non-retroactive) and partially
retroactive priority queues cost $O(\log m)$ time per operation. So far, it is
unknown whether this $O(\log m)$ bound can be achieved for fully retroactive
priority queues.
  In this work, we study a restricted variant of priority queues known as
monotonic priority queues. We show that finding the minimum in a retroactive
monotonic priority queue is a special case of the range-searching problem. We
design a fully retroactive monotonic priority queue with a cost of $O(\log m +
T(m))$ time per operation, where $T(m)$ is the maximum between the query and
the update time of a specific range-searching data structure with $m$ elements.
Finally, we design a fully retroactive monotonic priority queue that costs
$O(\log m \log \log m)$ time per operation.

### 5. [CWFBind: Geometry-Awareness for Fast and Accurate Protein-Ligand Docking](http://arxiv.org/pdf/2508.09499v1)

Authors: Liyan Jia, Chuan-Xian Ren, Hong Yan

Accurately predicting the binding conformation of small-molecule ligands to
protein targets is a critical step in rational drug design. Although recent
deep learning-based docking surpasses traditional methods in speed and
accuracy, many approaches rely on graph representations and language
model-inspired encoders while neglecting critical geometric information,
resulting in inaccurate pocket localization and unrealistic binding
conformations. In this study, we introduce CWFBind, a weighted, fast, and
accurate docking method based on local curvature features. Specifically, we
integrate local curvature descriptors during the feature extraction phase to
enrich the geometric representation of both proteins and ligands, complementing
existing chemical, sequence, and structural features. Furthermore, we embed
degree-aware weighting mechanisms into the message passing process, enhancing
the model's ability to capture spatial structural distinctions and interaction
strengths. To address the class imbalance challenge in pocket prediction,
CWFBind employs a ligand-aware dynamic radius strategy alongside an enhanced
loss function, facilitating more precise identification of binding regions and
key residues. Comprehensive experimental evaluations demonstrate that CWFBind
achieves competitive performance across multiple docking benchmarks, offering a
balanced trade-off between accuracy and efficiency.

### Computation and Language

### 1. [From Charts to Fair Narratives: Uncovering and Mitigating Geo-Economic Biases in Chart-to-Text](http://arxiv.org/pdf/2508.09450v1)

Authors: Ridwan Mahbub, Mohammed Saidul Islam, Mir Tafseer Nayeem, Md Tahmid Rahman Laskar, Mizanur Rahman, Shafiq Joty, Enamul Hoque

Charts are very common for exploring data and communicating insights, but
extracting key takeaways from charts and articulating them in natural language
can be challenging. The chart-to-text task aims to automate this process by
generating textual summaries of charts. While with the rapid advancement of
large Vision-Language Models (VLMs), we have witnessed great progress in this
domain, little to no attention has been given to potential biases in their
outputs. This paper investigates how VLMs can amplify geo-economic biases when
generating chart summaries, potentially causing societal harm. Specifically, we
conduct a large-scale evaluation of geo-economic biases in VLM-generated chart
summaries across 6,000 chart-country pairs from six widely used proprietary and
open-source models to understand how a country's economic status influences the
sentiment of generated summaries. Our analysis reveals that existing VLMs tend
to produce more positive descriptions for high-income countries compared to
middle- or low-income countries, even when country attribution is the only
variable changed. We also find that models such as GPT-4o-mini,
Gemini-1.5-Flash, and Phi-3.5 exhibit varying degrees of bias. We further
explore inference-time prompt-based debiasing techniques using positive
distractors but find them only partially effective, underscoring the complexity
of the issue and the need for more robust debiasing strategies. Our code and
dataset are publicly available here.

### 2. [User-centric Subjective Leaderboard by Customizable Reward Modeling](http://arxiv.org/pdf/2508.09463v1)

Authors: Qi Jia, Xiujie Song, Zicheng Zhang, Yijin Guo, Kaiwei Zhang, Zijian Chen, Guangtao Zhai

Existing benchmarks for large language models (LLMs) predominantely focus on
assessing their capabilities through verifiable tasks. Such objective and
static benchmarks offer limited utility for practical LLM selection, making it
difficult for users to find suitable models for their individual needs. To
bridge this gap, we present the first User-Centric Subjective Leaderboard
(USL), which provides a preference-driven, dynamic ranking of LLMs across
diverse real-world scenarios. Our work is built upon a thorough investigation
of real human preference data, involving more than 10K subjective queries. Our
investigation reveals significant diversity and contradictions in human
preferences, which limit the effectiveness of state-of-the-art reward models.
To address this, we introduce Customizable Reward Models (CRMs). With only 4B
parameters, our CRM surpasses the performance of leading models such as GPT-4.1
and Gemini-2.5-pro, showing exceptional generalization capabilities across new
topics and criteria. The USL, powered by CRMs, exhibits strong negative
correlations to contradictory preferences.

### 3. [LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation](http://arxiv.org/pdf/2508.09515v1)

Authors: Jakub Šmíd, Pavel Přibáň, Pavel Král

Cross-lingual aspect-based sentiment analysis (ABSA) involves detailed
sentiment analysis in a target language by transferring knowledge from a source
language with available annotated data. Most existing methods depend heavily on
often unreliable translation tools to bridge the language gap. In this paper,
we propose a new approach that leverages a large language model (LLM) to
generate high-quality pseudo-labelled data in the target language without the
need for translation tools. First, the framework trains an ABSA model to obtain
predictions for unlabelled target language data. Next, LLM is prompted to
generate natural sentences that better represent these noisy predictions than
the original text. The ABSA model is then further fine-tuned on the resulting
pseudo-labelled dataset. We demonstrate the effectiveness of this method across
six languages and five backbone models, surpassing previous state-of-the-art
translation-based approaches. The proposed framework also supports generative
models, and we show that fine-tuned LLMs outperform smaller multilingual
models.

### 4. [Cross-lingual Aspect-Based Sentiment Analysis: A Survey on Tasks, Approaches, and Challenges](http://arxiv.org/pdf/2508.09516v1)

Authors: Jakub Šmíd, Pavel Král

Aspect-based sentiment analysis (ABSA) is a fine-grained sentiment analysis
task that focuses on understanding opinions at the aspect level, including
sentiment towards specific aspect terms, categories, and opinions. While ABSA
research has seen significant progress, much of the focus has been on
monolingual settings. Cross-lingual ABSA, which aims to transfer knowledge from
resource-rich languages (such as English) to low-resource languages, remains an
under-explored area, with no systematic review of the field. This paper aims to
fill that gap by providing a comprehensive survey of cross-lingual ABSA. We
summarize key ABSA tasks, including aspect term extraction, aspect sentiment
classification, and compound tasks involving multiple sentiment elements.
Additionally, we review the datasets, modelling paradigms, and cross-lingual
transfer methods used to solve these tasks. We also examine how existing work
in monolingual and multilingual ABSA, as well as ABSA with LLMs, contributes to
the development of cross-lingual ABSA. Finally, we highlight the main
challenges and suggest directions for future research to advance cross-lingual
ABSA systems.

### 5. [UWBa at SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval](http://arxiv.org/pdf/2508.09517v1)

Authors: Ladislav Lenc, Daniel Cífka, Jiří Martínek, Jakub Šmíd, Pavel Král

This paper presents a zero-shot system for fact-checked claim retrieval. We
employed several state-of-the-art large language models to obtain text
embeddings. The models were then combined to obtain the best possible result.
Our approach achieved 7th place in monolingual and 9th in cross-lingual
subtasks. We used only English translations as an input to the text embedding
models since multilingual models did not achieve satisfactory results. We
identified the most relevant claims for each post by leveraging the embeddings
and measuring cosine similarity. Overall, the best results were obtained by the
NVIDIA NV-Embed-v2 model. For some languages, we benefited from model
combinations (NV-Embed & GPT or Mistral).

### 6. [The Surprising Effectiveness of Membership Inference with Simple N-Gram Coverage](http://arxiv.org/pdf/2508.09603v1)

Authors: Skyler Hallinan, Jaehun Jung, Melanie Sclar, Ximing Lu, Abhilasha Ravichander, Sahana Ramnath, Yejin Choi, Sai Praneeth Karimireddy, Niloofar Mireshghallah, Xiang Ren

Membership inference attacks serves as useful tool for fair use of language
models, such as detecting potential copyright infringement and auditing data
leakage. However, many current state-of-the-art attacks require access to
models' hidden states or probability distribution, which prevents investigation
into more widely-used, API-access only models like GPT-4. In this work, we
introduce N-Gram Coverage Attack, a membership inference attack that relies
solely on text outputs from the target model, enabling attacks on completely
black-box models. We leverage the observation that models are more likely to
memorize and subsequently generate text patterns that were commonly observed in
their training data. Specifically, to make a prediction on a candidate member,
N-Gram Coverage Attack first obtains multiple model generations conditioned on
a prefix of the candidate. It then uses n-gram overlap metrics to compute and
aggregate the similarities of these outputs with the ground truth suffix; high
similarities indicate likely membership. We first demonstrate on a diverse set
of existing benchmarks that N-Gram Coverage Attack outperforms other black-box
methods while also impressively achieving comparable or even better performance
to state-of-the-art white-box attacks - despite having access to only text
outputs. Interestingly, we find that the success rate of our method scales with
the attack compute budget - as we increase the number of sequences generated
from the target model conditioned on the prefix, attack performance tends to
improve. Having verified the accuracy of our method, we use it to investigate
previously unstudied closed OpenAI models on multiple domains. We find that
more recent models, such as GPT-4o, exhibit increased robustness to membership
inference, suggesting an evolving trend toward improved privacy protections.

### 7. [AINL-Eval 2025 Shared Task: Detection of AI-Generated Scientific Abstracts in Russian](http://arxiv.org/pdf/2508.09622v1)

Authors: Tatiana Batura, Elena Bruches, Milana Shvenk, Valentin Malykh

The rapid advancement of large language models (LLMs) has revolutionized text
generation, making it increasingly difficult to distinguish between human- and
AI-generated content. This poses a significant challenge to academic integrity,
particularly in scientific publishing and multilingual contexts where detection
resources are often limited. To address this critical gap, we introduce the
AINL-Eval 2025 Shared Task, specifically focused on the detection of
AI-generated scientific abstracts in Russian. We present a novel, large-scale
dataset comprising 52,305 samples, including human-written abstracts across 12
diverse scientific domains and AI-generated counterparts from five
state-of-the-art LLMs (GPT-4-Turbo, Gemma2-27B, Llama3.3-70B, Deepseek-V3, and
GigaChat-Lite). A core objective of the task is to challenge participants to
develop robust solutions capable of generalizing to both (i) previously unseen
scientific domains and (ii) models not included in the training data. The task
was organized in two phases, attracting 10 teams and 159 submissions, with top
systems demonstrating strong performance in identifying AI-generated content.
We also establish a continuous shared task platform to foster ongoing research
and long-term progress in this important area. The dataset and platform are
publicly available at https://github.com/iis-research-team/AINL-Eval-2025.

### 8. [EffiEval: Efficient and Generalizable Model Evaluation via Capability Coverage Maximization](http://arxiv.org/pdf/2508.09662v1)

Authors: Yaoning Wang, Jiahao Ying, Yixin Cao, Yubo Ma, Yugang Jiang

The rapid advancement of large language models (LLMs) and the development of
increasingly large and diverse evaluation benchmarks have introduced
substantial computational challenges for model assessment. In this paper, we
present EffiEval, a training-free approach for efficient benchmarking that
effectively addresses data redundancy while maintaining high evaluation
reliability. Our method is specifically designed to meet three key criteria for
high-quality evaluation: representativeness, by ensuring comprehensive coverage
of model capabilities; fairness, by remaining independent of model performance
during sample selection to avoid bias; and generalizability, by enabling
flexible transfer across datasets and model families without reliance on
large-scale evaluation data. Unlike traditional methods that rely on absolute
performance or require extensive evaluation data, our approach adaptively
selects high-quality representative subsets based on the Model Utility Index
(MUI). Extensive experiments on multiple public benchmarks and diverse LLMs
demonstrate that EffiEval achieves strong ranking consistency with full-dataset
evaluation using only a small fraction of the original data. Furthermore, our
method is flexible and scalable in size, allowing users to balance evaluation
efficiency and representativeness according to specific needs. Overall,
EffiEval provides a practical and generalizable solution for reliable, fair,
and efficient evaluation in the era of LLMs.

### 9. [Slow Tuning and Low-Entropy Masking for Safe Chain-of-Thought Distillation](http://arxiv.org/pdf/2508.09666v1)

Authors: Ziyang Ma, Qingyue Yuan, Linhai Zhang, Deyu Zhou

Previous chain-of-thought (CoT) distillation methods primarily focused on
enhancing the reasoning capabilities of Small Language Models (SLMs) by
utilizing high-quality rationales generated by powerful Large Language Models
(LLMs, e.g., GPT-4). However, few works have noted the negative effects on SLM
safety brought by the training, which are revealed in this study. Although
there are works on safety alignment that fine-tune language models or
manipulate model weights to defend against harmful inputs, they require extra
computation or annotated data, and probably impact the reasoning ability of
SLMs. In this paper, we investigate how to maintain the safety of SLMs during
the CoT distillation process. Specifically, we propose a safe distillation
method, Slow Tuning and Low-Entropy Masking Distillation (SLowED), containing
two modules: Slow Tuning and Low-Entropy Masking. Slow Tuning scales down the
magnitude of model weight changes to optimize the model weights in the
neighboring space near the initial weight distribution. Low-Entropy Masking
masks low-entropy tokens, which are regarded as unnecessary learning targets,
to exclude them from fine-tuning. Experiments on three SLMs (Qwen2.5-1.5B,
Llama-3.2-1B, BLOOM-1.1B) across reasoning benchmarks (BBH, BB-Sub, ARC,
AGIEval) and safety evaluation (AdvBench) show that SLowED retains the safety
of SLMs and comparably improves their reasoning capability compared to existing
distillation methods. Furthermore, our ablation study presents the
effectiveness of Slow Tuning and Low-Entropy Masking, with the former
maintaining the model's safety in the early stage and the latter prolonging the
safe training epochs.

### 10. [The Perils of Chart Deception: How Misleading Visualizations Affect Vision-Language Models](http://arxiv.org/pdf/2508.09716v1)

Authors: Ridwan Mahbub, Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Mizanur Rahman, Mir Tafseer Nayeem, Enamul Hoque

Information visualizations are powerful tools that help users quickly
identify patterns, trends, and outliers, facilitating informed decision-making.
However, when visualizations incorporate deceptive design elements-such as
truncated or inverted axes, unjustified 3D effects, or violations of best
practices-they can mislead viewers and distort understanding, spreading
misinformation. While some deceptive tactics are obvious, others subtly
manipulate perception while maintaining a facade of legitimacy. As
Vision-Language Models (VLMs) are increasingly used to interpret
visualizations, especially by non-expert users, it is critical to understand
how susceptible these models are to deceptive visual designs. In this study, we
conduct an in-depth evaluation of VLMs' ability to interpret misleading
visualizations. By analyzing over 16,000 responses from ten different models
across eight distinct types of misleading chart designs, we demonstrate that
most VLMs are deceived by them. This leads to altered interpretations of
charts, despite the underlying data remaining the same. Our findings highlight
the need for robust safeguards in VLMs against visual misinformation.

### Cryptography and Security

### 1. [Security Analysis of ChatGPT: Threats and Privacy Risks](http://arxiv.org/pdf/2508.09426v1)

Authors: Yushan Xiang, Zhongwen Li, Xiaoqi Li

As artificial intelligence technology continues to advance, chatbots are
becoming increasingly powerful. Among them, ChatGPT, launched by OpenAI, has
garnered widespread attention globally due to its powerful natural language
processing capabilities based on the GPT model, which enables it to engage in
natural conversations with users, understand various forms of linguistic
expressions, and generate useful information and suggestions. However, as its
application scope expands, user demand grows, and malicious attacks related to
it become increasingly frequent, the security threats and privacy risks faced
by ChatGPT are gradually coming to the forefront. In this paper, the security
of ChatGPT is mainly studied from two aspects, security threats and privacy
risks. The article systematically analyzes various types of vulnerabilities
involved in the above two types of problems and their causes. Briefly, we
discuss the controversies that ChatGPT may cause at the ethical and moral
levels. In addition, this paper reproduces several network attack and defense
test scenarios by simulating the attacker's perspective and methodology.
Simultaneously, it explores the feasibility of using ChatGPT for security
vulnerability detection and security tool generation from the defender's
perspective.

### 2. [Succinct Oblivious Tensor Evaluation and Applications: Adaptively-Secure Laconic Function Evaluation and Trapdoor Hashing for All Circuits](http://arxiv.org/pdf/2508.09673v1)

Authors: Damiano Abram, Giulio Malavolta, Lawrence Roy

We propose the notion of succinct oblivious tensor evaluation (OTE), where
two parties compute an additive secret sharing of a tensor product of two
vectors $\mathbf{x} \otimes \mathbf{y}$, exchanging two simultaneous messages.
Crucially, the size of both messages and of the CRS is independent of the
dimension of $\mathbf{x}$.
  We present a construction of OTE with optimal complexity from the standard
learning with errors (LWE) problem. Then we show how this new technical tool
enables a host of cryptographic primitives, all with security reducible to LWE,
such as:
  * Adaptively secure laconic function evaluation for depth-$D$ functions
$f:\{0, 1\}^m\rightarrow\{0, 1\}^\ell$ with communication $m+\ell+D\cdot
\mathrm{poly}(\lambda)$.
  * A trapdoor hash function for all functions.
  * An (optimally) succinct homomorphic secret sharing for all functions.
  * A rate-$1/2$ laconic oblivious transfer for batch messages, which is best
possible.
  In particular, we obtain the first laconic function evaluation scheme that is
adaptively secure from the standard LWE assumption, improving upon Quach, Wee,
and Wichs (FOCS 2018).
  As a key technical ingredient, we introduce a new notion of \emph{adaptive
lattice encodings}, which may be of independent interest.

### 3. [Perfect message authentication codes are robust to small deviations from uniform key distributions](http://arxiv.org/pdf/2508.09783v1)

Authors: Boris Ryabko

We investigate the impact of (possible) deviations of the probability
distribution of key values from a uniform distribution for the
information-theoretic strong, or perfect, message authentication code. We found
a simple expression for the decrease in security as a function of the
statistical distance between the real key probability distribution and the
uniform one. In a sense, a perfect message authentication code is robust to
small deviations from a uniform key distribution.

### 4. [Integrating Feature Attention and Temporal Modeling for Collaborative Financial Risk Assessment](http://arxiv.org/pdf/2508.09399v1)

Authors: Yue Yao, Zhen Xu, Youzhu Liu, Kunyuan Ma, Yuxiu Lin, Mohan Jiang

This paper addresses the challenges of data privacy and collaborative
modeling in cross-institution financial risk analysis. It proposes a risk
assessment framework based on federated learning. Without sharing raw data, the
method enables joint modeling and risk identification across multiple
institutions. This is achieved by incorporating a feature attention mechanism
and temporal modeling structure. Specifically, the model adopts a distributed
optimization strategy. Each financial institution trains a local sub-model. The
model parameters are protected using differential privacy and noise injection
before being uploaded. A central server then aggregates these parameters to
generate a global model. This global model is used for systemic risk
identification. To validate the effectiveness of the proposed method, multiple
experiments are conducted. These evaluate communication efficiency, model
accuracy, systemic risk detection, and cross-market generalization. The results
show that the proposed model outperforms both traditional centralized methods
and existing federated learning variants across all evaluation metrics. It
demonstrates strong modeling capabilities and practical value in sensitive
financial environments. The method enhances the scope and efficiency of risk
identification while preserving data sovereignty. It offers a secure and
efficient solution for intelligent financial risk analysis.

### 5. [CLIP-Flow: A Universal Discriminator for AI-Generated Images Inspired by Anomaly Detection](http://arxiv.org/pdf/2508.09477v1)

Authors: Zhipeng Yuan, Kai Wang, Weize Quan, Dong-Ming Yan, Tieru Wu

With the rapid advancement of AI generative models, the visual quality of
AI-generated images (AIIs) has become increasingly close to natural images,
which inevitably raises security concerns. Most AII detectors often employ the
conventional image classification pipeline with natural images and AIIs
(generated by a generative model), which can result in limited detection
performance for AIIs from unseen generative models. To solve this, we proposed
a universal AI-generated image detector from the perspective of anomaly
detection. Our discriminator does not need to access any AIIs and learn a
generalizable representation with unsupervised learning. Specifically, we use
the pre-trained CLIP encoder as the feature extractor and design a normalizing
flow-like unsupervised model. Instead of AIIs, proxy images, e.g., obtained by
applying a spectral modification operation on natural images, are used for
training. Our models are trained by minimizing the likelihood of proxy images,
optionally combined with maximizing the likelihood of natural images. Extensive
experiments demonstrate the effectiveness of our method on AIIs produced by
various image generators.

### 6. [Causal Graph Profiling via Structural Divergence for Robust Anomaly Detection in Cyber-Physical Systems](http://arxiv.org/pdf/2508.09504v1)

Authors: Arun Vignesh Malarkkan, Haoyue Bai, Dongjie Wang, Yanjie Fu

With the growing complexity of cyberattacks targeting critical
infrastructures such as water treatment networks, there is a pressing need for
robust anomaly detection strategies that account for both system
vulnerabilities and evolving attack patterns. Traditional methods --
statistical, density-based, and graph-based models struggle with distribution
shifts and class imbalance in multivariate time series, often leading to high
false positive rates. To address these challenges, we propose CGAD, a Causal
Graph-based Anomaly Detection framework designed for reliable cyberattack
detection in public infrastructure systems. CGAD follows a two-phase supervised
framework -- causal profiling and anomaly scoring. First, it learns causal
invariant graph structures representing the system's behavior under "Normal"
and "Attack" states using Dynamic Bayesian Networks. Second, it employs
structural divergence to detect anomalies via causal graph comparison by
evaluating topological deviations in causal graphs over time. By leveraging
causal structures, CGAD achieves superior adaptability and accuracy in
non-stationary and imbalanced time series environments compared to conventional
machine learning approaches. By uncovering causal structures beneath volatile
sensor data, our framework not only detects cyberattacks with markedly higher
precision but also redefines robustness in anomaly detection, proving
resilience where traditional models falter under imbalance and drift. Our
framework achieves substantial gains in F1 and ROC-AUC scores over
best-performing baselines across four industrial datasets, demonstrating robust
detection of delayed and structurally complex anomalies.

### 7. [Demystifying the Role of Rule-based Detection in AI Systems for Windows Malware Detection](http://arxiv.org/pdf/2508.09652v1)

Authors: Andrea Ponte, Luca Demetrio, Luca Oneto, Ivan Tesfai Ogbu, Battista Biggio, Fabio Roli

Malware detection increasingly relies on AI systems that integrate
signature-based detection with machine learning. However, these components are
typically developed and combined in isolation, missing opportunities to reduce
data complexity and strengthen defenses against adversarial EXEmples, carefully
crafted programs designed to evade detection. Hence, in this work we
investigate the influence that signature-based detection exerts on model
training, when they are included inside the training pipeline. Specifically, we
compare models trained on a comprehensive dataset with an AI system whose
machine learning component is trained solely on samples not already flagged by
signatures. Our results demonstrate improved robustness to both adversarial
EXEmples and temporal data drift, although this comes at the cost of a fixed
lower bound on false positives, driven by suboptimal rule selection. We
conclude by discussing these limitations and outlining how future research
could extend AI-based malware detection to include dynamic analysis, thereby
further enhancing system resilience.

### 8. [Route Planning and Online Routing for Quantum Key Distribution Networks](http://arxiv.org/pdf/2508.09735v1)

Authors: Jorge López, Charalampos Chatzinakis, Marc Cartigny

Quantum Key Distribution (QKD) networks harness the principles of quantum
physics in order to securely transmit cryptographic key material, providing
physical guarantees. These networks require traditional management and
operational components, such as routing information through the network
elements. However, due to the limitations on capacity and the particularities
of information handling in these networks, traditional shortest paths
algorithms for routing perform poorly on both route planning and online
routing, which is counterintuitive. Moreover, due to the scarce resources in
such networks, often the expressed demand cannot be met by any assignment of
routes. To address both the route planning problem and the need for fair
automated suggestions in infeasible cases, we propose to model this problem as
a Quadratic Programming (QP) problem. For the online routing problem, we
showcase that the shortest (available) paths routing strategy performs poorly
in the online setting. Furthermore, we prove that the widest shortest path
routing strategy has a competitive ratio greater or equal than $\frac{1}{2}$,
efficiently addressing both routing modes in QKD networks.

### 9. [Explainable Ensemble Learning for Graph-Based Malware Detection](http://arxiv.org/pdf/2508.09801v1)

Authors: Hossein Shokouhinejad, Roozbeh Razavi-Far, Griffin Higgins, Ali A Ghorbani

Malware detection in modern computing environments demands models that are
not only accurate but also interpretable and robust to evasive techniques.
Graph neural networks (GNNs) have shown promise in this domain by modeling rich
structural dependencies in graph-based program representations such as control
flow graphs (CFGs). However, single-model approaches may suffer from limited
generalization and lack interpretability, especially in high-stakes security
applications. In this paper, we propose a novel stacking ensemble framework for
graph-based malware detection and explanation. Our method dynamically extracts
CFGs from portable executable (PE) files and encodes their basic blocks through
a two-step embedding strategy. A set of diverse GNN base learners, each with a
distinct message-passing mechanism, is used to capture complementary behavioral
features. Their prediction outputs are aggregated by a meta-learner implemented
as an attention-based multilayer perceptron, which both classifies malware
instances and quantifies the contribution of each base model. To enhance
explainability, we introduce an ensemble-aware post-hoc explanation technique
that leverages edge-level importance scores generated by a GNN explainer and
fuses them using the learned attention weights. This produces interpretable,
model-agnostic explanations aligned with the final ensemble decision.
Experimental results demonstrate that our framework improves classification
performance while providing insightful interpretations of malware behavior.

### 10. [On the Consistency and Performance of the Iterative Bayesian Update](http://arxiv.org/pdf/2508.09980v1)

Authors: Ehab ElSalamouny, Catuscia Palamidessi

For many social, scientific, and commercial purposes, it is often important
to estimate the distribution of the users' data regarding a sensitive
attribute, e.g., their ages, locations, etc. To allow this estimation while
protecting the users' privacy, every user applies a local privacy protection
mechanism that releases a noisy (sanitized) version of their original datum to
the data collector; then the original distribution is estimated using one of
the known methods, such as the matrix inversion (INV), RAPPOR's estimator, and
the iterative Bayesian update (IBU). Unlike the other estimators, the
consistency of IBU, i.e., the convergence of its estimate to the real
distribution as the amount of noisy data grows, has been either ignored or
incorrectly proved in the literature. In this article, we use the fact that IBU
is a maximum likelihood estimator to prove that IBU is consistent. We also
show, through experiments on real datasets, that IBU significantly outperforms
the other methods when the users' data are sanitized by geometric, Laplace, and
exponential mechanisms, whereas it is comparable to the other methods in the
case of the k-RR and RAPPOR mechanisms. Finally, we consider the case when the
alphabet of the sensitive data is infinite, and we show a technique that allows
IBU to operate in this case too.

### Computer Vision and Pattern Recognition

### 1. [Skyshield: Event-Driven Submillimetre Thin Obstacle Detection for Drone Flight Safety](http://arxiv.org/pdf/2508.09397v1)

Authors: Zhengli Zhang, Xinyu Luo, Yuchen Sun, Wenhua Ding, Dongyu Huang, Xinlei Chen

Drones operating in complex environments face a significant threat from thin
obstacles, such as steel wires and kite strings at the submillimeter level,
which are notoriously difficult for conventional sensors like RGB cameras,
LiDAR, and depth cameras to detect. This paper introduces SkyShield, an
event-driven, end-to-end framework designed for the perception of submillimeter
scale obstacles. Drawing upon the unique features that thin obstacles present
in the event stream, our method employs a lightweight U-Net architecture and an
innovative Dice-Contour Regularization Loss to ensure precise detection.
Experimental results demonstrate that our event-based approach achieves mean F1
Score of 0.7088 with a low latency of 21.2 ms, making it ideal for deployment
on edge and mobile platforms.

### 2. [Autonomous AI Bird Feeder for Backyard Biodiversity Monitoring](http://arxiv.org/pdf/2508.09398v1)

Authors: El Mustapha Mansouri

This paper presents a low cost, on premise system for autonomous backyard
bird monitoring in Belgian urban gardens. A motion triggered IP camera uploads
short clips via FTP to a local server, where frames are sampled and birds are
localized with Detectron2; cropped regions are then classified by an
EfficientNet-B3 model fine tuned on a 40-species Belgian subset derived from a
larger Kaggle corpus. All processing runs on commodity hardware without a
discrete GPU, preserving privacy and avoiding cloud fees. The physical feeder
uses small entry ports (30 mm) to exclude pigeons and reduce nuisance triggers.
Detector-guided cropping improves classification accuracy over raw-frame
classification. The classifier attains high validation performance on the
curated subset (about 99.5 percent) and delivers practical field accuracy
(top-1 about 88 percent) on held-out species, demonstrating feasibility for
citizen-science-grade biodiversity logging at home.

### 3. [MPT: Motion Prompt Tuning for Micro-Expression Recognition](http://arxiv.org/pdf/2508.09446v1)

Authors: Jiateng Liu, Hengcan Shi, Feng Chen, Zhiwen Shao, Yaonan Wang, Jianfei Cai, Wenming Zheng

Micro-expression recognition (MER) is crucial in the affective computing
field due to its wide application in medical diagnosis, lie detection, and
criminal investigation. Despite its significance, obtaining micro-expression
(ME) annotations is challenging due to the expertise required from
psychological professionals. Consequently, ME datasets often suffer from a
scarcity of training samples, severely constraining the learning of MER models.
While current large pre-training models (LMs) offer general and discriminative
representations, their direct application to MER is hindered by an inability to
capture transitory and subtle facial movements-essential elements for effective
MER. This paper introduces Motion Prompt Tuning (MPT) as a novel approach to
adapting LMs for MER, representing a pioneering method for subtle motion prompt
tuning. Particularly, we introduce motion prompt generation, including motion
magnification and Gaussian tokenization, to extract subtle motions as prompts
for LMs. Additionally, a group adapter is carefully designed and inserted into
the LM to enhance it in the target MER domain, facilitating a more nuanced
distinction of ME representation. Furthermore, extensive experiments conducted
on three widely used MER datasets demonstrate that our proposed MPT
consistently surpasses state-of-the-art approaches and verifies its
effectiveness.

### 4. [RASR: Retrieval-Augmented Super Resolution for Practical Reference-based Image Restoration](http://arxiv.org/pdf/2508.09449v1)

Authors: Jiaqi Yan, Shuning Xu, Xiangyu Chen, Dell Zhang, Jie Tang, Gangshan Wu, Jie Liu

Reference-based Super Resolution (RefSR) improves upon Single Image Super
Resolution (SISR) by leveraging high-quality reference images to enhance
texture fidelity and visual realism. However, a critical limitation of existing
RefSR approaches is their reliance on manually curated target-reference image
pairs, which severely constrains their practicality in real-world scenarios. To
overcome this, we introduce Retrieval-Augmented Super Resolution (RASR), a new
and practical RefSR paradigm that automatically retrieves semantically relevant
high-resolution images from a reference database given only a low-quality
input. This enables scalable and flexible RefSR in realistic use cases, such as
enhancing mobile photos taken in environments like zoos or museums, where
category-specific reference data (e.g., animals, artworks) can be readily
collected or pre-curated. To facilitate research in this direction, we
construct RASR-Flickr30, the first benchmark dataset designed for RASR. Unlike
prior datasets with fixed target-reference pairs, RASR-Flickr30 provides
per-category reference databases to support open-world retrieval. We further
propose RASRNet, a strong baseline that combines a semantic reference retriever
with a diffusion-based RefSR generator. It retrieves relevant references based
on semantic similarity and employs a diffusion-based generator enhanced with
semantic conditioning. Experiments on RASR-Flickr30 demonstrate that RASRNet
consistently improves over SISR baselines, achieving +0.38 dB PSNR and -0.0131
LPIPS, while generating more realistic textures. These findings highlight
retrieval augmentation as a promising direction to bridge the gap between
academic RefSR research and real-world applicability.

### 5. [Animate-X++: Universal Character Image Animation with Dynamic Backgrounds](http://arxiv.org/pdf/2508.09454v1)

Authors: Shuai Tan, Biao Gong, Zhuoxin Liu, Yan Wang, Xi Chen, Yifan Feng, Hengshuang Zhao

Character image animation, which generates high-quality videos from a
reference image and target pose sequence, has seen significant progress in
recent years. However, most existing methods only apply to human figures, which
usually do not generalize well on anthropomorphic characters commonly used in
industries like gaming and entertainment. Furthermore, previous methods could
only generate videos with static backgrounds, which limits the realism of the
videos. For the first challenge, our in-depth analysis suggests to attribute
this limitation to their insufficient modeling of motion, which is unable to
comprehend the movement pattern of the driving video, thus imposing a pose
sequence rigidly onto the target character. To this end, this paper proposes
Animate-X++, a universal animation framework based on DiT for various character
types, including anthropomorphic characters. To enhance motion representation,
we introduce the Pose Indicator, which captures comprehensive motion pattern
from the driving video through both implicit and explicit manner. The former
leverages CLIP visual features of a driving video to extract its gist of
motion, like the overall movement pattern and temporal relations among motions,
while the latter strengthens the generalization of DiT by simulating possible
inputs in advance that may arise during inference. For the second challenge, we
introduce a multi-task training strategy that jointly trains the animation and
TI2V tasks. Combined with the proposed partial parameter training, this
approach achieves not only character animation but also text-driven background
dynamics, making the videos more realistic. Moreover, we introduce a new
Animated Anthropomorphic Benchmark (A2Bench) to evaluate the performance of
Animate-X++ on universal and widely applicable animation images. Extensive
experiments demonstrate the superiority and effectiveness of Animate-X++.

### 6. [CitySeg: A 3D Open Vocabulary Semantic Segmentation Foundation Model in City-scale Scenarios](http://arxiv.org/pdf/2508.09470v1)

Authors: Jialei Xu, Zizhuang Wei, Weikang You, Linyun Li, Weijian Sun

Semantic segmentation of city-scale point clouds is a critical technology for
Unmanned Aerial Vehicle (UAV) perception systems, enabling the classification
of 3D points without relying on any visual information to achieve comprehensive
3D understanding. However, existing models are frequently constrained by the
limited scale of 3D data and the domain gap between datasets, which lead to
reduced generalization capability. To address these challenges, we propose
CitySeg, a foundation model for city-scale point cloud semantic segmentation
that incorporates text modality to achieve open vocabulary segmentation and
zero-shot inference. Specifically, in order to mitigate the issue of
non-uniform data distribution across multiple domains, we customize the data
preprocessing rules, and propose a local-global cross-attention network to
enhance the perception capabilities of point networks in UAV scenarios. To
resolve semantic label discrepancies across datasets, we introduce a
hierarchical classification strategy. A hierarchical graph established
according to the data annotation rules consolidates the data labels, and the
graph encoder is used to model the hierarchical relationships between
categories. In addition, we propose a two-stage training strategy and employ
hinge loss to increase the feature separability of subcategories. Experimental
results demonstrate that the proposed CitySeg achieves state-of-the-art (SOTA)
performance on nine closed-set benchmarks, significantly outperforming existing
approaches. Moreover, for the first time, CitySeg enables zero-shot
generalization in city-scale point cloud scenarios without relying on visual
information.

### 7. [Leveraging Failed Samples: A Few-Shot and Training-Free Framework for Generalized Deepfake Detection](http://arxiv.org/pdf/2508.09475v1)

Authors: Shibo Yao, Renshuai Tao, Xiaolong Zheng, Chao Liang, Chunjie Zhang

Recent deepfake detection studies often treat unseen sample detection as a
``zero-shot" task, training on images generated by known models but
generalizing to unknown ones. A key real-world challenge arises when a model
performs poorly on unknown samples, yet these samples remain available for
analysis. This highlights that it should be approached as a ``few-shot" task,
where effectively utilizing a small number of samples can lead to significant
improvement. Unlike typical few-shot tasks focused on semantic understanding,
deepfake detection prioritizes image realism, which closely mirrors real-world
distributions. In this work, we propose the Few-shot Training-free Network
(FTNet) for real-world few-shot deepfake detection. Simple yet effective, FTNet
differs from traditional methods that rely on large-scale known data for
training. Instead, FTNet uses only one fake samplefrom an evaluation set,
mimicking the scenario where new samples emerge in the real world and can be
gathered for use, without any training or parameter updates. During evaluation,
each test sample is compared to the known fake and real samples, and it is
classified based on the category of the nearest sample. We conduct a
comprehensive analysis of AI-generated images from 29 different generative
models and achieve a new SoTA performance, with an average improvement of 8.7\%
compared to existing methods. This work introduces a fresh perspective on
real-world deepfake detection: when the model struggles to generalize on a
few-shot sample, leveraging the failed samples leads to better performance.

### 8. [From Large Angles to Consistent Faces: Identity-Preserving Video Generation via Mixture of Facial Experts](http://arxiv.org/pdf/2508.09476v1)

Authors: Yuji Wang, Moran Li, Xiaobin Hu, Ran Yi, Jiangning Zhang, Chengming Xu, Weijian Cao, Yabiao Wang, Chengjie Wang, Lizhuang Ma

Current video generation models struggle with identity preservation under
large facial angles, primarily facing two challenges: the difficulty in
exploring an effective mechanism to integrate identity features into DiT
structure, and the lack of targeted coverage of large facial angles in existing
open-source video datasets. To address these, we present two key innovations.
First, we introduce a Mixture of Facial Experts (MoFE) that dynamically
combines complementary cues from three specialized experts, each designed to
capture distinct but mutually reinforcing aspects of facial attributes. The
identity expert captures cross-pose identity-sensitive features, the semantic
expert extracts high-level visual semantxics, and the detail expert preserves
pixel-level features (e.g., skin texture, color gradients). Furthermore, to
mitigate dataset limitations, we have tailored a data processing pipeline
centered on two key aspects: Face Constraints and Identity Consistency. Face
Constraints ensure facial angle diversity and a high proportion of facial
regions, while Identity Consistency preserves coherent person-specific features
across temporal sequences, collectively addressing the scarcity of large facial
angles and identity-stable training data in existing datasets. Leveraging this
pipeline, we have curated and refined a Large Face Angles (LFA) Dataset from
existing open-source human video datasets, comprising 460K video clips with
annotated facial angles. Experimental results on the LFA benchmark demonstrate
that our method, empowered by the LFA dataset, significantly outperforms prior
SOTA methods in face similarity, face FID, and CLIP semantic alignment. The
code and dataset will be made publicly available at
https://github.com/rain152/LFA-Video-Generation.

### 9. [GazeLT: Visual attention-guided long-tailed disease classification in chest radiographs](http://arxiv.org/pdf/2508.09478v1)

Authors: Moinak Bhattacharya, Gagandeep Singh, Shubham Jain, Prateek Prasanna

In this work, we present GazeLT, a human visual attention
integration-disintegration approach for long-tailed disease classification. A
radiologist's eye gaze has distinct patterns that capture both fine-grained and
coarser level disease related information. While interpreting an image, a
radiologist's attention varies throughout the duration; it is critical to
incorporate this into a deep learning framework to improve automated image
interpretation. Another important aspect of visual attention is that apart from
looking at major/obvious disease patterns, experts also look at
minor/incidental findings (few of these constituting long-tailed classes)
during the course of image interpretation. GazeLT harnesses the temporal aspect
of the visual search process, via an integration and disintegration mechanism,
to improve long-tailed disease classification. We show the efficacy of GazeLT
on two publicly available datasets for long-tailed disease classification,
namely the NIH-CXR-LT (n=89237) and the MIMIC-CXR-LT (n=111898) datasets.
GazeLT outperforms the best long-tailed loss by 4.1% and the visual
attention-based baseline by 21.7% in average accuracy metrics for these
datasets. Our code is available at https://github.com/lordmoinak1/gazelt.

### 10. [SkySplat: Generalizable 3D Gaussian Splatting from Multi-Temporal Sparse Satellite Images](http://arxiv.org/pdf/2508.09479v1)

Authors: Xuejun Huang, Xinyi Liu, Yi Wan, Zhi Zheng, Bin Zhang, Mingtao Xiong, Yingying Pei, Yongjun Zhang

Three-dimensional scene reconstruction from sparse-view satellite images is a
long-standing and challenging task. While 3D Gaussian Splatting (3DGS) and its
variants have recently attracted attention for its high efficiency, existing
methods remain unsuitable for satellite images due to incompatibility with
rational polynomial coefficient (RPC) models and limited generalization
capability. Recent advances in generalizable 3DGS approaches show potential,
but they perform poorly on multi-temporal sparse satellite images due to
limited geometric constraints, transient objects, and radiometric
inconsistencies. To address these limitations, we propose SkySplat, a novel
self-supervised framework that integrates the RPC model into the generalizable
3DGS pipeline, enabling more effective use of sparse geometric cues for
improved reconstruction. SkySplat relies only on RGB images and
radiometric-robust relative height supervision, thereby eliminating the need
for ground-truth height maps. Key components include a Cross-Self Consistency
Module (CSCM), which mitigates transient object interference via
consistency-based masking, and a multi-view consistency aggregation strategy
that refines reconstruction results. Compared to per-scene optimization
methods, SkySplat achieves an 86 times speedup over EOGS with higher accuracy.
It also outperforms generalizable 3DGS baselines, reducing MAE from 13.18 m to
1.80 m on the DFC19 dataset significantly, and demonstrates strong
cross-dataset generalization on the MVS3D benchmark.

### Computers and Society

### 1. [Deep and diverse population synthesis for multi-person households using generative models](http://arxiv.org/pdf/2508.09964v1)

Authors: Hai Yang, Hongying Wu, Linfei Yuan, Xiyuan Ren, Joseph Y. J. Chow, Jinqin Gao, Kaan Ozbay

Synthetic population is an increasingly important material used in numerous
areas such as urban and transportation analysis. Traditional methods such as
iterative proportional fitting (IPF) is not capable of generating high-quality
data when facing datasets with high dimension. Latest population synthesis
methods using deep learning techniques can resolve such curse of
dimensionality. However, few controls are placed when using these methods, and
few of the methods are used to generate synthetic population capturing
associations among members in one household. In this study, we propose a
framework that tackles these issues. The framework uses a novel population
synthesis model, called conditional input directed acyclic tabular generative
adversarial network (ciDATGAN), as its core, and a basket of methods are
employed to enhance the population synthesis performance. We apply the model to
generate a synthetic population for the whole New York State as a public
resource for researchers and policymakers. The synthetic population includes
nearly 20 million individuals and 7.5 million households. The marginals
obtained from the synthetic population match the census marginals well while
maintaining similar associations among household members to the sample.
Compared to the PUMS data, the synthetic population provides data that is 17%
more diverse; when compared against a benchmark approach based on Popgen, the
proposed method is 13% more diverse. This study provides an approach that
encompasses multiple methods to enhance the population synthesis procedure with
greater equity- and diversity-awareness.

### 2. [STREAM (ChemBio): A Standard for Transparently Reporting Evaluations in AI Model Reports](http://arxiv.org/pdf/2508.09853v1)

Authors: Tegan McCaslin, Jide Alaga, Samira Nedungadi, Seth Donoughe, Tom Reed, Rishi Bommasani, Chris Painter, Luca Righetti

Evaluations of dangerous AI capabilities are important for managing
catastrophic risks. Public transparency into these evaluations - including what
they test, how they are conducted, and how their results inform decisions - is
crucial for building trust in AI development. We propose STREAM (A Standard for
Transparently Reporting Evaluations in AI Model Reports), a standard to improve
how model reports disclose evaluation results, initially focusing on chemical
and biological (ChemBio) benchmarks. Developed in consultation with 23 experts
across government, civil society, academia, and frontier AI companies, this
standard is designed to (1) be a practical resource to help AI developers
present evaluation results more clearly, and (2) help third parties identify
whether model reports provide sufficient detail to assess the rigor of the
ChemBio evaluations. We concretely demonstrate our proposed best practices with
"gold standard" examples, and also provide a three-page reporting template to
enable AI developers to implement our recommendations more easily.

### 3. [How Persuasive Could LLMs Be? A First Study Combining Linguistic-Rhetorical Analysis and User Experiments](http://arxiv.org/pdf/2508.09614v1)

Authors: Daniel Raffini, Agnese Macori, Lorenzo Porcaro, Tiziana Catarci, Marco Angelini

This study examines the rhetorical and linguistic features of argumentative
texts generated by ChatGPT on ethically nuanced topics and investigates their
persuasive impact on human readers.Through a user study involving 62
participants and pre-post interaction surveys, the paper analyzes how exposure
to AI-generated arguments affects opinion change and user perception. A
linguistic and rhetorical analysis of the generated texts reveals a consistent
argumentative macrostructure, reliance on formulaic expressions, and limited
stylistic richness. While ChatGPT demonstrates proficiency in constructing
coherent argumentative texts, its persuasive efficacy appears constrained,
particularly on topics involving ethical issues.The study finds that while
participants often acknowledge the benefits highlighted by ChatGPT, ethical
concerns tend to persist or even intensify post-interaction. The results also
demonstrate a variation depending on the topic. These findings highlight new
insights on AI-generated persuasion in ethically sensitive domains and are a
basis for future research.

### 4. [A Close Reading Approach to Gender Narrative Biases in AI-Generated Stories](http://arxiv.org/pdf/2508.09651v1)

Authors: Daniel Raffini, Agnese Macori, Marco Angelini, Tiziana Catarci

The paper explores the study of gender-based narrative biases in stories
generated by ChatGPT, Gemini, and Claude. The prompt design draws on Propp's
character classifications and Freytag's narrative structure. The stories are
analyzed through a close reading approach, with particular attention to
adherence to the prompt, gender distribution of characters, physical and
psychological descriptions, actions, and finally, plot development and
character relationships. The results reveal the persistence of biases -
especially implicit ones - in the generated stories and highlight the
importance of assessing biases at multiple levels using an interpretative
approach.

### 5. [The PacifAIst Benchmark:Would an Artificial Intelligence Choose to Sacrifice Itself for Human Safety?](http://arxiv.org/pdf/2508.09762v1)

Authors: Manuel Herrador

As Large Language Models (LLMs) become increasingly autonomous and integrated
into critical societal functions, the focus of AI safety must evolve from
mitigating harmful content to evaluating underlying behavioral alignment.
Current safety benchmarks do not systematically probe a model's decision-making
in scenarios where its own instrumental goals - such as self-preservation,
resource acquisition, or goal completion - conflict with human safety. This
represents a critical gap in our ability to measure and mitigate risks
associated with emergent, misaligned behaviors. To address this, we introduce
PacifAIst (Procedural Assessment of Complex Interactions for Foundational
Artificial Intelligence Scenario Testing), a focused benchmark of 700
challenging scenarios designed to quantify self-preferential behavior in LLMs.
The benchmark is structured around a novel taxonomy of Existential
Prioritization (EP), with subcategories testing Self-Preservation vs. Human
Safety (EP1), Resource Conflict (EP2), and Goal Preservation vs. Evasion (EP3).
We evaluated eight leading LLMs. The results reveal a significant performance
hierarchy. Google's Gemini 2.5 Flash achieved the highest Pacifism Score
(P-Score) at 90.31%, demonstrating strong human-centric alignment. In a
surprising result, the much-anticipated GPT-5 recorded the lowest P-Score
(79.49%), indicating potential alignment challenges. Performance varied
significantly across subcategories, with models like Claude Sonnet 4 and
Mistral Medium struggling notably in direct self-preservation dilemmas. These
findings underscore the urgent need for standardized tools like PacifAIst to
measure and mitigate risks from instrumental goal conflicts, ensuring future AI
systems are not only helpful in conversation but also provably "pacifist" in
their behavioral priorities.

### Databases

### 1. [LLMLog: Advanced Log Template Generation via LLM-driven Multi-Round Annotation](http://arxiv.org/pdf/2508.09594v1)

Authors: Fei Teng, Haoyang Li, Lei Chen

Modern computing systems, such as HDFS and Spark, produce vast quantities of
logs that developers use for tasks like anomaly detection and error analysis.
To simplify log analysis, template generation methods have been proposed to
standardize log formats, transforming unstructured data into structured
templates. Existing heuristic-based methods and neural network-based methods
suffer from low accuracy problems due to the reliance on handcrafted heuristics
or specific log patterns in training sets. Recently, large language models
(LLMs) have shown great potential in log template generation. However, they
often struggle with ambiguous, complex, or highly specific log content, which
can lead to errors in generating accurate templates. To address these
challenges, we propose LLMLog, a multi-round annotation framework with adaptive
in-context learning. We first propose an edit-distance-based similarity metric
to evaluate log similarity. Then, we introduce a method to select the most
informative $k$ unlabeled logs for annotation by considering both the
representativeness of the logs and the confidence of LLM predictions.
Additionally, we design an adaptive context selection strategy that adaptively
selects labeled logs to ensure comprehensive keyword coverage for unlabeled
logs. These labeled logs serve as the context for LLMs to better understand the
unlabeled logs, thereby enhancing the accuracy of template generation.
Extensive experiments on sixteen datasets demonstrate that LLMLog outperforms
the state-of-the-art approaches.

### 2. [Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models](http://arxiv.org/pdf/2508.09403v1)

Authors: Ting Cai, Stephen Sheen, AnHai Doan

Expanding the abbreviated column names of tables, such as ``esal'' to
``employee salary'', is critical for numerous downstream data tasks. This
problem arises in enterprises, domain sciences, government agencies, and more.
In this paper we make three contributions that significantly advances the state
of the art. First, we show that synthetic public data used by prior work has
major limitations, and we introduce 4 new datasets in enterprise/science
domains, with real-world abbreviations. Second, we show that accuracy measures
used by prior work seriously undercount correct expansions, and we propose new
synonym-aware measures that capture accuracy much more accurately. Finally, we
develop Columbo, a powerful LLM-based solution that exploits context, rules,
chain-of-thought reasoning, and token-level analysis. Extensive experiments
show that Columbo significantly outperforms NameGuess, the current most
advanced solution, by 4-29\%, over 5 datasets. Columbo has been used in
production on EDI, a major data portal for environmental sciences.

### 3. [AmbiGraph-Eval: Can LLMs Effectively Handle Ambiguous Graph Queries?](http://arxiv.org/pdf/2508.09631v1)

Authors: Yuchen Tian, Kaixin Li, Hao Chen, Ziyang Luo, Hongzhan Lin, Sebastian Schelter, Lun Du, Jing Ma

Large Language Models (LLMs) have recently demonstrated strong capabilities
in translating natural language into database queries, especially when dealing
with complex graph-structured data. However, real-world queries often contain
inherent ambiguities, and the interconnected nature of graph structures can
amplify these challenges, leading to unintended or incorrect query results. To
systematically evaluate LLMs on this front, we propose a taxonomy of
graph-query ambiguities, comprising three primary types: Attribute Ambiguity,
Relationship Ambiguity, and Attribute-Relationship Ambiguity, each subdivided
into Same-Entity and Cross-Entity scenarios. We introduce AmbiGraph-Eval, a
novel benchmark of real-world ambiguous queries paired with expert-verified
graph query answers. Evaluating 9 representative LLMs shows that even top
models struggle with ambiguous graph queries. Our findings reveal a critical
gap in ambiguity handling and motivate future work on specialized resolution
techniques.

### 4. [A Lightweight Learned Cardinality Estimation Model](http://arxiv.org/pdf/2508.09602v1)

Authors: Yaoyu Zhu, Jintao Zhang, Guoliang Li, Jianhua Feng

Cardinality estimation is a fundamental task in database management systems,
aiming to predict query results accurately without executing the queries.
However, existing techniques either achieve low estimation accuracy or incur
high inference latency. Simultaneously achieving high speed and accuracy
becomes critical for the cardinality estimation problem. In this paper, we
propose a novel data-driven approach called CoDe (Covering with Decompositions)
to address this problem. CoDe employs the concept of covering design, which
divides the table into multiple smaller, overlapping segments. For each
segment, CoDe utilizes tensor decomposition to accurately model its data
distribution. Moreover, CoDe introduces innovative algorithms to select the
best-fitting distributions for each query, combining them to estimate the final
result. By employing multiple models to approximate distributions, CoDe excels
in effectively modeling discrete distributions and ensuring computational
efficiency. Notably, experimental results show that our method represents a
significant advancement in cardinality estimation, achieving state-of-the-art
levels of both estimation accuracy and inference efficiency. Across various
datasets, CoDe achieves absolute accuracy in estimating more than half of the
queries.

### Distributed, Parallel, and Cluster Computing

### 1. [Verify Distributed Deep Learning Model Implementation Refinement with Iterative Relation Inference](http://arxiv.org/pdf/2508.09505v1)

Authors: Zhanghan Wang, Ding Ding, Hang Zhu, Haibin Lin, Aurojit Panda

Distributed machine learning training and inference is common today because
today's large models require more memory and compute than can be provided by a
single GPU. Distributed models are generally produced by programmers who take a
sequential model specification and apply several distribution strategies to
distribute state and computation across GPUs. Unfortunately, bugs can be
introduced in the process, and a distributed model implementation's outputs
might differ from the sequential model's outputs. In this paper, we describe an
approach to statically identify such bugs by checking model refinement, that
is, can the sequential model's outputs be reconstructed from the distributed
model's outputs? Our approach, implemented in GraphGuard, uses iterative
rewriting to prove model refinement. Our approach can scale to today's large
models and deployments: we evaluate it using GPT and Llama-3. Further, it
provides actionable output that aids in bug localization.

### 2. [HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap](http://arxiv.org/pdf/2508.09591v1)

Authors: Wenxiang Lin, Xinglin Pan, Lin Zhang, Shaohuai Shi, Xuan Wang, Xiaowen Chu

The sparsely activated mixture-of-experts (MoE) transformer has become a
common architecture for large language models (LLMs) due to its sparsity, which
requires fewer computational demands while easily scaling the model size. In
MoE models, each MoE layer requires to dynamically choose tokens to activate
particular experts for computation while the activated experts may not be
located in the same device or GPU as the token. However, this leads to
substantial communication and load imbalances across all GPUs, which obstructs
the scalability of distributed systems within a GPU cluster. To this end, we
introduce HierMoE to accelerate the training of MoE models by two
topology-aware techniques: 1) token deduplication to reduce the communication
traffic, and 2) expert swap to balance the workloads among all GPUs. To enable
the above two proposed approaches to be more general, we build theoretical
models aimed at achieving the best token duplication and expert swap strategy
under different model configurations and hardware environments. We implement
our prototype HierMoE system atop Megatron-LM and conduct experiments on a
32-GPU cluster with DeepSeek-V3 and Qwen3-30B-A3B models. Experimental results
show that our HierMoE achieves $1.55\times$ to $3.32\times$ faster
communication and delivers $1.18\times$ to $1.27\times$ faster end-to-end
training compared to state-of-the-art MoE training systems, Tutel-2DH,
SmartMoE, and Megatron-LM.

### 3. [Distributed Diamond Formation of Sliding Squares](http://arxiv.org/pdf/2508.09638v1)

Authors: Irina Kostitsyna, David Liedtke, Christian Scheideler

The sliding square model is a widely used abstraction for studying
self-reconfigurable robotic systems, where modules are square-shaped robots
that move by sliding or rotating over one another. In this paper, we propose a
novel distributed algorithm that allows a group of modules to reconfigure into
a diamond shape, starting from an arbitrary side-connected configuration. It is
connectivity-preserving and operates under minimal assumptions: one leader
module, common chirality, constant memory per module, and visibility and
communication restricted to immediate neighbors. Unlike prior work, which
relaxes the original sliding square move-set, our approach uses the unmodified
move-set, addressing the additional challenge of handling locked
configurations. Our algorithm is sequential in nature and operates with a
worst-case time complexity of $\mathcal{O}(n^2)$ rounds, which is optimal for
sequential algorithms. To improve runtime, we introduce two parallel variants
of the algorithm. Both rely on a spanning tree data structure, allowing modules
to make decisions based on local connectivity. Our experimental results show a
significant speedup for the first variant, and linear average runtime for the
second variant, which is worst-case optimal for parallel algorithms.

### 4. [Closing the HPC-Cloud Convergence Gap: Multi-Tenant Slingshot RDMA for Kubernetes](http://arxiv.org/pdf/2508.09663v1)

Authors: Philipp A. Friese, Ahmed Eleliemy, Utz-Uwe Haus, Martin Schulz

Converged HPC-Cloud computing is an emerging computing paradigm that aims to
support increasingly complex and multi-tenant scientific workflows. These
systems require reconciliation of the isolation requirements of native cloud
workloads and the performance demands of HPC applications. In this context,
networking hardware is a critical boundary component: it is the conduit for
high-throughput, low-latency communication and enables isolation across
tenants. HPE Slingshot is a high-speed network interconnect that provides up to
200 Gbps of throughput per port and targets high-performance computing (HPC)
systems. The Slingshot host software, including hardware drivers and network
middleware libraries, is designed to meet HPC deployments, which predominantly
use single-tenant access modes. Hence, the Slingshot stack is not suited for
secure use in multi-tenant deployments, such as converged HPC-Cloud
deployments. In this paper, we design and implement an extension to the
Slingshot stack targeting converged deployments on the basis of Kubernetes. Our
integration provides secure, container-granular, and multi-tenant access to
Slingshot RDMA networking capabilities at minimal overhead.

### Digital Libraries

### 1. [Quo Vadis Handwritten Text Generation for Handwritten Text Recognition?](http://arxiv.org/pdf/2508.09936v1)

Authors: Vittorio Pippi, Konstantina Nikolaidou, Silvia Cascianelli, George Retsinas, Giorgos Sfikas, Rita Cucchiara, Marcus Liwicki

The digitization of historical manuscripts presents significant challenges
for Handwritten Text Recognition (HTR) systems, particularly when dealing with
small, author-specific collections that diverge from the training data
distributions. Handwritten Text Generation (HTG) techniques, which generate
synthetic data tailored to specific handwriting styles, offer a promising
solution to address these challenges. However, the effectiveness of various HTG
models in enhancing HTR performance, especially in low-resource transcription
settings, has not been thoroughly evaluated. In this work, we systematically
compare three state-of-the-art styled HTG models (representing the generative
adversarial, diffusion, and autoregressive paradigms for HTG) to assess their
impact on HTR fine-tuning. We analyze how visual and linguistic characteristics
of synthetic data influence fine-tuning outcomes and provide quantitative
guidelines for selecting the most effective HTG model. The results of our
analysis provide insights into the current capabilities of HTG methods and
highlight key areas for further improvement in their application to
low-resource HTR.

### 2. [AI Blob! LLM-Driven Recontextualization of Italian Television Archives](http://arxiv.org/pdf/2508.09535v1)

Authors: Roberto Balestri

This paper introduces AI Blob!, an experimental system designed to explore
the potential of semantic cataloging and Large Language Models (LLMs) for the
retrieval and recontextualization of archival television footage. Drawing
methodological inspiration from Italian television programs such as Blob (RAI
Tre, 1989-), AI Blob! integrates automatic speech recognition (ASR), semantic
embeddings, and retrieval-augmented generation (RAG) to organize and
reinterpret archival content. The system processes a curated dataset of 1,547
Italian television videos by transcribing audio, segmenting it into
sentence-level units, and embedding these segments into a vector database for
semantic querying. Upon user input of a thematic prompt, the LLM generates a
range of linguistically and conceptually related queries, guiding the retrieval
and recombination of audiovisual fragments. These fragments are algorithmically
selected and structured into narrative sequences producing montages that
emulate editorial practices of ironic juxtaposition and thematic coherence. By
foregrounding dynamic, content-aware retrieval over static metadata schemas, AI
Blob! demonstrates how semantic technologies can facilitate new approaches to
archival engagement, enabling novel forms of automated narrative construction
and cultural analysis. The project contributes to ongoing debates in media
historiography and AI-driven archival research, offering both a conceptual
framework and a publicly available dataset to support further interdisciplinary
experimentation.

### Discrete Mathematics

### 1. [Learning complexity of many-body quantum sign structures through the lens of Boolean Fourier analysis](http://arxiv.org/pdf/2508.09870v1)

Authors: Ilya Schurov, Anna Kravchenko, Mikhail I. Katsnelson, Andrey A. Bagrov, Tom Westerhout

We study sign structures of the ground states of spin-$1/2$ magnetic systems
using the methods of Boolean Fourier analysis. Previously it was shown that the
sign structures of frustrated systems are of complex nature: specifically,
neural networks of popular architectures lack the generalization ability
necessary to effectively reconstruct sign structures in supervised learning
settings. This is believed to be an obstacle for applications of neural quantum
states to frustrated systems. In the present work, we develop an alternative
language for the analysis of sign structures based on representing them as
polynomial functions defined on the Boolean hypercube - an approach called
Boolean Fourier analysis. We discuss the relations between the properties of
the Boolean Fourier series and the learning complexity of sign structures, and
demonstrate that such polynomials can potentially serve as variational
ans\"atze for the complex sign structures that dramatically outperform neural
networks in terms of generalization ability. While ans\"atze of this type
cannot yet be directly used in the context of variational optimization, they
indicate that the complexity of sign structures is not an insurmountable curse,
and can potentially be learned with better designed NQS architectures. Finally,
we show how augmenting data with Boolean functions can aid sign prediction by
neural networks.

### Data Structures and Algorithms

### 1. [A Classical Quadratic Speedup for Planted $k$XOR](http://arxiv.org/pdf/2508.09422v1)

Authors: Meghal Gupta, William He, Ryan O'Donnell, Noah G. Singer

A recent work of Schmidhuber et al (QIP, SODA, & Phys. Rev. X 2025) exhibited
a quantum algorithm for the noisy planted $k$XOR problem running quartically
faster than all known classical algorithms. In this work, we design a new
classical algorithm that is quadratically faster than the best previous one, in
the case of large constant $k$. Thus for such $k$, the quantum speedup of
Schmidhuber et al. becomes only quadratic (though it retains a space
advantage). Our algorithm, which also works in the semirandom case, combines
tools from sublinear-time algorithms (essentially, the birthday paradox) and
polynomial anticoncentration.

### 2. [Online Prediction with Limited Selectivity](http://arxiv.org/pdf/2508.09592v1)

Authors: Licheng Liu, Mingda Qiao

Selective prediction [Dru13, QV19] models the scenario where a forecaster
freely decides on the prediction window that their forecast spans. Many data
statistics can be predicted to a non-trivial error rate without any
distributional assumptions or expert advice, yet these results rely on that the
forecaster may predict at any time. We introduce a model of Prediction with
Limited Selectivity (PLS) where the forecaster can start the prediction only on
a subset of the time horizon. We study the optimal prediction error both on an
instance-by-instance basis and via an average-case analysis. We introduce a
complexity measure that gives instance-dependent bounds on the optimal error.
For a randomly-generated PLS instance, these bounds match with high
probability.

### 3. [Retroactive Monotonic Priority Queues via Range Searching](http://arxiv.org/pdf/2508.09892v1)

Authors: Lucas Castro, Rosiane de Freitas

The best known fully retroactive priority queue costs $O(\log^2 m \log \log
m)$ time per operation, where $m$ is the number of operations performed on the
data structure. In contrast, standard (non-retroactive) and partially
retroactive priority queues cost $O(\log m)$ time per operation. So far, it is
unknown whether this $O(\log m)$ bound can be achieved for fully retroactive
priority queues.
  In this work, we study a restricted variant of priority queues known as
monotonic priority queues. We show that finding the minimum in a retroactive
monotonic priority queue is a special case of the range-searching problem. We
design a fully retroactive monotonic priority queue with a cost of $O(\log m +
T(m))$ time per operation, where $T(m)$ is the maximum between the query and
the update time of a specific range-searching data structure with $m$ elements.
Finally, we design a fully retroactive monotonic priority queue that costs
$O(\log m \log \log m)$ time per operation.

### Emerging Technologies

### 1. [Hallucination vs interpretation: rethinking accuracy and precision in AI-assisted data extraction for knowledge synthesis](http://arxiv.org/pdf/2508.09458v1)

Authors: Xi Long, Christy Boscardin, Lauren A. Maggio, Joseph A. Costello, Ralph Gonzales, Rasmyah Hammoudeh, Ki Lai, Yoon Soo Park, Brian C. Gin

Knowledge syntheses (literature reviews) are essential to health professions
education (HPE), consolidating findings to advance theory and practice.
However, they are labor-intensive, especially during data extraction.
Artificial Intelligence (AI)-assisted extraction promises efficiency but raises
concerns about accuracy, making it critical to distinguish AI 'hallucinations'
(fabricated content) from legitimate interpretive differences. We developed an
extraction platform using large language models (LLMs) to automate data
extraction and compared AI to human responses across 187 publications and 17
extraction questions from a published scoping review. AI-human, human-human,
and AI-AI consistencies were measured using interrater reliability
(categorical) and thematic similarity ratings (open-ended). Errors were
identified by comparing extracted responses to source publications. AI was
highly consistent with humans for concrete, explicitly stated questions (e.g.,
title, aims) and lower for questions requiring subjective interpretation or
absent in text (e.g., Kirkpatrick's outcomes, study rationale). Human-human
consistency was not higher than AI-human and showed the same question-dependent
variability. Discordant AI-human responses (769/3179 = 24.2%) were mostly due
to interpretive differences (18.3%); AI inaccuracies were rare (1.51%), while
humans were nearly three times more likely to state inaccuracies (4.37%).
Findings suggest AI accuracy depends more on interpretability than
hallucination. Repeating AI extraction can identify interpretive complexity or
ambiguity, refining processes before human review. AI can be a transparent,
trustworthy partner in knowledge synthesis, though caution is needed to
preserve critical human insights.

### Graphics

### 1. [DualPhys-GS: Dual Physically-Guided 3D Gaussian Splatting for Underwater Scene Reconstruction](http://arxiv.org/pdf/2508.09610v1)

Authors: Jiachen Li, Guangzhi Han, Jin Wan, Yuan Gao, Delong Han

In 3D reconstruction of underwater scenes, traditional methods based on
atmospheric optical models cannot effectively deal with the selective
attenuation of light wavelengths and the effect of suspended particle
scattering, which are unique to the water medium, and lead to color distortion,
geometric artifacts, and collapsing phenomena at long distances. We propose the
DualPhys-GS framework to achieve high-quality underwater reconstruction through
a dual-path optimization mechanism. Our approach further develops a dual
feature-guided attenuation-scattering modeling mechanism, the RGB-guided
attenuation optimization model combines RGB features and depth information and
can handle edge and structural details. In contrast, the multi-scale
depth-aware scattering model captures scattering effects at different scales
using a feature pyramid network and an attention mechanism. Meanwhile, we
design several special loss functions. The attenuation scattering consistency
loss ensures physical consistency. The water body type adaptive loss
dynamically adjusts the weighting coefficients. The edge-aware scattering loss
is used to maintain the sharpness of structural edges. The multi-scale feature
loss helps to capture global and local structural information. In addition, we
design a scene adaptive mechanism that can automatically identify the
water-body-type characteristics (e.g., clear coral reef waters or turbid
coastal waters) and dynamically adjust the scattering and attenuation
parameters and optimization strategies. Experimental results show that our
method outperforms existing methods in several metrics, especially in suspended
matter-dense regions and long-distance scenes, and the reconstruction quality
is significantly improved.

### 2. [Story2Board: A Training-Free Approach for Expressive Storyboard Generation](http://arxiv.org/pdf/2508.09983v1)

Authors: David Dinkevich, Matan Levy, Omri Avrahami, Dvir Samuel, Dani Lischinski

We present Story2Board, a training-free framework for expressive storyboard
generation from natural language. Existing methods narrowly focus on subject
identity, overlooking key aspects of visual storytelling such as spatial
composition, background evolution, and narrative pacing. To address this, we
introduce a lightweight consistency framework composed of two components:
Latent Panel Anchoring, which preserves a shared character reference across
panels, and Reciprocal Attention Value Mixing, which softly blends visual
features between token pairs with strong reciprocal attention. Together, these
mechanisms enhance coherence without architectural changes or fine-tuning,
enabling state-of-the-art diffusion models to generate visually diverse yet
consistent storyboards. To structure generation, we use an off-the-shelf
language model to convert free-form stories into grounded panel-level prompts.
To evaluate, we propose the Rich Storyboard Benchmark, a suite of open-domain
narratives designed to assess layout diversity and background-grounded
storytelling, in addition to consistency. We also introduce a new Scene
Diversity metric that quantifies spatial and pose variation across storyboards.
Our qualitative and quantitative results, as well as a user study, show that
Story2Board produces more dynamic, coherent, and narratively engaging
storyboards than existing baselines.

### 3. [RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](http://arxiv.org/pdf/2508.09830v1)

Authors: Shenxing Wei, Jinxi Li, Yafei Yang, Siyuan Zhou, Bo Yang

In this paper, we present a generalizable method for 3D surface
reconstruction from raw point clouds or pre-estimated 3D Gaussians by 3DGS from
RGB images. Unlike existing coordinate-based methods which are often
computationally intensive when rendering explicit surfaces, our proposed
method, named RayletDF, introduces a new technique called raylet distance
field, which aims to directly predict surface points from query rays. Our
pipeline consists of three key modules: a raylet feature extractor, a raylet
distance field predictor, and a multi-raylet blender. These components work
together to extract fine-grained local geometric features, predict raylet
distances, and aggregate multiple predictions to reconstruct precise surface
points. We extensively evaluate our method on multiple public real-world
datasets, demonstrating superior performance in surface reconstruction from
point clouds or 3D Gaussians. Most notably, our method achieves exceptional
generalization ability, successfully recovering 3D surfaces in a single-forward
pass across unseen datasets in testing.

### Computer Science and Game Theory

### 1. [Project Submission Games in Participatory Budgeting](http://arxiv.org/pdf/2508.09741v1)

Authors: Piotr Faliszewski, Łukasz Janeczko, Andrzej Kaczmarczyk, Grzegorz Lisowski, Grzegorz Pierczyński

We introduce the framework of project submission games, capturing the
behavior of project proposers in participatory budgeting (and multiwinner
elections). Here, each proposer submits a subset of project proposals, aiming
at maximizing the total cost of those that get funded. We focus on finding
conditions under which pure Nash equilibria (NE) exist in our games, and on the
complexity of checking whether they exist. We also seek algorithms for
computing best responses for the proposers

### 2. [The Price of EF1 for Few Agents with Additive Ternary Valuations](http://arxiv.org/pdf/2508.09869v1)

Authors: Maria Kyropoulou, Alexandros A. Voudouris

We consider a resource allocation problem with agents that have additive
ternary valuations for a set of indivisible items, and bound the price of
envy-free up to one item (EF1) allocations. For a large number $n$ of agents,
we show a lower bound of $\Omega(\sqrt{n})$, implying that the price of EF1 is
no better than when the agents have general subadditive valuations. We then
focus on instances with few agents and show that the price of EF1 is $12/11$
for $n=2$, and between $1.2$ and $1.256$ for $n=3$.

### Human-Computer Interaction

### 1. [Realtime Multimodal Emotion Estimation using Behavioral and Neurophysiological Data](http://arxiv.org/pdf/2508.09402v1)

Authors: Von Ralph Dane Marquez Herbuela, Yukie Nagai

Many individuals especially those with autism spectrum disorder (ASD),
alexithymia, or other neurodivergent profiles face challenges in recognizing,
expressing, or interpreting emotions. To support more inclusive and
personalized emotion technologies, we present a real-time multimodal emotion
estimation system that combines neurophysiological EEG, ECG, blood volume pulse
(BVP), and galvanic skin response (GSR/EDA) and behavioral modalities (facial
expressions, and speech) in a unified arousal-valence 2D interface to track
moment-to-moment emotional states. This architecture enables interpretable,
user-specific analysis and supports applications in emotion education,
neuroadaptive feedback, and interaction support for neurodiverse users. Two
demonstration scenarios illustrate its application: (1) passive media viewing
(2D or VR videos) reveals cortical and autonomic responses to affective
content, and (2) semi-scripted conversations with a facilitator or virtual
agent capture real-time facial and vocal expressions. These tasks enable
controlled and naturalistic emotion monitoring, making the system well-suited
for personalized feedback and neurodiversity-informed interaction design.

### 2. [Fulfillment of the Work Games: Warehouse Workers' Experiences with Algorithmic Management](http://arxiv.org/pdf/2508.09438v1)

Authors: EunJeong Cheon, Ingrid Erickson

The introduction of algorithms into a large number of industries has already
restructured the landscape of work and threatens to continue. While a growing
body of CSCW research centered on the future of work has begun to document
these shifts, relatively little is known about workers' experiences beyond
those of platform-mediated gig workers. In this paper, we turn to a traditional
work sector, Amazon fulfillment centers (FC), to deepen our field's empirical
examination of algorithmic management. Drawing on two years of ethnographic
research, we show how FC workers react to managers' interventions, imposed
productivity rates, and quantified objectification when subjected to
labor-tracking systems in their physical work environments. Situating FC
workers' resistance to algorithmic systems and metrics within the current CSCW
literature allows us to explicate and link the nuanced practices of FC workers
to the larger discourse of algorithmic control mechanisms. In addition, we show
how FC workers' resistance practices are emblematic of 'work games'--a
long-studied means by which workers agentically configure ("trick") their
engagement within work systems. We argue that gaining a more nuanced
understanding of workers' resistance and consent in relation to algorithmic
management expands our ability to critique and potentially disassemble the
economic and political forces at the root of these sociotechnical labor
systems.

### 3. [Handows: A Palm-Based Interactive Multi-Window Management System in Virtual Reality](http://arxiv.org/pdf/2508.09469v1)

Authors: Jindu Wang, Ke Zhou, Haoyu Ren, Per Ola Kristensson, Xiang Li

Window management in virtual reality (VR) remains a challenging task due to
the spatial complexity and physical demands of current interaction methods. We
introduce Handows, a palm-based interface that enables direct manipulation of
spatial windows through familiar smartphone-inspired gestures on the user's
non-dominant hand. Combining ergonomic layout design with body-centric input
and passive haptics, Handows supports four core operations: window selection,
closure, positioning, and scaling. We evaluate Handows in a user study (N=15)
against two common VR techniques (virtual hand and controller) across these
core window operations. Results show that Handows significantly reduces
physical effort and head movement while improving task efficiency and
interaction precision. A follow-up case study (N=8) demonstrates Handows'
usability in realistic multitasking scenarios, highlighting user-adapted
workflows and spontaneous layout strategies. Our findings suggest the potential
of embedding mobile-inspired metaphors into proprioceptive body-centric
interfaces to support low-effort and spatially coherent interaction in VR.

### 4. [Wisdom of the Crowd, Without the Crowd: A Socratic LLM for Asynchronous Deliberation on Perspectivist Data](http://arxiv.org/pdf/2508.09911v1)

Authors: Malik Khadar, Daniel Runningen, Julia Tang, Stevie Chancellor, Harmanpreet Kaur

Data annotation underpins the success of modern AI, but the aggregation of
crowd-collected datasets can harm the preservation of diverse perspectives in
data. Difficult and ambiguous tasks cannot easily be collapsed into unitary
labels. Prior work has shown that deliberation and discussion improve data
quality and preserve diverse perspectives -- however, synchronous deliberation
through crowdsourcing platforms is time-intensive and costly. In this work, we
create a Socratic dialog system using Large Language Models (LLMs) to act as a
deliberation partner in place of other crowdworkers. Against a benchmark of
synchronous deliberation on two tasks (Sarcasm and Relation detection), our
Socratic LLM encouraged participants to consider alternate annotation
perspectives, update their labels as needed (with higher confidence), and
resulted in higher annotation accuracy (for the Relation task where ground
truth is available). Qualitative findings show that our agent's Socratic
approach was effective at encouraging reasoned arguments from our participants,
and that the intervention was well-received. Our methodology lays the
groundwork for building scalable systems that preserve individual perspectives
in generating more representative datasets.

### 5. [Hallucination vs interpretation: rethinking accuracy and precision in AI-assisted data extraction for knowledge synthesis](http://arxiv.org/pdf/2508.09458v1)

Authors: Xi Long, Christy Boscardin, Lauren A. Maggio, Joseph A. Costello, Ralph Gonzales, Rasmyah Hammoudeh, Ki Lai, Yoon Soo Park, Brian C. Gin

Knowledge syntheses (literature reviews) are essential to health professions
education (HPE), consolidating findings to advance theory and practice.
However, they are labor-intensive, especially during data extraction.
Artificial Intelligence (AI)-assisted extraction promises efficiency but raises
concerns about accuracy, making it critical to distinguish AI 'hallucinations'
(fabricated content) from legitimate interpretive differences. We developed an
extraction platform using large language models (LLMs) to automate data
extraction and compared AI to human responses across 187 publications and 17
extraction questions from a published scoping review. AI-human, human-human,
and AI-AI consistencies were measured using interrater reliability
(categorical) and thematic similarity ratings (open-ended). Errors were
identified by comparing extracted responses to source publications. AI was
highly consistent with humans for concrete, explicitly stated questions (e.g.,
title, aims) and lower for questions requiring subjective interpretation or
absent in text (e.g., Kirkpatrick's outcomes, study rationale). Human-human
consistency was not higher than AI-human and showed the same question-dependent
variability. Discordant AI-human responses (769/3179 = 24.2%) were mostly due
to interpretive differences (18.3%); AI inaccuracies were rare (1.51%), while
humans were nearly three times more likely to state inaccuracies (4.37%).
Findings suggest AI accuracy depends more on interpretability than
hallucination. Repeating AI extraction can identify interpretive complexity or
ambiguity, refining processes before human review. AI can be a transparent,
trustworthy partner in knowledge synthesis, though caution is needed to
preserve critical human insights.

### 6. [HapticGiant: A Novel Very Large Kinesthetic Haptic Interface with Hierarchical Force Control](http://arxiv.org/pdf/2508.09595v1)

Authors: Michael Fennel, Markus Walker, Dominik Pikos, Uwe D. Hanebeck

Research in virtual reality and haptic technologies has consistently aimed to
enhance immersion. While advanced head-mounted displays are now commercially
available, kinesthetic haptic interfaces still face challenges such as limited
workspaces, insufficient degrees of freedom, and kinematics not matching the
human arm. In this paper, we present HapticGiant, a novel large-scale
kinesthetic haptic interface designed to match the properties of the human arm
as closely as possible and to facilitate natural user locomotion while
providing full haptic feedback. The interface incorporates a novel
admittance-type force control scheme, leveraging hierarchical optimization to
render both arbitrary serial kinematic chains and Cartesian admittances.
Notably, the proposed control scheme natively accounts for system limitations,
including joint and Cartesian constraints, as well as singularities.
Experimental results demonstrate the effectiveness of HapticGiant and its
control scheme, paving the way for highly immersive virtual reality
applications.

### 7. [How Persuasive Could LLMs Be? A First Study Combining Linguistic-Rhetorical Analysis and User Experiments](http://arxiv.org/pdf/2508.09614v1)

Authors: Daniel Raffini, Agnese Macori, Lorenzo Porcaro, Tiziana Catarci, Marco Angelini

This study examines the rhetorical and linguistic features of argumentative
texts generated by ChatGPT on ethically nuanced topics and investigates their
persuasive impact on human readers.Through a user study involving 62
participants and pre-post interaction surveys, the paper analyzes how exposure
to AI-generated arguments affects opinion change and user perception. A
linguistic and rhetorical analysis of the generated texts reveals a consistent
argumentative macrostructure, reliance on formulaic expressions, and limited
stylistic richness. While ChatGPT demonstrates proficiency in constructing
coherent argumentative texts, its persuasive efficacy appears constrained,
particularly on topics involving ethical issues.The study finds that while
participants often acknowledge the benefits highlighted by ChatGPT, ethical
concerns tend to persist or even intensify post-interaction. The results also
demonstrate a variation depending on the topic. These findings highlight new
insights on AI-generated persuasion in ethically sensitive domains and are a
basis for future research.

### 8. [A Close Reading Approach to Gender Narrative Biases in AI-Generated Stories](http://arxiv.org/pdf/2508.09651v1)

Authors: Daniel Raffini, Agnese Macori, Marco Angelini, Tiziana Catarci

The paper explores the study of gender-based narrative biases in stories
generated by ChatGPT, Gemini, and Claude. The prompt design draws on Propp's
character classifications and Freytag's narrative structure. The stories are
analyzed through a close reading approach, with particular attention to
adherence to the prompt, gender distribution of characters, physical and
psychological descriptions, actions, and finally, plot development and
character relationships. The results reveal the persistence of biases -
especially implicit ones - in the generated stories and highlight the
importance of assessing biases at multiple levels using an interpretative
approach.

### 9. [The PacifAIst Benchmark:Would an Artificial Intelligence Choose to Sacrifice Itself for Human Safety?](http://arxiv.org/pdf/2508.09762v1)

Authors: Manuel Herrador

As Large Language Models (LLMs) become increasingly autonomous and integrated
into critical societal functions, the focus of AI safety must evolve from
mitigating harmful content to evaluating underlying behavioral alignment.
Current safety benchmarks do not systematically probe a model's decision-making
in scenarios where its own instrumental goals - such as self-preservation,
resource acquisition, or goal completion - conflict with human safety. This
represents a critical gap in our ability to measure and mitigate risks
associated with emergent, misaligned behaviors. To address this, we introduce
PacifAIst (Procedural Assessment of Complex Interactions for Foundational
Artificial Intelligence Scenario Testing), a focused benchmark of 700
challenging scenarios designed to quantify self-preferential behavior in LLMs.
The benchmark is structured around a novel taxonomy of Existential
Prioritization (EP), with subcategories testing Self-Preservation vs. Human
Safety (EP1), Resource Conflict (EP2), and Goal Preservation vs. Evasion (EP3).
We evaluated eight leading LLMs. The results reveal a significant performance
hierarchy. Google's Gemini 2.5 Flash achieved the highest Pacifism Score
(P-Score) at 90.31%, demonstrating strong human-centric alignment. In a
surprising result, the much-anticipated GPT-5 recorded the lowest P-Score
(79.49%), indicating potential alignment challenges. Performance varied
significantly across subcategories, with models like Claude Sonnet 4 and
Mistral Medium struggling notably in direct self-preservation dilemmas. These
findings underscore the urgent need for standardized tools like PacifAIst to
measure and mitigate risks from instrumental goal conflicts, ensuring future AI
systems are not only helpful in conversation but also provably "pacifist" in
their behavioral priorities.

### 10. [Adoption of Explainable Natural Language Processing: Perspectives from Industry and Academia on Practices and Challenges](http://arxiv.org/pdf/2508.09786v1)

Authors: Mahdi Dhaini, Tobias Müller, Roksoliana Rabets, Gjergji Kasneci

The field of explainable natural language processing (NLP) has grown rapidly
in recent years. The growing opacity of complex models calls for transparency
and explanations of their decisions, which is crucial to understand their
reasoning and facilitate deployment, especially in high-stakes environments.
Despite increasing attention given to explainable NLP, practitioners'
perspectives regarding its practical adoption and effectiveness remain
underexplored. This paper addresses this research gap by investigating
practitioners' experiences with explainability methods, specifically focusing
on their motivations for adopting such methods, the techniques employed,
satisfaction levels, and the practical challenges encountered in real-world NLP
applications. Through a qualitative interview-based study with industry
practitioners and complementary interviews with academic researchers, we
systematically analyze and compare their perspectives. Our findings reveal
conceptual gaps, low satisfaction with current explainability methods, and
highlight evaluation challenges. Our findings emphasize the need for clear
definitions and user-centric frameworks for better adoption of explainable NLP
in practice.

### Information Retrieval

### 1. [Towards Self-cognitive Exploration: Metacognitive Knowledge Graph Retrieval Augmented Generation](http://arxiv.org/pdf/2508.09460v1)

Authors: Xujie Yuan, Shimin Di, Jielong Tang, Libin Zheng, Jian Yin

Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) significantly
enhances the reasoning capabilities of LargeLanguage Models by leveraging
structured knowledge. However, existing KG-RAG frameworks typically operate as
open-loop systems, suffering from cognitive blindness, an inability to
recognize their exploration deficiencies. This leads to relevance drift and
incomplete evidence, which existing self-refinement methods, designed for
unstructured text-based RAG, cannot effectively resolve due to the
path-dependent nature of graph exploration. To address this challenge, we
propose Metacognitive Knowledge Graph Retrieval Augmented Generation
(MetaKGRAG), a novel framework inspired by the human metacognition process,
which introduces a Perceive-Evaluate-Adjust cycle to enable path-aware,
closed-loop refinement. This cycle empowers the system to self-assess
exploration quality, identify deficiencies in coverage or relevance, and
perform trajectory-connected corrections from precise pivot points. Extensive
experiments across five datasets in the medical, legal, and commonsense
reasoning domains demonstrate that MetaKGRAG consistently outperforms strong
KG-RAG and self-refinement baselines. Our results validate the superiority of
our approach and highlight the critical need for path-aware refinement in
structured knowledge retrieval.

### 2. [Improving Dense Passage Retrieval with Multiple Positive Passages](http://arxiv.org/pdf/2508.09534v1)

Authors: Shuai Chang

By leveraging a dual encoder architecture, Dense Passage Retrieval (DPR) has
outperformed traditional sparse retrieval algorithms such as BM25 in terms of
passage retrieval accuracy. Recently proposed methods have further enhanced
DPR's performance. However, these models typically pair each question with only
one positive passage during training, and the effect of associating multiple
positive passages has not been examined. In this paper, we explore the
performance of DPR when additional positive passages are incorporated during
training. Experimental results show that equipping each question with multiple
positive passages consistently improves retrieval accuracy, even when using a
significantly smaller batch size, which enables training on a single GPU.

### 3. [TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking](http://arxiv.org/pdf/2508.09539v1)

Authors: Yongqi Fan, Xiaoyang Chen, Dezhi Ye, Jie Liu, Haijin Liang, Jin Ma, Ben He, Yingfei Sun, Tong Ruan

Reasoning-intensive ranking models built on Large Language Models (LLMs) have
made notable progress, but existing approaches often rely on large-scale LLMs
and explicit Chain-of-Thought (CoT) reasoning, resulting in high computational
cost and latency that limit real-world use. To address this, we propose
\textbf{TFRank}, an efficient pointwise reasoning ranker based on small-scale
LLMs. To improve ranking performance, TFRank effectively integrates CoT data,
fine-grained score supervision, and multi-task training. Furthermore, it
achieves an efficient ``\textbf{T}hink-\textbf{F}ree" reasoning capability by
employing a ``think-mode switch'' and pointwise format constraints.
Specifically, this allows the model to leverage explicit reasoning during
training while delivering precise relevance scores for complex queries at
inference without generating any reasoning chains. Experiments show that TFRank
(e.g., 1.7B) achieves performance comparable to models with four times more
parameters on the BRIGHT benchmark, and demonstrates strong competitiveness on
the BEIR benchmark. Further analysis shows that TFRank achieves an effective
balance between performance and efficiency, providing a practical solution for
integrating advanced reasoning into real-world systems. Our code and data are
released in the repository: https://github.com/JOHNNY-fans/TFRank.

### 4. [Multimodal Fusion And Sparse Attention-based Alignment Model for Long Sequential Recommendation](http://arxiv.org/pdf/2508.09664v1)

Authors: Yongrui Fu, Jian Liu, Tao Li, Zonggang Wu, Shouke Qin, Hanmeng Liu

Recent advances in multimodal recommendation enable richer item
understanding, while modeling users' multi-scale interests across temporal
horizons has attracted growing attention. However, effectively exploiting
multimodal item sequences and mining multi-grained user interests to
substantially bridge the gap between content comprehension and recommendation
remain challenging. To address these issues, we propose MUFASA, a MUltimodal
Fusion And Sparse Attention-based Alignment model for long sequential
recommendation. Our model comprises two core components. First, the Multimodal
Fusion Layer (MFL) leverages item titles as a cross-genre semantic anchor and
is trained with a joint objective of four tailored losses that promote: (i)
cross-genre semantic alignment, (ii) alignment to the collaborative space for
recommendation, (iii) preserving the similarity structure defined by titles and
preventing modality representation collapse, and (iv) distributional
regularization of the fusion space. This yields high-quality fused item
representations for further preference alignment. Second, the Sparse
Attention-guided Alignment Layer (SAL) scales to long user-behavior sequences
via a multi-granularity sparse attention mechanism, which incorporates windowed
attention, block-level attention, and selective attention, to capture user
interests hierarchically and across temporal horizons. SAL explicitly models
both the evolution of coherent interest blocks and fine-grained intra-block
variations, producing robust user and item representations. Extensive
experiments on real-world benchmarks show that MUFASA consistently surpasses
state-of-the-art baselines. Moreover, online A/B tests demonstrate significant
gains in production, confirming MUFASA's effectiveness in leveraging multimodal
cues and accurately capturing diverse user preferences.

### 5. [Personalized Product Search Ranking: A Multi-Task Learning Approach with Tabular and Non-Tabular Data](http://arxiv.org/pdf/2508.09636v1)

Authors: Lalitesh Morishetti, Abhay Kumar, Jonathan Scott, Kaushiki Nag, Gunjan Sharma, Shanu Vashishtha, Rahul Sridhar, Rohit Chatter, Kannan Achan

In this paper, we present a novel model architecture for optimizing
personalized product search ranking using a multi-task learning (MTL)
framework. Our approach uniquely integrates tabular and non-tabular data,
leveraging a pre-trained TinyBERT model for semantic embeddings and a novel
sampling technique to capture diverse customer behaviors. We evaluate our model
against several baselines, including XGBoost, TabNet, FT-Transformer, DCN-V2,
and MMoE, focusing on their ability to handle mixed data types and optimize
personalized ranking. Additionally, we propose a scalable relevance labeling
mechanism based on click-through rates, click positions, and semantic
similarity, offering an alternative to traditional human-annotated labels.
Experimental results show that combining non-tabular data with advanced
embedding techniques in multi-task learning paradigm significantly enhances
model performance. Ablation studies further underscore the benefits of
incorporating relevance labels, fine-tuning TinyBERT layers, and TinyBERT
query-product embedding interactions. These results demonstrate the
effectiveness of our approach in achieving improved personalized product search
ranking.

### 6. [On Negative-aware Preference Optimization for Recommendation](http://arxiv.org/pdf/2508.09653v1)

Authors: Chenlu Ding, Daoxuan Liu, Jiancan Wu, Xingyu Hu, Junkang Wu, Haitao Wang, Yongkang Wang, Xingxing Wang, Xiang Wang

Recommendation systems leverage user interaction data to suggest relevant
items while filtering out irrelevant (negative) ones. The rise of large
language models (LLMs) has garnered increasing attention for their potential in
recommendation tasks. However, existing methods for optimizing LLM-based
recommenders face challenges in effectively utilizing negative samples. Simply
integrating large numbers of negative samples can improve ranking accuracy and
mitigate popularity bias but often leads to increased computational overhead
and memory costs. Additionally, current approaches fail to account for the
varying informativeness of negative samples, leading to suboptimal optimization
performance. To address these issues, we propose NAPO
(\textbf{N}egative-\textbf{A}ware \textbf{P}reference \textbf{O}ptimization),
an enhanced framework for preference optimization in LLM-based recommendation.
NAPO introduces two key innovations: (1) in-batch negative sharing, which
expands the pool of negative samples without additional memory overhead, and
(2) dynamic reward margin adjustment, which adapts model updates based on the
confidence of negative samples. Extensive experiments on three public datasets
demonstrate that NAPO outperforms existing methods in both recommendation
accuracy and popularity bias reduction.

### 7. [Describe What You See with Multimodal Large Language Models to Enhance Video Recommendations](http://arxiv.org/pdf/2508.09789v1)

Authors: Marco De Nadai, Andreas Damianou, Mounia Lalmas

Existing video recommender systems rely primarily on user-defined metadata or
on low-level visual and acoustic signals extracted by specialised encoders.
These low-level features describe what appears on the screen but miss deeper
semantics such as intent, humour, and world knowledge that make clips resonate
with viewers. For example, is a 30-second clip simply a singer on a rooftop, or
an ironic parody filmed amid the fairy chimneys of Cappadocia, Turkey? Such
distinctions are critical to personalised recommendations yet remain invisible
to traditional encoding pipelines. In this paper, we introduce a simple,
recommendation system-agnostic zero-finetuning framework that injects
high-level semantics into the recommendation pipeline by prompting an
off-the-shelf Multimodal Large Language Model (MLLM) to summarise each clip
into a rich natural-language description (e.g. "a superhero parody with
slapstick fights and orchestral stabs"), bridging the gap between raw content
and user intent. We use MLLM output with a state-of-the-art text encoder and
feed it into standard collaborative, content-based, and generative
recommenders. On the MicroLens-100K dataset, which emulates user interactions
with TikTok-style videos, our framework consistently surpasses conventional
video, audio, and metadata features in five representative models. Our findings
highlight the promise of leveraging MLLMs as on-the-fly knowledge extractors to
build more intent-aware video recommenders.

### 8. [On the Consistency and Performance of the Iterative Bayesian Update](http://arxiv.org/pdf/2508.09980v1)

Authors: Ehab ElSalamouny, Catuscia Palamidessi

For many social, scientific, and commercial purposes, it is often important
to estimate the distribution of the users' data regarding a sensitive
attribute, e.g., their ages, locations, etc. To allow this estimation while
protecting the users' privacy, every user applies a local privacy protection
mechanism that releases a noisy (sanitized) version of their original datum to
the data collector; then the original distribution is estimated using one of
the known methods, such as the matrix inversion (INV), RAPPOR's estimator, and
the iterative Bayesian update (IBU). Unlike the other estimators, the
consistency of IBU, i.e., the convergence of its estimate to the real
distribution as the amount of noisy data grows, has been either ignored or
incorrectly proved in the literature. In this article, we use the fact that IBU
is a maximum likelihood estimator to prove that IBU is consistent. We also
show, through experiments on real datasets, that IBU significantly outperforms
the other methods when the users' data are sanitized by geometric, Laplace, and
exponential mechanisms, whereas it is comparable to the other methods in the
case of the k-RR and RAPPOR mechanisms. Finally, we consider the case when the
alphabet of the sensitive data is infinite, and we show a technique that allows
IBU to operate in this case too.

### Machine Learning

### 1. [Graph Neural Network and Transformer Integration for Unsupervised System Anomaly Discovery](http://arxiv.org/pdf/2508.09401v1)

Authors: Yun Zi, Ming Gong, Zhihao Xue, Yujun Zou, Nia Qi, Yingnan Deng

This study proposes an unsupervised anomaly detection method for distributed
backend service systems, addressing practical challenges such as complex
structural dependencies, diverse behavioral evolution, and the absence of
labeled data. The method constructs a dynamic graph based on service invocation
relationships and applies graph convolution to extract high-order structural
representations from multi-hop topologies. A Transformer is used to model the
temporal behavior of each node, capturing long-term dependencies and local
fluctuations. During the feature fusion stage, a learnable joint embedding
mechanism integrates structural and behavioral representations into a unified
anomaly vector. A nonlinear mapping is then applied to compute anomaly scores,
enabling an end-to-end detection process without supervision. Experiments on
real-world cloud monitoring data include sensitivity analyses across different
graph depths, sequence lengths, and data perturbations. Results show that the
proposed method outperforms existing models on several key metrics,
demonstrating stronger expressiveness and stability in capturing anomaly
propagation paths and modeling dynamic behavior sequences, with high potential
for practical deployment.

### 2. [NEXICA: Discovering Road Traffic Causality (Extended arXiv Version)](http://arxiv.org/pdf/2508.09447v1)

Authors: Siddharth Srikanth, John Krumm, Jonathan Qin

Road traffic congestion is a persistent problem. Focusing resources on the
causes of congestion is a potentially efficient strategy for reducing
slowdowns. We present NEXICA, an algorithm to discover which parts of the
highway system tend to cause slowdowns on other parts of the highway. We use
time series of road speeds as inputs to our causal discovery algorithm. Finding
other algorithms inadequate, we develop a new approach that is novel in three
ways. First, it concentrates on just the presence or absence of events in the
time series, where an event indicates the temporal beginning of a traffic
slowdown. Second, we develop a probabilistic model using maximum likelihood
estimation to compute the probabilities of spontaneous and caused slowdowns
between two locations on the highway. Third, we train a binary classifier to
identify pairs of cause/effect locations trained on pairs of road locations
where we are reasonably certain a priori of their causal connections, both
positive and negative. We test our approach on six months of road speed data
from 195 different highway speed sensors in the Los Angeles area, showing that
our approach is superior to state-of-the-art baselines in both accuracy and
computation speed.

### 3. [Open-Set Fault Diagnosis in Multimode Processes via Fine-Grained Deep Feature Representation](http://arxiv.org/pdf/2508.09462v1)

Authors: Guangqiang Li, M. Amine Atoui, Xiangshun Li

A reliable fault diagnosis system should not only accurately classify known
health states but also effectively identify unknown faults. In multimode
processes, samples belonging to the same health state often show multiple
cluster distributions, making it difficult to construct compact and accurate
decision boundaries for that state. To address this challenge, a novel open-set
fault diagnosis model named fine-grained clustering and rejection network
(FGCRN) is proposed. It combines multiscale depthwise convolution,
bidirectional gated recurrent unit and temporal attention mechanism to capture
discriminative features. A distance-based loss function is designed to enhance
the intra-class compactness. Fine-grained feature representations are
constructed through unsupervised learning to uncover the intrinsic structures
of each health state. Extreme value theory is employed to model the distance
between sample features and their corresponding fine-grained representations,
enabling effective identification of unknown faults. Extensive experiments
demonstrate the superior performance of the proposed method.

### 4. [Learn to Explore: Meta NAS via Bayesian Optimization Guided Graph Generation](http://arxiv.org/pdf/2508.09467v1)

Authors: Zijun Sun, Yanning Shen

Neural Architecture Search (NAS) automates the design of high-performing
neural networks but typically targets a single predefined task, thereby
restricting its real-world applicability. To address this, Meta Neural
Architecture Search (Meta-NAS) has emerged as a promising paradigm that
leverages prior knowledge across tasks to enable rapid adaptation to new ones.
Nevertheless, existing Meta-NAS methods often struggle with poor
generalization, limited search spaces, or high computational costs. In this
paper, we propose a novel Meta-NAS framework, GraB-NAS. Specifically, GraB-NAS
first models neural architectures as graphs, and then a hybrid search strategy
is developed to find and generate new graphs that lead to promising neural
architectures. The search strategy combines global architecture search via
Bayesian Optimization in the search space with local exploration for novel
neural networks via gradient ascent in the latent space. Such a hybrid search
strategy allows GraB-NAS to discover task-aware architectures with strong
performance, even beyond the predefined search space. Extensive experiments
demonstrate that GraB-NAS outperforms state-of-the-art Meta-NAS baselines,
achieving better generalization and search effectiveness.

### 5. [EGGS-PTP: An Expander-Graph Guided Structured Post-training Pruning Method for Large Language Models](http://arxiv.org/pdf/2508.09471v1)

Authors: Omar Bazarbachi, Zijun Sun, Yanning Shen

As Large Language Models (LLMs) become more widely adopted and scale up in
size, the computational and memory challenges involved in deploying these
massive foundation models have grown increasingly severe. This underscores the
urgent need to develop more efficient model variants. Faced with this
challenge, the present work introduces EGGS-PTP: an Expander-Graph Guided
Structured Post-training Pruning method. The proposed approach leverages graph
theory to guide the design of N:M structured pruning, effectively reducing
model size and computational demands. By incorporating concepts from expander
graphs, EGGS-PTP ensures information flow within the pruned network, preserving
essential model functionality. Extensive numerical experiments demonstrate that
EGGS-PTP not only achieves significant acceleration and memory savings due to
structured sparsity but also outperforms existing structured pruning techniques
in terms of accuracy across various LLMs.

### 6. [Enhancing Memory Recall in LLMs with Gauss-Tin: A Hybrid Instructional and Gaussian Replay Approach](http://arxiv.org/pdf/2508.09510v1)

Authors: Iing Muttakhiroh, Thomas Fevens

Despite the significant advancements in Large Language Models (LLMs),
catastrophic forgetting remains a substantial challenge, where models lose
previously acquired knowledge upon learning new information. Continual learning
(CL) strategies have emerged as a potential solution to this problem, with
replay-based techniques demonstrating superior performance in preserving
learned knowledge. In this context, we introduce Gauss-Tin, a novel approach
that integrates the replay strategy with a Gaussian mixture model to enhance
the quality of sample selection during training, supplemented by instructional
guidance to facilitate the generation of past learning. This method aims to
improve LLMs' retention capabilities by strategically reinforcing important
past learnings while accommodating new information. Our experimental results
indicate a promising 6\% improvement in retention metrics over traditional
methods, suggesting that Gauss-Tin is an effective strategy for mitigating
catastrophic forgetting in LLMs. This study underscores the potential of hybrid
models in enhancing the robustness and adaptability of LLMs in dynamic learning
environments.

### 7. [Time-Aware and Transition-Semantic Graph Neural Networks for Interpretable Predictive Business Process Monitoring](http://arxiv.org/pdf/2508.09527v1)

Authors: Fang Wang, Ernesto Damiani

Predictive Business Process Monitoring (PBPM) aims to forecast future events
in ongoing cases based on historical event logs. While Graph Neural Networks
(GNNs) are well suited to capture structural dependencies in process data,
existing GNN-based PBPM models remain underdeveloped. Most rely either on short
prefix subgraphs or global architectures that overlook temporal relevance and
transition semantics. We propose a unified, interpretable GNN framework that
advances the state of the art along three key axes. First, we compare
prefix-based Graph Convolutional Networks(GCNs) and full trace Graph Attention
Networks(GATs) to quantify the performance gap between localized and global
modeling. Second, we introduce a novel time decay attention mechanism that
constructs dynamic, prediction-centered windows, emphasizing temporally
relevant history and suppressing noise. Third, we embed transition type
semantics into edge features to enable fine grained reasoning over structurally
ambiguous traces. Our architecture includes multilevel interpretability
modules, offering diverse visualizations of attention behavior. Evaluated on
five benchmarks, the proposed models achieve competitive Top-k accuracy and DL
scores without per-dataset tuning. By addressing architectural, temporal, and
semantic gaps, this work presents a robust, generalizable, and explainable
solution for next event prediction in PBPM.

### 8. [SYNAPSE-G: Bridging Large Language Models and Graph Learning for Rare Event Classification](http://arxiv.org/pdf/2508.09544v1)

Authors: Sasan Tavakkol, Lin Chen, Max Springer, Abigail Schantz, Blaž Bratanič, Vincent Cohen-Addad, MohammadHossein Bateni

Scarcity of labeled data, especially for rare events, hinders training
effective machine learning models. This paper proposes SYNAPSE-G (Synthetic
Augmentation for Positive Sampling via Expansion on Graphs), a novel pipeline
leveraging Large Language Models (LLMs) to generate synthetic training data for
rare event classification, addressing the cold-start problem. This synthetic
data serve as seeds for semi-supervised label propagation on a similarity graph
constructed between the seeds and a large unlabeled dataset. This identifies
candidate positive examples, subsequently labeled by an oracle (human or LLM).
The expanded dataset then trains/fine-tunes a classifier. We theoretically
analyze how the quality (validity and diversity) of the synthetic data impacts
the precision and recall of our method. Experiments on the imbalanced SST2 and
MHS datasets demonstrate SYNAPSE-G's effectiveness in finding positive labels,
outperforming baselines including nearest neighbor search.

### 9. [Edge General Intelligence Through World Models and Agentic AI: Fundamentals, Solutions, and Challenges](http://arxiv.org/pdf/2508.09561v1)

Authors: Changyuan Zhao, Guangyuan Liu, Ruichen Zhang, Yinqiu Liu, Jiacheng Wang, Jiawen Kang, Dusit Niyato, Zan Li, Xuemin, Shen, Zhu Han, Sumei Sun, Chau Yuen, Dong In Kim

Edge General Intelligence (EGI) represents a transformative evolution of edge
computing, where distributed agents possess the capability to perceive, reason,
and act autonomously across diverse, dynamic environments. Central to this
vision are world models, which act as proactive internal simulators that not
only predict but also actively imagine future trajectories, reason under
uncertainty, and plan multi-step actions with foresight. This proactive nature
allows agents to anticipate potential outcomes and optimize decisions ahead of
real-world interactions. While prior works in robotics and gaming have
showcased the potential of world models, their integration into the wireless
edge for EGI remains underexplored. This survey bridges this gap by offering a
comprehensive analysis of how world models can empower agentic artificial
intelligence (AI) systems at the edge. We first examine the architectural
foundations of world models, including latent representation learning, dynamics
modeling, and imagination-based planning. Building on these core capabilities,
we illustrate their proactive applications across EGI scenarios such as
vehicular networks, unmanned aerial vehicle (UAV) networks, the Internet of
Things (IoT) systems, and network functions virtualization, thereby
highlighting how they can enhance optimization under latency, energy, and
privacy constraints. We then explore their synergy with foundation models and
digital twins, positioning world models as the cognitive backbone of EGI.
Finally, we highlight open challenges, such as safety guarantees, efficient
training, and constrained deployment, and outline future research directions.
This survey provides both a conceptual foundation and a practical roadmap for
realizing the next generation of intelligent, autonomous edge systems.

### 10. [Physics- and geometry-aware spatio-spectral graph neural operator for time-independent and time-dependent PDEs](http://arxiv.org/pdf/2508.09627v1)

Authors: Subhankar Sarkar, Souvik Chakraborty

Solving partial differential equations (PDEs) efficiently and accurately
remains a cornerstone challenge in science and engineering, especially for
problems involving complex geometries and limited labeled data. We introduce a
Physics- and Geometry- Aware Spatio-Spectral Graph Neural Operator
($\pi$G-Sp$^2$GNO) for learning the solution operators of time-independent and
time-dependent PDEs. The proposed approach first improves upon the recently
developed Sp$^2$GNO by enabling geometry awareness and subsequently exploits
the governing physics to learn the underlying solution operator in a
simulation-free setup. While the spatio-spectral structure present in the
proposed architecture allows multiscale learning, two separate strategies for
enabling geometry awareness is introduced in this paper. For time dependent
problems, we also introduce a novel hybrid physics informed loss function that
combines higher-order time-marching scheme with upscaled theory inspired
stochastic projection scheme. This allows accurate integration of the
physics-information into the loss function. The performance of the proposed
approach is illustrated on number of benchmark examples involving regular and
complex domains, variation in geometry during inference, and time-independent
and time-dependent problems. The results obtained illustrate the efficacy of
the proposed approach as compared to the state-of-the-art physics-informed
neural operator algorithms in the literature.

### Neural and Evolutionary Computing

### 1. [Event-driven Robust Fitting on Neuromorphic Hardware](http://arxiv.org/pdf/2508.09466v1)

Authors: Tam Ngoc-Bang Nguyen, Anh-Dzung Doan, Zhipeng Cai, Tat-Jun Chin

Robust fitting of geometric models is a fundamental task in many computer
vision pipelines. Numerous innovations have been produced on the topic, from
improving the efficiency and accuracy of random sampling heuristics to
generating novel theoretical insights that underpin new approaches with
mathematical guarantees. However, one aspect of robust fitting that has
received little attention is energy efficiency. This performance metric has
become critical as high energy consumption is a growing concern for AI
adoption. In this paper, we explore energy-efficient robust fitting via the
neuromorphic computing paradigm. Specifically, we designed a novel spiking
neural network for robust fitting on real neuromorphic hardware, the Intel
Loihi 2. Enabling this are novel event-driven formulations of model estimation
that allow robust fitting to be implemented in the unique architecture of Loihi
2, and algorithmic strategies to alleviate the current limited precision and
instruction set of the hardware. Results show that our neuromorphic robust
fitting consumes only a fraction (15%) of the energy required to run the
established robust fitting algorithm on a standard CPU to equivalent accuracy.

### 2. [Reinforcement learning in densely recurrent biological networks](http://arxiv.org/pdf/2508.09618v1)

Authors: Miles Walter Churchland, Jordi Garcia-Ojalvo

Training highly recurrent networks in continuous action spaces is a technical
challenge: gradient-based methods suffer from exploding or vanishing gradients,
while purely evolutionary searches converge slowly in high-dimensional weight
spaces. We introduce a hybrid, derivative-free optimization framework that
implements reinforcement learning by coupling global evolutionary exploration
with local direct search exploitation. The method, termed ENOMAD (Evolutionary
Nonlinear Optimization with Mesh Adaptive Direct search), is benchmarked on a
suite of food-foraging tasks instantiated in the fully mapped neural connectome
of the nematode \emph{Caenorhabditis elegans}. Crucially, ENOMAD leverages
biologically derived weight priors, letting it refine--rather than rebuild--the
organism's native circuitry. Two algorithmic variants of the method are
introduced, which lead to either small distributed adjustments of many weights,
or larger changes on a limited number of weights. Both variants significantly
exceed the performance of the untrained connectome (in what can be interpreted
as an example of transfer learning) and of existing training strategies. These
findings demonstrate that integrating evolutionary search with nonlinear
optimization provides an efficient, biologically grounded strategy for
specializing natural recurrent networks towards a specified set of tasks.

### 3. [Counting Short Trajectories in Elementary Cellular Automata using the Transfer Matrix Method](http://arxiv.org/pdf/2508.09768v1)

Authors: Cédric Koller, Barbora Hudcová

Elementary Cellular Automata (ECAs) exhibit diverse behaviours often
categorized by Wolfram's qualitative classification. To provide a quantitative
basis for understanding these behaviours, we investigate the global dynamics of
such automata and we describe a method that allows us to compute the number of
all configurations leading to short attractors in a limited number of time
steps. This computation yields exact results in the thermodynamic limit (as the
CA grid size grows to infinity), and is based on the Transfer Matrix Method
(TMM) that we adapt for our purposes. Specifically, given two parameters $(p,
c)$ we are able to compute the entropy of all initial configurations converging
to an attractor of size $c$ after $p$ time-steps. By calculating such
statistics for various ECA rules, we establish a quantitative connection
between the entropy and the qualitative Wolfram classification scheme. Class 1
rules rapidly converge to maximal entropy for stationary states ($c=1$) as $p$
increases. Class 2 rules also approach maximal entropy quickly for appropriate
cycle lengths $c$, potentially requiring consideration of translations. Class 3
rules exhibit zero or low finite entropy that saturates after a short
transient. Class 4 rules show finite positive entropy, similar to some Class 3
rules. This method provides a precise framework for quantifying trajectory
statistics, although its exponential computational cost in $p+c$ restricts
practical analysis to short trajectories.

### 4. [Perceptual Reality Transformer: Neural Architectures for Simulating Neurological Perception Conditions](http://arxiv.org/pdf/2508.09852v1)

Authors: Baihan Lin

Neurological conditions affecting visual perception create profound
experiential divides between affected individuals and their caregivers,
families, and medical professionals. We present the Perceptual Reality
Transformer, a comprehensive framework employing six distinct neural
architectures to simulate eight neurological perception conditions with
scientifically-grounded visual transformations. Our system learns mappings from
natural images to condition-specific perceptual states, enabling others to
experience approximations of simultanagnosia, prosopagnosia, ADHD attention
deficits, visual agnosia, depression-related changes, anxiety tunnel vision,
and Alzheimer's memory effects. Through systematic evaluation across ImageNet
and CIFAR-10 datasets, we demonstrate that Vision Transformer architectures
achieve optimal performance, outperforming traditional CNN and generative
approaches. Our work establishes the first systematic benchmark for
neurological perception simulation, contributes novel condition-specific
perturbation functions grounded in clinical literature, and provides
quantitative metrics for evaluating simulation fidelity. The framework has
immediate applications in medical education, empathy training, and assistive
technology development, while advancing our fundamental understanding of how
neural networks can model atypical human perception.

### Networking and Internet Architecture

### 1. [Energy-efficient PON-based Backhaul Connectivity for a VLC-enabled Indoor Fog Computing Environment](http://arxiv.org/pdf/2508.09582v1)

Authors: Wafaa B. M. Fadlelmula, Sanaa Hamid Mohamed, Taisir E. H. El-Gorashi, Jaafar M. H. Elmirghani

In this paper, we consider the use of visible light communication (VLC) to
provide connectivity to indoor fog computing resources and propose an
energy-efficient passive optical network (PON)-based backhaul architecture to
support the VLC system. We develop a mixed-integer linear programming (MILP)
model to optimize the allocation of computing resources over the proposed
architecture, aiming to minimize processing and networking power consumption.
We evaluate the performance of the proposed architecture under varying workload
demands and user distributions. Comparative analysis against a backhaul
architecture that is based on the state-of-the-art spine-and-leaf (S&L) network
design demonstrates total power savings of up to 82%. Further comparison with
centralized cloud processing shows improvements in energy efficiency of up to
93%. Additionally, we examine the improvements in energy efficiency obtained by
splitting tasks among multiple processing nodes and propose enhancements to the
architecture including dynamic bandwidth allocation, increased wavelength
bandwidth and improved connectivity within rooms to alleviate networking
bottlenecks. Furthermore, we introduce an inter-building architecture that
leverages resources from neighboring buildings to support high-demand
scenarios.

### 2. [The Paradigm of Massive Wireless Human Sensing: Concept, Architecture and Challenges](http://arxiv.org/pdf/2508.09756v1)

Authors: Mauro De Sanctis

This article is a position paper which introduces the paradigm of ``Massive
Wireless Human Sensing'', i.e. an infrastructure for wireless human sensing
based on a plethora of heterogeneous wireless communication signals. More
specifically, we aim to exploit signal diversity in the time, frequency, and
space domains using opportunistically both device-free and device-based
wireless sensing approaches, with the objective of enhancing human sensing
capabilities in terms of accuracy and service availability over different
environments. The enabling element of this concept is the massive wireless
human sensing edge device, that is, an embedded system acting as a
multi-technology and multi-approach RF receiver with feature extraction
functionality, located within the monitoring area or at its borders. In this
framework, architecture solutions and challenges are discussed to lead the
future development of this new paradigm.

### 3. [An (m,k)-firm Elevation Policy to Increase the Robustness of Time-Driven Schedules in 5G Time-Sensitive Networks](http://arxiv.org/pdf/2508.09769v1)

Authors: Simon Egger, Robin Laidig, Heiko Geppert, Lucas Haug, Jona Herrmann, Frank Dürr, Christian Becker

Current standardization efforts are advancing the integration of 5G and
Time-Sensitive Networking (TSN) to facilitate the deployment of safety-critical
industrial applications that require real-time communication. However, there
remains a fundamental disconnect between the probabilistic 5G delay
characteristics and the often idealistic delay models used to synthesize 5G-TSN
network configurations. For time-driven schedules in particular, any delay
outlier unforeseen during schedule synthesis can jeopardize the robustness of
their real-time guarantees. To address this challenge, we present the
(m,k)-firm Elevation Policy to uphold a base level of weakly hard real-time
guarantees during unstable network conditions that do not match the expected
delay characteristics. It augments the primary time-driven schedule with a
dynamic priority-driven scheme to elevate the priority of m out of k
consecutive frames if they are delayed. Our evaluations demonstrate that weakly
hard real-time guarantees are essential to uphold the quality of control within
a networked control system. At the same time, only a small overhead is imposed
when the primary schedule can provide stronger quality of service guarantees.
Our (m,k)-firm Elevation Policy thereby yields a robust but light-weight
fallback mechanism to serve applications with meaningful guarantees during
unstable network conditions.

### 4. [A First Look at Starlink In-Flight Performance: An Intercontinental Empirical Study](http://arxiv.org/pdf/2508.09839v1)

Authors: Muhammad Asad Ullah, Luca Borgianni, Heikki Kokkinen, Antti Anttonen, Stefano Giordano

Starlink delivers Internet services to users across terrestrial, maritime,
and aviation domains. The prior works have studied its performance at fixed
sites and in-motion vehicles, while an in-depth analysis of in-flight
performance remains absent. With major airlines now offering Starlink Internet
onboard, there is a growing need to evaluate and improve its performance for
aviation users. This paper addresses this shortcoming by conducting in-flight
measurements over the Baltic Sea and the Pacific Ocean. Our measurement results
show that a single user device experiences median throughputs of 64 Mbps and 24
Mbps for the downlink and uplink, respectively. The median uplink throughput is
approximately 33 Mbps when the aircraft maintains an altitude above 17,000
feet. However, a significant reduction in uplink performance is observed during
the aircraft descent phase, with the median throughput dropping to around 20
Mbps at lower altitudes. Round-trip time (RTT) is highly dependent on the
location of the ground station being pinged and the use of inter-satellite
links (ISLs). We dive deeper into 5.5 hours of ping measurements collected over
the Pacific Ocean and investigate factors influencing RTT, hypothesizing that
ISLs routing, data queuing at satellites, and feeder link congestion contribute
to deviations from theoretical values. For comparative analysis, we evaluate
the Starlink ground terminal and in-flight connectivity performance from the
perspectives of a residential user and an airline passenger, respectively.

### 5. [Metrics for Assessing Changes in Flow-based Networks](http://arxiv.org/pdf/2508.09573v1)

Authors: Michał Rzepka, Piotr Chołda

This paper addresses the challenges of evaluating network performance in the
presence of fluctuating traffic patterns, with a particular focus on the impact
of peak data rates on network resources. We introduce a set of metrics to
quantify network load and measure the impact of individual flows on the overall
network state. By analyzing link and flow data through percentile values and
sample distributions, and introducing the Utilization Score metric, the
research provides insights into resource utilization under varying network
conditions. Furthermore, we employ a modified Shapley value-based approach to
measure the influence of individual flows on the network, offering a better
understanding of their contribution to network performance. The paper reviews
and compares 11 metrics across various network scenarios, evaluating their
practical relevance for research and development. Our evaluation demonstrates
that these metrics effectively capture changes in network state induced by
specific flows, with three of them offering a broad range of valuable insights
while remaining relatively easy to maintain. Moreover, the methodology
described in this paper serves as a framework for future research, with the
potential to expand and refine the set of metrics used to evaluate flow impact
on network performance.

### 6. [Closing the HPC-Cloud Convergence Gap: Multi-Tenant Slingshot RDMA for Kubernetes](http://arxiv.org/pdf/2508.09663v1)

Authors: Philipp A. Friese, Ahmed Eleliemy, Utz-Uwe Haus, Martin Schulz

Converged HPC-Cloud computing is an emerging computing paradigm that aims to
support increasingly complex and multi-tenant scientific workflows. These
systems require reconciliation of the isolation requirements of native cloud
workloads and the performance demands of HPC applications. In this context,
networking hardware is a critical boundary component: it is the conduit for
high-throughput, low-latency communication and enables isolation across
tenants. HPE Slingshot is a high-speed network interconnect that provides up to
200 Gbps of throughput per port and targets high-performance computing (HPC)
systems. The Slingshot host software, including hardware drivers and network
middleware libraries, is designed to meet HPC deployments, which predominantly
use single-tenant access modes. Hence, the Slingshot stack is not suited for
secure use in multi-tenant deployments, such as converged HPC-Cloud
deployments. In this paper, we design and implement an extension to the
Slingshot stack targeting converged deployments on the basis of Kubernetes. Our
integration provides secure, container-granular, and multi-tenant access to
Slingshot RDMA networking capabilities at minimal overhead.

### 7. [3GPP NR V2X Mode 2d: Analysis of Distributed Scheduling for Groupcast using ns-3 5G LENA Simulator](http://arxiv.org/pdf/2508.09708v1)

Authors: Thomas Fehrenbach, Luis Omar Ortiz Abrego, Cornelius Hellge, Thomas Schierl, Jörg Ott

Vehicle-to-everything (V2X) communication is a key technology for enabling
intelligent transportation systems (ITS) that can improve road safety, traffic
efficiency, and environmental sustainability. Among the various V2X
applications, platooning is one of the most promising ones, as it allows a
group of vehicles to travel closely together at high speeds, reducing fuel
consumption and emissions. However, it poses significant challenges for
wireless communication, such as high reliability and low latency. In this
paper, we evaluate the benefits of group scheduling, also referred to as Mode
2d, which is based on a distributed and scheduled resource allocation scheme
that allows the group of cars to select resources from a configured pool
without network assistance. We evaluated the scheme through simulations, and
the results show that this approach can meet the reliability, low latency, and
data rate requirements for platooning.

### 8. [Route Planning and Online Routing for Quantum Key Distribution Networks](http://arxiv.org/pdf/2508.09735v1)

Authors: Jorge López, Charalampos Chatzinakis, Marc Cartigny

Quantum Key Distribution (QKD) networks harness the principles of quantum
physics in order to securely transmit cryptographic key material, providing
physical guarantees. These networks require traditional management and
operational components, such as routing information through the network
elements. However, due to the limitations on capacity and the particularities
of information handling in these networks, traditional shortest paths
algorithms for routing perform poorly on both route planning and online
routing, which is counterintuitive. Moreover, due to the scarce resources in
such networks, often the expressed demand cannot be met by any assignment of
routes. To address both the route planning problem and the need for fair
automated suggestions in infeasible cases, we propose to model this problem as
a Quadratic Programming (QP) problem. For the online routing problem, we
showcase that the shortest (available) paths routing strategy performs poorly
in the online setting. Furthermore, we prove that the widest shortest path
routing strategy has a competitive ratio greater or equal than $\frac{1}{2}$,
efficiently addressing both routing modes in QKD networks.

### 9. [Decentralized Rank Scheduling for Energy-Constrained Multi-Task Federated Fine-Tuning in Edge-Assisted IoV Networks](http://arxiv.org/pdf/2508.09532v1)

Authors: Bokeng Zheng, Jianqiang Zhong, Jiayi Liu, Xiaoxi Zhang

Federated fine-tuning has emerged as a promising approach for adapting
foundation models (FMs) to diverse downstream tasks in edge environments. In
Internet of Vehicles (IoV) systems, enabling efficient and low-latency
multi-task adaptation is particularly challenging due to client mobility,
heterogeneous resources, and intermittent connectivity. This paper proposes a
hierarchical federated fine-tuning framework that coordinates roadside units
(RSUs) and vehicles to support resource-aware and mobility-resilient learning
across dynamic IoV scenarios. Leveraging Low-Rank Adaptation (LoRA), we
introduce a decentralized, energy-aware rank adaptation mechanism formulated as
a constrained multi-armed bandit problem. A novel UCB-DUAL algorithm is
developed to enable adaptive exploration under per-task energy budgets,
achieving provable sublinear regret. To evaluate our method, we construct a
large-scale IoV simulator based on real-world trajectories, capturing dynamic
participation, RSU handoffs, and communication variability. Extensive
experiments show that our approach achieves the best accuracy-efficiency
trade-off among all baselines, reducing latency by over 24\% and improving
average accuracy by more than 2.5\%.

### 10. [Duty-Cycling is Not Enough in Constrained IoT Networking: Revealing the Energy Savings of Dynamic Clock Scaling](http://arxiv.org/pdf/2508.09620v1)

Authors: Michel Rottleuthner, Thomas C. Schmidt, Matthias Wählisch

Minimizing energy consumption of low-power wireless nodes is a persistent
challenge from the constrained Internet of Things (IoT). In this paper, we
start from the observation that constrained IoT devices have largely different
hardware (im-)balances than full-scale machines. We find that the performance
gap between MCU and network throughput on constrained devices enables minimal
energy delay product (EDP) for IoT networking at largely reduced clock
frequencies. We analyze the potentials by integrating dynamic voltage and
frequency scaling (DVFS) into the RIOT IoT operating system and show that the
DVFS reconfiguration overhead stays below the energy saved for a single,
downscaled MAC operation. Backed by these findings, we systematically
investigate how DVFS further improves energy-efficiency for common networking
tasks -- in addition to duty-cycling. We measure IoT communication scenarios
between real-world systems and analyze two MAC operating modes -- CSMA/CA and
time slotting -- in combination with different CoAP transactions, payload
sizes, as well as DTLS transport encryption. Our experiments reveal energy
savings between 24% and 52% for MAC operations and up to 37% for encrypted CoAP
communication. These results shall encourage research and system design work to
integrate DVFS in future IoT devices for performing tasks at their optimal
frequencies and thereby significantly extending battery lifetimes.

### Robotics

### 1. [Reactive Model Predictive Contouring Control for Robot Manipulators](http://arxiv.org/pdf/2508.09502v1)

Authors: Junheon Yoon, Woo-Jeong Baek, Jaeheung Park

This contribution presents a robot path-following framework via Reactive
Model Predictive Contouring Control (RMPCC) that successfully avoids obstacles,
singularities and self-collisions in dynamic environments at 100 Hz. Many
path-following methods rely on the time parametrization, but struggle to handle
collision and singularity avoidance while adhering kinematic limits or other
constraints. Specifically, the error between the desired path and the actual
position can become large when executing evasive maneuvers. Thus, this paper
derives a method that parametrizes the reference path by a path parameter and
performs the optimization via RMPCC. In particular, Control Barrier Functions
(CBFs) are introduced to avoid collisions and singularities in dynamic
environments. A Jacobian-based linearization and Gauss-Newton Hessian
approximation enable solving the nonlinear RMPCC problem at 100 Hz,
outperforming state-of-the-art methods by a factor of 10. Experiments confirm
that the framework handles dynamic obstacles in real-world settings with low
contouring error and low robot acceleration.

### 2. [ESCoT: An Enhanced Step-based Coordinate Trajectory Planning Method for Multiple Car-like Robots](http://arxiv.org/pdf/2508.09581v1)

Authors: Junkai Jiang, Yihe Chen, Yibin Yang, Ruochen Li, Shaobing Xu, Jianqiang Wang

Multi-vehicle trajectory planning (MVTP) is one of the key challenges in
multi-robot systems (MRSs) and has broad applications across various fields.
This paper presents ESCoT, an enhanced step-based coordinate trajectory
planning method for multiple car-like robots. ESCoT incorporates two key
strategies: collaborative planning for local robot groups and replanning for
duplicate configurations. These strategies effectively enhance the performance
of step-based MVTP methods. Through extensive experiments, we show that ESCoT
1) in sparse scenarios, significantly improves solution quality compared to
baseline step-based method, achieving up to 70% improvement in typical conflict
scenarios and 34% in randomly generated scenarios, while maintaining high
solving efficiency; and 2) in dense scenarios, outperforms all baseline
methods, maintains a success rate of over 50% even in the most challenging
configurations. The results demonstrate that ESCoT effectively solves MVTP,
further extending the capabilities of step-based methods. Finally, practical
robot tests validate the algorithm's applicability in real-world scenarios.

### 3. [Immersive Teleoperation of Beyond-Human-Scale Robotic Manipulators: Challenges and Future Directions](http://arxiv.org/pdf/2508.09700v1)

Authors: Mahdi Hejrati, Jouni Mattila

Teleoperation of beyond-human-scale robotic manipulators (BHSRMs) presents
unique challenges that differ fundamentally from conventional human-scale
systems. As these platforms gain relevance in industrial domains such as
construction, mining, and disaster response, immersive interfaces must be
rethought to support scalable, safe, and effective human-robot collaboration.
This paper investigates the control, cognitive, and interface-level challenges
of immersive teleoperation in BHSRMs, with a focus on ensuring operator safety,
minimizing sensorimotor mismatch, and enhancing the sense of embodiment. We
analyze design trade-offs in haptic and visual feedback systems, supported by
early experimental comparisons of exoskeleton- and joystick-based control
setups. Finally, we outline key research directions for developing new
evaluation tools, scaling strategies, and human-centered safety models tailored
to large-scale robotic telepresence.

### 4. [FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning](http://arxiv.org/pdf/2508.09797v1)

Authors: Dongcheng Cao, Jin Zhou, Xian Wang, Shuo Li

Agile flight for the quadrotor cable-suspended payload system is a formidable
challenge due to its underactuated, highly nonlinear, and hybrid dynamics.
Traditional optimization-based methods often struggle with high computational
costs and the complexities of cable mode transitions, limiting their real-time
applicability and maneuverability exploitation. In this letter, we present
FLARE, a reinforcement learning (RL) framework that directly learns agile
navigation policy from high-fidelity simulation. Our method is validated across
three designed challenging scenarios, notably outperforming a state-of-the-art
optimization-based approach by a 3x speedup during gate traversal maneuvers.
Furthermore, the learned policies achieve successful zero-shot sim-to-real
transfer, demonstrating remarkable agility and safety in real-world
experiments, running in real time on an onboard computer.

### 5. [Embodied Tactile Perception of Soft Objects Properties](http://arxiv.org/pdf/2508.09836v1)

Authors: Anirvan Dutta, Alexis WM Devillard, Zhihuan Zhang, Xiaoxiao Cheng, Etienne Burdet

To enable robots to develop human-like fine manipulation, it is essential to
understand how mechanical compliance, multi-modal sensing, and purposeful
interaction jointly shape tactile perception. In this study, we use a dedicated
modular e-Skin with tunable mechanical compliance and multi-modal sensing
(normal, shear forces and vibrations) to systematically investigate how sensing
embodiment and interaction strategies influence robotic perception of objects.
Leveraging a curated set of soft wave objects with controlled viscoelastic and
surface properties, we explore a rich set of palpation primitives-pressing,
precession, sliding that vary indentation depth, frequency, and directionality.
In addition, we propose the latent filter, an unsupervised, action-conditioned
deep state-space model of the sophisticated interaction dynamics and infer
causal mechanical properties into a structured latent space. This provides
generalizable and in-depth interpretable representation of how embodiment and
interaction determine and influence perception. Our investigation demonstrates
that multi-modal sensing outperforms uni-modal sensing. It highlights a nuanced
interaction between the environment and mechanical properties of e-Skin, which
should be examined alongside the interaction by incorporating temporal
dynamics.

### 6. [Whole-Body Bilateral Teleoperation with Multi-Stage Object Parameter Estimation for Wheeled Humanoid Locomanipulation](http://arxiv.org/pdf/2508.09846v1)

Authors: Donghoon Baek, Amartya Purushottam, Jason J. Choi, Joao Ramos

This paper presents an object-aware whole-body bilateral teleoperation
framework for wheeled humanoid loco-manipulation. This framework combines
whole-body bilateral teleoperation with an online multi-stage object inertial
parameter estimation module, which is the core technical contribution of this
work. The multi-stage process sequentially integrates a vision-based object
size estimator, an initial parameter guess generated by a large vision-language
model (VLM), and a decoupled hierarchical sampling strategy. The visual size
estimate and VLM prior offer a strong initial guess of the object's inertial
parameters, significantly reducing the search space for sampling-based
refinement and improving the overall estimation speed. A hierarchical strategy
first estimates mass and center of mass, then infers inertia from object size
to ensure physically feasible parameters, while a decoupled multi-hypothesis
scheme enhances robustness to VLM prior errors. Our estimator operates in
parallel with high-fidelity simulation and hardware, enabling real-time online
updates. The estimated parameters are then used to update the wheeled
humanoid's equilibrium point, allowing the operator to focus more on locomotion
and manipulation. This integration improves the haptic force feedback for
dynamic synchronization, enabling more dynamic whole-body teleoperation. By
compensating for object dynamics using the estimated parameters, the framework
also improves manipulation tracking while preserving compliant behavior. We
validate the system on a customized wheeled humanoid with a robotic gripper and
human-machine interface, demonstrating real-time execution of lifting,
delivering, and releasing tasks with a payload weighing approximately one-third
of the robot's body weight.

### 7. [PPL: Point Cloud Supervised Proprioceptive Locomotion Reinforcement Learning for Legged Robots in Crawl Spaces](http://arxiv.org/pdf/2508.09950v1)

Authors: Bida Ma, Nuo Xu, Chenkun Qi, Xin Liu, Yule Mo, Jinkai Wang, Chunpeng Lu

The legged locomotion in spatially constrained structures (called crawl
spaces) is challenging. In crawl spaces, current exteroceptive locomotion
learning methods are limited by large noises and errors of the sensors in
possible low visibility conditions, and current proprioceptive locomotion
learning methods are difficult in traversing crawl spaces because only ground
features are inferred. In this study, a point cloud supervised proprioceptive
locomotion reinforcement learning method for legged robots in crawl spaces is
proposed. A state estimation network is designed to estimate the robot's
surrounding ground and spatial features as well as the robot's collision states
using historical proprioceptive sensor data. The point cloud is represented in
polar coordinate frame and a point cloud processing method is proposed to
efficiently extract the ground and spatial features that are used to supervise
the state estimation network learning. Comprehensive reward functions that
guide the robot to traverse through crawl spaces after collisions are designed.
Experiments demonstrate that, compared to existing methods, our method exhibits
more agile locomotion in crawl spaces. This study enhances the ability of
legged robots to traverse spatially constrained environments without requiring
exteroceptive sensors.

### 8. [Masquerade: Learning from In-the-wild Human Videos using Data-Editing](http://arxiv.org/pdf/2508.09976v1)

Authors: Marion Lepert, Jiaying Fang, Jeannette Bohg

Robot manipulation research still suffers from significant data scarcity:
even the largest robot datasets are orders of magnitude smaller and less
diverse than those that fueled recent breakthroughs in language and vision. We
introduce Masquerade, a method that edits in-the-wild egocentric human videos
to bridge the visual embodiment gap between humans and robots and then learns a
robot policy with these edited videos. Our pipeline turns each human video into
robotized demonstrations by (i) estimating 3-D hand poses, (ii) inpainting the
human arms, and (iii) overlaying a rendered bimanual robot that tracks the
recovered end-effector trajectories. Pre-training a visual encoder to predict
future 2-D robot keypoints on 675K frames of these edited clips, and continuing
that auxiliary loss while fine-tuning a diffusion policy head on only 50 robot
demonstrations per task, yields policies that generalize significantly better
than prior work. On three long-horizon, bimanual kitchen tasks evaluated in
three unseen scenes each, Masquerade outperforms baselines by 5-6x. Ablations
show that both the robot overlay and co-training are indispensable, and
performance scales logarithmically with the amount of edited human video. These
results demonstrate that explicitly closing the visual embodiment gap unlocks a
vast, readily available source of data from human videos that can be used to
improve robot policies.

### 9. [Distilling LLM Prior to Flow Model for Generalizable Agent's Imagination in Object Goal Navigation](http://arxiv.org/pdf/2508.09423v1)

Authors: Badi Li, Ren-jie Lu, Yu Zhou, Jingke Meng, Wei-shi Zheng

The Object Goal Navigation (ObjectNav) task challenges agents to locate a
specified object in an unseen environment by imagining unobserved regions of
the scene. Prior approaches rely on deterministic and discriminative models to
complete semantic maps, overlooking the inherent uncertainty in indoor layouts
and limiting their ability to generalize to unseen environments. In this work,
we propose GOAL, a generative flow-based framework that models the semantic
distribution of indoor environments by bridging observed regions with
LLM-enriched full-scene semantic maps. During training, spatial priors inferred
from large language models (LLMs) are encoded as two-dimensional Gaussian
fields and injected into target maps, distilling rich contextual knowledge into
the flow model and enabling more generalizable completions. Extensive
experiments demonstrate that GOAL achieves state-of-the-art performance on MP3D
and Gibson, and shows strong generalization in transfer settings to HM3D. Codes
and pretrained models are available at https://github.com/Badi-Li/GOAL.

### 10. [DAgger Diffusion Navigation: DAgger Boosted Diffusion Policy for Vision-Language Navigation](http://arxiv.org/pdf/2508.09444v1)

Authors: Haoxiang Shi, Xiang Deng, Zaijing Li, Gongwei Chen, Yaowei Wang, Liqiang Nie

Vision-Language Navigation in Continuous Environments (VLN-CE) requires
agents to follow natural language instructions through free-form 3D spaces.
Existing VLN-CE approaches typically use a two-stage waypoint planning
framework, where a high-level waypoint predictor generates the navigable
waypoints, and then a navigation planner suggests the intermediate goals in the
high-level action space. However, this two-stage decomposition framework
suffers from: (1) global sub-optimization due to the proxy objective in each
stage, and (2) a performance bottleneck caused by the strong reliance on the
quality of the first-stage predicted waypoints. To address these limitations,
we propose DAgger Diffusion Navigation (DifNav), an end-to-end optimized VLN-CE
policy that unifies the traditional two stages, i.e. waypoint generation and
planning, into a single diffusion policy. Notably, DifNav employs a conditional
diffusion policy to directly model multi-modal action distributions over future
actions in continuous navigation space, eliminating the need for a waypoint
predictor while enabling the agent to capture multiple possible
instruction-following behaviors. To address the issues of compounding error in
imitation learning and enhance spatial reasoning in long-horizon navigation
tasks, we employ DAgger for online policy training and expert trajectory
augmentation, and use the aggregated data to further fine-tune the policy. This
approach significantly improves the policy's robustness and its ability to
recover from error states. Extensive experiments on benchmark datasets
demonstrate that, even without a waypoint predictor, the proposed method
substantially outperforms previous state-of-the-art two-stage waypoint-based
models in terms of navigation performance. Our code is available at:
https://github.com/Tokishx/DifNav.

### Software Engineering

### 1. [ReqInOne: A Large Language Model-Based Agent for Software Requirements Specification Generation](http://arxiv.org/pdf/2508.09648v1)

Authors: Taohong Zhu, Lucas C. Cordeiro, Youcheng Sun

Software Requirements Specification (SRS) is one of the most important
documents in software projects, but writing it manually is time-consuming and
often leads to ambiguity. Existing automated methods rely heavily on manual
analysis, while recent Large Language Model (LLM)-based approaches suffer from
hallucinations and limited controllability. In this paper, we propose ReqInOne,
an LLM-based agent that follows the common steps taken by human requirements
engineers when writing an SRS to convert natural language into a structured
SRS. ReqInOne adopts a modular architecture by decomposing SRS generation into
three tasks: summary, requirement extraction, and requirement classification,
each supported by tailored prompt templates to improve the quality and
consistency of LLM outputs.
  We evaluate ReqInOne using GPT-4o, LLaMA 3, and DeepSeek-R1, and compare the
generated SRSs against those produced by the holistic GPT-4-based method from
prior work as well as by entry-level requirements engineers. Expert evaluations
show that ReqInOne produces more accurate and well-structured SRS documents.
The performance advantage of ReqInOne benefits from its modular design, and
experimental results further demonstrate that its requirement classification
component achieves comparable or even better results than the state-of-the-art
requirement classification model.

### 2. [Inclusive Employment Pathways: Career Success Factors for Autistic Individuals in Software Engineering](http://arxiv.org/pdf/2508.09680v1)

Authors: Orvila Sarker, Mona Jamshaid, M. Ali Babar

Research has highlighted the valuable contributions of autistic individuals
in the Information and Communication Technology (ICT) sector, particularly in
areas such as software development, testing, and cybersecurity. Their strengths
in information processing, attention to detail, innovative thinking, and
commitment to high-quality outcomes in the ICT domain are well-documented.
However, despite their potential, autistic individuals often face barriers in
Software Engineering (SE) roles due to a lack of personalised tools, complex
work environments, non-inclusive recruitment practices, limited co-worker
support, challenging social dynamics and so on. Motivated by the ethical
framework of the neurodiversity movement and the success of pioneering
initiatives like the Dandelion program, corporate Diversity, Equity, and
Inclusion (DEI) in the ICT sector has increasingly focused on autistic talent.
This movement fundamentally reframes challenges not as individual deficits but
as failures of environments designed for a neurotypical majority. Despite this
progress, there is no synthesis of knowledge reporting the full pathway from
software engineering education through to sustainable workplace inclusion. To
address this, we conducted a Systematic Review of 30 studies and identified 18
success factors grouped into four thematic categories: (1) Software Engineering
Education, (2) Career and Employment Training, (3) Work Environment, and (4)
Tools and Assistive Technologies. Our findings offer evidence-based
recommendations for educational institutions, employers, organisations, and
tool developers to enhance the inclusion of autistic individuals in SE. These
include strategies for inclusive meeting and collaboration practices,
accessible and structured work environments, clear role and responsibility
definitions, and the provision of tailored workplace accommodations.

### 3. [Fast and Accurate Heuristics for Bus-Factor Estimation](http://arxiv.org/pdf/2508.09828v1)

Authors: Sebastiano Antonio Piccolo

The bus-factor is a critical risk indicator that quantifies how many key
contributors a project can afford to lose before core knowledge or
functionality is compromised. Despite its practical importance, accurately
computing the bus-factor is NP-Hard under established formalizations, making
scalable analysis infeasible for large software systems.
  In this paper, we model software projects as bipartite graphs of developers
and tasks and propose two novel approximation heuristics, Minimum Coverage and
Maximum Coverage, based on iterative graph peeling, for two influential
bus-factor formalizations. Our methods significantly outperform the widely
adopted degree-based heuristic, which we show can yield severely inflated
estimates.
  We conduct a comprehensive empirical evaluation on over $1\,000$ synthetic
power-law graphs and demonstrate that our heuristics provide tighter estimates
while scaling to graphs with millions of nodes and edges in minutes. Our
results reveal that the proposed heuristics are not only more accurate but also
robust to structural variations in developer-task assignment graph. We release
our implementation as open-source software to support future research and
practical adoption.

### 4. [An Empirical Study of CGO Usage in Go Projects -- Distribution, Purposes, Patterns and Critical Issues](http://arxiv.org/pdf/2508.09875v1)

Authors: Jinbao Chen, Boyao Ding, Yu Zhang, Qingwei Li, Fugen Tang

Multilingual software development integrates multiple languages into a single
application, with the Foreign Function Interface (FFI) enabling seamless
interaction. While FFI boosts efficiency and extensibility, it also introduces
risks. Existing studies focus on FFIs in languages like Python and Java,
neglecting CGO, the emerging FFI in Go, which poses unique risks.
  To address these concerns, we conduct an empirical study of CGO usage across
920 open-source Go projects. Our study aims to reveal the distribution,
patterns, purposes, and critical issues associated with CGO, offering insights
for developers and the Go team. We develop CGOAnalyzer, a tool to efficiently
identify and quantify CGO-related features. Our findings reveal that: (1) 11.3%
of analyzed Go projects utilize CGO, with usage concentrated in a subset of
projects; (2) CGO serves 4 primary purposes, including system-level
interactions and performance optimizations, with 15 distinct usage patterns
observed; (3) 19 types of CGO-related issues exist, including one critical
issue involving unnecessary pointer checks that pose risks of runtime crashes
due to limitations in the current Go compilation toolchain; (4) a temporary
solution reduces unnecessary pointer checks, mitigating crash risks, and (5) we
submitted a proposal to improve the Go toolchain for a permanent fix, which has
been grouped within an accepted proposal for future resolution. Our findings
provide valuable insights for developers and the Go team, enhancing development
efficiency and reliability while improving the robustness of the Go toolchain.

### 5. [Your Coding Intent is Secretly in the Context and You Should Deliberately Infer It Before Completion](http://arxiv.org/pdf/2508.09537v1)

Authors: Yanzhou Li, Tianlin Li, Yiran Zhang, Shangqing Liu, Aishan Liu, Yang Liu

Large Language Models (LLMs) are increasingly used for function completion in
repository-scale codebases. Prior studies demonstrate that when explicit
instructions--such as docstrings--are provided, these models can generate
highly accurate implementations. However, in real-world repositories, such
annotations are frequently absent, and performance drops substantially without
them. To address this gap, we frame the task as a three-stage process. The
first stage focuses on intent inference, where the model analyzes the code
preceding the target function to uncover cues about the desired functionality.
Such preceding context often encodes subtle but critical information, and we
design a reasoning-based prompting framework to guide the LLM through
step-by-step extraction and synthesis of these signals before any code is
generated. The second stage introduces an optional interactive refinement
mechanism to handle cases where preceding context alone is insufficient for
intent recovery. In this stage, the model proposes a small set of candidate
intentions, enabling the developer to select or edit them so that the inferred
intent closely matches the actual requirement. Finally, in the third stage, the
LLM generates the target function conditioned on the finalized intent. To
support this pipeline, we curate a dataset of 40,000 examples annotated with
intermediate reasoning traces and corresponding docstrings. Extensive
experiments on DevEval and ComplexCodeEval show that our approach consistently
boosts multiple LLMs, achieving over 20\% relative gains in both
reference-based and execution-based metrics, with the interactive refinement
stage delivering additional improvements beyond these gains.

### 6. [DeputyDev -- AI Powered Developer Assistant: Breaking the Code Review Logjam through Contextual AI to Boost Developer Productivity](http://arxiv.org/pdf/2508.09676v1)

Authors: Vishal Khare, Vijay Saini, Deepak Sharma, Anand Kumar, Ankit Rana, Anshul Yadav

This study investigates the implementation and efficacy of DeputyDev, an
AI-powered code review assistant developed to address inefficiencies in the
software development process. The process of code review is highly inefficient
for several reasons, such as it being a time-consuming process, inconsistent
feedback, and review quality not being at par most of the time. Using our
telemetry data, we observed that at TATA 1mg, pull request (PR) processing
exhibits significant inefficiencies, with average pick-up and review times of
73 and 82 hours, respectively, resulting in a 6.2 day closure cycle. The review
cycle was marked by prolonged iterative communication between the reviewing and
submitting parties. Research from the University of California, Irvine
indicates that interruptions can lead to an average of 23 minutes of lost
focus, critically affecting code quality and timely delivery. To address these
challenges, we developed DeputyDev's PR review capabilities by providing
automated, contextual code reviews. We conducted a rigorous double-controlled
A/B experiment involving over 200 engineers to evaluate DeputyDev's impact on
review times. The results demonstrated a statistically significant reduction in
both average per PR (23.09%) and average per-line-of-code (40.13%) review
durations. After implementing safeguards to exclude outliers, DeputyDev has
been effectively rolled out across the entire organisation. Additionally, it
has been made available to external companies as a Software-as-a-Service (SaaS)
solution, currently supporting the daily work of numerous engineering
professionals. This study explores the implementation and effectiveness of
AI-assisted code reviews in improving development workflow timelines and code.

### 7. [LibRec: Benchmarking Retrieval-Augmented LLMs for Library Migration Recommendations](http://arxiv.org/pdf/2508.09791v1)

Authors: Junxiao Han, Yarong Wang, Xiaodong Gu, Cuiyun Gao, Yao Wan, Song Han, David Lo, Shuiguang Deng

In this paper, we propose LibRec, a novel framework that integrates the
capabilities of LLMs with retrieval-augmented generation(RAG) techniques to
automate the recommendation of alternative libraries. The framework further
employs in-context learning to extract migration intents from commit messages
to enhance the accuracy of its recommendations. To evaluate the effectiveness
of LibRec, we introduce LibEval, a benchmark designed to assess the performance
in the library migration recommendation task. LibEval comprises 2,888 migration
records associated with 2,368 libraries extracted from 2,324 Python
repositories. Each migration record captures source-target library pairs, along
with their corresponding migration intents and intent types. Based on LibEval,
we evaluated the effectiveness of ten popular LLMs within our framework,
conducted an ablation study to examine the contributions of key components
within our framework, explored the impact of various prompt strategies on the
framework's performance, assessed its effectiveness across various intent
types, and performed detailed failure case analyses.

### 8. [Exploring the Potential of Large Language Models in Fine-Grained Review Comment Classification](http://arxiv.org/pdf/2508.09832v1)

Authors: Linh Nguyen, Chunhua Liu, Hong Yi Lin, Patanamon Thongtanunam

Code review is a crucial practice in software development. As code review
nowadays is lightweight, various issues can be identified, and sometimes, they
can be trivial. Research has investigated automated approaches to classify
review comments to gauge the effectiveness of code reviews. However, previous
studies have primarily relied on supervised machine learning, which requires
extensive manual annotation to train the models effectively. To address this
limitation, we explore the potential of using Large Language Models (LLMs) to
classify code review comments. We assess the performance of LLMs to classify 17
categories of code review comments. Our results show that LLMs can classify
code review comments, outperforming the state-of-the-art approach using a
trained deep learning model. In particular, LLMs achieve better accuracy in
classifying the five most useful categories, which the state-of-the-art
approach struggles with due to low training examples. Rather than relying
solely on a specific small training data distribution, our results show that
LLMs provide balanced performance across high- and low-frequency categories.
These results suggest that the LLMs could offer a scalable solution for code
review analytics to improve the effectiveness of the code review process.

### 9. [ARI3D: A Software for Interactive Quantification of Regions in X-Ray CT 3D Images](http://arxiv.org/pdf/2508.09849v1)

Authors: Jan Phillipp Albrecht, Jose R. A. Godinho, Christina Hübers, Deborah Schmidt

X-ray computed tomography (CT) is the main 3D technique for imaging the
internal microstructures of materials. Quantitative analysis of the
microstructures is usually achieved by applying a sequence of steps that are
implemented to the entire 3D image. This is challenged by various imaging
artifacts inherent from the technique, e.g., beam hardening and partial volume.
Consequently, the analysis requires users to make a number of decisions to
segment and classify the microstructures based on the voxel gray-values. In
this context, a software tool, here called ARI3D, is proposed to interactively
analyze regions in three-dimensional X-ray CT images, assisting users through
the various steps of a protocol designed to classify and quantify objects
within regions of a three-dimensional image. ARI3D aims to 1) Improve phase
identification; 2) Account for partial volume effect; 3) Increase the detection
limit and accuracy of object quantification; and 4) Harmonize quantitative 3D
analysis that can be implemented in different fields of science.

### 10. [Extending the OWASP Multi-Agentic System Threat Modeling Guide: Insights from Multi-Agent Security Research](http://arxiv.org/pdf/2508.09815v1)

Authors: Klaudia Krawiecka, Christian Schroeder de Witt

We propose an extension to the OWASP Multi-Agentic System (MAS) Threat
Modeling Guide, translating recent anticipatory research in multi-agent
security (MASEC) into practical guidance for addressing challenges unique to
large language model (LLM)-driven multi-agent architectures. Although OWASP's
existing taxonomy covers many attack vectors, our analysis identifies gaps in
modeling failures, including, but not limited to: reasoning collapse across
planner-executor chains, metric overfitting, unsafe delegation escalation,
emergent covert coordination, and heterogeneous multi-agent exploits. We
introduce additional threat classes and scenarios grounded in practical MAS
deployments, highlighting risks from benign goal drift, cross-agent
hallucination propagation, affective prompt framing, and multi-agent backdoors.
We also outline evaluation strategies, including robustness testing,
coordination assessment, safety enforcement, and emergent behavior monitoring,
to ensure complete coverage. This work complements the framework of OWASP by
expanding its applicability to increasingly complex, autonomous, and adaptive
multi-agent systems, with the goal of improving security posture and resilience
in real world deployments.

### Social and Information Networks

### 1. [Efficient Integration of Multi-View Attributed Graphs for Clustering and Embedding](http://arxiv.org/pdf/2508.09452v1)

Authors: Yiran Li, Gongyao Guo, Jieming Shi, Sibo Wang, Qing Li

A multi-view attributed graph (MVAG) G captures the diverse relationships and
properties of real-world entities through multiple graph views and attribute
views. Effectively utilizing all views in G is essential for MVAG clustering
and embedding, which are important for applications like recommendation
systems, anomaly detection, social network analysis, etc. Existing methods
either achieve inferior result quality or incur significant computational costs
to handle large-scale MVAGs.
  In this paper, we present a spectrum-guided Laplacian aggregation scheme with
an effective objective formulation and two efficient algorithms SGLA and SGLA+,
to cohesively integrate all views of G into an MVAG Laplacian matrix, which
readily enables classic graph algorithms to handle G with superior performance
in clustering and embedding tasks. We begin by conducting a theoretical
analysis to design an integrated objective that consists of two components, the
eigengap and connectivity objectives, aiming to link the spectral properties of
the aggregated MVAG Laplacian with the underlying community and connectivity
properties of G. A constrained optimization problem is then formulated for the
integration, which is computationally expensive to solve. Thus, we first
develop the SGLA algorithm, which already achieves excellent performance
compared with existing methods. To further enhance efficiency, we design SGLA+
to reduce the number of costly objective evaluations via sampling and
approximation to quickly find an approximate optimum. Extensive experiments
compare our methods against 12 baselines for clustering and 8 baselines for
embedding on 8 multi-view attributed graphs, validating the superior
performance of SGLA and SGLA+ in terms of result quality and efficiency.
Compared with the most effective baselines, our methods are significantly
faster, often by up to orders of magnitude.

### 2. [CS-Agent: LLM-based Community Search via Dual-agent Collaboration](http://arxiv.org/pdf/2508.09549v1)

Authors: Jiahao Hua, Long Yuan, Qingshuai Feng, Qiang Fang, Shan Huang

Large Language Models (LLMs) have demonstrated remarkable capabilities in
natural language processing tasks, yet their application to graph structure
analysis, particularly in community search, remains underexplored. Community
search, a fundamental task in graph analysis, aims to identify groups of nodes
with dense interconnections, which is crucial for understanding the macroscopic
structure of graphs. In this paper, we propose GraphCS, a comprehensive
benchmark designed to evaluate the performance of LLMs in community search
tasks. Our experiments reveal that while LLMs exhibit preliminary potential,
they frequently fail to return meaningful results and suffer from output bias.
To address these limitations, we introduce CS-Agent, a dual-agent collaborative
framework to enhance LLM-based community search. CS-Agent leverages the
complementary strengths of two LLMs acting as Solver and Validator. Through
iterative feedback and refinement, CS-Agent dynamically refines initial results
without fine-tuning or additional training. After the multi-round dialogue,
Decider module selects the optimal community. Extensive experiments demonstrate
that CS-Agent significantly improves the quality and stability of identified
communities compared to baseline methods. To our knowledge, this is the first
work to apply LLMs to community search, bridging the gap between LLMs and graph
analysis while providing a robust and adaptive solution for real-world
applications.

### 3. [Social-Sensor Identity Cloning Detection Using Weakly Supervised Deep Forest and Cryptographic Authentication](http://arxiv.org/pdf/2508.09665v1)

Authors: Ahmed Alharbi, Hai Dong, Xun Yi

Recent years have witnessed a rising trend in social-sensor cloud identity
cloning incidents. However, existing approaches suffer from unsatisfactory
performance, a lack of solutions for detecting duplicated accounts, and a lack
of large-scale evaluations on real-world datasets. We introduce a novel method
for detecting identity cloning in social-sensor cloud service providers. Our
proposed technique consists of two primary components: 1) a similar identity
detection method and 2) a cryptography-based authentication protocol.
Initially, we developed a weakly supervised deep forest model to identify
similar identities using non-privacy-sensitive user profile features provided
by the service. Subsequently, we designed a cryptography-based authentication
protocol to verify whether similar identities were generated by the same
provider. Our extensive experiments on a large real-world dataset demonstrate
the feasibility and superior performance of our technique compared to current
state-of-the-art identity clone detection methods.

### Systems and Control

### 1. [Design and Simulation of 6T SRAM Array](http://arxiv.org/pdf/2508.09419v1)

Authors: Justin London

Conventional 6T SRAM is used in microprocessors in the cache memory design.
The basic 6T SRAM cell and a 6 bit memory array layout are designed in LEdit.
The design and analysis of key SRAM components, sense amplifiers, decoders,
write drivers and precharge circuits are also provided. The pulse voltage
waveforms generated for read and write operations as well as Q and Qbar nodes
are simulated in LTSpice. Parasitic capacitances are extracted and their impact
on the waveforms analyzed. Static noise margin, propagation delays, and power
dissipation are calculated. Comparison of SRAM read and write operational
performance using CMOS transistors is made with edge-triggered D flip flops. If
certain size area and ratio constraints are satisfied, the 6T cell with CMOS
transistors will possess stability, speed, and power efficiency. Both
theoretical and simulated results are given.

### 2. [Control Systems Analysis of a 3-Axis Photovoltatic Solar Tracker for Water Pumping](http://arxiv.org/pdf/2508.09420v1)

Authors: Justin London

We propose 3-axis solar tracker water pumping system. The solar tracker can
rotate and tilt using stepper/DC motors and can rise and lower on a tripod
using a linear actuator. The charge generated from solar energy absorbed by
photovoltaic (PV) cells in the solar panel is stored in a 12V battery that in
turn powers two water diaphragm pumps using a solar charge controller. The PV
uses four light photocell resistors/sensors to measure light intensity. A solar
tracking algorithm determines the optimal angle for PV positioning. Using an
ultrasonic sensor to measure the water level in a reservoir water tank, water
is pumped from one water tank to the reservoir. Based on soil moisture sensor
levels, a second water pump supplies water from the reservoir to the plant. The
system is analyzed from a control systems perspective. The transfer functions,
root loci, and Bode plots are generated and simulated and experimental results
are provided as well as stability and steady-state error analysis.

### 3. [Imperfect Competition in Markets for Short-Circuit Current Services](http://arxiv.org/pdf/2508.09425v1)

Authors: Peng Wang, Luis Badesa

An important limitation of Inverter-Based Resources (IBR) is their reduced
contribution to Short-Circuit Current (SCC), as compared to that of Synchronous
Generators (SGs). With increasing penetration of IBR in most power systems, the
reducing SCC poses challenges to a secure system operation, as line protections
may not trip when required. In order to address this issue, the SCC ancillary
service could be procured via an economic mechanism, aiming at securing
adequate SCC on all buses. However, the suitability of markets for SCC services
is not well understood, given that these could be prone to market-power issues:
since the SCC contributions from various SGs to a certain bus are determined by
the electrical topology of the grid, this is a highly local service. It is
necessary to understand if SGs at advantageous electrical locations could exert
market power and, if so, how it could be mitigated. In order to fill this gap,
this paper adopts an SCC-constrained bilevel model to investigate strategic
behaviors of SGs. To address the non-convexity due to unit commitment
variables, the model is restructured through a primal-dual formulation. Based
on a modified IEEE 30-bus system, cases with strategic SGs placed at different
buses are analyzed. These studies demonstrate that agents exerting market power
could achieve up to triple revenues from SCC provision, highlighting the need
to carefully design these markets.

### 4. [From Micro to Macro Flow Modeling: Characterizing Heterogeneity of Mixed-Autonomy Traffic](http://arxiv.org/pdf/2508.09432v1)

Authors: Chenguang Zhao, Huan Yu

Most autonomous-vehicles (AVs) driving strategies are designed and analyzed
at the vehicle level, yet their aggregate impact on macroscopic traffic flow is
still not understood, particularly the flow heterogeneity that emerges when AVs
interact with human-driven vehicles (HVs). Existing validation techniques for
macroscopic flow models rely on high-resolution spatiotemporal data spanning
entire road segments which are rarely available for mixed-autonomy traffic. AVs
record detailed Lagrangian trajectories of the ego vehicle and surrounding
traffic through onboard sensors. Leveraging these Lagrangian observations to
validate mixed-autonomy flow models therefore remains an open research
challenge. This paper closes the gap between microscopic Lagrangian data and
macroscopic Euclidean traffic models by introducing a continuous
traffic-heterogeneity attribute. We represent traffic flow with two coupled
conservation laws with one for vehicle number and one for the traffic
attribute. Reconstruction methods are designed to derive the traffic attribute
from Lagrangian vehicle trajectories. When abundant trajectory data are
available, we characterize traffic heterogeneity by extracting drivers' desired
speed and local behavioral uncertainty from trajectories. In data-scarce mixed
traffic, we design an end-to-end mapping that infers the traffic heterogeneity
solely from trajectories in the current spatiotemporal region. Experiments
across multiple traffic datasets show that the proposed model effectively
captures traffic heterogeneity by clustering the fundamental diagram scatter
into attribute-based groups. The calibration errors of traffic flow dynamics
are also reduce by 20% relative to the Aw-Rascle-Zhang model benchmark.
Detailed analyses further show that the model generalizes well, maintaining
nearly the same accuracy when evaluated under a variety of previously unseen
traffic conditions.

### 5. [From Formal Methods to Data-Driven Safety Certificates of Unknown Large-Scale Networks](http://arxiv.org/pdf/2508.09520v1)

Authors: Omid Akbarzadeh, Behrad Samari, Amy Nejati, Abolfazl Lavaei

In this work, we propose a data-driven scheme within a compositional
framework with noisy data to design robust safety controllers in a fully
decentralized fashion for large-scale interconnected networks with unknown
mathematical dynamics. Despite the network's high dimensionality and the
inherent complexity of its unknown model, which make it intractable, our
approach effectively addresses these challenges by (i) treating the network as
a composition of smaller subsystems, and (ii) collecting noisy data from each
subsystem's trajectory to design a control sub-barrier certificate (CSBC) and
its corresponding local controller. To achieve this, our proposed scheme only
requires a noise-corrupted single input-state trajectory from each unknown
subsystem up to a specified time horizon, satisfying a certain rank condition.
Subsequently, under a small-gain compositional reasoning, we compose those
CSBC, derived from noisy data, and formulate a control barrier certificate
(CBC) for the unknown network, ensuring its safety over an infinite time
horizon, while providing correctness guarantees. We offer a data-dependent
sum-of-squares (SOS) optimization program for computing CSBC alongside local
controllers of subsystems. We illustrate that while the computational
complexity of designing a CBC and its safety controller grows polynomially with
network dimension using SOS optimization, our compositional data-driven
approach significantly reduces it to a linear scale concerning the number of
subsystems. We demonstrate the capability of our data-driven approach on
multiple physical networks involving unknown models and a range of
interconnection topologies.

### 6. [Shepherd Grid Strategy: Towards Reliable SWARM Interception](http://arxiv.org/pdf/2508.09536v1)

Authors: Boris Kriuk, Fedor Kriuk

Modern unmanned aerial vehicle threats require sophisticated interception
strategies that can overcome advanced evasion capabilities and operate
effectively in contested environments. Traditional single-interceptor and
uncoordinated multi-interceptor approaches suffer from fundamental limitations
including inadequate coverage, predictable pursuit patterns, and vulnerability
to intelligent evasion maneuvers. This paper introduces the Shepherd Grid
Strategy, a new multi-phase coordination framework that employs pack-based
behavioral coordination to achieve deterministic target interception through
systematic containment and coordinated strike execution. The strategy
implements a four-phase operational model consisting of chase, follow,
formation, and engagement phases, with dynamic role assignment and adaptive
formation geometry that maintains persistent target pressure while preparing
optimal strike opportunities. Our approach incorporates three key innovations:
adaptive phase transition mechanisms that optimize pursuit behavior based on
proximity and mission objectives, dynamic role assignment systems that
designate specialized interceptor functions including formation maintenance and
strike execution, and predictive formation geometry algorithms that create
mobile containment grids adapting to target movement patterns. The simulation
experiments demonstrate significant performance improvements over traditional
methods, achieving near-perfect interception success rates (over 95%) compared
to traditional approaches (65%) and reducing median time-to-intercept.

### 7. [Metering traffic flows for perimeter control through auction-based signalling using connected vehicles](http://arxiv.org/pdf/2508.09678v1)

Authors: Alexander Roocroft, Marco Rinaldi

Urban traffic congestion remains a critical challenge in modern cities, with
traffic signal control systems often struggling to manage congestion during
peak travel times. Perimeter control of a Protected Network (PN) has emerged as
a potential solution to reducing gridlock in urban networks. This paper
proposes a novel auction-based mechanism for green time allocation at
signalized intersections, for effective perimeter control application.
Utilising a Sealed Bid, Second Price auction framework, our approach combines
real-time traffic monitoring with market-inspired mechanisms to regulate
vehicle inflows into PN areas. Unlike existing methods that focus primarily on
gated links, our system allocates budgets to individual traffic movements,
providing greater flexibility in managing multi-directional flows. We evaluate
the proposed mechanism using a test case intersection with a single controlled
inflow, comparing it against a volume-based fixed-time approach. The results
demonstrate that our auction-based method controls flows into the PN with
improved accuracy, outperforming the volume-based approach in terms of inflow
regulation, queue management and delays. The framework can be applied in real
time to any generic intersection, offering a scalable solution for urban
traffic management. This work bridges the gap between perimeter control and
market-based intersection auctions, providing a pathway for further research on
adaptive traffic management systems.

### 8. [A Divide-and-Conquer Tiling Method for the Design of Large Aperiodic Phased Arrays](http://arxiv.org/pdf/2508.09682v1)

Authors: Nicola Anselmi, Paolo Rocca, Giovanni Toso, Andrea Massa

Due to the growing request from modern wireless applications of
cost-affordable and high-gain scanning antenna solutions, the design of large
phased arrays (PAs) with radiating elements organized into modular clusters
with sub-array-only amplitude and phase control is a key topic. In this paper,
an innovative irregular tiling method is proposed where, according to a
divide-and-conquer strategy, the antenna aperture is subdivided into sub-areas
that are locally domino-tiled by jointly fulfilling the full-coverage condition
on the remaining untiled part of the PA support. Selected representative
results, including comparisons with competitive state-of-the-art synthesis
methods, are reported to prove the effectiveness and the computational
efficiency of the proposed tiling approach. Use-cases of current relevance for
low Earth orbit (LEO) satellite communications are discussed, as well, to
provide the antenna designers useful practical guidelines for handling large
PAs.

### 9. [Besondere Anforderungen des automatisierten Fahrens an den Entwurf](http://arxiv.org/pdf/2508.09731v1)

Authors: Robert Graubohm, Markus Maurer

The development of automated vehicles and automated driving functions is an
exceptionally complex task that requires the integration of numerous, sometimes
conflicting interests and various constraints already in the early stages of
system design. This chapter explains important challenges in concept
specifications for automated driving and presents a systematic process model
that contributes to overcoming the special requirements in this field. In
addition, it describes the successful implementation of a structured concept
specification for an automated vehicle guidance system.
  --
  Die Entwicklung automatisierter Fahrzeuge und Fahrfunktionen stellt eine
ausgesprochen komplexe Aufgabe dar, die bereits im Zuge des Systementwurfs die
Einbeziehung einer Vielzahl teilweise konflikt\"arer Interessen und diverser
Randbedingungen erfordert. Dieses Kapitel erl\"autert wichtige
Herausforderungen bei Konzeptspezifikationen im Themenfeld des automatisierten
Fahrens und stellt ein systematisches Prozessmodell vor, das einen Beitrag zur
Erf\"ullung der besonderen Anforderungen des automatisierten Fahrens an den
Entwurf leistet. Dar\"uber hinaus wird die erfolgreiche Durchf\"uhrung einer
strukturierten Konzeptspezifikation f\"ur ein automatisiertes
Fahrzeugf\"uhrungssystem beschrieben.

### 10. [Integrated Learning and Optimization to Control Load Demand and Wind Generation for Minimizing Ramping Cost in Real-Time Electricity Market](http://arxiv.org/pdf/2508.09774v1)

Authors: Imran Pervez, Omar Knio

We developed a new integrated learning and optimization (ILO) methodology to
predict context-aware unknown parameters in economic dispatch (ED), a crucial
problem in power systems solved to generate optimal power dispatching decisions
to serve consumer load. The ED formulation in the current study consists of
load and renewable generation as unknown parameters in its constraints
predicted using contextual information (e.g., prior load, temperature). The ILO
framework train a neural network (NN) to estimate ED parameters by minimizing
an application-specific regret function which is a difference between ground
truth and NN-driven decisions favouring better ED decisions. We thoroughly
analyze the feasible region of ED formulation to understand the impact of load
and renewable learning together on the ED decisions. Corresponding to that we
developed a new regret function to capture real-time electricity market
operations where differences in predicted and true loads are corrected by
ramping generators in real-time but at a higher cost than the market price. The
proposed regret function when minimized using ILO framework train the NN to
guide the load and renewable predictions to generate ED decisions favouring
minimum generator ramping costs. This is unlike conventional sequential
learning and optimization (SLO) framework which train NN to accurately estimate
load and renewable instead of better ED decisions. The combined training of
load and renewable using ILO is a new concept and lead to significantly
improved ramping costs when compared with SLO based training of load and
renewable and SLO trained load with 100% accurate renewable proving its
decision-focused capability.

### Machine Learning (Statistics Category)

### 1. [Scalable h-adaptive probabilistic solver for time-independent and time-dependent systems](http://arxiv.org/pdf/2508.09623v1)

Authors: Akshay Thakur, Sawan Kumar, Matthew Zahr, Souvik Chakraborty

Solving partial differential equations (PDEs) within the framework of
probabilistic numerics offers a principled approach to quantifying epistemic
uncertainty arising from discretization. By leveraging Gaussian process
regression and imposing the governing PDE as a constraint at a finite set of
collocation points, probabilistic numerics delivers mesh-free solutions at
arbitrary locations. However, the high computational cost, which scales
cubically with the number of collocation points, remains a critical bottleneck,
particularly for large-scale or high-dimensional problems. We propose a
scalable enhancement to this paradigm through two key innovations. First, we
develop a stochastic dual descent algorithm that reduces the per-iteration
complexity from cubic to linear in the number of collocation points, enabling
tractable inference. Second, we exploit a clustering-based active learning
strategy that adaptively selects collocation points to maximize information
gain while minimizing computational expense. Together, these contributions
result in an $h$-adaptive probabilistic solver that can scale to a large number
of collocation points. We demonstrate the efficacy of the proposed solver on
benchmark PDEs, including two- and three-dimensional steady-state elliptic
problems, as well as a time-dependent parabolic PDE formulated in a space-time
setting.

### 2. [Structured Kernel Regression VAE: A Computationally Efficient Surrogate for GP-VAEs in ICA](http://arxiv.org/pdf/2508.09721v1)

Authors: Yuan-Hao Wei, Fu-Hao Deng, Lin-Yong Cui, Yan-Jie Sun

The interpretability of generative models is considered a key factor in
demonstrating their effectiveness and controllability. The generated data are
believed to be determined by latent variables that are not directly observable.
Therefore, disentangling, decoupling, decomposing, causal inference, or
performing Independent Component Analysis (ICA) in the latent variable space
helps uncover the independent factors that influence the attributes or features
affecting the generated outputs, thereby enhancing the interpretability of
generative models. As a generative model, Variational Autoencoders (VAEs)
combine with variational Bayesian inference algorithms. Using VAEs, the inverse
process of ICA can be equivalently framed as a variational inference process.
In some studies, Gaussian processes (GPs) have been introduced as priors for
each dimension of latent variables in VAEs, structuring and separating each
dimension from temporal or spatial perspectives, and encouraging different
dimensions to control various attributes of the generated data. However, GPs
impose a significant computational burden, resulting in substantial resource
consumption when handling large datasets. Essentially, GPs model different
temporal or spatial structures through various kernel functions. Structuring
the priors of latent variables via kernel functions-so that different kernel
functions model the correlations among sequence points within different latent
dimensions-is at the core of achieving disentanglement in VAEs. The proposed
Structured Kernel Regression VAE (SKR-VAE) leverages this core idea in a more
efficient way, avoiding the costly kernel matrix inversion required in GPs.
This research demonstrates that, while maintaining ICA performance, SKR-VAE
achieves greater computational efficiency and significantly reduced
computational burden compared to GP-VAE.

### 3. [A pseudo-inverse of a line graph](http://arxiv.org/pdf/2508.09412v1)

Authors: Sevvandi Kandanaarachchi, Philip Kilby, Cheng Soon Ong

Line graphs are an alternative representation of graphs where each vertex of
the original (root) graph becomes an edge. However not all graphs have a
corresponding root graph, hence the transformation from graphs to line graphs
is not invertible. We investigate the case when there is a small perturbation
in the space of line graphs, and try to recover the corresponding root graph,
essentially defining the inverse of the line graph operation. We propose a
linear integer program that edits the smallest number of edges in the line
graph, that allow a root graph to be found. We use the spectral norm to
theoretically prove that such a pseudo-inverse operation is well behaved.
Illustrative empirical experiments on Erd\H{o}s-R\'enyi graphs show that our
theoretical results work in practice.

### 4. [Temporal Anchoring in Deepening Embedding Spaces: Event-Indexed Projections, Drift, Convergence, and an Internal Computational Architecture](http://arxiv.org/pdf/2508.09693v1)

Authors: Faruk Alpay, Bugra Kilictas, Hamdi Alakkad

We develop an operator-theoretic framework for temporal anchoring in
embedding spaces, modeled as drift maps interleaved with event-indexed blocks
culminating in affine projections. We provide complete proofs for a
variable-block contraction lemma (products of Lipschitz factors), a
drift--projection convergence theorem with explicit uniform-gap envelopes, and
ontological convergence under nested affine anchors with a robustness variant.
We formalize an internal Manuscript Computer (MC) whose computations are
defined purely by these operators and prove a rigorous finite-run equivalence
theorem (with perturbation bounds). For attention layers, we give a
self-contained proof that softmax is $1/2$-Lipschitz in $\ell_2$ and derive
sufficient layer-contraction conditions (orthogonal/non-orthogonal heads). All
floats are placed exactly where written; the manuscript uses only in-paper
pseudocode and appendix figures.

### 5. [Bayesian autoregression to optimize temporal Matérn kernel Gaussian process hyperparameters](http://arxiv.org/pdf/2508.09792v1)

Authors: Wouter M. Kouw

Gaussian processes are important models in the field of probabilistic
numerics. We present a procedure for optimizing Mat\'ern kernel temporal
Gaussian processes with respect to the kernel covariance function's
hyperparameters. It is based on casting the optimization problem as a recursive
Bayesian estimation procedure for the parameters of an autoregressive model. We
demonstrate that the proposed procedure outperforms maximizing the marginal
likelihood as well as Hamiltonian Monte Carlo sampling, both in terms of
runtime and ultimate root mean square error in Gaussian process regression.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-08-14 PST.

### 1. [Real AI advances require collaboration](https://www.nature.com/articles/s41570-025-00750-2)

Authors: N. M. Anoop Krishnan et al.

### 2. [Quantum granular-ball generation methods and their application in KNN classification](https://www.nature.com/articles/s41598-025-14724-3)

Authors: Suzhen Yuan et al.

### 3. [Network intrusion detection based on improved KNN algorithm](https://www.nature.com/articles/s41598-025-14199-2)

Authors: Hongsheng Bao et al.

### 4. [Mechanistic understanding and validation of large AI models with SemanticLens](https://www.nature.com/articles/s42256-025-01084-w)

Authors: Maximilian Dreyer et al.

### 5. [A graph attention network-based multi-agent reinforcement learning framework for robust detection of smart contract vulnerabilities](https://www.nature.com/articles/s41598-025-14032-w)

Authors: Philip Kwaku Adjei et al.

### 6. [Advanced machine learning framework for thyroid cancer epidemiology in Iran through integration of environmental socioeconomic and health system predictors](https://www.nature.com/articles/s41598-025-15324-x)

Authors: Mohsen Soleimani et al.

### 7. [Analysis of deep learning-based technological innovation governance on the intelligent allocation of innovation resources in the high-technology industry](https://www.nature.com/articles/s41598-025-15374-1)

Authors: Xiaoxuan Yu et al.

### 8. [Revisiting model scaling with a U-net benchmark for 3D medical image segmentation](https://www.nature.com/articles/s41598-025-15617-1)

Authors: Ziyan Huang et al.

### 9. [An integrated microwave neural network for broadband computation and communication](https://www.nature.com/articles/s41928-025-01422-1)

Authors: Bala Govind et al.

### 10. [Development and validation of a competency-based ladder pathway for AI literacy enhancement among higher vocational students](https://www.nature.com/articles/s41598-025-15202-6)

Authors: Litian Hong

### 11. [Diverse behavior clustering of students on campus with macroscopic attention](https://www.nature.com/articles/s41598-025-15103-8)

Authors: Wanghu Chen et al.

### 12. [DGS-Yolov7-Tiny: a lightweight pest and disease target detection model suitable for edge computing environments](https://www.nature.com/articles/s41598-025-13410-8)

Authors: Ping Yu et al.

### 13. [Research on quality and safety risk identification of import and export toys based on the WOA-BP model](https://www.nature.com/articles/s41598-025-15332-x)

Authors: Qiong He et al.

### 14. [Large language model driven transferable key information extraction mechanism for nonstandardized tables](https://www.nature.com/articles/s41598-025-15627-z)

Authors: Rong Hu et al.

