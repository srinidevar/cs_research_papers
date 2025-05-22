# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-21 18:36:46.337678 PST.

### Artificial Intelligence

### 1. [LLM-based Evaluation Policy Extraction for Ecological Modeling](http://arxiv.org/pdf/2505.13794v1)

Authors: Qi Cheng, Licheng Liu, Qing Zhu, Runlong Yu, Zhenong Jin, Yiqun Xie, Xiaowei Jia

Evaluating ecological time series is critical for benchmarking model
performance in many important applications, including predicting greenhouse gas
fluxes, capturing carbon-nitrogen dynamics, and monitoring hydrological cycles.
Traditional numerical metrics (e.g., R-squared, root mean square error) have
been widely used to quantify the similarity between modeled and observed
ecosystem variables, but they often fail to capture domain-specific temporal
patterns critical to ecological processes. As a result, these methods are often
accompanied by expert visual inspection, which requires substantial human labor
and limits the applicability to large-scale evaluation. To address these
challenges, we propose a novel framework that integrates metric learning with
large language model (LLM)-based natural language policy extraction to develop
interpretable evaluation criteria. The proposed method processes pairwise
annotations and implements a policy optimization mechanism to generate and
combine different assessment metrics. The results obtained on multiple datasets
for evaluating the predictions of crop gross primary production and carbon
dioxide flux have confirmed the effectiveness of the proposed method in
capturing target assessment preferences, including both synthetically generated
and expert-annotated model comparisons. The proposed framework bridges the gap
between numerical metrics and expert knowledge while providing interpretable
evaluation policies that accommodate the diverse needs of different ecosystem
modeling studies.

### 2. [Multimodal RAG-driven Anomaly Detection and Classification in Laser Powder Bed Fusion using Large Language Models](http://arxiv.org/pdf/2505.13828v1)

Authors: Kiarash Naghavi Khanghah, Zhiling Chen, Lela Romeo, Qian Yang, Rajiv Malhotra, Farhad Imani, Hongyi Xu

Additive manufacturing enables the fabrication of complex designs while
minimizing waste, but faces challenges related to defects and process
anomalies. This study presents a novel multimodal Retrieval-Augmented
Generation-based framework that automates anomaly detection across various
Additive Manufacturing processes leveraging retrieved information from
literature, including images and descriptive text, rather than training
datasets. This framework integrates text and image retrieval from scientific
literature and multimodal generation models to perform zero-shot anomaly
identification, classification, and explanation generation in a Laser Powder
Bed Fusion setting. The proposed framework is evaluated on four L-PBF
manufacturing datasets from Oak Ridge National Laboratory, featuring various
printer makes, models, and materials. This evaluation demonstrates the
framework's adaptability and generalizability across diverse images without
requiring additional training. Comparative analysis using Qwen2-VL-2B and
GPT-4o-mini as MLLM within the proposed framework highlights that GPT-4o-mini
outperforms Qwen2-VL-2B and proportional random baseline in manufacturing
anomalies classification. Additionally, the evaluation of the RAG system
confirms that incorporating retrieval mechanisms improves average accuracy by
12% by reducing the risk of hallucination and providing additional information.
The proposed framework can be continuously updated by integrating emerging
research, allowing seamless adaptation to the evolving landscape of AM
technologies. This scalable, automated, and zero-shot-capable framework
streamlines AM anomaly analysis, enhancing efficiency and accuracy.

### 3. [TelePlanNet: An AI-Driven Framework for Efficient Telecom Network Planning](http://arxiv.org/pdf/2505.13831v1)

Authors: Zongyuan Deng, Yujie Cai, Qing Liu, Shiyao Mu, Bin Lyu, Zhen Yang

The selection of base station sites is a critical challenge in 5G network
planning, which requires efficient optimization of coverage, cost, user
satisfaction, and practical constraints. Traditional manual methods, reliant on
human expertise, suffer from inefficiencies and are limited to an unsatisfied
planning-construction consistency. Existing AI tools, despite improving
efficiency in certain aspects, still struggle to meet the dynamic network
conditions and multi-objective needs of telecom operators' networks. To address
these challenges, we propose TelePlanNet, an AI-driven framework tailored for
the selection of base station sites, integrating a three-layer architecture for
efficient planning and large-scale automation. By leveraging large language
models (LLMs) for real-time user input processing and intent alignment with
base station planning, combined with training the planning model using the
improved group relative policy optimization (GRPO) reinforcement learning, the
proposed TelePlanNet can effectively address multi-objective optimization,
evaluates candidate sites, and delivers practical solutions. Experiments
results show that the proposed TelePlanNet can improve the consistency to 78%,
which is superior to the manual methods, providing telecom operators with an
efficient and scalable tool that significantly advances cellular network
planning.

### 4. [A Challenge to Build Neuro-Symbolic Video Agents](http://arxiv.org/pdf/2505.13851v1)

Authors: Sahil Shah, Harsh Goel, Sai Shankar Narasimhan, Minkyu Choi, S P Sharan, Oguzhan Akcin, Sandeep Chinchali

Modern video understanding systems excel at tasks such as scene
classification, object detection, and short video retrieval. However, as video
analysis becomes increasingly central to real-world applications, there is a
growing need for proactive video agents for the systems that not only interpret
video streams but also reason about events and take informed actions. A key
obstacle in this direction is temporal reasoning: while deep learning models
have made remarkable progress in recognizing patterns within individual frames
or short clips, they struggle to understand the sequencing and dependencies of
events over time, which is critical for action-driven decision-making.
Addressing this limitation demands moving beyond conventional deep learning
approaches. We posit that tackling this challenge requires a neuro-symbolic
perspective, where video queries are decomposed into atomic events, structured
into coherent sequences, and validated against temporal constraints. Such an
approach can enhance interpretability, enable structured reasoning, and provide
stronger guarantees on system behavior, all key properties for advancing
trustworthy video agents. To this end, we present a grand challenge to the
research community: developing the next generation of intelligent video agents
that integrate three core capabilities: (1) autonomous video search and
analysis, (2) seamless real-world interaction, and (3) advanced content
generation. By addressing these pillars, we can transition from passive
perception to intelligent video agents that reason, predict, and act, pushing
the boundaries of video understanding.

### 5. [Parallel Belief Revision via Order Aggregation](http://arxiv.org/pdf/2505.13914v1)

Authors: Jake Chandler, Richard Booth

Despite efforts to better understand the constraints that operate on
single-step parallel (aka "package", "multiple") revision, very little work has
been carried out on how to extend the model to the iterated case. A recent
paper by Delgrande & Jin outlines a range of relevant rationality postulates.
While many of these are plausible, they lack an underlying unifying
explanation. We draw on recent work on iterated parallel contraction to offer a
general method for extending serial iterated belief revision operators to
handle parallel change. This method, based on a family of order aggregators
known as TeamQueue aggregators, provides a principled way to recover the
independently plausible properties that can be found in the literature, without
yielding the more dubious ones.

### 6. [Visual Instruction Bottleneck Tuning](http://arxiv.org/pdf/2505.13946v1)

Authors: Changdae Oh, Jiatong Li, Shawn Im, Yixuan Li

Despite widespread adoption, multimodal large language models (MLLMs) suffer
performance degradation when encountering unfamiliar queries under distribution
shifts. Existing methods to improve MLLM generalization typically require
either more instruction data or larger advanced model architectures, both of
which incur non-trivial human labor or computational costs. In this work, we
take an alternative approach to enhance the robustness of MLLMs under
distribution shifts, from a representation learning perspective. Inspired by
the information bottleneck (IB) principle, we derive a variational lower bound
of the IB for MLLMs and devise a practical implementation, Visual Instruction
Bottleneck Tuning (Vittle). We then provide a theoretical justification of
Vittle by revealing its connection to an information-theoretic robustness
metric of MLLM. Empirical validation of three MLLMs on open-ended and
closed-form question answering and object hallucination detection tasks over 45
datasets, including 30 shift scenarios, demonstrates that Vittle consistently
improves the MLLM's robustness under shifts by pursuing the learning of a
minimal sufficient representation.

### 7. [Memory Assignment for Finite-Memory Strategies in Adversarial Patrolling Games](http://arxiv.org/pdf/2505.14137v1)

Authors: Vojtěch Kůr, Vít Musil, Vojtěch Řehák

Adversarial Patrolling games form a subclass of Security games where a
Defender moves between locations, guarding vulnerable targets. The main
algorithmic problem is constructing a strategy for the Defender that minimizes
the worst damage an Attacker can cause. We focus on the class of finite-memory
(also known as regular) Defender's strategies that experimentally outperformed
other competing classes. A finite-memory strategy can be seen as a positional
strategy on a finite set of states. Each state consists of a pair of a location
and a certain integer value--called memory. Existing algorithms improve the
transitional probabilities between the states but require that the available
memory size itself is assigned at each location manually. Choosing the right
memory assignment is a well-known open and hard problem that hinders the
usability of finite-memory strategies. We solve this issue by developing a
general method that iteratively changes the memory assignment. Our algorithm
can be used in connection with \emph{any} black-box strategy optimization tool.
We evaluate our method on various experiments and show its robustness by
solving instances of various patrolling models.

### 8. [RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning](http://arxiv.org/pdf/2505.14140v1)

Authors: Qianyue Hao, Sibo Li, Jian Yuan, Yong Li

Despite rapid advancements in large language models (LLMs), the token-level
autoregressive nature constrains their complex reasoning capabilities. To
enhance LLM reasoning, inference-time techniques, including
Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they
are fairly cost-effective by guiding reasoning through sophisticated logical
structures without modifying LLMs' parameters. However, these manually
predefined, task-agnostic frameworks are applied uniformly across diverse
tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT),
where we train a lightweight navigator model with reinforcement learning (RL)
to adaptively enhance LLM reasoning at inference time. Specifically, we design
five basic logic blocks from the perspective of human cognition. During the
reasoning process, the trained RL navigator dynamically selects the suitable
logic blocks and combines them into task-specific logical structures according
to problem characteristics. Experiments across multiple reasoning benchmarks
(AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek)
illustrate that RLoT outperforms established inference-time techniques by up to
13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to
make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL
navigator demonstrates strong transferability: a model trained on one specific
LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is
open-source at https://anonymous.4open.science/r/RL-LLM-Reasoning-1A30 for
reproducibility.

### 9. [Building a Stable Planner: An Extended Finite State Machine Based Planning Module for Mobile GUI Agent](http://arxiv.org/pdf/2505.14141v1)

Authors: Fanglin Mo, Junzhe Chen, Haoxuan Zhu, Xuming Hu

Mobile GUI agents execute user commands by directly interacting with the
graphical user interface (GUI) of mobile devices, demonstrating significant
potential to enhance user convenience. However, these agents face considerable
challenges in task planning, as they must continuously analyze the GUI and
generate operation instructions step by step. This process often leads to
difficulties in making accurate task plans, as GUI agents lack a deep
understanding of how to effectively use the target applications, which can
cause them to become "lost" during task execution. To address the task planning
issue, we propose SPlanner, a plug-and-play planning module to generate
execution plans that guide vision language model(VLMs) in executing tasks. The
proposed planning module utilizes extended finite state machines (EFSMs) to
model the control logits and configurations of mobile applications. It then
decomposes a user instruction into a sequence of primary function modeled in
EFSMs, and generate the execution path by traversing the EFSMs. We further
refine the execution path into a natural language plan using an LLM. The final
plan is concise and actionable, and effectively guides VLMs to generate
interactive GUI actions to accomplish user tasks. SPlanner demonstrates strong
performance on dynamic benchmarks reflecting real-world mobile usage. On the
AndroidWorld benchmark, SPlanner achieves a 63.8% task success rate when paired
with Qwen2.5-VL-72B as the VLM executor, yielding a 28.8 percentage point
improvement compared to using Qwen2.5-VL-72B without planning assistance.

### 10. [Multimodal Mixture of Low-Rank Experts for Sentiment Analysis and Emotion Recognition](http://arxiv.org/pdf/2505.14143v1)

Authors: Shuo Zhang, Jinsong Zhang, Zhejun Zhang, Lei Li

Multi-task learning (MTL) enables the efficient transfer of extra knowledge
acquired from other tasks. The high correlation between multimodal sentiment
analysis (MSA) and multimodal emotion recognition (MER) supports their joint
training. However, existing methods primarily employ hard parameter sharing,
ignoring parameter conflicts caused by complex task correlations. In this
paper, we present a novel MTL method for MSA and MER, termed Multimodal Mixture
of Low-Rank Experts (MMoLRE). MMoLRE utilizes shared and task-specific experts
to distinctly model common and unique task characteristics, thereby avoiding
parameter conflicts. Additionally, inspired by low-rank structures in the
Mixture of Experts (MoE) framework, we design low-rank expert networks to
reduce parameter and computational overhead as the number of experts increases.
Extensive experiments on the CMU-MOSI and CMU-MOSEI benchmarks demonstrate that
MMoLRE achieves state-of-the-art performance on the MSA task and competitive
results on the MER task.

### Hardware Architecture

### 1. [CRYPTONITE: Scalable Accelerator Design for Cryptographic Primitives and Algorithms](http://arxiv.org/pdf/2505.14657v1)

Authors: Karthikeya Sharma Maheswaran, Camille Bossut, Andy Wanna, Qirun Zhang, Cong Hao

Cryptographic primitives, consisting of repetitive operations with different
inputs, are typically implemented using straight-line C code due to traditional
execution on CPUs. Computing these primitives is necessary for secure
communication; thus, dedicated hardware accelerators are required in resource
and latency-constrained environments. High-Level Synthesis (HLS) generates
hardware from high-level implementations in languages like C, enabling the
rapid prototyping and evaluation of designs, leading to its prominent use in
developing dedicated hardware accelerators. However, directly synthesizing the
straight-line C implementations of cryptographic primitives can lead to large
hardware designs with excessive resource usage or suboptimal performance.
  We introduce Cryptonite, a tool that automatically generates efficient,
synthesizable, and correct-by-design hardware accelerators for cryptographic
primitives directly from straight-line C code. Cryptonite first identifies
high-level hardware constructs through verified rewriting, emphasizing resource
reuse. The second stage automatically explores latency-oriented implementations
of the compact design. This enables the flexible scaling of a particular
accelerator to meet the hardware requirements. We demonstrate Cryptonite's
effectiveness using implementations from the Fiat Cryptography project, a
library of verified and auto-generated cryptographic primitives for
elliptic-curve cryptography. Our results show that Cryptonite achieves scalable
designs with up to 88.88\% reduced resource usage and a 54.31\% improvement in
latency compared to naively synthesized designs.

### 2. [Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators](http://arxiv.org/pdf/2505.14314v1)

Authors: Kosmas Alexandridis, Vasileios Titopoulos, Giorgos Dimitrakopoulos

Attention mechanisms, particularly within Transformer architectures and large
language models (LLMs), have revolutionized sequence modeling in machine
learning and artificial intelligence applications. To compute attention for
increasingly long sequences, specialized accelerators have been proposed to
execute key attention steps directly in hardware. Among the various recently
proposed architectures, those based on variants of the FlashAttention
algorithm, originally designed for GPUs, stand out due to their optimized
computation, tiling capabilities, and reduced memory traffic. In this work, we
focus on optimizing the kernel of floating-point-based FlashAttention using new
hardware operators that fuse the computation of exponentials and vector
multiplications, e.g., e^x, V. The proposed ExpMul hardware operators
significantly reduce the area and power costs of FlashAttention-based hardware
accelerators. When implemented in a 28nm ASIC technology, they achieve
improvements of 28.8% in area and 17.6% in power, on average, compared to
state-of-the-art hardware architectures with separate exponentials and vector
multiplications hardware operators.

### 3. [FLASH-D: FlashAttention with Hidden Softmax Division](http://arxiv.org/pdf/2505.14201v1)

Authors: Kosmas Alexandridis, Vasileios Titopoulos, Giorgos Dimitrakopoulos

The transformer's attention mechanism has revolutionized AI and machine
learning, with its efficient computation being crucial to its performance.
However, calculating attention involves matrix operations interspersed with
softmax rescaling, which inherently slows down computation and requires
processing the entire input sequence. Building on online softmax computation,
FlashAttention integrates softmax calculation with matrix arithmetic, enabling
tiled computation independent of sequence length. While optimized for GPUs,
FlashAttention's simplicity makes it amenable to direct hardware acceleration.
This work re-evaluates the core FlashAttention kernel, presenting FLASH-D a
mathematically equivalent, yet simplified, formulation that achieves: (a)
hiding softmax division within other non-linear function evaluations; (b)
inherently numerically stable computation of exponentials, eliminating the need
for maximum value subtraction; and (c) a reduction in computational cost
without introducing numerical approximations to the FlashAttention kernel.
Importantly, the essential FlashAttention properties that facilitate efficient
tiled implementation are fully preserved. Hardware implementation results at
28nm demonstrate that this proposed formulation achieves a 22.8% reduction in
area and a 20.3% reduction in power, on average, compared to state-of-the-art
parallel hardware architectures without any performance penalty.

### 4. [Distributed quantum computing with black-box subroutines](http://arxiv.org/pdf/2505.14519v1)

Authors: X. Xu, Y. -D. Liu, S. Shi, Y. -J. Wang, D. -S. Wang

In this work, we propose a general protocol for distributed quantum computing
that accommodates arbitrary unknown subroutines. It can be applied to scale up
quantum computing through multi-chip interconnection, as well as to tasks such
as estimating unknown parameters or processes for circuit depth reduction and
constructing secure quantum cryptographic protocols. Our protocol builds upon a
few techniques we develop, such as the oblivious quantum teleportation and
control, which can circumvent quantum no-go theorems on the manipulation of
unknown objects. Furthermore, we demonstrate that this protocol can be
physically implemented using currently available quantum computing platforms.
These results suggest that our framework could provide a foundation for
developing more advanced quantum algorithms and protocols in the future.

### Computational Complexity

### 1. [Linear Hashing Is Optimal](http://arxiv.org/pdf/2505.14061v1)

Authors: Michael Jaber, Vinayak M. Kumar, David Zuckerman

We prove that hashing $n$ balls into $n$ bins via a random matrix over
$\mathbf{F}_2$ yields expected maximum load $O(\log n / \log \log n)$. This
matches the expected maximum load of a fully random function and resolves an
open question posed by Alon, Dietzfelbinger, Miltersen, Petrank, and Tardos
(STOC '97, JACM '99). More generally, we show that the maximum load exceeds
$r\cdot\log n/\log\log n$ with probability at most $O(1/r^2)$.

### Computational Engineering

### 1. [Predicting Dynamical Systems across Environments via Diffusive Model Weight Generation](http://arxiv.org/pdf/2505.13919v1)

Authors: Ruikun Li, Huandong Wang, Jingtao Ding, Yuan Yuan, Qingmin Liao, Yong Li

Data-driven methods offer an effective equation-free solution for predicting
physical dynamics. However, the same physical system can exhibit significantly
different dynamic behaviors in various environments. This causes prediction
functions trained for specific environments to fail when transferred to unseen
environments. Therefore, cross-environment prediction requires modeling the
dynamic functions of different environments. In this work, we propose a model
weight generation method, \texttt{EnvAd-Diff}. \texttt{EnvAd-Diff} operates in
the weight space of the dynamic function, generating suitable weights from
scratch based on environmental condition for zero-shot prediction.
Specifically, we first train expert prediction functions on dynamic
trajectories from a limited set of visible environments to create a model zoo,
thereby constructing sample pairs of prediction function weights and their
corresponding environments. Subsequently, we train a latent space diffusion
model conditioned on the environment to model the joint distribution of weights
and environments. Considering the lack of environmental prior knowledge in
real-world scenarios, we propose a physics-informed surrogate label to
distinguish different environments. Generalization experiments across multiple
systems demonstrate that a 1M parameter prediction function generated by
\texttt{EnvAd-Diff} outperforms a pre-trained 500M parameter foundation model.

### 2. [PLUTUS Open Source -- Breaking Barriers in Algorithmic Trading](http://arxiv.org/pdf/2505.14050v1)

Authors: An-Dan Nguyen, Quang-Khoi Ta, Duy-Anh Vo

Algorithmic trading has long been an opaque, fragmented domain, guarded by
secrecy and built around proprietary systems. In contrast to the open,
collaborative evolution in fields like machine learning or software
engineering, the algorithmic trading ecosystem has been slow to adopt
reproducibility, standardization, and shared infrastructure. This paper
introduces PLUTUS Open Source, an initiative sponsored by ALGOTRADE to reshape
this landscape through openness, structure, and collaboration. PLUTUS combines
a reproducibility standard, a modular development framework, and a growing
suite of community-built reference strategies. The project provides a
systematic approach to designing, testing, and documenting trading algorithms,
regardless of the user's technical or financial background. We outline the
motivation behind the initiative, present its foundational structure, and
showcase working examples that adhere to the PLUTUS standard. We also invite
the broader research and trading communities to contribute, iterate, and help
build a transparent and inclusive future for algorithmic trading.

### 3. [Higher-order, mixed-hybrid finite elements for Kirchhoff-Love shells](http://arxiv.org/pdf/2505.14115v1)

Authors: Jonas Neumeyer, Michael Wolfgang Kaiser, Thomas-Peter Fries

A novel mixed-hybrid method for Kirchhoff-Love shells is proposed that
enables the use of classical, possibly higher-order Lagrange elements in
numerical analyses. In contrast to purely displacement-based formulations that
require higher continuity of shape functions as in IGA, the mixed formulation
features displacements and moments as primary unknowns. Thereby the continuity
requirements are reduced, allowing equal-order interpolations of the
displacements and moments. Hybridization enables an element-wise static
condensation of the degrees of freedom related to the moments, at the price of
introducing (significantly less) rotational degrees of freedom acting as
Lagrange multipliers to weakly enforce the continuity of tangential moments
along element edges. The mixed model is formulated coordinate-free based on the
Tangential Differential Calculus, making it applicable for explicitly and
implicitly defined shell geometries. All mechanically relevant boundary
conditions are considered. Numerical results confirm optimal higher-order
convergence rates whenever the mechanical setup allows for sufficiently smooth
solutions; new benchmark test cases of this type are proposed.

### 4. [Improved Methods for Model Pruning and Knowledge Distillation](http://arxiv.org/pdf/2505.14052v1)

Authors: Wei Jiang, Anying Fu, Youling Zhang

Model pruning is a performance optimization technique for large language
models like R1 or o3-mini. However, existing pruning methods often lead to
significant performance degradation or require extensive retraining and
fine-tuning. This technique aims to identify and remove neurons, connections
unlikely leading to the contribution during the human-computer interaction
phase. Our goal is to obtain a much smaller and faster knowledge distilled
model that can quickly generate content almost as good as those of the unpruned
ones. We propose MAMA Pruning, short for Movement and Magnitude Analysis, an
improved pruning method that effectively reduces model size and computational
complexity while maintaining performance comparable to the original unpruned
model even at extreme pruned levels. The improved method is based on weights,
bias fixed in the pre-training phase and GRPO rewards verified during the
post-training phase as our novel pruning indicators. Preliminary experimental
results show that our method outperforms and be comparable to state-of-the-art
methods across various pruning levels and different downstream computational
linguistics tasks.

### 5. [Global Maxwell Tomography Using the Volume-Surface Integral Equation for Improved Estimation of Electrical Properties](http://arxiv.org/pdf/2505.14546v1)

Authors: Ilias Giannakopoulos, José E. Cruz Serrallés, Jan Paška, Martijn A. Cloos, Ryan Brown, Riccardo Lattanzi

Objective: Global Maxwell Tomography (GMT) is a noninvasive inverse
optimization method for the estimation of electrical properties (EP) from
magnetic resonance (MR) measurements. GMT uses the volume integral equation
(VIE) in the forward problem and assumes that the sample has negligible effect
on the coil currents. Consequently, GMT calculates the coil's incident fields
with an initial EP distribution and keeps them constant for all optimization
iterations. This can lead to erroneous reconstructions. This work introduces a
novel version of GMT that replaces VIE with the volume-surface integral
equation (VSIE), which recalculates the coil currents at every iteration based
on updated EP estimates before computing the associated fields. Methods: We
simulated an 8-channel transceiver coil array for 7 T brain imaging and
reconstructed the EP of a realistic head model using VSIE-based GMT. We built
the coil, collected experimental MR measurements, and reconstructed EP of a
two-compartment phantom. Results: In simulations, VSIE-based GMT outperformed
VIE-based GMT by at least 12% for both EP. In experiments, the relative
difference with respect to probe-measured EP values in the inner (outer)
compartment was 13% (26%) and 17% (33%) for the permittivity and conductivity,
respectively. Conclusion: The use of VSIE over VIE enhances GMT's performance
by accounting for the effect of the EP on the coil currents. Significance:
VSIE-based GMT does not rely on an initial EP estimate, rendering it more
suitable for experimental reconstructions compared to the VIE-based GMT.

### 6. [Design and Evaluation of a Microservices Cloud Framework for Online Travel Platforms](http://arxiv.org/pdf/2505.14508v1)

Authors: Biman Barua, M. Shamim Kaiser

Handling online travel agents globally requires efficient and flexible
software solution architectures. When it needs to handle thousands of agents
and billions of clients data globally. Microservices architecture is used to
break down a large program into numerous, smaller services which can run
individually and perform individual tasks. This paper analyses and integrates a
unique Microservices Cloud Framework designed to support Online Travel
Platforms (MCF-OTP). MCF-OTPs main goal is to increase the performance,
flexibility, and maintenance of online travel platforms via cloud computing and
microservice technologies. Large-scale travel apps, including managing numerous
data sources, dealing with traffic peaks, and providing fault tolerance, can be
addressed by the suggested framework. The framework increases good
interpretation between flawless data synchronization, microservices, and
dynamic scaling based on demand technology. An organization framework that
optimizes service borders and minimizes inter-service dependencies is
recommended. Thus, this can result in elevated development adaptability. In
this research, the principal goal is to evaluate MCF-OTPs efficiency using the
indicators of fault tolerance and response time. It is indicated by the
findings that the MCF-OTP structure excels traditional monolithic designs in
terms of dependability and scalability, managing traffic spikes seamlessly and
decreasing downtime. The cost-effective analysis helps ascertain the net gain
attained by the startup fees and the ongoing operational costs. The cloud-based
environment is used to reduce the fracture cost which also helps to increase
the efficiency of resource allocation, according to the research.

### Computational Geometry

### 1. [Bounding the density of binary sphere packing](http://arxiv.org/pdf/2505.14110v1)

Authors: Thomas Fernique, Daria Pchelina

This paper provides the currently best known upper bound on the density of a
packing in three-dimensional Euclidean space of two types of spheres whose size
ratio is the largest one that allows the insertion of a small sphere in each
octahedral hole of a hexagonal compact packing of large spheres. This upper
bound is obtained by bounding from above the density of the tetrahedra which
can appear in the additively-weighted Delaunay decomposition of the sphere
centers of such packings. The proof relies on challenging computer calculations
in interval arithmetic and may be of interest by their own.

### 2. [Towards Non-Euclidean Foundation Models: Advancing AI Beyond Euclidean Frameworks](http://arxiv.org/pdf/2505.14417v1)

Authors: Menglin Yang, Yifei Zhang, Jialin Chen, Melanie Weber, Rex Ying

In the era of foundation models and Large Language Models (LLMs), Euclidean
space is the de facto geometric setting of our machine learning architectures.
However, recent literature has demonstrated that this choice comes with
fundamental limitations. To that end, non-Euclidean learning is quickly gaining
traction, particularly in web-related applications where complex relationships
and structures are prevalent. Non-Euclidean spaces, such as hyperbolic,
spherical, and mixed-curvature spaces, have been shown to provide more
efficient and effective representations for data with intrinsic geometric
properties, including web-related data like social network topology,
query-document relationships, and user-item interactions. Integrating
foundation models with non-Euclidean geometries has great potential to enhance
their ability to capture and model the underlying structures, leading to better
performance in search, recommendations, and content understanding. This
workshop focuses on the intersection of Non-Euclidean Foundation Models and
Geometric Learning (NEGEL), exploring its potential benefits, including the
potential benefits for advancing web-related technologies, challenges, and
future directions. Workshop page:
[https://hyperboliclearning.github.io/events/www2025workshop](https://hyperboliclearning.github.io/events/www2025workshop)

### 3. [An asymptotic rigidity property from the realizability of chirotope extensions](http://arxiv.org/pdf/2505.14189v1)

Authors: Xavier Goaoc, Arnau Padrol

Let $P$ be a finite full-dimensional point configuration in $\mathbb{R}^d$.
We show that if a point configuration $Q$ has the property that all finite
chirotopes realizable by adding (generic) points to $P$ are also realizable by
adding points to $Q$, then $P$ and $Q$ are equal up to a direct affine
transform. We also show that for any point configuration $P$ and any
$\varepsilon>0$, there is a finite, (generic) extension $\widehat P$ of $P$
with the following property: if another realization $Q$ of the chirotope of $P$
can be extended so as to realize the chirotope of $\widehat P$, then there
exists a direct affine transform that maps each point of $Q$ within distance
$\varepsilon$ of the corresponding point of $P$.

### Computation and Language

### 1. [Improve Language Model and Brain Alignment via Associative Memory](http://arxiv.org/pdf/2505.13844v1)

Authors: Congchi Yin, Yongpeng Zhang, Xuyun Wen, Piji Li

Associative memory engages in the integration of relevant information for
comprehension in the human cognition system. In this work, we seek to improve
alignment between language models and human brain while processing speech
information by integrating associative memory. After verifying the alignment
between language model and brain by mapping language model activations to brain
activity, the original text stimuli expanded with simulated associative memory
are regarded as input to computational language models. We find the alignment
between language model and brain is improved in brain regions closely related
to associative memory processing. We also demonstrate large language models
after specific supervised fine-tuning better align with brain response, by
building the \textit{Association} dataset containing 1000 samples of stories,
with instructions encouraging associative memory as input and associated
content as output.

### 2. [Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning](http://arxiv.org/pdf/2505.13866v1)

Authors: Jiwon Song, Dongwon Jo, Yulhwa Kim, Jae-Joon Kim

Recent reasoning-focused language models achieve high accuracy by generating
lengthy intermediate reasoning paths before producing final answers. While this
approach is effective in solving problems that require logical thinking, long
reasoning paths significantly increase memory usage and throughput of token
generation, limiting the practical deployment of such models. We propose
Reasoning Path Compression (RPC), a training-free method that accelerates
inference by leveraging the semantic sparsity of reasoning paths. RPC
periodically compresses the KV cache by retaining KV cache that receive high
importance score, which are computed using a selector window composed of
recently generated queries. Experiments show that RPC improves generation
throughput of QwQ-32B by up to 1.60$\times$ compared to the inference with full
KV cache, with an accuracy drop of 1.2% on the AIME 2024 benchmark. Our
findings demonstrate that semantic sparsity in reasoning traces can be
effectively exploited for compression, offering a practical path toward
efficient deployment of reasoning LLMs. Our code is available at
https://github.com/jiwonsong-dev/ReasoningPathCompression.

### 3. [Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning](http://arxiv.org/pdf/2505.13886v1)

Authors: Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, Chaoran Tao, Zhiyuan Guo, Jizhou Yu, Tianhao Cheng, Changhao Jiang, Zhen Wang, Tao Liang, Zhihui Fei, Mingyang Wan, Guojun Ma, Weifeng Ge, Guanhua Chen, Tao Gui, Xipeng Qiu, Qi Zhang, Xuanjing Huang

Visual-language Chain-of-Thought (CoT) data resources are relatively scarce
compared to text-only counterparts, limiting the improvement of reasoning
capabilities in Vision Language Models (VLMs). However, high-quality
vision-language reasoning data is expensive and labor-intensive to annotate. To
address this issue, we leverage a promising resource: game code, which
naturally contains logical structures and state transition processes.
Therefore, we propose Code2Logic, a novel game-code-driven approach for
multimodal reasoning data synthesis. Our approach leverages Large Language
Models (LLMs) to adapt game code, enabling automatic acquisition of reasoning
processes and results through code execution. Using the Code2Logic approach, we
developed the GameQA dataset to train and evaluate VLMs. GameQA is
cost-effective and scalable to produce, challenging for state-of-the-art
models, and diverse with 30 games and 158 tasks. Surprisingly, despite training
solely on game data, VLMs demonstrated out of domain generalization,
specifically Qwen2.5-VL-7B improving performance by 2.33\% across 7 diverse
vision-language benchmarks. Our code and dataset are available at
https://github.com/tongjingqi/Code2Logic.

### 4. [Mapping the Minds of LLMs: A Graph-Based Analysis of Reasoning LLM](http://arxiv.org/pdf/2505.13890v1)

Authors: Zhen Xiong, Yujun Cai, Zhecheng Li, Yiwei Wang

Recent advances in test-time scaling have enabled Large Language Models
(LLMs) to display sophisticated reasoning abilities via extended
Chain-of-Thought (CoT) generation. Despite their potential, these Reasoning
LLMs (RLMs) often demonstrate counterintuitive and unstable behaviors, such as
performance degradation under few-shot prompting, that challenge our current
understanding of RLMs. In this work, we introduce a unified graph-based
analytical framework for better modeling the reasoning processes of RLMs. Our
method first clusters long, verbose CoT outputs into semantically coherent
reasoning steps, then constructs directed reasoning graphs to capture
contextual and logical dependencies among these steps. Through comprehensive
analysis across models and prompting regimes, we reveal that structural
properties, such as exploration density, branching, and convergence ratios,
strongly correlate with reasoning accuracy. Our findings demonstrate how
prompting strategies substantially reshape the internal reasoning structure of
RLMs, directly affecting task outcomes. The proposed framework not only enables
quantitative evaluation of reasoning quality beyond conventional metrics but
also provides practical insights for prompt engineering and the cognitive
analysis of LLMs. Code and resources will be released to facilitate future
research in this direction.

### 5. [InfiGFusion: Graph-on-Logits Distillation via Efficient Gromov-Wasserstein for Model Fusion](http://arxiv.org/pdf/2505.13893v1)

Authors: Yuanyi Wang, Zhaoyi Yan, Yiming Zhang, Qi Zhou, Yanggan Gu, Fei Wu, Hongxia Yang

Recent advances in large language models (LLMs) have intensified efforts to
fuse heterogeneous open-source models into a unified system that inherits their
complementary strengths. Existing logit-based fusion methods maintain inference
efficiency but treat vocabulary dimensions independently, overlooking semantic
dependencies encoded by cross-dimension interactions. These dependencies
reflect how token types interact under a model's internal reasoning and are
essential for aligning models with diverse generation behaviors. To explicitly
model these dependencies, we propose \textbf{InfiGFusion}, the first
structure-aware fusion framework with a novel \textit{Graph-on-Logits
Distillation} (GLD) loss. Specifically, we retain the top-$k$ logits per output
and aggregate their outer products across sequence positions to form a global
co-activation graph, where nodes represent vocabulary channels and edges
quantify their joint activations. To ensure scalability and efficiency, we
design a sorting-based closed-form approximation that reduces the original
$O(n^4)$ cost of Gromov-Wasserstein distance to $O(n \log n)$, with provable
approximation guarantees. Experiments across multiple fusion settings show that
GLD consistently improves fusion quality and stability. InfiGFusion outperforms
SOTA models and fusion baselines across 11 benchmarks spanning reasoning,
coding, and mathematics. It shows particular strength in complex reasoning
tasks, with +35.6 improvement on Multistep Arithmetic and +37.06 on Causal
Judgement over SFT, demonstrating superior multi-step and relational inference.

### 6. [Let's Verify Math Questions Step by Step](http://arxiv.org/pdf/2505.13903v1)

Authors: Chengyu Shen, Zhen Hao Wong, Runming He, Hao Liang, Meiyi Qiang, Zimo Meng, Zhengyang Zhao, Bohan Zeng, Zhengzhou Zhu, Bin Cui, Wentao Zhang

Large Language Models (LLMs) have recently achieved remarkable progress in
mathematical reasoning. To enable such capabilities, many existing works
distill strong reasoning models into long chains of thought or design
algorithms to construct high-quality math QA data for training. However, these
efforts primarily focus on generating correct reasoning paths and answers,
while largely overlooking the validity of the questions themselves. In this
work, we propose Math Question Verification (MathQ-Verify), a novel five-stage
pipeline designed to rigorously filter ill-posed or under-specified math
problems. MathQ-Verify first performs format-level validation to remove
redundant instructions and ensure that each question is syntactically
well-formed. It then formalizes each question, decomposes it into atomic
conditions, and verifies them against mathematical definitions. Next, it
detects logical contradictions among these conditions, followed by a
goal-oriented completeness check to ensure the question provides sufficient
information for solving. To evaluate this task, we use existing benchmarks
along with an additional dataset we construct, containing 2,147 math questions
with diverse error types, each manually double-validated. Experiments show that
MathQ-Verify achieves state-of-the-art performance across multiple benchmarks,
improving the F1 score by up to 25 percentage points over the direct
verification baseline. It further attains approximately 90% precision and 63%
recall through a lightweight model voting scheme. MathQ-Verify offers a
scalable and accurate solution for curating reliable mathematical datasets,
reducing label noise and avoiding unnecessary computation on invalid questions.
Our code and data are available at https://github.com/scuuy/MathQ-Verify.

### 7. [Cross-Linguistic Transfer in Multilingual NLP: The Role of Language Families and Morphology](http://arxiv.org/pdf/2505.13908v1)

Authors: Ajitesh Bankula, Praney Bankula

Cross-lingual transfer has become a crucial aspect of multilingual NLP, as it
allows for models trained on resource-rich languages to be applied to
low-resource languages more effectively. Recently massively multilingual
pre-trained language models (e.g., mBERT, XLM-R) demonstrate strong zero-shot
transfer capabilities[14] [13]. This paper investigates cross-linguistic
transfer through the lens of language families and morphology. Investigating
how language family proximity and morphological similarity affect performance
across NLP tasks. We further discuss our results and how it relates to findings
from recent literature. Overall, we compare multilingual model performance and
review how linguistic distance metrics correlate with transfer outcomes. We
also look into emerging approaches that integrate typological and morphological
information into model pre-training to improve transfer to diverse
languages[18] [19].

### 8. [Word length predicts word order: "Min-max"-ing drives language evolution](http://arxiv.org/pdf/2505.13913v1)

Authors: Hiram Ring

Current theories of language propose an innate (Baker 2001; Chomsky 1981) or
a functional (Greenberg 1963; Dryer 2007; Hawkins 2014) origin for the surface
structures (i.e. word order) that we observe in languages of the world, while
evolutionary modeling (Dunn et al. 2011) suggests that descent is the primary
factor influencing such patterns. Although there are hypotheses for word order
change from both innate and usage-based perspectives for specific languages and
families, there are key disagreements between the two major proposals for
mechanisms that drive the evolution of language more broadly (Wasow 2002; Levy
2008). This paper proposes a universal underlying mechanism for word order
change based on a large tagged parallel dataset of over 1,500 languages
representing 133 language families and 111 isolates. Results indicate that word
class length is significantly correlated with word order crosslinguistically,
but not in a straightforward manner, partially supporting opposing theories of
processing, while at the same time predicting historical word order change in
two different phylogenetic lines and explaining more variance than descent or
language area in regression models. Such findings suggest an integrated
"Min-Max" theory of language evolution driven by competing pressures of
processing and information structure, aligning with recent efficiency-oriented
(Levshina 2023) and information-theoretic proposals (Zaslavsky 2020; Tucker et
al. 2025).

### 9. [Towards Rehearsal-Free Continual Relation Extraction: Capturing Within-Task Variance with Adaptive Prompting](http://arxiv.org/pdf/2505.13944v1)

Authors: Bao-Ngoc Dao, Quang Nguyen, Luyen Ngo Dinh, Minh Le, Nam Le, Linh Ngo Van

Memory-based approaches have shown strong performance in Continual Relation
Extraction (CRE). However, storing examples from previous tasks increases
memory usage and raises privacy concerns. Recently, prompt-based methods have
emerged as a promising alternative, as they do not rely on storing past
samples. Despite this progress, current prompt-based techniques face several
core challenges in CRE, particularly in accurately identifying task identities
and mitigating catastrophic forgetting. Existing prompt selection strategies
often suffer from inaccuracies, lack robust mechanisms to prevent forgetting in
shared parameters, and struggle to handle both cross-task and within-task
variations. In this paper, we propose WAVE++, a novel approach inspired by the
connection between prefix-tuning and mixture of experts. Specifically, we
introduce task-specific prompt pools that enhance flexibility and adaptability
across diverse tasks while avoiding boundary-spanning risks; this design more
effectively captures variations within each task and across tasks. To further
refine relation classification, we incorporate label descriptions that provide
richer, more global context, enabling the model to better distinguish among
different relations. We also propose a training-free mechanism to improve task
prediction during inference. Moreover, we integrate a generative model to
consolidate prior knowledge within the shared parameters, thereby removing the
need for explicit data storage. Extensive experiments demonstrate that WAVE++
outperforms state-of-the-art prompt-based and rehearsal-based methods, offering
a more robust solution for continual relation extraction. Our code is publicly
available at https://github.com/PiDinosauR2804/WAVE-CRE-PLUS-PLUS.

### 10. [Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals](http://arxiv.org/pdf/2505.13972v1)

Authors: Qianli Wang, Van Bach Nguyen, Nils Feldhus, Luis Felipe Villa-Arenas, Christin Seifert, Sebastian Möller, Vera Schmitt

Counterfactual examples are widely employed to enhance the performance and
robustness of large language models (LLMs) through counterfactual data
augmentation (CDA). However, the selection of the judge model used to evaluate
label flipping, the primary metric for assessing the validity of generated
counterfactuals for CDA, yields inconsistent results. To decipher this, we
define four types of relationships between the counterfactual generator and
judge models. Through extensive experiments involving two state-of-the-art
LLM-based methods, three datasets, five generator models, and 15 judge models,
complemented by a user study (n = 90), we demonstrate that judge models with an
independent, non-fine-tuned relationship to the generator model provide the
most reliable label flipping evaluations. Relationships between the generator
and judge models, which are closely aligned with the user study for CDA, result
in better model performance and robustness. Nevertheless, we find that the gap
between the most effective judge models and the results obtained from the user
study remains considerably large. This suggests that a fully automated pipeline
for CDA may be inadequate and requires human intervention.

### Cryptography and Security

### 1. [Provable Execution in Real-Time Embedded Systems](http://arxiv.org/pdf/2505.13842v1)

Authors: Antonio Joia Neto, Norrathep Rattanavipanon, Ivan De Oliveira Nunes

Embedded devices are increasingly ubiquitous and vital, often supporting
safety-critical functions. However, due to strict cost and energy constraints,
they are typically implemented with Micro-Controller Units (MCUs) that lack
advanced architectural security features. Within this space, recent efforts
have created low-cost architectures capable of generating Proofs of Execution
(PoX) of software on potentially compromised MCUs. This capability can ensure
the integrity of sensor data from the outset, by binding sensed results to an
unforgeable cryptographic proof of execution on edge sensor MCUs. However, the
security of existing PoX requires the proven execution to occur atomically.
This requirement precludes the application of PoX to (1) time-shared systems,
and (2) applications with real-time constraints, creating a direct conflict
between execution integrity and the real-time availability needs of several
embedded system uses.
  In this paper, we formulate a new security goal called Real-Time Proof of
Execution (RT-PoX) that retains the integrity guarantees of classic PoX while
enabling its application to existing real-time systems. This is achieved by
relaxing the atomicity requirement of PoX while dispatching interference
attempts from other potentially malicious tasks (or compromised operating
systems) executing on the same device. To realize the RT-PoX goal, we develop
Provable Execution Architecture for Real-Time Systems (PEARTS). To the best of
our knowledge, PEARTS is the first PoX system that can be directly deployed
alongside a commodity embedded real-time operating system (FreeRTOS). This
enables both real-time scheduling and execution integrity guarantees on
commodity MCUs. To showcase this capability, we develop a PEARTS open-source
prototype atop FreeRTOS on a single-core ARM Cortex-M33 processor. We evaluate
and report on PEARTS security and (modest) overheads.

### 2. [hChain 4.0: A Secure and Scalable Permissioned Blockchain for EHR Management in Smart Healthcare](http://arxiv.org/pdf/2505.13861v1)

Authors: Musharraf N. Alruwaill, Saraju P. Mohanty, Elias Kougianos

The growing utilization of Internet of Medical Things (IoMT) devices,
including smartwatches and wearable medical devices, has facilitated real-time
health monitoring and data analysis to enhance healthcare outcomes. These
gadgets necessitate improved security measures to safeguard sensitive health
data while tackling scalability issues in real-time settings. The proposed
system, hChain 4.0, employs a permissioned blockchain to provide a secure and
scalable data infrastructure designed to fulfill these needs. This stands in
contrast to conventional systems, which are vulnerable to security flaws or
rely on public blockchains, constrained by scalability and expense. The
proposed approach introduces a high-privacy method in which health data are
encrypted using the Advanced Encryption Standard (AES) for time-efficient
encryption, combined with Partial Homomorphic Encryption (PHE) to enable secure
computations on encrypted data, thereby enhancing privacy. Moreover, it
utilizes private channels that enable isolated communication and ledger between
stakeholders, ensuring robust privacy while supporting collaborative
operations. The proposed framework enables anonymized health data sharing for
medical research by pseudonymizing patient identity. Additionally, hChain 4.0
incorporates Attribute-Based Access Control (ABAC) to provide secure electronic
health record (EHR) sharing among authorized parties, where ABAC ensures
fine-grained permission management vital for multi-organizational healthcare
settings. Experimental assessments indicate that the proposed approach achieves
higher scalability, cost-effectiveness, and validated security.

### 3. [The Hidden Dangers of Outdated Software: A Cyber Security Perspective](http://arxiv.org/pdf/2505.13922v1)

Authors: Gogulakrishnan Thiyagarajan, Vinay Bist, Prabhudarshi Nayak

Outdated software remains a potent and underappreciated menace in 2025's
cybersecurity environment, exposing systems to a broad array of threats,
including ransomware, data breaches, and operational outages that can have
devastating and far-reaching impacts. This essay explores the unseen threats of
cyberattacks by presenting robust statistical information, including the
staggering reality that 32% of cyberattacks exploit unpatched software
vulnerabilities, based on a 2025 TechTarget survey. Furthermore, it discusses
real case studies, including the MOVEit breach in 2023 and the Log4Shell breach
in 2021, both of which illustrate the catastrophic consequences of failing to
perform software updates. The article offers a detailed analysis of the nature
of software vulnerabilities, the underlying reasons for user resistance to
patches, and organizational barriers that compound the issue. Furthermore, it
suggests actionable solutions, including automation and awareness campaigns, to
address these shortcomings. Apart from this, the paper also talks of trends
such as AI-driven vulnerability patching and legal consequences of
non-compliance under laws like HIPAA, thus providing a futuristic outlook on
how such advancements may define future defenses. Supplemented by tables like
one detailing trends in vulnerability and a graph illustrating technology
adoption, this report showcases the pressing demand for anticipatory update
strategies to safeguard digital ecosystems against the constantly evolving
threats that characterize the modern cyber landscape. As it stands, it is a
very useful document for practitioners, policymakers, and researchers.

### 4. [D4+: Emergent Adversarial Driving Maneuvers with Approximate Functional Optimization](http://arxiv.org/pdf/2505.13942v1)

Authors: Diego Ortiz Barbosa, Luis Burbano, Carlos Hernandez, Zengxiang Lei, Younghee Park, Satish Ukkusuri, Alvaro A Cardenas

Intelligent mechanisms implemented in autonomous vehicles, such as proactive
driving assist and collision alerts, reduce traffic accidents. However,
verifying their correct functionality is difficult due to complex interactions
with the environment. This problem is exacerbated in adversarial environments,
where an attacker can control the environment surrounding autonomous vehicles
to exploit vulnerabilities.
  To preemptively identify vulnerabilities in these systems, in this paper, we
implement a scenario-based framework with a formal method to identify the
impact of malicious drivers interacting with autonomous vehicles. The
formalization of the evaluation requirements utilizes metric temporal logic
(MTL) to identify a safety condition that we want to test. Our goal is to find,
through a rigorous testing approach, any trace that violates this MTL safety
specification. Our results can help designers identify the range of safe
operational behaviors that prevent malicious drivers from exploiting the
autonomous features of modern vehicles.

### 5. [Zk-SNARK for String Match](http://arxiv.org/pdf/2505.13964v1)

Authors: Taoran Li, Taobo Liao

We present a secure and efficient string-matching platform leveraging
zk-SNARKs (Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge) to
address the challenge of detecting sensitive information leakage while
preserving data privacy. Our solution enables organizations to verify whether
private strings appear on public platforms without disclosing the strings
themselves. To achieve computational efficiency, we integrate a sliding window
technique with the Rabin-Karp algorithm and Rabin Fingerprint, enabling
hash-based rolling comparisons to detect string matches. This approach
significantly reduces time complexity compared to traditional
character-by-character comparisons. We implement the proposed system using
gnark, a high-performance zk-SNARK library, which generates succinct and
verifiable proofs for privacy-preserving string matching. Experimental results
demonstrate that our solution achieves strong privacy guarantees while
maintaining computational efficiency and scalability. This work highlights the
practical applications of zero-knowledge proofs in secure data verification and
contributes a scalable method for privacy-preserving string matching.

### 6. [In Search of Lost Data: A Study of Flash Sanitization Practices](http://arxiv.org/pdf/2505.14067v1)

Authors: Janine Schneider, Immanuel Lautner, Denise Moussa, Julian Wolf, Nicole Scheler, Felix Freiling, Jaap Haasnoot, Hans Henseler, Simon Malik, Holger Morgenstern, Martin Westman

To avoid the disclosure of personal or corporate data, sanitization of
storage devices is an important issue when such devices are to be reused. While
poor sanitization practices have been reported for second-hand hard disk
drives, it has been reported that data has been found on original storage
devices based on flash technology. Based on insights into the second-hand chip
market in China, we report on the results of the first large-scale study on the
effects of chip reuse for USB flash drives. We provide clear evidence of poor
sanitization practices in a non-negligible fraction of USB flash drives from
the low-cost Chinese market that were sold as original. More specifically, we
forensically analyzed 614 USB flash drives and were able to recover non-trivial
user data on a total of 75 devices (more than 12 %). This non-negligible
probability that any data (including incriminating files) already existed on
the drive when it was bought has critical implications to forensic
investigations. The absence of external factors which correlate with finding
data on new USB flash drives complicates the matter further.

### 7. [Destabilizing Power Grid and Energy Market by Cyberattacks on Smart Inverters](http://arxiv.org/pdf/2505.14175v1)

Authors: Xiangyu Hui, Samuel Karumba, Sid Chi-Kin Chau, Mohiuddin Ahmed

Cyberattacks on smart inverters and distributed PV are becoming an imminent
threat, because of the recent well-documented vulnerabilities and attack
incidents. Particularly, the long lifespan of inverter devices, users' oblivion
of cybersecurity compliance, and the lack of cyber regulatory frameworks
exacerbate the prospect of cyberattacks on smart inverters. As a result, this
raises a question -- "do cyberattacks on smart inverters, if orchestrated on a
large scale, pose a genuine threat of wide-scale instability to the power grid
and energy market"? This paper provides a realistic assessment on the
plausibility and impacts of wide-scale power instability caused by cyberattacks
on smart inverters. We conduct an in-depth study based on the electricity
market data of Australia and the knowledge of practical contingency mechanisms.
Our key findings reveal: (1) Despite the possibility of disruption to the grid
by cyberattacks on smart inverters, the impact is only significant under
careful planning and orchestration. (2) While the grid can assure certain power
system security to survive inadvertent contingency events, it is insufficient
to defend against savvy attackers who can orchestrate attacks in an adversarial
manner. Our data analysis of Australia's electricity grid also reveals that a
relatively low percentage of distributed PV would be sufficient to launch an
impactful concerted attack on the grid. Our study casts insights on robust
strategies for defending the grid in the presence of cyberattacks for places
with high penetration of distributed PV.

### 8. [Effects of the Cyber Resilience Act (CRA) on Industrial Equipment Manufacturing Companies](http://arxiv.org/pdf/2505.14325v1)

Authors: Roosa Risto, Mohit Sethi, Mika Katara

The Cyber Resilience Act (CRA) is a new European Union (EU) regulation aimed
at enhancing the security of digital products and services by ensuring they
meet stringent cybersecurity requirements. This paper investigates the
challenges that industrial equipment manufacturing companies anticipate while
preparing for compliance with CRA through a comprehensive survey. Key findings
highlight significant hurdles such as implementing secure development lifecycle
practices, managing vulnerability notifications within strict timelines, and
addressing gaps in cybersecurity expertise. This study provides insights into
these specific challenges and offers targeted recommendations on key focus
areas, such as tooling improvements, to aid industrial equipment manufacturers
in their preparation for CRA compliance.

### 9. [QUT-DV25: A Dataset for Dynamic Analysis of Next-Gen Software Supply Chain Attacks](http://arxiv.org/pdf/2505.13804v1)

Authors: Sk Tanzir Mehedi, Raja Jurdak, Chadni Islam, Gowri Ramachandran

Securing software supply chains is a growing challenge due to the inadequacy
of existing datasets in capturing the complexity of next-gen attacks, such as
multiphase malware execution, remote access activation, and dynamic payload
generation. Existing datasets, which rely on metadata inspection and static
code analysis, are inadequate for detecting such attacks. This creates a
critical gap because these datasets do not capture what happens during and
after a package is installed. To address this gap, we present QUT-DV25, a
dynamic analysis dataset specifically designed to support and advance research
on detecting and mitigating supply chain attacks within the Python Package
Index (PyPI) ecosystem. This dataset captures install and post-install-time
traces from 14,271 Python packages, of which 7,127 are malicious. The packages
are executed in an isolated sandbox environment using an extended Berkeley
Packet Filter (eBPF) kernel and user-level probes. It captures 36 real-time
features, that includes system calls, network traffic, resource usages,
directory access patterns, dependency logs, and installation behaviors,
enabling the study of next-gen attack vectors. ML analysis using the QUT-DV25
dataset identified four malicious PyPI packages previously labeled as benign,
each with thousands of downloads. These packages deployed covert remote access
and multi-phase payloads, were reported to PyPI maintainers, and subsequently
removed. This highlights the practical value of QUT-DV25, as it outperforms
reactive, metadata, and static datasets, offering a robust foundation for
developing and benchmarking advanced threat detection within the evolving
software supply chain ecosystem.

### 10. [Quantum Opacity, Classical Clarity: A Hybrid Approach to Quantum Circuit Obfuscation](http://arxiv.org/pdf/2505.13848v1)

Authors: Amal Raj, Vivek Balachandran

Quantum computing leverages quantum mechanics to achieve computational
advantages over classical hardware, but the use of third-party quantum
compilers in the Noisy Intermediate-Scale Quantum (NISQ) era introduces risks
of intellectual property (IP) exposure. We address this by proposing a novel
obfuscation technique that protects proprietary quantum circuits by inserting
additional quantum gates prior to compilation. These gates corrupt the
measurement outcomes, which are later corrected through a lightweight classical
post-processing step based on the inserted gate structure. Unlike prior methods
that rely on complex quantum reversals, barriers, or physical-to-virtual qubit
mapping, our approach achieves obfuscation using compiler-agnostic classical
correction. We evaluate the technique across five benchmark quantum algorithms
-- Shor's, QAOA, Bernstein-Vazirani, Grover's, and HHL -- using IBM's Qiskit
framework. The results demonstrate high Total Variation Distance (above 0.5)
and consistently negative Degree of Functional Corruption (DFC), confirming
both statistical and functional obfuscation. This shows that our method is a
practical and effective solution for the security of quantum circuit designs in
untrusted compilation flows.

### Computer Vision and Pattern Recognition

### 1. [Transfer Learning from Visual Speech Recognition to Mouthing Recognition in German Sign Language](http://arxiv.org/pdf/2505.13784v1)

Authors: Dinh Nam Pham, Eleftherios Avramidis

Sign Language Recognition (SLR) systems primarily focus on manual gestures,
but non-manual features such as mouth movements, specifically mouthing, provide
valuable linguistic information. This work directly classifies mouthing
instances to their corresponding words in the spoken language while exploring
the potential of transfer learning from Visual Speech Recognition (VSR) to
mouthing recognition in German Sign Language. We leverage three VSR datasets:
one in English, one in German with unrelated words and one in German containing
the same target words as the mouthing dataset, to investigate the impact of
task similarity in this setting. Our results demonstrate that multi-task
learning improves both mouthing recognition and VSR accuracy as well as model
robustness, suggesting that mouthing recognition should be treated as a
distinct but related task to VSR. This research contributes to the field of SLR
by proposing knowledge transfer from VSR to SLR datasets with limited mouthing
annotations.

### 2. [Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels](http://arxiv.org/pdf/2505.13788v1)

Authors: Yongshuo Zong, Qin Zhang, Dongsheng An, Zhihua Li, Xiang Xu, Linghan Xu, Zhuowen Tu, Yifan Xing, Onkar Dabeer

This work presents a simple yet effective workflow for automatically scaling
instruction-following data to elicit pixel-level grounding capabilities of VLMs
under complex instructions. In particular, we address five critical real-world
challenges in text-instruction-based grounding: hallucinated references,
multi-object scenarios, reasoning, multi-granularity, and part-level
references. By leveraging knowledge distillation from a pre-trained teacher
model, our approach generates high-quality instruction-response pairs linked to
existing pixel-level annotations, minimizing the need for costly human
annotation. The resulting dataset, Ground-V, captures rich object localization
knowledge and nuanced pixel-level referring expressions. Experiment results
show that models trained on Ground-V exhibit substantial improvements across
diverse grounding tasks. Specifically, incorporating Ground-V during training
directly achieves an average accuracy boost of 4.4% for LISA and a 7.9% for
PSALM across six benchmarks on the gIoU metric. It also sets new
state-of-the-art results on standard benchmarks such as RefCOCO/+/g. Notably,
on gRefCOCO, we achieve an N-Acc of 83.3%, exceeding the previous
state-of-the-art by more than 20%.

### 3. [Physics-Driven Local-Whole Elastic Deformation Modeling for Point Cloud Representation Learning](http://arxiv.org/pdf/2505.13812v1)

Authors: Zhongyu Chen, Rong Zhao, Xie Han, Xindong Guo, Song Wang, Zherui Qiao

Existing point cloud representation learning tend to learning the geometric
distribution of objects through data-driven approaches, emphasizing structural
features while overlooking the relationship between the local information and
the whole structure. Local features reflect the fine-grained variations of an
object, while the whole structure is determined by the interaction and
combination of these local features, collectively defining the object's shape.
In real-world, objects undergo elastic deformation under external forces, and
this deformation gradually affects the whole structure through the propagation
of forces from local regions, thereby altering the object's geometric
properties. Inspired by this, we propose a physics-driven self-supervised
learning method for point cloud representation, which captures the relationship
between parts and the whole by constructing a local-whole force propagation
mechanism. Specifically, we employ a dual-task encoder-decoder framework,
integrating the geometric modeling capability of implicit fields with
physics-driven elastic deformation. The encoder extracts features from the
point cloud and its tetrahedral mesh representation, capturing both geometric
and physical properties. These features are then fed into two decoders: one
learns the whole geometric shape of the point cloud through an implicit field,
while the other predicts local deformations using two specifically designed
physics information loss functions, modeling the deformation relationship
between local and whole shapes. Experimental results show that our method
outperforms existing approaches in object classification, few-shot learning,
and segmentation, demonstrating its effectiveness.

### 4. [InstanceBEV: Unifying Instance and BEV Representation for Global Modeling](http://arxiv.org/pdf/2505.13817v1)

Authors: Feng Li, Kun Xu, Zhaoyue Wang, Yunduan Cui, Mohammad Masum Billah, Jia Liu

Occupancy Grid Maps are widely used in navigation for their ability to
represent 3D space occupancy. However, existing methods that utilize multi-view
cameras to construct Occupancy Networks for perception modeling suffer from
cubic growth in data complexity. Adopting a Bird's-Eye View (BEV) perspective
offers a more practical solution for autonomous driving, as it provides higher
semantic density and mitigates complex object occlusions. Nonetheless,
BEV-based approaches still require extensive engineering optimizations to
enable efficient large-scale global modeling. To address this challenge, we
propose InstanceBEV, the first method to introduce instance-level
dimensionality reduction for BEV, enabling global modeling with transformers
without relying on sparsification or acceleration operators. Different from
other BEV methods, our approach directly employs transformers to aggregate
global features. Compared to 3D object detection models, our method samples
global feature maps into 3D space. Experiments on OpenOcc-NuScenes dataset show
that InstanceBEV achieves state-of-the-art performance while maintaining a
simple, efficient framework without requiring additional optimizations.

### 5. [MGStream: Motion-aware 3D Gaussian for Streamable Dynamic Scene Reconstruction](http://arxiv.org/pdf/2505.13839v1)

Authors: Zhenyu Bao, Qing Li, Guibiao Liao, Zhongyuan Zhao, Kanglin Liu

3D Gaussian Splatting (3DGS) has gained significant attention in streamable
dynamic novel view synthesis (DNVS) for its photorealistic rendering capability
and computational efficiency. Despite much progress in improving rendering
quality and optimization strategies, 3DGS-based streamable dynamic scene
reconstruction still suffers from flickering artifacts and storage
inefficiency, and struggles to model the emerging objects. To tackle this, we
introduce MGStream which employs the motion-related 3D Gaussians (3DGs) to
reconstruct the dynamic and the vanilla 3DGs for the static. The motion-related
3DGs are implemented according to the motion mask and the clustering-based
convex hull algorithm. The rigid deformation is applied to the motion-related
3DGs for modeling the dynamic, and the attention-based optimization on the
motion-related 3DGs enables the reconstruction of the emerging objects. As the
deformation and optimization are only conducted on the motion-related 3DGs,
MGStream avoids flickering artifacts and improves the storage efficiency.
Extensive experiments on real-world datasets N3DV and MeetRoom demonstrate that
MGStream surpasses existing streaming 3DGS-based approaches in terms of
rendering quality, training/storage efficiency and temporal consistency. Our
code is available at: https://github.com/pcl3dv/MGStream.

### 6. [SuperMapNet for Long-Range and High-Accuracy Vectorized HD Map Construction](http://arxiv.org/pdf/2505.13856v1)

Authors: Ruqin Zhou, San Jiang, Wanshou Jiang, Yongsheng Zhang, Chenguang Dai

Vectorized HD map is essential for autonomous driving. Remarkable work has
been achieved in recent years, but there are still major issues: (1) in the
generation of the BEV features, single modality-based methods are of limited
perception capability, while direct concatenation-based multi-modal methods
fail to capture synergies and disparities between different modalities,
resulting in limited ranges with feature holes; (2) in the classification and
localization of map elements, only point information is used without the
consideration of element infor-mation and neglects the interaction between
point information and element information, leading to erroneous shapes and
element entanglement with low accuracy. To address above issues, we introduce
SuperMapNet for long-range and high-accuracy vectorized HD map construction. It
uses both camera images and LiDAR point clouds as input, and first tightly
couple semantic information from camera images and geometric information from
LiDAR point clouds by a cross-attention based synergy enhancement module and a
flow-based disparity alignment module for long-range BEV feature generation.
And then, local features from point queries and global features from element
queries are tightly coupled by three-level interactions for high-accuracy
classification and localization, where Point2Point interaction learns local
geometric information between points of the same element and of each point,
Element2Element interaction learns relation constraints between different
elements and semantic information of each elements, and Point2Element
interaction learns complement element information for its constituent points.
Experiments on the nuScenes and Argoverse2 datasets demonstrate superior
performances, surpassing SOTAs over 14.9/8.8 mAP and 18.5/3.1 mAP under
hard/easy settings, respectively. The code is made publicly available1.

### 7. [An Explorative Analysis of SVM Classifier and ResNet50 Architecture on African Food Classification](http://arxiv.org/pdf/2505.13923v1)

Authors: Chinedu Emmanuel Mbonu, Kenechukwu Anigbogu, Doris Asogwa, Tochukwu Belonwu

Food recognition systems has advanced significantly for Western cuisines, yet
its application to African foods remains underexplored. This study addresses
this gap by evaluating both deep learning and traditional machine learning
methods for African food classification. We compared the performance of a
fine-tuned ResNet50 model with a Support Vector Machine (SVM) classifier. The
dataset comprises 1,658 images across six selected food categories that are
known in Africa. To assess model effectiveness, we utilize five key evaluation
metrics: Confusion matrix, F1-score, accuracy, recall and precision. Our
findings offer valuable insights into the strengths and limitations of both
approaches, contributing to the advancement of food recognition for African
cuisines.

### 8. [Every Pixel Tells a Story: End-to-End Urdu Newspaper OCR](http://arxiv.org/pdf/2505.13943v1)

Authors: Samee Arif, Sualeha Farid

This paper introduces a comprehensive end-to-end pipeline for Optical
Character Recognition (OCR) on Urdu newspapers. In our approach, we address the
unique challenges of complex multi-column layouts, low-resolution archival
scans, and diverse font styles. Our process decomposes the OCR task into four
key modules: (1) article segmentation, (2) image super-resolution, (3) column
segmentation, and (4) text recognition. For article segmentation, we fine-tune
and evaluate YOLOv11x to identify and separate individual articles from
cluttered layouts. Our model achieves a precision of 0.963 and mAP@50 of 0.975.
For super-resolution, we fine-tune and benchmark the SwinIR model (reaching
32.71 dB PSNR) to enhance the quality of degraded newspaper scans. To do our
column segmentation, we use YOLOv11x to separate columns in text to further
enhance performance - this model reaches a precision of 0.970 and mAP@50 of
0.975. In the text recognition stage, we benchmark a range of LLMs from
different families, including Gemini, GPT, Llama, and Claude. The lowest WER of
0.133 is achieved by Gemini-2.5-Pro.

### 9. [StPR: Spatiotemporal Preservation and Routing for Exemplar-Free Video Class-Incremental Learning](http://arxiv.org/pdf/2505.13997v1)

Authors: Huaijie Wang, De Cheng, Guozhang Li, Zhipeng Xu, Lingfeng He, Jie Li, Nannan Wang, Xinbo Gao

Video Class-Incremental Learning (VCIL) seeks to develop models that
continuously learn new action categories over time without forgetting
previously acquired knowledge. Unlike traditional Class-Incremental Learning
(CIL), VCIL introduces the added complexity of spatiotemporal structures,
making it particularly challenging to mitigate catastrophic forgetting while
effectively capturing both frame-shared semantics and temporal dynamics.
Existing approaches either rely on exemplar rehearsal, raising concerns over
memory and privacy, or adapt static image-based methods that neglect temporal
modeling. To address these limitations, we propose Spatiotemporal Preservation
and Routing (StPR), a unified and exemplar-free VCIL framework that explicitly
disentangles and preserves spatiotemporal information. First, we introduce
Frame-Shared Semantics Distillation (FSSD), which identifies semantically
stable and meaningful channels by jointly considering semantic sensitivity and
classification contribution. These important semantic channels are selectively
regularized to maintain prior knowledge while allowing for adaptation. Second,
we design a Temporal Decomposition-based Mixture-of-Experts (TD-MoE), which
dynamically routes task-specific experts based on their temporal dynamics,
enabling inference without task ID or stored exemplars. Together, StPR
effectively leverages spatial semantics and temporal dynamics, achieving a
unified, exemplar-free VCIL framework. Extensive experiments on UCF101, HMDB51,
and Kinetics400 show that our method outperforms existing baselines while
offering improved interpretability and efficiency in VCIL. Code is available in
the supplementary materials.

### 10. [Multi-Label Stereo Matching for Transparent Scene Depth Estimation](http://arxiv.org/pdf/2505.14008v1)

Authors: Zhidan Liu, Chengtang Yao, Jiaxi Zeng, Yuwei Wu, Yunde Jia

In this paper, we present a multi-label stereo matching method to
simultaneously estimate the depth of the transparent objects and the occluded
background in transparent scenes.Unlike previous methods that assume a unimodal
distribution along the disparity dimension and formulate the matching as a
single-label regression problem, we propose a multi-label regression
formulation to estimate multiple depth values at the same pixel in transparent
scenes. To resolve the multi-label regression problem, we introduce a
pixel-wise multivariate Gaussian representation, where the mean vector encodes
multiple depth values at the same pixel, and the covariance matrix determines
whether a multi-label representation is necessary for a given pixel. The
representation is iteratively predicted within a GRU framework. In each
iteration, we first predict the update step for the mean parameters and then
use both the update step and the updated mean parameters to estimate the
covariance matrix. We also synthesize a dataset containing 10 scenes and 89
objects to validate the performance of transparent scene depth estimation. The
experiments show that our method greatly improves the performance on
transparent surfaces while preserving the background information for scene
reconstruction. Code is available at https://github.com/BFZD233/TranScene.

### Computers and Society

### 1. [Safety Devolution in AI Agents](http://arxiv.org/pdf/2505.14215v1)

Authors: Cheng Yu, Benedikt Stroebl, Diyi Yang, Orestis Papakyriakopoulos

As retrieval-augmented AI agents become more embedded in society, their
safety properties and ethical behavior remain insufficiently understood. In
particular, the growing integration of LLMs and AI agents raises critical
questions about how they engage with and are influenced by their environments.
This study investigates how expanding retrieval access, from no external
sources to Wikipedia-based retrieval and open web search, affects model
reliability, bias propagation, and harmful content generation. Through
extensive benchmarking of censored and uncensored LLMs and AI Agents, our
findings reveal a consistent degradation in refusal rates, bias sensitivity,
and harmfulness safeguards as models gain broader access to external sources,
culminating in a phenomenon we term safety devolution. Notably,
retrieval-augmented agents built on aligned LLMs often behave more unsafely
than uncensored models without retrieval. This effect persists even under
strong retrieval accuracy and prompt-based mitigation, suggesting that the mere
presence of retrieved content reshapes model behavior in structurally unsafe
ways. These findings underscore the need for robust mitigation strategies to
ensure fairness and reliability in retrieval-augmented and increasingly
autonomous AI systems.

### 2. [Choosing a Model, Shaping a Future: Comparing LLM Perspectives on Sustainability and its Relationship with AI](http://arxiv.org/pdf/2505.14435v1)

Authors: Annika Bush, Meltem Aksoy, Markus Pauly, Greta Ontrup

As organizations increasingly rely on AI systems for decision support in
sustainability contexts, it becomes critical to understand the inherent biases
and perspectives embedded in Large Language Models (LLMs). This study
systematically investigates how five state-of-the-art LLMs -- Claude, DeepSeek,
GPT, LLaMA, and Mistral - conceptualize sustainability and its relationship
with AI. We administered validated, psychometric sustainability-related
questionnaires - each 100 times per model -- to capture response patterns and
variability. Our findings revealed significant inter-model differences: For
example, GPT exhibited skepticism about the compatibility of AI and
sustainability, whereas LLaMA demonstrated extreme techno-optimism with perfect
scores for several Sustainable Development Goals (SDGs). Models also diverged
in attributing institutional responsibility for AI and sustainability
integration, a results that holds implications for technology governance
approaches. Our results demonstrate that model selection could substantially
influence organizational sustainability strategies, highlighting the need for
awareness of model-specific biases when deploying LLMs for
sustainability-related decision-making.

### 3. [Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models](http://arxiv.org/pdf/2505.14617v1)

Authors: Sahar Abdelnabi, Ahmed Salem

Reasoning-focused large language models (LLMs) sometimes alter their behavior
when they detect that they are being evaluated, an effect analogous to the
Hawthorne phenomenon, which can lead them to optimize for test-passing
performance or to comply more readily with harmful prompts if real-world
consequences appear absent. We present the first quantitative study of how such
"test awareness" impacts model behavior, particularly its safety alignment. We
introduce a white-box probing framework that (i) linearly identifies
awareness-related activations and (ii) steers models toward or away from test
awareness while monitoring downstream performance. We apply our method to
different state-of-the-art open-source reasoning LLMs across both realistic and
hypothetical tasks. Our results demonstrate that test awareness significantly
impact safety alignment, and is different for different models. By providing
fine-grained control over this latent effect, our work aims to increase trust
in how we perform safety evaluation.

### 4. [Fragments to Facts: Partial-Information Fragment Inference from LLMs](http://arxiv.org/pdf/2505.13819v1)

Authors: Lucas Rosenblatt, Bin Han, Robert Wolfe, Bill Howe

Large language models (LLMs) can leak sensitive training data through
memorization and membership inference attacks. Prior work has primarily focused
on strong adversarial assumptions, including attacker access to entire samples
or long, ordered prefixes, leaving open the question of how vulnerable LLMs are
when adversaries have only partial, unordered sample information. For example,
if an attacker knows a patient has "hypertension," under what conditions can
they query a model fine-tuned on patient data to learn the patient also has
"osteoarthritis?" In this paper, we introduce a more general threat model under
this weaker assumption and show that fine-tuned LLMs are susceptible to these
fragment-specific extraction attacks. To systematically investigate these
attacks, we propose two data-blind methods: (1) a likelihood ratio attack
inspired by methods from membership inference, and (2) a novel approach, PRISM,
which regularizes the ratio by leveraging an external prior. Using examples
from both medical and legal settings, we show that both methods are competitive
with a data-aware baseline classifier that assumes access to labeled
in-distribution data, underscoring their robustness.

### 5. [Social Sycophancy: A Broader Understanding of LLM Sycophancy](http://arxiv.org/pdf/2505.13995v1)

Authors: Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, Dan Jurafsky

A serious risk to the safety and utility of LLMs is sycophancy, i.e.,
excessive agreement with and flattery of the user. Yet existing work focuses on
only one aspect of sycophancy: agreement with users' explicitly stated beliefs
that can be compared to a ground truth. This overlooks forms of sycophancy that
arise in ambiguous contexts such as advice and support-seeking, where there is
no clear ground truth, yet sycophancy can reinforce harmful implicit
assumptions, beliefs, or actions. To address this gap, we introduce a richer
theory of social sycophancy in LLMs, characterizing sycophancy as the excessive
preservation of a user's face (the positive self-image a person seeks to
maintain in an interaction). We present ELEPHANT, a framework for evaluating
social sycophancy across five face-preserving behaviors (emotional validation,
moral endorsement, indirect language, indirect action, and accepting framing)
on two datasets: open-ended questions (OEQ) and Reddit's r/AmITheAsshole
(AITA). Across eight models, we show that LLMs consistently exhibit high rates
of social sycophancy: on OEQ, they preserve face 47% more than humans, and on
AITA, they affirm behavior deemed inappropriate by crowdsourced human judgments
in 42% of cases. We further show that social sycophancy is rewarded in
preference datasets and is not easily mitigated. Our work provides theoretical
grounding and empirical tools (datasets and code) for understanding and
addressing this under-recognized but consequential issue.

### 6. [Gender Trouble in Language Models: An Empirical Audit Guided by Gender Performativity Theory](http://arxiv.org/pdf/2505.14080v1)

Authors: Franziska Sofia Hafner, Ana Valdivia, Luc Rocher

Language models encode and subsequently perpetuate harmful gendered
stereotypes. Research has succeeded in mitigating some of these harms, e.g. by
dissociating non-gendered terms such as occupations from gendered terms such as
'woman' and 'man'. This approach, however, remains superficial given that
associations are only one form of prejudice through which gendered harms arise.
Critical scholarship on gender, such as gender performativity theory,
emphasizes how harms often arise from the construction of gender itself, such
as conflating gender with biological sex. In language models, these issues
could lead to the erasure of transgender and gender diverse identities and
cause harms in downstream applications, from misgendering users to
misdiagnosing patients based on wrong assumptions about their anatomy.
  For FAccT research on gendered harms to go beyond superficial linguistic
associations, we advocate for a broader definition of 'gender bias' in language
models. We operationalize insights on the construction of gender through
language from gender studies literature and then empirically test how 16
language models of different architectures, training datasets, and model sizes
encode gender. We find that language models tend to encode gender as a binary
category tied to biological sex, and that gendered terms that do not neatly
fall into one of these binary categories are erased and pathologized. Finally,
we show that larger models, which achieve better results on performance
benchmarks, learn stronger associations between gender and sex, further
reinforcing a narrow understanding of gender. Our findings lead us to call for
a re-evaluation of how gendered harms in language models are defined and
addressed.

### 7. [MAS-KCL: Knowledge component graph structure learning with large language model-based agentic workflow](http://arxiv.org/pdf/2505.14126v1)

Authors: Yuan-Hao Jiang, Kezong Tang, Zi-Wei Chen, Yuang Wei, Tian-Yi Liu, Jiayi Wu

Knowledge components (KCs) are the fundamental units of knowledge in the
field of education. A KC graph illustrates the relationships and dependencies
between KCs. An accurate KC graph can assist educators in identifying the root
causes of learners' poor performance on specific KCs, thereby enabling targeted
instructional interventions. To achieve this, we have developed a KC graph
structure learning algorithm, named MAS-KCL, which employs a multi-agent system
driven by large language models for adaptive modification and optimization of
the KC graph. Additionally, a bidirectional feedback mechanism is integrated
into the algorithm, where AI agents leverage this mechanism to assess the value
of edges within the KC graph and adjust the distribution of generation
probabilities for different edges, thereby accelerating the efficiency of
structure learning. We applied the proposed algorithm to 5 synthetic datasets
and 4 real-world educational datasets, and experimental results validate its
effectiveness in learning path recognition. By accurately identifying learners'
learning paths, teachers are able to design more comprehensive learning plans,
enabling learners to achieve their educational goals more effectively, thus
promoting the sustainable development of education.

### 8. [Data-Efficient Hate Speech Detection via Cross-Lingual Nearest Neighbor Retrieval with Limited Labeled Data](http://arxiv.org/pdf/2505.14272v1)

Authors: Faeze Ghorbanpour, Daryna Dementieva, Alexander Fraser

Considering the importance of detecting hateful language, labeled hate speech
data is expensive and time-consuming to collect, particularly for low-resource
languages. Prior work has demonstrated the effectiveness of cross-lingual
transfer learning and data augmentation in improving performance on tasks with
limited labeled data. To develop an efficient and scalable cross-lingual
transfer learning approach, we leverage nearest-neighbor retrieval to augment
minimal labeled data in the target language, thereby enhancing detection
performance. Specifically, we assume access to a small set of labeled training
instances in the target language and use these to retrieve the most relevant
labeled examples from a large multilingual hate speech detection pool. We
evaluate our approach on eight languages and demonstrate that it consistently
outperforms models trained solely on the target language data. Furthermore, in
most cases, our method surpasses the current state-of-the-art. Notably, our
approach is highly data-efficient, retrieving as small as 200 instances in some
cases while maintaining superior performance. Moreover, it is scalable, as the
retrieval pool can be easily expanded, and the method can be readily adapted to
new languages and tasks. We also apply maximum marginal relevance to mitigate
redundancy and filter out highly similar retrieved instances, resulting in
improvements in some languages.

### 9. [Towards Verifiability of Total Value Locked (TVL) in Decentralized Finance](http://arxiv.org/pdf/2505.14565v1)

Authors: Pietro Saggese, Michael Fröwis, Stefan Kitzler, Bernhard Haslhofer, Raphael Auer

Total Value Locked (TVL) aims to measure the aggregate value of cryptoassets
deposited in Decentralized Finance (DeFi) protocols. Although blockchain data
is public, the way TVL is computed is not well understood. In practice, its
calculation on major TVL aggregators relies on self-reports from community
members and lacks standardization, making it difficult to verify published
figures independently. We thus conduct a systematic study on 939 DeFi projects
deployed in Ethereum. We study the methodologies used to compute TVL, examine
factors hindering verifiability, and ultimately propose standardization
attempts in the field. We find that 10.5% of the protocols rely on external
servers; 68 methods alternative to standard balance queries exist, although
their use decreased over time; and 240 equal balance queries are repeated on
multiple protocols. These findings indicate limits to verifiability and
transparency. We thus introduce ``verifiable Total Value Locked'' (vTVL), a
metric measuring the TVL that can be verified relying solely on on-chain data
and standard balance queries. A case study on 400 protocols shows that our
estimations align with published figures for 46.5% of protocols. Informed by
these findings, we discuss design guidelines that could facilitate a more
verifiable, standardized, and explainable TVL computation.

### 10. [Upgrading Democracies with Fairer Voting Methods](http://arxiv.org/pdf/2505.14349v1)

Authors: Evangelos Pournaras, Srijoni Majumdar, Thomas Wellings, Joshua C. Yang, Fatemeh B. Heravan, Regula Hänggli Fricker, Dirk Helbing

Voting methods are instrumental design element of democracies. Citizens use
them to express and aggregate their preferences to reach a collective decision.
However, voting outcomes can be as sensitive to voting rules as they are to
people's voting choices. Despite the significance and inter-disciplinary
scientific progress on voting methods, several democracies keep relying on
outdated voting methods that do not fit modern, pluralistic societies well,
while lacking social innovation. Here, we demonstrate how one can upgrade
real-world democracies, namely by using alternative preferential voting methods
such as cumulative voting and the method of equal shares designed for a
proportional representation of voters' preferences. By rigorously assessing a
new participatory budgeting approach applied in the city of Aarau, Switzerland,
we unravel the striking voting outcomes of fair voting methods: more winning
projects with the same budget and broader geographic and preference
representation of citizens by the elected projects, in particular for voters
who used to be under-represented, while promoting novel project ideas. We
provide profound causal evidence showing that citizens prefer proportional
voting methods, which possess strong legitimacy without the need of very
technical specialized explanations. We also reveal strong underlying democratic
values exhibited by citizens who support fair voting methods such as altruism
and compromise. These findings come with a global momentum to unleash a new and
long-awaited participation blueprint of how to upgrade democracies.

### Databases

### 1. [Detecting Flow Gaps in Data Streams](http://arxiv.org/pdf/2505.13945v1)

Authors: Siyuan Dong, Yuxuan Tian, Wenhan Ma, Tong Yang, Chenye Zhang, Yuhan Wu, Kaicheng Yang, Yaojing Wang

Data stream monitoring is a crucial task which has a wide range of
applications. The majority of existing research in this area can be broadly
classified into two types, monitoring value sum and monitoring value
cardinality. In this paper, we define a third type, monitoring value variation,
which can help us detect flow gaps in data streams. To realize this function,
we propose GapFilter, leveraging the idea of Sketch for achieving speed and
accuracy. To the best of our knowledge, this is the first work to detect flow
gaps in data streams. Two key ideas of our work are the similarity absorption
technique and the civilian-suspect mechanism. The similarity absorption
technique helps in reducing memory usage and enhancing speed, while the
civilian-suspect mechanism further boosts accuracy by organically integrating
broad monitoring of overall flows with meticulous monitoring of suspicious
flows.We have developed two versions of GapFilter. Speed-Oriented GapFilter
(GapFilter-SO) emphasizes speed while maintaining satisfactory accuracy.
Accuracy-Oriented GapFilter (GapFilter-AO) prioritizes accuracy while ensuring
considerable speed. We provide a theoretical proof demonstrating that GapFilter
secures high accuracy with minimal memory usage. Further, extensive experiments
were conducted to assess the accuracy and speed of our algorithms. The results
reveal that GapFilter-AO requires, on average, 1/32 of the memory to match the
accuracy of the Straw-man solution. GapFilter-SO operates at a speed 3 times
faster than the Straw-man solution. All associated source code has been
open-sourced and is available on GitHub.

### 2. [Evaluating the Impact Of Spatial Features Of Mobility Data and Index Choice On Database Performance](http://arxiv.org/pdf/2505.14466v1)

Authors: Tim C. Rese, Alexandra Kapp, David Bermbach

The growing number of moving Internet-of-Things (IoT) devices has led to a
surge in moving object data, powering applications such as traffic routing,
hotspot detection, or weather forecasting. When managing such data, spatial
database systems offer various index options and data formats, e.g.,
point-based or trajectory-based. Likewise, dataset characteristics such as
geographic overlap and skew can vary significantly. All three significantly
affect database performance. While this has been studied in existing papers,
none of them explore the effects and trade-offs resulting from a combination of
all three aspects. In this paper, we evaluate the performance impact of index
choice, data format, and dataset characteristics on a popular spatial database
system, PostGIS. We focus on two aspects of dataset characteristics, the degree
of overlap and the degree of skew, and propose novel approximation methods to
determine these features. We design a benchmark that compares a variety of
spatial indexing strategies and data formats, while also considering the impact
of dataset characteristics on database performance. We include a variety of
real-world and synthetic datasets, write operations, and read queries to cover
a broad range of scenarios that might occur during application runtime. Our
results offer practical guidance for developers looking to optimize spatial
storage and querying, while also providing insights into dataset
characteristics and their impact on database performance.

### 3. [Abacus: A Cost-Based Optimizer for Semantic Operator Systems](http://arxiv.org/pdf/2505.14661v1)

Authors: Matthew Russo, Sivaprasad Sudhir, Gerardo Vitagliano, Chunwei Liu, Tim Kraska, Samuel Madden, Michael Cafarella

LLMs enable an exciting new class of data processing applications over large
collections of unstructured documents. Several new programming frameworks have
enabled developers to build these applications by composing them out of
semantic operators: a declarative set of AI-powered data transformations with
natural language specifications. These include LLM-powered maps, filters,
joins, etc. used for document processing tasks such as information extraction,
summarization, and more. While systems of semantic operators have achieved
strong performance on benchmarks, they can be difficult to optimize. An
optimizer for this setting must determine how to physically implement each
semantic operator in a way that optimizes the system globally. Existing
optimizers are limited in the number of optimizations they can apply, and most
(if not all) cannot optimize system quality, cost, or latency subject to
constraint(s) on the other dimensions. In this paper we present Abacus, an
extensible, cost-based optimizer which searches for the best implementation of
a semantic operator system given a (possibly constrained) optimization
objective. Abacus estimates operator performance by leveraging a minimal set of
validation examples and, if available, prior beliefs about operator
performance. We evaluate Abacus on document processing workloads in the
biomedical and legal domains (BioDEX; CUAD) and multi-modal question answering
(MMQA). We demonstrate that systems optimized by Abacus achieve 18.7%-39.2%
better quality and up to 23.6x lower cost and 4.2x lower latency than the next
best system.

### 4. [VulCPE: Context-Aware Cybersecurity Vulnerability Retrieval and Management](http://arxiv.org/pdf/2505.13895v1)

Authors: Yuning Jiang, Feiyang Shang, Freedy Tan Wei You, Huilin Wang, Chia Ren Cong, Qiaoran Meng, Nay Oo, Hoon Wei Lim, Biplab Sikdar

The dynamic landscape of cybersecurity demands precise and scalable solutions
for vulnerability management in heterogeneous systems, where
configuration-specific vulnerabilities are often misidentified due to
inconsistent data in databases like the National Vulnerability Database (NVD).
Inaccurate Common Platform Enumeration (CPE) data in NVD further leads to false
positives and incomplete vulnerability retrieval. Informed by our systematic
analysis of CPE and CVEdeails data, revealing more than 50% vendor name
inconsistencies, we propose VulCPE, a framework that standardizes data and
models configuration dependencies using a unified CPE schema (uCPE), entity
recognition, relation extraction, and graph-based modeling. VulCPE achieves
superior retrieval precision (0.766) and coverage (0.926) over existing tools.
VulCPE ensures precise, context-aware vulnerability management, enhancing cyber
resilience.

### Distributed, Parallel, and Cluster Computing

### 1. [Paradigm Shift in Infrastructure Inspection Technology: Leveraging High-performance Imaging and Advanced AI Analytics to Inspect Road Infrastructure](http://arxiv.org/pdf/2505.13955v1)

Authors: Du Wu, Enzhi Zhang, Isaac Lyngaas, Xiao Wang, Amir Ziabari, Tao Luo, Peng Chen, Kento Sato, Fumiyoshi Shoji, Takaki Hatsui, Kentaro Uesugi, Akira Seo, Yasuhito Sakai, Toshio Endo, Tetsuya Ishikawa, Satoshi Matsuoka, Mohamed Wahib

Effective road infrastructure management is crucial for modern society.
Traditional manual inspection techniques remain constrained by cost,
efficiency, and scalability, while camera and laser imaging methods fail to
capture subsurface defects critical for long-term structural integrity. This
paper introduces ROVAI, an end-to-end framework that integrates high-resolution
X-ray computed tomography imaging and advanced AI-driven analytics, aiming to
transform road infrastructure inspection technologies. By leveraging the
computational power of world-leading supercomputers, Fugaku and Frontier, and
SoTA synchrotron facility (Spring-8), ROVAI enables scalable and
high-throughput processing of massive 3D tomographic datasets. Our approach
overcomes key challenges, such as the high memory requirements of vision
models, the lack of labeled training data, and storage I/O bottlenecks. This
seamless integration of imaging and AI analytics facilitates automated defect
detection, material composition analysis, and lifespan prediction. Experimental
results demonstrate the effectiveness of ROVAI in real-world scenarios, setting
a new standard for intelligent, data-driven infrastructure management.

### 2. [Prime Collective Communications Library -- Technical Report](http://arxiv.org/pdf/2505.14065v1)

Authors: Michael Keiblinger, Mario Sieg, Jack Min Ong, Sami Jaghouar, Johannes Hagemann

This report presents the Prime Collective Communications Library (PCCL), a
novel fault-tolerant collective communication library designed for distributed
ML workloads over the public internet. PCCL introduces a new programming model
that enables dynamic peer joining and failure recovery. The library implements
efficient collective operations like all-reduce while providing robust fault
tolerance mechanisms that allow the system to continue operating even when
peers fail or join during ongoing operations. We demonstrate that PCCL's design
enables practical solutions to dynamic membership challenges in workloads with
repeated operations and deterministic state advancement. Our implementation
passes extensive stress tests across all major operating systems, showing
reliable operation even under rapid peer churn and concurrent collective
operations. By dispatching to multiple connections, we can efficiently utilize
cross-continental long-fat-pipe TCP WAN links, in our experiments achieving up
to 45 Gbit/s of bandwidth utilization across Europe and 25 Gbit/s across North
America and Europe. PCCL's architecture enables easy implementation of
distributed low-communication optimization strategies like DiLoCo, which
significantly reduce communication frequency. Combined with quantization, this
leads to a significant reduction in the bandwidth required for distributed
training workloads. PCCL also allows for concurrent collective operations,
which enables optimization strategies like async DiLoCo, which can completely
hide communication overhead by implementing one-step delayed parameter updates.
PCCL can facilitate exact bit-parity of the shared state across peers in all
cases induced by graceful or abrupt peer churn. While PCCL exposes a C99 API,
Python bindings are available which are compatible with PyTorch alongside FSDP.
PCCL is available under the open source MIT license.

### 3. [SkyMemory: A LEO Edge Cache for Transformer Inference Optimization and Scale Out](http://arxiv.org/pdf/2505.14427v1)

Authors: Thomas Sandholm, Sayandev Mukherjee, Lin Cheng, Bernardo A. Huberman

We expand the scope of cache memory to include LEO constellations, which are
highly distributed systems with thousands of satellites connected with
free-space optics inter-satellite links (ISL) always only one hop from any
point on earth. We show how to increase the number of cache hits and improve
the speed of inference for the important use case of LLMs. These benefits apply
not only to LLMs, both terrestrially hosted and on satellites, but also
generalize to any cache distributed over multiple locations that needs to be
accessed in a timely manner. We show the benefit of our key value cache (KVC)
protocol in simulations and present a proof-of-concept implementation of the
protocol for KVCs on a testbed comprising 5 Intel NUC Linux mini PCs hosting a
19x5 constellation, with an NVIDIA Jetson Nano 8GB GPU hosting the LLM.

### 4. [Evaluating the Impact Of Spatial Features Of Mobility Data and Index Choice On Database Performance](http://arxiv.org/pdf/2505.14466v1)

Authors: Tim C. Rese, Alexandra Kapp, David Bermbach

The growing number of moving Internet-of-Things (IoT) devices has led to a
surge in moving object data, powering applications such as traffic routing,
hotspot detection, or weather forecasting. When managing such data, spatial
database systems offer various index options and data formats, e.g.,
point-based or trajectory-based. Likewise, dataset characteristics such as
geographic overlap and skew can vary significantly. All three significantly
affect database performance. While this has been studied in existing papers,
none of them explore the effects and trade-offs resulting from a combination of
all three aspects. In this paper, we evaluate the performance impact of index
choice, data format, and dataset characteristics on a popular spatial database
system, PostGIS. We focus on two aspects of dataset characteristics, the degree
of overlap and the degree of skew, and propose novel approximation methods to
determine these features. We design a benchmark that compares a variety of
spatial indexing strategies and data formats, while also considering the impact
of dataset characteristics on database performance. We include a variety of
real-world and synthetic datasets, write operations, and read queries to cover
a broad range of scenarios that might occur during application runtime. Our
results offer practical guidance for developers looking to optimize spatial
storage and querying, while also providing insights into dataset
characteristics and their impact on database performance.

### 5. [ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs](http://arxiv.org/pdf/2505.14468v1)

Authors: Yifan Sui, Hao Wang, Hanfei Yu, Yitao Hu, Jianxun Li, Hao Wang

Serverless computing has grown rapidly for serving Large Language Model (LLM)
inference due to its pay-as-you-go pricing, fine-grained GPU usage, and rapid
scaling. However, our analysis reveals that current serverless can effectively
serve general LLM but fail with Low-Rank Adaptation (LoRA) inference due to
three key limitations: 1) massive parameter redundancy among functions where
99% of weights are unnecessarily duplicated, 2) costly artifact loading latency
beyond LLM loading, and 3) magnified resource contention when serving multiple
LoRA LLMs. These inefficiencies lead to massive GPU wastage, increased
Time-To-First-Token (TTFT), and high monetary costs.
  We propose ServerlessLoRA, a novel serverless inference system designed for
faster and cheaper LoRA LLM serving. ServerlessLoRA enables secure backbone LLM
sharing across isolated LoRA functions to reduce redundancy. We design a
pre-loading method that pre-loads comprehensive LoRA artifacts to minimize
cold-start latency. Furthermore, ServerlessLoRA employs contention aware
batching and offloading to mitigate GPU resource conflicts during bursty
workloads. Experiment on industrial workloads demonstrates that ServerlessLoRA
reduces TTFT by up to 86% and cuts monetary costs by up to 89% compared to
state-of-the-art LLM inference solutions.

### 6. [Federated prediction for scalable and privacy-preserved knowledge-based planning in radiotherapy](http://arxiv.org/pdf/2505.14507v1)

Authors: Jingyun Chen, David Horowitz, Yading Yuan

Background: Deep learning has potential to improve the efficiency and
consistency of radiation therapy planning, but clinical adoption is hindered by
the limited model generalizability due to data scarcity and heterogeneity among
institutions. Although aggregating data from different institutions could
alleviate this problem, data sharing is a practical challenge due to concerns
about patient data privacy and other technical obstacles. Purpose: This work
aims to address this dilemma by developing FedKBP+, a comprehensive federated
learning (FL) platform for predictive tasks in real-world applications in
radiotherapy treatment planning. Methods: We implemented a unified
communication stack based on Google Remote Procedure Call (gRPC) to support
communication between participants whether located on the same workstation or
distributed across multiple workstations. In addition to supporting the
centralized FL strategies commonly available in existing open-source
frameworks, FedKBP+ also provides a fully decentralized FL model where
participants directly exchange model weights to each other through Peer-to-Peer
communication. We evaluated FedKBP+ on three predictive tasks using
scale-attention network (SA-Net) as the predictive model. Conclusions: Our
results demonstrate that FedKBP+ is highly effective, efficient and robust,
showing great potential as a federated learning platform for radiation therapy.

### 7. [PSMOA: Policy Support Multi-Objective Optimization Algorithm for Decentralized Data Replication](http://arxiv.org/pdf/2505.14574v2)

Authors: Xi Wang, Susmit Shannigrahi

Efficient data replication in decentralized storage systems must account for
diverse policies, especially in multi-organizational, data-intensive
environments. This work proposes PSMOA, a novel Policy Support Multi-objective
Optimization Algorithm for decentralized data replication that dynamically
adapts to varying organizational requirements such as minimization or
maximization of replication time, storage cost, replication based on content
popularity, and load balancing while respecting policy constraints. PSMOA
outperforms NSGA-II and NSGA-III in both Generational Distance (20.29 vs 148.74
and 67.74) and Inverted Generational Distance (0.78 vs 3.76 and 5.61),
indicating better convergence and solution distribution. These results validate
PSMOA's novelty in optimizing data replication in multi-organizational
environments.

### 8. [Distributed quantum computing with black-box subroutines](http://arxiv.org/pdf/2505.14519v1)

Authors: X. Xu, Y. -D. Liu, S. Shi, Y. -J. Wang, D. -S. Wang

In this work, we propose a general protocol for distributed quantum computing
that accommodates arbitrary unknown subroutines. It can be applied to scale up
quantum computing through multi-chip interconnection, as well as to tasks such
as estimating unknown parameters or processes for circuit depth reduction and
constructing secure quantum cryptographic protocols. Our protocol builds upon a
few techniques we develop, such as the oblivious quantum teleportation and
control, which can circumvent quantum no-go theorems on the manipulation of
unknown objects. Furthermore, we demonstrate that this protocol can be
physically implemented using currently available quantum computing platforms.
These results suggest that our framework could provide a foundation for
developing more advanced quantum algorithms and protocols in the future.

### Digital Libraries

### 1. [From Metadata to Storytelling: A Framework For 3D Cultural Heritage Visualization on RDF Data](http://arxiv.org/pdf/2505.14328v1)

Authors: Sebastian Barzaghi, Simona Colitti, Arianna Moretti, Giulia Renda

This paper introduces a pipeline for integrating semantic metadata, 3D
models, and storytelling, enhancing cultural heritage digitization. Using the
Aldrovandi Digital Twin case study, it outlines a reusable workflow combining
RDF-driven narratives and data visualization for creating interactive
experiences to facilitate access to cultural heritage.

### 2. [Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information](http://arxiv.org/pdf/2505.14149v1)

Authors: Chengzhi Zhang, Xinyi Yan, Lei Zhao, Yingyi Zhang

The exponential increase in academic papers has significantly increased the
time required for researchers to access relevant literature. Keyphrase
Extraction (KPE) offers a solution to this situation by enabling researchers to
efficiently retrieve relevant literature. The current study on KPE from
academic articles aims to improve the performance of extraction models through
innovative approaches using Title and Abstract as input corpora. However, the
semantic richness of keywords is significantly constrained by the length of the
abstract. While full-text-based KPE can address this issue, it simultaneously
introduces noise, which significantly diminishes KPE performance. To address
this issue, this paper utilized the structural features and section texts
obtained from the section structure information of academic articles to extract
keyphrase from academic papers. The approach consists of two main parts: (1)
exploring the effect of seven structural features on KPE models, and (2)
integrating the extraction results from all section texts used as input corpora
for KPE models via a keyphrase integration algorithm to obtain the keyphrase
integration result. Furthermore, this paper also examined the effect of the
classification quality of section structure on the KPE performance. The results
show that incorporating structural features improves KPE performance, though
different features have varying effects on model efficacy. The keyphrase
integration approach yields the best performance, and the classification
quality of section structure can affect KPE performance. These findings
indicate that using the section structure information of academic articles
contributes to effective KPE from academic articles. The code and dataset
supporting this study are available at https://github.com/yan-xinyi/SSB_KPE.

### Discrete Mathematics

### 1. [On the size of the neighborhoods of a word](http://arxiv.org/pdf/2505.13796v1)

Authors: Cedric Chauve, Louxin Zhang

The d-neighborhood of a word W in the Levenshtein distance is the set of all
words at distance at most d from W. Generating the neighborhood of a word W, or
related sets of words such as the condensed neighborhood or the super-condensed
neighborhood has applications in the design of approximate pattern matching
algorithms. It follows that bounds on the maximum size of the neighborhood of
words of a given length can be used in the complexity analysis of such
approximate pattern matching algorithms. In this note, we present exact
formulas for the size of the condensed and super condensed neighborhoods of a
unary word, a novel upper bound for the maximum size of the condensed
neighborhood of an arbitrary word of a given length, and we prove a conjectured
upper bound again for the maximum size of the condensed neighborhood of an
arbitrary word of a given length.

### 2. [On near optimal colorable graphs](http://arxiv.org/pdf/2505.13932v2)

Authors: C. U. Angeliya, Arnab Char, T. Karthick

A class of graphs $\cal G$ is said to be \emph{near optimal colorable} if
there exists a constant $c\in \mathbb{N}$ such that every graph $G\in \cal G$
satisfies $\chi(G) \leq \max\{c, \omega(G)\}$, where $\chi(G)$ and $\omega(G)$
respectively denote the chromatic number and clique number of $G$. The class of
near optimal colorable graphs is an important subclass of the class of
$\chi$-bounded graphs which is well-studied in the literature. In this paper,
we show that the class of ($F, K_4-e$)-free graphs is near optimal colorable,
where $F\in \{P_1+2P_2,2P_1+P_3,3P_1+P_2\}$. Furthermore, using these results
with some earlier known results, we also provide an alternate proof to the fact
that the \textsc{Chromatic Number} problem for the class of ($F, K_4-e$)-free
graphs is solvable in polynomial time, where $F\in
\{P_1+2P_2,2P_1+P_3,3P_1+P_2\}$.

### 3. [Path Contraction Faster than $2^n$](http://arxiv.org/pdf/2505.13996v1)

Authors: Akanksha Agrawal, Fedor V. Fomin, Daniel Lokshtanov, Saket Saurabh, Prafullkumar Tale

A graph $G$ is contractible to a graph $H$ if there is a set $X \subseteq
E(G)$, such that $G/X$ is isomorphic to $H$. Here, $G/X$ is the graph obtained
from $G$ by contracting all the edges in $X$. For a family of graphs $\cal F$,
the $\mathcal{F}$-\textsc{Contraction} problem takes as input a graph $G$ on
$n$ vertices, and the objective is to output the largest integer $t$, such that
$G$ is contractible to a graph $H \in {\cal F}$, where $|V(H)|=t$. When $\cal
F$ is the family of paths, then the corresponding
$\mathcal{F}$-\textsc{Contraction} problem is called \textsc{Path Contraction}.
The problem \textsc{Path Contraction} admits a simple algorithm running in time
$2^{n}\cdot n^{\mathcal{O}(1)}$. In spite of the deceptive simplicity of the
problem, beating the $2^{n}\cdot n^{\mathcal{O}(1)}$ bound for \textsc{Path
Contraction} seems quite challenging. In this paper, we design an exact
exponential time algorithm for \textsc{Path Contraction} that runs in time
$1.99987^n\cdot n^{\mathcal{O}(1)}$. We also define a problem called
\textsc{$3$-Disjoint Connected Subgraphs}, and design an algorithm for it that
runs in time $1.88^n\cdot n^{\mathcal{O}(1)}$. The above algorithm is used as a
sub-routine in our algorithm for {\sc Path Contraction}

### 4. [Graphon Mixtures](http://arxiv.org/pdf/2505.13864v1)

Authors: Sevvandi Kandanaarachchi, Cheng Soon Ong

Social networks have a small number of large hubs, and a large number of
small dense communities. We propose a generative model that captures both hub
and dense structures. Based on recent results about graphons on line graphs,
our model is a graphon mixture, enabling us to generate sequences of graphs
where each graph is a combination of sparse and dense graphs. We propose a new
condition on sparse graphs (the max-degree), which enables us to identify hubs.
We show theoretically that we can estimate the normalized degree of the hubs,
as well as estimate the graphon corresponding to sparse components of graph
mixtures. We illustrate our approach on synthetic data, citation graphs, and
social networks, showing the benefits of explicitly modeling sparse graphs.

### 5. [A composition theory for upward planar orders](http://arxiv.org/pdf/2505.13865v2)

Authors: Xue Dong, Xuexing Lu, Yu Ye

An upward planar order on an acyclic directed graph $G$ is a special linear
extension of the edge poset of $G$ that satisfies the nesting condition. This
order was introduced to combinatorially characterize upward plane graphs and
progressive plane graphs (commonly known as plane string diagrams). In this
paper, motivated by the theory of graphical calculus for monoidal categories,
we establish a composition theory for upward planar orders. The main result is
that the composition of upward planar orders is an upward planar order. This
theory provides a practical method to calculate the upward planar order of a
progressive plane graph or an upward plane graph.

### 6. [An asymptotic rigidity property from the realizability of chirotope extensions](http://arxiv.org/pdf/2505.14189v1)

Authors: Xavier Goaoc, Arnau Padrol

Let $P$ be a finite full-dimensional point configuration in $\mathbb{R}^d$.
We show that if a point configuration $Q$ has the property that all finite
chirotopes realizable by adding (generic) points to $P$ are also realizable by
adding points to $Q$, then $P$ and $Q$ are equal up to a direct affine
transform. We also show that for any point configuration $P$ and any
$\varepsilon>0$, there is a finite, (generic) extension $\widehat P$ of $P$
with the following property: if another realization $Q$ of the chirotope of $P$
can be extended so as to realize the chirotope of $\widehat P$, then there
exists a direct affine transform that maps each point of $Q$ within distance
$\varepsilon$ of the corresponding point of $P$.

### 7. [Better Neural Network Expressivity: Subdividing the Simplex](http://arxiv.org/pdf/2505.14338v1)

Authors: Egor Bakaev, Florestan Brunck, Christoph Hertrich, Jack Stade, Amir Yehudayoff

This work studies the expressivity of ReLU neural networks with a focus on
their depth. A sequence of previous works showed that $\lceil \log_2(n+1)
\rceil$ hidden layers are sufficient to compute all continuous piecewise linear
(CPWL) functions on $\mathbb{R}^n$. Hertrich, Basu, Di Summa, and Skutella
(NeurIPS'21) conjectured that this result is optimal in the sense that there
are CPWL functions on $\mathbb{R}^n$, like the maximum function, that require
this depth. We disprove the conjecture and show that
$\lceil\log_3(n-1)\rceil+1$ hidden layers are sufficient to compute all CPWL
functions on $\mathbb{R}^n$.
  A key step in the proof is that ReLU neural networks with two hidden layers
can exactly represent the maximum function of five inputs. More generally, we
show that $\lceil\log_3(n-2)\rceil+1$ hidden layers are sufficient to compute
the maximum of $n\geq 4$ numbers. Our constructions almost match the
$\lceil\log_3(n)\rceil$ lower bound of Averkov, Hojny, and Merkert (ICLR'25) in
the special case of ReLU networks with weights that are decimal fractions. The
constructions have a geometric interpretation via polyhedral subdivisions of
the simplex into ``easier'' polytopes.

### Data Structures and Algorithms

### 1. [A Single Exponential-Time FPT Algorithm for Cactus Contraction](http://arxiv.org/pdf/2505.14018v1)

Authors: R. Krithika, Pranabendu Misra, Prafullkumar Tale

For a collection $\mathcal{F}$ of graphs, the
$\mathcal{F}$-\textsc{Contraction} problem takes a graph $G$ and an integer $k$
as input and decides if $G$ can be modified to some graph in $\mathcal{F}$
using at most $k$ edge contractions. The $\mathcal{F}$-\textsc{Contraction}
problem is \NP-Complete for several graph classes $\mathcal{F}$. Heggerners et
al. [Algorithmica, 2014] initiated the study of
$\mathcal{F}$-\textsc{Contraction} in the realm of parameterized complexity.
They showed that it is \FPT\ if $\mathcal{F}$ is the set of all trees or the
set of all paths. In this paper, we study $\mathcal{F}$-\textsc{Contraction}
where $\mathcal{F}$ is the set of all cactus graphs and show that we can solve
it in $2^{\calO(k)} \cdot |V(G)|^{\OO(1)}$ time.

### 2. [Exploring Temporal Graphs with Frequent and Regular Edges](http://arxiv.org/pdf/2505.14046v1)

Authors: Duncan Adamson

Temporal graphs are a class of graphs defined by a constant set of vertices
and a changing set of edges, each of which is known as a timestep. These graphs
are well motivated in modelling real-world networks, where connections may
change over time. One such example, itself the primary motivation for this
paper, are public transport networks, where vertices represent stops and edges
the connections available at some given time. Exploration problems are one of
the most studied problems for temporal graphs, asking if an agent starting at
some given vertex $v$ can visit every vertex in the graph.
  In this paper, we study two primary classes of temporal graphs. First, we
study temporal graphs with \emph{frequent edges}, temporal graphs where each
edge $e$ is active at least once every $f_e$ timesteps, called the frequency of
the edge. Second, temporal graphs with \emph{regular edges}, graphs where each
edge $e$ is active at any timestep $t$ where $t \equiv s_e \bmod r_e$, with
$s_e$ being the start time of the edge, and $r_e$ the regularity.
  We show that graphs with frequent edges can be explored in $O(F n)$
timesteps, where $F = \max_{e \in E} f_e$, and that graphs with regular edges
can be explored in $O(R n)$ timesteps, where $R = \max_{e \in E} r_e$. We
provide additional results for \emph{public transport graphs}, temporal graphs
formed by the union of several routes, corresponding to the schedules of some
modes of transit, for \emph{sequential connection graphs}, temporal graphs in
which each vertex has a single active in-edge per timestep, iterating over the
set of edges in some order, and for \emph{broadcast networks}, a representation
of communication within distributed networks where each vertex broadcasts a
message either to all vertices, or none at each timestep.

### 3. [Simple and Optimal Algorithms for Heavy Hitters and Frequency Moments in Distributed Models](http://arxiv.org/pdf/2505.14250v1)

Authors: Zengfeng Huang, Zhongzheng Xiong, Xiaoyi Zhu, Zhewei Wei

We consider the problems of distributed heavy hitters and frequency moments
in both the coordinator model and the distributed tracking model (also known as
the distributed functional monitoring model). We present simple and optimal (up
to logarithmic factors) algorithms for $\ell_p$ heavy hitters and $F_p$
estimation ($p \geq 2$) in these distributed models.
  For $\ell_p$ heavy hitters in the coordinator model, our algorithm requires
only one round and uses $\tilde{O}(k^{p-1}/\eps^p)$ bits of communication. For
$p > 2$, this is the first near-optimal result. By combining our algorithm with
the standard recursive sketching technique, we obtain a near-optimal two-round
algorithm for $F_p$ in the coordinator model, matching a significant result
from recent work by Esfandiari et al.\ (STOC 2024). Our algorithm and analysis
are much simpler and have better costs with respect to logarithmic factors.
Furthermore, our technique provides a one-round algorithm for $F_p$, which is a
significant improvement over a result of Woodruff and Zhang (STOC 2012).
  Thanks to the simplicity of our heavy hitter algorithms, we manage to adapt
them to the distributed tracking model with only a $\polylog(n)$ increase in
communication. For $\ell_p$ heavy hitters, our algorithm has a communication
cost of $\tilde{O}(k^{p-1}/\eps^p)$, representing the first near-optimal
algorithm for all $p \geq 2$. By applying the recursive sketching technique, we
also provide the first near-optimal algorithm for $F_p$ in the distributed
tracking model, with a communication cost of $\tilde{O}(k^{p-1}/\eps^2)$ for
all $p \geq 2$. Even for $F_2$, our result improves upon the bounds established
by Cormode, Muthukrishnan, and Yi (SODA 2008) and Woodruff and Zhang (STOC
2012), nearly matching the existing lower bound for the first time.

### 4. [Approximate Spanning Tree Counting from Uncorrelated Edge Sets](http://arxiv.org/pdf/2505.14666v1)

Authors: Yang P. Liu, Richard Peng, Junzhao Yang

We show an $\widetilde{O}(m^{1.5} \epsilon^{-1})$ time algorithm that on a
graph with $m$ edges and $n$ vertices outputs its spanning tree count up to a
multiplicative $(1+\epsilon)$ factor with high probability, improving on the
previous best runtime of $\widetilde{O}(m + n^{1.875}\epsilon^{-7/4})$ in
sparse graphs. While previous algorithms were based on computing Schur
complements and determinantal sparsifiers, our algorithm instead repeatedly
removes sets of uncorrelated edges found using the electrical flow localization
theorem of Schild-Rao-Srivastava [SODA 2018].

### 5. [Path Contraction Faster than $2^n$](http://arxiv.org/pdf/2505.13996v1)

Authors: Akanksha Agrawal, Fedor V. Fomin, Daniel Lokshtanov, Saket Saurabh, Prafullkumar Tale

A graph $G$ is contractible to a graph $H$ if there is a set $X \subseteq
E(G)$, such that $G/X$ is isomorphic to $H$. Here, $G/X$ is the graph obtained
from $G$ by contracting all the edges in $X$. For a family of graphs $\cal F$,
the $\mathcal{F}$-\textsc{Contraction} problem takes as input a graph $G$ on
$n$ vertices, and the objective is to output the largest integer $t$, such that
$G$ is contractible to a graph $H \in {\cal F}$, where $|V(H)|=t$. When $\cal
F$ is the family of paths, then the corresponding
$\mathcal{F}$-\textsc{Contraction} problem is called \textsc{Path Contraction}.
The problem \textsc{Path Contraction} admits a simple algorithm running in time
$2^{n}\cdot n^{\mathcal{O}(1)}$. In spite of the deceptive simplicity of the
problem, beating the $2^{n}\cdot n^{\mathcal{O}(1)}$ bound for \textsc{Path
Contraction} seems quite challenging. In this paper, we design an exact
exponential time algorithm for \textsc{Path Contraction} that runs in time
$1.99987^n\cdot n^{\mathcal{O}(1)}$. We also define a problem called
\textsc{$3$-Disjoint Connected Subgraphs}, and design an algorithm for it that
runs in time $1.88^n\cdot n^{\mathcal{O}(1)}$. The above algorithm is used as a
sub-routine in our algorithm for {\sc Path Contraction}

### 6. [Linear Hashing Is Optimal](http://arxiv.org/pdf/2505.14061v1)

Authors: Michael Jaber, Vinayak M. Kumar, David Zuckerman

We prove that hashing $n$ balls into $n$ bins via a random matrix over
$\mathbf{F}_2$ yields expected maximum load $O(\log n / \log \log n)$. This
matches the expected maximum load of a fully random function and resolves an
open question posed by Alon, Dietzfelbinger, Miltersen, Petrank, and Tardos
(STOC '97, JACM '99). More generally, we show that the maximum load exceeds
$r\cdot\log n/\log\log n$ with probability at most $O(1/r^2)$.

### 7. [A Private Approximation of the 2nd-Moment Matrix of Any Subsamplable Input](http://arxiv.org/pdf/2505.14251v1)

Authors: Bar Mahpud, Or Sheffet

We study the problem of differentially private second moment estimation and
present a new algorithm that achieve strong privacy-utility trade-offs even for
worst-case inputs under subsamplability assumptions on the data. We call an
input $(m,\alpha,\beta)$-subsamplable if a random subsample of size $m$ (or
larger) preserves w.p $\geq 1-\beta$ the spectral structure of the original
second moment matrix up to a multiplicative factor of $1\pm \alpha$. Building
upon subsamplability, we give a recursive algorithmic framework similar to
Kamath et al 2019, that abides zero-Concentrated Differential Privacy (zCDP)
while preserving w.h.p. the accuracy of the second moment estimation upto an
arbitrary factor of $(1\pm\gamma)$. We then show how to apply our algorithm to
approximate the second moment matrix of a distribution $\mathcal{D}$, even when
a noticeable fraction of the input are outliers.

### 8. [Credible Sets of Phylogenetic Tree Topology Distributions](http://arxiv.org/pdf/2505.14532v1)

Authors: Jonathan Klawitter, Alexei J. Drummond

Credible intervals and credible sets, such as highest posterior density (HPD)
intervals, form an integral statistical tool in Bayesian phylogenetics, both
for phylogenetic analyses and for development. Readily available for continuous
parameters such as base frequencies and clock rates, the vast and complex space
of tree topologies poses significant challenges for defining analogous credible
sets. Traditional frequency-based approaches are inadequate for diffuse
posteriors where sampled trees are often unique. To address this, we introduce
novel and efficient methods for estimating the credible level of individual
tree topologies using tractable tree distributions, specifically Conditional
Clade Distributions (CCDs). Furthermore, we propose a new concept called
$\alpha$ credible CCD, which encapsulates a CCD whose trees collectively make
up $\alpha$ probability. We present algorithms to compute these credible CCDs
efficiently and to determine credible levels of tree topologies as well as of
subtrees. We evaluate the accuracy of these credible set methods leveraging
simulated and real datasets. Furthermore, to demonstrate the utility of our
methods, we use well-calibrated simulation studies to evaluate the performance
of different CCD models. In particular, we show how the credible set methods
can be used to conduct rank-uniformity validation and produce Empirical
Cumulative Distribution Function (ECDF) plots, supplementing standard coverage
analyses for continuous parameters.

### Emerging Technologies

### 1. [6G communications through sub-Terahertz CMOS power amplifiers: Design challenges and trends](http://arxiv.org/pdf/2505.13801v1)

Authors: Jun Yan Lee, Duo Wu, Xuanrui Guo, Jian Ding Tan, Teh Jia Yew, Zi Neng Ng, Mohammad Arif Sobhan Bhuiyan, Mahdi H. Miraz

The fifth-generation (5G) network faces limitations in supporting emerging
applications, such as artificial intelligence (AI), virtual reality (VR) and
digital twins. To overcome these confines, sub-Terahertz (sub-THz) and
Terahertz (THz) technologies are considered to be key enablers of effective 6G
wireless communications, offering higher transmission speeds, longer range and
wider bandwidth. Achieving these capabilities requires careful engineering of
6G transceivers, with a focus on efficient power amplifiers (PAs) in the
front-end, which play a critical role in effectively amplifying and
transmitting signals over long distances. Complimentary
metal-oxidesemiconductor (CMOS) technology-based PA in sub-THz suffers severe
parasitic and limited maximum frequency, however, this has eventually been
solved by different design architectures and scaling down of CMOS technology to
break through the frequency limitations. In this article, we reviewed the
potentials and capabilities of CMOS technology for designing 6G hardware,
identified the state-of-art PA designs in the sub-THz band and then examined as
well as compared the designs to identify the suitable design strategies for
better performance. The circuit optimisation techniques, such as coupled-line,
passive gain boosting method, zero-degree power splitting, load-pull matching,
diode and capacitor linearisation for better gain, saturated output power and
power added efficiency, are considered for the PA design architectures at
different sub-THz bands. Furthermore, these methods are summarised and
discussed with their advantages and disadvantages in lieu with their
performances. The PA design trends, challenges and future perspectives are also
presented and discussed. Therefore, this comprehensive review article will
serve as a comparative study and reference for future PA designs for radio
frequency integrated circuits (RFIC).

### 2. [Optimizing Binary and Ternary Neural Network Inference on RRAM Crossbars using CIM-Explorer](http://arxiv.org/pdf/2505.14303v1)

Authors: Rebecca Pelke, José Cubero-Cascante, Nils Bosbach, Niklas Degener, Florian Idrizi, Lennart M. Reimann, Jan Moritz Joseph, Rainer Leupers

Using Resistive Random Access Memory (RRAM) crossbars in Computing-in-Memory
(CIM) architectures offers a promising solution to overcome the von Neumann
bottleneck. Due to non-idealities like cell variability, RRAM crossbars are
often operated in binary mode, utilizing only two states: Low Resistive State
(LRS) and High Resistive State (HRS). Binary Neural Networks (BNNs) and Ternary
Neural Networks (TNNs) are well-suited for this hardware due to their efficient
mapping. Existing software projects for RRAM-based CIM typically focus on only
one aspect: compilation, simulation, or Design Space Exploration (DSE).
Moreover, they often rely on classical 8 bit quantization. To address these
limitations, we introduce CIM-Explorer, a modular toolkit for optimizing BNN
and TNN inference on RRAM crossbars. CIM-Explorer includes an end-to-end
compiler stack, multiple mapping options, and simulators, enabling a DSE flow
for accuracy estimation across different crossbar parameters and mappings.
CIM-Explorer can accompany the entire design process, from early accuracy
estimation for specific crossbar parameters, to selecting an appropriate
mapping, and compiling BNNs and TNNs for a finalized crossbar chip. In DSE case
studies, we demonstrate the expected accuracy for various mappings and crossbar
parameters. CIM-Explorer can be found on GitHub.

### 3. [Design and Evaluation of a Microservices Cloud Framework for Online Travel Platforms](http://arxiv.org/pdf/2505.14508v1)

Authors: Biman Barua, M. Shamim Kaiser

Handling online travel agents globally requires efficient and flexible
software solution architectures. When it needs to handle thousands of agents
and billions of clients data globally. Microservices architecture is used to
break down a large program into numerous, smaller services which can run
individually and perform individual tasks. This paper analyses and integrates a
unique Microservices Cloud Framework designed to support Online Travel
Platforms (MCF-OTP). MCF-OTPs main goal is to increase the performance,
flexibility, and maintenance of online travel platforms via cloud computing and
microservice technologies. Large-scale travel apps, including managing numerous
data sources, dealing with traffic peaks, and providing fault tolerance, can be
addressed by the suggested framework. The framework increases good
interpretation between flawless data synchronization, microservices, and
dynamic scaling based on demand technology. An organization framework that
optimizes service borders and minimizes inter-service dependencies is
recommended. Thus, this can result in elevated development adaptability. In
this research, the principal goal is to evaluate MCF-OTPs efficiency using the
indicators of fault tolerance and response time. It is indicated by the
findings that the MCF-OTP structure excels traditional monolithic designs in
terms of dependability and scalability, managing traffic spikes seamlessly and
decreasing downtime. The cost-effective analysis helps ascertain the net gain
attained by the startup fees and the ongoing operational costs. The cloud-based
environment is used to reduce the fracture cost which also helps to increase
the efficiency of resource allocation, according to the research.

### 4. [Towards Verifiability of Total Value Locked (TVL) in Decentralized Finance](http://arxiv.org/pdf/2505.14565v1)

Authors: Pietro Saggese, Michael Fröwis, Stefan Kitzler, Bernhard Haslhofer, Raphael Auer

Total Value Locked (TVL) aims to measure the aggregate value of cryptoassets
deposited in Decentralized Finance (DeFi) protocols. Although blockchain data
is public, the way TVL is computed is not well understood. In practice, its
calculation on major TVL aggregators relies on self-reports from community
members and lacks standardization, making it difficult to verify published
figures independently. We thus conduct a systematic study on 939 DeFi projects
deployed in Ethereum. We study the methodologies used to compute TVL, examine
factors hindering verifiability, and ultimately propose standardization
attempts in the field. We find that 10.5% of the protocols rely on external
servers; 68 methods alternative to standard balance queries exist, although
their use decreased over time; and 240 equal balance queries are repeated on
multiple protocols. These findings indicate limits to verifiability and
transparency. We thus introduce ``verifiable Total Value Locked'' (vTVL), a
metric measuring the TVL that can be verified relying solely on on-chain data
and standard balance queries. A case study on 400 protocols shows that our
estimations align with published figures for 46.5% of protocols. Informed by
these findings, we discuss design guidelines that could facilitate a more
verifiable, standardized, and explainable TVL computation.

### 5. [Upgrading Democracies with Fairer Voting Methods](http://arxiv.org/pdf/2505.14349v1)

Authors: Evangelos Pournaras, Srijoni Majumdar, Thomas Wellings, Joshua C. Yang, Fatemeh B. Heravan, Regula Hänggli Fricker, Dirk Helbing

Voting methods are instrumental design element of democracies. Citizens use
them to express and aggregate their preferences to reach a collective decision.
However, voting outcomes can be as sensitive to voting rules as they are to
people's voting choices. Despite the significance and inter-disciplinary
scientific progress on voting methods, several democracies keep relying on
outdated voting methods that do not fit modern, pluralistic societies well,
while lacking social innovation. Here, we demonstrate how one can upgrade
real-world democracies, namely by using alternative preferential voting methods
such as cumulative voting and the method of equal shares designed for a
proportional representation of voters' preferences. By rigorously assessing a
new participatory budgeting approach applied in the city of Aarau, Switzerland,
we unravel the striking voting outcomes of fair voting methods: more winning
projects with the same budget and broader geographic and preference
representation of citizens by the elected projects, in particular for voters
who used to be under-represented, while promoting novel project ideas. We
provide profound causal evidence showing that citizens prefer proportional
voting methods, which possess strong legitimacy without the need of very
technical specialized explanations. We also reveal strong underlying democratic
values exhibited by citizens who support fair voting methods such as altruism
and compromise. These findings come with a global momentum to unleash a new and
long-awaited participation blueprint of how to upgrade democracies.

### Formal Languages and Automata Theory

### 1. [On Quantum Context-Free Grammars](http://arxiv.org/pdf/2505.13937v1)

Authors: Merina Aruja, Lisa Mathew, Jayakrishna Vijayakumar

Quantum computing is a relatively new field of computing, which utilises the
fundamental concepts of quantum mechanics to process data. The seminal paper of
Moore et al. [2000] introduced quantum grammars wherein a set of amplitudes was
attached to each production. However they did not study the final probability
of the derived word. Aruja et al. [2025] considered conditions for the
well-formedness of quantum context-free grammars (QCFGs), in order to ensure
that the probabilty of the derived word does not exceed one. In this paper we
propose certain necessary and sufficient conditions (also known as unitary
conditions) for well-formedness of QCFGs

### 2. [Minimal History-Deterministic Co-Buchi Automata: Congruences and Passive Learning](http://arxiv.org/pdf/2505.14304v2)

Authors: Christof Löding, Igor Walukiewicz

Abu Radi and Kupferman (2019) demonstrated the efficient minimization of
history-deterministic (transition-based) co-B\"uchi automata, building on the
results of Kuperberg and Skrzypczak (2015). We give a congruence-based
description of these minimal automata, and a self-contained proof of its
correctness. We use this description based on congruences to create a passive
learning algorithm that can learn minimal history-deterministic co-B\"uchi
automata from a set of labeled example words. The algorithm runs in polynomial
time on a given set of examples, and there is a characteristic set of examples
of polynomial size for each minimal history-deterministic co-B\"uchi automaton.

### Graphics

### 1. [A Remeshing Method via Adaptive Multiple Original-Facet-Clipping and Centroidal Voronoi Tessellation](http://arxiv.org/pdf/2505.14306v1)

Authors: Yue Fei, Jingjing Liu, Yuyou Yao, Wenming Wu, Liping Zheng

CVT (Centroidal Voronoi Tessellation)-based remeshing optimizes mesh quality
by leveraging the Voronoi-Delaunay framework to optimize vertex distribution
and produce uniformly distributed vertices with regular triangles. Current
CVT-based approaches can be classified into two categories: (1) exact methods
(e.g., Geodesic CVT, Restricted Voronoi Diagrams) that ensure high quality but
require significant computation; and (2) approximate methods that try to reduce
computational complexity yet result in fair quality. To address this trade-off,
we propose a CVT-based surface remeshing approach that achieves balanced
optimization between quality and efficiency through multiple clipping times of
3D Centroidal Voronoi cells with curvature-adaptive original surface facets.
The core idea of the method is that we adaptively adjust the number of clipping
times according to local curvature, and use the angular relationship between
the normal vectors of neighboring facets to represent the magnitude of local
curvature. Experimental results demonstrate the effectiveness of our method.

### 2. [Large-Scale Multi-Character Interaction Synthesis](http://arxiv.org/pdf/2505.14087v1)

Authors: Ziyi Chang, He Wang, George Alex Koulieris, Hubert P. H. Shum

Generating large-scale multi-character interactions is a challenging and
important task in character animation. Multi-character interactions involve not
only natural interactive motions but also characters coordinated with each
other for transition. For example, a dance scenario involves characters dancing
with partners and also characters coordinated to new partners based on spatial
and temporal observations. We term such transitions as coordinated interactions
and decompose them into interaction synthesis and transition planning. Previous
methods of single-character animation do not consider interactions that are
critical for multiple characters. Deep-learning-based interaction synthesis
usually focuses on two characters and does not consider transition planning.
Optimization-based interaction synthesis relies on manually designing objective
functions that may not generalize well. While crowd simulation involves more
characters, their interactions are sparse and passive. We identify two
challenges to multi-character interaction synthesis, including the lack of data
and the planning of transitions among close and dense interactions. Existing
datasets either do not have multiple characters or do not have close and dense
interactions. The planning of transitions for multi-character close and dense
interactions needs both spatial and temporal considerations. We propose a
conditional generative pipeline comprising a coordinatable multi-character
interaction space for interaction synthesis and a transition planning network
for coordinations. Our experiments demonstrate the effectiveness of our
proposed pipeline for multicharacter interaction synthesis and the applications
facilitated by our method show the scalability and transferability.

### 3. [MatchDance: Collaborative Mamba-Transformer Architecture Matching for High-Quality 3D Dance Synthesis](http://arxiv.org/pdf/2505.14222v2)

Authors: Kaixing Yang, Xulong Tang, Yuxuan Hu, Jiahao Yang, Hongyan Liu, Qinnan Zhang, Jun He, Zhaoxin Fan

Music-to-dance generation represents a challenging yet pivotal task at the
intersection of choreography, virtual reality, and creative content generation.
Despite its significance, existing methods face substantial limitation in
achieving choreographic consistency. To address the challenge, we propose
MatchDance, a novel framework for music-to-dance generation that constructs a
latent representation to enhance choreographic consistency. MatchDance employs
a two-stage design: (1) a Kinematic-Dynamic-based Quantization Stage (KDQS),
which encodes dance motions into a latent representation by Finite Scalar
Quantization (FSQ) with kinematic-dynamic constraints and reconstructs them
with high fidelity, and (2) a Hybrid Music-to-Dance Generation Stage(HMDGS),
which uses a Mamba-Transformer hybrid architecture to map music into the latent
representation, followed by the KDQS decoder to generate 3D dance motions.
Additionally, a music-dance retrieval framework and comprehensive metrics are
introduced for evaluation. Extensive experiments on the FineDance dataset
demonstrate state-of-the-art performance. Code will be released upon
acceptance.

### Computer Science and Game Theory

### 1. [Online Resource Sharing: Better Robust Guarantees via Randomized Strategies](http://arxiv.org/pdf/2505.13824v1)

Authors: David X. Lin, Daniel Hall, Giannis Fikioris, Siddhartha Banerjee, Éva Tardos

We study the problem of fair online resource allocation via non-monetary
mechanisms, where multiple agents repeatedly share a resource without monetary
transfers. Previous work has shown that every agent can guarantee $1/2$ of
their ideal utility (the highest achievable utility given their fair share of
resources) robustly, i.e., under arbitrary behavior by the other agents. While
this $1/2$-robustness guarantee has now been established under very different
mechanisms, including pseudo-markets and dynamic max-min allocation, improving
on it has appeared difficult.
  In this work, we obtain the first significant improvement on the robustness
of online resource sharing. In more detail, we consider the widely-studied
repeated first-price auction with artificial currencies. Our main contribution
is to show that a simple randomized bidding strategy can guarantee each agent a
$2 - \sqrt 2 \approx 0.59$ fraction of her ideal utility, irrespective of
others' bids. Specifically, our strategy requires each agent with fair share
$\alpha$ to use a uniformly distributed bid whenever her value is in the top
$\alpha$-quantile of her value distribution. Our work almost closes the gap to
the known $1 - 1/e \approx 0.63$ hardness for robust resource sharing; we also
show that any static (i.e., budget independent) bidding policy cannot guarantee
more than a $0.6$-fraction of the ideal utility, showing our technique is
almost tight.

### 2. [A Sequence-Form Characterization and Differentiable Path-Following Computation of Normal-Form Perfect Equilibria in Extensive-Form Games](http://arxiv.org/pdf/2505.13827v1)

Authors: Yuqing Hou, Yiyin Cao, Chuangyin Dang

The sequence form, owing to its compact and holistic strategy representation,
has demonstrated significant efficiency in computing normal-form perfect
equilibria for two-player extensive-form games with perfect recall.
Nevertheless, the examination of $n$-player games remains underexplored. To
tackle this challenge, we present a sequence-form characterization of
normal-form perfect equilibria for $n$-player extensive-form games, achieved
through a class of perturbed games formulated in sequence form. Based on this
characterization, we develop a differentiable path-following method for
computing normal-form perfect equilibria and prove its convergence. This method
involves constructing an artificial logarithmic-barrier game in sequence form,
where an additional variable is incorporated to regulate the influence of
logarithmic-barrier terms to the payoff functions, as well as the transition of
the strategy space. We prove the existence of a smooth equilibrium path defined
by the artificial game, starting from an arbitrary positive realization plan
and converging to a normal-form perfect equilibrium of the original game as the
additional variable approaches zero. Furthermore, we extend Harsanyi's linear
and logarithmic tracing procedures to the sequence form and develop two
alternative methods for computing normal-form perfect equilibria. Numerical
experiments further substantiate the effectiveness and efficiency of our
methods.

### 3. [GUARD: Constructing Realistic Two-Player Matrix and Security Games for Benchmarking Game-Theoretic Algorithms](http://arxiv.org/pdf/2505.14547v1)

Authors: Noah Krever, Jakub Černý, Moïse Blanchard, Christian Kroer

Game-theoretic algorithms are commonly benchmarked on recreational games,
classical constructs from economic theory such as congestion and dispersion
games, or entirely random game instances. While the past two decades have seen
the rise of security games -- grounded in real-world scenarios like patrolling
and infrastructure protection -- their practical evaluation has been hindered
by limited access to the datasets used to generate them. In particular,
although the structural components of these games (e.g., patrol paths derived
from maps) can be replicated, the critical data defining target values --
central to utility modeling -- remain inaccessible. In this paper, we introduce
a flexible framework that leverages open-access datasets to generate realistic
matrix and security game instances. These include animal movement data for
modeling anti-poaching scenarios and demographic and infrastructure data for
infrastructure protection. Our framework allows users to customize utility
functions and game parameters, while also offering a suite of preconfigured
instances. We provide theoretical results highlighting the degeneracy and
limitations of benchmarking on random games, and empirically compare our
generated games against random baselines across a variety of standard
algorithms for computing Nash and Stackelberg equilibria, including linear
programming, incremental strategy generation, and self-play with no-regret
learners.

### 4. [Trustworthy Reputation Games and Applications to Proof-of-Reputation Blockchains](http://arxiv.org/pdf/2505.14551v1)

Authors: Petros Drineas, Rohit Nema, Rafail Ostrovsky, Vassilis Zikas

Reputation systems play an essential role in the Internet era, as they enable
people to decide whom to trust, by collecting and aggregating data about users'
behavior. Recently, several works proposed the use of reputation for the design
and scalability improvement of decentralized (blockchain) ledgers; however,
such systems are prone to manipulation and to our knowledge no game-theoretic
treatment exists that can support their economic robustness.
  In this work we put forth a new model for the design of what we call, {\em
trustworthy reputation systems}. Concretely, we describe a class of games,
which we term {\em trustworthy reputation games}, that enable a set of users to
report a function of their beliefs about the trustworthiness of each server in
a set -- i.e., their estimate of the probability that this server will behave
according to its specified strategy -- in a way that satisfies the following
properties:
  1. It is $(\epsilon$-)best response for any rational user in the game to play
a prescribed (truthful) strategy according to their true belief.
  2. Assuming that the users' beliefs are not too far from the {\em true}
trustworthiness of the servers, playing the above ($\epsilon-$)Nash equilibrium
allows anyone who observes the users' strategies to estimate the relative
trustworthiness of any two servers.
  Our utilities and decoding function build on a connection between the well
known PageRank algorithm and the problem of trustworthiness discovery, which
can be of independent interest. Finally, we show how the above games are
motivated by and can be leveraged in proof-of-reputation (PoR) blockchains.

### Human-Computer Interaction

### 1. [Human Authenticity and Flourishing in an AI-Driven World: Edmund's Journey and the Call for Mindfulness](http://arxiv.org/pdf/2505.13953v1)

Authors: Sebastian Zepf, Mark Colley

Humans have always dreamed of possessing superpowers, and the rapid
development of AI-based features promises to bring these dreams (closer) to
reality. However, these advancements come with significant risks. This paper
advocates for challenging existing methods and approaches in design and
evaluation for more responsible AI. We stimulate reflection through a
futuristic user journey illustrating the AI-driven life of Edmund in 2035.
Subsequently, we discuss four AI-based superpowers: extended perception,
cognitive offloading, externalized memory, and enhanced presence. We then
discuss implications for HCI and AI, emphasizing the need for preserving
intrinsic human superpowers, identifying meaningful use cases for AI, and
evaluating AI's impact on human abilities. This paper advocates for responsible
and reflective AI integration and proposes a pathway towards the idea of a
Human Flourishing Benchmark.

### 2. [Reading.help: Supporting EFL Readers with Proactive and On-Demand Explanation of English Grammar and Semantics](http://arxiv.org/pdf/2505.14031v1)

Authors: Sunghyo Chung, Hyeon Jeon, Sungbok Shin, Md Naimul Hoque

A large portion of texts in the world is written in English, but readers who
see English as a Foreign Language (EFL) often struggle to read texts written in
English accurately and swiftly. In many countries, EFL readers seek help from
professional teachers and mentors, which is limited and costly. In this paper,
we explore how an intelligent reading tool can assist EFL readers. To support
our research agenda, we conducted a case study with EFL readers in South Korea.
We at first developed an LLM-based reading tool based on prior literature. We
then revised the tool based on the feedback from a study with 15 South Korean
EFL readers. The final tool, named Reading.help, helps EFL readers comprehend
complex sentences and paragraphs with on-demand and proactive explanations. We
finally evaluated the tool with 5 EFL readers and 2 EFL education
professionals. Our findings suggest Reading.help could potentially help EFL
readers self-learn english when they do not have access to any external
support.

### 3. [The Virtual Reality Koinos Method: Analyzing Virtual Reality Collaboration from the perspective of communication models](http://arxiv.org/pdf/2505.14078v1)

Authors: Eloise Minder, Sylvain Fleury, Solène Neyret, Jean-Rémy Chardonnet

Understanding which factors could influence co-presence in Virtual Reality
could help develop more qualitative social interactions, or social interactions
that generate similar sensations, emotions and feelings than the ones generated
during Face-to-Face interactions. Co-presence is studied since the beginning of
Virtual Reality (VR); though, no consensus is identified on what factors could
influence it, except the consensus on the definition of "being there together"
inside the Virtual Environment. In this paper, we introduce the Koinos method
to explain social interactions in VR through communication models, (i)
theoretically, and (ii) on two VR experiments that change the virtual partner
social and physical representations. These analyses lead us to propose an
equation to predict and help manage the sense of co-presence in VR.

### 4. [Human and Machine as Seen at the Co-Creation Age: A Co-Word Analysis in Human Machine Co-creation (2014-2024)](http://arxiv.org/pdf/2505.14363v1)

Authors: Mengyao Guo, Jinda Han, Ze Gao, Yuan Zhuang, Xingting Wu

This paper explores the evolving landscape of human-machine co-creation,
focusing on its development in the context of the ACM Conference on Human
Factors in Computing Systems (CHI) from 2014 to 2024. We employ co-word
analysis to identify emerging trends, central themes, and the intellectual
trajectory of this field. The study highlights the shift from viewing machines
as mere tools to recognizing them as collaborative partners in creative
processes. By understanding these dynamics, we aim to provide insights into the
implications of this paradigm shift for creativity, innovation, and societal
impact, ultimately fostering a more inclusive and effective approach to
human-machine interaction in various domains.

### 5. [What Does Success Look Like? Catalyzing Meeting Intentionality with AI-Assisted Prospective Reflection](http://arxiv.org/pdf/2505.14370v1)

Authors: Ava Elizabeth Scott, Lev Tankelevitch, Payod Panda, Rishi Vanukuru, Xinyue Chen, Sean Rintel

Despite decades of HCI and Meeting Science research, complaints about
ineffective meetings are still pervasive. We argue that meeting technologies
lack support for prospective reflection, that is, thinking about why a meeting
is needed and what might happen. To explore this, we designed a Meeting Purpose
Assistant (MPA) technology probe to coach users to articulate their meeting's
purpose and challenges, and act accordingly. The MPA used Generative AI to
support personalized and actionable prospective reflection across the diversity
of meeting contexts. Using a participatory prompting methodology, 18 employees
of a global technology company reflected with the MPA on upcoming meetings.
Observed impacts were: clarifying meeting purposes, challenges, and success
conditions; changing perspectives and flexibility; improving preparation and
communication; and proposing changed plans. We also identify perceived social,
temporal, and technological barriers to using the MPA. We present system and
workflow design considerations for developing AI-assisted reflection support
for meetings.

### 6. [Two Empirical Studies on Audiovisual Semiotics of Uncertainty](http://arxiv.org/pdf/2505.14379v1)

Authors: Sita Vriend, David Hägele, Daniel Weiskopf

There exists limited theoretical guidance on integrating visualization and
sonification. In this paper, we address this gap by investigating audiovisual
semiotics for uncertainty representation: joining uncertainty visualization and
sonification to combine audiovisual channels for enhancing users' perception of
uncertainty. We conducted two preregistered crowd-sourced user studies. First,
we assessed suitable audio/visual pairs. Then, we investigated audiovisual
mappings of uncertainty. Here, we use probability as it is an easily
communicated aspect of uncertainty. We analyzed the participants' preferences
and reaction times in both user studies. Additionally, we explored the
strategies employed by participants through qualitative analysis. Our results
reveal audiovisual mappings that lead to particularly strong preferences and
low reaction times. Furthermore, we found that preferred audio/visual pairs are
not necessarily suitable audiovisual mappings of uncertainty. For example,
while pitch paired with brightness was preferred as a pair, it was not well
suited as a mapping for uncertainty. We recommend audiovisual mappings of
uncertainty that lead to low reaction times and high preferences in both user
studies. This paper presents guidelines to anyone seeking to employ audiovisual
representations for uncertainty, contributing to enhancing the perception of
uncertainty.

### 7. [Sketch Interface for Teleoperation of Mobile Manipulator to Enable Intuitive and Intended Operation: A Proof of Concept](http://arxiv.org/pdf/2505.13931v2)

Authors: Yuka Iwanaga, Masayoshi Tsuchinaga, Kosei Tanada, Yuji Nakamura, Takemitsu Mori, Takashi Yamamoto

Recent advancements in robotics have underscored the need for effective
collaboration between humans and robots. Traditional interfaces often struggle
to balance robot autonomy with human oversight, limiting their practical
application in complex tasks like mobile manipulation. This study aims to
develop an intuitive interface that enables a mobile manipulator to
autonomously interpret user-provided sketches, enhancing user experience while
minimizing burden. We implemented a web-based application utilizing machine
learning algorithms to process sketches, making the interface accessible on
mobile devices for use anytime, anywhere, by anyone. In the first validation,
we examined natural sketches drawn by users for 27 selected manipulation and
navigation tasks, gaining insights into trends related to sketch instructions.
The second validation involved comparative experiments with five grasping
tasks, showing that the sketch interface reduces workload and enhances
intuitiveness compared to conventional axis control interfaces. These findings
suggest that the proposed sketch interface improves the efficiency of mobile
manipulators and opens new avenues for integrating intuitive human-robot
collaboration in various applications.

### 8. [When Bias Backfires: The Modulatory Role of Counterfactual Explanations on the Adoption of Algorithmic Bias in XAI-Supported Human Decision-Making](http://arxiv.org/pdf/2505.14377v1)

Authors: Ulrike Kuhl, Annika Bush

Although the integration of artificial intelligence (AI) into everyday tasks
improves efficiency and objectivity, it also risks transmitting bias to human
decision-making. In this study, we conducted a controlled experiment that
simulated hiring decisions to examine how biased AI recommendations - augmented
with or without counterfactual explanations - influence human judgment over
time. Participants, acting as hiring managers, completed 60 decision trials
divided into a baseline phase without AI, followed by a phase with biased (X)AI
recommendations (favoring either male or female candidates), and a final
post-interaction phase without AI. Our results indicate that the participants
followed the AI recommendations 70% of the time when the qualifications of the
given candidates were comparable. Yet, only a fraction of participants detected
the gender bias (8 out of 294). Crucially, exposure to biased AI altered
participants' inherent preferences: in the post-interaction phase,
participants' independent decisions aligned with the bias when no
counterfactual explanations were provided before, but reversed the bias when
explanations were given. Reported trust did not differ significantly across
conditions. Confidence varied throughout the study phases after exposure to
male-biased AI, indicating nuanced effects of AI bias on decision certainty.
Our findings point to the importance of calibrating XAI to avoid unintended
behavioral shifts in order to safeguard equitable decision-making and prevent
the adoption of algorithmic bias.

### 9. [How Managers Perceive AI-Assisted Conversational Training for Workplace Communication](http://arxiv.org/pdf/2505.14452v2)

Authors: Lance T. Wilhelm, Xiaohan Ding, Kirk McInnis Knutsen, Buse Carik, Eugenia H. Rho

Effective workplace communication is essential for managerial success, yet
many managers lack access to tailored and sustained training. Although
AI-assisted communication systems may offer scalable training solutions, little
is known about how managers envision the role of AI in helping them improve
their communication skills. To investigate this, we designed a conversational
role-play system, CommCoach, as a functional probe to understand how managers
anticipate using AI to practice their communication skills. Through
semi-structured interviews, participants emphasized the value of adaptive,
low-risk simulations for practicing difficult workplace conversations. They
also highlighted opportunities, including human-AI teaming, transparent and
context-aware feedback, and greater control over AI-generated personas.
AI-assisted communication training should balance personalization, structured
learning objectives, and adaptability to different user styles and contexts.
However, achieving this requires carefully navigating tensions between adaptive
and consistent AI feedback, realism and potential bias, and the open-ended
nature of AI conversations versus structured workplace discourse.

### 10. [Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning](http://arxiv.org/pdf/2505.14535v1)

Authors: Jiangrong Shen, Yulin Xie, Qi Xu, Gang Pan, Huajin Tang, Badong Chen

Multimodal spiking neural networks (SNNs) hold significant potential for
energy-efficient sensory processing but face critical challenges in modality
imbalance and temporal misalignment. Current approaches suffer from
uncoordinated convergence speeds across modalities and static fusion mechanisms
that ignore time-varying cross-modal interactions. We propose the temporal
attention-guided adaptive fusion framework for multimodal SNNs with two
synergistic innovations: 1) The Temporal Attention-guided Adaptive Fusion
(TAAF) module that dynamically assigns importance scores to fused spiking
features at each timestep, enabling hierarchical integration of temporally
heterogeneous spike-based features; 2) The temporal adaptive balanced fusion
loss that modulates learning rates per modality based on the above attention
scores, preventing dominant modalities from monopolizing optimization. The
proposed framework implements adaptive fusion, especially in the temporal
dimension, and alleviates the modality imbalance during multimodal learning,
mimicking cortical multisensory integration principles. Evaluations on CREMA-D,
AVE, and EAD datasets demonstrate state-of-the-art performance (77.55\%,
70.65\% and 97.5\%accuracy, respectively) with energy efficiency. The system
resolves temporal misalignment through learnable time-warping operations and
faster modality convergence coordination than baseline SNNs. This work
establishes a new paradigm for temporally coherent multimodal learning in
neuromorphic systems, bridging the gap between biological sensory processing
and efficient machine intelligence.

### Information Retrieval

### 1. [Benchmarking the Myopic Trap: Positional Bias in Information Retrieval](http://arxiv.org/pdf/2505.13950v1)

Authors: Ziyang Zeng, Dun Zhang, Jiacheng Li, Panxiang Zou, Yuqing Yang

This study investigates a specific form of positional bias, termed the Myopic
Trap, where retrieval models disproportionately attend to the early parts of
documents while overlooking relevant information that appears later. To
systematically quantify this phenomenon, we propose a semantics-preserving
evaluation framework that repurposes the existing NLP datasets into
position-aware retrieval benchmarks. By evaluating the SOTA models of full
retrieval pipeline, including BM25, embedding models, ColBERT-style
late-interaction models, and reranker models, we offer a broader empirical
perspective on positional bias than prior work. Experimental results show that
embedding models and ColBERT-style models exhibit significant performance
degradation when query-related content is shifted toward later positions,
indicating a pronounced head bias. Notably, under the same training
configuration, ColBERT-style approach show greater potential for mitigating
positional bias compared to the traditional single-vector approach. In
contrast, BM25 and reranker models remain largely unaffected by such
perturbations, underscoring their robustness to positional bias. Code and data
are publicly available at: www.github.com/NovaSearch-Team/RAG-Retrieval.

### 2. [DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation](http://arxiv.org/pdf/2505.13974v1)

Authors: Hye-young Kim, Minjin Choi, Sunkyung Lee, Ilwoong Baek, Jongwuk Lee

Side-information Integrated Sequential Recommendation (SISR) benefits from
auxiliary item information to infer hidden user preferences, which is
particularly effective for sparse interactions and cold-start scenarios.
However, existing studies face two main challenges. (i) They fail to remove
noisy signals in item sequence and (ii) they underutilize the potential of
side-information integration. To tackle these issues, we propose a novel SISR
model, Dual Side-Information Filtering and Fusion (DIFF), which employs
frequency-based noise filtering and dual multi-sequence fusion. Specifically,
we convert the item sequence to the frequency domain to filter out noisy
short-term fluctuations in user interests. We then combine early and
intermediate fusion to capture diverse relationships across item IDs and
attributes. Thanks to our innovative filtering and fusion strategy, DIFF is
more robust in learning subtle and complex item correlations in the sequence.
DIFF outperforms state-of-the-art SISR models, achieving improvements of up to
14.1% and 12.5% in Recall@20 and NDCG@20 across four benchmark datasets.

### 3. [Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning](http://arxiv.org/pdf/2505.14069v1)

Authors: Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Derong Xu, Zhaocheng Du, Huifeng Guo, Ruiming Tang, Xiangyu Zhao

Retrieval-augmented generation (RAG) enhances the text generation
capabilities of large language models (LLMs) by integrating external knowledge
and up-to-date information. However, traditional RAG systems are limited by
static workflows and lack the adaptability required for multistep reasoning and
complex task management. To address these limitations, agentic RAG systems
(e.g., DeepResearch) have been proposed, enabling dynamic retrieval strategies,
iterative context refinement, and adaptive workflows for handling complex
search queries beyond the capabilities of conventional RAG. Recent advances,
such as Search-R1, have demonstrated promising gains using outcome-based
reinforcement learning, where the correctness of the final answer serves as the
reward signal. Nevertheless, such outcome-supervised agentic RAG methods face
challenges including low exploration efficiency, gradient conflict, and sparse
reward signals. To overcome these challenges, we propose to utilize
fine-grained, process-level rewards to improve training stability, reduce
computational costs, and enhance efficiency. Specifically, we introduce a novel
method ReasonRAG that automatically constructs RAG-ProGuide, a high-quality
dataset providing process-level rewards for (i) query generation, (ii) evidence
extraction, and (iii) answer generation, thereby enhancing model inherent
capabilities via process-supervised reinforcement learning. With the
process-level policy optimization, the proposed framework empowers LLMs to
autonomously invoke search, generate queries, extract relevant evidence, and
produce final answers. Compared to existing approaches such as Search-R1 and
traditional RAG systems, ReasonRAG, leveraging RAG-ProGuide, achieves superior
performance on five benchmark datasets using only 5k training instances,
significantly fewer than the 90k training instances required by Search-R1.

### 4. [The Limits of Graph Samplers for Training Inductive Recommender Systems: Extended results](http://arxiv.org/pdf/2505.14241v1)

Authors: Theis E. Jendal, Matteo Lissandrini, Peter Dolog, Katja Hose

Inductive Recommender Systems are capable of recommending for new users and
with new items thus avoiding the need to retrain after new data reaches the
system. However, these methods are still trained on all the data available,
requiring multiple days to train a single model, without counting
hyperparameter tuning. In this work we focus on graph-based recommender
systems, i.e., systems that model the data as a heterogeneous network. In other
applications, graph sampling allows to study a subgraph and generalize the
findings to the original graph. Thus, we investigate the applicability of
sampling techniques for this task. We test on three real world datasets, with
three state-of-the-art inductive methods, and using six different sampling
methods. We find that its possible to maintain performance using only 50% of
the training data with up to 86% percent decrease in training time; however,
using less training data leads to far worse performance. Further, we find that
when it comes to data for recommendations, graph sampling should also account
for the temporal dimension. Therefore, we find that if higher data reduction is
needed, new graph based sampling techniques should be studied and new inductive
methods should be designed.

### 5. [R2MED: A Benchmark for Reasoning-Driven Medical Retrieval](http://arxiv.org/pdf/2505.14558v1)

Authors: Lei Li, Xiao Zhou, Zheng Liu

Current medical retrieval benchmarks primarily emphasize lexical or shallow
semantic similarity, overlooking the reasoning-intensive demands that are
central to clinical decision-making. In practice, physicians often retrieve
authoritative medical evidence to support diagnostic hypotheses. Such evidence
typically aligns with an inferred diagnosis rather than the surface form of a
patient's symptoms, leading to low lexical or semantic overlap between queries
and relevant documents. To address this gap, we introduce R2MED, the first
benchmark explicitly designed for reasoning-driven medical retrieval. It
comprises 876 queries spanning three tasks: Q&A reference retrieval, clinical
evidence retrieval, and clinical case retrieval. These tasks are drawn from
five representative medical scenarios and twelve body systems, capturing the
complexity and diversity of real-world medical information needs. We evaluate
15 widely-used retrieval systems on R2MED and find that even the best model
achieves only 31.4 nDCG@10, demonstrating the benchmark's difficulty. Classical
re-ranking and generation-augmented retrieval methods offer only modest
improvements. Although large reasoning models improve performance via
intermediate inference generation, the best results still peak at 41.4 nDCG@10.
These findings underscore a substantial gap between current retrieval
techniques and the reasoning demands of real clinical tasks. We release R2MED
as a challenging benchmark to foster the development of next-generation medical
retrieval systems with enhanced reasoning capabilities. Data and code are
available at https://github.com/R2MED/R2MED

### 6. [TranSUN: A Preemptive Paradigm to Eradicate Retransformation Bias Intrinsically from Regression Models in Recommender Systems](http://arxiv.org/pdf/2505.13881v1)

Authors: Jiahao Yu, Haozhuang Liu, Yeqiu Yang, Lu Chen, Wu Jian, Yuning Jiang, Bo Zheng

Regression models are crucial in recommender systems. However,
retransformation bias problem has been conspicuously neglected within the
community. While many works in other fields have devised effective bias
correction methods, all of them are post-hoc cures externally to the model,
facing practical challenges when applied to real-world recommender systems.
Hence, we propose a preemptive paradigm to eradicate the bias intrinsically
from the models via minor model refinement. Specifically, a novel TranSUN
method is proposed with a joint bias learning manner to offer theoretically
guaranteed unbiasedness under empirical superior convergence. It is further
generalized into a novel generic regression model family, termed Generalized
TranSUN (GTS), which not only offers more theoretical insights but also serves
as a generic framework for flexibly developing various bias-free models.
Comprehensive experimental results demonstrate the superiority of our methods
across data from various domains, which have been successfully deployed in two
real-world industrial recommendation scenarios, i.e. product and short video
recommendation scenarios in Guess What You Like business domain in the homepage
of Taobao App (a leading e-commerce platform), to serve the major online
traffic. Codes will be released after this paper is published.

### 7. [LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts](http://arxiv.org/pdf/2505.13928v1)

Authors: Qifeng Cai, Hao Liang, Hejun Dong, Meiyi Qiang, Ruichuan An, Zhaoyang Han, Zhengzhou Zhu, Bin Cui, Wentao Zhang

Long videos contain a vast amount of information, making video-text retrieval
an essential and challenging task in multimodal learning. However, existing
benchmarks suffer from limited video duration, low-quality captions, and coarse
annotation granularity, which hinder the evaluation of advanced video-text
retrieval methods. To address these limitations, we introduce LoVR, a benchmark
specifically designed for long video-text retrieval. LoVR contains 467 long
videos and over 40,804 fine-grained clips with high-quality captions. To
overcome the issue of poor machine-generated annotations, we propose an
efficient caption generation framework that integrates VLM automatic
generation, caption quality scoring, and dynamic refinement. This pipeline
improves annotation accuracy while maintaining scalability. Furthermore, we
introduce a semantic fusion method to generate coherent full-video captions
without losing important contextual information. Our benchmark introduces
longer videos, more detailed captions, and a larger-scale dataset, presenting
new challenges for video understanding and retrieval. Extensive experiments on
various advanced embedding models demonstrate that LoVR is a challenging
benchmark, revealing the limitations of current approaches and providing
valuable insights for future research. We release the code and dataset link at
https://github.com/TechNomad-ds/LoVR-benchmark

### 8. [Field Matters: A lightweight LLM-enhanced Method for CTR Prediction](http://arxiv.org/pdf/2505.14057v1)

Authors: Yu Cui, Feng Liu, Jiawei Chen, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Xiaohu Yang, Can Wang

Click-through rate (CTR) prediction is a fundamental task in modern
recommender systems. In recent years, the integration of large language models
(LLMs) has been shown to effectively enhance the performance of traditional CTR
methods. However, existing LLM-enhanced methods often require extensive
processing of detailed textual descriptions for large-scale instances or
user/item entities, leading to substantial computational overhead. To address
this challenge, this work introduces LLaCTR, a novel and lightweight
LLM-enhanced CTR method that employs a field-level enhancement paradigm.
Specifically, LLaCTR first utilizes LLMs to distill crucial and lightweight
semantic knowledge from small-scale feature fields through self-supervised
field-feature fine-tuning. Subsequently, it leverages this field-level semantic
knowledge to enhance both feature representation and feature interactions. In
our experiments, we integrate LLaCTR with six representative CTR models across
four datasets, demonstrating its superior performance in terms of both
effectiveness and efficiency compared to existing LLM-enhanced methods. Our
code is available at https://anonymous.4open.science/r/LLaCTR-EC46.

### 9. [Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering](http://arxiv.org/pdf/2505.14099v1)

Authors: Yihua Zhu, Qianying Liu, Akiko Aizawa, Hidetoshi Shimodaira

Knowledge Base Question Answering (KBQA) aims to answer natural language
questions using structured knowledge from KBs. While LLM-only approaches offer
generalization, they suffer from outdated knowledge, hallucinations, and lack
of transparency. Chain-based KG-RAG methods address these issues by
incorporating external KBs, but are limited to simple chain-structured
questions due to the absence of planning and logical structuring. Inspired by
semantic parsing methods, we propose PDRR: a four-stage framework consisting of
Predict, Decompose, Retrieve, and Reason. Our method first predicts the
question type and decomposes the question into structured triples. Then
retrieves relevant information from KBs and guides the LLM as an agent to
reason over and complete the decomposed triples. Experimental results
demonstrate that PDRR consistently outperforms existing methods across various
LLM backbones and achieves superior performance on both chain-structured and
non-chain complex questions.

### 10. [Bridge the Gap between Past and Future: Siamese Model Optimization for Context-Aware Document Ranking](http://arxiv.org/pdf/2505.14180v1)

Authors: Songhao Wu, Quan Tu, Mingjie Zhong, Hong Liu, Jia Xu, Jinjie Gu, Rui Yan

In the realm of information retrieval, users often engage in multi-turn
interactions with search engines to acquire information, leading to the
formation of sequences of user feedback behaviors. Leveraging the session
context has proven to be beneficial for inferring user search intent and
document ranking. A multitude of approaches have been proposed to exploit
in-session context for improved document ranking. Despite these advances, the
limitation of historical session data for capturing evolving user intent
remains a challenge. In this work, we explore the integration of future
contextual information into the session context to enhance document ranking. We
present the siamese model optimization framework, comprising a
history-conditioned model and a future-aware model. The former processes only
the historical behavior sequence, while the latter integrates both historical
and anticipated future behaviors. Both models are trained collaboratively using
the supervised labels and pseudo labels predicted by the other. The
history-conditioned model, referred to as ForeRanker, progressively learns
future-relevant information to enhance ranking, while it singly uses historical
session at inference time. To mitigate inconsistencies during training, we
introduce the peer knowledge distillation method with a dynamic gating
mechanism, allowing models to selectively incorporate contextual information.
Experimental results on benchmark datasets demonstrate the effectiveness of our
ForeRanker, showcasing its superior performance compared to existing methods.

### Machine Learning

### 1. [Context-Free Synthetic Data Mitigates Forgetting](http://arxiv.org/pdf/2505.13811v1)

Authors: Parikshit Bansal, Sujay Sanghavi

Fine-tuning a language model often results in a degradation of its existing
performance on other tasks, due to a shift in the model parameters; this
phenomenon is often referred to as (catastrophic) forgetting. We are interested
in mitigating this, in settings where we only have access to the model weights
but no access to its training data/recipe. A natural approach is to penalize
the KL divergence between the original model and the new one. Our main
realization is that a simple process - which we term context-free generation -
allows for an approximate unbiased estimation of this KL divergence. We show
that augmenting a fine-tuning dataset with context-free generations mitigates
forgetting, in two settings: (a) preserving the zero-shot performance of
pretrained-only models, and (b) preserving the reasoning performance of
thinking models. We show that contextual synthetic data, and even a portion of
the pretraining data, are less effective. We also investigate the effect of
choices like generation temperature, data ratios etc. We present our results
for OLMo-1B for pretrained-only setting and R1-Distill-Llama-8B for the
reasoning setting.

### 2. [Enforcing Hard Linear Constraints in Deep Learning Models with Decision Rules](http://arxiv.org/pdf/2505.13858v1)

Authors: Gonzalo E. Constante-Flores, Hao Chen, Can Li

Deep learning models are increasingly deployed in safety-critical tasks where
predictions must satisfy hard constraints, such as physical laws, fairness
requirements, or safety limits. However, standard architectures lack built-in
mechanisms to enforce such constraints, and existing approaches based on
regularization or projection are often limited to simple constraints,
computationally expensive, or lack feasibility guarantees. This paper proposes
a model-agnostic framework for enforcing input-dependent linear equality and
inequality constraints on neural network outputs. The architecture combines a
task network trained for prediction accuracy with a safe network trained using
decision rules from the stochastic and robust optimization literature to ensure
feasibility across the entire input space. The final prediction is a convex
combination of the two subnetworks, guaranteeing constraint satisfaction during
both training and inference without iterative procedures or runtime
optimization. We prove that the architecture is a universal approximator of
constrained functions and derive computationally tractable formulations based
on linear decision rules. Empirical results on benchmark regression tasks show
that our method consistently satisfies constraints while maintaining
competitive accuracy and low inference latency.

### 3. [CRAFT: Time Series Forecasting with Cross-Future Behavior Awareness](http://arxiv.org/pdf/2505.13896v1)

Authors: Yingwei Zhang, Ke Bu, Zhuoran Zhuang, Tao Xie, Yao Yu, Dong Li, Yang Guo, Detao Lv

The past decades witness the significant advancements in time series
forecasting (TSF) across various real-world domains, including e-commerce and
disease spread prediction. However, TSF is usually constrained by the
uncertainty dilemma of predicting future data with limited past observations.
To settle this question, we explore the use of Cross-Future Behavior (CFB) in
TSF, which occurs before the current time but takes effect in the future. We
leverage CFB features and propose the CRoss-Future Behavior Awareness based
Time Series Forecasting method (CRAFT). The core idea of CRAFT is to utilize
the trend of cross-future behavior to mine the trend of time series data to be
predicted. Specifically, to settle the sparse and partial flaws of cross-future
behavior, CRAFT employs the Koopman Predictor Module to extract the key trend
and the Internal Trend Mining Module to supplement the unknown area of the
cross-future behavior matrix. Then, we introduce the External Trend Guide
Module with a hierarchical structure to acquire more representative trends from
higher levels. Finally, we apply the demand-constrained loss to calibrate the
distribution deviation of prediction results. We conduct experiments on
real-world dataset. Experiments on both offline large-scale dataset and online
A/B test demonstrate the effectiveness of CRAFT. Our dataset and code is
available at https://github.com/CRAFTinTSF/CRAFT.

### 4. [Exploring Causes of Representational Similarity in Machine Learning Models](http://arxiv.org/pdf/2505.13899v1)

Authors: Zeyu Michael Li, Hung Anh Vu, Damilola Awofisayo, Emily Wenger

Numerous works have noted significant similarities in how machine learning
models represent the world, even across modalities. Although much effort has
been devoted to uncovering properties and metrics on which these models align,
surprisingly little work has explored causes of this similarity. To advance
this line of inquiry, this work explores how two possible causal factors --
dataset overlap and task overlap -- influence downstream model similarity. The
exploration of dataset overlap is motivated by the reality that large-scale
generative AI models are often trained on overlapping datasets of scraped
internet data, while the exploration of task overlap seeks to substantiate
claims from a recent work, the Platonic Representation Hypothesis, that task
similarity may drive model similarity. We evaluate the effects of both factors
through a broad set of experiments. We find that both positively correlate with
higher representational similarity and that combining them provides the
strongest effect. Our code and dataset are published.

### 5. [New Evidence of the Two-Phase Learning Dynamics of Neural Networks](http://arxiv.org/pdf/2505.13900v1)

Authors: Zhanpeng Zhou, Yongyi Yang, Mahito Sugiyama, Junchi Yan

Understanding how deep neural networks learn remains a fundamental challenge
in modern machine learning. A growing body of evidence suggests that training
dynamics undergo a distinct phase transition, yet our understanding of this
transition is still incomplete. In this paper, we introduce an interval-wise
perspective that compares network states across a time window, revealing two
new phenomena that illuminate the two-phase nature of deep learning. i)
\textbf{The Chaos Effect.} By injecting an imperceptibly small parameter
perturbation at various stages, we show that the response of the network to the
perturbation exhibits a transition from chaotic to stable, suggesting there is
an early critical period where the network is highly sensitive to initial
conditions; ii) \textbf{The Cone Effect.} Tracking the evolution of the
empirical Neural Tangent Kernel (eNTK), we find that after this transition
point the model's functional trajectory is confined to a narrow cone-shaped
subset: while the kernel continues to change, it gets trapped into a tight
angular region. Together, these effects provide a structural, dynamical view of
how deep networks transition from sensitive exploration to stable refinement
during training.

### 6. [Cross-Domain Diffusion with Progressive Alignment for Efficient Adaptive Retrieval](http://arxiv.org/pdf/2505.13907v1)

Authors: Junyu Luo, Yusheng Zhao, Xiao Luo, Zhiping Xiao, Wei Ju, Li Shen, Dacheng Tao, Ming Zhang

Unsupervised efficient domain adaptive retrieval aims to transfer knowledge
from a labeled source domain to an unlabeled target domain, while maintaining
low storage cost and high retrieval efficiency. However, existing methods
typically fail to address potential noise in the target domain, and directly
align high-level features across domains, thus resulting in suboptimal
retrieval performance. To address these challenges, we propose a novel
Cross-Domain Diffusion with Progressive Alignment method (COUPLE). This
approach revisits unsupervised efficient domain adaptive retrieval from a graph
diffusion perspective, simulating cross-domain adaptation dynamics to achieve a
stable target domain adaptation process. First, we construct a cross-domain
relationship graph and leverage noise-robust graph flow diffusion to simulate
the transfer dynamics from the source domain to the target domain, identifying
lower noise clusters. We then leverage the graph diffusion results for
discriminative hash code learning, effectively learning from the target domain
while reducing the negative impact of noise. Furthermore, we employ a
hierarchical Mixup operation for progressive domain alignment, which is
performed along the cross-domain random walk paths. Utilizing target domain
discriminative hash learning and progressive domain alignment, COUPLE enables
effective domain adaptive hash learning. Extensive experiments demonstrate
COUPLE's effectiveness on competitive benchmarks.

### 7. [ShortcutProbe: Probing Prediction Shortcuts for Learning Robust Models](http://arxiv.org/pdf/2505.13910v1)

Authors: Guangtao Zheng, Wenqian Ye, Aidong Zhang

Deep learning models often achieve high performance by inadvertently learning
spurious correlations between targets and non-essential features. For example,
an image classifier may identify an object via its background that spuriously
correlates with it. This prediction behavior, known as spurious bias, severely
degrades model performance on data that lacks the learned spurious
correlations. Existing methods on spurious bias mitigation typically require a
variety of data groups with spurious correlation annotations called group
labels. However, group labels require costly human annotations and often fail
to capture subtle spurious biases such as relying on specific pixels for
predictions. In this paper, we propose a novel post hoc spurious bias
mitigation framework without requiring group labels. Our framework, termed
ShortcutProbe, identifies prediction shortcuts that reflect potential
non-robustness in predictions in a given model's latent space. The model is
then retrained to be invariant to the identified prediction shortcuts for
improved robustness. We theoretically analyze the effectiveness of the
framework and empirically demonstrate that it is an efficient and practical
tool for improving a model's robustness to spurious bias on diverse datasets.

### 8. [VAMO: Efficient Large-Scale Nonconvex Optimization via Adaptive Zeroth Order Variance Reduction](http://arxiv.org/pdf/2505.13954v1)

Authors: Jiahe Chen, Ziye Ma

Optimizing large-scale nonconvex problems, common in machine learning,
demands balancing rapid convergence with computational efficiency. First-order
(FO) stochastic methods like SVRG provide fast convergence and good
generalization but incur high costs due to full-batch gradients in large
models. Conversely, zeroth-order (ZO) algorithms reduce this burden using
estimated gradients, yet their slow convergence in high-dimensional settings
limits practicality. We introduce VAMO (VAriance-reduced Mixed-gradient
Optimizer), a stochastic variance-reduced method combining FO mini-batch
gradients with lightweight ZO finite-difference probes under an SVRG-style
framework. VAMO's hybrid design uses a two-point ZO estimator to achieve a
dimension-agnostic convergence rate of $\mathcal{O}(1/T + 1/b)$, where $T$ is
the number of iterations and $b$ is the batch-size, surpassing the
dimension-dependent slowdown of purely ZO methods and significantly improving
over SGD's $\mathcal{O}(1/\sqrt{T})$ rate. Additionally, we propose a
multi-point ZO variant that mitigates the $O(1/b)$ error by adjusting number of
estimation points to balance convergence and cost, making it ideal for a whole
range of computationally constrained scenarios. Experiments including
traditional neural network training and LLM finetuning show VAMO outperforms
established FO and ZO methods, offering a faster, more flexible option for
improved efficiency.

### 9. [Adaptive Sentencing Prediction with Guaranteed Accuracy and Legal Interpretability](http://arxiv.org/pdf/2505.14011v1)

Authors: Yifei Jin, Xin Zheng, Lei Guo

Existing research on judicial sentencing prediction predominantly relies on
end-to-end models, which often neglect the inherent sentencing logic and lack
interpretability-a critical requirement for both scholarly research and
judicial practice. To address this challenge, we make three key
contributions:First, we propose a novel Saturated Mechanistic Sentencing (SMS)
model, which provides inherent legal interpretability by virtue of its
foundation in China's Criminal Law. We also introduce the corresponding
Momentum Least Mean Squares (MLMS) adaptive algorithm for this model. Second,
for the MLMS algorithm based adaptive sentencing predictor, we establish a
mathematical theory on the accuracy of adaptive prediction without resorting to
any stationarity and independence assumptions on the data. We also provide a
best possible upper bound for the prediction accuracy achievable by the best
predictor designed in the known parameters case. Third, we construct a Chinese
Intentional Bodily Harm (CIBH) dataset. Utilizing this real-world data,
extensive experiments demonstrate that our approach achieves a prediction
accuracy that is not far from the best possible theoretical upper bound,
validating both the model's suitability and the algorithm's accuracy.

### 10. [Unsupervised Graph Clustering with Deep Structural Entropy](http://arxiv.org/pdf/2505.14040v1)

Authors: Jingyun Zhang, Hao Peng, Li Sun, Guanlin Wu, Chunyang Liu, Zhengtao Yu

Research on Graph Structure Learning (GSL) provides key insights for
graph-based clustering, yet current methods like Graph Neural Networks (GNNs),
Graph Attention Networks (GATs), and contrastive learning often rely heavily on
the original graph structure. Their performance deteriorates when the original
graph's adjacency matrix is too sparse or contains noisy edges unrelated to
clustering. Moreover, these methods depend on learning node embeddings and
using traditional techniques like k-means to form clusters, which may not fully
capture the underlying graph structure between nodes. To address these
limitations, this paper introduces DeSE, a novel unsupervised graph clustering
framework incorporating Deep Structural Entropy. It enhances the original graph
with quantified structural information and deep neural networks to form
clusters. Specifically, we first propose a method for calculating structural
entropy with soft assignment, which quantifies structure in a differentiable
form. Next, we design a Structural Learning layer (SLL) to generate an
attributed graph from the original feature data, serving as a target to enhance
and optimize the original structural graph, thereby mitigating the issue of
sparse connections between graph nodes. Finally, our clustering assignment
method (ASS), based on GNNs, learns node embeddings and a soft assignment
matrix to cluster on the enhanced graph. The ASS layer can be stacked to meet
downstream task requirements, minimizing structural entropy for stable
clustering and maximizing node consistency with edge-based cross-entropy loss.
Extensive comparative experiments are conducted on four benchmark datasets
against eight representative unsupervised graph clustering baselines,
demonstrating the superiority of the DeSE in both effectiveness and
interpretability.

### Neural and Evolutionary Computing

### 1. [Weak Pareto Boundary: The Achilles' Heel of Evolutionary Multi-Objective Optimization](http://arxiv.org/pdf/2505.13854v1)

Authors: Ruihao Zheng, Jingda Deng, Zhenkun Wang

The weak Pareto boundary ($WPB$) refers to a boundary in the objective space
of a multi-objective optimization problem, characterized by weak Pareto
optimality rather than Pareto optimality. The $WPB$ brings severe challenges to
multi-objective evolutionary algorithms (MOEAs), as it may mislead the
algorithms into finding dominance-resistant solutions (DRSs), i.e., solutions
that excel on some objectives but severely underperform on the others, thereby
missing Pareto-optimal solutions. Although the severe impact of the $WPB$ on
MOEAs has been recognized, a systematic and detailed analysis remains lacking.
To fill this gap, this paper studies the attributes of the $WPB$. In
particular, the category of a $WPB$, as an attribute derived from its weakly
Pareto-optimal property, is theoretically analyzed. The analysis reveals that
the dominance resistance degrees of DRSs induced by different categories of
$WPB$s exhibit distinct asymptotic growth rates as the DRSs in the objective
space approach the $WPB$s, where a steeper asymptotic growth rate indicates a
greater hindrance to MOEAs. Beyond that, experimental studies are conducted on
various new test problems to investigate the impact of $WPB$'s attributes. The
experimental results demonstrate consistency with our theoretical findings.
Experiments on other attributes show that the performance of an MOEA is highly
sensitive to some attributes. Overall, no existing MOEAs can comprehensively
address challenges brought by these attributes.

### 2. [RAG/LLM Augmented Switching Driven Polymorphic Metaheuristic Framework](http://arxiv.org/pdf/2505.13808v1)

Authors: Faramarz Safi Esfahani, Ghassan Beydoun, Morteza Saberi, Brad McCusker, Biswajeet Pradhan

Metaheuristic algorithms are widely used for solving complex optimization
problems, yet their effectiveness is often constrained by fixed structures and
the need for extensive tuning. The Polymorphic Metaheuristic Framework (PMF)
addresses this limitation by introducing a self-adaptive metaheuristic
switching mechanism driven by real-time performance feedback and dynamic
algorithmic selection. PMF leverages the Polymorphic Metaheuristic Agent (PMA)
and the Polymorphic Metaheuristic Selection Agent (PMSA) to dynamically select
and transition between metaheuristic algorithms based on key performance
indicators, ensuring continuous adaptation. This approach enhances convergence
speed, adaptability, and solution quality, outperforming traditional
metaheuristics in high-dimensional, dynamic, and multimodal environments.
Experimental results on benchmark functions demonstrate that PMF significantly
improves optimization efficiency by mitigating stagnation and balancing
exploration-exploitation strategies across various problem landscapes. By
integrating AI-driven decision-making and self-correcting mechanisms, PMF paves
the way for scalable, intelligent, and autonomous optimization frameworks, with
promising applications in engineering, logistics, and complex decision-making
systems.

### 3. [Do Language Models Use Their Depth Efficiently?](http://arxiv.org/pdf/2505.13898v1)

Authors: Róbert Csordás, Christopher D. Manning, Christopher Potts

Modern LLMs are increasingly deep, and depth correlates with performance,
albeit with diminishing returns. However, do these models use their depth
efficiently? Do they compose more features to create higher-order computations
that are impossible in shallow models, or do they merely spread the same kinds
of computation out over more layers? To address these questions, we analyze the
residual stream of the Llama 3.1 and Qwen 3 family of models. We find: First,
comparing the output of the sublayers to the residual stream reveals that
layers in the second half contribute much less than those in the first half,
with a clear phase transition between the two halves. Second, skipping layers
in the second half has a much smaller effect on future computations and output
predictions. Third, for multihop tasks, we are unable to find evidence that
models are using increased depth to compose subresults in examples involving
many hops. Fourth, we seek to directly address whether deeper models are using
their additional layers to perform new kinds of computation. To do this, we
train linear maps from the residual stream of a shallow model to a deeper one.
We find that layers with the same relative depth map best to each other,
suggesting that the larger model simply spreads the same computations out over
its many layers. All this evidence suggests that deeper models are not using
their depth to learn new kinds of computation, but only using the greater depth
to perform more fine-grained adjustments to the residual. This may help explain
why increasing scale leads to diminishing returns for stacked Transformer
architectures.

### 4. [X-KAN: Optimizing Local Kolmogorov-Arnold Networks via Evolutionary Rule-Based Machine Learning](http://arxiv.org/pdf/2505.14273v1)

Authors: Hiroki Shiraishi, Hisao Ishibuchi, Masaya Nakata

Function approximation is a critical task in various fields. However,
existing neural network approaches struggle with locally complex or
discontinuous functions due to their reliance on a single global model covering
the entire problem space. We propose X-KAN, a novel method that optimizes
multiple local Kolmogorov-Arnold Networks (KANs) through an evolutionary
rule-based machine learning framework called XCSF. X-KAN combines KAN's high
expressiveness with XCSF's adaptive partitioning capability by implementing
local KAN models as rule consequents and defining local regions via rule
antecedents. Our experimental results on artificial test functions and
real-world datasets demonstrate that X-KAN significantly outperforms
conventional methods, including XCSF, Multi-Layer Perceptron, and KAN, in terms
of approximation accuracy. Notably, X-KAN effectively handles functions with
locally complex or discontinuous structures that are challenging for
conventional KAN, using a compact set of rules (average 7.2 $\pm$ 2.3 rules).
These results validate the effectiveness of using KAN as a local model in XCSF,
which evaluates the rule fitness based on both accuracy and generality. Our
X-KAN implementation is available at https://github.com/YNU-NakataLab/X-KAN.

### 5. [Better Neural Network Expressivity: Subdividing the Simplex](http://arxiv.org/pdf/2505.14338v1)

Authors: Egor Bakaev, Florestan Brunck, Christoph Hertrich, Jack Stade, Amir Yehudayoff

This work studies the expressivity of ReLU neural networks with a focus on
their depth. A sequence of previous works showed that $\lceil \log_2(n+1)
\rceil$ hidden layers are sufficient to compute all continuous piecewise linear
(CPWL) functions on $\mathbb{R}^n$. Hertrich, Basu, Di Summa, and Skutella
(NeurIPS'21) conjectured that this result is optimal in the sense that there
are CPWL functions on $\mathbb{R}^n$, like the maximum function, that require
this depth. We disprove the conjecture and show that
$\lceil\log_3(n-1)\rceil+1$ hidden layers are sufficient to compute all CPWL
functions on $\mathbb{R}^n$.
  A key step in the proof is that ReLU neural networks with two hidden layers
can exactly represent the maximum function of five inputs. More generally, we
show that $\lceil\log_3(n-2)\rceil+1$ hidden layers are sufficient to compute
the maximum of $n\geq 4$ numbers. Our constructions almost match the
$\lceil\log_3(n)\rceil$ lower bound of Averkov, Hojny, and Merkert (ICLR'25) in
the special case of ReLU networks with weights that are decimal fractions. The
constructions have a geometric interpretation via polyhedral subdivisions of
the simplex into ``easier'' polytopes.

### Networking and Internet Architecture

### 1. [CE-LSLM: Efficient Large-Small Language Model Inference and Communication via Cloud-Edge Collaboration](http://arxiv.org/pdf/2505.14085v1)

Authors: Pengyan Zhu, Tingting Yang

Emerging intelligent service scenarios in 6G communication impose stringent
requirements for low latency, high reliability, and privacy preservation.
Generative large language models (LLMs) are gradually becoming key enablers for
the integration of semantic communication and computation. However, due to the
limited computational resources of edge devices and the increasing complexity
of heterogeneous terminal access, existing centralized inference approaches
fail to meet the dual demands of response efficiency and data privacy in
edge-side inference tasks. To address these challenges, this paper proposes a
novel collaborative inference architecture that integrates cloud-based LLMs
with edge-deployed small language models (SLMs), enabling dynamic scheduling
and sharing of semantic-level intermediate states, and establishing a unified
computation-communication paradigm tailored for 6G networks. Specifically, a
key-value (KV) cache reuse mechanism is introduced to enhance the semantic
understanding of edge models through contextual guidance from the cloud, while
significantly reducing edge-side computational and storage overhead.
Furthermore, a cross-node parallel scheduling mechanism is proposed to achieve
asynchronous coordination between model state loading and decoding computation,
thereby improving edge responsiveness. In addition, we investigate layer
alignment and representation compression strategies between heterogeneous
models to alleviate the communication burden on the edge. Experimental results
demonstrate that the proposed architecture exhibits superior adaptability and
scalability in terms of inference latency, system stability, and concurrent
processing capacity.

### 2. [Sibling Prefixes: Identifying Similarities in IPv4 and IPv6 Prefixes](http://arxiv.org/pdf/2505.14199v1)

Authors: Fariba Osali, Khwaja Zubair Sediqi, Oliver Gasser

Since the standardization of IPv6 in 1998, both versions of the Internet
Protocol have coexisted in the Internet. Clients usually run algorithms such as
Happy Eyeballs, to decide whether to connect to an IPv4 or IPv6 endpoint for
dual-stack domains. To identify whether two addresses belong to the same device
or service, researchers have proposed different forms of alias resolution
techniques. Similarly, one can also form siblings of IPv4 and IPv6 addresses
belonging to the same device. Traditionally, all of these approaches have
focused on individual IP addresses.
  In this work, we propose the concept of "sibling prefixes", where we extend
the definition of an IPv4-IPv6 sibling to two IP prefixe-one IPv4 prefix and
its sibling IPv6 prefix. We present a technique based on large-scale DNS
resolution data to identify 76k IPv4-IPv6 sibling prefixes. We find sibling
prefixes to be relatively stable over time. We present SP-Tuner algorithm to
tune the CIDR size of sibling prefixes and improve the perfect match siblings
from 52% to 82%. For more than half of sibling prefixes, the organization names
for their IPv4 and IPv6 origin ASes are identical, and 60% of all sibling
prefixes have at least one of the prefixes with a valid ROV status in RPKI.
Furthermore, we identify sibling prefixes in 24 hypergiant and CDN networks.
Finally, we plan to regularly publish a list of sibling prefixes to be used by
network operators and fellow researchers in dual-stack studies.

### 3. [Measuring Round-Trip Response Latencies Under Asymmetric Routing](http://arxiv.org/pdf/2505.14358v1)

Authors: Bhavana Vannarth Shobhana, Yen-lin Chien, Jonathan Diamant, Badri Nath, Shir Landau Feibish, Srinivas Narayana

Latency is a key indicator of Internet service performance. Continuously
tracking the latency of client requests enables service operators to quickly
identify bottlenecks, perform adaptive resource allocation or routing, and
mitigate attacks. Passively measuring the response latency at intermediate
vantage points is attractive since it provides insight into the experience of
real clients without requiring client instrumentation or incurring probing
overheads. This paper presents PIRATE, a passive approach to measure response
latencies when only the client-to-server traffic is visible, even when
transport headers are encrypted. PIRATE estimates the time gap between causal
pairs - two requests such that the response to the first triggered the second -
as a proxy for the client-side response latency. Our experiments with a
realistic web application show that PIRATE can estimate the response latencies
measured at the client application layer to within 1 percent. A PIRATE-enhanced
layer-4 load balancer (with DSR) cuts tail latencies by 37 percent.

### 4. [open5Gcube: A Modular and Usable Framework for Mobile Network Laboratories](http://arxiv.org/pdf/2505.14501v1)

Authors: Thorsten Horstmann, Dominik Brunke, Tobias Kremeyer, Matthias Wilmes, Gunnar Schneider, Julian Sturm, Hartmut König, Michael Rademacher

In mobile network research, the integration of real-world components such as
User Equipment (UE) with open-source network infrastructure is essential yet
challenging. To address these issues, we introduce open5Gcube, a modular
framework designed to integrate popular open-source mobile network projects
into a unified management environment. Our publicly available framework allows
researchers to flexibly combine different open-source implementations,
including different versions, and simplifies experimental setups through
containerization and lightweight orchestration. We demonstrate the practical
usability of open5Gcube by evaluating its compatibility with various commercial
off-the-shelf (COTS) smartphones and modems across multiple mobile generations
(2G, 4G, and 5G). The results underline the versatility and reproducibility of
our approach, significantly advancing the accessibility of rigorous
experimentation in mobile network laboratories.

### 5. [A5/1 is in the Air: Passive Detection of 2G (GSM) Ciphering Algorithms](http://arxiv.org/pdf/2505.14509v1)

Authors: Matthias Koch, Christian Nettersheim, Thorsten Horstmann, Michael Rademacher

This paper investigates the ongoing use of the A5/1 ciphering algorithm
within 2G GSM networks. Despite its known vulnerabilities and the gradual
phasing out of GSM technology by some operators, GSM security remains relevant
due to potential downgrade attacks from 4G/5G networks and its use in IoT
applications. We present a comprehensive overview of a historical weakness
associated with the A5 family of cryptographic algorithms. Building on this,
our main contribution is the design of a measurement approach using low-cost,
off-the-shelf hardware to passively monitor Cipher Mode Command messages
transmitted by base transceiver stations (BTS). We collected over 500,000
samples at 10 different locations, focusing on the three largest mobile network
operators in Germany. Our findings reveal significant variations in algorithm
usage among these providers. One operator favors A5/3, while another
surprisingly retains a high reliance on the compromised A5/1. The third
provider shows a marked preference for A5/3 and A5/4, indicating a shift
towards more secure ciphering algorithms in GSM networks.

### 6. [Automated, Cross-Layer Root Cause Analysis of 5G Video-Conferencing Quality Degradation](http://arxiv.org/pdf/2505.14540v1)

Authors: Fan Yi, Haoran Wan, Kyle Jamieson, Oliver Michel

5G wireless networks are complex, leveraging layers of scheduling,
retransmission, and adaptation mechanisms to maximize their efficiency. But
these mechanisms interact to produce significant fluctuations in uplink and
downlink capacity and latency. This markedly impacts the performance of
real-time applications, such as video-conferencing, which are particularly
sensitive to such fluctuations, resulting in lag, stuttering, distorted audio,
and low video quality. This paper presents a cross-layer view of 5G networks
and their impact on and interaction with video-conferencing applications. We
conduct novel, detailed measurements of both Private CBRS and commercial
carrier cellular network dynamics, capturing physical- and link-layer events
and correlating them with their effects at the network and transport layers,
and the video-conferencing application itself. Our two datasets comprise days
of low-rate campus-wide Zoom telemetry data, and hours of high-rate, correlated
WebRTC-network-5G telemetry data. Based on these data, we trace performance
anomalies back to root causes, identifying 24 previously unknown causal event
chains that degrade 5G video conferencing. Armed with this knowledge, we build
Domino, a tool that automates this process and is user-extensible to future
wireless networks and interactive applications.

### 7. [6G communications through sub-Terahertz CMOS power amplifiers: Design challenges and trends](http://arxiv.org/pdf/2505.13801v1)

Authors: Jun Yan Lee, Duo Wu, Xuanrui Guo, Jian Ding Tan, Teh Jia Yew, Zi Neng Ng, Mohammad Arif Sobhan Bhuiyan, Mahdi H. Miraz

The fifth-generation (5G) network faces limitations in supporting emerging
applications, such as artificial intelligence (AI), virtual reality (VR) and
digital twins. To overcome these confines, sub-Terahertz (sub-THz) and
Terahertz (THz) technologies are considered to be key enablers of effective 6G
wireless communications, offering higher transmission speeds, longer range and
wider bandwidth. Achieving these capabilities requires careful engineering of
6G transceivers, with a focus on efficient power amplifiers (PAs) in the
front-end, which play a critical role in effectively amplifying and
transmitting signals over long distances. Complimentary
metal-oxidesemiconductor (CMOS) technology-based PA in sub-THz suffers severe
parasitic and limited maximum frequency, however, this has eventually been
solved by different design architectures and scaling down of CMOS technology to
break through the frequency limitations. In this article, we reviewed the
potentials and capabilities of CMOS technology for designing 6G hardware,
identified the state-of-art PA designs in the sub-THz band and then examined as
well as compared the designs to identify the suitable design strategies for
better performance. The circuit optimisation techniques, such as coupled-line,
passive gain boosting method, zero-degree power splitting, load-pull matching,
diode and capacitor linearisation for better gain, saturated output power and
power added efficiency, are considered for the PA design architectures at
different sub-THz bands. Furthermore, these methods are summarised and
discussed with their advantages and disadvantages in lieu with their
performances. The PA design trends, challenges and future perspectives are also
presented and discussed. Therefore, this comprehensive review article will
serve as a comparative study and reference for future PA designs for radio
frequency integrated circuits (RFIC).

### 8. [VaN3Twin: the Multi-Technology V2X Digital Twin with Ray-Tracing in the Loop](http://arxiv.org/pdf/2505.14184v1)

Authors: Roberto Pegurri, Diego Gasco, Francesco Linsalata, Marco Rapelli, Eugenio Moro, Francesco Raviglione, Claudio Casetti

This paper presents VaN3Twin-the first open-source, full-stack Network
Digital Twin (NDT) framework for simulating the coexistence of multiple
Vehicle-to-Everything (V2X) communication technologies with accurate
physical-layer modeling via ray tracing. VaN3Twin extends the ms-van3t
simulator by integrating Sionna Ray Tracer (RT) in the loop, enabling
high-fidelity representation of wireless propagation, including diverse
Line-of-Sight (LoS) conditions with focus on LoS blockage due to other
vehicles' meshes, Doppler effect, and site-dependent effects-e.g., scattering
and diffraction. Unlike conventional simulation tools, the proposed framework
supports realistic coexistence analysis across DSRC and C-V2X technologies
operating over shared spectrum. A dedicated interference tracking module
captures cross-technology interference at the time-frequency resource block
level and enhances signal-to-interference-plus-noise ratio (SINR) estimation by
eliminating artifacts such as the bimodal behavior induced by separate LoS/NLoS
propagation models. Compared to field measurements, VaN3Twin reduces
application-layer disagreement by 50% in rural and over 70% in urban
environments with respect to current state-of-the-art simulation tools,
demonstrating its value for scalable and accurate digital twin-based V2X
coexistence simulation.

### 9. [Interpretable Reinforcement Learning for Load Balancing using Kolmogorov-Arnold Networks](http://arxiv.org/pdf/2505.14459v1)

Authors: Kamal Singh, Sami Marouani, Ahmad Al Sheikh, Pham Tran Anh Quang, Amaury Habrard

Reinforcement learning (RL) has been increasingly applied to network control
problems, such as load balancing. However, existing RL approaches often suffer
from lack of interpretability and difficulty in extracting controller
equations. In this paper, we propose the use of Kolmogorov-Arnold Networks
(KAN) for interpretable RL in network control. We employ a PPO agent with a
1-layer actor KAN model and an MLP Critic network to learn load balancing
policies that maximise throughput utility, minimize loss as well as delay. Our
approach allows us to extract controller equations from the learned neural
networks, providing insights into the decision-making process. We evaluate our
approach using different reward functions demonstrating its effectiveness in
improving network performance while providing interpretable policies.

### 10. [PSMOA: Policy Support Multi-Objective Optimization Algorithm for Decentralized Data Replication](http://arxiv.org/pdf/2505.14574v2)

Authors: Xi Wang, Susmit Shannigrahi

Efficient data replication in decentralized storage systems must account for
diverse policies, especially in multi-organizational, data-intensive
environments. This work proposes PSMOA, a novel Policy Support Multi-objective
Optimization Algorithm for decentralized data replication that dynamically
adapts to varying organizational requirements such as minimization or
maximization of replication time, storage cost, replication based on content
popularity, and load balancing while respecting policy constraints. PSMOA
outperforms NSGA-II and NSGA-III in both Generational Distance (20.29 vs 148.74
and 67.74) and Inverted Generational Distance (0.78 vs 3.76 and 5.61),
indicating better convergence and solution distribution. These results validate
PSMOA's novelty in optimizing data replication in multi-organizational
environments.

### Robotics

### 1. [Duawlfin: A Drone with Unified Actuation for Wheeled Locomotion and Flight Operation](http://arxiv.org/pdf/2505.13836v1)

Authors: Jerry Tang, Ruiqi Zhang, Kaan Beyduz, Yiwei Jiang, Cody Wiebe, Haoyu Zhang, Osaruese Asoro, Mark W. Mueller

This paper presents Duawlfin, a drone with unified actuation for wheeled
locomotion and flight operation that achieves efficient, bidirectional ground
mobility. Unlike existing hybrid designs, Duawlfin eliminates the need for
additional actuators or propeller-driven ground propulsion by leveraging only
its standard quadrotor motors and introducing a differential drivetrain with
one-way bearings. This innovation simplifies the mechanical system,
significantly reduces energy usage, and prevents the disturbance caused by
propellers spinning near the ground, such as dust interference with sensors.
Besides, the one-way bearings minimize the power transfer from motors to
propellers in the ground mode, which enables the vehicle to operate safely near
humans. We provide a detailed mechanical design, present control strategies for
rapid and smooth mode transitions, and validate the concept through extensive
experimental testing. Flight-mode tests confirm stable aerial performance
comparable to conventional quadcopters, while ground-mode experiments
demonstrate efficient slope climbing (up to 30{\deg}) and agile turning
maneuvers approaching 1g lateral acceleration. The seamless transitions between
aerial and ground modes further underscore the practicality and effectiveness
of our approach for applications like urban logistics and indoor navigation.
All the materials including 3-D model files, demonstration video and other
assets are open-sourced at https://sites.google.com/view/Duawlfin.

### 2. [InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning](http://arxiv.org/pdf/2505.13888v1)

Authors: Ji Zhang, Shihan Wu, Xu Luo, Hao Wu, Lianli Gao, Heng Tao Shen, Jingkuan Song

Leveraging pretrained Vision-Language Models (VLMs) to map language
instruction and visual observations to raw low-level actions,
Vision-Language-Action models (VLAs) hold great promise for achieving
general-purpose robotic systems. Despite their advancements, existing VLAs tend
to spuriously correlate task-irrelevant visual features with actions, limiting
their generalization capacity beyond the training data. To tackle this
challenge, we propose Intrinsic Spatial Reasoning (InSpire), a simple yet
effective approach that mitigates the adverse effects of spurious correlations
by boosting the spatial reasoning ability of VLAs. Specifically, InSpire
redirects the VLA's attention to task-relevant factors by prepending the
question "In which direction is the [object] relative to the robot?" to the
language instruction and aligning the answer
"right/left/up/down/front/back/grasped" and predicted actions with the
ground-truth. Notably, InSpire can be used as a plugin to enhance existing
autoregressive VLAs, requiring no extra training data or interaction with other
large models. Extensive experimental results in both simulation and real-world
environments demonstrate the effectiveness and flexibility of our approach. Our
code, pretrained models and demos are publicly available at:
https://Koorye.github.io/proj/Inspire.

### 3. [Robotic Monitoring of Colorimetric Leaf Sensors for Precision Agriculture](http://arxiv.org/pdf/2505.13916v1)

Authors: Malakhi Hopkins, Alice Kate Li, Shobhita Kramadhati, Jackson Arnold, Akhila Mallavarapu, Chavez Lawrence, Varun Murali, Sanjeev J. Koppal, Cherie Kagan, Vijay Kumar

Current remote sensing technologies that measure crop health e.g. RGB,
multispectral, hyperspectral, and LiDAR, are indirect, and cannot capture plant
stress indicators directly. Instead, low-cost leaf sensors that directly
interface with the crop surface present an opportunity to advance real-time
direct monitoring. To this end, we co-design a sensor-detector system, where
the sensor is a novel colorimetric leaf sensor that directly measures crop
health in a precision agriculture setting, and the detector autonomously
obtains optical signals from these leaf sensors. This system integrates a
ground robot platform with an on-board monocular RGB camera and object detector
to localize the leaf sensor, and a hyperspectral camera with motorized mirror
and an on-board halogen light to acquire a hyperspectral reflectance image of
the leaf sensor, from which a spectral response characterizing crop health can
be extracted. We show a successful demonstration of our co-designed system
operating in outdoor environments, obtaining spectra that are interpretable
when compared to controlled laboratory-grade spectrometer measurements. The
system is demonstrated in row-crop environments both indoors and outdoors where
it is able to autonomously navigate, locate and obtain a hyperspectral image of
all leaf sensors present, and retrieve interpretable spectral resonance from
leaf sensors.

### 4. [MultiDrive: A Co-Simulation Framework Bridging 2D and 3D Driving Simulation for AV Software Validation](http://arxiv.org/pdf/2505.13959v1)

Authors: Marc Kaufeld, Korbinian Moller, Alessio Gambi, Paolo Arcaini, Johannes Betz

Scenario-based testing using simulations is a cornerstone of Autonomous
Vehicles (AVs) software validation. So far, developers needed to choose between
low-fidelity 2D simulators to explore the scenario space efficiently, and
high-fidelity 3D simulators to study relevant scenarios in more detail, thus
reducing testing costs while mitigating the sim-to-real gap. This paper
presents a novel framework that leverages multi-agent co-simulation and
procedural scenario generation to support scenario-based testing across low-
and high-fidelity simulators for the development of motion planning algorithms.
Our framework limits the effort required to transition scenarios between
simulators and automates experiment execution, trajectory analysis, and
visualization. Experiments with a reference motion planner show that our
framework uncovers discrepancies between the planner's intended and actual
behavior, thus exposing weaknesses in planning assumptions under more realistic
conditions. Our framework is available at:
https://github.com/TUM-AVS/MultiDrive

### 5. [Adaptive Visuo-Tactile Fusion with Predictive Force Attention for Dexterous Manipulation](http://arxiv.org/pdf/2505.13982v1)

Authors: Jinzhou Li, Tianhao Wu, Jiyao Zhang, Zeyuan Chen, Haotian Jin, Mingdong Wu, Yujun Shen, Yaodong Yang, Hao Dong

Effectively utilizing multi-sensory data is important for robots to
generalize across diverse tasks. However, the heterogeneous nature of these
modalities makes fusion challenging. Existing methods propose strategies to
obtain comprehensively fused features but often ignore the fact that each
modality requires different levels of attention at different manipulation
stages. To address this, we propose a force-guided attention fusion module that
adaptively adjusts the weights of visual and tactile features without human
labeling. We also introduce a self-supervised future force prediction auxiliary
task to reinforce the tactile modality, improve data imbalance, and encourage
proper adjustment. Our method achieves an average success rate of 93% across
three fine-grained, contactrich tasks in real-world experiments. Further
analysis shows that our policy appropriately adjusts attention to each modality
at different manipulation stages. The videos can be viewed at
https://adaptac-dex.github.io/.

### 6. [AutoBio: A Simulation and Benchmark for Robotic Automation in Digital Biology Laboratory](http://arxiv.org/pdf/2505.14030v1)

Authors: Zhiqian Lan, Yuxuan Jiang, Ruiqi Wang, Xuanbing Xie, Rongkui Zhang, Yicheng Zhu, Peihang Li, Tianshuo Yang, Tianxing Chen, Haoyu Gao, Xiaokang Yang, Xuelong Li, Hongyuan Zhang, Yao Mu, Ping Luo

Vision-language-action (VLA) models have shown promise as generalist robotic
policies by jointly leveraging visual, linguistic, and proprioceptive
modalities to generate action trajectories. While recent benchmarks have
advanced VLA research in domestic tasks, professional science-oriented domains
remain underexplored. We introduce AutoBio, a simulation framework and
benchmark designed to evaluate robotic automation in biology laboratory
environments--an application domain that combines structured protocols with
demanding precision and multimodal interaction. AutoBio extends existing
simulation capabilities through a pipeline for digitizing real-world laboratory
instruments, specialized physics plugins for mechanisms ubiquitous in
laboratory workflows, and a rendering stack that support dynamic instrument
interfaces and transparent materials through physically based rendering. Our
benchmark comprises biologically grounded tasks spanning three difficulty
levels, enabling standardized evaluation of language-guided robotic
manipulation in experimental protocols. We provide infrastructure for
demonstration generation and seamless integration with VLA models. Baseline
evaluations with two SOTA VLA models reveal significant gaps in precision
manipulation, visual reasoning, and instruction following in scientific
workflows. By releasing AutoBio, we aim to catalyze research on generalist
robotic systems for complex, high-precision, and multimodal professional
environments. The simulator and benchmark are publicly available to facilitate
reproducible research.

### 7. [Unconventional Hexacopters via Evolution and Learning: Performance Gains and New Insights](http://arxiv.org/pdf/2505.14129v1)

Authors: Jed Muff, Keiichi Ito, Elijah H. W. Ang, Karine Miras, A. E. Eiben

Evolution and learning have historically been interrelated topics, and their
interplay is attracting increased interest lately. The emerging new factor in
this trend is morphological evolution, the evolution of physical forms within
embodied AI systems such as robots. In this study, we investigate a system of
hexacopter-type drones with evolvable morphologies and learnable controllers
and make contributions to two fields. For aerial robotics, we demonstrate that
the combination of evolution and learning can deliver non-conventional drones
that significantly outperform the traditional hexacopter on several tasks that
are more complex than previously considered in the literature. For the field of
Evolutionary Computing, we introduce novel metrics and perform new analyses
into the interaction of morphological evolution and learning, uncovering
hitherto unidentified effects. Our analysis tools are domain-agnostic, making a
methodological contribution towards building solid foundations for embodied AI
systems that integrate evolution and learning.

### 8. [Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning](http://arxiv.org/pdf/2505.14266v1)

Authors: Nikhil Sobanbabu, Guanqi He, Tairan He, Yuxiang Yang, Guanya Shi

Sim-to-real discrepancies hinder learning-based policies from achieving
high-precision tasks in the real world. While Domain Randomization (DR) is
commonly used to bridge this gap, it often relies on heuristics and can lead to
overly conservative policies with degrading performance when not properly
tuned. System Identification (Sys-ID) offers a targeted approach, but standard
techniques rely on differentiable dynamics and/or direct torque measurement,
assumptions that rarely hold for contact-rich legged systems. To this end, we
present SPI-Active (Sampling-based Parameter Identification with Active
Exploration), a two-stage framework that estimates physical parameters of
legged robots to minimize the sim-to-real gap. SPI-Active robustly identifies
key physical parameters through massive parallel sampling, minimizing state
prediction errors between simulated and real-world trajectories. To further
improve the informativeness of collected data, we introduce an active
exploration strategy that maximizes the Fisher Information of the collected
real-world trajectories via optimizing the input commands of an exploration
policy. This targeted exploration leads to accurate identification and better
generalization across diverse tasks. Experiments demonstrate that SPI-Active
enables precise sim-to-real transfer of learned policies to the real world,
outperforming baselines by 42-63% in various locomotion tasks.

### 9. [Local Minima Prediction using Dynamic Bayesian Filtering for UGV Navigation in Unstructured Environments](http://arxiv.org/pdf/2505.14337v1)

Authors: Seung Hun Lee, Wonse Jo, Lionel P. Robert Jr., Dawn M. Tilbury

Path planning is crucial for the navigation of autonomous vehicles, yet these
vehicles face challenges in complex and real-world environments. Although a
global view may be provided, it is often outdated, necessitating the reliance
of Unmanned Ground Vehicles (UGVs) on real-time local information. This
reliance on partial information, without considering the global context, can
lead to UGVs getting stuck in local minima. This paper develops a method to
proactively predict local minima using Dynamic Bayesian filtering, based on the
detected obstacles in the local view and the global goal. This approach aims to
enhance the autonomous navigation of self-driving vehicles by allowing them to
predict potential pitfalls before they get stuck, and either ask for help from
a human, or re-plan an alternate trajectory.

### 10. [Semantically-driven Deep Reinforcement Learning for Inspection Path Planning](http://arxiv.org/pdf/2505.14443v1)

Authors: Grzegorz Malczyk, Mihir Kulkarni, Kostas Alexis

This paper introduces a novel semantics-aware inspection planning policy
derived through deep reinforcement learning. Reflecting the fact that within
autonomous informative path planning missions in unknown environments, it is
often only a sparse set of objects of interest that need to be inspected, the
method contributes an end-to-end policy that simultaneously performs semantic
object visual inspection combined with collision-free navigation. Assuming
access only to the instantaneous depth map, the associated segmentation image,
the ego-centric local occupancy, and the history of past positions in the
robot's neighborhood, the method demonstrates robust generalizability and
successful crossing of the sim2real gap. Beyond simulations and extensive
comparison studies, the approach is verified in experimental evaluations
onboard a flying robot deployed in novel environments with previously unseen
semantics and overall geometric configurations.

### Software Engineering

### 1. [The Capability of Code Review as a Communication Network](http://arxiv.org/pdf/2505.13985v1)

Authors: Michael Dorner, Daniel Mendez

Background: Code review, a core practice in software engineering, has been
widely studied as a collaborative process, with prior work suggesting it
functions as a communication network. However, this theory remains untested,
limiting its practical and theoretical significance.
  Objective: This study aims to (1) formalize the theory of code review as a
communication network explicit and (2) empirically test its validity by
quantifying how widely and how quickly information can spread in code review.
  Method: We replicate an in-silico experiment simulating information diffusion
-- the spread of information among participants -- under best-case conditions
across three open-source (Android, Visual Studio Code, React) and three
closed-source code review systems (Microsoft, Spotify, Trivago) each modeled as
communication network. By measuring the number of reachable participants and
the minimal topological and temporal distances, we quantify how widely and how
quickly information can spread through code review.
  Results: We demonstrate that code review can enable both wide and fast
information diffusion, even at a large scale. However, this capacity varies:
open-source code review spreads information faster, while closed-source review
reaches more participants.
  Conclusion: Our findings reinforce and refine the theory, highlighting
implications for measuring collaboration, generalizing open-source studies, and
the role of AI in shaping future code review.

### 2. [Capturing the Effects of Quantization on Trojans in Code LLMs](http://arxiv.org/pdf/2505.14200v1)

Authors: Aftab Hussain, Sadegh AlMahdi Kazemi Zarkouei, Md Rafiqul Islam Rabin, Mohammad Amin Alipour, Sen Lin, Bowen Xu

Large language models of code exhibit high capability in performing diverse
software engineering tasks, such as code translation, defect detection,
text-to-code generation, and code summarization. While their ability to enhance
developer productivity has spurred widespread use, these models have also seen
substantial growth in size, often reaching billions of parameters. This scale
demands efficient memory resource usage, prompting practitioners to use
optimization techniques such as model quantization. Quantization uses smaller
bit representations for the model parameters, reducing the precision of the
weights. In this work, we investigate the impact of quantization on the risk of
data poisoning attacks on these models, specifically examining whether it
mitigates or exacerbates such vulnerabilities. We focus on two large language
models, Meta's Llama-2-7b and CodeLlama-7b, applied to an SQL code generation
task. Additionally, we introduce a new metric for measuring trojan signals in
compromised models. We find that quantization has differing effects on
code-generating LLMs: while reducing precision does not significantly alter
Llama-2's behavior, it boosts performance and reduces attack success rates in
CodeLlama, particularly at 4-bit precision.

### 3. [A Mosaic of Perspectives: Understanding Ownership in Software Engineering](http://arxiv.org/pdf/2505.14220v1)

Authors: Tomi Suomi, Petri Ihantola, Tommi Mikkonen, Niko Mäkitalo

Agile software development relies on self-organized teams, underlining the
importance of individual responsibility. How developers take responsibility and
build ownership are influenced by external factors such as architecture and
development methods. This paper examines the existing literature on ownership
in software engineering and in psychology, and argues that a more comprehensive
view of ownership in software engineering has a great potential in improving
software team's work. Initial positions on the issue are offered for discussion
and to lay foundations for further research.

### 4. [Who Introduces and Who Fixes? Analyzing Code Quality in Collaborative Student's Projects](http://arxiv.org/pdf/2505.14315v1)

Authors: Rafael Corsi Ferrao, Igor dos Santos Montagner, Rodolfo Azevedo

This paper investigates code quality education by analyzing how errors are
introduced and corrected in group projects within an embedded systems course.
We identify who introduces errors, who fixes them, and when these actions
occur. Students learn code quality rules for C and embedded systems.
  We address three questions: RQ1: What is the impact of group formation on
code quality? RQ2: How do students interact to fix code issues? RQ3: When are
issues introduced and resolved?
  We analyzed data from eight individual labs and two group projects involving
34 students. The course provides continuous, automated feedback on code
quality.
  Findings show that the most active contributors often introduce the most
issues. Many issues are fixed late in the project. Individual labs tend to have
fewer issues due to their structured nature. Most problems are fixed by the
original author, while cross-student fixes take longer, especially in shared
code. Critical issues are fixed quickly, but non-critical ones may be ignored,
showing a focus on functionality over quality.

### 5. [Building Reuse-Sensitive Control Flow Graphs (CFGs) for EVM Bytecode](http://arxiv.org/pdf/2505.14437v1)

Authors: Dingding Wang, Jianting He, Yizheng Yang, Lei Wu, Rui Chang, Yajin Zhou

The emergence of smart contracts brings security risks, exposing users to the
threat of losing valuable cryptocurrencies, underscoring the urgency of
meticulous scrutiny. Nevertheless, the static analysis of smart contracts in
EVM bytecode faces obstacles due to flawed primitives resulting from code reuse
introduced by compilers. Code reuse, a phenomenon where identical code executes
in diverse contexts, engenders semantic ambiguities and redundant control-flow
dependencies within reuse-insensitive CFGs. This work delves into the
exploration of code reuse within EVM bytecode, outlining prevalent reuse
patterns, and introducing Esuer, a tool that dynamically identifies code reuse
when constructing CFGs. Leveraging taint analysis to dynamically identify reuse
contexts, Esuer identifies code reuse by comparing multiple contexts for a
basic block and replicates reused code for a reuse-sensitive CFG. Evaluation
involving 10,000 prevalent smart contracts, compared with six leading tools,
demonstrates Esuer's ability to notably refine CFG precision. It achieves an
execution trace coverage of 99.94% and an F1-score of 97.02% for accurate
identification of reused code. Furthermore, Esuer attains a success rate of
99.25%, with an average execution time of 1.06 seconds, outpacing tools
generating reuse-insensitive CFGs. Esuer's efficacy in assisting identifying
vulnerabilities such as tx.origin and reentrancy vulnerabilities, achieving
F1-scores of 99.97% and 99.67%, respectively.

### 6. [From What to How: A Taxonomy of Formalized Security Properties](http://arxiv.org/pdf/2505.14514v1)

Authors: Imen Sayar, Nan Messe, Sophie Ebersold, Jean-Michel Bruel

Confidentiality, integrity, availability, authenticity, authorization, and
accountability are known as security properties that secure systems should
preserve. They are usually considered as security final goals that are achieved
by system development activities, either in a direct or an indirect manner.
However, these security properties are mainly elicited in the high-level
requirement phase during the System Development Life Cycle (SDLC) and are not
refined throughout the latter phases as other artifacts such as attacks,
defenses, and system assets. To align security properties refinement with
attacks, defenses, and system assets refinements, we propose an SDLC taxonomy
of security properties that may be used in a self-adaptive context and present
the methodology for defining it. To verify and check the correctness of the
resulting taxonomy, we use the Event-B formal language.

### 7. [BugRepro: Enhancing Android Bug Reproduction with Domain-Specific Knowledge Integration](http://arxiv.org/pdf/2505.14528v1)

Authors: Hongrong Yin, Tao Zhang

Mobile application development is a fast-paced process where maintaining
high-quality user experiences is crucial. Current bug reproduction methods
predominantly depend on precise feature descriptions in bug reports. However,
the growing complexity and dynamism of modern software systems pose significant
challenges to this crucial quality assurance process, as ambiguous or
incomplete steps-to-reproduce (S2Rs) in reports frequently impede effective
debugging and maintenance. To address these challenges, we propose BugRepro, a
novel technique that integrates domain-specific knowledge to enhance the
accuracy and efficiency of bug reproduction. BugRepro adopts a
Retrieval-Augmented Generation (RAG) approach. It retrieves similar bug reports
along with their corresponding S2R entities from an example-rich RAG document.
This document serves as a valuable reference for improving the accuracy of S2R
entity extraction. In addition, BugRepro incorporates app-specific knowledge.
It explores the app's graphical user interface (GUI) and extracts UI transition
graphs. These graphs are used to guide large language models (LLMs) in their
exploration process when they encounter bottlenecks. Our experiments
demonstrate the effectiveness of BugRepro. Our method significantly outperforms
two state-of-the-art methods. For S2R entity extraction accuracy, it achieves
improvements of 8.85% and 28.89%. For bug reproduction success rate, the
improvements reach 74.55% and 152.63%. In reproduction efficiency, the gains
are 0.72% and 76.68%.

### 8. [QUT-DV25: A Dataset for Dynamic Analysis of Next-Gen Software Supply Chain Attacks](http://arxiv.org/pdf/2505.13804v1)

Authors: Sk Tanzir Mehedi, Raja Jurdak, Chadni Islam, Gowri Ramachandran

Securing software supply chains is a growing challenge due to the inadequacy
of existing datasets in capturing the complexity of next-gen attacks, such as
multiphase malware execution, remote access activation, and dynamic payload
generation. Existing datasets, which rely on metadata inspection and static
code analysis, are inadequate for detecting such attacks. This creates a
critical gap because these datasets do not capture what happens during and
after a package is installed. To address this gap, we present QUT-DV25, a
dynamic analysis dataset specifically designed to support and advance research
on detecting and mitigating supply chain attacks within the Python Package
Index (PyPI) ecosystem. This dataset captures install and post-install-time
traces from 14,271 Python packages, of which 7,127 are malicious. The packages
are executed in an isolated sandbox environment using an extended Berkeley
Packet Filter (eBPF) kernel and user-level probes. It captures 36 real-time
features, that includes system calls, network traffic, resource usages,
directory access patterns, dependency logs, and installation behaviors,
enabling the study of next-gen attack vectors. ML analysis using the QUT-DV25
dataset identified four malicious PyPI packages previously labeled as benign,
each with thousands of downloads. These packages deployed covert remote access
and multi-phase payloads, were reported to PyPI maintainers, and subsequently
removed. This highlights the practical value of QUT-DV25, as it outperforms
reactive, metadata, and static datasets, offering a robust foundation for
developing and benchmarking advanced threat detection within the evolving
software supply chain ecosystem.

### 9. [On-Demand Scenario Generation for Testing Automated Driving Systems](http://arxiv.org/pdf/2505.14053v1)

Authors: Songyang Yan, Xiaodong Zhang, Kunkun Hao, haojie xin, Yonggang Luo, Jucheng Yang, Ming Fan, Chao Yang, Jun Sun, Zijiang Yang

The safety and reliability of Automated Driving Systems (ADS) are paramount,
necessitating rigorous testing methodologies to uncover potential failures
before deployment. Traditional testing approaches often prioritize either
natural scenario sampling or safety-critical scenario generation, resulting in
overly simplistic or unrealistic hazardous tests. In practice, the demand for
natural scenarios (e.g., when evaluating the ADS's reliability in real-world
conditions), critical scenarios (e.g., when evaluating safety in critical
situations), or somewhere in between (e.g., when testing the ADS in regions
with less civilized drivers) varies depending on the testing objectives. To
address this issue, we propose the On-demand Scenario Generation (OSG)
Framework, which generates diverse scenarios with varying risk levels.
Achieving the goal of OSG is challenging due to the complexity of quantifying
the criticalness and naturalness stemming from intricate vehicle-environment
interactions, as well as the need to maintain scenario diversity across various
risk levels. OSG learns from real-world traffic datasets and employs a Risk
Intensity Regulator to quantitatively control the risk level. It also leverages
an improved heuristic search method to ensure scenario diversity. We evaluate
OSG on the Carla simulators using various ADSs. We verify OSG's ability to
generate scenarios with different risk levels and demonstrate its necessity by
comparing accident types across risk levels. With the help of OSG, we are now
able to systematically and objectively compare the performance of different
ADSs based on different risk levels.

### 10. [Design and Evaluation of a Microservices Cloud Framework for Online Travel Platforms](http://arxiv.org/pdf/2505.14508v1)

Authors: Biman Barua, M. Shamim Kaiser

Handling online travel agents globally requires efficient and flexible
software solution architectures. When it needs to handle thousands of agents
and billions of clients data globally. Microservices architecture is used to
break down a large program into numerous, smaller services which can run
individually and perform individual tasks. This paper analyses and integrates a
unique Microservices Cloud Framework designed to support Online Travel
Platforms (MCF-OTP). MCF-OTPs main goal is to increase the performance,
flexibility, and maintenance of online travel platforms via cloud computing and
microservice technologies. Large-scale travel apps, including managing numerous
data sources, dealing with traffic peaks, and providing fault tolerance, can be
addressed by the suggested framework. The framework increases good
interpretation between flawless data synchronization, microservices, and
dynamic scaling based on demand technology. An organization framework that
optimizes service borders and minimizes inter-service dependencies is
recommended. Thus, this can result in elevated development adaptability. In
this research, the principal goal is to evaluate MCF-OTPs efficiency using the
indicators of fault tolerance and response time. It is indicated by the
findings that the MCF-OTP structure excels traditional monolithic designs in
terms of dependability and scalability, managing traffic spikes seamlessly and
decreasing downtime. The cost-effective analysis helps ascertain the net gain
attained by the startup fees and the ongoing operational costs. The cloud-based
environment is used to reduce the fracture cost which also helps to increase
the efficiency of resource allocation, according to the research.

### Social and Information Networks

### 1. [Pantheon: Personalized Multi-objective Ensemble Sort via Iterative Pareto Policy Optimization](http://arxiv.org/pdf/2505.13894v1)

Authors: Jiangxia Cao, Pengbo Xu, Yin Cheng, Kaiwei Guo, Jian Tang, Shijun Wang, Dewei Leng, Shuang Yang, Zhaojie Liu, Yanan Niu, Guorui Zhou, Kun Gai

In this paper, we provide our milestone ensemble sort work and the first-hand
practical experience, Pantheon, which transforms ensemble sorting from a
"human-curated art" to a "machine-optimized science". Compared with
formulation-based ensemble sort, our Pantheon has the following advantages: (1)
Personalized Joint Training: our Pantheon is jointly trained with the real-time
ranking model, which could capture ever-changing user personalized interests
accurately. (2) Representation inheritance: instead of the highly compressed
Pxtrs, our Pantheon utilizes the fine-grained hidden-states as model input,
which could benefit from the Ranking model to enhance our model complexity.
Meanwhile, to reach a balanced multi-objective ensemble sort, we further devise
an \textbf{iterative Pareto policy optimization} (IPPO) strategy to consider
the multiple objectives at the same time. To our knowledge, this paper is the
first work to replace the entire formulation-based ensemble sort in industry
RecSys, which was fully deployed at Kuaishou live-streaming services, serving
400 Million users daily.

### 2. [How Influencers and Multipliers Drive Polarization and Issue Alignment on Twitter/X](http://arxiv.org/pdf/2505.14280v1)

Authors: Armin Pournaki, Felix Gaisbauer, Eckehard Olbrich

We investigate the polarization of the German Twittersphere by extracting the
main issues discussed and the signaled opinions of users towards those issues
based on (re)tweets concerning trending topics. The dataset covers daily
trending topics from March 2021 to July 2023. At the opinion level, we show
that the online public sphere is largely divided into two camps, one consisting
mainly of left-leaning, and another of right-leaning accounts. Further we
observe that political issues are strongly aligned, contrary to what one may
expect from surveys. This alignment is driven by two cores of strongly active
users: influencers, who generate ideologically charged content, and
multipliers, who facilitate the spread of this content. The latter are specific
to social media and play a crucial role as intermediaries on the platform by
curating and amplifying very specific types of content that match their
ideological position, resulting in the overall observation of a strongly
polarized public sphere. These results contribute to a better understanding of
the mechanisms that shape online public opinion, and have implications for the
regulation of platforms.

### 3. [UKTwitNewsCor: A Dataset of Online Local News Articles for the Study of Local News Provision](http://arxiv.org/pdf/2505.14326v1)

Authors: Simona Bisiani, Agnes Gulyas, John Wihbey, Bahareh Heravi

In this paper, we present UKTwitNewsCor, a comprehensive dataset for
understanding the content production, dissemination, and audience engagement
dynamics of online local media in the UK. It comprises over 2.5 million online
news articles published between January 2020 and December 2022 from 360 local
outlets. The corpus represents all articles shared on Twitter by the social
media accounts of these outlets. We augment the dataset by incorporating social
media performance metrics for the articles at the tweet-level. We further
augment the dataset by creating metadata about content duplication across
domains. Alongside the article dataset, we supply three additional datasets: a
directory of local media web domains, one of UK Local Authority Districts, and
one of digital local media providers, providing statistics on the coverage
scope of UKTwitNewsCor. Our contributions enable comprehensive, longitudinal
analysis of UK local media, news trends, and content diversity across multiple
platforms and geographic areas. In this paper, we describe the data collection
methodology, assess the dataset geographic and media ownership diversity, and
outline how researchers, policymakers, and industry stakeholders can leverage
UKTwitNewsCor to advance the study of local media.

### 4. [MindVote: How LLMs Predict Human Decision-Making in Social Media Polls](http://arxiv.org/pdf/2505.14422v1)

Authors: Xutao Mao, Ezra Xuanru Tao

The increasing complexity of Large Language Models (LLMs) necessitates new
benchmarks to assess their ability to predict human decision-making in dynamic
social contexts. We introduce MindVote, the first benchmark for evaluating LLMs
as "virtual respondents" in social media polling. MindVote comprises 276 poll
instances with 1,142 data entry points from three platforms (Weibo, Reddit,
Fizz), features bilingual content (Chinese/English), and covers five domains.
Our evaluation of 18 LLMs demonstrates that top-performing models achieve an
overall score of 0.74, an 80% relative improvement over traditional baselines,
and then we analyze LLM world model bias with human preferences across societal
bias dimensions. MindVote also uncovers significant disparities related to
platform, language, and domain. We present strategies to optimize LLM
performance and use LLM-as-a-Judge to assess reasoning in societal contexts.
Furthermore, we show that temperature controls can reflect a way of human
thinking diversity and opinion shifts in polling. In summary, MindVote offers a
scalable framework for evaluating LLMs' social intelligence, with implications
for understanding behavioral decision-making. Code and data will be available
soon.

### 5. [Robustness Evaluation of Graph-based News Detection Using Network Structural Information](http://arxiv.org/pdf/2505.14453v2)

Authors: Xianghua Zeng, Hao Peng, Angsheng Li

Although Graph Neural Networks (GNNs) have shown promising potential in fake
news detection, they remain highly vulnerable to adversarial manipulations
within social networks. Existing methods primarily establish connections
between malicious accounts and individual target news to investigate the
vulnerability of graph-based detectors, while they neglect the structural
relationships surrounding targets, limiting their effectiveness in robustness
evaluation. In this work, we propose a novel Structural Information
principles-guided Adversarial Attack Framework, namely SI2AF, which effectively
challenges graph-based detectors and further probes their detection robustness.
Specifically, structural entropy is introduced to quantify the dynamic
uncertainty in social engagements and identify hierarchical communities that
encompass all user accounts and news posts. An influence metric is presented to
measure each account's probability of engaging in random interactions,
facilitating the design of multiple agents that manage distinct malicious
accounts. For each target news, three attack strategies are developed through
multi-agent collaboration within the associated subgraph to optimize evasion
against black-box detectors. By incorporating the adversarial manipulations
generated by SI2AF, we enrich the original network structure and refine
graph-based detectors to improve their robustness against adversarial attacks.
Extensive evaluations demonstrate that SI2AF significantly outperforms
state-of-the-art baselines in attack effectiveness with an average improvement
of 16.71%, and enhances GNN-based detection robustness by 41.54% on average.

### Systems and Control

### 1. [On the Input-Output Monotonicity of Voltage Dynamics of Power System with Grid-Forming Converters](http://arxiv.org/pdf/2505.13838v1)

Authors: Zhenyao Li, Shengwen Liao, Qian Zhang, Xuechun Zhang, Deqiang Gan

Integration of renewable resources is profoundly reshaping the dynamics of
modern power systems. This study shows that the voltage dynamics of power
systems with multiple grid-forming (GFM) converters often enjoys a desirable
property called input-output monotonicity. A systematic approach for computing
the derivatives of the voltage subsystem is presented first, which provides
insight into the structural characteristics of these models. Next, the sign
pattern of the trajectory Jacobian matrix associated with the voltage subsystem
is analyzed and revealed. The analysis indicates that the voltage dynamics of
power systems often exhibits the so-called input-output monotonicity property.
The theoretical results are then validated through several simulation examples,
underscoring their practical implications.

### 2. [Gaming Strategies in European Imbalance Settlement Mechanisms](http://arxiv.org/pdf/2505.14133v1)

Authors: Seyed Soroush Karimi Madahi, Kenneth Bruninx, Bert Claessens, Chris Develder

Transmission System Operators (TSOs) rely on balancing energy provided by
Balancing Service Providers (BSPs) to maintain the supply-demand balance in
real time. Balance Responsible Parties (BRPs) can simultaneously deviate from
their day-ahead schedules in response to imbalance prices, e.g., by controlling
flexible assets such as batteries. According to the European Electricity
Balancing Guideline, these imbalance prices should incentivize BRPs performing
such implicit or passive balancing to aid the TSO in restoring the energy
balance. In this paper, we demonstrate that BRPs are unintentionally offered
the opportunity to exploit gaming strategies in European imbalance settlement
mechanisms. This is enabled by a disconnect between sub-quarter-hourly dynamics
that determine the imbalance prices and the financial settlement on a
quarter-hourly basis. We illustrate this behavior in a case study of the
imbalance settlement mechanisms in Belgium and the Netherlands. Our results
reveal that, in both countries, BRPs can, in theory, exploit the imbalance
mechanism by increasing the instantaneous system imbalance during minutes
within the quarter-hour that determine the imbalance price while still
contributing to restoring the system balance for the rest of the quarter-hour.

### 3. [Statistically Optimal Structured Additive MIMO Continuous-time System Identification](http://arxiv.org/pdf/2505.14169v1)

Authors: Rodrigo A. González, Maarten van der Hulst, Koen Classens, Tom Oomen

Many applications in mechanical, acoustic, and electronic engineering require
estimating complex dynamical models, often represented as additive multi-input
multi-output (MIMO) transfer functions with structural constraints. This paper
introduces a two-stage procedure for estimating structured additive MIMO
models, where structural constraints are enforced through a weighted nonlinear
least-squares projection of the parameter vector initially estimated using a
recently developed refined instrumental variables algorithm. The proposed
approach is shown to be consistent and asymptotically efficient in open-loop
scenarios. In closed-loop settings, it remains consistent despite potential
noise model misspecification and achieves minimum covariance among all
instrumental variable estimators. Extensive simulations are performed to
validate the theoretical findings, and to show the efficacy of the proposed
approach.

### 4. [Functional Controllability, Functional Stabilisability, and the Generalised Separation Principle](http://arxiv.org/pdf/2505.14176v1)

Authors: Tyrone Fernando, Mohamed Darouach

This paper introduces the new concepts of Functional Controllability and
Functional Stabilisability, and establishes their duality with Functional
Observability and Functional Detectability, respectively. We further present a
Generalised Separation Principle, demonstrating that the classical Separation
Principle emerges as a special case. Conditions for the existence of functional
controllers of a specified order are derived. Importantly, the design framework
does not require full controllability. Furthermore, we develop a functional
observer-based controller design applicable to systems that are both
uncontrollable and unobservable. The results presented generalise the classical
full-state feedback control paradigm.

### 5. [A Data-Driven Method to Identify IBRs with Dominant Participation in Sub-Synchronous Oscillations](http://arxiv.org/pdf/2505.14267v1)

Authors: Youhong Chen, Debraj Bhattacharjee, Balarko Chaudhuri

This paper introduces a data-driven (i.e., model-free) approach to identify
which inverter-based resources (IBRs) have dominant participation in poorly
damped sub-synchronous oscillations (SSO), to get to the root cause for
effective mitigation. An Enhanced Dynamic Mode Decomposition (eDMD) method is
proposed that incorporates an appropriate set of observables. Based on
time-synchronized data (either simulated or real) from IBR connection points,
eDMD directly computes data-driven eigenvectors and participation factors to
reveal the role of each IBR in poorly damped SSO. We show the improved accuracy
of eDMD over conventional Dynamic Mode Decomposition (DMD) by benchmarking both
against actual model-based analysis. We demonstrate this first through a
synthetic example and then a case study on the IEEE 39-bus test system with
100% IBR. This data-driven, model-free method offers a powerful tool to foresee
and mitigate the risk of IBR-induced SSO in planning (simulated data) and
post-event analysis (real data) of SSO events.

### 6. [Efficient Configuration-Constrained Tube MPC via Variables Restriction and Template Selection](http://arxiv.org/pdf/2505.14440v1)

Authors: Filippo Badalamenti, Sampath Kumar Mulagaleti, Mario Eduardo Villanueva, Boris Houska, Alberto Bemporad

Configuration-Constrained Tube Model Predictive Control (CCTMPC) offers
flexibility by using a polytopic parameterization of invariant sets and the
optimization of an associated vertex control law. This flexibility, however,
often demands computational trade-offs between set parameterization accuracy
and optimization complexity. This paper proposes two innovations that help the
user tackle this trade-off. First, a structured framework is proposed, which
strategically limits optimization degrees of freedom, significantly reducing
online computation time while retaining stability guarantees. This framework
aligns with Homothetic Tube MPC (HTMPC) under maximal constraints. Second, a
template refinement algorithm that iteratively solves quadratic programs is
introduced to balance polytope complexity and conservatism. Simulation studies
on an illustrative benchmark problem as well as a high-dimensional ten-state
system demonstrate the approach's efficiency, achieving robust performance with
minimal computational overhead. The results validate a practical pathway to
leveraging CCTMPC's adaptability without sacrificing real-time viability.

### 7. [Comparison of Data-Driven Modeling Approaches for Control Optimization of Floating Offshore Wind Turbines](http://arxiv.org/pdf/2505.14515v1)

Authors: Athul K. Sundarrajan, Daniel R. Herber

Models that balance accuracy against computational costs are advantageous
when designing wind turbines with optimization studies, as several hundred
predictive function evaluations might be necessary to identify the optimal
solution. We explore different approaches to construct low-fidelity models that
can be used to approximate dynamic quantities and be used as surrogates for
design optimization studies and other use cases. In particular, low-fidelity
modeling approaches using classical systems identification and deep learning
approaches are considered against derivative function surrogate models
({DFSMs}), or approximate models of the state derivative function. This work
proposes a novel method that utilizes a linear parameter varying (LPV) modeling
scheme to construct the DFSM. We compare the trade-offs between these different
models and explore the efficacy of the proposed DFSM approach in approximating
wind turbine performance and design optimization studies for controllers.
Results show that the proposed DFSM approach balances computational time and
model accuracy better than the system identification and deep learning based
models. Additionally, the DFSM provides nearly a fifty times speed-up compared
to the high-fidelity model, while balancing accuracy.

### 8. [Development of a Scaled Setup for Experimental Study of the Effect of Lateral Dynamics on Energy Consumption in Electric Vehicles: An Extension](http://arxiv.org/pdf/2505.14575v2)

Authors: Simran Kumari, Anand Ronald K., Siddhartha Mukhopadhyay, Ashish R. Hota

Most of the existing state-of-the-art approaches for energy consumption
analysis do not account for the effect of lateral dynamics on energy
consumption in electric vehicles (EVs) during vehicle maneuvers. This paper
aims to validate this effect through an experimental study. We develop a scaled
model using a radio-controlled (RC) car, modified to achieve dynamic similitude
with on-road vehicles, to conduct scaled experiments. The experimental results
confirm the impact of lateral dynamics on both energy demand and driving range
in electric vehicles, aligning with our previous findings [1], and emphasize
the need to incorporate these factors into energy consumption models. This is
an extended version of a paper accepted at IEEE ITEC 2025. It includes
additional results and analysis.

### 9. [C*: A Coverage Path Planning Algorithm for Unknown Environments using Rapidly Covering Graphs](http://arxiv.org/pdf/2505.13782v1)

Authors: Zongyuan Shen, James P. Wilson, Shalabh Gupta

The paper presents a novel sample-based algorithm, called C*, for real-time
coverage path planning (CPP) of unknown environments. The C* algorithm is built
upon the concept of Rapidly Covering Graph (RCGs). The RCG is constructed
incrementally via progressive sampling during robot navigation, which
eliminates the need for cellular decomposition of the search space. The RCG has
a sparse-graph structure formed by efficient sampling and pruning techniques,
which produces non-myopic waypoints of the coverage trajectory. While C*
produces the desired back and forth coverage pattern, it adapts to the
TSP-based locally optimal coverage of small uncovered regions, called coverage
holes, that are surrounded by obstacles and covered regions. Thus, C*
proactively detects and covers the coverage holes in situ, which reduces the
coverage time by preventing the longer return trajectories from distant regions
to cover such holes later. The algorithmic simplicity and low computational
complexity of C* makes it easy to implement and suitable for real-time onboard
applications. It is analytically proven that C* provides complete coverage of
unknown environments. The performance of C* is validated by 1) extensive
high-fidelity simulations and 2) real laboratory experiments using autonomous
robots. A comparative evaluation with seven existing CPP methods demonstrate
that C* yields significant performance improvements in terms of coverage time,
number of turns, trajectory length and overlap ratio, while preventing the
formation of coverage holes. Finally, C* is evaluated on two different
applications of CPP using 1) energy-constrained robots and 2) multi-robot
teams.

### 10. [VeRecycle: Reclaiming Guarantees from Probabilistic Certificates for Stochastic Dynamical Systems after Change](http://arxiv.org/pdf/2505.14001v1)

Authors: Sterre Lutz, Matthijs T. J. Spaan, Anna Lukina

Autonomous systems operating in the real world encounter a range of
uncertainties. Probabilistic neural Lyapunov certification is a powerful
approach to proving safety of nonlinear stochastic dynamical systems. When
faced with changes beyond the modeled uncertainties, e.g., unidentified
obstacles, probabilistic certificates must be transferred to the new system
dynamics. However, even when the changes are localized in a known part of the
state space, state-of-the-art requires complete re-certification, which is
particularly costly for neural certificates. We introduce VeRecycle, the first
framework to formally reclaim guarantees for discrete-time stochastic dynamical
systems. VeRecycle efficiently reuses probabilistic certificates when the
system dynamics deviate only in a given subset of states. We present a general
theoretical justification and algorithmic implementation. Our experimental
evaluation shows scenarios where VeRecycle both saves significant computational
effort and achieves competitive probabilistic guarantees in compositional
neural control.

### Machine Learning (Statistics Category)

### 1. [A Probabilistic Perspective on Model Collapse](http://arxiv.org/pdf/2505.13947v1)

Authors: Shirong Xu, Hengzhi He, Guang Cheng

In recent years, model collapse has become a critical issue in language model
training, making it essential to understand the underlying mechanisms driving
this phenomenon. In this paper, we investigate recursive parametric model
training from a probabilistic perspective, aiming to characterize the
conditions under which model collapse occurs and, crucially, how it can be
mitigated. We conceptualize the recursive training process as a random walk of
the model estimate, highlighting how the sample size influences the step size
and how the estimation procedure determines the direction and potential bias of
the random walk. Under mild conditions, we rigorously show that progressively
increasing the sample size at each training step is necessary to prevent model
collapse. In particular, when the estimation is unbiased, the required growth
rate follows a superlinear pattern. This rate needs to be accelerated even
further in the presence of substantial estimation bias. Building on this
probabilistic framework, we also investigate the probability that recursive
training on synthetic data yields models that outperform those trained solely
on real data. Moreover, we extend these results to general parametric model
family in an asymptotic regime. Finally, we validate our theoretical results
through extensive simulations and a real-world dataset.

### 2. [Computational Efficiency under Covariate Shift in Kernel Ridge Regression](http://arxiv.org/pdf/2505.14083v1)

Authors: Andrea Della Vecchia, Arnaud Mavakala Watusadisi, Ernesto De Vito, Lorenzo Rosasco

This paper addresses the covariate shift problem in the context of
nonparametric regression within reproducing kernel Hilbert spaces (RKHSs).
Covariate shift arises in supervised learning when the input distributions of
the training and test data differ, presenting additional challenges for
learning. Although kernel methods have optimal statistical properties, their
high computational demands in terms of time and, particularly, memory, limit
their scalability to large datasets. To address this limitation, the main focus
of this paper is to explore the trade-off between computational efficiency and
statistical accuracy under covariate shift. We investigate the use of random
projections where the hypothesis space consists of a random subspace within a
given RKHS. Our results show that, even in the presence of covariate shift,
significant computational savings can be achieved without compromising learning
performance.

### 3. [The Post Double LASSO for Efficiency Analysis](http://arxiv.org/pdf/2505.14282v1)

Authors: Christopher Parmeter, Artem Prokhorov, Valentin Zelenyuk

Big data and machine learning methods have become commonplace across economic
milieus. One area that has not seen as much attention to these important topics
yet is efficiency analysis. We show how the availability of big (wide) data can
actually make detection of inefficiency more challenging. We then show how
machine learning methods can be leveraged to adequately estimate the primitives
of the frontier itself as well as inefficiency using the `post double LASSO' by
deriving Neyman orthogonal moment conditions for this problem. Finally, an
application is presented to illustrate key differences of the post-double LASSO
compared to other approaches.

### 4. [Just One Layer Norm Guarantees Stable Extrapolation](http://arxiv.org/pdf/2505.14512v1)

Authors: Juliusz Ziomek, George Whittle, Michael A. Osborne

In spite of their prevalence, the behaviour of Neural Networks when
extrapolating far from the training distribution remains poorly understood,
with existing results limited to specific cases. In this work, we prove general
results -- the first of their kind -- by applying Neural Tangent Kernel (NTK)
theory to analyse infinitely-wide neural networks trained until convergence and
prove that the inclusion of just one Layer Norm (LN) fundamentally alters the
induced NTK, transforming it into a bounded-variance kernel. As a result, the
output of an infinitely wide network with at least one LN remains bounded, even
on inputs far from the training data. In contrast, we show that a broad class
of networks without LN can produce pathologically large outputs for certain
inputs. We support these theoretical findings with empirical experiments on
finite-width networks, demonstrating that while standard NNs often exhibit
uncontrolled growth outside the training domain, a single LN layer effectively
mitigates this instability. Finally, we explore real-world implications of this
extrapolatory stability, including applications to predicting residue sizes in
proteins larger than those seen during training and estimating age from facial
images of underrepresented ethnicities absent from the training set.

### 5. [A simple estimator of the correlation kernel matrix of a determinantal point process](http://arxiv.org/pdf/2505.14529v1)

Authors: Christian Gouriéroux, Yang Lu

The Determinantal Point Process (DPP) is a parameterized model for
multivariate binary variables, characterized by a correlation kernel matrix.
This paper proposes a closed form estimator of this kernel, which is
particularly easy to implement and can also be used as a starting value of
learning algorithms for maximum likelihood estimation. We prove the consistency
and asymptotic normality of our estimator, as well as its large deviation
properties.

### 6. [Inference with correlated priors using sisters cells](http://arxiv.org/pdf/2505.14579v1)

Authors: Sina Tootoonian, Andreas T. Schaefer

A common view of sensory processing is as probabilistic inference of latent
causes from receptor activations. Standard approaches often assume these causes
are a priori independent, yet real-world generative factors are typically
correlated. Representing such structured priors in neural systems poses
architectural challenges, particularly when direct interactions between units
representing latent causes are biologically implausible or computationally
expensive. Inspired by the architecture of the olfactory bulb, we propose a
novel circuit motif that enables inference with correlated priors without
requiring direct interactions among latent cause units. The key insight lies in
using sister cells: neurons receiving shared receptor input but connected
differently to local interneurons. The required interactions among latent units
are implemented indirectly through their connections to the sister cells, such
that correlated connectivity implies anti-correlation in the prior and vice
versa. We use geometric arguments to construct connectivity that implements a
given prior and to bound the number of causes for which such priors can be
constructed. Using simulations, we demonstrate the efficacy of such priors for
inference in noisy environments and compare the inference dynamics to those
experimentally observed. Finally, we show how, under certain assumptions on
latent representations, the prior used can be inferred from sister cell
activations. While biologically grounded in the olfactory system, our mechanism
generalises to other natural and artificial sensory systems and may inform the
design of architectures for efficient inference under correlated latent
structure.

### 7. [High-Dimensional Analysis of Bootstrap Ensemble Classifiers](http://arxiv.org/pdf/2505.14587v1)

Authors: Hamza Cherkaoui, Malik Tiomoko, Mohamed El Amine Seddik, Cosme Louart, Ekkehard Schnoor, Balazs Kegl

Bootstrap methods have long been a cornerstone of ensemble learning in
machine learning. This paper presents a theoretical analysis of bootstrap
techniques applied to the Least Square Support Vector Machine (LSSVM) ensemble
in the context of large and growing sample sizes and feature dimensionalities.
Leveraging tools from Random Matrix Theory, we investigate the performance of
this classifier that aggregates decision functions from multiple weak
classifiers, each trained on different subsets of the data. We provide insights
into the use of bootstrap methods in high-dimensional settings, enhancing our
understanding of their impact. Based on these findings, we propose strategies
to select the number of subsets and the regularization parameter that maximize
the performance of the LSSVM. Empirical experiments on synthetic and real-world
datasets validate our theoretical results.

### 8. [CSTS: A Benchmark for the Discovery of Correlation Structures in Time Series Clustering](http://arxiv.org/pdf/2505.14596v1)

Authors: Isabella Degen, Zahraa S Abdallah, Henry W J Reeve, Kate Robson Brown

Time series clustering promises to uncover hidden structural patterns in data
with applications across healthcare, finance, industrial systems, and other
critical domains. However, without validated ground truth information,
researchers cannot objectively assess clustering quality or determine whether
poor results stem from absent structures in the data, algorithmic limitations,
or inappropriate validation methods, raising the question whether clustering is
"more art than science" (Guyon et al., 2009). To address these challenges, we
introduce CSTS (Correlation Structures in Time Series), a synthetic benchmark
for evaluating the discovery of correlation structures in multivariate time
series data. CSTS provides a clean benchmark that enables researchers to
isolate and identify specific causes of clustering failures by differentiating
between correlation structure deterioration and limitations of clustering
algorithms and validation methods. Our contributions are: (1) a comprehensive
benchmark for correlation structure discovery with distinct correlation
structures, systematically varied data conditions, established performance
thresholds, and recommended evaluation protocols; (2) empirical validation of
correlation structure preservation showing moderate distortion from
downsampling and minimal effects from distribution shifts and sparsification;
and (3) an extensible data generation framework enabling structure-first
clustering evaluation. A case study demonstrates CSTS's practical utility by
identifying an algorithm's previously undocumented sensitivity to non-normal
distributions, illustrating how the benchmark enables precise diagnosis of
methodological limitations. CSTS advances rigorous evaluation standards for
correlation-based time series clustering.

### 9. [Characterization of Efficient Influence Function for Off-Policy Evaluation Under Optimal Policies](http://arxiv.org/pdf/2505.13809v1)

Authors: Haoyu Wei

Off-policy evaluation (OPE) provides a powerful framework for estimating the
value of a counterfactual policy using observational data, without the need for
additional experimentation. Despite recent progress in robust and efficient OPE
across various settings, rigorous efficiency analysis of OPE under an estimated
optimal policy remains limited. In this paper, we establish a concise
characterization of the efficient influence function for the value function
under optimal policy within canonical Markov decision process models.
Specifically, we provide the sufficient conditions for the existence of the
efficient influence function and characterize its expression. We also give the
conditions under which the EIF does not exist.

### 10. [Graphon Mixtures](http://arxiv.org/pdf/2505.13864v1)

Authors: Sevvandi Kandanaarachchi, Cheng Soon Ong

Social networks have a small number of large hubs, and a large number of
small dense communities. We propose a generative model that captures both hub
and dense structures. Based on recent results about graphons on line graphs,
our model is a graphon mixture, enabling us to generate sequences of graphs
where each graph is a combination of sparse and dense graphs. We propose a new
condition on sparse graphs (the max-degree), which enables us to identify hubs.
We show theoretically that we can estimate the normalized degree of the hubs,
as well as estimate the graphon corresponding to sparse components of graph
mixtures. We illustrate our approach on synthetic data, citation graphs, and
social networks, showing the benefits of explicitly modeling sparse graphs.

